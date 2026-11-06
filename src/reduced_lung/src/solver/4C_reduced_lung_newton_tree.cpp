// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_reduced_lung_newton_tree.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_reduced_lung_newton_sparse.hpp"
#include "4C_reduced_lung_tree_linear_solver.hpp"
#include "4C_solver_nonlin_nox_globaldata.hpp"
#include "4C_solver_nonlin_nox_group.hpp"
#include "4C_solver_nonlin_nox_interface_required_base.hpp"
#include "4C_solver_nonlin_nox_linearsystem.hpp"
#include "4C_solver_nonlin_nox_linearsystem_base.hpp"
#include "4C_solver_nonlin_nox_solver_factory.hpp"
#include "4C_solver_nonlin_nox_vector.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_shared_ptr_from_ref.hpp"

#include <NOX_Solver_Generic.H>
#include <NOX_StatusTest_Combo.H>
#include <NOX_StatusTest_FiniteValue.H>
#include <NOX_StatusTest_MaxIters.H>
#include <NOX_StatusTest_NormF.H>
#include <Teuchos_ParameterList.hpp>

#include <algorithm>
#include <format>
#include <iostream>

FOUR_C_NAMESPACE_OPEN

namespace ReducedLung
{
  namespace
  {
    /**
     * @brief Evaluates reduced-lung residuals and tree linearizations for NOX callbacks.
     *
     * The model state (locally relevant dofs and state updaters) is synced in every residual
     * evaluation. The tree linearization reuses that state: @ref TreeGroup only requests a Jacobian
     * at the state of the most recent residual evaluation, and NOX's final residual evaluation
     * happens at the converged state, so the models are up to date after a solve.
     */
    class TreeModelEvaluator
    {
     public:
      TreeModelEvaluator(const NewtonTreeContext& context, double initial_time)
          : locally_relevant_dofs_(context.locally_relevant_dofs),
            assembly_pipeline_(context.assembly_pipeline),
            tree_linear_solver_(context.tree_linear_solver),
            dt_(context.dynamics.time_increment),
            current_time_(initial_time)
      {
      }

      void set_time(double time) { current_time_ = time; }

      void evaluate_res(const Core::LinAlg::Vector<double>& x, Core::LinAlg::Vector<double>& res)
      {
        sync_state(x);

        res.put_scalar(0.0);
        for (const auto& assemble_res : assembly_pipeline_.residual_assemblers)
        {
          assemble_res(res, locally_relevant_dofs_, current_time_, dt_);
        }
      }

      /**
       * @brief Whether the models are synced to @p x, i.e. the last residual was evaluated at x.
       */
      [[nodiscard]] bool is_synced_to(const Core::LinAlg::Vector<double>& x) const
      {
        return std::ranges::equal(
            x.local_values_as_span(), locally_relevant_dofs_.local_values_as_span());
      }

      /**
       * @brief Assemble the tree linearization at the state of the most recent residual evaluation.
       */
      void evaluate_tree_linearization()
      {
        if (!tree_linearization_static_initialized_)
        {
          for (const auto& static_assembler :
              assembly_pipeline_.tree_linearization_static_assemblers)
          {
            static_assembler(*tree_linear_solver_);
          }
          tree_linearization_static_initialized_ = true;
        }
        for (const auto& tree_linearization_assembler :
            assembly_pipeline_.tree_linearization_assemblers)
        {
          tree_linearization_assembler(
              *tree_linear_solver_, locally_relevant_dofs_, current_time_, dt_);
        }
      }

      /**
       * @brief Compute the Newton direction -J^{-1} r directly with the tree solver.
       */
      void solve_newton_direction(
          const Core::LinAlg::Vector<double>& res, Core::LinAlg::Vector<double>& direction)
      {
        tree_linear_solver_->solve(res, direction);
      }

     private:
      /**
       * @brief Copy x to the locally relevant dof vector and update the models.
       *
       * The tree solver is serial-only, where x and the locally relevant dof vector share the same
       * local layout, so a plain copy suffices.
       */
      void sync_state(const Core::LinAlg::Vector<double>& x)
      {
        locally_relevant_dofs_.update(1.0, x, 0.0);

        for (const auto& update_state : assembly_pipeline_.state_updaters)
        {
          update_state(locally_relevant_dofs_, dt_);
        }
      }

      Core::LinAlg::Vector<double>& locally_relevant_dofs_;
      NonlinearSolverAssemblyPipeline assembly_pipeline_;
      std::shared_ptr<TreeLinearSolver> tree_linear_solver_;

      bool tree_linearization_static_initialized_ = false;

      double dt_;
      double current_time_;
    };

    /**
     * @brief NOX group that does not re-evaluate an already valid residual.
     *
     * NOX::Nln::Group::compute_f_and_jacobian() re-evaluates r whenever the Jacobian is invalid,
     * even if r is valid. In a line-search Newton iteration r has always just been evaluated at
     * the new iterate for the status test, so this would assemble the residual twice per
     * iteration. This group only evaluates the Jacobian in that case.
     */
    class TreeGroup : public NOX::Nln::Group
    {
     public:
      TreeGroup(Teuchos::ParameterList& print_params, Teuchos::ParameterList& group_options,
          const std::shared_ptr<NOX::Nln::Interface::RequiredBase>& required_interface,
          const NOX::Nln::Vector& x, const Teuchos::RCP<NOX::Nln::LinearSystemBase>& linear_system,
          TreeModelEvaluator& evaluator)
          : NOX::Nln::Group(print_params, group_options, required_interface, x, linear_system),
            evaluator_(evaluator)
      {
      }

      TreeGroup(const TreeGroup& source, ::NOX::CopyType type)
          : NOX::Nln::Group(source, type), evaluator_(source.evaluator_)
      {
      }

      Teuchos::RCP<::NOX::Abstract::Group> clone(::NOX::CopyType type) const override
      {
        return Teuchos::make_rcp<TreeGroup>(*this, type);
      }

      ::NOX::Abstract::Group::ReturnType compute_f_and_jacobian() override
      {
        if (isF() && !isJacobian()) return computeJacobian();
        return NOX::Nln::Group::compute_f_and_jacobian();
      }

      /**
       * @brief Let the tree solver write the Newton direction directly.
       *
       * Avoids zero-initializing the direction and negating it twice (the tree solver already
       * returns -J^{-1} r).
       */
      ::NOX::Abstract::Group::ReturnType computeNewton(Teuchos::ParameterList& /*p*/) override
      {
        if (isNewton()) return ::NOX::Abstract::Group::Ok;
        FOUR_C_ASSERT(isF() && isJacobian(),
            "ReducedLung::TreeGroup::computeNewton() requires a valid residual and Jacobian.");

        evaluator_.solve_newton_direction(
            RHSVector.get_linalg_vector(), NewtonVector.get_linalg_vector());
        isValidNewton = true;
        return ::NOX::Abstract::Group::Ok;
      }

     private:
      TreeModelEvaluator& evaluator_;
    };

    /**
     * @brief NOX residual interface forwarding to the @ref TreeModelEvaluator.
     */
    class TreeRequiredInterface : public NOX::Nln::Interface::RequiredBase
    {
     public:
      explicit TreeRequiredInterface(TreeModelEvaluator& evaluator) : evaluator_(evaluator) {}

      bool compute_f(const Core::LinAlg::Vector<double>& x, Core::LinAlg::Vector<double>& res,
          NOX::Nln::FillType /*fill_flag*/) override
      {
        evaluator_.evaluate_res(x, res);
        return true;
      }

     private:
      TreeModelEvaluator& evaluator_;
    };

    /**
     * @brief NOX linear system that assembles the tree linearization in the tree linear solver.
     *
     * There is no sparse Jacobian and no linear solve through this class: @ref TreeGroup computes
     * the Newton direction with the tree solver.
     */
    class TreeLinearSystem : public NOX::Nln::LinearSystemBase
    {
     public:
      explicit TreeLinearSystem(TreeModelEvaluator& evaluator) : evaluator_(evaluator) {}

      bool compute_jacobian(const NOX::Nln::Vector& x) override
      {
        FOUR_C_ASSERT(evaluator_.is_synced_to(x.get_linalg_vector()),
            "ReducedLung::NewtonTree: the tree linearization is requested at a state other than "
            "the one of the most recent residual evaluation.");
        evaluator_.evaluate_tree_linearization();
        return true;
      }

      bool compute_f_and_jacobian(const NOX::Nln::Vector& x, NOX::Nln::Vector& rhs) override
      {
        evaluator_.evaluate_res(x.get_linalg_vector(), rhs.get_linalg_vector());
        evaluator_.evaluate_tree_linearization();
        return true;
      }

      bool apply_jacobian(
          const NOX::Nln::Vector& /*input*/, NOX::Nln::Vector& /*result*/) const override
      {
        FOUR_C_THROW("ReducedLung::TreeLinearSystem does not support Jacobian application.");
      }

      bool apply_jacobian_transpose(
          const NOX::Nln::Vector& /*input*/, NOX::Nln::Vector& /*result*/) const override
      {
        FOUR_C_THROW(
            "ReducedLung::TreeLinearSystem does not support transposed Jacobian application.");
      }

      bool apply_jacobian_inverse(Teuchos::ParameterList& /*linear_solver_params*/,
          const NOX::Nln::Vector& /*input*/, NOX::Nln::Vector& /*result*/) override
      {
        FOUR_C_THROW(
            "ReducedLung::TreeLinearSystem does not support inverse Jacobian application; the "
            "Newton direction is computed by TreeGroup.");
      }

      std::shared_ptr<const Core::LinAlg::SparseOperator> get_jacobian_operator() const override
      {
        FOUR_C_THROW("ReducedLung::TreeLinearSystem has no sparse Jacobian operator.");
      }

      std::shared_ptr<Core::LinAlg::SparseOperator> get_jacobian_operator() override
      {
        FOUR_C_THROW("ReducedLung::TreeLinearSystem has no sparse Jacobian operator.");
      }

     private:
      TreeModelEvaluator& evaluator_;
    };

    /**
     * @brief NOX parameter list: full-step Newton with silent printing.
     *
     * The "Status Test" sublist is only filled because NOX::Nln::GlobalData insists on it; the
     * outer status tests are built directly in C++ below.
     */
    Teuchos::ParameterList create_newton_tree_parameter_list(
        const ReducedLungParameters::Dynamics& dynamics)
    {
      Teuchos::ParameterList nox_params = create_nox_parameter_list(dynamics);

      auto& printing = nox_params.sublist("Printing");
      printing.set("Error", true);
      printing.set("Warning", false);
      printing.set("Outer Iteration", false);
      printing.set("Inner Iteration", false);
      printing.set("Outer Iteration StatusTest", false);

      return nox_params;
    }
  }  // namespace

  struct NewtonTree::Implementation
  {
    Implementation(const NewtonTreeContext& context, double initial_time)
        : evaluator(context, initial_time),
          nox_params(create_newton_tree_parameter_list(context.dynamics))
    {
      FOUR_C_ASSERT_ALWAYS(context.dynamics.max_nonlinear_iterations > 0,
          "ReducedLung::NewtonTree requires a positive max_nonlinear_iterations, got {}.",
          context.dynamics.max_nonlinear_iterations);
      FOUR_C_ASSERT_ALWAYS(Core::Communication::num_mpi_ranks(context.comm) == 1,
          "ReducedLung::NewtonTree is serial-only, but the run uses {} MPI ranks.",
          Core::Communication::num_mpi_ranks(context.comm));
      FOUR_C_ASSERT_ALWAYS(context.tree_linear_solver != nullptr,
          "ReducedLung::NewtonTree requires a valid tree linear solver.");
      FOUR_C_ASSERT_ALWAYS(!context.assembly_pipeline.residual_assemblers.empty(),
          "ReducedLung::NewtonTree requires at least one residual assembler callback.");
      FOUR_C_ASSERT_ALWAYS(
          !context.assembly_pipeline.tree_linearization_static_assemblers.empty() ||
              !context.assembly_pipeline.tree_linearization_assemblers.empty(),
          "ReducedLung::NewtonTree requires tree-linearization assembler callbacks.");

      // The tree linear system needs neither a Core::LinAlg::Solver nor a Jacobian interface.
      const std::shared_ptr<NOX::Nln::Interface::RequiredBase> required_interface =
          std::make_shared<TreeRequiredInterface>(evaluator);
      global_data = Teuchos::make_rcp<NOX::Nln::GlobalData>(context.comm, nox_params,
          NOX::Nln::LinearSystem::SolverMap{}, required_interface, nullptr);

      // View on the solution vector: NOX starts from it and the converged state is written back.
      x_nox = std::make_shared<NOX::Nln::Vector>(
          Core::Utils::shared_ptr_from_ref(context.x), NOX::Nln::Vector::MemoryType::View);

      // The NOX solver owns the group, which owns the linear system.
      Teuchos::ParameterList& params = global_data->get_nln_parameter_list();
      auto group = Teuchos::make_rcp<TreeGroup>(params.sublist("Printing"),
          params.sublist("Group Options"), required_interface, *x_nox,
          Teuchos::make_rcp<TreeLinearSystem>(evaluator), evaluator);

      res_test =
          Teuchos::make_rcp<::NOX::StatusTest::NormF>(context.dynamics.nonlinear_residual_tolerance,
              ::NOX::Abstract::Vector::TwoNorm, ::NOX::StatusTest::NormF::Unscaled);
      finite_value_test = Teuchos::make_rcp<::NOX::StatusTest::FiniteValue>();
      auto max_iterations_test =
          Teuchos::make_rcp<::NOX::StatusTest::MaxIters>(context.dynamics.max_nonlinear_iterations);

      auto outer_status_test =
          Teuchos::make_rcp<::NOX::StatusTest::Combo>(::NOX::StatusTest::Combo::OR);
      outer_status_test->addStatusTest(finite_value_test);
      outer_status_test->addStatusTest(res_test);
      outer_status_test->addStatusTest(max_iterations_test);

      nox_solver = NOX::Nln::Solver::build_solver(group, outer_status_test,
          Teuchos::RCP<NOX::Nln::Inner::StatusTest::Generic>(), *global_data);
    }

    TreeModelEvaluator evaluator;
    /// NOX only keeps a non-owning reference to the parameter list, so it has to be kept alive.
    Teuchos::ParameterList nox_params;
    Teuchos::RCP<NOX::Nln::GlobalData> global_data;
    std::shared_ptr<NOX::Nln::Vector> x_nox;
    /// Status tests queried after a solve; the remaining tests are owned by the NOX solver.
    Teuchos::RCP<::NOX::StatusTest::NormF> res_test;
    Teuchos::RCP<::NOX::StatusTest::FiniteValue> finite_value_test;
    Teuchos::RCP<::NOX::Solver::Generic> nox_solver;
  };

  NewtonTree::NewtonTree(const NewtonTreeContext& context, double initial_time)
      : implementation_(std::make_unique<Implementation>(context, initial_time))
  {
  }

  NewtonTree::~NewtonTree() = default;

  unsigned int NewtonTree::solve(double time)
  {
    Implementation& impl = *implementation_;
    impl.evaluator.set_time(time);

    impl.nox_solver->reset(*impl.x_nox);
    const ::NOX::StatusTest::StatusType status = impl.nox_solver->solve();

    const auto iterations = static_cast<unsigned int>(impl.nox_solver->getNumIterations());
    const double res_norm = impl.res_test->getNormF();

    FOUR_C_ASSERT_ALWAYS(impl.finite_value_test->getStatus() != ::NOX::StatusTest::Failed,
        "ReducedLung::NewtonTree encountered a non-finite residual at time {} after {} Newton "
        "corrections.",
        time, iterations);
    FOUR_C_ASSERT_ALWAYS(status == ::NOX::StatusTest::Converged,
        "ReducedLung::NewtonTree did not converge at time {} after {} Newton corrections. "
        "Final residual norm: {}.",
        time, iterations, res_norm);

    impl.x_nox->update(1.0, impl.nox_solver->getSolutionGroup().getX(), 0.0);

    std::cout << std::format("Newton iters: {} | ||F||: {:.2e}\n", iterations, res_norm)
              << std::flush;
    return iterations;
  }
}  // namespace ReducedLung

FOUR_C_NAMESPACE_CLOSE

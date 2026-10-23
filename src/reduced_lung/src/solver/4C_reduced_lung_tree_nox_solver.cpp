// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_reduced_lung_tree_nox_solver.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_linalg_map.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_reduced_lung_tree_linear_solver.hpp"
#include "4C_solver_nonlin_nox_aux.hpp"
#include "4C_solver_nonlin_nox_globaldata.hpp"
#include "4C_solver_nonlin_nox_group.hpp"
#include "4C_solver_nonlin_nox_interface_jacobian.hpp"
#include "4C_solver_nonlin_nox_interface_required.hpp"
#include "4C_solver_nonlin_nox_linearsystem.hpp"
#include "4C_solver_nonlin_nox_problem.hpp"
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

#include <cmath>
#include <map>

FOUR_C_NAMESPACE_OPEN

namespace ReducedLung
{
  namespace
  {
    /**
     * Compute the two-norm with a serial fast path (same as TreeNewtonSolver).
     */
    double compute_norm_2(const Core::LinAlg::Vector<double>& vector)
    {
      if (Core::Communication::num_mpi_ranks(vector.get_comm()) != 1)
      {
        double norm = 0.0;
        vector.norm_2(&norm);
        return norm;
      }

      double norm_square = 0.0;
      for (const double value : vector.local_values_as_span())
      {
        norm_square += value * value;
      }
      return std::sqrt(norm_square);
    }

    /**
     * @brief Evaluates reduced-lung residuals and tree linearizations for NOX callbacks.
     *
     * The model state (dof vectors and state updaters) is synced in every residual evaluation.
     * The tree linearization reuses that state: @ref TreeGroup only requests a Jacobian at the
     * state of the most recent residual evaluation, and NOX's final residual evaluation happens at
     * the converged state, so the models are up to date after a solve.
     */
    class TreeModelEvaluator
    {
     public:
      TreeModelEvaluator(const TreeNoxSolverContext& context, double initial_time)
          : dofs_(context.dofs),
            locally_relevant_dofs_(context.locally_relevant_dofs),
            assembly_pipeline_(context.assembly_pipeline),
            tree_linear_solver_(context.tree_linear_solver),
            dt_(context.dynamics.time_increment),
            current_time_(initial_time)
      {
      }

      /// Model history and time change between nonlinear solves.
      void begin_solve(double time) { current_time_ = time; }

      void evaluate_res(const Core::LinAlg::Vector<double>& x, Core::LinAlg::Vector<double>& res)
      {
        sync_state(x);

        res.put_scalar(0.0);
        for (const auto& assemble_res : assembly_pipeline_.residual_assemblers)
        {
          assemble_res(res, locally_relevant_dofs_, current_time_, dt_);
        }
      }

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
       * @brief Compute result = J^{-1} r with the tree solver.
       *
       * The tree solver returns the Newton correction delta = -J^{-1} r, while NOX expects the
       * plain inverse application (it negates the result itself).
       */
      void apply_jacobian_inverse(
          const Core::LinAlg::Vector<double>& res, Core::LinAlg::Vector<double>& result)
      {
        tree_linear_solver_->solve(res, result);
        result.scale(-1.0);
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
      /// Export x to the dof vectors and update the models.
      void sync_state(const Core::LinAlg::Vector<double>& x)
      {
        if (Core::Communication::num_mpi_ranks(x.get_comm()) == 1)
        {
          dofs_.update(1.0, x, 0.0);
          locally_relevant_dofs_.update(1.0, x, 0.0);
        }
        else
        {
          Core::LinAlg::export_to(x, dofs_);
          Core::LinAlg::export_to(dofs_, locally_relevant_dofs_);
        }

        for (const auto& update_state : assembly_pipeline_.state_updaters)
        {
          update_state(locally_relevant_dofs_, dt_);
        }
      }

      Core::LinAlg::Vector<double>& dofs_;
      Core::LinAlg::Vector<double>& locally_relevant_dofs_;
      NonlinearSolverAssemblyPipeline assembly_pipeline_;
      std::shared_ptr<TreeNewtonLinearSolver> tree_linear_solver_;

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

      /**
       * @brief Copy the state needed by line searches (x and r) but not the stale Newton vector.
       *
       * The line-search based solver copies the current group into the previous-solution group in
       * every iteration. The previous group's Newton direction and gradient are never used.
       */
      ::NOX::Abstract::Group& operator=(const ::NOX::Abstract::Group& source) override
      {
        const auto& tree_source = dynamic_cast<const TreeGroup&>(source);

        xVector = tree_source.xVector;
        isValidRHS = tree_source.isValidRHS;
        isValidJacobian = tree_source.isValidJacobian;
        isValidGrad = false;
        isValidNewton = false;
        if (isValidRHS) RHSVector = tree_source.RHSVector;

        return *this;
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

      double getNormF() const override
      {
        FOUR_C_ASSERT(isF(), "ReducedLung::TreeGroup::getNormF() - invalid residual");
        return compute_norm_2(RHSVector.get_linalg_vector());
      }

     private:
      TreeModelEvaluator& evaluator_;
    };

    /**
     * @brief NOX required/Jacobian interface forwarding to the @ref TreeModelEvaluator.
     */
    class TreeNoxInterface : public NOX::Nln::Interface::Required,
                             public NOX::Nln::Interface::Jacobian
    {
     public:
      explicit TreeNoxInterface(TreeModelEvaluator& evaluator) : evaluator_(evaluator) {}

      bool compute_f(const Core::LinAlg::Vector<double>& x, Core::LinAlg::Vector<double>& res,
          NOX::Nln::FillType /*fill_flag*/) override
      {
        evaluator_.evaluate_res(x, res);
        return true;
      }

      bool compute_jacobian(
          const Core::LinAlg::Vector<double>& /*x*/, Core::LinAlg::SparseOperator& /*jac*/) override
      {
        evaluator_.evaluate_tree_linearization();
        return true;
      }

      bool compute_f_and_jacobian(const Core::LinAlg::Vector<double>& x,
          Core::LinAlg::Vector<double>& res, Core::LinAlg::SparseOperator& jac) override
      {
        return compute_f(x, res, NOX::Nln::FillType::Residual) && compute_jacobian(x, jac);
      }

      Teuchos::RCP<Core::LinAlg::SparseMatrix>
      calc_jacobian_contributions_from_element_level_for_ptc() override
      {
        FOUR_C_THROW("ReducedLung::TreeNoxSolver: PTC is not supported.");
      }

      double get_primary_rhs_norms(const Core::LinAlg::Vector<double>& res,
          const NOX::Nln::StatusTest::QuantityType& /*checkquantity*/,
          const ::NOX::Abstract::Vector::NormType& type, const bool& isscaled) const override
      {
        return NOX::Nln::Aux::calc_vector_norm(res, type, isscaled);
      }

      double get_primary_solution_update_rms(const Core::LinAlg::Vector<double>& xnew,
          const Core::LinAlg::Vector<double>& xold, const double& aTol, const double& rTol,
          const NOX::Nln::StatusTest::QuantityType& /*checkquantity*/,
          const bool& disable_implicit_weighting) const override
      {
        Core::LinAlg::Vector<double> xincr(xnew);
        xincr.update(-1.0, xold, 1.0);
        return NOX::Nln::Aux::root_mean_square_norm(
            aTol, rTol, xnew, xincr, disable_implicit_weighting);
      }

      double get_primary_solution_update_norms(const Core::LinAlg::Vector<double>& xnew,
          const Core::LinAlg::Vector<double>& xold,
          const NOX::Nln::StatusTest::QuantityType& /*checkquantity*/,
          const ::NOX::Abstract::Vector::NormType& type, const bool& isscaled) const override
      {
        Core::LinAlg::Vector<double> xincr(xold);
        xincr.update(1.0, xnew, -1.0);
        return NOX::Nln::Aux::calc_vector_norm(xincr, type, isscaled);
      }

      double get_previous_primary_solution_norms(const Core::LinAlg::Vector<double>& xold,
          const NOX::Nln::StatusTest::QuantityType& /*checkquantity*/,
          const ::NOX::Abstract::Vector::NormType& type, const bool& isscaled) const override
      {
        return NOX::Nln::Aux::calc_vector_norm(xold, type, isscaled);
      }

      double calc_ref_norm_force() override { return 1.0; }

     private:
      TreeModelEvaluator& evaluator_;
    };

    /**
     * @brief NOX linear system whose inverse Jacobian application is the tree solver.
     *
     * PROTOTYPE: derives from the concrete NOX::Nln::LinearSystem because NOX::Nln::Group casts
     * to it. The sparse Jacobian passed to the base class is a never-filled placeholder; only the
     * Jacobian evaluation callbacks and @ref apply_jacobian_inverse are used.
     */
    class TreeLinearSystem : public NOX::Nln::LinearSystem
    {
     public:
      TreeLinearSystem(Teuchos::ParameterList& print_params,
          Teuchos::ParameterList& linear_solver_params, const SolverMap& solvers,
          const std::shared_ptr<NOX::Nln::Interface::RequiredBase>& required_interface,
          const std::shared_ptr<NOX::Nln::Interface::JacobianBase>& jacobian_interface,
          const std::shared_ptr<Core::LinAlg::SparseOperator>& placeholder_jacobian,
          const NOX::Nln::Vector& clone_vector, TreeModelEvaluator& evaluator)
          : NOX::Nln::LinearSystem(print_params, linear_solver_params, solvers, required_interface,
                jacobian_interface, placeholder_jacobian, clone_vector),
            evaluator_(evaluator)
      {
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
          const NOX::Nln::Vector& input, NOX::Nln::Vector& result) override
      {
        evaluator_.apply_jacobian_inverse(input.get_linalg_vector(), result.get_linalg_vector());
        return true;
      }

     protected:
      Core::LinAlg::SolverParams set_solver_options(Teuchos::ParameterList& /*p*/,
          Teuchos::RCP<Core::LinAlg::Solver>& /*solver*/,
          const NOX::Nln::SolutionType& /*solver_type*/) override
      {
        FOUR_C_THROW("ReducedLung::TreeLinearSystem does not use a Core::LinAlg::Solver.");
      }

      NOX::Nln::SolutionType get_active_lin_solver(
          const std::map<NOX::Nln::SolutionType, Teuchos::RCP<Core::LinAlg::Solver>>& /*solvers*/,
          Teuchos::RCP<Core::LinAlg::Solver>& /*current_solver*/) override
      {
        FOUR_C_THROW("ReducedLung::TreeLinearSystem does not use a Core::LinAlg::Solver.");
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
    Teuchos::ParameterList create_tree_nox_parameter_list(
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

  struct TreeNoxSolver::Implementation
  {
    Implementation(const TreeNoxSolverContext& context, double initial_time)
        : x_solution(context.x),
          evaluator(context, initial_time),
          nox_params(create_tree_nox_parameter_list(context.dynamics)),
          max_nonlinear_iterations(context.dynamics.max_nonlinear_iterations)
    {
      FOUR_C_ASSERT_ALWAYS(context.dynamics.max_nonlinear_iterations > 0,
          "ReducedLung::TreeNoxSolver requires a positive max_nonlinear_iterations, got {}.",
          context.dynamics.max_nonlinear_iterations);
      FOUR_C_ASSERT_ALWAYS(context.tree_linear_solver != nullptr,
          "ReducedLung::TreeNoxSolver requires a valid tree Newton linear solver.");
      FOUR_C_ASSERT_ALWAYS(!context.assembly_pipeline.residual_assemblers.empty(),
          "ReducedLung::TreeNoxSolver requires at least one residual assembler callback.");

      // The tree linear system never touches a Core::LinAlg::Solver; NOX only warns about a null
      // entry in the solver map.
      std::map<NOX::Nln::SolutionType, Teuchos::RCP<Core::LinAlg::Solver>> solvers;
      solvers[NOX::Nln::sol_generic] = Teuchos::null;

      auto interface = std::make_shared<TreeNoxInterface>(evaluator);
      auto required_interface =
          std::static_pointer_cast<NOX::Nln::Interface::RequiredBase>(interface);
      auto jacobian_interface =
          std::static_pointer_cast<NOX::Nln::Interface::JacobianBase>(interface);

      global_data = Teuchos::make_rcp<NOX::Nln::GlobalData>(
          context.comm, nox_params, solvers, required_interface, jacobian_interface);

      x_nox = std::make_shared<NOX::Nln::Vector>(
          Core::Utils::shared_ptr_from_ref(context.x), NOX::Nln::Vector::MemoryType::View);
      // NOX only inspects the operator type of the Jacobian; an empty matrix is sufficient.
      const Core::LinAlg::Map empty_map(0, 0, context.comm);
      placeholder_jacobian = std::make_shared<Core::LinAlg::SparseMatrix>(empty_map, 1);

      problem = Teuchos::make_rcp<NOX::Nln::Problem>(global_data, x_nox, placeholder_jacobian);

      Teuchos::ParameterList& params = global_data->get_nln_parameter_list();
      linear_system = Teuchos::make_rcp<TreeLinearSystem>(params.sublist("Printing"),
          params.sublist("Direction").sublist("Newton").sublist("Linear Solver"),
          global_data->get_linear_solvers(), required_interface, jacobian_interface,
          placeholder_jacobian, *x_nox, evaluator);
      group = Teuchos::make_rcp<TreeGroup>(params.sublist("Printing"),
          params.sublist("Group Options"), required_interface, *x_nox, linear_system, evaluator);

      res_test =
          Teuchos::make_rcp<::NOX::StatusTest::NormF>(context.dynamics.nonlinear_residual_tolerance,
              ::NOX::Abstract::Vector::TwoNorm, ::NOX::StatusTest::NormF::Unscaled);
      finite_value_test = Teuchos::make_rcp<::NOX::StatusTest::FiniteValue>();
      max_iterations_test =
          Teuchos::make_rcp<::NOX::StatusTest::MaxIters>(context.dynamics.max_nonlinear_iterations);

      auto combo = Teuchos::make_rcp<::NOX::StatusTest::Combo>(::NOX::StatusTest::Combo::OR);
      combo->addStatusTest(finite_value_test);
      combo->addStatusTest(res_test);
      combo->addStatusTest(max_iterations_test);
      outer_status_test = combo;

      nox_solver = NOX::Nln::Solver::build_solver(group, outer_status_test,
          Teuchos::RCP<NOX::Nln::Inner::StatusTest::Generic>(), *global_data);
    }

    Core::LinAlg::Vector<double>& x_solution;
    TreeModelEvaluator evaluator;
    Teuchos::ParameterList nox_params;
    int max_nonlinear_iterations;

    Teuchos::RCP<NOX::Nln::GlobalData> global_data;
    std::shared_ptr<NOX::Nln::Vector> x_nox;
    std::shared_ptr<Core::LinAlg::SparseMatrix> placeholder_jacobian;
    Teuchos::RCP<NOX::Nln::Problem> problem;
    Teuchos::RCP<TreeLinearSystem> linear_system;
    Teuchos::RCP<::NOX::Abstract::Group> group;
    Teuchos::RCP<::NOX::StatusTest::NormF> res_test;
    Teuchos::RCP<::NOX::StatusTest::FiniteValue> finite_value_test;
    Teuchos::RCP<::NOX::StatusTest::MaxIters> max_iterations_test;
    Teuchos::RCP<::NOX::StatusTest::Generic> outer_status_test;
    Teuchos::RCP<::NOX::Solver::Generic> nox_solver;

    double last_res_norm = 0.0;
    double current_time = 0.0;
  };

  TreeNoxSolver::TreeNoxSolver(const TreeNoxSolverContext& context, double initial_time)
      : implementation_(std::make_unique<Implementation>(context, initial_time))
  {
  }

  TreeNoxSolver::~TreeNoxSolver() = default;

  unsigned int TreeNoxSolver::solve(double time)
  {
    Implementation& impl = *implementation_;
    impl.current_time = time;
    impl.evaluator.begin_solve(time);

    impl.nox_solver->reset(*impl.x_nox);
    const ::NOX::StatusTest::StatusType status = impl.nox_solver->solve();

    const auto iterations = static_cast<unsigned int>(impl.nox_solver->getNumIterations());
    impl.last_res_norm = impl.res_test->getNormF();

    FOUR_C_ASSERT_ALWAYS(impl.finite_value_test->getStatus() != ::NOX::StatusTest::Failed,
        "ReducedLung::TreeNoxSolver encountered a non-finite residual at time {} after {} Newton "
        "corrections.",
        time, iterations);
    FOUR_C_ASSERT_ALWAYS(status == ::NOX::StatusTest::Converged,
        "ReducedLung::TreeNoxSolver did not converge at time {} after {} Newton corrections. "
        "Final residual norm: {}.",
        time, iterations, impl.last_res_norm);

    const auto& final_x =
        dynamic_cast<const NOX::Nln::Vector&>(impl.nox_solver->getSolutionGroup().getX());
    impl.x_solution.update(1.0, final_x.get_linalg_vector(), 0.0);

    return iterations;
  }

  double TreeNoxSolver::last_residual_norm() const { return implementation_->last_res_norm; }
}  // namespace ReducedLung

FOUR_C_NAMESPACE_CLOSE

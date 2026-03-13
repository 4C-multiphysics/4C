// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_solver_nonlin_nox_adapter.hpp"

#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_solver_nonlin_nox_aux.hpp"
#include "4C_solver_nonlin_nox_globaldata.hpp"
#include "4C_solver_nonlin_nox_group.hpp"
#include "4C_solver_nonlin_nox_interface_jacobian.hpp"
#include "4C_solver_nonlin_nox_interface_required.hpp"
#include "4C_solver_nonlin_nox_problem.hpp"
#include "4C_solver_nonlin_nox_solver_factory.hpp"
#include "4C_solver_nonlin_nox_vector.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_shared_ptr_from_ref.hpp"

#include <map>

FOUR_C_NAMESPACE_OPEN

namespace
{
  const NOX::Nln::Adapter::Options default_options;

  class AdapterInterface : public NOX::Nln::Interface::Required,
                           public NOX::Nln::Interface::Jacobian
  {
   public:
    using ResidualCallback = std::function<bool(
        const Core::LinAlg::Vector<double>&, Core::LinAlg::Vector<double>&, NOX::Nln::FillType)>;
    using JacobianCallback =
        std::function<bool(const Core::LinAlg::Vector<double>&, Core::LinAlg::SparseOperator&)>;

    AdapterInterface(ResidualCallback* residual_cb, JacobianCallback* jacobian_cb,
        const NOX::Nln::Adapter::Options* options)
        : residual_cb_(residual_cb), jacobian_cb_(jacobian_cb), options_(options)
    {
    }

    bool compute_f(const Core::LinAlg::Vector<double>& x, Core::LinAlg::Vector<double>& rhs,
        NOX::Nln::FillType fill_flag) override
    {
      if (residual_cb_ == nullptr || !(*residual_cb_))
        FOUR_C_THROW("Adapter: residual callback not set.");

      return (*residual_cb_)(x, rhs, fill_flag);
    }

    bool compute_jacobian(
        const Core::LinAlg::Vector<double>& x, Core::LinAlg::SparseOperator& jac) override
    {
      if (jacobian_cb_ == nullptr || !(*jacobian_cb_))
        FOUR_C_THROW("Adapter: jacobian callback not set.");

      return (*jacobian_cb_)(x, jac);
    }

    bool compute_f_and_jacobian(const Core::LinAlg::Vector<double>& x,
        Core::LinAlg::Vector<double>& rhs, Core::LinAlg::SparseOperator& jac) override
    {
      if (!compute_f(x, rhs, NOX::Nln::FillType::Residual)) return false;
      return compute_jacobian(x, jac);
    }

    Teuchos::RCP<Core::LinAlg::SparseMatrix>
    calc_jacobian_contributions_from_element_level_for_ptc() override
    {
      if (options_ && options_->ptc_element_scaling) return options_->ptc_element_scaling();

      FOUR_C_THROW("Adapter: PTC is not supported.");
      return Teuchos::null;
    }

    double get_primary_rhs_norms(const Core::LinAlg::Vector<double>& residual,
        const NOX::Nln::StatusTest::QuantityType& checkquantity,
        const ::NOX::Abstract::Vector::NormType& type, const bool& isscaled) const override
    {
      if (options_ && options_->residual_norm)
        return options_->residual_norm(residual, checkquantity, type, isscaled);

      (void)checkquantity;
      return NOX::Nln::Aux::calc_vector_norm(residual, type, isscaled);
    }

    double get_primary_solution_update_rms(const Core::LinAlg::Vector<double>& xnew,
        const Core::LinAlg::Vector<double>& xold, const double& aTol, const double& rTol,
        const NOX::Nln::StatusTest::QuantityType& checkQuantity,
        const bool& disable_implicit_weighting) const override
    {
      if (options_ && options_->solution_update_rms)
        return options_->solution_update_rms(
            xnew, xold, aTol, rTol, checkQuantity, disable_implicit_weighting);

      (void)checkQuantity;

      Core::LinAlg::Vector<double> xincr(xnew);
      xincr.update(-1.0, xold, 1.0);

      return NOX::Nln::Aux::root_mean_square_norm(
          aTol, rTol, xnew, xincr, disable_implicit_weighting);
    }

    double get_primary_solution_update_norms(const Core::LinAlg::Vector<double>& xnew,
        const Core::LinAlg::Vector<double>& xold,
        const NOX::Nln::StatusTest::QuantityType& checkquantity,
        const ::NOX::Abstract::Vector::NormType& type, const bool& isscaled) const override
    {
      if (options_ && options_->solution_update_norm)
        return options_->solution_update_norm(xnew, xold, checkquantity, type, isscaled);

      (void)checkquantity;

      Core::LinAlg::Vector<double> xincr(xold);
      xincr.update(1.0, xnew, -1.0);

      return NOX::Nln::Aux::calc_vector_norm(xincr, type, isscaled);
    }

    double get_previous_primary_solution_norms(const Core::LinAlg::Vector<double>& xold,
        const NOX::Nln::StatusTest::QuantityType& checkquantity,
        const ::NOX::Abstract::Vector::NormType& type, const bool& isscaled) const override
    {
      if (options_ && options_->previous_solution_norm)
        return options_->previous_solution_norm(xold, checkquantity, type, isscaled);

      (void)checkquantity;
      return NOX::Nln::Aux::calc_vector_norm(xold, type, isscaled);
    }

    double calc_ref_norm_force() override
    {
      if (options_ && options_->ref_norm_force) return options_->ref_norm_force();

      return 1.0;
    }

   private:
    ResidualCallback* residual_cb_;
    JacobianCallback* jacobian_cb_;
    const NOX::Nln::Adapter::Options* options_;
  };
}  // namespace

NOX::Nln::Adapter::Adapter(
    MPI_Comm comm, const Teuchos::ParameterList& nox_params, SolverMap solvers)
    : Adapter(comm, nox_params, std::move(solvers), default_options)
{
}

NOX::Nln::Adapter::Adapter(MPI_Comm comm, const Teuchos::ParameterList& nox_params,
    SolverMap solvers, const Options& options)
    : comm_(comm), nox_params_(nox_params), solvers_(std::move(solvers)), options_(options)
{
  if (solvers_.empty())
    FOUR_C_THROW("Adapter requires at least one Core::LinAlg::Solver instance.");

  for (const auto& [type, solver] : solvers_)
  {
    if (!solver)
      FOUR_C_THROW("Adapter solver map contains a null solver pointer (SolutionType = {}).",
          solution_type_to_string(type));
  }

  auto interface = std::make_shared<AdapterInterface>(&residual, &jacobian, &options_);
  interface_required_ = std::static_pointer_cast<NOX::Nln::Interface::RequiredBase>(interface);
  interface_jacobian_ = std::static_pointer_cast<NOX::Nln::Interface::JacobianBase>(interface);

  std::map<NOX::Nln::SolutionType, Teuchos::RCP<Core::LinAlg::Solver>> lin_solvers;
  for (const auto& [type, solver] : solvers_)
  {
    lin_solvers[type] = Teuchos::rcp(solver.get(), /*owns_mem=*/false);
  }

  nox_global_data_ = Teuchos::make_rcp<NOX::Nln::GlobalData>(
      comm_, nox_params_, lin_solvers, interface_required_, interface_jacobian_);
}

unsigned int NOX::Nln::Adapter::solve(Core::LinAlg::Vector<double>& x,
    Core::LinAlg::Vector<double>& residual_vector, Core::LinAlg::SparseOperator& jacobian)
{
  if (nox_solver_.is_null() || x_ptr_ != &x || jacobian_ptr_ != &jacobian)
  {
    build_solver(x, jacobian);
  }

  group_->setX(*x_nox_);
  const auto status = nox_solver_->solve();
  nox_problem_->check_final_status(status);

  const auto* final_group = dynamic_cast<const NOX::Nln::Group*>(&nox_solver_->getSolutionGroup());
  if (!final_group)
  {
    FOUR_C_THROW(
        "Adapter::solve: Expected NOX::Nln::Group from solver solution group, but "
        "dynamic_cast failed.");
  }

  const auto* x_vector = dynamic_cast<const NOX::Nln::Vector*>(&final_group->getX());
  if (!x_vector)
  {
    FOUR_C_THROW(
        "Adapter::solve: Expected NOX::Nln::Vector from final_group->getX(), but "
        "dynamic_cast failed.");
  }

  const auto* f_vector = dynamic_cast<const NOX::Nln::Vector*>(&final_group->getF());
  if (!f_vector)
  {
    FOUR_C_THROW(
        "Adapter::solve: Expected NOX::Nln::Vector from final_group->getF(), but "
        "dynamic_cast failed.");
  }

  Core::LinAlg::export_to(x_vector->get_linalg_vector(), x);
  Core::LinAlg::export_to(f_vector->get_linalg_vector(), residual_vector);

  return static_cast<unsigned int>(nox_solver_->getNumIterations());
}

void NOX::Nln::Adapter::reset()
{
  if (x_ptr_ == nullptr || jacobian_ptr_ == nullptr)
    FOUR_C_THROW("Adapter::reset called before first solve; no cached context available.");

  nox_solver_ = Teuchos::null;
  group_ = Teuchos::null;
  linsys_ = Teuchos::null;
  nox_problem_ = Teuchos::null;
  x_nox_.reset();

  build_solver(*x_ptr_, *jacobian_ptr_);
}

void NOX::Nln::Adapter::build_solver(
    Core::LinAlg::Vector<double>& x, Core::LinAlg::SparseOperator& jacobian)
{
  x_ptr_ = &x;
  jacobian_ptr_ = &jacobian;

  x_nox_ = std::make_shared<NOX::Nln::Vector>(
      Core::Utils::shared_ptr_from_ref(x), NOX::Nln::Vector::MemoryType::View);

  auto jac_shared = Core::Utils::shared_ptr_from_ref(jacobian);

  nox_problem_ = Teuchos::make_rcp<NOX::Nln::Problem>(nox_global_data_, x_nox_, jac_shared);
  linsys_ = nox_problem_->create_linear_system();
  group_ = nox_problem_->create_group(linsys_);
  nox_problem_->create_status_tests(ostatus_, istatus_);
  nox_solver_ = NOX::Nln::Solver::build_solver(group_, ostatus_, istatus_, *nox_global_data_);
}

FOUR_C_NAMESPACE_CLOSE

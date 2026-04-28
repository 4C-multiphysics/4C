// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_solver_nonlin_nox_adapter.hpp"

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
  class AdapterInterface : public NOX::Nln::Interface::Required,
                           public NOX::Nln::Interface::Jacobian
  {
   public:
    using ResidualCallback = NOX::Nln::Adapter::ResidualCallback;
    using JacobianCallback = NOX::Nln::Adapter::JacobianCallback;

    AdapterInterface(ResidualCallback residual_cb, JacobianCallback jacobian_cb)
        : residual_cb_(std::move(residual_cb)), jacobian_cb_(std::move(jacobian_cb))
    {
    }

    bool compute_f(const Core::LinAlg::Vector<double>& x, Core::LinAlg::Vector<double>& rhs,
        NOX::Nln::FillType fill_flag) override
    {
      FOUR_C_ASSERT_ALWAYS(residual_cb_, "Adapter: residual callback not set.");

      return (residual_cb_)(x, rhs, fill_flag);
    }

    bool compute_jacobian(
        const Core::LinAlg::Vector<double>& x, Core::LinAlg::SparseOperator& jac) override
    {
      FOUR_C_ASSERT_ALWAYS(jacobian_cb_, "Adapter: jacobian callback not set.");

      return (jacobian_cb_)(x, jac);
    }

    bool compute_f_and_jacobian(const Core::LinAlg::Vector<double>& x,
        Core::LinAlg::Vector<double>& rhs, Core::LinAlg::SparseOperator& jac) override
    {
      return compute_f(x, rhs, NOX::Nln::FillType::Residual) && compute_jacobian(x, jac);
    }

    Teuchos::RCP<Core::LinAlg::SparseMatrix>
    calc_jacobian_contributions_from_element_level_for_ptc() override
    {
      FOUR_C_THROW("Adapter: PTC is not supported.");
      return Teuchos::null;
    }

    double get_primary_rhs_norms(const Core::LinAlg::Vector<double>& residual,
        const NOX::Nln::StatusTest::QuantityType& checkquantity,
        const ::NOX::Abstract::Vector::NormType& type, const bool& isscaled) const override
    {
      (void)checkquantity;
      return NOX::Nln::Aux::calc_vector_norm(residual, type, isscaled);
    }

    double get_primary_solution_update_rms(const Core::LinAlg::Vector<double>& xnew,
        const Core::LinAlg::Vector<double>& xold, const double& aTol, const double& rTol,
        const NOX::Nln::StatusTest::QuantityType& checkquantity,
        const bool& disable_implicit_weighting) const override
    {
      (void)checkquantity;

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
      (void)checkquantity;

      Core::LinAlg::Vector<double> xincr(xold);
      xincr.update(1.0, xnew, -1.0);

      return NOX::Nln::Aux::calc_vector_norm(xincr, type, isscaled);
    }

    double get_previous_primary_solution_norms(const Core::LinAlg::Vector<double>& xold,
        const NOX::Nln::StatusTest::QuantityType& checkquantity,
        const ::NOX::Abstract::Vector::NormType& type, const bool& isscaled) const override
    {
      (void)checkquantity;
      return NOX::Nln::Aux::calc_vector_norm(xold, type, isscaled);
    }

    double calc_ref_norm_force() override { return 1.0; }

   private:
    ResidualCallback residual_cb_;
    JacobianCallback jacobian_cb_;
  };
}  // namespace

NOX::Nln::Adapter::Adapter(MPI_Comm comm, const Teuchos::ParameterList& nox_params,
    const std::map<NOX::Nln::SolutionType, Teuchos::RCP<Core::LinAlg::Solver>>& solvers,
    Core::LinAlg::Vector<double>& x, Core::LinAlg::SparseOperator& jacobian,
    NOX::Nln::Adapter::ResidualCallback residual_callback,
    NOX::Nln::Adapter::JacobianCallback jacobian_callback)
    : nox_params_(nox_params),
      x_nox_(std::make_shared<NOX::Nln::Vector>(
          Core::Utils::shared_ptr_from_ref(x), NOX::Nln::Vector::MemoryType::View)),
      jacobian_(Core::Utils::shared_ptr_from_ref(jacobian))
{
  FOUR_C_ASSERT_ALWAYS(
      !solvers.empty(), "Adapter requires at least one Core::LinAlg::Solver instance.");
  FOUR_C_ASSERT_ALWAYS(residual_callback && jacobian_callback,
      "Adapter requires non-null residual and jacobian callbacks.");

  auto interface = std::make_shared<AdapterInterface>(
      std::move(residual_callback), std::move(jacobian_callback));
  auto interface_required = std::static_pointer_cast<NOX::Nln::Interface::RequiredBase>(interface);
  auto interface_jacobian = std::static_pointer_cast<NOX::Nln::Interface::JacobianBase>(interface);

  nox_global_data_ = Teuchos::make_rcp<NOX::Nln::GlobalData>(
      comm, nox_params_, solvers, interface_required, interface_jacobian);

  build_solver();
}

unsigned int NOX::Nln::Adapter::solve()
{
  FOUR_C_ASSERT_ALWAYS(!nox_solver_.is_null(), "Adapter is not fully initialized.");
  nox_solver_->reset(*x_nox_);

  const auto status = nox_solver_->solve();
  nox_problem_->check_final_status(status);

  const auto* final_group = dynamic_cast<const NOX::Nln::Group*>(&nox_solver_->getSolutionGroup());
  FOUR_C_ASSERT_ALWAYS(final_group,
      "Adapter::solve: Expected NOX::Nln::Group from solver solution group, but dynamic_cast "
      "failed.");

  const auto* x_vector = dynamic_cast<const NOX::Nln::Vector*>(&final_group->getX());
  FOUR_C_ASSERT_ALWAYS(x_vector,
      "Adapter::solve: Expected NOX::Nln::Vector from final_group->getX(), but dynamic_cast "
      "failed.");

  x_nox_->get_linalg_vector().scale(1.0, x_vector->get_linalg_vector());
  return static_cast<unsigned int>(nox_solver_->getNumIterations());
}

void NOX::Nln::Adapter::build_solver()
{
  FOUR_C_ASSERT_ALWAYS(jacobian_, "Adapter build_solver requires bound jacobian object.");
  nox_problem_ = Teuchos::make_rcp<NOX::Nln::Problem>(nox_global_data_, x_nox_, jacobian_);
  linsys_ = nox_problem_->create_linear_system();
  group_ = nox_problem_->create_group(linsys_);
  nox_problem_->create_status_tests(ostatus_, istatus_);
  nox_solver_ = NOX::Nln::Solver::build_solver(group_, ostatus_, istatus_, *nox_global_data_);
}

FOUR_C_NAMESPACE_CLOSE

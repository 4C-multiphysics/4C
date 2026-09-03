// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_structure_new_timint_implicit.hpp"

#include "4C_adapter_str_structure.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_blocksparsematrix.hpp"
#include "4C_linalg_utils_sparse_algebra_io.hpp"
#include "4C_solver_nonlin_nox_group.hpp"
#include "4C_solver_nonlin_nox_linearsystem.hpp"
#include "4C_solver_nonlin_nox_vector.hpp"
#include "4C_structure_new_impl_generic.hpp"
#include "4C_structure_new_input.hpp"
#include "4C_structure_new_nln_solver_factory.hpp"
#include "4C_structure_new_nln_solver_generic.hpp"
#include "4C_structure_new_predict_factory.hpp"
#include "4C_structure_new_predict_generic.hpp"
#include "4C_structure_new_timint_base.hpp"
#include "4C_structure_new_timint_noxinterface.hpp"
#include "4C_structure_new_utils.hpp"
#include "4C_timestepping_time_step_control.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_exceptions.hpp"

#include <NOX_Abstract_Group.H>

#include <iostream>
#include <string>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::Implicit::setup()
{
  // safety check
  check_init();

  Solid::TimeInt::Base::setup();

  // ---------------------------------------------------------------------------
  // cast the base class integrator
  // ---------------------------------------------------------------------------
  implint_ptr_ = std::dynamic_pointer_cast<Solid::IMPLICIT::Generic>(integrator_ptr());

  // ---------------------------------------------------------------------------
  // build NOX interface
  // ---------------------------------------------------------------------------
  std::shared_ptr<Solid::TimeInt::NoxInterface> noxinterface_ptr =
      std::make_shared<Solid::TimeInt::NoxInterface>();
  noxinterface_ptr->init(
      data_global_state_ptr(), implint_ptr_, dbc_ptr(), Core::Utils::shared_ptr_from_ref(*this));
  noxinterface_ptr->setup();

  // ---------------------------------------------------------------------------
  // build predictor
  // ---------------------------------------------------------------------------
  const Solid::PredEnum predtype = data_sdyn().get_predictor_type();
  predictor_ptr_ = Solid::Predict::build_predictor(predtype);
  predictor_ptr_->init(predtype, implint_ptr_, dbc_ptr(), data_global_state_ptr(), data_io_ptr(),
      data_sdyn().get_nox_params_ptr());
  predictor_ptr_->setup();

  // ---------------------------------------------------------------------------
  // build non-linear solver
  // ---------------------------------------------------------------------------
  const Solid::NonlinSolTech nlnSolverType = data_sdyn().get_nln_solver_type();
  if (nlnSolverType == Solid::soltech_singlestep)
    std::cout << "WARNING!!! You are trying to solve implicitly using the \"singlestep\" nonlinear "
                 "solver. This is not encouraged, since it only works for linear statics analysis. "
                 "Please use NLNSOL as \"fullnewton\" for reliable result."
              << std::endl;
  nlnsolver_ptr_ = Solid::Nln::SOLVER::build_nln_solver(nlnSolverType, data_global_state_ptr(),
      data_s_dyn_ptr(), noxinterface_ptr, implint_ptr_, Core::Utils::shared_ptr_from_ref(*this));

  // set setup flag
  issetup_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::Implicit::set_state(const std::shared_ptr<Core::LinAlg::Vector<double>>& x)
{
  integrator_ptr()->set_state(*x);
  NOX::Nln::Vector x_nox(x, NOX::Nln::Vector::MemoryType::View);
  nln_solver().get_solution_group().setX(x_nox);
  set_state_in_sync_with_nox_group(true);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::Implicit::prepare_partition_step()
{
  check_init_setup();
  FOUR_C_THROW("Not yet implemented!");
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::Implicit::prepare_time_step()
{
  check_init_setup();

  ::NOX::Abstract::Group& grp = nln_solver().get_solution_group();
  predictor().predict(grp);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::Implicit::integrate()
{
  check_init_setup();
  FOUR_C_THROW(
      "The function is unused since the Adapter::StructureTimeLoop "
      "wrapper gives you all the flexibility you need.");
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::Implicit::integrate_step()
{
  check_init_setup();
  // do the predictor step
  ::NOX::Abstract::Group& grp = nln_solver().get_solution_group();
  predictor().predict(grp);
  const auto solve_status = solve();
  FOUR_C_ASSERT_ALWAYS(solve_status == Solid::StepStatus::no_errors, "Nonlinear solve failed.");
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::StepStatus Solid::TimeInt::Implicit::solve()
{
  check_init_setup();

  throw_if_state_not_in_sync_with_nox_group();
  // reset the non-linear solver
  nln_solver().reset();
  // solve the non-linear problem
  return nln_solver().solve();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::Implicit::update_state_incrementally(
    std::shared_ptr<const Core::LinAlg::Vector<double>> disiterinc)
{
  if (disiterinc == nullptr) return;

  check_init_setup();
  throw_if_state_not_in_sync_with_nox_group();
  ::NOX::Abstract::Group& grp = nln_solver().get_solution_group();

  auto* grp_ptr = dynamic_cast<NOX::Nln::Group*>(&grp);
  FOUR_C_ASSERT(grp_ptr != nullptr, "Dynamic cast failed!");

  // cast away const-qualifier for building the Nox Vector
  std::shared_ptr<Core::LinAlg::Vector<double>> mutable_disiterinc =
      Core::Utils::shared_ptr_from_ref(
          *const_cast<Core::LinAlg::Vector<double>*>(disiterinc.get()));

  // wrap the displacement vector in a NOX::Nln::Vector
  const NOX::Nln::Vector nox_disiterinc_ptr(mutable_disiterinc, NOX::Nln::Vector::MemoryType::View);

  // updated the state vector in the nox group
  grp_ptr->computeX(*grp_ptr, nox_disiterinc_ptr, 1.0);

  // Reset the state variables
  const auto& x_nox = dynamic_cast<const NOX::Nln::Vector&>(grp_ptr->getX());
  // set the consistent state in the models (e.g. structure and contact models)
  impl_int().reset_model_states(Core::LinAlg::Vector<double>(x_nox.get_linalg_vector()));
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::Implicit::determine_stress_strain() { impl_int().determine_stress_strain(); }

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::Implicit::evaluate(
    std::shared_ptr<const Core::LinAlg::Vector<double>> disiterinc)
{
  update_state_incrementally(disiterinc);

  evaluate();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::Implicit::evaluate()
{
  check_init_setup();
  throw_if_state_not_in_sync_with_nox_group();
  ::NOX::Abstract::Group& grp = nln_solver().get_solution_group();

  auto* grp_ptr = dynamic_cast<NOX::Nln::Group*>(&grp);
  FOUR_C_ASSERT(grp_ptr != nullptr, "Dynamic cast failed!");

  // you definitely have to evaluate here. You might be called from a coupled
  // problem and the group might not be aware, that a different state than
  // the internally stored displacements may have changed.
  // This is a hack to get NOX to set IsValid to false.
  grp_ptr->setX(grp_ptr->getX());

  // compute the rhs vector and the stiffness matrix
  grp_ptr->compute_f_and_jacobian();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const ::NOX::Abstract::Group& Solid::TimeInt::Implicit::get_solution_group() const
{
  check_init_setup();
  return nln_solver().get_solution_group();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::StepAction Solid::TimeInt::Implicit::perform_error_action(Solid::StepStatus solve_status)
{
  check_init_setup();

  const bool write_cout = (data_global_state().get_my_rank() == 0);

  switch (solve_status)
  {
    case StepStatus::no_errors:
      return StepAction::accept_step;
    case StepStatus::nonlinear_solver_failed:
    {
      if (write_cout) std::cout << "Nonlinear solver did not converge!\n";
      const auto divergence_action = get_divergence_action();
      switch (divergence_action)
      {
        case DivContAct::adapt_step:
        {
          prepare_retry_with_reduced_time_step();
          return StepAction::retry_step;
        }
        case DivContAct::ignore:
        {
          if (write_cout) std::cout << "WARNING: Nonlinear solver non-convergence is ignored.\n";
          return StepAction::accept_step;
        }
        case DivContAct::stop:
        {
          output(true);
          FOUR_C_THROW("Abort due to DIVERCONT: {}.", divergence_action);
        }
        case DivContAct::adapt_penaltycontact:
        {
          FOUR_C_THROW(
              "The option DIVERCONT: {} is not implemented in the new solid time integration.",
              divergence_action);
        }
        default:
        {
          FOUR_C_THROW("Inconsistent divergence action.");
        }
      }
    }
    case StepStatus::linear_solver_failed:
    {
      FOUR_C_THROW("Linear solver did not converge!");
    }
    case StepStatus::evaluation_failed:
    {
      FOUR_C_THROW("Evaluation of the residual or jacobian failed!");
    }
    default:
      FOUR_C_THROW("Inconsistent step status.");
  }
}  // PerformErrorAction()

void Solid::TimeInt::Implicit::prepare_retry_with_reduced_time_step()
{
  check_init_setup();

  const double new_dt = TimeStepping::compute_reduced_time_step(
      get_delta_time(), get_data_sdyn().time_step_control_settings());

  if (get_data_global_state().get_my_rank() == 0)
  {
    std::cout << "Retrying the current time-step with a reduced time-step size.\n";
  }
  set_new_time_step_size(new_dt);

  reset_step();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::Implicit::finalize_successful_step()
{
  ImplicitBase::finalize_successful_step();

  time_step_control_after_successful_step();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::Implicit::time_step_control_after_successful_step()
{
  check_init_setup();

  const auto& settings = get_data_sdyn().time_step_control_settings();

  num_steps_at_current_dt_++;
  number_of_nonlinear_iterations_per_step_.emplace_back(nln_solver().get_num_nln_iterations());
  if (number_of_nonlinear_iterations_per_step_.size() > settings.steps_to_increase)
  {
    number_of_nonlinear_iterations_per_step_.pop_front();
  }

  const double new_dt = TimeStepping::compute_time_step_after_successful_step(get_delta_time(),
      settings, num_steps_at_current_dt_, number_of_nonlinear_iterations_per_step_,
      get_time_end() - get_time_n(), get_step_end() - get_step_n());

  if (std::abs(new_dt - get_delta_time()) > std::numeric_limits<double>::epsilon())
  {
    if (get_data_global_state().get_my_rank() == 0)
    {
      std::cout << "Updating the time-step size for the next step:\n";
    }
    set_new_time_step_size(new_dt);
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::Implicit::set_new_time_step_size(const double new_dt)
{
  check_init_setup();

  const double old_dt = get_delta_time();

  FOUR_C_ASSERT(new_dt > 0.0, "Time-step size must be positive.");
  FOUR_C_ASSERT(std::abs(new_dt - old_dt) > std::numeric_limits<double>::epsilon(),
      "New time-step size must be different from the current time-step size.");

  set_delta_time(new_dt);
  set_time_np(get_time_n() + get_delta_time());

  num_steps_at_current_dt_ = 0;

  integrator().update_constant_state_contributions();

  if (data_global_state().get_my_rank() == 0)
  {
    std::cout << "Old time-step size: " << old_dt << "\n"
              << "New time-step size: " << new_dt << "\n"
              << std::string(72, '*') << "\n";
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::Implicit::print_jacobian_in_matlab_format(
    const NOX::Nln::Group& curr_grp) const
{
  using LinalgBlockSparseMatrix =
      Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>;

  if (not get_data_io().is_write_jacobian_to_matlab()) return;

  // create file name
  std::stringstream filebase;

  filebase << "str_jacobian" << "_step-" << get_data_global_state().get_step_np() << "_nlniter-"
           << nlnsolver_ptr_->get_num_nln_iterations();

  std::stringstream filename;
  filename << get_data_io().get_output_ptr()->output().file_name() << "_" << filebase.str()
           << ".mtl";

  if (get_data_global_state().get_my_rank() == 0)
    std::cout << "Writing structural jacobian to \"" << filename.str() << "\"\n";

  Teuchos::RCP<const NOX::Nln::LinearSystemBase> linear_system = curr_grp.get_linear_system();

  Teuchos::RCP<const NOX::Nln::LinearSystem> nln_lin_system =
      Teuchos::rcp_dynamic_cast<const NOX::Nln::LinearSystem>(linear_system, true);

  const NOX::Nln::LinSystem::OperatorType jac_type = nln_lin_system->get_jacobian_operator_type();

  auto jac_ptr = nln_lin_system->get_jacobian_operator();

  switch (jac_type)
  {
    case NOX::Nln::LinSystem::LinalgSparseMatrix:
    {
      auto sparse_matrix = std::dynamic_pointer_cast<const Core::LinAlg::SparseMatrix>(jac_ptr);
      Core::LinAlg::print_matrix_in_matlab_format(filename.str().c_str(), *sparse_matrix);

      break;
    }
    case NOX::Nln::LinSystem::LinalgBlockSparseMatrix:
    {
      auto block_matrix = std::dynamic_pointer_cast<const LinalgBlockSparseMatrix>(jac_ptr);
      Core::LinAlg::print_block_matrix_in_matlab_format(filename.str(), *block_matrix);

      break;
    }
    default:
    {
      FOUR_C_THROW("Unsupported NOX::Nln::LinSystem::OperatorType: \"{}\"",
          NOX::Nln::LinSystem::operator_type_to_string(jac_type));
    }
  }
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::DynamicType Solid::TimeInt::Implicit::method_name() const
{
  return implint_ptr_->method_name();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
int Solid::TimeInt::Implicit::method_steps() const { return implint_ptr_->method_steps(); }

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
int Solid::TimeInt::Implicit::method_order_of_accuracy_dis() const
{
  return implint_ptr_->method_order_of_accuracy_dis();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
int Solid::TimeInt::Implicit::method_order_of_accuracy_vel() const
{
  return implint_ptr_->method_order_of_accuracy_vel();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double Solid::TimeInt::Implicit::method_lin_err_coeff_dis() const
{
  return implint_ptr_->method_lin_err_coeff_dis();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double Solid::TimeInt::Implicit::method_lin_err_coeff_vel() const
{
  return implint_ptr_->method_lin_err_coeff_vel();
}

FOUR_C_NAMESPACE_CLOSE

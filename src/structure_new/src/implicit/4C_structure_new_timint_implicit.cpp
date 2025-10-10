// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_structure_new_timint_implicit.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_io_pstream.hpp"
#include "4C_linalg_blocksparsematrix.hpp"
#include "4C_linalg_utils_sparse_algebra_print.hpp"
#include "4C_solver_nonlin_nox_group.hpp"
#include "4C_solver_nonlin_nox_linearsystem.hpp"
#include "4C_structure_new_impl_generic.hpp"
#include "4C_structure_new_nln_solver_factory.hpp"
#include "4C_structure_new_nln_solver_generic.hpp"
#include "4C_structure_new_predict_factory.hpp"
#include "4C_structure_new_predict_generic.hpp"
#include "4C_structure_new_timint_noxinterface.hpp"
#include "4C_structure_new_utils.hpp"

#include <NOX_Abstract_Group.H>

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
  const enum Inpar::Solid::PredEnum predtype = data_sdyn().get_predictor_type();
  predictor_ptr_ = Solid::Predict::build_predictor(predtype);
  predictor_ptr_->init(predtype, implint_ptr_, dbc_ptr(), data_global_state_ptr(), data_io_ptr(),
      data_sdyn().get_nox_params_ptr());
  predictor_ptr_->setup();

  // ---------------------------------------------------------------------------
  // build non-linear solver
  // ---------------------------------------------------------------------------
  const enum Inpar::Solid::NonlinSolTech nlnSolverType = data_sdyn().get_nln_solver_type();
  if (nlnSolverType == Inpar::Solid::soltech_singlestep)
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
  ::NOX::Epetra::Vector x_nox(
      Teuchos::rcpFromRef(x->get_ref_of_epetra_vector()), ::NOX::Epetra::Vector::CreateView);
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
  // update end time \f$t_{n+1}\f$ of this time step to cope with time step size adaptivity
  /* ToDo Check if this is still necessary. I moved this part to the Update(const double endtime)
   * routine, such it becomes consistent with non-adaptive update routine! See the
   * update_step_time() routine for more information.                             hiermeier 12/2015
   *
  double& time_np = data_global_state().get_time_np();
  time_np = data_global_state().get_time_n() + (*data_global_state().get_delta_time())[0]; */

  ::NOX::Abstract::Group& grp = nln_solver().get_solution_group();
  predictor().predict(grp);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
int Solid::TimeInt::Implicit::integrate()
{
  check_init_setup();
  FOUR_C_THROW(
      "The function is unused since the Adapter::StructureTimeLoop "
      "wrapper gives you all the flexibility you need.");
  return 0;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
int Solid::TimeInt::Implicit::integrate_step()
{
  check_init_setup();
  // do the predictor step
  ::NOX::Abstract::Group& grp = nln_solver().get_solution_group();
  predictor().predict(grp);
  return solve();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Inpar::Solid::ConvergenceStatus Solid::TimeInt::Implicit::solve()
{
  check_init_setup();
  throw_if_state_not_in_sync_with_nox_group();
  // reset the non-linear solver
  nln_solver().reset();
  // solve the non-linear problem
  Inpar::Solid::ConvergenceStatus convstatus = nln_solver().solve();
  // return convergence status
  return perform_error_action(convstatus);
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

  // wrap the displacement vector in a nox_epetra_Vector
  const ::NOX::Epetra::Vector nox_disiterinc_ptr(
      Teuchos::rcpFromRef(mutable_disiterinc->get_ref_of_epetra_vector()),
      ::NOX::Epetra::Vector::CreateView);

  // updated the state vector in the nox group
  grp_ptr->computeX(*grp_ptr, nox_disiterinc_ptr, 1.0);

  // Reset the state variables
  const auto& x_eptra = dynamic_cast<const ::NOX::Epetra::Vector&>(grp_ptr->getX());
  // set the consistent state in the models (e.g. structure and contact models)
  impl_int().reset_model_states(Core::LinAlg::Vector<double>(x_eptra.getEpetraVector()));
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
Inpar::Solid::ConvergenceStatus Solid::TimeInt::Implicit::perform_error_action(
    Inpar::Solid::ConvergenceStatus nonlinsoldiv)
{
  check_init_setup();

  if (nonlinsoldiv == Inpar::Solid::conv_success)
  {
    // Only relevant, if the input parameter DIVERCONT is used and set to divcontype_ == adapt_step:
    // In this case, the time step size is halved as consequence of a non-converging nonlinear
    // solver. After a prescribed number of converged time steps, the time step is doubled again.
    // The following methods checks, if the time step size can be increased again.
    check_for_time_step_increase(nonlinsoldiv);
    return Inpar::Solid::conv_success;
  }
  // get ID of actual processor in parallel
  const int& myrank = data_global_state().get_my_rank();

  // what to do when nonlinear solver does not converge
  switch (get_divergence_action())
  {
    case Inpar::Solid::divcont_stop:
    {
      // write restart output of last converged step before stopping
      output(true);

      // we should not get here, FOUR_C_THROW for safety
      FOUR_C_THROW("Nonlinear solver did not converge! ");
      return Inpar::Solid::conv_nonlin_fail;
      break;
    }
    case Inpar::Solid::divcont_continue:
    {
      if (myrank == 0)
      {
        Core::IO::cout
            << "\n WARNING: You are continuing your simulation although the nonlinear solver\n"
               " did not converge in the current time step.\n"
            << Core::IO::endl;
      }
      return Inpar::Solid::conv_success;
      break;
    }
    case Inpar::Solid::divcont_repeat_step:
    {
      if (myrank == 0)
        Core::IO::cout << "Nonlinear solver failed to converge repeat time step" << Core::IO::endl;

      // reset step (e.g. quantities on element level or model specific stuff)
      reset_step();

      return Inpar::Solid::conv_fail_repeat;
      break;
    }
    case Inpar::Solid::divcont_halve_step:
    {
      if (myrank == 0)
      {
        Core::IO::cout << "Nonlinear solver failed to converge at time t= " << get_time_np()
                       << ". Divide timestep in half. "
                       << "Old time step: " << get_delta_time() << Core::IO::endl
                       << "New time step: " << 0.5 * get_delta_time() << Core::IO::endl
                       << Core::IO::endl;
      }

      // halve the time step size
      set_delta_time(get_delta_time() * 0.5);
      // update the number of max time steps if it does not exceed the largest possible value for
      // the type int
      if ((get_step_end() - get_step_np() + 1) > std::numeric_limits<int>::max() - get_step_end())
        FOUR_C_THROW(" Your updated step number exceeds largest possible value for type int");
      int endstep = get_step_end() + (get_step_end() - get_step_np()) + 1;
      set_step_end(endstep);
      // reset timen_ because it is set in the constructor
      set_time_np(get_time_n() + get_delta_time());
      // reset step (e.g. quantities on element level or model specific stuff)
      reset_step();

      integrator().update_constant_state_contributions();

      return Inpar::Solid::conv_fail_repeat;
      break;
    }
    case Inpar::Solid::divcont_adapt_step:
    {
      if (myrank == 0)
      {
        Core::IO::cout << "Nonlinear solver failed to converge at time t= " << get_time_np()
                       << ". Divide timestep in half. "
                       << "Old time step: " << get_delta_time() << Core::IO::endl
                       << "New time step: " << 0.5 * get_delta_time() << Core::IO::endl
                       << Core::IO::endl;
      }

      // halve the time step size
      set_delta_time(get_delta_time() * 0.5);
      // update the number of max time steps if it does not exceed the largest possible value for
      // the type int
      if ((get_step_end() - get_step_np() + 1) > std::numeric_limits<int>::max() - get_step_end())
        FOUR_C_THROW(" Your updated step number exceeds largest possible value for type int");
      int endstep = get_step_end() + (get_step_end() - get_step_np()) + 1;
      set_step_end(endstep);
      // reset timen_ because it is set in the constructor
      set_time_np(get_time_n() + get_delta_time());

      set_div_con_refine_level(get_div_con_refine_level() + 1);
      set_div_con_num_fine_step(0);

      if (get_div_con_refine_level() == get_max_div_con_refine_level())
        FOUR_C_THROW(
            "Maximal divercont refinement level reached. Adapt your time basic time step size!");

      // reset step (e.g. quantities on element level or model specific stuff)
      reset_step();

      integrator().update_constant_state_contributions();

      return Inpar::Solid::conv_fail_repeat;
      break;
    }
    case Inpar::Solid::divcont_rand_adapt_step:
    case Inpar::Solid::divcont_rand_adapt_step_ele_err:
    {
      // generate random number between 0.51 and 1.99 (as mean value of random
      // numbers generated on all processors), alternating values larger
      // and smaller than 1.0
      double proc_randnum_get = ((double)rand() / (double)RAND_MAX);
      double proc_randnum = proc_randnum_get;
      double randnum = 1.0;
      MPI_Comm comm = discretization()->get_comm();
      randnum = Core::Communication::sum_all(proc_randnum, comm);
      const double numproc = Core::Communication::num_mpi_ranks(comm);
      randnum /= numproc;
      if (get_random_time_step_factor() > 1.0)
        set_random_time_step_factor(randnum * 0.49 + 0.51);
      else if (get_random_time_step_factor() < 1.0)
        set_random_time_step_factor(randnum * 0.99 + 1.0);
      else
        set_random_time_step_factor(randnum * 1.48 + 0.51);

      if (myrank == 0)
      {
        Core::IO::cout << "Nonlinear solver failed to converge: modifying time-step size by random "
                          "number between 0.51 and 1.99 -> here: "
                       << get_random_time_step_factor() << " !" << Core::IO::endl;
      }
      // multiply time-step size by random number
      set_delta_time(get_delta_time() * get_random_time_step_factor());
      // update maximum number of time steps
      int endstep = (1.0 / get_random_time_step_factor()) * get_step_end() +
                    (1.0 - (1.0 / get_random_time_step_factor())) * get_step_np() + 1;
      if (endstep > std::numeric_limits<int>::max())
        FOUR_C_THROW(" Your updated step number exceeds largest possible value for type int");
      set_step_end(endstep);
      // reset timen_ because it is set in the constructor
      set_time_np(get_time_n() + get_delta_time());
      // reset step (e.g. quantities on element level or model specific stuff)
      reset_step();

      integrator().update_constant_state_contributions();

      return Inpar::Solid::conv_fail_repeat;
      break;
    }
    case Inpar::Solid::divcont_adapt_penaltycontact:
    {
      // adapt penalty and search parameter
      FOUR_C_THROW("Not yet implemented for new structure time integration");
      break;
    }
    case Inpar::Solid::divcont_repeat_simulation:
    {
      if (nonlinsoldiv == Inpar::Solid::conv_nonlin_fail and myrank == 0)
      {
        Core::IO::cout << "Nonlinear solver failed to converge and DIVERCONT = "
                          "repeat_simulation, hence leaving structural time integration "
                       << Core::IO::endl;
      }
      else if (nonlinsoldiv == Inpar::Solid::conv_lin_fail and myrank == 0)
      {
        Core::IO::cout << "Linear solver failed to converge and DIVERCONT = "
                          "repeat_simulation, hence leaving structural time integration "
                       << Core::IO::endl;
      }
      else if (nonlinsoldiv == Inpar::Solid::conv_ele_fail and myrank == 0)
      {
        Core::IO::cout
            << "Element failure in form of a negative Jacobian determinant and DIVERCONT = "
               "repeat_simulation, hence leaving structural time integration "
            << Core::IO::endl;
      }
      return nonlinsoldiv;  // so that time loop will be aborted
      break;
    }
    default:
      FOUR_C_THROW("Unknown DIVER_CONT case");
      return Inpar::Solid::conv_nonlin_fail;
      break;
  }
  return Inpar::Solid::conv_success;  // make compiler happy
}  // PerformErrorAction()

/*-----------------------------------------------------------------------------*
 * check, if according to divercont flag                             meier 01/15
 * time step size can be increased
 *-----------------------------------------------------------------------------*/
void Solid::TimeInt::Implicit::check_for_time_step_increase(Inpar::Solid::ConvergenceStatus& status)
{
  check_init_setup();

  const int maxnumfinestep = 4;

  if (get_divergence_action() != Inpar::Solid::divcont_adapt_step)
    return;
  else if (status == Inpar::Solid::conv_success and get_div_con_refine_level() != 0)
  {
    set_div_con_num_fine_step(get_div_con_num_fine_step() + 1);

    if (get_div_con_num_fine_step() == maxnumfinestep)
    {
      // increase the step size if the remaining number of steps is a even number
      if (((get_step_end() - get_step_np()) % 2) == 0 and get_step_end() != get_step_np())
      {
        if (data_global_state().get_my_rank() == 0)
          Core::IO::cout << "Nonlinear solver successful. Double timestep size!" << Core::IO::endl;

        set_div_con_refine_level(get_div_con_refine_level() - 1);
        set_div_con_num_fine_step(0);

        set_step_end(get_step_end() - (get_step_end() - get_step_np()) / 2);

        // double the time step size
        set_delta_time(get_delta_time() * 2.0);
      }
      else  // otherwise we have to wait one more time step until the step size can be increased
      {
        set_div_con_num_fine_step(get_div_con_num_fine_step() - 1);
      }
    }
    return;
  }
}  // check_for_time_step_increase()

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

  filebase << "str_jacobian"
           << "_step-" << get_data_global_state().get_step_np() << "_nlniter-"
           << nlnsolver_ptr_->get_num_nln_iterations();

  std::stringstream filename;
  filename << get_data_io().get_output_ptr()->output()->file_name() << "_" << filebase.str()
           << ".mtl";

  if (get_data_global_state().get_my_rank() == 0)
    std::cout << "Writing structural jacobian to \"" << filename.str() << "\"\n";

  Teuchos::RCP<const NOX::Nln::LinearSystemBase> linear_system = curr_grp.get_linear_system();

  Teuchos::RCP<const NOX::Nln::LinearSystem> nln_lin_system =
      Teuchos::rcp_dynamic_cast<const NOX::Nln::LinearSystem>(linear_system, true);

  const enum NOX::Nln::LinSystem::OperatorType jac_type =
      nln_lin_system->get_jacobian_operator_type();

  Teuchos::RCP<const Epetra_Operator> jac_ptr = nln_lin_system->get_jacobian_operator();

  switch (jac_type)
  {
    case NOX::Nln::LinSystem::LinalgSparseMatrix:
    {
      Teuchos::RCP<const Core::LinAlg::SparseMatrix> sparse_matrix =
          Teuchos::rcp_dynamic_cast<const Core::LinAlg::SparseMatrix>(jac_ptr, true);
      Core::LinAlg::print_matrix_in_matlab_format(filename.str().c_str(), *sparse_matrix);

      break;
    }
    case NOX::Nln::LinSystem::LinalgBlockSparseMatrix:
    {
      Teuchos::RCP<const LinalgBlockSparseMatrix> block_matrix =
          Teuchos::rcp_dynamic_cast<const LinalgBlockSparseMatrix>(jac_ptr, true);
      Core::LinAlg::print_block_matrix_in_matlab_format(filename.str(), *block_matrix);

      break;
    }
    default:
    {
      FOUR_C_THROW("Unsupported NOX::Nln::LinSystem::OperatorType: \"{}\"",
          NOX::Nln::LinSystem::operator_type_to_string(jac_type));
    }
  }

  // print sparsity pattern to file
  //  Core::LinAlg::PrintSparsityToPostscript( *(system_matrix()->EpetraMatrix()) );
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
enum Inpar::Solid::DynamicType Solid::TimeInt::Implicit::method_name() const
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

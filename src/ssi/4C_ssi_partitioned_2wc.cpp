// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_ssi_partitioned_2wc.hpp"

#include "4C_adapter_scatra_base_algorithm.hpp"
#include "4C_adapter_str_ssiwrapper.hpp"
#include "4C_contact_nitsche_strategy_ssi.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_scatra_ele.hpp"
#include "4C_scatra_timint_implicit.hpp"
#include "4C_structure_new_model_evaluator_data.hpp"

#include <Teuchos_Time.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | constructor                                               Thon 12/14 |
 *----------------------------------------------------------------------*/
SSI::SSIPart2WC::SSIPart2WC(MPI_Comm comm, const Teuchos::ParameterList& globaltimeparams)
    : SSIPart(comm, globaltimeparams)
{
  // Keep this constructor empty!
  // First do everything on the more basic objects like the discretizations, like e.g.
  // redistribution of elements. Only then call the setup to this class. This will call the setup to
  // all classes in the inheritance hierarchy. This way, this class may also override a method that
  // is called during setup() in a base class.
}

/*----------------------------------------------------------------------*
 | Init this class                                          rauch 08/16 |
 *----------------------------------------------------------------------*/
void SSI::SSIPart2WC::init(MPI_Comm comm, const Teuchos::ParameterList& globaltimeparams,
    const Teuchos::ParameterList& scatraparams, const Teuchos::ParameterList& structparams,
    const std::string& struct_disname, const std::string& scatra_disname, bool isAle)
{
  // call setup of base class
  SSI::SSIPart::init(
      comm, globaltimeparams, scatraparams, structparams, struct_disname, scatra_disname, isAle);

  // call the SSI parameter lists
  const Teuchos::ParameterList& ssicontrol = Global::Problem::instance()->ssi_control_params();
  const Teuchos::ParameterList& ssicontrolpart =
      Global::Problem::instance()->ssi_control_params().sublist("PARTITIONED");

  // do some checks
  {
    auto structtimealgo =
        Teuchos::getIntegralValue<Inpar::Solid::DynamicType>(structparams, "DYNAMICTYPE");
    if (structtimealgo == Inpar::Solid::DynamicType::Statics)
    {
      FOUR_C_THROW(
          "If you use statics as the structural time integrator no velocities will be calculated "
          "and hence"
          "the deformations will not be applied to the scalar transport problem!");
    }
  }

  auto convform = Teuchos::getIntegralValue<Inpar::ScaTra::ConvForm>(scatraparams, "CONVFORM");
  if (convform == Inpar::ScaTra::convform_convective)
  {
    FOUR_C_THROW(
        "If the scalar transport problem is solved on the deforming domain, the conservative "
        "form must be used to include volume changes. Set 'CONVFORM' to 'conservative' in the "
        "SCALAR TRANSPORT DYNAMIC section.");
  }

  if (diff_time_step_size())
  {
    FOUR_C_THROW("Different time stepping for two way coupling not implemented yet.");
  }

  // Get the parameters for the convergence_check
  itmax_ = ssicontrol.get<int>("ITEMAX");          // default: =10
  ittol_ = ssicontrolpart.get<double>("CONVTOL");  // default: =1e-6
}

/*----------------------------------------------------------------------*
 | Setup this class                                          rauch 08/16 |
 *----------------------------------------------------------------------*/
void SSI::SSIPart2WC::setup()
{
  // call setup in base class
  SSI::SSIPart::setup();

  // construct increment vectors
  scaincnp_ = std::make_shared<Core::LinAlg::Vector<double>>(
      *scatra_field()->discretization()->dof_row_map(0), true);
  dispincnp_ =
      std::make_shared<Core::LinAlg::Vector<double>>(*structure_field()->dof_row_map(0), true);
}

/*----------------------------------------------------------------------*
 | Timeloop for 2WC SSI problems                             Thon 12/14 |
 *----------------------------------------------------------------------*/
void SSI::SSIPart2WC::timeloop()
{
  // safety checks
  check_is_init();
  check_is_setup();

  prepare_time_loop();

  // time loop
  while (not_finished() and scatra_field()->not_finished())
  {
    prepare_time_step();

    // store time before calling nonlinear solver
    double time = Teuchos::Time::wallTime();

    outer_loop();

    // determine time spent by nonlinear solver and take maximum over all processors via
    // communication
    double mydtnonlinsolve(Teuchos::Time::wallTime() - time), dtnonlinsolve(0.);
    dtnonlinsolve = Core::Communication::max_all(mydtnonlinsolve, get_comm());

    // output performance statistics associated with nonlinear solver into *.csv file if applicable
    if (scatra_field()->scatra_parameter_list()->get<bool>("OUTPUTNONLINSOLVERSTATS"))
      scatra_field()->output_nonlin_solver_stats(
          iteration_count(), dtnonlinsolve, step(), get_comm());

    update_and_output();
  }
}

/*----------------------------------------------------------------------*
 | Solve structure filed                                     Thon 12/14 |
 *----------------------------------------------------------------------*/
void SSI::SSIPart2WC::do_struct_step()
{
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    std::cout << "\n***********************\n STRUCTURE SOLVER \n***********************\n";
  }

  // Newton-Raphson iteration
  structure_field()->solve();

  if (is_s2i_kinetics_with_pseudo_contact()) structure_field()->determine_stress_strain();

  //  set mesh displacement and velocity fields
  return set_struct_solution(*structure_field()->dispnp(), structure_field()->velnp(),
      is_s2i_kinetics_with_pseudo_contact());
}

/*----------------------------------------------------------------------*
 | Solve Scatra field                                        Thon 12/14 |
 *----------------------------------------------------------------------*/
void SSI::SSIPart2WC::do_scatra_step()
{
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    std::cout << "\n***********************\n  TRANSPORT SOLVER \n***********************\n";
  }

  // -------------------------------------------------------------------
  //                  solve nonlinear / linear equation
  // -------------------------------------------------------------------
  scatra_field()->solve();

  // set structure-based scalar transport values
  set_scatra_solution(scatra_field()->phinp());

  // set micro scale value (projected to macro scale) to structure field
  if (macro_scale()) set_micro_scatra_solution(scatra_field()->phinp_micro());

  // evaluate temperature from function and set to structural discretization
  evaluate_and_set_temperature_field();
}

/*----------------------------------------------------------------------*/
// prepare time step
/*----------------------------------------------------------------------*/
void SSI::SSIPart2WC::prepare_time_loop()
{
  // initial output
  constexpr bool force_prepare = true;
  structure_field()->prepare_output(force_prepare);
  structure_field()->output();
  set_struct_solution(*structure_field()->dispnp(), structure_field()->velnp(), false);
  scatra_field()->prepare_time_loop();
}

/*----------------------------------------------------------------------*/
// prepare time step
/*----------------------------------------------------------------------*/
void SSI::SSIPart2WC::prepare_time_step(bool printheader)
{
  increment_time_and_step();

  set_struct_solution(*structure_field()->dispnp(), structure_field()->velnp(), false);
  scatra_field()->prepare_time_step();

  // if adaptive time stepping and different time step size: calculate time step in scatra
  // (prepare_time_step() of Scatra) and pass to other fields
  if (scatra_field()->time_step_adapted()) set_dt_from_scatra_to_ssi();

  set_scatra_solution(scatra_field()->phinp());
  if (macro_scale()) set_micro_scatra_solution(scatra_field()->phinp_micro());

  // NOTE: the predictor of the structure is called in here
  structure_field()->prepare_time_step();

  if (printheader) print_header();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSIPart2WC::update_and_output()
{
  constexpr bool force_prepare = false;
  structure_field()->prepare_output(force_prepare);

  if (ssi_interface_contact())
  {
    // re-evaluate the contact to re-obtain the displ state
    const auto& model_eval = structure_field()->model_evaluator(Inpar::Solid::model_structure);
    const auto& cparams = model_eval.eval_data().contact_ptr();
    nitsche_strategy_ssi()->integrate(*cparams);
  }

  structure_field()->update();
  scatra_field()->update();

  scatra_field()->evaluate_error_compared_to_analytical_sol();

  structure_field()->output();
  scatra_field()->check_and_write_output_and_restart();
}


/*----------------------------------------------------------------------*
 | update the current states in every iteration             rauch 05/16 |
 *----------------------------------------------------------------------*/
void SSI::SSIPart2WC::iter_update_states()
{
  // store last solutions (current states).
  // will be compared in convergence_check to the solutions,
  // obtained from the next Struct and Scatra steps.
  scaincnp_->update(1.0, *scatra_field()->phinp(), 0.0);
  dispincnp_->update(1.0, *structure_field()->dispnp(), 0.0);
}  // iter_update_states()


/*----------------------------------------------------------------------*
 | Outer Timeloop for 2WC SSi without relaxation            rauch 06/17 |
 *----------------------------------------------------------------------*/
void SSI::SSIPart2WC::outer_loop()
{
  // reset iteration number
  reset_iteration_count();
  bool stopnonliniter = false;

  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    std::cout << "\n****************************************\n          OUTER ITERATION "
                 "LOOP\n****************************************\n";
  }

  while (!stopnonliniter)
  {
    // increment iteration number
    increment_iteration_count();

    // update the states to the last solutions obtained
    iter_update_states();

    // standard ssi_2wc :
    // 1.) solve structural system
    // 2.) set disp and vel states in scatra field
    pre_operator1();
    operator1();
    post_operator1();

    // standard ssi_2wc :
    // 1.) solve scalar transport equation
    // 2.) set phi state in structure field
    pre_operator2();
    operator2();
    post_operator2();

    // check convergence for all fields
    // stop iteration loop if converged
    stopnonliniter = convergence_check(iteration_count());
  }
}

/*----------------------------------------------------------------------*
 | convergence check for both fields (scatra & structure) (copied form tsi)
 *----------------------------------------------------------------------*/
bool SSI::SSIPart2WC::convergence_check(int itnum)
{
  // convergence check based on the scalar increment
  bool stopnonliniter = false;

  //    | scalar increment |_2
  //  -------------------------------- < Tolerance
  //    | scalar state n+1 |_2
  //
  // AND
  //
  //    | scalar increment |_2
  //  -------------------------------- < Tolerance , with n := global length of vector
  //        dt * sqrt(n)
  //
  // The same is checked for the structural displacements.
  //

  // variables to save different L2 - Norms
  // define L2-norm of incremental scalar and scalar
  double scaincnorm_L2(0.0);
  double scanorm_L2(0.0);
  double dispincnorm_L2(0.0);
  double dispnorm_L2(0.0);

  // build the current scalar increment Inc T^{i+1}
  // \f Delta T^{k+1} = Inc T^{k+1} = T^{k+1} - T^{k}  \f
  scaincnp_->update(1.0, *(scatra_field()->phinp()), -1.0);
  dispincnp_->update(1.0, *(structure_field()->dispnp()), -1.0);

  // build the L2-norm of the scalar increment and the scalar
  scaincnp_->norm_2(&scaincnorm_L2);
  scatra_field()->phinp()->norm_2(&scanorm_L2);
  dispincnp_->norm_2(&dispincnorm_L2);
  structure_field()->dispnp()->norm_2(&dispnorm_L2);

  // care for the case that there is (almost) zero scalar
  if (scanorm_L2 < 1e-6) scanorm_L2 = 1.0;
  if (dispnorm_L2 < 1e-6) dispnorm_L2 = 1.0;

  // print the incremental based convergence check to the screen
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    std::cout << "\n";
    std::cout
        << "***********************************************************************************\n";
    std::cout << "    OUTER ITERATION STEP    \n";
    std::cout
        << "***********************************************************************************\n";
    printf(
        "+--------------+---------------------+----------------+------------------+----------------"
        "----+------------------+\n");
    printf(
        "|-  step/max  -|-  tol      [norm]  -|-  scalar-inc  -|-  disp-inc      -|-  "
        "scalar-rel-inc  -|-  disp-rel-inc  -|\n");
    printf(
        "|   %3d/%3d    |  %10.3E[L_2 ]   |  %10.3E    |  %10.3E      |  %10.3E        |  %10.3E   "
        "   |",
        itnum, itmax_, ittol_, scaincnorm_L2 / dt() / sqrt(scaincnp_->global_length()),
        dispincnorm_L2 / dt() / sqrt(dispincnp_->global_length()), scaincnorm_L2 / scanorm_L2,
        dispincnorm_L2 / dispnorm_L2);
    printf("\n");
    printf(
        "+--------------+---------------------+----------------+------------------+----------------"
        "----+------------------+\n");
  }

  // converged
  if (((scaincnorm_L2 / scanorm_L2) <= ittol_) and ((dispincnorm_L2 / dispnorm_L2) <= ittol_) and
      ((dispincnorm_L2 / dt() / sqrt(dispincnp_->global_length())) <= ittol_) and
      ((scaincnorm_L2 / dt() / sqrt(scaincnp_->global_length())) <= ittol_))
  {
    stopnonliniter = true;
    if (Core::Communication::my_mpi_rank(get_comm()) == 0)
    {
      printf(
          "|  Outer Iteration loop converged after iteration %3d/%3d !                             "
          "                         |\n",
          itnum, itmax_);
      printf(
          "+--------------+---------------------+----------------+------------------+--------------"
          "------+------------------+\n");
    }
  }

  // stop if itemax is reached without convergence
  // timestep
  if ((itnum == itmax_) and
      (((scaincnorm_L2 / scanorm_L2) > ittol_) or ((dispincnorm_L2 / dispnorm_L2) > ittol_) or
          ((dispincnorm_L2 / dt() / sqrt(dispincnp_->global_length())) > ittol_) or
          (scaincnorm_L2 / dt() / sqrt(scaincnp_->global_length())) > ittol_))
  {
    stopnonliniter = true;
    if ((Core::Communication::my_mpi_rank(get_comm()) == 0))
    {
      printf(
          "|     >>>>>> not converged in itemax steps!                                             "
          "                         |\n");
      printf(
          "+--------------+---------------------+----------------+------------------+--------------"
          "------+------------------+\n");
      printf("\n");
      printf("\n");
    }
    FOUR_C_THROW("The partitioned SSI solver did not converge in ITEMAX steps!");
  }

  return stopnonliniter;
}

/*----------------------------------------------------------------------*
 | calculate velocities by a FD approximation                Thon 14/11 |
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> SSI::SSIPart2WC::calc_velocity(
    const Core::LinAlg::Vector<double>& dispnp)
{
  std::shared_ptr<Core::LinAlg::Vector<double>> vel = nullptr;
  // copy D_n onto V_n+1
  vel = std::make_shared<Core::LinAlg::Vector<double>>(*(structure_field()->dispn()));
  // calculate velocity with timestep Dt()
  //  V_n+1^k = (D_n+1^k - D_n) / Dt
  vel->update(1. / dt(), dispnp, -1. / dt());

  return vel;
}  // calc_velocity()

/*----------------------------------------------------------------------*
 | Constructor                                               Thon 12/14 |
 *----------------------------------------------------------------------*/
SSI::SSIPart2WCSolidToScatraRelax::SSIPart2WCSolidToScatraRelax(
    MPI_Comm comm, const Teuchos::ParameterList& globaltimeparams)
    : SSIPart2WC(comm, globaltimeparams), omega_(-1.0)
{
  // Keep this constructor empty!
  // First do everything on the more basic objects like the discretizations, like e.g.
  // redistribution of elements. Only then call the setup to this class. This will call the setup to
  // all classes in the inheritance hierarchy. This way, this class may also override a method that
  // is called during setup() in a base class.
}

/*----------------------------------------------------------------------*
 | Init this class                                          rauch 08/16 |
 *----------------------------------------------------------------------*/
void SSI::SSIPart2WCSolidToScatraRelax::init(MPI_Comm comm,
    const Teuchos::ParameterList& globaltimeparams, const Teuchos::ParameterList& scatraparams,
    const Teuchos::ParameterList& structparams, const std::string& struct_disname,
    const std::string& scatra_disname, bool isAle)
{
  // call init of base class
  SSI::SSIPart2WC::init(
      comm, globaltimeparams, scatraparams, structparams, struct_disname, scatra_disname, isAle);

  const Teuchos::ParameterList& ssicontrolpart =
      Global::Problem::instance()->ssi_control_params().sublist("PARTITIONED");

  // Get minimal relaxation parameter from input file
  omega_ = ssicontrolpart.get<double>("STARTOMEGA");
}

/*----------------------------------------------------------------------*
 | Outer Timeloop for 2WC SSi with relaxed displacements     Thon 12/14 |
 *----------------------------------------------------------------------*/
void SSI::SSIPart2WCSolidToScatraRelax::outer_loop()
{
  reset_iteration_count();
  bool stopnonliniter = false;

  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    std::cout << "\n****************************************\n          OUTER ITERATION "
                 "LOOP\n****************************************\n";
  }

  // these are the relaxed inputs
  std::shared_ptr<Core::LinAlg::Vector<double>> dispnp =
      std::make_shared<Core::LinAlg::Vector<double>>(*(structure_field()->dof_row_map(0)), true);
  std::shared_ptr<Core::LinAlg::Vector<double>> velnp =
      std::make_shared<Core::LinAlg::Vector<double>>(*(structure_field()->dof_row_map(0)), true);

  while (!stopnonliniter)
  {
    increment_iteration_count();

    if (iteration_count() == 1)
    {
      dispnp->update(1.0, *(structure_field()->dispnp()), 0.0);  // TSI does Dispn()
      velnp->update(1.0, *(structure_field()->velnp()), 0.0);
    }

    // store scalars and displacements for the convergence check later
    scaincnp_->update(1.0, *scatra_field()->phinp(), 0.0);
    dispincnp_->update(1.0, *dispnp, 0.0);

    // begin nonlinear solver / outer iteration ***************************

    // set relaxed mesh displacements and velocity field
    set_struct_solution(*dispnp, velnp, is_s2i_kinetics_with_pseudo_contact());

    // solve scalar transport equation
    do_scatra_step();

    // solve structural system
    do_struct_step();

    // end nonlinear solver / outer iteration *****************************

    // check convergence for all fields and stop iteration loop if
    // convergence is achieved overall
    stopnonliniter = convergence_check(iteration_count());

    // get relaxation parameter
    calc_omega(omega_, iteration_count());

    // do the relaxation
    // d^{i+1} = omega^{i+1} . d^{i+1} + (1- omega^{i+1}) d^i
    //         = d^i + omega^{i+1} * ( d^{i+1} - d^i )
    dispnp->update(omega_, *dispincnp_, 1.0);

    // since the velocity field has to fit to the relaxated displacements we also have to relaxate
    // them.
    {
      // consistent derivation of velocity field from displacement field
      // set_state(dispnp) will automatically be undone during the next evaluation of the structural
      // field
      structure_field()->set_state(dispnp);
      velnp->update(1., *structure_field()->velnp(), 0.);
    }
  }
}

/*----------------------------------------------------------------------*
 | Calculate relaxation parameter                            Thon 12/14 |
 *----------------------------------------------------------------------*/
void SSI::SSIPart2WCSolidToScatraRelax::calc_omega(double& omega, const int itnum)
{
  // nothing to do in here since we have a constant relaxation parameter: omega != startomega_;
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
    std::cout << "Fixed relaxation parameter omega is: " << omega << std::endl;
}

/*----------------------------------------------------------------------*
 | Constructor                                               Thon 12/14 |
 *----------------------------------------------------------------------*/
SSI::SSIPart2WCSolidToScatraRelaxAitken::SSIPart2WCSolidToScatraRelaxAitken(
    MPI_Comm comm, const Teuchos::ParameterList& globaltimeparams)
    : SSIPart2WCSolidToScatraRelax(comm, globaltimeparams)
{
  // Keep this constructor empty!
  // First do everything on the more basic objects like the discretizations, like e.g.
  // redistribution of elements. Only then call the setup to this class. This will call he setup to
  // all classes in the inheritance hierarchy. This way, this class may also override a method that
  // is called during setup() in a base class.
}

/*----------------------------------------------------------------------*
 | Setup this class                                          fang 01/18 |
 *----------------------------------------------------------------------*/
void SSI::SSIPart2WCSolidToScatraRelaxAitken::setup()
{
  // call setup of base class
  SSI::SSIPart2WC::setup();

  // setup old scatra increment vector
  dispincnpold_ =
      std::make_shared<Core::LinAlg::Vector<double>>(*structure_field()->dof_row_map(0), true);
}

/*----------------------------------------------------------------------*
 | Calculate relaxation parameter via Aitken                 Thon 12/14 |
 *----------------------------------------------------------------------*/
void SSI::SSIPart2WCSolidToScatraRelaxAitken::calc_omega(double& omega, const int itnum)
{
  const Teuchos::ParameterList& ssicontrolpart =
      Global::Problem::instance()->ssi_control_params().sublist("PARTITIONED");

  // Get maximal relaxation parameter from input file
  const double maxomega = ssicontrolpart.get<double>("MAXOMEGA");
  // Get minimal relaxation parameter from input file
  const double minomega = ssicontrolpart.get<double>("MINOMEGA");

  // calculate difference of current (i+1) and old (i) residual vector
  // dispincnpdiff = ( r^{i+1}_{n+1} - r^i_{n+1} )
  std::shared_ptr<Core::LinAlg::Vector<double>> dispincnpdiff =
      std::make_shared<Core::LinAlg::Vector<double>>(*(structure_field()->dof_row_map(0)), true);
  dispincnpdiff->update(
      1.0, *dispincnp_, (-1.0), *dispincnpold_, 0.0);  // update r^{i+1}_{n+1} - r^i_{n+1}

  double dispincnpdiffnorm = 0.0;
  dispincnpdiff->norm_2(&dispincnpdiffnorm);
  if (dispincnpdiffnorm <= 1e-06 and Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    std::cout << "Warning: The structure increment is to small in order to use it for Aitken "
                 "relaxation. Using the previous Omega instead!"
              << std::endl;
  }

  // calculate dot product
  double dispincsdot = 0.0;  // delsdot = ( r^{i+1}_{n+1} - r^i_{n+1} )^T . r^{i+1}_{n+1}
  dispincnpdiff->dot(*dispincnp_, &dispincsdot);

  if (itnum != 1 and dispincnpdiffnorm > 1e-06)
  {  // relaxation parameter
    // omega^{i+1} = 1- mu^{i+1} and nu^{i+1} = nu^i + (nu^i -1) . (r^{i+1} - r^i)^T . (-r^{i+1}) /
    // |r^{i+1} - r^{i}|^2 results in
    omega = omega *
            (1 - (dispincsdot) / (dispincnpdiffnorm *
                                     dispincnpdiffnorm));  // compare e.g. PhD thesis U. Kuettler

    // we force omega to be in the range defined in the input file
    if (omega < minomega)
    {
      if (Core::Communication::my_mpi_rank(get_comm()) == 0)
      {
        std::cout << "Warning: The calculation of the relaxation parameter omega via Aitken did "
                     "lead to a value smaller than MINOMEGA!"
                  << std::endl;
      }
      omega = minomega;
    }
    if (omega > maxomega)
    {
      if (Core::Communication::my_mpi_rank(get_comm()) == 0)
      {
        std::cout << "Warning: The calculation of the relaxation parameter omega via Aitken did "
                     "lead to a value bigger than MAXOMEGA!"
                  << std::endl;
      }
      omega = maxomega;
    }
  }

  // else //if itnum==1 nothing is to do here since we want to take the last omega from the previous
  // step
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
    std::cout << "Using Aitken the relaxation parameter omega was estimated to: " << omega
              << std::endl;

  // update history vector old increment r^i_{n+1}
  dispincnpold_->update(1.0, *dispincnp_, 0.0);
}


/*----------------------------------------------------------------------*
 | Constructor                                               Thon 12/14 |
 *----------------------------------------------------------------------*/
SSI::SSIPart2WCScatraToSolidRelax::SSIPart2WCScatraToSolidRelax(
    MPI_Comm comm, const Teuchos::ParameterList& globaltimeparams)
    : SSIPart2WC(comm, globaltimeparams), omega_(-1.0)
{
  // Keep this constructor empty!
  // First do everything on the more basic objects like the discretizations, like e.g.
  // redistribution of elements. Only then call the setup to this class. This will call the setup to
  // all classes in the inheritance hierarchy. This way, this class may also override a method that
  // is called during setup() in a base class.
}

/*----------------------------------------------------------------------*
 | Setup this class                                         rauch 08/16 |
 *----------------------------------------------------------------------*/
void SSI::SSIPart2WCScatraToSolidRelax::init(MPI_Comm comm,
    const Teuchos::ParameterList& globaltimeparams, const Teuchos::ParameterList& scatraparams,
    const Teuchos::ParameterList& structparams, const std::string& struct_disname,
    const std::string& scatra_disname, bool isAle)
{
  // call setup of base class
  SSI::SSIPart2WC::init(
      comm, globaltimeparams, scatraparams, structparams, struct_disname, scatra_disname, isAle);

  const Teuchos::ParameterList& ssicontrolpart =
      Global::Problem::instance()->ssi_control_params().sublist("PARTITIONED");

  // Get start relaxation parameter from input file
  omega_ = ssicontrolpart.get<double>("STARTOMEGA");

  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    std::cout << "\n#########################################################################\n  "
              << std::endl;
    std::cout << "The ScatraToSolid relaxations are not well tested . Keep your eyes open!  "
              << std::endl;
    std::cout << "\n#########################################################################\n  "
              << std::endl;
  }
}

/*----------------------------------------------------------------------*
 | Outer Timeloop for 2WC SSi with relaxed scalar             Thon 12/14 |
 *----------------------------------------------------------------------*/
void SSI::SSIPart2WCScatraToSolidRelax::outer_loop()
{
  reset_iteration_count();
  bool stopnonliniter = false;

  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    std::cout << "\n****************************************\n          OUTER ITERATION "
                 "LOOP\n****************************************\n";
  }

  // this is the relaxed input
  std::shared_ptr<Core::LinAlg::Vector<double>> phinp =
      std::make_shared<Core::LinAlg::Vector<double>>(
          *scatra_field()->discretization()->dof_row_map(0), true);

  while (!stopnonliniter)
  {
    increment_iteration_count();

    if (iteration_count() == 1)
    {
      phinp->update(1.0, *(scatra_field()->phinp()), 0.0);  // TSI does Dispn()
    }

    // store scalars and displacements for the convergence check later
    scaincnp_->update(1.0, *phinp, 0.0);
    dispincnp_->update(1.0, *structure_field()->dispnp(), 0.0);


    // begin nonlinear solver / outer iteration ***************************

    // set relaxed scalars
    set_scatra_solution(phinp);

    // evaluate temperature from function and set to structural discretization
    evaluate_and_set_temperature_field();

    // solve structural system
    do_struct_step();

    // solve scalar transport equation
    do_scatra_step();

    // check convergence for all fields and stop iteration loop if
    // convergence is achieved overall
    stopnonliniter = convergence_check(iteration_count());

    // get relaxation parameter
    calc_omega(omega_, iteration_count());

    // do the relaxation
    // d^{i+1} = omega^{i+1} . d^{i+1} + (1- omega^{i+1}) d^i
    //         = d^i + omega^{i+1} * ( d^{i+1} - d^i )
    phinp->update(omega_, *scaincnp_, 1.0);
  }
}

/*----------------------------------------------------------------------*
 | Calculate relaxation parameter                            Thon 12/14 |
 *----------------------------------------------------------------------*/
void SSI::SSIPart2WCScatraToSolidRelax::calc_omega(double& omega, const int itnum)
{
  // nothing to do in here since we have a constant relaxation parameter: omega != startomega_;
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
    std::cout << "Fixed relaxation parameter omega is: " << omega << std::endl;
}

/*----------------------------------------------------------------------*
 | Constructor                                               Thon 12/14 |
 *----------------------------------------------------------------------*/
SSI::SSIPart2WCScatraToSolidRelaxAitken::SSIPart2WCScatraToSolidRelaxAitken(
    MPI_Comm comm, const Teuchos::ParameterList& globaltimeparams)
    : SSIPart2WCScatraToSolidRelax(comm, globaltimeparams)
{
  // Keep this constructor empty!
  // First do everything on the more basic objects like the discretizations, like e.g.
  // redistribution of elements. Only then call the setup to this class. This will call the setup to
  // all classes in the inheritance hierarchy. This way, this class may also override a method that
  // is called during setup() in a base class.
}

/*----------------------------------------------------------------------*
 | Setup this class                                          fang 01/18 |
 *----------------------------------------------------------------------*/
void SSI::SSIPart2WCScatraToSolidRelaxAitken::setup()
{
  // call setup of base class
  SSI::SSIPart2WC::setup();

  // setup old scatra increment vector
  scaincnpold_ = std::make_shared<Core::LinAlg::Vector<double>>(
      *scatra_field()->discretization()->dof_row_map(), true);
}

/*----------------------------------------------------------------------*
 | Calculate relaxation parameter via Aitken                 Thon 12/14 |
 *----------------------------------------------------------------------*/
void SSI::SSIPart2WCScatraToSolidRelaxAitken::calc_omega(double& omega, const int itnum)
{
  const Teuchos::ParameterList& ssicontrolpart =
      Global::Problem::instance()->ssi_control_params().sublist("PARTITIONED");

  // Get maximal relaxation parameter from input file
  const double maxomega = ssicontrolpart.get<double>("MAXOMEGA");
  // Get minimal relaxation parameter from input file
  const double minomega = ssicontrolpart.get<double>("MINOMEGA");

  // scaincnpdiff =  r^{i+1}_{n+1} - r^i_{n+1}
  std::shared_ptr<Core::LinAlg::Vector<double>> scaincnpdiff =
      std::make_shared<Core::LinAlg::Vector<double>>(
          *scatra_field()->discretization()->dof_row_map(0), true);
  scaincnpdiff->update(1.0, *scaincnp_, (-1.0), *scaincnpold_, 0.0);

  double scaincnpdiffnorm = 0.0;
  scaincnpdiff->norm_2(&scaincnpdiffnorm);

  if (scaincnpdiffnorm <= 1e-06 and Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    std::cout << "Warning: The scalar increment is to small in order to use it for Aitken "
                 "relaxation. Using the previous omega instead!"
              << std::endl;
  }

  // calculate dot product
  double scaincsdot = 0.0;  // delsdot = ( r^{i+1}_{n+1} - r^i_{n+1} )^T . r^{i+1}_{n+1}
  scaincnpdiff->dot(*scaincnp_, &scaincsdot);

  if (itnum != 1 and scaincnpdiffnorm > 1e-06)
  {  // relaxation parameter
    // omega^{i+1} = 1- mu^{i+1} and nu^{i+1} = nu^i + (nu^i -1) . (r^{i+1} - r^i)^T . (-r^{i+1}) /
    // |r^{i+1} - r^{i}|^2 results in
    omega = omega *
            (1 - (scaincsdot) /
                     (scaincnpdiffnorm * scaincnpdiffnorm));  // compare e.g. PhD thesis U. Kuettler

    // we force omega to be in the range defined in the input file
    if (omega < minomega)
    {
      if (Core::Communication::my_mpi_rank(get_comm()) == 0)
      {
        std::cout << "Warning: The calculation of the relaxation parameter omega via Aitken did "
                     "lead to a value smaller than MINOMEGA!"
                  << std::endl;
      }
      omega = minomega;
    }
    if (omega > maxomega)
    {
      if (Core::Communication::my_mpi_rank(get_comm()) == 0)
      {
        std::cout << "Warning: The calculation of the relaxation parameter omega via Aitken did "
                     "lead to a value bigger than MAXOMEGA!"
                  << std::endl;
      }
      omega = maxomega;
    }
  }

  // else //if itnum==1 nothing is to do here since we want to take the last omega from the previous
  // step
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
    std::cout << "Using Aitken the relaxation parameter omega was estimated to: " << omega
              << std::endl;

  // update history vector old increment r^i_{n+1}
  scaincnpold_->update(1.0, *scaincnp_, 0.0);
}

FOUR_C_NAMESPACE_CLOSE

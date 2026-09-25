// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_algorithm.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_coupling_adapter_volmortar.hpp"
#include "4C_fluid_turbulence_statistic_manager.hpp"
#include "4C_global_data.hpp"
#include "4C_scatra_timint_implicit.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
ScaTra::ScaTraAlgorithm::ScaTraAlgorithm(Global::Problem& problem,  ///< global problem
    MPI_Comm comm,                                                  ///< communicator
    const Teuchos::ParameterList& scatradyn,                        ///< scatra parameter list
    const Teuchos::ParameterList& fdyn,                             ///< fluid parameter list
    const std::string scatra_disname,                               ///< scatra discretization name
    const Teuchos::ParameterList& solverparams                      ///< solver parameter list
    )
    : ScaTraFluidCouplingAlgorithm(problem, comm, scatradyn, false, scatra_disname, solverparams),
      natconv_(scatradyn.get<bool>("NATURAL_CONVECTION")),
      natconvitmax_(scatradyn.sublist("NONLINEAR").get<int>("ITEMAX_OUTER")),
      natconvittol_(scatradyn.sublist("NONLINEAR").get<double>("CONVTOL_OUTER")),
      velincnp_(nullptr),
      phiincnp_(nullptr),
      samstart_(fdyn.sublist("TURBULENCE MODEL").get<int>("SAMPLING_START")),
      samstop_(fdyn.sublist("TURBULENCE MODEL").get<int>("SAMPLING_STOP"))
{
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ScaTra::ScaTraAlgorithm::setup()
{
  // call setup in base class
  Adapter::ScaTraFluidCouplingAlgorithm::setup();

  // create vectors
  velincnp_ = std::make_shared<Core::LinAlg::Vector<double>>(
      *(fluid_field()->extract_velocity_part(fluid_field()->velnp())));
  phiincnp_ = std::make_shared<Core::LinAlg::Vector<double>>(*(scatra_field()->phinp()));

  if (velincnp_ == nullptr) FOUR_C_THROW("velincnp_ == nullptr");
  if (phiincnp_ == nullptr) FOUR_C_THROW("phiincnp_ == nullptr");
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ScaTra::ScaTraAlgorithm::init()
{
  // call init in base class
  Adapter::ScaTraFluidCouplingAlgorithm::init();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraAlgorithm::time_loop()
{
  // safety checks
  check_is_init();
  check_is_setup();

  // provide information about initial field
  prepare_time_loop();

  if (!natconv_)
    time_loop_one_way();  // one-way coupling
  else
    time_loop_two_way();  // two-way coupling (natural convection)
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraAlgorithm::time_loop_one_way()
{
  // safety checks
  check_is_init();
  check_is_setup();

  // write velocities since they may be needed in prepare_first_time_step() of the scatra field
  set_velocity_field();
  // write initial output to file
  if (step() == 0) output();

  // time loop (no-subcycling at the moment)
  while (not_finished())
  {
    // prepare next time step
    prepare_time_step();

    // solve nonlinear Navier-Stokes system
    do_fluid_step();

    // solve transport equations
    do_transport_step();

    // update all field solvers
    update();

    // compute error for problems with analytical solution
    scatra_field()->evaluate_error_compared_to_analytical_sol();

    // write output to screen and files
    output();
  }
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void ScaTra::ScaTraAlgorithm::time_loop_two_way()
{
  // safety checks
  check_is_init();
  check_is_setup();

  prepare_time_loop_two_way();

  // time loop
  while (not_finished())
  {
    increment_time_and_step();

    // prepare next time step
    prepare_time_step_convection();

    // Outer iteration loop
    outer_iteration_convection();

    // update all single field solvers and density
    update_convection();

    // write output to screen and files
    output();
  }
}

/*----------------------------------------------------------------------*
 | Initial calculations for two-way coupling time loop       fang 08/14 |
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraAlgorithm::prepare_time_loop_two_way()
{
  // safety checks
  check_is_init();
  check_is_setup();

  // safety check
  switch ((fluid_field()->tim_int_scheme()))
  {
    case FLUID::timeint_afgenalpha:
    case FLUID::timeint_bdf2:
    case FLUID::timeint_one_step_theta:
    case FLUID::timeint_stationary:
      break;
    default:
    {
      FOUR_C_THROW("Selected time integration scheme is not available!");
      break;
    }
  }

  // compute initial mean concentrations and load densification coefficients
  scatra_field()->setup_nat_conv();

  // compute initial density
  scatra_field()->compute_density();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ScaTra::ScaTraAlgorithm::prepare_time_step()
{
  // safety checks
  check_is_init();
  check_is_setup();

  increment_time_and_step();

  fluid_field()->prepare_time_step();

  // prepare time step
  /* remark: initial velocity field has been transferred to scalar transport field in constructor of
   * ScaTraFluidCouplingAlgorithm (initialvelset_ == true). Time integration schemes, such as
   * the one-step-theta scheme, are thus initialized correctly.
   */
  scatra_field()->prepare_time_step();

  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    std::cout << "\n******************\n   TIME STEP     \n******************\n";
    std::cout << "\nStep:   " << step() << " / " << n_step() << "\n";
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ScaTra::ScaTraAlgorithm::prepare_time_step_convection()
{
  // safety checks
  check_is_init();
  check_is_setup();

  // OST/BDF2 time integration schemes are implemented
  // pass density to fluid discretization at time steps n+1 and n
  // density time derivative is not used for OST and BDF2 (pass zero vector)
  // thermodynamic pressure values are set to 1.0 and its derivative to 0.0

  switch ((fluid_field()->tim_int_scheme()))
  {
    case FLUID::timeint_afgenalpha:
    case FLUID::timeint_bdf2:
    case FLUID::timeint_one_step_theta:
    case FLUID::timeint_stationary:
    {
      fluid_field()->set_iter_scalar_fields(scatra_field()->densafnp(),
          scatra_field()->densafnp(),  // not needed, provided as dummy vector
          nullptr, scatra_field()->discretization());
      break;
    }
    default:
    {
      FOUR_C_THROW("Selected time integration scheme is not available!");
    }
  }

  fluid_field()->prepare_time_step();

  // transfer the initial(!!) convective velocity
  //(fluid initial field was set inside the constructor of fluid base class)
  if (step() == 1)
  {
    scatra_field()->set_acceleration_field(*fluid_field()->hist());
    scatra_field()->set_convective_velocity(*fluid_field()->velnp());
    scatra_field()->set_velocity_field(*fluid_field()->velnp());
  }

  // prepare time step (+ initialize one-step-theta scheme correctly with
  // velocity given above)
  scatra_field()->prepare_time_step();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraAlgorithm::print_scatra_solver()
{
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
    std::cout
        << "\n****************************\n      TRANSPORT SOLVER\n****************************\n";
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ScaTra::ScaTraAlgorithm::do_fluid_step()
{
  // solve nonlinear Navier-Stokes system
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
    std::cout << "\n************************\n      FLUID SOLVER\n************************\n";

  // currently only required for forced homogeneous isotropic turbulence with
  // passive scalar transport; does nothing otherwise
  fluid_field()->calc_intermediate_solution();

  fluid_field()->solve();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ScaTra::ScaTraAlgorithm::do_transport_step()
{
  print_scatra_solver();

  // transfer velocities to scalar transport field solver
  set_velocity_field();

  // solve the transport equation(s)
  scatra_field()->solve();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ScaTra::ScaTraAlgorithm::set_velocity_field()
{
  // safety checks
  check_is_init();
  check_is_setup();

  // transfer velocities to scalar transport field solver
  // NOTE: so far, the convective velocity is chosen to equal the fluid velocity
  //       since it is not yet clear how the grid velocity should be interpolated
  //       properly -> hence, ScaTraAlgorithm does not support moving
  //       meshes yet
  switch (fluid_field()->tim_int_scheme())
  {
    case FLUID::timeint_npgenalpha:
    case FLUID::timeint_afgenalpha:
    {
      scatra_field()->set_acceleration_field(*fluid_to_scatra(fluid_field()->accam()));
      scatra_field()->set_convective_velocity(*fluid_to_scatra(fluid_field()->velaf()));
      scatra_field()->set_velocity_field(*fluid_to_scatra(fluid_field()->velaf()));
      if (scatra_field()->fine_scale_velocity_field_required() and
          fluid_field()->fs_vel() != nullptr)
      {
        scatra_field()->set_fine_scale_velocity(*fluid_to_scatra(fluid_field()->fs_vel()));
      }
      break;
    }
    case FLUID::timeint_one_step_theta:
    case FLUID::timeint_bdf2:
    case FLUID::timeint_stationary:
    {
      scatra_field()->set_acceleration_field(*fluid_to_scatra(fluid_field()->hist()));
      scatra_field()->set_convective_velocity(*fluid_to_scatra(fluid_field()->velnp()));
      scatra_field()->set_velocity_field(*fluid_to_scatra(fluid_field()->velnp()));
      if (scatra_field()->fine_scale_velocity_field_required() and
          fluid_field()->fs_vel() != nullptr)
      {
        scatra_field()->set_fine_scale_velocity(*fluid_to_scatra(fluid_field()->fs_vel()));
      }
      break;
    }
    default:
    {
      FOUR_C_THROW("Time integration scheme not supported");
    }
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ScaTra::ScaTraAlgorithm::outer_iteration_convection()
{
  // safety checks
  check_is_init();
  check_is_setup();

  int natconvitnum = 0;
  bool stopnonliniter = false;

  // Outer Iteration loop starts
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    std::cout << "\n";
    std::cout << "**************************************************************\n";
    std::cout << "      OUTER ITERATION LOOP (" << scatra_field()->method_title() << ")\n";
    printf("      Time Step %3d/%3d \n", step(), scatra_field()->n_step());
    std::cout << "**************************************************************\n";
  }

  while (!stopnonliniter)
  {
    natconvitnum++;

    phiincnp_->update(1.0, *scatra_field()->phinp(), 0.0);
    velincnp_->update(1.0, *fluid_field()->extract_velocity_part(fluid_field()->velnp()), 0.0);

    // solve nonlinear Navier-Stokes system with body forces
    do_fluid_step();

    // solve scalar transport equation
    do_transport_step();

    // compute new density field and pass it to the fluid discretization
    scatra_field()->compute_density();
    fluid_field()->set_scalar_fields(
        scatra_field()->densafnp(), 0.0, nullptr, scatra_field()->discretization());

    // convergence check based on incremental values
    stopnonliniter = convergence_check(natconvitnum, natconvitmax_, natconvittol_);
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ScaTra::ScaTraAlgorithm::update()
{
  fluid_field()->update();
  scatra_field()->update();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ScaTra::ScaTraAlgorithm::update_convection()
{
  // update scatra and fluid fields
  scatra_field()->update();
  fluid_field()->update();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ScaTra::ScaTraAlgorithm::output()
{
  // Note: The order is important here! In here control file entries are
  // written. And these entries define the order in which the filters handle
  // the discretizations, which in turn defines the dof number ordering of the
  // discretizations.

  if ((step() >= samstart_) and (step() <= samstop_))
  {
    // if statistics for one-way coupled problems is performed, provide
    // the field for the first scalar!
    fluid_field()->set_scalar_fields(
        scatra_field()->phinp(), 0.0, nullptr, scatra_field()->discretization(),
        0  // do statistics for FIRST dof at every node!!
    );
  }

  fluid_field()->statistics_and_output();
  scatra_field()->check_and_write_output_and_restart();

  // we have to call the output of averaged fields for scatra separately
  if (fluid_field()->turbulence_statistic_manager() != nullptr)
    fluid_field()->turbulence_statistic_manager()->do_output_for_scatra(
        *scatra_field()->disc_writer(), scatra_field()->step());
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool ScaTra::ScaTraAlgorithm::convergence_check(
    int natconvitnum, const int natconvitmax, const double natconvittol)
{
  // convergence check based on phi and velocity increment

  //   | phi increment |_2
  //  -------------------------- < Tolerance
  //     | phi_n+1 |_2

  bool stopnonliniter = false;
  std::shared_ptr<Core::LinAlg::MapExtractor> phisplitter = scatra_field()->splitter();
  // Variables to save different L2 - Norms

  double velincnorm_L2(0.0);
  double velnorm_L2(0.0);
  double phiincnorm_L2(0.0);
  double phinorm_L2(0.0);

  // Calculate velocity increment and velocity L2 - Norm
  // velincnp_ = 1.0 * convelnp_ - 1.0 * conveln_

  velincnp_->update(1.0, *fluid_field()->extract_velocity_part(fluid_field()->velnp()), -1.0);
  velincnp_->norm_2(&velincnorm_L2);  // Estimation of the L2 - norm save values to both variables
                                      // (velincnorm_L2 and velnorm_L2)
  fluid_field()->extract_velocity_part(fluid_field()->velnp())->norm_2(&velnorm_L2);

  // Calculate phi increment and phi L2 - Norm
  // tempincnp_ includes the concentration and the potential increment
  // tempincnp_ = 1.0 * phinp_ - 1.0 * phin_

  phiincnp_->update(1.0, *scatra_field()->phinp(), -1.0);
  phiincnp_->norm_2(&phiincnorm_L2);
  scatra_field()->phinp()->norm_2(&phinorm_L2);

  // care for the case that there is (almost) zero temperature or velocity
  // (usually not required for temperature)
  if (velnorm_L2 < 1e-6) velnorm_L2 = 1.0;
  if (phinorm_L2 < 1e-6) phinorm_L2 = 1.0;

  // Print the incremental based convergence check to the screen
  if (natconvitnum != 1)
  {
    if (Core::Communication::my_mpi_rank(get_comm()) == 0)
    {
      std::cout << "\n";
      std::cout
          << "*****************************************************************************\n";
      std::cout << "                          OUTER ITERATION STEP\n";
      std::cout
          << "*****************************************************************************\n";
      printf("+------------+-------------------+-------------+-------------+\n");
      printf("|- step/max -|- tol      [norm] -|-- con-inc --|-- vel-inc --|\n");
      printf("|  %3d/%3d   | %10.3E[L_2 ]  | %10.3E  | %10.3E  |", natconvitnum, natconvitmax,
          natconvittol, phiincnorm_L2 / phinorm_L2, velincnorm_L2 / velnorm_L2);
      printf("\n");
      printf("+------------+-------------------+-------------+-------------+\n");
    }

    // Converged or Not
    if ((phiincnorm_L2 / phinorm_L2 <= natconvittol) &&
        (velincnorm_L2 / velnorm_L2 <= natconvittol))
    // if ((incconnorm_L2/connorm_L2 <= natconvittol))
    {
      stopnonliniter = true;
      if (Core::Communication::my_mpi_rank(get_comm()) == 0)
      {
        printf("| Outer Iteration loop converged after iteration %3d/%3d                    |\n",
            natconvitnum, natconvitmax);
        printf("+---------------------------------------------------------------------------+");
        printf("\n");
        printf("\n");
      }
    }
    else
    {
      if (Core::Communication::my_mpi_rank(get_comm()) == 0)
      {
        printf("| Outer Iteration loop is not converged after iteration %3d/%3d             |\n",
            natconvitnum, natconvitmax);
        printf("+---------------------------------------------------------------------------+");
        printf("\n");
        printf("\n");
      }
    }
  }
  else
  {
    // first outer iteration loop: fluid solver has not got the new density yet
    // => minimum two outer iteration loops
    stopnonliniter = false;
    if (Core::Communication::my_mpi_rank(get_comm()) == 0)
    {
      std::cout << "\n";
      std::cout
          << "*****************************************************************************\n";
      std::cout << "                          OUTER ITERATION STEP\n";
      std::cout
          << "*****************************************************************************\n";
      printf("+------------+-------------------+-------------+-------------+\n");
      printf("|- step/max -|- tol      [norm] -|-- con-inc --|-- vel-inc --|\n");
      printf("|  %3d/%3d   | %10.3E[L_2 ]  |       -     |      -      |", natconvitnum,
          natconvitmax, natconvittol);
      printf("\n");
      printf("+------------+-------------------+-------------+-------------+\n");
    }
  }

  // warn if natconvitemax is reached without convergence, but proceed to next timestep
  // natconvitemax = 1 is also possible for segregated coupling approaches (not fully implicit)
  if (natconvitnum == natconvitmax)
  {
    if (((phiincnorm_L2 / phinorm_L2 > natconvittol) ||
            (velincnorm_L2 / velnorm_L2 > natconvittol)) or
        (natconvitmax == 1))
    {
      stopnonliniter = true;
      if ((Core::Communication::my_mpi_rank(get_comm()) == 0))
      {
        printf("|     >>>>>> not converged in itemax steps!     |\n");
        printf("+-----------------------------------------------+\n");
        printf("\n");
        printf("\n");
      }
    }
  }

  return stopnonliniter;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ScaTra::ScaTraAlgorithm::read_inflow_restart(int restart)
{
  // in case a inflow generation in the inflow section has been performed,
  // there are not any scatra results available and the initial field is used
  fluid_field()->read_restart(restart);
  // as read_restart is only called for the fluid_field
  // time and step have not been set in the superior class and the ScaTraField
  set_time_step(fluid_field()->time(), fluid_field()->step());
  scatra_field()->set_time_step(fluid_field()->time(), fluid_field()->step());
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ScaTra::ScaTraAlgorithm::test_results()
{
  auto& problem = AlgorithmBase::problem();
  problem.add_field_test(fluid_field()->create_field_test());
  problem.add_field_test(create_scatra_field_test());
  problem.test_all(get_comm());
}
FOUR_C_NAMESPACE_CLOSE

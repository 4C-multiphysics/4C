// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_porofluid_pressure_based_elast_partitioned.hpp"

#include "4C_adapter_art_net.hpp"
#include "4C_adapter_porofluid_pressure_based_wrapper.hpp"
#include "4C_adapter_str_wrapper.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | constructor                                              vuong 08/16 |
 *----------------------------------------------------------------------*/
PoroPressureBased::PorofluidElastPartitionedAlgorithm::PorofluidElastPartitionedAlgorithm(
    MPI_Comm comm, const Teuchos::ParameterList& globaltimeparams)
    : PorofluidElastAlgorithm(comm, globaltimeparams),
      phiincnp_(nullptr),
      dispincnp_(nullptr),
      fluidphinp_(nullptr),
      fluidphioldnp_(nullptr),
      fluidphiincnp_(nullptr),
      ittol_(0.0),
      omega_(1.0),
      startomega_(1.0),
      omegamin_(1.0),
      omegamax_(1.0),
      itmax_(0),
      itnum_(0),
      writerestartevery_(-1),
      artery_coupling_active_(false),
      relaxationmethod_(PoroPressureBased::RelaxationMethods::none)
{
}

/*----------------------------------------------------------------------*
 | initialization                                            vuong 08/16 |
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastPartitionedAlgorithm::init(
    const Teuchos::ParameterList& globaltimeparams, const Teuchos::ParameterList& algoparams,
    const Teuchos::ParameterList& structparams, const Teuchos::ParameterList& fluidparams,
    const std::string& struct_disname, const std::string& fluid_disname, bool isale, int nds_disp,
    int nds_vel, int nds_solidpressure, int ndsporofluid_scatra,
    const std::map<int, std::set<int>>* nearby_ele_pairs)
{
  // call base class
  PorofluidElastAlgorithm::init(globaltimeparams, algoparams, structparams, fluidparams,
      struct_disname, fluid_disname, isale, nds_disp, nds_vel, nds_solidpressure,
      ndsporofluid_scatra, nearby_ele_pairs);

  artery_coupling_active_ = fluidparams.get<bool>("artery_coupling_active");

  // initialize increment vectors
  phiincnp_ = Core::LinAlg::create_vector(*porofluid_algo()->dof_row_map(0), true);
  if (artery_coupling_active_)
    arterypressincnp_ = Core::LinAlg::create_vector(*porofluid_algo()->artery_dof_row_map(), true);
  dispincnp_ = Core::LinAlg::create_vector(*structure_algo()->dof_row_map(0), true);

  // initialize fluid vectors
  fluidphinp_ =
      Core::LinAlg::create_vector(*porofluid_algo()->discretization()->dof_row_map(), true);
  fluidphioldnp_ =
      Core::LinAlg::create_vector(*porofluid_algo()->discretization()->dof_row_map(), true);
  fluidphiincnp_ =
      Core::LinAlg::create_vector(*porofluid_algo()->discretization()->dof_row_map(), true);
  fluidphiincnpold_ =
      Core::LinAlg::create_vector(*porofluid_algo()->discretization()->dof_row_map(), true);

  // Get the parameters for the convergence_check
  itmax_ = algoparams.sublist("nonlinear_solver").get<int>("maximum_number_of_iterations");
  ittol_ = algoparams.sublist("partitioned").get<double>("convergence_tolerance");

  // restart
  writerestartevery_ = globaltimeparams.sublist("output").get<int>("restart_data_every");

  // relaxation parameters
  startomega_ = algoparams.sublist("partitioned").sublist("relaxation").get<double>("start_omega");
  omegamin_ = algoparams.sublist("partitioned").sublist("relaxation").get<double>("minimum_omega");
  omegamax_ = algoparams.sublist("partitioned").sublist("relaxation").get<double>("maximum_omega");

  relaxationmethod_ = Teuchos::getIntegralValue<PoroPressureBased::RelaxationMethods>(
      algoparams.sublist("partitioned").sublist("relaxation"), "type");
}

/*----------------------------------------------------------------------*
 | setup the system if necessary                             vuong 08/16 |
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastPartitionedAlgorithm::setup_system()
{
  // Do nothing, just monolithic coupling needs this method
  return;
}

/*----------------------------------------------------------------------*
 | Outer Timeloop  without relaxation                        vuong 08/16 |
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastPartitionedAlgorithm::outer_loop()
{
  // reset counter
  itnum_ = 0;
  bool stopnonliniter = false;

  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    std::cout << "********************************************************************************"
              << "***********************************************\n";
    std::cout << "* PARTITIONED OUTER ITERATION LOOP ----- FLUID  <-------> STRUCTURE         "
              << "                                                  *\n";
    std::cout << "* STEP: " << std::setw(5) << std::setprecision(4) << std::scientific << step()
              << "/" << std::setw(5) << std::setprecision(4) << std::scientific << n_step()
              << ", Time: " << std::setw(11) << std::setprecision(4) << std::scientific << time()
              << "/" << std::setw(11) << std::setprecision(4) << std::scientific << max_time()
              << ", Dt: " << std::setw(11) << std::setprecision(4) << std::scientific << dt()
              << "                                                           *" << std::endl;
  }

  while (stopnonliniter == false)
  {
    // increment number of iteration
    itnum_++;

    // update the states to the last solutions obtained
    iter_update_states();

    if (solve_structure_)
    {
      // 1.) solve structural system, note: in first iteration solid pressure has already been set
      // in prepare_time_step()
      do_struct_step();
      // 2.) set disp and vel states in porofluid field
      set_structure_solution(structure_algo()->dispnp(), structure_algo()->velnp());
    }
    else
    {
      // Inform user that structure field has been disabled
      print_structure_disabled_info();
      // just set displacements and velocities to zero
      set_structure_solution(Core::LinAlg::create_vector(*structure_algo()->dof_row_map(), true),
          Core::LinAlg::create_vector(*structure_algo()->dof_row_map(), true));
    }

    // 1.) solve scalar transport equation
    do_fluid_step();

    // perform relaxation
    perform_relaxation(porofluid_algo()->phinp(), itnum_);

    // 2.) set fluid solution in structure field
    set_relaxed_fluid_solution();

    // check convergence for all fields
    // stop iteration loop if converged
    stopnonliniter = convergence_check(itnum_);
  }

  return;
}

/*----------------------------------------------------------------------*
 | convergence check for both fields (copied form tsi)       vuong 08/16 |
 *----------------------------------------------------------------------*/
bool PoroPressureBased::PorofluidElastPartitionedAlgorithm::convergence_check(int itnum)
{
  // convergence check based on the scalar increment
  bool stopnonliniter = false;

  //    | fluid increment |_2
  //  -------------------------------- < Tolerance
  //    | fluid state n+1 |_2
  //
  // The same is checked for the structural displacements
  // and 1D artery pressure (if present)

  // variables to save different L2 - Norms
  // define L2-norm of increments
  double phiincnorm_L2(0.0);
  double phinorm_L2(0.0);
  double dispincnorm_L2(0.0);
  double dispnorm_L2(0.0);
  double artpressincnorm_L2(0.0);
  double artpressnorm_L2(0.0);

  // build the current scalar increment Inc T^{i+1}
  // \f Delta T^{k+1} = Inc T^{k+1} = T^{k+1} - T^{k}  \f
  phiincnp_->update(1.0, *(porofluid_algo()->phinp()), -1.0);
  if (artery_coupling_active_)
    arterypressincnp_->update(1.0, *porofluid_algo()->art_net_tim_int()->pressurenp(), -1.0);
  dispincnp_->update(1.0, *(structure_algo()->dispnp()), -1.0);

  // build the L2-norm of the scalar increment and the scalar
  phiincnp_->norm_2(&phiincnorm_L2);
  porofluid_algo()->phinp()->norm_2(&phinorm_L2);
  dispincnp_->norm_2(&dispincnorm_L2);
  structure_algo()->dispnp()->norm_2(&dispnorm_L2);
  if (artery_coupling_active_)
  {
    arterypressincnp_->norm_2(&artpressincnorm_L2);
    porofluid_algo()->art_net_tim_int()->pressurenp()->norm_2(&artpressnorm_L2);
  }

  // care for the case that there is (almost) zero scalar
  if (phinorm_L2 < 1e-6) phinorm_L2 = 1.0;
  if (dispnorm_L2 < 1e-6) dispnorm_L2 = 1.0;
  if (artpressnorm_L2 < 1e-6) artpressnorm_L2 = 1.0;

  // print the incremental based convergence check to the screen
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    std::cout << "                                                                                 "
                 "                                             *\n";
    std::cout << "+--------------------------------------------------------------------------------"
                 "--------------------------------+            *\n";
    std::cout << "| PARTITIONED OUTER ITERATION STEP ----- FLUID  <-------> STRUCTURE              "
                 "                                |            *\n";
    printf(
        "+--------------+---------------------+-----------------+-----------------+----------------"
        "-+---------------------+            *\n");
    printf(
        "|-  step/max  -|-  tol      [norm]  -|- fluid-rel-inc -|-  disp-rel-inc -|-   1D-rel-inc  "
        "-|                                  *\n");
    printf("|   %3d/%3d    |  %10.3E[L_2 ]   |   %10.3E    |   %10.3E    |   %10.3E    |", itnum,
        itmax_, ittol_, phiincnorm_L2 / phinorm_L2, dispincnorm_L2 / dispnorm_L2,
        artpressincnorm_L2 / artpressnorm_L2);
    printf("                                  *\n");
    printf(
        "+--------------+---------------------+-----------------+-----------------+----------------"
        "-+                                  *\n");
  }

  // converged
  if (((phiincnorm_L2 / phinorm_L2) <= ittol_) and ((dispincnorm_L2 / dispnorm_L2) <= ittol_) and
      ((artpressincnorm_L2 / artpressnorm_L2) <= ittol_))
  {
    stopnonliniter = true;
    if (Core::Communication::my_mpi_rank(get_comm()) == 0)
    {
      printf(
          "* FLUID  <-------> STRUCTURE Outer Iteration loop converged after iteration %3d/%3d !   "
          "                                      *\n",
          itnum, itmax_);
      printf(
          "****************************************************************************************"
          "***************************************\n");
    }
  }

  // stop if itemax is reached without convergence
  // timestep
  if ((itnum == itmax_) and
      (((phiincnorm_L2 / phinorm_L2) > ittol_) or ((dispincnorm_L2 / dispnorm_L2) > ittol_) or
          ((artpressincnorm_L2 / artpressnorm_L2) > ittol_)))
  {
    stopnonliniter = true;
    if ((Core::Communication::my_mpi_rank(get_comm()) == 0))
    {
      printf(
          "|     >>>>>> not converged in itemax steps!                                             "
          "                         |\n");
      printf(
          "+--------------+---------------------+--------------------+----------------+--------"
          "------------+----------------+            *\n");
      printf("\n");
      printf("\n");
    }
    FOUR_C_THROW("The partitioned solver did not converge in ITEMAX steps!");
  }

  return stopnonliniter;
}

/*----------------------------------------------------------------------*
 | Solve structure filed                                     vuong 08/16 |
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastPartitionedAlgorithm::do_struct_step()
{
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    std::cout << "\n";
    std::cout << "*********************************************************************************"
                 "********************************\n";
    std::cout << "STRUCTURE SOLVER   \n";
    std::cout << "*********************************************************************************"
                 "********************************\n";
  }

  // Newton-Raphson iteration
  structure_algo()->solve();

  return;
}

/*----------------------------------------------------------------------*
 | Solve poro fluid field                                    vuong 08/16 |
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastPartitionedAlgorithm::do_fluid_step()
{
  // -------------------------------------------------------------------
  //                  solve nonlinear / linear equation
  // -------------------------------------------------------------------
  porofluid_algo()->solve();

  return;
}

/*----------------------------------------------------------------------*
 | Set relaxed fluid solution on structure             kremheller 09/16 |
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastPartitionedAlgorithm::set_relaxed_fluid_solution()
{
  // set fluid solution on structure
  structure_algo()->discretization()->set_state(1, "porofluid", *fluidphinp_);

  return;
}

/*----------------------------------------------------------------------*
 | Calculate relaxation parameter omega                kremheller 09/16 |
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastPartitionedAlgorithm::perform_relaxation(
    std::shared_ptr<const Core::LinAlg::Vector<double>> phi, const int itnum)
{
  // get the increment vector
  fluidphiincnp_->update(1.0, *phi, -1.0, *fluidphioldnp_, 0.0);

  // perform relaxation
  switch (relaxationmethod_)
  {
    case PoroPressureBased::RelaxationMethods::none:
    {
      // no relaxation
      omega_ = 1.0;
      break;
    }

    case PoroPressureBased::RelaxationMethods::constant:
    {
      // constant relaxation parameter omega
      omega_ = startomega_;
      if (Core::Communication::my_mpi_rank(get_comm()) == 0)
        std::cout << "Fixed relaxation parameter omega is: " << omega_ << std::endl;
      break;
    }

    case PoroPressureBased::RelaxationMethods::aitken:
    {
      // Aitken
      aitken_relaxation(omega_, itnum);
      if (Core::Communication::my_mpi_rank(get_comm()) == 0)
        std::cout << "Aitken relaxation parameter omega is: " << omega_ << std::endl;
      break;
    }

    default:
    {
      FOUR_C_THROW("Relaxation method not yet implemented!");
      break;
    }
  }

  // calculate the relaxed fluid solution as phi,n+1^i+1 = phi,n+1^i + \omega*(phi,n+1^i+1 -
  // phi,n+1^i) note: in first iteration step, omega = 1.0
  fluidphinp_->update(1.0, *fluidphioldnp_, omega_, *fluidphiincnp_, 0.0);

  // save the old fluid solution
  fluidphioldnp_->update(1.0, *fluidphinp_, 0.0);

  return;
}

/*----------------------------------------------------------------------*
 | Perform Aitken relaxation                           kremheller 09/16 |
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastPartitionedAlgorithm::aitken_relaxation(
    double& omega, const int itnum)
{
  // fluidphiincnpdiff =  r^{i+1}_{n+1} - r^i_{n+1}
  std::shared_ptr<Core::LinAlg::Vector<double>> fluidphiincnpdiff =
      Core::LinAlg::create_vector(*porofluid_algo()->discretization()->dof_row_map(), true);
  fluidphiincnpdiff->update(1.0, *fluidphiincnp_, (-1.0), *fluidphiincnpold_, 0.0);

  double fluidphiincnpdiffnorm = 0.0;
  fluidphiincnpdiff->norm_2(&fluidphiincnpdiffnorm);

  if (fluidphiincnpdiffnorm <= 1e-06 and Core::Communication::my_mpi_rank(get_comm()) == 0)
    std::cout << "Warning: The scalar increment is too small in order to use it for Aitken "
                 "relaxation. Using the previous omega instead!"
              << std::endl;

  // calculate dot product
  double fluidphiincsdot = 0.0;  // delsdot = ( r^{i+1}_{n+1} - r^i_{n+1} )^T . r^{i+1}_{n+1}
  fluidphiincnpdiff->dot(*fluidphiincnp_, &fluidphiincsdot);

  if (itnum != 1 and fluidphiincnpdiffnorm > 1e-06)
  {
    // relaxation parameter
    // omega^{i+1} = 1- mu^{i+1} and nu^{i+1} = nu^i + (nu^i -1) . (r^{i+1} - r^i)^T . (-r^{i+1}) /
    // |r^{i+1} - r^{i}|^2 results in
    omega = omega * (1.0 - (fluidphiincsdot) /
                               (fluidphiincnpdiffnorm *
                                   fluidphiincnpdiffnorm));  // compare e.g. PhD thesis U. Kuettler

    // we force omega to be in the range defined in the input file
    if (omega < omegamin_)
    {
      if (Core::Communication::my_mpi_rank(get_comm()) == 0)
        std::cout << "Warning: The calculation of the relaxation parameter omega via Aitken did "
                     "lead to a value smaller than MINOMEGA!"
                  << std::endl;
      omega = omegamin_;
    }
    if (omega > omegamax_)
    {
      if (Core::Communication::my_mpi_rank(get_comm()) == 0)
        std::cout << "Warning: The calculation of the relaxation parameter omega via Aitken did "
                     "lead to a value bigger than MAXOMEGA!"
                  << std::endl;
      omega = omegamax_;
    }
  }

  // update history vector old increment r^i_{n+1}
  fluidphiincnpold_->update(1.0, *fluidphiincnp_, 0.0);
}

/*----------------------------------------------------------------------*
 | update the current states in every iteration             vuong 08/16 |
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastPartitionedAlgorithm::iter_update_states()
{
  // store last solutions (current states).
  // will be compared in convergence_check to the solutions,
  // obtained from the next Struct and Scatra steps.
  phiincnp_->update(1.0, *porofluid_algo()->phinp(), 0.0);
  if (artery_coupling_active_)
    arterypressincnp_->update(1.0, *porofluid_algo()->art_net_tim_int()->pressurenp(), 0.0);
  dispincnp_->update(1.0, *structure_algo()->dispnp(), 0.0);

  return;
}  // iter_update_states()

/*-------------------------------------------------------------------------*
 | read restart information for given time step (public) kremheller 09/17  |
 *-------------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastPartitionedAlgorithm::read_restart(int restart)
{
  if (restart)
  {
    // call base class
    PoroPressureBased::PorofluidElastAlgorithm::read_restart(restart);

    Core::IO::DiscretizationReader reader(porofluid_algo()->discretization(),
        Global::Problem::instance()->input_control_file(), restart);
    if (restart != reader.read_int("step"))
      FOUR_C_THROW("Time step on file not equal to given step");

    // get omega_ from restart
    omega_ = reader.read_double("omega_");

    // get omega_ from restart
    reader.read_vector(fluidphioldnp_, "fluidphioldnp_");
  }


  return;
}

/*----------------------------------------------------------------------*
 | update fields and output results                    kremheller 09/17 |
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastPartitionedAlgorithm::update_and_output()
{
  // call base class
  PoroPressureBased::PorofluidElastAlgorithm::update_and_output();

  // write interface force and relaxation parameter in restart
  if (writerestartevery_ and step() % writerestartevery_ == 0)
  {
    porofluid_algo()->discretization()->writer()->write_double("omega_", omega_);
    porofluid_algo()->discretization()->writer()->write_vector("fluidphioldnp_", fluidphioldnp_);
  }
}

FOUR_C_NAMESPACE_CLOSE

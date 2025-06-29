// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_porofluid_pressure_based_elast_scatra_partitioned.hpp"

#include "4C_adapter_art_net.hpp"
#include "4C_adapter_porofluid_pressure_based_wrapper.hpp"
#include "4C_adapter_scatra_base_algorithm.hpp"
#include "4C_adapter_str_wrapper.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_porofluid_pressure_based_elast_base.hpp"
#include "4C_scatra_timint_implicit.hpp"
#include "4C_scatra_timint_meshtying_strategy_artery.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
PoroPressureBased::PorofluidElastScatraPartitionedAlgorithm::
    PorofluidElastScatraPartitionedAlgorithm(
        MPI_Comm comm, const Teuchos::ParameterList& globaltimeparams)
    : PorofluidElastScatraBaseAlgorithm(comm, globaltimeparams),
      scatra_inc_np_(nullptr),
      structure_inc_np_(nullptr),
      porofluid_inc_np_(nullptr),
      iter_max_(-1),
      iter_tol_(-1),
      artery_coupling_active_(false)
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraPartitionedAlgorithm::init(
    const Teuchos::ParameterList& globaltimeparams, const Teuchos::ParameterList& algoparams,
    const Teuchos::ParameterList& poroparams, const Teuchos::ParameterList& structparams,
    const Teuchos::ParameterList& fluidparams, const Teuchos::ParameterList& scatraparams,
    const std::string& struct_disname, const std::string& fluid_disname,
    const std::string& scatra_disname, bool isale, int nds_disp, int nds_vel, int nds_solidpressure,
    int ndsporofluid_scatra, const std::map<int, std::set<int>>* nearby_ele_pairs)
{
  // call base class
  PoroPressureBased::PorofluidElastScatraBaseAlgorithm::init(globaltimeparams, algoparams,
      poroparams, structparams, fluidparams, scatraparams, struct_disname, fluid_disname,
      scatra_disname, isale, nds_disp, nds_vel, nds_solidpressure, ndsporofluid_scatra,
      nearby_ele_pairs);

  // read input variables
  iter_max_ = algoparams.sublist("nonlinear_solver").get<int>("maximum_number_of_iterations");
  iter_tol_ = algoparams.sublist("partitioned").get<double>("convergence_tolerance");

  artery_coupling_active_ = algoparams.get<bool>("artery_coupling_active");

  // initialize increment vectors
  scatra_inc_np_ = std::make_shared<Core::LinAlg::Vector<double>>(
      *(scatra_algo()->scatra_field()->discretization()->dof_row_map()));
  structure_inc_np_ = std::make_shared<Core::LinAlg::Vector<double>>(
      *(porofluid_elast_algo()->structure_dof_row_map()));
  porofluid_inc_np_ = (std::make_shared<Core::LinAlg::Vector<double>>(
      *(porofluid_elast_algo()->porofluid_dof_row_map())));
  if (artery_coupling_active_)
  {
    artery_pressure_inc_np_ = std::make_shared<Core::LinAlg::Vector<double>>(
        *(porofluid_elast_algo()->porofluid_algo()->artery_dof_row_map()));
    artery_scatra_inc_np_ = std::make_shared<Core::LinAlg::Vector<double>>(
        *(scatra_meshtying_strategy_->art_scatra_dof_row_map()));
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraPartitionedAlgorithm::setup_system()
{
  porofluid_elast_algo()->setup_system();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraPartitionedAlgorithm::setup_solver()
{
  porofluid_elast_algo()->setup_solver();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraPartitionedAlgorithm::do_poro_step()
{
  // Newton-Raphson iteration
  porofluid_elast_algo()->time_step();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraPartitionedAlgorithm::do_scatra_step()
{
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    std::cout << "\n";
    std::cout << "*********************************************************************************"
                 "********************************\n";
    std::cout << "TRANSPORT SOLVER   \n";
    std::cout << "*********************************************************************************"
                 "********************************\n";
  }

  // -------------------------------------------------------------------
  //                  solve nonlinear / linear equation
  // -------------------------------------------------------------------
  scatra_algo()->scatra_field()->solve();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraPartitionedAlgorithm::print_header_partitioned()
{
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    std::cout << "\n";
    std::cout << "********************************************************************************"
              << "***************************************************************\n";
    std::cout << "* PARTITIONED OUTER ITERATION LOOP ----- MULTIPORO  <-------> SCATRA         "
              << "                                                                 *\n";
    std::cout << "* STEP: " << std::setw(5) << std::setprecision(4) << std::scientific << step()
              << "/" << std::setw(5) << std::setprecision(4) << std::scientific << n_step()
              << ", Time: " << std::setw(11) << std::setprecision(4) << std::scientific << time()
              << "/" << std::setw(11) << std::setprecision(4) << std::scientific << max_time()
              << ", Dt: " << std::setw(11) << std::setprecision(4) << std::scientific << dt()
              << "                                                                           *"
              << std::endl;
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraPartitionedAlgorithm::iter_update_states()
{
  // store scalar from first solution for convergence check (like in
  // elch_algorithm: use current values)
  scatra_inc_np_->update(1.0, *scatra_algo()->scatra_field()->phinp(), 0.0);
  structure_inc_np_->update(1.0, *porofluid_elast_algo()->structure_dispnp(), 0.0);
  porofluid_inc_np_->update(1.0, *porofluid_elast_algo()->fluid_phinp(), 0.0);
  if (artery_coupling_active_)
  {
    artery_pressure_inc_np_->update(
        1.0, *(porofluid_elast_algo()->porofluid_algo()->art_net_tim_int()->pressurenp()), 0.0);
    artery_scatra_inc_np_->update(
        1.0, *(scatra_meshtying_strategy_->art_scatra_field()->phinp()), 0.0);
  }

}  // iter_update_states()

/*----------------------------------------------------------------------*
 | convergence check for both fields (scatra & poro) (copied from tsi)
 *----------------------------------------------------------------------*/
bool PoroPressureBased::PorofluidElastScatraPartitionedAlgorithm::convergence_check(int itnum)
{
  // convergence check based on the scalar increment
  bool stop_nonlinear_iter = false;

  //    | scalar increment |_2
  //  -------------------------------- < Tolerance
  //     | scalar+1 |_2

  // variables to save different L2 - Norms
  // define L2-norm of incremental scalar and scalar
  double scatra_inc_norm(0.0);
  double scatra_norm(0.0);
  double structure_inc_norm(0.0);
  double structure_norm(0.0);
  double porofluid_inc_norm(0.0);
  double porofluid_norm(0.0);
  double artery_pressure_inc_norm(0.0);
  double artery_pressure_norm(0.0);
  double artery_scatra_inc_norm(0.0);
  double artery_scatra_norm(0.0);

  // build the current scalar increment Inc T^{i+1}
  // \f Delta T^{k+1} = Inc T^{k+1} = T^{k+1} - T^{k}  \f
  scatra_inc_np_->update(1.0, *(scatra_algo()->scatra_field()->phinp()), -1.0);
  structure_inc_np_->update(1.0, *(porofluid_elast_algo()->structure_dispnp()), -1.0);
  porofluid_inc_np_->update(1.0, *(porofluid_elast_algo()->fluid_phinp()), -1.0);
  if (artery_coupling_active_)
  {
    artery_pressure_inc_np_->update(
        1.0, *(porofluid_elast_algo()->porofluid_algo()->art_net_tim_int()->pressurenp()), -1.0);
    artery_scatra_inc_np_->update(
        1.0, *(scatra_meshtying_strategy_->art_scatra_field()->phinp()), -1.0);
  }

  // build the L2-norm of the scalar increment and the scalar
  scatra_inc_np_->norm_2(&scatra_inc_norm);
  scatra_algo()->scatra_field()->phinp()->norm_2(&scatra_norm);
  structure_inc_np_->norm_2(&structure_inc_norm);
  porofluid_elast_algo()->structure_dispnp()->norm_2(&structure_norm);
  porofluid_inc_np_->norm_2(&porofluid_inc_norm);
  porofluid_elast_algo()->fluid_phinp()->norm_2(&porofluid_norm);
  if (artery_coupling_active_)
  {
    artery_pressure_inc_np_->norm_2(&artery_pressure_inc_norm);
    porofluid_elast_algo()->porofluid_algo()->art_net_tim_int()->pressurenp()->norm_2(
        &artery_pressure_norm);
    artery_scatra_inc_np_->norm_2(&artery_scatra_inc_norm);
    porofluid_elast_algo()->porofluid_algo()->art_net_tim_int()->pressurenp()->norm_2(
        &artery_scatra_norm);
  }

  // care for the case that there is (almost) zero scalar
  if (scatra_norm < 1e-6) scatra_norm = 1.0;
  if (structure_norm < 1e-6) structure_norm = 1.0;
  if (porofluid_norm < 1e-6) porofluid_norm = 1.0;
  if (artery_pressure_norm < 1e-6) artery_pressure_norm = 1.0;
  if (artery_scatra_norm < 1e-6) artery_scatra_norm = 1.0;

  // print the incremental based convergence check to the screen
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    std::cout << "                                                                                 "
                 "                                                             *\n";
    std::cout << "+--------------------------------------------------------------------------------"
                 "-----------------------------------------+                   *\n";
    std::cout << "| PARTITIONED OUTER ITERATION STEP ----- MULTIPORO  <-------> SCATRA             "
                 "                                         |                   *\n";
    printf(
        "+--------------+---------------------+----------------+----------------+-----"
        "-----------+----------------+----------------+                   *\n");
    printf(
        "|-  step/max  -|-  tol      [norm]  -|-- scalar-inc --|-- disp-inc   --|-- "
        "fluid-inc  --|--  1Dp-inc   --|--  1Ds-inc   --|                   *\n");
    printf(
        "|   %3d/%3d    |  %10.3E[L_2 ]   | %10.3E     | %10.3E     | %10.3E     | "
        "%10.3E     | %10.3E     |",
        itnum, iter_max_, iter_tol_, scatra_inc_norm / scatra_norm,
        structure_inc_norm / structure_norm, porofluid_inc_norm / porofluid_norm,
        artery_pressure_inc_norm / artery_pressure_norm,
        artery_scatra_inc_norm / artery_scatra_norm);
    printf("                   *\n");
    printf(
        "+--------------+---------------------+----------------+----------------+-----"
        "-----------+----------------+----------------+                   *\n");
  }

  // converged
  if ((scatra_inc_norm / scatra_norm <= iter_tol_) and
      (structure_inc_norm / structure_norm <= iter_tol_) and
      (porofluid_inc_norm / porofluid_norm <= iter_tol_) and
      ((artery_pressure_inc_norm / artery_pressure_norm) <= iter_tol_) and
      ((artery_scatra_inc_norm / artery_scatra_norm) <= iter_tol_))
  {
    stop_nonlinear_iter = true;
    if (Core::Communication::my_mpi_rank(get_comm()) == 0)
    {
      printf(
          "* MULTIPORO  <-------> SCATRA Outer Iteration loop converged after iteration %3d/%3d !  "
          "                                                      *\n",
          itnum, iter_max_);
      printf(
          "****************************************************************************************"
          "*******************************************************\n");
    }
  }

  // break the loop
  // timestep
  if ((itnum == iter_max_) and
      ((scatra_inc_norm / scatra_norm > iter_tol_) or
          (structure_inc_norm / structure_norm > iter_tol_) or
          (porofluid_inc_norm / porofluid_norm > iter_tol_) or
          ((artery_pressure_inc_norm / artery_pressure_norm) > iter_tol_) or
          ((artery_scatra_inc_norm / artery_scatra_norm) > iter_tol_)))
  {
    stop_nonlinear_iter = true;
    if ((Core::Communication::my_mpi_rank(get_comm()) == 0))
    {
      printf(
          "* MULTIPORO  <-------> SCATRA Outer Iteration loop not converged in itemax steps        "
          "                                                      *\n");
      printf(
          "****************************************************************************************"
          "*******************************************************\n");
      printf("\n");
      printf("\n");
    }
    handle_divergence();
  }

  return stop_nonlinear_iter;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
PoroPressureBased::PorofluidElastScatraNestedPartitionedAlgorithm::
    PorofluidElastScatraNestedPartitionedAlgorithm(
        MPI_Comm comm, const Teuchos::ParameterList& globaltimeparams)
    : PorofluidElastScatraPartitionedAlgorithm(comm, globaltimeparams)
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraNestedPartitionedAlgorithm::init(
    const Teuchos::ParameterList& globaltimeparams, const Teuchos::ParameterList& algoparams,
    const Teuchos::ParameterList& poroparams, const Teuchos::ParameterList& structparams,
    const Teuchos::ParameterList& fluidparams, const Teuchos::ParameterList& scatraparams,
    const std::string& struct_disname, const std::string& fluid_disname,
    const std::string& scatra_disname, bool isale, int nds_disp, int nds_vel, int nds_solidpressure,
    int ndsporofluid_scatra, const std::map<int, std::set<int>>* nearby_ele_pairs)
{
  // call base class
  PoroPressureBased::PorofluidElastScatraPartitionedAlgorithm::init(globaltimeparams, algoparams,
      poroparams, structparams, fluidparams, scatraparams, struct_disname, fluid_disname,
      scatra_disname, isale, nds_disp, nds_vel, nds_solidpressure, ndsporofluid_scatra,
      nearby_ele_pairs);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraNestedPartitionedAlgorithm::solve()
{
  int itnum = 0;
  bool stopnonliniter = false;

  print_header_partitioned();

  while (stopnonliniter == false)
  {
    itnum++;

    // update the states to the last solutions obtained
    iter_update_states();

    // set structure-based scalar transport values
    set_scatra_solution();

    // solve structural system
    do_poro_step();

    // set mesh displacement and velocity fields
    set_porofluid_elast_solution();

    // solve scalar transport equation
    do_scatra_step();

    // check convergence for all fields and stop iteration loop if
    // convergence is achieved overall
    stopnonliniter = convergence_check(itnum);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
PoroPressureBased::PorofluidElastScatraSequentialPartitionedAlgorithm::
    PorofluidElastScatraSequentialPartitionedAlgorithm(
        MPI_Comm comm, const Teuchos::ParameterList& globaltimeparams)
    : PorofluidElastScatraPartitionedAlgorithm(comm, globaltimeparams)
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraSequentialPartitionedAlgorithm::init(
    const Teuchos::ParameterList& globaltimeparams, const Teuchos::ParameterList& algoparams,
    const Teuchos::ParameterList& poroparams, const Teuchos::ParameterList& structparams,
    const Teuchos::ParameterList& fluidparams, const Teuchos::ParameterList& scatraparams,
    const std::string& struct_disname, const std::string& fluid_disname,
    const std::string& scatra_disname, bool isale, int nds_disp, int nds_vel, int nds_solidpressure,
    int ndsporofluid_scatra, const std::map<int, std::set<int>>* nearby_ele_pairs)
{
  // call base class
  PoroPressureBased::PorofluidElastScatraPartitionedAlgorithm::init(globaltimeparams, algoparams,
      poroparams, structparams, fluidparams, scatraparams, struct_disname, fluid_disname,
      scatra_disname, isale, nds_disp, nds_vel, nds_solidpressure, ndsporofluid_scatra,
      nearby_ele_pairs);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraSequentialPartitionedAlgorithm::solve()
{
  int itnum = 0;
  bool stopnonliniter = false;

  print_header_partitioned();

  while (stopnonliniter == false)
  {
    itnum++;

    // update the states to the last solutions obtained
    iter_update_states();

    // 1) set scatra and structure solution (on fluid field)
    set_scatra_solution();
    porofluid_elast_algo()->set_structure_solution(
        porofluid_elast_algo()->structure_algo()->dispnp(),
        porofluid_elast_algo()->structure_algo()->velnp());

    // 2) solve fluid
    porofluid_elast_algo()->porofluid_algo()->solve();

    // 3) relaxation
    porofluid_elast_algo()->perform_relaxation(
        porofluid_elast_algo()->porofluid_algo()->phinp(), itnum);

    // 4) set relaxed fluid solution on structure field
    porofluid_elast_algo()->set_relaxed_fluid_solution();

    // 5) solve structure
    porofluid_elast_algo()->structure_algo()->solve();

    // 6) set mesh displacement and velocity fields on ScaTra
    set_porofluid_elast_solution();

    // 7) solve scalar transport equation
    do_scatra_step();

    // check convergence for all fields and stop iteration loop if
    // convergence is achieved overall
    stopnonliniter = convergence_check(itnum);
  }
}

FOUR_C_NAMESPACE_CLOSE

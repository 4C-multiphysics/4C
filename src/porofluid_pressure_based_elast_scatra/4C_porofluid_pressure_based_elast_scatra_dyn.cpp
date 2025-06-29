// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_porofluid_pressure_based_elast_scatra_dyn.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_porofluid_pressure_based_elast_scatra_base.hpp"
#include "4C_porofluid_pressure_based_elast_scatra_utils.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void porofluid_pressure_based_elast_scatra_dyn(int restart)
{
  // define the discretization names
  const std::string struct_disname = "structure";
  const std::string porofluid_disname = "porofluid";
  const std::string scatra_disname = "scatra";

  // access the problem
  Global::Problem* problem = Global::Problem::instance();

  // access the communicator
  MPI_Comm comm = problem->get_dis(struct_disname)->get_comm();

  // print problem type
  if (Core::Communication::my_mpi_rank(comm) == 0)
  {
    std::cout << "###################################################" << '\n';
    std::cout << "# YOUR PROBLEM TYPE: " << problem->problem_name() << '\n';
    std::cout << "###################################################" << '\n';
  }

  // Parameter reading
  // access porofluid-elasticity-scatra params list
  const Teuchos::ParameterList& porofluid_elast_scatra_params =
      problem->poro_multi_phase_scatra_dynamic_params();
  // access porofluid-elasticity params list
  const Teuchos::ParameterList& porofluid_elast_params = problem->poro_multi_phase_dynamic_params();
  // access structure params list
  const Teuchos::ParameterList& structure_params = problem->structural_dynamic_params();
  // access porofluid dynamic params list
  const Teuchos::ParameterList& porofluid_params =
      problem->porofluid_pressure_based_dynamic_params();
  // access scatra dynamic params list
  const Teuchos::ParameterList& scatra_params = problem->scalar_transport_dynamic_params();

  // do we perform coupling with 1D artery
  const bool artery_coupling = porofluid_elast_scatra_params.get<bool>("artery_coupling_active");

  // initialize variables for dof set numbers
  int nds_disp(-1);
  int nds_vel(-1);
  int nds_solidpressure(-1);
  int nds_porofluid_scatra(-1);

  // Setup discretizations and coupling. Assign the dof sets and return the numbers
  std::map<int, std::set<int>> nearby_ele_pairs =
      PoroPressureBased::setup_discretizations_and_field_coupling_porofluid_elast_scatra(comm,
          struct_disname, porofluid_disname, scatra_disname, nds_disp, nds_vel, nds_solidpressure,
          nds_porofluid_scatra, artery_coupling);

  // -------------------------------------------------------------------
  // algorithm construction depending on
  // coupling scheme
  // -------------------------------------------------------------------
  auto solution_scheme =
      Teuchos::getIntegralValue<PoroPressureBased::SolutionSchemePorofluidElastScatra>(
          porofluid_elast_scatra_params, "coupling_scheme");

  std::shared_ptr<PoroPressureBased::PorofluidElastScatraBaseAlgorithm> algo =
      PoroPressureBased::create_algorithm_porofluid_elast_scatra(
          solution_scheme, porofluid_elast_scatra_params, comm);

  algo->init(porofluid_elast_scatra_params, porofluid_elast_scatra_params, porofluid_elast_params,
      structure_params, porofluid_params, scatra_params, struct_disname, porofluid_disname,
      scatra_disname, true, nds_disp, nds_vel, nds_solidpressure, nds_porofluid_scatra,
      &nearby_ele_pairs);

  // read the restart information, set vectors and variables
  if (restart)
  {
    algo->read_restart(restart);
  }
  else
  {
    algo->post_init();
  }

  // assign materials
  // note: to be done after potential restart, as in read_restart() the secondary material is
  // destroyed
  PoroPressureBased::assign_material_pointers_porofluid_elast_scatra(
      struct_disname, porofluid_disname, scatra_disname, artery_coupling);

  // Setup Solver (necessary if poro-structure coupling solved monolithically)
  algo->setup_solver();

  // Run of the actual problem.

  // Some setup needed for the subproblems.
  algo->setup_system();

  // Solve the whole problem
  algo->time_loop();

  // Summarize the performance measurements
  Teuchos::TimeMonitor::summarize();

  // perform the result test if required
  algo->create_field_test();
  problem->test_all(comm);
}

FOUR_C_NAMESPACE_CLOSE

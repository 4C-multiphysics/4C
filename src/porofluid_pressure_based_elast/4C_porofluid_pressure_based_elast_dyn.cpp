// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_porofluid_pressure_based_elast_dyn.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_porofluid_pressure_based_elast_adapter.hpp"
#include "4C_porofluid_pressure_based_elast_input.hpp"
#include "4C_porofluid_pressure_based_elast_utils.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | Main control routine                                      vuong 08/16 |
 *----------------------------------------------------------------------*/
void poromultiphase_dyn(int restart)
{
  // define the discretization names
  const std::string struct_disname = "structure";
  const std::string fluid_disname = "porofluid";

  // access the problem
  Global::Problem* problem = Global::Problem::instance();

  // access the communicator
  MPI_Comm comm = problem->get_dis(struct_disname)->get_comm();

  // print problem type
  if (Core::Communication::my_mpi_rank(comm) == 0)
  {
    POROMULTIPHASE::print_logo();
    std::cout << "###################################################" << std::endl;
    std::cout << "# YOUR PROBLEM TYPE: " << problem->problem_name() << std::endl;
    std::cout << "###################################################" << std::endl;
  }

  // initialize variables for dof set numbers
  int nds_disp(-1);
  int nds_vel(-1);
  int nds_solidpressure(-1);

  // Setup discretizations and coupling. Assign the dof sets and return the numbers
  std::map<int, std::set<int>> nearbyelepairs =
      POROMULTIPHASE::Utils::setup_discretizations_and_field_coupling(
          comm, struct_disname, fluid_disname, nds_disp, nds_vel, nds_solidpressure);

  // Parameter reading
  const Teuchos::ParameterList& poroparams = problem->poro_multi_phase_dynamic_params();
  // access scatra params list
  const Teuchos::ParameterList& structdyn = problem->structural_dynamic_params();
  // access poro fluid dynamic params list
  const Teuchos::ParameterList& fluiddyn = problem->poro_fluid_multi_phase_dynamic_params();

  // -------------------------------------------------------------------
  // algorithm construction depending on
  // coupling scheme
  // -------------------------------------------------------------------
  auto solscheme =
      Teuchos::getIntegralValue<POROMULTIPHASE::SolutionSchemeOverFields>(poroparams, "COUPALGO");

  std::shared_ptr<POROMULTIPHASE::PoroMultiPhase> algo =
      POROMULTIPHASE::Utils::create_poro_multi_phase_algorithm(solscheme, poroparams, comm);

  // initialize
  algo->init(poroparams, poroparams, structdyn, fluiddyn, struct_disname, fluid_disname, true,
      nds_disp, nds_vel, nds_solidpressure,
      -1,  // no scalar field
      &nearbyelepairs);

  // read the restart information, set vectors and variables
  if (restart)
  {
    algo->read_restart(restart);
  }
  else
  {
    algo->post_init();
  }

  // assign poro material for evaluation of porosity
  // note: to be done after potential restart, as in read_restart()
  //       the secondary material is destroyed
  POROMULTIPHASE::Utils::assign_material_pointers(struct_disname, fluid_disname);

  // Setup the solver (only for the monolithic problem)
  algo->setup_solver();

  // Run of the actual problem.

  // Some setup needed for the subproblems.
  algo->setup_system();

  // Solve the whole problem
  algo->timeloop();

  // Summarize the performance measurements
  Teuchos::TimeMonitor::summarize();

  // perform the result test if required
  algo->create_field_test();
  problem->test_all(comm);

  return;

}  // poromultiphase_dyn

FOUR_C_NAMESPACE_CLOSE

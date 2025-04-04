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
void poromultiphasescatra_dyn(int restart)
{
  // define the discretization names
  const std::string struct_disname = "structure";
  const std::string fluid_disname = "porofluid";
  const std::string scatra_disname = "scatra";

  // access the problem
  Global::Problem* problem = Global::Problem::instance();

  // access the communicator
  MPI_Comm comm = problem->get_dis(struct_disname)->get_comm();

  // print problem type
  if (Core::Communication::my_mpi_rank(comm) == 0)
  {
    PoroMultiPhaseScaTra::print_logo();
    std::cout << "###################################################" << std::endl;
    std::cout << "# YOUR PROBLEM TYPE: " << problem->problem_name() << std::endl;
    std::cout << "###################################################" << std::endl;
  }

  // Parameter reading
  // access poro multiphase scatra params list
  const Teuchos::ParameterList& poroscatraparams =
      problem->poro_multi_phase_scatra_dynamic_params();
  // access poro multiphase params list
  const Teuchos::ParameterList& poroparams = problem->poro_multi_phase_dynamic_params();
  // access scatra params list
  const Teuchos::ParameterList& structparams = problem->structural_dynamic_params();
  // access poro fluid dynamic params list
  const Teuchos::ParameterList& fluidparams = problem->poro_fluid_multi_phase_dynamic_params();
  // access scatra dynamic params list
  const Teuchos::ParameterList& scatraparams = problem->scalar_transport_dynamic_params();

  // do we perform coupling with 1D artery
  const bool artery_coupl = poroscatraparams.get<bool>("ARTERY_COUPLING");

  // initialize variables for dof set numbers
  int ndsporo_disp(-1);
  int ndsporo_vel(-1);
  int ndsporo_solidpressure(-1);
  int ndsporofluid_scatra(-1);

  // Setup discretizations and coupling. Assign the dof sets and return the numbers
  std::map<int, std::set<int>> nearbyelepairs =
      PoroMultiPhaseScaTra::Utils::setup_discretizations_and_field_coupling(comm, struct_disname,
          fluid_disname, scatra_disname, ndsporo_disp, ndsporo_vel, ndsporo_solidpressure,
          ndsporofluid_scatra, artery_coupl);

  // -------------------------------------------------------------------
  // algorithm construction depending on
  // coupling scheme
  // -------------------------------------------------------------------
  auto solscheme = Teuchos::getIntegralValue<PoroMultiPhaseScaTra::SolutionSchemeOverFields>(
      poroscatraparams, "COUPALGO");

  std::shared_ptr<PoroMultiPhaseScaTra::PoroMultiPhaseScaTraBase> algo =
      PoroMultiPhaseScaTra::Utils::create_poro_multi_phase_scatra_algorithm(
          solscheme, poroscatraparams, comm);

  algo->init(poroscatraparams, poroscatraparams, poroparams, structparams, fluidparams,
      scatraparams, struct_disname, fluid_disname, scatra_disname, true, ndsporo_disp, ndsporo_vel,
      ndsporo_solidpressure, ndsporofluid_scatra, &nearbyelepairs);

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
  // note: to be done after potential restart, as in read_restart()
  //       the secondary material is destroyed
  PoroMultiPhaseScaTra::Utils::assign_material_pointers(
      struct_disname, fluid_disname, scatra_disname, artery_coupl);

  // Setup Solver (necessary if poro-structure coupling solved monolithically)
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


}  // poromultiphase_dyn

FOUR_C_NAMESPACE_CLOSE

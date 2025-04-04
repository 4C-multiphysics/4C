// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_tsi_dyn.hpp"

#include "4C_adapter_str_structure.hpp"
#include "4C_comm_mpi_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_tsi.hpp"
#include "4C_thermo_adapter.hpp"
#include "4C_tsi_algorithm.hpp"
#include "4C_tsi_monolithic.hpp"
#include "4C_tsi_partitioned.hpp"
#include "4C_tsi_utils.hpp"
#include "4C_utils_enum.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 | entry point for TSI in discretization management          dano 12/09 |
 *----------------------------------------------------------------------*/
void tsi_dyn_drt()
{
  // create a communicator
  MPI_Comm comm = Global::Problem::instance()->get_dis("structure")->get_comm();

  // print TSI-Logo to screen
  if (Core::Communication::my_mpi_rank(comm) == 0) TSI::printlogo();

  // setup of the discretizations, including clone strategy
  TSI::Utils::setup_tsi(comm);

  // access the problem-specific parameter list
  const Teuchos::ParameterList& tsidyn = Global::Problem::instance()->tsi_dynamic_params();
  // access the problem-specific parameter list
  const Teuchos::ParameterList& sdynparams =
      Global::Problem::instance()->structural_dynamic_params();
  const auto coupling =
      Teuchos::getIntegralValue<Inpar::TSI::SolutionSchemeOverFields>(tsidyn, "COUPALGO");

  // create an empty TSI::Algorithm instance
  std::shared_ptr<TSI::Algorithm> tsi;

  // choose algorithm depending on solution type
  switch (coupling)
  {
    case Inpar::TSI::Monolithic:
    {
      // create an TSI::Monolithic instance
      tsi = std::make_shared<TSI::Monolithic>(comm, sdynparams);
      break;
    }
    case Inpar::TSI::OneWay:
    case Inpar::TSI::SequStagg:
    case Inpar::TSI::IterStagg:
    case Inpar::TSI::IterStaggAitken:
    case Inpar::TSI::IterStaggAitkenIrons:
    case Inpar::TSI::IterStaggFixedRel:
    {
      // Any partitioned algorithm. Stable of working horses.
      // create an TSI::Algorithm instance
      tsi = std::make_shared<TSI::Partitioned>(comm);
      break;
    }
    default:
      FOUR_C_THROW("Unknown solutiontype for thermo-structure interaction: {}", coupling);
      break;
  }  // end switch

  const int restart = Global::Problem::instance()->restart();
  if (restart)
  {
    // read the restart information, set vectors and variables
    tsi->read_restart(restart);
  }
  else
  {
    // run post_setup to update the structure field
    tsi->post_setup();
  }

  // now do the coupling setup and create the combined dofmap
  tsi->setup_system();

  // solve the whole tsi problem
  tsi->time_loop();

  // summarize the performance measurements
  Teuchos::TimeMonitor::summarize();

  // perform the result test
  Global::Problem::instance()->add_field_test(tsi->structure_field()->create_field_test());
  Global::Problem::instance()->add_field_test(tsi->thermo_field()->create_field_test());
  Global::Problem::instance()->test_all(comm);

  return;
}  // tsi_dyn_drt()


/*----------------------------------------------------------------------*/

FOUR_C_NAMESPACE_CLOSE

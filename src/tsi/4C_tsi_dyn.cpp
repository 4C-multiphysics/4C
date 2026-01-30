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
#include "4C_thermo_adapter.hpp"
#include "4C_tsi_algorithm.hpp"
#include "4C_tsi_input.hpp"
#include "4C_tsi_monolithic.hpp"
#include "4C_tsi_partitioned.hpp"
#include "4C_tsi_utils.hpp"
#include "4C_utils_enum.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

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
      Teuchos::getIntegralValue<TSI::SolutionSchemeOverFields>(tsidyn, "COUPALGO");

  // create an empty TSI::Algorithm instance
  std::shared_ptr<TSI::Algorithm> tsi;

  // choose algorithm depending on solution type
  switch (coupling)
  {
    case TSI::SolutionSchemeOverFields::Monolithic:
    {
      // create an TSI::Monolithic instance
      tsi = std::make_shared<TSI::Monolithic>(comm, sdynparams);
      break;
    }
    case TSI::SolutionSchemeOverFields::OneWay:
    case TSI::SolutionSchemeOverFields::SequStagg:
    case TSI::SolutionSchemeOverFields::IterStagg:
    case TSI::SolutionSchemeOverFields::IterStaggAitken:
    case TSI::SolutionSchemeOverFields::IterStaggAitkenIrons:
    case TSI::SolutionSchemeOverFields::IterStaggFixedRel:
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

  // write initial state if requested and not a restart
  const bool write_initial_state =
      Global::Problem::instance()->io_params().get<bool>("WRITE_INITIAL_STATE") and (restart < 1);
  if (write_initial_state)
  {
    tsi->prepare_output();
    tsi->output();
  }

  // solve the whole tsi problem
  tsi->time_loop();

  // perform the result test
  Global::Problem::instance()->add_field_test(tsi->structure_field()->create_field_test());
  Global::Problem::instance()->add_field_test(tsi->thermo_field()->create_field_test());
  Global::Problem::instance()->test_all(comm);
}  // tsi_dyn_drt()


/*----------------------------------------------------------------------*/

FOUR_C_NAMESPACE_CLOSE

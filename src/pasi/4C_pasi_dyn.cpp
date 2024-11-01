// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_pasi_dyn.hpp"

#include "4C_comm_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_pasi.hpp"
#include "4C_pasi_partitioned_onewaycoup.hpp"
#include "4C_pasi_partitioned_twowaycoup.hpp"
#include "4C_pasi_utils.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
void pasi_dyn()
{
  // get pointer to global problem
  Global::Problem* problem = Global::Problem::instance();

  // create a communicator
  const Epetra_Comm& comm = problem->get_dis("structure")->get_comm();

  // print pasi logo to screen
  if (comm.MyPID() == 0) PaSI::Utils::logo();

  // get parameter list
  const Teuchos::ParameterList& params = problem->pasi_dynamic_params();

  // modification of time parameters of subproblems
  PaSI::Utils::change_time_parameter(comm, params,
      const_cast<Teuchos::ParameterList&>(problem->particle_params()),
      const_cast<Teuchos::ParameterList&>(problem->structural_dynamic_params()));

  // create particle structure interaction algorithm
  Teuchos::RCP<PaSI::PartitionedAlgo> algo = Teuchos::null;

  // get type of partitioned coupling
  const auto coupling =
      Teuchos::getIntegralValue<Inpar::PaSI::PartitionedCouplingType>(params, "COUPLING");

  // query algorithm
  switch (coupling)
  {
    case Inpar::PaSI::partitioned_onewaycoup:
    {
      algo = Teuchos::make_rcp<PaSI::PasiPartOneWayCoup>(comm, params);
      break;
    }
    case Inpar::PaSI::partitioned_twowaycoup:
    {
      algo = Teuchos::make_rcp<PaSI::PasiPartTwoWayCoup>(comm, params);
      break;
    }
    case Inpar::PaSI::partitioned_twowaycoup_disprelax:
    {
      algo = Teuchos::make_rcp<PaSI::PasiPartTwoWayCoupDispRelax>(comm, params);
      break;
    }
    case Inpar::PaSI::partitioned_twowaycoup_disprelaxaitken:
    {
      algo = Teuchos::make_rcp<PaSI::PasiPartTwoWayCoupDispRelaxAitken>(comm, params);
      break;
    }
    default:
    {
      FOUR_C_THROW("no valid coupling type for particle structure interaction specified!");
      break;
    }
  }

  // init pasi algorithm
  algo->init();

  // read restart information
  const int restart = problem->restart();
  if (restart) algo->read_restart(restart);

  // setup pasi algorithm
  algo->setup();

  // solve partitioned particle structure interaction
  algo->timeloop();

  // perform result tests
  algo->test_results(comm);

  // print summary statistics for all timers
  Teuchos::RCP<const Teuchos::Comm<int>> TeuchosComm =
      Core::Communication::to_teuchos_comm<int>(comm);
  Teuchos::TimeMonitor::summarize(TeuchosComm.ptr(), std::cout, false, true, false);
}

FOUR_C_NAMESPACE_CLOSE

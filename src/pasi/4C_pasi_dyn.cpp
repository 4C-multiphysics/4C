// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_pasi_dyn.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_comm_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_pasi_input.hpp"
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
  MPI_Comm comm = problem->get_dis("structure")->get_comm();

  // print pasi logo to screen
  if (Core::Communication::my_mpi_rank(comm) == 0) PaSI::Utils::logo();

  // get parameter list
  const Teuchos::ParameterList& params = problem->pasi_dynamic_params();

  // modification of time parameters of subproblems
  PaSI::Utils::change_time_parameter(comm, params,
      const_cast<Teuchos::ParameterList&>(problem->particle_params()),
      const_cast<Teuchos::ParameterList&>(problem->structural_dynamic_params()));

  // create particle structure interaction algorithm
  std::shared_ptr<PaSI::PartitionedAlgo> algo = nullptr;

  // get type of partitioned coupling
  const auto coupling =
      Teuchos::getIntegralValue<PaSI::PartitionedCouplingType>(params, "COUPLING");

  // query algorithm
  switch (coupling)
  {
    case PaSI::partitioned_onewaycoup:
    {
      algo = std::make_shared<PaSI::PasiPartOneWayCoup>(comm, params);
      break;
    }
    case PaSI::partitioned_twowaycoup:
    {
      algo = std::make_shared<PaSI::PasiPartTwoWayCoup>(comm, params);
      break;
    }
    case PaSI::partitioned_twowaycoup_disprelax:
    {
      algo = std::make_shared<PaSI::PasiPartTwoWayCoupDispRelax>(comm, params);
      break;
    }
    case PaSI::partitioned_twowaycoup_disprelaxaitken:
    {
      algo = std::make_shared<PaSI::PasiPartTwoWayCoupDispRelaxAitken>(comm, params);
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
  if (restart)
  {
    algo->read_restart(restart);
  }
  else
  {
    algo->post_init();
  }

  // setup pasi algorithm
  algo->setup();

  // solve partitioned particle structure interaction
  algo->timeloop();

  // perform result tests
  algo->test_results(comm);

  // print summary statistics for all timers
  std::shared_ptr<const Teuchos::Comm<int>> TeuchosComm =
      Core::Communication::to_teuchos_comm<int>(comm);
  Teuchos::TimeMonitor::summarize(Teuchos::Ptr(TeuchosComm.get()), std::cout, false, true, false);
}

FOUR_C_NAMESPACE_CLOSE

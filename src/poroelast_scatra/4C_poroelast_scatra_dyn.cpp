// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_poroelast_scatra_dyn.hpp"

#include "4C_poroelast_scatra_base.hpp"
#include "4C_poroelast_scatra_utils.hpp"
#include "4C_poroelast_scatra_utils_clonestrategy.hpp"
#include "4C_poroelast_scatra_utils_setup.hpp"
#include "4C_poroelast_utils_clonestrategy.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

void poro_scatra_drt()
{
  Global::Problem* problem = Global::Problem::instance();

  // 1.- Initialization
  MPI_Comm comm = problem->get_dis("structure")->get_comm();

  // 2.- Parameter reading
  const Teuchos::ParameterList& poroscatradynparams = problem->poro_scatra_control_params();

  PoroElastScaTra::Utils::setup_poro_scatra_discretizations<
      PoroElastScaTra::Utils::PoroelastCloneStrategyforScatraElements,
      PoroElastScaTra::Utils::PoroScatraCloneStrategy>();

  // 3.- Creation of Poroelastic + Scalar_Transport problem. (discretization called inside)
  std::shared_ptr<PoroElastScaTra::PoroScatraBase> poro_scatra =
      PoroElastScaTra::Utils::create_poro_scatra_algorithm(poroscatradynparams, comm);

  // 3.1- Read restart if needed. (discretization called inside)
  const int restart = problem->restart();
  poro_scatra->read_restart(restart);

  // 4.- Run of the actual problem.

  // 4.1.- Some setup needed for the poroelastic subproblem.
  poro_scatra->setup_system();

  // 4.2.- Solve the whole problem
  poro_scatra->timeloop();

  // 4.3.- Summarize the performance measurements
  Teuchos::TimeMonitor::summarize();

  // 5. - perform the result test
  poro_scatra->test_results(comm);
}

FOUR_C_NAMESPACE_CLOSE

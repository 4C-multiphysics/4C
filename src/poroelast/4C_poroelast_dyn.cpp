// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_poroelast_dyn.hpp"

#include "4C_poroelast_base.hpp"
#include "4C_poroelast_utils.hpp"
#include "4C_poroelast_utils_clonestrategy.hpp"
#include "4C_poroelast_utils_setup.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN


void poroelast_drt()
{
  Global::Problem* problem = Global::Problem::instance();

  // create a communicator
  MPI_Comm comm = problem->get_dis("structure")->get_comm();

  // print Logo to screen
  if (Core::Communication::my_mpi_rank(comm) == 0) PoroElast::print_logo();

  // setup of the discretizations, including clone strategy
  PoroElast::Utils::setup_poro<PoroElast::Utils::PoroelastCloneStrategy>();

  // access the problem-specific parameter list
  const Teuchos::ParameterList& poroelastdyn = problem->poroelast_dynamic_params();

  // choose algorithm depending on solution type
  std::shared_ptr<PoroElast::PoroBase> poroalgo =
      PoroElast::Utils::create_poro_algorithm(poroelastdyn, comm);

  // read the restart information, set vectors and variables
  const int restart = problem->restart();
  if (restart)
  {
    poroalgo->read_restart(restart);
  }
  else
  {
    poroalgo->post_setup();
  }

  // now do the coupling setup and create the combined dofmap
  poroalgo->setup_system();

  // solve the whole problem
  poroalgo->time_loop();

  // summarize the performance measurements
  Teuchos::TimeMonitor::summarize();

  // perform the result test
  poroalgo->test_results(comm);
}

FOUR_C_NAMESPACE_CLOSE

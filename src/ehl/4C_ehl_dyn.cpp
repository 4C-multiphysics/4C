// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_ehl_dyn.hpp"

#include "4C_ehl_monolithic.hpp"
#include "4C_ehl_partitioned.hpp"
#include "4C_ehl_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | 4C Logo for  EHL problems                             Faraji 05/19 |
 *----------------------------------------------------------------------*/
void printehllogo()
{
  std::cout << "---------------------------------------------------------------------------------"
            << std::endl;
  std::cout << "---------------------------------------------------------------------------------"
            << std::endl;
  std::cout << "-----------  Welcome to the Elasto-Hydrodynamic Lubrication problem  ------------"
            << std::endl;
  std::cout << "---------------------------------------------------------------------------------"
            << std::endl;
  std::cout << "---------------------------------------------------------------------------------"
            << std::endl;
  return;
}

void printehlmixlogo()
{
  std::cout << "---------------------------------------------------------------------------------"
            << std::endl;
  std::cout << "-----------------        Welcome to the problem type EHL        -----------------"
            << std::endl;
  std::cout << "-----------------               Mixed Lubrication               -----------------"
            << std::endl;
  std::cout << "-----------------           Averaged Reynolds Equation          -----------------"
            << std::endl;
  std::cout << "---------------------------------------------------------------------------------"
            << std::endl;
  return;
}

/*----------------------------------------------------------------------*
 | Main control routine for EHL problems                    wirtz 12/15 |
 *----------------------------------------------------------------------*/
void ehl_dyn()
{
  Global::Problem* problem = Global::Problem::instance();

  // 1.- Initialization
  MPI_Comm comm = problem->get_dis("structure")->get_comm();

  // 2.- Parameter reading
  Teuchos::ParameterList& ehlparams =
      const_cast<Teuchos::ParameterList&>(problem->elasto_hydro_dynamic_params());
  // access lubrication params list
  Teuchos::ParameterList& lubricationdyn =
      const_cast<Teuchos::ParameterList&>(problem->lubrication_dynamic_params());
  // do we want to use Modified Reynolds Equation?
  const bool modifiedreynolds = lubricationdyn.get<bool>("MODIFIED_REYNOLDS_EQU");

  // print problem specific logo
  if (!Core::Communication::my_mpi_rank(problem->get_dis("structure")->get_comm()))
  {
    if (!modifiedreynolds)
      printehllogo();
    else
      printehlmixlogo();
  }

  if (!Core::Communication::my_mpi_rank(problem->get_dis("structure")->get_comm()))
    EHL::printlogo();

  // access structural dynamic params list which will be possibly modified while creating the time
  // integrator
  Teuchos::ParameterList& sdyn =
      const_cast<Teuchos::ParameterList&>(Global::Problem::instance()->structural_dynamic_params());


  //  //Modification of time parameter list
  EHL::Utils::change_time_parameter(comm, ehlparams, lubricationdyn, sdyn);

  const auto coupling =
      Teuchos::getIntegralValue<EHL::SolutionSchemeOverFields>(ehlparams, "COUPALGO");

  // 3.- Creation of Lubrication + Structure problem. (discretization called inside)
  std::shared_ptr<EHL::Base> ehl = nullptr;

  // 3.1 choose algorithm depending on solution type
  switch (coupling)
  {
    case EHL::ehl_IterStagg:
      ehl = std::make_shared<EHL::Partitioned>(
          comm, ehlparams, lubricationdyn, sdyn, "structure", "lubrication");
      break;
    case EHL::ehl_Monolithic:
      ehl = std::make_shared<EHL::Monolithic>(
          comm, ehlparams, lubricationdyn, sdyn, "structure", "lubrication");
      break;
    default:
      FOUR_C_THROW("unknown coupling algorithm for EHL!");
      break;
  }

  // 3.2- Read restart if needed. (discretization called inside)
  const int restart = problem->restart();
  if (restart)
  {
    ehl->read_restart(restart);
  }
  else
  {
    // run post_setup
    ehl->post_setup();
  }

  // 4.- Run of the actual problem.

  // 4.1.- Some setup needed for the elastohydrodynamic lubrication problem.
  ehl->setup_system();

  // 4.2.- Solve the whole problem
  ehl->timeloop();

  // 4.3.- Summarize the performance measurements
  Teuchos::TimeMonitor::summarize();

  // 5. - perform the result test
  ehl->test_results(comm);
}

FOUR_C_NAMESPACE_CLOSE

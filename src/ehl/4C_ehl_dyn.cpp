/*--------------------------------------------------------------------------*/
/*! \file

\brief control routine for elastohydrodynamic lubrication (lubrication structure interaction)

\level 3

*/
/*--------------------------------------------------------------------------*/

#include "4C_ehl_dyn.hpp"

#include "4C_ehl_monolithic.hpp"
#include "4C_ehl_partitioned.hpp"
#include "4C_ehl_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"

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
  Global::Problem* problem = Global::Problem::Instance();

  // 1.- Initialization
  const Epetra_Comm& comm = problem->GetDis("structure")->Comm();

  // 2.- Parameter reading
  Teuchos::ParameterList& ehlparams =
      const_cast<Teuchos::ParameterList&>(problem->elasto_hydro_dynamic_params());
  // access lubrication params list
  Teuchos::ParameterList& lubricationdyn =
      const_cast<Teuchos::ParameterList&>(problem->lubrication_dynamic_params());
  // do we want to use Modified Reynolds Equation?
  bool modifiedreynolds =
      (Core::UTILS::IntegralValue<int>(lubricationdyn, "MODIFIED_REYNOLDS_EQU"));

  // print problem specific logo
  if (!problem->GetDis("structure")->Comm().MyPID())
  {
    if (!modifiedreynolds)
      printehllogo();
    else
      printehlmixlogo();
  }

  if (!problem->GetDis("structure")->Comm().MyPID()) EHL::printlogo();

  // access structural dynamic params list which will be possibly modified while creating the time
  // integrator
  Teuchos::ParameterList& sdyn =
      const_cast<Teuchos::ParameterList&>(Global::Problem::Instance()->structural_dynamic_params());


  //  //Modification of time parameter list
  EHL::Utils::ChangeTimeParameter(comm, ehlparams, lubricationdyn, sdyn);

  const Inpar::EHL::SolutionSchemeOverFields coupling =
      Core::UTILS::IntegralValue<Inpar::EHL::SolutionSchemeOverFields>(ehlparams, "COUPALGO");

  // 3.- Creation of Lubrication + Structure problem. (discretization called inside)
  Teuchos::RCP<EHL::Base> ehl = Teuchos::null;

  // 3.1 choose algorithm depending on solution type
  switch (coupling)
  {
    case Inpar::EHL::ehl_IterStagg:
      ehl = Teuchos::rcp(
          new EHL::Partitioned(comm, ehlparams, lubricationdyn, sdyn, "structure", "lubrication"));
      break;
    case Inpar::EHL::ehl_Monolithic:
      ehl = Teuchos::rcp(
          new EHL::Monolithic(comm, ehlparams, lubricationdyn, sdyn, "structure", "lubrication"));
      break;
    default:
      FOUR_C_THROW("unknown coupling algorithm for EHL!");
      break;
  }

  // 3.2- Read restart if needed. (discretization called inside)
  const int restart = problem->Restart();
  if (restart) ehl->read_restart(restart);

  // 4.- Run of the actual problem.

  // 4.1.- Some setup needed for the elastohydrodynamic lubrication problem.
  ehl->SetupSystem();

  // 4.2.- Solve the whole problem
  ehl->Timeloop();

  // 4.3.- Summarize the performance measurements
  Teuchos::TimeMonitor::summarize();

  // 5. - perform the result test
  ehl->TestResults(comm);
}

FOUR_C_NAMESPACE_CLOSE

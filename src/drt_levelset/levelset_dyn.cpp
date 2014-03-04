/*----------------------------------------------------------------------*/
/*!
\file levelset_dyn.cpp
\brief entry point for level-set transport problems

<pre>
Maintainer: Ursula Rasthofer
            rasthofer@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15236
</pre>
*/
/*----------------------------------------------------------------------*/

#include "levelset_dyn.H"
#include "levelset_algorithm.H"
#include "../drt_inpar/inpar_scatra.H"
#include "../drt_lib/drt_discret.H"
#include "../drt_lib/drt_globalproblem.H"
#include "../drt_adapter/adapter_scatra_base_algorithm.H"
#include <Epetra_MpiComm.h>
#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_TimeMonitor.hpp>
#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <iostream>


/*----------------------------------------------------------------------*
 * Main control routine for level set problems
 *----------------------------------------------------------------------*/
void levelset_dyn(int restart)
{
  // define abbreviation
  DRT::Problem* problem = DRT::Problem::Instance();

  // access the scatra discretization
  Teuchos::RCP<DRT::Discretization> scatradis = problem->GetDis("scatra");

  // access the communicator
  const Epetra_Comm& comm = scatradis->Comm();

  // print warning to screen
  if (comm.MyPID()==0)
   std::cout << "You are now about to enter the module for level-set problems!" <<std::endl;

  // access the level-set-specific parameter list
  const Teuchos::ParameterList& levelsetcontrol = problem->LevelSetControl();

  // access the scatra-specific parameter list
  const Teuchos::ParameterList& scatradyn = problem->ScalarTransportDynamicParams();

  // ensure that all dofs are assigned in the right order
  scatradis->FillComplete();

  // check velocity field
  const INPAR::SCATRA::VelocityField veltype
    = DRT::INPUT::IntegralValue<INPAR::SCATRA::VelocityField>(scatradyn,"VELOCITYFIELD");
  if (veltype != INPAR::SCATRA::velocity_function and veltype != INPAR::SCATRA::velocity_function_and_curve)
    dserror("Other velocity fields than a field given by a function not yet supported for level-set problems");

  // we directly use the elements from the scalar transport elements section
  if (scatradis->NumGlobalNodes()==0)
    dserror("No elements in the ---TRANSPORT ELEMENTS section");

  // get linear solver id from SCALAR TRANSPORT DYNAMIC
  const int linsolvernumber = scatradyn.get<int>("LINEAR_SOLVER");
  if (linsolvernumber == (-1))
    dserror("no linear solver defined for SCALAR_TRANSPORT problem. Please set LINEAR_SOLVER in SCALAR TRANSPORT DYNAMIC to a valid number!");

  // create instance of scalar transport basis algorithm (empty fluid discretization)
  Teuchos::RCP<ADAPTER::ScaTraBaseAlgorithm> scatrabase = Teuchos::rcp(new ADAPTER::ScaTraBaseAlgorithm(levelsetcontrol,false,"scatra",DRT::Problem::Instance()->SolverParams(linsolvernumber)));

  // get pointer to time integrator
  Teuchos::RCP<SCATRA::ScaTraTimIntImpl> levelsetalgo = scatrabase->ScaTraField();

  // read the restart information, set vectors and variables
  if (restart) levelsetalgo->ReadRestart(restart);

  // set initial velocity field
  // note: The order ReadRestart() before SetVelocityField() is important here!!
  //       The velocity field is not initialized in the constructor of the basic scalar field. Moreover, it is not
  //       read from restart data. Therefore, we first have to set the restart time in the function ReadRestart() and
  //       then in case of time-dependent velocity fields to evaluate the velocity function and curve.
  // bool true allows for setting old convective velocity required for particle coupling
  Teuchos::rcp_dynamic_cast<SCATRA::LevelSetAlgorithm>(levelsetalgo)->SetVelocityField(true);

  // time measurement: time loop
  TEUCHOS_FUNC_TIME_MONITOR("LEVEL SET:  + time loop");

  // enter time loop
  levelsetalgo->TimeLoop();

  // summarize performance measurements
  Teuchos::TimeMonitor::summarize();

  // perform result test if required
  Teuchos::rcp_dynamic_cast<SCATRA::LevelSetAlgorithm>(levelsetalgo)->TestResults();

  return;

} // end of levelset_dyn()


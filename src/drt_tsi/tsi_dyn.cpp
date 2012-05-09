/*!----------------------------------------------------------------------*/
/*!
\file tsi_dyn.cpp
\brief Control routine for thermo-structure-interaction problems.


<pre>
Maintainer: Caroline Danowski
            danowski@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15253
</pre>
*/

/*----------------------------------------------------------------------*
 | definitions                                               dano 12/09 |
 *----------------------------------------------------------------------*/
#include <mpi.h>
#include <Epetra_MpiComm.h>

/*----------------------------------------------------------------------*
 |  headers                                                  dano 12/09 |
 *----------------------------------------------------------------------*/
#include "tsi_dyn.H"
#include "tsi_algorithm.H"
#include "tsi_partitioned.H"
#include "tsi_monolithic.H"
#include "tsi_utils.H"
#include "../drt_inpar/inpar_tsi.H"
#include "../drt_lib/drt_globalproblem.H"
#include "../drt_adapter/adapter_thermo.H"

#include <Teuchos_TimeMonitor.hpp>

/*----------------------------------------------------------------------*
 | general problem data                                     m.gee 06/01 |
 | global variable GENPROB genprob is defined in global_control.c       |
 *----------------------------------------------------------------------*/
extern struct _GENPROB     genprob;

/*----------------------------------------------------------------------*
 | entry point for TSI in DRT                                dano 12/09 |
 *----------------------------------------------------------------------*/
//void tsi_dyn(int disnumsf,int disnumtf,int restart)
void tsi_dyn_drt()
{
  // create a communicator
  const Epetra_Comm& comm = DRT::Problem::Instance()->Dis(genprob.numsf,0)->Comm();

  // print TSI-Logo to screen
  if (comm.MyPID()==0) TSI::printlogo();

  // setup of the discretizations, including clone strategy
  TSI::UTILS::SetupTSI(comm);

  // access the problem-specific parameter list
  const Teuchos::ParameterList& tsidyn
    = DRT::Problem::Instance()->TSIDynamicParams();
  // access the problem-specific parameter list
  const Teuchos::ParameterList& sdynparams
    = DRT::Problem::Instance()->StructuralDynamicParams();
  const INPAR::TSI::SolutionSchemeOverFields coupling
    = DRT::INPUT::IntegralValue<INPAR::TSI::SolutionSchemeOverFields>(tsidyn,"COUPALGO");

  // create an empty TSI::Algorithm instance
  Teuchos::RCP<TSI::Algorithm> tsi = Teuchos::null;

  // choose algorithm depending on solution type
  switch (coupling)
  {
  case INPAR::TSI::Monolithic :
  {
    // create an TSI::Monolithic instance
    tsi = Teuchos::rcp(new TSI::Monolithic(comm,sdynparams));
    break;
  }
  case INPAR::TSI::OneWay :
  case INPAR::TSI::SequStagg :
  case INPAR::TSI::IterStagg :
  case INPAR::TSI::IterStaggAitken :
  case INPAR::TSI::IterStaggAitkenIrons :
  {
    // Any partitioned algorithm. Stable of working horses.
    // create an TSI::Algorithm instance
    tsi = Teuchos::rcp(new TSI::Partitioned(comm));
    break;
  }
  default:
     dserror("Unknown solutiontype for thermo-structure interaction: %d",coupling);
  }  // end switch

  const int restart = DRT::Problem::Instance()->Restart();
  if (restart)
  {
    // read the restart information, set vectors and variables
    tsi->ReadRestart(restart);
  }

  // now do the coupling setup and create the combined dofmap
  tsi->SetupSystem();

  // solve the whole tsi problem
  tsi->TimeLoop();

  // summarize the performance measurements
  Teuchos::TimeMonitor::summarize();

  // perform the result test
  DRT::Problem::Instance()->AddFieldTest(tsi->StructureField().CreateFieldTest());
  DRT::Problem::Instance()->AddFieldTest(tsi->ThermoField()->CreateFieldTest());
  DRT::Problem::Instance()->TestAll(comm);

  return;
} // tsi_dyn_drt()


/*----------------------------------------------------------------------*/

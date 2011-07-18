/*!---------------------------------------------------------------------*
\file combust_dyn.cpp

\brief control of dynamic combustion analysis

<pre>
Maintainer: Florian Henke
            henke@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15265
</pre>
*----------------------------------------------------------------------*/
#ifdef CCADISCRET

#ifdef PARALLEL
#include <mpi.h>
#include <Epetra_MpiComm.h>
#else
#include <Epetra_SerialComm.h>
#endif

#include "../drt_lib/drt_globalproblem.H"
#include "../drt_lib/drt_utils_createdis.H"
#include "../drt_inpar/inpar_fluid.H"
#include "combust_dyn.H"
#include "combust_utils.H"
#include "combust_algorithm.H"
#include "../drt_scatra/scatra_utils.H"
#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <Teuchos_TimeMonitor.hpp>

/*----------------------------------------------------------------------*
 |                                                       m.gee 06/01    |
 | general problem data                                                 |
 | global variable GENPROB genprob is defined in global_control.c       |
 *----------------------------------------------------------------------*/
extern struct _GENPROB     genprob;

/*------------------------------------------------------------------------------------------------*
 | main  control routine for dynamic combustion analysis                              henke 06/08 |
 *------------------------------------------------------------------------------------------------*/
void combust_dyn()
{
  // create a communicator
#ifdef PARALLEL
  Epetra_MpiComm comm(MPI_COMM_WORLD);
#else
  Epetra_SerialComm comm;
#endif

  //------------------------------------------------------------------------------------------------
  // print COMBUST-Logo on screen
  //------------------------------------------------------------------------------------------------
  if (comm.MyPID()==0) COMBUST::printCombustLogo();

  //------------------------------------------------------------------------------------------------
  // create G-function discretization by copying the fluid discretization (fill with scatra elements)
  //------------------------------------------------------------------------------------------------
  // get discretization ids
  int disnumff = genprob.numff; // discretization number fluid; typically 0
  int disnumgff = genprob.numscatra; // discretization number G-function; typically 1
  /* remark: here the ScaTra discretization (genprob.numscatra) is renamed according to its physical
   * meaning in the combustion context, namely the G-function (disnumgff).*/

  // access fluid discretization
  RCP<DRT::Discretization> fluiddis = DRT::Problem::Instance()->Dis(disnumff,0);
  if (!fluiddis->Filled()) fluiddis->FillComplete(false,false,false);
  if (fluiddis->NumGlobalNodes()==0)
    dserror("No fluid discretization found!");

  // access G-function discretization (it should be empty)
  RCP<DRT::Discretization> gfuncdis = DRT::Problem::Instance()->Dis(disnumgff,0);
  if (!gfuncdis->Filled()) gfuncdis->FillComplete(false,false,false);

  if (gfuncdis->NumGlobalNodes()==0)
  {
    Epetra_Time time(comm);

    // access the scalar transport parameter list
    const Teuchos::ParameterList& scatracontrol = DRT::Problem::Instance()->ScalarTransportDynamicParams();
    const int matid = SCATRA::GetScaTraMatID(scatracontrol);

    // create the scatra discretization
    {
    Teuchos::RCP<DRT::UTILS::DiscretizationCreator<SCATRA::ScatraFluidCloneStrategy> > clonewizard =
          Teuchos::rcp(new DRT::UTILS::DiscretizationCreator<SCATRA::ScatraFluidCloneStrategy>() );

    clonewizard->CreateMatchingDiscretization(fluiddis,gfuncdis,matid);
    }
    if (comm.MyPID()==0)
      cout<<"Created G-function discretization from fluid discretization in...."
          <<time.ElapsedTime() << " secs\n\n";
  }
  else
    dserror("G-function discretization is not empty. Fluid and G-function already present!");

  //------------------------------------------------------------------------------------------------
  // create a combustion algorithm
  //------------------------------------------------------------------------------------------------
  // get the combustion parameter list
  const Teuchos::ParameterList combustdyn = DRT::Problem::Instance()->CombustionDynamicParams();
  // create a COMBUST::Algorithm instance
  Teuchos::RCP<COMBUST::Algorithm> combust_ = Teuchos::rcp(new COMBUST::Algorithm(comm, combustdyn));

  //------------------------------------------------------------------------------------------------
  // restart
  //------------------------------------------------------------------------------------------------
  if (genprob.restart)
  {
    // read the restart information, set vectors and variables
    combust_->RestartNew(genprob.restart);
    //combust_->Restart(genprob.restart);
  }
  //------------------------------------------------------------------------------------------------
  // call one of the available time integration schemes
  //------------------------------------------------------------------------------------------------
  INPAR::FLUID::TimeIntegrationScheme timeintscheme = DRT::INPUT::IntegralValue<INPAR::FLUID::TimeIntegrationScheme>(combustdyn,"TIMEINT");

  if (timeintscheme == INPAR::FLUID::timeint_one_step_theta or
      timeintscheme == INPAR::FLUID::timeint_afgenalpha)
  {
    // solve a dynamic combustion problem
    combust_->TimeLoop();
    /* remark: Hier kann auch z.B. Genalpha mit combust->TimeLoop() gerufen werden, weil der
     * combustion Algorithmus ja schon weiss welche Zeitintegration er hat. Es muss dann eine Klasse
     * "GenalphaTimeInt" existieren, die eine Funktion TimeLoop() hat. Dann muss allerdings auch
     * das ADAPTER::FluidCombust ein entsprechendes Object GenalphaTimeInt haben. Momentan hat ein
     * FluidCombust immer ein CombustFluidImplicitTimeInt!
     */
  }
  else if (timeintscheme == INPAR::FLUID::timeint_stationary)
  {
    // solve a static combustion problem
    combust_->SolveStationaryProblem();
  }
  else
  {
    // every time integration scheme must be either static or dynamic
    dserror("the combustion module can not handle this type of time integration scheme");
  }

  //------------------------------------------------------------------------------------------------
  // validate the results
  //------------------------------------------------------------------------------------------------
    // summarize the performance measurements
  Teuchos::TimeMonitor::summarize();

  // perform the result test
  DRT::Problem::Instance()->AddFieldTest(combust_->FluidField().CreateFieldTest());
  DRT::Problem::Instance()->AddFieldTest(combust_->CreateScaTraFieldTest());
  DRT::Problem::Instance()->TestAll(comm);

  return;

} // combust_dyn()

#endif  // #ifdef CCADISCRET

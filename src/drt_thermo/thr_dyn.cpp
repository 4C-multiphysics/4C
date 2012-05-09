/*----------------------------------------------------------------------*/
/*!
\file thr_dyn.cpp
\brief entry point for (in)stationary heat conduction

<pre>
Maintainer: Burkhard Bornemann
            bornemann@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15237
</pre>
*/

/*----------------------------------------------------------------------*
 |  definitions                                               gjb 01/08 |
 *----------------------------------------------------------------------*/
#ifdef CCADISCRET

#include <Teuchos_TimeMonitor.hpp>
#include <mpi.h>

/*----------------------------------------------------------------------*
 |  headers                                                   gjb 01/08 |
 *----------------------------------------------------------------------*/
#include "thr_dyn.H"
#include "../drt_inpar/drt_validparameters.H"
#include "../drt_lib/drt_globalproblem.H"
#include "../drt_adapter/adapter_thermo.H"
#include "thr_resulttest.H"

/*----------------------------------------------------------------------*
 | general problem data                                     m.gee 06/01 |
 | global variable GENPROB genprob is defined in global_control.c       |
 *----------------------------------------------------------------------*/
extern struct _GENPROB     genprob;

/*----------------------------------------------------------------------*
 | Main control routine for (in)stationary heat conduction              |
 *----------------------------------------------------------------------*/
void thr_dyn_drt()
{
  // access the discretization
  Teuchos::RCP<DRT::Discretization> thermodis = Teuchos::null;
  thermodis = DRT::Problem::Instance()->Dis(genprob.numtf, 0);

  // set degrees of freedom in the discretization
  if (not thermodis->Filled()) thermodis->FillComplete();

  const Teuchos::ParameterList& tdyn
    = DRT::Problem::Instance()->ThermalDynamicParams();

  // the adapter expects a couple of variables that do not exist in the
  // ThermalDynamicParams() list so rename them here to the expected name
  // (like in stru_dyn_nln_drt.cpp)
  int upres = DRT::Problem::Instance()->ThermalDynamicParams().get<int>("RESEVRYGLOB");
  const_cast<Teuchos::ParameterList&>(DRT::Problem::Instance()->ThermalDynamicParams()).set<int>("UPRES",upres);

  // create instance of thermo basis algorithm (no structure discretization)
  Teuchos::RCP<ADAPTER::ThermoBaseAlgorithm> thermoonly
    = rcp(new ADAPTER::ThermoBaseAlgorithm(tdyn));

  // do restart if demanded from input file
  const int restart = DRT::Problem::Instance()->Restart();
  if (restart){thermoonly->ThermoField().ReadRestart(restart);}

  // enter time loop to solve problem
  (thermoonly->ThermoField()).Integrate();

  // perform the result test if required
  DRT::Problem::Instance()->AddFieldTest(thermoonly->ThermoField().CreateFieldTest());
  DRT::Problem::Instance()->TestAll(thermodis->Comm());

  // done
  return;

} // end of thr_dyn_drt()


/*----------------------------------------------------------------------*/
#endif  // #ifdef CCADISCRET

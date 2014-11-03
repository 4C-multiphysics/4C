/*----------------------------------------------------------------------------*/
/*!
\file ale_dyn.cpp

<pre>
Maintainer: Matthias Mayr
            mayr@mhpc.mw.tum.de
            089 - 289-10362
</pre>
*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
#include <Teuchos_RCP.hpp>

#include "ale_dyn.H"

#include "ale_resulttest.H"

#include "../drt_adapter/ad_ale.H"

#include "../drt_lib/drt_globalproblem.H"
#include "../drt_lib/drt_discret.H"

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void dyn_ale_new_drt()
{
  // -------------------------------------------------------------------
  // access the discretization
  // -------------------------------------------------------------------
  Teuchos::RCP<DRT::Discretization> actdis = DRT::Problem::Instance()->GetDis("ale");

  // -------------------------------------------------------------------
  // ask ALE::AleBaseAlgorithm for the ale time integrator
  // -------------------------------------------------------------------
  Teuchos::RCP< ::ADAPTER::AleNewBaseAlgorithm> ale =
      Teuchos::rcp(new ::ADAPTER::AleNewBaseAlgorithm(DRT::Problem::Instance()->AleDynamicParams(), actdis));
  Teuchos::RCP< ::ADAPTER::Ale> aletimint = ale->AleField();

  // -------------------------------------------------------------------
  // read the restart information, set vectors and variables if necessary
  // -------------------------------------------------------------------
  const int restart = DRT::Problem::Instance()->Restart();
  if (restart)
    aletimint->ReadRestart(restart);

  // -------------------------------------------------------------------
  // call time loop
  // -------------------------------------------------------------------
  aletimint->CreateSystemMatrix();
  aletimint->Integrate();

  // -------------------------------------------------------------------
  // do the result test
  // -------------------------------------------------------------------
  // test results
  DRT::Problem::Instance()->AddFieldTest(aletimint->CreateFieldTest());
  DRT::Problem::Instance()->TestAll(actdis->Comm());

  return;
}



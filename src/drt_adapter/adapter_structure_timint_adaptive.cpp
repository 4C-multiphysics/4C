/*----------------------------------------------------------------------*/
/*!
\file adapter_structure_timint.cpp

\brief Structure field adapter

<pre>
Maintainer: Burkhard Bornemann
            bornemann@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15237
</pre>
*/

/*----------------------------------------------------------------------*/
/* headers */
#include "../drt_structure/strtimint_create.H"
#include "../drt_structure/strtimada.H"
#include "../drt_structure/strtimint.H"
#include "adapter_structure_timint_adaptive.H"


/*======================================================================*/
/* constructor */
ADAPTER::StructureTimIntAda::StructureTimIntAda(
  Teuchos::RCP<STR::TimAda> sta,
  Teuchos::RCP<STR::TimInt> sti
)
: FSIStructureWrapper(sti),
  structure_(sta)
{
  // make sure
  if (structure_ == Teuchos::null)
    dserror("Failed to create structural integrator");

}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ADAPTER::StructureTimIntAda::Integrate()
{
  structure_->Integrate();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ADAPTER::StructureTimIntAda::PrepareOutput()
{
  structure_->PrepareOutputPeriod();
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ADAPTER::StructureTimIntAda::Output()
{
  structure_->OutputPeriod();
}


/*----------------------------------------------------------------------*/

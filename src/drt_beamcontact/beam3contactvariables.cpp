/*!-----------------------------------------------------------------------------------------------------------
\file beam3contactvariabless.cpp
\brief One beam contact segment living on an element pair

<pre>
Maintainer: Christoph Meier
            meier@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15262
</pre>

*-----------------------------------------------------------------------------------------------------------*/

#include "beam3contact.H"
#include "beam3contactvariables.H"
#include "beam3contact_defines.H"
#include "beam3contact_utils.H"
#include "beam3contact_tangentsmoothing.H"
#include "../drt_inpar/inpar_beamcontact.H"
#include "../drt_inpar/inpar_contact.H"
#include "../drt_lib/drt_discret.H"
#include "../drt_lib/drt_exporter.H"
#include "../drt_lib/drt_dserror.H"
#include "../linalg/linalg_utils.H"
#include "../drt_fem_general/drt_utils_fem_shapefunctions.H"
#include "../drt_lib/drt_globalproblem.H"

#include "../drt_structure/strtimint_impl.H"
#include "../drt_beam3/beam3.H"
#include "../drt_beam3ii/beam3ii.H"
#include "../drt_beam3eb/beam3eb.H"
#include "../drt_inpar/inpar_statmech.H"

#include "Teuchos_TimeMonitor.hpp"

/*----------------------------------------------------------------------*
 |  constructor (public)                                     meier 01/14|
 *----------------------------------------------------------------------*/
template<const int numnodes , const int numnodalvalues>
CONTACT::Beam3contactvariables<numnodes, numnodalvalues>::Beam3contactvariables(std::pair<TYPE,TYPE>& closestpoint,
                                                                                std::pair<int,int>& segids,
                                                                                const double& pp):
closestpoint_(closestpoint),
segids_(segids),
gap_(0.0),
normal_(LINALG::TMatrix<TYPE,3,1>(true)),
pp_(pp),
fp_(0.0),
dfp_(0.0),
angle_(0.0)
{
  return;
}
/*----------------------------------------------------------------------*
 |  end: constructor
 *----------------------------------------------------------------------*/

//Possible template cases: this is necessary for the compiler
template class CONTACT::Beam3contactvariables<2,1>;
template class CONTACT::Beam3contactvariables<3,1>;
template class CONTACT::Beam3contactvariables<4,1>;
template class CONTACT::Beam3contactvariables<5,1>;
template class CONTACT::Beam3contactvariables<2,2>;

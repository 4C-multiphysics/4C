/*----------------------------------------------------------------------*/
/*!
\file fluid_ele_calc_std.cpp

\brief standard routines for calculation of fluid element

<pre>
Maintainer: Ursula Rasthofer & Volker Gravemeier
            {rasthofer,vgravem}@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15236/-245
</pre>
*/
/*----------------------------------------------------------------------*/

#include "fluid_ele_calc_std.H"
#include "fluid_ele_parameter_std.H"
#include "../drt_inpar/inpar_fpsi.H"
#include "../drt_lib/drt_globalproblem.H"
#include "../drt_lib/drt_discret.H"


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template<DRT::Element::DiscretizationType distype>
DRT::ELEMENTS::FluidEleCalcStd<distype>* DRT::ELEMENTS::FluidEleCalcStd<distype>::Instance( bool create )
{
  static FluidEleCalcStd<distype> * instance;
  if ( create )
  {
    if ( instance==NULL )
    {
      instance = new FluidEleCalcStd<distype>();
    }
  }
  else
  {
    if ( instance!=NULL )
      delete instance;
    instance = NULL;
  }
  return instance;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::FluidEleCalcStd<distype>::Done()
{
  // delete this pointer! Afterwards we have to go! But since this is a
  // cleanup call, we can do it this way.
    Instance( false );
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
DRT::ELEMENTS::FluidEleCalcStd<distype>::FluidEleCalcStd()
  : DRT::ELEMENTS::FluidEleCalc<distype>::FluidEleCalc()
{
  my::fldpara_=DRT::ELEMENTS::FluidEleParameterStd::Instance();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/

// template classes
template class DRT::ELEMENTS::FluidEleCalcStd<DRT::Element::hex8>;
template class DRT::ELEMENTS::FluidEleCalcStd<DRT::Element::hex20>;
template class DRT::ELEMENTS::FluidEleCalcStd<DRT::Element::hex27>;
template class DRT::ELEMENTS::FluidEleCalcStd<DRT::Element::tet4>;
template class DRT::ELEMENTS::FluidEleCalcStd<DRT::Element::tet10>;
template class DRT::ELEMENTS::FluidEleCalcStd<DRT::Element::wedge6>;
template class DRT::ELEMENTS::FluidEleCalcStd<DRT::Element::pyramid5>;
template class DRT::ELEMENTS::FluidEleCalcStd<DRT::Element::quad4>;
template class DRT::ELEMENTS::FluidEleCalcStd<DRT::Element::quad8>;
template class DRT::ELEMENTS::FluidEleCalcStd<DRT::Element::quad9>;
template class DRT::ELEMENTS::FluidEleCalcStd<DRT::Element::tri3>;
template class DRT::ELEMENTS::FluidEleCalcStd<DRT::Element::tri6>;
template class DRT::ELEMENTS::FluidEleCalcStd<DRT::Element::nurbs9>;
template class DRT::ELEMENTS::FluidEleCalcStd<DRT::Element::nurbs27>;

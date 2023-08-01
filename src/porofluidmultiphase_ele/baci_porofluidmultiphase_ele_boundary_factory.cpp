/*----------------------------------------------------------------------*/
/*! \file
 \brief factory class providing the implementations of the porofluidmultiphase
        boundary element evaluation routines

   \level 3

 *----------------------------------------------------------------------*/


#include "baci_porofluidmultiphase_ele_boundary_factory.H"

#include "baci_lib_element.H"
#include "baci_lib_globalproblem.H"
#include "baci_porofluidmultiphase_ele_boundary_calc.H"
#include "baci_porofluidmultiphase_ele_interface.H"


/*--------------------------------------------------------------------------*
 | provide the implementation of evaluation class      (public) vuong 08/16 |
 *--------------------------------------------------------------------------*/
DRT::ELEMENTS::PoroFluidMultiPhaseEleInterface*
DRT::ELEMENTS::PoroFluidMultiPhaseBoundaryFactory::ProvideImpl(
    const DRT::Element* ele, const int numdofpernode, const std::string& disname)
{
  switch (ele->Shape())
  {
    case DRT::Element::quad4:
    {
      return DefineProblemType<DRT::Element::quad4>(numdofpernode, disname);
    }
    case DRT::Element::quad8:
    {
      return DefineProblemType<DRT::Element::quad8>(numdofpernode, disname);
    }
    case DRT::Element::quad9:
    {
      return DefineProblemType<DRT::Element::quad9>(numdofpernode, disname);
    }
    case DRT::Element::tri3:
    {
      return DefineProblemType<DRT::Element::tri3>(numdofpernode, disname);
    }
    case DRT::Element::tri6:
    {
      return DefineProblemType<DRT::Element::tri6>(numdofpernode, disname);
    }
    case DRT::Element::line2:
    {
      return DefineProblemType<DRT::Element::line2>(numdofpernode, disname);
    }
    case DRT::Element::line3:
    {
      return DefineProblemType<DRT::Element::line3>(numdofpernode, disname);
    }
    default:
    {
      dserror(
          "Element shape %d (%d nodes) not activated. Just do it.", ele->Shape(), ele->NumNode());
      break;
    }
  }

  return NULL;
}


/*--------------------------------------------------------------------------*
 | provide the implementation of evaluation class      (public) vuong 08/16 |
 *--------------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
DRT::ELEMENTS::PoroFluidMultiPhaseEleInterface*
DRT::ELEMENTS::PoroFluidMultiPhaseBoundaryFactory::DefineProblemType(
    const int numdofpernode, const std::string& disname)
{
  return DRT::ELEMENTS::PoroFluidMultiPhaseEleBoundaryCalc<distype>::Instance(
      numdofpernode, disname);
}

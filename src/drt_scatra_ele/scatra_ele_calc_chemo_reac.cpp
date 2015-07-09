/*----------------------------------------------------------------------*/
/*!
 \file scatra_ele_calc_chemo_reac.cpp

 \brief main file containing routines for calculation of scatra element with chemotactic AND reactive scalars


 <pre>
   Maintainer: Moritz Thon
               thon@mhpc.mw.tum.de
               http://www.lnm.mw.tum.de
               089 - 289-10364
 </pre>
 *----------------------------------------------------------------------*/

#include "scatra_ele_calc_chemo_reac.H"

#include "scatra_ele_parameter.H"

#include "../drt_lib/drt_globalproblem.H"
#include "../drt_lib/drt_discret.H"
#include "../drt_lib/drt_element.H"

#include "../drt_mat/matlist_chemoreac.H"
#include "../drt_mat/matlist_chemotaxis.H"
#include "../drt_mat/matlist_reactions.H"
#include "../drt_mat/scatra_mat.H"
#include "../drt_mat/matlist.H"


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype, int probdim>
DRT::ELEMENTS::ScaTraEleCalcChemoReac<distype,probdim>::ScaTraEleCalcChemoReac(const int numdofpernode,const int numscal)
  : DRT::ELEMENTS::ScaTraEleCalc<distype,probdim>::ScaTraEleCalc(numdofpernode,numscal),
    DRT::ELEMENTS::ScaTraEleCalcChemo<distype,probdim>::ScaTraEleCalcChemo(numdofpernode,numscal),
    DRT::ELEMENTS::ScaTraEleCalcAdvReac<distype,probdim>::ScaTraEleCalcAdvReac(numdofpernode,numscal)
{

}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype, int probdim>
DRT::ELEMENTS::ScaTraEleCalcChemoReac<distype,probdim> * DRT::ELEMENTS::ScaTraEleCalcChemoReac<distype,probdim>::Instance(
  const int numdofpernode,
  const int numscal,
  bool create )
{
  static ScaTraEleCalcChemoReac<distype,probdim> * instance;
  if ( create )
  {
    if ( instance==NULL )
    {
      instance = new ScaTraEleCalcChemoReac<distype,probdim>(numdofpernode,numscal);
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
template <DRT::Element::DiscretizationType distype, int probdim>
void DRT::ELEMENTS::ScaTraEleCalcChemoReac<distype,probdim>::Done()
{
  // delete this pointer! Afterwards we have to go! But since this is a
  // cleanup call, we can do it this way.
  Instance( 0, 0, false );
}

/*----------------------------------------------------------------------*
 |  get the material constants  (private)                    thon 06/15 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype, int probdim>
void DRT::ELEMENTS::ScaTraEleCalcChemoReac<distype,probdim>::GetMaterialParams(
      const DRT::Element* ele,       //!< the element we are dealing with
      double&             densn,     //!< density at t_(n)
      double&             densnp,    //!< density at t_(n+1) or t_(n+alpha_F)
      double&             densam,    //!< density at t_(n+alpha_M)
      double&             visc,       //!< fluid viscosity
      const int           iquad    //!< id of current gauss point (default = -1)
    )
{
  // get the material
  Teuchos::RCP<MAT::Material> material = ele->Material();

  // We may have some reactive and some non-reactive elements in one discretisation.
  // But since the calculation classes are singleton, we have to reset all reactive stuff in case
  // of non-reactive elements:
  advreac::ClearAdvancedReactionTerms();

  // We may have some chemotactic and some non-chemotactic discretisation.
  // But since the calculation classes are singleton, we have to reset all chemotaxis stuff each time
  chemo::ClearChemotaxisTerms();

  if (material->MaterialType() == INPAR::MAT::m_matlist)
  {
    const Teuchos::RCP<const MAT::MatList>& actmat = Teuchos::rcp_dynamic_cast<const MAT::MatList>(material);
    if (actmat->NumMat() != my::numscal_) dserror("Not enough materials in MatList.");

    for (int k = 0;k<my::numscal_;++k)
    {
      int matid = actmat->MatID(k);
      Teuchos::RCP< MAT::Material> singlemat = actmat->MaterialById(matid);

      my::Materials(singlemat,k,densn,densnp,densam,visc,iquad);
    }
  }

  else if (material->MaterialType() == INPAR::MAT::m_matlist_reactions)
  {
    const Teuchos::RCP<const MAT::MatListReactions>& actmat = Teuchos::rcp_dynamic_cast<const MAT::MatListReactions>(material);
    if (actmat->NumMat() != my::numscal_) dserror("Not enough materials in MatList.");

    advreac::GetAdvancedReactionCoefficients(actmat); // read all reaction input from material and copy it into local variables

    for (int k = 0;k<my::numscal_;++k)
    {
      int matid = actmat->MatID(k);
      Teuchos::RCP< MAT::Material> singlemat = actmat->MaterialById(matid);

      //Note: order is important here!!
      advreac::Materials(singlemat,k,densn,densnp,densam,visc,iquad);

      advreac::SetAdvancedReactionTerms(k,1.0); //every reaction calculation stuff happens in here!!
    }
  }

  else if (material->MaterialType() == INPAR::MAT::m_matlist_chemotaxis)
  {
    const Teuchos::RCP<const MAT::MatListChemotaxis>& actmat = Teuchos::rcp_dynamic_cast<const MAT::MatListChemotaxis>(material);
    if (actmat->NumMat() != my::numscal_) dserror("Not enough materials in MatList.");

    chemo::GetChemotaxisCoefficients(actmat); // read all chemotaxis input from material and copy it into local variables

    for (int k = 0;k<my::numscal_;++k)
    {
      int matid = actmat->MatID(k);
      Teuchos::RCP< MAT::Material> singlemat = actmat->MaterialById(matid);

      my::Materials(singlemat,k,densn,densnp,densam,visc,iquad);
    }

  }

  else if (material->MaterialType() == INPAR::MAT::m_matlist_chemoreac)
  {
    const Teuchos::RCP<const MAT::MatListChemoReac>& actmat = Teuchos::rcp_dynamic_cast<const MAT::MatListChemoReac>(material);
    if (actmat->NumMat() != my::numscal_) dserror("Not enough materials in MatList.");

    chemo::GetChemotaxisCoefficients(actmat); // read all chemotaxis input from material and copy it into local variables
    advreac::GetAdvancedReactionCoefficients(actmat); // read all reaction input from material and copy it into local variables

    for (int k = 0;k<my::numscal_;++k)
    {
      int matid = actmat->MatID(k);
      Teuchos::RCP< MAT::Material> singlemat = actmat->MaterialById(matid);

      //Note: order is important here!!
      my::Materials(singlemat,k,densn,densnp,densam,visc,iquad);
      advreac::SetAdvancedReactionTerms(k,1.0); //every reaction calculation stuff happens in here!!
    }
  }

  else
  {
    advreac::Materials(material,0,densn,densnp,densam,visc,iquad);
  }

  return;
} //ScaTraEleCalc::GetMaterialParams


// template classes

// 1D elements
template class DRT::ELEMENTS::ScaTraEleCalcChemoReac<DRT::Element::line2>;
template class DRT::ELEMENTS::ScaTraEleCalcChemoReac<DRT::Element::line3>;

// 2D elements
//template class DRT::ELEMENTS::ScaTraEleCalcChemoReac<DRT::Element::tri3>;
//template class DRT::ELEMENTS::ScaTraEleCalcChemoReac<DRT::Element::tri6>;
template class DRT::ELEMENTS::ScaTraEleCalcChemoReac<DRT::Element::quad4>;
//template class DRT::ELEMENTS::ScaTraEleCalcChemoReac<DRT::Element::quad8>;
template class DRT::ELEMENTS::ScaTraEleCalcChemoReac<DRT::Element::quad9>;

// 3D elements
template class DRT::ELEMENTS::ScaTraEleCalcChemoReac<DRT::Element::hex8>;
//template class DRT::ELEMENTS::ScaTraEleCalcChemoReac<DRT::Element::hex20>;
template class DRT::ELEMENTS::ScaTraEleCalcChemoReac<DRT::Element::hex27>;
template class DRT::ELEMENTS::ScaTraEleCalcChemoReac<DRT::Element::tet4>;
template class DRT::ELEMENTS::ScaTraEleCalcChemoReac<DRT::Element::tet10>;
//template class DRT::ELEMENTS::ScaTraEleCalcChemoReac<DRT::Element::wedge6>;
template class DRT::ELEMENTS::ScaTraEleCalcChemoReac<DRT::Element::pyramid5>;
template class DRT::ELEMENTS::ScaTraEleCalcChemoReac<DRT::Element::nurbs9>;
//template class DRT::ELEMENTS::ScaTraEleCalcChemoReac<DRT::Element::nurbs27>;

/*----------------------------------------------------------------------*/
/*! \file

\brief evaluation class containing routines for dissolving reactions within
       porous medium for ECM modeling

\level 2

 *----------------------------------------------------------------------*/

#include "baci_scatra_ele_calc_poro_reac_ECM.H"

#include "baci_lib_discret.H"
#include "baci_lib_element.H"
#include "baci_lib_globalproblem.H"
#include "baci_mat_list_reactions.H"
#include "baci_mat_scatra_mat.H"
#include "baci_mat_scatra_mat_poro_ecm.H"
#include "baci_mat_structporo.H"
#include "baci_mat_structporo_reaction_ecm.H"
#include "baci_scatra_ele_parameter_std.H"
#include "baci_utils_singleton_owner.H"


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
DRT::ELEMENTS::ScaTraEleCalcPoroReacECM<distype>::ScaTraEleCalcPoroReacECM(
    const int numdofpernode, const int numscal, const std::string& disname)
    : DRT::ELEMENTS::ScaTraEleCalc<distype>::ScaTraEleCalc(numdofpernode, numscal, disname),
      DRT::ELEMENTS::ScaTraEleCalcPoro<distype>::ScaTraEleCalcPoro(numdofpernode, numscal, disname),
      DRT::ELEMENTS::ScaTraEleCalcAdvReac<distype>::ScaTraEleCalcAdvReac(
          numdofpernode, numscal, disname),
      DRT::ELEMENTS::ScaTraEleCalcPoroReac<distype>::ScaTraEleCalcPoroReac(
          numdofpernode, numscal, disname)
{
  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
DRT::ELEMENTS::ScaTraEleCalcPoroReacECM<distype>*
DRT::ELEMENTS::ScaTraEleCalcPoroReacECM<distype>::Instance(
    const int numdofpernode, const int numscal, const std::string& disname)
{
  static auto singleton_map = CORE::UTILS::MakeSingletonMap<std::string>(
      [](const int numdofpernode, const int numscal, const std::string& disname)
      {
        return std::unique_ptr<ScaTraEleCalcPoroReacECM<distype>>(
            new ScaTraEleCalcPoroReacECM<distype>(numdofpernode, numscal, disname));
      });

  return singleton_map[disname].Instance(
      CORE::UTILS::SingletonAction::create, numdofpernode, numscal, disname);
}


/*----------------------------------------------------------------------*
 |  evaluate single material  (protected)                    thon 02/14 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::ScaTraEleCalcPoroReacECM<distype>::Materials(
    const Teuchos::RCP<const MAT::Material> material,  //!< pointer to current material
    const int k,                                       //!< id of current scalar
    double& densn,                                     //!< density at t_(n)
    double& densnp,                                    //!< density at t_(n+1) or t_(n+alpha_F)
    double& densam,                                    //!< density at t_(n+alpha_M)
    double& visc,                                      //!< fluid viscosity
    const int iquad                                    //!< id of current gauss point
)
{
  switch (material->MaterialType())
  {
    case INPAR::MAT::m_scatra:
      pororeac::MatScaTra(material, k, densn, densnp, densam, visc, iquad);
      break;
    default:
      dserror("Material type %i is not supported", material->MaterialType());
      break;
  }
  return;
}

/*----------------------------------------------------------------------*
 |  evaluate single material  (protected)                    vuong 10/14 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::ScaTraEleCalcPoroReacECM<distype>::GetMaterialParams(
    const DRT::Element* ele,      //!< the element we are dealing with
    std::vector<double>& densn,   //!< density at t_(n)
    std::vector<double>& densnp,  //!< density at t_(n+1) or t_(n+alpha_F)
    std::vector<double>& densam,  //!< density at t_(n+alpha_M)
    double& visc,                 //!< fluid viscosity
    const int iquad               //!< id of current gauss point
)
{
  // call poro base class to compute porosity
  poro::ComputePorosity(ele);

  // get the material
  Teuchos::RCP<MAT::Material> material = ele->Material();

  if (material->MaterialType() == INPAR::MAT::m_matlist_reactions)
  {
    const Teuchos::RCP<MAT::MatListReactions> actmat =
        Teuchos::rcp_dynamic_cast<MAT::MatListReactions>(material);
    if (actmat->NumMat() != my::numscal_) dserror("Not enough materials in MatList.");

    for (int k = 0; k < actmat->NumReac(); ++k)
    {
      int matid = actmat->ReacID(k);
      Teuchos::RCP<MAT::Material> singlemat = actmat->MaterialById(matid);

      Teuchos::RCP<MAT::ScatraMatPoroECM> scatramat =
          Teuchos::rcp_dynamic_cast<MAT::ScatraMatPoroECM>(singlemat);

      if (scatramat != Teuchos::null)
      {
        Teuchos::RCP<MAT::StructPoroReactionECM> structmat =
            Teuchos::rcp_dynamic_cast<MAT::StructPoroReactionECM>(my::ele_->Material(1));
        if (structmat == Teuchos::null) dserror("cast to MAT::StructPoroReactionECM failed!");
        double structpot = ComputeStructChemPotential(structmat, iquad);

        scatramat->ComputeReacCoeff(structpot);
      }
    }
  }

  // call base class
  advreac::GetMaterialParams(ele, densn, densnp, densam, visc, iquad);

  return;
}

/*----------------------------------------------------------------------*
 |  evaluate single material  (protected)                    vuong 19/15 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
double DRT::ELEMENTS::ScaTraEleCalcPoroReacECM<distype>::ComputeStructChemPotential(
    Teuchos::RCP<MAT::StructPoroReactionECM>& structmat, const int gp)
{
  // gauss point displacements
  CORE::LINALG::Matrix<nsd_, 1> dispint(false);
  dispint.Multiply(my::edispnp_, my::funct_);

  // transposed jacobian "dX/ds"
  CORE::LINALG::Matrix<nsd_, nsd_> xjm0;
  xjm0.MultiplyNT(my::deriv_, poro::xyze0_);

  // inverse of transposed jacobian "ds/dX"
  CORE::LINALG::Matrix<nsd_, nsd_> xji0(true);
  xji0.Invert(xjm0);

  // inverse of transposed jacobian "ds/dX"
  const double det0 = xjm0.Determinant();

  my::xjm_.MultiplyNT(my::deriv_, my::xyze_);
  const double det = my::xjm_.Determinant();

  // determinant of deformationgradient det F = det ( d x / d X ) = det (dx/ds) * ( det(dX/ds) )^-1
  const double J = det / det0;

  // ----------------------compute derivatives N_XYZ_ at gp w.r.t. material coordinates
  /// first derivatives of shape functions w.r.t. material coordinates
  CORE::LINALG::Matrix<nsd_, nen_> N_XYZ;
  N_XYZ.Multiply(xji0, my::deriv_);

  // -------------------------(material) deformation gradient F = d xyze_ / d XYZE = xyze_ *
  // N_XYZ_^T
  static CORE::LINALG::Matrix<nsd_, nsd_> defgrd(false);
  defgrd.MultiplyNT(my::xyze_, N_XYZ);

  // GL strain vector glstrain={E11,E22,E33,2*E12,2*E23,2*E31}
  static CORE::LINALG::Matrix<6, 1> glstrain(true);
  glstrain.Clear();
  // if (kinemtype_ == INPAR::STR::kinem_nonlinearTotLag)
  {
    // Right Cauchy-Green tensor = F^T * F
    CORE::LINALG::Matrix<nsd_, nsd_> cauchygreen;
    cauchygreen.MultiplyTN(defgrd, defgrd);
    // Green-Lagrange strains matrix E = 0.5 * (Cauchygreen - Identity)
    // GL strain vector glstrain={E11,E22,E33,2*E12,2*E23,2*E31}
    if (nsd_ == 3)
    {
      glstrain(0) = 0.5 * (cauchygreen(0, 0) - 1.0);
      glstrain(1) = 0.5 * (cauchygreen(1, 1) - 1.0);
      glstrain(2) = 0.5 * (cauchygreen(2, 2) - 1.0);
      glstrain(3) = cauchygreen(0, 1);
      glstrain(4) = cauchygreen(1, 2);
      glstrain(5) = cauchygreen(2, 0);
    }
    else if (nsd_ == 2)
    {
      glstrain(0) = 0.5 * (cauchygreen(0, 0) - 1.0);
      glstrain(1) = 0.5 * (cauchygreen(1, 1) - 1.0);
      glstrain(2) = 0.0;
      glstrain(3) = cauchygreen(0, 1);
      glstrain(4) = 0.0;
      glstrain(5) = 0.0;
    }
  }

  // fluid pressure at gauss point
  const double pres = my::eprenp_.Dot(my::funct_);

  double pot = 0.0;

  structmat->ChemPotential(
      glstrain, poro::DiffManager()->GetPorosity(0), pres, J, my::eid_, pot, gp);

  return pot;
}


// template classes

// 1D elements
template class DRT::ELEMENTS::ScaTraEleCalcPoroReacECM<DRT::Element::line2>;
template class DRT::ELEMENTS::ScaTraEleCalcPoroReacECM<DRT::Element::line3>;

// 2D elements
template class DRT::ELEMENTS::ScaTraEleCalcPoroReacECM<DRT::Element::tri3>;
template class DRT::ELEMENTS::ScaTraEleCalcPoroReacECM<DRT::Element::tri6>;
template class DRT::ELEMENTS::ScaTraEleCalcPoroReacECM<DRT::Element::quad4>;
// template class DRT::ELEMENTS::ScaTraEleCalcPoroReacECM<DRT::Element::quad8>;
template class DRT::ELEMENTS::ScaTraEleCalcPoroReacECM<DRT::Element::quad9>;

// 3D elements
template class DRT::ELEMENTS::ScaTraEleCalcPoroReacECM<DRT::Element::hex8>;
// template class DRT::ELEMENTS::ScaTraEleCalcPoroReacECM<DRT::Element::hex20>;
template class DRT::ELEMENTS::ScaTraEleCalcPoroReacECM<DRT::Element::hex27>;
template class DRT::ELEMENTS::ScaTraEleCalcPoroReacECM<DRT::Element::tet4>;
template class DRT::ELEMENTS::ScaTraEleCalcPoroReacECM<DRT::Element::tet10>;
// template class DRT::ELEMENTS::ScaTraEleCalcPoroReacECM<DRT::Element::wedge6>;
template class DRT::ELEMENTS::ScaTraEleCalcPoroReacECM<DRT::Element::pyramid5>;
template class DRT::ELEMENTS::ScaTraEleCalcPoroReacECM<DRT::Element::nurbs9>;
// template class DRT::ELEMENTS::ScaTraEleCalcPoroReacECM<DRT::Element::nurbs27>;

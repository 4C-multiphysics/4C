// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_ele_calc_poro_reac_ECM.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_tensor_conversion.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_mat_list_reactions.hpp"
#include "4C_mat_scatra.hpp"
#include "4C_mat_scatra_poro_ecm.hpp"
#include "4C_mat_structporo.hpp"
#include "4C_mat_structporo_reaction_ecm.hpp"
#include "4C_scatra_ele_parameter_std.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_singleton_owner.hpp"

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
Discret::Elements::ScaTraEleCalcPoroReacECM<distype>::ScaTraEleCalcPoroReacECM(
    const int numdofpernode, const int numscal, const std::string& disname)
    : Discret::Elements::ScaTraEleCalc<distype>::ScaTraEleCalc(numdofpernode, numscal, disname),
      Discret::Elements::ScaTraEleCalcPoro<distype>::ScaTraEleCalcPoro(
          numdofpernode, numscal, disname),
      Discret::Elements::ScaTraEleCalcAdvReac<distype>::ScaTraEleCalcAdvReac(
          numdofpernode, numscal, disname),
      Discret::Elements::ScaTraEleCalcPoroReac<distype>::ScaTraEleCalcPoroReac(
          numdofpernode, numscal, disname)
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
Discret::Elements::ScaTraEleCalcPoroReacECM<distype>*
Discret::Elements::ScaTraEleCalcPoroReacECM<distype>::instance(
    const int numdofpernode, const int numscal, const std::string& disname)
{
  static auto singleton_map = Core::Utils::make_singleton_map<std::string>(
      [](const int numdofpernode, const int numscal, const std::string& disname)
      {
        return std::unique_ptr<ScaTraEleCalcPoroReacECM<distype>>(
            new ScaTraEleCalcPoroReacECM<distype>(numdofpernode, numscal, disname));
      });

  return singleton_map[disname].instance(
      Core::Utils::SingletonAction::create, numdofpernode, numscal, disname);
}


/*----------------------------------------------------------------------*
 |  evaluate single material  (protected)                    thon 02/14 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcPoroReacECM<distype>::materials(
    const std::shared_ptr<const Core::Mat::Material> material,  //!< pointer to current material
    const int k,                                                //!< id of current scalar
    double& densn,                                              //!< density at t_(n)
    double& densnp,  //!< density at t_(n+1) or t_(n+alpha_F)
    double& densam,  //!< density at t_(n+alpha_M)
    double& visc,    //!< fluid viscosity
    const int iquad  //!< id of current gauss point
)
{
  switch (material->material_type())
  {
    case Core::Materials::m_scatra:
      pororeac::mat_scatra(material, k, densn, densnp, densam, visc, iquad);
      break;
    default:
      FOUR_C_THROW("Material type {} is not supported", material->material_type());
      break;
  }
}

/*----------------------------------------------------------------------*
 |  evaluate single material  (protected)                    vuong 10/14 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcPoroReacECM<distype>::get_material_params(
    const Core::Elements::Element* ele,  //!< the element we are dealing with
    std::vector<double>& densn,          //!< density at t_(n)
    std::vector<double>& densnp,         //!< density at t_(n+1) or t_(n+alpha_F)
    std::vector<double>& densam,         //!< density at t_(n+alpha_M)
    double& visc,                        //!< fluid viscosity
    const int iquad                      //!< id of current gauss point
)
{
  // call poro base class to compute porosity
  poro::compute_porosity(ele);

  // get the material
  std::shared_ptr<Core::Mat::Material> material = ele->material();

  if (material->material_type() == Core::Materials::m_matlist_reactions)
  {
    const std::shared_ptr<Mat::MatListReactions> actmat =
        std::dynamic_pointer_cast<Mat::MatListReactions>(material);
    if (actmat->num_mat() != my::numscal_) FOUR_C_THROW("Not enough materials in MatList.");

    for (int k = 0; k < actmat->num_reac(); ++k)
    {
      int matid = actmat->reac_id(k);
      std::shared_ptr<Core::Mat::Material> singlemat = actmat->material_by_id(matid);

      std::shared_ptr<Mat::ScatraMatPoroECM> scatramat =
          std::dynamic_pointer_cast<Mat::ScatraMatPoroECM>(singlemat);

      if (scatramat != nullptr)
      {
        std::shared_ptr<Mat::StructPoroReactionECM> structmat =
            std::dynamic_pointer_cast<Mat::StructPoroReactionECM>(my::ele_->material(1));
        if (structmat == nullptr) FOUR_C_THROW("cast to Mat::StructPoroReactionECM failed!");
        double structpot = compute_struct_chem_potential(*structmat, iquad);

        scatramat->compute_reac_coeff(structpot);
      }
    }
  }

  // call base class
  advreac::get_material_params(ele, densn, densnp, densam, visc, iquad);
}

/*----------------------------------------------------------------------*
 |  evaluate single material  (protected)                    vuong 19/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
double Discret::Elements::ScaTraEleCalcPoroReacECM<distype>::compute_struct_chem_potential(
    Mat::StructPoroReactionECM& structmat, const int gp)
{
  // gauss point displacements
  Core::LinAlg::Matrix<nsd_, 1> dispint(Core::LinAlg::Initialization::uninitialized);
  dispint.multiply(my::edispnp_, my::funct_);

  // transposed jacobian "dX/ds"
  Core::LinAlg::Matrix<nsd_, nsd_> xjm0;
  xjm0.multiply_nt(my::deriv_, poro::xyze0_);

  // inverse of transposed jacobian "ds/dX"
  Core::LinAlg::Matrix<nsd_, nsd_> xji0(Core::LinAlg::Initialization::zero);
  xji0.invert(xjm0);

  // inverse of transposed jacobian "ds/dX"
  const double det0 = xjm0.determinant();

  my::xjm_.multiply_nt(my::deriv_, my::xyze_);
  const double det = my::xjm_.determinant();

  // determinant of deformationgradient det F = det ( d x / d X ) = det (dx/ds) * ( det(dX/ds) )^-1
  const double J = det / det0;

  // ----------------------compute derivatives N_XYZ_ at gp w.r.t. material coordinates
  /// first derivatives of shape functions w.r.t. material coordinates
  Core::LinAlg::Matrix<nsd_, nen_> N_XYZ;
  N_XYZ.multiply(xji0, my::deriv_);

  // -------------------------(material) deformation gradient F = d xyze_ / d XYZE = xyze_ *
  // N_XYZ_^T
  static Core::LinAlg::Matrix<nsd_, nsd_> defgrd(Core::LinAlg::Initialization::uninitialized);
  defgrd.multiply_nt(my::xyze_, N_XYZ);

  // GL strain vector glstrain={E11,E22,E33,2*E12,2*E23,2*E31}
  static Core::LinAlg::SymmetricTensor<double, 3, 3> glstrain;
  glstrain = {};
  // if (kinemtype_ == Inpar::Solid::KinemType::nonlinearTotLag)
  {
    // Right Cauchy-Green tensor = F^T * F
    Core::LinAlg::Matrix<nsd_, nsd_> cauchygreen;
    cauchygreen.multiply_tn(defgrd, defgrd);
    // Green-Lagrange strains matrix E = 0.5 * (Cauchygreen - Identity)
    // GL strain vector glstrain={E11,E22,E33,2*E12,2*E23,2*E31}
    if constexpr (nsd_ == 3)
    {
      glstrain = 0.5 * (Core::LinAlg::assume_symmetry(Core::LinAlg::make_tensor_view(cauchygreen)) -
                           Core::LinAlg::TensorGenerators::identity<double, 3, 3>);
    }
    else if constexpr (nsd_ == 2)
    {
      glstrain(0, 0) = 0.5 * (cauchygreen(0, 0) - 1.0);
      glstrain(1, 1) = 0.5 * (cauchygreen(1, 1) - 1.0);
      glstrain(2, 2) = 0.0;
      glstrain(0, 1) = 0.5 * cauchygreen(0, 1);
      glstrain(1, 2) = 0.0;
      glstrain(0, 2) = 0.0;
    }
  }

  // fluid pressure at gauss point
  const double pres = my::eprenp_.dot(my::funct_);

  double pot = 0.0;

  structmat.chem_potential(
      glstrain, poro::diff_manager()->get_porosity(0), pres, J, my::eid_, pot, gp);

  return pot;
}


// template classes

// 1D elements
template class Discret::Elements::ScaTraEleCalcPoroReacECM<Core::FE::CellType::line2>;
template class Discret::Elements::ScaTraEleCalcPoroReacECM<Core::FE::CellType::line3>;

// 2D elements
template class Discret::Elements::ScaTraEleCalcPoroReacECM<Core::FE::CellType::tri3>;
template class Discret::Elements::ScaTraEleCalcPoroReacECM<Core::FE::CellType::tri6>;
template class Discret::Elements::ScaTraEleCalcPoroReacECM<Core::FE::CellType::quad4>;
// template class Discret::Elements::ScaTraEleCalcPoroReacECM<Core::FE::CellType::quad8>;
template class Discret::Elements::ScaTraEleCalcPoroReacECM<Core::FE::CellType::quad9>;

// 3D elements
template class Discret::Elements::ScaTraEleCalcPoroReacECM<Core::FE::CellType::hex8>;
// template class Discret::Elements::ScaTraEleCalcPoroReacECM<Core::FE::CellType::hex20>;
template class Discret::Elements::ScaTraEleCalcPoroReacECM<Core::FE::CellType::hex27>;
template class Discret::Elements::ScaTraEleCalcPoroReacECM<Core::FE::CellType::tet4>;
template class Discret::Elements::ScaTraEleCalcPoroReacECM<Core::FE::CellType::tet10>;
// template class Discret::Elements::ScaTraEleCalcPoroReacECM<Core::FE::CellType::wedge6>;
template class Discret::Elements::ScaTraEleCalcPoroReacECM<Core::FE::CellType::pyramid5>;
template class Discret::Elements::ScaTraEleCalcPoroReacECM<Core::FE::CellType::nurbs9>;
// template class
// Discret::Elements::ScaTraEleCalcPoroReacECM<Core::FE::CellType::nurbs27>;

FOUR_C_NAMESPACE_CLOSE

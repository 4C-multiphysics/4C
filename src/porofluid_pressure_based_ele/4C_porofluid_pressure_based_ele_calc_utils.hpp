// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_POROFLUID_PRESSURE_BASED_ELE_CALC_UTILS_HPP
#define FOUR_C_POROFLUID_PRESSURE_BASED_ELE_CALC_UTILS_HPP

#include "4C_config.hpp"

#include "4C_fem_general_element.hpp"
#include "4C_fem_general_utils_integration.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  class Material;
  class FluidPoroSinglePhase;
  class FluidPoroSingleVolFrac;
  class FluidPoroVolFracPressure;
  class FluidPoroMultiPhase;
  class FluidPoroMultiPhaseReactions;
  class FluidPoroSingleReaction;
  class FluidPoroVolFracPressureBloodLung;
}  // namespace Mat

namespace PoroPressureBased
{
  namespace ElementUtils
  {
    //! get the single phase material from the element material
    const Mat::FluidPoroSinglePhase& get_single_phase_mat_from_material(
        const Core::Mat::Material& material, int phasenum);

    //! get the single phase material from the element multiphase material
    const Mat::FluidPoroSinglePhase& get_single_phase_mat_from_multi_material(
        const Mat::FluidPoroMultiPhase& multiphasemat, int phasenum);

    //! get the single volume fraction material from the element material
    const Mat::FluidPoroSingleVolFrac& get_single_vol_frac_mat_from_material(
        const Core::Mat::Material& material, int volfracnum);

    //! get the single volume fraction material from the element multiphase material
    const Mat::FluidPoroSingleVolFrac& get_single_vol_frac_mat_from_multi_material(
        const Mat::FluidPoroMultiPhase& multiphasemat, int volfracnum);

    //! get the single volume fraction pressure blood lung material from the element material
    const Mat::FluidPoroVolFracPressureBloodLung&
    get_single_vol_frac_pressure_blood_lung_mat_from_material(
        const Core::Mat::Material& material, int volfracnum);

    //! get the single volume fraction pressure blood lung material from the element multiphase
    //! material
    const Mat::FluidPoroVolFracPressureBloodLung&
    get_single_vol_frac_pressure_blood_lung_mat_from_multi_material(
        const Mat::FluidPoroMultiPhase& multiphasemat, int volfracnum);

    //! get the volume fraction pressure material from the element material
    const Mat::FluidPoroVolFracPressure& get_vol_frac_pressure_mat_from_material(
        const Core::Mat::Material& material, int volfracnum);

    //! get the volume fraction pressure material from the element multiphase material
    const Mat::FluidPoroVolFracPressure& get_vol_frac_pressure_mat_from_multi_material(
        const Mat::FluidPoroMultiPhase& multiphasemat, int volfracnum);

    //! get the single phase material from the element multiphase reactions material
    Mat::FluidPoroSingleReaction& get_single_reaction_mat_from_multi_reactions_material(
        const Mat::FluidPoroMultiPhaseReactions& multiphasereacmat, int phasenum);

    /*!
    \brief Decide, whether second derivatives are needed  (template version)
     *  In convection-diffusion problems, ONLY N,xx , N,yy and N,zz are needed
     *  to evaluate the laplacian operator for the residual-based stabilization.
     *  Hence, unlike to the Navier-Stokes equations, hex8, wedge6 and pyramid5
     *  return false although they have non-zero MIXED second derivatives.*/
    template <Core::FE::CellType distype>
    struct Use2ndDerivs
    {
    };
    template <>
    struct Use2ndDerivs<Core::FE::CellType::hex8>
    {
      static constexpr bool use = false;
    };
    template <>
    struct Use2ndDerivs<Core::FE::CellType::tet4>
    {
      static constexpr bool use = false;
    };
    template <>
    struct Use2ndDerivs<Core::FE::CellType::wedge6>
    {
      static constexpr bool use = false;
    };
    template <>
    struct Use2ndDerivs<Core::FE::CellType::pyramid5>
    {
      static constexpr bool use = false;
    };
    template <>
    struct Use2ndDerivs<Core::FE::CellType::nurbs8>
    {
      static constexpr bool use = false;
    };
    template <>
    struct Use2ndDerivs<Core::FE::CellType::quad4>
    {
      static constexpr bool use = false;
    };
    template <>
    struct Use2ndDerivs<Core::FE::CellType::nurbs4>
    {
      static constexpr bool use = false;
    };
    template <>
    struct Use2ndDerivs<Core::FE::CellType::tri3>
    {
      static constexpr bool use = false;
    };
    template <>
    struct Use2ndDerivs<Core::FE::CellType::line2>
    {
      static constexpr bool use = false;
    };
    template <>
    struct Use2ndDerivs<Core::FE::CellType::nurbs2>
    {
      static constexpr bool use = false;
    };
    template <>
    struct Use2ndDerivs<Core::FE::CellType::hex20>
    {
      static constexpr bool use = true;
    };
    template <>
    struct Use2ndDerivs<Core::FE::CellType::hex27>
    {
      static constexpr bool use = true;
    };
    template <>
    struct Use2ndDerivs<Core::FE::CellType::nurbs27>
    {
      static constexpr bool use = true;
    };
    template <>
    struct Use2ndDerivs<Core::FE::CellType::tet10>
    {
      static constexpr bool use = true;
    };
    template <>
    struct Use2ndDerivs<Core::FE::CellType::quad8>
    {
      static constexpr bool use = true;
    };
    template <>
    struct Use2ndDerivs<Core::FE::CellType::quad9>
    {
      static constexpr bool use = true;
    };
    template <>
    struct Use2ndDerivs<Core::FE::CellType::nurbs9>
    {
      static constexpr bool use = true;
    };
    template <>
    struct Use2ndDerivs<Core::FE::CellType::tri6>
    {
      static constexpr bool use = true;
    };
    template <>
    struct Use2ndDerivs<Core::FE::CellType::line3>
    {
      static constexpr bool use = true;
    };
    template <>
    struct Use2ndDerivs<Core::FE::CellType::nurbs3>
    {
      static constexpr bool use = true;
    };


    //! Template Meta Programming version of switch over discretization type
    template <Core::FE::CellType distype>
    struct DisTypeToOptGaussRule
    {
    };
    template <>
    struct DisTypeToOptGaussRule<Core::FE::CellType::hex8>
    {
      static constexpr Core::FE::GaussRule3D rule = Core::FE::GaussRule3D::hex_8point;
    };
    template <>
    struct DisTypeToOptGaussRule<Core::FE::CellType::hex20>
    {
      static constexpr Core::FE::GaussRule3D rule = Core::FE::GaussRule3D::hex_27point;
    };
    template <>
    struct DisTypeToOptGaussRule<Core::FE::CellType::hex27>
    {
      static constexpr Core::FE::GaussRule3D rule = Core::FE::GaussRule3D::hex_27point;
    };
    template <>
    struct DisTypeToOptGaussRule<Core::FE::CellType::tet4>
    {
      static constexpr Core::FE::GaussRule3D rule = Core::FE::GaussRule3D::tet_4point;
    };
    template <>
    struct DisTypeToOptGaussRule<Core::FE::CellType::tet10>
    {
      static constexpr Core::FE::GaussRule3D rule = Core::FE::GaussRule3D::tet_5point;
    };
    template <>
    struct DisTypeToOptGaussRule<Core::FE::CellType::wedge6>
    {
      static constexpr Core::FE::GaussRule3D rule = Core::FE::GaussRule3D::wedge_6point;
    };
    template <>
    struct DisTypeToOptGaussRule<Core::FE::CellType::pyramid5>
    {
      static constexpr Core::FE::GaussRule3D rule = Core::FE::GaussRule3D::pyramid_8point;
    };
    template <>
    struct DisTypeToOptGaussRule<Core::FE::CellType::nurbs8>
    {
      static constexpr Core::FE::GaussRule3D rule = Core::FE::GaussRule3D::hex_8point;
    };
    template <>
    struct DisTypeToOptGaussRule<Core::FE::CellType::nurbs27>
    {
      static constexpr Core::FE::GaussRule3D rule = Core::FE::GaussRule3D::hex_27point;
    };
    template <>
    struct DisTypeToOptGaussRule<Core::FE::CellType::quad4>
    {
      static constexpr Core::FE::GaussRule2D rule = Core::FE::GaussRule2D::quad_4point;
    };
    template <>
    struct DisTypeToOptGaussRule<Core::FE::CellType::quad8>
    {
      static constexpr Core::FE::GaussRule2D rule = Core::FE::GaussRule2D::quad_9point;
    };
    template <>
    struct DisTypeToOptGaussRule<Core::FE::CellType::quad9>
    {
      static constexpr Core::FE::GaussRule2D rule = Core::FE::GaussRule2D::quad_9point;
    };
    template <>
    struct DisTypeToOptGaussRule<Core::FE::CellType::tri3>
    {
      static constexpr Core::FE::GaussRule2D rule = Core::FE::GaussRule2D::tri_3point;
    };
    template <>
    struct DisTypeToOptGaussRule<Core::FE::CellType::tri6>
    {
      static constexpr Core::FE::GaussRule2D rule = Core::FE::GaussRule2D::tri_6point;
    };
    template <>
    struct DisTypeToOptGaussRule<Core::FE::CellType::nurbs4>
    {
      static constexpr Core::FE::GaussRule2D rule = Core::FE::GaussRule2D::quad_4point;
    };
    template <>
    struct DisTypeToOptGaussRule<Core::FE::CellType::nurbs9>
    {
      static constexpr Core::FE::GaussRule2D rule = Core::FE::GaussRule2D::quad_9point;
    };
    template <>
    struct DisTypeToOptGaussRule<Core::FE::CellType::line2>
    {
      static constexpr Core::FE::GaussRule1D rule = Core::FE::GaussRule1D::line_2point;
    };
    template <>
    struct DisTypeToOptGaussRule<Core::FE::CellType::line3>
    {
      static constexpr Core::FE::GaussRule1D rule = Core::FE::GaussRule1D::line_3point;
    };
    template <>
    struct DisTypeToOptGaussRule<Core::FE::CellType::nurbs2>
    {
      static constexpr Core::FE::GaussRule1D rule = Core::FE::GaussRule1D::line_2point;
    };
    template <>
    struct DisTypeToOptGaussRule<Core::FE::CellType::nurbs3>
    {
      static constexpr Core::FE::GaussRule1D rule = Core::FE::GaussRule1D::line_3point;
    };


    //! Template Meta Programming version of switch over discretization type
    template <Core::FE::CellType distype>
    struct DisTypeToGaussRuleForExactSol
    {
    };
    template <>
    struct DisTypeToGaussRuleForExactSol<Core::FE::CellType::hex8>
    {
      static constexpr Core::FE::GaussRule3D rule = Core::FE::GaussRule3D::hex_27point;
    };
    template <>
    struct DisTypeToGaussRuleForExactSol<Core::FE::CellType::hex20>
    {
      static constexpr Core::FE::GaussRule3D rule = Core::FE::GaussRule3D::hex_27point;
    };
    template <>
    struct DisTypeToGaussRuleForExactSol<Core::FE::CellType::hex27>
    {
      static constexpr Core::FE::GaussRule3D rule = Core::FE::GaussRule3D::hex_27point;
    };
    template <>
    struct DisTypeToGaussRuleForExactSol<Core::FE::CellType::tet4>
    {
      static constexpr Core::FE::GaussRule3D rule = Core::FE::GaussRule3D::undefined;
    };
    template <>
    struct DisTypeToGaussRuleForExactSol<Core::FE::CellType::tet10>
    {
      static constexpr Core::FE::GaussRule3D rule = Core::FE::GaussRule3D::undefined;
    };
    template <>
    struct DisTypeToGaussRuleForExactSol<Core::FE::CellType::wedge6>
    {
      static constexpr Core::FE::GaussRule3D rule = Core::FE::GaussRule3D::undefined;
    };
    template <>
    struct DisTypeToGaussRuleForExactSol<Core::FE::CellType::pyramid5>
    {
      static constexpr Core::FE::GaussRule3D rule = Core::FE::GaussRule3D::undefined;
    };
    template <>
    struct DisTypeToGaussRuleForExactSol<Core::FE::CellType::nurbs8>
    {
      static constexpr Core::FE::GaussRule3D rule = Core::FE::GaussRule3D::hex_27point;
    };
    template <>
    struct DisTypeToGaussRuleForExactSol<Core::FE::CellType::nurbs27>
    {
      static constexpr Core::FE::GaussRule3D rule = Core::FE::GaussRule3D::hex_27point;
    };
    template <>
    struct DisTypeToGaussRuleForExactSol<Core::FE::CellType::quad4>
    {
      static constexpr Core::FE::GaussRule2D rule = Core::FE::GaussRule2D::quad_9point;
    };
    template <>
    struct DisTypeToGaussRuleForExactSol<Core::FE::CellType::quad8>
    {
      static constexpr Core::FE::GaussRule2D rule = Core::FE::GaussRule2D::quad_9point;
    };
    template <>
    struct DisTypeToGaussRuleForExactSol<Core::FE::CellType::quad9>
    {
      static constexpr Core::FE::GaussRule2D rule = Core::FE::GaussRule2D::quad_9point;
    };
    template <>
    struct DisTypeToGaussRuleForExactSol<Core::FE::CellType::tri3>
    {
      static constexpr Core::FE::GaussRule2D rule = Core::FE::GaussRule2D::undefined;
    };
    template <>
    struct DisTypeToGaussRuleForExactSol<Core::FE::CellType::tri6>
    {
      static constexpr Core::FE::GaussRule2D rule = Core::FE::GaussRule2D::undefined;
    };
    template <>
    struct DisTypeToGaussRuleForExactSol<Core::FE::CellType::nurbs4>
    {
      static constexpr Core::FE::GaussRule2D rule = Core::FE::GaussRule2D::quad_4point;
    };
    template <>
    struct DisTypeToGaussRuleForExactSol<Core::FE::CellType::nurbs9>
    {
      static constexpr Core::FE::GaussRule2D rule = Core::FE::GaussRule2D::quad_9point;
    };
    template <>
    struct DisTypeToGaussRuleForExactSol<Core::FE::CellType::line2>
    {
      static constexpr Core::FE::GaussRule1D rule = Core::FE::GaussRule1D::line_2point;
    };  // not tested
    template <>
    struct DisTypeToGaussRuleForExactSol<Core::FE::CellType::line3>
    {
      static constexpr Core::FE::GaussRule1D rule = Core::FE::GaussRule1D::undefined;
    };
    template <>
    struct DisTypeToGaussRuleForExactSol<Core::FE::CellType::nurbs2>
    {
      static constexpr Core::FE::GaussRule1D rule = Core::FE::GaussRule1D::undefined;
    };
    template <>
    struct DisTypeToGaussRuleForExactSol<Core::FE::CellType::nurbs3>
    {
      static constexpr Core::FE::GaussRule1D rule = Core::FE::GaussRule1D::undefined;
    };

  }  // namespace ElementUtils

}  // namespace PoroPressureBased



FOUR_C_NAMESPACE_CLOSE

#endif

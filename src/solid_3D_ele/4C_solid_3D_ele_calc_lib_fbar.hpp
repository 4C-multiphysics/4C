// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SOLID_3D_ELE_CALC_LIB_FBAR_HPP
#define FOUR_C_SOLID_3D_ELE_CALC_LIB_FBAR_HPP

#include "4C_config.hpp"

#include "4C_fem_general_cell_type.hpp"
#include "4C_fem_general_cell_type_traits.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_solid_3D_ele_calc_lib.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Discret::Elements
{
  /*!
   * @brief A small container storing information needed to compute the linearization of an element
   * with FBAR
   */
  template <Core::FE::CellType celltype>
  struct FBarLinearizationContainer
  {
    Core::LinAlg::Matrix<Core::FE::dim<celltype>*(Core::FE::dim<celltype> + 1) / 2,
        Core::FE::num_nodes<celltype> * Core::FE::dim<celltype>>
        Bop{};

    Core::LinAlg::Matrix<Core::FE::num_nodes<celltype> * Core::FE::dim<celltype>, 1> Hop{};

    Core::LinAlg::Matrix<Core::FE::dim<celltype>, Core::FE::dim<celltype>> cauchygreen{};

    double fbar_factor = 1.0;
  };

  /*!
   * @brief Evaluate the fbar factor \f[ \frac{\mathbf{F}_{\mathrm{centroid}}}{\mathbf{F}}^{1/3}
   * \f]
   *
   * @param defgrd_centroid (in) : Deformation gradient evaluated at the element centroid
   * @param defgrd_gp (in) : Deformation gradient evaluated at the Gauss point
   * @return double : Fbar factor
   */
  inline double evaluate_fbar_factor(const double& defgrd_centroid, const double& defgrd_gp)
  {
    const double fbar_factor = std::pow(defgrd_centroid / defgrd_gp, 1.0 / 3.0);
    return fbar_factor;
  }

  /*!
   * @brief Evaluates the H-Operator used in F-bar of the specified element
   *
   * @tparam celltype : Cell type
   * @param shape_function_derivs (in) : Derivative of the shape functions w.r.t. XYZ at the Gauss
   * point
   * @param shape_function_derivs_centroid (in) : Derivative of the shape functions w.r.t. XYZ at
   * the element center
   * @param spatial_material_mapping (in) :An object holding quantities of the spatial material
   * mapping (deformation_gradient, inverse_deformation_gradient,
   * determinant_deformation_gradient) evaluated at the Gauss point
   * @param spatial_material_mapping_centroid (in) : An object holding quantities of the spatial
   * material mapping (deformation_gradient, inverse_deformation_gradient,
   * determinant_deformation_gradient) evaluated at the element centroid
   * @return Core::LinAlg::Matrix<num_dof_per_ele, 1> : H-Operator
   */
  template <Core::FE::CellType celltype>
  inline Core::LinAlg::Matrix<Core::FE::dim<celltype> * Core::FE::num_nodes<celltype>, 1>
  evaluate_fbar_h_operator(
      const Core::LinAlg::Matrix<Core::FE::dim<celltype>, Core::FE::num_nodes<celltype>>&
          shape_function_derivs,
      const Core::LinAlg::Matrix<Core::FE::dim<celltype>, Core::FE::num_nodes<celltype>>&
          shape_function_derivs_centroid,
      const Discret::Elements::SpatialMaterialMapping<celltype>& spatial_material_mapping,
      const Discret::Elements::SpatialMaterialMapping<celltype>& spatial_material_mapping_centroid)
    requires(Core::FE::dim<celltype> == 3)
  {
    // inverse deformation gradient at centroid
    Core::LinAlg::Matrix<Core::FE::dim<celltype>, Core::FE::dim<celltype>> invdefgrd_centroid;
    invdefgrd_centroid.invert(spatial_material_mapping_centroid.deformation_gradient_);

    // inverse deformation gradient at gp
    Core::LinAlg::Matrix<Core::FE::dim<celltype>, Core::FE::dim<celltype>> invdefgrd;
    invdefgrd.invert(spatial_material_mapping.deformation_gradient_);

    Core::LinAlg::Matrix<Core::FE::dim<celltype> * Core::FE::num_nodes<celltype>, 1> Hop(
        Core::LinAlg::Initialization::zero);
    for (int idof = 0; idof < Core::FE::dim<celltype> * Core::FE::num_nodes<celltype>; idof++)
    {
      for (int idim = 0; idim < Core::FE::dim<celltype>; idim++)
      {
        Hop(idof) += invdefgrd_centroid(idim, idof % Core::FE::dim<celltype>) *
                     shape_function_derivs_centroid(idim, idof / Core::FE::dim<celltype>);
        Hop(idof) -= invdefgrd(idim, idof % Core::FE::dim<celltype>) *
                     shape_function_derivs(idim, idof / Core::FE::dim<celltype>);
      }
    }

    return Hop;
  }

  /*!
   * @brief Add fbar stiffness matrix contribution of one Gauss point
   *
   * @tparam celltype : Cell type
   * @param Bop (in) : Strain gradient (B-Operator)
   * @param Hop (in) : H-Operator
   * @param f_bar_factor (in) : f_bar_factor
   * @param integration_fac (in) : Integration factor (Gauss point weight times the determinant of
   * the jacobian)
   * @param cauchyGreen (in) : An object holding the right Cauchy-Green deformation tensor and
   * its inverse
   * @param stress_bar (in) : Deviatoric part of stress measures
   * @param stiffness_matrix (in/out) : stiffness matrix where the local contribution is added to
   */
  template <Core::FE::CellType celltype>
  inline void add_fbar_stiffness_matrix(
      const Core::LinAlg::Matrix<Core::FE::dim<celltype>*(Core::FE::dim<celltype> + 1) / 2,
          Core::FE::dim<celltype> * Core::FE::num_nodes<celltype>>& Bop,
      const Core::LinAlg::Matrix<Core::FE::dim<celltype> * Core::FE::num_nodes<celltype>, 1>& Hop,
      const double f_bar_factor, const double integration_fac,
      const Core::LinAlg::Matrix<Core::FE::dim<celltype>, Core::FE::dim<celltype>> cauchyGreen,
      const Discret::Elements::Stress<celltype> stress_bar,
      Core::LinAlg::Matrix<Core::FE::dim<celltype> * Core::FE::num_nodes<celltype>,
          Core::FE::dim<celltype> * Core::FE::num_nodes<celltype>>& stiffness_matrix)
  {
    constexpr int num_dof_per_ele = Core::FE::dim<celltype> * Core::FE::num_nodes<celltype>;

    Core::LinAlg::Matrix<Core::FE::dim<celltype>*(Core::FE::dim<celltype> + 1) / 2, 1>
        rcg_bar_voigt;
    Core::LinAlg::Voigt::Strains::matrix_to_vector(cauchyGreen, rcg_bar_voigt);

    Core::LinAlg::Matrix<Core::FE::dim<celltype>*(Core::FE::dim<celltype> + 1) / 2, 1> ccg;
    ccg.multiply_nn(stress_bar.cmat_, rcg_bar_voigt);

    // auxiliary integrated stress_bar
    Core::LinAlg::Matrix<num_dof_per_ele, 1> bopccg(Core::LinAlg::Initialization::uninitialized);
    bopccg.multiply_tn(integration_fac * f_bar_factor / 3.0, Bop, ccg);

    Core::LinAlg::Matrix<num_dof_per_ele, 1> bops(Core::LinAlg::Initialization::uninitialized);
    bops.multiply_tn(-integration_fac / f_bar_factor / 3.0, Bop, stress_bar.pk2_);

    for (int idof = 0; idof < num_dof_per_ele; idof++)
    {
      for (int jdof = 0; jdof < num_dof_per_ele; jdof++)
      {
        stiffness_matrix(idof, jdof) += Hop(jdof) * (bops(idof, 0) + bopccg(idof, 0));
      }
    }
  }
}  // namespace Discret::Elements

FOUR_C_NAMESPACE_CLOSE
#endif
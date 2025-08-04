// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SOLID_3D_ELE_CALC_FBAR_HPP
#define FOUR_C_SOLID_3D_ELE_CALC_FBAR_HPP

#include "4C_config.hpp"

#include "4C_fem_general_cell_type_traits.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_solid_3D_ele_calc.hpp"
#include "4C_solid_3D_ele_calc_lib.hpp"
#include "4C_solid_3D_ele_calc_lib_fbar.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Discret::Elements
{
  template <Core::FE::CellType celltype>
  struct FBarPreparationData
  {
    /// jacobian mapping evaluated at element centroid
    JacobianMapping<celltype> jacobian_mapping_centroid;

    /// deformation gradient at element centroid
    SpatialMaterialMapping<celltype> spatial_material_mapping_centroid;
  };

  struct FBarHistoryData
  {
    // no history data needed
  };

  /*!
   * @brief A displacement based solid element formulation with FBAR element technology
   *
   * @tparam celltype
   */
  template <Core::FE::CellType celltype>
  struct FBarFormulation
  {
    static constexpr bool has_gauss_point_history = false;
    static constexpr bool has_global_history = false;
    static constexpr bool has_preparation_data = true;
    static constexpr bool has_condensed_contribution = false;

    using LinearizationContainer = FBarLinearizationContainer<celltype>;
    using PreparationData = FBarPreparationData<celltype>;


    static FBarPreparationData<celltype> prepare(
        const Core::Elements::Element& ele, const ElementNodes<celltype>& nodal_coordinates)
    {
      const JacobianMapping<celltype> jacobian_mapping_centroid =
          evaluate_jacobian_mapping_centroid(nodal_coordinates);

      return {jacobian_mapping_centroid,
          evaluate_spatial_material_mapping(jacobian_mapping_centroid, nodal_coordinates)};
    }

    template <typename Evaluator>
    static auto evaluate(const Core::Elements::Element& ele,
        const ElementNodes<celltype>& nodal_coordinates,
        const Core::LinAlg::Tensor<double, Core::FE::dim<celltype>>& xi,
        const ShapeFunctionsAndDerivatives<celltype>& shape_functions,
        const JacobianMapping<celltype>& jacobian_mapping,
        const FBarPreparationData<celltype>& preparation_data, Evaluator evaluator)
    {
      const SpatialMaterialMapping<celltype> spatial_material_mapping =
          evaluate_spatial_material_mapping(jacobian_mapping, nodal_coordinates);

      // factor (detF0/detF)^1/3
      const double fbar_factor = evaluate_fbar_factor(
          preparation_data.spatial_material_mapping_centroid.determinant_deformation_gradient_,
          spatial_material_mapping.determinant_deformation_gradient_);

      const FBarLinearizationContainer<celltype> linearization = std::invoke(
          [&]()
          {
            FBarLinearizationContainer<celltype> linearization{};
            linearization.Bop =
                evaluate_strain_gradient(jacobian_mapping, spatial_material_mapping);

            linearization.Hop = evaluate_fbar_h_operator(jacobian_mapping.N_XYZ,
                preparation_data.jacobian_mapping_centroid.N_XYZ, spatial_material_mapping,
                preparation_data.spatial_material_mapping_centroid);

            linearization.fbar_factor = fbar_factor;

            linearization.cauchygreen = evaluate_cauchy_green(spatial_material_mapping);

            return linearization;
          });

      // deformation gradient F_bar and resulting strains: F_bar = (detF_0/detF)^1/3 F
      const SpatialMaterialMapping<celltype> spatial_material_mapping_bar =
          evaluate_spatial_material_mapping(jacobian_mapping, nodal_coordinates, fbar_factor);

      const Core::LinAlg::SymmetricTensor<double, Core::FE::dim<celltype>, Core::FE::dim<celltype>>
          cauchygreen_bar = evaluate_cauchy_green(spatial_material_mapping_bar);

      Core::LinAlg::SymmetricTensor<double, Core::FE::dim<celltype>, Core::FE::dim<celltype>>
          gl_strain_bar = evaluate_green_lagrange_strain(cauchygreen_bar);

      return evaluator(
          spatial_material_mapping_bar.deformation_gradient_, gl_strain_bar, linearization);
    }

    static inline Core::LinAlg::Matrix<9, Core::FE::num_nodes(celltype) * Core::FE::dim<celltype>>
    evaluate_d_deformation_gradient_d_displacements(const Core::Elements::Element& ele,
        const ElementNodes<celltype>& element_nodes,
        const Core::LinAlg::Tensor<double, Core::FE::dim<celltype>>& xi,
        const ShapeFunctionsAndDerivatives<celltype>& shape_functions,
        const JacobianMapping<celltype>& jacobian_mapping,
        const Core::LinAlg::Tensor<double, Internal::num_dim<celltype>,
            Internal::num_dim<celltype>>& deformation_gradient,
        const FBarPreparationData<celltype>& preparation_data)
    {
      FOUR_C_THROW(
          "This derivative of the deformation gradient w.r.t. the displacements is not "
          "implemented");
    }

    static inline Core::LinAlg::Matrix<9, Core::FE::dim<celltype>>
    evaluate_d_deformation_gradient_d_xi(const Core::Elements::Element& ele,
        const ElementNodes<celltype>& element_nodes,
        const Core::LinAlg::Tensor<double, Core::FE::dim<celltype>>& xi,
        const ShapeFunctionsAndDerivatives<celltype>& shape_functions,
        const JacobianMapping<celltype>& jacobian_mapping,
        const Core::LinAlg::Tensor<double, Internal::num_dim<celltype>,
            Internal::num_dim<celltype>>& deformation_gradient,
        const FBarPreparationData<celltype>& preparation_data)
    {
      FOUR_C_THROW("This derivative of the deformation gradient w.r.t. xi is not implemented");
    }

    static inline Core::LinAlg::Matrix<9,
        Core::FE::num_nodes(celltype) * Core::FE::dim<celltype> * Core::FE::dim<celltype>>
    evaluate_d_deformation_gradient_d_displacements_d_xi(const Core::Elements::Element& ele,
        const ElementNodes<celltype>& element_nodes,
        const Core::LinAlg::Tensor<double, Core::FE::dim<celltype>>& xi,
        const ShapeFunctionsAndDerivatives<celltype>& shape_functions,
        const JacobianMapping<celltype>& jacobian_mapping,
        const Core::LinAlg::Tensor<double, Internal::num_dim<celltype>,
            Internal::num_dim<celltype>>& deformation_gradient,
        const FBarPreparationData<celltype>& preparation_data)
    {
      FOUR_C_THROW(
          "This second derivative of the deformation gradient w.r.t. the displacements and xi is "
          "not implemented");
    }

    static void add_internal_force_vector(const JacobianMapping<celltype>& jacobian_mapping,
        const Core::LinAlg::Tensor<double, Core::FE::dim<celltype>, Core::FE::dim<celltype>>& F,
        const FBarLinearizationContainer<celltype>& linearization, const Stress<celltype>& stress,
        const double integration_factor, const FBarPreparationData<celltype>& preparation_data,
        Core::LinAlg::Matrix<Core::FE::num_nodes(celltype) * Core::FE::dim<celltype>, 1>&
            force_vector)
    {
      Discret::Elements::add_internal_force_vector(
          linearization.Bop, stress, integration_factor / linearization.fbar_factor, force_vector);
    }

    static void add_stiffness_matrix(const JacobianMapping<celltype>& jacobian_mapping,
        const Core::LinAlg::Tensor<double, Core::FE::dim<celltype>, Core::FE::dim<celltype>>& F,
        const Core::LinAlg::Tensor<double, Core::FE::dim<celltype>>& xi,
        const ShapeFunctionsAndDerivatives<celltype>& shape_functions,
        const FBarLinearizationContainer<celltype>& linearization, const Stress<celltype>& stress,
        const double integration_factor, const FBarPreparationData<celltype>& preparation_data,
        Core::LinAlg::Matrix<Core::FE::num_nodes(celltype) * Core::FE::dim<celltype>,
            Core::FE::num_nodes(celltype) * Core::FE::dim<celltype>>& stiffness_matrix)
    {
      Discret::Elements::add_elastic_stiffness_matrix(linearization.Bop, stress,
          integration_factor * linearization.fbar_factor, stiffness_matrix);
      Discret::Elements::add_geometric_stiffness_matrix(jacobian_mapping, stress,
          integration_factor / linearization.fbar_factor, stiffness_matrix);

      // additional stiffness matrix needed for fbar method
      add_fbar_stiffness_matrix(linearization.Bop, linearization.Hop, linearization.fbar_factor,
          integration_factor, linearization.cauchygreen, stress, stiffness_matrix);
    }
  };

  template <Core::FE::CellType celltype>
  using FBarSolidIntegrator = SolidEleCalc<celltype, FBarFormulation<celltype>>;


}  // namespace Discret::Elements

FOUR_C_NAMESPACE_CLOSE
#endif

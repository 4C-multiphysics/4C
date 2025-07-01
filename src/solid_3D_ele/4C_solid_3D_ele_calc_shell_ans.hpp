// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SOLID_3D_ELE_CALC_SHELL_ANS_HPP
#define FOUR_C_SOLID_3D_ELE_CALC_SHELL_ANS_HPP

#include "4C_config.hpp"

#include "4C_fem_general_cell_type.hpp"
#include "4C_fem_general_cell_type_traits.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_solid_3D_ele_calc.hpp"
#include "4C_solid_3D_ele_calc_eas_helpers.hpp"
#include "4C_solid_3D_ele_calc_lib.hpp"
#include "4C_solid_3D_ele_calc_shell_ans_lib.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Discret::Elements
{
  template <std::size_t num_sampling_points, Core::FE::CellType celltype>
  struct ShellANSPreparationData
  {
    std::array<SamplingPointData<celltype>, num_sampling_points> sampling_point_data{};
  };

  template <Core::FE::CellType celltype>
  struct ShellANSLinearizationContainer
  {
    Core::LinAlg::Matrix<Internal::num_str<celltype>,
        Core::FE::num_nodes(celltype) * Core::FE::dim<celltype>>
        Bop_{};

    Core::LinAlg::Matrix<Internal::num_str<celltype>, Internal::num_str<celltype>> TinvT{};
  };

  /*!
   * @brief A solid-shell element formulation with ANS
   *
   * @note The ordering of the nodes within the element determine the direction of the thickness of
   * the shell. The thickness-direction is assumed to be the t-direction of the element's parameter
   * space.
   *
   * @tparam celltype
   */
  template <Core::FE::CellType celltype>
  struct ShellANSFormulation
  {
    static constexpr bool has_gauss_point_history = false;
    static constexpr bool has_global_history = false;
    static constexpr bool has_preparation_data = true;
    static constexpr bool has_condensed_contribution = false;

    static constexpr auto ans_sampling_points = SamplingPoints<celltype>::value;

    using LinearizationContainer = ShellANSLinearizationContainer<celltype>;
    using PreparationData = ShellANSPreparationData<ans_sampling_points.size(), celltype>;

    static ShellANSPreparationData<ans_sampling_points.size(), celltype> prepare(
        const Core::Elements::Element& ele, const ElementNodes<celltype>& nodal_coordinates)
    {
      ShellANSPreparationData<ans_sampling_points.size(), celltype> shell_ans_data{};
      shell_ans_data.sampling_point_data =
          evaluate_sampling_point_data(ele, nodal_coordinates, ans_sampling_points);
      return shell_ans_data;
    }

    template <typename Evaluator>
    static inline auto evaluate(const Core::Elements::Element& ele,
        const ElementNodes<celltype>& element_nodes,
        const Core::LinAlg::Tensor<double, Core::FE::dim<celltype>>& xi,
        const ShapeFunctionsAndDerivatives<celltype>& shape_functions,
        const JacobianMapping<celltype>& jacobian_mapping,
        const ShellANSPreparationData<ans_sampling_points.size(), celltype>& preparation_data,
        Evaluator evaluator)
    {
      // evaluate local b-op
      Core::LinAlg::Matrix<num_str<celltype>, num_dof_per_ele<celltype>> bop_local =
          evaluate_local_b_operator(element_nodes, xi, shape_functions, jacobian_mapping,
              preparation_data.sampling_point_data);

      const ShellANSLinearizationContainer<celltype> linearization = std::invoke(
          [&]()
          {
            ShellANSLinearizationContainer<celltype> linearization{};
            linearization.TinvT = evaluate_voigt_transformation_matrix(jacobian_mapping);
            linearization.Bop_.multiply(linearization.TinvT, bop_local);
            return linearization;
          });

      Core::LinAlg::Matrix<6, 1> gl_strain_local = evaluate_local_glstrain(element_nodes, xi,
          shape_functions, jacobian_mapping, preparation_data.sampling_point_data);

      Core::LinAlg::Matrix<6, 1> gl_strain;
      gl_strain.multiply(linearization.TinvT, gl_strain_local);

      Core::LinAlg::Voigt::Strains::to_stress_like(gl_strain, gl_strain);
      Core::LinAlg::SymmetricTensor<double, 3, 3> gl_strain_tensor;
      std::ranges::copy_n(gl_strain.data(), 6, gl_strain_tensor.data());



      const SpatialMaterialMapping<celltype> spatial_material_mapping =
          evaluate_spatial_material_mapping(jacobian_mapping, element_nodes);

      Core::LinAlg::Tensor<double, 3, 3> consistent_defgrd =
          compute_deformation_gradient_from_gl_strains(
              spatial_material_mapping.deformation_gradient_, gl_strain_tensor);

      return evaluator(consistent_defgrd, gl_strain_tensor, linearization);
    }

    static inline Core::LinAlg::Matrix<9, Core::FE::num_nodes(celltype) * Core::FE::dim<celltype>>
    evaluate_d_deformation_gradient_d_displacements(const Core::Elements::Element& ele,
        const ElementNodes<celltype>& element_nodes,
        const Core::LinAlg::Tensor<double, Core::FE::dim<celltype>>& xi,
        const ShapeFunctionsAndDerivatives<celltype>& shape_functions,
        const JacobianMapping<celltype>& jacobian_mapping,
        const Core::LinAlg::Tensor<double, Internal::num_dim<celltype>,
            Internal::num_dim<celltype>>& deformation_gradient,
        const ShellANSPreparationData<ans_sampling_points.size(), celltype>& preparation_data)
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
        const ShellANSPreparationData<ans_sampling_points.size(), celltype>& preparation_data)
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
        const ShellANSPreparationData<ans_sampling_points.size(), celltype>& preparation_data)
    {
      FOUR_C_THROW(
          "This second derivative of the deformation gradient w.r.t. the displacements and xi "
          "is "
          "not implemented");
    }

    static void add_internal_force_vector(const JacobianMapping<celltype>& jacobian_mapping,
        const Core::LinAlg::Tensor<double, Core::FE::dim<celltype>, Core::FE::dim<celltype>>& F,
        const ShellANSLinearizationContainer<celltype>& linearization,
        const Stress<celltype>& stress, const double integration_factor,
        const ShellANSPreparationData<ans_sampling_points.size(), celltype>& preparation_data,
        Core::LinAlg::Matrix<Core::FE::num_nodes(celltype) * Core::FE::dim<celltype>, 1>&
            force_vector)
    {
      Discret::Elements::add_internal_force_vector(
          linearization.Bop_, stress, integration_factor, force_vector);
    }

    static void add_stiffness_matrix(const JacobianMapping<celltype>& jacobian_mapping,
        const Core::LinAlg::Tensor<double, Core::FE::dim<celltype>, Core::FE::dim<celltype>>& F,
        const Core::LinAlg::Tensor<double, Core::FE::dim<celltype>>& xi,
        const ShapeFunctionsAndDerivatives<celltype>& shape_functions,
        const ShellANSLinearizationContainer<celltype>& linearization,
        const Stress<celltype>& stress, const double integration_factor,
        const ShellANSPreparationData<ans_sampling_points.size(), celltype>& preparation_data,
        Core::LinAlg::Matrix<Core::FE::num_nodes(celltype) * Core::FE::dim<celltype>,
            Core::FE::num_nodes(celltype) * Core::FE::dim<celltype>>& stiffness_matrix)
    {
      Discret::Elements::add_elastic_stiffness_matrix(
          linearization.Bop_, stress, integration_factor, stiffness_matrix);


      add_ans_geometric_stiffness(xi, shape_functions, stress, integration_factor,
          linearization.TinvT, preparation_data.sampling_point_data, stiffness_matrix);
    }
  };

  template <Core::FE::CellType celltype>
  using ANSSolidShellIntegrator = SolidEleCalc<celltype, ShellANSFormulation<celltype>>;


}  // namespace Discret::Elements

FOUR_C_NAMESPACE_CLOSE
#endif

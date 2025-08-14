// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SOLID_3D_ELE_CALC_SHELL_EAS_ANS_HPP
#define FOUR_C_SOLID_3D_ELE_CALC_SHELL_EAS_ANS_HPP

#include "4C_config.hpp"

#include "4C_fem_general_cell_type.hpp"
#include "4C_fem_general_cell_type_traits.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_solid_3D_ele_calc.hpp"
#include "4C_solid_3D_ele_calc_eas_helpers.hpp"
#include "4C_solid_3D_ele_calc_lib.hpp"
#include "4C_solid_3D_ele_calc_shell_ans_lib.hpp"
#include "4C_solid_3D_ele_formulation.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Discret::Elements
{

  template <std::size_t num_sampling_points, Core::FE::CellType celltype>
  struct ShellEASANSPreparationData
  {
    std::array<SamplingPointData<celltype>, num_sampling_points> sampling_point_data{};

    CentroidTransformation<celltype> centeroid_transformation{};
  };

  template <Core::FE::CellType celltype, Discret::Elements::EasType eastype>
  struct ShellEASANSLinearizationContainer
  {
    static constexpr int num_str = Core::FE::dim<celltype> * (Core::FE::dim<celltype> + 1) / 2;

    Core::LinAlg::Matrix<Internal::num_str<celltype>,
        Core::FE::num_nodes(celltype) * Core::FE::dim<celltype>>
        b_op{};

    Core::LinAlg::Matrix<Internal::num_str<celltype>, Internal::num_str<celltype>> TinvT{};

    Core::LinAlg::Matrix<num_str, Discret::Elements::EasTypeToNumEas<eastype>::num_eas> m_tilde{};
  };

  template <Core::FE::CellType celltype, Discret::Elements::EasType eastype>
  struct EASANSHistoryData
  {
    /// EAS matrices and vectors to be stored between iterations
    Discret::Elements::EasIterationData<celltype, eastype> eas_iteration_data = {};

    // line search parameter (old step length)
    double old_step_length = 1.0;
  };

  /*!
   * @brief A solid-shell element formulation with EAS and ANS
   *
   * @note The ordering of the nodes within the element determine the direction of the thickness of
   * the shell. The thickness-direction is assumed to be the t-direction of the element's parameter
   * space.
   *
   * @tparam celltype
   */
  template <Core::FE::CellType celltype, Discret::Elements::EasType eastype>
  struct ShellEASANSFormulation
  {
    static constexpr bool has_gauss_point_history = false;
    static constexpr bool has_global_history = true;
    static constexpr bool has_preparation_data = true;
    static constexpr bool has_condensed_contribution = true;

    static constexpr auto ans_sampling_points = SamplingPoints<celltype>::value;

    using LinearizationContainer = ShellEASANSLinearizationContainer<celltype, eastype>;
    using PreparationData = ShellEASANSPreparationData<ans_sampling_points.size(), celltype>;
    using GlobalHistory = EASANSHistoryData<celltype, eastype>;
    using CondensedContributionData = Core::LinAlg::Matrix<num_dof_per_ele<celltype>,
        Discret::Elements::EasTypeToNumEas<eastype>::num_eas>;

    static PreparationData prepare(const Core::Elements::Element& ele,
        const ElementNodes<celltype>& nodal_coordinates, const GlobalHistory& history_data)
    {
      PreparationData shell_eas_ans_data{};
      shell_eas_ans_data.sampling_point_data =
          evaluate_sampling_point_data(ele, nodal_coordinates, ans_sampling_points);

      shell_eas_ans_data.centeroid_transformation =
          evaluate_centroid_transformation<celltype>(nodal_coordinates);
      return shell_eas_ans_data;
    }

    template <typename Evaluator>
    static inline auto evaluate(const Core::Elements::Element& ele,
        const ElementNodes<celltype>& element_nodes,
        const Core::LinAlg::Tensor<double, Core::FE::dim<celltype>>& xi,
        const ShapeFunctionsAndDerivatives<celltype>& shape_functions,
        const JacobianMapping<celltype>& jacobian_mapping, const PreparationData& preparation_data,
        const GlobalHistory& history_data, Evaluator evaluator)
    {
      // evaluate local b-op
      Core::LinAlg::Matrix<num_str<celltype>, num_dof_per_ele<celltype>> bop_local =
          evaluate_local_b_operator(element_nodes, xi, shape_functions, jacobian_mapping,
              preparation_data.sampling_point_data);

      const LinearizationContainer linearization = std::invoke(
          [&]()
          {
            LinearizationContainer linearization{};
            linearization.TinvT = evaluate_voigt_transformation_matrix(jacobian_mapping);
            linearization.b_op.multiply(linearization.TinvT, bop_local);
            linearization.m_tilde = evaluate_eas_shape_functions_material_config<celltype, eastype>(
                jacobian_mapping.determinant_, preparation_data.centeroid_transformation, xi);
            return linearization;
          });

      Core::LinAlg::Matrix<6, 1> gl_strain_local = evaluate_local_glstrain(element_nodes, xi,
          shape_functions, jacobian_mapping, preparation_data.sampling_point_data);

      Core::LinAlg::Matrix<6, 1> gl_strain;
      gl_strain.multiply(linearization.TinvT, gl_strain_local);

      Core::LinAlg::Voigt::Strains::to_stress_like(gl_strain, gl_strain);
      Core::LinAlg::SymmetricTensor<double, 3, 3> gl_strain_tensor;
      std::ranges::copy_n(gl_strain.data(), 6, gl_strain_tensor.data());

      Core::LinAlg::SymmetricTensor<double, 3, 3> gl_strain_eas =
          gl_strain_tensor + evaluate_enhanced_assumed_gl_strains<celltype, eastype>(
                                 linearization.m_tilde, history_data.eas_iteration_data.alpha);



      const SpatialMaterialMapping<celltype> spatial_material_mapping =
          evaluate_spatial_material_mapping(jacobian_mapping, element_nodes);

      Core::LinAlg::Tensor<double, 3, 3> consistent_defgrd =
          compute_deformation_gradient_from_gl_strains(
              spatial_material_mapping.deformation_gradient_, gl_strain_eas);

      return evaluator(consistent_defgrd, gl_strain_eas, linearization);
    }

    static inline Core::LinAlg::Matrix<9, Core::FE::num_nodes(celltype) * Core::FE::dim<celltype>>
    evaluate_d_deformation_gradient_d_displacements(const Core::Elements::Element& ele,
        const ElementNodes<celltype>& element_nodes,
        const Core::LinAlg::Tensor<double, Core::FE::dim<celltype>>& xi,
        const ShapeFunctionsAndDerivatives<celltype>& shape_functions,
        const JacobianMapping<celltype>& jacobian_mapping,
        const Core::LinAlg::Tensor<double, Internal::num_dim<celltype>,
            Internal::num_dim<celltype>>& deformation_gradient,
        const PreparationData& preparation_data, const GlobalHistory& history_data)
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
        const PreparationData& preparation_data, const GlobalHistory& history_data)
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
        const PreparationData& preparation_data, const GlobalHistory& history_data)
    {
      FOUR_C_THROW(
          "This second derivative of the deformation gradient w.r.t. the displacements and xi "
          "is "
          "not implemented");
    }

    static void add_internal_force_vector(const JacobianMapping<celltype>& jacobian_mapping,
        const Core::LinAlg::Tensor<double, Core::FE::dim<celltype>, Core::FE::dim<celltype>>& F,
        const LinearizationContainer& linearization, const Stress<celltype>& stress,
        const double integration_factor, const PreparationData& preparation_data,
        const GlobalHistory& history_data,
        Core::LinAlg::Matrix<Core::FE::num_nodes(celltype) * Core::FE::dim<celltype>, 1>&
            force_vector)
    {
      Discret::Elements::add_internal_force_vector(
          linearization.b_op, stress, integration_factor, force_vector);
    }

    static void add_stiffness_matrix(const JacobianMapping<celltype>& jacobian_mapping,
        const Core::LinAlg::Tensor<double, Core::FE::dim<celltype>, Core::FE::dim<celltype>>& F,
        const Core::LinAlg::Tensor<double, Core::FE::dim<celltype>>& xi,
        const ShapeFunctionsAndDerivatives<celltype>& shape_functions,
        const LinearizationContainer& linearization, const Stress<celltype>& stress,
        const double integration_factor, const PreparationData& preparation_data,
        const GlobalHistory& history_data,
        Core::LinAlg::Matrix<Core::FE::num_nodes(celltype) * Core::FE::dim<celltype>,
            Core::FE::num_nodes(celltype) * Core::FE::dim<celltype>>& stiffness_matrix)
    {
      Discret::Elements::add_elastic_stiffness_matrix(
          linearization.b_op, stress, integration_factor, stiffness_matrix);


      add_ans_geometric_stiffness(xi, shape_functions, stress, integration_factor,
          linearization.TinvT, preparation_data.sampling_point_data, stiffness_matrix);
    }

    static void reset_condensed_variable_integration(const Core::Elements::Element& ele,
        const ElementNodes<celltype>& nodal_coordinatesconst,
        const PreparationData& centeroid_transformation, GlobalHistory& global_history)
    {
      global_history.eas_iteration_data.invKaa.clear();
      global_history.eas_iteration_data.Kda.clear();
      global_history.eas_iteration_data.s.clear();
    }

    static void integrate_condensed_contribution(
        const LinearizationContainer& linearization_container, const Stress<celltype>& stress,
        const double integration_factor, const PreparationData& centeroid_transformation,
        GlobalHistory& eas_data)
    {
      integrate_eas<celltype, eastype>(stress, linearization_container.m_tilde,
          linearization_container.b_op, integration_factor, eas_data.eas_iteration_data);
    }

    static CondensedContributionData prepare_condensed_contribution(
        const PreparationData& centeroid_transformation, GlobalHistory& eas_data)
    {
      // invert Kaa with solver. eas_iteration_data_.invKaa then is Kaa^{-1}
      solve_for_inverse_ignoring_errors(eas_data.eas_iteration_data.invKaa);

      // compute the product (- Kda Kaa^{-1}) which is later needed for force and stiffness update
      Core::LinAlg::Matrix<num_dof_per_ele<celltype>,
          Discret::Elements::EasTypeToNumEas<eastype>::num_eas>
          minusKdainvKaa(Core::LinAlg::Initialization::zero);
      minusKdainvKaa.multiply_nn(
          -1.0, eas_data.eas_iteration_data.Kda, eas_data.eas_iteration_data.invKaa);

      return minusKdainvKaa;
    }

    static void update_condensed_variables(const Core::Elements::Element& ele,
        FourC::Solid::Elements::ParamsInterface* params_interface,
        const ElementNodes<celltype>& element_nodes,
        const Core::LinAlg::Matrix<Internal::num_dof_per_ele<celltype>, 1>& displacement_increments,
        const double linesearch_step_length, const PreparationData& preparation_data,
        GlobalHistory& eas_data)
    {
      if (params_interface)
      {
        // params_interface is optional and only available when called from the new time integration
        // framework
        params_interface->sum_into_my_previous_sol_norm(NOX::Nln::StatusTest::quantity_eas,
            Discret::Elements::EasTypeToNumEas<eastype>::num_eas,
            eas_data.eas_iteration_data.alpha.data(), ele.owner());
      }

      update_alpha(eas_data.eas_iteration_data, displacement_increments, linesearch_step_length);

      eas_data.old_step_length = linesearch_step_length;

      if (params_interface)
      {
        params_interface->sum_into_my_update_norm(NOX::Nln::StatusTest::quantity_eas,
            Discret::Elements::EasTypeToNumEas<eastype>::num_eas,
            eas_data.eas_iteration_data.alpha_inc.data(), eas_data.eas_iteration_data.alpha.data(),
            linesearch_step_length, ele.owner());
      }
    }

    static void correct_condensed_variables_for_linesearch(const Core::Elements::Element& ele,
        FourC::Solid::Elements::ParamsInterface* params_interface,
        const double linesearch_step_length, const PreparationData& preparation_data,
        GlobalHistory& eas_data)
    {
      correct_alpha(eas_data.eas_iteration_data, linesearch_step_length, eas_data.old_step_length);

      eas_data.old_step_length = linesearch_step_length;

      if (params_interface)
      {
        params_interface->sum_into_my_update_norm(NOX::Nln::StatusTest::quantity_eas,
            Discret::Elements::EasTypeToNumEas<eastype>::num_eas,
            eas_data.eas_iteration_data.alpha_inc.data(), eas_data.eas_iteration_data.alpha.data(),
            linesearch_step_length, ele.owner());
      }
    }

    static void add_condensed_contribution_to_force_vector(
        const CondensedContributionData& minusKdainvKaa,
        const PreparationData& centeroid_transformation, GlobalHistory& eas_data,
        Core::LinAlg::Matrix<Core::FE::num_nodes(celltype) * Core::FE::dim<celltype>, 1>&
            force_vector)
    {
      add_eas_internal_force<celltype, eastype>(
          minusKdainvKaa, eas_data.eas_iteration_data.s, force_vector);
    }

    static void add_condensed_contribution_to_stiffness_matrix(
        const CondensedContributionData& minusKdainvKaa,
        const PreparationData& centeroid_transformation, GlobalHistory& eas_data,
        Core::LinAlg::Matrix<Core::FE::num_nodes(celltype) * Core::FE::dim<celltype>,
            Core::FE::num_nodes(celltype) * Core::FE::dim<celltype>>& stiffness_matrix)
    {
      add_eas_stiffness_matrix<celltype, eastype>(
          minusKdainvKaa, eas_data.eas_iteration_data.Kda, stiffness_matrix);
    }

    static void pack(const GlobalHistory& history_data, Core::Communication::PackBuffer& data)
    {
      add_to_pack(data, history_data.eas_iteration_data.alpha_inc);
      add_to_pack(data, history_data.eas_iteration_data.alpha);
      add_to_pack(data, history_data.eas_iteration_data.s);
      add_to_pack(data, history_data.eas_iteration_data.invKaa);
      add_to_pack(data, history_data.eas_iteration_data.Kda);
      add_to_pack(data, history_data.old_step_length);
    }

    static void unpack(Core::Communication::UnpackBuffer& buffer, GlobalHistory& history_data)
    {
      extract_from_pack(buffer, history_data.eas_iteration_data.alpha_inc);
      extract_from_pack(buffer, history_data.eas_iteration_data.alpha);
      extract_from_pack(buffer, history_data.eas_iteration_data.s);
      extract_from_pack(buffer, history_data.eas_iteration_data.invKaa);
      extract_from_pack(buffer, history_data.eas_iteration_data.Kda);
      extract_from_pack(buffer, history_data.old_step_length);
    }
  };

  template <Core::FE::CellType celltype, Discret::Elements::EasType eastype>
  using EasAnsSolidShellIntegrator =
      SolidEleCalc<celltype, ShellEASANSFormulation<celltype, eastype>>;


}  // namespace Discret::Elements

FOUR_C_NAMESPACE_CLOSE
#endif

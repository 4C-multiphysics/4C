// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SOLID_3D_ELE_CALC_EAS_HPP
#define FOUR_C_SOLID_3D_ELE_CALC_EAS_HPP

#include "4C_config.hpp"

#include "4C_fem_general_cell_type.hpp"
#include "4C_fem_general_cell_type_traits.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_solid_3D_ele_calc.hpp"
#include "4C_solid_3D_ele_calc_eas_helpers.hpp"
#include "4C_solid_3D_ele_calc_lib.hpp"
#include "4C_solid_3D_ele_formulation.hpp"
#include "4C_solid_3D_ele_properties.hpp"
#include "4C_solver_nonlin_nox_enum_lists.hpp"
#include "4C_structure_new_elements_paramsinterface.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Discret::Elements
{
  template <Core::FE::CellType celltype, Discret::Elements::EasType eastype>
  struct EASHistoryData
  {
    /// EAS matrices and vectors to be stored between iterations
    Discret::Elements::EasIterationData<celltype, eastype> eas_iteration_data = {};

    // line search parameter (old step length)
    double old_step_length = 1.0;
  };

  /*!
   * @brief A solid element formulation with EAS
   *
   * @tparam celltype
   */
  template <Core::FE::CellType celltype, Discret::Elements::EasType eastype,
      Inpar::Solid::KinemType kinematic_type>
  struct EASFormulation
  {
    static constexpr bool has_gauss_point_history = false;
    static constexpr bool has_global_history = true;
    static constexpr bool has_preparation_data = true;
    static constexpr bool is_prestress_updatable = false;
    static constexpr bool has_condensed_contribution = true;

    using LinearizationContainer = EASKinematics<celltype, eastype>;
    using GlobalHistory = EASHistoryData<celltype, eastype>;
    using PreparationData = CentroidTransformation<celltype>;
    using CondensedContributionData = Core::LinAlg::Matrix<num_dof_per_ele<celltype>,
        Discret::Elements::EasTypeToNumEas<eastype>::num_eas>;

    static PreparationData prepare(const Core::Elements::Element& ele,
        const ElementNodes<celltype>& nodal_coordinates, GlobalHistory& global_history)
    {
      return evaluate_centroid_transformation<celltype>(nodal_coordinates);
    }

    template <typename Evaluator>
    static auto evaluate(const Core::Elements::Element& ele,
        const ElementNodes<celltype>& element_nodes,
        const Core::LinAlg::Tensor<double, Core::FE::dim<celltype>>& xi,
        const ShapeFunctionsAndDerivatives<celltype>& shape_functions,
        const JacobianMapping<celltype>& jacobian_mapping,
        const PreparationData& centeroid_transformation,
        const EASHistoryData<celltype, eastype>& eas_data, Evaluator evaluator)
    {
      const EASKinematics<celltype, eastype> kinematic_quantities =
          evaluate_eas_kinematics<celltype, eastype, kinematic_type>(element_nodes,
              centeroid_transformation, xi, jacobian_mapping, eas_data.eas_iteration_data);


      return evaluator(kinematic_quantities.enhanced_deformation_gradient,
          kinematic_quantities.enhanced_gl, kinematic_quantities);
    }

    static inline Core::LinAlg::Matrix<9, Core::FE::dim<celltype>>
    evaluate_d_deformation_gradient_d_xi(const Core::Elements::Element& ele,
        const ElementNodes<celltype>& element_nodes,
        const Core::LinAlg::Tensor<double, Core::FE::dim<celltype>>& xi,
        const ShapeFunctionsAndDerivatives<celltype>& shape_functions,
        const JacobianMapping<celltype>& jacobian_mapping,
        const Core::LinAlg::Tensor<double, Internal::num_dim<celltype>,
            Internal::num_dim<celltype>>& deformation_gradient,
        const PreparationData& centeroid_transformation,
        const EASHistoryData<celltype, eastype>& eas_data)
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
        const PreparationData& centeroid_transformation,
        const EASHistoryData<celltype, eastype>& eas_data)
    {
      FOUR_C_THROW(
          "This second derivative of the deformation gradient w.r.t. the displacements and xi is "
          "not implemented");
    }

    static inline Core::LinAlg::Matrix<9, Core::FE::num_nodes(celltype) * Core::FE::dim<celltype>>
    evaluate_d_deformation_gradient_d_displacements(const Core::Elements::Element& ele,
        const ElementNodes<celltype>& element_nodes,
        const Core::LinAlg::Tensor<double, Core::FE::dim<celltype>>& xi,
        const ShapeFunctionsAndDerivatives<celltype>& shape_functions,
        const JacobianMapping<celltype>& jacobian_mapping,
        const Core::LinAlg::Tensor<double, Internal::num_dim<celltype>,
            Internal::num_dim<celltype>>& deformation_gradient,
        const PreparationData& centeroid_transformation,
        const EASHistoryData<celltype, eastype>& eas_data)
    {
      FOUR_C_THROW(
          "This derivative of the deformation gradient w.r.t. the displacements is not "
          "implemented");
    }

    static void add_internal_force_vector(const JacobianMapping<celltype>& jacobian_mapping,
        const Core::LinAlg::Tensor<double, Core::FE::dim<celltype>, Core::FE::dim<celltype>>& F,
        const EASKinematics<celltype, eastype>& eas_kinematics, const Stress<celltype>& stress,
        const double integration_factor, const PreparationData& centeroid_transformation,
        EASHistoryData<celltype, eastype>& eas_data,
        Core::LinAlg::Matrix<Core::FE::num_nodes(celltype) * Core::FE::dim<celltype>, 1>&
            force_vector)
    {
      Discret::Elements::add_internal_force_vector(
          eas_kinematics.b_op, stress, integration_factor, force_vector);
    }

    static void add_stiffness_matrix(const JacobianMapping<celltype>& jacobian_mapping,
        const Core::LinAlg::Tensor<double, Core::FE::dim<celltype>, Core::FE::dim<celltype>>& F,
        const Core::LinAlg::Tensor<double, Core::FE::dim<celltype>>& xi,
        const ShapeFunctionsAndDerivatives<celltype>& shape_functions,
        const EASKinematics<celltype, eastype>& eas_kinematics, const Stress<celltype>& stress,
        const double integration_factor, const PreparationData& preparation_data,
        EASHistoryData<celltype, eastype>& eas_data,
        Core::LinAlg::Matrix<Core::FE::num_nodes(celltype) * Core::FE::dim<celltype>,
            Core::FE::num_nodes(celltype) * Core::FE::dim<celltype>>& stiffness_matrix)
    {
      add_elastic_stiffness_matrix(
          eas_kinematics.b_op, stress, integration_factor, stiffness_matrix);
      add_geometric_stiffness_matrix(
          jacobian_mapping, stress, integration_factor, stiffness_matrix);
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
        const EASKinematics<celltype, eastype>& eas_kinematics, const Stress<celltype>& stress,
        const double integration_factor, const PreparationData& centeroid_transformation,
        EASHistoryData<celltype, eastype>& eas_data)
    {
      integrate_eas<celltype, eastype>(stress, eas_kinematics.m_tilde, eas_kinematics.b_op,
          integration_factor, eas_data.eas_iteration_data);
    }

    static CondensedContributionData prepare_condensed_contribution(
        const PreparationData& centeroid_transformation,
        EASHistoryData<celltype, eastype>& eas_data)
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
        EASHistoryData<celltype, eastype>& eas_data)
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
        EASHistoryData<celltype, eastype>& eas_data)
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
        const PreparationData& centeroid_transformation,
        EASHistoryData<celltype, eastype>& eas_data,
        Core::LinAlg::Matrix<Core::FE::num_nodes(celltype) * Core::FE::dim<celltype>, 1>&
            force_vector)
    {
      add_eas_internal_force<celltype, eastype>(
          minusKdainvKaa, eas_data.eas_iteration_data.s, force_vector);
    }

    static void add_condensed_contribution_to_stiffness_matrix(
        const CondensedContributionData& minusKdainvKaa,
        const PreparationData& centeroid_transformation,
        EASHistoryData<celltype, eastype>& eas_data,
        Core::LinAlg::Matrix<Core::FE::num_nodes(celltype) * Core::FE::dim<celltype>,
            Core::FE::num_nodes(celltype) * Core::FE::dim<celltype>>& stiffness_matrix)
    {
      add_eas_stiffness_matrix<celltype, eastype>(
          minusKdainvKaa, eas_data.eas_iteration_data.Kda, stiffness_matrix);
    }

    static void pack(const EASHistoryData<celltype, eastype>& history_data,
        Core::Communication::PackBuffer& data)
    {
      add_to_pack(data, history_data.eas_iteration_data.alpha_inc);
      add_to_pack(data, history_data.eas_iteration_data.alpha);
      add_to_pack(data, history_data.eas_iteration_data.s);
      add_to_pack(data, history_data.eas_iteration_data.invKaa);
      add_to_pack(data, history_data.eas_iteration_data.Kda);
      add_to_pack(data, history_data.old_step_length);
    }

    static void unpack(
        Core::Communication::UnpackBuffer& buffer, EASHistoryData<celltype, eastype>& history_data)
    {
      extract_from_pack(buffer, history_data.eas_iteration_data.alpha_inc);
      extract_from_pack(buffer, history_data.eas_iteration_data.alpha);
      extract_from_pack(buffer, history_data.eas_iteration_data.s);
      extract_from_pack(buffer, history_data.eas_iteration_data.invKaa);
      extract_from_pack(buffer, history_data.eas_iteration_data.Kda);
      extract_from_pack(buffer, history_data.old_step_length);
    }
  };

  template <Core::FE::CellType celltype, Discret::Elements::EasType eastype,
      Inpar::Solid::KinemType kinematic_type>
  using EASSolidIntegrator =
      SolidEleCalc<celltype, EASFormulation<celltype, eastype, kinematic_type>>;
}  // namespace Discret::Elements

FOUR_C_NAMESPACE_CLOSE

#endif

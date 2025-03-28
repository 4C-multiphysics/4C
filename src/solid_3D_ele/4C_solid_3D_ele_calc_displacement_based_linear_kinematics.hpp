// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SOLID_3D_ELE_CALC_DISPLACEMENT_BASED_LINEAR_KINEMATICS_HPP
#define FOUR_C_SOLID_3D_ELE_CALC_DISPLACEMENT_BASED_LINEAR_KINEMATICS_HPP

#include "4C_config.hpp"

#include "4C_fem_general_cell_type.hpp"
#include "4C_fem_general_cell_type_traits.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_solid_3D_ele_calc.hpp"
#include "4C_solid_3D_ele_calc_lib.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Discret::Elements
{
  /*!
   * @brief A container holding the linearization of the displacement based linear kinematics solid
   * element formulation
   */
  template <Core::FE::CellType celltype>
  struct DisplacementBasedLinearKinematicsLinearizationContainer
  {
    Core::LinAlg::Matrix<Internal::num_str<celltype>,
        Core::FE::num_nodes<celltype> * Core::FE::dim<celltype>>
        linear_b_operator_{};
  };

  /*!
   * @brief A displacement based solid element formulation with linear kinematics (i.e. small
   * displacements)
   *
   * @tparam celltype
   */
  template <Core::FE::CellType celltype>
  struct DisplacementBasedLinearKinematicsFormulation
  {
    static constexpr bool has_gauss_point_history = false;
    static constexpr bool has_global_history = false;
    static constexpr bool has_preparation_data = false;
    static constexpr bool has_condensed_contribution = false;


    using LinearizationContainer =
        DisplacementBasedLinearKinematicsLinearizationContainer<celltype>;

    template <typename Evaluator>
    static inline auto evaluate(const Core::Elements::Element& ele,
        const ElementNodes<celltype>& nodal_coordinates,
        const Core::LinAlg::Matrix<Internal::num_dim<celltype>, 1>& xi,
        const ShapeFunctionsAndDerivatives<celltype>& shape_functions,
        const JacobianMapping<celltype>& jacobian_mapping, Evaluator evaluator)
    {
      const DisplacementBasedLinearKinematicsLinearizationContainer<celltype> linearization{
          Discret::Elements::evaluate_linear_strain_gradient(jacobian_mapping)};

      Core::LinAlg::Matrix<Internal::num_str<celltype>, 1> gl_strain =
          evaluate_linear_gl_strain(nodal_coordinates, linearization.linear_b_operator_);

      return evaluator(
          Core::LinAlg::identity_matrix<Core::FE::dim<celltype>>(), gl_strain, linearization);
    }

    static inline Core::LinAlg::Matrix<9, Core::FE::num_nodes<celltype> * Core::FE::dim<celltype>>
    evaluate_d_deformation_gradient_d_displacements(const Core::Elements::Element& ele,
        const ElementNodes<celltype>& element_nodes,
        const Core::LinAlg::Matrix<Internal::num_dim<celltype>, 1>& xi,
        const ShapeFunctionsAndDerivatives<celltype>& shape_functions,
        const JacobianMapping<celltype>& jacobian_mapping,
        const Core::LinAlg::Matrix<Internal::num_dim<celltype>, Internal::num_dim<celltype>>&
            deformation_gradient)
    {
      // linearization is zero for small displacements
      return Core::LinAlg::Matrix<9, Core::FE::num_nodes<celltype> * Core::FE::dim<celltype>>(
          Core::LinAlg::Initialization::zero);
    }

    static inline Core::LinAlg::Matrix<9, Core::FE::dim<celltype>>
    evaluate_d_deformation_gradient_d_xi(const Core::Elements::Element& ele,
        const ElementNodes<celltype>& element_nodes,
        const Core::LinAlg::Matrix<Internal::num_dim<celltype>, 1>& xi,
        const ShapeFunctionsAndDerivatives<celltype>& shape_functions,
        const JacobianMapping<celltype>& jacobian_mapping,
        const Core::LinAlg::Matrix<Internal::num_dim<celltype>, Internal::num_dim<celltype>>&
            deformation_gradient)
    {
      // linearization is zero for small displacements
      return Core::LinAlg::Matrix<9, Core::FE::dim<celltype>>(Core::LinAlg::Initialization::zero);
    }

    static inline Core::LinAlg::Matrix<9,
        Core::FE::num_nodes<celltype> * Core::FE::dim<celltype> * Core::FE::dim<celltype>>
    evaluate_d_deformation_gradient_d_displacements_d_xi(const Core::Elements::Element& ele,
        const ElementNodes<celltype>& element_nodes,
        const Core::LinAlg::Matrix<Internal::num_dim<celltype>, 1>& xi,
        const ShapeFunctionsAndDerivatives<celltype>& shape_functions,
        const JacobianMapping<celltype>& jacobian_mapping,
        const Core::LinAlg::Matrix<Internal::num_dim<celltype>, Internal::num_dim<celltype>>&
            deformation_gradient)
    {
      // linearization is zero for small displacements
      return Core::LinAlg::Matrix<9,
          Core::FE::num_nodes<celltype> * Core::FE::dim<celltype> * Core::FE::dim<celltype>>(
          Core::LinAlg::Initialization::zero);
    }

    static Core::LinAlg::Matrix<Internal::num_str<celltype>,
        Core::FE::num_nodes<celltype> * Core::FE::dim<celltype>>
    get_linear_b_operator(
        const DisplacementBasedLinearKinematicsLinearizationContainer<celltype>& linearization)
    {
      return linearization.linear_b_operator_;
    }


    static void add_internal_force_vector(
        const DisplacementBasedLinearKinematicsLinearizationContainer<celltype>& linearization,
        const Stress<celltype>& stress, const double integration_factor,
        Core::LinAlg::Matrix<Core::FE::num_nodes<celltype> * Core::FE::dim<celltype>, 1>&
            force_vector)
    {
      Discret::Elements::add_internal_force_vector(
          linearization.linear_b_operator_, stress, integration_factor, force_vector);
    }

    static void add_stiffness_matrix(const Core::LinAlg::Matrix<Internal::num_dim<celltype>, 1>& xi,
        const ShapeFunctionsAndDerivatives<celltype>& shape_functions,
        const DisplacementBasedLinearKinematicsLinearizationContainer<celltype>& linearization,
        const JacobianMapping<celltype>& jacobian_mapping, const Stress<celltype>& stress,
        const double integration_factor,
        Core::LinAlg::Matrix<Core::FE::num_nodes<celltype> * Core::FE::dim<celltype>,
            Core::FE::num_nodes<celltype> * Core::FE::dim<celltype>>& stiffness_matrix)
    {
      Discret::Elements::add_elastic_stiffness_matrix(
          linearization.linear_b_operator_, stress, integration_factor, stiffness_matrix);
    }
  };

  template <Core::FE::CellType celltype>
  using DisplacementBasedLinearKinematicsSolidIntegrator =
      SolidEleCalc<celltype, DisplacementBasedLinearKinematicsFormulation<celltype>>;


}  // namespace Discret::Elements

FOUR_C_NAMESPACE_CLOSE
#endif
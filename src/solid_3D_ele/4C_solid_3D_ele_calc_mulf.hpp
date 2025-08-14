// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SOLID_3D_ELE_CALC_MULF_HPP
#define FOUR_C_SOLID_3D_ELE_CALC_MULF_HPP

#include "4C_config.hpp"

#include "4C_fem_general_cell_type_traits.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_solid_3D_ele_calc.hpp"
#include "4C_solid_3D_ele_calc_lib.hpp"
#include "4C_solid_3D_ele_calc_lib_mulf.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Discret::Elements
{
  template <Core::FE::CellType celltype>
  struct MulfLinearizationContainer
  {
    Core::LinAlg::Matrix<Internal::num_str<celltype>,
        Core::FE::num_nodes(celltype) * Core::FE::dim<celltype>>
        Bop{};
  };

  /*!
   * @brief A displacement based solid element formulation with MULF prestressing
   *
   * @tparam celltype
   */
  template <Core::FE::CellType celltype>
  struct MulfFormulation
  {
    static constexpr bool has_gauss_point_history = true;
    static constexpr bool has_global_history = false;
    static constexpr bool has_preparation_data = false;
    static constexpr bool is_prestress_updatable = true;
    static constexpr bool has_condensed_contribution = false;

    using LinearizationContainer = MulfLinearizationContainer<celltype>;
    using GaussPointHistory = MulfHistoryData<celltype>;

    template <typename Evaluator>
    static auto evaluate(const Core::Elements::Element& ele,
        const ElementNodes<celltype>& element_nodes,
        const Core::LinAlg::Tensor<double, Core::FE::dim<celltype>>& xi,
        const ShapeFunctionsAndDerivatives<celltype>& shape_functions,
        const JacobianMapping<celltype>& jacobian_mapping, MulfHistoryData<celltype>& history_data,
        Evaluator evaluator)
    {
      if (!history_data.is_setup)
      {
        history_data.inverse_jacobian = jacobian_mapping.inverse_jacobian_;
        history_data.is_setup = true;
      }

      const SpatialMaterialMapping<celltype> spatial_material_mapping =
          evaluate_mulf_spatial_material_mapping(
              jacobian_mapping, shape_functions, element_nodes.displacements, history_data);

      const Core::LinAlg::SymmetricTensor<double, Core::FE::dim<celltype>, Core::FE::dim<celltype>>
          cauchygreen = evaluate_cauchy_green<celltype>(spatial_material_mapping);

      const Core::LinAlg::SymmetricTensor<double, Core::FE::dim<celltype>, Core::FE::dim<celltype>>
          gl_strain = evaluate_green_lagrange_strain(cauchygreen);

      Core::LinAlg::Matrix<Internal::num_str<celltype>,
          Core::FE::num_nodes(celltype) * Core::FE::dim<celltype>>
          Bop = evaluate_strain_gradient(jacobian_mapping, spatial_material_mapping);

      const MulfLinearizationContainer<celltype> linearization = std::invoke(
          [&]()
          {
            return MulfLinearizationContainer<celltype>{
                evaluate_strain_gradient(jacobian_mapping, spatial_material_mapping)};
          });

      return evaluator(spatial_material_mapping.deformation_gradient_, gl_strain, linearization);
    }

    static void add_internal_force_vector(const JacobianMapping<celltype>& jacobian_mapping,
        const Core::LinAlg::Tensor<double, Core::FE::dim<celltype>, Core::FE::dim<celltype>>& F,
        const MulfLinearizationContainer<celltype>& linearization, const Stress<celltype>& stress,
        const double integration_factor, MulfHistoryData<celltype>& history_data,
        Core::LinAlg::Matrix<Core::FE::num_nodes(celltype) * Core::FE::dim<celltype>, 1>&
            force_vector)
    {
      Discret::Elements::add_internal_force_vector(
          linearization.Bop, stress, integration_factor, force_vector);
    }

    static void add_stiffness_matrix(const JacobianMapping<celltype>& jacobian_mapping,
        const Core::LinAlg::Tensor<double, Core::FE::dim<celltype>, Core::FE::dim<celltype>>& F,
        const Core::LinAlg::Tensor<double, Core::FE::dim<celltype>>& xi,
        const ShapeFunctionsAndDerivatives<celltype>& shape_functions,
        const MulfLinearizationContainer<celltype>& linearization, const Stress<celltype>& stress,
        const double integration_factor, MulfHistoryData<celltype>& history_data,
        Core::LinAlg::Matrix<Core::FE::num_nodes(celltype) * Core::FE::dim<celltype>,
            Core::FE::num_nodes(celltype) * Core::FE::dim<celltype>>& stiffness_matrix)
    {
      Discret::Elements::add_elastic_stiffness_matrix(
          linearization.Bop, stress, integration_factor, stiffness_matrix);
      Discret::Elements::add_geometric_stiffness_matrix(
          jacobian_mapping, stress, integration_factor, stiffness_matrix);
    }

    static void pack(
        const MulfHistoryData<celltype>& history_data, Core::Communication::PackBuffer& data)
    {
      add_to_pack(data, history_data);
    }

    static void unpack(
        Core::Communication::UnpackBuffer& buffer, MulfHistoryData<celltype>& history_data)
    {
      extract_from_pack(buffer, history_data);
    }

    static inline void update_prestress(const Core::Elements::Element& ele,
        const ElementNodes<celltype>& element_nodes,
        const Core::LinAlg::Tensor<double, Core::FE::dim<celltype>>& xi,
        const ShapeFunctionsAndDerivatives<celltype>& shape_functions,
        const JacobianMapping<celltype>& jacobian_mapping,
        const Core::LinAlg::Tensor<double, Internal::num_dim<celltype>,
            Internal::num_dim<celltype>>& deformation_gradient,
        MulfHistoryData<celltype>& history_data)
    {
      Core::LinAlg::Tensor<double, Core::FE::dim<celltype>, Core::FE::dim<celltype>> delta_defgrd =
          evaluate_mulf_deformation_gradient_update(
              shape_functions, element_nodes.displacements, history_data);

      // update mulf history data only if prestress is active
      Core::LinAlg::Tensor<double, Core::FE::dim<celltype>, Core::FE::dim<celltype>>
          inv_delta_defgrd = Core::LinAlg::inv(delta_defgrd);

      history_data.deformation_gradient = deformation_gradient;
      history_data.inverse_jacobian =
          Core::LinAlg::transpose(inv_delta_defgrd) * history_data.inverse_jacobian;
    }
  };

  template <Core::FE::CellType celltype>
  using MulfSolidIntegrator = SolidEleCalc<celltype, MulfFormulation<celltype>>;

  template <typename T>
  concept IsPrestressUpdateable = requires(T t, const Core::Elements::Element& ele,
      Mat::So3Material& mat, const Core::FE::Discretization& discretization,
      const std::vector<int>& lm, Teuchos::ParameterList& params) {
    t->update_prestress(ele, mat, discretization, lm, params);
  };

  namespace Internal
  {
    struct UpdatePrestressAction
    {
      UpdatePrestressAction(const Core::Elements::Element& e, Mat::So3Material& m,
          const Core::FE::Discretization& d, const std::vector<int>& lmvec,
          Teuchos::ParameterList& p)
          : element(e), mat(m), discretization(d), lm(lmvec), params(p)
      {
      }

      template <typename T>
        requires(IsPrestressUpdateable<T>)
      void operator()(T& updateable)
      {
        updateable->update_prestress(element, mat, discretization, lm, params);
      }

      template <typename T>
        requires(!IsPrestressUpdateable<T>)
      void operator()(T& other)
      {
        FOUR_C_THROW(
            "Your element evaluation {} does not allow to update prestress. You may need to add "
            "MULF to your element line definitions.",
            Core::Utils::get_type_name<T>().c_str());
      }

      const Core::Elements::Element& element;
      Mat::So3Material& mat;
      const Core::FE::Discretization& discretization;
      const std::vector<int>& lm;
      Teuchos::ParameterList& params;
    };
  }  // namespace Internal

  template <typename VariantType>
  void update_prestress(VariantType& variant, const Core::Elements::Element& element,
      Mat::So3Material& mat, const Core::FE::Discretization& discretization,
      const std::vector<int>& lm, Teuchos::ParameterList& params)
  {
    std::visit(Internal::UpdatePrestressAction(element, mat, discretization, lm, params), variant);
  }
}  // namespace Discret::Elements

FOUR_C_NAMESPACE_CLOSE

#endif

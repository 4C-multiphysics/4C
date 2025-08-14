// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SOLID_3D_ELE_FORMULATION_HPP
#define FOUR_C_SOLID_3D_ELE_FORMULATION_HPP

#include "4C_config.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_fem_general_cell_type.hpp"
#include "4C_solid_3D_ele_calc_lib.hpp"
#include "4C_structure_new_elements_paramsinterface.hpp"
#include "4C_utils_exceptions.hpp"

#include <tuple>
#include <type_traits>
#include <utility>


FOUR_C_NAMESPACE_OPEN

// forward declaration
namespace Solid::Elements
{
  class ParamsInterface;
}

namespace Discret::Elements
{
  /*!
   * @brief a trait for the solid formulation determining whether the formulation has a gauss point
   * history
   *
   * @tparam SolidFormulation
   */
  template <typename SolidFormulation>
  constexpr bool has_gauss_point_history = SolidFormulation::has_gauss_point_history;

  /*!
   * @brief a trait for the solid formulation determining whether the formulation has a global
   * history
   *
   * @tparam SolidFormulation
   */
  template <typename SolidFormulation>
  constexpr bool has_global_history = SolidFormulation::has_global_history;

  /*!
   * @brief a trait for the solid formulation determining whether the formulation has preparation
   * data (i.e. data executed once per element)
   *
   * @tparam SolidFormulation
   */
  template <typename SolidFormulation>
  constexpr bool has_preparation_data = SolidFormulation::has_preparation_data;

  /*!
   * @brief a trait for solid formulation determining whether the formulation has a contribution of
   * condensed dofs (e.g. for EAS)
   *
   * @tparam SolidFormulation
   */
  template <typename SolidFormulation>
  constexpr bool has_condensed_contribution = SolidFormulation::has_condensed_contribution;

  namespace Internal
  {
    /*!
     * @brief A dummy type that is used if a solid formulation does not need preparation data
     */
    struct NoneType
    {
    };

    template <typename SolidFormulation>
    struct PreparationTypeTrait;

    template <typename SolidFormulation>
      requires(has_preparation_data<SolidFormulation>)
    struct PreparationTypeTrait<SolidFormulation>
    {
      using type = typename SolidFormulation::PreparationData;
    };

    template <typename SolidFormulation>
      requires(!has_preparation_data<SolidFormulation>)
    struct PreparationTypeTrait<SolidFormulation>
    {
      using type = NoneType;
    };
  }  // namespace Internal

  /*!
   * @brief A type trait that is the type of the Preparation data if SolidFormulation needs to
   * prepare data, otherwise, it is @p Internal::NoneType
   *
   * @tparam SolidFormulation
   */
  template <typename SolidFormulation>
  using PreparationData = typename Internal::PreparationTypeTrait<SolidFormulation>::type;

  /*!
   * @brief An object holding the history data of the solid formulation.
   *
   * Can hold none, gauss point history and global history.
   *
   * @note The data is only stored if needed by the solid formulation
   *
   * @tparam SolidFormulation
   * @tparam T
   */
  template <typename SolidFormulation>
  struct SolidFormulationHistory;

  template <typename SolidFormulation>
    requires(has_global_history<SolidFormulation> && has_gauss_point_history<SolidFormulation>)
  struct SolidFormulationHistory<SolidFormulation>
  {
    typename SolidFormulation::GlobalHistory global_history;
    std::vector<typename SolidFormulation::GaussPointHistory> gp_history;
  };

  template <typename SolidFormulation>
    requires(!has_global_history<SolidFormulation> && has_gauss_point_history<SolidFormulation>)
  struct SolidFormulationHistory<SolidFormulation>
  {
    std::vector<typename SolidFormulation::GaussPointHistory> gp_history;
  };

  template <typename SolidFormulation>
    requires(has_global_history<SolidFormulation> && !has_gauss_point_history<SolidFormulation>)
  struct SolidFormulationHistory<SolidFormulation>
  {
    typename SolidFormulation::GlobalHistory global_history;
  };

  template <typename SolidFormulation>
    requires(!has_global_history<SolidFormulation> && !has_gauss_point_history<SolidFormulation>)
  struct SolidFormulationHistory<SolidFormulation>
  {
  };

  template <typename SolidFormulation>
  void resize_gp_history(SolidFormulationHistory<SolidFormulation>& history_data,
      [[maybe_unused]] const std::size_t num_gps)
  {
    if constexpr (has_gauss_point_history<SolidFormulation>)
    {
      history_data.gp_history.resize(num_gps);
    }
  }

  namespace Internal
  {
    template <typename SolidFormulation>
    auto get_additional_preparation_tuple(const PreparationData<SolidFormulation>& preparation_data)
    {
      if constexpr (has_preparation_data<SolidFormulation>)
        return std::tie(preparation_data);
      else
        return std::tie();
    }

    template <typename SolidFormulation>
    auto get_additional_global_history_tuple(
        SolidFormulationHistory<SolidFormulation>& history_data)
    {
      if constexpr (has_global_history<SolidFormulation>)
        return std::tie(history_data.global_history);
      else
        return std::tie();
    }

    template <typename SolidFormulation>
    auto get_additional_gauss_point_history_tuple(
        SolidFormulationHistory<SolidFormulation>& history_data, [[maybe_unused]] const int gp)
    {
      if constexpr (has_gauss_point_history<SolidFormulation>)
        return std::tie(history_data.gp_history[gp]);
      else
        return std::tie();
    }

    template <typename SolidFormulation>
    auto get_additional_gauss_point_history_tuple(
        SolidFormulationHistory<SolidFormulation>& history_data)
    {
      static_assert(!has_gauss_point_history<SolidFormulation>,
          "The solid formulation has a Gauss point history and, therefore, needs the Gauss point "
          "id!");

      return std::tie();
    }


    template <typename SolidFormulation>
    auto get_additional_tuple(const PreparationData<SolidFormulation>& preparation_data,
        SolidFormulationHistory<SolidFormulation>& history_data, const int gp)
    {
      return std::tuple_cat(get_additional_preparation_tuple<SolidFormulation>(preparation_data),
          get_additional_global_history_tuple<SolidFormulation>(history_data),
          get_additional_gauss_point_history_tuple<SolidFormulation>(history_data, gp));
    }

    template <typename SolidFormulation>
    auto get_additional_tuple(const PreparationData<SolidFormulation>& preparation_data,
        SolidFormulationHistory<SolidFormulation>& history_data)
    {
      return std::tuple_cat(get_additional_preparation_tuple<SolidFormulation>(preparation_data),
          get_additional_global_history_tuple<SolidFormulation>(history_data),
          get_additional_gauss_point_history_tuple<SolidFormulation>(history_data));
    }
  }  // namespace Internal

  /*!
   * @brief Pack the solid formulation history data
   *
   * @note Calls the respective @p pack(...) method for the SolidFormulation if needed by the solid
   * formulation
   *
   * @tparam SolidFormulation
   * @param data (out) : Buffer where the data is packed to
   * @param solid_formulation_history (in) : History data to be packed
   */
  template <typename SolidFormulation>
  void pack(Core::Communication::PackBuffer& data,
      const SolidFormulationHistory<SolidFormulation>& solid_formulation_history)
  {
    if constexpr (has_global_history<SolidFormulation>)
    {
      SolidFormulation::pack(solid_formulation_history.global_history, data);
    }

    if constexpr (has_gauss_point_history<SolidFormulation>)
    {
      data.add_to_pack(solid_formulation_history.gp_history.size());
      for (const auto& item : solid_formulation_history.gp_history)
      {
        SolidFormulation::pack(item, data);
      }
    }
  }

  /*!
   * @brief Unpack the solid formulation history data
   *
   * @note Calls the respective @p unpack(...) method for the SolidFormulation if needed by the
   * solid formulation
   *
   * @tparam SolidFormulation
   * @param position (in/out) : Position to start to unpack (will be incremented)
   * @param data (in) : Data to unpack from
   * @param solid_formulation_history (out) : History data to be unpacked
   */
  template <typename SolidFormulation>
  void unpack(Core::Communication::UnpackBuffer& buffer,
      SolidFormulationHistory<SolidFormulation>& solid_formulation_history)
  {
    if constexpr (has_global_history<SolidFormulation>)
    {
      SolidFormulation::unpack(buffer, solid_formulation_history.global_history);
    }

    if constexpr (has_gauss_point_history<SolidFormulation>)
    {
      std::size_t num_gps;
      extract_from_pack(buffer, num_gps);
      solid_formulation_history.gp_history.resize(num_gps);
      for (auto& item : solid_formulation_history.gp_history)
      {
        SolidFormulation::unpack(buffer, item);
      }
    }
  }

  /*!
   * @brief Calls the @p Prepare(...) on the solid formulation of the solid formulation needs to
   * prepare stuff
   *
   * @tparam SolidFormulation
   * @tparam celltype
   * @param ele (in) : Solid element
   * @param nodal_coordinates (in) : element coordinates
   * @param history_data (in/out) : history data
   * @return PreparationData<SolidFormulation>
   */
  template <typename SolidFormulation, Core::FE::CellType celltype>
  PreparationData<SolidFormulation> prepare(const Core::Elements::Element& ele,
      const ElementNodes<celltype>& nodal_coordinates,
      SolidFormulationHistory<SolidFormulation>& history_data)
  {
    if constexpr (has_preparation_data<SolidFormulation>)
    {
      if constexpr (has_global_history<SolidFormulation>)
        return SolidFormulation::prepare(ele, nodal_coordinates, history_data.global_history);
      else
      {
        return SolidFormulation::prepare(ele, nodal_coordinates);
      }
    }
    else
    {
      // nothing to prepare
      return {};
    }
  }

  /*!
   * @brief The purpose of this function is to evaluate a solid formulation for a given element. It
   * combines all necessary data (element, nodes, local coordinates, shape functions, Jacobian,
   * preparation data, history data, and an evaluator) and calls the evaluate method of the
   * SolidFormulation with these parameters. The use of perfect forwarding ensures that the
   * arguments are passed efficiently, preserving their value categories (lvalue or rvalue).
   *
   * In a more specific context, i.e., for displacement-based nonlinear kinematics element, it
   * evaluates the deformation gradient and Green-Lagrange strain tensor for the solid element
   * formulation and pass them to the Evaluator to compute the internal forces and linearization.
   *
   * @note: This method does not support solid formulation with Gauss point history since the method
   * does not necessarily be called on a Gauss point. If called on a Gauss point, prefer the other
   * @p evaluate() call with the Gauss point id as parameter.
   */
  template <typename SolidFormulation, Core::FE::CellType celltype, typename Evaluator>
  inline auto evaluate(const Core::Elements::Element& ele,
      const ElementNodes<celltype>& element_nodes,
      const Core::LinAlg::Tensor<double, Internal::num_dim<celltype>>& xi,
      const ShapeFunctionsAndDerivatives<celltype>& shape_functions,
      const JacobianMapping<celltype>& jacobian_mapping,
      const PreparationData<SolidFormulation>& preparation_data,
      SolidFormulationHistory<SolidFormulation>& history_data, Evaluator&& evaluator)
  {
    return std::apply([](auto&&... args)
        { return SolidFormulation::evaluate(std::forward<decltype(args)>(args)...); },
        std::tuple_cat(
            std::forward_as_tuple(ele, element_nodes, xi, shape_functions, jacobian_mapping),
            Internal::get_additional_tuple<SolidFormulation>(preparation_data, history_data),
            std::forward_as_tuple(evaluator)));
  }


  /*!
   * @brief Evaluate the deformation gradient and Green-Lagrange strain tensor for the solid element
   * formulation.
   *
   * @note: This method should be preferred if called at a Gauss point since it also supports
   * element formulations with gauss point history.
   */
  template <typename SolidFormulation, Core::FE::CellType celltype, typename Evaluator>
  inline auto evaluate(const Core::Elements::Element& ele,
      const ElementNodes<celltype>& element_nodes,
      const Core::LinAlg::Tensor<double, Core::FE::dim<celltype>>& xi,
      const ShapeFunctionsAndDerivatives<celltype>& shape_functions,
      const JacobianMapping<celltype>& jacobian_mapping,
      const PreparationData<SolidFormulation>& preparation_data,
      SolidFormulationHistory<SolidFormulation>& history_data, const int gp, Evaluator&& evaluator)
  {
    return std::apply([](auto&&... args)
        { return SolidFormulation::evaluate(std::forward<decltype(args)>(args)...); },
        std::tuple_cat(
            std::forward_as_tuple(ele, element_nodes, xi, shape_functions, jacobian_mapping),
            Internal::get_additional_tuple(preparation_data, history_data, gp),
            std::forward_as_tuple(evaluator)));
  }

  /*!
   * @brief Evaluate the derivative of the deformation gradient w.r.t. the displacements
   */
  template <typename SolidFormulation, Core::FE::CellType celltype>
  inline Core::LinAlg::Matrix<9, Core::FE::num_nodes(celltype) * Core::FE::dim<celltype>>
  evaluate_d_deformation_gradient_d_displacements(const Core::Elements::Element& ele,
      const ElementNodes<celltype>& element_nodes,
      const Core::LinAlg::Tensor<double, Internal::num_dim<celltype>>& xi,
      const ShapeFunctionsAndDerivatives<celltype>& shape_functions,
      const JacobianMapping<celltype>& jacobian_mapping,
      const Core::LinAlg::Tensor<double, Internal::num_dim<celltype>, Internal::num_dim<celltype>>&
          deformation_gradient,
      const PreparationData<SolidFormulation>& preparation_data,
      SolidFormulationHistory<SolidFormulation>& history_data)
  {
    if constexpr (has_gauss_point_history<SolidFormulation>)
    {
      FOUR_C_THROW("The Solid element formulation can only be evaluated at the Gauss points.");
    }
    else
    {
      return std::apply(
          [](auto&&... args)
          {
            return SolidFormulation::evaluate_d_deformation_gradient_d_displacements(
                std::forward<decltype(args)>(args)...);
          },
          std::tuple_cat(std::forward_as_tuple(ele, element_nodes, xi, shape_functions,
                             jacobian_mapping, deformation_gradient),
              Internal::get_additional_tuple(preparation_data, history_data)));
    }
  }

  /*!
   * @brief Evaluate the derivative of the deformation gradient w.r.t. the xi
   */
  template <typename SolidFormulation, Core::FE::CellType celltype>
  inline Core::LinAlg::Matrix<9, Core::FE::dim<celltype>> evaluate_d_deformation_gradient_d_xi(
      const Core::Elements::Element& ele, const ElementNodes<celltype>& element_nodes,
      const Core::LinAlg::Tensor<double, Internal::num_dim<celltype>>& xi,
      const ShapeFunctionsAndDerivatives<celltype>& shape_functions,
      const JacobianMapping<celltype>& jacobian_mapping,
      const Core::LinAlg::Tensor<double, Internal::num_dim<celltype>, Internal::num_dim<celltype>>&
          deformation_gradient,
      const PreparationData<SolidFormulation>& preparation_data,
      SolidFormulationHistory<SolidFormulation>& history_data)
  {
    if constexpr (has_gauss_point_history<SolidFormulation>)
    {
      FOUR_C_THROW("The Solid element formulation can only be evaluated at the Gauss points.");
    }
    else
    {
      return std::apply(
          [](auto&&... args)
          {
            return SolidFormulation::evaluate_d_deformation_gradient_d_xi(
                std::forward<decltype(args)>(args)...);
          },
          std::tuple_cat(std::forward_as_tuple(ele, element_nodes, xi, shape_functions,
                             jacobian_mapping, deformation_gradient),
              Internal::get_additional_tuple(preparation_data, history_data)));
    }
  }

  /*!
   * @brief Evaluate the second derivative of the deformation gradient w.r.t. the xi and the
   * displacements
   */
  template <typename SolidFormulation, Core::FE::CellType celltype>
  inline Core::LinAlg::Matrix<9,
      Core::FE::num_nodes(celltype) * Core::FE::dim<celltype> * Core::FE::dim<celltype>>
  evaluate_d_deformation_gradient_d_displacements_d_xi(const Core::Elements::Element& ele,
      const ElementNodes<celltype>& element_nodes,
      const Core::LinAlg::Tensor<double, Internal::num_dim<celltype>>& xi,
      const ShapeFunctionsAndDerivatives<celltype>& shape_functions,
      const JacobianMapping<celltype>& jacobian_mapping,
      const Core::LinAlg::Tensor<double, Internal::num_dim<celltype>, Internal::num_dim<celltype>>&
          deformation_gradient,
      const PreparationData<SolidFormulation>& preparation_data,
      SolidFormulationHistory<SolidFormulation>& history_data)
  {
    if constexpr (has_gauss_point_history<SolidFormulation>)
    {
      FOUR_C_THROW("The Solid element formulation can only be evaluated at the Gauss points.");
    }
    else
    {
      return std::apply(
          [](auto&&... args)
          {
            return SolidFormulation::evaluate_d_deformation_gradient_d_displacements_d_xi(
                std::forward<decltype(args)>(args)...);
          },
          std::tuple_cat(std::forward_as_tuple(ele, element_nodes, xi, shape_functions,
                             jacobian_mapping, deformation_gradient),
              Internal::get_additional_tuple(preparation_data, history_data)));
    }
  }

  /*!
   * @brief Add the internal force vector contribution of the Gauss point to @p force_vector
   */
  template <typename SolidFormulation, Core::FE::CellType celltype>
  static inline void add_internal_force_vector(const JacobianMapping<celltype>& jacobian_mapping,
      const Core::LinAlg::Tensor<double, Core::FE::dim<celltype>, Core::FE::dim<celltype>>& F,
      const typename SolidFormulation::LinearizationContainer& linearization,
      const Stress<celltype>& stress, const double integration_factor,
      const PreparationData<SolidFormulation>& preparation_data,
      SolidFormulationHistory<SolidFormulation>& history_data, const int gp,
      Core::LinAlg::Matrix<Core::FE::num_nodes(celltype) * Core::FE::dim<celltype>, 1>&
          force_vector)
  {
    std::apply([](auto&&... args)
        { SolidFormulation::add_internal_force_vector(std::forward<decltype(args)>(args)...); },
        std::tuple_cat(
            std::forward_as_tuple(jacobian_mapping, F, linearization, stress, integration_factor),
            Internal::get_additional_tuple(preparation_data, history_data, gp),
            std::forward_as_tuple(force_vector)));
  }

  /*!
   * @brief Add stiffness matrix contribution of the Gauss point to @p stiffness_matrix
   */
  template <typename SolidFormulation, Core::FE::CellType celltype>
  static inline void add_stiffness_matrix(const JacobianMapping<celltype>& jacobian_mapping,
      const Core::LinAlg::Tensor<double, Core::FE::dim<celltype>, Core::FE::dim<celltype>>& F,
      const Core::LinAlg::Tensor<double, Internal::num_dim<celltype>>& xi,
      const ShapeFunctionsAndDerivatives<celltype>& shape_functions,
      const typename SolidFormulation::LinearizationContainer& linearization,
      const Stress<celltype>& stress, const double integration_factor,
      const PreparationData<SolidFormulation>& preparation_data,
      SolidFormulationHistory<SolidFormulation>& history_data, const int gp,
      Core::LinAlg::Matrix<Core::FE::num_nodes(celltype) * Core::FE::dim<celltype>,
          Core::FE::num_nodes(celltype) * Core::FE::dim<celltype>>& stiffness_matrix)
  {
    std::apply([](auto&&... args)
        { SolidFormulation::add_stiffness_matrix(std::forward<decltype(args)>(args)...); },
        std::tuple_cat(std::forward_as_tuple(jacobian_mapping, F, xi, shape_functions,
                           linearization, stress, integration_factor),
            Internal::get_additional_tuple(preparation_data, history_data, gp),
            std::forward_as_tuple(stiffness_matrix)));
  }

  /*!
   * @brief Resed any condensed variables before integration. Called before the Gauss-point loop.
   *
   * @tparam SolidFormulation
   * @tparam celltype
   * @param ele
   * @param element_nodes
   * @param preparation_data (in) : Preparation that is forwarded to the solid formulation if needed
   * @param history_data (in/out) : History data that is forwarded to the solid formulation if
   * needed
   */
  template <typename SolidFormulation, Core::FE::CellType celltype>
  static inline void reset_condensed_variable_integration(const Core::Elements::Element& ele,
      const ElementNodes<celltype>& element_nodes,
      const PreparationData<SolidFormulation>& preparation_data,
      SolidFormulationHistory<SolidFormulation>& history_data)
  {
    std::apply(
        [](auto&&... args)
        {
          SolidFormulation::reset_condensed_variable_integration(
              std::forward<decltype(args)>(args)...);
        },
        std::tuple_cat(std::forward_as_tuple(ele, element_nodes),
            Internal::get_additional_tuple(preparation_data, history_data)));
  }

  /*!
   * @brief Perform the integration of any condensed variables.
   *
   * @tparam SolidFormulation
   * @tparam celltype
   * @param linearization
   * @param stress
   * @param integration_factor
   * @param preparation_data (in) : Preparation that is forwarded to the solid formulation if needed
   * @param history_data (in/out) : History data that is forwarded to the solid formulation if
   * @param gp
   */
  template <typename SolidFormulation, Core::FE::CellType celltype>
  static inline void integrate_condensed_contribution(
      const typename SolidFormulation::LinearizationContainer& linearization,
      const Stress<celltype>& stress, const double integration_factor,
      const PreparationData<SolidFormulation>& preparation_data,
      SolidFormulationHistory<SolidFormulation>& history_data, const int gp)
  {
    // only needed if there are condensed variables
    std::apply(
        [](auto&&... args)
        {
          SolidFormulation::integrate_condensed_contribution(std::forward<decltype(args)>(args)...);
        },
        std::tuple_cat(std::forward_as_tuple(linearization, stress, integration_factor),
            Internal::get_additional_tuple(preparation_data, history_data, gp)));
  }

  /*!
   * @brief Evaluate the contributions of condensed variables
   *
   * @tparam SolidFormulation
   * @param preparation_data (in) : Preparation that is forwarded to the solid formulation if needed
   * @param history_data (in/out) : History data that is forwarded to the solid formulation if
   * @return SolidFormulation::CondensedContributionData
   */
  template <typename SolidFormulation>
  static inline typename SolidFormulation::CondensedContributionData prepare_condensed_contribution(
      const PreparationData<SolidFormulation>& preparation_data,
      SolidFormulationHistory<SolidFormulation>& history_data)
  {
    // only needed if there are condensed variables
    return std::apply(
        [](auto&&... args)
        {
          return SolidFormulation::prepare_condensed_contribution(
              std::forward<decltype(args)>(args)...);
        },
        Internal::get_additional_tuple(preparation_data, history_data));
  }

  /*!
   * @brief Update internal condensed variables
   *
   * @tparam SolidFormulation
   * @tparam celltype
   * @param ele
   * @param params_interface
   * @param element_nodes
   * @param displacement_increments
   * @param linesearch_step_length (in) : step length for line search algorithms (default 1.0)
   * @param preparation_data (in) : Preparation that is forwarded to the solid formulation if needed
   * @param history_data (in/out) : History data that is forwarded to the solid formulation if
   */
  template <typename SolidFormulation, Core::FE::CellType celltype>
  static inline void update_condensed_variables(const Core::Elements::Element& ele,
      FourC::Solid::Elements::ParamsInterface* params_interface,
      const ElementNodes<celltype>& element_nodes,
      const Core::LinAlg::Matrix<Internal::num_dof_per_ele<celltype>, 1>& displacement_increments,
      const double linesearch_step_length,
      const PreparationData<SolidFormulation>& preparation_data,
      SolidFormulationHistory<SolidFormulation>& history_data)
  {
    std::apply([](auto&&... args)
        { SolidFormulation::update_condensed_variables(std::forward<decltype(args)>(args)...); },
        std::tuple_cat(std::forward_as_tuple(ele, params_interface, element_nodes,
                           displacement_increments, linesearch_step_length),
            Internal::get_additional_tuple(preparation_data, history_data)));
  }

  /*!
   * @brief Correction of the condensed variables for line search algorithms
   *
   * @tparam SolidFormulation
   * @param ele
   * @param params_interface
   * @param linesearch_step_length (in) : new line search step length
   * @param preparation_data (in) : Preparation that is forwarded to the solid formulation if needed
   * @param history_data (in/out) : History data that is forwarded to the solid formulation if
   */
  template <typename SolidFormulation>
  static inline void correct_condensed_variables_for_linesearch(const Core::Elements::Element& ele,
      FourC::Solid::Elements::ParamsInterface* params_interface,
      const double linesearch_step_length,
      const PreparationData<SolidFormulation>& preparation_data,
      SolidFormulationHistory<SolidFormulation>& history_data)
  {
    std::apply(
        [](auto&&... args)
        {
          SolidFormulation::correct_condensed_variables_for_linesearch(
              std::forward<decltype(args)>(args)...);
        },
        std::tuple_cat(std::forward_as_tuple(ele, params_interface, linesearch_step_length),
            Internal::get_additional_tuple(preparation_data, history_data)));
  }

  /*!
   * @brief Add the contributions of condensed variables to the element force vector
   *
   * @tparam celltype
   * @tparam SolidFormulation
   * @param condensed_contribution_data
   * @param preparation_data (in) : Preparation that is forwarded to the solid formulation if needed
   * @param history_data (in/out) : History data that is forwarded to the solid formulation if
   * @param force_vector
   */
  template <Core::FE::CellType celltype, typename SolidFormulation>
  static inline void add_condensed_contribution_to_force_vector(
      const typename SolidFormulation::CondensedContributionData& condensed_contribution_data,
      const PreparationData<SolidFormulation>& preparation_data,
      SolidFormulationHistory<SolidFormulation>& history_data,
      Core::LinAlg::Matrix<Core::FE::num_nodes(celltype) * Core::FE::dim<celltype>, 1>&
          force_vector)
  {
    std::apply(
        [](auto&&... args)
        {
          SolidFormulation::add_condensed_contribution_to_force_vector(
              std::forward<decltype(args)>(args)...);
        },
        std::tuple_cat(std::forward_as_tuple(condensed_contribution_data),
            Internal::get_additional_tuple(preparation_data, history_data),
            std::forward_as_tuple(force_vector)));
  }

  /*!
   * @brief Add the contributions of condensed variables to the element stiffness matrix
   *
   * @tparam celltype
   * @tparam SolidFormulation
   * @param condensed_contribution_data
   * @param preparation_data (in) : Preparation that is forwarded to the solid formulation if needed
   * @param history_data (in/out) : History data that is forwarded to the solid formulation if
   * @param stiffness_matrix
   */
  template <Core::FE::CellType celltype, typename SolidFormulation>
  static inline void add_condensed_contribution_to_stiffness_matrix(
      const typename SolidFormulation::CondensedContributionData& condensed_contribution_data,
      const PreparationData<SolidFormulation>& preparation_data,
      SolidFormulationHistory<SolidFormulation>& history_data,
      Core::LinAlg::Matrix<Core::FE::num_nodes(celltype) * Core::FE::dim<celltype>,
          Core::FE::num_nodes(celltype) * Core::FE::dim<celltype>>& stiffness_matrix)
  {
    std::apply(
        [](auto&&... args)
        {
          SolidFormulation::add_condensed_contribution_to_stiffness_matrix(
              std::forward<decltype(args)>(args)...);
        },
        std::tuple_cat(std::forward_as_tuple(condensed_contribution_data),
            Internal::get_additional_tuple(preparation_data, history_data),
            std::forward_as_tuple(stiffness_matrix)));
  }

  template <typename SolidFormulation, Core::FE::CellType celltype>
  static inline void update_prestress(const Core::Elements::Element& ele,
      const ElementNodes<celltype>& element_nodes,
      const PreparationData<SolidFormulation>& preparation_data,
      SolidFormulationHistory<SolidFormulation>& history_data)
  {
    if constexpr (has_global_history<SolidFormulation>)
    {
      std::apply([](auto&&... args)
          { SolidFormulation::update_prestress(std::forward<decltype(args)>(args)...); },
          std::tuple_cat(std::forward_as_tuple(ele, element_nodes),
              Internal::get_additional_preparation_tuple<SolidFormulation>(preparation_data),
              Internal::get_additional_global_history_tuple<SolidFormulation>(history_data)));
    }
  }

  template <typename SolidFormulation, Core::FE::CellType celltype>
  static inline void update_prestress(const Core::Elements::Element& ele,
      const ElementNodes<celltype>& element_nodes,
      const Core::LinAlg::Tensor<double, Internal::num_dim<celltype>>& xi,
      const ShapeFunctionsAndDerivatives<celltype>& shape_functions,
      const JacobianMapping<celltype>& jacobian_mapping,
      const Core::LinAlg::Tensor<double, Internal::num_dim<celltype>, Internal::num_dim<celltype>>&
          deformation_gradient,
      const PreparationData<SolidFormulation>& preparation_data,
      SolidFormulationHistory<SolidFormulation>& history_data, [[maybe_unused]] const int gp)
  {
    std::apply([](auto&&... args)
        { SolidFormulation::update_prestress(std::forward<decltype(args)>(args)...); },
        std::tuple_cat(std::forward_as_tuple(ele, element_nodes, xi, shape_functions,
                           jacobian_mapping, deformation_gradient),
            Internal::get_additional_tuple(preparation_data, history_data, gp)));
  }


}  // namespace Discret::Elements

FOUR_C_NAMESPACE_CLOSE

#endif

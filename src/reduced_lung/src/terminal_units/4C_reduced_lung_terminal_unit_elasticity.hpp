// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_REDUCED_LUNG_TERMINAL_UNIT_ELASTICITY_HPP
#define FOUR_C_REDUCED_LUNG_TERMINAL_UNIT_ELASTICITY_HPP

#include "4C_config.hpp"

#include "4C_reduced_lung_input.hpp"
#include "4C_reduced_lung_terminal_unit_common.hpp"
#include "4C_utils_exceptions.hpp"

#include <functional>
#include <utility>
#include <variant>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace ReducedLung::TerminalUnits
{
  /**
   * @brief Linear elasticity data used by terminal-unit spring response.
   */
  struct LinearElasticity
  {
    std::vector<double> elasticity_E;
    std::vector<double> elastic_pressure_p_el;
    std::vector<double> elastic_pressure_grad_dp_el;
  };

  /**
   * @brief Ogden hyperelasticity data used by terminal-unit spring response.
   */
  struct OgdenHyperelasticity
  {
    std::vector<double> bulk_modulus_kappa;
    std::vector<double> nonlinear_stiffening_beta;
    std::vector<double> elastic_pressure_p_el;
    std::vector<double> elastic_pressure_grad_dp_el;
  };

  /**
   * @brief Variant containing all supported terminal-unit elasticity model data structs.
   */
  using ElasticityModel = std::variant<LinearElasticity, OgdenHyperelasticity>;
}  // namespace ReducedLung::TerminalUnits

namespace ReducedLung::TerminalUnits::Elasticity
{
  /**
   * @brief Enum type used in reduced-lung input for elasticity model selection.
   */
  using ElasticityModelType =
      ReducedLungParameters::LungTree::TerminalUnits::ElasticityModel::ElasticityModelType;

  /**
   * @brief Human-readable name for elasticity enum values.
   */
  inline const char* elasticity_model_name(const ElasticityModelType elasticity_model_type)
  {
    switch (elasticity_model_type)
    {
      case ElasticityModelType::Linear:
        return "Linear";
      case ElasticityModelType::Ogden:
        return "Ogden";
    }
    FOUR_C_THROW("Unknown elasticity model type enum value.");
  }

  /**
   * @brief Dispatch an elasticity enum value to its concrete C++ model type.
   *
   * The callable must provide templated overloads via
   * `callable.template operator()<LinearElasticity>()` and
   * `callable.template operator()<OgdenHyperelasticity>()`.
   */
  template <typename Callable>
  void dispatch_elasticity_model_type(
      const ElasticityModelType elasticity_model_type, Callable&& callable)
  {
    switch (elasticity_model_type)
    {
      case ElasticityModelType::Linear:
        std::forward<Callable>(callable).template operator()<LinearElasticity>();
        return;
      case ElasticityModelType::Ogden:
        std::forward<Callable>(callable).template operator()<OgdenHyperelasticity>();
        return;
    }
    FOUR_C_THROW("Unknown elasticity model type enum value.");
  }

  /**
   * @brief Evaluates elastic pressure for one terminal-unit model block.
   */
  using ElasticPressureEvaluator = std::function<std::vector<double>&(
      TerminalUnitData&, const Core::LinAlg::Vector<double>&, double)>;

  /**
   * @brief Evaluates derivative of elastic pressure w.r.t. q for one model block.
   */
  using ElasticPressureGradientEvaluator = std::function<std::vector<double>&(
      TerminalUnitData&, const Core::LinAlg::Vector<double>&, double)>;

  /**
   * @brief Build pressure evaluator callback for the concrete elasticity variant in
   * @p elasticity_model.
   */
  ElasticPressureEvaluator make_elastic_pressure_evaluator(ElasticityModel& elasticity_model);

  /**
   * @brief Build pressure-gradient evaluator callback for the concrete elasticity variant.
   */
  ElasticPressureGradientEvaluator make_elastic_pressure_gradient_evaluator(
      ElasticityModel& elasticity_model);

  /**
   * @brief Build output evaluator callback for the concrete elasticity variant.
   */
  OutputEvaluator make_output_evaluator(ElasticityModel& elasticity_model);

  /**
   * @brief Append element parameters and initialize internal vectors for a concrete elasticity
   * model.
   */
  void append_model_parameters(ElasticityModel& elasticity_model, int global_element_id,
      const ReducedLungParameters::LungTree::TerminalUnits::ElasticityModel& parameters);
}  // namespace ReducedLung::TerminalUnits::Elasticity

FOUR_C_NAMESPACE_CLOSE

#endif

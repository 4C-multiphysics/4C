// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_REDUCED_LUNG_TERMINAL_UNIT_RHEOLOGY_HPP
#define FOUR_C_REDUCED_LUNG_TERMINAL_UNIT_RHEOLOGY_HPP

#include "4C_config.hpp"

#include "4C_reduced_lung_input.hpp"
#include "4C_reduced_lung_terminal_unit_common.hpp"
#include "4C_reduced_lung_terminal_unit_elasticity.hpp"
#include "4C_utils_exceptions.hpp"

#include <utility>
#include <variant>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace ReducedLung::TerminalUnits
{
  /**
   * @brief Kelvin-Voigt rheology data (parallel spring-dashpot branch).
   */
  struct KelvinVoigt
  {
    std::vector<double> viscosity_eta;
  };

  /**
   * @brief Four-element Maxwell rheology data.
   *
   * Stores the Kelvin-Voigt branch viscosity and Maxwell branch parameters plus the history
   * variable for Maxwell pressure.
   */
  struct FourElementMaxwell
  {
    std::vector<double> viscosity_eta;
    std::vector<double> elasticity_E_m;
    std::vector<double> viscosity_eta_m;
    std::vector<double> maxwell_pressure_p_m;
  };

  /**
   * @brief Variant containing all supported terminal-unit rheology model data structs.
   */
  using RheologicalModel = std::variant<KelvinVoigt, FourElementMaxwell>;
}  // namespace ReducedLung::TerminalUnits

namespace ReducedLung::TerminalUnits::Rheology
{
  /**
   * @brief Enum type used in reduced-lung input for rheological model selection.
   */
  using RheologicalModelType =
      ReducedLungParameters::LungTree::TerminalUnits::RheologicalModel::RheologicalModelType;

  /**
   * @brief Human-readable name for rheology enum values.
   */
  inline const char* rheological_model_name(const RheologicalModelType rheological_model_type)
  {
    switch (rheological_model_type)
    {
      case RheologicalModelType::KelvinVoigt:
        return "KelvinVoigt";
      case RheologicalModelType::FourElementMaxwell:
        return "FourElementMaxwell";
    }
    FOUR_C_THROW("Unknown rheological model type enum value.");
  }

  /**
   * @brief Dispatch a rheology enum value to its concrete C++ model type.
   *
   * The callable must provide templated overloads via
   * `callable.template operator()<KelvinVoigt>()` and
   * `callable.template operator()<FourElementMaxwell>()`.
   */
  template <typename Callable>
  void dispatch_rheological_model_type(
      const RheologicalModelType rheological_model_type, Callable&& callable)
  {
    switch (rheological_model_type)
    {
      case RheologicalModelType::KelvinVoigt:
        std::forward<Callable>(callable).template operator()<KelvinVoigt>();
        return;
      case RheologicalModelType::FourElementMaxwell:
        std::forward<Callable>(callable).template operator()<FourElementMaxwell>();
        return;
    }
    FOUR_C_THROW("Unknown rheological model type enum value.");
  }

  /**
   * @brief Build residual evaluator callback for the concrete rheology variant.
   */
  ResidualEvaluator make_residual_evaluator(
      RheologicalModel& rheological_model, Elasticity::ElasticPressureEvaluator pressure_evaluator);

  /**
   * @brief Build Jacobian evaluator callback for the concrete rheology variant.
   */
  JacobianEvaluator make_jacobian_evaluator(RheologicalModel& rheological_model,
      Elasticity::ElasticPressureGradientEvaluator pressure_gradient_evaluator);

  /**
   * @brief Build internal-state updater callback for the concrete rheology variant.
   */
  InternalStateUpdater make_internal_state_updater(RheologicalModel& rheological_model);

  /**
   * @brief Build end-of-timestep callback for the concrete rheology variant.
   */
  EndOfTimestepRoutine make_end_of_timestep_routine(RheologicalModel& rheological_model);

  /**
   * @brief Build output evaluator callback for the concrete rheology variant.
   */
  OutputEvaluator make_output_evaluator(RheologicalModel& rheological_model);

  /**
   * @brief Append element parameters and initialize internal vectors for a concrete rheology model.
   */
  void append_model_parameters(RheologicalModel& rheological_model, int global_element_id,
      const ReducedLungParameters::LungTree::TerminalUnits::RheologicalModel& parameters);
}  // namespace ReducedLung::TerminalUnits::Rheology

FOUR_C_NAMESPACE_CLOSE

#endif

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_REDUCED_LUNG_AIRWAYS_FLOW_RESISTANCE_HPP
#define FOUR_C_REDUCED_LUNG_AIRWAYS_FLOW_RESISTANCE_HPP

#include "4C_config.hpp"

#include "4C_reduced_lung_airways_common.hpp"
#include "4C_reduced_lung_input.hpp"
#include "4C_utils_exceptions.hpp"

#include <functional>
#include <span>
#include <utility>
#include <variant>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace ReducedLung::Airways
{
  /**
   * @brief Linear resistive airway flow model data.
   */
  struct LinearResistive
  {
    std::vector<bool> has_inertia;
  };

  /**
   * @brief Nonlinear resistive airway flow model data. Model of van Ertbruggen (2004).
   */
  struct NonLinearResistive
  {
    std::vector<double> turbulence_factor_gamma;
    std::vector<bool> has_inertia;
    std::vector<double> k_turb;
  };

  /**
   * @brief Variant containing all supported airway flow resistance models.
   */
  using FlowModel = std::variant<LinearResistive, NonLinearResistive>;

  // Forward declaration: defined in wall mechanics module.
  struct KelvinVoigtWall;
}  // namespace ReducedLung::Airways

namespace ReducedLung::Airways::FlowResistance
{
  /**
   * @brief Enum type used in reduced-lung input for airway flow-model selection.
   */
  using FlowModelType = ReducedLungParameters::LungTree::Airways::FlowModel::ResistanceType;

  /**
   * @brief Readable name for airway flow-model enum values.
   */
  inline const char* flow_model_name(const FlowModelType flow_model_type)
  {
    switch (flow_model_type)
    {
      case FlowModelType::Linear:
        return "Linear";
      case FlowModelType::NonLinear:
        return "NonLinear";
    }
    FOUR_C_THROW("Unknown airway flow model type enum value.");
  }

  /**
   * @brief Dispatch a flow-model enum value to its concrete C++ model type.
   */
  template <typename Callable>
  void dispatch_flow_model_type(const FlowModelType flow_model_type, Callable&& callable)
  {
    switch (flow_model_type)
    {
      case FlowModelType::Linear:
        std::forward<Callable>(callable).template operator()<LinearResistive>();
        return;
      case FlowModelType::NonLinear:
        std::forward<Callable>(callable).template operator()<NonLinearResistive>();
        return;
    }
    FOUR_C_THROW("Unknown airway flow model type enum value.");
  }

  /**
   * @brief Compile-time number of flow state equations per airway element for model type @p F.
   */
  template <typename F>
  struct FlowModelStateCount
  {
  };

  template <>
  struct FlowModelStateCount<LinearResistive>
  {
    static constexpr int value = 1;
  };

  template <>
  struct FlowModelStateCount<NonLinearResistive>
  {
    static constexpr int value = 1;
  };

  /**
   * @brief Evaluator type for inertia contributions.
   */
  using InertiaEvaluator = std::function<void(
      const AirwayData&, const std::vector<double>& area, std::span<double> inertia)>;

  /**
   * @brief Evaluator type for flow-resistance values.
   *
   * The evaluator writes into caller-owned scratch storage to avoid temporary allocations during
   * repeated residual/Jacobian/tree assembly.
   */
  using FlowResistanceEvaluator = std::function<void(const AirwayData&,
      const Core::LinAlg::Vector<double>&, const std::vector<double>&, std::span<double>)>;

  /**
   * @brief Evaluator type for flow-resistance derivative in rigid-wall equations.
   *
   * Results are written into the supplied span and later used as dynamic q-coefficients in sparse
   * and structured airway linearizations.
   */
  using FlowResistanceDerivativeEvaluatorRigid = std::function<void(
      const AirwayData&, const Core::LinAlg::Vector<double>&, double, std::span<double>)>;

  /**
   * @brief Evaluator type for flow-resistance derivatives in Kelvin-Voigt wall equations.
   *
   * Writes q1 and q2 derivatives into caller-owned spans for batched structured coefficient
   * replacement.
   */
  using FlowResistanceDerivativeEvaluatorKelvinVoigt = std::function<void(const AirwayData&,
      const Core::LinAlg::Vector<double>&, double, std::span<double>, std::span<double>)>;

  /**
   * @brief Evaluator type for inertia derivatives in Kelvin-Voigt wall equations.
   *
   * Writes q1 and q2 inertia derivatives into caller-owned spans for sparse and structured airway
   * linearization assembly.
   */
  using InertiaDerivativeEvaluatorKelvinVoigt = std::function<void(const AirwayData&,
      const Core::LinAlg::Vector<double>&, double, std::span<double>, std::span<double>)>;

  /**
   * @brief Callback type for flow-model internal state updates.
   */
  using InternalStateUpdaterFlowModel = FlowModelInternalStateUpdater;

  /**
   * @brief Build inertia evaluator callback for the concrete flow-model variant.
   */
  InertiaEvaluator make_inertia_evaluator(FlowModel& flow_model);

  /**
   * @brief Build rigid-wall flow-resistance evaluator callback for the concrete flow-model variant.
   */
  FlowResistanceEvaluator make_flow_resistance_evaluator_rigid(FlowModel& flow_model);

  /**
   * @brief Build Kelvin-Voigt flow-resistance evaluator callback for the concrete flow-model
   * variant.
   */
  FlowResistanceEvaluator make_flow_resistance_evaluator_kelvin_voigt(FlowModel& flow_model);

  /**
   * @brief Build rigid-wall flow-resistance-derivative evaluator callback for the concrete
   * flow-model variant.
   */
  FlowResistanceDerivativeEvaluatorRigid make_flow_resistance_derivative_evaluator_rigid(
      FlowModel& flow_model);

  /**
   * @brief Build Kelvin-Voigt flow-resistance-derivative evaluator callback for the concrete
   * flow-model variant.
   */
  FlowResistanceDerivativeEvaluatorKelvinVoigt
  make_flow_resistance_derivative_evaluator_kelvin_voigt(
      FlowModel& flow_model, KelvinVoigtWall& kelvin_voigt_wall_model);

  /**
   * @brief Build Kelvin-Voigt inertia-derivative evaluator callback for the concrete flow-model
   * variant.
   */
  InertiaDerivativeEvaluatorKelvinVoigt make_inertia_derivative_evaluator_kelvin_voigt(
      FlowModel& flow_model, KelvinVoigtWall& kelvin_voigt_wall_model);

  /**
   * @brief Build internal-state updater callback for the concrete flow-model variant.
   */
  InternalStateUpdaterFlowModel make_internal_state_updater(FlowModel& flow_model);

  /**
   * @brief Build output evaluator callback for the concrete flow-model variant.
   */
  OutputEvaluator make_output_evaluator(FlowModel& flow_model);

  /**
   * @brief Append element parameters and initialize vectors for a concrete flow model.
   */
  void append_model_parameters(FlowModel& flow_model, int global_element_id,
      const ReducedLungParameters::LungTree::Airways::FlowModel& parameters);
}  // namespace ReducedLung::Airways::FlowResistance

FOUR_C_NAMESPACE_CLOSE

#endif

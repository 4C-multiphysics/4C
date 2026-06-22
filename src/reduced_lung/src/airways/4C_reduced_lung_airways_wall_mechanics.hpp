// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_REDUCED_LUNG_AIRWAYS_WALL_MECHANICS_HPP
#define FOUR_C_REDUCED_LUNG_AIRWAYS_WALL_MECHANICS_HPP

#include "4C_config.hpp"

#include "4C_reduced_lung_airways_common.hpp"
#include "4C_reduced_lung_airways_flow_resistance.hpp"
#include "4C_reduced_lung_input.hpp"
#include "4C_utils_exceptions.hpp"

#include <utility>
#include <variant>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace ReducedLung::Airways
{
  /**
   * @brief Rigid airway wall model data.
   */
  struct RigidWall
  {
  };

  /**
   * @brief Kelvin-Voigt airway wall model data based on dissertation M. Ismail (2014).
   */
  struct KelvinVoigtWall
  {
    std::vector<double> wall_poisson_ratio;
    std::vector<double> wall_elasticity;
    std::vector<double> wall_thickness;
    std::vector<double> viscous_time_constant;
    std::vector<double> viscous_phase_shift;
    std::vector<double> area_n;

    std::vector<double> area;
    std::vector<double> viscous_resistance_Rvisc;
    std::vector<double> compliance_C;
    std::vector<double> gamma_w;
    std::vector<double> beta_w;
  };

  /**
   * @brief Variant containing all supported airway wall models.
   */
  using WallModel = std::variant<RigidWall, KelvinVoigtWall>;
}  // namespace ReducedLung::Airways

namespace ReducedLung::Airways::WallMechanics
{
  /**
   * @brief Enum type used in reduced-lung input for airway wall-model selection.
   */
  using WallModelType = ReducedLungParameters::LungTree::Airways::WallModelType;

  /**
   * @brief Human-readable name for airway wall-model enum values.
   */
  inline const char* wall_model_name(const WallModelType wall_model_type)
  {
    switch (wall_model_type)
    {
      case WallModelType::Rigid:
        return "Rigid";
      case WallModelType::KelvinVoigt:
        return "KelvinVoigt";
    }
    FOUR_C_THROW("Unknown airway wall model type enum value.");
  }

  /**
   * @brief Dispatch a wall-model enum value to its concrete C++ model type.
   */
  template <typename Callable>
  void dispatch_wall_model_type(const WallModelType wall_model_type, Callable&& callable)
  {
    switch (wall_model_type)
    {
      case WallModelType::Rigid:
        std::forward<Callable>(callable).template operator()<RigidWall>();
        return;
      case WallModelType::KelvinVoigt:
        std::forward<Callable>(callable).template operator()<KelvinVoigtWall>();
        return;
    }
    FOUR_C_THROW("Unknown airway wall model type enum value.");
  }

  /**
   * @brief Compile-time number of wall state equations per airway element for model type @p W.
   */
  template <typename W>
  struct WallModelStateCount
  {
  };

  template <>
  struct WallModelStateCount<RigidWall>
  {
    static constexpr int value = 0;
  };

  template <>
  struct WallModelStateCount<KelvinVoigtWall>
  {
    static constexpr int value = 1;
  };

  /**
   * @brief Build residual evaluator callback for the concrete wall-model variant.
   */
  ResidualEvaluator make_residual_evaluator(WallModel& wall_model, FlowModel& flow_model);

  /**
   * @brief Build Jacobian evaluator callback for the concrete wall-model variant.
   */
  JacobianEvaluator make_jacobian_evaluator(WallModel& wall_model, FlowModel& flow_model);

  /**
   * @brief Build internal-state updater callback for the concrete wall-model variant.
   */
  InternalStateUpdater make_internal_state_updater(
      WallModel& wall_model, FlowModelInternalStateUpdater flow_state_updater);

  /**
   * @brief Build output evaluator callback for the concrete wall-model variant.
   */
  OutputEvaluator make_output_evaluator(WallModel& wall_model);

  /**
   * @brief Build end-of-timestep callback for the concrete wall-model variant.
   */
  EndOfTimestepRoutine make_end_of_timestep_routine(WallModel& wall_model);

  /**
   * @brief Append element parameters and initialize vectors for a concrete wall model.
   */
  void append_model_parameters(WallModel& wall_model, int global_element_id,
      const ReducedLungParameters::LungTree::Airways::WallModel& parameters, double ref_area);
}  // namespace ReducedLung::Airways::WallMechanics

FOUR_C_NAMESPACE_CLOSE

#endif

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_REDUCED_LUNG_TERMINAL_UNIT_RECRUITMENT_HPP
#define FOUR_C_REDUCED_LUNG_TERMINAL_UNIT_RECRUITMENT_HPP

#include "4C_config.hpp"

#include "4C_reduced_lung_terminal_unit_common.hpp"
#include "4C_utils_exceptions.hpp"

#include <utility>
#include <variant>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace ReducedLung::TerminalUnits::Recruitment
{
  /**
   * @brief Enum types used in reduced-lung input for recruitment model selection.
   */
  using PressureLawType =
      ReducedLungParameters::LungTree::TerminalUnits::RecruitmentModel::PressureLawType;
  using TimeLawType = ReducedLungParameters::LungTree::TerminalUnits::RecruitmentModel::TimeLawType;
  using HysteresisPath =
      ReducedLungParameters::LungTree::TerminalUnits::RecruitmentModel::HysteresisPath;
  using ReferenceVolumeLinearization = ReducedLungParameters::LungTree::TerminalUnits::
      RecruitmentModel::ReferenceVolumeLinearization;

  /**
   * @brief Terminal units whose reference volume is the geometry-derived constant.
   */
  struct NoRecruitment
  {
    ///< Reference volume of each element, derived from the geometry and never changed.
    std::vector<double> v0;
  };

  /**
   * @brief Element-wise relaxation of the reference volume towards its quasi-static target.
   *
   * The time law composes with whichever pressure law drives the target, so it is kept separate
   * from the pressure law parameters.
   */
  struct TimeLaw
  {
    ///< Time law delaying the reference volume response, applied on top of the pressure law.
    std::vector<TimeLawType> type;
    ///< Relaxation time of the exponential relaxation time law; 0 without that time law.
    std::vector<double> tau;
  };

  /**
   * @brief Piecewise-linear pressure law on an opening and a closing hysteresis branch.
   */
  struct LinearPressureLaw
  {
    ///< Hysteresis branch of the last converged time step, initialized from the input field.
    std::vector<HysteresisPath> active_path_n;

    ///< Reference volume of the fully derecruited and the fully recruited element.
    std::vector<double> v0_min;
    std::vector<double> v0_max;
    ///< Transmural pressure at which recruitment sets in on the closing and the opening branch.
    std::vector<double> p_closing_min;
    std::vector<double> p_opening_min;
    ///< Pressure interval over which the reference volume sweeps from v0_min to v0_max.
    std::vector<double> delta_p_minmax;
    ///< Relative distance to v0_min/v0_max at which the hysteresis branch is switched.
    std::vector<double> epsilon_v0_switch;
  };

  /**
   * @brief Recruitment data for a block of terminal units on the piecewise-linear pressure law.
   *
   * Recruitment turns the reference volume into a state variable driven by the transmural
   * pressure: the pressure law sets a quasi-static target, the time law relaxes towards it. The
   * state vectors hold the last converged time step, denoted by the suffix `_n`; the value of the
   * new time step only exists within update_recruitment_state().
   */
  struct LinearPressureRecruitment
  {
    LinearPressureLaw pressure_law;
    TimeLaw time_law;

    ///< Whether the reference volume derivative enters the Jacobian or is dropped.
    std::vector<ReferenceVolumeLinearization> reference_volume_linearization;

    ///< Reference volume of the last converged time step.
    std::vector<double> v0_n;
    ///< Quasi-static reference volume the time law relaxes towards. Output only; seeded with the
    ///< initial reference volume until the first time step advances it.
    std::vector<double> v0_target;
  };

  /**
   * @brief Variant containing all supported terminal-unit recruitment model data structs.
   */
  using RecruitmentModel = std::variant<NoRecruitment, LinearPressureRecruitment>;

  /**
   * @brief Human-readable name for recruitment pressure-law enum values.
   */
  inline const char* pressure_law_name(const PressureLawType pressure_law_type)
  {
    switch (pressure_law_type)
    {
      case PressureLawType::None:
        return "None";
      case PressureLawType::LinearPressure:
        return "LinearPressure";
    }
    FOUR_C_THROW("Unknown recruitment pressure-law type enum value.");
  }

  /**
   * @brief Dispatch a recruitment pressure-law enum value to its concrete C++ model type.
   *
   * The callable must provide templated overloads via
   * `callable.template operator()<NoRecruitment>()` and
   * `callable.template operator()<LinearPressureRecruitment>()`.
   */
  template <typename Callable>
  void dispatch_pressure_law_type(const PressureLawType pressure_law_type, Callable&& callable)
  {
    switch (pressure_law_type)
    {
      case PressureLawType::None:
        std::forward<Callable>(callable).template operator()<NoRecruitment>();
        return;
      case PressureLawType::LinearPressure:
        std::forward<Callable>(callable).template operator()<LinearPressureRecruitment>();
        return;
    }
    FOUR_C_THROW("Unknown recruitment pressure-law type enum value.");
  }

  /**
   * @brief Reference volume of one element at the last converged time step.
   *
   * The `_n` suffix matches the state vectors this reads. Blocks without a recruitment law
   * report their constant geometry-derived value.
   */
  [[nodiscard]] double reference_volume_n(
      const RecruitmentModel& recruitment_model, size_t element_index);

  /**
   * @brief Effective reference volume of one element and its pressure derivatives.
   *
   * Under the frozen linearization the reference volume stays at the last converged value and the
   * derivatives vanish; under the coupled one it follows the recruitment law within the Newton
   * step, exactly as update_recruitment_state() will advance it. Blocks without a recruitment law
   * always return the constant geometry-derived reference volume.
   */
  ReferenceVolumeContext evaluate_recruitment_context(const TerminalUnitData& data,
      const RecruitmentModel& recruitment_model,
      const Core::LinAlg::Vector<double>& locally_relevant_dofs, size_t element_index, double dt);

  /**
   * @brief Advance reference volume and hysteresis branch of one model block by one time step.
   */
  void update_recruitment_state(const TerminalUnitData& data, RecruitmentModel& recruitment_model,
      const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt);

  /**
   * @brief Append element parameters and initialize the recruitment state vectors.
   */
  void append_model_parameters(RecruitmentModel& recruitment_model, int global_element_id,
      double geometry_volume,
      const ReducedLungParameters::LungTree::TerminalUnits::RecruitmentModel& parameters);

  /**
   * @brief Build output evaluator callback for the concrete recruitment variant.
   *
   * Blocks without a recruitment law contribute no output fields.
   */
  OutputEvaluator make_output_evaluator(const RecruitmentModel& recruitment_model);
}  // namespace ReducedLung::TerminalUnits::Recruitment

FOUR_C_NAMESPACE_CLOSE

#endif

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_reduced_lung_terminal_unit_recruitment.hpp"

#include "4C_reduced_lung_helpers.hpp"
#include "4C_utils_exceptions.hpp"

#include <cmath>

FOUR_C_NAMESPACE_OPEN

namespace ReducedLung::TerminalUnits::Recruitment
{
  namespace
  {
    /**
     * Quasi-static recruitment target of one element.
     */
    struct TargetContext
    {
      double v0_target = 0.0;
    };

    /**
     * Transmural pressure across one terminal unit.
     */
    [[nodiscard]] double transmural_pressure(const TerminalUnitData& data,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs, const size_t element_index)
    {
      return locally_relevant_dofs.local_values_as_span()[data.lid_p1[element_index]] -
             locally_relevant_dofs.local_values_as_span()[data.lid_p2[element_index]];
    }

    /**
     * Evaluate the piecewise-linear pressure law on the active hysteresis branch.
     */
    [[nodiscard]] TargetContext evaluate_linear_pressure_law(const LinearPressureLaw& pressure_law,
        const size_t element_index, const double transmural_pressure_p_tm)
    {
      const double v0_min = pressure_law.v0_min[element_index];
      const double v0_max = pressure_law.v0_max[element_index];
      const double p_min = pressure_law.active_path_n[element_index] == HysteresisPath::Opening
                               ? pressure_law.p_opening_min[element_index]
                               : pressure_law.p_closing_min[element_index];
      const double p_max = p_min + pressure_law.delta_p_minmax[element_index];

      if (transmural_pressure_p_tm <= p_min) return {.v0_target = v0_min};
      if (transmural_pressure_p_tm >= p_max) return {.v0_target = v0_max};

      const double slope = (v0_max - v0_min) / pressure_law.delta_p_minmax[element_index];
      return {.v0_target = v0_min + (transmural_pressure_p_tm - p_min) * slope};
    }

    /**
     * Relax the reference volume of the last converged step towards the quasi-static target.
     *
     * Composes with any pressure law: it only sees the target and the state it relaxes from.
     */
    [[nodiscard]] double apply_time_law(const TimeLaw& time_law, const size_t element_index,
        const TargetContext& target, const double v0_n, const double dt)
    {
      switch (time_law.type[element_index])
      {
        case TimeLawType::None:
          return target.v0_target;
        case TimeLawType::ExponentialRelaxation:
        {
          const double relaxation = std::exp(-dt / time_law.tau[element_index]);
          return relaxation * v0_n + (1.0 - relaxation) * target.v0_target;
        }
      }
      FOUR_C_THROW("Unknown recruitment time-law type enum value.");
    }

    /**
     * Reference volume of the new time step: the pressure law target, relaxed by the time law and
     * kept inside [v0_min, v0_max].
     */
    [[nodiscard]] double evaluate_recruitment_law(const TerminalUnitData& data,
        const LinearPressureRecruitment& recruitment_model,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs, const size_t element_index,
        const double dt, TargetContext& target)
    {
      target = evaluate_linear_pressure_law(recruitment_model.pressure_law, element_index,
          transmural_pressure(data, locally_relevant_dofs, element_index));
      const double v0_next = apply_time_law(recruitment_model.time_law, element_index, target,
          recruitment_model.v0_n[element_index], dt);

      const double v0_min = recruitment_model.pressure_law.v0_min[element_index];
      const double v0_max = recruitment_model.pressure_law.v0_max[element_index];
      if (v0_next <= v0_min) return v0_min;
      if (v0_next >= v0_max) return v0_max;
      return v0_next;
    }

    /**
     * Switch the hysteresis branch once an element is almost fully recruited or derecruited.
     */
    [[nodiscard]] HysteresisPath maybe_switch_path(const HysteresisPath active_path,
        const double v0_next, const double v0_min, const double v0_max,
        const double epsilon_v0_switch)
    {
      const double v0_range = v0_max - v0_min;
      const double full_closed_threshold = v0_min + epsilon_v0_switch * v0_range;
      const double full_open_threshold = v0_max - epsilon_v0_switch * v0_range;

      if (active_path == HysteresisPath::Opening && v0_next >= full_open_threshold)
      {
        return HysteresisPath::Closing;
      }
      if (active_path == HysteresisPath::Closing && v0_next <= full_closed_threshold)
      {
        return HysteresisPath::Opening;
      }
      return active_path;
    }

    /**
     * Check the linear-pressure recruitment input parameters of one element for consistency.
     */
    void validate_linear_pressure_parameters(const TimeLawType time_law_type, const double v0_min,
        const double v0_max, const double initial_v0, const double p_closing_min,
        const double p_opening_min, const double delta_p_minmax, const double epsilon_v0_switch,
        const double tau, const int global_element_id)
    {
      if (!(v0_min > 0.0))
      {
        FOUR_C_THROW(
            "Terminal unit {}: recruitment parameter 'v0_min' must be > 0.", global_element_id);
      }
      if (!(v0_max > v0_min))
      {
        FOUR_C_THROW("Terminal unit {}: recruitment parameter 'v0_max' must be > 'v0_min'.",
            global_element_id);
      }
      if (!(initial_v0 >= v0_min && initial_v0 <= v0_max))
      {
        FOUR_C_THROW(
            "Terminal unit {}: recruitment parameter 'initial_v0' must be in [v0_min, v0_max].",
            global_element_id);
      }
      if (!(delta_p_minmax > 0.0))
      {
        FOUR_C_THROW("Terminal unit {}: recruitment parameter 'delta_p_minmax' must be > 0.",
            global_element_id);
      }
      if (!(p_opening_min >= p_closing_min))
      {
        FOUR_C_THROW(
            "Terminal unit {}: recruitment parameter 'p_opening_min' must be >= 'p_closing_min'.",
            global_element_id);
      }
      if (!(epsilon_v0_switch >= 0.0 && epsilon_v0_switch < 0.5))
      {
        FOUR_C_THROW(
            "Terminal unit {}: recruitment parameter 'epsilon_v0_switch' must be in [0, 0.5).",
            global_element_id);
      }

      if (time_law_type == TimeLawType::ExponentialRelaxation && !(tau > 0.0))
      {
        FOUR_C_THROW(
            "Terminal unit {}: recruitment parameter 'tau' must be > 0 for exponential "
            "relaxation time law.",
            global_element_id);
      }
    }

    /**
     * Append one element to the linear-pressure recruitment block.
     */
    void append_linear_pressure_parameters(LinearPressureRecruitment& recruitment_model,
        const int global_element_id,
        const ReducedLungParameters::LungTree::TerminalUnits::RecruitmentModel& parameters)
    {
      const TimeLawType time_law_type =
          parameters.time_law_type.at(global_element_id, "time_law_type");
      const double v0_min = parameters.linear_pressure.v0_min.at(global_element_id, "v0_min");
      const double v0_max = parameters.linear_pressure.v0_max.at(global_element_id, "v0_max");
      const double p_closing_min =
          parameters.linear_pressure.p_closing_min.at(global_element_id, "p_closing_min");
      const double p_opening_min =
          parameters.linear_pressure.p_opening_min.at(global_element_id, "p_opening_min");
      const double delta_p_minmax =
          parameters.linear_pressure.delta_p_minmax.at(global_element_id, "delta_p_minmax");
      const double epsilon_v0_switch =
          parameters.linear_pressure.epsilon_v0_switch.at(global_element_id, "epsilon_v0_switch");
      const double initial_v0 =
          parameters.linear_pressure.initial_v0.at(global_element_id, "initial_v0");
      const double tau = time_law_type == TimeLawType::ExponentialRelaxation
                             ? parameters.exponential_relaxation.tau.at(global_element_id, "tau")
                             : 0.0;

      validate_linear_pressure_parameters(time_law_type, v0_min, v0_max, initial_v0, p_closing_min,
          p_opening_min, delta_p_minmax, epsilon_v0_switch, tau, global_element_id);

      recruitment_model.time_law.type.push_back(time_law_type);
      recruitment_model.time_law.tau.push_back(tau);
      recruitment_model.pressure_law.active_path_n.push_back(
          parameters.linear_pressure.initial_path.at(global_element_id, "initial_path"));
      recruitment_model.pressure_law.v0_min.push_back(v0_min);
      recruitment_model.pressure_law.v0_max.push_back(v0_max);
      recruitment_model.pressure_law.p_closing_min.push_back(p_closing_min);
      recruitment_model.pressure_law.p_opening_min.push_back(p_opening_min);
      recruitment_model.pressure_law.delta_p_minmax.push_back(delta_p_minmax);
      recruitment_model.pressure_law.epsilon_v0_switch.push_back(epsilon_v0_switch);
      recruitment_model.v0_n.push_back(initial_v0);
      recruitment_model.v0_target.push_back(initial_v0);
    }
  }  // namespace

  /**
   * Resolve the reference volume of one element at the last converged time step.
   */
  double reference_volume_n(const RecruitmentModel& recruitment_model, const size_t element_index)
  {
    return std::visit(
        [&](const auto& model) -> double
        {
          using ModelType = std::decay_t<decltype(model)>;
          if constexpr (std::is_same_v<ModelType, NoRecruitment>)
          {
            return model.v0[element_index];
          }
          else if constexpr (std::is_same_v<ModelType, LinearPressureRecruitment>)
          {
            return model.v0_n[element_index];
          }
          else
          {
            FOUR_C_THROW("Unknown terminal-unit recruitment model.");
          }
        },
        recruitment_model);
  }

  /**
   * Evaluate the effective reference volume of one element.
   */
  ReferenceVolumeContext evaluate_recruitment_context(const TerminalUnitData& data,
      const RecruitmentModel& recruitment_model,
      const Core::LinAlg::Vector<double>& locally_relevant_dofs, const size_t element_index,
      const double dt)
  {
    return make_reference_volume_context(reference_volume_n(recruitment_model, element_index));
  }

  /**
   * Advance reference volume and hysteresis branch of all elements in one model block.
   */
  void update_recruitment_state(const TerminalUnitData& data, RecruitmentModel& recruitment_model,
      const Core::LinAlg::Vector<double>& locally_relevant_dofs, const double dt)
  {
    std::visit(
        [&](auto& model)
        {
          using ModelType = std::decay_t<decltype(model)>;
          if constexpr (std::is_same_v<ModelType, NoRecruitment>)
          {
            // The reference volume is the geometry-derived constant; there is no state to advance.
          }
          else if constexpr (std::is_same_v<ModelType, LinearPressureRecruitment>)
          {
            for (size_t i = 0; i < data.number_of_elements(); ++i)
            {
              TargetContext target;
              const double v0_next =
                  evaluate_recruitment_law(data, model, locally_relevant_dofs, i, dt, target);

              // The law reads the state of the last converged step, so overwrite it afterwards.
              model.v0_target[i] = target.v0_target;
              model.pressure_law.active_path_n[i] = maybe_switch_path(
                  model.pressure_law.active_path_n[i], v0_next, model.pressure_law.v0_min[i],
                  model.pressure_law.v0_max[i], model.pressure_law.epsilon_v0_switch[i]);
              model.v0_n[i] = v0_next;
            }
          }
          else
          {
            FOUR_C_THROW("Unknown terminal-unit recruitment model.");
          }
        },
        recruitment_model);
  }

  /**
   * Append input parameters and initialize the recruitment state vectors.
   */
  void append_model_parameters(RecruitmentModel& recruitment_model, const int global_element_id,
      const double geometry_volume,
      const ReducedLungParameters::LungTree::TerminalUnits::RecruitmentModel& parameters)
  {
    std::visit(
        [&](auto& model)
        {
          using ModelType = std::decay_t<decltype(model)>;
          if constexpr (std::is_same_v<ModelType, NoRecruitment>)
          {
            if (parameters.time_law_type.at(global_element_id, "time_law_type") !=
                TimeLawType::None)
            {
              FOUR_C_THROW(
                  "Terminal unit {}: recruitment time law requires an active pressure law.",
                  global_element_id);
            }
            model.v0.push_back(geometry_volume);
          }
          else if constexpr (std::is_same_v<ModelType, LinearPressureRecruitment>)
          {
            append_linear_pressure_parameters(model, global_element_id, parameters);
          }
          else
          {
            FOUR_C_THROW("Unknown terminal-unit recruitment model.");
          }
        },
        recruitment_model);
  }

  /**
   * Resolve variant-based output evaluator.
   */
  OutputEvaluator make_output_evaluator(const RecruitmentModel& recruitment_model)
  {
    return std::visit(
        [&](const auto& model) -> OutputEvaluator
        {
          using ModelType = std::decay_t<decltype(model)>;
          if constexpr (std::is_same_v<ModelType, NoRecruitment>)
          {
            return [](const TerminalUnitData&, RuntimeOutputCollector&,
                       ReducedLungParameters::OutputVerbosity) {};
          }
          else if constexpr (std::is_same_v<ModelType, LinearPressureRecruitment>)
          {
            return [&model](const TerminalUnitData& data, RuntimeOutputCollector& collector,
                       ReducedLungParameters::OutputVerbosity verbosity)
            {
              if (verbosity < ReducedLungParameters::OutputVerbosity::high) return;

              auto& v0 = collector.get_or_create_vector("v_0");
              auto& v0_target = collector.get_or_create_vector("v0_target");

              for (size_t i = 0; i < data.number_of_elements(); ++i)
              {
                v0.replace_local_value(data.local_element_id[i], model.v0_n[i]);
                v0_target.replace_local_value(data.local_element_id[i], model.v0_target[i]);
              }
            };
          }
          else
          {
            FOUR_C_THROW("Unknown terminal-unit recruitment model.");
          }
        },
        recruitment_model);
  }
}  // namespace ReducedLung::TerminalUnits::Recruitment

FOUR_C_NAMESPACE_CLOSE

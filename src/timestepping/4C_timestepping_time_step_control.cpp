// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_timestepping_time_step_control.hpp"

#include "4C_io_input_spec_builders.hpp"
#include "4C_io_input_spec_validators.hpp"
#include "4C_utils_exceptions.hpp"

#include <algorithm>
#include <limits>
#include <numeric>

FOUR_C_NAMESPACE_OPEN

namespace TimeStepping
{

  TimeStepControlSettings::TimeStepControlSettings(
      const InputParameters& input, const double initial_time_step, const int max_iter)
      : decrease_factor(input.decrease_factor),
        min_time_step(initial_time_step * input.min_time_step_ratio),
        steps_to_increase(input.steps_to_increase),
        max_time_step(input.max_time_step.value_or(initial_time_step)),
        increase_factor(input.increase_factor.value_or(1.0 / input.decrease_factor)),
        max_average_nonlinear_iterations(input.max_average_nonlinear_iterations.value_or(max_iter)),
        reduce_to_max_time(input.reduce_to_max_time)
  {
  }

  [[nodiscard]] Core::IO::InputSpec TimeStepControlSettings::input_spec()
  {
    using namespace Core::IO::InputSpecBuilders;
    using namespace Core::IO::InputSpecBuilders::Validators;

    return group<InputParameters>("TIMESTEP CONTROL",
        {
            parameter<double>("DECREASE_FACTOR",
                {.description = "Factor applied to the global time-step size when a "
                                "time-step retry is requested",
                    .default_value = 0.5,
                    .validator = Validators::in_range(Validators::excl(0.0), Validators::excl(1.0)),
                    .store =
                        in_struct(&TimeStepControlSettings::InputParameters::decrease_factor)}),
            parameter<double>("MIN_TIMESTEP_RATIO",
                {.description = "Minimum allowed global time-step size during "
                                "time-step reduction, relative to the initial time-step size",
                    .default_value = 1.0e-3,
                    .validator = Validators::in_range(Validators::excl(0.0), Validators::excl(1.0)),
                    .store =
                        in_struct(&TimeStepControlSettings::InputParameters::min_time_step_ratio)}),
            parameter<int>("STEPS_TO_INCREASE",
                {.description = "Number of consecutively accepted time-steps on a reduced "
                                "time-step size before increasing the global time-step size again",
                    .default_value = 10,
                    .validator = Validators::positive<int>(),
                    .store =
                        in_struct(&TimeStepControlSettings::InputParameters::steps_to_increase)}),
            parameter<std::optional<double>>("MAX_TIMESTEP",
                {.description =
                        "Maximum allowed global time-step size during "
                        "time-step recovery. If omitted, the initially "
                        "configured TIMESTEP is used and time-step size increase is only attempted "
                        "if the current time-step size is smaller than the initial time-step size. "
                        "If this is larger than the initial time-step size, time-step increase is "
                        "attempted even if the time-step size has never been reduced."
                        "If this is smaller than the initial time-step size, time-step increase is "
                        "only attempted once the current time-step size has been reduced below "
                        "this value.",
                    .validator = Validators::null_or(Validators::positive<double>()),
                    .store = in_struct(&TimeStepControlSettings::InputParameters::max_time_step)}),
            parameter<std::optional<double>>("INCREASE_FACTOR",
                {.description = "Factor applied when increasing the global time-step size "
                                "during time-step increase. If omitted, "
                                "INCREASE_FACTOR = 1/DECREASE_FACTOR. If 1.0, the time-step is "
                                "never increased.",
                    .validator = Validators::null_or(Validators::in_range(
                        Validators::incl(1.0), std::numeric_limits<double>::max())),
                    .store =
                        in_struct(&TimeStepControlSettings::InputParameters::increase_factor)}),
            parameter<std::optional<double>>("MAX_AVERAGE_NONLINEAR_ITERATIONS",
                {.description =
                        "Time-step increase is attempted only if the average number of Newton "
                        "iterations over the last STEPS_TO_INCREASE steps is below this value. "
                        "If omitted, time-step increase is attempted regardless of the "
                        "number of Newton iterations.",
                    .validator = Validators::null_or(Validators::positive<double>()),
                    .store = in_struct(&TimeStepControlSettings::InputParameters::
                            max_average_nonlinear_iterations)}),
            parameter<bool>("REDUCE_TO_MAXTIME",
                {.description = "If true, the last time-step size of the simulation is reduced to "
                                "exactly reach the "
                                "specified MAXTIME.",
                    .default_value = false,
                    .store =
                        in_struct(&TimeStepControlSettings::InputParameters::reduce_to_max_time)}),
        },
        {.description = "Settings for time-step control", .required = false});
  }

  double compute_reduced_time_step(const double current_dt, const TimeStepControlSettings& settings)
  {
    const double new_dt = current_dt * settings.decrease_factor;
    if (new_dt < settings.min_time_step)
    {
      FOUR_C_THROW(
          "A reduced time-step was requested, but reducing dt would violate the specified minimum "
          "time-step size. Current dt = {}, requested dt = {}, minimum dt = {}.",
          current_dt, new_dt, settings.min_time_step);
    }

    return new_dt;
  }

  double compute_time_step_after_successful_step(const double current_dt,
      const TimeStepControlSettings& settings, const size_t num_successful_steps_at_current_dt,
      const std::deque<size_t>& newton_iterations, const double remaining_time,
      const int remaining_steps)
  {
    if (remaining_time <= std::numeric_limits<double>::epsilon() or remaining_steps == 0)
    {
      return current_dt;
    }

    double new_dt = current_dt;

    // current_dt might be larger than max_time_step if the initial time-step was set larger than
    // max_time_step.
    if (current_dt < settings.max_time_step and
        num_successful_steps_at_current_dt >= settings.steps_to_increase)
    {
      const double average_nonlinear_iterations =
          std::accumulate(newton_iterations.begin(), newton_iterations.end(), 0.0) /
          static_cast<double>(newton_iterations.size());

      if (average_nonlinear_iterations <= settings.max_average_nonlinear_iterations)
      {
        new_dt = std::min(current_dt * settings.increase_factor, settings.max_time_step);
      }
    }

    if (settings.reduce_to_max_time and new_dt > remaining_time)
    {
      new_dt = remaining_time;
    }

    return new_dt;
  }
}  // namespace TimeStepping
FOUR_C_NAMESPACE_CLOSE

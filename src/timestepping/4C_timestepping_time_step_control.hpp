// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_TIMESTEPPING_TIME_STEP_CONTROL_HPP
#define FOUR_C_TIMESTEPPING_TIME_STEP_CONTROL_HPP

#include "4C_config.hpp"

#include "4C_io_input_spec.hpp"

#include <cstddef>
#include <deque>
#include <optional>

FOUR_C_NAMESPACE_OPEN

namespace TimeStepping
{
  struct TimeStepControlSettings
  {
    /// Input parameters used to construct TimeStepControlSettings.
    struct InputParameters
    {
      double decrease_factor;
      double min_time_step_ratio;
      int steps_to_increase;
      std::optional<double> max_time_step;
      std::optional<double> increase_factor;
      std::optional<double> max_average_nonlinear_iterations;
      bool reduce_to_max_time;
    };

    /// Return the input specification for the TIMESTEP CONTROL parameter group.
    [[nodiscard]] static Core::IO::InputSpec input_spec();

    /// Factor used to reduce the time-step size after a failed step.
    double decrease_factor;
    /// Minimum allowed absolute time-step size.
    double min_time_step;
    /// Number of accepted steps before a time-step increase may be attempted.
    size_t steps_to_increase;
    /// Maximum allowed time-step size during recovery.
    double max_time_step;
    /// Factor used to increase the time-step size during recovery.
    double increase_factor;
    /// Maximum average Newton iterations allowed for a time-step increase.
    double max_average_nonlinear_iterations;
    /// Whether the last step may be shortened to reach the final time exactly.
    bool reduce_to_max_time;

    /// Default constructor.
    TimeStepControlSettings() = default;

    /**
     * \brief Construct settings from parsed input parameters.
     *
     * @param input parsed TIME STEP CONTROL input parameters
     * @param initial_time_step initial global time-step size
     * @param max_iter maximum number of nonlinear iterations per time-step
     */
    TimeStepControlSettings(
        const InputParameters& input, const double initial_time_step, const int max_iter);
  };

  /**
   * \brief Compute the reduced time-step size after a failed step.
   *
   * @param current_dt current time-step size
   * @param settings time-step control settings used by the calling algorithm
   * @return reduced time-step size
   *
   * @throws Core::Exception if the reduced time-step size is below the configured minimum
   */
  double compute_reduced_time_step(
      const double current_dt, const TimeStepControlSettings& settings);

  /**
   * \brief Compute the time-step size to use after an accepted step.
   *
   * @param current_dt current time-step size
   * @param settings time-step control settings used by the calling algorithm
   * @param num_successful_steps_at_current_dt number of accepted steps since the current time-step
   * size was selected
   * @param newton_iterations Newton iteration counts from the recent accepted steps considered for
   * time-step recovery
   * @param remaining_time remaining simulation time after the accepted step
   * @param remaining_steps remaining number of steps after the accepted step
   * @return time-step size for the next step
   */
  double compute_time_step_after_successful_step(const double current_dt,
      const TimeStepControlSettings& settings, const size_t num_successful_steps_at_current_dt,
      const std::deque<size_t>& newton_iterations, const double remaining_time,
      const int remaining_steps);
}  // namespace TimeStepping

FOUR_C_NAMESPACE_CLOSE

#endif

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_timestepping_time_step_control.hpp"

#include "4C_unittest_utils_assertions_test.hpp"
#include "4C_utils_exceptions.hpp"

#include <cstddef>
#include <deque>
#include <optional>
#include <string>

namespace
{
  using namespace FourC;

  using namespace TimeStepping;

  const TimeStepControlSettings settings(
      TimeStepControlSettings::InputParameters{
          .decrease_factor = 0.5,
          .min_time_step_ratio = 1.0e-1,  // minimum time-step is 0.05
          .steps_to_increase = 2,
          .max_time_step = 0.5,
          .increase_factor = 2.0,
          .max_average_nonlinear_iterations = 3,
          .reduce_to_max_time = true,
      },
      0.5, 5);

  // number of newton iterations per step such that the average is below the threshold of 3.0
  const std::deque<size_t> number_of_newton_iterations_per_step{1, 1};

  // Test construction

  TEST(TimeStepControlTest, ConstructorHandlesNullopt)
  {
    const TimeStepControlSettings::InputParameters input{
        .decrease_factor = 0.5,
        .min_time_step_ratio = 1.0e-1,
        .steps_to_increase = 2,
        .max_time_step = std::nullopt,
        .increase_factor = std::nullopt,
        .max_average_nonlinear_iterations = std::nullopt,
        .reduce_to_max_time = true,
    };

    const double initial_time_step = 0.5;
    const int max_iter = 5;
    const TimeStepControlSettings settings(input, initial_time_step, max_iter);

    EXPECT_EQ(settings.max_time_step, initial_time_step);
    EXPECT_EQ(settings.increase_factor, 1.0 / input.decrease_factor);
    EXPECT_EQ(settings.max_average_nonlinear_iterations, max_iter);
  }

  TEST(TimeStepControlTest, ConstructorDoesNotOverrideOptionals)
  {
    const TimeStepControlSettings::InputParameters input{
        .decrease_factor = 0.5,
        .min_time_step_ratio = 1.0e-1,
        .steps_to_increase = 2,
        .max_time_step = 0.75,
        .increase_factor = 1.5,
        .max_average_nonlinear_iterations = 3,
        .reduce_to_max_time = true,
    };

    const TimeStepControlSettings settings(input, 0.5, 5);

    EXPECT_EQ(settings.max_time_step, 0.75);
    EXPECT_EQ(settings.increase_factor, 1.5);
    EXPECT_EQ(settings.max_average_nonlinear_iterations, 3);
  }

  // Test compute_reduced_time_step

  TEST(TimeStepControlTest, ReducedTimeStepUsesDecreaseFactor)
  {
    EXPECT_DOUBLE_EQ(compute_reduced_time_step(0.5, settings), 0.25);
  }

  TEST(TimeStepControlTest, ReducedTimeStepThrowsBelowMinimum)
  {
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        compute_reduced_time_step(0.08, settings), Core::Exception, "minimum time-step size");
  }

  // Test compute_time_step_after_successful_step

  TEST(TimeStepControlTest, ReturnsCurrentDtWhenNoRemainingTime)
  {
    double current_dt = 0.2;

    EXPECT_EQ(compute_time_step_after_successful_step(current_dt, settings,
                  settings.steps_to_increase, number_of_newton_iterations_per_step, 0.0, 10),
        current_dt);
  }

  TEST(TimeStepControlTest, ReturnsCurrentDtWhenNoRemainingSteps)
  {
    double current_dt = 0.2;

    EXPECT_EQ(compute_time_step_after_successful_step(current_dt, settings,
                  settings.steps_to_increase, number_of_newton_iterations_per_step, 10.0, 0),
        current_dt);
  }

  TEST(TimeStepControlTest, ReturnsCurrentDtWhenLargerOrEqualThanMaxTimeStep)
  {
    // larger
    double current_dt = 0.75;
    EXPECT_EQ(compute_time_step_after_successful_step(current_dt, settings,
                  settings.steps_to_increase, number_of_newton_iterations_per_step, 10.0, 10),
        current_dt);
    // equal
    current_dt = settings.max_time_step;
    EXPECT_EQ(compute_time_step_after_successful_step(current_dt, settings,
                  settings.steps_to_increase, number_of_newton_iterations_per_step, 10.0, 10),
        current_dt);
  }

  TEST(TimeStepControlTest, WaitsUntilEnoughSuccessfulSteps)
  {
    double current_dt = 0.2;

    EXPECT_DOUBLE_EQ(compute_time_step_after_successful_step(
                         current_dt, settings, 1, number_of_newton_iterations_per_step, 10.0, 10),
        current_dt);
  }

  TEST(TimeStepControlTest, BlocksIncreaseWhenAverageNewtonIterationsIsTooHigh)
  {
    const std::deque<size_t> newton_iterations{2, 5};  // average = 3.5 > 3

    double current_dt = 0.2;
    EXPECT_DOUBLE_EQ(compute_time_step_after_successful_step(current_dt, settings,
                         settings.steps_to_increase, newton_iterations, 10.0, 10),
        current_dt);
  }

  TEST(TimeStepControlTest, IncreasesAfterEnoughSuccessfulStepsAndAverageBelowThreshold)
  {
    const std::deque<size_t> newton_iterations{2, 3};  // average = 2.5 < 3

    double current_dt = 0.2;
    EXPECT_DOUBLE_EQ(compute_time_step_after_successful_step(current_dt, settings,
                         settings.steps_to_increase, newton_iterations, 10.0, 10),
        current_dt * settings.increase_factor);
  }

  TEST(TimeStepControlTest, CapsIncreaseAtMaxTimeStep)
  {
    double current_dt = 0.3;
    EXPECT_DOUBLE_EQ(
        compute_time_step_after_successful_step(current_dt, settings, settings.steps_to_increase,
            number_of_newton_iterations_per_step, 10.0, 10),
        settings.max_time_step);
  }

  TEST(TimeStepControlTest, DecreasesStepToMatchMaxTime)
  {
    double current_dt = 0.2;
    EXPECT_DOUBLE_EQ(compute_time_step_after_successful_step(current_dt, settings,
                         settings.steps_to_increase, number_of_newton_iterations_per_step, 0.15, 1),
        0.15);
  }

  TEST(TimeStepControlTest, DoesNotReduceToMaxTimeWhenDisabled)
  {
    TimeStepControlSettings settings_no_reduce_to_maxtime = settings;
    settings_no_reduce_to_maxtime.reduce_to_max_time = false;

    double current_dt = 0.5;
    EXPECT_EQ(compute_time_step_after_successful_step(current_dt, settings_no_reduce_to_maxtime,
                  settings_no_reduce_to_maxtime.steps_to_increase,
                  number_of_newton_iterations_per_step, 0.15, 1),
        current_dt);
  }
}  // namespace

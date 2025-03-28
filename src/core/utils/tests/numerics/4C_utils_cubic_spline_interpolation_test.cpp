// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_utils_cubic_spline_interpolation.hpp"

#include "4C_utils_exceptions.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace
{
  class CubicSplineInterpolationTest : public ::testing::Test
  {
   protected:
    CubicSplineInterpolationTest()
    {
      const std::vector<double> x = {0.30, 0.35, 0.40, 0.45};
      const std::vector<double> y = {4.40, 4.30, 4.25, 4.10};

      cubic_spline_ = std::make_shared<Core::Utils::CubicSplineInterpolation>(x, y);
    }

    std::shared_ptr<Core::Utils::CubicSplineInterpolation> cubic_spline_;
  };

  TEST_F(CubicSplineInterpolationTest, InputArgumentSortedAscending)
  {
    const std::vector<double> x = {0.3, 0.6, 0.5};
    const std::vector<double> y(x.size(), 0.0);

    EXPECT_THROW(Core::Utils::CubicSplineInterpolation(x, y), Core::Exception);
  }

  TEST_F(CubicSplineInterpolationTest, InputArgumentsDifferentLength)
  {
    const std::vector<double> x = {0.3, 0.5, 0.7};
    const std::vector<double> y(x.size() - 1, 0.0);

    EXPECT_THROW(Core::Utils::CubicSplineInterpolation(x, y), Core::Exception);
  }

  TEST_F(CubicSplineInterpolationTest, EvaluateOutsideValidityBounds)
  {
    const std::vector<double> x_test = {0.2, 0.5};

    for (double x : x_test)
    {
      EXPECT_THROW((void)cubic_spline_->evaluate(x), Core::Exception);
      EXPECT_THROW((void)cubic_spline_->evaluate_derivative(x, 1), Core::Exception);
      EXPECT_THROW((void)cubic_spline_->evaluate_derivative(x, 2), Core::Exception);
    }
  }

  TEST_F(CubicSplineInterpolationTest, EvaluateScalar)
  {
    const std::vector<double> x_test = {0.33, 0.36, 0.4, 0.42};
    const std::vector<double> reference_solution = {4.33232, 4.29, 4.25, 4.20152};

    for (std::size_t i = 0; i < x_test.size(); ++i)
      EXPECT_NEAR(cubic_spline_->evaluate(x_test[i]), reference_solution[i], 1.0e-12);
  }

  TEST_F(CubicSplineInterpolationTest, EvaluateFirstDerivative)
  {
    const std::vector<double> x_test = {0.33, 0.36, 0.4, 0.42};
    const std::vector<double> reference_solution = {-1.968, -0.84, -1.8, -2.952};

    for (std::size_t i = 0; i < x_test.size(); ++i)
      EXPECT_NEAR(cubic_spline_->evaluate_derivative(x_test[i], 1), reference_solution[i], 1.0e-12);
  }

  TEST_F(CubicSplineInterpolationTest, EvaluateSecondDerivative)
  {
    const std::vector<double> x_test = {0.33, 0.36, 0.4, 0.42};
    const std::vector<double> reference_solution = {28.8, 24.0, -72.0, -43.2};

    for (std::size_t i = 0; i < x_test.size(); ++i)
      EXPECT_NEAR(cubic_spline_->evaluate_derivative(x_test[i], 2), reference_solution[i], 1.0e-12);
  }
}  // namespace
FOUR_C_NAMESPACE_CLOSE

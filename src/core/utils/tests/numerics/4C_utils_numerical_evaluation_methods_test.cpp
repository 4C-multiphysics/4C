// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_utils_numerical_evaluation_methods.hpp"

#include "4C_unittest_utils_assertions_test.hpp"
#include "4C_utils_exceptions.hpp"

#include <limits>

FOUR_C_NAMESPACE_OPEN

namespace
{
  TEST(CoreUtilsTest, Bisection)
  {
    std::function<double(double)> function = [](double x) { return std::pow(x, 2) - 1; };

    double root = Core::Utils::bisection(function, 0.0, 5.0, 1e-12, 200);

    EXPECT_NEAR(root, 1.0, 1e-12);
  }

  TEST(CoreUtilsTest, FirstDerivativeCentralDifferences)
  {
    const double x = 1.5;
    const double h = 0.001;

    // lets say f(x) = x^2 + 1
    const double f_xplus = std::pow(x + h, 2) + 1;
    const double f_xminus = std::pow(x - h, 2) + 1;

    // correct dfdx would be dfdx = 2x
    const double ref_dfdx = 2 * x;

    double dfdx = Core::Utils::first_derivative_central_differences(f_xminus, f_xplus, h);

    EXPECT_NEAR(dfdx, ref_dfdx, std::pow(h, 2.0));
  }

  TEST(CoreUtilsTest, SecondDerivativeCentralDifferences)
  {
    const double x = 1.5;
    const double h = 0.001;

    // lets say f(x) = x^2 + 1
    const double f_x = std::pow(x, 2) + 1;
    const double f_xplus = std::pow(x + h, 2) + 1;
    const double f_xminus = std::pow(x - h, 2) + 1;

    // correct dfdx would be ddfddx = 2
    const double ref_ddfddx = 2;

    double ddfddx = Core::Utils::second_derivative_central_differences(f_xminus, f_x, f_xplus, h);

    EXPECT_NEAR(ddfddx, ref_ddfddx, std::pow(h, 2.0));
  }

  TEST(CoreUtilsTest, SignedLogSumExp)
  {
    const double max_double = std::numeric_limits<double>::max();

    // consistency tests on vector sizes
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(std::ignore = Core::Utils::signed_log_sum_exp({}, {}),
        Core::Exception, "No values were given for this computation");
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        std::ignore = Core::Utils::signed_log_sum_exp({1.0, 2.0}, {1.0}), Core::Exception,
        "The number of values must match the number of prefactors: currently 2 versus 1, "
        "respectively");

    // single double test
    const auto lse_single_double = Core::Utils::signed_log_sum_exp({std::log(2.0)}, {1.0});
    EXPECT_EQ(lse_single_double.eval_status, Core::Utils::SignedLogSumExpStatus::success);
    EXPECT_DOUBLE_EQ(lse_single_double.log_abs, std::log(2.0));
    EXPECT_EQ(lse_single_double.sign, 1.0);

    const auto lse_single_negative_double =
        Core::Utils::signed_log_sum_exp({std::log(2.0)}, {-1.0});
    EXPECT_EQ(lse_single_negative_double.eval_status, Core::Utils::SignedLogSumExpStatus::success);
    EXPECT_DOUBLE_EQ(lse_single_negative_double.log_abs, std::log(2.0));
    EXPECT_EQ(lse_single_negative_double.sign, -1.0);

    // different tests for two doubles
    const auto lse_diff_two_doubles =
        Core::Utils::signed_log_sum_exp({std::log(2.0), std::log(1.0)}, {1.0, -1.0});
    EXPECT_EQ(lse_diff_two_doubles.eval_status, Core::Utils::SignedLogSumExpStatus::success);
    EXPECT_DOUBLE_EQ(lse_diff_two_doubles.log_abs, 0.0);
    EXPECT_EQ(lse_diff_two_doubles.sign, 1.0);

    const auto lse_sum_two_negative_doubles =
        Core::Utils::signed_log_sum_exp({std::log(2.0), std::log(1.0)}, {-1.0, -1.0});
    EXPECT_EQ(
        lse_sum_two_negative_doubles.eval_status, Core::Utils::SignedLogSumExpStatus::success);
    EXPECT_DOUBLE_EQ(lse_sum_two_negative_doubles.log_abs, std::log(3.0));
    EXPECT_EQ(lse_sum_two_negative_doubles.sign, -1.0);

    const auto lse_sum_negative_exponents =
        Core::Utils::signed_log_sum_exp({-2.0, -1.0}, {-1.0, -1.0});
    EXPECT_EQ(lse_sum_negative_exponents.eval_status, Core::Utils::SignedLogSumExpStatus::success);
    EXPECT_DOUBLE_EQ(lse_sum_negative_exponents.log_abs, std::log(std::exp(-2.0) + std::exp(-1.0)));
    EXPECT_EQ(lse_sum_negative_exponents.sign, -1.0);

    const auto lse_sum_high_doubles = Core::Utils::signed_log_sum_exp({1000.0, 999.0}, {1.0, 1.0});
    EXPECT_EQ(lse_sum_high_doubles.eval_status, Core::Utils::SignedLogSumExpStatus::success);
    EXPECT_DOUBLE_EQ(lse_sum_high_doubles.log_abs, 1000.0 + std::log(1 + std::exp(-1.0)));
    EXPECT_EQ(lse_sum_high_doubles.sign, 1.0);

    const auto lse_sum_low_doubles = Core::Utils::signed_log_sum_exp({-1000.0, -999.0}, {1.0, 1.0});
    EXPECT_EQ(lse_sum_low_doubles.eval_status, Core::Utils::SignedLogSumExpStatus::success);
    EXPECT_DOUBLE_EQ(lse_sum_low_doubles.log_abs, -999.0 + std::log(1 + std::exp(-1.0)));
    EXPECT_EQ(lse_sum_low_doubles.sign, 1.0);
    // verification: naive summation gives 0.0, which would mean that computing the logarithm this
    // way instead of using log-sum-exp would result in \f$ - \infty \f$
    EXPECT_DOUBLE_EQ(std::exp(-1000.0) + std::exp(-999.0), 0.0);

    // tests for three and four doubles
    const auto lse_mixed_sum_of_three_doubles = Core::Utils::signed_log_sum_exp(
        {std::log(2.0), std::log(1.0), std::log(1.0)}, {1.0, 1.0, -1.0});
    EXPECT_EQ(
        lse_mixed_sum_of_three_doubles.eval_status, Core::Utils::SignedLogSumExpStatus::success);
    EXPECT_DOUBLE_EQ(lse_mixed_sum_of_three_doubles.log_abs, std::log(2.0));
    EXPECT_EQ(lse_mixed_sum_of_three_doubles.sign, 1.0);

    const auto lse_mixed_sum_of_four_doubles_negative = Core::Utils::signed_log_sum_exp(
        {std::log(2.0), std::log(1.0), std::log(1.0), 2.0 * std::log(2.0)}, {1.0, 1.0, -1.0, -1.0});
    EXPECT_EQ(lse_mixed_sum_of_four_doubles_negative.eval_status,
        Core::Utils::SignedLogSumExpStatus::success);
    EXPECT_DOUBLE_EQ(lse_mixed_sum_of_four_doubles_negative.log_abs, std::log(2.0));
    EXPECT_EQ(lse_mixed_sum_of_four_doubles_negative.sign, -1.0);

    const auto lse_stable_even_for_max_prefactors = Core::Utils::signed_log_sum_exp(
        {std::log(1.0), std::log(1.0), std::log(1.0)}, {max_double, max_double, -max_double});
    EXPECT_EQ(lse_stable_even_for_max_prefactors.eval_status,
        Core::Utils::SignedLogSumExpStatus::success);
    EXPECT_DOUBLE_EQ(lse_stable_even_for_max_prefactors.log_abs, std::log(max_double));
    EXPECT_EQ(lse_stable_even_for_max_prefactors.sign, 1.0);

    // --> trigger different error / zero_argument / overflow cases

    // provided arguments must be evaluable
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        std::ignore = Core::Utils::signed_log_sum_exp({max_double * 2}, {1.0}), Core::Exception,
        "Given log_value inf with index 0 is non-finite! The calling procedure must deal with such "
        "cases itself!");
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        std::ignore = Core::Utils::signed_log_sum_exp({0.0, -1000.0}, {0.0, 1.0}), Core::Exception,
        "Given prefactor 0 with index 0 has an absolute value smaller than precision epsilon");
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        std::ignore = Core::Utils::signed_log_sum_exp({1.0}, {max_double * 2}), Core::Exception,
        "Given prefactor inf with index 0 is non-finite! The calling procedure must deal with such "
        "cases itself!");

    // zero_argument happening due to cancellation of the terms (reduced sum = 0)
    EXPECT_EQ(Core::Utils::signed_log_sum_exp({0.0, 0.0}, {1.0, -1.0}).eval_status,
        Core::Utils::SignedLogSumExpStatus::zero_argument);
    EXPECT_EQ(Core::Utils::signed_log_sum_exp({1.0, 1.0, 1.0}, {1.0, 1.0, -2.0}).eval_status,
        Core::Utils::SignedLogSumExpStatus::zero_argument);  // also exercises prefactors
                                                             // different than 1.0 / -1.0
    EXPECT_EQ(Core::Utils::signed_log_sum_exp({1.0, 1.0 - 1.0e-15}, {1.0, -1.0}).eval_status,
        Core::Utils::SignedLogSumExpStatus::success);  // no complete cancellation!

    // overflow occurring through very high values
    EXPECT_EQ(Core::Utils::signed_log_sum_exp({max_double, max_double}, {max_double, max_double})
                  .eval_status,
        Core::Utils::SignedLogSumExpStatus::overflow);
  }

}  // namespace
FOUR_C_NAMESPACE_CLOSE

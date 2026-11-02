// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_unittest_utils_assertions_test.hpp"

#include <gtest/gtest-spi.h>

#include <array>
#include <limits>
#include <vector>

namespace
{
  using namespace FourC;

  TEST(UnittestUtilsAssertionsTest, AbsoluteComparisonRespectsTolerance)
  {
    constexpr double passing_tolerance = 0.5;
    constexpr double failing_tolerance = 0.125;

    // actual values: (0.0, 0.25)
    const std::vector<double> actual_vector = {0.0, 0.25};
    const std::array<double, 2> actual_array = {0.0, 0.25};
    const Core::LinAlg::Matrix<1, 2> actual_matrix(actual_array.data());
    Core::LinAlg::SerialDenseMatrix actual_dense_matrix(1, 2, true);
    actual_dense_matrix(0, 1) = 0.25;
    Core::LinAlg::SerialDenseVector actual_dense_vector(2, true);
    actual_dense_vector(1) = 0.25;
    const Core::LinAlg::Tensor<double, 2> actual_tensor = {{0.0, 0.25}};

    // expected values: (0.0, 0.0)
    const std::vector<double> expected_vector = {0.0, 0.0};
    const std::array<double, 2> expected_array = {0.0, 0.0};
    const Core::LinAlg::Matrix<1, 2> expected_matrix(expected_array.data());
    const Core::LinAlg::SerialDenseMatrix expected_dense_matrix(1, 2, true);
    const Core::LinAlg::SerialDenseVector expected_dense_vector(2, true);
    const Core::LinAlg::Tensor<double, 2> expected_tensor = {{0.0, 0.0}};

    // difference within tolerance: 0.25 < 0.5
    FOUR_C_EXPECT_NEAR(actual_vector, expected_vector, passing_tolerance);
    FOUR_C_EXPECT_ITERABLE_NEAR(
        actual_array.data(), expected_array.data(), actual_array.size(), passing_tolerance);
    FOUR_C_EXPECT_NEAR(actual_matrix, expected_matrix, passing_tolerance);
    FOUR_C_EXPECT_NEAR(actual_matrix, expected_array, passing_tolerance);
    FOUR_C_EXPECT_NEAR(actual_dense_matrix, expected_dense_matrix, passing_tolerance);
    FOUR_C_EXPECT_NEAR(actual_dense_vector, expected_dense_vector, passing_tolerance);
    FOUR_C_EXPECT_NEAR(actual_tensor, expected_tensor, passing_tolerance);

    // difference exceeds tolerance: 0.25 > 0.125
    EXPECT_NONFATAL_FAILURE(
        FOUR_C_EXPECT_NEAR(actual_vector, expected_vector, failing_tolerance), "[1]: ");
    EXPECT_NONFATAL_FAILURE(FOUR_C_EXPECT_ITERABLE_NEAR(actual_array.data(), expected_array.data(),
                                actual_array.size(), failing_tolerance),
        "[1]: ");
    EXPECT_NONFATAL_FAILURE(
        FOUR_C_EXPECT_NEAR(actual_matrix, expected_matrix, failing_tolerance), "(0,1): ");
    EXPECT_NONFATAL_FAILURE(
        FOUR_C_EXPECT_NEAR(actual_matrix, expected_array, failing_tolerance), "(0,1) vs. [1]: ");
    EXPECT_NONFATAL_FAILURE(
        FOUR_C_EXPECT_NEAR(actual_dense_matrix, expected_dense_matrix, failing_tolerance),
        "(0,1): ");
    EXPECT_NONFATAL_FAILURE(
        FOUR_C_EXPECT_NEAR(actual_dense_vector, expected_dense_vector, failing_tolerance), "[1]: ");
    EXPECT_NONFATAL_FAILURE(
        FOUR_C_EXPECT_NEAR(actual_tensor, expected_tensor, failing_tolerance), "(1): ");
  }

  TEST(UnittestUtilsAssertionsTest, ComparisonRejectsNonfiniteEntries)
  {
    constexpr double nan = std::numeric_limits<double>::quiet_NaN();
    constexpr double tolerance = 1.0e-6;  // irrelevant here

    // actual values: (1.0, NaN)
    const std::vector<double> nan_vector = {1.0, nan};
    const std::array<double, 2> nan_array = {1.0, nan};
    const Core::LinAlg::Matrix<1, 2> nan_matrix(nan_array.data());
    Core::LinAlg::SerialDenseMatrix nan_dense_matrix(1, 2, true);
    nan_dense_matrix(0, 0) = 1.0;
    nan_dense_matrix(0, 1) = nan;
    Core::LinAlg::SerialDenseVector nan_dense_vector(2, true);
    nan_dense_vector(0) = 1.0;
    nan_dense_vector(1) = nan;
    const Core::LinAlg::Tensor<double, 2> nan_tensor = {{1.0, nan}};

    // expected values: (1.0, 0.0)
    const std::vector<double> finite_vector = {1.0, 0.0};
    const std::array<double, 2> finite_array = {1.0, 0.0};
    const Core::LinAlg::Matrix<1, 2> finite_matrix(finite_array.data());
    Core::LinAlg::SerialDenseMatrix finite_dense_matrix(1, 2, true);
    finite_dense_matrix(0, 0) = 1.0;
    Core::LinAlg::SerialDenseVector finite_dense_vector(2, true);
    finite_dense_vector(0) = 1.0;
    const Core::LinAlg::Tensor<double, 2> finite_tensor = {{1.0, 0.0}};

    // comparison fails due to NaN entries
    EXPECT_NONFATAL_FAILURE(FOUR_C_EXPECT_NEAR(nan_vector, finite_vector, tolerance), "[1]: nan");
    EXPECT_NONFATAL_FAILURE(FOUR_C_EXPECT_ITERABLE_NEAR(
                                nan_array.data(), finite_array.data(), nan_array.size(), tolerance),
        "[1]: nan");
    EXPECT_NONFATAL_FAILURE(FOUR_C_EXPECT_NEAR(nan_matrix, finite_matrix, tolerance), "(0,1): nan");
    EXPECT_NONFATAL_FAILURE(
        FOUR_C_EXPECT_NEAR(nan_matrix, finite_array, tolerance), "(0,1) vs. [1]: nan");
    EXPECT_NONFATAL_FAILURE(
        FOUR_C_EXPECT_NEAR(nan_dense_matrix, finite_dense_matrix, tolerance), "(0,1): nan");
    EXPECT_NONFATAL_FAILURE(
        FOUR_C_EXPECT_NEAR(nan_dense_vector, finite_dense_vector, tolerance), "[1]: nan");
    EXPECT_NONFATAL_FAILURE(FOUR_C_EXPECT_NEAR(nan_tensor, finite_tensor, tolerance), "(1): nan");
  }
}  // namespace

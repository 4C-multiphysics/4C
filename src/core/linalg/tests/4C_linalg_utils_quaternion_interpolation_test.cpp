// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_utlis_quaternion_interpolation.hpp"
#include "4C_unittest_utils_assertions_test.hpp"
#include "4C_utils_exceptions.hpp"

#include <array>
#include <cmath>
#include <iomanip>
#include <ios>
#include <utility>
#include <vector>

FOUR_C_NAMESPACE_OPEN

/**
 * @brief Unit test for quaternion interpolation using Slerp with identity quaternions.
 *
 * This test verifies that interpolating between two identical identity quaternions
 * using the GeneralizedSphericalLinearInterpolator yields the identity quaternion itself.
 * The test constructs two identity quaternions, assigns equal weights, performs the interpolation,
 * and asserts that the resulting quaternion matches the identity quaternion within a small
 * tolerance.
 */
TEST(QuaternionInterpolationTest, SlerpIdentity)
{
  // Interpolating between identity quaternion and itself (x, y, z, w)
  Core::LinAlg::Matrix<4, 1> q(Core::LinAlg::Initialization::zero);
  q(0, 0) = 0.0;  // x
  q(1, 0) = 0.0;  // y
  q(2, 0) = 0.0;  // z
  q(3, 0) = 1.0;  // w

  std::vector<Core::LinAlg::Matrix<4, 1>> quats = {q, q};
  std::vector<double> weights = {0.5, 0.5};

  Core::LinAlg::GeneralizedSphericalLinearInterpolator interpolator(quats, weights);
  Core::LinAlg::Matrix<4, 1> result = interpolator.get_interpolated_quaternion();

  ASSERT_NEAR(result(0, 0), 0.0, 1e-12);  // x
  ASSERT_NEAR(result(1, 0), 0.0, 1e-12);  // y
  ASSERT_NEAR(result(2, 0), 0.0, 1e-12);  // z
  ASSERT_NEAR(result(3, 0), 1.0, 1e-12);  // w
}

/**
 * @brief Unit test for verifying the Slerp (Spherical Linear Interpolation) functionality for
 * quaternions.
 *
 * This test checks the correctness of quaternion interpolation at the halfway point (t = 0.5)
 * using the Slerp algorithm. It ensures that the interpolated quaternion is as expected
 * when transitioning between two given quaternions.
 */
TEST(QuaternionInterpolationTest, SlerpHalfway)
{
  // Interpolating halfway between identity and 180-degree rotation about z (x, y, z, w)
  Core::LinAlg::Matrix<4, 1> q1(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<4, 1> q2(Core::LinAlg::Initialization::zero);
  q1(3, 0) = 1.0;  // w
  q2(2, 0) = 1.0;  // z

  // Use the SLERP function directly
  double t = 0.5;
  Core::LinAlg::Matrix<4, 1> slerp_result = Core::LinAlg::spherical_linear_interpolation(q1, q2, t);

  // Analytical result for SLERP between (0,0,0,1) and (0,0,1,0) at t=0.5:
  // angle = acos(0) = pi/2
  // slerp = (sin((1-t)*theta)*q1 + sin(t*theta)*q2) / sin(theta)
  // theta = pi/2, sin(theta) = 1
  // sin((1-t)*theta) = sin(0.5*pi/2) = sin(pi/4)
  // sin(t*theta) = sin(0.5*pi/2) = sin(pi/4)
  double theta = std::numbers::pi_v<double> / 2.0;
  double sin_theta = std::sin(theta);
  double sin_1mt_theta = std::sin((1.0 - t) * theta);
  double sin_t_theta = std::sin(t * theta);

  double w = sin_1mt_theta / sin_theta;
  double z = sin_t_theta / sin_theta;
  double norm = std::sqrt(w * w + z * z);

  double w_expected = w / norm;
  double z_expected = z / norm;

  ASSERT_NEAR(slerp_result(0, 0), 0.0, 1e-12);         // x
  ASSERT_NEAR(slerp_result(1, 0), 0.0, 1e-12);         // y
  ASSERT_NEAR(slerp_result(2, 0), z_expected, 1e-12);  // z
  ASSERT_NEAR(slerp_result(3, 0), w_expected, 1e-12);  // w

  // Also check the GeneralizedSphericalLinearInterpolator result matches
  std::vector<Core::LinAlg::Matrix<4, 1>> quats = {q1, q2};
  std::vector<double> weights = {0.5, 0.5};
  Core::LinAlg::GeneralizedSphericalLinearInterpolator interpolator(quats, weights);
  Core::LinAlg::Matrix<4, 1> result = interpolator.get_interpolated_quaternion();

  ASSERT_NEAR(result(0, 0), 0.0, 1e-12);         // x
  ASSERT_NEAR(result(1, 0), 0.0, 1e-12);         // y
  ASSERT_NEAR(result(2, 0), z_expected, 1e-12);  // z
  ASSERT_NEAR(result(3, 0), w_expected, 1e-12);  // w
}

/**
 * @brief Unit test for the SlerpWeighted function in quaternion interpolation utilities.
 *
 * This test verifies the correctness of the SlerpWeighted implementation, which performs
 * spherical linear interpolation (SLERP) between quaternions with specified weights.
 * It checks that the interpolated quaternion maintains expected properties such as normalization
 * and correct orientation between the input quaternions.
 */
TEST(QuaternionInterpolationTest, SlerpWeighted)
{
  // Weighted interpolation: 75% q1, 25% q2 (x, y, z, w order)
  Core::LinAlg::Matrix<4, 1> q1(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<4, 1> q2(Core::LinAlg::Initialization::zero);
  q1(3, 0) = 1.0;  // w
  q2(0, 0) = 1.0;  // x

  std::vector<Core::LinAlg::Matrix<4, 1>> quats = {q1, q2};
  std::vector<double> weights = {0.75, 0.25};

  Core::LinAlg::GeneralizedSphericalLinearInterpolator interpolator(quats, weights);
  Core::LinAlg::Matrix<4, 1> result = interpolator.get_interpolated_quaternion();

  double norm = std::sqrt(result(0, 0) * result(0, 0) + result(1, 0) * result(1, 0) +
                          result(2, 0) * result(2, 0) + result(3, 0) * result(3, 0));
  ASSERT_NEAR(norm, 1.0, 1e-12);
  ASSERT_GT(result(3, 0), result(0, 0));  // w should be greater than x
}

/**
 * @brief Unit test for generalized spherical linear interpolation (Slerp) among three quaternions.
 *
 * This test verifies the behavior of the GeneralizedSphericalLinearInterpolator when interpolating
 * between three orthogonal quaternions (representing the x, y, and w axes) with equal weights.
 *
 * The test sets up three quaternions:
 *   - q1: (0, 0, 0, 1)  // w-axis
 *   - q2: (1, 0, 0, 0)  // x-axis
 *   - q3: (0, 1, 0, 0)  // y-axis
 *
 * Each quaternion is assigned a weight of 1/3. The interpolator computes the resulting quaternion,
 * which is expected to be normalized and close to (1/sqrt(3), 1/sqrt(3), 0, 1/sqrt(3)).
 *
 * The test asserts that each component of the resulting quaternion matches the expected value
 * within a tolerance of 1e-12.
 */
TEST(QuaternionInterpolationTest, SlerpThreeQuaternions)
{
  // Interpolating among three orthogonal quaternions with equal weights (x, y, z, w order)
  Core::LinAlg::Matrix<4, 1> q1(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<4, 1> q2(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<4, 1> q3(Core::LinAlg::Initialization::zero);
  q1(3, 0) = 1.0;  // w
  q2(0, 0) = 1.0;  // x
  q3(1, 0) = 1.0;  // y

  std::vector<Core::LinAlg::Matrix<4, 1>> quats = {q1, q2, q3};
  std::vector<double> weights = {1.0 / 3, 1.0 / 3, 1.0 / 3};

  Core::LinAlg::GeneralizedSphericalLinearInterpolator<2> interpolator(quats, weights);
  Core::LinAlg::Matrix<4, 1> result = interpolator.get_interpolated_quaternion();

  // For three orthogonal quaternions with equal weights, the result should be close to normalized
  // (1/sqrt(3), 1/sqrt(3), 0, 1/sqrt(3))
  double inv_sqrt3 = 1.0 / std::sqrt(3.0);
  ASSERT_NEAR(result(0, 0), inv_sqrt3, 1e-12);  // x
  ASSERT_NEAR(result(1, 0), inv_sqrt3, 1e-12);  // y
  ASSERT_NEAR(result(2, 0), 0.0, 1e-12);        // z
  ASSERT_NEAR(result(3, 0), inv_sqrt3, 1e-12);  // w
}

// same test as above, but with 2D reference locations
// This test verifies that the interpolation works correctly with 2D reference locations
// and that the resulting quaternion is normalized and has the expected values.
TEST(QuaternionInterpolationTest, SlerpThreeQuaternions_2DRefLoc_EquidistantInterp)
{
  // Interpolating among three orthogonal quaternions with 2D reference locations and a center
  // interpolation point
  Core::LinAlg::Matrix<4, 1> q1(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<4, 1> q2(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<4, 1> q3(Core::LinAlg::Initialization::zero);
  q1(3, 0) = 1.0;  // w
  q2(0, 0) = 1.0;  // x
  q3(1, 0) = 1.0;  // y

  std::vector<Core::LinAlg::Matrix<4, 1>> quats = {q1, q2, q3};

  // Reference locations as 2D matrices (for loc_dim = 2), forming an equilateral triangle
  std::vector<Core::LinAlg::Matrix<2, 1>> ref_locs(3);
  ref_locs[0](0, 0) = 0.0;
  ref_locs[0](1, 0) = 0.0;  // (0,0)
  ref_locs[1](0, 0) = 1.0;
  ref_locs[1](1, 0) = 0.0;  // (1,0)
  ref_locs[2](0, 0) = 0.5;
  ref_locs[2](1, 0) = std::sqrt(3.0) / 2.0;  // (0.5, sqrt(3)/2)

  // Interpolation location at the centroid of the triangle (equidistant from all nodes)
  Core::LinAlg::Matrix<2, 1> interp_loc(Core::LinAlg::Initialization::zero);
  interp_loc(0, 0) = (ref_locs[0](0, 0) + ref_locs[1](0, 0) + ref_locs[2](0, 0)) / 3.0;
  interp_loc(1, 0) = (ref_locs[0](1, 0) + ref_locs[1](1, 0) + ref_locs[2](1, 0)) / 3.0;

  // Use inverse distance weighting function and default params
  auto weight_func = Core::LinAlg::ScalarInterpolation::WeightingFunction::inverse_distance;
  Core::LinAlg::ScalarInterpolation::InterpParams interp_params;

  Core::LinAlg::GeneralizedSphericalLinearInterpolator<2> interpolator(
      quats, ref_locs, weight_func, interp_params);

  Core::LinAlg::Matrix<4, 1> result = interpolator.get_interpolated_quaternion(&interp_loc);

  // For three orthogonal quaternions with equal 2D spacing and centroid interpolation, expect all
  // components to be equal (normalized)
  double inv_sqrt3 = 1.0 / std::sqrt(3.0);
  double norm = std::sqrt(result(0, 0) * result(0, 0) + result(1, 0) * result(1, 0) +
                          result(2, 0) * result(2, 0) + result(3, 0) * result(3, 0));
  // For three orthogonal quaternions with equal weights, the result should be close to normalized
  // (1/sqrt(3), 1/sqrt(3), 0, 1/sqrt(3))
  ASSERT_NEAR(norm, 1.0, 1e-12);
  ASSERT_NEAR(result(0, 0), inv_sqrt3, 1e-12);  // x
  ASSERT_NEAR(result(1, 0), inv_sqrt3, 1e-12);  // y
  ASSERT_NEAR(result(2, 0), 0.0, 1e-12);        // z
  ASSERT_NEAR(result(3, 0), inv_sqrt3, 1e-12);  // w
}

/**
 * @brief Unit test for generalized spherical linear interpolation (Slerp) among four quaternions.
 *
 * This test verifies the behavior of the GeneralizedSphericalLinearInterpolator when interpolating
 * between four orthogonal quaternions (representing the x, y, z, and w axes) with equal weights.
 *
 * The test sets up four quaternions:
 *   - q1: (1, 0, 0, 0)  // x-axis
 *   - q2: (0, 1, 0, 0)  // y-axis
 *   - q3: (0, 0, 1, 0)  // z-axis
 *   - q4: (0, 0, 0, 1)  // w-axis
 *
 * Each quaternion is assigned a weight of 1/4. The interpolator computes the resulting quaternion,
 * which is expected to be normalized and close to (0.5, 0.5, 0.5, 0.5).
 *
 * The test asserts that each component of the resulting quaternion matches the expected value
 * within a tolerance of 1e-12.
 */
TEST(QuaternionInterpolationTest, SlerpFourQuaternions)
{
  // Interpolating among four orthogonal quaternions with equal weights (x, y, z, w order)
  Core::LinAlg::Matrix<4, 1> q1(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<4, 1> q2(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<4, 1> q3(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<4, 1> q4(Core::LinAlg::Initialization::zero);
  q1(0, 0) = 1.0;  // x
  q2(1, 0) = 1.0;  // y
  q3(2, 0) = 1.0;  // z
  q4(3, 0) = 1.0;  // w

  std::vector<Core::LinAlg::Matrix<4, 1>> quats = {q1, q2, q3, q4};
  std::vector<double> weights = {0.25, 0.25, 0.25, 0.25};

  Core::LinAlg::GeneralizedSphericalLinearInterpolator<3> interpolator(quats, weights);
  Core::LinAlg::Matrix<4, 1> result = interpolator.get_interpolated_quaternion();

  // For four orthogonal quaternions with equal weights, the result should be close to normalized
  // (0.5, 0.5, 0.5, 0.5)
  ASSERT_NEAR(result(0, 0), 0.5, 1e-12);  // x
  ASSERT_NEAR(result(1, 0), 0.5, 1e-12);  // y
  ASSERT_NEAR(result(2, 0), 0.5, 1e-12);  // z
  ASSERT_NEAR(result(3, 0), 0.5, 1e-12);  // w
}

// Same test as above, but with 2D reference locations and a center interpolation point
// This test verifies that the interpolation works correctly with 2D reference locations
// and that the resulting quaternion is normalized and has the expected values.
TEST(QuaternionInterpolationTest, SlerpFourQuaternions_2DRefLoc_EquidistantInterp)
{
  // Interpolating among four orthogonal quaternions with equal weights (x, y, z, w order)
  Core::LinAlg::Matrix<4, 1> q1(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<4, 1> q2(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<4, 1> q3(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<4, 1> q4(Core::LinAlg::Initialization::zero);
  q1(0, 0) = 1.0;  // x
  q2(1, 0) = 1.0;  // y
  q3(2, 0) = 1.0;  // z
  q4(3, 0) = 1.0;  // w
  std::vector<Core::LinAlg::Matrix<4, 1>> quats = {q1, q2, q3, q4};

  // Reference locations as 2D matrices (for loc_dim = 2), forming a square
  std::vector<Core::LinAlg::Matrix<2, 1>> ref_locs(4);
  ref_locs[0](0, 0) = 0.0;
  ref_locs[0](1, 0) = 0.0;  // (0,0)
  ref_locs[1](0, 0) = 1.0;
  ref_locs[1](1, 0) = 0.0;  // (1,0)
  ref_locs[2](0, 0) = 1.0;
  ref_locs[2](1, 0) = 1.0;  // (1,1)
  ref_locs[3](0, 0) = 0.0;
  ref_locs[3](1, 0) = 1.0;  // (0,1)

  // Interpolation location at the center of the square (0.5, 0.5), equidistant from all nodes
  Core::LinAlg::Matrix<2, 1> interp_loc(Core::LinAlg::Initialization::zero);
  interp_loc(0, 0) = 0.5;
  interp_loc(1, 0) = 0.5;

  // Use inverse distance weighting function and default params
  auto weight_func = Core::LinAlg::ScalarInterpolation::WeightingFunction::inverse_distance;
  Core::LinAlg::ScalarInterpolation::InterpParams interp_params;

  Core::LinAlg::GeneralizedSphericalLinearInterpolator<2> interpolator(
      quats, ref_locs, weight_func, interp_params);

  Core::LinAlg::Matrix<4, 1> result = interpolator.get_interpolated_quaternion(&interp_loc);

  // For four orthogonal quaternions with equal 2D spacing and center interpolation, expect all
  // components to be equal
  double norm = std::sqrt(result(0, 0) * result(0, 0) + result(1, 0) * result(1, 0) +
                          result(2, 0) * result(2, 0) + result(3, 0) * result(3, 0));
  ASSERT_NEAR(norm, 1.0, 1e-12);
  ASSERT_NEAR(result(0, 0), 0.5, 1e-12);  // x
  ASSERT_NEAR(result(1, 0), 0.5, 1e-12);  // y
  ASSERT_NEAR(result(2, 0), 0.5, 1e-12);  // z
  ASSERT_NEAR(result(3, 0), 0.5, 1e-12);  // w
}

/**
 * @brief Unit test for quaternion spherical linear interpolation (SLERP) and generalized SLERP.
 *
 * This test verifies the correctness of the SLERP implementation between two quaternions:
 * - q1: Identity quaternion (0, 0, 0, 1)
 * - q2: 180-degree rotation about the x-axis (1, 0, 0, 0)
 *
 * The test performs the following checks:
 * 1. Computes the SLERP result at t = 0.25 (weighted interpolation: 75% q1, 25% q2) using the
 *    `Core::LinAlg::spherical_linear_interpolation` function.
 * 2. Calculates the expected analytical result for this interpolation using the SLERP formula.
 * 3. Asserts that the computed SLERP result matches the analytical result within a tight tolerance.
 * 4. Uses the `GeneralizedSphericalLinearInterpolator` with the same weights and verifies that its
 *    result matches the analytical expectation.
 *
 * The test ensures that both the direct SLERP and the generalized interpolator produce correct and
 * consistent results for a simple, analytically tractable case.
 */
TEST(QuaternionInterpolationTest, SlerpWeighted_AnalyticalCheck)
{
  // Weighted interpolation: 75% q1, 25% q2 (x, y, z, w order)
  Core::LinAlg::Matrix<4, 1> q1(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<4, 1> q2(Core::LinAlg::Initialization::zero);
  q1(3, 0) = 1.0;  // w (identity quaternion)
  q2(0, 0) = 1.0;  // x (180-degree rotation about x)

  // Use the SLERP function directly
  double t = 0.25;
  Core::LinAlg::Matrix<4, 1> slerp_result = Core::LinAlg::spherical_linear_interpolation(q1, q2, t);

  // Analytical result for SLERP between (0,0,0,1) and (1,0,0,0) at t=0.25:
  // angle = acos(0) = pi/2, so
  // slerp = (sin((1-t)*theta)*q1 + sin(t*theta)*q2) / sin(theta)
  // theta = pi/2, sin(theta) = 1
  // sin((1-t)*theta) = sin(0.75*pi/2) = sin(3*pi/8)
  // sin(t*theta) = sin(0.25*pi/2) = sin(pi/8)
  double theta = std::numbers::pi_v<double> / 2.0;
  double sin_theta = std::sin(theta);
  double sin_1mt_theta = std::sin((1.0 - t) * theta);
  double sin_t_theta = std::sin(t * theta);

  double x = sin_t_theta / sin_theta;
  double w = sin_1mt_theta / sin_theta;
  double norm = std::sqrt(w * w + x * x);

  double x_expected = x / norm;
  double w_expected = w / norm;

  ASSERT_NEAR(slerp_result(0, 0), x_expected, 1e-12);  // x
  ASSERT_NEAR(slerp_result(1, 0), 0.0, 1e-12);         // y
  ASSERT_NEAR(slerp_result(2, 0), 0.0, 1e-12);         // z
  ASSERT_NEAR(slerp_result(3, 0), w_expected, 1e-12);  // w

  // Also check the GeneralizedSphericalLinearInterpolator result matches
  std::vector<Core::LinAlg::Matrix<4, 1>> quats = {q1, q2};
  std::vector<double> weights = {0.75, 0.25};
  Core::LinAlg::GeneralizedSphericalLinearInterpolator interpolator(quats, weights);
  Core::LinAlg::Matrix<4, 1> result = interpolator.get_interpolated_quaternion();

  ASSERT_NEAR(result(0, 0), x_expected, 1e-12);  // x
  ASSERT_NEAR(result(1, 0), 0.0, 1e-12);         // y
  ASSERT_NEAR(result(2, 0), 0.0, 1e-12);         // z
  ASSERT_NEAR(result(3, 0), w_expected, 1e-12);  // w
}

FOUR_C_NAMESPACE_CLOSE
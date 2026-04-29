// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_linalg_utils_scalar_interpolation.hpp"

#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_unittest_utils_assertions_test.hpp"
#include "4C_utils_exceptions.hpp"

#include <array>
#include <cmath>
#include <iomanip>
#include <ios>
#include <utility>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace
{
  TEST(LinalgScalarInterpolationTest, CalculateNormalizedWeights1DInverseDistance)
  {
    using namespace Core::LinAlg;

    // Setup dimension of interpolation locations
    constexpr unsigned int loc_dim = 3;

    // Reference locations: -1.0, 1.0
    std::vector<Core::LinAlg::Matrix<loc_dim, 1>> ref_locs;
    Core::LinAlg::Matrix<loc_dim, 1> loc1(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<loc_dim, 1> loc2(Core::LinAlg::Initialization::zero);
    loc1(0, 0) = -1.0;
    loc2(0, 0) = 1.0;
    ref_locs.push_back(loc1);
    ref_locs.push_back(loc2);

    // Interpolation location: 0.5
    Core::LinAlg::Matrix<loc_dim, 1> interp_loc(Core::LinAlg::Initialization::zero);
    interp_loc(0, 0) = 0.5;

    // Weighting function: inverse distance
    ScalarInterpolationWeightingFunction weight_func =
        ScalarInterpolationWeightingFunction::inverse_distance;

    // Interpolation parameters
    ScalarInterpolationParams interp_params;
    interp_params.distance_threshold = 1e-12;
    interp_params.inverse_distance_power = 1.0;  // Use power of 1 for inverse distance

    // Call the function
    std::vector<double> weights = Core::LinAlg::calculate_normalized_weights(
        ref_locs, interp_loc, weight_func, interp_params);

    // So weights should be [0.25, 0.75]
    std::vector<double> weights_ref = {0.25, 0.75};

    ASSERT_EQ(weights.size(), 2);
    for (size_t i = 0; i < weights.size(); ++i) ASSERT_NEAR(weights[i], weights_ref[i], 1e-8);
  }


  TEST(LinalgScalarInterpolationTest, CalculateNormalizedWeights1DInverseDistanceAtRefLocations)
  {
    using namespace Core::LinAlg;

    // Setup dimension of interpolation locations
    constexpr unsigned int loc_dim = 3;

    // Reference locations: 0.0, 1.0
    std::vector<Core::LinAlg::Matrix<loc_dim, 1>> ref_locs;
    Core::LinAlg::Matrix<loc_dim, 1> loc1(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<loc_dim, 1> loc2(Core::LinAlg::Initialization::zero);
    loc1(0, 0) = 0.0;
    loc2(0, 0) = 1.0;
    ref_locs.push_back(loc1);
    ref_locs.push_back(loc2);

    // Interpolation location: 0.0
    Core::LinAlg::Matrix<loc_dim, 1> interp_loc(Core::LinAlg::Initialization::zero);
    interp_loc(0, 0) = 0.0;

    // Weighting function: inverse distance
    ScalarInterpolationWeightingFunction weight_func =
        ScalarInterpolationWeightingFunction::inverse_distance;

    // Interpolation parameters
    ScalarInterpolationParams interp_params;
    interp_params.distance_threshold = 1e-12;
    interp_params.inverse_distance_power = 1.0;  // Use power of 1 for inverse distance

    // Call the function
    std::vector<double> weights = Core::LinAlg::calculate_normalized_weights(
        ref_locs, interp_loc, weight_func, interp_params);

    // So weights should be [1.0, 0.0]
    std::vector<double> weights_ref = {1.0, 0.0};

    ASSERT_EQ(weights.size(), 2);
    for (size_t i = 0; i < weights.size(); ++i) ASSERT_NEAR(weights[i], weights_ref[i], 1e-8);


    // And the same for interpolation location: 1.0
    interp_loc(0, 0) = 1.0;
    weights = Core::LinAlg::calculate_normalized_weights(
        ref_locs, interp_loc, weight_func, interp_params);
    ASSERT_EQ(weights.size(), 2);
    weights_ref = {0.0, 1.0};
    for (size_t i = 0; i < weights.size(); ++i) ASSERT_NEAR(weights[i], weights_ref[i], 1e-8);
  }



  TEST(LinalgScalarInterpolationTest, CalculateNormalizedWeights1DExponential)
  {
    using namespace Core::LinAlg;

    // Setup dimension of interpolation locations
    constexpr unsigned int loc_dim = 3;

    // Reference locations: -1.0, 1.0
    std::vector<Core::LinAlg::Matrix<loc_dim, 1>> ref_locs;
    Core::LinAlg::Matrix<loc_dim, 1> loc1(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<loc_dim, 1> loc2(Core::LinAlg::Initialization::zero);
    loc1(0, 0) = -1.0;
    loc2(0, 0) = 1.0;
    ref_locs.push_back(loc1);
    ref_locs.push_back(loc2);

    // Interpolation location: 0.5
    Core::LinAlg::Matrix<loc_dim, 1> interp_loc(Core::LinAlg::Initialization::zero);
    interp_loc(0, 0) = 0.5;

    // Weighting function: exponential
    ScalarInterpolationWeightingFunction weight_func =
        ScalarInterpolationWeightingFunction::exponential;

    // Interpolation parameters
    ScalarInterpolationParams interp_params;
    interp_params.distance_threshold = 1e-12;
    interp_params.exponential_decay_c = 1.0;  // decay parameter

    // Call the function
    std::vector<double> weights = Core::LinAlg::calculate_normalized_weights(
        ref_locs, interp_loc, weight_func, interp_params);

    // Calculate expected weights manually
    // distances: |0.5 - (-1.0)| = 1.5, |0.5 - 1.0| = 0.5
    // weights: exp(-c * 1.5^2) = exp(-2.25), exp(-c * 0.5^2) = exp(-0.25)
    double w0 = std::exp(-interp_params.exponential_decay_c * 1.5 * 1.5);
    double w1 = std::exp(-interp_params.exponential_decay_c * 0.5 * 0.5);
    double sum = w0 + w1;
    std::vector<double> weights_ref = {w0 / sum, w1 / sum};

    ASSERT_EQ(weights.size(), 2);
    for (size_t i = 0; i < weights.size(); ++i) ASSERT_NEAR(weights[i], weights_ref[i], 1e-8);
  }


  TEST(LinalgScalarInterpolationTest, CalculateNormalizedWeights1DExponentialAtRefLocations)
  {
    using namespace Core::LinAlg;

    // Setup dimension of interpolation locations
    constexpr unsigned int loc_dim = 3;

    // Reference locations: 0.0, 1.0
    std::vector<Core::LinAlg::Matrix<loc_dim, 1>> ref_locs;
    Core::LinAlg::Matrix<loc_dim, 1> loc1(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<loc_dim, 1> loc2(Core::LinAlg::Initialization::zero);
    loc1(0, 0) = 0.0;
    loc2(0, 0) = 1.0;
    ref_locs.push_back(loc1);
    ref_locs.push_back(loc2);

    // Interpolation location: 0.0
    Core::LinAlg::Matrix<loc_dim, 1> interp_loc(Core::LinAlg::Initialization::zero);
    interp_loc(0, 0) = 0.0;

    // Weighting function: exponential
    ScalarInterpolationWeightingFunction weight_func =
        ScalarInterpolationWeightingFunction::exponential;

    // Interpolation parameters
    ScalarInterpolationParams interp_params;
    interp_params.distance_threshold = 1e-12;
    interp_params.exponential_decay_c = 1.0;  // decay parameter

    // Call the function
    std::vector<double> weights = Core::LinAlg::calculate_normalized_weights(
        ref_locs, interp_loc, weight_func, interp_params);

    // Expected weights: 1.0 and 0.0
    std::vector<double> weights_ref = {1.0, 0.0};

    ASSERT_EQ(weights.size(), 2);
    for (size_t i = 0; i < weights.size(); ++i) ASSERT_NEAR(weights[i], weights_ref[i], 1e-8);


    // And the same for interpolation location: 1.0
    interp_loc(0, 0) = 1.0;
    weights = Core::LinAlg::calculate_normalized_weights(
        ref_locs, interp_loc, weight_func, interp_params);
    ASSERT_EQ(weights.size(), 2);
    weights_ref = {0.0, 1.0};
    for (size_t i = 0; i < weights.size(); ++i) ASSERT_NEAR(weights[i], weights_ref[i], 1e-8);
  }


  TEST(LinalgScalarInterpolationTest, LogInterpolation_ManualData)
  {
    // see Satheesh et al., 2024, 10.1002/nme.7373, Section 2.4.4 Comparison of eigenvalue
    // interpolation methods

    using namespace Core::LinAlg;

    constexpr unsigned int loc_dim = 1;

    // Reference locations and scalar data
    std::vector<Core::LinAlg::Matrix<loc_dim, 1>> ref_locs;
    std::vector<std::vector<double>> scalar_data;

    Core::LinAlg::Matrix<loc_dim, 1> x1(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<loc_dim, 1> x2(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<loc_dim, 1> x3(Core::LinAlg::Initialization::zero);
    x1(0, 0) = 1.0;
    x2(0, 0) = 2.0;
    x3(0, 0) = 3.0;
    ref_locs.push_back(x1);
    ref_locs.push_back(x2);
    ref_locs.push_back(x3);

    scalar_data.push_back({0.1});
    scalar_data.push_back({0.1});
    scalar_data.push_back({1.0});

    ScalarInterpolationWeightingFunction weight_func =
        ScalarInterpolationWeightingFunction::exponential;
    ScalarInterpolationParams interp_params;
    interp_params.distance_threshold = 1e-12;
    interp_params.exponential_decay_c = 10.0;

    ScalarInterpolator<loc_dim> interpolator(
        ScalarInterpolationType::logarithmic_weighted_average, weight_func, interp_params);

    // Interpolate at several points in [1, 3]
    std::vector<double> interp_points = {
        1.0, 1.14141414141414, 2.33333333333333, 2.91919191919192, 3.0};
    std::vector<double> expected = {0.1, 0.1, 0.10825430938265, 0.999474046355807, 1.0};
    for (size_t i = 0; i < interp_points.size(); ++i)
    {
      Core::LinAlg::Matrix<loc_dim, 1> interp_loc(Core::LinAlg::Initialization::zero);
      interp_loc(0, 0) = interp_points[i];

      std::vector<double> result =
          interpolator.get_interpolated_scalar(scalar_data, ref_locs, interp_loc);

      ASSERT_EQ(result.size(), 1);
      ASSERT_NEAR(result[0], expected[i], 1e-8);
    }
  }

  TEST(LinalgScalarInterpolationTest, MLSInterpolation_ManualData)
  {
    // see Satheesh et al., 2024, 10.1002/nme.7373, Section 2.4.4 Comparison of eigenvalue
    // interpolation methods

    using namespace Core::LinAlg;

    constexpr unsigned int loc_dim = 1;
    constexpr unsigned int poly_order = 2;
    constexpr unsigned int num_cofeffcients = 3;

    // Reference locations and scalar data
    std::vector<Core::LinAlg::Matrix<loc_dim, 1>> ref_locs;
    std::vector<std::vector<double>> scalar_data;

    Core::LinAlg::Matrix<loc_dim, 1> x1(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<loc_dim, 1> x2(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<loc_dim, 1> x3(Core::LinAlg::Initialization::zero);
    x1(0, 0) = 1.0;
    x2(0, 0) = 2.0;
    x3(0, 0) = 3.0;
    ref_locs.push_back(x1);
    ref_locs.push_back(x2);
    ref_locs.push_back(x3);

    scalar_data.push_back({0.1, 0.1, 0.1});
    scalar_data.push_back({0.1, 0.1, 0.1});
    scalar_data.push_back({1.0, 1.0, 1.0});

    ScalarInterpolationWeightingFunction weight_func =
        ScalarInterpolationWeightingFunction::exponential;
    ScalarInterpolationParams interp_params;
    interp_params.distance_threshold = 1e-12;
    interp_params.exponential_decay_c = 1.0;

    ScalarInterpolator<loc_dim, poly_order, num_cofeffcients> interpolator(
        ScalarInterpolationType::moving_least_squares, weight_func, interp_params);

    // Interpolate at several points in [1, 3]
    std::vector<double> interp_points = {
        1.0, 1.14141414141414, 2.33333333333333, 2.91919191919192, 3.0};
    std::vector<double> expected = {
        0.1, 0.0453627180900034, 0.300000000000004, 0.893847566574832, 1.0};
    for (size_t i = 0; i < interp_points.size(); ++i)
    {
      Core::LinAlg::Matrix<loc_dim, 1> interp_loc(Core::LinAlg::Initialization::zero);
      interp_loc(0, 0) = interp_points[i];

      std::vector<double> result =
          interpolator.get_interpolated_scalar(scalar_data, ref_locs, interp_loc);

      ASSERT_EQ(result.size(), scalar_data[0].size());
      for (size_t j = 0; j < result.size(); ++j) ASSERT_NEAR(result[j], expected[i], 1e-8);
    }
  }

  TEST(LinalgScalarInterpolationTest, MLSInterpolation_3DTrilinear_Hex8Nodes)
  {
    // -----------------------------------------------------------------------------
    // HEX8 Linear Element Analogy
    // The 8 input nodes represent the corners of a reference hexahedral (HEX8) element,
    // positioned in the natural coordinate system:
    //      $ (\xi, \eta, \zeta) \in \{ -1, +1 \}^3 $
    //
    // Each node is assigned a scalar value (e.g., 1 through 8)
    // The interpolation is performed at the centroid of the element:
    //      \xi = 0, \eta = 0, \zeta = 0
    // For a HEX8 element with trilinear shape functions, this is the center of the cube.
    // All shape functions evaluate to 1/8 at the center:
    //      N_i(0, 0, 0) = \frac{1}{8}, \quad \text{for } i = 1 \ldots 8
    // Use FEM interpolation formula:
    //      u_\text{centroid} = \sum_{i=1}^8 N_i u_i
    // Since all N_i = 1/8:
    //      u_\text{centroid} = \frac{1}{8} \sum_{i=1}^8 u_i
    //                       = \frac{1}{8} (1 + 2 + \ldots + 8)
    //                       = 4.5
    // -----------------------------------------------------------------------------

    using namespace Core::LinAlg;

    constexpr unsigned int loc_dim = 3;
    constexpr unsigned int poly_order = 1;
    constexpr unsigned int num_cofeffcients = 8;

    // Reference locations (cube nodes)
    std::vector<Core::LinAlg::Matrix<loc_dim, 1>> ref_locs;
    std::vector<std::vector<double>> scalar_data;

    // Node coordinates as per your table
    double coords[8][3] = {{-1, -1, -1}, {+1, -1, -1}, {+1, +1, -1}, {-1, +1, -1}, {-1, -1, +1},
        {+1, -1, +1}, {+1, +1, +1}, {-1, +1, +1}};

    // Fill ref_locs and scalar_data (values 1,2,...,8)
    for (int i = 0; i < 8; ++i)
    {
      Core::LinAlg::Matrix<loc_dim, 1> node(Core::LinAlg::Initialization::zero);
      node(0, 0) = coords[i][0];
      node(1, 0) = coords[i][1];
      node(2, 0) = coords[i][2];
      ref_locs.push_back(node);
      scalar_data.push_back({static_cast<double>(i + 1)});
    }

    ScalarInterpolationWeightingFunction weight_func =
        ScalarInterpolationWeightingFunction::exponential;
    ScalarInterpolationParams interp_params;
    interp_params.distance_threshold = 1e-12;
    interp_params.exponential_decay_c = 1.0;

    ScalarInterpolator<loc_dim, poly_order, num_cofeffcients> interpolator(
        ScalarInterpolationType::moving_least_squares, weight_func, interp_params);

    // Interpolate at the center of the cube (0,0,0)
    Core::LinAlg::Matrix<loc_dim, 1> interp_loc(Core::LinAlg::Initialization::zero);
    interp_loc(0, 0) = 0.0;
    interp_loc(1, 0) = 0.0;
    interp_loc(2, 0) = 0.0;

    std::vector<double> result =
        interpolator.get_interpolated_scalar(scalar_data, ref_locs, interp_loc);

    ASSERT_EQ(result.size(), 1);
    ASSERT_NEAR(result[0], 4.5, 1e-8);
  }

  TEST(LinalgScalarInterpolationTest, MLSInterpolation_3DQuadratic_Hex20Nodes)
  {
    // -----------------------------------------------------------------------------
    // HEX20 Quadratic Serendipity Element Analogy
    // The 20 input nodes represent the corners and mid-edge nodes of a reference
    // hexahedral (HEX20) element, positioned in the natural coordinate system:
    //
    //      Corner nodes at $(\xi, \eta, \zeta) \in \{ -1, +1 \}^3$
    //      Mid-edge nodes lie at the midpoint of edges where one coordinate is zero
    //      and the other two are $\pm1$.
    //
    // Each node is assigned a scalar value (e.g., 1 through 20).
    // The interpolation is performed at the centroid of the element:
    //
    //      $\xi = 0, \eta = 0, \zeta = 0$
    //
    // At the centroid, the HEX20 shape functions evaluate as:
    //      - Corner nodes: $N_i(0, 0, 0) = -0.25$ (for $i = 1 \ldots 8$)
    //      - Mid-edge nodes: $N_i(0, 0, 0) = 0.25$ (for $i = 9 \ldots 20$)
    //
    // Using the FEM interpolation formula:
    //      $u_\text{centroid} = \sum N_i u_i$
    //
    // For nodal values $u_i = 1 \ldots 20$, this evaluates to:
    //      $u_\text{centroid} = (-0.25) \sum_{i=1}^8 u_i + 0.25 \sum_{i=9}^{20} u_i$
    //                        $= (-0.25) \times 36 + 0.25 \times 174$
    //                        $= -9 + 43.5$
    //                        $= 34.5$
    // Thus, the displacement at the centroid is $34.5$.
    // -----------------------------------------------------------------------------

    using namespace Core::LinAlg;

    constexpr unsigned int loc_dim = 3;
    constexpr unsigned int poly_order = 2;
    constexpr unsigned int num_cofeffcients = 10;

    // 20-node HEX element: 8 corners + 12 mid-edge nodes
    double coords[20][3] = {
        {-1, -1, -1},  // 1
        {+1, -1, -1},  // 2
        {+1, +1, -1},  // 3
        {-1, +1, -1},  // 4
        {-1, -1, +1},  // 5
        {+1, -1, +1},  // 6
        {+1, +1, +1},  // 7
        {-1, +1, +1},  // 8
        {0, -1, -1},   // 9   (1-2)
        {+1, 0, -1},   // 10  (2-3)
        {0, +1, -1},   // 11  (3-4)
        {-1, 0, -1},   // 12  (4-1)
        {0, -1, +1},   // 13  (5-6)
        {+1, 0, +1},   // 14  (6-7)
        {0, +1, +1},   // 15  (7-8)
        {-1, 0, +1},   // 16  (8-5)
        {-1, -1, 0},   // 17  (1-5)
        {+1, -1, 0},   // 18  (2-6)
        {+1, +1, 0},   // 19  (3-7)
        {-1, +1, 0}    // 20  (4-8)
    };

    std::vector<Core::LinAlg::Matrix<loc_dim, 1>> ref_locs;
    std::vector<std::vector<double>> scalar_data;

    for (int i = 0; i < 20; ++i)
    {
      Core::LinAlg::Matrix<loc_dim, 1> node(Core::LinAlg::Initialization::zero);
      node(0, 0) = coords[i][0];
      node(1, 0) = coords[i][1];
      node(2, 0) = coords[i][2];
      ref_locs.push_back(node);
      scalar_data.push_back({static_cast<double>(i + 1)});
    }

    ScalarInterpolationWeightingFunction weight_func =
        ScalarInterpolationWeightingFunction::exponential;
    ScalarInterpolationParams interp_params;
    interp_params.distance_threshold = 1e-12;
    interp_params.exponential_decay_c = 1.0;

    ScalarInterpolator<loc_dim, poly_order, num_cofeffcients> interpolator(
        ScalarInterpolationType::moving_least_squares, weight_func, interp_params);

    // Interpolate at the center of the cube (0,0,0)
    Core::LinAlg::Matrix<loc_dim, 1> interp_loc(Core::LinAlg::Initialization::zero);

    std::vector<double> result =
        interpolator.get_interpolated_scalar(scalar_data, ref_locs, interp_loc);

    ASSERT_EQ(result.size(), 1);
    ASSERT_NEAR(result[0], 34.5, 1e-8);
  }

  TEST(LinalgScalarInterpolationTest, LOGMLSInterpolation_ManualData)
  {
    // see Satheesh et al., 2024, 10.1002/nme.7373, Section 2.4.4 Comparison of eigenvalue
    // interpolation methods

    using namespace Core::LinAlg;

    constexpr unsigned int loc_dim = 1;
    constexpr unsigned int poly_order = 2;
    constexpr unsigned int num_cofeffcients = 3;

    // Reference locations and scalar data
    std::vector<Core::LinAlg::Matrix<loc_dim, 1>> ref_locs;
    std::vector<std::vector<double>> scalar_data;

    Core::LinAlg::Matrix<loc_dim, 1> x1(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<loc_dim, 1> x2(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<loc_dim, 1> x3(Core::LinAlg::Initialization::zero);
    x1(0, 0) = 1.0;
    x2(0, 0) = 2.0;
    x3(0, 0) = 3.0;
    ref_locs.push_back(x1);
    ref_locs.push_back(x2);
    ref_locs.push_back(x3);

    scalar_data.push_back({0.1, 0.1, 0.1});
    scalar_data.push_back({0.1, 0.1, 0.1});
    scalar_data.push_back({1.0, 1.0, 1.0});

    ScalarInterpolationWeightingFunction weight_func =
        ScalarInterpolationWeightingFunction::exponential;
    ScalarInterpolationParams interp_params;
    interp_params.distance_threshold = 1e-12;
    interp_params.exponential_decay_c = 1.0;

    ScalarInterpolator<loc_dim, poly_order, num_cofeffcients> interpolator(
        ScalarInterpolationType::logarithmic_moving_least_squares, weight_func, interp_params);

    // Interpolate at several points in [1, 3]
    std::vector<double> interp_points = {
        1.0, 1.14141414141414, 2.33333333333333, 2.91919191919192, 3.0};
    std::vector<double> expected = {
        0.1, 0.0869544693275927, 0.166810053719997, 0.762171757370172, 1.0};
    for (size_t i = 0; i < interp_points.size(); ++i)
    {
      Core::LinAlg::Matrix<loc_dim, 1> interp_loc(Core::LinAlg::Initialization::zero);
      interp_loc(0, 0) = interp_points[i];

      std::vector<double> result =
          interpolator.get_interpolated_scalar(scalar_data, ref_locs, interp_loc);

      ASSERT_EQ(result.size(), scalar_data[0].size());
      for (size_t j = 0; j < result.size(); ++j) ASSERT_NEAR(result[j], expected[i], 1e-8);
    }
  }

}  // namespace
FOUR_C_NAMESPACE_CLOSE

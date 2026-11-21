// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_mortar_coupling3d_classes.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>

namespace
{
  using namespace FourC;

  using Derivative = Core::Gen::Pairedvector<int, double>;
  using CoordinateLinearization = std::vector<Derivative>;

  class IntCellTest : public ::testing::Test
  {
   protected:
    static constexpr int linearization_capacity = 32;

    IntCellTest()
        : vertex_linearizations_{
          empty_coordinate_linearization(),
          empty_coordinate_linearization(),
          empty_coordinate_linearization(),
      },
      auxiliary_normal_linearization_(empty_coordinate_linearization())
    {
      for (std::size_t vertex = 0; vertex < 3; ++vertex)
      {
        for (std::size_t dim = 0; dim < 3; ++dim)
        {
          const int degree_of_freedom = static_cast<int>(3 * vertex + dim);
          vertex_linearizations_[vertex][dim][degree_of_freedom] = 1.0;
        }
      }
    }

    static CoordinateLinearization empty_coordinate_linearization()
    {
      CoordinateLinearization linearization;
      linearization.reserve(3);

      for (std::size_t dim = 0; dim < 3; ++dim) linearization.emplace_back(linearization_capacity);

      return linearization;
    }

    std::unique_ptr<Mortar::IntCell> make_cell(
        Core::LinAlg::Matrix<3, 3> coordinates, const Core::FE::CellType shape)
    {
      return std::make_unique<Mortar::IntCell>(coordinates, auxiliary_normal_.data(), shape,
          vertex_linearizations_[0], vertex_linearizations_[1], vertex_linearizations_[2],
          auxiliary_normal_linearization_);
    }

    static Core::LinAlg::Matrix<3, 3> triangle_coordinates()
    {
      Core::LinAlg::Matrix<3, 3> coordinates(Core::LinAlg::Initialization::zero);

      // x_0 = (1, 2, 3)
      coordinates(0, 0) = 1.0;
      coordinates(1, 0) = 2.0;
      coordinates(2, 0) = 3.0;

      // x_1 = (3, 2, 4)
      coordinates(0, 1) = 3.0;
      coordinates(1, 1) = 2.0;
      coordinates(2, 1) = 4.0;

      // x_2 = (1, 5, 3)
      coordinates(0, 2) = 1.0;
      coordinates(1, 2) = 5.0;
      coordinates(2, 2) = 3.0;

      return coordinates;
    }

    static Core::LinAlg::Matrix<3, 3> segment_coordinates()
    {
      Core::LinAlg::Matrix<3, 3> coordinates(Core::LinAlg::Initialization::zero);

      // x_0 = (1, 2, 3)
      coordinates(0, 0) = 1.0;
      coordinates(1, 0) = 2.0;
      coordinates(2, 0) = 3.0;

      // x_1 = (4, 6, 15)
      coordinates(0, 1) = 4.0;
      coordinates(1, 1) = 6.0;
      coordinates(2, 1) = 15.0;

      return coordinates;
    }

    std::array<CoordinateLinearization, 3> vertex_linearizations_;
    CoordinateLinearization auxiliary_normal_linearization_;
    std::array<double, 3> auxiliary_normal_{0.0, 0.0, 1.0};
  };

  // Test for a triangular integration cell, area() is the physical area A and
  // jacobian() is twice the physical area 2A.
  TEST_F(IntCellTest, TriangleAreaAndJacobian)
  {
    auto cell = make_cell(triangle_coordinates(), Core::FE::CellType::tri3);

    // (x_1 - x_0) x (x_2 - x_0) = (-3, 0, 6)
    const double expected_jacobian = 3.0 * std::sqrt(5.0);
    const double expected_area = 0.5 * expected_jacobian;

    EXPECT_NEAR(cell->area(), expected_area, 1.0e-14);
    EXPECT_NEAR(cell->jacobian(), expected_jacobian, 1.0e-14);
  }

  // Test that triangle shape functions interpolate the physical coordinates using
  // barycentric coordinates.
  TEST_F(IntCellTest, TriangleShapeFunctionsAndMapping)
  {
    auto cell = make_cell(triangle_coordinates(), Core::FE::CellType::tri3);
    const std::array<double, 2> local_coordinate = {0.2, 0.3};

    Core::LinAlg::Matrix<3, 1> values(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<2, 3> derivatives(Core::LinAlg::Initialization::zero);

    ASSERT_TRUE(cell->evaluate_shape(local_coordinate.data(), values, derivatives));

    EXPECT_DOUBLE_EQ(values(0), 0.5);
    EXPECT_DOUBLE_EQ(values(1), 0.2);
    EXPECT_DOUBLE_EQ(values(2), 0.3);
    EXPECT_DOUBLE_EQ(values(0) + values(1) + values(2), 1.0);

    EXPECT_DOUBLE_EQ(derivatives(0, 0), -1.0);
    EXPECT_DOUBLE_EQ(derivatives(1, 0), -1.0);
    EXPECT_DOUBLE_EQ(derivatives(0, 1), 1.0);
    EXPECT_DOUBLE_EQ(derivatives(1, 1), 0.0);
    EXPECT_DOUBLE_EQ(derivatives(0, 2), 0.0);
    EXPECT_DOUBLE_EQ(derivatives(1, 2), 1.0);

    std::array<double, 3> global_coordinate{};
    ASSERT_TRUE(cell->local_to_global(local_coordinate.data(), global_coordinate.data(), 0));

    EXPECT_NEAR(global_coordinate[0], 1.4, 1.0e-14);
    EXPECT_NEAR(global_coordinate[1], 2.9, 1.0e-14);
    EXPECT_NEAR(global_coordinate[2], 3.2, 1.0e-14);

    // Mapping with shape-function derivatives returns the two tangents.
    std::array<double, 3> tangent_xi{};
    ASSERT_TRUE(cell->local_to_global(local_coordinate.data(), tangent_xi.data(), 1));

    EXPECT_DOUBLE_EQ(tangent_xi[0], 2.0);
    EXPECT_DOUBLE_EQ(tangent_xi[1], 0.0);
    EXPECT_DOUBLE_EQ(tangent_xi[2], 1.0);

    std::array<double, 3> tangent_eta{};
    ASSERT_TRUE(cell->local_to_global(local_coordinate.data(), tangent_eta.data(), 2));

    EXPECT_DOUBLE_EQ(tangent_eta[0], 0.0);
    EXPECT_DOUBLE_EQ(tangent_eta[1], 3.0);
    EXPECT_DOUBLE_EQ(tangent_eta[2], 0.0);
  }

  // Test that the analytical Jacobian linearization agrees with central finite
  // differences with respect to every vertex coordinate.
  TEST_F(IntCellTest, TriangleJacobianLinearization)
  {
    const auto coordinates = triangle_coordinates();
    auto cell = make_cell(coordinates, Core::FE::CellType::tri3);

    Derivative analytical_derivative(linearization_capacity);
    cell->deriv_jacobian(analytical_derivative);

    constexpr double perturbation = 1.0e-7;

    for (std::size_t vertex = 0; vertex < 3; ++vertex)
    {
      for (std::size_t dim = 0; dim < 3; ++dim)
      {
        auto coordinates_plus = coordinates;
        auto coordinates_minus = coordinates;

        coordinates_plus(dim, vertex) += perturbation;
        coordinates_minus(dim, vertex) -= perturbation;

        const double jacobian_plus =
            make_cell(coordinates_plus, Core::FE::CellType::tri3)->jacobian();

        const double jacobian_minus =
            make_cell(coordinates_minus, Core::FE::CellType::tri3)->jacobian();

        const double finite_difference = (jacobian_plus - jacobian_minus) / (2.0 * perturbation);
        const int degree_of_freedom = static_cast<int>(3 * vertex + dim);
        const double tolerance = 1.0e-7 * std::max(1.0, std::abs(finite_difference));

        EXPECT_NEAR(analytical_derivative[degree_of_freedom], finite_difference, tolerance);
      }
    }
  }

  // For a line integration cell, area() stores its length L and jacobian()
  // contains the reference-to-physical mapping factor L/2.
  TEST_F(IntCellTest, SegmentLengthAndJacobian)
  {
    auto cell = make_cell(segment_coordinates(), Core::FE::CellType::line2);

    EXPECT_DOUBLE_EQ(cell->area(), 13.0);
    EXPECT_DOUBLE_EQ(cell->jacobian(), 6.5);
  }

  // Test that segment shape functions interpolate the physical coordinates
  TEST_F(IntCellTest, SegmentShapeFunctionsAndMapping)
  {
    auto cell = make_cell(segment_coordinates(), Core::FE::CellType::line2);
    const std::array<double, 2> local_coordinate = {0.25, 0.0};

    Core::LinAlg::Matrix<3, 1> values(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<2, 3> derivatives(Core::LinAlg::Initialization::zero);

    ASSERT_TRUE(cell->evaluate_shape(local_coordinate.data(), values, derivatives));

    EXPECT_DOUBLE_EQ(values(0), 0.375);
    EXPECT_DOUBLE_EQ(values(1), 0.625);
    EXPECT_DOUBLE_EQ(values(0) + values(1), 1.0);

    EXPECT_DOUBLE_EQ(derivatives(0, 0), -0.5);
    EXPECT_DOUBLE_EQ(derivatives(0, 1), 0.5);

    std::array<double, 3> global_coordinate{};
    ASSERT_TRUE(cell->local_to_global(local_coordinate.data(), global_coordinate.data(), 0));

    EXPECT_DOUBLE_EQ(global_coordinate[0], 2.875);
    EXPECT_DOUBLE_EQ(global_coordinate[1], 4.5);
    EXPECT_DOUBLE_EQ(global_coordinate[2], 10.5);

    std::array<double, 3> tangent{};
    ASSERT_TRUE(cell->local_to_global(local_coordinate.data(), tangent.data(), 1));

    EXPECT_DOUBLE_EQ(tangent[0], 1.5);
    EXPECT_DOUBLE_EQ(tangent[1], 2.0);
    EXPECT_DOUBLE_EQ(tangent[2], 6.0);
  }

  // Test that the analytical segment Jacobian linearization agrees with central finite
  // differences with respect to every coordinate of both segment vertices.
  TEST_F(IntCellTest, SegmentJacobianLinearization)
  {
    const auto coordinates = segment_coordinates();
    auto cell = make_cell(coordinates, Core::FE::CellType::line2);

    Derivative analytical_derivative(linearization_capacity);
    cell->deriv_jacobian(analytical_derivative);

    constexpr double perturbation = 1.0e-7;

    for (std::size_t vertex = 0; vertex < 2; ++vertex)
    {
      for (std::size_t dim = 0; dim < 3; ++dim)
      {
        auto coordinates_plus = coordinates;
        auto coordinates_minus = coordinates;

        coordinates_plus(dim, vertex) += perturbation;
        coordinates_minus(dim, vertex) -= perturbation;

        const double jacobian_plus =
            make_cell(coordinates_plus, Core::FE::CellType::line2)->jacobian();

        const double jacobian_minus =
            make_cell(coordinates_minus, Core::FE::CellType::line2)->jacobian();

        const double finite_difference = (jacobian_plus - jacobian_minus) / (2.0 * perturbation);
        const int degree_of_freedom = static_cast<int>(3 * vertex + dim);
        const double tolerance = 1.0e-7 * std::max(1.0, std::abs(finite_difference));

        EXPECT_NEAR(analytical_derivative[degree_of_freedom], finite_difference, tolerance);
      }
    }
  }
}  // namespace
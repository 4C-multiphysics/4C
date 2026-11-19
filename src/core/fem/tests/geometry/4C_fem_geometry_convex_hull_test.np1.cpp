// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_fem_geometry_polygon_utils.hpp"

#include <algorithm>
#include <array>
#include <vector>

namespace
{
  using namespace FourC;

  constexpr double tolerance = 1.0e-12;

  // Convex hull tests
  namespace ConvexHull2D
  {
    using Points = std::vector<std::array<double, 2>>;

    // Compute the convex hull and return its points
    Points convex_hull(const Points& points)
    {
      Points hull;
      for (int i : Core::Geo::convex_hull_2d(points)) hull.push_back(points[i]);
      return hull;
    }

    // The hull consists of the corners (no inner points, no points on edges), consecutive along the
    // hull, starting at an arbitrary corner and running in an arbitrary direction.
    TEST(ConvexHull2D, CornersAreConsecutive)
    {
      const Points shuffled_square_with_inner_and_edge_points = {
          {1.0, 1.0}, {0.5, 0.5}, {0.5, 0.0}, {0.0, 0.0}, {1.0, 0.0}, {1.0, 0.5}, {0.0, 1.0}};
      const Points square = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}, {1.0, 0.0}};

      Points hull = convex_hull(shuffled_square_with_inner_and_edge_points);
      std::ranges::rotate(hull, std::ranges::find(hull, square[0]));
      if (hull.size() > 1 and hull[1] != square[1]) std::reverse(hull.begin() + 1, hull.end());

      EXPECT_EQ(hull, square);
    }

    // If the hull degenerates to a line or a point, it is empty.
    TEST(ConvexHull2D, DegenerateHullIsEmpty)
    {
      EXPECT_TRUE(convex_hull({{0.0, 0.0}, {2.0, 0.0}, {1.0, 0.0}}).empty());
      EXPECT_TRUE(convex_hull({{1.0, 1.0}, {1.0, 1.0}, {1.0, 1.0}}).empty());
      EXPECT_TRUE(convex_hull({{0.0, 0.0}, {1.0, 0.0}}).empty());
    }
  }  // namespace ConvexHull2D


  // Polygon center tests
  namespace PolygonCenter
  {
    void expect_center_near(
        const std::optional<std::array<double, 3>>& center, const std::array<double, 3>& expected)
    {
      ASSERT_TRUE(center.has_value());

      for (std::size_t dim = 0; dim < 3; ++dim)
        EXPECT_NEAR((*center)[dim], expected[dim], tolerance);
    }

    // Test that the center of a triangle is the arithmetic mean of its vertices in a plane.
    TEST(PolygonCenter, TriangleCenterIsVertexAverage)
    {
      const std::vector<std::array<double, 3>> triangle = {
          {0.0, 0.0, 2.0}, {3.0, 0.0, 2.0}, {0.0, 3.0, 2.0}};

      expect_center_near(Core::Geo::compute_polygon_center(triangle, 0.0), {1.0, 1.0, 2.0});
    }

    // Test that the center is weighted by area and is generally different from the arithmetic mean
    // of the polygon vertices in a plane.
    TEST(PolygonCenter, ComputesAreaWeightedCenter)
    {
      const std::vector<std::array<double, 3>> trapezoid = {
          {0.0, 0.0, 3.0},
          {4.0, 0.0, 3.0},
          {3.0, 2.0, 3.0},
          {0.0, 2.0, 3.0},
      };

      expect_center_near(
          Core::Geo::compute_polygon_center(trapezoid, 0.0), {37.0 / 21.0, 20.0 / 21.0, 3.0});
    }

    // Test that the calculation works for polygons in arbitrarily oriented planes.
    TEST(PolygonCenter, ComputesCenterInThreeDimensionalSpace)
    {
      const std::vector<std::array<double, 3>> rectangle = {
          {2.0, -1.0, 4.0},
          {4.0, -1.0, 6.0},
          {4.0, 1.0, 5.0},
          {2.0, 1.0, 3.0},
      };

      expect_center_near(Core::Geo::compute_polygon_center(rectangle, 0.0), {3.0, 0.0, 4.5});
    }

    // Test that the polygon orientation does not change its center calculation.
    TEST(PolygonCenter, IsIndependentOfVertexOrientation)
    {
      const std::vector<std::array<double, 3>> counterclockwise = {
          {0.0, 0.0, 0.0},
          {2.0, 0.0, 0.0},
          {2.0, 1.0, 0.0},
          {0.0, 1.0, 0.0},
      };

      auto clockwise = counterclockwise;
      std::ranges::reverse(clockwise);

      const auto counterclockwise_center = Core::Geo::compute_polygon_center(counterclockwise, 0.0);

      const auto clockwise_center = Core::Geo::compute_polygon_center(clockwise, 0.0);

      ASSERT_TRUE(counterclockwise_center.has_value());
      ASSERT_TRUE(clockwise_center.has_value());

      for (std::size_t dim = 0; dim < 3; ++dim)
        EXPECT_NEAR((*clockwise_center)[dim], (*counterclockwise_center)[dim], tolerance);
    }

    // Test that a polygon whose area does not exceed the prescribed tolerance has no center.
    TEST(PolygonCenter, AreaToleranceRejectsPolygon)
    {
      const std::vector<std::array<double, 3>> unit_square = {
          {0.0, 0.0, 0.0},
          {1.0, 0.0, 0.0},
          {1.0, 1.0, 0.0},
          {0.0, 1.0, 0.0},
      };

      EXPECT_FALSE(Core::Geo::compute_polygon_center(unit_square, 1.0).has_value());
      EXPECT_TRUE(Core::Geo::compute_polygon_center(unit_square, 0.5).has_value());
    }

    // Test that a polygon degenerating to a line or a point has zero area and therefore no center.
    TEST(PolygonCenter, DegeneratePolygonHasNoCenter)
    {
      const std::vector<std::array<double, 3>> line = {
          {0.0, 0.0, 0.0},
          {1.0, 0.0, 0.0},
          {2.0, 0.0, 0.0},
      };

      const std::vector<std::array<double, 3>> point = {
          {1.0, 2.0, 3.0},
          {1.0, 2.0, 3.0},
          {1.0, 2.0, 3.0},
      };

      EXPECT_FALSE(Core::Geo::compute_polygon_center(line, 0.0).has_value());
      EXPECT_FALSE(Core::Geo::compute_polygon_center(point, 0.0).has_value());
    }
  }  // namespace PolygonCenter
}  // namespace

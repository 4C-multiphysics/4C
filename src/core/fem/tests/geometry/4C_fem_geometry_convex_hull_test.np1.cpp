// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_fem_geometry_convex_hull.hpp"

#include <algorithm>
#include <array>
#include <vector>

namespace
{
  using namespace FourC;

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
        const std::array<double, 3>& center, const std::array<double, 3>& expected)
    {
      for (std::size_t dim = 0; dim < 3; ++dim) EXPECT_NEAR(center[dim], expected[dim], 1e-12);
    }

    // Test that the center of a triangle is the arithmetic mean of its vertices in a plane.
    TEST(PolygonCenter, TriangleCenterIsVertexAverage)
    {
      const std::vector<std::array<double, 3>> triangle = {
          {0.0, 0.0, 2.0}, {3.0, 0.0, 2.0}, {0.0, 3.0, 2.0}};

      expect_center_near(Core::Geo::compute_polygon_center(triangle), {1.0, 1.0, 2.0});
    }

    // Test that the center is weighted by area and is generally different from the arithmetic mean
    // of the polygon vertices in a plane.
    TEST(PolygonCenter, ComputesAreaWeightedCenter)
    {
      const std::vector<std::array<double, 3>> trapezoid = {
          {0.0, 0.0, 3.0}, {4.0, 0.0, 3.0}, {3.0, 2.0, 3.0}, {0.0, 2.0, 3.0}};

      expect_center_near(
          Core::Geo::compute_polygon_center(trapezoid), {37.0 / 21.0, 20.0 / 21.0, 3.0});
    }

    // Test that the calculation works for polygons in arbitrarily oriented planes.
    TEST(PolygonCenter, ComputesCenterInThreeDimensionalSpace)
    {
      const std::vector<std::array<double, 3>> rectangle = {
          {2.0, -1.0, 4.0}, {4.0, -1.0, 6.0}, {4.0, 1.0, 5.0}, {2.0, 1.0, 3.0}};

      expect_center_near(Core::Geo::compute_polygon_center(rectangle), {3.0, 0.0, 4.5});
    }

    // Test that the polygon orientation does not change its center calculation.
    TEST(PolygonCenter, IsIndependentOfVertexOrientation)
    {
      const std::vector<std::array<double, 3>> counterclockwise = {
          {0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {2.0, 1.0, 0.0}, {0.0, 1.0, 0.0}};

      auto clockwise = counterclockwise;
      std::ranges::reverse(clockwise);

      const auto counterclockwise_center = Core::Geo::compute_polygon_center(counterclockwise);
      const auto clockwise_center = Core::Geo::compute_polygon_center(clockwise);

      expect_center_near(counterclockwise_center, clockwise_center);
    }

    // Test that a polygon degenerating to a line or a point has zero area and therefore no center.
    TEST(PolygonCenter, DegeneratePolygonHasNoCenter)
    {
      const std::vector<std::array<double, 3>> line = {
          {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {2.0, 0.0, 0.0}};

      const std::vector<std::array<double, 3>> point = {
          {1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}};

      EXPECT_ANY_THROW(Core::Geo::compute_polygon_center(line));
      EXPECT_ANY_THROW(Core::Geo::compute_polygon_center(point));
    }

    // Test that the points spanning the polygon must be coplanar
    TEST(PolygonCenter, TiltedPolygonMustBeCoplanar)
    {
      const double inverse_sqrt_two = 1.0 / std::sqrt(2.0);
      const std::array<double, 3> unit_normal{0.0, -inverse_sqrt_two, inverse_sqrt_two};

      std::vector<std::array<double, 3>> tilted_rectangle{{0.0, 0.0, 0.0}, {2.0, 0.0, 0.0},
          {2.0, 3.0 * inverse_sqrt_two, 3.0 * inverse_sqrt_two},
          {0.0, 3.0 * inverse_sqrt_two, 3.0 * inverse_sqrt_two}};

      // The planar tilted rectangle is accepted.
      const auto center = Core::Geo::compute_polygon_center(tilted_rectangle);

      EXPECT_NEAR(center[0], 1.0, 1.0e-12);
      EXPECT_NEAR(center[1], 1.5 * inverse_sqrt_two, 1.0e-12);
      EXPECT_NEAR(center[2], 1.5 * inverse_sqrt_two, 1.0e-12);

      // Move one vertex out of the plane.
      for (std::size_t dim = 0; dim < 3; ++dim) tilted_rectangle[3][dim] += unit_normal[dim] * 2.0;

      EXPECT_ANY_THROW(Core::Geo::compute_polygon_center(tilted_rectangle));
    }

    // Test that a non-coplanar polygon does not have a well-defined center
    TEST(PolygonCenter, NonplanarPolygonHasNoCenter)
    {
      const std::vector<std::array<double, 3>> nonplanar_rectangle = {
          {2.0, -1.0, 4.0}, {4.0, -1.0, 6.0}, {4.0, 1.0, 5.0 + 2.0e-8}, {2.0, 1.0, 3.0}};

      EXPECT_ANY_THROW(Core::Geo::compute_polygon_center(nonplanar_rectangle));
    }
  }  // namespace PolygonCenter
}  // namespace

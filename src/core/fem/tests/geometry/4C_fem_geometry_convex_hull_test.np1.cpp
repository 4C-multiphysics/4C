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
#include <cmath>
#include <vector>

namespace
{
  using namespace FourC;

  // Convex hull tests
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

  // Polygon area tests
  class PolygonAreaTest : public ::testing::Test
  {
   protected:
    using Point = std::array<double, 3>;
    using Polygon = std::vector<Point>;

    static constexpr double tolerance = 1.0e-12;

    static Point create_point(const double x, const double y, const double z) { return {x, y, z}; }

    inline static const std::array<double, 3> unit_z_normal = {0.0, 0.0, 1.0};

    static Polygon create_polygon()
    {
      // Rectangle in the xy-plane with area 2 * 3 = 6.
      return {
          create_point(0.0, 0.0, 0.0),
          create_point(2.0, 0.0, 0.0),
          create_point(2.0, 3.0, 0.0),
          create_point(0.0, 3.0, 0.0),
      };
    }

    static Polygon create_concave_polygon()
    {
      // L-shaped polygon with area 4 - 1 = 3.
      return {
          create_point(0.0, 0.0, 0.0),
          create_point(2.0, 0.0, 0.0),
          create_point(2.0, 1.0, 0.0),
          create_point(1.0, 1.0, 0.0),
          create_point(1.0, 2.0, 0.0),
          create_point(0.0, 2.0, 0.0),
      };
    }
  };

  // Test that for fewer than three points the method returns a zero area
  TEST_F(PolygonAreaTest, ReturnsZeroForFewerThanThreePoints)
  {
    EXPECT_DOUBLE_EQ(Core::Geo::polygon_area(Polygon{}, unit_z_normal), 0.0);
    EXPECT_DOUBLE_EQ(
        Core::Geo::polygon_area(Polygon{create_point(0.0, 0.0, 0.0)}, unit_z_normal), 0.0);
    EXPECT_DOUBLE_EQ(
        Core::Geo::polygon_area(
            Polygon{create_point(0.0, 0.0, 0.0), create_point(1.0, 0.0, 0.0)}, unit_z_normal),
        0.0);
  }

  // Test that the method calculates the area of a rectangle correctly
  TEST_F(PolygonAreaTest, ComputesAreaOfPlanarRectangle)
  {
    const Polygon polygon = create_polygon();

    EXPECT_NEAR(Core::Geo::polygon_area(polygon, unit_z_normal), 6.0, tolerance);
  }

  // Test that the calculation is independent of the vertex orientation
  TEST_F(PolygonAreaTest, IsIndependentOfVertexOrientation)
  {
    Polygon polygon = create_polygon();
    std::reverse(polygon.begin(), polygon.end());

    EXPECT_NEAR(Core::Geo::polygon_area(polygon, unit_z_normal), 6.0, tolerance);
  }

  // Test that the calculation is independent of the normal orientation
  TEST_F(PolygonAreaTest, IsIndependentOfNormalOrientation)
  {
    const Polygon polygon = create_polygon();
    const std::array<double, 3> opposite_normal = {0.0, 0.0, -1.0};

    EXPECT_NEAR(Core::Geo::polygon_area(polygon, opposite_normal), 6.0, tolerance);
  }

  // Test that the calculation is independent to translation of the polygon
  TEST_F(PolygonAreaTest, IsIndependentOfTranslation)
  {
    Polygon polygon = create_polygon();

    const Point translation = create_point(7.0, -3.0, 4.0);
    for (Point& point : polygon)
      for (std::size_t dim = 0; dim < 3; ++dim) point[dim] += translation[dim];

    EXPECT_NEAR(Core::Geo::polygon_area(polygon, unit_z_normal), 6.0, tolerance);
  }

  // Test that the method correctly computes the area of concave polygons
  TEST_F(PolygonAreaTest, ComputesAreaOfConcavePolygon)
  {
    const Polygon polygon = create_concave_polygon();

    EXPECT_NEAR(Core::Geo::polygon_area(polygon, unit_z_normal), 3.0, tolerance);
  }

  // Test that the method correctly calculates the area for tilted polygons
  TEST_F(PolygonAreaTest, ComputesAreaInInclinedPlane)
  {
    // Rectangle spanned by u = (1, 0, 1) and  v = (0, 2, 0)
    // Its area is |u x v| = 2 * sqrt(2).
    const Polygon polygon{
        create_point(3.0, -2.0, 5.0),
        create_point(4.0, -2.0, 6.0),
        create_point(4.0, 0.0, 6.0),
        create_point(3.0, 0.0, 5.0),
    };

    const double inverse_sqrt_two = 1.0 / std::sqrt(2.0);
    const std::array<double, 3> unit_normal = {-inverse_sqrt_two, 0.0, inverse_sqrt_two};

    EXPECT_NEAR(Core::Geo::polygon_area(polygon, unit_normal), 2.0 * std::sqrt(2.0), tolerance);
  }

  // Test that the area is zero for collinear points
  TEST_F(PolygonAreaTest, ReturnsZeroForCollinearPoints)
  {
    const Polygon polygon{
        create_point(0.0, 0.0, 0.0),
        create_point(1.0, 1.0, 1.0),
        create_point(2.0, 2.0, 2.0),
        create_point(3.0, 3.0, 3.0),
    };

    const std::array<double, 3> unit_normal = {0.0, -1.0 / std::sqrt(2.0), 1.0 / std::sqrt(2.0)};

    EXPECT_NEAR(Core::Geo::polygon_area(polygon, unit_normal), 0.0, tolerance);
  }

  // Test that a vertex whose distance from the plane exceeds the admissible tolerance
  // violates the planar-polygon assumption.
  TEST_F(PolygonAreaTest, RejectsPointOutsidePlanarityTolerance)
  {
    const std::array<double, 3> unit_normal{0.0, 0.0, 1.0};
    const Polygon nonplanar_rectangle{
        {0.0, 0.0, 2.0}, {2.0, 0.0, 2.0}, {2.0, 3.0, 2.0 + 2.0 * 1e-08}, {0.0, 3.0, 2.0}};

    EXPECT_ANY_THROW(Core::Geo::polygon_area(nonplanar_rectangle, unit_normal));
  }
}  // namespace

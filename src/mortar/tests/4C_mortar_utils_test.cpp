// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_mortar_utils.hpp"

#include "4C_linalg_serialdensematrix.hpp"

#include <array>
#include <vector>

namespace
{
  using namespace FourC;

  // Points of a candidate clip polygon in the coordinates of the auxiliary plane
  using Points = std::vector<std::array<double, 2>>;

  // Sort the candidate points and return the resulting clip polygon
  Points sort_convex_hull(const Points& points, double clipping_tolerance)
  {
    Core::LinAlg::SerialDenseMatrix coordinates(2, points.size());
    for (std::size_t i = 0; i < points.size(); ++i)
    {
      coordinates(0, i) = points[i][0];
      coordinates(1, i) = points[i][1];
    }

    Points result;
    for (int i : Mortar::sort_convex_hull_points(coordinates, clipping_tolerance))
      result.push_back(points[i]);
    return result;
  }

  // expected outcome for a unit square, starting at the bottom left corner and running clockwise
  const Points unit_square = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}, {1.0, 0.0}};

  // The clip polygon starts at the point with the smallest x-coordinate and runs clockwise.
  TEST(SortConvexHullPoints, ClockwiseFromSmallestX)
  {
    const Points shuffled_square = {{1.0, 1.0}, {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};

    EXPECT_EQ(sort_convex_hull(shuffled_square, 0.01), unit_square);
  }

  // Points inside the hull, duplicate points and points exactly on an edge are removed, even
  // without a tolerance.
  TEST(SortConvexHullPoints, InnerDuplicateAndCollinearPointsAreRemoved)
  {
    const Points square_with_extra_points = {{0.0, 0.0}, {0.5, 0.5}, {0.0, 1.0}, {1.0, 1.0},
        {1.0, 1.0}, {1.0, 0.5}, {1.0, 0.0}, {0.5, 0.0}};

    EXPECT_EQ(sort_convex_hull(square_with_extra_points, 0.0), unit_square);
  }

  // Points slightly outside an edge are part of the convex hull, but removed by the tolerance.
  // Points clearly outside an edge are kept as corners.
  TEST(SortConvexHullPoints, OnlyCornersBeyondToleranceAreKept)
  {
    const Points square_with_points_near_edges = {
        {0.0, 0.0}, {0.0, 1.0}, {0.5, 1.001}, {1.0, 1.0}, {1.0, 0.0}, {0.5, -0.015}};

    EXPECT_EQ(sort_convex_hull(square_with_points_near_edges, 0.01),
        (Points{{0.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}, {1.0, 0.0}, {0.5, -0.015}}));
  }

  // A corner is only judged against its neighbors on the convex hull. The point (0.99, 0.95) lies
  // inside the square, almost on the line from (0, 1) to the corner (1, 1). Judged against it,
  // this corner would look almost straight.
  TEST(SortConvexHullPoints, CornerIsNotJudgedAgainstInnerPoint)
  {
    const Points square_with_point_near_corner = {
        {0.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}, {1.0, 0.0}, {0.99, 0.95}};

    EXPECT_EQ(sort_convex_hull(square_with_point_near_corner, 0.1), unit_square);
  }

  // If several points are below the tolerance, the straightest one is removed first and the others
  // are judged against their new neighbors. Here, the distances from the line through the
  // neighbors are 0.24 at (3, 0) and 1.0 at (4, 0). After removing (3, 0), the distance of (4, 0)
  // is 2.2, so this corner is kept.
  TEST(SortConvexHullPoints, StraightestPointIsRemovedFirst)
  {
    const Points quadrilateral = {{0.0, 1.0}, {3.0, 2.0}, {4.0, 0.0}, {3.0, 0.0}};

    EXPECT_EQ(sort_convex_hull(quadrilateral, 1.5), (Points{{0.0, 1.0}, {3.0, 2.0}, {4.0, 0.0}}));
  }

  // The tolerance is a length. The corners of a small square are 0.007 away from the line through
  // their neighbors, which is above the tolerance, although twice the triangle areas are only 1e-4.
  TEST(SortConvexHullPoints, SmallPolygonKeepsItsCorners)
  {
    const Points small_square = {{0.0, 0.0}, {0.0, 0.01}, {0.01, 0.01}, {0.01, 0.0}};

    EXPECT_EQ(sort_convex_hull(small_square, 0.001), small_square);
  }

  // If less than three points remain, there is no clip polygon.
  TEST(SortConvexHullPoints, DegeneratePolygonHasLessThanThreePoints)
  {
    const Points almost_straight_line = {{0.0, 0.0}, {1.0, 0.001}, {2.0, 0.0}};

    EXPECT_LT(sort_convex_hull(almost_straight_line, 0.01).size(), 3u);
  }
}  // namespace

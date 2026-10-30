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

  using Points = std::vector<std::array<double, 2>>;

  // Compute the convex hull and return its points
  Points convex_hull(const Points& points)
  {
    Points hull;
    for (int i : Core::Geo::convex_hull_2d(points)) hull.push_back(points[i]);
    return hull;
  }

  // The hull consists of the corners, consecutive along the hull, starting at an arbitrary corner
  // and running in an arbitrary direction.
  TEST(ConvexHull2D, CornersAreConsecutive)
  {
    const Points shuffled_square_with_inner_point = {
        {1.0, 1.0}, {0.5, 0.5}, {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};
    const Points square = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}, {1.0, 0.0}};

    Points hull = convex_hull(shuffled_square_with_inner_point);
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
}  // namespace

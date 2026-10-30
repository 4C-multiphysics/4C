// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_geometry_convex_hull.hpp"

#include <boost/geometry/algorithms/area.hpp>
#include <boost/geometry/algorithms/convex_hull.hpp>
#include <boost/geometry/geometries/multi_point.hpp>
#include <boost/geometry/geometries/register/point.hpp>
#include <boost/geometry/geometries/ring.hpp>

FOUR_C_NAMESPACE_OPEN

namespace
{
  // point, which remembers its position in the input
  struct IndexedPoint
  {
    double x;
    double y;
    int index;
  };
}  // namespace

FOUR_C_NAMESPACE_CLOSE

// Register IndexedPoint as a 2D point type
BOOST_GEOMETRY_REGISTER_POINT_2D(FourC::IndexedPoint, double, boost::geometry::cs::cartesian, x, y)

FOUR_C_NAMESPACE_OPEN

std::vector<int> Core::Geo::convex_hull_2d(std::span<const std::array<double, 2>> points)
{
  boost::geometry::model::multi_point<IndexedPoint> indexed_points;
  for (int i = 0; i < static_cast<int>(points.size()); ++i)
    indexed_points.push_back({.x = points[i][0], .y = points[i][1], .index = i});

  // template arguments: clockwise (boost requires an orientation, the order is not guaranteed
  // to callers) and open, i.e. the first point is not repeated at the end
  boost::geometry::model::ring<IndexedPoint, true, false> hull;
  boost::geometry::convex_hull(indexed_points, hull);

  // boost pads a degenerate hull (a segment or a point) to the minimum ring size by repeating a
  // point; such a ring has exactly zero area and is reported as an empty hull
  if (boost::geometry::area(hull) == 0.0) return {};

  std::vector<int> hull_indices;
  for (const auto& point : hull) hull_indices.push_back(point.index);
  return hull_indices;
}

FOUR_C_NAMESPACE_CLOSE

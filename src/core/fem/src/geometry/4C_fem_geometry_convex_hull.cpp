// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_geometry_convex_hull.hpp"

#include "4C_fem_geometry_intersection_math.hpp"
#include "4C_linalg_fixedsizematrix.hpp"

#include <boost/geometry/algorithms/area.hpp>
#include <boost/geometry/algorithms/convex_hull.hpp>
#include <boost/geometry/geometries/multi_point.hpp>
#include <boost/geometry/geometries/register/point.hpp>
#include <boost/geometry/geometries/ring.hpp>
#include <boost/geometry/strategies/strategies.hpp>

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
  indexed_points.reserve(points.size());
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


double Core::Geo::polygon_area(
    std::span<const std::array<double, 3>> points, const std::array<double, 3>& unit_normal)
{
  if (points.size() < 3) return 0.0;

  const LinAlg::Matrix<3, 1> normal(unit_normal.data(), true);

  FOUR_C_ASSERT(std::abs(normal.norm2() - 1.0) < TOL12,
      "Expected a unit normal, but its norm is {}.", normal.norm2());

  const Core::LinAlg::Matrix<3, 1> plane_point(points.front().data(), true);
  double twice_oriented_area = 0.0;

  for (std::size_t k = 0; k < points.size(); ++k)
  {
    Core::LinAlg::Matrix<3, 1> current(Core::LinAlg::Initialization::uninitialized);
    current.update(1.0, Core::LinAlg::Matrix<3, 1>(points[k].data(), true), -1.0, plane_point, 0.0);

    Core::LinAlg::Matrix<3, 1> next(Core::LinAlg::Initialization::uninitialized);
    next.update(1.0, Core::LinAlg::Matrix<3, 1>(points[(k + 1) % points.size()].data(), true), -1.0,
        plane_point, 0.0);

    FOUR_C_ASSERT(std::abs(current.dot(normal)) <= TOL8,
        "Polygon point has distance {} from the polygon plane, exceeding tolerance {}.",
        std::abs(current.dot(normal)), TOL8);

    Core::LinAlg::Matrix<3, 1> cross_product(Core::LinAlg::Initialization::uninitialized);
    cross_product.cross_product(current, next);
    twice_oriented_area += cross_product.dot(normal);
  }

  return 0.5 * std::abs(twice_oriented_area);
}

FOUR_C_NAMESPACE_CLOSE

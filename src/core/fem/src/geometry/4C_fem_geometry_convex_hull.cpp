// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_geometry_convex_hull.hpp"

#include "4C_fem_geometry_intersection_math.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_utils_exceptions.hpp"

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


std::array<double, 3> Core::Geo::compute_polygon_center(
    std::span<const std::array<double, 3>> points)
{
  FOUR_C_ASSERT_ALWAYS(points.size() >= 3, "A polygon must contain at least three vertices.");

#ifdef FOUR_C_ENABLE_ASSERTIONS
  const LinAlg::Matrix<3, 1> plane_point(points[0].data(), true);

  LinAlg::Matrix<3, 1> edge(Core::LinAlg::Initialization::uninitialized);
  edge.update(1.0, LinAlg::Matrix<3, 1>(points[1].data(), true), -1.0, plane_point, 0.0);

  LinAlg::Matrix<3, 1> next_edge(Core::LinAlg::Initialization::uninitialized);
  next_edge.update(1.0, LinAlg::Matrix<3, 1>(points[2].data(), true), -1.0, plane_point, 0.0);

  LinAlg::Matrix<3, 1> plane_normal(Core::LinAlg::Initialization::uninitialized);
  plane_normal.cross_product(edge, next_edge);
  plane_normal.scale(1.0 / plane_normal.norm2());

  for (const auto& point : points)
  {
    LinAlg::Matrix<3, 1> offset(Core::LinAlg::Initialization::uninitialized);
    offset.update(1.0, LinAlg::Matrix<3, 1>(point.data(), true), -1.0, plane_point, 0.0);

    FOUR_C_ASSERT_ALWAYS(std::abs(offset.dot(plane_normal)) < TOL10,
        "Polygon vertices must be coplanar. The given offset measure is {} with the given "
        "tolerance {}.",
        std::abs(offset.dot(plane_normal)), TOL10);
  }
#endif

  LinAlg::Matrix<3, 1> reference_point(Core::LinAlg::Initialization::zero);
  for (const auto& point : points) reference_point += LinAlg::Matrix<3, 1>(point.data(), true);

  reference_point.scale(1.0 / static_cast<double>(points.size()));

  LinAlg::Matrix<3, 1> weighted_center(Core::LinAlg::Initialization::zero);
  double polygon_area = 0.0;

  for (std::size_t point = 0; point < points.size(); ++point)
  {
    const std::size_t next_point = (point + 1) % points.size();

    LinAlg::Matrix<3, 1> first_edge(Core::LinAlg::Initialization::uninitialized);
    first_edge.update(
        1.0, LinAlg::Matrix<3, 1>(points[point].data(), true), -1.0, reference_point, 0.0);

    LinAlg::Matrix<3, 1> second_edge(Core::LinAlg::Initialization::uninitialized);
    second_edge.update(
        1.0, LinAlg::Matrix<3, 1>(points[next_point].data(), true), -1.0, reference_point, 0.0);

    LinAlg::Matrix<3, 1> cross_product(Core::LinAlg::Initialization::uninitialized);
    cross_product.cross_product(first_edge, second_edge);

    const double triangle_area = 0.5 * cross_product.norm2();
    polygon_area += triangle_area;

    LinAlg::Matrix<3, 1> triangle_center = reference_point;
    triangle_center += LinAlg::Matrix<3, 1>(points[point].data(), true);
    triangle_center += LinAlg::Matrix<3, 1>(points[next_point].data(), true);
    triangle_center.scale(triangle_area / 3.0);

    weighted_center += triangle_center;
  }

  FOUR_C_ASSERT_ALWAYS(polygon_area > 0.0 && std::isfinite(polygon_area),
      "The polygon area {} must be greater than zero.", polygon_area);

  weighted_center.scale(1.0 / polygon_area);

  return std::array{weighted_center(0), weighted_center(1), weighted_center(2)};
}

FOUR_C_NAMESPACE_CLOSE

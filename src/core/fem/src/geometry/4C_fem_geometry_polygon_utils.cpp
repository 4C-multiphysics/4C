// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_geometry_polygon_utils.hpp"

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

/*----------------------------------------------------------------------* |
 *----------------------------------------------------------------------*/
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

/*----------------------------------------------------------------------* |
 *----------------------------------------------------------------------*/
std::optional<std::array<double, 3>> Core::Geo::compute_polygon_center(
    const std::vector<std::array<double, 3>>& polygon, const double area_tolerance)
{
  FOUR_C_ASSERT_ALWAYS(polygon.size() >= 3, "A polygon must contain at least three vertices.");

  std::array<double, 3> reference_point{0.0, 0.0, 0.0};

  for (const auto& vertex : polygon)
  {
    for (std::size_t dim = 0; dim < 3; ++dim) reference_point[dim] += vertex[dim];
  }

  for (double& coordinate : reference_point) coordinate /= static_cast<double>(polygon.size());

  std::array<double, 3> weighted_center{0.0, 0.0, 0.0};
  double polygon_area = 0.0;

  for (std::size_t vertex = 0; vertex < polygon.size(); ++vertex)
  {
    const std::size_t next_vertex = (vertex + 1) % polygon.size();

    const std::array<double, 3> first_edge{
        polygon[vertex][0] - reference_point[0],
        polygon[vertex][1] - reference_point[1],
        polygon[vertex][2] - reference_point[2],
    };

    const std::array<double, 3> second_edge{
        polygon[next_vertex][0] - reference_point[0],
        polygon[next_vertex][1] - reference_point[1],
        polygon[next_vertex][2] - reference_point[2],
    };

    const std::array<double, 3> cross_product{
        first_edge[1] * second_edge[2] - first_edge[2] * second_edge[1],
        first_edge[2] * second_edge[0] - first_edge[0] * second_edge[2],
        first_edge[0] * second_edge[1] - first_edge[1] * second_edge[0],
    };

    const double triangle_area =
        0.5 * std::sqrt(cross_product[0] * cross_product[0] + cross_product[1] * cross_product[1] +
                        cross_product[2] * cross_product[2]);

    polygon_area += triangle_area;

    for (std::size_t dim = 0; dim < 3; ++dim)
    {
      weighted_center[dim] +=
          triangle_area *
          (reference_point[dim] + polygon[vertex][dim] + polygon[next_vertex][dim]) / 3.0;
    }
  }

  if (polygon_area <= area_tolerance) return std::nullopt;

  for (double& coordinate : weighted_center) coordinate /= polygon_area;

  return weighted_center;
}

FOUR_C_NAMESPACE_CLOSE

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FEM_GEOMETRY_DELAUNAY_TRIANGULATION_HPP
#define FOUR_C_FEM_GEOMETRY_DELAUNAY_TRIANGULATION_HPP

#include "4C_config.hpp"

#include "4C_utils_exceptions.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <span>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::Geometry
{
  template <std::size_t dim>
  using Point = std::array<double, dim>;
  template <std::size_t dim>
  using Simplex = std::array<int, dim + 1>;

  /**
   * @brief Compute the squared Euclidean norm of a point.
   */
  template <std::size_t dim>
  double squared_norm(const Point<dim>& point)
  {
    return std::inner_product(point.begin(), point.end(), point.begin(), 0.0);
  }

  /**
   * @brief Center and uniformly scale a point set.
   *
   * The centroid is translated to the origin and all coordinates are divided
   * by the maximum distance from the centroid. Consequently, the normalized
   * point set has approximately unit scale.
   *
   * Translation and uniform scaling preserve Delaunay connectivity in exact
   * arithmetic. The original point ordering is retained.
   *
   * @throws Core::Exception if all points coincide.
   */
  template <std::size_t dim>
  std::vector<Point<dim>> normalize_points(std::span<const Point<dim>> points)
  {
    Point<dim> center{};

    for (const auto& point : points)
      for (std::size_t d = 0; d < dim; ++d)
        center[d] += point[d] / static_cast<double>(points.size());

    std::vector<Point<dim>> normalized_points(points.size());
    double scale = 0.0;

    for (std::size_t i = 0; i < points.size(); ++i)
    {
      for (std::size_t d = 0; d < dim; ++d) normalized_points[i][d] = points[i][d] - center[d];
      scale = std::max(scale, std::sqrt(squared_norm(normalized_points[i])));
    }

    if (!(scale > 0.0)) FOUR_C_THROW("Cannot triangulate a point set with zero diameter.");

    // Uniform scaling preserves the Delaunay triangulation.
    for (auto& point : normalized_points)
      for (double& coordinate : point) coordinate /= scale;

    return normalized_points;
  }

  /**
   * @brief Compute twice the signed area of a triangle.
   *
   * A positive value denotes counterclockwise orientation, a negative value
   * denotes clockwise orientation, and zero denotes a degenerate triangle.
   */
  inline double signed_measure(std::span<const Point<2>> points, const Simplex<2>& triangle)
  {
    const auto& a = points[triangle[0]];
    const auto& b = points[triangle[1]];
    const auto& c = points[triangle[2]];

    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
  }

  /**
   * @brief Compute six times the signed volume of a tetrahedron.
   *
   * The sign describes the orientation of the tetrahedron. A zero value
   * denotes a degenerate tetrahedron.
   */
  inline double signed_measure(std::span<const Point<3>> points, const Simplex<3>& tetrahedron)
  {
    const auto& a = points[tetrahedron[0]];
    const auto& b = points[tetrahedron[1]];
    const auto& c = points[tetrahedron[2]];
    const auto& d = points[tetrahedron[3]];

    const Point<3> ab{b[0] - a[0], b[1] - a[1], b[2] - a[2]};
    const Point<3> ac{c[0] - a[0], c[1] - a[1], c[2] - a[2]};
    const Point<3> ad{d[0] - a[0], d[1] - a[1], d[2] - a[2]};

    return ab[0] * (ac[1] * ad[2] - ac[2] * ad[1]) - ab[1] * (ac[0] * ad[2] - ac[2] * ad[0]) +
           ab[2] * (ac[0] * ad[1] - ac[1] * ad[0]);
  }

  /**
   * @brief Convert a simplex to a deterministic, positively oriented form.
   *
   * The point indices are first sorted to obtain deterministic connectivity.
   * If this ordering has negative orientation, the final two indices are
   * exchanged.
   *
   * Since the coordinates have previously been normalized, a fixed tolerance
   * can be used to identify numerically degenerate simplices.
   *
   * @return `true` if the simplex is full-dimensional and has been
   *         canonicalized, or `false` if it is degenerate.
   */
  template <std::size_t dim>
  bool canonicalize_simplex(Simplex<dim>& simplex, const std::span<const Point<dim>> points)
  {
    std::ranges::sort(simplex);
    const double measure = signed_measure(points, simplex);

    // Points have already been normalized to approximately unit scale.
    constexpr double tolerance = 100.0 * std::numeric_limits<double>::epsilon();
    if (std::abs(measure) <= tolerance) return false;

    // Sorting may reverse orientation. One transposition restores it.
    if (measure < 0.0) std::swap(simplex[dim - 1], simplex[dim]);

    return true;
  }

  /**
   * Compute the Delaunay triangulation of a full-dimensional 2D point set.
   *
   * Every input point must be unique and must occur in the resulting
   * triangulation. Returned triangles have positive orientation.
   */
  [[nodiscard]] std::vector<Simplex<2>> delaunay_triangulation(std::span<const Point<2>> points);

  /**
   * Compute the Delaunay tetrahedralization of a full-dimensional 3D point set.
   *
   * Every input point must be unique and must occur in the resulting
   * tetrahedralization. Returned tetrahedra have positive orientation.
   */
  [[nodiscard]] std::vector<Simplex<3>> delaunay_triangulation(std::span<const Point<3>> points);

}  // namespace Core::Geometry

FOUR_C_NAMESPACE_CLOSE

#endif

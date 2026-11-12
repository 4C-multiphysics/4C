// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FEM_GEOMETRY_CONVEX_HULL_HPP
#define FOUR_C_FEM_GEOMETRY_CONVEX_HULL_HPP

#include "4C_config.hpp"

#include <array>
#include <span>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::Geo
{
  /**
   * @brief Compute the convex hull of a set of points in 2D.
   *
   * Points inside the hull or on one of its edges are not part of the hull.
   *
   * @param points coordinates of the points
   * @return indices of the points on the hull, consecutive along the hull (empty if the hull
   * degenerates to a line or a point)
   */
  std::vector<int> convex_hull_2d(std::span<const std::array<double, 2>> points);
}  // namespace Core::Geo

FOUR_C_NAMESPACE_CLOSE

#endif

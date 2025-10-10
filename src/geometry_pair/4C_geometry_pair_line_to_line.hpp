// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_GEOMETRY_PAIR_LINE_TO_LINE_HPP
#define FOUR_C_GEOMETRY_PAIR_LINE_TO_LINE_HPP


#include "4C_config.hpp"

#include "4C_geometry_pair_element.hpp"


FOUR_C_NAMESPACE_OPEN

// Forward declarations.
namespace GeometryPair
{
  enum class ProjectionResult;
}  // namespace GeometryPair


namespace GeometryPair
{
  /**
   * \brief Closest point projection between two curves.
   */
  template <typename ScalarType, typename LineA, typename LineB>
  ProjectionResult line_to_line_closest_point_projection(
      const ElementData<LineA, ScalarType>& element_data_line_a,
      const ElementData<LineB, ScalarType>& element_data_line_b, ScalarType& eta_a,
      ScalarType& eta_b, const bool min_one_iteration = false);
}  // namespace GeometryPair

FOUR_C_NAMESPACE_CLOSE

#endif

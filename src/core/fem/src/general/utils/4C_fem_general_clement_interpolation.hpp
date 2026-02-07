// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FEM_GENERAL_CLEMENT_INTERPOLATION_HPP
#define FOUR_C_FEM_GENERAL_CLEMENT_INTERPOLATION_HPP

#include "4C_config.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_linalg_multi_vector.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  /**
   * @brief Compute nodal values using a Clement interpolant.
   *
   * This function performs a nodal Clement interpolation of element based values
   * onto mesh nodes. For each node, the interpolated value is computed as a
   * volume weighted average of the values of all elements adjacent to that node.
   *
   * More precisely, for each node and each vector component, the function:
   * - Collects all adjacent elements (the element patch of the node),
   * - Computes the geometric volume of each element,
   * - Weights the element value by its volume,
   * - Normalizes by the total volume of the patch.
   *
   * The operation is repeated independently for each column of the input
   * MultiVector.
   *
   * @param[in] dis Finite element discretization containing mesh topology, geometry, and parallel
   * distribution information.
   * @param[in] values MultiVector holding element-wise values. The map of this vector is assumed to
   * be compatible with the element global IDs used by the discretization.  Each column is
   * interpolated independently.
   *
   * @return A shared pointer to a MultiVector containing the interpolated nodal values. The
   * returned vector is defined on the node row map of the discretization and has the same number
   * of columns as the input values.
   *
   * @note The input element value vector needs to have been build on the column element map.
   *
   * P. Clement: Approximation by finite element functions using local regularization,
   * RAIRO. Analyse numerique, 2:77-84, 1975
   */
  std::shared_ptr<Core::LinAlg::MultiVector<double>> compute_nodal_clement_interpolation(
      const Core::FE::Discretization& dis, const Core::LinAlg::MultiVector<double>& values);
}  // namespace Core::FE

FOUR_C_NAMESPACE_CLOSE

#endif

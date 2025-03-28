// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FEM_GENERAL_UTILS_INTERPOLATION_HPP
#define FOUR_C_FEM_GENERAL_UTILS_INTERPOLATION_HPP

#include "4C_config.hpp"

#include "4C_fem_general_cell_type.hpp"
#include "4C_fem_general_cell_type_traits.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_linalg_fixedsizematrix.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  /*!
   * @brief calculate and return the value of the quantity at position xi based on the
   * quantity node vector
   *
   * @tparam distype        discretization type of element
   * @param xi              position to project to in local coordinates
   * @param nodal_quantity  nodal vector of the quantity to be projected
   * @return quantities projected to position xi
   */
  template <CellType celltype>
  std::vector<double> interpolate_to_xi(const Core::LinAlg::Matrix<Core::FE::dim<celltype>, 1>& xi,
      const std::vector<double>& nodal_quantity)
  {
    constexpr int num_nodes = Core::FE::num_nodes<celltype>;

    Core::LinAlg::Matrix<num_nodes, 1> shapefunct(Core::LinAlg::Initialization::zero);
    Core::FE::shape_function<celltype>(xi, shapefunct);

    const int num_dof_per_node = static_cast<int>(nodal_quantity.size()) / num_nodes;
    std::vector<double> projected_quantities(num_dof_per_node, 0.0);

    for (int dof = 0; dof < num_dof_per_node; ++dof)
    {
      for (int i = 0; i < num_nodes; ++i)
      {
        projected_quantities[dof] += nodal_quantity[i * num_dof_per_node + dof] * shapefunct(i);
      }
    }

    return projected_quantities;
  }
}  // namespace Core::FE

FOUR_C_NAMESPACE_CLOSE

#endif
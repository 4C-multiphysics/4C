// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_deal_ii_dofs.hpp"

#include "4C_deal_ii_element_conversion.hpp"
#include "4C_fem_discretization.hpp"


FOUR_C_NAMESPACE_OPEN

namespace DealiiWrappers
{

  template <int dim, int spacedim>
  void assign_fes_and_dofs(
      const Context<dim, spacedim>& context, dealii::DoFHandler<dim, spacedim>& dof_handler)
  {
    // Loop all cells again and set the correct fe index within the collection
    for (const auto& cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned()) continue;
      // this is necessary since we can only set active_fe_indices for
      // cell_iterators from a DoFHandler. While it refers to the same cell as in the triangulation
      // (and actually decays to it), it has capabilities that the usual tria_cell_iterator does
      // not. since we don't build a DofHandler for the triangulation we cannot set it there,
      // instead we store the information manually within the context as it is needed to understand
      // which finite element is used on which cell. We thus can use that information here
      cell->set_active_fe_index(context.get_active_fe_index(cell));
    }
    // Now distribute the dofs, which will use the previously set index
    dof_handler.distribute_dofs(context.get_finite_elements());
    // Depending on the element types we need to construct an appropriate mapping collection
    FOUR_C_ASSERT(context.get_finite_elements().size() == 1,
        "The current implementation only supports a single finite element type per "
        "discretization.");
  }


  // explicit instantiation
  template void assign_fes_and_dofs(const Context<3, 3>& context, dealii::DoFHandler<3, 3>&);
}  // namespace DealiiWrappers

FOUR_C_NAMESPACE_CLOSE

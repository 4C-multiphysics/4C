// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_DEAL_II_LINALG_SPARSITY_HPP
#define FOUR_C_DEAL_II_LINALG_SPARSITY_HPP

#include "4C_config.hpp"

#include "4C_deal_ii_context.hpp"

FOUR_C_NAMESPACE_OPEN

namespace DealiiWrappers
{
  /**
   * Make a sparsity pattern for the given context. It is a sparsity pattern for the coupling
   * from the domain discretization (4C) to the range discretization (deal.II).
   * It builds the sparsity pattern for coupling all dofs on the equivalent cells given in the
   * context object. This means that the underlying triangulation in the context must be the
   * same as in the provided dof_handler.
   * @tparam four_c_is_range template bool to indicate which of the two discretizations is the range
   * space (i.e. which one is used for the rows of the sparsity pattern).
   */
  template <bool four_c_is_range, int dim, int spacedim>
  void make_context_sparsity_pattern(const Context<dim, spacedim>& context,
      const dealii::DoFHandler<dim, spacedim>& dof_handler,
      dealii::SparsityPatternBase& sparsity_pattern);
}  // namespace DealiiWrappers


// ===============================================================================================
// Implementation of the make_context_sparsity_pattern function

namespace DealiiWrappers
{
  template <bool four_c_is_range, int dim, int spacedim>
  void make_context_sparsity_pattern(const Context<dim, spacedim>& context,
      const dealii::DoFHandler<dim, spacedim>& dof_handler,
      dealii::SparsityPatternBase& sparsity_pattern)
  {
    // Assert that the sparsity pattern is already sized correctly
    FOUR_C_ASSERT(sparsity_pattern.n_rows() == four_c_is_range
                      ? static_cast<unsigned int>(context.get_discretization().num_global_nodes())
                      : dof_handler.n_dofs(),
        "The sparsity pattern must be sized to the number of dofs in the range discretization.");
    FOUR_C_ASSERT(sparsity_pattern.n_cols() == four_c_is_range
                      ? dof_handler.n_dofs()
                      : static_cast<unsigned int>(context.get_discretization().num_global_nodes()),
        "The sparsity pattern must be sized to the number of dofs in the domain discretization.");

    FOUR_C_ASSERT(&dof_handler.get_triangulation() == &context.get_triangulation(),
        "The triangulation of the dof_handler must be the same as the one in the context.");

    std::vector<dealii::types::global_dof_index> dofs_four_c;
    std::vector<dealii::types::global_dof_index> dofs_dealii;
    dofs_four_c.reserve(context.get_finite_elements().max_dofs_per_cell());
    dofs_dealii.reserve(dof_handler.get_fe_collection().max_dofs_per_cell());

    for (const auto& cell : dof_handler.active_cell_iterators())
    {
      // skip ghost cells
      if (!cell->is_locally_owned()) continue;


      const unsigned int n_dofs_on_cell_dealii = cell->get_fe().dofs_per_cell;
      dofs_dealii.resize(n_dofs_on_cell_dealii);
      cell->get_dof_indices(dofs_dealii);

      const unsigned int n_dofs_on_cell_four_c = context.fe_on_cell(cell).dofs_per_cell;
      dofs_four_c.resize(n_dofs_on_cell_four_c);
      context.get_dof_indices_four_c_ordering(cell, dofs_four_c);

      auto& domain_indices = four_c_is_range ? dofs_dealii : dofs_four_c;
      auto& range_indices = four_c_is_range ? dofs_four_c : dofs_dealii;
      std::sort(domain_indices.begin(), domain_indices.end());

      for (auto dof : range_indices)
      {
        // Add the coupling from the range dof to the domain dofs
        sparsity_pattern.add_row_entries(dof, domain_indices, true);
      }
    }
  }
}  // namespace DealiiWrappers
FOUR_C_NAMESPACE_CLOSE

#endif  // FOUR_C_DEAL_II_LINALG_SPARSITY_HPP

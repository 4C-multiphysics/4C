// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_DEAL_II_TRIANGULATION_HPP
#define FOUR_C_DEAL_II_TRIANGULATION_HPP

#include "4C_config.hpp"

#include <deal.II/grid/tria.h>
#include <deal.II/hp/fe_collection.h>


FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}

namespace DealiiWrappers
{
  template <int dim, int spacedim>
  class Context;

  /**
   * Create an equivalent dealii::Triangulation @p tria from a given Core::FE::Discretization
   * @p discretization. The given @p tria can be either
   *
   *  - (serial) dealii::Triangulation
   *  - dealii::parallel::fullydistributed::Triangulation (abbreviated p:f:T).
   *
   * The nodes and elements of the @p discretization are translated into vertices and cells in the
   * target Triangulation. Note that, the serial Triangulation stores the full (coarse) mesh which
   * is extracted from @p discretization. In contrast, the p:f:T variant only stores the
   * relevant parts of a coarse mesh on every process. For large meshes (> 10,000 cells) this often
   * becomes necessary to save memory. The partitioning of the cells of a p:f:T will be identical to
   * the partitioning of the @p discretization.
   *
   * @attention
   * If the @p discretization is partitioned but the @p tria that's passed is serial, the resulting
   * Triangulation will contain cells corresponding to all cells in the discretization (not only the
   * local ones) on all processes.
   * The @r context object that is returned will however only contain the FE information for the
   * elements that are locally owned by the discretization on the calling process.
   * Objects that are initialized on all local cells of the @p triangulation will have invalid
   * numbers on the cells that are not locally owned. Specifically:
   * - The cell_index_to_element_lid map will have -1 for all cells that are not locally owned.
   * - The active_fe_indices will be set to dealii::numbers::invalid_unsigned_int for all cells that
   *   are not locally owned.
   *
   * @return A Context that maps Discretization elements to Triangulation cells and vice versa. The
   * details of this type are not relevant for a user but it may be passed on to other functions in
   * this namespace.
   *
   * @note Limitations:
   *  - no copying of block ids or nodesets.
   *  - for tet meshes, deal.II does not support the GridTools::consistently_order_cells
   * function, so we just use the order of the vertices as they are given in the @p discretization.
   * This is NOT guaranteed to work.
   */
  template <int dim, int spacedim>
  Context<dim, spacedim> create_triangulation(
      dealii::Triangulation<dim, spacedim>& tria, const Core::FE::Discretization& discretization);



  /**
   * Create the dealii::hp::FECollection with all FE types that appear in the given @p
   * discretization. This also included FEs that appear only on other MPI ranks. In addition,
   * this function returns the names of these elements in the same order as in the FECollection.
   *
   * @note The ordering of the FEs in the collection is the same on all ranks.
   */
  template <int dim, int spacedim>
  std::pair<dealii::hp::FECollection<dim, spacedim>, std::vector<std::string>>
  create_required_finite_element_collection(const Core::FE::Discretization& discretization);


}  // namespace DealiiWrappers

FOUR_C_NAMESPACE_CLOSE

#endif

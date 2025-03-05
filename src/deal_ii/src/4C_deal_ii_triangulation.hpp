// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_DEAL_II_TRIANGULATION_HPP
#define FOUR_C_DEAL_II_TRIANGULATION_HPP

#include "4C_config.hpp"

#include "4C_deal_ii_context.hpp"

#include <deal.II/grid/tria.h>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}

namespace DealiiWrappers
{
  /**
   * Create an equivalent dealii::Triangulation @p tria from a given Core::FE::Discretization
   * @p discretization. The given @p tria can be either
   *
   *  - (serial) dealii::Triangulation
   *  - dealii::parallel::distributed::Triangulation (abbreviated p:d:T)
   *  - dealii::parallel::fullydistributed::Triangulation (abbreviated p:f:T).
   *
   * The nodes and elements of the @p discretization are translated into vertices and cells in the
   * target Triangulation. Note that, the serial and p:d:T Triangulation store the  full (coarse)
   * mesh which is extracted from @p discretization. In contrast, the p:f:T variant only stores the
   * relevant parts of a coarse mesh on every process. For large meshes (> 10,000 cells) this often
   * becomes necessary to save memory. The partitioning of the cells of a p:f:T will be identical to
   * the partitioning of the @p discretization.
   *
   * @return A Context that maps Discretization elements to Triangulation cells and vice versa. The
   * details of this type are not relevant for a user but it may be passed on to other functions in
   * this namespace.
   *
   * @note Limitations:
   *  - no copying of block ids or nodesets.
   */
  template <int dim, int spacedim>
  Context<dim, spacedim> create_triangulation(
      dealii::Triangulation<dim, spacedim>& tria, const Core::FE::Discretization& discretization);

}  // namespace DealiiWrappers

FOUR_C_NAMESPACE_CLOSE

#endif

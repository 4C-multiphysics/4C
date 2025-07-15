// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_DEAL_II_DOFS_HPP
#define FOUR_C_DEAL_II_DOFS_HPP

#include "4C_config.hpp"

#include "4C_deal_ii_context.hpp"

#include <deal.II/dofs/dof_handler.h>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}

namespace DealiiWrappers
{
  /**
   * @brief Assigns a finite element and degrees of freedom to the given @p dof_handler.
   *
   * The information about the finite element and degrees of freedom is taken from the
   * @p context object, which was created during create_triangulation(). After calling this
   * function, the @p dof_handler will be set up to mimic the finite element space inside
   * the @p discretization object.
   */
  template <int dim, int spacedim>
  void assign_fes_and_dofs(
      const Context<dim, spacedim>& context, dealii::DoFHandler<dim, spacedim>& dof_handler);

}  // namespace DealiiWrappers

FOUR_C_NAMESPACE_CLOSE

#endif

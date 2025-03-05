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
  template <int dim, int spacedim>
  std::unique_ptr<dealii::hp::MappingCollection<dim, spacedim>> assign_fes_and_dofs(
      dealii::DoFHandler<dim, spacedim>& dof_handler,
      const Core::FE::Discretization& discretization, Context<dim, spacedim>& context);

}  // namespace DealiiWrappers

FOUR_C_NAMESPACE_CLOSE

#endif

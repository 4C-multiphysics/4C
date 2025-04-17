// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_deal_ii_dofs.hpp"

#include "4C_deal_ii_context_implementation.hpp"
#include "4C_deal_ii_element_conversion.hpp"
#include "4C_fem_discretization.hpp"

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/hp/mapping_collection.h>

FOUR_C_NAMESPACE_OPEN

namespace DealiiWrappers
{

  template <int dim, int spacedim>
  std::unique_ptr<dealii::hp::MappingCollection<dim, spacedim>> assign_fes_and_dofs(
      dealii::DoFHandler<dim, spacedim>& dof_handler,
      const Core::FE::Discretization& discretization, Context<dim, spacedim>& context)
  {
    const auto& fe_collection = context.pimpl_->finite_elements;
    const auto& fe_names = context.pimpl_->finite_element_names;

    // Loop all cells again and set the correct fe index within the collection
    for (const auto& cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned()) continue;

      const auto four_c_element_lid = context.pimpl_->cell_index_to_element_lid[cell->index()];
      const auto* four_c_element = discretization.l_row_element(four_c_element_lid);

      auto iter = std::find(fe_names.begin(), fe_names.end(),
          ElementConversion::dealii_fe_name(four_c_element->shape()));
      FOUR_C_ASSERT(iter != fe_names.end(),
          "The finite element name '{}' is not in the list of finite element names.",
          ElementConversion::dealii_fe_name(four_c_element->shape()));

      const unsigned index = std::distance(fe_names.begin(), iter);
      cell->set_active_fe_index(index);
    }

    // Now distribute the dofs, which will use the previously set index
    dof_handler.distribute_dofs(fe_collection);

    // Depending on the element types we need to construct an appropriate mapping collection
    FOUR_C_ASSERT(fe_collection.size() == 1,
        "The current implementation only supports a single finite element type per "
        "discretization.");

    if (fe_collection[0].degree == 1)
    {
      // simple case: just return the isoparametric linear mapping
      return std::make_unique<dealii::hp::MappingCollection<dim, spacedim>>(
          dealii::MappingQ<dim, spacedim>(1));
    }
    else if (fe_collection[0].degree == 2)
    {
      // complicated case: we need to find the shift of the 4C nodes and create a mapping with
      // these

      return std::make_unique<dealii::hp::MappingCollection<dim, spacedim>>(
          dealii::MappingQ<dim, spacedim>(2));
    }
    else
    {
      FOUR_C_THROW("Polynomial FEs with degree higher than 2 are not supported.");
    }
  }


  // explicit instantiation

  template std::unique_ptr<dealii::hp::MappingCollection<3, 3>> assign_fes_and_dofs(
      dealii::DoFHandler<3, 3>&, const Core::FE::Discretization& discretization,
      Context<3, 3>& context);
}  // namespace DealiiWrappers

FOUR_C_NAMESPACE_CLOSE

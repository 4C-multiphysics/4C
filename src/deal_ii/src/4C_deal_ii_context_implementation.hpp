// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_DEAL_II_CONTEXT_IMPLEMENTATION_HPP
#define FOUR_C_DEAL_II_CONTEXT_IMPLEMENTATION_HPP

#include "4C_config.hpp"

#include "4C_deal_ii_context.hpp"
#include "4C_utils_exceptions.hpp"

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <map>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace DealiiWrappers::Internal
{
  template <int dim, int spacedim = dim>
  struct ContextImplementation
  {
    //! Store the local mapping between deal.II cells and 4C elements
    std::map<int, int> cell_index_to_element_lid;

    //! All dealii::FiniteElement objects that are required for the original 4C discretization.
    dealii::hp::FECollection<dim, spacedim> finite_elements;

    //! The names of the FiniteElements in #finite_elements.
    std::vector<std::string> finite_element_names;

    //! Store the geometric mapping necessary for higher order boundary approximation.
    dealii::hp::MappingCollection<dim, spacedim> mapping;

    /**
     * Helper struct to store the information necessary to construct a MappingQEulerian.
     */
    struct MappingContext
    {
      std::unique_ptr<dealii::DoFHandler<dim, spacedim>> mapping_dofs;
      std::unique_ptr<dealii::LinearAlgebra::distributed::Vector<double>> mapping_vector;
    };
  };


  /**
   * Fill mapping data in the context.
   *
   * @pre FiniteElements have already been set in context.
   */
  template <int dim, int spacedim>
  void fill_mapping(Context<dim, spacedim>& context, const Core::FE::Discretization& discretization)
  {
    FOUR_C_ASSERT(context.pimpl_->finite_elements.size() == 1, "Internal error.");

    const auto& fe = context.pimpl_->finite_elements[0];

    if (fe.degree == 1)
    {
      // simple linear mapping
      context.pimpl_->mapping.push_back(dealii::MappingQ<dim, spacedim>(1));
    }
    else if (fe.degree == 2)
    {
      // create a MappingQEulerian that takes the shift into account
    }
    else
      FOUR_C_THROW("Only finite elements up to degree 2 are supported.");
  }
}  // namespace DealiiWrappers::Internal

FOUR_C_NAMESPACE_CLOSE

#endif

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_deal_ii_vector_conversion.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_deal_ii_context_implementation.hpp"
#include "4C_deal_ii_element_conversion.hpp"

#include <deal.II/base/exceptions.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

FOUR_C_NAMESPACE_OPEN

template <int dim, int spacedim>
Core::LinAlg::Map DealiiWrappers::create_dealii_to_four_c_map(
    const dealii::DoFHandler<dim, spacedim>& dof_handler,
    const Core::FE::Discretization& discretization, const Context<dim, spacedim>& context)
{
  const auto& locally_owned_dofs = dof_handler.locally_owned_dofs();

  std::set<std::pair<dealii::types::global_dof_index, int>> local_dealii_four_c_mapping;
  std::set<std::pair<dealii::types::global_dof_index, int>> nonlocal_dealii_four_c_mapping;

  // Go over all local cells and figure out which deal.II dofs map to which 4C dof GIDs
  // We are guaranteed to hit all dof GIDs, but not necessarily on the owning process, so we
  // communicate everything non-local afterwards.
  for (const auto& cell : dof_handler.active_cell_iterators())
  {
    if (!cell->is_locally_owned()) continue;

    const auto& fe = cell->get_fe();
    std::vector<dealii::types::global_dof_index> dof_indices(fe.n_dofs_per_cell());
    cell->get_dof_indices(dof_indices);

    const int element_lid = context.pimpl_->cell_index_to_element_lid.at(cell->index());
    Core::Elements::LocationArray location_array{1};
    const auto* four_c_ele = discretization.l_row_element(element_lid);
    four_c_ele->location_vector(discretization, location_array);


    auto reindexing = ElementConversion::reindex_dealii_to_four_c(four_c_ele->shape());
    FOUR_C_ASSERT(location_array[0].lm_.size() == dof_indices.size(), "Internal error.");

    for (unsigned i = 0; i < dof_indices.size(); ++i)
    {
      const auto [component, index] = fe.system_to_component_index(i);

      const int four_c_la_index = fe.n_components() * reindexing[index] + component;
      const int four_c_gid = location_array[0].lm_[four_c_la_index];

      // sort the mapping from deal.II Dof to 4C GID into a local and nonlocal part
      if (locally_owned_dofs.is_element(dof_indices[i]))
        local_dealii_four_c_mapping.emplace(dof_indices[i], four_c_gid);
      else
        nonlocal_dealii_four_c_mapping.emplace(dof_indices[i], four_c_gid);
    }
  }

  {
    // communicate the nonlocal part so other processes may find data they need
    // first convert to a vector so deal.II can handle the communication
    const auto other_dealii_four_c_mappings = dealii::Utilities::MPI::all_gather(MPI_COMM_WORLD,
        std::vector<std::pair<dealii::types::global_dof_index, int>>(
            nonlocal_dealii_four_c_mapping.begin(), nonlocal_dealii_four_c_mapping.end()));

    for (const auto& proc_results : other_dealii_four_c_mappings)
    {
      for (const auto& [dealii_dof, four_c_gid] : proc_results)
      {
        if (locally_owned_dofs.is_element(dealii_dof))
          local_dealii_four_c_mapping.emplace(dealii_dof, four_c_gid);
      }
    }

    // At this point we must have received all data, i.e., there must be a GID for every local
    // dof
    FOUR_C_ASSERT(
        local_dealii_four_c_mapping.size() == locally_owned_dofs.n_elements(), "Internal error.");
  }

  // Now create an Epetra_map that can convert a deal.II vector to 4C layout

  std::vector<std::pair<dealii::types::global_dof_index, int>> local_mapping(
      local_dealii_four_c_mapping.begin(), local_dealii_four_c_mapping.end());
  std::ranges::sort(local_mapping);

  std::vector<int> my_gids(locally_owned_dofs.n_elements());
  for (unsigned i = 0; i < local_mapping.size(); ++i)
  {
    const auto& [dealii_dof, four_c_gid] = local_mapping[i];
    FOUR_C_ASSERT(dealii_dof == locally_owned_dofs.nth_index_in_set(i), "Internal error.");
    my_gids[i] = four_c_gid;
  }

  Core::LinAlg::Map dealii_to_four_c_map(dof_handler.n_dofs(), locally_owned_dofs.n_elements(),
      my_gids.data(), 0, discretization.get_comm());

  return dealii_to_four_c_map;
}



template <typename VectorType, int dim, int spacedim>
DealiiWrappers::VectorConverter<VectorType, dim, spacedim>::VectorConverter(
    const dealii::DoFHandler<dim, spacedim>& dof_handler,
    const Core::FE::Discretization& discretization, const Context<dim, spacedim>& context)
    : dealii_to_four_c_map_(create_dealii_to_four_c_map(dof_handler, discretization, context)),
      dealii_to_four_c_importer_(
          Core::LinAlg::Map{discretization.dof_row_map()->get_epetra_map()}, dealii_to_four_c_map_),
      vector_in_dealii_layout_(dealii_to_four_c_map_, false)
{
}



template <typename VectorType, int dim, int spacedim>
void DealiiWrappers::VectorConverter<VectorType, dim, spacedim>::to_dealii(
    VectorType& dealii_vector, const Core::LinAlg::Vector<double>& four_c_vector) const
{
  FOUR_C_ASSERT_ALWAYS(four_c_vector.get_map().point_same_as(
                           dealii_to_four_c_importer_.get_epetra_import().TargetMap()),
      "The 4C vector passed to the converter needs to have dof_row_map layout.");
  const int n_local_elements = dealii_vector.locally_owned_size();
  FOUR_C_ASSERT(n_local_elements == dealii_to_four_c_map_.num_my_elements(), "Internal error.");


  vector_in_dealii_layout_.export_to(four_c_vector, dealii_to_four_c_importer_, Insert);

  std::copy(vector_in_dealii_layout_.get_values(),
      vector_in_dealii_layout_.get_values() + n_local_elements, dealii_vector.begin());
}


template <typename VectorType, int dim, int spacedim>
void DealiiWrappers::VectorConverter<VectorType, dim, spacedim>::to_four_c(
    Core::LinAlg::Vector<double>& four_c_vector, const VectorType& dealii_vector) const
{
  FOUR_C_ASSERT_ALWAYS(four_c_vector.get_map().point_same_as(
                           dealii_to_four_c_importer_.get_epetra_import().TargetMap()),
      "The 4C vector passed to the converter needs to have dof_row_map layout.");
  const int n_local_elements = dealii_vector.locally_owned_size();
  FOUR_C_ASSERT(n_local_elements == dealii_to_four_c_map_.num_my_elements(), "Internal error.");

  std::vector<int> indices(n_local_elements);
  std::iota(indices.begin(), indices.end(), 0);
  vector_in_dealii_layout_.replace_local_values(
      n_local_elements, dealii_vector.begin(), indices.data());

  four_c_vector.import(vector_in_dealii_layout_, dealii_to_four_c_importer_, Insert);
}


// --- explicit instantiations --- //
template Core::LinAlg::Map DealiiWrappers::create_dealii_to_four_c_map(
    const dealii::DoFHandler<3, 3>& dof_handler, const Core::FE::Discretization& discretization,
    const Context<3, 3>& context);

template class DealiiWrappers::VectorConverter<dealii::LinearAlgebra::distributed::Vector<double>,
    3, 3>;

FOUR_C_NAMESPACE_CLOSE

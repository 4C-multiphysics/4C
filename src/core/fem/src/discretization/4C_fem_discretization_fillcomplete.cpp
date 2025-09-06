// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_comm_parobjectfactory.hpp"
#include "4C_fem_condition.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_io_pstream.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

void Core::FE::Discretization::reset(bool killdofs, bool killcond)
{
  filled_ = false;
  if (killdofs)
  {
    havedof_ = false;
    for (auto& dofset : dofsets_) dofset->reset();
  }

  elerowmap_ = nullptr;
  elecolmap_ = nullptr;
  elerowptr_.clear();
  elecolptr_.clear();
  noderowmap_ = nullptr;
  nodecolmap_ = nullptr;
  noderowptr_.clear();
  nodecolptr_.clear();

  // delete all old geometries that are attached to any conditions
  // as early as possible
  if (killcond)
  {
    for (const auto& cond : condition_ | std::views::values)
    {
      cond->clear_geometry();
    }
  }
}


int Core::FE::Discretization::fill_complete(
    bool assigndegreesoffreedom, bool initelements, bool doboundaryconditions)
{
  // my processor id
  const int myrank = Core::Communication::my_mpi_rank(get_comm());

  // print information to screen
  if (myrank == 0)
  {
    Core::IO::cout(Core::IO::verbose)
        << "\n+--------------------------------------------------------------------+"
        << Core::IO::endl;
    Core::IO::cout(Core::IO::verbose)
        << "| fill_complete() on discretization " << std::setw(33) << std::left << name()
        << std::setw(1) << std::right << "|" << Core::IO::endl;
  }

  // set all maps to nullptr
  reset(assigndegreesoffreedom, doboundaryconditions);

  // (re)build map of nodes noderowmap_, nodecolmap_, noderowptr and nodecolptr
  build_node_row_map();
  build_node_col_map();

  // (re)build map of elements elemap_
  build_element_row_map();
  build_element_col_map();

  // (re)construct element -> node pointers
  build_element_to_node_pointers();

  // (re)construct node -> element pointers
  build_node_to_element_pointers();

  // set the flag indicating Filled()==true
  // as the following methods make use of maps
  // which we just built
  filled_ = true;

  // Assign degrees of freedom to elements and nodes
  if (assigndegreesoffreedom)
  {
    if (myrank == 0)
    {
      Core::IO::cout(Core::IO::verbose)
          << "| assign_degrees_of_freedom() ...                                    |"
          << Core::IO::endl;
    }
    assign_degrees_of_freedom(0);
  }

  // call element routines to initialize
  if (initelements)
  {
    if (myrank == 0)
    {
      Core::IO::cout(Core::IO::verbose)
          << "| initialize_elements() ...                                          |"
          << Core::IO::endl;
    }
    initialize_elements();
  }

  // (Re)build the geometry of the boundary conditions
  if (doboundaryconditions)
  {
    if (myrank == 0)
    {
      Core::IO::cout(Core::IO::verbose)
          << "| boundary_conditions_geometry() ...                                 |"
          << Core::IO::endl;
    }

    boundary_conditions_geometry();
  }

  if (myrank == 0)
  {
    Core::IO::cout(Core::IO::verbose)
        << "+--------------------------------------------------------------------+"
        << Core::IO::endl;
  }

  return 0;
}


void Core::FE::Discretization::initialize_elements()
{
  if (!filled()) FOUR_C_THROW("fill_complete was not called");

  Core::Communication::ParObjectFactory::instance().initialize_elements(*this);
}

/**
 * Find all entities and put them into a Map. Also create a vector of local pointers to the
 * entities. If local_only is true, only the entities owned by this proc are put into the
 * map and the local pointer vector. If local_only is false, all entities are put in.
 */
template <typename T>
static void make_map_and_local_pointers(MPI_Comm comm,
    const std::map<int, std::shared_ptr<T>>& entities, std::vector<T*>& local_ptrs,
    std::shared_ptr<Core::LinAlg::Map>& map, bool local_only)
{
  const int myrank = Core::Communication::my_mpi_rank(comm);
  const size_t num_entities = local_only ? std::ranges::count(entities | std::views::values, myrank,
                                               [&](const auto& ele) { return ele->owner(); })
                                         : entities.size();

  std::vector<int> local_ids(num_entities);
  local_ptrs.resize(num_entities);

  size_t count = 0;
  for (const auto& [id, entity] : entities)
  {
    FOUR_C_ASSERT(id == entity->id(), "Internal error: mapping is corrupted.");
    if (!local_only || entity->owner() == myrank)
    {
      local_ids[count] = entity->id();
      local_ptrs[count] = entity.get();

      if (!local_only) entity->set_lid(count);

      ++count;
    }
  }

  if (count != num_entities) FOUR_C_THROW("Mismatch in no. of nodes");

  map = std::make_unique<Core::LinAlg::Map>(-1, num_entities, local_ids.data(), 0, comm);
}


void Core::FE::Discretization::build_node_row_map()
{
  make_map_and_local_pointers(get_comm(), node_, noderowptr_, noderowmap_, /*local_only=*/true);
}

void Core::FE::Discretization::build_node_col_map()
{
  make_map_and_local_pointers(get_comm(), node_, nodecolptr_, nodecolmap_, /*local_only=*/false);

  // Build an index into the nodecolptr_ data

  all_local_node_ids_.resize(nodecolptr_.size());
  locally_owned_local_node_ids_.resize(noderowptr_.size());

  size_t count_locally_owned = 0;
  int my_rank = Core::Communication::my_mpi_rank(get_comm());
  for (size_t i = 0; i < nodecolptr_.size(); ++i)
  {
    all_local_node_ids_[i] = i;
    if (nodecolptr_[i]->owner() == my_rank)
    {
      locally_owned_local_node_ids_[count_locally_owned] = i;
      ++count_locally_owned;
    }
  }
  FOUR_C_ASSERT(count_locally_owned == noderowptr_.size(),
      "Internal error: counted {}, expected {}", count_locally_owned, noderowptr_.size());
}


void Core::FE::Discretization::build_element_row_map()
{
  make_map_and_local_pointers(get_comm(), element_, elerowptr_, elerowmap_, /*local_only=*/true);
}

void Core::FE::Discretization::build_element_col_map()
{
  make_map_and_local_pointers(get_comm(), element_, elecolptr_, elecolmap_, /*local_only=*/false);
}


void Core::FE::Discretization::build_element_to_node_pointers()
{
  for (const auto& ele : element_ | std::views::values)
  {
    bool success = ele->build_nodal_pointers(node_);
    if (!success) FOUR_C_THROW("Building element <-> node topology failed");
  }

  std::vector<std::vector<int>> element_connectivity(element_.size());
  for (const auto& ele : element_ | std::views::values)
  {
    const int nnode = ele->num_node();
    element_connectivity[ele->lid()].resize(nnode);
    const int* nodes = ele->node_ids();
    for (int j = 0; j < nnode; ++j) element_connectivity[ele->lid()][j] = nodes[j];
  }
  element_connectivity_.from_nested(element_connectivity);
}


void Core::FE::Discretization::build_node_to_element_pointers()
{
  for (const auto& node : node_ | std::views::values) node->clear_my_element_topology();

  std::vector<std::vector<int>> node_to_element_ids(node_.size());

  for (const auto& ele : element_ | std::views::values)
  {
    const int nnode = ele->num_node();
    const int* nodes = ele->node_ids();
    for (int j = 0; j < nnode; ++j)
    {
      Core::Nodes::Node* node = g_node(nodes[j]);
      FOUR_C_ASSERT_ALWAYS(
          node, "Node {} is not on this proc {}", j, Core::Communication::my_mpi_rank(get_comm()));

      node->add_element_ptr(ele.get());
      node_to_element_ids[node->lid()].push_back(ele->lid());
    }
  }

  node_to_element_lids_.from_nested(node_to_element_ids);
}


int Core::FE::Discretization::assign_degrees_of_freedom(int start)
{
  if (!filled()) FOUR_C_THROW("Filled()==false");
  if (!node_row_map()->unique_gids()) FOUR_C_THROW("Nodal row map is not unique");
  if (!element_row_map()->unique_gids()) FOUR_C_THROW("Element row map is not unique");

  // Set the havedof flag before dofs are assigned. Some dof set
  // implementations do query the discretization after the assignment has been
  // done and this query demands the havedof flag to be set. An unexpected
  // implicit dependency here.
  havedof_ = true;

  for (unsigned i = 0; i < dofsets_.size(); ++i)
    start = dofsets_[i]->assign_degrees_of_freedom(*this, i, start);

  callbacks().post_assign_dofs.call_all(*this);

  return start;
}

FOUR_C_NAMESPACE_CLOSE

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

/*----------------------------------------------------------------------*
 |  Finalize construction (public)                           mwgee 11/06|
 *----------------------------------------------------------------------*/
void Core::FE::Discretization::reset(bool killdofs, bool killcond)
{
  filled_ = false;
  if (killdofs)
  {
    havedof_ = false;
    for (unsigned i = 0; i < dofsets_.size(); ++i) dofsets_[i]->reset();
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
    std::multimap<std::string, std::shared_ptr<Core::Conditions::Condition>>::iterator fool;
    for (fool = condition_.begin(); fool != condition_.end(); ++fool)
    {
      fool->second->clear_geometry();
    }
  }
}


/*----------------------------------------------------------------------*
 |  Finalize construction (public)                           mwgee 11/06|
 *----------------------------------------------------------------------*/
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

  // (re)construct element -> element pointers for interface-elements
  build_element_to_element_pointers();

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


/*----------------------------------------------------------------------*
 |  init elements (public)                                   mwgee 12/06|
 *----------------------------------------------------------------------*/
void Core::FE::Discretization::initialize_elements()
{
  if (!filled()) FOUR_C_THROW("fill_complete was not called");

  Core::Communication::ParObjectFactory::instance().initialize_elements(*this);

  return;
}


/*----------------------------------------------------------------------*
 |  Build noderowmap_ (private)                              mwgee 11/06|
 *----------------------------------------------------------------------*/
void Core::FE::Discretization::build_node_row_map()
{
  const int myrank = Core::Communication::my_mpi_rank(get_comm());
  int nummynodes = 0;
  std::map<int, std::shared_ptr<Core::Nodes::Node>>::iterator curr;
  for (curr = node_.begin(); curr != node_.end(); ++curr)
    if (curr->second->owner() == myrank) ++nummynodes;
  std::vector<int> nodeids(nummynodes);
  noderowptr_.resize(nummynodes);

  int count = 0;
  for (curr = node_.begin(); curr != node_.end(); ++curr)
    if (curr->second->owner() == myrank)
    {
      nodeids[count] = curr->second->id();
      noderowptr_[count] = curr->second.get();
      ++count;
    }
  if (count != nummynodes) FOUR_C_THROW("Mismatch in no. of nodes");
  noderowmap_ = std::make_shared<Core::LinAlg::Map>(-1, nummynodes, nodeids.data(), 0, get_comm());
  return;
}

/*----------------------------------------------------------------------*
 |  Build nodecolmap_ (private)                              mwgee 11/06|
 *----------------------------------------------------------------------*/
void Core::FE::Discretization::build_node_col_map()
{
  int nummynodes = (int)node_.size();
  std::vector<int> nodeids(nummynodes);
  nodecolptr_.resize(nummynodes);

  int count = 0;
  std::map<int, std::shared_ptr<Core::Nodes::Node>>::iterator curr;
  for (curr = node_.begin(); curr != node_.end(); ++curr)
  {
    nodeids[count] = curr->second->id();
    nodecolptr_[count] = curr->second.get();
    curr->second->set_lid(count);
    ++count;
  }
  if (count != nummynodes) FOUR_C_THROW("Mismatch in no. of nodes");
  nodecolmap_ = std::make_shared<Core::LinAlg::Map>(-1, nummynodes, nodeids.data(), 0, get_comm());
  return;
}


/*----------------------------------------------------------------------*
 |  Build elerowmap_ (private)                                mwgee 11/06|
 *----------------------------------------------------------------------*/
void Core::FE::Discretization::build_element_row_map()
{
  const int myrank = Core::Communication::my_mpi_rank(get_comm());
  int nummyeles = 0;
  std::map<int, std::shared_ptr<Core::Elements::Element>>::iterator curr;
  for (curr = element_.begin(); curr != element_.end(); ++curr)
    if (curr->second->owner() == myrank) nummyeles++;
  std::vector<int> eleids(nummyeles);
  elerowptr_.resize(nummyeles);
  int count = 0;
  for (curr = element_.begin(); curr != element_.end(); ++curr)
    if (curr->second->owner() == myrank)
    {
      eleids[count] = curr->second->id();
      elerowptr_[count] = curr->second.get();
      ++count;
    }
  if (count != nummyeles) FOUR_C_THROW("Mismatch in no. of elements");
  elerowmap_ = std::make_shared<Core::LinAlg::Map>(-1, nummyeles, eleids.data(), 0, get_comm());
  return;
}

/*----------------------------------------------------------------------*
 |  Build elecolmap_ (private)                                mwgee 11/06|
 *----------------------------------------------------------------------*/
void Core::FE::Discretization::build_element_col_map()
{
  int nummyeles = (int)element_.size();
  std::vector<int> eleids(nummyeles);
  elecolptr_.resize(nummyeles);
  std::map<int, std::shared_ptr<Core::Elements::Element>>::iterator curr;
  int count = 0;
  for (curr = element_.begin(); curr != element_.end(); ++curr)
  {
    eleids[count] = curr->second->id();
    elecolptr_[count] = curr->second.get();
    curr->second->set_lid(count);
    ++count;
  }
  if (count != nummyeles) FOUR_C_THROW("Mismatch in no. of elements");
  elecolmap_ = std::make_shared<Core::LinAlg::Map>(-1, nummyeles, eleids.data(), 0, get_comm());
  return;
}

/*----------------------------------------------------------------------*
 |  Build ptrs element -> node (private)                      mwgee 11/06|
 *----------------------------------------------------------------------*/
void Core::FE::Discretization::build_element_to_node_pointers()
{
  std::map<int, std::shared_ptr<Core::Elements::Element>>::iterator elecurr;
  for (elecurr = element_.begin(); elecurr != element_.end(); ++elecurr)
  {
    bool success = elecurr->second->build_nodal_pointers(node_);
    if (!success) FOUR_C_THROW("Building element <-> node topology failed");
  }
  return;
}

/*----------------------------------------------------------------------*
 |  Build ptrs element -> element (private)                      mwgee 11/06|
 *----------------------------------------------------------------------*/
void Core::FE::Discretization::build_element_to_element_pointers()
{
  std::map<int, std::shared_ptr<Core::Elements::Element>>::iterator elecurr;
  for (elecurr = element_.begin(); elecurr != element_.end(); ++elecurr)
  {
    bool success = elecurr->second->build_element_pointers(element_);
    if (!success) FOUR_C_THROW("Building element <-> element topology failed");
  }
  return;
}

/*----------------------------------------------------------------------*
 |  Build ptrs node -> element (private)                      mwgee 11/06|
 *----------------------------------------------------------------------*/
void Core::FE::Discretization::build_node_to_element_pointers()
{
  std::map<int, std::shared_ptr<Core::Nodes::Node>>::iterator nodecurr;
  for (nodecurr = node_.begin(); nodecurr != node_.end(); ++nodecurr)
    nodecurr->second->clear_my_element_topology();

  std::map<int, std::shared_ptr<Core::Elements::Element>>::iterator elecurr;
  for (elecurr = element_.begin(); elecurr != element_.end(); ++elecurr)
  {
    const int nnode = elecurr->second->num_node();
    const int* nodes = elecurr->second->node_ids();
    for (int j = 0; j < nnode; ++j)
    {
      Core::Nodes::Node* node = g_node(nodes[j]);
      if (!node)
        FOUR_C_THROW(
            "Node {} is not on this proc {}", j, Core::Communication::my_mpi_rank(get_comm()));
      else
        node->add_element_ptr(elecurr->second.get());
    }
  }
  return;
}

/*----------------------------------------------------------------------*
 |  set degrees of freedom (protected)                       mwgee 03/07|
 *----------------------------------------------------------------------*/
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

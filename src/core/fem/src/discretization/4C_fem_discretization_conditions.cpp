// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_condition_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_linalg_utils_densematrix_communication.hpp"
#include "4C_utils_exceptions.hpp"

#include <algorithm>

FOUR_C_NAMESPACE_OPEN

void Core::FE::Discretization::boundary_conditions_geometry()
{
  // As a first step we delete ALL references to any conditions
  // in the discretization
  for (int i = 0; i < num_my_col_nodes(); ++i) l_col_node(i)->clear_conditions();

  // now we delete all old geometries that are attached to any conditions
  // and set a communicator to the condition
  for (auto& [name, condition] : condition_)
  {
    condition->clear_geometry();
  }

  // for all conditions, we set a ptr in the nodes to the condition
  for (auto& [name, condition] : condition_)
  {
    const std::vector<int>* nodes = condition->get_nodes();
    // There might be conditions that do not have a nodal cloud
    if (!nodes) continue;
    for (int node : *nodes)
    {
      if (!node_col_map()->my_gid(node)) continue;
      Core::Nodes::Node* actnode = g_node(node);
      if (!actnode) FOUR_C_THROW("Cannot find global node");
      actnode->set_condition(name, condition);
    }
  }

  // create a map that holds the overall number of created elements
  // associated with a specific condition type
  std::map<std::string, int> numele;

  // Loop all conditions and build geometry description if desired
  for (auto& [name, condition] : condition_)
  {
    // flag whether new elements have been created for this condition
    bool havenewelements = false;

    // do not build geometry description for this condition
    if (!condition->geometry_description()) continue;
    // do not build geometry description for this condition
    else if (condition->g_type() == Core::Conditions::geometry_type_no_geom)
      continue;
    // do not build anything for point wise conditions
    else if (condition->g_type() == Core::Conditions::geometry_type_point)
      continue;
    // build element geometry description without creating new elements if the
    // condition is not a boundary condition; this would be:
    //  - line conditions in 1D
    //  - surface conditions in 2D
    //  - volume conditions in 3D
    else if (Core::Conditions::geometry_type_to_dim.at(condition->g_type()) == n_dim_)
      havenewelements = build_volumes_in_condition(name, *condition);
    // dimension of condition must not larger than the one of the problem itself
    else if (Core::Conditions::geometry_type_to_dim.at(condition->g_type()) > n_dim_)
      FOUR_C_THROW("Dimension of condition is larger than the problem dimension.");
    // build a line element geometry description
    else if (condition->g_type() == Core::Conditions::geometry_type_line)
      havenewelements = build_lines_in_condition(name, *condition);
    // build a surface element geometry description
    else if (condition->g_type() == Core::Conditions::geometry_type_surface)
      havenewelements = build_surfaces_in_condition(name, *condition);
    // this should be it. if not: FOUR_C_THROW.
    else
      FOUR_C_THROW("Somehow the condition geometry does not fit to the problem dimension.");

    if (havenewelements)
    {
      // determine the local number of created elements associated with
      // the active condition
      int localcount = 0;
      for (const auto& [ele_id, ele] : condition->geometry())
      {
        // do not count ghosted elements
        if (ele->owner() == Core::Communication::my_mpi_rank(get_comm()))
        {
          localcount += 1;
        }
      }

      // determine the global number of created elements associated with
      // the active condition
      int count;
      Core::Communication::sum_all(&localcount, &count, 1, get_comm());

      if (numele.find(name) == numele.end())
      {
        numele[name] = 0;
      }

      // adjust the IDs of the elements associated with the active
      // condition in order to obtain unique IDs within one condition type
      condition->adjust_id(numele[name]);

      // adjust the number of elements associated with the current
      // condition type
      numele[name] += count;
    }
  }
}


void Core::FE::Discretization::assign_global_ids(MPI_Comm comm,
    const std::map<std::vector<int>, std::shared_ptr<Core::Elements::Element>>& elementmap,
    std::map<int, std::shared_ptr<Core::Elements::Element>>& finalgeometry)
{
  // pack elements on all processors

  int size = 0;
  for (const auto& [nodes, ele] : elementmap)
  {
    size += nodes.size() + 1;
  }
  std::vector<int> sendblock;
  sendblock.reserve(size);
  for (const auto& [nodes, ele] : elementmap)
  {
    sendblock.push_back(nodes.size());
    std::copy(nodes.begin(), nodes.end(), std::back_inserter(sendblock));
  }

  // communicate elements to processor 0

  int mysize = sendblock.size();
  Core::Communication::sum_all(&mysize, &size, 1, comm);
  int mypos = Core::LinAlg::find_my_pos(sendblock.size(), comm);

  std::vector<int> send(size);
  std::ranges::fill(send, 0);
  std::ranges::copy(sendblock, send.data() + mypos);
  sendblock.clear();
  std::vector<int> recv(size);
  Core::Communication::sum_all(send.data(), recv.data(), size, comm);

  send.clear();

  // unpack, unify and sort elements on processor 0

  if (Core::Communication::my_mpi_rank(comm) == 0)
  {
    std::set<std::vector<int>> elements;
    int index = 0;
    while (index < static_cast<int>(recv.size()))
    {
      int esize = recv[index];
      index += 1;
      std::vector<int> element;
      element.reserve(esize);
      std::copy(recv.begin() + index, recv.begin() + index + esize, std::back_inserter(element));
      index += esize;
      elements.insert(element);
    }
    recv.clear();

    // pack again to distribute pack to all processors

    send.reserve(index);
    for (const auto& ele : elements)
    {
      send.push_back(ele.size());
      std::copy(ele.begin(), ele.end(), std::back_inserter(send));
    }
    size = send.size();
  }
  else
  {
    recv.clear();
  }

  // broadcast sorted elements to all processors

  Core::Communication::broadcast(&size, 1, 0, comm);
  send.resize(size);
  Core::Communication::broadcast(send.data(), send.size(), 0, comm);

  // Unpack sorted elements. Take element position for gid.

  int index = 0;
  int gid = 0;
  while (index < static_cast<int>(send.size()))
  {
    int esize = send[index];
    index += 1;
    std::vector<int> element;
    element.reserve(esize);
    std::copy(send.begin() + index, send.begin() + index + esize, std::back_inserter(element));
    index += esize;

    // set gid to my elements
    auto iter = elementmap.find(element);
    if (iter != elementmap.end())
    {
      iter->second->set_id(gid);
      finalgeometry[gid] = iter->second;
    }

    gid += 1;
  }
}  // assign_global_ids


bool Core::FE::Discretization::build_lines_in_condition(
    const std::string& name, Core::Conditions::Condition& cond)
{
  /* First: Create the line objects that belong to the condition. */

  // get ptrs to all node ids that have this condition
  const std::vector<int>* nodeids = cond.get_nodes();
  if (!nodeids) FOUR_C_THROW("Cannot find array 'Node Ids' in condition");

  // ptrs to my row/column nodes of those
  std::map<int, Core::Nodes::Node*> colnodes;

  for (const auto& nodeid : *nodeids)
  {
    if (node_col_map()->my_gid(nodeid))
    {
      Core::Nodes::Node* actnode = g_node(nodeid);
      if (!actnode) FOUR_C_THROW("Cannot find global node");
      colnodes[actnode->id()] = actnode;
    }
  }

  // map of lines in our cloud: (node_ids) -> line
  std::map<std::vector<int>, std::shared_ptr<Core::Elements::Element>> linemap;
  // loop these nodes and build all lines attached to them
  for (const auto& [node_id, actnode] : colnodes)
  {
    // loop all elements attached to actnode
    std::span<Core::Elements::Element*> elements(actnode->elements(), actnode->num_element());
    for (const auto& element : elements)
    {
      // loop all lines of all elements attached to actnode
      const int numlines = element->num_line();
      if (!numlines) continue;
      std::vector<std::shared_ptr<Core::Elements::Element>> lines = element->lines();
      if (lines.size() == 0) FOUR_C_THROW("Element returned no lines");
      for (int j = 0; j < numlines; ++j)
      {
        std::shared_ptr<Core::Elements::Element> actline = lines[j];
        // find lines that are attached to actnode
        const int nnodeperline = actline->num_node();
        Core::Nodes::Node** nodesperline = actline->nodes();
        if (!nodesperline) FOUR_C_THROW("Line returned no nodes");
        for (int k = 0; k < nnodeperline; ++k)
        {
          if (nodesperline[k]->id() == actnode->id())
          {
            // line is attached to actnode
            // see whether all nodes on the line are in our nodal cloud
            bool allin = true;
            for (int l = 0; l < nnodeperline; ++l)
            {
              if (colnodes.find(nodesperline[l]->id()) == colnodes.end())
              {
                allin = false;
                break;
              }
            }  // for (int l=0; l<nnodeperline; ++l)
            // if all nodes on line are in our cloud, add line
            if (allin)
            {
              std::vector<int> nodes(actline->num_node());
              std::transform(actline->nodes(), actline->nodes() + actline->num_node(),
                  nodes.begin(), std::mem_fn(&Core::Nodes::Node::id));
              std::ranges::sort(nodes);

              if (!linemap.contains(nodes))
              {
                auto line = std::shared_ptr<Core::Elements::Element>(actline->clone());
                line->set_owner(element->owner());
                linemap[nodes] = line;
              }
            }
            break;
          }
        }
      }
    }
  }


  // Lines be added to the condition: (line_id) -> (line).
  std::map<int, std::shared_ptr<Core::Elements::Element>> finallines;

  assign_global_ids(get_comm(), linemap, finallines);

  cond.set_geometry(std::move(finallines));

  // elements were created that need new unique ids
  return true;
}


bool Core::FE::Discretization::build_surfaces_in_condition(
    const std::string& name, Core::Conditions::Condition& cond)
{
  /* First: Create the surface objects that belong to the condition. */

  // get ptrs to all node ids that have this condition
  const std::vector<int>* nodeids = cond.get_nodes();
  if (!nodeids) FOUR_C_THROW("Cannot find array 'Node Ids' in condition");

  // ptrs to my row/column nodes of those
  std::map<int, Core::Nodes::Node*> myrownodes;
  std::map<int, Core::Nodes::Node*> mycolnodes;
  for (const auto& nodeid : *nodeids)
  {
    if (node_col_map()->my_gid(nodeid))
    {
      Core::Nodes::Node* actnode = g_node(nodeid);
      if (!actnode) FOUR_C_THROW("Cannot find global node");
      mycolnodes[actnode->id()] = actnode;
    }
    if (node_row_map()->my_gid(nodeid))
    {
      Core::Nodes::Node* actnode = g_node(nodeid);
      if (!actnode) FOUR_C_THROW("Cannot find global node");
      myrownodes[actnode->id()] = actnode;
    }
  }

  // map of surfaces in this cloud: (node_ids) -> (surface)
  std::map<std::vector<int>, std::shared_ptr<Core::Elements::Element>> surfmap;

  // loop these column nodes and build all surfs attached to them.
  // we have to loop over column nodes because it can happen that
  // we want to create a surface on an element face which has only
  // ghosted nodes. if the resulting condition geometry is then used for
  // cloning a discretization from it, we copy the condition to the
  // cloned surface discretization, and if we build the geometry of
  // this surface discretization, this way we make sure that we do not
  // miss a surface element. otherwise, we would miss the surface element
  // of the face which has now only ghosted nodes because we would only
  // look at row nodes but the considered face has no row node.
  for (const auto& [node_id, actnode] : mycolnodes)
  {
    Core::Elements::Element** elements = actnode->elements();

    // loop all elements attached to actnode
    for (int i = 0; i < actnode->num_element(); ++i)
    {
      // loop all surfaces of all elements attached to actnode
      const int numsurfs = elements[i]->num_surface();
      if (!numsurfs) continue;
      std::vector<std::shared_ptr<Core::Elements::Element>> surfs = elements[i]->surfaces();
      if (surfs.size() == 0) FOUR_C_THROW("Element does not return any surfaces");

      // loop all surfaces of all elements attached to actnode
      for (int j = 0; j < numsurfs; ++j)
      {
        std::shared_ptr<Core::Elements::Element> actsurf = surfs[j];
        // find surfs attached to actnode
        const int nnodepersurf = actsurf->num_node();
        Core::Nodes::Node** nodespersurf = actsurf->nodes();
        if (!nodespersurf) FOUR_C_THROW("Surface returned no nodes");
        for (int k = 0; k < nnodepersurf; ++k)
        {
          if (nodespersurf[k]->id() == actnode->id())
          {
            // surface is attached to actnode
            // see whether all  nodes on the surface are in our cloud
            bool is_conditioned_surface = true;
            for (int l = 0; l < nnodepersurf; ++l)
            {
              if (mycolnodes.find(nodespersurf[l]->id()) == mycolnodes.end())
              {
                is_conditioned_surface = false;
                // continue with next element surface
                break;
              }
            }
            // if all nodes are in our cloud, add surface
            if (is_conditioned_surface)
            {
              // get sorted vector of node ids
              std::vector<int> nodes(actsurf->num_node());
              transform(actsurf->nodes(), actsurf->nodes() + actsurf->num_node(), nodes.begin(),
                  std::mem_fn(&Core::Nodes::Node::id));
              sort(nodes.begin(), nodes.end());

              // now we can add the surface
              if (surfmap.find(nodes) == surfmap.end())
              {
                auto surf = std::shared_ptr<Core::Elements::Element>(actsurf->clone());
                // Set owning processor of surface owner of underlying volume element.
                surf->set_owner(elements[i]->owner());
                surfmap[nodes] = surf;
              }  // if surface not yet in map
            }  // if all nodes of surface belong to condition (is_conditioned_surface == true)
          }  // if surface contains conditioned row node
        }  // loop over all nodes of element surface
      }  // loop over all element surfaces
    }  // loop over all adjacent elements of conditioned row node
  }  // loop over all conditioned row nodes

  // surfaces be added to the condition: (surf_id) -> (surface).
  std::map<int, std::shared_ptr<Core::Elements::Element>> final_geometry;

  assign_global_ids(get_comm(), surfmap, final_geometry);
  cond.set_geometry(std::move(final_geometry));

  // elements were created that need new unique ids
  return true;
}


bool Core::FE::Discretization::build_volumes_in_condition(
    const std::string& name, Core::Conditions::Condition& cond)
{
  // get ptrs to all node ids that have this condition
  const std::vector<int>* nodeids = cond.get_nodes();
  if (!nodeids) FOUR_C_THROW("Cannot find array 'Node Ids' in condition");

  // extract colnodes on this proc from condition
  const Core::LinAlg::Map* colmap = node_col_map();
  std::set<int> mynodes;

  std::ranges::remove_copy_if(*nodeids, std::inserter(mynodes, mynodes.begin()),
      std::not_fn(Core::Conditions::MyGID(colmap)));

  // this is the map we want to construct
  std::map<int, std::shared_ptr<Core::Elements::Element>> geom;

  for (const auto& [ele_id, actele] : element_)
  {
    std::span<const int> myelenodes(actele->node_ids(), actele->node_ids() + actele->num_node());

    // check whether all node ids of the element are nodes belonging
    // to the condition and stored on this proc
    bool allin = true;
    for (const auto& myid : myelenodes)
    {
      if (mynodes.find(myid) == mynodes.end())
      {
        // myid is not in the condition
        allin = false;
        break;
      }
    }

    if (allin)
    {
      geom[ele_id] = actele;
    }
  }

  cond.set_geometry(std::move(geom));

  // no elements where created to assign new unique ids to
  return false;
}


FOUR_C_NAMESPACE_CLOSE

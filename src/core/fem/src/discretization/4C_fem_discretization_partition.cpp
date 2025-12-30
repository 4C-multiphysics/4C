// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_comm_exporter.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_dofset_pbc.hpp"
#include "4C_linalg_utils_densematrix_communication.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::FE::Discretization::export_row_nodes(
    const Core::LinAlg::Map& newmap, bool killdofs, bool killcond)
{
  // test whether newmap is non-overlapping
  if (!newmap.unique_gids()) FOUR_C_THROW("new map not unique");

  // destroy all ghosted nodes
  const int myrank = Core::Communication::my_mpi_rank(get_comm());
  for (auto curr = node_.begin(); curr != node_.end();)
  {
    if (curr->second->owner() != myrank)
      node_.erase(curr++);
    else
      ++curr;
  }

  // build rowmap of nodes noderowmap_ if it does not exist
  if (noderowmap_ == nullptr) build_node_row_map();
  const Core::LinAlg::Map& oldmap = *noderowmap_;

  // create an exporter object that will figure out the communication pattern
  Core::Communication::Exporter exporter(oldmap, newmap, get_comm());

  // Do the communication
  exporter.do_export(node_);

  // update all ownership flags
  for (auto& val : node_ | std::views::values) val->set_owner(myrank);

  // maps and pointers are no longer correct and need rebuilding
  reset(killdofs, killcond);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::FE::Discretization::export_column_nodes(
    const Core::LinAlg::Map& newmap, bool killdofs, bool killcond)
{
  // destroy all ghosted nodes
  const int myrank = Core::Communication::my_mpi_rank(get_comm());
  for (auto curr = node_.begin(); curr != node_.end();)
  {
    if (curr->second->owner() != myrank)
      node_.erase(curr++);
    else
      ++curr;
  }

  // build rowmap of nodes noderowmap_ if it does not exist
  if (noderowmap_ == nullptr) build_node_row_map();
  const Core::LinAlg::Map& oldmap = *noderowmap_;

  // test whether all nodes in oldmap are also in newmap, otherwise
  // this would be a change of owner which is not allowed here
  for (int i = 0; i < oldmap.num_my_elements(); ++i)
  {
    int gid = oldmap.gid(i);
    if (!(newmap.my_gid(gid)))
      FOUR_C_THROW("Proc {}: Node gid={} from oldmap is not in newmap", myrank, gid);
  }

  // create an exporter object that will figure out the communication pattern
  Core::Communication::Exporter exporter(oldmap, newmap, get_comm());
  // Do the communication
  exporter.do_export(node_);

  // maps and pointers are no longer correct and need rebuilding
  reset(killdofs, killcond);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::FE::Discretization::proc_zero_distribute_elements_to_all(
    Core::LinAlg::Map& target, std::vector<int>& gidlist)
{
  const int myrank = Core::Communication::my_mpi_rank(get_comm());

  const size_t n_elements_to_distribute = gidlist.size();
  std::vector<int> target_process(n_elements_to_distribute);
  int err = target.remote_id_list(
      n_elements_to_distribute, gidlist.data(), target_process.data(), nullptr);

  if (err < 0) FOUR_C_THROW("Core::LinAlg::Map::remoted_id_list() returned err={}", err);

  // Raw data that is to be sent to other processors
  std::map<int, std::vector<char>> sendmap;
  if (myrank == 0)
  {
    std::map<int, Core::Communication::PackBuffer> pack_buffers;
    for (size_t i = 0; i < n_elements_to_distribute; ++i)
    {
      if (target_process[i] == myrank or target_process[i] < 0) continue;  // do not send to myself

      Core::Elements::Element* actele = g_element(gidlist[i]);
      if (!actele) FOUR_C_THROW("Cannot find global element {}", gidlist[i]);
      actele->pack(pack_buffers[target_process[i]]);
      element_.erase(actele->id());
    }
    for (auto& [pid, pack_buffer] : pack_buffers) swap(sendmap[pid], pack_buffer());
  }

  // tell everybody who is to receive something
  std::vector<int> receivers;
  receivers.reserve(sendmap.size());

  for (const auto& pid : sendmap | std::views::keys) receivers.push_back(pid);

  size_t n_destination_processes = receivers.size();
  Core::Communication::broadcast(n_destination_processes, 0, get_comm());
  if (myrank != 0) receivers.resize(n_destination_processes);
  Core::Communication::broadcast(receivers.data(), n_destination_processes, 0, get_comm());

  const int this_process_receives_index = std::invoke(
      [&]()
      {
        if (myrank != 0)
        {
          size_t index = std::distance(receivers.begin(), std::ranges::find(receivers, myrank));
          if (index < receivers.size()) return static_cast<int>(index);
        }
        return -1;
      });

  // proc 0 sends out messages
  Core::Communication::Exporter exporter(get_comm());
  std::vector<MPI_Request> request(n_destination_processes);
  if (!myrank)
  {
    size_t tag = 0;
    for (auto& [pid, data] : sendmap)
    {
      exporter.i_send(0, pid, data.data(), (int)data.size(), tag, request[tag]);
      tag++;
    }
    if (tag != n_destination_processes) FOUR_C_THROW("Number of messages is mixed up");
  }

  // all other procs listen to message and put element into dis
  if (this_process_receives_index != -1)
  {
    std::vector<char> recvdata;
    int length = 0;
    int source = -1;
    int tag = -1;
    exporter.receive_any(source, tag, recvdata, length);
    if (source != 0 || tag != this_process_receives_index) FOUR_C_THROW("Messages got mixed up");
    // Put received elements into discretization
    Communication::UnpackBuffer buffer(recvdata);
    while (!buffer.at_end())
    {
      Core::Communication::ParObject* object = Core::Communication::factory(buffer);
      auto* ele = dynamic_cast<Core::Elements::Element*>(object);
      if (!ele) FOUR_C_THROW("Received object is not an element");
      ele->set_owner(myrank);
      std::shared_ptr<Core::Elements::Element> rcpele(ele);
      add_element(rcpele);
    }
  }

  // wait for all communication to finish
  if (!myrank)
    for (size_t i = 0; i < n_destination_processes; ++i) exporter.wait(request[i]);

  Core::Communication::barrier(get_comm());  // I feel better this way ;-)
  reset();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::FE::Discretization::export_row_elements(
    const Core::LinAlg::Map& newmap, bool killdofs, bool killcond)
{
  // test whether newmap is non-overlapping
  if (!newmap.unique_gids()) FOUR_C_THROW("new map not unique");

  // destroy all ghosted elements
  const int myrank = Core::Communication::my_mpi_rank(get_comm());
  for (auto curr = element_.begin(); curr != element_.end();)
  {
    if (curr->second->owner() != myrank)
      element_.erase(curr++);
    else
      ++curr;
  }

  // build map of elements elerowmap_ if it does not exist
  if (elerowmap_ == nullptr) build_element_row_map();
  const Core::LinAlg::Map& oldmap = *elerowmap_;

  // create an exporter object that will figure out the communication pattern
  Core::Communication::Exporter exporter(oldmap, newmap, get_comm());

  exporter.do_export(element_);

  for (auto& ele : element_ | std::views::values)
  {
    ele->set_owner(myrank);
    ele->discretization_ = this;
  }

  // maps and pointers are no longer correct and need rebuilding
  reset(killdofs, killcond);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::FE::Discretization::export_column_elements(
    const Core::LinAlg::Map& newmap, bool killdofs, bool killcond)
{
  // destroy all ghosted elements
  const int myrank = Core::Communication::my_mpi_rank(get_comm());
  for (auto curr = element_.begin(); curr != element_.end();)
  {
    if (curr->second->owner() != myrank)
      element_.erase(curr++);
    else
      ++curr;
  }

  // build map of elements elerowmap_ if it does not exist
  if (elerowmap_ == nullptr) build_element_row_map();
  const Core::LinAlg::Map& oldmap = *elerowmap_;

  // test whether all elements in oldmap are also in newmap
  // Otherwise, this would be a change of owner which is not allowed here
  for (int i = 0; i < oldmap.num_my_elements(); ++i)
  {
    int gid = oldmap.gid(i);
    if (!(newmap.my_gid(gid)))
      FOUR_C_THROW("Proc {}: Element gid={} from oldmap is not in newmap", myrank, gid);
  }

  // create an exporter object that will figure out the communication pattern
  Core::Communication::Exporter exporter(oldmap, newmap, get_comm());
  exporter.do_export(element_);

  // maps and pointers are no longer correct and need rebuilding
  reset(killdofs, killcond);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Graph> Core::FE::Discretization::build_node_graph() const
{
  if (!filled()) FOUR_C_THROW("fill_complete() was not called on this discretization");

  // get nodal row map
  const Core::LinAlg::Map* noderowmap = node_row_map();

  // allocate graph
  auto graph = std::make_shared<Core::LinAlg::Graph>(*noderowmap, 108);

  // iterate all elements on this proc including ghosted ones
  // Note:
  // if a proc stores the appropriate ghosted elements, the resulting
  // graph will be the correct and complete graph of the distributed
  // discretization even if nodes are not ghosted.
  for (const auto& val : element_ | std::views::values)
  {
    const int nnode = val->num_node();
    const int* nodeids = val->node_ids();
    for (int row = 0; row < nnode; ++row)
    {
      const int rownode = nodeids[row];
      if (!noderowmap->my_gid(rownode)) continue;
      for (int col = 0; col < nnode; ++col)
      {
        int colnode = nodeids[col];
        auto indices = std::span(&colnode, 1);
        graph->insert_global_indices(rownode, indices);
      }
    }
  }
  graph->fill_complete();
  graph->optimize_storage();

  return graph;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::pair<std::shared_ptr<Core::LinAlg::Map>, std::shared_ptr<Core::LinAlg::Map>>
Core::FE::Discretization::build_element_row_column(const Core::LinAlg::Map& noderowmap,
    const Core::LinAlg::Map& nodecolmap, bool find_ghost_elements_with_no_owned_node) const
{
  const int myrank = Core::Communication::my_mpi_rank(get_comm());
  const int numproc = Core::Communication::num_mpi_ranks(get_comm());

  // note:
  // - noderowmap need not match distribution of nodes in this
  //   discretization at all.
  // - noderowmap is a non-overlapping map, that's tested
  if (!noderowmap.unique_gids()) FOUR_C_THROW("noderowmap is not a unique map");

  // find all owners for the overlapping node map
  const int ncnode = nodecolmap.num_my_elements();
  std::vector<int> cnodeowner(ncnode);
  int err = noderowmap.remote_id_list(
      ncnode, nodecolmap.my_global_elements(), cnodeowner.data(), nullptr);
  if (err) FOUR_C_THROW("Core::LinAlg::Map::RemoteIDLis returned err={}", err);

  // build connectivity of elements
  // storing :  element gid
  //            no. of nodes
  //            nodeids
  int stoposize = 2000;
  int count = 0;
  std::vector<int> stopo(stoposize);
  std::map<int, std::shared_ptr<Core::Elements::Element>>::const_iterator ecurr;
  for (ecurr = element_.begin(); ecurr != element_.end(); ++ecurr)
  {
    const Core::Elements::Element& actele = *(ecurr->second);
    int gid = actele.id();
    int nnode = actele.num_node();
    const int* nodeids = actele.node_ids();
    if (count + nnode + 2 >= stoposize)
    {
      stoposize += (nnode + 2) * 300;
      stopo.resize(stoposize);
    }
    stopo[count++] = gid;
    stopo[count++] = nnode;
    for (int j = 0; j < nnode; ++j) stopo[count++] = nodeids[j];
  }
  stoposize = count;
  stopo.resize(stoposize);

  std::vector<int> rtopo(stoposize);

  // communicate number of nodes per proc
  std::vector<int> nodesperproc(numproc);
  int nummynodes = noderowmap.num_my_elements();
  Core::Communication::gather_all(&nummynodes, nodesperproc.data(), 1, get_comm());

  // estimate no. of elements equal to no. of nodes
  std::vector<int> myele(nummynodes);
  int nummyele = 0;
  // estimate no. of ghosted elements much lower
  std::vector<int> myghostele(nummynodes / 4);
  int nummyghostele = 0;

  // loop processors and sort elements into
  // elements owned by a proc
  // elements ghosted by a proc
  for (int proc = 0; proc < numproc; ++proc)
  {
    int size = stoposize;
    Core::Communication::broadcast(&size, 1, proc, get_comm());
    if (size > (int)rtopo.size()) rtopo.resize(size);
    if (proc == myrank)
      for (int i = 0; i < size; ++i) rtopo[i] = stopo[i];
    Core::Communication::broadcast(rtopo.data(), size, proc, get_comm());
    for (int i = 0; i < size;)
    {
      const int elegid = rtopo[i++];
      const int numnode = rtopo[i++];
      const int* nodeids = &rtopo[i];
      i += numnode;

      // resize arrays
      if (nummyele >= (int)myele.size()) myele.resize(myele.size() + 500);
      if (nummyghostele >= (int)myghostele.size()) myghostele.resize(myghostele.size() + 500);

      // count nodes I own of this element
      int nummine = 0;
      for (int j = 0; j < numnode; ++j)
        if (noderowmap.my_gid(nodeids[j])) ++nummine;

      // Check if I own nodes of this element
      if (!nummine)
      {
        // If the rebalance type is monolithic, activate additional ghosting
        if (find_ghost_elements_with_no_owned_node)
        {
          // If all nodes of the element are in col map we still ghost it
          bool all_nodes_in_col = true;
          for (int j = 0; j < numnode; ++j)
            if (!nodecolmap.my_gid(nodeids[j])) all_nodes_in_col = false;

          if (all_nodes_in_col)
          {
            myghostele[nummyghostele++] = elegid;
          }
          continue;
        }

        // if I do not own any of the nodes, it is definitely not my element
        // and I do not ghost it
        else
          continue;
      }

      // check whether I ghost all nodes of this element
      // this is necessary to be able to own or ghost the element
      for (int j = 0; j < numnode; ++j)
        if (!nodecolmap.my_gid(nodeids[j]))
          FOUR_C_THROW("I do not have own/ghosted node gid={}", nodeids[j]);

      // find out who owns how many of the nodes
      std::vector<int> nodeowner(numnode);
      std::vector<int> numperproc(numproc);
      for (int j = 0; j < numproc; ++j) numperproc[j] = 0;
      for (int j = 0; j < numnode; ++j)
      {
        const int lid = nodecolmap.lid(nodeids[j]);
        const int owner = cnodeowner[lid];
        nodeowner[j] = owner;
        numperproc[owner]++;
      }

      // the proc with the largest number of nodes owns the element,
      // all others ghost it
      //
      // tie-breaking if number of nodes is equal among some procs:
      // the processor with the smaller number of row nodes owns the element;
      // if still tied, the last node owner with equal number of nodes owns
      // the element
      int owner = -1;
      int maxnode = 0;
      int minrownodes = noderowmap.num_global_elements();
      for (int j = 0; j < numnode; ++j)
      {
        int currentproc = nodeowner[j];
        int ownhowmany = numperproc[currentproc];
        if (ownhowmany > maxnode ||
            (ownhowmany == maxnode && nodesperproc[currentproc] <= minrownodes))
        {
          owner = currentproc;
          maxnode = ownhowmany;
          minrownodes = nodesperproc[currentproc];
        }
      }
      if (myrank == owner)
      {
        myele[nummyele++] = elegid;
        continue;
      }
      else
      {
        myghostele[nummyghostele++] = elegid;
        continue;
      }
      FOUR_C_THROW("Error in logic of element ownerships");

    }  // for (int i=0; i<size;)
  }  // for (int proc=0; proc<numproc; ++proc)

  // at this point we have
  // myele, length nummyele
  // myghostele, length nummyghostele
  myele.resize(nummyele);
  myghostele.resize(nummyghostele);

  // allreduced nummyele must match the total no. of elements in this
  // discretization, otherwise we lost some
  // build the rowmap of elements
  std::shared_ptr<Core::LinAlg::Map> elerowmap =
      std::make_shared<Core::LinAlg::Map>(-1, nummyele, myele.data(), 0, get_comm());
  if (!elerowmap->unique_gids()) FOUR_C_THROW("Element row map is not unique");

  // build elecolmap
  std::vector<int> elecol(nummyele + nummyghostele);
  for (int i = 0; i < nummyele; ++i) elecol[i] = myele[i];
  for (int i = 0; i < nummyghostele; ++i) elecol[nummyele + i] = myghostele[i];
  std::shared_ptr<Core::LinAlg::Map> elecolmap = std::make_shared<Core::LinAlg::Map>(
      -1, nummyghostele + nummyele, elecol.data(), 0, get_comm());

  return {elerowmap, elecolmap};
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::FE::Discretization::redistribute(
    const RowColMaps& node_maps, OptionsRedistribution options)
{
  // build the overlapping and non-overlapping element maps
  const auto& [elerowmap, elecolmap] =
      build_element_row_column(node_maps.row_map, node_maps.col_map, options.do_extended_ghosting);

  redistribute(node_maps, {.row_map = *elerowmap, .col_map = *elecolmap}, options);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::FE::Discretization::redistribute(
    const RowColMaps& node_maps, const RowColMaps& element_maps, OptionsRedistribution options)
{
  // export nodes and elements to the new maps
  export_row_nodes(node_maps.row_map, options.kill_dofs, options.kill_cond);
  export_column_nodes(node_maps.col_map, options.kill_dofs, options.kill_cond);
  export_row_elements(element_maps.row_map, options.kill_dofs, options.kill_cond);
  export_column_elements(element_maps.col_map, options.kill_dofs, options.kill_cond);

  // these exports have set Filled()=false as all maps are invalid now
  if (options.fill_complete)
  {
    int err = fill_complete(*options.fill_complete);

    if (err) FOUR_C_THROW("fill_complete() returned err={}", err);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::FE::Discretization::extended_ghosting(const Core::LinAlg::Map& elecolmap,
    bool assigndegreesoffreedom, bool initelements, bool doboundaryconditions, bool checkghosting)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (filled())
  {
    const Core::LinAlg::Map* oldelecolmap = element_col_map();
    // check whether standard ghosting is included in extended ghosting
    for (int i = 0; i < oldelecolmap->num_my_elements(); ++i)
    {
      bool hasgid = elecolmap.my_gid(oldelecolmap->gid(i));
      if (!hasgid)
        FOUR_C_THROW("standard ghosting of ele {} is not included in extended ghosting",
            oldelecolmap->gid(i));
    }

    if (checkghosting)
    {
      int diff = elecolmap.num_global_elements() - oldelecolmap->num_global_elements();
      if (diff == 0 and Core::Communication::my_mpi_rank(get_comm()) == 0)
        FOUR_C_THROW("no additional elements have been ghosted");
    }
  }
#endif

  // first export the elements according to the processor local element column maps
  export_column_elements(elecolmap);

  // periodic boundary conditions require ghosting of all master and slave nodes,
  // if node of pbc set is contained in list of owned and ghosted elements
  // in case of pbcs, this has to be restored
  bool have_pbc = false;
  std::shared_ptr<Core::DOFSets::PBCDofSet> pbcdofset = nullptr;
  // map of master nodes and corresponding slave nodes
  std::map<int, std::set<int>> pbcmap;
  // create the inverse map --- slavenode -> masternode
  std::map<int, int> inversenodecoupling;
  // map to be filled with new (extended) master nodes and corresponding slave nodes (in col layout)
  std::map<int, std::set<int>> pbcmapnew;

  // check for pbcs
  for (int nds = 0; nds < num_dof_sets(); nds++)
  {
    pbcdofset = std::dynamic_pointer_cast<Core::DOFSets::PBCDofSet>(dofsets_[nds]);

    if (pbcdofset != nullptr)
    {
      have_pbc = true;
      // fill content of pbcmap int std::map<int, std::set<int> > in preparation for gather_all
      const std::map<int, std::vector<int>>* tmp = pbcdofset->get_coupled_nodes();
      for (auto& [gid, nodes] : *tmp) pbcmap[gid].insert(nodes.begin(), nodes.end());

      // it is assumed that, if one pbc set is available, all other potential dofsets hold the same
      // layout
      break;
    }
  }

  // if pbcs are available, get master and slave information
  if (have_pbc)
  {
    // communicate all master and slave pairs
    // caution: we build redundant maps here, containing all master nodes
    Core::LinAlg::gather_all(pbcmap, comm_);

    // and build slave master pairs
    for (std::map<int, std::set<int>>::iterator curr = pbcmap.begin(); curr != pbcmap.end(); ++curr)
      for (std::set<int>::const_iterator it = curr->second.begin(); it != curr->second.end(); ++it)
        inversenodecoupling[*it] = curr->first;
  }

  // get the node ids of the elements that have to be ghosted and create a proper node column map
  // for their export
  std::set<int> nodes;
  for (int lid = 0; lid < elecolmap.num_my_elements(); ++lid)
  {
    Core::Elements::Element* ele = this->g_element(elecolmap.gid(lid));
    const int* nodeids = ele->node_ids();
    for (int inode = 0; inode < ele->num_node(); ++inode)
    {
      nodes.insert(nodeids[inode]);

      // for pbcs, take into account all master and slave pairs
      if (have_pbc)
      {
        // is present node a master node?
        std::map<int, std::set<int>>::iterator foundmaster = pbcmap.find(nodeids[inode]);

        if (foundmaster != pbcmap.end())
        {
          // also store all corresponding slave nodes in set of col nodes
          nodes.insert(foundmaster->second.begin(), foundmaster->second.end());

          // add master and corresponding slaves to new col list of master and slave pairs
          pbcmapnew[foundmaster->first] = foundmaster->second;
        }
        else
        {
          // is present node a slave node?
          std::map<int, int>::iterator foundslave = inversenodecoupling.find(nodeids[inode]);

          if (foundslave != inversenodecoupling.end())
          {
            // add corresponding master to set of col nodes
            nodes.insert(foundslave->second);

            // store also all further slave nodes of this master (if multiple pbcs are used)
            nodes.insert(pbcmap[foundslave->second].begin(), pbcmap[foundslave->second].end());

            // add master and corresponding slaves to new col list of master and slave pairs
            pbcmapnew[foundslave->second] = pbcmap[foundslave->second];
          }
        }
      }
    }
  }

  // copy data from std::set<int> to std::vector<int>
  std::shared_ptr<std::map<int, std::vector<int>>> pbcmapvec =
      std::make_shared<std::map<int, std::vector<int>>>();
  for (std::map<int, std::set<int>>::const_iterator it = pbcmapnew.begin(); it != pbcmapnew.end();
      ++it)
    std::copy(it->second.begin(), it->second.end(), std::back_inserter((*pbcmapvec)[it->first]));

  // transfer master and slave information to pbc dofset
  if (have_pbc) pbcdofset->set_coupled_nodes(pbcmapvec);

  std::vector<int> colnodes(nodes.begin(), nodes.end());
  Core::LinAlg::Map nodecolmap(-1, (int)colnodes.size(), colnodes.data(), 0, get_comm());

  // now ghost the nodes
  export_column_nodes(nodecolmap);

  // these exports have set Filled()=false as all maps are invalid now
  int err = fill_complete({
      .assign_degrees_of_freedom = assigndegreesoffreedom,
      .init_elements = initelements,
      .do_boundary_conditions = doboundaryconditions,
  });
  if (err) FOUR_C_THROW("fill_complete() threw error code {}", err);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::FE::Discretization::setup_ghosting(OptionsFillComplete options)
{
  if (filled())
    FOUR_C_THROW(
        "there is really no need to setup ghosting if the discretization is already filled");

  // build the graph ourselves
  std::map<int, std::set<int>> localgraph;
  for (std::map<int, std::shared_ptr<Core::Elements::Element>>::iterator i = element_.begin();
      i != element_.end(); ++i)
  {
    int numnodes = i->second->num_node();
    const int* nodes = i->second->node_ids();

    // loop nodes and add this topology to the row in the graph of every node
    for (int n = 0; n < numnodes; ++n)
    {
      int nodelid = nodes[n];
      copy(nodes, nodes + numnodes, inserter(localgraph[nodelid], localgraph[nodelid].begin()));
    }
  }

  // Create node row map. Only the row nodes go there.

  std::vector<int> gids;
  std::vector<int> entriesperrow;

  gids.reserve(localgraph.size());
  entriesperrow.reserve(localgraph.size());

  for (std::map<int, std::shared_ptr<Core::Nodes::Node>>::iterator i = node_.begin();
      i != node_.end(); ++i)
  {
    gids.push_back(i->first);
    entriesperrow.push_back(localgraph[i->first].size());
  }

  Core::LinAlg::Map rownodes(-1, gids.size(), gids.data(), 0, comm_);

  // Construct FE graph. This graph allows processor off-rows to be inserted
  // as well. The communication issue is solved.

  auto graph = std::make_shared<Core::LinAlg::Graph>(
      rownodes, entriesperrow.data(), Core::LinAlg::Graph::GraphType::FE_GRAPH);

  gids.clear();
  entriesperrow.clear();

  // Insert all rows into the graph, including the off ones.

  for (std::map<int, std::set<int>>::iterator i = localgraph.begin(); i != localgraph.end(); ++i)
  {
    std::set<int>& rowset = i->second;
    auto row = std::vector(rowset.begin(), rowset.end());
    rowset.clear();

    auto indices = std::span(row.data(), row.size());
    graph->insert_global_indices(i->first, indices);
  }

  localgraph.clear();

  // Finalize construction of this graph. Here the communication
  // happens. The ghosting problem is solved at this point.

  graph->fill_complete(rownodes, rownodes);

  // replace rownodes, colnodes with row and column maps from the graph
  const Core::LinAlg::Map& brow = graph->row_map();
  const Core::LinAlg::Map& bcol = graph->col_map();
  Core::LinAlg::Map noderowmap(
      brow.num_global_elements(), brow.num_my_elements(), brow.my_global_elements(), 0, comm_);
  Core::LinAlg::Map nodecolmap(
      bcol.num_global_elements(), bcol.num_my_elements(), bcol.my_global_elements(), 0, comm_);

  graph = nullptr;

  // Redistribute discretization to match the new maps.
  redistribute(
      {
          .row_map = noderowmap,
          .col_map = nodecolmap,
      },
      {.fill_complete = options});
}

FOUR_C_NAMESPACE_CLOSE

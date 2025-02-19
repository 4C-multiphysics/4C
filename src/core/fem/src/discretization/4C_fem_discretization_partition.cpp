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

#include <Epetra_FECrsGraph.h>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::FE::Discretization::export_row_nodes(
    const Epetra_Map& newmap, bool killdofs, bool killcond)
{
  // test whether newmap is non-overlapping
  if (!newmap.UniqueGIDs()) FOUR_C_THROW("new map not unique");

  // destroy all ghosted nodes
  const int myrank = Core::Communication::my_mpi_rank(get_comm());
  std::map<int, std::shared_ptr<Core::Nodes::Node>>::iterator curr;
  for (curr = node_.begin(); curr != node_.end();)
  {
    if (curr->second->owner() != myrank)
      node_.erase(curr++);
    else
      ++curr;
  }

  // build rowmap of nodes noderowmap_ if it does not exist
  if (noderowmap_ == nullptr) build_node_row_map();
  const Epetra_Map& oldmap = *noderowmap_;

  // create an exporter object that will figure out the communication pattern
  Core::Communication::Exporter exporter(oldmap, newmap, get_comm());

  // Do the communication
  exporter.do_export(node_);

  // update all ownership flags
  for (curr = node_.begin(); curr != node_.end(); ++curr) curr->second->set_owner(myrank);

  // maps and pointers are no longer correct and need rebuilding
  reset(killdofs, killcond);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::FE::Discretization::export_column_nodes(
    const Epetra_Map& newmap, bool killdofs, bool killcond)
{
  // destroy all ghosted nodes
  const int myrank = Core::Communication::my_mpi_rank(get_comm());
  std::map<int, std::shared_ptr<Core::Nodes::Node>>::iterator curr;
  for (curr = node_.begin(); curr != node_.end();)
  {
    if (curr->second->owner() != myrank)
      node_.erase(curr++);
    else
      ++curr;
  }

  // build rowmap of nodes noderowmap_ if it does not exist
  if (noderowmap_ == nullptr) build_node_row_map();
  const Epetra_Map& oldmap = *noderowmap_;

  // test whether all nodes in oldmap are also in newmap, otherwise
  // this would be a change of owner which is not allowed here
  for (int i = 0; i < oldmap.NumMyElements(); ++i)
  {
    int gid = oldmap.GID(i);
    if (!(newmap.MyGID(gid)))
      FOUR_C_THROW("Proc %d: Node gid=%d from oldmap is not in newmap", myrank, gid);
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
    Epetra_Map& target, std::vector<int>& gidlist)
{
  const int myrank = Core::Communication::my_mpi_rank(get_comm());

  // proc 0 looks for elements that are to be send to other procs
  int size = (int)gidlist.size();
  std::vector<int> pidlist(size);  // gids on proc 0
  int err = target.RemoteIDList(size, gidlist.data(), pidlist.data(), nullptr);
  if (err < 0) FOUR_C_THROW("Epetra_BlockMap::RemoteIDList returned err=%d", err);

  std::map<int, std::vector<char>> sendmap;  // proc to send a set of elements to
  if (!myrank)
  {
    std::map<int, Core::Communication::PackBuffer> sendpb;  // proc to send a set of elements to
    for (int i = 0; i < size; ++i)
    {
      if (pidlist[i] == myrank or pidlist[i] < 0) continue;  // do not send to myself
      Core::Elements::Element* actele = g_element(gidlist[i]);
      if (!actele) FOUR_C_THROW("Cannot find global element %d", gidlist[i]);
      actele->pack(sendpb[pidlist[i]]);
      element_.erase(actele->id());
    }
    for (std::map<int, Core::Communication::PackBuffer>::iterator fool = sendpb.begin();
        fool != sendpb.end(); ++fool)
      swap(sendmap[fool->first], fool->second());
  }

  // tell everybody who is to receive something
  std::vector<int> receivers;

  receivers.reserve(sendmap.size());
  for (auto& fool : sendmap) receivers.push_back(fool.first);

  size = (int)receivers.size();
  Core::Communication::broadcast(&size, 1, 0, get_comm());
  if (myrank != 0) receivers.resize(size);
  Core::Communication::broadcast(receivers.data(), size, 0, get_comm());
  int foundme = -1;
  if (myrank != 0)
    for (int i = 0; i < size; ++i)
      if (receivers[i] == myrank)
      {
        foundme = i;
        break;
      }

  // proc 0 sends out messages
  int tag = 0;
  Core::Communication::Exporter exporter(get_comm());
  std::vector<MPI_Request> request(size);
  if (!myrank)
  {
    for (std::map<int, std::vector<char>>::iterator fool = sendmap.begin(); fool != sendmap.end();
        ++fool)
    {
      exporter.i_send(
          0, fool->first, fool->second.data(), (int)fool->second.size(), tag, request[tag]);
      tag++;
    }
    if (tag != size) FOUR_C_THROW("Number of messages is mixed up");
    // do not delete sendmap until Wait has returned!
  }

  // all other procs listen to message and put element into dis
  if (foundme != -1)
  {
    std::vector<char> recvdata;
    int length = 0;
    int source = -1;
    int tag = -1;
    exporter.receive_any(source, tag, recvdata, length);
    if (source != 0 || tag != foundme) FOUR_C_THROW("Messages got mixed up");
    // Put received elements into discretization
    Communication::UnpackBuffer buffer(recvdata);
    while (!buffer.at_end())
    {
      Core::Communication::ParObject* object = Core::Communication::factory(buffer);
      Core::Elements::Element* ele = dynamic_cast<Core::Elements::Element*>(object);
      if (!ele) FOUR_C_THROW("Received object is not an element");
      ele->set_owner(myrank);
      std::shared_ptr<Core::Elements::Element> rcpele(ele);
      add_element(rcpele);
    }
  }

  // wait for all communication to finish
  if (!myrank)
    for (int i = 0; i < size; ++i) exporter.wait(request[i]);

  Core::Communication::barrier(get_comm());  // I feel better this way ;-)
  reset();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::FE::Discretization::proc_zero_distribute_nodes_to_all(Epetra_Map& target)
{
  const int myrank = Core::Communication::my_mpi_rank(get_comm());

  // proc 0 looks for nodes that are to be distributed
  reset();
  build_node_row_map();
  const Epetra_Map& oldmap = *noderowmap_;
  int size = oldmap.NumMyElements();
  if (myrank) size = 0;
  std::vector<int> pidlist(size, -1);
  {
    int err = target.RemoteIDList(size, oldmap.MyGlobalElements(), pidlist.data(), nullptr);
    if (err) FOUR_C_THROW("Epetra_BlockMap::RemoteIDLis returned err=%d", err);
  }

  std::map<int, std::vector<char>> sendmap;
  if (!myrank)
  {
    std::map<int, Core::Communication::PackBuffer> sendpb;
    for (int i = 0; i < size; ++i)
    {
      // proc 0 does not send to itself
      if (pidlist[i] == myrank || pidlist[i] == -1) continue;
      Core::Nodes::Node* node = g_node(oldmap.MyGlobalElements()[i]);
      if (!node) FOUR_C_THROW("Proc 0 cannot find global node %d", oldmap.MyGlobalElements()[i]);
      node->pack(sendpb[pidlist[i]]);
      node_.erase(node->id());
    }
    for (std::map<int, Core::Communication::PackBuffer>::iterator fool = sendpb.begin();
        fool != sendpb.end(); ++fool)
      swap(sendmap[fool->first], fool->second());
  }

  // tell everybody who is to receive something
  std::vector<int> receivers;
  for (std::map<int, std::vector<char>>::iterator fool = sendmap.begin(); fool != sendmap.end();
      ++fool)
    receivers.push_back(fool->first);
  size = (int)receivers.size();
  Core::Communication::broadcast(&size, 1, 0, get_comm());
  if (myrank != 0) receivers.resize(size);
  Core::Communication::broadcast(receivers.data(), size, 0, get_comm());
  int foundme = -1;
  if (myrank != 0)
    for (int i = 0; i < size; ++i)
      if (receivers[i] == myrank)
      {
        foundme = i;
        break;
      }

  // proc 0 sends out messages
  int tag = 0;
  Core::Communication::Exporter exporter(get_comm());
  std::vector<MPI_Request> request(size);
  if (!myrank)
  {
    for (std::map<int, std::vector<char>>::iterator fool = sendmap.begin(); fool != sendmap.end();
        ++fool)
    {
      exporter.i_send(
          0, fool->first, fool->second.data(), (int)fool->second.size(), tag, request[tag]);
      tag++;
    }
    if (tag != size) FOUR_C_THROW("Number of messages is mixed up");
    // do not delete sendmap until Wait has returned!
  }

  // all other procs listen to message and put node into dis
  if (foundme != -1)
  {
    std::vector<char> recvdata;
    int length = 0;
    int source = -1;
    int tag = -1;
    exporter.receive_any(source, tag, recvdata, length);
    // printf("Proc %d received tag %d length %d\n",myrank,tag,length); fflush(stdout);
    if (source != 0 || tag != foundme) FOUR_C_THROW("Messages got mixed up");
    // Put received nodes into discretization
    Communication::UnpackBuffer buffer(recvdata);
    while (!buffer.at_end())
    {
      Core::Communication::ParObject* object = Core::Communication::factory(buffer);
      Core::Nodes::Node* node = dynamic_cast<Core::Nodes::Node*>(object);
      if (!node) FOUR_C_THROW("Received object is not a node");
      node->set_owner(myrank);
      std::shared_ptr<Core::Nodes::Node> rcpnode(node);
      add_node(rcpnode);
    }
  }


  // wait for all communication to finish
  if (!myrank)
    for (int i = 0; i < size; ++i) exporter.wait(request[i]);

  Core::Communication::barrier(get_comm());  // feel better this way ;-)
  reset();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::FE::Discretization::export_row_elements(
    const Epetra_Map& newmap, bool killdofs, bool killcond)
{
  // test whether newmap is non-overlapping
  if (!newmap.UniqueGIDs()) FOUR_C_THROW("new map not unique");

  // destroy all ghosted elements
  const int myrank = Core::Communication::my_mpi_rank(get_comm());
  std::map<int, std::shared_ptr<Core::Elements::Element>>::iterator curr;
  for (curr = element_.begin(); curr != element_.end();)
  {
    if (curr->second->owner() != myrank)
      element_.erase(curr++);
    else
      ++curr;
  }

  // build map of elements elerowmap_ if it does not exist
  if (elerowmap_ == nullptr) build_element_row_map();
  const Epetra_Map& oldmap = *elerowmap_;

  // create an exporter object that will figure out the communication pattern
  Core::Communication::Exporter exporter(oldmap, newmap, get_comm());

  exporter.do_export(element_);

  // update ownerships and kick out everything that's not in newmap
  for (curr = element_.begin(); curr != element_.end(); ++curr) curr->second->set_owner(myrank);

  // maps and pointers are no longer correct and need rebuilding
  reset(killdofs, killcond);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::FE::Discretization::export_column_elements(
    const Epetra_Map& newmap, bool killdofs, bool killcond)
{
  // destroy all ghosted elements
  const int myrank = Core::Communication::my_mpi_rank(get_comm());
  std::map<int, std::shared_ptr<Core::Elements::Element>>::iterator curr;
  for (curr = element_.begin(); curr != element_.end();)
  {
    if (curr->second->owner() != myrank)
      element_.erase(curr++);
    else
      ++curr;
  }

  // build map of elements elerowmap_ if it does not exist
  if (elerowmap_ == nullptr) build_element_row_map();
  const Epetra_Map& oldmap = *elerowmap_;

  // test whether all elements in oldmap are also in newmap
  // Otherwise, this would be a change of owner which is not allowed here
  for (int i = 0; i < oldmap.NumMyElements(); ++i)
  {
    int gid = oldmap.GID(i);
    if (!(newmap.MyGID(gid)))
      FOUR_C_THROW("Proc %d: Element gid=%d from oldmap is not in newmap", myrank, gid);
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
  const Epetra_Map* noderowmap = node_row_map();

  // allocate graph
  std::shared_ptr<Core::LinAlg::Graph> graph =
      std::make_shared<Core::LinAlg::Graph>(Copy, *noderowmap, 108, false);

  // iterate all elements on this proc including ghosted ones
  // Note:
  // if a proc stores the appropriate ghosted elements, the resulting
  // graph will be the correct and complete graph of the distributed
  // discretization even if nodes are not ghosted.
  std::map<int, std::shared_ptr<Core::Elements::Element>>::const_iterator curr;
  for (curr = element_.begin(); curr != element_.end(); ++curr)
  {
    const int nnode = curr->second->num_node();
    const int* nodeids = curr->second->node_ids();
    for (int row = 0; row < nnode; ++row)
    {
      const int rownode = nodeids[row];
      if (!noderowmap->MyGID(rownode)) continue;
      for (int col = 0; col < nnode; ++col)
      {
        int colnode = nodeids[col];
        int err = graph->InsertGlobalIndices(rownode, 1, &colnode);
        if (err < 0) FOUR_C_THROW("graph->InsertGlobalIndices returned err=%d", err);
      }
    }
  }
  int err = graph->FillComplete();
  if (err) FOUR_C_THROW("graph->FillComplete() returned err=%d", err);
  err = graph->OptimizeStorage();
  if (err) FOUR_C_THROW("graph->OptimizeStorage() returned err=%d", err);

  return graph;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::MultiVector<double>> Core::FE::Discretization::build_node_coordinates(
    std::shared_ptr<const Epetra_Map> noderowmap) const
{
  // get nodal row map if not given
  if (noderowmap == nullptr)
    noderowmap = Core::Utils::shared_ptr_from_ref<const Epetra_Map>(*node_row_map());

  std::shared_ptr<Core::LinAlg::MultiVector<double>> coordinates =
      std::make_shared<Core::LinAlg::MultiVector<double>>(*noderowmap, 3, true);

  for (int lid = 0; lid < noderowmap->NumMyElements(); ++lid)
  {
    if (!node_.count(noderowmap->GID(lid))) continue;
    for (int dim = 0; dim < 3; ++dim)
      coordinates->ReplaceMyValue(lid, dim, node_.at(noderowmap->GID(lid))->x()[dim]);
  }

  return coordinates;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::pair<std::shared_ptr<Epetra_Map>, std::shared_ptr<Epetra_Map>>
Core::FE::Discretization::build_element_row_column(
    const Epetra_Map& noderowmap, const Epetra_Map& nodecolmap, bool do_extended_ghosting) const
{
  const int myrank = Core::Communication::my_mpi_rank(get_comm());
  const int numproc = Core::Communication::num_mpi_ranks(get_comm());

  // note:
  // - noderowmap need not match distribution of nodes in this
  //   discretization at all.
  // - noderowmap is a non-overlapping map, that's tested
  if (!noderowmap.UniqueGIDs()) FOUR_C_THROW("noderowmap is not a unique map");

  // find all owners for the overlapping node map
  const int ncnode = nodecolmap.NumMyElements();
  std::vector<int> cnodeowner(ncnode);
  int err =
      noderowmap.RemoteIDList(ncnode, nodecolmap.MyGlobalElements(), cnodeowner.data(), nullptr);
  if (err) FOUR_C_THROW("Epetra_BlockMap::RemoteIDLis returned err=%d", err);

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
  int nummynodes = noderowmap.NumMyElements();
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
        if (noderowmap.MyGID(nodeids[j])) ++nummine;

      // Check if I own nodes of this element
      if (!nummine)
      {
        // If the rebalance type is monolithic, activate additional ghosting
        if (do_extended_ghosting)
        {
          // If all nodes of the element are in col map we still ghost it
          bool all_nodes_in_col = true;
          for (int j = 0; j < numnode; ++j)
            if (!nodecolmap.MyGID(nodeids[j])) all_nodes_in_col = false;

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
        if (!nodecolmap.MyGID(nodeids[j]))
          FOUR_C_THROW("I do not have own/ghosted node gid=%d", nodeids[j]);

      // find out who owns how many of the nodes
      std::vector<int> nodeowner(numnode);
      std::vector<int> numperproc(numproc);
      for (int j = 0; j < numproc; ++j) numperproc[j] = 0;
      for (int j = 0; j < numnode; ++j)
      {
        const int lid = nodecolmap.LID(nodeids[j]);
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
      int minrownodes = noderowmap.NumGlobalElements();
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
  std::shared_ptr<Epetra_Map> elerowmap = std::make_shared<Epetra_Map>(
      -1, nummyele, myele.data(), 0, Core::Communication::as_epetra_comm(get_comm()));
  if (!elerowmap->UniqueGIDs()) FOUR_C_THROW("Element row map is not unique");

  // build elecolmap
  std::vector<int> elecol(nummyele + nummyghostele);
  for (int i = 0; i < nummyele; ++i) elecol[i] = myele[i];
  for (int i = 0; i < nummyghostele; ++i) elecol[nummyele + i] = myghostele[i];
  std::shared_ptr<Epetra_Map> elecolmap = std::make_shared<Epetra_Map>(-1, nummyghostele + nummyele,
      elecol.data(), 0, Core::Communication::as_epetra_comm(get_comm()));

  return {elerowmap, elecolmap};
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::FE::Discretization::redistribute(const Epetra_Map& noderowmap,
    const Epetra_Map& nodecolmap, OptionsRedistribution options_redistribution)
{
  // build the overlapping and non-overlapping element maps
  const auto& [elerowmap, elecolmap] =
      build_element_row_column(noderowmap, nodecolmap, options_redistribution.do_extended_ghosting);

  // export nodes and elements to the new maps
  export_row_nodes(noderowmap, options_redistribution.kill_dofs, options_redistribution.kill_cond);
  export_column_nodes(
      nodecolmap, options_redistribution.kill_dofs, options_redistribution.kill_cond);
  export_row_elements(
      *elerowmap, options_redistribution.kill_dofs, options_redistribution.kill_cond);
  export_column_elements(
      *elecolmap, options_redistribution.kill_dofs, options_redistribution.kill_cond);

  // these exports have set Filled()=false as all maps are invalid now
  int err = fill_complete(options_redistribution.assign_degrees_of_freedom,
      options_redistribution.init_elements, options_redistribution.do_boundary_conditions);

  if (err) FOUR_C_THROW("fill_complete() returned err=%d", err);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::FE::Discretization::redistribute(const Epetra_Map& noderowmap,
    const Epetra_Map& nodecolmap, const Epetra_Map& elerowmap, const Epetra_Map& elecolmap,
    bool assigndegreesoffreedom, bool initelements, bool doboundaryconditions, bool killdofs,
    bool killcond)
{
  // export nodes and elements to the new maps
  export_row_nodes(noderowmap, killdofs, killcond);
  export_column_nodes(nodecolmap, killdofs, killcond);
  export_row_elements(elerowmap, killdofs, killcond);
  export_column_elements(elecolmap, killdofs, killcond);

  // these exports have set Filled()=false as all maps are invalid now
  int err = fill_complete(assigndegreesoffreedom, initelements, doboundaryconditions);

  if (err) FOUR_C_THROW("fill_complete() returned err=%d", err);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::FE::Discretization::extended_ghosting(const Epetra_Map& elecolmap,
    bool assigndegreesoffreedom, bool initelements, bool doboundaryconditions, bool checkghosting)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (filled())
  {
    const Epetra_Map* oldelecolmap = element_col_map();
    // check whether standard ghosting is included in extended ghosting
    for (int i = 0; i < oldelecolmap->NumMyElements(); ++i)
    {
      bool hasgid = elecolmap.MyGID(oldelecolmap->GID(i));
      if (!hasgid)
        FOUR_C_THROW("standard ghosting of ele %d is not included in extended ghosting",
            oldelecolmap->GID(i));
    }

    if (checkghosting)
    {
      int diff = elecolmap.NumGlobalElements() - oldelecolmap->NumGlobalElements();
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
      std::map<int, std::vector<int>>* tmp = pbcdofset->get_coupled_nodes();
      for (std::map<int, std::vector<int>>::const_iterator it = tmp->begin(); it != tmp->end();
          ++it)
        pbcmap[it->first].insert(it->second.begin(), it->second.end());

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
  for (int lid = 0; lid < elecolmap.NumMyElements(); ++lid)
  {
    Core::Elements::Element* ele = this->g_element(elecolmap.GID(lid));
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
  Epetra_Map nodecolmap(-1, (int)colnodes.size(), colnodes.data(), 0,
      Core::Communication::as_epetra_comm(get_comm()));

  // now ghost the nodes
  export_column_nodes(nodecolmap);

  // these exports have set Filled()=false as all maps are invalid now
  int err = fill_complete(assigndegreesoffreedom, initelements, doboundaryconditions);
  if (err) FOUR_C_THROW("fill_complete() threw error code %d", err);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::FE::Discretization::setup_ghosting(
    bool assigndegreesoffreedom, bool initelements, bool doboundaryconditions)
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

  Epetra_Map rownodes(-1, gids.size(), gids.data(), 0, Core::Communication::as_epetra_comm(comm_));

  // Construct FE graph. This graph allows processor off-rows to be inserted
  // as well. The communication issue is solved.

  std::shared_ptr<Epetra_FECrsGraph> graph =
      std::make_shared<Epetra_FECrsGraph>(Copy, rownodes, entriesperrow.data(), false);

  gids.clear();
  entriesperrow.clear();

  // Insert all rows into the graph, including the off ones.

  for (std::map<int, std::set<int>>::iterator i = localgraph.begin(); i != localgraph.end(); ++i)
  {
    std::set<int>& rowset = i->second;
    std::vector<int> row;
    row.reserve(rowset.size());
    row.assign(rowset.begin(), rowset.end());
    rowset.clear();

    int err = graph->InsertGlobalIndices(1, &i->first, row.size(), row.data());
    if (err < 0) FOUR_C_THROW("graph->InsertGlobalIndices returned %d", err);
  }

  localgraph.clear();

  // Finalize construction of this graph. Here the communication
  // happens. The ghosting problem is solved at this point.

  int err = graph->GlobalAssemble(rownodes, rownodes);
  if (err) FOUR_C_THROW("graph->GlobalAssemble returned %d", err);

  // replace rownodes, colnodes with row and column maps from the graph
  // do stupid conversion from Epetra_BlockMap to Epetra_Map
  const Epetra_BlockMap& brow = graph->RowMap();
  const Epetra_BlockMap& bcol = graph->ColMap();
  Epetra_Map noderowmap(brow.NumGlobalElements(), brow.NumMyElements(), brow.MyGlobalElements(), 0,
      Core::Communication::as_epetra_comm(comm_));
  Epetra_Map nodecolmap(bcol.NumGlobalElements(), bcol.NumMyElements(), bcol.MyGlobalElements(), 0,
      Core::Communication::as_epetra_comm(comm_));

  graph = nullptr;

  // Redistribute discretization to match the new maps.
  redistribute(noderowmap, nodecolmap,
      {.assign_degrees_of_freedom = assigndegreesoffreedom,
          .init_elements = initelements,
          .do_boundary_conditions = doboundaryconditions});
}

FOUR_C_NAMESPACE_CLOSE

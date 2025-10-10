// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_dofset_transparent.hpp"

#include "4C_fem_condition.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"

FOUR_C_NAMESPACE_OPEN

Core::DOFSets::TransparentDofSet::TransparentDofSet(
    std::shared_ptr<Core::FE::Discretization> sourcedis, bool parallel)
    : Core::DOFSets::DofSet(), sourcedis_(sourcedis), parallel_(parallel)
{
  return;
}

int Core::DOFSets::TransparentDofSet::assign_degrees_of_freedom(
    const Core::FE::Discretization& dis, const unsigned dspos, const int start)
{
  // first, we call the standard assign_degrees_of_freedom from the base class
  int count = DofSet::assign_degrees_of_freedom(dis, dspos, start);
  if (pccdofhandling_)
    FOUR_C_THROW("ERROR: Point coupling cinditions not yet implemented for TransparentDofSet");

  if (!parallel_)
  {
    transfer_degrees_of_freedom(*sourcedis_, dis, start);
  }
  else
  {
    parallel_transfer_degrees_of_freedom(*sourcedis_, dis, start);
  }

  // tell all proxies (again!)
  notify_assigned();

  return count;
}

/// Assign dof numbers for new discretization using dof numbering from source discretization.
void Core::DOFSets::TransparentDofSet::transfer_degrees_of_freedom(
    const Core::FE::Discretization& sourcedis, const Core::FE::Discretization& newdis,
    const int start)
{
  if (!sourcedis.dof_row_map()->unique_gids()) FOUR_C_THROW("dof_row_map is not unique");
  if (!sourcedis.node_row_map()->unique_gids()) FOUR_C_THROW("NodeRowMap is not unique");
  if (!sourcedis.element_row_map()->unique_gids()) FOUR_C_THROW("ElementRowMap is not unique");

  if (!newdis.dof_row_map()->unique_gids()) FOUR_C_THROW("dof_row_map is not unique");
  if (!newdis.node_row_map()->unique_gids()) FOUR_C_THROW("NodeRowMap is not unique");
  if (!newdis.element_row_map()->unique_gids()) FOUR_C_THROW("ElementRowMap is not unique");

  // build dofrowmap
  std::set<int> dofrowset;
  std::vector<int> dofrowvec;
  dofrowvec.reserve(dofrowmap_->num_my_elements());
  for (int inode = 0; inode != newdis.num_my_row_nodes(); ++inode)
  {
    const Core::Nodes::Node* newnode = newdis.l_row_node(inode);
    const Core::Nodes::Node* sourcenode = sourcedis.g_node(newnode->id());

    const std::vector<int> dofs = sourcedis.dof(0, sourcenode);

    const int newlid = newnode->lid();
    const int numdofs = (numdfcolnodes_->get_local_values())[newlid];
    if (numdofs > 0)
    {
      (idxcolnodes_->get_local_values())[newlid] = dofs[0];
      for (int idof = 0; idof < numdofs; ++idof)
      {
        dofrowset.insert(dofs[idof]);
      }
    }
  }

  for (std::set<int>::iterator idof = dofrowset.begin(); idof != dofrowset.end(); ++idof)
  {
    dofrowvec.push_back(*idof);
  }

  dofrowmap_ = std::make_shared<Core::LinAlg::Map>(
      -1, dofrowvec.size(), dofrowvec.data(), 0, newdis.get_comm());

  // build dofcolvec
  std::set<int> dofcolset;
  std::vector<int> dofcolvec;
  dofcolvec.reserve(dofcolmap_->num_my_elements());
  for (int inode = 0; inode != newdis.num_my_col_nodes(); ++inode)
  {
    const Core::Nodes::Node* newnode = newdis.l_col_node(inode);
    const Core::Nodes::Node* sourcenode = sourcedis.g_node(newnode->id());

    const int lid = sourcenode->lid();
    if (lid == -1)
    {
      FOUR_C_THROW("required node {} not on proc", newnode->id());
    }
    const std::vector<int> dofs = sourcedis.dof(0, sourcenode);
    const int newlid = newnode->lid();
    const int numdofs = (numdfcolnodes_->get_local_values())[newlid];
    if (numdofs > 0)
    {
      (idxcolnodes_->get_local_values())[newlid] = dofs[0];

      for (int idof = 0; idof < numdofs; ++idof)
      {
        dofcolset.insert(dofs[idof]);
      }
    }
  }

  for (std::set<int>::iterator idof = dofcolset.begin(); idof != dofcolset.end(); ++idof)
  {
    dofcolvec.push_back(*idof);
  }

  dofcolmap_ = std::make_shared<Core::LinAlg::Map>(
      -1, dofcolvec.size(), dofcolvec.data(), 0, newdis.get_comm());
}

/// Assign dof numbers for new discretization using dof numbering from source discretization.
void Core::DOFSets::TransparentDofSet::parallel_transfer_degrees_of_freedom(
    const Core::FE::Discretization& sourcedis, const Core::FE::Discretization& newdis,
    const int start)
{
  if (!sourcedis.dof_row_map()->unique_gids()) FOUR_C_THROW("dof_row_map is not unique");
  if (!sourcedis.node_row_map()->unique_gids()) FOUR_C_THROW("NodeRowMap is not unique");
  if (!sourcedis.element_row_map()->unique_gids()) FOUR_C_THROW("ElementRowMap is not unique");

  if (!newdis.dof_row_map()->unique_gids()) FOUR_C_THROW("dof_row_map is not unique");
  if (!newdis.node_row_map()->unique_gids()) FOUR_C_THROW("NodeRowMap is not unique");
  if (!newdis.element_row_map()->unique_gids()) FOUR_C_THROW("ElementRowMap is not unique");

  // list all my rownode ids

  //
  // we need a mapping
  //
  // colnode gid -> std::vector<int> dofs of sourcenode
  //
  // problem: sourcenode not necessarily on this proc -> communicate
  //
  // the idea is to search for the sourcerownode on some proc and to get
  // this unique number
  //
  std::map<int, std::vector<int>> gid_to_dofs;

  for (int inode = 0; inode != newdis.num_my_col_nodes(); ++inode)
  {
    const Core::Nodes::Node* newnode = newdis.l_col_node(inode);
    int gid = newnode->id();
    std::vector<int> emptyvec;
    gid_to_dofs.insert(std::pair<int, std::vector<int>>(gid, emptyvec));
  }

  {
    // create an exporter for point to point communication
    Core::Communication::Exporter exporter(sourcedis.get_comm());

    // necessary variables
    MPI_Request request;

    // define send and receive blocks
    std::vector<char> sblock;
    std::vector<char> rblock;

    // get number of processors and the current processors id
    int numproc = Core::Communication::num_mpi_ranks(sourcedis.get_comm());
    int myrank = Core::Communication::my_mpi_rank(sourcedis.get_comm());

    //----------------------------------------------------------------------
    // communication is done in a round robin loop
    //----------------------------------------------------------------------
    for (int np = 0; np < numproc + 1; ++np)
    {
      // in the first step, we cannot receive anything
      if (np > 0)
      {
        receive_block(numproc, myrank, rblock, exporter, request);

        // Unpack info from the receive block from the last proc
        unpack_local_source_dofs(gid_to_dofs, rblock);
      }

      // in the last step, we keep everything on this proc
      if (np < numproc)
      {
        // -----------------------
        // do what we wanted to do
        set_source_dofs_available_on_this_proc(gid_to_dofs);

        // Pack info into block to send
        Core::Communication::PackBuffer data;
        pack_local_source_dofs(gid_to_dofs, data);
        gid_to_dofs.clear();
        swap(sblock, data());

        send_block(numproc, myrank, sblock, exporter, request);
      }
    }
  }

  std::set<int> slaveset;
  std::vector<const Core::Conditions::Condition*> mypbcs;

  // get periodic surface boundary conditions
  sourcedis_->get_condition("SurfacePeriodic", mypbcs);

  if (mypbcs.empty())
  {
    sourcedis_->get_condition("LinePeriodic", mypbcs);
  }

  for (unsigned numcond = 0; numcond < mypbcs.size(); ++numcond)
  {
    const Core::Conditions::Condition* thiscond = mypbcs[numcond];

    // see whether we have a slave condition
    const std::string& mymasterslavetoggle =
        thiscond->parameters().get<std::string>("MASTER_OR_SLAVE");

    if (!(mymasterslavetoggle == "Master"))
    {
      const std::vector<int>* pbcids;
      pbcids = (*thiscond).get_nodes();

      for (std::vector<int>::const_iterator iter = pbcids->begin(); iter != pbcids->end(); ++iter)
      {
        slaveset.insert(*iter);
      }
    }
  }

  // build dofrowmap
  std::set<int> dofrowset;
  std::vector<int> dofrowvec;
  dofrowvec.reserve(dofrowmap_->num_my_elements());
  for (int inode = 0; inode != newdis.num_my_row_nodes(); ++inode)
  {
    const Core::Nodes::Node* newnode = newdis.l_row_node(inode);

    const std::vector<int> dofs = gid_to_dofs[newnode->id()];

    const int newlid = newnode->lid();
    const int numdofs = (numdfcolnodes_->get_local_values())[newlid];

    if (numdofs != (int)dofs.size())
    {
      printf("This is node %d  (%12.5e,%12.5e,%12.5e)\n", newnode->id(), newnode->x()[0],
          newnode->x()[1], newnode->x()[2]);

      FOUR_C_THROW("spooky, isn't it? dofs to overwrite {} != {} dofs.size() to set \n", numdofs,
          dofs.size());
    }

    if (numdofs > 0)
    {
      idxcolnodes_->get_local_values()[newlid] = dofs[0];

      // slave-dofs must not enter the dofrowset (if master&slave are on different procs)
      std::set<int>::iterator curr = slaveset.find(newnode->id());

      if (curr == slaveset.end())
      {
        for (int idof = 0; idof < numdofs; ++idof)
        {
          dofrowset.insert(dofs[idof]);
        }
      }
    }
  }

  for (std::set<int>::iterator idof = dofrowset.begin(); idof != dofrowset.end(); ++idof)
  {
    dofrowvec.push_back(*idof);
  }

  dofrowmap_ = std::make_shared<Core::LinAlg::Map>(
      -1, dofrowvec.size(), dofrowvec.data(), 0, newdis.get_comm());

  // build dofcolvec
  std::set<int> dofcolset;
  std::vector<int> dofcolvec;
  dofcolvec.reserve(dofcolmap_->num_my_elements());
  for (int inode = 0; inode != newdis.num_my_col_nodes(); ++inode)
  {
    const Core::Nodes::Node* newnode = newdis.l_col_node(inode);

    const std::vector<int> dofs = gid_to_dofs[newnode->id()];

    const int newlid = newnode->lid();
    // const int newfirstidx = (*idxcolnodes_)[newlid];
    const int numdofs = (numdfcolnodes_->get_local_values())[newlid];
    if (numdofs > 0)
    {
      idxcolnodes_->get_local_values()[newlid] = dofs[0];
      if (numdofs != (int)dofs.size())
        FOUR_C_THROW("numdofs {}!={} for node {}", numdofs, dofs.size(), newnode->id());

      for (int idof = 0; idof < numdofs; ++idof)
      {
        dofcolset.insert(dofs[idof]);
      }
    }
  }

  for (std::set<int>::iterator idof = dofcolset.begin(); idof != dofcolset.end(); ++idof)
  {
    dofcolvec.push_back(*idof);
  }

  dofcolmap_ = std::make_shared<Core::LinAlg::Map>(
      -1, dofcolvec.size(), dofcolvec.data(), 0, newdis.get_comm());


  return;
}


//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 | write dof info into map                                    (private) |
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void Core::DOFSets::TransparentDofSet::set_source_dofs_available_on_this_proc(
    std::map<int, std::vector<int>>& gid_to_dofs)
{
  for (std::map<int, std::vector<int>>::iterator curr = gid_to_dofs.begin();
      curr != gid_to_dofs.end(); ++curr)
  {
    const int lid = sourcedis_->node_row_map()->lid(curr->first);

    if (lid > -1)
    {
      curr->second.clear();

      const Core::Nodes::Node* sourcenode = sourcedis_->g_node(curr->first);

      const std::vector<int> dofs = sourcedis_->dof(0, sourcenode);

      for (std::vector<int>::const_iterator iter = dofs.begin(); iter != dofs.end(); ++iter)
      {
        curr->second.push_back(*iter);
      }
    }
    else
    {
      int numproc = Core::Communication::num_mpi_ranks(sourcedis_->get_comm());
      if (numproc == 1)
      {
        FOUR_C_THROW(
            "I have a one-processor problem but the node is not on the proc. "
            "sourcedis_->NodeRowMap() is probably corrupted.");
      }
    }
  }
  return;
}

//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 | pack all values into a send block                          (private) |
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void Core::DOFSets::TransparentDofSet::pack_local_source_dofs(
    std::map<int, std::vector<int>>& gid_to_dofs, Core::Communication::PackBuffer& sblock)
{
  int size = gid_to_dofs.size();

  // add size  to sendblock
  add_to_pack(sblock, size);

  for (std::map<int, std::vector<int>>::iterator curr = gid_to_dofs.begin();
      curr != gid_to_dofs.end(); ++curr)
  {
    int gid = curr->first;
    std::vector<int> mydofs = curr->second;
    int numdofs = (int)mydofs.size();

    add_to_pack(sblock, gid);
    add_to_pack(sblock, numdofs);
    for (int ll = 0; ll < numdofs; ++ll)
    {
      add_to_pack(sblock, mydofs[ll]);
    }
  }

  return;
}


//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 | unpack all values contained in receive block               (private) |
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void Core::DOFSets::TransparentDofSet::unpack_local_source_dofs(
    std::map<int, std::vector<int>>& gid_to_dofs, std::vector<char>& rblock)
{
  gid_to_dofs.clear();

  // extract size
  int size = 0;
  Communication::UnpackBuffer buffer(rblock);
  extract_from_pack(buffer, size);

  for (int rr = 0; rr < size; ++rr)
  {
    int gid = -1;
    std::vector<int> mydofs;
    int numdofs = 0;

    extract_from_pack(buffer, gid);
    extract_from_pack(buffer, numdofs);

    for (int ll = 0; ll < numdofs; ++ll)
    {
      int thisdof = 0;

      extract_from_pack(buffer, thisdof);
      mydofs.push_back(thisdof);
    }

    gid_to_dofs.insert(std::pair<int, std::vector<int>>(gid, mydofs));
  }

  rblock.clear();
  return;
}

//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 | receive a block in the round robin communication pattern   (private) |
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void Core::DOFSets::TransparentDofSet::receive_block(int numproc, int myrank,
    std::vector<char>& rblock, Core::Communication::Exporter& exporter, MPI_Request& request)
{
  // necessary variables

  int length = -1;
  int frompid = (myrank + numproc - 1) % numproc;
  int tag = frompid;

  // make sure that you do not think you received something if
  // you didn't
  if (rblock.empty() == false)
  {
    FOUR_C_THROW("rblock not empty");
  }

  // receive from predecessor
  exporter.receive_any(frompid, tag, rblock, length);

  if (tag != (myrank + numproc - 1) % numproc)
  {
    FOUR_C_THROW("received wrong message (ReceiveAny)");
  }

  exporter.wait(request);

  // for safety
  Core::Communication::barrier(exporter.get_comm());

  return;
}  // receive_block


//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 | send a block in the round robin communication pattern      (private) |
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void Core::DOFSets::TransparentDofSet::send_block(int numproc, int myrank,
    std::vector<char>& sblock, Core::Communication::Exporter& exporter, MPI_Request& request)
{
  // Send block to next proc.
  int tag = myrank;
  int frompid = myrank;
  int topid = (myrank + 1) % numproc;

  exporter.i_send(frompid, topid, sblock.data(), sblock.size(), tag, request);


  // for safety
  Core::Communication::barrier(exporter.get_comm());

  return;
}  // send_block

FOUR_C_NAMESPACE_CLOSE

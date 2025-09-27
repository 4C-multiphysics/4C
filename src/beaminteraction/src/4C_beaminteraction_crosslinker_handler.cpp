// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_crosslinker_handler.hpp"

#include "4C_binstrategy_meshfree_multibin.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_io.hpp"
#include "4C_io_pstream.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"

#include <Teuchos_TimeMonitor.hpp>

#include <unordered_set>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
BeamInteraction::BeamCrosslinkerHandler::BeamCrosslinkerHandler()
    : binstrategy_(nullptr), myrank_(-1), bincolmap_(nullptr)
{
  // empty constructor
}

void BeamInteraction::BeamCrosslinkerHandler::init(
    int myrank, std::shared_ptr<Core::Binstrategy::BinningStrategy> binstrategy)
{
  binstrategy_ = binstrategy;
  myrank_ = myrank;
}

void BeamInteraction::BeamCrosslinkerHandler::setup()
{
  // so far nothing to do
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void BeamInteraction::BeamCrosslinkerHandler::distribute_linker_to_bins(
    std::shared_ptr<Core::LinAlg::Map> const& linkerrowmap)
{
  std::list<std::shared_ptr<Core::Nodes::Node>> homelesslinker;
  for (int lid = 0; lid < linkerrowmap->num_my_elements(); ++lid)
  {
    Core::Nodes::Node* node = binstrategy_->bin_discret()->g_node(linkerrowmap->gid(lid));
    const double* currpos = node->x().data();
    place_node_correctly(Core::Utils::shared_ptr_from_ref(*node), currpos, homelesslinker);
  }

  // start round robin loop to fill linker into their correct bins
  fill_linker_into_bins_round_robin(homelesslinker);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void BeamInteraction::BeamCrosslinkerHandler::remove_all_linker()
{
  // 1st) loop over bins and remove initial linker info
  const int numrowbin = bin_strategy()->bin_discret()->num_my_col_elements();
  for (int ibin = 0; ibin < numrowbin; ++ibin)
  {
    Core::Elements::Element* actele = bin_strategy()->bin_discret()->l_col_element(ibin);
    dynamic_cast<Core::FE::MeshFree::MeshfreeMultiBin*>(actele)->delete_nodes();
  }

  // 2nd) initial linker need to be removed from bindis_
  bin_strategy()->bin_discret()->delete_nodes();
}

/*----------------------------------------------------------------------*
| fill linker into their correct bin on according proc   ghamm 09/12 |
 *----------------------------------------------------------------------*/
void BeamInteraction::BeamCrosslinkerHandler::fill_linker_into_bins_round_robin(
    std::list<std::shared_ptr<Core::Nodes::Node>>& homelesslinker)
{
  const int numproc = Core::Communication::num_mpi_ranks(binstrategy_->bin_discret()->get_comm());
  const int myrank =
      Core::Communication::my_mpi_rank(binstrategy_->bin_discret()->get_comm());  // me
  const int torank = (myrank + 1) % numproc;                                      // to
  const int fromrank = (myrank + numproc - 1) % numproc;                          // from

  Core::Communication::Exporter exporter(binstrategy_->bin_discret()->get_comm());

  for (int irobin = 0; irobin < numproc; ++irobin)
  {
    std::vector<char> sdata;
    std::vector<char> rdata;

    // ---- pack data for sending -----
    {
      Core::Communication::PackBuffer data;
      for (std::list<std::shared_ptr<Core::Nodes::Node>>::const_iterator currlinker =
               homelesslinker.begin();
          currlinker != homelesslinker.end(); ++currlinker)
      {
        (*currlinker)->pack(data);
        binstrategy_->bin_discret()->delete_node((*currlinker)->id());
      }
      std::swap(sdata, data());
    }


    // ---- send ----
    MPI_Request request;
    exporter.i_send(myrank, torank, sdata.data(), (int)sdata.size(), 1234, request);


    // ---- receive ----
    int length = rdata.size();
    int tag = -1;
    int from = -1;
    exporter.receive_any(from, tag, rdata, length);
    if (tag != 1234 or from != fromrank)
      FOUR_C_THROW("Received data from the wrong proc soll({} -> {}) is({} -> {})", fromrank,
          myrank, from, myrank);


    // ---- unpack ----
    {
      // Put received nodes either into discretization or into list of homeless linker
      homelesslinker.clear();
      Core::Communication::UnpackBuffer buffer(rdata);
      while (!buffer.at_end())
      {
        auto object =
            std::shared_ptr<Core::Communication::ParObject>(Core::Communication::factory(buffer));
        std::shared_ptr<Core::Nodes::Node> node =
            std::dynamic_pointer_cast<Core::Nodes::Node>(object);
        if (node == nullptr) FOUR_C_THROW("Received object is not a node");

        // process received linker
        const double* currpos = node->x().data();
        place_node_correctly(node, currpos, homelesslinker);
      }
    }


    // wait for all communication to finish
    exporter.wait(request);
    Core::Communication::barrier(
        binstrategy_->bin_discret()->get_comm());  // I feel better this way ;-)
  }  // end for irobin

  if (homelesslinker.size())
  {
    std::cout << " There are " << homelesslinker.size()
              << " linker which have left the computational domain on rank " << myrank << std::endl;
    // erase everything that is left
    homelesslinker.clear();
  }

  return;
}

/*----------------------------------------------------------------------*
| fill linker into their correct bin on according proc   ghamm 03/16 |
 *----------------------------------------------------------------------*/
std::shared_ptr<std::list<int>>
BeamInteraction::BeamCrosslinkerHandler::fill_linker_into_bins_remote_id_list(
    std::list<std::shared_ptr<Core::Nodes::Node>>& homelesslinker)
{
  TEUCHOS_FUNC_TIME_MONITOR(
      "BeamInteraction::beam_crosslinker_handler::fill_linker_into_bins_remote_id_list");
  const int numproc = Core::Communication::num_mpi_ranks(binstrategy_->bin_discret()->get_comm());
  std::shared_ptr<std::list<int>> removedlinker = std::make_shared<std::list<int>>(0);

  // parallel case
  // ---- find new host procs for linker -----
  const int fullsize = (int)homelesslinker.size();
  std::vector<int> targetbinIdlist;
  targetbinIdlist.reserve(fullsize);
  std::list<std::shared_ptr<Core::Nodes::Node>>::const_iterator hlp;
  for (hlp = homelesslinker.begin(); hlp != homelesslinker.end(); ++hlp)
  {
    const int binId = binstrategy_->convert_pos_to_gid((*hlp)->x().data());
    targetbinIdlist.push_back(binId);
  }

  // get proc which will be the future host of homeless linker
  std::vector<int> pidlist(fullsize);
  {
    // only unique id lists are accepted in RemoteIDList
    // 1) make gid list unique
    std::set<int> unique_targetbinIdlist(targetbinIdlist.begin(), targetbinIdlist.end());
    std::vector<int> uniquevec_targetbinIdlist(
        unique_targetbinIdlist.begin(), unique_targetbinIdlist.end());
    const int uniquesize = (int)unique_targetbinIdlist.size();

    // 2) communication
    std::vector<int> unique_pidlist(uniquesize);
    int err = binstrategy_->bin_discret()->element_row_map()->remote_id_list(
        uniquesize, uniquevec_targetbinIdlist.data(), unique_pidlist.data(), nullptr);
    if (err < 0) FOUR_C_THROW("Core::LinAlg::Map::RemoteIDList returned err={}", err);

    // 3) build full pid list via lookup table
    std::map<int, int> lookuptable;
    for (int s = 0; s < uniquesize; ++s)
      lookuptable.insert(
          lookuptable.end(), std::pair<int, int>(uniquevec_targetbinIdlist[s], unique_pidlist[s]));
    for (int s = 0; s < fullsize; ++s) pidlist[s] = lookuptable[targetbinIdlist[s]];
  }

  // ---- pack data for sending -----
  std::map<int, std::vector<char>> sdata;
  std::vector<int> targetprocs(numproc, 0);
  int iter = 0;
  for (hlp = homelesslinker.begin(); hlp != homelesslinker.end(); ++hlp)
  {
    std::shared_ptr<Core::Nodes::Node> iterhomelesslinker = *hlp;

    // ---- pack data for sending -----
    const int targetproc = pidlist[iter];
    if (targetproc != -1)
    {
      Core::Communication::PackBuffer data;
      iterhomelesslinker->pack(data);
      binstrategy_->bin_discret()->delete_node(iterhomelesslinker->id());
      sdata[targetproc].insert(sdata[targetproc].end(), data().begin(), data().end());
      targetprocs[targetproc] = 1;
    }
    else
    {
      const int removeid = iterhomelesslinker->id();
      removedlinker->push_back(removeid);
      binstrategy_->bin_discret()->delete_node(removeid);
    }
    ++iter;
  }
  if (removedlinker->size() != 0)
    std::cout << " There are " << removedlinker->size()
              << " linker which have left the computational domain on rank " << myrank_
              << std::endl;
  homelesslinker.clear();

  // ---- prepare receiving procs -----
  std::vector<int> summedtargets(numproc, 0);
  summedtargets =
      Core::Communication::sum_all(targetprocs, binstrategy_->bin_discret()->get_comm());

  // ---- send ----
  Core::Communication::Exporter exporter(binstrategy_->bin_discret()->get_comm());
  const int length = sdata.size();
  std::vector<MPI_Request> request(length);
  int tag = 0;
  for (std::map<int, std::vector<char>>::const_iterator p = sdata.begin(); p != sdata.end(); ++p)
  {
    exporter.i_send(
        myrank_, p->first, (p->second).data(), (int)(p->second).size(), 1234, request[tag]);
    ++tag;
  }
  if (tag != length) FOUR_C_THROW("Number of messages is mixed up");

  receive_linker_and_fill_them_in_bins(summedtargets[myrank_], exporter, homelesslinker);

  // wait for all communications to finish
  {
    for (int i = 0; i < length; ++i) exporter.wait(request[i]);
  }

  Core::Communication::barrier(
      binstrategy_->bin_discret()->get_comm());  // I feel better this way ;-)

  return removedlinker;
}

/*-----------------------------------------------------------------------------------------*
| fill linker into their correct bin on according proc using ghosting   eichinger 02/17 |
 *-----------------------------------------------------------------------------------------*/
std::shared_ptr<std::list<int>>
BeamInteraction::BeamCrosslinkerHandler::fill_linker_into_bins_using_ghosting(
    std::list<std::shared_ptr<Core::Nodes::Node>>& homelesslinker)
{
  TEUCHOS_FUNC_TIME_MONITOR(
      "BeamInteraction::beam_crosslinker_handler::fill_linker_into_bins_using_ghosting");

  const int numproc = Core::Communication::num_mpi_ranks(binstrategy_->bin_discret()->get_comm());
  std::shared_ptr<std::list<int>> removedlinker = std::make_shared<std::list<int>>(0);

  // parallel case
  // ---- find new host procs for linker -----
  std::list<std::shared_ptr<Core::Nodes::Node>>::const_iterator hlp;
  std::map<int, std::list<std::shared_ptr<Core::Nodes::Node>>> towhomwhat;
  int binId, binowner;
  for (hlp = homelesslinker.begin(); hlp != homelesslinker.end(); ++hlp)
  {
    binId = binstrategy_->convert_pos_to_gid((*hlp)->x().data());
    if (binId == -1)
    {
      binowner = -1;
    }
    else
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      // safety check
      if (not binstrategy_->bin_discret()->have_global_element(binId))
        FOUR_C_THROW(
            "To transfer linker using ghosting you need to provide a one layer ghosting,"
            " that is not the case. Bin with gid {} not ghosted on rank {} ",
            binId, myrank_);
#endif
      binowner = binstrategy_->bin_discret()->g_element(binId)->owner();
    }
    towhomwhat[binowner].push_back((*hlp));
  }

  // -----------------------------------------------------------------------
  // send
  // -----------------------------------------------------------------------
  // ---- pack data for sending -----
  std::map<int, std::vector<char>> sdata;
  std::vector<int> targetprocs(numproc, 0);
  std::map<int, std::list<std::shared_ptr<Core::Nodes::Node>>>::const_iterator p;
  for (p = towhomwhat.begin(); p != towhomwhat.end(); ++p)
  {
    if (p->first != -1)
    {
      std::list<std::shared_ptr<Core::Nodes::Node>>::const_iterator iter;
      for (iter = p->second.begin(); iter != p->second.end(); ++iter)
      {
        Core::Communication::PackBuffer data;
        (*iter)->pack(data);
        binstrategy_->bin_discret()->delete_node((*iter)->id());
        sdata[p->first].insert(sdata[p->first].end(), data().begin(), data().end());
      }
      targetprocs[p->first] = 1;
    }
    else
    {
      std::list<std::shared_ptr<Core::Nodes::Node>>::const_iterator iter;
      for (iter = p->second.begin(); iter != p->second.end(); ++iter)
      {
        const int removeid = (*iter)->id();
        removedlinker->push_back(removeid);
        binstrategy_->bin_discret()->delete_node(removeid);
      }
    }
  }

  // ---- send ----
  Core::Communication::Exporter exporter(binstrategy_->bin_discret()->get_comm());
  const int length = sdata.size();
  std::vector<MPI_Request> request(length);
  int tag = 0;
  for (std::map<int, std::vector<char>>::const_iterator p = sdata.begin(); p != sdata.end(); ++p)
  {
    exporter.i_send(
        myrank_, p->first, (p->second).data(), (int)(p->second).size(), 1234, request[tag]);
    ++tag;
  }
  if (tag != length) FOUR_C_THROW("Number of messages is mixed up");


  if (removedlinker->size() != 0)
    std::cout << "There are " << removedlinker->size()
              << " linker which have "
                 "left the computational domain on rank "
              << myrank_ << std::endl;
  homelesslinker.clear();

  // -----------------------------------------------------------------------
  // receive
  // -----------------------------------------------------------------------
  // ---- prepare receiving procs -----
  std::vector<int> summedtargets(numproc, 0);
  summedtargets =
      Core::Communication::sum_all(targetprocs, binstrategy_->bin_discret()->get_comm());

  // ---- receive -----
  receive_linker_and_fill_them_in_bins(summedtargets[myrank_], exporter, homelesslinker);

  // wait for all communications to finish
  for (int i = 0; i < length; ++i) exporter.wait(request[i]);

  // should be no time operation (if we have done everything correctly)
  Core::Communication::barrier(binstrategy_->bin_discret()->get_comm());

  return removedlinker;
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
void BeamInteraction::BeamCrosslinkerHandler::receive_linker_and_fill_them_in_bins(int const numrec,
    Core::Communication::Exporter& exporter,
    std::list<std::shared_ptr<Core::Nodes::Node>>& homelesslinker)
{
  // ---- receive ----
  for (int rec = 0; rec < numrec; ++rec)
  {
    std::vector<char> rdata;
    int length = 0;
    int tag = -1;
    int from = -1;
    exporter.receive_any(from, tag, rdata, length);
    if (tag != 1234)
      FOUR_C_THROW("Received on proc {} data with wrong tag from proc {}", myrank_, from);

    // ---- unpack ----
    {
      // Put received nodes into discretization
      Core::Communication::UnpackBuffer buffer(rdata);
      while (!buffer.at_end())
      {
        auto object =
            std::shared_ptr<Core::Communication::ParObject>(Core::Communication::factory(buffer));
        std::shared_ptr<Core::Nodes::Node> node =
            std::dynamic_pointer_cast<Core::Nodes::Node>(object);
        if (node == nullptr) FOUR_C_THROW("Received object is not a node");

        // process received linker
        const double* currpos = node->x().data();
        place_node_correctly(node, currpos, homelesslinker);
        if (homelesslinker.size())
          FOUR_C_THROW(
              "linker (id: {}) was sent to proc {} but corresponding bin (gid: {}) "
              " is missing",
              node->id(), myrank_, binstrategy_->convert_pos_to_gid(currpos));
      }
    }
  }
}

/*----------------------------------------------------------------------*
| node is placed into the correct row bin                   ghamm 09/12 |
 *----------------------------------------------------------------------*/
bool BeamInteraction::BeamCrosslinkerHandler::place_node_correctly(
    std::shared_ptr<Core::Nodes::Node> node, const double* currpos,
    std::list<std::shared_ptr<Core::Nodes::Node>>& homelesslinker)
{
  //  std::cout << "on proc: " << myrank_ << " node with ID: " << node->Id() << " and owner: " <<
  //  node->Owner() << " arrived in PlaceNodeCorrectly" << std::endl;
  const int binId = binstrategy_->convert_pos_to_gid(currpos);

  // check whether the current node belongs into a bin on this proc
  const bool found = binstrategy_->bin_discret()->have_global_element(binId);

  // either fill linker into correct bin on this proc or mark it as homeless
  if (found == true)
  {
    Core::FE::MeshFree::MeshfreeMultiBin* currbin =
        dynamic_cast<Core::FE::MeshFree::MeshfreeMultiBin*>(
            binstrategy_->bin_discret()->g_element(binId));
#ifdef FOUR_C_ENABLE_ASSERTIONS
    if (currbin == nullptr)
      FOUR_C_THROW(
          "dynamic cast from Core::Elements::Element to Core::FE:MeshFree::MeshfreeMultiBin "
          "failed");
#endif
    // check whether it is a row bin
    if (currbin->owner() == myrank_)  // row bin
    {
      //      std::cout << "on proc: " << myrank_ << " for node " << node->Id() << " a row bin was
      //      found" << std::endl;
      // node already exists (either row or ghost)
      if (binstrategy_->bin_discret()->have_global_node(node->id()) == true)
      {
        Core::Nodes::Node* existingnode = binstrategy_->bin_discret()->g_node(node->id());
        // existing node is a row node, this means that node is equal existingnode
        if (existingnode->owner() == myrank_)
        {
          //          std::cout << "on proc: " << myrank_ << " existingnode row node " <<
          //          existingnode->Id() << " (ID from outside node: " << node->Id() << ") is added
          //          to element: " << currbin->Id() << std::endl;

          // assign node to the correct bin
          currbin->add_node(existingnode);
        }
        else  // delete existing node, insert received node into discretization and add it to
              // correct bin
        {
          // delete existing node
          binstrategy_->bin_discret()->delete_node(existingnode->id());
          // update ownership
          node->set_owner(myrank_);
          // add node
          binstrategy_->bin_discret()->add_node(node->x(), node->id(), node);
          // assign node to the correct bin
          currbin->add_node(node.get());

          //        std::cout << "on proc: " << myrank_ << " node " << node->Id() << " is added to
          //        the discretization and assigned to element: " << currbin->Id() << std::endl;
        }
      }
      else  // fill newly received node into discretization
      {
        // change owner of the node to this proc and add it to the discretization
        node->set_owner(myrank_);
        binstrategy_->bin_discret()->add_node(node->x(), node->id(), node);
        //        std::cout << "on proc: " << myrank_ << " node " << node->Id() << " is added to the
        //        discretization and assigned to element: " << currbin->Id() << std::endl;
        // assign node to the correct bin
        currbin->add_node(node.get());
      }
      return true;
    }
    else  // ghost bin
    {
      homelesslinker.push_back(node);
      //      std::cout << "on proc: " << myrank_ << " node " << node->Id() << " is added to
      //      homeless because of becoming a future ghost node" << std::endl;
      return false;
    }
  }
  else  // bin not found on this proc
  {
    homelesslinker.push_back(node);
    //    std::cout << "on proc: " << myrank_ << " node " << node->Id() << " is added to homeless
    //    because bin is not on this proc " << std::endl;
    return false;
  }
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
std::shared_ptr<std::list<int>> BeamInteraction::BeamCrosslinkerHandler::transfer_linker(
    bool const fill_using_ghosting)
{
  TEUCHOS_FUNC_TIME_MONITOR("BeamInteraction::beam_crosslinker_handler::TransferLinker");

  // set of homeless linker
  std::list<std::shared_ptr<Core::Nodes::Node>> homelesslinker;

  std::vector<int> examinedbins(binstrategy_->bin_discret()->num_my_row_elements(), 0);
  // first run over linker and then process whole bin in which linker is located
  // until all linker have been checked
  const int numrownodes = binstrategy_->bin_discret()->num_my_row_nodes();
  for (int i = 0; i < numrownodes; ++i)
  {
    Core::Nodes::Node* currlinker = binstrategy_->bin_discret()->l_row_node(i);

#ifdef FOUR_C_ENABLE_ASSERTIONS
    if (currlinker->num_element() != 1)
      FOUR_C_THROW("ERROR: A linker is assigned to more than one bin!");
#endif

    Core::FE::MeshFree::MeshfreeMultiBin* currbin =
        dynamic_cast<Core::FE::MeshFree::MeshfreeMultiBin*>(
            currlinker->adjacent_elements()[0].user_element());
    // as checked above, there is only one element in currele array
    const int binId = currbin->id();
    const int rlid = binstrategy_->bin_discret()->element_row_map()->lid(binId);

    // if a bin has already been examined --> continue with next linker
    if (examinedbins[rlid]) continue;
    // else: bin is examined for the first time --> new entry in examinedbins_
    else
      examinedbins[rlid] = 1;

    Core::Nodes::Node** linker = currbin->nodes();
    std::vector<int> tobemoved;
    for (int ilinker = 0; ilinker < currbin->num_node(); ++ilinker)
    {
      // get current node
      Core::Nodes::Node* currnode = linker[ilinker];

      // transform to array
      std::vector<double> pos(3, 0.0);
      for (int dim = 0; dim < 3; ++dim) pos[dim] = currnode->x()[dim];

      const int gidofbin = binstrategy_->convert_pos_to_gid(pos.data());
      // linker has left current bin
      if (gidofbin != binId)
      {
        // (looping over nodes and deleting at the same time is detrimental)
        tobemoved.push_back(currnode->id());
        // find new bin for linker
        place_node_correctly(
            Core::Utils::shared_ptr_from_ref(*currnode), pos.data(), homelesslinker);
      }
    }

    // finally remove nodes from their old bin
    for (size_t iter = 0; iter < tobemoved.size(); ++iter) currbin->delete_node(tobemoved[iter]);
  }

#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (homelesslinker.size())
    std::cout << "There are " << homelesslinker.size() << " homeless linker on proc" << myrank_
              << std::endl;
#endif

  // store linker that have left the computational domain
  std::shared_ptr<std::list<int>> deletedlinker = std::make_shared<std::list<int>>(0);

  //---------------------------------------------------------------------------
  // numproc == 1
  //---------------------------------------------------------------------------
  if (Core::Communication::num_mpi_ranks(binstrategy_->bin_discret()->get_comm()) == 1)
  {
    if (homelesslinker.size())
    {
      std::cout << " There are " << homelesslinker.size()
                << " linker which have left the"
                   " computational domain on rank "
                << myrank_ << std::endl;
      std::list<std::shared_ptr<Core::Nodes::Node>>::const_iterator hlp;
      for (hlp = homelesslinker.begin(); hlp != homelesslinker.end(); ++hlp)
      {
        const int removeid = (*hlp)->id();
        deletedlinker->push_back(removeid);
        binstrategy_->bin_discret()->delete_node(removeid);
      }
      homelesslinker.clear();
    }
    return deletedlinker;
  }

  //---------------------------------------------------------------------------
  // numproc > 1
  //---------------------------------------------------------------------------
  // homeless linker are sent to their new processors where they are inserted into their correct
  // bin
  if (fill_using_ghosting)
  {
    deletedlinker = fill_linker_into_bins_using_ghosting(homelesslinker);
  }
  else
  {
    deletedlinker = fill_linker_into_bins_remote_id_list(homelesslinker);
  }

  return deletedlinker;
}

/*-----------------------------------------------------------------------------*
 | build reduced bin col map based on boundary row bins       eichinger 01/17  |
 *-----------------------------------------------------------------------------*/
void BeamInteraction::BeamCrosslinkerHandler::
    get_neighbouring_bins_of_linker_containing_boundary_row_bins(std::set<int>& colbins) const
{
  colbins.clear();

  std::list<Core::Elements::Element*> const boundaryrowbins = binstrategy_->boundary_row_bins();

  if (boundaryrowbins.size() == 0)
    FOUR_C_THROW("Boundary row bins unknown, call function determine_boundary_row_bins() first!");

  // loop over boundary row bins and add neighbors of filled row bins
  std::list<Core::Elements::Element*>::const_iterator it;
  for (it = boundaryrowbins.begin(); it != boundaryrowbins.end(); ++it)
  {
    if ((*it)->num_node() != 0)
    {
      std::vector<int> binvec;
      binvec.reserve(26);
      // get neighboring bins
      binstrategy_->get_neighbor_bin_ids((*it)->id(), binvec);
      colbins.insert(binvec.begin(), binvec.end());
    }
  }
}

FOUR_C_NAMESPACE_CLOSE

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_engine_unique_global_id.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_comm_pack_helpers.hpp"
#include "4C_comm_parobject.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_particle_engine_communication_utils.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
Particle::UniqueGlobalIdHandler::UniqueGlobalIdHandler(MPI_Comm comm, const std::string& objectname)
    : comm_(comm),
      myrank_(Core::Communication::my_mpi_rank(comm)),
      masterrank_(0),
      objectname_(objectname),
      maxglobalid_(-1)
{
  // empty constructor
}

void Particle::UniqueGlobalIdHandler::write_restart(
    std::shared_ptr<Core::IO::DiscretizationWriter> writer) const
{
  // write maximum global id in restart
  writer->write_int(objectname_ + "maxglobalid", maxglobalid_);

  // write reusable global ids
  {
    std::vector<char> buffer;

    Core::Communication::PackBuffer data;
    add_to_pack(data, reusableglobalids_);

    buffer.insert(buffer.end(), data().begin(), data().end());

    writer->write_char_data(objectname_ + "reusableglobalids", buffer);
  }
}

void Particle::UniqueGlobalIdHandler::read_restart(
    const std::shared_ptr<Core::IO::DiscretizationReader> reader)
{
  // get maximum global id from restart
  maxglobalid_ = reader->read_int(objectname_ + "maxglobalid");

  // get reusable global ids from restart
  {
    std::shared_ptr<std::vector<char>> buffer = std::make_shared<std::vector<char>>();

    reader->read_char_vector(buffer, objectname_ + "reusableglobalids");


    Core::Communication::UnpackBuffer data(*buffer);
    while (!data.at_end())
    {
      extract_from_pack(data, reusableglobalids_);
    }
  }
}

void Particle::UniqueGlobalIdHandler::draw_requested_number_of_global_ids(
    std::vector<int>& requesteduniqueglobalids)
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::UniqueGlobalIdHandler::draw_requested_number_of_global_ids");

  // get number of requested global ids
  const int numberofrequestedgids = requesteduniqueglobalids.capacity();

  // gather reusable global ids from all processors on master processor
  gather_reusable_global_ids_from_all_procs_on_master_proc();

  // prepare requested global ids for all processors
  std::map<int, std::vector<int>> preparedglobalids;
  prepare_requested_global_ids_for_all_procs(numberofrequestedgids, preparedglobalids);

#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (myrank_ != masterrank_ and (not preparedglobalids.empty()))
    FOUR_C_THROW("generated global ids on processor {}", myrank_);
#endif

  // extract requested global ids on master processor
  extract_requested_global_ids_on_master_proc(preparedglobalids, requesteduniqueglobalids);

  // distribute requested global ids from master processor to all processors
  distribute_requested_global_ids_from_master_proc_to_all_procs(
      preparedglobalids, requesteduniqueglobalids);

#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (numberofrequestedgids != static_cast<int>(requesteduniqueglobalids.size()))
    FOUR_C_THROW("requested {} global ids on processor {} but received only {} global ids!",
        numberofrequestedgids, myrank_, requesteduniqueglobalids.size());
#endif

  // sort drawn unique global ids
  std::sort(requesteduniqueglobalids.begin(), requesteduniqueglobalids.end());
}

void Particle::UniqueGlobalIdHandler::gather_reusable_global_ids_from_all_procs_on_master_proc()
{
  // prepare buffer for sending and receiving
  std::map<int, std::vector<char>> sdata;
  std::map<int, std::vector<char>> rdata;

  if (myrank_ != masterrank_)
  {
    // pack data for sending
    Core::Communication::PackBuffer data;
    add_to_pack(data, reusableglobalids_);

    // clear reusable global ids
    reusableglobalids_.clear();

    // communicate reusable global ids to master processor
    sdata[masterrank_].insert(sdata[masterrank_].end(), data().begin(), data().end());
  }

  // communicate data via non-buffered send from proc to proc
  ParticleUtils::immediate_recv_blocking_send(comm_, sdata, rdata);

#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (myrank_ != masterrank_)
  {
    for (const auto& p : rdata)
    {
      const int msgsource = p.first;
      const std::vector<char>& rmsg = p.second;

      if (rmsg.size() != 0)
        FOUR_C_THROW(
            "not expected to received reusable global ids on processor {} from processor {}!",
            myrank_, msgsource);
    }
  }
#endif

  if (myrank_ == masterrank_)
  {
    // init receiving vector
    std::vector<int> receivedreusableglobalids;

    // unpack and store received data
    for (const auto& p : rdata)
    {
      const std::vector<char>& rmsg = p.second;


      Core::Communication::UnpackBuffer buffer(rmsg);
      while (!buffer.at_end())
      {
        extract_from_pack(buffer, receivedreusableglobalids);

        reusableglobalids_.insert(reusableglobalids_.end(), receivedreusableglobalids.begin(),
            receivedreusableglobalids.end());
      }
    }
  }

  // sort reusable global ids
  std::sort(reusableglobalids_.begin(), reusableglobalids_.end());

#ifdef FOUR_C_ENABLE_ASSERTIONS
  auto it = std::unique(reusableglobalids_.begin(), reusableglobalids_.end());
  if (it != reusableglobalids_.end()) FOUR_C_THROW("duplicate entries in reusable global ids!");
#endif
}

void Particle::UniqueGlobalIdHandler::prepare_requested_global_ids_for_all_procs(
    int numberofrequestedgids, std::map<int, std::vector<int>>& preparedglobalids)
{
  // gather number of requested global ids of each processor on master processor
  std::vector<int> numberofrequestedgidsofallprocs(Core::Communication::num_mpi_ranks(comm_), 0);
  MPI_Gather(&numberofrequestedgids, 1, MPI_INT, numberofrequestedgidsofallprocs.data(), 1, MPI_INT,
      masterrank_, comm_);

  if (myrank_ == masterrank_)
  {
    // get total number of requested global ids over all processors
    int totalnumberofrequestedgids = 0;
    for (const auto& n : numberofrequestedgidsofallprocs) totalnumberofrequestedgids += n;

    // prepare all requested global ids
    std::vector<int> allrequestedglobalids;
    allrequestedglobalids.reserve(totalnumberofrequestedgids);

    // enough global ids can be reused
    if (static_cast<int>(reusableglobalids_.size()) >= totalnumberofrequestedgids)
    {
      allrequestedglobalids.insert(allrequestedglobalids.begin(), reusableglobalids_.begin(),
          reusableglobalids_.begin() + totalnumberofrequestedgids);

      reusableglobalids_.erase(
          reusableglobalids_.begin(), reusableglobalids_.begin() + totalnumberofrequestedgids);
    }
    // reuse all global ids and generate additional global ids
    else
    {
      allrequestedglobalids.insert(
          allrequestedglobalids.begin(), reusableglobalids_.begin(), reusableglobalids_.end());

      reusableglobalids_.clear();

      int missingnumberofrequestedgids = totalnumberofrequestedgids - allrequestedglobalids.size();

      for (int i = 0; i < missingnumberofrequestedgids; ++i)
        allrequestedglobalids.push_back(++maxglobalid_);
    }

    // iterators for range of global ids to be set
    auto curr_range_begin = allrequestedglobalids.begin();

    for (int rank = 0; rank < Core::Communication::num_mpi_ranks(comm_); ++rank)
    {
      // get current number of requested global ids
      const int currnumberofrequestedgids = numberofrequestedgidsofallprocs[rank];

      if (currnumberofrequestedgids == 0) continue;

      // insert current requested global ids
      preparedglobalids[rank].insert(preparedglobalids[rank].begin(), curr_range_begin,
          curr_range_begin + currnumberofrequestedgids);

      // set current iterator for range begin
      curr_range_begin += currnumberofrequestedgids;
    }
  }

  // broadcast current maximum global id to all processors
  MPI_Bcast(&maxglobalid_, 1, MPI_INT, masterrank_, comm_);
}

void Particle::UniqueGlobalIdHandler::extract_requested_global_ids_on_master_proc(
    std::map<int, std::vector<int>>& preparedglobalids,
    std::vector<int>& requesteduniqueglobalids) const
{
  if (myrank_ == masterrank_)
  {
    if (not preparedglobalids.contains(masterrank_)) return;

    requesteduniqueglobalids.insert(requesteduniqueglobalids.begin(),
        preparedglobalids[masterrank_].begin(), preparedglobalids[masterrank_].end());

    preparedglobalids.erase(masterrank_);
  }
}

void Particle::UniqueGlobalIdHandler::distribute_requested_global_ids_from_master_proc_to_all_procs(
    std::map<int, std::vector<int>>& tobesendglobalids,
    std::vector<int>& requesteduniqueglobalids) const
{
  // prepare buffer for sending and receiving
  std::map<int, std::vector<char>> sdata;
  std::map<int, std::vector<char>> rdata;

  if (myrank_ == masterrank_)
  {
    // pack data for sending
    for (int torank = 0; torank < Core::Communication::num_mpi_ranks(comm_); ++torank)
    {
      if (tobesendglobalids[torank].empty()) continue;

      // pack data for sending
      Core::Communication::PackBuffer data;
      add_to_pack(data, tobesendglobalids[torank]);

      sdata[torank].insert(sdata[torank].end(), data().begin(), data().end());
    }
  }

  // communicate data via non-buffered send from proc to proc
  ParticleUtils::immediate_recv_blocking_send(comm_, sdata, rdata);

#ifdef FOUR_C_ENABLE_ASSERTIONS
  for (const auto& p : rdata)
  {
    const int msgsource = p.first;
    const std::vector<char>& rmsg = p.second;

    if (msgsource != masterrank_ and rmsg.size() != 0)
      FOUR_C_THROW("not expected to received global ids on processor {} from processor {}!",
          myrank_, msgsource);
  }
#endif

  if (myrank_ != masterrank_)
  {
    // unpack and store received data
    {
      std::vector<char>& rmsg = rdata[masterrank_];


      Core::Communication::UnpackBuffer buffer(rmsg);
      while (!buffer.at_end())
      {
        extract_from_pack(buffer, requesteduniqueglobalids);
      }
    }
  }
}

FOUR_C_NAMESPACE_CLOSE

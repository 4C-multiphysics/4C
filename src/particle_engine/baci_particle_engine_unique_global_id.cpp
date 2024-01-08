/*---------------------------------------------------------------------------*/
/*! \file
\brief unique global identifier handler for particle simulations
\level 1
*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "baci_particle_engine_unique_global_id.H"

#include "baci_comm_parobject.H"
#include "baci_io.H"
#include "baci_io_control.H"
#include "baci_particle_engine_communication_utils.H"

#include <Teuchos_TimeMonitor.hpp>

BACI_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
PARTICLEENGINE::UniqueGlobalIdHandler::UniqueGlobalIdHandler(
    const Epetra_Comm& comm, const std::string& objectname)
    : comm_(comm), myrank_(comm.MyPID()), masterrank_(0), objectname_(objectname), maxglobalid_(-1)
{
  // empty constructor
}

void PARTICLEENGINE::UniqueGlobalIdHandler::Init()
{
  // nothing to do
}

void PARTICLEENGINE::UniqueGlobalIdHandler::Setup()
{
  // nothing to do
}

void PARTICLEENGINE::UniqueGlobalIdHandler::WriteRestart(
    std::shared_ptr<IO::DiscretizationWriter> writer) const
{
  // write maximum global id in restart
  writer->WriteInt(objectname_ + "maxglobalid", maxglobalid_);

  // write reusable global ids
  {
    Teuchos::RCP<std::vector<char>> buffer = Teuchos::rcp(new std::vector<char>);

    CORE::COMM::PackBuffer data;
    CORE::COMM::ParObject::AddtoPack(data, reusableglobalids_);
    data.StartPacking();
    CORE::COMM::ParObject::AddtoPack(data, reusableglobalids_);

    buffer->insert(buffer->end(), data().begin(), data().end());

    writer->WriteCharVector(objectname_ + "reusableglobalids", buffer);
  }
}

void PARTICLEENGINE::UniqueGlobalIdHandler::ReadRestart(
    const std::shared_ptr<IO::DiscretizationReader> reader)
{
  // get maximum global id from restart
  maxglobalid_ = reader->ReadInt(objectname_ + "maxglobalid");

  // get reusable global ids from restart
  {
    Teuchos::RCP<std::vector<char>> buffer = Teuchos::rcp(new std::vector<char>);

    reader->ReadCharVector(buffer, objectname_ + "reusableglobalids");

    std::vector<char>::size_type position = 0;

    while (position < buffer->size())
    {
      CORE::COMM::ParObject::ExtractfromPack(position, *buffer, reusableglobalids_);
    }

    if (position != buffer->size())
      dserror("mismatch in size of data %d <-> %d", static_cast<int>(buffer->size()), position);
  }
}

void PARTICLEENGINE::UniqueGlobalIdHandler::DrawRequestedNumberOfGlobalIds(
    std::vector<int>& requesteduniqueglobalids)
{
  TEUCHOS_FUNC_TIME_MONITOR(
      "PARTICLEENGINE::UniqueGlobalIdHandler::DrawRequestedNumberOfGlobalIds");

  // get number of requested global ids
  const int numberofrequestedgids = requesteduniqueglobalids.capacity();

  // gather reusable global ids from all processors on master processor
  GatherReusableGlobalIdsFromAllProcsOnMasterProc();

  // prepare requested global ids for all processors
  std::map<int, std::vector<int>> preparedglobalids;
  PrepareRequestedGlobalIdsForAllProcs(numberofrequestedgids, preparedglobalids);

#ifdef BACI_DEBUG
  if (myrank_ != masterrank_ and (not preparedglobalids.empty()))
    dserror("generated global ids on processor %d", myrank_);
#endif

  // extract requested global ids on master processor
  ExtractRequestedGlobalIdsOnMasterProc(preparedglobalids, requesteduniqueglobalids);

  // distribute requested global ids from master processor to all processors
  DistributeRequestedGlobalIdsFromMasterProcToAllProcs(preparedglobalids, requesteduniqueglobalids);

#ifdef BACI_DEBUG
  if (numberofrequestedgids != static_cast<int>(requesteduniqueglobalids.size()))
    dserror("requested %d global ids on processor %d but received only %d global ids!",
        numberofrequestedgids, myrank_, requesteduniqueglobalids.size());
#endif

  // sort drawn unique global ids
  std::sort(requesteduniqueglobalids.begin(), requesteduniqueglobalids.end());
}

void PARTICLEENGINE::UniqueGlobalIdHandler::GatherReusableGlobalIdsFromAllProcsOnMasterProc()
{
  // prepare buffer for sending and receiving
  std::map<int, std::vector<char>> sdata;
  std::map<int, std::vector<char>> rdata;

  if (myrank_ != masterrank_)
  {
    // pack data for sending
    CORE::COMM::PackBuffer data;
    CORE::COMM::ParObject::AddtoPack(data, reusableglobalids_);
    data.StartPacking();
    CORE::COMM::ParObject::AddtoPack(data, reusableglobalids_);

    // clear reusable global ids
    reusableglobalids_.clear();

    // communicate reusable global ids to master processor
    sdata[masterrank_].insert(sdata[masterrank_].end(), data().begin(), data().end());
  }

  // communicate data via non-buffered send from proc to proc
  COMMUNICATION::ImmediateRecvBlockingSend(comm_, sdata, rdata);

#ifdef BACI_DEBUG
  if (myrank_ != masterrank_)
  {
    for (const auto& p : rdata)
    {
      const int msgsource = p.first;
      const std::vector<char>& rmsg = p.second;

      if (rmsg.size() != 0)
        dserror("not expected to received reusable global ids on processor %d from processor %d!",
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

      std::vector<char>::size_type position = 0;

      while (position < rmsg.size())
      {
        CORE::COMM::ParObject::ExtractfromPack(position, rmsg, receivedreusableglobalids);

        reusableglobalids_.insert(reusableglobalids_.end(), receivedreusableglobalids.begin(),
            receivedreusableglobalids.end());
      }

      if (position != rmsg.size())
        dserror("mismatch in size of data %d <-> %d", static_cast<int>(rmsg.size()), position);
    }
  }

  // sort reusable global ids
  std::sort(reusableglobalids_.begin(), reusableglobalids_.end());

#ifdef BACI_DEBUG
  auto it = std::unique(reusableglobalids_.begin(), reusableglobalids_.end());
  if (it != reusableglobalids_.end()) dserror("duplicate entries in reusable global ids!");
#endif
}

void PARTICLEENGINE::UniqueGlobalIdHandler::PrepareRequestedGlobalIdsForAllProcs(
    int numberofrequestedgids, std::map<int, std::vector<int>>& preparedglobalids)
{
  // mpi communicator
  const auto* mpicomm = dynamic_cast<const Epetra_MpiComm*>(&comm_);
  if (!mpicomm) dserror("dynamic cast to Epetra_MpiComm failed!");

  // gather number of requested global ids of each processor on master processor
  std::vector<int> numberofrequestedgidsofallprocs(comm_.NumProc(), 0);
  MPI_Gather(&numberofrequestedgids, 1, MPI_INT, numberofrequestedgidsofallprocs.data(), 1, MPI_INT,
      masterrank_, mpicomm->Comm());

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

    for (int rank = 0; rank < comm_.NumProc(); ++rank)
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
  MPI_Bcast(&maxglobalid_, 1, MPI_INT, masterrank_, mpicomm->Comm());
}

void PARTICLEENGINE::UniqueGlobalIdHandler::ExtractRequestedGlobalIdsOnMasterProc(
    std::map<int, std::vector<int>>& preparedglobalids,
    std::vector<int>& requesteduniqueglobalids) const
{
  if (myrank_ == masterrank_)
  {
    if (not preparedglobalids.count(masterrank_)) return;

    requesteduniqueglobalids.insert(requesteduniqueglobalids.begin(),
        preparedglobalids[masterrank_].begin(), preparedglobalids[masterrank_].end());

    preparedglobalids.erase(masterrank_);
  }
}

void PARTICLEENGINE::UniqueGlobalIdHandler::DistributeRequestedGlobalIdsFromMasterProcToAllProcs(
    std::map<int, std::vector<int>>& tobesendglobalids,
    std::vector<int>& requesteduniqueglobalids) const
{
  // prepare buffer for sending and receiving
  std::map<int, std::vector<char>> sdata;
  std::map<int, std::vector<char>> rdata;

  if (myrank_ == masterrank_)
  {
    // pack data for sending
    for (int torank = 0; torank < comm_.NumProc(); ++torank)
    {
      if (tobesendglobalids[torank].empty()) continue;

      // pack data for sending
      CORE::COMM::PackBuffer data;
      CORE::COMM::ParObject::AddtoPack(data, tobesendglobalids[torank]);
      data.StartPacking();
      CORE::COMM::ParObject::AddtoPack(data, tobesendglobalids[torank]);

      sdata[torank].insert(sdata[torank].end(), data().begin(), data().end());
    }
  }

  // communicate data via non-buffered send from proc to proc
  COMMUNICATION::ImmediateRecvBlockingSend(comm_, sdata, rdata);

#ifdef BACI_DEBUG
  for (const auto& p : rdata)
  {
    const int msgsource = p.first;
    const std::vector<char>& rmsg = p.second;

    if (msgsource != masterrank_ and rmsg.size() != 0)
      dserror("not expected to received global ids on processor %d from processor %d!", myrank_,
          msgsource);
  }
#endif

  if (myrank_ != masterrank_)
  {
    // unpack and store received data
    {
      std::vector<char>& rmsg = rdata[masterrank_];

      std::vector<char>::size_type position = 0;

      while (position < rmsg.size())
      {
        CORE::COMM::ParObject::ExtractfromPack(position, rmsg, requesteduniqueglobalids);
      }

      if (position != rmsg.size())
        dserror("mismatch in size of data %d <-> %d", static_cast<int>(rmsg.size()), position);
    }
  }
}

BACI_NAMESPACE_CLOSE

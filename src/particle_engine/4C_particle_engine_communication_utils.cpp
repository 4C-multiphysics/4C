// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_engine_communication_utils.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
void PARTICLEENGINE::COMMUNICATION::immediate_recv_blocking_send(
    MPI_Comm comm, std::map<int, std::vector<char>>& sdata, std::map<int, std::vector<char>>& rdata)
{
  // number of processors
  int const numproc = Core::Communication::num_mpi_ranks(comm);

  // processor id
  int const myrank = Core::Communication::my_mpi_rank(comm);

  // number of processors receiving data from this processor
  int const numsendtoprocs = sdata.size();

  // ---- communicate target processors to all processors ----
  std::vector<int> targetprocs(numproc, 0);
  std::vector<int> summedtargets(numproc, 0);

  for (const auto& p : sdata) targetprocs[p.first] = 1;

  summedtargets = Core::Communication::sum_all(targetprocs, comm);

  // number of processors this processor receives data from
  int const numrecvfromprocs = summedtargets[myrank];

  // ---- send size of messages to receiving processors ----
  std::vector<MPI_Request> sizerequest(numsendtoprocs);
  std::vector<int> sizetargets(numsendtoprocs);
  int counter = 0;
  for (const auto& p : sdata)
  {
    int const torank = p.first;
    if (myrank == torank) FOUR_C_THROW("processor should not send messages to itself!");
    if (torank < 0) FOUR_C_THROW("processor can not send messages to processor < 0!");

    sizetargets[counter] = torank;

    int msgsizetosend = static_cast<int>((p.second).size());

    // check sending size of message
    if (not(msgsizetosend > 0))
      FOUR_C_THROW("sending non-positive message size {} to proc {}!", msgsizetosend, torank);

    // perform non-blocking send operation
    MPI_Isend(&msgsizetosend, 1, MPI_INT, torank, 1234, comm, &sizerequest[counter]);

    ++counter;
  }
  if (counter != numsendtoprocs) FOUR_C_THROW("number of messages is mixed up!");

  // ---- receive size of messages ----
  std::vector<MPI_Request> recvrequest(numrecvfromprocs);
  for (int rec = 0; rec < numrecvfromprocs; ++rec)
  {
    // probe for any message to come
    MPI_Status status;
    MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &status);

    // get message sender and tag
    int const msgsource = status.MPI_SOURCE;
    int const msgtag = status.MPI_TAG;

    // check message tag
    if (msgtag != 1234)
      FOUR_C_THROW("received data on proc {} with wrong tag from proc {}", myrank, msgsource);

    // get message size
    int msgsize = -1;
    MPI_Get_count(&status, MPI_INT, &msgsize);

    // check size of message
    if (msgsize != 1) FOUR_C_THROW("message size not correct (one int expected)!");

    // perform blocking receive operation
    int msgsizetorecv = -1;
    MPI_Recv(&msgsizetorecv, msgsize, MPI_INT, msgsource, msgtag, comm, MPI_STATUS_IGNORE);

    // check received size of message
    if (not(msgsizetorecv > 0))
      FOUR_C_THROW("received non-positive message size {} from proc {}!", msgsizetorecv, msgsource);

    // resize receiving buffer to received size
    std::vector<char>& rbuffer = rdata[msgsource];
    rbuffer.resize(msgsizetorecv);

    // perform non-blocking receive operation
    MPI_Irecv(
        (void*)(rbuffer.data()), msgsizetorecv, MPI_CHAR, msgsource, 5678, comm, &recvrequest[rec]);
  }

  // ---- send data to already waiting processors ----
  std::vector<MPI_Request> sendrequest(numsendtoprocs);
  counter = 0;
  while (counter != numsendtoprocs)
  {
    // test for non-blocking send operation
    int index = -1;
    int flag = 0;
    MPI_Testany(numsendtoprocs, sizerequest.data(), &index, &flag, MPI_STATUS_IGNORE);

    if (flag)
    {
      int const torank = sizetargets[index];
      if (myrank == torank) FOUR_C_THROW("processor should not send messages to itself!");
      if (torank < 0) FOUR_C_THROW("processor can not send messages to processor < 0!");

      // reference to send buffer
      std::vector<char>& sbuffer = sdata[torank];

      // perform non-blocking send operation
      MPI_Isend((void*)(sbuffer.data()), static_cast<int>(sbuffer.size()), MPI_CHAR, torank, 5678,
          comm, &sendrequest[index]);

      ++counter;
    }
  }

  // ---- wait for completion of send operations ----
  MPI_Waitall(numsendtoprocs, sendrequest.data(), MPI_STATUSES_IGNORE);

  // clear send buffer after successful communication
  sdata.clear();

  // ---- wait for completion of receive operations ----
  MPI_Waitall(numrecvfromprocs, recvrequest.data(), MPI_STATUSES_IGNORE);
}

FOUR_C_NAMESPACE_CLOSE

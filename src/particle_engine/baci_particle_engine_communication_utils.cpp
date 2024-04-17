/*---------------------------------------------------------------------------*/
/*! \file
\brief communication utils for particle problem
\level 1
*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "baci_particle_engine_communication_utils.hpp"

#include "baci_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
void PARTICLEENGINE::COMMUNICATION::ImmediateRecvBlockingSend(const Epetra_Comm& comm,
    std::map<int, std::vector<char>>& sdata, std::map<int, std::vector<char>>& rdata)
{
  // number of processors
  int const numproc = comm.NumProc();

  // processor id
  int const myrank = comm.MyPID();

  // number of processors receiving data from this processor
  int const numsendtoprocs = sdata.size();

  // mpi communicator
  const auto* mpicomm = dynamic_cast<const Epetra_MpiComm*>(&comm);
  if (!mpicomm) dserror("dynamic cast to Epetra_MpiComm failed!");

  // ---- communicate target processors to all processors ----
  std::vector<int> targetprocs(numproc, 0);
  std::vector<int> summedtargets(numproc, 0);

  for (const auto& p : sdata) targetprocs[p.first] = 1;

  comm.SumAll(targetprocs.data(), summedtargets.data(), numproc);

  // number of processors this processor receives data from
  int const numrecvfromprocs = summedtargets[myrank];

  // ---- send size of messages to receiving processors ----
  std::vector<MPI_Request> sizerequest(numsendtoprocs);
  std::vector<int> sizetargets(numsendtoprocs);
  int counter = 0;
  for (const auto& p : sdata)
  {
    int const torank = p.first;
    if (myrank == torank) dserror("processor should not send messages to itself!");
    if (torank < 0) dserror("processor can not send messages to processor < 0!");

    sizetargets[counter] = torank;

    int msgsizetosend = static_cast<int>((p.second).size());

    // check sending size of message
    if (not(msgsizetosend > 0))
      dserror("sending non-positive message size %i to proc %i!", msgsizetosend, torank);

    // perform non-blocking send operation
    MPI_Isend(&msgsizetosend, 1, MPI_INT, torank, 1234, mpicomm->Comm(), &sizerequest[counter]);

    ++counter;
  }
  if (counter != numsendtoprocs) dserror("number of messages is mixed up!");

  // ---- receive size of messages ----
  std::vector<MPI_Request> recvrequest(numrecvfromprocs);
  for (int rec = 0; rec < numrecvfromprocs; ++rec)
  {
    // probe for any message to come
    MPI_Status status;
    MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, mpicomm->Comm(), &status);

    // get message sender and tag
    int const msgsource = status.MPI_SOURCE;
    int const msgtag = status.MPI_TAG;

    // check message tag
    if (msgtag != 1234)
      dserror("received data on proc %i with wrong tag from proc %i", myrank, msgsource);

    // get message size
    int msgsize = -1;
    MPI_Get_count(&status, MPI_INT, &msgsize);

    // check size of message
    if (msgsize != 1) dserror("message size not correct (one int expected)!");

    // perform blocking receive operation
    int msgsizetorecv = -1;
    MPI_Recv(
        &msgsizetorecv, msgsize, MPI_INT, msgsource, msgtag, mpicomm->Comm(), MPI_STATUS_IGNORE);

    // check received size of message
    if (not(msgsizetorecv > 0))
      dserror("received non-positive message size %i from proc %i!", msgsizetorecv, msgsource);

    // resize receiving buffer to received size
    std::vector<char>& rbuffer = rdata[msgsource];
    rbuffer.resize(msgsizetorecv);

    // perform non-blocking receive operation
    MPI_Irecv((void*)(rbuffer.data()), msgsizetorecv, MPI_CHAR, msgsource, 5678, mpicomm->Comm(),
        &recvrequest[rec]);
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
      if (myrank == torank) dserror("processor should not send messages to itself!");
      if (torank < 0) dserror("processor can not send messages to processor < 0!");

      // reference to send buffer
      std::vector<char>& sbuffer = sdata[torank];

      // perform non-blocking send operation
      MPI_Isend((void*)(sbuffer.data()), static_cast<int>(sbuffer.size()), MPI_CHAR, torank, 5678,
          mpicomm->Comm(), &sendrequest[index]);

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

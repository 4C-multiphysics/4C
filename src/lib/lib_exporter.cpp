/*---------------------------------------------------------------------*/
/*! \file

\brief Implementation of exporter class

\level 0

*/
/*---------------------------------------------------------------------*/

#include "vector"
#include "lib_exporter.H"
#include "lib_utils.H"
#include "lib_dserror.H"


DRT::Exporter::Exporter(const Epetra_Comm& comm)
    : dummymap_(0, 0, comm),
      frommap_(dummymap_),
      tomap_(dummymap_),
      comm_(comm),
      myrank_(comm.MyPID()),
      numproc_(comm.NumProc())
{
}

DRT::Exporter::Exporter(const Epetra_Map& frommap, const Epetra_Map& tomap, const Epetra_Comm& comm)
    : dummymap_(0, 0, comm),
      frommap_(frommap),
      tomap_(tomap),
      comm_(comm),
      myrank_(comm.MyPID()),
      numproc_(comm.NumProc())
{
  ConstructExporter();
}

void DRT::Exporter::ISend(const int frompid, const int topid, const char* data, const int dsize,
    const int tag, MPI_Request& request) const
{
  if (MyPID() != frompid) return;
  const auto* comm = dynamic_cast<const Epetra_MpiComm*>(&(Comm()));
  if (!comm) dserror("Comm() is not a Epetra_MpiComm\n");
  MPI_Isend((void*)data, dsize, MPI_CHAR, topid, tag, comm->Comm(), &request);
}

void DRT::Exporter::ISend(const int frompid, const int topid, const int* data, const int dsize,
    const int tag, MPI_Request& request) const
{
  if (MyPID() != frompid) return;
  const auto* comm = dynamic_cast<const Epetra_MpiComm*>(&(Comm()));
  if (!comm) dserror("Comm() is not a Epetra_MpiComm\n");
  MPI_Isend((void*)data, dsize, MPI_INT, topid, tag, comm->Comm(), &request);
}

void DRT::Exporter::ISend(const int frompid, const int topid, const double* data, const int dsize,
    const int tag, MPI_Request& request) const
{
  if (MyPID() != frompid) return;
  const auto* comm = dynamic_cast<const Epetra_MpiComm*>(&(Comm()));
  if (!comm) dserror("Comm() is not a Epetra_MpiComm\n");
  MPI_Isend((void*)data, dsize, MPI_DOUBLE, topid, tag, comm->Comm(), &request);
}

void DRT::Exporter::ReceiveAny(
    int& source, int& tag, std::vector<char>& recvbuff, int& length) const
{
  const auto* comm = dynamic_cast<const Epetra_MpiComm*>(&(Comm()));
  if (!comm) dserror("Comm() is not a Epetra_MpiComm\n");
  MPI_Status status;
  // probe for any message to come
  MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm->Comm(), &status);
  // get sender, tag and length
  source = status.MPI_SOURCE;
  tag = status.MPI_TAG;
  MPI_Get_count(&status, MPI_CHAR, &length);
  if (length > (int)recvbuff.size()) recvbuff.resize(length);
  // receive the message
  MPI_Recv(recvbuff.data(), length, MPI_CHAR, source, tag, comm->Comm(), &status);
}

void DRT::Exporter::Receive(
    const int source, const int tag, std::vector<char>& recvbuff, int& length) const
{
  const auto* comm = dynamic_cast<const Epetra_MpiComm*>(&(Comm()));
  if (!comm) dserror("Comm() is not a Epetra_MpiComm\n");
  MPI_Status status;
  // probe for any message to come
  MPI_Probe(source, tag, comm->Comm(), &status);
  MPI_Get_count(&status, MPI_CHAR, &length);
  if (length > (int)recvbuff.size()) recvbuff.resize(length);
  // receive the message
  MPI_Recv(recvbuff.data(), length, MPI_CHAR, source, tag, comm->Comm(), &status);
}

void DRT::Exporter::ReceiveAny(int& source, int& tag, std::vector<int>& recvbuff, int& length) const
{
  const auto* comm = dynamic_cast<const Epetra_MpiComm*>(&(Comm()));
  if (!comm) dserror("Comm() is not a Epetra_MpiComm\n");
  MPI_Status status;
  // probe for any message to come
  MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm->Comm(), &status);
  // get sender, tag and length
  source = status.MPI_SOURCE;
  tag = status.MPI_TAG;
  MPI_Get_count(&status, MPI_INT, &length);
  if (length > (int)recvbuff.size()) recvbuff.resize(length);
  // receive the message
  MPI_Recv(recvbuff.data(), length, MPI_INT, source, tag, comm->Comm(), &status);
}

void DRT::Exporter::Receive(
    const int source, const int tag, std::vector<int>& recvbuff, int& length) const
{
  const auto* comm = dynamic_cast<const Epetra_MpiComm*>(&(Comm()));
  if (!comm) dserror("Comm() is not a Epetra_MpiComm\n");
  MPI_Status status;
  // probe for any message to come
  MPI_Probe(source, tag, comm->Comm(), &status);
  MPI_Get_count(&status, MPI_INT, &length);
  if (length > (int)recvbuff.size()) recvbuff.resize(length);
  // receive the message
  MPI_Recv(recvbuff.data(), length, MPI_INT, source, tag, comm->Comm(), &status);
}

void DRT::Exporter::ReceiveAny(
    int& source, int& tag, std::vector<double>& recvbuff, int& length) const
{
  const auto* comm = dynamic_cast<const Epetra_MpiComm*>(&(Comm()));
  if (!comm) dserror("Comm() is not a Epetra_MpiComm\n");
  MPI_Status status;
  // probe for any message to come
  MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm->Comm(), &status);
  // get sender, tag and length
  source = status.MPI_SOURCE;
  tag = status.MPI_TAG;
  MPI_Get_count(&status, MPI_DOUBLE, &length);
  if (length > (int)recvbuff.size()) recvbuff.resize(length);
  // receive the message
  MPI_Recv(recvbuff.data(), length, MPI_DOUBLE, source, tag, comm->Comm(), &status);
}

void DRT::Exporter::Receive(
    const int source, const int tag, std::vector<double>& recvbuff, int& length) const
{
  const auto* comm = dynamic_cast<const Epetra_MpiComm*>(&(Comm()));
  if (!comm) dserror("Comm() is not a Epetra_MpiComm\n");
  MPI_Status status;
  // probe for any message to come
  MPI_Probe(source, tag, comm->Comm(), &status);
  MPI_Get_count(&status, MPI_DOUBLE, &length);
  if (length > (int)recvbuff.size()) recvbuff.resize(length);
  // receive the message
  MPI_Recv(recvbuff.data(), length, MPI_DOUBLE, source, tag, comm->Comm(), &status);
}

void DRT::Exporter::Allreduce(std::vector<int>& sendbuff, std::vector<int>& recvbuff, MPI_Op mpi_op)
{
  const auto* comm = dynamic_cast<const Epetra_MpiComm*>(&(Comm()));
  if (!comm) dserror("Comm() is not a Epetra_MpiComm\n");

  int length = (int)sendbuff.size();
  if (length > (int)recvbuff.size()) recvbuff.resize(length);

  MPI_Allreduce(sendbuff.data(), recvbuff.data(), length, MPI_INT, mpi_op, comm->Comm());
}

void DRT::Exporter::Broadcast(const int frompid, std::vector<char>& data, const int tag) const
{
  const auto* comm = dynamic_cast<const Epetra_MpiComm*>(&(Comm()));
  if (!comm) dserror("Comm() is not a Epetra_MpiComm\n");

  int length = static_cast<int>(data.size());
  MPI_Bcast(&length, 1, MPI_INT, frompid, comm->Comm());
  if (MyPID() != frompid)
  {
    data.resize(length);
  }
  MPI_Bcast((void*)data.data(), length, MPI_CHAR, frompid, comm->Comm());
}

void DRT::Exporter::ConstructExporter()
{
  if (SourceMap().SameAs(TargetMap())) return;

  // allocate a sendplan array and init to zero
  // SendPlan():
  // SendPlan()(lid,proc)    = 1 for data with local id lid needs sending to proc
  // SendPlan()(lid,proc)    = 0 otherwise
  // SendPlan()(lid,MyPID()) = 0 always! (I never send to myself)
  SendPlan().resize(NumProc());

  // To build these plans, everybody has to communicate what he has and wants:
  // bundle this info to save on communication:
  int sizes[2];
  sizes[0] = SourceMap().NumMyElements();
  sizes[1] = TargetMap().NumMyElements();
  const int sendsize = sizes[0] + sizes[1];
  std::vector<int> sendbuff;
  sendbuff.reserve(sendsize);
  std::copy(SourceMap().MyGlobalElements(),
      SourceMap().MyGlobalElements() + SourceMap().NumMyElements(), std::back_inserter(sendbuff));
  std::copy(TargetMap().MyGlobalElements(),
      TargetMap().MyGlobalElements() + TargetMap().NumMyElements(), std::back_inserter(sendbuff));

  for (int proc = 0; proc < NumProc(); ++proc)
  {
    int recvsizes[2];
    recvsizes[0] = sizes[0];
    recvsizes[1] = sizes[1];
    Comm().Broadcast(recvsizes, 2, proc);
    const int recvsize = recvsizes[0] + recvsizes[1];
    std::vector<int> recvbuff(recvsize);
    if (proc == MyPID()) std::copy(sendbuff.begin(), sendbuff.end(), recvbuff.data());
    Comm().Broadcast(recvbuff.data(), recvsize, proc);
    // const int* have = recvbuff.data();            // this is what proc has
    const int* want = &recvbuff[recvsizes[0]];  // this is what proc needs

    // Loop what proc wants and what I have (SendPlan)
    if (proc != MyPID())
    {
      for (int i = 0; i < recvsizes[1]; ++i)
      {
        const int gid = want[i];
        if (SourceMap().MyGID(gid))
        {
          const int lid = SourceMap().LID(gid);
          SendPlan()[proc].insert(lid);
        }
      }
    }
    Comm().Barrier();
  }  // for (int proc=0; proc<NumProc(); ++proc)
}

void DRT::Exporter::GenericExport(ExporterHelper& helper)
{
  if (SendPlan().size() == 0) return;
  // if (SourceMap().SameAs(TargetMap())) return;

  helper.PreExportTest(this);

  //------------------------------------------------ do the send/recv loop
  for (int i = 0; i < NumProc() - 1; ++i)
  {
    int tproc = MyPID() + 1 + i;
    int sproc = MyPID() - 1 - i;
    if (tproc < 0) tproc += NumProc();
    if (sproc < 0) sproc += NumProc();
    if (tproc > NumProc() - 1) tproc -= NumProc();
    if (sproc > NumProc() - 1) sproc -= NumProc();
    // cout << "Proc " << MyPID() << " tproc " << tproc << " sproc " << sproc << endl;
    // fflush(stdout);

    //------------------------------------------------ do sending to tproc
    // gather all objects to be send
    DRT::PackBuffer sendblock;
    std::vector<int> sendgid;
    sendgid.reserve(SendPlan()[tproc].size());

    // count

    for (int lid : SendPlan()[tproc])
    {
      const int gid = SourceMap().GID(lid);
      helper.PackObject(gid, sendblock);
    }

    // pack

    sendblock.StartPacking();

    for (int lid : SendPlan()[tproc])
    {
      const int gid = SourceMap().GID(lid);
      if (helper.PackObject(gid, sendblock)) sendgid.push_back(gid);
    }

    // send tproc no. of chars tproc must receive
    std::vector<int> snmessages(2);
    snmessages[0] = sendblock().size();
    snmessages[1] = sendgid.size();

    MPI_Request sizerequest;
    ISend(MyPID(), tproc, snmessages.data(), 2, 1, sizerequest);

    // do the sending of the objects
    MPI_Request sendrequest;
    ISend(MyPID(), tproc, sendblock().data(), sendblock().size(), 2, sendrequest);

    MPI_Request sendgidrequest;
    ISend(MyPID(), tproc, sendgid.data(), sendgid.size(), 3, sendgidrequest);

    //---------------------------------------- do the receiving from sproc
    // receive how many messages I will receive from sproc
    std::vector<int> rnmessages(2);
    int source = sproc;
    int length = 0;
    int tag = 1;
    // do a blocking specific receive
    Receive(source, tag, rnmessages, length);
    if (length != 2 or tag != 1) dserror("Messages got mixed up");

    // receive the objects
    std::vector<char> recvblock(rnmessages[0]);
    tag = 2;
    ReceiveAny(source, tag, recvblock, length);
    if (tag != 2) dserror("Messages got mixed up");

    // receive the gids
    std::vector<int> recvgid(rnmessages[1]);
    tag = 3;
    ReceiveAny(source, tag, recvgid, length);
    if (tag != 3) dserror("Messages got mixed up");

    std::vector<char>::size_type index = 0;
    int j = 0;
    while (index < recvblock.size())
    {
      int gid = recvgid[j];
      helper.UnpackObject(gid, index, recvblock);
      j += 1;
    }

    //----------------------------------- do waiting for messages to tproc to leave
    Wait(sizerequest);
    Wait(sendrequest);
    Wait(sendgidrequest);

    // make sure we do not get mixed up messages as we use wild card receives here
    Comm().Barrier();
  }  // for (int i=0; i<NumProc()-1; ++i)

  helper.PostExportCleanup(this);
}

void DRT::Exporter::Export(std::map<int, int>& data)
{
  PODExporterHelper<int> helper(data);
  GenericExport(helper);
}

void DRT::Exporter::Export(std::map<int, double>& data)
{
  PODExporterHelper<double> helper(data);
  GenericExport(helper);
}

void DRT::Exporter::Export(std::map<int, Teuchos::RCP<Epetra_SerialDenseMatrix>>& data)
{
  AnyObjectExporterHelper<Epetra_SerialDenseMatrix> helper(data);
  GenericExport(helper);
}

/*!----------------------------------------------------------------------
\file drt_discret_partition.cpp
\brief

<pre>
-------------------------------------------------------------------------
                 BACI finite element library subsystem
            Copyright (2008) Technical University of Munich

Under terms of contract T004.008.000 there is a non-exclusive license for use
of this work by or on behalf of Rolls-Royce Ltd & Co KG, Germany.

This library is proprietary software. It must not be published, distributed,
copied or altered in any form or any media without written permission
of the copyright holder. It may be used under terms and conditions of the
above mentioned license by or on behalf of Rolls-Royce Ltd & Co KG, Germany.

This library may solemnly used in conjunction with the BACI contact library
for purposes described in the above mentioned contract.

This library contains and makes use of software copyrighted by Sandia Corporation
and distributed under LGPL licence. Licensing does not apply to this or any
other third party software used here.

Questions? Contact Dr. Michael W. Gee (gee@lnm.mw.tum.de)
                   or
                   Prof. Dr. Wolfgang A. Wall (wall@lnm.mw.tum.de)

http://www.lnm.mw.tum.de

-------------------------------------------------------------------------
</pre>

<pre>
Maintainer: Michael Gee
            gee@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15239
</pre>

*----------------------------------------------------------------------*/

#include <Epetra_FECrsGraph.h>
#include <Epetra_Time.h>

#include "drt_discret.H"
#include "drt_exporter.H"
#include "drt_dserror.H"

/*----------------------------------------------------------------------*
 |  Export nodes owned by a proc (public)                    mwgee 11/06|
 *----------------------------------------------------------------------*/
void DRT::Discretization::ExportRowNodes(const Epetra_Map& newmap)
{
  // test whether newmap is non-overlapping
  if (!newmap.UniqueGIDs()) dserror("new map not unique");

  // destroy all ghosted nodes
  const int myrank = Comm().MyPID();
  map<int,RCP<DRT::Node> >::iterator curr;
  for (curr=node_.begin(); curr!=node_.end();)
  {
    if (curr->second->Owner() != myrank)
      node_.erase(curr++);
    else
      ++curr;
  }

  // build rowmap of nodes noderowmap_ if it does not exist
  if (noderowmap_==null) BuildNodeRowMap();
  const Epetra_Map& oldmap = *noderowmap_;

  // create an exporter object that will figure out the communication pattern
  DRT::Exporter exporter(oldmap,newmap,Comm());

  // Do the communication
  exporter.Export(node_);

  // update all ownership flags
  for (curr=node_.begin(); curr!=node_.end(); ++curr)
    curr->second->SetOwner(myrank);

  // maps and pointers are no longer correct and need rebuilding
  Reset();

  return;
}

/*----------------------------------------------------------------------*
 |  Export nodes owned by a proc (public)                    mwgee 11/06|
 *----------------------------------------------------------------------*/
void DRT::Discretization::ExportColumnNodes(const Epetra_Map& newmap)
{
  // destroy all ghosted nodes
  const int myrank = Comm().MyPID();
  map<int,RefCountPtr<DRT::Node> >::iterator curr;
  for (curr=node_.begin(); curr!=node_.end();)
  {
    if (curr->second->Owner() != myrank)
    {
      node_.erase(curr++);
    }
    else
    {
      ++curr;
    }
  }
  // build rowmap of nodes noderowmap_ if it does not exist
  if (noderowmap_==null) BuildNodeRowMap();
  const Epetra_Map& oldmap = *noderowmap_;

  // test whether all nodes in oldmap are also in newmap, otherwise
  // this would be a change of owner which is not allowed here
  for (int i=0; i<oldmap.NumMyElements(); ++i)
  {
    int gid = oldmap.GID(i);
    if (!(newmap.MyGID(gid)))
      dserror("Proc %d: Node gid=%d from oldmap is not in newmap",myrank,gid);
  }

  // create an exporter object that will figure out the communication pattern
  DRT::Exporter exporter(oldmap,newmap,Comm());
  // Do the communication
  exporter.Export(node_);

  // maps and pointers are no longer correct and need rebuilding
  Reset();

  return;
}


/*----------------------------------------------------------------------*
 |  Export elements (public)                                 mwgee 02/11|
 *----------------------------------------------------------------------*/
void DRT::Discretization::ProcZeroDistributeElementsToAll(Epetra_Map& target,
                                                          vector<int>& gidlist)
{
  const int myrank = Comm().MyPID();

  // proc 0 looks for elements that are to be send to other procs
  int size = (int)gidlist.size();
  vector<int> pidlist(size); // gids on proc 0
  int err = target.RemoteIDList(size,&gidlist[0],&pidlist[0],NULL);
  if (err < 0) dserror("Epetra_BlockMap::RemoteIDList returned err=%d",err);

  map<int,vector<char> > sendmap; // proc to send a set of elements to
  if (!myrank)
  {
    map<int,DRT::PackBuffer > sendpb; // proc to send a set of elements to
    for (int i=0; i<size; ++i)
    {
      if (pidlist[i]==myrank or pidlist[i]<0) continue; // do not send to myself
      Element* actele = gElement(gidlist[i]);
      if (!actele) dserror("Cannot find global element %d",gidlist[i]);
      actele->Pack(sendpb[pidlist[i]]);
    }
    for (map<int,DRT::PackBuffer >::iterator fool = sendpb.begin(); fool != sendpb.end(); ++fool)
      fool->second.StartPacking();
    for (int i=0; i<size; ++i)
    {
      if (pidlist[i]==myrank or pidlist[i]<0) continue; // do not send to myself
      Element* actele = gElement(gidlist[i]);
      actele->Pack(sendpb[pidlist[i]]);
      element_.erase(actele->Id());
    }
    for (map<int,DRT::PackBuffer >::iterator fool = sendpb.begin(); fool != sendpb.end(); ++fool)
      swap(sendmap[fool->first],fool->second());
  }


#ifdef PARALLEL
  // tell everybody who is to receive something
  vector<int> receivers;

  for (map<int,vector<char> >::iterator fool = sendmap.begin(); fool !=sendmap.end(); ++fool)
    receivers.push_back(fool->first);
  size = (int)receivers.size();
  Comm().Broadcast(&size,1,0);
  if (myrank != 0) receivers.resize(size);
  Comm().Broadcast(&receivers[0],size,0);
  int foundme = -1;
  if (myrank != 0)
    for (int i=0; i<size; ++i)
      if (receivers[i]==myrank)
      {
        foundme = i;
        break;
      }


  // proc 0 sends out messages
  int tag = 0;
  DRT::Exporter exporter(Comm());
  vector<MPI_Request> request(size);
  if (!myrank)
  {
    for (map<int,vector<char> >::iterator fool = sendmap.begin(); fool !=sendmap.end(); ++fool)
    {
      exporter.ISend(0,fool->first,&fool->second[0],(int)fool->second.size(),tag,request[tag]);
      tag++;
    }
    if (tag != size) dserror("Number of messages is mixed up");
    // do not delete sendmap until Wait has returned!
  }


  // all other procs listen to message and put element into dis
  if (foundme != -1)
  {
    vector<char> recvdata;
    int length = 0;
    int source = -1;
    int tag = -1;
    exporter.ReceiveAny(source,tag,recvdata,length);
    if (source != 0 || tag != foundme)
      dserror("Messages got mixed up");
    // Put received elements into discretization
    vector<char>::size_type index = 0;
    while (index < recvdata.size())
    {
      vector<char> data;
      ParObject::ExtractfromPack(index,recvdata,data);
      DRT::ParObject* object = DRT::UTILS::Factory(data);
      DRT::Element* ele = dynamic_cast<DRT::Element*>(object);
      if (!ele) dserror("Received object is not an element");
      ele->SetOwner(myrank);
      RCP<DRT::Element> rcpele = rcp(ele);
      AddElement(rcpele);
      //printf("proc %d index %d\n",myrank,index); fflush(stdout);
    }
  }

  // wait for all communication to finish
  if (!myrank)
  {
    for (int i=0; i<size; ++i)
      exporter.Wait(request[i]);
  }
#endif

  Comm().Barrier(); // I feel better this way ;-)
  Reset();
  return;
}

/*----------------------------------------------------------------------*
 |  Export elements (public)                                 mwgee 03/11|
 *----------------------------------------------------------------------*/
void DRT::Discretization::ProcZeroDistributeNodesToAll(Epetra_Map& target)
{
  const int myrank = Comm().MyPID();

#if 0
  Epetra_Time timer(Comm());
  double t1 = timer.ElapsedTime();
#endif

  // proc 0 looks for nodes that are to be distributed
  Reset();
  BuildNodeRowMap();
  const Epetra_Map& oldmap = *noderowmap_;
  int size = oldmap.NumMyElements();
  if (myrank) size = 0;
  vector<int> pidlist(size,-1);
  {
    int err = target.RemoteIDList(size,oldmap.MyGlobalElements(),&pidlist[0],NULL);
    if (err) dserror("Epetra_BlockMap::RemoteIDLis returned err=%d",err);
  }

#if 0
  for (int proc=0; proc<Comm().NumProc(); ++proc)
  {
    if (proc==myrank)
    {
      printf("\nProc %d numnode %d\n",myrank,size);
      for (int i=0; i<size; ++i)
        printf("Proc %d gid %d pid %d\n",myrank,oldmap.MyGlobalElements()[i],pidlist[i]);
    }
    fflush(stdout);
    Comm().Barrier();
  }
#endif

#if 0
  double t2 = timer.ElapsedTime();
  if (!myrank) printf("\nTime 1 %10.5e\n",t2-t1); fflush(stdout);
#endif

  map<int,vector<char> > sendmap;
  if (!myrank)
  {
    map<int,DRT::PackBuffer > sendpb;
    for (int i=0; i<size; ++i)
    {
      // proc 0 does not send to itself
      if (pidlist[i]==myrank || pidlist[i]==-1) continue;
      Node* node = gNode(oldmap.MyGlobalElements()[i]);
      if (!node) dserror("Proc 0 cannot find global node %d",oldmap.MyGlobalElements()[i]);
      node->Pack(sendpb[pidlist[i]]);
    }
    for (map<int,DRT::PackBuffer >::iterator fool = sendpb.begin(); fool != sendpb.end(); ++fool)
      fool->second.StartPacking();
    for (int i=0; i<size; ++i)
    {
      // proc 0 does not send to itself
      if (pidlist[i]==myrank || pidlist[i]==-1) continue;
      Node* node = gNode(oldmap.MyGlobalElements()[i]);
      node->Pack(sendpb[pidlist[i]]);
      node_.erase(node->Id());
    }
    for (map<int,DRT::PackBuffer >::iterator fool = sendpb.begin(); fool != sendpb.end(); ++fool)
      swap(sendmap[fool->first],fool->second());
  }

#if 0
  double t3 = timer.ElapsedTime();
  if (!myrank) printf("Time 2 %10.5e\n",t3-t2); fflush(stdout);
#endif

#ifdef PARALLEL
  // tell everybody who is to receive something
  vector<int> receivers;
  for (map<int,vector<char> >::iterator fool = sendmap.begin(); fool !=sendmap.end(); ++fool)
    receivers.push_back(fool->first);
  size = (int)receivers.size();
  Comm().Broadcast(&size,1,0);
  if (myrank != 0) receivers.resize(size);
  Comm().Broadcast(&receivers[0],size,0);
  int foundme = -1;
  if (myrank != 0)
    for (int i=0; i<size; ++i)
      if (receivers[i]==myrank)
      {
        foundme = i;
        break;
      }

  // proc 0 sends out messages
  int tag = 0;
  DRT::Exporter exporter(Comm());
  vector<MPI_Request> request(size);
  if (!myrank)
  {
    for (map<int,vector<char> >::iterator fool = sendmap.begin(); fool !=sendmap.end(); ++fool)
    {
      exporter.ISend(0,fool->first,&fool->second[0],(int)fool->second.size(),tag,request[tag]);
      tag++;
    }
    if (tag != size) dserror("Number of messages is mixed up");
    // do not delete sendmap until Wait has returned!
  }

  // all other procs listen to message and put node into dis
  if (foundme != -1)
  {
    vector<char> recvdata;
    int length = 0;
    int source = -1;
    int tag = -1;
    exporter.ReceiveAny(source,tag,recvdata,length);
    //printf("Proc %d received tag %d length %d\n",myrank,tag,length); fflush(stdout);
    if (source != 0 || tag != foundme)
      dserror("Messages got mixed up");
    // Put received nodes into discretization
    vector<char>::size_type index = 0;
    while (index < recvdata.size())
    {
      vector<char> data;
      ParObject::ExtractfromPack(index,recvdata,data);
      DRT::ParObject* object = DRT::UTILS::Factory(data);
      DRT::Node* node = dynamic_cast<DRT::Node*>(object);
      if (!node) dserror("Received object is not a node");
      node->SetOwner(myrank);
      RCP<DRT::Node> rcpnode = rcp(node);
      AddNode(rcpnode);
    }
  }


  // wait for all communication to finish
  if (!myrank)
  {
    for (int i=0; i<size; ++i)
      exporter.Wait(request[i]);
  }

#if 0
  Comm().Barrier(); // feel better this way ;-)
  double t4 = timer.ElapsedTime();
  if (!myrank) printf("Time 3 %10.5e\n",t4-t3); fflush(stdout);
#endif


#endif

  Comm().Barrier(); // feel better this way ;-)
  Reset();
  return;
}

/*----------------------------------------------------------------------*
 |  Export elements (public)                                 mwgee 11/06|
 *----------------------------------------------------------------------*/
void DRT::Discretization::ExportRowElements(const Epetra_Map& newmap)
{
  // destroy all ghosted elements
  const int myrank = Comm().MyPID();
  map<int,RCP<DRT::Element> >::iterator curr;
  for (curr=element_.begin(); curr!=element_.end();)
  {
    if (curr->second->Owner() != myrank)
    {
      element_.erase(curr++);
    }
    else
    {
      ++curr;
    }
  }

  // build map of elements elerowmap_ if it does not exist
  if (elerowmap_==null) BuildElementRowMap();
  const Epetra_Map& oldmap = *elerowmap_;

  // create an exporter object that will figure out the communication pattern
  DRT::Exporter exporter(oldmap,newmap,Comm());

  exporter.Export(element_);

  // update ownerships and kick out everything that's not in newmap
  for (curr=element_.begin(); curr!=element_.end(); ++curr)
    curr->second->SetOwner(myrank);

  // maps and pointers are no longer correct and need rebuilding
  Reset();

  return;
}

/*----------------------------------------------------------------------*
 |  Export elements (public)                                 mwgee 11/06|
 *----------------------------------------------------------------------*/
void DRT::Discretization::ExportColumnElements(const Epetra_Map& newmap)
{
  // destroy all ghosted elements
  const int myrank = Comm().MyPID();
  map<int,RefCountPtr<DRT::Element> >::iterator curr;
  for (curr=element_.begin(); curr!=element_.end();)
  {
    if (curr->second->Owner() != myrank)
    {
      element_.erase(curr++);
    }
    else
    {
      ++curr;
    }
  }

  // build map of elements elerowmap_ if it does not exist
  if (elerowmap_==null) BuildElementRowMap();
  const Epetra_Map& oldmap = *elerowmap_;

  // test whether all elements in oldmap are also in newmap
  // Otherwise, this would be a change of owner which is not allowed here
  for (int i=0; i<oldmap.NumMyElements(); ++i)
  {
    int gid = oldmap.GID(i);
    if (!(newmap.MyGID(gid))) dserror("Proc %d: Element gid=%d from oldmap is not in newmap",myrank,gid);
  }

  // create an exporter object that will figure out the communication pattern
  DRT::Exporter exporter(oldmap,newmap,Comm());
  exporter.Export(element_);

  // maps and pointers are no longer correct and need rebuilding
  Reset();

  return;
}

/*----------------------------------------------------------------------*
 |  build nodal graph from discretization (public)           mwgee 11/06|
 *----------------------------------------------------------------------*/
RefCountPtr<Epetra_CrsGraph> DRT::Discretization::BuildNodeGraph() const
{
  if (!Filled()) dserror("FillComplete() was not called on this discretization");

  // get nodal row map
  const Epetra_Map* noderowmap = NodeRowMap();

  // allocate graph
  RefCountPtr<Epetra_CrsGraph> graph =
                     rcp( new Epetra_CrsGraph(Copy,*noderowmap,108,false));

  // iterate all elements on this proc including ghosted ones
  // Note:
  // if a proc stores the appropiate ghosted elements, the resulting
  // graph will be the correct and complete graph of the distributed
  // discretization even if nodes are not ghosted.
  map<int,RefCountPtr<DRT::Element> >::const_iterator curr;
  for (curr=element_.begin(); curr!=element_.end(); ++curr)
  {
    const int  nnode   = curr->second->NumNode();
    const int* nodeids = curr->second->NodeIds();
    for (int row=0; row<nnode; ++row)
    {
      const int rownode = nodeids[row];
      if (!noderowmap->MyGID(rownode)) continue;
      for (int col=0; col<nnode; ++col)
      {
        int colnode = nodeids[col];
        int err = graph->InsertGlobalIndices(rownode,1,&colnode);
        if (err<0) dserror("graph->InsertGlobalIndices returned err=%d",err);
      }
    }
  }
  int err = graph->FillComplete();
  if (err) dserror("graph->FillComplete() returned err=%d",err);
  err = graph->OptimizeStorage();
  if (err) dserror("graph->OptimizeStorage() returned err=%d",err);
  return graph;
}


/*----------------------------------------------------------------------*
 |  build element map from discretization (public)           mwgee 11/06|
 *----------------------------------------------------------------------*/
void DRT::Discretization::BuildElementRowColumn(
                                    const Epetra_Map& noderowmap,
                                    const Epetra_Map& nodecolmap,
                                    RCP<Epetra_Map>& elerowmap,
                                    RCP<Epetra_Map>& elecolmap) const
{
  const int myrank = Comm().MyPID();
  const int numproc = Comm().NumProc();

  // note:
  // - noderowmap need not match distribution of nodes in this
  //   discretization at all.
  // - noderowmap is a non-overlapping map, that's tested
  if (!noderowmap.UniqueGIDs()) dserror("noderowmap is not a unique map");

  // find all owners for the overlapping node map
  const int ncnode = nodecolmap.NumMyElements();
  vector<int> cnodeowner(ncnode);
  int err = noderowmap.RemoteIDList(ncnode,nodecolmap.MyGlobalElements(),&cnodeowner[0],NULL);
  if (err) dserror("Epetra_BlockMap::RemoteIDLis returned err=%d",err);

  // build connectivity of elements
  // storing :  element gid
  //            no. of nodes
  //            nodeids
  int stoposize = 2000;
  int count     = 0;
  vector<int> stopo(stoposize);
  map<int,RCP<DRT::Element> >::const_iterator ecurr;
  for (ecurr=element_.begin(); ecurr!=element_.end(); ++ecurr)
  {
    const DRT::Element& actele = *(ecurr->second);
    int        gid     = actele.Id();
    int        nnode   = actele.NumNode();
    const int* nodeids = actele.NodeIds();
    if (count+nnode+2>=stoposize)
    {
      stoposize += (nnode+2)*300;
      stopo.resize(stoposize);
    }
    stopo[count++] = gid;
    stopo[count++] = nnode;
    for (int j=0; j<nnode; ++j) stopo[count++] = nodeids[j];
  }
  stoposize = count;
  stopo.resize(stoposize);

  vector<int> rtopo(stoposize);

  // estimate no. of elements equal to no. of nodes
  vector<int> myele(noderowmap.NumMyElements());
  int nummyele=0;
  // estimate no. of ghosted elements much lower
  vector<int> myghostele(noderowmap.NumMyElements()/4);
  int nummyghostele=0;

  // loop processors and sort elements into
  // elements owned by a proc
  // elements ghosted by a proc
  for (int proc=0; proc<numproc; ++proc)
  {
    int size = stoposize;
    Comm().Broadcast(&size,1,proc);
    if (size>(int)rtopo.size()) rtopo.resize(size);
    if (proc==myrank)
      for (int i=0; i<size; ++i) rtopo[i] = stopo[i];
    Comm().Broadcast(&rtopo[0],size,proc);
    for (int i=0; i<size;)
    {
      const int  elegid  = rtopo[i++];
      const int  numnode = rtopo[i++];
      const int* nodeids = &rtopo[i];
      i += numnode;

      // resize arrays
      if (nummyele>=(int)myele.size()) myele.resize(myele.size()+500);
      if (nummyghostele>=(int)myghostele.size()) myghostele.resize(myghostele.size()+500);

      // count nodes I own of this element
      int nummine=0;
      for (int j=0; j<numnode; ++j)
        if (noderowmap.MyGID(nodeids[j]))
          ++nummine;

      // if I do not own any of the nodes, it is definitely not my element
      // and I do not ghost it
      if (!nummine)
        continue;

      // check whether I ghost all nodes of this element
      // this is neccessary to be able to own or ghost the element
      for (int j=0; j<numnode; ++j)
        if (!nodecolmap.MyGID(nodeids[j]))
          dserror("I do not have own/ghosted node gid=%d",nodeids[j]);

      // find out who owns how many of the nodes
      vector<int> nodeowner(numnode);
      vector<int> numperproc(numproc);
      for (int j=0; j<numproc; ++j) numperproc[j] = 0;
      for (int j=0; j<numnode; ++j)
      {
        const int lid   = nodecolmap.LID(nodeids[j]);
        const int owner = cnodeowner[lid];
        nodeowner[j] = owner;
        numperproc[owner]++;
      }

      // the proc with the largest number of nodes owns the element,
      // all others ghost it
      // if no. of nodes is equal among some procs,
      // the last node owner with equal number of nodes owns the element
      int owner   = -1;
      int maxnode = 0;
      for (int j=0; j<numnode; ++j)
      {
        int currentproc = nodeowner[j];
        int ownhowmany  = numperproc[currentproc];
        if (ownhowmany>=maxnode)
        {
          owner   = currentproc;
          maxnode = ownhowmany;
        }
      }
      if (myrank==owner)
      {
        myele[nummyele++] = elegid;
        continue;
      }
      else
      {
        myghostele[nummyghostele++] = elegid;
        continue;
      }
      dserror("Error in logic of element ownerships");

    } // for (int i=0; i<size;)
  } // for (int proc=0; proc<numproc; ++proc)

  // at this point we have
  // myele, length nummyele
  // myghostele, length nummyghostele
  myele.resize(nummyele);
  myghostele.resize(nummyghostele);

  // allreduced nummyele must match the total no. of elements in this
  // discretization, otherwise we lost some
  // build the rowmap of elements
  elerowmap = rcp(new Epetra_Map(-1,nummyele,&myele[0],0,Comm()));
  if (!elerowmap->UniqueGIDs())
    dserror("Element row map is not unique");

  // build elecolmap
  vector<int> elecol(nummyele+nummyghostele);
  for (int i=0; i<nummyele; ++i) elecol[i] = myele[i];
  for (int i=0; i<nummyghostele; ++i) elecol[nummyele+i] = myghostele[i];
  elecolmap = rcp(new Epetra_Map(-1,nummyghostele+nummyele,
                                 &elecol[0],0,Comm()));

  return;
}

/*----------------------------------------------------------------------*
 |  redistribute discretization (public)                     mwgee 11/06|
 *----------------------------------------------------------------------*/
void DRT::Discretization::Redistribute(const Epetra_Map& noderowmap,
                                       const Epetra_Map& nodecolmap,
                                       bool assigndegreesoffreedom ,
                                       bool initelements           ,
                                       bool doboundaryconditions   )
{
  // build the overlapping and non-overlapping element maps
  RefCountPtr<Epetra_Map> elerowmap;
  RefCountPtr<Epetra_Map> elecolmap;
  BuildElementRowColumn(noderowmap,nodecolmap,elerowmap,elecolmap);

  // export nodes and elements to the new maps
  ExportRowNodes(noderowmap);
  ExportColumnNodes(nodecolmap);
  ExportRowElements(*elerowmap);
  ExportColumnElements(*elecolmap);

  // these exports have set Filled()=false as all maps are invalid now
  int err = FillComplete(assigndegreesoffreedom,initelements,doboundaryconditions);

  if (err) dserror("FillComplete() returned err=%d",err);

  return;
}


/*----------------------------------------------------------------------*
// this is to go away!!!!
 *----------------------------------------------------------------------*/
void DRT::Discretization::SetupGhostingWrongNameDoNotUse(
                                        bool assigndegreesoffreedom ,
                                        bool initelements           ,
                                        bool doboundaryconditions   )
{
  if (Filled())
    dserror("there is really no need to setup ghosting if the discretization is already filled");

  // build the graph ourselves
  std::map<int,std::set<int> > localgraph;
  for (std::map<int,RCP<DRT::Element> >::iterator i=element_.begin();
       i!=element_.end();
       ++i)
  {
    int numnodes = i->second->NumNode();
    const int* nodes = i->second->NodeIds();

    // loop nodes and add this topology to the row in the graph of every node
    for (int n=0; n<numnodes; ++n)
    {
      int nodelid = nodes[n];
      copy(nodes,
           nodes+numnodes,
           inserter(localgraph[nodelid],
                    localgraph[nodelid].begin()));
    }
  }

  // Create node row map. Only the row nodes go there.

  std::vector<int> gids;
  std::vector<int> entriesperrow;

  gids.reserve(localgraph.size());
  entriesperrow.reserve(localgraph.size());

  for (std::map<int,RCP<DRT::Node> >::iterator i=node_.begin();
       i!=node_.end();
       ++i)
  {
    gids.push_back(i->first);
    entriesperrow.push_back(localgraph[i->first].size());
  }

  Epetra_Map rownodes(-1,gids.size(),&gids[0],0,*comm_);

  // Construct FE graph. This graph allows processor off-rows to be inserted
  // as well. The communication issue is solved.

  Teuchos::RCP<Epetra_FECrsGraph> graph = rcp(new Epetra_FECrsGraph(Copy,rownodes,&entriesperrow[0],false));

  gids.clear();
  entriesperrow.clear();

  // Insert all rows into the graph, including the off ones.

  for (std::map<int,std::set<int> >::iterator i=localgraph.begin();
       i!=localgraph.end();
       ++i)
  {
    set<int>& rowset = i->second;
    vector<int> row;
    row.reserve(rowset.size());
    row.assign(rowset.begin(),rowset.end());
    rowset.clear();

    int err = graph->InsertGlobalIndices(1,&i->first,row.size(),&row[0]);
    if (err<0) dserror("graph->InsertGlobalIndices returned %d",err);
  }

  localgraph.clear();

  // Finalize construction of this graph. Here the communication
  // happens. The ghosting problem is solved at this point.

  int err = graph->GlobalAssemble(rownodes,rownodes);
  if (err) dserror("graph->GlobalAssemble returned %d",err);

  // partition graph using metis
  Epetra_Vector weights(graph->RowMap(),false);
  weights.PutScalar(1.0);
  Teuchos::RCP<Epetra_CrsGraph> gr = DRT::UTILS::PartGraphUsingMetis(*graph,weights);
  graph = Teuchos::null;

  // replace rownodes, colnodes with row and column maps from the graph
  // do stupid conversion from Epetra_BlockMap to Epetra_Map
  const Epetra_BlockMap& brow = gr->RowMap();
  const Epetra_BlockMap& bcol = gr->ColMap();
  RCP<Epetra_Map> noderowmap = rcp(new Epetra_Map(brow.NumGlobalElements(),
                                                  brow.NumMyElements(),
                                                  brow.MyGlobalElements(),
                                                  0,
                                                  *comm_));
  RCP<Epetra_Map> nodecolmap = rcp(new Epetra_Map(bcol.NumGlobalElements(),
                                                  bcol.NumMyElements(),
                                                  bcol.MyGlobalElements(),
                                                  0,
                                                  *comm_));

  gr = Teuchos::null;

  // Redistribute discretization to match the new maps.

  Redistribute(*noderowmap,
               *nodecolmap,
               assigndegreesoffreedom,
               initelements,
               doboundaryconditions);

}


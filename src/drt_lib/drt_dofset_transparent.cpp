/*!----------------------------------------------------------------------
\file drt_dofset_transparent.cpp

\brief A set of degrees of freedom special for contact

<pre>
Maintainer: Thomas Kloeppel
            kloeppel@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15257
</pre>

*----------------------------------------------------------------------*/
#ifdef CCADISCRET

#include "drt_dofset_transparent.H"
#include "../linalg/linalg_utils.H"

DRT::TransparentDofSet::TransparentDofSet(RCP<DRT::Discretization> sourcedis,bool parallel) :
DRT::DofSet(),
sourcedis_(sourcedis),
parallel_(parallel)
{
  return;
}

int DRT::TransparentDofSet::AssignDegreesOfFreedom(const DRT::Discretization& dis, const unsigned dspos, const int start)
{

  // first, we call the standard AssignDegreesOfFreedom from the base class
  int count = DRT::DofSet::AssignDegreesOfFreedom(dis,dspos,start);

  if(!parallel_)
  {
    TransferDegreesOfFreedom(*sourcedis_, dis, start);
  }
  else
  {
    ParallelTransferDegreesOfFreedom(*sourcedis_, dis, start);
  }

  // tell all proxies (again!)
  NotifyAssigned();

  return count;
}

/// Assign dof numbers for new discretization using dof numbering from source discretization.
void DRT::TransparentDofSet::TransferDegreesOfFreedom(
        const DRT::Discretization& sourcedis,
        const DRT::Discretization& newdis,
        const int start
        )
{
    if (!sourcedis.DofRowMap()->UniqueGIDs()) dserror("DofRowMap is not unique");
    if (!sourcedis.NodeRowMap()->UniqueGIDs()) dserror("NodeRowMap is not unique");
    if (!sourcedis.ElementRowMap()->UniqueGIDs()) dserror("ElementRowMap is not unique");

    if (!newdis.DofRowMap()->UniqueGIDs()) dserror("DofRowMap is not unique");
    if (!newdis.NodeRowMap()->UniqueGIDs()) dserror("NodeRowMap is not unique");
    if (!newdis.ElementRowMap()->UniqueGIDs()) dserror("ElementRowMap is not unique");

    //build dofrowmap
    set<int> dofrowset;
    vector<int> dofrowvec;
    dofrowvec.reserve(dofrowmap_->NumMyElements());
    for (int inode = 0; inode != newdis.NumMyRowNodes(); ++inode)
    {
      const DRT::Node* newnode = newdis.lRowNode(inode);
      const DRT::Node* sourcenode = sourcedis.gNode(newnode->Id());

      const vector<int> dofs = sourcedis.Dof(sourcenode);

      const int newlid = newnode->LID();
      const int numdofs = (*numdfcolnodes_)[newlid];
      if (numdofs > 0)
      {
        (*idxcolnodes_)[newlid] = dofs[0];
        for (int idof = 0; idof < numdofs; ++idof)
        {
          dofrowset.insert(dofs[idof]);
        }
      }
    }

    for(set<int>::iterator idof=dofrowset.begin();
        idof!=dofrowset.end();++idof)
    {
      dofrowvec.push_back(*idof);
    }

    dofrowmap_ = rcp(new Epetra_Map(-1, dofrowvec.size(), &dofrowvec[0], 0, newdis.Comm()));

    //build dofcolvec
    set<int> dofcolset;
    vector<int> dofcolvec;
    dofcolvec.reserve(dofcolmap_->NumMyElements());
    for (int inode = 0; inode != newdis.NumMyColNodes(); ++inode)
    {
      const DRT::Node* newnode = newdis.lColNode(inode);
      const DRT::Node* sourcenode = sourcedis.gNode(newnode->Id());

      const int lid = sourcenode->LID();
      if (lid==-1)
      {
        dserror("required node %d not on proc",newnode->Id());
      }
      const vector<int> dofs = sourcedis.Dof(sourcenode);
      const int newlid = newnode->LID();
      //const int newfirstidx = (*idxcolnodes_)[newlid];
      const int numdofs = (*numdfcolnodes_)[newlid];
      if (numdofs > 0)
      {
        (*idxcolnodes_)[newlid] = dofs[0];
//        if(numdofs!=(int)dofs.size())
//        dserror("numdofs %d!=%d for node %d",numdofs,(int)dofs.size(),newnode->Id());

        if(dofs[0]==0)
        {
          printf("dofs %d %d %d %d on PID %d, vector size %d\n",dofs[0],dofs[1],dofs[2],dofs[3],sourcedis.Comm().MyPID(),(int)dofs.size());
        }

        for (int idof = 0; idof < numdofs; ++idof)
        {
          dofcolset.insert(dofs[idof]);
        }
      }
    }

    for(set<int>::iterator idof=dofcolset.begin();
        idof!=dofcolset.end();++idof)
    {
      dofcolvec.push_back(*idof);
    }

    dofcolmap_ = rcp(new Epetra_Map(-1, dofcolvec.size(), &dofcolvec[0], 0, newdis.Comm()));
}

/// Assign dof numbers for new discretization using dof numbering from source discretization.
void DRT::TransparentDofSet::ParallelTransferDegreesOfFreedom(
  const DRT::Discretization& sourcedis,
  const DRT::Discretization& newdis,
  const int start
  )
{
  if (!sourcedis.DofRowMap()->UniqueGIDs()) dserror("DofRowMap is not unique");
  if (!sourcedis.NodeRowMap()->UniqueGIDs()) dserror("NodeRowMap is not unique");
  if (!sourcedis.ElementRowMap()->UniqueGIDs()) dserror("ElementRowMap is not unique");

  if (!newdis.DofRowMap()->UniqueGIDs()) dserror("DofRowMap is not unique");
  if (!newdis.NodeRowMap()->UniqueGIDs()) dserror("NodeRowMap is not unique");
  if (!newdis.ElementRowMap()->UniqueGIDs()) dserror("ElementRowMap is not unique");

  // list all my rownode ids

  //
  // we need a mapping
  //
  // colnode gid -> vector<int> dofs of sourcenode
  //
  // problem: sourcenode not necessarily on this proc -> communicate
  //
  // the idea is to search for the sourcerownode on some proc and to get
  // this unique number
  //
  map<int,vector<int> >  gid_to_dofs;

  for (int inode = 0; inode != newdis.NumMyColNodes(); ++inode)
  {
    const DRT::Node* newnode = newdis.lColNode(inode);
    int gid=newnode->Id();
    vector<int> emptyvec;
    gid_to_dofs.insert(pair<int, vector<int> >(gid,emptyvec));
  }

  {
#ifdef PARALLEL
    // create an exporter for point to point comunication
    DRT::Exporter exporter(sourcedis.Comm());

    // necessary variables
    MPI_Request request;
#endif

    // define send and receive blocks
    vector<char> sblock;
    vector<char> rblock;

    // get number of processors and the current processors id
    int numproc=sourcedis.Comm().NumProc();
    int myrank=sourcedis.Comm().MyPID();

    //----------------------------------------------------------------------
    // communication is done in a round robin loop
    //----------------------------------------------------------------------
    for (int np=0;np<numproc+1;++np)
    {
      // in the first step, we cannot receive anything
      if(np >0)
      {
#ifdef PARALLEL
        ReceiveBlock(numproc,myrank,rblock,exporter,request);
#else
        rblock=sblock;
#endif

        // Unpack info from the receive block from the last proc
        UnpackLocalSourceDofs(gid_to_dofs,rblock);
      }

      // in the last step, we keep everything on this proc
      if(np < numproc)
      {
        // -----------------------
        // do what we wanted to do
        SetSourceDofsAvailableOnThisProc(gid_to_dofs);

        // Pack info into block to send
        PackLocalSourceDofs(gid_to_dofs,sblock);

#ifdef PARALLEL
        SendBlock(numproc,myrank,sblock,exporter,request);
#endif
      }
    }
  }

  set<int> slaveset;
  std::vector<DRT::Condition*> mypbcs;

  // get periodic surface boundary conditions
  sourcedis_->GetCondition("SurfacePeriodic",mypbcs);

  if(mypbcs.empty())
  {
    sourcedis_->GetCondition("LinePeriodic",mypbcs);
  }

  for (unsigned numcond=0;numcond<mypbcs.size();++numcond)
  {
    DRT::Condition* thiscond= mypbcs[numcond];

    // see whether we have a slave condition
    const string* mymasterslavetoggle
      = thiscond->Get<string>("Is slave periodic boundary condition");

    if(!(*mymasterslavetoggle=="Master"))
    {

      const vector <int>* pbcids;
      pbcids = (*thiscond).Nodes();

      for(vector<int>::const_iterator iter=pbcids->begin();iter!=pbcids->end();++iter)
      {
        slaveset.insert(*iter);
      }
    }
  }

  //build dofrowmap
  set<int> dofrowset;
  vector<int> dofrowvec;
  dofrowvec.reserve(dofrowmap_->NumMyElements());
  for (int inode = 0; inode != newdis.NumMyRowNodes(); ++inode)
  {
    const DRT::Node* newnode = newdis.lRowNode(inode);

    const vector<int> dofs = gid_to_dofs[newnode->Id()];

    const int newlid = newnode->LID();
    const int numdofs = (*numdfcolnodes_)[newlid];

    if(numdofs!=(int)dofs.size())
    {
      dserror("spooky, isn't it?");
    }

    if (numdofs > 0)
    {
      (*idxcolnodes_)[newlid] = dofs[0];

      // slave-dofs must not enter the dofrowset (if master&slave are on different procs)
      std::set<int>::iterator curr=slaveset.find(newnode->Id());

      if(curr==slaveset.end())
      {
        for (int idof = 0; idof < numdofs; ++idof)
        {
          dofrowset.insert(dofs[idof]);
        }
      }
    }
  }

  for(set<int>::iterator idof=dofrowset.begin();
      idof!=dofrowset.end();++idof)
  {
    dofrowvec.push_back(*idof);
  }

  dofrowmap_ = rcp(new Epetra_Map(-1, dofrowvec.size(), &dofrowvec[0], 0, newdis.Comm()));

  //build dofcolvec
  set<int> dofcolset;
  vector<int> dofcolvec;
  dofcolvec.reserve(dofcolmap_->NumMyElements());
  for (int inode = 0; inode != newdis.NumMyColNodes(); ++inode)
  {
    const DRT::Node* newnode = newdis.lColNode(inode);

    const vector<int> dofs = gid_to_dofs[newnode->Id()];

    const int newlid = newnode->LID();
    //const int newfirstidx = (*idxcolnodes_)[newlid];
    const int numdofs = (*numdfcolnodes_)[newlid];
    if (numdofs > 0)
    {
      (*idxcolnodes_)[newlid] = dofs[0];
      if(numdofs!=(int)dofs.size())
      dserror("numdofs %d!=%d for node %d",numdofs,dofs.size(),newnode->Id());

      for (int idof = 0; idof < numdofs; ++idof)
      {
        dofcolset.insert(dofs[idof]);
      }
    }
  }

  for(set<int>::iterator idof=dofcolset.begin();
      idof!=dofcolset.end();++idof)
  {
    dofcolvec.push_back(*idof);
  }

  dofcolmap_ = rcp(new Epetra_Map(-1, dofcolvec.size(), &dofcolvec[0], 0, newdis.Comm()));


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
void DRT::TransparentDofSet::SetSourceDofsAvailableOnThisProc(
  map<int,vector<int> >   & gid_to_dofs
  )
{
  for(map<int,vector<int> >::iterator curr=gid_to_dofs.begin();
    curr!=gid_to_dofs.end();++curr)
  {

    const int lid = sourcedis_->NodeRowMap()->LID(curr->first);

    if (lid>-1)
    {
      curr->second.clear();

      const DRT::Node* sourcenode = sourcedis_->gNode(curr->first);

      const vector<int> dofs = sourcedis_->Dof(sourcenode);

      for (vector<int>::const_iterator iter=dofs.begin();iter!=dofs.end();++iter)
      {
        curr->second.push_back(*iter);
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
void DRT::TransparentDofSet::PackLocalSourceDofs(
  map<int,vector<int> >   & gid_to_dofs  ,
  vector<char>            & sblock
  )
{
  sblock.clear();

  int size=gid_to_dofs.size();

  // add size  to sendblock
  DRT::ParObject::AddtoPack<int>(sblock,size);

  for(map<int,vector<int> >::iterator curr=gid_to_dofs.begin();
    curr!=gid_to_dofs.end();++curr)
  {
    int         gid    = curr->first;
    vector<int> mydofs = curr->second;
    int numdofs = (int)mydofs.size();

    DRT::ParObject::AddtoPack<int>(sblock,gid);
    DRT::ParObject::AddtoPack<int>(sblock,numdofs);
    for(int ll=0;ll<numdofs;++ll)
    {
      DRT::ParObject::AddtoPack<int>(sblock,mydofs[ll]);
    }
  }

  gid_to_dofs.clear();

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
void DRT::TransparentDofSet::UnpackLocalSourceDofs(
  map<int,vector<int> >   & gid_to_dofs  ,
  vector<char>            & rblock
  )
{
  gid_to_dofs.clear();

  // position to extract
  vector<char>::size_type position = 0;

  // extract size
  int size=0;
  DRT::ParObject::ExtractfromPack<int>(position,rblock,size);

  for(int rr=0;rr<size;++rr)
  {
    int         gid    = -1;
    vector<int> mydofs ;
    int numdofs = 0;

    DRT::ParObject::ExtractfromPack<int>(position,rblock,gid);
    DRT::ParObject::ExtractfromPack<int>(position,rblock,numdofs);

    for(int ll=0;ll<numdofs;++ll)
    {
      int thisdof=0;

      DRT::ParObject::ExtractfromPack<int>(position,rblock,thisdof);
      mydofs.push_back(thisdof);
    }

    gid_to_dofs.insert(pair<int, vector<int> >(gid,mydofs));
  }

  rblock.clear();
  return;
}

#ifdef PARALLEL
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 | receive a block in the round robin communication pattern   (private) |
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void DRT::TransparentDofSet::ReceiveBlock(
  int             numproc ,
  int             myrank  ,
  vector<char>   & rblock,
  DRT::Exporter  & exporter,
  MPI_Request    & request)
{

  // necessary variables

  int         length =-1;
  int         frompid=(myrank+numproc-1)%numproc;
  int         tag    =frompid;

  // make sure that you do not think you received something if
  // you didn't
  if(rblock.empty()==false)
  {
    dserror("rblock not empty");
  }

  // receive from predecessor
  exporter.ReceiveAny(frompid,tag,rblock,length);

  if(tag!=(myrank+numproc-1)%numproc)
  {
    dserror("received wrong message (ReceiveAny)");
  }

  exporter.Wait(request);

  // for safety
  exporter.Comm().Barrier();

  return;
} // ReceiveBlock
#endif



#ifdef PARALLEL
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 | send a block in the round robin communication pattern      (private) |
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void DRT::TransparentDofSet::SendBlock(
  int             numproc ,
  int             myrank  ,
  vector<char>  & sblock  ,
  DRT::Exporter & exporter,
  MPI_Request   & request )
{

  // Send block to next proc.
  int         tag    =myrank;
  int         frompid=myrank;
  int         topid  =(myrank+1)%numproc;

  exporter.ISend(frompid,topid,
                 &(sblock[0]),sblock.size(),
                 tag,request);


  // for safety
  exporter.Comm().Barrier();

  return;
} //SendBlock
#endif


#endif  // #ifdef CCADISCRET

/*!----------------------------------------------------------------------
\file drt_turbulent_inflow.H

\brief Methods to transfer a turbulent inflow profile from a (usually
periodic boundary condition) separate domain to a name Dirichlet
boundary of the actual domain

<pre>
Maintainer: Ursula Rasthofer
            rasthofer@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15236
</pre>

*----------------------------------------------------------------------*/
#include "drt_transfer_turb_inflow.H"
#include "../drt_lib/drt_nodematchingoctree.H"
#include "../drt_lib/drt_timecurve.H"
#include "../drt_lib/drt_globalproblem.H"
#include "../drt_lib/drt_discret.H"



//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 |  Constructor (public)                                     gammi 03/10|
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
FLD::TransferTurbulentInflowCondition::TransferTurbulentInflowCondition(
  RCP<DRT::Discretization>  dis    ,
  RCP<LINALG::MapExtractor> dbcmaps
    )
  : dis_(dis),
    dbcmaps_(dbcmaps),
    curve_(-1)
{
  active_=false;

  // vector of pointers to all node clouds i.e. conditions to couple
  std::vector<DRT::Condition*>     nodecloudstocouple;

  // get surfaces to couple
  dis_->GetCondition("TransferTurbulentInflow",nodecloudstocouple);

  if(not nodecloudstocouple.empty())
  {
    // activate
    active_=true;

    // master and slave sets to couple
    std::set<int> masterset;
    std::set<int> slaveset;

    // global master node Ids and global slave node Ids
    vector <int> masternodeids;
    vector <int> slavenodeids;

    // the (at the moment) one and only direction to couple
    int dir=-1;

    // loop all conditions and check whether they are of master or slave
    // type
    for(vector<DRT::Condition*>::iterator cond=nodecloudstocouple.begin();
        cond!=nodecloudstocouple.end();
        ++cond)
    {
      // get id, direction info and toggle
      int        id       =-1;
      int        direction=-1;
      ToggleType toggle   =none;

      GetData(id,direction,toggle,*cond);

      if(dir==-1)
      {
        dir=direction;
      }
      else
      {
        if(dir!=direction)
        {
          dserror("multiple directions are not supported yet");
        }
      }

      if(id!=0)
      {
        dserror("expecting only one group of coupling surfaces (up to now), its %d",id);
      }

      switch (toggle)
      {
      case master:
      {
        //--------------------------------------------------
        // get global master node Ids
        const vector <int>* masteridstoadd;

        masteridstoadd = (*cond)->Nodes();

        for(vector<int>::const_iterator idtoadd =(*masteridstoadd).begin();
            idtoadd!=(*masteridstoadd).end();
            ++idtoadd)
        {
          masterset.insert(*idtoadd);
        }

        break;
      }
      case slave:
      {
        //--------------------------------------------------
        // get global slave node Ids
        const vector <int>* slaveidstoadd;

        slaveidstoadd = (*cond)->Nodes();

        for(vector<int>::const_iterator idtoadd =(*slaveidstoadd).begin();
            idtoadd!=(*slaveidstoadd).end();
            ++idtoadd)
        {
          slaveset.insert(*idtoadd);
        }

        break;
      }
      default :
        dserror("toggle non master or slave");
      }

    }

    //--------------------------------------------------
    // just write sets into vectors
    (masternodeids).clear();
    (slavenodeids ).clear();

    for(std::set<int>::iterator appendednode = masterset.begin();
        appendednode != masterset.end();
        ++appendednode)
    {
      masternodeids.push_back(*appendednode);
    }

    for(std::set<int>::iterator appendednode = slaveset.begin();
        appendednode != slaveset.end();
        ++appendednode)
    {
      slavenodeids.push_back(*appendednode);
    }

    // these are just parameter definitions for the octree search algorithm
    double tol            = 1e-6;
    int    maxnodeperleaf = 250;
    double rotangle       = 0.0;

    vector<int> dofsforpbcplane(2);
    {
      int mm=0;
      for(int rr=0;rr<3;++rr)
      {
        if(rr!=dir)
        {
          dofsforpbcplane[mm]=rr;
          ++mm;
        }
      }
    }

    // build processor local octree
    DRT::UTILS::NodeMatchingOctree nodematchingoctree(
      *dis_         ,
      masternodeids ,
      maxnodeperleaf,
      tol
      );

    // create map from gid masternode -> gid corresponding slavenode
    nodematchingoctree.CreateGlobalNodeMatching(
        slavenodeids   ,
        dofsforpbcplane,
        rotangle,
        midtosid_
    );
    // sanity check
    for (std::map<int,std::vector<int> >::iterator pair=midtosid_.begin();
         pair!=midtosid_.end();
         ++pair)
    {
      if(pair->second.size()!=1)
      {
        dserror("expected one node to match, got %d out of %d",
        pair->second.size(),
        slavenodeids.size());
      }
    }
  }
  return;
} // TransferTurbulentInflowCondition::TransferTurbulentInflowCondition

//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 |  Perform transfer process (public)                        gammi 03/10|
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void FLD::TransferTurbulentInflowCondition::Transfer(
  const Teuchos::RCP<Epetra_Vector> veln ,
  Teuchos::RCP<Epetra_Vector>       velnp,
  const double                      time)
{
  const Epetra_Map* dofrowmap = dis_->DofRowMap();

  vector<int>             mymasters;
  vector<std::vector<double> > mymasters_vel(3);

  if(active_)
  {

      // initialization of time curve factor
      double curvefac    = 1.0;
      if (curve_ >= 0) // yes, we have a time curve
      {
        // time factor for the intermediate step
        if(time >= 0.0)
        {
          curvefac = DRT::Problem::Instance()->Curve(curve_).f(time);
        }
        else
        {
          // do not compute an "alternative" curvefac here since a negative time value
          // indicates an error.
          dserror("Negative time value: time = %f",time);
        }
      }
      else // we do not have a time curve --- time curve factor is constant equal 1
      {
        // nothing to do
      }

    // collect masters on this proc and associated velocities
    for (std::map<int,std::vector<int> >::iterator pair=midtosid_.begin();
         pair!=midtosid_.end();
         ++pair)
    {
      int gid=pair->first;

      if(dis_->HaveGlobalNode(gid))
      {
        mymasters.push_back(gid);

        DRT::Node* master=dis_->gNode(gid);

        vector<int> masterdofs=dis_->Dof(master);

        for(int rr=0;rr<3;++rr)
        {
          int lid = dofrowmap->LID(masterdofs[rr]);

          (mymasters_vel[rr]).push_back(((*veln)[lid])*curvefac);
        }
      }
      else
      {
	dserror("master %d in midtosid but not on proc. This was unexpected",gid);
      }
    }

#ifdef PARALLEL
    // create an exporter for point to point comunication
    DRT::Exporter exporter(dis_->Comm());

    // necessary variables
    MPI_Request request;
#endif

    // define send and receive blocks
    vector<char> sblock;
    vector<char> rblock;

    // get number of processors and the current processors id
    int numproc=dis_->Comm().NumProc();

    //----------------------------------------------------------------------
    // communication is done in a round robin loop
    //----------------------------------------------------------------------
    for (int np=0;np<numproc+1;++np)
    {
      // in the first step, we cannot receive anything
      if(np >0)
      {
#ifdef PARALLEL
        ReceiveBlock(rblock,exporter,request);
#else
        rblock=sblock;
#endif

        // Unpack info from the receive block from the last proc
        UnpackLocalMasterValues(mymasters,mymasters_vel,rblock);
      }

      // in the last step, we keep everything on this proc
      if(np < numproc)
      {
        // -----------------------
        // do what we wanted to do
        SetValuesAvailableOnThisProc(mymasters,mymasters_vel,velnp);

        // Pack info into block to send
        DRT::PackBuffer data;
        PackLocalMasterValues(mymasters,mymasters_vel,data);
        data.StartPacking();
        PackLocalMasterValues(mymasters,mymasters_vel,data);
        swap( sblock, data() );

#ifdef PARALLEL
        SendBlock(sblock,exporter,request);
#endif
      }
    }
  }
  return;
} // Transfer

//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 | Destructor dtor  (public)                                 gammi 03/10|
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
FLD::TransferTurbulentInflowCondition::~TransferTurbulentInflowCondition()
{
  return;
}// ~TransferTurbulentInflowCondition

//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 | get condition id etc (private)                            gammi 03/10|
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void FLD::TransferTurbulentInflowCondition::GetData(
  int                 & id       ,
  int                 & direction,
  ToggleType          & type     ,
  const DRT::Condition* cond     )
{

  const std::vector<int>* myid = cond->Get<std::vector<int> >("id");
  id=(*myid)[0];

  const string* mydirection = cond->Get<string>("transfer direction");
  if      (*mydirection == "x")
  {
    direction=0;
  }
  else if (*mydirection == "y")
  {
    direction=1;
  }
  else if (*mydirection == "z")
  {
    direction=2;
  }
  else
  {
    dserror("unknown direction");
  }

  const string* mytoggle = cond->Get<string>("toggle");
  if      (*mytoggle == "master")
  {
    type=master;
  }
  else if (*mytoggle == "slave")
  {
    type=slave;
  }
  else
  {
    dserror("expecting either master or slave");
  }

  // find out whether we will use a time curve
  if (curve_ == -1)
  {
    const std::vector<int>* curve = cond->Get<std::vector<int> >("curve");

    // set curve number
    if (curve) curve_ = (*curve)[0];
  }

  return;
}


#ifdef PARALLEL
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 | receive a block in the round robin communication pattern   (private) |
 |                                                          gammi 03/10 |
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void FLD::TransferTurbulentInflowCondition::ReceiveBlock(
    vector<char>   & rblock,
    DRT::Exporter  & exporter,
    MPI_Request    & request)
{
  // get number of processors and the current processors id
  int numproc=dis_->Comm().NumProc();
  int myrank =dis_->Comm().MyPID();

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
} // TransferTurbulentInflowCondition::ReceiveBlock
#endif



#ifdef PARALLEL
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 | send a block in the round robin communication pattern      (private) |
 |                                                          gammi 03/10 |
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void FLD::TransferTurbulentInflowCondition::SendBlock(
    vector<char>  & sblock  ,
    DRT::Exporter & exporter,
    MPI_Request   & request )
{
  // get number of processors and the current processors id
  int numproc=dis_->Comm().NumProc();
  int myrank =dis_->Comm().MyPID();

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
} // TransferTurbulentInflowCondition::SendBlock
#endif



//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 | unpack all master values contained in receive block        (private) |
 |                                                          gammi 03/10 |
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void FLD::TransferTurbulentInflowCondition::UnpackLocalMasterValues(
  vector<int>             & mymasters    ,
  vector<std::vector<double> > & mymasters_vel,
  vector<char>            & rblock
  )
{
  mymasters.clear();

  midtosid_.clear();

  if((int)mymasters_vel.size()!=3)
  {
    dserror("expecting three spatial dimensions in mymasters_vel to unpack into");
  }

  for(int rr=0;rr<3;++rr)
  {
    (mymasters_vel[rr]).clear();
  }

  // position to extract
  std::vector<char>::size_type position = 0;

  // extract size
  int size=0;
  DRT::ParObject::ExtractfromPack(position,rblock,size);

  // extract master ids
  for(int i=0;i<size;++i)
  {
    int id;

    DRT::ParObject::ExtractfromPack(position,rblock,id);
    mymasters.push_back(id);

    map<int,std::vector<int> >::iterator iter=midtosid_.find(id);

    if(iter!=midtosid_.end())
    {
      iter->second.clear();
    }
    else
    {
      midtosid_[id].clear();
    }
  }

  // extract slave ids
  for(int rr=0;rr<size;++rr)
  {
    int slavesize;

    DRT::ParObject::ExtractfromPack(position,rblock,slavesize);

    for(int ll=0;ll<slavesize;++ll)
    {
      int sid;
      DRT::ParObject::ExtractfromPack(position,rblock,sid);

      map<int,std::vector<int> >::iterator iter=midtosid_.find(mymasters[rr]);

      if(iter!=midtosid_.end())
      {
	iter->second.push_back(sid);
      }
      else
      {
	dserror("master id %d was not in midtosid_",mymasters[rr]);
      }
    }

    if(midtosid_[mymasters[rr]].size()<1)
    {
      dserror("require at least one slave to master %d, got %d",mymasters[rr],midtosid_[mymasters[rr]].size());
    }
  }

  // extract values (first u, then v, then w)
  for(int mm=0;mm<3;++mm)
  {
    for(int rr=0;rr<size;++rr)
    {
      double value;

      DRT::ParObject::ExtractfromPack(position,rblock,value);

      (mymasters_vel[mm]).push_back(value);
    }
  }

  rblock.clear();
  return;
} // UnpackLocalMasterValues



//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 | pack all master values into a send block                   (private) |
 |                                                          gammi 03/10 |
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void FLD::TransferTurbulentInflowCondition::PackLocalMasterValues(
    vector<int>             & mymasters    ,
    vector<std::vector<double> > & mymasters_vel,
    DRT::PackBuffer         & sblock
    )
{
  int size=mymasters.size();

  if(mymasters_vel.size()!=3)
  {
    dserror("expecting three spatial dimensions in mymasters_vel to pack");
  }

  for(int rr=0;rr<3;++rr)
  {
    if((int)(mymasters_vel[rr]).size()!=size)
    {
      dserror("at least one of the components of mymasters_vel has the wrong size");
    }
  }

  // add size  to sendblock
  DRT::ParObject::AddtoPack(sblock,size);

  // add master ids
  for(int rr=0;rr<size;++rr)
  {
    DRT::ParObject::AddtoPack(sblock,mymasters[rr]);
  }

  // add slave ids
  for(int rr=0;rr<size;++rr)
  {
    map<int,std::vector<int> >::iterator iter=midtosid_.find(mymasters[rr]);

    if(iter==midtosid_.end())
    {
      dserror("tried to pack slaves to master master %d, got none",mymasters[rr]);
    }
    else
    {
      vector<int> slaves=iter->second;

      int slavesize=(int)slaves.size();

      DRT::ParObject::AddtoPack(sblock,slavesize);
      for(int ll=0;ll<slavesize;++ll)
      {
	DRT::ParObject::AddtoPack(sblock,slaves[ll]);
      }
    }
  }

  // add values (first u, then v, then w)
  for(int mm=0;mm<3;++mm)
  {
    for(int rr=0;rr<size;++rr)
    {
      DRT::ParObject::AddtoPack(sblock,(mymasters_vel[mm])[rr]);
    }
  }

  return;
} // PackLocalMasterValues



//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 | for all values avaible on the processor, do the final setting of the |
 | value                          (private)                 gammi 03/10 |
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void FLD::TransferTurbulentInflowCondition::SetValuesAvailableOnThisProc(
    vector<int>                 & mymasters,
    vector<std::vector<double> >     & mymasters_vel,
    Teuchos::RCP<Epetra_Vector>   velnp)
{
  const Teuchos::RCP<const Epetra_Map> activedbcdofs=dbcmaps_->CondMap();

  for(unsigned nn=0;nn<mymasters.size();++nn)
  {
    map<int,std::vector<int> >::iterator iter=midtosid_.find(mymasters[nn]);

    if(iter!=midtosid_.end())
    {
      vector<int> myslaves(iter->second);

      for(vector<int>::iterator sid=myslaves.begin();sid!=myslaves.end();++sid)
      {
        // is this slave id on this proc?
        if(dis_->NodeRowMap()->MyGID(*sid))
        {
          DRT::Node* slave=dis_->gNode(*sid);

          // get dofs
          vector<int> slavedofs=dis_->Dof(slave);

	  for(int rr=0;rr<3;++rr)
          {
            int gid = slavedofs[rr];

	    // only set if DBC is active, otherwise throw error
	    if(activedbcdofs->MyGID(gid))
	    {
	      double value=(mymasters_vel[rr])[nn];

	      velnp->ReplaceGlobalValues(1,&value,&gid);
	    }
	    else
	    {
	      int    id=slave->Id();

	      double x=slave->X()[0];
	      double y=slave->X()[1];
	      double z=slave->X()[2];

	      dserror("Dirichlet condition required on slave node (%12.5e,%12.5e,%12.5e), id %d, dof %d of transfer condition",x,y,z,id,rr);
	    }
          }
        }
      }
    }
  }

  return;
} // SetValuesAvailableOnThisProc



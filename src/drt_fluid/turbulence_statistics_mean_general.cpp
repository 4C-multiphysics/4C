/*
         Computation of mean values of nodal/cp quantities.

<pre>

Maintainer: Peter Gamnitzer
            gamnitzer@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15235

</pre>

*/

#ifdef CCADISCRET

#include "turbulence_statistics_mean_general.H"

//----------------------------------------------------------------------
//
//                                 Constructor
//
//----------------------------------------------------------------------
FLD::TurbulenceStatisticsGeneralMean::TurbulenceStatisticsGeneralMean(
  RCP<DRT::Discretization> discret        ,
  string                   homdir         ,
  double &                 density        ,
  LINALG::MapExtractor&    velpressplitter,
  const bool               withscatra
  )
  :
  discret_(discret),
  density_(density),
  velpressplitter_(velpressplitter),
  withscatra_(withscatra)
{
  // get directions to do spatial averaging
  homdir_.clear();

  if(homdir=="xy")
  {
    homdir_.push_back(0);
    homdir_.push_back(1);
  }
  else if(homdir=="xz")
  {
    homdir_.push_back(0);
    homdir_.push_back(2);
  }
  else if(homdir=="yz")
  {
    homdir_.push_back(1);
    homdir_.push_back(2);
  }
  else if(homdir=="x")
  {
    homdir_.push_back(0);
  }
  else if(homdir=="y")
  {
    homdir_.push_back(1);
  }
  else if(homdir=="z")
  {
    homdir_.push_back(2);
  }

  // initialise all counters, timers and vectors to zero
  ResetComplete();

  return;
} // FLD::TurbulenceStatisticsGeneralMean::TurbulenceStatisticsGeneralMean

//----------------------------------------------------------------------
//
//                                 Destructor
//
//----------------------------------------------------------------------
FLD::TurbulenceStatisticsGeneralMean::~TurbulenceStatisticsGeneralMean()
{
  return;
} // FLD::TurbulenceStatisticsGeneralMean::~TurbulenceStatisticsGeneralMean


//----------------------------------------------------------------------
//
//                    Add vector to current time average
//
//----------------------------------------------------------------------
void FLD::TurbulenceStatisticsGeneralMean::AddToCurrentTimeAverage(
  const double             dt ,
  const RCP<Epetra_Vector> vec,
  const RCP<Epetra_Vector> scavec)
{
  // remember time included in this average
  const double old_time = curr_avg_time_;

  // increase time counter
  curr_avg_time_+=dt;

  // add vector to average (this is an arithmetic mean!)
  /*
  //                        n
  //      - n+1   - n      t                 dt
  //      u     = u   * -------- + vec * ---------
  //                      n                n
  //                     t + dt           t  + dt
  */

  const double oldfac = old_time/curr_avg_time_;
  const double incfac =       dt/curr_avg_time_;

  curr_avg_->Update(incfac,*vec,oldfac);

  if(withscatra_)
  {
    if ( curr_avg_sca_ != Teuchos::null)
      curr_avg_sca_->Update(incfac,*scavec,oldfac);
    else
      dserror("curr_avg_sca_ is Teuchos::null");
  }

  // increase number of steps included in this sample
  ++curr_n_;

  return;
} // FLD::TurbulenceStatisticsGeneralMean::AddToCurrentTimeAverage


//----------------------------------------------------------------------
//
//       Perform a averaging of the current, already time averaged
//              vector, in space in a homogeneous direction.
//
//----------------------------------------------------------------------
void FLD::TurbulenceStatisticsGeneralMean::SpaceAverageInOneDirection(
  const int dim )
{
  // get a communicator
  const Epetra_Comm& avgcomm=discret_->Comm();

  // get rowmap for dofs
  const Epetra_Map* dofrowmap  = discret_->DofRowMap ();

  // get a tolerance
  const double eps = 1e-7;

  // get other dimensions
  int odim[2];
  {
    switch (dim)
    {
    case 0:
    {
      odim[0]=1;
      odim[1]=2;
      break;
    }
    case 1:
    {
      odim[0]=0;
      odim[1]=2;
      break;
    }
    case 2:
    {
      odim[0]=0;
      odim[1]=1;
      break;
    }
    default:
    {
      odim[0]=-1;
      odim[1]=-1;

      dserror("dimension to average not in 0-2\n",dim);
    }
    }
  }


  // get minimal coordinate in direction dim
  double minxdim = 0.0;
  {
    // first on this proc
    double lminxdim = 1e9;

    for(int nn=0;nn<discret_->NumMyRowNodes();++nn)
    {
      // get the processor local node
      DRT::Node*  lnode      = discret_->lRowNode(nn);

      double xdim = (lnode->X())[dim];

      if(lminxdim>xdim)
      {
        lminxdim=xdim;
      }
    }

    // do communication for global mins
    avgcomm.MinAll(&lminxdim,&minxdim,1);
  }


  // get a vector of all ((X[odim[0]],X[odim[1]]),X[dim) pairs on
  // this proc

  vector<double> x;
  vector<double> y;
  {

    double xdim    ;
    double xodim[2];

    for(int nn=0;nn<discret_->NumMyRowNodes();++nn)
    {
      // get the processor local node
      DRT::Node*  lnode      = discret_->lRowNode(nn);

      // check for slave nodes  to skip them
      vector<DRT::Condition*> mypbcs;
      lnode->GetCondition("SurfacePeriodic",mypbcs);

      // check whether a periodic boundary condition is active on this node
      if (mypbcs.size()>0)
      {
        bool is_slave=false;

        // yes, we have one
        for (unsigned numcond=0;numcond<mypbcs.size();++numcond)
        {
          DRT::Condition* pbc=mypbcs[numcond];

          // see whether pbc is active in plane orthogonal to sampling plane
          const string* dofsforpbcplanename
            =
            pbc->Get<string>("degrees of freedom for the pbc plane");

          bool active=false;

          if(*dofsforpbcplanename=="xyz")
          {
            active=true;
          }
          else if(*dofsforpbcplanename=="xy")
          {
            if(dim==2)
            {
              active=true;
            }
          }
          else if(*dofsforpbcplanename=="xz")
          {
            if(dim==1)
            {
              active=true;
            }
          }
          else if(*dofsforpbcplanename=="yz")
          {
            if(dim==0)
            {
              active=true;
            }
          }

          if(active)
          {
            // see whether we have a slave node
            const string* mymasterslavetoggle
              = pbc->Get<string>("Is slave periodic boundary condition");

            if(*mymasterslavetoggle=="Slave")
            {
              is_slave=true;
            }
          }
        }
        if(is_slave)
        {
          continue;
        }
      }

      xdim = (lnode->X())[dim];

      // this is a value on the very bottom in dim direction
      if(xdim-eps<minxdim)
      {
        xodim[0]= (lnode->X())[odim[0]];
        xodim[1]= (lnode->X())[odim[1]];

        x.push_back(xodim[0]);
        y.push_back(xodim[1]);
      }
    }
  }

  // get a global number of lines in direction dim, i.e. the global
  // number of (X[odim[0]],X[odim[1]]) pairs
  int numlines=0;
  {
    int lnumlines=x.size();

    avgcomm.SumAll(&lnumlines,&numlines,1);
  }


  // get an empty vector for the averages
  vector<double> avg_u(x.size(),0.0);
  vector<double> avg_v(x.size(),0.0);
  vector<double> avg_w(x.size(),0.0);
  vector<double> avg_p(x.size(),0.0);

  vector<int>    count(x.size(),0  );

  //----------------------------------------------------------------------
  // do a round robin loop
  //
  // 1) receive (x,y) pairs and avg from previous processor
  // 2) on each processor, construct map x->(y->i)
  // 3) for each node on the proc: search in map, add value to to avgs at i
  // 4) pass (x,y) pairs and avg to the next proc
  //
  // 1) is skipped in the first step, 4) in the last
  //----------------------------------------------------------------------

  const int myrank  =avgcomm.MyPID();
  const int numprocs=avgcomm.NumProc();

  vector<char> sblock;
  vector<char> rblock;

  sblock.clear();
  rblock.clear();

  // stl map to construct
  map<double,map<double,int,doublecomp>,doublecomp>           xtoy;
  map<double,map<double,int,doublecomp>,doublecomp>::iterator x_and_y;
  map<double,int,doublecomp>::iterator                        y_and_i;

#ifdef PARALLEL
  // create an exporter for point to point comunication
  DRT::Exporter exporter(avgcomm);

  // necessary variables
  MPI_Request request;

  int         tag    =-1;
  int         frompid=-1;
  int         topid  =-1;
  int         length =-1;

#endif

  for (int np=0;np<numprocs+1;++np)
  {
    // in the first step, we cannot receive anything
    if(np >0)
    {
#ifdef PARALLEL
      //--------------------------------------------------
      // Receive a block from the last proc

      // make sure that you do not think you received something if
      // you didn't
      if(rblock.empty()==false)
      {
        dserror("rblock not empty");
      }

      rblock.clear();

      // receive from predecessor
      frompid=(myrank+numprocs-1)%numprocs;
      exporter.ReceiveAny(frompid,tag,rblock,length);

      if(tag!=(myrank+numprocs-1)%numprocs)
      {
        dserror("received wrong message (ReceiveAny)");
      }

      exporter.Wait(request);

      // for safety
      exporter.Comm().Barrier();
#endif

      //--------------------------------------------------
      // Unpack received block

      // clear all old stuff
      x.clear();
      y.clear();
      count.clear();
      avg_u.clear();
      avg_v.clear();
      avg_w.clear();
      avg_p.clear();

      vector<char>::size_type position=0;

      // size
      int size;
      DRT::ParObject::ExtractfromPack(position,rblock,size);

      x    .resize(size,0.0);
      y    .resize(size,0.0);

      count.resize(size,0  );
      avg_u.resize(size,0.0);
      avg_v.resize(size,0.0);
      avg_w.resize(size,0.0);
      avg_p.resize(size,0.0);

      // x and y
      DRT::ParObject::ExtractfromPack(position,rblock,x);
      DRT::ParObject::ExtractfromPack(position,rblock,y);

      // counters
      DRT::ParObject::ExtractfromPack(position,rblock,count);

      // avgs
      DRT::ParObject::ExtractfromPack(position,rblock,avg_u);
      DRT::ParObject::ExtractfromPack(position,rblock,avg_v);
      DRT::ParObject::ExtractfromPack(position,rblock,avg_w);
      DRT::ParObject::ExtractfromPack(position,rblock,avg_p);

      rblock.clear();
    }


    // in the last step, we keep everything on this proc
    if(np < numprocs)
    {

      // 2) use (x,y) pairs and avg to construct map x->(y->avg)
      xtoy.clear();

      for(unsigned i=0;i<x.size();++i)
      {
        // check whether x is already in the map
        x_and_y=xtoy.find(x[i]);

        if(x_and_y!=xtoy.end())
        {
          // it is already in the map. This y cannot overwrite
          // something since pairs (x,y) are unique

          (x_and_y->second).insert(pair<double,int>(y[i],i));
        }
        else
        {
          // it's not in the map yet. construct second map with
          // one initial connection
          map<double,int,doublecomp> y_to_i_map;
          y_to_i_map.insert(pair<double,int>(y[i],i));

          xtoy.insert(pair<double,map<double,int,doublecomp> >(x[i],y_to_i_map));
        }
      }

      // 3) for each node on this proc: search in map, add
      //    value to avg
      for(int nn=0;nn<discret_->NumMyRowNodes();++nn)
      {
        // get the processor local node
        DRT::Node*  lnode      = discret_->lRowNode(nn);

        // check for slave nodes  to skip them
        vector<DRT::Condition*> mypbcs;
        lnode->GetCondition("SurfacePeriodic",mypbcs);

        // check whether a periodic boundary condition is active on this node
        if (mypbcs.size()>0)
        {
          bool is_slave=false;

          // yes, we have one
          for (unsigned numcond=0;numcond<mypbcs.size();++numcond)
          {
            DRT::Condition* pbc=mypbcs[numcond];

            // see whether pbc is active in plane orthogonal to sampling plane
            const string* dofsforpbcplanename
              =
              pbc->Get<string>("degrees of freedom for the pbc plane");

            bool active=false;

            if(*dofsforpbcplanename=="xyz")
            {
              active=true;
            }
            else if(*dofsforpbcplanename=="xy")
            {
              if(dim==2)
              {
                active=true;
              }
            }
            else if(*dofsforpbcplanename=="xz")
            {
              if(dim==1)
              {
                active=true;
              }
            }
            else if(*dofsforpbcplanename=="yz")
            {
              if(dim==0)
              {
                active=true;
              }
            }

            if(active)
            {
              // see whether we have a slave node
              const string* mymasterslavetoggle
                = pbc->Get<string>("Is slave periodic boundary condition");

              if(*mymasterslavetoggle=="Slave")
              {
                is_slave=true;
              }
            }
          }
          if(is_slave)
          {
            continue;
          }
        }

        double xodim[2];

        xodim[0]= (lnode->X())[odim[0]];

        x_and_y=xtoy.find(xodim[0]);

        if(x_and_y!=xtoy.end())
        {
          xodim[1]= (lnode->X())[odim[1]];

          y_and_i=(x_and_y->second).find(xodim[1]);

          if(y_and_i!=(x_and_y->second).end())
          {
            const int pos = y_and_i->second;

            // get dofs of vector to average
            int gid;
            int lid;

            // the set of degrees of freedom associated with the node
            vector<int> nodedofset = discret_->Dof(lnode);

            // u velocity
            gid = nodedofset[0];
            lid = dofrowmap->LID(gid);

            avg_u[pos]+=(*curr_avg_)[lid];

            // v velocity
            gid = nodedofset[1];
            lid = dofrowmap->LID(gid);

            avg_v[pos]+=(*curr_avg_)[lid];

            // w velocity
            gid = nodedofset[2];
            lid = dofrowmap->LID(gid);

            avg_w[pos]+=(*curr_avg_)[lid];

            // pressure p
            gid = nodedofset[3];
            lid = dofrowmap->LID(gid);

            avg_p[pos]+=(*curr_avg_)[lid];

            // count nodes
            count[pos] +=1;
          }
          else
          {
            if(numprocs==1)
            {
              dserror("didn\'t find node %d on single proc\n",lnode->Id());
            }
          }
        }
        else
        {
          if(numprocs==1)
          {
            dserror("didn\'t find node %d on single proc\n",lnode->Id());
          }
        }
      }

      //--------------------------------------------------
      // Pack block to send
      DRT::PackBuffer data;

      // size
      int size=x.size();
      DRT::ParObject::AddtoPack(data,size);

      // x and y
      DRT::ParObject::AddtoPack(data,x);
      DRT::ParObject::AddtoPack(data,y);

      // counters
      DRT::ParObject::AddtoPack(data,count);

      // avgs
      DRT::ParObject::AddtoPack(data,avg_u);
      DRT::ParObject::AddtoPack(data,avg_v);
      DRT::ParObject::AddtoPack(data,avg_w);
      DRT::ParObject::AddtoPack(data,avg_p);

      data.StartPacking();

      DRT::ParObject::AddtoPack(data,size);

      // x and y
      DRT::ParObject::AddtoPack(data,x);
      DRT::ParObject::AddtoPack(data,y);

      // counters
      DRT::ParObject::AddtoPack(data,count);

      // avgs
      DRT::ParObject::AddtoPack(data,avg_u);
      DRT::ParObject::AddtoPack(data,avg_v);
      DRT::ParObject::AddtoPack(data,avg_w);
      DRT::ParObject::AddtoPack(data,avg_p);

      swap( sblock, data() );

#ifdef PARALLEL
      //--------------------------------------------------
      // Send block to next proc.

      tag    =myrank;
      frompid=myrank;
      topid  =(myrank+1)%numprocs;

      exporter.ISend(frompid,topid,
                     &(sblock[0]),sblock.size(),
                     tag,request);

#endif
    }
  }

  //----------------------------------------------------------------------
  // divide vectors by number of layers along lines
  for(unsigned i=0;i<x.size();++i)
  {
    if(count[i]==0)
    {
      dserror("no layers have been detected along line %d\n",i);
    }

    avg_u[i]/=count[i];
    avg_v[i]/=count[i];
    avg_w[i]/=count[i];
    avg_p[i]/=count[i];
  }

  //----------------------------------------------------------------------
  // repeat communication to redistribute stuff into the global vector

  for (int np=0;np<numprocs+1;++np)
  {
    // in the first step, we cannot receive anything
    if(np >0)
    {
#ifdef PARALLEL
      //--------------------------------------------------
      // Receive a block from the last proc

      // make sure that you do not think you received something if
      // you didn't
      if(rblock.empty()==false)
      {
        dserror("rblock not empty");
      }

      // receive from predecessor
      frompid=(myrank+numprocs-1)%numprocs;
      exporter.ReceiveAny(frompid,tag,rblock,length);

      if(tag!=(myrank+numprocs-1)%numprocs)
      {
        dserror("received wrong message (ReceiveAny)");
      }

      exporter.Wait(request);

      // for safety
      exporter.Comm().Barrier();
#endif

      //--------------------------------------------------
      // Unpack received block

      // clear all old stuff
      x.clear();
      y.clear();
      count.clear();
      avg_u.clear();
      avg_v.clear();
      avg_w.clear();
      avg_p.clear();

      vector<char>::size_type position=0;

      // size
      int size;
      DRT::ParObject::ExtractfromPack(position,rblock,size);

      count.resize(size,0  );
      avg_u.resize(size,0.0);
      avg_v.resize(size,0.0);
      avg_w.resize(size,0.0);
      avg_p.resize(size,0.0);

      // x and y
      DRT::ParObject::ExtractfromPack(position,rblock,x);
      DRT::ParObject::ExtractfromPack(position,rblock,y);

      // counters
      DRT::ParObject::ExtractfromPack(position,rblock,count);

      // avgs
      DRT::ParObject::ExtractfromPack(position,rblock,avg_u);
      DRT::ParObject::ExtractfromPack(position,rblock,avg_v);
      DRT::ParObject::ExtractfromPack(position,rblock,avg_w);
      DRT::ParObject::ExtractfromPack(position,rblock,avg_p);

      rblock.clear();
    }

    // 2) use (x,y) pairs and avg to construct map x->(y->avg)
    xtoy.clear();

    for(unsigned i=0;i<x.size();++i)
    {
      // check whether x is already in the map
      x_and_y=xtoy.find(x[i]);

      if(x_and_y!=xtoy.end())
      {
        // it is already in the map. This y cannot overwrite
        // something since pairs (x,y) are unique

        (x_and_y->second).insert(pair<double,int>(y[i],i));
      }
      else
      {
        // it's not in the map yet. construct second map with
        // one initial connection
        map<double,int,doublecomp> y_to_i_map;
        y_to_i_map.insert(pair<double,int>(y[i],i));

        xtoy.insert(pair<double,map<double,int,doublecomp> >(x[i],y_to_i_map));
      }
    }

    // 3) for each node on this proc: search in map, insert
    //    avg into global vector
    for(int nn=0;nn<discret_->NumMyRowNodes();++nn)
    {
      // get the processor local node
      DRT::Node*  lnode      = discret_->lRowNode(nn);

      // check for slave nodes  to skip them
      vector<DRT::Condition*> mypbcs;
      lnode->GetCondition("SurfacePeriodic",mypbcs);

      // check whether a periodic boundary condition is active on this node
      if (mypbcs.size()>0)
      {
        bool is_slave=false;

        // yes, we have one
        for (unsigned numcond=0;numcond<mypbcs.size();++numcond)
        {
          DRT::Condition* pbc=mypbcs[numcond];

          // see whether pbc is active in plane orthogonal to sampling plane
          const string* dofsforpbcplanename
            =
            pbc->Get<string>("degrees of freedom for the pbc plane");

          bool active=false;

          if(*dofsforpbcplanename=="xyz")
          {
            active=true;
          }
          else if(*dofsforpbcplanename=="xy")
          {
            if(dim==2)
            {
              active=true;
            }
          }
          else if(*dofsforpbcplanename=="xz")
          {
            if(dim==1)
            {
              active=true;
            }
          }
          else if(*dofsforpbcplanename=="yz")
          {
            if(dim==0)
            {
              active=true;
            }
          }

          if(active)
          {
            // see whether we have a slave node
            const string* mymasterslavetoggle
              = pbc->Get<string>("Is slave periodic boundary condition");

            if(*mymasterslavetoggle=="Slave")
            {
              is_slave=true;
            }
          }
        }
        if(is_slave)
        {
          continue;
        }
      }

      double xodim[2];

      xodim[0]= (lnode->X())[odim[0]];

      x_and_y=xtoy.find(xodim[0]);

      if(x_and_y!=xtoy.end())
      {
	xodim[1]= (lnode->X())[odim[1]];

	y_and_i=(x_and_y->second).find(xodim[1]);

	if(y_and_i!=(x_and_y->second).end())
	{
	  int pos = y_and_i->second;

	  // get dofs of vector to average
	  int gid;
	  int lid;

	  // the set of degrees of freedom associated with the node
	  vector<int> nodedofset = discret_->Dof(lnode);

	  int err=0;

	  // u velocity
	  gid = nodedofset[0];
	  lid = dofrowmap->LID(gid);

	  err += curr_avg_->ReplaceMyValues(1,&(avg_u[pos]),&lid);

	  // v velocity
	  gid = nodedofset[1];
	  lid = dofrowmap->LID(gid);

	  err += curr_avg_->ReplaceMyValues(1,&(avg_v[pos]),&lid);

	  // w velocity
	  gid = nodedofset[2];
	  lid = dofrowmap->LID(gid);

	  err += curr_avg_->ReplaceMyValues(1,&(avg_w[pos]),&lid);

	  // pressure p
	  gid = nodedofset[3];
	  lid = dofrowmap->LID(gid);

	  err += curr_avg_->ReplaceMyValues(1,&(avg_p[pos]),&lid);

          if(err>0)
          {
            dserror("lid was not on proc %d\n",myrank);
          }
	}
      }
    }

    // in the last step, we keep everything on this proc
    if(np < numprocs)
    {
      //--------------------------------------------------
      // Pack block to send
      DRT::PackBuffer data;

      // size
      int size=x.size();

      DRT::ParObject::AddtoPack(data,size);

      // x and y
      DRT::ParObject::AddtoPack(data,x);
      DRT::ParObject::AddtoPack(data,y);

      // counters
      DRT::ParObject::AddtoPack(data,count);

      // avgs
      DRT::ParObject::AddtoPack(data,avg_u);
      DRT::ParObject::AddtoPack(data,avg_v);
      DRT::ParObject::AddtoPack(data,avg_w);
      DRT::ParObject::AddtoPack(data,avg_p);

      data.StartPacking();

      DRT::ParObject::AddtoPack(data,size);

      // x and y
      DRT::ParObject::AddtoPack(data,x);
      DRT::ParObject::AddtoPack(data,y);

      // counters
      DRT::ParObject::AddtoPack(data,count);

      // avgs
      DRT::ParObject::AddtoPack(data,avg_u);
      DRT::ParObject::AddtoPack(data,avg_v);
      DRT::ParObject::AddtoPack(data,avg_w);
      DRT::ParObject::AddtoPack(data,avg_p);

      swap( sblock, data() );

#ifdef PARALLEL
      //--------------------------------------------------
      // Send block to next proc.

      tag    =myrank;
      frompid=myrank;
      topid  =(myrank+1)%numprocs;

      exporter.ISend(frompid,topid,
                     &(sblock[0]),sblock.size(),
                     tag,request);

#endif
    }
  }

  return;
} // FLD::TurbulenceStatisticsGeneralMean::SpaceAverageInOneDirection


//----------------------------------------------------------------------
//
//           Add vector to time average from previous steps
//
//----------------------------------------------------------------------
void FLD::TurbulenceStatisticsGeneralMean::AddToTotalTimeAverage()
{
  // remember time included in this average
  const double old_time = prev_avg_time_;

  // increase time counter
  prev_avg_time_+=curr_avg_time_;

  // add vector to average (this is an arithmetic mean!)
  /*
  //                            old                    inc
  //      - new   - old        t          - inc       t
  //      u     = u     * ------------- + u     * -------------
  //                        old    inc              old    inc
  //                       t    + t                t    + t
  */

  const double oldfac =       old_time/prev_avg_time_;
  const double incfac = curr_avg_time_/prev_avg_time_;

  prev_avg_->Update(incfac,*curr_avg_,oldfac);

  if(withscatra_)
  {
    prev_avg_sca_->Update(incfac,*curr_avg_sca_,oldfac);
  }

  // increase number of steps included in this sample
  prev_n_+=curr_n_;

  // reinitialise curr(ent) counter and averages
  TimeReset();

  return;
} // FLD::TurbulenceStatisticsGeneralMean::AddToTotalTimeAverage


//----------------------------------------------------------------------
//
//          Read previous statistics from a file (for restart)
//
//----------------------------------------------------------------------
void FLD::TurbulenceStatisticsGeneralMean::ReadOldStatistics(
  IO::DiscretizationReader&  input
  )
{
  prev_n_        = input.ReadInt   ("num_steps_in_sample");
  prev_avg_time_ = input.ReadDouble("sampling_time"      );

  input.ReadVector(prev_avg_,"averaged_velnp");
  if(withscatra_)
    input.ReadVector(prev_avg_sca_,"averaged_scanp");

  return;
} // FLD::TurbulenceStatisticsGeneralMean::ReadOldStatistics


//----------------------------------------------------------------------
//
//                 Write the statistics to a file
//
//----------------------------------------------------------------------
void FLD::TurbulenceStatisticsGeneralMean::WriteOldAverageVec(
  IO::DiscretizationWriter&  output
)
{

  // loop homogeneous directions, do averaging
  for(unsigned i=0;i<homdir_.size();++i)
  {
    SpaceAverageInOneDirection(homdir_[i]);
  }

  AddToTotalTimeAverage();

  if(discret_->Comm().MyPID()==0)
  {
    cout << "XXXXXXXXXXXXXXXXXXXXX              ";
    cout << " Wrote averaged vector             ";
    cout << "XXXXXXXXXXXXXXXXXXXXX";
    cout << "\n\n";
  }

  output.WriteInt   ("num_steps_in_sample", prev_n_       );
  output.WriteDouble("sampling_time"      , prev_avg_time_);

  output.WriteVector("averaged_velnp",prev_avg_);
  if(withscatra_)
    output.WriteVector("averaged_scanp",prev_avg_sca_);

  // output real pressure
  Teuchos::RCP<Epetra_Vector> pressure = velpressplitter_.ExtractCondVector(prev_avg_);
  pressure->Scale(density_);
  output.WriteVector("averaged_pressure", pressure);

  return;
} // FLD::TurbulenceStatisticsGeneralMean::WriteOldAverageVec


//----------------------------------------------------------------------
//
//     Clear all statistics collected in the current period
//
//----------------------------------------------------------------------
void FLD::TurbulenceStatisticsGeneralMean::TimeReset()
{
  const Epetra_Map* dofrowmap = discret_->DofRowMap();

  curr_avg_     = Teuchos::null;
  curr_avg_     = LINALG::CreateVector(*dofrowmap,true);
  if(withscatra_)
  {
    curr_avg_sca_     = Teuchos::null;
    curr_avg_sca_     = LINALG::CreateVector(*dofrowmap,true);
  }

  curr_n_       = 0;
  curr_avg_time_= 0.0;

  return;
} // FLD::TurbulenceStatisticsGeneralMean::TimeReset


//----------------------------------------------------------------------
//
//          Clear all statistics collected up to now
//
//----------------------------------------------------------------------
void FLD::TurbulenceStatisticsGeneralMean::ResetComplete()
{
  const Epetra_Map* dofrowmap = discret_->DofRowMap();

  curr_avg_     = Teuchos::null;
  curr_avg_     = LINALG::CreateVector(*dofrowmap,true);

  curr_n_       = 0;
  curr_avg_time_= 0.0;

  prev_avg_     = Teuchos::null;
  prev_avg_     = LINALG::CreateVector(*dofrowmap,true);

  prev_n_       = 0;
  prev_avg_time_= 0.0;

  if(withscatra_)
  {
    curr_avg_sca_     = Teuchos::null;
    curr_avg_sca_     = LINALG::CreateVector(*dofrowmap,true);
    prev_avg_sca_     = Teuchos::null;
    prev_avg_sca_     = LINALG::CreateVector(*dofrowmap,true);
  }

  return;
} // FLD::TurbulenceStatisticsGeneralMean::ResetComplete




#endif


/*!----------------------------------------------------------------------
\file airwayimplicitintegration.cpp
\brief Control routine for reduced airway solvers,

<pre>
Maintainer: Mahmoud Ismail
            ismail@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15268
</pre>
*----------------------------------------------------------------------*/
#ifdef CCADISCRET

#include <stdio.h>

#include "airwayimplicitintegration.H"

#include "../drt_lib/drt_condition_utils.H"
#include "../drt_lib/drt_globalproblem.H"
#include "../linalg/linalg_ana.H"
#include "../linalg/linalg_utils.H"
#include "../linalg/linalg_solver.H"
#include "../drt_lib/drt_dserror.H"
#include "../drt_lib/drt_nodematchingoctree.H"
#include "../drt_lib/drt_function.H"



//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 |  Constructor (public)                                    ismail 01/10|
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//

AIRWAY::RedAirwayImplicitTimeInt::RedAirwayImplicitTimeInt(RCP<DRT::Discretization>  actdis,
                                                           LINALG::Solver  &         solver,
                                                           ParameterList&            params,
                                                           IO::DiscretizationWriter& output)
  :
  // call constructor for "nontrivial" objects
  discret_(actdis),
  solver_ (solver),
  params_ (params),
  output_ (output),
  time_(0.0),
  step_(0),
  uprestart_(params.get("write restart every", -1)),
  upres_(params.get("write solution every", -1)),
  coupledTo3D_(false)
{

  // -------------------------------------------------------------------
  // get the processor ID from the communicator
  // -------------------------------------------------------------------
  myrank_  = discret_->Comm().MyPID();

  // time measurement: initialization
  if(!coupledTo3D_)
  {
    // time measurement: initialization
    TEUCHOS_FUNC_TIME_MONITOR(" + initialization");
  }

  // -------------------------------------------------------------------
  // get the basic parameters first
  // -------------------------------------------------------------------
  // time-step size
  dtp_ = dta_ = params_.get<double>("time step size");
  // maximum number of timesteps
  stepmax_  = params_.get<int>   ("max number timesteps");
  // maximum simulation time
  maxtime_  = dtp_*double(stepmax_);
  // maximum iteration steps
  maxiter_     = params_.get<int> ("maximum iteration steps");
  // tolerance of nonlinear solution
  non_lin_tol_ = params_.get<double> ("tolerance");


  // ensure that degrees of freedom in the discretization have been set
  if (!discret_->Filled() || !actdis->HaveDofs()) discret_->FillComplete();

  // -------------------------------------------------------------------
  // get a vector layout from the discretization to construct matching
  // vectors and matrices
  //                 local <-> global dof numbering
  // -------------------------------------------------------------------
  const Epetra_Map* dofrowmap      = discret_->DofRowMap();
  const Epetra_Map* elementcolmap  = discret_->ElementColMap();

  // -------------------------------------------------------------------
  // get a vector layout from the discretization for a vector which only
  // contains the volumetric flow rate dofs and for one vector which only
  // contains cross-sectional area degrees of freedom.
  // -------------------------------------------------------------------


  // This is a first estimate for the number of non zeros in a row of
  // the matrix. Each node has 3 adjacent nodes (including itself), each
  // with 1 dofs. (3*1=3)
  // We do not need the exact number here, just for performance reasons
  // a 'good' estimate

  // initialize standard (stabilized) system matrix
  sysmat_  = Teuchos::rcp(new LINALG::SparseMatrix(*dofrowmap,3,false,true));

  // Vectors passed to the element
  // -----------------------------

  // Pressures at time n+1, n and n-1
  pnp_          = LINALG::CreateVector(*dofrowmap,true);
  pn_           = LINALG::CreateVector(*dofrowmap,true);
  pnm_          = LINALG::CreateVector(*dofrowmap,true);

  p_nonlin_     = LINALG::CreateVector(*dofrowmap,true);

  // Inlet volumetric flow rates at time n+1, n and n-1
  qin_np_       = LINALG::CreateVector(*elementcolmap,true);
  qin_n_        = LINALG::CreateVector(*elementcolmap,true);
  qin_nm_       = LINALG::CreateVector(*elementcolmap,true);

  qi_nl_np_     = LINALG::CreateVector(*elementcolmap,true);

  // outlet volumetric flow rates at time n+1, n and n-1
  qout_np_      = LINALG::CreateVector(*elementcolmap,true);
  qout_n_       = LINALG::CreateVector(*elementcolmap,true);
  qout_nm_      = LINALG::CreateVector(*elementcolmap,true);

  qo_nl_np_     = LINALG::CreateVector(*elementcolmap,true);


  // Volumetric flow rates at time n+1, n and n-1
  //  qcnp_          = LINALG::CreateVector(*elementrowmap,true);
  //  qcn_           = LINALG::CreateVector(*elementrowmap,true);
  //  qcnm_          = LINALG::CreateVector(*elementrowmap,true);

  // vectors for postprocessing, Element Node Ids and raduis 
  nodeIds_      = LINALG::CreateVector(*dofrowmap,true);
  radii_        = LINALG::CreateVector(*dofrowmap,true);

  // a vector of zeros to be used to enforce zero dirichlet boundary conditions
  // This part might be optimized later
  bcval_   = LINALG::CreateVector(*dofrowmap,true);
  dbctog_  = LINALG::CreateVector(*dofrowmap,true);

  //  abc_  = LINALG::CreateVector(*dofrowmap,true);

  // Vectors used for solution process
  // ---------------------------------


  // right hand side vector and right hand side corrector
  rhs_     = LINALG::CreateVector(*dofrowmap,true);

  // ---------------------------------------------------------------------------------------
  // Initialize all the arteries' cross-sectional areas to the initial crossectional area Ao
  // and the volumetric flow rate to 0
  // ---------------------------------------------------------------------------------------
  ParameterList eleparams;

  // loop all elements and initialize all of the values

  eleparams.set("p0np",pnp_);
  eleparams.set("p0n",pn_);
  eleparams.set("p0nm",pnm_);
  
  //  eleparams.set("qc0np",qcnp_);
  //  eleparams.set("qc0n",qcn_);
  //  eleparams.set("qc0nm",qcnm_);
  eleparams.set("radii",radii_);


  //  eleparams.set("abc",abc_);
  eleparams.set("action","get_initial_state");
  discret_->Evaluate(eleparams,Teuchos::null,Teuchos::null,Teuchos::null,Teuchos::null,Teuchos::null);

  
  // Fill the NodeId vector
  for (int nele=0;nele<discret_->NumMyColElements();++nele)
  {
    // get the element
    DRT::Element* ele = discret_->lColElement(nele);

    // get element location vector, dirichlet flags and ownerships
    vector<int> lm;
    vector<int> lmstride;
    //vector<int> lmowner;
    RCP<vector<int> > lmowner = rcp(new vector<int>);
    ele->LocationVector(*discret_,lm,*lmowner,lmstride);

    // loop all nodes of this element, add values to the global vectors

    if(myrank_ == (*lmowner)[0])
    {
      int    gid = lm[0];
      double val = gid;
      nodeIds_->ReplaceGlobalValues(1,&val,&gid);
      //      double val2 = 0.0;
      //      if (ele->Nodes()[0]->GetCondition("RedLungAcinusCond"))
      //      {
      //        double val2 = 1.0+(*abc_)[gid];
      //        abc_->ReplaceGlobalValues(1,&val2,&gid);
      //      }

    }
    if(myrank_ == (*lmowner)[1])
    {
      int    gid = lm[1];
      double val = gid;
      nodeIds_->ReplaceGlobalValues(1,&val,&gid);
      //      if (ele->Nodes()[1]->GetCondition("RedLungAcinusCond"))
      //      {
      //        double val2 = 1.0+(*abc_)[gid];
      //        abc_->ReplaceGlobalValues(1,&val2,&gid);
      //      }

    }
  }


} // RedAirwayImplicitTimeInt::RedAirwayImplicitTimeInt


//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 | Start the time integration.                                          |
 |                                                          ismail 01/10|
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void AIRWAY::RedAirwayImplicitTimeInt::Integrate()
{
  RCP<ParameterList> param;
  Integrate(false, param);
} //RedAirwayImplicitTimeInt::Integrate()


//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 | Start the time integration.                                          |
 |                                                          ismail 01/10|
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void AIRWAY::RedAirwayImplicitTimeInt::Integrate(bool CoupledTo3D,
                                                 RCP<ParameterList> CouplingParams)
{
  coupledTo3D_ = CoupledTo3D;
  if (CoupledTo3D && CouplingParams.get() == NULL)
  {
    dserror("Coupling parameter list is not allowed to be empty, If a 3-D/reduced-D coupling is defined\n");
  }

  TimeLoop(CoupledTo3D,CouplingParams);

  // print the results of time measurements
  if (!coupledTo3D_)
  {
    TimeMonitor::summarize();
  }

  return;
} // RedAirwayImplicitTimeInt::Integrate



//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 | contains the time loop                                   ismail 01/10|
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void AIRWAY::RedAirwayImplicitTimeInt::TimeLoop(bool CoupledTo3D,
                                                RCP<ParameterList> CouplingTo3DParams)
{
  coupledTo3D_ = CoupledTo3D;
  // time measurement: time loop
  if(!coupledTo3D_)
  {
    TEUCHOS_FUNC_TIME_MONITOR(" + time loop");
  }

  while (step_<stepmax_ and time_<maxtime_)
  {
    PrepareTimeStep();
    // -------------------------------------------------------------------
    //                       output to screen
    // -------------------------------------------------------------------
    if (myrank_==0)
    {
      if(!coupledTo3D_)
      {
        printf("TIME: %11.4E/%11.4E  DT = %11.4E   Solving Reduced Dimensional Airways    STEP = %4d/%4d \n",
               time_,maxtime_,dta_,step_,stepmax_);
      }
      else
      {
        printf("SUBSCALE_TIME: %11.4E/%11.4E  SUBSCALE_DT = %11.4E   Solving Reduced Dimensional Airways    SUBSCALE_STEP = %4d/%4d \n",
               time_,maxtime_,dta_,step_,stepmax_);
      }
    }


    // -------------------------------------------------------------------
    //                            Solve
    // -------------------------------------------------------------------
    if(params_.get<string> ("solver type") == "Nonlinear")
    {
      NonLin_Solve(CouplingTo3DParams);
      if (!myrank_)
        cout<<endl;
    }
    else if (params_.get<string> ("solver type") == "Linear")
    {
      Solve(CouplingTo3DParams);
      if (!myrank_)
        cout<<endl<<endl;
    }
    else
    {
      dserror("[%s] is not a defined solver",(params_.get<string> ("solver type")).c_str() );
    }

    // -------------------------------------------------------------------
    //                         update solution
    //        current solution becomes old solution of next timestep
    // -------------------------------------------------------------------
    TimeUpdate();

    // -------------------------------------------------------------------
    //  lift'n'drag forces, statistics time sample and output of solution
    //  and statistics
    // -------------------------------------------------------------------
    if (!CoupledTo3D)
    {
      Output(CoupledTo3D,CouplingTo3DParams);
    }

    // -------------------------------------------------------------------
    //                       update time step sizes
    // -------------------------------------------------------------------
    dtp_ = dta_;

    // -------------------------------------------------------------------
    //                    stop criterium for timeloop
    // -------------------------------------------------------------------
    if (CoupledTo3D)
    {
      break;
    }
  }

} // RedAirwayImplicitTimeInt::TimeLoop


//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 | setup the variables to do a new time step                ismail 01/10|
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void AIRWAY::RedAirwayImplicitTimeInt::PrepareTimeStep()
{
  rhs_->PutScalar(0.0);
  // -------------------------------------------------------------------
  //              set time dependent parameters
  // -------------------------------------------------------------------
  step_ += 1;
  time_ += dta_;
} //RedAirwayImplicitTimeInt::PrepareTimeStep


//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 | the nonlinear iterative solver for reduced  ismail             01/11 |
 | dimensional airway                                                   |
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*
Some detials!!
*/
void AIRWAY::RedAirwayImplicitTimeInt::NonLin_Solve(Teuchos::RCP<ParameterList> CouplingTo3DParams)
{
  //--------------------------------------------------------------------
  // 
  //--------------------------------------------------------------------
  double error_norm = 1.e7;

  for (int i =1; i<=maxiter_; i++)
  {
    //------------------------------------------------------------------
    // update the pressures of the previous time step
    //------------------------------------------------------------------
    p_nonlin_->Update(1.0,*pnp_     ,0.0);

    //------------------------------------------------------------------
    // Solve the reduced dimensional model
    //------------------------------------------------------------------
    this->Solve(CouplingTo3DParams);

    //------------------------------------------------------------------
    // Find the change of pressure between the last two iteration steps
    //------------------------------------------------------------------
    p_nonlin_->Update(1.0,*pnp_     ,-1.0);

    //------------------------------------------------------------------
    // Evaluate the L2 norm of the difference
    //------------------------------------------------------------------
    p_nonlin_->Norm2 (&error_norm);
    error_norm  /= sqrt(double(p_nonlin_->MyLength()));

    //------------------------------------------------------------------
    // if L2 norm is smaller then tolerance then proceed
    //------------------------------------------------------------------
    if (!myrank_)
    {
      printf("iteration step %4d/%4d ",i,maxiter_);
      printf(" | ||P{%d}-P{%d}||_L2 = %f\n",i-1,i,error_norm);
    }
    if(error_norm <= non_lin_tol_)
      break;
    
  }
  if (!myrank_)
    cout<<endl;
  
}

//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 | the solver for reduced dimensional airway               ismail 01/10 |
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*
Some detials!!
*/
void AIRWAY::RedAirwayImplicitTimeInt::Solve(Teuchos::RCP<ParameterList> CouplingTo3DParams)
{
  // time measurement: Airways
  if(!coupledTo3D_)
  {
    TEUCHOS_FUNC_TIME_MONITOR("   + solving reduced dimensional airways");
  }

  // -------------------------------------------------------------------
  // call elements to calculate system matrix
  // -------------------------------------------------------------------

  // get cpu time
  //  const double tcpuele = Teuchos::Time::wallTime();
  {
    //    cout<<"get sysmat_"<<endl;
    // time measurement: element
    if(!coupledTo3D_)
    {
      TEUCHOS_FUNC_TIME_MONITOR("      + element calls");
    }

    // set both system matrix and rhs vector to zero
    sysmat_->Zero();
    rhs_->PutScalar(0.0);


    // create the parameters for the discretization
    ParameterList eleparams;

    // action for elements
    eleparams.set("action","calc_sys_matrix_rhs");
    eleparams.set("time step size",dta_);

    // other parameters that might be needed by the elements
    eleparams.set("total time",time_);

    // set vector values needed by elements
    discret_->ClearState();
    discret_->SetState("pnp",pnp_);
    discret_->SetState("pn" ,pn_ );
    discret_->SetState("pnm",pnm_);


    
    
    eleparams.set("qin_np",qin_np_);
    eleparams.set("qin_n",qin_n_);
    //    discret_->SetState("qin_n" ,qin_n_ );
    //    discret_->SetState("qin_nm",qin_nm_);

    eleparams.set("qout_np",qout_np_);
    eleparams.set("qout_n" ,qout_n_ );
    //    discret_->SetState("qout_nm",qout_nm_);

    // call standard loop over all elements
    discret_->Evaluate(eleparams,sysmat_,rhs_);
    discret_->ClearState();

    // finalize the complete matrix
    sysmat_->Complete();
    discret_->ClearState();
    

#if 0  // Exporting some values for debugging purposes

    {
      cout<<"----------------------- My SYSMAT IS ("<<myrank_<<"-----------------------"<<endl;
      RCP<LINALG::SparseMatrix> A_debug = Teuchos::rcp_dynamic_cast<LINALG::SparseMatrix>(sysmat_);
      if (A_debug != Teuchos::null)
      {
        // print to screen
        (A_debug->EpetraMatrix())->Print(cout);
      }
      cout<<"Map is: ("<<myrank_<<")"<<endl<<*(discret_->DofRowMap())<<endl;
      cout<<"---------------------------------------("<<myrank_<<"------------------------"<<endl;
    }
#endif 

  }
  // end time measurement for element


  // -------------------------------------------------------------------
  // Solve the boundary conditions 
  // -------------------------------------------------------------------
  bcval_->PutScalar(0.0);
  dbctog_->PutScalar(0.0);
  // Solve terminal BCs
  {
    // create the parameters for the discretization
    ParameterList eleparams;

    // action for elements
    eleparams.set("action","set_bc");

    // set vecotr values needed by elements
    discret_->ClearState();
    discret_->SetState("pnp",pnp_);
    discret_->SetState("pn" ,pn_ );
    discret_->SetState("pnm",pnm_);
    //    discret_->SetState("qcnp",qcnp_);
    //    discret_->SetState("qcn" ,qcn_ );
    //    discret_->SetState("qcnm",qcnm_);

    eleparams.set("qin_np",qin_np_);
    eleparams.set("qin_n",qin_n_);
    //    discret_->SetState("qin_n" ,qin_n_ );
    //    discret_->SetState("qin_nm",qin_nm_);

    eleparams.set("qout_np",qout_np_);
    //    discret_->SetState("qout_n" ,qout_n_ );
    //    discret_->SetState("qout_nm",qout_nm_);

    eleparams.set("time step size",dta_);
    eleparams.set("total time",time_);
    eleparams.set("bcval",bcval_);
    eleparams.set("dbctog",dbctog_);
    //    eleparams.set("abc",abc_);
    //    eleparams.set("rhs",rhs_);

    // Add the parameters to solve terminal BCs coupled to 3D fluid boundary
    eleparams.set("coupling with 3D fluid params",CouplingTo3DParams);

    // call standard loop over all elements
    discret_->Evaluate(eleparams,sysmat_,rhs_);
    discret_->ClearState();
  }
  

  double norm_bc_tog = 0.0;
  rhs_->Norm1(&norm_bc_tog);
  
  // -------------------------------------------------------------------
  // Apply the BCs to the system matrix and rhs
  // -------------------------------------------------------------------
  {
    // time measurement: application of dbc
    if(!coupledTo3D_)
    {
      TEUCHOS_FUNC_TIME_MONITOR("      + apply DBC");
    }

    LINALG::ApplyDirichlettoSystem(sysmat_,pnp_,rhs_,bcval_,dbctog_);
  }

#if 0  // Exporting some values for debugging purposes

    {
      cout<<"----------------------- My SYSMAT IS ("<<myrank_<<"-----------------------"<<endl;
      RCP<LINALG::SparseMatrix> A_debug = Teuchos::rcp_dynamic_cast<LINALG::SparseMatrix>(sysmat_);
      if (A_debug != Teuchos::null)
      {
        // print to screen
        (A_debug->EpetraMatrix())->Print(cout);
      }
      cout<<"Map is: ("<<myrank_<<")"<<endl<<*(discret_->DofRowMap())<<endl;
      cout<<"---------------------------------------("<<myrank_<<"------------------------"<<endl;
    }
    cout<<"rhs: "<<*rhs_<<endl;
#endif 

  //-------solve for total new velocities and pressures
  // get cpu time
  const double tcpusolve = Teuchos::Time::wallTime();
  {
    // time measurement: solver
    if(!coupledTo3D_)
    {
      TEUCHOS_FUNC_TIME_MONITOR("      + solver calls");
    }

#if 0 // Exporting some values for debugging purposes

    RCP<LINALG::SparseMatrix> A_debug = Teuchos::rcp_dynamic_cast<LINALG::SparseMatrix>(sysmat_);
    if (A_debug != Teuchos::null)
    {
      // print to screen
      (A_debug->EpetraMatrix())->Print(cout);
    }
    //#else    
    cout<<"DOF row map"<<*(discret_->DofRowMap())<<endl;
    cout<<"bcval: "<<*bcval_<<endl;
    cout<<"bctog: "<<*dbctog_<<endl;
    cout<<"pnp: "<<*pnp_<<endl;
    cout<<"rhs: "<<*rhs_<<endl;

#endif 
    // call solver
    solver_.Solve(sysmat_->EpetraOperator(),pnp_,rhs_,true,true);
    //    cout<<"Solved"<<endl;
  }

  // end time measurement for solver
  dtsolve_ = Teuchos::Time::wallTime() - tcpusolve;

  if (myrank_ == 0)
  printf("te=%f, ts=%4.7f |",dtele_, dtsolve_);
  //  cout << "te=" << dtele_ << ", ts=" << dtsolve_ <<" | ";

  // find the flow rates
  {
    // create the parameters for the discretization
    ParameterList eleparams;

    // action for elements
    eleparams.set("action","calc_flow_rates");

    // set solution type
    eleparams.set("solver type", params_.get<string> ("solver type"));

    // set vecotr values needed by elements
    discret_->ClearState();
    discret_->SetState("pnp",pnp_);
    discret_->SetState("pn" ,pn_ );
    discret_->SetState("pnm",pnm_);

    //    discret_->SetState("qcn" ,qcn_ );


    //    discret_->SetState("qin_np",qin_np_);
    //    discret_->SetState("qin_n" ,qin_n_ );
    //    discret_->SetState("qin_nm",qin_nm_);

    //    discret_->SetState("qout_np",qout_np_);
    //    discret_->SetState("qout_n" ,qout_n_ );
    //    discret_->SetState("qout_nm",qout_nm_);

    //    eleparams.set("qcnp",qcnp_);
    eleparams.set("time step size",dta_);
    eleparams.set("total time",time_);
    eleparams.set("qin_n",qin_n_);
    eleparams.set("qout_n",qout_n_);
    eleparams.set("qin_np",qin_np_);
    eleparams.set("qout_np",qout_np_);

    // call standard loop over all elements
    discret_->Evaluate(eleparams,sysmat_,rhs_);
    discret_->ClearState();
  }

  if(coupledTo3D_)
  {
    // create the parameters for the discretization
    ParameterList eleparams;

    // action for elements
    eleparams.set("action","get_coupled_values");

    // set vecotr values needed by elements
    discret_->ClearState();
    discret_->SetState("pnp",pnp_);
    //    discret_->SetState("qcn" ,qcn_ );

    eleparams.set("qin_np",qin_np_);
    eleparams.set("qin_n",qin_n_);
    //    discret_->SetState("qin_n" ,qin_n_ );
    //    discret_->SetState("qin_nm",qin_nm_);

    eleparams.set("qout_np",qout_np_);
    eleparams.set("qout_n" ,qout_n_ );
    //    discret_->SetState("qout_nm",qout_nm_);

    //    eleparams.set("qcnp",qcnp_);
    eleparams.set("time step size",dta_);
    eleparams.set("total time",time_);

    // Add the parameters to solve terminal BCs coupled to 3D fluid boundary
    eleparams.set("coupling with 3D fluid params",CouplingTo3DParams);

    // call standard loop over all elements
    discret_->Evaluate(eleparams,sysmat_,rhs_);
    discret_->ClearState();
  }

} // RedAirwayImplicitTimeInt::Solve




//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 | call elements to calculate system matrix/rhs and assemble            |
 | this function will be kept empty untill further use     ismail 01/10 |
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void AIRWAY::RedAirwayImplicitTimeInt::AssembleMatAndRHS()
{

  dtele_    = 0.0;
  dtfilter_ = 0.0;
  // time measurement: element
  if(!coupledTo3D_)
  {
    TEUCHOS_FUNC_TIME_MONITOR("      + element calls");
  }

  // get cpu time
  //  const double tcpu=Teuchos::Time::wallTime();

} // RedAirwayImplicitTimeInt::AssembleMatAndRHS




//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 | build system matrix and rhs                              ismail 01/10|
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void AIRWAY::RedAirwayImplicitTimeInt::Evaluate(Teuchos::RCP<const Epetra_Vector> qael)
{
  
}


//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 | current solution becomes most recent solution of next timestep       |
 |                                                                      |
 |  pnm_  =  pn_                                                        |
 |  pn_   =  pnp_                                                       |
 |                                                                      |
 |                                                          ismail 06/09|
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void AIRWAY::RedAirwayImplicitTimeInt::TimeUpdate()
{


  // Volumetric Flow rate/Cross-sectional area of this step become most recent
  pnm_->Update(1.0,*pn_ ,0.0);
  pn_ ->Update(1.0,*pnp_,0.0);

  qin_nm_->Update(1.0,*qin_n_ ,0.0);
  qin_n_ ->Update(1.0,*qin_np_,0.0);

  qout_nm_->Update(1.0,*qout_n_ ,0.0);
  qout_n_ ->Update(1.0,*qout_np_,0.0);
  //  qcnm_->Update(1.0,*qcn_ ,0.0);
  //  qcn_ ->Update(1.0,*qcnp_,0.0);

  return;
}// RedAirwayImplicitTimeInt::TimeUpdate


//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 | output of solution vector to binio                       ismail 07/09|
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void AIRWAY::RedAirwayImplicitTimeInt::Output(bool               CoupledTo3D,
                                              RCP<ParameterList> CouplingParams)
{
  int step      = 0;
  int upres     = 0;
  int uprestart = 0;
  double time_backup = 0.0;
  // -------------------------------------------------------------------
  // if coupled to 3D problem, then get the export information from 
  // the 3D problem
  // -------------------------------------------------------------------
  if (CoupledTo3D)
  {
    step        = step_;
    upres       = upres_;
    uprestart   = uprestart_;
    time_backup = time_;
    step_      = CouplingParams->get<int>("step");
    upres_     = CouplingParams->get<int>("upres");
    uprestart_ = CouplingParams->get<int>("uprestart");
    time_      = CouplingParams->get<double>("time");
  }


  if (step_%upres_ == 0)
  {
    // step number and time
    output_.NewStep(step_,time_);

    // "pressure" vectors
    output_.WriteVector("pnm",pnm_);
    output_.WriteVector("pn",pn_);
    output_.WriteVector("pnp",pnp_);

    // "flow" vectors for capacitances
    //    output_.WriteVector("qcnm",qcnm_);
    //    output_.WriteVector("qcn",qcn_);
    //    output_.WriteVector("qcnp",qcnp_);

    // write domain decomposition for visualization
    output_.WriteElementData();

    // write the flow values
    output_.WriteVector("qin_nm",qin_nm_);
    output_.WriteVector("qin_n" ,qin_n_ );
    output_.WriteVector("qin_np",qin_np_);
//
    output_.WriteVector("qout_nm",qout_nm_);
    output_.WriteVector("qout_n" ,qout_n_);
    output_.WriteVector("qout_np",qout_np_);


    if (step_==upres_)
    {
      output_.WriteVector("NodeIDs",nodeIds_);
      output_.WriteVector("radii",radii_);
      //      output_.WriteVector("abc",abc_);
    }

    // write mesh in each restart step --- the elements are required since
    // they contain history variables (the time dependent subscales)
    output_.WriteMesh(step_,time_);

    if (CoupledTo3D)
    {
      output_.WriteDouble("Actual_RedD_step", step);
    }
  }
  // write restart also when uprestart_ is not a integer multiple of upres_
  else if (uprestart_ != 0 && step_%uprestart_ == 0)
  {
    // step number and time
    output_.NewStep(step_,time_);

    // "pressure" vectors
    output_.WriteVector("pnm",pnm_);
    output_.WriteVector("pn",pn_);
    output_.WriteVector("pnp",pnp_);

    // "flow" vectors for capacitances
    //    output_.WriteVector("qcnm",qcnm_);
    //    output_.WriteVector("qcn",qcn_);
    //  output_.WriteVector("qcnp",qcnp_);


    // write domain decomposition for visualization
    output_.WriteElementData();
    
    // write the flow values
    output_.WriteVector("qin_nm",qin_nm_);
    output_.WriteVector("qin_n" ,qin_n_ );
    output_.WriteVector("qin_np",qin_np_);

    output_.WriteVector("qout_nm",qout_nm_);
    output_.WriteVector("qout_n" ,qout_n_);
    output_.WriteVector("qout_np",qout_np_);

    // write mesh in each restart step --- the elements are required since
    // they contain history variables (the time dependent subscales)
    output_.WriteMesh(step_,time_);

    if (CoupledTo3D)
    {
      output_.WriteDouble("Actual_RedD_step", step);
    }
  }

  // -------------------------------------------------------------------
  // if coupled to 3D problem, then retrieve the old information of the
  // the reduced model problem
  // -------------------------------------------------------------------
  if (CoupledTo3D)
  {
    step_     = step;
    upres_    = upres;
    uprestart_= uprestart;
    time_     = time_backup;
  }

  return;
} // RedAirwayImplicitTimeInt::Output


//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 | ReadRestart (public)                                     ismail 01/10|
 -----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>// 
void AIRWAY::RedAirwayImplicitTimeInt::ReadRestart(int step)
{

  IO::DiscretizationReader reader(discret_,step);
  time_ = reader.ReadDouble("time");

  if (coupledTo3D_)
  {
    step_ = reader.ReadInt("Actual_RedD_step");
  }
  else
  {
    step_ = reader.ReadInt("step");
  }

  reader.ReadVector(pnp_,"pnp");
  reader.ReadVector(pn_,"pn");
  reader.ReadVector(pnm_,"pnm");

  reader.ReadVector(qin_nm_, "qin_nm");
  reader.ReadVector(qin_n_ , "qin_n" );
  reader.ReadVector(qin_np_, "qin_np");
  
  reader.ReadVector(qout_nm_, "qout_nm");
  reader.ReadVector(qout_n_ , "qout_n" );
  reader.ReadVector(qout_np_, "qout_np");
  //reader.ReadVector(qcnp_,"qcnp");
  //reader.ReadVector(qcn_,"qcn");
  //reader.ReadVector(qcnm_,"qcnm");
  // read the previously written elements including the history data
  reader.ReadMesh(step_);

}//RedAirwayImplicitTimeInt::ReadRestart


/*----------------------------------------------------------------------*
 | Destructor dtor (public)                                 ismail 01/10|
 *----------------------------------------------------------------------*/
AIRWAY::RedAirwayImplicitTimeInt::~RedAirwayImplicitTimeInt()
{
  return;
}

#endif


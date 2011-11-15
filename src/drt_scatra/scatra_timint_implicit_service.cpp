/*----------------------------------------------------------------------*/
/*!
\file scatra_timint_implicit_service.cpp
\brief Service routines of the scalar transport time integration class

<pre>
Maintainer: Georg Bauer
            bauer@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15252
</pre>
*/
/*----------------------------------------------------------------------*/
#ifdef CCADISCRET

#include "scatra_timint_implicit.H"
#include "scatra_utils.H"
#include "../linalg/linalg_solver.H"
#include "../linalg/linalg_utils.H"
#include "../drt_lib/drt_timecurve.H"
#include "../drt_nurbs_discret/drt_nurbs_discret.H"
#include "../drt_fluid/fluid_rotsym_periodicbc_utils.H"
#include <Teuchos_TimeMonitor.hpp>
#include <Teuchos_StandardParameterEntryValidators.hpp>
// for AVM3 solver:
#include <MLAPI_Workspace.h>
#include <MLAPI_Aggregation.h>
// for printing electrode status to file
#include "../drt_lib/drt_globalproblem.H"
#include "../drt_io/io.H"
#include "../drt_io/io_control.H"
#include "../drt_io/io_gmsh.H"
//access to the material data (ELCH)
#include "../drt_mat/material.H"
#include "../drt_mat/ion.H"
#include "../drt_mat/matlist.H"
#include "../drt_inpar/inpar_elch.H"

/*----------------------------------------------------------------------*
 | calculate initial time derivative of phi at t=t_0           gjb 08/08|
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::CalcInitialPhidt()
{
  // assemble system: M phidt^0 = f^n - K\phi^n - C(u_n)\phi^n
  CalcInitialPhidtAssemble();
  // solve for phidt_0
  CalcInitialPhidtSolve();
}

/*----------------------------------------------------------------------*
 | calculate initial time derivative of phi at t=t_0 (assembly)gjb 08/08|
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::CalcInitialPhidtAssemble()
{
  // time measurement:
  TEUCHOS_FUNC_TIME_MONITOR("SCATRA:       + calc initial phidt");
  if (myrank_ == 0)
    std::cout<<"SCATRA: calculating initial time derivative of phi"<<endl;

  // are we really at step 0?
  dsassert(step_==0,"Step counter is not 0");

  // evaluate Dirichlet boundary conditions at time t=0
  // the values should match your initial field at the boundary!
  //ApplyDirichletBC(time_,phin_,phidtn_);
  ApplyDirichletBC(time_,phin_,Teuchos::null);

  {
    // evaluate Neumann boundary conditions at time t=0
    neumann_loads_->PutScalar(0.0);
    ParameterList p;
    p.set("total time",time_);
    p.set<int>("scatratype",scatratype_);
    p.set("isale",isale_);
    discret_->ClearState();
    discret_->EvaluateNeumann(p,*neumann_loads_);
    discret_->ClearState();

    // add potential Neumann boundary condition at time t=0
    // and zero out the residual_ vector!
    residual_->Update(1.0,*neumann_loads_,0.0);
  }

  // call elements to calculate matrix and right-hand-side
  {
    // create the parameters for the discretization
    ParameterList eleparams;

    // action for elements
    eleparams.set("action","calc_initial_time_deriv");

    // set type of scalar transport problem
    eleparams.set<int>("scatratype",scatratype_);

    // add additional parameters (timefac_ remains unused at element level!!!!)
    AddSpecificTimeIntegrationParameters(eleparams);

    // other parameters that are needed by the elements
    eleparams.set("incremental solver",incremental_);
    eleparams.set<int>("form of convective term",convform_);
    if (IsElch(scatratype_))
      eleparams.set("frt",frt_); // factor F/RT
    else if (scatratype_==INPAR::SCATRA::scatratype_loma)
    {
      eleparams.set("thermodynamic pressure",thermpressn_);
      eleparams.set("time derivative of thermodynamic pressure",thermpressdtn_);
    }

    // provide velocity field and potentially acceleration/pressure field
    // (export to column map necessary for parallel evaluation)
    AddMultiVectorToParameterList(eleparams,"velocity field",convel_);
    AddMultiVectorToParameterList(eleparams,"acceleration/pressure field",accpre_);

    // set switch for reinitialization
    eleparams.set("reinitswitch",reinitswitch_);

    // parameters for stabilization (here required for material evaluation location)
    eleparams.sublist("STABILIZATION") = params_->sublist("STABILIZATION");

    //provide displacement field in case of ALE
    eleparams.set("isale",isale_);
    if (isale_)
      AddMultiVectorToParameterList(eleparams,"dispnp",dispnp_);

    // set vector values needed by elements
    discret_->ClearState();
    discret_->SetState("phi0",phin_);

    // call loop over elements
    discret_->Evaluate(eleparams,sysmat_,residual_);
    discret_->ClearState();

    // finalize the complete matrix
    sysmat_->Complete();
  }

  // what's next?
  return;
}

/*----------------------------------------------------------------------*
 | calculate initial time derivative of phi at t=t_0 (solver) gjb 08/08 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::CalcInitialPhidtSolve()
{
  // are we really at step 0?
  dsassert(step_==0,"Step counter is not 0");

  // We determine phidtn at every node (including boundaries)
  // To be consistent with time integration scheme we do not prescribe
  // any values at Dirichlet boundaries for phidtn !!!
  // apply Dirichlet boundary conditions to system matrix
  // LINALG::ApplyDirichlettoSystem(sysmat_,phidtn_,residual_,phidtn_,*(dbcmaps_->CondMap()));

  // solve for phidtn
  solver_->Solve(sysmat_->EpetraOperator(),phidtn_,residual_,true,true);

  // copy solution also to phidtnp
  phidtnp_->Update(1.0,*phidtn_,0.0);

  // reset the matrix (and its graph!) since we solved
  // a very special problem here that has a different sparsity pattern
  if (DRT::INPUT::IntegralValue<int>(*params_,"BLOCKPRECOND"))
    BlockSystemMatrix()->Reset();
  else
    SystemMatrix()->Reset();
  // reset the solver as well
  solver_->Reset();

  // that's it
  return;
}

/*----------------------------------------------------------------------*
 | calculate initial electric potential field at t=t_0         gjb 04/10|
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::CalcInitialPotentialField()
{
  if (IsElch(scatratype_))
  {
    if (DRT::INPUT::IntegralValue<int>(*params_,"INITPOTCALC"))
    {
      // time measurement:
      TEUCHOS_FUNC_TIME_MONITOR("SCATRA:       + calc initial potential field");
      if (myrank_ == 0)
        std::cout<<"SCATRA: calculating initial field for electric potential"<<endl;

      // are we really at step 0?
      dsassert(step_==0,"Step counter is not 0");

      // construct intermediate vectors
      const Epetra_Map* dofrowmap = discret_->DofRowMap();
      RCP<Epetra_Vector> rhs = LINALG::CreateVector(*dofrowmap,true);
      RCP<Epetra_Vector> phi0 = LINALG::CreateVector(*dofrowmap,true);

      // zero out matrix entries
      sysmat_->Zero();

      // evaluate Dirichlet boundary conditions at time t=0
      // the values should match your initial field at the boundary!
      //ApplyDirichletBC(time_,phin_,phidtn_);
      ApplyDirichletBC(time_,phin_,Teuchos::null);

      // ToDo:
      // contributions due to Neumann b.c. or ElectrodeKinetics b.c.
      // have to be summed up here, and applied
      // as a current flux condition at the potential field!

      // evaluate Neumann boundary conditions at time t=0
      /*{
        neumann_loads_->PutScalar(0.0);
        ParameterList p;
        p.set("total time",time_);
        p.set<int>("scatratype",scatratype_);
        p.set("isale",isale_);
        discret_->ClearState();
        discret_->EvaluateNeumann(p,*neumann_loads_);
        discret_->ClearState();

        // add potential Neumann boundary condition at time t=0
        // and zero out the residual_ vector!
        // residual_->Update(1.0,*neumann_loads_,0.0);
      }*/

      // call elements to calculate matrix and right-hand-side
      {

        // create the parameters for the discretization
        ParameterList eleparams;

        // action for elements
        eleparams.set("action","calc_initial_potential_field");

        // set type of scalar transport problem
        eleparams.set<int>("scatratype",scatratype_);

        // factor F/RT
        eleparams.set("frt",frt_);

        // parameters for stabilization
        eleparams.sublist("STABILIZATION") = params_->sublist("STABILIZATION");

        //provide displacement field in case of ALE
        eleparams.set("isale",isale_);
        if (isale_)
          AddMultiVectorToParameterList(eleparams,"dispnp",dispnp_);

        // set vector values needed by elements
        discret_->ClearState();
        discret_->SetState("phi0",phin_);

        // call loop over elements
        discret_->Evaluate(eleparams,sysmat_,rhs);
        discret_->ClearState();

        // finalize the complete matrix
        sysmat_->Complete();
      }

      // apply Dirichlet boundary conditions to system matrix
      LINALG::ApplyDirichlettoSystem(sysmat_,phi0,rhs,phi0,*(dbcmaps_->CondMap()));

      // solve
      solver_->Solve(sysmat_->EpetraOperator(),phi0,rhs,true,true);

      // copy solution of initial potential field to the solution vectors
      Teuchos::RCP<Epetra_Vector> onlypot = splitter_->ExtractCondVector(phi0);
      // insert values into the whole solution vectors
      splitter_->InsertCondVector(onlypot,phinp_);
      splitter_->InsertCondVector(onlypot,phin_);

      // reset the matrix (and its graph!) since we solved
      // a very special problem here that has a different sparsity pattern
      if (DRT::INPUT::IntegralValue<int>(*params_,"BLOCKPRECOND"))
        BlockSystemMatrix()->Reset();
      else
        SystemMatrix()->Reset();
    }
  }
  // go on!
  return;
}

/*----------------------------------------------------------------------*
 | evaluate contribution of electrode kinetics to eq. system  gjb 02/09 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::EvaluateElectrodeKinetics(
    RCP<LINALG::SparseOperator> matrix,
    RCP<Epetra_Vector>          rhs
)
{
  // time measurement: evaluate condition 'ElectrodeKinetics'
  TEUCHOS_FUNC_TIME_MONITOR("SCATRA:       + evaluate condition 'ElectrodeKinetics'");

  discret_->ClearState();

  // create an parameter list
  ParameterList condparams;

  // action for elements
  condparams.set("action","calc_elch_electrode_kinetics");
  condparams.set<int>("scatratype",scatratype_);
  condparams.set("frt",frt_); // factor F/RT
  condparams.set("isale",isale_);
  if (isale_)   //provide displacement field in case of ALE
    AddMultiVectorToParameterList(condparams,"dispnp",dispnp_);

  // add element parameters and set state vectors according to time-integration scheme
  AddSpecificTimeIntegrationParameters(condparams);

  std::string condstring("ElectrodeKinetics");
  // evaluate ElectrodeKinetics conditions at time t_{n+1} or t_{n+alpha_F}
  discret_->EvaluateCondition(condparams,matrix,Teuchos::null,rhs,Teuchos::null,Teuchos::null,condstring);
  discret_->ClearState();

  return;
}


/*----------------------------------------------------------------------*
 | evaluate Neumann inflow boundary condition                  vg 03/09 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::ComputeNeumannInflow(
    RCP<LINALG::SparseOperator> matrix,
    RCP<Epetra_Vector>          rhs)
{
  // time measurement: evaluate condition 'Neumann inflow'
  TEUCHOS_FUNC_TIME_MONITOR("SCATRA:       + evaluate condition 'TransportNeumannInflow'");

  // create parameter list
  ParameterList condparams;

  // action for elements
  condparams.set("action","calc_Neumann_inflow");
  condparams.set<int>("scatratype",scatratype_);
  condparams.set("incremental solver",incremental_);
  condparams.set("isale",isale_);

  // provide velocity field and potentially acceleration/pressure field
  // as well as displacement field in case of ALE
  // (export to column map necessary for parallel evaluation)
  AddMultiVectorToParameterList(condparams,"velocity field",convel_);
  AddMultiVectorToParameterList(condparams,"acceleration/pressure field",accpre_);
  if (isale_) AddMultiVectorToParameterList(condparams,"dispnp",dispnp_);

  // clear state
  discret_->ClearState();

  // add element parameters and vectors according to time-integration scheme
  AddSpecificTimeIntegrationParameters(condparams);

  std::string condstring("TransportNeumannInflow");
  discret_->EvaluateCondition(condparams,matrix,Teuchos::null,rhs,Teuchos::null,Teuchos::null,condstring);
  discret_->ClearState();

  return;
}

/*----------------------------------------------------------------------*
 | evaluate boundary cond. due to convective heat transfer     vg 10/11 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::EvaluateConvectiveHeatTransfer(
    RCP<LINALG::SparseOperator> matrix,
    RCP<Epetra_Vector>          rhs)
{
  // time measurement: evaluate condition 'ThermoConvections'
  TEUCHOS_FUNC_TIME_MONITOR("SCATRA:       + evaluate condition 'ThermoConvections'");

  // create parameter list
  ParameterList condparams;

  // action for elements
  condparams.set("action","calc_convective_heat_transfer");
  condparams.set<int>("scatratype",scatratype_);
  condparams.set("incremental solver",incremental_);
  condparams.set("isale",isale_);

  // clear state
  discret_->ClearState();

  // add element parameters and vectors according to time-integration scheme
  AddSpecificTimeIntegrationParameters(condparams);

  std::string condstring("ThermoConvections");
  discret_->EvaluateCondition(condparams,matrix,Teuchos::null,rhs,Teuchos::null,Teuchos::null,condstring);
  discret_->ClearState();

  return;
}

/*----------------------------------------------------------------------*
 | Evaluate surface/interface permeability                              |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::SurfacePermeability(
    RCP<LINALG::SparseOperator> matrix,
    RCP<Epetra_Vector>          rhs)
{
  // time measurement: evaluate condition 'SurfacePermeability'
  TEUCHOS_FUNC_TIME_MONITOR("SCATRA:       + evaluate condition 'ScaTraCoupling'");

  // create parameter list
  ParameterList condparams;

  // action for elements
  condparams.set("action","calc_surface_permeability");
  condparams.set<int>("scatratype",scatratype_);
  condparams.set("incremental solver",incremental_);

  // provide displacement field in case of ALE
  condparams.set("isale",isale_);
  if (isale_) AddMultiVectorToParameterList(condparams,"dispnp",dispnp_);

  // set vector values needed by elements
  discret_->ClearState();

  // add element parameters according to time-integration scheme
  AddSpecificTimeIntegrationParameters(condparams);

  std::string condstring("ScaTraCoupling");
  discret_->EvaluateCondition(condparams,matrix,Teuchos::null,rhs,Teuchos::null,Teuchos::null,condstring);
  discret_->ClearState();

  matrix->Complete();

  double scaling = 1.0/ResidualScaling();

  rhs->Scale(scaling);
  matrix->Scale(scaling);

  return;
}

/*----------------------------------------------------------------------*
 | construct toggle vector for Dirichlet dofs                  gjb 11/08|
 | assures backward compatibility for avm3 solver; should go away once  |
 *----------------------------------------------------------------------*/
const Teuchos::RCP<const Epetra_Vector> SCATRA::ScaTraTimIntImpl::DirichletToggle()
{
  if (dbcmaps_ == Teuchos::null)
    dserror("Dirichlet map has not been allocated");
  Teuchos::RCP<Epetra_Vector> dirichones = LINALG::CreateVector(*(dbcmaps_->CondMap()),false);
  dirichones->PutScalar(1.0);
  Teuchos::RCP<Epetra_Vector> dirichtoggle = LINALG::CreateVector(*(discret_->DofRowMap()),true);
  dbcmaps_->InsertCondVector(dirichones, dirichtoggle);
  return dirichtoggle;
}


/*----------------------------------------------------------------------*
 |  prepare AVM3-based scale separation                        vg 10/08 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::AVM3Preparation()
{
  // time measurement: avm3
  TEUCHOS_FUNC_TIME_MONITOR("SCATRA:            + avm3");

  // create normalized all-scale subgrid-diffusivity matrix
  sysmat_sd_->Zero();

  // create the parameters for the discretization
  ParameterList eleparams;

  // action for elements, time factor and stationary flag
  eleparams.set("action","calc_subgrid_diffusivity_matrix");

  // set type of scalar transport problem
  eleparams.set<int>("scatratype",scatratype_);

  //provide displacement field in case of ALE
  eleparams.set("isale",isale_);
  if (isale_)
    AddMultiVectorToParameterList(eleparams,"dispnp",dispnp_);

  // add element parameters according to time-integration scheme
  AddSpecificTimeIntegrationParameters(eleparams);

  // call loop over elements
  discret_->Evaluate(eleparams,sysmat_sd_,residual_);
  discret_->ClearState();

  // finalize the normalized all-scale subgrid-diffusivity matrix
  sysmat_sd_->Complete();

  // apply DBC to normalized all-scale subgrid-diffusivity matrix
  LINALG::ApplyDirichlettoSystem(sysmat_sd_,phinp_,residual_,phinp_,*(dbcmaps_->CondMap()));

  // get normalized fine-scale subgrid-diffusivity matrix
  {
    // this is important to have!!!
    MLAPI::Init();

    // extract the ML parameters
    ParameterList&  mlparams = solver_->Params().sublist("ML Parameters");

    // get toggle vector for Dirchlet boundary conditions
    const Epetra_Vector& dbct = *DirichletToggle();

    // get nullspace parameters
    double* nullspace = mlparams.get("null space: vectors",(double*)NULL);
    if (!nullspace) dserror("No nullspace supplied in parameter list");
    int nsdim = mlparams.get("null space: dimension",1);

    // modify nullspace to ensure that DBC are fully taken into account
    if (nullspace)
    {
      const int length = sysmat_sd_->OperatorRangeMap().NumMyElements();
      for (int i=0; i<nsdim; ++i)
        for (int j=0; j<length; ++j)
          if (dbct[j]!=0.0) nullspace[i*length+j] = 0.0;
    }

    // get plain aggregation Ptent
    RCP<Epetra_CrsMatrix> crsPtent;
    MLAPI::GetPtent(*sysmat_sd_->EpetraMatrix(),mlparams,nullspace,crsPtent);
    LINALG::SparseMatrix Ptent(crsPtent);

    // compute scale-separation matrix: S = I - Ptent*Ptent^T
    Sep_ = LINALG::Multiply(Ptent,false,Ptent,true);
    Sep_->Scale(-1.0);
    RCP<Epetra_Vector> tmp = LINALG::CreateVector(Sep_->RowMap(),false);
    tmp->PutScalar(1.0);
    RCP<Epetra_Vector> diag = LINALG::CreateVector(Sep_->RowMap(),false);
    Sep_->ExtractDiagonalCopy(*diag);
    diag->Update(1.0,*tmp,1.0);
    Sep_->ReplaceDiagonalValues(*diag);

    //complete scale-separation matrix and check maps
    Sep_->Complete(Sep_->DomainMap(),Sep_->RangeMap());
    if (!Sep_->RowMap().SameAs(sysmat_sd_->RowMap())) dserror("rowmap not equal");
    if (!Sep_->RangeMap().SameAs(sysmat_sd_->RangeMap())) dserror("rangemap not equal");
    if (!Sep_->DomainMap().SameAs(sysmat_sd_->DomainMap())) dserror("domainmap not equal");

    // precomputation of unscaled diffusivity matrix:
    // either two-sided S^T*M*S: multiply M by S from left- and right-hand side
    // or one-sided M*S: only multiply M by S from left-hand side
    if (not incremental_)
    {
      Mnsv_ = LINALG::Multiply(*sysmat_sd_,false,*Sep_,false);
      //Mnsv_ = LINALG::Multiply(*Sep_,true,*Mnsv_,false);
    }
  }

  return;
}


/*----------------------------------------------------------------------*
 |  scaling of AVM3-based subgrid-diffusivity matrix           vg 10/08 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::AVM3Scaling(ParameterList& eleparams)
{
  // time measurement: avm3
  TEUCHOS_FUNC_TIME_MONITOR("SCATRA:            + avm3");

  // some necessary definitions
  int ierr;
  double* sgvsqrt = 0;
  int length = subgrdiff_->MyLength();

  // square-root of subgrid-viscosity-scaling vector for left and right scaling
  sgvsqrt = (double*)subgrdiff_->Values();
  for (int i = 0; i < length; ++i)
  {
    sgvsqrt[i] = sqrt(sgvsqrt[i]);
    subgrdiff_->ReplaceMyValues(1,&sgvsqrt[i],&i);
  }

  // get unscaled S^T*M*S from Sep
  sysmat_sd_ = rcp(new LINALG::SparseMatrix(*Mnsv_));

  // left and right scaling of normalized fine-scale subgrid-viscosity matrix
  ierr = sysmat_sd_->LeftScale(*subgrdiff_);
  if (ierr) dserror("Epetra_CrsMatrix::LeftScale returned err=%d",ierr);
  ierr = sysmat_sd_->RightScale(*subgrdiff_);
  if (ierr) dserror("Epetra_CrsMatrix::RightScale returned err=%d",ierr);

  // add the subgrid-viscosity-scaled fine-scale matrix to obtain complete matrix
  Teuchos::RCP<LINALG::SparseMatrix> sysmat = SystemMatrix();
  sysmat->Add(*sysmat_sd_,false,1.0,1.0);

  // set subgrid-diffusivity vector to zero after scaling procedure
  subgrdiff_->PutScalar(0.0);

  return;
}


/*----------------------------------------------------------------------*
 | set initial thermodynamic pressure                          vg 07/09 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::SetInitialThermPressure()
{
  // get thermodynamic pressure and gas constant from material parameters
  // (if no temperature equation, zero values are returned)
  ParameterList eleparams;
  eleparams.set("action","get_material_parameters");
  eleparams.set<int>("scatratype",scatratype_);
  eleparams.set("isale",isale_);
  discret_->Evaluate(eleparams,null,null,null,null,null);
  thermpressn_ = eleparams.get("thermodynamic pressure", 98100.0);

  // initialize also value at n+1
  // (computed if not constant, otherwise prescribed value remaining)
  thermpressnp_ = thermpressn_;

  // initialize time derivative of thermodynamic pressure at n+1 and n
  // (computed if not constant, otherwise remaining zero)
  thermpressdtnp_ = 0.0;
  thermpressdtn_  = 0.0;

  // compute values at intermediate time steps
  // (only for generalized-alpha time-integration scheme)
  // -> For constant thermodynamic pressure, this is done here once and
  // for all simulation time.
  ComputeThermPressureIntermediateValues();

  return;
}


/*----------------------------------------------------------------------*
 | compute initial time derivative of thermodynamic pressure   vg 07/09 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::ComputeInitialThermPressureDeriv()
{
  // define element parameter list
  ParameterList eleparams;

  // DO THIS BEFORE PHINP IS SET (ClearState() is called internally!!!!)
  // compute flux approximation and add it to the parameter list
  AddFluxApproxToParameterList(eleparams,INPAR::SCATRA::flux_diffusive_domain);

  // set scalar vector values needed by elements
  discret_->ClearState();
  discret_->SetState("phinp",phin_);

  // provide velocity field and potentially acceleration/pressure field
  // (export to column map necessary for parallel evaluation)
  AddMultiVectorToParameterList(eleparams,"velocity field",convel_);
  AddMultiVectorToParameterList(eleparams,"acceleration/pressure field",accpre_);

  // provide displacement field in case of ALE
  eleparams.set("isale",isale_);
  if (isale_) AddMultiVectorToParameterList(eleparams,"dispnp",dispnp_);

  // set parameters for element evaluation
  eleparams.set("action","calc_domain_and_bodyforce");
  eleparams.set<int>("scatratype",scatratype_);
  eleparams.set("total time",0.0);

  // variables for integrals of domain and bodyforce
  Teuchos::RCP<Epetra_SerialDenseVector> scalars
    = Teuchos::rcp(new Epetra_SerialDenseVector(2));

  // evaluate domain and bodyforce integral
  discret_->EvaluateScalars(eleparams, scalars);

  // get global integral values
  double pardomint  = (*scalars)[0];
  double parbofint  = (*scalars)[1];

  // set action for elements
  eleparams.set("action","calc_therm_press");

  // variables for integrals of normal velocity and diffusive flux
  double normvelint      = 0.0;
  double normdifffluxint = 0.0;
  eleparams.set("normal velocity integral",normvelint);
  eleparams.set("normal diffusive flux integral",normdifffluxint);

  // evaluate velocity-divergence and diffusive (minus sign!) flux on boundaries
  // We may use the flux-calculation condition for calculation of fluxes for
  // thermodynamic pressure, since it is usually at the same boundary.
  vector<std::string> condnames;
  condnames.push_back("ScaTraFluxCalc");
  for (unsigned int i=0; i < condnames.size(); i++)
  {
    discret_->EvaluateCondition(eleparams,Teuchos::null,Teuchos::null,Teuchos::null,Teuchos::null,Teuchos::null,condnames[i]);
  }

  // get integral values on this proc
  normvelint      = eleparams.get<double>("normal velocity integral");
  normdifffluxint = eleparams.get<double>("normal diffusive flux integral");

  // get integral values in parallel case
  double parnormvelint      = 0.0;
  double parnormdifffluxint = 0.0;
  discret_->Comm().SumAll(&normvelint,&parnormvelint,1);
  discret_->Comm().SumAll(&normdifffluxint,&parnormdifffluxint,1);

  // clean up
  discret_->ClearState();

  // compute initial time derivative of thermodynamic pressure
  // (with specific heat ratio fixed to be 1.4)
  const double shr = 1.4;
  thermpressdtn_ = (-shr*thermpressn_*parnormvelint
                    + (shr-1.0)*(-parnormdifffluxint+parbofint))/pardomint;

  // set time derivative of thermodynamic pressure at n+1 equal to the one at n
  // for following evaluation of intermediate values
  thermpressdtnp_ = thermpressdtn_;

  // compute values at intermediate time steps
  // (only for generalized-alpha time-integration scheme)
  ComputeThermPressureIntermediateValues();

  return;
}


/*----------------------------------------------------------------------*
 | compute initial total mass in domain                        vg 01/09 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::ComputeInitialMass()
{
  // set scalar values needed by elements
  discret_->ClearState();
  discret_->SetState("phinp",phin_);
  // set action for elements
  ParameterList eleparams;
  eleparams.set("action","calc_mean_scalars");
  eleparams.set<int>("scatratype",scatratype_);
  // inverted scalar values are required here
  eleparams.set("inverting",true);

  //provide displacement field in case of ALE
  eleparams.set("isale",isale_);
  if (isale_)
    AddMultiVectorToParameterList(eleparams,"dispnp",dispnp_);

  // evaluate integral of inverse temperature
  Teuchos::RCP<Epetra_SerialDenseVector> scalars
    = Teuchos::rcp(new Epetra_SerialDenseVector(numscal_+1));
  discret_->EvaluateScalars(eleparams, scalars);
  discret_->ClearState();   // clean up

  // compute initial mass times gas constant: R*M_0 = int(1/T_0)*tp
  initialmass_ = (*scalars)[0]*thermpressn_;

  // print out initial total mass
  if (myrank_ == 0)
  {
    cout << endl;
    cout << "+--------------------------------------------------------------------------------------------+" << endl;
    cout << "Initial total mass in domain (times gas constant): " << initialmass_ << endl;
    cout << "+--------------------------------------------------------------------------------------------+" << endl;
  }

  return;
}


/*----------------------------------------------------------------------*
 | compute thermodynamic pressure from mass conservation       vg 01/09 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::ComputeThermPressureFromMassCons()
{
  // set scalar values needed by elements
  discret_->ClearState();
  discret_->SetState("phinp",phinp_);
  // set action for elements
  ParameterList eleparams;
  eleparams.set("action","calc_mean_scalars");
  eleparams.set<int>("scatratype",scatratype_);
  // inverted scalar values are required here
  eleparams.set("inverting",true);

  //provide displacement field in case of ALE
  eleparams.set("isale",isale_);
  if (isale_) AddMultiVectorToParameterList(eleparams,"dispnp",dispnp_);

  // evaluate integral of inverse temperature
  Teuchos::RCP<Epetra_SerialDenseVector> scalars
    = Teuchos::rcp(new Epetra_SerialDenseVector(numscal_+1));
  discret_->EvaluateScalars(eleparams, scalars);
  discret_->ClearState();   // clean up

  // compute thermodynamic pressure: tp = R*M_0/int(1/T)
  thermpressnp_ = initialmass_/(*scalars)[0];

  // print out thermodynamic pressure
  if (myrank_ == 0)
  {
    cout << endl;
    cout << "+--------------------------------------------------------------------------------------------+" << endl;
    cout << "Thermodynamic pressure from mass conservation: " << thermpressnp_ << endl;
    cout << "+--------------------------------------------------------------------------------------------+" << endl;
  }

  // compute time derivative of thermodynamic pressure at time step n+1
  ComputeThermPressureTimeDerivative();

  // compute values at intermediate time steps
  // (only for generalized-alpha time-integration scheme)
  ComputeThermPressureIntermediateValues();

  return;
}


/*----------------------------------------------------------------------*
 | perform setup of natural convection applications (ELCH)    gjb 07/09 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::SetupElchNatConv()
{
  // loads densification coefficients and the initial mean concentration

  // only required for ELCH with natural convection
  if (prbtype_ == "elch")
  {
    if (DRT::INPUT::IntegralValue<int>(extraparams_->sublist("ELCH CONTROL"),"NATURAL_CONVECTION") == true)
    {
      // allocate denselch_ with *dofrowmap and initialize it
      const Epetra_Map* dofrowmap = discret_->DofRowMap();
      elchdensnp_ = LINALG::CreateVector(*dofrowmap,true);
      elchdensnp_->PutScalar(1.0);

      // Calculate the initial mean concentration value
      if (numscal_ < 1) dserror("Error since numscal = %d. Not allowed since < 1",numscal_);
      c0_.resize(numscal_);

      discret_->ClearState();
      discret_->SetState("phinp",phinp_);
      // set action for elements
      ParameterList eleparams;
      eleparams.set("action","calc_mean_scalars");
      eleparams.set<int>("scatratype",scatratype_);
      eleparams.set("inverting",false);

      //provide displacement field in case of ALE
      eleparams.set("isale",isale_);
      if (isale_)
        AddMultiVectorToParameterList(eleparams,"dispnp",dispnp_);

      // evaluate integrals of concentrations and domain
      Teuchos::RCP<Epetra_SerialDenseVector> scalars
      = Teuchos::rcp(new Epetra_SerialDenseVector(numscal_+1));
      discret_->EvaluateScalars(eleparams, scalars);
      discret_->ClearState();   // clean up

      // calculate mean_concentration
      const double domint  = (*scalars)[numscal_];
      for(int k=0;k<numscal_;k++)
      {
        c0_[k] = (*scalars)[k]/domint;
      }

      //initialization of the densification coefficient vector
      densific_.resize(numscal_);
      DRT::Element*   element = discret_->lRowElement(0);
      RefCountPtr<MAT::Material>  mat = element->Material();

      if (mat->MaterialType() == INPAR::MAT::m_matlist)
      {
        const MAT::MatList* actmat = static_cast<const MAT::MatList*>(mat.get());

        for (int k = 0;k<numscal_;++k)
        {
          const int matid = actmat->MatID(k);
          Teuchos::RCP<const MAT::Material> singlemat = actmat->MaterialById(matid);

          if (singlemat->MaterialType() == INPAR::MAT::m_ion)
          {
            const MAT::Ion* actsinglemat = static_cast<const MAT::Ion*>(singlemat.get());
            densific_[k] = actsinglemat->Densification();
            if (densific_[k] < 0.0) dserror("received negative densification value");
          }
          else
            dserror("material type is not allowed");
        }
      }
      if (mat->MaterialType() == INPAR::MAT::m_ion) // for a single species calculation
      {
        const MAT::Ion* actmat = static_cast<const MAT::Ion*>(mat.get());
        densific_[0] = actmat->Densification();
        if (densific_[0] < 0.0) dserror("received negative densification value");
        if (numscal_ > 1) dserror("Single species calculation but numscal = %d > 1",numscal_);
      }
    }
  }

  return;
}


/*----------------------------------------------------------------------*
 | compute density from ion concentrations                    gjb 07/09 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::ComputeDensity(double density0)
{
  double newdensity(0.0);
  int err(0);

  // loop over all local nodes
  for(int lnodeid=0; lnodeid<discret_->NumMyRowNodes(); lnodeid++)
  {
    // get the processor's local node
    DRT::Node* lnode = discret_->lRowNode(lnodeid);

    // get the degrees of freedom associated with this node
    vector<int> nodedofs;
    nodedofs = discret_->Dof(lnode);
    int numdof = nodedofs.size();

    newdensity= 1.0;
    // loop over all ionic species
    for(int k=0; k<numscal_; k++)
    {
      /*
        //                  k=numscal_-1
        //          /       ----                         \
        //         |        \                            |
        // rho_0 * | 1 +    /       alfa_k * (c_k - c_0) |
        //         |        ----                         |
        //          \       k=0                          /
        //
        // For use of molar mass M_k:  alfa_k = M_k/rho_0  !!
       */

      // global and processor's local DOF ID
      const int globaldofid = nodedofs[k];
      const int localdofid = phinp_->Map().LID(globaldofid);
      if (localdofid < 0)
        dserror("localdofid not found in map for given globaldofid");

      // compute contribution to density due to ionic species k
      newdensity += densific_[k]*((*phinp_)[localdofid]-c0_[k]);
    }
    newdensity *= density0;

    // insert the current density value for this node
    // (has to be at position of el potential/ the position of the last dof!
    const int globaldofid = nodedofs[numdof-1];
    const int localdofid = phinp_->Map().LID(globaldofid);
    if (localdofid < 0)
      dserror("localdofid not found in map for given globaldofid");
    err = elchdensnp_->ReplaceMyValue(localdofid,0,newdensity);
    if (err != 0) dserror("error while inserting a value into elchdensnp_");

  } // loop over all local nodes
  return;
}


/*----------------------------------------------------------------------*
 | convergence check (only for low-Mach-number flow)           vg 09/11 |
 *----------------------------------------------------------------------*/
bool SCATRA::ScaTraTimIntImpl::ConvergenceCheck(int          itnum,
                                                int          itmax,
                                                const double ittol)
{
  bool stopnonliniter = false;

  // define L2-norm of residual, incremental scalar and scalar
  double resnorm_L2(0.0);
  double phiincnorm_L2(0.0);
  double phinorm_L2(0.0);

  // for the time being, only one scalar considered for low-Mach-number flow
  /*if (numscal_>1)
  {
    Teuchos::RCP<Epetra_Vector> onlyphi = splitter_->ExtractCondVector(increment_);
    onlyphi->Norm2(&phiincnorm_L2);

    splitter_->ExtractCondVector(phinp_,onlyphi);
    onlyphi->Norm2(&phinorm_L2);
  }
  else*/
  residual_ ->Norm2(&resnorm_L2);
  increment_->Norm2(&phiincnorm_L2);
  phinp_    ->Norm2(&phinorm_L2);

  // check for any INF's and NaN's
  if (std::isnan(resnorm_L2) or
      std::isnan(phiincnorm_L2) or
      std::isnan(phinorm_L2))
    dserror("At least one of the calculated vector norms is NaN.");

  if (abs(std::isinf(resnorm_L2)) or
      abs(std::isinf(phiincnorm_L2)) or
      abs(std::isinf(phinorm_L2)))
    dserror("At least one of the calculated vector norms is INF.");

  // for scalar norm being (close to) zero, set to one
  if (phinorm_L2 < 1e-5) phinorm_L2 = 1.0;

  if (myrank_==0)
  {
    printf("+------------+-------------------+--------------+--------------+\n");
    printf("|- step/max -|- tol      [norm] -|- residual   -|- scalar-inc -|\n");
    printf("|  %3d/%3d   | %10.3E[L_2 ]  | %10.3E   | %10.3E   |",
         itnum,itmax,ittol,resnorm_L2,phiincnorm_L2/phinorm_L2);
    printf("\n");
    printf("+------------+-------------------+--------------+--------------+\n");
  }

  if ((resnorm_L2 <= ittol) and
      (phiincnorm_L2/phinorm_L2 <= ittol)) stopnonliniter=true;

  // warn if itemax is reached without convergence, but proceed to next timestep
  if ((itnum == itmax) and
      ((resnorm_L2 > ittol) or (phiincnorm_L2/phinorm_L2 > ittol)))
  {
    stopnonliniter=true;
    if (myrank_==0)
    {
      printf("|            >>>>>> not converged in itemax steps!             |\n");
      printf("+--------------------------------------------------------------+\n");
    }
  }

  return stopnonliniter;
}


/*----------------------------------------------------------------------*
 | write state vectors to Gmsh postprocessing files        henke   12/09|
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::OutputToGmsh(
    const int step,
    const double time
    ) const
{
  // turn on/off screen output for writing process of Gmsh postprocessing file
  const bool screen_out = true;

  // create Gmsh postprocessing file
  const std::string filename = IO::GMSH::GetNewFileNameAndDeleteOldFiles("solution_field_scalar", step, 500, screen_out, discret_->Comm().MyPID());
  std::ofstream gmshfilecontent(filename.c_str());
//  {
//    // add 'View' to Gmsh postprocessing file
//    gmshfilecontent << "View \" " << "Phin \" {" << endl;
//    // draw scalar field 'Phindtp' for every element
//    IO::GMSH::ScalarFieldToGmsh(discret_,phin_,gmshfilecontent);
//    gmshfilecontent << "};" << endl;
//  }
  {
    // add 'View' to Gmsh postprocessing file
    gmshfilecontent << "View \" " << "Phinp \" {" << endl;
    // draw scalar field 'Phinp' for every element
    IO::GMSH::ScalarFieldToGmsh(discret_,phinp_,gmshfilecontent);
    gmshfilecontent << "};" << endl;
  }
//  {
//    // add 'View' to Gmsh postprocessing file
//    gmshfilecontent << "View \" " << "Phidtn \" {" << endl;
//    // draw scalar field 'Phindtn' for every element
//    IO::GMSH::ScalarFieldToGmsh(discret_,phidtn_,gmshfilecontent);
//    gmshfilecontent << "};" << endl;
//  }
//  {
//    // add 'View' to Gmsh postprocessing file
//    gmshfilecontent << "View \" " << "Phidtnp \" {" << endl;
//    // draw scalar field 'Phindtp' for every element
//    IO::GMSH::ScalarFieldToGmsh(discret_,phidtnp_,gmshfilecontent);
//    gmshfilecontent << "};" << endl;
//  }
  {
    // add 'View' to Gmsh postprocessing file
    gmshfilecontent << "View \" " << "Convective Velocity \" {" << endl;
    // draw vector field 'Convective Velocity' for every element
    IO::GMSH::VectorFieldNodeBasedToGmsh(discret_,convel_,gmshfilecontent);
    gmshfilecontent << "};" << endl;
  }
  gmshfilecontent.close();
  if (screen_out) std::cout << " done" << endl;
}


/*----------------------------------------------------------------------*
 |  output of some mean values                               gjb   01/09|
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::OutputMeanScalars()
{
  // set scalar values needed by elements
  discret_->ClearState();
  discret_->SetState("phinp",phinp_);
  // set action for elements
  ParameterList eleparams;
  eleparams.set("action","calc_mean_scalars");
  eleparams.set("inverting",false);
  eleparams.set<int>("scatratype",scatratype_);

  //provide displacement field in case of ALE
  eleparams.set("isale",isale_);
  if (isale_)
    AddMultiVectorToParameterList(eleparams,"dispnp",dispnp_);

  // evaluate integrals of scalar(s) and domain
  Teuchos::RCP<Epetra_SerialDenseVector> scalars
    = Teuchos::rcp(new Epetra_SerialDenseVector(numscal_+1));
  discret_->EvaluateScalars(eleparams, scalars);
  discret_->ClearState();   // clean up

  const double domint = (*scalars)[numscal_];

  // print out values
  if (myrank_ == 0)
  {
    if (scatratype_==INPAR::SCATRA::scatratype_loma)
      cout << "Mean scalar: " << (*scalars)[0]/domint << endl;
    else
    {
      cout << "Domain integral:          " << domint << endl;
      for (int k = 0; k < numscal_; k++)
      {
        //cout << "Total concentration (c_"<<k+1<<"): "<< (*scalars)[k] << endl;
        cout << "Mean concentration (c_"<<k+1<<"): "<< (*scalars)[k]/domint << endl;
      }
    }
  }

  // print out results to file as well
  if (myrank_ == 0)
  {
    const std::string fname
    = DRT::Problem::Instance()->OutputControlFile()->FileName()+".meanvalues.txt";

    std::ofstream f;
    if (Step() <= 1)
    {
      f.open(fname.c_str(),std::fstream::trunc);
      if (scatratype_==INPAR::SCATRA::scatratype_loma)
        f << "#| Step | Time | Mean scalar |\n";
      else
      {
        f << "#| Step | Time | Domain integral ";
        for (int k = 0; k < numscal_; k++)
        {
          f << "| Mean concentration (c_"<<k+1<<") ";
        }
        f << "\n";
      }
    }
    else
      f.open(fname.c_str(),std::fstream::ate | std::fstream::app);

    f << Step() << " " << Time() << " ";
    if (scatratype_==INPAR::SCATRA::scatratype_loma)
      f << (*scalars)[0]/domint << "\n";
    else
    {
      f << domint << " ";
      for (int k = 0; k < numscal_; k++)
      {
        f << (*scalars)[k]/domint << " ";
      }
      f << "\n";
    }
    f.flush();
    f.close();
  }

  return;
}


/*----------------------------------------------------------------------*
 |  output of electrode status information to screen         gjb  01/09 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::OutputElectrodeInfo(
    bool printtoscreen,
    bool printtofile)
{
  // evaluate the following type of boundary conditions:
  std::string condname("ElectrodeKinetics");
  vector<DRT::Condition*> cond;
  discret_->GetCondition(condname,cond);

  // leave method, if there's nothing to do!
  if (!cond.size()) return;

  double sum(0.0);

  if ((myrank_ == 0) and printtoscreen)
  {
    cout<<"Status of '"<<condname<<"':\n"
    <<"++----+---------------------+------------------+----------------------+--------------------+----------------+----------------+"<<endl;
    printf("|| ID |    Total current    | Area of boundary | Mean current density | Mean overpotential | Electrode pot. | Mean Concentr. |\n");
  }

  // first, add to all conditions of interest a ConditionID
  for (int condid = 0; condid < (int) cond.size(); condid++)
  {
    // is there already a ConditionID?
    const vector<int>*    CondIDVec  = cond[condid]->Get<vector<int> >("ConditionID");
    if (CondIDVec)
    {
      if ((*CondIDVec)[0] != condid)
        dserror("Condition %s has non-matching ConditionID",condname.c_str());
    }
    else
    {
      // let's add a ConditionID
      cond[condid]->Add("ConditionID",condid);
    }
  }
  // now we evaluate the conditions and separate via ConditionID
  for (int condid = 0; condid < (int) cond.size(); condid++)
  {
    double currtangent(0.0); // this value remains unused here!
    double currresidual(0.0); // this value remains unused here!
    double electrodesurface(0.0); // this value remains unused here!

    OutputSingleElectrodeInfo(
        cond[condid],
        condid,
        printtoscreen,
        printtofile,
        sum,
        currtangent,
        currresidual,
        electrodesurface);
  } // loop over condid

  if ((myrank_==0) and printtoscreen)
  {
    cout<<"++----+---------------------+------------------+----------------------+--------------------+----------------+----------------+"<<endl;
    // print out the net total current for all indicated boundaries
    printf("Net total current over boundary: %10.3E\n\n",sum);
  }

  // clean up
  discret_->ClearState();

  return;
} // ScaTraImplicitTimeInt::OutputElectrodeInfo


/*----------------------------------------------------------------------*
 |  get electrode status for single boundary condition       gjb  11/09 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::OutputSingleElectrodeInfo(
    DRT::Condition* condition,
    const int condid,
    const bool printtoscreen,
    const bool printtofile,
    double& currentsum,
    double& currtangent,
    double& currresidual,
    double& electrodesurface)
{
  // safety check: is there already a ConditionID?
  const vector<int>* CondIDVec  = condition->Get<vector<int> >("ConditionID");
  if (not CondIDVec) dserror("Condition has not yet a ConditionID");

  // set vector values needed by elements
  discret_->ClearState();
  discret_->SetState("phinp",phinp_);
  discret_->SetState("timederivative",phidtnp_); // needed for double-layer capacity!

  // set action for elements
  ParameterList eleparams;
  eleparams.set("action","calc_elch_electrode_kinetics");
  eleparams.set<int>("scatratype",scatratype_);
  eleparams.set("calc_status",true); // just want to have a status ouput!
  eleparams.set("frt",frt_);

  //provide displacement field in case of ALE
  eleparams.set("isale",isale_);
  if (isale_)
    AddMultiVectorToParameterList(eleparams,"dispnp",dispnp_);

  // Since we just want to have the status ouput for t_{n+1},
  // we have to take care for Gen.Alpha!
  // AddSpecificTimeIntegrationParameters cannot be used since we do not want
  // an evaluation for t_{n+\alpha_f} !!!

  // add element parameters according to time-integration scheme
  AddSpecificTimeIntegrationParameters(eleparams);

  // values to be computed
  eleparams.set("currentintegral",0.0);
  eleparams.set("boundaryintegral",0.0);
  eleparams.set("overpotentialintegral",0.0);
  eleparams.set("concentrationintegral",0.0);
  eleparams.set("currentderiv",0.0);
  eleparams.set("currentresidual",0.0);

  // would be nice to have a EvaluateScalar for conditions!!!
  discret_->EvaluateCondition(eleparams,Teuchos::null,Teuchos::null,Teuchos::null,Teuchos::null,Teuchos::null,"ElectrodeKinetics",condid);

  // get integral of current on this proc
  double currentintegral = eleparams.get<double>("currentintegral");
  // get area of the boundary on this proc
  double boundaryint = eleparams.get<double>("boundaryintegral");
  // get integral of overpotential on this proc
  double overpotentialint = eleparams.get<double>("overpotentialintegral");
  // get integral of reactant concentration on this proc
  double cint = eleparams.get<double>("concentrationintegral");
  // tangent of current w.r.t. electrode potential on this proc
  double currderiv = eleparams.get<double>("currentderiv");
  // get negative current residual (rhs of galvanostatic balance equation)
  double currentresidual = eleparams.get<double>("currentresidual");

  // care for the parallel case
  double parcurrentintegral = 0.0;
  discret_->Comm().SumAll(&currentintegral,&parcurrentintegral,1);
  double parboundaryint = 0.0;
  discret_->Comm().SumAll(&boundaryint,&parboundaryint,1);
  double paroverpotentialint = 0.0;
  discret_->Comm().SumAll(&overpotentialint,&paroverpotentialint,1);
  double parcint = 0.0;
  discret_->Comm().SumAll(&cint,&parcint,1);
  double parcurrderiv = 0.0;
  discret_->Comm().SumAll(&currderiv,&parcurrderiv ,1);
  double parcurrentresidual = 0.0;
  discret_->Comm().SumAll(&currentresidual,&parcurrentresidual ,1);

  // access some parameters of the actual condition
  double pot = condition->GetDouble("pot");
  const int curvenum = condition->GetInt("curve");
  if (curvenum>=0)
  {
    const double curvefac = DRT::Problem::Instance()->Curve(curvenum).f(time_);
    // adjust potential at metal side accordingly
    pot *= curvefac;
  }

  // specify some return values
  currentsum += parcurrentintegral; // sum of currents
  currtangent  = parcurrderiv;      // tangent w.r.t. electrode potential on metal side
  currresidual = parcurrentresidual;
  electrodesurface = parboundaryint;

  // clean up
  discret_->ClearState();

  // print out results to screen/file if desired
  if (myrank_ == 0)
  {
    if (printtoscreen) // print out results to screen
    {
      printf("|| %2d |     %10.3E      |    %10.3E    |      %10.3E      |     %10.3E     |   %10.3E   |   %10.3E   |\n",
          condid,parcurrentintegral,parboundaryint,parcurrentintegral/parboundaryint,paroverpotentialint/parboundaryint, pot, parcint/parboundaryint);
    }

    if (printtofile)// write results to file
    {
      ostringstream temp;
      temp << condid;
      const std::string fname
      = DRT::Problem::Instance()->OutputControlFile()->FileName()+".electrode_status_"+temp.str()+".txt";

      std::ofstream f;
      if (Step() <= 1)
      {
        f.open(fname.c_str(),std::fstream::trunc);
        f << "#| ID | Step | Time | Total current | Area of boundary | Mean current density | Mean overpotential | Electrode pot. | Mean Concentr. |\n";
      }
      else
        f.open(fname.c_str(),std::fstream::ate | std::fstream::app);

      f << condid << " " << Step() << " " << Time() << " " << parcurrentintegral << " " << parboundaryint
      << " " << parcurrentintegral/parboundaryint << " " << paroverpotentialint/parboundaryint << " "
      << pot << " " << parcint/parboundaryint << " " <<"\n";
      f.flush();
      f.close();
    }
  } // if (myrank_ == 0)

  return;
}


/*----------------------------------------------------------------------*
 |  write mass / heat flux vector to BINIO                   gjb   08/08|
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::OutputFlux(RCP<Epetra_MultiVector> flux)
{
  //safety check
  if (flux == Teuchos::null)
    dserror("Null pointer for flux vector output. Output() called before Update() ??");

  // WORK-AROUND FOR NURBS DISCRETIZATIONS
  // using noderowmap is problematic. Thus, we do not add normal vectors
  // to the scalar result field (scalar information is enough anyway)
  DRT::NURBS::NurbsDiscretization* nurbsdis
  = dynamic_cast<DRT::NURBS::NurbsDiscretization*>(&(*discret_));
  if(nurbsdis!=NULL)
  {
    RCP<Epetra_Vector> normalflux = rcp(((*flux)(0)),false);
    output_->WriteVector("normalflux", normalflux, IO::DiscretizationWriter::dofvector);
    return; // leave here
  }

  // post_drt_ensight does not support multivectors based on the dofmap
  // for now, I create single vectors that can be handled by the filter

  // get the noderowmap
  const Epetra_Map* noderowmap = discret_->NodeRowMap();
  Teuchos::RCP<Epetra_MultiVector> fluxk = rcp(new Epetra_MultiVector(*noderowmap,3,true));
  for (vector<int>::iterator it = writefluxids_.begin(); it!=writefluxids_.end(); ++it)
  {
    int k=(*it);

    ostringstream temp;
    temp << k;
    string name = "flux_phi_"+temp.str();
    for (int i = 0;i<fluxk->MyLength();++i)
    {
      DRT::Node* actnode = discret_->lRowNode(i);
      int dofgid = discret_->Dof(actnode,k-1);
      // get value for each component of flux vector
      double xvalue = ((*flux)[0])[(flux->Map()).LID(dofgid)];
      double yvalue = ((*flux)[1])[(flux->Map()).LID(dofgid)];
      double zvalue = ((*flux)[2])[(flux->Map()).LID(dofgid)];
      // care for the slave nodes of rotationally symm. periodic boundary conditions
      double rotangle(0.0); //already converted to radians
      bool havetorotate = FLD::IsSlaveNodeOfRotSymPBC(actnode,rotangle);
      if (havetorotate)
      {
        double xvalue_rot = (xvalue*cos(rotangle)) - (yvalue*sin(rotangle));
        double yvalue_rot = (xvalue*sin(rotangle)) + (yvalue*(cos(rotangle)));
        xvalue = xvalue_rot;
        yvalue = yvalue_rot;
      }
      // insert values
      fluxk->ReplaceMyValue(i,0,xvalue);
      fluxk->ReplaceMyValue(i,1,yvalue);
      fluxk->ReplaceMyValue(i,2,zvalue);
    }
    if (numscal_==1)
      output_->WriteVector("flux", fluxk, IO::DiscretizationWriter::nodevector);
    else
      output_->WriteVector(name, fluxk, IO::DiscretizationWriter::nodevector);
  }
  // that's it
  return;
}


/*----------------------------------------------------------------------*
 |  calculate mass / heat flux vector                        gjb   04/08|
 *----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_MultiVector> SCATRA::ScaTraTimIntImpl::CalcFlux(const bool writetofile)
{
  switch(writeflux_)
  {
  case INPAR::SCATRA::flux_total_domain:
  case INPAR::SCATRA::flux_diffusive_domain:
  {
    return CalcFluxInDomain(writeflux_);
    break;
  }
  case INPAR::SCATRA::flux_total_boundary:
  case INPAR::SCATRA::flux_diffusive_boundary:
  {
    // calculate normal flux vector field only for the user-defined boundary conditions:
    vector<std::string> condnames;
    condnames.push_back("ScaTraFluxCalc");

    return CalcFluxAtBoundary(condnames, writetofile);
    break;
  }
  default:
    break;
  }
  // else: we just return a zero vector field (needed for result testing)
  const Epetra_Map* dofrowmap = discret_->DofRowMap();
  return rcp(new Epetra_MultiVector(*dofrowmap,3,true));
}


/*----------------------------------------------------------------------*
 |  calculate mass / heat flux vector field in comp. domain    gjb 06/09|
 *----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_MultiVector> SCATRA::ScaTraTimIntImpl::CalcFluxInDomain
(const INPAR::SCATRA::FluxType fluxtype)
{
  // get a vector layout from the discretization to construct matching
  // vectors and matrices    local <-> global dof numbering
  const Epetra_Map* dofrowmap = discret_->DofRowMap();

  // empty vector for (normal) mass or heat flux vectors (always 3D)
  Teuchos::RCP<Epetra_MultiVector> flux = rcp(new Epetra_MultiVector(*dofrowmap,3,true));

  // We have to treat each spatial direction separately
  Teuchos::RCP<Epetra_Vector> fluxx = LINALG::CreateVector(*dofrowmap,true);
  Teuchos::RCP<Epetra_Vector> fluxy = LINALG::CreateVector(*dofrowmap,true);
  Teuchos::RCP<Epetra_Vector> fluxz = LINALG::CreateVector(*dofrowmap,true);

  // we need a vector for the integrated shape functions
  Teuchos::RCP<Epetra_Vector> integratedshapefcts = LINALG::CreateVector(*dofrowmap,true);

  {
    ParameterList eleparams;
    eleparams.set("action","integrate_shape_functions");
    eleparams.set<int>("scatratype",scatratype_);
    // we integrate shape functions for the first numscal_ dofs per node!!
    Epetra_IntSerialDenseVector dofids(7); // make it big enough!
    for(int rr=0;rr < numscal_;rr++)
    {
      dofids(rr) = rr;
    }
    for(int rr=numscal_;rr<7;rr++)
    {
      dofids(rr) = -1; // do not integrate shape functions for these dofs
    }
    eleparams.set("dofids",dofids);
    eleparams.set("isale",isale_);
    if (isale_)
      AddMultiVectorToParameterList(eleparams,"dispnp",dispnp_);
    // evaluate fluxes in the whole computational domain
    // (e.g., for visualization of particle path-lines) or L2 projection for better consistency
    discret_->Evaluate(eleparams,Teuchos::null,Teuchos::null,integratedshapefcts,Teuchos::null,Teuchos::null);
  }

  // set action for elements
  ParameterList params;
  params.set("action","calc_condif_flux");
  params.set<int>("scatratype",scatratype_);
  params.set("frt",frt_);
  params.set<int>("fluxtype",fluxtype);

  // provide velocity field and potentially acceleration/pressure field
  // (export to column map necessary for parallel evaluation)
  AddMultiVectorToParameterList(params,"velocity field",convel_);
  AddMultiVectorToParameterList(params,"acceleration/pressure field",accpre_);

  //provide displacement field in case of ALE
  params.set("isale",isale_);
  if (isale_)
    AddMultiVectorToParameterList(params,"dispnp",dispnp_);

  // parameters for stabilization
  params.sublist("STABILIZATION") = params_->sublist("STABILIZATION");

  // set vector values needed by elements
  discret_->ClearState();
  discret_->SetState("phinp",phinp_);

  // evaluate fluxes in the whole computational domain (e.g., for visualization of particle path-lines)
  discret_->Evaluate(params,Teuchos::null,Teuchos::null,fluxx,fluxy,fluxz);

  // insert values into final flux vector for visualization
  // we do not solve a global equation system for the flux values here
  // but perform a lumped mass matrix approach, i.e., dividing by the values of
  // integrated shape functions
  for (int i = 0;i<flux->MyLength();++i)
  {
    const double intshapefct = (*integratedshapefcts)[i];
    // is zero at electric potential dofs
    if (abs(intshapefct) > EPS13)
    {
      flux->ReplaceMyValue(i,0,((*fluxx)[i])/intshapefct);
      flux->ReplaceMyValue(i,1,((*fluxy)[i])/intshapefct);
      flux->ReplaceMyValue(i,2,((*fluxz)[i])/intshapefct);
    }
  }

  // clean up
  discret_->ClearState();

  return flux;
}


/*----------------------------------------------------------------------*
 |  calculate mass / heat normal flux at specified boundaries  gjb 06/09|
 *----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_MultiVector> SCATRA::ScaTraTimIntImpl::CalcFluxAtBoundary(
    std::vector<string>& condnames,
    const bool writetofile)
{
  // The normal flux calculation is based on the idea proposed in
  // GRESHO ET AL.,
  // "THE CONSISTENT GALERKIN FEM FOR COMPUTING DERIVED BOUNDARY
  // QUANTITIES IN THERMAL AND/OR FLUIDS PROBLEMS",
  // INTERNATIONAL JOURNAL FOR NUMERICAL METHODS IN FLUIDS, VOL. 7, 371-394 (1987)
  // For the moment, we are lumping the 'boundary mass matrix' instead of solving
  // a small linear system!


  // get a vector layout from the discretization to construct matching
  // vectors and matrices
  //                 local <-> global dof numbering
  const Epetra_Map* dofrowmap = discret_->DofRowMap();

  // empty vector for (normal) mass or heat flux vectors (always 3D)
  Teuchos::RCP<Epetra_MultiVector> flux = rcp(new Epetra_MultiVector(*dofrowmap,3,true));

  // determine the averaged normal vector field for indicated boundaries
  // used for the output of the normal flux as a vector field
  // is computed only once; for ALE formulation recalculation is necessary
  if ((normals_ == Teuchos::null) or (isale_== true))
    normals_ = ComputeNormalVectors(condnames);

  // was the residual already prepared? (Important only for the result test)
  if ((solvtype_!=INPAR::SCATRA::solvertype_nonlinear) and (lastfluxoutputstep_ != step_))
  {
    lastfluxoutputstep_ = step_;

    // For nonlinear problems we already have the actual residual vector
    // from the last convergence test!
    // For linear problems we have to compute this information first, since
    // the residual (w.o. Neumann boundary) has not been computed after the last solve!

    // zero out matrix entries
    sysmat_->Zero();

    // zero out residual vector
    residual_->PutScalar(0.0);

    ParameterList eleparams;
    // action for elements
    eleparams.set("action","calc_condif_systemmat_and_residual");

    // other parameters that might be needed by the elements
    eleparams.set("time-step length",dta_);
    eleparams.set<int>("scatratype",scatratype_);
    eleparams.set("incremental solver",true); // say yes and you get the residual!!
    eleparams.set<int>("form of convective term",convform_);
    eleparams.set<int>("fs subgrid diffusivity",fssgd_);
    eleparams.set("turbulence model",turbmodel_);
    eleparams.set("frt",frt_);

    // provide velocity field and potentially acceleration/pressure field
    // (export to column map necessary for parallel evaluation)
    AddMultiVectorToParameterList(eleparams,"velocity field",convel_);
    AddMultiVectorToParameterList(eleparams,"acceleration/pressure field",accpre_);

    //provide displacement field in case of ALE
    eleparams.set("isale",isale_);
    if (isale_)
      AddMultiVectorToParameterList(eleparams,"dispnp",dispnp_);

    // parameters for stabilization
    eleparams.sublist("STABILIZATION") = params_->sublist("STABILIZATION");

    // clear state
    discret_->ClearState();

    // we have to perform some dirty action here...
    bool incremental_old = incremental_;
    incremental_ = true;
    // add element parameters according to time-integration scheme
    AddSpecificTimeIntegrationParameters(eleparams);
    // undo
    incremental_ = incremental_old;

    {
      // call standard loop over elements
      discret_->Evaluate(eleparams,sysmat_,null,residual_,null,null);
      discret_->ClearState();
    }

    // scaling to get true residual vector for all time integration schemes
    trueresidual_->Update(ResidualScaling(),*residual_,0.0);

  } // if ((solvtype_!=INPAR::SCATRA::solvertype_nonlinear) && (lastfluxoutputstep_ != step_))

  // if total flux is desired add the convective flux contribution
  // to the trueresidual_ now.
  if(writeflux_==INPAR::SCATRA::flux_total_boundary)
  {
    if (myrank_==0)
      cout<<"Convective flux contribution is added to trueresidual_ vector.\n"
      "Be sure not to address the same boundary part twice!\n";

    // now we evaluate the conditions and separate via ConditionID
    for (unsigned int i=0; i < condnames.size(); i++)
    {
      vector<DRT::Condition*> cond;
      discret_->GetCondition(condnames[i],cond);

      discret_->ClearState();
      ParameterList params;

      params.set("action","add_convective_mass_flux");
      params.set<int>("scatratype",scatratype_);

      // add element parameters according to time-integration scheme
      AddSpecificTimeIntegrationParameters(params);

      // provide velocity field
      // (export to column map necessary for parallel evaluation)
      AddMultiVectorToParameterList(params,"velocity field",convel_);

      //provide displacement field in case of ALE
      params.set("isale",isale_);
      if (isale_)
        AddMultiVectorToParameterList(params,"dispnp",dispnp_);

      // call loop over boundary elements and add integrated fluxes to trueresidual_
      discret_->EvaluateCondition(params,trueresidual_,condnames[i]);
      discret_->ClearState();
    }
  }

  vector<double> normfluxsum(numscal_);

  for (unsigned int i=0; i < condnames.size(); i++)
  {
    vector<DRT::Condition*> cond;
    discret_->GetCondition(condnames[i],cond);

    // go to the next condition type, if there's nothing to do!
    if (!cond.size()) continue;

    if (myrank_ == 0)
    {
      cout<<"Normal fluxes at boundary '"<<condnames[i]<<"':\n"
      <<"+----+-----+-------------------------+------------------+--------------------------+"<<endl;
      printf("| ID | DOF | Integral of normal flux | Area of boundary | Mean normal flux density |\n");
    }

    // first, add to all conditions of interest a ConditionID
    for (int condid = 0; condid < (int) cond.size(); condid++)
    {
      // is there already a ConditionID?
      const vector<int>*    CondIDVec  = cond[condid]->Get<vector<int> >("ConditionID");
      if (CondIDVec)
      {
        if ((*CondIDVec)[0] != condid)
          dserror("Condition %s has non-matching ConditionID",condnames[i].c_str());
      }
      else
      {
        // let's add a ConditionID
        cond[condid]->Add("ConditionID",condid);
      }
    }

    // now we evaluate the conditions and separate via ConditionID
    for (int condid = 0; condid < (int) cond.size(); condid++)
    {
      ParameterList params;

      // calculate integral of shape functions over indicated boundary and it's area
      params.set("boundaryint",0.0);
      params.set("action","integrate_shape_functions");
      params.set<int>("scatratype",scatratype_);

      //provide displacement field in case of ALE
      params.set("isale",isale_);
      if (isale_)
        AddMultiVectorToParameterList(params,"dispnp",dispnp_);

      // create vector (+ initialization with zeros)
      Teuchos::RCP<Epetra_Vector> integratedshapefunc = LINALG::CreateVector(*dofrowmap,true);

      // call loop over elements
      discret_->ClearState();
      discret_->EvaluateCondition(params,integratedshapefunc,condnames[i],condid);
      discret_->ClearState();

      vector<double> normfluxintegral(numscal_);

      // insert values into final flux vector for visualization
      int numrownodes = discret_->NumMyRowNodes();
      for (int lnodid = 0; lnodid < numrownodes; ++lnodid )
      {
        DRT::Node* actnode = discret_->lRowNode(lnodid);
        for (int idof = 0; idof < numscal_; ++idof)
        {
          int dofgid = discret_->Dof(actnode,idof);
          int doflid = dofrowmap->LID(dofgid);

          if ((*integratedshapefunc)[doflid] != 0.0)
          {
            // this is the value of the normal flux density
            double normflux = ((*trueresidual_)[doflid])/(*integratedshapefunc)[doflid];
            // compute integral value for every degree of freedom
            normfluxintegral[idof] += (*trueresidual_)[doflid];

            // care for the slave nodes of rotationally symm. periodic boundary conditions
            double rotangle(0.0);
            bool havetorotate = FLD::IsSlaveNodeOfRotSymPBC(actnode,rotangle);

            // do not insert slave node values here, since they would overwrite the
            // master node values owning the same dof
            // (rotation of slave node vectors is performed later during output)
            if (not havetorotate)
            {
              // for visualization, we plot the normal flux with
              // outward pointing normal vector
              for (int idim = 0; idim < 3; idim++)
              {
                Epetra_Vector* normalcomp = (*normals_)(idim);
                double normalveccomp =(*normalcomp)[lnodid];
                flux->ReplaceMyValue(doflid,idim,normflux*normalveccomp);
              }
            }
          }
        }
      }

      // get area of the boundary on this proc
      double boundaryint = params.get<double>("boundaryint");

      // care for the parallel case
      vector<double> parnormfluxintegral(numscal_);
      discret_->Comm().SumAll(&normfluxintegral[0],&parnormfluxintegral[0],numscal_);
      double parboundaryint = 0.0;
      discret_->Comm().SumAll(&boundaryint,&parboundaryint,1);

      for (int idof = 0; idof < numscal_; ++idof)
      {
        // print out results
        if (myrank_ == 0)
        {
          printf("| %2d | %2d  |       %10.4E        |    %10.4E    |        %10.4E        |\n",
              condid,idof,parnormfluxintegral[idof],parboundaryint,parnormfluxintegral[idof]/parboundaryint);
        }
        normfluxsum[idof]+=parnormfluxintegral[idof];
      }

      // statistics section for normfluxintegral
      if (DoBoundaryFluxStatistics())
      {
        // add current flux value to the sum!
        (*sumnormfluxintegral_)[condid] += parnormfluxintegral[0]; // only first scalar!
        int samstep = step_-samstart_+1;

        // dump every dumperiod steps (i.e., write to screen)
        bool dumpit(false);
        if (dumperiod_==0)
          {dumpit=true;}
        else
          {if(samstep%dumperiod_==0) dumpit=true;}

        if(dumpit)
        {
          double meannormfluxintegral = (*sumnormfluxintegral_)[condid]/samstep;
          // dump statistical results
          if (myrank_ == 0)
          {
            printf("| %2d | Mean normal-flux integral (step %5d -- step %5d) :   %12.5E |\n", condid,samstart_,step_,meannormfluxintegral);
          }
        }
      }

      // print out results to file as well (only if really desired)
      if ((myrank_ == 0) and (writetofile==true))
      {
        ostringstream temp;
        temp << condid;
        const std::string fname
        = DRT::Problem::Instance()->OutputControlFile()->FileName()+".boundaryflux_"+condnames[i]+"_"+temp.str()+".txt";

        std::ofstream f;
        if (Step() <= 1)
        {
          f.open(fname.c_str(),std::fstream::trunc);
          f << "#| ID | Step | Time | Area of boundary |";
          for(int idof = 0; idof < numscal_; ++idof)
          {
            f<<" Integral of normal flux "<<idof<<" | Mean normal flux density "<<idof<<" |";
          }
          f<<"\n";
        }
        else
          f.open(fname.c_str(),std::fstream::ate | std::fstream::app);

        f << condid << " " << Step() << " " << Time() << " "<< parboundaryint<< " ";
        for (int idof = 0; idof < numscal_; ++idof)
        {
          f << parnormfluxintegral[idof] << " "<< parnormfluxintegral[idof]/parboundaryint<< " ";
        }
        f << "\n";
        f.flush();
        f.close();
      } // write to file

    } // loop over condid

    if (myrank_==0)
      cout<<"+----+-----+-------------------------+------------------+--------------------------+"<<endl;
  }

  // print out the accumulated normal flux over all indicated boundaries
  if (myrank_ == 0)
  {
    for (int idof = 0; idof < numscal_; ++idof)
    {
      printf("Sum of all normal flux boundary integrals for scalar %d: %10.5E\n",idof,normfluxsum[idof]);
    }
    cout<<endl;
  }
  // clean up
  discret_->ClearState();

  return flux;
}


/*----------------------------------------------------------------------*
 | compute outward pointing unit normal vectors at given b.c.  gjb 01/09|
 *----------------------------------------------------------------------*/
RCP<Epetra_MultiVector> SCATRA::ScaTraTimIntImpl::ComputeNormalVectors(
    const vector<string>& condnames
)
{
  // create vectors for x,y and z component of average normal vector field
  // get noderowmap of discretization
  const Epetra_Map* noderowmap = discret_->NodeRowMap();
  RCP<Epetra_MultiVector> normal = rcp(new Epetra_MultiVector(*noderowmap,3,true));

  discret_->ClearState();

  // set action for elements
  ParameterList eleparams;
  eleparams.set("action","calc_normal_vectors");
  eleparams.set<int>("scatratype",scatratype_);
  eleparams.set<RCP<Epetra_MultiVector> >("normal vectors",normal);

  //provide displacement field in case of ALE
  eleparams.set("isale",isale_);
  if (isale_)
    AddMultiVectorToParameterList(eleparams,"dispnp",dispnp_);

  // loop over all intended types of conditions
  for (unsigned int i=0; i < condnames.size(); i++)
  {
    discret_->EvaluateCondition(eleparams,condnames[i]);
  }

  // clean up
  discret_->ClearState();

  // the normal vector field is not properly scaled up to now. We do this here
  int numrownodes = discret_->NumMyRowNodes();
  Epetra_Vector* xcomp = (*normal)(0);
  Epetra_Vector* ycomp = (*normal)(1);
  Epetra_Vector* zcomp = (*normal)(2);
  for (int i=0; i<numrownodes; ++i)
  {
    double x = (*xcomp)[i];
    double y = (*ycomp)[i];
    double z = (*zcomp)[i];
    double norm = sqrt(x*x + y*y + z*z);
    // form the unit normal vector
    if (norm > EPS15)
    {
      normal->ReplaceMyValue(i,0,x/norm);
      normal->ReplaceMyValue(i,1,y/norm);
      normal->ReplaceMyValue(i,2,z/norm);
    }
  }

  return normal;
}


/*----------------------------------------------------------------------*
 |  calculate error compared to analytical solution            gjb 10/08|
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::EvaluateErrorComparedToAnalyticalSol()
{
  const INPAR::SCATRA::CalcError calcerr
    = DRT::INPUT::IntegralValue<INPAR::SCATRA::CalcError>(*params_,"CALCERROR");

  switch (calcerr)
  {
  case INPAR::SCATRA::calcerror_no: // do nothing (the usual case)
    break;
  case INPAR::SCATRA::calcerror_Kwok_Wu:
  {
    //------------------------------------------------ Kwok and Wu,1995
    //   Reference:
    //   Kwok, Yue-Kuen and Wu, Charles C. K.
    //   "Fractional step algorithm for solving a multi-dimensional
    //   diffusion-migration equation"
    //   Numerical Methods for Partial Differential Equations
    //   1995, Vol 11, 389-397

    // create the parameters for the discretization
    ParameterList p;

    // parameters for the elements
    p.set("action","calc_error");
    p.set<int>("scatratype",scatratype_);
    p.set("total time",time_);
    p.set("frt",frt_);
    p.set<int>("calcerrorflag",calcerr);
    //provide displacement field in case of ALE
    p.set("isale",isale_);
    if (isale_)
      AddMultiVectorToParameterList(p,"dispnp",dispnp_);

    // set vector values needed by elements
    discret_->ClearState();
    discret_->SetState("phinp",phinp_);

    // get (squared) error values
    Teuchos::RCP<Epetra_SerialDenseVector> errors
      = Teuchos::rcp(new Epetra_SerialDenseVector(3));
    discret_->EvaluateScalars(p, errors);
    discret_->ClearState();

    // for the L2 norm, we need the square root
    double conerr1 = sqrt((*errors)[0]);
    double conerr2 = sqrt((*errors)[1]);
    double poterr  = sqrt((*errors)[2]);

    if (myrank_ == 0)
    {
      printf("\nL2_err for Kwok and Wu:\n");
      printf(" concentration1 %15.8e\n concentration2 %15.8e\n potential      %15.8e\n\n",
             conerr1,conerr2,poterr);
    }
  }
  break;
  case INPAR::SCATRA::calcerror_cylinder:
  {
    //   Reference:
    //   G. Bauer, V. Gravemeier, W.A. Wall, A 3D finite element approach for the coupled
    //   numerical simulation of electrochemical systems and fluid flow,
    //   International Journal for Numerical Methods in Engineering, 2011

    // create the parameters for the discretization
    ParameterList p;

    // parameters for the elements
    p.set("action","calc_error");
    p.set<int>("scatratype",scatratype_);
    p.set("total time",time_);
    p.set("frt",frt_);
    p.set<int>("calcerrorflag",calcerr);
    //provide displacement field in case of ALE
    p.set("isale",isale_);
    if (isale_)
      AddMultiVectorToParameterList(p,"dispnp",dispnp_);

    // set vector values needed by elements
    discret_->ClearState();
    discret_->SetState("phinp",phinp_);

    // get (squared) error values
    Teuchos::RCP<Epetra_SerialDenseVector> errors
      = Teuchos::rcp(new Epetra_SerialDenseVector(3));
    discret_->EvaluateScalars(p, errors);
    discret_->ClearState();

    // for the L2 norm, we need the square root
    double conerr1 = sqrt((*errors)[0]);
    double conerr2 = sqrt((*errors)[1]);
    double poterr  = sqrt((*errors)[2]);

    if (myrank_ == 0)
    {
      printf("\nL2_err for concentric cylinders:\n");
      printf(" concentration1 %15.8e\n concentration2 %15.8e\n potential      %15.8e\n\n",
             conerr1,conerr2,poterr);
    }
  }
  break;
  default:
    dserror("Cannot calculate error. Unknown type of analytical test problem");
  }
  return;
}


/*----------------------------------------------------------------------*
 |  calculate conductivity of electrolyte solution             gjb 07/09|
 *----------------------------------------------------------------------*/
Epetra_SerialDenseVector SCATRA::ScaTraTimIntImpl::ComputeConductivity()
{
  // we perform the calculation on element level hiding the material access!
  // the initial concentration distribution has to be uniform to do so!!

  // create the parameters for the elements
  ParameterList p;
  p.set("action","calc_elch_conductivity");
  p.set<int>("scatratype",scatratype_);
  p.set("frt",frt_);

  //provide displacement field in case of ALE
  p.set("isale",isale_);
  if (isale_)
    AddMultiVectorToParameterList(p,"dispnp",dispnp_);

  // set vector values needed by elements
  discret_->ClearState();
  discret_->SetState("phinp",phinp_);

  // pointer to current element
  DRT::Element* actele = discret_->lRowElement(0);

  // get element location vector, dirichlet flags and ownerships
  std::vector<int> lm;  // location vector
  std::vector<int> lmowner;  // processor which owns DOFs
  std::vector<int> lmstride;  // nodal block sizes in element matrices

  actele->LocationVector(*discret_,lm,lmowner,lmstride);

  // define element matrices and vectors
  // -- which are empty and unused, just to satisfy element Evaluate()
  Epetra_SerialDenseMatrix elematrix1;
  Epetra_SerialDenseMatrix elematrix2;
  Epetra_SerialDenseVector elevector2;
  Epetra_SerialDenseVector elevector3;

  // define element vector
  Epetra_SerialDenseVector sigma(numscal_+1);

  // call the element evaluate method of the first row element
  int err = actele->Evaluate(p,*discret_,lm,elematrix1,elematrix2,sigma,elevector2,elevector3);
  if (err) dserror("error while computing conductivity");
  discret_->ClearState();

  return sigma;
}


/*----------------------------------------------------------------------*
 | apply galvanostatic control                                gjb 11/09 |
 *----------------------------------------------------------------------*/
bool SCATRA::ScaTraTimIntImpl::ApplyGalvanostaticControl()
{
  // for galvanostatic ELCH applications we have to adjust the
  // applied cell voltage and continue Newton-Raphson iterations until
  // we reach the desired value for the electric current.

  // leave method, if there's nothing to do!
  if (extraparams_->isSublist("ELCH CONTROL") == false) return true;

  if (DRT::INPUT::IntegralValue<int>(extraparams_->sublist("ELCH CONTROL"),"GALVANOSTATIC"))
  {
    vector<DRT::Condition*> cond;
    discret_->GetCondition("ElectrodeKinetics",cond);
    if (!cond.empty())
    {
      const unsigned condid_cathode = extraparams_->sublist("ELCH CONTROL").get<int>("GSTATCONDID_CATHODE");
      const unsigned condid_anode = extraparams_->sublist("ELCH CONTROL").get<int>("GSTATCONDID_ANODE");
      int gstatitemax = (extraparams_->sublist("ELCH CONTROL").get<int>("GSTATITEMAX"));
      double gstatcurrenttol = (extraparams_->sublist("ELCH CONTROL").get<double>("GSTATCURTOL"));
      const int curvenum = extraparams_->sublist("ELCH CONTROL").get<int>("GSTATCURVENO");
      const double tol = extraparams_->sublist("ELCH CONTROL").get<double>("GSTATCONVTOL");
      const double effective_length = extraparams_->sublist("ELCH CONTROL").get<double>("GSTAT_LENGTH_CURRENTPATH");

      const double potold = cond[condid_cathode]->GetDouble("pot");
      double potnew = potold;
      double actualcurrent(0.0);
      double currtangent(0.0);
      double currresidual(0.0);
      double electrodesurface(0.0);
      //Assumption: Residual at BV1 is the negative of the value at BV2, therefore only the first residual is calculated
      double newtonrhs(0.0);

      // for all time integration schemes, compute the current value for phidtnp
      // this is needed for evaluating charging currents due to double-layer capacity
      // This may only be called here and not inside OutputSingleElectrodeInfo!!!!
      // Otherwise you modify your output to file called during Output()
      ComputeTimeDerivative();

      double targetcurrent = DRT::Problem::Instance()->Curve(curvenum-1).f(time_);
      double timefac = 1.0/ResidualScaling();

      double currtangent_anode(0.0);
      double currtangent_cathode(0.0);
      double potinc_ohm(0.0);

      // loop over all BV
      // degenerated to a loop over 2 (user-specified) BV conditions
      for (unsigned int icond = 0; icond < cond.size(); icond++)
      {
        // consider only the specified electrode kinetics boundaries!
        if ((icond != condid_cathode)and((icond != condid_anode)))
          continue;

        actualcurrent = 0.0;
        currtangent = 0.0;
        currresidual = 0.0;

        // note: only the potential at the boundary with id condid_cathode will be adjusted!
        OutputSingleElectrodeInfo(cond[icond],icond,false,false,actualcurrent,currtangent,currresidual,electrodesurface);

        // store the tangent for later usage
        if (icond==condid_cathode)
          currtangent_cathode=currtangent;
        if (icond==condid_anode)
          currtangent_anode=currtangent;

        if (icond==condid_cathode)
        {
          //Assumption: Residual at BV1 is the negative of the value at BV2, therefore only the first residual is calculated
          // newtonrhs = -residual, with the definition:  residual := timefac*(-I + I_target)
          newtonrhs = + currresidual - (timefac*targetcurrent); // newtonrhs is stored only from cathode!
          if (myrank_==0)
          {
            cout<<"\nGALVANOSTATIC MODE:\n";
            cout<<"iteration "<<gstatnumite_<<" / "<<gstatitemax<<endl;
            cout<<"  actual reaction current = "<<scientific<<actualcurrent<<endl;
            cout<<"  required total current  = "<<targetcurrent<<endl;
            cout<<"  negative residual (rhs) = "<<newtonrhs<<endl<<endl;
          }

          if (gstatnumite_ > gstatitemax)
          {
            if (myrank_==0) cout<< endl <<"  --> maximum number iterations reached. Not yet converged!"<<endl<<endl;
            return true; // we proceed to next time step
          }
          else if (abs(newtonrhs)< gstatcurrenttol)
          {
            if (myrank_==0) cout<< endl <<"  --> Newton-RHS-Residual is smaller than " << gstatcurrenttol<< "!" <<endl<<endl;
            return true; // we proceed to next time step
          }
          // electric potential increment of the last iteration
          else if ((gstatnumite_ > 1) and (abs(gstatincrement_)< (1+abs(potold))*tol)) // < ATOL + |pot|* RTOL
          {
            if (myrank_==0) cout<< endl <<"  --> converged: |"<<gstatincrement_<<"| < "<<(1+abs(potold))*tol<<endl<<endl;
            return true; // galvanostatic control has converged
          }

          // update applied electric potential
          // potential drop ButlerVolmer conditions (surface ovepotential) and in the electrolyte (ohmic overpotential) are conected in parallel:
          //
          // I_0 = I_BV1 = I_ohmic = I_BV2
          // R(I_target, I) = R_BV1(I_target, I) = R_ohmic(I_target, I) = -R_BV2(I_target, I)
          // delta E_0 = delta U_BV1 + delta U_ohmic - (delta U_BV2)
          // => delta E_0 = (R_BV1(I_target, I)/J) + (R_ohmic(I_target, I)/J) - (-R_BV2(I_target, I)/J)

          potinc_ohm=(-1.0*effective_length*newtonrhs)/(sigma_(numscal_)*timefac*electrodesurface);

          // print additional information
          if (myrank_==0)
          {
            cout<< "  area                          = " << electrodesurface << endl;
            cout<< "  actualcurrent - targetcurrent = " << (actualcurrent-targetcurrent) << endl;
            cout<< "  conductivity                  = " << sigma_(numscal_) << endl;
          }
        }

        // safety check
        if (abs(currtangent)<EPS13)
          dserror("Tangent in galvanostatic control is near zero: %lf",currtangent);

      }
      // end loop over electrode kinetics

      // Newton step:  Jacobian * \Delta pot = - Residual
      const double potinc_cathode = newtonrhs/currtangent_cathode;
      double potinc_anode = 0.0;
      if (abs(currtangent_anode)>EPS13) // anode surface overpotential is optional
        potinc_anode = newtonrhs/currtangent_anode;
      gstatincrement_ = (potinc_cathode+potinc_anode+potinc_ohm);
      // update electric potential
      potnew += gstatincrement_;

      // print info to screen
      if (myrank_==0)
      {
        cout<<endl<< "  ohmic overpotential                        = " << potinc_ohm << endl;
        cout<< "  overpotential increment cathode (condid " << condid_cathode <<") = " << potinc_cathode << endl;
        if (abs(potinc_anode)>EPS12) // prevents output if an anode is not considered
          cout<< "  overpotential increment anode   (condid " << condid_anode <<") = " << potinc_anode << endl;

        cout<< "  total increment for potential              = " << potinc_cathode+potinc_anode+potinc_ohm << endl;
        cout<< endl;
        cout<< "  old electrode potential (condid "<<condid_cathode <<") = "<<potold<<endl;
        cout<< "  new electrode potential (condid "<<condid_cathode <<") = "<<potnew<<endl<<endl;
      }
      // replace potential value of the boundary condition (on all processors)
      cond[condid_cathode]->Add("pot",potnew);
      gstatnumite_++;
      return false; // not yet converged -> continue Newton iteration with updated potential
    }
  }
  return true; //default

} // end ApplyGalvanostaticControl()


/*----------------------------------------------------------------------*
 | check for zero/negative concentration values               gjb 01/10 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::CheckConcentrationValues(RCP<Epetra_Vector> vec)
{
  // action only for ELCH applications
  if (IsElch(scatratype_))
  {
    // for NURBS discretizations we skip the following check.
    // Control points (i.e., the "nodes" and its associated dofs can be located
    // outside the domain of interest. Thus, they can have negative
    // concentration values although the concentration solution is positive
    // in the whole computational domain!
    if(dynamic_cast<DRT::NURBS::NurbsDiscretization*>(discret_.get())!=NULL)
      return;
  
    // this option can be helpful in some rare situations
    bool makepositive(false);

    vector<int> numfound(numscal_,0);
#if 0
    stringstream myerrormessage;
#endif
    for (int i = 0; i < discret_->NumMyRowNodes(); i++)
    {
      DRT::Node* lnode = discret_->lRowNode(i);
      vector<int> dofs;
      dofs = discret_->Dof(lnode);

      for (int k = 0; k < numscal_; k++)
      {
        const int lid = discret_->DofRowMap()->LID(dofs[k]);
        if (((*vec)[lid]) < EPS13 )
        {
          numfound[k]++;
          if (makepositive)
            ((*vec)[lid]) = EPS13;
#if 0
          myerrormessage<<"PROC "<<myrank_<<" dof index: "<<k<<setprecision(7)<<scientific<<
              " val: "<<((*vec)[lid])<<" node gid: "<<lnode->Id()<<
              " coord: [x] "<< lnode->X()[0]<<" [y] "<< lnode->X()[1]<<" [z] "<< lnode->X()[2]<<endl;
#endif
        }
      }
    }

    // print warning to screen
    for (int k = 0; k < numscal_; k++)
    {
      if (numfound[k] > 0)
      {
        cout<<"WARNING: PROC "<<myrank_<<" has "<<numfound[k]<<
        " nodes with zero/neg. concentration values for species "<<k;
        if (makepositive)
          cout<<"-> were made positive (set to 1.0e-13)"<<endl;
        else
          cout<<endl;
      }
    }

#if 0
    // print detailed info to error file
    for(int p=0; p < discret_->Comm().NumProc(); p++)
    {
      if (p==myrank_) // is it my turn?
      {
        // finish error message
        myerrormessage.flush();

        // write info to error file
        if ((errfile_!=NULL) and (myerrormessage.str()!=""))
        {
          fprintf(errfile_,myerrormessage.str().c_str());
          // cout<<myerrormessage.str()<<endl;
        }
      }
      // give time to finish writing to file before going to next proc ?
      discret_->Comm().Barrier();
    }
#endif

  }
  // so much code for a simple check!
  return;
}


/*----------------------------------------------------------------------*
 | define the magnetic field                                  gjb 05/10 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::SetMagneticField(const int funcno)
{
  if (funcno > 0)
  {
    int err(0);
    const int numdim = 3; // the magnetic field is always 3D

    // loop all nodes on the processor
    for(int lnodeid=0;lnodeid<discret_->NumMyRowNodes();lnodeid++)
    {
      // get the processor local node
      DRT::Node*  lnode = discret_->lRowNode(lnodeid);
      for(int index=0;index<numdim;++index)
      {
        double value = DRT::Problem::Instance()->Funct(funcno-1).Evaluate(index,lnode->X(),0.0,NULL);
        // no time-dependency included, yet!
        err = magneticfield_->ReplaceMyValue(lnodeid, index, value);
        if (err!=0) dserror("error while inserting a value into magneticfield_");
      }
    }
  }
  return;

} // ScaTraImplicitTimeInt::SetMagneticField


/*----------------------------------------------------------------------*
 | add approximation to flux vectors to a parameter list      gjb 05/10 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::AddFluxApproxToParameterList(
    Teuchos::ParameterList& p,
    const enum INPAR::SCATRA::FluxType fluxtype
)
{
Teuchos::RCP<Epetra_MultiVector> flux
 = CalcFluxInDomain(fluxtype);

// post_drt_ensight does not support multivectors based on the dofmap
// for now, I create single vectors that can be handled by the filter

// get the noderowmap
const Epetra_Map* noderowmap = discret_->NodeRowMap();
Teuchos::RCP<Epetra_MultiVector> fluxk = rcp(new Epetra_MultiVector(*noderowmap,3,true));
for(int k=0;k<numscal_;++k)
{
  ostringstream temp;
  temp << k;
  string name = "flux_phi_"+temp.str();
  for (int i = 0;i<fluxk->MyLength();++i)
  {
    DRT::Node* actnode = discret_->lRowNode(i);
    int dofgid = discret_->Dof(actnode,k);
    fluxk->ReplaceMyValue(i,0,((*flux)[0])[(flux->Map()).LID(dofgid)]);
    fluxk->ReplaceMyValue(i,1,((*flux)[1])[(flux->Map()).LID(dofgid)]);
    fluxk->ReplaceMyValue(i,2,((*flux)[2])[(flux->Map()).LID(dofgid)]);
  }
  AddMultiVectorToParameterList(p,name,fluxk);
}
return;
}

/*----------------------------------------------------------------------*
 | return dof row map                                          vg 09/11 |
 *----------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Map> SCATRA::ScaTraTimIntImpl::DofRowMap()
{
  const Epetra_Map* dofrowmap = discret_->DofRowMap();
  return Teuchos::rcp(dofrowmap, false);
}

/*----------------------------------------------------------------------*
 | return dof row map for multiple discretizations             vg 09/11 |
 *----------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Map> SCATRA::ScaTraTimIntImpl::DofRowMap(unsigned nds)
{
  const Epetra_Map* dofrowmap = discret_->DofRowMap(nds);
  return Teuchos::rcp(dofrowmap, false);
}


#endif /* CCADISCRET       */

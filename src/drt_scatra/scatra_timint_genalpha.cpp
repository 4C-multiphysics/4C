/*----------------------------------------------------------------------*/
/*!
\file scatra_timint_genalpha.cpp
\brief Generalized-alpha time-integration scheme

<pre>
Maintainer: Volker Gravemeier
            vgravem@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15245
</pre>
*/
/*----------------------------------------------------------------------*/

#include "scatra_timint_genalpha.H"
#include "../drt_scatra_ele/scatra_ele_action.H"
#include "turbulence_hit_scalar_forcing.H"
#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <Teuchos_TimeMonitor.hpp>
#include "../drt_io/io.H"
#include "../linalg/linalg_solver.H"
#include "../drt_fluid_turbulence/dyn_smag.H"
#include "../drt_fluid_turbulence/dyn_vreman.H"
#include "../drt_lib/drt_globalproblem.H"
#include "../drt_inpar/drt_validparameters.H"


/*----------------------------------------------------------------------*
 |  Constructor (public)                                       vg 11/08 |
 *----------------------------------------------------------------------*/
SCATRA::TimIntGenAlpha::TimIntGenAlpha(
    Teuchos::RCP<DRT::Discretization>      actdis,
  Teuchos::RCP<LINALG::Solver>           solver,
  Teuchos::RCP<Teuchos::ParameterList>   params,
  Teuchos::RCP<Teuchos::ParameterList>   extraparams,
  Teuchos::RCP<IO::DiscretizationWriter> output)
: ScaTraTimIntImpl(actdis,solver,params,extraparams,output),
  phiaf_(Teuchos::null),
  phiam_(Teuchos::null),
  phidtam_(Teuchos::null),
  fsphiaf_(Teuchos::null),
  alphaM_(params_->get<double>("ALPHA_M")),
  alphaF_(params_->get<double>("ALPHA_F")),
  gamma_ (params_->get<double>("GAMMA")),
  genalphafac_(0.0)
{
  // DO NOT DEFINE ANY STATE VECTORS HERE (i.e., vectors based on row or column maps)
  // this is important since we have problems which require an extended ghosting
  // this has to be done before all state vectors are initialized
  return;
}


/*----------------------------------------------------------------------*
 |  initialize time integration                         rasthofer 09/13 |
 *----------------------------------------------------------------------*/
void SCATRA::TimIntGenAlpha::Init()
{
  // initialize base class
  ScaTraTimIntImpl::Init();

  // -------------------------------------------------------------------
  // get a vector layout from the discretization to construct matching
  // vectors and matrices
  //                 local <-> global dof numbering
  // -------------------------------------------------------------------
  const Epetra_Map* dofrowmap = discret_->DofRowMap();

  // Vectors passed to the element
  // -----------------------------

  // scalar at times n+alpha_F and n+alpha_M
  phiaf_ = LINALG::CreateVector(*dofrowmap,true);
  phiam_ = LINALG::CreateVector(*dofrowmap,true);

  // temporal derivative of scalar at times n+1, n and n+alpha_M
  phidtam_ = LINALG::CreateVector(*dofrowmap,true);

  // compute specific time factor for generalized-alpha time integration:
  // genalphatimefac = gamma*alpha_F/alpha_M
  if (alphaM_ < EPS12) dserror("factor alpha_M lower than or equal zero");
  genalphafac_ = gamma_/alphaM_;

  // fine-scale vector at time n+alpha_F
  if (fssgd_ != INPAR::SCATRA::fssugrdiff_no or turbmodel_ == INPAR::FLUID::multifractal_subgrid_scales)
    fsphiaf_ = LINALG::CreateVector(*dofrowmap,true);

  // -------------------------------------------------------------------
  // set element parameters
  // -------------------------------------------------------------------
  // note: - this has to be done before element routines are called
  //       - order is important here: for savety checks in SetElementGeneralScaTraParameter(),
  //         we have to konw the time-integration parameters
  SetElementTimeParameter();
  SetElementGeneralScaTraParameter();
  SetElementTurbulenceParameter();

  // for initializing phiaf_, phiam based on the initial field that was
  // set for phinp_, phin_ in the TimInt base class constructor
  // this method has to be called before method OutputElectrodeInfo() is called
  // otherwise phi_af is initialized with zeros instead of the initial field
  ComputeIntermediateValues();

  // Important: this adds the required ConditionID's to the single conditions.
  // It is necessary to do this BEFORE ReadRestart() is called!
  // Output to screen and file is suppressed
  //OutputElectrodeInfo(false,false);

  // setup krylov
  PrepareKrylovProjection();

  // -------------------------------------------------------------------
  // initialize forcing for homogeneous isotropic turbulence
  // -------------------------------------------------------------------
  // note: this constructor has to be called after the forcing_ vector has
  //       been initialized; this is done in ScaTraTimIntImpl::Init() called before
  if (special_flow_ == "scatra_forced_homogeneous_isotropic_turbulence")
  {
    if (extraparams_->sublist("TURBULENCE MODEL").get<std::string>("SCALAR_FORCING")=="isotropic")
    {
      homisoturb_forcing_ = Teuchos::rcp(new SCATRA::HomIsoTurbScalarForcing(this));
      // initialize forcing algorithm
      homisoturb_forcing_->SetInitialSpectrum(DRT::INPUT::IntegralValue<INPAR::SCATRA::InitialField>(*params_,"INITIALFIELD"));
    }
  }

  return;
}


/*----------------------------------------------------------------------*
| Destructor dtor (public)                                     vg 11/08 |
*-----------------------------------------------------------------------*/
SCATRA::TimIntGenAlpha::~TimIntGenAlpha()
{
  return;
}


/*----------------------------------------------------------------------*
 |  set time parameter for element evaluation (usual call)   ehrl 11/13 |
 *----------------------------------------------------------------------*/
void SCATRA::TimIntGenAlpha::SetElementTimeParameter(bool forcedincrementalsolver)
{
  Teuchos::ParameterList eleparams;

  eleparams.set<int>("action",SCATRA::set_time_parameter);
  // set type of scalar transport problem (after preevaluate evaluate, which need scatratype is called)
  eleparams.set<int>("scatratype",scatratype_);

  eleparams.set<bool>("using generalized-alpha time integration",true);
  eleparams.set<bool>("using stationary formulation",false);
  if(forcedincrementalsolver==false)
    eleparams.set<bool>("incremental solver",incremental_);
  else
    eleparams.set<bool>("incremental solver",true);

  eleparams.set<double>("time-step length",dta_);
  eleparams.set<double>("total time",time_-(1-alphaF_)*dta_);
  eleparams.set<double>("time factor",genalphafac_*dta_);
  eleparams.set<double>("alpha_F",alphaF_);

  // call standard loop over elements
  discret_->Evaluate(eleparams,Teuchos::null,Teuchos::null,Teuchos::null,Teuchos::null,Teuchos::null);

  return;
}


/*--------------------------------------------------------------------------*
 | set time for evaluation of POINT -Neumann boundary conditions   vg 12/08 |
 *--------------------------------------------------------------------------*/
void SCATRA::TimIntGenAlpha::SetTimeForNeumannEvaluation(
  Teuchos::ParameterList& params)
{
  params.set("total time",time_-(1-alphaF_)*dta_);
  return;
}


/*----------------------------------------------------------------------*
 |  set time parameter for element evaluation (usual call)   ehrl 11/13 |
 *----------------------------------------------------------------------*/
void SCATRA::TimIntGenAlpha::SetElementTimeParameterBackwardEuler()
{
  Teuchos::ParameterList eleparams;

  eleparams.set<int>("action",SCATRA::set_time_parameter);
  // set type of scalar transport problem (after preevaluate evaluate, which need scatratype is called)
  eleparams.set<int>("scatratype",scatratype_);

  eleparams.set<bool>("using generalized-alpha time integration",false);
  eleparams.set<bool>("using stationary formulation",false);
  eleparams.set<bool>("incremental solver",true);

  eleparams.set<double>("time-step length",dta_);
  eleparams.set<double>("total time",time_);
  eleparams.set<double>("time factor",1.0*dta_);
  eleparams.set<double>("alpha_F",1.0);

  // call standard loop over elements
  discret_->Evaluate(eleparams,Teuchos::null,Teuchos::null,Teuchos::null,Teuchos::null,Teuchos::null);

  return;
}


/*----------------------------------------------------------------------*
 | set part of the residual vector belonging to the old timestep        |
 |                                                             vg 11/08 |
 *----------------------------------------------------------------------*/
void SCATRA::TimIntGenAlpha::SetOldPartOfRighthandside()
{
  // calculation of history vector only for non-incremental formulation:
  // (History vector is used in both cases, but in incremental case, it
  // contains time derivatives of scalar, see below.)
  // hist_ = phin_ + dt*(1-(gamma/alpha_M))*phidtn_
  if (not incremental_)
    hist_->Update(1.0, *phin_, dta_*(1.0-genalphafac_), *phidtn_, 0.0);

  return;
}


/*----------------------------------------------------------------------*
 | perform an explicit predictor step                          vg 11/08 |
 *----------------------------------------------------------------------*/
void SCATRA::TimIntGenAlpha::ExplicitPredictor()
{
  // constant predictor
  phinp_->Update(1.0,*phin_,0.0);
  return;
}


/*----------------------------------------------------------------------*
 | compute values at intermediate time steps                   vg 09/09 |
 *----------------------------------------------------------------------*/
void SCATRA::TimIntGenAlpha::ComputeIntermediateValues()
{
  // compute phi at n+alpha_F and n+alpha_M
  phiaf_->Update(alphaF_,*phinp_,(1.0-alphaF_),*phin_,0.0);
  phiam_->Update(alphaM_,*phinp_,(1.0-alphaM_),*phin_,0.0);

  // accelerations are not independent but rather have to be computed
  // from phinp_, phin_ and phidtn_
  ComputeTimeDerivative();

  // compute time derivative of phi at n+alpha_M
  phidtam_->Update(alphaM_,*phidtnp_,(1.0-alphaM_),*phidtn_,0.0);

  return;
}


/*----------------------------------------------------------------------*
 | add actual Neumann loads                                             |
 | scaled with a factor resulting from time discretization     vg 11/08 |
 *----------------------------------------------------------------------*/
void SCATRA::TimIntGenAlpha::AddNeumannToResidual()
{
  residual_->Update(genalphafac_*dta_,*neumann_loads_,1.0);
  return;
}


/*----------------------------------------------------------------------*
 | AVM3-based scale separation                                 vg 03/09 |
 *----------------------------------------------------------------------*/
void SCATRA::TimIntGenAlpha::AVM3Separation()
{
  // time measurement: avm3
  TEUCHOS_FUNC_TIME_MONITOR("SCATRA:            + avm3");

  // AVM3 separation
  Sep_->Multiply(false,*phiaf_,*fsphiaf_);

  // set fine-scale velocity for parallel nigthly tests
  // separation matrix depends on the number of proc here
  if (turbmodel_==INPAR::FLUID::multifractal_subgrid_scales and
      (DRT::INPUT::IntegralValue<int>(extraparams_->sublist("MULTIFRACTAL SUBGRID SCALES"),"SET_FINE_SCALE_VEL")))
      fsphiaf_->PutScalar(0.01);

  // set fine-scale vector
  discret_->SetState("fsphinp",fsphiaf_);

  return;
}


/*----------------------------------------------------------------------*
 | dynamic Smagorinsky model                           rasthofer  08/12 |
 *----------------------------------------------------------------------*/
void SCATRA::TimIntGenAlpha::DynamicComputationOfCs()
{
  if (turbmodel_==INPAR::FLUID::dynamic_smagorinsky)
  {
    // perform filtering and computation of Prt
    // compute averaged values for LkMk and MkMk
    const Teuchos::RCP<const Epetra_Vector> dirichtoggle = DirichletToggle();
    DynSmag_->ApplyFilterForDynamicComputationOfPrt(convel_,phiaf_,0.0,dirichtoggle,*extraparams_);
  }

  return;
}

/*----------------------------------------------------------------------*
 | dynamic Vreman model                                krank  09/13     |
 *----------------------------------------------------------------------*/
void SCATRA::TimIntGenAlpha::DynamicComputationOfCv()
{
  if (turbmodel_==INPAR::FLUID::dynamic_vreman)
  {
    const Teuchos::RCP<const Epetra_Vector> dirichtoggle = DirichletToggle();
    Vrem_->ApplyFilterForDynamicComputationOfDt(convel_,phiaf_,0.0,dirichtoggle,*extraparams_);
  }

  return;
}


/*----------------------------------------------------------------------*
 | add parameters specific for time-integration scheme         vg 11/08 |
 *----------------------------------------------------------------------*/
void SCATRA::TimIntGenAlpha::AddTimeIntegrationSpecificVectors(bool forcedincrementalsolver)
{
  discret_->SetState("phinp",phiaf_);

  if (incremental_ or forcedincrementalsolver)
    discret_->SetState("hist",phidtam_);
  else
  {
    discret_->SetState("hist",hist_);
    discret_->SetState("phin",phin_);
  }

  return;
}


/*----------------------------------------------------------------------*
 | compute time derivative                                     vg 09/09 |
 *----------------------------------------------------------------------*/
void SCATRA::TimIntGenAlpha::ComputeTimeDerivative()
{
  // time derivative of phi:
  // phidt(n+1) = (phi(n+1)-phi(n)) / (gamma*dt) + (1-(1/gamma))*phidt(n)
  const double fact1 = 1.0/(gamma_*dta_);
  const double fact2 = 1.0 - (1.0/gamma_);
  phidtnp_->Update(fact2,*phidtn_,0.0);
  phidtnp_->Update(fact1,*phinp_,-fact1,*phin_,1.0);

  // We know the first time derivative on Dirichlet boundaries
  // so we do not need an approximation of these values!
  // However, we do not want to break the linear relationship
  // as stated above. We do not want to set Dirichlet values for
  // dependent values like phidtnp_. This turned out to be inconsistent.
  // Such an inconsistency can cause different results for
  // our different Gen. Alpha formulations (linear_full <-> linear_incremental).
  // We don't want this to happen.
  // ApplyDirichletBC(time_,Teuchos::null,phidtnp_);

  return;
}


/*----------------------------------------------------------------------*
 | current solution becomes most recent solution of next timestep       |
 |                                                             vg 11/08 |
 *----------------------------------------------------------------------*/
void SCATRA::TimIntGenAlpha::Update(const int num)
{
  // set history variable to zero for not spoiling flux calculation
  //if (not incremental_) hist_->PutScalar(0.0);

  // compute flux vector field for later output BEFORE time shift of results
  // is performed below !!
  if (writeflux_!=INPAR::SCATRA::flux_no)
  {
    if (DoOutput() or DoBoundaryFluxStatistics())
      flux_ = CalcFlux(true, num);
    //else
      // necessary to print statistical values after each time step but the solution only
      //flux_ = CalcFlux(true);
  }

  // compute time derivative at time n+1
  ComputeTimeDerivative();

  // solution of this step becomes most recent solution of last step
  phin_->Update(1.0,*phinp_,0.0);

  // time deriv. of this step becomes most recent time derivative of
  // last step
  phidtn_->Update(1.0,*phidtnp_,0.0);

  // call time update of forcing routine
  if (homisoturb_forcing_ != Teuchos::null)
    homisoturb_forcing_->TimeUpdateForcing();

  return;
}


/*----------------------------------------------------------------------*
 | write additional data required for restart                  vg 11/08 |
 *----------------------------------------------------------------------*/
void SCATRA::TimIntGenAlpha::OutputRestart()
{
  // additional state vectors that are needed for generalized-alpha restart
  output_->WriteVector("phidtnp",phidtnp_);
  output_->WriteVector("phidtn", phidtn_);
  output_->WriteVector("phin",   phin_);

  // for elch problems with moving boundary
  if (isale_)
    output_->WriteVector("trueresidual", trueresidual_);

  return;
}


/*----------------------------------------------------------------------*
 |                                                             vg 11/08 |
 -----------------------------------------------------------------------*/
void SCATRA::TimIntGenAlpha::ReadRestart(int step)
{
  IO::DiscretizationReader reader(discret_,step);
  time_ = reader.ReadDouble("time");
  step_ = reader.ReadInt("step");

  if (myrank_==0)
    std::cout<<"Reading ScaTra restart data (time="<<time_<<" ; step="<<step_<<")"<<std::endl;

  // read state vectors that are needed for generalized-alpha restart
  reader.ReadVector(phinp_,  "phinp");
  reader.ReadVector(phin_,   "phin");
  reader.ReadVector(phidtnp_,"phidtnp");
  reader.ReadVector(phidtn_, "phidtn");

  // for elch problems with moving boundary
  if(isale_)
    reader.ReadVector(trueresidual_, "trueresidual");

  if (fssgd_ != INPAR::SCATRA::fssugrdiff_no or
      turbmodel_ == INPAR::FLUID::multifractal_subgrid_scales)
    AVM3Preparation();

  return;
}


/*----------------------------------------------------------------------*
 | Initialization procedure before the first time step         vg 11/08 |
 -----------------------------------------------------------------------*/
void SCATRA::TimIntGenAlpha::PrepareFirstTimeStep()
{
  // evaluate Dirichlet boundary conditions at time t=0
  // the values should match your initial field at the boundary!
  // ApplyDirichletBC(time_,phin_,phidtn_);
  ApplyDirichletBC(time_,phin_,Teuchos::null);

  // compute initial field for electric potential (ELCH)
  CalcInitialPotentialField();

  // for calculation of initial time derivative, we have to switch off all stabilization and
  // turbulence modeling terms
  // therefore, we have another PerEvaluate call here
  Teuchos::ParameterList eleparams;

  // standard general element parameter without stabilization
  SetElementGeneralScaTraParameterDeactivatedStab();

  // we also have to modify the time-parameter list (incremental solve)
  // actually we do not need a time integration scheme for calculating the initial time derivatives,
  // but the rhs of the standard element routine is used as starting point for this special system of equations.
  // Therefore, the rhs vector has to be scaled correctly.
  // Since the genalpha scheme cannot be adapted easily, the backward Euler scheme is used instead.
  SetElementTimeParameterBackwardEuler();

  // deactivate turbulence settings
  SetElementTurbulenceParameterDeactivated();

  // compute time derivative of phi at time t=0
  CalcInitialPhidt();

  // and finally undo our temporary settings
  SetElementGeneralScaTraParameter();
  SetElementTimeParameter();
  SetElementTurbulenceParameter();

  return;
}




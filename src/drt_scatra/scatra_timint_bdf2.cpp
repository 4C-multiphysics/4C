/*----------------------------------------------------------------------*/
/*!
\file scatra_timint_bdf2.cpp
\brief BDF2 time-integration scheme

<pre>
Maintainer: Andreas Ehrl
            ehrl@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15252
</pre>
*/
/*----------------------------------------------------------------------*/

#include "scatra_timint_bdf2.H"
#include "../drt_scatra_ele/scatra_ele_action.H"
#include "turbulence_hit_scalar_forcing.H"
#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <Teuchos_TimeMonitor.hpp>
#include "../drt_inpar/inpar_elch.H"
#include "../drt_io/io.H"
#include "../drt_fluid_turbulence/dyn_smag.H"
#include "../drt_fluid_turbulence/dyn_vreman.H"


/*----------------------------------------------------------------------*
 |  Constructor (public)                                      gjb 08/08 |
 *----------------------------------------------------------------------*/
SCATRA::TimIntBDF2::TimIntBDF2(
    Teuchos::RCP<DRT::Discretization>      actdis,
    Teuchos::RCP<LINALG::Solver>           solver,
    Teuchos::RCP<Teuchos::ParameterList>   params,
    Teuchos::RCP<Teuchos::ParameterList>   extraparams,
    Teuchos::RCP<IO::DiscretizationWriter> output)
: ScaTraTimIntImpl(actdis,solver,params,extraparams,output),
  theta_(1.0),
  phinm_(Teuchos::null),
  fsphinp_(Teuchos::null)
{
  // DO NOT DEFINE ANY STATE VECTORS HERE (i.e., vectors based on row or column maps)
  // this is important since we have problems which require an extended ghosting
  // this has to be done before all state vectors are initialized
  return;
}


/*----------------------------------------------------------------------*
 |  initialize time integration                         rasthofer 09/13 |
 *----------------------------------------------------------------------*/
void SCATRA::TimIntBDF2::Init()
{
  // initialize base class
  ScaTraTimIntImpl::Init();

  // -------------------------------------------------------------------
  // get a vector layout from the discretization to construct matching
  // vectors and matrices
  //                 local <-> global dof numbering
  // -------------------------------------------------------------------
  const Epetra_Map* dofrowmap = discret_->DofRowMap();

  // state vector for solution at time t_{n-1}
  phinm_ = LINALG::CreateVector(*dofrowmap,true);

  // fine-scale vector at time n+1
  if (fssgd_ != INPAR::SCATRA::fssugrdiff_no or turbmodel_ == INPAR::FLUID::multifractal_subgrid_scales)
    fsphinp_ = LINALG::CreateVector(*dofrowmap,true);

  // -------------------------------------------------------------------
  // set element parameters
  // -------------------------------------------------------------------
  // note: - this has to be done before element routines are called
  //       - order is important here: for savety checks in SetElementGeneralScaTraParameter(),
  //         we have to konw the time-integration parameters
  SetElementTimeParameter();
  SetElementGeneralScaTraParameter();
  SetElementTurbulenceParameter();

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
| Destructor dtor (public)                                   gjb 08/08 |
*----------------------------------------------------------------------*/
SCATRA::TimIntBDF2::~TimIntBDF2()
{
  return;
}


/*----------------------------------------------------------------------*
 |  set time parameter for element evaluation (usual call)   ehrl 11/13 |
 *----------------------------------------------------------------------*/
void SCATRA::TimIntBDF2::SetElementTimeParameter()
{
  Teuchos::ParameterList eleparams;

  eleparams.set<int>("action",SCATRA::set_time_parameter);
  // set type of scalar transport problem (after preevaluate evaluate, which need scatratype is called)
  eleparams.set<int>("scatratype",scatratype_);

  eleparams.set<bool>("using generalized-alpha time integration",false);
  eleparams.set<bool>("using stationary formulation",false);
  eleparams.set<bool>("incremental solver",incremental_);

  eleparams.set<double>("time-step length",dta_);
  eleparams.set<double>("total time",time_);
  eleparams.set<double>("time factor",theta_*dta_);
  eleparams.set<double>("alpha_F",1.0);

  // call standard loop over elements
  discret_->Evaluate(eleparams,Teuchos::null,Teuchos::null,Teuchos::null,Teuchos::null,Teuchos::null);

  return;
}


/*----------------------------------------------------------------------*
 |  set time parameter for element evaluation                ehrl 11/13 |
 *----------------------------------------------------------------------*/
void SCATRA::TimIntBDF2::SetElementTimeParameterForForcedIncrementalSolve()
{
  Teuchos::ParameterList eleparams;

  eleparams.set<int>("action",SCATRA::set_time_parameter);
  // set type of scalar transport problem (after preevaluate evaluate, which need scatratype is called)
  eleparams.set<int>("scatratype",scatratype_);

  eleparams.set<bool>("using generalized-alpha time integration",false);
  eleparams.set<bool>("using stationary formulation",false);
  // this is important to have here and the only difference compared to SetElementTimeParameter()
  eleparams.set<bool>("incremental solver",true);

  eleparams.set<double>("time-step length",dta_);
  eleparams.set<double>("total time",time_);
  eleparams.set<double>("time factor",theta_*dta_);
  eleparams.set<double>("alpha_F",1.0);

  // call standard loop over elements
  discret_->Evaluate(eleparams,Teuchos::null,Teuchos::null,Teuchos::null,Teuchos::null,Teuchos::null);

  return;
}


/*----------------------------------------------------------------------*
 | set part of the residual vector belonging to the old timestep        |
 |                                                            gjb 08/08 |
 *----------------------------------------------------------------------*/
void SCATRA::TimIntBDF2::SetOldPartOfRighthandside()
{
  /*
  BDF2: for variable time step:

                 hist_ = (1+omega)^2/(1+ 2*omega) * phin_
                           - omega^2/(1+ 2*omega) * phinm_

  BDF2: for constant time step:

                 hist_ = 4/3 phin_ - 1/3 phinm_
  */
  if (step_>1)
  {
    double fact1 = 4.0/3.0;
    double fact2 = -1.0/3.0;
    hist_->Update(fact1, *phin_, fact2, *phinm_, 0.0);

    // for BDF2 theta is set to 2/3 for constant time-step length dt
    theta_ = 2.0/3.0;
  }
  else
  {
    // for start-up of BDF2 we do one step with backward Euler
    hist_->Update(1.0, *phin_, 0.0);

    // backward Euler => use theta=1.0
    theta_=1.0;
  }

  // for electrochemical applications
  ElectrodeKineticsSetOldPartOfRHS();

  return;
}


/*----------------------------------------------------------------------*
 | perform an explicit predictor step                         gjb 11/08 |
 *----------------------------------------------------------------------*/
void SCATRA::TimIntBDF2::ExplicitPredictor()
{
  if (step_>1) phinp_->Update(-1.0, *phinm_,2.0);
  // for step == 1 phinp_ is already correctly initialized with the
  // initial field phin_

  return;
}


/*----------------------------------------------------------------------*
 | add actual Neumann loads                                             |
 | scaled with a factor resulting from time discretization     vg 11/08 |
 *----------------------------------------------------------------------*/
void SCATRA::TimIntBDF2::AddNeumannToResidual()
{
  residual_->Update(theta_*dta_,*neumann_loads_,1.0);
  return;
}


/*----------------------------------------------------------------------*
 | AVM3-based scale separation                                 vg 03/09 |
 *----------------------------------------------------------------------*/
void SCATRA::TimIntBDF2::AVM3Separation()
{
  // time measurement: avm3
  TEUCHOS_FUNC_TIME_MONITOR("SCATRA:            + avm3");

  // AVM3 separation
  Sep_->Multiply(false,*phinp_,*fsphinp_);

  // set fine-scale vector
  discret_->SetState("fsphinp",fsphinp_);

  return;
}

/*----------------------------------------------------------------------*
 | dynamic Smagorinsky model                           rasthofer  08/12 |
 *----------------------------------------------------------------------*/
void SCATRA::TimIntBDF2::DynamicComputationOfCs()
{
  if (turbmodel_==INPAR::FLUID::dynamic_smagorinsky)
  {
    // perform filtering and computation of Prt
    // compute averaged values for LkMk and MkMk
    const Teuchos::RCP<const Epetra_Vector> dirichtoggle = DirichletToggle();
    DynSmag_->ApplyFilterForDynamicComputationOfPrt(convel_,phinp_,0.0,dirichtoggle,*extraparams_);
  }

  return;
}


/*----------------------------------------------------------------------*
 | dynamic Vreman model                                krank      09/13 |
 *----------------------------------------------------------------------*/
void SCATRA::TimIntBDF2::DynamicComputationOfCv()
{
  if (turbmodel_==INPAR::FLUID::dynamic_vreman)
  {
    const Teuchos::RCP<const Epetra_Vector> dirichtoggle = DirichletToggle();
    Vrem_->ApplyFilterForDynamicComputationOfDt(convel_,phinp_,0.0,dirichtoggle,*extraparams_);
  }

  return;
}


/*----------------------------------------------------------------------*
 | add parameters specific for time-integration scheme         vg 11/08 |
 *----------------------------------------------------------------------*/
void SCATRA::TimIntBDF2::AddTimeIntegrationSpecificVectors()
{
  discret_->SetState("hist",hist_);
  discret_->SetState("phinp",phinp_);

  return;
}


/*----------------------------------------------------------------------*
 | compute time derivative                                     vg 09/09 |
 *----------------------------------------------------------------------*/
void SCATRA::TimIntBDF2::ComputeTimeDerivative()
{
  if (step_ == 1)
  {
    // time derivative of phi for first time step:
    // phidt(n+1) = (phi(n+1)-phi(n))/dt
    const double fact = 1.0/dta_;
    phidtnp_->Update(fact,*phinp_,-fact,*phin_,0.0);
  }
  else
  {
    // time derivative of phi:
    // phidt(n+1) = ((3/2)*phi(n+1)-2*phi(n)+(1/2)*phi(n-1))/dt
    const double fact1 = 3.0/(2.0*dta_);
    const double fact2 = -2.0/dta_;
    const double fact3 = 1.0/(2.0*dta_);
    phidtnp_->Update(fact3,*phinm_,0.0);
    phidtnp_->Update(fact1,*phinp_,fact2,*phin_,1.0);
  }

  // We know the first time derivative on Dirichlet boundaries
  // so we do not need an approximation of these values!
  // However, we do not want to break the linear relationship
  // as stated above. We do not want to set Dirichlet values for
  // dependent values like phidtnp_. This turned out to be inconsistent.
  // ApplyDirichletBC(time_,Teuchos::null,phidtnp_);

  return;
}


/*----------------------------------------------------------------------*
 | current solution becomes most recent solution of next timestep       |
 |                                                            gjb 08/08 |
 *----------------------------------------------------------------------*/
void SCATRA::TimIntBDF2::Update(const int num)
{
  // compute flux vector field for later output BEFORE time shift of results
  // is performed below !!
  if (writeflux_!=INPAR::SCATRA::flux_no)
  {
    if (DoOutput() or DoBoundaryFluxStatistics())
      flux_ = CalcFlux(true, num);
  }

  // solution of this step becomes most recent solution of the last step
  phinm_->Update(1.0,*phin_ ,0.0);
  phin_ ->Update(1.0,*phinp_,0.0);

  // call time update of forcing routine
  if (homisoturb_forcing_ != Teuchos::null)
    homisoturb_forcing_->TimeUpdateForcing();

  return;
}


/*----------------------------------------------------------------------*
 | write additional data required for restart                 gjb 08/08 |
 *----------------------------------------------------------------------*/
void SCATRA::TimIntBDF2::OutputRestart()
{
  // additional state vectors that are needed for BDF2 restart
  output_->WriteVector("phin", phin_);
  output_->WriteVector("phinm", phinm_);

  // for elch problems with moving boundary
  if (isale_)
    output_->WriteVector("trueresidual", trueresidual_);

  return;
}


/*----------------------------------------------------------------------*
 |                                                            gjb 08/08 |
 -----------------------------------------------------------------------*/
void SCATRA::TimIntBDF2::ReadRestart(int step)
{
  IO::DiscretizationReader reader(discret_,step);
  time_ = reader.ReadDouble("time");
  step_ = reader.ReadInt("step");

  if (myrank_==0)
    std::cout<<"Reading ScaTra restart data (time="<<time_<<" ; step="<<step_<<")"<<std::endl;

  // read state vectors that are needed for BDF2 restart
  reader.ReadVector(phinp_,"phinp");
  reader.ReadVector(phin_, "phin");
  reader.ReadVector(phinm_,"phinm");

  // for elch problems with moving boundary
  if(isale_)
    reader.ReadVector(trueresidual_, "trueresidual");

  if (fssgd_ != INPAR::SCATRA::fssugrdiff_no or
      turbmodel_ == INPAR::FLUID::multifractal_subgrid_scales)
    AVM3Preparation();

  return;
}


/*----------------------------------------------------------------------*
 | initialization procedure before the first time step         gjb 07/08|
 *----------------------------------------------------------------------*/
void SCATRA::TimIntBDF2::PrepareFirstTimeStep()
{
  ApplyDirichletBC(time_, phin_,Teuchos::null);

  // compute initial field for electric potential (ELCH)
  CalcInitialPotentialField();

  return;
}



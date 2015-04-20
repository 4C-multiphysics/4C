/*----------------------------------------------------------------------*/
/*!
\file scatra_timint_implicit.cpp
\brief Control routine for convection-diffusion (in)stationary solvers,

     including instationary solvers based on

     o one-step-theta time-integration scheme

     o two-step BDF2 time-integration scheme

     o generalized-alpha time-integration scheme

     and stationary solver.

<pre>
Maintainer: Andreas Ehrl
            ehrl@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15252
</pre>
*/
/*----------------------------------------------------------------------*/

#include "scatra_timint_meshtying_strategy_fluid.H"
#include "scatra_timint_meshtying_strategy_s2i.H"
#include "scatra_timint_meshtying_strategy_std.H"
#include "turbulence_hit_initial_scalar_field.H"
#include "turbulence_hit_scalar_forcing.H"

#include "../drt_scatra_ele/scatra_ele_action.H"

#include "../linalg/linalg_solver.H"
#include "../linalg/linalg_krylov_projector.H"

#include "../drt_lib/drt_globalproblem.H"

#include "../drt_inpar/drt_validparameters.H"

#include "../drt_mat/matlist.H"
#include "../drt_mat/scatra_mat.H"

#include "../drt_fluid/drt_periodicbc.H"
#include "../drt_fluid/fluid_rotsym_periodicbc_utils.H"
#include "../drt_fluid_turbulence/dyn_vreman.H"

#include "../drt_nurbs_discret/drt_apply_nurbs_initial_condition.H"

#include "../drt_meshfree_discret/drt_meshfree_discret.H"

#include "../drt_io/io.H"

#include "scatra_timint_implicit.H"

// for the condition writer output
/*
#include "../drt_adapter/adapter_coupling.H"
#include "../drt_lib/drt_utils.H"
#include "../drt_lib/drt_condition_utils.H"
#include "../drt_io/io_control.H"
#include "../drt_io/io.H"
*/

/*
// for output of intermediate states in Newton loop
#include "../drt_io/io.H"
#include "../drt_io/io_control.H"
*/

//#define VISUALIZE_ELEDATA_GMSH
//only if VISUALIZE_ELEDATA_GMSH
//#include "../drt_io/io_gmsh.H"


/*==========================================================================*/
// Constructors and destructors and related methods
/*==========================================================================*/

/*----------------------------------------------------------------------*
 |  Constructor                                        (public) vg 05/07|
 *----------------------------------------------------------------------*/
SCATRA::ScaTraTimIntImpl::ScaTraTimIntImpl(
    Teuchos::RCP<DRT::Discretization>      actdis,
    Teuchos::RCP<LINALG::Solver>           solver,
    Teuchos::RCP<Teuchos::ParameterList>   params,
    Teuchos::RCP<Teuchos::ParameterList>   extraparams,
    Teuchos::RCP<IO::DiscretizationWriter> output) :
  // call constructor for "nontrivial" objects
  solver_ (solver),
  params_ (params),
  extraparams_(extraparams),
  myrank_ (actdis->Comm().MyPID()),
  splitter_(Teuchos::null),
  errfile_  (extraparams->get<FILE*>("err file")),
  strategy_(Teuchos::null),
  isale_    (extraparams->get<bool>("isale")),
  solvtype_ (DRT::INPUT::IntegralValue<INPAR::SCATRA::SolverType>(*params,"SOLVERTYPE")),
  incremental_(true),
  initialvelset_(false),
  fssgd_ (DRT::INPUT::IntegralValue<INPAR::SCATRA::FSSUGRDIFF>(*params,"FSSUGRDIFF")),
  turbmodel_(INPAR::FLUID::no_model),
  s2icoupling_(actdis->GetCondition("S2ICouplingSlave") != NULL),
  writeflux_(DRT::INPUT::IntegralValue<INPAR::SCATRA::FluxType>(*params,"WRITEFLUX")),
  writefluxids_(Teuchos::rcp(new std::vector<int>)),
  flux_(Teuchos::null),
  sumnormfluxintegral_(Teuchos::null),
  lastfluxoutputstep_(-1),
  outmean_  (DRT::INPUT::IntegralValue<int>(*params,"OUTMEAN")),
  outputgmsh_(DRT::INPUT::IntegralValue<int>(*params,"OUTPUT_GMSH")),
  output_state_matlab_(DRT::INPUT::IntegralValue<int>(*params,"MATLAB_STATE_OUTPUT")),
  fdcheck_(DRT::INPUT::IntegralValue<INPAR::SCATRA::FDCheck>(*params,"FDCHECK")),
  fdcheckeps_(params->get<double>("FDCHECKEPS")),
  fdchecktol_(params->get<double>("FDCHECKTOL")),
  time_   (0.0),
  maxtime_  (params->get<double>("MAXTIME")),
  step_   (0),
  stepmax_  (params->get<int>("NUMSTEP")),
  dta_      (params->get<double>("TIMESTEP")),
  dtele_(0.0),
  dtsolve_(0.0),
  iternum_(0),
  timealgo_ (DRT::INPUT::IntegralValue<INPAR::SCATRA::TimeIntegrationScheme>(*params,"TIMEINTEGR")),
  nsd_(DRT::Problem::Instance()->NDim()),
  numscal_(0),
  numdofpernode_(0),
  // Initialization of degrees of freedom variables
  phin_(Teuchos::null),
  phinp_(Teuchos::null),
  phiatmeshfreenodes_(Teuchos::null),
  phidtn_(Teuchos::null),
  phidtnp_(Teuchos::null),
  hist_(Teuchos::null),
  densafnp_(Teuchos::null),
  vel_(Teuchos::null),
  convel_(Teuchos::null),
  fsvel_(Teuchos::null),
  acc_(Teuchos::null),
  pre_(Teuchos::null),
  dispnp_(Teuchos::null),
  cdvel_(DRT::INPUT::IntegralValue<INPAR::SCATRA::VelocityField>(*params,"VELOCITYFIELD")),
  wss_(Teuchos::null),
  pressure_(Teuchos::null),
  meanconc_(Teuchos::null),
  densific_(0,0.0),
  c0_(0,0.0),
  discret_(actdis),
  output_ (output),
  convform_ (DRT::INPUT::IntegralValue<INPAR::SCATRA::ConvForm>(*params,"CONVFORM")),
  sysmat_(Teuchos::null),
  zeros_(Teuchos::null),
  dbcmaps_(Teuchos::null),
  neumann_loads_(Teuchos::null),
  normals_(Teuchos::null),
  residual_(Teuchos::null),
  trueresidual_(Teuchos::null),
  increment_(Teuchos::null),
  msht_(DRT::INPUT::IntegralValue<INPAR::FLUID::MeshTying>(*params,"MESHTYING")),
  // Initialization of AVM3 variables
  sysmat_sd_(Teuchos::null),
  Sep_(Teuchos::null),
  Mnsv_(Teuchos::null),
  // Initialization of turbulent flow variables
  DynSmag_(Teuchos::null),
  Vrem_(Teuchos::null),
  samstart_(0),
  samstop_(0),
  dumperiod_(0),
  turbinflow_(DRT::INPUT::IntegralValue<int>(extraparams->sublist("TURBULENT INFLOW"),"TURBULENTINFLOW")),
  numinflowsteps_(extraparams->sublist("TURBULENT INFLOW").get<int>("NUMINFLOWSTEP")),
  special_flow_("initialization"),
  forcing_(Teuchos::null),
  homisoturb_forcing_(Teuchos::null),
  // Initialization of Krylov
  updateprojection_(false),
  projector_ (Teuchos::null),
  // Initialization of
  upres_    (params->get<int>("UPRES")),
  uprestart_(params->get<int>("RESTARTEVRY")),
  neumanninflow_(DRT::INPUT::IntegralValue<int>(*params,"NEUMANNINFLOW")),
  convheatrans_(DRT::INPUT::IntegralValue<int>(*params,"CONV_HEAT_TRANS")),
  skipinitder_(DRT::INPUT::IntegralValue<int>(*params,"SKIPINITDER")),
  // Initialization of Biofilm specific stuff
  scfldgrdisp_(Teuchos::null),
  scstrgrdisp_(Teuchos::null),
  outintegrreac_(DRT::INPUT::IntegralValue<int>(*params,"OUTINTEGRREAC"))
{
  // DO NOT DEFINE ANY STATE VECTORS HERE (i.e., vectors based on row or column maps)
  // this is important since we have problems which require an extended ghosting
  // this has to be done before all state vectors are initialized

  // -------------------------------------------------------------------
  // connect degrees of freedom for periodic boundary conditions
  // -------------------------------------------------------------------
  // note: pbcs have to be correctly set up before extended ghosting is applied
  Teuchos::RCP<PeriodicBoundaryConditions> pbc = Teuchos::rcp(new PeriodicBoundaryConditions (discret_, false));
  pbc->UpdateDofsForPeriodicBoundaryConditions();

  return;
}


/*------------------------------------------------------------------------*
 |  initialize time integration                           rasthofer 09/13 |
 *------------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::Init()
{
  // print problem type
  if (myrank_ == 0)
  {
    std::cout << "###################################################"<< std::endl;
    std::cout << "# YOUR PROBLEM TYPE: " << DRT::Problem::Instance()->ProblemName() << std::endl;
    std::cout << "###################################################" << std::endl;
  }

  // -------------------------------------------------------------------
  // determine whether linear incremental or nonlinear solver
  // -------------------------------------------------------------------
  switch(solvtype_)
  {
  case INPAR::SCATRA::solvertype_nonlinear:
  case INPAR::SCATRA::solvertype_linear_incremental:
  {
    incremental_ = true;
  }
  break;
  case INPAR::SCATRA::solvertype_linear_full:
  {
    incremental_ = false;
  }
  break;
  default:
    dserror("Received illegal scatra solvertype enum.");
    break;
  }

  // -----------------------------------------------------------------------
  // determine number of degrees of freedom and transported scalars per node
  // -----------------------------------------------------------------------
  int mynumdofpernode = 0;
  if (discret_->NumMyRowNodes() > 0)
    mynumdofpernode = discret_->NumDof(0,discret_->lRowNode(0));

  // to support completely empty procs, communication is required
  discret_->Comm().MaxAll(&mynumdofpernode,&numdofpernode_,1);

  numscal_ = numdofpernode_;

  // adapt number of transported scalars if necessary
  AdaptNumScal();

  // -----------------------------------------------------------------------------
  // initialize meshtying strategy (including standard strategy without meshtying)
  // -----------------------------------------------------------------------------
  // safety checks
  if(msht_ != INPAR::FLUID::no_meshtying and s2icoupling_)
    dserror("Fluid-fluid meshtying in combination with scatra-scatra interface coupling is not implemented yet!");
  if(s2icoupling_ and !incremental_)
    dserror("Scatra-scatra interface coupling only working for incremental solve so far!\n"
        "Set the parameter SOLVERTYPE in SCALAR TRANSPORT DYNAMIC section to 'nonlinear' or 'linear_incremental'!");

  CreateMeshtyingStrategy();

  // initialize strategy
  strategy_->InitMeshtying();

  // -------------------------------------------------------------------
  // check compatibility of boundary conditions
  // -------------------------------------------------------------------
  if (neumanninflow_ and convheatrans_)
    dserror("Neumann inflow and convective heat transfer boundary conditions must not appear simultaneously for the same problem!");

  discret_->ComputeNullSpaceIfNecessary(solver_->Params(),true);

  // ensure that degrees of freedom in the discretization have been set
  if ((not discret_->Filled()) or (not discret_->HaveDofs()))
    discret_->FillComplete();

  // -------------------------------------------------------------------
  // get a vector layout from the discretization to construct matching
  // vectors and matrices: local <-> global dof numbering
  // -------------------------------------------------------------------
  const Epetra_Map* dofrowmap = discret_->DofRowMap();

  // -------------------------------------------------------------------
  // create empty system matrix (27 adjacent nodes as 'good' guess)
  // -------------------------------------------------------------------
  if (fssgd_ != INPAR::SCATRA::fssugrdiff_no and not incremental_)
    // do not save the graph if fine-scale subgrid diffusivity is used in non-incremental case (very special case)
    sysmat_ = Teuchos::rcp(new LINALG::SparseMatrix(*(discret_->DofRowMap()),27));
  else
    sysmat_ = strategy_->InitSystemMatrix();

  // -------------------------------------------------------------------
  // create vectors containing problem variables
  // -------------------------------------------------------------------
  // solutions at time n+1 and n
  phinp_ = LINALG::CreateVector(*dofrowmap,true);
  phin_  = LINALG::CreateVector(*dofrowmap,true);

  // phi at nodes for meshfree non-interpolatory basis functions
  Teuchos::RCP<DRT::MESHFREE::MeshfreeDiscretization> meshfreediscret
    = Teuchos::rcp_dynamic_cast<DRT::MESHFREE::MeshfreeDiscretization>(discret_);
  if (meshfreediscret!=Teuchos::null)
    phiatmeshfreenodes_ = LINALG::CreateVector(*dofrowmap,true);

  // temporal solution derivative at time n+1
  phidtnp_ = LINALG::CreateVector(*dofrowmap,true);
  // temporal solution derivative at time n
  phidtn_ = LINALG::CreateVector(*dofrowmap,true);

  // history vector (a linear combination of phinm, phin (BDF)
  // or phin, phidtn (One-Step-Theta, Generalized-alpha))
  hist_ = LINALG::CreateVector(*dofrowmap,true);

  // velocities (always three velocity components per node)
  // (get noderowmap of discretization for creating this multivector)
  const Epetra_Map* noderowmap = discret_->NodeRowMap();
  convel_ = Teuchos::rcp(new Epetra_MultiVector(*noderowmap,nsd_,true));
  vel_ = Teuchos::rcp(new Epetra_MultiVector(*noderowmap,nsd_,true));
  wss_ = LINALG::CreateVector(*dofrowmap,true);
  pressure_ = LINALG::CreateVector(*dofrowmap,true);
  meanconc_ = LINALG::CreateVector(*dofrowmap,true);

  // acceleration and pressure required for computation of subgrid-scale
  acc_ = Teuchos::rcp(new Epetra_MultiVector(*noderowmap,nsd_,true));
  pre_ = LINALG::CreateVector(*noderowmap,true);

  if (isale_)
  {
    // displacement field for moving mesh applications using ALE
    // (get noderowmap of discretization for creating this multivector)
    dispnp_ = Teuchos::rcp(new Epetra_MultiVector(*noderowmap,nsd_,true));
  }

  // -------------------------------------------------------------------
  // create vectors associated to boundary conditions
  // -------------------------------------------------------------------
  // a vector of zeros to be used to enforce zero dirichlet boundary conditions
  zeros_ = LINALG::CreateVector(*dofrowmap,true);

  // object holds maps/subsets for DOFs subjected to Dirichlet BCs and otherwise
  dbcmaps_ = Teuchos::rcp(new LINALG::MapExtractor());
  {
    Teuchos::ParameterList eleparams;
    // other parameters needed by the elements
    eleparams.set("total time",time_);
    discret_->EvaluateDirichlet(eleparams, zeros_, Teuchos::null, Teuchos::null,
                                Teuchos::null, dbcmaps_);
    zeros_->PutScalar(0.0); // just in case of change
  }

  // -------------------------------------------------------------------
  // ensure that the Transport std::string was removed from conditions
  // -------------------------------------------------------------------
  {
    DRT::Condition* cond = discret_->GetCondition("TransportDirichlet");
    if (cond) dserror("Found a Transport Dirichlet condition. Remove Transport std::string!");

    cond = discret_->GetCondition("TransportPointNeumann");
    if (cond) dserror("Found a Transport Point Neumann condition. Remove Transport std::string!");
    cond = discret_->GetCondition("TransportLineNeumann");
    if (cond) dserror("Found a Transport Line Neumann condition. Remove Transport std::string!");
    cond = discret_->GetCondition("TransportSurfaceNeumann");
    if (cond) dserror("Found a Transport Surface Neumann condition. Remove Transport std::string!");
  }

  // -------------------------------------------------------------------
  // create vectors associated to solution process
  // -------------------------------------------------------------------
  // the vector containing body and surface forces
  neumann_loads_= LINALG::CreateVector(*dofrowmap,true);

  // the residual vector --- more or less the rhs
  residual_ = LINALG::CreateVector(*dofrowmap,true);

  // residual vector containing the normal boundary fluxes
  trueresidual_ = LINALG::CreateVector(*dofrowmap,true);

  // incremental solution vector
  increment_ = LINALG::CreateVector(*dofrowmap,true);

  // subgrid-diffusivity(-scaling) vector
  // (used either for AVM3 approach or temperature equation
  //  with all-scale subgrid-diffusivity model)
  if (fssgd_ != INPAR::SCATRA::fssugrdiff_no)
    subgrdiff_ = LINALG::CreateVector(*dofrowmap,true);

  // -------------------------------------------------------------------
  // set parameters associated to potential statistical flux evaluations
  // -------------------------------------------------------------------

  // initialize vector for statistics (assume a maximum of 10 conditions)
  sumnormfluxintegral_ = Teuchos::rcp(new Epetra_SerialDenseVector(10));

  // get desired scalar id's for flux output
  {
    // write one by one of scalars (as flux output in the input file defined)
    // to the temporary variable word1
    int word1 = 0;
    std::istringstream mystream(Teuchos::getNumericStringParameter(*params_,"WRITEFLUX_IDS"));
    while (mystream >> word1)

    writefluxids_->push_back(word1);

    // default value (-1): flux is written for all dof's
    // scalar transport: numdofpernode_ = numscal_
    // elch:             numdofpernode_ = numscal_+1
    // -> current flux for potential only if div i is used to close the system otherwise zero
    if ((*writefluxids_)[0]==(-1)) //default is to perform flux output for ALL scalars
    {
      writefluxids_->resize(numdofpernode_);
      for(int k=0;k<numdofpernode_;++k)
        (*writefluxids_)[k]=k+1;
    }
    // flux_ vector is initized when CalcFlux() is called
    if ((writeflux_!=INPAR::SCATRA::flux_no) and (myrank_ == 0))
    {
      IO::cout << "Flux output is performed for scalars: ";
      for (unsigned int i=0; i < writefluxids_->size();i++)
      {
        const int id = (*writefluxids_)[i];
        IO::cout << (*writefluxids_)[i] << " ";
        if ((id<1) or (id > numdofpernode_)) // check validity of these numbers as well !
          dserror("Received illegal scalar id for flux output: %d",id);
      }
      IO::cout << IO::endl;
    }
  }

  // -------------------------------------------------------------------
  // preparations for turbulence models
  // -------------------------------------------------------------------
  InitTurbulenceModel(dofrowmap, noderowmap);

  // -------------------------------------------------------------------
  // set initial field
  // -------------------------------------------------------------------
  SetInitialField(DRT::INPUT::IntegralValue<INPAR::SCATRA::InitialField>(*params_,"INITIALFIELD"),
      params_->get<int>("INITFUNCNO"));

  // -------------------------------------------------------------------
  // preparations for natural convection
  // -------------------------------------------------------------------
  if (DRT::INPUT::IntegralValue<int>(*params_,"NATURAL_CONVECTION") == true)
  {
    // allocate global density vector and initialize
    densafnp_ = LINALG::CreateVector(*discret_->DofRowMap(),true);
    densafnp_->PutScalar(1.);
  }

  return;
} // ScaTraTimIntImpl::Init()


/*----------------------------------------------------------------------*
 | perform setup of natural convection                       fang 08/14 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::SetupNatConv()
{
  // calculate the initial mean concentration value
  if (numscal_ < 1) dserror("Error since numscal = %d. Not allowed since < 1",numscal_);
  c0_.resize(numscal_);

  discret_->ClearState();
  discret_->SetState("phinp",phinp_);

  // set action for elements
  Teuchos::ParameterList eleparams;
  eleparams.set<int>("action",SCATRA::calc_mean_scalars);
  eleparams.set("inverting",false);

  // provide displacement field in case of ALE
  if (isale_)
    discret_->AddMultiVectorToParameterList(eleparams,"dispnp",dispnp_);

  // evaluate integrals of concentrations and domain
  Teuchos::RCP<Epetra_SerialDenseVector> scalars
  = Teuchos::rcp(new Epetra_SerialDenseVector(numscal_+1));
  discret_->EvaluateScalars(eleparams, scalars);
  discret_->ClearState();   // clean up

  // calculate mean concentrations
  const double domint = (*scalars)[numscal_];
  if (std::abs(domint) < EPS15)
    dserror("Domain has zero volume!");
  for(int k=0; k<numscal_; ++k)
    c0_[k] = (*scalars)[k]/domint;

  // initialization of the densification coefficient vector
  densific_.resize(numscal_);
  DRT::Element*   element = discret_->lRowElement(0);
  Teuchos::RCP<MAT::Material>  mat = element->Material();

  if (mat->MaterialType() == INPAR::MAT::m_matlist or mat->MaterialType() == INPAR::MAT::m_matlist_reactions)
  {
    Teuchos::RCP<const MAT::MatList> actmat = Teuchos::rcp_static_cast<const MAT::MatList>(mat);

    for (int k = 0;k<numscal_;++k)
    {
      const int matid = actmat->MatID(k);
      Teuchos::RCP<const MAT::Material> singlemat = actmat->MaterialById(matid);

      if (singlemat->MaterialType() == INPAR::MAT::m_scatra)
      {
        Teuchos::RCP<const MAT::ScatraMat> actsinglemat = Teuchos::rcp_static_cast<const MAT::ScatraMat>(singlemat);

        densific_[k] = actsinglemat->Densification();

        if (densific_[k] < 0.0) dserror("received negative densification value");
      }
      else
        dserror("Material type is not allowed!");
    }
  }

  // for a single species calculation
  else if (mat->MaterialType() == INPAR::MAT::m_scatra)
  {
    Teuchos::RCP<const MAT::ScatraMat> actmat = Teuchos::rcp_static_cast<const MAT::ScatraMat>(mat);

    densific_[0] = actmat->Densification();

    if (densific_[0] < 0.0) dserror("received negative densification value");
    if (numscal_ > 1) dserror("Single species calculation but numscal = %d > 1",numscal_);
  }
  else
    dserror("Material type is not allowed!");

  return;
} // ScaTraTimIntImpl::SetupNatConv


/*----------------------------------------------------------------------*
 | initialization of turbulence model                        ehrl 05/14 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::InitTurbulenceModel(
    const Epetra_Map* dofrowmap,
    const Epetra_Map* noderowmap)
{
  // get fluid turbulence sublist
  Teuchos::ParameterList * turbparams =&(extraparams_->sublist("TURBULENCE MODEL"));

  // parameters for statistical evaluation of normal fluxes
  samstart_  = turbparams->get<int>("SAMPLING_START");
  samstop_   = turbparams->get<int>("SAMPLING_STOP" );
  dumperiod_ = turbparams->get<int>("DUMPING_PERIOD");
  if (dumperiod_ < 0) dserror("dumperiod_ is negative!");

  // -------------------------------------------------------------------
  // necessary only for AVM3 approach:
  // initialize subgrid-diffusivity matrix + respective output
  // -------------------------------------------------------------------
  if (fssgd_ != INPAR::SCATRA::fssugrdiff_no and DRT::INPUT::IntegralValue<int>(*turbparams,"TURBMODEL_LS"))
  {
    sysmat_sd_ = Teuchos::rcp(new LINALG::SparseMatrix(*dofrowmap,27));

    // fine-scale velocities (always three velocity components per node)
    // transferred from the fluid field
    // only Smagorinsky small
    if (fssgd_ == INPAR::SCATRA::fssugrdiff_smagorinsky_small)
      fsvel_ = Teuchos::rcp(new Epetra_MultiVector(*noderowmap,nsd_,true));

    // Output
    if (myrank_ == 0)
    {
      std::cout << "SCATRA: Fine-scale subgrid-diffusivity approach based on AVM3: ";
      std::cout << fssgd_;
      std::cout << " with turbulent Prandtl number Prt= ";
      std::cout << extraparams_->sublist("SUBGRID VISCOSITY").get<double>("C_TURBPRANDTL") ;
      std::cout << &std::endl << &std::endl;
    }

    if (turbparams->get<std::string>("PHYSICAL_MODEL") != "Multifractal_Subgrid_Scales")
    {
      if (fssgd_ == INPAR::SCATRA::fssugrdiff_smagorinsky_small
          and turbparams->get<std::string>("FSSUGRVISC") != "Smagorinsky_small")
        dserror ("Same subgrid-viscosity approach expected!");
      if (fssgd_ == INPAR::SCATRA::fssugrdiff_smagorinsky_all
          and turbparams->get<std::string>("FSSUGRVISC") != "Smagorinsky_all")
        dserror ("Same subgrid-viscosity approach expected!");
    }
  }
  else fssgd_ = INPAR::SCATRA::fssugrdiff_no; // in case of not "TURBMODEL_LS"

  // -------------------------------------------------------------------
  // get turbulence model and parameters
  // -------------------------------------------------------------------
  turbmodel_ = INPAR::FLUID::no_model;

  if (DRT::INPUT::IntegralValue<int>(*turbparams,"TURBMODEL_LS"))
  {
    // set turbulence model
    if (turbparams->get<std::string>("PHYSICAL_MODEL") == "Smagorinsky")
    {
      turbmodel_ = INPAR::FLUID::smagorinsky;

      // Output
      if (turbmodel_ and myrank_ == 0)
      {
        std::cout << "All-scale subgrid-diffusivity model: ";
        std::cout << turbparams->get<std::string>("PHYSICAL_MODEL");
        std::cout << &std::endl << &std::endl;
      }
    }
    else if (turbparams->get<std::string>("PHYSICAL_MODEL") == "Dynamic_Smagorinsky")
    {
      turbmodel_ = INPAR::FLUID::dynamic_smagorinsky;
      // access to the dynamic Smagorinsky class will provided by the
      // scatra fluid couling algorithm
    }
    else if (turbparams->get<std::string>("PHYSICAL_MODEL") == "Multifractal_Subgrid_Scales")
    {
      turbmodel_ = INPAR::FLUID::multifractal_subgrid_scales;

      // initalize matrix used to build the scale separation operator
      sysmat_sd_ = Teuchos::rcp(new LINALG::SparseMatrix(*dofrowmap,27));

      // fine-scale velocities (always three velocity components per node)
      // transferred from the fluid field
      fsvel_ = Teuchos::rcp(new Epetra_MultiVector(*noderowmap,nsd_,true));

      Teuchos::ParameterList * mfsparams =&(extraparams_->sublist("MULTIFRACTAL SUBGRID SCALES"));
      if (mfsparams->get<std::string>("SCALE_SEPARATION")!= "algebraic_multigrid_operator")
       dserror("Only scale separation by plain algebraic multigrid available in scatra!");

      // Output
      if (turbmodel_ and myrank_ == 0)
      {
        std::cout << "Multifractal subgrid-scale model: ";
        std::cout << turbparams->get<std::string>("PHYSICAL_MODEL");
        std::cout << &std::endl << &std::endl;
      }
    }
    else if (turbparams->get<std::string>("PHYSICAL_MODEL") == "Dynamic_Vreman")
    {
      // equivalent to dynamic smagorinsky
      turbmodel_ = INPAR::FLUID::dynamic_vreman;
    }

    // warning No. 1: if classical (all-scale) turbulence model other than
    // Smagorinsky or multifractal subrgid-scale modeling
    // is intended to be used
    if (turbparams->get<std::string>("PHYSICAL_MODEL") != "Smagorinsky" and
        turbparams->get<std::string>("PHYSICAL_MODEL") != "Dynamic_Smagorinsky" and
        turbparams->get<std::string>("PHYSICAL_MODEL") != "Multifractal_Subgrid_Scales" and
        turbparams->get<std::string>("PHYSICAL_MODEL") != "Dynamic_Vreman" and
        turbparams->get<std::string>("PHYSICAL_MODEL") != "no_model")
      dserror("No classical (all-scale) turbulence model other than constant-coefficient Smagorinsky model and multifractal subrgid-scale modeling currently possible!");

    // warning No. 2: if classical (all-scale) turbulence model and fine-scale
    // subgrid-viscosity approach are intended to be used simultaneously
    if (turbmodel_==INPAR::FLUID::smagorinsky and fssgd_ != INPAR::SCATRA::fssugrdiff_no)
      dserror("No combination of classical turbulence model and fine-scale subgrid-diffusivity approach currently possible!");
  }

  if (turbmodel_ != INPAR::FLUID::no_model and numscal_ > 1)
    dserror("Turbulent passive scalar transport not supported for more than one scalar!");

  // -------------------------------------------------------------------
  // initialize forcing for homogeneous isotropic turbulence
  // -------------------------------------------------------------------
  // flag for special flow
  special_flow_ = extraparams_->sublist("TURBULENCE MODEL").get<std::string>("CANONICAL_FLOW","no");
  if (special_flow_ == "scatra_forced_homogeneous_isotropic_turbulence")
  {
    if (extraparams_->sublist("TURBULENCE MODEL").get<std::string>("SCALAR_FORCING")=="isotropic")
    {
      forcing_ = LINALG::CreateVector(*dofrowmap,true);
      forcing_->PutScalar(0.0);
    }
  }

  return;
} // ScaTraTimIntImpl::InitTurbulenceModel()


/*----------------------------------------------------------------------*
 | create vectors for Krylov projection if necessary         ehrl 12/13 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::PrepareKrylovProjection()
{
  // sysmat might be singular (some modes are defined only up to a constant)
  // in this case, we need basis vectors for the nullspace/kernel

  // get condition "KrylovSpaceProjection" from discretization
  std::vector<DRT::Condition*> KSPCond;
  discret_->GetCondition("KrylovSpaceProjection",KSPCond);
  int numcond = KSPCond.size();
  int numscatra = 0;

  DRT::Condition* kspcond = NULL;
  // check if for scatra Krylov projection is required
  for(int icond = 0; icond < numcond; icond++)
  {
    const std::string* name = KSPCond[icond]->Get<std::string>("discretization");
    if (*name == "scatra")
    {
      numscatra++;
      kspcond = KSPCond[icond];
    }
  }

  // initialize variables for Krylov projection if necessary
  if (numscatra == 1)
  {
    SetupKrylovSpaceProjection(kspcond);
    if (myrank_ == 0)
      std::cout << "\nSetup of KrylovSpaceProjection in scatra field\n" << std::endl;
  }
  else if (numscatra == 0)
  {
    projector_ = Teuchos::null;
  }
  else
    dserror("Received more than one KrylovSpaceCondition for scatra field");

  return;
}


/*----------------------------------------------------------------------*
 | Destructor dtor                                   (public) gjb 04/08 |
 *----------------------------------------------------------------------*/
SCATRA::ScaTraTimIntImpl::~ScaTraTimIntImpl()
{
  return;
}


/*========================================================================*/
//! set element parameters
/*========================================================================*/

/*--------------------------------------------------------------------------------*
 | set all general parameters for element                              fang 10/14 |
 *--------------------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::SetElementGeneralParameters(bool calcinitialtimederivative)
{
  Teuchos::ParameterList eleparams;

  eleparams.set<int>("action",SCATRA::set_general_scatra_parameter);

  eleparams.set<int>("form of convective term",convform_);
  eleparams.set("isale",isale_);

  // set flag for writing the flux vector fields
  eleparams.set<int>("writeflux",writeflux_);

  //! set vector containing ids of scalars for which flux vectors are calculated
  eleparams.set<Teuchos::RCP<std::vector<int> > >("writefluxids",writefluxids_);

  // parameters for stabilization
  eleparams.sublist("STABILIZATION") = params_->sublist("STABILIZATION");
  if(calcinitialtimederivative)      // deactivate stabilization when calculating initial time derivative
  {
    Teuchos::setStringToIntegralParameter<int>("STABTYPE",
        "no_stabilization",
        "type of stabilization (if any)",
        Teuchos::tuple<std::string>("no_stabilization"),
        Teuchos::tuple<std::string>("Do not use any stabilization"),
        Teuchos::tuple<int>(INPAR::SCATRA::stabtype_no_stabilization),
        &eleparams.sublist("STABILIZATION"));
    DRT::INPUT::BoolParameter("SUGRVEL","no","potential incorporation of subgrid-scale velocity",&eleparams.sublist("STABILIZATION"));
    DRT::INPUT::BoolParameter("ASSUGRDIFF","no",
          "potential incorporation of all-scale subgrid diffusivity (a.k.a. discontinuity-capturing) term",&eleparams.sublist("STABILIZATION"));
  }

  // parameters for finite difference check
  if(calcinitialtimederivative)      // deactivate finite difference check when calculating initial time derivative
    eleparams.set<int>("fdcheck",INPAR::SCATRA::fdcheck_none);
  else
    eleparams.set<int>("fdcheck",fdcheck_);
  eleparams.set<double>("fdcheckeps",fdcheckeps_);
  eleparams.set<double>("fdchecktol",fdchecktol_);

  // additional problem-specific parameters for non-standard scalar transport problems (electrochemistry etc.)
  SetElementSpecificScaTraParameters(eleparams);

  // call standard loop over elements
  discret_->Evaluate(eleparams,Teuchos::null,Teuchos::null,Teuchos::null,Teuchos::null,Teuchos::null);

  return;
}


/*----------------------------------------------------------------------*
 | set turbulence parameters for element                rasthofer 11/13 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::SetElementTurbulenceParameters(bool calcinitialtimederivative)
{
  Teuchos::ParameterList eleparams;

  eleparams.set<int>("action",SCATRA::set_turbulence_scatra_parameter);

  eleparams.sublist("TURBULENCE MODEL") = extraparams_->sublist("TURBULENCE MODEL");
  if(calcinitialtimederivative)      // deactivate turbulence model when calculating initial time derivative
    Teuchos::setStringToIntegralParameter<int>(
      "PHYSICAL_MODEL",
      "no_model",
      "Classical LES approaches require an additional model for\nthe turbulent viscosity.",
      Teuchos::tuple<std::string>("no_model"),
      Teuchos::tuple<std::string>("If classical LES is our turbulence approach, this is a contradiction and should cause a dserror."),
      Teuchos::tuple<int>(INPAR::FLUID::no_model),
      &eleparams.sublist("TURBULENCE MODEL"));

  // set model-dependent parameters
  eleparams.sublist("SUBGRID VISCOSITY") = extraparams_->sublist("SUBGRID VISCOSITY");

  // set parameters for multifractal subgrid-scale modeling
  eleparams.sublist("MULTIFRACTAL SUBGRID SCALES") = extraparams_->sublist("MULTIFRACTAL SUBGRID SCALES");

  eleparams.set<bool>("turbulent inflow",turbinflow_);

  if(calcinitialtimederivative)
    eleparams.set<int>("fs subgrid diffusivity",INPAR::SCATRA::fssugrdiff_no);
  else
    eleparams.set<int>("fs subgrid diffusivity",fssgd_);

  // call standard loop over elements
  discret_->Evaluate(eleparams,Teuchos::null,Teuchos::null,Teuchos::null,Teuchos::null,Teuchos::null);

  return;
}


/*==========================================================================*/
// general framework
/*==========================================================================*/

/*--- set, prepare, and predict --------------------------------------------*/

/*----------------------------------------------------------------------*
 | setup the variables to do a new time step          (public)  vg 08/07|
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::PrepareTimeStep()
{
  // time measurement: prepare time step
  TEUCHOS_FUNC_TIME_MONITOR("SCATRA:    + prepare time step");

  // -------------------------------------------------------------------
  //                       initialization
  // -------------------------------------------------------------------
  if (step_ == 0)
  {
    // if initial velocity field has not been set here, the initial time derivative of phi will be
    // calculated wrongly for some time integration schemes
    if(not skipinitder_)
    {
      if (initialvelset_) PrepareFirstTimeStep();
      else dserror("Initial velocity field has not been set");
    }

    // Initialize Nernst-BC
    InitNernstBC();
  }

  // -------------------------------------------------------------------
  //              set time dependent parameters
  // -------------------------------------------------------------------
  // note the order of the following three functions is important
  IncrementTimeAndStep();

  // -------------------------------------------------------------------
  // set part of the rhs vector belonging to the old timestep
  // -------------------------------------------------------------------
  SetOldPartOfRighthandside();
  SetElementTimeParameter();

  // -------------------------------------------------------------------
  //         evaluate Dirichlet and Neumann boundary conditions
  // -------------------------------------------------------------------
  // TODO: dirichlet auch im Fall von genalpha phinp
  // neuman n + alfaf
  ApplyDirichletBC(time_,phinp_,Teuchos::null);
  ApplyNeumannBC(neumann_loads_);

  // By definition: Applying DC on the slave side of an internal interface is not allowed
  //                since it leads to an over-constraint system
  // Therefore, nodes belonging to the slave side of an internal interface have to be excluded from the DC.
  // However, a velocity value (projected from the Dirichlet condition on the master side)
  // has to be assigned to the DOF's on the slave side in order to evaluate the system matrix completely

  // Preparation for including DC on the master side in the condensation process
  strategy_->IncludeDirichletInCondensation();

  // -------------------------------------------------------------------
  //     update velocity field if given by function AND time curve
  // -------------------------------------------------------------------
  if (cdvel_ == INPAR::SCATRA::velocity_function_and_curve)
    SetVelocityField();

  // -------------------------------------------------------------------
  //           preparation of AVM3-based scale separation
  // -------------------------------------------------------------------
  if ((step_==1 or (turbinflow_ and step_==numinflowsteps_+1)) and
      (fssgd_ != INPAR::SCATRA::fssugrdiff_no or turbmodel_ == INPAR::FLUID::multifractal_subgrid_scales))
     AVM3Preparation();

  // -------------------------------------------------------------------
  // compute values at intermediate time steps (only for gen.-alpha)
  // -------------------------------------------------------------------
  ComputeIntermediateValues();

  return;
} // ScaTraTimIntImpl::PrepareTimeStep


/*----------------------------------------------------------------------*
 | preparations for solve                                (public) mr.x  |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::PrepareLinearSolve()
{
  // special preparations for multifractal subgrid-scale model
  if (turbmodel_==INPAR::FLUID::multifractal_subgrid_scales)
    RecomputeMeanCsgsB();

  // call elements to calculate system matrix and rhs and assemble
  AssembleMatAndRHS();

  // potential residual scaling and potential addition of Neumann terms
  ScalingAndNeumann();

  // apply Dirichlet boundary conditions
  ApplyDirichletToSystem();
}


/*----------------------------------------------------------------------*
 | update the velocity field                                  gjb 04/08 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::SetVelocityField()
{
  if (cdvel_ == INPAR::SCATRA::velocity_zero)
  {
    convel_->PutScalar(0.); // just to be sure!
    vel_->PutScalar(0.);
  }
  else if ((cdvel_ == INPAR::SCATRA::velocity_function)
      or (cdvel_ == INPAR::SCATRA::velocity_function_and_curve))
  {
    int err(0);
    const int velfuncno = params_->get<int>("VELFUNCNO");
    const int velcurveno = params_->get<int>("VELCURVENO");
    // loop all nodes on the processor
    for(int lnodeid=0;lnodeid<discret_->NumMyRowNodes();lnodeid++)
    {
      // get the processor local node
      DRT::Node*  lnode      = discret_->lRowNode(lnodeid);
      for(int index=0;index<nsd_;++index)
      {
        double value = DRT::Problem::Instance()->Funct(velfuncno-1).Evaluate(index,lnode->X(),time_,NULL);
        if (cdvel_ == INPAR::SCATRA::velocity_function_and_curve)
        {
          value *= DRT::Problem::Instance()->Curve(velcurveno-1).f(time_);
        }
        err = convel_->ReplaceMyValue (lnodeid, index, value);
        if (err!=0) dserror("error while inserting a value into convel_");
        err = vel_->ReplaceMyValue (lnodeid, index, value);
        if (err!=0) dserror("error while inserting a value into vel_");
      }
    }
  }
  else
    dserror("Wrong SetVelocity() action for velocity field type %d!",cdvel_);

  // initial velocity field has now been set
  if (step_ == 0) initialvelset_ = true;

  return;

} // ScaTraImplicitTimeInt::SetVelocityField


/*----------------------------------------------------------------------*
 | Set Wall Shear Stresses                                hemmler 05/14 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::SetWallShearStresses(Teuchos::RCP<const Epetra_Vector> wss, Teuchos::RCP<const DRT::DofSet>   dofset, Teuchos::RCP<DRT::Discretization> dis)
{
  // We have to be sure that everything is still matching.
  if (not dis->NodeRowMap()->SameAs(*(discret_->NodeRowMap())))
    dserror("Fluid/Poro and Scatra noderowmaps are NOT identical. Emergency!");

  //---------------------------------------------------------------------------
  // transfer of dofs
  // (We rely on the fact that the scatra discretization is a clone of the
  // fluid or poro mesh, respectively, meaning that a scatra node has the
  // same local (and global) ID as its corresponding fluid/structure node.)
  //---------------------------------------------------------------------------
  // loop over all local nodes of scatra discretization
  for (int lnodeid=0; lnodeid < discret_->NumMyRowNodes(); lnodeid++)
  {
    // get local fluid/poro node with the same lnodeid
    DRT::Node* lnode = dis->lRowNode(lnodeid);

    // care for the slave nodes of rotationally symm. periodic boundary conditions
    double rotangle(0.0);
    bool havetorotate = FLD::IsSlaveNodeOfRotSymPBC(lnode,rotangle);

    // get degrees of freedom associated with this fluid/poro node
    // two particular cases have to be considered:
    // - in non-XFEM case, the first dofset is always considered, allowing for
    //   using multiple dof sets, e.g., for structure-based scalar transport
    // - for XFEM, a different nodeset is required
    std::vector<int> nodedofs;
    if (dofset == Teuchos::null) nodedofs = dis->Dof(0,lnode);
    else                         nodedofs = (*dofset).Dof(lnode);

    // determine number of space dimensions
    const int numdim = DRT::Problem::Instance()->NDim();

    //-------------------------------------------------------------------------
    // transfer of wss dofs
    //-------------------------------------------------------------------------
    double tmp=0;

    for (int index=0;index < numdim; ++index)
    {
      // get global and local ID
      const int gid = nodedofs[index];
      const int lid = wss->Map().LID(gid);
      if (lid < 0) dserror("Local ID not found in map for given global ID!");

      //-----------------------------------------------------------------------
      // get wss and calculate the euclidean norm
      //-----------------------------------------------------------------------
      double WallShearStress = (*wss)[lid];

      // component of rotated vector field
      if (havetorotate)  WallShearStress = FLD::GetComponentOfRotatedVectorField(index,wss,lid,rotangle);

      // insert WSS of Fluid discretization in Map of Scatra discretization
      //squares of the euclidean norm
      tmp+=WallShearStress*WallShearStress;
    }

    //for case of multi scalars
    for(int i=0;i<numscal_;i++)
    {
      (*wss_)[lnodeid * numscal_ + i]=sqrt(tmp);  //square root of euclidean norm
    }
  } //for

  return;
} // ScaTraTimIntImpl::SetWallShearStresses


/*----------------------------------------------------------------------*
 | Set Pressure Field                                     hemmler 05/14 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::SetPressureFields(Teuchos::RCP<const Epetra_Vector> pres, Teuchos::RCP<const DRT::DofSet>   dofset, Teuchos::RCP<DRT::Discretization> dis)
{
  if (not dis->NodeRowMap()->SameAs(*(discret_->NodeRowMap())))
    dserror("Fluid/Structure and Scatra noderowmaps are NOT identical. Emergency!");

  //---------------------------------------------------------------------------
  // transfer of dofs
  // (We rely on the fact that the scatra discretization is a clone of the
  // fluid or structure mesh, respectively, meaning that a scatra node has the
  // same local (and global) ID as its corresponding fluid/structure node.)
  //---------------------------------------------------------------------------
  // loop over all local nodes of scatra discretization
  for (int lnodeid=0; lnodeid < discret_->NumMyRowNodes(); lnodeid++)
  {
    // get local fluid/structure node with the same lnodeid
    DRT::Node* lnode = dis->lRowNode(lnodeid);

    // get degrees of freedom associated with this fluid/structure node
    // two particular cases have to be considered:
    // - in non-XFEM case, the first dofset is always considered, allowing for
    //   using multiple dof sets, e.g., for structure-based scalar transport
    // - for XFEM, a different nodeset is required
    std::vector<int> nodedofs;
    if (dofset == Teuchos::null) nodedofs = dis->Dof(0,lnode);
    else                         nodedofs = (*dofset).Dof(lnode);

    // determine number of space dimensions
    const int numdim = DRT::Problem::Instance()->NDim();

    //-------------------------------------------------------------------------
    // transfer of pressure dofs
    //-------------------------------------------------------------------------
    // get global and local ID
    const int gid = nodedofs[numdim]; //this entry corresponds to the pressure
    // const int lid = dofrowmap->LID(gid);
    const int lid = pres->Map().LID(gid);
    if (lid < 0) dserror("Local ID not found in map for given global ID!");

    // insert pressure of Fluid discretization in Map of Scatra discretization
    for(int i=0;i<numscal_;i++)
    {
      (*pressure_)[lnodeid * numscal_ + i]=(*pres)[lid];
    }

  } //for

  return;
}// ScaTraTimIntImpl::SetPressureFields

/*----------------------------------------------------------------------*
 | Set Pressure Field                                     hemmler 05/14 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::SetMeanConcentration(Teuchos::RCP<Epetra_Vector> MeanConc)
{
meanconc_=MeanConc;
return;
} // ScaTraTimIntImpl::SetMeanConcentration


/*----------------------------------------------------------------------*
 | set convective velocity field (+ pressure and acceleration field as  |
 | well as fine-scale velocity field, if required)            gjb 05/09 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::SetVelocityField(
Teuchos::RCP<const Epetra_Vector> convvel,
Teuchos::RCP<const Epetra_Vector> acc,
Teuchos::RCP<const Epetra_Vector> vel,
Teuchos::RCP<const Epetra_Vector> fsvel,
Teuchos::RCP<const DRT::DofSet>   dofset,
Teuchos::RCP<DRT::Discretization> dis,
bool setpressure)
{
  //---------------------------------------------------------------------------
  // preliminaries
  //---------------------------------------------------------------------------
  if (cdvel_ != INPAR::SCATRA::velocity_Navier_Stokes)
    dserror("Wrong SetVelocityField() called for velocity field type %d!",cdvel_);

  TEUCHOS_FUNC_TIME_MONITOR("SCATRA: set convective velocity field");

//#ifdef DEBUG   // is this costly, when we do this test always?
  // We rely on the fact, that the nodal distribution of both fields is the same.
  // Although Scatra discretization was constructed as a clone of the fluid or
  // structure mesh, respectively, at the beginning, the nodal distribution may
  // have changed meanwhile (e.g., due to periodic boundary conditions applied only
  // to the fluid field)!
  // We have to be sure that everything is still matching.
  if (not dis->NodeRowMap()->SameAs(*(discret_->NodeRowMap())))
    dserror("Fluid/Structure and Scatra noderowmaps are NOT identical. Emergency!");
//#endif

  // define error variable
  int err(0);

  // boolean indicating whether acceleration vector exists
  // -> if yes, subgrid-scale velocity may need to be computed on element level
  //bool sgvelswitch = (acc != Teuchos::null);
  const bool accswitch = (acc != Teuchos::null);

  // boolean indicating whether fine-scale velocity vector exists
  // -> if yes, multifractal subgrid-scale modeling is applied
  bool fsvelswitch = (fsvel != Teuchos::null);

  // some thing went wrong if we want to use multifractal subgrid-scale modeling
  // and have not got the fine-scale velocity
  if (step_>=1 and (turbmodel_ == INPAR::FLUID::multifractal_subgrid_scales
       or fssgd_ == INPAR::SCATRA::fssugrdiff_smagorinsky_small)
       and not fsvelswitch)
    dserror("Fine-scale velocity expected for multifractal subgrid-scale modeling!");
  // as fsvelswitch is also true for smagorinsky_all, we have to reset fsvelswitch
  // as the corresponding vector, which is not necessary, is not provided in scatra
  if (fssgd_ == INPAR::SCATRA::fssugrdiff_smagorinsky_all and fsvelswitch)
    fsvelswitch = false;
  // as fsvelswitch is true in case of turned-off model in scalar field,
  // we have to ensure false
  if (turbmodel_ == INPAR::FLUID::no_model and fssgd_ == INPAR::SCATRA::fssugrdiff_no)
    fsvelswitch = false;

  //---------------------------------------------------------------------------
  // transfer of dofs
  // (We rely on the fact that the scatra discretization is a clone of the
  // fluid or structure mesh, respectively, meaning that a scatra node has the
  // same local (and global) ID as its corresponding fluid/structure node.)
  //---------------------------------------------------------------------------
  // loop over all local nodes of scatra discretization
  for (int lnodeid=0; lnodeid < discret_->NumMyRowNodes(); lnodeid++)
  {
    // get local fluid/structure node with the same lnodeid
    DRT::Node* lnode = dis->lRowNode(lnodeid);

    // care for the slave nodes of rotationally symm. periodic boundary conditions
    double rotangle(0.0);
    bool havetorotate = FLD::IsSlaveNodeOfRotSymPBC(lnode,rotangle);

    // get degrees of freedom associated with this fluid/structure node
    // two particular cases have to be considered:
    // - in non-XFEM case, the first dofset is always considered, allowing for
    //   using multiple dof sets, e.g., for structure-based scalar transport
    // - for XFEM, a different nodeset is required
    std::vector<int> nodedofs;
    if (dofset == Teuchos::null) nodedofs = dis->Dof(0,lnode);
    else                         nodedofs = (*dofset).Dof(lnode);

    // determine number of space dimensions
    const int numdim = DRT::Problem::Instance()->NDim();

    //-------------------------------------------------------------------------
    // transfer of velocity dofs
    //-------------------------------------------------------------------------
    for (int index=0;index < numdim; ++index)
    {
      // get global and local ID
      const int gid = nodedofs[index];
      // const int lid = dofrowmap->LID(gid);
      const int lid = convvel->Map().LID(gid);
      if (lid < 0) dserror("Local ID not found in map for given global ID!");

      //-----------------------------------------------------------------------
      // get convective velocity
      //-----------------------------------------------------------------------
      double convelocity = (*convvel)[lid];

      // component of rotated vector field
      if (havetorotate)  convelocity = FLD::GetComponentOfRotatedVectorField(index,convvel,lid,rotangle);

      // insert velocity value into node-based vector
      err = convel_->ReplaceMyValue(lnodeid,index,convelocity);
      if (err != 0) dserror("Error while inserting value into vector convel_!");

      //-----------------------------------------------------------------------
      // get velocity
      //-----------------------------------------------------------------------
      if (vel != Teuchos::null)
      {
        // get value of corresponding velocity component
        double velocity = (*vel)[lid];

        // component of rotated vector field
        if (havetorotate) velocity = FLD::GetComponentOfRotatedVectorField(index,vel,lid,rotangle);

        // insert velocity value into node-based vector
        err = vel_->ReplaceMyValue(lnodeid,index,velocity);
        if (err != 0) dserror("Error while inserting value into vector vel_!");
      }
      else
      {
        // if velocity vector is not provided by the respective algorithm, we
        // assume that it equals the given convective velocity:
        // insert velocity value into node-based vector
        err = vel_->ReplaceMyValue(lnodeid,index,convelocity);
        if (err != 0) dserror("Error while inserting value into vector vel_!");
      }

      //-----------------------------------------------------------------------
      // get acceleration, if required
      //-----------------------------------------------------------------------
      if (accswitch)
      {
        // get value of corresponding acceleration component
        double acceleration = (*acc)[lid];

        // component of rotated vector field
        if (havetorotate) acceleration = FLD::GetComponentOfRotatedVectorField(index,acc,lid,rotangle);

        // insert acceleration value into node-based vector
        err = acc_->ReplaceMyValue(lnodeid,index,acceleration);
        if (err != 0) dserror("Error while inserting value into vector acc_!");
      }

      //-----------------------------------------------------------------------
      // get fine-scale velocity, if required
      //-----------------------------------------------------------------------
      if (fsvelswitch)
      {
        // get value of corresponding fine-scale velocity component
        double fsvelocity = (*fsvel)[lid];

        // component of rotated vector field
        if (havetorotate) fsvelocity = FLD::GetComponentOfRotatedVectorField(index,fsvel,lid,rotangle);

        // insert fine-scale velocity value into node-based vector
        err = fsvel_->ReplaceMyValue(lnodeid,index,fsvelocity);
        if (err != 0) dserror("Error while inserting value into vector fsvel_!");
      }
    }

    //-------------------------------------------------------------------------
    // transfer of pressure dofs, if required
    //-------------------------------------------------------------------------
    if (setpressure)
    {
      if(convvel==Teuchos::null)
        dserror("Velocity state is Teuchos::null. Velocity state is needed to extract pressure from");

      // get global and local ID
      const int gid = nodedofs[numdim];
      // const int lid = dofrowmap->LID(gid);
      const int lid = convvel->Map().LID(gid);
      if (lid < 0) dserror("Local ID not found in map for given global ID!");

      // get value of corresponding pressure component
      double pressure = (*convvel)[lid];

      // insert pressure value into node-based vector
      err = pre_->ReplaceMyValue(lnodeid,0,pressure);
      if (err != 0) dserror("Error while inserting value into vector pre_!");
    }

  }

  // confirm that initial velocity field has now been set
  if (step_ == 0) initialvelset_ = true;

  return;

} // ScaTraTimIntImpl::SetVelocityField


/*----------------------------------------------------------------------*
 | contains the time loop                                       vg 05/07|
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::TimeLoop()
{
  // provide information about initial field (do not do for restarts!)
  if (Step()==0)
  {
    // write out initial state
    Output();

    // compute error for problems with analytical solution (initial field!)
    EvaluateErrorComparedToAnalyticalSol();
  }

  // time measurement: time loop
  TEUCHOS_FUNC_TIME_MONITOR("SCATRA:  + time loop");

  while ((step_<stepmax_) and ((time_+ EPS12) < maxtime_))
  {
    // -------------------------------------------------------------------
    // prepare time step
    // -------------------------------------------------------------------
    PrepareTimeStep();

    // -------------------------------------------------------------------
    //                  solve nonlinear / linear equation
    // -------------------------------------------------------------------
    Solve();

    // -------------------------------------------------------------------
    //                         update solution
    //        current solution becomes old solution of next timestep
    // -------------------------------------------------------------------
    Update();

    // -------------------------------------------------------------------
    // evaluate error for problems with analytical solution
    // -------------------------------------------------------------------
    EvaluateErrorComparedToAnalyticalSol();

    // -------------------------------------------------------------------
    //                         output of solution
    // -------------------------------------------------------------------
    Output();

  } // while

  // print the results of time measurements
  Teuchos::TimeMonitor::summarize();

  return;
} // ScaTraTimIntImpl::TimeLoop


/*----------------------------------------------------------------------*
 | contains the call of linear/nonlinear solver               gjb 02/10 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::Solve()
{
  // -----------------------------------------------------------------
  // intermediate solution step for homogeneous isotropic turbulence
  // -----------------------------------------------------------------
  if (solvtype_==INPAR::SCATRA::solvertype_nonlinear)
    CalcIntermediateSolution();

  // -----------------------------------------------------------------
  //                     solve (non-)linear equation
  // -----------------------------------------------------------------
  if (solvtype_==INPAR::SCATRA::solvertype_nonlinear)
    NonlinearSolve();
  else
    LinearSolve();
  //that's all

  return;
}


/*----------------------------------------------------------------------*
 | apply moving mesh data                                     gjb 05/09 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::ApplyMeshMovement(
    Teuchos::RCP<const Epetra_Vector> dispnp,
    Teuchos::RCP<DRT::Discretization> dis
)
{
  //---------------------------------------------------------------------------
  // only required in ALE case
  //---------------------------------------------------------------------------
  if (isale_)
  {
    TEUCHOS_FUNC_TIME_MONITOR("SCATRA: apply mesh movement");

    // check existence of displacement vector
    if (dispnp == Teuchos::null) dserror("Got null pointer for displacements!");

    // define error variable
    int err(0);

    // get dofrowmap of discretization
    const Epetra_Map* dofrowmap = dis->DofRowMap();

    //-------------------------------------------------------------------------
    // transfer of dofs
    // (We rely on the fact that the scatra discretization is a clone of the
    // fluid or structure mesh, respectively, meaning that a scatra node has the
    // same local (and global) ID as its corresponding fluid/structure node.)
    //-------------------------------------------------------------------------
    // loop over all local nodes of scatra discretization
    for (int lnodeid=0; lnodeid < discret_->NumMyRowNodes(); lnodeid++)
    {
      // get local fluid/structure node with the same lnodeid
      DRT::Node* lnode = dis->lRowNode(lnodeid);

      // get degrees of freedom associated with this fluid/structure node
      // (first dofset always considered, allowing for using multiple
      //  dof sets, e.g., for structure-based scalar transport)
      std::vector<int> nodedofs = dis->Dof(0,lnode);

      // determine number of space dimensions
      const int numdim = DRT::Problem::Instance()->NDim();

      for (int index=0;index < numdim; ++index)
      {
        // get global and local ID
        const int gid = nodedofs[index];
        const int lid = dofrowmap->LID(gid);

        //---------------------------------------------------------------------
        // get displacement
        //---------------------------------------------------------------------
        double disp = (*dispnp)[lid];

        // insert displacement value into node-based vector
        err = dispnp_->ReplaceMyValue(lnodeid,index,disp);
        if (err != 0) dserror("Error while inserting value into vector dispnp_!");
      }
    } // for lnodeid
  } // if (isale_)

  return;
} // ScaTraTimIntImpl::ApplyMeshMovement


/*----------------------------------------------------------------------*
 |  print information about current time step to screen        mr. x    |
 *----------------------------------------------------------------------*/
inline void SCATRA::ScaTraTimIntImpl::PrintTimeStepInfo()
{
  if (myrank_==0)
    printf("TIME: %11.4E/%11.4E  DT = %11.4E  %s  STEP = %4d/%4d \n",
           time_,maxtime_,dta_,MethodTitle().c_str(),step_,stepmax_);
} // SCATRA::ScaTraTimIntImpl::PrintTimeStepInfo


/*----------------------------------------------------------------------*
 | return system matrix downcasted as sparse matrix           gjb 02/11 |
 | implemented here to be able to use forward declaration in .H         |
 *----------------------------------------------------------------------*/
Teuchos::RCP<LINALG::SparseMatrix> SCATRA::ScaTraTimIntImpl::SystemMatrix()
{
  return Teuchos::rcp_dynamic_cast<LINALG::SparseMatrix>(sysmat_);
}


/*----------------------------------------------------------------------*
 | return system matrix downcasted as block sparse matrix     gjb 06/10 |
 | implemented here to be able to use forward declaration in .H         |
 *----------------------------------------------------------------------*/
Teuchos::RCP<LINALG::BlockSparseMatrixBase> SCATRA::ScaTraTimIntImpl::BlockSystemMatrix()
{
  return Teuchos::rcp_dynamic_cast<LINALG::BlockSparseMatrixBase>(sysmat_);
}


/*----------------------------------------------------------------------*
 | output of solution vector to BINIO                          gjb 08/08|
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::Output(const int num)
{
  // time measurement: output of solution
  TEUCHOS_FUNC_TIME_MONITOR("SCATRA:    + output of solution");

  // solution output and potentially restart data and/or flux data
  if (DoOutput())
  {
    // step number and time (only after that data output is possible)
    output_->NewStep(step_,time_);

    // write domain decomposition for visualization (only once at step "upres"!)
    if (step_==upres_) output_->WriteElementData(true);

    // write state vectors
    OutputState();

    // write output to Gmsh postprocessing files
    if (outputgmsh_) OutputToGmsh(step_, time_);

    // add restart data
    if (step_%uprestart_==0 and step_ !=0) OutputRestart();

    // write flux vector field (only writing, calculation was done during Update() call)
    if (writeflux_!=INPAR::SCATRA::flux_no)
    {
      // for flux output of initial field (before first solve) do:
      // flux_ vector is initialized when CalcFlux() is called
      if (step_==0)
        flux_=CalcFlux(true, num);

      OutputFlux(flux_);
    }

    // write mean values of scalar(s)
    OutputMeanScalars(num);

    // write integral values of reaction(s)
    OutputIntegrReac(num);

    // problem-specific outputs
    OutputProblemSpecific();

    // biofilm growth
    if (scfldgrdisp_!=Teuchos::null)
    {
      output_->WriteVector("scfld_growth_displ", scfldgrdisp_);
    }

    // biofilm growth
    if (scstrgrdisp_!=Teuchos::null)
    {
      output_->WriteVector("scstr_growth_displ", scstrgrdisp_);
    }

  }

  if ( (step_ != 0) and (output_state_matlab_) )
      {
        std::ostringstream filename;
        filename << "Result_Step" << step_ << ".m";
        LINALG::PrintVectorInMatlabFormat(filename.str(),*phinp_);
      }
  // NOTE:
  // statistics output for normal fluxes at boundaries was already done during Update()

  return;
} // ScaTraTimIntImpl::Output


/*==========================================================================*/
// scalar degrees of freedom and related
/*==========================================================================*/

/*----------------------------------------------------------------------*
 |  set initial field for phi                                 gjb 04/08 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::SetInitialField(
    const INPAR::SCATRA::InitialField init,
    const int startfuncno)
{
  switch(init)
  {
  case INPAR::SCATRA::initfield_zero_field:
  {
    phin_-> PutScalar(0.0);
    phinp_-> PutScalar(0.0);
    break;
  }
  case INPAR::SCATRA::initfield_field_by_function:
  case INPAR::SCATRA::initfield_disturbed_field_by_function:
  {
    const Epetra_Map* dofrowmap = discret_->DofRowMap();

    // loop all nodes on the processor
    for(int lnodeid=0;lnodeid<discret_->NumMyRowNodes();lnodeid++)
    {
      // get the processor local node
      DRT::Node*  lnode      = discret_->lRowNode(lnodeid);
      // the set of degrees of freedom associated with the node
      std::vector<int> nodedofset = discret_->Dof(0,lnode);

      int numdofs = nodedofset.size();
      for (int k=0;k< numdofs;++k)
      {
        const int dofgid = nodedofset[k];
        int doflid = dofrowmap->LID(dofgid);
        // evaluate component k of spatial function
        double initialval = DRT::Problem::Instance()->Funct(startfuncno-1).Evaluate(k,lnode->X(),time_,NULL);
        int err = phin_->ReplaceMyValues(1,&initialval,&doflid);
        if (err != 0) dserror("dof not on proc");
      }
    }

    // for NURBS discretizations we have to solve a least squares problem,
    // with high accuracy! (do nothing for Lagrangian polynomials)
    const Teuchos::ParameterList& scatradyn =
      DRT::Problem::Instance()->ScalarTransportDynamicParams();
    const int lstsolver = scatradyn.get<int>("LINEAR_SOLVER");
    if (lstsolver == (-1))
      dserror("no linear solver defined for least square NURBS problem. Please set LINEAR_SOLVER in SCALAR TRANSPORT DYNAMIC to a valid number! Note: this solver block is misused for the least square problem. Maybe one should add a separate parameter for this.");

    DRT::NURBS::apply_nurbs_initial_condition(
        *discret_  ,
        errfile_,
        DRT::Problem::Instance()->SolverParams(lstsolver),
        startfuncno,
        phin_     );

    // initialize also the solution vector. These values are a pretty good guess for the
    // solution after the first time step (much better than starting with a zero vector)
    phinp_->Update(1.0,*phin_ ,0.0);

    // add random perturbation for initial field of turbulent flows
    if(init==INPAR::SCATRA::initfield_disturbed_field_by_function)
    {
      int err = 0;

      // random noise is relative to difference of max-min values of initial profile
      double perc = extraparams_->sublist("TURBULENCE MODEL").get<double>("CHAN_AMPL_INIT_DIST",0.1);

      // out to screen
      if (myrank_==0)
      {
        std::cout << "Disturbed initial scalar profile:   max. " << perc*100 << "% random perturbation\n";
        std::cout << "\n\n";
      }

      // get overall max and min values and range between min and max
      double maxphi(0.0);
      double minphi(0.0);
      err = phinp_->MaxValue(&maxphi);
      if (err > 0) dserror("Error during evaluation of maximum value.");
      err = phinp_->MinValue(&minphi);
      if (err > 0) dserror("Error during evaluation of minimum value.");
      double range = abs(maxphi - minphi);

      // disturb initial field for all degrees of freedom
      for (int k=0; k < phinp_->MyLength(); ++k)
      {
        double randomnumber = DRT::Problem::Instance()->Random()->Uni();
        double noise = perc * range * randomnumber;
        err += phinp_->SumIntoMyValues(1,&noise,&k);
        err += phin_ ->SumIntoMyValues(1,&noise,&k);
        if (err!=0) dserror("Error while disturbing initial field.");
      }
    }
    break;
  }
  case INPAR::SCATRA::initfield_field_by_condition:
  {
    // set initial field for ALL existing scatra fields
    const std::string field = "ScaTra";
    std::vector<int> localdofs(numdofpernode_);

    for (int i = 0; i < numdofpernode_; i++)
    {
      localdofs[i] = i;
    }
    discret_->EvaluateInitialField(field,phin_,localdofs);

    // initialize also the solution vector. These values are a pretty good guess for the
    // solution after the first time step (much better than starting with a zero vector)
    phinp_->Update(1.0,*phin_ ,0.0);

    break;
  }
  // discontinuous 0-1 field for progress variable in 1-D
  case INPAR::SCATRA::initfield_discontprogvar_1D:
  {
    const Epetra_Map* dofrowmap = discret_->DofRowMap();

    // loop all nodes on the processor
    for(int lnodeid=0;lnodeid<discret_->NumMyRowNodes();lnodeid++)
    {
      // get the processor local node
      DRT::Node*  lnode      = discret_->lRowNode(lnodeid);
      // the set of degrees of freedom associated with the node
      std::vector<int> nodedofset = discret_->Dof(lnode);

      // get coordinate
      const double x = lnode->X()[0];

      int numdofs = nodedofset.size();
      for (int k=0;k< numdofs;++k)
      {
        const int dofgid = nodedofset[k];
        int doflid = dofrowmap->LID(dofgid);

        double initialval = 0.0;
        if (x > -EPS10) initialval = 1.0;

        int err = 0;
        err += phin_->ReplaceMyValues(1,&initialval,&doflid);
        // initialize also the solution vector. These values are a pretty good guess for the
        // solution after the first time step (much better than starting with a zero vector)
        err += phinp_->ReplaceMyValues(1,&initialval,&doflid);
        if (err != 0) dserror("dof not on proc");
      }
    }
    break;
  }
  // reconstructed initial profile for progress variable in x2-direction from
  // Lessani and Papalexandris (2006), also used in Moureau et al. (2007, 2009),
  // for two-dimensional flame-vortex interaction problem (x2=0-200)
  case INPAR::SCATRA::initfield_flame_vortex_interaction:
  {
    // locations separating region 1 from region 2 and region 2 from region 3
    const double loc12 = 98.5;
    const double loc23 = 103.0;

    // define parameters for region 1 (exponential function for curve fitting)
    const double beta1  = 1.65;
    const double delta1 = 1.0;
    const double trans1 = 100.0;

    // define parameters for region 2 (linear function for curve fitting)
    const double abs2 = 0.0879;
    const double fac2 = 0.139309333;
    const double trans2 = 98.5;

    // define parameters for region 3 (exponential function for curve fitting)
    const double beta3  = 3.506209;
    const double delta3 = 4.28875;
    const double trans3 = 103.0;

    const Epetra_Map* dofrowmap = discret_->DofRowMap();

    // define variable
    double initialval = 0.0;

    // loop all nodes on the processor
    for(int lnodeid=0;lnodeid<discret_->NumMyRowNodes();lnodeid++)
    {
      // get the processor local node
      DRT::Node*  lnode      = discret_->lRowNode(lnodeid);
      // the set of degrees of freedom associated with the node
      std::vector<int> nodedofset = discret_->Dof(lnode);

      // get x2-coordinate
      const double x2 = lnode->X()[1];

      int numdofs = nodedofset.size();
      for (int k=0;k< numdofs;++k)
      {
        const int dofgid = nodedofset[k];
        int doflid = dofrowmap->LID(dofgid);

        if (x2 < loc12-EPS10)
          initialval = (1.0-(1.0/beta1))*exp((x2-trans1)/delta1);
        else if (x2 > loc23+EPS10)
          initialval = 1.0-(exp((1.0-beta3)*(x2-trans3)/delta3)/beta3);
        else
          initialval = fac2*(x2-trans2) + abs2;

        int err = 0;
        err += phin_->ReplaceMyValues(1,&initialval,&doflid);
        // initialize also the solution vector. These values are a pretty good guess for the
        // solution after the first time step (much better than starting with a zero vector)
        err += phinp_->ReplaceMyValues(1,&initialval,&doflid);
        if (err != 0) dserror("dof not on proc");
      }
    }
    break;
  }
  // initial mixture-fraction profile for Rayleigh-Taylor instability
  case INPAR::SCATRA::initfield_raytaymixfrac:
  {
    // define interface thickness, sinusoidal disturbance wave amplitude and pi
    const double delta = 0.002;
    const double alpha = 0.001;

    const Epetra_Map* dofrowmap = discret_->DofRowMap();

    // loop all nodes on the processor
    for(int lnodeid=0;lnodeid<discret_->NumMyRowNodes();lnodeid++)
    {
      // get the processor local node
      DRT::Node*  lnode      = discret_->lRowNode(lnodeid);
      // the set of degrees of freedom associated with the node
      std::vector<int> nodedofset = discret_->Dof(lnode);

      // get x1- and x2-coordinate
      const double x1 = lnode->X()[0];
      const double x2 = lnode->X()[1];

      // interface disturbance
      //double x2_int = 0.05*cos(pi*(x1+0.5));
      //double x2_int = 0.05*cos(2.0*pi*x1);
      double x2_int = 0.0;
      x2_int -= cos(4*M_PI*x1);
      x2_int -= cos(14*M_PI*x1);
      x2_int -= cos(23*M_PI*x1);
      x2_int -= cos(28*M_PI*x1);
      x2_int -= cos(33*M_PI*x1);
      x2_int -= cos(42*M_PI*x1);
      x2_int -= cos(51*M_PI*x1);
      x2_int -= cos(59*M_PI*x1);
      x2_int *= alpha;

      const double value = (x2_int-x2)/(2.0*delta);

      // values required for tanh-distribution
      const double vp = exp(value);
      const double vm = exp(-value);

      int numdofs = nodedofset.size();
      for (int k=0;k< numdofs;++k)
      {
        const int dofgid = nodedofset[k];
        int doflid = dofrowmap->LID(dofgid);

        // compute tanh-distribution
        double initialval = 0.0;
        initialval = 0.5*(1.0+(vp-vm)/(vp+vm));

        int err = 0;
        err += phin_->ReplaceMyValues(1,&initialval,&doflid);
        // initialize also the solution vector. These values are a pretty good guess for the
        // solution after the first time step (much better than starting with a zero vector)
        err += phinp_->ReplaceMyValues(1,&initialval,&doflid);
        if (err != 0) dserror("dof not on proc");
      }
    }
    break;
  }
  // initial field for skew convection of L-shaped domain
  case INPAR::SCATRA::initfield_Lshapeddomain:
  {
    const Epetra_Map* dofrowmap = discret_->DofRowMap();

    // loop all nodes on the processor
    for(int lnodeid=0;lnodeid<discret_->NumMyRowNodes();lnodeid++)
    {
      // get the processor local node
      DRT::Node*  lnode      = discret_->lRowNode(lnodeid);
      // the set of degrees of freedom associated with the node
      std::vector<int> nodedofset = discret_->Dof(lnode);

      // get x1- and x2-coordinate
      const double x1 = lnode->X()[0];
      const double x2 = lnode->X()[1];

      int numdofs = nodedofset.size();
      for (int k=0;k< numdofs;++k)
      {
        const int dofgid = nodedofset[k];
        int doflid = dofrowmap->LID(dofgid);

        // compute initial values 0.0 or 1.0 depending on geometrical location
        double initialval = 0.0;
        if ((x1 <= 0.25 and x2 <= 0.5) or (x1 <= 0.5 and x2 <= 0.25))
          initialval = 1.0;

        int err = 0;
        err += phin_->ReplaceMyValues(1,&initialval,&doflid);
        // initialize also the solution vector. These values are a pretty good
        // guess for the solution after the first time step (much better than
        // starting with a zero vector)
        err += phinp_->ReplaceMyValues(1,&initialval,&doflid);
        if (err != 0) dserror("dof not on proc");
      }
    }
    break;
  }
  case INPAR::SCATRA::initfield_facing_flame_fronts:
  {
    const Epetra_Map* dofrowmap = discret_->DofRowMap();

    // loop all nodes on the processor
    for(int lnodeid=0;lnodeid<discret_->NumMyRowNodes();lnodeid++)
    {
      // get the processor local node
      DRT::Node*  lnode      = discret_->lRowNode(lnodeid);
      // the set of degrees of freedom associated with the node
      std::vector<int> nodedofset = discret_->Dof(lnode);

      // get x1- and x2-coordinate
      const double x1 = lnode->X()[0];
      //const double x2 = lnode->X()[1];

      int numdofs = nodedofset.size();
      for (int k=0;k< numdofs;++k)
      {
        const int dofgid = nodedofset[k];
        int doflid = dofrowmap->LID(dofgid);
        // evaluate component k of spatial function

        double initialval;
        if (x1 < 0.0) initialval = -(x1+0.75);
        else initialval = x1-0.75;

        int err = 0;
        err += phin_->ReplaceMyValues(1,&initialval,&doflid);
        // initialize also the solution vector. These values are a pretty good guess for the
        // solution after the first time step (much better than starting with a zero vector)
        err += phinp_->ReplaceMyValues(1,&initialval,&doflid);
        if (err != 0) dserror("dof not on proc");
      }
    }
    break;
  }
  case INPAR::SCATRA::initfield_oracles_flame:
  {
    const Epetra_Map* dofrowmap = discret_->DofRowMap();

    const double eps = 0.00152;
    //const double xsing = 0.2;
    //const double zsing = 0.7525-0.05;//0.0354;

    // loop all nodes on the processor
    for(int lnodeid=0;lnodeid<discret_->NumMyRowNodes();lnodeid++)
    {
      // get the processor local node
      DRT::Node*  lnode      = discret_->lRowNode(lnodeid);
      // the set of degrees of freedom associated with the node
      std::vector<int> nodedofset = discret_->Dof(lnode);

      // get x1, x2 and x3-coordinate
      //const double x1 = lnode->X()[0];
      const double x2 = lnode->X()[1];
      //const double x3 = lnode->X()[2];

      int numdofs = nodedofset.size();
      for (int k=0;k< numdofs;++k)
      {
        const int dofgid = nodedofset[k];
        int doflid = dofrowmap->LID(dofgid);
        // evaluate component k of spatial function

        double initval = 0.0;

        // initial plane implementation for periodic spanwise boundary
        if (x2 >= 0.0)
          initval = (x2-0.0354) - eps;
        else
          initval = (-0.0354-x2) - eps;

#if 0
        // initial wedge implementation for periodic spanwise boundary
        if (x1 <= 0.0)
        {
          if (x2 >= 0.0)
            initval = (x2-0.0354) - eps;
          else
            initval = (-0.0354-x2) - eps;
        }
        else if (x1 > 0.0 and x1 < xsing)
        {
          initval = abs(x2)-0.0354*(xsing-x1)/xsing - eps;
        }
        else if (x1 >= xsing)
          initval = x1 - xsing - eps;
        else
          dserror("impossible!");
#endif

#if 0
        // initial wedge implementation for spanwise walls
        if (x1 <= 0.0)
        {
          if ( x3 <= -zsing and abs(x2) <= abs(x3+zsing) )
          {
            initval = (-0.7525-x3) - eps;
          }
          else if ( x3 >= zsing and abs(x2) <= (x3-zsing) )
          {
            initval = (x3-0.7525) - eps;
          }
          else if ( x2 >= 0.0 and ( x2 > abs(x3+zsing) or x2 > (x3-zsing) ))
          {
            initval = (x2-0.0354) - eps;
          }
          else if ( x2 < 0.0 and (-x2 > abs(x3+zsing) or -x2 > (x3-zsing) ))
          {
            initval = (-0.0354-x2) - eps;
          }
          else
            dserror("coordinate out of range of ORACLES initial function");
        }
        else if (x1 > 0.0 and x1 < xsing)
        {
          if (abs(x3) <= 0.07)
            initval = abs(x2)-0.0354*(xsing-x1)/xsing - eps;
          else
          {
            initval = 0.07525-0.07;
          }
        }
        else if (x1 >= xsing)
          initval = x1 - xsing - eps;
        else
          dserror("impossible!");
#endif
        int err = 0;
        err += phin_->ReplaceMyValues(1,&initval,&doflid);
        // initialize also the solution vector. These values are a pretty good guess for the
        // solution after the first time step (much better than starting with a zero vector)
        err += phinp_->ReplaceMyValues(1,&initval,&doflid);
        if (err != 0) dserror("dof not on proc");
      }
    }
  break;
  }
  case INPAR::SCATRA::initialfield_forced_hit_high_Sc:
  case INPAR::SCATRA::initialfield_forced_hit_low_Sc:
  {
    // initialize calculation of initial field based on fast Fourier transformation
    Teuchos::RCP<HomIsoTurbInitialScalarField> HitInitialScalarField = Teuchos::rcp(new SCATRA::HomIsoTurbInitialScalarField(*this,init));
    // calculate initial field
    HitInitialScalarField->CalculateInitialField();

    break;
  }
  default:
    dserror("Unknown option for initial field: %d", init); break;
  } // switch(init)

  return;
} // ScaTraTimIntImpl::SetInitialField


/*----------------------------------------------------------------------*
 | iterative update of concentrations                                   |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::UpdateIter(const Teuchos::RCP<const Epetra_Vector> inc)
{
  // store incremental vector to be available for convergence check
  // if incremental vector is received from outside for coupled problem
  increment_->Update(1.0,*inc,0.0);

  // update scalar values by adding increments
  phinp_->Update(1.0,*inc,1.0);
} // UpdateIter


/*==========================================================================*
 |                                                                          |
 | protected:                                                               |
 |                                                                          |
 *==========================================================================*/

/*==========================================================================*/
// general framework
/*==========================================================================*/

/*--------------------------------------------------------------------------*
 | setup Krylov projector including first fill                    nis Feb13 |
 *--------------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::SetupKrylovSpaceProjection(DRT::Condition* kspcond)
{
  // previously, scatra was able to define actual modes that formed a
  // nullspace. factors when assigned to scalars in the dat file. it could
  // take several scatra Krylov conditions each forming one mode like:
  // 3.0*c_1 + 2.0*c_2 = const
  // since this was never used, not even in ELCH-problems, and for the sake of
  // consistency, now only a singel scatra Krylov condition can be given, with
  // flags that triggers the scalars that are to be levelled by a projection
  // (like the pressure in a pure Dirichlet fluid problem).
  // furthermore, this is a step towards the ability to have projection on
  // more than one field.
  // to see the handling of different modes, the ability that is now lost, see
  // revision 17615.

  // confirm that mode flags are number of nodal dofs/scalars
  const int nummodes = kspcond->GetInt("NUMMODES");
  if (nummodes!=numdofpernode_)
    dserror("Expecting as many mode flags as nodal dofs in Krylov projection definition. Check dat-file!");

  // get vector of mode flags as given in dat-file
  const std::vector<int>* modeflags = kspcond->Get<std::vector<int> >("ONOFF");

  // count actual active modes selected in dat-file
  std::vector<int> activemodeids;
  for(int rr=0;rr<numdofpernode_;++rr)
  {
    if(((*modeflags)[rr])!=0)
    {
      activemodeids.push_back(rr);
    }
  }

  // get from dat-file definition how weights are to be computed
  const std::string* weighttype = kspcond->Get<std::string>("weight vector definition");

  // set flag for projection update true only if ALE and integral weights
  if (isale_ and (*weighttype=="integration"))
    updateprojection_ = true;

  // create the projector
  projector_ = Teuchos::rcp(new LINALG::KrylovProjector(activemodeids,weighttype,discret_->DofRowMap()));

  // update the projector
  UpdateKrylovSpaceProjection();
} // SCATRA::ScaTraTimIntImpl::SetupKrylovSpaceProjection


/*--------------------------------------------------------------------------*
 | update projection vectors w_ and c_ for Krylov projection      nis Feb13 |
 *--------------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::UpdateKrylovSpaceProjection()
{
  // loop over modes to create vectors within multi-vector
  // one could tailor a MapExtractor to extract necessary dofs, however this
  // seems to be an overkill, since normally only a single scalar is
  // projected. for only a second projected scalar it seems worthwhile. feel
  // free! :)

  // get Teuchos::RCP to kernel vector of projector
  Teuchos::RCP<Epetra_MultiVector> c = projector_->GetNonConstKernel();
  c->PutScalar(0.0);

  const std::string* weighttype = projector_->WeightType();
  // compute w_ as defined in dat-file
  if(*weighttype == "pointvalues")
  {
    dserror("option pointvalues not implemented");
  }
  else if(*weighttype == "integration")
  {
    // get Teuchos::RCP to weight vector of projector
    Teuchos::RCP<Epetra_MultiVector> w = projector_->GetNonConstWeights();
    w->PutScalar(0.0);

    // get number of modes and their ids
    int nummodes = projector_->Nsdim();
    std::vector<int> modeids = projector_->Modes();

    // initialize dofid vector to -1
    Epetra_IntSerialDenseVector dofids(numdofpernode_);
    for (int rr=0;rr<numdofpernode_;++rr)
    {
      dofids[rr] = -1;
    }

    Teuchos::ParameterList mode_params;

    // set parameters for elements that do not change over mode
    mode_params.set<int>("action",SCATRA::integrate_shape_functions);
    if (isale_)
      discret_->AddMultiVectorToParameterList(mode_params,"dispnp",dispnp_);

    // loop over all activemodes
    for (int imode = 0; imode < nummodes; ++imode)
    {
      // activate dof of current mode and add dofids to parameter list
      dofids[modeids[imode]] = 1;
      mode_params.set("dofids",dofids);

      /*
      // evaluate KrylovSpaceProjection condition in order to get
      // integrated nodal basis functions w_
      // Note that in the case of definition integration based, the average
      // increment of the scalar quantity c will vanish in an integral sense
      //
      //                    /              /                      /
      //   /    \          |              |  /          \        |  /    \
      //  | w_*c | = c_i * | N_i(x) dx =  | | N_i(x)*c_i | dx =  | | c(x) | dx = 0
      //   \    /          |              |  \          /        |  \    /
      //                   /              /                      /
      */

      // get an Teuchos::RCP of the current column Epetra_Vector of the MultiVector
      Teuchos::RCP<Epetra_Vector> wi = Teuchos::rcp((*w)(imode),false);

      // compute integral of shape functions
      discret_->EvaluateCondition
          (mode_params           ,
          Teuchos::null      ,
          Teuchos::null      ,
          wi                 ,
          Teuchos::null      ,
          Teuchos::null      ,
          "KrylovSpaceProjection");

      // deactivate dof of current mode
      dofids[modeids[imode]] = -1;

      // set the current kernel basis vector - not very nice
      for (int inode = 0; inode < discret_->NumMyRowNodes(); inode++)
      {
        DRT::Node* node = discret_->lRowNode(inode);
        std::vector<int> gdof = discret_->Dof(node);
        int err = c->ReplaceGlobalValue(gdof[modeids[imode]],imode,1);
        if (err != 0) dserror("error while inserting value into c");
      }

    } // loop over modes

    // adapt weight vector according to meshtying case
    if (msht_ != INPAR::FLUID::no_meshtying)
    {
      dserror("Since meshtying for scatra is not tested under Krylov projection dserror is introduced. Remove at own responsibility.");
      //meshtying_->AdaptKrylovProjector(w);
    }

  } // endif integration
  else
  {
    dserror("unknown definition of weight vector w for restriction of Krylov space");
  }

  // adapt kernel vector according to meshtying case
  if (msht_ != INPAR::FLUID::no_meshtying)
  {
    dserror("Since meshtying for scatra is not tested under Krylov projection dserror is introduced. Remove at own responsibility.");
    //meshtying_->AdaptKrylovProjector(c);
  }

  // fillcomplete the projector to compute (w^T c)^(-1)
  projector_->FillComplete();

  return;
} // ScaTraTimIntImpl::UpdateKrylovSpaceProjection


/*----------------------------------------------------------------------------------------*
 | initialize meshtying strategy (including standard case without meshtying)   fang 12/14 |
 *----------------------------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::CreateMeshtyingStrategy()
{
  // fluid meshtying
  if(msht_ != INPAR::FLUID::no_meshtying)
    strategy_ = Teuchos::rcp(new MeshtyingStrategyFluid(this));

  // scatra-scatra interface coupling
  else if(s2icoupling_)
    strategy_ = Teuchos::rcp(new MeshtyingStrategyS2I(this));

  // standard case without meshtying
  else
    strategy_ = Teuchos::rcp(new MeshtyingStrategyStd(this));

  return;
} // ScaTraTimIntImpl::UpdateKrylovSpaceProjection


/*----------------------------------------------------------------------*
 | add approximation to flux vectors to a parameter list      gjb 05/10 |
 *----------------------------------------------------------------------*/
// void SCATRA::ScaTraTimIntImpl::AddFluxApproxToParameterList
// defined in scalar_timint_implicit_service.cpp

/*----------------------------------------------------------------------*
 | application of Dirichlet boundary conditions                         |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::ApplyDirichletToSystem()
{
  // -------------------------------------------------------------------
  // Apply Dirichlet boundary conditions to system matrix
  // -------------------------------------------------------------------
  if (incremental_)
  {
    // blank residual DOFs which are on Dirichlet BC
    // We can do this because the values at the Dirichlet positions
    // are not used anyway.
    // We could avoid this though, if the dofrowmap would not include
    // the Dirichlet values as well. But it is expensive to avoid that.
    dbcmaps_->InsertCondVector(dbcmaps_->ExtractCondVector(zeros_), residual_);

    //--------- Apply Dirichlet boundary conditions to system of equations
    // residual values are supposed to be zero at Dirichlet boundaries
    increment_->PutScalar(0.0);

    {
      // time measurement: application of DBC to system
      TEUCHOS_FUNC_TIME_MONITOR("SCATRA:       + apply DBC to system");

      LINALG::ApplyDirichlettoSystem(sysmat_,increment_,residual_,zeros_,*(dbcmaps_->CondMap()));
    }
  }
  else
  {
    // time measurement: application of DBC to system
    TEUCHOS_FUNC_TIME_MONITOR("SCATRA:       + apply DBC to system");

    LINALG::ApplyDirichlettoSystem(sysmat_,phinp_,residual_,phinp_,*(dbcmaps_->CondMap()));
  }
  return;
} // SCATRA::ScaTraTimIntImpl::ApplyDirichletToSystem


/*----------------------------------------------------------------------*
 | evaluate Dirichlet boundary conditions at t_{n+1}           gjb 07/08|
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::ApplyDirichletBC
(
  const double time,
  Teuchos::RCP<Epetra_Vector> phinp,
  Teuchos::RCP<Epetra_Vector> phidt
)
{
  // time measurement: apply Dirichlet conditions
  TEUCHOS_FUNC_TIME_MONITOR("SCATRA:      + apply dirich cond.");

  // Todo: what happens in  the case of generalized alpha
  // needed parameters
  Teuchos::ParameterList p;
  p.set("total time",time);  // actual time t_{n+1}

  // predicted Dirichlet values
  // \c  phinp then also holds prescribed new Dirichlet values
  discret_->ClearState();
  discret_->EvaluateDirichlet(p,phinp,phidt,Teuchos::null,Teuchos::null,dbcmaps_);
  discret_->ClearState();

  return;
} // SCATRA::ScaTraTimIntImpl::ApplyDirichletBC


/*----------------------------------------------------------------------*
 | contains the residual scaling and addition of Neumann terms vg 08/09 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::ScalingAndNeumann()
{
  // scaling to get true residual vector for all time integration schemes
  // in incremental case: boundary flux values can be computed from trueresidual
  if (incremental_) trueresidual_->Update(ResidualScaling(),*residual_,0.0);

  // add Neumann b.c. scaled with a factor due to time discretization
  AddNeumannToResidual();

  // add potential Neumann inflow or convective heat transfer boundary
  // conditions (simultaneous evaluation of both conditions not allowed!)
  if (neumanninflow_)     ComputeNeumannInflow(sysmat_,residual_);
  else if (convheatrans_) EvaluateConvectiveHeatTransfer(sysmat_,residual_);

  return;
} // ScaTraTimIntImpl::ScalingAndNeumann


/*----------------------------------------------------------------------*
 | evaluate Neumann boundary conditions                      fang 01/15 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::ApplyNeumannBC
(
    const Teuchos::RCP<Epetra_Vector>&   neumann_loads //!< Neumann loads
)
{
  // prepare load vector
  neumann_loads->PutScalar(0.0);

  // create parameter list
  Teuchos::ParameterList condparams;

  // action for elements
  condparams.set<int>("action",SCATRA::bd_calc_Neumann);

  // specific parameters
  AddProblemSpecificParametersAndVectors(condparams);

  // set time for evaluation of point Neumann conditions as parameter depending on time integration scheme
  // line/surface/volume Neumann conditions use the time stored in the time parameter class
  SetTimeForNeumannEvaluation(condparams);

  // provide displacement field in case of ALE
  if (isale_) discret_->AddMultiVectorToParameterList(condparams,"dispnp",dispnp_);

  // evaluate Neumann boundary conditions at time t_{n+alpha_F} (generalized alpha) or time t_{n+1} (otherwise)
  discret_->EvaluateNeumann(condparams,*neumann_loads);
  discret_->ClearState();

  return;
} // SCATRA::ScaTraTimIntImpl::ApplyNeumannBC


/*----------------------------------------------------------------------------*
 | evaluate solution-depending boundary and interface conditions   fang 10/14 |
 *----------------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::EvaluateSolutionDependingConditions(
    Teuchos::RCP<LINALG::SparseOperator> systemmatrix,      //!< system matrix
    Teuchos::RCP<Epetra_Vector>          rhs                //!< rhs vector
)
{
  // evaluate meshtying
  strategy_->EvaluateMeshtying();

  // evaluate Robin type boundary condition (needed for photoacoustic simulations)
  EvaluateRobinBoundaryConditions(systemmatrix,rhs);

  return;
} // ScaTraTimIntImpl::EvaluateSolutionDependingConditions


/*----------------------------------------------------------------------------*
 | evaluate Robin boundary conditions                          schoeder 03/15 |
 *----------------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::EvaluateRobinBoundaryConditions(
    Teuchos::RCP<LINALG::SparseOperator> matrix,      //!< system matrix
    Teuchos::RCP<Epetra_Vector>          rhs                //!< rhs vector
)
{
  discret_->ClearState();

  // create parameter list
  Teuchos::ParameterList condparams;

  // action for elements
  condparams.set<int>("action",SCATRA::bd_calc_Robin);

  // add element parameters and set state vectors according to time-integration scheme
  AddTimeIntegrationSpecificVectors();

  // evaluate ElchBoundaryKinetics conditions at time t_{n+1} or t_{n+alpha_F}
  discret_->EvaluateCondition(condparams,matrix,Teuchos::null,rhs,Teuchos::null,Teuchos::null,"ScatraRobin");
  discret_->ClearState();

  return;
} // ScaTraTimIntImpl::EvaluateRobinBoundaryConditions


/*----------------------------------------------------------------------*
 | contains the assembly process for matrix and rhs            vg 08/09 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::AssembleMatAndRHS()
{
  // time measurement: element calls
  TEUCHOS_FUNC_TIME_MONITOR("SCATRA:       + element calls");

  // get cpu time
  const double tcpuele = Teuchos::Time::wallTime();

  // zero out matrix entries
  sysmat_->Zero();

  // reset the residual vector
  residual_->PutScalar(0.0);

  // create parameter list for elements
  Teuchos::ParameterList eleparams;

  // action for elements
  eleparams.set<int>("action",SCATRA::calc_mat_and_rhs);

  // DO THIS AT VERY FIRST!!!
  // compute reconstructed diffusive fluxes for better consistency
  const enum INPAR::SCATRA::Consistency consistency
  = DRT::INPUT::IntegralValue<INPAR::SCATRA::Consistency>(params_->sublist("STABILIZATION"),"CONSISTENCY");
  if (consistency == INPAR::SCATRA::consistency_l2_projection_lumped)
  {
    // compute flux approximation and add it to the parameter list
    AddFluxApproxToParameterList(eleparams,INPAR::SCATRA::flux_diffusive_domain);
  }

  // prepare dynamic Smagorinsky model if required,
  // i.e. calculate turbulent Prandtl number
  if (timealgo_ != INPAR::SCATRA::timeint_stationary)
  {
    DynamicComputationOfCs();
    DynamicComputationOfCv();
  }
  // this parameter list is required here to get the element-based filtered constants
  eleparams.sublist("TURBULENCE MODEL") = extraparams_->sublist("TURBULENCE MODEL");

  // provide velocity field and potentially acceleration/pressure field
  // (export to column map necessary for parallel evaluation)
  discret_->AddMultiVectorToParameterList(eleparams,"convective velocity field",convel_);
  discret_->AddMultiVectorToParameterList(eleparams,"velocity field",vel_);
  discret_->AddMultiVectorToParameterList(eleparams,"acceleration field",acc_);
  discret_->AddMultiVectorToParameterList(eleparams,"pressure field",pre_);
  // and provide fine-scale velocity for multifractal subgrid-scale modeling only
  if(turbmodel_==INPAR::FLUID::multifractal_subgrid_scales or fssgd_ == INPAR::SCATRA::fssugrdiff_smagorinsky_small)
    discret_->AddMultiVectorToParameterList(eleparams,"fine-scale velocity field",fsvel_);

  // provide displacement field in case of ALE
  if(isale_)
    discret_->AddMultiVectorToParameterList(eleparams,"dispnp",dispnp_);

  // set vector values needed by elements
  discret_->ClearState();

  // AVM3 separation for incremental solver: get fine-scale part of scalar
  if(incremental_ and step_ > 0 and (fssgd_ != INPAR::SCATRA::fssugrdiff_no or turbmodel_ == INPAR::FLUID::multifractal_subgrid_scales))
    AVM3Separation();

  // add state vectors according to time-integration scheme
  AddTimeIntegrationSpecificVectors();

  // set external volume force (required, e.g., for forced homogeneous isotropic turbulence)
  if (homisoturb_forcing_ != Teuchos::null)
    homisoturb_forcing_->UpdateForcing(step_);

  if (forcing_!=Teuchos::null)
    discret_->SetState("forcing",forcing_);

  // add problem specific time-integration parameters
  AddProblemSpecificParametersAndVectors(eleparams);

  // call loop over elements (with or without subgrid-diffusivity(-scaling) vector)
  if (fssgd_ != INPAR::SCATRA::fssugrdiff_no)
    discret_->Evaluate(eleparams,sysmat_,Teuchos::null,residual_,subgrdiff_,Teuchos::null);
  else
    discret_->Evaluate(eleparams,sysmat_,residual_);

//  (SystemMatrix()->EpetraMatrix())->Print(std::cout); // kn nis

  discret_->ClearState();

  //----------------------------------------------------------------------
  // apply weak Dirichlet boundary conditions
  //----------------------------------------------------------------------
  {
    Teuchos::ParameterList mhdbcparams;

    // set action for elements
    mhdbcparams.set<int>("action",SCATRA::bd_calc_weak_Dirichlet);

    discret_->AddMultiVectorToParameterList(mhdbcparams,"convective velocity field",convel_);
    discret_->AddMultiVectorToParameterList(mhdbcparams,"velocity field",vel_);
    AddTimeIntegrationSpecificVectors();

    // evaluate all mixed hybrid Dirichlet boundary conditions
    discret_->EvaluateCondition
      (mhdbcparams          ,
       sysmat_              ,
       Teuchos::null        ,
       residual_            ,
       Teuchos::null        ,
       Teuchos::null        ,
       "LineWeakDirichlet");

    discret_->EvaluateCondition
      (mhdbcparams          ,
       sysmat_              ,
       Teuchos::null        ,
       residual_            ,
       Teuchos::null        ,
       Teuchos::null        ,
       "SurfaceWeakDirichlet");

    // clear state
    discret_->ClearState();
  }

  // AVM3 scaling for non-incremental solver: scaling of normalized AVM3-based
  // fine-scale subgrid-diffusivity matrix by subgrid diffusivity
  if (not incremental_ and fssgd_ != INPAR::SCATRA::fssugrdiff_no)
    AVM3Scaling(eleparams);

  // potential residual scaling and potential addition of Neumann terms
  ScalingAndNeumann();

  // evaluate solution-depending boundary and interface conditions
  EvaluateSolutionDependingConditions(sysmat_,residual_);

  // finalize assembly of system matrix
  sysmat_->Complete();

  // end time measurement for element
  dtele_=Teuchos::Time::wallTime()-tcpuele;

  return;
} // ScaTraTimIntImpl::AssembleMatAndRHS


/*----------------------------------------------------------------------*
 | contains the linear solver                                  vg 08/09 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::LinearSolve()
{
  // -------------------------------------------------------------------
  //                        output to screen
  // -------------------------------------------------------------------
  PrintTimeStepInfo();

  // -------------------------------------------------------------------
  //                     preparations for solve
  // -------------------------------------------------------------------
  PrepareLinearSolve();

  // -------------------------------------------------------------------
  // Solve system in incremental or non-incremental case
  // -------------------------------------------------------------------
  if (incremental_)
  {
    TEUCHOS_FUNC_TIME_MONITOR("SCATRA:       + solver calls");

    // get cpu time
    const double tcpusolve=Teuchos::Time::wallTime();

    strategy_->Solve(solver_,sysmat_,increment_,residual_,phinp_,1,Teuchos::null);

    // end time measurement for solver
    dtsolve_=Teuchos::Time::wallTime()-tcpusolve;

    //------------------------------------------------ update solution vector
    UpdateIter(increment_);

    //--------------------------------------------- compute norm of increment
    double incnorm_L2(0.0);
    double scalnorm_L2(0.0);
    increment_->Norm2(&incnorm_L2);
    phinp_    ->Norm2(&scalnorm_L2);

    if (myrank_ == 0)
    {
      printf("+-------------------------------+-------------+\n");
      {
        if (scalnorm_L2 > EPS10)
          printf("|  relative increment (L2 norm) | %10.3E  |",incnorm_L2/scalnorm_L2);
        else // prevent division by an almost zero value
          printf("|  absolute increment (L2 norm) | %10.3E  |\n",incnorm_L2);
      }
      printf(" (ts=%10.3E,te=%10.3E)\n",dtsolve_,dtele_);
      printf("+-------------------------------+-------------+\n");
    }
  }
  else
  {
    TEUCHOS_FUNC_TIME_MONITOR("SCATRA:       + solver calls");

    // get cpu time
    const double tcpusolve=Teuchos::Time::wallTime();

    strategy_->Solve(solver_,sysmat_,phinp_,residual_,phinp_,1,Teuchos::null);

    // end time measurement for solver
    dtsolve_=Teuchos::Time::wallTime()-tcpusolve;

    if (myrank_==0)
      printf("Solvertype linear_full (ts=%10.3E,te=%10.3E)\n",dtsolve_,dtele_);
  }

  // -------------------------------------------------------------------
  // compute values at intermediate time steps (only for gen.-alpha)
  // -------------------------------------------------------------------
  ComputeIntermediateValues();

  return;
} // ScaTraTimIntImpl::LinearSolve


/*----------------------------------------------------------------------*
 | contains the nonlinear iteration loop                       gjb 09/08|
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::NonlinearSolve()
{
  // time measurement: nonlinear iteration
  TEUCHOS_FUNC_TIME_MONITOR("SCATRA:   + nonlin. iteration/lin. solve");

  // out to screen
  PrintTimeStepInfo();

  // special preparations for multifractal subgrid-scale model
  if (turbmodel_==INPAR::FLUID::multifractal_subgrid_scales)
    RecomputeMeanCsgsB();

  // print header of convergence table to screen
  PrintConvergenceHeader();

  // ---------------------------------------------- nonlinear iteration
  // stop nonlinear iteration when increment-norm is below this bound
  const double  ittol = params_->sublist("NONLINEAR").get<double>("CONVTOL");

  //------------------------------ turn adaptive solver tolerance on/off
  const bool   isadapttol    = (DRT::INPUT::IntegralValue<int>(params_->sublist("NONLINEAR"),"ADAPTCONV"));
  const double adaptolbetter = params_->sublist("NONLINEAR").get<double>("ADAPTCONV_BETTER");
  const double abstolres = params_->sublist("NONLINEAR").get<double>("ABSTOLRES");
  double       actresidual(0.0);

  // prepare Newton-Raphson iteration
  iternum_ = 0;
  int itemax = params_->sublist("NONLINEAR").get<int>("ITEMAX");

  // perform explicit predictor step (-> better starting point for nonlinear solver)
  const bool explpredictor = (DRT::INPUT::IntegralValue<int>(params_->sublist("NONLINEAR"),"EXPLPREDICT") == 1);
  if (explpredictor)
    ExplicitPredictor();

  // start Newton-Raphson iteration
  while(true)
  {
    iternum_++;

    // check for negative/zero concentration values (in case of ELCH only)
    CheckConcentrationValues(phinp_);

    // call elements to calculate system matrix and rhs and assemble
    AssembleMatAndRHS();

    // perform finite difference check on time integrator level
    if(fdcheck_ == INPAR::SCATRA::fdcheck_global)
      FDCheck();

    // project residual such that only part orthogonal to nullspace is considered
    if (projector_!=Teuchos::null)
      projector_->ApplyPT(*residual_);

    // Apply Dirichlet boundary conditions to system of equations
    // residual values are supposed to be zero at Dirichlet boundaries
    {
      // time measurement: application of DBC to system
      TEUCHOS_FUNC_TIME_MONITOR("SCATRA:       + apply DBC to system");

      LINALG::ApplyDirichlettoSystem(sysmat_,increment_,residual_,zeros_,*(dbcmaps_->CondMap()));
    }

    // abort nonlinear iteration if desired
    if (AbortNonlinIter(iternum_,itemax,ittol,abstolres,actresidual))
      break;

    // initialize increment vector
    increment_->PutScalar(0.0);

    {
      // get cpu time
      const double tcpusolve=Teuchos::Time::wallTime();

      // time measurement: call linear solver
      TEUCHOS_FUNC_TIME_MONITOR("SCATRA:       + call linear solver");

      // do adaptive linear solver tolerance (not in first solve)
      if (isadapttol && iternum_>1)
      {
        solver_->AdaptTolerance(ittol,actresidual,adaptolbetter);
      }

      // reprepare Krylov projection only if ale and projection required
      if (updateprojection_)
        UpdateKrylovSpaceProjection();

      strategy_->Solve(solver_,sysmat_,increment_,residual_,phinp_,iternum_,projector_);

      solver_->ResetTolerance();

      // end time measurement for solver
      dtsolve_=Teuchos::Time::wallTime()-tcpusolve;
    }

    //------------------------------------------------ update solution vector
    phinp_->Update(1.0,*increment_,1.0);

    //-------- update values at intermediate time steps (only for gen.-alpha)
    ComputeIntermediateValues();

  } // nonlinear iteration

  return;
} // ScaTraTimIntImpl::NonlinearSolve


/*----------------------------------------------------------------------*
 | check if to stop the nonlinear iteration                    gjb 09/08|
 *----------------------------------------------------------------------*/
bool SCATRA::ScaTraTimIntImpl::AbortNonlinIter(
    const int itnum,
    const int itemax,
    const double ittol,
    const double abstolres,
    double& actresidual)
{
  //----------------------------------------------------- compute norms
  double incconnorm_L2(0.0);
  double incpotnorm_L2(0.0);

  double connorm_L2(0.0);
  double potnorm_L2(0.0);

  double conresnorm(0.0);
  double potresnorm(0.0);

  double conresnorminf(0.0);

  // Calculate problem-specific norms
  CalcProblemSpecificNorm(conresnorm,incconnorm_L2,connorm_L2,incpotnorm_L2,potnorm_L2,potresnorm,conresnorminf);

  // care for the case that nothing really happens in the concentration
  // or potential field
  if (connorm_L2 < 1e-5)
  {
    connorm_L2 = 1.0;
  }
  if (potnorm_L2 < 1e-5)
  {
    potnorm_L2 = 1.0;
  }

  // absolute tolerance for deciding if residual is (already) zero
  // prevents additional solver calls that will not improve the residual anymore

  //-------------------------------------------------- output to screen
  /* special case of very first iteration step:
      - solution increment is not yet available
      - do not perform a solver call when the initial residuals are < EPS14*/
  if (itnum == 1)
  {
    // print first line of convergence table to screen
    PrintConvergenceValuesFirstIter(itnum,itemax,ittol,conresnorm,potresnorm,conresnorminf);

    // abort iteration, when there's nothing more to do
    if ((conresnorm < abstolres) && (potresnorm < abstolres))
    {
      // print finish line of convergence table to screen
      PrintConvergenceFinishLine();

      return true;
    }
  }
  /* ordinary case later iteration steps:
      - solution increment can be printed
      - convergence check should be done*/
  else
  {
    // print current line of convergence table to screen
    PrintConvergenceValues(itnum,itemax,ittol,conresnorm,potresnorm,incconnorm_L2,connorm_L2,incpotnorm_L2,potnorm_L2,conresnorminf);

    // this is the convergence check
    // We always require at least one solve. We test the L_2-norm of the
    // current residual. Norm of residual is just printed for information
    if (conresnorm <= ittol and potresnorm <= ittol and
        incconnorm_L2/connorm_L2 <= ittol and incpotnorm_L2/potnorm_L2 <= ittol)
    {
      // print finish line of convergence table to screen
      PrintConvergenceFinishLine();

      // write info to error file
      if (myrank_ == 0)
        if (errfile_!=NULL)
          fprintf(errfile_,"elch solve:   %3d/%3d  tol=%10.3E[L_2 ]  cres=%10.3E  pres=%10.3E  cinc=%10.3E  pinc=%10.3E\n",
              itnum,itemax,ittol,conresnorm,potresnorm,
              incconnorm_L2/connorm_L2,incpotnorm_L2/potnorm_L2);

      return true;
    }

    // abort iteration, when there's nothing more to do! -> more robustness
    if ((conresnorm < abstolres) && (potresnorm < abstolres))
    {
      // print finish line of convergence table to screen
      PrintConvergenceFinishLine();

      return true;
    }

    // if not yet converged go on...
  }

  // warn if itemax is reached without convergence, but proceed to
  // next timestep...
  if ((itnum == itemax))
  {
    if (myrank_ == 0)
    {
      std::cout << "+---------------------------------------------------------------+" << std::endl;
      std::cout << "|            >>>>>> not converged in itemax steps!              |" << std::endl;
      std::cout << "+---------------------------------------------------------------+" << std::endl << std::endl;

      if (errfile_!=NULL)
      {
        fprintf(errfile_,"elch divergent solve:   %3d/%3d  tol=%10.3E[L_2 ]  cres=%10.3E  pres=%10.3E  cinc=%10.3E  pinc=%10.3E\n",
            itnum,itemax,ittol,conresnorm,potresnorm,
            incconnorm_L2/connorm_L2,incpotnorm_L2/potnorm_L2);
      }
    }
    // yes, we stop the iteration
    return true;
  }

  // return the maximum residual value -> used for adaptivity of linear solver tolerance
  actresidual = std::max(conresnorm,potresnorm);
  actresidual = std::max(actresidual,incconnorm_L2/connorm_L2);
  actresidual = std::max(actresidual,incpotnorm_L2/potnorm_L2);

  // check for INF's and NaN's before going on...
  if (std::isnan(incconnorm_L2) or
      std::isnan(incpotnorm_L2) or
      std::isnan(connorm_L2) or
      std::isnan(potnorm_L2) or
      std::isnan(conresnorm) or
      std::isnan(potresnorm))
    dserror("calculated vector norm is NaN.");

  if (abs(std::isinf(incconnorm_L2)) or
      abs(std::isinf(incpotnorm_L2))  or
      abs(std::isinf(connorm_L2))  or
      abs(std::isinf(potnorm_L2))  or
      abs(std::isinf(conresnorm))  or
      abs(std::isinf(potresnorm)) )
    dserror("calculated vector norm is INF.");

  return false;
} // ScaTraTimIntImpl::AbortNonlinIter


/*----------------------------------------------------------------------*
 | Calculate problem specific norm                            ehrl 01/14|
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::CalcProblemSpecificNorm(
    double& conresnorm,
    double& incconnorm_L2,
    double& connorm_L2,
    double& incpotnorm_L2,
    double& potnorm_L2,
    double& potresnorm,
    double& conresnorminf)
{
  residual_ ->Norm2(&conresnorm);
  increment_->Norm2(&incconnorm_L2);
  phinp_    ->Norm2(&connorm_L2);
  residual_ ->NormInf(&conresnorminf);

  return;
}


/*----------------------------------------------------------------------*
 | print header of convergence table to screen               fang 11/14 |
 *----------------------------------------------------------------------*/
inline void SCATRA::ScaTraTimIntImpl::PrintConvergenceHeader()
{
  if (myrank_ == 0)
    std::cout << "+------------+-------------------+--------------+--------------+------------------+\n"
             << "|- step/max -|- tol      [norm] -|-- con-res ---|-- con-inc ---|-- con-res-inf ---|" << std::endl;

  return;
} // SCATRA::ScaTraTimIntImpl::PrintConvergenceHeader


/*----------------------------------------------------------------------*
 | print first line of convergence table to screen           fang 11/14 |
 *----------------------------------------------------------------------*/
inline void SCATRA::ScaTraTimIntImpl::PrintConvergenceValuesFirstIter(
    const int&              itnum,          //!< current Newton-Raphson iteration step
    const int&              itemax,         //!< maximum number of Newton-Raphson iteration steps
    const double&           ittol,          //!< relative tolerance for Newton-Raphson scheme
    const double&           conresnorm,     //!< L2 norm of concentration residual
    const double&           potresnorm,     //!< L2 norm of potential residual (only relevant for electrochemistry)
    const double&           conresnorminf   //!< infinity norm of concentration residual
)
{
  if (myrank_ == 0)
    std::cout << "|  " << std::setw(3) << itnum << "/" << std::setw(3) << itemax << "   | "
             << std::setw(10) << std::setprecision(3) << std::scientific << ittol << "[L_2 ]  | "
             << std::setw(10) << std::setprecision(3) << std::scientific << conresnorm << "   |      --      | "
             << std::setw(10) << std::setprecision(3) << std::scientific << conresnorminf << "       | (      --     ,te="
             << std::setw(10) << std::setprecision(3) << std::scientific << dtele_ << ")" << std::endl;

  return;
} // SCATRA::ScaTraTimIntImpl::PrintConvergenceValuesFirstIter


/*----------------------------------------------------------------------*
 | print current line of convergence table to screen         fang 11/14 |
 *----------------------------------------------------------------------*/
inline void SCATRA::ScaTraTimIntImpl::PrintConvergenceValues(
    const int&              itnum,           //!< current Newton-Raphson iteration step
    const int&              itemax,          //!< maximum number of Newton-Raphson iteration steps
    const double&           ittol,           //!< relative tolerance for Newton-Raphson scheme
    const double&           conresnorm,      //!< L2 norm of concentration residual
    const double&           potresnorm,      //!< L2 norm of potential residual (only relevant for electrochemistry)
    const double&           incconnorm_L2,   //!< L2 norm of concentration increment
    const double&           connorm_L2,      //!< L2 norm of concentration state vector
    const double&           incpotnorm_L2,   //!< L2 norm of potential increment
    const double&           potnorm_L2,      //!< L2 norm of potential state vector
    const double&           conresnorminf    //!< infinity norm of concentration residual
)
{
  if (myrank_ == 0)
    std::cout << "|  " << std::setw(3) << itnum << "/" << std::setw(3) << itemax << "   | "
             << std::setw(10) << std::setprecision(3) << std::scientific << ittol << "[L_2 ]  | "
             << std::setw(10) << std::setprecision(3) << std::scientific << conresnorm << "   | "
             << std::setw(10) << std::setprecision(3) << std::scientific << incconnorm_L2/connorm_L2 << "   | "
             << std::setw(10) << std::setprecision(3) << std::scientific << conresnorminf << "       | (ts="
             << std::setw(10) << std::setprecision(3) << std::scientific << dtsolve_ << ",te="
             << std::setw(10) << std::setprecision(3) << std::scientific << dtele_ << ")" << std::endl;

  return;
} // SCATRA::ScaTraTimIntImpl::PrintConvergenceValues


/*----------------------------------------------------------------------*
 | print finish line of convergence table to screen          fang 11/14 |
 *----------------------------------------------------------------------*/
inline void SCATRA::ScaTraTimIntImpl::PrintConvergenceFinishLine()
{
  if (myrank_ == 0)
    std::cout << "+------------+-------------------+--------------+--------------+------------------+" << std::endl << std::endl;

  return;
} // SCATRA::ScaTraTimIntImpl::PrintConvergenceFinishLine


/*----------------------------------------------------------------------*
| returns matching std::string for each time integration scheme   gjb 08/08 |
*----------------------------------------------------------------------*/
std::string SCATRA::ScaTraTimIntImpl::MapTimIntEnumToString
(
   const enum INPAR::SCATRA::TimeIntegrationScheme term
)
{
  // length of return std::string is 14 due to usage in formated screen output
  switch (term)
  {
  case INPAR::SCATRA::timeint_one_step_theta :
    return "One-Step-Theta";
    break;
  case INPAR::SCATRA::timeint_bdf2 :
    return "     BDF2     ";
    break;
  case INPAR::SCATRA::timeint_stationary :
    return "  Stationary  ";
    break;
  case INPAR::SCATRA::timeint_gen_alpha :
    return "  Gen. Alpha  ";
    break;
  default :
    dserror("Cannot cope with name enum %d", term);
    return "";
    break;
  }

  return "";
} // ScaTraTimIntImpl::MapTimIntEnumToString


/*----------------------------------------------------------------------*
 |  write current state to BINIO                             gjb   08/08|
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::OutputState()
{
  // solution
  output_->WriteVector("phinp", phinp_);

  // convective velocity (not written in case of coupled simulations)
  if (cdvel_ != INPAR::SCATRA::velocity_Navier_Stokes)
    output_->WriteVector("convec_velocity", convel_,IO::DiscretizationWriter::nodevector);

  // displacement field
  if (isale_) output_->WriteVector("dispnp", dispnp_);

  return;
} // ScaTraTimIntImpl::OutputState


/*----------------------------------------------------------------------*
 | increment time and step for next iteration                     mr. x |
 *----------------------------------------------------------------------*/
inline void SCATRA::ScaTraTimIntImpl::IncrementTimeAndStep()
{
  step_ += 1;
  time_ += dta_;
}


/*----------------------------------------------------------------------*
 |  add dirichlet dofs to dbcmaps_                         rauch   04/15|
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::AddDirichCond(const Teuchos::RCP<const Epetra_Map> maptoadd)
{
  std::vector<Teuchos::RCP<const Epetra_Map> > condmaps;
  condmaps.push_back(maptoadd);
  condmaps.push_back(dbcmaps_->CondMap());
  Teuchos::RCP<Epetra_Map> condmerged = LINALG::MultiMapExtractor::MergeMaps(condmaps);
  *dbcmaps_ = LINALG::MapExtractor(*(discret_->DofRowMap()), condmerged);
  return;
} // ScaTraTimIntImpl::AddDirichCond


/*----------------------------------------------------------------------*
 |  remove dirichlet dofs from dbcmaps_                    rauch   04/15|
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::RemoveDirichCond(const Teuchos::RCP<const Epetra_Map> maptoremove)
{
  std::vector<Teuchos::RCP<const Epetra_Map> > othermaps;
  othermaps.push_back(maptoremove);
  othermaps.push_back(dbcmaps_->OtherMap());
  Teuchos::RCP<Epetra_Map> othermerged = LINALG::MultiMapExtractor::MergeMaps(othermaps);
  *dbcmaps_ = LINALG::MapExtractor(*(discret_->DofRowMap()), othermerged, false);
  return;
} // ScaTraTimIntImpl::RemoveDirichCond


/*----------------------------------------------------------------------*
 |  return pointer to const dofrowmap                      rauch   04/15|
 *----------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Map>
SCATRA::ScaTraTimIntImpl::DofRowMap()
{ return DofRowMap(0); }


/*----------------------------------------------------------------------*
 |  return pointer to const dofrowmap of specified dofset  rauch   04/15|
 *----------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Map> SCATRA::ScaTraTimIntImpl::DofRowMap(int nds)
{
  const Epetra_Map* dofrowmap = discret_->DofRowMap(nds);
  return Teuchos::rcp(dofrowmap, false);
}


/*----------------------------------------------------------------------*
 |  add contribution to rhs                                rauch   04/15|
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntImpl::AddContributionToRHS(const Teuchos::RCP<const Epetra_Vector> contributing_vector)
{
  int err = neumann_loads_->Update(1.0,*contributing_vector,1.0);

  if(err!=0)
    dserror(" Epetra_Vector update threw error code %i ",err);

  return;
}

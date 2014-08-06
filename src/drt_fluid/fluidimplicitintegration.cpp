/*!----------------------------------------------------------------------
\file fluidimplicitintegration.cpp
\brief Control routine for fluid (in)stationary solvers,

     including instationary solvers based on

     o a one-step-theta time-integration scheme,

     o a two-step BDF2 time-integration scheme
       (with potential one-step-theta start algorithm),

     o two variants of a generalized-alpha time-integration scheme

     and a stationary solver.

<pre>
Maintainers: Ursula Rasthofer & Volker Gravemeier
             {rasthofer,vgravem}@lnm.mw.tum.de
             http://www.lnm.mw.tum.de
             089 - 289-15236/-245
</pre>

*----------------------------------------------------------------------*/
#undef WRITEOUTSTATISTICS

#include "fluidimplicitintegration.H"
#include "fluid_utils.H"
#include "fluidresulttest.H"
#include "../drt_fluid_turbulence/dyn_smag.H"
#include "../drt_fluid_turbulence/dyn_vreman.H"
#include "../drt_fluid_turbulence/boxfilter.H"
#include "../drt_fluid_turbulence/scale_sep_gmo.H"
#include "../drt_fluid_turbulence/turbulence_statistic_manager.H"
#include "fluid_utils_mapextractor.H"
#include "fluid_meshtying.H"
#include "fluid_MHD_evaluate.H"
#include "../drt_fluid_turbulence/turbulence_hit_initial_field.H"
#include "../drt_fluid_turbulence/turbulence_hit_forcing.H"
#include "../drt_fluid_turbulence/drt_transfer_turb_inflow.H"
#include "fluid_utils_infnormscaling.H"
#include "../drt_fluid_ele/fluid_ele.H"
#include "../linalg/linalg_ana.H"
#include "../linalg/linalg_utils.H"
#include "../linalg/linalg_solver.H"
#include "../linalg/linalg_krylov_projector.H"
#include "../linalg/linalg_mapextractor.H"
#include "../drt_io/io.H"
#include "../drt_lib/drt_dserror.H"
#include "../drt_lib/drt_condition.H"
#include "../drt_lib/drt_condition_utils.H"
#include "../drt_lib/drt_globalproblem.H"
#include "../drt_lib/drt_assemblestrategy.H"
#include "../drt_lib/drt_locsys.H"
#include "../drt_comm/comm_utils.H"
#include "../drt_adapter/adapter_coupling_mortar.H"
#include "../drt_adapter/ad_opt.H"
#include "../drt_nurbs_discret/drt_apply_nurbs_initial_condition.H"
#include "../drt_fluid_ele/fluid_ele_action.H"
#include "../drt_mat/matpar_bundle.H"
#include "../drt_mat/newtonianfluid.H"

#include "../drt_meshfree_discret/drt_meshfree_discret.H"

// print error file for function EvaluateErrorComparedToAnalyticalSol()
#include "../drt_io/io_control.H"

// for AVM3 solver:
#include <MLAPI_Workspace.h>
#include <MLAPI_Aggregation.h>

#include "../drt_io/io.H"
#include "../drt_io/io_control.H"
#include "../drt_io/io_gmsh.H"

#include <cmath>

// allows for dealing with edged-based stabilization
#include "../drt_lib/drt_discret_faces.H"

#include <Teuchos_StrUtils.hpp>


#include "../drt_geometry/element_coordtrafo.H"
#include "../drt_geometry/position_array.H"


/*----------------------------------------------------------------------*
 |  Constructor (public)                                     gammi 04/07|
 *----------------------------------------------------------------------*/
FLD::FluidImplicitTimeInt::FluidImplicitTimeInt(
    const Teuchos::RCP<DRT::Discretization>&      actdis,
    const Teuchos::RCP<LINALG::Solver>&           solver,
    const Teuchos::RCP<Teuchos::ParameterList>&   params,
    const Teuchos::RCP<IO::DiscretizationWriter>& output,
    bool                                          alefluid /*= false*/
):TimInt(actdis,solver,params,output),
  // call constructor for "nontrivial" objects
  alefluid_(alefluid),
  writestresses_(params_->get<int>("write stresses", 0)),
  write_wall_shear_stresses_(params_->get<int>("write wall shear stresses", 0)),
  dtele_(0.0),
  dtfilter_(0.0),
  dtsolve_(0.0),
  external_loads_(Teuchos::null),
  forcing_(Teuchos::null),
  homisoturb_forcing_(Teuchos::null),
  surfacesplitter_(NULL),
  inrelaxation_(false),
  msht_(INPAR::FLUID::no_meshtying),
  facediscret_(Teuchos::null),
  fldgrdisp_(Teuchos::null),
  velatmeshfreenodes_(Teuchos::null),
  locsysman_(Teuchos::null),
  massmat_(Teuchos::null),
  logenergy_(Teuchos::null)
{
  return;
}


/*----------------------------------------------------------------------*
 |  initialize algorithm                                rasthofer 04/14 |
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::Init()
{
  // time measurement: initialization
  TEUCHOS_FUNC_TIME_MONITOR(" + initialization");

  // -------------------------------------------------------------------
  // get the basic parameters first
  // -------------------------------------------------------------------
  //genalpha integration scheme (afgenalpha or npgenalpha)
  //move to adapterif (timealgo_==INPAR::FLUID::timeint_afgenalpha or timealgo_==INPAR::FLUID::timeint_npgenalpha)
  // time-step size
  dtp_ = params_->get<double>("time step size");

  // parameter theta for time-integration schemes (required for all schemes)
  theta_ = params_->get<double>("theta");

  // number of steps for starting algorithm, only for GenAlpha so far
  numstasteps_ = params_->get<int> ("number of start steps");

  // parameter for linearization scheme (fixed-point-like or Newton)
  newton_ = DRT::INPUT::get<INPAR::FLUID::LinearisationAction>(*params_, "Linearisation");

  predictor_ = params_->get<std::string>("predictor","steady_state_predictor");

  // flag to skip calculation of residual after solution has converged
  inconsistent_ = params_->get<bool>("INCONSISTENT_RESIDUAL",false);
  if (inconsistent_ and myrank_==0)
    std::cout << "Warning: residual will not be adapted to the final solution of the nonlinear solution procedure!" << std::endl;

  // form of convective term
  convform_ = params_->get<std::string>("form of convective term","convective");

  // -------------------------------------------------------------------
  // flag for potential nonlinear boundary conditions
  // -------------------------------------------------------------------
  nonlinearbc_ = false;
  if (params_->get<std::string>("Nonlinear boundary conditions","no") == "yes")
    nonlinearbc_ = true;

  discret_->ComputeNullSpaceIfNecessary(solver_->Params(),true);

  // ensure that degrees of freedom in the discretization have been set
  if (!discret_->Filled() || !discret_->HaveDofs()) discret_->FillComplete();

  // -------------------------------------------------------------------
  // get a vector layout from the discretization to construct matching
  // vectors and matrices
  //                 local <-> global dof numbering
  // -------------------------------------------------------------------
  const Epetra_Map* dofrowmap = discret_->DofRowMap();

  // -------------------------------------------------------------------
  // get a vector layout from the discretization for a vector which only
  // contains the velocity dofs and for one vector which only contains
  // pressure degrees of freedom.
  // -------------------------------------------------------------------
  numdim_ = params_->get<int>("number of velocity degrees of freedom");

  if (velpressplitter_.NumMaps() == 0)
    FLD::UTILS::SetupFluidSplit(*discret_,numdim_,velpressplitter_);
  // if the pressure map is empty, the user obviously specified a wrong
  // number of space dimensions in the input file
  if (velpressplitter_.CondMap()->NumGlobalElements()<1)
    dserror("Pressure map empty. Wrong DIM value in input file?");

  // -------------------------------------------------------------------
  // create empty vectors
  // -------------------------------------------------------------------
  Reset();

  // initialize flow-rate and flow-volume vectors (fixed to length of four,
  // for the time being) in case of flow-dependent pressure boundary conditions,
  // including check of respective conditions
  if (nonlinearbc_)
  {
    std::vector<DRT::Condition*> flowdeppressureline;
    discret_->GetCondition("LineFlowDepPressure",flowdeppressureline);
    std::vector<DRT::Condition*> flowdeppressuresurf;
    discret_->GetCondition("SurfaceFlowDepPressure",flowdeppressuresurf);

    // check number of boundary conditions
    if (flowdeppressureline.size() > 0 or flowdeppressuresurf.size() > 0)
    {
      // initialize vectors for flow rate and volume
      flowratenp_(true);
      flowratenpi_(true);
      flowraten_(true);
      flowratenm_(true);

      flowvolumenp_(true);
      flowvolumenpi_(true);
      flowvolumen_(true);
      flowvolumenm_(true);

      if (flowdeppressureline.size() > 4 or flowdeppressuresurf.size() > 4)
        dserror("More than four flow-dependent pressure line or surface conditions assigned -> correction required!");
    }
  }

  // ---------------------------------------------------------------------
  // Create LocSysManager, if needed (used for LocSys-Dirichlet BCs)
  // ---------------------------------------------------------------------
  {
    std::vector<DRT::Condition*> locsysconditions(0);
    discret_->GetCondition("Locsys", locsysconditions);
    if (locsysconditions.size())
    {
      // Initialize locsys manager
      locsysman_ = Teuchos::rcp(new DRT::UTILS::LocsysManager(*discret_));
    }
  }

  // Vectors associated to boundary conditions
  // -----------------------------------------

  // a vector of zeros to be used to enforce zero dirichlet boundary conditions
  zeros_ = LINALG::CreateVector(*dofrowmap,true);

  // object holds maps/subsets for DOFs subjected to Dirichlet BCs and otherwise
  dbcmaps_ = Teuchos::rcp(new LINALG::MapExtractor());
  {
    Teuchos::ParameterList eleparams;
    // other parameters needed by the elements
    eleparams.set("total time",time_);

    ApplyDirichletBC(eleparams, zeros_, Teuchos::null, Teuchos::null, true);
    //zeros_ has to be reset to zero here, since it has a different value after call of ApplyDirichletBC(...)
    zeros_->PutScalar(0.0);
  }

  // a vector containing the integrated traction in boundary normal direction for slip boundary conditions (Unit: Newton [N])
  slip_bc_normal_tractions_ = LINALG::CreateVector(*dofrowmap,true);

  // -------------------------------------------------------------------
  // create empty system matrix --- stiffness and mass are assembled in
  // one system matrix!
  // -------------------------------------------------------------------

  // This is a first estimate for the number of non zeros in a row of
  // the matrix. Assuming a structured 3d-fluid mesh we have 27 adjacent
  // nodes with 4 dofs each. (27*4=108)
  // We do not need the exact number here, just for performance reasons
  // a 'good' estimate

  if (not params_->get<int>("Simple Preconditioner",0) && not params_->get<int>("AMG BS Preconditioner",0)
      && params_->get<int>("MESHTYING") == INPAR::FLUID::no_meshtying)
  {
    // initialize standard (stabilized) system matrix (construct its graph already)
    sysmat_ = Teuchos::rcp(new LINALG::SparseMatrix(*dofrowmap,108,false,true));
  }
  else if(params_->get<int>("MESHTYING")!= INPAR::FLUID::no_meshtying)
    SetupMeshtying();
  else
  {
    Teuchos::RCP<LINALG::BlockSparseMatrix<FLD::UTILS::VelPressSplitStrategy> > blocksysmat =
      Teuchos::rcp(new LINALG::BlockSparseMatrix<FLD::UTILS::VelPressSplitStrategy>(velpressplitter_,velpressplitter_,108,false,true));
    blocksysmat->SetNumdim(numdim_);
    sysmat_ = blocksysmat;
  }

  // the vector containing body and surface forces
  neumann_loads_= LINALG::CreateVector(*dofrowmap,true);

  // Vectors used for solution process
  // ---------------------------------
  // rhs: standard (stabilized) residual vector (rhs for the incremental form)
  residual_      = LINALG::CreateVector(*dofrowmap,true);
  trueresidual_  = LINALG::CreateVector(*dofrowmap,true);

  // right hand side vector for linearised solution;
//  rhs_ = LINALG::CreateVector(*dofrowmap,true);

  // Nonlinear iteration increment vector
  incvel_ = LINALG::CreateVector(*dofrowmap,true);

  // -------------------------------------------------------------------
  // initialize vectors and flags for turbulence approach
  // -------------------------------------------------------------------

  turbmodel_ = INPAR::FLUID::no_model;

  std::string physmodel = params_->sublist("TURBULENCE MODEL").get<std::string>("PHYSICAL_MODEL","no_model");

  // flag for special flow
  special_flow_ = params_->sublist("TURBULENCE MODEL").get<std::string>("CANONICAL_FLOW","no");

  // scale-separation
  scale_sep_ = INPAR::FLUID::no_scale_sep;

  // fine-scale subgrid viscosity?
  fssgv_ = DRT::INPUT::IntegralValue<INPAR::FLUID::FineSubgridVisc>(params_->sublist("TURBULENCE MODEL"),"FSSUGRVISC");

  // warning if classical (all-scale) turbulence model and fine-scale
  // subgrid-viscosity approach are intended to be used simultaneously
  if (fssgv_ != INPAR::FLUID::no_fssgv
      and (physmodel == "Smagorinsky"
        or physmodel == "Dynamic_Smagorinsky"
        or physmodel == "Smagorinsky_with_van_Driest_damping"))
    dserror("No combination of classical all-scale subgrid-viscosity turbulence model and fine-scale subgrid-viscosity approach currently possible!");

  if (params_->sublist("TURBULENCE MODEL").get<std::string>("TURBULENCE_APPROACH","DNS_OR_RESVMM_LES") == "CLASSICAL_LES")
  {

    if(physmodel == "Dynamic_Smagorinsky")
    {
      turbmodel_ = INPAR::FLUID::dynamic_smagorinsky;

      // get one instance of the dynamic Smagorinsky class
      DynSmag_=Teuchos::rcp(new FLD::DynSmagFilter(discret_, *params_));
    }
    else if (physmodel == "Smagorinsky")
      turbmodel_ = INPAR::FLUID::smagorinsky;
    else if (physmodel == "Smagorinsky_with_van_Driest_damping")
      turbmodel_ = INPAR::FLUID::smagorinsky_with_van_Driest_damping;
    else if(physmodel == "Scale_Similarity" or physmodel == "Scale_Similarity_basic")
    {
      if (physmodel == "Scale_Similarity")
        turbmodel_ = INPAR::FLUID::scale_similarity;
      else
        turbmodel_ = INPAR::FLUID::scale_similarity_basic;
      Teuchos::ParameterList *  modelparams =&(params_->sublist("MULTIFRACTAL SUBGRID SCALES"));
      const std::string scale_sep = modelparams->get<std::string>("SCALE_SEPARATION");
      if (scale_sep == "box_filter")
      {
        scale_sep_ = INPAR::FLUID::box_filter;
        // get one instance of the Boxfilter class
        Boxf_=Teuchos::rcp(new FLD::Boxfilter(discret_, *params_));
      }
      else if (scale_sep == "algebraic_multigrid_operator")
      {
        scale_sep_ = INPAR::FLUID::algebraic_multigrid_operator;
      }
      else if (scale_sep == "geometric_multigrid_operator")
      {
        dserror("Not yet implemented!");
      }
      else
      {
        dserror("Unknown filter type!");
      }

      const Epetra_Map* nodecolmap = discret_->NodeColMap();
      filteredvel_ = Teuchos::rcp(new Epetra_MultiVector(*nodecolmap,3,true));
      finescalevel_ = Teuchos::rcp(new Epetra_MultiVector(*nodecolmap,3,true));
      filteredreystr_ = Teuchos::rcp(new Epetra_MultiVector(*nodecolmap,9,true));

      fsvelaf_  = LINALG::CreateVector(*dofrowmap,true);
    }
    else if(physmodel == "Multifractal_Subgrid_Scales")
    {
      turbmodel_ = INPAR::FLUID::multifractal_subgrid_scales;

      fsvelaf_  = LINALG::CreateVector(*dofrowmap,true);

      Teuchos::ParameterList *  modelparams =&(params_->sublist("MULTIFRACTAL SUBGRID SCALES"));

      const std::string scale_sep = modelparams->get<std::string>("SCALE_SEPARATION");
      if (scale_sep == "box_filter")
      {
        scale_sep_ = INPAR::FLUID::box_filter;

        // get one instance of the Boxfilter class
        Boxf_=Teuchos::rcp(new FLD::Boxfilter(discret_, *params_));

        if (fssgv_ != INPAR::FLUID::no_fssgv)
          dserror("No fine-scale subgrid viscosity for this scale separation operator!");
      }
      else if (scale_sep == "algebraic_multigrid_operator")
      {
        scale_sep_ = INPAR::FLUID::algebraic_multigrid_operator;
      }
      else if (scale_sep == "geometric_multigrid_operator")
      {
        scale_sep_ = INPAR::FLUID::geometric_multigrid_operator;
        ScaleSepGMO_ = Teuchos::rcp(new LESScaleSeparation(scale_sep_,discret_));

        if (fssgv_ != INPAR::FLUID::no_fssgv)
          dserror("No fine-scale subgrid viscosity for this scale separation operator!");
      }
      else
      {
        dserror("Unknown filter type!");
      }

      // fine-scale scalar at time n+alpha_F/n+1 and n+alpha_M/n
      // (only required for low-Mach-number case)
      fsscaaf_ = LINALG::CreateVector(*dofrowmap,true);
    }
    else if (physmodel == "Vreman")
    {
      turbmodel_ = INPAR::FLUID::vreman;
    }
    else if (physmodel == "Dynamic_Vreman")
    {
      turbmodel_ = INPAR::FLUID::dynamic_vreman;
      Vrem_=Teuchos::rcp(new FLD::Vreman(discret_, *params_));
    }
    else if (physmodel == "no_model")
      dserror("Turbulence model for LES expected!");
    else
      dserror("Undefined turbulence model!");

    PrintTurbulenceModel();
  }
  else
  {
    if (turbmodel_ != INPAR::FLUID::no_model)
      dserror("Set TURBULENCE APPROACH to CLASSICAL LES to activate turbulence model!");
  }

  // -------------------------------------------------------------------
  // necessary only for the AVM3 approach:
  // fine-scale solution vector + respective output
  // -------------------------------------------------------------------
  if (fssgv_ != INPAR::FLUID::no_fssgv)
  {
    fsvelaf_  = LINALG::CreateVector(*dofrowmap,true);

    if (myrank_ == 0)
    {
      // Output
      std::cout << "FLUID: Fine-scale subgrid-viscosity approach based on AVM3: ";
      std::cout << &std::endl << &std::endl;
      std::cout << fssgv_;
      std::cout << " with Smagorinsky constant Cs= ";
      std::cout << params_->sublist("SUBGRID VISCOSITY").get<double>("C_SMAGORINSKY") ;
      std::cout << &std::endl << &std::endl << &std::endl;
    }
  }

  // -------------------------------------------------------------------
  // check whether we have a coupling to a turbulent inflow generating
  // computation and initialize the transfer if necessary
  // -------------------------------------------------------------------
  turbulent_inflow_condition_
    = Teuchos::rcp(new TransferTurbulentInflowCondition(discret_,dbcmaps_));

  // -------------------------------------------------------------------
  // initialize turbulence-statistics evaluation
  // -------------------------------------------------------------------
  statisticsmanager_=Teuchos::rcp(new FLD::TurbulenceStatisticManager(*this));
  // parameter for sampling/dumping period
  if (special_flow_ != "no")
    samstart_ = params_->sublist("TURBULENCE MODEL").get<int>("SAMPLING_START",1);

  // set gas constant to 1.0 for incompressible flow
  gasconstant_ = 1.0;

  // object for a redistributed evaluation of the mixed-hybrid Dirichlet condition
  MHD_evaluator_=Teuchos::null;

  std::vector<DRT::Condition*> MHDcndSurf;
  discret_->GetCondition("SurfaceMixHybDirichlet",MHDcndSurf);
  if(MHDcndSurf.size()!=0) // redistributed evaluation currently only for surface conditions!
  {
    bool mhd_redis_eval = false; // get this info from the input file
    if(mhd_redis_eval)
      MHD_evaluator_=Teuchos::rcp(new FLD::FluidMHDEvaluate(discret_));
  }

  // initialize density_scaling_ as null
  density_scaling_ = Teuchos::null;

  if (params_->get<bool>("INFNORMSCALING"))
  {
    fluid_infnormscaling_ = Teuchos::rcp(new FLD::UTILS::FluidInfNormScaling(velpressplitter_));
  }

  // ------------------------------------------------------------------------------
  // Check, if features are used with the locsys manager that are not supported,
  // or better, not implemented yet.
  // ------------------------------------------------------------------------------
  if (locsysman_ != Teuchos::null)
  {
    // TangVel predictor
    if (predictor_ == "TangVel") {
      dserror("No problem types involving TangVel predictors are supported for use with locsys conditions!");
    }

    // Meshtying
    if (msht_ != INPAR::FLUID::no_meshtying) {
      dserror("No problem types involving meshtying are supported for use with locsys conditions!");
    }

    // Additionally, locsys doesn't work yet with AVM3 and LinearRelaxationSolve. Those
    // checks needed to be put in the corresponding functions, so they are not listed here.
  }

  return;
} // FluidImplicitTimeInt::Init()


/*----------------------------------------------------------------------*
 | complete initialization                                              |
 |                                                                      |
 |  o is called at the end of the constructor of the time integrators   |
 |  o used for init functions that require the time integrators to exist|
 |                                                              bk 01/14|
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::CompleteGeneralInit()
{
  // -------------------------------------------------------------------
  // initialize forcing for homogeneous isotropic turbulence
  // -------------------------------------------------------------------
  if (special_flow_ == "forced_homogeneous_isotropic_turbulence"
      or special_flow_ == "scatra_forced_homogeneous_isotropic_turbulence"
      or special_flow_ == "decaying_homogeneous_isotropic_turbulence")
  {
    forcing_ = LINALG::CreateVector(*(discret_->DofRowMap()),true);
    forcing_->PutScalar(0.0);
    homisoturb_forcing_ = Teuchos::rcp(new FLD::HomIsoTurbForcing(*this));
  }

  //Set general parameters:
  //the following two functions are overloaded (switched off) in TimIntPoro
  SetElementGeneralFluidParameter();
  SetElementTurbulenceParameter();

  // set special parameter for faces/edges when using edge-based fluid stabilizations
  if(params_->sublist("RESIDUAL-BASED STABILIZATION").get<std::string>("STABTYPE")=="edge_based")
    SetFaceGeneralFluidParameter();

  // sysmat might be singular (if we have a purely Dirichlet constrained
  // problem, the pressure mode is defined only up to a constant)
  // in this case, we need a basis vector for the nullspace/kernel

  //initialize Krylov space projection
  InitKrylovSpaceProjection();

  // ------------------------------------------------------------------------------
  // Pre-compute mass matrix in case the user wants output of kinetic energy
  // ------------------------------------------------------------------------------
  if (params_->get<bool>("COMPUTE_EKIN"))
  {
    // write energy-file
    {
      std::string fileiter = DRT::Problem::Instance()->OutputControlFile()->FileName();
      fileiter.append(".fluidenergy");
      logenergy_ = Teuchos::rcp(new std::ofstream(fileiter.c_str()));

      // write header of energy-file (if energy file is desired by user)
      if (myrank_ == 0 and (not logenergy_.is_null()))
      {
        (*logenergy_) << "# Kinetic energy in fluid field\n"
                      << "# num procs = " << Discretization()->Comm().NumProc()
                      << std::endl
                      << std::right << std::setw(9) << "# step"
                      << std::right << std::setw(16) << "time"
                      << std::right << std::setw(16) << "kinetic_energy"
                      << std::endl;

        (*logenergy_) << "#" << std::endl;
      }
    }

    massmat_ = Teuchos::rcp(new LINALG::SparseMatrix(*DofRowMap(), 108, false, true));
    EvaluateMassMatrix();
  }

  return;
}

/*----------------------------------------------------------------------*
 | Start the time integration. Allows                                   |
 |                                                                      |
 |  o starting steps with different algorithms                          |
 |  o the "standard" time integration                                   |
 |                                                           gammi 04/07|
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::Integrate()
{
  PrintStabilizationDetails();

  Teuchos::ParameterList* stabparams = &(params_->sublist("RESIDUAL-BASED STABILIZATION"));
  if(DRT::INPUT::IntegralValue<INPAR::FLUID::StabType>(*stabparams, "STABTYPE") == INPAR::FLUID::stabtype_edgebased)
  {
    // if the definition of internal faces would be included
    // in the standard discretization, these lines can be removed
    // and CreateInternalFacesExtension() can be called once
    // in the constructor of the fluid time integration
    // since we want to keep the standard discretization as clean as
    // possible, we create interal faces via an enhanced discretization
    // including the faces between elements
    facediscret_ = Teuchos::rcp_dynamic_cast<DRT::DiscretizationFaces>(discret_, true);
    facediscret_->CreateInternalFacesExtension(true);
  }

  // TimeLoop() calls SolveStationaryProblem() in stationary case
  TimeLoop();

  // print the results of time measurements
  if (DRT::Problem::Instance()->ProblemType() != prb_fluid_topopt)
  {
    Teuchos::RCP<const Teuchos::Comm<int> > TeuchosComm = COMM_UTILS::toTeuchosComm<int>(discret_->Comm());
    Teuchos::TimeMonitor::summarize(TeuchosComm.ptr(), std::cout, false, true, false);
  }

  return;
} // FluidImplicitTimeInt::Integrate

//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 | contains the time loop                                    gammi 04/07|
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void FLD::FluidImplicitTimeInt::TimeLoop()
{
  // time measurement: time loop
  TEUCHOS_FUNC_TIME_MONITOR(" + time loop");

  while (step_<stepmax_ and time_<maxtime_)
  {
    PrepareTimeStep();
    // -------------------------------------------------------------------
    //                       output to screen
    // -------------------------------------------------------------------
    PrintTimeStepInfo();

    // -----------------------------------------------------------------
    // intermediate solution step for homogeneous isotropic turbulence
    // -----------------------------------------------------------------
    CalcIntermediateSolution();

    // -----------------------------------------------------------------
    //                     solve nonlinear equation
    // -----------------------------------------------------------------
    Solve();

    // -------------------------------------------------------------------
    //                         update solution
    //        current solution becomes old solution of next timestep
    // -------------------------------------------------------------------
    TimeUpdate();

    // -------------------------------------------------------------------
    //  lift'n'drag forces, statistics time sample and output of solution
    //  and statistics
    // -------------------------------------------------------------------
    StatisticsAndOutput();

    // -------------------------------------------------------------------
    //                       update time step sizes
    // -------------------------------------------------------------------
    dtp_ = dta_;

    // -------------------------------------------------------------------
    //                    stop criterion for timeloop
    // -------------------------------------------------------------------
  }

} // FluidImplicitTimeInt::TimeLoop


/*----------------------------------------------------------------------*
 | setup the variables to do a new time step                 u.kue 06/07|
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::PrepareTimeStep()
{
  // -------------------------------------------------------------------
  //              set time-dependent parameters
  // -------------------------------------------------------------------
  EvaluateDtWithCFL();
  IncrementTimeAndStep();

  // Sets theta_ to a specific value for bdf2 and calculates
  // a pseudo-theta for genalpha (the latter in case of startalgo_)
  SetTheta();

  // -------------------------------------------------------------------
  // set part(s) of the rhs vector(s) belonging to the old timestep
  // (only meaningful for momentum part)
  //
  // stationary/af-generalized-alpha: hist_ = 0.0
  //
  // one-step-Theta:                  hist_ = veln_ + dt*(1-Theta)*accn_
  //
  // BDF2: for constant time step:    hist_ = 4/3*veln_ - 1/3*velnm_
  //
  // -------------------------------------------------------------------
  SetOldPartOfRighthandside();

  // -------------------------------------------------------------------
  //                     do explicit predictor step
  // -------------------------------------------------------------------

  // no predictor in first time step
  if (step_>1)
  {
    if (predictor_ != "TangVel")
    {
      ExplicitPredictor();
    }
    else
    {
      PredictTangVelConsistAcc();
    }
  }

  // -------------------------------------------------------------------
  //  Set time parameter for element call
  // -------------------------------------------------------------------
  SetElementTimeParameter();

  // -------------------------------------------------------------------
  // Update local coordinate systems (which may be time dependent)
  // -------------------------------------------------------------------
  if (locsysman_ != Teuchos::null) {
    discret_->ClearState();
    if (alefluid_)
    {
      discret_->SetState("dispnp", dispnp_);
    }
    locsysman_->Setup(time_);
    discret_->ClearState();
  }

  // -------------------------------------------------------------------
  //  evaluate Dirichlet and Neumann boundary conditions
  // -------------------------------------------------------------------
  {
    Teuchos::ParameterList eleparams;

    // total time required for Dirichlet conditions
    eleparams.set("total time",time_);

    // set vector values needed by elements
    discret_->ClearState();
    discret_->SetState("velnp",velnp_);

    // predicted dirichlet values
    // velnp then also holds prescribed new dirichlet values
    ApplyDirichletBC(eleparams,velnp_,Teuchos::null,Teuchos::null,false);

    // additionally evaluate problem-specific boundary conditions
    DoProblemSpecificBoundaryConditions();

    // By definition: Applying DC on the slave side of an internal interface is not allowed
    //                since it leads to an over-constraint system
    // Therefore, nodes belonging to the slave side of an internal interface have to be excluded from the DC.
    // However, a velocity value (projected from the Dirichlet condition on the master side)
    // has to be assigned to the DOF's on the slave side in order to evaluate the system matrix completely

    // Preparation for including DC on the master side in the condensation process
    if(msht_ != INPAR::FLUID::no_meshtying)
      meshtying_->IncludeDirichletInCondensation(velnp_, veln_);

    discret_->ClearState();

    // Transfer of boundary data if necessary
    turbulent_inflow_condition_->Transfer(veln_,velnp_,time_);

    // add problem-dependent parameters, e.g., thermodynamic pressure in case of loma
    SetCustomEleParamsApplyNonlinearBoundaryConditions(eleparams);

    if (alefluid_)
    {
      discret_->SetState("dispnp", dispnp_);
    }

    // evaluate Neumann conditions
    neumann_loads_->PutScalar(0.0);
    discret_->SetState("scaaf",scaaf_);
    discret_->EvaluateNeumann(eleparams,*neumann_loads_);
    discret_->ClearState();
  }


  // --------------------------------------------------
  // adjust accnp according to Dirichlet values of velnp for GenAlpha
  //
  GenAlphaUpdateAcceleration();
  // ----------------------------------------------------------------
  // compute values at intermediate time steps for GenAlpha
  // ----------------------------------------------------------------
  GenAlphaIntermediateValues();

  // -------------------------------------------------------------------
  // meshtying: evaluation of matrix P with potential mesh relocation
  // in ALE case
  // -------------------------------------------------------------------
  if (msht_ != INPAR::FLUID::no_meshtying and alefluid_)
    meshtying_->EvaluateWithMeshRelocation(dispnp_);

  // -------------------------------------------------------------------
  //           preparation of AVM3-based scale separation
  // -------------------------------------------------------------------
  if (step_==1 and (fssgv_ != INPAR::FLUID::no_fssgv or scale_sep_ == INPAR::FLUID::algebraic_multigrid_operator))
   AVM3Preparation();

  return;
}

/*----------------------------------------------------------------------*
 | nonlinear solve, i.e., (multiple) corrector                 vg 02/09 |
 | overloaded in TimIntTopOpt                                  bk 12/13 |
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::Solve()
{
  // -------------------------------------------------------------------
  // time measurement: nonlinear iteration
  // -------------------------------------------------------------------
  TEUCHOS_FUNC_TIME_MONITOR("   + corrector");

  dtsolve_  = 0.0;

  // -------------------------------------------------------------------
  // parameters and variables for nonlinear iteration
  // -------------------------------------------------------------------
  int          itnum = 0;
  int          itmax = 0;
  const double ittol = params_->get<double>("tolerance for nonlin iter");
  bool         stopnonliniter = false;

  // -------------------------------------------------------------------
  // currently default for turbulent channel flow:
  // only one iteration before sampling
  // -------------------------------------------------------------------
  // REMARK:
  // commented reduced number of iterations out as it seems that more iterations
  // are necessary before sampling to obtain a converged result
//  if (special_flow_ == "channel_flow_of_height_2" && step_<samstart_ )
//       itmax = 1;
//  else
  itmax = params_->get<int>("max nonlin iter steps");

  // -------------------------------------------------------------------
  // turn adaptive solver tolerance on/off
  // -------------------------------------------------------------------
  const bool   isadapttol    = params_->get<bool>("ADAPTCONV",true);
  const double adaptolbetter = params_->get<double>("ADAPTCONV_BETTER",0.01);

  // -------------------------------------------------------------------
  // option for multifractal subgrid-scale modeling approach within
  // variable-density flow at low Mach number:
  // adaption of CsgsD to resolution dependent CsgsB
  // when near-wall limit is used
  // see also comment within function
  // -------------------------------------------------------------------
  if ((physicaltype_ == INPAR::FLUID::loma or statisticsmanager_->WithScaTra()) and turbmodel_==INPAR::FLUID::multifractal_subgrid_scales)
    RecomputeMeanCsgsB();

  // -------------------------------------------------------------------
  // prepare print out for (multiple) corrector
  // -------------------------------------------------------------------
  if (myrank_ == 0)
  {
    printf("+------------+-------------------+--------------+--------------+--------------+--------------+\n");
    printf("|- step/max -|- tol      [norm] -|-- vel-res ---|-- pre-res ---|-- vel-inc ---|-- pre-inc ---|\n");
  }

  // -------------------------------------------------------------------
  // nonlinear iteration loop
  // -------------------------------------------------------------------
  while (stopnonliniter==false)
  {
    itnum++;

    // -------------------------------------------------------------------
    // preparatives for solver
    // -------------------------------------------------------------------
    PrepareSolve();

    // -------------------------------------------------------------------
    // solver:
    // - It is solved for velocity and pressure increments.
    // - Adaptive linear solver tolerance is used from second corrector
    //   step on.
    // - Time for solver is measured.
    // -------------------------------------------------------------------
    {
      TEUCHOS_FUNC_TIME_MONITOR("      + solver calls");

      const double tcpusolve=Teuchos::Time::wallTime();

      if (isadapttol and itnum>1)
      {
        double currresidual = std::max(vresnorm_,presnorm_);
        currresidual = std::max(currresidual,incvelnorm_L2_/velnorm_L2_);
        currresidual = std::max(currresidual,incprenorm_L2_/prenorm_L2_);
        solver_->AdaptTolerance(ittol,currresidual,adaptolbetter);
      }

      if (updateprojection_)
      {
        UpdateKrylovSpaceProjection();
      }

      // if Krylov space projection is used, check whether constant pressure
      // is in nullspace of sysmat_
      // !!! only done for FEM since for NURBS- and meshfree-approximations,
      //     the integration error can already disturb matrix nullspace too
      //     much for sensitive problems
      if (DRT::Problem::Instance()->SpatialApproximation()=="Polynomial")
        CheckMatrixNullspace();

      if (msht_== INPAR::FLUID::no_meshtying)
      {
        // scale system prior to solver call
        if (fluid_infnormscaling_!= Teuchos::null)
          fluid_infnormscaling_->ScaleSystem(sysmat_, *residual_);

        bool resetSolver = (itnum == 1);
        if (step_%10 != 0 && (uprestart_ == 0 || step_%uprestart_!=1) && step_>3)
          resetSolver = false;

         // solve the system
        solver_->Solve(sysmat_->EpetraOperator(),incvel_,residual_,true,resetSolver, projector_);

        // unscale solution
        if (fluid_infnormscaling_!= Teuchos::null)
          fluid_infnormscaling_->UnscaleSolution(sysmat_, *incvel_,*residual_);
      }
      else
        meshtying_->SolveMeshtying(*solver_, sysmat_, incvel_, residual_, velnp_, itnum, projector_);

      solver_->ResetTolerance();

      dtsolve_ = Teuchos::Time::wallTime()-tcpusolve;
    }

    // -------------------------------------------------------------------
    // update within iteration
    // -------------------------------------------------------------------
    IterUpdate(incvel_);

    // -------------------------------------------------------------------
    // convergence check
    // -------------------------------------------------------------------
    stopnonliniter = ConvergenceCheck(itnum,itmax,ittol);

    // -------------------------------------------------------------------
    // free surface flow update
    // -------------------------------------------------------------------
    FreeSurfaceFlowUpdate();

    // -------------------------------------------------------------------
    // local lagrange conditions update
    // -------------------------------------------------------------------
    LocalLagrangeUpdate();
  }

  // -------------------------------------------------------------------
  // recompute residual (i.e., residual belonging to the final solution)
  // -------------------------------------------------------------------
  if (not inconsistent_)
  {
    AssembleMatAndRHS();

    if (locsysman_ != Teuchos::null) {
        // Transform newly built residual to local coordinate system
        // in order to later properly erase the lines containing
        // Dirichlet conditions in function ConvergenceCheck()
        locsysman_->RotateGlobalToLocal(residual_, false);
     }

    // prepare meshtying system
    if (msht_ != INPAR::FLUID::no_meshtying)
      meshtying_->PrepareMeshtyingSystem(sysmat_,residual_,velnp_);

    // print to screen
    ConvergenceCheck(0,itmax,ittol);
  }
} // FluidImplicitTimeInt::Solve

/*----------------------------------------------------------------------*
 | preparatives for solver                                     vg 09/11 |
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::PrepareSolve()
{
  // call elements to calculate system matrix and rhs and assemble
  AssembleMatAndRHS();

  // prepare meshtying system
  if (msht_ != INPAR::FLUID::no_meshtying)
    meshtying_->PrepareMeshtyingSystem(sysmat_,residual_,velnp_);

  // update local coordinate systems for ALE fluid case
  // (which may be time and displacement dependent)
  if ((locsysman_ != Teuchos::null) && (alefluid_)){
    discret_->ClearState();
    discret_->SetState("dispnp", dispnp_);
    locsysman_->Setup(time_);
    discret_->ClearState();
  }

  // apply Dirichlet boundary conditions to system of equations
  ApplyDirichletToSystem();
} // FluidImplicitTimeInt::PrepareSolve


/*----------------------------------------------------------------------*
 | call elements to calculate system matrix/rhs and assemble   vg 02/09 |
 | overloaded in TimIntRedModels                               bk 12/13 |
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::AssembleMatAndRHS()
{
  dtele_    = 0.0;
  dtfilter_ = 0.0;

  // time measurement: element
  TEUCHOS_FUNC_TIME_MONITOR("      + element calls");

  // get cpu time
  const double tcpu=Teuchos::Time::wallTime();

  sysmat_->Zero();

  if (shapederivatives_ != Teuchos::null)
    shapederivatives_->Zero();

  // set old residual to zero and add Neumann loads
  residual_->Update(1.0,*neumann_loads_,0.0);

  // add external loads
  if(external_loads_ != Teuchos::null)
    residual_->Update(1.0/ResidualScaling(),*external_loads_,1.0);

  //----------------------------------------------------------------------
  // evaluate elements
  //----------------------------------------------------------------------

  discret_->ClearState();

  // create the parameters for the discretization
  Teuchos::ParameterList eleparams;

  //----------------------------------------------------------------------
  // set general vector values needed by elements
  //----------------------------------------------------------------------

  //set parameters for turbulence models
  TreatTurbulenceModels(eleparams);

  //set additional pseudo-porosity field for topology optimization
  //set thermodynamic pressures
  //set parameters for poro
  //set parameters for HDG
  SetCustomEleParamsAssembleMatAndRHS(eleparams);

  discret_->SetState("hist" ,hist_ );
  discret_->SetState("veln" ,veln_ );
  discret_->SetState("accam",accam_);
  discret_->SetState("scaaf",scaaf_);
  discret_->SetState("scaam",scaam_);
  if (alefluid_)
  {
    discret_->SetState("dispnp", dispnp_);
    discret_->SetState("gridv", gridv_);
  }

  // set scheme-specific element parameters and vector values
  SetStateTimInt();

  // set external volume force (required, e.g., for forced homogeneous isotropic turbulence)
  if (DRT::INPUT::IntegralValue<INPAR::FLUID::ForcingType>(params_->sublist("TURBULENCE MODEL"),"FORCING_TYPE")
      == INPAR::FLUID::fixed_power_input)
  {
    // calculate required forcing
    homisoturb_forcing_->CalculateForcing(step_);
    homisoturb_forcing_->ActivateForcing(true);
  }

  if (homisoturb_forcing_ != Teuchos::null)
    homisoturb_forcing_->UpdateForcing(step_);

  if (forcing_!=Teuchos::null)
  {
    eleparams.set("forcing",true);
    discret_->SetState("forcing",forcing_);
  }

  //----------------------------------------------------------------------
  // AVM3-based solution approach if required
  //----------------------------------------------------------------------
  if (fssgv_ != INPAR::FLUID::no_fssgv)
    AVM3Separation();

  //----------------------------------------------------------------------
  // multifractal subgrid-scale modeling
  //----------------------------------------------------------------------
  if (turbmodel_==INPAR::FLUID::multifractal_subgrid_scales)
  {
    this->ApplyScaleSeparationForLES();
    discret_->SetState("fsscaaf",fsscaaf_);
  }

  //----------------------------------------------------------------------
  // Add further problem dependent vectors
  //----------------------------------------------------------------------
  AddProblemDependentVectors();

  // call standard loop over elements
  discret_->Evaluate(eleparams,sysmat_,shapederivatives_,residual_,Teuchos::null,Teuchos::null);
  ClearStateAssembleMatAndRHS();

  //----------------------------------------------------------------------
  // add potential edge-based stabilization terms
  //----------------------------------------------------------------------
  AssembleEdgeBasedMatandRHS();

  //----------------------------------------------------------------------
  // application of potential nonlinear boundary conditions to system
  //----------------------------------------------------------------------
  if (nonlinearbc_) ApplyNonlinearBoundaryConditions();

  //----------------------------------------------------------------------
  // update surface tension (free surface flow only)
  //----------------------------------------------------------------------
  FreeSurfaceFlowSurfaceTensionUpdate();

  // scaling to get true residual vector
  trueresidual_->Update(ResidualScaling(),*residual_,0.0);

  // finalize the complete matrix
  sysmat_->Complete();

  if (shapederivatives_ != Teuchos::null)
  {
    shapederivatives_->Complete();
    // apply Dirichlet conditions to a non-diagonal matrix
    // (The Dirichlet rows will become all zero, no diagonal one.)
    shapederivatives_->ApplyDirichlet(*(dbcmaps_->CondMap()),false);
  }

  // end time measurement for element
  dtele_=Teuchos::Time::wallTime()-tcpu;

} // FluidImplicitTimeInt::AssembleMatAndRHS

/*----------------------------------------------------------------------------*
 | Evaluate mass matrix                                       mayr.mt 05/2014 |
 *----------------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::EvaluateMassMatrix()
{
  massmat_->Zero();

  // create the parameters for the discretization
  Teuchos::ParameterList eleparams;

  // action for elements
  eleparams.set<int>("action", FLD::calc_mass_matrix);

  if (alefluid_)
  {
    discret_->SetState("dispnp", dispnp_);
  }

  discret_->Evaluate(eleparams, massmat_, Teuchos::null, Teuchos::null, Teuchos::null, Teuchos::null);
  discret_->ClearState();

  // finalize the complete matrix
  massmat_->Complete();

  return;
}

/*----------------------------------------------------------------------*|
 | Set Eleparams for turbulence models                          bk 12/13 |
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::TreatTurbulenceModels(Teuchos::ParameterList& eleparams)
{
  //----------------------------------------------------------------------
  // apply filter for turbulence models (only if required)
  //----------------------------------------------------------------------
   if (turbmodel_==INPAR::FLUID::dynamic_smagorinsky
    or turbmodel_ == INPAR::FLUID::scale_similarity
    or turbmodel_ == INPAR::FLUID::scale_similarity_basic
    or turbmodel_ == INPAR::FLUID::dynamic_vreman)
  {
    //compute filtered velocity
    // time measurement
    const double tcpufilter=Teuchos::Time::wallTime();
    this->ApplyScaleSeparationForLES();
    dtfilter_=Teuchos::Time::wallTime()-tcpufilter;
  }

   // set action type
   eleparams.set<int>("action",FLD::calc_fluid_systemmat_and_residual);
   eleparams.set<int>("physical type",physicaltype_);
  // parameters for turbulence model
  // TODO: rename list
  eleparams.sublist("TURBULENCE MODEL") = params_->sublist("TURBULENCE MODEL");
//      if (turbmodel_==INPAR::FLUID::scale_similarity
//       or turbmodel_ == INPAR::FLUID::multifractal_subgrid_scales)
   if (turbmodel_==INPAR::FLUID::scale_similarity
    or turbmodel_ == INPAR::FLUID::scale_similarity_basic)
  {
    eleparams.set("Filtered velocity",filteredvel_);
    eleparams.set("Fine scale velocity",finescalevel_);
    eleparams.set("Filtered reynoldsstress",filteredreystr_);
  }
  if (turbmodel_==INPAR::FLUID::dynamic_vreman)
  {
    double cv=0.0;
    cv=params_->get<double>("C_vreman");
    eleparams.set<double>("C_vreman", cv );
  }
  return;
}

/*----------------------------------------------------------------------*
 | application of nonlinear boundary conditions to system, such as      |
 | 1) Neumann inflow boundary conditions                                |
 | 2) flow-dependent pressure boundary conditions                       |
 | 3) weak Dirichlet boundary conditions                                |
 | 4) mixed/hybrid Dirichlet boundary conditions                        |
 | 5) Slip Supplemental Curved Boundary conditions                      |
 | 6) Navier-slip boundary conditions                                   |
 |                                                             vg 06/13 |
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::ApplyNonlinearBoundaryConditions()
{
  //----------------------------------------------------------------------
  // 1) Neumann inflow boundary conditions
  //----------------------------------------------------------------------
  // check whether there are Neumann inflow boundary conditions
  std::vector<DRT::Condition*> neumanninflow;
  discret_->GetCondition("FluidNeumannInflow",neumanninflow);

  if (neumanninflow.size() != 0)
  {
    // create parameter list
    Teuchos::ParameterList neuminparams;

    // set action for elements
    neuminparams.set<int>("action",FLD::calc_Neumann_inflow);

    // set thermodynamic pressure
    SetCustomEleParamsApplyNonlinearBoundaryConditions(neuminparams);

    // set required state vectors
    // (no contribution due to pressure or continuity equation for Neumann inflow
    // -> no difference between af_genalpha and np_genalpha)
    discret_->ClearState();
    discret_->SetState("scaaf",scaaf_);
    SetStateTimInt();
    if (alefluid_) discret_->SetState("dispnp",dispnp_);

    // evaluate all Neumann inflow boundary conditions
    discret_->EvaluateCondition(neuminparams,sysmat_,Teuchos::null,
                                residual_,Teuchos::null,Teuchos::null,
                                "FluidNeumannInflow");

    // clear state
    discret_->ClearState();
  }

  //----------------------------------------------------------------------
  // 2) flow-dependent pressure boundary conditions
  //    (either based on (out)flow rate or on (out)flow volume (e.g.,
  //     for air cushion outside of boundary))
  //----------------------------------------------------------------------
  // check whether there are flow-dependent pressure boundary conditions
  std::vector<DRT::Condition*> flowdeppressureline;
  discret_->GetCondition("LineFlowDepPressure",flowdeppressureline);
  std::vector<DRT::Condition*> flowdeppressuresurf;
  discret_->GetCondition("SurfaceFlowDepPressure",flowdeppressuresurf);

  if (flowdeppressureline.size() != 0 or flowdeppressuresurf.size() != 0)
  {
    // get a vector layout from the discretization to construct matching
    // vectors and matrices local <-> global dof numbering
    const Epetra_Map* dofrowmap = discret_->DofRowMap();

    // decide on whether it is a line or a surface condition and
    // set condition name accordingly
    std::string fdpcondname;
    if (flowdeppressureline.size() != 0)      fdpcondname = "LineFlowDepPressure";
    else if (flowdeppressuresurf.size() != 0) fdpcondname = "SurfaceFlowDepPressure";
    else dserror("Line and surface flow-dependent pressure boundary conditions simultaneously prescribed!");

    // get condition vector
    std::vector<DRT::Condition*> fdpcond;
    discret_->GetCondition(fdpcondname,fdpcond);

    // define vectors for flow rate and volume for actual evaluation of boundary
    // conditions according to time-integration scheme and potential relaxation
    // within nonlinear iteration loop
    // (relaxation parameter 1.0, for the time being, that is, no relaxation)
    LINALG::Matrix<4,1> flowraterel(true);
    LINALG::Matrix<4,1> flowvolumerel(true);
    const double relaxpara = 1.0;

    double timefac = 1.0;
    timefac=SetTimeFac();

    // assign ID to all conditions
    for (int fdpcondid = 0; fdpcondid < (int) fdpcond.size(); fdpcondid++)
    {
      // check for already existing ID and add ID
      const std::vector<int>* fdpcondidvec = fdpcond[fdpcondid]->Get<std::vector<int> >("ConditionID");
      if (fdpcondidvec)
      {
        if ((*fdpcondidvec)[0] != fdpcondid) dserror("Flow-dependent pressure condition %s has non-matching ID",fdpcondname.c_str());
      }
      else fdpcond[fdpcondid]->Add("ConditionID",fdpcondid);
    }

    // create or append to output file
    if (myrank_ == 0)
    {
      const std::string fname1 = DRT::Problem::Instance()->OutputControlFile()->FileName()+".fdpressure";

      std::ofstream f1;

      // create file for output in first time step or append to existing file
      // in subsequent time steps
      if (step_ <= 1)
      {
        f1.open(fname1.c_str(),std::fstream::trunc);
        f1 << "#| Step | Time |";
        for (int fdpcondid = 0; fdpcondid < (int) fdpcond.size(); fdpcondid++)
        {
          f1 << " Flow rate " << fdpcondid << " | Flow volume " << fdpcondid << " | Mean pressure " << fdpcondid << " |";
        }
        f1 <<"\n";
      }
      else f1.open(fname1.c_str(),std::fstream::ate | std::fstream::app);

      // write step number and time
      f1 << step_ << " " << time_ << " ";
    }

    //----------------------------------------------------------------------
    // compute
    // a) flow rate (current value and value used for evaluation of bc)
    // b) flow volume (current value and value used for evaluation of bc)
    // c) surface area,
    // d) pressure integral, and
    // e) mean pressure
    // for each flow-dependent pressure boundary condition
    //----------------------------------------------------------------------
    for (int fdpcondid = 0; fdpcondid < (int) fdpcond.size(); fdpcondid++)
    {
      // create parameter list
      Teuchos::ParameterList flowdeppressureparams;

      // initialization of values at first time step
      if (step_ <= 1)
      {
        flowraten_(fdpcondid) = 0.0;
        flowratenm_(fdpcondid) = 0.0;
        flowvolumen_(fdpcondid) = 0.0;
        flowvolumenm_(fdpcondid) = 0.0;
      }

      //--------------------------------------------------------------------
      // a) flow rate
      //--------------------------------------------------------------------
      // set action for elements
      flowdeppressureparams.set<int>("action",FLD::calc_flowrate);

      // create vector and initialize with zeros
      Teuchos::RCP<Epetra_Vector> flowrates = LINALG::CreateVector(*dofrowmap,true);

      // set required state vectors (no ALE so far)
      discret_->ClearState();
      SetStateTimInt();
      //if (alefluid_) discret_->SetState("dispnp",dispnp_);

      // evaluate flow rate
      discret_->EvaluateCondition(flowdeppressureparams,flowrates,fdpcondname,fdpcondid);

      // sum up local flow rate on this processor
      double local_flowrate = 0.0;
      for (int i=0; i < dofrowmap->NumMyElements(); i++)
      {
        local_flowrate +=((*flowrates)[i]);
      }

      // sum up global flow rate over all processors and set to global value
      double flowrate = 0.0;
      dofrowmap->Comm().SumAll(&local_flowrate,&flowrate,1);

      // set current flow rate
      flowratenp_(fdpcondid) = flowrate;

      // compute flow rate used for evaluation of boundary condition below
      flowraterel(fdpcondid) = (1.0-timefac)*flowraten_(fdpcondid) + timefac*((1.0-relaxpara)*flowratenpi_(fdpcondid) + relaxpara*flowratenp_(fdpcondid));

      // clear state
      discret_->ClearState();

      //--------------------------------------------------------------------
      // b) flow volume
      //--------------------------------------------------------------------
      // compute current flow volume as integral of flow rate according to
      // trapezoidal rule
      flowvolumenp_(fdpcondid) = flowvolumen_(fdpcondid) + 0.5*dta_*(flowratenp_(fdpcondid)+flowraten_(fdpcondid));

      // set current flow volume to zero if value smaller than zero,
      // meaning that no flow volume may be sucked in from outside
      if (flowvolumenp_(fdpcondid) < 0.0) flowvolumenp_(fdpcondid) = 0.0;

      // compute flow volume used for evaluation of boundary condition below
      flowvolumerel(fdpcondid) = (1.0-timefac)*flowvolumen_(fdpcondid) + timefac*((1.0-relaxpara)*flowvolumenpi_(fdpcondid) + relaxpara*flowvolumenp_(fdpcondid));

      //--------------------------------------------------------------------
      // c) surface area
      //--------------------------------------------------------------------
      // set action and parameter for elements
      flowdeppressureparams.set<int>("action",FLD::calc_area);
      flowdeppressureparams.set<double>("area",0.0);

      // set required state vectors (no ALE so far)
      //if (alefluid_) discret_->SetState("dispnp",dispnp_);

      // evaluate surface area
      discret_->EvaluateCondition(flowdeppressureparams,fdpcondname,fdpcondid);

      // sum up local surface area on this processor
      double localarea = flowdeppressureparams.get<double>("area");

      // sum up global surface area over all processors
      double area = 0.0;
      discret_->Comm().SumAll(&localarea,&area,1);

      // clear state
      discret_->ClearState();

      //--------------------------------------------------------------------
      // d) pressure integral
      //--------------------------------------------------------------------
      // set action for elements
      flowdeppressureparams.set<int>("action",FLD::calc_pressure_bou_int);
      flowdeppressureparams.set<double>("pressure boundary integral",0.0);

      // set required state vectors (no ALE so far)
      discret_->ClearState();
      SetStateTimInt();
      //if (alefluid_) discret_->SetState("dispnp",dispnp_);

      // evaluate pressure integral
      discret_->EvaluateCondition(flowdeppressureparams,fdpcondname,fdpcondid);

      // sum up local pressure integral on this processor
      double localpressint = flowdeppressureparams.get<double>("pressure boundary integral");

      // sum up global pressure integral over all processors
      double pressint = 0.0;
      discret_->Comm().SumAll(&localpressint,&pressint,1);

      // clear state
      discret_->ClearState();

      //--------------------------------------------------------------------
      // e) mean pressure
      //--------------------------------------------------------------------
      const double meanpressure = pressint/area;

      // append values to output file
      if (myrank_ == 0)
      {
        const std::string fname1 = DRT::Problem::Instance()->OutputControlFile()->FileName()+".fdpressure";

        std::ofstream f1;
        f1.open(fname1.c_str(),std::fstream::ate | std::fstream::app);

        f1 << flowratenp_(fdpcondid) << " " << flowvolumenp_(fdpcondid) << " " << meanpressure << "   ";
      }
    }

    // append values to output file
    if (myrank_ == 0)
    {
      const std::string fname1 = DRT::Problem::Instance()->OutputControlFile()->FileName()+".fdpressure";

      std::ofstream f1;
      f1.open(fname1.c_str(),std::fstream::ate | std::fstream::app);

      f1 << "\n";
      f1.flush();
      f1.close();
    }

    //----------------------------------------------------------------------
    // evaluate flow-dependent pressure boundary conditions
    // (proceeds in separate loop to enable, e.g., implementation of
    //  flow-rate sums of more than one condition in between)
    //----------------------------------------------------------------------
    for (int fdpcondid = 0; fdpcondid < (int) fdpcond.size(); fdpcondid++)
    {
      // create parameter list
      Teuchos::ParameterList flowdeppressureparams;

      // set action for elements
      flowdeppressureparams.set<int>("action",FLD::flow_dep_pressure_bc);

      // set required state vectors (no ALE so far)
      SetStateTimInt();
      if (alefluid_) discret_->SetState("dispnp",dispnp_);

      // set values for elements
      flowdeppressureparams.set<LINALG::Matrix<4,1> >("flow rate",flowraterel);
      flowdeppressureparams.set<LINALG::Matrix<4,1> >("flow volume",flowvolumerel);

      // evaluate all flow-dependent pressure boundary conditions
      discret_->EvaluateCondition(flowdeppressureparams,sysmat_,Teuchos::null,
                                  residual_,Teuchos::null,Teuchos::null,
                                  fdpcondname,fdpcondid);

      // clear state
      discret_->ClearState();

      // update iteration values
      flowratenpi_(fdpcondid)   = flowratenp_(fdpcondid);
      flowvolumenpi_(fdpcondid) = flowvolumenp_(fdpcondid);
    }
  }

  //----------------------------------------------------------------------
  // 3) weak Dirichlet boundary conditions
  //----------------------------------------------------------------------
  // check whether there are weak Dirichlet boundary conditions
  std::vector<DRT::Condition*> weakdbcline;
  discret_->GetCondition("LineWeakDirichlet",weakdbcline);
  std::vector<DRT::Condition*> weakdbcsurf;
  discret_->GetCondition("SurfaceWeakDirichlet",weakdbcsurf);

  if (weakdbcline.size() != 0 or weakdbcsurf.size() != 0)
  {
    // create parameter list
    Teuchos::ParameterList weakdbcparams;

    // set action for elements
    weakdbcparams.set<int>("action", FLD::enforce_weak_dbc);

    // set required state vectors
    SetStateTimInt();
    if (alefluid_) discret_->SetState("dispnp",dispnp_);

    // evaluate all line weak Dirichlet boundary conditions
    discret_->EvaluateCondition(weakdbcparams,sysmat_,Teuchos::null,
                                residual_,Teuchos::null,Teuchos::null, "LineWeakDirichlet");

    // evaluate all surface weak Dirichlet boundary conditions
    discret_->EvaluateCondition(weakdbcparams, sysmat_, Teuchos::null,
                               residual_, Teuchos::null, Teuchos::null,
                               "SurfaceWeakDirichlet");

    // clear state
    discret_->ClearState();
  }

  //----------------------------------------------------------------------
  // 4) mixed/hybrid Dirichlet boundary conditions
  //----------------------------------------------------------------------
  // check whether there are mixed/hybrid Dirichlet boundary conditions
  std::vector<DRT::Condition*> mhdbcline;
  discret_->GetCondition("LineMixHybDirichlet",mhdbcline);
  std::vector<DRT::Condition*> mhdbcsurf;
  discret_->GetCondition("SurfaceMixHybDirichlet",mhdbcsurf);

  if (mhdbcline.size() != 0 or mhdbcsurf.size() != 0)
  {
    // create parameter list
    Teuchos::ParameterList mhdbcparams;

    // set action for elements
    mhdbcparams.set<int>("action", FLD::mixed_hybrid_dbc);

    // set required state vectors
    SetStateTimInt();
    if (alefluid_) discret_->SetState("dispnp",dispnp_);

    // evaluate all line mixed/hybrid Dirichlet boundary conditions
    discret_->EvaluateCondition(mhdbcparams, sysmat_, Teuchos::null,
                                residual_, Teuchos::null, Teuchos::null, "LineMixHybDirichlet");

    // evaluate all surface mixed/hybrid Dirichlet boundary conditions
    // with checking of parallel redistribution of boundary elements
    if (MHD_evaluator_ != Teuchos::null)
    {
      dserror("Redistributed MHD evaluation needs to be verified");
      MHD_evaluator_->BoundaryElementLoop(mhdbcparams, velaf_, velnp_,
                                          residual_, SystemMatrix());
    }
    else discret_->EvaluateCondition(mhdbcparams, sysmat_, Teuchos::null,
                                     residual_, Teuchos::null, Teuchos::null,
                                     "SurfaceMixHybDirichlet");

    // clear state
    discret_->ClearState();
  }

  //------------------------------------------------------------------------
  // 5) Slip Supplemental Curved Boundary conditions            [hahn 07/14]
  //    (Boundary condition used for counteracting spurious velocities at
  //     curved boundaries with slip-conditions. For details see Behr M.,
  //     2003, "On the Application of Slip Boundary Condition on Curved
  //     Boundaries" and Coppola-Owen H. & Codina R., 2011, "A free surface
  //     finite element model for low Froude number mould filling problems
  //     on fixed meshes".)
  //------------------------------------------------------------------------

  // check whether there are Slip Supplemental Curved Boundary conditions
  std::vector<DRT::Condition*> slipsuppline;
  discret_->GetCondition("LineSlipSupp",slipsuppline);
  std::vector<DRT::Condition*> slipsuppsurf;
  discret_->GetCondition("SurfaceSlipSupp",slipsuppsurf);

  if (slipsuppline.size() != 0 or slipsuppsurf.size() != 0)
  {
    // get a vector layout from the discretization to construct matching
    // vectors and matrices local <-> global dof numbering
    const Epetra_Map* dofrowmap = discret_->DofRowMap();

    // initialize global slip bc normal traction variable
    slip_bc_normal_tractions_ = LINALG::CreateVector(*dofrowmap,true);

    // decide on whether it is a line or a surface condition and set condition
    // name accordingly. Both types simultaneously is not supported
    if ((slipsuppline.size() != 0) && (slipsuppsurf.size() != 0))
      dserror("Line and surface slip supplemental curved boundary conditions simultaneously prescribed!");

    std::string sscbcondname;
    if (slipsuppline.size() != 0)      sscbcondname = "LineSlipSupp";
    else if (slipsuppsurf.size() != 0) sscbcondname = "SurfaceSlipSupp";

    // get condition vector
    std::vector<DRT::Condition*> sscbcond;
    discret_->GetCondition(sscbcondname,sscbcond);

    // assign ID to all conditions
    for (int sscbcondid = 0; sscbcondid < (int) sscbcond.size(); sscbcondid++)
    {
      // check for already existing ID and add ID
      const std::vector<int>* sscbcondidvec = sscbcond[sscbcondid]->Get<std::vector<int> >("ConditionID");
      if (sscbcondidvec)
      {
        if ((*sscbcondidvec)[0] != sscbcondid) dserror("Slip Supplemental Curved Boundary condition %s has non-matching ID",sscbcondname.c_str());
      }
      else sscbcond[sscbcondid]->Add("ConditionID",sscbcondid);
    }

    //----------------------------------------------------------------------
    // evaluate slip supplemental curved boundary conditions
    //----------------------------------------------------------------------

    for (int sscbcondid = 0; sscbcondid < (int) sscbcond.size(); sscbcondid++)
    {
      // Evaluate condition
      // ******************************************************************
      // create parameter list
      Teuchos::ParameterList slipsuppparams;

      // set action for elements
      slipsuppparams.set<int>("action",FLD::slip_supp_bc);

      // set required state vectors
      SetStateTimInt();
      if (alefluid_) discret_->SetState("dispnp",dispnp_);
      //discret_->SetState("nodenormal",nodeNormal);

      // temporary variable holding the scaled residual contribution
      Teuchos::RCP<Epetra_Vector>    slip_bc_normal_tractions_scaled;
      slip_bc_normal_tractions_scaled = LINALG::CreateVector(*dofrowmap,true);

      // evaluate all slip supplemental curved boundary conditions
      discret_->EvaluateCondition(slipsuppparams,sysmat_,Teuchos::null,
                                  slip_bc_normal_tractions_scaled,Teuchos::null,Teuchos::null,
                                  sscbcondname,sscbcondid);

      // Update residual vector
      residual_->Update(1.0, *slip_bc_normal_tractions_scaled, 1.0);

      // Add to tractions vector
      slip_bc_normal_tractions_->Update((-1)*ResidualScaling(), *slip_bc_normal_tractions_scaled, 1.0);

      // clear state
      discret_->ClearState();
    }
  }

  //------------------------------------------------------------------------
  // 6) Navier-slip boundary conditions                         [hahn 03/14]
  //    At the boundary, apply a shear stress which is proportional to the
  //    tangential/bi-tangential velocity. In BACI, this is achieved by
  //    applying h = sigma*n = -beta*u under the condition that u*n=0 has
  //    been set as Dirichlet BC! For details on the Navier slip condition
  //    please refer to e.g. Behr M., 2003, "On the Application of Slip
  //    Boundary Condition on Curved Boundaries".
  //------------------------------------------------------------------------

  // check whether there are navier-slip boundary conditions
  std::vector<DRT::Condition*> navierslipline;
  discret_->GetCondition("LineNavierSlip",navierslipline);
  std::vector<DRT::Condition*> navierslipsurf;
  discret_->GetCondition("SurfNavierSlip",navierslipsurf);

  if (navierslipline.size() != 0 or navierslipsurf.size() != 0)
  {
    // decide on whether it is a line or a surface condition and set condition
    // name accordingly. Both types simultaneously is not supported
    if ((navierslipline.size() != 0) && (navierslipsurf.size() != 0))
      dserror("Line and surface Navier slip boundary conditions simultaneously prescribed!");

    std::string nscondname;
    if (navierslipline.size() != 0)      nscondname = "LineNavierSlip";
    else if (navierslipsurf.size() != 0) nscondname = "SurfNavierSlip";

    // get condition vector
    std::vector<DRT::Condition*> nscond;
    discret_->GetCondition(nscondname,nscond);

    // assign ID to all conditions
    for (int nscondid = 0; nscondid < (int) nscond.size(); nscondid++)
    {
      // check for already existing ID and add ID
      const std::vector<int>* nscondidvec = nscond[nscondid]->Get<std::vector<int> >("ConditionID");
      if (nscondidvec)
      {
        if ((*nscondidvec)[0] != nscondid) dserror("Navier slip boundary condition %s has non-matching ID",nscondname.c_str());
      }
      else nscond[nscondid]->Add("ConditionID",nscondid);
    }

    //----------------------------------------------------------------------
    // evaluate navier slip boundary conditions
    //----------------------------------------------------------------------

    for (int nscondid = 0; nscondid < (int) nscond.size(); nscondid++)
    {
      // create parameter list
      Teuchos::ParameterList navierslipparams;

      // set action for elements
      navierslipparams.set<int>("action",FLD::navier_slip_bc);

      // set required state vectors
      SetStateTimInt();
      if (alefluid_) discret_->SetState("dispnp",dispnp_);

      // set slip coefficient
      DRT::Condition* currnavierslip = nscond[nscondid];
      const double beta = currnavierslip->GetDouble("slipcoefficient");
      navierslipparams.set<double>("beta",beta);

      // evaluate navier slip boundary condition
      discret_->EvaluateCondition(navierslipparams,sysmat_,Teuchos::null,
                                  residual_,Teuchos::null,Teuchos::null,
                                  nscondname,nscondid);

      // clear state
      discret_->ClearState();
    }
  }

  return;
}


/*----------------------------------------------------------------------*
 | add potential edge-based stabilization terms         rasthofer 06/13 |
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::AssembleEdgeBasedMatandRHS()
{
  // add edged-based stabilization, if selected
  if(params_->sublist("RESIDUAL-BASED STABILIZATION").get<std::string>("STABTYPE")=="edge_based")
  {
    // set the only required state vectors
    SetStateTimInt();

     if (alefluid_)
     {
       dserror("Edge-based stabilization not yet supported for ale problems!");
     }

     facediscret_->EvaluateEdgeBased(sysmat_,residual_);

     discret_->ClearState();
  }

  return;
}


/*----------------------------------------------------------------------*
 | update surface tension                               rasthofer 06/13 |
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::FreeSurfaceFlowSurfaceTensionUpdate()
{
  if (alefluid_ and surfacesplitter_->FSCondRelevant())
  {
    // employs the divergence theorem acc. to Saksono eq. (24) and does
    // not require second derivatives.

    // select free surface elements
    std::string condname = "FREESURFCoupling";

    Teuchos::ParameterList eleparams;

    // set action for elements
    eleparams.set<int>("action", FLD::calc_surface_tension);

    discret_->ClearState();
    discret_->SetState("dispnp", dispnp_);
    discret_->EvaluateCondition(eleparams, Teuchos::null,
                                Teuchos::null, residual_, Teuchos::null,
                                Teuchos::null, condname);
    discret_->ClearState();
  }
  return;
}


/*----------------------------------------------------------------------*
 | application of Dirichlet boundary conditions to system      vg 09/11 |
 | overloaded in TimIntRedModels                              bk 12/13 |
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::ApplyDirichletToSystem()
{
  // -------------------------------------------------------------------
  // apply Dirichlet boundary conditions to system of equations:
  // - Residual displacements are supposed to be zero for resp. dofs.
  // - Time for applying Dirichlet boundary conditions is measured.
  // -------------------------------------------------------------------
  incvel_->PutScalar(0.0);

  Teuchos::RCP<const LINALG::SparseMatrix> locsysTrafo = Teuchos::null;

  if (locsysman_ != Teuchos::null) {
     // Transform system matrix and rhs to local co-ordinate systems
     locsysman_->RotateGlobalToLocal(SystemMatrix(), residual_);

     locsysTrafo = locsysman_->Trafo();
  }

  {
    TEUCHOS_FUNC_TIME_MONITOR("      + apply DBC");
    LINALG::ApplyDirichlettoSystem(sysmat_,incvel_,residual_,locsysTrafo,zeros_,*(dbcmaps_->CondMap()));
  }

} // FluidImplicitTimeInt::ApplyDirichletToSystem


void FLD::FluidImplicitTimeInt::InitKrylovSpaceProjection()
{
  // get condition "KrylovSpaceProjection" from discretization
  std::vector<DRT::Condition*> KSPcond;
  discret_->GetCondition("KrylovSpaceProjection",KSPcond);
  int numcond = KSPcond.size();
  int numfluid = 0;

  DRT::Condition* kspcond = NULL;
  // check if for fluid Krylov projection is required
  for(int icond = 0; icond < numcond; icond++)
  {
    const std::string* name = KSPcond[icond]->Get<std::string>("discretization");
    if (*name == "fluid")
    {
      numfluid++;
      kspcond = KSPcond[icond];
    }
  }

  // initialize variables for Krylov projection if necessary
  if (numfluid == 1)
  {
    SetupKrylovSpaceProjection(kspcond);
    if (myrank_ == 0)
      std::cout << "\nSetup of KrylovSpaceProjection in fluid field\n" << std::endl;
  }
  else if (numfluid == 0)
  {
    updateprojection_ = false;
    projector_ = Teuchos::null;
  }
  else
    dserror("Received more than one KrylovSpaceCondition for fluid field");
  return;
}

/*--------------------------------------------------------------------------*
 | setup Krylov projector including first fill                    nis Feb13 |
 *--------------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::SetupKrylovSpaceProjection(DRT::Condition* kspcond)
{
  // confirm that mode flags are number of nodal dofs
  const int nummodes = kspcond->GetInt("NUMMODES");
  if (nummodes!=(numdim_+1))
    dserror("Expecting numdim_+1 modes in Krylov projection definition. Check dat-file!");

  // get vector of mode flags as given in dat-file
  const std::vector<int>* modeflags = kspcond->Get<std::vector<int> >("ONOFF");

  // confirm that only the pressure mode is selected for Krylov projection in dat-file
  for(int rr=0;rr<numdim_;++rr)
  {
    if(((*modeflags)[rr])!=0)
    {
      dserror("Expecting only an undetermined pressure. Check dat-file!");
    }
  }
  if(((*modeflags)[numdim_])!=1)
    dserror("Expecting an undetermined pressure. Check dat-file!");
  std::vector<int> activemodeids(1,numdim_);

  // allocate kspsplitter_
  kspsplitter_ = Teuchos::rcp(new FLD::UTILS::KSPMapExtractor());
  // create map of nodes involved in Krylov projection

  kspsplitter_->Setup(*discret_);

  // get from dat-file definition how weights are to be computed
  const std::string* weighttype = kspcond->Get<std::string>("weight vector definition");

  // set flag for projection update true only if ALE and integral weights
  if (alefluid_ and (*weighttype=="integration"))
    updateprojection_ = true;

  projector_ = Teuchos::rcp(new LINALG::KrylovProjector(activemodeids,weighttype,discret_->DofRowMap()));

  // update the projector
  UpdateKrylovSpaceProjection();

  return;

} // FLD::FluidImplicitTimeInt::SetupKrylovSpaceProjection


/*--------------------------------------------------------------------------*
 | update projection vectors w_ and c_ for Krylov projection      nis Feb13 |
 *--------------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::UpdateKrylovSpaceProjection()
{
  // get Teuchos::RCP to kernel vector of projector
  Teuchos::RCP<Epetra_MultiVector> c = projector_->GetNonConstKernel();
  Teuchos::RCP<Epetra_Vector> c0 = Teuchos::rcp((*c)(0),false);
  c0->PutScalar(0.0);
  // extract vector of pressure-dofs
  Teuchos::RCP<Epetra_Vector> presmode = velpressplitter_.ExtractCondVector(*c0);

  const std::string* weighttype = projector_->WeightType();
  Teuchos::RCP<Epetra_Vector> w0_update=Teuchos::null;
  // compute w_ as defined in dat-file
  if(*weighttype == "pointvalues")
  {
    /*
    // export to vector to normalize against
    // Note that in the case of definition pointvalue based,
    // the average pressure will vanish in a pointwise sense
    //
    //    +---+
    //     \
    //      +   p_i  = 0
    //     /
    //    +---+
    //
    // (everything is done below)
    */
  }
  else if(*weighttype == "integration")
  {
    // get Teuchos::RCP to weight vector of projector
    Teuchos::RCP<Epetra_MultiVector> w = projector_->GetNonConstWeights();
    Teuchos::RCP<Epetra_Vector> w0 = Teuchos::rcp((*w)(0),false);
    w0->PutScalar(0.0);

    // create parameter list for condition evaluate and ...
    Teuchos::ParameterList mode_params;
    // ... set action for elements to integration of shape functions
    mode_params.set<int>("action",FLD::integrate_shape);

    if (alefluid_)
    {
      discret_->SetState("dispnp",dispnp_);
    }

    /*
    // evaluate KrylovSpaceProjection condition in order to get
    // integrated nodal basis functions w_
    // Note that in the case of definition integration based,
    // the average pressure will vanish in an integral sense
    //
    //                    /              /                      /
    //   /    \          |              |  /          \        |  /    \
    //  | w_*p | = p_i * | N_i(x) dx =  | | N_i(x)*p_i | dx =  | | p(x) | dx = 0
    //   \    /          |              |  \          /        |  \    /
    //                   /              /                      /
    */

    // compute w_ by evaluating the integrals of all pressure basis functions
    discret_->EvaluateCondition
      (mode_params        ,
       Teuchos::null      ,
       Teuchos::null      ,
       w0                 ,
       Teuchos::null      ,
       Teuchos::null      ,
       "KrylovSpaceProjection");

    // adapt weight vector according to meshtying case
    if (msht_ != INPAR::FLUID::no_meshtying)
      w0_update=meshtying_->AdaptKrylovProjector(w0);


  }
  else
  {
    dserror("unknown definition of weight vector w for restriction of Krylov space");
  }

  // construct c by setting all pressure values to 1.0 and export to c
  presmode->PutScalar(1.0);
  Teuchos::RCP<Epetra_Vector> tmpc = LINALG::CreateVector(*(discret_->DofRowMap()),true);
  LINALG::Export(*presmode,*tmpc);
  Teuchos::RCP<Epetra_Vector> tmpkspc = kspsplitter_->ExtractKSPCondVector(*tmpc);
  LINALG::Export(*tmpkspc,*c0);
  // adapt kernel vector according to meshtying case

  if (msht_ != INPAR::FLUID::no_meshtying)
  {
    Teuchos::RCP<Epetra_Vector> c0_update;
    if(*weighttype != "integration")
      dserror("Fluidmeshtying supports only an integration - like Krylov projector");
    c0_update=meshtying_->AdaptKrylovProjector(c0);
    if(msht_ == INPAR::FLUID::condensed_bmat || msht_ == INPAR::FLUID::condensed_bmat_merged)
    {
      const Epetra_BlockMap* mergedmap = meshtying_->GetMergedMap();
      projector_->SetCW(c0_update,w0_update, mergedmap);
    }
    else
    {
      projector_->SetCW(c0_update,w0_update);
    }
  }

  // fillcomplete the projector to compute (w^T c)^(-1)
  projector_->FillComplete();
  return;

} // FluidImplicitTimeInt::UpdateKrylovSpaceProjection


/*--------------------------------------------------------------------------*
 | check if constant pressure mode is in kernel of sysmat_     nissen Jan13 |
 *--------------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::CheckMatrixNullspace()
{
  //Note: this check is expensive and should only be used in the debug mode
  if (projector_ != Teuchos::null)
  {
    Teuchos::RCP<Epetra_MultiVector> c = projector_->GetNonConstKernel();
    projector_->FillComplete();
    int nsdim = c->NumVectors();
    if (nsdim != 1)
      dserror("Only one mode, namely the constant pressure mode, expected.");

    Epetra_Vector result(c->Map(),false);

    sysmat_->Apply(*c,result);

    double norm=1e9;

    result.Norm2(&norm);

    if(norm>1e-12)
    {
      std::cout << "#####################################################" << std::endl;
      std::cout << "Nullspace check for sysmat_ failed!                  " << std::endl;
      std::cout << "This might be caused by:                             " << std::endl;
      std::cout << " - you don't have pure Dirichlet boundary conditions " << std::endl;
      std::cout << "   or pbcs. pressure level is fixed. -> check datfile" << std::endl;
      std::cout << " - you don't integrate pressure dofs accurately      " << std::endl;
      std::cout << "   enough for sysmat_. constant pressure is not in   " << std::endl;
      std::cout << "   kernel of sysmat_. -> use more gauss points (often" << std::endl;
      std::cout << "   problem with nurbs)                               " << std::endl;
      std::cout << " - unlikely but not impossible: nullspace vector is  " << std::endl;
      std::cout << "   not the constant pressure mode (not totally clear " << std::endl;
      std::cout << "   for xfem, yet). In this case sysmat_ could be     " << std::endl;
      std::cout << "   correct. -> adapt nullspace vector                " << std::endl;
      std::cout << "#####################################################" << std::endl;
      dserror("Nullspace check for sysmat_ failed, Ac returned %12.5e",norm);
    }
  }

  return;
}


/*----------------------------------------------------------------------*
 | update within iteration                                     vg 09/11 |
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::IterUpdate(
  const Teuchos::RCP<const Epetra_Vector> increment)
{
  // store incremental vector to be available for convergence check
  // if incremental vector is received from outside for coupled problems
  incvel_->Update(1.0,*increment,0.0);

  // update velocity and pressure values by adding increments
  velnp_->Update(1.0,*increment,1.0);

  // -------------------------------------------------------------------
  // For af-generalized-alpha: update accelerations
  // Furthermore, calculate velocities, pressures, scalars and
  // accelerations at intermediate time steps n+alpha_F and n+alpha_M,
  // respectively, for next iteration.
  // This has to be done at the end of the iteration, since we might
  // need the velocities at n+alpha_F in a potential coupling
  // algorithm, for instance.
  // -------------------------------------------------------------------
  GenAlphaUpdateAcceleration();
  GenAlphaIntermediateValues();


} // FluidImplicitTimeInt::IterUpdate

/*----------------------------------------------------------------------*
 | convergence check                                           vg 09/11 |
 *----------------------------------------------------------------------*/
bool FLD::FluidImplicitTimeInt::ConvergenceCheck(int          itnum,
                                                 int          itmax,
                                                 const double ittol)
{
  // -------------------------------------------------------------------
  // calculate and print out norms for convergence check
  // (blank residual DOFs which are on Dirichlet BC
  // We can do this because the values at the dirichlet positions
  // are not used anyway.
  // We could avoid this though, if velrowmap_ and prerowmap_ would
  // not include the dirichlet values as well. But it is expensive
  // to avoid that.)
  // -------------------------------------------------------------------
  dbcmaps_->InsertCondVector(dbcmaps_->ExtractCondVector(zeros_),residual_);

  // -------------------------------------------------------------------
  // take surface volumetric flow rate into account
  //    Teuchos::RCP<Epetra_Vector> temp_vec = Teuchos::rcp(new Epetra_Vector(*vol_surf_flow_bcmaps_,true));
  //    vol_surf_flow_bc_->InsertCondVector( *temp_vec , *residual_);
  // -------------------------------------------------------------------
  InsertVolumetricSurfaceFlowCondVector(zeros_,residual_);

  // remove contributions of pressure mode that would not vanish due to the
  // projection
  // In meshtying with block matrix, the projector might have another length
  // compared to residual. Thus, the projector is applied differently in this case.
  if (projector_ != Teuchos::null)
  {
    if (msht_ == INPAR::FLUID::condensed_bmat_merged or
        msht_ == INPAR::FLUID::condensed_bmat)
      meshtying_->ApplyPTToResidual(sysmat_,residual_,projector_);
    else
      projector_->ApplyPT(*residual_);
  }



  Teuchos::RCP<Epetra_Vector> onlyvel = velpressplitter_.ExtractOtherVector(residual_);

  onlyvel->Norm2(&vresnorm_);

  velpressplitter_.ExtractOtherVector(incvel_,onlyvel);

  onlyvel->Norm2(&incvelnorm_L2_);

  velpressplitter_.ExtractOtherVector(velnp_,onlyvel);

  onlyvel->Norm2(&velnorm_L2_);

  Teuchos::RCP<Epetra_Vector> onlypre = velpressplitter_.ExtractCondVector(residual_);
  onlypre->Norm2(&presnorm_);

  velpressplitter_.ExtractCondVector(incvel_,onlypre);
  onlypre->Norm2(&incprenorm_L2_);

  velpressplitter_.ExtractCondVector(velnp_,onlypre);
  onlypre->Norm2(&prenorm_L2_);

  // check for any INF's and NaN's
  if (std::isnan(vresnorm_) or
      std::isnan(incvelnorm_L2_) or
      std::isnan(velnorm_L2_) or
      std::isnan(presnorm_) or
      std::isnan(incprenorm_L2_) or
      std::isnan(prenorm_L2_))
    dserror("At least one of the calculated vector norms is NaN.");

  if (std::abs(std::isinf(vresnorm_)) or
      std::abs(std::isinf(incvelnorm_L2_)) or
      std::abs(std::isinf(velnorm_L2_)) or
      std::abs(std::isinf(presnorm_)) or
      std::abs(std::isinf(incprenorm_L2_)) or
      std::abs(std::isinf(prenorm_L2_)))
    dserror("At least one of the calculated vector norms is INF.");

  // care for the case that nothing really happens in velocity
  // or pressure field
  if (velnorm_L2_ < 1e-5) velnorm_L2_ = 1.0;
  if (prenorm_L2_ < 1e-5) prenorm_L2_ = 1.0;

  if (myrank_ == 0)
  {
    if (itnum>0)
    {
      printf("|  %3d/%3d   | %10.3E[L_2 ]  | %10.3E   | %10.3E   | %10.3E   | %10.3E   |",
        itnum,itmax,ittol,vresnorm_,presnorm_,incvelnorm_L2_/velnorm_L2_,
        incprenorm_L2_/prenorm_L2_);
      printf(" (ts=%10.3E,te=%10.3E",dtsolve_,dtele_);
      if (turbmodel_==INPAR::FLUID::dynamic_smagorinsky or turbmodel_ == INPAR::FLUID::scale_similarity)
        printf(",tf=%10.3E",dtfilter_);
      printf(")\n");
    }
    else
    {
      printf("|   --/%3d   | %10.3E[L_2 ]  | %10.3E   | %10.3E   |      --      |      --      |",
        itmax,ittol,vresnorm_,presnorm_);
      printf(" (      --     ,te=%10.3E",dtele_);
      if (turbmodel_==INPAR::FLUID::dynamic_smagorinsky or turbmodel_ == INPAR::FLUID::scale_similarity)
        printf(",tf=%10.3E",dtfilter_);
      printf(")\n");
    }
  }

  // -------------------------------------------------------------------
  // check convergence and print out respective information:
  // - stop if convergence is achieved
  // - warn if itemax is reached without convergence, but proceed to
  //   next timestep
  // -------------------------------------------------------------------
  if (vresnorm_ <= ittol and
      presnorm_ <= ittol and
      incvelnorm_L2_/velnorm_L2_ <= ittol and
      incprenorm_L2_/prenorm_L2_ <= ittol)
  {
    if (myrank_ == 0 and (inconsistent_ or (not inconsistent_ and itnum==0)))
    {
      printf("+------------+-------------------+--------------+--------------+--------------+--------------+\n");
      FILE* errfile = params_->get<FILE*>("err file",NULL);
      if (errfile!=NULL)
      {
        fprintf(errfile,"fluid solve:   %3d/%3d  tol=%10.3E[L_2 ]  vres=%10.3E  pres=%10.3E  vinc=%10.3E  pinc=%10.3E\n",
          itnum,itmax,ittol,vresnorm_,presnorm_,
          incvelnorm_L2_/velnorm_L2_,incprenorm_L2_/prenorm_L2_);
      }
    }
    return true;
  }
  else
  {
    if (itnum == itmax or (not inconsistent_ and itnum==0))
    {
      if (myrank_ == 0 and ((itnum == itmax and inconsistent_) or (not inconsistent_ and itnum==0)))
      {
        printf("+---------------------------------------------------------------+\n");
        printf("|            >>>>>> not converged in itemax steps!              |\n");
        printf("+---------------------------------------------------------------+\n");

        FILE* errfile = params_->get<FILE*>("err file",NULL);
        if (errfile!=NULL)
        {
          fprintf(errfile,"fluid unconverged solve:   %3d/%3d  tol=%10.3E[L_2 ]  vres=%10.3E  pres=%10.3E  vinc=%10.3E  pinc=%10.3E\n",
                  itnum,itmax,ittol,vresnorm_,presnorm_,
                  incvelnorm_L2_/velnorm_L2_,incprenorm_L2_/prenorm_L2_);
        }
      }
      return true;
    }
  }

  return false;

} // FluidImplicitTimeInt::ConvergenceCheck


/*----------------------------------------------------------------------*
 | free surface flow update                             rasthofer 06/13 |
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::FreeSurfaceFlowUpdate()
{
  if (alefluid_ and surfacesplitter_->FSCondRelevant())
  {
    Teuchos::RCP<Epetra_Vector> fsvelnp = surfacesplitter_->ExtractFSCondVector(velnp_);
    Teuchos::RCP<Epetra_Vector> fsdisp = surfacesplitter_->ExtractFSCondVector(dispn_);
    Teuchos::RCP<Epetra_Vector> fsdispnp = Teuchos::rcp(new Epetra_Vector(*surfacesplitter_->FSCondMap()));

    // select free surface elements
    std::string condname = "FREESURFCoupling";

    std::vector<DRT::Condition*> conds;
    discret_->GetCondition(condname, conds);

    // select only heightfunction conditions here
    std::vector<DRT::Condition*> hfconds;
    for (unsigned i=0; i<conds.size(); ++i)
    {
      if (*conds[i]->Get<std::string>("coupling")=="heightfunction")
        hfconds.push_back(conds[i]);
    }

    conds.clear();

    // ================ HEIGHTFUNCTION =======================================
    // This is the heightfunction implementation for the partitioned algorithm
    // as it is. This is a very basic implementation - and it is not
    // mass-consistent! We need another evaluate call here to find a
    // mass-consistent heightfunction manipulation matrix.
    //
    // local lagrange is mass-consistent of course.

    if (hfconds.size()>0)
    {
      Teuchos::ParameterList eleparams;
      // set action for elements
      eleparams.set<int>("action",FLD::ba_calc_node_normal);

      // get a vector layout from the discretization to construct matching
      // vectors and matrices
      //                 local <-> global dof numbering
      const Epetra_Map* dofrowmap = discret_->DofRowMap();

      //vector ndnorm0 with pressure-entries is needed for EvaluateCondition
      Teuchos::RCP<Epetra_Vector> ndnorm0 = LINALG::CreateVector(*dofrowmap,true);

      //call loop over elements, note: normal vectors do not yet have length = 1.0
      discret_->ClearState();
      discret_->SetState("dispnp", dispnp_);
      discret_->EvaluateCondition(eleparams,ndnorm0,condname);
      discret_->ClearState();

      //ndnorm contains fsnodes' normal vectors (with arbitrary length). no pressure-entries
      Teuchos::RCP<Epetra_Vector> ndnorm = surfacesplitter_->ExtractFSCondVector(ndnorm0);

      std::vector< int > GIDfsnodes;  //GIDs of free surface nodes
      std::vector< int > GIDdof;      //GIDs of current fsnode's dofs
      std::vector< int > rfs;         //local indices for ndnorm and fsvelnp for current node

      //get GIDs of free surface nodes for this processor
      DRT::UTILS::FindConditionedNodes(*discret_,hfconds,GIDfsnodes);

      for (unsigned int node=0; node<(GIDfsnodes.size()); node++)
      {
        //get LID for this node
        int ndLID = (discret_->NodeRowMap())->LID(GIDfsnodes[node]);
        if (ndLID == -1) dserror("No LID for free surface node");

        //get vector of this node's dof GIDs
        GIDdof.clear();
        discret_->Dof(discret_->lRowNode(ndLID), GIDdof);
        GIDdof.pop_back();  //free surface nodes: pop pressure dof

        //numdof = dim, no pressure
        int numdof = GIDdof.size();
        rfs.clear();
        rfs.resize(numdof);

        //get local indices for dofs in ndnorm and fsvelnp
        for (int i=0; i<numdof; i++)
        {
          int rgid = GIDdof[i];
          if (!ndnorm->Map().MyGID(rgid) or !fsvelnp->Map().MyGID(rgid)
              or ndnorm->Map().MyGID(rgid) != fsvelnp->Map().MyGID(rgid))
            dserror("Sparse vector does not have global row  %d or vectors don't match",rgid);
          rfs[i] = ndnorm->Map().LID(rgid);
        }

        double length = 0.0;
        for (int i=0; i<numdof; i++)
          length += (*ndnorm)[rfs[i]] * (*ndnorm)[rfs[i]];
        length = sqrt(length);

        double pointproduct = 0.0;
        for (int i=0; i<numdof; i++)
        {
          (*ndnorm)[rfs[i]] = (1.0/length) * (*ndnorm)[rfs[i]];
          //height function approach: left hand side of eq. 15
          pointproduct += (*ndnorm)[rfs[i]] * (*fsvelnp)[rfs[i]];
        }

        for (int i=0; i<numdof; i++)
        {
          //height function approach: last entry of u_G is delta_phi/delta_t,
          //the other entries are zero
          if (i == numdof-1)
            (*fsvelnp)[rfs[i]]  = pointproduct / (*ndnorm)[rfs[i]];
           else
            (*fsvelnp)[rfs[i]] = 0.0;
         }
      }
    }

    fsdispnp->Update(1.0,*fsdisp,dta_,*fsvelnp,0.0);

    surfacesplitter_->InsertFSCondVector(fsdispnp,dispnp_);
    surfacesplitter_->InsertFSCondVector(fsvelnp,gridv_);
  }

  return;
}

/*----------------------------------------------------------------------*
 | local lagrange update                                     hahn 03/14 |
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::LocalLagrangeUpdate()
{
  if (alefluid_ and surfacesplitter_->LLCondRelevant())
  {
    Teuchos::RCP<Epetra_Vector> llvelnp = surfacesplitter_->ExtractLLCondVector(velnp_);
    Teuchos::RCP<Epetra_Vector> lldisp = surfacesplitter_->ExtractLLCondVector(dispn_);
    Teuchos::RCP<Epetra_Vector> lldispnp = Teuchos::rcp(new Epetra_Vector(*surfacesplitter_->LLCondMap()));

    lldispnp->Update(1.0,*lldisp,dta_,*llvelnp,0.0);

    surfacesplitter_->InsertLLCondVector(lldispnp,dispnp_);
    surfacesplitter_->InsertLLCondVector(llvelnp,gridv_);
  }

  return;
}


/*----------------------------------------------------------------------*
 | Assemble Mat and RHS and apply Dirichlet Conditions          bk 12/13|
 | Call routine from outside of fluid,                                  |
 | e.g. FSI, FPSI, Poro, ...                                            |
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::Evaluate(Teuchos::RCP<const Epetra_Vector> stepinc)
{
  // update solution by adding step increment to previous converged solution
  if (stepinc!=Teuchos::null)
  {
    // Add stepinc to veln_ for non-Dirichlet values.
    Teuchos::RCP<Epetra_Vector> aux = LINALG::CreateVector(*(discret_->DofRowMap()),true);
    aux->Update(1.0, *veln_, 1.0, *stepinc, 0.0);

    // Set Dirichlet values
    dbcmaps_->InsertCondVector(dbcmaps_->ExtractCondVector(velnp_), aux);

    InsertVolumetricSurfaceFlowCondVector(velnp_,aux);

    *velnp_ = *aux;
  }

  // --------------------------------------------------
  //the following steps have to be repeated after that the velocity has been updated
  // --------------------------------------------------

  // adjust accnp_ according to Dirichlet values of velnp_ for GenAlpha
  GenAlphaUpdateAcceleration();

  // compute values at intermediate time steps for GenAlpha
  // ----------------------------------------------------------------
  GenAlphaIntermediateValues();

  if (alefluid_)
  {
    // account for potentially moving Neumann boundaries
    Teuchos::ParameterList eleparams;
    discret_->SetState("dispnp", dispnp_);

    // evaluate Neumann conditions
    neumann_loads_->PutScalar(0.0);
    discret_->SetState("scaaf",scaaf_);
    discret_->EvaluateNeumann(eleparams,*neumann_loads_);
    discret_->ClearState();
  }

  PrepareSolve();

  return;
}

/*----------------------------------------------------------------------*
 | current solution becomes most recent solution of next timestep       |
 |                                                                      |
 | One-step-Theta: (step>1)                                             |
 |                                                                      |
 |  accn_  = (velnp_-veln_) / (Theta * dt) - (1/Theta -1) * accn_"(n+1) |
 |                                                                      |
 |  velnm_ =veln_                                                       |
 |  veln_  =velnp_                                                      |
 |                                                                      |
 |  BDF2:           (step>1)                                            |
 |                                                                      |
 |               2*dt(n)+dt(n-1)              dt(n)+dt(n-1)             |
 |  accn_   = --------------------- velnp_ - --------------- veln_      |
 |            dt(n)*[dt(n)+dt(n-1)]           dt(n)*dt(n-1)             |
 |                                                                      |
 |                     dt(n)                                            |
 |           + ----------------------- velnm_                           |
 |             dt(n-1)*[dt(n)+dt(n-1)]                                  |
 |                                                                      |
 |  velnm_ =veln_                                                       |
 |  veln_  =velnp_                                                      |
 |                                                                      |
 |  BDF2 and  One-step-Theta: (step==1)                                 |
 |                                                                      |
 |  The given formulas are only valid from the second timestep. In the  |
 |  first step, the acceleration is calculated simply by                |
 |                                                                      |
 |  accn_  = (velnp_-veln_) / (dt)                                      |
 |                                                                      |
 |                                                           gammi 04/07|
 |  overloaded in TimIntRedModels                              bk 12/13 |
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::TimeUpdate()
{
  Teuchos::ParameterList *  stabparams;
  stabparams=&(params_->sublist("RESIDUAL-BASED STABILIZATION"));

  if(stabparams->get<std::string>("STABTYPE")=="residual_based")
  {
    if(stabparams->get<std::string>("TDS") == "time_dependent")
    {
      const double tcpu=Teuchos::Time::wallTime();

      if(myrank_==0)
      {
        std::cout << "time update for subscales";
      }

      // call elements to calculate system matrix and rhs and assemble
      // this is required for the time update of the subgrid scales and
      // makes sure that the current subgrid scales correspond to the
      // current residual
      AssembleMatAndRHS();

      // create the parameters for the discretization
      Teuchos::ParameterList eleparams;
      // action for elements
      eleparams.set<int>("action",FLD::calc_fluid_genalpha_update_for_subscales);

      // update time parameters
      SetGamma(eleparams);

      eleparams.set("dt"     ,dta_    );

      // call loop over elements to update subgrid scales
      discret_->Evaluate(eleparams,Teuchos::null,Teuchos::null,Teuchos::null,Teuchos::null,Teuchos::null);

      if(myrank_==0)
        std::cout << "("<<Teuchos::Time::wallTime()-tcpu<<")\n";
    }
  }

  // compute accelerations
  TimIntCalculateAcceleration();

  // acceleration of this step becomes most recent
  // acceleration of the last step
  accnm_->Update(1.0,*accn_,0.0);
  accn_->Update(1.0,*accnp_,0.0);

  // velocities/pressures of this step become most recent
  // velocities/pressures of the last step
  velnm_->Update(1.0,*veln_ ,0.0);
  veln_ ->Update(1.0,*velnp_,0.0);

  // compute values at nodes from nodal values for non-interpolatory basis functions
  if (velatmeshfreenodes_!=Teuchos::null)
  {
    Teuchos::rcp_dynamic_cast<DRT::MESHFREE::MeshfreeDiscretization>(discret_)->ComputeValuesAtNodes(veln_, velatmeshfreenodes_);
  }

  // displacement vectors for ALE
  if (alefluid_)
  {
    dispnm_->Update(1.0,*dispn_,0.0);
    dispn_ ->Update(1.0,*dispnp_,0.0);
  }

  // update flow-rate and flow-volume vectors in case of flow-dependent pressure
  // boundary conditions,
  if (nonlinearbc_)
  {
    std::vector<DRT::Condition*> flowdeppressureline;
    discret_->GetCondition("LineFlowDepPressure",flowdeppressureline);
    std::vector<DRT::Condition*> flowdeppressuresurf;
    discret_->GetCondition("SurfaceFlowDepPressure",flowdeppressuresurf);

    if (flowdeppressureline.size() != 0 or flowdeppressuresurf.size() != 0)
    {
      for (int i = 0; i < 4; i++)
      {
        flowratenm_(i) = flowraten_(i);
        flowraten_(i)  = flowratenp_(i);

        flowvolumenm_(i) = flowvolumen_(i);
        flowvolumen_(i)  = flowvolumenp_(i);
      }

      // close this time step also in output file
      if (myrank_ == 0)
      {
        const std::string fname1 = DRT::Problem::Instance()->OutputControlFile()->FileName()+".fdpressure";
        std::ofstream f1;
        f1.open(fname1.c_str(),std::fstream::ate | std::fstream::app);

        f1 << "\n";
        f1.flush();
        f1.close();
      }
    }
  }

  // call time update of forcing routine
  if (homisoturb_forcing_ != Teuchos::null)
    homisoturb_forcing_->TimeUpdateForcing();

  // account for possible changes in time step size and update previous
  // time step size accordingly
  dtp_ = dta_;
  discret_->ClearState();

  return;
}// FluidImplicitTimeInt::TimeUpdate

/*----------------------------------------------------------------------*
 | Calculate Acceleration                                               |
 | overloaded in TimIntPoro                                    bk 12/13 |
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::TimIntCalculateAcceleration()
{
  Teuchos::RCP<Epetra_Vector> onlyaccn = Teuchos::null;
  Teuchos::RCP<Epetra_Vector> onlyaccnp = Teuchos::null;
  Teuchos::RCP<Epetra_Vector> onlyvelnm = Teuchos::null;
  Teuchos::RCP<Epetra_Vector> onlyveln = Teuchos::null;
  Teuchos::RCP<Epetra_Vector> onlyvelnp = Teuchos::null;

  if ( physicaltype_ == INPAR::FLUID::artcomp) //artcomp case
  {
    onlyaccn = accn_;
    onlyaccnp = accnp_;
    onlyvelnm = velnm_;
    onlyveln = veln_;
    onlyvelnp = velnp_;
  }
  else //standard case
  {
    onlyaccn  = velpressplitter_.ExtractOtherVector(accn_);
    onlyaccnp = velpressplitter_.ExtractOtherVector(accnp_);
    onlyvelnm = velpressplitter_.ExtractOtherVector(velnm_);
    onlyveln  = velpressplitter_.ExtractOtherVector(veln_);
    onlyvelnp = velpressplitter_.ExtractOtherVector(velnp_);
  }

  CalculateAcceleration(onlyvelnp,
                        onlyveln ,
                        onlyvelnm,
                        onlyaccn ,
                        onlyaccnp);

  // copy back into global vector
  LINALG::Export(*onlyaccnp,*accnp_);
  return;
}

/*----------------------------------------------------------------------*
 | calculate intermediate solution                       rasthofer 05/13|
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::CalcIntermediateSolution()
{
  if ((special_flow_ == "forced_homogeneous_isotropic_turbulence"
      or special_flow_ == "scatra_forced_homogeneous_isotropic_turbulence"
      or special_flow_ == "decaying_homogeneous_isotropic_turbulence") and
      DRT::INPUT::IntegralValue<INPAR::FLUID::ForcingType>(params_->sublist("TURBULENCE MODEL"),"FORCING_TYPE")
      == INPAR::FLUID::linear_compensation_from_intermediate_spectrum)
  {
    bool activate = true;
    if (special_flow_ == "decaying_homogeneous_isotropic_turbulence"
        and step_ > params_->sublist("TURBULENCE MODEL").get<int>("FORCING_TIME_STEPS",0))
      activate = false;

    if (activate)
    {
      if (homisoturb_forcing_ == Teuchos::null)
        dserror("Forcing expected!");

      if (myrank_ == 0)
      {
        std::cout << "+------------------------------------------------------------------------+\n";
        std::cout << "|     calculate intermediate solution\n";
        std::cout << "|"<< std::endl;
      }

      // turn off forcing in Solve()
      homisoturb_forcing_->ActivateForcing(false);

      // temporary store velnp_ since it will be modified in Solve()
      const Epetra_Map* dofrowmap = discret_->DofRowMap();
      Teuchos::RCP<Epetra_Vector> tmp = LINALG::CreateVector(*dofrowmap,true);
      tmp->Update(1.0,*velnp_,0.0);

      // compute intermediate solution without forcing
      forcing_->PutScalar(0.0); // just to be sure
      Solve();

      // calculate required forcing
      homisoturb_forcing_->CalculateForcing(step_);

      // reset velnp_
      velnp_->Update(1.0,*tmp,0.0);

      // recompute intermediate values, since they have been likewise overwritten
      // --------------------------------------------------
      // adjust accnp according to Dirichlet values of velnp
      //
      //                                  n+1     n
      //                               vel   - vel
      //       n+1      n  gamma-1.0      (0)
      //    acc    = acc * --------- + ------------
      //       (0)           gamma      gamma * dt
      //
      GenAlphaUpdateAcceleration();

      // ----------------------------------------------------------------
      // compute values at intermediate time steps
      // ----------------------------------------------------------------
      GenAlphaIntermediateValues();

      homisoturb_forcing_->ActivateForcing(true);

      if (myrank_ == 0)
      {
        std::cout << "|\n";
        std::cout << "+------------------------------------------------------------------------+\n";
        std::cout << "+------------------------------------------------------------------------+\n";
        std::cout << "|" << std::endl;
      }
    }
    else
      // set force to zero
      forcing_->PutScalar(0.0);
  }

  return;
}


/*----------------------------------------------------------------------*
 | lift'n'drag forces, statistics time sample and output of solution    |
 | and statistics                                              vg 11/08 |
 -----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::StatisticsAndOutput()
{
  // time measurement: output and statistics
  TEUCHOS_FUNC_TIME_MONITOR("      + output and statistics");

  // -------------------------------------------------------------------
  //          calculate lift'n'drag forces from the residual
  // -------------------------------------------------------------------
  LiftDrag();

  // store subfilter stresses for additional output
  if (turbmodel_==INPAR::FLUID::scale_similarity)
  {
    Teuchos::RCP<Epetra_Vector> stress12 = CalcSFS(1,2);
    statisticsmanager_->StoreNodalValues(step_, stress12);
  }
  // -------------------------------------------------------------------
  //   add calculated velocity to mean value calculation (statistics)
  // -------------------------------------------------------------------
  if (meshtying_ !=Teuchos::null)
    statisticsmanager_->GetCurrentVelnp(velnp_);

  CallStatisticsManager();

  // -------------------------------------------------------------------
  //                        compute flow rates
  // -------------------------------------------------------------------
  ComputeFlowRates();

  // -------------------------------------------------------------------
  //                         output of solution
  // -------------------------------------------------------------------
  Output();

  // -------------------------------------------------------------------
  //          dumping of turbulence statistics if required
  // -------------------------------------------------------------------
  statisticsmanager_->DoOutput(*output_,step_);

  // -------------------------------------------------------------------
  // evaluate error for test flows with analytical solutions
  // -------------------------------------------------------------------
  EvaluateErrorComparedToAnalyticalSol();

  // -------------------------------------------------------------------
  // evaluate divergence u
  // -------------------------------------------------------------------
  EvaluateDivU();

} // FluidImplicitTimeInt::StatisticsAndOutput

/*----------------------------------------------------------------------*
 | statistics time sample, overloaded in TimIntLoma            bk 12/13 |
 -----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::CallStatisticsManager()
{
  // -------------------------------------------------------------------
  //   add calculated velocity to mean value calculation (statistics)
  // -------------------------------------------------------------------
  statisticsmanager_->DoTimeSample(step_,0.0,
                                   0.0,0.0,
                                   0.0,0.0);
  return;
}

/*----------------------------------------------------------------------*
 | statistics time sample and output of statistics      rasthofer 06/11 |
 -----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::StatisticsOutput()
{
  // store subfilter stresses for additional output
  if (turbmodel_==INPAR::FLUID::scale_similarity)
  {
    Teuchos::RCP<Epetra_Vector> stress12 = CalcSFS(1,2);
    statisticsmanager_->StoreNodalValues(step_, stress12);
  }
  // -------------------------------------------------------------------
  //   add calculated velocity to mean value calculation (statistics)
  // -------------------------------------------------------------------
  CallStatisticsManager();

  // -------------------------------------------------------------------
  //          dumping of turbulence statistics if required
  // -------------------------------------------------------------------
  statisticsmanager_->DoOutput(*output_,step_,true);

  if (params_->get<bool>("GMSH_OUTPUT"))
    OutputToGmsh(step_, time_,true);
} // FluidImplicitTimeInt::StatisticsOutput


/*----------------------------------------------------------------------*
 | output of solution vector to binio                        gammi 04/07|
 | overloaded in TimIntPoro                                             |
 | overloaded in TimIntRedModels                               bk 12/13|
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::Output()
{
  // output of solution
  if (step_%upres_ == 0)
  {
    // step number and time
    output_->NewStep(step_,time_);

    // velocity/pressure vector
    output_->WriteVector("velnp",velnp_);
    // (hydrodynamic) pressure
    Teuchos::RCP<Epetra_Vector> pressure = velpressplitter_.ExtractCondVector(velnp_);
    output_->WriteVector("pressure", pressure);

    // velocity/pressure at nodes for meshfree problems with non-interpolatory basis functions
    if(velatmeshfreenodes_!=Teuchos::null)
    {
      // velocity/pressure values at meshfree nodes
      output_->WriteVector("velatmeshfreenodes",velatmeshfreenodes_);
      // (hydrodynamic) pressure values at meshfree nodes
      Teuchos::RCP<Epetra_Vector> pressureatmeshfreenodes = velpressplitter_.ExtractCondVector(velatmeshfreenodes_);
      output_->WriteVector("pressureatmeshfreenodes", pressureatmeshfreenodes);
    }

    if (params_->get<bool>("GMSH_OUTPUT"))
      OutputToGmsh(step_, time_,false);

    //output_->WriteVector("residual", trueresidual_);
    if (alefluid_) output_->WriteVector("dispnp", dispnp_);

    if (physicaltype_ == INPAR::FLUID::varying_density or physicaltype_ == INPAR::FLUID::boussinesq)
    {
      Teuchos::RCP<Epetra_Vector> scalar_field = velpressplitter_.ExtractCondVector(scaaf_);
      output_->WriteVector("scalar_field", scalar_field);
    }

    //only perform stress calculation when output is needed
    if (writestresses_)
    {
      Teuchos::RCP<Epetra_Vector> traction = CalcStresses();
      output_->WriteVector("traction",traction);
      if (myrank_==0)
        std::cout<<"Writing stresses"<<std::endl;
      //only perform wall shear stress calculation when output is needed
      if (write_wall_shear_stresses_)
      {
        Teuchos::RCP<Epetra_Vector> wss = CalcWallShearStresses();
        output_->WriteVector("wss",wss);
      }
    }

    //biofilm growth
    if (fldgrdisp_!=Teuchos::null)
    {
      output_->WriteVector("fld_growth_displ", fldgrdisp_);
    }

    // cavitation: void fraction
    WriteOutputCavitationVoidFraction();

    if (params_->get<bool>("COMPUTE_EKIN"))
      WriteOutputKineticEnergy();

    // don't write output in case of separate inflow computation
    // Sep_-Matrix needed for algebraic-multigrid filter has never been build
#if 0
    // output of coarse and fine scale velocities
    // at time n+1 or n+af depending on the time
    // integration scheme
    if (turbmodel_==INPAR::FLUID::scale_similarity or turbmodel_ == INPAR::FLUID::multifractal_subgrid_scales) // or dynamic_smagorinsky_)
    {
      const Epetra_Map* dofrowmap = discret_->DofRowMap();
      Teuchos::RCP<Epetra_Vector> filteredvel = LINALG::CreateVector(*dofrowmap,true);
      Teuchos::RCP<Epetra_Vector> fsvel = LINALG::CreateVector(*dofrowmap,true);
      if (scale_sep_ == INPAR::FLUID::algebraic_multigrid_operator
        or scale_sep_ == INPAR::FLUID::geometric_multigrid_operator)
      {
        OutputofFilteredVel(filteredvel,fsvel);
      }
      if (scale_sep_ == INPAR::FLUID::box_filter)
      {
        Boxf_->OutputofAveragedVel(filteredvel);
        Boxf_->OutputofFineScaleVel(fsvel);
      }
      output_->WriteVector("filteredvel",filteredvel);
      output_->WriteVector("fsvelaf",fsvel);
      if (turbmodel_==INPAR::FLUID::scale_similarity)
      {
        if (myrank_==0)
           std::cout << "output of subfilter stresses for scale similarity model ..." << std::endl;
        Teuchos::RCP<Epetra_Vector> stress11 = CalcSFS(1,1);
        output_->WriteVector("sfs11",stress11);
        Teuchos::RCP<Epetra_Vector> stress12 = CalcSFS(1,2);
        output_->WriteVector("sfs12",stress12);
        Teuchos::RCP<Epetra_Vector> stress13 = CalcSFS(1,3);
        output_->WriteVector("sfs13",stress13);
        Teuchos::RCP<Epetra_Vector> stress22 = CalcSFS(2,2);
        output_->WriteVector("sfs22",stress22);
        Teuchos::RCP<Epetra_Vector> stress23 = CalcSFS(2,3);
        output_->WriteVector("sfs23",stress23);
        Teuchos::RCP<Epetra_Vector> stress33 = CalcSFS(3,3);
        output_->WriteVector("sfs33",stress33);
      }
    }
#endif

    // write domain decomposition for visualization (only once!)
    if (step_==upres_ or step_ == 0) output_->WriteElementData(true);

    if (uprestart_ != 0 && step_%uprestart_ == 0) //add restart data
    {
      // acceleration vector at time n+1 and n, velocity/pressure vector at time n and n-1
      output_->WriteVector("accnp",accnp_);
      output_->WriteVector("accn", accn_);
      output_->WriteVector("veln", veln_);
      output_->WriteVector("velnm",velnm_);

      if (alefluid_)
      {
        output_->WriteVector("dispn", dispn_);
        output_->WriteVector("dispnm",dispnm_);
      }

      // flow rate and flow volume in case of flow-dependent pressure bc
      if (nonlinearbc_)
      {
        std::vector<DRT::Condition*> flowdeppressureline;
        discret_->GetCondition("LineFlowDepPressure",flowdeppressureline);
        std::vector<DRT::Condition*> flowdeppressuresurf;
        discret_->GetCondition("SurfaceFlowDepPressure",flowdeppressuresurf);

        if (flowdeppressureline.size() != 0 or flowdeppressuresurf.size() != 0)
        {
          output_->WriteDouble("flowratenp0",flowratenp_(0));
          output_->WriteDouble("flowratenp1",flowratenp_(1));
          output_->WriteDouble("flowratenp2",flowratenp_(2));
          output_->WriteDouble("flowratenp3",flowratenp_(3));
          output_->WriteDouble("flowraten0", flowraten_(0));
          output_->WriteDouble("flowraten1", flowraten_(1));
          output_->WriteDouble("flowraten2", flowraten_(2));
          output_->WriteDouble("flowraten3", flowraten_(3));
          output_->WriteDouble("flowratenm0",flowratenm_(0));
          output_->WriteDouble("flowratenm1",flowratenm_(1));
          output_->WriteDouble("flowratenm2",flowratenm_(2));
          output_->WriteDouble("flowratenm3",flowratenm_(3));

          output_->WriteDouble("flowvolumenp0",flowvolumenp_(0));
          output_->WriteDouble("flowvolumenp1",flowvolumenp_(1));
          output_->WriteDouble("flowvolumenp2",flowvolumenp_(2));
          output_->WriteDouble("flowvolumenp3",flowvolumenp_(3));
          output_->WriteDouble("flowvolumen0", flowvolumen_(0));
          output_->WriteDouble("flowvolumen1", flowvolumen_(1));
          output_->WriteDouble("flowvolumen2", flowvolumen_(2));
          output_->WriteDouble("flowvolumen3", flowvolumen_(3));
          output_->WriteDouble("flowvolumenm0",flowvolumenm_(0));
          output_->WriteDouble("flowvolumenm1",flowvolumenm_(1));
          output_->WriteDouble("flowvolumenm2",flowvolumenm_(2));
          output_->WriteDouble("flowvolumenm3",flowvolumenm_(3));
        }
      }

      // write mesh in each restart step --- the elements are required since
      // they contain history variables (the time dependent subscales)
      // But never do this for step 0 (visualization of initial field) since
      // it would lead to writing the mesh twice for step 0
      // (FluidBaseAlgorithm already wrote the mesh) -> HDF5 writer will claim!
      if ((step_!=0) and ((params_->sublist("RESIDUAL-BASED STABILIZATION").get<std::string>("TDS")) != "quasistatic"))
        output_->WriteMesh(step_,time_);
    }

  }
  // write restart also when uprestart_ is not a integer multiple of upres_
  else if (uprestart_ > 0 && step_%uprestart_ == 0)
  {
    // step number and time
    output_->NewStep(step_,time_);

    // velocity/pressure vector
    output_->WriteVector("velnp",velnp_);

    //output_->WriteVector("residual", trueresidual_);
    if (alefluid_)
    {
      output_->WriteVector("dispnp", dispnp_);
      output_->WriteVector("dispn", dispn_);
      output_->WriteVector("dispnm",dispnm_);
    }

    // write mesh in each restart step --- the elements are required since
    // they contain history variables (the time dependent subscales)
    // But never do this for step 0 (visualization of initial field) since
    // it would lead to writing the mesh twice for step 0
    // (FluidBaseAlgorithm already wrote the mesh) -> HDF5 writer will claim!
    if ((step_!=0) and ((params_->sublist("RESIDUAL-BASED STABILIZATION").get<std::string>("TDS")) != "quasistatic"))
      output_->WriteMesh(step_,time_);

    //only perform stress calculation when output is needed
    if (writestresses_)
    {
      Teuchos::RCP<Epetra_Vector> traction = CalcStresses();
      output_->WriteVector("traction",traction);
      //only perform wall shear stress calculation when output is needed
      if (write_wall_shear_stresses_)
      {
        Teuchos::RCP<Epetra_Vector> wss = CalcWallShearStresses();
        output_->WriteVector("wss",wss);
      }
    }

    // acceleration vector at time n+1 and n, velocity/pressure vector at time n and n-1
    output_->WriteVector("accnp",accnp_);
    output_->WriteVector("accn", accn_);
    output_->WriteVector("veln", veln_);
    output_->WriteVector("velnm",velnm_);

    // flow rate and flow volume in case of flow-dependent pressure bc
    if (nonlinearbc_)
    {
      std::vector<DRT::Condition*> flowdeppressureline;
      discret_->GetCondition("LineFlowDepPressure",flowdeppressureline);
      std::vector<DRT::Condition*> flowdeppressuresurf;
      discret_->GetCondition("SurfaceFlowDepPressure",flowdeppressuresurf);

      if (flowdeppressureline.size() != 0 or flowdeppressuresurf.size() != 0)
      {
        output_->WriteDouble("flowratenp0",flowratenp_(0));
        output_->WriteDouble("flowratenp1",flowratenp_(1));
        output_->WriteDouble("flowratenp2",flowratenp_(2));
        output_->WriteDouble("flowratenp3",flowratenp_(3));
        output_->WriteDouble("flowraten0", flowraten_(0));
        output_->WriteDouble("flowraten1", flowraten_(1));
        output_->WriteDouble("flowraten2", flowraten_(2));
        output_->WriteDouble("flowraten3", flowraten_(3));
        output_->WriteDouble("flowratenm0",flowratenm_(0));
        output_->WriteDouble("flowratenm1",flowratenm_(1));
        output_->WriteDouble("flowratenm2",flowratenm_(2));
        output_->WriteDouble("flowratenm3",flowratenm_(3));

        output_->WriteDouble("flowvolumenp0",flowvolumenp_(0));
        output_->WriteDouble("flowvolumenp1",flowvolumenp_(1));
        output_->WriteDouble("flowvolumenp2",flowvolumenp_(2));
        output_->WriteDouble("flowvolumenp3",flowvolumenp_(3));
        output_->WriteDouble("flowvolumen0", flowvolumen_(0));
        output_->WriteDouble("flowvolumen1", flowvolumen_(1));
        output_->WriteDouble("flowvolumen2", flowvolumen_(2));
        output_->WriteDouble("flowvolumen3", flowvolumen_(3));
        output_->WriteDouble("flowvolumenm0",flowvolumenm_(0));
        output_->WriteDouble("flowvolumenm1",flowvolumenm_(1));
        output_->WriteDouble("flowvolumenm2",flowvolumenm_(2));
        output_->WriteDouble("flowvolumenm3",flowvolumenm_(3));
      }
    }
  }

//#define PRINTALEDEFORMEDNODECOORDS // flag for printing all ALE nodes and xspatial in current configuration - only works for 1 processor  devaal 02.2011

// output ALE nodes and xspatial in current configuration - devaal 02.2011
#ifdef PRINTALEDEFORMEDNODECOORDS

  if (discret_->Comm().NumProc() != 1)
    dserror("The flag PRINTALEDEFORMEDNODECOORDS has been switched on, and only works for 1 processor");

  std::cout << "ALE DISCRETIZATION IN THE DEFORMED CONFIGURATIONS" << std::endl;
  // does discret_ exist here?
  //std::cout << "discret_->NodeRowMap()" << discret_->NodeRowMap() << std::endl;

  //Teuchos::RCP<Epetra_Vector> mynoderowmap = Teuchos::rcp(new Epetra_Vector(discret_->NodeRowMap()));
  //Teuchos::RCP<Epetra_Vector> noderowmap_ = Teuchos::rcp(new Epetra_Vector(discret_->NodeRowMap()));
  //dofrowmap_  = Teuchos::rcp(new discret_->DofRowMap());
  const Epetra_Map* noderowmap = discret_->NodeRowMap();
  const Epetra_Map* dofrowmap = discret_->DofRowMap();

  for (int lid=0; lid<noderowmap->NumGlobalPoints(); lid++)
  {
    int gid;
    // get global id of a node
    gid = noderowmap->GID(lid);
    // get the node
    DRT::Node * node = discret_->gNode(gid);
    // get the coordinates of the node
    const double * X = node->X();
    // get degrees of freedom of a node
    std::vector<int> gdofs = discret_->Dof(node);
    //std::cout << "for node:" << *node << std::endl;
    //std::cout << "this is my gdof vector" << gdofs[0] << " " << gdofs[1] << " " << gdofs[2] << std::endl;

    // get displacements of a node
    std::vector<double> mydisp (3,0.0);
    for (int ldof = 0; ldof<3; ldof ++)
    {
      int displid = dofrowmap->LID(gdofs[ldof]);
      //std::cout << "displacement local id - in the rowmap" << displid << std::endl;
      mydisp[ldof] = (*dispnp_)[displid];
      //make zero if it is too small
      if (abs(mydisp[ldof]) < 0.00001)
      {
        mydisp[ldof] = 0.0;
      }
    }
    // Export disp, X
    double newX = mydisp[0]+X[0];
    double newY = mydisp[1]+X[1];
    double newZ = mydisp[2]+X[2];
    //std::cout << "NODE " << gid << "  COORD  " << newX << " " << newY << " " << newZ << std::endl;
    std::cout << gid << " " << newX << " " << newY << " " << newZ << std::endl;
  }
#endif //PRINTALEDEFORMEDNODECOORDS

  return;
} // FluidImplicitTimeInt::Output


void FLD::FluidImplicitTimeInt::OutputToGmsh(
    const int step,
    const double time,
    const bool inflow
    ) const
{
  // turn on/off screen output for writing process of Gmsh postprocessing file
  const bool screen_out = true;

  // create Gmsh postprocessing file
  // 20 steps are kept
  std::string filename = "dummy";
  if (inflow)
  {
    filename = IO::GMSH::GetNewFileNameAndDeleteOldFiles("solution_velpres_inflow", step, 20, screen_out, discret_->Comm().MyPID());
    //std::ofstream gmshfilecontent(filename.c_str());
  }
  else
  {
    filename = IO::GMSH::GetNewFileNameAndDeleteOldFiles("solution_velpres", step, 20, screen_out, discret_->Comm().MyPID());
    //std::ofstream gmshfilecontent(filename.c_str());
  }
  std::ofstream gmshfilecontent(filename.c_str());

  {
    // add 'View' to Gmsh postprocessing file
    gmshfilecontent << "View \" " << "velocity solution \" {" << std::endl;
    IO::GMSH::VelocityPressureFieldDofBasedToGmsh(discret_, velnp_ , "velocity", gmshfilecontent);
    gmshfilecontent << "};" << std::endl;
  }
  {
    // add 'View' to Gmsh postprocessing file
    gmshfilecontent << "View \" " << "pressure solution\" {" << std::endl;
    IO::GMSH::VelocityPressureFieldDofBasedToGmsh(discret_, velnp_, "pressure",gmshfilecontent);
    gmshfilecontent << "};" << std::endl;
  }

  gmshfilecontent.close();
  if (screen_out) std::cout << " done" << std::endl;

 return;
}


/*----------------------------------------------------------------------*
 |                                                             kue 04/07|
 -----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::ReadRestart(int step)
{
  IO::DiscretizationReader reader(discret_,step);
  time_ = reader.ReadDouble("time");
  step_ = reader.ReadInt("step");

  reader.ReadVector(velnp_,"velnp");
  reader.ReadVector(veln_, "veln");
  reader.ReadVector(velnm_,"velnm");
  reader.ReadVector(accnp_,"accnp");
  reader.ReadVector(accn_ ,"accn");

  // set element time parameter after restart:
  // Here it is already needed by AVM3 and impedance boundary condition!!
  SetElementTimeParameter();

  statisticsmanager_->Restart(reader,step);

  if ((fssgv_ != INPAR::FLUID::no_fssgv) or
      (scale_sep_ == INPAR::FLUID::algebraic_multigrid_operator))
  {
    AVM3Preparation();
  }

  if (alefluid_)
  {
    reader.ReadVector(dispnp_,"dispnp");
    reader.ReadVector(dispn_ , "dispn");
    reader.ReadVector(dispnm_,"dispnm");
  }

  // flow rate and flow volume in case of flow-dependent pressure bc
  if (nonlinearbc_)
  {
    std::vector<DRT::Condition*> flowdeppressureline;
    discret_->GetCondition("LineFlowDepPressure",flowdeppressureline);
    std::vector<DRT::Condition*> flowdeppressuresurf;
    discret_->GetCondition("SurfaceFlowDepPressure",flowdeppressuresurf);

    if (flowdeppressureline.size() != 0 or flowdeppressuresurf.size() != 0)
    {
      flowratenp_(0) = reader.ReadDouble("flowratenp0");
      flowratenp_(1) = reader.ReadDouble("flowratenp1");
      flowratenp_(2) = reader.ReadDouble("flowratenp2");
      flowratenp_(3) = reader.ReadDouble("flowratenp3");
      flowraten_(0)  = reader.ReadDouble("flowraten0");
      flowraten_(1)  = reader.ReadDouble("flowraten1");
      flowraten_(2)  = reader.ReadDouble("flowraten2");
      flowraten_(3)  = reader.ReadDouble("flowraten3");
      flowratenm_(0) = reader.ReadDouble("flowratenm0");
      flowratenm_(1) = reader.ReadDouble("flowratenm1");
      flowratenm_(2) = reader.ReadDouble("flowratenm2");
      flowratenm_(3) = reader.ReadDouble("flowratenm3");

      flowvolumenp_(0) = reader.ReadDouble("flowvolumenp0");
      flowvolumenp_(1) = reader.ReadDouble("flowvolumenp1");
      flowvolumenp_(2) = reader.ReadDouble("flowvolumenp2");
      flowvolumenp_(3) = reader.ReadDouble("flowvolumenp3");
      flowvolumen_(0)  = reader.ReadDouble("flowvolumen0");
      flowvolumen_(1)  = reader.ReadDouble("flowvolumen1");
      flowvolumen_(2)  = reader.ReadDouble("flowvolumen2");
      flowvolumen_(3)  = reader.ReadDouble("flowvolumen3");
      flowvolumenm_(0) = reader.ReadDouble("flowvolumenm0");
      flowvolumenm_(1) = reader.ReadDouble("flowvolumenm1");
      flowvolumenm_(2) = reader.ReadDouble("flowvolumenm2");
      flowvolumenm_(3) = reader.ReadDouble("flowvolumenm3");
    }
  }

  // read the previously written elements including the history data
  // only avalaible+required for time-dependent subgrid scales!
  if ((params_->sublist("RESIDUAL-BASED STABILIZATION").get<std::string>("TDS")) != "quasistatic")
    reader.ReadMesh(step_);

    // ensure that the overall dof numbering is identical to the one
  // that was used when the restart data was written. Especially
  // in case of multiphysics problems & periodic boundary conditions
  // it is better to check the consistency of the maps here:
  if (not (discret_->DofRowMap())->SameAs(velnp_->Map()))
    dserror("Global dof numbering in maps does not match");
  if (not (discret_->DofRowMap())->SameAs(veln_->Map()))
    dserror("Global dof numbering in maps does not match");
  if (not (discret_->DofRowMap())->SameAs(accn_->Map()))
    dserror("Global dof numbering in maps does not match");

}


/*----------------------------------------------------------------------*
 |set restart values (turbulent inflow only)             rasthofer 06/11|
 -----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::SetRestart(
  const int step,
  const double time,
  Teuchos::RCP<const Epetra_Vector> readvelnp,
  Teuchos::RCP<const Epetra_Vector> readveln,
  Teuchos::RCP<const Epetra_Vector> readvelnm,
  Teuchos::RCP<const Epetra_Vector> readaccnp,
  Teuchos::RCP<const Epetra_Vector> readaccn)
{
  time_ = time;
  step_ = step;

  velnp_->Update(1.0,*readvelnp,0.0);
  veln_->Update(1.0,*readveln,0.0);
  velnm_->Update(1.0,*readvelnm,0.0);
  accnp_->Update(1.0,*readaccnp,0.0);
  accn_->Update(1.0,*readaccn,0.0);

  if ((fssgv_ != INPAR::FLUID::no_fssgv) or
      (scale_sep_ == INPAR::FLUID::algebraic_multigrid_operator))
  {
    SetElementTimeParameter();
    AVM3Preparation();
  }

}


/*----------------------------------------------------------------------*
 |                                                           chfoe 01/08|
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::UpdateGridv()
{
  // get order of accuracy of grid velocity determination
  // from input file data
  const Teuchos::ParameterList& fluiddynparams =  DRT::Problem::Instance()->FluidDynamicParams();
  const int order = DRT::INPUT::IntegralValue<INPAR::FLUID::Gridvel>(fluiddynparams, "GRIDVEL");

  switch (order)
  {
    case INPAR::FLUID::BE:
      /* get gridvelocity from BE time discretisation of mesh motion:
           -> cheap
           -> easy
           -> limits FSI algorithm to first order accuracy in time

                  x^n+1 - x^n
             uG = -----------
                    Delta t                        */
      gridv_->Update(1/dta_, *dispnp_, -1/dta_, *dispn_, 0.0);
    break;
    case INPAR::FLUID::BDF2:
      /* get gridvelocity from BDF2 time discretisation of mesh motion:
           -> requires one more previous mesh position or displacement
           -> somewhat more complicated
           -> allows second order accuracy for the overall flow solution  */
      gridv_->Update(1.5/dta_, *dispnp_, -2.0/dta_, *dispn_, 0.0);
      gridv_->Update(0.5/dta_, *dispnm_, 1.0);
    break;
    case INPAR::FLUID::OST:
      /* get gridvelocity from OST time discretisation of mesh motion:
         -> needed to allow consistent linearization of FPSI problem  */
      const double theta = fluiddynparams.get<double>("THETA");
      gridv_->Update(1/(theta*dta_), *dispnp_, -1/(theta*dta_), *dispn_, 0.0);
      gridv_->Update(-((1.0/theta)-1.0),*veln_,1.0);
    break;
  }

  // Set proper grid velocities at the free-surface and for the local lagrange conditions
  Teuchos::RCP<Epetra_Vector> llveln = surfacesplitter_->ExtractLLCondVector(veln_);
  surfacesplitter_->InsertLLCondVector(llveln,gridv_);

  Teuchos::RCP<Epetra_Vector> fsveln = surfacesplitter_->ExtractFSCondVector(veln_);
  surfacesplitter_->InsertFSCondVector(fsveln,gridv_);
}


/*----------------------------------------------------------------------*
 | prepare AVM3-based scale separation                         vg 10/08 |
 | overloaded in TimIntRedModels and TimIntLoma               bk 12/13 |
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::AVM3Preparation()
{

  // time measurement: avm3
  TEUCHOS_FUNC_TIME_MONITOR("           + avm3");

  // create the parameters for the discretization
  Teuchos::ParameterList eleparams;

  //necessary here, because some application time integrations add something to the residual
  //before the Neumann loads are added
  residual_->PutScalar(0.0);

  AVM3AssembleMatAndRHS(eleparams);

  // get scale-separation matrix
  AVM3GetScaleSeparationMatrix();

  return;
}// FluidImplicitTimeInt::AVM3Preparation


/*----------------------------------------------------------------------*
 | prepare AVM3-based scale separation:                        vg 10/08 |
 | assemble mat and rhs                                                 |
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::AVM3AssembleMatAndRHS(Teuchos::ParameterList& eleparams)
{
  // AVM3 can't be used with locsys conditions cause it hasn't been implemented yet
  if (locsysman_ != Teuchos::null) {
    dserror("AVM3 can't be used with locsys conditions cause it hasn't been implemented yet!");
  }

  // zero matrix
  sysmat_->Zero();

  // add Neumann loads
  // has been set to zero before
  residual_->Update(1.0,*neumann_loads_,1.0);

  // set action type
  eleparams.set<int>("action",FLD::calc_fluid_systemmat_and_residual);
  eleparams.set<int>("physical type",physicaltype_);

  // parameters for turbulence approach
  eleparams.sublist("TURBULENCE MODEL") = params_->sublist("TURBULENCE MODEL");
  // dummy vectors initialized with zeros
  // see remark fine scale velocity vector
  if (turbmodel_==INPAR::FLUID::scale_similarity
   or turbmodel_==INPAR::FLUID::scale_similarity_basic)
  {
    //compute filtered velocity
    //set filtered velocity
    eleparams.set("Filtered velocity",filteredvel_);
    eleparams.set("Fine scale velocity",finescalevel_);
    eleparams.set("Filtered reynoldsstress",filteredreystr_);
  }

  // set general vector values needed by elements
  discret_->ClearState();
  discret_->SetState("hist" ,hist_ );
  discret_->SetState("veln" ,veln_ );
  discret_->SetState("accam",accam_);
  // this vector contains only zeros unless SetIterScalarFields is called
  // as this function has not been called yet
  // we have to replace the zeros by ones
  // otherwise nans are occur
  scaaf_->PutScalar(1.0);
  discret_->SetState("scaaf",scaaf_);
  scaam_->PutScalar(1.0);
  discret_->SetState("scaam",scaam_);

  // set fine-scale vector
  // dummy vector initialized with zeros
  // Remark:
  // This is necessary because the fssgv_ flag
  // has already been set in SetParameters()
  // Therefore, the function Evaluate() already
  // expects the state vector "fsvelaf" and "fsscaaf" for loma
  if (fssgv_ != INPAR::FLUID::no_fssgv or turbmodel_ == INPAR::FLUID::multifractal_subgrid_scales)
  {
    discret_->SetState("fsvelaf",fsvelaf_);
    if (turbmodel_ == INPAR::FLUID::multifractal_subgrid_scales)
      discret_->SetState("fsscaaf",fsscaaf_);
  }

  if (alefluid_)
  {
    discret_->SetState("dispnp", dispnp_);
    discret_->SetState("gridv", gridv_);
  }

  // set scheme-specific element parameters and vector values
  // set the only required state vectors
  SetStateTimInt();

  // set external volume force (required, e.g., for forced homogeneous isotropic turbulence)
  // dummy
  if (forcing_!=Teuchos::null)
  {
    eleparams.set("forcing",true);
    discret_->SetState("forcing",forcing_);
  }

  // element evaluation for getting system matrix
  // -> we merely need matrix "structure" below, not the actual contents
  discret_->Evaluate(eleparams,sysmat_,Teuchos::null,residual_,Teuchos::null,Teuchos::null);
  discret_->ClearState();
  // reset the vector modified above
  scaaf_->PutScalar(0.0);
  scaam_->PutScalar(0.0);

  // complete system matrix
  sysmat_->Complete();

  // apply DBC to system matrix
  LINALG::ApplyDirichlettoSystem(sysmat_,incvel_,residual_,zeros_,*(dbcmaps_->CondMap()));

  return;
}


/*----------------------------------------------------------------------*
 | prepare AVM3-based scale separation:                         vg 10/08 |
 | get scale separation matrix                                  bk 12/13 |
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::AVM3GetScaleSeparationMatrix()
{
  // this is important to have!!!
  // MLAPI::Init() without arguments uses internally MPI_COMM_WOLRD
  MLAPI::Init();

  // extract the ML parameters:
  Teuchos::ParameterList&  mlparams = solver_->Params().sublist("ML Parameters");
  // remark: we create a new solver with ML preconditioner here, since this allows for also using other solver setups
  // to solve the system of equations
  // get the solver number used form the multifractal subgrid-scale model parameter list
  const int scale_sep_solvernumber = params_->sublist("MULTIFRACTAL SUBGRID SCALES").get<int>("ML_SOLVER");
  if (scale_sep_solvernumber != (-1))    // create a dummy solver
  {
    Teuchos::RCP<LINALG::Solver> solver = Teuchos::rcp(new LINALG::Solver(DRT::Problem::Instance()->SolverParams(scale_sep_solvernumber),
                                          discret_->Comm(),
                                          DRT::Problem::Instance()->ErrorFile()->Handle()));
    // compute the null space,
    discret_->ComputeNullSpaceIfNecessary(solver->Params(),true);
    // and, finally, extract the ML parameters
    mlparams = solver->Params().sublist("ML Parameters");
  }

  // get toggle vector for Dirchlet boundary conditions
  const Teuchos::RCP<const Epetra_Vector> dbct = Dirichlet();

  // get nullspace parameters
  double* nullspace = mlparams.get("null space: vectors",(double*)NULL);
  if (!nullspace) dserror("No nullspace supplied in parameter list");
  int nsdim = mlparams.get("null space: dimension",1);

  // modify nullspace to ensure that DBC are fully taken into account
  if (nullspace)
  {
    const int length = SystemMatrix()->OperatorRangeMap().NumMyElements();
    for (int i=0; i<nsdim; ++i)
      for (int j=0; j<length; ++j)
        if ((*dbct)[j]!=0.0) nullspace[i*length+j] = 0.0;
  }

  // get plain aggregation Ptent
  Teuchos::RCP<Epetra_CrsMatrix> crsPtent;
  MLAPI::GetPtent(*SystemMatrix()->EpetraMatrix(),mlparams,nullspace,crsPtent);
  LINALG::SparseMatrix Ptent(crsPtent);

  // compute scale-separation matrix: S = I - Ptent*Ptent^T
  Sep_ = LINALG::Multiply(Ptent,false,Ptent,true);
  Sep_->Scale(-1.0);
  Teuchos::RCP<Epetra_Vector> tmp = LINALG::CreateVector(Sep_->RowMap(),false);
  tmp->PutScalar(1.0);
  Teuchos::RCP<Epetra_Vector> diag = LINALG::CreateVector(Sep_->RowMap(),false);
  Sep_->ExtractDiagonalCopy(*diag);
  diag->Update(1.0,*tmp,1.0);
  Sep_->ReplaceDiagonalValues(*diag);

  //complete scale-separation matrix and check maps
  Sep_->Complete(Sep_->DomainMap(),Sep_->RangeMap());
  if (!Sep_->RowMap().SameAs(SystemMatrix()->RowMap())) dserror("rowmap not equal");
  if (!Sep_->RangeMap().SameAs(SystemMatrix()->RangeMap())) dserror("rangemap not equal");
  if (!Sep_->DomainMap().SameAs(SystemMatrix()->DomainMap())) dserror("domainmap not equal");

  return;
}

/*----------------------------------------------------------------------*
 | AVM3-based scale separation                                 vg 10/08 |
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::AVM3Separation()
{
  // time measurement: avm3
  TEUCHOS_FUNC_TIME_MONITOR("           + avm3");

  // get fine-scale part of velocity at time n+alpha_F or n+1
  Sep_Multiply();

  // set fine-scale vector
  discret_->SetState("fsvelaf",fsvelaf_);

  return;
}// FluidImplicitTimeInt::AVM3Separation


/*----------------------------------------------------------------------*
 |  set initial flow field for test cases                    gammi 04/07|
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::SetInitialFlowField(
  const INPAR::FLUID::InitialField initfield,
  const int startfuncno
  )
{
  // initial field by (undisturbed) function (init==2)
  // or disturbed function (init==3)
  if (initfield == INPAR::FLUID::initfield_field_by_function or
      initfield == INPAR::FLUID::initfield_disturbed_field_from_function)
  {
    // loop all nodes on the processor
    for(int lnodeid=0;lnodeid<discret_->NumMyRowNodes();lnodeid++)
    {
      // get the processor local node
      DRT::Node*  lnode      = discret_->lRowNode(lnodeid);
      // the set of degrees of freedom associated with the node
      const std::vector<int> nodedofset = discret_->Dof(0,lnode);

      for(int index=0;index<numdim_+1;++index)
      {
        int gid = nodedofset[index];

        double initialval=DRT::Problem::Instance()->Funct(startfuncno-1).Evaluate(index,lnode->X(),time_,NULL);

        velnp_->ReplaceGlobalValues(1,&initialval,&gid);
      }
    }

    // for NURBS discretizations we have to solve a least squares problem,
    // with high accuracy! (do nothing for Lagrangian polynomials)
    DRT::NURBS::apply_nurbs_initial_condition(
      *discret_,
      DRT::Problem::Instance()->ErrorFile()->Handle(),
      DRT::Problem::Instance()->UMFPACKSolverParams(),
      startfuncno,
      velnp_);

    // initialize veln_ as well. That's what we actually want to do here!
    veln_->Update(1.0,*velnp_ ,0.0);

    // add random perturbation of certain percentage to function
    if (initfield == INPAR::FLUID::initfield_disturbed_field_from_function)
    {
      const Epetra_Map* dofrowmap = discret_->DofRowMap();

      int err =0;

      // random noise is perc percent of the initial profile
      double perc = params_->sublist("TURBULENCE MODEL").get<double>("CHAN_AMPL_INIT_DIST",0.1);

      // out to screen
      if (myrank_==0)
      {
        std::cout << "Disturbed initial profile:   max. " << perc*100 << "% random perturbation\n";
        std::cout << "\n\n";
      }

      double bmvel=0;
      double mybmvel=0;
      double thisvel=0;
      // loop all nodes on the processor
      for(int lnodeid=0;lnodeid<discret_->NumMyRowNodes();++lnodeid)
      {
        // get the processor local node
        DRT::Node*  lnode      = discret_->lRowNode(lnodeid);
        // the set of degrees of freedom associated with the node
        std::vector<int> nodedofset = discret_->Dof(lnode);

        for(int index=0;index<numdim_;++index)
        {
          int gid = nodedofset[index];
          int lid = dofrowmap->LID(gid);

          thisvel=(*velnp_)[lid];
          if (mybmvel*mybmvel < thisvel*thisvel)
            mybmvel=thisvel;
        }
      }

      // the noise is proportional to the bulk mean velocity of the
      // undisturbed initial field (=2/3*maximum velocity)
      mybmvel=2*mybmvel/3;
      discret_->Comm().MaxAll(&mybmvel,&bmvel,1);

      // loop all nodes on the processor
      for(int lnodeid=0;lnodeid<discret_->NumMyRowNodes();++lnodeid)
      {
        // get the processor local node
        DRT::Node*  lnode      = discret_->lRowNode(lnodeid);
        // the set of degrees of freedom associated with the node
        std::vector<int> nodedofset = discret_->Dof(lnode);

        // check whether we have a pbc condition on this node
        std::vector<DRT::Condition*> mypbc;

        lnode->GetCondition("SurfacePeriodic",mypbc);

        // check whether a periodic boundary condition is active on this node
        if (mypbc.size()>0)
        {
          // yes, we have one

          // get the list of all his slavenodes
          std::map<int, std::vector<int> >::iterator master = (discret_->GetAllPBCCoupledColNodes())->find(lnode->Id());

          // slavenodes are ignored
          if(master == (discret_->GetAllPBCCoupledColNodes())->end()) continue;
        }

        // add random noise on initial function field
        for (int index = 0; index < numdim_; ++index)
        {
          int gid = nodedofset[index];

          double randomnumber = DRT::Problem::Instance()->Random()->Uni();

          double noise = perc * bmvel * randomnumber;

          err += velnp_->SumIntoGlobalValues(1,&noise,&gid);
          err += veln_ ->SumIntoGlobalValues(1,&noise,&gid);
        }

        if (err != 0)
        {
          dserror("dof not on proc");
        }
      }
      //meshtying: this is necessary for the disturbed field. the interface does not work
      //otherwise due to the non-linear terms in the matrix.
      if (msht_ != INPAR::FLUID::no_meshtying)
      {
        meshtying_->UpdateSlaveDOF(velnp_,velnp_);
        meshtying_->UpdateSlaveDOF(veln_,veln_);
      }
    }
  }
  // special initial function: two counter-rotating vortices (2-D) and flame front
  // for flame-vortex interaction problem
  else if (initfield == INPAR::FLUID::initfield_flame_vortex_interaction)
  {
    const Epetra_Map* dofrowmap = discret_->DofRowMap();

    int err = 0;

    // define vectors for velocity field, node coordinates and coordinates
    // of left and right vortex
    std::vector<double> u(numdim_);
    std::vector<double> xy(numdim_);
    std::vector<double> xy0_left(numdim_);
    std::vector<double> xy0_right(numdim_);

    // check whether present flow is indeed two-dimensional
    if (numdim_!=2) dserror("Counter-rotating vortices are a two-dimensional flow!");

    // set laminar burning velocity, vortex strength C (scaled by laminar
    // burning velocity and (squared) vortex radius R
    const double sl = 1.0;
    const double C = 70.0*sl;
    const double R_squared = 16.0;

    // set density in unburnt and burnt phase and initialize actual density
    const double densu = 1.161;
    // -> for "pure fluid" computation: rhob = rhou = 1.161
    //const double densb = 1.161;
    const double densb = 0.157;
    double dens = 1.161;

    // initialize progress variable
    double pv = 0.0;

    // variables for evaluation of progress-variable profile
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

    // set (scaled) vortex strength C, (squared) vortex radius R and define variables
    double r_squared_left;
    double r_squared_right;

    // set initial locations of vortices
    xy0_left[0] = 37.5;
    xy0_left[1] = 75.0;
    xy0_right[0] = 62.5;
    xy0_right[1] = 75.0;

    // loop all nodes on the processor
    for(int lnodeid=0;lnodeid<discret_->NumMyRowNodes();lnodeid++)
    {
      // get the processor local node
      DRT::Node*  lnode      = discret_->lRowNode(lnodeid);

      // the set of degrees of freedom associated with the node
      std::vector<int> nodedofset = discret_->Dof(lnode);

      // set node coordinates
      for(int dim=0;dim<numdim_;dim++)
      {
        xy[dim]=lnode->X()[dim];
      }

      // compute preliminary values for both vortices
      r_squared_left  = ((xy[0]-xy0_left[0])*(xy[0]-xy0_left[0])
                        +(xy[1]-xy0_left[1])*(xy[1]-xy0_left[1]))/R_squared;
      r_squared_right = ((xy[0]-xy0_right[0])*(xy[0]-xy0_right[0])
                        +(xy[1]-xy0_right[1])*(xy[1]-xy0_right[1]))/R_squared;

      // compute value of progress variable
      if (xy[1] < loc12-EPS10)
        pv = (1.0-(1.0/beta1))*exp((xy[1]-trans1)/delta1);
      else if (xy[1] > loc23+EPS10)
        pv = 1.0-(exp((1.0-beta3)*(xy[1]-trans3)/delta3)/beta3);
      else
        pv = fac2*(xy[1]-trans2) + abs2;

      // compute current density
      dens = densu+(densb-densu)*pv;

      // compute initial velocity components
      // including initial velocity distribution velocity in x2-direction
      u[0] = (C/R_squared)*(-(xy[1]-xy0_left[1])*exp(-r_squared_left/2.0)
                            +(xy[1]-xy0_right[1])*exp(-r_squared_right/2.0));
      u[1] = (C/R_squared)*( (xy[0]-xy0_left[0])*exp(-r_squared_left/2.0)
                            -(xy[0]-xy0_right[0])*exp(-r_squared_right/2.0))
                            + sl*densu/dens;

      // velocity profile due to flame without vortices:
      //u[1] = sl*densu/dens;

      // set initial velocity components
      for(int nveldof=0;nveldof<numdim_;nveldof++)
      {
        const int gid = nodedofset[nveldof];
        int lid = dofrowmap->LID(gid);
        err += velnp_->ReplaceMyValues(1,&(u[nveldof]),&lid);
        err += veln_ ->ReplaceMyValues(1,&(u[nveldof]),&lid);
        err += velnm_->ReplaceMyValues(1,&(u[nveldof]),&lid);
      }
    } // end loop nodes lnodeid

    if (err!=0) dserror("dof not on proc");
  }
  // special initial function: Beltrami flow (3-D)
  else if (initfield == INPAR::FLUID::initfield_beltrami_flow)
  {
    const Epetra_Map* dofrowmap = discret_->DofRowMap();

    int err = 0;

    const int npredof = numdim_;

    double         p;
    std::vector<double> u  (numdim_);
    std::vector<double> acc(numdim_);
    std::vector<double> xyz(numdim_);

    // check whether present flow is indeed three-dimensional
    if (numdim_!=3) dserror("Beltrami flow is a three-dimensional flow!");

    // set constants for analytical solution
    const double a = M_PI/4.0;
    const double d = M_PI/2.0;

    // loop all nodes on the processor
    for(int lnodeid=0;lnodeid<discret_->NumMyRowNodes();lnodeid++)
    {
      // get the processor local node
      DRT::Node*  lnode      = discret_->lRowNode(lnodeid);

      // the set of degrees of freedom associated with the node
      std::vector<int> nodedofset = discret_->Dof(lnode);

      // set node coordinates
      for(int dim=0;dim<numdim_;dim++)
      {
        xyz[dim]=lnode->X()[dim];
      }

      // compute initial velocity components
      u[0] = -a * ( exp(a*xyz[0]) * sin(a*xyz[1] + d*xyz[2]) +
                    exp(a*xyz[2]) * cos(a*xyz[0] + d*xyz[1]) );
      u[1] = -a * ( exp(a*xyz[1]) * sin(a*xyz[2] + d*xyz[0]) +
                    exp(a*xyz[0]) * cos(a*xyz[1] + d*xyz[2]) );
      u[2] = -a * ( exp(a*xyz[2]) * sin(a*xyz[0] + d*xyz[1]) +
                    exp(a*xyz[1]) * cos(a*xyz[2] + d*xyz[0]) );

      // compute initial pressure
      int id = DRT::Problem::Instance()->Materials()->FirstIdByType(INPAR::MAT::m_fluid);
      if (id==-1) dserror("Newtonian fluid material could not be found");
      const MAT::PAR::Parameter* mat = DRT::Problem::Instance()->Materials()->ParameterById(id);
      const MAT::PAR::NewtonianFluid* actmat = static_cast<const MAT::PAR::NewtonianFluid*>(mat);
      double dens = actmat->density_;
      double visc = actmat->viscosity_;

      p = -a*a/2.0 * dens *
        ( exp(2.0*a*xyz[0])
          + exp(2.0*a*xyz[1])
          + exp(2.0*a*xyz[2])
          + 2.0 * sin(a*xyz[0] + d*xyz[1]) * cos(a*xyz[2] + d*xyz[0]) * exp(a*(xyz[1]+xyz[2]))
          + 2.0 * sin(a*xyz[1] + d*xyz[2]) * cos(a*xyz[0] + d*xyz[1]) * exp(a*(xyz[2]+xyz[0]))
          + 2.0 * sin(a*xyz[2] + d*xyz[0]) * cos(a*xyz[1] + d*xyz[2]) * exp(a*(xyz[0]+xyz[1]))
          );

      // Beltrami is always 3D
      acc[0] = u[0]*(-1.0*d*d*visc/dens);
      acc[1] = u[1]*(-1.0*d*d*visc/dens);
      acc[2] = u[2]*(-1.0*d*d*visc/dens);

      // set initial velocity components
      for(int nveldof=0;nveldof<numdim_;nveldof++)
      {
        const int gid = nodedofset[nveldof];
        int lid = dofrowmap->LID(gid);
        err += velnp_->ReplaceMyValues(1,&(u[nveldof]),&lid);
        err += veln_ ->ReplaceMyValues(1,&(u[nveldof]),&lid);
        err += velnm_->ReplaceMyValues(1,&(u[nveldof]),&lid);

        // set additionally the values for the time derivative to start with an exact acceleration in case of OST (theta!=1.0)
        // set initial acceleration components
        err += accnp_->ReplaceMyValues(1,&(acc[nveldof]),&lid);
        err += accn_ ->ReplaceMyValues(1,&(acc[nveldof]),&lid);
        err += accam_->ReplaceMyValues(1,&(acc[nveldof]),&lid);
      }

      // set initial pressure
      const int gid = nodedofset[npredof];
      int lid = dofrowmap->LID(gid);
      err += velnp_->ReplaceMyValues(1,&p,&lid);
      err += veln_ ->ReplaceMyValues(1,&p,&lid);
      err += velnm_->ReplaceMyValues(1,&p,&lid);
    } // end loop nodes lnodeid

    if (err!=0) dserror("dof not on proc");
  }
  // special initial function: test case due to Bochev et al. (2007) (2-D)
  else if (initfield == INPAR::FLUID::initfield_bochev_test)
  {
    const Epetra_Map* dofrowmap = discret_->DofRowMap();

    int err = 0;

    // check whether present flow is indeed two-dimensional
    if (numdim_!=2) dserror("Bochev test case is a two-dimensional flow!");

    // define vectors for velocity and pressure field as well as node coordinates
    std::vector<double> up(numdim_+1);
    std::vector<double> xy(numdim_);

    // loop all nodes on the processor
    for(int lnodeid=0;lnodeid<discret_->NumMyRowNodes();lnodeid++)
    {
      // get the processor local node
      DRT::Node*  lnode      = discret_->lRowNode(lnodeid);

      // the set of degrees of freedom associated with the node
      std::vector<int> nodedofset = discret_->Dof(lnode);

      // set node coordinates
      for(int dim=0;dim<numdim_;dim++)
      {
        xy[dim]=lnode->X()[dim];
      }

      // compute initial velocity and pressure components
      up[0] = sin(M_PI*xy[0]-0.7)*sin(M_PI*xy[1]+0.2);
      up[1] = cos(M_PI*xy[0]-0.7)*cos(M_PI*xy[1]+0.2);
      up[2] = sin(xy[0])*cos(xy[1])+(cos(1.0)-1.0)*sin(1.0);

      // set initial velocity and pressure components
      for(int ndof=0;ndof<numdim_+1;ndof++)
      {
        const int gid = nodedofset[ndof];
        int lid = dofrowmap->LID(gid);
        err += velnp_->ReplaceMyValues(1,&(up[ndof]),&lid);
        err += veln_ ->ReplaceMyValues(1,&(up[ndof]),&lid);
        err += velnm_->ReplaceMyValues(1,&(up[ndof]),&lid);
      }
    } // end loop nodes lnodeid

    if (err!=0) dserror("dof not on proc");
  }
  else if (initfield == INPAR::FLUID::initfield_hit_comte_bellot_corrsin
           or initfield == INPAR::FLUID::initfield_forced_hit_simple_algebraic_spectrum
           or initfield == INPAR::FLUID::initfield_forced_hit_numeric_spectrum
           or initfield == INPAR::FLUID::initfield_passive_hit_const_input)
  {
    // initialize calculation of initial field based on fast Fourier transformation
    Teuchos::RCP<HomIsoTurbInitialField> HitInitialField = Teuchos::rcp(new FLD::HomIsoTurbInitialField(*this,initfield));
    // calculate initial field
    HitInitialField->CalculateInitialField();

    // get statistics of initial field
    CallStatisticsManager();

    // initialize  forcing depending on initial field
    homisoturb_forcing_->SetInitialSpectrum(initfield);
  }
  else
  {
    dserror("Only initial fields such as a zero field, initial fields by (un-)disturbed functions, three special initial fields (counter-rotating vortices, Beltrami flow and Bochev test) as well as initial fields for homegeneous isotropic turbulence are available up to now!");
  }

  return;
} // end SetInitialFlowField


/*----------------------------------------------------------------------*
 | set fields for scatra - fluid coupling, esp.                         |
 | set fields for low-Mach-number flow within iteration loop   vg 09/09 |
 | overloaded in TimIntLoma                                    bk 12/13 |
 | overloaded in TimIntTwoPhase                                mw 07/14 |
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::SetIterScalarFields(
   Teuchos::RCP<const Epetra_Vector> scalaraf,
   Teuchos::RCP<const Epetra_Vector> scalaram,
   Teuchos::RCP<const Epetra_Vector> scalardtam,
   Teuchos::RCP<DRT::Discretization> scatradis)
{
  // initializations
  int err(0);
  double value(0.0);
  std::vector<int> nodedofs;

  //--------------------------------------------------------------------------
  // Filling the scaaf-vector and scaam-vector at time n+alpha_F/n+1 and
  // n+alpha_M/n, respectively, with scalar at pressure dofs
  // Additionally, filling the scaam-vector at time n+alpha_M/n with
  // velocity at time n at velocity dofs for OST/BDF2
  // Filling the accam-vector at time n+alpha_M/n+1, respectively, with
  // scalar time derivative values at pressure dofs
  //--------------------------------------------------------------------------
  // get velocity values at time n in scaam-vector as copy from veln-vector
  scaam_->Update(1.0,*veln_,0.0);

  // loop all nodes on the processor
  for(int lnodeid=0;lnodeid<discret_->NumMyRowNodes();lnodeid++)
  {
    // get the processor's local scatra node
    DRT::Node* lscatranode = scatradis->lRowNode(lnodeid);

    // find out the global dof id of the last(!) dof at the scatra node
    const int numscatradof = scatradis->NumDof(0,lscatranode);
    const int globalscatradofid = scatradis->Dof(0,lscatranode,numscatradof-1);
    const int localscatradofid = scalaraf->Map().LID(globalscatradofid);
    if (localscatradofid < 0)
      dserror("localdofid not found in map for given globaldofid");

    // get the processor's local fluid node
    DRT::Node* lnode = discret_->lRowNode(lnodeid);
    // get the global ids of degrees of freedom associated with this node
    nodedofs = discret_->Dof(0,lnode);
    // get global and processor's local pressure dof id (using the map!)
    const int numdof = discret_->NumDof(0,lnode);
    const int globaldofid = discret_->Dof(0,lnode,numdof-1);
    const int localdofid = scaam_->Map().LID(globaldofid);
    if (localdofid < 0)
      dserror("localdofid not found in map for given globaldofid");

    // now copy the values
    value = (*scalaraf)[localscatradofid];
    err = scaaf_->ReplaceMyValue(localdofid,0,value);
    if (err != 0) dserror("error while inserting value into scaaf_");

    value = (*scalaram)[localscatradofid];
    err = scaam_->ReplaceMyValue(localdofid,0,value);
    if (err != 0) dserror("error while inserting value into scaam_");

    if (scalardtam != Teuchos::null)
    {
      value = (*scalardtam)[localscatradofid];
    }
    else
    {
      value = 0.0; // for safety reasons: set zeros in accam_
    }
    err = accam_->ReplaceMyValue(localdofid,0,value);
    if (err != 0) dserror("error while inserting value into accam_");

  }
  return;
} // FluidImplicitTimeInt::SetIterScalarFields

/*----------------------------------------------------------------------*
 | set fields for scatra - fluid coupling, esp.                         |
 | set scalar fields     vg 09/09 |
 | overloaded in TimIntLoma                                    bk 12/13 |
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::SetScalarFields(
   Teuchos::RCP<const Epetra_Vector> scalarnp,
   const double             thermpressnp,
   Teuchos::RCP<const Epetra_Vector> scatraresidual,
   Teuchos::RCP<DRT::Discretization> scatradis,
   const int                whichscalar)
{
  // initializations
  int err(0);
  double value(0.0);
  std::vector<int> nodedofs;

  //--------------------------------------------------------------------------
  // Filling the scaaf-vector with scalar at time n+1 at pressure dofs
  //--------------------------------------------------------------------------
  // loop all nodes on the processor
  for(int lnodeid=0;lnodeid<discret_->NumMyRowNodes();lnodeid++)
  {
    // get the processor's local scatra node
    DRT::Node* lscatranode = scatradis->lRowNode(lnodeid);

    // find out the global dof id of the last(!) dof at the scatra node
    const int numscatradof = scatradis->NumDof(0,lscatranode);
    int globalscatradofid(-1);
    if (whichscalar == (-1))
    {
      // default: always take the LAST scatra dof at each node
      globalscatradofid = scatradis->Dof(0,lscatranode,numscatradof-1);
    }
    else
    {
      // respect the explicit wish of the user
      globalscatradofid = scatradis->Dof(0,lscatranode,whichscalar);
    }
    const int localscatradofid = scalarnp->Map().LID(globalscatradofid);
    if (localscatradofid < 0)
      dserror("localdofid not found in map for given globaldofid");

    // get the processor's local fluid node
    DRT::Node* lnode = discret_->lRowNode(lnodeid);
    // get the global ids of degrees of freedom associated with this node
    nodedofs = discret_->Dof(0,lnode);
    // get global and processor's local pressure dof id (using the map!)
    const int globaldofid = nodedofs[numdim_];
    const int localdofid = scaam_->Map().LID(globaldofid);
    if (localdofid < 0)
      dserror("localdofid not found in map for given globaldofid");

    value = (*scalarnp)[localscatradofid];
    err = scaaf_->ReplaceMyValue(localdofid,0,value);
    if (err != 0) dserror("error while inserting value into scaaf_");

    //--------------------------------------------------------------------------
    // Filling the trueresidual vector with scatraresidual at pre-dofs
    //--------------------------------------------------------------------------
    if (scatraresidual != Teuchos::null)
    {
      value = (*scatraresidual)[localscatradofid];
      trueresidual_->ReplaceMyValue(localdofid,0,value);
    }

  }


  return;

} // FluidImplicitTimeInt::SetScalarFields

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Vector> FLD::FluidImplicitTimeInt::ExtractVelocityPart(Teuchos::RCP<const Epetra_Vector> velpres)
{
   return VelPresSplitter().ExtractOtherVector(velpres);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Vector> FLD::FluidImplicitTimeInt::ExtractPressurePart(Teuchos::RCP<const Epetra_Vector> velpres)
{
   return VelPresSplitter().ExtractCondVector(velpres);
}

/*----------------------------------------------------------------------*
 | evaluate error for test cases with analytical solutions   gammi 04/07|
 *----------------------------------------------------------------------*/
Teuchos::RCP<std::vector<double> > FLD::FluidImplicitTimeInt::EvaluateErrorComparedToAnalyticalSol()
{

  INPAR::FLUID::CalcError calcerr = DRT::INPUT::get<INPAR::FLUID::CalcError>(*params_,"calculate error");

  switch(calcerr)
  {
  case INPAR::FLUID::no_error_calculation:
  {
    // do nothing --- no analytical solution available
    return Teuchos::null;
    break;
  }
  case INPAR::FLUID::beltrami_flow:
  case INPAR::FLUID::channel2D:
  case INPAR::FLUID::topoptchannel:
  case INPAR::FLUID::gravitation:
  case INPAR::FLUID::shear_flow:
  case INPAR::FLUID::fsi_fluid_pusher:
  case INPAR::FLUID::byfunct1:
  {
    // std::vector containing
    // [0]: relative L2 velocity error
    // [1]: relative L2 pressure error
    // [2]: relative H1 velocity error
    Teuchos::RCP<std::vector<double> > relerror = Teuchos::rcp(new std::vector<double>(3));

    // create the parameters for the discretization
    Teuchos::ParameterList eleparams;

    // action for elements
    eleparams.set<int>("action",FLD::calc_fluid_error);
    eleparams.set<int>("calculate error",calcerr);

    // set scheme-specific element parameters and vector values
    SetStateTimInt();

    if (alefluid_)
    {
      discret_->SetState("dispnp", dispnp_);
    }

    // get (squared) error values
    // 0: delta velocity for L2-error norm
    // 1: delta p for L2-error norm
    // 2: delta velocity for H1-error norm
    // 3: analytical velocity for L2 norm
    // 4: analytical p for L2 norm
    // 5: analytical velocity for H1 norm
    Teuchos::RCP<Epetra_SerialDenseVector> errors
      = Teuchos::rcp(new Epetra_SerialDenseVector(3+3));

    // call loop over elements (assemble nothing)
    discret_->EvaluateScalars(eleparams, errors);
    discret_->ClearState();

    (*relerror)[0] = sqrt((*errors)[0])/sqrt((*errors)[3]);
    (*relerror)[1] = sqrt((*errors)[1])/sqrt((*errors)[4]);

    if ((calcerr==INPAR::FLUID::beltrami_flow) or
        (calcerr==INPAR::FLUID::topoptchannel))
      (*relerror)[2] = sqrt((*errors)[2])/sqrt((*errors)[5]);
    else
    {
      (*relerror)[2] = 0.0;
      if (myrank_ == 0)
      {
        std::cout << std::endl << "Warning: H_1 velocity error norm for analytical solution Nr. " <<
          DRT::INPUT::get<INPAR::FLUID::CalcError>(*params_,"calculate error") <<
          " is not implemented yet!!" << std::endl;
      }
    }

    if (myrank_ == 0)
    {
      {
        std::cout.precision(8);
        std::cout << std::endl << "---- error norm for analytical solution Nr. " <<
          DRT::INPUT::get<INPAR::FLUID::CalcError>(*params_,"calculate error") <<
          " ----------" << std::endl;
        std::cout << "| relative L_2 velocity error norm:     " << (*relerror)[0] << std::endl;
        std::cout << "| relative L_2 pressure error norm:     " << (*relerror)[1] << std::endl;
        if ((*relerror)[2] != 0.0)
          std::cout << "| relative H_1 velocity error norm:     " << (*relerror)[2] << std::endl;
        std::cout << "--------------------------------------------------------------------" << std::endl << std::endl;
        if ((*relerror)[2] != 0.0)
          std::cout << "H1 velocity scaling  " << sqrt((*errors)[5]) << std::endl;
      }

      // print last error in a seperate file

      // append error of the last time step to the error file
      if ((step_==stepmax_) or (time_==maxtime_))// write results to file
      {
        std::ostringstream temp;
        const std::string simulation = DRT::Problem::Instance()->OutputControlFile()->FileName();
        const std::string fname = simulation+".relerror";

        std::ofstream f;
        f.open(fname.c_str(),std::fstream::ate | std::fstream::app);
        f << "#| " << simulation << "\n";
        f << "#| Step | Time | rel. L2-error velocity  |  rel. L2-error pressure  |  rel. H1-error velocity  |\n";
        f << step_ << " " << time_ << " " << (*relerror)[0] << " " << (*relerror)[1] << " "<< (*relerror)[2] << "\n";
        f.flush();
        f.close();
      }


      std::ostringstream temp;
      const std::string simulation = DRT::Problem::Instance()->OutputControlFile()->FileName();
      const std::string fname = simulation+"_time.relerror";

      if(step_==1)
      {
        std::ofstream f;
        f.open(fname.c_str());
        f << "#| Step | Time | rel. L2-error velocity  |  rel. L2-error pressure  |  rel. H1-error velocity  |\n";
        f << std::setprecision(10) << step_ << " " << std::setw(1)<< std::setprecision(5)
          << time_ << std::setw(1) << std::setprecision(6) << " "
          << (*relerror)[0] << std::setw(1) << std::setprecision(6) << " " << (*relerror)[1] << std::setprecision(6) << " "<< (*relerror)[2] <<"\n";

        f.flush();
        f.close();
      }
      else
      {
        std::ofstream f;
        f.open(fname.c_str(),std::fstream::ate | std::fstream::app);
        f << std::setprecision(10) << step_ << " " << std::setw(3)<< std::setprecision(5)
        << time_ << std::setw(1) << std::setprecision(6) << " "
        << (*relerror)[0] << std::setw(1) << std::setprecision(6) << " " << (*relerror)[1] << std::setprecision(6) << " "<< (*relerror)[2] <<"\n";

        f.flush();
        f.close();
      }
    }
    return relerror;
  }
  break;
  default:
    dserror("Cannot calculate error. Unknown type of analytical test problem");
    break;
  }
  return Teuchos::null;
} // end EvaluateErrorComparedToAnalyticalSol


/*----------------------------------------------------------------------*
 | evaluate divergence u                                      ehrl 12/12|
 *----------------------------------------------------------------------*/
Teuchos::RCP<double> FLD::FluidImplicitTimeInt::EvaluateDivU()
{
  // Evaluate div u only at the last step
  //if ((step_==stepmax_) or (time_==maxtime_))// write results to file
  if (params_->get<bool>("COMPUTE_DIVU"))
  {
    // create the parameters for the discretization
    Teuchos::ParameterList eleparams;

    // action for elements
    eleparams.set<int>("action",FLD::calc_div_u);

    // set vector values needed by elements
    // div u is always evaluated at time n+af (generalized alpha time integration schemes) and
    // at time  n+1 (one-step-theta)
    // set scheme-specific element parameters and vector values
    // continuity equation in np-genalpha is also evaluated at time n+1
    SetStateTimInt();

    const Epetra_Map* elementrowmap = discret_->ElementRowMap();
    Teuchos::RCP<Epetra_MultiVector> divu = Teuchos::rcp(new Epetra_MultiVector(*elementrowmap,1,true));

    // optional: elementwise defined div u may be written to standard output file (not implemented yet)
    discret_->EvaluateScalars(eleparams, divu);

    discret_->ClearState();

    double maxdivu = 0.0;
    Teuchos::RCP<double> sumdivu = Teuchos::rcp(new double(0.0));
    divu->Norm1(&(*sumdivu));
    divu->NormInf(&maxdivu);

    if(myrank_==0)
    {
      std::cout << "---------------------------------------------------" << std::endl;
      std::cout << "| divergence-free condition:                      |" << std::endl;
      std::cout << "| Norm(inf) = " << maxdivu <<  " | Norm(1) = " << *sumdivu << "  |" << std::endl ;
      std::cout << "---------------------------------------------------" << std::endl << std::endl;

      const std::string simulation = DRT::Problem::Instance()->OutputControlFile()->FileName();
      const std::string fname = simulation+".divu";

      std::ofstream f;
      f.open(fname.c_str());
      if(step_==1)
      {
        f << "#| " << simulation << "\n";
        f << "#| Step | Time | max. div u | div u (Norm(1)) |\n";
        f << step_ << " " << time_ << " " << maxdivu << " " << *sumdivu << " " << "\n";
        f.flush();
        f.close();
      }
      else
      {
        f << step_ << " " << time_ << " " << maxdivu << " " << *sumdivu << " " << "\n";
        f.flush();
        f.close();
      }
    }
    return sumdivu;
  }
  else
    return Teuchos::null;
} // end EvaluateDivU

/*----------------------------------------------------------------------*
 | calculate adaptive time step with the CFL number             bk 08/14|
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::EvaluateDtWithCFL()
{
  double CFL = params_->get<double>("CFL_NUMBER",-1.0);
  int stependadaptivedt =     params_->get<int>   ("FREEZE_ADAPTIVE_DT_AT",10000000);
  if(step_+1==stependadaptivedt&&myrank_==0)
    std::cout << "\n    !!time step is kept constant from now on for sampling of turbulence statistics!!\n" << std::endl;
  if(CFL>0.0 && step_+1<stependadaptivedt)
  {
    // create the parameters for the discretization
    Teuchos::ParameterList eleparams;

    // action for elements
    eleparams.set<int>("action",FLD::calc_dt_via_cfl);

    discret_->SetState("velnp", velnp_);

    const Epetra_Map* elementrowmap = discret_->ElementRowMap();
    Teuchos::RCP<Epetra_MultiVector> h_u = Teuchos::rcp(new Epetra_MultiVector(*elementrowmap,1,true));

    // optional: elementwise defined div u may be written to standard output file (not implemented yet)
    discret_->EvaluateScalars(eleparams, h_u);

    discret_->ClearState();

    double min_h_u = 0.0;

    h_u->MinValue(&min_h_u);

    //if the initial velocity field is zero and there are no non-zero Dirichlet-boundaries,
    //min_h_u is zero. In this case, we use the time step stated in the input file
    //and write this to the screen
    if(min_h_u<1.0e3)
    {
      if(step_>0)
        dta_+=0.8*(CFL*min_h_u-dtp_);
      else //start of simulation
        dta_=CFL*min_h_u;
    }
    else if(myrank_==0)
      std::cout << "Calculated time step is zero due to zero velocity field: use time step stated in input file for the first step!" << std::endl;
  }

  return;
} // end EvaluateDtWithCFL



/*----------------------------------------------------------------------*
 |  calculate traction vector at (Dirichlet) boundary (public) gjb 07/07|
 *----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector> FLD::FluidImplicitTimeInt::CalcStresses()
{
  std::string condstring("FluidStressCalc");
  Teuchos::RCP<Epetra_Vector> integratedshapefunc = IntegrateInterfaceShape(condstring);

  // compute traction values at specified nodes; otherwise do not touch the zero values
  for (int i=0;i<integratedshapefunc->MyLength();i++)
  {
    if ((*integratedshapefunc)[i] != 0.0)
    {
      // overwrite integratedshapefunc values with the calculated traction coefficients,
      // which are reconstructed out of the nodal forces (trueresidual_) using the
      // same shape functions on the boundary as for velocity and pressure.
      (*integratedshapefunc)[i] = (*trueresidual_)[i]/(*integratedshapefunc)[i];
    }
  }

  return integratedshapefunc;

} // FluidImplicitTimeInt::CalcStresses()


/*----------------------------------------------------------------------*
 | Destructor dtor (public)                                  gammi 04/07|
 *----------------------------------------------------------------------*/
FLD::FluidImplicitTimeInt::~FluidImplicitTimeInt()
{
  return;
}


/*----------------------------------------------------------------------*
 | calculate lift and drag forces as well as angular moment: chfoe 11/07|
 | Lift and drag forces are based upon the right-hand side              |
 | true-residual entities of the corresponding nodes.                   |
 | The contribution of the end node of a line is entirely               |
 | added to a present L&D force.                                        |
 | For computing the angular moment, potential displacements            |
 | are taken into account when calculating the distance to              |
 | the center of rotation.                                              |
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::LiftDrag() const
{
  // initially check whether computation of lift and drag values is required
  if (params_->get<bool>("LIFTDRAG"))
  {
    // in this map, the results of the lift drag calculation are stored
    Teuchos::RCP<std::map<int,std::vector<double> > > liftdragvals;

    // check whether there are slip supplemental curved boundary conditions
    std::vector<DRT::Condition*> slipsuppline;
    discret_->GetCondition("LineSlipSupp",slipsuppline);
    std::vector<DRT::Condition*> slipsuppsurf;
    discret_->GetCondition("SurfaceSlipSupp",slipsuppsurf);

    if (slipsuppline.size() != 0 or slipsuppsurf.size() != 0)
    {
      // Create temporary variable that holds a vector containing the real forces
      // acting on the node, i.e. taking into account the previously neglected
      // forces perpendicular to the boundary due to slip boundary condition(s)
      Teuchos::RCP<Epetra_Vector>    forces;
      forces = LINALG::CreateVector(*(discret_->DofRowMap()),true);

      forces->Update(1.0, *trueresidual_, 1.0, *slip_bc_normal_tractions_, 0.0);

      FLD::UTILS::LiftDrag(discret_,forces,dispnp_,numdim_,liftdragvals,alefluid_);
    } else {
      FLD::UTILS::LiftDrag(discret_,trueresidual_,dispnp_,numdim_,liftdragvals,alefluid_);
    }

    if (liftdragvals!=Teuchos::null and discret_->Comm().MyPID() == 0)
      FLD::UTILS::WriteLiftDragToFile(time_, step_, *liftdragvals);
  }

  return;
}


/*----------------------------------------------------------------------*
 | compute flow rates through desired boundary parts        u.may 01/10 |
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::ComputeFlowRates() const
{
  std::vector<DRT::Condition*> flowratecond;
  std::string condstring;

  if(numdim_ == 2)
  {
    condstring = "LineFlowRate";
    discret_->GetCondition("LineFlowRate", flowratecond);
    // if no flowrate condition is present we do not compute anything
    if((int) flowratecond.size()== 0)
      return;
  }
  else if (numdim_ == 3)
  {
    condstring = "SurfFlowRate";
    discret_->GetCondition("SurfFlowRate", flowratecond);
    // if no flowrate condition is present we do not compute anything
    if((int) flowratecond.size()== 0)
      return;
  }
  else
    dserror("flow rate computation is not implemented for the 1D case");

  if (alefluid_)
  {
    const std::map<int,double> flowrates = FLD::UTILS::ComputeFlowRates(*discret_, velnp_,gridv_,dispnp_, condstring);
    //const std::map<int,double> volume = FLD::UTILS::ComputeVolume(*discret_, velnp_,gridv_,dispnp_);

    // write to file
    if(discret_->Comm().MyPID() == 0)
    {
      FLD::UTILS::WriteDoublesToFile(time_, step_, flowrates,"flowrate" );
      //FLD::UTILS::WriteDoublesToFile(time_, step_, volume,"volume" );
    }
  }
  else
  {
    const std::map<int,double> flowrates = FLD::UTILS::ComputeFlowRates(*discret_, velnp_, condstring);

    // write to file
    if(discret_->Comm().MyPID() == 0)
      FLD::UTILS::WriteDoublesToFile(time_, step_, flowrates,"flowrate" );
  }

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector> FLD::FluidImplicitTimeInt::IntegrateInterfaceShape(std::string condname)
{
  Teuchos::ParameterList eleparams;
  // set action for elements
  eleparams.set<int>("action",FLD::integrate_Shapefunction);

  // get a vector layout from the discretization to construct matching
  // vectors and matrices
  //                 local <-> global dof numbering
  const Epetra_Map* dofrowmap = discret_->DofRowMap();

  // create vector (+ initialization with zeros)
  Teuchos::RCP<Epetra_Vector> integratedshapefunc = LINALG::CreateVector(*dofrowmap,true);

  // call loop over elements
  discret_->ClearState();
  if (alefluid_)
  {
    discret_->SetState("dispnp", dispnp_);
  }
  discret_->EvaluateCondition(eleparams,integratedshapefunc,condname);
  discret_->ClearState();

  return integratedshapefunc;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::UseBlockMatrix(Teuchos::RCP<std::set<int> >     condelements,
                                               const LINALG::MultiMapExtractor& domainmaps,
                                               const LINALG::MultiMapExtractor& rangemaps,
                                               bool splitmatrix)
{
  Teuchos::RCP<LINALG::BlockSparseMatrix<FLD::UTILS::InterfaceSplitStrategy> > mat;

  if (splitmatrix)
  {
    // (re)allocate system matrix
    mat = Teuchos::rcp(new LINALG::BlockSparseMatrix<FLD::UTILS::InterfaceSplitStrategy>(domainmaps,rangemaps,108,false,true));
    mat->SetCondElements(condelements);
    sysmat_ = mat;
  }

  // if we never build the matrix nothing will be done
  if (params_->get<bool>("shape derivatives"))
  {
    // allocate special mesh moving matrix
    mat = Teuchos::rcp(new LINALG::BlockSparseMatrix<FLD::UTILS::InterfaceSplitStrategy>(domainmaps,rangemaps,108,false,true));
    mat->SetCondElements(condelements);
    shapederivatives_ = mat;
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::LinearRelaxationSolve(Teuchos::RCP<Epetra_Vector> relax)
{
  // LinearRelaxationSolve can't be used with locsys conditions cause it hasn't been implemented yet
  if (locsysman_ != Teuchos::null) {
    dserror("LinearRelaxationSolve can't be used with locsys conditions cause it hasn't been implemented yet!");
  }

  TEUCHOS_FUNC_TIME_MONITOR("FluidImplicitTimeInt::LinearRelaxationSolve");

  //
  // Special linear solve used for steepest descent relaxation as well as
  // Jacobian-free Newton-Krylov on the FSI interface equations. The later one
  // presents a special challenge, as we have to solve the same linear system
  // repeatedly for different rhs. That is why we need the inrelaxation_ flag.
  //
  // Additionally we might want to include the mesh derivatives to get optimal
  // convergence in the Newton loop.
  //
  // This adds even more state to the fluid algorithm class, which is a bad
  // thing. And the explicit storage of the Dirichlet lines is
  // required. However, we do not need any special element code to perform the
  // steepest descent calculation. This is quite a benefit as the special code
  // in the old discretization was a real nightmare.
  //

  if (not inrelaxation_)
  {
    // setup relaxation matrices just once
    //
    // We use these matrices for several solves in Jacobian-free Newton-Krylov
    // solves of the FSI interface equations.

    const Epetra_Map* dofrowmap = discret_->DofRowMap();
    Teuchos::RCP<Epetra_Vector> griddisp = LINALG::CreateVector(*dofrowmap,false);

    // set the grid displacement independent of the trial value at the
    // interface
    griddisp->Update(1., *dispnp_, -1., *dispn_, 0.);

    // dbcmaps_ has already been set up

    // zero out the stiffness matrix
    sysmat_->Zero();

    // zero out residual, no neumann bc
    residual_->PutScalar(0.0);

    // Get matrix for mesh derivatives. This is not meant to be efficient.
    if (params_->get<bool>("shape derivatives"))
    {
      if (meshmatrix_==Teuchos::null)
      {
        meshmatrix_ = Teuchos::rcp(new LINALG::SparseMatrix(*SystemMatrix()));
      }
      else
      {
        meshmatrix_->Zero();
      }
    }

    // general fluid and time parameter are set in PrepareTimeStep()
    Teuchos::ParameterList eleparams;

    // parameters for stabilization
    eleparams.sublist("TURBULENCE MODEL") = params_->sublist("TURBULENCE MODEL");

    // set thermodynamic pressures
    SetCustomEleParamsLinearRelaxationSolve(eleparams);

    // set general vector values needed by elements
    discret_->ClearState();
    discret_->SetState("hist" ,hist_ );
    discret_->SetState("veln" ,veln_ );
    discret_->SetState("accam",accam_);
    discret_->SetState("scaaf",scaaf_);
    discret_->SetState("scaam",scaam_);
    discret_->SetState("dispnp", griddisp);
    discret_->SetState("gridv", zeros_);

    eleparams.set<int>("action",FLD::calc_fluid_systemmat_and_residual);
    eleparams.set<int>("physical type",physicaltype_);
    // set scheme-specific element parameters and vector values
    SetStateTimInt();

    // call loop over elements
    discret_->Evaluate(eleparams,sysmat_,meshmatrix_,residual_,Teuchos::null,Teuchos::null);
    discret_->ClearState();

    // finalize the system matrix
    sysmat_->Complete();

    if (meshmatrix_!=Teuchos::null)
    {
      meshmatrix_->Complete();
    }

    //--------- Apply dirichlet boundary conditions to system of equations
    //          residual displacements are supposed to be zero at
    //          boundary conditions
    dirichletlines_ = Teuchos::null;
    dirichletlines_ = SystemMatrix()->ExtractDirichletRows(*(dbcmaps_->CondMap()));
    sysmat_->ApplyDirichlet(*(dbcmaps_->CondMap()));
  }

  // No, we do not want to have any rhs. There cannot be any.
  residual_->PutScalar(0.0);

  if (meshmatrix_!=Teuchos::null)
  {
    // Calculate rhs due to mesh movement induced by the interface
    // displacement.
    meshmatrix_->Apply(*gridv_,*residual_);
    residual_->Scale(-dta_);
  }

  //--------- Apply dirichlet boundary conditions to system of equations
  //          residual displacements are supposed to be zero at
  //          boundary conditions
  incvel_->PutScalar(0.0);

  LINALG::ApplyDirichlettoSystem(incvel_,residual_,relax,*(dbcmaps_->CondMap()));

  CustomSolve(relax);
  //-------solve for residual displacements to correct incremental displacements
  solver_->Solve(sysmat_->EpetraOperator(),incvel_,residual_,not inrelaxation_,not inrelaxation_);

  // and now we need the reaction forces

  if (dirichletlines_->Apply(*incvel_, *trueresidual_)!=0)
    dserror("dirichletlines_->Apply() failed");

  if (meshmatrix_!=Teuchos::null)
  {
    // Calculate rhs due to mesh movement induced by the interface
    // displacement.
    meshmatrix_->Apply(*gridv_,*residual_);
    trueresidual_->Update(dta_,*residual_,1.0);
  }

  trueresidual_->Scale(-ResidualScaling());

  if (not inrelaxation_)
    inrelaxation_ = true;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::AddDirichCond(const Teuchos::RCP<const Epetra_Map> maptoadd)
{
  std::vector<Teuchos::RCP<const Epetra_Map> > condmaps;
  condmaps.push_back(maptoadd);
  condmaps.push_back(dbcmaps_->CondMap());
  Teuchos::RCP<Epetra_Map> condmerged = LINALG::MultiMapExtractor::MergeMaps(condmaps);
  *dbcmaps_ = LINALG::MapExtractor(*(discret_->DofRowMap()), condmerged);
  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::RemoveDirichCond(const Teuchos::RCP<const Epetra_Map> maptoremove)
{
  std::vector<Teuchos::RCP<const Epetra_Map> > othermaps;
  othermaps.push_back(maptoremove);
  othermaps.push_back(dbcmaps_->OtherMap());
  Teuchos::RCP<Epetra_Map> othermerged = LINALG::MultiMapExtractor::MergeMaps(othermaps);
  *dbcmaps_ = LINALG::MapExtractor(*(discret_->DofRowMap()), othermerged, false);
  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const Teuchos::RCP<const Epetra_Vector> FLD::FluidImplicitTimeInt::Dirichlet()
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
 *----------------------------------------------------------------------*/
const Teuchos::RCP<const Epetra_Vector> FLD::FluidImplicitTimeInt::InvDirichlet()
{
  if (dbcmaps_ == Teuchos::null)
    dserror("Dirichlet map has not been allocated");
  Teuchos::RCP<Epetra_Vector> dirichzeros = LINALG::CreateVector(*(dbcmaps_->CondMap()),true);
  Teuchos::RCP<Epetra_Vector> invtoggle = LINALG::CreateVector(*(discret_->DofRowMap()),false);
  invtoggle->PutScalar(1.0);
  dbcmaps_->InsertCondVector(dirichzeros, invtoggle);
  return invtoggle;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Map> FLD::FluidImplicitTimeInt::VelocityRowMap()
{ return velpressplitter_.OtherMap(); }

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Map> FLD::FluidImplicitTimeInt::PressureRowMap()
{ return velpressplitter_.CondMap(); }


/*----------------------------------------------------------------------*
 |  calculate wall sheer stress at (Dirichlet) boundary (public)        |
 |                                                          ismail 08/10|
 *----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector> FLD::FluidImplicitTimeInt::CalcWallShearStresses()
{
  // -------------------------------------------------------------------
  // first evaluate the normals at the nodes
  // -------------------------------------------------------------------

  Teuchos::ParameterList eleparams;
  // set action for elements
  eleparams.set<int>("action",FLD::ba_calc_node_normal);

  // get a vector layout from the discretization to construct matching
  // vectors and matrices
  //                 local <-> global dof numbering
  const Epetra_Map* dofrowmap = discret_->DofRowMap();

  //vector ndnorm0 with pressure-entries is needed for EvaluateCondition
  Teuchos::RCP<Epetra_Vector> ndnorm0 = LINALG::CreateVector(*dofrowmap,true);

  //call loop over elements, note: normal vectors do not yet have length = 1.0
  discret_->ClearState();
  if (alefluid_)
  {
    discret_->SetState("dispnp", dispnp_);
  }
  // evaluate the normals of the surface
  discret_->EvaluateCondition(eleparams,ndnorm0,"FluidStressCalc");
  discret_->ClearState();

  // -------------------------------------------------------------------
  // normalize the normal vectors
  // -------------------------------------------------------------------
  for (int i = 0; i < ndnorm0->MyLength();i+=numdim_+1)
  {
    // calculate the length of the normal
    double L = 0.0;
    for (int j = 0; j<numdim_; j++)
    {
      L += ((*ndnorm0)[i+j])*((*ndnorm0)[i+j]);
    }
    L = sqrt(L);

    // normalise the normal vector (if present for the current node)
    if (L > EPS15)
    {
      for (int j = 0; j < numdim_; j++)
      {
        (*ndnorm0)[i+j] /=  L;
      }
    }
  }

  // -------------------------------------------------------------------
  // evaluate the wall shear stress from the traction by removing
  // the normal stresses
  // -------------------------------------------------------------------

  // get traction
  Teuchos::RCP<Epetra_Vector> wss = CalcStresses();

  // loop over all entities within the traction vector
  for (int i = 0; i < ndnorm0->MyLength();i+=numdim_+1)
  {
    // evaluate the normal stress = < traction . normal >
    double normal_stress = 0.0;
    for (int j = 0; j<numdim_; j++)
    {
      normal_stress += (*wss)[i+j] * (*ndnorm0)[i+j];
    }

    // subtract the normal stresses from traction
    for (int j = 0; j<numdim_; j++)
    {
      (*wss)[i+j] -= normal_stress * (*ndnorm0)[i+j];
    }
  }


  // -------------------------------------------------------------------
  // return the wall_shear_stress vector
  // -------------------------------------------------------------------
  return wss;
}


Teuchos::RCP<Epetra_Vector> FLD::FluidImplicitTimeInt::CalcSFS(
   const int   i,
   const int   j
)
{
  // compute filtered quantities with the latest velocity field
  this->ApplyScaleSeparationForLES();

  const Epetra_Map* noderowmap = discret_->NodeRowMap();
  // create vector (+ initialization with zeros)
  Teuchos::RCP<Epetra_Vector> tauSFS = LINALG::CreateVector(*noderowmap,true);

  // get filtered velocity
  Teuchos::RCP<Epetra_Vector> filteredvel = LINALG::CreateVector(*noderowmap,true);
  Boxf_->FilteredVelComp(filteredvel, i, j);

  // get filtered reynoldsstress
  Teuchos::RCP<Epetra_Vector> filteredreystr = LINALG::CreateVector(*noderowmap,true);
  Boxf_->FilteredReyStrComp(filteredreystr, i , j);

  tauSFS->Update(1.0,*filteredreystr,-1.0, *filteredvel,0.0);

  return tauSFS;
}

// -------------------------------------------------------------------
// set general fluid parameter (AE 01/2011)
// -------------------------------------------------------------------
void FLD::FluidImplicitTimeInt::SetElementGeneralFluidParameter()
{
  Teuchos::ParameterList eleparams;

  eleparams.set<int>("action",FLD::set_general_fluid_parameter);

  // set general element parameters
  eleparams.set("form of convective term",convform_);
  eleparams.set<int>("Linearisation",newton_);
  eleparams.set<int>("Physical Type", physicaltype_);

  // parameter for stabilization
  eleparams.sublist("RESIDUAL-BASED STABILIZATION") = params_->sublist("RESIDUAL-BASED STABILIZATION");
  eleparams.sublist("EDGE-BASED STABILIZATION") = params_->sublist("EDGE-BASED STABILIZATION");

  // get function number of given Oseen advective field if necessary
  if (physicaltype_==INPAR::FLUID::oseen)
    eleparams.set<int>("OSEENFIELDFUNCNO", params_->get<int>("OSEENFIELDFUNCNO"));

  // call standard loop over elements
  discret_->Evaluate(eleparams,Teuchos::null,Teuchos::null,Teuchos::null,Teuchos::null,Teuchos::null);
  return;
}

// -------------------------------------------------------------------
// set turbulence parameters                         rasthofer 11/2011
// -------------------------------------------------------------------
void FLD::FluidImplicitTimeInt::SetElementTurbulenceParameter()
{
  Teuchos::ParameterList eleparams;

  eleparams.set<int>("action",FLD::set_turbulence_parameter);

  // set general parameters for turbulent flow
  eleparams.sublist("TURBULENCE MODEL") = params_->sublist("TURBULENCE MODEL");

  // set model-dependent parameters
  eleparams.sublist("SUBGRID VISCOSITY") = params_->sublist("SUBGRID VISCOSITY");
  eleparams.sublist("MULTIFRACTAL SUBGRID SCALES") = params_->sublist("MULTIFRACTAL SUBGRID SCALES");

  // call standard loop over elements
  discret_->Evaluate(eleparams,Teuchos::null,Teuchos::null,Teuchos::null,Teuchos::null,Teuchos::null);
  return;
}

// -------------------------------------------------------------------
// set general face fluid parameter for face/edge-oriented fluid stabilizations (BS 06/2014)
// -------------------------------------------------------------------
void FLD::FluidImplicitTimeInt::SetFaceGeneralFluidParameter()
{

  Teuchos::ParameterList faceparams;

  faceparams.set<int>("action",FLD::set_general_face_fluid_parameter);

  // set general fluid face parameters are contained in the following two sublists
  faceparams.sublist("EDGE-BASED STABILIZATION")     = params_->sublist("EDGE-BASED STABILIZATION");

  faceparams.set<int>("STABTYPE", DRT::INPUT::IntegralValue<INPAR::FLUID::StabType>( params_->sublist("RESIDUAL-BASED STABILIZATION"), "STABTYPE"));

  DRT::ELEMENTS::FluidIntFaceType::Instance().PreEvaluate(*discret_,faceparams,Teuchos::null,Teuchos::null,Teuchos::null,Teuchos::null,Teuchos::null);

}

/*----------------------------------------------------------------------*
 | update Newton step                                                   |
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::UpdateNewton(Teuchos::RCP<const Epetra_Vector> vel)
{
  UpdateIterIncrementally(vel);
}


// -------------------------------------------------------------------
// provide access to turbulence statistics manager (gjb 06/2011)
// -------------------------------------------------------------------
Teuchos::RCP<FLD::TurbulenceStatisticManager> FLD::FluidImplicitTimeInt::TurbulenceStatisticManager()
{
  return statisticsmanager_;
}


// -------------------------------------------------------------------
// provide access to box filter for dynamic Smagorinsk model     rasthofer/krank
// -------------------------------------------------------------------
Teuchos::RCP<FLD::DynSmagFilter> FLD::FluidImplicitTimeInt::DynSmagFilter()
{
  return DynSmag_;
}

// -------------------------------------------------------------------
// provide access to box filter for dynamic Vreman model         rasthofer/krank
// -------------------------------------------------------------------
Teuchos::RCP<FLD::Vreman> FLD::FluidImplicitTimeInt::Vreman()
{
  return Vrem_;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
//Overloaded in TimIntPoro and TimIntRedModels bk 12/13
void FLD::FluidImplicitTimeInt::UpdateIterIncrementally(
  Teuchos::RCP<const Epetra_Vector> vel
  )
{
  // set the new solution we just got
  if (vel != Teuchos::null)
  {
    // Take Dirichlet values from velnp and add vel to veln for non-Dirichlet
    // values.
    Teuchos::RCP<Epetra_Vector> aux = LINALG::CreateVector(
        *(discret_->DofRowMap(0)), true);
    aux->Update(1.0, *velnp_, 1.0, *vel, 0.0);
    //    dbcmaps_->InsertOtherVector(dbcmaps_->ExtractOtherVector(aux), velnp_);
    dbcmaps_->InsertCondVector(dbcmaps_->ExtractCondVector(velnp_), aux);

    *velnp_ = *aux;
  }

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::PrintStabilizationDetails() const
{
  // output of stabilization details
  Teuchos::ParameterList* stabparams=&(params_->sublist("RESIDUAL-BASED STABILIZATION"));
  if (myrank_ == 0)
  {
    std::cout << "Stabilization type         : " << stabparams->get<std::string>("STABTYPE") << "\n";
    std::cout << "                             " << "Evaluation Tau  = " << stabparams->get<std::string>("EVALUATION_TAU") <<"\n";
    std::cout << "                             " << "Evaluation Mat  = " << stabparams->get<std::string>("EVALUATION_MAT") <<"\n";
    std::cout << "\n";

    if(DRT::INPUT::IntegralValue<INPAR::FLUID::StabType>(*stabparams, "STABTYPE") == INPAR::FLUID::stabtype_residualbased)
    {

      std::cout << "                             " << stabparams->get<std::string>("TDS")<< "\n";
      std::cout << "\n";
      std::cout << "                             " << "Tau Type        = " << stabparams->get<std::string>("DEFINITION_TAU") <<"\n";

      if(stabparams->get<std::string>("TDS") == "quasistatic")
      {
        if(stabparams->get<std::string>("TRANSIENT")=="yes_transient")
        {
          dserror("The quasistatic version of the residual-based stabilization currently does not support the incorporation of the transient term.");
        }
      }

      std::cout <<  "                             " << "SUPG            = " << stabparams->get<std::string>("SUPG")           <<"\n";
      std::cout <<  "                             " << "PSPG            = " << stabparams->get<std::string>("PSPG")           <<"\n";
      std::cout <<  "                             " << "GRAD_DIV        = " << stabparams->get<std::string>("GRAD_DIV")          <<"\n";
      std::cout <<  "                             " << "CROSS-STRESS    = " << stabparams->get<std::string>("CROSS-STRESS")   <<"\n";
      std::cout <<  "                             " << "REYNOLDS-STRESS = " << stabparams->get<std::string>("REYNOLDS-STRESS")<<"\n";
      std::cout <<  "                             " << "VSTAB           = " << stabparams->get<std::string>("VSTAB")          <<"\n";
      std::cout <<  "                             " << "RSTAB           = " << stabparams->get<std::string>("RSTAB")          <<"\n";
      std::cout <<  "                             " << "TRANSIENT       = " << stabparams->get<std::string>("TRANSIENT")      <<"\n";
      std::cout << "\n";
      std::cout << std::endl;
    }
    else if(DRT::INPUT::IntegralValue<INPAR::FLUID::StabType>(*stabparams, "STABTYPE") == INPAR::FLUID::stabtype_edgebased)
    {
      Teuchos::ParameterList *  stabparams_edgebased      =&(params_->sublist("EDGE-BASED STABILIZATION"));

      std::cout << "\n\nEDGE-BASED (EOS) fluid stabilizations " << "\n";

      std::cout << "+---------------------------------------------------------------------------------+\n";
      std::cout << "|  WARNING: edge-based stabilization requires face discretization                 |\n";
      std::cout << "|           face discretization currently only available for pure fluid problems  |\n";
      std::cout << "+---------------------------------------------------------------------------------+\n";

      std::cout <<  "                    " << "EOS_PRES             = " << stabparams_edgebased->get<std::string>("EOS_PRES")      <<"\n";
      std::cout <<  "                    " << "EOS_CONV_STREAM      = " << stabparams_edgebased->get<std::string>("EOS_CONV_STREAM")      <<"\n";
      std::cout <<  "                    " << "EOS_CONV_CROSS       = " << stabparams_edgebased->get<std::string>("EOS_CONV_CROSS")      <<"\n";
      std::cout <<  "                    " << "EOS_DIV              = " << stabparams_edgebased->get<std::string>("EOS_DIV")      <<"\n";
      std::cout <<  "                    " << "EOS_DEFINITION_TAU   = " << stabparams_edgebased->get<std::string>("EOS_DEFINITION_TAU")      <<"\n";
      std::cout <<  "                    " << "EOS_H_DEFINITION     = " << stabparams_edgebased->get<std::string>("EOS_H_DEFINITION")      <<"\n";
      std::cout << "+---------------------------------------------------------------------------------+\n" << std::endl;
    }
  }
}

// -------------------------------------------------------------------
// print informations about turbulence model         rasthofer 04/2011
// -------------------------------------------------------------------
void FLD::FluidImplicitTimeInt::PrintTurbulenceModel()
{
    // a canonical flow with homogeneous directions would allow a
    // spatial averaging of data
    std::string homdir = params_->sublist("TURBULENCE MODEL").get<std::string>("HOMDIR","not_specified");

    if (myrank_ == 0 and turbmodel_!=INPAR::FLUID::no_model)
    {
      std::cout << "Turbulence model        : ";
      std::cout << params_->sublist("TURBULENCE MODEL").get<std::string>("PHYSICAL_MODEL","no_model");
      std::cout << &std::endl;

      if (turbmodel_ == INPAR::FLUID::smagorinsky)
      {
        std::cout << "                             " ;
        std::cout << "with Smagorinsky constant Cs= ";
        std::cout << params_->sublist("SUBGRID VISCOSITY").get<double>("C_SMAGORINSKY") << "\n";
        std::cout << &std::endl;
      }
      else if(turbmodel_ == INPAR::FLUID::smagorinsky_with_van_Driest_damping)
      {
        if (special_flow_ != "channel_flow_of_height_2"
            ||
            homdir != "xz")
        {
          dserror("The van Driest damping is only implemented for a channel flow with wall \nnormal direction y");
        }

        std::cout << "                             "          ;
        std::cout << "\n";
        std::cout << "- Smagorinsky constant:   Cs   = "      ;
        std::cout << params_->sublist("SUBGRID VISCOSITY").get<double>("C_SMAGORINSKY");
        std::cout << &std::endl;
        std::cout << "- viscous length      :   l_tau= "      ;
        std::cout << params_->sublist("SUBGRID VISCOSITY").get<double>("CHANNEL_L_TAU") << "\n";
        std::cout << &std::endl;
      }
      else if(turbmodel_ == INPAR::FLUID::dynamic_smagorinsky)
      {
        if (homdir == "not_specified")
        {
          std::cout << "      no homogeneous directions specified --- so we just use pointwise clipping for Cs\n";
          std::cout << &std::endl;
        }
      }
        else if(turbmodel_ == INPAR::FLUID::scale_similarity or turbmodel_ == INPAR::FLUID::scale_similarity_basic)
      {
        std::cout << "                             "      ;
        std::cout << "\n";
        std::cout << "- Constant:  Cl   = "      ;
        std::cout << params_->sublist("MULTIFRACTAL SUBGRID SCALES").get<double>("C_SCALE_SIMILARITY") << "\n";
        std::cout << "- Scale separation:  " << params_->sublist("MULTIFRACTAL SUBGRID SCALES").get<std::string>("SCALE_SEPARATION") << "\n";
        std::cout << &std::endl;
      }
      else if(turbmodel_ == INPAR::FLUID::multifractal_subgrid_scales)
      {
        Teuchos::ParameterList *  modelparams =&(params_->sublist("MULTIFRACTAL SUBGRID SCALES"));
        std::cout << "                             "      ;
        std::cout << "\n";
        std::cout << "- Csgs:              " << modelparams->get<double>("CSGS") << "\n";
        std::cout << "- Scale separation:  " << modelparams->get<std::string>("SCALE_SEPARATION") << "\n";
        if ((DRT::INPUT::IntegralValue<int>(*modelparams,"CALC_N")))
        {
          std::cout << "- Re_length:         " << modelparams->get<std::string>("REF_LENGTH") << "\n";
          std::cout << "- Re_vel:            " << modelparams->get<std::string>("REF_VELOCITY") << "\n";
          std::cout << "- c_nu:              " << modelparams->get<double>("C_NU") << "\n";
        }
        else
          std::cout << "- N:                 " << modelparams->get<double>("N") << "\n";
        std::cout << "- near-wall limit:   " << DRT::INPUT::IntegralValue<int>(*modelparams,"NEAR_WALL_LIMIT") << "\n";
        std::cout << "- beta:              " << modelparams->get<double>("BETA") << "\n";
        std::cout << "- evaluation B:      " << modelparams->get<std::string>("EVALUATION_B") << "\n";
        std::cout << "- conservative:      " << modelparams->get<std::string>("CONVFORM") << "\n";
        if ((DRT::INPUT::IntegralValue<int>(*modelparams,"SET_FINE_SCALE_VEL")))
            std::cout << "WARNING: fine-scale velocity is set for nightly tests!" << "\n";
        std::cout << &std::endl;
      }
      else if(turbmodel_ == INPAR::FLUID::vreman)
      {
        std::cout << "                             "          ;
        std::cout << "\n";
        std::cout << "- Vreman model with constant coefficient\n"      ;
        std::cout << "- Use filter width method:  " << params_->sublist("SUBGRID VISCOSITY").get<std::string>("FILTER_WIDTH","CubeRootVol") << "\n";
        std::cout << &std::endl;
      }
      else if(turbmodel_ == INPAR::FLUID::dynamic_vreman)
      {
        std::cout << "                             "          ;
        std::cout << "\n";
        std::cout << "- Vreman model with dynamic calculation of coefficient\n"      ;
        std::cout << "- Use filter width method:  Only cube root volume implemented for dynamic coefficient" << "\n";
        std::cout << &std::endl;
      }
    }

  return;
}


/*----------------------------------------------------------------------*
 | filtered quantities for classical LES models          rasthofer 02/11|
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::ApplyScaleSeparationForLES()
{
  if (turbmodel_==INPAR::FLUID::dynamic_smagorinsky)
  {
    // perform filtering and computation of Cs
    // compute averaged values for LijMij and MijMij
    const Teuchos::RCP<const Epetra_Vector> dirichtoggle = Dirichlet();
    DynSmag_->ApplyFilterForDynamicComputationOfCs(EvaluationVel(),scaaf_,ReturnThermpressaf(),dirichtoggle);
  }
  else if (turbmodel_==INPAR::FLUID::scale_similarity or turbmodel_==INPAR::FLUID::scale_similarity_basic)
  {
    switch (scale_sep_)
    {
    case INPAR::FLUID::box_filter:
    {
      // perform filtering
      const Teuchos::RCP<const Epetra_Vector> dirichtoggle = Dirichlet();
      // call only filtering
      Boxf_->ApplyFilter(EvaluationVel(),scaaf_,ReturnThermpressaf(),dirichtoggle);

      // get filtered fields
      filteredvel_->PutScalar(0.0);
      filteredreystr_->PutScalar(0.0);

      Boxf_->GetFilteredVelocity(filteredvel_);
      Boxf_->GetFilteredReynoldsStress(filteredreystr_);
      Boxf_->GetFineScaleVelocity(finescalevel_);

      // store fine-scale velocity
      Boxf_->OutputofFineScaleVel(fsvelaf_);
      break;
    }
    case INPAR::FLUID::algebraic_multigrid_operator:
    {
      const Epetra_Map* dofrowmap = discret_->DofRowMap();
      const Epetra_Map* dofcolmap = discret_->DofColMap();

      Teuchos::RCP<Epetra_Vector> row_filteredveltmp;
      row_filteredveltmp = Teuchos::rcp(new Epetra_Vector(*dofrowmap,true));
      Teuchos::RCP<Epetra_Vector> col_filteredveltmp;
      col_filteredveltmp = Teuchos::rcp(new Epetra_Vector(*dofcolmap,true));

      Teuchos::RCP<Epetra_Vector> row_finescaleveltmp;
      row_finescaleveltmp = Teuchos::rcp(new Epetra_Vector(*dofrowmap,true));
      Teuchos::RCP<Epetra_Vector> col_finescaleveltmp;
      col_finescaleveltmp = Teuchos::rcp(new Epetra_Vector(*dofcolmap,true));

      Teuchos::RCP<Epetra_MultiVector> row_filteredreystretmp;
      row_filteredreystretmp = Teuchos::rcp(new Epetra_MultiVector(*dofrowmap,3,true));
      Teuchos::RCP<Epetra_MultiVector> col_filteredreystretmp;
      col_filteredreystretmp = Teuchos::rcp(new Epetra_MultiVector(*dofcolmap,3,true));
      Teuchos::RCP<Epetra_MultiVector> row_reystretmp;
      row_reystretmp = Teuchos::rcp(new Epetra_MultiVector(*dofrowmap,3,true));
      Teuchos::RCP<Epetra_MultiVector> row_finescalereystretmp;
      row_finescalereystretmp = Teuchos::rcp(new Epetra_MultiVector(*dofrowmap,3,true));

      /*-------------------------------------------------------------------
         * remark:
         * - first, the fine scale velocity is computed
         * - second, we get the coarse scale velocity by subtracting
         *   the fine scale velocity from the resolved velocity
         * - the same procedure is applied to the reynolds stresses
         * - One might think of directly computing the coarse scale quantities
         *   by applying the respective scale-separation operator. However, in
         *   doing so, coarse scale quantities are set to zero on dirichlet
         *   boundaries due to the way the scale-separation operators are
         *   constructed. Setting the fine scale part of the velocity equal zero
         *   is a reasonable choice. However, this strategy is not reasonable for
         *   coarse scale quantities. The coarse scale quantities should rather
         *   contain the exact values. This is ensured by the proposed approach.
      *-------------------------------------------------------------------*/
      // get fine scale velocity
      Sep_->Multiply(false,*EvaluationVel(),*row_finescaleveltmp);
      // get filtered or coarse scale velocity
      row_filteredveltmp->Update(1.0,*EvaluationVel(),-1.0,*row_finescaleveltmp,0.0);

      /*-------------------------------------------------------------------
         * idea:
         * - calculate reynolds stress tensor at each node
         * - filter the reynolds stress tensor by multiplication with
         *   large scale separation operator
      *-------------------------------------------------------------------*/
      // calculate reynoldsstress
      // loop all nodes on this proc
      for (int nid=0;nid<discret_->NumMyRowNodes();++nid)
      {
        // get the node
        DRT::Node* node = discret_->lRowNode(nid);

        // get the dofs of the node
        std::vector<int> dofs= discret_->Dof(node);
        //we only loop over all velocity dofs
        for(int di=0;di<discret_->NumDof(node)-1;++di)
        {
          // get global id of the dof
          int gidi = dofs[di];
          // get local id of the dof
          int lidi = discret_->DofRowMap()->LID(gidi);
          // get the velocity
          double veli=(*EvaluationVel())[lidi];
          for(int dj=0;dj<discret_->NumDof(node)-1;++dj)
          {
            // get the second velocity in the same way
            int gidj =dofs[dj];
            int lidj = discret_->DofRowMap()->LID(gidj);
            double velj=(*EvaluationVel())[lidj];
            // multiply the velocity to get the final component of the reynoldsstress tensor
            double velivelj = veli*velj;
            // store it
            ((*row_reystretmp)(dj))->ReplaceGlobalValues(1,&velivelj,&gidi);
          }
        }
      }

      //get the filtered reynoldsstress
      Sep_->Multiply(false,*row_reystretmp,*row_finescalereystretmp);
      row_filteredreystretmp->Update(1.0,*row_reystretmp,-1.0,*row_finescalereystretmp,0.0);

      // export quantities in dofrowmap format to dofcolumnmap format
      LINALG::Export(*row_filteredveltmp,*col_filteredveltmp);
      LINALG::Export(*row_finescaleveltmp,*col_finescaleveltmp);
      LINALG::Export(*row_filteredreystretmp,*col_filteredreystretmp);

      // transfer quantities from dofcolumnmap to nodecolmap
      // filtered velocity and fine scale subgrid velocity
      // loop all nodes on this proc (including ghosted ones)
      for (int nid=0;nid<discret_->NumMyColNodes();++nid)
      {
        // get the node
        DRT::Node* node = discret_->lColNode(nid);
        // get global ids of all dofs of the node
        std::vector<int> dofs= discret_->Dof(node);

        //we only loop over all velocity dofs
        for(int di=0;di<discret_->NumDof(node)-1;++di)
        {
          // get global id of the dof
          int gidi = dofs[di];
          // get local dof id corresponding to the global id
          int lidi = discret_->DofColMap()->LID(gidi);
          // get the values of the dof
          double valvel = (*col_filteredveltmp)[lidi];
          double valfsvel = (*col_finescaleveltmp)[lidi];
          // store them
          int err = 0;
          err += ((*filteredvel_)(di))->ReplaceMyValues(1,&valvel,&nid);
          err += ((*finescalevel_)(di))->ReplaceMyValues(1,&valfsvel,&nid);
          if (err!=0) dserror("dof not on proc");
        }
      }

          // transfer filtered reynoldsstress
          // loop all nodes on this proc (including ghosted ones)
          for (int nid=0;nid<discret_->NumMyColNodes();++nid)
          {
            // get the node
            DRT::Node* node = discret_->lColNode(nid);
            // get global ids of all dofs of the node
            std::vector<int> dofs= discret_->Dof(node);

            //we only loop over all velocity dofs
            for(int di=0;di<discret_->NumDof(node)-1;++di)
            {
              // get global id of the dof
              int gidi = dofs[di];
              // get local dof id corresponding to the global id
              int lidi = discret_->DofColMap()->LID(gidi);
              //loop over all components
              for(int dj=0;dj<discret_->NumDof(node)-1;++dj)
              {
                // get the values
                double val = (*((*col_filteredreystretmp)(dj)))[lidi];
                // and store it
                const int ij = di*3+dj;
                int err = ((*filteredreystr_)(ij))->ReplaceMyValues(1,&val,&nid);
                if (err!=0) dserror("dof not on proc");
              }
            }
          }

      // store fine-scale velocity
      fsvelaf_->Update(1.0,*row_finescaleveltmp,0.0);

      break;
    }
    case INPAR::FLUID::geometric_multigrid_operator:
    {
      dserror("Not available for scale-similarity type models!");
      break;
    }
    default:
    {
      dserror("Unknown filter type!");
      break;
    }
    }
  }
  else if (turbmodel_==INPAR::FLUID::multifractal_subgrid_scales)
  {
    switch (scale_sep_)
    {
    case INPAR::FLUID::box_filter:
    {
      // perform filtering
      const Teuchos::RCP<const Epetra_Vector> dirichtoggle = Dirichlet();
      // call only filtering
      Boxf_->ApplyFilter(EvaluationVel(),scaaf_,ReturnThermpressaf(),dirichtoggle);

      // get fine-scale velocity
      Boxf_->OutputofFineScaleVel(fsvelaf_);

      break;
    }
    case INPAR::FLUID::algebraic_multigrid_operator:
    {
      // get fine-scale part of velocity at time n+alpha_F or n+1
      Sep_->Multiply(false,*EvaluationVel(),*fsvelaf_);

      // set fine-scale velocity for parallel nigthly tests
      // separation matrix depends on the number of proc here
      if (turbmodel_==INPAR::FLUID::multifractal_subgrid_scales and
          (DRT::INPUT::IntegralValue<int>(params_->sublist("MULTIFRACTAL SUBGRID SCALES"),"SET_FINE_SCALE_VEL")))
        fsvelaf_->PutScalar(0.01);

      break;
    }
    case INPAR::FLUID::geometric_multigrid_operator:
    {
      ScaleSepGMO_->ApplyScaleSeparation(EvaluationVel(),fsvelaf_);

      break;
    }
    default:
    {
      dserror("Unknown filter type!");
      break;
    }
    }

    // set fine-scale vector
    discret_->SetState("fsvelaf",fsvelaf_);
  }
  else if (turbmodel_==INPAR::FLUID::dynamic_vreman)
  {
    // perform filtering
    const Teuchos::RCP<const Epetra_Vector> dirichtoggle = Dirichlet();

    Vrem_->ApplyFilterForDynamicComputationOfCv(EvaluationVel(),scaaf_,ReturnThermpressaf(),dirichtoggle);

  }
  else
    dserror("Unknown turbulence model!");

  return;
}


//-------------------------------------------------------------------------
// calculate mean CsgsB to estimate CsgsD
// for multifractal subgrid-scale model                    rasthofer 08/12
//-------------------------------------------------------------------------
void FLD::FluidImplicitTimeInt::RecomputeMeanCsgsB()
{
  // For loma, this function is required at the respective position to set up CsgsD of the scalar field for including the subgrid-scale
  // temperature in the physical properties and the subgrid-scale terms arising in the continuity equation. This recomputation
  // avoids transferring the respective value from the scalar to the fluid field.
  // The so computed value is also used for calculating statistical data for MFS, although this is not the final value for gen-alpha
  // that is seen by the scalar field. However, note that vel_n ~ vel_np ~ vel_af for statistically stationary flow. Hence, the expected
  // error is marginal, but another computation is avoided.

  if (DRT::INPUT::IntegralValue<int>(params_->sublist("MULTIFRACTAL SUBGRID SCALES"),"ADAPT_CSGS_PHI"))
  {
    // mean Cai
    double meanCai = 0.0;

    // variables required for calculation
    // local sums
    double local_sumCai = 0.0;
    double local_sumVol = 0.0;
    // global sums
    double global_sumCai = 0.0;
    double global_sumVol = 0.0;

    // define element matrices and vectors --- dummies
    Epetra_SerialDenseMatrix emat1;
    Epetra_SerialDenseMatrix emat2;
    Epetra_SerialDenseVector evec1;
    Epetra_SerialDenseVector evec2;
    Epetra_SerialDenseVector evec3;

    // generate a parameterlist for communication and control
    Teuchos::ParameterList myparams;
    // action for elements
    myparams.set<int>("action",FLD::calc_mean_Cai);
    myparams.set<int>("physical type",physicaltype_);

    // set state vector to pass distributed vector to the element
    // set velocity
    discret_->ClearState();
    SetStateTimInt();
    // set temperature
    discret_->SetState("scalar",scaaf_);
    // set thermodynamic pressures
    SetCustomEleParamsApplyNonlinearBoundaryConditions(myparams);

    // loop all elements on this proc (excluding ghosted ones)
    for (int nele=0;nele<discret_->NumMyRowElements();++nele)
    {
      // get the element
      DRT::Element* ele = discret_->lRowElement(nele);

      // get element location vector, dirichlet flags and ownerships
      std::vector<int> lm;
      std::vector<int> lmowner;
      std::vector<int> lmstride;
      ele->LocationVector(*discret_,lm,lmowner,lmstride);

      // call the element evaluate method to integrate functions
      int err = ele->Evaluate(myparams,*discret_,lm,
                            emat1,emat2,
                            evec1,evec2,evec2);
      if (err) dserror("Proc %d: Element %d returned err=%d",myrank_,ele->Id(),err);

      // get contributions of this element and add it up
      local_sumCai += myparams.get<double>("Cai_int");
      local_sumVol += myparams.get<double>("ele_vol");
    }
    discret_->ClearState();

    // gather contributions of all procs
    discret_->Comm().SumAll(&local_sumCai,&global_sumCai,1);
    discret_->Comm().SumAll(&local_sumVol,&global_sumVol,1);

    // calculate mean Cai
    meanCai = global_sumCai/global_sumVol;

    //std::cout << "Proc:  " << myrank_ << "  local vol and Cai   "
    //<< local_sumVol << "   " << local_sumCai << "  global vol and Cai   "
    //<< global_sumVol << "   " << global_sumCai << "  mean   " << meanCai << std::endl;

    if (myrank_ == 0)
    {
      std::cout << "\n+--------------------------------------------------------------------------------------------+" << std::endl;
      std::cout << "Multifractal subgrid scales: adaption of CsgsD from near-wall limit of CsgsB:  " << std::setprecision (8) << meanCai << std::endl;
      std::cout << "+--------------------------------------------------------------------------------------------+\n" << std::endl;
    }

    // store value in element parameter list
    myparams.set<int>("action",FLD::set_mean_Cai);
    myparams.set<double>("meanCai",meanCai);
    for (int nele=0;nele<discret_->NumMyRowElements();++nele)
    {
      // get the element
      DRT::Element* ele = discret_->lRowElement(nele);

      // get element location vector, dirichlet flags and ownerships
      std::vector<int> lm;
      std::vector<int> lmowner;
      std::vector<int> lmstride;
      ele->LocationVector(*discret_,lm,lmowner,lmstride);

      // call the element evaluate method to integrate functions
      int err = ele->Evaluate(myparams,*discret_,lm,
                            emat1,emat2,
                            evec1,evec2,evec2);
      if (err) dserror("Proc %d: Element %d returned err=%d",myrank_,ele->Id(),err);
    }
  }

  return;
}

// -------------------------------------------------------------------
// extrapolate from time mid-point to end-point         (mayr 12/2011)
// overloaded in TimIntGenAlpha                            bk 12/13
// -------------------------------------------------------------------
Teuchos::RCP<Epetra_Vector> FLD::FluidImplicitTimeInt::ExtrapolateEndPoint
(
  Teuchos::RCP<Epetra_Vector> vecn,
  Teuchos::RCP<Epetra_Vector> vecm
)
{
  Teuchos::RCP<Epetra_Vector> vecnp = Teuchos::rcp(new Epetra_Vector(*vecm));

  return vecnp;
}


// -------------------------------------------------------------------
// apply external forces to the fluid                    ghamm 03/2013
// -------------------------------------------------------------------
void FLD::FluidImplicitTimeInt::ApplyExternalForces
(
  Teuchos::RCP<Epetra_MultiVector> fext
)
{
  if(external_loads_ == Teuchos::null)
    external_loads_ = LINALG::CreateVector(*discret_->DofRowMap(),true);

  external_loads_->Update(1.0, *fext, 0.0);
  return;
}


/*------------------------------------------------------------------------------------------------*
 | create field test
 *------------------------------------------------------------------------------------------------*/
Teuchos::RCP<DRT::ResultTest> FLD::FluidImplicitTimeInt::CreateFieldTest()
{
  return Teuchos::rcp(new FLD::FluidResultTest(*this));
}


/*------------------------------------------------------------------------------------------------*
 |
 *------------------------------------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Vector> FLD::FluidImplicitTimeInt::ConvectiveVel()
{
  if (GridVel() == Teuchos::null)
    return Velnp(); // no moving mesh present
  else
  {
    // make an intermediate copy of velnp
    Teuchos::RCP<Epetra_Vector> convel = Teuchos::rcp(new Epetra_Vector(*(Velnp())));
    // now subtract the grid velocity
    convel->Update(-1.0,*(GridVel()),1.0);

    return convel;
  }
}


/*------------------------------------------------------------------------------------------------*
 | Calculate an integrated divergence operator                                    (mayr.mt 04/12) |
 *------------------------------------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector> FLD::FluidImplicitTimeInt::CalcDivOp()
{
  // set action in order to calculate the integrated divergence operator
  Teuchos::ParameterList params;
  params.set<int>("action",FLD::calc_divop);

  // integrated divergence operator B in vector form
  Teuchos::RCP<Epetra_Vector> divop = Teuchos::rcp(new Epetra_Vector(velnp_->Map(),true));

  // copy row map of mesh displacement to column map (only if ALE is used)
  discret_->ClearState();
  if (alefluid_)
    discret_->SetState("dispnp", dispnp_);

  // construct the operator on element level as a column vector
  discret_->Evaluate(params, Teuchos::null, Teuchos::null, divop, Teuchos::null,Teuchos::null);

  // clear column maps after the evaluate call
  discret_->ClearState();

//  // blank DOFs which are on Dirichlet BC, since they may not be modified
//  dbcmaps_->InsertCondVector(dbcmaps_->ExtractCondVector(zeros_), divop);

  return divop;
}


/*------------------------------------------------------------------------------------------------*
 |
 *------------------------------------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::Reset(
    bool completeReset,
    bool newFiles,
    int iter)
{
  const Epetra_Map* dofrowmap = discret_->DofRowMap();

  // Vectors passed to the element
  // -----------------------------
  // velocity/pressure at time n+1, n and n-1
  velnp_ = LINALG::CreateVector(*dofrowmap,true);
  veln_  = LINALG::CreateVector(*dofrowmap,true);
  velnm_ = LINALG::CreateVector(*dofrowmap,true);

  // acceleration/(scalar time derivative) at time n+1 and n
  accnp_ = LINALG::CreateVector(*dofrowmap,true);
  accn_  = LINALG::CreateVector(*dofrowmap,true);
  accnm_ = LINALG::CreateVector(*dofrowmap,true);

  // velocity/pressure at time n+alpha_F
  velaf_ = LINALG::CreateVector(*dofrowmap,true);

  // acceleration/(scalar time derivative) at time n+alpha_M/(n+alpha_M/n)
  accam_ = LINALG::CreateVector(*dofrowmap,true);

  // scalar at time n+alpha_F/n+1 and n+alpha_M/n
  // (only required for low-Mach-number case)
  scaaf_ = LINALG::CreateVector(*dofrowmap,true);
  scaam_ = LINALG::CreateVector(*dofrowmap,true);

  // velocity/pressure at nodes for meshfree non-interpolatory basis functions
  Teuchos::RCP<DRT::MESHFREE::MeshfreeDiscretization> meshfreediscret
    = Teuchos::rcp_dynamic_cast<DRT::MESHFREE::MeshfreeDiscretization>(discret_);
  if (meshfreediscret!=Teuchos::null)
    velatmeshfreenodes_ = LINALG::CreateVector(*dofrowmap,true);

  // history vector
  hist_ = LINALG::CreateVector(*dofrowmap,true);

  if (alefluid_)
  {
    dispnp_ = LINALG::CreateVector(*dofrowmap,true);
    dispn_  = LINALG::CreateVector(*dofrowmap,true);
    dispnm_ = LINALG::CreateVector(*dofrowmap,true);
    gridv_  = LINALG::CreateVector(*dofrowmap,true);

  }

  if (completeReset)
  {
    time_ = 0.0;
    step_ = 0;

    if (newFiles)
    {
      if (iter<0) dserror("iteration number <0");
      output_->NewResultFile((iter));
    }
    else
      output_->OverwriteResultFile();

    output_->WriteMesh(0,0.0);
  }

  return;
}


/*------------------------------------------------------------------------------------------------*
 |
 *------------------------------------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::PredictTangVelConsistAcc()
{
  // message to screen
  if(discret_->Comm().MyPID()==0)
  {
    std::cout << "fluid: doing TangVel predictor" << std::endl;
  }

  // total time required for evaluation of Dirichlet conditions
  Teuchos::ParameterList eleparams;
  eleparams.set("total time",time_);

  // initialize
  velnp_->Update(1.0, *veln_, 0.0);
  accnp_->Update(1.0, *accn_, 0.0);
  incvel_->PutScalar(0.0);

  // for solution increments on Dirichlet boundary
  Teuchos::RCP<Epetra_Vector> dbcinc
    = LINALG::CreateVector(*(discret_->DofRowMap()), true);

  // copy last converged solution
  dbcinc->Update(1.0, *veln_, 0.0);

  // get Dirichlet values at t_{n+1}
  // set vector values needed by elements
  discret_->ClearState();
  discret_->SetState("velnp",velnp_);

  // predicted Dirichlet values
  // velnp_ then also holds prescribed new dirichlet values
  discret_->EvaluateDirichlet(eleparams,velnp_,Teuchos::null,Teuchos::null,Teuchos::null);

  // subtract the displacements of the last converged step
  // DBC-DOFs hold increments of current step
  // free-DOFs hold zeros
  dbcinc->Update(-1.0, *veln_, 1.0);

  // compute residual forces residual_ and stiffness sysmat_
  // at velnp_, etc which are unchanged
  Evaluate(Teuchos::null);

  // add linear reaction forces to residual
  // linear reactions
  Teuchos::RCP<Epetra_Vector> freact
    = LINALG::CreateVector(*(discret_->DofRowMap()), true);
  sysmat_->Multiply(false, *dbcinc, *freact);

  // add linear reaction forces due to prescribed Dirichlet BCs
  residual_->Update(1.0, *freact, 1.0);

  // extract reaction forces
  freact->Update(1.0, *residual_, 0.0);
  dbcmaps_->InsertOtherVector(dbcmaps_->ExtractOtherVector(zeros_), freact);

  // blank residual at DOFs on Dirichlet BC
  dbcmaps_->InsertCondVector(dbcmaps_->ExtractCondVector(zeros_), residual_);

  // apply Dirichlet BCs to system of equations
  incvel_->PutScalar(0.0);
  sysmat_->Complete();
  LINALG::ApplyDirichlettoSystem(sysmat_, incvel_, residual_,
                                 Teuchos::null, zeros_, *(dbcmaps_->CondMap()));

  // solve for incvel_
  solver_->Solve(sysmat_->EpetraOperator(), incvel_, residual_, true, true);

  // set Dirichlet increments in solution increments
  incvel_->Update(1.0, *dbcinc, 1.0);

  // update end-point velocities and pressure
  UpdateIterIncrementally(incvel_);

  // keep pressure values from previous time step
  velpressplitter_.InsertCondVector(velpressplitter_.ExtractCondVector(veln_),velnp_);

  // Note: accelerations on Dirichlet DOFs are not set.

  // reset to zero
  incvel_->PutScalar(0.0);

  return;
}

/*----------------------------------------------------------------------*/
/* set fluid displacement vector due to biofilm growth          */
void FLD::FluidImplicitTimeInt::SetFldGrDisp(Teuchos::RCP<Epetra_Vector> fluid_growth_disp)
{
  fldgrdisp_= fluid_growth_disp;  return;
}

//overloaded in TimIntRedModels bk 12/13
void FLD::FluidImplicitTimeInt::SetupMeshtying()
{
  msht_ = DRT::INPUT::get<INPAR::FLUID::MeshTying>(*params_,"MESHTYING");
  bool alldofcoupled = params_->get<bool>("ALLDOFCOUPLED");

  // meshtying: all dofs (velocity + pressure) are coupled
  //            -> vector of length velocity dofs (numdim_) + pressure dof initialiued with ones
  // coupleddof [1, 1, 1, 1]
  std::vector<int> coupleddof(numdim_+1, 1);

  if (alldofcoupled == false)
  {
    // meshtying: only velocity dofs are coupled
    // meshtying: all dofs (velocity + pressure) are coupled
    //            -> vector of length velocity dofs (numdim_) + pressure dof initialiued with ones
    //            -> last entry (pressure) is set to zero -> pressure is not included into the coupling algorithm
    // coupleddof [1, 1, 1, 0]
    coupleddof[numdim_]=0;
  }

  meshtying_ = Teuchos::rcp(new Meshtying(discret_, *solver_, msht_, numdim_, surfacesplitter_));
  sysmat_ = meshtying_->Setup(coupleddof);

  // Check if there are DC defined on the master side of the internal interface
  meshtying_->DirichletOnMaster(dbcmaps_->CondMap());

  if(predictor_!= "steady_state")
  {
    if(myrank_==0)
      dserror("The meshtying framework does only support a steady-state predictor");
  }

  //meshtying_->OutputSetUp();

  return;
}

/*------------------------------------------------------------------------------------------------*
 | set void volume for cavitation problems                                          (ghamm 08/13) |
 *------------------------------------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::SetVoidVolume(Teuchos::RCP<Epetra_MultiVector> voidvolume)
{
  if(density_scaling_ == Teuchos::null)
    density_scaling_ = LINALG::CreateVector(*discret_->ElementRowMap(),false);
  density_scaling_->Update(1.0, *voidvolume, 0.0);
}

/*------------------------------------------------------------------------------------------------*
 | cavitation: output void fraction, overloaded (switched off) in TimIntTopOpt           bk 12/13 |
 *------------------------------------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::WriteOutputCavitationVoidFraction()
{
  if (density_scaling_ != Teuchos::null)
  {
    output_->WriteVector("voidfraction", density_scaling_, IO::DiscretizationWriter::elementvector);
  }
  return;
}

/*----------------------------------------------------------------------------*
 | Compute kinetic energy and write it to file                mayr.mt 05/2014 |
 *----------------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::WriteOutputKineticEnergy()
{
  // take care of possibly changed element geometry
  if (alefluid_)
    EvaluateMassMatrix();

  // compute kinetic energy
  double energy = 0.0;
  Teuchos::RCP<Epetra_Vector> mtimesu = Teuchos::rcp(new Epetra_Vector(massmat_->OperatorRangeMap(), true));
  massmat_->Apply(*velnp_, *mtimesu);
  velnp_->Dot(*mtimesu, &energy);
  energy *= 0.5;

  // write to file
  if (myrank_ == 0 and (not logenergy_.is_null()))
  {
    (*logenergy_) << std::right << std::setw(9) << step_
                  << std::right << std::setw(16) << time_
                  << std::right << std::setw(16) << energy
                  << std::endl;
                  ;
  }

  return;
}

/*----------------------------------------------------------------------------*
 | Set time step size                                         mayr.mt 09/2013 |
 *----------------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::SetDt(const double dtnew)
{
  dta_ = dtnew;

  return;
}

/*----------------------------------------------------------------------------*
 | Set time and step                                          mayr.mt 09/2013 |
 *----------------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::SetTimeStep(const double time, const int step)
{
  step_ = step;
  time_ = time;

  return;
}

/*---------------------------------------------------------------*/
/* Apply Dirichlet boundary conditions on provided state vectors */
void FLD::FluidImplicitTimeInt::ApplyDirichletBC
(
  Teuchos::ParameterList& params,
  Teuchos::RCP<Epetra_Vector> systemvector,   //!< (may be Teuchos::null)
  Teuchos::RCP<Epetra_Vector> systemvectord,  //!< (may be Teuchos::null)
  Teuchos::RCP<Epetra_Vector> systemvectordd, //!< (may be Teuchos::null)
  bool recreatemap  //!< recreate mapextractor/toggle-vector
)
{
  // In the case of local coordinate systems, we have to rotate forward ...
  // --------------------------------------------------------------------------------
  if (locsysman_ != Teuchos::null)
  {
    if (systemvector != Teuchos::null)
        locsysman_->RotateGlobalToLocal(systemvector);
    if (systemvectord != Teuchos::null)
        locsysman_->RotateGlobalToLocal(systemvectord);
    if (systemvectordd != Teuchos::null)
        locsysman_->RotateGlobalToLocal(systemvectordd);
  }

  // Apply DBCs
  // --------------------------------------------------------------------------------
  discret_->ClearState();
  if (recreatemap)
  {
    discret_->EvaluateDirichlet(params, systemvector, systemvectord, systemvectordd,
                                Teuchos::null, dbcmaps_);
  }
  else
  {
    discret_->EvaluateDirichlet(params, systemvector, systemvectord, systemvectordd,
                               Teuchos::null, Teuchos::null);
  }
  discret_->ClearState();

  // In the case of local coordinate systems, we have to rotate back into global Cartesian frame
  // --------------------------------------------------------------------------------
  if (locsysman_ != Teuchos::null)
  {
    if (systemvector != Teuchos::null)
        locsysman_->RotateLocalToGlobal(systemvector);
    if (systemvectord != Teuchos::null)
        locsysman_->RotateLocalToGlobal(systemvectord);
    if (systemvectordd != Teuchos::null)
        locsysman_->RotateLocalToGlobal(systemvectordd);
  }

  return;
}

/*----------------------------------------------------------------------*
 * Explicit predictor                                   rasthofer 12/13 |
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::ExplicitPredictor()
{

  if(discret_->Comm().MyPID()==0)
  {
    printf("fluid: using explicit predictor %s",predictor_.c_str());
  }

  if (predictor_=="steady_state")
  {
    // steady state predictor
    //
    //       n+1    n
    //      u    = u
    //       (0)
    //
    //  and
    //
    //       n+1    n
    //      p    = p
    //       (0)

    // this has already been done in TimeUpdate()
  }
  else if(predictor_=="zero_acceleration")
  {
    // zero acceleration predictor
    //
    //       n+1    n                   n
    //      u    = u  + (1-gamma)*dt*acc
    //       (0)
    //
    //  and
    //
    //       n+1    n
    //      p    = p
    //       (0)
    //
    velnp_->Update(1.0,*veln_,0.0);

    // split between acceleration and pressure
    Teuchos::RCP<Epetra_Vector> inc = velpressplitter_.ExtractOtherVector(accn_);
    inc->Scale((1.0-theta_)*dta_);

    velpressplitter_.AddOtherVector(inc,velnp_);
  }
  else if(predictor_=="constant_acceleration")
  {
    // constant acceleration predictor
    //
    //       n+1    n         n
    //      u    = u  + dt*acc
    //       (0)
    //
    //  and
    //
    //       n+1    n
    //      p    = p
    //       (0)
    //
    velnp_->Update(1.0,*veln_,0.0);

    Teuchos::RCP<Epetra_Vector> inc = velpressplitter_.ExtractOtherVector(accn_);
    inc->Scale(dta_);

    velpressplitter_.AddOtherVector(inc,velnp_);
  }
  else if(predictor_=="constant_increment")
  {
    // constant increment predictor
    //
    //       n+1      n    n-1
    //      u    = 2*u  - u
    //       (0)
    //
    //  and
    //
    //       n+1    n
    //      p    = p
    //       (0)
    //
    velnp_->Update(1.0,*veln_,0.0);

    Teuchos::RCP<Epetra_Vector> un  = velpressplitter_.ExtractOtherVector(veln_ );
    Teuchos::RCP<Epetra_Vector> unm = velpressplitter_.ExtractOtherVector(velnm_);
    unm->Scale(-1.0);

    velpressplitter_.AddOtherVector(un ,velnp_);
    velpressplitter_.AddOtherVector(unm,velnp_);
  }
  else if(predictor_=="explicit_second_order_midpoint")
  {
    // the conventional explicit second order predictor (assuming constant dt)
    // also known as leapfrog integration
    /*
    //                        /          n    n-1 \
    //       n+1    n        |      n   u  - u     |
    //      u    = u  + dt * | 2*acc  - ---------  |
    //       (0)             |             dt      |
    //                        \                   /
    // respectively
    //
    //       n+1    n-1               n
    //      u    = u    + 2 * dt * acc
    //       (0)
    //
    //  and
    //
    //       n+1    n
    //      p    = p
    //       (0)
    */
    velnp_->Update(1.0,*veln_,0.0);

    // split between acceleration and pressure
    Teuchos::RCP<Epetra_Vector> unm = velpressplitter_.ExtractOtherVector(velnm_);
    Teuchos::RCP<Epetra_Vector> an  = velpressplitter_.ExtractOtherVector(accn_ );

    unm->Update(2.0*dta_,*an,1.0);

    velpressplitter_.InsertOtherVector(unm,velnp_);
  }
  else
    dserror("Unknown fluid predictor %s", predictor_.c_str());

  if(discret_->Comm().MyPID()==0)
  {
    printf("\n");
  }

  return;
}

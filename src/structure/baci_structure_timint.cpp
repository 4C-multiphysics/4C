/*----------------------------------------------------------------------*/
/*! \file
\brief Time integration for structural dynamics

\level 1

*/
/*----------------------------------------------------------------------*/

/*----------------------------------------------------------------------*/
/* headers */
#include "baci_structure_timint.H"

#include "baci_beamcontact_beam3contact_manager.H"
#include "baci_cardiovascular0d_manager.H"
#include "baci_comm_utils.H"
#include "baci_constraint_manager.H"
#include "baci_constraint_solver.H"
#include "baci_constraint_springdashpot_manager.H"
#include "baci_contact_meshtying_contact_bridge.H"
#include "baci_inpar_beamcontact.H"
#include "baci_inpar_contact.H"
#include "baci_inpar_mortar.H"
#include "baci_io.H"
#include "baci_io_control.H"
#include "baci_io_gmsh.H"
#include "baci_io_pstream.H"
#include "baci_lib_discret_faces.H"
#include "baci_lib_globalproblem.H"
#include "baci_lib_locsys.H"
#include "baci_linalg_blocksparsematrix.H"
#include "baci_linalg_multiply.H"
#include "baci_linalg_serialdensevector.H"
#include "baci_linalg_sparsematrix.H"
#include "baci_linalg_utils_densematrix_communication.H"
#include "baci_linalg_utils_sparse_algebra_create.H"
#include "baci_linalg_utils_sparse_algebra_manipulation.H"
#include "baci_linear_solver_method_linalg.H"
#include "baci_mat_micromaterial.H"
#include "baci_mat_par_bundle.H"
#include "baci_mor_pod.H"
#include "baci_mortar_manager_base.H"
#include "baci_mortar_strategy_base.H"
#include "baci_mortar_utils.H"
#include "baci_poroelast_utils.H"
#include "baci_so3_sh8p8.H"
#include "baci_stru_multi_microstatic.H"
#include "baci_structure_resulttest.H"
#include "baci_structure_timint_genalpha.H"

#include <Teuchos_TimeMonitor.hpp>

#include <iostream>

BACI_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/* print tea time logo */
void STR::TimInt::Logo()
{
  IO::cout << "Welcome to Structural Time Integration " << IO::endl;
  IO::cout << "     __o__                          __o__" << IO::endl;
  IO::cout << "__  /-----\\__                  __  /-----\\__" << IO::endl;
  IO::cout << "\\ \\/       \\ \\    |       \\    \\ \\/       \\ \\" << IO::endl;
  IO::cout << " \\ |  tea  | |    |-------->    \\ |  tea  | |" << IO::endl;
  IO::cout << "  \\|       |_/    |       /      \\|       |_/" << IO::endl;
  IO::cout << "    \\_____/   ._                   \\_____/   ._ _|_ /|" << IO::endl;
  IO::cout << "              | |                            | | |   |" << IO::endl;
  IO::cout << IO::endl;
}

/*----------------------------------------------------------------------*/
/* constructor */
STR::TimInt::TimInt(const Teuchos::ParameterList& timeparams,
    const Teuchos::ParameterList& ioparams, const Teuchos::ParameterList& sdynparams,
    const Teuchos::ParameterList& xparams, Teuchos::RCP<DRT::Discretization> actdis,
    Teuchos::RCP<CORE::LINALG::Solver> solver, Teuchos::RCP<CORE::LINALG::Solver> contactsolver,
    Teuchos::RCP<IO::DiscretizationWriter> output)
    : discret_(actdis),
      facediscret_(Teuchos::null),
      myrank_(actdis->Comm().MyPID()),
      solver_(solver),
      contactsolver_(contactsolver),
      solveradapttol_(INPUT::IntegralValue<int>(sdynparams, "ADAPTCONV") == 1),
      solveradaptolbetter_(sdynparams.get<double>("ADAPTCONV_BETTER")),
      dbcmaps_(Teuchos::rcp(new CORE::LINALG::MapExtractor())),
      divcontype_(INPUT::IntegralValue<INPAR::STR::DivContAct>(sdynparams, "DIVERCONT")),
      divconrefinementlevel_(0),
      divconnumfinestep_(0),
      sdynparams_(sdynparams),
      output_(output),
      printscreen_(ioparams.get<int>("STDOUTEVRY")),
      printlogo_(bool(printscreen_)),  // no std out no logo
      printiter_(true),                // ADD INPUT PARAMETER
      outputeveryiter_((bool)INPUT::IntegralValue<int>(ioparams, "OUTPUT_EVERY_ITER")),
      oei_filecounter_(ioparams.get<int>("OEI_FILE_COUNTER")),
      writerestartevery_(timeparams.get<int>("RESTARTEVRY")),
      writeele_((bool)INPUT::IntegralValue<int>(ioparams, "STRUCT_ELE")),
      writestate_((bool)INPUT::IntegralValue<int>(ioparams, "STRUCT_DISP")),
      writevelacc_((bool)INPUT::IntegralValue<int>(ioparams, "STRUCT_VEL_ACC")),
      writeresultsevery_(timeparams.get<int>("RESULTSEVRY")),
      writestress_(INPUT::IntegralValue<INPAR::STR::StressType>(ioparams, "STRUCT_STRESS")),
      writecouplstress_(
          INPUT::IntegralValue<INPAR::STR::StressType>(ioparams, "STRUCT_COUPLING_STRESS")),
      writestrain_(INPUT::IntegralValue<INPAR::STR::StrainType>(ioparams, "STRUCT_STRAIN")),
      writeplstrain_(
          INPUT::IntegralValue<INPAR::STR::StrainType>(ioparams, "STRUCT_PLASTIC_STRAIN")),
      writeoptquantity_(
          INPUT::IntegralValue<INPAR::STR::OptQuantityType>(ioparams, "STRUCT_OPTIONAL_QUANTITY")),
      writeenergyevery_(sdynparams.get<int>("RESEVRYERGY")),
      writesurfactant_((bool)INPUT::IntegralValue<int>(ioparams, "STRUCT_SURFACTANT")),
      writerotation_((bool)INPUT::IntegralValue<int>(ioparams, "OUTPUT_ROT")),
      energyfile_(Teuchos::null),
      damping_(INPUT::IntegralValue<INPAR::STR::DampKind>(sdynparams, "DAMPING")),
      dampk_(sdynparams.get<double>("K_DAMP")),
      dampm_(sdynparams.get<double>("M_DAMP")),
      conman_(Teuchos::null),
      consolv_(Teuchos::null),
      cardvasc0dman_(Teuchos::null),
      springman_(Teuchos::null),
      cmtbridge_(Teuchos::null),
      beamcman_(Teuchos::null),
      locsysman_(Teuchos::null),
      pressure_(Teuchos::null),
      gmsh_out_(false),
      time_(Teuchos::null),
      timen_(0.0),
      dt_(Teuchos::null),
      timemax_(timeparams.get<double>("MAXTIME")),
      stepmax_(timeparams.get<int>("NUMSTEP")),
      step_(0),
      stepn_(0),
      rand_tsfac_(1.0),
      firstoutputofrun_(true),
      lumpmass_(INPUT::IntegralValue<int>(sdynparams, "LUMPMASS") == 1),
      zeros_(Teuchos::null),
      dis_(Teuchos::null),
      dismat_(Teuchos::null),
      vel_(Teuchos::null),
      acc_(Teuchos::null),
      disn_(Teuchos::null),
      dismatn_(Teuchos::null),
      veln_(Teuchos::null),
      accn_(Teuchos::null),
      fifc_(Teuchos::null),
      fresn_str_(Teuchos::null),
      fintn_str_(Teuchos::null),
      stiff_(Teuchos::null),
      mass_(Teuchos::null),
      damp_(Teuchos::null),
      timer_(Teuchos::rcp(new Teuchos::Time("", true))),
      dtsolve_(0.0),
      dtele_(0.0),
      dtcmt_(0.0),
      strgrdisp_(Teuchos::null),
      mor_(Teuchos::null),
      issetup_(false),
      isinit_(false)
{
  // Keep this constructor empty except some basic input error catching!
  // First do everything on the more basic objects like the discretizations, like e.g.
  // redistribution of elements. Only then call the setup to this class. This will call the
  // setup to all classes in the inheritance hierarchy. This way, this class may also override
  // a method that is called during Setup() in a base class.

  if (sdynparams.get<int>("OUTPUT_STEP_OFFSET") != 0)
  {
    dserror(
        "Output step offset (\"OUTPUT_STEP_OFFSET\" != 0) is not supported in the old structural "
        "time integration");
  }
}

/*----------------------------------------------------------------------------------------------*
 * Initialize this class                                                            rauch 09/16 |
 *----------------------------------------------------------------------------------------------*/
void STR::TimInt::Init(const Teuchos::ParameterList& timeparams,
    const Teuchos::ParameterList& sdynparams, const Teuchos::ParameterList& xparams,
    Teuchos::RCP<DRT::Discretization> actdis, Teuchos::RCP<CORE::LINALG::Solver> solver)
{
  // invalidate setup
  SetIsSetup(false);

  // welcome user
  if ((printlogo_) and (myrank_ == 0))
  {
    Logo();
  }

  // check whether discretisation has been completed
  if (not discret_->Filled() || not actdis->HaveDofs())
  {
    dserror("Discretisation is not complete or has no dofs!");
  }

  // time state
  time_ = Teuchos::rcp(new TIMESTEPPING::TimIntMStep<double>(
      0, 0, 0.0));  // HERE SHOULD BE SOMETHING LIKE (sdynparams.get<double>("TIMEINIT"))
  dt_ =
      Teuchos::rcp(new TIMESTEPPING::TimIntMStep<double>(0, 0, timeparams.get<double>("TIMESTEP")));
  step_ = 0;
  timen_ = (*time_)[0] + (*dt_)[0];  // set target time to initial time plus step size
  stepn_ = step_ + 1;

  // output file for energy
  if ((writeenergyevery_ != 0) and (myrank_ == 0)) AttachEnergyFile();

  // initialize constraint manager
  conman_ = Teuchos::rcp(new UTILS::ConstrManager());
  conman_->Init(discret_, sdynparams_);

  // create stiffness, mass matrix and other fields
  CreateFields();

  // stay with us

  // we have successfully initialized this class
  SetIsInit(true);
}

/*----------------------------------------------------------------------------------------------*
 * Setup this class                                                                 rauch 09/16 |
 *----------------------------------------------------------------------------------------------*/
void STR::TimInt::Setup()
{
  // we have to call Init() before
  CheckIsInit();

  CreateAllSolutionVectors();

  // create stiffness, mass matrix and other fields
  CreateFields();

  // set initial fields
  SetInitialFields();

  // setup constraint manager
  conman_->Setup((*dis_)(0), sdynparams_);

  // model order reduction
  mor_ = Teuchos::rcp(new UTILS::MOR(discret_));

  // initialize 0D cardiovascular manager
  cardvasc0dman_ = Teuchos::rcp(new UTILS::Cardiovascular0DManager(discret_, (*dis_)(0),
      sdynparams_, DRT::Problem::Instance()->Cardiovascular0DStructuralParams(), *solver_, mor_));

  // initialize spring dashpot manager
  springman_ = Teuchos::rcp(new UTILS::SpringDashpotManager(discret_));


  // initialize constraint solver if constraints are defined
  if (conman_->HaveConstraint())
  {
    consolv_ = Teuchos::rcp(new UTILS::ConstraintSolver(discret_, *solver_, dbcmaps_, sdynparams_));
  }

  // check for beam contact
  {
    // If beam contact (no statistical mechanics) is chosen in the input file, then a
    // corresponding manager object stored via #beamcman_ is created and all relevant
    // stuff is initialized. Else, #beamcman_ remains a Teuchos::null pointer.
    PrepareBeamContact(sdynparams_);
  }
  // check for mortar contact or meshtying
  {
    // If mortar contact or meshtying is chosen in the input file, then a
    // corresponding manager object stored via #cmtman_ is created and all relevant
    // stuff is initialized. Else, #cmtman_ remains a Teuchos::null pointer.
    PrepareContactMeshtying(sdynparams_);
  }

  // check whether we have locsys BCs and create LocSysManager if so
  // after checking
  {
    std::vector<DRT::Condition*> locsysconditions(0);
    discret_->GetCondition("Locsys", locsysconditions);
    if (locsysconditions.size())
    {
      locsysman_ = Teuchos::rcp(new DRT::UTILS::LocsysManager(*discret_));
      // in case we have no time dependent locsys conditions in our problem,
      // this is the only time where the whole setup routine is conducted.
      locsysman_->Update(-1.0, {});
    }
  }

  // check if we have elements which use a continuous displacement and pressure
  // field
  {
    int locnumsosh8p8 = 0;
    // Loop through all elements on processor
    for (int i = 0; i < discret_->NumMyColElements(); ++i)
    {
      // get the actual element

      if (discret_->lColElement(i)->ElementType() == DRT::ELEMENTS::So_sh8p8Type::Instance())
        locnumsosh8p8 += 1;
    }
    // Was at least one SoSh8P8 found on one processor?
    int glonumsosh8p8 = 0;
    discret_->Comm().MaxAll(&locnumsosh8p8, &glonumsosh8p8, 1);
    // Yes, it was. Go ahead for all processors (even if they do not carry any SoSh8P8 elements)
    if (glonumsosh8p8 > 0)
    {
      pressure_ = Teuchos::rcp(new CORE::LINALG::MapExtractor());
      const int ndim = 3;
      CORE::LINALG::CreateMapExtractorFromDiscretization(*discret_, ndim, *pressure_);
    }
  }

  // check if we have elements with a micro-material
  havemicromat_ = false;
  for (int i = 0; i < discret_->NumMyColElements(); i++)
  {
    DRT::Element* actele = discret_->lColElement(i);
    Teuchos::RCP<MAT::Material> mat = actele->Material();
    if (mat != Teuchos::null && mat->MaterialType() == INPAR::MAT::m_struct_multiscale)
    {
      havemicromat_ = true;
      break;
    }
  }


  // Check for porosity dofs within the structure and build a map extractor if necessary
  porositysplitter_ = POROELAST::UTILS::BuildPoroSplitter(discret_);


  // we have successfully set up this class
  SetIsSetup(true);
}

/*----------------------------------------------------------------------------------------------*
 * Create all solution vectors
 *----------------------------------------------------------------------------------------------*/
void STR::TimInt::CreateAllSolutionVectors()
{
  // displacements D_{n}
  dis_ = Teuchos::rcp(new TIMESTEPPING::TimIntMStep<Epetra_Vector>(0, 0, DofRowMapView(), true));
  // velocities V_{n}
  vel_ = Teuchos::rcp(new TIMESTEPPING::TimIntMStep<Epetra_Vector>(0, 0, DofRowMapView(), true));
  // accelerations A_{n}
  acc_ = Teuchos::rcp(new TIMESTEPPING::TimIntMStep<Epetra_Vector>(0, 0, DofRowMapView(), true));

  // displacements D_{n+1} at t_{n+1}
  disn_ = CORE::LINALG::CreateVector(*DofRowMapView(), true);

  if ((DRT::Problem::Instance()->GetProblemType() == ProblemType::struct_ale and
          (DRT::Problem::Instance()->WearParams()).get<double>("WEARCOEFF") > 0.0))
  {
    // material displacements Dm_{n+1} at t_{n+1}
    dismatn_ = CORE::LINALG::CreateVector(*DofRowMapView(), true);

    // material_displacements D_{n}
    dismat_ =
        Teuchos::rcp(new TIMESTEPPING::TimIntMStep<Epetra_Vector>(0, 0, DofRowMapView(), true));
  }

  // velocities V_{n+1} at t_{n+1}
  veln_ = CORE::LINALG::CreateVector(*DofRowMapView(), true);
  // accelerations A_{n+1} at t_{n+1}
  accn_ = CORE::LINALG::CreateVector(*DofRowMapView(), true);
  // create empty interface force vector
  fifc_ = CORE::LINALG::CreateVector(*DofRowMapView(), true);
}

/*-------------------------------------------------------------------------------------------*
 * Create matrices when setting up time integrator
 *-------------------------------------------------------------------------------------------*/
void STR::TimInt::CreateFields()
{
  // a zero vector of full length
  zeros_ = CORE::LINALG::CreateVector(*DofRowMapView(), true);

  // Map containing Dirichlet DOFs
  {
    Teuchos::ParameterList p;
    p.set("total time", timen_);
    discret_->EvaluateDirichlet(p, zeros_, Teuchos::null, Teuchos::null, Teuchos::null, dbcmaps_);
    zeros_->PutScalar(0.0);  // just in case of change
  }

  // create empty matrices
  stiff_ = Teuchos::rcp(new CORE::LINALG::SparseMatrix(*DofRowMapView(), 81, false, true));
  mass_ = Teuchos::rcp(new CORE::LINALG::SparseMatrix(*DofRowMapView(), 81, false, true));
  if (damping_ != INPAR::STR::damp_none)
  {
    if (HaveNonlinearMass() == INPAR::STR::ml_none)
    {
      damp_ = Teuchos::rcp(new CORE::LINALG::SparseMatrix(*DofRowMapView(), 81, false, true));
    }
    else
    {
      // Since our element evaluate routine is only designed for two input matrices
      //(stiffness and damping or stiffness and mass) its not possible, to have nonlinear
      // inertia forces AND material damping.
      dserror("So far its not possible to model nonlinear inertia forces and damping!");
    }
  }
}

/*----------------------------------------------------------------------*/
/* Set intitial fields in structure (e.g. initial velocities */
void STR::TimInt::SetInitialFields()
{
  //***************************************************
  // Data that needs to be handed into discretization:
  // - std::string field: name of initial field to be set
  // - std::vector<int> localdofs: local dof ids affected
  //***************************************************

  // set initial velocity field if existing
  const std::string field = "Velocity";
  std::vector<int> localdofs;
  localdofs.push_back(0);
  localdofs.push_back(1);
  localdofs.push_back(2);
  discret_->EvaluateInitialField(field, (*vel_)(0), localdofs);

  // set initial porosity field if existing
  const std::string porosityfield = "Porosity";
  std::vector<int> porositylocaldofs;
  porositylocaldofs.push_back(DRT::Problem::Instance()->NDim());
  discret_->EvaluateInitialField(porosityfield, (*dis_)(0), porositylocaldofs);
}

/*----------------------------------------------------------------------*/
/* Check for beam contact and do preparations */
void STR::TimInt::PrepareBeamContact(const Teuchos::ParameterList& sdynparams)
{
  // some parameters
  const Teuchos::ParameterList& beamcontact = DRT::Problem::Instance()->BeamContactParams();
  INPAR::BEAMCONTACT::Strategy strategy =
      INPUT::IntegralValue<INPAR::BEAMCONTACT::Strategy>(beamcontact, "BEAMS_STRATEGY");

  // conditions for potential-based beam interaction
  std::vector<DRT::Condition*> beampotconditions(0);
  discret_->GetCondition("BeamPotentialLineCharge", beampotconditions);

  // only continue if beam contact unmistakably chosen in input file or beam potential conditions
  // applied
  if (strategy != INPAR::BEAMCONTACT::bstr_none or (int) beampotconditions.size() != 0)
  {
    // store integration parameter alphaf into beamcman_ as well
    // (for all cases except OST, GenAlpha and GEMM this is zero)
    // (note that we want to hand in theta in the OST case, which
    // is defined just the other way round as alphaf in GenAlpha schemes.
    // Thus, we have to hand in 1.0-theta for OST!!!)
    double alphaf = TimIntParam();

    // create beam contact manager
    beamcman_ = Teuchos::rcp(new CONTACT::Beam3cmanager(*discret_, alphaf));

    // gmsh output at beginning of simulation
#ifdef GMSHTIMESTEPS
    beamcman_->GmshOutput(*disn_, 0, 0, true);
#endif
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void STR::TimInt::PrepareContactMeshtying(const Teuchos::ParameterList& sdynparams)
{
  TEUCHOS_FUNC_TIME_MONITOR("STR::TimInt::PrepareContactMeshtying");

  // some parameters
  const Teuchos::ParameterList& smortar = DRT::Problem::Instance()->MortarCouplingParams();
  const Teuchos::ParameterList& scontact = DRT::Problem::Instance()->ContactDynamicParams();
  INPAR::MORTAR::ShapeFcn shapefcn =
      INPUT::IntegralValue<INPAR::MORTAR::ShapeFcn>(smortar, "LM_SHAPEFCN");
  INPAR::CONTACT::SolvingStrategy soltype =
      INPUT::IntegralValue<INPAR::CONTACT::SolvingStrategy>(scontact, "STRATEGY");
  INPAR::CONTACT::SystemType systype =
      INPUT::IntegralValue<INPAR::CONTACT::SystemType>(scontact, "SYSTEM");
  INPAR::MORTAR::AlgorithmType algorithm =
      INPUT::IntegralValue<INPAR::MORTAR::AlgorithmType>(smortar, "ALGORITHM");

  // check mortar contact or meshtying conditions
  std::vector<DRT::Condition*> mortarconditions(0);
  std::vector<DRT::Condition*> contactconditions(0);

  discret_->GetCondition("Mortar", mortarconditions);
  discret_->GetCondition("Contact", contactconditions);

  // double-check for contact/meshtying conditions
  if (mortarconditions.size() == 0 and contactconditions.size() == 0) return;

  // check if only beam-to-solid contact / meshtying conditions (and leave if so)
  bool realcontactconditions = false;
  for (const auto& contactCondition : contactconditions)
  {
    if (*contactCondition->Get<std::string>("Application") != "Beamtosolidcontact" &&
        *contactCondition->Get<std::string>("Application") != "Beamtosolidmeshtying")
      realcontactconditions = true;
  }
  if (mortarconditions.size() == 0 and !realcontactconditions) return;

  // store integration parameter alphaf into cmtman_ as well
  // (for all cases except OST, GenAlpha and GEMM this is zero)
  // (note that we want to hand in theta in the OST case, which
  // is defined just the other way round as alphaf in GenAlpha schemes.
  // Thus, we have to hand in 1-theta for OST!!!)
  double time_integration_factor = 0.0;
  const bool do_endtime = INPUT::IntegralValue<int>(scontact, "CONTACTFORCE_ENDTIME");
  if (!do_endtime) time_integration_factor = TimIntParam();

  // create instance for meshtying contact bridge
  cmtbridge_ = Teuchos::rcp(new CONTACT::MeshtyingContactBridge(
      *discret_, mortarconditions, contactconditions, time_integration_factor));

  cmtbridge_->StoreDirichletStatus(dbcmaps_);
  cmtbridge_->SetState(zeros_);

  // contact and constraints together not yet implemented
  if (conman_->HaveConstraint())
    dserror("Constraints and contact cannot be treated at the same time yet");

  // print messages for multifield problems (e.g FSI)
  const ProblemType probtype = DRT::Problem::Instance()->GetProblemType();
  const std::string probname = DRT::Problem::Instance()->ProblemName();
  if (probtype != ProblemType::structure && !myrank_)
  {
    // warnings
#ifdef CONTACTPSEUDO2D
    std::cout << "WARNING: The flag CONTACTPSEUDO2D is switched on. If this "
              << "is a real 3D problem, switch it off!" << std::endl;
#else
    std::cout << "WARNING: The flag CONTACTPSEUDO2D is switched off. If this "
              << "is a 2D problem modeled pseudo-3D, switch it on!" << std::endl;
#endif  // #ifdef CONTACTPSEUDO2D
  }

  // initialization of meshtying
  if (cmtbridge_->HaveMeshtying())
  {
    // FOR MESHTYING (ONLY ONCE), NO FUNCTIONALITY FOR CONTACT CASES
    // (1) do mortar coupling in reference configuration
    cmtbridge_->MtManager()->GetStrategy().MortarCoupling(zeros_);

    // perform mesh initialization if required by input parameter MESH_RELOCATION
    auto mesh_relocation_parameter = INPUT::IntegralValue<INPAR::MORTAR::MeshRelocation>(
        DRT::Problem::Instance()->MortarCouplingParams(), "MESH_RELOCATION");

    if (mesh_relocation_parameter == INPAR::MORTAR::relocation_initial)
    {
      // (2) perform mesh initialization for rotational invariance (interface)
      // and return the modified slave node positions in vector Xslavemod
      Teuchos::RCP<const Epetra_Vector> Xslavemod =
          cmtbridge_->MtManager()->GetStrategy().MeshInitialization();

      // (3) apply result of mesh initialization to underlying problem discretization
      ApplyMeshInitialization(Xslavemod);
    }
    else if (mesh_relocation_parameter == INPAR::MORTAR::relocation_timestep)
    {
      dserror(
          "Meshtying with MESH_RELOCATION every_timestep not permitted. Change to MESH_RELOCATION "
          "initial or MESH_RELOCATION no.");
    }
  }

  // initialization of contact
  if (cmtbridge_->HaveContact())
  {
    // FOR PENALTY CONTACT (ONLY ONCE), NO FUNCTIONALITY FOR OTHER CASES
    // (1) Explicitly store gap-scaling factor kappa
    cmtbridge_->ContactManager()->GetStrategy().SaveReferenceState(zeros_);

    // FOR CONTACT FORMULATIONS (ONLY ONCE)
    // (1) Evaluate reference state for friction and initialize gap
    cmtbridge_->ContactManager()->GetStrategy().EvaluateReferenceState();
  }

  // visualization of initial configuration
#ifdef MORTARGMSH3
  bool gmsh = INPUT::IntegralValue<int>(DRT::Problem::Instance()->IOParams(), "OUTPUT_GMSH");
  if (gmsh) cmtbridge_->VisualizeGmsh(0);
#endif  // #ifdef MORTARGMSH3

  //**********************************************************************
  // prepare solvers for contact/meshtying problem
  //**********************************************************************
  {
    // only plausibility check, that a contact solver is available
    if (contactsolver_ == Teuchos::null)
      dserror("No contact solver in STR::TimInt::PrepareContactMeshtying? Cannot be!");
  }

  // output of strategy / shapefcn / system type to screen
  {
    // output
    if (!myrank_)
    {
      if (algorithm == INPAR::MORTAR::algorithm_mortar)
      {
        // saddle point formulation
        if (systype == INPAR::CONTACT::system_saddlepoint)
        {
          if (soltype == INPAR::CONTACT::solution_lagmult &&
              shapefcn == INPAR::MORTAR::shape_standard)
          {
            std::cout << "================================================================"
                      << std::endl;
            std::cout << "===== Standard Lagrange multiplier strategy ===================="
                      << std::endl;
            std::cout << "===== (Saddle point formulation) ==============================="
                      << std::endl;
            std::cout << "================================================================\n"
                      << std::endl;
          }
          else if (soltype == INPAR::CONTACT::solution_lagmult &&
                   shapefcn == INPAR::MORTAR::shape_dual)
          {
            std::cout << "================================================================"
                      << std::endl;
            std::cout << "===== Dual Lagrange multiplier strategy ========================"
                      << std::endl;
            std::cout << "===== (Saddle point formulation) ==============================="
                      << std::endl;
            std::cout << "================================================================\n"
                      << std::endl;
          }
          else if (soltype == INPAR::CONTACT::solution_multiscale &&
                   shapefcn == INPAR::MORTAR::shape_standard)
          {
            std::cout << "================================================================"
                      << std::endl;
            std::cout << "===== Standard Multi Scale strategy ================================"
                      << std::endl;
            std::cout << "===== (Pure displacement formulation) =========================="
                      << std::endl;
            std::cout << "================================================================\n"
                      << std::endl;
          }
          else if (soltype == INPAR::CONTACT::solution_multiscale &&
                   shapefcn == INPAR::MORTAR::shape_dual)
          {
            std::cout << "================================================================"
                      << std::endl;
            std::cout << "===== Dual Multi Scale strategy ===================================="
                      << std::endl;
            std::cout << "===== (Pure displacement formulation) =========================="
                      << std::endl;
            std::cout << "================================================================\n"
                      << std::endl;
          }
          else if (soltype == INPAR::CONTACT::solution_lagmult &&
                   INPUT::IntegralValue<INPAR::MORTAR::LagMultQuad>(smortar, "LM_QUAD") ==
                       INPAR::MORTAR::lagmult_const)
          {
            std::cout << "================================================================"
                      << std::endl;
            std::cout << "===== const Lagrange multiplier strategy ======================="
                      << std::endl;
            std::cout << "===== (Saddle point formulation) ==============================="
                      << std::endl;
            std::cout << "================================================================\n"
                      << std::endl;
          }
          else if (soltype == INPAR::CONTACT::solution_lagmult &&
                   shapefcn == INPAR::MORTAR::shape_petrovgalerkin)
          {
            std::cout << "================================================================"
                      << std::endl;
            std::cout << "===== Petrov-Galerkin Lagrange multiplier strategy ============="
                      << std::endl;
            std::cout << "===== (Saddle point formulation) ==============================="
                      << std::endl;
            std::cout << "================================================================\n"
                      << std::endl;
          }
          else if (soltype == INPAR::CONTACT::solution_penalty &&
                   shapefcn == INPAR::MORTAR::shape_standard)
          {
            std::cout << "================================================================"
                      << std::endl;
            std::cout << "===== Standard Penalty strategy ================================"
                      << std::endl;
            std::cout << "===== (Pure displacement formulation) =========================="
                      << std::endl;
            std::cout << "================================================================\n"
                      << std::endl;
          }
          else if (soltype == INPAR::CONTACT::solution_penalty &&
                   shapefcn == INPAR::MORTAR::shape_dual)
          {
            std::cout << "================================================================"
                      << std::endl;
            std::cout << "===== Dual Penalty strategy ===================================="
                      << std::endl;
            std::cout << "===== (Pure displacement formulation) =========================="
                      << std::endl;
            std::cout << "================================================================\n"
                      << std::endl;
          }
          else if (soltype == INPAR::CONTACT::solution_uzawa &&
                   shapefcn == INPAR::MORTAR::shape_standard)
          {
            std::cout << "================================================================"
                      << std::endl;
            std::cout << "===== Uzawa Augmented Lagrange strategy ========================"
                      << std::endl;
            std::cout << "===== (Pure displacement formulation) =========================="
                      << std::endl;
            std::cout << "================================================================\n"
                      << std::endl;
          }
          else if (soltype == INPAR::CONTACT::solution_uzawa &&
                   shapefcn == INPAR::MORTAR::shape_dual)
          {
            std::cout << "================================================================"
                      << std::endl;
            std::cout << "===== Dual Uzawa Augmented Lagrange strategy ==================="
                      << std::endl;
            std::cout << "===== (Pure displacement formulation) =========================="
                      << std::endl;
            std::cout << "================================================================\n"
                      << std::endl;
          }
          else if (soltype == INPAR::CONTACT::solution_augmented &&
                   shapefcn == INPAR::MORTAR::shape_standard)
          {
            std::cout << "================================================================"
                      << std::endl;
            std::cout << "===== Augmented Lagrange strategy =============================="
                      << std::endl;
            std::cout << "===== (Saddle point formulation) ==============================="
                      << std::endl;
            std::cout << "================================================================\n"
                      << std::endl;
          }
          else
            dserror("Invalid strategy or shape function type for contact/meshtying");
        }

        // condensed formulation
        else if (systype == INPAR::CONTACT::system_condensed ||
                 systype == INPAR::CONTACT::system_condensed_lagmult)
        {
          if (soltype == INPAR::CONTACT::solution_lagmult && shapefcn == INPAR::MORTAR::shape_dual)
          {
            std::cout << "================================================================"
                      << std::endl;
            std::cout << "===== Dual Lagrange multiplier strategy ========================"
                      << std::endl;
            std::cout << "===== (Condensed formulation) =================================="
                      << std::endl;
            std::cout << "================================================================\n"
                      << std::endl;
          }
          else if (soltype == INPAR::CONTACT::solution_lagmult &&
                   shapefcn == INPAR::MORTAR::shape_petrovgalerkin)
          {
            std::cout << "================================================================"
                      << std::endl;
            std::cout << "===== Petrov-Galerkin Lagrange multiplier strategy ============="
                      << std::endl;
            std::cout << "===== (Condensed formulation) =================================="
                      << std::endl;
            std::cout << "================================================================\n"
                      << std::endl;
          }
          else if (soltype == INPAR::CONTACT::solution_lagmult &&
                   INPUT::IntegralValue<INPAR::MORTAR::LagMultQuad>(smortar, "LM_QUAD") ==
                       INPAR::MORTAR::lagmult_const)
          {
            std::cout << "================================================================"
                      << std::endl;
            std::cout << "===== const Lagrange multiplier strategy ======================="
                      << std::endl;
            std::cout << "===== (Condensed formulation) =================================="
                      << std::endl;
            std::cout << "================================================================\n"
                      << std::endl;
          }
          else if (soltype == INPAR::CONTACT::solution_multiscale &&
                   shapefcn == INPAR::MORTAR::shape_standard)
          {
            std::cout << "================================================================"
                      << std::endl;
            std::cout << "===== Standard Rough Contact strategy ================================"
                      << std::endl;
            std::cout << "===== (Pure displacement formulation) =========================="
                      << std::endl;
            std::cout << "================================================================\n"
                      << std::endl;
          }
          else if (soltype == INPAR::CONTACT::solution_multiscale &&
                   shapefcn == INPAR::MORTAR::shape_dual)
          {
            std::cout << "================================================================"
                      << std::endl;
            std::cout << "===== Dual Rough Contact strategy ===================================="
                      << std::endl;
            std::cout << "===== (Pure displacement formulation) =========================="
                      << std::endl;
            std::cout << "================================================================\n"
                      << std::endl;
          }
          else if (soltype == INPAR::CONTACT::solution_penalty &&
                   shapefcn == INPAR::MORTAR::shape_standard)
          {
            std::cout << "================================================================"
                      << std::endl;
            std::cout << "===== Standard Penalty strategy ================================"
                      << std::endl;
            std::cout << "===== (Pure displacement formulation) =========================="
                      << std::endl;
            std::cout << "================================================================\n"
                      << std::endl;
          }
          else if (soltype == INPAR::CONTACT::solution_penalty &&
                   shapefcn == INPAR::MORTAR::shape_dual)
          {
            std::cout << "================================================================"
                      << std::endl;
            std::cout << "===== Dual Penalty strategy ===================================="
                      << std::endl;
            std::cout << "===== (Pure displacement formulation) =========================="
                      << std::endl;
            std::cout << "================================================================\n"
                      << std::endl;
          }
          else if (soltype == INPAR::CONTACT::solution_uzawa &&
                   shapefcn == INPAR::MORTAR::shape_standard)
          {
            std::cout << "================================================================"
                      << std::endl;
            std::cout << "===== Uzawa Augmented Lagrange strategy ========================"
                      << std::endl;
            std::cout << "===== (Pure displacement formulation) =========================="
                      << std::endl;
            std::cout << "================================================================\n"
                      << std::endl;
          }
          else if (soltype == INPAR::CONTACT::solution_uzawa &&
                   shapefcn == INPAR::MORTAR::shape_dual)
          {
            std::cout << "================================================================"
                      << std::endl;
            std::cout << "===== Dual Uzawa Augmented Lagrange strategy ==================="
                      << std::endl;
            std::cout << "===== (Pure displacement formulation) =========================="
                      << std::endl;
            std::cout << "================================================================\n"
                      << std::endl;
          }
          else
            dserror("Invalid strategy or shape function type for contact/meshtying");
        }
      }
      else if (algorithm == INPAR::MORTAR::algorithm_nts)
      {
        std::cout << "================================================================"
                  << std::endl;
        std::cout << "===== Node-To-Segment approach ================================="
                  << std::endl;
        std::cout << "================================================================\n"
                  << std::endl;
      }
      else if (algorithm == INPAR::MORTAR::algorithm_lts)
      {
        std::cout << "================================================================"
                  << std::endl;
        std::cout << "===== Line-To-Segment approach ================================="
                  << std::endl;
        std::cout << "================================================================\n"
                  << std::endl;
      }
      else if (algorithm == INPAR::MORTAR::algorithm_ltl)
      {
        std::cout << "================================================================"
                  << std::endl;
        std::cout << "===== Line-To-Line approach ===================================="
                  << std::endl;
        std::cout << "================================================================\n"
                  << std::endl;
      }
      else if (algorithm == INPAR::MORTAR::algorithm_stl)
      {
        std::cout << "================================================================"
                  << std::endl;
        std::cout << "===== Segment-To-Line approach ================================="
                  << std::endl;
        std::cout << "================================================================\n"
                  << std::endl;
      }
      else if (algorithm == INPAR::MORTAR::algorithm_gpts)
      {
        std::cout << "================================================================"
                  << std::endl;
        std::cout << "===== Gauss-Point-To-Segment approach =========================="
                  << std::endl;
        std::cout << "================================================================\n"
                  << std::endl;
      }
      // invalid system type
      else
        dserror("Invalid system type for contact/meshtying");
    }
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void STR::TimInt::ApplyMeshInitialization(Teuchos::RCP<const Epetra_Vector> Xslavemod)
{
  // check modified positions vector
  if (Xslavemod == Teuchos::null) return;

  // create fully overlapping slave node map
  Teuchos::RCP<Epetra_Map> slavemap = cmtbridge_->MtManager()->GetStrategy().SlaveRowNodes();
  Teuchos::RCP<Epetra_Map> allreduceslavemap = CORE::LINALG::AllreduceEMap(*slavemap);

  // export modified node positions to column map of problem discretization
  Teuchos::RCP<Epetra_Vector> Xslavemodcol =
      CORE::LINALG::CreateVector(*discret_->DofColMap(), false);
  CORE::LINALG::Export(*Xslavemod, *Xslavemodcol);

  const int numnode = allreduceslavemap->NumMyElements();
  const int numdim = DRT::Problem::Instance()->NDim();
  const Epetra_Vector& gvector = *Xslavemodcol;

  // loop over all slave nodes (for all procs)
  for (int index = 0; index < numnode; ++index)
  {
    int gid = allreduceslavemap->GID(index);

    // only do someting for nodes in my column map
    int ilid = discret_->NodeColMap()->LID(gid);
    if (ilid < 0) continue;

    DRT::Node* mynode = discret_->gNode(gid);

    // get degrees of freedom associated with this fluid/structure node
    std::vector<int> nodedofs = discret_->Dof(0, mynode);
    std::vector<double> nvector(3, 0.0);

    // create new position vector
    for (int i = 0; i < numdim; ++i)
    {
      const int lid = gvector.Map().LID(nodedofs[i]);

      if (lid < 0)
        dserror(
            "Proc %d: Cannot find gid=%d in Epetra_Vector", gvector.Comm().MyPID(), nodedofs[i]);

      nvector[i] += gvector[lid];
    }

    // set new reference position
    mynode->SetPos(nvector);
  }

  // re-initialize finite elements
  CORE::COMM::ParObjectFactory::Instance().InitializeElements(*discret_);
}

/*----------------------------------------------------------------------*/
/* Prepare contact for new time step */
void STR::TimInt::PrepareStepContact()
{
  // just do something here if contact is present
  if (HaveContactMeshtying())
  {
    if (cmtbridge_->HaveContact())
    {
      cmtbridge_->GetStrategy().Inttime_init();
      cmtbridge_->GetStrategy().RedistributeContact((*dis_)(0), (*vel_)(0));
    }
  }
}

/*----------------------------------------------------------------------*/
/* things that should be done after the actual time loop is finished */
void STR::TimInt::PostTimeLoop()
{
  if (HaveMicroMat())
  {
    // stop supporting processors in multi scale simulations
    STRUMULTI::stop_np_multiscale();
  }
}

/*----------------------------------------------------------------------*/
/* equilibrate system at initial state
 * and identify consistent accelerations */
void STR::TimInt::DetermineMassDampConsistAccel()
{
  // temporary right hand sinde vector in this routing
  Teuchos::RCP<Epetra_Vector> rhs =
      CORE::LINALG::CreateVector(*DofRowMapView(), true);  // right hand side
  // temporary force vectors in this routine
  Teuchos::RCP<Epetra_Vector> fext =
      CORE::LINALG::CreateVector(*DofRowMapView(), true);  // external force
  Teuchos::RCP<Epetra_Vector> fint =
      CORE::LINALG::CreateVector(*DofRowMapView(), true);  // internal force

  // initialise matrices
  stiff_->Zero();
  mass_->Zero();

  // auxiliary vector in order to store accelerations of inhomogeneous Dirichilet-DoFs
  // Meier 2015: This contribution is necessary in order to determine correct initial
  // accelerations in case of inhomogeneous Dirichlet conditions
  Teuchos::RCP<Epetra_Vector> acc_aux = CORE::LINALG::CreateVector(*DofRowMapView(), true);
  acc_aux->PutScalar(0.0);

  // overwrite initial state vectors with DirichletBCs
  ApplyDirichletBC((*time_)[0], (*dis_)(0), (*vel_)(0), acc_aux, false);

  /* get external force (no linearization since we assume Rayleigh damping
   * to be independent of follower loads) */
  ApplyForceExternal((*time_)[0], (*dis_)(0), disn_, (*vel_)(0), fext);

  // get initial internal force and stiffness and mass
  {
    // compute new inner radius
    discret_->ClearState();
    discret_->SetState(0, "displacement", (*dis_)(0));

    // for structure ale
    if (dismat_ != Teuchos::null) discret_->SetState(0, "material_displacement", (*dismat_)(0));

    // create the parameters for the discretization
    Teuchos::ParameterList p;

    // action for elements
    if (lumpmass_ == false) p.set("action", "calc_struct_nlnstiffmass");
    // lumping the mass matrix
    else
      p.set("action", "calc_struct_nlnstifflmass");
    // other parameters that might be needed by the elements
    p.set("total time", (*time_)[0]);
    p.set("delta time", (*dt_)[0]);
    if (pressure_ != Teuchos::null) p.set("volume", 0.0);
    if (fresn_str_ != Teuchos::null)
    {
      p.set<int>("MyPID", myrank_);
      p.set<double>("cond_rhs_norm", 0.);
    }

    // set vector values needed by elements
    discret_->ClearState();
    // extended SetState(0,...) in case of multiple dofsets (e.g. TSI)
    discret_->SetState(0, "residual displacement", zeros_);
    discret_->SetState(0, "displacement", (*dis_)(0));
    discret_->SetState(0, "velocity", (*vel_)(0));

    // The acceleration is only used as a dummy here and should not be applied inside an element,
    // since this is not the consistent initial acceleration vector which will be determined later
    // on
    discret_->SetState(0, "acceleration", acc_aux);

    if (damping_ == INPAR::STR::damp_material) discret_->SetState(0, "velocity", (*vel_)(0));

    // for structure ale
    if (dismat_ != Teuchos::null) discret_->SetState(0, "material_displacement", (*dismat_)(0));

    discret_->Evaluate(p, stiff_, mass_, fint, Teuchos::null, fintn_str_);
    discret_->ClearState();
  }

  // finish mass matrix
  mass_->Complete();

  // close stiffness matrix
  stiff_->Complete();

  // build Rayleigh damping matrix if desired
  if (damping_ == INPAR::STR::damp_rayleigh)
  {
    damp_->Add(*stiff_, false, dampk_, 0.0);
    damp_->Add(*mass_, false, dampm_, 1.0);
    damp_->Complete();
  }

  // in case of C0 pressure field, we need to get rid of
  // pressure equations
  Teuchos::RCP<CORE::LINALG::SparseOperator> mass = Teuchos::null;
  // Meier 2015: Here, we apply a deep copy in order to not perform the Dirichlet conditions on the
  // constant matrix mass_ later on. This is necessary since we need the original mass matrix mass_
  // (without blanked rows) on the Dirichlet DoFs in order to calculate correct reaction forces
  // (Christoph Meier)
  mass = Teuchos::rcp(new CORE::LINALG::SparseMatrix(*MassMatrix(), CORE::LINALG::Copy));

  /* calculate consistent initial accelerations
   * WE MISS:
   *   - surface stress forces
   *   - potential forces
   *   - linearization of follower loads
   */
  {
    // Contribution to rhs due to damping forces
    if (damping_ == INPAR::STR::damp_rayleigh)
    {
      damp_->Multiply(false, (*vel_)[0], *rhs);
    }

    // add initial forces due to 0D cardiovascular for consistent initial acceleration calculation!
    // needed in case of initial ventricular pressures != 0
    Teuchos::ParameterList pwindk;
    if (cardvasc0dman_->HaveCardiovascular0D())
    {
      pwindk.set("scale_timint", 1.0);
      pwindk.set("time_step_size", (*dt_)[0]);
      cardvasc0dman_->EvaluateForceStiff((*time_)[0], (*dis_)(0), fint, stiff_, pwindk);
    }

    // Contribution to rhs due to internal and external forces
    rhs->Update(-1.0, *fint, 1.0, *fext, -1.0);

    // Contribution to rhs due to beam contact
    if (HaveBeamContact())
    {
      // create empty parameter list
      Teuchos::ParameterList beamcontactparams;
      beamcontactparams.set("iter", 0);
      beamcontactparams.set("dt", (*dt_)[0]);
      beamcontactparams.set("numstep", step_);
      beamcman_->Evaluate(*SystemMatrix(), *rhs, (*dis_)[0], beamcontactparams, true, timen_);
    }

    // Contribution to rhs due to inertia forces of inhomogeneous Dirichlet conditions
    Teuchos::RCP<Epetra_Vector> finert0 = CORE::LINALG::CreateVector(*DofRowMapView(), true);
    finert0->PutScalar(0.0);
    mass_->Multiply(false, *acc_aux, *finert0);
    rhs->Update(-1.0, *finert0, 1.0);

    // blank RHS and system matrix on DBC DOFs
    dbcmaps_->InsertCondVector(dbcmaps_->ExtractCondVector(zeros_), rhs);

    // Apply Dirichlet conditions also to mass matrix (which represents the system matrix of
    // the considered linear system of equations)
    mass->ApplyDirichlet(*(dbcmaps_->CondMap()));

    if (pressure_ != Teuchos::null)
    {
      pressure_->InsertCondVector(pressure_->ExtractCondVector(zeros_), rhs);
      mass->ApplyDirichlet(*(pressure_->CondMap()));
    }
    if (porositysplitter_ != Teuchos::null)
    {
      porositysplitter_->InsertCondVector(porositysplitter_->ExtractCondVector(zeros_), rhs);
      mass->ApplyDirichlet(*(porositysplitter_->CondMap()));
    }

    // Meier 2015: Due to the Dirichlet conditions applied to the mass matrix, we solely solve
    // for the accelerations at non-Dirichlet DoFs while the resulting accelerations at the
    // Dirichlet-DoFs will be zero. Therefore, the accelerations at DoFs with inhomogeneous
    // Dirichlet conditions will be added below at *).
    CORE::LINALG::SolverParams solver_params;
    solver_params.refactor = true;
    solver_params.reset = true;
    solver_->Solve(mass->EpetraOperator(), (*acc_)(0), rhs, solver_params);

    //*) Add contributions of inhomogeneous DBCs
    (*acc_)(0)->Update(1.0, *acc_aux, 1.0);
  }

  // We need to reset the stiffness matrix because its graph (topology)
  // is not finished yet in case of constraints and possibly other side
  // effects (basically managers).
  stiff_->Reset();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void STR::TimInt::DetermineMass()
{
  dserror(
      "(Re-)Evaluation of only the mass matrix and intertial forces is "
      "not implemented in the base class.\n Set 'MASSLIN' to 'No' in "
      "--STRUCTURAL DYNAMIC if you want to use the chosen timint scheme.");
}

/*---------------------------------------------------------------*/
/* Apply Dirichlet boundary conditions on provided state vectors */
void STR::TimInt::ApplyDirichletBC(const double time, Teuchos::RCP<Epetra_Vector> dis,
    Teuchos::RCP<Epetra_Vector> vel, Teuchos::RCP<Epetra_Vector> acc, bool recreatemap)
{
  // In the case of local coordinate systems, we have to rotate forward ...
  // --------------------------------------------------------------------------------
  if (locsysman_ != Teuchos::null)
  {
    if (dis != Teuchos::null) locsysman_->RotateGlobalToLocal(dis, true);
    if (vel != Teuchos::null) locsysman_->RotateGlobalToLocal(vel);
    if (acc != Teuchos::null) locsysman_->RotateGlobalToLocal(acc);
  }

  // Apply DBCs
  // --------------------------------------------------------------------------------
  // needed parameters
  Teuchos::ParameterList p;
  p.set("total time", time);  // target time

  // predicted Dirichlet values
  // \c dis then also holds prescribed new Dirichlet displacements
  discret_->ClearState();
  if (recreatemap)
  {
    discret_->EvaluateDirichlet(p, dis, vel, acc, Teuchos::null, dbcmaps_);
  }
  else
  {
    discret_->EvaluateDirichlet(p, dis, vel, acc, Teuchos::null, Teuchos::null);
  }
  discret_->ClearState();

  // In the case of local coordinate systems, we have to rotate back into global Cartesian frame
  // --------------------------------------------------------------------------------
  if (locsysman_ != Teuchos::null)
  {
    if (dis != Teuchos::null) locsysman_->RotateLocalToGlobal(dis, true);
    if (vel != Teuchos::null) locsysman_->RotateLocalToGlobal(vel);
    if (acc != Teuchos::null) locsysman_->RotateLocalToGlobal(acc);
  }
}

/*----------------------------------------------------------------------*/
/* Update time and step counter */
void STR::TimInt::UpdateStepTime()
{
  // update time and step
  time_->UpdateSteps(timen_);  // t_{n} := t_{n+1}, etc
  step_ = stepn_;              // n := n+1
  //
  timen_ += (*dt_)[0];
  stepn_ += 1;
}

/*----------------------------------------------------------------------*/
/* Update contact and meshtying */
void STR::TimInt::UpdateStepContactMeshtying()
{
  if (HaveContactMeshtying())
  {
    cmtbridge_->Update(disn_);
#ifdef MORTARGMSH1
    bool gmsh = INPUT::IntegralValue<int>(DRT::Problem::Instance()->IOParams(), "OUTPUT_GMSH");
    if (gmsh) cmtbridge_->VisualizeGmsh(stepn_);
#endif  // #ifdef MORTARGMSH1
  }
}

/*----------------------------------------------------------------------*/
/* Update beam contact */
void STR::TimInt::UpdateStepBeamContact()
{
  if (HaveBeamContact()) beamcman_->Update(*disn_, stepn_, 99);
}

/*----------------------------------------------------------------------*/
/* Velocity update method (VUM) for contact */
void STR::TimInt::UpdateStepContactVUM()
{
  if (HaveContactMeshtying())
  {
    bool do_vum = INPUT::IntegralValue<int>(cmtbridge_->GetStrategy().Params(), "VELOCITY_UPDATE");

    //********************************************************************
    // VELOCITY UPDATE METHOD
    //********************************************************************
    if (do_vum)
    {
      // check for actual contact and leave if active set empty
      bool isincontact = cmtbridge_->GetStrategy().IsInContact();
      if (!isincontact) return;

      // check for contact force evaluation
      bool do_end =
          INPUT::IntegralValue<int>(cmtbridge_->GetStrategy().Params(), "CONTACTFORCE_ENDTIME");
      if (do_end == false)
      {
        dserror(
            "***** WARNING: VelUpdate ONLY for contact force evaluated at the end time -> skipping "
            "****");
        return;
      }

      // parameter list
      const Teuchos::ParameterList& sdynparams =
          DRT::Problem::Instance()->StructuralDynamicParams();

      // time integration parameter
      double alpham = 0.0;
      double beta = 0.0;
      double gamma = 0.0;
      if (INPUT::IntegralValue<INPAR::STR::DynamicType>(sdynparams, "DYNAMICTYP") ==
          INPAR::STR::dyna_genalpha)
      {
        auto genAlpha = dynamic_cast<STR::TimIntGenAlpha*>(this);
        alpham = genAlpha->TimIntParamAlpham();
        beta = genAlpha->TimIntParamBeta();
        gamma = genAlpha->TimIntParamGamma();
      }
      else if (INPUT::IntegralValue<INPAR::STR::DynamicType>(sdynparams, "DYNAMICTYP") ==
               INPAR::STR::dyna_gemm)
      {
        alpham = sdynparams.sublist("GEMM").get<double>("ALPHA_M");
        beta = sdynparams.sublist("GEMM").get<double>("BETA");
        gamma = sdynparams.sublist("GEMM").get<double>("GAMMA");
      }
      else
      {
        dserror("***** WARNING: VelUpdate ONLY for Gen-alpha and GEMM -> skipping ****");
        return;
      }

      // the four velocity update constants
      double R1 = 2 * (alpham - 1) / (gamma * (*dt_)[0]);
      double R2 = (1 - alpham) / gamma;
      double R3 = (*dt_)[0] * (1 - 2 * beta - alpham) / (2 * gamma);
      double R4 = beta * (alpham - 1) / pow(gamma, 2);

      // maps
      const Epetra_Map* dofmap = discret_->DofRowMap();
      Teuchos::RCP<Epetra_Map> activenodemap = cmtbridge_->GetStrategy().ActiveRowNodes();
      Teuchos::RCP<Epetra_Map> slavenodemap = cmtbridge_->GetStrategy().SlaveRowNodes();
      Teuchos::RCP<Epetra_Map> notredistslavedofmap =
          cmtbridge_->GetStrategy().NotReDistSlaveRowDofs();
      Teuchos::RCP<Epetra_Map> notredistmasterdofmap =
          cmtbridge_->GetStrategy().NotReDistMasterRowDofs();
      Teuchos::RCP<Epetra_Map> notactivenodemap =
          CORE::LINALG::SplitMap(*slavenodemap, *activenodemap);

      // the lumped mass matrix and its inverse
      if (lumpmass_ == false)
      {
        dserror("***** WARNING: VelUpdate ONLY for lumped mass matrix -> skipping ****");
        return;
      }
      Teuchos::RCP<CORE::LINALG::SparseMatrix> Mass =
          Teuchos::rcp_dynamic_cast<CORE::LINALG::SparseMatrix>(mass_);
      Teuchos::RCP<CORE::LINALG::SparseMatrix> Minv =
          Teuchos::rcp(new CORE::LINALG::SparseMatrix(*Mass));
      Teuchos::RCP<Epetra_Vector> diag = CORE::LINALG::CreateVector(*dofmap, true);
      int err = 0;
      Minv->ExtractDiagonalCopy(*diag);
      err = diag->Reciprocal(*diag);
      if (err != 0) dserror("Reciprocal: Zero diagonal entry!");
      err = Minv->ReplaceDiagonalValues(*diag);
      Minv->Complete(*dofmap, *dofmap);

      // displacement increment Dd
      Teuchos::RCP<Epetra_Vector> Dd = CORE::LINALG::CreateVector(*dofmap, true);
      Dd->Update(1.0, *disn_, 0.0);
      Dd->Update(-1.0, (*dis_)[0], 1.0);

      // mortar operator Bc
      Teuchos::RCP<CORE::LINALG::SparseMatrix> Mmat = cmtbridge_->GetStrategy().MMatrix();
      Teuchos::RCP<CORE::LINALG::SparseMatrix> Dmat = cmtbridge_->GetStrategy().DMatrix();
      Teuchos::RCP<Epetra_Map> slavedofmap = Teuchos::rcp(new Epetra_Map(Dmat->RangeMap()));
      Teuchos::RCP<CORE::LINALG::SparseMatrix> Bc =
          Teuchos::rcp(new CORE::LINALG::SparseMatrix(*dofmap, 10));
      Teuchos::RCP<CORE::LINALG::SparseMatrix> M =
          Teuchos::rcp(new CORE::LINALG::SparseMatrix(*slavedofmap, 10));
      Teuchos::RCP<CORE::LINALG::SparseMatrix> D =
          Teuchos::rcp(new CORE::LINALG::SparseMatrix(*slavedofmap, 10));
      if (Teuchos::getIntegralValue<INPAR::MORTAR::ParallelRedist>(
              cmtbridge_->GetStrategy().Params().sublist("PARALLEL REDISTRIBUTION"),
              "PARALLEL_REDIST") != INPAR::MORTAR::ParallelRedist::redist_none)
      {
        M = MORTAR::MatrixColTransform(Mmat, notredistmasterdofmap);
        D = MORTAR::MatrixColTransform(Dmat, notredistslavedofmap);
      }
      else
      {
        M = Mmat;
        D = Dmat;
      }
      Bc->Add(*M, true, -1.0, 1.0);
      Bc->Add(*D, true, 1.0, 1.0);
      Bc->Complete(*slavedofmap, *dofmap);
      Bc->ApplyDirichlet(*(dbcmaps_->CondMap()), false);

      // matrix of the normal vectors
      Teuchos::RCP<CORE::LINALG::SparseMatrix> N = cmtbridge_->GetStrategy().EvaluateNormals(disn_);

      // lagrange multiplier z
      Teuchos::RCP<Epetra_Vector> LM = cmtbridge_->GetStrategy().LagrMult();
      Teuchos::RCP<Epetra_Vector> Z = CORE::LINALG::CreateVector(*slavenodemap, true);
      Teuchos::RCP<Epetra_Vector> z = CORE::LINALG::CreateVector(*activenodemap, true);
      N->Multiply(false, *LM, *Z);
      CORE::LINALG::Export(*Z, *z);

      // auxiliary operator BN = Bc * N
      Teuchos::RCP<CORE::LINALG::SparseMatrix> BN =
          CORE::LINALG::MLMultiply(*Bc, false, *N, true, false, false, true);

      // operator A
      Teuchos::RCP<CORE::LINALG::SparseMatrix> tempmtx1;
      Teuchos::RCP<CORE::LINALG::SparseMatrix> tempmtx2;
      Teuchos::RCP<CORE::LINALG::SparseMatrix> tempmtx3;
      Teuchos::RCP<CORE::LINALG::SparseMatrix> A;
      Teuchos::RCP<CORE::LINALG::SparseMatrix> Atemp1 =
          CORE::LINALG::MLMultiply(*BN, true, *Minv, false, false, false, true);
      Teuchos::RCP<CORE::LINALG::SparseMatrix> Atemp2 =
          CORE::LINALG::MLMultiply(*Atemp1, false, *BN, false, false, false, true);
      Atemp2->Scale(R4);
      CORE::LINALG::SplitMatrix2x2(Atemp2, notactivenodemap, activenodemap, notactivenodemap,
          activenodemap, tempmtx1, tempmtx2, tempmtx3, A);
      A->Complete(*activenodemap, *activenodemap);

      // diagonal of A
      Teuchos::RCP<Epetra_Vector> AD = CORE::LINALG::CreateVector(*activenodemap, true);
      A->ExtractDiagonalCopy(*AD);

      // operator b
      Teuchos::RCP<Epetra_Vector> btemp1 = CORE::LINALG::CreateVector(*dofmap, true);
      Teuchos::RCP<Epetra_Vector> btemp2 = CORE::LINALG::CreateVector(*slavenodemap, true);
      Teuchos::RCP<Epetra_Vector> b = CORE::LINALG::CreateVector(*activenodemap, true);
      btemp1->Update(R1, *Dd, 0.0);
      btemp1->Update(R2, (*vel_)[0], 1.0);
      btemp1->Update(R3, (*acc_)[0], 1.0);
      BN->Multiply(true, *btemp1, *btemp2);
      CORE::LINALG::Export(*btemp2, *b);

      // operatior c
      Teuchos::RCP<Epetra_Vector> ctemp = CORE::LINALG::CreateVector(*slavenodemap, true);
      Teuchos::RCP<Epetra_Vector> c = CORE::LINALG::CreateVector(*activenodemap, true);
      BN->Multiply(true, *Dd, *ctemp);
      CORE::LINALG::Export(*ctemp, *c);

      // contact work wc
      Teuchos::RCP<Epetra_Vector> wc = CORE::LINALG::CreateVector(*activenodemap, true);
      wc->Multiply(1.0, *c, *z, 0.0);

      // gain and loss of energy
      double gain = 0;
      double loss = 0;
      Teuchos::RCP<Epetra_Vector> wp = CORE::LINALG::CreateVector(*activenodemap, true);
      Teuchos::RCP<Epetra_Vector> wn = CORE::LINALG::CreateVector(*activenodemap, true);
      Teuchos::RCP<Epetra_Vector> wd = CORE::LINALG::CreateVector(*activenodemap, true);
      Teuchos::RCP<Epetra_Vector> wt = CORE::LINALG::CreateVector(*activenodemap, true);
      for (int i = 0; i < activenodemap->NumMyElements(); ++i)
      {
        if ((*wc)[i] > 0)
        {
          (*wp)[i] = (*wc)[i];
          (*wn)[i] = 0;
          (*wd)[i] = 0;
          (*wt)[i] = 0;
        }
        else
        {
          (*wp)[i] = 0;
          (*wn)[i] = (*wc)[i];
          (*wd)[i] = pow((*b)[i], 2) / (4 * (*AD)[i]);
          if ((*wc)[i] > (*wd)[i])
          {
            (*wt)[i] = (*wc)[i];
          }
          else
          {
            (*wt)[i] = (*wd)[i];
          }
        }
      }
      wp->Norm1(&loss);
      wn->Norm1(&gain);

      // manipulated contact work w
      double tolerance = 0.01;
      Teuchos::RCP<Epetra_Vector> wtemp1 = CORE::LINALG::CreateVector(*activenodemap, true);
      Teuchos::RCP<Epetra_Vector> wtemp2 = CORE::LINALG::CreateVector(*activenodemap, true);
      Teuchos::RCP<Epetra_Vector> w = CORE::LINALG::CreateVector(*activenodemap, true);
      if (abs(gain - loss) < 1.0e-8)
      {
        return;
      }
      else if (gain > loss)
      {
        double C = (gain - loss) / gain;
        wtemp1->Update(C, *wn, 0.0);
        for (int i = 0; i < activenodemap->NumMyElements(); ++i)
        {
          (*wtemp2)[i] = pow((*b)[i], 2) / (4 * (*AD)[i]);
          if ((*wtemp1)[i] > (*wtemp2)[i])
          {
            (*w)[i] = (*wtemp1)[i];
          }
          else
          {
            (*w)[i] = (1 - tolerance) * (*wtemp2)[i];
            std::cout << "***** WARNING: VelUpdate is not able to compensate the gain of energy****"
                      << std::endl;
          }
        }
      }
      else
      {
        double C = (loss - gain) / loss;
        w->Update(C, *wp, 0.0);
      }

      // (1) initial solution p_0
      Teuchos::RCP<Epetra_Vector> p1 = CORE::LINALG::CreateVector(*activenodemap, true);
      Teuchos::RCP<Epetra_Vector> p2 = CORE::LINALG::CreateVector(*activenodemap, true);
      Teuchos::RCP<Epetra_Vector> p = CORE::LINALG::CreateVector(*activenodemap, true);
      if (gain > loss)
      {
        for (int i = 0; i < activenodemap->NumMyElements(); ++i)
        {
          (*p1)[i] =
              (-(*b)[i] + pow(pow((*b)[i], 2) - 4 * (*AD)[i] * (*w)[i], 0.5)) / (2 * (*AD)[i]);

          (*p2)[i] =
              (-(*b)[i] - pow(pow((*b)[i], 2) - 4 * (*AD)[i] * (*w)[i], 0.5)) / (2 * (*AD)[i]);
          if ((*w)[i] == 0)
            (*p)[i] = 0;
          else if (abs((*p1)[i]) < abs((*p2)[i]))
            (*p)[i] = (*p1)[i];
          else
            (*p)[i] = (*p2)[i];
        }
      }
      else
      {
        for (int i = 0; i < activenodemap->NumMyElements(); ++i)
        {
          (*p1)[i] =
              (-(*b)[i] + pow(pow((*b)[i], 2) - 4 * (*AD)[i] * (*w)[i], 0.5)) / (2 * (*AD)[i]);

          (*p2)[i] =
              (-(*b)[i] - pow(pow((*b)[i], 2) - 4 * (*AD)[i] * (*w)[i], 0.5)) / (2 * (*AD)[i]);
          if ((*w)[i] == 0)
            (*p)[i] = 0;
          else if (((*p1)[i] > 0) == ((*b)[i] < 0))
            (*p)[i] = (*p1)[i];
          else
            (*p)[i] = (*p2)[i];
        }
      }

      // (2) initial residual f_0, |f_0|, DF_0
      Teuchos::RCP<Epetra_Vector> x = CORE::LINALG::CreateVector(*activenodemap, true);
      Teuchos::RCP<Epetra_Vector> f = CORE::LINALG::CreateVector(*activenodemap, true);
      int NumEntries = 0;
      int* Indices = nullptr;
      double* Values = nullptr;
      double res = 1.0;
      double initres = 1.0;
      double dfik = 0;
      Teuchos::RCP<CORE::LINALG::SparseMatrix> DF =
          Teuchos::rcp(new CORE::LINALG::SparseMatrix(*activenodemap, 10));

      // rhs f
      for (int i = 0; i < activenodemap->NumMyElements(); ++i)
      {
        x->PutScalar(0.0);
        (A->EpetraMatrix())->ExtractMyRowView(i, NumEntries, Values, Indices);
        x->ReplaceMyValues(NumEntries, Values, Indices);
        (*f)[i] = (*b)[i] * (*p)[i] + (*w)[i];
        for (int j = 0; j < activenodemap->NumMyElements(); ++j)
        {
          (*f)[i] += (*x)[j] * (*p)[i] * (*p)[j];
        }
      }

      // residual res
      f->Norm2(&initres);
      res = initres;

      // jacobian DF
      for (int i = 0; i < activenodemap->NumMyElements(); ++i)
      {
        x->PutScalar(0.0);
        (A->EpetraMatrix())->ExtractMyRowView(i, NumEntries, Values, Indices);
        x->ReplaceMyValues(NumEntries, Values, Indices);
        for (int k = 0; k < activenodemap->NumMyElements(); ++k)
        {
          if (k == i)
          {
            dfik = (*x)[i] * (*p)[i] + (*b)[i];
            for (int j = 0; j < activenodemap->NumMyElements(); ++j)
            {
              dfik += (*x)[j] * (*p)[j];
            }
            DF->Assemble(dfik, activenodemap->GID(i), activenodemap->GID(k));
          }
          else
            DF->Assemble((*x)[k] * (*p)[i], activenodemap->GID(i), activenodemap->GID(k));
        }
      }
      DF->Complete(*activenodemap, *activenodemap);

      // (3) Newton-Iteration
      Teuchos::RCP<Epetra_Vector> mf = CORE::LINALG::CreateVector(*activenodemap, true);
      Teuchos::RCP<Epetra_Vector> dp = CORE::LINALG::CreateVector(*activenodemap, true);
      double tol = 0.00000001;
      double numiter = 0;
      double stopcrit = 100;

      while (res > tol)
      {
        // solver for linear equations DF * dp = -f
        mf->Update(-1.0, *f, 0.0);
        CORE::LINALG::SolverParams solver_params;
        solver_params.refactor = true;
        solver_->Solve(DF->EpetraOperator(), dp, mf, solver_params);

        // Update solution p_n = p_n-1 + dp
        p->Update(1.0, *dp, 1.0);

        // rhs f
        for (int i = 0; i < activenodemap->NumMyElements(); ++i)
        {
          x->PutScalar(0.0);
          (A->EpetraMatrix())->ExtractMyRowView(i, NumEntries, Values, Indices);
          x->ReplaceMyValues(NumEntries, Values, Indices);
          (*f)[i] = (*b)[i] * (*p)[i] + (*w)[i];
          for (int j = 0; j < activenodemap->NumMyElements(); ++j)
          {
            (*f)[i] += (*x)[j] * (*p)[i] * (*p)[j];
          }
        }

        // residual res
        f->Norm2(&res);
        res /= initres;

        // jacobian DF
        DF->PutScalar(0.0);
        for (int i = 0; i < activenodemap->NumMyElements(); ++i)
        {
          x->PutScalar(0.0);
          (A->EpetraMatrix())->ExtractMyRowView(i, NumEntries, Values, Indices);
          x->ReplaceMyValues(NumEntries, Values, Indices);
          for (int k = 0; k < activenodemap->NumMyElements(); ++k)
          {
            if (k == i)
            {
              dfik = (*x)[i] * (*p)[i] + (*b)[i];
              for (int j = 0; j < activenodemap->NumMyElements(); ++j)
              {
                dfik += (*x)[j] * (*p)[j];
              }
              DF->Assemble(dfik, activenodemap->GID(i), activenodemap->GID(k));
            }
            else
              DF->Assemble((*x)[k] * (*p)[i], activenodemap->GID(i), activenodemap->GID(k));
          }
        }

        // stop criteria
        numiter += 1;
        if (numiter == stopcrit)
        {
          std::cout << "***** WARNING: VelUpdate is not able to converge -> skipping ****"
                    << std::endl;
          return;
        }
      }

      // (4) VelocityUpdate
      Teuchos::RCP<Epetra_Vector> ptemp1 = CORE::LINALG::CreateVector(*slavenodemap, true);
      Teuchos::RCP<Epetra_Vector> ptemp2 = CORE::LINALG::CreateVector(*dofmap, true);
      Teuchos::RCP<Epetra_Vector> VU = CORE::LINALG::CreateVector(*dofmap, true);
      CORE::LINALG::Export(*p, *ptemp1);
      BN->Multiply(false, *ptemp1, *ptemp2);
      Minv->Multiply(false, *ptemp2, *VU);
      veln_->Update(1.0, *VU, 1.0);
    }
  }
}

/*----------------------------------------------------------------------*/
/* Reset configuration after time step */
void STR::TimInt::ResetStep()
{
  // reset state vectors
  disn_->Update(1.0, (*dis_)[0], 0.0);
  if (dismatn_ != Teuchos::null) dismatn_->Update(1.0, (*dismat_)[0], 0.0);
  veln_->Update(1.0, (*vel_)[0], 0.0);
  accn_->Update(1.0, (*acc_)[0], 0.0);

  // reset anything that needs to be reset at the element level
  {
    // create the parameters for the discretization
    Teuchos::ParameterList p;
    p.set("action", "calc_struct_reset_istep");
    // go to elements
    discret_->Evaluate(
        p, Teuchos::null, Teuchos::null, Teuchos::null, Teuchos::null, Teuchos::null);
    discret_->ClearState();
  }

  // reset 0D cardiovascular model if we have monolithic 0D cardiovascular-structure coupling (mhv
  // 02/2015)
  if (cardvasc0dman_->HaveCardiovascular0D()) cardvasc0dman_->ResetStep();
}

/*----------------------------------------------------------------------*/
/* Read and set restart values */
void STR::TimInt::ReadRestart(const int step)
{
  IO::DiscretizationReader reader(discret_, step);
  if (step != reader.ReadInt("step")) dserror("Time step on file not equal to given step");

  step_ = step;
  stepn_ = step_ + 1;
  time_ = Teuchos::rcp(new TIMESTEPPING::TimIntMStep<double>(0, 0, reader.ReadDouble("time")));
  timen_ = (*time_)[0] + (*dt_)[0];

  ReadRestartState();

  ReadRestartConstraint();
  ReadRestartCardiovascular0D();
  ReadRestartContactMeshtying();
  ReadRestartBeamContact();
  ReadRestartMultiScale();
  ReadRestartSpringDashpot();

  ReadRestartForce();
}

/*----------------------------------------------------------------------*/
/* Set restart values passed down from above */
void STR::TimInt::SetRestart(int step, double time, Teuchos::RCP<Epetra_Vector> disn,
    Teuchos::RCP<Epetra_Vector> veln, Teuchos::RCP<Epetra_Vector> accn,
    Teuchos::RCP<std::vector<char>> elementdata, Teuchos::RCP<std::vector<char>> nodedata)
{
  step_ = step;
  stepn_ = step_ + 1;
  time_ = Teuchos::rcp(new TIMESTEPPING::TimIntMStep<double>(0, 0, time));
  timen_ = (*time_)[0] + (*dt_)[0];

  SetRestartState(disn, veln, accn, elementdata, nodedata);

  // ---------------------------------------------------------------------------
  // set restart is only for simple structure problems
  // hence we put some security measures in place

  // constraints
  if (conman_->HaveConstraint()) dserror("Set restart not implemented for constraints");

  // Cardiovascular0D
  if (cardvasc0dman_->HaveCardiovascular0D())
    dserror("Set restart not implemented for Cardiovascular0D");

  // contact / meshtying
  if (HaveContactMeshtying()) dserror("Set restart not implemented for contact / meshtying");

  // beam contact
  if (HaveBeamContact()) dserror("Set restart not implemented for beam contact");

  // biofilm growth
  if (HaveBiofilmGrowth()) dserror("Set restart not implemented for biofilm growth");
}

/*----------------------------------------------------------------------*/
/* Read and set restart state */
void STR::TimInt::ReadRestartState()
{
  IO::DiscretizationReader reader(discret_, step_);

  reader.ReadVector(disn_, "displacement");
  dis_->UpdateSteps(*disn_);

  if ((dismatn_ != Teuchos::null))
  {
    reader.ReadVector(dismatn_, "material_displacement");
    dismat_->UpdateSteps(*dismatn_);
  }

  reader.ReadVector(veln_, "velocity");
  vel_->UpdateSteps(*veln_);
  reader.ReadVector(accn_, "acceleration");
  acc_->UpdateSteps(*accn_);
  reader.ReadHistoryData(step_);
}

/*----------------------------------------------------------------------*/
/* Read and set restart state */
void STR::TimInt::SetRestartState(Teuchos::RCP<Epetra_Vector> disn,
    Teuchos::RCP<Epetra_Vector> veln, Teuchos::RCP<Epetra_Vector> accn,
    Teuchos::RCP<std::vector<char>> elementdata, Teuchos::RCP<std::vector<char>> nodedata

)
{
  dis_->UpdateSteps(*disn);
  vel_->UpdateSteps(*veln);
  acc_->UpdateSteps(*accn);

  // the following is copied from Readmesh()
  // before we unpack nodes/elements we store a copy of the nodal row/col map
  Teuchos::RCP<Epetra_Map> noderowmap = Teuchos::rcp(new Epetra_Map(*discret_->NodeRowMap()));
  Teuchos::RCP<Epetra_Map> nodecolmap = Teuchos::rcp(new Epetra_Map(*discret_->NodeColMap()));

  // unpack nodes and elements
  // so everything should be OK
  discret_->UnPackMyNodes(nodedata);
  discret_->UnPackMyElements(elementdata);
  discret_->Redistribute(*noderowmap, *nodecolmap);
}
/*----------------------------------------------------------------------*/
/* Read and set restart values for constraints */
void STR::TimInt::ReadRestartConstraint()
{
  if (conman_->HaveConstraint())
  {
    IO::DiscretizationReader reader(discret_, step_);
    double uzawatemp = reader.ReadDouble("uzawaparameter");
    consolv_->SetUzawaParameter(uzawatemp);

    conman_->ReadRestart(reader, (*time_)[0]);
  }
}

/*----------------------------------------------------------------------*/
/* Read and set restart values for 0D cardiovascular models */
void STR::TimInt::ReadRestartCardiovascular0D()
{
  if (cardvasc0dman_->HaveCardiovascular0D())
  {
    IO::DiscretizationReader reader(discret_, step_);
    cardvasc0dman_->ReadRestart(reader, (*time_)[0]);
  }
}

/*----------------------------------------------------------------------*/
/* Read and set restart values for spring dashpot */
void STR::TimInt::ReadRestartSpringDashpot()
{
  if (springman_->HaveSpringDashpot())
  {
    IO::DiscretizationReader reader(discret_, step_);
    springman_->ReadRestart(reader, (*time_)[0]);
  }
}

/*----------------------------------------------------------------------*/
/* Read and set restart values for contact / meshtying */
void STR::TimInt::ReadRestartContactMeshtying()
{
  //**********************************************************************
  // NOTE: There is an important difference here between contact and
  // meshtying simulations. In both cases, the current coupling operators
  // have to be re-computed for restart, but in the meshtying case this
  // means evaluating DM in the reference configuration!
  // Thus, both dis_ (current displacement state) and zero_ are handed
  // in and contact / meshtying managers choose the correct state.
  //**********************************************************************
  IO::DiscretizationReader reader(discret_, step_);

  if (HaveContactMeshtying()) cmtbridge_->ReadRestart(reader, (*dis_)(0), zeros_);
}

/*----------------------------------------------------------------------*/
/* Read and set restart values for beam contact */
void STR::TimInt::ReadRestartBeamContact()
{
  if (HaveBeamContact())
  {
    IO::DiscretizationReader reader(discret_, step_);
    beamcman_->ReadRestart(reader);
  }
}

/*----------------------------------------------------------------------*/
/* Read and set restart values for multi-scale */
void STR::TimInt::ReadRestartMultiScale()
{
  Teuchos::RCP<MAT::PAR::Bundle> materials = DRT::Problem::Instance()->Materials();
  for (std::map<int, Teuchos::RCP<MAT::PAR::Material>>::const_iterator i =
           materials->Map()->begin();
       i != materials->Map()->end(); ++i)
  {
    Teuchos::RCP<MAT::PAR::Material> mat = i->second;
    if (mat->Type() == INPAR::MAT::m_struct_multiscale)
    {
      // create the parameters for the discretization
      Teuchos::ParameterList p;
      // action for elements
      p.set("action", "multi_readrestart");
      discret_->Evaluate(
          p, Teuchos::null, Teuchos::null, Teuchos::null, Teuchos::null, Teuchos::null);
      discret_->ClearState();
      break;
    }
  }
}

/*----------------------------------------------------------------------*/
/* Calculate all output quantities that depend on a potential material history */
void STR::TimInt::PrepareOutput(bool force_prepare_timestep)
{
  DetermineStressStrain();
  DetermineEnergy();
  DetermineOptionalQuantity();
  if (havemicromat_) PrepareOutputMicro();
}

/*----------------------------------------------------------------------*
 *   Write Output while the Newton Iteration         by hiermeier 09/13 *
 *   (useful for debugging purposes)                                    */
void STR::TimInt::OutputEveryIter(bool nw, bool ls)
{
  // prevents repeated initialization of output writer
  bool datawritten = false;

  // Reinitialize the result file in the initial step
  if (outputcounter_ == 0)
  {
    firstoutputofrun_ = true;
    /*--------------------------------------------------------------------------------*
     | We modify the maximum number of steps per output file here, because we use the |
     | step number as indicator for the differentiation between time-,Newton- and     |
     | Line Search-steps. This is the minimal invasive change to prevent the output   |
     | routine to generate too many output files. We assume that the Newton method    |
     | needs an average cumulated number of 5 Newton/Line-Search steps per time step. |
     *--------------------------------------------------------------------------------*/
    int newFileSteps = 0;
    if (output_->Output()->FileSteps() >= std::numeric_limits<int>::max() / 50000)
      newFileSteps = std::numeric_limits<int>::max();
    else
      newFileSteps = output_->Output()->FileSteps() * 50000;

    output_->Output()->SetFileSteps(newFileSteps);

    std::string resultname = output_->Output()->FileName() + "_EveryIter";
    output_->NewResultFile(resultname, oei_filecounter_);
    output_->WriteMesh(0, 0.0);
  }

  // increase counter value
  if (ls)
    // for line search steps the outputcounter_ is increased by one
    outputcounter_++;
  else if (nw)
    // for Newton steps the outputcounter_ is increased by 100
    outputcounter_ += 100 - (outputcounter_ % 100);
  else
    // for time steps the outputcounter_ is increased by 100 000
    outputcounter_ += 100000 - (outputcounter_ % 100000);
  // time and step number

  output_->WriteMesh(outputcounter_, (double)outputcounter_);  //(*time_)[0]

  //  output_->OverwriteResultFile();
  OutputState(datawritten);
}

/*----------------------------------------------------------------------*/
/* output to file
 * originally by mwgee 03/07 */
void STR::TimInt::OutputStep(const bool forced_writerestart)
{
  // print iterations instead of steps
  if (outputeveryiter_)
  {
    OutputEveryIter();
    return;
  }

  // special treatment is necessary when restart is forced
  if (forced_writerestart)
  {
    // reset possible history data on element level
    ResetStep();
    // restart has already been written or simulation has just started
    if ((writerestartevery_ and (step_ % writerestartevery_ == 0)) or
        step_ == DRT::Problem::Instance()->Restart())
      return;
    // if state already exists, add restart information
    if (writeresultsevery_ and (step_ % writeresultsevery_ == 0))
    {
      AddRestartToOutputState();
      return;
    }
  }

  // this flag is passed along subroutines and prevents
  // repeated initialising of output writer, printing of
  // state vectors, or similar
  bool datawritten = false;

  // output restart (try this first)
  // write restart step
  if ((writerestartevery_ and (step_ % writerestartevery_ == 0) and step_ != 0) or
      forced_writerestart or
      DRT::Problem::Instance()->RestartManager()->Restart(step_, discret_->Comm()))
  {
    OutputRestart(datawritten);
    lastwrittenresultsstep_ = step_;
  }

  // output results (not necessary if restart in same step)
  if (writestate_ and writeresultsevery_ and (step_ % writeresultsevery_ == 0) and
      (not datawritten))
  {
    OutputState(datawritten);
    lastwrittenresultsstep_ = step_;
  }

  // output stress & strain
  if (writeresultsevery_ and
      ((writestress_ != INPAR::STR::stress_none) or
          (writecouplstress_ != INPAR::STR::stress_none) or
          (writestrain_ != INPAR::STR::strain_none) or
          (writeplstrain_ != INPAR::STR::strain_none)) and
      (step_ % writeresultsevery_ == 0))
  {
    OutputStressStrain(datawritten);
  }

  // output energy
  if (writeenergyevery_ and (step_ % writeenergyevery_ == 0))
  {
    OutputEnergy();
  }

  // output optional quantity
  if (writeresultsevery_ and (writeoptquantity_ != INPAR::STR::optquantity_none) and
      (step_ % writeresultsevery_ == 0))
  {
    OutputOptQuantity(datawritten);
  }

  // output active set, energies and momentum for contact
  OutputContact();

  // print error norms
  OutputErrorNorms();

  OutputVolumeMass();

  // output of nodal positions in current configuration
  OutputNodalPositions();

  // write output on micro-scale (multi-scale analysis)
  if (havemicromat_) OutputMicro();
}

/*-----------------------------------------------------------------------------*
 * write GMSH output of displacement field
 *-----------------------------------------------------------------------------*/
void STR::TimInt::WriteGmshStrucOutputStep()
{
  if (not gmsh_out_) return;

  const std::string filename = IO::GMSH::GetFileName("struct", stepn_, false, myrank_);
  std::ofstream gmshfilecontent(filename.c_str());

  // add 'View' to Gmsh postprocessing file
  gmshfilecontent << "View \" "
                  << "struct displacement \" {" << std::endl;
  // draw vector field 'struct displacement' for every element
  IO::GMSH::VectorFieldDofBasedToGmsh(discret_, Dispn(), gmshfilecontent, 0, true);
  gmshfilecontent << "};" << std::endl;
}

bool STR::TimInt::HasFinalStateBeenWritten() const { return step_ == lastwrittenresultsstep_; }
/*----------------------------------------------------------------------*/
/* We need the restart data to perform on "restarts" on the fly for parameter
 * continuation
 */
void STR::TimInt::GetRestartData(Teuchos::RCP<int> step, Teuchos::RCP<double> time,
    Teuchos::RCP<Epetra_Vector> disn, Teuchos::RCP<Epetra_Vector> veln,
    Teuchos::RCP<Epetra_Vector> accn, Teuchos::RCP<std::vector<char>> elementdata,
    Teuchos::RCP<std::vector<char>> nodedata)
{
  // at some point we have to create a copy
  *step = step_;
  *time = (*time_)[0];
  *disn = *disn_;
  *veln = *veln_;
  *accn = *accn_;
  *elementdata = *(discret_->PackMyElements());
  *nodedata = *(discret_->PackMyNodes());

  // get restart data is only for simple structure problems
  // hence

  // constraints
  if (conman_->HaveConstraint()) dserror("Get restart data not implemented for constraints");

  // contact / meshtying
  if (HaveContactMeshtying()) dserror("Get restart data not implemented for contact / meshtying");

  // beam contact
  if (HaveBeamContact()) dserror("Get restart data not implemented for beam contact");

  // biofilm growth
  if (HaveBiofilmGrowth()) dserror("Get restart data not implemented for biofilm growth");
}
/*----------------------------------------------------------------------*/
/* write restart
 * originally by mwgee 03/07 */
void STR::TimInt::OutputRestart(bool& datawritten)
{
  // Yes, we are going to write...
  datawritten = true;

  // write restart output, please
  if (step_ != 0) output_->WriteMesh(step_, (*time_)[0]);
  output_->NewStep(step_, (*time_)[0]);
  output_->WriteVector("displacement", (*dis_)(0));
  if (dismat_ != Teuchos::null) output_->WriteVector("material_displacement", (*dismat_)(0));
  output_->WriteVector("velocity", (*vel_)(0));
  output_->WriteVector("acceleration", (*acc_)(0));
  output_->WriteElementData(firstoutputofrun_);
  output_->WriteNodeData(firstoutputofrun_);
  WriteRestartForce(output_);
  // owner of elements is just written once because it does not change during simulation (so far)
  firstoutputofrun_ = false;

  // constraints
  if (conman_->HaveConstraint())
  {
    output_->WriteDouble("uzawaparameter", consolv_->GetUzawaParameter());
    output_->WriteVector("lagrmultiplier", conman_->GetLagrMultVector());
    output_->WriteVector("refconval", conman_->GetRefBaseValues());
  }

  // 0D cardiovascular models
  if (cardvasc0dman_->HaveCardiovascular0D())
  {
    output_->WriteVector("cv0d_df_np", cardvasc0dman_->Get0D_df_np());
    output_->WriteVector("cv0d_f_np", cardvasc0dman_->Get0D_f_np());

    output_->WriteVector("cv0d_dof_np", cardvasc0dman_->Get0D_dof_np());
    output_->WriteVector("vol_np", cardvasc0dman_->Get0D_vol_np());
  }

  // contact and meshtying
  if (HaveContactMeshtying())
  {
    cmtbridge_->WriteRestart(output_);
    cmtbridge_->PostprocessQuantities(output_);

    {
      Teuchos::RCP<Teuchos::ParameterList> cmtOutputParams =
          Teuchos::rcp(new Teuchos::ParameterList());
      cmtOutputParams->set<int>("step", step_);
      cmtOutputParams->set<double>("time", (*time_)[0]);
      cmtOutputParams->set<Teuchos::RCP<const Epetra_Vector>>("displacement", (*dis_)(0));
      cmtbridge_->PostprocessQuantitiesPerInterface(cmtOutputParams);
    }
  }

  // beam contact
  if (HaveBeamContact())
  {
    beamcman_->WriteRestart(output_);
  }

  // biofilm growth
  if (HaveBiofilmGrowth())
  {
    output_->WriteVector("str_growth_displ", strgrdisp_);
  }

  // springdashpot output
  if (springman_->HaveSpringDashpot()) springman_->OutputRestart(output_, discret_, disn_);

  // info dedicated to user's eyes staring at standard out
  if ((myrank_ == 0) and printscreen_ and (StepOld() % printscreen_ == 0))
  {
    IO::cout << "====== Restart for field '" << discret_->Name() << "' written in step " << step_
             << IO::endl;
  }
}

/*----------------------------------------------------------------------*/
/* output displacements, velocities and accelerations
 * originally by mwgee 03/07 */
void STR::TimInt::OutputState(bool& datawritten)
{
  // Yes, we are going to write...
  datawritten = true;

  // write now
  if (outputeveryiter_)
  {
    output_->NewStep(outputcounter_, (double)outputcounter_);
    output_->WriteVector("displacement", Teuchos::rcp_static_cast<Epetra_MultiVector>(disn_));
  }
  else
  {
    output_->NewStep(step_, (*time_)[0]);
    output_->WriteVector("displacement", (*dis_)(0));
  }

  if ((dismatn_ != Teuchos::null)) output_->WriteVector("material_displacement", (*dismat_)(0));

  // for visualization of vel and acc do not forget to comment in corresponding lines in
  // StructureEnsightWriter
  if (writevelacc_)
  {
    output_->WriteVector("velocity", (*vel_)(0));
    output_->WriteVector("acceleration", (*acc_)(0));
  }

  // biofilm growth
  if (HaveBiofilmGrowth())
  {
    output_->WriteVector("str_growth_displ", strgrdisp_);
  }

  // owner of elements is just written once because it does not change during simulation (so far)
  if (writeele_) output_->WriteElementData(firstoutputofrun_);
  output_->WriteNodeData(firstoutputofrun_);
  firstoutputofrun_ = false;

  // meshtying and contact output
  if (HaveContactMeshtying())
  {
    cmtbridge_->PostprocessQuantities(output_);

    {
      Teuchos::RCP<Teuchos::ParameterList> cmtOutputParams =
          Teuchos::rcp(new Teuchos::ParameterList());
      cmtOutputParams->set<int>("step", step_);
      cmtOutputParams->set<double>("time", (*time_)[0]);
      cmtOutputParams->set<Teuchos::RCP<const Epetra_Vector>>("displacement", (*dis_)(0));
      cmtbridge_->PostprocessQuantitiesPerInterface(cmtOutputParams);
    }
  }

  if (porositysplitter_ != Teuchos::null)
  {
    Teuchos::RCP<Epetra_Vector> porosity = porositysplitter_->ExtractCondVector((*dis_)(0));
    output_->WriteVector("porosity_p1", porosity);
  }

  // springdashpot output
  if (springman_->HaveSpringDashpot()) springman_->Output(output_, discret_, disn_);
}

/*----------------------------------------------------------------------*/
/* add restart information to OutputState */
void STR::TimInt::AddRestartToOutputState()
{
  // add velocity and acceleration if necessary
  if (!writevelacc_)
  {
    output_->WriteVector("velocity", (*vel_)(0));
    output_->WriteVector("acceleration", (*acc_)(0));
  }

  WriteRestartForce(output_);

  // constraints
  if (conman_->HaveConstraint())
  {
    output_->WriteDouble("uzawaparameter", consolv_->GetUzawaParameter());
    output_->WriteVector("lagrmultiplier", conman_->GetLagrMultVector());
    output_->WriteVector("refconval", conman_->GetRefBaseValues());
  }

  // 0D cardiovascular models
  if (cardvasc0dman_->HaveCardiovascular0D())
  {
    output_->WriteVector("cv0d_df_np", cardvasc0dman_->Get0D_df_np());
    output_->WriteVector("cv0d_f_np", cardvasc0dman_->Get0D_f_np());

    output_->WriteVector("cv0d_dof_np", cardvasc0dman_->Get0D_dof_np());
    output_->WriteVector("vol_np", cardvasc0dman_->Get0D_vol_np());
  }

  // springdashpot output
  if (springman_->HaveSpringDashpot()) springman_->OutputRestart(output_, discret_, disn_);

  // contact/meshtying
  if (HaveContactMeshtying()) cmtbridge_->WriteRestart(output_, true);

  // TODO: add missing restart data for surface stress and contact/meshtying here

  // beam contact
  if (HaveBeamContact()) beamcman_->WriteRestart(output_);


  // finally add the missing mesh information, order is important here
  output_->WriteMesh(step_, (*time_)[0]);

  // info dedicated to user's eyes staring at standard out
  if ((myrank_ == 0) and printscreen_ and (StepOld() % printscreen_ == 0))
  {
    IO::cout << "====== Restart for field '" << discret_->Name() << "' written in step " << step_
             << IO::endl;
  }
}

/*----------------------------------------------------------------------*/
/* Calculation of stresses and strains */
void STR::TimInt::DetermineStressStrain()
{
  if (writeresultsevery_ and
      ((writestress_ != INPAR::STR::stress_none) or
          (writecouplstress_ != INPAR::STR::stress_none) or
          (writestrain_ != INPAR::STR::strain_none) or
          (writeplstrain_ != INPAR::STR::strain_none)) and
      (stepn_ % writeresultsevery_ == 0))
  {
    //-------------------------------
    // create the parameters for the discretization
    Teuchos::ParameterList p;
    // action for elements
    p.set("action", "calc_struct_stress");
    // other parameters that might be needed by the elements
    p.set("total time", timen_);
    p.set("delta time", (*dt_)[0]);

    stressdata_ = Teuchos::rcp(new std::vector<char>());
    p.set("stress", stressdata_);
    p.set<int>("iostress", writestress_);

    // write stress data that arise from the coupling with another field, e.g.
    // in TSI: couplstress corresponds to thermal stresses
    couplstressdata_ = Teuchos::rcp(new std::vector<char>());
    p.set("couplstress", couplstressdata_);
    p.set<int>("iocouplstress", writecouplstress_);

    straindata_ = Teuchos::rcp(new std::vector<char>());
    p.set("strain", straindata_);
    p.set<int>("iostrain", writestrain_);

    // plastic strain
    plstraindata_ = Teuchos::rcp(new std::vector<char>());
    p.set("plstrain", plstraindata_);
    p.set<int>("ioplstrain", writeplstrain_);

    // rotation tensor
    rotdata_ = Teuchos::rcp(new std::vector<char>());
    p.set("rotation", rotdata_);

    // set vector values needed by elements
    discret_->ClearState();
    // extended SetState(0,...) in case of multiple dofsets (e.g. TSI)
    discret_->SetState(0, "residual displacement", zeros_);
    discret_->SetState(0, "displacement", disn_);

    if ((dismatn_ != Teuchos::null)) discret_->SetState(0, "material_displacement", dismatn_);

    discret_->Evaluate(
        p, Teuchos::null, Teuchos::null, Teuchos::null, Teuchos::null, Teuchos::null);
    discret_->ClearState();
  }
}

/*----------------------------------------------------------------------*/
/* Calculation of internal, external and kinetic energy */
void STR::TimInt::DetermineEnergy()
{
  if (writeenergyevery_ and (stepn_ % writeenergyevery_ == 0))
  {
    // internal/strain energy
    intergy_ = 0.0;  // total internal energy
    {
      Teuchos::ParameterList p;
      // other parameters needed by the elements
      p.set("action", "calc_struct_energy");

      // set vector values needed by elements
      discret_->ClearState();
      discret_->SetState("displacement", disn_);
      // get energies
      Teuchos::RCP<CORE::LINALG::SerialDenseVector> energies =
          Teuchos::rcp(new CORE::LINALG::SerialDenseVector(1));
      discret_->EvaluateScalars(p, energies);
      discret_->ClearState();
      intergy_ = (*energies)(0);
    }

    // global calculation of kinetic energy
    kinergy_ = 0.0;  // total kinetic energy
    {
      Teuchos::RCP<Epetra_Vector> linmom = CORE::LINALG::CreateVector(*DofRowMapView(), true);
      mass_->Multiply(false, *veln_, *linmom);
      linmom->Dot(*veln_, &kinergy_);
      kinergy_ *= 0.5;
    }

    // external energy
    extergy_ = 0.0;  // total external energy
    {
      // WARNING: This will only work with dead loads and implicit time
      // integration (otherwise there is no fextn_)!!!
      Teuchos::RCP<Epetra_Vector> fext = FextNew();
      fext->Dot(*disn_, &extergy_);
    }
  }
}

/*----------------------------------------------------------------------*/
/* Calculation of an optional quantity */
void STR::TimInt::DetermineOptionalQuantity()
{
  if (writeresultsevery_ and (writeoptquantity_ != INPAR::STR::optquantity_none) and
      (stepn_ % writeresultsevery_ == 0))
  {
    //-------------------------------
    // create the parameters for the discretization
    Teuchos::ParameterList p;

    // action for elements
    if (writeoptquantity_ == INPAR::STR::optquantity_membranethickness)
      p.set("action", "calc_struct_thickness");
    else
      dserror("requested optional quantity type not supported");

    // other parameters that might be needed by the elements
    p.set("total time", timen_);
    p.set("delta time", (*dt_)[0]);

    optquantitydata_ = Teuchos::rcp(new std::vector<char>());
    p.set("optquantity", optquantitydata_);
    p.set<int>("iooptquantity", writeoptquantity_);

    // set vector values needed by elements
    discret_->ClearState();
    // extended SetState(0,...) in case of multiple dofsets (e.g. TSI)
    discret_->SetState(0, "residual displacement", zeros_);
    discret_->SetState(0, "displacement", disn_);

    if ((dismatn_ != Teuchos::null)) discret_->SetState(0, "material_displacement", dismatn_);

    discret_->Evaluate(
        p, Teuchos::null, Teuchos::null, Teuchos::null, Teuchos::null, Teuchos::null);
    discret_->ClearState();
  }
}

/*----------------------------------------------------------------------*/
/* stress calculation and output */
void STR::TimInt::OutputStressStrain(bool& datawritten)
{
  // Make new step
  if (not datawritten)
  {
    output_->NewStep(step_, (*time_)[0]);
  }
  datawritten = true;

  // write stress
  if (writestress_ != INPAR::STR::stress_none)
  {
    std::string stresstext = "";
    if (writestress_ == INPAR::STR::stress_cauchy)
    {
      stresstext = "gauss_cauchy_stresses_xyz";
    }
    else if (writestress_ == INPAR::STR::stress_2pk)
    {
      stresstext = "gauss_2PK_stresses_xyz";
    }
    else
    {
      dserror("requested stress type not supported");
    }
    output_->WriteVector(stresstext, *stressdata_, *(discret_->ElementRowMap()));
    // we don't need this anymore
    stressdata_ = Teuchos::null;
  }

  // write coupling stress
  if (writecouplstress_ != INPAR::STR::stress_none)
  {
    std::string couplstresstext = "";
    if (writecouplstress_ == INPAR::STR::stress_cauchy)
    {
      couplstresstext = "gauss_cauchy_coupling_stresses_xyz";
    }
    else if (writecouplstress_ == INPAR::STR::stress_2pk)
    {
      couplstresstext = "gauss_2PK_coupling_stresses_xyz";
    }
    else
    {
      dserror("requested stress type not supported");
    }
    output_->WriteVector(couplstresstext, *couplstressdata_, *(discret_->ElementRowMap()));
    // we don't need this anymore
    couplstressdata_ = Teuchos::null;
  }

  // write strain
  if (writestrain_ != INPAR::STR::strain_none)
  {
    std::string straintext = "";
    if (writestrain_ == INPAR::STR::strain_ea)
    {
      straintext = "gauss_EA_strains_xyz";
    }
    else if (writestrain_ == INPAR::STR::strain_gl)
    {
      straintext = "gauss_GL_strains_xyz";
    }
    else if (writestrain_ == INPAR::STR::strain_log)
    {
      straintext = "gauss_LOG_strains_xyz";
    }
    else
    {
      dserror("requested strain type not supported");
    }
    output_->WriteVector(straintext, *straindata_, *(discret_->ElementRowMap()));
    // we don't need this anymore
    straindata_ = Teuchos::null;
  }

  // write plastic strain
  if (writeplstrain_ != INPAR::STR::strain_none)
  {
    std::string plstraintext = "";
    if (writeplstrain_ == INPAR::STR::strain_ea)
    {
      plstraintext = "gauss_pl_EA_strains_xyz";
    }
    else if (writeplstrain_ == INPAR::STR::strain_gl)
    {
      plstraintext = "gauss_pl_GL_strains_xyz";
    }
    else
    {
      dserror("requested plastic strain type not supported");
    }
    output_->WriteVector(plstraintext, *plstraindata_, *(discret_->ElementRowMap()));
    // we don't need this anymore
    plstraindata_ = Teuchos::null;
  }

  // write structural rotation tensor
  if (writerotation_) output_->WriteVector("rotation", *rotdata_, *(discret_->ElementRowMap()));
}

/*----------------------------------------------------------------------*/
/* output system energies */
void STR::TimInt::OutputEnergy()
{
  // total energy
  double totergy = kinergy_ + intergy_ - extergy_;

  // the output
  if (myrank_ == 0)
  {
    (*energyfile_) << " " << std::setw(9) << step_ << std::scientific << std::setprecision(16)
                   << " " << (*time_)[0] << " " << totergy << " " << kinergy_ << " " << intergy_
                   << " " << extergy_ << std::endl;
  }
}

/*----------------------------------------------------------------------*/
/* stress calculation and output */
void STR::TimInt::OutputOptQuantity(bool& datawritten)
{
  // Make new step
  if (not datawritten)
  {
    output_->NewStep(step_, (*time_)[0]);
  }
  datawritten = true;

  // write optional quantity
  if (writeoptquantity_ != INPAR::STR::optquantity_none)
  {
    std::string optquantitytext = "";
    if (writeoptquantity_ == INPAR::STR::optquantity_membranethickness)
      optquantitytext = "gauss_membrane_thickness";
    else
      dserror("requested optional quantity type not supported");

    output_->WriteVector(optquantitytext, *optquantitydata_, *(discret_->ElementRowMap()));
    // we don't need this anymore
    optquantitydata_ = Teuchos::null;
  }
}

/*----------------------------------------------------------------------*/
/* output active set, energies and momentum for contact */
void STR::TimInt::OutputContact()
{
  // only for contact / meshtying simulations
  if (HaveContactMeshtying())
  {
    // print active set
    cmtbridge_->GetStrategy().PrintActiveSet();

    // check chosen output option
    INPAR::CONTACT::EmOutputType emtype = INPUT::IntegralValue<INPAR::CONTACT::EmOutputType>(
        cmtbridge_->GetStrategy().Params(), "EMOUTPUT");

    // get out of here if no energy/momentum output wanted
    if (emtype == INPAR::CONTACT::output_none) return;

    // get some parameters from parameter list
    double timen = (*time_)[0];
    double dt = (*dt_)[0];
    int dim = cmtbridge_->GetStrategy().Dim();

    // global linear momentum (M*v)
    Teuchos::RCP<Epetra_Vector> mv = CORE::LINALG::CreateVector(*(discret_->DofRowMap()), true);
    mass_->Multiply(false, (*vel_)[0], *mv);

    // linear / angular momentum
    std::vector<double> sumlinmom(3);
    std::vector<double> sumangmom(3);
    std::vector<double> angmom(3);
    std::vector<double> linmom(3);

    // vectors of nodal properties
    std::vector<double> nodelinmom(3);
    std::vector<double> nodeangmom(3);
    std::vector<double> position(3);

    // loop over all nodes belonging to the respective processor
    for (int k = 0; k < (discret_->NodeRowMap())->NumMyElements(); ++k)
    {
      // get current node
      int gid = (discret_->NodeRowMap())->GID(k);
      DRT::Node* mynode = discret_->gNode(gid);
      std::vector<int> globaldofs = discret_->Dof(mynode);

      // loop over all DOFs comprised by this node
      for (int i = 0; i < dim; i++)
      {
        nodelinmom[i] = (*mv)[mv->Map().LID(globaldofs[i])];
        sumlinmom[i] += nodelinmom[i];
        position[i] = (mynode->X())[i] + ((*dis_)[0])[mv->Map().LID(globaldofs[i])];
      }

      // calculate vector product position x linmom
      nodeangmom[0] = position[1] * nodelinmom[2] - position[2] * nodelinmom[1];
      nodeangmom[1] = position[2] * nodelinmom[0] - position[0] * nodelinmom[2];
      nodeangmom[2] = position[0] * nodelinmom[1] - position[1] * nodelinmom[0];

      // loop over all DOFs comprised by this node
      for (int i = 0; i < 3; ++i) sumangmom[i] += nodeangmom[i];
    }

    // global quantities (sum over all processors)
    for (int i = 0; i < 3; ++i)
    {
      cmtbridge_->Comm().SumAll(&sumangmom[i], &angmom[i], 1);
      cmtbridge_->Comm().SumAll(&sumlinmom[i], &linmom[i], 1);
    }

    //--------------------------Calculation of total kinetic energy
    double kinen = 0.0;
    mv->Dot((*vel_)[0], &kinen);
    kinen *= 0.5;

    //-------------------------Calculation of total internal energy
    // BE CAREFUL HERE!!!
    // Currently this is only valid for materials without(!) history
    // data, because we have already updated everything and thus
    // any potential material history has already been overwritten.
    // When we are interested in internal energy output for contact
    // or meshtying simulations with(!) material history, we should
    // move the following block before the first UpdateStep() method.
    double inten = 0.0;
    Teuchos::ParameterList p;
    p.set("action", "calc_struct_energy");
    discret_->ClearState();
    discret_->SetState("displacement", (*dis_)(0));
    Teuchos::RCP<CORE::LINALG::SerialDenseVector> energies =
        Teuchos::rcp(new CORE::LINALG::SerialDenseVector(1));
    energies->putScalar(0.0);
    discret_->EvaluateScalars(p, energies);
    discret_->ClearState();
    inten = (*energies)(0);

    //-------------------------Calculation of total external energy
    double exten = 0.0;
    // WARNING: This will only work with dead loads!!!
    // fext_->Dot(*dis_, &exten);

    //----------------------------------------Print results to file
    if (emtype == INPAR::CONTACT::output_file || emtype == INPAR::CONTACT::output_both)
    {
      // processor 0 does all the work
      if (!myrank_)
      {
        // path and filename
        std::ostringstream filename;
        const std::string filebase = DRT::Problem::Instance()->OutputControlFile()->FileName();
        filename << filebase << ".energymomentum";

        // open file
        FILE* MyFile = nullptr;
        if (timen < 2 * dt)
        {
          MyFile = fopen(filename.str().c_str(), "wt");

          // initialize file pointer for writing contact interface forces/moments
          FILE* MyConForce = nullptr;
          std::ostringstream filenameif;
          filenameif << filebase << ".energymomentum";
          MyConForce = fopen(filenameif.str().c_str(), "wt");
          if (MyConForce != nullptr)
            fclose(MyConForce);
          else
            dserror("File for writing contact interface forces/moments could not be generated.");
        }
        else
          MyFile = fopen(filename.str().c_str(), "at+");

        // add current values to file
        if (MyFile != nullptr)
        {
          std::stringstream filec;
          fprintf(MyFile, "% e\t", timen);
          for (int i = 0; i < 3; i++) fprintf(MyFile, "% e\t", linmom[i]);
          for (int i = 0; i < 3; i++) fprintf(MyFile, "% e\t", angmom[i]);
          fprintf(MyFile, "% e\t% e\t% e\t% e\n", kinen, inten, exten, kinen + inten - exten);
          fclose(MyFile);
        }
        else
          dserror("File for writing momentum and energy data could not be opened.");
      }
    }

    //-------------------------------Print energy results to screen
    if (emtype == INPAR::CONTACT::output_screen || emtype == INPAR::CONTACT::output_both)
    {
      // processor 0 does all the work
      if (!myrank_)
      {
        printf("******************************");
        printf("\nMECHANICAL ENERGIES:");
        printf("\nE_kinetic \t %e", kinen);
        printf("\nE_internal \t %e", inten);
        printf("\nE_external \t %e", exten);
        printf("\n------------------------------");
        printf("\nE_total \t %e", kinen + inten - exten);
        printf("\n******************************");

        printf("\n\n********************************************");
        printf("\nLINEAR / ANGULAR MOMENTUM:");
        printf("\nL_x  % e \t H_x  % e", linmom[0], angmom[0]);
        printf("\nL_y  % e \t H_y  % e", linmom[1], angmom[1]);
        printf("\nL_z  % e \t H_z  % e", linmom[2], angmom[2]);
        printf("\n********************************************\n\n");
        fflush(stdout);
      }
    }

    //-------------------------- Compute and output interface forces
    cmtbridge_->GetStrategy().InterfaceForces(true);
  }
}

/*----------------------------------------------------------------------*/
/* output solution error norms */
void STR::TimInt::OutputErrorNorms()
{
  // get out of here if no output wanted
  const Teuchos::ParameterList& listcmt = DRT::Problem::Instance()->ContactDynamicParams();
  INPAR::CONTACT::ErrorNorms entype =
      INPUT::IntegralValue<INPAR::CONTACT::ErrorNorms>(listcmt, "ERROR_NORMS");
  if (entype == INPAR::CONTACT::errornorms_none) return;

  // initialize variables
  Teuchos::RCP<CORE::LINALG::SerialDenseVector> norms =
      Teuchos::rcp(new CORE::LINALG::SerialDenseVector(3));
  norms->putScalar(0.0);

  // vector for output
  Teuchos::RCP<Epetra_MultiVector> normvec =
      Teuchos::rcp(new Epetra_MultiVector(*discret_->ElementRowMap(), 3));

  // call discretization to evaluate error norms
  Teuchos::ParameterList p;
  p.set("action", "calc_struct_errornorms");
  discret_->ClearState();
  discret_->SetState("displacement", (*dis_)(0));
  discret_->EvaluateScalars(p, norms);
  discret_->EvaluateScalars(p, normvec);
  discret_->ClearState();

  // proc 0 writes output to screen
  if (!myrank_)
  {
    printf("**********************************");
    printf("\nSOLUTION ERROR NORMS:");
    printf("\nL_2 norm:     %.10e", sqrt((*norms)(0)));
    printf("\nH_1 norm:     %.10e", sqrt((*norms)(1)));
    printf("\nEnergy norm:  %.10e", sqrt((*norms)(2)));
    printf("\n**********************************\n\n");
    fflush(stdout);
  }

  Teuchos::RCP<Epetra_MultiVector> L2_norm =
      Teuchos::rcp(new Epetra_MultiVector(*discret_->ElementRowMap(), 1));
  Teuchos::RCP<Epetra_MultiVector> H1_norm =
      Teuchos::rcp(new Epetra_MultiVector(*discret_->ElementRowMap(), 1));
  Teuchos::RCP<Epetra_MultiVector> Energy_norm =
      Teuchos::rcp(new Epetra_MultiVector(*discret_->ElementRowMap(), 1));
  Epetra_MultiVector& sca = *(normvec.get());
  Epetra_MultiVector& scaL2 = *(L2_norm.get());
  Epetra_MultiVector& scaH1 = *(H1_norm.get());
  Epetra_MultiVector& scaEn = *(Energy_norm.get());

  for (int i = 0; i < discret_->NumMyRowElements(); ++i)
  {
    (*scaL2(0))[i] = (*sca(0))[i];
    (*scaH1(0))[i] = (*sca(1))[i];
    (*scaEn(0))[i] = (*sca(2))[i];
  }

  // output to file
  output_->WriteVector("L2_norm", L2_norm);
  output_->WriteVector("H1_norm", H1_norm);
  output_->WriteVector("Energy_norm", Energy_norm);
}

/*----------------------------------------------------------------------*/
/* output volume and mass */
void STR::TimInt::OutputVolumeMass()
{
  const Teuchos::ParameterList& listwear = DRT::Problem::Instance()->WearParams();
  bool massvol = INPUT::IntegralValue<int>(listwear, "VOLMASS_OUTPUT");
  if (!massvol) return;

  // initialize variables
  Teuchos::RCP<CORE::LINALG::SerialDenseVector> norms =
      Teuchos::rcp(new CORE::LINALG::SerialDenseVector(6));
  norms->putScalar(0.0);

  // call discretization to evaluate error norms
  Teuchos::ParameterList p;
  p.set("action", "calc_struct_mass_volume");
  discret_->ClearState();
  discret_->SetState("displacement", (*dis_)(0));
  if ((dismatn_ != Teuchos::null)) discret_->SetState(0, "material_displacement", dismatn_);
  discret_->EvaluateScalars(p, norms);
  discret_->ClearState();

  // proc 0 writes output to screen
  if (!myrank_)
  {
    printf("**********************************");
    printf("\nVOLUMES:");
    printf("\nVolume ref.:     %.10e", ((*norms)(0)));
    printf("\nVolume mat.:     %.10e", ((*norms)(1)));
    printf("\nDIFF.:           %.10e", ((*norms)(0)) - ((*norms)(1)));
    printf("\nVolume cur.:     %.10e", ((*norms)(2)));
    printf("\n**********************************");
    printf("\nMass:");
    printf("\nMass ref.:       %.10e", ((*norms)(3)));
    printf("\nMass mat.:       %.10e", ((*norms)(4)));
    printf("\nDIFF.:           %.10e", ((*norms)(3)) - ((*norms)(4)));
    printf("\nMass cur.:       %.10e", ((*norms)(5)));
    printf("\n**********************************\n\n");
    fflush(stdout);
  }
}

/*----------------------------------------------------------------------*/
/* output on micro-scale */
void STR::TimInt::OutputMicro()
{
  for (int i = 0; i < discret_->NumMyRowElements(); i++)
  {
    DRT::Element* actele = discret_->lRowElement(i);
    Teuchos::RCP<MAT::Material> mat = actele->Material();
    if (mat->MaterialType() == INPAR::MAT::m_struct_multiscale)
    {
      MAT::MicroMaterial* micro = static_cast<MAT::MicroMaterial*>(mat.get());
      micro->Output();
    }
  }
}

/*----------------------------------------------------------------------*/
/* calculate stresses and strains on micro-scale */
void STR::TimInt::PrepareOutputMicro()
{
  for (int i = 0; i < discret_->NumMyRowElements(); i++)
  {
    DRT::Element* actele = discret_->lRowElement(i);

    Teuchos::RCP<MAT::Material> mat = actele->Material();
    if (mat->MaterialType() == INPAR::MAT::m_struct_multiscale)
    {
      MAT::MicroMaterial* micro = static_cast<MAT::MicroMaterial*>(mat.get());
      micro->PrepareOutput();
    }
  }
}

/*----------------------------------------------------------------------*/
/* output nodal positions */
void STR::TimInt::OutputNodalPositions()
{
#ifdef PRINTSTRUCTDEFORMEDNODECOORDS

  /////////////////////////////////////////////////////////////////
  // from here I want to output my ale displacements - devaal 14.12.2010
  /////////////////////////////////////////////////////////////////

  if (discret_->Comm().NumProc() != 1)
    dserror("The flag PRINTSTRUCTDEFORMEDNODECOORDS is on and only works with 1 processor");

  std::cout << "STRUCT DISCRETIZATION IN THE DEFORMED CONFIGURATIONS" << std::endl;
  // does discret_ exist here?
  // std::cout << "discret_->NodeRowMap()" << discret_->NodeRowMap() << std::endl;

  // Teuchos::RCP<Epetra_Vector> mynoderowmap = Teuchos::rcp(new
  // Epetra_Vector(discret_->NodeRowMap())); Teuchos::RCP<Epetra_Vector> noderowmap_ =
  // Teuchos::rcp(new Epetra_Vector(discret_->NodeRowMap())); DofRowMapView()  = Teuchos::rcp(new
  // discret_->DofRowMap());
  const Epetra_Map* noderowmap = discret_->NodeRowMap();
  const Epetra_Map* dofrowmap = discret_->DofRowMap();


  for (int lid = 0; lid < noderowmap->NumGlobalPoints(); lid++)
  {
    int gid;
    // get global id of a node
    gid = noderowmap->GID(lid);
    // get the node
    DRT::Node* node = discret_->gNode(gid);
    // std::cout<<"mynode"<<*node<<std::endl;

    // get the coordinates of the node
    const double* X = node->X();
    // get degrees of freedom of a node

    std::vector<int> gdofs = discret_->Dof(node);
    // std::cout << "for node:" << *node << std::endl;
    // std::cout << "this is my gdof vector" << gdofs[0] << " " << gdofs[1] << " " << gdofs[2] <<
    // std::endl;

    // get displacements of a node
    std::vector<double> mydisp(3, 0.0);
    for (int ldof = 0; ldof < 3; ldof++)
    {
      int displid = dofrowmap->LID(gdofs[ldof]);

      // std::cout << "displacement local id - in the rowmap" << displid << std::endl;
      mydisp[ldof] = ((*dis_)[0])[displid];
      // std::cout << "at node" << gid << "mydisplacement in each direction" << mydisp[ldof] <<
      // std::endl; make zero if it is too small
      if (abs(mydisp[ldof]) < 0.00001)
      {
        mydisp[ldof] = 0.0;
      }
    }

    // Export disp, X
    double newX = mydisp[0] + X[0];
    double newY = mydisp[1] + X[1];
    double newZ = mydisp[2] + X[2];
    // std::cout << "NODE " << gid << "  COORD  " << newX << " " << newY << " " << newZ <<
    // std::endl;
    std::cout << gid << " " << newX << " " << newY << " " << newZ << std::endl;
  }

#endif  // PRINTSTRUCTDEFORMEDNODECOORDS
}

/*----------------------------------------------------------------------*/
/* evaluate external forces at t_{n+1} */
void STR::TimInt::ApplyForceExternal(const double time, const Teuchos::RCP<Epetra_Vector> dis,
    const Teuchos::RCP<Epetra_Vector> disn, const Teuchos::RCP<Epetra_Vector> vel,
    Teuchos::RCP<Epetra_Vector>& fext)
{
  Teuchos::ParameterList p;
  // other parameters needed by the elements
  p.set("total time", time);

  // set vector values needed by elements
  discret_->ClearState();
  discret_->SetState(0, "displacement", dis);
  discret_->SetState(0, "displacement new", disn);

  if (damping_ == INPAR::STR::damp_material) discret_->SetState(0, "velocity", vel);

  discret_->EvaluateNeumann(p, *fext);
}

/*----------------------------------------------------------------------*/
/* check whether we have nonlinear inertia forces or not */
int STR::TimInt::HaveNonlinearMass() const
{
  const Teuchos::ParameterList& sdyn = DRT::Problem::Instance()->StructuralDynamicParams();
  int masslin = INPUT::IntegralValue<INPAR::STR::MassLin>(sdyn, "MASSLIN");

  return masslin;
}

/*----------------------------------------------------------------------*/
/* check whether the initial conditions are fulfilled */
void STR::TimInt::NonlinearMassSanityCheck(Teuchos::RCP<const Epetra_Vector> fext,
    Teuchos::RCP<const Epetra_Vector> dis, Teuchos::RCP<const Epetra_Vector> vel,
    Teuchos::RCP<const Epetra_Vector> acc, const Teuchos::ParameterList* sdynparams) const
{
  double fextnorm;
  fext->Norm2(&fextnorm);

  double dispnorm;
  dis->Norm2(&dispnorm);

  double velnorm;
  vel->Norm2(&velnorm);

  double accnorm;
  acc->Norm2(&accnorm);

  if (fextnorm > 1.0e-14)
  {
    dserror(
        "Initial configuration does not fulfill equilibrium, check your "
        "initial external forces, velocities and accelerations!!!");
  }

  if ((dispnorm > 1.0e-14) or (velnorm > 1.0e-14) or (accnorm > 1.0e-14))
  {
    dserror(
        "Nonlinear inertia terms (input flag 'MASSLIN' not set to 'none') "
        "are only possible for vanishing initial displacements, velocities and "
        "accelerations so far!!!\n"
        "norm disp = %f \n"
        "norm vel  = %f \n"
        "norm acc  = %f",
        dispnorm, velnorm, accnorm);
  }

  if (HaveNonlinearMass() == INPAR::STR::ml_rotations and
      INPUT::IntegralValue<INPAR::STR::PredEnum>(*sdynparams, "PREDICT") !=
          INPAR::STR::pred_constdis)
  {
    dserror(
        "Only constant displacement consistent velocity and acceleration "
        "predictor possible for multiplicative Genalpha time integration!");
  }

  if (sdynparams != nullptr)
  {
    if (HaveNonlinearMass() == INPAR::STR::ml_rotations and
        INPUT::IntegralValue<INPAR::STR::DynamicType>(*sdynparams, "DYNAMICTYP") !=
            INPAR::STR::dyna_genalpha)
      dserror(
          "Nonlinear inertia forces for rotational DoFs only implemented "
          "for GenAlpha time integration so far!");
  }
}

/*----------------------------------------------------------------------*/
/* evaluate ordinary internal force */
void STR::TimInt::ApplyForceInternal(const double time, const double dt,
    Teuchos::RCP<const Epetra_Vector> dis, Teuchos::RCP<const Epetra_Vector> disi,
    Teuchos::RCP<const Epetra_Vector> vel, Teuchos::RCP<Epetra_Vector> fint)
{
  // create the parameters for the discretization
  Teuchos::ParameterList p;
  // action for elements
  std::string action = "calc_struct_internalforce";

  p.set("action", action);
  // other parameters that might be needed by the elements
  p.set("total time", time);
  p.set("delta time", dt);

  if (pressure_ != Teuchos::null) p.set("volume", 0.0);
  // set vector values needed by elements
  discret_->ClearState();
  discret_->SetState("residual displacement", disi);  // these are incremental
  discret_->SetState("displacement", dis);

  if (damping_ == INPAR::STR::damp_material) discret_->SetState("velocity", vel);
  // fintn_->PutScalar(0.0);  // initialise internal force vector
  discret_->Evaluate(p, Teuchos::null, Teuchos::null, fint, Teuchos::null, Teuchos::null);

  discret_->ClearState();
}

/*----------------------------------------------------------------------*/
INPAR::STR::ConvergenceStatus STR::TimInt::PerformErrorAction(
    INPAR::STR::ConvergenceStatus nonlinsoldiv)
{
  // what to do when nonlinear solver does not converge
  switch (divcontype_)
  {
    case INPAR::STR::divcont_stop:
    {
      // write restart output of last converged step before stopping
      Output(true);

      // we should not get here, dserror for safety
      dserror("Nonlinear solver did not converge! ");
      return INPAR::STR::conv_nonlin_fail;
    }
    case INPAR::STR::divcont_continue:
    {
      // we should not get here, dserror for safety
      dserror("Nonlinear solver did not converge! ");
      return INPAR::STR::conv_nonlin_fail;
    }
    break;
    case INPAR::STR::divcont_repeat_step:
    {
      IO::cout << "Nonlinear solver failed to converge repeat time step" << IO::endl;

      // reset step (e.g. quantities on element level)
      ResetStep();

      return INPAR::STR::conv_success;
    }
    break;
    case INPAR::STR::divcont_halve_step:
    {
      IO::cout << "Nonlinear solver failed to converge at time t= " << timen_
               << ". Divide timestep in half. "
               << "Old time step: " << (*dt_)[0] << IO::endl
               << "New time step: " << 0.5 * (*dt_)[0] << IO::endl
               << IO::endl;

      // halve the time step size
      (*dt_)[0] = (*dt_)[0] * 0.5;
      // update the number of max time steps
      stepmax_ = stepmax_ + (stepmax_ - stepn_) + 1;
      // reset timen_ because it is set in the constructor
      timen_ = (*time_)[0] + (*dt_)[0];
      // reset step (e.g. quantities on element level)
      ResetStep();

      return INPAR::STR::conv_success;
    }
    break;
    case INPAR::STR::divcont_adapt_step:
    {
      // maximal possible refinementlevel
      const int maxdivconrefinementlevel = 10;
      const int maxstepmax = 1000000;
      IO::cout << "Nonlinear solver failed to converge at time t= " << timen_
               << ". Divide timestep in half. "
               << "Old time step: " << (*dt_)[0] << IO::endl
               << "New time step: " << 0.5 * (*dt_)[0] << IO::endl
               << IO::endl;

      // halve the time step size
      (*dt_)[0] = (*dt_)[0] * 0.5;

      // update the number of max time steps
      stepmax_ = stepmax_ + (stepmax_ - stepn_) + 1;
      // reset timen_ because it is set in the constructor
      timen_ = (*time_)[0] + (*dt_)[0];

      divconrefinementlevel_++;
      divconnumfinestep_ = 0;

      if (divconrefinementlevel_ == maxdivconrefinementlevel)
        dserror(
            "Maximal divercont refinement level reached. Adapt your time basic time step size!");

      if (stepmax_ > maxstepmax) dserror("Upper level for stepmax_ reached!");

      // reset step (e.g. quantities on element level)
      ResetStep();

      return INPAR::STR::conv_success;
    }
    break;
    case INPAR::STR::divcont_rand_adapt_step:
    case INPAR::STR::divcont_rand_adapt_step_ele_err:
    {
      // generate random number between 0.51 and 1.99 (as mean value of random
      // numbers generated on all processors), alternating values larger
      // and smaller than 1.0
      double proc_randnum_get = ((double)rand() / (double)RAND_MAX);
      double proc_randnum = proc_randnum_get;
      double randnum = 1.0;
      discret_->Comm().SumAll(&proc_randnum, &randnum, 1);
      const double numproc = discret_->Comm().NumProc();
      randnum /= numproc;
      if (rand_tsfac_ > 1.0)
        rand_tsfac_ = randnum * 0.49 + 0.51;
      else if (rand_tsfac_ < 1.0)
        rand_tsfac_ = randnum * 0.99 + 1.0;
      else
        rand_tsfac_ = randnum * 1.48 + 0.51;
      if (myrank_ == 0)
        IO::cout << "Nonlinear solver failed to converge: modifying time-step size by random "
                    "number between 0.51 and 1.99 -> here: "
                 << rand_tsfac_ << " !" << IO::endl;
      // multiply time-step size by random number
      (*dt_)[0] = (*dt_)[0] * rand_tsfac_;
      // update maximum number of time steps
      stepmax_ = (1.0 / rand_tsfac_) * stepmax_ + (1.0 - (1.0 / rand_tsfac_)) * stepn_ + 1;
      // reset timen_ because it is set in the constructor
      timen_ = (*time_)[0] + (*dt_)[0];
      // reset step (e.g. quantities on element level)
      ResetStep();

      return INPAR::STR::conv_success;
    }
    break;
    case INPAR::STR::divcont_adapt_penaltycontact:
    {
      // adapt penalty and search parameter
      if (HaveContactMeshtying())
      {
        cmtbridge_->GetStrategy().ModifyPenalty();
      }
    }
    break;
    case INPAR::STR::divcont_repeat_simulation:
    {
      if (nonlinsoldiv == INPAR::STR::conv_nonlin_fail)
        IO::cout << "Nonlinear solver failed to converge and DIVERCONT = "
                    "repeat_simulation, hence leaving structural time integration "
                 << IO::endl;
      else if (nonlinsoldiv == INPAR::STR::conv_lin_fail)
        IO::cout << "Linear solver failed to converge and DIVERCONT = "
                    "repeat_simulation, hence leaving structural time integration "
                 << IO::endl;
      else if (nonlinsoldiv == INPAR::STR::conv_ele_fail)
        IO::cout << "Element failure in form of a negative Jacobian determinant and DIVERCONT = "
                    "repeat_simulation, hence leaving structural time integration "
                 << IO::endl;
      return nonlinsoldiv;  // so that time loop will be aborted
    }
    break;
    case INPAR::STR::divcont_adapt_3D0Dptc_ele_err:
    {
      // maximal possible refinementlevel
      const int maxdivconrefinementlevel_ptc = 15;
      const int adapt_penaltycontact_after = 7;
      const double sum = 10.0;
      const double fac = 2.0;

      if (divconrefinementlevel_ < (maxdivconrefinementlevel_ptc))
      {
        if (myrank_ == 0)
        {
          if (cardvasc0dman_->Get_k_ptc() == 0.0)
          {
            IO::cout << "Nonlinear solver failed to converge at time t= " << timen_
                     << ". Increase PTC parameter. "
                     << "Old PTC parameter: " << cardvasc0dman_->Get_k_ptc() << IO::endl
                     << "New PTC parameter: " << sum + cardvasc0dman_->Get_k_ptc() << IO::endl
                     << IO::endl;
          }
          else
          {
            IO::cout << "Nonlinear solver failed to converge at time t= " << timen_
                     << ". Increase PTC parameter. "
                     << "Old PTC parameter: " << cardvasc0dman_->Get_k_ptc() << IO::endl
                     << "New PTC parameter: " << fac * cardvasc0dman_->Get_k_ptc() << IO::endl
                     << IO::endl;
          }
        }
        // increase PTC factor
        cardvasc0dman_->Modify_k_ptc(sum, fac);

        // adapt penalty parameter
        if (HaveContactMeshtying() and divconrefinementlevel_ > adapt_penaltycontact_after)
        {
          if (myrank_ == 0)
          {
            IO::cout << "Nonlinear solver still did not converge. Slightly adapt penalty parameter "
                        "for contact."
                     << IO::endl;
          }

          cmtbridge_->GetStrategy().ModifyPenalty();
        }

        divconrefinementlevel_++;
        divconnumfinestep_ = 0;
      }

      else
        dserror(
            "Maximal divercont refinement level reached. Finally nonlinear solver did not "
            "converge. :-(");

      // reset step (e.g. quantities on element level)
      ResetStep();

      return INPAR::STR::conv_success;
    }

    default:
      dserror("Unknown DIVER_CONT case");
      return INPAR::STR::conv_nonlin_fail;
      break;
  }
  return INPAR::STR::conv_success;  // make compiler happy
}

/*----------------------------------------------------------------------*/
/* Set forces due to interface with fluid,
 * the force is expected external-force-like */
void STR::TimInt::SetForceInterface(
    Teuchos::RCP<Epetra_MultiVector> iforce  ///< the force on interface
)
{
  fifc_->Update(1.0, *iforce, 0.0);
}

/*----------------------------------------------------------------------*/
/* apply the new material_displacements        mgit 05/11 / rauch 01/16 */
void STR::TimInt::ApplyDisMat(Teuchos::RCP<Epetra_Vector> dismat)
{
  // The values in dismatn_ are replaced, because the new absolute material
  // displacement is provided in the argument (not an increment)
  CORE::LINALG::Export(*dismat, *dismatn_);
}

/*----------------------------------------------------------------------*/
/* Attach file handle for energy file #energyfile_                      */
void STR::TimInt::AttachEnergyFile()
{
  if (energyfile_.is_null())
  {
    std::string energyname = DRT::Problem::Instance()->OutputControlFile()->FileName() + ".energy";
    energyfile_ = Teuchos::rcp(new std::ofstream(energyname.c_str()));
    (*energyfile_) << "# timestep time total_energy"
                   << " kinetic_energy internal_energy external_energy" << std::endl;
  }
}

/*----------------------------------------------------------------------*/
/* Return (rotatory) transformation matrix of local co-ordinate systems */
Teuchos::RCP<const CORE::LINALG::SparseMatrix> STR::TimInt::GetLocSysTrafo() const
{
  if (locsysman_ != Teuchos::null) return locsysman_->Trafo();

  return Teuchos::null;
}

/*----------------------------------------------------------------------*/
/* Return stiffness matrix as CORE::LINALG::SparseMatrix                      */
Teuchos::RCP<CORE::LINALG::SparseMatrix> STR::TimInt::SystemMatrix()
{
  return Teuchos::rcp_dynamic_cast<CORE::LINALG::SparseMatrix>(stiff_);
}

/*----------------------------------------------------------------------*/
/* Return stiffness matrix as CORE::LINALG::BlockSparseMatrix */
Teuchos::RCP<CORE::LINALG::BlockSparseMatrixBase> STR::TimInt::BlockSystemMatrix()
{
  return Teuchos::rcp_dynamic_cast<CORE::LINALG::BlockSparseMatrixBase>(stiff_);
}

/*----------------------------------------------------------------------*/
/* Return sparse mass matrix                                            */
Teuchos::RCP<CORE::LINALG::SparseMatrix> STR::TimInt::MassMatrix()
{
  return Teuchos::rcp_dynamic_cast<CORE::LINALG::SparseMatrix>(mass_);
}


/*----------------------------------------------------------------------*/
/* Return domain map of mass matrix                                     */
const Epetra_Map& STR::TimInt::DomainMap() const { return mass_->DomainMap(); }

/*----------------------------------------------------------------------*/
/* Creates the field test                                               */
Teuchos::RCP<DRT::ResultTest> STR::TimInt::CreateFieldTest()
{
  return Teuchos::rcp(new StruResultTest(*this));
}

/*----------------------------------------------------------------------*/
/* dof map of vector of unknowns                                        */
Teuchos::RCP<const Epetra_Map> STR::TimInt::DofRowMap()
{
  const Epetra_Map* dofrowmap = discret_->DofRowMap();
  return Teuchos::rcp(new Epetra_Map(*dofrowmap));
}

/*----------------------------------------------------------------------*/
/* dof map of vector of unknowns                                        */
/* new method for multiple dofsets                                      */
Teuchos::RCP<const Epetra_Map> STR::TimInt::DofRowMap(unsigned nds)
{
  const Epetra_Map* dofrowmap = discret_->DofRowMap(nds);
  return Teuchos::rcp(new Epetra_Map(*dofrowmap));
}

/*----------------------------------------------------------------------*/
/* view of dof map of vector of unknowns                                */
const Epetra_Map* STR::TimInt::DofRowMapView() { return discret_->DofRowMap(); }

/*----------------------------------------------------------------------*/
/* reset everything (needed for biofilm simulations)                    */
void STR::TimInt::Reset()
{
  // displacements D_{n}
  dis_ = Teuchos::rcp(new TIMESTEPPING::TimIntMStep<Epetra_Vector>(0, 0, DofRowMapView(), true));
  // displacements D_{n}
  dismat_ = Teuchos::rcp(new TIMESTEPPING::TimIntMStep<Epetra_Vector>(0, 0, DofRowMapView(), true));
  // velocities V_{n}
  vel_ = Teuchos::rcp(new TIMESTEPPING::TimIntMStep<Epetra_Vector>(0, 0, DofRowMapView(), true));
  // accelerations A_{n}
  acc_ = Teuchos::rcp(new TIMESTEPPING::TimIntMStep<Epetra_Vector>(0, 0, DofRowMapView(), true));

  // displacements D_{n+1} at t_{n+1}
  disn_ = CORE::LINALG::CreateVector(*DofRowMapView(), true);
  // velocities V_{n+1} at t_{n+1}
  veln_ = CORE::LINALG::CreateVector(*DofRowMapView(), true);
  // accelerations A_{n+1} at t_{n+1}
  accn_ = CORE::LINALG::CreateVector(*DofRowMapView(), true);
  // create empty interface force vector
  fifc_ = CORE::LINALG::CreateVector(*DofRowMapView(), true);

  // set initial fields
  SetInitialFields();
}

/*----------------------------------------------------------------------*/
/* set structure displacement vector due to biofilm growth              */
void STR::TimInt::SetStrGrDisp(Teuchos::RCP<Epetra_Vector> struct_growth_disp)
{
  strgrdisp_ = struct_growth_disp;
}

/*----------------------------------------------------------------------*/
/* Resize MStep Object due to time adaptivity in FSI                    */
void STR::TimInt::ResizeMStepTimAda()
{
  // safety checks
  CheckIsInit();
  CheckIsSetup();

  // resize time and stepsize fields
  time_->Resize(-1, 0, (*time_)[0]);
  dt_->Resize(-1, 0, (*dt_)[0]);

  // resize state vectors, AB2 is a 2-step method, thus we need two
  // past steps at t_{n} and t_{n-1}
  dis_->Resize(-1, 0, DofRowMapView(), true);
  vel_->Resize(-1, 0, DofRowMapView(), true);
  acc_->Resize(-1, 0, DofRowMapView(), true);
}

/*----------------------------------------------------------------------*/
/* Expand the dbc map by dofs provided in Epetra_Map maptoadd.          */
void STR::TimInt::AddDirichDofs(const Teuchos::RCP<const Epetra_Map> maptoadd)
{
  std::vector<Teuchos::RCP<const Epetra_Map>> condmaps;
  condmaps.push_back(maptoadd);
  condmaps.push_back(GetDBCMapExtractor()->CondMap());
  Teuchos::RCP<Epetra_Map> condmerged = CORE::LINALG::MultiMapExtractor::MergeMaps(condmaps);
  *dbcmaps_ = CORE::LINALG::MapExtractor(*(discret_->DofRowMap()), condmerged);
}

/*----------------------------------------------------------------------*/
/* Contract the dbc map by dofs provided in Epetra_Map maptoremove.     */
void STR::TimInt::RemoveDirichDofs(const Teuchos::RCP<const Epetra_Map> maptoremove)
{
  std::vector<Teuchos::RCP<const Epetra_Map>> othermaps;
  othermaps.push_back(maptoremove);
  othermaps.push_back(GetDBCMapExtractor()->OtherMap());
  Teuchos::RCP<Epetra_Map> othermerged = CORE::LINALG::MultiMapExtractor::MergeMaps(othermaps);
  *dbcmaps_ = CORE::LINALG::MapExtractor(*(discret_->DofRowMap()), othermerged, false);
}

BACI_NAMESPACE_CLOSE

/*----------------------------------------------------------------------*/
/*! \file
\brief Time integration for structural dynamics

\level 1

*/
/*----------------------------------------------------------------------*/

/*----------------------------------------------------------------------*/
/* headers */
#include "4C_structure_timint.hpp"

#include "4C_beamcontact_beam3contact_manager.hpp"
#include "4C_cardiovascular0d_manager.hpp"
#include "4C_comm_utils.hpp"
#include "4C_constraint_manager.hpp"
#include "4C_constraint_solver.hpp"
#include "4C_constraint_springdashpot_manager.hpp"
#include "4C_contact_meshtying_contact_bridge.hpp"
#include "4C_fem_condition_locsys.hpp"
#include "4C_fem_discretization_faces.hpp"
#include "4C_fem_discretization_utils.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_beamcontact.hpp"
#include "4C_inpar_contact.hpp"
#include "4C_inpar_mortar.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_io_gmsh.hpp"
#include "4C_io_pstream.hpp"
#include "4C_linalg_blocksparsematrix.hpp"
#include "4C_linalg_multiply.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_utils_densematrix_communication.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_mat_micromaterial.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mor_pod.hpp"
#include "4C_mortar_manager_base.hpp"
#include "4C_mortar_strategy_base.hpp"
#include "4C_mortar_utils.hpp"
#include "4C_poroelast_utils.hpp"
#include "4C_solid_3D_ele.hpp"
#include "4C_stru_multi_microstatic.hpp"
#include "4C_structure_resulttest.hpp"
#include "4C_structure_timint_genalpha.hpp"

#include <Teuchos_TimeMonitor.hpp>

#include <algorithm>
#include <iostream>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/* print tea time logo */
void Solid::TimInt::logo()
{
  Core::IO::cout << "Welcome to Structural Time Integration " << Core::IO::endl;
  Core::IO::cout << "     __o__                          __o__" << Core::IO::endl;
  Core::IO::cout << "__  /-----\\__                  __  /-----\\__" << Core::IO::endl;
  Core::IO::cout << "\\ \\/       \\ \\    |       \\    \\ \\/       \\ \\" << Core::IO::endl;
  Core::IO::cout << " \\ |  tea  | |    |-------->    \\ |  tea  | |" << Core::IO::endl;
  Core::IO::cout << "  \\|       |_/    |       /      \\|       |_/" << Core::IO::endl;
  Core::IO::cout << "    \\_____/   ._                   \\_____/   ._ _|_ /|" << Core::IO::endl;
  Core::IO::cout << "              | |                            | | |   |" << Core::IO::endl;
  Core::IO::cout << Core::IO::endl;
}

/*----------------------------------------------------------------------*/
/* constructor */
Solid::TimInt::TimInt(const Teuchos::ParameterList& timeparams,
    const Teuchos::ParameterList& ioparams, const Teuchos::ParameterList& sdynparams,
    const Teuchos::ParameterList& xparams, Teuchos::RCP<Core::FE::Discretization> actdis,
    Teuchos::RCP<Core::LinAlg::Solver> solver, Teuchos::RCP<Core::LinAlg::Solver> contactsolver,
    Teuchos::RCP<Core::IO::DiscretizationWriter> output)
    : discret_(actdis),
      facediscret_(Teuchos::null),
      myrank_(actdis->get_comm().MyPID()),
      solver_(solver),
      contactsolver_(contactsolver),
      solveradapttol_(Core::UTILS::IntegralValue<int>(sdynparams, "ADAPTCONV") == 1),
      solveradaptolbetter_(sdynparams.get<double>("ADAPTCONV_BETTER")),
      dbcmaps_(Teuchos::rcp(new Core::LinAlg::MapExtractor())),
      divcontype_(Core::UTILS::IntegralValue<Inpar::Solid::DivContAct>(sdynparams, "DIVERCONT")),
      divconrefinementlevel_(0),
      divconnumfinestep_(0),
      sdynparams_(sdynparams),
      output_(output),
      printscreen_(ioparams.get<int>("STDOUTEVRY")),
      printlogo_(bool(printscreen_)),  // no std out no logo
      printiter_(true),                // ADD INPUT PARAMETER
      outputeveryiter_((bool)Core::UTILS::IntegralValue<int>(ioparams, "OUTPUT_EVERY_ITER")),
      oei_filecounter_(ioparams.get<int>("OEI_FILE_COUNTER")),
      writerestartevery_(timeparams.get<int>("RESTARTEVRY")),
      writeele_((bool)Core::UTILS::IntegralValue<int>(ioparams, "STRUCT_ELE")),
      writestate_((bool)Core::UTILS::IntegralValue<int>(ioparams, "STRUCT_DISP")),
      writevelacc_((bool)Core::UTILS::IntegralValue<int>(ioparams, "STRUCT_VEL_ACC")),
      writeresultsevery_(timeparams.get<int>("RESULTSEVRY")),
      writestress_(Core::UTILS::IntegralValue<Inpar::Solid::StressType>(ioparams, "STRUCT_STRESS")),
      writecouplstress_(
          Core::UTILS::IntegralValue<Inpar::Solid::StressType>(ioparams, "STRUCT_COUPLING_STRESS")),
      writestrain_(Core::UTILS::IntegralValue<Inpar::Solid::StrainType>(ioparams, "STRUCT_STRAIN")),
      writeplstrain_(
          Core::UTILS::IntegralValue<Inpar::Solid::StrainType>(ioparams, "STRUCT_PLASTIC_STRAIN")),
      writeoptquantity_(Core::UTILS::IntegralValue<Inpar::Solid::OptQuantityType>(
          ioparams, "STRUCT_OPTIONAL_QUANTITY")),
      writeenergyevery_(sdynparams.get<int>("RESEVRYERGY")),
      writesurfactant_((bool)Core::UTILS::IntegralValue<int>(ioparams, "STRUCT_SURFACTANT")),
      writerotation_((bool)Core::UTILS::IntegralValue<int>(ioparams, "OUTPUT_ROT")),
      energyfile_(Teuchos::null),
      damping_(Core::UTILS::IntegralValue<Inpar::Solid::DampKind>(sdynparams, "DAMPING")),
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
      lumpmass_(Core::UTILS::IntegralValue<int>(sdynparams, "LUMPMASS") == 1),
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
  // a method that is called during setup() in a base class.

  if (sdynparams.get<int>("OUTPUT_STEP_OFFSET") != 0)
  {
    FOUR_C_THROW(
        "Output step offset (\"OUTPUT_STEP_OFFSET\" != 0) is not supported in the old structural "
        "time integration");
  }
}

/*----------------------------------------------------------------------------------------------*
 * Initialize this class                                                            rauch 09/16 |
 *----------------------------------------------------------------------------------------------*/
void Solid::TimInt::init(const Teuchos::ParameterList& timeparams,
    const Teuchos::ParameterList& sdynparams, const Teuchos::ParameterList& xparams,
    Teuchos::RCP<Core::FE::Discretization> actdis, Teuchos::RCP<Core::LinAlg::Solver> solver)
{
  // invalidate setup
  set_is_setup(false);

  // welcome user
  if ((printlogo_) and (myrank_ == 0))
  {
    logo();
  }

  // check whether discretisation has been completed
  if (not discret_->filled() || not actdis->have_dofs())
  {
    FOUR_C_THROW("Discretisation is not complete or has no dofs!");
  }

  // time state
  time_ = Teuchos::rcp(new TimeStepping::TimIntMStep<double>(
      0, 0, 0.0));  // HERE SHOULD BE SOMETHING LIKE (sdynparams.get<double>("TIMEINIT"))
  dt_ =
      Teuchos::rcp(new TimeStepping::TimIntMStep<double>(0, 0, timeparams.get<double>("TIMESTEP")));
  step_ = 0;
  timen_ = (*time_)[0] + (*dt_)[0];  // set target time to initial time plus step size
  stepn_ = step_ + 1;

  // output file for energy
  if ((writeenergyevery_ != 0) and (myrank_ == 0)) attach_energy_file();

  // initialize constraint manager
  conman_ = Teuchos::rcp(new CONSTRAINTS::ConstrManager());
  conman_->init(discret_, sdynparams_);

  // create stiffness, mass matrix and other fields
  create_fields();

  // stay with us

  // we have successfully initialized this class
  set_is_init(true);
}

/*----------------------------------------------------------------------------------------------*
 * Setup this class                                                                 rauch 09/16 |
 *----------------------------------------------------------------------------------------------*/
void Solid::TimInt::setup()
{
  // we have to call init() before
  check_is_init();

  create_all_solution_vectors();

  // create stiffness, mass matrix and other fields
  create_fields();

  // set initial fields
  set_initial_fields();

  // setup constraint manager
  conman_->setup((*dis_)(0), sdynparams_);

  // model order reduction
  mor_ = Teuchos::rcp(new ModelOrderRed::ProperOrthogonalDecomposition(discret_));

  // initialize 0D cardiovascular manager
  cardvasc0dman_ =
      Teuchos::rcp(new FourC::UTILS::Cardiovascular0DManager(discret_, (*dis_)(0), sdynparams_,
          Global::Problem::instance()->cardiovascular0_d_structural_params(), *solver_, mor_));

  // initialize spring dashpot manager
  springman_ = Teuchos::rcp(new CONSTRAINTS::SpringDashpotManager(discret_));


  // initialize constraint solver if constraints are defined
  if (conman_->have_constraint())
  {
    consolv_ =
        Teuchos::rcp(new CONSTRAINTS::ConstraintSolver(discret_, *solver_, dbcmaps_, sdynparams_));
  }

  // check for beam contact
  {
    // If beam contact (no statistical mechanics) is chosen in the input file, then a
    // corresponding manager object stored via #beamcman_ is created and all relevant
    // stuff is initialized. Else, #beamcman_ remains a Teuchos::null pointer.
    prepare_beam_contact(sdynparams_);
  }
  // check for mortar contact or meshtying
  {
    // If mortar contact or meshtying is chosen in the input file, then a
    // corresponding manager object stored via #cmtman_ is created and all relevant
    // stuff is initialized. Else, #cmtman_ remains a Teuchos::null pointer.
    prepare_contact_meshtying(sdynparams_);
  }

  // check whether we have locsys BCs and create LocSysManager if so
  // after checking
  {
    std::vector<Core::Conditions::Condition*> locsysconditions(0);
    discret_->get_condition("Locsys", locsysconditions);
    if (locsysconditions.size())
    {
      locsysman_ = Teuchos::rcp(
          new Core::Conditions::LocsysManager(*discret_, Global::Problem::instance()->n_dim()));
      // in case we have no time dependent locsys conditions in our problem,
      // this is the only time where the whole setup routine is conducted.
      locsysman_->update(-1.0, {}, Global::Problem::instance()->function_manager());
    }
  }

  // check if we have elements with a micro-material
  havemicromat_ = false;
  for (int i = 0; i < discret_->num_my_col_elements(); i++)
  {
    Core::Elements::Element* actele = discret_->l_col_element(i);
    Teuchos::RCP<Core::Mat::Material> mat = actele->material();
    if (mat != Teuchos::null && mat->material_type() == Core::Materials::m_struct_multiscale)
    {
      havemicromat_ = true;
      break;
    }
  }


  // Check for porosity dofs within the structure and build a map extractor if necessary
  porositysplitter_ = PoroElast::UTILS::BuildPoroSplitter(discret_);


  // we have successfully set up this class
  set_is_setup(true);
}

/*----------------------------------------------------------------------------------------------*
 * Create all solution vectors
 *----------------------------------------------------------------------------------------------*/
void Solid::TimInt::create_all_solution_vectors()
{
  // displacements D_{n}
  dis_ = Teuchos::rcp(new TimeStepping::TimIntMStep<Epetra_Vector>(0, 0, dof_row_map_view(), true));
  // velocities V_{n}
  vel_ = Teuchos::rcp(new TimeStepping::TimIntMStep<Epetra_Vector>(0, 0, dof_row_map_view(), true));
  // accelerations A_{n}
  acc_ = Teuchos::rcp(new TimeStepping::TimIntMStep<Epetra_Vector>(0, 0, dof_row_map_view(), true));

  // displacements D_{n+1} at t_{n+1}
  disn_ = Core::LinAlg::CreateVector(*dof_row_map_view(), true);

  if ((Global::Problem::instance()->get_problem_type() == Core::ProblemType::struct_ale and
          (Global::Problem::instance()->wear_params()).get<double>("WEARCOEFF") > 0.0))
  {
    // material displacements Dm_{n+1} at t_{n+1}
    dismatn_ = Core::LinAlg::CreateVector(*dof_row_map_view(), true);

    // material_displacements D_{n}
    dismat_ =
        Teuchos::rcp(new TimeStepping::TimIntMStep<Epetra_Vector>(0, 0, dof_row_map_view(), true));
  }

  // velocities V_{n+1} at t_{n+1}
  veln_ = Core::LinAlg::CreateVector(*dof_row_map_view(), true);
  // accelerations A_{n+1} at t_{n+1}
  accn_ = Core::LinAlg::CreateVector(*dof_row_map_view(), true);
  // create empty interface force vector
  fifc_ = Core::LinAlg::CreateVector(*dof_row_map_view(), true);
}

/*-------------------------------------------------------------------------------------------*
 * Create matrices when setting up time integrator
 *-------------------------------------------------------------------------------------------*/
void Solid::TimInt::create_fields()
{
  // a zero vector of full length
  zeros_ = Core::LinAlg::CreateVector(*dof_row_map_view(), true);

  // Map containing Dirichlet DOFs
  {
    Teuchos::ParameterList p;
    p.set("total time", timen_);
    p.set<const Core::UTILS::FunctionManager*>(
        "function_manager", &Global::Problem::instance()->function_manager());

    // if this is a NURBS discretization, get the solver parameters for solving the least squares
    // problem
    if (Teuchos::rcp_dynamic_cast<Core::FE::Nurbs::NurbsDiscretization>(discret_) != Teuchos::null)
    {
      const Teuchos::ParameterList& nurbs_params = Global::Problem::instance()->nurbs_params();

      const bool is_ls_dbc_needed =
          Core::UTILS::IntegralValue<bool>(nurbs_params, "DO_LS_DBC_PROJECTION");

      if (is_ls_dbc_needed)
      {
        const int ls_dbc_solver_num = nurbs_params.get<int>("SOLVER_LS_DBC_PROJECTION");

        if (ls_dbc_solver_num == (-1))
          FOUR_C_THROW(
              "No linear solver defined for the projection of least squares Dirichlet "
              "boundary conditions for the NURBS discretization. Please set "
              "SOLVER_LS_DBC_PROJECTION in NURBS to a valid number!");

        // Save solver parameters
        p.sublist("ls_dbc_solver_params")
            .setParameters(Global::Problem::instance()->solver_params(ls_dbc_solver_num));
      }
    }

    discret_->evaluate_dirichlet(p, zeros_, Teuchos::null, Teuchos::null, Teuchos::null, dbcmaps_);
    zeros_->PutScalar(0.0);  // just in case of change
  }

  // create empty matrices
  stiff_ = Teuchos::rcp(new Core::LinAlg::SparseMatrix(*dof_row_map_view(), 81, false, true));
  mass_ = Teuchos::rcp(new Core::LinAlg::SparseMatrix(*dof_row_map_view(), 81, false, true));
  if (damping_ != Inpar::Solid::damp_none)
  {
    if (have_nonlinear_mass() == Inpar::Solid::ml_none)
    {
      damp_ = Teuchos::rcp(new Core::LinAlg::SparseMatrix(*dof_row_map_view(), 81, false, true));
    }
    else
    {
      // Since our element evaluate routine is only designed for two input matrices
      //(stiffness and damping or stiffness and mass) its not possible, to have nonlinear
      // inertia forces AND material damping.
      FOUR_C_THROW("So far its not possible to model nonlinear inertia forces and damping!");
    }
  }
}

/*----------------------------------------------------------------------*/
/* Set intitial fields in structure (e.g. initial velocities */
void Solid::TimInt::set_initial_fields()
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
  discret_->evaluate_initial_field(
      Global::Problem::instance()->function_manager(), field, (*vel_)(0), localdofs);

  // set initial porosity field if existing
  const std::string porosityfield = "Porosity";
  std::vector<int> porositylocaldofs;
  porositylocaldofs.push_back(Global::Problem::instance()->n_dim());

  discret_->evaluate_initial_field(Global::Problem::instance()->function_manager(), porosityfield,
      (*dis_)(0), porositylocaldofs);
}

/*----------------------------------------------------------------------*/
/* Check for beam contact and do preparations */
void Solid::TimInt::prepare_beam_contact(const Teuchos::ParameterList& sdynparams)
{
  // some parameters
  const Teuchos::ParameterList& beamcontact = Global::Problem::instance()->beam_contact_params();
  Inpar::BEAMCONTACT::Strategy strategy =
      Core::UTILS::IntegralValue<Inpar::BEAMCONTACT::Strategy>(beamcontact, "BEAMS_STRATEGY");

  // conditions for potential-based beam interaction
  std::vector<Core::Conditions::Condition*> beampotconditions(0);
  discret_->get_condition("BeamPotentialLineCharge", beampotconditions);

  // only continue if beam contact unmistakably chosen in input file or beam potential conditions
  // applied
  if (strategy != Inpar::BEAMCONTACT::bstr_none or (int) beampotconditions.size() != 0)
  {
    // store integration parameter alphaf into beamcman_ as well
    // (for all cases except OST, GenAlpha and GEMM this is zero)
    // (note that we want to hand in theta in the OST case, which
    // is defined just the other way round as alphaf in GenAlpha schemes.
    // Thus, we have to hand in 1.0-theta for OST!!!)
    double alphaf = tim_int_param();

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
void Solid::TimInt::prepare_contact_meshtying(const Teuchos::ParameterList& sdynparams)
{
  TEUCHOS_FUNC_TIME_MONITOR("Solid::TimInt::prepare_contact_meshtying");

  // some parameters
  const Teuchos::ParameterList& smortar = Global::Problem::instance()->mortar_coupling_params();
  const Teuchos::ParameterList& scontact = Global::Problem::instance()->contact_dynamic_params();
  Inpar::Mortar::ShapeFcn shapefcn =
      Core::UTILS::IntegralValue<Inpar::Mortar::ShapeFcn>(smortar, "LM_SHAPEFCN");
  Inpar::CONTACT::SolvingStrategy soltype =
      Core::UTILS::IntegralValue<Inpar::CONTACT::SolvingStrategy>(scontact, "STRATEGY");
  Inpar::CONTACT::SystemType systype =
      Core::UTILS::IntegralValue<Inpar::CONTACT::SystemType>(scontact, "SYSTEM");
  Inpar::Mortar::AlgorithmType algorithm =
      Core::UTILS::IntegralValue<Inpar::Mortar::AlgorithmType>(smortar, "ALGORITHM");

  // check mortar contact or meshtying conditions
  std::vector<Core::Conditions::Condition*> mortarconditions(0);
  std::vector<Core::Conditions::Condition*> contactconditions(0);

  discret_->get_condition("Mortar", mortarconditions);
  discret_->get_condition("Contact", contactconditions);

  // double-check for contact/meshtying conditions
  if (mortarconditions.size() == 0 and contactconditions.size() == 0) return;

  // check if only beam-to-solid contact / meshtying conditions (and leave if so)
  bool realcontactconditions = false;
  for (const auto& contactCondition : contactconditions)
  {
    if (contactCondition->parameters().get<std::string>("Application") != "Beamtosolidcontact" &&
        contactCondition->parameters().get<std::string>("Application") != "Beamtosolidmeshtying")
      realcontactconditions = true;
  }
  if (mortarconditions.size() == 0 and !realcontactconditions) return;

  // store integration parameter alphaf into cmtman_ as well
  // (for all cases except OST, GenAlpha and GEMM this is zero)
  // (note that we want to hand in theta in the OST case, which
  // is defined just the other way round as alphaf in GenAlpha schemes.
  // Thus, we have to hand in 1-theta for OST!!!)
  double time_integration_factor = 0.0;
  const bool do_endtime = Core::UTILS::IntegralValue<int>(scontact, "CONTACTFORCE_ENDTIME");
  if (!do_endtime) time_integration_factor = tim_int_param();

  // create instance for meshtying contact bridge
  cmtbridge_ = Teuchos::rcp(new CONTACT::MeshtyingContactBridge(
      *discret_, mortarconditions, contactconditions, time_integration_factor));

  cmtbridge_->store_dirichlet_status(dbcmaps_);
  cmtbridge_->set_state(zeros_);

  // contact and constraints together not yet implemented
  if (conman_->have_constraint())
    FOUR_C_THROW("Constraints and contact cannot be treated at the same time yet");

  // print messages for multifield problems (e.g FSI)
  const Core::ProblemType probtype = Global::Problem::instance()->get_problem_type();
  const std::string probname = Global::Problem::instance()->problem_name();
  if (probtype != Core::ProblemType::structure && !myrank_)
  {
    // warnings
#ifdef CONTACTPSEUDO2D
    std::cout << "WARNING: The flag CONTACTPSEUDO2D is switched on. If this "
              << "is a real 3D problem, switch it off!" << std::endl;
#else
    std::cout << "WARNING: The flag CONTACTPSEUDO2D is switched off. If this "
              << "is a 2D problem modeled pseudo-3D, switch it on!" << std::endl;
#endif
  }

  // initialization of meshtying
  if (cmtbridge_->have_meshtying())
  {
    // FOR MESHTYING (ONLY ONCE), NO FUNCTIONALITY FOR CONTACT CASES
    // (1) do mortar coupling in reference configuration
    cmtbridge_->mt_manager()->get_strategy().mortar_coupling(zeros_);

    // perform mesh initialization if required by input parameter MESH_RELOCATION
    auto mesh_relocation_parameter = Core::UTILS::IntegralValue<Inpar::Mortar::MeshRelocation>(
        Global::Problem::instance()->mortar_coupling_params(), "MESH_RELOCATION");

    if (mesh_relocation_parameter == Inpar::Mortar::relocation_initial)
    {
      // (2) perform mesh initialization for rotational invariance (interface)
      // and return the modified slave node positions in vector Xslavemod
      Teuchos::RCP<const Epetra_Vector> Xslavemod =
          cmtbridge_->mt_manager()->get_strategy().mesh_initialization();

      // (3) apply result of mesh initialization to underlying problem discretization
      apply_mesh_initialization(Xslavemod);
    }
    else if (mesh_relocation_parameter == Inpar::Mortar::relocation_timestep)
    {
      FOUR_C_THROW(
          "Meshtying with MESH_RELOCATION every_timestep not permitted. Change to MESH_RELOCATION "
          "initial or MESH_RELOCATION no.");
    }
  }

  // initialization of contact
  if (cmtbridge_->have_contact())
  {
    // FOR PENALTY CONTACT (ONLY ONCE), NO FUNCTIONALITY FOR OTHER CASES
    // (1) Explicitly store gap-scaling factor kappa
    cmtbridge_->contact_manager()->get_strategy().save_reference_state(zeros_);

    // FOR CONTACT FORMULATIONS (ONLY ONCE)
    // (1) Evaluate reference state for friction and initialize gap
    cmtbridge_->contact_manager()->get_strategy().evaluate_reference_state();
  }

  // visualization of initial configuration
#ifdef MORTARGMSH3
  bool gmsh =
      Core::UTILS::IntegralValue<int>(Global::Problem::instance()->IOParams(), "OUTPUT_GMSH");
  if (gmsh) cmtbridge_->VisualizeGmsh(0);
#endif  // #ifdef MORTARGMSH3

  //**********************************************************************
  // prepare solvers for contact/meshtying problem
  //**********************************************************************
  {
    // only plausibility check, that a contact solver is available
    if (contactsolver_ == Teuchos::null)
      FOUR_C_THROW("No contact solver in Solid::TimInt::prepare_contact_meshtying? Cannot be!");
  }

  // output of strategy / shapefcn / system type to screen
  {
    // output
    if (!myrank_)
    {
      if (algorithm == Inpar::Mortar::algorithm_mortar)
      {
        // saddle point formulation
        if (systype == Inpar::CONTACT::system_saddlepoint)
        {
          if (soltype == Inpar::CONTACT::solution_lagmult &&
              shapefcn == Inpar::Mortar::shape_standard)
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
          else if (soltype == Inpar::CONTACT::solution_lagmult &&
                   shapefcn == Inpar::Mortar::shape_dual)
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
          else if (soltype == Inpar::CONTACT::solution_multiscale &&
                   shapefcn == Inpar::Mortar::shape_standard)
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
          else if (soltype == Inpar::CONTACT::solution_multiscale &&
                   shapefcn == Inpar::Mortar::shape_dual)
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
          else if (soltype == Inpar::CONTACT::solution_lagmult &&
                   Core::UTILS::IntegralValue<Inpar::Mortar::LagMultQuad>(smortar, "LM_QUAD") ==
                       Inpar::Mortar::lagmult_const)
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
          else if (soltype == Inpar::CONTACT::solution_lagmult &&
                   shapefcn == Inpar::Mortar::shape_petrovgalerkin)
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
          else if (soltype == Inpar::CONTACT::solution_penalty &&
                   shapefcn == Inpar::Mortar::shape_standard)
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
          else if (soltype == Inpar::CONTACT::solution_penalty &&
                   shapefcn == Inpar::Mortar::shape_dual)
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
          else if (soltype == Inpar::CONTACT::solution_uzawa &&
                   shapefcn == Inpar::Mortar::shape_standard)
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
          else if (soltype == Inpar::CONTACT::solution_uzawa &&
                   shapefcn == Inpar::Mortar::shape_dual)
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
            FOUR_C_THROW("Invalid strategy or shape function type for contact/meshtying");
        }

        // condensed formulation
        else if (systype == Inpar::CONTACT::system_condensed ||
                 systype == Inpar::CONTACT::system_condensed_lagmult)
        {
          if (soltype == Inpar::CONTACT::solution_lagmult && shapefcn == Inpar::Mortar::shape_dual)
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
          else if (soltype == Inpar::CONTACT::solution_lagmult &&
                   shapefcn == Inpar::Mortar::shape_petrovgalerkin)
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
          else if (soltype == Inpar::CONTACT::solution_lagmult &&
                   Core::UTILS::IntegralValue<Inpar::Mortar::LagMultQuad>(smortar, "LM_QUAD") ==
                       Inpar::Mortar::lagmult_const)
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
          else if (soltype == Inpar::CONTACT::solution_multiscale &&
                   shapefcn == Inpar::Mortar::shape_standard)
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
          else if (soltype == Inpar::CONTACT::solution_multiscale &&
                   shapefcn == Inpar::Mortar::shape_dual)
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
          else if (soltype == Inpar::CONTACT::solution_penalty &&
                   shapefcn == Inpar::Mortar::shape_standard)
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
          else if (soltype == Inpar::CONTACT::solution_penalty &&
                   shapefcn == Inpar::Mortar::shape_dual)
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
          else if (soltype == Inpar::CONTACT::solution_uzawa &&
                   shapefcn == Inpar::Mortar::shape_standard)
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
          else if (soltype == Inpar::CONTACT::solution_uzawa &&
                   shapefcn == Inpar::Mortar::shape_dual)
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
            FOUR_C_THROW("Invalid strategy or shape function type for contact/meshtying");
        }
      }
      else if (algorithm == Inpar::Mortar::algorithm_nts)
      {
        std::cout << "================================================================"
                  << std::endl;
        std::cout << "===== Node-To-Segment approach ================================="
                  << std::endl;
        std::cout << "================================================================\n"
                  << std::endl;
      }
      else if (algorithm == Inpar::Mortar::algorithm_lts)
      {
        std::cout << "================================================================"
                  << std::endl;
        std::cout << "===== Line-To-Segment approach ================================="
                  << std::endl;
        std::cout << "================================================================\n"
                  << std::endl;
      }
      else if (algorithm == Inpar::Mortar::algorithm_ltl)
      {
        std::cout << "================================================================"
                  << std::endl;
        std::cout << "===== Line-To-Line approach ===================================="
                  << std::endl;
        std::cout << "================================================================\n"
                  << std::endl;
      }
      else if (algorithm == Inpar::Mortar::algorithm_stl)
      {
        std::cout << "================================================================"
                  << std::endl;
        std::cout << "===== Segment-To-Line approach ================================="
                  << std::endl;
        std::cout << "================================================================\n"
                  << std::endl;
      }
      else if (algorithm == Inpar::Mortar::algorithm_gpts)
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
        FOUR_C_THROW("Invalid system type for contact/meshtying");
    }
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Solid::TimInt::apply_mesh_initialization(Teuchos::RCP<const Epetra_Vector> Xslavemod)
{
  // check modified positions vector
  if (Xslavemod == Teuchos::null) return;

  // create fully overlapping slave node map
  Teuchos::RCP<Epetra_Map> slavemap =
      cmtbridge_->mt_manager()->get_strategy().slave_row_nodes_ptr();
  Teuchos::RCP<Epetra_Map> allreduceslavemap = Core::LinAlg::AllreduceEMap(*slavemap);

  // export modified node positions to column map of problem discretization
  Teuchos::RCP<Epetra_Vector> Xslavemodcol =
      Core::LinAlg::CreateVector(*discret_->dof_col_map(), false);
  Core::LinAlg::export_to(*Xslavemod, *Xslavemodcol);

  const int numnode = allreduceslavemap->NumMyElements();
  const int numdim = Global::Problem::instance()->n_dim();
  const Epetra_Vector& gvector = *Xslavemodcol;

  // loop over all slave nodes (for all procs)
  for (int index = 0; index < numnode; ++index)
  {
    int gid = allreduceslavemap->GID(index);

    // only do someting for nodes in my column map
    int ilid = discret_->node_col_map()->LID(gid);
    if (ilid < 0) continue;

    Core::Nodes::Node* mynode = discret_->g_node(gid);

    // get degrees of freedom associated with this fluid/structure node
    std::vector<int> nodedofs = discret_->dof(0, mynode);
    std::vector<double> nvector(3, 0.0);

    // create new position vector
    for (int i = 0; i < numdim; ++i)
    {
      const int lid = gvector.Map().LID(nodedofs[i]);

      if (lid < 0)
        FOUR_C_THROW(
            "Proc %d: Cannot find gid=%d in Epetra_Vector", gvector.Comm().MyPID(), nodedofs[i]);

      nvector[i] += gvector[lid];
    }

    // set new reference position
    mynode->set_pos(nvector);
  }

  // re-initialize finite elements
  Core::Communication::ParObjectFactory::instance().initialize_elements(*discret_);
}

/*----------------------------------------------------------------------*/
/* Prepare contact for new time step */
void Solid::TimInt::prepare_step_contact()
{
  // just do something here if contact is present
  if (have_contact_meshtying())
  {
    if (cmtbridge_->have_contact())
    {
      cmtbridge_->get_strategy().inttime_init();
      cmtbridge_->get_strategy().redistribute_contact((*dis_)(0), (*vel_)(0));
    }
  }
}

/*----------------------------------------------------------------------*/
/* things that should be done after the actual time loop is finished */
void Solid::TimInt::post_time_loop()
{
  if (have_micro_mat())
  {
    // stop supporting processors in multi scale simulations
    MultiScale::stop_np_multiscale();
  }
}

/*----------------------------------------------------------------------*/
/* equilibrate system at initial state
 * and identify consistent accelerations */
void Solid::TimInt::determine_mass_damp_consist_accel()
{
  // temporary right hand sinde vector in this routing
  Teuchos::RCP<Epetra_Vector> rhs =
      Core::LinAlg::CreateVector(*dof_row_map_view(), true);  // right hand side
  // temporary force vectors in this routine
  Teuchos::RCP<Epetra_Vector> fext =
      Core::LinAlg::CreateVector(*dof_row_map_view(), true);  // external force
  Teuchos::RCP<Epetra_Vector> fint =
      Core::LinAlg::CreateVector(*dof_row_map_view(), true);  // internal force

  // initialise matrices
  stiff_->zero();
  mass_->zero();

  // auxiliary vector in order to store accelerations of inhomogeneous Dirichilet-DoFs
  // Meier 2015: This contribution is necessary in order to determine correct initial
  // accelerations in case of inhomogeneous Dirichlet conditions
  Teuchos::RCP<Epetra_Vector> acc_aux = Core::LinAlg::CreateVector(*dof_row_map_view(), true);
  acc_aux->PutScalar(0.0);

  // overwrite initial state vectors with DirichletBCs
  apply_dirichlet_bc((*time_)[0], (*dis_)(0), (*vel_)(0), acc_aux, false);

  /* get external force (no linearization since we assume Rayleigh damping
   * to be independent of follower loads) */
  apply_force_external((*time_)[0], (*dis_)(0), disn_, (*vel_)(0), fext);

  // get initial internal force and stiffness and mass
  {
    // compute new inner radius
    discret_->clear_state();
    discret_->set_state(0, "displacement", (*dis_)(0));

    // for structure ale
    if (dismat_ != Teuchos::null) discret_->set_state(0, "material_displacement", (*dismat_)(0));

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
    discret_->clear_state();
    // extended set_state(0,...) in case of multiple dofsets (e.g. TSI)
    discret_->set_state(0, "residual displacement", zeros_);
    discret_->set_state(0, "displacement", (*dis_)(0));
    discret_->set_state(0, "velocity", (*vel_)(0));

    // The acceleration is only used as a dummy here and should not be applied inside an element,
    // since this is not the consistent initial acceleration vector which will be determined later
    // on
    discret_->set_state(0, "acceleration", acc_aux);

    if (damping_ == Inpar::Solid::damp_material) discret_->set_state(0, "velocity", (*vel_)(0));

    // for structure ale
    if (dismat_ != Teuchos::null) discret_->set_state(0, "material_displacement", (*dismat_)(0));

    discret_->evaluate(p, stiff_, mass_, fint, Teuchos::null, fintn_str_);
    discret_->clear_state();
  }

  // finish mass matrix
  mass_->complete();

  // close stiffness matrix
  stiff_->complete();

  // build Rayleigh damping matrix if desired
  if (damping_ == Inpar::Solid::damp_rayleigh)
  {
    damp_->add(*stiff_, false, dampk_, 0.0);
    damp_->add(*mass_, false, dampm_, 1.0);
    damp_->complete();
  }

  // in case of C0 pressure field, we need to get rid of
  // pressure equations
  Teuchos::RCP<Core::LinAlg::SparseOperator> mass = Teuchos::null;
  // Meier 2015: Here, we apply a deep copy in order to not perform the Dirichlet conditions on the
  // constant matrix mass_ later on. This is necessary since we need the original mass matrix mass_
  // (without blanked rows) on the Dirichlet DoFs in order to calculate correct reaction forces
  // (Christoph Meier)
  mass = Teuchos::rcp(new Core::LinAlg::SparseMatrix(*mass_matrix(), Core::LinAlg::Copy));

  /* calculate consistent initial accelerations
   * WE MISS:
   *   - surface stress forces
   *   - potential forces
   *   - linearization of follower loads
   */
  {
    // Contribution to rhs due to damping forces
    if (damping_ == Inpar::Solid::damp_rayleigh)
    {
      damp_->multiply(false, (*vel_)[0], *rhs);
    }

    // add initial forces due to 0D cardiovascular for consistent initial acceleration calculation!
    // needed in case of initial ventricular pressures != 0
    Teuchos::ParameterList pwindk;
    if (cardvasc0dman_->have_cardiovascular0_d())
    {
      pwindk.set("scale_timint", 1.0);
      pwindk.set("time_step_size", (*dt_)[0]);
      cardvasc0dman_->evaluate_force_stiff((*time_)[0], (*dis_)(0), fint, stiff_, pwindk);
    }

    // Contribution to rhs due to internal and external forces
    rhs->Update(-1.0, *fint, 1.0, *fext, -1.0);

    // Contribution to rhs due to beam contact
    if (have_beam_contact())
    {
      // create empty parameter list
      Teuchos::ParameterList beamcontactparams;
      beamcontactparams.set("iter", 0);
      beamcontactparams.set("dt", (*dt_)[0]);
      beamcontactparams.set("numstep", step_);
      beamcman_->evaluate(*system_matrix(), *rhs, (*dis_)[0], beamcontactparams, true, timen_);
    }

    // Contribution to rhs due to inertia forces of inhomogeneous Dirichlet conditions
    Teuchos::RCP<Epetra_Vector> finert0 = Core::LinAlg::CreateVector(*dof_row_map_view(), true);
    finert0->PutScalar(0.0);
    mass_->multiply(false, *acc_aux, *finert0);
    rhs->Update(-1.0, *finert0, 1.0);

    // blank RHS and system matrix on DBC DOFs
    dbcmaps_->insert_cond_vector(dbcmaps_->extract_cond_vector(zeros_), rhs);

    // Apply Dirichlet conditions also to mass matrix (which represents the system matrix of
    // the considered linear system of equations)
    mass->apply_dirichlet(*(dbcmaps_->cond_map()));

    if (pressure_ != Teuchos::null)
    {
      pressure_->insert_cond_vector(pressure_->extract_cond_vector(zeros_), rhs);
      mass->apply_dirichlet(*(pressure_->cond_map()));
    }
    if (porositysplitter_ != Teuchos::null)
    {
      porositysplitter_->insert_cond_vector(porositysplitter_->extract_cond_vector(zeros_), rhs);
      mass->apply_dirichlet(*(porositysplitter_->cond_map()));
    }

    // Meier 2015: Due to the Dirichlet conditions applied to the mass matrix, we solely solve
    // for the accelerations at non-Dirichlet DoFs while the resulting accelerations at the
    // Dirichlet-DoFs will be zero. Therefore, the accelerations at DoFs with inhomogeneous
    // Dirichlet conditions will be added below at *).
    Core::LinAlg::SolverParams solver_params;
    solver_params.refactor = true;
    solver_params.reset = true;
    solver_->solve(mass->epetra_operator(), (*acc_)(0), rhs, solver_params);

    //*) Add contributions of inhomogeneous DBCs
    (*acc_)(0)->Update(1.0, *acc_aux, 1.0);
  }

  // We need to reset the stiffness matrix because its graph (topology)
  // is not finished yet in case of constraints and possibly other side
  // effects (basically managers).
  stiff_->reset();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Solid::TimInt::determine_mass()
{
  FOUR_C_THROW(
      "(Re-)Evaluation of only the mass matrix and intertial forces is "
      "not implemented in the base class.\n Set 'MASSLIN' to 'No' in "
      "--STRUCTURAL DYNAMIC if you want to use the chosen timint scheme.");
}

/*---------------------------------------------------------------*/
/* Apply Dirichlet boundary conditions on provided state vectors */
void Solid::TimInt::apply_dirichlet_bc(const double time, Teuchos::RCP<Epetra_Vector> dis,
    Teuchos::RCP<Epetra_Vector> vel, Teuchos::RCP<Epetra_Vector> acc, bool recreatemap)
{
  // In the case of local coordinate systems, we have to rotate forward ...
  // --------------------------------------------------------------------------------
  if (locsysman_ != Teuchos::null)
  {
    if (dis != Teuchos::null) locsysman_->rotate_global_to_local(dis, true);
    if (vel != Teuchos::null) locsysman_->rotate_global_to_local(vel);
    if (acc != Teuchos::null) locsysman_->rotate_global_to_local(acc);
  }

  // Apply DBCs
  // --------------------------------------------------------------------------------
  // needed parameters
  Teuchos::ParameterList p;
  p.set("total time", time);  // target time
  p.set<const Core::UTILS::FunctionManager*>(
      "function_manager", &Global::Problem::instance()->function_manager());

  // if this is a NURBS discretization, get the solver parameters for solving the least squares
  // problem
  if (Teuchos::rcp_dynamic_cast<Core::FE::Nurbs::NurbsDiscretization>(discret_) != Teuchos::null)
  {
    const Teuchos::ParameterList& nurbs_params = Global::Problem::instance()->nurbs_params();

    const bool is_ls_dbc_needed =
        Core::UTILS::IntegralValue<bool>(nurbs_params, "DO_LS_DBC_PROJECTION");

    if (is_ls_dbc_needed)
    {
      const int ls_dbc_solver_num = nurbs_params.get<int>("SOLVER_LS_DBC_PROJECTION");

      if (ls_dbc_solver_num == (-1))
        FOUR_C_THROW(
            "No linear solver defined for the projection of least squares Dirichlet "
            "boundary conditions for the NURBS discretization. Please set "
            "SOLVER_LS_DBC_PROJECTION in NURBS to a valid number!");

      // Save solver parameters
      p.sublist("ls_dbc_solver_params")
          .setParameters(Global::Problem::instance()->solver_params(ls_dbc_solver_num));
    }
  }

  // predicted Dirichlet values
  // \c dis then also holds prescribed new Dirichlet displacements
  discret_->clear_state();
  if (recreatemap)
  {
    discret_->evaluate_dirichlet(p, dis, vel, acc, Teuchos::null, dbcmaps_);
  }
  else
  {
    discret_->evaluate_dirichlet(p, dis, vel, acc, Teuchos::null, Teuchos::null);
  }
  discret_->clear_state();

  // In the case of local coordinate systems, we have to rotate back into global Cartesian frame
  // --------------------------------------------------------------------------------
  if (locsysman_ != Teuchos::null)
  {
    if (dis != Teuchos::null) locsysman_->rotate_local_to_global(dis, true);
    if (vel != Teuchos::null) locsysman_->rotate_local_to_global(vel);
    if (acc != Teuchos::null) locsysman_->rotate_local_to_global(acc);
  }
}

/*----------------------------------------------------------------------*/
/* Update time and step counter */
void Solid::TimInt::update_step_time()
{
  // update time and step
  time_->update_steps(timen_);  // t_{n} := t_{n+1}, etc
  step_ = stepn_;               // n := n+1
  //
  timen_ += (*dt_)[0];
  stepn_ += 1;
}

/*----------------------------------------------------------------------*/
/* Update contact and meshtying */
void Solid::TimInt::update_step_contact_meshtying()
{
  if (have_contact_meshtying())
  {
    cmtbridge_->update(disn_);
#ifdef MORTARGMSH1
    bool gmsh =
        Core::UTILS::IntegralValue<int>(Global::Problem::instance()->IOParams(), "OUTPUT_GMSH");
    if (gmsh) cmtbridge_->VisualizeGmsh(stepn_);
#endif  // #ifdef MORTARGMSH1
  }
}

/*----------------------------------------------------------------------*/
/* Update beam contact */
void Solid::TimInt::update_step_beam_contact()
{
  if (have_beam_contact()) beamcman_->update(*disn_, stepn_, 99);
}

/*----------------------------------------------------------------------*/
/* Velocity update method (VUM) for contact */
void Solid::TimInt::update_step_contact_vum()
{
  if (have_contact_meshtying())
  {
    bool do_vum =
        Core::UTILS::IntegralValue<int>(cmtbridge_->get_strategy().params(), "VELOCITY_UPDATE");

    //********************************************************************
    // VELOCITY UPDATE METHOD
    //********************************************************************
    if (do_vum)
    {
      // check for actual contact and leave if active set empty
      bool isincontact = cmtbridge_->get_strategy().is_in_contact();
      if (!isincontact) return;

      // check for contact force evaluation
      bool do_end = Core::UTILS::IntegralValue<int>(
          cmtbridge_->get_strategy().params(), "CONTACTFORCE_ENDTIME");
      if (do_end == false)
      {
        FOUR_C_THROW(
            "***** WARNING: VelUpdate ONLY for contact force evaluated at the end time -> skipping "
            "****");
        return;
      }

      // parameter list
      const Teuchos::ParameterList& sdynparams =
          Global::Problem::instance()->structural_dynamic_params();

      // time integration parameter
      double alpham = 0.0;
      double beta = 0.0;
      double gamma = 0.0;
      if (Core::UTILS::IntegralValue<Inpar::Solid::DynamicType>(sdynparams, "DYNAMICTYP") ==
          Inpar::Solid::dyna_genalpha)
      {
        auto genAlpha = dynamic_cast<Solid::TimIntGenAlpha*>(this);
        alpham = genAlpha->tim_int_param_alpham();
        beta = genAlpha->tim_int_param_beta();
        gamma = genAlpha->tim_int_param_gamma();
      }
      else
      {
        FOUR_C_THROW("***** WARNING: VelUpdate ONLY for Gen-alpha -> skipping ****");
        return;
      }

      // the four velocity update constants
      double R1 = 2 * (alpham - 1) / (gamma * (*dt_)[0]);
      double R2 = (1 - alpham) / gamma;
      double R3 = (*dt_)[0] * (1 - 2 * beta - alpham) / (2 * gamma);
      double R4 = beta * (alpham - 1) / pow(gamma, 2);

      // maps
      const Epetra_Map* dofmap = discret_->dof_row_map();
      Teuchos::RCP<Epetra_Map> activenodemap = cmtbridge_->get_strategy().active_row_nodes();
      Teuchos::RCP<Epetra_Map> slavenodemap = cmtbridge_->get_strategy().slave_row_nodes_ptr();
      Teuchos::RCP<Epetra_Map> notredistslavedofmap =
          cmtbridge_->get_strategy().non_redist_slave_row_dofs();
      Teuchos::RCP<Epetra_Map> notredistmasterdofmap =
          cmtbridge_->get_strategy().non_redist_master_row_dofs();
      Teuchos::RCP<Epetra_Map> notactivenodemap =
          Core::LinAlg::SplitMap(*slavenodemap, *activenodemap);

      // the lumped mass matrix and its inverse
      if (lumpmass_ == false)
      {
        FOUR_C_THROW("***** WARNING: VelUpdate ONLY for lumped mass matrix -> skipping ****");
        return;
      }
      Teuchos::RCP<Core::LinAlg::SparseMatrix> Mass =
          Teuchos::rcp_dynamic_cast<Core::LinAlg::SparseMatrix>(mass_);
      Teuchos::RCP<Core::LinAlg::SparseMatrix> Minv =
          Teuchos::rcp(new Core::LinAlg::SparseMatrix(*Mass));
      Teuchos::RCP<Epetra_Vector> diag = Core::LinAlg::CreateVector(*dofmap, true);
      int err = 0;
      Minv->extract_diagonal_copy(*diag);
      err = diag->Reciprocal(*diag);
      if (err != 0) FOUR_C_THROW("Reciprocal: Zero diagonal entry!");
      err = Minv->replace_diagonal_values(*diag);
      Minv->complete(*dofmap, *dofmap);

      // displacement increment Dd
      Teuchos::RCP<Epetra_Vector> Dd = Core::LinAlg::CreateVector(*dofmap, true);
      Dd->Update(1.0, *disn_, 0.0);
      Dd->Update(-1.0, (*dis_)[0], 1.0);

      // mortar operator Bc
      Teuchos::RCP<Core::LinAlg::SparseMatrix> Mmat = cmtbridge_->get_strategy().m_matrix();
      Teuchos::RCP<Core::LinAlg::SparseMatrix> Dmat = cmtbridge_->get_strategy().d_matrix();
      Teuchos::RCP<Epetra_Map> slavedofmap = Teuchos::rcp(new Epetra_Map(Dmat->range_map()));
      Teuchos::RCP<Core::LinAlg::SparseMatrix> Bc =
          Teuchos::rcp(new Core::LinAlg::SparseMatrix(*dofmap, 10));
      Teuchos::RCP<Core::LinAlg::SparseMatrix> M =
          Teuchos::rcp(new Core::LinAlg::SparseMatrix(*slavedofmap, 10));
      Teuchos::RCP<Core::LinAlg::SparseMatrix> D =
          Teuchos::rcp(new Core::LinAlg::SparseMatrix(*slavedofmap, 10));
      if (Teuchos::getIntegralValue<Inpar::Mortar::ParallelRedist>(
              cmtbridge_->get_strategy().params().sublist("PARALLEL REDISTRIBUTION"),
              "PARALLEL_REDIST") != Inpar::Mortar::ParallelRedist::redist_none)
      {
        M = Mortar::MatrixColTransform(Mmat, notredistmasterdofmap);
        D = Mortar::MatrixColTransform(Dmat, notredistslavedofmap);
      }
      else
      {
        M = Mmat;
        D = Dmat;
      }
      Bc->add(*M, true, -1.0, 1.0);
      Bc->add(*D, true, 1.0, 1.0);
      Bc->complete(*slavedofmap, *dofmap);
      Bc->apply_dirichlet(*(dbcmaps_->cond_map()), false);

      // matrix of the normal vectors
      Teuchos::RCP<Core::LinAlg::SparseMatrix> N =
          cmtbridge_->get_strategy().evaluate_normals(disn_);

      // lagrange multiplier z
      Teuchos::RCP<Epetra_Vector> LM = cmtbridge_->get_strategy().lagrange_multiplier();
      Teuchos::RCP<Epetra_Vector> Z = Core::LinAlg::CreateVector(*slavenodemap, true);
      Teuchos::RCP<Epetra_Vector> z = Core::LinAlg::CreateVector(*activenodemap, true);
      N->multiply(false, *LM, *Z);
      Core::LinAlg::export_to(*Z, *z);

      // auxiliary operator BN = Bc * N
      Teuchos::RCP<Core::LinAlg::SparseMatrix> BN =
          Core::LinAlg::MLMultiply(*Bc, false, *N, true, false, false, true);

      // operator A
      Teuchos::RCP<Core::LinAlg::SparseMatrix> tempmtx1;
      Teuchos::RCP<Core::LinAlg::SparseMatrix> tempmtx2;
      Teuchos::RCP<Core::LinAlg::SparseMatrix> tempmtx3;
      Teuchos::RCP<Core::LinAlg::SparseMatrix> A;
      Teuchos::RCP<Core::LinAlg::SparseMatrix> Atemp1 =
          Core::LinAlg::MLMultiply(*BN, true, *Minv, false, false, false, true);
      Teuchos::RCP<Core::LinAlg::SparseMatrix> Atemp2 =
          Core::LinAlg::MLMultiply(*Atemp1, false, *BN, false, false, false, true);
      Atemp2->scale(R4);
      Core::LinAlg::SplitMatrix2x2(Atemp2, notactivenodemap, activenodemap, notactivenodemap,
          activenodemap, tempmtx1, tempmtx2, tempmtx3, A);
      A->complete(*activenodemap, *activenodemap);

      // diagonal of A
      Teuchos::RCP<Epetra_Vector> AD = Core::LinAlg::CreateVector(*activenodemap, true);
      A->extract_diagonal_copy(*AD);

      // operator b
      Teuchos::RCP<Epetra_Vector> btemp1 = Core::LinAlg::CreateVector(*dofmap, true);
      Teuchos::RCP<Epetra_Vector> btemp2 = Core::LinAlg::CreateVector(*slavenodemap, true);
      Teuchos::RCP<Epetra_Vector> b = Core::LinAlg::CreateVector(*activenodemap, true);
      btemp1->Update(R1, *Dd, 0.0);
      btemp1->Update(R2, (*vel_)[0], 1.0);
      btemp1->Update(R3, (*acc_)[0], 1.0);
      BN->multiply(true, *btemp1, *btemp2);
      Core::LinAlg::export_to(*btemp2, *b);

      // operatior c
      Teuchos::RCP<Epetra_Vector> ctemp = Core::LinAlg::CreateVector(*slavenodemap, true);
      Teuchos::RCP<Epetra_Vector> c = Core::LinAlg::CreateVector(*activenodemap, true);
      BN->multiply(true, *Dd, *ctemp);
      Core::LinAlg::export_to(*ctemp, *c);

      // contact work wc
      Teuchos::RCP<Epetra_Vector> wc = Core::LinAlg::CreateVector(*activenodemap, true);
      wc->Multiply(1.0, *c, *z, 0.0);

      // gain and loss of energy
      double gain = 0;
      double loss = 0;
      Teuchos::RCP<Epetra_Vector> wp = Core::LinAlg::CreateVector(*activenodemap, true);
      Teuchos::RCP<Epetra_Vector> wn = Core::LinAlg::CreateVector(*activenodemap, true);
      Teuchos::RCP<Epetra_Vector> wd = Core::LinAlg::CreateVector(*activenodemap, true);
      Teuchos::RCP<Epetra_Vector> wt = Core::LinAlg::CreateVector(*activenodemap, true);
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
      Teuchos::RCP<Epetra_Vector> wtemp1 = Core::LinAlg::CreateVector(*activenodemap, true);
      Teuchos::RCP<Epetra_Vector> wtemp2 = Core::LinAlg::CreateVector(*activenodemap, true);
      Teuchos::RCP<Epetra_Vector> w = Core::LinAlg::CreateVector(*activenodemap, true);
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
      Teuchos::RCP<Epetra_Vector> p1 = Core::LinAlg::CreateVector(*activenodemap, true);
      Teuchos::RCP<Epetra_Vector> p2 = Core::LinAlg::CreateVector(*activenodemap, true);
      Teuchos::RCP<Epetra_Vector> p = Core::LinAlg::CreateVector(*activenodemap, true);
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
      Teuchos::RCP<Epetra_Vector> x = Core::LinAlg::CreateVector(*activenodemap, true);
      Teuchos::RCP<Epetra_Vector> f = Core::LinAlg::CreateVector(*activenodemap, true);
      int NumEntries = 0;
      int* Indices = nullptr;
      double* Values = nullptr;
      double res = 1.0;
      double initres = 1.0;
      double dfik = 0;
      Teuchos::RCP<Core::LinAlg::SparseMatrix> DF =
          Teuchos::rcp(new Core::LinAlg::SparseMatrix(*activenodemap, 10));

      // rhs f
      for (int i = 0; i < activenodemap->NumMyElements(); ++i)
      {
        x->PutScalar(0.0);
        (A->epetra_matrix())->ExtractMyRowView(i, NumEntries, Values, Indices);
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
        (A->epetra_matrix())->ExtractMyRowView(i, NumEntries, Values, Indices);
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
            DF->assemble(dfik, activenodemap->GID(i), activenodemap->GID(k));
          }
          else
            DF->assemble((*x)[k] * (*p)[i], activenodemap->GID(i), activenodemap->GID(k));
        }
      }
      DF->complete(*activenodemap, *activenodemap);

      // (3) Newton-Iteration
      Teuchos::RCP<Epetra_Vector> mf = Core::LinAlg::CreateVector(*activenodemap, true);
      Teuchos::RCP<Epetra_Vector> dp = Core::LinAlg::CreateVector(*activenodemap, true);
      double tol = 0.00000001;
      double numiter = 0;
      double stopcrit = 100;

      while (res > tol)
      {
        // solver for linear equations DF * dp = -f
        mf->Update(-1.0, *f, 0.0);
        Core::LinAlg::SolverParams solver_params;
        solver_params.refactor = true;
        solver_->solve(DF->epetra_operator(), dp, mf, solver_params);

        // Update solution p_n = p_n-1 + dp
        p->Update(1.0, *dp, 1.0);

        // rhs f
        for (int i = 0; i < activenodemap->NumMyElements(); ++i)
        {
          x->PutScalar(0.0);
          (A->epetra_matrix())->ExtractMyRowView(i, NumEntries, Values, Indices);
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
        DF->put_scalar(0.0);
        for (int i = 0; i < activenodemap->NumMyElements(); ++i)
        {
          x->PutScalar(0.0);
          (A->epetra_matrix())->ExtractMyRowView(i, NumEntries, Values, Indices);
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
              DF->assemble(dfik, activenodemap->GID(i), activenodemap->GID(k));
            }
            else
              DF->assemble((*x)[k] * (*p)[i], activenodemap->GID(i), activenodemap->GID(k));
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
      Teuchos::RCP<Epetra_Vector> ptemp1 = Core::LinAlg::CreateVector(*slavenodemap, true);
      Teuchos::RCP<Epetra_Vector> ptemp2 = Core::LinAlg::CreateVector(*dofmap, true);
      Teuchos::RCP<Epetra_Vector> VU = Core::LinAlg::CreateVector(*dofmap, true);
      Core::LinAlg::export_to(*p, *ptemp1);
      BN->multiply(false, *ptemp1, *ptemp2);
      Minv->multiply(false, *ptemp2, *VU);
      veln_->Update(1.0, *VU, 1.0);
    }
  }
}

/*----------------------------------------------------------------------*/
/* Reset configuration after time step */
void Solid::TimInt::reset_step()
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
    discret_->evaluate(
        p, Teuchos::null, Teuchos::null, Teuchos::null, Teuchos::null, Teuchos::null);
    discret_->clear_state();
  }

  // reset 0D cardiovascular model if we have monolithic 0D cardiovascular-structure coupling (mhv
  // 02/2015)
  if (cardvasc0dman_->have_cardiovascular0_d()) cardvasc0dman_->reset_step();
}

/*----------------------------------------------------------------------*/
/* Read and set restart values */
void Solid::TimInt::read_restart(const int step)
{
  Core::IO::DiscretizationReader reader(
      discret_, Global::Problem::instance()->input_control_file(), step);
  if (step != reader.read_int("step")) FOUR_C_THROW("Time step on file not equal to given step");

  step_ = step;
  stepn_ = step_ + 1;
  time_ = Teuchos::rcp(new TimeStepping::TimIntMStep<double>(0, 0, reader.read_double("time")));
  timen_ = (*time_)[0] + (*dt_)[0];

  read_restart_state();

  read_restart_constraint();
  read_restart_cardiovascular0_d();
  read_restart_contact_meshtying();
  read_restart_beam_contact();
  read_restart_multi_scale();
  read_restart_spring_dashpot();

  read_restart_force();
}

/*----------------------------------------------------------------------*/
/* Set restart values passed down from above */
void Solid::TimInt::set_restart(int step, double time, Teuchos::RCP<Epetra_Vector> disn,
    Teuchos::RCP<Epetra_Vector> veln, Teuchos::RCP<Epetra_Vector> accn,
    Teuchos::RCP<std::vector<char>> elementdata, Teuchos::RCP<std::vector<char>> nodedata)
{
  step_ = step;
  stepn_ = step_ + 1;
  time_ = Teuchos::rcp(new TimeStepping::TimIntMStep<double>(0, 0, time));
  timen_ = (*time_)[0] + (*dt_)[0];

  set_restart_state(disn, veln, accn, elementdata, nodedata);

  // ---------------------------------------------------------------------------
  // set restart is only for simple structure problems
  // hence we put some security measures in place

  // constraints
  if (conman_->have_constraint()) FOUR_C_THROW("Set restart not implemented for constraints");

  // Cardiovascular0D
  if (cardvasc0dman_->have_cardiovascular0_d())
    FOUR_C_THROW("Set restart not implemented for Cardiovascular0D");

  // contact / meshtying
  if (have_contact_meshtying()) FOUR_C_THROW("Set restart not implemented for contact / meshtying");

  // beam contact
  if (have_beam_contact()) FOUR_C_THROW("Set restart not implemented for beam contact");

  // biofilm growth
  if (have_biofilm_growth()) FOUR_C_THROW("Set restart not implemented for biofilm growth");
}

/*----------------------------------------------------------------------*/
/* Read and set restart state */
void Solid::TimInt::read_restart_state()
{
  Core::IO::DiscretizationReader reader(
      discret_, Global::Problem::instance()->input_control_file(), step_);

  reader.read_vector(disn_, "displacement");
  dis_->update_steps(*disn_);

  if ((dismatn_ != Teuchos::null))
  {
    reader.read_vector(dismatn_, "material_displacement");
    dismat_->update_steps(*dismatn_);
  }

  reader.read_vector(veln_, "velocity");
  vel_->update_steps(*veln_);
  reader.read_vector(accn_, "acceleration");
  acc_->update_steps(*accn_);
  reader.read_history_data(step_);
}

/*----------------------------------------------------------------------*/
/* Read and set restart state */
void Solid::TimInt::set_restart_state(Teuchos::RCP<Epetra_Vector> disn,
    Teuchos::RCP<Epetra_Vector> veln, Teuchos::RCP<Epetra_Vector> accn,
    Teuchos::RCP<std::vector<char>> elementdata, Teuchos::RCP<std::vector<char>> nodedata

)
{
  dis_->update_steps(*disn);
  vel_->update_steps(*veln);
  acc_->update_steps(*accn);

  // the following is copied from read_mesh()
  // before we unpack nodes/elements we store a copy of the nodal row/col map
  Teuchos::RCP<Epetra_Map> noderowmap = Teuchos::rcp(new Epetra_Map(*discret_->node_row_map()));
  Teuchos::RCP<Epetra_Map> nodecolmap = Teuchos::rcp(new Epetra_Map(*discret_->node_col_map()));

  // unpack nodes and elements
  // so everything should be OK
  discret_->un_pack_my_nodes(nodedata);
  discret_->un_pack_my_elements(elementdata);
  discret_->redistribute(*noderowmap, *nodecolmap);
}
/*----------------------------------------------------------------------*/
/* Read and set restart values for constraints */
void Solid::TimInt::read_restart_constraint()
{
  if (conman_->have_constraint())
  {
    Core::IO::DiscretizationReader reader(
        discret_, Global::Problem::instance()->input_control_file(), step_);
    double uzawatemp = reader.read_double("uzawaparameter");
    consolv_->set_uzawa_parameter(uzawatemp);

    conman_->read_restart(reader, (*time_)[0]);
  }
}

/*----------------------------------------------------------------------*/
/* Read and set restart values for 0D cardiovascular models */
void Solid::TimInt::read_restart_cardiovascular0_d()
{
  if (cardvasc0dman_->have_cardiovascular0_d())
  {
    Core::IO::DiscretizationReader reader(
        discret_, Global::Problem::instance()->input_control_file(), step_);
    cardvasc0dman_->read_restart(reader, (*time_)[0]);
  }
}

/*----------------------------------------------------------------------*/
/* Read and set restart values for spring dashpot */
void Solid::TimInt::read_restart_spring_dashpot()
{
  if (springman_->have_spring_dashpot())
  {
    Core::IO::DiscretizationReader reader(
        discret_, Global::Problem::instance()->input_control_file(), step_);
    springman_->read_restart(reader, (*time_)[0]);
  }
}

/*----------------------------------------------------------------------*/
/* Read and set restart values for contact / meshtying */
void Solid::TimInt::read_restart_contact_meshtying()
{
  //**********************************************************************
  // NOTE: There is an important difference here between contact and
  // meshtying simulations. In both cases, the current coupling operators
  // have to be re-computed for restart, but in the meshtying case this
  // means evaluating DM in the reference configuration!
  // Thus, both dis_ (current displacement state) and zero_ are handed
  // in and contact / meshtying managers choose the correct state.
  //**********************************************************************
  Core::IO::DiscretizationReader reader(
      discret_, Global::Problem::instance()->input_control_file(), step_);

  if (have_contact_meshtying()) cmtbridge_->read_restart(reader, (*dis_)(0), zeros_);
}

/*----------------------------------------------------------------------*/
/* Read and set restart values for beam contact */
void Solid::TimInt::read_restart_beam_contact()
{
  if (have_beam_contact())
  {
    Core::IO::DiscretizationReader reader(
        discret_, Global::Problem::instance()->input_control_file(), step_);
    beamcman_->read_restart(reader);
  }
}

/*----------------------------------------------------------------------*/
/* Read and set restart values for multi-scale */
void Solid::TimInt::read_restart_multi_scale()
{
  Teuchos::RCP<Mat::PAR::Bundle> materials = Global::Problem::instance()->materials();

  if (std::any_of(materials->map().begin(), materials->map().end(),
          [](const auto& item)
          { return item.second->type() == Core::Materials::m_struct_multiscale; }))
  {
    int my_pid = Global::Problem::instance()->get_dis("structure")->get_comm().MyPID();
    // set dummy displacements
    discret_->set_state("displacement", zeros_);
    Core::Elements::Element::LocationArray la(discret_->num_dof_sets());
    for (const auto* ele : discret_->my_col_element_range())
    {
      ele->location_vector(*discret_, la, false);

      const auto* solid_ele = dynamic_cast<const Discret::ELEMENTS::Solid*>(ele);
      FOUR_C_THROW_UNLESS(solid_ele,
          "Multiscale simulations are currently only possible with the new solid elements");

      solid_ele->for_each_gauss_point(*discret_, la[0].lm_,
          [&](Mat::So3Material& solid_material, double integration_factor, int gp)
          {
            if (solid_material.material_type() == Core::Materials::m_struct_multiscale)
            {
              auto& micro = dynamic_cast<Mat::MicroMaterial&>(solid_material);
              const bool eleowner = my_pid == ele->owner();

              micro.read_restart(gp, ele->id(), eleowner);
            }
          });
    }
  }
}

/*----------------------------------------------------------------------*/
/* Calculate all output quantities that depend on a potential material history */
void Solid::TimInt::prepare_output(bool force_prepare_timestep)
{
  determine_stress_strain();
  determine_energy();
  determine_optional_quantity();
  if (havemicromat_) prepare_output_micro();
}

/*----------------------------------------------------------------------*
 *   Write Output while the Newton Iteration         by hiermeier 09/13 *
 *   (useful for debugging purposes)                                    */
void Solid::TimInt::output_every_iter(bool nw, bool ls)
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
    if (output_->output()->file_steps() >= std::numeric_limits<int>::max() / 50000)
      newFileSteps = std::numeric_limits<int>::max();
    else
      newFileSteps = output_->output()->file_steps() * 50000;

    output_->output()->set_file_steps(newFileSteps);

    std::string resultname = output_->output()->file_name() + "_EveryIter";
    output_->new_result_file(resultname, oei_filecounter_);
    output_->write_mesh(0, 0.0);
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

  output_->write_mesh(outputcounter_, (double)outputcounter_);  //(*time_)[0]

  //  output_->overwrite_result_file();
  output_state(datawritten);
}

/*----------------------------------------------------------------------*/
/* output to file
 * originally by mwgee 03/07 */
void Solid::TimInt::output_step(const bool forced_writerestart)
{
  // print iterations instead of steps
  if (outputeveryiter_)
  {
    output_every_iter();
    return;
  }

  // special treatment is necessary when restart is forced
  if (forced_writerestart)
  {
    // reset possible history data on element level
    reset_step();
    // restart has already been written or simulation has just started
    if ((writerestartevery_ and (step_ % writerestartevery_ == 0)) or
        step_ == Global::Problem::instance()->restart())
      return;
    // if state already exists, add restart information
    if (writeresultsevery_ and (step_ % writeresultsevery_ == 0))
    {
      add_restart_to_output_state();
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
      Global::Problem::instance()->restart_manager()->restart(step_, discret_->get_comm()))
  {
    output_restart(datawritten);
    lastwrittenresultsstep_ = step_;
  }

  // output results (not necessary if restart in same step)
  if (writestate_ and writeresultsevery_ and (step_ % writeresultsevery_ == 0) and
      (not datawritten))
  {
    output_state(datawritten);
    lastwrittenresultsstep_ = step_;
  }

  // output stress & strain
  if (writeresultsevery_ and
      ((writestress_ != Inpar::Solid::stress_none) or
          (writecouplstress_ != Inpar::Solid::stress_none) or
          (writestrain_ != Inpar::Solid::strain_none) or
          (writeplstrain_ != Inpar::Solid::strain_none)) and
      (step_ % writeresultsevery_ == 0))
  {
    output_stress_strain(datawritten);
  }

  // output energy
  if (writeenergyevery_ and (step_ % writeenergyevery_ == 0))
  {
    output_energy();
  }

  // output optional quantity
  if (writeresultsevery_ and (writeoptquantity_ != Inpar::Solid::optquantity_none) and
      (step_ % writeresultsevery_ == 0))
  {
    output_opt_quantity(datawritten);
  }

  // output active set, energies and momentum for contact
  output_contact();

  output_volume_mass();

  // write output on micro-scale (multi-scale analysis)
  if (havemicromat_) output_micro();
}

/*-----------------------------------------------------------------------------*
 * write GMSH output of displacement field
 *-----------------------------------------------------------------------------*/
void Solid::TimInt::write_gmsh_struc_output_step()
{
  if (not gmsh_out_) return;

  const std::string filename = Core::IO::Gmsh::GetFileName(
      "struct", discret_->writer()->output()->file_name(), stepn_, false, myrank_);
  std::ofstream gmshfilecontent(filename.c_str());

  // add 'View' to Gmsh postprocessing file
  gmshfilecontent << "View \" "
                  << "struct displacement \" {" << std::endl;
  // draw vector field 'struct displacement' for every element
  Core::IO::Gmsh::VectorFieldDofBasedToGmsh(discret_, dispn(), gmshfilecontent, 0, true);
  gmshfilecontent << "};" << std::endl;
}

bool Solid::TimInt::has_final_state_been_written() const
{
  return step_ == lastwrittenresultsstep_;
}
/*----------------------------------------------------------------------*/
/* We need the restart data to perform on "restarts" on the fly for parameter
 * continuation
 */
void Solid::TimInt::get_restart_data(Teuchos::RCP<int> step, Teuchos::RCP<double> time,
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
  *elementdata = *(discret_->pack_my_elements());
  *nodedata = *(discret_->pack_my_nodes());

  // get restart data is only for simple structure problems
  // hence

  // constraints
  if (conman_->have_constraint()) FOUR_C_THROW("Get restart data not implemented for constraints");

  // contact / meshtying
  if (have_contact_meshtying())
    FOUR_C_THROW("Get restart data not implemented for contact / meshtying");

  // beam contact
  if (have_beam_contact()) FOUR_C_THROW("Get restart data not implemented for beam contact");

  // biofilm growth
  if (have_biofilm_growth()) FOUR_C_THROW("Get restart data not implemented for biofilm growth");
}
/*----------------------------------------------------------------------*/
/* write restart
 * originally by mwgee 03/07 */
void Solid::TimInt::output_restart(bool& datawritten)
{
  // Yes, we are going to write...
  datawritten = true;

  // write restart output, please
  if (step_ != 0) output_->write_mesh(step_, (*time_)[0]);
  output_->new_step(step_, (*time_)[0]);
  output_->write_vector("displacement", (*dis_)(0));
  if (dismat_ != Teuchos::null) output_->write_vector("material_displacement", (*dismat_)(0));
  output_->write_vector("velocity", (*vel_)(0));
  output_->write_vector("acceleration", (*acc_)(0));
  output_->write_element_data(firstoutputofrun_);
  output_->write_node_data(firstoutputofrun_);
  write_restart_force(output_);
  // owner of elements is just written once because it does not change during simulation (so far)
  firstoutputofrun_ = false;

  // constraints
  if (conman_->have_constraint())
  {
    output_->write_double("uzawaparameter", consolv_->get_uzawa_parameter());
    output_->write_vector("lagrmultiplier", conman_->get_lagr_mult_vector());
    output_->write_vector("refconval", conman_->get_ref_base_values());
  }

  // 0D cardiovascular models
  if (cardvasc0dman_->have_cardiovascular0_d())
  {
    output_->write_vector("cv0d_df_np", cardvasc0dman_->get0_d_df_np());
    output_->write_vector("cv0d_f_np", cardvasc0dman_->get0_d_f_np());

    output_->write_vector("cv0d_dof_np", cardvasc0dman_->get0_d_dof_np());
    output_->write_vector("vol_np", cardvasc0dman_->get0_d_vol_np());
  }

  // contact and meshtying
  if (have_contact_meshtying())
  {
    cmtbridge_->write_restart(output_);
    cmtbridge_->postprocess_quantities(output_);

    {
      Teuchos::RCP<Teuchos::ParameterList> cmtOutputParams =
          Teuchos::rcp(new Teuchos::ParameterList());
      cmtOutputParams->set<int>("step", step_);
      cmtOutputParams->set<double>("time", (*time_)[0]);
      cmtOutputParams->set<Teuchos::RCP<const Epetra_Vector>>("displacement", (*dis_)(0));
      cmtbridge_->postprocess_quantities_per_interface(cmtOutputParams);
    }
  }

  // beam contact
  if (have_beam_contact())
  {
    beamcman_->write_restart(output_);
  }

  // biofilm growth
  if (have_biofilm_growth())
  {
    output_->write_vector("str_growth_displ", strgrdisp_);
  }

  // springdashpot output
  if (springman_->have_spring_dashpot()) springman_->output_restart(output_, discret_, disn_);

  // info dedicated to user's eyes staring at standard out
  if ((myrank_ == 0) and printscreen_ and (step_old() % printscreen_ == 0))
  {
    Core::IO::cout << "====== Restart for field '" << discret_->name() << "' written in step "
                   << step_ << Core::IO::endl;
  }
}

/*----------------------------------------------------------------------*/
/* output displacements, velocities and accelerations
 * originally by mwgee 03/07 */
void Solid::TimInt::output_state(bool& datawritten)
{
  // Yes, we are going to write...
  datawritten = true;

  // write now
  if (outputeveryiter_)
  {
    output_->new_step(outputcounter_, (double)outputcounter_);
    output_->write_vector("displacement", Teuchos::rcp_static_cast<Epetra_MultiVector>(disn_));
  }
  else
  {
    output_->new_step(step_, (*time_)[0]);
    output_->write_vector("displacement", (*dis_)(0));
  }

  if ((dismatn_ != Teuchos::null)) output_->write_vector("material_displacement", (*dismat_)(0));

  // for visualization of vel and acc do not forget to comment in corresponding lines in
  // StructureEnsightWriter
  if (writevelacc_)
  {
    output_->write_vector("velocity", (*vel_)(0));
    output_->write_vector("acceleration", (*acc_)(0));
  }

  // biofilm growth
  if (have_biofilm_growth())
  {
    output_->write_vector("str_growth_displ", strgrdisp_);
  }

  // owner of elements is just written once because it does not change during simulation (so far)
  if (writeele_) output_->write_element_data(firstoutputofrun_);
  output_->write_node_data(firstoutputofrun_);
  firstoutputofrun_ = false;

  // meshtying and contact output
  if (have_contact_meshtying())
  {
    cmtbridge_->postprocess_quantities(output_);

    {
      Teuchos::RCP<Teuchos::ParameterList> cmtOutputParams =
          Teuchos::rcp(new Teuchos::ParameterList());
      cmtOutputParams->set<int>("step", step_);
      cmtOutputParams->set<double>("time", (*time_)[0]);
      cmtOutputParams->set<Teuchos::RCP<const Epetra_Vector>>("displacement", (*dis_)(0));
      cmtbridge_->postprocess_quantities_per_interface(cmtOutputParams);
    }
  }

  if (porositysplitter_ != Teuchos::null)
  {
    Teuchos::RCP<Epetra_Vector> porosity = porositysplitter_->extract_cond_vector((*dis_)(0));
    output_->write_vector("porosity_p1", porosity);
  }

  // springdashpot output
  if (springman_->have_spring_dashpot()) springman_->output(output_, discret_, disn_);
}

/*----------------------------------------------------------------------*/
/* add restart information to output_state */
void Solid::TimInt::add_restart_to_output_state()
{
  // add velocity and acceleration if necessary
  if (!writevelacc_)
  {
    output_->write_vector("velocity", (*vel_)(0));
    output_->write_vector("acceleration", (*acc_)(0));
  }

  write_restart_force(output_);

  // constraints
  if (conman_->have_constraint())
  {
    output_->write_double("uzawaparameter", consolv_->get_uzawa_parameter());
    output_->write_vector("lagrmultiplier", conman_->get_lagr_mult_vector());
    output_->write_vector("refconval", conman_->get_ref_base_values());
  }

  // 0D cardiovascular models
  if (cardvasc0dman_->have_cardiovascular0_d())
  {
    output_->write_vector("cv0d_df_np", cardvasc0dman_->get0_d_df_np());
    output_->write_vector("cv0d_f_np", cardvasc0dman_->get0_d_f_np());

    output_->write_vector("cv0d_dof_np", cardvasc0dman_->get0_d_dof_np());
    output_->write_vector("vol_np", cardvasc0dman_->get0_d_vol_np());
  }

  // springdashpot output
  if (springman_->have_spring_dashpot()) springman_->output_restart(output_, discret_, disn_);

  // contact/meshtying
  if (have_contact_meshtying()) cmtbridge_->write_restart(output_, true);

  // TODO: add missing restart data for surface stress and contact/meshtying here

  // beam contact
  if (have_beam_contact()) beamcman_->write_restart(output_);


  // finally add the missing mesh information, order is important here
  output_->write_mesh(step_, (*time_)[0]);

  // info dedicated to user's eyes staring at standard out
  if ((myrank_ == 0) and printscreen_ and (step_old() % printscreen_ == 0))
  {
    Core::IO::cout << "====== Restart for field '" << discret_->name() << "' written in step "
                   << step_ << Core::IO::endl;
  }
}

/*----------------------------------------------------------------------*/
/* Calculation of stresses and strains */
void Solid::TimInt::determine_stress_strain()
{
  if (writeresultsevery_ and
      ((writestress_ != Inpar::Solid::stress_none) or
          (writecouplstress_ != Inpar::Solid::stress_none) or
          (writestrain_ != Inpar::Solid::strain_none) or
          (writeplstrain_ != Inpar::Solid::strain_none)) and
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
    discret_->clear_state();
    // extended set_state(0,...) in case of multiple dofsets (e.g. TSI)
    discret_->set_state(0, "residual displacement", zeros_);
    discret_->set_state(0, "displacement", disn_);

    if ((dismatn_ != Teuchos::null)) discret_->set_state(0, "material_displacement", dismatn_);

    Teuchos::RCP<Core::LinAlg::SparseOperator> system_matrix = Teuchos::null;
    Teuchos::RCP<Epetra_Vector> system_vector = Teuchos::null;
    Core::FE::UTILS::evaluate(
        *discret_, p, system_matrix, system_vector, discret_->element_row_map());
    discret_->clear_state();
  }
}

/*----------------------------------------------------------------------*/
/* Calculation of internal, external and kinetic energy */
void Solid::TimInt::determine_energy()
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
      discret_->clear_state();
      discret_->set_state("displacement", disn_);
      // get energies
      Teuchos::RCP<Core::LinAlg::SerialDenseVector> energies =
          Teuchos::rcp(new Core::LinAlg::SerialDenseVector(1));
      discret_->evaluate_scalars(p, energies);
      discret_->clear_state();
      intergy_ = (*energies)(0);
    }

    // global calculation of kinetic energy
    kinergy_ = 0.0;  // total kinetic energy
    {
      Teuchos::RCP<Epetra_Vector> linmom = Core::LinAlg::CreateVector(*dof_row_map_view(), true);
      mass_->multiply(false, *veln_, *linmom);
      linmom->Dot(*veln_, &kinergy_);
      kinergy_ *= 0.5;
    }

    // external energy
    extergy_ = 0.0;  // total external energy
    {
      // WARNING: This will only work with dead loads and implicit time
      // integration (otherwise there is no fextn_)!!!
      Teuchos::RCP<Epetra_Vector> fext = fext_new();
      fext->Dot(*disn_, &extergy_);
    }
  }
}

/*----------------------------------------------------------------------*/
/* Calculation of an optional quantity */
void Solid::TimInt::determine_optional_quantity()
{
  if (writeresultsevery_ and (writeoptquantity_ != Inpar::Solid::optquantity_none) and
      (stepn_ % writeresultsevery_ == 0))
  {
    //-------------------------------
    // create the parameters for the discretization
    Teuchos::ParameterList p;

    // action for elements
    if (writeoptquantity_ == Inpar::Solid::optquantity_membranethickness)
      p.set("action", "calc_struct_thickness");
    else
      FOUR_C_THROW("requested optional quantity type not supported");

    // other parameters that might be needed by the elements
    p.set("total time", timen_);
    p.set("delta time", (*dt_)[0]);

    optquantitydata_ = Teuchos::rcp(new std::vector<char>());
    p.set("optquantity", optquantitydata_);
    p.set<int>("iooptquantity", writeoptquantity_);

    // set vector values needed by elements
    discret_->clear_state();
    // extended set_state(0,...) in case of multiple dofsets (e.g. TSI)
    discret_->set_state(0, "residual displacement", zeros_);
    discret_->set_state(0, "displacement", disn_);

    if ((dismatn_ != Teuchos::null)) discret_->set_state(0, "material_displacement", dismatn_);

    discret_->evaluate(
        p, Teuchos::null, Teuchos::null, Teuchos::null, Teuchos::null, Teuchos::null);
    discret_->clear_state();
  }
}

/*----------------------------------------------------------------------*/
/* stress calculation and output */
void Solid::TimInt::output_stress_strain(bool& datawritten)
{
  // Make new step
  if (not datawritten)
  {
    output_->new_step(step_, (*time_)[0]);
  }
  datawritten = true;

  // write stress
  if (writestress_ != Inpar::Solid::stress_none)
  {
    std::string stresstext = "";
    if (writestress_ == Inpar::Solid::stress_cauchy)
    {
      stresstext = "gauss_cauchy_stresses_xyz";
    }
    else if (writestress_ == Inpar::Solid::stress_2pk)
    {
      stresstext = "gauss_2PK_stresses_xyz";
    }
    else
    {
      FOUR_C_THROW("requested stress type not supported");
    }
    output_->write_vector(stresstext, *stressdata_, *(discret_->element_row_map()));
    // we don't need this anymore
    stressdata_ = Teuchos::null;
  }

  // write coupling stress
  if (writecouplstress_ != Inpar::Solid::stress_none)
  {
    std::string couplstresstext = "";
    if (writecouplstress_ == Inpar::Solid::stress_cauchy)
    {
      couplstresstext = "gauss_cauchy_coupling_stresses_xyz";
    }
    else if (writecouplstress_ == Inpar::Solid::stress_2pk)
    {
      couplstresstext = "gauss_2PK_coupling_stresses_xyz";
    }
    else
    {
      FOUR_C_THROW("requested stress type not supported");
    }
    output_->write_vector(couplstresstext, *couplstressdata_, *(discret_->element_row_map()));
    // we don't need this anymore
    couplstressdata_ = Teuchos::null;
  }

  // write strain
  if (writestrain_ != Inpar::Solid::strain_none)
  {
    std::string straintext = "";
    if (writestrain_ == Inpar::Solid::strain_ea)
    {
      straintext = "gauss_EA_strains_xyz";
    }
    else if (writestrain_ == Inpar::Solid::strain_gl)
    {
      straintext = "gauss_GL_strains_xyz";
    }
    else if (writestrain_ == Inpar::Solid::strain_log)
    {
      straintext = "gauss_LOG_strains_xyz";
    }
    else
    {
      FOUR_C_THROW("requested strain type not supported");
    }
    output_->write_vector(straintext, *straindata_, *(discret_->element_row_map()));
    // we don't need this anymore
    straindata_ = Teuchos::null;
  }

  // write plastic strain
  if (writeplstrain_ != Inpar::Solid::strain_none)
  {
    std::string plstraintext = "";
    if (writeplstrain_ == Inpar::Solid::strain_ea)
    {
      plstraintext = "gauss_pl_EA_strains_xyz";
    }
    else if (writeplstrain_ == Inpar::Solid::strain_gl)
    {
      plstraintext = "gauss_pl_GL_strains_xyz";
    }
    else
    {
      FOUR_C_THROW("requested plastic strain type not supported");
    }
    output_->write_vector(plstraintext, *plstraindata_, *(discret_->element_row_map()));
    // we don't need this anymore
    plstraindata_ = Teuchos::null;
  }

  // write structural rotation tensor
  if (writerotation_) output_->write_vector("rotation", *rotdata_, *(discret_->element_row_map()));
}

/*----------------------------------------------------------------------*/
/* output system energies */
void Solid::TimInt::output_energy()
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
void Solid::TimInt::output_opt_quantity(bool& datawritten)
{
  // Make new step
  if (not datawritten)
  {
    output_->new_step(step_, (*time_)[0]);
  }
  datawritten = true;

  // write optional quantity
  if (writeoptquantity_ != Inpar::Solid::optquantity_none)
  {
    std::string optquantitytext = "";
    if (writeoptquantity_ == Inpar::Solid::optquantity_membranethickness)
      optquantitytext = "gauss_membrane_thickness";
    else
      FOUR_C_THROW("requested optional quantity type not supported");

    output_->write_vector(optquantitytext, *optquantitydata_, *(discret_->element_row_map()));
    // we don't need this anymore
    optquantitydata_ = Teuchos::null;
  }
}

/*----------------------------------------------------------------------*/
/* output active set, energies and momentum for contact */
void Solid::TimInt::output_contact()
{
  // only for contact / meshtying simulations
  if (have_contact_meshtying())
  {
    // print active set
    cmtbridge_->get_strategy().print_active_set();

    // check chosen output option
    Inpar::CONTACT::EmOutputType emtype = Core::UTILS::IntegralValue<Inpar::CONTACT::EmOutputType>(
        cmtbridge_->get_strategy().params(), "EMOUTPUT");

    // get out of here if no energy/momentum output wanted
    if (emtype == Inpar::CONTACT::output_none) return;

    // get some parameters from parameter list
    double timen = (*time_)[0];
    double dt = (*dt_)[0];
    int dim = cmtbridge_->get_strategy().n_dim();

    // global linear momentum (M*v)
    Teuchos::RCP<Epetra_Vector> mv = Core::LinAlg::CreateVector(*(discret_->dof_row_map()), true);
    mass_->multiply(false, (*vel_)[0], *mv);

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
    for (int k = 0; k < (discret_->node_row_map())->NumMyElements(); ++k)
    {
      // get current node
      int gid = (discret_->node_row_map())->GID(k);
      Core::Nodes::Node* mynode = discret_->g_node(gid);
      std::vector<int> globaldofs = discret_->dof(mynode);

      // loop over all DOFs comprised by this node
      for (int i = 0; i < dim; i++)
      {
        nodelinmom[i] = (*mv)[mv->Map().LID(globaldofs[i])];
        sumlinmom[i] += nodelinmom[i];
        position[i] = (mynode->x())[i] + ((*dis_)[0])[mv->Map().LID(globaldofs[i])];
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
      cmtbridge_->get_comm().SumAll(&sumangmom[i], &angmom[i], 1);
      cmtbridge_->get_comm().SumAll(&sumlinmom[i], &linmom[i], 1);
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
    discret_->clear_state();
    discret_->set_state("displacement", (*dis_)(0));
    Teuchos::RCP<Core::LinAlg::SerialDenseVector> energies =
        Teuchos::rcp(new Core::LinAlg::SerialDenseVector(1));
    energies->putScalar(0.0);
    discret_->evaluate_scalars(p, energies);
    discret_->clear_state();
    inten = (*energies)(0);

    //-------------------------Calculation of total external energy
    double exten = 0.0;
    // WARNING: This will only work with dead loads!!!
    // fext_->Dot(*dis_, &exten);

    //----------------------------------------Print results to file
    if (emtype == Inpar::CONTACT::output_file || emtype == Inpar::CONTACT::output_both)
    {
      // processor 0 does all the work
      if (!myrank_)
      {
        // path and filename
        std::ostringstream filename;
        const std::string filebase =
            Global::Problem::instance()->output_control_file()->file_name();
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
            FOUR_C_THROW(
                "File for writing contact interface forces/moments could not be generated.");
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
          FOUR_C_THROW("File for writing momentum and energy data could not be opened.");
      }
    }

    //-------------------------------Print energy results to screen
    if (emtype == Inpar::CONTACT::output_screen || emtype == Inpar::CONTACT::output_both)
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
    cmtbridge_->get_strategy().interface_forces(true);
  }
}

/*----------------------------------------------------------------------*/
/* output volume and mass */
void Solid::TimInt::output_volume_mass()
{
  const Teuchos::ParameterList& listwear = Global::Problem::instance()->wear_params();
  bool massvol = Core::UTILS::IntegralValue<int>(listwear, "VOLMASS_OUTPUT");
  if (!massvol) return;

  // initialize variables
  Teuchos::RCP<Core::LinAlg::SerialDenseVector> norms =
      Teuchos::rcp(new Core::LinAlg::SerialDenseVector(6));
  norms->putScalar(0.0);

  // call discretization to evaluate error norms
  Teuchos::ParameterList p;
  p.set("action", "calc_struct_mass_volume");
  discret_->clear_state();
  discret_->set_state("displacement", (*dis_)(0));
  if ((dismatn_ != Teuchos::null)) discret_->set_state(0, "material_displacement", dismatn_);
  discret_->evaluate_scalars(p, norms);
  discret_->clear_state();

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
void Solid::TimInt::output_micro()
{
  for (int i = 0; i < discret_->num_my_row_elements(); i++)
  {
    Core::Elements::Element* actele = discret_->l_row_element(i);
    Teuchos::RCP<Core::Mat::Material> mat = actele->material();
    if (mat->material_type() == Core::Materials::m_struct_multiscale)
    {
      Mat::MicroMaterial* micro = static_cast<Mat::MicroMaterial*>(mat.get());
      micro->output();
    }
  }
}

/*----------------------------------------------------------------------*/
/* calculate stresses and strains on micro-scale */
void Solid::TimInt::prepare_output_micro()
{
  for (int i = 0; i < discret_->num_my_row_elements(); i++)
  {
    Core::Elements::Element* actele = discret_->l_row_element(i);

    Teuchos::RCP<Core::Mat::Material> mat = actele->material();
    if (mat->material_type() == Core::Materials::m_struct_multiscale)
    {
      Mat::MicroMaterial* micro = static_cast<Mat::MicroMaterial*>(mat.get());
      micro->prepare_output();
    }
  }
}

/*----------------------------------------------------------------------*/
/* evaluate external forces at t_{n+1} */
void Solid::TimInt::apply_force_external(const double time, const Teuchos::RCP<Epetra_Vector> dis,
    const Teuchos::RCP<Epetra_Vector> disn, const Teuchos::RCP<Epetra_Vector> vel,
    Teuchos::RCP<Epetra_Vector>& fext)
{
  Teuchos::ParameterList p;
  // other parameters needed by the elements
  p.set("total time", time);
  p.set<const Core::UTILS::FunctionManager*>(
      "function_manager", &Global::Problem::instance()->function_manager());

  // set vector values needed by elements
  discret_->clear_state();
  discret_->set_state(0, "displacement", dis);
  discret_->set_state(0, "displacement new", disn);

  if (damping_ == Inpar::Solid::damp_material) discret_->set_state(0, "velocity", vel);

  discret_->evaluate_neumann(p, *fext);
}

/*----------------------------------------------------------------------*/
/* check whether we have nonlinear inertia forces or not */
int Solid::TimInt::have_nonlinear_mass() const
{
  const Teuchos::ParameterList& sdyn = Global::Problem::instance()->structural_dynamic_params();
  int masslin = Core::UTILS::IntegralValue<Inpar::Solid::MassLin>(sdyn, "MASSLIN");

  return masslin;
}

/*----------------------------------------------------------------------*/
/* check whether the initial conditions are fulfilled */
void Solid::TimInt::nonlinear_mass_sanity_check(Teuchos::RCP<const Epetra_Vector> fext,
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
    FOUR_C_THROW(
        "Initial configuration does not fulfill equilibrium, check your "
        "initial external forces, velocities and accelerations!!!");
  }

  if ((dispnorm > 1.0e-14) or (velnorm > 1.0e-14) or (accnorm > 1.0e-14))
  {
    FOUR_C_THROW(
        "Nonlinear inertia terms (input flag 'MASSLIN' not set to 'none') "
        "are only possible for vanishing initial displacements, velocities and "
        "accelerations so far!!!\n"
        "norm disp = %f \n"
        "norm vel  = %f \n"
        "norm acc  = %f",
        dispnorm, velnorm, accnorm);
  }

  if (have_nonlinear_mass() == Inpar::Solid::ml_rotations and
      Core::UTILS::IntegralValue<Inpar::Solid::PredEnum>(*sdynparams, "PREDICT") !=
          Inpar::Solid::pred_constdis)
  {
    FOUR_C_THROW(
        "Only constant displacement consistent velocity and acceleration "
        "predictor possible for multiplicative Genalpha time integration!");
  }

  if (sdynparams != nullptr)
  {
    if (have_nonlinear_mass() == Inpar::Solid::ml_rotations and
        Core::UTILS::IntegralValue<Inpar::Solid::DynamicType>(*sdynparams, "DYNAMICTYP") !=
            Inpar::Solid::dyna_genalpha)
      FOUR_C_THROW(
          "Nonlinear inertia forces for rotational DoFs only implemented "
          "for GenAlpha time integration so far!");
  }
}

/*----------------------------------------------------------------------*/
/* evaluate ordinary internal force */
void Solid::TimInt::apply_force_internal(const double time, const double dt,
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
  discret_->clear_state();
  discret_->set_state("residual displacement", disi);  // these are incremental
  discret_->set_state("displacement", dis);

  if (damping_ == Inpar::Solid::damp_material) discret_->set_state("velocity", vel);
  // fintn_->PutScalar(0.0);  // initialise internal force vector
  discret_->evaluate(p, Teuchos::null, Teuchos::null, fint, Teuchos::null, Teuchos::null);

  discret_->clear_state();
}

/*----------------------------------------------------------------------*/
Inpar::Solid::ConvergenceStatus Solid::TimInt::perform_error_action(
    Inpar::Solid::ConvergenceStatus nonlinsoldiv)
{
  // what to do when nonlinear solver does not converge
  switch (divcontype_)
  {
    case Inpar::Solid::divcont_stop:
    {
      // write restart output of last converged step before stopping
      output(true);

      // we should not get here, FOUR_C_THROW for safety
      FOUR_C_THROW("Nonlinear solver did not converge! ");
      return Inpar::Solid::conv_nonlin_fail;
    }
    case Inpar::Solid::divcont_continue:
    {
      // we should not get here, FOUR_C_THROW for safety
      FOUR_C_THROW("Nonlinear solver did not converge! ");
      return Inpar::Solid::conv_nonlin_fail;
    }
    break;
    case Inpar::Solid::divcont_repeat_step:
    {
      Core::IO::cout << "Nonlinear solver failed to converge repeat time step" << Core::IO::endl;

      // reset step (e.g. quantities on element level)
      reset_step();

      return Inpar::Solid::conv_success;
    }
    break;
    case Inpar::Solid::divcont_halve_step:
    {
      Core::IO::cout << "Nonlinear solver failed to converge at time t= " << timen_
                     << ". Divide timestep in half. "
                     << "Old time step: " << (*dt_)[0] << Core::IO::endl
                     << "New time step: " << 0.5 * (*dt_)[0] << Core::IO::endl
                     << Core::IO::endl;

      // halve the time step size
      (*dt_)[0] = (*dt_)[0] * 0.5;
      // update the number of max time steps
      stepmax_ = stepmax_ + (stepmax_ - stepn_) + 1;
      // reset timen_ because it is set in the constructor
      timen_ = (*time_)[0] + (*dt_)[0];
      // reset step (e.g. quantities on element level)
      reset_step();

      return Inpar::Solid::conv_success;
    }
    break;
    case Inpar::Solid::divcont_adapt_step:
    {
      // maximal possible refinementlevel
      const int maxdivconrefinementlevel = 10;
      const int maxstepmax = 1000000;
      Core::IO::cout << "Nonlinear solver failed to converge at time t= " << timen_
                     << ". Divide timestep in half. "
                     << "Old time step: " << (*dt_)[0] << Core::IO::endl
                     << "New time step: " << 0.5 * (*dt_)[0] << Core::IO::endl
                     << Core::IO::endl;

      // halve the time step size
      (*dt_)[0] = (*dt_)[0] * 0.5;

      // update the number of max time steps
      stepmax_ = stepmax_ + (stepmax_ - stepn_) + 1;
      // reset timen_ because it is set in the constructor
      timen_ = (*time_)[0] + (*dt_)[0];

      divconrefinementlevel_++;
      divconnumfinestep_ = 0;

      if (divconrefinementlevel_ == maxdivconrefinementlevel)
        FOUR_C_THROW(
            "Maximal divercont refinement level reached. Adapt your time basic time step size!");

      if (stepmax_ > maxstepmax) FOUR_C_THROW("Upper level for stepmax_ reached!");

      // reset step (e.g. quantities on element level)
      reset_step();

      return Inpar::Solid::conv_success;
    }
    break;
    case Inpar::Solid::divcont_rand_adapt_step:
    case Inpar::Solid::divcont_rand_adapt_step_ele_err:
    {
      // generate random number between 0.51 and 1.99 (as mean value of random
      // numbers generated on all processors), alternating values larger
      // and smaller than 1.0
      double proc_randnum_get = ((double)rand() / (double)RAND_MAX);
      double proc_randnum = proc_randnum_get;
      double randnum = 1.0;
      discret_->get_comm().SumAll(&proc_randnum, &randnum, 1);
      const double numproc = discret_->get_comm().NumProc();
      randnum /= numproc;
      if (rand_tsfac_ > 1.0)
        rand_tsfac_ = randnum * 0.49 + 0.51;
      else if (rand_tsfac_ < 1.0)
        rand_tsfac_ = randnum * 0.99 + 1.0;
      else
        rand_tsfac_ = randnum * 1.48 + 0.51;
      if (myrank_ == 0)
        Core::IO::cout << "Nonlinear solver failed to converge: modifying time-step size by random "
                          "number between 0.51 and 1.99 -> here: "
                       << rand_tsfac_ << " !" << Core::IO::endl;
      // multiply time-step size by random number
      (*dt_)[0] = (*dt_)[0] * rand_tsfac_;
      // update maximum number of time steps
      stepmax_ = (1.0 / rand_tsfac_) * stepmax_ + (1.0 - (1.0 / rand_tsfac_)) * stepn_ + 1;
      // reset timen_ because it is set in the constructor
      timen_ = (*time_)[0] + (*dt_)[0];
      // reset step (e.g. quantities on element level)
      reset_step();

      return Inpar::Solid::conv_success;
    }
    break;
    case Inpar::Solid::divcont_adapt_penaltycontact:
    {
      // adapt penalty and search parameter
      if (have_contact_meshtying())
      {
        cmtbridge_->get_strategy().modify_penalty();
      }
    }
    break;
    case Inpar::Solid::divcont_repeat_simulation:
    {
      if (nonlinsoldiv == Inpar::Solid::conv_nonlin_fail)
        Core::IO::cout << "Nonlinear solver failed to converge and DIVERCONT = "
                          "repeat_simulation, hence leaving structural time integration "
                       << Core::IO::endl;
      else if (nonlinsoldiv == Inpar::Solid::conv_lin_fail)
        Core::IO::cout << "Linear solver failed to converge and DIVERCONT = "
                          "repeat_simulation, hence leaving structural time integration "
                       << Core::IO::endl;
      else if (nonlinsoldiv == Inpar::Solid::conv_ele_fail)
        Core::IO::cout
            << "Element failure in form of a negative Jacobian determinant and DIVERCONT = "
               "repeat_simulation, hence leaving structural time integration "
            << Core::IO::endl;
      return nonlinsoldiv;  // so that time loop will be aborted
    }
    break;
    case Inpar::Solid::divcont_adapt_3D0Dptc_ele_err:
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
          if (cardvasc0dman_->get_k_ptc() == 0.0)
          {
            Core::IO::cout << "Nonlinear solver failed to converge at time t= " << timen_
                           << ". Increase PTC parameter. "
                           << "Old PTC parameter: " << cardvasc0dman_->get_k_ptc() << Core::IO::endl
                           << "New PTC parameter: " << sum + cardvasc0dman_->get_k_ptc()
                           << Core::IO::endl
                           << Core::IO::endl;
          }
          else
          {
            Core::IO::cout << "Nonlinear solver failed to converge at time t= " << timen_
                           << ". Increase PTC parameter. "
                           << "Old PTC parameter: " << cardvasc0dman_->get_k_ptc() << Core::IO::endl
                           << "New PTC parameter: " << fac * cardvasc0dman_->get_k_ptc()
                           << Core::IO::endl
                           << Core::IO::endl;
          }
        }
        // increase PTC factor
        cardvasc0dman_->modify_k_ptc(sum, fac);

        // adapt penalty parameter
        if (have_contact_meshtying() and divconrefinementlevel_ > adapt_penaltycontact_after)
        {
          if (myrank_ == 0)
          {
            Core::IO::cout
                << "Nonlinear solver still did not converge. Slightly adapt penalty parameter "
                   "for contact."
                << Core::IO::endl;
          }

          cmtbridge_->get_strategy().modify_penalty();
        }

        divconrefinementlevel_++;
        divconnumfinestep_ = 0;
      }

      else
        FOUR_C_THROW(
            "Maximal divercont refinement level reached. Finally nonlinear solver did not "
            "converge. :-(");

      // reset step (e.g. quantities on element level)
      reset_step();

      return Inpar::Solid::conv_success;
    }

    default:
      FOUR_C_THROW("Unknown DIVER_CONT case");
      return Inpar::Solid::conv_nonlin_fail;
      break;
  }
  return Inpar::Solid::conv_success;  // make compiler happy
}

/*----------------------------------------------------------------------*/
/* Set forces due to interface with fluid,
 * the force is expected external-force-like */
void Solid::TimInt::set_force_interface(
    Teuchos::RCP<Epetra_MultiVector> iforce  ///< the force on interface
)
{
  fifc_->Update(1.0, *iforce, 0.0);
}

/*----------------------------------------------------------------------*/
/* apply the new material_displacements        mgit 05/11 / rauch 01/16 */
void Solid::TimInt::apply_dis_mat(Teuchos::RCP<Epetra_Vector> dismat)
{
  // The values in dismatn_ are replaced, because the new absolute material
  // displacement is provided in the argument (not an increment)
  Core::LinAlg::export_to(*dismat, *dismatn_);
}

/*----------------------------------------------------------------------*/
/* Attach file handle for energy file #energyfile_                      */
void Solid::TimInt::attach_energy_file()
{
  if (energyfile_.is_null())
  {
    std::string energyname =
        Global::Problem::instance()->output_control_file()->file_name() + ".energy";
    energyfile_ = Teuchos::rcp(new std::ofstream(energyname.c_str()));
    (*energyfile_) << "# timestep time total_energy"
                   << " kinetic_energy internal_energy external_energy" << std::endl;
  }
}

/*----------------------------------------------------------------------*/
/* Return (rotatory) transformation matrix of local co-ordinate systems */
Teuchos::RCP<const Core::LinAlg::SparseMatrix> Solid::TimInt::get_loc_sys_trafo() const
{
  if (locsysman_ != Teuchos::null) return locsysman_->trafo();

  return Teuchos::null;
}

/*----------------------------------------------------------------------*/
/* Return stiffness matrix as Core::LinAlg::SparseMatrix                      */
Teuchos::RCP<Core::LinAlg::SparseMatrix> Solid::TimInt::system_matrix()
{
  return Teuchos::rcp_dynamic_cast<Core::LinAlg::SparseMatrix>(stiff_);
}

/*----------------------------------------------------------------------*/
/* Return stiffness matrix as Core::LinAlg::BlockSparseMatrix */
Teuchos::RCP<Core::LinAlg::BlockSparseMatrixBase> Solid::TimInt::block_system_matrix()
{
  return Teuchos::rcp_dynamic_cast<Core::LinAlg::BlockSparseMatrixBase>(stiff_);
}

/*----------------------------------------------------------------------*/
/* Return sparse mass matrix                                            */
Teuchos::RCP<Core::LinAlg::SparseMatrix> Solid::TimInt::mass_matrix()
{
  return Teuchos::rcp_dynamic_cast<Core::LinAlg::SparseMatrix>(mass_);
}


/*----------------------------------------------------------------------*/
/* Return domain map of mass matrix                                     */
const Epetra_Map& Solid::TimInt::domain_map() const { return mass_->domain_map(); }

/*----------------------------------------------------------------------*/
/* Creates the field test                                               */
Teuchos::RCP<Core::UTILS::ResultTest> Solid::TimInt::create_field_test()
{
  return Teuchos::rcp(new StruResultTest(*this));
}

/*----------------------------------------------------------------------*/
/* dof map of vector of unknowns                                        */
Teuchos::RCP<const Epetra_Map> Solid::TimInt::dof_row_map()
{
  const Epetra_Map* dofrowmap = discret_->dof_row_map();
  return Teuchos::rcp(new Epetra_Map(*dofrowmap));
}

/*----------------------------------------------------------------------*/
/* dof map of vector of unknowns                                        */
/* new method for multiple dofsets                                      */
Teuchos::RCP<const Epetra_Map> Solid::TimInt::dof_row_map(unsigned nds)
{
  const Epetra_Map* dofrowmap = discret_->dof_row_map(nds);
  return Teuchos::rcp(new Epetra_Map(*dofrowmap));
}

/*----------------------------------------------------------------------*/
/* view of dof map of vector of unknowns                                */
const Epetra_Map* Solid::TimInt::dof_row_map_view() { return discret_->dof_row_map(); }

/*----------------------------------------------------------------------*/
/* reset everything (needed for biofilm simulations)                    */
void Solid::TimInt::reset()
{
  // displacements D_{n}
  dis_ = Teuchos::rcp(new TimeStepping::TimIntMStep<Epetra_Vector>(0, 0, dof_row_map_view(), true));
  // displacements D_{n}
  dismat_ =
      Teuchos::rcp(new TimeStepping::TimIntMStep<Epetra_Vector>(0, 0, dof_row_map_view(), true));
  // velocities V_{n}
  vel_ = Teuchos::rcp(new TimeStepping::TimIntMStep<Epetra_Vector>(0, 0, dof_row_map_view(), true));
  // accelerations A_{n}
  acc_ = Teuchos::rcp(new TimeStepping::TimIntMStep<Epetra_Vector>(0, 0, dof_row_map_view(), true));

  // displacements D_{n+1} at t_{n+1}
  disn_ = Core::LinAlg::CreateVector(*dof_row_map_view(), true);
  // velocities V_{n+1} at t_{n+1}
  veln_ = Core::LinAlg::CreateVector(*dof_row_map_view(), true);
  // accelerations A_{n+1} at t_{n+1}
  accn_ = Core::LinAlg::CreateVector(*dof_row_map_view(), true);
  // create empty interface force vector
  fifc_ = Core::LinAlg::CreateVector(*dof_row_map_view(), true);

  // set initial fields
  set_initial_fields();
}

/*----------------------------------------------------------------------*/
/* set structure displacement vector due to biofilm growth              */
void Solid::TimInt::set_str_gr_disp(Teuchos::RCP<Epetra_Vector> struct_growth_disp)
{
  strgrdisp_ = struct_growth_disp;
}

/*----------------------------------------------------------------------*/
/* Resize MStep Object due to time adaptivity in FSI                    */
void Solid::TimInt::resize_m_step_tim_ada()
{
  // safety checks
  check_is_init();
  check_is_setup();

  // resize time and stepsize fields
  time_->resize(-1, 0, (*time_)[0]);
  dt_->resize(-1, 0, (*dt_)[0]);

  // resize state vectors, AB2 is a 2-step method, thus we need two
  // past steps at t_{n} and t_{n-1}
  dis_->resize(-1, 0, dof_row_map_view(), true);
  vel_->resize(-1, 0, dof_row_map_view(), true);
  acc_->resize(-1, 0, dof_row_map_view(), true);
}

/*----------------------------------------------------------------------*/
/* Expand the dbc map by dofs provided in Epetra_Map maptoadd.          */
void Solid::TimInt::add_dirich_dofs(const Teuchos::RCP<const Epetra_Map> maptoadd)
{
  std::vector<Teuchos::RCP<const Epetra_Map>> condmaps;
  condmaps.push_back(maptoadd);
  condmaps.push_back(get_dbc_map_extractor()->cond_map());
  Teuchos::RCP<Epetra_Map> condmerged = Core::LinAlg::MultiMapExtractor::merge_maps(condmaps);
  *dbcmaps_ = Core::LinAlg::MapExtractor(*(discret_->dof_row_map()), condmerged);
}

/*----------------------------------------------------------------------*/
/* Contract the dbc map by dofs provided in Epetra_Map maptoremove.     */
void Solid::TimInt::remove_dirich_dofs(const Teuchos::RCP<const Epetra_Map> maptoremove)
{
  std::vector<Teuchos::RCP<const Epetra_Map>> othermaps;
  othermaps.push_back(maptoremove);
  othermaps.push_back(get_dbc_map_extractor()->other_map());
  Teuchos::RCP<Epetra_Map> othermerged = Core::LinAlg::MultiMapExtractor::merge_maps(othermaps);
  *dbcmaps_ = Core::LinAlg::MapExtractor(*(discret_->dof_row_map()), othermerged, false);
}

FOUR_C_NAMESPACE_CLOSE

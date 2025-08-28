// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_structure_timint.hpp"

#include "4C_beamcontact_input.hpp"
#include "4C_cardiovascular0d_manager.hpp"
#include "4C_cardiovascular0d_mor_pod.hpp"
#include "4C_comm_utils.hpp"
#include "4C_constraint_manager.hpp"
#include "4C_constraint_solver.hpp"
#include "4C_constraint_springdashpot_manager.hpp"
#include "4C_contact_input.hpp"
#include "4C_contact_meshtying_contact_bridge.hpp"
#include "4C_fem_condition_locsys.hpp"
#include "4C_fem_discretization_faces.hpp"
#include "4C_fem_discretization_utils.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_mortar.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_io_gmsh.hpp"
#include "4C_io_pstream.hpp"
#include "4C_linalg_blocksparsematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_utils_densematrix_communication.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_mat_micromaterial.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mortar_manager_base.hpp"
#include "4C_mortar_strategy_base.hpp"
#include "4C_mortar_utils.hpp"
#include "4C_poroelast_utils.hpp"
#include "4C_solid_3D_ele.hpp"
#include "4C_stru_multi_microstatic.hpp"
#include "4C_structure_resulttest.hpp"
#include "4C_structure_timint_genalpha.hpp"
#include "4C_utils_enum.hpp"

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
    const Teuchos::ParameterList& xparams, std::shared_ptr<Core::FE::Discretization> actdis,
    std::shared_ptr<Core::LinAlg::Solver> solver,
    std::shared_ptr<Core::LinAlg::Solver> contactsolver,
    std::shared_ptr<Core::IO::DiscretizationWriter> output)
    : discret_(actdis),
      myrank_(Core::Communication::my_mpi_rank(actdis->get_comm())),
      solver_(solver),
      contactsolver_(contactsolver),
      solveradapttol_(sdynparams.get<bool>("ADAPTCONV")),
      solveradaptolbetter_(sdynparams.get<double>("ADAPTCONV_BETTER")),
      dbcmaps_(std::make_shared<Core::LinAlg::MapExtractor>()),
      divcontype_(Teuchos::getIntegralValue<Inpar::Solid::DivContAct>(sdynparams, "DIVERCONT")),
      divconrefinementlevel_(0),
      divconnumfinestep_(0),
      sdynparams_(sdynparams),
      output_(output),
      printscreen_(ioparams.get<int>("STDOUTEVERY")),
      printlogo_(bool(printscreen_)),  // no std out no logo
      printiter_(true),                // ADD INPUT PARAMETER
      writerestartevery_(timeparams.get<int>("RESTARTEVERY")),
      writeele_(ioparams.get<bool>("STRUCT_ELE")),
      writestate_(ioparams.get<bool>("STRUCT_DISP")),
      writeresultsevery_(timeparams.get<int>("RESULTSEVERY")),
      writestress_(Teuchos::getIntegralValue<Inpar::Solid::StressType>(ioparams, "STRUCT_STRESS")),
      writestrain_(Teuchos::getIntegralValue<Inpar::Solid::StrainType>(ioparams, "STRUCT_STRAIN")),
      writeplstrain_(
          Teuchos::getIntegralValue<Inpar::Solid::StrainType>(ioparams, "STRUCT_PLASTIC_STRAIN")),
      writeenergyevery_(sdynparams.get<int>("RESEVERYERGY")),
      writesurfactant_(ioparams.get<bool>("STRUCT_SURFACTANT")),
      writerotation_(ioparams.get<bool>("OUTPUT_ROT")),
      energyfile_(nullptr),
      damping_(Teuchos::getIntegralValue<Inpar::Solid::DampKind>(sdynparams, "DAMPING")),
      dampk_(sdynparams.get<double>("K_DAMP")),
      dampm_(sdynparams.get<double>("M_DAMP")),
      conman_(nullptr),
      consolv_(nullptr),
      cardvasc0dman_(nullptr),
      springman_(nullptr),
      cmtbridge_(nullptr),
      beamcman_(nullptr),
      locsysman_(nullptr),
      pressure_(nullptr),
      gmsh_out_(false),
      time_(nullptr),
      timen_(0.0),
      dt_(nullptr),
      timemax_(timeparams.get<double>("MAXTIME")),
      stepmax_(timeparams.get<int>("NUMSTEP")),
      step_(0),
      stepn_(0),
      rand_tsfac_(1.0),
      firstoutputofrun_(true),
      lumpmass_(sdynparams.get<bool>("LUMPMASS")),
      zeros_(nullptr),
      dis_(nullptr),
      vel_(nullptr),
      acc_(nullptr),
      disn_(nullptr),
      veln_(nullptr),
      accn_(nullptr),
      fifc_(nullptr),
      fresn_str_(nullptr),
      fintn_str_(nullptr),
      stiff_(nullptr),
      mass_(nullptr),
      damp_(nullptr),
      timer_(std::make_shared<Teuchos::Time>("", true)),
      dtsolve_(0.0),
      dtele_(0.0),
      dtcmt_(0.0),
      strgrdisp_(nullptr),
      mor_(nullptr),
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
    std::shared_ptr<Core::FE::Discretization> actdis, std::shared_ptr<Core::LinAlg::Solver> solver)
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
  time_ = std::make_shared<TimeStepping::TimIntMStep<double>>(
      0, 0, 0.0);  // HERE SHOULD BE SOMETHING LIKE (sdynparams.get<double>("TIMEINIT"))
  dt_ =
      std::make_shared<TimeStepping::TimIntMStep<double>>(0, 0, timeparams.get<double>("TIMESTEP"));
  step_ = 0;
  timen_ = (*time_)[0] + (*dt_)[0];  // set target time to initial time plus step size
  stepn_ = step_ + 1;

  // output file for energy
  if ((writeenergyevery_ != 0) and (myrank_ == 0)) attach_energy_file();

  // initialize constraint manager
  conman_ = std::make_shared<Constraints::ConstrManager>();
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
  mor_ = std::make_shared<Cardiovascular0D::ProperOrthogonalDecomposition>(dof_row_map(),
      Global::Problem::instance()->mor_params().get<std::string>("POD_MATRIX"),
      Global::Problem::instance()->output_control_file()->input_file_name());

  // initialize 0D cardiovascular manager
  cardvasc0dman_ =
      std::make_shared<FourC::Utils::Cardiovascular0DManager>(discret_, (*dis_)(0), sdynparams_,
          Global::Problem::instance()->cardiovascular0_d_structural_params(), *solver_, mor_);

  // initialize spring dashpot manager
  springman_ = std::make_shared<Constraints::SpringDashpotManager>(discret_);


  // initialize constraint solver if constraints are defined
  if (conman_->have_constraint())
  {
    consolv_ =
        std::make_shared<Constraints::ConstraintSolver>(discret_, *solver_, dbcmaps_, sdynparams_);
  }

  // check for mortar contact or meshtying
  {
    // If mortar contact or meshtying is chosen in the input file, then a
    // corresponding manager object stored via #cmtman_ is created and all relevant
    // stuff is initialized. Else, #cmtman_ remains a nullptr pointer.
    prepare_contact_meshtying(sdynparams_);
  }

  // check whether we have locsys BCs and create LocSysManager if so
  // after checking
  {
    std::vector<const Core::Conditions::Condition*> locsysconditions;
    discret_->get_condition("Locsys", locsysconditions);
    if (locsysconditions.size())
    {
      locsysman_ = std::make_shared<Core::Conditions::LocsysManager>(
          *discret_, Global::Problem::instance()->n_dim());
      // in case we have no time dependent locsys conditions in our problem,
      // this is the only time where the whole setup routine is conducted.
      locsysman_->update(-1.0, {}, Global::Problem::instance()->function_manager());
    }
  }

  // Check for porosity dofs within the structure and build a map extractor if necessary
  porositysplitter_ = PoroElast::Utils::build_poro_splitter(*discret_);

  // ensure that no runtime output is activated because it is not implemented
  FOUR_C_ASSERT_ALWAYS(not Global::Problem::instance()
                           ->io_params()
                           .sublist("RUNTIME VTK OUTPUT")
                           .sublist("STRUCTURE")
                           .get<bool>("OUTPUT_STRUCTURE"),
      "Runtime output is not available in the old structure time integration! You need to take the "
      "new one, i.e. set `INT_STRATEGY: Standard`!");

  // we have successfully set up this class
  set_is_setup(true);
}

/*----------------------------------------------------------------------------------------------*
 * Create all solution vectors
 *----------------------------------------------------------------------------------------------*/
void Solid::TimInt::create_all_solution_vectors()
{
  // displacements D_{n}
  dis_ = std::make_shared<TimeStepping::TimIntMStep<Core::LinAlg::Vector<double>>>(
      0, 0, dof_row_map_view(), true);
  // velocities V_{n}
  vel_ = std::make_shared<TimeStepping::TimIntMStep<Core::LinAlg::Vector<double>>>(
      0, 0, dof_row_map_view(), true);
  // accelerations A_{n}
  acc_ = std::make_shared<TimeStepping::TimIntMStep<Core::LinAlg::Vector<double>>>(
      0, 0, dof_row_map_view(), true);

  // displacements D_{n+1} at t_{n+1}
  disn_ = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map_view(), true);

  // velocities V_{n+1} at t_{n+1}
  veln_ = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map_view(), true);
  // accelerations A_{n+1} at t_{n+1}
  accn_ = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map_view(), true);
  // create empty interface force vector
  fifc_ = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map_view(), true);
}

/*-------------------------------------------------------------------------------------------*
 * Create matrices when setting up time integrator
 *-------------------------------------------------------------------------------------------*/
void Solid::TimInt::create_fields()
{
  // a zero vector of full length
  zeros_ = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map_view(), true);

  // Map containing Dirichlet DOFs
  {
    Teuchos::ParameterList p;
    p.set("total time", timen_);
    p.set<const Core::Utils::FunctionManager*>(
        "function_manager", &Global::Problem::instance()->function_manager());

    // if this is a NURBS discretization, get the solver parameters for solving the least squares
    // problem
    if (std::dynamic_pointer_cast<Core::FE::Nurbs::NurbsDiscretization>(discret_) != nullptr)
    {
      const Teuchos::ParameterList& nurbs_params = Global::Problem::instance()->nurbs_params();

      const bool is_ls_dbc_needed = nurbs_params.get<bool>("DO_LS_DBC_PROJECTION");

      if (is_ls_dbc_needed)
      {
        const int ls_dbc_solver_num = nurbs_params.get<int>("SOLVER_LS_DBC_PROJECTION");

        if (ls_dbc_solver_num == (-1))
        {
          FOUR_C_THROW(
              "No linear solver defined for the projection of least squares Dirichlet "
              "boundary conditions for the NURBS discretization. Please set "
              "SOLVER_LS_DBC_PROJECTION in NURBS to a valid number!");
        }

        // Save solver parameters
        p.sublist("ls_dbc_solver_params")
            .setParameters(Global::Problem::instance()->solver_params(ls_dbc_solver_num));
      }
    }

    discret_->evaluate_dirichlet(p, zeros_, nullptr, nullptr, nullptr, dbcmaps_);
    zeros_->put_scalar(0.0);  // just in case of change
  }

  // create empty matrices
  stiff_ = std::make_shared<Core::LinAlg::SparseMatrix>(*dof_row_map_view(), 81, false, true);
  mass_ = std::make_shared<Core::LinAlg::SparseMatrix>(*dof_row_map_view(), 81, false, true);
  if (damping_ != Inpar::Solid::damp_none)
  {
    if (have_nonlinear_mass() == Inpar::Solid::MassLin::ml_none)
    {
      damp_ = std::make_shared<Core::LinAlg::SparseMatrix>(*dof_row_map_view(), 81, false, true);
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
/* Set initial fields in structure (e.g. initial velocities */
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
      Global::Problem::instance()->function_manager(), field, *(*vel_)(0), localdofs);

  // set initial porosity field if existing
  const std::string porosityfield = "Porosity";
  std::vector<int> porositylocaldofs;
  porositylocaldofs.push_back(Global::Problem::instance()->n_dim());

  discret_->evaluate_initial_field(Global::Problem::instance()->function_manager(), porosityfield,
      *(*dis_)(0), porositylocaldofs);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Solid::TimInt::prepare_contact_meshtying(const Teuchos::ParameterList& sdynparams)
{
  TEUCHOS_FUNC_TIME_MONITOR("Solid::TimInt::prepare_contact_meshtying");

  // some parameters
  const Teuchos::ParameterList& smortar = Global::Problem::instance()->mortar_coupling_params();
  const Teuchos::ParameterList& scontact = Global::Problem::instance()->contact_dynamic_params();
  auto shapefcn = Teuchos::getIntegralValue<Inpar::Mortar::ShapeFcn>(smortar, "LM_SHAPEFCN");
  auto soltype = Teuchos::getIntegralValue<CONTACT::SolvingStrategy>(scontact, "STRATEGY");
  auto systype = Teuchos::getIntegralValue<CONTACT::SystemType>(scontact, "SYSTEM");
  auto algorithm = Teuchos::getIntegralValue<Inpar::Mortar::AlgorithmType>(smortar, "ALGORITHM");

  // check mortar contact or meshtying conditions
  std::vector<const Core::Conditions::Condition*> mortarconditions;
  std::vector<const Core::Conditions::Condition*> contactconditions;

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
  const bool do_endtime = scontact.get<bool>("CONTACTFORCE_ENDTIME");
  if (!do_endtime) time_integration_factor = tim_int_param();

  // create instance for meshtying contact bridge
  cmtbridge_ = std::make_shared<CONTACT::MeshtyingContactBridge>(
      *discret_, mortarconditions, contactconditions, time_integration_factor);

  cmtbridge_->store_dirichlet_status(dbcmaps_);
  cmtbridge_->set_state(*zeros_);

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
    auto mesh_relocation_parameter = Teuchos::getIntegralValue<Inpar::Mortar::MeshRelocation>(
        Global::Problem::instance()->mortar_coupling_params(), "MESH_RELOCATION");

    if (mesh_relocation_parameter == Inpar::Mortar::relocation_initial)
    {
      // (2) perform mesh initialization for rotational invariance (interface)
      // and return the modified slave node positions in vector Xslavemod
      std::shared_ptr<const Core::LinAlg::Vector<double>> Xslavemod =
          cmtbridge_->mt_manager()->get_strategy().mesh_initialization();

      // (3) apply result of mesh initialization to underlying problem discretization
      apply_mesh_initialization(Xslavemod);
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

  //**********************************************************************
  // prepare solvers for contact/meshtying problem
  //**********************************************************************
  {
    // only plausibility check, that a contact solver is available
    if (contactsolver_ == nullptr)
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
        if (systype == CONTACT::SystemType::saddlepoint)
        {
          if (soltype == CONTACT::SolvingStrategy::lagmult &&
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
          else if (soltype == CONTACT::SolvingStrategy::lagmult &&
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
          else if (soltype == CONTACT::SolvingStrategy::multiscale &&
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
          else if (soltype == CONTACT::SolvingStrategy::multiscale &&
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
          else if (soltype == CONTACT::SolvingStrategy::lagmult &&
                   Teuchos::getIntegralValue<Inpar::Mortar::LagMultQuad>(smortar, "LM_QUAD") ==
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
          else if (soltype == CONTACT::SolvingStrategy::lagmult &&
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
          else if (soltype == CONTACT::SolvingStrategy::penalty &&
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
          else if (soltype == CONTACT::SolvingStrategy::penalty &&
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
          else if (soltype == CONTACT::SolvingStrategy::uzawa &&
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
          else if (soltype == CONTACT::SolvingStrategy::uzawa &&
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
        else if (systype == CONTACT::SystemType::condensed ||
                 systype == CONTACT::SystemType::condensed_lagmult)
        {
          if (soltype == CONTACT::SolvingStrategy::lagmult && shapefcn == Inpar::Mortar::shape_dual)
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
          else if (soltype == CONTACT::SolvingStrategy::lagmult &&
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
          else if (soltype == CONTACT::SolvingStrategy::lagmult &&
                   Teuchos::getIntegralValue<Inpar::Mortar::LagMultQuad>(smortar, "LM_QUAD") ==
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
          else if (soltype == CONTACT::SolvingStrategy::multiscale &&
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
          else if (soltype == CONTACT::SolvingStrategy::multiscale &&
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
          else if (soltype == CONTACT::SolvingStrategy::penalty &&
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
          else if (soltype == CONTACT::SolvingStrategy::penalty &&
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
          else if (soltype == CONTACT::SolvingStrategy::uzawa &&
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
          else if (soltype == CONTACT::SolvingStrategy::uzawa &&
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
          {
            FOUR_C_THROW("Invalid strategy or shape function type for contact/meshtying");
          }
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
      {
        FOUR_C_THROW("Invalid system type for contact/meshtying");
      }
    }
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Solid::TimInt::apply_mesh_initialization(
    std::shared_ptr<const Core::LinAlg::Vector<double>> Xslavemod)
{
  // check modified positions vector
  if (Xslavemod == nullptr) return;

  // create fully overlapping slave node map
  std::shared_ptr<const Core::LinAlg::Map> slavemap =
      cmtbridge_->mt_manager()->get_strategy().slave_row_nodes_ptr();
  std::shared_ptr<Core::LinAlg::Map> allreduceslavemap = Core::LinAlg::allreduce_e_map(*slavemap);

  // export modified node positions to column map of problem discretization
  std::shared_ptr<Core::LinAlg::Vector<double>> Xslavemodcol =
      std::make_shared<Core::LinAlg::Vector<double>>(*discret_->dof_col_map(), false);
  Core::LinAlg::export_to(*Xslavemod, *Xslavemodcol);

  const int numnode = allreduceslavemap->num_my_elements();
  const int numdim = Global::Problem::instance()->n_dim();
  const Core::LinAlg::Vector<double>& gvector = *Xslavemodcol;

  // loop over all slave nodes (for all procs)
  for (int index = 0; index < numnode; ++index)
  {
    int gid = allreduceslavemap->gid(index);

    // only do something for nodes in my column map
    int ilid = discret_->node_col_map()->lid(gid);
    if (ilid < 0) continue;

    Core::Nodes::Node* mynode = discret_->g_node(gid);

    // get degrees of freedom associated with this fluid/structure node
    std::vector<int> nodedofs = discret_->dof(0, mynode);
    std::vector<double> nvector(numdim, 0.0);

    // create new position vector
    for (int i = 0; i < numdim; ++i)
    {
      const int lid = gvector.get_map().lid(nodedofs[i]);

      if (lid < 0)
        FOUR_C_THROW("Proc {}: Cannot find gid={} in Core::LinAlg::Vector<double>",
            Core::Communication::my_mpi_rank(gvector.get_comm()), nodedofs[i]);

      nvector[i] += gvector.local_values_as_span()[lid];
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
/* equilibrate system at initial state
 * and identify consistent accelerations */
void Solid::TimInt::determine_mass_damp_consist_accel()
{
  // temporary right hand sinde vector in this routing
  std::shared_ptr<Core::LinAlg::Vector<double>> rhs =
      std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map_view(), true);  // right hand side
  // temporary force vectors in this routine
  std::shared_ptr<Core::LinAlg::Vector<double>> fext =
      std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map_view(), true);  // external force
  std::shared_ptr<Core::LinAlg::Vector<double>> fint =
      std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map_view(), true);  // internal force

  // initialise matrices
  stiff_->zero();
  mass_->zero();

  // auxiliary vector in order to store accelerations of inhomogeneous Dirichilet-DoFs
  // Meier 2015: This contribution is necessary in order to determine correct initial
  // accelerations in case of inhomogeneous Dirichlet conditions
  std::shared_ptr<Core::LinAlg::Vector<double>> acc_aux =
      std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map_view(), true);
  acc_aux->put_scalar(0.0);

  // overwrite initial state vectors with DirichletBCs
  apply_dirichlet_bc((*time_)[0], (*dis_)(0), (*vel_)(0), acc_aux, false);

  /* get external force (no linearization since we assume Rayleigh damping
   * to be independent of follower loads) */
  apply_force_external((*time_)[0], (*dis_)(0), disn_, *(*vel_)(0), *fext);

  // get initial internal force and stiffness and mass
  {
    // compute new inner radius
    discret_->clear_state();
    discret_->set_state(0, "displacement", *(*dis_)(0));

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
    if (pressure_ != nullptr) p.set("volume", 0.0);
    if (fresn_str_ != nullptr)
    {
      p.set<int>("MyPID", myrank_);
      p.set<double>("cond_rhs_norm", 0.);
    }

    // set vector values needed by elements
    discret_->clear_state();
    // extended set_state(0,...) in case of multiple dofsets (e.g. TSI)
    discret_->set_state(0, "residual displacement", *zeros_);
    discret_->set_state(0, "displacement", *(*dis_)(0));
    discret_->set_state(0, "velocity", *(*vel_)(0));

    // The acceleration is only used as a dummy here and should not be applied inside an element,
    // since this is not the consistent initial acceleration vector which will be determined later
    // on
    discret_->set_state(0, "acceleration", *acc_aux);

    if (damping_ == Inpar::Solid::damp_material) discret_->set_state(0, "velocity", *(*vel_)(0));

    discret_->evaluate(p, stiff_, mass_, fint, nullptr, fintn_str_);
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
  std::shared_ptr<Core::LinAlg::SparseOperator> mass = nullptr;
  // Meier 2015: Here, we apply a deep copy in order to not perform the Dirichlet conditions on the
  // constant matrix mass_ later on. This is necessary since we need the original mass matrix mass_
  // (without blanked rows) on the Dirichlet DoFs in order to calculate correct reaction forces
  // (Christoph Meier)
  mass =
      std::make_shared<Core::LinAlg::SparseMatrix>(*mass_matrix(), Core::LinAlg::DataAccess::Copy);

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
    rhs->update(-1.0, *fint, 1.0, *fext, -1.0);

    // Contribution to rhs due to inertia forces of inhomogeneous Dirichlet conditions
    std::shared_ptr<Core::LinAlg::Vector<double>> finert0 =
        std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map_view(), true);
    finert0->put_scalar(0.0);
    mass_->multiply(false, *acc_aux, *finert0);
    rhs->update(-1.0, *finert0, 1.0);

    // blank RHS and system matrix on DBC DOFs
    dbcmaps_->insert_cond_vector(*dbcmaps_->extract_cond_vector(*zeros_), *rhs);

    // Apply Dirichlet conditions also to mass matrix (which represents the system matrix of
    // the considered linear system of equations)
    mass->apply_dirichlet(*(dbcmaps_->cond_map()));

    if (pressure_ != nullptr)
    {
      pressure_->insert_cond_vector(*pressure_->extract_cond_vector(*zeros_), *rhs);
      mass->apply_dirichlet(*(pressure_->cond_map()));
    }
    if (porositysplitter_ != nullptr)
    {
      porositysplitter_->insert_cond_vector(*porositysplitter_->extract_cond_vector(*zeros_), *rhs);
      mass->apply_dirichlet(*(porositysplitter_->cond_map()));
    }

    // Meier 2015: Due to the Dirichlet conditions applied to the mass matrix, we solely solve
    // for the accelerations at non-Dirichlet DoFs while the resulting accelerations at the
    // Dirichlet-DoFs will be zero. Therefore, the accelerations at DoFs with inhomogeneous
    // Dirichlet conditions will be added below at *).
    Core::LinAlg::SolverParams solver_params;
    solver_params.refactor = true;
    solver_params.reset = true;
    solver_->solve(mass, (*acc_)(0), rhs, solver_params);

    //*) Add contributions of inhomogeneous DBCs
    (*acc_)(0)->update(1.0, *acc_aux, 1.0);
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
      "(Re-)Evaluation of only the mass matrix and inertial forces is "
      "not implemented in the base class.\n Set 'MASSLIN' to 'No' in "
      "--STRUCTURAL DYNAMIC if you want to use the chosen timint scheme.");
}

/*---------------------------------------------------------------*/
/* Apply Dirichlet boundary conditions on provided state vectors */
void Solid::TimInt::apply_dirichlet_bc(const double time,
    std::shared_ptr<Core::LinAlg::Vector<double>> dis,
    std::shared_ptr<Core::LinAlg::Vector<double>> vel,
    std::shared_ptr<Core::LinAlg::Vector<double>> acc, bool recreatemap)
{
  // In the case of local coordinate systems, we have to rotate forward ...
  // --------------------------------------------------------------------------------
  if (locsysman_ != nullptr)
  {
    if (dis != nullptr) locsysman_->rotate_global_to_local(*dis, true);
    if (vel != nullptr) locsysman_->rotate_global_to_local(*vel);
    if (acc != nullptr) locsysman_->rotate_global_to_local(*acc);
  }

  // Apply DBCs
  // --------------------------------------------------------------------------------
  // needed parameters
  Teuchos::ParameterList p;
  p.set("total time", time);  // target time
  p.set<const Core::Utils::FunctionManager*>(
      "function_manager", &Global::Problem::instance()->function_manager());

  // if this is a NURBS discretization, get the solver parameters for solving the least squares
  // problem
  if (std::dynamic_pointer_cast<Core::FE::Nurbs::NurbsDiscretization>(discret_) != nullptr)
  {
    const Teuchos::ParameterList& nurbs_params = Global::Problem::instance()->nurbs_params();

    const bool is_ls_dbc_needed = nurbs_params.get<bool>("DO_LS_DBC_PROJECTION");

    if (is_ls_dbc_needed)
    {
      const int ls_dbc_solver_num = nurbs_params.get<int>("SOLVER_LS_DBC_PROJECTION");

      if (ls_dbc_solver_num == (-1))
      {
        FOUR_C_THROW(
            "No linear solver defined for the projection of least squares Dirichlet "
            "boundary conditions for the NURBS discretization. Please set "
            "SOLVER_LS_DBC_PROJECTION in NURBS to a valid number!");
      }

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
    discret_->evaluate_dirichlet(p, dis, vel, acc, nullptr, dbcmaps_);
  }
  else
  {
    discret_->evaluate_dirichlet(p, dis, vel, acc, nullptr, nullptr);
  }
  discret_->clear_state();

  // In the case of local coordinate systems, we have to rotate back into global Cartesian frame
  // --------------------------------------------------------------------------------
  if (locsysman_ != nullptr)
  {
    if (dis != nullptr) locsysman_->rotate_local_to_global(*dis, true);
    if (vel != nullptr) locsysman_->rotate_local_to_global(*vel);
    if (acc != nullptr) locsysman_->rotate_local_to_global(*acc);
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
  }
}

/*----------------------------------------------------------------------*/
/* Velocity update method (VUM) for contact */
void Solid::TimInt::update_step_contact_vum()
{
  if (have_contact_meshtying())
  {
    const bool do_vum = cmtbridge_->get_strategy().params().get<bool>("VELOCITY_UPDATE");

    //********************************************************************
    // VELOCITY UPDATE METHOD
    //********************************************************************
    if (do_vum)
    {
      // check for actual contact and leave if active set empty
      bool isincontact = cmtbridge_->get_strategy().is_in_contact();
      if (!isincontact) return;

      // check for contact force evaluation
      const bool do_end = cmtbridge_->get_strategy().params().get<bool>("CONTACTFORCE_ENDTIME");
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
      if (Teuchos::getIntegralValue<Inpar::Solid::DynamicType>(sdynparams, "DYNAMICTYPE") ==
          Inpar::Solid::DynamicType::GenAlpha)
      {
        auto* genAlpha = dynamic_cast<Solid::TimIntGenAlpha*>(this);
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
      const Core::LinAlg::Map* dofmap = discret_->dof_row_map();
      std::shared_ptr<Core::LinAlg::Map> activenodemap =
          std::make_shared<Core::LinAlg::Map>(*cmtbridge_->get_strategy().active_row_nodes());
      std::shared_ptr<const Core::LinAlg::Map> slavenodemap =
          cmtbridge_->get_strategy().slave_row_nodes_ptr();
      std::shared_ptr<const Core::LinAlg::Map> notredistslavedofmap =
          cmtbridge_->get_strategy().non_redist_slave_row_dofs();
      std::shared_ptr<const Core::LinAlg::Map> notredistmasterdofmap =
          cmtbridge_->get_strategy().non_redist_master_row_dofs();
      std::shared_ptr<Core::LinAlg::Map> notactivenodemap =
          Core::LinAlg::split_map(*slavenodemap, *activenodemap);

      // the lumped mass matrix and its inverse
      if (lumpmass_ == false)
      {
        FOUR_C_THROW("***** WARNING: VelUpdate ONLY for lumped mass matrix -> skipping ****");
        return;
      }
      std::shared_ptr<Core::LinAlg::SparseMatrix> Mass =
          std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(mass_);
      Core::LinAlg::SparseMatrix Minv(*Mass);
      std::shared_ptr<Core::LinAlg::Vector<double>> diag =
          std::make_shared<Core::LinAlg::Vector<double>>(*dofmap, true);
      Minv.extract_diagonal_copy(*diag);
      diag->reciprocal(*diag);
      Minv.replace_diagonal_values(*diag);
      Minv.complete(*dofmap, *dofmap);

      // displacement increment Dd
      std::shared_ptr<Core::LinAlg::Vector<double>> Dd =
          std::make_shared<Core::LinAlg::Vector<double>>(*dofmap, true);
      Dd->update(1.0, *disn_, 0.0);
      Dd->update(-1.0, (*dis_)[0], 1.0);

      // mortar operator Bc
      std::shared_ptr<const Core::LinAlg::SparseMatrix> Mmat =
          cmtbridge_->get_strategy().m_matrix();
      std::shared_ptr<const Core::LinAlg::SparseMatrix> Dmat =
          cmtbridge_->get_strategy().d_matrix();
      Core::LinAlg::Map slavedofmap(Dmat->range_map());
      Core::LinAlg::SparseMatrix Bc(*dofmap, 10);
      std::shared_ptr<const Core::LinAlg::SparseMatrix> M =
          std::make_shared<Core::LinAlg::SparseMatrix>(slavedofmap, 10);
      std::shared_ptr<const Core::LinAlg::SparseMatrix> D =
          std::make_shared<Core::LinAlg::SparseMatrix>(slavedofmap, 10);
      if (Teuchos::getIntegralValue<Inpar::Mortar::ParallelRedist>(
              cmtbridge_->get_strategy().params().sublist("PARALLEL REDISTRIBUTION"),
              "PARALLEL_REDIST") != Inpar::Mortar::ParallelRedist::redist_none)
      {
        M = Core::LinAlg::matrix_col_transform(*Mmat, *notredistmasterdofmap);
        D = Core::LinAlg::matrix_col_transform(*Dmat, *notredistslavedofmap);
      }
      else
      {
        M = Mmat;
        D = Dmat;
      }
      Bc.add(*M, true, -1.0, 1.0);
      Bc.add(*D, true, 1.0, 1.0);
      Bc.complete(slavedofmap, *dofmap);
      Bc.apply_dirichlet(*(dbcmaps_->cond_map()), false);

      // matrix of the normal vectors
      std::shared_ptr<Core::LinAlg::SparseMatrix> N =
          cmtbridge_->get_strategy().evaluate_normals(disn_);

      // lagrange multiplier z
      std::shared_ptr<const Core::LinAlg::Vector<double>> LM =
          cmtbridge_->get_strategy().lagrange_multiplier();
      std::shared_ptr<Core::LinAlg::Vector<double>> Z =
          std::make_shared<Core::LinAlg::Vector<double>>(*slavenodemap, true);
      std::shared_ptr<Core::LinAlg::Vector<double>> z =
          std::make_shared<Core::LinAlg::Vector<double>>(*activenodemap, true);
      N->multiply(false, *LM, *Z);
      Core::LinAlg::export_to(*Z, *z);

      // auxiliary operator BN = Bc * N
      std::shared_ptr<Core::LinAlg::SparseMatrix> BN =
          Core::LinAlg::matrix_multiply(Bc, false, *N, true, false, false, true);

      // operator A
      std::shared_ptr<Core::LinAlg::SparseMatrix> tempmtx1;
      std::shared_ptr<Core::LinAlg::SparseMatrix> tempmtx2;
      std::shared_ptr<Core::LinAlg::SparseMatrix> tempmtx3;
      std::shared_ptr<Core::LinAlg::SparseMatrix> A;
      std::shared_ptr<Core::LinAlg::SparseMatrix> Atemp1 =
          Core::LinAlg::matrix_multiply(*BN, true, Minv, false, false, false, true);
      std::shared_ptr<Core::LinAlg::SparseMatrix> Atemp2 =
          Core::LinAlg::matrix_multiply(*Atemp1, false, *BN, false, false, false, true);
      Atemp2->scale(R4);
      Core::LinAlg::split_matrix2x2(Atemp2, notactivenodemap, activenodemap, notactivenodemap,
          activenodemap, tempmtx1, tempmtx2, tempmtx3, A);
      A->complete(*activenodemap, *activenodemap);

      // diagonal of A
      std::shared_ptr<Core::LinAlg::Vector<double>> AD =
          std::make_shared<Core::LinAlg::Vector<double>>(*activenodemap, true);
      A->extract_diagonal_copy(*AD);

      // operator b
      std::shared_ptr<Core::LinAlg::Vector<double>> btemp1 =
          std::make_shared<Core::LinAlg::Vector<double>>(*dofmap, true);
      std::shared_ptr<Core::LinAlg::Vector<double>> btemp2 =
          std::make_shared<Core::LinAlg::Vector<double>>(*slavenodemap, true);
      std::shared_ptr<Core::LinAlg::Vector<double>> b =
          std::make_shared<Core::LinAlg::Vector<double>>(*activenodemap, true);
      btemp1->update(R1, *Dd, 0.0);
      btemp1->update(R2, (*vel_)[0], 1.0);
      btemp1->update(R3, (*acc_)[0], 1.0);
      BN->multiply(true, *btemp1, *btemp2);
      Core::LinAlg::export_to(*btemp2, *b);

      // operator c
      std::shared_ptr<Core::LinAlg::Vector<double>> ctemp =
          std::make_shared<Core::LinAlg::Vector<double>>(*slavenodemap, true);
      std::shared_ptr<Core::LinAlg::Vector<double>> c =
          std::make_shared<Core::LinAlg::Vector<double>>(*activenodemap, true);
      BN->multiply(true, *Dd, *ctemp);
      Core::LinAlg::export_to(*ctemp, *c);

      // contact work wc
      std::shared_ptr<Core::LinAlg::Vector<double>> wc =
          std::make_shared<Core::LinAlg::Vector<double>>(*activenodemap, true);
      wc->multiply(1.0, *c, *z, 0.0);

      // gain and loss of energy
      double gain = 0;
      double loss = 0;
      std::shared_ptr<Core::LinAlg::Vector<double>> wp =
          std::make_shared<Core::LinAlg::Vector<double>>(*activenodemap, true);
      std::shared_ptr<Core::LinAlg::Vector<double>> wn =
          std::make_shared<Core::LinAlg::Vector<double>>(*activenodemap, true);
      std::shared_ptr<Core::LinAlg::Vector<double>> wd =
          std::make_shared<Core::LinAlg::Vector<double>>(*activenodemap, true);
      std::shared_ptr<Core::LinAlg::Vector<double>> wt =
          std::make_shared<Core::LinAlg::Vector<double>>(*activenodemap, true);
      for (int i = 0; i < activenodemap->num_my_elements(); ++i)
      {
        if (wc->local_values_as_span()[i] > 0)
        {
          (*wp).get_values()[i] = wc->local_values_as_span()[i];
          (*wn).get_values()[i] = 0;
          (*wd).get_values()[i] = 0;
          (*wt).get_values()[i] = 0;
        }
        else
        {
          (*wp).get_values()[i] = 0;
          (*wn).get_values()[i] = wc->local_values_as_span()[i];
          (*wd).get_values()[i] =
              pow(b->local_values_as_span()[i], 2) / (4 * AD->local_values_as_span()[i]);
          if (wc->local_values_as_span()[i] > wd->local_values_as_span()[i])
          {
            (*wt).get_values()[i] = wc->local_values_as_span()[i];
          }
          else
          {
            (*wt).get_values()[i] = wd->local_values_as_span()[i];
          }
        }
      }
      wp->norm_1(&loss);
      wn->norm_1(&gain);

      // manipulated contact work w
      double tolerance = 0.01;
      std::shared_ptr<Core::LinAlg::Vector<double>> wtemp1 =
          std::make_shared<Core::LinAlg::Vector<double>>(*activenodemap, true);
      std::shared_ptr<Core::LinAlg::Vector<double>> wtemp2 =
          std::make_shared<Core::LinAlg::Vector<double>>(*activenodemap, true);
      std::shared_ptr<Core::LinAlg::Vector<double>> w =
          std::make_shared<Core::LinAlg::Vector<double>>(*activenodemap, true);
      if (abs(gain - loss) < 1.0e-8)
      {
        return;
      }
      else if (gain > loss)
      {
        double C = (gain - loss) / gain;
        wtemp1->update(C, *wn, 0.0);
        for (int i = 0; i < activenodemap->num_my_elements(); ++i)
        {
          (*wtemp2).get_values()[i] =
              pow(b->local_values_as_span()[i], 2) / (4 * AD->local_values_as_span()[i]);
          if (wtemp1->local_values_as_span()[i] > wtemp2->local_values_as_span()[i])
          {
            (*w).get_values()[i] = wtemp1->local_values_as_span()[i];
          }
          else
          {
            (*w).get_values()[i] = (1 - tolerance) * wtemp2->local_values_as_span()[i];
            std::cout << "***** WARNING: VelUpdate is not able to compensate the gain of energy****"
                      << std::endl;
          }
        }
      }
      else
      {
        double C = (loss - gain) / loss;
        w->update(C, *wp, 0.0);
      }

      // (1) initial solution p_0
      std::shared_ptr<Core::LinAlg::Vector<double>> p1 =
          std::make_shared<Core::LinAlg::Vector<double>>(*activenodemap, true);
      std::shared_ptr<Core::LinAlg::Vector<double>> p2 =
          std::make_shared<Core::LinAlg::Vector<double>>(*activenodemap, true);
      std::shared_ptr<Core::LinAlg::Vector<double>> p =
          std::make_shared<Core::LinAlg::Vector<double>>(*activenodemap, true);
      if (gain > loss)
      {
        for (int i = 0; i < activenodemap->num_my_elements(); ++i)
        {
          (*p1).get_values()[i] =
              (-b->local_values_as_span()[i] +
                  pow(pow(b->local_values_as_span()[i], 2) -
                          4 * AD->local_values_as_span()[i] * w->local_values_as_span()[i],
                      0.5)) /
              (2 * AD->local_values_as_span()[i]);

          (*p2).get_values()[i] =
              (-b->local_values_as_span()[i] -
                  pow(pow(b->local_values_as_span()[i], 2) -
                          4 * AD->local_values_as_span()[i] * w->local_values_as_span()[i],
                      0.5)) /
              (2 * AD->local_values_as_span()[i]);
          if (w->local_values_as_span()[i] == 0)
            (*p).get_values()[i] = 0;
          else if (abs(p1->local_values_as_span()[i]) < abs(p2->local_values_as_span()[i]))
            (*p).get_values()[i] = p1->local_values_as_span()[i];
          else
            (*p).get_values()[i] = p2->local_values_as_span()[i];
        }
      }
      else
      {
        for (int i = 0; i < activenodemap->num_my_elements(); ++i)
        {
          (*p1).get_values()[i] =
              (-b->local_values_as_span()[i] +
                  pow(pow(b->local_values_as_span()[i], 2) -
                          4 * AD->local_values_as_span()[i] * w->local_values_as_span()[i],
                      0.5)) /
              (2 * AD->local_values_as_span()[i]);

          (*p2).get_values()[i] =
              -b->local_values_as_span()[i] -
              pow(pow(b->local_values_as_span()[i], 2) -
                      4 * AD->local_values_as_span()[i] * w->local_values_as_span()[i],
                  0.5) /
                  (2 * AD->local_values_as_span()[i]);
          if (w->local_values_as_span()[i] == 0)
            (*p).get_values()[i] = 0;
          else if ((p1->local_values_as_span()[i] > 0) == (b->local_values_as_span()[i] < 0))
            (*p).get_values()[i] = p1->local_values_as_span()[i];
          else
            (*p).get_values()[i] = p2->local_values_as_span()[i];
        }
      }

      // (2) initial residual f_0, |f_0|, DF_0
      std::shared_ptr<Core::LinAlg::Vector<double>> x =
          std::make_shared<Core::LinAlg::Vector<double>>(*activenodemap, true);
      std::shared_ptr<Core::LinAlg::Vector<double>> f =
          std::make_shared<Core::LinAlg::Vector<double>>(*activenodemap, true);
      int NumEntries = 0;
      int* Indices = nullptr;
      double* Values = nullptr;
      double res = 1.0;
      double initres = 1.0;
      double dfik = 0;
      Core::LinAlg::SparseMatrix DF(*activenodemap, 10);

      // rhs f
      for (int i = 0; i < activenodemap->num_my_elements(); ++i)
      {
        x->put_scalar(0.0);
        A->extract_global_row_view(activenodemap->gid(i), NumEntries, Values, Indices);
        x->replace_global_values(NumEntries, Values, Indices);
        (*f).get_values()[i] = b->local_values_as_span()[i] * p->local_values_as_span()[i] +
                               w->local_values_as_span()[i];
        for (int j = 0; j < activenodemap->num_my_elements(); ++j)
        {
          (*f).get_values()[i] += x->local_values_as_span()[j] * p->local_values_as_span()[i] *
                                  p->local_values_as_span()[j];
        }
      }

      // residual res
      f->norm_2(&initres);
      res = initres;

      // jacobian DF
      for (int i = 0; i < activenodemap->num_my_elements(); ++i)
      {
        x->put_scalar(0.0);
        A->extract_global_row_view(activenodemap->gid(i), NumEntries, Values, Indices);
        x->replace_global_values(NumEntries, Values, Indices);
        for (int k = 0; k < activenodemap->num_my_elements(); ++k)
        {
          if (k == i)
          {
            dfik = x->local_values_as_span()[i] * p->local_values_as_span()[i] +
                   b->local_values_as_span()[i];
            for (int j = 0; j < activenodemap->num_my_elements(); ++j)
            {
              dfik += x->local_values_as_span()[j] * p->local_values_as_span()[j];
            }
            DF.assemble(dfik, activenodemap->gid(i), activenodemap->gid(k));
          }
          else
          {
            DF.assemble(x->local_values_as_span()[k] * p->local_values_as_span()[i],
                activenodemap->gid(i), activenodemap->gid(k));
          }
        }
      }
      DF.complete(*activenodemap, *activenodemap);

      // (3) Newton-Iteration
      std::shared_ptr<Core::LinAlg::Vector<double>> mf =
          std::make_shared<Core::LinAlg::Vector<double>>(*activenodemap, true);
      std::shared_ptr<Core::LinAlg::Vector<double>> dp =
          std::make_shared<Core::LinAlg::Vector<double>>(*activenodemap, true);
      double tol = 0.00000001;
      double numiter = 0;
      double stopcrit = 100;

      while (res > tol)
      {
        // solver for linear equations DF * dp = -f
        mf->update(-1.0, *f, 0.0);
        Core::LinAlg::SolverParams solver_params;
        solver_params.refactor = true;
        solver_->solve(Core::Utils::shared_ptr_from_ref(DF), dp, mf, solver_params);

        // Update solution p_n = p_n-1 + dp
        p->update(1.0, *dp, 1.0);

        // rhs f
        for (int i = 0; i < activenodemap->num_my_elements(); ++i)
        {
          x->put_scalar(0.0);
          A->extract_global_row_view(activenodemap->gid(i), NumEntries, Values, Indices);
          x->replace_global_values(NumEntries, Values, Indices);
          (*f).get_values()[i] = (*b).get_values()[i] * (*p).get_values()[i] + (*w).get_values()[i];
          for (int j = 0; j < activenodemap->num_my_elements(); ++j)
          {
            (*f).get_values()[i] +=
                (*x).get_values()[j] * (*p).get_values()[i] * (*p).get_values()[j];
          }
        }

        // residual res
        f->norm_2(&res);
        res /= initres;

        // jacobian DF
        DF.put_scalar(0.0);
        for (int i = 0; i < activenodemap->num_my_elements(); ++i)
        {
          x->put_scalar(0.0);
          A->extract_global_row_view(activenodemap->gid(i), NumEntries, Values, Indices);
          x->replace_global_values(NumEntries, Values, Indices);
          for (int k = 0; k < activenodemap->num_my_elements(); ++k)
          {
            if (k == i)
            {
              dfik = x->local_values_as_span()[i] * p->local_values_as_span()[i] +
                     b->local_values_as_span()[i];
              for (int j = 0; j < activenodemap->num_my_elements(); ++j)
              {
                dfik += x->local_values_as_span()[j] * p->local_values_as_span()[j];
              }
              DF.assemble(dfik, activenodemap->gid(i), activenodemap->gid(k));
            }
            else
            {
              DF.assemble(x->local_values_as_span()[k] * p->local_values_as_span()[i],
                  activenodemap->gid(i), activenodemap->gid(k));
            }
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
      std::shared_ptr<Core::LinAlg::Vector<double>> ptemp1 =
          std::make_shared<Core::LinAlg::Vector<double>>(*slavenodemap, true);
      std::shared_ptr<Core::LinAlg::Vector<double>> ptemp2 =
          std::make_shared<Core::LinAlg::Vector<double>>(*dofmap, true);
      std::shared_ptr<Core::LinAlg::Vector<double>> VU =
          std::make_shared<Core::LinAlg::Vector<double>>(*dofmap, true);
      Core::LinAlg::export_to(*p, *ptemp1);
      BN->multiply(false, *ptemp1, *ptemp2);
      Minv.multiply(false, *ptemp2, *VU);
      veln_->update(1.0, *VU, 1.0);
    }
  }
}

/*----------------------------------------------------------------------*/
/* Reset configuration after time step */
void Solid::TimInt::reset_step()
{
  // reset state vectors
  disn_->update(1.0, (*dis_)[0], 0.0);
  veln_->update(1.0, (*vel_)[0], 0.0);
  accn_->update(1.0, (*acc_)[0], 0.0);

  // reset anything that needs to be reset at the element level
  {
    // create the parameters for the discretization
    Teuchos::ParameterList p;
    p.set("action", "calc_struct_reset_istep");
    // go to elements
    discret_->evaluate(p, nullptr, nullptr, nullptr, nullptr, nullptr);
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
      *discret_, Global::Problem::instance()->input_control_file(), step);
  if (step != reader.read_int("step")) FOUR_C_THROW("Time step on file not equal to given step");

  step_ = step;
  stepn_ = step_ + 1;
  time_ = std::make_shared<TimeStepping::TimIntMStep<double>>(0, 0, reader.read_double("time"));
  timen_ = (*time_)[0] + (*dt_)[0];

  read_restart_state();

  read_restart_constraint();
  read_restart_cardiovascular0_d();
  read_restart_contact_meshtying();
  read_restart_spring_dashpot();

  read_restart_force();
}

/*----------------------------------------------------------------------*/
/* Set restart values passed down from above */
void Solid::TimInt::set_restart(int step, double time,
    std::shared_ptr<Core::LinAlg::Vector<double>> disn,
    std::shared_ptr<Core::LinAlg::Vector<double>> veln,
    std::shared_ptr<Core::LinAlg::Vector<double>> accn,
    std::shared_ptr<std::vector<char>> elementdata, std::shared_ptr<std::vector<char>> nodedata)
{
  step_ = step;
  stepn_ = step_ + 1;
  time_ = std::make_shared<TimeStepping::TimIntMStep<double>>(0, 0, time);
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

  // biofilm growth
  if (have_biofilm_growth()) FOUR_C_THROW("Set restart not implemented for biofilm growth");
}

/*----------------------------------------------------------------------*/
/* Read and set restart state */
void Solid::TimInt::read_restart_state()
{
  Core::IO::DiscretizationReader reader(
      *discret_, Global::Problem::instance()->input_control_file(), step_);

  reader.read_vector(disn_, "displacement");
  dis_->update_steps(*disn_);

  reader.read_vector(veln_, "velocity");
  vel_->update_steps(*veln_);
  reader.read_vector(accn_, "acceleration");
  acc_->update_steps(*accn_);
  reader.read_history_data(step_);
}

/*----------------------------------------------------------------------*/
/* Read and set restart state */
void Solid::TimInt::set_restart_state(std::shared_ptr<Core::LinAlg::Vector<double>> disn,
    std::shared_ptr<Core::LinAlg::Vector<double>> veln,
    std::shared_ptr<Core::LinAlg::Vector<double>> accn,
    std::shared_ptr<std::vector<char>> elementdata, std::shared_ptr<std::vector<char>> nodedata

)
{
  dis_->update_steps(*disn);
  vel_->update_steps(*veln);
  acc_->update_steps(*accn);

  // the following is copied from read_mesh()
  // before we unpack nodes/elements we store a copy of the nodal row/col map
  Core::LinAlg::Map noderowmap(*discret_->node_row_map());
  Core::LinAlg::Map nodecolmap(*discret_->node_col_map());

  // unpack nodes and elements
  // so everything should be OK
  discret_->unpack_my_nodes(*nodedata);
  discret_->unpack_my_elements(*elementdata);
  discret_->redistribute({noderowmap, nodecolmap});
}
/*----------------------------------------------------------------------*/
/* Read and set restart values for constraints */
void Solid::TimInt::read_restart_constraint()
{
  if (conman_->have_constraint())
  {
    Core::IO::DiscretizationReader reader(
        *discret_, Global::Problem::instance()->input_control_file(), step_);
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
        *discret_, Global::Problem::instance()->input_control_file(), step_);
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
        *discret_, Global::Problem::instance()->input_control_file(), step_);
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
      *discret_, Global::Problem::instance()->input_control_file(), step_);

  if (have_contact_meshtying()) cmtbridge_->read_restart(reader, (*dis_)(0), zeros_);
}

/*----------------------------------------------------------------------*/
/* Calculate all output quantities that depend on a potential material history */
void Solid::TimInt::prepare_output(bool force_prepare_timestep)
{
  determine_stress_strain();
  determine_energy();
}


/*----------------------------------------------------------------------*/
/* output to file
 * originally by mwgee 03/07 */
void Solid::TimInt::output_step(const bool forced_writerestart)
{
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
  }

  // output results (not necessary if restart in same step)
  if (writestate_ and writeresultsevery_ and (step_ % writeresultsevery_ == 0) and
      (not datawritten))
  {
    output_state(datawritten);
  }

  // output stress & strain
  if (writeresultsevery_ and
      ((writestress_ != Inpar::Solid::stress_none) or (writestrain_ != Inpar::Solid::strain_none) or
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

  // output active set, energies and momentum for contact
  output_contact();
}

/*-----------------------------------------------------------------------------*
 * write GMSH output of displacement field
 *-----------------------------------------------------------------------------*/
void Solid::TimInt::write_gmsh_struct_output_step()
{
  if (not gmsh_out_) return;

  const std::string filename = Core::IO::Gmsh::get_file_name(
      "struct", discret_->writer()->output().file_name(), stepn_, false, myrank_);
  std::ofstream gmshfilecontent(filename.c_str());

  // add 'View' to Gmsh postprocessing file
  gmshfilecontent << "View \" "
                  << "struct displacement \" {" << std::endl;
  // draw vector field 'struct displacement' for every element
  Core::IO::Gmsh::vector_field_dof_based_to_gmsh(*discret_, dispn(), gmshfilecontent, 0, true);
  gmshfilecontent << "};" << std::endl;
}

/*----------------------------------------------------------------------*/
/* We need the restart data to perform on "restarts" on the fly for parameter
 * continuation
 */
void Solid::TimInt::get_restart_data(std::shared_ptr<int> step, std::shared_ptr<double> time,
    std::shared_ptr<Core::LinAlg::Vector<double>> disn,
    std::shared_ptr<Core::LinAlg::Vector<double>> veln,
    std::shared_ptr<Core::LinAlg::Vector<double>> accn,
    std::shared_ptr<std::vector<char>> elementdata, std::shared_ptr<std::vector<char>> nodedata)
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
    cmtbridge_->write_restart(*output_);
    cmtbridge_->postprocess_quantities(*output_);

    {
      std::shared_ptr<Teuchos::ParameterList> cmtOutputParams =
          std::make_shared<Teuchos::ParameterList>();
      cmtOutputParams->set<int>("step", step_);
      cmtOutputParams->set<double>("time", (*time_)[0]);
      cmtOutputParams->set<std::shared_ptr<const Core::LinAlg::Vector<double>>>(
          "displacement", (*dis_)(0));
      cmtbridge_->postprocess_quantities_per_interface(cmtOutputParams);
    }
  }

  // biofilm growth
  if (have_biofilm_growth())
  {
    output_->write_vector("str_growth_displ", strgrdisp_);
  }

  // springdashpot output
  if (springman_->have_spring_dashpot()) springman_->output_restart(output_, *discret_, *disn_);

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
  {
    output_->new_step(step_, (*time_)[0]);
    output_->write_vector("displacement", (*dis_)(0));
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
    cmtbridge_->postprocess_quantities(*output_);

    {
      std::shared_ptr<Teuchos::ParameterList> cmtOutputParams =
          std::make_shared<Teuchos::ParameterList>();
      cmtOutputParams->set<int>("step", step_);
      cmtOutputParams->set<double>("time", (*time_)[0]);
      cmtOutputParams->set<std::shared_ptr<const Core::LinAlg::Vector<double>>>(
          "displacement", (*dis_)(0));
      cmtbridge_->postprocess_quantities_per_interface(cmtOutputParams);
    }
  }

  if (porositysplitter_ != nullptr)
  {
    std::shared_ptr<Core::LinAlg::Vector<double>> porosity =
        porositysplitter_->extract_cond_vector(*(*dis_)(0));
    output_->write_vector("porosity_p1", porosity);
  }

  // springdashpot output
  if (springman_->have_spring_dashpot()) springman_->output(*output_, *discret_, *disn_);
}

/*----------------------------------------------------------------------*/
/* add restart information to output_state */
void Solid::TimInt::add_restart_to_output_state()
{
  // add velocity and acceleration
  output_->write_vector("velocity", (*vel_)(0));
  output_->write_vector("acceleration", (*acc_)(0));

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
  if (springman_->have_spring_dashpot()) springman_->output_restart(output_, *discret_, *disn_);

  // contact/meshtying
  if (have_contact_meshtying()) cmtbridge_->write_restart(*output_, true);

  // TODO: add missing restart data for surface stress and contact/meshtying here

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
      ((writestress_ != Inpar::Solid::stress_none) or (writestrain_ != Inpar::Solid::strain_none) or
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

    stressdata_ = std::make_shared<std::vector<char>>();
    p.set("stress", stressdata_);

    p.set<Inpar::Solid::StressType>("iostress", writestress_);

    straindata_ = std::make_shared<std::vector<char>>();
    p.set("strain", straindata_);
    p.set<Inpar::Solid::StrainType>("iostrain", writestrain_);

    // plastic strain
    plstraindata_ = std::make_shared<std::vector<char>>();
    p.set("plstrain", plstraindata_);
    p.set<Inpar::Solid::StrainType>("ioplstrain", writeplstrain_);

    // rotation tensor
    rotdata_ = std::make_shared<std::vector<char>>();
    p.set("rotation", rotdata_);

    // set vector values needed by elements
    discret_->clear_state();
    // extended set_state(0,...) in case of multiple dofsets (e.g. TSI)
    discret_->set_state(0, "residual displacement", *zeros_);
    discret_->set_state(0, "displacement", *disn_);

    std::shared_ptr<Core::LinAlg::SparseOperator> system_matrix = nullptr;
    std::shared_ptr<Core::LinAlg::Vector<double>> system_vector = nullptr;
    Core::FE::evaluate(*discret_, p, system_matrix, system_vector, discret_->element_row_map());
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
      discret_->set_state("displacement", *disn_);
      // get energies
      Core::LinAlg::SerialDenseVector energies(1);
      discret_->evaluate_scalars(p, energies);
      discret_->clear_state();
      intergy_ = (energies)(0);
    }

    // global calculation of kinetic energy
    kinergy_ = 0.0;  // total kinetic energy
    {
      std::shared_ptr<Core::LinAlg::Vector<double>> linmom =
          std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map_view(), true);
      mass_->multiply(false, *veln_, *linmom);
      linmom->dot(*veln_, &kinergy_);
      kinergy_ *= 0.5;
    }

    // external energy
    extergy_ = 0.0;  // total external energy
    {
      // WARNING: This will only work with dead loads and implicit time
      // integration (otherwise there is no fextn_)!!!
      std::shared_ptr<Core::LinAlg::Vector<double>> fext = fext_new();
      fext->dot(*disn_, &extergy_);
    }
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
    stressdata_ = nullptr;
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
    straindata_ = nullptr;
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
    plstraindata_ = nullptr;
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
/* output active set, energies and momentum for contact */
void Solid::TimInt::output_contact()
{
  // only for contact / meshtying simulations
  if (have_contact_meshtying())
  {
    // print active set
    cmtbridge_->get_strategy().print_active_set();
  }
}

/*----------------------------------------------------------------------*/
/* evaluate external forces at t_{n+1} */
void Solid::TimInt::apply_force_external(const double time,
    const std::shared_ptr<Core::LinAlg::Vector<double>> dis,
    const std::shared_ptr<Core::LinAlg::Vector<double>> disn, Core::LinAlg::Vector<double>& vel,
    Core::LinAlg::Vector<double>& fext)
{
  Teuchos::ParameterList p;
  // other parameters needed by the elements
  p.set("total time", time);
  p.set<const Core::Utils::FunctionManager*>(
      "function_manager", &Global::Problem::instance()->function_manager());

  // set vector values needed by elements
  discret_->clear_state();
  discret_->set_state(0, "displacement", *dis);
  discret_->set_state(0, "displacement new", *disn);

  if (damping_ == Inpar::Solid::damp_material) discret_->set_state(0, "velocity", vel);

  discret_->evaluate_neumann(p, fext);
}

/*----------------------------------------------------------------------*/
/* check whether we have nonlinear inertia forces or not */
Inpar::Solid::MassLin Solid::TimInt::have_nonlinear_mass() const
{
  const Teuchos::ParameterList& sdyn = Global::Problem::instance()->structural_dynamic_params();
  auto masslin = Teuchos::getIntegralValue<Inpar::Solid::MassLin>(sdyn, "MASSLIN");

  return masslin;
}

/*----------------------------------------------------------------------*/
/* check whether the initial conditions are fulfilled */
void Solid::TimInt::nonlinear_mass_sanity_check(
    std::shared_ptr<const Core::LinAlg::Vector<double>> fext,
    std::shared_ptr<const Core::LinAlg::Vector<double>> dis,
    std::shared_ptr<const Core::LinAlg::Vector<double>> vel,
    std::shared_ptr<const Core::LinAlg::Vector<double>> acc,
    const Teuchos::ParameterList* sdynparams) const
{
  double fextnorm;
  fext->norm_2(&fextnorm);

  double dispnorm;
  dis->norm_2(&dispnorm);

  double velnorm;
  vel->norm_2(&velnorm);

  double accnorm;
  acc->norm_2(&accnorm);

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
        "norm disp = {} \n"
        "norm vel  = {} \n"
        "norm acc  = {}",
        dispnorm, velnorm, accnorm);
  }

  if (have_nonlinear_mass() == Inpar::Solid::MassLin::ml_rotations and
      Teuchos::getIntegralValue<Inpar::Solid::PredEnum>(*sdynparams, "PREDICT") !=
          Inpar::Solid::pred_constdis)
  {
    FOUR_C_THROW(
        "Only constant displacement consistent velocity and acceleration "
        "predictor possible for multiplicative Genalpha time integration!");
  }

  if (sdynparams != nullptr)
  {
    if (have_nonlinear_mass() == Inpar::Solid::MassLin::ml_rotations and
        Teuchos::getIntegralValue<Inpar::Solid::DynamicType>(*sdynparams, "DYNAMICTYPE") !=
            Inpar::Solid::DynamicType::GenAlpha)
    {
      FOUR_C_THROW(
          "Nonlinear inertia forces for rotational DoFs only implemented "
          "for GenAlpha time integration so far!");
    }
  }
}

/*----------------------------------------------------------------------*/
/* evaluate ordinary internal force */
void Solid::TimInt::apply_force_internal(const double time, const double dt,
    std::shared_ptr<const Core::LinAlg::Vector<double>> dis,
    std::shared_ptr<const Core::LinAlg::Vector<double>> disi,
    const Core::LinAlg::Vector<double>& vel, std::shared_ptr<Core::LinAlg::Vector<double>> fint)
{
  // create the parameters for the discretization
  Teuchos::ParameterList p;
  // action for elements
  std::string action = "calc_struct_internalforce";

  p.set("action", action);
  // other parameters that might be needed by the elements
  p.set("total time", time);
  p.set("delta time", dt);

  if (pressure_ != nullptr) p.set("volume", 0.0);
  // set vector values needed by elements
  discret_->clear_state();
  discret_->set_state("residual displacement", *disi);  // these are incremental
  discret_->set_state("displacement", *dis);

  if (damping_ == Inpar::Solid::damp_material) discret_->set_state("velocity", vel);
  // fintn_->PutScalar(0.0);  // initialise internal force vector
  discret_->evaluate(p, nullptr, nullptr, fint, nullptr, nullptr);

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
      randnum = Core::Communication::sum_all(proc_randnum, discret_->get_comm());
      const double numproc = Core::Communication::num_mpi_ranks(discret_->get_comm());
      randnum /= numproc;
      if (rand_tsfac_ > 1.0)
        rand_tsfac_ = randnum * 0.49 + 0.51;
      else if (rand_tsfac_ < 1.0)
        rand_tsfac_ = randnum * 0.99 + 1.0;
      else
        rand_tsfac_ = randnum * 1.48 + 0.51;
      if (myrank_ == 0)
      {
        Core::IO::cout << "Nonlinear solver failed to converge: modifying time-step size by random "
                          "number between 0.51 and 1.99 -> here: "
                       << rand_tsfac_ << " !" << Core::IO::endl;
      }
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
      {
        Core::IO::cout << "Nonlinear solver failed to converge and DIVERCONT = "
                          "repeat_simulation, hence leaving structural time integration "
                       << Core::IO::endl;
      }
      else if (nonlinsoldiv == Inpar::Solid::conv_lin_fail)
      {
        Core::IO::cout << "Linear solver failed to converge and DIVERCONT = "
                          "repeat_simulation, hence leaving structural time integration "
                       << Core::IO::endl;
      }
      else if (nonlinsoldiv == Inpar::Solid::conv_ele_fail)
      {
        Core::IO::cout
            << "Element failure in form of a negative Jacobian determinant and DIVERCONT = "
               "repeat_simulation, hence leaving structural time integration "
            << Core::IO::endl;
      }
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
      {
        FOUR_C_THROW(
            "Maximal divercont refinement level reached. Finally nonlinear solver did not "
            "converge. :-(");
      }

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
    const Core::LinAlg::MultiVector<double>& iforce  ///< the force on interface
)
{
  fifc_->update(1.0, iforce, 0.0);
}

std::string Solid::TimInt::method_title() const
{
  return std::string(EnumTools::enum_name(method_name()));
}

/*----------------------------------------------------------------------*/
/* Attach file handle for energy file #energyfile_                      */
void Solid::TimInt::attach_energy_file()
{
  if (!energyfile_)
  {
    std::string energyname =
        Global::Problem::instance()->output_control_file()->file_name() + ".energy";
    energyfile_ = std::make_shared<std::ofstream>(energyname.c_str());
    (*energyfile_) << "# timestep time total_energy"
                   << " kinetic_energy internal_energy external_energy" << std::endl;
  }
}

/*----------------------------------------------------------------------*/
/* Return (rotatory) transformation matrix of local co-ordinate systems */
std::shared_ptr<const Core::LinAlg::SparseMatrix> Solid::TimInt::get_loc_sys_trafo() const
{
  if (locsysman_ != nullptr) return locsysman_->trafo();

  return nullptr;
}

/*----------------------------------------------------------------------*/
/* Return stiffness matrix as Core::LinAlg::SparseMatrix                      */
std::shared_ptr<Core::LinAlg::SparseMatrix> Solid::TimInt::system_matrix()
{
  return std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(stiff_);
}

/*----------------------------------------------------------------------*/
/* Return stiffness matrix as Core::LinAlg::BlockSparseMatrix */
std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> Solid::TimInt::block_system_matrix()
{
  return std::dynamic_pointer_cast<Core::LinAlg::BlockSparseMatrixBase>(stiff_);
}

/*----------------------------------------------------------------------*/
/* Return sparse mass matrix                                            */
std::shared_ptr<Core::LinAlg::SparseMatrix> Solid::TimInt::mass_matrix()
{
  return std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(mass_);
}


/*----------------------------------------------------------------------*/
/* Return domain map of mass matrix                                     */
const Core::LinAlg::Map& Solid::TimInt::domain_map() const { return mass_->domain_map(); }

/*----------------------------------------------------------------------*/
/* Creates the field test                                               */
std::shared_ptr<Core::Utils::ResultTest> Solid::TimInt::create_field_test()
{
  return std::make_shared<StruResultTest>(*this);
}

/*----------------------------------------------------------------------*/
/* dof map of vector of unknowns                                        */
std::shared_ptr<const Core::LinAlg::Map> Solid::TimInt::dof_row_map()
{
  const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map();
  return std::make_shared<Core::LinAlg::Map>(*dofrowmap);
}

/*----------------------------------------------------------------------*/
/* dof map of vector of unknowns                                        */
/* new method for multiple dofsets                                      */
std::shared_ptr<const Core::LinAlg::Map> Solid::TimInt::dof_row_map(unsigned nds)
{
  const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map(nds);
  return std::make_shared<Core::LinAlg::Map>(*dofrowmap);
}

/*----------------------------------------------------------------------*/
/* view of dof map of vector of unknowns                                */
const Core::LinAlg::Map* Solid::TimInt::dof_row_map_view() { return discret_->dof_row_map(); }

/*----------------------------------------------------------------------*/
/* reset everything (needed for biofilm simulations)                    */
void Solid::TimInt::reset()
{
  // displacements D_{n}
  dis_ = std::make_shared<TimeStepping::TimIntMStep<Core::LinAlg::Vector<double>>>(
      0, 0, dof_row_map_view(), true);
  // velocities V_{n}
  vel_ = std::make_shared<TimeStepping::TimIntMStep<Core::LinAlg::Vector<double>>>(
      0, 0, dof_row_map_view(), true);
  // accelerations A_{n}
  acc_ = std::make_shared<TimeStepping::TimIntMStep<Core::LinAlg::Vector<double>>>(
      0, 0, dof_row_map_view(), true);

  // displacements D_{n+1} at t_{n+1}
  disn_ = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map_view(), true);
  // velocities V_{n+1} at t_{n+1}
  veln_ = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map_view(), true);
  // accelerations A_{n+1} at t_{n+1}
  accn_ = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map_view(), true);
  // create empty interface force vector
  fifc_ = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map_view(), true);

  // set initial fields
  set_initial_fields();
}

/*----------------------------------------------------------------------*/
/* set structure displacement vector due to biofilm growth              */
void Solid::TimInt::set_str_gr_disp(
    std::shared_ptr<Core::LinAlg::Vector<double>> struct_growth_disp)
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
/* Expand the dbc map by dofs provided in Core::LinAlg::Map maptoadd.          */
void Solid::TimInt::add_dirich_dofs(const std::shared_ptr<const Core::LinAlg::Map> maptoadd)
{
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> condmaps;
  condmaps.push_back(maptoadd);
  condmaps.push_back(get_dbc_map_extractor()->cond_map());
  std::shared_ptr<Core::LinAlg::Map> condmerged =
      Core::LinAlg::MultiMapExtractor::merge_maps(condmaps);
  *dbcmaps_ = Core::LinAlg::MapExtractor(*(discret_->dof_row_map()), condmerged);
}

/*----------------------------------------------------------------------*/
/* Contract the dbc map by dofs provided in Core::LinAlg::Map maptoremove.     */
void Solid::TimInt::remove_dirich_dofs(const std::shared_ptr<const Core::LinAlg::Map> maptoremove)
{
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> othermaps;
  othermaps.push_back(maptoremove);
  othermaps.push_back(get_dbc_map_extractor()->other_map());
  std::shared_ptr<Core::LinAlg::Map> othermerged =
      Core::LinAlg::MultiMapExtractor::merge_maps(othermaps);
  *dbcmaps_ = Core::LinAlg::MapExtractor(*(discret_->dof_row_map()), othermerged, false);
}

FOUR_C_NAMESPACE_CLOSE

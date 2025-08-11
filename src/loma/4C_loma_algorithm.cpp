// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_loma_algorithm.hpp"

#include "4C_fem_general_assemblestrategy.hpp"
#include "4C_fluid_ele_action.hpp"
#include "4C_global_data.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_blocksparsematrix.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linear_solver_method.hpp"
#include "4C_linear_solver_method_input.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_linear_solver_method_parameters.hpp"
#include "4C_scatra_timint_loma.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
LowMach::Algorithm::Algorithm(
    MPI_Comm comm, const Teuchos::ParameterList& prbdyn, const Teuchos::ParameterList& solverparams)
    : ScaTraFluidCouplingAlgorithm(comm, prbdyn, false, "scatra", solverparams),
      monolithic_(false),
      lomadbcmap_(nullptr),
      lomaincrement_(nullptr),
      lomarhs_(nullptr),
      zeros_(nullptr),
      lomasystemmatrix_(nullptr),
      lomasolver_(nullptr),
      dt_(0.0),
      maxtime_(0.0),
      stepmax_(0),
      itmax_(0),
      itmaxpre_(0),
      itmaxbs_(0),
      ittol_(1.0),
      samstart_(-1),
      turbinflow_(false),
      numinflowsteps_(-1),
      probdyn_(prbdyn)
{
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void LowMach::Algorithm::init()
{
  // call init() in base class
  Adapter::ScaTraFluidCouplingAlgorithm::init();

  // flag for monolithic solver
  monolithic_ = (probdyn_.get<bool>("MONOLITHIC"));

  // time-step length, maximum time and maximum number of steps
  dt_ = probdyn_.get<double>("TIMESTEP");
  maxtime_ = probdyn_.get<double>("MAXTIME");
  stepmax_ = probdyn_.get<int>("NUMSTEP");

  // (preliminary) maximum number of iterations and tolerance for outer iteration
  ittol_ = probdyn_.get<double>("CONVTOL");
  itmaxpre_ = probdyn_.get<int>("ITEMAX");
  // maximum number of iterations before sampling (turbulent flow only)
  itmaxbs_ = probdyn_.get<int>("ITEMAX_BEFORE_SAMPLING");

  // flag for constant thermodynamic pressure
  consthermpress_ = probdyn_.get<std::string>("CONSTHERMPRESS");

  // flag for special flow and start of sampling period from fluid parameter list
  const Teuchos::ParameterList& fluiddyn = Global::Problem::instance()->fluid_dynamic_params();
  special_flow_ = fluiddyn.sublist("TURBULENCE MODEL").get<std::string>("CANONICAL_FLOW");
  samstart_ = fluiddyn.sublist("TURBULENCE MODEL").get<int>("SAMPLING_START");

  // check scatra solver type, which should be incremental, for the time being
  if (not scatra_field()->is_incremental())
    FOUR_C_THROW("Incremental ScaTra formulation required for low-Mach-number flow");

  // flag for turbulent inflow
  turbinflow_ = fluiddyn.sublist("TURBULENT INFLOW").get<bool>("TURBULENTINFLOW");
  // number of inflow steps
  numinflowsteps_ = fluiddyn.sublist("TURBULENT INFLOW").get<int>("NUMINFLOWSTEP");
  if (turbinflow_)
  {
    if (Core::Communication::my_mpi_rank(get_comm()) == 0)
    {
      std::cout << "##############################################################" << '\n';
      std::cout << "#                     TURBULENT INFLOW                       #" << '\n';
      std::cout << "# Caution!                                                   #" << '\n';
      std::cout << "# Assumptions: - constant thermodynamic pressure in main     #" << '\n';
      std::cout << "#                problem domain                              #" << '\n';
      std::cout << "#              - inflow domain is closed system without in-/ #" << '\n';
      std::cout << "#                outflow and heating                         #" << '\n';
      std::cout << "#                -> constant thermodynamic pressure          #" << '\n';
      std::cout << "##############################################################" << '\n';
    }

    if (special_flow_ != "loma_backward_facing_step")
      FOUR_C_THROW("Turbulent inflow generation only for backward-facing step!");
    if (consthermpress_ != "Yes")
      FOUR_C_THROW("Constant thermodynamic pressure in main problem domain!");
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void LowMach::Algorithm::setup()
{
  // call setup() in base class
  Adapter::ScaTraFluidCouplingAlgorithm::setup();

  const Teuchos::ParameterList& fluiddyn = Global::Problem::instance()->fluid_dynamic_params();

  // preparatives for monolithic solver
  if (monolithic_)
  {
    // check whether turbulent inflow is included,
    // which is currently not possible for monolithic solver
    if (turbinflow_) FOUR_C_THROW("No turbulent inflow for monolithic low-Mach-number solver");

    // check whether (fluid) linearization scheme is a fixed-point-like scheme,
    // which is the only one enabled for monolithic solver, for the time being
    auto linearization = fluiddyn.get<Inpar::FLUID::LinearisationAction>("NONLINITER");
    if (linearization != Inpar::FLUID::fixed_point_like)
      FOUR_C_THROW(
          "Only a fixed-point-like iteration scheme is enabled for monolithic low-Mach-number "
          "solver, for the time being!");

    // generate proxy of scatra dof set to be used by fluid field
    std::shared_ptr<Core::DOFSets::DofSetInterface> scatradofset =
        scatra_field()->discretization()->get_dof_set_proxy();

    // check number of dof sets in respective fields
    if (fluid_field()->discretization()->add_dof_set(scatradofset) != 1)
      FOUR_C_THROW("Incorrect number of dof sets in fluid field!");

    // create combined map for loma problem
    std::vector<std::shared_ptr<const Core::LinAlg::Map>> dofrowmaps;

    // insert actual (zeroth) map of the discretization: first fluid, then scatra
    {
      dofrowmaps.push_back(fluid_field()->dof_row_map(0));
      const Core::LinAlg::Map* dofrowmapscatra = (scatra_field()->discretization())->dof_row_map(0);
      dofrowmaps.push_back(Core::Utils::shared_ptr_from_ref(*dofrowmapscatra));
    }

    // check existence of elements
    if (dofrowmaps[0]->num_global_elements() == 0) FOUR_C_THROW("No fluid elements!");
    if (dofrowmaps[1]->num_global_elements() == 0) FOUR_C_THROW("No scatra elements!");

    std::shared_ptr<Core::LinAlg::Map> fullmap =
        Core::LinAlg::MultiMapExtractor::merge_maps(dofrowmaps);

    // full loma block dofrowmap
    lomablockdofrowmap_.setup(*fullmap, dofrowmaps);

    // get solver number used for LOMA solver
    const int linsolvernumber = probdyn_.get<int>("LINEAR_SOLVER");
    // check if LOMA solvers has a valid number
    if (linsolvernumber == (-1))
      FOUR_C_THROW(
          "no linear solver defined for LOMA. Please set LINEAR_SOLVER in LOMA CONTROL to a valid "
          "number! This solver has to be an iterative solver with BGS2x2 block preconditioner.");

    // create loma solver
    // get solver parameter list of linear LOMA solver
    const Teuchos::ParameterList& lomasolverparams =
        Global::Problem::instance()->solver_params(linsolvernumber);

    const auto solvertype =
        Teuchos::getIntegralValue<Core::LinearSolver::SolverType>(lomasolverparams, "SOLVER");

    if (solvertype != Core::LinearSolver::SolverType::belos)
      FOUR_C_THROW(
          "SOLVER {} is not valid for LOMA. It has to be an iterative Solver (with BGS2x2 block "
          "preconditioner)",
          linsolvernumber);

    const auto azprectype = Teuchos::getIntegralValue<Core::LinearSolver::PreconditionerType>(
        lomasolverparams, "AZPREC");
    if (azprectype != Core::LinearSolver::PreconditionerType::block_teko)
      FOUR_C_THROW(
          "SOLVER {} is not valid for LOMA. It has to be an iterative Solver with 2x2 block "
          "preconditioner",
          linsolvernumber);

    // use loma solver object
    lomasolver_ = std::make_shared<Core::LinAlg::Solver>(lomasolverparams,
        fluid_field()->discretization()->get_comm(),
        Global::Problem::instance()->solver_params_callback(),
        Global::Problem::instance()->io_params().get<Core::IO::Verbositylevel>("VERBOSITY"));

    // todo extract ScalarTransportFluidSolver
    const int fluidsolver = fluiddyn.get<int>("LINEAR_SOLVER");
    if (fluidsolver == (-1))
      FOUR_C_THROW(
          "no linear solver defined for fluid LOMA (inflow) problem. Please set LINEAR_SOLVER in "
          "FLUID DYNAMIC to a valid number! This solver block is used for the primary variables "
          "(Inverse1 block) within BGS2x2 preconditioner.");

    lomasolver_->put_solver_params_to_sub_params("Inverse1",
        Global::Problem::instance()->solver_params(fluidsolver),
        Global::Problem::instance()->solver_params_callback(),
        Global::Problem::instance()->io_params().get<Core::IO::Verbositylevel>("VERBOSITY"));


    // get linear solver id from SCALAR TRANSPORT DYNAMIC
    const Teuchos::ParameterList& scatradyn =
        Global::Problem::instance()->scalar_transport_dynamic_params();
    const int scalartransportsolvernumber = scatradyn.get<int>("LINEAR_SOLVER");
    if (scalartransportsolvernumber == (-1))
      FOUR_C_THROW(
          "no linear solver defined for LOMA problem. Please set LINEAR_SOLVER in SCALAR TRANSPORT "
          "DYNAMIC to a valid number! This solver block is used for the secondary variables "
          "(Inverse2 block) within BGS2x2 preconditioner.");

    lomasolver_->put_solver_params_to_sub_params("Inverse2",
        Global::Problem::instance()->solver_params(scalartransportsolvernumber),
        Global::Problem::instance()->solver_params_callback(),
        Global::Problem::instance()->io_params().get<Core::IO::Verbositylevel>("VERBOSITY"));

    Core::LinearSolver::Parameters::compute_solver_parameters(
        *fluid_field()->discretization(), lomasolver_->params().sublist("Inverse1"));
    Core::LinearSolver::Parameters::compute_solver_parameters(
        *scatra_field()->discretization(), lomasolver_->params().sublist("Inverse2"));

    // create loma block matrix
    lomasystemmatrix_ =
        std::make_shared<Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(
            lomablockdofrowmap_, lomablockdofrowmap_, 135, false, true);

    // create loma rhs vector
    lomarhs_ =
        std::make_shared<Core::LinAlg::Vector<double>>(*lomablockdofrowmap_.full_map(), true);

    // create loma increment vector
    lomaincrement_ =
        std::make_shared<Core::LinAlg::Vector<double>>(*lomablockdofrowmap_.full_map(), true);

    // create vector of zeros for enforcing zero Dirichlet boundary conditions
    zeros_ = std::make_shared<Core::LinAlg::Vector<double>>(*lomablockdofrowmap_.full_map(), true);

    // create combined Dirichlet boundary condition map
    const std::shared_ptr<const Core::LinAlg::Map> fdbcmap =
        fluid_field()->get_dbc_map_extractor()->cond_map();
    const std::shared_ptr<const Core::LinAlg::Map> sdbcmap =
        scatra_field()->dirich_maps()->cond_map();
    lomadbcmap_ = Core::LinAlg::merge_map(fdbcmap, sdbcmap, false);
  }

  return;
}



/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void LowMach::Algorithm::time_loop()
{
  check_is_init();
  check_is_setup();

  // do initial calculations
  // if and only if it is the first time step
  // do not do initial calculations after restarts
  if (step() == 0 or (turbinflow_ and step() == numinflowsteps_))
    initial_calculations();
  else
    // set scalar field and thermodynamic pressure for evaluation of
    // Neumann boundary conditions in FLUID at beginning of first time step
    fluid_field()->set_scalar_fields(scatra_field()->phinp(),
        std::dynamic_pointer_cast<ScaTra::ScaTraTimIntLoma>(scatra_field())->therm_press_np(),
        nullptr, scatra_field()->discretization());

  // time loop
  while (not_finished())
  {
    increment_time_and_step();

    // prepare time step
    prepare_time_step();

    // do outer iteration loop for particular type of algorithm
    if (monolithic_)
      mono_loop();
    else
      outer_loop();

    // update for next time step
    time_update();

    // write output to screen and files
    output();

  }  // time loop

  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void LowMach::Algorithm::initial_calculations()
{
  // set initial velocity field for evaluation of initial scalar time derivative in SCATRA
  scatra_field()->set_convective_velocity(*fluid_field()->velnp());
  scatra_field()->set_velocity_field(*fluid_field()->velnp());
  if (scatra_field()->fine_scale_velocity_field_required())
  {
    scatra_field()->set_fine_scale_velocity(*fluid_field()->fs_vel());
  }

  // set initial value of thermodynamic pressure in SCATRA
  std::dynamic_pointer_cast<ScaTra::ScaTraTimIntLoma>(scatra_field())->set_initial_therm_pressure();

  // mass conservation: compute initial mass (initial time deriv. assumed zero)
  if (consthermpress_ == "No_mass")
    std::dynamic_pointer_cast<ScaTra::ScaTraTimIntLoma>(scatra_field())->compute_initial_mass();

  // set initial scalar field and thermodynamic pressure for evaluation of
  // Neumann boundary conditions in FLUID at beginning of first time step
  fluid_field()->set_scalar_fields(scatra_field()->phinp(),
      std::dynamic_pointer_cast<ScaTra::ScaTraTimIntLoma>(scatra_field())->therm_press_np(),
      nullptr, scatra_field()->discretization());
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void LowMach::Algorithm::prepare_time_step()
{
  check_is_init();
  check_is_setup();

  // prepare scalar transport time step
  // (+ computation of initial scalar time derivative in first time step)
  scatra_field()->prepare_time_step();

  // predict thermodynamic pressure and time derivative
  // (if not constant or based on mass conservation)
  if (consthermpress_ == "No_energy")
    std::dynamic_pointer_cast<ScaTra::ScaTraTimIntLoma>(scatra_field())->predict_therm_pressure();

  // prepare fluid time step, among other things, predict velocity field
  fluid_field()->prepare_time_step();

  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void LowMach::Algorithm::outer_loop()
{
  check_is_init();
  check_is_setup();

  int itnum = 0;
  bool stopnonliniter = false;

  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    std::cout << "\n****************************************\n          OUTER ITERATION "
                 "LOOP\n****************************************\n";

    printf("TIME: %11.4E/%11.4E  DT = %11.4E  %s  STEP = %4d/%4d\n", time(), maxtime_, dt_,
        scatra_field()->method_title().c_str(), step(), stepmax_);
  }

  //  // maximum number of iterations tolerance for outer iteration
  //  // currently default for turbulent channel flow: only one iteration before sampling
  //  if (special_flow_ == "loma_channel_flow_of_height_2" && Step() < samstart_ )
  //       itmax_ = 1;
  //  else itmax_ = itmaxpre_;

  // maximum number of iterations tolerance for outer iteration
  // reduced number of iterations for turbulent flow: only before sampling
  if (special_flow_ != "no" && step() < samstart_)
  {
    itmax_ = itmaxbs_;
    if (Core::Communication::my_mpi_rank(get_comm()) == 0 and
        (step() == 1 or (turbinflow_ and step() == (numinflowsteps_ + 1))))
    {
      std::cout << "\n+----------------------------------------------------------------------------"
                   "----------------+"
                << std::endl;
      std::cout << "Special turbulent variable-density flow: reduced number of iterations before "
                   "sampling: "
                << itmax_ << std::endl;
      std::cout << "+------------------------------------------------------------------------------"
                   "--------------+\n"
                << std::endl;
    }
  }
  else
  {
    itmax_ = itmaxpre_;
    if (Core::Communication::my_mpi_rank(get_comm()) == 0 and special_flow_ != "no" and
        step() == samstart_)
    {
      std::cout << "\n+----------------------------------------------------------------------------"
                   "----------------+"
                << std::endl;
      std::cout << "Special turbulent variable-density flow: maximum number of iterations allowed: "
                << itmax_ << std::endl;
      std::cout << "+------------------------------------------------------------------------------"
                   "--------------+\n"
                << std::endl;
    }
  }

  // evaluate fluid predictor step (currently not performed)
  // fluid_field()->Predictor();

  // set fluid values required in scatra
  set_fluid_values_in_scatra();

  // initially solve scalar transport equation
  // (values for intermediate time steps were calculated at the end of PerpareTimeStep)
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
    std::cout << "\n****************************************\n        SCALAR TRANSPORT "
                 "SOLVER\n****************************************\n";
  scatra_field()->solve();

  while (stopnonliniter == false)
  {
    itnum++;

    // in case of non-constant thermodynamic pressure: compute
    // (either based on energy conservation or based on mass conservation)
    if (consthermpress_ == "No_energy")
      std::dynamic_pointer_cast<ScaTra::ScaTraTimIntLoma>(scatra_field())->compute_therm_pressure();
    else if (consthermpress_ == "No_mass")
      std::dynamic_pointer_cast<ScaTra::ScaTraTimIntLoma>(scatra_field())
          ->compute_therm_pressure_from_mass_cons();

    // set scatra values required in fluid
    set_scatra_values_in_fluid();

    // solve low-Mach-number flow equations
    if (Core::Communication::my_mpi_rank(get_comm()) == 0)
      std::cout << "\n****************************************\n              FLUID "
                   "SOLVER\n****************************************\n";
    fluid_field()->solve();

    // set fluid values required in scatra
    set_fluid_values_in_scatra();

    // solve scalar transport equation
    if (Core::Communication::my_mpi_rank(get_comm()) == 0)
      std::cout << "\n****************************************\n        SCALAR TRANSPORT "
                   "SOLVER\n****************************************\n";
    scatra_field()->solve();

    // check convergence and stop iteration loop if convergence is achieved
    stopnonliniter = convergence_check(itnum);
  }

  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void LowMach::Algorithm::mono_loop()
{
  int itnum = 0;
  bool stopnonliniter = false;

  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    std::cout << "\n****************************************\n       MONOLITHIC ITERATION "
                 "LOOP\n****************************************\n";

    printf("TIME: %11.4E/%11.4E  DT = %11.4E  %s  STEP = %4d/%4d\n", time(), maxtime_, dt_,
        scatra_field()->method_title().c_str(), step(), stepmax_);
  }

  // maximum number of iterations tolerance for monolithic iteration
  // currently default for turbulent channel flow: only one iteration before sampling
  if (special_flow_ == "loma_channel_flow_of_height_2" && step() < samstart_)
    itmax_ = 1;
  else
    itmax_ = itmaxpre_;

  // evaluate fluid predictor step (currently not performed)
  // fluid_field()->Predictor();

  while (stopnonliniter == false)
  {
    itnum++;

    // set fluid values required in scatra
    set_fluid_values_in_scatra();

    // in case of non-constant thermodynamic pressure: compute
    // (either based on energy conservation or based on mass conservation)
    if (consthermpress_ == "No_energy")
      std::dynamic_pointer_cast<ScaTra::ScaTraTimIntLoma>(scatra_field())->compute_therm_pressure();
    else if (consthermpress_ == "No_mass")
      std::dynamic_pointer_cast<ScaTra::ScaTraTimIntLoma>(scatra_field())
          ->compute_therm_pressure_from_mass_cons();

    // set scatra values required in fluid
    set_scatra_values_in_fluid();

    // preparatives for scalar transport and fluid solver
    scatra_field()->prepare_linear_solve();
    fluid_field()->prepare_solve();

    // set up matrix and right-hand-side for monolithic low-Mach-number system
    setup_mono_loma_matrix();
    setup_mono_loma_rhs();

    // solve monolithic low-Mach-number system
    mono_loma_system_solve();

    // update for next iteration step
    iter_update();

    // check convergence and stop iteration loop if convergence is achieved
    stopnonliniter = convergence_check(itnum);
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void LowMach::Algorithm::set_fluid_values_in_scatra()
{
  // set respective field vectors for velocity/pressure, acceleration
  // and discretization based on time-integration scheme
  switch (fluid_field()->tim_int_scheme())
  {
    case Inpar::FLUID::timeint_afgenalpha:
    {
      scatra_field()->set_acceleration_field(*fluid_field()->accam());
      scatra_field()->set_convective_velocity(*fluid_field()->velaf());
      scatra_field()->set_velocity_field(*fluid_field()->velaf());
      if (scatra_field()->fine_scale_velocity_field_required() and
          fluid_field()->fs_vel() != nullptr)
      {
        scatra_field()->set_fine_scale_velocity(*fluid_field()->fs_vel());
      }
    }
    break;
    case Inpar::FLUID::timeint_one_step_theta:
    case Inpar::FLUID::timeint_bdf2:
    {
      scatra_field()->set_acceleration_field(*fluid_field()->hist());
      scatra_field()->set_convective_velocity(*fluid_field()->velnp());
      scatra_field()->set_velocity_field(*fluid_field()->velnp());
      if (scatra_field()->fine_scale_velocity_field_required() and
          fluid_field()->fs_vel() != nullptr)
      {
        scatra_field()->set_fine_scale_velocity(*fluid_field()->fs_vel());
      }
    }
    break;
    default:
      FOUR_C_THROW("Time integration scheme not supported");
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void LowMach::Algorithm::set_scatra_values_in_fluid()
{
  // set scalar and thermodynamic pressure values as well as time
  // derivatives and discretization based on time-integration scheme
  switch (fluid_field()->tim_int_scheme())
  {
    case Inpar::FLUID::timeint_afgenalpha:
    {
      if (fluid_field()->physical_type() == Inpar::FLUID::tempdepwater)
        fluid_field()->set_iter_scalar_fields(scatra_field()->phiaf(), scatra_field()->phiam(),
            scatra_field()->phidtam(), scatra_field()->discretization());
      else
        fluid_field()->set_loma_iter_scalar_fields(scatra_field()->phiaf(), scatra_field()->phiam(),
            scatra_field()->phidtam(), scatra_field()->fs_phi(),
            std::dynamic_pointer_cast<ScaTra::ScaTraTimIntLoma>(scatra_field())->therm_press_af(),
            std::dynamic_pointer_cast<ScaTra::ScaTraTimIntLoma>(scatra_field())->therm_press_am(),
            std::dynamic_pointer_cast<ScaTra::ScaTraTimIntLoma>(scatra_field())
                ->therm_press_dt_af(),
            std::dynamic_pointer_cast<ScaTra::ScaTraTimIntLoma>(scatra_field())
                ->therm_press_dt_am(),
            scatra_field()->discretization());
    }
    break;
    case Inpar::FLUID::timeint_one_step_theta:
    {
      if (fluid_field()->physical_type() == Inpar::FLUID::tempdepwater)
        fluid_field()->set_iter_scalar_fields(scatra_field()->phinp(), scatra_field()->phin(),
            scatra_field()->phidtnp(), scatra_field()->discretization());
      else
        fluid_field()->set_loma_iter_scalar_fields(scatra_field()->phinp(), scatra_field()->phin(),
            scatra_field()->phidtnp(), scatra_field()->fs_phi(),
            std::dynamic_pointer_cast<ScaTra::ScaTraTimIntLoma>(scatra_field())->therm_press_np(),
            std::dynamic_pointer_cast<ScaTra::ScaTraTimIntLoma>(scatra_field())->therm_press_n(),
            std::dynamic_pointer_cast<ScaTra::ScaTraTimIntLoma>(scatra_field())
                ->therm_press_dt_np(),
            std::dynamic_pointer_cast<ScaTra::ScaTraTimIntLoma>(scatra_field())
                ->therm_press_dt_np(),
            scatra_field()->discretization());
    }
    break;
    default:
      FOUR_C_THROW("Time integration scheme not supported");
      break;
  }
  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void LowMach::Algorithm::setup_mono_loma_matrix()
{
  // set loma block matrix to zero
  lomasystemmatrix_->zero();

  //----------------------------------------------------------------------
  // 1st diagonal block (upper left): fluid weighting - fluid solution
  //----------------------------------------------------------------------
  // get matrix block
  std::shared_ptr<Core::LinAlg::SparseMatrix> mat_ff = fluid_field()->system_matrix();

  // uncomplete matrix block (appears to be required in certain cases)
  mat_ff->un_complete();

  // assign matrix block
  lomasystemmatrix_->assign(0, 0, Core::LinAlg::DataAccess::View, *mat_ff);

  //----------------------------------------------------------------------
  // 2nd diagonal block (lower right): scatra weighting - scatra solution
  //----------------------------------------------------------------------
  // get matrix block
  std::shared_ptr<Core::LinAlg::SparseMatrix> mat_ss = scatra_field()->system_matrix();

  // uncomplete matrix block (appears to be required in certain cases)
  mat_ss->un_complete();

  // assign matrix block
  lomasystemmatrix_->assign(1, 1, Core::LinAlg::DataAccess::View, *mat_ss);

  // complete loma block matrix
  lomasystemmatrix_->complete();

  //----------------------------------------------------------------------
  // 1st off-diagonal block (upper right): fluid weighting - scatra solution
  //----------------------------------------------------------------------
  // create matrix block
  std::shared_ptr<Core::LinAlg::SparseMatrix> mat_fs = nullptr;
  mat_fs = std::make_shared<Core::LinAlg::SparseMatrix>(
      *(fluid_field()->discretization()->dof_row_map(0)), 27, true, true);

  // evaluate loma off-diagonal matrix block in fluid
  evaluate_loma_od_block_mat_fluid(mat_fs);

  // uncomplete matrix block (appears to be required in certain cases)
  mat_fs->un_complete();

  // assign matrix block
  lomasystemmatrix_->assign(0, 1, Core::LinAlg::DataAccess::View, *mat_fs);

  //----------------------------------------------------------------------
  // 2nd off-diagonal block (lower left): scatra weighting - fluid solution
  //----------------------------------------------------------------------
  // create matrix block
  std::shared_ptr<Core::LinAlg::SparseMatrix> mat_sf = nullptr;
  mat_sf = std::make_shared<Core::LinAlg::SparseMatrix>(
      *(scatra_field()->discretization()->dof_row_map(0)), 108, true, true);

  // evaluate loma off-diagonal matrix block in scatra
  // (for present fixed-point-like iteration: no entries)
  // EvaluateLomaODBlockMatScaTra(mat_sf);

  // uncomplete matrix block (appears to be required in certain cases)
  mat_sf->un_complete();

  // assign matrix block
  lomasystemmatrix_->assign(1, 0, Core::LinAlg::DataAccess::View, *mat_sf);

  // complete loma block matrix
  lomasystemmatrix_->complete();
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void LowMach::Algorithm::evaluate_loma_od_block_mat_fluid(
    std::shared_ptr<Core::LinAlg::SparseMatrix> mat_fs)
{
  // create parameters for fluid discretization
  Teuchos::ParameterList fparams;

  // set action type
  fparams.set<FLD::Action>("action", FLD::calc_loma_mono_odblock);

  // set general vector values needed by elements
  fluid_field()->discretization()->clear_state();
  fluid_field()->discretization()->set_state(0, "hist", *fluid_field()->hist());
  fluid_field()->discretization()->set_state(0, "accam", *fluid_field()->accam());
  fluid_field()->discretization()->set_state(0, "scaaf", *fluid_field()->scaaf());
  fluid_field()->discretization()->set_state(0, "scaam", *fluid_field()->scaam());

  // set time-integration-scheme-specific element parameters and vector values
  if (fluid_field()->tim_int_scheme() == Inpar::FLUID::timeint_afgenalpha)
  {
    // set thermodynamic pressures
    fparams.set("thermpress at n+alpha_F/n+1",
        std::dynamic_pointer_cast<ScaTra::ScaTraTimIntLoma>(scatra_field())->therm_press_af());
    fparams.set("thermpress at n+alpha_M/n",
        std::dynamic_pointer_cast<ScaTra::ScaTraTimIntLoma>(scatra_field())->therm_press_am());
    fparams.set("thermpressderiv at n+alpha_F/n+1",
        std::dynamic_pointer_cast<ScaTra::ScaTraTimIntLoma>(scatra_field())->therm_press_dt_af());
    fparams.set("thermpressderiv at n+alpha_M/n+1",
        std::dynamic_pointer_cast<ScaTra::ScaTraTimIntLoma>(scatra_field())->therm_press_dt_am());

    // set velocity vector
    fluid_field()->discretization()->set_state(0, "velaf", *fluid_field()->velaf());
  }
  else if (fluid_field()->tim_int_scheme() == Inpar::FLUID::timeint_one_step_theta)
  {
    // set thermodynamic pressures
    fparams.set("thermpress at n+alpha_F/n+1",
        std::dynamic_pointer_cast<ScaTra::ScaTraTimIntLoma>(scatra_field())->therm_press_np());
    fparams.set("thermpress at n+alpha_M/n",
        std::dynamic_pointer_cast<ScaTra::ScaTraTimIntLoma>(scatra_field())->therm_press_n());
    fparams.set("thermpressderiv at n+alpha_F/n+1",
        std::dynamic_pointer_cast<ScaTra::ScaTraTimIntLoma>(scatra_field())->therm_press_dt_np());
    fparams.set("thermpressderiv at n+alpha_M/n+1",
        std::dynamic_pointer_cast<ScaTra::ScaTraTimIntLoma>(scatra_field())->therm_press_dt_np());

    // set velocity vector
    fluid_field()->discretization()->set_state(0, "velaf", *fluid_field()->velnp());
  }
  else
    FOUR_C_THROW("Time integration scheme not supported");

  // build specific assemble strategy for this off-diagonal matrix block,
  // which is assembled in fluid solver
  // fluid dof set = 0, scatra dof set = 1
  Core::FE::AssembleStrategy fluidstrategy(0,  // rows: fluid dof set
      1,                                       // columns: scatra dof set
      mat_fs, nullptr, nullptr, nullptr, nullptr);

  // evaluate off-diagonal matrix block entries for fluid element
  fluid_field()->discretization()->evaluate(fparams, fluidstrategy);
  fluid_field()->discretization()->clear_state();
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void LowMach::Algorithm::setup_mono_loma_rhs()
{
  // define fluid and scatra residual vectors
  std::shared_ptr<const Core::LinAlg::Vector<double>> fluidres = fluid_field()->rhs();
  std::shared_ptr<const Core::LinAlg::Vector<double>> scatrares = scatra_field()->residual();

  // insert fluid and scatra residual vectors into loma residual vector
  lomablockdofrowmap_.insert_vector(*fluidres, 0, *lomarhs_);
  lomablockdofrowmap_.insert_vector(*scatrares, 1, *lomarhs_);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void LowMach::Algorithm::mono_loma_system_solve()
{
  check_is_init();
  check_is_setup();

  // set incremental solution vector to zero
  lomaincrement_->put_scalar(0.0);

  // apply Dirichlet boundary conditions to system
  Core::LinAlg::apply_dirichlet_to_system(
      *lomasystemmatrix_, *lomaincrement_, *lomarhs_, *zeros_, *lomadbcmap_);

  // solve monolithic low-Mach-number system
  Core::LinAlg::SolverParams solver_params;
  solver_params.refactor = true;
  solver_params.reset = true;
  lomasolver_->solve(lomasystemmatrix_, lomaincrement_, lomarhs_, solver_params);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void LowMach::Algorithm::iter_update()
{
  // define incremental fluid and scatra solution vectors
  std::shared_ptr<const Core::LinAlg::Vector<double>> incfluid;
  std::shared_ptr<const Core::LinAlg::Vector<double>> incscatra;

  // extract incremental fluid and scatra solution vectors
  // from incremental low-Mach-number solution vector
  incfluid = lomablockdofrowmap_.extract_vector(*lomaincrement_, 0);
  incscatra = lomablockdofrowmap_.extract_vector(*lomaincrement_, 1);

  // add incremental fluid and scatra solution vectors to
  // respective solution vectors from last iteration step
  fluid_field()->iter_update(incfluid);
  scatra_field()->update_iter(*incscatra);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool LowMach::Algorithm::convergence_check(int itnum)
{
  // define flags for fluid and scatra convergence check
  bool fluidstopnonliniter = false;
  bool scatrastopnonliniter = false;

  // fluid convergence check
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    std::cout << "\n****************************************\n  CONVERGENCE CHECK FOR ITERATION "
                 "STEP\n****************************************\n";
    std::cout << "\n****************************************\n              FLUID "
                 "CHECK\n****************************************\n";
  }
  fluidstopnonliniter =
      fluid_field()->convergence_check(itnum, itmax_, ittol_, ittol_, ittol_, ittol_);

  // scatra convergence check
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
    std::cout << "\n****************************************\n         SCALAR TRANSPORT "
                 "CHECK\n****************************************\n";
  scatrastopnonliniter = std::dynamic_pointer_cast<ScaTra::ScaTraTimIntLoma>(scatra_field())
                             ->convergence_check(itnum, itmax_, ittol_);

  if (fluidstopnonliniter == true and scatrastopnonliniter == true)
    return true;
  else
    return false;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void LowMach::Algorithm::time_update()
{
  // update scalar
  scatra_field()->update();

  // in case of non-constant thermodynamic pressure: update
  if (consthermpress_ == "No_energy" or consthermpress_ == "No_mass")
    std::dynamic_pointer_cast<ScaTra::ScaTraTimIntLoma>(scatra_field())->update_therm_pressure();

  // update fluid
  fluid_field()->update();

  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void LowMach::Algorithm::output()
{
  // set scalar and thermodynamic pressure at n+1 and SCATRA trueresidual
  // for statistical evaluation and evaluation of Neumann boundary
  // conditions at the beginning of the subsequent time step
  fluid_field()->set_scalar_fields(scatra_field()->phinp(),
      std::dynamic_pointer_cast<ScaTra::ScaTraTimIntLoma>(scatra_field())->therm_press_np(),
      scatra_field()->true_residual(), scatra_field()->discretization());

  // Note: The order is important here! Herein, control file entries are
  // written, defining the order in which the filters handle the
  // discretizations, which in turn defines the dof number ordering of the
  // discretizations.
  fluid_field()->statistics_and_output();

  scatra_field()->check_and_write_output_and_restart();

  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void LowMach::Algorithm::read_inflow_restart(int restart)
{
  // in case a inflow generation in the inflow section has been performed,
  // there are not any scatra results available and the initial field is used
  // caution: if avm3_preparation is called ,e.g., for multifractal subgrid-scale
  //          modeling the physical parameters (dens, visc, diff) are required
  //          to obtain non-zero values which otherwise cause troubles when dividing by them
  //          we have to set the temperature field here
  // set initial scalar field
  fluid_field()->set_scalar_fields(
      scatra_field()->phinp(), 0.0, nullptr, scatra_field()->discretization());
  fluid_field()->read_restart(restart);
  // as read_restart is only called for the fluid_field
  // time and step have not been set in the superior class and the ScaTraField
  set_time_step(fluid_field()->time(), fluid_field()->step());
  scatra_field()->set_time_step(fluid_field()->time(), fluid_field()->step());
  return;
}

FOUR_C_NAMESPACE_CLOSE

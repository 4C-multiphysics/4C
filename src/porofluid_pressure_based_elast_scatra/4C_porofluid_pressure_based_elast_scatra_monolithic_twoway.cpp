// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_porofluid_pressure_based_elast_scatra_monolithic_twoway.hpp"

#include "4C_adapter_art_net.hpp"
#include "4C_adapter_porofluid_pressure_based_wrapper.hpp"
#include "4C_adapter_scatra_base_algorithm.hpp"
#include "4C_adapter_str_structure.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_assemblestrategy.hpp"
#include "4C_global_data.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_equilibrate.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linear_solver_method.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_linear_solver_method_parameters.hpp"
#include "4C_porofluid_pressure_based_elast.hpp"
#include "4C_scatra_ele_action.hpp"
#include "4C_scatra_timint_implicit.hpp"
#include "4C_scatra_timint_meshtying_strategy_artery.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_parameter_list.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
PoroPressureBased::PoroMultiPhaseScaTraMonolithicTwoWay::PoroMultiPhaseScaTraMonolithicTwoWay(
    MPI_Comm comm, const Teuchos::ParameterList& globaltimeparams)
    : PoroMultiPhaseScaTraMonolithic(comm, globaltimeparams),
      ittolinc_(0.0),
      ittolres_(0.0),
      itmax_(0),
      itmin_(1),
      itnum_(0),
      blockrowdofmap_(nullptr),
      equilibration_(nullptr),
      equilibration_method_(Core::LinAlg::EquilibrationMethod::none),
      solveradaptolbetter_(0.0),
      solveradapttol_(false),
      solve_structure_(true),
      struct_offset_(1),
      tolinc_(0.0),
      tolfres_(0.0),
      tolinc_struct_(0.0),
      tolfres_struct_(0.0),
      tolinc_fluid_(0.0),
      tolfres_fluid_(0.0),
      tolinc_scatra_(0.0),
      tolfres_scatra_(0.0),
      normrhs_(0.0),
      normrhsfluid_(0.0),
      normincfluid_(0.0),
      normrhsstruct_(0.0),
      normincstruct_(0.0),
      normrhsscatra_(0.0),
      normincscatra_(0.0),
      normrhsart_(0.0),
      normincart_(0.0),
      arterypressnorm_(0.0),
      normrhsartsca_(0.0),
      normincartsca_(0.0),
      arteryscanorm_(0.0),
      maxinc_(0.0),
      maxres_(0.0),
      vectornormfres_(VectorNorm::undefined),
      vectornorminc_(VectorNorm::undefined),
      timernewton_("", true),
      dtsolve_(0.0),
      dtele_(0.0),
      fdcheck_(PoroPressureBased::FdCheck::none)
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraMonolithicTwoWay::init(
    const Teuchos::ParameterList& globaltimeparams, const Teuchos::ParameterList& algoparams,
    const Teuchos::ParameterList& poroparams, const Teuchos::ParameterList& structparams,
    const Teuchos::ParameterList& fluidparams, const Teuchos::ParameterList& scatraparams,
    const std::string& struct_disname, const std::string& fluid_disname,
    const std::string& scatra_disname, bool isale, int nds_disp, int nds_vel, int nds_solidpressure,
    int ndsporofluid_scatra, const std::map<int, std::set<int>>* nearbyelepairs)
{
  // call base class
  PoroPressureBased::PoroMultiPhaseScaTraMonolithic::init(globaltimeparams, algoparams, poroparams,
      structparams, fluidparams, scatraparams, struct_disname, fluid_disname, scatra_disname, isale,
      nds_disp, nds_vel, nds_solidpressure, ndsporofluid_scatra, nearbyelepairs);

  // read input variables
  itmax_ = algoparams.get<int>("ITEMAX");
  ittolinc_ = algoparams.sublist("MONOLITHIC").get<double>("TOLINC_GLOBAL");
  ittolres_ = algoparams.sublist("MONOLITHIC").get<double>("TOLRES_GLOBAL");

  blockrowdofmap_ = std::make_shared<Core::LinAlg::MultiMapExtractor>();

  fdcheck_ = Teuchos::getIntegralValue<PoroPressureBased::FdCheck>(
      algoparams.sublist("MONOLITHIC"), "FDCHECK");

  equilibration_method_ = Teuchos::getIntegralValue<Core::LinAlg::EquilibrationMethod>(
      algoparams.sublist("MONOLITHIC"), "EQUILIBRATION");

  solveradaptolbetter_ = algoparams.sublist("MONOLITHIC").get<double>("ADAPTCONV_BETTER");
  solveradapttol_ = algoparams.sublist("MONOLITHIC").get<bool>("ADAPTCONV");

  // do we also solve the structure, this is helpful in case of fluid-scatra coupling without mesh
  // deformation
  solve_structure_ = poroparams.get<bool>("SOLVE_STRUCTURE");
  if (!solve_structure_) struct_offset_ = 0;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraMonolithicTwoWay::setup_system()
{
  // setup the poro subsystem first
  poro_field()->setup_system();

  // -------------------------------------------------------------create combined map
  setup_maps();

  //-----------------------------------build map of global dofs with DBC
  build_combined_dbc_map();
  // -------------------------------------------------------------

  // initialize Poroscatra-systemmatrix_
  systemmatrix_ =
      std::make_shared<Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(
          *extractor(), *extractor(), 81, false, true);

  //! structure-scatra coupling matrix k_pss_ --> equal to zero so far
  //! fluid-scatra coupling matrix
  k_pfs_ = std::make_shared<Core::LinAlg::SparseMatrix>(*(poro_field()->fluid_dof_row_map()),
      //*(fluid_field()->dof_row_map()),
      81, true, true);

  //! scatra-structure coupling matrix
  k_sps_ = std::make_shared<Core::LinAlg::SparseMatrix>(
      *(scatra_algo()->scatra_field()->discretization()->dof_row_map()), 81, true, true);
  //! scatra-fluid coupling matrix
  k_spf_ = std::make_shared<Core::LinAlg::SparseMatrix>(
      *(scatra_algo()->scatra_field()->discretization()->dof_row_map()),
      //*(fluid_field()->dof_row_map()),
      81, true, true);

  // instantiate appropriate equilibration class
  auto equilibration_method =
      std::vector<Core::LinAlg::EquilibrationMethod>(1, equilibration_method_);
  equilibration_ = Core::LinAlg::build_equilibration(
      Core::LinAlg::MatrixType::block_field, equilibration_method, fullmap_);

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraMonolithicTwoWay::setup_maps()
{
  // create combined map
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> vecSpaces;

  if (solve_structure_)
  {
    vecSpaces.push_back(poro_field()->struct_dof_row_map());
    vecSpaces.push_back(poro_field()->fluid_dof_row_map());
    const Core::LinAlg::Map* dofrowmapscatra =
        (scatra_algo()->scatra_field()->discretization())->dof_row_map(0);
    vecSpaces.push_back(Core::Utils::shared_ptr_from_ref(*dofrowmapscatra));

    if (vecSpaces[0]->NumGlobalElements() == 0) FOUR_C_THROW("No poro structure equation. Panic.");
    if (vecSpaces[1]->NumGlobalElements() == 0) FOUR_C_THROW("No poro fluid equation. Panic.");
    if (vecSpaces[2]->NumGlobalElements() == 0) FOUR_C_THROW("No scatra equation. Panic.");
  }
  else
  {
    vecSpaces.push_back(poro_field()->fluid_dof_row_map());
    const Core::LinAlg::Map* dofrowmapscatra =
        (scatra_algo()->scatra_field()->discretization())->dof_row_map(0);
    vecSpaces.push_back(Core::Utils::shared_ptr_from_ref(*dofrowmapscatra));

    if (vecSpaces[0]->NumGlobalElements() == 0) FOUR_C_THROW("No poro fluid equation. Panic.");
    if (vecSpaces[1]->NumGlobalElements() == 0) FOUR_C_THROW("No scatra equation. Panic.");
  }

  // full fluid-structure-scatra-map
  fullmap_ = Core::LinAlg::MultiMapExtractor::merge_maps(vecSpaces);

  // full Poromultiphase-elasticity-blockmap
  blockrowdofmap_->setup(*fullmap_, vecSpaces);

  // check global map extractor
  blockrowdofmap_->check_for_valid_map_extractor();

  return;
}

/*-----------------------------------------------------------------------/
/-----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraMonolithicTwoWay::build_combined_dbc_map()
{
  // Combined DBC map of poromultielast-problem
  const std::shared_ptr<const Core::LinAlg::Map> porocondmap = poro_field()->combined_dbc_map();
  const std::shared_ptr<const Core::LinAlg::Map> scatracondmap =
      scatra_algo()->scatra_field()->dirich_maps()->cond_map();
  combinedDBCMap_ = Core::LinAlg::merge_map(porocondmap, scatracondmap, false);

  return;
}

/*-----------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraMonolithicTwoWay::build_block_null_spaces()
{
  // Build block null spaces of structure and fluid-field
  if (solve_structure_) poro_field()->build_block_null_spaces(solver_);
  // only fluid
  else
  {
    Teuchos::ParameterList& blocksmootherparams1 = solver_->params().sublist("Inverse1");
    Core::LinearSolver::Parameters::compute_solver_parameters(
        *poro_field()->fluid_field()->discretization(), blocksmootherparams1);
  }

  Teuchos::ParameterList& blocksmootherparams =
      solver_->params().sublist("Inverse" + std::to_string(struct_offset_ + 2));

  Core::LinearSolver::Parameters::compute_solver_parameters(
      *scatra_algo()->scatra_field()->discretization(), blocksmootherparams);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraMonolithicTwoWay::setup_solver()
{
  //  solver
  // create a linear solver
  // get dynamic section of poroelasticity
  const Teuchos::ParameterList& poromultscatradyn =
      Global::Problem::instance()->poro_multi_phase_scatra_dynamic_params();
  // get the solver number used for linear poroelasticity solver
  const int linsolvernumber = poromultscatradyn.sublist("MONOLITHIC").get<int>("LINEAR_SOLVER");
  // check if the poroelasticity solver has a valid solver number
  if (linsolvernumber == (-1))
    FOUR_C_THROW(
        "no linear solver defined for poromultiphaseflow with scatra coupling.\n"
        " Please set LINEAR_SOLVER in POROMULTIPHASESCATRA DYNAMIC to a valid number!");
  const Teuchos::ParameterList& solverparams =
      Global::Problem::instance()->solver_params(linsolvernumber);
  const auto solvertype =
      Teuchos::getIntegralValue<Core::LinearSolver::SolverType>(solverparams, "SOLVER");

  create_linear_solver(solverparams, solvertype);

  vectornormfres_ = Teuchos::getIntegralValue<PoroPressureBased::VectorNorm>(
      poromultscatradyn.sublist("MONOLITHIC"), "VECTORNORM_RESF");
  vectornorminc_ = Teuchos::getIntegralValue<PoroPressureBased::VectorNorm>(
      poromultscatradyn.sublist("MONOLITHIC"), "VECTORNORM_INC");
}

/*-----------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraMonolithicTwoWay::create_linear_solver(
    const Teuchos::ParameterList& solverparams, const Core::LinearSolver::SolverType solvertype)
{
  solver_ = std::make_shared<Core::LinAlg::Solver>(solverparams, get_comm(),
      Global::Problem::instance()->solver_params_callback(),
      Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
          Global::Problem::instance()->io_params(), "VERBOSITY"));
  // no need to do the rest for direct solvers
  if (solvertype == Core::LinearSolver::SolverType::umfpack or
      solvertype == Core::LinearSolver::SolverType::superlu)
    return;

  if (solvertype != Core::LinearSolver::SolverType::belos)
  {
    std::cout << "!!!!!!!!!!!!!!!!!!!!!! ATTENTION !!!!!!!!!!!!!!!!!!!!!" << std::endl;
    std::cout << " Note: the BGS2x2 preconditioner now " << std::endl;
    std::cout << " uses the structural solver and fluid solver blocks" << std::endl;
    std::cout << " for building the internal inverses" << std::endl;
    std::cout << " Remove the old BGS PRECONDITIONER BLOCK entries " << std::endl;
    std::cout << " in the input files!" << std::endl;
    std::cout << "!!!!!!!!!!!!!!!!!!!!!! ATTENTION !!!!!!!!!!!!!!!!!!!!!" << std::endl;
    FOUR_C_THROW("Iterative solver expected");
  }
  const auto azprectype =
      Teuchos::getIntegralValue<Core::LinearSolver::PreconditionerType>(solverparams, "AZPREC");

  // plausibility check
  switch (azprectype)
  {
    case Core::LinearSolver::PreconditionerType::block_teko:
    {
      // no plausibility checks here
      // if you forget to declare an xml file you will get an error message anyway
    }
    break;
    default:
      FOUR_C_THROW("Block preconditioner expected");
      break;
  }

  // build the null spaces of the single blocks
  build_block_null_spaces();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraMonolithicTwoWay::time_step()
{
  // Prepare stuff
  setup_newton();
  print_header();

  // Evaluate
  evaluate(iterinc_);

  // Newton-Loop
  while ((not converged() and itnum_ < itmax_) or (itnum_ < itmin_))
  {
    // increment number of iteration
    itnum_++;

    // Solve
    linear_solve();
    solver_->reset_tolerance();

    // Build Convergence Norms
    build_convergence_norms();

    // Evaluate
    if (not converged())
    {
      evaluate(iterinc_);
      // perform FD Check of monolithic system matrix
      if (fdcheck_ == FdCheck::global) poro_multi_phase_scatra_fd_check();
    }
    else
    {
      // convergence check is based on residual(phi_i) < tol and phi_i+1 - phi_i < tol
      // in this function we update phi_i+1 as phi_i+1 = phi_i + iterinc for all fields
      // even though we have not evaluated the residual of phi_i+1 it will still be more exact than
      // the one at phi_i
      update_fields_after_convergence();
    }

    // print output
    newton_output();
  }

  // Error-Check
  newton_error_check();

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraMonolithicTwoWay::evaluate(
    std::shared_ptr<const Core::LinAlg::Vector<double>> iterinc)
{
  TEUCHOS_FUNC_TIME_MONITOR("PoroMultiPhaseScaTra::PoroMultiPhaseScaTraMonolithicTwoWay::Evaluate");

  // reset timer
  timernewton_.reset();
  // *********** time measurement ***********
  double dtcpu = timernewton_.wallTime();
  // *********** time measurement ***********

  // displacement, fluid variable and scatra variable incremental vector
  std::shared_ptr<const Core::LinAlg::Vector<double>> porostructinc;
  std::shared_ptr<const Core::LinAlg::Vector<double>> porofluidinc;
  std::shared_ptr<const Core::LinAlg::Vector<double>> scatrainc;
  extract_field_vectors(iterinc, porostructinc, porofluidinc, scatrainc);

  // (1) Newton update of the scatra field
  update_scatra(scatrainc);

  // (2) set scatra solution on fluid field
  set_scatra_solution();

  // (3) access poro problem to build poro-poro block
  poro_field()->evaluate(porostructinc, porofluidinc, itnum_ == 0);

  // (4) set fluid and structure solution on scatra field
  set_poro_solution();

  // (5) access ScaTra problem to build scatra-scatra block
  evaluate_scatra();

  // (6) Build the monolithic system matrix
  setup_system_matrix();

  // check whether we have a sanely filled tangent matrix
  if (not systemmatrix_->filled())
  {
    FOUR_C_THROW("Effective tangent matrix must be filled here");
  }

  // (7) Build the monolithic system vector
  setup_rhs();

  // *********** time measurement ***********
  double mydtele = timernewton_.wallTime() - dtcpu;
  Core::Communication::max_all(&mydtele, &dtele_, 1, get_comm());
  // *********** time measurement ***********
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraMonolithicTwoWay::setup_system_matrix()
{
  // set loma block matrix to zero
  systemmatrix_->zero();

  //----------------------------------------------------------------------
  // 1st diagonal block (upper left): poro weighting - poro solution
  // has dimensions ((ndim+n_phases)*n_nodes)x((ndim+n_phases)*n_nodes)
  //----------------------------------------------------------------------
  // get matrix block
  std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> mat_pp = poro_field()->block_system_matrix();

  // uncomplete matrix block (appears to be required in certain cases (locsys+iterative solver))
  mat_pp->un_complete();

  // assign matrix block
  if (solve_structure_)
  {
    systemmatrix_->assign(0, 0, Core::LinAlg::DataAccess::View, mat_pp->matrix(0, 0));
    systemmatrix_->assign(0, 1, Core::LinAlg::DataAccess::View, mat_pp->matrix(0, 1));
    systemmatrix_->assign(1, 0, Core::LinAlg::DataAccess::View, mat_pp->matrix(1, 0));
  }
  systemmatrix_->assign(
      struct_offset_, struct_offset_, Core::LinAlg::DataAccess::View, mat_pp->matrix(1, 1));

  //----------------------------------------------------------------------
  // 2nd diagonal block (lower right): scatra weighting - scatra solution
  // has dimensions (n_species*n_nodes)x(n_species*n_nodes)
  //----------------------------------------------------------------------
  // get matrix block
  std::shared_ptr<Core::LinAlg::SparseMatrix> mat_ss =
      scatra_algo()->scatra_field()->system_matrix();

  // uncomplete matrix block (appears to be required in certain cases)
  mat_ss->un_complete();

  // assign matrix block
  systemmatrix_->assign(
      struct_offset_ + 1, struct_offset_ + 1, Core::LinAlg::DataAccess::View, *mat_ss);

  // complete scatra block matrix
  systemmatrix_->complete();

  //----------------------------------------------------------------------
  // 1st off-diagonal block k_ps (upper right): poro weighting - scatra solution
  // has dimensions ((ndim+n_phases)*n_nodes)x(n_species*n_nodes)
  // so far no coupling of structure with scatra --> k_pss_ = 0
  // --> dimensions (n_phases*n_nodes)x(n_species*n_nodes)
  //----------------------------------------------------------------------

  // create empty matrix
  std::shared_ptr<Core::LinAlg::SparseMatrix> k_pfs = poro_fluid_scatra_coupling_matrix();

  // call the porofluid-elements and calculate the off-diagonal scatra matrix block
  apply_poro_fluid_scatra_coupl_matrix(k_pfs);

  // apply DBC's also on off-diagonal fluid-scatra coupling block (main-diagonal blocks have already
  // been set, either in poromultielast_monolithic.cpp or in the respective evaluate calls)
  k_pfs->apply_dirichlet(*poro_field()->fluid_field()->get_dbc_map_extractor()->cond_map(), false);

  // uncomplete matrix block (appears to be required in certain cases)
  // k_pss_->UnComplete();
  k_pfs->un_complete();

  // assign matrix block
  // systemmatrix_->Assign(0,2,Core::LinAlg::DataAccess::View,*(k_pss_)); --> zero
  systemmatrix_->assign(
      struct_offset_, struct_offset_ + 1, Core::LinAlg::DataAccess::View, *(k_pfs));

  //----------------------------------------------------------------------
  // 2nd off-diagonal block k_sp (lower left): scatra weighting - poro solution
  // has dimensions (n_species*n_nodes)x((ndim+n_phases)*n_nodes)
  //----------------------------------------------------------------------

  // create empty matrix
  std::shared_ptr<Core::LinAlg::SparseMatrix> k_sps = scatra_struct_coupling_matrix();

  // call the scatra-elements and calculate the off-diagonal structure matrix block
  apply_scatra_struct_coupl_matrix(k_sps);

  // apply DBC's also on off-diagonal scatra-structure coupling block (main-diagonal blocks have
  // already been set, either in poromultielast_monolithic.cpp or in the respective evaluate calls)
  k_sps->apply_dirichlet(*scatra_algo()->scatra_field()->dirich_maps()->cond_map(), false);

  // create empty matrix
  std::shared_ptr<Core::LinAlg::SparseMatrix> k_spf = scatra_poro_fluid_coupling_matrix();

  // call the scatra-elements and calculate the off-diagonal structure matrix block
  apply_scatra_poro_fluid_coupl_matrix(k_spf);

  // apply DBC's also on off-diagonal scatra-fluid coupling block (main-diagonal blocks have already
  // been set, either in poromultielast_monolithic.cpp or in the respective evaluate calls)
  k_spf->apply_dirichlet(*scatra_algo()->scatra_field()->dirich_maps()->cond_map(), false);

  // uncomplete matrix block (appears to be required in certain cases (locsys+iterative solver))
  k_sps->un_complete();
  k_spf->un_complete();

  // assign matrix block
  if (solve_structure_) systemmatrix_->assign(2, 0, Core::LinAlg::DataAccess::View, *(k_sps));
  systemmatrix_->assign(
      struct_offset_ + 1, struct_offset_, Core::LinAlg::DataAccess::View, *(k_spf));

  // complete block matrix
  systemmatrix_->complete();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::SparseMatrix>
PoroPressureBased::PoroMultiPhaseScaTraMonolithicTwoWay::poro_fluid_scatra_coupling_matrix()
{
  std::shared_ptr<Core::LinAlg::SparseMatrix> sparse =
      std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(k_pfs_);
  if (sparse == nullptr) FOUR_C_THROW("cast to Core::LinAlg::SparseMatrix failed!");

  return sparse;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::SparseMatrix>
PoroPressureBased::PoroMultiPhaseScaTraMonolithicTwoWay::scatra_struct_coupling_matrix()
{
  std::shared_ptr<Core::LinAlg::SparseMatrix> sparse =
      std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(k_sps_);
  if (sparse == nullptr) FOUR_C_THROW("cast to Core::LinAlg::SparseMatrix failed!");

  return sparse;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::SparseMatrix>
PoroPressureBased::PoroMultiPhaseScaTraMonolithicTwoWay::scatra_poro_fluid_coupling_matrix()
{
  std::shared_ptr<Core::LinAlg::SparseMatrix> sparse =
      std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(k_spf_);
  if (sparse == nullptr) FOUR_C_THROW("cast to Core::LinAlg::SparseMatrix failed!");

  return sparse;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraMonolithicTwoWay::evaluate_scatra()
{
  scatra_algo()->scatra_field()->prepare_linear_solve();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraMonolithicTwoWay::apply_poro_fluid_scatra_coupl_matrix(
    std::shared_ptr<Core::LinAlg::SparseOperator> k_pfs  //!< off-diagonal tangent matrix term
)
{
  // reset
  k_pfs->zero();
  // evaluate
  poro_field()->fluid_field()->assemble_fluid_scatra_coupling_mat(k_pfs);
  // complete
  k_pfs->complete(scatra_algo()->scatra_field()->system_matrix()->range_map(),
      poro_field()->fluid_field()->system_matrix()->range_map());
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraMonolithicTwoWay::apply_scatra_struct_coupl_matrix(
    std::shared_ptr<Core::LinAlg::SparseOperator> k_sps  //!< off-diagonal tangent matrix term
)
{
  // create the parameters for the discretization
  Teuchos::ParameterList sparams_struct;

  k_sps->zero();

  if (solve_structure_)
  {
    Core::Utils::add_enum_class_to_parameter_list<ScaTra::Action>(
        "action", ScaTra::Action::calc_scatra_mono_odblock_mesh, sparams_struct);
    // other parameters that might be needed by the elements
    sparams_struct.set("delta time", dt());
    sparams_struct.set("total time", time());

    // we cannot employ L2-projection for monolithic coupling yet
    sparams_struct.set<bool>("L2-projection", false);

    scatra_algo()->scatra_field()->discretization()->clear_state();
    scatra_algo()->scatra_field()->discretization()->set_state(
        0, "hist", *scatra_algo()->scatra_field()->hist());
    scatra_algo()->scatra_field()->discretization()->set_state(
        0, "phinp", *scatra_algo()->scatra_field()->phinp());

    // build specific assemble strategy for mechanical-fluid system matrix
    // from the point of view of structure_field:
    // structdofset = 0, fluiddofset = 1
    Core::FE::AssembleStrategy scatrastrategy_struct(0,  // scatradofset for row
        1,                                               // structuredofset for column
        k_sps,                                           // scatra-structure coupling matrix
        nullptr, nullptr, nullptr, nullptr);

    scatra_algo()->scatra_field()->discretization()->evaluate(
        sparams_struct, scatrastrategy_struct);
  }

  // complete
  k_sps->complete(poro_field()->structure_field()->system_matrix()->range_map(),
      scatra_algo()->scatra_field()->system_matrix()->range_map());

  scatra_algo()->scatra_field()->discretization()->clear_state();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraMonolithicTwoWay::apply_scatra_poro_fluid_coupl_matrix(
    std::shared_ptr<Core::LinAlg::SparseOperator> k_spf  //!< off-diagonal tangent matrix term
)
{
  // create the parameters for the discretization
  Teuchos::ParameterList sparams_fluid;

  k_spf->zero();

  Core::Utils::add_enum_class_to_parameter_list<ScaTra::Action>(
      "action", ScaTra::Action::calc_scatra_mono_odblock_fluid, sparams_fluid);
  // other parameters that might be needed by the elements
  sparams_fluid.set("delta time", dt());
  sparams_fluid.set("total time", time());

  // we cannot employ L2-projection for monolithic coupling yet
  sparams_fluid.set<bool>("L2-projection", false);

  scatra_algo()->scatra_field()->discretization()->clear_state();
  scatra_algo()->scatra_field()->discretization()->set_state(
      0, "hist", *scatra_algo()->scatra_field()->hist());
  scatra_algo()->scatra_field()->discretization()->set_state(
      0, "phinp", *scatra_algo()->scatra_field()->phinp());


  // build specific assemble strategy for mechanical-fluid system matrix
  // from the point of view of structure_field:
  // structdofset = 0, fluiddofset = 1
  Core::FE::AssembleStrategy scatrastrategy_fluid(0,  // scatradofset for row
      2,                                              // fluiddofset for column
      k_spf,                                          // scatra-structure coupling matrix
      nullptr, nullptr, nullptr, nullptr);

  scatra_algo()->scatra_field()->discretization()->evaluate(sparams_fluid, scatrastrategy_fluid);

  // complete
  k_spf->complete(poro_field()->fluid_field()->system_matrix()->range_map(),
      scatra_algo()->scatra_field()->system_matrix()->range_map());

  scatra_algo()->scatra_field()->discretization()->clear_state();
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraMonolithicTwoWay::update_fields_after_convergence()
{
  // displacement, fluid variable and scatra variable incremental vector
  std::shared_ptr<const Core::LinAlg::Vector<double>> porostructinc;
  std::shared_ptr<const Core::LinAlg::Vector<double>> porofluidinc;
  std::shared_ptr<const Core::LinAlg::Vector<double>> scatrainc;
  extract_field_vectors(iterinc_, porostructinc, porofluidinc, scatrainc);

  // update ScaTra field
  update_scatra(scatrainc);

  // update structure and fluid field
  poro_field()->update_fields_after_convergence(porostructinc, porofluidinc);
}
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraMonolithicTwoWay::update_scatra(
    std::shared_ptr<const Core::LinAlg::Vector<double>> scatrainc)
{
  scatra_algo()->scatra_field()->update_iter(*scatrainc);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraMonolithicTwoWay::setup_rhs()
{
  // create full monolithic rhs vector
  if (rhs_ == nullptr) rhs_ = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map(), true);

  // note: rhs of fluid-structure system already setup in evaluate call

  // fill the Poroelasticity rhs vector rhs_ with the single field rhss
  setup_vector(*rhs_, poro_field()->rhs(), scatra_algo()->scatra_field()->residual());
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraMonolithicTwoWay::setup_vector(
    Core::LinAlg::Vector<double>& f, std::shared_ptr<const Core::LinAlg::Vector<double>> pv,
    std::shared_ptr<const Core::LinAlg::Vector<double>> sv)
{
  // extract dofs of the two fields
  // and put the poro/scatra field vector into the global vector f
  // noticing the block number

  //  std::shared_ptr<const Core::LinAlg::Vector<double>> psx;
  //  std::shared_ptr<const Core::LinAlg::Vector<double>> pfx;

  if (solve_structure_)
    extractor()->insert_vector(*(poro_field()->extractor()->extract_vector(*pv, 0)), 0, f);
  extractor()->insert_vector(
      *(poro_field()->extractor()->extract_vector(*pv, 1)), struct_offset_, f);
  extractor()->insert_vector(*sv, struct_offset_ + 1, f);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraMonolithicTwoWay::extract_field_vectors(
    std::shared_ptr<const Core::LinAlg::Vector<double>> x,
    std::shared_ptr<const Core::LinAlg::Vector<double>>& stx,
    std::shared_ptr<const Core::LinAlg::Vector<double>>& flx,
    std::shared_ptr<const Core::LinAlg::Vector<double>>& scx)
{
  TEUCHOS_FUNC_TIME_MONITOR(
      "PoroMultiPhaseScaTra::PoroMultiPhaseScaTraMonolithicTwoWay::extract_field_vectors");

  // process structure unknowns of the first field
  if (solve_structure_)
    stx = extractor()->extract_vector(*x, 0);
  else
    stx = std::make_shared<Core::LinAlg::Vector<double>>(*poro_field()->struct_dof_row_map(), true);

  // process fluid unknowns of the second field
  flx = extractor()->extract_vector(*x, struct_offset_);

  // process scatra unknowns of the third field
  scx = extractor()->extract_vector(*x, struct_offset_ + 1);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraMonolithicTwoWay::extract_3d_field_vectors(
    std::shared_ptr<const Core::LinAlg::Vector<double>> x,
    std::shared_ptr<const Core::LinAlg::Vector<double>>& stx,
    std::shared_ptr<const Core::LinAlg::Vector<double>>& flx,
    std::shared_ptr<const Core::LinAlg::Vector<double>>& scx)
{
  PoroPressureBased::PoroMultiPhaseScaTraMonolithicTwoWay::extract_field_vectors(x, stx, flx, scx);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraMonolithicTwoWay::linear_solve()
{
  // reset timer
  timernewton_.reset();
  // *********** time measurement ***********
  double dtcpu = timernewton_.wallTime();
  // *********** time measurement ***********
  Core::LinAlg::SolverParams solver_params;
  if (solveradapttol_ and (itnum_ > 1))
  {
    solver_params.nonlin_tolerance = ittolres_;
    solver_params.nonlin_residual = std::max(maxinc_, maxres_);
    solver_params.lin_tol_better = solveradaptolbetter_;
  }
  iterinc_->put_scalar(0.0);  // Useful? depends on solver and more

  // equilibrate global system of equations if necessary
  equilibration_->equilibrate_system(systemmatrix_, rhs_, blockrowdofmap_);

  // standard solver call
  // system is ready to solve since Dirichlet Boundary conditions have been applied in
  // setup_system_matrix or Evaluate
  solver_params.refactor = true;
  solver_params.reset = itnum_ == 1;
  solver_->solve(systemmatrix_->epetra_operator(), iterinc_, rhs_, solver_params);

  equilibration_->unequilibrate_increment(iterinc_);

  // *********** time measurement ***********
  double mydtsolve = timernewton_.wallTime() - dtcpu;
  Core::Communication::max_all(&mydtsolve, &dtsolve_, 1, get_comm());
  // *********** time measurement ***********
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool PoroPressureBased::PoroMultiPhaseScaTraMonolithicTwoWay::converged()
{
  return (normincfluid_ < ittolinc_ && normincstruct_ < ittolinc_ && normincscatra_ < ittolinc_ &&
          normincart_ < ittolinc_ && normincartsca_ < ittolinc_ && normrhs_ < ittolres_ &&
          normrhsfluid_ < ittolres_ && normrhsstruct_ < ittolres_ && normrhsscatra_ < ittolres_ &&
          normrhsart_ < ittolres_ && normrhsartsca_ < ittolres_);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraMonolithicTwoWay::build_convergence_norms()
{
  //------------------------------------------------------------ build residual force norms
  normrhs_ = calculate_vector_norm(vectornormfres_, *rhs_);
  std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_st;
  std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_fl;
  std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_sc;

  // get structure and fluid RHS
  extract_3d_field_vectors(rhs_, rhs_st, rhs_fl, rhs_sc);

  // build also norms for structure, fluid and scatra
  normrhsstruct_ = calculate_vector_norm(vectornormfres_, *rhs_st);
  normrhsfluid_ = calculate_vector_norm(vectornormfres_, *rhs_fl);
  normrhsscatra_ = calculate_vector_norm(vectornormfres_, *rhs_sc);

  //------------------------------------------------------------- build residual increment norms
  // displacement and fluid velocity & pressure incremental vector
  std::shared_ptr<const Core::LinAlg::Vector<double>> iterincst;
  std::shared_ptr<const Core::LinAlg::Vector<double>> iterincfl;
  std::shared_ptr<const Core::LinAlg::Vector<double>> iterincsc;

  // get structure and fluid increment
  extract_3d_field_vectors(iterinc_, iterincst, iterincfl, iterincsc);

  // build also norms for fluid and structure
  normincstruct_ = calculate_vector_norm(vectornorminc_, *iterincst);
  normincfluid_ = calculate_vector_norm(vectornorminc_, *iterincfl);
  normincscatra_ = calculate_vector_norm(vectornorminc_, *iterincsc);

  double dispnorm =
      calculate_vector_norm(vectornorminc_, *poro_field()->structure_field()->dispnp());
  double fluidnorm = calculate_vector_norm(vectornorminc_, *poro_field()->fluid_field()->phinp());
  double scatranorm =
      calculate_vector_norm(vectornorminc_, *scatra_algo()->scatra_field()->phinp());

  // take care of very small norms
  if (dispnorm < 1.0e-6) dispnorm = 1.0;
  if (fluidnorm < 1.0e-6) fluidnorm = 1.0;
  if (scatranorm < 1.0e-6) scatranorm = 1.0;
  if (arterypressnorm_ < 1.0e-6) arterypressnorm_ = 1.0;
  if (arteryscanorm_ < 1.0e-6) arteryscanorm_ = 1.0;

  // build relative increment norm
  normincstruct_ /= dispnorm;
  normincfluid_ /= fluidnorm;
  normincscatra_ /= scatranorm;
  normincart_ /= arterypressnorm_;
  normincartsca_ /= arteryscanorm_;

  // build the maximum value of the residuals and increments
  maxinc_ = std::max({normincfluid_, normincstruct_, normincscatra_, normincart_, normincartsca_});
  maxres_ = std::max(
      {normrhs_, normrhsfluid_, normrhsstruct_, normrhsscatra_, normrhsart_, normrhsartsca_});
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraMonolithicTwoWay::setup_newton()
{
  // initialise equilibrium loop and norms
  itnum_ = 0;
  normrhs_ = 0.0;
  normrhsfluid_ = 0.0;
  normincfluid_ = 0.0;
  normrhsstruct_ = 0.0;
  normincstruct_ = 0.0;
  normrhsscatra_ = 0.0;
  normincscatra_ = 0.0;
  tolinc_ = 0.0;
  tolfres_ = 0.0;
  tolinc_struct_ = 0.0;
  tolfres_struct_ = 0.0;
  tolinc_fluid_ = 0.0;
  tolfres_fluid_ = 0.0;
  tolinc_scatra_ = 0.0;
  tolfres_scatra_ = 0.0;
  normrhsart_ = 0.0;
  normincart_ = 0.0;
  arterypressnorm_ = 0.0;
  normrhsartsca_ = 0.0;
  normincartsca_ = 0.0;
  arteryscanorm_ = 0.0;
  maxinc_ = 0.0;
  maxres_ = 0.0;

  // incremental solution vector with length of all dofs
  if (iterinc_ == nullptr)
    iterinc_ = Core::LinAlg::create_vector(*dof_row_map(), true);
  else
    iterinc_->put_scalar(0.0);

  // a zero vector of full length
  if (zeros_ == nullptr)
    zeros_ = Core::LinAlg::create_vector(*dof_row_map(), true);
  else
    zeros_->put_scalar(0.0);

  // AitkenReset();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraMonolithicTwoWay::newton_output()
{
  // print the incremental based convergence check to the screen
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    if (itnum_ == 1)
      printf(
          "+--------------+-------------+-------------+--------------+------------+-----"
          "-------+-----------------+\n");
    printf(
        "|-  step/max  -|- fluid-inc -|- displ-inc -|- scatra-inc -|-  1Dp-inc -|- "
        " 1Ds-inc -|- norm(tot-rhs) -| (ts =%10.3E,",
        dtsolve_);
    printf("\n");
    printf(
        "|   %3d/%3d    | %10.3E  | %10.3E  |  %10.3E  | %10.3E | %10.3E |   %10.3E    |  "
        "te =%10.3E)",
        itnum_, itmax_, normincfluid_, normincstruct_, normincscatra_, normincart_, normincartsca_,
        normrhs_, dtele_);
    printf("\n");
    printf(
        "+--------------+-------------+-------------+--------------+------------+-----"
        "-------+-----------------+\n");
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraMonolithicTwoWay::newton_error_check()
{
  // print the incremental based convergence check to the screen
  if (converged())  // norminc_ < ittolinc_ && normrhs_ < ittolinc_ && normincfluid_ < ittolinc_ &&
                    // normincstruct_ < ittolinc_
  {
    if (Core::Communication::my_mpi_rank(get_comm()) == 0)
    {
      printf(
          "|  Monolithic iteration loop converged after iteration %3d/%3d !                        "
          "              |\n",
          itnum_, itmax_);
      printf(
          "|  Quantity           [norm]:                 TOL                                       "
          "              |\n");
      printf(
          "|  Max. rel. increment [%3s]:  %10.3E  < %10.3E                                        "
          "       |\n",
          EnumTools::enum_name(vectornorminc_).data(), maxinc_, ittolinc_);
      printf(
          "|  Maximum    residual [%3s]:  %10.3E  < %10.3E                                        "
          "       |\n",
          EnumTools::enum_name(vectornormfres_).data(), maxres_, ittolres_);
      printf(
          "+--------------+-------------+-------------+--------------+------------+-----"
          "-------+-----------------+\n");
      printf("\n");
    }
  }
  else
  {
    if ((Core::Communication::my_mpi_rank(get_comm()) == 0))
    {
      printf(
          "|     >>>>>> not converged in %3d steps!                                                "
          "       |\n",
          itmax_);
      printf(
          "|  Quantity           [norm]:                 TOL                                       "
          "       |\n");
      printf(
          "|  Max. rel. increment [%3s]:  %10.3E    %10.3E                                        "
          "|\n",
          EnumTools::enum_name(vectornorminc_).data(), maxinc_, ittolinc_);
      printf(
          "|  Maximum    residual [%3s]:  %10.3E    %10.3E                                        "
          "|\n",
          EnumTools::enum_name(vectornormfres_).data(), maxres_, ittolres_);
      printf(
          "+--------------+-------------+-------------+--------------+------------+-----"
          "-------+-----------------+\n");
      printf("\n");
      printf("\n");
    }
    handle_divergence();
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Map>
PoroPressureBased::PoroMultiPhaseScaTraMonolithicTwoWay::dof_row_map()
{
  return blockrowdofmap_->full_map();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraMonolithicTwoWay::print_header()
{
  if (!solve_structure_) print_structure_disabled_info();
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    std::cout << "+--------------------------------------------------------------------------------"
                 "---------------------+"
              << std::endl;
    std::cout << "| MONOLITHIC POROMULTIPHASE-SCATRA SOLVER                                        "
                 "                     |"
              << std::endl;
    std::cout << "| STEP: " << std::setw(5) << std::setprecision(4) << std::scientific << step()
              << "/" << std::setw(5) << std::setprecision(4) << std::scientific << n_step()
              << ", Time: " << std::setw(11) << std::setprecision(4) << std::scientific << time()
              << "/" << std::setw(11) << std::setprecision(4) << std::scientific << max_time()
              << ", Dt: " << std::setw(11) << std::setprecision(4) << std::scientific << dt()
              << "                                   |" << std::endl;
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraMonolithicTwoWay::print_structure_disabled_info()
{
  // print out Info
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    std::cout << "\n";
    std::cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
                 "++++++++++++++++++++++++++++++++\n";
    std::cout << " INFO:    STRUCTURE FIELD IS NOT SOLVED; MAKE SURE YOU HAVE CONSTRAINED ALL DOFS "
                 "IN YOUR STRUCTURE WITH A DBC\n";
    std::cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
                 "++++++++++++++++++++++++++++++++\n";
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraMonolithicTwoWay::poro_multi_phase_scatra_fd_check()
{
  std::cout << "\n******************finite difference check***************" << std::endl;

  int dof_struct = (poro_field()->structure_field()->dof_row_map()->NumGlobalElements());
  int dof_fluid = (poro_field()->fluid_field()->dof_row_map()->NumGlobalElements());
  int dof_scatra = (scatra_algo()->scatra_field()->dof_row_map()->NumGlobalElements());

  std::cout << "structure field has " << dof_struct << " DOFs" << std::endl;
  std::cout << "fluid field has " << dof_fluid << " DOFs" << std::endl;
  std::cout << "scatra field has " << dof_scatra << " DOFs" << std::endl;
  if (artery_coupl_)
  {
    int dof_artery = (poro_field()->fluid_field()->artery_dof_row_map()->NumGlobalElements());
    int dof_artscatra = (scatramsht_->art_scatra_field()->dof_row_map()->NumGlobalElements());
    std::cout << "artery field has " << dof_artery << " DOFs" << std::endl;
    std::cout << "artery-scatra field has " << dof_artscatra << " DOFs" << std::endl;

    std::cout << "\n\n============================================================\n"
                 "WARNING: THIS FD CHECK DOES NOT WORK FOR NODE BASED COUPLING\n"
                 "============================================================\n\n";
  }

  std::shared_ptr<Core::LinAlg::Vector<double>> iterinc = nullptr;
  iterinc = Core::LinAlg::create_vector(*dof_row_map(), true);

  const int dofs = iterinc->global_length();
  std::cout << "in total " << dofs << " DOFs" << std::endl;
  const double delta = 1e-8;

  iterinc->put_scalar(0.0);

  iterinc->replace_global_value(0, 0, delta);

  std::shared_ptr<Epetra_CrsMatrix> stiff_approx = nullptr;
  stiff_approx = Core::LinAlg::create_matrix(*dof_row_map(), 81);

  Core::LinAlg::Vector<double> rhs_old(*dof_row_map(), true);
  rhs_old.update(1.0, *rhs_, 0.0);
  Core::LinAlg::Vector<double> rhs_copy(*dof_row_map(), true);

  std::shared_ptr<Core::LinAlg::SparseMatrix> sparse = systemmatrix_->merge();
  Core::LinAlg::SparseMatrix sparse_copy(*sparse, Core::LinAlg::DataAccess::Copy);


  const int zeilennr = -1;
  const int spaltenr = -1;
  for (int i = 0; i < dofs; ++i)
  {
    if (combined_dbc_map()->MyGID(i))
    {
      iterinc->replace_global_value(i, 0, 0.0);
    }

    if (i == spaltenr)
      std::cout << "\n******************" << spaltenr + 1 << ". Spalte!!***************"
                << std::endl;

    evaluate(iterinc);
    setup_rhs();

    rhs_copy.update(1.0, *rhs_, 0.0);

    iterinc_->put_scalar(0.0);  // Useful? depends on solver and more
    Core::LinAlg::apply_dirichlet_to_system(
        sparse_copy, *iterinc_, rhs_copy, *zeros_, *combined_dbc_map());


    if (i == spaltenr)
    {
      std::cout << "rhs_: " << (rhs_copy)[zeilennr] << std::endl;
      std::cout << "rhs_old: " << (rhs_old)[zeilennr] << std::endl;
    }

    rhs_copy.update(-1.0, rhs_old, 1.0);
    rhs_copy.scale(-1.0 / delta);

    int* index = &i;
    for (int j = 0; j < dofs; ++j)
    {
      double value = (rhs_copy)[j];
      stiff_approx->InsertGlobalValues(j, 1, &value, index);

      if ((j == zeilennr) and (i == spaltenr))
      {
        std::cout << "\n******************" << zeilennr + 1 << ". Zeile!!***************"
                  << std::endl;
        // std::cout << "iterinc_" << std::endl << *iterinc_ << std::endl;
        // std::cout << "iterinc" << std::endl << *iterinc << std::endl;
        //  std::cout << "meshdisp: " << std::endl << *(poro_field()->fluid_field()->Dispnp());
        //  std::cout << "meshdisp scatra: " << std::endl <<
        //  *(ScaTraField()->discretization()->GetState(ScaTraField()->NdsDisp(),"dispnp"));
        // std::cout << "disp: " << std::endl << *(poro_field()->structure_field()->dispnp());
        //  std::cout << "fluid vel" << std::endl << *(poro_field()->fluid_field()->Velnp());
        //  std::cout << "scatra vel" << std::endl <<
        //  *(ScaTraField()->discretization()->GetState(ScaTraField()->NdsVel(),"velocity field"));
        //  std::cout << "fluid acc" << std::endl << *(poro_field()->fluid_field()->Accnp());
        //  std::cout << "gridvel fluid" << std::endl << *(poro_field()->fluid_field()->GridVel());
        // std::cout << "gridvel struct" << std::endl <<
        // *(poro_field()->structure_field()->velnp());

        // std::cout << "stiff_apprx(" << zeilennr << "," << spaltenr << "): " <<
        // (*rhs_copy)[zeilennr]
        //          << std::endl;

        // std::cout << "value(" << zeilennr << "," << spaltenr << "): " << value << std::endl;
        std::cout << "\n******************" << zeilennr + 1 << ". Zeile End!!***************"
                  << std::endl;
      }
    }

    if (not combined_dbc_map()->MyGID(i)) iterinc->replace_global_value(i, 0, -delta);

    iterinc->replace_global_value(i - 1, 0, 0.0);

    if (i != dofs - 1) iterinc->replace_global_value(i + 1, 0, delta);

    if (i == spaltenr)
      std::cout << "\n******************" << spaltenr + 1 << ". Spalte End!!***************"
                << std::endl;
  }

  evaluate(iterinc);
  setup_rhs();

  stiff_approx->FillComplete();

  std::shared_ptr<Core::LinAlg::SparseMatrix> stiff_approx_sparse = nullptr;
  stiff_approx_sparse =
      std::make_shared<Core::LinAlg::SparseMatrix>(stiff_approx, Core::LinAlg::DataAccess::Copy);

  stiff_approx_sparse->add(sparse_copy, false, -1.0, 1.0);

  std::shared_ptr<Epetra_CrsMatrix> sparse_crs = sparse_copy.epetra_matrix();

  std::shared_ptr<Epetra_CrsMatrix> error_crs = stiff_approx_sparse->epetra_matrix();

  error_crs->FillComplete();
  sparse_crs->FillComplete();

  bool success = true;
  double error_max_rel = 0.0;
  double error_max_abs = 0.0;
  for (int i = 0; i < dofs; ++i)
  {
    if (not combined_dbc_map()->MyGID(i))
    {
      for (int j = 0; j < dofs; ++j)
      {
        if (not combined_dbc_map()->MyGID(j))
        {
          double stiff_approx_ij = 0.0;
          double sparse_ij = 0.0;
          double error_ij = 0.0;

          {
            // get error_crs entry ij
            int errornumentries;
            int errorlength = error_crs->NumGlobalEntries(i);
            std::vector<double> errorvalues(errorlength);
            std::vector<int> errorindices(errorlength);
            // int errorextractionstatus =
            error_crs->ExtractGlobalRowCopy(
                i, errorlength, errornumentries, errorvalues.data(), errorindices.data());
            for (int k = 0; k < errorlength; ++k)
            {
              if (errorindices[k] == j)
              {
                error_ij = errorvalues[k];
                break;
              }
              else
                error_ij = 0.0;
            }
          }

          // get sparse_ij entry ij
          {
            int sparsenumentries;
            int sparselength = sparse_crs->NumGlobalEntries(i);
            std::vector<double> sparsevalues(sparselength);
            std::vector<int> sparseindices(sparselength);
            // int sparseextractionstatus =
            sparse_crs->ExtractGlobalRowCopy(
                i, sparselength, sparsenumentries, sparsevalues.data(), sparseindices.data());
            for (int k = 0; k < sparselength; ++k)
            {
              if (sparseindices[k] == j)
              {
                sparse_ij = sparsevalues[k];
                break;
              }
              else
                sparse_ij = 0.0;
            }
          }

          // get stiff_approx entry ij
          {
            int approxnumentries;
            int approxlength = stiff_approx->NumGlobalEntries(i);
            std::vector<double> approxvalues(approxlength);
            std::vector<int> approxindices(approxlength);
            // int approxextractionstatus =
            stiff_approx->ExtractGlobalRowCopy(
                i, approxlength, approxnumentries, approxvalues.data(), approxindices.data());
            for (int k = 0; k < approxlength; ++k)
            {
              if (approxindices[k] == j)
              {
                stiff_approx_ij = approxvalues[k];
                break;
              }
              else
                stiff_approx_ij = 0.0;
            }
          }

          double error = 0.0;
          if (abs(stiff_approx_ij) > 1e-5)
            error = error_ij / (stiff_approx_ij);
          else if (abs(sparse_ij) > 1e-5)
            error = error_ij / (sparse_ij);

          if (abs(error) > abs(error_max_rel)) error_max_rel = abs(error);
          if (abs(error_ij) > abs(error_max_abs)) error_max_abs = abs(error_ij);

          if ((abs(error) > 1e-4))
          {
            if ((abs(error_ij) > 1e-5))
            //  if( (sparse_ij>1e-1) or (stiff_approx_ij>1e-1) )
            {
              std::cout << "finite difference check failed entry (" << i << "," << j
                        << ")! stiff: " << sparse_ij << ", approx: " << stiff_approx_ij
                        << " ,abs. error: " << error_ij << " , rel. error: " << error << std::endl;

              success = false;
            }
          }
        }
      }
    }
  }

  if (success)
  {
    std::cout << "finite difference check successful, max. rel. error: " << error_max_rel
              << " , max. abs. error: " << error_max_abs << std::endl;
    std::cout << "******************finite difference check done***************\n\n" << std::endl;
  }
  else
    FOUR_C_THROW("PoroFDCheck failed");
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
PoroPressureBased::PoroMultiPhaseScaTraMonolithicTwoWayArteryCoupling::
    PoroMultiPhaseScaTraMonolithicTwoWayArteryCoupling(
        MPI_Comm comm, const Teuchos::ParameterList& globaltimeparams)
    : PoroMultiPhaseScaTraMonolithicTwoWay(comm, globaltimeparams)
{
  blockrowdofmap_artscatra_ = std::make_shared<Core::LinAlg::MultiMapExtractor>();
  blockrowdofmap_artporo_ = std::make_shared<Core::LinAlg::MultiMapExtractor>();
  nodal_coupl_inactive_ = false;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraMonolithicTwoWayArteryCoupling::setup_system()
{
  PoroMultiPhaseScaTraMonolithicTwoWay::setup_system();

  //! arteryscatra-artery coupling matrix, this matrix has the full map of all coupled + uncoupled
  //! DOFs
  k_asa_ = std::make_shared<Core::LinAlg::SparseMatrix>(
      *(scatramsht_->art_scatra_field()->discretization()->dof_row_map()), 81, true, true);

  //! simple check if nodal coupling active or not, if condensed and un-condensed dofrowmaps have
  //! equal size
  nodal_coupl_inactive_ =
      ((poro_field()->artery_dof_row_map()->NumGlobalElements() == poro_field()
                                                                       ->fluid_field()
                                                                       ->art_net_tim_int()
                                                                       ->discretization()
                                                                       ->dof_row_map(0)
                                                                       ->NumGlobalElements())) &&
      (scatramsht_->art_scatra_dof_row_map()->NumGlobalElements() ==
          scatramsht_->art_scatra_field()->discretization()->dof_row_map(0)->NumGlobalElements());
}
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraMonolithicTwoWayArteryCoupling::setup_maps()
{
  // create combined map
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> vecSpaces;

  if (solve_structure_)
  {
    vecSpaces.push_back(poro_field()->struct_dof_row_map());
    vecSpaces.push_back(poro_field()->fluid_dof_row_map());
    const Core::LinAlg::Map* dofrowmapscatra =
        (scatra_algo()->scatra_field()->discretization())->dof_row_map(0);
    vecSpaces.push_back(Core::Utils::shared_ptr_from_ref(*dofrowmapscatra));
    vecSpaces.push_back(poro_field()->artery_dof_row_map());
    vecSpaces.push_back(scatramsht_->art_scatra_dof_row_map());
    if (vecSpaces[0]->NumGlobalElements() == 0) FOUR_C_THROW("No poro structure equation. Panic.");
    if (vecSpaces[1]->NumGlobalElements() == 0) FOUR_C_THROW("No poro fluid equation. Panic.");
    if (vecSpaces[2]->NumGlobalElements() == 0) FOUR_C_THROW("No scatra equation. Panic.");
    if (vecSpaces[3]->NumGlobalElements() == 0) FOUR_C_THROW("No artery equation. Panic.");
    if (vecSpaces[4]->NumGlobalElements() == 0) FOUR_C_THROW("No artery scatra equation. Panic.");
  }
  else
  {
    vecSpaces.push_back(poro_field()->fluid_dof_row_map());
    const Core::LinAlg::Map* dofrowmapscatra =
        (scatra_algo()->scatra_field()->discretization())->dof_row_map(0);
    vecSpaces.push_back(Core::Utils::shared_ptr_from_ref(*dofrowmapscatra));
    vecSpaces.push_back(poro_field()->artery_dof_row_map());
    vecSpaces.push_back(scatramsht_->art_scatra_dof_row_map());
    if (vecSpaces[0]->NumGlobalElements() == 0) FOUR_C_THROW("No poro fluid equation. Panic.");
    if (vecSpaces[1]->NumGlobalElements() == 0) FOUR_C_THROW("No scatra equation. Panic.");
    if (vecSpaces[2]->NumGlobalElements() == 0) FOUR_C_THROW("No artery equation. Panic.");
    if (vecSpaces[3]->NumGlobalElements() == 0) FOUR_C_THROW("No artery scatra equation. Panic.");
  }

  // full fluid-structure-scatra-artery-arteryscatra map
  fullmap_ = Core::LinAlg::MultiMapExtractor::merge_maps(vecSpaces);

  // full Poromultiphasescatra block map coupled with artery network
  blockrowdofmap_->setup(*fullmap_, vecSpaces);

  // check global map extractor
  blockrowdofmap_->check_for_valid_map_extractor();

  // full porofluid-artery map
  fullmap_artporo_ = Core::LinAlg::MultiMapExtractor::merge_maps(
      {vecSpaces[struct_offset_], vecSpaces[struct_offset_ + 2]});

  // full porofluid-artery blockmap
  blockrowdofmap_artporo_->setup(
      *fullmap_artporo_, {vecSpaces[struct_offset_], vecSpaces[struct_offset_ + 2]});

  // full artery-arteryscatra map
  fullmap_artscatra_ = Core::LinAlg::MultiMapExtractor::merge_maps(
      {vecSpaces[struct_offset_ + 1], vecSpaces[struct_offset_ + 3]});

  // full artery-arteryscatra blockmap
  blockrowdofmap_artscatra_->setup(
      *fullmap_artscatra_, {vecSpaces[struct_offset_ + 1], vecSpaces[struct_offset_ + 3]});
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraMonolithicTwoWayArteryCoupling::update_scatra(
    std::shared_ptr<const Core::LinAlg::Vector<double>> scatrainc)
{
  scatra_algo()->scatra_field()->update_iter(
      *blockrowdofmap_artscatra_->extract_vector(*scatrainc, 0));
  scatramsht_->update_art_scatra_iter(scatrainc);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraMonolithicTwoWayArteryCoupling::extract_field_vectors(
    std::shared_ptr<const Core::LinAlg::Vector<double>> x,
    std::shared_ptr<const Core::LinAlg::Vector<double>>& stx,
    std::shared_ptr<const Core::LinAlg::Vector<double>>& flx,
    std::shared_ptr<const Core::LinAlg::Vector<double>>& scx)
{
  TEUCHOS_FUNC_TIME_MONITOR(
      "PoroMultiPhaseScaTra::PoroMultiPhaseScaTraMonolithicTwoWay::extract_field_vectors");

  // process structure unknowns of the first field
  if (solve_structure_)
    stx = extractor()->extract_vector(*x, 0);
  else
    stx = std::make_shared<Core::LinAlg::Vector<double>>(*poro_field()->struct_dof_row_map(), true);

  // process artery and porofluid unknowns
  std::shared_ptr<const Core::LinAlg::Vector<double>> porofluid =
      extractor()->extract_vector(*x, struct_offset_);
  std::shared_ptr<const Core::LinAlg::Vector<double>> artery =
      extractor()->extract_vector(*x, struct_offset_ + 2);

  std::shared_ptr<Core::LinAlg::Vector<double>> dummy1 =
      std::make_shared<Core::LinAlg::Vector<double>>(*fullmap_artporo_);

  // build the combined increment of porofluid and artery
  blockrowdofmap_artporo_->insert_vector(*porofluid, 0, *dummy1);
  blockrowdofmap_artporo_->insert_vector(*artery, 1, *dummy1);

  flx = dummy1;

  // process scatra and artery scatra unknowns of the third field
  std::shared_ptr<const Core::LinAlg::Vector<double>> scatra =
      extractor()->extract_vector(*x, struct_offset_ + 1);
  std::shared_ptr<const Core::LinAlg::Vector<double>> artscatra =
      extractor()->extract_vector(*x, struct_offset_ + 3);

  std::shared_ptr<Core::LinAlg::Vector<double>> dummy2 =
      std::make_shared<Core::LinAlg::Vector<double>>(*fullmap_artscatra_);

  // build the combined increment of artery and artery-scatra
  blockrowdofmap_artscatra_->insert_vector(*scatra, 0, *dummy2);
  blockrowdofmap_artscatra_->insert_vector(*artscatra, 1, *dummy2);

  scx = dummy2;
}
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraMonolithicTwoWayArteryCoupling::setup_system_matrix()
{
  PoroMultiPhaseScaTraMonolithicTwoWay::setup_system_matrix();

  // --------------------------------------------------------------------------- artery-porofluid
  // get matrix block
  std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> mat_pp = poro_field()->block_system_matrix();

  // artery part
  systemmatrix_->assign(
      struct_offset_ + 2, struct_offset_ + 2, Core::LinAlg::DataAccess::View, mat_pp->matrix(2, 2));
  // artery-porofluid part
  systemmatrix_->assign(
      struct_offset_ + 2, struct_offset_, Core::LinAlg::DataAccess::View, mat_pp->matrix(2, 1));
  // porofluid-artery part
  systemmatrix_->assign(
      struct_offset_, struct_offset_ + 2, Core::LinAlg::DataAccess::View, mat_pp->matrix(1, 2));

  // -------------------------------------------------------------------------arteryscatra-scatra
  // arteryscatra part
  systemmatrix_->assign(struct_offset_ + 3, struct_offset_ + 3, Core::LinAlg::DataAccess::View,
      scatramsht_->combined_system_matrix()->matrix(1, 1));
  // scatra-arteryscatra part
  systemmatrix_->assign(struct_offset_ + 1, struct_offset_ + 3, Core::LinAlg::DataAccess::View,
      scatramsht_->combined_system_matrix()->matrix(0, 1));
  // arteryscatra-scatra part
  systemmatrix_->assign(struct_offset_ + 3, struct_offset_ + 1, Core::LinAlg::DataAccess::View,
      scatramsht_->combined_system_matrix()->matrix(1, 0));

  if (nodal_coupl_inactive_)
  {
    // create empty matrix
    std::shared_ptr<Core::LinAlg::SparseMatrix> k_asa = artery_scatra_artery_coupling_matrix();

    // call the scatra-elements and calculate the off-diagonal structure matrix block
    apply_artery_scatra_artery_coupl_matrix(k_asa);

    // apply DBC's also on off-diagonal scatra-fluid coupling block (main-diagonal blocks have
    // already been set, either in poromultielast_monolithic.cpp or in the respective evaluate
    // calls)
    k_asa->apply_dirichlet(*scatramsht_->art_scatra_field()->dirich_maps()->cond_map(), false);

    // arteryscatra-scatra part
    systemmatrix_->assign(
        struct_offset_ + 3, struct_offset_ + 2, Core::LinAlg::DataAccess::View, *k_asa);
  }

  systemmatrix_->complete();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraMonolithicTwoWayArteryCoupling::setup_rhs()
{
  // create full monolithic rhs vector
  if (rhs_ == nullptr) rhs_ = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map(), true);

  // structure
  if (solve_structure_)
    extractor()->insert_vector(
        *(poro_field()->extractor()->extract_vector(*poro_field()->rhs(), 0)), 0, *rhs_);
  // porofluid
  extractor()->insert_vector(
      *(poro_field()->extractor()->extract_vector(*poro_field()->rhs(), 1)), struct_offset_, *rhs_);
  // scatra
  extractor()->insert_vector(
      *(blockrowdofmap_artscatra_->extract_vector(*scatramsht_->combined_rhs(), 0)),
      struct_offset_ + 1, *rhs_);

  // artery
  extractor()->insert_vector(*(poro_field()->extractor()->extract_vector(*poro_field()->rhs(), 2)),
      struct_offset_ + 2, *rhs_);
  // arteryscatra
  extractor()->insert_vector(
      *(blockrowdofmap_artscatra_->extract_vector(*scatramsht_->combined_rhs(), 1)),
      struct_offset_ + 3, *rhs_);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraMonolithicTwoWayArteryCoupling::
    build_convergence_norms()
{
  std::shared_ptr<const Core::LinAlg::Vector<double>> arteryrhs =
      extractor()->extract_vector(*rhs_, struct_offset_ + 2);
  std::shared_ptr<const Core::LinAlg::Vector<double>> arteryinc =
      extractor()->extract_vector(*iterinc_, struct_offset_ + 2);

  // build also norms for artery
  normrhsart_ = calculate_vector_norm(vectornormfres_, *arteryrhs);
  normincart_ = calculate_vector_norm(vectornorminc_, *arteryinc);
  arterypressnorm_ = calculate_vector_norm(
      vectornorminc_, (*poro_field()->fluid_field()->art_net_tim_int()->pressurenp()));

  std::shared_ptr<const Core::LinAlg::Vector<double>> arteryscarhs =
      extractor()->extract_vector(*rhs_, struct_offset_ + 3);
  std::shared_ptr<const Core::LinAlg::Vector<double>> arteryscainc =
      extractor()->extract_vector(*iterinc_, struct_offset_ + 3);

  // build also norms for artery
  normrhsartsca_ = calculate_vector_norm(vectornormfres_, *arteryscarhs);
  normincartsca_ = calculate_vector_norm(vectornorminc_, *arteryscainc);
  arteryscanorm_ =
      calculate_vector_norm(vectornorminc_, *(scatramsht_->art_scatra_field()->phinp()));

  // call base class
  PoroPressureBased::PoroMultiPhaseScaTraMonolithicTwoWay::build_convergence_norms();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraMonolithicTwoWayArteryCoupling::evaluate_scatra()
{
  PoroMultiPhaseScaTraMonolithicTwoWay::evaluate_scatra();
  scatramsht_->setup_system(
      scatra_algo()->scatra_field()->system_matrix(), scatra_algo()->scatra_field()->residual());
}

/*-----------------------------------------------------------------------/
/-----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraMonolithicTwoWayArteryCoupling::build_combined_dbc_map()
{
  PoroMultiPhaseScaTraMonolithicTwoWay::build_combined_dbc_map();

  const std::shared_ptr<const Core::LinAlg::Map> artscatracondmap =
      scatramsht_->art_scatra_field()->dirich_maps()->cond_map();

  combinedDBCMap_ = Core::LinAlg::merge_map(combinedDBCMap_, artscatracondmap, false);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::SparseMatrix> PoroPressureBased::
    PoroMultiPhaseScaTraMonolithicTwoWayArteryCoupling::artery_scatra_artery_coupling_matrix()
{
  std::shared_ptr<Core::LinAlg::SparseMatrix> sparse =
      std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(k_asa_);
  if (sparse == nullptr) FOUR_C_THROW("cast to Core::LinAlg::SparseMatrix failed!");

  return sparse;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraMonolithicTwoWayArteryCoupling::
    apply_artery_scatra_artery_coupl_matrix(
        std::shared_ptr<Core::LinAlg::SparseOperator> k_asa  //!< off-diagonal tangent matrix term
    )
{
  // create the parameters for the discretization
  Teuchos::ParameterList sparams_artery;

  k_asa->zero();

  Core::Utils::add_enum_class_to_parameter_list<ScaTra::Action>(
      "action", ScaTra::Action::calc_scatra_mono_odblock_fluid, sparams_artery);
  // other parameters that might be needed by the elements
  sparams_artery.set("delta time", dt());
  sparams_artery.set("total time", time());

  scatramsht_->art_scatra_field()->discretization()->clear_state();
  scatramsht_->art_scatra_field()->discretization()->set_state(
      0, "phinp", *scatramsht_->art_scatra_field()->phinp());
  scatramsht_->art_scatra_field()->discretization()->set_state(
      0, "hist", *scatramsht_->art_scatra_field()->hist());
  scatramsht_->art_scatra_field()->discretization()->set_state(
      2, "one_d_artery_pressure", *poro_field()->fluid_field()->art_net_tim_int()->pressurenp());

  // build specific assemble strategy for mechanical-fluid system matrix
  Core::FE::AssembleStrategy artscatrastrategy_artery(0,  // scatradofset for row
      2,                                                  // arterydofset for column
      k_asa,                                              // scatra-artery coupling matrix
      nullptr, nullptr, nullptr, nullptr);

  scatramsht_->art_scatra_field()->discretization()->evaluate(
      sparams_artery, artscatrastrategy_artery);

  // complete
  k_asa->complete(poro_field()->fluid_field()->art_net_tim_int()->system_matrix()->range_map(),
      scatramsht_->art_scatra_field()->system_matrix()->range_map());

  scatramsht_->art_scatra_field()->discretization()->clear_state();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraMonolithicTwoWayArteryCoupling::
    build_block_null_spaces()
{
  // base class -> structure, porofluid, scatra
  PoroMultiPhaseScaTraMonolithicTwoWay::build_block_null_spaces();

  // artery
  poro_field()->build_artery_block_null_space(solver_, struct_offset_ + 3);

  // artery-scatra
  Teuchos::ParameterList& blocksmootherparams5 =
      solver_->params().sublist("Inverse" + std::to_string(struct_offset_ + 4));

  // build null space of complete discretization
  Core::LinearSolver::Parameters::compute_solver_parameters(
      *scatramsht_->art_scatra_field()->discretization(), blocksmootherparams5);

  // fix the null space if some DOFs are condensed out
  Core::LinearSolver::Parameters::fix_null_space("ArteryScatra",
      *(scatramsht_->art_scatra_field()->discretization()->dof_row_map(0)),
      *(scatramsht_->art_scatra_dof_row_map()), blocksmootherparams5);
}

FOUR_C_NAMESPACE_CLOSE

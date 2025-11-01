// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_porofluid_pressure_based_elast_scatra_monolithic.hpp"

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
#include "4C_porofluid_pressure_based_elast_base.hpp"
#include "4C_scatra_ele_action.hpp"
#include "4C_scatra_timint_implicit.hpp"
#include "4C_scatra_timint_meshtying_strategy_artery.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_parameter_list.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
PoroPressureBased::PorofluidElastScatraMonolithicAlgorithm::PorofluidElastScatraMonolithicAlgorithm(
    MPI_Comm comm, const Teuchos::ParameterList& globaltimeparams)
    : PorofluidElastScatraBaseAlgorithm(comm, globaltimeparams),
      iter_tol_inc_(0.0),
      iter_tol_res_(0.0),
      iter_max_(0),
      iter_min_(1),
      iter_number_(0),
      blockrowdofmap_(nullptr),
      equilibration_(nullptr),
      equilibration_method_(Core::LinAlg::EquilibrationMethod::none),
      solveradaptolbetter_(0.0),
      solveradapttol_(false),
      solve_structure_(true),
      struct_offset_(1),
      tol_inc_(0.0),
      tol_res_(0.0),
      tol_inc_structure_(0.0),
      tol_res_structure_(0.0),
      tol_inc_porofluid_(0.0),
      tol_res_porofluid_(0.0),
      tol_inc_scatra_(0.0),
      tol_res_scatra_(0.0),
      norm_rhs_(0.0),
      norm_rhs_porofluid_(0.0),
      norm_inc_porofluid_(0.0),
      norm_rhs_structure_(0.0),
      norm_inc_structure_(0.0),
      norm_rhs_scatra_(0.0),
      norm_inc_scatra_(0.0),
      norm_rhs_artery_(0.0),
      norm_inc_artery_(0.0),
      norm_artery_pressure_(0.0),
      norm_rhs_artery_scatra_(0.0),
      norm_inc_artery_scatra_(0.0),
      norm_artery_scatra_(0.0),
      max_inc_(0.0),
      max_res_(0.0),
      vector_norm_res_(VectorNorm::undefined),
      vector_norm_inc_(VectorNorm::undefined),
      timernewton_("", true),
      dtsolve_(0.0),
      dtele_(0.0),
      fdcheck_(false)
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraMonolithicAlgorithm::init(
    const Teuchos::ParameterList& globaltimeparams, const Teuchos::ParameterList& algoparams,
    const Teuchos::ParameterList& poroparams, const Teuchos::ParameterList& structparams,
    const Teuchos::ParameterList& fluidparams, const Teuchos::ParameterList& scatraparams,
    const std::string& struct_disname, const std::string& fluid_disname,
    const std::string& scatra_disname, bool isale, int nds_disp, int nds_vel, int nds_solidpressure,
    int ndsporofluid_scatra, const std::map<int, std::set<int>>* nearby_ele_pairs)
{
  // call base class
  PoroPressureBased::PorofluidElastScatraBaseAlgorithm::init(globaltimeparams, algoparams,
      poroparams, structparams, fluidparams, scatraparams, struct_disname, fluid_disname,
      scatra_disname, isale, nds_disp, nds_vel, nds_solidpressure, ndsporofluid_scatra,
      nearby_ele_pairs);

  // read input variables
  iter_max_ = algoparams.sublist("nonlinear_solver").get<int>("maximum_number_of_iterations");
  iter_tol_inc_ = algoparams.sublist("monolithic")
                      .sublist("nonlinear_solver")
                      .sublist("increment")
                      .get<double>("global_tolerance");
  iter_tol_res_ = algoparams.sublist("monolithic")
                      .sublist("nonlinear_solver")
                      .sublist("residual")
                      .get<double>("global_tolerance");

  blockrowdofmap_ = std::make_shared<Core::LinAlg::MultiMapExtractor>();

  fdcheck_ = algoparams.sublist("monolithic").get<bool>("fd_check");

  equilibration_method_ = Teuchos::getIntegralValue<Core::LinAlg::EquilibrationMethod>(
      algoparams.sublist("monolithic").sublist("nonlinear_solver"), "equilibration");

  solveradaptolbetter_ = algoparams.sublist("monolithic")
                             .sublist("nonlinear_solver")
                             .sublist("convergence_criteria_adaptivity")
                             .get<double>("nonlinear_to_linear_tolerance_ratio");
  solveradapttol_ = algoparams.sublist("monolithic")
                        .sublist("nonlinear_solver")
                        .sublist("convergence_criteria_adaptivity")
                        .get<bool>("active");

  // do we also solve the structure, this is helpful in case of fluid-scatra coupling without mesh
  // deformation
  solve_structure_ = poroparams.get<bool>("solve_structure");
  if (!solve_structure_) struct_offset_ = 0;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraMonolithicAlgorithm::setup_system()
{
  // setup the poro subsystem first
  porofluid_elast_algo()->setup_system();

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
  k_pfs_ = std::make_shared<Core::LinAlg::SparseMatrix>(
      *(porofluid_elast_algo()->porofluid_dof_row_map()),
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
void PoroPressureBased::PorofluidElastScatraMonolithicAlgorithm::setup_maps()
{
  // create combined map
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> vecSpaces;

  if (solve_structure_)
  {
    vecSpaces.push_back(porofluid_elast_algo()->structure_dof_row_map());
    vecSpaces.push_back(porofluid_elast_algo()->porofluid_dof_row_map());
    const Core::LinAlg::Map* dofrowmapscatra =
        (scatra_algo()->scatra_field()->discretization())->dof_row_map(0);
    vecSpaces.push_back(Core::Utils::shared_ptr_from_ref(*dofrowmapscatra));

    if (vecSpaces[0]->num_global_elements() == 0)
      FOUR_C_THROW("No poro structure equation. Panic.");
    if (vecSpaces[1]->num_global_elements() == 0) FOUR_C_THROW("No poro fluid equation. Panic.");
    if (vecSpaces[2]->num_global_elements() == 0) FOUR_C_THROW("No scatra equation. Panic.");
  }
  else
  {
    vecSpaces.push_back(porofluid_elast_algo()->porofluid_dof_row_map());
    const Core::LinAlg::Map* dofrowmapscatra =
        (scatra_algo()->scatra_field()->discretization())->dof_row_map(0);
    vecSpaces.push_back(Core::Utils::shared_ptr_from_ref(*dofrowmapscatra));

    if (vecSpaces[0]->num_global_elements() == 0) FOUR_C_THROW("No poro fluid equation. Panic.");
    if (vecSpaces[1]->num_global_elements() == 0) FOUR_C_THROW("No scatra equation. Panic.");
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
void PoroPressureBased::PorofluidElastScatraMonolithicAlgorithm::build_combined_dbc_map()
{
  // Combined DBC map of poromultielast-problem
  const std::shared_ptr<const Core::LinAlg::Map> porocondmap =
      porofluid_elast_algo()->combined_dbc_map();
  const std::shared_ptr<const Core::LinAlg::Map> scatracondmap =
      scatra_algo()->scatra_field()->dirich_maps()->cond_map();
  combinedDBCMap_ = Core::LinAlg::merge_map(porocondmap, scatracondmap, false);

  return;
}

/*-----------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraMonolithicAlgorithm::build_block_null_spaces()
{
  // Build block null spaces of structure and fluid-field
  if (solve_structure_) porofluid_elast_algo()->build_block_null_spaces(solver_);
  // only fluid
  else
  {
    Teuchos::ParameterList& blocksmootherparams1 = solver_->params().sublist("Inverse1");
    Core::LinearSolver::Parameters::compute_solver_parameters(
        *porofluid_elast_algo()->porofluid_algo()->discretization(), blocksmootherparams1);
  }

  Teuchos::ParameterList& blocksmootherparams =
      solver_->params().sublist("Inverse" + std::to_string(struct_offset_ + 2));

  Core::LinearSolver::Parameters::compute_solver_parameters(
      *scatra_algo()->scatra_field()->discretization(), blocksmootherparams);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraMonolithicAlgorithm::setup_solver()
{
  //  solver
  // create a linear solver
  // get dynamic section of poroelasticity
  const Teuchos::ParameterList& poromultscatradyn =
      Global::Problem::instance()->poro_multi_phase_scatra_dynamic_params();
  // get the solver number used for linear poroelasticity solver
  const int linsolvernumber = poromultscatradyn.sublist("monolithic")
                                  .sublist("nonlinear_solver")
                                  .get<int>("linear_solver_id");
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

  vector_norm_res_ = Teuchos::getIntegralValue<PoroPressureBased::VectorNorm>(
      poromultscatradyn.sublist("monolithic").sublist("nonlinear_solver").sublist("residual"),
      "vector_norm");
  vector_norm_inc_ = Teuchos::getIntegralValue<PoroPressureBased::VectorNorm>(
      poromultscatradyn.sublist("monolithic").sublist("nonlinear_solver").sublist("increment"),
      "vector_norm");
}

/*-----------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraMonolithicAlgorithm::create_linear_solver(
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
void PoroPressureBased::PorofluidElastScatraMonolithicAlgorithm::time_step()
{
  // Prepare stuff
  setup_newton();
  print_header();

  // Evaluate
  evaluate(iter_inc_);

  // Newton-Loop
  while ((not converged() and iter_number_ < iter_max_) or (iter_number_ < iter_min_))
  {
    // increment number of iteration
    iter_number_++;

    // Solve
    linear_solve();
    solver_->reset_tolerance();

    // Build Convergence Norms
    build_convergence_norms();

    // Evaluate
    if (not converged())
    {
      evaluate(iter_inc_);
      // perform FD Check of monolithic system matrix
      if (fdcheck_) poro_multi_phase_scatra_fd_check();
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
void PoroPressureBased::PorofluidElastScatraMonolithicAlgorithm::evaluate(
    std::shared_ptr<const Core::LinAlg::Vector<double>> iterinc)
{
  TEUCHOS_FUNC_TIME_MONITOR(
      "PoroMultiPhaseScaTra::PorofluidElastScatraMonolithicAlgorithm::Evaluate");

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
  porofluid_elast_algo()->evaluate(porostructinc, porofluidinc, iter_number_ == 0);

  // (4) set fluid and structure solution on scatra field
  set_porofluid_elast_solution();

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
  dtele_ = Core::Communication::max_all(mydtele, get_comm());
  // *********** time measurement ***********
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraMonolithicAlgorithm::setup_system_matrix()
{
  // set loma block matrix to zero
  systemmatrix_->zero();

  //----------------------------------------------------------------------
  // 1st diagonal block (upper left): poro weighting - poro solution
  // has dimensions ((ndim+n_phases)*n_nodes)x((ndim+n_phases)*n_nodes)
  //----------------------------------------------------------------------
  // get matrix block
  std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> mat_pp =
      porofluid_elast_algo()->block_system_matrix();

  // uncomplete matrix block (appears to be required in certain cases (locsys+iterative solver))
  mat_pp->un_complete();

  // assign matrix block
  if (solve_structure_)
  {
    systemmatrix_->assign(0, 0, Core::LinAlg::DataAccess::Share, mat_pp->matrix(0, 0));
    systemmatrix_->assign(0, 1, Core::LinAlg::DataAccess::Share, mat_pp->matrix(0, 1));
    systemmatrix_->assign(1, 0, Core::LinAlg::DataAccess::Share, mat_pp->matrix(1, 0));
  }
  systemmatrix_->assign(
      struct_offset_, struct_offset_, Core::LinAlg::DataAccess::Share, mat_pp->matrix(1, 1));

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
      struct_offset_ + 1, struct_offset_ + 1, Core::LinAlg::DataAccess::Share, *mat_ss);

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
  k_pfs->apply_dirichlet(
      *porofluid_elast_algo()->porofluid_algo()->get_dbc_map_extractor()->cond_map(), false);

  // uncomplete matrix block (appears to be required in certain cases)
  // k_pss_->UnComplete();
  k_pfs->un_complete();

  // assign matrix block
  // systemmatrix_->Assign(0,2,Core::LinAlg::DataAccess::Share,*(k_pss_)); --> zero
  systemmatrix_->assign(
      struct_offset_, struct_offset_ + 1, Core::LinAlg::DataAccess::Share, *(k_pfs));

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
  if (solve_structure_) systemmatrix_->assign(2, 0, Core::LinAlg::DataAccess::Share, *(k_sps));
  systemmatrix_->assign(
      struct_offset_ + 1, struct_offset_, Core::LinAlg::DataAccess::Share, *(k_spf));

  // complete block matrix
  systemmatrix_->complete();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::SparseMatrix>
PoroPressureBased::PorofluidElastScatraMonolithicAlgorithm::poro_fluid_scatra_coupling_matrix()
{
  std::shared_ptr<Core::LinAlg::SparseMatrix> sparse =
      std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(k_pfs_);
  if (sparse == nullptr) FOUR_C_THROW("cast to Core::LinAlg::SparseMatrix failed!");

  return sparse;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::SparseMatrix>
PoroPressureBased::PorofluidElastScatraMonolithicAlgorithm::scatra_struct_coupling_matrix()
{
  std::shared_ptr<Core::LinAlg::SparseMatrix> sparse =
      std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(k_sps_);
  if (sparse == nullptr) FOUR_C_THROW("cast to Core::LinAlg::SparseMatrix failed!");

  return sparse;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::SparseMatrix>
PoroPressureBased::PorofluidElastScatraMonolithicAlgorithm::scatra_poro_fluid_coupling_matrix()
{
  std::shared_ptr<Core::LinAlg::SparseMatrix> sparse =
      std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(k_spf_);
  if (sparse == nullptr) FOUR_C_THROW("cast to Core::LinAlg::SparseMatrix failed!");

  return sparse;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraMonolithicAlgorithm::evaluate_scatra()
{
  scatra_algo()->scatra_field()->prepare_linear_solve();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraMonolithicAlgorithm::
    apply_poro_fluid_scatra_coupl_matrix(
        std::shared_ptr<Core::LinAlg::SparseOperator> k_pfs  //!< off-diagonal tangent matrix term
    )
{
  // reset
  k_pfs->zero();
  // evaluate
  porofluid_elast_algo()->porofluid_algo()->assemble_fluid_scatra_coupling_mat(k_pfs);
  // complete
  k_pfs->complete(scatra_algo()->scatra_field()->system_matrix()->range_map(),
      porofluid_elast_algo()->porofluid_algo()->system_matrix()->range_map());
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraMonolithicAlgorithm::apply_scatra_struct_coupl_matrix(
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
  k_sps->complete(porofluid_elast_algo()->structure_algo()->system_matrix()->range_map(),
      scatra_algo()->scatra_field()->system_matrix()->range_map());

  scatra_algo()->scatra_field()->discretization()->clear_state();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraMonolithicAlgorithm::
    apply_scatra_poro_fluid_coupl_matrix(
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
  k_spf->complete(porofluid_elast_algo()->porofluid_algo()->system_matrix()->range_map(),
      scatra_algo()->scatra_field()->system_matrix()->range_map());

  scatra_algo()->scatra_field()->discretization()->clear_state();
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraMonolithicAlgorithm::update_fields_after_convergence()
{
  // displacement, fluid variable and scatra variable incremental vector
  std::shared_ptr<const Core::LinAlg::Vector<double>> porostructinc;
  std::shared_ptr<const Core::LinAlg::Vector<double>> porofluidinc;
  std::shared_ptr<const Core::LinAlg::Vector<double>> scatrainc;
  extract_field_vectors(iter_inc_, porostructinc, porofluidinc, scatrainc);

  // update ScaTra field
  update_scatra(scatrainc);

  // update structure and fluid field
  porofluid_elast_algo()->update_fields_after_convergence(porostructinc, porofluidinc);
}
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraMonolithicAlgorithm::update_scatra(
    std::shared_ptr<const Core::LinAlg::Vector<double>> scatrainc)
{
  scatra_algo()->scatra_field()->update_iter(*scatrainc);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraMonolithicAlgorithm::setup_rhs()
{
  // create full monolithic rhs vector
  if (rhs_ == nullptr) rhs_ = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map(), true);

  // note: rhs of fluid-structure system already setup in evaluate call

  // fill the Poroelasticity rhs vector rhs_ with the single field rhss
  setup_vector(*rhs_, porofluid_elast_algo()->rhs(), scatra_algo()->scatra_field()->residual());
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraMonolithicAlgorithm::setup_vector(
    Core::LinAlg::Vector<double>& f, std::shared_ptr<const Core::LinAlg::Vector<double>> pv,
    std::shared_ptr<const Core::LinAlg::Vector<double>> sv)
{
  // extract dofs of the two fields
  // and put the poro/scatra field vector into the global vector f
  // noticing the block number

  //  std::shared_ptr<const Core::LinAlg::Vector<double>> psx;
  //  std::shared_ptr<const Core::LinAlg::Vector<double>> pfx;

  if (solve_structure_)
    extractor()->insert_vector(
        *(porofluid_elast_algo()->extractor()->extract_vector(*pv, 0)), 0, f);
  extractor()->insert_vector(
      *(porofluid_elast_algo()->extractor()->extract_vector(*pv, 1)), struct_offset_, f);
  extractor()->insert_vector(*sv, struct_offset_ + 1, f);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraMonolithicAlgorithm::extract_field_vectors(
    std::shared_ptr<const Core::LinAlg::Vector<double>> x,
    std::shared_ptr<const Core::LinAlg::Vector<double>>& stx,
    std::shared_ptr<const Core::LinAlg::Vector<double>>& flx,
    std::shared_ptr<const Core::LinAlg::Vector<double>>& scx)
{
  TEUCHOS_FUNC_TIME_MONITOR(
      "PoroMultiPhaseScaTra::PorofluidElastScatraMonolithicAlgorithm::extract_field_vectors");

  // process structure unknowns of the first field
  if (solve_structure_)
    stx = extractor()->extract_vector(*x, 0);
  else
    stx = std::make_shared<Core::LinAlg::Vector<double>>(
        *porofluid_elast_algo()->structure_dof_row_map(), true);

  // process fluid unknowns of the second field
  flx = extractor()->extract_vector(*x, struct_offset_);

  // process scatra unknowns of the third field
  scx = extractor()->extract_vector(*x, struct_offset_ + 1);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraMonolithicAlgorithm::extract_3d_field_vectors(
    std::shared_ptr<const Core::LinAlg::Vector<double>> x,
    std::shared_ptr<const Core::LinAlg::Vector<double>>& stx,
    std::shared_ptr<const Core::LinAlg::Vector<double>>& flx,
    std::shared_ptr<const Core::LinAlg::Vector<double>>& scx)
{
  PoroPressureBased::PorofluidElastScatraMonolithicAlgorithm::extract_field_vectors(
      x, stx, flx, scx);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraMonolithicAlgorithm::linear_solve()
{
  // reset timer
  timernewton_.reset();
  // *********** time measurement ***********
  double dtcpu = timernewton_.wallTime();
  // *********** time measurement ***********
  Core::LinAlg::SolverParams solver_params;
  if (solveradapttol_ and (iter_number_ > 1))
  {
    solver_params.nonlin_tolerance = iter_tol_res_;
    solver_params.nonlin_residual = std::max(max_inc_, max_res_);
    solver_params.lin_tol_better = solveradaptolbetter_;
  }
  iter_inc_->put_scalar(0.0);  // Useful? depends on solver and more

  // equilibrate global system of equations if necessary
  equilibration_->equilibrate_system(systemmatrix_, rhs_, blockrowdofmap_);

  // standard solver call
  // system is ready to solve since Dirichlet Boundary conditions have been applied in
  // setup_system_matrix or Evaluate
  solver_params.refactor = true;
  solver_params.reset = iter_number_ == 1;
  solver_->solve(systemmatrix_, iter_inc_, rhs_, solver_params);

  equilibration_->unequilibrate_increment(iter_inc_);

  // *********** time measurement ***********
  double mydtsolve = timernewton_.wallTime() - dtcpu;
  dtsolve_ = Core::Communication::max_all(mydtsolve, get_comm());
  // *********** time measurement ***********
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool PoroPressureBased::PorofluidElastScatraMonolithicAlgorithm::converged()
{
  return (norm_inc_porofluid_ < iter_tol_inc_ && norm_inc_structure_ < iter_tol_inc_ &&
          norm_inc_scatra_ < iter_tol_inc_ && norm_inc_artery_ < iter_tol_inc_ &&
          norm_inc_artery_scatra_ < iter_tol_inc_ && norm_rhs_ < iter_tol_res_ &&
          norm_rhs_porofluid_ < iter_tol_res_ && norm_rhs_structure_ < iter_tol_res_ &&
          norm_rhs_scatra_ < iter_tol_res_ && norm_rhs_artery_ < iter_tol_res_ &&
          norm_rhs_artery_scatra_ < iter_tol_res_);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraMonolithicAlgorithm::build_convergence_norms()
{
  //------------------------------------------------------------ build residual force norms
  norm_rhs_ = calculate_vector_norm(vector_norm_res_, *rhs_);
  std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_structure;
  std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_porofluid;
  std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_scatra;

  // get structure, porofluid and scatra RHS
  extract_3d_field_vectors(rhs_, rhs_structure, rhs_porofluid, rhs_scatra);

  // build norms for structure, porofluid and scatra RHS
  norm_rhs_structure_ = calculate_vector_norm(vector_norm_res_, *rhs_structure);
  norm_rhs_porofluid_ = calculate_vector_norm(vector_norm_res_, *rhs_porofluid);
  norm_rhs_scatra_ = calculate_vector_norm(vector_norm_res_, *rhs_scatra);

  // build residual increment norms
  // structure, porofluid and scatra incremental vector
  std::shared_ptr<const Core::LinAlg::Vector<double>> inc_structure;
  std::shared_ptr<const Core::LinAlg::Vector<double>> inc_porofluid;
  std::shared_ptr<const Core::LinAlg::Vector<double>> inc_scatra;

  // get structure, porofluid and scatra increment
  extract_3d_field_vectors(iter_inc_, inc_structure, inc_porofluid, inc_scatra);

  // build norms for structure, porofluid and scatra increment
  norm_inc_structure_ = calculate_vector_norm(vector_norm_inc_, *inc_structure);
  norm_inc_porofluid_ = calculate_vector_norm(vector_norm_inc_, *inc_porofluid);
  norm_inc_scatra_ = calculate_vector_norm(vector_norm_inc_, *inc_scatra);

  double norm_structure =
      calculate_vector_norm(vector_norm_inc_, *porofluid_elast_algo()->structure_algo()->dispnp());
  double norm_porofluid =
      calculate_vector_norm(vector_norm_inc_, *porofluid_elast_algo()->porofluid_algo()->phinp());
  double norm_scatra =
      calculate_vector_norm(vector_norm_inc_, *scatra_algo()->scatra_field()->phinp());

  // take care of very small norms
  if (norm_structure < 1.0e-6) norm_structure = 1.0;
  if (norm_porofluid < 1.0e-6) norm_porofluid = 1.0;
  if (norm_scatra < 1.0e-6) norm_scatra = 1.0;
  if (norm_artery_pressure_ < 1.0e-6) norm_artery_pressure_ = 1.0;
  if (norm_artery_scatra_ < 1.0e-6) norm_artery_scatra_ = 1.0;

  // build relative increment norm
  norm_inc_structure_ /= norm_structure;
  norm_inc_porofluid_ /= norm_porofluid;
  norm_inc_scatra_ /= norm_scatra;
  norm_inc_artery_ /= norm_artery_pressure_;
  norm_inc_artery_scatra_ /= norm_artery_scatra_;

  // build the maximum value of the residuals and increments
  max_inc_ = std::max({norm_inc_porofluid_, norm_inc_structure_, norm_inc_scatra_, norm_inc_artery_,
      norm_inc_artery_scatra_});
  max_res_ = std::max({norm_rhs_, norm_rhs_porofluid_, norm_rhs_structure_, norm_rhs_scatra_,
      norm_rhs_artery_, norm_rhs_artery_scatra_});
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraMonolithicAlgorithm::setup_newton()
{
  // initialise equilibrium loop and norms
  iter_number_ = 0;
  norm_rhs_ = 0.0;
  norm_rhs_porofluid_ = 0.0;
  norm_inc_porofluid_ = 0.0;
  norm_rhs_structure_ = 0.0;
  norm_inc_structure_ = 0.0;
  norm_rhs_scatra_ = 0.0;
  norm_inc_scatra_ = 0.0;
  tol_inc_ = 0.0;
  tol_res_ = 0.0;
  tol_inc_structure_ = 0.0;
  tol_res_structure_ = 0.0;
  tol_inc_porofluid_ = 0.0;
  tol_res_porofluid_ = 0.0;
  tol_inc_scatra_ = 0.0;
  tol_res_scatra_ = 0.0;
  norm_rhs_artery_ = 0.0;
  norm_inc_artery_ = 0.0;
  norm_artery_pressure_ = 0.0;
  norm_rhs_artery_scatra_ = 0.0;
  norm_inc_artery_scatra_ = 0.0;
  norm_artery_scatra_ = 0.0;
  max_inc_ = 0.0;
  max_res_ = 0.0;

  // incremental solution vector with length of all dofs
  if (iter_inc_ == nullptr)
    iter_inc_ = Core::LinAlg::create_vector(*dof_row_map(), true);
  else
    iter_inc_->put_scalar(0.0);

  // a zero vector of full length
  if (zeros_ == nullptr)
    zeros_ = Core::LinAlg::create_vector(*dof_row_map(), true);
  else
    zeros_->put_scalar(0.0);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraMonolithicAlgorithm::newton_output()
{
  // print the incremental based convergence check to the screen
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    if (iter_number_ == 1)
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
        iter_number_, iter_max_, norm_inc_porofluid_, norm_inc_structure_, norm_inc_scatra_,
        norm_inc_artery_, norm_inc_artery_scatra_, norm_rhs_, dtele_);
    printf("\n");
    printf(
        "+--------------+-------------+-------------+--------------+------------+-----"
        "-------+-----------------+\n");
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraMonolithicAlgorithm::newton_error_check()
{
  // print the incremental based convergence check to the screen
  if (converged())  // norminc_ < iter_tol_inc_ && norm_rhs_ < iter_tol_inc_ && norm_inc_porofluid_
                    // < iter_tol_inc_ && norm_inc_structure_ < iter_tol_inc_
  {
    if (Core::Communication::my_mpi_rank(get_comm()) == 0)
    {
      printf(
          "|  Monolithic iteration loop converged after iteration %3d/%3d !                        "
          "              |\n",
          iter_number_, iter_max_);
      printf(
          "|  Quantity           [norm]:                 TOL                                       "
          "              |\n");
      printf(
          "|  Max. rel. increment [%3s]:  %10.3E  < %10.3E                                        "
          "       |\n",
          EnumTools::enum_name(vector_norm_inc_).data(), max_inc_, iter_tol_inc_);
      printf(
          "|  Maximum    residual [%3s]:  %10.3E  < %10.3E                                        "
          "       |\n",
          EnumTools::enum_name(vector_norm_res_).data(), max_res_, iter_tol_res_);
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
          iter_max_);
      printf(
          "|  Quantity           [norm]:                 TOL                                       "
          "       |\n");
      printf(
          "|  Max. rel. increment [%3s]:  %10.3E    %10.3E                                        "
          "|\n",
          EnumTools::enum_name(vector_norm_inc_).data(), max_inc_, iter_tol_inc_);
      printf(
          "|  Maximum    residual [%3s]:  %10.3E    %10.3E                                        "
          "|\n",
          EnumTools::enum_name(vector_norm_res_).data(), max_res_, iter_tol_res_);
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
PoroPressureBased::PorofluidElastScatraMonolithicAlgorithm::dof_row_map()
{
  return blockrowdofmap_->full_map();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraMonolithicAlgorithm::print_header()
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
void PoroPressureBased::PorofluidElastScatraMonolithicAlgorithm::print_structure_disabled_info()
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
void PoroPressureBased::PorofluidElastScatraMonolithicAlgorithm::poro_multi_phase_scatra_fd_check()
{
  std::cout << "\n******************finite difference check***************" << std::endl;

  int dof_struct = (porofluid_elast_algo()->structure_algo()->dof_row_map()->num_global_elements());
  int dof_fluid = (porofluid_elast_algo()->porofluid_algo()->dof_row_map()->num_global_elements());
  int dof_scatra = (scatra_algo()->scatra_field()->dof_row_map()->num_global_elements());

  std::cout << "structure field has " << dof_struct << " DOFs" << std::endl;
  std::cout << "fluid field has " << dof_fluid << " DOFs" << std::endl;
  std::cout << "scatra field has " << dof_scatra << " DOFs" << std::endl;
  if (artery_coupling_)
  {
    int dof_artery =
        (porofluid_elast_algo()->porofluid_algo()->artery_dof_row_map()->num_global_elements());
    int dof_artscatra =
        (scatra_meshtying_strategy_->art_scatra_field()->dof_row_map()->num_global_elements());
    std::cout << "artery field has " << dof_artery << " DOFs" << std::endl;
    std::cout << "artery-scatra field has " << dof_artscatra << " DOFs" << std::endl;

    std::cout << "\n\n============================================================\n"
                 "WARNING: THIS FD CHECK DOES NOT WORK FOR NODE BASED COUPLING\n"
                 "============================================================\n\n";
  }

  std::shared_ptr<Core::LinAlg::Vector<double>> iter_inc;
  iter_inc = Core::LinAlg::create_vector(*dof_row_map(), true);

  const int dofs = iter_inc->global_length();
  std::cout << "in total " << dofs << " DOFs" << std::endl;
  const double delta = 1e-8;

  iter_inc->put_scalar(0.0);

  iter_inc->replace_global_value(0, delta);

  Core::LinAlg::SparseMatrix stiff_approx(*dof_row_map(), 81);

  Core::LinAlg::Vector<double> rhs_old(*dof_row_map(), true);
  rhs_old.update(1.0, *rhs_, 0.0);
  Core::LinAlg::Vector<double> rhs_copy(*dof_row_map(), true);

  std::shared_ptr<Core::LinAlg::SparseMatrix> sparse = systemmatrix_->merge();
  Core::LinAlg::SparseMatrix sparse_copy(*sparse, Core::LinAlg::DataAccess::Copy);


  const int row_id = -1;
  const int column_id = -1;
  for (int i = 0; i < dofs; ++i)
  {
    if (combined_dbc_map()->my_gid(i))
    {
      iter_inc->replace_global_value(i, 0.0);
    }

    if (i == column_id)
      std::cout << "\n******************" << column_id + 1 << "th column ***************"
                << std::endl;

    evaluate(iter_inc);
    setup_rhs();

    rhs_copy.update(1.0, *rhs_, 0.0);

    iter_inc_->put_scalar(0.0);  // Useful? depends on solver and more
    Core::LinAlg::apply_dirichlet_to_system(
        sparse_copy, *iter_inc_, rhs_copy, *zeros_, *combined_dbc_map());


    if (i == column_id)
    {
      std::cout << "rhs_: " << rhs_copy.local_values_as_span()[row_id] << std::endl;
      std::cout << "rhs_old: " << rhs_old.local_values_as_span()[row_id] << std::endl;
    }

    rhs_copy.update(-1.0, rhs_old, 1.0);
    rhs_copy.scale(-1.0 / delta);

    int* index = &i;
    for (int j = 0; j < dofs; ++j)
    {
      double value = rhs_copy.local_values_as_span()[j];
      stiff_approx.insert_global_values(j, 1, &value, index);

      if ((j == row_id) and (i == column_id))
      {
        std::cout << "\n******************" << row_id + 1 << "th row ***************" << std::endl;
        std::cout << "\n******************" << row_id + 1 << "the row end ***************"
                  << std::endl;
      }
    }

    if (not combined_dbc_map()->my_gid(i)) iter_inc->replace_global_value(i, -delta);

    if (i - 1 >= 0 && i - 1 < dofs && iter_inc->get_map().my_gid(i - 1))
    {
      iter_inc->replace_global_value(i - 1, 0.0);
    }

    if (i != dofs - 1) iter_inc->replace_global_value(i + 1, delta);

    if (i == column_id)
      std::cout << "\n******************" << column_id + 1 << "the column end ***************"
                << std::endl;
  }

  evaluate(iter_inc);
  setup_rhs();

  stiff_approx.complete();

  auto stiff_approx_sparse = std::make_shared<Core::LinAlg::SparseMatrix>(stiff_approx);

  stiff_approx_sparse->add(sparse_copy, false, -1.0, 1.0);

  Core::LinAlg::SparseMatrix sparse_crs(sparse_copy);
  std::shared_ptr<Core::LinAlg::SparseMatrix> error_crs = stiff_approx_sparse;

  error_crs->complete();
  sparse_crs.complete();

  bool success = true;
  double error_max_rel = 0.0;
  double error_max_abs = 0.0;
  for (int i = 0; i < dofs; ++i)
  {
    if (not combined_dbc_map()->my_gid(i))
    {
      for (int j = 0; j < dofs; ++j)
      {
        if (not combined_dbc_map()->my_gid(j))
        {
          double stiff_approx_ij = 0.0;
          double sparse_ij = 0.0;
          double error_ij = 0.0;

          {
            // get error_crs entry ij
            int error_num_entries;
            int error_length = error_crs->num_global_entries(i);
            std::vector<double> error_values(error_length);
            std::vector<int> error_indices(error_length);

            error_crs->extract_global_row_copy(
                i, error_length, error_num_entries, error_values.data(), error_indices.data());
            for (int k = 0; k < error_length; ++k)
            {
              if (error_indices[k] == j)
              {
                error_ij = error_values[k];
                break;
              }
              else
                error_ij = 0.0;
            }
          }

          // get sparse_ij entry ij
          {
            int sparse_num_entries;
            int sparse_length = sparse_crs.num_global_entries(i);
            std::vector<double> sparse_values(sparse_length);
            std::vector<int> sparse_indices(sparse_length);

            sparse_crs.extract_global_row_copy(
                i, sparse_length, sparse_num_entries, sparse_values.data(), sparse_indices.data());
            for (int k = 0; k < sparse_length; ++k)
            {
              if (sparse_indices[k] == j)
              {
                sparse_ij = sparse_values[k];
                break;
              }
              else
                sparse_ij = 0.0;
            }
          }

          // get stiff_approx entry ij
          {
            int approx_num_entries;
            int approx_length = stiff_approx.num_global_entries(i);
            std::vector<double> approx_values(approx_length);
            std::vector<int> approx_indices(approx_length);

            stiff_approx.extract_global_row_copy(
                i, approx_length, approx_num_entries, approx_values.data(), approx_indices.data());
            for (int k = 0; k < approx_length; ++k)
            {
              if (approx_indices[k] == j)
              {
                stiff_approx_ij = approx_values[k];
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
PoroPressureBased::PorofluidElastScatraMonolithicArteryCouplingAlgorithm::
    PorofluidElastScatraMonolithicArteryCouplingAlgorithm(
        MPI_Comm comm, const Teuchos::ParameterList& globaltimeparams)
    : PorofluidElastScatraMonolithicAlgorithm(comm, globaltimeparams)
{
  blockrowdofmap_artscatra_ = std::make_shared<Core::LinAlg::MultiMapExtractor>();
  blockrowdofmap_artporo_ = std::make_shared<Core::LinAlg::MultiMapExtractor>();
  nodal_coupl_inactive_ = false;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraMonolithicArteryCouplingAlgorithm::setup_system()
{
  PorofluidElastScatraMonolithicAlgorithm::setup_system();

  //! arteryscatra-artery coupling matrix, this matrix has the full map of all coupled + uncoupled
  //! DOFs
  k_asa_ = std::make_shared<Core::LinAlg::SparseMatrix>(
      *(scatra_meshtying_strategy_->art_scatra_field()->discretization()->dof_row_map()), 81, true,
      true);

  //! simple check if nodal coupling active or not, if condensed and un-condensed dofrowmaps have
  //! equal size
  nodal_coupl_inactive_ =
      ((porofluid_elast_algo()->artery_dof_row_map()->num_global_elements() ==
          porofluid_elast_algo()
              ->porofluid_algo()
              ->art_net_tim_int()
              ->discretization()
              ->dof_row_map(0)
              ->num_global_elements())) &&
      (scatra_meshtying_strategy_->art_scatra_dof_row_map()->num_global_elements() ==
          scatra_meshtying_strategy_->art_scatra_field()
              ->discretization()
              ->dof_row_map(0)
              ->num_global_elements());
}
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraMonolithicArteryCouplingAlgorithm::setup_maps()
{
  // create combined map
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> vecSpaces;

  if (solve_structure_)
  {
    vecSpaces.push_back(porofluid_elast_algo()->structure_dof_row_map());
    vecSpaces.push_back(porofluid_elast_algo()->porofluid_dof_row_map());
    const Core::LinAlg::Map* dofrowmapscatra =
        (scatra_algo()->scatra_field()->discretization())->dof_row_map(0);
    vecSpaces.push_back(Core::Utils::shared_ptr_from_ref(*dofrowmapscatra));
    vecSpaces.push_back(porofluid_elast_algo()->artery_dof_row_map());
    vecSpaces.push_back(scatra_meshtying_strategy_->art_scatra_dof_row_map());
    if (vecSpaces[0]->num_global_elements() == 0)
      FOUR_C_THROW("No poro structure equation. Panic.");
    if (vecSpaces[1]->num_global_elements() == 0) FOUR_C_THROW("No poro fluid equation. Panic.");
    if (vecSpaces[2]->num_global_elements() == 0) FOUR_C_THROW("No scatra equation. Panic.");
    if (vecSpaces[3]->num_global_elements() == 0) FOUR_C_THROW("No artery equation. Panic.");
    if (vecSpaces[4]->num_global_elements() == 0) FOUR_C_THROW("No artery scatra equation. Panic.");
  }
  else
  {
    vecSpaces.push_back(porofluid_elast_algo()->porofluid_dof_row_map());
    const Core::LinAlg::Map* dofrowmapscatra =
        (scatra_algo()->scatra_field()->discretization())->dof_row_map(0);
    vecSpaces.push_back(Core::Utils::shared_ptr_from_ref(*dofrowmapscatra));
    vecSpaces.push_back(porofluid_elast_algo()->artery_dof_row_map());
    vecSpaces.push_back(scatra_meshtying_strategy_->art_scatra_dof_row_map());
    if (vecSpaces[0]->num_global_elements() == 0) FOUR_C_THROW("No poro fluid equation. Panic.");
    if (vecSpaces[1]->num_global_elements() == 0) FOUR_C_THROW("No scatra equation. Panic.");
    if (vecSpaces[2]->num_global_elements() == 0) FOUR_C_THROW("No artery equation. Panic.");
    if (vecSpaces[3]->num_global_elements() == 0) FOUR_C_THROW("No artery scatra equation. Panic.");
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
void PoroPressureBased::PorofluidElastScatraMonolithicArteryCouplingAlgorithm::update_scatra(
    std::shared_ptr<const Core::LinAlg::Vector<double>> scatrainc)
{
  scatra_algo()->scatra_field()->update_iter(
      *blockrowdofmap_artscatra_->extract_vector(*scatrainc, 0));
  scatra_meshtying_strategy_->update_art_scatra_iter(scatrainc);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraMonolithicArteryCouplingAlgorithm::
    extract_field_vectors(std::shared_ptr<const Core::LinAlg::Vector<double>> x,
        std::shared_ptr<const Core::LinAlg::Vector<double>>& stx,
        std::shared_ptr<const Core::LinAlg::Vector<double>>& flx,
        std::shared_ptr<const Core::LinAlg::Vector<double>>& scx)
{
  TEUCHOS_FUNC_TIME_MONITOR(
      "PoroPressureBased::PorofluidElastScatraMonolithicArteryCouplingAlgorithm::extract_field_"
      "vectors");

  // process structure unknowns of the first field
  if (solve_structure_)
    stx = extractor()->extract_vector(*x, 0);
  else
    stx = std::make_shared<Core::LinAlg::Vector<double>>(
        *porofluid_elast_algo()->structure_dof_row_map(), true);

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
void PoroPressureBased::PorofluidElastScatraMonolithicArteryCouplingAlgorithm::setup_system_matrix()
{
  PorofluidElastScatraMonolithicAlgorithm::setup_system_matrix();

  // --------------------------------------------------------------------------- artery-porofluid
  // get matrix block
  std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> mat_pp =
      porofluid_elast_algo()->block_system_matrix();

  // artery part
  systemmatrix_->assign(struct_offset_ + 2, struct_offset_ + 2, Core::LinAlg::DataAccess::Share,
      mat_pp->matrix(2, 2));
  // artery-porofluid part
  systemmatrix_->assign(
      struct_offset_ + 2, struct_offset_, Core::LinAlg::DataAccess::Share, mat_pp->matrix(2, 1));
  // porofluid-artery part
  systemmatrix_->assign(
      struct_offset_, struct_offset_ + 2, Core::LinAlg::DataAccess::Share, mat_pp->matrix(1, 2));

  // -------------------------------------------------------------------------arteryscatra-scatra
  // arteryscatra part
  systemmatrix_->assign(struct_offset_ + 3, struct_offset_ + 3, Core::LinAlg::DataAccess::Share,
      scatra_meshtying_strategy_->combined_system_matrix()->matrix(1, 1));
  // scatra-arteryscatra part
  systemmatrix_->assign(struct_offset_ + 1, struct_offset_ + 3, Core::LinAlg::DataAccess::Share,
      scatra_meshtying_strategy_->combined_system_matrix()->matrix(0, 1));
  // arteryscatra-scatra part
  systemmatrix_->assign(struct_offset_ + 3, struct_offset_ + 1, Core::LinAlg::DataAccess::Share,
      scatra_meshtying_strategy_->combined_system_matrix()->matrix(1, 0));

  if (nodal_coupl_inactive_)
  {
    // create empty matrix
    std::shared_ptr<Core::LinAlg::SparseMatrix> k_asa = artery_scatra_artery_coupling_matrix();

    // call the scatra-elements and calculate the off-diagonal structure matrix block
    apply_artery_scatra_artery_coupl_matrix(k_asa);

    // apply DBC's also on off-diagonal scatra-fluid coupling block (main-diagonal blocks have
    // already been set, either in poromultielast_monolithic.cpp or in the respective evaluate
    // calls)
    k_asa->apply_dirichlet(
        *scatra_meshtying_strategy_->art_scatra_field()->dirich_maps()->cond_map(), false);

    // arteryscatra-scatra part
    systemmatrix_->assign(
        struct_offset_ + 3, struct_offset_ + 2, Core::LinAlg::DataAccess::Share, *k_asa);
  }

  systemmatrix_->complete();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraMonolithicArteryCouplingAlgorithm::setup_rhs()
{
  // create full monolithic rhs vector
  if (rhs_ == nullptr) rhs_ = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map(), true);

  // structure
  if (solve_structure_)
    extractor()->insert_vector(
        *(porofluid_elast_algo()->extractor()->extract_vector(*porofluid_elast_algo()->rhs(), 0)),
        0, *rhs_);
  // porofluid
  extractor()->insert_vector(
      *(porofluid_elast_algo()->extractor()->extract_vector(*porofluid_elast_algo()->rhs(), 1)),
      struct_offset_, *rhs_);
  // scatra
  extractor()->insert_vector(
      *(blockrowdofmap_artscatra_->extract_vector(*scatra_meshtying_strategy_->combined_rhs(), 0)),
      struct_offset_ + 1, *rhs_);

  // artery
  extractor()->insert_vector(
      *(porofluid_elast_algo()->extractor()->extract_vector(*porofluid_elast_algo()->rhs(), 2)),
      struct_offset_ + 2, *rhs_);
  // arteryscatra
  extractor()->insert_vector(
      *(blockrowdofmap_artscatra_->extract_vector(*scatra_meshtying_strategy_->combined_rhs(), 1)),
      struct_offset_ + 3, *rhs_);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraMonolithicArteryCouplingAlgorithm::
    build_convergence_norms()
{
  std::shared_ptr<const Core::LinAlg::Vector<double>> arteryrhs =
      extractor()->extract_vector(*rhs_, struct_offset_ + 2);
  std::shared_ptr<const Core::LinAlg::Vector<double>> arteryinc =
      extractor()->extract_vector(*iter_inc_, struct_offset_ + 2);

  // build also norms for artery
  norm_rhs_artery_ = calculate_vector_norm(vector_norm_res_, *arteryrhs);
  norm_inc_artery_ = calculate_vector_norm(vector_norm_inc_, *arteryinc);
  norm_artery_pressure_ = calculate_vector_norm(vector_norm_inc_,
      (*porofluid_elast_algo()->porofluid_algo()->art_net_tim_int()->pressurenp()));

  std::shared_ptr<const Core::LinAlg::Vector<double>> arteryscarhs =
      extractor()->extract_vector(*rhs_, struct_offset_ + 3);
  std::shared_ptr<const Core::LinAlg::Vector<double>> arteryscainc =
      extractor()->extract_vector(*iter_inc_, struct_offset_ + 3);

  // build also norms for artery
  norm_rhs_artery_scatra_ = calculate_vector_norm(vector_norm_res_, *arteryscarhs);
  norm_inc_artery_scatra_ = calculate_vector_norm(vector_norm_inc_, *arteryscainc);
  norm_artery_scatra_ = calculate_vector_norm(
      vector_norm_inc_, *(scatra_meshtying_strategy_->art_scatra_field()->phinp()));

  // call base class
  PoroPressureBased::PorofluidElastScatraMonolithicAlgorithm::build_convergence_norms();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraMonolithicArteryCouplingAlgorithm::evaluate_scatra()
{
  PorofluidElastScatraMonolithicAlgorithm::evaluate_scatra();
  scatra_meshtying_strategy_->setup_system(
      scatra_algo()->scatra_field()->system_matrix(), scatra_algo()->scatra_field()->residual());
}

/*-----------------------------------------------------------------------/
/-----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraMonolithicArteryCouplingAlgorithm::
    build_combined_dbc_map()
{
  PorofluidElastScatraMonolithicAlgorithm::build_combined_dbc_map();

  const std::shared_ptr<const Core::LinAlg::Map> artscatracondmap =
      scatra_meshtying_strategy_->art_scatra_field()->dirich_maps()->cond_map();

  combinedDBCMap_ = Core::LinAlg::merge_map(combinedDBCMap_, artscatracondmap, false);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::SparseMatrix> PoroPressureBased::
    PorofluidElastScatraMonolithicArteryCouplingAlgorithm::artery_scatra_artery_coupling_matrix()
{
  std::shared_ptr<Core::LinAlg::SparseMatrix> sparse =
      std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(k_asa_);
  if (sparse == nullptr) FOUR_C_THROW("cast to Core::LinAlg::SparseMatrix failed!");

  return sparse;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraMonolithicArteryCouplingAlgorithm::
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

  scatra_meshtying_strategy_->art_scatra_field()->discretization()->clear_state();
  scatra_meshtying_strategy_->art_scatra_field()->discretization()->set_state(
      0, "phinp", *scatra_meshtying_strategy_->art_scatra_field()->phinp());
  scatra_meshtying_strategy_->art_scatra_field()->discretization()->set_state(
      0, "hist", *scatra_meshtying_strategy_->art_scatra_field()->hist());
  scatra_meshtying_strategy_->art_scatra_field()->discretization()->set_state(2,
      "one_d_artery_pressure",
      *porofluid_elast_algo()->porofluid_algo()->art_net_tim_int()->pressurenp());

  // build specific assemble strategy for mechanical-fluid system matrix
  Core::FE::AssembleStrategy artscatrastrategy_artery(0,  // scatradofset for row
      2,                                                  // arterydofset for column
      k_asa,                                              // scatra-artery coupling matrix
      nullptr, nullptr, nullptr, nullptr);

  scatra_meshtying_strategy_->art_scatra_field()->discretization()->evaluate(
      sparams_artery, artscatrastrategy_artery);

  // complete
  k_asa->complete(
      porofluid_elast_algo()->porofluid_algo()->art_net_tim_int()->system_matrix()->range_map(),
      scatra_meshtying_strategy_->art_scatra_field()->system_matrix()->range_map());

  scatra_meshtying_strategy_->art_scatra_field()->discretization()->clear_state();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraMonolithicArteryCouplingAlgorithm::
    build_block_null_spaces()
{
  // base class -> structure, porofluid, scatra
  PorofluidElastScatraMonolithicAlgorithm::build_block_null_spaces();

  // artery
  porofluid_elast_algo()->build_artery_block_null_space(solver_, struct_offset_ + 3);

  // artery-scatra
  Teuchos::ParameterList& blocksmootherparams5 =
      solver_->params().sublist("Inverse" + std::to_string(struct_offset_ + 4));

  // build null space of complete discretization
  Core::LinearSolver::Parameters::compute_solver_parameters(
      *scatra_meshtying_strategy_->art_scatra_field()->discretization(), blocksmootherparams5);

  // fix the null space if some DOFs are condensed out
  Core::LinearSolver::Parameters::fix_null_space("ArteryScatra",
      *(scatra_meshtying_strategy_->art_scatra_field()->discretization()->dof_row_map(0)),
      *(scatra_meshtying_strategy_->art_scatra_dof_row_map()), blocksmootherparams5);
}

FOUR_C_NAMESPACE_CLOSE

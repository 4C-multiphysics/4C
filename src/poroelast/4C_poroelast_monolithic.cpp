// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_poroelast_monolithic.hpp"

#include "4C_adapter_fld_base_algorithm.hpp"
#include "4C_adapter_fld_poro.hpp"
#include "4C_adapter_str_fpsiwrapper.hpp"
#include "4C_contact_lagrange_strategy_poro.hpp"
#include "4C_contact_meshtying_contact_bridge.hpp"
#include "4C_contact_meshtying_poro_lagrange_strategy.hpp"
#include "4C_contact_nitsche_strategy_poro.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_assemblestrategy.hpp"
#include "4C_fem_general_elements_paramsminimal.hpp"
#include "4C_fluid_ele_action.hpp"
#include "4C_fluid_utils_mapextractor.hpp"
#include "4C_global_data.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_equilibrate.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linear_solver_method.hpp"
#include "4C_linear_solver_method_input.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_linear_solver_method_parameters.hpp"
#include "4C_mortar_manager_base.hpp"
#include "4C_structure_aux.hpp"
#include "4C_structure_new_model_evaluator_contact.hpp"
#include "4C_utils_enum.hpp"

#include <Teuchos_Time.hpp>
#include <Teuchos_TimeMonitor.hpp>

#include <sstream>

FOUR_C_NAMESPACE_OPEN



PoroElast::Monolithic::Monolithic(MPI_Comm comm, const Teuchos::ParameterList& timeparams,
    std::shared_ptr<Core::LinAlg::MapExtractor> porosity_splitter)
    : PoroBase(comm, timeparams, porosity_splitter),
      printscreen_(true),  // ADD INPUT PARAMETER
      printiter_(true),    // ADD INPUT PARAMETER
      zeros_(nullptr),
      blockrowdofmap_(nullptr),
      normtypeinc_(PoroElast::convnorm_undefined),
      normtypefres_(PoroElast::convnorm_undefined),
      combincfres_(PoroElast::bop_undefined),
      vectornormfres_(PoroElast::norm_undefined),
      vectornorminc_(PoroElast::norm_undefined),
      tolinc_(0.0),
      tolfres_(0.0),
      tolinc_struct_(0.0),
      tolfres_struct_(0.0),
      tolinc_velocity_(0.0),
      tolfres_velocity_(0.0),
      tolinc_pressure_(0.0),
      tolfres_pressure_(0.0),
      tolinc_porosity_(0.0),
      tolfres_porosity_(0.0),
      itermax_(0),
      itermin_(0),
      normrhs_(0.0),
      norminc_(0.0),
      normrhsfluidvel_(0.0),
      normincfluidvel_(0.0),
      normrhsfluidpres_(0.0),
      normincfluidpres_(0.0),
      normrhsfluid_(0.0),
      normincfluid_(0.0),
      normrhsstruct_(0.0),
      normincstruct_(0.0),
      normrhsporo_(0.0),
      normincporo_(0.0),
      timer_(std::make_shared<Teuchos::Time>("", false)),
      iter_(-1),
      iterinc_(nullptr),
      directsolve_(true),
      del_(nullptr),
      delhist_(nullptr),
      mu_(0.0),
      equilibration_(nullptr),
      equilibration_method_(Core::LinAlg::EquilibrationMethod::none)
{
  const Teuchos::ParameterList& sdynparams =
      Global::Problem::instance()->structural_dynamic_params();

  // some solver parameters are red form the structure dynamic list (this is not the best way to do
  // it ...)
  solveradapttol_ = (sdynparams.get<bool>("ADAPTCONV"));
  solveradaptolbetter_ = (sdynparams.get<double>("ADAPTCONV_BETTER"));

  const Teuchos::ParameterList& poroparams =
      Global::Problem::instance()->poroelast_dynamic_params();
  equilibration_method_ =
      Teuchos::getIntegralValue<Core::LinAlg::EquilibrationMethod>(poroparams, "EQUILIBRATION");

  strmethodname_ = Teuchos::getIntegralValue<Inpar::Solid::DynamicType>(sdynparams, "DYNAMICTYPE");
  no_penetration_ = false;
  nit_contact_ = false;
  // if inpar is set to nopenetration for contact!!! to be done!
  // clean up as soon as old time integration is unused!
  if (oldstructimint_)
  {
    if (structure_field()->meshtying_contact_bridge() != nullptr)
    {
      if (structure_field()->meshtying_contact_bridge()->have_contact())
      {
        auto* poro_lagrange_strategy = dynamic_cast<CONTACT::LagrangeStrategyPoro*>(
            &structure_field()->meshtying_contact_bridge()->contact_manager()->get_strategy());
        if (poro_lagrange_strategy)
          no_penetration_ = poro_lagrange_strategy->has_poro_no_penetration();
        else
        {
          auto* co_nitsche_strategy = dynamic_cast<CONTACT::NitscheStrategyPoro*>(
              &structure_field()->meshtying_contact_bridge()->contact_manager()->get_strategy());
          if (co_nitsche_strategy)
          {
            nit_contact_ = true;
            no_penetration_ = co_nitsche_strategy->has_poro_no_penetration();
          }
        }
      }
    }
  }
  blockrowdofmap_ = std::make_shared<Core::LinAlg::MultiMapExtractor>();

  // contact no penetration constraint not yet works for non-matching structure and fluid
  // discretizations
  if (no_penetration_ and (not matchinggrid_))
  {
    FOUR_C_THROW(
        "The contact no penetration constraint does not yet work for non-matching "
        "discretizations!");
  }
}

void PoroElast::Monolithic::do_time_step()
{
  // counter and print header
  // predict solution of both field (call the adapter)
  prepare_time_step();

  // Newton-Raphson iteration
  solve();

  // calculate stresses, strains, energies
  constexpr bool force_prepare = false;
  prepare_output(force_prepare);

  // update all single field solvers
  update();

  // write output to screen and files
  output();
}

void PoroElast::Monolithic::solve()
{
  // we do a Newton-Raphson iteration here.
  // the specific time integration has set the following
  // --> On #rhs_ is the positive force residuum
  // --> On #systemmatrix_ is the effective dynamic tangent matrix

  //---------------------------------------- initialise equilibrium loop and norms
  setup_newton();

  //---------------------------------------------- iteration loop
  // equilibrium iteration loop (loop over k)
  while (((not converged()) and (iter_ <= itermax_)) or (iter_ <= itermin_))
  {
    timer_->start();
    Teuchos::Time timer("eval", true);
    timer.start();
    // compute residual forces #rhs_ and tangent #tang_
    // whose components are globally oriented
    // build linear system stiffness matrix and rhs/force residual for each
    // field, here e.g. for structure field: field want the iteration increment
    // 1.) Update(iterinc_),
    // 2.) evaluate_force_stiff_residual(),
    // 3.) prepare_system_for_newton_solve()
    evaluate(iterinc_, iter_ == 1);
    // std::cout << "  time for Evaluate : " <<  timer.totalElapsedTime(true) << "\n";
    // timer.reset();

    // if (iter_>1 and Step()>2 )
    // PoroFDCheck();

    // build norms
    build_convergence_norms();
    if ((not converged()) or combincfres_ == PoroElast::bop_or)
    {
      // (Newton-ready) residual with blanked Dirichlet DOFs (see adapter_timint!)
      // is done in prepare_system_for_newton_solve() within evaluate(iterinc_)
      linear_solve();
      // std::cout << "  time for Evaluate linear_solve: " << timer.totalElapsedTime(true) << "\n";
      // timer.reset();

      // reset solver tolerance
      solver_->reset_tolerance();

      // rebuild norms
      build_convergence_norms();
    }

    // print stuff
    print_newton_iter();

    // Aitken();

    // Recover Lagrangean Multiplier in Newton Iteration (for contact & contact no penetration!)
    // adjust LM Recovery for the meshtying case
    recover_lagrange_multiplier_after_newton_step(iterinc_);

    // increment equilibrium loop index
    iter_ += 1;
  }  // end equilibrium loop

  // correct iteration counter
  iter_ -= 1;

  // test whether max iterations was hit
  if ((converged()) and (Core::Communication::my_mpi_rank(get_comm()) == 0))
  {
    print_newton_conv();
  }
  else if (iter_ >= itermax_)
  {
    FOUR_C_THROW("Newton unconverged in {} iterations", iter_);
  }
}

void PoroElast::Monolithic::update_state_incrementally(
    std::shared_ptr<const Core::LinAlg::Vector<double>> iterinc)
{
  TEUCHOS_FUNC_TIME_MONITOR("PoroElast::Monolithic::update_state_incrementally");

  // displacement and fluid velocity & pressure incremental vector
  std::shared_ptr<const Core::LinAlg::Vector<double>> s_iterinc = nullptr;
  std::shared_ptr<const Core::LinAlg::Vector<double>> f_iterinc = nullptr;

  // do nothing in the case no increment vector is given
  if (iterinc == nullptr)
  {
    return;
  }
  else
  {
    // extract displacement sx and fluid fx incremental vector of global
    // unknown incremental vector x (different for splits)
    extract_field_vectors(iterinc, s_iterinc, f_iterinc);

    // update poro iterinc
    if (is_part_of_multifield_problem_) update_poro_iterinc(*iterinc);
  }

  update_state_incrementally(s_iterinc, f_iterinc);
}

void PoroElast::Monolithic::update_state_incrementally(
    std::shared_ptr<const Core::LinAlg::Vector<double>> s_iterinc,
    std::shared_ptr<const Core::LinAlg::Vector<double>> f_iterinc)
{
  // Newton update of the fluid field
  // update velocities and pressures before passed to the structural field
  //  update_iter_incrementally(fx),
  fluid_field()->update_newton(f_iterinc);

  // call all elements and assemble rhs and matrices
  // structural field

  // structure Evaluate (builds tangent, residual and applies DBC)

  // apply current velocity and pressures to structure
  set_fluid_solution();

  // Monolithic Poroelasticity accesses the linearised structure problem:
  //   UpdaterIterIncrementally(sx),
  structure_field()->update_state_incrementally(s_iterinc);
  // fluid field

  // fluid Evaluate
  // (builds tangent, residual and applies DBC and recent coupling values)

  // set structure displacements onto the fluid
  set_struct_solution();

  // apply current velocity of fluid to ContactManager if contact problem
  if (no_penetration_)
    set_poro_contact_states();  // ATM svel is set in structure evaluate as the vel of the structure
                                // is evaluated there ...
}

void PoroElast::Monolithic::evaluate(
    std::shared_ptr<const Core::LinAlg::Vector<double>> iterinc, bool firstiter)
{
  TEUCHOS_FUNC_TIME_MONITOR("PoroElast::Monolithic::Evaluate");

  // Evaluate Stiffness and RHS of Structural & Fluid field
  evaluate_fields(iterinc);

  // fill off diagonal blocks and fill global system matrix
  setup_system_matrix();

  // check whether we have a sanely filled tangent matrix
  if (not systemmatrix_->filled())
  {
    FOUR_C_THROW("Effective tangent matrix must be filled here");
  }

  // create full monolithic rhs vector
  setup_rhs(firstiter);

  // Modify System for Contact or Meshtying!
  eval_poro_mortar();
}

void PoroElast::Monolithic::evaluate(std::shared_ptr<const Core::LinAlg::Vector<double>> s_iterinc,
    std::shared_ptr<const Core::LinAlg::Vector<double>> f_iterinc, bool firstiter)
{
  TEUCHOS_FUNC_TIME_MONITOR("PoroElast::Monolithic::Evaluate");

  // Evaluate Stiffness and RHS of Structural & Fluid field
  evaluate_fields(s_iterinc, f_iterinc);

  // fill off diagonal blocks and fill global system matrix
  setup_system_matrix();

  // check whether we have a sanely filled tangent matrix
  if (not systemmatrix_->filled())
  {
    FOUR_C_THROW("Effective tangent matrix must be filled here");
  }

  // create full monolithic rhs vector
  setup_rhs(firstiter);

  // Modify System for Contact or Meshtying!
  eval_poro_mortar();
}

void PoroElast::Monolithic::evaluate_fields(
    std::shared_ptr<const Core::LinAlg::Vector<double>> s_iterinc,
    std::shared_ptr<const Core::LinAlg::Vector<double>> f_iterinc)
{
  // Update State incrementally
  update_state_incrementally(s_iterinc, f_iterinc);

  // Monolithic Poroelasticity accesses the linearised structure problem:
  //   evaluate_force_stiff_residual()
  //   prepare_system_for_newton_solve()
  structure_field()->evaluate(nullptr);

  // monolithic Poroelasticity accesses the linearised fluid problem
  //   evaluate_rhs_tang_residual() and
  //   prepare_system_for_newton_solve()
  fluid_field()->evaluate(nullptr);
}

void PoroElast::Monolithic::evaluate_fields(
    std::shared_ptr<const Core::LinAlg::Vector<double>> iterinc)
{
  // Update State incrementally
  update_state_incrementally(iterinc);

  // Monolithic Poroelasticity accesses the linearised structure problem:
  //   evaluate_force_stiff_residual()
  //   prepare_system_for_newton_solve()
  structure_field()->evaluate(nullptr);

  // monolithic Poroelasticity accesses the linearised fluid problem
  //   evaluate_rhs_tang_residual() and
  //   prepare_system_for_newton_solve()
  fluid_field()->evaluate(nullptr);
}

void PoroElast::Monolithic::extract_field_vectors(
    std::shared_ptr<const Core::LinAlg::Vector<double>> x,
    std::shared_ptr<const Core::LinAlg::Vector<double>>& sx,
    std::shared_ptr<const Core::LinAlg::Vector<double>>& fx, bool firstcall)
{
  TEUCHOS_FUNC_TIME_MONITOR("PoroElast::Monolithic::extract_field_vectors");

  // process structure unknowns of the first field
  sx = extractor()->extract_vector(*x, 0);

  // process fluid unknowns of the second field
  fx = extractor()->extract_vector(*x, 1);
}

void PoroElast::Monolithic::setup_system()
{
  // new timint with nitsche contact. For old timint this is in the constructor but for new it has
  // to be after poroalgo->read_restart(restart);
  if ((!oldstructimint_) && structure_field()->have_model(Inpar::Solid::model_contact))
  {
    auto& model_evaluator_contact = dynamic_cast<Solid::ModelEvaluator::Contact&>(
        structure_field()->model_evaluator(Inpar::Solid::model_contact));

    std::shared_ptr<CONTACT::NitscheStrategyPoro> contact_strategy_nitsche_poro =
        std::dynamic_pointer_cast<CONTACT::NitscheStrategyPoro>(
            model_evaluator_contact.strategy_ptr());

    if (contact_strategy_nitsche_poro != nullptr)
    {
      nit_contact_ = true;
      no_penetration_ = contact_strategy_nitsche_poro->has_poro_no_penetration();
    }
  }

  if (no_penetration_ && not(strmethodname_ == Inpar::Solid::DynamicType::OneStepTheta))
  {
    FOUR_C_THROW(
        "Porous contact with no penetration is only implemented for OneStepTheta. Please set "
        "DYNAMICTYPE to OneStepTheta.");
  }


  {
    // -------------------------------------------------------------create combined map
    std::vector<std::shared_ptr<const Core::LinAlg::Map>> vecSpaces;

    // Note:
    // when using constraints applied via Lagrange-Multipliers there is a
    // difference between structure_field()->dof_row_map() and structure_field()->dof_row_map(0).
    // structure_field()->dof_row_map(0) returns the dof_row_map
    // known to the discretization (without lagrange multipliers)
    // while structure_field()->dof_row_map() returns the dof_row_map known to
    // the constraint manager (with lagrange multipliers)
    // In the constrained case we want the "whole" RowDofMap,
    // otherwise both calls are equivalent

    vecSpaces.push_back(structure_field()->dof_row_map());
    // use its own dof_row_map, that is the 0th map of the discretization
    vecSpaces.push_back(fluid_field()->dof_row_map(0));

    if (vecSpaces[0]->num_global_elements() == 0) FOUR_C_THROW("No structure equation. Panic.");
    if (vecSpaces[1]->num_global_elements() == 0) FOUR_C_THROW("No fluid equation. Panic.");

    // full Poroelasticity-map
    fullmap_ = Core::LinAlg::MultiMapExtractor::merge_maps(vecSpaces);
    // full Poroelasticity-blockmap
    blockrowdofmap_->setup(*fullmap_, vecSpaces);
  }
  // -------------------------------------------------------------

  //-----------------------------------build map of global dofs with DBC
  build_combined_dbc_map();
  // -------------------------------------------------------------

  // initialize Poroelasticity-systemmatrix_
  systemmatrix_ =
      std::make_shared<Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(
          *extractor(), *extractor(), 81, false, true);

  k_sf_ = std::make_shared<Core::LinAlg::SparseMatrix>(
      *(structure_field()->dof_row_map()), 81, true, true);
  k_fs_ = std::make_shared<Core::LinAlg::SparseMatrix>(
      *(fluid_field()->discretization()->dof_row_map(0)),
      //*(fluid_field()->dof_row_map()),
      81, true, true);

  nopen_handle_->setup(*dof_row_map(), (fluid_field()->discretization()->dof_row_map(0)));

  setup_equilibration();
}  // SetupSystem()

void PoroElast::Monolithic::setup_equilibration()
{
  // instantiate appropriate equilibration class
  auto equilibration_method =
      std::vector<Core::LinAlg::EquilibrationMethod>(1, equilibration_method_);
  equilibration_ = Core::LinAlg::build_equilibration(
      Core::LinAlg::MatrixType::block_field, equilibration_method, fullmap_);
}

void PoroElast::Monolithic::setup_system_matrix(Core::LinAlg::BlockSparseMatrixBase& mat)
{
  TEUCHOS_FUNC_TIME_MONITOR("PoroElast::Monolithic::setup_system_matrix");

  // pure structural part k_ss (3nx3n)

  // build pure structural block k_ss
  // build block matrix
  // The maps of the block matrix have to match the maps of the blocks we
  // insert here. Extract Jacobian matrices and put them into composite system
  // matrix W
  std::shared_ptr<Core::LinAlg::SparseMatrix> k_ss = structure_field()->system_matrix();

  if (k_ss == nullptr) FOUR_C_THROW("structure system matrix null pointer!");

  /*----------------------------------------------------------------------*/
  // structural part k_sf (3nxn)
  // build mechanical-fluid block

  // create empty matrix
  std::shared_ptr<Core::LinAlg::SparseMatrix> k_sf = struct_fluid_coupling_matrix();

  // call the element and calculate the matrix block
  apply_str_coupl_matrix(k_sf);

  if (!matchinggrid_)
  {
    k_sf->complete(*(structure_field()->discretization()->dof_row_map(1)),
        *(structure_field()->discretization()->dof_row_map(0)));

    k_sf = volcoupl_->apply_matrix_mapping12(*k_sf);
  }

  /*----------------------------------------------------------------------*/
  // pure fluid part k_ff ( (3n+1)x(3n+1) )

  // build pure fluid block k_ff
  // build block matrix
  // The maps of the block matrix have to match the maps of the blocks we
  // insert here. Extract Jacobian matrices and put them into composite system
  // matrix W
  std::shared_ptr<Core::LinAlg::SparseMatrix> k_ff = fluid_field()->system_matrix();

  if (k_ff == nullptr) FOUR_C_THROW("fuid system matrix null pointer!");

  if (nopen_handle_->has_cond())
  {
    // Evaluate poroelasticity specific conditions
    evaluate_condition(*k_ff);
  }

  /*----------------------------------------------------------------------*/
  // fluid part k_fs ( (3n+1)x3n )
  // build fluid-mechanical block

  // create empty matrix
  std::shared_ptr<Core::LinAlg::SparseMatrix> k_fs = fluid_struct_coupling_matrix();

  // call the element and calculate the matrix block
  apply_fluid_coupl_matrix(k_fs);

  if (!matchinggrid_)
  {
    k_fs->complete(*(fluid_field()->discretization()->dof_row_map(1)),
        *(fluid_field()->discretization()->dof_row_map(0)));

    k_fs = volcoupl_->apply_matrix_mapping21(*k_fs);
  }

  // assign structure part to the Poroelasticity matrix
  mat.assign(0, 0, Core::LinAlg::DataAccess::Share, *k_ss);
  // assign coupling part to the Poroelasticity matrix
  mat.assign(0, 1, Core::LinAlg::DataAccess::Share, *k_sf);
  // assign fluid part to the poroelasticity matrix
  mat.assign(1, 1, Core::LinAlg::DataAccess::Share, *k_ff);
  // assign coupling part to the Poroelasticity matrix
  mat.assign(1, 0, Core::LinAlg::DataAccess::Share, *k_fs);

  /*----------------------------------------------------------------------*/
  // done. make sure all blocks are filled.
  mat.complete();
}

void PoroElast::Monolithic::setup_rhs(bool firstcall)
{
  TEUCHOS_FUNC_TIME_MONITOR("PoroElast::Monolithic::setup_rhs");

  // create full monolithic rhs vector
  if (rhs_ == nullptr) rhs_ = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map(), true);

  // fill the Poroelasticity rhs vector rhs_ with the single field rhss
  setup_vector(*rhs_, structure_field()->rhs(), fluid_field()->rhs());

  // add rhs terms due to no penetration condition
  if (iterinc_) nopen_handle_->apply_cond_rhs(*iterinc_, *rhs_);
}

void PoroElast::Monolithic::prepare_time_step() { PoroBase::prepare_time_step(); }

void PoroElast::Monolithic::linear_solve()
{
  Core::LinAlg::SolverParams solver_params;
  if (solveradapttol_ and (iter_ > 1))
  {
    solver_params.nonlin_tolerance = tolfres_;
    solver_params.nonlin_residual = normrhs_;
    solver_params.lin_tol_better = solveradaptolbetter_;
  }
  iterinc_->put_scalar(0.0);  // Useful? depends on solver and more

  // equilibrate global system of equations if necessary
  equilibration_->equilibrate_system(systemmatrix_, rhs_, blockrowdofmap_);

  if (directsolve_)
  {
    // merge blockmatrix to SparseMatrix
    std::shared_ptr<Core::LinAlg::SparseMatrix> sparse = systemmatrix_->merge();

    // apply dirichlet boundary conditions
    Core::LinAlg::apply_dirichlet_to_system(
        *sparse, *iterinc_, *rhs_, *zeros_, *combined_dbc_map());

    // standard solver call

    solver_params.refactor = true;
    solver_params.reset = iter_ == 1;
    solver_->solve(sparse, iterinc_, rhs_, solver_params);
  }
  else  // use bgs2x2_operator
  {
    // apply dirichlet boundary conditions
    Core::LinAlg::apply_dirichlet_to_system(
        *systemmatrix_, *iterinc_, *rhs_, *zeros_, *combined_dbc_map());

    // standard solver call
    solver_params.refactor = true;
    solver_params.reset = iter_ == 1;
    solver_->solve(systemmatrix_, iterinc_, rhs_, solver_params);
  }

  equilibration_->unequilibrate_increment(iterinc_);
}

void PoroElast::Monolithic::create_linear_solver()
{
  // get dynamic section
  const Teuchos::ParameterList& porodyn = Global::Problem::instance()->poroelast_dynamic_params();

  // get the linear solver number
  const int linsolvernumber = porodyn.get<int>("LINEAR_SOLVER");
  if (linsolvernumber == (-1))
  {
    FOUR_C_THROW(
        "no linear solver defined for monolithic Poroelasticity. Please set LINEAR_SOLVER in "
        "POROELASTICITY DYNAMIC to a valid number!");
  }

  // get parameter list of structural dynamics
  const Teuchos::ParameterList& sdyn = Global::Problem::instance()->structural_dynamic_params();
  // use solver blocks for structure
  // get the solver number used for structural solver
  const int slinsolvernumber = sdyn.get<int>("LINEAR_SOLVER");
  // check if the structural solver has a valid solver number
  if (slinsolvernumber == (-1))
  {
    FOUR_C_THROW(
        "no linear solver defined for structural field. Please set LINEAR_SOLVER in STRUCTURAL "
        "DYNAMIC to a valid number!");
  }

  // get parameter list of fluid dynamics
  const Teuchos::ParameterList& fdyn = Global::Problem::instance()->fluid_dynamic_params();
  // use solver blocks for fluid
  // get the solver number used for fluid solver
  const int flinsolvernumber = fdyn.get<int>("LINEAR_SOLVER");
  // check if the fluid solver has a valid solver number
  if (flinsolvernumber == (-1))
  {
    FOUR_C_THROW(
        "no linear solver defined for fluid field. Please set LINEAR_SOLVER in FLUID DYNAMIC to a "
        "valid number!");
  }

  // get solver parameter list of linear Poroelasticity solver
  const Teuchos::ParameterList& porosolverparams =
      Global::Problem::instance()->solver_params(linsolvernumber);

  const auto solvertype =
      Teuchos::getIntegralValue<Core::LinearSolver::SolverType>(porosolverparams, "SOLVER");

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
      Teuchos::getIntegralValue<Core::LinearSolver::PreconditionerType>(porosolverparams, "AZPREC");

  solver_ = std::make_shared<Core::LinAlg::Solver>(porosolverparams, get_comm(),
      Global::Problem::instance()->solver_params_callback(),
      Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
          Global::Problem::instance()->io_params(), "VERBOSITY"));

  // use solver blocks for structure and fluid
  const Teuchos::ParameterList& ssolverparams =
      Global::Problem::instance()->solver_params(slinsolvernumber);
  const Teuchos::ParameterList& fsolverparams =
      Global::Problem::instance()->solver_params(flinsolvernumber);

  solver_->put_solver_params_to_sub_params("Inverse1", ssolverparams,
      Global::Problem::instance()->solver_params_callback(),
      Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
          Global::Problem::instance()->io_params(), "VERBOSITY"),
      get_comm());
  solver_->put_solver_params_to_sub_params("Inverse2", fsolverparams,
      Global::Problem::instance()->solver_params_callback(),
      Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
          Global::Problem::instance()->io_params(), "VERBOSITY"),
      get_comm());

  switch (azprectype)
  {
    case Core::LinearSolver::PreconditionerType::block_teko:
    {
      Core::LinearSolver::Parameters::compute_solver_parameters(
          *structure_field()->discretization(), solver_->params().sublist("Inverse1"));
      Core::LinearSolver::Parameters::compute_solver_parameters(
          *fluid_field()->discretization(), solver_->params().sublist("Inverse2"));
    }
    break;
    default:
      FOUR_C_THROW("Block Gauss-Seidel preconditioner expected.");
  }
}

void PoroElast::Monolithic::initial_guess(std::shared_ptr<Core::LinAlg::Vector<double>> ig)
{
  TEUCHOS_FUNC_TIME_MONITOR("PoroElast::Monolithic::initial_guess");

  // InitialGuess() is called of the single fields and results are put in
  // increment vector ig
  setup_vector(*ig,
      // returns residual displacements \f$\Delta D_{n+1}^{<k>}\f$ - disi_
      structure_field()->initial_guess(),
      // returns residual velocities or iterative fluid increment - incvel_
      fluid_field()->initial_guess());
}

void PoroElast::Monolithic::setup_vector(Core::LinAlg::Vector<double>& f,
    std::shared_ptr<const Core::LinAlg::Vector<double>> sv,
    std::shared_ptr<const Core::LinAlg::Vector<double>> fv)
{
  // extract dofs of the two fields
  // and put the structural/fluid field vector into the global vector f
  // noticing the block number

  extractor()->insert_vector(*sv, 0, f);
  if (not oldstructimint_) f.scale(-1);
  extractor()->insert_vector(*fv, 1, f);
}

bool PoroElast::Monolithic::converged()
{
  // check for single norms
  bool convinc = false;
  bool convfres = false;

  // residual increments
  switch (normtypeinc_)
  {
    case PoroElast::convnorm_abs_global:
      convinc = norminc_ < tolinc_;
      break;
    case PoroElast::convnorm_abs_singlefields:
      convinc = (normincstruct_ < tolinc_struct_ and normincfluidvel_ < tolinc_velocity_ and
                 normincfluidpres_ < tolinc_pressure_ and normincporo_ < tolinc_porosity_);
      break;
    default:
      FOUR_C_THROW("Cannot check for convergence of residual values!");
      break;
  }

  // residual forces
  switch (normtypefres_)
  {
    case PoroElast::convnorm_abs_global:
      convfres = normrhs_ < tolfres_;
      break;
    case PoroElast::convnorm_abs_singlefields:
      convfres = (normrhsstruct_ < tolfres_struct_ and normrhsfluidvel_ < tolfres_velocity_ and
                  normrhsfluidpres_ < tolfres_pressure_ and normrhsporo_ < tolfres_porosity_);
      break;
    default:
      FOUR_C_THROW("Cannot check for convergence of residual forces!");
      break;
  }

  // combine increments and forces
  bool conv = false;
  if (combincfres_ == PoroElast::bop_and)
    conv = convinc and convfres;
  else if (combincfres_ == PoroElast::bop_or)
    conv = convinc or convfres;
  else
    FOUR_C_THROW("Something went terribly wrong with binary operator!");

  if (conv)
  {
    // clean up as soon as old time integration is unused!
    if (oldstructimint_)
    {
      if (structure_field()->meshtying_contact_bridge() != nullptr)
      // assume dual mortar lagmult contact!!! ... for general case store member into algo!
      {
        if (structure_field()->meshtying_contact_bridge()->have_contact())
        {
          conv =
              structure_field()->meshtying_contact_bridge()->get_strategy().active_set_converged();
        }
      }
    }
  }

  // return things
  return conv;
}

void PoroElast::Monolithic::print_newton_iter()
{
  // print to standard out
  // replace myrank_ here general by Core::Communication::my_mpi_rank(Comm())
  if ((Core::Communication::my_mpi_rank(get_comm()) == 0) and printscreen_ and
      (step() % printscreen_ == 0) and printiter_)
  {
    if (iter_ == 1) print_newton_iter_header(stdout);
    print_newton_iter_text(stdout);
  }
}

void PoroElast::Monolithic::print_newton_iter_header(FILE* ofile)
{
  // open outstringstream
  std::ostringstream oss;

  print_newton_iter_header_stream(oss);

  // add solution time
  oss << std::setw(14) << "wct";
  // finish oss
  oss << std::ends;

  // print to screen (could be done differently...)
  if (ofile == nullptr) FOUR_C_THROW("no ofile available");
  fprintf(ofile, "%s\n", oss.str().c_str());

  // print it, now
  fflush(ofile);
}

void PoroElast::Monolithic::print_newton_iter_header_stream(std::ostringstream& oss)
{
  oss << "------------------------------------------------------------" << std::endl;
  oss << "                   Newton-Raphson Scheme                    " << std::endl;
  oss << "                NormRES " << EnumTools::enum_name(vectornormfres_);
  oss << "     NormINC " << EnumTools::enum_name(vectornorminc_) << "                    "
      << std::endl;
  oss << "------------------------------------------------------------" << std::endl;

  // enter converged state etc
  oss << "numiter";

  // different style due relative or absolute error checking

  // --------------------------------------------------------global system test
  // residual forces
  switch (normtypefres_)
  {
    case PoroElast::convnorm_abs_global:
      oss << std::setw(15) << "abs-res"
          << "(" << std::setw(5) << std::setprecision(2) << tolfres_ << ")";
      break;
    case PoroElast::convnorm_abs_singlefields:
      oss << std::setw(15) << "abs-s-res"
          << "(" << std::setw(5) << std::setprecision(2) << tolfres_struct_ << ")";
      if (porosity_dof_)
        oss << std::setw(15) << "abs-poro-res"
            << "(" << std::setw(5) << std::setprecision(2) << tolfres_porosity_ << ")";
      oss << std::setw(15) << "abs-fvel-res"
          << "(" << std::setw(5) << std::setprecision(2) << tolfres_velocity_ << ")";
      oss << std::setw(15) << "abs-fpres-res"
          << "(" << std::setw(5) << std::setprecision(2) << tolfres_pressure_ << ")";
      break;
    default:
      FOUR_C_THROW("Unknown or undefined convergence form for residual.");
      break;
  }

  switch (normtypeinc_)
  {
    case PoroElast::convnorm_abs_global:
      oss << std::setw(15) << "abs-inc"
          << "(" << std::setw(5) << std::setprecision(2) << tolinc_ << ")";
      break;
      break;
    case PoroElast::convnorm_abs_singlefields:
      oss << std::setw(15) << "abs-s-inc"
          << "(" << std::setw(5) << std::setprecision(2) << tolinc_struct_ << ")";
      if (porosity_dof_)
        oss << std::setw(15) << "abs-poro-inc"
            << "(" << std::setw(5) << std::setprecision(2) << tolinc_porosity_ << ")";
      oss << std::setw(15) << "abs-fvel-inc"
          << "(" << std::setw(5) << std::setprecision(2) << tolinc_velocity_ << ")";
      oss << std::setw(15) << "abs-fpres-inc"
          << "(" << std::setw(5) << std::setprecision(2) << tolinc_pressure_ << ")";
      break;
    default:
      FOUR_C_THROW("Unknown or undefined convergence form for increment.");
      break;
  }
}

void PoroElast::Monolithic::print_newton_iter_text(FILE* ofile)
{
  // open outstringstream
  std::ostringstream oss;

  print_newton_iter_text_stream(oss);

  // add solution time
  oss << std::setw(14) << std::setprecision(2) << std::scientific << timer_->totalElapsedTime(true);
  // finish oss
  oss << std::ends;

  // print to screen (could be done differently...)
  if (ofile == nullptr) FOUR_C_THROW("no ofile available");
  fprintf(ofile, "%s\n", oss.str().c_str());

  // print it, now
  fflush(ofile);
}

void PoroElast::Monolithic::print_newton_iter_text_stream(std::ostringstream& oss)
{
  // enter converged state etc
  oss << std::setw(7) << iter_;

  // different style due relative or absolute error checking

  // --------------------------------------------------------global system test
  // residual forces
  switch (normtypefres_)
  {
    case PoroElast::convnorm_abs_global:
      oss << std::setw(22) << std::setprecision(5) << std::scientific << normrhs_;
      break;
    case PoroElast::convnorm_abs_singlefields:
      break;
    default:
      FOUR_C_THROW("Unknown or undefined convergence form for global residual.");
      break;
  }
  // increments
  switch (normtypeinc_)
  {
    case PoroElast::convnorm_abs_global:
      oss << std::setw(22) << std::setprecision(5) << std::scientific << norminc_;
      break;
    case PoroElast::convnorm_abs_singlefields:
      break;
    default:
      FOUR_C_THROW("Unknown or undefined convergence form for global increment.");
      break;
  }

  // --------------------------------------------------------single field test
  switch (normtypefres_)
  {
    case PoroElast::convnorm_abs_singlefields:
      oss << std::setw(22) << std::setprecision(5) << std::scientific << normrhsstruct_;
      if (porosity_dof_)
        oss << std::setw(22) << std::setprecision(5) << std::scientific << normrhsporo_;
      oss << std::setw(22) << std::setprecision(5) << std::scientific << normrhsfluidvel_;
      oss << std::setw(22) << std::setprecision(5) << std::scientific << normrhsfluidpres_;
      break;
    case PoroElast::convnorm_abs_global:
      break;
    default:
      FOUR_C_THROW("Unknown or undefined convergence form for single field residual.");
      break;
  }

  switch (normtypeinc_)
  {
    case PoroElast::convnorm_abs_singlefields:
      oss << std::setw(22) << std::setprecision(5) << std::scientific << normincstruct_;
      if (porosity_dof_)
        oss << std::setw(22) << std::setprecision(5) << std::scientific << normincporo_;
      oss << std::setw(22) << std::setprecision(5) << std::scientific << normincfluidvel_;
      oss << std::setw(22) << std::setprecision(5) << std::scientific << normincfluidpres_;
      break;
    case PoroElast::convnorm_abs_global:
      break;
    default:
      FOUR_C_THROW("Unknown or undefined convergence form for single field increment.");
      break;
  }
}

void PoroElast::Monolithic::print_newton_conv() {}

void PoroElast::Monolithic::apply_str_coupl_matrix(
    std::shared_ptr<Core::LinAlg::SparseOperator> k_sf)
{
  k_sf->zero();

  // create the parameters for the discretization
  Teuchos::ParameterList sparams;

  if (oldstructimint_)
  {
    const std::string action = "struct_poro_calc_fluidcoupling";
    sparams.set("action", action);
    // other parameters that might be needed by the elements
    sparams.set("delta time", dt());
    sparams.set("total time", time());
  }
  else
  {
    //! pointer to the model evaluator data container
    std::shared_ptr<Core::Elements::ParamsMinimal> params =
        std::make_shared<Core::Elements::ParamsMinimal>();

    // set parameters needed for element evaluation
    params->set_action_type(Core::Elements::struct_poro_calc_fluidcoupling);
    params->set_total_time(time());
    params->set_delta_time(dt());

    sparams.set<std::shared_ptr<Core::Elements::ParamsInterface>>("interface", params);
  }

  structure_field()->discretization()->clear_state();
  structure_field()->discretization()->set_state(0, "displacement", *structure_field()->dispnp());
  structure_field()->discretization()->set_state(0, "velocity", *structure_field()->velnp());

  // build specific assemble strategy for mechanical-fluid system matrix
  // from the point of view of structure_field:
  // structdofset = 0, fluiddofset = 1
  Core::FE::AssembleStrategy structuralstrategy(0,  // structdofset for row
      1,                                            // fluiddofset for column
      k_sf,                                         // mechanical-fluid coupling matrix
      nullptr, nullptr, nullptr, nullptr);

  // evaluate the mechanical-fluid system matrix on the structural element
  structure_field()->discretization()->evaluate_condition(
      sparams, structuralstrategy, "PoroCoupling");
  // structure_field()->discretization()->evaluate( sparams, structuralstrategy);
  structure_field()->discretization()->clear_state();

  // scale with time integration factor
  k_sf->scale(1.0 - structure_field()->tim_int_param());
}

void PoroElast::Monolithic::apply_fluid_coupl_matrix(
    std::shared_ptr<Core::LinAlg::SparseOperator> k_fs)
{
  k_fs->zero();

  // create the parameters for the discretization
  Teuchos::ParameterList fparams;
  // action for elements
  fparams.set<FLD::Action>("action", FLD::calc_porousflow_fluid_coupling);
  // physical type
  fparams.set<Inpar::FLUID::PhysicalType>("Physical Type", fluid_field()->physical_type());
  // other parameters that might be needed by the elements
  fparams.set("delta time", dt());
  fparams.set("total time", time());

  fluid_field()->discretization()->clear_state();

  // set general vector values needed by elements
  fluid_field()->discretization()->set_state(0, "dispnp", *fluid_field()->dispnp());
  fluid_field()->discretization()->set_state(0, "dispn", *fluid_field()->dispn());
  fluid_field()->discretization()->set_state(0, "gridv", *fluid_field()->grid_vel());
  fluid_field()->discretization()->set_state(0, "gridvn", *fluid_field()->grid_veln());
  fluid_field()->discretization()->set_state(0, "veln", *fluid_field()->veln());
  fluid_field()->discretization()->set_state(0, "accnp", *fluid_field()->accnp());
  fluid_field()->discretization()->set_state(0, "accam", *fluid_field()->accam());
  fluid_field()->discretization()->set_state(0, "accn", *fluid_field()->accn());

  fluid_field()->discretization()->set_state(0, "scaaf", *fluid_field()->scaaf());

  fluid_field()->discretization()->set_state(0, "hist", *fluid_field()->hist());

  // set scheme-specific element parameters and vector values
  if (fluid_field()->tim_int_scheme() == Inpar::FLUID::timeint_npgenalpha or
      fluid_field()->tim_int_scheme() == Inpar::FLUID::timeint_npgenalpha)
    fluid_field()->discretization()->set_state(0, "velaf", *fluid_field()->velaf());
  else
    fluid_field()->discretization()->set_state(0, "velaf", *fluid_field()->velnp());

  fluid_field()->discretization()->set_state(0, "velnp", *fluid_field()->velnp());

  // build specific assemble strategy for the fluid-mechanical system matrix
  // from the point of view of fluid_field:
  // fluiddofset = 0, structdofset = 1
  Core::FE::AssembleStrategy fluidstrategy(0,  // fluiddofset for row
      1,                                       // structdofset for column
      k_fs,                                    // fluid-mechanical matrix
      nullptr,                                 // no other matrix or vectors
      nullptr, nullptr, nullptr);

  // evaluate the fluid-mechanical system matrix on the fluid element
  fluid_field()->discretization()->evaluate_condition(fparams, fluidstrategy, "PoroCoupling");
  // fluid_field()->discretization()->evaluate( fparams, fluidstrategy );

  // evaluate coupling terms from partial integration of continuity equation
  if (part_int_cond_)
  {
    // create the parameters for the discretization
    Teuchos::ParameterList params;
    // action for elements
    params.set<FLD::BoundaryAction>("action", FLD::poro_boundary);
    params.set("total time", time());
    params.set("delta time", dt());
    params.set<PoroElast::Coupltype>("coupling", PoroElast::fluidstructure);
    params.set("timescale", fluid_field()->residual_scaling());
    params.set<Inpar::FLUID::PhysicalType>("Physical Type", fluid_field()->physical_type());

    fluid_field()->discretization()->clear_state();
    fluid_field()->discretization()->set_state(0, "dispnp", *fluid_field()->dispnp());
    fluid_field()->discretization()->set_state(0, "gridv", *fluid_field()->grid_vel());
    fluid_field()->discretization()->set_state(0, "velnp", *fluid_field()->velnp());
    fluid_field()->discretization()->set_state(0, "scaaf", *fluid_field()->scaaf());

    fluid_field()->discretization()->evaluate_condition(params, fluidstrategy, "PoroPartInt");

    fluid_field()->discretization()->clear_state();
  }
  if (pres_int_cond_)
  {
    // create the parameters for the discretization
    Teuchos::ParameterList params;
    // action for elements
    params.set<FLD::BoundaryAction>("action", FLD::poro_prescoupl);
    params.set<PoroElast::Coupltype>("coupling", PoroElast::fluidstructure);
    params.set<Inpar::FLUID::PhysicalType>("Physical Type", fluid_field()->physical_type());

    fluid_field()->discretization()->clear_state();
    fluid_field()->discretization()->set_state(0, "dispnp", *fluid_field()->dispnp());
    fluid_field()->discretization()->set_state(0, "velnp", *fluid_field()->velnp());

    fluid_field()->discretization()->evaluate_condition(params, fluidstrategy, "PoroPresInt");

    fluid_field()->discretization()->clear_state();
  }

  // apply normal flux condition on coupling part
  if (nopen_handle_->has_cond())
  {
    k_fs->complete(structure_field()->system_matrix()->range_map(),
        fluid_field()->system_matrix()->range_map());
    evaluate_condition(*k_fs, PoroElast::fluidstructure);
  }

  fluid_field()->discretization()->clear_state();
}

[[maybe_unused]] void PoroElast::Monolithic::poro_fd_check()
{
  std::cout << "\n******************finite difference check***************" << std::endl;

  int dof_struct = (dof_row_map_structure()->num_global_elements());
  int dof_fluid = (dof_row_map_fluid()->num_global_elements());

  std::cout << "structure field has " << dof_struct << " DOFs" << std::endl;
  std::cout << "fluid field has " << dof_fluid << " DOFs" << std::endl;

  std::shared_ptr<Core::LinAlg::Vector<double>> iterinc = nullptr;
  std::shared_ptr<Core::LinAlg::Vector<double>> abs_iterinc = nullptr;
  iterinc = Core::LinAlg::create_vector(*dof_row_map(), true);
  abs_iterinc = Core::LinAlg::create_vector(*dof_row_map(), true);

  const int dofs = iterinc->global_length();
  std::cout << "in total " << dofs << " DOFs" << std::endl;
  const double delta = 1e-8;

  iterinc->put_scalar(0.0);

  iterinc->replace_global_value(0, delta);

  abs_iterinc->update(1.0, *iterinc_, 0.0);

  Core::LinAlg::SparseMatrix stiff_approx(*dof_row_map(), 81);

  Core::LinAlg::Vector<double> rhs_old(*dof_row_map(), true);
  rhs_old.update(1.0, *rhs_, 0.0);
  Core::LinAlg::Vector<double> rhs_copy(*dof_row_map(), true);

  Core::LinAlg::SparseMatrix sparse(*systemmatrix_->merge());

  bool output = false;
  if (output)
  {
    std::cout << "iterinc_" << std::endl;
    iterinc_->print(std::cout);
    std::cout << "meshdisp: " << std::endl;
    fluid_field()->dispnp()->print(std::cout);
    std::cout << "disp: " << std::endl;
    structure_field()->dispnp()->print(std::cout);
    std::cout << "fluid vel" << std::endl;
    fluid_field()->velnp()->print(std::cout);
    std::cout << "fluid acc" << std::endl;
    fluid_field()->accnp()->print(std::cout);
    std::cout << "gridvel fluid" << std::endl;
    fluid_field()->grid_vel()->print(std::cout);
    std::cout << "gridvel struct" << std::endl;
    structure_field()->velnp()->print(std::cout);
  }

  const int row_number = -1;
  const int column_number = -1;
  for (int i = 0; i < dofs; ++i)
  {
    if (combined_dbc_map()->my_gid(i))
    {
      iterinc->replace_global_value(i, 0.0);
    }
    abs_iterinc->update(1.0, *iterinc, 1.0);

    if (i == column_number)
      std::cout << "\n******************" << column_number + 1 << ". Spalte!!***************"
                << std::endl;

    evaluate(iterinc, iter_ == 1);

    rhs_copy.update(1.0, *rhs_, 0.0);

    iterinc_->put_scalar(0.0);  // Useful? depends on solver and more
    Core::LinAlg::apply_dirichlet_to_system(
        sparse, *iterinc_, rhs_copy, *zeros_, *combined_dbc_map());


    if (i == column_number)
    {
      std::cout << "rhs_: " << rhs_copy.local_values_as_span()[row_number] << std::endl;
      std::cout << "rhs_old: " << rhs_old.local_values_as_span()[row_number] << std::endl;
    }

    rhs_copy.update(-1.0, rhs_old, 1.0);
    rhs_copy.scale(-1.0 / delta);

    int* index = &i;
    for (int j = 0; j < dofs; ++j)
    {
      double value = rhs_copy.local_values_as_span()[j];
      stiff_approx.insert_global_values(j, 1, &value, index);

      if ((j == row_number) and (i == column_number))
      {
        std::cout << "\n******************" << row_number + 1 << ". Row!!***************"
                  << std::endl;
        std::cout << "iterinc_" << std::endl;
        iterinc_->print(std::cout);
        std::cout << "meshdisp: " << std::endl;
        fluid_field()->dispnp()->print(std::cout);
        std::cout << "disp: " << std::endl;
        structure_field()->dispnp()->print(std::cout);
        std::cout << "fluid vel" << std::endl;
        fluid_field()->velnp()->print(std::cout);
        std::cout << "fluid acc" << std::endl;
        fluid_field()->accnp()->print(std::cout);
        std::cout << "gridvel fluid" << std::endl;
        fluid_field()->grid_vel()->print(std::cout);
        std::cout << "gridvel struct" << std::endl;
        structure_field()->velnp()->print(std::cout);

        std::cout << "stiff_apprx(" << row_number << "," << column_number
                  << "): " << rhs_copy.local_values_as_span()[row_number] << std::endl;

        std::cout << "value(" << row_number << "," << column_number << "): " << value << std::endl;
        std::cout << "\n******************" << row_number + 1 << ". Row End!!***************"
                  << std::endl;
      }
    }

    if (not combined_dbc_map()->my_gid(i)) iterinc->replace_global_value(i, -delta);

    iterinc->replace_global_value(i - 1, 0.0);

    if (i != dofs - 1) iterinc->replace_global_value(i + 1, delta);

    if (i == column_number)
      std::cout << "\n******************" << column_number + 1 << ". Column End!!***************"
                << std::endl;
  }

  evaluate(iterinc, iter_ == 1);

  stiff_approx.complete();

  Core::LinAlg::SparseMatrix stiff_approx_sparse(stiff_approx);
  stiff_approx_sparse.add(sparse, false, -1.0, 1.0);

  stiff_approx_sparse.complete();
  sparse.complete();

  bool success = true;
  double error_max = 0.0;
  double abs_error_max = 0.0;
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
            int errornumentries;
            int errorlength = stiff_approx_sparse.num_global_entries(i);
            std::vector<double> errorvalues(errorlength);
            std::vector<int> errorindices(errorlength);
            // int errorextractionstatus =
            stiff_approx_sparse.extract_global_row_copy(
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
            int sparselength = sparse.num_global_entries(i);
            std::vector<double> sparsevalues(sparselength);
            std::vector<int> sparseindices(sparselength);
            // int sparseextractionstatus =
            sparse.extract_global_row_copy(
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
            int approxlength = stiff_approx.num_global_entries(i);
            std::vector<double> approxvalues(approxlength);
            std::vector<int> approxindices(approxlength);
            // int approxextractionstatus =
            stiff_approx.extract_global_row_copy(
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

          if (abs(error) > abs(error_max)) error_max = abs(error);
          if (abs(error_ij) > abs(abs_error_max)) abs_error_max = abs(error_ij);

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
    std::cout << "finite difference check successful, max. rel. error: " << error_max
              << "  (max. abs. error: " << abs_error_max << ")" << std::endl;
    std::cout << "******************finite difference check done***************\n\n" << std::endl;
  }
  else
    FOUR_C_THROW("PoroFDCheck failed in step: {}, iter: {}", step(), iter_);
}

void PoroElast::Monolithic::evaluate_condition(
    Core::LinAlg::SparseOperator& Sysmat, PoroElast::Coupltype coupltype)
{
  nopen_handle_->clear(coupltype);
  std::shared_ptr<Core::LinAlg::SparseMatrix> ConstraintMatrix =
      nopen_handle_->constraint_matrix(coupltype);
  std::shared_ptr<Core::LinAlg::SparseMatrix> struct_vel_constraint_matrix =
      nopen_handle_->struct_vel_constraint_matrix(coupltype);

  // evaluate condition on elements and assemble matrices
  fluid_field()->evaluate_no_penetration_cond(nopen_handle_->rhs(), ConstraintMatrix,
      struct_vel_constraint_matrix, nopen_handle_->cond_vector(), *nopen_handle_->cond_ids(),
      coupltype);

  if (coupltype == PoroElast::fluidfluid)  // fluid fluid part
  {
    ConstraintMatrix->complete();
    nopen_handle_->build_no_penetration_map(
        fluid_field()->discretization()->get_comm(), fluid_field()->dof_row_map());
  }
  else  // fluid structure part
  {
    // double timescale = fluid_field()->TimeScaling();
    double timescale = fluid_field()->residual_scaling();
    struct_vel_constraint_matrix->scale(timescale);
    struct_vel_constraint_matrix->complete(structure_field()->system_matrix()->range_map(),
        fluid_field()->system_matrix()->range_map());
    ConstraintMatrix->add(*struct_vel_constraint_matrix, false, 1.0, 1.0);
    ConstraintMatrix->complete(structure_field()->system_matrix()->range_map(),
        fluid_field()->system_matrix()->range_map());
  }

  const std::shared_ptr<const Core::LinAlg::Map>& nopenetrationmap =
      nopen_handle_->extractor()->map(1);
  const std::shared_ptr<const Core::LinAlg::Map>& othermap = nopen_handle_->extractor()->map(0);
  ConstraintMatrix->apply_dirichlet(*othermap, false);
  Sysmat.apply_dirichlet(*nopenetrationmap, false);
  Sysmat.un_complete();
  Sysmat.add(*ConstraintMatrix, false, 1.0, 1.0);
}

std::shared_ptr<Core::LinAlg::SparseMatrix> PoroElast::Monolithic::struct_fluid_coupling_matrix()
{
  std::shared_ptr<Core::LinAlg::SparseMatrix> sparse =
      std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(k_sf_);
  if (sparse == nullptr) FOUR_C_THROW("cast to Core::LinAlg::SparseMatrix failed!");

  return sparse;
}

std::shared_ptr<Core::LinAlg::SparseMatrix> PoroElast::Monolithic::fluid_struct_coupling_matrix()
{
  std::shared_ptr<Core::LinAlg::SparseMatrix> sparse =
      std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(k_fs_);
  if (sparse == nullptr) FOUR_C_THROW("cast to Core::LinAlg::SparseMatrix failed!");

  return sparse;
}

std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase>
PoroElast::Monolithic::struct_fluid_coupling_block_matrix()
{
  std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> blocksparse =
      std::dynamic_pointer_cast<Core::LinAlg::BlockSparseMatrixBase>(k_sf_);
  if (blocksparse == nullptr) FOUR_C_THROW("cast to Core::LinAlg::BlockSparseMatrixBase failed!");

  return blocksparse;
}


std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase>
PoroElast::Monolithic::fluid_struct_coupling_block_matrix()
{
  std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> blocksparse =
      std::dynamic_pointer_cast<Core::LinAlg::BlockSparseMatrixBase>(k_fs_);
  if (blocksparse == nullptr) FOUR_C_THROW("cast to Core::LinAlg::BlockSparseMatrixBase failed!");

  return blocksparse;
}

void PoroElast::Monolithic::setup_newton()
{
  // initialise equilibrium loop and norms
  iter_ = 1;
  normrhs_ = 0.0;
  norminc_ = 0.0;
  normrhsfluid_ = 0.0;
  normincfluid_ = 0.0;
  normrhsfluidvel_ = 0.0;
  normincfluidvel_ = 0.0;
  normrhsfluidpres_ = 0.0;
  normincfluidpres_ = 0.0;
  normrhsstruct_ = 0.0;
  normincstruct_ = 0.0;
  normrhsporo_ = 0.0;
  normincporo_ = 0.0;

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

void PoroElast::Monolithic::build_convergence_norms()
{
  //------------------------------------------------------------ build residual force norms
  normrhs_ = Utils::calculate_vector_norm(vectornormfres_, *rhs_);
  std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_s;
  std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_f;
  std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_fvel;
  std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_fpres;

  // process structure unknowns of the first field
  rhs_s = extractor()->extract_vector(*rhs_, 0);
  // process fluid unknowns of the second field
  rhs_f = extractor()->extract_vector(*rhs_, 1);
  rhs_fvel = fluid_field()->extract_velocity_part(rhs_f);
  rhs_fpres = fluid_field()->extract_pressure_part(rhs_f);

  if (porosity_dof_)
  {
    std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_poro =
        porosity_splitter_->extract_cond_vector(*rhs_s);
    std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_sdisp =
        porosity_splitter_->extract_other_vector(*rhs_s);

    normrhsstruct_ = Utils::calculate_vector_norm(vectornormfres_, *rhs_sdisp);
    normrhsporo_ = Utils::calculate_vector_norm(vectornormfres_, *rhs_poro);
  }
  else
    normrhsstruct_ = Utils::calculate_vector_norm(vectornormfres_, *rhs_s);

  normrhsfluid_ = Utils::calculate_vector_norm(vectornormfres_, *rhs_f);
  normrhsfluidvel_ = Utils::calculate_vector_norm(vectornormfres_, *rhs_fvel);
  normrhsfluidpres_ = Utils::calculate_vector_norm(vectornormfres_, *rhs_fpres);


  //------------------------------------------------------------- build residual increment norms
  iterinc_->norm_2(&norminc_);

  // displacement and fluid velocity & pressure incremental vector
  std::shared_ptr<const Core::LinAlg::Vector<double>> interincs;
  std::shared_ptr<const Core::LinAlg::Vector<double>> interincf;
  std::shared_ptr<const Core::LinAlg::Vector<double>> interincfvel;
  std::shared_ptr<const Core::LinAlg::Vector<double>> interincfpres;
  // process structure unknowns of the first field
  interincs = extractor()->extract_vector(*iterinc_, 0);
  // process fluid unknowns of the second field
  interincf = extractor()->extract_vector(*iterinc_, 1);
  interincfvel = fluid_field()->extract_velocity_part(interincf);
  interincfpres = fluid_field()->extract_pressure_part(interincf);

  if (porosity_dof_)
  {
    std::shared_ptr<const Core::LinAlg::Vector<double>> interincporo =
        porosity_splitter_->extract_cond_vector(*interincs);
    std::shared_ptr<const Core::LinAlg::Vector<double>> interincsdisp =
        porosity_splitter_->extract_other_vector(*interincs);

    normincstruct_ = Utils::calculate_vector_norm(vectornorminc_, *interincsdisp);
    normincporo_ = Utils::calculate_vector_norm(vectornorminc_, *interincporo);
  }
  else
    normincstruct_ = Utils::calculate_vector_norm(vectornorminc_, *interincs);

  normincfluid_ = Utils::calculate_vector_norm(vectornorminc_, *interincf);
  normincfluidvel_ = Utils::calculate_vector_norm(vectornorminc_, *interincfvel);
  normincfluidpres_ = Utils::calculate_vector_norm(vectornorminc_, *interincfpres);
}

bool PoroElast::Monolithic::setup_solver()
{
  //  solver
  // create a linear solver
  // get dynamic section of poroelasticity
  const Teuchos::ParameterList& poroelastdyn =
      Global::Problem::instance()->poroelast_dynamic_params();
  // get the solver number used for linear poroelasticity solver
  const int linsolvernumber = poroelastdyn.get<int>("LINEAR_SOLVER");
  // check if the poroelasticity solver has a valid solver number
  if (linsolvernumber == (-1))
  {
    FOUR_C_THROW(
        "no linear solver defined for poroelasticity. Please set LINEAR_SOLVER in POROELASTICITY "
        "DYNAMIC to a valid number!");
  }
  const Teuchos::ParameterList& solverparams =
      Global::Problem::instance()->solver_params(linsolvernumber);
  const auto solvertype =
      Teuchos::getIntegralValue<Core::LinearSolver::SolverType>(solverparams, "SOLVER");

  directsolve_ = (solvertype == Core::LinearSolver::SolverType::umfpack or
                  solvertype == Core::LinearSolver::SolverType::superlu);

  if (directsolve_)
  {
    solver_ = std::make_shared<Core::LinAlg::Solver>(solverparams, get_comm(),
        Global::Problem::instance()->solver_params_callback(),
        Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
            Global::Problem::instance()->io_params(), "VERBOSITY"));
  }
  else
    // create a linear solver
    create_linear_solver();

  // Get the parameters for the Newton iteration
  itermax_ = poroelastdyn.get<int>("ITEMAX");
  itermin_ = poroelastdyn.get<int>("ITEMIN");
  normtypeinc_ = Teuchos::getIntegralValue<PoroElast::ConvNorm>(poroelastdyn, "NORM_INC");
  normtypefres_ = Teuchos::getIntegralValue<PoroElast::ConvNorm>(poroelastdyn, "NORM_RESF");
  combincfres_ = Teuchos::getIntegralValue<PoroElast::BinaryOp>(poroelastdyn, "NORMCOMBI_RESFINC");
  vectornormfres_ =
      Teuchos::getIntegralValue<PoroElast::VectorNorm>(poroelastdyn, "VECTORNORM_RESF");
  vectornorminc_ = Teuchos::getIntegralValue<PoroElast::VectorNorm>(poroelastdyn, "VECTORNORM_INC");

  tolinc_ = poroelastdyn.get<double>("TOLINC_GLOBAL");
  tolfres_ = poroelastdyn.get<double>("TOLRES_GLOBAL");

  tolinc_struct_ = poroelastdyn.get<double>("TOLINC_DISP");
  tolinc_velocity_ = poroelastdyn.get<double>("TOLINC_VEL");
  tolinc_pressure_ = poroelastdyn.get<double>("TOLINC_PRES");
  tolinc_porosity_ = poroelastdyn.get<double>("TOLINC_PORO");

  tolfres_struct_ = poroelastdyn.get<double>("TOLRES_DISP");
  tolfres_velocity_ = poroelastdyn.get<double>("TOLRES_VEL");
  tolfres_pressure_ = poroelastdyn.get<double>("TOLRES_PRES");
  tolfres_porosity_ = poroelastdyn.get<double>("TOLRES_PORO");

  return true;
}

std::shared_ptr<Core::LinAlg::SparseMatrix> PoroElast::Monolithic::system_matrix()
{
  return systemmatrix_->merge();
}

void PoroElast::Monolithic::increment_poro_iter() { iter_ += 1; }

void PoroElast::Monolithic::update_poro_iterinc(const Core::LinAlg::Vector<double>& poroinc)
{
  iterinc_->put_scalar(0.0);
  iterinc_->update(1.0, poroinc, 0.0);
}

void PoroElast::Monolithic::clear_poro_iterinc() { iterinc_->put_scalar(0.0); }

void PoroElast::Monolithic::aitken()
{
  // initialise increment vector with solution of last iteration (i)
  // update del_ with current residual vector
  // difference of last two solutions
  if (del_ == nullptr)  // first iteration, itnum==1
  {
    del_ = Core::LinAlg::create_vector(*dof_row_map(), true);
    delhist_ = Core::LinAlg::create_vector(*dof_row_map(), true);
    del_->put_scalar(1.0e20);
    delhist_->put_scalar(0.0);
  }

  // calculate difference of current (i+1) and old (i) residual vector
  // delhist = ( r^{i+1}_{n+1} - r^i_{n+1} )
  // update history vector old increment r^i_{n+1}
  delhist_->update(1.0, *del_, 0.0);         // r^i_{n+1}
  delhist_->update(1.0, *iterinc_, (-1.0));  // update r^{i+1}_{n+1}


  // del_ = r^{i+1}_{n+1} = T^{i+1}_{n+1} - T^{i}_{n+1}
  del_->update(1.0, *iterinc_, 0.0);
  // den = |r^{i+1} - r^{i}|^2
  double den = 0.0;
  delhist_->norm_2(&den);
  // calculate dot product
  // dot = delhist_ . del_ = ( r^{i+1}_{n+1} - r^i_{n+1} )^T . r^{i+1}_{n+1}
  double top = 0.0;
  delhist_->dot(*del_, &top);

  // mu_: Aikten factor in Mok's version
  // mu_: relaxation parameter in Irons & Tuck
  // mu^{i+1} = mu^i + (mu^i -1) . (r^{i+1} - r^i)^T . (-r^{i+1}) / |r^{i+1} - r^{i}|^2
  // top = ( r^{i+1} - r^i )^T . r^{i+1} --> use -top
  // Uli's implementation: mu_ = mu_ + (mu_ - 1.0) * top / (den*den). with '-' included in top
  mu_ = mu_ + (mu_ - 1) * (-top) / (den * den);

  iterinc_->scale(1.0 - mu_);
}

[[maybe_unused]] void PoroElast::Monolithic::aitken_reset()
{
  if (del_ == nullptr)  // first iteration, itnum==1
  {
    del_ = Core::LinAlg::create_vector(*dof_row_map(), true);
    delhist_ = Core::LinAlg::create_vector(*dof_row_map(), true);
  }
  del_->put_scalar(1.0e20);
  delhist_->put_scalar(0.0);
  mu_ = 0.0;
}

const Core::LinAlg::Map& PoroElast::Monolithic::fluid_range_map()
{
  return fluid_field()->system_matrix()->range_map();
}

const Core::LinAlg::Map& PoroElast::Monolithic::fluid_domain_map()
{
  return fluid_field()->system_matrix()->domain_map();
}

const Core::LinAlg::Map& PoroElast::Monolithic::structure_domain_map()
{
  return structure_field()->domain_map();
}

std::shared_ptr<const Core::LinAlg::Map> PoroElast::Monolithic::dof_row_map()
{
  return blockrowdofmap_->full_map();
}

std::shared_ptr<const Core::LinAlg::Map> PoroElast::Monolithic::dof_row_map_structure()
{
  return blockrowdofmap_->map(0);
}

std::shared_ptr<const Core::LinAlg::Map> PoroElast::Monolithic::dof_row_map_fluid()
{
  return blockrowdofmap_->map(1);
}

void PoroElast::Monolithic::recover_lagrange_multiplier_after_newton_step(
    std::shared_ptr<const Core::LinAlg::Vector<double>> iterinc)
{
  // clean up as soon as old time integration is unused!
  if (oldstructimint_)
  {
    if (structure_field()->meshtying_contact_bridge() != nullptr)
    {
      if (structure_field()->meshtying_contact_bridge()->have_contact() && !nit_contact_)
      {
        // Recover structural contact lagrange multiplier !!! For Poro & FPSI Problems this is
        // deactivated in the Structure!!!

        CONTACT::LagrangeStrategyPoro& costrategy = static_cast<CONTACT::LagrangeStrategyPoro&>(
            structure_field()->meshtying_contact_bridge()->contact_manager()->get_strategy());

        // displacement and fluid velocity & pressure incremental vector
        std::shared_ptr<const Core::LinAlg::Vector<double>> s_iterinc;
        std::shared_ptr<const Core::LinAlg::Vector<double>> f_iterinc;
        extract_field_vectors(iterinc, s_iterinc, f_iterinc);

        // RecoverStructuralLM
        std::shared_ptr<Core::LinAlg::Vector<double>> tmpsx =
            std::make_shared<Core::LinAlg::Vector<double>>(*s_iterinc);
        std::shared_ptr<Core::LinAlg::Vector<double>> tmpfx =
            std::make_shared<Core::LinAlg::Vector<double>>(*f_iterinc);

        costrategy.recover_coupled(tmpsx, tmpfx);
        if (no_penetration_) costrategy.recover_poro_no_pen(tmpsx, tmpfx);
      }
      else if (structure_field()->meshtying_contact_bridge()->have_meshtying())
      {  // if meshtying  //h.Willmann

        CONTACT::PoroMtLagrangeStrategy& costrategy = static_cast<CONTACT::PoroMtLagrangeStrategy&>(
            structure_field()->meshtying_contact_bridge()->mt_manager()->get_strategy());

        // displacement and fluid velocity & pressure incremental vector
        std::shared_ptr<const Core::LinAlg::Vector<double>> s_iterinc;
        std::shared_ptr<const Core::LinAlg::Vector<double>> f_iterinc;
        extract_field_vectors(iterinc, s_iterinc, f_iterinc);

        Core::LinAlg::Vector<double> tmpfx(*f_iterinc);

        // Recover part of LM stemming from offdiagonal coupling matrix
        costrategy.recover_coupling_matrix_partof_lmp(tmpfx);
      }
    }
  }
}

void PoroElast::Monolithic::set_poro_contact_states()
{
  // clean up as soon as old time integration is unused!
  if (oldstructimint_)
  {
    if (structure_field()->meshtying_contact_bridge() != nullptr)
    {
      if (structure_field()->meshtying_contact_bridge()->have_contact())
      {
        if (!nit_contact_)  // Lagmult contact
        {
          CONTACT::LagrangeStrategyPoro& costrategy = static_cast<CONTACT::LagrangeStrategyPoro&>(
              structure_field()->meshtying_contact_bridge()->contact_manager()->get_strategy());
          std::shared_ptr<Core::LinAlg::Vector<double>> fvel =
              std::make_shared<Core::LinAlg::Vector<double>>(
                  *fluid_field()->extract_velocity_part(fluid_field()->velnp()));
          fvel = fluid_structure_coupling().slave_to_master(*fvel);
          costrategy.set_state(Mortar::state_fvelocity, *fvel);

          // To get pressure dofs into first structural component!!! - any idea for nice
          // implementation?
          std::shared_ptr<const Core::LinAlg::Vector<double>> fpres =
              fluid_field()->extract_pressure_part(fluid_field()->velnp());
          std::shared_ptr<Core::LinAlg::Vector<double>> modfpres =
              std::make_shared<Core::LinAlg::Vector<double>>(
                  *fluid_field()->velocity_row_map(), true);

          int* mygids = fpres->get_map().my_global_elements();
          const double* val = fpres->get_values();
          const int ndim = Global::Problem::instance()->n_dim();
          for (int i = 0; i < fpres->local_length(); ++i)
          {
            int gid = mygids[i] - ndim;
            modfpres->replace_global_values(1, &val[i], &gid);
          }

          modfpres = fluid_structure_coupling().slave_to_master(*modfpres);
          costrategy.set_state(Mortar::state_fpressure, *modfpres);

          Core::LinAlg::Vector<double> dis(*structure_field()->dispnp());
          costrategy.set_parent_state(Mortar::StateType::state_new_displacement, dis,
              *structure_field()->discretization());  // add displacements of the parent element!!!
        }
        else
        {
          CONTACT::NitscheStrategyPoro& costrategy = dynamic_cast<CONTACT::NitscheStrategyPoro&>(
              structure_field()->meshtying_contact_bridge()->contact_manager()->get_strategy());
          std::shared_ptr<Core::FE::Discretization> dis =
              Global::Problem::instance()->get_dis("porofluid");
          if (dis == nullptr) FOUR_C_THROW("didn't get my discretization");
          costrategy.set_parent_state(Mortar::state_fvelocity, *fluid_field()->velnp(), *dis);
        }
      }
    }
  }
  else if (nit_contact_)  // new time integration with nitsche contact
  {
    auto& model_evaluator_contact = dynamic_cast<Solid::ModelEvaluator::Contact&>(
        structure_field()->model_evaluator(Inpar::Solid::model_contact));

    std::shared_ptr<CONTACT::NitscheStrategyPoro> contact_strategy_nitsche_poro =
        std::dynamic_pointer_cast<CONTACT::NitscheStrategyPoro>(
            model_evaluator_contact.strategy_ptr());

    if (contact_strategy_nitsche_poro != nullptr)
    {
      std::shared_ptr<Core::FE::Discretization> poro_dis =
          Global::Problem::instance()->get_dis("porofluid");
      if (poro_dis == nullptr) FOUR_C_THROW("didn't get my poro discretization");

      contact_strategy_nitsche_poro->set_parent_state(
          Mortar::state_fvelocity, *fluid_field()->velnp(), *poro_dis);

      std::shared_ptr<Core::FE::Discretization> struct_dis =
          Global::Problem::instance()->get_dis("structure");
      if (struct_dis == nullptr) FOUR_C_THROW("didn't get my structure discretization");

      contact_strategy_nitsche_poro->set_parent_state(
          Mortar::state_svelocity, *structure_field()->velnp(), *struct_dis);
    }
  }
}

void PoroElast::Monolithic::eval_poro_mortar()
{
  // clean up as soon as old time integration is unused!
  if (oldstructimint_)
  {
    if (structure_field()->meshtying_contact_bridge() != nullptr)
    {
      if (structure_field()->meshtying_contact_bridge()->have_contact())
      {
        if (!nit_contact_)  // Lagmult contact
        {
          CONTACT::LagrangeStrategyPoro& costrategy = static_cast<CONTACT::LagrangeStrategyPoro&>(
              structure_field()->meshtying_contact_bridge()->contact_manager()->get_strategy());
          //---Modify coupling matrix k_sf

          // Get matrix block!
          std::shared_ptr<Core::LinAlg::SparseOperator> k_ss =
              std::make_shared<Core::LinAlg::SparseMatrix>(systemmatrix_->matrix(0, 0));
          std::shared_ptr<Core::LinAlg::SparseOperator> k_sf =
              std::make_shared<Core::LinAlg::SparseMatrix>(systemmatrix_->matrix(0, 1));
          std::shared_ptr<Core::LinAlg::Vector<double>> rhs_s =
              extractor()->extract_vector(*rhs_, 0);

          // Evaluate Poro Contact Condensation for K_ss, K_sf
          costrategy.apply_force_stiff_cmt_coupled(
              structure_field()->write_access_dispnp(), k_ss, k_sf, rhs_s, step(), iter_, false);

          // Assign modified matrixes & vectors
          systemmatrix_->assign(0, 0, Core::LinAlg::DataAccess::Copy,
              *std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(k_ss));
          systemmatrix_->assign(0, 1, Core::LinAlg::DataAccess::Copy,
              *std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(k_sf));
          extractor()->insert_vector(*rhs_s, 0, *rhs_);

          //---Modify fluid matrix, coupling matrix k_fs and rhs of fluid
          if (no_penetration_)
          {
            //---Initialize Poro Contact
            costrategy.poro_initialize(fluid_structure_coupling(),
                *fluid_field()->dof_row_map());  // true stands for the no_penetration condition !!!
            // Get matrix blocks & rhs vector!
            std::shared_ptr<Core::LinAlg::SparseMatrix> f =
                std::make_shared<Core::LinAlg::SparseMatrix>(systemmatrix_->matrix(1, 1));
            std::shared_ptr<Core::LinAlg::SparseMatrix> k_fs =
                std::make_shared<Core::LinAlg::SparseMatrix>(systemmatrix_->matrix(1, 0));

            std::shared_ptr<Core::LinAlg::Vector<double>> frhs =
                extractor()->extract_vector(*rhs_, 1);

            // Evaluate Poro No Penetration Contact Condensation
            costrategy.evaluate_poro_no_pen_contact(k_fs, f, frhs);

            // Assign modified matrixes & vectors
            systemmatrix_->assign(1, 1, Core::LinAlg::DataAccess::Copy, *f);
            systemmatrix_->assign(1, 0, Core::LinAlg::DataAccess::Copy, *k_fs);

            extractor()->insert_vector(*frhs, 1, *rhs_);
          }
        }
        else  // add nitsche contributions to sysmat and rhs
        {
          CONTACT::NitscheStrategyPoro* pstrat = dynamic_cast<CONTACT::NitscheStrategyPoro*>(
              &structure_field()->meshtying_contact_bridge()->contact_manager()->get_strategy());

          systemmatrix_->un_complete();
          systemmatrix_->matrix(0, 1).add(
              *pstrat->get_matrix_block_ptr(CONTACT::MatBlockType::displ_porofluid), false, 1.0,
              1.0);
          systemmatrix_->matrix(1, 1).add(
              *pstrat->get_matrix_block_ptr(CONTACT::MatBlockType::porofluid_porofluid), false, 1.0,
              1.0);
          systemmatrix_->matrix(1, 0).add(
              *pstrat->get_matrix_block_ptr(CONTACT::MatBlockType::porofluid_displ), false, 1.0,
              1.0);
          systemmatrix_->complete();

          extractor()->add_vector(
              *pstrat->get_rhs_block_ptr(CONTACT::VecBlockType::porofluid), 1, *rhs_);
        }
      }
      else if (structure_field()->meshtying_contact_bridge()->have_meshtying())
      {  // if meshtying  //h.Willmann

        CONTACT::PoroMtLagrangeStrategy& costrategy = static_cast<CONTACT::PoroMtLagrangeStrategy&>(
            structure_field()->meshtying_contact_bridge()->mt_manager()->get_strategy());


        //---Modify coupling matrix k_sf

        // Get matrix block!
        std::shared_ptr<Core::LinAlg::SparseMatrix> k_sf =
            std::make_shared<Core::LinAlg::SparseMatrix>(systemmatrix_->matrix(0, 1));

        // initialize poro meshtying
        costrategy.initialize_poro_mt(k_sf);

        // Evaluate Poro Meshtying Condensation for K_sf
        costrategy.evaluate_meshtying_poro_off_diag(k_sf);

        // Assign modified matrix
        systemmatrix_->assign(0, 1, Core::LinAlg::DataAccess::Copy, *k_sf);
      }
    }
  }
  else if (nit_contact_)  // new time integration with nitsche contact
  {
    auto& model_evaluator_contact = dynamic_cast<Solid::ModelEvaluator::Contact&>(
        structure_field()->model_evaluator(Inpar::Solid::model_contact));

    std::shared_ptr<CONTACT::NitscheStrategyPoro> contact_strategy_nitsche_poro =
        std::dynamic_pointer_cast<CONTACT::NitscheStrategyPoro>(
            model_evaluator_contact.strategy_ptr());

    if (contact_strategy_nitsche_poro != nullptr)
    {
      systemmatrix_->un_complete();
      systemmatrix_->matrix(0, 1).add(*contact_strategy_nitsche_poro->get_matrix_block_ptr(
                                          CONTACT::MatBlockType::displ_porofluid),
          false, 1.0, 1.0);
      systemmatrix_->matrix(1, 1).add(*contact_strategy_nitsche_poro->get_matrix_block_ptr(
                                          CONTACT::MatBlockType::porofluid_porofluid),
          false, 1.0, 1.0);
      systemmatrix_->matrix(1, 0).add(*contact_strategy_nitsche_poro->get_matrix_block_ptr(
                                          CONTACT::MatBlockType::porofluid_displ),
          false, 1.0, 1.0);
      systemmatrix_->complete();

      extractor()->add_vector(
          *contact_strategy_nitsche_poro->get_rhs_block_ptr(CONTACT::VecBlockType::porofluid), 1,
          *rhs_);
    }
  }
}

void PoroElast::Monolithic::build_combined_dbc_map()
{
  const std::shared_ptr<const Core::LinAlg::Map> scondmap =
      structure_field()->get_dbc_map_extractor()->cond_map();
  const std::shared_ptr<const Core::LinAlg::Map> fcondmap =
      fluid_field()->get_dbc_map_extractor()->cond_map();
  combinedDBCMap_ = Core::LinAlg::merge_map(scondmap, fcondmap, false);
}

void PoroElast::Monolithic::read_restart(const int step)
{
  // call base class
  PoroElast::PoroBase::read_restart(step);

  // set states for porous contact
  // apply current velocity of fluid to ContactManager if contact problem
  if (no_penetration_) set_poro_contact_states();
}

FOUR_C_NAMESPACE_CLOSE

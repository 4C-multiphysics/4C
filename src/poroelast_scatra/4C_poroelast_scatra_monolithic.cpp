// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_poroelast_scatra_monolithic.hpp"

#include "4C_adapter_fld_poro.hpp"
#include "4C_adapter_str_fpsiwrapper.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_assemblestrategy.hpp"
#include "4C_fluid_ele_action.hpp"
#include "4C_global_data.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_linear_solver_method.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_poroelast_scatra_utils.hpp"
#include "4C_scatra_ele_action.hpp"
#include "4C_scatra_timint_implicit.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_parameter_list.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 |                                                         vuong 08/13  |
 *----------------------------------------------------------------------*/
PoroElastScaTra::PoroScatraMono::PoroScatraMono(
    MPI_Comm comm, const Teuchos::ParameterList& timeparams)
    : PoroScatraBase(comm, timeparams),
      printscreen_(true),  // ADD INPUT PARAMETER
      printiter_(true),    // ADD INPUT PARAMETER
      timer_("PoroScatraMonoSolve", true),
      iterinc_(nullptr),
      zeros_(nullptr),
      systemmatrix_(nullptr),
      rhs_(nullptr),
      blockrowdofmap_(nullptr),
      directsolve_(true)
{
  const Teuchos::ParameterList& sdynparams =
      Global::Problem::instance()->structural_dynamic_params();

  // some solver parameters are red form the structure dynamic list (this is not the best way to do
  // it ...)
  solveradapttol_ = (sdynparams.get<bool>("ADAPTCONV"));
  solveradaptolbetter_ = (sdynparams.get<double>("ADAPTCONV_BETTER"));

  blockrowdofmap_ = std::make_shared<Core::LinAlg::MultiMapExtractor>();
}

/*----------------------------------------------------------------------*
 |                                                         vuong 08/13  |
 *----------------------------------------------------------------------*/
void PoroElastScaTra::PoroScatraMono::timeloop()
{
  while (not_finished())
  {
    do_time_step();
  }
}

/*----------------------------------------------------------------------*
 |                                                         vuong 08/13  |
 *----------------------------------------------------------------------*/
void PoroElastScaTra::PoroScatraMono::do_time_step()
{
  // counter and print header
  // predict solution of both field (call the adapter)
  prepare_time_step();

  // Newton-Raphson iteration
  solve();

  // calculate stresses, strains, energies
  prepare_output();

  // update all single field solvers
  update();

  // write output to screen and files
  output();
}

/*----------------------------------------------------------------------*
 |                                                         vuong 08/13  |
 *----------------------------------------------------------------------*/
void PoroElastScaTra::PoroScatraMono::read_restart(int restart)
{
  // read restart information, set vectors and variables
  // (Note that dofmaps might have changed in a redistribution call!)
  if (restart)
  {
    set_scatra_solution();
    set_poro_solution();

    poro_field()->read_restart(restart);
    scatra_field()->read_restart(restart);

    // in case of submeshes, we need to rebuild the subproxies, also (they are reset during restart)
    if (poro_field()->has_submeshes())
      replace_dof_sets(structure_field()->discretization(), fluid_field()->discretization(),
          scatra_field()->discretization());

    // the variables need to be set on other field
    set_scatra_solution();
    set_poro_solution();

    // second restart needed due to two way coupling.
    scatra_field()->read_restart(restart);
    poro_field()->read_restart(restart);

    // in case of submeshes, we need to rebuild the subproxies, also (they are reset during restart)
    if (poro_field()->has_submeshes())
      replace_dof_sets(structure_field()->discretization(), fluid_field()->discretization(),
          scatra_field()->discretization());

    set_time_step(poro_field()->time(), restart);

    // Material pointers to other field were deleted during read_restart().
    // They need to be reset.
    PoroElast::Utils::set_material_pointers_matching_grid(
        *poro_field()->structure_field()->discretization(), *scatra_field()->discretization());
    PoroElast::Utils::set_material_pointers_matching_grid(
        *poro_field()->fluid_field()->discretization(), *scatra_field()->discretization());
  }
}

/*----------------------------------------------------------------------*/
// prepare time step
/*----------------------------------------------------------------------*/
void PoroElastScaTra::PoroScatraMono::prepare_time_step(bool printheader)
{
  // the global control routine has its own time_ and step_ variables, as well as the single fields
  // keep them in sinc!
  increment_time_and_step();

  if (printheader) print_header();

  set_poro_solution();
  scatra_field()->prepare_time_step();
  // set structure-based scalar transport values
  set_scatra_solution();

  poro_field()->prepare_time_step();
  set_poro_solution();
}

/*----------------------------------------------------------------------*
 |                                                         vuong 08/13  |
 *----------------------------------------------------------------------*/
void PoroElastScaTra::PoroScatraMono::prepare_output()
{
  constexpr bool force_prepare = false;
  poro_field()->prepare_output(force_prepare);
}

/*----------------------------------------------------------------------*
 |                                                         vuong 08/13  |
 *----------------------------------------------------------------------*/
void PoroElastScaTra::PoroScatraMono::output()
{
  poro_field()->output();
  scatra_field()->check_and_write_output_and_restart();
}

/*----------------------------------------------------------------------*
 |                                                         vuong 08/13  |
 *----------------------------------------------------------------------*/
void PoroElastScaTra::PoroScatraMono::update()
{
  poro_field()->update();

  scatra_field()->update();
  scatra_field()->evaluate_error_compared_to_analytical_sol();
}

/*----------------------------------------------------------------------*
 |                                                         vuong 08/13  |
 *----------------------------------------------------------------------*/
void PoroElastScaTra::PoroScatraMono::solve()
{
  // we do a Newton-Raphson iteration here.
  // the specific time integration has set the following
  // --> On #rhs_ is the positive force residuum
  // --> On #systemmatrix_ is the effective dynamic tangent matrix

  //-------------------------------------- initialize variables needed in newton loop
  iter_ = 1;
  normrhs_ = 0.0;
  norminc_ = 0.0;

  // incremental solution vector with length of all dofs
  iterinc_ = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map(), true);
  iterinc_->put_scalar(0.0);

  // a zero vector of full length
  zeros_ = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map(), true);
  zeros_->put_scalar(0.0);

  //---------------------------------------------- iteration loop

  // equilibrium iteration loop (loop over k)
  while (((not converged()) and (iter_ <= itermax_)) or (iter_ <= itermin_))
  {
    timer_.reset();
    Teuchos::Time timer("PoroScatraMonoSolve", true);

    // compute residual forces #rhs_ and tangent #tang_
    // whose components are globally oriented
    // build linear system stiffness matrix and rhs/force residual for each
    // field, here e.g. for structure field: field want the iteration increment
    // 1.) Update(iterinc_),
    // 2.) evaluate_force_stiff_residual(),
    // 3.) prepare_system_for_newton_solve()
    evaluate(iterinc_);

    // check whether we have a sanely filled tangent matrix
    if (not systemmatrix_->filled())
    {
      FOUR_C_THROW("Effective tangent matrix must be filled here");
    }

    // create full monolithic rhs vector
    setup_rhs(iter_ == 1);

    // fd_check();

    // build norms
    build_convergence_norms();

    if ((not converged()) or combincfres_ == PoroElast::bop_or)
    {
      // (Newton-ready) residual with blanked Dirichlet DOFs (see adapter_timint!)
      // is done in prepare_system_for_newton_solve() within evaluate(iterinc_)
      linear_solve();
      // cout << "  time for Evaluate linear_solve: " << timer.totalElapsedTime(true) << "\n";
      // timer.reset();

      // reset solver tolerance
      solver_->reset_tolerance();

      // build norms
      build_convergence_norms();
    }

    // print stuff
    print_newton_iter();

    // increment equilibrium loop index
    iter_ += 1;
  }  // end equilibrium loop

  //---------------------------------------------- iteration loop

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

}  // Solve()

/*----------------------------------------------------------------------*
 | evaluate the single fields                              vuong 01/12   |
 *----------------------------------------------------------------------*/
void PoroElastScaTra::PoroScatraMono::evaluate(
    std::shared_ptr<const Core::LinAlg::Vector<double>> stepinc)
{
  TEUCHOS_FUNC_TIME_MONITOR("PoroScatraMono::Monolithic::Evaluate");

  // displacement and fluid velocity & pressure incremental vector
  std::shared_ptr<const Core::LinAlg::Vector<double>> porostructinc;
  std::shared_ptr<const Core::LinAlg::Vector<double>> porofluidinc;
  std::shared_ptr<const Core::LinAlg::Vector<double>> scatrainc;

  // if an increment vector exists
  if (stepinc != nullptr)
  {
    // process structure unknowns of the first field
    porostructinc = extractor()->extract_vector(*stepinc, 0);
    porofluidinc = extractor()->extract_vector(*stepinc, 1);

    // process fluid unknowns of the second field
    scatrainc = extractor()->extract_vector(*stepinc, 2);
  }

  // Newton update of the fluid field
  // update velocities and pressures before passed to the structural field
  //  update_iter_incrementally(fx),
  scatra_field()->update_iter(*scatrainc);

  // call all elements and assemble rhs and matrices
  /// poro field

  // structure Evaluate (builds tangent, residual and applies DBC)
  // apply current velocity and pressures to structure
  set_scatra_solution();

  // access poro problem to build poro-poro block
  poro_field()->evaluate(porostructinc, porofluidinc, iter_ == 1);

  /// scatra field

  // Scatra Evaluate
  // (builds tangent, residual and applies DBC and recent coupling values)
  set_poro_solution();

  // access ScaTra problem to build scatra-scatra block
  scatra_field()->prepare_linear_solve();

  // fill off diagonal blocks and build monolithic system matrix
  setup_system_matrix();
}

/*----------------------------------------------------------------------*
 | setup system (called in poro_dyn.cpp)                 vuong 01/12    |
 *----------------------------------------------------------------------*/
void PoroElastScaTra::PoroScatraMono::setup_system()
{
  // setup the poro subsystem first
  poro_field()->setup_system();

  // create combined map
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> vecSpaces;

  {
    // vecSpaces.push_back(poro_field()->dof_row_map());
    vecSpaces.push_back(poro_field()->dof_row_map_structure());
    vecSpaces.push_back(poro_field()->dof_row_map_fluid());
    const Core::LinAlg::Map* dofrowmapscatra = (scatra_field()->discretization())->dof_row_map(0);
    vecSpaces.push_back(Core::Utils::shared_ptr_from_ref(*dofrowmapscatra));
  }

  if (vecSpaces[0]->num_global_elements() == 0) FOUR_C_THROW("No poro structure equation. Panic.");
  if (vecSpaces[1]->num_global_elements() == 0) FOUR_C_THROW("No poro fluid equation. Panic.");
  if (vecSpaces[2]->num_global_elements() == 0) FOUR_C_THROW("No scatra equation. Panic.");

  // build dof row map of monolithic system
  set_dof_row_maps(vecSpaces);

  // build dbc map of monolithic system
  {
    const std::shared_ptr<const Core::LinAlg::Map> porocondmap = poro_field()->combined_dbc_map();
    const std::shared_ptr<const Core::LinAlg::Map> scatracondmap =
        scatra_field()->dirich_maps()->cond_map();
    std::shared_ptr<const Core::LinAlg::Map> dbcmap =
        Core::LinAlg::merge_map(porocondmap, scatracondmap, false);

    // Finally, create the global FSI Dirichlet map extractor
    dbcmaps_ = std::make_shared<Core::LinAlg::MapExtractor>(*dof_row_map(), dbcmap, true);
    if (dbcmaps_ == nullptr) FOUR_C_THROW("Creation of Dirichlet map extractor failed.");
  }

  // initialize Poroscatra-systemmatrix_
  systemmatrix_ =
      std::make_shared<Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(
          *extractor(), *extractor(), 81, false, true);

  {
    std::vector<std::shared_ptr<const Core::LinAlg::Map>> scatravecSpaces;
    const Core::LinAlg::Map* dofrowmapscatra = (scatra_field()->discretization())->dof_row_map(0);
    scatravecSpaces.push_back(Core::Utils::shared_ptr_from_ref(*dofrowmapscatra));
    scatrarowdofmap_.setup(*dofrowmapscatra, scatravecSpaces);
  }

  k_pss_ = std::make_shared<Core::LinAlg::SparseMatrix>(
      *(poro_field()->dof_row_map_structure()), 81, true, true);
  k_pfs_ = std::make_shared<Core::LinAlg::SparseMatrix>(*(poro_field()->dof_row_map_fluid()),
      //*(fluid_field()->dof_row_map()),
      81, true, true);

  k_sps_ = std::make_shared<Core::LinAlg::SparseMatrix>(
      *(scatra_field()->discretization()->dof_row_map()), 81, true, true);
  k_spf_ = std::make_shared<Core::LinAlg::SparseMatrix>(
      *(scatra_field()->discretization()->dof_row_map()),
      //*(fluid_field()->dof_row_map()),
      81, true, true);
}

/*----------------------------------------------------------------------*
 | setup RHS (like fsimon)                                 vuong 01/12  |
 *----------------------------------------------------------------------*/
void PoroElastScaTra::PoroScatraMono::setup_rhs(bool firstcall)
{
  // create full monolithic rhs vector
  if (rhs_ == nullptr) rhs_ = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map(), true);

  // fill the Poroelasticity rhs vector rhs_ with the single field rhss
  setup_vector(*rhs_, poro_field()->rhs(), scatra_field()->residual());
}

/*----------------------------------------------------------------------*
 | setup vector of the structure and fluid field            vuong 01/12|
 *----------------------------------------------------------------------*/
void PoroElastScaTra::PoroScatraMono::setup_vector(Core::LinAlg::Vector<double>& f,
    std::shared_ptr<const Core::LinAlg::Vector<double>> pv,
    std::shared_ptr<const Core::LinAlg::Vector<double>> sv)
{
  // extract dofs of the two fields
  // and put the poro/scatra field vector into the global vector f
  // noticing the block number

  //  std::shared_ptr<const Core::LinAlg::Vector<double>> psx;
  //  std::shared_ptr<const Core::LinAlg::Vector<double>> pfx;

  extractor()->insert_vector(*(poro_field()->extractor()->extract_vector(*pv, 0)), 0, f);
  extractor()->insert_vector(*(poro_field()->extractor()->extract_vector(*pv, 1)), 1, f);
  extractor()->insert_vector(*sv, 2, f);
}

/*----------------------------------------------------------------------*
 | setup system matrix of poroelasticity                   vuong 01/12  |
 *----------------------------------------------------------------------*/
void PoroElastScaTra::PoroScatraMono::setup_system_matrix()
{
  // set loma block matrix to zero
  systemmatrix_->zero();

  //----------------------------------------------------------------------
  // 1st diagonal block (upper left): poro weighting - poro solution
  //----------------------------------------------------------------------
  // get matrix block
  std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> mat_pp = poro_field()->block_system_matrix();

  // uncomplete matrix block (appears to be required in certain cases)
  mat_pp->un_complete();

  // assign matrix block
  systemmatrix_->assign(0, 0, Core::LinAlg::DataAccess::Share, mat_pp->matrix(0, 0));
  systemmatrix_->assign(0, 1, Core::LinAlg::DataAccess::Share, mat_pp->matrix(0, 1));
  systemmatrix_->assign(1, 0, Core::LinAlg::DataAccess::Share, mat_pp->matrix(1, 0));
  systemmatrix_->assign(1, 1, Core::LinAlg::DataAccess::Share, mat_pp->matrix(1, 1));

  //----------------------------------------------------------------------
  // 2nd diagonal block (lower right): scatra weighting - scatra solution
  //----------------------------------------------------------------------
  // get matrix block
  std::shared_ptr<Core::LinAlg::SparseMatrix> mat_ss = scatra_field()->system_matrix();

  // uncomplete matrix block (appears to be required in certain cases)
  mat_ss->un_complete();

  // assign matrix block
  systemmatrix_->assign(2, 2, Core::LinAlg::DataAccess::Share, *mat_ss);

  // complete scatra block matrix
  systemmatrix_->complete();

  //----------------------------------------------------------------------
  // 1st off-diagonal block (upper right): poro weighting - scatra solution
  //----------------------------------------------------------------------

  // evaluate off-diagonal matrix block in fluid
  evaluate_od_block_mat_poro();

  // k_ps_->Complete(mat_pp->DomainMap(),mat_ss->RangeMap());
  //  k_ps_->Complete();
  //
  //  std::shared_ptr<Core::LinAlg::SparseMatrix> k_ps_sparse = k_ps_->Merge();

  // uncomplete matrix block (appears to be required in certain cases)
  // k_ps_sparse->UnComplete();
  k_pss_->un_complete();
  k_pfs_->un_complete();

  // assign matrix block
  systemmatrix_->assign(0, 2, Core::LinAlg::DataAccess::Share, *(k_pss_));
  systemmatrix_->assign(1, 2, Core::LinAlg::DataAccess::Share, *(k_pfs_));

  //----------------------------------------------------------------------
  // 2nd off-diagonal block (lower left): scatra weighting - poro solution
  //----------------------------------------------------------------------

  // evaluate off-diagonal matrix block in scatra
  evaluate_od_block_mat_scatra();

  //  k_sp_->Complete();
  //
  //  std::shared_ptr<Core::LinAlg::SparseMatrix> k_sp_sparse = k_sp_->Merge();

  // uncomplete matrix block (appears to be required in certain cases)
  // k_sp_sparse->UnComplete();
  k_sps_->un_complete();
  k_spf_->un_complete();

  // assign matrix block
  systemmatrix_->assign(2, 0, Core::LinAlg::DataAccess::Share, *(k_sps_));
  systemmatrix_->assign(2, 1, Core::LinAlg::DataAccess::Share, *(k_spf_));

  // complete block matrix
  systemmatrix_->complete();
}

/*----------------------------------------------------------------------*
 | Solve linear Poroelast_SCATRAicity system                      vuong 01/12   |
 *----------------------------------------------------------------------*/
void PoroElastScaTra::PoroScatraMono::linear_solve()
{
  // Solve for inc_ = [disi_,tempi_]
  // Solve K_Teffdyn . IncX = -R  ===>  IncX_{n+1} with X=[d,T]
  // \f$x_{i+1} = x_i + \Delta x_i\f$
  Core::LinAlg::SolverParams solver_params;
  if (solveradapttol_ and (iter_ > 1))
  {
    solver_params.nonlin_tolerance = tolfres_;
    solver_params.nonlin_residual = normrhs_;
    solver_params.lin_tol_better = solveradaptolbetter_;
  }
  // apply Dirichlet BCs to system of equations
  iterinc_->put_scalar(0.0);  // Useful? depends on solver and more

  if (directsolve_)
  {
    // merge blockmatrix to SparseMatrix and solve
    std::shared_ptr<Core::LinAlg::SparseMatrix> sparse = systemmatrix_->merge();

    Core::LinAlg::apply_dirichlet_to_system(
        *sparse, *iterinc_, *rhs_, *zeros_, *combined_dbc_map());
    //  if ( Core::Communication::my_mpi_rank(Comm())==0 ) { cout << " DBC applied to system" <<
    //  endl; }

    // standard solver call
    solver_params.refactor = true;
    solver_params.reset = iter_ == 1;
    solver_->solve(sparse, iterinc_, rhs_, solver_params);
    //  if ( Core::Communication::my_mpi_rank(Comm())==0 ) { cout << " Solved" << endl; }
  }
  else
  {
    // in case of inclined boundary conditions
    // rotate systemmatrix_ using get_loc_sys_trafo()!=nullptr
    Core::LinAlg::apply_dirichlet_to_system(
        *systemmatrix_, *iterinc_, *rhs_, *zeros_, *combined_dbc_map());
    solver_params.refactor = true;
    solver_params.reset = iter_ == 1;
    solver_->solve(systemmatrix_, iterinc_, rhs_, solver_params);
  }
}

/*----------------------------------------------------------------------*
 | setup solver for monolithic system                    vuong 01/12     |
 *----------------------------------------------------------------------*/
bool PoroElastScaTra::PoroScatraMono::setup_solver()
{
  //  solver
  // create a linear solver
  // get dynamic section of poroelasticity
  const Teuchos::ParameterList& poroscatradyn =
      Global::Problem::instance()->poro_scatra_control_params();
  // get the solver number used for linear poroelasticity solver
  const int linsolvernumber = poroscatradyn.get<int>("LINEAR_SOLVER");
  // check if the poroelasticity solver has a valid solver number
  if (linsolvernumber == (-1))
    FOUR_C_THROW(
        "no linear solver defined for scalar transport in porous media. Please set LINEAR_SOLVER "
        "in POROSCATRA CONTROL to a valid number!");
  const Teuchos::ParameterList& solverparams =
      Global::Problem::instance()->solver_params(linsolvernumber);
  const auto solvertype =
      Teuchos::getIntegralValue<Core::LinearSolver::SolverType>(solverparams, "SOLVER");

  directsolve_ = (solvertype == Core::LinearSolver::SolverType::umfpack);

  if (directsolve_)
  {
    solver_ = std::make_shared<Core::LinAlg::Solver>(solverparams, get_comm(),
        Global::Problem::instance()->solver_params_callback(),
        Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
            Global::Problem::instance()->io_params(), "VERBOSITY"));
  }
  else
    // create a linear solver
    // create_linear_solver();
    FOUR_C_THROW("no implicit solver supported yet!");

  // Get the parameters for the Newton iteration
  itermax_ = poroscatradyn.get<int>("ITEMAX");
  itermin_ = poroscatradyn.get<int>("ITEMIN");
  normtypeinc_ = Teuchos::getIntegralValue<PoroElast::ConvNorm>(poroscatradyn, "NORM_INC");
  normtypeinc_ = Teuchos::getIntegralValue<PoroElast::ConvNorm>(poroscatradyn, "NORM_INC");
  normtypefres_ = Teuchos::getIntegralValue<PoroElast::ConvNorm>(poroscatradyn, "NORM_RESF");
  combincfres_ = Teuchos::getIntegralValue<PoroElast::BinaryOp>(poroscatradyn, "NORMCOMBI_RESFINC");
  vectornormfres_ =
      Teuchos::getIntegralValue<PoroElast::VectorNorm>(poroscatradyn, "VECTORNORM_RESF");
  vectornorminc_ =
      Teuchos::getIntegralValue<PoroElast::VectorNorm>(poroscatradyn, "VECTORNORM_INC");

  tolinc_ = poroscatradyn.get<double>("TOLINC_GLOBAL");
  tolfres_ = poroscatradyn.get<double>("TOLRES_GLOBAL");

  tolinc_struct_ = poroscatradyn.get<double>("TOLINC_DISP");
  tolinc_velocity_ = poroscatradyn.get<double>("TOLINC_VEL");
  tolinc_pressure_ = poroscatradyn.get<double>("TOLINC_PRES");
  tolinc_scalar_ = poroscatradyn.get<double>("TOLINC_SCALAR");
  // tolinc_porosity_= poroscatradyn.get<double> ("TOLINC_PORO");

  tolfres_struct_ = poroscatradyn.get<double>("TOLRES_DISP");
  tolfres_velocity_ = poroscatradyn.get<double>("TOLRES_VEL");
  tolfres_pressure_ = poroscatradyn.get<double>("TOLRES_PRES");
  tolfres_scalar_ = poroscatradyn.get<double>("TOLINC_SCALAR");
  // tolfres_porosity_= poroscatradyn.get<double> ("TOLRES_PORO");

  return true;
}

/*----------------------------------------------------------------------*
 | check convergence of Newton iteration (public)         vuong 01/12   |
 *----------------------------------------------------------------------*/
bool PoroElastScaTra::PoroScatraMono::converged()
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
                 normincfluidpres_ < tolinc_pressure_ and normincscalar_ < tolinc_scalar_
          //                  normincporo_      < tolinc_porosity_
      );
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
                  normrhsfluidpres_ < tolfres_pressure_ and normrhsscalar_ < tolfres_scalar_
          //                   normrhsporo_      < tolfres_porosity_
      );
      break;
    default:
      FOUR_C_THROW("Cannot check for convergence of residual forces!");
      break;
  }

  // combine increments and forces
  bool conv = false;
  switch (combincfres_)
  {
    case PoroElast::bop_and:
      conv = convinc and convfres;
      break;
    case PoroElast::bop_or:
      conv = convinc or convfres;
      break;
    default:
      FOUR_C_THROW("Something went terribly wrong with binary operator!");
      break;
  }

  // return things
  return conv;
}  // Converged()

/*----------------------------------------------------------------------*
 | print Newton-Raphson iteration to screen and error file    vuong 08/13   |
 | originally by lw 12/07, tk 01/08                                     |
 *----------------------------------------------------------------------*/
void PoroElastScaTra::PoroScatraMono::print_newton_iter()
{
  // print to standard out
  // replace myrank_ here general by Core::Communication::my_mpi_rank(Comm())
  if ((Core::Communication::my_mpi_rank(get_comm()) == 0) and printscreen_ and
      (step() % printscreen_ == 0) and printiter_)
  {
    if (iter_ == 1) print_newton_iter_header(stdout);
    print_newton_iter_text(stdout);
  }

}  // print_newton_iter()


/*----------------------------------------------------------------------*
 | print Newton-Raphson iteration to screen and error file    vuong 08/13  |
 | originally by lw 12/07, tk 01/08                                     |
 *----------------------------------------------------------------------*/
void PoroElastScaTra::PoroScatraMono::print_newton_iter_header(FILE* ofile)
{
  // open outstringstream
  std::ostringstream oss;

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
      //    case PoroElastScaTra::convnorm_rel_global:
      //      oss << std::setw(18) << "rel-res";
      break;
    case PoroElast::convnorm_abs_singlefields:
      //  case PoroElastScaTra::convnorm_rel_singlefields:
      break;
    default:
      FOUR_C_THROW("You should not turn up here.");
      break;
  }

  switch (normtypeinc_)
  {
    case PoroElast::convnorm_abs_global:
      oss << std::setw(15) << "abs-inc"
          << "(" << std::setw(5) << std::setprecision(2) << tolinc_ << ")";
      break;
      //    case PoroElastScaTra::convnorm_rel_global:
      //      oss << std::setw(18) << "rel-inc";
      break;
    case PoroElast::convnorm_abs_singlefields:
      // case PoroElastScaTra::convnorm_rel_singlefields:
      break;
    default:
      FOUR_C_THROW("You should not turn up here.");
      break;
  }

  // --------------------------------------------------------single field test
  switch (normtypefres_)
  {
    case PoroElast::convnorm_abs_global:
      break;
    case PoroElast::convnorm_abs_singlefields:
      oss << std::setw(15) << "abs-s-res"
          << "(" << std::setw(5) << std::setprecision(2) << tolfres_struct_ << ")";
      //      if(porositydof_)
      //        oss <<std::setw(15)<< "abs-poro-res"<<"("<<std::setprecision(2)
      //        <<tolfres_porosity_<<")";
      oss << std::setw(15) << "abs-fvel-res"
          << "(" << std::setw(5) << std::setprecision(2) << tolfres_velocity_ << ")";
      oss << std::setw(15) << "abs-fpres-res"
          << "(" << std::setw(5) << std::setprecision(2) << tolfres_pressure_ << ")";
      oss << std::setw(15) << "abs-sca-res"
          << "(" << std::setw(5) << std::setprecision(2) << tolfres_scalar_ << ")";
      break;
    default:
      FOUR_C_THROW("You should not turn up here.");
      break;
  }

  switch (normtypeinc_)
  {
    case PoroElast::convnorm_abs_global:
      break;
    case PoroElast::convnorm_abs_singlefields:
      oss << std::setw(15) << "abs-s-inc"
          << "(" << std::setw(5) << std::setprecision(2) << tolinc_struct_ << ")";
      //      if(porositydof_)
      //        oss <<std::setw(15)<< "abs-poro-inc"<<"("<<std::setprecision(2)
      //        <<tolinc_porosity_<<")";
      oss << std::setw(15) << "abs-fvel-inc"
          << "(" << std::setw(5) << std::setprecision(2) << tolinc_velocity_ << ")";
      oss << std::setw(15) << "abs-fpres-inc"
          << "(" << std::setw(5) << std::setprecision(2) << tolinc_pressure_ << ")";
      oss << std::setw(15) << "abs-sca-inc"
          << "(" << std::setw(5) << std::setprecision(2) << tolinc_scalar_ << ")";
      break;
    default:
      FOUR_C_THROW("You should not turn up here.");
      break;
  }

  // add solution time
  oss << std::setw(14) << "wct";

  // finish oss
  oss << std::ends;

  // print to screen (could be done differently...)
  if (ofile == nullptr) FOUR_C_THROW("no ofile available");
  fprintf(ofile, "%s\n", oss.str().c_str());

  // print it, now
  fflush(ofile);

  // nice to have met you
  return;
}  // print_newton_iter_header()


/*----------------------------------------------------------------------*
 | print Newton-Raphson iteration to screen              vuong 08/13 |
 | originally by lw 12/07, tk 01/08                                     |
 *----------------------------------------------------------------------*/
void PoroElastScaTra::PoroScatraMono::print_newton_iter_text(FILE* ofile)
{
  // open outstringstream
  std::ostringstream oss;

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
      FOUR_C_THROW("You should not turn up here.");
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
      FOUR_C_THROW("You should not turn up here.");
      break;
  }

  // --------------------------------------------------------single field test
  switch (normtypefres_)
  {
    case PoroElast::convnorm_abs_singlefields:
      oss << std::setw(22) << std::setprecision(5) << std::scientific << normrhsstruct_;
      //      if(porositydof_)
      //        oss << std::setw(22) << std::setprecision(5) << std::scientific << normrhsporo_;
      oss << std::setw(22) << std::setprecision(5) << std::scientific << normrhsfluidvel_;
      oss << std::setw(22) << std::setprecision(5) << std::scientific << normrhsfluidpres_;
      oss << std::setw(22) << std::setprecision(5) << std::scientific << normrhsscalar_;
      break;
    case PoroElast::convnorm_abs_global:
      break;
    default:
      FOUR_C_THROW("You should not turn up here.");
      break;
  }

  switch (normtypeinc_)
  {
    case PoroElast::convnorm_abs_singlefields:
      oss << std::setw(22) << std::setprecision(5) << std::scientific << normincstruct_;
      //      if(porositydof_)
      //        oss << std::setw(22) << std::setprecision(5) << std::scientific << normincporo_;
      oss << std::setw(22) << std::setprecision(5) << std::scientific << normincfluidvel_;
      oss << std::setw(22) << std::setprecision(5) << std::scientific << normincfluidpres_;
      oss << std::setw(22) << std::setprecision(5) << std::scientific << normincscalar_;
      break;
    case PoroElast::convnorm_abs_global:
      break;
    default:
      FOUR_C_THROW("You should not turn up here.");
      break;
  }

  // add solution time
  oss << std::setw(14) << std::setprecision(2) << std::scientific << timer_.totalElapsedTime(true);

  // finish oss
  oss << std::ends;

  // print to screen (could be done differently...)
  if (ofile == nullptr) FOUR_C_THROW("no ofile available");
  fprintf(ofile, "%s\n", oss.str().c_str());

  // print it, now
  fflush(ofile);

  // nice to have met you
  return;

}  // print_newton_iter_text

/*----------------------------------------------------------------------*
 | print statistics of converged NRI                      vuong 08/13    |
 | originally by bborn 08/09                                            |
 *----------------------------------------------------------------------*/
void PoroElastScaTra::PoroScatraMono::print_newton_conv()
{
  // somebody did the door
  return;
}

/*----------------------------------------------------------------------*
 |                                                         vuong 08/13  |
 *----------------------------------------------------------------------*/
void PoroElastScaTra::PoroScatraMono::build_convergence_norms()
{
  //------------------------------------------------------------ build residual force norms

  // global norm
  normrhs_ = PoroElast::Utils::calculate_vector_norm(vectornormfres_, *rhs_);

  // split vectors
  std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_s;
  std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_f;
  std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_fvel;
  std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_fpres;
  std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_scalar;

  // process structure unknowns of the first field
  rhs_s = extractor()->extract_vector(*rhs_, 0);

  // process fluid unknowns of the second field
  rhs_f = extractor()->extract_vector(*rhs_, 1);
  rhs_fvel = poro_field()->fluid_field()->extract_velocity_part(rhs_f);
  rhs_fpres = poro_field()->fluid_field()->extract_pressure_part(rhs_f);

  // process scalar unknowns of the third field
  rhs_scalar = extractor()->extract_vector(*rhs_, 2);

  //  if(porositydof_)
  //  {
  //    std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_poro =
  //    porositysplitter_->extract_cond_vector(rhs_s); std::shared_ptr<const
  //    Core::LinAlg::Vector<double>> rhs_sdisp = porositysplitter_->extract_other_vector(rhs_s);
  //
  //    normrhsstruct_ = Utils::calculate_vector_norm(vectornormfres_,rhs_sdisp);
  //    normrhsporo_ = Utils::calculate_vector_norm(vectornormfres_,rhs_poro);
  //  }
  //  else
  normrhsstruct_ = PoroElast::Utils::calculate_vector_norm(vectornormfres_, *rhs_s);

  normrhsfluid_ = PoroElast::Utils::calculate_vector_norm(vectornormfres_, *rhs_f);
  normrhsfluidvel_ = PoroElast::Utils::calculate_vector_norm(vectornormfres_, *rhs_fvel);
  normrhsfluidpres_ = PoroElast::Utils::calculate_vector_norm(vectornormfres_, *rhs_fpres);

  normrhsscalar_ = PoroElast::Utils::calculate_vector_norm(vectornormfres_, *rhs_scalar);


  //------------------------------------------------------------- build residual increment norms
  iterinc_->norm_2(&norminc_);

  // displacement and fluid velocity & pressure incremental vector
  std::shared_ptr<const Core::LinAlg::Vector<double>> interincs;
  std::shared_ptr<const Core::LinAlg::Vector<double>> interincf;
  std::shared_ptr<const Core::LinAlg::Vector<double>> interincfvel;
  std::shared_ptr<const Core::LinAlg::Vector<double>> interincfpres;
  std::shared_ptr<const Core::LinAlg::Vector<double>> interincscalar;

  // process structure unknowns of the first field
  interincs = extractor()->extract_vector(*iterinc_, 0);

  // process fluid unknowns of the second field
  interincf = extractor()->extract_vector(*iterinc_, 1);
  interincfvel = poro_field()->fluid_field()->extract_velocity_part(interincf);
  interincfpres = poro_field()->fluid_field()->extract_pressure_part(interincf);

  // process scalar unknowns of the third field
  interincscalar = extractor()->extract_vector(*iterinc_, 2);

  //  if(porositydof_)
  //  {
  //    std::shared_ptr<const Core::LinAlg::Vector<double>> interincporo =
  //    porositysplitter_->extract_cond_vector(interincs); std::shared_ptr<const
  //    Core::LinAlg::Vector<double>> interincsdisp =
  //    porositysplitter_->extract_other_vector(interincs);
  //
  //    normincstruct_     = Utils::calculate_vector_norm(vectornorminc_,interincsdisp);
  //    normincporo_       = Utils::calculate_vector_norm(vectornorminc_,interincporo);
  //  }
  //  else
  normincstruct_ = PoroElast::Utils::calculate_vector_norm(vectornorminc_, *interincs);

  normincfluid_ = PoroElast::Utils::calculate_vector_norm(vectornorminc_, *interincf);
  normincfluidvel_ = PoroElast::Utils::calculate_vector_norm(vectornorminc_, *interincfvel);
  normincfluidpres_ = PoroElast::Utils::calculate_vector_norm(vectornorminc_, *interincfpres);

  normincscalar_ = PoroElast::Utils::calculate_vector_norm(vectornorminc_, *interincscalar);

  return;
}

/*----------------------------------------------------------------------*
 |                                                         vuong 08/13  |
 *----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Map> PoroElastScaTra::PoroScatraMono::dof_row_map() const
{
  return blockrowdofmap_->full_map();
}

/*----------------------------------------------------------------------*
 |                                                         vuong 08/13  |
 *----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Map> PoroElastScaTra::PoroScatraMono::combined_dbc_map() const
{
  return dbcmaps_->cond_map();
}

/*----------------------------------------------------------------------*
 |                                                         vuong 08/13  |
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::SparseMatrix> PoroElastScaTra::PoroScatraMono::system_matrix()
{
  return systemmatrix_->merge();
}

/*----------------------------------------------------------------------*
 | put the single maps to one full                                      |
 | Poroelast_SCATRAicity map together                              vuong 01/12 |
 *----------------------------------------------------------------------*/
void PoroElastScaTra::PoroScatraMono::set_dof_row_maps(
    const std::vector<std::shared_ptr<const Core::LinAlg::Map>>& maps)
{
  std::shared_ptr<Core::LinAlg::Map> fullmap = Core::LinAlg::MultiMapExtractor::merge_maps(maps);

  // full monolithic-blockmap
  blockrowdofmap_->setup(*fullmap, maps);
}


/*----------------------------------------------------------------------*
 |  Evaluate off diagonal matrix in poro row                  vuong 08/13   |
 *---------------------------------------------------------------------*/
void PoroElastScaTra::PoroScatraMono::evaluate_od_block_mat_poro()
{
  k_pfs_->zero();

  // create the parameters for the discretization
  Teuchos::ParameterList fparams;
  // action for elements
  fparams.set<FLD::Action>("action", FLD::calc_poroscatra_mono_odblock);
  // physical type
  fparams.set<Inpar::FLUID::PhysicalType>(
      "Physical Type", poro_field()->fluid_field()->physical_type());

  // other parameters that might be needed by the elements
  fparams.set("delta time", dt());
  fparams.set("total time", time());

  const std::shared_ptr<Core::FE::Discretization>& porofluiddis =
      poro_field()->fluid_field()->discretization();
  porofluiddis->clear_state();

  // set general vector values needed by elements
  porofluiddis->set_state(0, "dispnp", *poro_field()->fluid_field()->dispnp());
  porofluiddis->set_state(0, "gridv", *poro_field()->fluid_field()->grid_vel());
  porofluiddis->set_state(0, "veln", *poro_field()->fluid_field()->veln());
  porofluiddis->set_state(0, "accnp", *poro_field()->fluid_field()->accnp());
  porofluiddis->set_state(0, "hist", *poro_field()->fluid_field()->hist());

  poro_field()->fluid_field()->discretization()->set_state(
      0, "scaaf", *poro_field()->fluid_field()->scaaf());

  porofluiddis->set_state(0, "velaf", *poro_field()->fluid_field()->velnp());
  porofluiddis->set_state(0, "velnp", *poro_field()->fluid_field()->velnp());

  // build specific assemble strategy for the fluid-mechanical system matrix
  // from the point of view of fluid_field:
  // fluiddofset = 0, structdofset = 1
  Core::FE::AssembleStrategy fluidstrategy(0,  // fluiddofset for row
      2,                                       // scatradofset for column
      k_pfs_,                                  // scatra-mechanical matrix
      nullptr,                                 // no other matrix or vectors
      nullptr, nullptr, nullptr);

  // evaluate the fluid-mechanical system matrix on the fluid element
  poro_field()->fluid_field()->discretization()->evaluate_condition(
      fparams, fluidstrategy, "PoroCoupling");
  // poro_field()->fluid_field()->discretization()->evaluate( fparams, fluidstrategy );

  porofluiddis->clear_state(true);

  //************************************************************************************
  //************************************************************************************

  k_pss_->zero();


  // create the parameters for the discretization
  Teuchos::ParameterList sparams;

  const std::string action = "struct_poro_calc_scatracoupling";
  sparams.set("action", action);
  // other parameters that might be needed by the elements
  sparams.set("delta time", dt());
  sparams.set("total time", time());

  poro_field()->structure_field()->discretization()->clear_state();
  poro_field()->structure_field()->discretization()->set_state(
      0, "displacement", *poro_field()->structure_field()->dispnp());
  poro_field()->structure_field()->discretization()->set_state(
      0, "velocity", *poro_field()->structure_field()->velnp());

  // build specific assemble strategy for mechanical-fluid system matrix
  // from the point of view of structure_field:
  // structdofset = 0, fluiddofset = 1
  Core::FE::AssembleStrategy structuralstrategy(0,  // structdofset for row
      2,                                            // scatradofset for column
      k_pss_,                                       // mechanical-scatra coupling matrix
      nullptr, nullptr, nullptr, nullptr);

  // evaluate the mechanical-fluid system matrix on the structural element
  poro_field()->structure_field()->discretization()->evaluate_condition(
      sparams, structuralstrategy, "PoroCoupling");
  // structure_field()->discretization()->evaluate( sparams, structuralstrategy);
  poro_field()->structure_field()->discretization()->clear_state(true);

  return;
}

/*----------------------------------------------------------------------*
 |  Evaluate off diagonal matrix in scatra row                    |
 *----------------------------------------------------------------------*/
void PoroElastScaTra::PoroScatraMono::evaluate_od_block_mat_scatra()
{
  // create the parameters for the discretization
  Teuchos::ParameterList sparams_struct;

  k_sps_->zero();

  Core::Utils::add_enum_class_to_parameter_list<ScaTra::Action>(
      "action", ScaTra::Action::calc_scatra_mono_odblock_mesh, sparams_struct);
  // other parameters that might be needed by the elements
  sparams_struct.set("delta time", dt());
  sparams_struct.set("total time", time());

  scatra_field()->discretization()->clear_state();
  scatra_field()->discretization()->set_state(0, "hist", *scatra_field()->hist());
  scatra_field()->discretization()->set_state(0, "phinp", *scatra_field()->phinp());

  // build specific assemble strategy for mechanical-fluid system matrix
  // from the point of view of structure_field:
  // structdofset = 0, fluiddofset = 1
  Core::FE::AssembleStrategy scatrastrategy_struct(0,  // scatradofset for row
      1,                                               // structuredofset for column
      k_sps_,                                          // scatra-structure coupling matrix
      nullptr, nullptr, nullptr, nullptr);

  // evaluate the mechanical-fluid system matrix on the structural element
  scatra_field()->discretization()->evaluate_condition(
      sparams_struct, scatrastrategy_struct, "PoroCoupling");
  // structure_field()->discretization()->evaluate( sparams, structuralstrategy);

  scatra_field()->discretization()->clear_state();
  // FOUR_C_THROW("stop");
  //************************************************************************************
  //************************************************************************************

  k_spf_->zero();

  // create the parameters for the discretization
  Teuchos::ParameterList sparams_fluid;

  Core::Utils::add_enum_class_to_parameter_list<ScaTra::Action>(
      "action", ScaTra::Action::calc_scatra_mono_odblock_fluid, sparams_fluid);
  // other parameters that might be needed by the elements
  sparams_fluid.set("delta time", dt());
  sparams_fluid.set("total time", time());

  scatra_field()->discretization()->clear_state();
  scatra_field()->discretization()->set_state(0, "hist", *scatra_field()->hist());
  scatra_field()->discretization()->set_state(0, "phinp", *scatra_field()->phinp());

  // build specific assemble strategy for mechanical-fluid system matrix
  // from the point of view of structure_field:
  // structdofset = 0, fluiddofset = 1
  Core::FE::AssembleStrategy scatrastrategy_fluid(0,  // scatradofset for row
      2,                                              // fluiddofset for column
      k_spf_,                                         // scatra-fluid coupling matrix
      nullptr, nullptr, nullptr, nullptr);

  // evaluate the mechanical-fluid system matrix on the structural element
  scatra_field()->discretization()->evaluate_condition(
      sparams_fluid, scatrastrategy_fluid, "PoroCoupling");
  // structure_field()->discretization()->evaluate( sparams, structuralstrategy);
  scatra_field()->discretization()->clear_state();

  return;
}

/*----------------------------------------------------------------------*
 |  check tangent stiffness matrix vie finite differences      vuong 08/13  |
 *----------------------------------------------------------------------*/
void PoroElastScaTra::PoroScatraMono::fd_check()
{
  std::cout << "\n******************finite difference check***************" << std::endl;

  int dof_struct = (poro_field()->structure_field()->discretization()->num_global_nodes()) * 3;
  int dof_fluid = (poro_field()->fluid_field()->discretization()->num_global_nodes()) * 4;
  int dof_scatra = (scatra_field()->discretization()->num_global_nodes());

  std::cout << "structure field has " << dof_struct << " DOFs" << std::endl;
  std::cout << "fluid field has " << dof_fluid << " DOFs" << std::endl;
  std::cout << "scatra field has " << dof_scatra << " DOFs" << std::endl;

  std::shared_ptr<Core::LinAlg::Vector<double>> iterinc = nullptr;
  iterinc = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map(), true);

  const int dofs = iterinc->global_length();
  std::cout << "in total " << dofs << " DOFs" << std::endl;
  const double delta = 1e-8;

  iterinc->put_scalar(0.0);

  iterinc->replace_global_value(0, delta);

  Core::LinAlg::SparseMatrix stiff_approx(*dof_row_map(), 81);

  Core::LinAlg::Vector<double> rhs_old(*dof_row_map(), true);
  rhs_old.update(1.0, *rhs_, 0.0);
  Core::LinAlg::Vector<double> rhs_copy(*dof_row_map(), true);

  std::shared_ptr<Core::LinAlg::SparseMatrix> sparse = systemmatrix_->merge();
  Core::LinAlg::SparseMatrix sparse_copy(*sparse, Core::LinAlg::DataAccess::Copy);


  const int zeilennr = -1;
  const int spaltenr = -1;
  for (int i = 0; i < dofs; ++i)
  {
    if (combined_dbc_map()->my_gid(i))
    {
      iterinc->replace_global_value(i, 0.0);
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
      std::cout << "rhs_: " << rhs_copy.local_values_as_span()[zeilennr] << std::endl;
      std::cout << "rhs_old: " << rhs_old.local_values_as_span()[zeilennr] << std::endl;
    }

    rhs_copy.update(-1.0, rhs_old, 1.0);
    rhs_copy.scale(-1.0 / delta);

    int* index = &i;
    for (int j = 0; j < dofs; ++j)
    {
      double value = rhs_copy.local_values_as_span()[j];
      stiff_approx.insert_global_values(j, 1, &value, index);

      if ((j == zeilennr) and (i == spaltenr))
      {
        std::cout << "\n******************" << zeilennr + 1 << ". Zeile!!***************"
                  << std::endl;
        /*std::cout << "iterinc_" << std::endl << *iterinc_ << std::endl;
        std::cout << "iterinc" << std::endl << *iterinc << std::endl;
        std::cout << "meshdisp: " << std::endl << *(poro_field()->fluid_field()->dispnp());
        std::cout << "meshdisp scatra: " << std::endl
                  << *(scatra_field()->discretization()->get_state(
                         scatra_field()->nds_disp(), "dispnp"));
        std::cout << "disp: " << std::endl << *(poro_field()->structure_field()->dispnp());
        std::cout << "fluid vel" << std::endl << *(poro_field()->fluid_field()->velnp());
        std::cout << "scatra vel" << std::endl
                  << *(scatra_field()->discretization()->get_state(
                         scatra_field()->nds_vel(), "velocity field"));
        std::cout << "fluid acc" << std::endl << *(poro_field()->fluid_field()->accnp());
        std::cout << "gridvel fluid" << std::endl << *(poro_field()->fluid_field()->grid_vel());
        std::cout << "gridvel struct" << std::endl << *(poro_field()->structure_field()->velnp());

        std::cout << "stiff_apprx(" << zeilennr << "," << spaltenr << "): " << (*rhs_copy)[zeilennr]
                  << std::endl;

        std::cout << "value(" << zeilennr << "," << spaltenr << "): " << value << std::endl;
        std::cout << "\n******************" << zeilennr + 1 << ". Zeile End!!***************"
                  << std::endl;
                  */
      }
    }

    if (not combined_dbc_map()->my_gid(i)) iterinc->replace_global_value(i, -delta);

    iterinc->replace_global_value(i - 1, 0.0);

    if (i != dofs - 1) iterinc->replace_global_value(i + 1, delta);

    if (i == spaltenr)
      std::cout << "\n******************" << spaltenr + 1 << ". Spalte End!!***************"
                << std::endl;
  }

  evaluate(iterinc);
  setup_rhs();

  stiff_approx.complete();

  Core::LinAlg::SparseMatrix stiff_approx_sparse(stiff_approx);

  Core::LinAlg::matrix_add(sparse_copy, false, -1.0, stiff_approx_sparse, 1.0);

  Core::LinAlg::SparseMatrix sparse_crs(sparse_copy);
  Core::LinAlg::SparseMatrix error_crs(stiff_approx_sparse);

  error_crs.complete();
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
            int errornumentries;
            int errorlength = error_crs.num_global_entries(i);
            std::vector<double> errorvalues(errorlength);
            std::vector<int> errorindices(errorlength);
            // int errorextractionstatus =
            error_crs.extract_global_row_copy(
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
            int sparselength = sparse_crs.num_global_entries(i);
            std::vector<double> sparsevalues(sparselength);
            std::vector<int> sparseindices(sparselength);
            // int sparseextractionstatus =
            sparse_crs.extract_global_row_copy(
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

  return;
}

FOUR_C_NAMESPACE_CLOSE

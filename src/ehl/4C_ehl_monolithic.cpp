// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_ehl_monolithic.hpp"

#include "4C_adapter_coupling_ehl_mortar.hpp"
#include "4C_adapter_str_structure.hpp"
#include "4C_contact_element.hpp"
#include "4C_contact_interface.hpp"
#include "4C_contact_node.hpp"
#include "4C_coupling_adapter_converter.hpp"
#include "4C_fem_condition_locsys.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_assemblestrategy.hpp"
#include "4C_global_data.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_blocksparsematrix.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linear_solver_method.hpp"
#include "4C_linear_solver_method_input.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_lubrication_adapter.hpp"
#include "4C_lubrication_ele_action.hpp"
#include "4C_lubrication_timint_implicit.hpp"
#include "4C_mat_lubrication_mat.hpp"
#include "4C_mortar_coupling3d_classes.hpp"
#include "4C_mortar_node.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN


//! Note: The order of calling the two BaseAlgorithm-constructors is
//! important here! In here control file entries are written. And these entries
//! define the order in which the filters handle the Discretizations, which in
//! turn defines the dof number ordering of the Discretizations.



/*----------------------------------------------------------------------*
 | monolithic                                               wirtz 01/16 |
 *----------------------------------------------------------------------*/
EHL::Monolithic::Monolithic(MPI_Comm comm, const Teuchos::ParameterList& globaltimeparams,
    const Teuchos::ParameterList& lubricationparams, const Teuchos::ParameterList& structparams,
    const std::string struct_disname, const std::string lubrication_disname)
    : Base(comm, globaltimeparams, lubricationparams, structparams, struct_disname,
          lubrication_disname),
      solveradapttol_(Global::Problem::instance()
              ->elasto_hydro_dynamic_params()
              .sublist("MONOLITHIC")
              .get<bool>("ADAPTCONV")),
      solveradaptolbetter_(Global::Problem::instance()
              ->elasto_hydro_dynamic_params()
              .sublist("MONOLITHIC")
              .get<double>("ADAPTCONV_BETTER")),
      printiter_(true),  // ADD INPUT PARAMETER
      zeros_(nullptr),
      strmethodname_(
          Teuchos::getIntegralValue<Inpar::Solid::DynamicType>(structparams, "DYNAMICTYPE")),
      ehldyn_(Global::Problem::instance()->elasto_hydro_dynamic_params()),
      ehldynmono_(
          (Global::Problem::instance()->elasto_hydro_dynamic_params()).sublist("MONOLITHIC")),
      blockrowdofmap_(nullptr),
      systemmatrix_(nullptr),
      k_sl_(nullptr),
      k_ls_(nullptr),
      iternorm_(Teuchos::getIntegralValue<EHL::VectorNorm>(ehldynmono_, "ITERNORM")),
      iter_(0),
      sdyn_(structparams),
      timernewton_("EHL_Monolithic_newton", true)
{
  blockrowdofmap_ = std::make_shared<Core::LinAlg::MultiMapExtractor>();

  // --------------------------------- EHL solver: create a linear solver

  // get solver parameter list of linear TSI solver
  const int linsolvernumber = ehldynmono_.get<int>("LINEAR_SOLVER");
  const Teuchos::ParameterList& ehlsolverparams =
      Global::Problem::instance()->solver_params(linsolvernumber);

  std::shared_ptr<Teuchos::ParameterList> solverparams = std::make_shared<Teuchos::ParameterList>();
  *solverparams = ehlsolverparams;

  solver_ = std::make_shared<Core::LinAlg::Solver>(*solverparams, Monolithic::get_comm(),
      Global::Problem::instance()->solver_params_callback(),
      Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
          Global::Problem::instance()->io_params(), "VERBOSITY"));

}  // Monolithic()


/*----------------------------------------------------------------------*
 | prepare time step (public)                               wirtz 01/16 |
 *----------------------------------------------------------------------*/
void EHL::Monolithic::prepare_time_step()
{
  // counter and print header
  // increment time and step counter
  increment_time_and_step();
  print_header();

  // apply current pressure to structure
  // set the external fluid force on the structure, which result from the fluid pressure
  set_lubrication_solution(lubrication_->lubrication_field()->prenp());

  // call the predictor
  structure_field()->prepare_time_step();
  lubrication_->lubrication_field()->prepare_time_step();

}  // prepare_time_step()

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void EHL::Monolithic::solve() { newton_full(); }

/*----------------------------------------------------------------------*
 | time loop of the monolithic system                       wirtz 01/16 |
 *----------------------------------------------------------------------*/
void EHL::Monolithic::timeloop()
{
  // time loop
  while (not_finished())
  {
    // counter and print header
    // predict solution of both field (call the adapter)
    prepare_time_step();

    // integrate time step
    solve();

    // calculate stresses, strains, energies
    constexpr bool force_prepare = false;
    prepare_output(force_prepare);

    // update all single field solvers
    update();

    // write output to screen and files
    output();

  }  // not_finished
}  // TimeLoop()


/*----------------------------------------------------------------------*
 | solution with full Newton-Raphson iteration              wirtz 01/16 |
 | in ehl_algorithm: NewtonFull()                                       |
 *----------------------------------------------------------------------*/
void EHL::Monolithic::newton_full()
{
  // we do a Newton-Raphson iteration here.
  // the specific time integration has set the following
  // --> On #rhs_ is the positive force residuum
  // --> On #systemmatrix_ is the effective dynamic EHL tangent matrix

  // initialise equilibrium loop
  iter_ = 1;

  // incremental solution vector with length of all EHL dofs
  iterinc_ = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map(), true);
  iterinc_->put_scalar(0.0);

  // a zero vector of full length
  zeros_ = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map(), true);
  zeros_->put_scalar(0.0);

  //------------------------------------------------------ iteration loop

  // equilibrium iteration loop (loop over k)
  while (((not converged()) and (iter_ <= itermax_)) or (iter_ <= itermin_))
  {
    // reset timer
    timernewton_.reset();

    // compute residual forces #rhs_ and tangent #systemmatrix_
    // whose components are globally oriented
    // build linear system stiffness matrix and rhs/force residual for each
    // field, here e.g. for structure field: field want the iteration increment
    // 1.) Update(iterinc_),
    // 2.) evaluate_force_stiff_residual(),
    evaluate(iterinc_);

    // create the linear system
    // \f$J(x_i) \Delta x_i = - R(x_i)\f$
    // create the systemmatrix
    setup_system_matrix();

    // check whether we have a sanely filled tangent matrix
    if (not systemmatrix_->filled()) FOUR_C_THROW("Effective tangent matrix must be filled here");

    // create full monolithic rhs vector
    // make negative residual not necessary: rhs_ is already negative
    // (STR/LUBRICATION)-RHS is put negative in prepare_system_for_newton_solve()
    setup_rhs();

    if (dry_contact_)
    {
      mortaradapter_->condense_contact(systemmatrix_, rhs_, structure_field()->dispnp(), dt());
      apply_dbc();
    }
    // *********** time measurement ***********
    double dtcpu = timernewton_.wallTime();
    // *********** time measurement ***********
    // (Newton-ready) residual with blanked Dirichlet DOFs (see adapter_timint!)
    // is done in prepare_system_for_newton_solve() within evaluate(iterinc_)
    linear_solve();
    // *********** time measurement ***********
    dtsolve_ = timernewton_.wallTime() - dtcpu;
    // *********** time measurement ***********

    // vector of displacement and pressure increments
    std::shared_ptr<Core::LinAlg::Vector<double>> sx;
    std::shared_ptr<Core::LinAlg::Vector<double>> lx;
    // extract field vectors
    extract_field_vectors(iterinc_, sx, lx);

    if (dry_contact_) mortaradapter_->recover_coupled(sx, lx);

    // reset solver tolerance
    solver_->reset_tolerance();

    // --------------------------------------------- build residual norms
    // include all stuff here related with convergence test
    normrhs_ = calculate_vector_norm(iternorm_, *rhs_);
    // vector of displacement and pressure residual
    std::shared_ptr<Core::LinAlg::Vector<double>> strrhs;
    std::shared_ptr<Core::LinAlg::Vector<double>> lubricationrhs;
    // extract field vectors
    extract_field_vectors(rhs_, strrhs, lubricationrhs);
    normstrrhs_ = calculate_vector_norm(iternormstr_, *strrhs);
    normlubricationrhs_ = calculate_vector_norm(iternormlubrication_, *lubricationrhs);

    // --------------------------------- build residual incremental norms
    norminc_ = calculate_vector_norm(iternorm_, *iterinc_);
    normdisi_ = calculate_vector_norm(iternormstr_, *sx);
    normprei_ = calculate_vector_norm(iternormlubrication_, *lx);

    // in case of 'Mix'-convergence criterion: save the norm of the 1st
    // iteration in (norm . iter0_)
    if (iter_ == 1)
    {
      // save initial residual norms
      normrhsiter0_ = normrhs_;
      normstrrhsiter0_ = normstrrhs_;
      normlubricationrhsiter0_ = normlubricationrhs_;
      // save initial incremental norms
      norminciter0_ = norminc_;
      normdisiiter0_ = normdisi_;
      normpreiiter0_ = normprei_;

      // set the minimum of iter0_ and tolrhs_, because we want to prevent the
      // case of a zero characteristic initial norm
      if (normrhsiter0_ == 0.0) normrhsiter0_ = tolrhs_;
      if (normstrrhsiter0_ == 0.0) normstrrhsiter0_ = tolstrrhs_;
      if (normlubricationrhsiter0_ == 0.0) normlubricationrhsiter0_ = tollubricationrhs_;
      if (norminciter0_ == 0.0) norminciter0_ = tolinc_;
      if (normdisiiter0_ == 0.0) normdisiiter0_ = toldisi_;
      if (normpreiiter0_ == 0.0) normpreiiter0_ = tolprei_;
    }

    // print stuff
    print_newton_iter();

    // increment equilibrium loop index
    iter_ += 1;

  }  // end equilibrium loop

  // ----------------------------------------------------- iteration loop

  // correct iteration counter
  iter_ -= 1;

  // test whether max iterations was hit
  if ((converged()) and (Core::Communication::my_mpi_rank(get_comm()) == 0))
  {
    print_newton_conv();
  }
  else if (Teuchos::getIntegralValue<Inpar::Solid::DivContAct>(sdyn_, "DIVERCONT") ==
           Inpar::Solid::divcont_continue)
    ;
  else if (iter_ >= itermax_)
    FOUR_C_THROW("Newton unconverged in {} iterations", iter_);

}  // NewtonFull()

/*----------------------------------------------------------------------*
 | evaluate the single fields                               wirtz 01/16 |
 *----------------------------------------------------------------------*/
void EHL::Monolithic::evaluate(std::shared_ptr<Core::LinAlg::Vector<double>> stepinc)
{
  TEUCHOS_FUNC_TIME_MONITOR("EHL::Monolithic::Evaluate");

  // displacement and pressure incremental vector
  std::shared_ptr<Core::LinAlg::Vector<double>> sx;
  std::shared_ptr<Core::LinAlg::Vector<double>> lx;

  // if an increment vector exists
  if (stepinc != nullptr)
  {
    // extract displacement sx and pressure lx incremental vector of global
    // unknown incremental vector x
    extract_field_vectors(stepinc, sx, lx);
  }

  // Newton update of the lubrication field
  // update pressure before passed to the structural field
  lubrication_->lubrication_field()->update_newton(
      lx);  // The same should be done for the structure field before updating the couplingh...

  // pass the structural values to the lubrication field,
  // i.e. set mesh displacement, velocity fields and film thickness
  // note: the iteration update has not been done yet
  std::shared_ptr<Core::LinAlg::Vector<double>> new_disp =
      std::make_shared<Core::LinAlg::Vector<double>>(*structure_->dispnp());
  new_disp->update(1., *sx, 1.);

  // set interface height, velocity etc to lubrication field
  set_struct_solution(new_disp);

  // apply current pressure to structure
  // set the external fluid force on the structure, which result from the fluid pressure
  set_lubrication_solution(lubrication_->lubrication_field()->prenp());

  // structure Evaluate (builds tangent, residual and applies DBC)
  structure_field()->evaluate(sx);
  structure_field()->discretization()->clear_state(true);

  /// lubrication field

  // lubrication Evaluate
  // (builds tangent, residual and applies DBC and recent coupling values)
  lubrication_->lubrication_field()->evaluate();
  lubrication_->lubrication_field()->discretization()->clear_state(true);

}  // evaluate()



/*----------------------------------------------------------------------*
 | extract field vectors for calling evaluate() of the      wirtz 01/16 |
 | single fields                                                        |
 *----------------------------------------------------------------------*/
void EHL::Monolithic::extract_field_vectors(std::shared_ptr<Core::LinAlg::Vector<double>> x,
    std::shared_ptr<Core::LinAlg::Vector<double>>& sx,
    std::shared_ptr<Core::LinAlg::Vector<double>>& lx)
{
  TEUCHOS_FUNC_TIME_MONITOR("EHL::Monolithic::extract_field_vectors");

  // process structure unknowns of the first field
  sx = extractor()->extract_vector(*x, 0);

  // process lubrication unknowns of the second field
  lx = extractor()->extract_vector(*x, 1);
}  // extract_field_vectors()


/*----------------------------------------------------------------------*
 | full monolithic dof row map                              wirtz 01/16 |
 *----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Map> EHL::Monolithic::dof_row_map() const
{
  return extractor()->full_map();
}  // dof_row_map()


/*----------------------------------------------------------------------*
 | setup system (called in ehl_dyn)                         wirtz 01/16 |
 *----------------------------------------------------------------------*/
void EHL::Monolithic::setup_system()
{
  // set parameters that remain the same in the whole calculation
  set_default_parameters();

  // create combined map
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> vecSpaces;

  // use its own dof_row_map, that is the 0th map of the discretization
  vecSpaces.push_back(structure_field()->dof_row_map(0));
  vecSpaces.push_back(lubrication_->lubrication_field()->dof_row_map(0));

  if (vecSpaces[0]->num_global_elements() == 0) FOUR_C_THROW("No structure equation. Panic.");
  if (vecSpaces[1]->num_global_elements() == 0) FOUR_C_THROW("No pressure equation. Panic.");

  set_dof_row_maps(vecSpaces);

  /*----------------------------------------------------------------------*/
  // initialise EHL-systemmatrix_
  systemmatrix_ =
      std::make_shared<Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(
          *extractor(), *extractor(), 81, false, true);

  // create empty matrix
  k_sl_ = std::make_shared<Core::LinAlg::SparseMatrix>(
      *(structure_field()->discretization()->dof_row_map(0)), 81, true, true);

  // create empty matrix
  k_ls_ = std::make_shared<Core::LinAlg::SparseMatrix>(
      *(lubrication_->lubrication_field()->discretization()->dof_row_map(0)), 81, true, true);

}  // SetupSystem()


/*----------------------------------------------------------------------*
 | put the single maps to one full EHL map together         wirtz 01/16 |
 *----------------------------------------------------------------------*/
void EHL::Monolithic::set_dof_row_maps(
    const std::vector<std::shared_ptr<const Core::LinAlg::Map>>& maps)
{
  std::shared_ptr<Core::LinAlg::Map> fullmap = Core::LinAlg::MultiMapExtractor::merge_maps(maps);

  // full EHL-blockmap
  extractor()->setup(*fullmap, maps);
}  // set_dof_row_maps()


/*----------------------------------------------------------------------*
 | setup system matrix of EHL                               wirtz 01/16 |
 *----------------------------------------------------------------------*/
void EHL::Monolithic::setup_system_matrix()
{
  TEUCHOS_FUNC_TIME_MONITOR("EHL::Monolithic::setup_system_matrix");



  //--------------------------------
  // 0. Prepare the linearizations...
  //--------------------------------

  // Time integration specific parameters..
  double alphaf = -1.;
  switch (Teuchos::getIntegralValue<Inpar::Solid::DynamicType>(sdyn_, "DYNAMICTYPE"))
  {
    case Inpar::Solid::DynamicType::GenAlpha:
    {
      alphaf = sdyn_.sublist("GENALPHA").get<double>("ALPHA_F");
      break;
    }
    case Inpar::Solid::DynamicType::Statics:
    {
      alphaf = 0.;
      break;
    }
    default:
      FOUR_C_THROW("unknown time integration strategy for structural problem in coupled EHL");
  }

  //--------------------------------------------------------------------------------------
  // 1. Calculate and assemble k_ss (Linearization of structure residual wrt displacements)
  //--------------------------------------------------------------------------------------

  // k_ss basically consists of the standard effective dynamic stiffness matrix plus the
  // linearization of the lubrication interface force wrt the displacements


  //----------------------------------------------
  // 1.1 Get the effective dynamic stiffness matrix
  //----------------------------------------------

  // Effective dynamic stiffness matrix, Dirichlet already applied.
  std::shared_ptr<Core::LinAlg::SparseMatrix> k_ss = structure_field()->system_matrix();
  k_ss->un_complete();

  //----------------------------------------------------------------------
  // 1.2 Linearization of lubrication interface force wrt the displacements
  //----------------------------------------------------------------------

  // Lubrication interface force: f_lub = B * t = D^T*t - M^T*t
  // Derivative of lubrication interface force: df_lub/dd = dB/dd * t + B * dt/dd

  // 1.2.1 Derivative due to deformation-dependent Mortar matrices
  // d(D^T)/dd * t
  std::shared_ptr<Core::LinAlg::Vector<double>> stritraction_D_col =
      std::make_shared<Core::LinAlg::Vector<double>>(
          *(mortaradapter_->interface()->slave_col_dofs()), true);
  Core::LinAlg::export_to(*stritraction_D_, *stritraction_D_col);
  std::shared_ptr<Core::LinAlg::SparseMatrix> slaveiforce_derivd1 =
      mortaradapter_->assemble_ehl_lin_d(stritraction_D_col);

  // d(-M^T)/dd * t
  std::shared_ptr<Core::LinAlg::Vector<double>> stritraction_M_col =
      std::make_shared<Core::LinAlg::Vector<double>>(
          *(mortaradapter_->interface()->slave_col_dofs()), true);
  Core::LinAlg::export_to(*stritraction_M_, *stritraction_M_col);
  std::shared_ptr<Core::LinAlg::SparseMatrix> masteriforce_derivd1 =
      mortaradapter_->assemble_ehl_lin_m(stritraction_M_col);
  masteriforce_derivd1->scale(-1.0);

  // Add the negative midpoint values (remember, we are linearizing an extforce-like term)
  Core::LinAlg::matrix_add(*slaveiforce_derivd1, false, -(1.0 - alphaf), *k_ss, 1.0);
  Core::LinAlg::matrix_add(*masteriforce_derivd1, false, -(1.0 - alphaf), *k_ss, 1.0);

  // linearization of D.t and M.t (the part with D.(dt/dd))
  std::shared_ptr<Core::LinAlg::SparseMatrix> ds_dd =
      std::make_shared<Core::LinAlg::SparseMatrix>(*mortaradapter_->slave_dof_map(), 81);
  std::shared_ptr<Core::LinAlg::SparseMatrix> dm_dd =
      std::make_shared<Core::LinAlg::SparseMatrix>(*mortaradapter_->master_dof_map(), 81);

  // pressure term
  lin_pressure_force_disp(*ds_dd, *dm_dd);
  // couette term
  lin_couette_force_disp(*ds_dd, *dm_dd);
  // poiseuille term
  lin_poiseuille_force_disp(*ds_dd, *dm_dd);

  // complete matrices
  ds_dd->complete(*mortaradapter_->s_mdof_map(), *mortaradapter_->slave_dof_map());
  dm_dd->complete(*mortaradapter_->s_mdof_map(), *mortaradapter_->master_dof_map());

  // Add the negative midpoint values (remember, we are linearizing an extforce-like term)
  Core::LinAlg::matrix_add(*ds_dd, false, -(1.0 - alphaf), *k_ss, 1.0);
  Core::LinAlg::matrix_add(*dm_dd, false, -(1.0 - alphaf), *k_ss, 1.0);

  k_ss->complete();
  k_ss->apply_dirichlet(*structure_field()->get_dbc_map_extractor()->cond_map(), true);

  // Assign k_ss to system matrix
  systemmatrix_->assign(0, 0, Core::LinAlg::DataAccess::Share, *k_ss);


  //---------------------------------------------------------------------------------
  // 2. Calculate and assemble k_sl (Linearization of structure residual wrt pressure)
  //---------------------------------------------------------------------------------

  // Derivative of the lubrication interface force: df_lub/dp = B * dt/dp

  k_sl_->reset();
  k_sl_->un_complete();

  // linearization of traction w.r.t. fluid pressure
  std::shared_ptr<Core::LinAlg::SparseMatrix> slaveiforce_derivp =
      std::make_shared<Core::LinAlg::SparseMatrix>(
          *mortaradapter_->slave_dof_map(), 81, true, true);
  std::shared_ptr<Core::LinAlg::SparseMatrix> masteriforce_derivp =
      std::make_shared<Core::LinAlg::SparseMatrix>(
          *mortaradapter_->master_dof_map(), 81, true, true);

  // pressure force
  lin_pressure_force_pres(*slaveiforce_derivp, *masteriforce_derivp);

  // poiseuille force
  lin_poiseuille_force_pres(*slaveiforce_derivp, *masteriforce_derivp);

  // couette force
  lin_couette_force_pres(*slaveiforce_derivp, *masteriforce_derivp);

  slaveiforce_derivp->complete(
      *lubrication_->lubrication_field()->dof_row_map(), *mortaradapter_->slave_dof_map());
  masteriforce_derivp->complete(
      *lubrication_->lubrication_field()->dof_row_map(), *mortaradapter_->master_dof_map());

  // Add the negative midpoint values (remember, we are linearizing an extforce-like term)
  Core::LinAlg::matrix_add(*slaveiforce_derivp, false, -(1.0 - alphaf), *k_sl_, 1.0);
  Core::LinAlg::matrix_add(*masteriforce_derivp, false, -(1.0 - alphaf), *k_sl_, 1.0);

  k_sl_->complete(*(extractor()->map(1)),  // pressure dof map
      *(extractor()->map(0))               // displacement dof map
  );

  // No DBC need to be applied, since lubrication interface disp-dofs must NOT have dbc conditions!
  k_sl_->apply_dirichlet(*structure_field()->get_dbc_map_extractor()->cond_map(), false);

  // Assign k_sl to system matrix
  systemmatrix_->assign(0, 1, Core::LinAlg::DataAccess::Share, *(k_sl_));


  //-----------------------------------------------------------------------------------
  // 3. Calculate and assemble k_ll (Linearization of lubrication residual wrt pressure)
  //-----------------------------------------------------------------------------------

  // Pure hydrodynamic stiffness matrix
  std::shared_ptr<Core::LinAlg::SparseMatrix> k_ll =
      lubrication_->lubrication_field()->system_matrix();

  // Assign k_ll to system matrix
  systemmatrix_->assign(1, 1, Core::LinAlg::DataAccess::Share, *(k_ll));


  //----------------------------------------------------------------------------------------
  // 4. Calculate and assemble k_ls (Linearization of lubrication residual wrt displacements)
  //----------------------------------------------------------------------------------------

  k_ls_->reset();
  k_ls_ = std::make_shared<Core::LinAlg::SparseMatrix>(
      *(lubrication_->lubrication_field()->discretization()->dof_row_map(0)), 81, false, true);

  /*
   * Three distinct types of derivatives of the lubrication residual wrt to interface displacement
   * exist: -Derivatives due to the deformation dependent film height -Derivatives due to the
   * deformation dependent tangential velocities -Derivatives due to the deformation of the
   * lubrication mesh (not implemented yet)
   */

  /*
   * Instead of assembling the element-wise contributions directly into k_ls, following strategy is
   * followed: The discrete nodal derivatives of the film height and the tangential velocities are
   * calculated and assembled globally. The integrals associated with those derivatives are
   * evaluated element-wise and assembled into global matrices. The product of the discret
   * derivatives with those matrices finally enter the stiffness matrix k_ls
   */

  //-------------------------------------------------------------------------------------------
  // 4.0 Calculate and assemble the matrices, which are associated with the discrete derivatives
  //-------------------------------------------------------------------------------------------

  // Global matrix associated with the discrete film height derivative
  std::shared_ptr<Core::LinAlg::SparseMatrix> k_ls_linH =
      std::make_shared<Core::LinAlg::SparseMatrix>(*(extractor()->map(1)), 81, false, false);

  // Global matrix associated with the discrete tangential velocity derivative
  std::shared_ptr<Core::LinAlg::SparseMatrix> k_ls_linV =
      std::make_shared<Core::LinAlg::SparseMatrix>(*(extractor()->map(1)), 81, false, false);

  // Call elements and assemble
  apply_lubrication_coupl_matrix(k_ls_linH, k_ls_linV);

  //-----------------------------------
  // 4.1 Discrete film height derivative
  //-----------------------------------
  std::shared_ptr<Core::LinAlg::SparseMatrix> ddgap_dd = mortaradapter_->nodal_gap_deriv();

  std::shared_ptr<Core::LinAlg::SparseMatrix> dh_dd = std::make_shared<Core::LinAlg::SparseMatrix>(
      *ada_strDisp_to_lubDisp_->slave_dof_map(), 81, true, true);

  Coupling::Adapter::MatrixRowTransform()(*ddgap_dd, 1.0,
      Coupling::Adapter::CouplingMasterConverter(*ada_strDisp_to_lubDisp_), *dh_dd, false);
  dh_dd->complete(*(extractor()->map(0)), *ada_strDisp_to_lubDisp_->slave_dof_map());

  // Multiply with associated matrix
  std::shared_ptr<Core::LinAlg::SparseMatrix> k_ls_H =
      Core::LinAlg::matrix_multiply(*k_ls_linH, false, *dh_dd, false, false, false, true);

  // Add the contribution of film height derivative to the stiffness matrix
  Core::LinAlg::matrix_add(*k_ls_H, false, 1.0, *k_ls_, 1.0);

  //-------------------------------------------
  // 4.2 Discrete tangential velocity derivative
  //-----------------------------------------
  std::shared_ptr<Core::LinAlg::SparseMatrix> avTangVelDeriv = mortaradapter_->av_tang_vel_deriv();
  std::shared_ptr<Core::LinAlg::SparseMatrix> dst = std::make_shared<Core::LinAlg::SparseMatrix>(
      *lubrication_->lubrication_field()->discretization()->dof_row_map(1), 81, true, false);
  Coupling::Adapter::MatrixRowTransform().operator()(*avTangVelDeriv, 1.,
      Coupling::Adapter::CouplingMasterConverter(*ada_strDisp_to_lubDisp_), *dst, false);
  dst->complete(*structure_field()->dof_row_map(),
      *lubrication_->lubrication_field()->discretization()->dof_row_map(1));
  std::shared_ptr<Core::LinAlg::SparseMatrix> tmp =
      Core::LinAlg::matrix_multiply(*k_ls_linV, false, *dst, false, true);
  Core::LinAlg::matrix_add(*tmp, false, -1.0, *k_ls_, 1.0);


  k_ls_->complete(*(extractor()->map(0)),  // displacement dof map
      *(extractor()->map(1))               // pressure dof map
  );

  // Apply Dirichet to k_ls
  k_ls_->apply_dirichlet(
      *lubrication_->lubrication_field()->get_dbc_map_extractor()->cond_map(), false);
  if (inf_gap_toggle_lub_ != nullptr) k_ls_->apply_dirichlet(*inf_gap_toggle_lub_, false);

  // Assign k_ls to system matrix
  systemmatrix_->assign(1, 0, Core::LinAlg::DataAccess::Share, *k_ls_);

  // Finished...
  systemmatrix_->complete();

}  // setup_system_matrix()


/*----------------------------------------------------------------------*
 | setup RHS (like fsimon)                                  wirtz 01/16 |
 *----------------------------------------------------------------------*/
void EHL::Monolithic::setup_rhs()
{
  TEUCHOS_FUNC_TIME_MONITOR("EHL::Monolithic::setup_rhs");

  // create full monolithic rhs vector
  rhs_ = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map(), true);

  // fill the EHL rhs vector rhs_ with the single field rhs
  setup_vector(*rhs_, structure_field()->rhs(), lubrication_->lubrication_field()->rhs());
}  // setup_rhs()


/*----------------------------------------------------------------------*
 | solve linear EHL system                                  wirtz 01/16 |
 *----------------------------------------------------------------------*/
void EHL::Monolithic::linear_solve()
{
  // Solve for inc_ = [disi_,prei_]
  // Solve K_Teffdyn . IncX = -R  ===>  IncX_{n+1} with X=[d,T]
  // \f$x_{i+1} = x_i + \Delta x_i\f$
  Core::LinAlg::SolverParams solver_params;
  if (solveradapttol_ and (iter_ > 1))
  {
    solver_params.nonlin_tolerance = normrhs_;
    solver_params.nonlin_tolerance = tolrhs_;
    solver_params.lin_tol_better = solveradaptolbetter_;
  }

  // Dirichlet boundary conditions are already applied to EHL system, i.e. EHL
  // system is prepared for solve, i.e. EHL systemmatrix, EHL rhs, EHL inc
  // --> in prepare_system_for_newton_solve(): done for rhs and diagonal blocks
  // --> in setup_system_matrix() done for off-diagonal blocks k_sl, k_ls

  // apply Dirichlet BCs to system of equations
  iterinc_->put_scalar(0.0);  // Useful? depends on solver and more

  // Infnormscaling: scale system before solving
  scale_system(*systemmatrix_, *rhs_);

  // merge blockmatrix to SparseMatrix and solve
  std::shared_ptr<Core::LinAlg::SparseMatrix> sparse = systemmatrix_->merge();

  // standard solver call
  solver_params.refactor = true;
  solver_params.reset = iter_ == 1;
  solver_->solve(sparse, iterinc_, rhs_, solver_params);

  // Infnormscaling: unscale system after solving
  unscale_solution(*systemmatrix_, *iterinc_, *rhs_);

}  // linear_solve()


/*----------------------------------------------------------------------*
 | setup vector of the structure and lubrication field      wirtz 01/16 |
 *----------------------------------------------------------------------*/
void EHL::Monolithic::setup_vector(Core::LinAlg::Vector<double>& f,
    std::shared_ptr<const Core::LinAlg::Vector<double>> sv,
    std::shared_ptr<const Core::LinAlg::Vector<double>> lv)
{
  // extract dofs of the two fields
  // and put the structural/lubrication field vector into the global vector f
  // noticing the block number
  extractor()->insert_vector(*sv, 0, f);
  extractor()->insert_vector(*lv, 1, f);

}  // setup_vector()


/*----------------------------------------------------------------------*
 | check convergence of Newton iteration (public)           wirtz 01/16 |
 *----------------------------------------------------------------------*/
bool EHL::Monolithic::converged()
{
  // check for single norms
  bool convrhs = false;
  bool convinc = false;
  bool convstrrhs = false;
  bool convdisp = false;
  bool convlubricationrhs = false;
  bool convpre = false;

  // ----------------------------------------------------------- EHL test
  // residual EHL forces
  switch (normtyperhs_)
  {
    case EHL::convnorm_abs:
      convrhs = normrhs_ < tolrhs_;
      break;
    case EHL::convnorm_rel:
      convrhs = normrhs_ < std::max(tolrhs_ * normrhsiter0_, 1.0e-15);
      break;
    case EHL::convnorm_mix:
      convrhs = ((normrhs_ < tolrhs_) and (normrhs_ < std::max(normrhsiter0_ * tolrhs_, 1.0e-15)));
      break;
    default:
      FOUR_C_THROW("Cannot check for convergence of residual forces!");
      break;
  }

  // residual EHL increments
  switch (normtypeinc_)
  {
    case EHL::convnorm_abs:
      convinc = norminc_ < tolinc_;
      break;
    case EHL::convnorm_rel:
      convinc = norminc_ < std::max(norminciter0_ * tolinc_, 1e-15);
      break;
    case EHL::convnorm_mix:
      convinc = norminc_ < std::max(norminciter0_ * tolinc_, 1e-15);
      break;
    default:
      FOUR_C_THROW("Cannot check for convergence of increments!");
      break;
  }  // switch (normtypeinc_)

  // -------------------------------------------------- single field test
  // ---------------------------------------------------------- structure
  // structural residual forces
  switch (normtypestrrhs_)
  {
    case Inpar::Solid::convnorm_abs:
      convstrrhs = normstrrhs_ < tolstrrhs_;
      break;
    case Inpar::Solid::convnorm_rel:
      convstrrhs = normstrrhs_ < std::max(normstrrhsiter0_ * tolstrrhs_, 1e-15);
      break;
    case Inpar::Solid::convnorm_mix:
      convstrrhs = ((normstrrhs_ < tolstrrhs_) or
                    (normstrrhs_ < std::max(normstrrhsiter0_ * tolstrrhs_, 1e-15)));
      break;
    default:
      FOUR_C_THROW("Cannot check for convergence of residual forces!");
      break;
  }  // switch (normtypestrrhs_)

  // residual displacements
  switch (normtypedisi_)
  {
    case Inpar::Solid::convnorm_abs:
      convdisp = normdisi_ < toldisi_;
      break;
    case Inpar::Solid::convnorm_rel:
      convdisp = normdisi_ < std::max(normdisiiter0_ * toldisi_, 1e-15);
      break;
    case Inpar::Solid::convnorm_mix:
      convdisp =
          ((normdisi_ < toldisi_) or (normdisi_ < std::max(normdisiiter0_ * toldisi_, 1e-15)));
      break;
    default:
      FOUR_C_THROW("Cannot check for convergence of displacements!");
      break;
  }  // switch (normtypedisi_)

  // ------------------------------------------------------------- lubrication
  // lubrication residual forces
  switch (normtypelubricationrhs_)
  {
    case Lubrication::convnorm_abs:
      convlubricationrhs = normlubricationrhs_ < tollubricationrhs_;
      break;
    case Lubrication::convnorm_rel:
      convlubricationrhs = normlubricationrhs_ < normlubricationrhsiter0_ * tollubricationrhs_;
      break;
    case Lubrication::convnorm_mix:
      convlubricationrhs = ((normlubricationrhs_ < tollubricationrhs_) or
                            (normlubricationrhs_ < normlubricationrhsiter0_ * tollubricationrhs_));
      break;
    default:
      FOUR_C_THROW("Cannot check for convergence of residual forces!");
      break;
  }  // switch (normtypelubricationrhs_)

  // residual pressures
  switch (normtypeprei_)
  {
    case Lubrication::convnorm_abs:
      convpre = normprei_ < tolprei_;
      break;
    case Lubrication::convnorm_rel:
      convpre = normprei_ < normpreiiter0_ * tolprei_;
      break;
    case Lubrication::convnorm_mix:
      convpre = ((normprei_ < tolprei_) or (normprei_ < normpreiiter0_ * tolprei_));
      break;
    default:
      FOUR_C_THROW("Cannot check for convergence of pressures!");
      break;
  }  // switch (normtypeprei_)

  // -------------------------------------------------------- convergence
  // combine increment-like and force-like residuals, combine EHL and single
  // field values
  bool conv = false;
  if (combincrhs_ == EHL::bop_and)
    conv = convinc and convrhs;
  else if (combincrhs_ == EHL::bop_or)
    conv = convinc or convrhs;
  else if (combincrhs_ == EHL::bop_coupl_and_single)
    conv = convinc and convrhs and convdisp and convstrrhs and convpre and convlubricationrhs;
  else if (combincrhs_ == EHL::bop_coupl_or_single)
    conv = (convinc and convrhs) or (convdisp and convstrrhs and convpre and convlubricationrhs);
  else if (combincrhs_ == EHL::bop_and_single)
    conv = convdisp and convstrrhs and convpre and convlubricationrhs;
  else if (combincrhs_ == EHL::bop_or_single)
    conv = (convdisp or convstrrhs or convpre or convlubricationrhs);
  else
    FOUR_C_THROW("Something went terribly wrong with binary operator!");

  // return things
  return conv;

}  // Converged()


/*----------------------------------------------------------------------*
 | print Newton-Raphson iteration to screen and error file  wirtz 01/16 |
 | originally by lw 12/07, tk 01/08                                     |
 *----------------------------------------------------------------------*/
void EHL::Monolithic::print_newton_iter()
{
  // print to standard out
  // replace myrank_ here general by Core::Communication::my_mpi_rank(Comm())
  if ((Core::Communication::my_mpi_rank(get_comm()) == 0) and print_screen_every() and
      (step() % print_screen_every() == 0) and printiter_)
  {
    if (iter_ == 1) print_newton_iter_header(stdout);
    print_newton_iter_text(stdout);
  }

}  // print_newton_iter()


/*----------------------------------------------------------------------*
 | print Newton-Raphson iteration to screen and error file  wirtz 01/16 |
 | originally by lw 12/07, tk 01/08                                     |
 *----------------------------------------------------------------------*/
void EHL::Monolithic::print_newton_iter_header(FILE* ofile)
{
  // open outstringstream
  std::ostringstream oss;

  // enter converged state etc
  oss << std::setw(6) << "numiter";

  // ---------------------------------------------------------------- EHL
  // different style due relative or absolute error checking
  // displacement
  switch (normtyperhs_)
  {
    case EHL::convnorm_abs:
      oss << std::setw(15) << "abs-res-norm";
      break;
    case EHL::convnorm_rel:
      oss << std::setw(15) << "rel-res-norm";
      break;
    case EHL::convnorm_mix:
      oss << std::setw(15) << "mix-res-norm";
      break;
    default:
      FOUR_C_THROW("You should not turn up here.");
      break;
  }

  switch (normtypeinc_)
  {
    case EHL::convnorm_abs:
      oss << std::setw(15) << "abs-inc-norm";
      break;
    case EHL::convnorm_rel:
      oss << std::setw(15) << "rel-inc-norm";
      break;
    default:
      FOUR_C_THROW("You should not turn up here.");
      break;
  }

  // -------------------------------------------------- single field test
  // ---------------------------------------------------------- structure
  switch (normtypestrrhs_)
  {
    case Inpar::Solid::convnorm_rel:
      oss << std::setw(18) << "rel-str-res-norm";
      break;
    case Inpar::Solid::convnorm_abs:
      oss << std::setw(18) << "abs-str-res-norm";
      break;
    case Inpar::Solid::convnorm_mix:
      oss << std::setw(18) << "mix-str-res-norm";
      break;
    default:
      FOUR_C_THROW("You should not turn up here.");
      break;
  }  // switch (normtypestrrhs_)

  switch (normtypedisi_)
  {
    case Inpar::Solid::convnorm_rel:
      oss << std::setw(16) << "rel-dis-norm";
      break;
    case Inpar::Solid::convnorm_abs:
      oss << std::setw(16) << "abs-dis-norm";
      break;
    case Inpar::Solid::convnorm_mix:
      oss << std::setw(16) << "mix-dis-norm";
      break;
    default:
      FOUR_C_THROW("You should not turn up here.");
      break;
  }  // switch (normtypedisi_)

  // ------------------------------------------------------------- lubrication
  switch (normtypelubricationrhs_)
  {
    case Lubrication::convnorm_rel:
      oss << std::setw(18) << "rel-lub-res-norm";
      break;
    case Lubrication::convnorm_abs:
      oss << std::setw(18) << "abs-lub-res-norm";
      break;
    case Lubrication::convnorm_mix:
      oss << std::setw(18) << "mix-lub-res-norm";
      break;
    default:
      FOUR_C_THROW("You should not turn up here.");
      break;
  }  // switch (normtypelubricationrhs_)

  switch (normtypeprei_)
  {
    case Lubrication::convnorm_rel:
      oss << std::setw(16) << "rel-pre-norm";
      break;
    case Lubrication::convnorm_abs:
      oss << std::setw(16) << "abs-pre-norm";
      break;
    case Lubrication::convnorm_mix:
      oss << std::setw(16) << "mix-pre-norm";
      break;
    default:
      FOUR_C_THROW("You should not turn up here.");
      break;
  }  // switch (normtypeprei_)

  if (mortaradapter_->has_contact())
  {
    oss << std::setw(16) << "L2-Cont-Res";
    oss << std::setw(16) << "L2-Cont-Incr";
    oss << std::setw(11) << "#active";
    oss << std::setw(10) << "#slip";
  }

  // add solution time
  oss << std::setw(12) << "ts";
  oss << std::setw(12) << "wct";

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
 | print Newton-Raphson iteration to screen                 wirtz 01/16 |
 | originally by lw 12/07, tk 01/08                                     |
 *----------------------------------------------------------------------*/
void EHL::Monolithic::print_newton_iter_text(FILE* ofile)
{
  // open outstringstream
  std::ostringstream oss;

  // enter converged state etc
  oss << std::setw(7) << iter_;

  // different style due relative or absolute error checking

  // ----------------------------------------------- test coupled problem
  switch (normtyperhs_)
  {
    case EHL::convnorm_abs:
      oss << std::setw(15) << std::setprecision(5) << std::scientific << normrhs_;
      break;
    case EHL::convnorm_rel:
      oss << std::setw(15) << std::setprecision(5) << std::scientific << normrhs_ / normrhsiter0_;
      break;
    case EHL::convnorm_mix:
      oss << std::setw(15) << std::setprecision(5) << std::scientific
          << std::min(normrhs_, normrhs_ / normrhsiter0_);
      break;
    default:
      FOUR_C_THROW("You should not turn up here.");
      break;
  }

  switch (normtypeinc_)
  {
    case EHL::convnorm_abs:
      oss << std::setw(15) << std::setprecision(5) << std::scientific << norminc_;
      break;
    case EHL::convnorm_rel:
      oss << std::setw(15) << std::setprecision(5) << std::scientific << norminc_ / norminciter0_;
      break;
    case EHL::convnorm_mix:
      oss << std::setw(15) << std::setprecision(5) << std::scientific
          << std::min(norminc_, norminc_ / norminciter0_);
      break;
    default:
      FOUR_C_THROW("You should not turn up here.");
      break;
  }  // switch (normtypeinc_)

  // ------------------------------------------------- test single fields
  // ---------------------------------------------------------- structure
  // different style due relative or absolute error checking
  // displacement
  switch (normtypestrrhs_)
  {
    case Inpar::Solid::convnorm_abs:
      oss << std::setw(18) << std::setprecision(5) << std::scientific << normstrrhs_;
      break;
    case Inpar::Solid::convnorm_rel:
      oss << std::setw(18) << std::setprecision(5) << std::scientific
          << normstrrhs_ / normstrrhsiter0_;
      break;
    case Inpar::Solid::convnorm_mix:
      oss << std::setw(18) << std::setprecision(5) << std::scientific
          << std::min(normstrrhs_, normstrrhs_ / normstrrhsiter0_);
      break;
    default:
      FOUR_C_THROW("You should not turn up here.");
      break;
  }  // switch (normtypestrrhs_)

  switch (normtypedisi_)
  {
    case Inpar::Solid::convnorm_abs:
      oss << std::setw(16) << std::setprecision(5) << std::scientific << normdisi_;
      break;
    case Inpar::Solid::convnorm_rel:
      oss << std::setw(16) << std::setprecision(5) << std::scientific << normdisi_ / normdisiiter0_;
      break;
    case Inpar::Solid::convnorm_mix:
      oss << std::setw(16) << std::setprecision(5) << std::scientific
          << std::min(normdisi_, normdisi_ / normdisiiter0_);
      break;
    default:
      FOUR_C_THROW("You should not turn up here.");
      break;
  }  // switch (normtypedisi_)

  // ------------------------------------------------------------- lubrication
  switch (normtypelubricationrhs_)
  {
    case Lubrication::convnorm_abs:
      oss << std::setw(18) << std::setprecision(5) << std::scientific << normlubricationrhs_;
      break;
    case Lubrication::convnorm_rel:
      oss << std::setw(18) << std::setprecision(5) << std::scientific
          << normlubricationrhs_ / normlubricationrhsiter0_;
      break;
    case Lubrication::convnorm_mix:
      oss << std::setw(18) << std::setprecision(5) << std::scientific
          << std::min(normlubricationrhs_, normlubricationrhs_ / normlubricationrhsiter0_);
      break;
    default:
      FOUR_C_THROW("You should not turn up here.");
      break;
  }  // switch (normtypelubricationrhs_)

  switch (normtypeprei_)
  {
    case Lubrication::convnorm_abs:
      oss << std::setw(16) << std::setprecision(5) << std::scientific << normprei_;
      break;
    case Lubrication::convnorm_rel:
      oss << std::setw(16) << std::setprecision(5) << std::scientific << normprei_ / normpreiiter0_;
      break;
    case Lubrication::convnorm_mix:
      oss << std::setw(16) << std::setprecision(5) << std::scientific
          << std::min(normprei_, normprei_ / normpreiiter0_);
      break;
    default:
      FOUR_C_THROW("You should not turn up here.");
      break;
  }  // switch (normtypeprei_)


  if (mortaradapter_->has_contact())
  {
    oss << std::setw(16) << std::setprecision(5) << std::scientific
        << mortaradapter_->contact_res();
    oss << std::setw(16) << std::setprecision(5) << std::scientific
        << mortaradapter_->contact_incr();
    oss << std::setw(11) << mortaradapter_->active_contact();
    oss << std::setw(10) << mortaradapter_->slip_contact();
  }

  // add solution time of to print to screen
  oss << std::setw(12) << std::setprecision(2) << std::scientific << dtsolve_;
  oss << std::setw(12) << std::setprecision(2) << std::scientific
      << timernewton_.totalElapsedTime(true);

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
 | print statistics of converged NRI                        wirtz 01/16 |
 | originally by bborn 08/09                                            |
 *----------------------------------------------------------------------*/
void EHL::Monolithic::print_newton_conv()
{
  // somebody did the door
  return;
}  // print_newton_conv()



/*----------------------------------------------------------------------*
 | evaluate lubrication-mechanical system matrix at state   wirtz 01/16 |
 *----------------------------------------------------------------------*/
void EHL::Monolithic::apply_lubrication_coupl_matrix(
    std::shared_ptr<Core::LinAlg::SparseMatrix> matheight,
    std::shared_ptr<Core::LinAlg::SparseMatrix> matvel)
{
  // create the parameters for the discretization
  Teuchos::ParameterList lparams;
  // action for elements
  Lubrication::Action action = Lubrication::calc_lubrication_coupltang;
  lparams.set<Lubrication::Action>("action", action);
  // other parameters that might be needed by the elements
  lparams.set("delta time", dt());

  // provide bool whether lubrication grid is displaced or not
  lparams.set<bool>("isale", true);

  lubrication_->lubrication_field()->discretization()->clear_state(true);
  // set the variables that are needed by the elements
  lubrication_->lubrication_field()->discretization()->set_state(
      0, "prenp", *lubrication_->lubrication_field()->prenp());

  set_struct_solution(structure_->dispnp());

  // build specific assemble strategy for the lubrication-mechanical system matrix
  // from the point of view of lubrication_->LubricationField:
  // thermdofset = 0, structdofset = 1
  Core::FE::AssembleStrategy lubricationstrategy(0,  // pressure dofset for row
      1,                                             // displacement dofset for column
      matheight,                                     // lubrication-mechanical matrix
      matvel,                                        // no other matrix or vectors
      nullptr, nullptr, nullptr);

  // evaluate the lubrication-mechanical system matrix on the lubrication element
  lubrication_->lubrication_field()->discretization()->evaluate(lparams, lubricationstrategy);

  matheight->complete(*(lubrication_->lubrication_field()->discretization()->dof_row_map(1)),
      *(lubrication_->lubrication_field()->discretization()->dof_row_map(0)));

  matvel->complete(*(lubrication_->lubrication_field()->discretization()->dof_row_map(1)),
      *(lubrication_->lubrication_field()->discretization()->dof_row_map(0)));

  lubrication_->lubrication_field()->discretization()->clear_state(true);

  return;
}  // apply_lubrication_coupl_matrix()

/*----------------------------------------------------------------------*
 | map containing the dofs with Dirichlet BC                wirtz 01/16 |
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Map> EHL::Monolithic::combined_dbc_map()
{
  const std::shared_ptr<const Core::LinAlg::Map> scondmap =
      structure_field()->get_dbc_map_extractor()->cond_map();
  const std::shared_ptr<const Core::LinAlg::Map> lcondmap =
      lubrication_->lubrication_field()->get_dbc_map_extractor()->cond_map();
  std::shared_ptr<Core::LinAlg::Map> condmap = Core::LinAlg::merge_map(scondmap, lcondmap, false);
  return condmap;

}  // combined_dbc_map()

/*----------------------------------------------------------------------*
 | scale system, i.e. apply infnorm scaling to linear       wirtz 01/16 |
 | block system before solving system                                   |
 *----------------------------------------------------------------------*/
void EHL::Monolithic::scale_system(
    Core::LinAlg::BlockSparseMatrixBase& mat, Core::LinAlg::Vector<double>& b)
{
  // should we scale the system?
  const bool scaling_infnorm = ehldynmono_.get<bool>("INFNORMSCALING");

  if (scaling_infnorm)
  {
    // TODO: The matrices are modified here. Do we have to change them back later on?

    Core::LinAlg::SparseMatrix& A_00 = mat.matrix(0, 0);
    srowsum_ = std::make_shared<Core::LinAlg::Vector<double>>(A_00.row_map(), false);
    scolsum_ = std::make_shared<Core::LinAlg::Vector<double>>(A_00.row_map(), false);
    A_00.inv_row_sums(*srowsum_);
    A_00.inv_col_sums(*scolsum_);
    A_00.left_scale(*srowsum_);
    A_00.right_scale(*scolsum_);
    mat.matrix(0, 1).left_scale(*srowsum_);
    mat.matrix(1, 0).right_scale(*scolsum_);

    Core::LinAlg::SparseMatrix& A_11 = mat.matrix(1, 1);
    lrowsum_ = std::make_shared<Core::LinAlg::Vector<double>>(A_11.row_map(), false);
    lcolsum_ = std::make_shared<Core::LinAlg::Vector<double>>(A_11.row_map(), false);
    A_11.inv_row_sums(*lrowsum_);
    A_11.inv_col_sums(*lcolsum_);
    A_11.left_scale(*lrowsum_);
    A_11.right_scale(*lcolsum_);
    mat.matrix(1, 0).left_scale(*lrowsum_);
    mat.matrix(0, 1).right_scale(*lcolsum_);

    std::shared_ptr<Core::LinAlg::Vector<double>> sx = extractor()->extract_vector(b, 0);
    std::shared_ptr<Core::LinAlg::Vector<double>> lx = extractor()->extract_vector(b, 1);

    sx->multiply(1.0, *srowsum_, *sx, 0.0);
    lx->multiply(1.0, *lrowsum_, *lx, 0.0);

    extractor()->insert_vector(*sx, 0, b);
    extractor()->insert_vector(*lx, 1, b);
  }
}


/*----------------------------------------------------------------------*
 | unscale solution after solving the linear system         wirtz 01/16 |
 *----------------------------------------------------------------------*/
void EHL::Monolithic::unscale_solution(Core::LinAlg::BlockSparseMatrixBase& mat,
    Core::LinAlg::Vector<double>& x, Core::LinAlg::Vector<double>& b)
{
  const bool scaling_infnorm = ehldynmono_.get<bool>("INFNORMSCALING");

  if (scaling_infnorm)
  {
    std::shared_ptr<Core::LinAlg::Vector<double>> sy = extractor()->extract_vector(x, 0);
    std::shared_ptr<Core::LinAlg::Vector<double>> ly = extractor()->extract_vector(x, 1);

    sy->multiply(1.0, *scolsum_, *sy, 0.0);
    ly->multiply(1.0, *lcolsum_, *ly, 0.0);

    extractor()->insert_vector(*sy, 0, x);
    extractor()->insert_vector(*ly, 1, x);

    std::shared_ptr<Core::LinAlg::Vector<double>> sx = extractor()->extract_vector(b, 0);
    std::shared_ptr<Core::LinAlg::Vector<double>> lx = extractor()->extract_vector(b, 1);

    sx->reciprocal_multiply(1.0, *srowsum_, *sx, 0.0);
    lx->reciprocal_multiply(1.0, *lrowsum_, *lx, 0.0);

    extractor()->insert_vector(*sx, 0, b);
    extractor()->insert_vector(*lx, 1, b);

    Core::LinAlg::SparseMatrix& A_00 = mat.matrix(0, 0);
    srowsum_->reciprocal(*srowsum_);
    scolsum_->reciprocal(*scolsum_);
    A_00.left_scale(*srowsum_);
    A_00.right_scale(*scolsum_);
    mat.matrix(0, 1).left_scale(*srowsum_);
    mat.matrix(1, 0).right_scale(*scolsum_);

    Core::LinAlg::SparseMatrix& A_11 = mat.matrix(1, 1);
    lrowsum_->reciprocal(*lrowsum_);
    lcolsum_->reciprocal(*lcolsum_);
    A_11.left_scale(*lrowsum_);
    A_11.right_scale(*lcolsum_);
    mat.matrix(1, 0).left_scale(*lrowsum_);
    mat.matrix(0, 1).right_scale(*lcolsum_);
  }
}


/*----------------------------------------------------------------------*
 | calculate vector norm                                    wirtz 01/16 |
 *----------------------------------------------------------------------*/
double EHL::Monolithic::calculate_vector_norm(
    const EHL::VectorNorm norm, Core::LinAlg::Vector<double>& vect)
{
  // L1 norm
  // norm = sum_0^i vect[i]
  if (norm == EHL::norm_l1)
  {
    double vectnorm;
    vect.norm_1(&vectnorm);
    return vectnorm;
  }
  // L2/Euclidian norm
  // norm = sqrt{sum_0^i vect[i]^2 }
  else if (norm == EHL::norm_l2)
  {
    double vectnorm;
    vect.norm_2(&vectnorm);
    return vectnorm;
  }
  // RMS norm
  // norm = sqrt{sum_0^i vect[i]^2 }/ sqrt{length_vect}
  else if (norm == EHL::norm_rms)
  {
    double vectnorm;
    vect.norm_2(&vectnorm);
    return vectnorm / sqrt((double)vect.global_length());
  }
  // infinity/maximum norm
  // norm = max( vect[i] )
  else if (norm == EHL::norm_inf)
  {
    double vectnorm;
    vect.norm_inf(&vectnorm);
    return vectnorm;
  }
  // norm = sum_0^i vect[i]/length_vect
  else if (norm == EHL::norm_l1_scaled)
  {
    double vectnorm;
    vect.norm_1(&vectnorm);
    return vectnorm / ((double)vect.global_length());
  }
  else
  {
    FOUR_C_THROW("Cannot handle vector norm");
    return 0;
  }
}  // calculate_vector_norm()


/*----------------------------------------------------------------------*
 | set parameters for EHL remaining constant over whole     wirtz 01/16 |
 | simulation                                                           |
 *----------------------------------------------------------------------*/
void EHL::Monolithic::set_default_parameters()
{
  // time parameters
  // call the EHL parameter list
  const Teuchos::ParameterList& ldyn = Global::Problem::instance()->lubrication_dynamic_params();

  // get the parameters for the Newton iteration
  itermax_ = ehldyn_.get<int>("ITEMAX");
  itermin_ = ehldyn_.get<int>("ITEMIN");

  // what kind of norm do we wanna test for coupled EHL problem
  normtypeinc_ = Teuchos::getIntegralValue<EHL::ConvNorm>(ehldynmono_, "NORM_INC");
  normtyperhs_ = Teuchos::getIntegralValue<EHL::ConvNorm>(ehldynmono_, "NORM_RESF");
  // what kind of norm do we wanna test for the single fields
  normtypedisi_ = Teuchos::getIntegralValue<Inpar::Solid::ConvNorm>(sdyn_, "NORM_DISP");
  normtypestrrhs_ = Teuchos::getIntegralValue<Inpar::Solid::ConvNorm>(sdyn_, "NORM_RESF");
  auto striternorm = Teuchos::getIntegralValue<Inpar::Solid::VectorNorm>(sdyn_, "ITERNORM");
  normtypeprei_ = Teuchos::getIntegralValue<Lubrication::ConvNorm>(ldyn, "NORM_PRE");
  normtypelubricationrhs_ = Teuchos::getIntegralValue<Lubrication::ConvNorm>(ldyn, "NORM_RESF");
  auto lubricationiternorm = Teuchos::getIntegralValue<Lubrication::VectorNorm>(ldyn, "ITERNORM");
  // in total when do we reach a converged state for complete problem
  combincrhs_ = Teuchos::getIntegralValue<EHL::BinaryOp>(ehldynmono_, "NORMCOMBI_RESFINC");

  switch (combincrhs_)
  {
    case EHL::bop_and:
    {
      if (Core::Communication::my_mpi_rank(get_comm()) == 0)
        std::cout << "Convergence test of EHL:\n res, inc with 'AND'." << std::endl;
      break;
    }
    case EHL::bop_or:
    {
      if (Core::Communication::my_mpi_rank(get_comm()) == 0)
        std::cout << "Convergence test of EHL:\n res, inc with 'OR'." << std::endl;
      break;
    }
    case EHL::bop_coupl_and_single:
    {
      if (Core::Communication::my_mpi_rank(get_comm()) == 0)
        std::cout << "Convergence test of EHL:\n res, inc, str-res, lub-res, dis, pre with 'AND'."
                  << std::endl;
      break;
    }
    case EHL::bop_coupl_or_single:
    {
      if (Core::Communication::my_mpi_rank(get_comm()) == 0)
        std::cout << "Convergence test of EHL:\n (res, inc) or (str-res, lub-res, dis, pre)."
                  << std::endl;
      break;
    }
    case EHL::bop_and_single:
    {
      if (Core::Communication::my_mpi_rank(get_comm()) == 0)
        std::cout << "Convergence test of EHL:\n str-res, lub-res, dis, pre with 'AND'."
                  << std::endl;
      break;
    }
    case EHL::bop_or_single:
    {
      if (Core::Communication::my_mpi_rank(get_comm()) == 0)
        std::cout << "Convergence test of EHL:\n str-res, lub-res, dis, pre with 'OR'."
                  << std::endl;
      break;
    }
    default:
    {
      FOUR_C_THROW("Something went terribly wrong with binary operator!");
      break;
    }
  }  // switch (combincrhs_)

  // convert the single field norms to be used within EHL
  // what norm is used for structure
  switch (striternorm)
  {
    case Inpar::Solid::norm_l1:
      iternormstr_ = EHL::norm_l1;
      break;
    case Inpar::Solid::norm_l2:
      iternormstr_ = EHL::norm_l2;
      break;
    case Inpar::Solid::norm_rms:
      iternormstr_ = EHL::norm_rms;
      break;
    case Inpar::Solid::norm_inf:
      iternormstr_ = EHL::norm_inf;
      break;
    case Inpar::Solid::norm_vague:
    default:
      FOUR_C_THROW("STR norm is not determined");
      break;
  }  // switch (striternorm)

  // what norm is used for lubrication
  switch (lubricationiternorm)
  {
    case Lubrication::norm_l1:
      iternormlubrication_ = EHL::norm_l1;
      break;
    case Lubrication::norm_l2:
      iternormlubrication_ = EHL::norm_l2;
      break;
    case Lubrication::norm_rms:
      iternormlubrication_ = EHL::norm_rms;
      break;
    case Lubrication::norm_inf:
      iternormlubrication_ = EHL::norm_inf;
      break;
    case Lubrication::norm_vague:
    default:
    {
      FOUR_C_THROW("LUBRICATION norm is not determined.");
      break;
    }
  }  // switch (lubricationiternorm)

  // if scaled L1-norm is wished to be used
  if ((iternorm_ == EHL::norm_l1_scaled) and
      ((combincrhs_ == EHL::bop_coupl_and_single) or (combincrhs_ == EHL::bop_coupl_or_single)))
  {
    iternormstr_ = EHL::norm_l1_scaled;
    iternormlubrication_ = EHL::norm_l1_scaled;
  }

  // test the EHL-residual and the EHL-increment
  tolinc_ = ehldynmono_.get<double>("TOLINC");
  tolrhs_ = ehldynmono_.get<double>("CONVTOL");

  // get the single field tolerances from this field itselves
  toldisi_ = sdyn_.get<double>("TOLDISP");
  tolstrrhs_ = sdyn_.get<double>("TOLRES");
  tolprei_ = ldyn.get<double>("TOLPRE");
  tollubricationrhs_ = ldyn.get<double>("TOLRES");

  // initialise norms for coupled EHL problem
  normrhs_ = 0.0;
  normrhsiter0_ = 0.0;
  norminc_ = 0.0;
  norminciter0_ = 0.0;

  // initialise norms for single field tests
  normdisi_ = 0.0;
  normstrrhs_ = 0.0;
  normstrrhsiter0_ = 0.0;
  normprei_ = 0.0;
  normlubricationrhs_ = 0.0;
  normlubricationrhsiter0_ = 0.0;

  return;

}  // SetDefaultParameter()

/*----------------------------------------------------------------------*
 | calculate stresses, strains, energies                    wirtz 01/16 |
 *----------------------------------------------------------------------*/
void EHL::Monolithic::prepare_output(bool force_prepare)
{
  // prepare output (i.e. calculate stresses, strains, energies)
  structure_field()->prepare_output(force_prepare);

  // reset states
  structure_field()->discretization()->clear_state(true);
}
/*----------------------------------------------------------------------*/


void EHL::Monolithic::lin_pressure_force_disp(
    Core::LinAlg::SparseMatrix& ds_dd, Core::LinAlg::SparseMatrix& dm_dd)
{
  std::shared_ptr<Core::LinAlg::SparseMatrix> p_deriv_normal =
      std::make_shared<Core::LinAlg::SparseMatrix>(*mortaradapter_->nderiv_matrix());
  std::shared_ptr<Core::LinAlg::Vector<double>> p_full =
      std::make_shared<Core::LinAlg::Vector<double>>(
          *lubrication_->lubrication_field()->dof_row_map(1));
  if (lubrimaptransform_->Apply(*lubrication_->lubrication_field()->prenp(), *p_full))
    FOUR_C_THROW("apply failed");
  std::shared_ptr<Core::LinAlg::Vector<double>> p_exp =
      std::make_shared<Core::LinAlg::Vector<double>>(*mortaradapter_->slave_dof_map());
  p_exp = ada_strDisp_to_lubDisp_->slave_to_master(*p_full);
  p_deriv_normal->left_scale(*p_exp);
  if (p_deriv_normal->scale(-1.)) FOUR_C_THROW("scale failed");

  std::shared_ptr<Core::LinAlg::SparseMatrix> tmp = Core::LinAlg::matrix_multiply(
      *mortaradapter_->get_mortar_matrix_d(), true, *p_deriv_normal, false, false, false, true);
  if (!tmp) FOUR_C_THROW("matrix_multiply failed");
  Core::LinAlg::matrix_add(*tmp, false, +1., ds_dd, 1.);

  tmp = nullptr;
  tmp = Core::LinAlg::matrix_multiply(
      *mortaradapter_->get_mortar_matrix_m(), true, *p_deriv_normal, false, false, false, true);
  if (!tmp) FOUR_C_THROW("matrix_multiply failed");
  Core::LinAlg::matrix_add(*tmp, false, -1., dm_dd, 1.);

  return;
}

void EHL::Monolithic::lin_poiseuille_force_disp(
    Core::LinAlg::SparseMatrix& ds_dd, Core::LinAlg::SparseMatrix& dm_dd)
{
  std::shared_ptr<Core::LinAlg::Vector<double>> p_int =
      ada_strDisp_to_lubPres_->slave_to_master(*lubrication_->lubrication_field()->prenp());
  Core::LinAlg::Vector<double> p_int_full(*mortaradapter_->slave_dof_map());
  Core::LinAlg::export_to(*p_int, p_int_full);

  Core::LinAlg::Vector<double> nodal_gap(*mortaradapter_->slave_dof_map());
  slavemaptransform_->multiply(false, *mortaradapter_->nodal_gap(), nodal_gap);

  Core::LinAlg::Vector<double> grad_p(*mortaradapter_->slave_dof_map());
  if (mortaradapter_->surf_grad_matrix()->Apply(p_int_full, grad_p)) FOUR_C_THROW("apply failed");

  std::shared_ptr<Core::LinAlg::SparseMatrix> deriv_Poiseuille =
      std::make_shared<Core::LinAlg::SparseMatrix>(
          *mortaradapter_->slave_dof_map(), 81, false, false);

  Core::LinAlg::SparseMatrix derivH_gradP(*mortaradapter_->nodal_gap_deriv());
  derivH_gradP.left_scale(grad_p);
  Core::LinAlg::matrix_add(derivH_gradP, false, -.5, *deriv_Poiseuille, 1.);

  Core::LinAlg::Vector<double> p_int_full_col(*mortaradapter_->interface()->slave_col_dofs());
  Core::LinAlg::export_to(p_int_full, p_int_full_col);
  std::shared_ptr<Core::LinAlg::SparseMatrix> h_derivGrad_nodalP =
      mortaradapter_->assemble_surf_grad_deriv(p_int_full_col);
  h_derivGrad_nodalP->left_scale(nodal_gap);
  Core::LinAlg::matrix_add(*h_derivGrad_nodalP, false, -.5, *deriv_Poiseuille, 1.);

  deriv_Poiseuille->complete(*mortaradapter_->s_mdof_map(), *mortaradapter_->slave_dof_map());

  std::shared_ptr<Core::LinAlg::SparseMatrix> tmp = Core::LinAlg::matrix_multiply(
      *mortaradapter_->get_mortar_matrix_d(), true, *deriv_Poiseuille, false, false, false, true);
  if (!tmp) FOUR_C_THROW("matrix_multiply failed");
  Core::LinAlg::matrix_add(*tmp, false, +1., ds_dd, 1.);

  tmp = nullptr;
  tmp = Core::LinAlg::matrix_multiply(
      *mortaradapter_->get_mortar_matrix_m(), true, *deriv_Poiseuille, false, false, false, true);
  if (!tmp) FOUR_C_THROW("matrix_multiply failed");
  Core::LinAlg::matrix_add(*tmp, false, +1., dm_dd, 1.);
}

void EHL::Monolithic::lin_couette_force_disp(
    Core::LinAlg::SparseMatrix& ds_dd, Core::LinAlg::SparseMatrix& dm_dd)
{
  const int ndim = Global::Problem::instance()->n_dim();
  Core::FE::Discretization& lub_dis = *lubrication_->lubrication_field()->discretization();
  Core::LinAlg::Vector<double> visc_vec(*lubrication_->lubrication_field()->dof_row_map(1));
  for (int i = 0; i < lub_dis.node_row_map()->num_my_elements(); ++i)
  {
    Core::Nodes::Node* lnode = lub_dis.l_row_node(i);
    if (!lnode) FOUR_C_THROW("node not found");
    const double p = lubrication_->lubrication_field()->prenp()->local_values_as_span()[(
        lubrication_->lubrication_field()->prenp()->get_map().lid(lub_dis.dof(0, lnode, 0)))];

    std::shared_ptr<Core::Mat::Material> mat =
        lnode->adjacent_elements()[0].user_element()->material(0);
    if (!mat) FOUR_C_THROW("null pointer");
    std::shared_ptr<Mat::LubricationMat> lmat = std::dynamic_pointer_cast<Mat::LubricationMat>(mat);
    const double visc = lmat->compute_viscosity(p);

    for (int d = 0; d < ndim; ++d) visc_vec.replace_global_value(lub_dis.dof(1, lnode, d), visc);
  }
  std::shared_ptr<Core::LinAlg::Vector<double>> visc_vec_str =
      ada_strDisp_to_lubDisp_->slave_to_master(visc_vec);

  Core::LinAlg::Vector<double> height(*mortaradapter_->slave_dof_map());
  slavemaptransform_->multiply(false, *mortaradapter_->nodal_gap(), height);
  Core::LinAlg::Vector<double> h_inv(*mortaradapter_->slave_dof_map());
  h_inv.reciprocal(height);

  Core::LinAlg::Vector<double> hinv_visc(*mortaradapter_->slave_dof_map());
  hinv_visc.multiply(1., h_inv, *visc_vec_str, 0.);

  std::shared_ptr<Core::LinAlg::SparseMatrix> deriv_Couette =
      std::make_shared<Core::LinAlg::SparseMatrix>(
          *mortaradapter_->slave_dof_map(), 81, false, false);

  {
    Core::LinAlg::SparseMatrix tmp(*mortaradapter_->rel_tang_vel_deriv());
    tmp.left_scale(hinv_visc);
    tmp.scale(-1.);
    Core::LinAlg::matrix_add(tmp, false, 1., *deriv_Couette, 1.);
  }
  {
    Core::LinAlg::Vector<double> hinv_hinv_visc(*mortaradapter_->slave_dof_map());
    hinv_hinv_visc.multiply(1., h_inv, hinv_visc, 0.);
    Core::LinAlg::Vector<double> hinv_hinv_visc_vel(*mortaradapter_->slave_dof_map());
    hinv_hinv_visc_vel.multiply(1., hinv_hinv_visc, *mortaradapter_->rel_tang_vel(), 0.);
    Core::LinAlg::SparseMatrix tmp(*mortaradapter_->nodal_gap_deriv());
    tmp.left_scale(hinv_hinv_visc_vel);
    Core::LinAlg::matrix_add(tmp, false, 1., *deriv_Couette, 1.);
  }
  deriv_Couette->complete(*mortaradapter_->s_mdof_map(), *mortaradapter_->slave_dof_map());

  std::shared_ptr<Core::LinAlg::SparseMatrix> tmp = Core::LinAlg::matrix_multiply(
      *mortaradapter_->get_mortar_matrix_d(), true, *deriv_Couette, false, false, false, true);
  if (!tmp) FOUR_C_THROW("matrix_multiply failed");
  Core::LinAlg::matrix_add(*tmp, false, +1., ds_dd, 1.);

  tmp = nullptr;
  tmp = Core::LinAlg::matrix_multiply(
      *mortaradapter_->get_mortar_matrix_m(), true, *deriv_Couette, false, false, false, true);
  if (!tmp) FOUR_C_THROW("matrix_multiply failed");
  Core::LinAlg::matrix_add(*tmp, false, -1., dm_dd, 1.);
}

void EHL::Monolithic::lin_pressure_force_pres(
    Core::LinAlg::SparseMatrix& ds_dp, Core::LinAlg::SparseMatrix& dm_dp)
{
  std::shared_ptr<Core::LinAlg::SparseMatrix> tmp = std::make_shared<Core::LinAlg::SparseMatrix>(
      *mortaradapter_->slave_dof_map(), 81, false, false);

  Coupling::Adapter::MatrixRowTransform().operator()(*lubrimaptransform_, 1.,
      Coupling::Adapter::CouplingSlaveConverter(*ada_strDisp_to_lubDisp_), *tmp, false);

  tmp->complete(
      *lubrication_->lubrication_field()->dof_row_map(0), *mortaradapter_->slave_dof_map());

  tmp->left_scale(*mortaradapter_->normals());
  if (tmp->scale(-1.)) FOUR_C_THROW("scale failed");

  std::shared_ptr<Core::LinAlg::SparseMatrix> a = Core::LinAlg::matrix_multiply(
      *mortaradapter_->get_mortar_matrix_d(), true, *tmp, false, false, false, true);
  if (!a) FOUR_C_THROW("matrix_multiply failed");
  Core::LinAlg::matrix_add(*a, false, +1., ds_dp, 1.);

  a = nullptr;
  a = Core::LinAlg::matrix_multiply(
      *mortaradapter_->get_mortar_matrix_m(), true, *tmp, false, false, false, true);
  if (!a) FOUR_C_THROW("matrix_multiply failed");
  Core::LinAlg::matrix_add(*a, false, -1., dm_dp, 1.);
}

void EHL::Monolithic::lin_poiseuille_force_pres(
    Core::LinAlg::SparseMatrix& ds_dp, Core::LinAlg::SparseMatrix& dm_dp)
{
  Core::LinAlg::Vector<double> nodal_gap(*mortaradapter_->slave_dof_map());
  slavemaptransform_->multiply(false, *mortaradapter_->nodal_gap(), nodal_gap);

  Core::LinAlg::SparseMatrix m(*mortaradapter_->surf_grad_matrix());
  m.left_scale(nodal_gap);
  m.scale(-.5);

  {
    std::shared_ptr<const Core::LinAlg::Map> r = mortaradapter_->slave_dof_map();
    std::shared_ptr<const Core::LinAlg::Map> d = lubrication_->lubrication_field()->dof_row_map(0);
    std::shared_ptr<Core::LinAlg::SparseMatrix> a = Core::LinAlg::matrix_multiply(
        *mortaradapter_->get_mortar_matrix_d(), true, m, false, true, false, true);

    Core::LinAlg::SparseMatrix b(a->row_map(), 81, false, false);

    Coupling::Adapter::MatrixColTransform().operator()(a->row_map(), a->col_map(), *a, 1.,
        Coupling::Adapter::CouplingMasterConverter(*ada_strDisp_to_lubPres_), b, true, false);
    b.complete(*d, *r);

    ds_dp.un_complete();
    Core::LinAlg::matrix_add(b, false, 1., ds_dp, 1.);
    ds_dp.complete(*d, *r);
  }

  {
    std::shared_ptr<const Core::LinAlg::Map> r = mortaradapter_->master_dof_map();
    std::shared_ptr<const Core::LinAlg::Map> d = lubrication_->lubrication_field()->dof_row_map(0);
    std::shared_ptr<Core::LinAlg::SparseMatrix> a = Core::LinAlg::matrix_multiply(
        *mortaradapter_->get_mortar_matrix_m(), true, m, false, true, false, true);

    Core::LinAlg::SparseMatrix b(a->row_map(), 81, false, false);

    Coupling::Adapter::MatrixColTransform().operator()(a->row_map(), a->col_map(), *a, 1.,
        Coupling::Adapter::CouplingMasterConverter(*ada_strDisp_to_lubPres_), b, true, false);
    b.complete(*d, *r);

    dm_dp.un_complete();
    Core::LinAlg::matrix_add(b, false, 1., dm_dp, 1.);
    dm_dp.complete(*d, *r);
  }
  return;
}

void EHL::Monolithic::lin_couette_force_pres(
    Core::LinAlg::SparseMatrix& ds_dp, Core::LinAlg::SparseMatrix& dm_dp)
{
  const int ndim = Global::Problem::instance()->n_dim();
  const std::shared_ptr<const Core::LinAlg::Vector<double>> relVel = mortaradapter_->rel_tang_vel();
  Core::LinAlg::Vector<double> height(*mortaradapter_->slave_dof_map());
  slavemaptransform_->multiply(false, *mortaradapter_->nodal_gap(), height);
  Core::LinAlg::Vector<double> h_inv(*mortaradapter_->slave_dof_map());
  h_inv.reciprocal(height);
  Core::LinAlg::Vector<double> hinv_relV(*mortaradapter_->slave_dof_map());
  hinv_relV.multiply(1., h_inv, *relVel, 0.);

  Core::FE::Discretization& lub_dis = *lubrication_->lubrication_field()->discretization();
  std::shared_ptr<Core::LinAlg::SparseMatrix> dVisc_dp =
      std::make_shared<Core::LinAlg::SparseMatrix>(*(lub_dis.dof_row_map(1)), 81);

  for (int i = 0; i < lub_dis.node_row_map()->num_my_elements(); ++i)
  {
    Core::Nodes::Node* lnode = lub_dis.l_row_node(i);
    if (!lnode) FOUR_C_THROW("node not found");
    const double p = lubrication_->lubrication_field()->prenp()->local_values_as_span()[(
        lubrication_->lubrication_field()->prenp()->get_map().lid(lub_dis.dof(0, lnode, 0)))];

    std::shared_ptr<Core::Mat::Material> mat =
        lnode->adjacent_elements()[0].user_element()->material(0);
    if (!mat) FOUR_C_THROW("null pointer");
    std::shared_ptr<Mat::LubricationMat> lmat = std::dynamic_pointer_cast<Mat::LubricationMat>(mat);
    const double visc = lmat->compute_viscosity(p);
    const double dvisc_dp = lmat->compute_viscosity_deriv(p, visc);

    for (int d = 0; d < ndim; ++d)
      dVisc_dp->assemble(dvisc_dp, lub_dis.dof(1, lnode, d), lub_dis.dof(0, lnode, 0));
  }
  dVisc_dp->complete(*lub_dis.dof_row_map(0), *lub_dis.dof_row_map(1));

  std::shared_ptr<Core::LinAlg::SparseMatrix> dVisc_str_dp =
      std::make_shared<Core::LinAlg::SparseMatrix>(*mortaradapter_->slave_dof_map(), 81);

  Coupling::Adapter::MatrixRowTransform().operator()(*dVisc_dp, 1.,
      Coupling::Adapter::CouplingSlaveConverter(*ada_strDisp_to_lubDisp_), *dVisc_str_dp, false);

  dVisc_str_dp->complete(*lub_dis.dof_row_map(0), *mortaradapter_->slave_dof_map());

  dVisc_str_dp->left_scale(hinv_relV);

  {
    std::shared_ptr<const Core::LinAlg::Map> r = mortaradapter_->slave_dof_map();
    std::shared_ptr<const Core::LinAlg::Map> d = lubrication_->lubrication_field()->dof_row_map(0);

    ds_dp.un_complete();
    Core::LinAlg::matrix_add(*Core::LinAlg::matrix_multiply(*mortaradapter_->get_mortar_matrix_d(),
                                 true, *dVisc_str_dp, false, true, false, true),
        false, 1., ds_dp, 1.);
    ds_dp.complete(*d, *r);
  }

  {
    std::shared_ptr<const Core::LinAlg::Map> r = mortaradapter_->master_dof_map();
    std::shared_ptr<const Core::LinAlg::Map> d = lubrication_->lubrication_field()->dof_row_map(0);

    dm_dp.un_complete();
    Core::LinAlg::matrix_add(*Core::LinAlg::matrix_multiply(*mortaradapter_->get_mortar_matrix_m(),
                                 true, *dVisc_str_dp, false, true, false, true),
        false, -1., dm_dp, 1.);
    dm_dp.complete(*d, *r);
  }
}


/*----------------------------------------------------------------------*
 |                                                                      |
 *----------------------------------------------------------------------*/
void EHL::Monolithic::apply_dbc()
{
  auto k_ss = std::make_shared<Core::LinAlg::SparseMatrix>(systemmatrix_->matrix(0, 0));
  auto k_sl = std::make_shared<Core::LinAlg::SparseMatrix>(systemmatrix_->matrix(0, 1));
  auto k_ls = std::make_shared<Core::LinAlg::SparseMatrix>(systemmatrix_->matrix(1, 0));
  auto k_ll = std::make_shared<Core::LinAlg::SparseMatrix>(systemmatrix_->matrix(1, 1));

  k_ss->apply_dirichlet(*structure_field()->get_dbc_map_extractor()->cond_map(), true);
  k_sl->apply_dirichlet(*structure_field()->get_dbc_map_extractor()->cond_map(), false);
  k_ls->apply_dirichlet(
      *lubrication_->lubrication_field()->get_dbc_map_extractor()->cond_map(), false);
  k_ll->apply_dirichlet(
      *lubrication_->lubrication_field()->get_dbc_map_extractor()->cond_map(), true);

  if (inf_gap_toggle_lub_ != nullptr)
  {
    k_ls->apply_dirichlet(*inf_gap_toggle_lub_, false);
    k_ll->apply_dirichlet(*inf_gap_toggle_lub_, true);
  }

  systemmatrix_->un_complete();
  systemmatrix_->assign(0, 0, Core::LinAlg::DataAccess::Share, *k_ss);
  systemmatrix_->assign(0, 1, Core::LinAlg::DataAccess::Share, *k_sl);
  systemmatrix_->assign(1, 0, Core::LinAlg::DataAccess::Share, *k_ls);
  systemmatrix_->assign(1, 1, Core::LinAlg::DataAccess::Share, *k_ll);
  systemmatrix_->complete();


  Core::LinAlg::apply_dirichlet_to_system(
      *rhs_, *zeros_, *structure_field()->get_dbc_map_extractor()->cond_map());
  Core::LinAlg::apply_dirichlet_to_system(
      *rhs_, *zeros_, *lubrication_->lubrication_field()->get_dbc_map_extractor()->cond_map());

  if (inf_gap_toggle_lub_ != nullptr)
    for (int i = 0; i < inf_gap_toggle_lub_->local_length(); ++i)
      if (abs(inf_gap_toggle_lub_->local_values_as_span()[i]) > 1.e-12)
        rhs_->replace_global_value(inf_gap_toggle_lub_->get_map().gid(i), 0.);
}

FOUR_C_NAMESPACE_CLOSE

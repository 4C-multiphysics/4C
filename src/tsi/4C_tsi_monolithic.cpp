// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_tsi_monolithic.hpp"

#include "4C_adapter_str_structure_new.hpp"
#include "4C_comm_mpi_utils.hpp"
#include "4C_contact_abstract_strategy.hpp"
#include "4C_contact_interface.hpp"
#include "4C_contact_lagrange_strategy_tsi.hpp"
#include "4C_contact_meshtying_contact_bridge.hpp"
#include "4C_contact_node.hpp"
#include "4C_fem_condition_locsys.hpp"
#include "4C_fem_discretization_nullspace.hpp"
#include "4C_fem_general_assemblestrategy.hpp"
#include "4C_fem_general_elements_paramsminimal.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_blocksparsematrix.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linear_solver_method.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_mortar_manager_base.hpp"
#include "4C_mortar_multifield_coupling.hpp"
#include "4C_structure_new_model_evaluator_contact.hpp"
#include "4C_structure_new_model_evaluator_data.hpp"
#include "4C_structure_new_model_evaluator_structure.hpp"
#include "4C_structure_new_timint_base.hpp"
#include "4C_thermo_ele_action.hpp"
#include "4C_thermo_timint.hpp"
#include "4C_tsi_utils.hpp"
#include "4C_utils_enum.hpp"

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

//! Note: The order of calling the two BaseAlgorithm-constructors is
//! important here! In here control file entries are written. And these entries
//! define the order in which the filters handle the Discretizations, which in
//! turn defines the dof number ordering of the Discretizations.



/*----------------------------------------------------------------------*
 | monolithic                                                dano 11/10 |
 *----------------------------------------------------------------------*/
TSI::Monolithic::Monolithic(MPI_Comm comm, const Teuchos::ParameterList& sdynparams)
    : Algorithm(comm),
      solveradapttol_(((Global::Problem::instance()->tsi_dynamic_params()).sublist("MONOLITHIC"))
              .get<bool>("ADAPTCONV")),
      solveradaptolbetter_(
          ((Global::Problem::instance()->tsi_dynamic_params()).sublist("MONOLITHIC"))
              .get<double>("ADAPTCONV_BETTER")),
      printiter_(true),  // ADD INPUT PARAMETER
      zeros_(nullptr),
      strmethodname_(
          Teuchos::getIntegralValue<Inpar::Solid::DynamicType>(sdynparams, "DYNAMICTYPE")),
      tsidyn_(Global::Problem::instance()->tsi_dynamic_params()),
      tsidynmono_((Global::Problem::instance()->tsi_dynamic_params()).sublist("MONOLITHIC")),
      blockrowdofmap_(nullptr),
      systemmatrix_(nullptr),
      k_st_(nullptr),
      k_ts_(nullptr),
      merge_tsi_blockmatrix_(tsidynmono_.get<bool>("MERGE_TSI_BLOCK_MATRIX")),
      soltech_(Teuchos::getIntegralValue<TSI::NlnSolTech>(tsidynmono_, "NLNSOL")),
      iternorm_(Teuchos::getIntegralValue<TSI::VectorNorm>(tsidynmono_, "ITERNORM")),
      iter_(0),
      sdyn_(sdynparams),
      timernewton_("", true),
      dtsolve_(0.),
      ptcdt_(tsidynmono_.get<double>("PTCDT")),
      dti_(1.0 / ptcdt_),
      ls_strategy_(Teuchos::getIntegralValue<TSI::LineSearch>(tsidynmono_, "TSI_LINE_SEARCH")),
      vel_(nullptr)
{
  fix_time_integration_params();

  // another setup of structural time integration with the correct initial temperature is required,
  // so get the temperature
  if (thermo_field()->tempnp() == nullptr) FOUR_C_THROW("this is nullptr");

  if (matchinggrid_)
    structure_field()->discretization()->set_state(1, "temperature", *thermo_field()->tempnp());
  else
    structure_field()->discretization()->set_state(
        1, "temperature", *volcoupl_->apply_vector_mapping12(*thermo_field()->tempnp()));

  // setup structural time integrator with initial temperature
  structure_->setup();
  structure_field()->discretization()->clear_state(true);

  blockrowdofmap_ = std::make_shared<Core::LinAlg::MultiMapExtractor>();

  // initialise internal variable with new velocities V_{n+1} at t_{n+1}
  vel_ = Core::LinAlg::create_vector(*(structure_field()->dof_row_map(0)), true);

  // --------------------------------- TSI solver: create a linear solver

  // get iterative solver
  if (merge_tsi_blockmatrix_ == false) create_linear_solver();
  // get direct solver, e.g. UMFPACK
  else  // (merge_tsi_blockmatrix_ == true)
  {
    if (Core::Communication::my_mpi_rank(get_comm()) == 0)
      std::cout << "Merged TSI block matrix is used!\n" << std::endl;

    // get solver parameter list of linear TSI solver
    const int linsolvernumber = tsidynmono_.get<int>("LINEAR_SOLVER");
    const Teuchos::ParameterList& tsisolverparams =
        Global::Problem::instance()->solver_params(linsolvernumber);

    Teuchos::ParameterList solverparams;
    solverparams = tsisolverparams;

    solver_ = std::make_shared<Core::LinAlg::Solver>(solverparams, get_comm(),
        Global::Problem::instance()->solver_params_callback(),
        Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
            Global::Problem::instance()->io_params(), "VERBOSITY"));
  }  // end BlockMatrixMerge

  // structure_field: check whether we have locsys BCs, i.e. inclined structural
  //  Dirichlet BC
  {
    std::vector<const Core::Conditions::Condition*> locsysconditions;
    (structure_field()->discretization())->get_condition("Locsys", locsysconditions);

    // if there are inclined structural Dirichlet BC, get the structural LocSysManager
    if (locsysconditions.size())
    {
      locsysman_ = structure_field()->locsys_manager();
    }
    else
      locsysman_ = nullptr;
  }

  if ((tsidynmono_.get<bool>("CALC_NECKING_TSI_VALUES")) and
      (Core::Communication::my_mpi_rank(get_comm()) == 0))
    std::cout
        << "CAUTION: calculation ONLY valid for necking of a cylindrical body!"
        << "\n Due to symmetry only 1/8 of the cylinder is simulated, i.e r/l = 6.413mm/53.334mm."
        << "\n The body is located between x: [0,6.413mm], y: [0,6.413mm], z: "
           "[-13.3335mm,13.3335mm]\n"
        << std::endl;

  // structural and thermal contact
  prepare_contact_strategy();
}  // Monolithic()


/*----------------------------------------------------------------------*
 | read restart information for given time step (public)     dano 11/10 |
 *----------------------------------------------------------------------*/
void TSI::Monolithic::read_restart(int step)
{
  thermo_field()->read_restart(step);
  structure_field()->read_restart(step);

  // structure_field()->read_restart destroyed the old object and created
  // a new one, so we update the pointers
  prepare_contact_strategy();

  // pass the current coupling variables to the respective field
  // second read_restart needed due to the coupling variables
  apply_struct_coupling_state(structure_field()->dispnp(), structure_field()->velnp());
  thermo_field()->read_restart(step);
  thermo_field()->discretization()->clear_state(true);

  apply_thermo_coupling_state(thermo_field()->tempnp());
  structure_field()->read_restart(step);
  structure_field()->discretization()->clear_state(true);

  // structure_field()->read_restart destroyed the old object and created
  // a new one, so we update the pointers
  prepare_contact_strategy();

  set_time_step(thermo_field()->time_old(), step);

  // Material pointers to other field were deleted during read_restart().
  // They need to be reset.
  if (matchinggrid_)
    TSI::Utils::set_material_pointers_matching_grid(
        *structure_field()->discretization(), *thermo_field()->discretization());
  else
  {
    std::shared_ptr<TSI::Utils::TSIMaterialStrategy> strategy =
        std::make_shared<TSI::Utils::TSIMaterialStrategy>();
    volcoupl_->assign_materials(structure_field()->discretization(),
        thermo_field()->discretization(), Global::Problem::instance()->volmortar_params(),
        Global::Problem::instance()->cut_general_params(), strategy);
  }

  Teuchos::ParameterList p;
  //! pointer to the model evaluator data container
  std::shared_ptr<Core::Elements::ParamsMinimal> EvalData =
      std::make_shared<Core::Elements::ParamsMinimal>();
  EvalData->set_action_type(Core::Elements::struct_calc_reset_istep);
  p.set<std::shared_ptr<Core::Elements::ParamsInterface>>("interface", EvalData);
  p.set<std::string>("action", "calc_struct_reset_istep");
  structure_field()->discretization()->evaluate(p);


  return;
}  // read_restart()


/*----------------------------------------------------------------------*
 | prepare time step (public)                                dano 11/10 |
 *----------------------------------------------------------------------*/
void TSI::Monolithic::prepare_time_step()
{
  // counter and print header
  // increment time and step counter
  increment_time_and_step();
  print_header();

  // pass the current coupling variables to the respective fields
  apply_struct_coupling_state(structure_field()->dispnp(), structure_field()->velnp());
  apply_thermo_coupling_state(thermo_field()->tempnp());

  // call the predictor
  structure_field()->prepare_time_step();
  thermo_field()->prepare_time_step();

}  // prepare_time_step()


/*----------------------------------------------------------------------*
 | create linear solver                                   wiesner 07/11 |
 *----------------------------------------------------------------------*/
void TSI::Monolithic::create_linear_solver()
{
  // get the solver number used for linear TSI solver
  const int linsolvernumber = tsidynmono_.get<int>("LINEAR_SOLVER");
  // check if the TSI solver has a valid solver number
  if (linsolvernumber == (-1))
    FOUR_C_THROW(
        "no linear solver defined for monolithic TSI. Please set LINEAR_SOLVER in TSI DYNAMIC to a "
        "valid number!");

  // get solver parameter list of linear TSI solver
  const Teuchos::ParameterList& tsisolverparams =
      Global::Problem::instance()->solver_params(linsolvernumber);

  const auto solvertype =
      Teuchos::getIntegralValue<Core::LinearSolver::SolverType>(tsisolverparams, "SOLVER");

  if (solvertype != Core::LinearSolver::SolverType::belos)
  {
    std::cout << "!!!!!!!!!!!!!!!!!!!!!! ATTENTION !!!!!!!!!!!!!!!!!!!!!" << std::endl;
    std::cout << " Note: the BGS2x2 preconditioner now " << std::endl;
    std::cout << " uses the structural solver and thermal solver blocks" << std::endl;
    std::cout << " for building the internal inverses" << std::endl;
    std::cout << " Remove the old BGS PRECONDITIONER BLOCK entries " << std::endl;
    std::cout << " in the input files!" << std::endl;
    std::cout << "!!!!!!!!!!!!!!!!!!!!!! ATTENTION !!!!!!!!!!!!!!!!!!!!!" << std::endl;
    FOUR_C_THROW("Iterative solver expected");
  }

  // prepare linear solvers and preconditioners
  solver_ = std::make_shared<Core::LinAlg::Solver>(tsisolverparams, get_comm(),
      Global::Problem::instance()->solver_params_callback(),
      Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
          Global::Problem::instance()->io_params(), "VERBOSITY"));

  const auto azprectype =
      Teuchos::getIntegralValue<Core::LinearSolver::PreconditionerType>(tsisolverparams, "AZPREC");

  switch (azprectype)
  {
    case Core::LinearSolver::PreconditionerType::multigrid_muelu:
    case Core::LinearSolver::PreconditionerType::block_teko:
    {
      solver_->put_solver_params_to_sub_params("Inverse1", tsisolverparams,
          Global::Problem::instance()->solver_params_callback(),
          Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
              Global::Problem::instance()->io_params(), "VERBOSITY"),
          get_comm());
      compute_null_space_if_necessary(
          *structure_field()->discretization(), solver_->params().sublist("Inverse1"));

      solver_->put_solver_params_to_sub_params("Inverse2", tsisolverparams,
          Global::Problem::instance()->solver_params_callback(),
          Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
              Global::Problem::instance()->io_params(), "VERBOSITY"),
          get_comm());
      compute_null_space_if_necessary(
          *thermo_field()->discretization(), solver_->params().sublist("Inverse2"));

      break;
    }
    case Core::LinearSolver::PreconditionerType::ilu:
      break;
    default:
      FOUR_C_THROW(
          "Block Gauss-Seidel BGS2x2 preconditioner expected. Alternatively you can define your "
          "own algebraic multigrid block preconditioner (using an xml file).");
      break;
  }

}  // create_linear_solver()


/*----------------------------------------------------------------------*
 | non-linear solve, i.e. (multiple) corrector (public)      dano 11/10 |
 *----------------------------------------------------------------------*/
void TSI::Monolithic::solve()
{
  // choose solution technique according to input file
  switch (soltech_)
  {
    // Newton-Raphson iteration
    case TSI::soltech_newtonfull:
      newton_full();
      break;
    // Pseudo-transient continuation
    case TSI::soltech_ptc:
      ptc();
      break;
    // catch problems
    default:
      FOUR_C_THROW("Solution technique \"{}\" is not implemented", soltech_);
      break;
  }  // end switch (soltechnique_)

  return;
}  // Solve()


/*----------------------------------------------------------------------*
 | time loop of the monolithic system                        dano 11/10 |
 *----------------------------------------------------------------------*/
void TSI::Monolithic::time_loop()
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
    prepare_output();

    // update all single field solvers
    update();

    // write output to screen and files
    output();

  }  // not_finished
}  // TimeLoop()


/*----------------------------------------------------------------------*
 | solution with full Newton-Raphson iteration               dano 10/10 |
 | in tsi_algorithm: NewtonFull()                                       |
 *----------------------------------------------------------------------*/
void TSI::Monolithic::newton_full()
{
  // we do a Newton-Raphson iteration here.
  // the specific time integration has set the following
  // --> On #rhs_ is the positive force residuum
  // --> On #systemmatrix_ is the effective dynamic TSI tangent matrix

  // initialise equilibrium loop
  iter_ = 0;

  // incremental solution vector with length of all TSI dofs
  iterinc_ = Core::LinAlg::create_vector(*dof_row_map(), true);
  iterinc_->put_scalar(0.0);
  // a zero vector of full length
  zeros_ = Core::LinAlg::create_vector(*dof_row_map(), true);
  zeros_->put_scalar(0.0);

  // compute residual forces #rhs_ and tangent #systemmatrix_
  // whose components are globally oriented
  // build linear system stiffness matrix and rhs/force residual for each
  // field, here e.g. for structure field: field want the iteration increment
  // 1.) Update(iterinc_),
  // 2.) evaluate_force_stiff_residual(),
  // 3.) prepare_system_for_newton_solve() --> if (locsysman_!=null) k_ss is rotated
  evaluate(iterinc_);

  // create the linear system
  // \f$J(x_i) \Delta x_i = - R(x_i)\f$
  // create the systemmatrix
  setup_system_matrix();

  // check whether we have a sanely filled tangent matrix
  if (not systemmatrix_->filled()) FOUR_C_THROW("Effective tangent matrix must be filled here");

  // create full monolithic rhs vector
  // make negative residual not necessary: rhs_ is already negative
  // (STR/Thermo)-RHS is put negative in prepare_system_for_newton_solve()
  setup_rhs();

  // do the thermo contact modifications all at once
  if (contact_strategy_lagrange_ != nullptr)
    contact_strategy_lagrange_->evaluate(
        system_matrix(), rhs_, coupST_, structure_field()->dispnp(), thermo_field()->tempnp());
  apply_dbc();

  // initialize with predictor values
  normrhsiter0_ = normrhs_ = calculate_vector_norm(iternorm_, *rhs_);
  normstrrhsiter0_ = normstrrhs_ = last_iter_res_.first =
      calculate_vector_norm(iternormstr_, *structure_field()->rhs());
  normthrrhsiter0_ = normthrrhs_ = last_iter_res_.second =
      calculate_vector_norm(iternormthr_, *thermo_field()->rhs());
  ls_step_length_ = 1.;
  normdisi_ = normtempi_ = norminc_ = 0.;
  normstrrhsiter0_ = normstrrhs_;
  norminciter0_ = normdisiiter0_ = normtempiiter0_ = -1.;

  // print stuff
  print_newton_iter();

  //------------------------------------------------------ iteration loop

  // equilibrium iteration loop (loop over k)
  while (((not converged()) and (iter_ <= itermax_)) or (iter_ <= itermin_))
  {
    // increment equilibrium loop index
    ++iter_;
    ls_step_length_ = 1.;

    // reset timer
    timernewton_.reset();

    // *********** time measurement ***********
    double dtcpu = timernewton_.wallTime();
    // *********** time measurement ***********
    // (Newton-ready) residual with blanked Dirichlet DOFs (see adapter_timint!)
    // is done in prepare_system_for_newton_solve() within evaluate(iterinc_)
    linear_solve();
    // *********** time measurement ***********
    dtsolve_ = timernewton_.wallTime() - dtcpu;
    // *********** time measurement ***********

    // recover LM in the case of contact
    if (contact_strategy_lagrange_ != nullptr) recover_struct_therm_lm();

    // reset solver tolerance
    solver_->reset_tolerance();

    // compute residual forces #rhs_ and tangent #systemmatrix_
    // whose components are globally oriented
    // build linear system stiffness matrix and rhs/force residual for each
    // field, here e.g. for structure field: field want the iteration increment
    // 1.) Update(iterinc_),
    // 2.) evaluate_force_stiff_residual(),
    // 3.) prepare_system_for_newton_solve() --> if (locsysman_!=null) k_ss is rotated
    evaluate(iterinc_);

    // create the linear system
    // \f$J(x_i) \Delta x_i = - R(x_i)\f$
    // create the systemmatrix
    setup_system_matrix();

    // check whether we have a sanely filled tangent matrix
    if (not systemmatrix_->filled()) FOUR_C_THROW("Effective tangent matrix must be filled here");

    // create full monolithic rhs vector
    // make negative residual not necessary: rhs_ is already negative
    // (STR/Thermo)-RHS is put negative in prepare_system_for_newton_solve()
    setup_rhs();

    // do the thermo contact modifications all at once
    if (contact_strategy_lagrange_ != nullptr)
    {
      // *********** time measurement ***********
      double dtcpu = timernewton_.wallTime();
      // *********** time measurement ***********

      contact_strategy_lagrange_->evaluate(
          system_matrix(), rhs_, coupST_, structure_field()->dispnp(), thermo_field()->tempnp());

      // *********** time measurement ***********
      dtcmt_ = timernewton_.wallTime() - dtcpu;
      // *********** time measurement ***********
    }
    apply_dbc();

    // do line search
    switch (ls_strategy_)
    {
      case TSI::LS_none:
        break;
      case TSI::LS_structure:
      case TSI::LS_thermo:
      case TSI::LS_and:
      case TSI::LS_or:
      {
        normstrrhs_ = calculate_vector_norm(iternormstr_, *structure_field()->rhs());
        normthrrhs_ = calculate_vector_norm(iternormthr_, *thermo_field()->rhs());
        iterinc_->scale(-1.);

        while (ls_step_length_ > 1.e-8 && !l_sadmissible())
        {
          iterinc_->scale(.5);
          ls_step_length_ *= .5;
          evaluate(iterinc_);
          normstrrhs_ = calculate_vector_norm(iternormstr_, *structure_field()->rhs());
          normthrrhs_ = calculate_vector_norm(iternormthr_, *thermo_field()->rhs());
        }

        last_iter_res_.first = calculate_vector_norm(iternormstr_, *structure_field()->rhs());
        last_iter_res_.second = calculate_vector_norm(iternormthr_, *thermo_field()->rhs());

        if (ls_step_length_ < 1.)
        {
          setup_system_matrix();
          setup_rhs();
          apply_dbc();
        }
      }
      break;
      default:
        FOUR_C_THROW("unknown line search strategy");
    }  // end line search

    // --------------------------------------------- build residual norms
    // include all stuff here related with convergence test
    normrhs_ = calculate_vector_norm(iternorm_, *rhs_);
    // vector of displacement and temperature residual
    std::shared_ptr<Core::LinAlg::Vector<double>> strrhs;
    std::shared_ptr<Core::LinAlg::Vector<double>> thrrhs;
    // extract field vectors
    extract_field_vectors(rhs_, strrhs, thrrhs);
    normstrrhs_ = calculate_vector_norm(iternormstr_, *strrhs);
    normthrrhs_ = calculate_vector_norm(iternormthr_, *thrrhs);

    // --------------------------------- build residual incremental norms
    // vector of displacement and temperature increments
    std::shared_ptr<Core::LinAlg::Vector<double>> sx;
    std::shared_ptr<Core::LinAlg::Vector<double>> tx;
    // extract field vectors
    extract_field_vectors(iterinc_, sx, tx);
    norminc_ = calculate_vector_norm(iternorm_, *iterinc_);
    normdisi_ = calculate_vector_norm(iternormstr_, *sx);
    normtempi_ = calculate_vector_norm(iternormthr_, *tx);

    // in case of 'Mix'-convergence criterion: save the norm of the 1st
    // iteration in (norm . iter0_)
    if (iter_ == 1)
    {
      // save initial incremental norms
      norminciter0_ = norminc_;
      normdisiiter0_ = normdisi_;
      normtempiiter0_ = normtempi_;

      // set the minimum of iter0_ and tolrhs_, because we want to prevent the
      // case of a zero characteristic initial norm
      if (normrhsiter0_ == 0.0) normrhsiter0_ = tolrhs_;
      if (normstrrhsiter0_ == 0.0) normstrrhsiter0_ = tolstrrhs_;
      if (normthrrhsiter0_ == 0.0) normthrrhsiter0_ = tolthrrhs_;
      if (norminciter0_ == 0.0) norminciter0_ = tolinc_;
      if (normdisiiter0_ == 0.0) normdisiiter0_ = toldisi_;
      if (normtempiiter0_ == 0.0) normtempiiter0_ = toltempi_;
    }

    // print stuff
    print_newton_iter();

  }  // end equilibrium loop

  // ----------------------------------------------------- iteration loop

  // correct iteration counter
  iter_ -= 1;

  // test whether max iterations was hit
  if ((converged()) and (Core::Communication::my_mpi_rank(get_comm()) == 0))
  {
    print_newton_conv();
  }
  else if (iter_ >= itermax_)
  {
    if (Teuchos::getIntegralValue<Inpar::Solid::DivContAct>(sdyn_, "DIVERCONT") ==
        Inpar::Solid::divcont_continue)
      ;  // do nothing
    else
      FOUR_C_THROW("Newton unconverged in {} iterations", iter_);
  }
  // for validation with literature calculate nodal TSI values
  if ((tsidynmono_.get<bool>("CALC_NECKING_TSI_VALUES")) == true) calculate_necking_tsi_results();

}  // NewtonFull()


/*----------------------------------------------------------------------*
 | solution with pseudo-transient continuation               dano 06/14 |
 *----------------------------------------------------------------------*/
void TSI::Monolithic::ptc()
{
  // do a PTC iteration here
  // implementation is based on the work of Gee, Kelley, Lehouq (2009):
  // "Pseudo-transient continuation for nonlinear transient elasticity"

  // the specific time integration has set the following
  // --> On #rhs_ is the positive force residuum
  // --> On #systemmatrix_ is the effective dynamic TSI tangent matrix

  // initialise equilibrium loop
  iter_ = 0;

  // incremental solution vector with length of all TSI dofs
  iterinc_ = Core::LinAlg::create_vector(*dof_row_map(), true);
  iterinc_->put_scalar(0.0);
  // a zero vector of full length
  zeros_ = Core::LinAlg::create_vector(*dof_row_map(), true);
  zeros_->put_scalar(0.0);

  // compute residual forces #rhs_ and tangent #systemmatrix_
  // whose components are globally oriented
  // build linear system stiffness matrix and rhs/force residual for each
  // field, here e.g. for structure field: field want the iteration increment
  // 1.) Update(iterinc_),
  // 2.) evaluate_force_stiff_residual(),
  // 3.) prepare_system_for_newton_solve() --> if (locsysman_!=null) k_ss is rotated
  evaluate(iterinc_);

  // create the linear system
  // \f$J(x_i) \Delta x_i = - R(x_i)\f$
  // create the systemmatrix
  setup_system_matrix();

  // check whether we have a sanely filled tangent matrix
  if (not systemmatrix_->filled()) FOUR_C_THROW("Effective tangent matrix must be filled here");

  // create full monolithic rhs vector
  // make negative residual not necessary: rhs_ is already negative
  // (STR/Thermo)-RHS is put negative in prepare_system_for_newton_solve()
  setup_rhs();

  apply_dbc();

  // initialize with predictor values
  normrhsiter0_ = normrhs_ = calculate_vector_norm(iternorm_, *rhs_);
  normrhsiter0_ = normstrrhs_ = last_iter_res_.first =
      calculate_vector_norm(iternormstr_, *structure_field()->rhs());
  normthrrhsiter0_ = normthrrhs_ = last_iter_res_.second =
      calculate_vector_norm(iternormthr_, *thermo_field()->rhs());
  ls_step_length_ = 1.;
  normdisi_ = normtempi_ = norminc_ = 0.;
  normstrrhsiter0_ = normstrrhs_;
  normrhsiter0_ = normstrrhsiter0_ = normthrrhsiter0_ = norminciter0_ = normdisiiter0_ =
      normtempiiter0_ = -1.;

  // ----------------------------------------------- special stuff of PTC

  // compute the PTC parameters
  double ptcdt = ptcdt_;
  // norm of residual of old iteration step
  double nc = 0.0;
  // as recommended by Michael, apply PTC to the whole TSI system, i.e. use TSI
  // residual here
  // here the new convergence test stuff has to be included

  normrhs_ = calculate_vector_norm(iternorm_, *rhs_);
  rhs_->norm_inf(&nc);
  // define the pseudo time step delta^{-1}
  double dti = 1 / ptcdt;

  // print stuff
  print_newton_iter();

  //------------------------------------------------------ iteration loop

  // equilibrium iteration loop (loop over k)
  while (((not converged()) and (iter_ <= itermax_)) or (iter_ <= itermin_))
  {
    // increment equilibrium loop index
    ++iter_;

    // reset timer
    timernewton_.reset();

    // ---------- modify diagonal blocks of systemmatrix according to PTC

    // modify structural diagonal block k_ss
    {
      std::shared_ptr<Core::LinAlg::Vector<double>> tmp_SS =
          Core::LinAlg::create_vector(structure_field()->system_matrix()->row_map(), false);
      tmp_SS->put_scalar(dti);
      std::shared_ptr<Core::LinAlg::Vector<double>> diag_SS =
          Core::LinAlg::create_vector(structure_field()->system_matrix()->row_map(), false);
      structure_field()->system_matrix()->extract_diagonal_copy(*diag_SS);

      diag_SS->update(1.0, *tmp_SS, 1.0);

      structure_field()->system_matrix()->replace_diagonal_values(*diag_SS);
    }
    // modify thermal diagonal block k_tt
    {
      std::shared_ptr<Core::LinAlg::Vector<double>> tmp_tt =
          Core::LinAlg::create_vector(thermo_field()->system_matrix()->row_map(), false);
      tmp_tt->put_scalar(dti);
      std::shared_ptr<Core::LinAlg::Vector<double>> diag_tt =
          Core::LinAlg::create_vector(thermo_field()->system_matrix()->row_map(), false);
      thermo_field()->system_matrix()->extract_diagonal_copy(*diag_tt);
      diag_tt->update(1.0, *tmp_tt, 1.0);
      thermo_field()->system_matrix()->replace_diagonal_values(*diag_tt);
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

    // reset solver tolerance
    solver_->reset_tolerance();

    // compute residual forces #rhs_ and tangent #systemmatrix_
    // whose components are globally oriented
    // build linear TSI tangent matrix and rhs/force residual for each field,
    // here e.g. for structure field: STR field wants the iteration increment
    // 1.) Update(iterinc_),
    // 2.) evaluate_force_stiff_residual(),
    // 3.) prepare_system_for_newton_solve() --> if (locsysman_ != null) k_ss is rotated
    evaluate(iterinc_);

    // create the linear system including PTC-modified systemmatrices k_ss and k_tt
    // \f$J(x_i) \Delta x_i = - R(x_i)\f$
    // create the systemmatrix
    setup_system_matrix();

    // check whether we have a sanely filled tangent matrix
    if (not systemmatrix_->filled()) FOUR_C_THROW("Effective tangent matrix must be filled here");

    // create full monolithic rhs vector
    // make negative residual not necessary: rhs_ is already negative
    // (STR/Thermo)-RHS is put negative in prepare_system_for_newton_solve()
    setup_rhs();

    // apply Dirichlet boundary conditions on System matrix and RHS
    apply_dbc();

    // --------------------------------------------- build residual norms
    // include all stuff here related with convergence test
    normrhs_ = calculate_vector_norm(iternorm_, *rhs_);
    // vector of displacement and temperature residual
    std::shared_ptr<Core::LinAlg::Vector<double>> strrhs;
    std::shared_ptr<Core::LinAlg::Vector<double>> thrrhs;
    // extract field vectors
    extract_field_vectors(rhs_, strrhs, thrrhs);
    normstrrhs_ = calculate_vector_norm(iternormstr_, *strrhs);
    normthrrhs_ = calculate_vector_norm(iternormthr_, *thrrhs);

    // --------------------------------- build residual incremental norms
    // vector of displacement and temperature increments
    std::shared_ptr<Core::LinAlg::Vector<double>> sx;
    std::shared_ptr<Core::LinAlg::Vector<double>> tx;
    // extract field vectors
    extract_field_vectors(iterinc_, sx, tx);
    norminc_ = calculate_vector_norm(iternorm_, *iterinc_);
    normdisi_ = calculate_vector_norm(iternormstr_, *sx);
    normtempi_ = calculate_vector_norm(iternormthr_, *tx);

    // in case of 'Mix'-convergence criterion: save the norm of the 1st
    // iteration in norm*iter0_
    if (iter_ == 1)
    {
      // set residuals
      normrhsiter0_ = normrhs_;
      normstrrhsiter0_ = normstrrhs_;
      normthrrhsiter0_ = normthrrhs_;
      // set increments
      norminciter0_ = norminc_;
      normdisiiter0_ = normdisi_;
      normtempiiter0_ = normtempi_;

      // we set the minimum of iter0_ and tolrhs_, because
      // we want to prevent the case of a zero characteristic initial norm
      if (normrhsiter0_ == -1.) normrhsiter0_ = tolrhs_;
      if (normstrrhsiter0_ == -1.) normstrrhsiter0_ = tolstrrhs_;
      if (normthrrhsiter0_ == -1.) normthrrhsiter0_ = tolthrrhs_;
      if (norminciter0_ == -1.) norminciter0_ = tolinc_;
      if (normdisiiter0_ == -1.) normdisiiter0_ = toldisi_;
      if (normtempiiter0_ == -1.) normtempiiter0_ = toltempi_;
    }

    // print stuff
    print_newton_iter();

    // save old pseudo-time step in dti_
    dti_ = dti;

    // update ptc
    {
      double np = 0.0;
      rhs_->norm_inf(&np);
      dti *= (np / nc);
      dti = std::max(dti, 0.0);
      nc = np;
    }

  }  // end equilibrium loop

  // ----------------------------------------------------- iteration loop

  // correct iteration counter
  iter_ -= 1;

  // test whether max iterations was hit
  if ((converged()) and (Core::Communication::my_mpi_rank(get_comm()) == 0))
  {
    print_newton_conv();
  }
  else if (iter_ >= itermax_)
    FOUR_C_THROW("PTC unconverged in {} iterations", iter_);

}  // PTC()


/*----------------------------------------------------------------------*
 | evaluate the single fields                                dano 11/10 |
 *----------------------------------------------------------------------*/
void TSI::Monolithic::evaluate(std::shared_ptr<Core::LinAlg::Vector<double>> stepinc)
{
  TEUCHOS_FUNC_TIME_MONITOR("TSI::Monolithic::Evaluate");

  // displacement and temperature incremental vector
  std::shared_ptr<Core::LinAlg::Vector<double>> sx;
  std::shared_ptr<Core::LinAlg::Vector<double>> tx;

  // if an increment vector exists
  if (stepinc != nullptr)
  {
    // extract displacement sx and temperature tx incremental vector of global
    // unknown incremental vector x
    extract_field_vectors(stepinc, sx, tx);
  }

  // Newton update of the thermo field
  // update temperature before passed to the structural field
  //   update_iter_incrementally(tx),
  thermo_field()->update_newton(tx);

  // call all elements and assemble rhs and matrices
  /// structural field

  // structure Evaluate (builds tangent, residual and applies DBC)
  Teuchos::Time timerstructure("", true);

  // apply current temperature to structure
  apply_thermo_coupling_state(thermo_field()->tempnp(), tx);

  // Monolithic TSI accesses the linearised structure problem:
  //   UpdaterIterIncrementally(sx),
  //   evaluate_force_stiff_residual()
  //   prepare_system_for_newton_solve()
  //     blank residual DOFs that are on Dirichlet BC
  //     in case of local coordinate systems rotate the residual forth and back
  //     Be AWARE: apply_dirichlet_to_system has to be called with rotated stiff_!
  if (iter_ == 0)
    structure_field()->evaluate();
  else
    structure_field()->evaluate(sx);
  structure_field()->discretization()->clear_state(true);

  /// thermal field

  // thermo Evaluate
  // (builds tangent, residual and applies DBC and recent coupling values)
  Teuchos::Time timerthermo("", true);

  // apply current displacements and velocities to the thermo field
  if (strmethodname_ == Inpar::Solid::DynamicType::Statics)
  {
    // calculate velocity V_n+1^k = (D_n+1^k-D_n)/Dt()
    vel_ = calc_velocity(*structure_field()->dispnp());
  }
  // else: use velnp
  else
    vel_ = structure_field()->velnp();

  // pass the structural values to the thermo field
  apply_struct_coupling_state(structure_field()->dispnp(), vel_);

  // monolithic TSI accesses the linearised thermo problem
  //   evaluate_rhs_tang_residual() and
  //   prepare_system_for_newton_solve()
  thermo_field()->evaluate();
  thermo_field()->discretization()->clear_state(true);
}  // evaluate()


/*----------------------------------------------------------------------*
 | extract field vectors for calling evaluate() of the       dano 11/10 |
 | single fields                                                        |
 *----------------------------------------------------------------------*/
void TSI::Monolithic::extract_field_vectors(std::shared_ptr<Core::LinAlg::Vector<double>> x,
    std::shared_ptr<Core::LinAlg::Vector<double>>& sx,
    std::shared_ptr<Core::LinAlg::Vector<double>>& tx)
{
  TEUCHOS_FUNC_TIME_MONITOR("TSI::Monolithic::extract_field_vectors");

  // process structure unknowns of the first field
  sx = extractor()->extract_vector(*x, 0);

  // process thermo unknowns of the second field
  tx = extractor()->extract_vector(*x, 1);
}  // extract_field_vectors()


/*----------------------------------------------------------------------*
 | full monolithic dof row map                               dano 05/12 |
 *----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Map> TSI::Monolithic::dof_row_map() const
{
  return extractor()->full_map();
}  // dof_row_map()


/*----------------------------------------------------------------------*
 | setup system (called in tsi_dyn)                          dano 11/10 |
 *----------------------------------------------------------------------*/
void TSI::Monolithic::setup_system()
{
  // set parameters that remain the same in the whole calculation
  set_default_parameters();

  set_dof_row_maps();

  /*----------------------------------------------------------------------*/
  // initialise TSI-systemmatrix_
  systemmatrix_ =
      std::make_shared<Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(
          *extractor(), *extractor(), 81, false, true);

  // create empty matrix
  k_st_ = std::make_shared<Core::LinAlg::SparseMatrix>(
      *(structure_field()->discretization()->dof_row_map(0)), 81, false, true);

  // create empty matrix
  k_ts_ = std::make_shared<Core::LinAlg::SparseMatrix>(
      *(thermo_field()->discretization()->dof_row_map(0)), 81, false, true);

}  // SetupSystem()


/*----------------------------------------------------------------------*
 | put the single maps to one full TSI map together          dano 11/10 |
 *----------------------------------------------------------------------*/
void TSI::Monolithic::set_dof_row_maps()
{
  // create combined map
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> vecSpaces;

  // use its own dof_row_map, that is the 0th map of the discretization
  vecSpaces.push_back(structure_field()->dof_row_map(0));
  vecSpaces.push_back(thermo_field()->dof_row_map(0));

  if (vecSpaces[0]->num_global_elements() == 0) FOUR_C_THROW("No structure equation. Panic.");
  if (vecSpaces[1]->num_global_elements() == 0) FOUR_C_THROW("No temperature equation. Panic.");

  std::shared_ptr<Core::LinAlg::Map> fullmap =
      Core::LinAlg::MultiMapExtractor::merge_maps(vecSpaces);

  // full TSI-blockmap
  extractor()->setup(*fullmap, vecSpaces);
}  // set_dof_row_maps()


/*----------------------------------------------------------------------*
 | setup system matrix of TSI                                dano 11/10 |
 *----------------------------------------------------------------------*/
void TSI::Monolithic::setup_system_matrix()
{
  TEUCHOS_FUNC_TIME_MONITOR("TSI::Monolithic::setup_system_matrix");

  /*----------------------------------------------------------------------*/
  // pure structural part k_ss (3nx3n)

  // build pure structural block k_ss
  // build block matrix
  // The maps of the block matrix have to match the maps of the blocks we
  // insert here. Extract Jacobian matrices and put them into composite system
  // matrix W
  std::shared_ptr<Core::LinAlg::SparseMatrix> k_ss = structure_field()->system_matrix();

  // assign structure part to the TSI matrix
  systemmatrix_->assign(0, 0, Core::LinAlg::DataAccess::Share, *k_ss);

  /*----------------------------------------------------------------------*/
  // structural block k_st (3nxn)
  // build mechanical-thermal block

  k_st_->reset();
  // call the element and calculate the matrix block
  apply_str_coupl_matrix(k_st_);

  k_st_->complete(*(structure_field()->discretization()->dof_row_map(1)),
      *(structure_field()->discretization()->dof_row_map(0)));

  if (!matchinggrid_) k_st_ = volcoupl_->apply_matrix_mapping12(*k_st_);

  k_st_->un_complete();

  // assign thermo part to the TSI matrix
  systemmatrix_->assign(0, 1, Core::LinAlg::DataAccess::Share, *(k_st_));

  /*----------------------------------------------------------------------*/
  // pure thermo part k_tt (nxn)

  // build pure thermal block k_tt
  // build block matrix
  // The maps of the block matrix have to match the maps of the blocks we
  // insert here. Extract Jacobian matrices and put them into composite system
  // matrix systemmatrix_
  std::shared_ptr<Core::LinAlg::SparseMatrix> k_tt = thermo_field()->system_matrix();

  // assign thermo part to the TSI matrix
  systemmatrix_->assign(1, 1, Core::LinAlg::DataAccess::Share, *(k_tt));

  /*----------------------------------------------------------------------*/
  // thermo part k_ts (nx3n)
  // build thermal-mechanical block

  k_ts_->reset();

  // call the element and calculate the matrix block
  apply_thermo_coupl_matrix(k_ts_);
  apply_thermo_coupl_matrix_conv_bc(k_ts_);

  k_ts_->complete(*(thermo_field()->discretization()->dof_row_map(1)),
      *(thermo_field()->discretization()->dof_row_map(0)));

  if (!matchinggrid_) k_ts_ = volcoupl_->apply_matrix_mapping21(*k_ts_);

  systemmatrix_->assign(1, 0, Core::LinAlg::DataAccess::Share, *k_ts_);

  /*----------------------------------------------------------------------*/
  // done. make sure all blocks are filled.
  systemmatrix_->complete();

  // apply mortar coupling
  if (mortar_coupling_ != nullptr) mortar_coupling_->condense_matrix(systemmatrix_);

}  // setup_system_matrix()


/*----------------------------------------------------------------------*
 | setup RHS (like fsimon)                                   dano 11/10 |
 *----------------------------------------------------------------------*/
void TSI::Monolithic::setup_rhs()
{
  TEUCHOS_FUNC_TIME_MONITOR("TSI::Monolithic::setup_rhs");

  // create full monolithic rhs vector
  rhs_ = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map(), true);

  // get the structural rhs
  std::shared_ptr<Core::LinAlg::Vector<double>> str_rhs =
      std::make_shared<Core::LinAlg::Vector<double>>(*structure_field()->rhs());
  if (Teuchos::getIntegralValue<Inpar::Solid::IntegrationStrategy>(
          Global::Problem::instance()->structural_dynamic_params(), "INT_STRATEGY") ==
      Inpar::Solid::int_standard)
    str_rhs->scale(-1.);

  // insert vectors to tsi rhs
  extractor()->insert_vector(*str_rhs, 0, *rhs_);
  extractor()->insert_vector(*thermo_field()->rhs(), 1, *rhs_);

  // apply mortar coupling
  if (mortar_coupling_ != nullptr) mortar_coupling_->condense_rhs(*rhs_);

}  // setup_rhs()


/*----------------------------------------------------------------------*
 | solve linear TSI system                                   dano 04/11 |
 *----------------------------------------------------------------------*/
void TSI::Monolithic::linear_solve()
{
  // Solve for inc_ = [disi_,tempi_]
  // Solve K_Teffdyn . IncX = -R  ===>  IncX_{n+1} with X=[d,T]
  // \f$x_{i+1} = x_i + \Delta x_i\f$
  Core::LinAlg::SolverParams solver_params;
  if (solveradapttol_ and (iter_ > 1))
  {
    solver_params.nonlin_tolerance = tolrhs_;
    solver_params.nonlin_residual = normrhs_;
    solver_params.lin_tol_better = solveradaptolbetter_;
  }

  // Dirichlet boundary conditions are already applied to TSI system, i.e. TSI
  // system is prepared for solve, i.e. TSI systemmatrix, TSI rhs, TSI inc
  // --> in prepare_system_for_newton_solve(): done for rhs and diagonal blocks
  // --> in setup_system_matrix() done for off-diagonal blocks k_st, k_ts

  // apply Dirichlet BCs to system of equations
  iterinc_->put_scalar(0.0);  // Useful? depends on solver and more

  // default: use block matrix
  if (merge_tsi_blockmatrix_ == false)
  {
    // Infnormscaling: scale system before solving
    scale_system(*systemmatrix_, *rhs_);

    // solve the problem, work is done here!
    solver_params.refactor = true;
    solver_params.reset = iter_ == 1;
    solver_->solve(systemmatrix_, iterinc_, rhs_, solver_params);

    // Infnormscaling: unscale system after solving
    unscale_solution(*systemmatrix_, *iterinc_, *rhs_);

  }  // use block matrix

  else  // (merge_tsi_blockmatrix_ == true)
  {
    // merge blockmatrix to SparseMatrix and solve
    std::shared_ptr<Core::LinAlg::SparseMatrix> sparse = systemmatrix_->merge();

    // standard solver call
    Core::LinAlg::SolverParams solver_params;
    solver_params.refactor = true;
    solver_params.reset = iter_ == 1;
    solver_->solve(sparse, iterinc_, rhs_, solver_params);
  }  // MergeBlockMatrix

  // apply mortar coupling
  if (mortar_coupling_ != nullptr) mortar_coupling_->recover_incr(*iterinc_);

}  // linear_solve()


/*----------------------------------------------------------------------*
 | initial guess of the displacements/temperatures           dano 11/10 |
 *----------------------------------------------------------------------*/
void TSI::Monolithic::initial_guess(std::shared_ptr<Core::LinAlg::Vector<double>> ig)
{
  TEUCHOS_FUNC_TIME_MONITOR("TSI::Monolithic::initial_guess");

  // InitialGuess() is called of the single fields and results are put in TSI
  // increment vector ig
  setup_vector(*ig,
      // returns residual displacements \f$\Delta D_{n+1}^{<k>}\f$ - disi_
      structure_field()->initial_guess(),
      // returns residual temperatures or iterative thermal increment - tempi_
      thermo_field()->initial_guess());

}  // initial_guess()


/*----------------------------------------------------------------------*
 | setup vector of the structure and thermo field            dano 11/10 |
 *----------------------------------------------------------------------*/
void TSI::Monolithic::setup_vector(Core::LinAlg::Vector<double>& f,
    std::shared_ptr<const Core::LinAlg::Vector<double>> sv,
    std::shared_ptr<const Core::LinAlg::Vector<double>> tv)
{
  // extract dofs of the two fields
  // and put the structural/thermal field vector into the global vector f
  // noticing the block number
  extractor()->insert_vector(*sv, 0, f);
  extractor()->insert_vector(*tv, 1, f);

}  // setup_vector()


/*----------------------------------------------------------------------*
 | check convergence of Newton iteration (public)            dano 11/10 |
 *----------------------------------------------------------------------*/
bool TSI::Monolithic::converged()
{
  // check for single norms
  bool convrhs = false;
  bool convinc = false;
  bool convstrrhs = false;
  bool convdisp = false;
  bool convthrrhs = false;
  bool convtemp = false;

  // ----------------------------------------------------------- TSI test
  // residual TSI forces
  switch (normtyperhs_)
  {
    case TSI::convnorm_abs:
      convrhs = normrhs_ < tolrhs_;
      break;
    case TSI::convnorm_rel:
      convrhs = normrhs_ < std::max(tolrhs_ * normrhsiter0_, 1.0e-15);
      break;
    case TSI::convnorm_mix:
      convrhs = ((normrhs_ < tolrhs_) and (normrhs_ < std::max(normrhsiter0_ * tolrhs_, 1.0e-15)));
      break;
    default:
      FOUR_C_THROW("Cannot check for convergence of residual forces!");
      break;
  }

  // residual TSI increments
  switch (normtypeinc_)
  {
    case TSI::convnorm_abs:
      convinc = norminc_ < tolinc_;
      break;
    case TSI::convnorm_rel:
      convinc = norminc_ < std::max(norminciter0_ * tolinc_, 1e-15);
      break;
    case TSI::convnorm_mix:
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

  // ------------------------------------------------------------- thermo
  // thermal residual forces
  switch (normtypethrrhs_)
  {
    case Thermo::convnorm_abs:
      convthrrhs = normthrrhs_ < tolthrrhs_;
      break;
    case Thermo::convnorm_rel:
      convthrrhs = normthrrhs_ < normthrrhsiter0_ * tolthrrhs_;
      break;
    case Thermo::convnorm_mix:
      convthrrhs = ((normthrrhs_ < tolthrrhs_) or (normthrrhs_ < normthrrhsiter0_ * tolthrrhs_));
      break;
    default:
      FOUR_C_THROW("Cannot check for convergence of residual forces!");
      break;
  }  // switch (normtypethrrhs_)

  // residual temperatures
  switch (normtypetempi_)
  {
    case Thermo::convnorm_abs:
      convtemp = normtempi_ < toltempi_;
      break;
    case Thermo::convnorm_rel:
      convtemp = normtempi_ < normtempiiter0_ * toltempi_;
      break;
    case Thermo::convnorm_mix:
      convtemp = ((normtempi_ < toltempi_) or (normtempi_ < normtempiiter0_ * toltempi_));
      break;
    default:
      FOUR_C_THROW("Cannot check for convergence of temperatures!");
      break;
  }  // switch (normtypetempi_)

  // -------------------------------------------------------- convergence
  // combine increment-like and force-like residuals, combine TSI and single
  // field values
  bool conv = false;
  if (combincrhs_ == TSI::bop_and)
    conv = convinc and convrhs;
  else if (combincrhs_ == TSI::bop_or)
    conv = convinc or convrhs;
  else if (combincrhs_ == TSI::bop_coupl_and_single)
    conv = convinc and convrhs and convdisp and convstrrhs and convtemp and convthrrhs;
  else if (combincrhs_ == TSI::bop_coupl_or_single)
    conv = (convinc and convrhs) or (convdisp and convstrrhs and convtemp and convthrrhs);
  else if (combincrhs_ == TSI::bop_and_single)
    conv = convdisp and convstrrhs and convtemp and convthrrhs;
  else if (combincrhs_ == TSI::bop_or_single)
    conv = (convdisp or convstrrhs or convtemp or convthrrhs);
  else
    FOUR_C_THROW("Something went terribly wrong with binary operator!");

  // convergence of active contact set
  if (contact_strategy_lagrange_ != nullptr)
  {
    conv = conv && (contact_strategy_lagrange_->mech_contact_res_ <
                       contact_strategy_lagrange_->params().get<double>("TOLCONTCONSTR"));
    conv = conv && (contact_strategy_lagrange_->mech_contact_incr_ <
                       contact_strategy_lagrange_->params().get<double>("TOLLAGR"));
    conv = conv && contact_strategy_lagrange_->active_set_converged();
  }

  // return things
  return conv;

}  // Converged()


/*----------------------------------------------------------------------*
 | print Newton-Raphson iteration to screen and error file   dano 11/10 |
 | originally by lw 12/07, tk 01/08                                     |
 *----------------------------------------------------------------------*/
void TSI::Monolithic::print_newton_iter()
{
  // print to standard out
  // replace myrank_ here general by Core::Communication::my_mpi_rank(Comm())
  if ((Core::Communication::my_mpi_rank(get_comm()) == 0) and print_screen_every() and
      (step() % print_screen_every() == 0) and printiter_)
  {
    if (iter_ == 0) print_newton_iter_header(stdout);
    print_newton_iter_text(stdout);
  }

}  // print_newton_iter()


/*----------------------------------------------------------------------*
 | print Newton-Raphson iteration to screen and error file   dano 11/10 |
 | originally by lw 12/07, tk 01/08                                     |
 *----------------------------------------------------------------------*/
void TSI::Monolithic::print_newton_iter_header(FILE* ofile)
{
  // open outstringstream
  std::ostringstream oss;

  // enter converged state etc
  oss << std::setw(5) << "# iter";

  // line search
  if (ls_strategy_) oss << std::setw(11) << " ls_step";

  // ---------------------------------------------------------------- TSI
  // different style due relative or absolute error checking
  // displacement
  switch (normtyperhs_)
  {
    case TSI::convnorm_abs:
      oss << std::setw(15) << "abs-res-norm";
      break;
    case TSI::convnorm_rel:
      oss << std::setw(15) << "rel-res-norm";
      break;
    case TSI::convnorm_mix:
      oss << std::setw(15) << "mix-res-norm";
      break;
    default:
      FOUR_C_THROW("You should not turn up here.");
      break;
  }

  switch (normtypeinc_)
  {
    case TSI::convnorm_abs:
      oss << std::setw(15) << "abs-inc-norm";
      break;
    case TSI::convnorm_rel:
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

  // ------------------------------------------------------------- thermo
  switch (normtypethrrhs_)
  {
    case Thermo::convnorm_rel:
      oss << std::setw(18) << "rel-thermo-res-norm";
      break;
    case Thermo::convnorm_abs:
      oss << std::setw(18) << "abs-thermo-res-norm";
      break;
    case Thermo::convnorm_mix:
      oss << std::setw(18) << "mix-thermo-res-norm";
      break;
    default:
      FOUR_C_THROW("You should not turn up here.");
      break;
  }  // switch (normtypethrrhs_)

  switch (normtypetempi_)
  {
    case Thermo::convnorm_rel:
      oss << std::setw(16) << "rel-temp-norm";
      break;
    case Thermo::convnorm_abs:
      oss << std::setw(16) << "abs-temp-norm";
      break;
    case Thermo::convnorm_mix:
      oss << std::setw(16) << "mix-temp-norm";
      break;
    default:
      FOUR_C_THROW("You should not turn up here.");
      break;
  }  // switch (normtypetempi_)

  if (contact_strategy_lagrange_ != nullptr)
  {
    oss << std::setw(16) << "sLMres-norm";
    oss << std::setw(16) << "sLMinc-norm";
    oss << std::setw(16) << "tLMinc-norm";
  }


  if (soltech_ == TSI::soltech_ptc)
  {
    oss << std::setw(16) << "        PTC-dti";
  }

  // add solution time
  oss << std::setw(12) << "ts";
  oss << std::setw(12) << "wct";

  // add contact set information
  if (contact_strategy_lagrange_ != nullptr)
  {
    oss << std::setw(12) << "tc";
    oss << std::setw(11) << "#active";
    if (contact_strategy_lagrange_->is_friction()) oss << std::setw(10) << "#slip";
  }

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
 | print Newton-Raphson iteration to screen                  dano 11/10 |
 | originally by lw 12/07, tk 01/08                                     |
 *----------------------------------------------------------------------*/
void TSI::Monolithic::print_newton_iter_text(FILE* ofile)
{
  // open outstringstream
  std::ostringstream oss;

  // enter converged state etc
  oss << std::setw(6) << iter_;

  // line search step
  if (ls_strategy_)
    oss << std::setw(11) << std::setprecision(3) << std::scientific << ls_step_length_;

  // different style due relative or absolute error checking

  // ----------------------------------------------- test coupled problem
  switch (normtyperhs_)
  {
    case TSI::convnorm_abs:
      oss << std::setw(15) << std::setprecision(5) << std::scientific << normrhs_;
      break;
    case TSI::convnorm_rel:
      oss << std::setw(15) << std::setprecision(5) << std::scientific << normrhs_ / normrhsiter0_;
      break;
    case TSI::convnorm_mix:
      oss << std::setw(15) << std::setprecision(5) << std::scientific
          << std::min(normrhs_, normrhs_ / normrhsiter0_);
      break;
    default:
      FOUR_C_THROW("You should not turn up here.");
      break;
  }

  switch (normtypeinc_)
  {
    case TSI::convnorm_abs:
      oss << std::setw(15) << std::setprecision(5) << std::scientific << norminc_;
      break;
    case TSI::convnorm_rel:
      oss << std::setw(15) << std::setprecision(5) << std::scientific << norminc_ / norminciter0_;
      break;
    case TSI::convnorm_mix:
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

  // ------------------------------------------------------------- thermo
  switch (normtypethrrhs_)
  {
    case Thermo::convnorm_abs:
      oss << std::setw(18) << std::setprecision(5) << std::scientific << normthrrhs_;
      break;
    case Thermo::convnorm_rel:
      oss << std::setw(18) << std::setprecision(5) << std::scientific
          << normthrrhs_ / normthrrhsiter0_;
      break;
    case Thermo::convnorm_mix:
      oss << std::setw(18) << std::setprecision(5) << std::scientific
          << std::min(normthrrhs_, normthrrhs_ / normthrrhsiter0_);
      break;
    default:
      FOUR_C_THROW("You should not turn up here.");
      break;
  }  // switch (normtypethrrhs_)

  switch (normtypetempi_)
  {
    case Thermo::convnorm_abs:
      oss << std::setw(16) << std::setprecision(5) << std::scientific << normtempi_;
      break;
    case Thermo::convnorm_rel:
      oss << std::setw(16) << std::setprecision(5) << std::scientific
          << normtempi_ / normtempiiter0_;
      break;
    case Thermo::convnorm_mix:
      oss << std::setw(16) << std::setprecision(5) << std::scientific
          << std::min(normtempi_, normtempi_ / normtempiiter0_);
      break;
    default:
      FOUR_C_THROW("You should not turn up here.");
      break;
  }  // switch (normtypetempi_)

  if (contact_strategy_lagrange_ != nullptr)
  {
    oss << std::setw(16) << std::setprecision(5) << std::scientific
        << contact_strategy_lagrange_->mech_contact_res_;
    oss << std::setw(16) << std::setprecision(5) << std::scientific
        << contact_strategy_lagrange_->mech_contact_incr_;
    oss << std::setw(16) << std::setprecision(5) << std::scientific
        << contact_strategy_lagrange_->thermo_contact_incr_;
  }

  if (soltech_ == TSI::soltech_ptc)
  {
    oss << std::setw(16) << std::setprecision(5) << std::scientific << dti_;
  }

  // add solution time of to print to screen
  oss << std::setw(12) << std::setprecision(2) << std::scientific << dtsolve_;
  oss << std::setw(12) << std::setprecision(2) << std::scientific
      << timernewton_.totalElapsedTime(true);

  // add contact information
  if (contact_strategy_lagrange_ != nullptr)
  {
    oss << std::setw(12) << std::setprecision(2) << std::scientific << dtcmt_;
    oss << std::setw(11) << contact_strategy_lagrange_->number_of_active_nodes();
    if (contact_strategy_lagrange_->is_friction())
      oss << std::setw(10) << contact_strategy_lagrange_->number_of_slip_nodes();
  }

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
 | print statistics of converged NRI                         dano 11/10 |
 | originally by bborn 08/09                                            |
 *----------------------------------------------------------------------*/
void TSI::Monolithic::print_newton_conv()
{
  // somebody did the door
  return;
}  // print_newton_conv()


/*----------------------------------------------------------------------*
 | evaluate mechanical-thermal system matrix at state        dano 03/11 |
 *----------------------------------------------------------------------*/
void TSI::Monolithic::apply_str_coupl_matrix(
    std::shared_ptr<Core::LinAlg::SparseMatrix> k_st  //!< off-diagonal tangent matrix term
)
{
  // create the parameters for the discretization
  Teuchos::ParameterList sparams;

  //! pointer to the model evaluator data container
  std::shared_ptr<Core::Elements::ParamsMinimal> EvalData =
      std::make_shared<Core::Elements::ParamsMinimal>();

  // set parameters needed for element evaluation
  EvalData->set_action_type(Core::Elements::struct_calc_stifftemp);
  EvalData->set_total_time(time());
  EvalData->set_delta_time(dt());

  const std::string action = "calc_struct_stifftemp";
  sparams.set("action", action);
  // other parameters that might be needed by the elements
  sparams.set("delta time", dt());
  sparams.set("total time", time());

  structure_field()->discretization()->clear_state(true);
  structure_field()->discretization()->set_state(0, "displacement", *structure_field()->dispnp());

  apply_thermo_coupling_state(thermo_field()->tempnp());

  // build specific assemble strategy for mechanical-thermal system matrix
  // from the point of view of structure_field:
  // structdofset = 0, thermdofset = 1
  Core::FE::AssembleStrategy structuralstrategy(0,  // structdofset for row
      1,                                            // thermdofset for column
      k_st,                                         // build mechanical-thermal matrix
      nullptr,                                      // no other matrix or vectors
      nullptr, nullptr, nullptr);


  sparams.set<std::shared_ptr<Core::Elements::ParamsInterface>>("interface", EvalData);
  structure_field()->discretization()->evaluate(sparams, structuralstrategy);
  structure_field()->discretization()->clear_state(true);

  // TODO 2013-11-11 move scaling to the so3_thermo element
  // --> consistent with thermo element and clearer, more consistent

  // for consistent linearisation scale k_st with time factor
  // major switch to different time integrators
  switch (strmethodname_)
  {
    case Inpar::Solid::DynamicType::Statics:
    {
      // continue
      break;
    }
    case Inpar::Solid::DynamicType::OneStepTheta:
    {
      double theta = sdyn_.sublist("ONESTEPTHETA").get<double>("THETA");
      // K_Teffdyn(T_n+1^i) = theta * k_st
      k_st->scale(theta);
      break;
    }
    case Inpar::Solid::DynamicType::GenAlpha:
    {
      double alphaf = sdyn_.sublist("GENALPHA").get<double>("ALPHA_F");
      // K_Teffdyn(T_n+1) = (1-alphaf_) . kst
      // Lin(dT_n+1-alphaf_/ dT_n+1) = (1-alphaf_)
      k_st->scale(1.0 - alphaf);
      break;
    }
    default:
      FOUR_C_THROW("Don't know what to do...");
      break;
  }  // end of switch(strmethodname_)

}  // apply_str_coupl_matrix()


/*----------------------------------------------------------------------*
 | evaluate thermal-mechanical system matrix at state        dano 03/11 |
 *----------------------------------------------------------------------*/
void TSI::Monolithic::apply_thermo_coupl_matrix(
    std::shared_ptr<Core::LinAlg::SparseMatrix> k_ts  //!< off-diagonal tangent matrix term
)
{
  // create the parameters for the discretization
  Teuchos::ParameterList tparams;
  // action for elements
  const Thermo::Action action = Thermo::calc_thermo_coupltang;
  tparams.set<Thermo::Action>("action", action);
  // other parameters that might be needed by the elements
  tparams.set("delta time", dt());
  tparams.set("total time", time());

  // create specific time integrator
  const Teuchos::ParameterList& tdyn = Global::Problem::instance()->thermal_dynamic_params();
  tparams.set<Thermo::DynamicType>(
      "time integrator", Teuchos::getIntegralValue<Thermo::DynamicType>(tdyn, "DYNAMICTYPE"));
  tparams.set<Inpar::Solid::DynamicType>("structural time integrator", strmethodname_);
  switch (Teuchos::getIntegralValue<Thermo::DynamicType>(tdyn, "DYNAMICTYPE"))
  {
    // static analysis
    case Thermo::DynamicType::Statics:
    {
      // continue
      break;
    }
    // dynamic analysis
    case Thermo::DynamicType::OneStepTheta:
    {
      // K_Td = theta . k_Td^e
      double theta = tdyn.sublist("ONESTEPTHETA").get<double>("THETA");
      tparams.set("theta", theta);
      break;
    }
    case Thermo::DynamicType::GenAlpha:
    {
      double alphaf = tdyn.sublist("GENALPHA").get<double>("ALPHA_F");
      tparams.set("alphaf", alphaf);
      break;
    }
    case Thermo::DynamicType::Undefined:
    default:
    {
      FOUR_C_THROW("Don't know what to do...");
      break;
    }
  }  // switch (Thermo::DynamicType)

  switch (strmethodname_)
  {
    case Inpar::Solid::DynamicType::Statics:
    {
      // continue
      break;
    }
    case Inpar::Solid::DynamicType::OneStepTheta:
    {
      // put the structural theta value to the thermal parameter list
      double str_theta = sdyn_.sublist("ONESTEPTHETA").get<double>("THETA");
      tparams.set("str_theta", str_theta);
      break;
    }
    case Inpar::Solid::DynamicType::GenAlpha:
    {
      // put the structural theta value to the thermal parameter list
      double str_beta = sdyn_.sublist("GENALPHA").get<double>("BETA");
      double str_gamma = sdyn_.sublist("GENALPHA").get<double>("GAMMA");

      tparams.set("str_beta", str_beta);
      tparams.set("str_gamma", str_gamma);
      break;
    }
    default:
      FOUR_C_THROW("Don't know what to do...");
      break;
  }


  thermo_field()->discretization()->clear_state(true);
  // set the variables that are needed by the elements
  thermo_field()->discretization()->set_state(0, "temperature", *thermo_field()->tempnp());

  apply_struct_coupling_state(structure_field()->dispnp(), vel_);

  // build specific assemble strategy for the thermal-mechanical system matrix
  // from the point of view of ThermoField:
  // thermdofset = 0, structdofset = 1
  Core::FE::AssembleStrategy thermostrategy(0,  // thermdofset for row
      1,                                        // structdofset for column
      k_ts,                                     // thermal-mechanical matrix
      nullptr,                                  // no other matrix or vectors
      nullptr, nullptr, nullptr);

  // evaluate the thermal-mechanical system matrix on the thermal element
  thermo_field()->discretization()->evaluate(tparams, thermostrategy);
  thermo_field()->discretization()->clear_state(true);
}  // ApplyThermoCouplMatrix()


/*----------------------------------------------------------------------*
 | evaluate thermal-mechanical system matrix at state        dano 12/12 |
 *----------------------------------------------------------------------*/
void TSI::Monolithic::apply_thermo_coupl_matrix_conv_bc(
    std::shared_ptr<Core::LinAlg::SparseMatrix> k_ts  //!< off-diagonal tangent matrix term
)
{
  std::vector<const Core::Conditions::Condition*> cond;
  std::string condstring("ThermoConvections");
  thermo_field()->discretization()->get_condition(condstring, cond);
  if (cond.size() > 0)
  {
    // create the parameters for the discretization
    Teuchos::ParameterList tparams;
    // action for elements
    tparams.set<Thermo::BoundaryAction>("action", Thermo::calc_thermo_fextconvection_coupltang);
    // other parameters that might be needed by the elements
    tparams.set("delta time", dt());
    tparams.set("total time", time());
    // create specific time integrator
    const Teuchos::ParameterList& tdyn = Global::Problem::instance()->thermal_dynamic_params();
    tparams.set<Thermo::DynamicType>(
        "time integrator", Teuchos::getIntegralValue<Thermo::DynamicType>(tdyn, "DYNAMICTYPE"));
    tparams.set<Inpar::Solid::DynamicType>("structural time integrator", strmethodname_);
    switch (Teuchos::getIntegralValue<Thermo::DynamicType>(tdyn, "DYNAMICTYPE"))
    {
      // static analysis
      case Thermo::DynamicType::Statics:
      {
        break;
      }
      // dynamic analysis
      case Thermo::DynamicType::OneStepTheta:
      {
        // K_Td = theta . k_Td^e
        double theta = tdyn.sublist("ONESTEPTHETA").get<double>("THETA");
        tparams.set("theta", theta);
        // put the structural theta value to the thermal parameter list
        double str_theta = sdyn_.sublist("ONESTEPTHETA").get<double>("THETA");
        tparams.set("str_theta", str_theta);
        break;
      }
      case Thermo::DynamicType::GenAlpha:
      {
        // K_Td = alphaf . k_Td^e
        double alphaf = tdyn.sublist("GENALPHA").get<double>("ALPHA_F");
        tparams.set("alphaf", alphaf);

        // put the structural theta value to the thermal parameter list
        double str_beta = sdyn_.sublist("GENALPHA").get<double>("BETA");
        double str_gamma = sdyn_.sublist("GENALPHA").get<double>("GAMMA");
        tparams.set("str_beta", str_beta);
        tparams.set("str_gamma", str_gamma);
        break;
      }
      case Thermo::DynamicType::Undefined:
      default:
      {
        FOUR_C_THROW("Don't know what to do...");
        break;
      }
    }  // end(switch)
    // clear all states set in discretization
    thermo_field()->discretization()->clear_state(true);
    // set the variables that are needed by the elements
    thermo_field()->discretization()->set_state(0, "temperature", *thermo_field()->tempnp());
    apply_struct_coupling_state(structure_field()->dispnp(), vel_);

    // build specific assemble strategy for the thermal-mechanical system matrix
    // from the point of view of ThermoField:
    // thermdofset = 0, structdofset = 1
    Core::FE::AssembleStrategy thermostrategy(0,  // thermdofset for row
        1,                                        // structdofset for column
        k_ts,                                     // thermal-mechanical matrix
        nullptr,                                  // no other matrix or vectors
        nullptr, nullptr, nullptr);

    // evaluate the thermal-mechanical system matrix on the thermal element
    thermo_field()->discretization()->evaluate_condition(tparams, thermostrategy, condstring);
    // clear all states set in discretization
    thermo_field()->discretization()->clear_state(true);
  }  // cond.size()>0

}  // ApplyThermoCouplMatrix()


/*----------------------------------------------------------------------*
 | map containing the dofs with Dirichlet BC                 dano 03/11 |
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Map> TSI::Monolithic::combined_dbc_map()
{
  const std::shared_ptr<const Core::LinAlg::Map> scondmap =
      structure_field()->get_dbc_map_extractor()->cond_map();
  const std::shared_ptr<const Core::LinAlg::Map> tcondmap =
      thermo_field()->get_dbc_map_extractor()->cond_map();
  std::shared_ptr<Core::LinAlg::Map> condmap = Core::LinAlg::merge_map(scondmap, tcondmap, false);
  return condmap;

}  // combined_dbc_map()



/*----------------------------------------------------------------------*
 | recover structural and thermal Lagrange multipliers from  seitz 11/15|
 | displacements and temperature                                        |
 *----------------------------------------------------------------------*/
void TSI::Monolithic::recover_struct_therm_lm()
{
  // only in the case of contact
  if (contact_strategy_lagrange_ == nullptr) return;

  // split the increment
  std::shared_ptr<Core::LinAlg::Vector<double>> sx;
  std::shared_ptr<Core::LinAlg::Vector<double>> tx;

  // extract field vectors
  extract_field_vectors(iterinc_, sx, tx);

  contact_strategy_lagrange_->recover_coupled(sx, tx, coupST_);

  return;
}


/*----------------------------------------------------------------------*
 | scale system, i.e. apply infnorm scaling to linear        dano 02/13 |
 | block system before solving system                                   |
 *----------------------------------------------------------------------*/
void TSI::Monolithic::scale_system(
    Core::LinAlg::BlockSparseMatrixBase& mat, Core::LinAlg::Vector<double>& b)
{
  // should we scale the system?
  const bool scaling_infnorm = tsidynmono_.get<bool>("INFNORMSCALING");

  if (scaling_infnorm)
  {
    // TODO: The matrices are modified here. Do we have to change them back later on?

    Core::LinAlg::SparseMatrix& A_00 = mat.matrix(0, 0);
    srowsum_ = std::make_shared<Core::LinAlg::Vector<double>>(A_00.row_map(), false);
    scolsum_ = std::make_shared<Core::LinAlg::Vector<double>>(A_00.row_map(), false);
    A_00.inv_row_sums(*srowsum_);
    A_00.inv_col_sums(*scolsum_);
    if (A_00.left_scale(*srowsum_) or A_00.right_scale(*scolsum_) or
        mat.matrix(0, 1).left_scale(*srowsum_) or mat.matrix(1, 0).right_scale(*scolsum_))
      FOUR_C_THROW("structure scaling failed");

    Core::LinAlg::SparseMatrix& A_11 = mat.matrix(1, 1);
    trowsum_ = std::make_shared<Core::LinAlg::Vector<double>>(A_11.row_map(), false);
    tcolsum_ = std::make_shared<Core::LinAlg::Vector<double>>(A_11.row_map(), false);
    A_11.inv_row_sums(*trowsum_);
    A_11.inv_col_sums(*tcolsum_);
    if (A_11.left_scale(*trowsum_) or A_11.right_scale(*tcolsum_) or
        mat.matrix(1, 0).left_scale(*trowsum_) or mat.matrix(0, 1).right_scale(*tcolsum_))
      FOUR_C_THROW("thermo scaling failed");

    std::shared_ptr<Core::LinAlg::Vector<double>> sx = extractor()->extract_vector(b, 0);
    std::shared_ptr<Core::LinAlg::Vector<double>> tx = extractor()->extract_vector(b, 1);

    sx->multiply(1.0, *srowsum_, *sx, 0.0);
    tx->multiply(1.0, *trowsum_, *tx, 0.0);

    extractor()->insert_vector(*sx, 0, b);
    extractor()->insert_vector(*tx, 1, b);
  }
}


/*----------------------------------------------------------------------*
 | unscale solution after solving the linear system          dano 02/13 |
 *----------------------------------------------------------------------*/
void TSI::Monolithic::unscale_solution(Core::LinAlg::BlockSparseMatrixBase& mat,
    Core::LinAlg::Vector<double>& x, Core::LinAlg::Vector<double>& b)
{
  const bool scaling_infnorm = tsidynmono_.get<bool>("INFNORMSCALING");

  if (scaling_infnorm)
  {
    std::shared_ptr<Core::LinAlg::Vector<double>> sy = extractor()->extract_vector(x, 0);
    std::shared_ptr<Core::LinAlg::Vector<double>> ty = extractor()->extract_vector(x, 1);

    sy->multiply(1.0, *scolsum_, *sy, 0.0);
    ty->multiply(1.0, *tcolsum_, *ty, 0.0);

    extractor()->insert_vector(*sy, 0, x);
    extractor()->insert_vector(*ty, 1, x);

    std::shared_ptr<Core::LinAlg::Vector<double>> sx = extractor()->extract_vector(b, 0);
    std::shared_ptr<Core::LinAlg::Vector<double>> tx = extractor()->extract_vector(b, 1);

    sx->reciprocal_multiply(1.0, *srowsum_, *sx, 0.0);
    tx->reciprocal_multiply(1.0, *trowsum_, *tx, 0.0);

    extractor()->insert_vector(*sx, 0, b);
    extractor()->insert_vector(*tx, 1, b);

    Core::LinAlg::SparseMatrix& A_00 = mat.matrix(0, 0);
    srowsum_->reciprocal(*srowsum_);
    scolsum_->reciprocal(*scolsum_);
    if (A_00.left_scale(*srowsum_) or A_00.right_scale(*scolsum_) or
        mat.matrix(0, 1).left_scale(*srowsum_) or mat.matrix(1, 0).right_scale(*scolsum_))
      FOUR_C_THROW("structure scaling failed");

    Core::LinAlg::SparseMatrix& A_11 = mat.matrix(1, 1);
    trowsum_->reciprocal(*trowsum_);
    tcolsum_->reciprocal(*tcolsum_);
    if (A_11.left_scale(*trowsum_) or A_11.right_scale(*tcolsum_) or
        mat.matrix(1, 0).left_scale(*trowsum_) or mat.matrix(0, 1).right_scale(*tcolsum_))
      FOUR_C_THROW("thermo scaling failed");
  }
}


/*----------------------------------------------------------------------*
 | calculate vector norm                                     dano 04/13 |
 *----------------------------------------------------------------------*/
double TSI::Monolithic::calculate_vector_norm(
    const enum TSI::VectorNorm norm, const Core::LinAlg::Vector<double>& vect)
{
  // L1 norm
  // norm = sum_0^i vect[i]
  if (norm == TSI::norm_l1)
  {
    double vectnorm;
    vect.norm_1(&vectnorm);
    return vectnorm;
  }
  // L2/Euclidian norm
  // norm = sqrt{sum_0^i vect[i]^2 }
  else if (norm == TSI::norm_l2)
  {
    double vectnorm;
    vect.norm_2(&vectnorm);
    return vectnorm;
  }
  // RMS norm
  // norm = sqrt{sum_0^i vect[i]^2 }/ sqrt{length_vect}
  else if (norm == TSI::norm_rms)
  {
    double vectnorm;
    vect.norm_2(&vectnorm);
    return vectnorm / sqrt((double)vect.global_length());
  }
  // infinity/maximum norm
  // norm = max( vect[i] )
  else if (norm == TSI::norm_inf)
  {
    double vectnorm;
    vect.norm_inf(&vectnorm);
    return vectnorm;
  }
  // norm = sum_0^i vect[i]/length_vect
  else if (norm == TSI::norm_l1_scaled)
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
 | set parameters for TSI remaining constant over whole      dano 04/13 |
 | simulation                                                           |
 *----------------------------------------------------------------------*/
void TSI::Monolithic::set_default_parameters()
{
  // time parameters
  // call the TSI parameter list
  const Teuchos::ParameterList& tdyn = Global::Problem::instance()->thermal_dynamic_params();

  // get the parameters for the Newton iteration
  itermax_ = tsidyn_.get<int>("ITEMAX");
  itermin_ = tsidyn_.get<int>("ITEMIN");

  // what kind of norm do we wanna test for coupled TSI problem
  normtypeinc_ = Teuchos::getIntegralValue<TSI::ConvNorm>(tsidyn_, "NORM_INC");
  normtyperhs_ = Teuchos::getIntegralValue<TSI::ConvNorm>(tsidynmono_, "NORM_RESF");
  // what kind of norm do we wanna test for the single fields
  normtypedisi_ = Teuchos::getIntegralValue<Inpar::Solid::ConvNorm>(sdyn_, "NORM_DISP");
  normtypestrrhs_ = Teuchos::getIntegralValue<Inpar::Solid::ConvNorm>(sdyn_, "NORM_RESF");
  enum Inpar::Solid::VectorNorm striternorm =
      Teuchos::getIntegralValue<Inpar::Solid::VectorNorm>(sdyn_, "ITERNORM");
  normtypetempi_ = Teuchos::getIntegralValue<Thermo::ConvNorm>(tdyn, "NORM_TEMP");
  normtypethrrhs_ = Teuchos::getIntegralValue<Thermo::ConvNorm>(tdyn, "NORM_RESF");
  enum Thermo::VectorNorm thriternorm =
      Teuchos::getIntegralValue<Thermo::VectorNorm>(tdyn, "ITERNORM");
  // in total when do we reach a converged state for complete problem
  combincrhs_ = Teuchos::getIntegralValue<TSI::BinaryOp>(tsidynmono_, "NORMCOMBI_RESFINC");

  switch (combincrhs_)
  {
    case TSI::bop_and:
    {
      if (Core::Communication::my_mpi_rank(get_comm()) == 0)
        std::cout << "Convergence test of TSI:\n res, inc with 'AND'." << std::endl;
      break;
    }
    case TSI::bop_or:
    {
      if (Core::Communication::my_mpi_rank(get_comm()) == 0)
        std::cout << "Convergence test of TSI:\n res, inc with 'OR'." << std::endl;
      break;
    }
    case TSI::bop_coupl_and_single:
    {
      if (Core::Communication::my_mpi_rank(get_comm()) == 0)
        std::cout
            << "Convergence test of TSI:\n res, inc, str-res, thermo-res, dis, temp with 'AND'."
            << std::endl;
      break;
    }
    case TSI::bop_coupl_or_single:
    {
      if (Core::Communication::my_mpi_rank(get_comm()) == 0)
        std::cout << "Convergence test of TSI:\n (res, inc) or (str-res, thermo-res, dis, temp)."
                  << std::endl;
      break;
    }
    case TSI::bop_and_single:
    {
      if (Core::Communication::my_mpi_rank(get_comm()) == 0)
        std::cout << "Convergence test of TSI:\n str-res, thermo-res, dis, temp with 'AND'."
                  << std::endl;
      break;
    }
    case TSI::bop_or_single:
    {
      if (Core::Communication::my_mpi_rank(get_comm()) == 0)
        std::cout << "Convergence test of TSI:\n str-res, thermo-res, dis, temp with 'OR'."
                  << std::endl;
      break;
    }
    default:
    {
      FOUR_C_THROW("Something went terribly wrong with binary operator!");
      break;
    }
  }  // switch (combincrhs_)

  // convert the single field norms to be used within TSI
  // what norm is used for structure
  switch (striternorm)
  {
    case Inpar::Solid::norm_l1:
      iternormstr_ = TSI::norm_l1;
      break;
    case Inpar::Solid::norm_l2:
      iternormstr_ = TSI::norm_l2;
      break;
    case Inpar::Solid::norm_rms:
      iternormstr_ = TSI::norm_rms;
      break;
    case Inpar::Solid::norm_inf:
      iternormstr_ = TSI::norm_inf;
      break;
    case Inpar::Solid::norm_vague:
    default:
      FOUR_C_THROW("STR norm is not determined");
      break;
  }  // switch (striternorm)

  // what norm is used for thermo
  switch (thriternorm)
  {
    case Thermo::norm_l1:
      iternormthr_ = TSI::norm_l1;
      break;
    case Thermo::norm_l2:
      iternormthr_ = TSI::norm_l2;
      break;
    case Thermo::norm_rms:
      iternormthr_ = TSI::norm_rms;
      break;
    case Thermo::norm_inf:
      iternormthr_ = TSI::norm_inf;
      break;
    case Thermo::norm_vague:
    default:
    {
      FOUR_C_THROW("Thermo norm is not determined.");
      break;
    }
  }  // switch (thriternorm)

  // if scaled L1-norm is wished to be used
  if ((iternorm_ == TSI::norm_l1_scaled) and
      ((combincrhs_ == TSI::bop_coupl_and_single) or (combincrhs_ == TSI::bop_coupl_or_single)))
  {
    iternormstr_ = TSI::norm_l1_scaled;
    iternormthr_ = TSI::norm_l1_scaled;
  }

  // test the TSI-residual and the TSI-increment
  tolinc_ = tsidynmono_.get<double>("TOLINC");
  tolrhs_ = tsidynmono_.get<double>("CONVTOL");

  // get the single field tolerances from this field itselves
  toldisi_ = sdyn_.get<double>("TOLDISP");
  tolstrrhs_ = sdyn_.get<double>("TOLRES");
  toltempi_ = tdyn.get<double>("TOLTEMP");
  tolthrrhs_ = tdyn.get<double>("TOLRES");

  // initialise norms for coupled TSI problem
  normrhs_ = 0.0;
  normrhsiter0_ = 0.0;
  norminc_ = 0.0;
  norminciter0_ = 0.0;

  // initialise norms for single field tests
  normdisi_ = 0.0;
  normstrrhs_ = 0.0;
  normstrrhsiter0_ = 0.0;
  normtempi_ = 0.0;
  normthrrhs_ = 0.0;
  normthrrhsiter0_ = 0.0;

  return;

}  // SetDefaultParameter()


/*----------------------------------------------------------------------*
 | calculate nodal TSI results for evaluation                dano 09/13 |
 | used for thermoplasticity and necking to calculate displacements,    |
 | temperatures, and reaction forces at different points                |
 *----------------------------------------------------------------------*/
void TSI::Monolithic::calculate_necking_tsi_results()
{
  // be aware, parallel bug arises in debug mode

  // --------------------------------------------- initialise/ define constants

  // initialise initial temperature required to calculate temperature increase
  const double inittemp = 293.0;

  // necking point A
  const double necking_x = 6.413;
  const double necking_y = 0.0;
  const double necking_z = -13.3335;
  // --> point A is extracted via restriction of coordinates

  // top point B
  const double top_x = 6.413;
  const double top_y = 0.0;
  const double top_z = 13.3335;
  // --> point B is extracted via DBC

  //---------------------------------------------------------------------------
  // -------------- get the nodes with STRUCTURAL Dirichlet boundary conditions
  //---------------------------------------------------------------------------

  // initialise a vector containing all structural DBC
  std::vector<const Core::Conditions::Condition*> dbc;
  structure_field()->discretization()->get_condition("Dirichlet", dbc);

  // initialise a vector containing all DBC in a special direction (here: in z)
  std::vector<int> one_dof_in_dbc(1);
  one_dof_in_dbc.at(0) = -1;

  // local list of found structural DOF IDs that have a DBC
  // in this special case DBC are applied at the nodes which are interested for
  // evaluation
  std::vector<int> sdata;

  // loop over Dirichlet boundary conditions
  for (int i = 0; i < (int)dbc.size(); ++i)
  {
    // get nodes which have DBCs
    const std::vector<int>* nodeids_withdbc = dbc[i]->get_nodes();
    if (!nodeids_withdbc) FOUR_C_THROW("Condition does not have Node Ids");

    // loop over DBC nodes
    for (int k = 0; k < (int)(*nodeids_withdbc).size(); ++k)
    {
      int gid = (*nodeids_withdbc)[k];
      // do only nodes which are in my discretisation
      if (structure_field()->discretization()->node_row_map()->my_gid(gid) == false) continue;

      // -------------------- evaluation in special direction, here z-direction
      // get node with global id gid
      Core::Nodes::Node* node = structure_field()->discretization()->g_node(gid);
      if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
      // check coordinates in z-direction, i.e. third value of X()
      double zcoord = node->x()[2];
      // possible push-back
      bool this_is_new_gid = true;
      // get the z-displacement DOFS located at the top surface (z=13.3335mm)
      if (abs(zcoord - top_z) < 1.0e-8)  // change here value for different geometries
      {
        for (unsigned j = 0; j < sdata.size(); j++)
        {
          if (sdata.at(j) == structure_field()->discretization()->dof(0, node, 2))
            this_is_new_gid = false;
        }
        if (this_is_new_gid) sdata.push_back(structure_field()->discretization()->dof(0, node, 2));
        one_dof_in_dbc.at(0) = structure_field()->discretization()->dof(0, node, 2);
      }  // top surface
    }  // loop over DBC nodes
  }  // loop over all STRUCTURAL DBC conditions

  // map containing all z-displacement DOFs which have a DBC
  Core::LinAlg::Map newdofmap(
      -1, (int)sdata.size(), sdata.data(), 0, structure_field()->discretization()->get_comm());

  //---------------------------------------------------------------------------
  // ------------------------------------ initialise STRUCTURAL output variables
  //---------------------------------------------------------------------------

  // ---------------------------------------------------------------- top force
  // nodal reaction force at outer edge for whole support area
  std::shared_ptr<Core::LinAlg::Vector<double>> tension =
      std::make_shared<Core::LinAlg::Vector<double>>(newdofmap,  // map containing
                                                                 // all DOFs at top surf with DBC
          false);
  // copy the structural reaction force to tension
  Core::LinAlg::export_to(*(structure_field()->freact()), *tension);
  double top_force_local = 0.0;  // local force
  for (int i = 0; i < tension->local_length(); i++) top_force_local -= (*tension)[i];

  // complete force pointing in axial direction
  double top_force_global = 0.0;

  // sum all nodal forces (top_force_local) in one global vector (top_force_global)
  top_force_global = Core::Communication::sum_all(
      top_force_local, structure_field()->discretization()->get_comm());

  // --------------------------------------------- reaction force of whole body
  // due to symmetry only 1/8 is simulated, i.e. only 1/4 of the surface is considered
  // R = force_top * 4
  double top_reaction_force = 4 * top_force_global;

  // --------------------------------------------------------- top displacement
  // top displacement, i.e. displacement at outer edge in axial-direction
  // (here: in z-direction)

  // get the DOFs corresponding to z-displacements and located at the top surface
  std::vector<double> top_disp_local(1);
  top_disp_local.at(0) = 0.0;

  std::vector<int> one_dof_in_dbc_global(1);
  one_dof_in_dbc_global.at(0) = -1;

  one_dof_in_dbc_global.at(0) = Core::Communication::max_all(
      one_dof_in_dbc.at(0), structure_field()->discretization()->get_comm());

  // extract axial displacements (here z-displacements) of top surface
  if (structure_field()->discretization()->dof_row_map()->my_gid(one_dof_in_dbc_global.at(0)))
  {
    top_disp_local =
        Core::FE::extract_values(*(structure_field()->dispnp()), one_dof_in_dbc_global);
  }

  // initialise the top displacement
  double top_disp_global = 0.0;
  // sum all nodal displacements (top_disp_local) in one global vector (top_disp_global)
  top_disp_global = Core::Communication::sum_all(
      top_disp_local.at(0), structure_field()->discretization()->get_comm());

  // ------------------------------------------------ necking radius at point A
  // necking, i.e. radial displacements in centre plane (here: xy-plane)

  // get the necking DOFs
  // i.e. displacement-DOFs of the xy-plane in the middle of the body
  std::vector<int> necking_radius_dof(1);
  necking_radius_dof.at(0) = -1.;
  for (int k = 0; k < (int)structure_field()->discretization()->node_row_map()->num_my_elements();
      k++)
  {
    Core::Nodes::Node* node = structure_field()->discretization()->l_row_node(k);
    // change here value for different geometries
    if ((abs(node->x()[0] - necking_x) < 1.e-8)      // x-direction
        and (abs(node->x()[1] - necking_y) < 1.e-8)  // y-direction
        and (abs(node->x()[2] - necking_z) < 1.e-8)  // z-direction
    )
    {
      // we choose point A (6.413mm / 0mm / -13.3335mm)
      necking_radius_dof.at(0) = structure_field()->discretization()->dof(0, node, 0);
      break;  // we only look for one specific node, if we have found it: stop

    }  // end point A(6.413/0/-13.3335)
  }  // sum all nodes
  std::vector<double> necking_radius(1);
  necking_radius.at(0) = 0.0;
  if (necking_radius_dof.at(0) != -1)
  {
    necking_radius = Core::FE::extract_values(*(structure_field()->dispnp()), necking_radius_dof);
  }

  // sum necking deformations in the global variable necking_radius_global
  double necking_radius_global = 0.0;
  necking_radius_global = Core::Communication::sum_all(
      necking_radius.at(0), structure_field()->discretization()->get_comm());

  //---------------------------------------------------------------------------
  // -------------------------------------------------- initialise TEMPERATURES
  //---------------------------------------------------------------------------

  // ------------------------------------------- necking temperature at point A

  // necking temperature at point A, i.e. temperature at the outer side in the middle
  // A (6.413mm / 0.0mm / 13.3335mm)
  std::vector<int> neck_temperature_dof(1);
  neck_temperature_dof.at(0) = -1.0;
  for (int k = 0; k < (int)thermo_field()->discretization()->node_row_map()->num_my_elements(); k++)
  {
    Core::Nodes::Node* node = thermo_field()->discretization()->l_row_node(k);
    // change here value for different geometries
    if ((abs(node->x()[0] - necking_x) < 1.e-8)      // x-direction
        and (abs(node->x()[1] - necking_y) < 1.e-8)  // y-direction
        and (abs(node->x()[2] - necking_z) < 1.e-8)  // z-direction
    )
    {
      neck_temperature_dof.at(0) = thermo_field()->discretization()->dof(0, node, 0);
      break;  // we only look for one specific node, if we have found it: stop
    }
  }
  std::vector<double> temperature(1);
  temperature.at(0) = 0.0;
  if (neck_temperature_dof.at(0) != -1)
  {
    temperature = Core::FE::extract_values(*(thermo_field()->tempnp()),  // global (i)
        neck_temperature_dof  // global ids to be extracted
    );
  }
  // sum necking temperatures in the variable temperature_global
  double necking_temperature_global = 0.0;
  necking_temperature_global =
      Core::Communication::sum_all(temperature.at(0), thermo_field()->discretization()->get_comm());

  // -----------------------------------------temperatures at top, i.e. point B

  // necking temperature at point B, i.e. temperature at the outer side at top
  // B (6.413mm / 0.0mm / -13.3335mm)
  std::vector<int> top_temperature_dof(1);
  top_temperature_dof.at(0) = -1.0;
  for (int k = 0; k < (int)thermo_field()->discretization()->node_row_map()->num_my_elements(); k++)
  {
    Core::Nodes::Node* node = thermo_field()->discretization()->l_row_node(k);
    // change here value for different geometries
    if ((abs(node->x()[0] - top_x) < 1.e-8)      // x-direction
        and (abs(node->x()[1] - top_y) < 1.e-8)  // y-direction
        and (abs(node->x()[2] - top_z) < 1.e-8)  // z-direction
    )
    {
      top_temperature_dof.at(0) = thermo_field()->discretization()->dof(0, node, 0);
      break;  // we only look for one specific node, if we have found it: stop
    }
  }  // loop over thermal nodes

  // extract top-temperatures of top surface out of global temperature vector
  std::vector<double> top_temperature_local(1);
  top_temperature_local.at(0) = 0.0;
  if (top_temperature_dof.at(0) != -1.)
  {
    top_temperature_local =
        Core::FE::extract_values(*(thermo_field()->tempnp()),  // global vector (i)
            top_temperature_dof                                // global ids to be extracted
        );
  }

  // sum top-temperatures in the variable top_temperature_global
  double top_temperature_global = 0.0;
  top_temperature_global = Core::Communication::sum_all(
      top_temperature_local.at(0), thermo_field()->discretization()->get_comm());

  // -------------------------------------------------- print results to screen
  std::cout.precision(7);
  std::cout << std::scientific;
  std::cout << std::fixed;
  if (Core::Communication::my_mpi_rank(thermo_field()->discretization()->get_comm()) == 0)
  {
    std::cout << "OUTPUT:\ttop-disp \ttop-freact \tneck-disp \tneck-tempi \ttop-tempi \ttop-force\n"
              << "\t" << top_disp_global << "\t" << top_reaction_force << "\t"
              << necking_radius_global << "\t" << (necking_temperature_global - inittemp) << "\t"
              << (top_temperature_global - inittemp) << "\t" << top_force_global << std::endl;
  }

}  // calculate_necking_tsi_results()

/*----------------------------------------------------------------------*
 |                                                                      |
 *----------------------------------------------------------------------*/
void TSI::Monolithic::prepare_output()
{
  // set temperatures on structure field for evaluating stresses
  apply_thermo_coupling_state(thermo_field()->tempnp());
  // prepare output (i.e. calculate stresses, strains, energies)
  constexpr bool force_prepare = false;
  structure_field()->prepare_output(force_prepare);

  // reset states
  structure_field()->discretization()->clear_state(true);
}

/*----------------------------------------------------------------------*
 |                                                                      |
 *----------------------------------------------------------------------*/
void TSI::Monolithic::prepare_contact_strategy() { TSI::Algorithm::prepare_contact_strategy(); }

/*----------------------------------------------------------------------*
 |                                                                      |
 *----------------------------------------------------------------------*/
void TSI::Monolithic::apply_struct_coupling_state(
    std::shared_ptr<const Core::LinAlg::Vector<double>> disp,
    std::shared_ptr<const Core::LinAlg::Vector<double>> vel)
{
  if (matchinggrid_)
  {
    if (disp != nullptr) thermo_field()->discretization()->set_state(1, "displacement", *disp);
    if (vel != nullptr) thermo_field()->discretization()->set_state(1, "velocity", *vel);
  }
  else
  {
    if (disp != nullptr)
      thermo_field()->discretization()->set_state(
          1, "displacement", *volcoupl_->apply_vector_mapping21(*disp));
    if (vel != nullptr)
      thermo_field()->discretization()->set_state(
          1, "velocity", *volcoupl_->apply_vector_mapping21(*vel));
  }
}  // apply_struct_coupling_state()



/*----------------------------------------------------------------------*
 |                                                                      |
 *----------------------------------------------------------------------*/
void TSI::Monolithic::fix_time_integration_params()
{
  if (Teuchos::getIntegralValue<Thermo::DynamicType>(
          Global::Problem::instance()->thermal_dynamic_params(), "DYNAMICTYPE") ==
      Thermo::DynamicType::GenAlpha)
  {
    Teuchos::ParameterList& ga = const_cast<Teuchos::ParameterList&>(
        Global::Problem::instance()->thermal_dynamic_params().sublist("GENALPHA"));
    double rhoinf = ga.get<double>("RHO_INF");

    if (rhoinf != -1.)
    {
      if ((rhoinf < 0.0) or (rhoinf > 1.0)) FOUR_C_THROW("rho_inf out of range [0.0,1.0]");
      double alpham = 0.5 * (3.0 - rhoinf) / (rhoinf + 1.0);
      double alphaf = 1.0 / (rhoinf + 1.0);
      double gamma = 0.5 + alpham - alphaf;
      ga.set<double>("GAMMA", gamma);
      ga.set<double>("ALPHA_F", alphaf);
      ga.set<double>("ALPHA_M", alpham);
    }
  }

  if (Teuchos::getIntegralValue<Inpar::Solid::DynamicType>(
          Global::Problem::instance()->structural_dynamic_params(), "DYNAMICTYPE") ==
      Inpar::Solid::DynamicType::GenAlpha)
  {
    Teuchos::ParameterList& ga = const_cast<Teuchos::ParameterList&>(
        Global::Problem::instance()->structural_dynamic_params().sublist("GENALPHA"));
    double rhoinf = ga.get<double>("RHO_INF");

    if (rhoinf != -1.)
    {
      double alpham = (2.0 * rhoinf - 1.0) / (rhoinf + 1.0);
      double alphaf = rhoinf / (rhoinf + 1.0);
      double beta = 0.25 * (1.0 - alpham + alphaf) * (1.0 - alpham + alphaf);
      double gamma = 0.5 - alpham + alphaf;

      ga.set<double>("BETA", beta);
      ga.set<double>("GAMMA", gamma);
      ga.set<double>("ALPHA_F", alphaf);
      ga.set<double>("ALPHA_M", alpham);
    }
  }
}

/*----------------------------------------------------------------------*
 |                                                                      |
 *----------------------------------------------------------------------*/
void TSI::Monolithic::apply_dbc()
{
  auto k_ss = std::make_shared<Core::LinAlg::SparseMatrix>(systemmatrix_->matrix(0, 0));
  auto k_st = std::make_shared<Core::LinAlg::SparseMatrix>(systemmatrix_->matrix(0, 1));
  auto k_ts = std::make_shared<Core::LinAlg::SparseMatrix>(systemmatrix_->matrix(1, 0));
  auto k_tt = std::make_shared<Core::LinAlg::SparseMatrix>(systemmatrix_->matrix(1, 1));

  if (locsysman_ != nullptr)
  {
    {
      locsysman_->rotate_global_to_local(k_ss);
      k_ss->apply_dirichlet_with_trafo(
          *locsysman_->trafo(), *structure_field()->get_dbc_map_extractor()->cond_map(), true);
      locsysman_->rotate_local_to_global(*k_ss);
    }
    {
      locsysman_->rotate_global_to_local(k_st);
      k_st->apply_dirichlet_with_trafo(
          *locsysman_->trafo(), *structure_field()->get_dbc_map_extractor()->cond_map(), false);
      locsysman_->rotate_local_to_global(*k_st);
    }
  }
  else
  {
    k_ss->apply_dirichlet(*structure_field()->get_dbc_map_extractor()->cond_map(), true);
    k_st->apply_dirichlet(*structure_field()->get_dbc_map_extractor()->cond_map(), false);
  }
  k_ts->apply_dirichlet(*thermo_field()->get_dbc_map_extractor()->cond_map(), false);
  k_tt->apply_dirichlet(*thermo_field()->get_dbc_map_extractor()->cond_map(), true);


  systemmatrix_->un_complete();
  systemmatrix_->assign(0, 0, Core::LinAlg::DataAccess::Share, *k_ss);
  systemmatrix_->assign(0, 1, Core::LinAlg::DataAccess::Share, *k_st);
  systemmatrix_->assign(1, 0, Core::LinAlg::DataAccess::Share, *k_ts);
  systemmatrix_->assign(1, 1, Core::LinAlg::DataAccess::Share, *k_tt);
  systemmatrix_->complete();


  if (locsysman_ != nullptr)
  {
    std::shared_ptr<Core::LinAlg::Vector<double>> s_rhs, t_rhs;
    extract_field_vectors(rhs_, s_rhs, t_rhs);
    locsysman_->rotate_global_to_local(*s_rhs);
    Core::LinAlg::apply_dirichlet_to_system(
        *s_rhs, *zeros_, *structure_field()->get_dbc_map_extractor()->cond_map());
    locsysman_->rotate_local_to_global(*s_rhs);

    Core::LinAlg::apply_dirichlet_to_system(
        *t_rhs, *zeros_, *thermo_field()->get_dbc_map_extractor()->cond_map());

    extractor()->insert_vector(*s_rhs, 0, *rhs_);
    extractor()->insert_vector(*t_rhs, 1, *rhs_);
  }
  else
  {
    Core::LinAlg::apply_dirichlet_to_system(
        *rhs_, *zeros_, *structure_field()->get_dbc_map_extractor()->cond_map());
    Core::LinAlg::apply_dirichlet_to_system(
        *rhs_, *zeros_, *thermo_field()->get_dbc_map_extractor()->cond_map());
  }
}

/*----------------------------------------------------------------------*
 |                                                                      |
 *----------------------------------------------------------------------*/
bool TSI::Monolithic::l_sadmissible()
{
  switch (ls_strategy_)
  {
    case TSI::LS_structure:
      return normstrrhs_ < last_iter_res_.first;
    case TSI::LS_thermo:
      return normthrrhs_ < last_iter_res_.second;
    case TSI::LS_or:
      return (normstrrhs_ < last_iter_res_.first || normthrrhs_ < last_iter_res_.second);
    case TSI::LS_and:
      return (normstrrhs_ < last_iter_res_.first && normthrrhs_ < last_iter_res_.second);
    case TSI::LS_none:
    default:
      FOUR_C_THROW("you should not be here");
      return false;
  }
}

FOUR_C_NAMESPACE_CLOSE

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_timint_elch_scl.hpp"

#include "4C_adapter_scatra_base_algorithm.hpp"
#include "4C_comm_mpi_utils.hpp"
#include "4C_comm_utils_gid_vector.hpp"
#include "4C_coupling_adapter.hpp"
#include "4C_coupling_adapter_converter.hpp"
#include "4C_fem_dofset_predefineddofnumber.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_equilibrate.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_linear_solver_method_parameters.hpp"
#include "4C_scatra_ele_action.hpp"
#include "4C_scatra_resulttest_elch.hpp"
#include "4C_scatra_timint_elch_service.hpp"
#include "4C_scatra_timint_meshtying_strategy_s2i_elch.hpp"
#include "4C_utils_parameter_list.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
ScaTra::ScaTraTimIntElchSCL::ScaTraTimIntElchSCL(std::shared_ptr<Core::FE::Discretization> dis,
    std::shared_ptr<Core::LinAlg::Solver> solver, std::shared_ptr<Teuchos::ParameterList> params,
    std::shared_ptr<Teuchos::ParameterList> sctratimintparams,
    std::shared_ptr<Teuchos::ParameterList> extraparams,
    std::shared_ptr<Core::IO::DiscretizationWriter> output)
    : ScaTraTimIntElch(dis, solver, params, sctratimintparams, extraparams, output),
      matrixtype_elch_scl_(
          Teuchos::getIntegralValue<Core::LinAlg::MatrixType>(params->sublist("SCL"), "MATRIXTYPE"))
{
  if (matrixtype_elch_scl_ != Core::LinAlg::MatrixType::sparse and
      matrixtype_elch_scl_ != Core::LinAlg::MatrixType::block_field)
    FOUR_C_THROW("Only sparse and block field matrices supported in SCL computations");

  if (elchparams_->get<bool>("INITPOTCALC"))
  {
    FOUR_C_THROW(
        "Must disable INITPOTCALC for a coupled SCL problem. Use INITPOTCALC in the SCL section "
        "instead.");
  }
  if (!params_->get<bool>("SKIPINITDER"))
  {
    FOUR_C_THROW(
        "Must enable SKIPINITDER. Currently, Neumann BCs are not supported in the SCL formulation "
        "and thus, the calculation of the initial time derivative is meaningless.");
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntElchSCL::setup()
{
  TEUCHOS_FUNC_TIME_MONITOR("SCL: setup");

  ScaTra::ScaTraTimIntElch::setup();

  auto* problem = Global::Problem::instance();

  auto sdyn_micro =
      std::make_shared<Teuchos::ParameterList>(problem->scalar_transport_dynamic_params());

  auto initial_field_type = Teuchos::getIntegralValue<Inpar::ScaTra::InitialField>(
      elchparams_->sublist("SCL"), "INITIALFIELD");
  if (!(initial_field_type == Inpar::ScaTra::initfield_zero_field ||
          initial_field_type == Inpar::ScaTra::initfield_field_by_function ||
          initial_field_type == Inpar::ScaTra::initfield_field_by_condition))
    FOUR_C_THROW("input type not supported");

  sdyn_micro->set("INITIALFIELD", initial_field_type);
  sdyn_micro->set("INITFUNCNO", elchparams_->sublist("SCL").get<int>("INITFUNCNO"));

  micro_timint_ = std::make_shared<Adapter::ScaTraBaseAlgorithm>(*sdyn_micro, *sdyn_micro,
      problem->solver_params(sdyn_micro->get<int>("LINEAR_SOLVER")), "scatra_micro", false);

  micro_timint_->init();

  auto dofset_vel = std::make_shared<Core::DOFSets::DofSetPredefinedDoFNumber>(3, 0, 0, true);
  if (micro_timint_->scatra_field()->discretization()->add_dof_set(dofset_vel) != 1)
    FOUR_C_THROW("unexpected number of dofsets in the scatra micro discretization");
  micro_scatra_field()->set_number_of_dof_set_velocity(1);

  micro_scatra_field()->discretization()->fill_complete();

  redistribute_micro_discretization();

  micro_scatra_field()->set_velocity_field_from_function();

  micro_timint_->setup();

  // setup coupling between macro and micro field
  setup_coupling();

  // setup maps for coupled problem
  full_map_elch_scl_ = Core::LinAlg::merge_map(dof_row_map(), micro_scatra_field()->dof_row_map());
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> block_map_vec_scl;
  switch (matrixtype_elch_scl_)
  {
    case Core::LinAlg::MatrixType::sparse:
      block_map_vec_scl = {full_map_elch_scl_};
      break;
    case Core::LinAlg::MatrixType::block_field:
      block_map_vec_scl = {dof_row_map(), micro_scatra_field()->dof_row_map()};
      break;
    default:
      FOUR_C_THROW("Matrix type not supported.");
      break;
  }
  full_block_map_elch_scl_ =
      std::make_shared<Core::LinAlg::MultiMapExtractor>(*full_map_elch_scl_, block_map_vec_scl);

  // setup matrix, rhs, and increment for coupled problem
  increment_elch_scl_ = Core::LinAlg::create_vector(*full_map_elch_scl_, true);
  residual_elch_scl_ = Core::LinAlg::create_vector(*full_map_elch_scl_, true);


  switch (matrixtype_elch_scl_)
  {
    case Core::LinAlg::MatrixType::sparse:
    {
      const int expected_entries_per_row = 27;
      const bool explicitdirichlet = false;
      const bool savegraph = true;
      system_matrix_elch_scl_ = std::make_shared<Core::LinAlg::SparseMatrix>(
          *full_map_elch_scl_, expected_entries_per_row, explicitdirichlet, savegraph);
      break;
    }
    case Core::LinAlg::MatrixType::block_field:
    {
      const int expected_entries_per_row = 81;
      const bool explicitdirichlet = false;
      const bool savegraph = true;
      system_matrix_elch_scl_ = std::make_shared<
          Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(

          *full_block_map_elch_scl_, *full_block_map_elch_scl_, expected_entries_per_row,
          explicitdirichlet, savegraph);
      break;
    }
    default:
      FOUR_C_THROW("Matrix type not supported.");
      break;
  }

  // extractor to get micro or macro dofs from global vector
  macro_micro_dofs_ = std::make_shared<Core::LinAlg::MapExtractor>(
      *full_map_elch_scl_, micro_scatra_field()->dof_row_map());

  dbcmaps_elch_scl_ =
      std::make_shared<Core::LinAlg::MapExtractor>(*full_map_elch_scl_, dbcmaps_->cond_map());

  // setup solver for coupled problem
  solver_elch_scl_ = std::make_shared<Core::LinAlg::Solver>(
      problem->solver_params(elchparams_->sublist("SCL").get<int>("SOLVER")), discret_->get_comm(),
      Global::Problem::instance()->solver_params_callback(),
      Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
          Global::Problem::instance()->io_params(), "VERBOSITY"));

  switch (matrixtype_elch_scl_)
  {
    case Core::LinAlg::MatrixType::sparse:
      break;
    case Core::LinAlg::MatrixType::block_field:
    {
      std::ostringstream scatrablockstr;
      scatrablockstr << 1;
      Teuchos::ParameterList& blocksmootherparamsscatra =
          solver_elch_scl_->params().sublist("Inverse" + scatrablockstr.str());

      Core::LinearSolver::Parameters::compute_solver_parameters(
          *discretization(), blocksmootherparamsscatra);

      std::ostringstream microblockstr;
      microblockstr << 2;
      Teuchos::ParameterList& blocksmootherparamsmicro =
          solver_elch_scl_->params().sublist("Inverse" + microblockstr.str());

      Core::LinearSolver::Parameters::compute_solver_parameters(
          *micro_scatra_field()->discretization(), blocksmootherparamsmicro);

      break;
    }
    default:
      FOUR_C_THROW("not supported");
  }
}

/*------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntElchSCL::prepare_time_step()
{
  if (elchparams_->sublist("SCL").get<int>("ADAPT_TIME_STEP") == step() + 1)
  {
    const double new_dt = elchparams_->sublist("SCL").get<double>("ADAPTED_TIME_STEP_SIZE");
    if (new_dt <= 0) FOUR_C_THROW("new time step size for SCL must be positive.");

    set_dt(new_dt);
    set_time_step(time(), step());

    micro_scatra_field()->set_dt(new_dt);
    micro_scatra_field()->set_time_step(time(), step());
    if (Core::Communication::my_mpi_rank(discret_->get_comm()) == 0)
      std::cout << "Time step size changed to " << new_dt << std::endl;
  }

  ScaTraTimIntElch::prepare_time_step();

  copy_solution_to_micro_field();
  micro_scatra_field()->prepare_time_step();
  copy_solution_to_micro_field();
}

/*------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntElchSCL::update()
{
  ScaTraTimIntElch::update();

  micro_scatra_field()->update();
}

/*------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntElchSCL::check_and_write_output_and_restart()
{
  ScaTraTimIntElch::check_and_write_output_and_restart();

  micro_scatra_field()->check_and_write_output_and_restart();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntElchSCL::nonlinear_solve()
{
  // safety checks
  check_is_init();
  check_is_setup();

  // time measurement: nonlinear iteration
  TEUCHOS_FUNC_TIME_MONITOR("SCATRA:   + nonlin. iteration/lin. solve");

  // out to screen
  print_time_step_info();

  // prepare Newton-Raphson iteration
  iternum_ = 0;

  copy_solution_to_micro_field();

  auto equilibration_method = std::vector<Core::LinAlg::EquilibrationMethod>(
      1, ScaTraTimIntElchSCL::equilibration_method());
  auto equilibration = Core::LinAlg::build_equilibration(
      matrixtype_elch_scl_, equilibration_method, full_map_elch_scl_);

  // start Newton-Raphson iteration
  while (true)
  {
    iternum_++;

    // prepare load vector
    neumann_loads_->put_scalar(0.0);

    {
      TEUCHOS_FUNC_TIME_MONITOR("SCL: evaluate");
      // assemble sub problems
      assemble_mat_and_rhs();
      micro_scatra_field()->assemble_mat_and_rhs();

      // scale micro problem to account for related macro area
      scale_micro_problem();

      // couple micro and macro field my nodal mesh tying
      assemble_and_apply_mesh_tying();

      system_matrix_elch_scl_->complete();

      // All DBCs are on the macro scale
      Core::LinAlg::apply_dirichlet_to_system(*system_matrix_elch_scl_, *increment_elch_scl_,
          *residual_elch_scl_, *zeros_, *dbcmaps_elch_scl_->cond_map());

      if (break_newton_loop_and_print_convergence()) break;
    }

    increment_elch_scl_->put_scalar(0.0);

    {
      TEUCHOS_FUNC_TIME_MONITOR("SCL: solve");

      equilibration->equilibrate_system(
          system_matrix_elch_scl_, residual_elch_scl_, full_block_map_elch_scl_);
      Core::LinAlg::SolverParams solver_params;
      solver_params.refactor = true;
      solver_params.reset = iternum_ == 1;
      solver_elch_scl_->solve(
          system_matrix_elch_scl_, increment_elch_scl_, residual_elch_scl_, solver_params);
      equilibration->unequilibrate_increment(increment_elch_scl_);
    }

    {
      TEUCHOS_FUNC_TIME_MONITOR("SCL: update");

      update_iter_micro_macro();

      //-------- update values at intermediate time steps (only for gen.-alpha)
      compute_intermediate_values();
      micro_scatra_field()->compute_intermediate_values();
      // compute values at the interior of the elements (required for hdg)
      compute_interior_values();
      micro_scatra_field()->compute_interior_values();

      compute_time_derivative();
      micro_scatra_field()->compute_time_derivative();
    }

  }  // nonlinear iteration
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntElchSCL::add_problem_specific_parameters_and_vectors(
    Teuchos::ParameterList& params)
{
  ScaTra::ScaTraTimIntElch::add_problem_specific_parameters_and_vectors(params);

  discret_->set_state("phinp", *phinp());
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntElchSCL::copy_solution_to_micro_field()
{
  // extract coupled values from macro, copy to micro, and insert into full micro vector
  auto macro_to_micro_coupled_nodes = macro_micro_coupling_adapter_->master_to_slave(
      *macro_coupling_dofs_->extract_cond_vector(*phinp()));
  micro_coupling_dofs_->insert_cond_vector(
      *macro_to_micro_coupled_nodes, *micro_scatra_field()->phinp());
}

/*----------------------------------------------------------------------------------------*
 *----------------------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntElchSCL::create_meshtying_strategy()
{
  strategy_ = std::make_shared<MeshtyingStrategyS2IElchSCL>(this, *params_);
}

/*----------------------------------------------------------------------------------------*
 *----------------------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntElchSCL::read_restart_problem_specific(
    int step, Core::IO::DiscretizationReader& reader)
{
  FOUR_C_THROW("Restart is not implemented for Elch with SCL.");
}

/*----------------------------------------------------------------------------------------*
 *----------------------------------------------------------------------------------------*/
std::shared_ptr<ScaTra::ScaTraTimIntImpl> ScaTra::ScaTraTimIntElchSCL::micro_scatra_field()
{
  return micro_timint_->scatra_field();
}

/*----------------------------------------------------------------------------------------*
 *----------------------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntElchSCL::write_coupling_to_csv(
    const std::map<int, int>& glob_micro_macro_coupled_node_gids,
    const std::map<int, int>& glob_macro_slave_node_master_node_gids)
{
  std::ofstream file;

  // write GID of coupled nodes to .csv file
  if (myrank_ == 0)
  {
    const std::string file_name_coupling =
        problem_->output_control_file()->file_name() + "_micro_macro_coupling.csv";

    file.open(file_name_coupling, std::fstream::trunc);
    file << "macro_slave_node_gid,macro_master_node_gid,micro_slave_node_gid,micro_master_"
            "node_"
            "gid\n";
    file.flush();
    file.close();

    for (const auto& glob_macro_slave_node_master_node_gid : glob_macro_slave_node_master_node_gids)
    {
      const int macro_slave_node_gid = glob_macro_slave_node_master_node_gid.first;
      const int macro_master_node_gid = glob_macro_slave_node_master_node_gid.second;
      const int micro_slave_node_gid =
          glob_micro_macro_coupled_node_gids.find(macro_slave_node_gid)->second;
      const int micro_master_node_gid =
          glob_micro_macro_coupled_node_gids.find(macro_master_node_gid)->second;

      file.open(file_name_coupling, std::fstream::app);
      file << macro_slave_node_gid << "," << macro_master_node_gid << "," << micro_slave_node_gid
           << "," << micro_master_node_gid << "\n";
      file.flush();
      file.close();
    }
  }

  // write node coordinates to .csv file
  const std::string file_name_coords =
      problem_->output_control_file()->file_name() + "_micro_macro_coupling_coords.csv";

  if (myrank_ == 0)
  {
    file.open(file_name_coords, std::fstream::trunc);
    file << "node_GID,x,y,z \n";
    file.flush();
    file.close();
  }

  // node coordinates only known by owning proc. Writing of data to file not possible by
  // multiple procs in parallel
  for (int iproc = 0; iproc < Core::Communication::num_mpi_ranks(discret_->get_comm()); ++iproc)
  {
    if (iproc == myrank_)
    {
      for (const auto& glob_micro_macro_coupled_node_gid : glob_micro_macro_coupled_node_gids)
      {
        const int macro_node_gid = glob_micro_macro_coupled_node_gid.first;
        const int mirco_node_gid = glob_micro_macro_coupled_node_gid.second;

        if (Core::Communication::is_node_gid_on_this_proc(*discret_, macro_node_gid))
        {
          const auto& macro_coords = discret_->g_node(macro_node_gid)->x();

          file.open(file_name_coords, std::fstream::app);
          file << std::setprecision(16) << std::scientific;
          file << macro_node_gid << "," << macro_coords[0] << "," << macro_coords[1] << ","
               << macro_coords[2] << "\n";
          file.flush();
          file.close();
        }

        if (Core::Communication::is_node_gid_on_this_proc(
                *micro_scatra_field()->discretization(), mirco_node_gid))
        {
          const auto& micro_coords =
              micro_scatra_field()->discretization()->g_node(mirco_node_gid)->x();

          file.open(file_name_coords, std::fstream::app);
          file << std::setprecision(16) << std::scientific;
          file << mirco_node_gid << "," << micro_coords[0] << "," << micro_coords[1] << ","
               << micro_coords[2] << "\n";
          file.flush();
          file.close();
        }
      }
    }
    Core::Communication::barrier(discret_->get_comm());
  }
}

/*----------------------------------------------------------------------------------------*
 *----------------------------------------------------------------------------------------*/
bool ScaTra::ScaTraTimIntElchSCL::break_newton_loop_and_print_convergence()
{
  // extract processor ID
  const int mypid = Core::Communication::my_mpi_rank(discret_->get_comm());

  const auto& params =
      Global::Problem::instance()->scalar_transport_dynamic_params().sublist("NONLINEAR");

  const int itermax = params.get<int>("ITEMAX");
  const double itertol = params.get<double>("CONVTOL");

  auto micro_residual = macro_micro_dofs_->extract_cond_vector(*residual_elch_scl_);
  auto macro_residual = macro_micro_dofs_->extract_other_vector(*residual_elch_scl_);
  auto micro_increment = macro_micro_dofs_->extract_cond_vector(*increment_elch_scl_);
  auto macro_increment = macro_micro_dofs_->extract_other_vector(*increment_elch_scl_);

  double residual_L2, micro_residual_L2, macro_residual_L2, increment_L2, micro_increment_L2,
      macro_increment_L2, micro_state_L2, macro_state_L2;
  residual_elch_scl_->norm_2(&residual_L2);
  micro_residual->norm_2(&micro_residual_L2);
  macro_residual->norm_2(&macro_residual_L2);
  increment_elch_scl_->norm_2(&increment_L2);
  micro_increment->norm_2(&micro_increment_L2);
  macro_increment->norm_2(&macro_increment_L2);
  micro_scatra_field()->phinp()->norm_2(&micro_state_L2);
  phinp()->norm_2(&macro_state_L2);

  // safety checks
  if (std::isnan(residual_L2) or std::isnan(micro_residual_L2) or std::isnan(macro_residual_L2) or
      std::isnan(increment_L2) or std::isnan(micro_increment_L2) or
      std::isnan(macro_increment_L2) or std::isnan(micro_state_L2) or std::isnan(macro_state_L2))
    FOUR_C_THROW("Calculated vector norm is not a number!");
  if (std::isinf(residual_L2) or std::isinf(micro_residual_L2) or std::isinf(macro_residual_L2) or
      std::isinf(increment_L2) or std::isinf(micro_increment_L2) or
      std::isinf(macro_increment_L2) or std::isinf(micro_state_L2) or std::isinf(macro_state_L2))
    FOUR_C_THROW("Calculated vector norm is infinity!");

  micro_state_L2 = micro_state_L2 < 1.0e-10 ? 1.0 : micro_state_L2;
  macro_state_L2 = macro_state_L2 < 1.0e-10 ? 1.0 : micro_state_L2;

  const double state_L2 = std::sqrt(std::pow(micro_state_L2, 2) + std::pow(macro_state_L2, 2));

  micro_increment_L2 = micro_increment_L2 / micro_state_L2;
  macro_increment_L2 = macro_increment_L2 / macro_state_L2;
  increment_L2 = increment_L2 / state_L2;

  const bool finished =
      (residual_L2 < itertol and micro_residual_L2 < itertol and macro_residual_L2 < itertol and
          increment_L2 < itertol and micro_increment_L2 < itertol and
          macro_increment_L2 < itertol and iternum_ > 1) or
      iternum_ == itermax;

  // special case: very first iteration step --> solution increment is not yet available
  if (mypid == 0)
  {
    if (iternum_ == 1)
    {
      // print header of convergence table to screen
      std::cout << "+------------+-------------------+-------------+-------------+-------------+---"
                   "----------+-------------+-------------+"
                << std::endl;
      std::cout << "|- step/max -|- tol      [norm] -|---  res  ---|---  inc  ---|- micro-res -|- "
                   "micro-inc -|- macro-res -|- macro-inc -| "
                << std::endl;

      // print first line of convergence table to screen
      std::cout << "|  " << std::setw(3) << iternum_ << "/" << std::setw(3) << itermax << "   | "
                << std::setw(10) << std::setprecision(3) << std::scientific << itertol
                << "[L_2 ]  | " << std::setw(10) << std::setprecision(3) << std::scientific
                << residual_L2 << "  |     --      | " << std::setw(10) << std::setprecision(3)
                << std::scientific << micro_residual_L2 << "  |     --      | " << std::setw(10)
                << std::setprecision(3) << std::scientific << macro_residual_L2
                << "  |     --      |  " << std::endl;
    }
    else
    {
      // print current line of convergence table to screen
      std::cout << "|  " << std::setw(3) << iternum_ << "/" << std::setw(3) << itermax << "   | "
                << std::setw(10) << std::setprecision(3) << std::scientific << itertol
                << "[L_2 ]  | " << std::setw(10) << std::setprecision(3) << std::scientific
                << residual_L2 << "  | " << std::setw(10) << std::setprecision(3) << std::scientific
                << increment_L2 << "  | " << std::setw(10) << std::setprecision(3)
                << std::scientific << micro_residual_L2 << "  | " << std::setw(10)
                << std::setprecision(3) << std::scientific << micro_increment_L2 << "  | "
                << std::setw(10) << std::setprecision(3) << std::scientific << macro_residual_L2
                << "  | " << std::setw(10) << std::setprecision(3) << std::scientific
                << macro_increment_L2 << "  | " << std::endl;


      // convergence check
      if (finished)
      {
        // print finish line of convergence table to screen
        std::cout
            << "+------------+-------------------+-------------+-------------+-------------+---"
               "----------+-------------+-------------+"
            << std::endl;
        if (iternum_ == itermax)
        {
          std::cout << "|      >> Newton-Raphson iteration did not converge! <<                    "
                       "                                          |"
                    << std::endl;
          std::cout
              << "+------------+-------------------+-------------+-------------+-------------+---"
                 "----------+-------------+-------------+"
              << std::endl;
        }
      }
    }
  }
  return finished;
}

/*----------------------------------------------------------------------------------------*
 *----------------------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntElchSCL::setup_coupling()
{
  TEUCHOS_FUNC_TIME_MONITOR("SCL: setup");

  auto microdis = micro_scatra_field()->discretization();
  const auto& comm = microdis->get_comm();

  // get coupling conditions
  std::vector<const Core::Conditions::Condition*> macro_coupling_conditions;
  discretization()->get_condition("S2ISCLCoupling", macro_coupling_conditions);

  // get all slave and master nodes on this proc from macro coupling condition
  std::vector<int> my_macro_slave_node_gids;
  std::vector<int> my_macro_master_node_gids;
  for (auto* coupling_condition : macro_coupling_conditions)
  {
    for (const int coupling_node_gid : *coupling_condition->get_nodes())
    {
      // is this node owned by this proc?
      if (!Core::Communication::is_node_gid_on_this_proc(*discret_, coupling_node_gid)) continue;

      switch (coupling_condition->parameters().get<Inpar::S2I::InterfaceSides>("INTERFACE_SIDE"))
      {
        case Inpar::S2I::side_slave:
          my_macro_slave_node_gids.emplace_back(coupling_node_gid);
          break;
        case Inpar::S2I::side_master:
          my_macro_master_node_gids.emplace_back(coupling_node_gid);
          break;
        default:
          FOUR_C_THROW("must be master or slave side");
          break;
      }
    }
  }

  // get master dof(!!) (any proc) to slave nodes(!!) (this proc) from macro coupling adapter
  auto macro_coupling_adapter =
      std::dynamic_pointer_cast<const ScaTra::MeshtyingStrategyS2I>(strategy())->coupling_adapter();

  std::map<int, int> my_macro_slave_node_master_dof_gids;
  for (auto my_macro_slave_node_gid : my_macro_slave_node_gids)
  {
    auto* macro_slave_node = discret_->g_node(my_macro_slave_node_gid);
    auto fist_macro_slave_dof_gid = discret_->dof(0, macro_slave_node)[0];

    for (int slave_dof_lid = 0;
        slave_dof_lid < macro_coupling_adapter->slave_dof_map()->num_my_elements(); ++slave_dof_lid)
    {
      const int slave_dof_gid = macro_coupling_adapter->slave_dof_map()->gid(slave_dof_lid);
      if (fist_macro_slave_dof_gid == slave_dof_gid)
      {
        const int first_macro_master_dof_gid =
            macro_coupling_adapter->perm_master_dof_map()->gid(slave_dof_lid);
        my_macro_slave_node_master_dof_gids.insert(
            std::make_pair(my_macro_slave_node_gid, first_macro_master_dof_gid));
        break;
      }
    }
  }
  // distribute all maps to all procs
  const auto glob_macro_slave_node_master_dof_gids =
      Core::Communication::all_reduce(my_macro_slave_node_master_dof_gids, comm);

  // get master node (this proc) to slave node (any proc)
  std::map<int, int> my_macro_slave_node_master_node_gids;
  for (const auto& glob_macro_slave_node_master_dof_gid : glob_macro_slave_node_master_dof_gids)
  {
    const int master_dof_gid = glob_macro_slave_node_master_dof_gid.second;
    const int slave_node_gid = glob_macro_slave_node_master_dof_gid.first;
    if (dof_row_map()->lid(master_dof_gid) == -1) continue;

    for (const auto my_macro_master_node_gid : my_macro_master_node_gids)
    {
      auto* macro_master_node = discret_->g_node(my_macro_master_node_gid);
      if (discret_->dof(0, macro_master_node)[0] == master_dof_gid)
      {
        my_macro_slave_node_master_node_gids.insert(
            std::make_pair(slave_node_gid, my_macro_master_node_gid));
        break;
      }
    }
  }
  // distribute all maps to all procs
  const auto glob_macro_slave_node_master_node_gids =
      Core::Communication::all_reduce(my_macro_slave_node_master_node_gids, comm);

  // we use Dirchlet conditions on micro side to achieve coupling by adapting the DBC value
  std::vector<const Core::Conditions::Condition*> micro_coupling_conditions;
  microdis->get_condition("Dirichlet", micro_coupling_conditions);

  if (micro_coupling_conditions.size() != 2) FOUR_C_THROW("only 2 DBCs allowed on micro dis");
  if (micro_coupling_conditions[0]->get_nodes()->size() !=
      micro_coupling_conditions[1]->get_nodes()->size())
    FOUR_C_THROW("Number of nodes in micro DBCs are not equal");

  // get all micro coupling nodes
  std::vector<int> my_micro_node_gids;
  for (auto* micro_coupling_condition : micro_coupling_conditions)
  {
    for (const int micro_node_gid : *micro_coupling_condition->get_nodes())
    {
      // is this node owned by this proc?
      if (Core::Communication::is_node_gid_on_this_proc(*microdis, micro_node_gid))
        my_micro_node_gids.emplace_back(micro_node_gid);
    }
  }

  // setup coupling between macro and micro problems: find micro problems for this proc (end of last
  // proc)
  int micro_problem_counter = 0;
  int my_micro_problem_counter = 0;
  const unsigned int num_my_macro_slave_node_gids = my_macro_slave_node_gids.size();
  for (int iproc = 0; iproc < Core::Communication::num_mpi_ranks(comm); ++iproc)
  {
    if (iproc == Core::Communication::my_mpi_rank(comm))
      micro_problem_counter += static_cast<int>(num_my_macro_slave_node_gids);
    Core::Communication::broadcast(&micro_problem_counter, 1, iproc, comm);

    // start of micro discretization of this proc is end of last proc
    if (iproc == Core::Communication::my_mpi_rank(comm) - 1)
      my_micro_problem_counter = micro_problem_counter;
  }

  // global  map between coupled macro nodes and micro nodes
  std::map<int, int> my_macro_micro_coupled_node_gids;
  for (unsigned i = 0; i < num_my_macro_slave_node_gids; ++i)
  {
    const int macro_slave_gid = my_macro_slave_node_gids[i];
    const int macro_master_gid = glob_macro_slave_node_master_node_gids.at(macro_slave_gid);
    const int micro_slave_gid =
        micro_coupling_conditions[0]->get_nodes()->at(my_micro_problem_counter);
    const int micro_master_gid =
        micro_coupling_conditions[1]->get_nodes()->at(my_micro_problem_counter);

    my_macro_micro_coupled_node_gids.insert(std::make_pair(macro_slave_gid, micro_slave_gid));
    my_macro_micro_coupled_node_gids.insert(std::make_pair(macro_master_gid, micro_master_gid));
    my_micro_problem_counter++;
  }
  const auto glob_macro_micro_coupled_node_gids =
      Core::Communication::all_reduce(my_macro_micro_coupled_node_gids, comm);

  // setup macro nodes on this proc and coupled micro nodes (can be other proc)
  std::vector<int> my_micro_permuted_node_gids;
  std::vector<int> my_macro_node_gids;
  for (const auto& glob_macro_micro_coupled_node_gid : glob_macro_micro_coupled_node_gids)
  {
    const int macro_node_gid = glob_macro_micro_coupled_node_gid.first;
    const int mirco_node_gid = glob_macro_micro_coupled_node_gid.second;

    if (!Core::Communication::is_node_gid_on_this_proc(*discret_, macro_node_gid)) continue;

    my_macro_node_gids.emplace_back(macro_node_gid);
    my_micro_permuted_node_gids.emplace_back(mirco_node_gid);
  }

  if (elchparams_->sublist("SCL").get<bool>("COUPLING_OUTPUT"))
    write_coupling_to_csv(
        glob_macro_micro_coupled_node_gids, glob_macro_slave_node_master_node_gids);

  // setup maps for coupled nodes
  Core::LinAlg::Map master_node_map(
      -1, static_cast<int>(my_macro_node_gids.size()), my_macro_node_gids.data(), 0, comm);
  Core::LinAlg::Map slave_node_map(
      -1, static_cast<int>(my_micro_node_gids.size()), my_micro_node_gids.data(), 0, comm);
  Core::LinAlg::Map perm_slave_node_map(-1, static_cast<int>(my_micro_permuted_node_gids.size()),
      my_micro_permuted_node_gids.data(), 0, comm);

  // setup coupling adapter between micro (slave) and macro (master) for all dof of the nodes
  Coupling::Adapter::Coupling macro_micro_coupling_adapter_temp;
  macro_micro_coupling_adapter_temp.setup_coupling(*discret_, *microdis, master_node_map,
      slave_node_map, perm_slave_node_map, num_dof_per_node());

  // setup actual coupling adapter only for dofs for which coupling is activated
  std::vector<int> my_slave_dofs;
  std::vector<int> my_perm_master_dofs;
  for (int slave_lid = 0;
      slave_lid < macro_micro_coupling_adapter_temp.slave_dof_map()->num_my_elements(); ++slave_lid)
  {
    const int slave_gid = macro_micro_coupling_adapter_temp.slave_dof_map()->gid(slave_lid);

    for (int dbc_lid = 0;
        dbc_lid < micro_scatra_field()->dirich_maps()->cond_map()->num_my_elements(); ++dbc_lid)
    {
      const int dbc_gid = micro_scatra_field()->dirich_maps()->cond_map()->gid(dbc_lid);
      if (slave_gid == dbc_gid)
      {
        my_slave_dofs.emplace_back(slave_gid);
        my_perm_master_dofs.emplace_back(
            macro_micro_coupling_adapter_temp.perm_master_dof_map()->gid(slave_lid));
        break;
      }
    }
  }

  const std::vector<int> glob_slave_dofs = Core::Communication::all_reduce(my_slave_dofs, comm);

  std::vector<int> my_master_dofs;
  std::vector<int> my_perm_slave_dofs;
  for (int master_lid = 0;
      master_lid < macro_micro_coupling_adapter_temp.master_dof_map()->num_my_elements();
      ++master_lid)
  {
    const int slave_gid = macro_micro_coupling_adapter_temp.perm_slave_dof_map()->gid(master_lid);
    const int master_gid = macro_micro_coupling_adapter_temp.master_dof_map()->gid(master_lid);
    if (std::find(glob_slave_dofs.begin(), glob_slave_dofs.end(), slave_gid) !=
        glob_slave_dofs.end())
    {
      my_master_dofs.emplace_back(master_gid);
      my_perm_slave_dofs.emplace_back(slave_gid);
    }
  }

  auto slave_dof_map = std::make_shared<Core::LinAlg::Map>(
      -1, static_cast<int>(my_slave_dofs.size()), my_slave_dofs.data(), 0, comm);
  auto perm_slave_dof_map = std::make_shared<Core::LinAlg::Map>(
      -1, static_cast<int>(my_perm_slave_dofs.size()), my_perm_slave_dofs.data(), 0, comm);
  auto master_dof_map = std::make_shared<Core::LinAlg::Map>(
      -1, static_cast<int>(my_master_dofs.size()), my_master_dofs.data(), 0, comm);
  auto perm_master_dof_map = std::make_shared<Core::LinAlg::Map>(
      -1, static_cast<int>(my_perm_master_dofs.size()), my_perm_master_dofs.data(), 0, comm);


  macro_micro_coupling_adapter_ = std::make_shared<Coupling::Adapter::Coupling>();
  macro_micro_coupling_adapter_->setup_coupling(
      slave_dof_map, perm_slave_dof_map, master_dof_map, perm_master_dof_map);

  macro_coupling_dofs_ = std::make_shared<Core::LinAlg::MapExtractor>(
      *dof_row_map(), macro_micro_coupling_adapter_->master_dof_map());

  micro_coupling_dofs_ = std::make_shared<Core::LinAlg::MapExtractor>(
      *microdis->dof_row_map(), macro_micro_coupling_adapter_->slave_dof_map());

  // setup relation between first node of micro sub problem and following nodes. This is required
  // for scaling (see scale_micro_problem())
  std::set<int> my_micro_coupling_nodes;
  for (int lid_micro = 0;
      lid_micro < macro_micro_coupling_adapter_->slave_dof_map()->num_my_elements(); ++lid_micro)
  {
    const int gid_micro = macro_micro_coupling_adapter_->slave_dof_map()->gid(lid_micro);
    my_micro_coupling_nodes.insert(gid_micro);
  }

  const auto glob_micro_coupling_nodes =
      Core::Communication::all_reduce(my_micro_coupling_nodes, discret_->get_comm());

  // by definition, the last node of a micro sub problem is coupled with the macro. Here, all nodes
  // in the sub problem are linked to the coupled node, by looping backwards through all nodes and
  // forwards through the coupled nodes
  for (int lid_micro = micro_scatra_field()->dof_row_map()->num_my_elements() - 1; lid_micro >= 0;
      --lid_micro)
  {
    const int gid_micro = micro_scatra_field()->dof_row_map()->gid(lid_micro);
    for (const auto& coupled_node : glob_micro_coupling_nodes)
    {
      if (coupled_node >= gid_micro)
      {
        coupled_micro_nodes_.insert(std::make_pair(gid_micro, coupled_node));
        break;
      }
    }
  }
}

/*----------------------------------------------------------------------------------------*
 *----------------------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntElchSCL::scale_micro_problem()
{
  Teuchos::ParameterList condparams;

  // scale micro problem with nodal area of macro discretiaztion
  Core::Utils::add_enum_class_to_parameter_list<ScaTra::BoundaryAction>(
      "action", ScaTra::BoundaryAction::calc_nodal_size, condparams);

  auto nodal_size_macro = Core::LinAlg::create_vector(*dof_row_map(), true);
  discret_->evaluate_condition(
      condparams, nullptr, nullptr, nodal_size_macro, nullptr, nullptr, "S2ISCLCoupling");

  // extract dof values to node values
  for (int row_lid = 0; row_lid < dof_row_map()->num_my_elements(); row_lid += 2)
  {
    const double row_value = (*nodal_size_macro).get_values()[row_lid];
    const double scale_fac = row_value == 0.0 ? 1.0 : row_value;
    for (int dof = 0; dof < num_dof_per_node(); ++dof)
      (*nodal_size_macro).get_values()[row_lid + dof] = scale_fac;
  }

  // transform to micro discretization
  auto nodal_size_micro = macro_micro_coupling_adapter_->master_to_slave(
      *macro_coupling_dofs_->extract_cond_vector(*nodal_size_macro));

  // communicate nodal size to all procs to be able to scale all rows in micro discretization
  // attached to a macro node
  std::map<int, double> my_nodal_size_micro;
  for (int lid_micro = 0;
      lid_micro < macro_micro_coupling_adapter_->slave_dof_map()->num_my_elements(); ++lid_micro)
  {
    const int gid_micro = macro_micro_coupling_adapter_->slave_dof_map()->gid(lid_micro);
    my_nodal_size_micro.insert(
        std::make_pair(gid_micro, nodal_size_micro->local_values_as_span()[lid_micro]));
  }

  const auto glob_nodal_size_micro =
      Core::Communication::all_reduce(my_nodal_size_micro, discret_->get_comm());

  auto micro_scale = Core::LinAlg::create_vector(*micro_scatra_field()->dof_row_map(), true);
  for (int lid_micro = micro_scatra_field()->dof_row_map()->num_my_elements() - 1; lid_micro >= 0;
      --lid_micro)
  {
    const int gid_micro = micro_scatra_field()->dof_row_map()->gid(lid_micro);
    const int coupled_node = coupled_micro_nodes_[gid_micro];
    const double scale_val = glob_nodal_size_micro.at(coupled_node);
    micro_scale->get_values()[lid_micro] = scale_val;
    micro_scatra_field()->residual()->get_values()[lid_micro] *= scale_val;
  }
  micro_scatra_field()->system_matrix()->left_scale(*micro_scale);
}

/*----------------------------------------------------------------------------------------*
 *----------------------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntElchSCL::assemble_and_apply_mesh_tying()
{
  // Meshtying + Assembly RHS
  auto micro_residual =
      micro_coupling_dofs_->extract_cond_vector(*micro_scatra_field()->residual());
  auto micro_residual_on_macro_side =
      macro_micro_coupling_adapter_->slave_to_master(*micro_residual);

  auto full_macro_vector = Core::LinAlg::create_vector(*dof_row_map(), true);
  macro_coupling_dofs_->insert_cond_vector(*micro_residual_on_macro_side, *full_macro_vector);

  residual_elch_scl_->put_scalar(0.0);
  system_matrix_elch_scl_->zero();

  macro_micro_dofs_->add_other_vector(*full_macro_vector, *residual_elch_scl_);

  macro_micro_dofs_->add_other_vector(*residual(), *residual_elch_scl_);

  // apply pseudo DBC on slave side
  micro_coupling_dofs_->cond_put_scalar(*micro_scatra_field()->residual(), 0.0);

  macro_micro_dofs_->add_cond_vector(
      *micro_scatra_field()->residual(), *residual_elch_scl_);  // add slave side to total residual

  switch (matrixtype_elch_scl_)
  {
    case Core::LinAlg::MatrixType::sparse:
    {
      auto sparse_systemmatrix =
          Core::LinAlg::cast_to_sparse_matrix_and_check_success(system_matrix_elch_scl_);

      sparse_systemmatrix->add(*system_matrix(), false, 1.0, 1.0);

      Coupling::Adapter::CouplingSlaveConverter micro_side_converter(
          *macro_micro_coupling_adapter_);

      // micro: interior - interior
      Coupling::Adapter::MatrixLogicalSplitAndTransform()(*micro_scatra_field()->system_matrix(),
          *micro_coupling_dofs_->other_map(), *micro_coupling_dofs_->other_map(), 1.0, nullptr,
          nullptr, *sparse_systemmatrix, true, true);

      // micro: interior - slave
      Coupling::Adapter::MatrixLogicalSplitAndTransform()(*micro_scatra_field()->system_matrix(),
          *micro_coupling_dofs_->other_map(), *micro_coupling_dofs_->cond_map(), 1.0, nullptr,
          &(micro_side_converter), *sparse_systemmatrix, true, true);

      // micro: slave - interior
      Coupling::Adapter::MatrixLogicalSplitAndTransform()(*micro_scatra_field()->system_matrix(),
          *micro_coupling_dofs_->cond_map(), *micro_coupling_dofs_->other_map(), 1.0,
          &(micro_side_converter), nullptr, *sparse_systemmatrix, true, true);

      // micro: slave - slave
      Coupling::Adapter::MatrixLogicalSplitAndTransform()(*micro_scatra_field()->system_matrix(),
          *micro_coupling_dofs_->cond_map(), *micro_coupling_dofs_->cond_map(), 1.0,
          &(micro_side_converter), &(micro_side_converter), *sparse_systemmatrix, true, true);
      break;
    }
    case Core::LinAlg::MatrixType::block_field:
    {
      auto block_systemmatrix =
          Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(system_matrix_elch_scl_);

      block_systemmatrix->matrix(0, 0).add(*system_matrix(), false, 1.0, 1.0);

      Coupling::Adapter::CouplingSlaveConverter micro_side_converter(
          *macro_micro_coupling_adapter_);

      // micro: interior - interior
      Coupling::Adapter::MatrixLogicalSplitAndTransform()(*micro_scatra_field()->system_matrix(),
          *micro_coupling_dofs_->other_map(), *micro_coupling_dofs_->other_map(), 1.0, nullptr,
          nullptr, block_systemmatrix->matrix(1, 1), true, true);

      // micro: interior - slave
      Coupling::Adapter::MatrixLogicalSplitAndTransform()(*micro_scatra_field()->system_matrix(),
          *micro_coupling_dofs_->other_map(), *micro_coupling_dofs_->cond_map(), 1.0, nullptr,
          &(micro_side_converter), block_systemmatrix->matrix(1, 0), true, true);

      // micro: slave - interior
      Coupling::Adapter::MatrixLogicalSplitAndTransform()(*micro_scatra_field()->system_matrix(),
          *micro_coupling_dofs_->cond_map(), *micro_coupling_dofs_->other_map(), 1.0,
          &(micro_side_converter), nullptr, block_systemmatrix->matrix(0, 1), true, true);

      // micro: slave - slave
      Coupling::Adapter::MatrixLogicalSplitAndTransform()(*micro_scatra_field()->system_matrix(),
          *micro_coupling_dofs_->cond_map(), *micro_coupling_dofs_->cond_map(), 1.0,
          &(micro_side_converter), &(micro_side_converter), block_systemmatrix->matrix(0, 0), true,
          true);
      break;
    }
    default:
      FOUR_C_THROW("not supported");
      break;
  }

  // pseudo DBCs on slave side
  Core::LinAlg::SparseMatrix& micromatrix =
      matrixtype_elch_scl_ == Core::LinAlg::MatrixType::sparse
          ? *Core::LinAlg::cast_to_sparse_matrix_and_check_success(system_matrix_elch_scl_)
          : Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(
                system_matrix_elch_scl_)
                ->matrix(1, 1);
  auto slavemaps = macro_micro_coupling_adapter_->slave_dof_map();
  const double one = 1.0;
  for (int doflid_slave = 0; doflid_slave < slavemaps->num_my_elements(); ++doflid_slave)
  {
    // extract global ID of current slave-side row
    const int dofgid_slave = slavemaps->gid(doflid_slave);
    if (dofgid_slave < 0) FOUR_C_THROW("Local ID not found!");

    // apply pseudo Dirichlet conditions to filled matrix, i.e., to local row and column indices
    if (micromatrix.filled())
    {
      const int rowlid_slave = micromatrix.row_map().lid(dofgid_slave);
      if (rowlid_slave < 0) FOUR_C_THROW("Global ID not found!");
      if (micromatrix.replace_my_values(rowlid_slave, 1, &one, &rowlid_slave))
        FOUR_C_THROW("ReplaceMyValues failed!");
    }

    // apply pseudo Dirichlet conditions to unfilled matrix, i.e., to global row and column
    // indices
    else
    {
      micromatrix.insert_global_values(dofgid_slave, 1, &one, &dofgid_slave);
    }
  }
}

/*----------------------------------------------------------------------------------------*
 *----------------------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntElchSCL::update_iter_micro_macro()
{
  auto increment_macro = macro_micro_dofs_->extract_other_vector(*increment_elch_scl_);
  auto increment_micro = macro_micro_dofs_->extract_cond_vector(*increment_elch_scl_);

  // reconstruct slave result from master side
  auto macro_extract = macro_coupling_dofs_->extract_cond_vector(*increment_macro);
  auto macro_extract_to_micro = macro_micro_coupling_adapter_->master_to_slave(*macro_extract);
  micro_coupling_dofs_->insert_cond_vector(*macro_extract_to_micro, *increment_micro);

  update_iter(*increment_macro);
  micro_scatra_field()->update_iter(*increment_micro);
}

/*----------------------------------------------------------------------------------------*
 *----------------------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntElchSCL::redistribute_micro_discretization()
{
  auto micro_dis = micro_scatra_field()->discretization();
  const int min_node_gid = micro_dis->node_row_map()->min_all_gid();
  const int num_nodes = micro_dis->node_row_map()->num_global_elements();
  const int num_proc = Core::Communication::num_mpi_ranks(micro_dis->get_comm());
  const int myPID = Core::Communication::my_mpi_rank(micro_dis->get_comm());

  const int num_node_per_proc = static_cast<int>(std::floor(num_nodes / num_proc));

  // new node row list: split node list by number of processors
  std::vector<int> my_row_nodes(num_node_per_proc, -1);
  if (myPID == num_proc - 1) my_row_nodes.resize(num_nodes - (num_proc - 1) * num_node_per_proc);
  std::iota(my_row_nodes.begin(), my_row_nodes.end(), min_node_gid + myPID * num_node_per_proc);

  // new node col list: add boundary nodes of other procs (first and last node of list)
  std::vector<int> my_col_nodes = my_row_nodes;
  if (myPID > 0) my_col_nodes.emplace_back(my_row_nodes[0] - 1);
  if (myPID < num_proc - 1) my_col_nodes.emplace_back(my_row_nodes.back() + 1);

  Core::LinAlg::Map new_node_row_map(num_nodes, static_cast<int>(my_row_nodes.size()),
      my_row_nodes.data(), 0, micro_dis->get_comm());

  Core::LinAlg::Map new_node_col_map(
      -1, static_cast<int>(my_col_nodes.size()), my_col_nodes.data(), 0, micro_dis->get_comm());

  micro_dis->redistribute({new_node_row_map, new_node_col_map});
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntElchSCL::prepare_time_loop()
{
  // call base class routine
  ScaTraTimIntElch::prepare_time_loop();

  if (elchparams_->sublist("SCL").get<bool>("INITPOTCALC")) calc_initial_potential_field();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntElchSCL::calc_initial_potential_field()
{
  pre_calc_initial_potential_field();
  std::dynamic_pointer_cast<ScaTra::ScaTraTimIntElch>(micro_scatra_field())
      ->pre_calc_initial_potential_field();

  // safety checks
  FOUR_C_ASSERT(step_ == 0, "Step counter is not zero!");

  if (equpot_ != ElCh::equpot_divi)
  {
    FOUR_C_THROW(
        "Initial potential field cannot be computed for chosen closing equation for electric "
        "potential!");
  }

  // screen output
  if (myrank_ == 0)
  {
    std::cout << "SCATRA: calculating initial field for electric potential" << std::endl;
    print_time_step_info();
    std::cout << "+------------+-------------------+--------------+--------------+" << std::endl;
    std::cout << "|- step/max -|- tol      [norm] -|--   res   ---|--   inc   ---|" << std::endl;
  }

  // prepare Newton-Raphson iteration
  iternum_ = 0;
  const int itermax = params_->sublist("NONLINEAR").get<int>("ITEMAX");
  const double itertol = params_->sublist("NONLINEAR").get<double>("CONVTOL");

  copy_solution_to_micro_field();

  // start Newton-Raphson iteration
  while (true)
  {
    // update iteration counter
    iternum_++;

    // prepare load vector
    neumann_loads_->put_scalar(0.0);

    // assemble sub problems
    assemble_mat_and_rhs();
    micro_scatra_field()->assemble_mat_and_rhs();

    // scale micro problem to account for related macro area
    scale_micro_problem();

    // couple micro and macro field my nodal mesh tying
    assemble_and_apply_mesh_tying();

    system_matrix_elch_scl_->complete();

    // All DBCs are on the macro scale
    Core::LinAlg::apply_dirichlet_to_system(*system_matrix_elch_scl_, *increment_elch_scl_,
        *residual_elch_scl_, *zeros_, *dbcmaps_elch_scl_->cond_map());

    // apply artificial Dirichlet boundary conditions to system of equations
    // to hold initial concentrations constant when solving for initial potential field
    auto pseudo_dbc_scl = Core::LinAlg::merge_map(
        splitter_->other_map(), micro_scatra_field()->splitter()->other_map());
    auto pseudo_zeros_scl = Core::LinAlg::create_vector(*pseudo_dbc_scl, true);

    Core::LinAlg::apply_dirichlet_to_system(*system_matrix_elch_scl_, *increment_elch_scl_,
        *residual_elch_scl_, *pseudo_zeros_scl, *pseudo_dbc_scl);

    // compute L2 norm of state vector
    double state_L2_macro, state_L2_micro;
    phinp()->norm_2(&state_L2_macro);
    micro_scatra_field()->phinp()->norm_2(&state_L2_micro);
    double state_L2 = std::sqrt(std::pow(state_L2_macro, 2) + std::pow(state_L2_micro, 2));

    // compute L2 residual vector
    double res_L2, inc_L2;
    residual_elch_scl_->norm_2(&res_L2);

    // compute L2 norm of increment vector
    increment_elch_scl_->norm_2(&inc_L2);

    // safety checks
    if (std::isnan(inc_L2) or std::isnan(res_L2)) FOUR_C_THROW("calculated vector norm is NaN.");
    if (std::isinf(inc_L2) or std::isinf(res_L2)) FOUR_C_THROW("calculated vector norm is INF.");

    // care for the case that nothing really happens
    if (state_L2 < 1.0e-5) state_L2 = 1.0;

    // first iteration step: solution increment is not yet available
    if (iternum_ == 1)
    {
      // print first line of convergence table to screen
      if (myrank_ == 0)
      {
        std::cout << "|  " << std::setw(3) << iternum_ << "/" << std::setw(3) << itermax << "   | "
                  << std::setw(10) << std::setprecision(3) << std::scientific << itertol
                  << "[L_2 ]  | " << std::setw(10) << std::setprecision(3) << std::scientific << 0.0
                  << "   |      --      | " << std::endl;
      }
    }

    // later iteration steps: solution increment can be printed
    else
    {
      // print current line of convergence table to screen
      if (myrank_ == 0)
      {
        std::cout << "|  " << std::setw(3) << iternum_ << "/" << std::setw(3) << itermax << "   | "
                  << std::setw(10) << std::setprecision(3) << std::scientific << itertol
                  << "[L_2 ]  | " << std::setw(10) << std::setprecision(3) << std::scientific
                  << res_L2 << "   | " << std::setw(10) << std::setprecision(3) << std::scientific
                  << inc_L2 / state_L2 << "   | " << std::endl;
      }

      // convergence check
      if (res_L2 <= itertol and inc_L2 / state_L2 <= itertol)
      {
        // print finish line of convergence table to screen
        if (myrank_ == 0)
        {
          std::cout << "+------------+-------------------+--------------+--------------+"
                    << std::endl
                    << std::endl;
        }

        // abort Newton-Raphson iteration
        break;
      }
    }

    // warn if maximum number of iterations is reached without convergence
    if (iternum_ == itermax)
    {
      if (myrank_ == 0)
      {
        std::cout << "+--------------------------------------------------------------+"
                  << std::endl;
        std::cout << "|            >>>>>> not converged!                             |"
                  << std::endl;
        std::cout << "+--------------------------------------------------------------+" << std::endl
                  << std::endl;
      }

      // abort Newton-Raphson iteration
      break;
    }

    // zero out increment vector
    increment_elch_scl_->put_scalar(0.0);

    Core::LinAlg::SolverParams solver_params;
    solver_params.refactor = true;
    solver_params.reset = iternum_ == 1;
    solver_elch_scl_->solve(
        system_matrix_elch_scl_, increment_elch_scl_, residual_elch_scl_, solver_params);

    update_iter_micro_macro();

    // copy initial state vector
    phin_->update(1., *phinp_, 0.);
    micro_scatra_field()->phin()->update(1.0, *micro_scatra_field()->phinp(), 0.0);

    // update state vectors for intermediate time steps (only for generalized alpha)
    compute_intermediate_values();
    micro_scatra_field()->compute_intermediate_values();
  }  // Newton-Raphson iteration

  // reset global system matrix and its graph, since we solved a very special problem with a
  // special sparsity pattern
  system_matrix_elch_scl_->reset();

  post_calc_initial_potential_field();

  std::dynamic_pointer_cast<ScaTra::ScaTraTimIntElch>(micro_scatra_field())
      ->post_calc_initial_potential_field();
}

std::shared_ptr<Core::Utils::ResultTest> ScaTra::ScaTraTimIntElchSCL::create_micro_field_test()
{
  return std::make_shared<ScaTra::ElchResultTest>(
      std::dynamic_pointer_cast<ScaTra::ScaTraTimIntElch>(micro_scatra_field()));
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntElchSCL::test_results()
{
  Global::Problem::instance()->add_field_test(create_scatra_field_test());
  Global::Problem::instance()->add_field_test(create_micro_field_test());
  Global::Problem::instance()->test_all(discret_->get_comm());
}

FOUR_C_NAMESPACE_CLOSE

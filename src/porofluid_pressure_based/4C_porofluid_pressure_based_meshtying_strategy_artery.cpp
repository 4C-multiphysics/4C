// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_porofluid_pressure_based_meshtying_strategy_artery.hpp"

#include "4C_art_net_input.hpp"
#include "4C_art_net_utils.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_utils_sparse_algebra_print.hpp"
#include "4C_linear_solver_method.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_linear_solver_method_parameters.hpp"
#include "4C_porofluid_pressure_based_algorithm.hpp"
#include "4C_porofluid_pressure_based_elast_scatra_artery_coupling_base.hpp"
#include "4C_porofluid_pressure_based_elast_scatra_utils.hpp"
#include "4C_porofluid_pressure_based_utils.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
PoroPressureBased::MeshtyingArtery::MeshtyingArtery(PorofluidAlgorithm* porofluid_algorithm,
    const Teuchos::ParameterList& problem_params, const Teuchos::ParameterList& porofluid_params)
    : porofluid_algorithm_(porofluid_algorithm),
      params_(problem_params),
      porofluid_params_(porofluid_params),
      vector_norm_res_(Teuchos::getIntegralValue<VectorNorm>(
          porofluid_params_.sublist("nonlinear_solver").sublist("residual"), "vector_norm")),
      vector_norm_inc_(Teuchos::getIntegralValue<VectorNorm>(
          porofluid_params_.sublist("nonlinear_solver").sublist("increment"), "vector_norm"))
{
  const Teuchos::ParameterList& artery_params =
      Global::Problem::instance()->arterial_dynamic_params();

  artery_dis_ = Global::Problem::instance()->get_dis("artery");

  if (!artery_dis_->filled()) artery_dis_->fill_complete();

  const auto time_integration_scheme =
      Teuchos::getIntegralValue<ArtDyn::TimeIntegrationScheme>(artery_params, "DYNAMICTYPE");

  std::shared_ptr<Core::IO::DiscretizationWriter> artery_output = artery_dis_->writer();
  artery_output->write_mesh(0, 0.0);

  // Translate updated porofluid input format to old artery format
  Teuchos::ParameterList artery_problem_params;
  artery_problem_params.set<int>(
      "RESTARTEVERY", problem_params.sublist("output").get<int>("restart_data_every"));
  artery_problem_params.set<int>(
      "RESULTSEVERY", problem_params.sublist("output").get<int>("result_data_every"));
  artery_problem_params.set<double>(
      "TIMESTEP", problem_params.sublist("time_integration").get<double>("time_step_size"));
  artery_problem_params.set<int>(
      "NUMSTEP", problem_params.sublist("time_integration").get<int>("number_of_time_steps"));

  // build artery network algorithm
  artery_algorithm_ = Arteries::Utils::create_algorithm(time_integration_scheme, artery_dis_,
      artery_params.get<int>("LINEAR_SOLVER"), artery_problem_params, artery_params,
      *artery_output);

  // set to false
  artery_algorithm_->set_solve_scatra(false);

  // initialize
  artery_algorithm_->init(problem_params, artery_params, "artery_scatra");

  // print user info
  if (Core::Communication::my_mpi_rank(porofluid_algorithm->discretization()->get_comm()) == 0)
  {
    std::cout << "\n";
    std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>" << '\n';
    std::cout << "<                                                  >" << '\n';
    std::cout << "<    Coupling with 1D Artery Network activated     >" << '\n';
  }

  const bool evaluate_on_lateral_surface =
      porofluid_params.sublist("artery_coupling").get<bool>("lateral_surface_coupling");

  const std::string coupling_condition_name = std::invoke(
      [&]()
      {
        if (Teuchos::getIntegralValue<ArteryNetwork::ArteryPorofluidElastScatraCouplingMethod>(
                Global::Problem::instance()->porofluid_pressure_based_dynamic_params().sublist(
                    "artery_coupling"),
                "coupling_method") ==
            ArteryNetwork::ArteryPorofluidElastScatraCouplingMethod::node_to_point)
        {
          return "ArtPorofluidCouplConNodeToPoint";
        }
        else
        {
          return "ArtPorofluidCouplConNodebased";
        }
      });

  // initialize meshtying object
  artery_porofluid_coupling_algorithm_ = create_and_init_artery_coupling_strategy(artery_dis_,
      porofluid_algorithm->discretization(), porofluid_params.sublist("artery_coupling"),
      coupling_condition_name, evaluate_on_lateral_surface);

  // Initialize rhs vector
  global_rhs_ = std::make_shared<Core::LinAlg::Vector<double>>(
      *artery_porofluid_coupling_algorithm_->full_map(), true);

  // Initialize increment vector
  global_increment_ = std::make_shared<Core::LinAlg::Vector<double>>(
      *artery_porofluid_coupling_algorithm_->full_map(), true);
  // Initialize phinp vector
  global_phinp_ = std::make_shared<Core::LinAlg::Vector<double>>(
      *artery_porofluid_coupling_algorithm_->full_map(), true);

  // initialize porofluid-elasticity system matrix
  global_sysmat_ =
      std::make_shared<Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(
          *artery_porofluid_coupling_algorithm_->global_extractor(),
          *artery_porofluid_coupling_algorithm_->global_extractor(), 81, false, true);
}



/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::MeshtyingArtery::prepare_time_loop() const
{
  artery_algorithm_->prepare_time_loop();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::MeshtyingArtery::prepare_time_step() const
{
  artery_algorithm_->prepare_time_step();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::MeshtyingArtery::update() const { artery_algorithm_->time_update(); }

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void PoroPressureBased::MeshtyingArtery::initialize_linear_solver(
    Core::LinAlg::Solver& solver) const
{
  const Teuchos::ParameterList& porofluid_params =
      Global::Problem::instance()->porofluid_pressure_based_dynamic_params();
  const int linear_solver_num =
      porofluid_params.sublist("nonlinear_solver").get<int>("linear_solver_id");
  const Teuchos::ParameterList& solver_params =
      Global::Problem::instance()->solver_params(linear_solver_num);
  const auto solver_type =
      Teuchos::getIntegralValue<Core::LinearSolver::SolverType>(solver_params, "SOLVER");
  // no need to do the rest for direct solvers
  if (solver_type == Core::LinearSolver::SolverType::umfpack or
      solver_type == Core::LinearSolver::SolverType::superlu)
    return;

  if (solver_type != Core::LinearSolver::SolverType::belos)
    FOUR_C_THROW("Iterative solver expected");

  const auto azprec_type =
      Teuchos::getIntegralValue<Core::LinearSolver::PreconditionerType>(solver_params, "AZPREC");

  // plausibility check
  switch (azprec_type)
  {
    case Core::LinearSolver::PreconditionerType::block_teko:
    {
      // no plausibility checks here
      // if you forget to declare an xml-file you will get an error message anyway
    }
    break;
    default:
      FOUR_C_THROW("Block Gauss-Seidel preconditioner expected.");
  }

  Teuchos::ParameterList& block_smoother_params_1 = solver.params().sublist("Inverse1");
  Core::LinearSolver::Parameters::compute_solver_parameters(
      *porofluid_algorithm_->discretization(), block_smoother_params_1);

  Teuchos::ParameterList& block_smoother_params_2 = solver.params().sublist("Inverse2");
  Core::LinearSolver::Parameters::compute_solver_parameters(*artery_dis_, block_smoother_params_2);
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void PoroPressureBased::MeshtyingArtery::linear_solve(
    const std::shared_ptr<Core::LinAlg::Solver> solver,
    Core::LinAlg::SolverParams& solver_params) const
{
  global_sysmat_->complete();

  global_increment_->put_scalar(0.0);

  // standard solver call
  // system is ready to solve since Dirichlet Boundary conditions have been applied in
  // setup_system_matrix or Evaluate
  solver_params.refactor = true;
  solver->solve(global_sysmat_, global_increment_, global_rhs_, solver_params);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::MeshtyingArtery::calculate_norms(
    std::vector<double>& residual_pressure_norm, std::vector<double>& increment_pressure_norm,
    std::vector<double>& pressure_norm) const
{
  residual_pressure_norm.resize(2);
  increment_pressure_norm.resize(2);
  pressure_norm.resize(2);

  pressure_norm[0] = calculate_vector_norm(vector_norm_inc_, *porofluid_algorithm_->phinp());
  pressure_norm[1] = calculate_vector_norm(vector_norm_inc_, *artery_algorithm_->pressurenp());

  std::shared_ptr<const Core::LinAlg::Vector<double>> artery_pressure_increment;
  std::shared_ptr<const Core::LinAlg::Vector<double>> porofluid_increment;

  artery_porofluid_coupling_algorithm_->extract_single_field_vectors(
      global_increment_, porofluid_increment, artery_pressure_increment);

  increment_pressure_norm[0] = calculate_vector_norm(vector_norm_inc_, *porofluid_increment);
  increment_pressure_norm[1] = calculate_vector_norm(vector_norm_inc_, *artery_pressure_increment);

  std::shared_ptr<const Core::LinAlg::Vector<double>> artery_pressure_rhs;
  std::shared_ptr<const Core::LinAlg::Vector<double>> porofluid_rhs;

  artery_porofluid_coupling_algorithm_->extract_single_field_vectors(
      global_rhs_, porofluid_rhs, artery_pressure_rhs);

  residual_pressure_norm[0] = calculate_vector_norm(vector_norm_res_, *porofluid_rhs);
  residual_pressure_norm[1] = calculate_vector_norm(vector_norm_res_, *artery_pressure_rhs);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::MeshtyingArtery::create_result_test() const
{
  const std::shared_ptr<Core::Utils::ResultTest> artery_result_test =
      artery_algorithm_->create_field_test();
  Global::Problem::instance()->add_field_test(artery_result_test);
}

/*----------------------------------------------------------------------*
 -----------------------------------------------------------------------*/
void PoroPressureBased::MeshtyingArtery::read_restart(const int step) const
{
  artery_algorithm_->read_restart(step);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::MeshtyingArtery::output() const
{
  if (porofluid_algorithm_->step() != 0) artery_algorithm_->output(false, nullptr);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::MeshtyingArtery::evaluate() const
{
  artery_porofluid_coupling_algorithm_->set_solution_vectors(
      porofluid_algorithm_->phinp(), porofluid_algorithm_->phin(), artery_algorithm_->pressurenp());

  // evaluate the coupling
  artery_porofluid_coupling_algorithm_->evaluate(global_sysmat_, global_rhs_);

  // evaluate artery
  artery_algorithm_->assemble_mat_and_rhs();
  // apply DBC
  artery_algorithm_->prepare_linear_solve();

  // SetupCoupledArteryPoroFluidSystem();
  artery_porofluid_coupling_algorithm_->setup_system(global_sysmat_, global_rhs_,
      porofluid_algorithm_->system_matrix(), artery_algorithm_->system_matrix(),
      porofluid_algorithm_->rhs(), artery_algorithm_->rhs(),
      porofluid_algorithm_->get_dbc_map_extractor(), artery_algorithm_->get_dbc_map_extractor());
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Vector<double>>
PoroPressureBased::MeshtyingArtery::extract_and_update_iter(
    const std::shared_ptr<const Core::LinAlg::Vector<double>> increment) const
{
  std::shared_ptr<const Core::LinAlg::Vector<double>> artery_pressure_increment;
  std::shared_ptr<const Core::LinAlg::Vector<double>> porofluid_increment;

  artery_porofluid_coupling_algorithm_->extract_single_field_vectors(
      increment, porofluid_increment, artery_pressure_increment);

  artery_algorithm_->update_iter(artery_pressure_increment);

  return porofluid_increment;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Map> PoroPressureBased::MeshtyingArtery::artery_dof_row_map()
    const
{
  return artery_porofluid_coupling_algorithm_->artery_dof_row_map();
}

/*-----------------------------------------------------------------------*
 *-----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase>
PoroPressureBased::MeshtyingArtery::artery_porofluid_sysmat() const
{
  return global_sysmat_;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Vector<double>>
PoroPressureBased::MeshtyingArtery::artery_porofluid_rhs() const
{
  return global_rhs_;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Vector<double>>
PoroPressureBased::MeshtyingArtery::combined_increment() const
{
  return global_increment_;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::MeshtyingArtery::check_initial_fields(
    const std::shared_ptr<const Core::LinAlg::Vector<double>> vector_homogenized) const
{
  artery_porofluid_coupling_algorithm_->check_initial_fields(
      vector_homogenized, artery_algorithm_->pressurenp());
}

/*-------------------------------------------------------------------------*
 *------------------------------------------------------------------------ */
void PoroPressureBased::MeshtyingArtery::set_nearby_ele_pairs(
    const std::map<int, std::set<int>>* nearby_ele_pairs) const
{
  artery_porofluid_coupling_algorithm_->set_nearby_ele_pairs(nearby_ele_pairs);
}

/*-------------------------------------------------------------------------*
 *------------------------------------------------------------------------ */
void PoroPressureBased::MeshtyingArtery::setup() const
{
  artery_porofluid_coupling_algorithm_->setup();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::MeshtyingArtery::apply_mesh_movement() const
{
  artery_porofluid_coupling_algorithm_->apply_mesh_movement();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Vector<double>>
PoroPressureBased::MeshtyingArtery::blood_vessel_volume_fraction() const
{
  return artery_porofluid_coupling_algorithm_->blood_vessel_volume_fraction();
}

FOUR_C_NAMESPACE_CLOSE

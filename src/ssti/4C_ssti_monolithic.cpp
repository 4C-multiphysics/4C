// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_ssti_monolithic.hpp"

#include "4C_adapter_scatra_base_algorithm.hpp"
#include "4C_adapter_str_ssiwrapper.hpp"
#include "4C_adapter_str_structure_new.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_equilibrate.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_scatra_timint_implicit.hpp"
#include "4C_scatra_timint_meshtying_strategy_s2i.hpp"
#include "4C_ssi_monolithic_evaluate_OffDiag.hpp"
#include "4C_ssti_algorithm.hpp"
#include "4C_ssti_monolithic_assemble_strategy.hpp"
#include "4C_ssti_monolithic_evaluate_OffDiag.hpp"
#include "4C_ssti_utils.hpp"
#include "4C_sti_monolithic_evaluate_OffDiag.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
SSTI::SSTIMono::SSTIMono(MPI_Comm comm, const Teuchos::ParameterList& globaltimeparams)
    : SSTIAlgorithm(comm, globaltimeparams),
      increment_(nullptr),
      residual_(nullptr),
      solver_(std::make_shared<Core::LinAlg::Solver>(
          Global::Problem::instance()->solver_params(
              globaltimeparams.sublist("MONOLITHIC").get<int>("LINEAR_SOLVER")),
          comm, Global::Problem::instance()->solver_params_callback(),
          Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
              Global::Problem::instance()->io_params(), "VERBOSITY"))),
      scatrastructureoffdiagcoupling_(nullptr),
      scatrathermooffdiagcoupling_(nullptr),
      thermostructureoffdiagcoupling_(nullptr),
      dtassemble_(0.0),
      dtevaluate_(0.0),
      dtnewton_(0.0),
      dtsolve_(0.0),
      timer_(std::make_shared<Teuchos::Time>("SSTI_Monolithic", true)),
      equilibration_method_{Teuchos::getIntegralValue<Core::LinAlg::EquilibrationMethod>(
                                globaltimeparams.sublist("MONOLITHIC"), "EQUILIBRATION"),
          Teuchos::getIntegralValue<Core::LinAlg::EquilibrationMethod>(
              globaltimeparams.sublist("MONOLITHIC"), "EQUILIBRATION_SCATRA"),
          Teuchos::getIntegralValue<Core::LinAlg::EquilibrationMethod>(
              globaltimeparams.sublist("MONOLITHIC"), "EQUILIBRATION_STRUCTURE"),
          Teuchos::getIntegralValue<Core::LinAlg::EquilibrationMethod>(
              globaltimeparams.sublist("MONOLITHIC"), "EQUILIBRATION_THERMO")},
      matrixtype_(Teuchos::getIntegralValue<Core::LinAlg::MatrixType>(
          globaltimeparams.sublist("MONOLITHIC"), "MATRIXTYPE")),
      convcheck_(std::make_shared<SSTI::ConvCheckMono>(globaltimeparams)),
      ssti_maps_mono_(nullptr),
      ssti_matrices_(nullptr),
      strategy_assemble_(nullptr),
      strategy_equilibration_(nullptr)
{
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void SSTI::SSTIMono::assemble_mat_and_rhs()
{
  double starttime = timer_->wallTime();

  ssti_matrices_->system_matrix()->zero();

  // assemble blocks of subproblems into system matrix
  strategy_assemble_->assemble_scatra(
      ssti_matrices_->system_matrix(), scatra_field()->system_matrix_operator());
  strategy_assemble_->assemble_structure(
      ssti_matrices_->system_matrix(), structure_field()->system_matrix());
  strategy_assemble_->assemble_thermo(
      ssti_matrices_->system_matrix(), thermo_field()->system_matrix_operator());

  // assemble domain contributions from coupling into system matrix
  strategy_assemble_->assemble_scatra_structure(ssti_matrices_->system_matrix(),
      ssti_matrices_->scatra_structure_domain(), ssti_matrices_->scatra_structure_interface());
  strategy_assemble_->assemble_structure_scatra(
      ssti_matrices_->system_matrix(), ssti_matrices_->structure_scatra_domain());
  strategy_assemble_->assemble_thermo_structure(ssti_matrices_->system_matrix(),
      ssti_matrices_->thermo_structure_domain(), ssti_matrices_->thermo_structure_interface());
  strategy_assemble_->assemble_structure_thermo(
      ssti_matrices_->system_matrix(), ssti_matrices_->structure_thermo_domain());
  strategy_assemble_->assemble_thermo_scatra(ssti_matrices_->system_matrix(),
      ssti_matrices_->thermo_scatra_domain(), ssti_matrices_->thermo_scatra_interface());
  strategy_assemble_->assemble_scatra_thermo_domain(
      ssti_matrices_->system_matrix(), ssti_matrices_->scatra_thermo_domain());

  // assemble interface contributions from coupling into system matrix
  if (interface_meshtying())
  {
    strategy_assemble_->assemble_scatra_thermo_interface(
        ssti_matrices_->system_matrix(), ssti_matrices_->scatra_thermo_interface());
  }

  // apply meshtying on structural linearizations
  strategy_assemble_->apply_meshtying_system_matrix(ssti_matrices_->system_matrix());

  // finalize global system matrix
  ssti_matrices_->system_matrix()->complete();

  // apply Dirichlet conditions
  ssti_matrices_->system_matrix()->apply_dirichlet(
      *scatra_field()->dirich_maps()->cond_map(), true);
  ssti_matrices_->system_matrix()->apply_dirichlet(
      *thermo_field()->dirich_maps()->cond_map(), true);
  strategy_assemble_->apply_structural_dbc_system_matrix(ssti_matrices_->system_matrix());

  // assemble RHS
  strategy_assemble_->assemble_rhs(
      residual_, scatra_field()->residual(), *structure_field()->rhs(), thermo_field()->residual());

  double mydt = timer_->wallTime() - starttime;
  Core::Communication::max_all(&mydt, &dtassemble_, 1, get_comm());
}

/*-------------------------------------------------------------------------------*
 *-------------------------------------------------------------------------------*/
void SSTI::SSTIMono::build_null_spaces()
{
  // build null spaces for scatra and thermo
  switch (scatra_field()->matrix_type())
  {
    case Core::LinAlg::MatrixType::block_condition:
    case Core::LinAlg::MatrixType::block_condition_dof:
    {
      scatra_field()->build_block_null_spaces(
          solver_, get_block_positions(Subproblem::scalar_transport).at(0));
      thermo_field()->build_block_null_spaces(
          solver_, get_block_positions(Subproblem::thermo).at(0));
      break;
    }
    case Core::LinAlg::MatrixType::sparse:
    {
      // equip smoother for scatra matrix block with empty parameter sub lists to trigger null space
      // computation
      std::ostringstream scatrablockstr;
      scatrablockstr << get_block_positions(Subproblem::scalar_transport).at(0) + 1;
      Teuchos::ParameterList& blocksmootherparamsscatra =
          solver_->params().sublist("Inverse" + scatrablockstr.str());

      blocksmootherparamsscatra.sublist("Belos Parameters");
      blocksmootherparamsscatra.sublist("MueLu Parameters");

      // equip smoother for scatra matrix block with null space associated with all degrees of
      // freedom on scatra discretization
      scatra_field()->discretization()->compute_null_space_if_necessary(blocksmootherparamsscatra);

      std::ostringstream thermoblockstr;
      thermoblockstr << get_block_positions(Subproblem::thermo).at(0) + 1;
      Teuchos::ParameterList& blocksmootherparamsthermo =
          solver_->params().sublist("Inverse" + thermoblockstr.str());
      blocksmootherparamsthermo.sublist("Belos Parameters");
      blocksmootherparamsthermo.sublist("MueLu Parameters");

      // equip smoother for scatra matrix block with null space associated with all degrees of
      // freedom on scatra discretization
      thermo_field()->discretization()->compute_null_space_if_necessary(blocksmootherparamsthermo);
      break;
    }
    default:
    {
      FOUR_C_THROW("Invalid matrix type associated with scalar transport field!");
      break;
    }
  }
  // build null spaces for structure
  {
    // store number of matrix block associated with structural field as string
    std::stringstream iblockstr;
    iblockstr << get_block_positions(Subproblem::structure).at(0) + 1;

    // equip smoother for structural matrix block with empty parameter sub lists to trigger null
    // space computation
    Teuchos::ParameterList& blocksmootherparams =
        solver_->params().sublist("Inverse" + iblockstr.str());
    blocksmootherparams.sublist("Belos Parameters");
    blocksmootherparams.sublist("MueLu Parameters");

    // equip smoother for structural matrix block with null space associated with all degrees of
    // freedom on structural discretization
    structure_field()->discretization()->compute_null_space_if_necessary(blocksmootherparams);
  }
}  // SSTI::SSTI_Mono::build_null_spaces

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void SSTI::SSTIMono::init(MPI_Comm comm, const Teuchos::ParameterList& sstitimeparams,
    const Teuchos::ParameterList& scatraparams, const Teuchos::ParameterList& thermoparams,
    const Teuchos::ParameterList& structparams)
{
  // check input parameters for scalar transport field
  if (Teuchos::getIntegralValue<Inpar::ScaTra::VelocityField>(scatraparams, "VELOCITYFIELD") !=
      Inpar::ScaTra::velocity_Navier_Stokes)
    FOUR_C_THROW("Invalid type of velocity field for scalar-structure interaction!");

  // call base class routine
  SSTIAlgorithm::init(comm, sstitimeparams, scatraparams, thermoparams, structparams);
}


/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void SSTI::SSTIMono::output()
{
  // print finish line of convergence table to screen
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    std::cout << "+------------+-------------------+--------------+--------------+--------------+--"
                 "------------+--------------+--------------+--------------+--------------+--------"
                 "------+"
              << std::endl;
    std::cout << "| Computation time for this timestep: " << std::setw(10) << time_statistics()[2]
              << "                                                                                 "
                 "                                       |"
              << std::endl;
    std::cout << "+--------------------------------------------------------------------------------"
                 "---------------------------------------------------------------------------------"
                 "------+"
              << std::endl;
  }

  scatra_field()->check_and_write_output_and_restart();
  thermo_field()->check_and_write_output_and_restart();
  structure_field()->output();
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void SSTI::SSTIMono::prepare_time_step()
{
  // update time and time step
  increment_time_and_step();

  distribute_solution_all_fields();

  // in first time step: solve to get initial derivatives
  scatra_field()->prepare_time_step();

  // if adaptive time stepping and different time step size: calculate time step in scatra
  // (prepare_time_step() of Scatra) and pass to structure and thermo
  if (scatra_field()->time_step_adapted()) distribute_dt_from_scatra();

  // in first time step: solve to get initial derivatives
  thermo_field()->prepare_time_step();

  // pass scalar transport degrees of freedom to structural discretization
  // has to be called AFTER ScaTraField()->prepare_time_step() to ensure
  // consistent scalar transport state vector with valid Dirichlet conditions
  structure_field()->prepare_time_step();

  scatra_field()->print_time_step_info();
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void SSTI::SSTIMono::setup()
{
  // call base class routine
  SSTIAlgorithm::setup();

  // safety checks
  if (scatra_field()->num_scal() != 1)
  {
    FOUR_C_THROW(
        "Since the ssti_monolithic framework is only implemented for usage in combination with "
        "volume change laws 'MAT_InelasticDefgradLinScalarIso' or  "
        "'MAT_InelasticDefgradLinScalarAniso' so far and these laws are implemented for only one "
        "transported scalar at the moment it is not reasonable to use them with more than one "
        "transported scalar. So you need to cope with it or change implementation! ;-)");
  }
  if (equilibration_method_.global != Core::LinAlg::EquilibrationMethod::local and
      (equilibration_method_.structure != Core::LinAlg::EquilibrationMethod::none or
          equilibration_method_.scatra != Core::LinAlg::EquilibrationMethod::none or
          equilibration_method_.thermo != Core::LinAlg::EquilibrationMethod::none))
    FOUR_C_THROW("Either global equilibration or local equilibration");

  if (matrixtype_ == Core::LinAlg::MatrixType::sparse and
      (equilibration_method_.structure != Core::LinAlg::EquilibrationMethod::none or
          equilibration_method_.scatra != Core::LinAlg::EquilibrationMethod::none or
          equilibration_method_.thermo != Core::LinAlg::EquilibrationMethod::none))
    FOUR_C_THROW("Block based equilibration only for block matrices");

  const bool equilibration_scatra_initial = Global::Problem::instance()
                                                ->ssti_control_params()
                                                .sublist("MONOLITHIC")
                                                .get<bool>("EQUILIBRATION_INIT_SCATRA");
  const bool calc_initial_pot =
      Global::Problem::instance()->elch_control_params().get<bool>("INITPOTCALC");

  if (!equilibration_scatra_initial and
      scatra_field()->equilibration_method() != Core::LinAlg::EquilibrationMethod::none)
  {
    FOUR_C_THROW(
        "You are within the monolithic SSTI framework but activated a pure scatra equilibration "
        "method. Delete this from 'SCALAR TRANSPORT DYNAMIC' section and set it in 'SSTI "
        "CONTROL/MONOLITHIC' instead.");
  }
  if (equilibration_scatra_initial and
      scatra_field()->equilibration_method() == Core::LinAlg::EquilibrationMethod::none)
  {
    FOUR_C_THROW(
        "You selected to equilibrate equations of initial potential but did not specify any "
        "equilibration method in ScaTra.");
  }
  if (equilibration_scatra_initial and !calc_initial_pot)
  {
    FOUR_C_THROW(
        "You selected to equilibrate equations of initial potential but did not activate "
        "INITPOTCALC in ELCH CONTROL");
  }

  if (!scatra_field()->is_incremental())
    FOUR_C_THROW("Must have incremental solution approach for monolithic SSTI!");
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void SSTI::SSTIMono::setup_system()
{
  if (interface_meshtying())
    ssti_structure_mesh_tying()->check_slave_side_has_dirichlet_conditions(
        structure_field()->get_dbc_map_extractor()->cond_map());

  // Setup all kind of maps
  ssti_maps_mono_ = std::make_shared<SSTI::SSTIMapsMono>(*this);

  // initialize global increment vector for Newton-Raphson iteration
  increment_ = Core::LinAlg::create_vector(*ssti_maps_mono_->maps_sub_problems()->full_map(), true);

  // initialize global residual vector
  residual_ = Core::LinAlg::create_vector(*ssti_maps_mono_->maps_sub_problems()->full_map(), true);

  if (matrixtype_ == Core::LinAlg::MatrixType::block_field)
  {
    if (!solver_->params().isSublist("AMGnxn Parameters"))
      FOUR_C_THROW(
          "Global system matrix with block structure requires AMGnxn block preconditioner!");

    // feed AMGnxn block preconditioner with null space information for each block of global
    // block system matrix
    build_null_spaces();
  }

  // initialize submatrices and system matrix
  ssti_matrices_ = std::make_shared<SSTI::SSTIMatrices>(
      ssti_maps_mono_, matrixtype_, scatra_field()->matrix_type(), interface_meshtying());

  // initialize strategy for assembly
  strategy_assemble_ = SSTI::build_assemble_strategy(
      Core::Utils::shared_ptr_from_ref(*this), matrixtype_, scatra_field()->matrix_type());

  // initialize evaluation objects for coupling between subproblems
  scatrastructureoffdiagcoupling_ = std::make_shared<SSI::ScatraStructureOffDiagCouplingSSTI>(
      ssti_maps_mono_->block_map_structure(),
      ssti_maps_mono_->maps_sub_problems()->map(get_problem_position(Subproblem::scalar_transport)),
      ssti_maps_mono_->maps_sub_problems()->map(get_problem_position(Subproblem::structure)),
      ssti_structure_mesh_tying(), meshtying_scatra(), scatra_field(), structure_field());

  thermostructureoffdiagcoupling_ = std::make_shared<SSTI::ThermoStructureOffDiagCoupling>(
      ssti_maps_mono_->block_map_structure(), ssti_maps_mono_->block_map_thermo(),
      ssti_maps_mono_->maps_sub_problems()->map(get_problem_position(Subproblem::structure)),
      ssti_maps_mono_->maps_sub_problems()->map(get_problem_position(Subproblem::thermo)),
      ssti_structure_mesh_tying(), meshtying_thermo(), structure_field(), thermo_field_base());

  // Note: STI evaluation of off diagonal coupling is designed to use interface maps for the
  // interface coupling matrices. In SSTI we always use the full maps and thus hand in the same map
  // multiple times for both domain and interface contributions.
  scatrathermooffdiagcoupling_ = std::make_shared<STI::ScatraThermoOffDiagCouplingMatchingNodes>(
      ssti_maps_mono_->block_map_thermo(), ssti_maps_mono_->block_map_thermo(),
      ssti_maps_mono_->block_map_thermo(),
      ssti_maps_mono_->maps_sub_problems()->map(get_problem_position(Subproblem::scalar_transport)),
      ssti_maps_mono_->maps_sub_problems()->map(get_problem_position(Subproblem::thermo)),
      ssti_maps_mono_->maps_sub_problems()->map(get_problem_position(Subproblem::scalar_transport)),
      ssti_maps_mono_->maps_sub_problems()->map(get_problem_position(Subproblem::thermo)), true,
      meshtying_scatra(), meshtying_thermo(), scatra_field_base(), thermo_field_base());

  // initialize equilibration class
  strategy_equilibration_ = Core::LinAlg::build_equilibration(
      matrixtype_, get_block_equilibration(), all_maps()->maps_sub_problems()->full_map());
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void SSTI::SSTIMono::newton_loop()
{
  double starttime = timer_->wallTime();

  // initialize counter for Newton-Raphson iteration
  reset_iter();

  // start Newton-Raphson iteration
  while (true)
  {
    prepare_newton_step();

    ssti_matrices_->un_complete_coupling_matrices();

    evaluate_subproblems();

    ssti_matrices_->complete_coupling_matrices();

    assemble_mat_and_rhs();

    if (convcheck_->converged(*this)) break;

    linear_solve();

    update_iter_states();
  }

  double mydt = timer_->wallTime() - starttime;
  Core::Communication::max_all(&mydt, &dtnewton_, 1, get_comm());
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void SSTI::SSTIMono::timeloop()
{
  // output initial scalar transport solution to screen and files
  if (step() == 0)
  {
    distribute_solution_all_fields();

    scatra_field()->prepare_time_loop();
    thermo_field()->prepare_time_loop();
  }
  // time loop
  while (not_finished() and scatra_field()->not_finished())
  {
    prepare_time_step();

    newton_loop();

    constexpr bool force_prepare = false;
    structure_field()->prepare_output(force_prepare);

    update();

    output();
  }
}

/*--------------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------------*/
void SSTI::SSTIMono::update()
{
  scatra_field()->update();
  thermo_field()->update();
  structure_field()->update();
}

/*--------------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> SSTI::SSTIMono::extract_sub_increment(Subproblem sub)
{
  std::shared_ptr<Core::LinAlg::Vector<double>> subincrement(nullptr);
  switch (sub)
  {
    case Subproblem::structure:
    {
      // First, extract increment from domain and master side
      subincrement = ssti_maps_mono_->maps_sub_problems()->extract_vector(
          *increment_, get_problem_position(Subproblem::structure));

      // Second, copy master side displacements and increments to slave side for meshtying
      if (interface_meshtying())
      {
        for (const auto& meshtying : ssti_structure_mesh_tying()->mesh_tying_handlers())
        {
          auto coupling_adapter = meshtying->slave_master_coupling();
          auto coupling_map_extractor = meshtying->slave_master_extractor();

          // displacements
          coupling_map_extractor->insert_vector(
              *coupling_adapter->master_to_slave(
                  *coupling_map_extractor->extract_vector(*structure_field()->dispnp(), 2)),
              1, *structure_field()->write_access_dispnp());
          structure_field()->set_state(structure_field()->write_access_dispnp());
          // increments
          coupling_map_extractor->insert_vector(
              *coupling_adapter->master_to_slave(
                  *coupling_map_extractor->extract_vector(*subincrement, 2)),
              1, *subincrement);
        }
      }
      break;
    }
    case Subproblem::scalar_transport:
    {
      subincrement = ssti_maps_mono_->maps_sub_problems()->extract_vector(
          *increment_, get_problem_position(Subproblem::scalar_transport));
      break;
    }
    case Subproblem::thermo:
    {
      subincrement = ssti_maps_mono_->maps_sub_problems()->extract_vector(
          *increment_, get_problem_position(Subproblem::thermo));
      break;
    }
    default:
    {
      FOUR_C_THROW("Unknown type of subproblem in SSTI");
    }
  }
  return subincrement;
}

/*--------------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------------*/
void SSTI::SSTIMono::evaluate_subproblems()
{
  double starttime = timer_->wallTime();

  // clear all matrices from previous Newton iteration
  ssti_matrices_->clear_matrices();

  // needed to communicate to NOX state
  structure_field()->set_state(structure_field()->write_access_dispnp());

  // distribute solution from all fields to each other
  distribute_solution_all_fields();

  // evaluate all subproblems
  structure_field()->evaluate();
  scatra_field()->prepare_linear_solve();
  thermo_field()->prepare_linear_solve();

  // evaluate domain contributions from coupling
  scatrastructureoffdiagcoupling_->evaluate_off_diag_block_scatra_structure_domain(
      ssti_matrices_->scatra_structure_domain());
  scatrastructureoffdiagcoupling_->evaluate_off_diag_block_structure_scatra_domain(
      ssti_matrices_->structure_scatra_domain());
  thermostructureoffdiagcoupling_->evaluate_off_diag_block_thermo_structure_domain(
      ssti_matrices_->thermo_structure_domain());
  thermostructureoffdiagcoupling_->evaluate_off_diag_block_structure_thermo_domain(
      ssti_matrices_->structure_thermo_domain());
  scatrathermooffdiagcoupling_->evaluate_off_diag_block_thermo_scatra_domain(
      ssti_matrices_->thermo_scatra_domain());
  scatrathermooffdiagcoupling_->evaluate_off_diag_block_scatra_thermo_domain(
      ssti_matrices_->scatra_thermo_domain());

  // evaluate interface contributions from coupling
  if (interface_meshtying())
  {
    scatrastructureoffdiagcoupling_->evaluate_off_diag_block_scatra_structure_interface(
        *ssti_matrices_->scatra_structure_interface());
    thermostructureoffdiagcoupling_->evaluate_off_diag_block_thermo_structure_interface(
        *ssti_matrices_->thermo_structure_interface());
    scatrathermooffdiagcoupling_->evaluate_off_diag_block_thermo_scatra_interface(
        ssti_matrices_->thermo_scatra_interface());
    scatrathermooffdiagcoupling_->evaluate_off_diag_block_scatra_thermo_interface(
        ssti_matrices_->scatra_thermo_interface());
  }

  double mydt = timer_->wallTime() - starttime;
  Core::Communication::max_all(&mydt, &dtevaluate_, 1, get_comm());
}

/*--------------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------------*/
void SSTI::SSTIMono::linear_solve()
{
  double starttime = timer_->wallTime();

  increment_->put_scalar(0.0);

  if (!ssti_matrices_->system_matrix()->filled())
    FOUR_C_THROW("Complete() has not been called on global system matrix yet!");

  strategy_equilibration_->equilibrate_system(
      ssti_matrices_->system_matrix(), residual_, all_maps()->block_map_system_matrix());

  Core::LinAlg::SolverParams solver_params;
  solver_params.refactor = true;
  solver_params.reset = iter() == 1;

  solver_->solve(ssti_matrices_->system_matrix(), increment_, residual_, solver_params);

  strategy_equilibration_->unequilibrate_increment(increment_);

  double mydt = timer_->wallTime() - starttime;
  Core::Communication::max_all(&mydt, &dtsolve_, 1, get_comm());
}

/*--------------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------------*/
void SSTI::SSTIMono::update_iter_states()
{
  scatra_field()->update_iter(*extract_sub_increment(Subproblem::scalar_transport));
  scatra_field()->compute_intermediate_values();

  thermo_field()->update_iter(*extract_sub_increment(Subproblem::thermo));
  thermo_field()->compute_intermediate_values();

  structure_field()->update_state_incrementally(extract_sub_increment(Subproblem::structure));
}

/*--------------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------------*/
void SSTI::SSTIMono::prepare_newton_step()
{
  // update iteration counter
  increment_iter();

  // reset timer
  timer_->reset();

  ssti_matrices_->system_matrix()->zero();
}

/*--------------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------------*/
std::vector<int> SSTI::SSTIMono::get_block_positions(Subproblem subproblem) const
{
  if (matrixtype_ == Core::LinAlg::MatrixType::sparse)
    FOUR_C_THROW("Sparse matrices have just one block");

  auto block_position = std::vector<int>(0);

  switch (subproblem)
  {
    case Subproblem::structure:
    {
      if (scatra_field()->matrix_type() == Core::LinAlg::MatrixType::sparse)
        block_position.emplace_back(1);
      else
        block_position.emplace_back(scatra_field()->block_maps()->num_maps());
      break;
    }
    case Subproblem::scalar_transport:
    {
      if (scatra_field()->matrix_type() == Core::LinAlg::MatrixType::sparse)
        block_position.emplace_back(0);
      else

      {
        for (int i = 0; i < static_cast<int>(scatra_field()->block_maps()->num_maps()); ++i)
          block_position.emplace_back(i);
      }
      break;
    }
    case Subproblem::thermo:
    {
      if (thermo_field()->matrix_type() == Core::LinAlg::MatrixType::sparse)
        block_position.emplace_back(2);
      else
      {
        for (int i = 0; i < static_cast<int>(thermo_field()->block_maps()->num_maps()); ++i)
          block_position.emplace_back(scatra_field()->block_maps()->num_maps() + 1 + i);
      }
      break;
    }
    default:
    {
      FOUR_C_THROW("Unknown type of subproblem");
      break;
    }
  }

  return block_position;
}

/*--------------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------------*/
int SSTI::SSTIMono::get_problem_position(Subproblem subproblem) const
{
  int position = -1;

  switch (subproblem)
  {
    case Subproblem::structure:
    {
      position = 1;
      break;
    }
    case Subproblem::scalar_transport:
    {
      position = 0;
      break;
    }
    case Subproblem::thermo:
    {
      position = 2;
      break;
    }
    default:
    {
      FOUR_C_THROW("Unknown type of subproblem");
      break;
    }
  }

  return position;
}

/*--------------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------------*/
std::vector<Core::LinAlg::EquilibrationMethod> SSTI::SSTIMono::get_block_equilibration()
{
  std::vector<Core::LinAlg::EquilibrationMethod> equilibration_method_vector;
  switch (matrixtype_)
  {
    case Core::LinAlg::MatrixType::sparse:
    {
      equilibration_method_vector =
          std::vector<Core::LinAlg::EquilibrationMethod>(1, equilibration_method_.global);
      break;
    }
    case Core::LinAlg::MatrixType::block_field:
    {
      if (equilibration_method_.global != Core::LinAlg::EquilibrationMethod::local)
      {
        equilibration_method_vector =
            std::vector<Core::LinAlg::EquilibrationMethod>(1, equilibration_method_.global);
      }
      else if (equilibration_method_.structure == Core::LinAlg::EquilibrationMethod::none and
               equilibration_method_.scatra == Core::LinAlg::EquilibrationMethod::none and
               equilibration_method_.thermo == Core::LinAlg::EquilibrationMethod::none)
      {
        equilibration_method_vector = std::vector<Core::LinAlg::EquilibrationMethod>(
            1, Core::LinAlg::EquilibrationMethod::none);
      }
      else
      {
        auto block_positions_scatra = get_block_positions(Subproblem::scalar_transport);
        auto block_position_structure = get_block_positions(Subproblem::structure);
        auto block_positions_thermo = get_block_positions(Subproblem::thermo);

        equilibration_method_vector = std::vector<Core::LinAlg::EquilibrationMethod>(
            block_positions_scatra.size() + block_position_structure.size() +
                block_positions_thermo.size(),
            Core::LinAlg::EquilibrationMethod::none);

        for (const int block_position_scatra : block_positions_scatra)
          equilibration_method_vector.at(block_position_scatra) = equilibration_method_.scatra;

        equilibration_method_vector.at(block_position_structure.at(0)) =
            equilibration_method_.structure;

        for (const int block_position_thermo : block_positions_thermo)
          equilibration_method_vector.at(block_position_thermo) = equilibration_method_.thermo;
      }

      break;
    }
    default:
    {
      FOUR_C_THROW("Invalid matrix type associated with system matrix field!");
      break;
    }
  }

  return equilibration_method_vector;
}

FOUR_C_NAMESPACE_CLOSE

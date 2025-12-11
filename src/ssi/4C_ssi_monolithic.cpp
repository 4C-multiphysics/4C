// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_ssi_monolithic.hpp"

#include "4C_adapter_scatra_base_algorithm.hpp"
#include "4C_adapter_str_ssiwrapper.hpp"
#include "4C_adapter_str_structure_new.hpp"
#include "4C_contact_nitsche_strategy_ssi.hpp"
#include "4C_fem_condition_locsys.hpp"
#include "4C_fem_discretization_nullspace.hpp"
#include "4C_fem_general_assemblestrategy.hpp"
#include "4C_global_data.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_equilibrate.hpp"
#include "4C_linalg_mapextractor.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_linalg_utils_sparse_algebra_print.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_linear_solver_method_parameters.hpp"
#include "4C_scatra_timint_elch.hpp"
#include "4C_scatra_timint_meshtying_strategy_s2i.hpp"
#include "4C_ssi_contact_strategy.hpp"
#include "4C_ssi_coupling.hpp"
#include "4C_ssi_input.hpp"
#include "4C_ssi_manifold_utils.hpp"
#include "4C_ssi_monolithic_assemble_strategy.hpp"
#include "4C_ssi_monolithic_convcheck_strategies.hpp"
#include "4C_ssi_monolithic_dbc_handler.hpp"
#include "4C_ssi_monolithic_evaluate_OffDiag.hpp"
#include "4C_ssi_monolithic_meshtying_strategy.hpp"
#include "4C_ssi_utils.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
SSI::SsiMono::SsiMono(MPI_Comm comm, const Teuchos::ParameterList& globaltimeparams)
    : SSIBase(comm, globaltimeparams),
      equilibration_method_{.global = Teuchos::getIntegralValue<Core::LinAlg::EquilibrationMethod>(
                                globaltimeparams.sublist("MONOLITHIC"), "EQUILIBRATION"),
          .scatra = Teuchos::getIntegralValue<Core::LinAlg::EquilibrationMethod>(
              globaltimeparams.sublist("MONOLITHIC"), "EQUILIBRATION_SCATRA"),
          .structure = Teuchos::getIntegralValue<Core::LinAlg::EquilibrationMethod>(
              globaltimeparams.sublist("MONOLITHIC"), "EQUILIBRATION_STRUCTURE")},
      matrixtype_(Teuchos::getIntegralValue<Core::LinAlg::MatrixType>(
          globaltimeparams.sublist("MONOLITHIC"), "MATRIXTYPE")),
      print_matlab_(globaltimeparams.sublist("MONOLITHIC").get<bool>("PRINT_MAT_RHS_MAP_MATLAB")),
      relax_lin_solver_tolerance_(
          globaltimeparams.sublist("MONOLITHIC").get<double>("RELAX_LIN_SOLVER_TOLERANCE")),
      relax_lin_solver_iter_step_(
          globaltimeparams.sublist("MONOLITHIC").get<int>("RELAX_LIN_SOLVER_STEP")),
      solver_(std::make_shared<Core::LinAlg::Solver>(
          Global::Problem::instance()->solver_params(
              globaltimeparams.sublist("MONOLITHIC").get<int>("LINEAR_SOLVER")),
          comm, Global::Problem::instance()->solver_params_callback(),
          Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
              Global::Problem::instance()->io_params(), "VERBOSITY"))),
      timer_(std::make_shared<Teuchos::Time>("SSI_Mono", true))
{
}

/*-------------------------------------------------------------------------------*
 *-------------------------------------------------------------------------------*/
void SSI::SsiMono::apply_contact_to_sub_problems() const
{
  // uncomplete matrices; we need to do this here since in contact simulations the dofs that
  // interact with each other can change and thus the graph of the matrix can also change.
  ssi_matrices_->scatra_matrix()->un_complete();
  ssi_matrices_->scatra_structure_matrix()->un_complete();
  ssi_matrices_->structure_scatra_matrix()->un_complete();

  // add contributions
  strategy_contact_->apply_contact_to_scatra_residual(ssi_vectors_->scatra_residual());
  strategy_contact_->apply_contact_to_scatra_scatra(ssi_matrices_->scatra_matrix());
  strategy_contact_->apply_contact_to_scatra_structure(ssi_matrices_->scatra_structure_matrix());
  strategy_contact_->apply_contact_to_structure_scatra(ssi_matrices_->structure_scatra_matrix());
}

/*-------------------------------------------------------------------------------*
 *-------------------------------------------------------------------------------*/
void SSI::SsiMono::apply_dbc_to_system() const
{
  // apply Dirichlet boundary conditions to global system matrix
  dbc_handler_->apply_dbc_to_system_matrix(ssi_matrices_->system_matrix());

  // apply Dirichlet boundary conditions to global RHS
  dbc_handler_->apply_dbc_to_rhs(ssi_vectors_->residual());
}

/*-------------------------------------------------------------------------------*
 *-------------------------------------------------------------------------------*/
bool SSI::SsiMono::is_uncomplete_of_matrices_necessary_for_mesh_tying() const
{
  // check for first iteration in calculation of initial time derivative
  if (iteration_count() == 0 and step() == 0 and !do_calculate_initial_potential_field())
    return true;

  if (iteration_count() <= 2)
  {
    // check for first iteration in standard Newton loop
    if (step() == 1 and !do_calculate_initial_potential_field()) return true;

    // check for first iterations in calculation of initial potential field
    if (step() == 0 and do_calculate_initial_potential_field()) return true;

    // check for first iteration in restart simulations
    if (is_restart())
    {
      auto* problem = Global::Problem::instance();
      // restart based on time step
      if (step() == problem->restart() + 1) return true;
    }
  }

  // if we have at least one contact interface the dofs that are in contact can change and therefore
  // also the matrices have to be uncompleted
  return ssi_interface_contact();
}

/*-------------------------------------------------------------------------------*
 *-------------------------------------------------------------------------------*/
void SSI::SsiMono::apply_meshtying_to_sub_problems() const
{
  TEUCHOS_FUNC_TIME_MONITOR("SSI mono: apply mesh tying");
  if (ssi_interface_meshtying())
  {
    // check if matrices are filled because they have to be for the below methods
    if (!ssi_matrices_->structure_scatra_matrix()->filled())
      ssi_matrices_->complete_structure_scatra_matrix();
    if (!ssi_matrices_->scatra_structure_matrix()->filled())
      ssi_matrices_->complete_scatra_structure_matrix();

    strategy_meshtying_->apply_meshtying_to_scatra_structure(
        ssi_matrices_->scatra_structure_matrix(), *ssi_maps(), *ssi_structure_mesh_tying(),
        is_uncomplete_of_matrices_necessary_for_mesh_tying());

    strategy_meshtying_->apply_meshtying_to_structure_matrix(*ssi_matrices_->structure_matrix(),
        *structure_field()->system_matrix(), *ssi_structure_mesh_tying(),
        is_uncomplete_of_matrices_necessary_for_mesh_tying());

    strategy_meshtying_->apply_meshtying_to_structure_scatra(
        ssi_matrices_->structure_scatra_matrix(), *ssi_maps(), *ssi_structure_mesh_tying(),
        is_uncomplete_of_matrices_necessary_for_mesh_tying());

    ssi_vectors_->structure_residual()->update(1.0,
        strategy_meshtying_->apply_meshtying_to_structure_rhs(
            *structure_field()->rhs(), *ssi_maps(), *ssi_structure_mesh_tying()),
        1.0);

    if (is_scatra_manifold())
    {
      if (!ssi_matrices_->scatra_manifold_structure_matrix()->filled())
        ssi_matrices_->complete_scatra_manifold_structure_matrix();
      if (!manifoldscatraflux_->matrix_manifold_structure()->filled())
        manifoldscatraflux_->complete_matrix_manifold_structure();
      if (!manifoldscatraflux_->matrix_scatra_structure()->filled())
        manifoldscatraflux_->complete_matrix_scatra_structure();

      strategy_meshtying_->apply_meshtying_to_scatra_manifold_structure(
          ssi_matrices_->scatra_manifold_structure_matrix(), *ssi_maps(),
          *ssi_structure_mesh_tying(), is_uncomplete_of_matrices_necessary_for_mesh_tying());

      strategy_meshtying_->apply_meshtying_to_scatra_manifold_structure(
          manifoldscatraflux_->matrix_manifold_structure(), *ssi_maps(),
          *ssi_structure_mesh_tying(), is_uncomplete_of_matrices_necessary_for_mesh_tying());

      strategy_meshtying_->apply_meshtying_to_scatra_structure(
          manifoldscatraflux_->matrix_scatra_structure(), *ssi_maps(), *ssi_structure_mesh_tying(),
          is_uncomplete_of_matrices_necessary_for_mesh_tying());
    }
  }
  // copy the structure residual and matrix if we do not have a mesh tying problem
  else
  {
    ssi_vectors_->structure_residual()->update(1.0, *(structure_field()->rhs()), 1.0);
    Core::LinAlg::matrix_add(
        *structure_field()->system_matrix(), false, 1.0, *ssi_matrices_->structure_matrix(), 1.0);
  }
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void SSI::SsiMono::apply_manifold_meshtying() const
{
  if (!manifoldscatraflux_->system_matrix_manifold()->filled())
    manifoldscatraflux_->system_matrix_manifold()->complete();

  if (!ssi_matrices_->scatra_manifold_structure_matrix()->filled())
    ssi_matrices_->complete_scatra_manifold_structure_matrix();

  if (!manifoldscatraflux_->matrix_manifold_structure()->filled())
    manifoldscatraflux_->complete_matrix_manifold_structure();

  if (!manifoldscatraflux_->matrix_scatra_manifold()->filled())
    manifoldscatraflux_->complete_matrix_scatra_manifold();

  if (!manifoldscatraflux_->matrix_manifold_scatra()->filled())
    manifoldscatraflux_->complete_matrix_manifold_scatra();

  // apply mesh tying to...
  // manifold - manifold
  strategy_manifold_meshtying_->apply_meshtying_to_manifold_matrix(
      ssi_matrices_->manifold_matrix(), scatra_manifold()->system_matrix_operator());
  strategy_manifold_meshtying_->apply_meshtying_to_manifold_matrix(
      ssi_matrices_->manifold_matrix(), manifoldscatraflux_->system_matrix_manifold());

  // manifold - structure
  strategy_manifold_meshtying_->apply_meshtying_to_manifold_structure_matrix(
      ssi_matrices_->scatra_manifold_structure_matrix(),
      manifoldscatraflux_->matrix_manifold_structure(),
      is_uncomplete_of_matrices_necessary_for_mesh_tying());

  // scatra - manifold
  strategy_manifold_meshtying_->apply_meshtying_to_scatra_manifold_matrix(
      ssi_matrices_->scatra_scatra_manifold_matrix(), manifoldscatraflux_->matrix_scatra_manifold(),
      is_uncomplete_of_matrices_necessary_for_mesh_tying());

  // manifold - scatra
  strategy_manifold_meshtying_->apply_meshtying_to_manifold_scatra_matrix(
      ssi_matrices_->scatra_manifold_scatra_matrix(),
      manifoldscatraflux_->matrix_manifold_scatra());

  // RHS
  strategy_manifold_meshtying_->apply_mesh_tying_to_manifold_rhs(
      *ssi_vectors_->manifold_residual());
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void SSI::SsiMono::assemble_mat_and_rhs() const
{
  TEUCHOS_FUNC_TIME_MONITOR("SSI mono: assemble global system");

  assemble_mat_scatra();

  assemble_mat_structure();

  if (is_scatra_manifold()) assemble_mat_scatra_manifold();

  // finalize global system matrix
  ssi_matrices_->system_matrix()->complete();

  // assemble monolithic RHS
  strategy_assemble_->assemble_rhs(ssi_vectors_->residual(), ssi_vectors_->scatra_residual(),
      ssi_vectors_->structure_residual(), ssi_vectors_->manifold_residual().get());
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void SSI::SsiMono::assemble_mat_scatra() const
{
  // assemble scatra-scatra block into system matrix
  strategy_assemble_->assemble_scatra_scatra(
      ssi_matrices_->system_matrix(), ssi_matrices_->scatra_matrix());

  // assemble scatra-structure block into system matrix
  strategy_assemble_->assemble_scatra_structure(
      ssi_matrices_->system_matrix(), ssi_matrices_->scatra_structure_matrix());
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void SSI::SsiMono::assemble_mat_scatra_manifold() const
{
  // assemble scatra manifold - scatra manifold block into system matrix
  strategy_assemble_->assemble_scatramanifold_scatramanifold(
      ssi_matrices_->system_matrix(), ssi_matrices_->manifold_matrix());

  // assemble scatra manifold-structure block into system matrix
  strategy_assemble_->assemble_scatramanifold_structure(
      ssi_matrices_->system_matrix(), ssi_matrices_->scatra_manifold_structure_matrix());

  // assemble contributions from scatra - scatra manifold coupling: derivs. of scatra side w.r.t.
  // scatra side
  strategy_assemble_->assemble_scatra_scatra(
      ssi_matrices_->system_matrix(), manifoldscatraflux_->system_matrix_scatra());

  // assemble contributions from scatra - scatra manifold coupling: derivs. of manifold side w.r.t.
  // scatra side
  strategy_assemble_->assemble_scatra_scatramanifold(
      ssi_matrices_->system_matrix(), ssi_matrices_->scatra_scatra_manifold_matrix());

  // assemble contributions from scatra - scatra manifold coupling: derivs. of scatra side w.r.t.
  // manifold side
  strategy_assemble_->assemble_scatramanifold_scatra(
      ssi_matrices_->system_matrix(), ssi_matrices_->scatra_manifold_scatra_matrix());

  strategy_assemble_->assemble_scatra_structure(
      ssi_matrices_->system_matrix(), manifoldscatraflux_->matrix_scatra_structure());
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void SSI::SsiMono::assemble_mat_structure() const
{  // assemble structure-scatra block into system matrix
  strategy_assemble_->assemble_structure_scatra(
      ssi_matrices_->system_matrix(), ssi_matrices_->structure_scatra_matrix());

  // assemble structure-structure block into system matrix
  strategy_assemble_->assemble_structure_structure(
      ssi_matrices_->system_matrix(), ssi_matrices_->structure_matrix());
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void SSI::SsiMono::evaluate_subproblems()
{
  TEUCHOS_FUNC_TIME_MONITOR("SSI mono: evaluate sub problems");

  // clear all matrices and residuals from previous Newton iteration
  ssi_matrices_->clear_matrices();
  ssi_vectors_->clear_residuals();

  // evaluate temperature from function and set to structural discretization
  evaluate_and_set_temperature_field();

  // build system matrix and residual for structure field
  structure_field()->evaluate();

  // build system matrix and residual for scalar transport field
  evaluate_scatra();

  // build system matrix and residual for scalar transport field on manifold
  if (is_scatra_manifold()) evaluate_scatra_manifold();

  // build all off diagonal matrices
  evaluate_off_diag_contributions();

  // apply mesh tying to sub problems
  apply_meshtying_to_sub_problems();

  // apply mesh tying to manifold domains
  if (is_scatra_manifold()) apply_manifold_meshtying();

  // apply contact contributions to sub problems
  if (ssi_interface_contact()) apply_contact_to_sub_problems();
}

/*-------------------------------------------------------------------------------*
 *-------------------------------------------------------------------------------*/
void SSI::SsiMono::evaluate_off_diag_contributions() const
{
  // evaluate off-diagonal scatra-structure block (domain contributions) of global system matrix
  scatrastructure_off_diagcoupling_->evaluate_off_diag_block_scatra_structure_domain(
      ssi_matrices_->scatra_structure_matrix());

  // evaluate off-diagonal scatra-structure block (interface contributions) of global system matrix
  if (ssi_interface_meshtying())
    scatrastructure_off_diagcoupling_->evaluate_off_diag_block_scatra_structure_interface(
        *ssi_matrices_->scatra_structure_matrix());

  // evaluate off-diagonal structure-scatra block (we only have domain contributions so far) of
  // global system matrix
  scatrastructure_off_diagcoupling_->evaluate_off_diag_block_structure_scatra_domain(
      ssi_matrices_->structure_scatra_matrix());

  if (is_scatra_manifold())
  {
    // evaluate off-diagonal manifold-structure block of global system matrix
    scatrastructure_off_diagcoupling_->evaluate_off_diag_block_scatra_manifold_structure_domain(
        ssi_matrices_->scatra_manifold_structure_matrix());
  }
}

/*-------------------------------------------------------------------------------*
 *-------------------------------------------------------------------------------*/
void SSI::SsiMono::build_null_spaces() const
{
  switch (scatra_field()->matrix_type())
  {
    case Core::LinAlg::MatrixType::block_condition:
    case Core::LinAlg::MatrixType::block_condition_dof:
    {
      // equip smoother for scatra matrix blocks with null space
      scatra_field()->build_block_null_spaces(
          *solver_, ssi_maps_->get_block_positions(Subproblem::scalar_transport).at(0));
      if (is_scatra_manifold())
      {
        scatra_manifold()->build_block_null_spaces(
            *solver_, ssi_maps_->get_block_positions(Subproblem::manifold).at(0));
      }
      break;
    }

    case Core::LinAlg::MatrixType::sparse:
    {
      std::ostringstream scatrablockstr;
      scatrablockstr << ssi_maps_->get_block_positions(Subproblem::scalar_transport).at(0) + 1;
      Teuchos::ParameterList& blocksmootherparamsscatra =
          solver_->params().sublist("Inverse" + scatrablockstr.str());

      Core::LinearSolver::Parameters::compute_solver_parameters(
          *scatra_field()->discretization(), blocksmootherparamsscatra);

      if (is_scatra_manifold())
      {
        std::ostringstream scatramanifoldblockstr;
        scatramanifoldblockstr << ssi_maps_->get_block_positions(Subproblem::manifold).at(0) + 1;
        Teuchos::ParameterList& blocksmootherparamsscatramanifold =
            solver_->params().sublist("Inverse" + scatramanifoldblockstr.str());

        Core::LinearSolver::Parameters::compute_solver_parameters(
            *scatra_manifold()->discretization(), blocksmootherparamsscatramanifold);
      }

      break;
    }

    default:
    {
      FOUR_C_THROW("Invalid matrix type associated with scalar transport field!");
    }
  }

  // store number of matrix block associated with structural field as string
  std::stringstream iblockstr;
  iblockstr << ssi_maps_->get_block_positions(Subproblem::structure).at(0) + 1;

  Teuchos::ParameterList& blocksmootherparams =
      solver_->params().sublist("Inverse" + iblockstr.str());

  if (solver_->params().isSublist("AMGnxn Parameters"))
  {
    blocksmootherparams.sublist("Belos Parameters");
    blocksmootherparams.sublist("MueLu Parameters");

    Core::FE::compute_null_space_if_necessary(
        *structure_field()->discretization(), blocksmootherparams);
  }
  else
  {
    Core::LinearSolver::Parameters::compute_solver_parameters(
        *structure_field()->discretization(), blocksmootherparams);
  }
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void SSI::SsiMono::complete_subproblem_matrices() const
{
  ssi_matrices_->scatra_matrix()->complete();
  ssi_matrices_->complete_scatra_structure_matrix();
  ssi_matrices_->complete_structure_scatra_matrix();
  ssi_matrices_->structure_matrix()->complete();

  if (is_scatra_manifold())
  {
    ssi_matrices_->manifold_matrix()->complete();
    ssi_matrices_->complete_scatra_manifold_structure_matrix();
    ssi_matrices_->complete_scatra_scatra_manifold_matrix();
    ssi_matrices_->complete_scatra_manifold_scatra_matrix();

    manifoldscatraflux_->complete_matrix_manifold_scatra();
    manifoldscatraflux_->complete_matrix_manifold_structure();
    manifoldscatraflux_->complete_matrix_scatra_manifold();
    manifoldscatraflux_->complete_matrix_scatra_structure();
    manifoldscatraflux_->complete_system_matrix_manifold();
    manifoldscatraflux_->complete_system_matrix_scatra();
  }
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
const std::shared_ptr<const Core::LinAlg::Map>& SSI::SsiMono::dof_row_map() const
{
  return maps_sub_problems()->full_map();
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void SSI::SsiMono::init(MPI_Comm comm, const Teuchos::ParameterList& globaltimeparams,
    const Teuchos::ParameterList& scatraparams, const Teuchos::ParameterList& structparams,
    const std::string& struct_disname, const std::string& scatra_disname, bool isAle)
{
  // check input parameters for scalar transport field
  if (Teuchos::getIntegralValue<Inpar::ScaTra::VelocityField>(scatraparams, "VELOCITYFIELD") !=
      Inpar::ScaTra::velocity_Navier_Stokes)
    FOUR_C_THROW("Invalid type of velocity field for scalar-structure interaction!");

  if (Teuchos::getIntegralValue<Inpar::Solid::DynamicType>(structparams, "DYNAMICTYPE") ==
      Inpar::Solid::DynamicType::Statics)
  {
    FOUR_C_THROW(
        "Mass conservation is not fulfilled if 'Statics' time integration is chosen since the "
        "deformation velocities are incorrectly calculated.\n"
        "Use 'NEGLECTINERTIA Yes' in combination with another time integration scheme instead!");
  }

  // initialize strategy for Newton-Raphson convergence check
  switch (Teuchos::getIntegralValue<SSI::ScaTraTimIntType>(globaltimeparams, "SCATRATIMINTTYPE"))
  {
    case SSI::ScaTraTimIntType::elch:
    {
      if (is_scatra_manifold())
      {
        strategy_convcheck_ =
            std::make_shared<ConvCheckStrategyElchScaTraManifold>(globaltimeparams);
      }
      else
      {
        strategy_convcheck_ = std::make_shared<ConvCheckStrategyElch>(globaltimeparams);
      }
      break;
    }

    case SSI::ScaTraTimIntType::standard:
    {
      strategy_convcheck_ = std::make_shared<ConvCheckStrategyStd>(globaltimeparams);
      break;
    }

    default:
    {
      FOUR_C_THROW("Type of scalar transport time integrator currently not supported!");
    }
  }

  // call base class routine
  SSIBase::init(
      comm, globaltimeparams, scatraparams, structparams, struct_disname, scatra_disname, isAle);
}


/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void SSI::SsiMono::output()
{
  // output scalar transport field
  scatra_field()->check_and_write_output_and_restart();
  if (is_scatra_manifold())
  {
    // domain output
    scatra_manifold()->check_and_write_output_and_restart();
    // coupling output
    if (manifoldscatraflux_->do_output()) manifoldscatraflux_->output();
  }

  // output structure field
  structure_field()->output();

  if (print_matlab_) print_system_matrix_rhs_to_mat_lab_format();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::SsiMono::read_restart(int restart)
{
  // call base class
  SSIBase::read_restart(restart);

  // do ssi contact specific tasks
  if (ssi_interface_contact())
  {
    setup_contact_strategy();
    set_ssi_contact_states(scatra_field()->phinp());
  }
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void SSI::SsiMono::prepare_time_loop()
{
  set_struct_solution(*structure_field()->dispnp(), structure_field()->velnp(),
      is_s2i_kinetics_with_pseudo_contact());

  // calculate initial potential field if needed
  if (do_calculate_initial_potential_field()) calc_initial_potential_field();

  // calculate initial time derivatives
  calc_initial_time_derivative();

  scatra_field()->prepare_time_loop();
  if (is_scatra_manifold()) scatra_manifold()->prepare_time_loop();
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void SSI::SsiMono::prepare_time_step()
{
  // update time and time step
  increment_time_and_step();

  // pass structural degrees of freedom to scalar transport discretization
  set_struct_solution(*structure_field()->dispnp(), structure_field()->velnp(),
      is_s2i_kinetics_with_pseudo_contact());

  // prepare time step for scalar transport field
  scatra_field()->prepare_time_step();
  if (is_scatra_manifold()) scatra_manifold()->prepare_time_step();

  // if adaptive time stepping and different time step size: calculate time step in scatra
  // (prepare_time_step() of Scatra) and pass to other fields
  if (scatra_field()->time_step_adapted()) set_dt_from_scatra_to_ssi();

  // pass scalar transport degrees of freedom to structural discretization
  // has to be called AFTER ScaTraField()->prepare_time_step() to ensure
  // consistent scalar transport state vector with valid Dirichlet conditions
  set_scatra_solution(scatra_field()->phinp());
  if (is_scatra_manifold()) set_scatra_manifold_solution(*scatra_manifold()->phinp());

  // evaluate temperature from function and set to structural discretization
  evaluate_and_set_temperature_field();

  // prepare time step for structural field
  structure_field()->prepare_time_step();

  // structure_field()->prepare_time_step() evaluates the DBC displaements on the master side. Now,
  // the master side displacements are copied to slave side to consider non zero DBC values in the
  // first Newton step on the slave side in case of interface mesh tying
  if (ssi_interface_meshtying())
  {
    for (const auto& meshtying : ssi_structure_mesh_tying()->mesh_tying_handlers())
    {
      auto coupling_adapter = meshtying->slave_master_coupling();
      auto coupling_map_extractor = meshtying->slave_master_extractor();

      // displacements
      coupling_map_extractor->insert_vector(
          *coupling_adapter->master_to_slave(
              *coupling_map_extractor->extract_vector(*structure_field()->dispnp(), 2)),
          1, *structure_field()->write_access_dispnp());
      structure_field()->set_state(structure_field()->write_access_dispnp());
    }
  }

  print_time_step_info();
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void SSI::SsiMono::setup()
{
  // call base class routine
  SSIBase::setup();

  // safety checks
  if (scatra_field()->num_scal() != 1)
  {
    FOUR_C_THROW(
        "Since the ssi_monolithic framework is only implemented for usage in combination with "
        "volume change laws 'MAT_InelasticDefgradLinScalarIso' or "
        "'MAT_InelasticDefgradLinScalarAniso' so far and these laws are implemented for only "
        "one transported scalar at the moment it is not reasonable to use them with more than one "
        "transported scalar. So you need to cope with it or change implementation! ;-)");
  }
  const auto ssi_params = Global::Problem::instance()->ssi_control_params();

  const bool calc_initial_pot_elch =
      Global::Problem::instance()->elch_control_params().get<bool>("INITPOTCALC");
  const bool calc_initial_pot_ssi = ssi_params.sublist("ELCH").get<bool>("INITPOTCALC");

  if (scatra_field()->equilibration_method() != Core::LinAlg::EquilibrationMethod::none)
  {
    FOUR_C_THROW(
        "You are within the monolithic solid scatra interaction framework but activated a pure "
        "scatra equilibration method. Delete this from 'SCALAR TRANSPORT DYNAMIC' section and set "
        "it in 'SSI CONTROL/MONOLITHIC' instead.");
  }
  if (equilibration_method_.global != Core::LinAlg::EquilibrationMethod::local and
      (equilibration_method_.structure != Core::LinAlg::EquilibrationMethod::none or
          equilibration_method_.scatra != Core::LinAlg::EquilibrationMethod::none))
    FOUR_C_THROW("Either global equilibration or local equilibration");

  if (matrixtype_ == Core::LinAlg::MatrixType::sparse and
      (equilibration_method_.structure != Core::LinAlg::EquilibrationMethod::none or
          equilibration_method_.scatra != Core::LinAlg::EquilibrationMethod::none))
    FOUR_C_THROW("Block based equilibration only for block matrices");

  if (not Global::Problem::instance()->scalar_transport_dynamic_params().get<bool>("SKIPINITDER"))
  {
    FOUR_C_THROW(
        "Initial derivatives are already calculated in monolithic SSI. Enable 'SKIPINITDER' in the "
        "input file.");
  }

  if (calc_initial_pot_elch)
    FOUR_C_THROW("Initial potential is calculated by SSI. Disable in Elch section.");
  if (calc_initial_pot_ssi and Teuchos::getIntegralValue<SSI::ScaTraTimIntType>(
                                   ssi_params, "SCATRATIMINTTYPE") != SSI::ScaTraTimIntType::elch)
    FOUR_C_THROW("Calculation of initial potential only in case of Elch");

  if (!scatra_field()->is_incremental())
    FOUR_C_THROW(
        "Must have incremental solution approach for monolithic scalar-structure interaction!");

  if (ssi_interface_meshtying() and
      meshtying_strategy_s2i()->coupling_type() != Inpar::S2I::coupling_matching_nodes)
  {
    FOUR_C_THROW(
        "Monolithic scalar-structure interaction only implemented for scatra-scatra "
        "interface coupling with matching interface nodes!");
  }

  if (ssi_interface_contact() and !is_restart()) setup_contact_strategy();
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void SSI::SsiMono::setup_system()
{
  SSIBase::setup_system();

  // setup the ssi maps object
  ssi_maps_ = std::make_shared<Utils::SSIMaps>(*this);

  // perform initializations associated with global system matrix
  switch (matrixtype_)
  {
    case Core::LinAlg::MatrixType::block_field:
    {
      // feed block preconditioner with null space information for each block of global block system
      // matrix
      build_null_spaces();

      break;
    }

    case Core::LinAlg::MatrixType::sparse:
    {
      // safety check
      if (scatra_field()->system_matrix() == nullptr)
        FOUR_C_THROW("Incompatible matrix type associated with scalar transport field!");
      break;
    }

    default:
    {
      FOUR_C_THROW("Type of global system matrix for scalar-structure interaction not recognized!");
    }
  }

  // initialize sub blocks and system matrix
  ssi_matrices_ = std::make_shared<Utils::SSIMatrices>(
      *ssi_maps_, matrixtype_, scatra_field()->matrix_type(), is_scatra_manifold());

  // initialize residual and increment vectors
  ssi_vectors_ = std::make_shared<Utils::SSIVectors>(*ssi_maps_, is_scatra_manifold());

  // initialize strategy for assembly
  strategy_assemble_ = build_assemble_strategy(
      ssi_maps_, is_scatra_manifold(), matrixtype_, scatra_field()->matrix_type());

  if (is_scatra_manifold())
  {
    // initialize object, that performs evaluations of OD coupling
    scatrastructure_off_diagcoupling_ = std::make_shared<ScatraManifoldStructureOffDiagCoupling>(
        block_map_structure(), ssi_maps()->structure_dof_row_map(), ssi_structure_mesh_tying(),
        meshtying_strategy_s2i(), scatra_field(), scatra_manifold(), structure_field());

    // initialize object, that performs evaluations of scatra - scatra on manifold coupling
    manifoldscatraflux_ = std::make_shared<ScaTraManifoldScaTraFluxEvaluator>(*this);

    // initialize object, that performs meshtying between manifold domains
    strategy_manifold_meshtying_ =
        SSI::build_manifold_mesh_tying_strategy(scatra_manifold()->discretization(), ssi_maps_,
            is_scatra_manifold_meshtying(), scatra_manifold()->matrix_type());
  }
  else
  {
    scatrastructure_off_diagcoupling_ = std::make_shared<SSI::ScatraStructureOffDiagCoupling>(
        block_map_structure(), ssi_maps()->structure_dof_row_map(), ssi_structure_mesh_tying(),
        meshtying_strategy_s2i(), scatra_field(), structure_field());
  }
  // instantiate appropriate equilibration class
  strategy_equilibration_ =
      build_equilibration(matrixtype_, get_block_equilibration(), maps_sub_problems()->full_map());

  // instantiate appropriate contact class
  strategy_contact_ =
      build_contact_strategy(nitsche_strategy_ssi(), ssi_maps_, scatra_field()->matrix_type());

  // instantiate appropriate mesh tying class
  strategy_meshtying_ =
      build_meshtying_strategy(is_scatra_manifold(), scatra_field()->matrix_type(), *ssi_maps());

  // instantiate Dirichlet boundary condition handler class
  dbc_handler_ = build_dbc_handler(is_scatra_manifold(), matrixtype_, scatra_field(),
      is_scatra_manifold() ? scatra_manifold() : nullptr, ssi_maps(), structure_field());
}

/*---------------------------------------------------------------------------------*
 *---------------------------------------------------------------------------------*/
void SSI::SsiMono::solve_linear_system() const
{
  TEUCHOS_FUNC_TIME_MONITOR("SSI mono: solve linear system");
  strategy_equilibration_->equilibrate_system(
      ssi_matrices_->system_matrix(), ssi_vectors_->residual(), block_map_system_matrix());

  // solve global system of equations
  // Dirichlet boundary conditions have already been applied to global system of equations
  Core::LinAlg::SolverParams solver_params;
  solver_params.refactor = true;
  solver_params.reset = iteration_count() == 1;
  if (relax_lin_solver_iter_step_ > 0)
  {
    solver_->reset_tolerance();
    if (iteration_count() <= relax_lin_solver_iter_step_)
    {
      solver_params.tolerance = solver_->get_tolerance() * relax_lin_solver_tolerance_;
    }
  }
  solver_->solve(ssi_matrices_->system_matrix(), ssi_vectors_->increment(),
      ssi_vectors_->residual(), solver_params);

  strategy_equilibration_->unequilibrate_increment(ssi_vectors_->increment());
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void SSI::SsiMono::newton_loop()
{
  TEUCHOS_FUNC_TIME_MONITOR("SSI mono: solve Newton loop");
  // reset counter for Newton-Raphson iteration
  reset_iteration_count();

  // start Newton-Raphson iteration
  while (true)
  {
    // update iteration counter
    increment_iteration_count();

    timer_->reset();

    // store time before evaluating elements and assembling global system of equations
    const double time_before_evaluate = timer_->wallTime();

    // set solution from last Newton step to all fields
    distribute_solution_all_fields();

    // evaluate sub problems and get all matrices and right-hand-sides
    evaluate_subproblems();

    // complete the sub problem matrices
    complete_subproblem_matrices();

    // assemble global system of equations
    assemble_mat_and_rhs();

    // apply the Dirichlet boundary conditions to global system
    apply_dbc_to_system();

    // time needed for evaluating elements and assembling global system of equations
    double my_evaluation_time = timer_->wallTime() - time_before_evaluate;
    dt_eval_ = Core::Communication::max_all(my_evaluation_time, get_comm());

    // safety check
    if (!ssi_matrices_->system_matrix()->filled())
      FOUR_C_THROW("Complete() has not been called on global system matrix yet!");

    // check termination criterion for Newton-Raphson iteration
    if (strategy_convcheck_->exit_newton_raphson(*this)) break;

    // clear the global increment vector
    ssi_vectors_->clear_increment();

    // store time before solving global system of equations
    const double time_before_solving = timer_->wallTime();

    solve_linear_system();

    // time needed for solving global system of equations
    double my_solve_time = timer_->wallTime() - time_before_solving;
    dt_solve_ = Core::Communication::max_all(my_solve_time, get_comm());

    // output performance statistics associated with linear solver into text file if
    // applicable
    if (scatra_field()->scatra_parameter_list()->get<bool>("OUTPUTLINSOLVERSTATS"))
      scatra_field()->output_lin_solver_stats(*solver_, dt_solve_, step(), iteration_count(),
          ssi_vectors_->residual()->get_map().num_global_elements());

    // update states for next Newton iteration
    update_iter_scatra();
    update_iter_structure();
  }
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void SSI::SsiMono::timeloop()
{
  if (step() == 0) prepare_time_loop();

  // time loop
  while (not_finished() and scatra_field()->not_finished())
  {
    TEUCHOS_FUNC_TIME_MONITOR("SSI mono: solve time step");
    // prepare time step
    prepare_time_step();

    // store time before calling nonlinear solver
    const double time = timer_->wallTime();

    // evaluate time step
    newton_loop();

    // determine time spent by nonlinear solver and take maximum over all processors via
    // communication
    double mydtnonlinsolve(timer_->wallTime() - time), dtnonlinsolve(0.);
    dtnonlinsolve = Core::Communication::max_all(mydtnonlinsolve, get_comm());

    // output performance statistics associated with nonlinear solver into *.csv file if
    // applicable
    if (scatra_field()->scatra_parameter_list()->get<bool>("OUTPUTNONLINSOLVERSTATS"))
      scatra_field()->output_nonlin_solver_stats(
          iteration_count(), dtnonlinsolve, step(), get_comm());

    prepare_output();

    // update scalar transport and structure fields
    update();

    // output solution to screen and files
    output();
  }
  strategy_convcheck_->print_non_converged_steps(Core::Communication::my_mpi_rank(get_comm()));
}

/*--------------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------------*/
void SSI::SsiMono::update()
{
  // update scalar transport field
  scatra_field()->update();
  if (is_scatra_manifold()) scatra_manifold()->update();

  // update structure field
  structure_field()->update();
}

/*--------------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------------*/
void SSI::SsiMono::update_iter_scatra() const
{
  // update scalar transport field
  scatra_field()->update_iter(*maps_sub_problems()->extract_vector(*ssi_vectors_->increment(),
      Utils::SSIMaps::get_problem_position(Subproblem::scalar_transport)));
  scatra_field()->compute_intermediate_values();

  if (is_scatra_manifold())
  {
    auto increment_manifold = maps_sub_problems()->extract_vector(
        *ssi_vectors_->increment(), Utils::SSIMaps::get_problem_position(Subproblem::manifold));

    // reconstruct slave side solution from master side
    if (is_scatra_manifold_meshtying())
    {
      for (const auto& meshtying :
          strategy_manifold_meshtying_->ssi_mesh_tying()->mesh_tying_handlers())
      {
        auto coupling_adapter = meshtying->slave_master_coupling();
        auto multimap = meshtying->slave_master_extractor();

        auto master_dofs = multimap->extract_vector(*increment_manifold, 2);
        auto master_dofs_to_slave = coupling_adapter->master_to_slave(*master_dofs);
        multimap->insert_vector(*master_dofs_to_slave, 1, *increment_manifold);
      }
    }

    scatra_manifold()->update_iter(*increment_manifold);
    scatra_manifold()->compute_intermediate_values();
  }
}

/*--------------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------------*/
void SSI::SsiMono::update_iter_structure() const
{
  // set up structural increment vector
  const std::shared_ptr<Core::LinAlg::Vector<double>> increment_structure =
      maps_sub_problems()->extract_vector(
          *ssi_vectors_->increment(), Utils::SSIMaps::get_problem_position(Subproblem::structure));

  // consider structural meshtying. Copy master increments and displacements to slave side.
  if (ssi_interface_meshtying())
  {
    for (const auto& meshtying : ssi_structure_mesh_tying()->mesh_tying_handlers())
    {
      auto coupling_adapter = meshtying->slave_master_coupling();
      auto coupling_map_extractor = meshtying->slave_master_extractor();

      // displacements
      coupling_map_extractor->insert_vector(
          *coupling_adapter->master_to_slave(
              *coupling_map_extractor->extract_vector(*structure_field()->dispnp(), 2)),
          1, *structure_field()->write_access_dispnp());
      structure_field()->set_state(structure_field()->write_access_dispnp());

      // increment
      coupling_map_extractor->insert_vector(
          *coupling_adapter->master_to_slave(
              *coupling_map_extractor->extract_vector(*increment_structure, 2)),
          1, *increment_structure);
    }
  }

  // update displacement of structure field
  structure_field()->update_state_incrementally(increment_structure);
}

/*--------------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------------*/
std::vector<Core::LinAlg::EquilibrationMethod> SSI::SsiMono::get_block_equilibration() const
{
  std::vector<Core::LinAlg::EquilibrationMethod> equilibration_method_vector;
  switch (matrixtype_)
  {
    case Core::LinAlg::MatrixType::sparse:
    {
      equilibration_method_vector = std::vector(1, equilibration_method_.global);
      break;
    }
    case Core::LinAlg::MatrixType::block_field:
    {
      if (equilibration_method_.global != Core::LinAlg::EquilibrationMethod::local)
      {
        equilibration_method_vector = std::vector(1, equilibration_method_.global);
      }
      else if (equilibration_method_.structure == Core::LinAlg::EquilibrationMethod::none and
               equilibration_method_.scatra == Core::LinAlg::EquilibrationMethod::none)
      {
        equilibration_method_vector = std::vector(1, Core::LinAlg::EquilibrationMethod::none);
      }
      else
      {
        auto block_positions_scatra = ssi_maps_->get_block_positions(Subproblem::scalar_transport);
        auto block_position_structure = ssi_maps_->get_block_positions(Subproblem::structure);
        if (is_scatra_manifold())
        {
          auto block_positions_scatra_manifold =
              ssi_maps_->get_block_positions(Subproblem::manifold);

          equilibration_method_vector =
              std::vector(block_positions_scatra.size() + block_position_structure.size() +
                              block_positions_scatra_manifold.size(),
                  Core::LinAlg::EquilibrationMethod::none);
        }
        else
        {
          equilibration_method_vector =
              std::vector(block_positions_scatra.size() + block_position_structure.size(),
                  Core::LinAlg::EquilibrationMethod::none);
        }


        for (const int block_position_scatra : block_positions_scatra)
          equilibration_method_vector.at(block_position_scatra) = equilibration_method_.scatra;

        equilibration_method_vector.at(block_position_structure.at(0)) =
            equilibration_method_.structure;

        if (is_scatra_manifold())
        {
          for (const int block_position_scatra_manifold :
              ssi_maps_->get_block_positions(Subproblem::manifold))
          {
            equilibration_method_vector.at(block_position_scatra_manifold) =
                equilibration_method_.scatra;
          }
        }
      }

      break;
    }
    default:
    {
      FOUR_C_THROW("Invalid matrix type associated with system matrix field!");
    }
  }
  return equilibration_method_vector;
}

/*--------------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------------*/
void SSI::SsiMono::evaluate_scatra() const
{
  // evaluate the scatra field
  scatra_field()->prepare_linear_solve();

  // copy the matrix to the corresponding ssi matrix and complete it such that additional
  // contributions like contact contributions can be added before assembly
  ssi_matrices_->scatra_matrix()->add(*scatra_field()->system_matrix_operator(), false, 1.0, 1.0);

  // copy the residual to the corresponding ssi vector to enable application of contact
  // contributions before assembly
  ssi_vectors_->scatra_residual()->update(1.0, *scatra_field()->residual(), 1.0);
}

/*--------------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------------*/
void SSI::SsiMono::evaluate_scatra_manifold() const
{
  // evaluate single problem
  scatra_manifold()->prepare_linear_solve();

  ssi_vectors_->manifold_residual()->update(1.0, *scatra_manifold()->residual(), 1.0);

  // evaluate coupling fluxes
  manifoldscatraflux_->evaluate();

  ssi_vectors_->manifold_residual()->update(1.0, *manifoldscatraflux_->rhs_manifold(), 1.0);
  ssi_vectors_->scatra_residual()->update(1.0, *manifoldscatraflux_->rhs_scatra(), 1.0);
}

/*--------------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------------*/
void SSI::SsiMono::prepare_output()
{
  constexpr bool force_prepare = false;
  structure_field()->prepare_output(force_prepare);

  // prepare output of coupling sctra manifold - scatra
  if (is_scatra_manifold() and manifoldscatraflux_->do_output())
  {
    distribute_solution_all_fields();
    manifoldscatraflux_->evaluate_scatra_manifold_inflow();
  }
}

/*--------------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------------*/
void SSI::SsiMono::distribute_solution_all_fields(const bool restore_velocity)
{
  // has to be called before the call of 'set_struct_solution()' to have updated stress/strain
  // states
  if (is_s2i_kinetics_with_pseudo_contact()) structure_field()->determine_stress_strain();

  // clear all states before redistributing the new states
  structure_field()->discretization()->clear_state(true);
  scatra_field()->discretization()->clear_state(true);
  if (is_scatra_manifold()) scatra_manifold()->discretization()->clear_state(true);

  // needed to communicate to NOX state
  if (restore_velocity)
  {
    auto vel_temp = *structure_field()->velnp();
    structure_field()->set_state(structure_field()->write_access_dispnp());
    structure_field()->write_access_velnp()->update(1.0, vel_temp, 0.0);
  }
  else
  {
    structure_field()->set_state(structure_field()->write_access_dispnp());
  }

  // distribute states to other fields
  set_struct_solution(*structure_field()->dispnp(), structure_field()->velnp(),
      is_s2i_kinetics_with_pseudo_contact());
  set_scatra_solution(scatra_field()->phinp());
  if (is_scatra_manifold()) set_scatra_manifold_solution(*scatra_manifold()->phinp());
}

/*--------------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------------*/
void SSI::SsiMono::calc_initial_potential_field()
{
  const auto equpot = Teuchos::getIntegralValue<ElCh::EquPot>(
      Global::Problem::instance()->elch_control_params(), "EQUPOT");
  if (equpot != ElCh::equpot_divi and equpot != ElCh::equpot_enc_pde and
      equpot != ElCh::equpot_enc_pde_elim)
  {
    FOUR_C_THROW(
        "Initial potential field cannot be computed for chosen closing equation for electric "
        "potential!");
  }

  // store initial velocity to restore them afterwards
  auto init_velocity = *structure_field()->velnp();

  // cast scatra time integrators to elch to call elch specific methods
  auto scatra_elch = std::dynamic_pointer_cast<ScaTra::ScaTraTimIntElch>(scatra_field());
  auto manifold_elch = is_scatra_manifold()
                           ? std::dynamic_pointer_cast<ScaTra::ScaTraTimIntElch>(scatra_manifold())
                           : nullptr;
  if (scatra_elch == nullptr or (is_scatra_manifold() and manifold_elch == nullptr))
    FOUR_C_THROW("Cast to Elch time integrator failed. Scatra is not an Elch problem");

  // prepare specific time integrators
  scatra_elch->pre_calc_initial_potential_field();
  if (is_scatra_manifold()) manifold_elch->pre_calc_initial_potential_field();

  auto scatra_elch_splitter = scatra_field()->splitter();
  auto manifold_elch_splitter = is_scatra_manifold() ? scatra_manifold()->splitter() : nullptr;

  reset_iteration_count();

  while (true)
  {
    increment_iteration_count();

    timer_->reset();

    // store time before evaluating elements and assembling global system of equations
    const double time_before_evaluate = timer_->wallTime();

    // prepare full SSI system
    distribute_solution_all_fields(true);
    evaluate_subproblems();

    // complete the sub problem matrices
    complete_subproblem_matrices();

    assemble_mat_and_rhs();
    apply_dbc_to_system();

    // apply artificial Dirichlet boundary conditions to system of equations (on concentration
    // dofs and on structure dofs)
    std::shared_ptr<Core::LinAlg::Map> pseudo_dbc_map;
    if (is_scatra_manifold())
    {
      auto conc_map = Core::LinAlg::merge_map(
          scatra_elch_splitter->other_map(), manifold_elch_splitter->other_map());
      pseudo_dbc_map = Core::LinAlg::merge_map(conc_map, structure_field()->dof_row_map());
    }
    else
    {
      pseudo_dbc_map = Core::LinAlg::merge_map(
          scatra_elch_splitter->other_map(), structure_field()->dof_row_map());
    }

    Core::LinAlg::Vector<double> dbc_zeros(*pseudo_dbc_map, true);

    auto rhs = ssi_vectors_->residual();
    apply_dirichlet_to_system(*ssi_matrices_->system_matrix(), *ssi_vectors_->increment(), *rhs,
        dbc_zeros, *pseudo_dbc_map);
    ssi_vectors_->residual()->update(1.0, *rhs, 0.0);

    // time needed for evaluating elements and assembling global system of equations
    double my_evaluation_time = timer_->wallTime() - time_before_evaluate;
    dt_eval_ = Core::Communication::max_all(my_evaluation_time, get_comm());

    if (strategy_convcheck_->exit_newton_raphson_init_pot_calc(*this)) break;

    // solve for potential increments
    ssi_vectors_->clear_increment();

    // store time before solving global system of equations
    const double time_before_solving = timer_->wallTime();

    solve_linear_system();

    // time needed for solving global system of equations
    double my_solve_time = timer_->wallTime() - time_before_solving;
    dt_solve_ = Core::Communication::max_all(my_solve_time, get_comm());

    // update potential dofs in scatra and manifold fields
    update_iter_scatra();

    // copy initial state vector
    scatra_field()->phin()->update(1.0, *scatra_field()->phinp(), 0.0);
    if (is_scatra_manifold())
      scatra_manifold()->phin()->update(1.0, *scatra_manifold()->phinp(), 0.0);

    // update state vectors for intermediate time steps (only for generalized alpha)
    scatra_field()->compute_intermediate_values();
    if (is_scatra_manifold()) scatra_manifold()->compute_intermediate_values();
  }

  scatra_elch->post_calc_initial_potential_field();
  if (is_scatra_manifold()) manifold_elch->post_calc_initial_potential_field();

  structure_field()->write_access_velnp()->update(1.0, init_velocity, 0.0);
}

/*--------------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------------*/
void SSI::SsiMono::calc_initial_time_derivative()
{
  // store initial velocity to restore them afterwards
  auto init_velocity = *structure_field()->velnp();

  const bool is_elch = is_elch_scatra_time_int_type();

  // prepare specific time integrators
  scatra_field()->pre_calc_initial_time_derivative();
  if (is_scatra_manifold()) scatra_manifold()->pre_calc_initial_time_derivative();

  auto scatra_elch_splitter = is_elch ? scatra_field()->splitter() : nullptr;
  auto manifold_elch_splitter =
      (is_elch and is_scatra_manifold()) ? scatra_manifold()->splitter() : nullptr;

  // initial screen output
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    std::cout << "Calculating initial time derivative of state variables on discretization "
              << scatra_field()->discretization()->name();
    if (is_scatra_manifold())
      std::cout << " and discretization " << scatra_manifold()->discretization()->name();
    std::cout << '\n';
  }

  // evaluate Dirichlet and Neumann boundary conditions
  scatra_field()->apply_bc_to_system();
  if (is_scatra_manifold()) scatra_manifold()->apply_bc_to_system();

  // clear history values (this is the first step)
  scatra_field()->hist()->put_scalar(0.0);
  if (is_scatra_manifold()) scatra_manifold()->hist()->put_scalar(0.0);

  // In a first step, we assemble the standard global system of equations (we need the residual)
  distribute_solution_all_fields(true);
  evaluate_subproblems();

  // complete the sub problem matrices
  complete_subproblem_matrices();

  assemble_mat_and_rhs();
  apply_dbc_to_system();

  // prepare mass matrices of sub problems and global system
  auto massmatrix_scatra =
      scatra_field()->matrix_type() == Core::LinAlg::MatrixType::sparse
          ? std::dynamic_pointer_cast<Core::LinAlg::SparseOperator>(
                Utils::SSIMatrices::setup_sparse_matrix(*scatra_field()->dof_row_map()))
          : std::dynamic_pointer_cast<Core::LinAlg::SparseOperator>(
                Utils::SSIMatrices::setup_block_matrix(
                    *scatra_field()->dof_block_maps(), *scatra_field()->dof_block_maps()));

  auto massmatrix_manifold =
      is_scatra_manifold() ? (scatra_manifold()->matrix_type() == Core::LinAlg::MatrixType::sparse
                                     ? std::dynamic_pointer_cast<Core::LinAlg::SparseOperator>(
                                           Utils::SSIMatrices::setup_sparse_matrix(
                                               *scatra_manifold()->dof_row_map()))
                                     : std::dynamic_pointer_cast<Core::LinAlg::SparseOperator>(
                                           Utils::SSIMatrices::setup_block_matrix(
                                               *scatra_manifold()->dof_block_maps(),
                                               *scatra_manifold()->dof_block_maps())))
                           : nullptr;

  auto massmatrix_system = matrix_type() == Core::LinAlg::MatrixType::sparse
                               ? std::dynamic_pointer_cast<Core::LinAlg::SparseOperator>(
                                     Utils::SSIMatrices::setup_sparse_matrix(*dof_row_map()))
                               : std::dynamic_pointer_cast<Core::LinAlg::SparseOperator>(
                                     Utils::SSIMatrices::setup_block_matrix(
                                         *block_map_system_matrix(), *block_map_system_matrix()));

  // fill ones on main diag of structure block (not solved)
  auto ones_struct =
      std::make_shared<Core::LinAlg::Vector<double>>(*structure_field()->dof_row_map(), true);
  ones_struct->put_scalar(1.0);
  matrix_type() == Core::LinAlg::MatrixType::sparse
      ? insert_my_row_diagonal_into_unfilled_matrix(
            *cast_to_sparse_matrix_and_check_success(massmatrix_system), *ones_struct)
      : insert_my_row_diagonal_into_unfilled_matrix(
            cast_to_block_sparse_matrix_base_and_check_success(massmatrix_system)
                ->matrix(ssi_maps_->get_block_positions(Subproblem::structure).at(0),
                    ssi_maps_->get_block_positions(Subproblem::structure).at(0)),
            *ones_struct);

  // extract residuals of scatra and manifold from global residual
  auto rhs_scatra =
      std::make_shared<Core::LinAlg::Vector<double>>(*scatra_field()->dof_row_map(), true);
  auto rhs_manifold = is_scatra_manifold() ? std::make_shared<Core::LinAlg::Vector<double>>(
                                                 *scatra_manifold()->dof_row_map(), true)
                                           : nullptr;

  rhs_scatra->update(1.0,
      *maps_sub_problems()->extract_vector(*ssi_vectors_->residual(),
          Utils::SSIMaps::get_problem_position(Subproblem::scalar_transport)),
      0.0);
  if (is_scatra_manifold())
  {
    rhs_manifold->update(1.0,
        *maps_sub_problems()->extract_vector(
            *ssi_vectors_->residual(), Utils::SSIMaps::get_problem_position(Subproblem::manifold)),
        0.0);
  }

  // In a second step, we need to modify the assembled system of equations, since we want to solve
  // M phidt^0 = f^n - K\phi^n - C(u_n)\phi^n
  // In particular, we need to replace the global system matrix by a global mass matrix,
  // and we need to remove all transient contributions associated with time discretization from the
  // global residual vector.

  // Evaluate mass matrix and modify residual
  scatra_field()->evaluate_initial_time_derivative(massmatrix_scatra, rhs_scatra);
  if (is_scatra_manifold())
    scatra_manifold()->evaluate_initial_time_derivative(massmatrix_manifold, rhs_manifold);

  // assemble global mass matrix from
  switch (matrix_type())
  {
    case Core::LinAlg::MatrixType::block_field:
    {
      switch (scatra_field()->matrix_type())
      {
        case Core::LinAlg::MatrixType::block_condition:
        case Core::LinAlg::MatrixType::block_condition_dof:
        {
          auto massmatrix_system_block =
              cast_to_block_sparse_matrix_base_and_check_success(massmatrix_system);

          auto massmatrix_scatra_block =
              cast_to_block_sparse_matrix_base_and_check_success(massmatrix_scatra);

          auto positions_scatra = ssi_maps_->get_block_positions(Subproblem::scalar_transport);

          for (int i = 0; i < static_cast<int>(positions_scatra.size()); ++i)
          {
            const int position_scatra = positions_scatra.at(i);
            Core::LinAlg::matrix_add(massmatrix_scatra_block->matrix(i, i), false, 1.0,
                massmatrix_system_block->matrix(position_scatra, position_scatra), 1.0);
          }
          if (is_scatra_manifold())
          {
            auto positions_manifold = ssi_maps_->get_block_positions(Subproblem::manifold);

            auto massmatrix_manifold_block =
                cast_to_block_sparse_matrix_base_and_check_success(massmatrix_manifold);

            for (int i = 0; i < static_cast<int>(positions_manifold.size()); ++i)
            {
              const int position_manifold = positions_manifold.at(i);
              Core::LinAlg::matrix_add(massmatrix_manifold_block->matrix(i, i), false, 1.0,
                  massmatrix_system_block->matrix(position_manifold, position_manifold), 1.0);
            }
          }

          break;
        }

        case Core::LinAlg::MatrixType::sparse:
        {
          auto massmatrix_system_block =
              cast_to_block_sparse_matrix_base_and_check_success(massmatrix_system);

          const int position_scatra =
              ssi_maps_->get_block_positions(Subproblem::scalar_transport).at(0);

          Core::LinAlg::matrix_add(*cast_to_sparse_matrix_and_check_success(massmatrix_scatra),
              false, 1.0, massmatrix_system_block->matrix(position_scatra, position_scatra), 1.0);

          if (is_scatra_manifold())
          {
            const int position_manifold =
                ssi_maps_->get_block_positions(Subproblem::manifold).at(0);

            Core::LinAlg::matrix_add(*cast_to_sparse_matrix_and_check_success(massmatrix_manifold),
                false, 1.0, massmatrix_system_block->matrix(position_manifold, position_manifold),
                1.0);
          }
          break;
        }

        default:
        {
          FOUR_C_THROW("Invalid matrix type associated with scalar transport field!");
        }
      }
      massmatrix_system->complete();
      break;
    }
    case Core::LinAlg::MatrixType::sparse:
    {
      auto massmatrix_system_sparse = cast_to_sparse_matrix_and_check_success(massmatrix_system);
      Core::LinAlg::matrix_add(*cast_to_sparse_matrix_and_check_success(massmatrix_scatra), false,
          1.0, *massmatrix_system_sparse, 1.0);

      if (is_scatra_manifold())
      {
        Core::LinAlg::matrix_add(*cast_to_sparse_matrix_and_check_success(massmatrix_manifold),
            false, 1.0, *massmatrix_system_sparse, 1.0);
      }

      massmatrix_system->complete(*dof_row_map(), *dof_row_map());
      break;
    }
    default:
    {
      FOUR_C_THROW("Type of global system matrix for scalar-structure interaction not recognized!");
    }
  }

  // reconstruct global residual from partial residuals
  auto rhs_system = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map(), true);
  maps_sub_problems()->insert_vector(
      *rhs_scatra, Utils::SSIMaps::get_problem_position(Subproblem::scalar_transport), *rhs_system);
  if (is_scatra_manifold())
    maps_sub_problems()->insert_vector(
        *rhs_manifold, Utils::SSIMaps::get_problem_position(Subproblem::manifold), *rhs_system);

  // apply artificial Dirichlet boundary conditions to system of equations to non-transported
  // scalars and structure
  std::shared_ptr<Core::LinAlg::Map> pseudo_dbc_map;
  if (is_scatra_manifold() and is_elch)
  {
    auto conc_map = merge_map(scatra_elch_splitter->cond_map(), manifold_elch_splitter->cond_map());
    pseudo_dbc_map = merge_map(conc_map, structure_field()->dof_row_map());
  }
  else if (is_elch)
  {
    pseudo_dbc_map = merge_map(scatra_elch_splitter->cond_map(), structure_field()->dof_row_map());
  }
  else
  {
    pseudo_dbc_map = std::make_shared<Core::LinAlg::Map>(*structure_field()->dof_row_map());
  }

  Core::LinAlg::Vector<double> dbc_zeros(*pseudo_dbc_map, true);

  // temporal derivative of transported scalars
  auto phidtnp_system = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map(), true);
  apply_dirichlet_to_system(
      *massmatrix_system, *phidtnp_system, *rhs_system, dbc_zeros, *(pseudo_dbc_map));

  // solve global system of equations for initial time derivative of state variables
  Core::LinAlg::SolverParams solver_params;
  solver_params.refactor = true;
  solver_params.reset = true;
  solver_->solve(massmatrix_system, phidtnp_system, rhs_system, solver_params);

  // copy solution to sub problems
  auto phidtnp_scatra = maps_sub_problems()->extract_vector(
      *phidtnp_system, Utils::SSIMaps::get_problem_position(Subproblem::scalar_transport));
  scatra_field()->phidtnp()->update(1.0, *phidtnp_scatra, 0.0);
  scatra_field()->phidtn()->update(1.0, *phidtnp_scatra, 0.0);

  if (is_scatra_manifold())
  {
    auto phidtnp_manifold = maps_sub_problems()->extract_vector(
        *phidtnp_system, Utils::SSIMaps::get_problem_position(Subproblem::manifold));
    scatra_manifold()->phidtnp()->update(1.0, *phidtnp_manifold, 0.0);
    scatra_manifold()->phidtn()->update(1.0, *phidtnp_manifold, 0.0);
  }

  // reset solver
  solver_->reset();

  scatra_field()->post_calc_initial_time_derivative();
  if (is_scatra_manifold()) scatra_manifold()->post_calc_initial_time_derivative();

  structure_field()->write_access_velnp()->update(1.0, init_velocity, 0.0);
}

/*--------------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::MultiMapExtractor> SSI::SsiMono::maps_sub_problems() const
{
  return ssi_maps_->maps_sub_problems();
}

/*--------------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::MultiMapExtractor> SSI::SsiMono::block_map_scatra() const
{
  return ssi_maps_->block_map_scatra();
}

/*--------------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::MultiMapExtractor> SSI::SsiMono::block_map_scatra_manifold()
    const
{
  return ssi_maps_->block_map_scatra_manifold();
}

/*--------------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::MultiMapExtractor> SSI::SsiMono::block_map_structure() const
{
  return ssi_maps_->block_map_structure();
}

/*--------------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::MultiMapExtractor> SSI::SsiMono::block_map_system_matrix() const
{
  return ssi_maps_->block_map_system_matrix();
}

/*--------------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------------*/
void SSI::SsiMono::print_time_step_info() const
{
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    std::cout << '\n'
              << "TIME: " << std::setw(11) << std::setprecision(4) << std::scientific << time()
              << "/" << max_time() << "  DT = " << dt() << "  STEP = " << step() << "/" << n_step()
              << '\n';
  }
}

/*--------------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------------*/
void SSI::SsiMono::print_system_matrix_rhs_to_mat_lab_format() const
{
  // print system matrix
  switch (matrixtype_)
  {
    case Core::LinAlg::MatrixType::block_field:
    {
      auto block_matrix =
          cast_to_const_block_sparse_matrix_base_and_check_success(ssi_matrices_->system_matrix());

      for (int row = 0; row < block_matrix->rows(); ++row)
      {
        for (int col = 0; col < block_matrix->cols(); ++col)
        {
          std::ostringstream filename;
          filename << Global::Problem::instance()->output_control_file()->file_name()
                   << "_block_system_matrix_" << row << "_" << col << ".csv";

          Core::LinAlg::print_matrix_in_matlab_format(
              filename.str(), block_matrix->matrix(row, col), true);
        }
      }
      break;
    }

    case Core::LinAlg::MatrixType::sparse:
    {
      auto sparse_matrix =
          cast_to_const_sparse_matrix_and_check_success(ssi_matrices_->system_matrix());

      const std::string filename = Global::Problem::instance()->output_control_file()->file_name() +
                                   "_sparse_system_matrix.csv";

      Core::LinAlg::print_matrix_in_matlab_format(filename, *sparse_matrix, true);
      break;
    }

    default:
    {
      FOUR_C_THROW("Type of global system matrix for scalar-structure interaction not recognized!");
    }
  }

  // print rhs
  {
    const std::string filename =
        Global::Problem::instance()->output_control_file()->file_name() + "_system_vector.csv";
    print_vector_in_matlab_format(filename, *ssi_vectors_->residual(), true);
  }

  // print full map
  {
    const std::string filename =
        Global::Problem::instance()->output_control_file()->file_name() + "_full_map.csv";
    print_map_in_matlab_format(filename, *ssi_maps_->map_system_matrix(), true);
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SsiMono::set_scatra_manifold_solution(const Core::LinAlg::Vector<double>& phi) const
{
  // scatra values on master side copied to manifold
  auto manifold_on_scatra = std::make_shared<Core::LinAlg::Vector<double>>(
      *scatra_field()->discretization()->dof_row_map(), true);

  for (const auto& coup : manifoldscatraflux_->scatra_manifold_couplings())
  {
    auto manifold_cond = coup->manifold_map_extractor()->extract_cond_vector(phi);
    auto manifold_on_scatra_cond = coup->coupling_adapter()->slave_to_master(*manifold_cond);
    coup->scatra_map_extractor()->insert_cond_vector(*manifold_on_scatra_cond, *manifold_on_scatra);
  }
  scatra_field()->discretization()->set_state(0, "manifold_on_scatra", *manifold_on_scatra);
}
FOUR_C_NAMESPACE_CLOSE

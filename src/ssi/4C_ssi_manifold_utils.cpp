// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_ssi_manifold_utils.hpp"

#include "4C_adapter_scatra_base_algorithm.hpp"
#include "4C_comm_utils_gid_vector.hpp"
#include "4C_coupling_adapter.hpp"
#include "4C_coupling_adapter_converter.hpp"
#include "4C_fem_condition_utils.hpp"
#include "4C_fem_general_assemblestrategy.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_s2i.hpp"
#include "4C_io_runtime_csv_writer.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_scatra_ele_action.hpp"
#include "4C_scatra_ele_parameter_boundary.hpp"
#include "4C_scatra_timint_implicit.hpp"
#include "4C_ssi_input.hpp"
#include "4C_ssi_monolithic.hpp"
#include "4C_ssi_utils.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN


/*---------------------------------------------------------------------------------*
 *---------------------------------------------------------------------------------*/
SSI::ManifoldScaTraCoupling::ManifoldScaTraCoupling(
    std::shared_ptr<Core::FE::Discretization> manifolddis,
    std::shared_ptr<Core::FE::Discretization> scatradis,
    const Core::Conditions::Condition* condition_manifold,
    const Core::Conditions::Condition* condition_kinetics, const int ndof_per_node)
    : condition_kinetics_(condition_kinetics),
      condition_manifold_(condition_manifold),
      coupling_adapter_(std::make_shared<Coupling::Adapter::Coupling>()),
      inv_thickness_(1.0 / condition_manifold->parameters().get<double>("thickness")),
      manifold_condition_id_(condition_manifold->parameters().get<int>("ConditionID")),
      kinetics_condition_id_(condition_kinetics->parameters().get<int>("ConditionID")),
      manifold_map_extractor_(nullptr),
      master_converter_(nullptr),
      scatra_map_extractor_(nullptr),
      size_matrix_graph_()
{
  std::vector<int> inodegidvec_manifold;
  Core::Communication::add_owned_node_gid_from_list(
      *manifolddis, *condition_manifold->get_nodes(), inodegidvec_manifold);

  std::vector<int> inodegidvec_scatra;
  Core::Communication::add_owned_node_gid_from_list(
      *scatradis, *condition_kinetics->get_nodes(), inodegidvec_scatra);

  coupling_adapter_->setup_coupling(*scatradis, *manifolddis, inodegidvec_scatra,
      inodegidvec_manifold, ndof_per_node, true, 1.0e-8);
  master_converter_ =
      std::make_shared<Coupling::Adapter::CouplingMasterConverter>(*coupling_adapter_);

  scatra_map_extractor_ = std::make_shared<Core::LinAlg::MapExtractor>(
      *scatradis->dof_row_map(), coupling_adapter_->master_dof_map(), true);

  manifold_map_extractor_ = std::make_shared<Core::LinAlg::MapExtractor>(
      *manifolddis->dof_row_map(), coupling_adapter_->slave_dof_map(), true);

  // initially, the matrices are empty
  size_matrix_graph_.insert(std::make_pair(BlockMatrixType::ManifoldScaTra, 0));
  size_matrix_graph_.insert(std::make_pair(BlockMatrixType::ManifoldStructure, 0));
  size_matrix_graph_.insert(std::make_pair(BlockMatrixType::ScaTraManifold, 0));
  size_matrix_graph_.insert(std::make_pair(BlockMatrixType::ScaTraStructure, 0));
  size_matrix_graph_.insert(std::make_pair(BlockMatrixType::SysMatManifold, 0));
  size_matrix_graph_.insert(std::make_pair(BlockMatrixType::SysMatScaTra, 0));
}

/*---------------------------------------------------------------------------------*
 *---------------------------------------------------------------------------------*/
bool SSI::ManifoldScaTraCoupling::check_and_set_size_of_matrix_graph(
    const BlockMatrixType block, const int size)
{
  // check, if size of matrix graph changed between last evaluation and this evaluation
  const bool changed_size = size != size_matrix_graph_.at(block);

  // update new size
  size_matrix_graph_.at(block) = size;

  return changed_size;
}

/*---------------------------------------------------------------------------------*
 *---------------------------------------------------------------------------------*/
SSI::ScaTraManifoldScaTraFluxEvaluator::ScaTraManifoldScaTraFluxEvaluator(
    const SSI::SsiMono& ssi_mono)
    : block_map_scatra_(ssi_mono.block_map_scatra()),
      block_map_scatra_manifold_(ssi_mono.block_map_scatra_manifold()),
      block_map_structure_(ssi_mono.block_map_structure()),
      do_output_(Global::Problem::instance()
              ->ssi_control_params()
              .sublist("MANIFOLD")
              .get<bool>("OUTPUT_INFLOW")),
      full_map_manifold_(ssi_mono.maps_sub_problems()->map(
          Utils::SSIMaps::get_problem_position(Subproblem::manifold))),
      full_map_scatra_(ssi_mono.maps_sub_problems()->map(
          Utils::SSIMaps::get_problem_position(Subproblem::scalar_transport))),
      full_map_structure_(ssi_mono.maps_sub_problems()->map(
          Utils::SSIMaps::get_problem_position(Subproblem::structure))),
      scatra_(ssi_mono.scatra_base_algorithm()),
      scatra_manifold_(ssi_mono.scatra_manifold_base_algorithm()),
      ssi_structure_meshtying_(ssi_mono.ssi_structure_mesh_tying())
{
  // safety check before setup of coupling
  if (ssi_mono.scatra_field()->num_dof_per_node() != ssi_mono.scatra_manifold()->num_dof_per_node())
    FOUR_C_THROW("Number of dofs per node of scatra field and scatra manifold field must be equal");

  std::vector<const Core::Conditions::Condition*> conditions_manifold;
  scatra_manifold_->scatra_field()->discretization()->get_condition(
      "SSISurfaceManifold", conditions_manifold);

  std::vector<const Core::Conditions::Condition*> conditions_manifold_kinetics_scatra;
  scatra_->scatra_field()->discretization()->get_condition(
      "SSISurfaceManifoldKinetics", conditions_manifold_kinetics_scatra);

  // create pair: manifold condition - kinetics condition
  for (const auto& condition_manifold : conditions_manifold)
  {
    for (const auto& condition_kinetics : conditions_manifold_kinetics_scatra)
    {
      if (condition_manifold->parameters().get<int>("ConditionID") ==
          condition_kinetics->parameters().get<int>("ManifoldConditionID"))
      {
        scatra_manifold_couplings_.emplace_back(std::make_shared<SSI::ManifoldScaTraCoupling>(
            scatra_manifold_->scatra_field()->discretization(),
            scatra_->scatra_field()->discretization(), condition_manifold, condition_kinetics,
            ssi_mono.scatra_manifold()->num_dof_per_node()));
      }
    }
  }

  rhs_manifold_ = std::make_shared<Core::LinAlg::Vector<double>>(*full_map_manifold_, true);
  rhs_scatra_ = std::make_shared<Core::LinAlg::Vector<double>>(*full_map_scatra_, true);

  switch (scatra_->scatra_field()->matrix_type())
  {
    case Core::LinAlg::MatrixType::block_condition:
    case Core::LinAlg::MatrixType::block_condition_dof:
    {
      systemmatrix_manifold_ = SSI::Utils::SSIMatrices::setup_block_matrix(
          *block_map_scatra_manifold_, *block_map_scatra_manifold_);
      systemmatrix_scatra_ =
          SSI::Utils::SSIMatrices::setup_block_matrix(*block_map_scatra_, *block_map_scatra_);
      matrix_manifold_structure_ = SSI::Utils::SSIMatrices::setup_block_matrix(
          *block_map_scatra_manifold_, *block_map_structure_);
      matrix_manifold_scatra_ = SSI::Utils::SSIMatrices::setup_block_matrix(
          *block_map_scatra_manifold_, *block_map_scatra_);
      matrix_scatra_manifold_ = SSI::Utils::SSIMatrices::setup_block_matrix(
          *block_map_scatra_, *block_map_scatra_manifold_);
      matrix_scatra_structure_ =
          SSI::Utils::SSIMatrices::setup_block_matrix(*block_map_scatra_, *block_map_structure_);

      break;
    }
    case Core::LinAlg::MatrixType::sparse:
    {
      systemmatrix_manifold_ = SSI::Utils::SSIMatrices::setup_sparse_matrix(*full_map_manifold_);
      systemmatrix_scatra_ = SSI::Utils::SSIMatrices::setup_sparse_matrix(*full_map_scatra_);
      matrix_manifold_structure_ =
          SSI::Utils::SSIMatrices::setup_sparse_matrix(*full_map_manifold_);
      matrix_manifold_scatra_ = SSI::Utils::SSIMatrices::setup_sparse_matrix(*full_map_manifold_);
      matrix_scatra_manifold_ = SSI::Utils::SSIMatrices::setup_sparse_matrix(*full_map_scatra_);
      matrix_scatra_structure_ = SSI::Utils::SSIMatrices::setup_sparse_matrix(*full_map_scatra_);

      break;
    }
    default:
    {
      FOUR_C_THROW("Invalid matrix type associated with scalar transport field!");
      break;
    }
  }

  rhs_manifold_cond_ = std::make_shared<Core::LinAlg::Vector<double>>(*full_map_manifold_, true);
  rhs_scatra_cond_ = std::make_shared<Core::LinAlg::Vector<double>>(*full_map_scatra_, true);

  systemmatrix_manifold_cond_ = SSI::Utils::SSIMatrices::setup_sparse_matrix(*full_map_manifold_);
  systemmatrix_scatra_cond_ = SSI::Utils::SSIMatrices::setup_sparse_matrix(*full_map_scatra_);
  matrix_manifold_scatra_cond_ = SSI::Utils::SSIMatrices::setup_sparse_matrix(*full_map_manifold_);
  matrix_manifold_structure_cond_ =
      SSI::Utils::SSIMatrices::setup_sparse_matrix(*full_map_manifold_);
  matrix_scatra_manifold_cond_ = SSI::Utils::SSIMatrices::setup_sparse_matrix(*full_map_scatra_);
  matrix_scatra_structure_cond_ = SSI::Utils::SSIMatrices::setup_sparse_matrix(*full_map_scatra_);

  // Prepare runtime csv writer
  if (do_output())
  {
    runtime_csvwriter_.emplace(Core::Communication::my_mpi_rank(ssi_mono.get_comm()),
        *Global::Problem::instance()->output_control_file(), "manifold_inflow");

    for (const auto& condition_manifold : conditions_manifold)
    {
      const std::string manifold_string =
          "manifold " + std::to_string(condition_manifold->parameters().get<int>("ConditionID"));

      runtime_csvwriter_->register_data_vector("Integral of " + manifold_string, 1, 16);

      for (int k = 0; k < ssi_mono.scatra_manifold()->num_dof_per_node(); ++k)
      {
        runtime_csvwriter_->register_data_vector(
            "Total flux of scalar " + std::to_string(k + 1) + " into " + manifold_string, 1, 16);

        runtime_csvwriter_->register_data_vector(
            "Mean flux of scalar " + std::to_string(k + 1) + " into " + manifold_string, 1, 16);
      }
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::ScaTraManifoldScaTraFluxEvaluator::complete_matrix_manifold_scatra()
{
  switch (scatra_->scatra_field()->matrix_type())
  {
    case Core::LinAlg::MatrixType::block_condition:
    case Core::LinAlg::MatrixType::block_condition_dof:
    {
      matrix_manifold_scatra_->complete();
      break;
    }
    case Core::LinAlg::MatrixType::sparse:
    {
      matrix_manifold_scatra_->complete(*full_map_scatra_, *full_map_manifold_);
      break;
    }
    default:
    {
      FOUR_C_THROW("Invalid matrix type associated with scalar transport field!");
      break;
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::ScaTraManifoldScaTraFluxEvaluator::complete_matrix_manifold_structure()
{
  switch (scatra_->scatra_field()->matrix_type())
  {
    case Core::LinAlg::MatrixType::block_condition:
    case Core::LinAlg::MatrixType::block_condition_dof:
    {
      matrix_manifold_structure_->complete();
      break;
    }
    case Core::LinAlg::MatrixType::sparse:
    {
      matrix_manifold_structure_->complete(*full_map_structure_, *full_map_manifold_);
      break;
    }
    default:
    {
      FOUR_C_THROW("Invalid matrix type associated with scalar transport field!");
      break;
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::ScaTraManifoldScaTraFluxEvaluator::complete_matrix_scatra_manifold()
{
  switch (scatra_->scatra_field()->matrix_type())
  {
    case Core::LinAlg::MatrixType::block_condition:
    case Core::LinAlg::MatrixType::block_condition_dof:
    {
      matrix_scatra_manifold_->complete();
      break;
    }
    case Core::LinAlg::MatrixType::sparse:
    {
      matrix_scatra_manifold_->complete(*full_map_manifold_, *full_map_scatra_);
      break;
    }
    default:
    {
      FOUR_C_THROW("Invalid matrix type associated with scalar transport field!");
      break;
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::ScaTraManifoldScaTraFluxEvaluator::complete_matrix_scatra_structure()
{
  switch (scatra_->scatra_field()->matrix_type())
  {
    case Core::LinAlg::MatrixType::block_condition:
    case Core::LinAlg::MatrixType::block_condition_dof:
    {
      matrix_scatra_structure_->complete();
      break;
    }
    case Core::LinAlg::MatrixType::sparse:
    {
      matrix_scatra_structure_->complete(*full_map_structure_, *full_map_scatra_);
      break;
    }
    default:
    {
      FOUR_C_THROW("Invalid matrix type associated with scalar transport field!");
      break;
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::ScaTraManifoldScaTraFluxEvaluator::complete_system_matrix_manifold()
{
  systemmatrix_manifold_->complete();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::ScaTraManifoldScaTraFluxEvaluator::complete_system_matrix_scatra()
{
  systemmatrix_scatra_->complete();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::ScaTraManifoldScaTraFluxEvaluator::evaluate()
{
  // clear matrices and rhs from last evaluation
  systemmatrix_manifold_->zero();
  systemmatrix_scatra_->zero();
  matrix_manifold_scatra_->zero();
  matrix_manifold_structure_->zero();
  matrix_scatra_manifold_->zero();
  matrix_scatra_structure_->zero();

  rhs_manifold_->put_scalar(0.0);
  rhs_scatra_->put_scalar(0.0);

  // evaluate all scatra-manifold coupling conditions
  for (const auto& scatra_manifold_coupling : scatra_manifold_couplings_)
  {
    // clear matrices and rhs from last condition. Maps are different for each condition (need for
    // UnComplete()).
    systemmatrix_manifold_cond_->zero();
    systemmatrix_manifold_cond_->un_complete();
    systemmatrix_scatra_cond_->zero();
    systemmatrix_scatra_cond_->un_complete();
    matrix_manifold_scatra_cond_->zero();
    matrix_manifold_scatra_cond_->un_complete();
    matrix_manifold_structure_cond_->zero();
    matrix_manifold_structure_cond_->un_complete();
    matrix_scatra_manifold_cond_->zero();
    matrix_scatra_manifold_cond_->un_complete();
    matrix_scatra_structure_cond_->zero();
    matrix_scatra_structure_cond_->un_complete();

    rhs_manifold_cond_->put_scalar(0.0);
    rhs_scatra_cond_->put_scalar(0.0);

    evaluate_bulk_side(*scatra_manifold_coupling);

    copy_scatra_scatra_manifold_side(scatra_manifold_coupling);

    // This is needed because the graph of the matrices could change from step to step in case we
    // have zero flux (and then zero entries in the matrices)
    un_complete_matrices_if_necessary(*scatra_manifold_coupling);

    add_condition_contribution();
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::ScaTraManifoldScaTraFluxEvaluator::evaluate_bulk_side(
    ManifoldScaTraCoupling& scatra_manifold_coupling)
{
  // First: Set parameters to elements
  pre_evaluate(scatra_manifold_coupling);

  scatra_->scatra_field()->discretization()->set_state("phinp", *scatra_->scatra_field()->phinp());

  // Second: Evaluate condition
  {
    // manifold-scatra coupling matrix evaluated on scatra side
    auto matrix_scatra_manifold_cond_on_scatra_side =
        std::make_shared<Core::LinAlg::SparseMatrix>(*full_map_scatra_, 27, false, true);

    Teuchos::ParameterList condparams;
    Core::Utils::add_enum_class_to_parameter_list<ScaTra::BoundaryAction>(
        "action", ScaTra::BoundaryAction::calc_s2icoupling, condparams);
    condparams.set<int>("evaluate_manifold_coupling", 1);

    scatra_->scatra_field()->add_time_integration_specific_vectors();

    // Evaluation of RHS and scatra-manifold coupling matrices
    {
      Core::Utils::add_enum_class_to_parameter_list<ScaTra::DifferentiationType>(
          "differentiationtype", ScaTra::DifferentiationType::elch, condparams);

      // dscatra_dscatra, dscatra_dmanifold (on scatra side)
      Core::FE::AssembleStrategy strategyscatra(0, 0, systemmatrix_scatra_cond_,
          matrix_scatra_manifold_cond_on_scatra_side, rhs_scatra_cond_, nullptr, nullptr);

      scatra_->scatra_field()->discretization()->evaluate_condition(condparams, strategyscatra,
          "SSISurfaceManifoldKinetics", scatra_manifold_coupling.kinetics_condition_id());

      systemmatrix_scatra_cond_->complete();
      matrix_scatra_manifold_cond_on_scatra_side->complete();

      // dscatra_dmanifold (on scatra side) -> dscatra_dmanifold
      Coupling::Adapter::MatrixLogicalSplitAndTransform()(
          *matrix_scatra_manifold_cond_on_scatra_side, *full_map_scatra_, *full_map_scatra_, 1.0,
          nullptr, &*scatra_manifold_coupling.master_converter(), *matrix_scatra_manifold_cond_,
          true, true);
      matrix_scatra_manifold_cond_->complete(*full_map_manifold_, *full_map_scatra_);
    }

    // Evaluation of linearization w.r.t. displacement
    {
      Core::Utils::add_enum_class_to_parameter_list<ScaTra::BoundaryAction>(
          "action", ScaTra::BoundaryAction::calc_s2icoupling_od, condparams);

      Core::Utils::add_enum_class_to_parameter_list<ScaTra::DifferentiationType>(
          "differentiationtype", ScaTra::DifferentiationType::disp, condparams);

      // dscatra_dstructure
      auto matrix_scatra_structure_cond_slave_side_disp_evaluate =
          std::make_shared<Core::LinAlg::SparseMatrix>(*full_map_scatra_, 27, false, true);

      Core::FE::AssembleStrategy strategyscatra(0, scatra_->scatra_field()->nds_disp(),
          matrix_scatra_structure_cond_slave_side_disp_evaluate, nullptr, nullptr, nullptr,
          nullptr);

      scatra_->scatra_field()->discretization()->evaluate_condition(condparams, strategyscatra,
          "SSISurfaceManifoldKinetics", scatra_manifold_coupling.kinetics_condition_id());

      matrix_scatra_structure_cond_slave_side_disp_evaluate->complete(
          *full_map_structure_, *full_map_scatra_);

      // "slave side" from manifold and from structure do not need to be the same nodes.
      // Linearization is evaluated on scatra slave side node --> Transformation needed
      Core::LinAlg::SparseMatrix matrix_scatra_structure_cond_slave_side_disp(
          *full_map_scatra_, 27, false, true);
      for (const auto& meshtying : ssi_structure_meshtying_->mesh_tying_handlers())
      {
        auto slave_slave_transformation = meshtying->slave_slave_transformation();
        // converter between old slave dofs from input and actual slave dofs from current mesh tying
        // adapter
        auto slave_slave_converter =
            Coupling::Adapter::CouplingSlaveConverter(*slave_slave_transformation);

        // old slave dofs from input
        auto slave_map = slave_slave_transformation->slave_dof_map();

        Coupling::Adapter::MatrixLogicalSplitAndTransform()(
            *matrix_scatra_structure_cond_slave_side_disp_evaluate, *full_map_scatra_, *slave_map,
            1.0, nullptr, &slave_slave_converter, matrix_scatra_structure_cond_slave_side_disp,
            true, true);
      }
      matrix_scatra_structure_cond_slave_side_disp.complete(
          *full_map_structure_, *full_map_scatra_);

      // Add slave side disp. contributions
      Core::LinAlg::matrix_add(matrix_scatra_structure_cond_slave_side_disp, false, 1.0,
          *matrix_scatra_structure_cond_, 0.0);

      // Add master side disp. contributions
      for (const auto& meshtying : ssi_structure_meshtying_->mesh_tying_handlers())
      {
        auto cond_slave_dof_map = meshtying->slave_master_coupling()->slave_dof_map();
        auto converter = meshtying->slave_side_converter();

        // assemble derivatives of x w.r.t. structure slave dofs
        Coupling::Adapter::MatrixLogicalSplitAndTransform()(
            matrix_scatra_structure_cond_slave_side_disp,
            matrix_scatra_structure_cond_slave_side_disp.range_map(), *cond_slave_dof_map, 1.0,
            nullptr, &(*converter), *matrix_scatra_structure_cond_, true, true);
      }
      matrix_scatra_structure_cond_->complete(*full_map_structure_, *full_map_scatra_);
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::ScaTraManifoldScaTraFluxEvaluator::copy_scatra_scatra_manifold_side(
    std::shared_ptr<ManifoldScaTraCoupling> scatra_manifold_coupling)
{
  const double inv_thickness = scatra_manifold_coupling->inv_thickness();
  {
    auto rhs_scatra_cond_extract =
        scatra_manifold_coupling->scatra_map_extractor()->extract_cond_vector(*rhs_scatra_cond_);

    auto rhs_manifold_cond_extract =
        scatra_manifold_coupling->coupling_adapter()->master_to_slave(*rhs_scatra_cond_extract);

    scatra_manifold_coupling->manifold_map_extractor()->add_cond_vector(
        *rhs_manifold_cond_extract, *rhs_manifold_cond_);
    rhs_manifold_cond_->scale(-inv_thickness);
  }

  // dmanifold_dscatra: scatra rows are transformed to manifold side (flux is scaled by -1.0)
  Coupling::Adapter::MatrixLogicalSplitAndTransform()(*systemmatrix_scatra_cond_, *full_map_scatra_,
      *full_map_scatra_, -inv_thickness, &*scatra_manifold_coupling->master_converter(), nullptr,
      *matrix_manifold_scatra_cond_, true, true);

  matrix_manifold_scatra_cond_->complete(*full_map_scatra_, *full_map_manifold_);

  // dmanifold_dmanifold: scatra rows are transformed to manifold side (flux is scaled by -1.0)
  Coupling::Adapter::MatrixLogicalSplitAndTransform()(*matrix_scatra_manifold_cond_,
      *full_map_scatra_, *full_map_manifold_, -inv_thickness,
      &*scatra_manifold_coupling->master_converter(), nullptr, *systemmatrix_manifold_cond_, true,
      true);

  systemmatrix_manifold_cond_->complete();

  // dmanifold_dstructure: scatra rows are transformed to manifold side (flux is scaled by -1.0)
  Coupling::Adapter::MatrixLogicalSplitAndTransform()(*matrix_scatra_structure_cond_,
      *full_map_scatra_, *full_map_structure_, -inv_thickness,
      &*scatra_manifold_coupling->master_converter(), nullptr, *matrix_manifold_structure_cond_,
      true, true);

  matrix_manifold_structure_cond_->complete(*full_map_structure_, *full_map_manifold_);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::ScaTraManifoldScaTraFluxEvaluator::add_condition_contribution()
{
  rhs_manifold_->update(1.0, *rhs_manifold_cond_, 1.0);
  rhs_scatra_->update(1.0, *rhs_scatra_cond_, 1.0);

  switch (scatra_->scatra_field()->matrix_type())
  {
    case Core::LinAlg::MatrixType::block_condition:
    case Core::LinAlg::MatrixType::block_condition_dof:
    {
      auto blockmaps_manifold = *scatra_manifold_->scatra_field()->dof_block_maps();

      /*
       * m: manifold
       * s: scatra
       * d: structure/displacements
       */
      auto flux_manifold_scatra_mm_block =
          Core::LinAlg::split_matrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
              *systemmatrix_manifold_cond_, blockmaps_manifold, blockmaps_manifold);
      auto flux_manifold_scatra_md_block =
          Core::LinAlg::split_matrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
              *matrix_manifold_structure_cond_, *block_map_structure_, blockmaps_manifold);
      auto flux_manifold_scatra_ms_block =
          Core::LinAlg::split_matrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
              *matrix_manifold_scatra_cond_, *block_map_scatra_, blockmaps_manifold);
      auto flux_manifold_scatra_sm_block =
          Core::LinAlg::split_matrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
              *matrix_scatra_manifold_cond_, blockmaps_manifold, *block_map_scatra_);
      auto flux_manifold_scatra_sd_block =
          Core::LinAlg::split_matrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
              *matrix_scatra_structure_cond_, *block_map_structure_, *block_map_scatra_);
      auto flux_manifold_scatra_ss_block =
          Core::LinAlg::split_matrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
              *systemmatrix_scatra_cond_, *block_map_scatra_, *block_map_scatra_);

      flux_manifold_scatra_mm_block->complete();
      flux_manifold_scatra_md_block->complete();
      flux_manifold_scatra_ms_block->complete();
      flux_manifold_scatra_sm_block->complete();
      flux_manifold_scatra_sd_block->complete();
      flux_manifold_scatra_ss_block->complete();

      systemmatrix_manifold_->add(*flux_manifold_scatra_mm_block, false, 1.0, 1.0);
      matrix_manifold_scatra_->add(*flux_manifold_scatra_ms_block, false, 1.0, 1.0);
      matrix_manifold_structure_->add(*flux_manifold_scatra_md_block, false, 1.0, 1.0);
      systemmatrix_scatra_->add(*flux_manifold_scatra_ss_block, false, 1.0, 1.0);
      matrix_scatra_manifold_->add(*flux_manifold_scatra_sm_block, false, 1.0, 1.0);
      matrix_scatra_structure_->add(*flux_manifold_scatra_sd_block, false, 1.0, 1.0);

      break;
    }
    case Core::LinAlg::MatrixType::sparse:
    {
      systemmatrix_manifold_->add(*systemmatrix_manifold_cond_, false, 1.0, 1.0);
      matrix_manifold_structure_->add(*matrix_manifold_structure_cond_, false, 1.0, 1.0);
      matrix_manifold_scatra_->add(*matrix_manifold_scatra_cond_, false, 1.0, 1.0);
      systemmatrix_scatra_->add(*systemmatrix_scatra_cond_, false, 1.0, 1.0);
      matrix_scatra_structure_->add(*matrix_scatra_structure_cond_, false, 1.0, 1.0);
      matrix_scatra_manifold_->add(*matrix_scatra_manifold_cond_, false, 1.0, 1.0);

      break;
    }
    default:
    {
      FOUR_C_THROW("Invalid matrix type associated with scalar transport field!");
      break;
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::ScaTraManifoldScaTraFluxEvaluator::evaluate_scatra_manifold_inflow()
{
  inflow_.clear();
  domainintegral_.clear();

  scatra_->scatra_field()->discretization()->set_state("phinp", *scatra_->scatra_field()->phinp());

  for (const auto& scatra_manifold_coupling : scatra_manifold_couplings_)
  {
    const int kineticsID = scatra_manifold_coupling->kinetics_condition_id();

    std::vector<double> zero_scalar_vector(scatra_->scatra_field()->num_dof_per_node(), 0.0);
    inflow_.insert(std::make_pair(kineticsID, zero_scalar_vector));

    // First: set parameters to elements
    pre_evaluate(*scatra_manifold_coupling);

    // Second: evaluate condition
    evaluate_scatra_manifold_inflow_integral(*scatra_manifold_coupling);

    // Third: evaluate domain integral
    evaluate_scatra_manifold_domain_integral(*scatra_manifold_coupling);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::ScaTraManifoldScaTraFluxEvaluator::evaluate_scatra_manifold_domain_integral(
    ManifoldScaTraCoupling& scatra_manifold_coupling)
{
  const int kineticsID = scatra_manifold_coupling.kinetics_condition_id();

  // integrate only if not done so far
  if (domainintegral_.find(kineticsID) == domainintegral_.end())
  {
    Teuchos::ParameterList condparams;

    Core::Utils::add_enum_class_to_parameter_list<ScaTra::BoundaryAction>(
        "action", ScaTra::BoundaryAction::calc_boundary_integral, condparams);

    // integrated domain of this condition
    Core::LinAlg::SerialDenseVector domainintegral_cond(1);

    scatra_->scatra_field()->discretization()->evaluate_scalars(
        condparams, domainintegral_cond, "SSISurfaceManifold", kineticsID);

    domainintegral_.insert(std::make_pair(kineticsID, domainintegral_cond.values()[0]));
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::ScaTraManifoldScaTraFluxEvaluator::evaluate_scatra_manifold_inflow_integral(
    ManifoldScaTraCoupling& scatra_manifold_coupling)
{
  const int kineticsID = scatra_manifold_coupling.kinetics_condition_id();

  Teuchos::ParameterList condparams;

  Core::Utils::add_enum_class_to_parameter_list<ScaTra::BoundaryAction>(
      "action", ScaTra::BoundaryAction::calc_s2icoupling_flux, condparams);

  condparams.set<bool>("only_positive_fluxes", true);

  condparams.set<int>("evaluate_manifold_coupling", 1);

  // integrated scalars of this condition
  auto inflow_cond = std::make_shared<Core::LinAlg::SerialDenseVector>(
      scatra_->scatra_field()->num_dof_per_node());

  scatra_->scatra_field()->discretization()->evaluate_scalars(
      condparams, *inflow_cond, "SSISurfaceManifold", kineticsID);

  for (int i = 0; i < inflow_cond->length(); ++i)
    inflow_.at(kineticsID).at(i) += inflow_cond->values()[i];
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::ScaTraManifoldScaTraFluxEvaluator::pre_evaluate(
    ManifoldScaTraCoupling& scatra_manifold_coupling)
{
  Teuchos::ParameterList eleparams;

  eleparams.set<Core::Conditions::ConditionType>(
      "condition type", Core::Conditions::ConditionType::S2IKinetics);

  switch (
      scatra_manifold_coupling.condition_kinetics()->parameters().get<Inpar::S2I::KineticModels>(
          "KINETIC_MODEL"))
  {
    case Inpar::S2I::kinetics_constantinterfaceresistance:
    {
      eleparams.set<int>("KINETIC_MODEL", Inpar::S2I::kinetics_constantinterfaceresistance);
      eleparams.set<double>("RESISTANCE",
          scatra_manifold_coupling.condition_kinetics()->parameters().get<double>("RESISTANCE"));
      eleparams.set<const std::vector<int>*>("ONOFF",
          &scatra_manifold_coupling.condition_kinetics()->parameters().get<std::vector<int>>(
              "ONOFF"));
      eleparams.set<int>("numelectrons",
          scatra_manifold_coupling.condition_kinetics()->parameters().get<int>("E-"));
      break;
    }
    case Inpar::S2I::kinetics_butlervolmerreduced:
    {
      eleparams.set<int>("KINETIC_MODEL", Inpar::S2I::kinetics_butlervolmerreduced);
      eleparams.set<int>("NUMSCAL",
          scatra_manifold_coupling.condition_kinetics()->parameters().get<int>("NUMSCAL"));
      eleparams.set<const std::vector<int>*>("STOICHIOMETRIES",
          &scatra_manifold_coupling.condition_kinetics()->parameters().get<std::vector<int>>(
              "STOICHIOMETRIES"));
      eleparams.set<int>("numelectrons",
          scatra_manifold_coupling.condition_kinetics()->parameters().get<int>("E-"));
      eleparams.set<double>(
          "K_R", scatra_manifold_coupling.condition_kinetics()->parameters().get<double>("K_R"));
      eleparams.set<double>("ALPHA_A",
          scatra_manifold_coupling.condition_kinetics()->parameters().get<double>("ALPHA_A"));
      eleparams.set<double>("ALPHA_C",
          scatra_manifold_coupling.condition_kinetics()->parameters().get<double>("ALPHA_C"));
      break;
    }
    case Inpar::S2I::kinetics_nointerfaceflux:
    {
      eleparams.set<int>("KINETIC_MODEL", Inpar::S2I::kinetics_nointerfaceflux);
      break;
    }
    default:
    {
      FOUR_C_THROW("Unknown kinetics type for manifold couplign");
      break;
    }
  }

  Discret::Elements::ScaTraEleParameterBoundary::instance("scatra")->set_parameters(eleparams);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::ScaTraManifoldScaTraFluxEvaluator::output()
{
  FOUR_C_ASSERT(runtime_csvwriter_.has_value(), "internal error: runtime csv writer not created.");

  std::map<std::string, std::vector<double>> output_data;
  for (const auto& inflow_comp : inflow_)
  {
    const std::string manifold_string = "manifold " + std::to_string(inflow_comp.first);

    // find pair of domain integrals with same key for current condition and get domain integral
    const auto domainintegral_cond = domainintegral_.find(inflow_comp.first);
    const double domainint = domainintegral_cond->second;

    output_data["Integral of " + manifold_string] = {domainint};

    for (int i = 0; i < static_cast<int>(inflow_comp.second.size()); ++i)
    {
      output_data["Total flux of scalar " + std::to_string(i + 1) + " into " + manifold_string] = {
          inflow_comp.second[i]};
      output_data["Mean flux of scalar " + std::to_string(i + 1) + " into " + manifold_string] = {
          inflow_comp.second[i] / domainint};
    }
  }
  runtime_csvwriter_->write_data_to_file(scatra_manifold_->scatra_field()->time(),
      scatra_manifold_->scatra_field()->step(), output_data);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::ScaTraManifoldScaTraFluxEvaluator::un_complete_matrices_if_necessary(
    ManifoldScaTraCoupling& scatra_manifold_coupling)
{
  // definition of lambda function to get size of graph for given matrix
  auto graph_size = [](std::shared_ptr<Core::LinAlg::SparseMatrix> matrix)
  { return matrix->num_global_nonzeros(); };

  // get size of graphs of conditions matrices
  const int size_manifold_scatra_graph_ = graph_size(matrix_manifold_scatra_cond_);
  const int size_manifold_structure_graph = graph_size(matrix_manifold_structure_cond_);
  const int size_scatra_manifold_graph = graph_size(matrix_scatra_manifold_cond_);
  const int size_scatra_structure_graph = graph_size(matrix_scatra_structure_cond_);
  const int size_manifold_sysmat_graph = graph_size(systemmatrix_manifold_cond_);
  const int size_scatra_sysmat_graph = graph_size(systemmatrix_scatra_cond_);

  // check if size of any condition matrix was updated and store new size
  bool do_uncomplete = false;
  if (scatra_manifold_coupling.check_and_set_size_of_matrix_graph(
          BlockMatrixType::ManifoldScaTra, size_manifold_scatra_graph_))
    do_uncomplete = true;
  if (scatra_manifold_coupling.check_and_set_size_of_matrix_graph(
          BlockMatrixType::ManifoldStructure, size_manifold_structure_graph))
    do_uncomplete = true;
  if (scatra_manifold_coupling.check_and_set_size_of_matrix_graph(
          BlockMatrixType::ScaTraManifold, size_scatra_manifold_graph))
    do_uncomplete = true;
  if (scatra_manifold_coupling.check_and_set_size_of_matrix_graph(
          BlockMatrixType::ScaTraStructure, size_scatra_structure_graph))
    do_uncomplete = true;
  if (scatra_manifold_coupling.check_and_set_size_of_matrix_graph(
          BlockMatrixType::SysMatManifold, size_manifold_sysmat_graph))
    do_uncomplete = true;
  if (scatra_manifold_coupling.check_and_set_size_of_matrix_graph(
          BlockMatrixType::SysMatScaTra, size_scatra_sysmat_graph))
    do_uncomplete = true;

  // uncomplete all global matrices if condition matrices have updated graph
  if (do_uncomplete)
  {
    matrix_manifold_scatra_->un_complete();
    matrix_manifold_structure_->un_complete();
    matrix_scatra_manifold_->un_complete();
    matrix_scatra_structure_->un_complete();
    systemmatrix_manifold_->un_complete();
    systemmatrix_scatra_->un_complete();
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
SSI::ManifoldMeshTyingStrategyBase::ManifoldMeshTyingStrategyBase(
    std::shared_ptr<Core::FE::Discretization> scatra_manifold_dis,
    std::shared_ptr<SSI::Utils::SSIMaps> ssi_maps, const bool is_manifold_meshtying)
    : is_manifold_meshtying_(is_manifold_meshtying),
      condensed_dof_map_(nullptr),
      ssi_maps_(std::move(ssi_maps)),
      ssi_meshtying_(nullptr)
{
  if (is_manifold_meshtying_)
  {
    ssi_meshtying_ = std::make_shared<SSI::Utils::SSIMeshTying>(
        "SSISurfaceManifold", *scatra_manifold_dis, false, false);

    if (ssi_meshtying_->mesh_tying_handlers().empty())
    {
      FOUR_C_THROW(
          "Could not create mesh tying between manifold fields. They are not intersecting. "
          "Disable 'MESHTYING_MANIFOLD' or create intersecting manifold conditions.");
    }

    // merge slave dof maps from all mesh tying conditions
    std::shared_ptr<Core::LinAlg::Map> slave_dof_map = nullptr;
    for (const auto& meshtying : ssi_meshtying_->mesh_tying_handlers())
    {
      auto coupling_adapter = meshtying->slave_master_coupling();
      if (slave_dof_map == nullptr)
        slave_dof_map = std::make_shared<Core::LinAlg::Map>(*coupling_adapter->slave_dof_map());
      else
      {
        auto slave_dof_map_old = std::make_shared<Core::LinAlg::Map>(*slave_dof_map);
        slave_dof_map =
            Core::LinAlg::merge_map(slave_dof_map_old, coupling_adapter->slave_dof_map());
      }
    }
    // exclusive interior and master dofs across all slave conditions
    condensed_dof_map_ =
        Core::LinAlg::split_map(*ssi_maps_->scatra_manifold_dof_row_map(), *slave_dof_map);
  }
  else
  {
    condensed_dof_map_ = ssi_maps_->scatra_manifold_dof_row_map();
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
SSI::ManifoldMeshTyingStrategySparse::ManifoldMeshTyingStrategySparse(
    std::shared_ptr<Core::FE::Discretization> scatra_manifold_dis,
    std::shared_ptr<Utils::SSIMaps> ssi_maps, const bool is_manifold_meshtying)
    : ManifoldMeshTyingStrategyBase(scatra_manifold_dis, ssi_maps, is_manifold_meshtying)
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
SSI::ManifoldMeshTyingStrategyBlock::ManifoldMeshTyingStrategyBlock(
    std::shared_ptr<Core::FE::Discretization> scatra_manifold_dis,
    std::shared_ptr<SSI::Utils::SSIMaps> ssi_maps, const bool is_manifold_meshtying)
    : ManifoldMeshTyingStrategyBase(scatra_manifold_dis, ssi_maps, is_manifold_meshtying),
      condensed_block_dof_map_(nullptr),
      meshtying_block_handler_()
{
  // split condensed_dof_map into blocks
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> partial_maps_condensed_block_dof_map;
  for (int i = 0; i < ssi_maps_->block_map_scatra_manifold()->num_maps(); ++i)
  {
    partial_maps_condensed_block_dof_map.emplace_back(Core::LinAlg::intersect_map(
        *condensed_dof_map_, *ssi_maps_->block_map_scatra_manifold()->map(i)));
  }

  condensed_block_dof_map_ = std::make_shared<Core::LinAlg::MultiMapExtractor>(
      *condensed_dof_map_, partial_maps_condensed_block_dof_map);

  if (is_manifold_meshtying_)
  {
    // couple meshyting_handler_ and condensed_block_dof_map_ to meshtying_block_handler_
    for (const auto& meshtying : ssi_mesh_tying()->mesh_tying_handlers())
    {
      auto coupling_adapter = meshtying->slave_master_coupling();
      auto slave_dof_map = coupling_adapter->slave_dof_map();
      auto perm_slave_dof_map = coupling_adapter->perm_slave_dof_map();
      auto master_dof_map = coupling_adapter->master_dof_map();
      auto perm_master_dof_map = coupling_adapter->perm_master_dof_map();

      // split maps according to split of matrix blocks, i.e. block maps
      std::vector<std::shared_ptr<const Core::LinAlg::Map>> partial_maps_slave_block_dof_map;
      std::vector<std::shared_ptr<Coupling::Adapter::Coupling>> partial_block_adapters;
      for (int i = 0; i < ssi_maps_->block_map_scatra_manifold()->num_maps(); ++i)
      {
        auto [slave_block_map, perm_master_block_map] =
            intersect_coupling_maps_block_map(*ssi_maps_->block_map_scatra_manifold()->map(i),
                *slave_dof_map, *perm_master_dof_map, scatra_manifold_dis->get_comm());

        auto [master_block_map, perm_slave_block_map] =
            intersect_coupling_maps_block_map(*ssi_maps_->block_map_scatra_manifold()->map(i),
                *master_dof_map, *perm_slave_dof_map, scatra_manifold_dis->get_comm());

        auto coupling_adapter_block = std::make_shared<Coupling::Adapter::Coupling>();
        coupling_adapter_block->setup_coupling(
            slave_block_map, perm_slave_block_map, master_block_map, perm_master_block_map);

        partial_maps_slave_block_dof_map.emplace_back(slave_block_map);
        partial_block_adapters.emplace_back(coupling_adapter_block);
      }

      meshtying_block_handler_.emplace_back(
          partial_block_adapters, partial_maps_slave_block_dof_map);
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::tuple<std::shared_ptr<const Core::LinAlg::Map>, std::shared_ptr<const Core::LinAlg::Map>>
SSI::ManifoldMeshTyingStrategyBlock::intersect_coupling_maps_block_map(
    const Core::LinAlg::Map& block_map, const Core::LinAlg::Map& intersecting_map,
    const Core::LinAlg::Map& permuted_map, MPI_Comm comm)
{
  std::vector<int> intersecting_map_vec, permuted_intersecting_map_vec;
  for (int slave_lid = 0; slave_lid < intersecting_map.num_my_elements(); ++slave_lid)
  {
    const int slave_gid = intersecting_map.gid(slave_lid);
    if (block_map.lid(slave_gid) != -1)
    {
      intersecting_map_vec.emplace_back(slave_gid);
      permuted_intersecting_map_vec.emplace_back(permuted_map.gid(slave_lid));
    }
  }

  auto intersected_map = std::make_shared<Core::LinAlg::Map>(
      -1, static_cast<int>(intersecting_map_vec.size()), intersecting_map_vec.data(), 0, comm);

  auto permuted_intersected_map = std::make_shared<Core::LinAlg::Map>(-1,
      static_cast<int>(permuted_intersecting_map_vec.size()), permuted_intersecting_map_vec.data(),
      0, comm);

  return {intersected_map, permuted_intersected_map};
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::ManifoldMeshTyingStrategyBase::apply_mesh_tying_to_manifold_rhs(
    Core::LinAlg::Vector<double>& rhs_manifold)
{
  if (is_manifold_meshtying_)
  {
    for (const auto& meshtying : ssi_mesh_tying()->mesh_tying_handlers())
    {
      auto coupling_adapter = meshtying->slave_master_coupling();
      auto multimap = meshtying->slave_master_extractor();

      auto slave_dofs = multimap->extract_vector(rhs_manifold, 1);
      auto slave_to_master_dofs = coupling_adapter->slave_to_master(*slave_dofs);
      multimap->add_vector(*slave_to_master_dofs, 2, rhs_manifold);
      multimap->put_scalar(rhs_manifold, 1, 0.0);
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::ManifoldMeshTyingStrategySparse::apply_meshtying_to_manifold_matrix(
    std::shared_ptr<Core::LinAlg::SparseOperator> ssi_manifold_matrix,
    std::shared_ptr<const Core::LinAlg::SparseOperator> manifold_matrix)
{
  auto ssi_manifold_sparse =
      Core::LinAlg::cast_to_sparse_matrix_and_check_success(ssi_manifold_matrix);
  auto manifold_sparse =
      Core::LinAlg::cast_to_const_sparse_matrix_and_check_success(manifold_matrix);

  // add derivs. of interior/master dofs. w.r.t. interior/master dofs
  Coupling::Adapter::MatrixLogicalSplitAndTransform()(*manifold_sparse, *condensed_dof_map_,
      *condensed_dof_map_, 1.0, nullptr, nullptr, *ssi_manifold_sparse, true, true);

  if (is_manifold_meshtying_)
  {
    for (const auto& meshtying : ssi_mesh_tying()->mesh_tying_handlers())
    {
      auto coupling_adapter = meshtying->slave_master_coupling();

      auto cond_slave_dof_map = coupling_adapter->slave_dof_map();
      auto converter = Coupling::Adapter::CouplingSlaveConverter(*coupling_adapter);

      // add derivs. of slave dofs. w.r.t. slave dofs
      Coupling::Adapter::MatrixLogicalSplitAndTransform()(*manifold_sparse, *cond_slave_dof_map,
          *cond_slave_dof_map, 1.0, &converter, &converter, *ssi_manifold_sparse, true, true);
      // add derivs. of slave dofs. w.r.t. interior/master dofs
      Coupling::Adapter::MatrixLogicalSplitAndTransform()(*manifold_sparse, *cond_slave_dof_map,
          *condensed_dof_map_, 1.0, &converter, nullptr, *ssi_manifold_sparse, true, true);
      // add derivs. of interior/master dofs w.r.t. slave dofs
      Coupling::Adapter::MatrixLogicalSplitAndTransform()(*manifold_sparse, *condensed_dof_map_,
          *cond_slave_dof_map, 1.0, nullptr, &converter, *ssi_manifold_sparse, true, true);
    }

    // Finalize: put 1.0 on main diag of slave dofs
    const double one = 1.0;
    for (const auto& meshtying : ssi_mesh_tying()->mesh_tying_handlers())
    {
      auto coupling_adapter = meshtying->slave_master_coupling();

      auto slave_dof_map = coupling_adapter->slave_dof_map();
      for (int doflid_slave = 0; doflid_slave < slave_dof_map->num_my_elements(); ++doflid_slave)
      {
        // extract global ID of current slave-side row
        const int dofgid_slave = slave_dof_map->gid(doflid_slave);
        if (dofgid_slave < 0) FOUR_C_THROW("Local ID not found!");

        // apply pseudo Dirichlet conditions to filled matrix, i.e., to local row and column indices
        if (ssi_manifold_sparse->filled())
        {
          const int rowlid_slave = ssi_manifold_sparse->row_map().lid(dofgid_slave);
          if (rowlid_slave < 0) FOUR_C_THROW("Global ID not found!");
          ssi_manifold_sparse->replace_my_values(rowlid_slave, 1, &one, &rowlid_slave);
        }

        // apply pseudo Dirichlet conditions to unfilled matrix, i.e., to global row and column
        // indices
        else
        {
          ssi_manifold_sparse->insert_global_values(dofgid_slave, 1, &one, &dofgid_slave);
        }
      }
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::ManifoldMeshTyingStrategyBlock::apply_meshtying_to_manifold_matrix(
    std::shared_ptr<Core::LinAlg::SparseOperator> ssi_manifold_matrix,
    std::shared_ptr<const Core::LinAlg::SparseOperator> manifold_matrix)
{
  auto ssi_manifold_block =
      Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(ssi_manifold_matrix);
  auto manifold_block =
      Core::LinAlg::cast_to_const_block_sparse_matrix_base_and_check_success(manifold_matrix);

  for (int row = 0; row < condensed_block_dof_map_->num_maps(); ++row)
  {
    for (int col = 0; col < condensed_block_dof_map_->num_maps(); ++col)
    {
      // add derivs. of interior/master dofs. w.r.t. interior/master dofs
      Coupling::Adapter::MatrixLogicalSplitAndTransform()(manifold_block->matrix(row, col),
          *condensed_block_dof_map_->map(row), *condensed_block_dof_map_->map(col), 1.0, nullptr,
          nullptr, ssi_manifold_block->matrix(row, col), true, true);

      if (is_manifold_meshtying_)
      {
        for (const auto& block_meshtying : mesh_tying_block_handler())
        {
          auto meshtying = block_meshtying.first;
          auto cond_block_slave_dof_map = block_meshtying.second;
          auto converter_row = Coupling::Adapter::CouplingSlaveConverter(*meshtying[row]);
          auto converter_col = Coupling::Adapter::CouplingSlaveConverter(*meshtying[col]);

          // add derivs. of slave dofs. w.r.t. slave dofs
          Coupling::Adapter::MatrixLogicalSplitAndTransform()(manifold_block->matrix(row, col),
              *cond_block_slave_dof_map[row], *cond_block_slave_dof_map[col], 1.0, &converter_row,
              &converter_col, ssi_manifold_block->matrix(row, col), true, true);
          // add derivs. of slave dofs. w.r.t. interior/master dofs
          Coupling::Adapter::MatrixLogicalSplitAndTransform()(manifold_block->matrix(row, col),
              *cond_block_slave_dof_map[row], *condensed_block_dof_map_->map(col), 1.0,
              &converter_row, nullptr, ssi_manifold_block->matrix(row, col), true, true);
          // add derivs. of interior/master dofs w.r.t. slave dofs
          Coupling::Adapter::MatrixLogicalSplitAndTransform()(manifold_block->matrix(row, col),
              *condensed_block_dof_map_->map(row), *cond_block_slave_dof_map[col], 1.0, nullptr,
              &converter_col, ssi_manifold_block->matrix(row, col), true, true);
        }

        // Finalize: put 1.0 on main diag of slave dofs
        const double one = 1.0;
        if (row == col)
        {
          for (const auto& block_meshtying : mesh_tying_block_handler())
          {
            auto coupling_adapter = block_meshtying.first[row];
            auto slave_dof_map = coupling_adapter->slave_dof_map();
            for (int doflid_slave = 0; doflid_slave < slave_dof_map->num_my_elements();
                ++doflid_slave)
            {
              // extract global ID of current slave-side row
              const int dofgid_slave = slave_dof_map->gid(doflid_slave);
              if (dofgid_slave < 0) FOUR_C_THROW("Local ID not found!");

              // apply pseudo Dirichlet conditions to filled matrix, i.e., to local row and column
              // indices
              if (ssi_manifold_block->matrix(row, row).filled())
              {
                const int rowlid_slave =
                    ssi_manifold_block->matrix(row, row).row_map().lid(dofgid_slave);
                if (rowlid_slave < 0) FOUR_C_THROW("Global ID not found!");
                ssi_manifold_block->matrix(row, row).replace_my_values(
                    rowlid_slave, 1, &one, &rowlid_slave);
              }

              // apply pseudo Dirichlet conditions to unfilled matrix, i.e., to global row and
              // column indices
              else
              {
                ssi_manifold_block->matrix(row, row).insert_global_values(
                    dofgid_slave, 1, &one, &dofgid_slave);
              }
            }
          }
        }
      }
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::ManifoldMeshTyingStrategySparse::apply_meshtying_to_manifold_scatra_matrix(
    std::shared_ptr<Core::LinAlg::SparseOperator> ssi_manifold_scatra_matrix,
    std::shared_ptr<const Core::LinAlg::SparseOperator> manifold_scatra_matrix)
{
  auto ssi_manifold_scatra_sparse =
      Core::LinAlg::cast_to_sparse_matrix_and_check_success(ssi_manifold_scatra_matrix);
  auto manifold_scatra_sparse =
      Core::LinAlg::cast_to_const_sparse_matrix_and_check_success(manifold_scatra_matrix);

  // add derivs. of interior/master dofs w.r.t. scatra dofs
  Coupling::Adapter::MatrixLogicalSplitAndTransform()(*manifold_scatra_sparse, *condensed_dof_map_,
      *ssi_maps_->scatra_dof_row_map(), 1.0, nullptr, nullptr, *ssi_manifold_scatra_sparse, true,
      true);

  if (is_manifold_meshtying_)
  {
    for (const auto& meshtying : ssi_mesh_tying()->mesh_tying_handlers())
    {
      auto coupling_adapter = meshtying->slave_master_coupling();

      auto cond_slave_dof_map = coupling_adapter->slave_dof_map();
      auto converter = Coupling::Adapter::CouplingSlaveConverter(*coupling_adapter);

      // add derivs. of slave dofs w.r.t. scatra dofs
      Coupling::Adapter::MatrixLogicalSplitAndTransform()(*manifold_scatra_sparse,
          *cond_slave_dof_map, *ssi_maps_->scatra_dof_row_map(), 1.0, &converter, nullptr,
          *ssi_manifold_scatra_sparse, true, true);
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::ManifoldMeshTyingStrategyBlock::apply_meshtying_to_manifold_scatra_matrix(
    std::shared_ptr<Core::LinAlg::SparseOperator> ssi_manifold_scatra_matrix,
    std::shared_ptr<const Core::LinAlg::SparseOperator> manifold_scatra_matrix)
{
  auto ssi_manifold_scatra_block =
      Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(ssi_manifold_scatra_matrix);
  auto manifold_scatra_block =
      Core::LinAlg::cast_to_const_block_sparse_matrix_base_and_check_success(
          manifold_scatra_matrix);

  for (int row = 0; row < condensed_block_dof_map_->num_maps(); ++row)
  {
    for (int col = 0; col < ssi_maps_->block_map_scatra()->num_maps(); ++col)
    {
      // add derivs. of interior/master dofs w.r.t. scatra dofs
      Coupling::Adapter::MatrixLogicalSplitAndTransform()(manifold_scatra_block->matrix(row, col),
          *condensed_block_dof_map_->map(row), *ssi_maps_->block_map_scatra()->map(col), 1.0,
          nullptr, nullptr, ssi_manifold_scatra_block->matrix(row, col), true, true);

      if (is_manifold_meshtying_)
      {
        for (const auto& block_meshtying : mesh_tying_block_handler())
        {
          auto meshtying = block_meshtying.first;
          auto cond_block_slave_dof_map = block_meshtying.second;
          auto converter_row = Coupling::Adapter::CouplingSlaveConverter(*meshtying[row]);

          // add derivs. of slave dofs w.r.t. scatra dofs
          Coupling::Adapter::MatrixLogicalSplitAndTransform()(
              manifold_scatra_block->matrix(row, col), *cond_block_slave_dof_map[row],
              *ssi_maps_->block_map_scatra()->map(col), 1.0, &converter_row, nullptr,
              ssi_manifold_scatra_block->matrix(row, col), true, true);
        }
      }
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::ManifoldMeshTyingStrategySparse::apply_meshtying_to_manifold_structure_matrix(
    std::shared_ptr<Core::LinAlg::SparseOperator> ssi_manifold_structure_matrix,
    std::shared_ptr<const Core::LinAlg::SparseOperator> manifold_structure_matrix,
    const bool do_uncomplete)
{
  auto ssi_manifold_structure_sparse =
      Core::LinAlg::cast_to_sparse_matrix_and_check_success(ssi_manifold_structure_matrix);
  auto manifold_structure_sparse =
      Core::LinAlg::cast_to_const_sparse_matrix_and_check_success(manifold_structure_matrix);

  auto temp_manifold_structure =
      SSI::Utils::SSIMatrices::setup_sparse_matrix(*ssi_maps_->scatra_manifold_dof_row_map());

  // add derivs. of interior/master dofs w.r.t. structure dofs
  Coupling::Adapter::MatrixLogicalSplitAndTransform()(*ssi_manifold_structure_sparse,
      *condensed_dof_map_, *ssi_maps_->structure_dof_row_map(), 1.0, nullptr, nullptr,
      *temp_manifold_structure, true, true);
  Coupling::Adapter::MatrixLogicalSplitAndTransform()(*manifold_structure_sparse,
      *condensed_dof_map_, *ssi_maps_->structure_dof_row_map(), 1.0, nullptr, nullptr,
      *temp_manifold_structure, true, true);

  if (is_manifold_meshtying_)
  {
    for (const auto& meshtying : ssi_mesh_tying()->mesh_tying_handlers())
    {
      auto coupling_adapter = meshtying->slave_master_coupling();

      auto cond_slave_dof_map = coupling_adapter->slave_dof_map();
      auto converter = Coupling::Adapter::CouplingSlaveConverter(*coupling_adapter);

      // add derivs. of slave dofs w.r.t. structure dofs
      Coupling::Adapter::MatrixLogicalSplitAndTransform()(*ssi_manifold_structure_sparse,
          *cond_slave_dof_map, *ssi_maps_->structure_dof_row_map(), 1.0, &converter, nullptr,
          *temp_manifold_structure, true, true);
      Coupling::Adapter::MatrixLogicalSplitAndTransform()(*manifold_structure_sparse,
          *cond_slave_dof_map, *ssi_maps_->structure_dof_row_map(), 1.0, &converter, nullptr,
          *temp_manifold_structure, true, true);
    }
  }

  ssi_manifold_structure_sparse->zero();
  if (do_uncomplete) ssi_manifold_structure_sparse->un_complete();
  temp_manifold_structure->complete(
      *ssi_maps_->structure_dof_row_map(), *ssi_maps_->scatra_manifold_dof_row_map());
  Core::LinAlg::matrix_add(
      *temp_manifold_structure, false, 1.0, *ssi_manifold_structure_sparse, 0.0);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::ManifoldMeshTyingStrategyBlock::apply_meshtying_to_manifold_structure_matrix(
    std::shared_ptr<Core::LinAlg::SparseOperator> ssi_manifold_structure_matrix,
    std::shared_ptr<const Core::LinAlg::SparseOperator> manifold_structure_matrix,
    const bool do_uncomplete)
{
  auto ssi_manifold_structure_block =
      Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(
          ssi_manifold_structure_matrix);
  auto manifold_structure_block =
      Core::LinAlg::cast_to_const_block_sparse_matrix_base_and_check_success(
          manifold_structure_matrix);

  auto temp_manifold_structure = SSI::Utils::SSIMatrices::setup_block_matrix(
      *ssi_maps_->block_map_scatra_manifold(), *ssi_maps_->block_map_structure());

  for (int row = 0; row < condensed_block_dof_map_->num_maps(); ++row)
  {
    // add derivs. of interior/master dofs w.r.t. structure dofs
    Coupling::Adapter::MatrixLogicalSplitAndTransform()(
        ssi_manifold_structure_block->matrix(row, 0), *condensed_block_dof_map_->map(row),
        *ssi_maps_->structure_dof_row_map(), 1.0, nullptr, nullptr,
        temp_manifold_structure->matrix(row, 0), true, true);
    Coupling::Adapter::MatrixLogicalSplitAndTransform()(manifold_structure_block->matrix(row, 0),
        *condensed_block_dof_map_->map(row), *ssi_maps_->structure_dof_row_map(), 1.0, nullptr,
        nullptr, temp_manifold_structure->matrix(row, 0), true, true);

    if (is_manifold_meshtying_)
    {
      for (const auto& block_meshtying : mesh_tying_block_handler())
      {
        auto meshtying = block_meshtying.first;
        auto cond_block_slave_dof_map = block_meshtying.second;
        auto converter_row = Coupling::Adapter::CouplingSlaveConverter(*meshtying[row]);

        // add derivs. of slave dofs w.r.t. structure dofs
        Coupling::Adapter::MatrixLogicalSplitAndTransform()(
            ssi_manifold_structure_block->matrix(row, 0), *cond_block_slave_dof_map[row],
            *ssi_maps_->structure_dof_row_map(), 1.0, &converter_row, nullptr,
            temp_manifold_structure->matrix(row, 0), true, true);
        Coupling::Adapter::MatrixLogicalSplitAndTransform()(
            manifold_structure_block->matrix(row, 0), *cond_block_slave_dof_map[row],
            *ssi_maps_->structure_dof_row_map(), 1.0, &converter_row, nullptr,
            temp_manifold_structure->matrix(row, 0), true, true);
      }
    }
  }

  ssi_manifold_structure_block->zero();
  if (do_uncomplete) ssi_manifold_structure_block->un_complete();
  temp_manifold_structure->complete();
  ssi_manifold_structure_block->add(*temp_manifold_structure, false, 1.0, 0.0);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::ManifoldMeshTyingStrategySparse::apply_meshtying_to_scatra_manifold_matrix(
    std::shared_ptr<Core::LinAlg::SparseOperator> ssi_scatra_manifold_matrix,
    std::shared_ptr<const Core::LinAlg::SparseOperator> scatra_manifold_matrix,
    const bool do_uncomplete)
{
  auto ssi_scatra_manifold_sparse =
      Core::LinAlg::cast_to_sparse_matrix_and_check_success(ssi_scatra_manifold_matrix);
  auto scatra_manifold_sparse =
      Core::LinAlg::cast_to_const_sparse_matrix_and_check_success(scatra_manifold_matrix);

  if (do_uncomplete) ssi_scatra_manifold_sparse->un_complete();

  // add derivs. of scatra w.r.t. interior/master dofs
  Coupling::Adapter::MatrixLogicalSplitAndTransform()(*scatra_manifold_sparse,
      *ssi_maps_->scatra_dof_row_map(), *condensed_dof_map_, 1.0, nullptr, nullptr,
      *ssi_scatra_manifold_sparse, true, true);

  if (is_manifold_meshtying_)
  {
    for (const auto& meshtying : ssi_mesh_tying()->mesh_tying_handlers())
    {
      auto coupling_adapter = meshtying->slave_master_coupling();

      auto cond_slave_dof_map = coupling_adapter->slave_dof_map();
      auto converter = Coupling::Adapter::CouplingSlaveConverter(*coupling_adapter);

      // add derivs. of scatra w.r.t. slave dofs
      Coupling::Adapter::MatrixLogicalSplitAndTransform()(*scatra_manifold_sparse,
          *ssi_maps_->scatra_dof_row_map(), *cond_slave_dof_map, 1.0, nullptr, &converter,
          *ssi_scatra_manifold_sparse, true, true);
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::ManifoldMeshTyingStrategyBlock::apply_meshtying_to_scatra_manifold_matrix(
    std::shared_ptr<Core::LinAlg::SparseOperator> ssi_scatra_manifold_matrix,
    std::shared_ptr<const Core::LinAlg::SparseOperator> scatra_manifold_matrix,
    const bool do_uncomplete)
{
  auto ssi_scatra_manifold_block =
      Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(ssi_scatra_manifold_matrix);
  auto scatra_manifold_block =
      Core::LinAlg::cast_to_const_block_sparse_matrix_base_and_check_success(
          scatra_manifold_matrix);

  if (do_uncomplete) ssi_scatra_manifold_block->un_complete();

  for (int row = 0; row < ssi_maps_->block_map_scatra()->num_maps(); ++row)
  {
    for (int col = 0; col < condensed_block_dof_map_->num_maps(); ++col)
    {
      // add derivs. of scatra w.r.t. interior/master dofs
      Coupling::Adapter::MatrixLogicalSplitAndTransform()(scatra_manifold_block->matrix(row, col),
          *ssi_maps_->block_map_scatra()->map(row), *condensed_block_dof_map_->map(col), 1.0,
          nullptr, nullptr, ssi_scatra_manifold_block->matrix(row, col), true, true);

      if (is_manifold_meshtying_)
      {
        for (const auto& block_meshtying : mesh_tying_block_handler())
        {
          auto meshtying = block_meshtying.first;
          auto cond_block_slave_dof_map = block_meshtying.second;
          auto converter_col = Coupling::Adapter::CouplingSlaveConverter(*meshtying[col]);

          // add derivs. of scatra w.r.t. slave dofs
          Coupling::Adapter::MatrixLogicalSplitAndTransform()(
              scatra_manifold_block->matrix(row, col), *ssi_maps_->block_map_scatra()->map(row),
              *cond_block_slave_dof_map[col], 1.0, nullptr, &converter_col,
              ssi_scatra_manifold_block->matrix(row, col), true, true);
        }
      }
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<SSI::ManifoldMeshTyingStrategyBase> SSI::build_manifold_mesh_tying_strategy(
    std::shared_ptr<Core::FE::Discretization> scatra_manifold_dis,
    std::shared_ptr<Utils::SSIMaps> ssi_maps, const bool is_manifold_meshtying,
    Core::LinAlg::MatrixType matrixtype_manifold)
{
  std::shared_ptr<SSI::ManifoldMeshTyingStrategyBase> meshtyingstrategy = nullptr;

  switch (matrixtype_manifold)
  {
    case Core::LinAlg::MatrixType::block_condition:
    case Core::LinAlg::MatrixType::block_condition_dof:
    {
      meshtyingstrategy = std::make_shared<SSI::ManifoldMeshTyingStrategyBlock>(
          scatra_manifold_dis, ssi_maps, is_manifold_meshtying);
      break;
    }
    case Core::LinAlg::MatrixType::sparse:
    {
      meshtyingstrategy = std::make_shared<SSI::ManifoldMeshTyingStrategySparse>(
          scatra_manifold_dis, ssi_maps, is_manifold_meshtying);
      break;
    }

    default:
    {
      FOUR_C_THROW("unknown matrix type of Manifold field");
      break;
    }
  }

  return meshtyingstrategy;
}

FOUR_C_NAMESPACE_CLOSE

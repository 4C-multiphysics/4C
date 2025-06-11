// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_ssi_monolithic_evaluate_OffDiag.hpp"

#include "4C_adapter_scatra_base_algorithm.hpp"
#include "4C_adapter_str_ssiwrapper.hpp"
#include "4C_coupling_adapter_converter.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_assemblestrategy.hpp"
#include "4C_linalg_mapextractor.hpp"
#include "4C_linalg_sparseoperator.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_scatra_ele_action.hpp"
#include "4C_scatra_timint_implicit.hpp"
#include "4C_scatra_timint_meshtying_strategy_s2i.hpp"
#include "4C_ssi_utils.hpp"
#include "4C_structure_new_enum_lists.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
SSI::ScatraStructureOffDiagCoupling::ScatraStructureOffDiagCoupling(
    std::shared_ptr<const Core::LinAlg::MultiMapExtractor> block_map_structure,
    std::shared_ptr<const Core::LinAlg::Map> full_map_structure,
    std::shared_ptr<const SSI::Utils::SSIMeshTying> ssi_structure_meshtying,
    std::shared_ptr<const ScaTra::MeshtyingStrategyS2I> meshtying_strategy_s2i,
    std::shared_ptr<ScaTra::ScaTraTimIntImpl> scatra,
    std::shared_ptr<Adapter::SSIStructureWrapper> structure)
    : block_map_structure_(std::move(block_map_structure)),
      full_map_structure_(std::move(full_map_structure)),
      meshtying_strategy_s2i_(std::move(meshtying_strategy_s2i)),
      scatra_(std::move(scatra)),
      structure_(std::move(structure)),
      ssi_structure_meshtying_(std::move(ssi_structure_meshtying))
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
SSI::ScatraManifoldStructureOffDiagCoupling::ScatraManifoldStructureOffDiagCoupling(
    std::shared_ptr<const Core::LinAlg::MultiMapExtractor> block_map_structure,
    std::shared_ptr<const Core::LinAlg::Map> full_map_structure,
    std::shared_ptr<const SSI::Utils::SSIMeshTying> ssi_structure_meshtying,
    std::shared_ptr<const ScaTra::MeshtyingStrategyS2I> meshtying_strategy_s2i,
    std::shared_ptr<ScaTra::ScaTraTimIntImpl> scatra,
    std::shared_ptr<ScaTra::ScaTraTimIntImpl> scatra_manifold,
    std::shared_ptr<Adapter::SSIStructureWrapper> structure)
    : ScatraStructureOffDiagCoupling(std::move(block_map_structure), std::move(full_map_structure),
          std::move(ssi_structure_meshtying), std::move(meshtying_strategy_s2i), std::move(scatra),
          std::move(structure)),
      scatra_manifold_(std::move(scatra_manifold))
{
}


/*-----------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------*/
void SSI::ScatraStructureOffDiagCoupling::evaluate_off_diag_block_scatra_structure_domain(
    std::shared_ptr<Core::LinAlg::SparseOperator> scatrastructureblock)
{
  // create parameter list for element evaluation
  Teuchos::ParameterList eleparams;

  // action for elements
  Core::Utils::add_enum_class_to_parameter_list<ScaTra::Action>(
      "action", ScaTra::Action::calc_scatra_mono_odblock_mesh, eleparams);

  // add state vectors to scalar transport discretization
  scatra_field()->add_time_integration_specific_vectors();

  // create strategy for assembly of scatra-structure matrix block
  Core::FE::AssembleStrategy strategyscatrastructure(
      0,  // row assembly based on number of dofset associated with scalar transport dofs on
          // scalar transport discretization
      1,  // column assembly based on number of dofset associated with structural dofs on scalar
      // transport discretization
      scatrastructureblock,  // scatra-structure matrix block
      nullptr,               // no additional matrices or vectors
      nullptr, nullptr, nullptr);

  // assemble scatra-structure matrix block
  scatra_field()->discretization()->evaluate(eleparams, strategyscatrastructure);
}

/*-----------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------*/
void SSI::ScatraStructureOffDiagCoupling::evaluate_off_diag_block_scatra_manifold_structure_domain(
    std::shared_ptr<Core::LinAlg::SparseOperator> scatramanifoldstructureblock)
{
  FOUR_C_THROW("not implemented");
}

/*-----------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------*/
void SSI::ScatraManifoldStructureOffDiagCoupling::
    evaluate_off_diag_block_scatra_manifold_structure_domain(
        std::shared_ptr<Core::LinAlg::SparseOperator> scatramanifoldstructureblock)
{
  // create parameter list for element evaluation
  Teuchos::ParameterList eleparams;

  // action for elements
  Core::Utils::add_enum_class_to_parameter_list<ScaTra::Action>(
      "action", ScaTra::Action::calc_scatra_mono_odblock_mesh, eleparams);

  // add state vectors to scalar transport discretization
  scatra_manifold_->add_time_integration_specific_vectors();

  // create strategy for assembly of scatra-structure matrix block
  Core::FE::AssembleStrategy strategyscatrastructure(
      0,  // row assembly based on number of dofset associated with scalar transport dofs on
          // scalar transport discretization
      1,  // column assembly based on number of dofset associated with structural dofs on scalar
      // transport discretization
      scatramanifoldstructureblock,  // scatra-structure matrix block
      nullptr,                       // no additional matrices or vectors
      nullptr, nullptr, nullptr);

  // assemble scatra-structure matrix block
  scatra_manifold_->discretization()->evaluate(eleparams, strategyscatrastructure);
}

/*-----------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------*/
void SSI::ScatraStructureOffDiagCoupling::evaluate_off_diag_block_scatra_structure_interface(
    Core::LinAlg::SparseOperator& scatrastructureinterface)
{
  // slave and master matrix for evaluation of conditions
  std::shared_ptr<Core::LinAlg::SparseOperator> slavematrix(nullptr);
  std::shared_ptr<Core::LinAlg::SparseOperator> mastermatrix(nullptr);
  switch (scatra_field()->matrix_type())
  {
    case Core::LinAlg::MatrixType::block_condition:
    case Core::LinAlg::MatrixType::block_condition_dof:
    {
      slavematrix = std::make_shared<
          Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(

          *block_map_structure_, meshtying_strategy_s2i_->block_maps_slave(), 81, false, true);
      mastermatrix = std::make_shared<
          Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(

          *block_map_structure_, meshtying_strategy_s2i_->block_maps_master(), 81, false, true);
      break;
    }
    case Core::LinAlg::MatrixType::sparse:
    {
      slavematrix = std::make_shared<Core::LinAlg::SparseMatrix>(
          *meshtying_strategy_s2i_->coupling_adapter()->slave_dof_map(), 27, false, true);
      mastermatrix = std::make_shared<Core::LinAlg::SparseMatrix>(
          *meshtying_strategy_s2i_->coupling_adapter()->master_dof_map(), 27, false, true);
      break;
    }
    default:
    {
      FOUR_C_THROW("Invalid matrix type associated with scalar transport field!");
      break;
    }
  }

  // evaluate symmetric interface contributions on slave side
  evaluate_scatra_structure_symmetric_interface_contributions_slave_side(slavematrix);

  // copy symmetric interface contributions from slave side to master side
  copy_slave_to_master_scatra_structure_symmetric_interface_contributions(
      slavematrix, mastermatrix);

  // evaluate non-symmetric interface contributions
  evaluate_scatra_structure_non_symmetric_interface_contributions_slave_side(
      slavematrix, mastermatrix);

  // add contributions from slave side and master side
  scatrastructureinterface.add(*slavematrix, false, 1.0, 1.0);
  scatrastructureinterface.add(*mastermatrix, false, 1.0, 1.0);
}

/*-----------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------*/
void SSI::ScatraStructureOffDiagCoupling::evaluate_off_diag_block_structure_scatra_domain(
    std::shared_ptr<Core::LinAlg::SparseOperator> structurescatradomain) const
{
  // create parameter list for element evaluation and fill it
  Teuchos::ParameterList eleparams;
  // set action
  eleparams.set("action", "calc_struct_stiffscalar");

  // linearization of structural residuals w.r.t. elch
  eleparams.set<int>("differentiationtype", static_cast<int>(Solid::DifferentiationType::elch));

  // set time
  eleparams.set<double>("total time", structure_->time());
  // set numscatradofspernode
  eleparams.set<int>("numscatradofspernode", scatra_field()->num_dof_per_node());

  // remove state vectors from structure discretization
  structure_->discretization()->clear_state();

  // set the current displacement state vector
  structure_->discretization()->set_state("displacement", *structure_->dispnp());

  // create strategy for assembly of structure-scatra matrix block
  Core::FE::AssembleStrategy strategystructurescatra(
      0,  // row assembly based on number of dofset associated with structure dofs on structural
          // discretization
      1,  // column assembly based on number of dofset associated with scalar transport dofs on
      // structural discretization
      structurescatradomain,  // structure-scatra matrix block
      nullptr,                // no additional matrices or vectors needed
      nullptr, nullptr, nullptr);

  // assemble structure-scatra matrix block
  structure_->discretization()->evaluate(eleparams, strategystructurescatra);

  // need to scale structurescatrablock_ with 'timefac' (e.g. with theta for OST-scheme) to get
  // correct implementation
  const double timeintparam = structure_->tim_int_param();
  // scale with theta
  structurescatradomain->scale(1.0 - timeintparam);
}

/*-----------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------*/
void SSI::ScatraStructureOffDiagCoupling::
    copy_slave_to_master_scatra_structure_symmetric_interface_contributions(
        std::shared_ptr<const Core::LinAlg::SparseOperator> slavematrix,
        std::shared_ptr<Core::LinAlg::SparseOperator>& mastermatrix)
{
  mastermatrix->zero();
  switch (scatra_field()->matrix_type())
  {
    case Core::LinAlg::MatrixType::block_condition:
    case Core::LinAlg::MatrixType::block_condition_dof:
    {
      const int numberscatrablocks = scatra_field()->dof_block_maps()->num_maps();

      // cast master and slave matrix
      auto blockslavematrix =
          Core::LinAlg::cast_to_const_block_sparse_matrix_base_and_check_success(slavematrix);
      auto blockmastermatrix =
          Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(mastermatrix);

      // initialize auxiliary system matrix for linearizations of master-side scatra fluxes w.r.t.
      // master-side structural dofs
      Core::LinAlg::SparseMatrix mastermatrixsparse(
          *meshtying_strategy_s2i_->coupling_adapter()->master_dof_map(), 27, false, true);

      // derive linearizations of master-side scatra fluxes w.r.t. master-side structural dofs and
      // assemble into auxiliary system matrix
      for (int iblock = 0; iblock < numberscatrablocks; ++iblock)
      {
        for (const auto& meshtying : ssi_structure_meshtying_->mesh_tying_handlers())
        {
          auto slave_dof_map = meshtying->slave_master_coupling()->slave_dof_map();
          auto slave_side_converter_struct = meshtying->slave_side_converter();

          auto slave_side_converter_scatra = Coupling::Adapter::CouplingSlaveConverter(
              *meshtying_strategy_s2i_->coupling_adapter());

          Coupling::Adapter::MatrixLogicalSplitAndTransform()(blockslavematrix->matrix(iblock, 0),
              *meshtying_strategy_s2i_->coupling_adapter()->slave_dof_map(), *slave_dof_map, -1.0,
              &slave_side_converter_scatra, &(*slave_side_converter_struct), mastermatrixsparse,
              true, true);
        }
      }

      // finalize auxiliary system matrix
      mastermatrixsparse.complete(*full_map_structure(), *scatra_field()->dof_row_map());

      // split auxiliary system matrix and assemble into scatra-structure matrix block
      auto mastermatrix_split =
          Core::LinAlg::split_matrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
              mastermatrixsparse, *block_map_structure_, *scatra_field()->dof_block_maps());
      mastermatrix_split->complete();
      blockmastermatrix->add(*mastermatrix_split, false, 1.0, 1.0);

      mastermatrix->complete();

      break;
    }

    case Core::LinAlg::MatrixType::sparse:
    {
      // cast master and slave matrix
      auto sparseslavematrix =
          Core::LinAlg::cast_to_const_sparse_matrix_and_check_success(slavematrix);
      auto sparsemastermatrix = Core::LinAlg::cast_to_sparse_matrix_and_check_success(mastermatrix);

      // copy slave side values to master side and scale with minus 1. Insert into
      // scatrastructureinterface_sparse
      for (const auto& meshtying : ssi_structure_meshtying_->mesh_tying_handlers())
      {
        auto slave_dof_map = meshtying->slave_master_coupling()->slave_dof_map();
        auto slave_side_converter_struct = meshtying->slave_side_converter();
        auto slave_side_converter_scatra =
            Coupling::Adapter::CouplingSlaveConverter(*meshtying_strategy_s2i_->coupling_adapter());

        Coupling::Adapter::MatrixLogicalSplitAndTransform()(*sparseslavematrix,
            *meshtying_strategy_s2i_->coupling_adapter()->slave_dof_map(), *slave_dof_map, -1.0,
            &slave_side_converter_scatra, &(*slave_side_converter_struct), *sparsemastermatrix,
            true, true);
      }
      // finalize
      mastermatrix->complete(
          *full_map_structure_, *meshtying_strategy_s2i_->coupling_adapter()->master_dof_map());
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
void SSI::ScatraStructureOffDiagCoupling::
    evaluate_scatra_structure_non_symmetric_interface_contributions_slave_side(
        std::shared_ptr<Core::LinAlg::SparseOperator> slavematrix,
        std::shared_ptr<Core::LinAlg::SparseOperator> mastermatrix)
{
  // create parameter list for element evaluation
  Teuchos::ParameterList condparams;

  // action for elements
  Core::Utils::add_enum_class_to_parameter_list<ScaTra::BoundaryAction>(
      "action", ScaTra::BoundaryAction::calc_s2icoupling_capacitance_od, condparams);

  // linearization of boundary flux w.r.t. displacement
  Core::Utils::add_enum_class_to_parameter_list<ScaTra::DifferentiationType>(
      "differentiationtype", ScaTra::DifferentiationType::disp, condparams);

  // add state vectors to scalar transport discretization
  scatra_field()->add_time_integration_specific_vectors();

  // set up necessary matrices
  std::shared_ptr<Core::LinAlg::SparseOperator>
      scatra_slave_flux_structure_slave_dofs_on_scatra_slave_matrix;
  std::shared_ptr<Core::LinAlg::SparseOperator>
      scatra_master_flux_on_scatra_slave_structure_slave_dofs_on_scatra_slave_matrix;
  std::shared_ptr<Core::LinAlg::SparseOperator>
      scatra_master_flux_on_scatra_slave_dofs_structure_slave_dofs_matrix;

  if (scatra_field()->matrix_type() == Core::LinAlg::MatrixType::sparse)
  {
    scatra_slave_flux_structure_slave_dofs_on_scatra_slave_matrix =
        std::make_shared<Core::LinAlg::SparseMatrix>(
            *meshtying_strategy_s2i_->coupling_adapter()->slave_dof_map(), 27, false, true);
    scatra_master_flux_on_scatra_slave_structure_slave_dofs_on_scatra_slave_matrix =
        std::make_shared<Core::LinAlg::SparseMatrix>(
            *meshtying_strategy_s2i_->coupling_adapter()->slave_dof_map(), 27, false, true);
    scatra_master_flux_on_scatra_slave_dofs_structure_slave_dofs_matrix =
        std::make_shared<Core::LinAlg::SparseMatrix>(
            *meshtying_strategy_s2i_->coupling_adapter()->slave_dof_map(), 27, false, true);
  }
  else
  {
    scatra_slave_flux_structure_slave_dofs_on_scatra_slave_matrix =
        std::make_shared<Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(
            *block_map_structure_, meshtying_strategy_s2i_->block_maps_slave(), 81, false, true);
    scatra_master_flux_on_scatra_slave_structure_slave_dofs_on_scatra_slave_matrix =
        std::make_shared<Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(
            *block_map_structure_, meshtying_strategy_s2i_->block_maps_slave(), 81, false, true);
    scatra_master_flux_on_scatra_slave_dofs_structure_slave_dofs_matrix =
        std::make_shared<Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(
            *block_map_structure_, meshtying_strategy_s2i_->block_maps_slave(), 81, false, true);
  }

  // create strategy for assembly of auxiliary system matrix
  Core::FE::AssembleStrategy strategyscatras2istructure(
      0,  // row assembly based on number of dofset associated with scalar transport dofs on
          // scalar transport discretization
      1,  // column assembly based on number of dofset associated with structural dofs on
      // structural discretization
      scatra_slave_flux_structure_slave_dofs_on_scatra_slave_matrix,
      scatra_master_flux_on_scatra_slave_structure_slave_dofs_on_scatra_slave_matrix,
      // no additional vectors
      nullptr, nullptr, nullptr);

  // evaluate scatra-scatra interface coupling
  for (auto kinetics_slave_cond :
      meshtying_strategy_s2i_->kinetics_conditions_meshtying_slave_side())
  {
    if (kinetics_slave_cond.second->parameters().get<Inpar::S2I::KineticModels>("KINETIC_MODEL") ==
        static_cast<int>(Inpar::S2I::kinetics_butlervolmerreducedcapacitance))
    {
      // collect condition specific data and store to scatra boundary parameter class
      meshtying_strategy_s2i_->set_condition_specific_scatra_parameters(
          *kinetics_slave_cond.second);
      // evaluate the condition
      scatra_field()->discretization()->evaluate_condition(
          condparams, strategyscatras2istructure, "S2IKinetics", kinetics_slave_cond.first);
    }
  }

  // finalize scatra-structure matrix block
  switch (scatra_field()->matrix_type())
  {
    case Core::LinAlg::MatrixType::sparse:
    {
      scatra_slave_flux_structure_slave_dofs_on_scatra_slave_matrix->complete(
          *full_map_structure_, *meshtying_strategy_s2i_->coupling_adapter()->slave_dof_map());
      scatra_master_flux_on_scatra_slave_structure_slave_dofs_on_scatra_slave_matrix->complete(
          *full_map_structure_, *meshtying_strategy_s2i_->coupling_adapter()->slave_dof_map());

      auto scatra_slave_flux_structure_slave_dofs_on_scatra_slave_matrix_sparse =
          Core::LinAlg::cast_to_const_sparse_matrix_and_check_success(
              scatra_slave_flux_structure_slave_dofs_on_scatra_slave_matrix);
      auto slavematrix_sparse = Core::LinAlg::cast_to_sparse_matrix_and_check_success(slavematrix);

      auto scatra_master_flux_on_scatra_slave_structure_slave_dofs_on_scatra_slave_matrix_sparse =
          Core::LinAlg::cast_to_const_sparse_matrix_and_check_success(
              scatra_master_flux_on_scatra_slave_structure_slave_dofs_on_scatra_slave_matrix);
      auto scatra_master_flux_on_scatra_slave_dofs_structure_slave_dofs_matrix_sparse =
          Core::LinAlg::cast_to_sparse_matrix_and_check_success(
              scatra_master_flux_on_scatra_slave_dofs_structure_slave_dofs_matrix);
      auto mastermatrix_sparse =
          Core::LinAlg::cast_to_sparse_matrix_and_check_success(mastermatrix);

      // "slave side" from scatra and from structure do not need to be the same nodes.
      // Linearization is evaluated on scatra slave side node --> Transformation needed
      for (const auto& meshtying : ssi_structure_meshtying_->mesh_tying_handlers())
      {
        auto slave_slave_transformation = meshtying->slave_slave_transformation();

        // converter between old slave dofs from input and actual slave dofs from current mesh tying
        // adapter
        auto slave_slave_converter =
            Coupling::Adapter::CouplingSlaveConverter(*slave_slave_transformation);

        // old slave dofs from input
        auto slave_map = slave_slave_transformation->slave_dof_map();

        // add slave contributions to slave matrix
        Coupling::Adapter::MatrixLogicalSplitAndTransform()(
            *scatra_slave_flux_structure_slave_dofs_on_scatra_slave_matrix_sparse,
            *meshtying_strategy_s2i_->coupling_adapter()->slave_dof_map(), *slave_map, 1.0, nullptr,
            &slave_slave_converter, *slavematrix_sparse, true, true);
        // convert structure slave dofs on scatra discretization to slave dofs on structure
        // discretization
        Coupling::Adapter::MatrixLogicalSplitAndTransform()(
            *scatra_master_flux_on_scatra_slave_structure_slave_dofs_on_scatra_slave_matrix_sparse,
            *meshtying_strategy_s2i_->coupling_adapter()->slave_dof_map(), *slave_map, 1.0, nullptr,
            &slave_slave_converter,
            *scatra_master_flux_on_scatra_slave_dofs_structure_slave_dofs_matrix_sparse, true,
            true);

        scatra_master_flux_on_scatra_slave_dofs_structure_slave_dofs_matrix_sparse->complete(
            *full_map_structure_, *meshtying_strategy_s2i_->coupling_adapter()->slave_dof_map());

        auto slave_dof_map = meshtying->slave_master_coupling()->slave_dof_map();
        auto slave_side_converter_struct = meshtying->slave_side_converter();
        auto slave_side_converter_scatra =
            Coupling::Adapter::CouplingSlaveConverter(*meshtying_strategy_s2i_->coupling_adapter());

        Coupling::Adapter::MatrixLogicalSplitAndTransform()(
            *scatra_master_flux_on_scatra_slave_dofs_structure_slave_dofs_matrix_sparse,
            *meshtying_strategy_s2i_->coupling_adapter()->slave_dof_map(), *slave_dof_map, 1.0,
            &slave_side_converter_scatra, &(*slave_side_converter_struct), *mastermatrix_sparse,
            true, true);
      }

      break;
    }

    case Core::LinAlg::MatrixType::block_condition:
    case Core::LinAlg::MatrixType::block_condition_dof:
    {
      scatra_slave_flux_structure_slave_dofs_on_scatra_slave_matrix->complete();
      scatra_master_flux_on_scatra_slave_structure_slave_dofs_on_scatra_slave_matrix->complete();

      slavematrix->un_complete();
      mastermatrix->un_complete();

      auto scatra_slave_flux_structure_slave_dofs_on_scatra_slave_matrix_block =
          Core::LinAlg::cast_to_const_block_sparse_matrix_base_and_check_success(
              scatra_slave_flux_structure_slave_dofs_on_scatra_slave_matrix);
      auto slavematrix_block =
          Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(slavematrix);

      auto scatra_master_flux_on_scatra_slave_structure_slave_dofs_on_scatra_slave_matrix_block =
          Core::LinAlg::cast_to_const_block_sparse_matrix_base_and_check_success(
              scatra_master_flux_on_scatra_slave_structure_slave_dofs_on_scatra_slave_matrix);
      auto mastermatrix_block =
          Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(mastermatrix);
      auto scatra_master_flux_on_scatra_slave_dofs_structure_slave_dofs_matrix_block =
          Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(
              scatra_master_flux_on_scatra_slave_dofs_structure_slave_dofs_matrix);

      // initialize auxiliary system matrix for linearizations of master-side scatra fluxes w.r.t.
      // master-side structural dofs
      Core::LinAlg::SparseMatrix mastermatrixsparse(
          *meshtying_strategy_s2i_->coupling_adapter()->master_dof_map(), 27, false, true);

      // "slave side" from scatra and from structure do not need to be the same nodes.
      // Linearization is evaluated on scatra slave side node --> Transformation needed
      for (const auto& meshtying : ssi_structure_meshtying_->mesh_tying_handlers())
      {
        auto slave_slave_transformation = meshtying->slave_slave_transformation();
        // converter between old slave dofs from input and actual slave dofs from current mesh tying
        // adapter
        auto slave_slave_converter =
            Coupling::Adapter::CouplingSlaveConverter(*slave_slave_transformation);

        // old slave dofs from input
        auto slave_map = slave_slave_transformation->slave_dof_map();

        for (int iblock = 0; iblock < scatra_field()->dof_block_maps()->num_maps(); ++iblock)
        {
          auto scatra_slave_flux_structure_slave_dofs_on_scatra_slave_iblock =
              scatra_slave_flux_structure_slave_dofs_on_scatra_slave_matrix_block->matrix(
                  iblock, 0);
          auto slave_iblock = slavematrix_block->matrix(iblock, 0);

          auto scatra_master_flux_on_scatra_slave_structure_slave_dofs_on_scatra_slave_iblock =
              scatra_master_flux_on_scatra_slave_structure_slave_dofs_on_scatra_slave_matrix_block
                  ->matrix(iblock, 0);
          auto scatra_master_flux_on_scatra_slave_dofs_structure_slave_dofs_iblock =
              scatra_master_flux_on_scatra_slave_dofs_structure_slave_dofs_matrix_block->matrix(
                  iblock, 0);

          auto scatra_block_mapi =
              Core::LinAlg::intersect_map(*scatra_field()->dof_block_maps()->map(iblock),
                  *meshtying_strategy_s2i_->coupling_adapter()->slave_dof_map());

          Coupling::Adapter::MatrixLogicalSplitAndTransform()(
              scatra_slave_flux_structure_slave_dofs_on_scatra_slave_iblock, *scatra_block_mapi,
              *slave_map, 1.0, nullptr, &slave_slave_converter, slave_iblock, true, true);
          Coupling::Adapter::MatrixLogicalSplitAndTransform()(
              scatra_master_flux_on_scatra_slave_structure_slave_dofs_on_scatra_slave_iblock,
              *scatra_block_mapi, *slave_map, 1.0, nullptr, &slave_slave_converter,
              scatra_master_flux_on_scatra_slave_dofs_structure_slave_dofs_iblock, true, true);

          scatra_master_flux_on_scatra_slave_dofs_structure_slave_dofs_iblock.complete();

          auto slave_dof_map = meshtying->slave_master_coupling()->slave_dof_map();
          auto slave_side_converter_struct = meshtying->slave_side_converter();
          auto slave_side_converter_scatra = Coupling::Adapter::CouplingSlaveConverter(
              *meshtying_strategy_s2i_->coupling_adapter());

          Coupling::Adapter::MatrixLogicalSplitAndTransform()(
              scatra_master_flux_on_scatra_slave_dofs_structure_slave_dofs_iblock,
              *meshtying_strategy_s2i_->coupling_adapter()->slave_dof_map(), *slave_dof_map, 1.0,
              &slave_side_converter_scatra, &(*slave_side_converter_struct), mastermatrixsparse,
              true, true);
        }
      }

      // finalize auxiliary system matrix
      mastermatrixsparse.complete(*full_map_structure(), *scatra_field()->dof_row_map());

      // split auxiliary system matrix and assemble into scatra-structure matrix block
      auto mastermatrix_split =
          Core::LinAlg::split_matrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
              mastermatrixsparse, *block_map_structure_, *scatra_field()->dof_block_maps());
      mastermatrix_split->complete();
      mastermatrix_block->add(*mastermatrix_split, false, 1.0, 1.0);

      mastermatrix->complete();
      slavematrix->complete();
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
void SSI::ScatraStructureOffDiagCoupling::
    evaluate_scatra_structure_symmetric_interface_contributions_slave_side(
        std::shared_ptr<Core::LinAlg::SparseOperator> slavematrix)
{
  // create parameter list for element evaluation
  Teuchos::ParameterList condparams;

  // action for elements
  Core::Utils::add_enum_class_to_parameter_list<ScaTra::BoundaryAction>(
      "action", ScaTra::BoundaryAction::calc_s2icoupling_od, condparams);

  // linearization of boundary flux w.r.t. displacement
  Core::Utils::add_enum_class_to_parameter_list<ScaTra::DifferentiationType>(
      "differentiationtype", ScaTra::DifferentiationType::disp, condparams);

  // add state vectors to scalar transport discretization
  scatra_field()->add_time_integration_specific_vectors();

  std::shared_ptr<Core::LinAlg::SparseOperator> evaluate_matrix;
  if (scatra_field()->matrix_type() == Core::LinAlg::MatrixType::sparse)
  {
    evaluate_matrix = std::make_shared<Core::LinAlg::SparseMatrix>(
        *meshtying_strategy_s2i_->coupling_adapter()->slave_dof_map(), 27, false, true);
  }
  else
  {
    evaluate_matrix =
        std::make_shared<Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(
            *block_map_structure_, meshtying_strategy_s2i_->block_maps_slave(), 81, false, true);
  }

  // create strategy for assembly of auxiliary system matrix
  Core::FE::AssembleStrategy strategyscatrastructures2i(
      0,  // row assembly based on number of dofset associated with scalar transport dofs on
          // scalar transport discretization
      1,  // column assembly based on number of dofset associated with structural dofs on
          // structural discretization
      evaluate_matrix,  // auxiliary system matrix
      nullptr,          // no additional matrices of vectors
      nullptr, nullptr, nullptr);

  // evaluate scatra-scatra interface coupling
  for (auto kinetics_slave_cond :
      meshtying_strategy_s2i_->kinetics_conditions_meshtying_slave_side())
  {
    if (kinetics_slave_cond.second->parameters().get<Inpar::S2I::KineticModels>("KINETIC_MODEL") !=
        static_cast<int>(Inpar::S2I::kinetics_nointerfaceflux))
    {
      // collect condition specific data and store to scatra boundary parameter class
      meshtying_strategy_s2i_->set_condition_specific_scatra_parameters(
          *kinetics_slave_cond.second);
      // evaluate the condition
      scatra_field()->discretization()->evaluate_condition(
          condparams, strategyscatrastructures2i, "S2IKinetics", kinetics_slave_cond.first);
    }
  }

  // finalize scatra-structure matrix block
  switch (scatra_field()->matrix_type())
  {
    case Core::LinAlg::MatrixType::block_condition:
    case Core::LinAlg::MatrixType::block_condition_dof:
    {
      evaluate_matrix->complete();

      auto evaluate_matrix_block =
          Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(evaluate_matrix);
      auto slavematrix_block =
          Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(slavematrix);

      // "slave side" from scatra and from structure do not need to be the same nodes.
      // Linearization is evaluated on scatra slave side node --> Transformation needed
      for (const auto& meshtying : ssi_structure_meshtying_->mesh_tying_handlers())
      {
        auto slave_slave_transformation = meshtying->slave_slave_transformation();
        // converter between old slave dofs from input and actual slave dofs from current mesh tying
        // adapter
        auto slave_slave_converter =
            Coupling::Adapter::CouplingSlaveConverter(*slave_slave_transformation);

        // old slave dofs from input
        auto slave_map = slave_slave_transformation->slave_dof_map();

        for (int iblock = 0; iblock < scatra_field()->dof_block_maps()->num_maps(); ++iblock)
        {
          auto evaluate_iblock = evaluate_matrix_block->matrix(iblock, 0);
          auto slave_iblock = slavematrix_block->matrix(iblock, 0);

          auto scatra_slave_block_mapi =
              Core::LinAlg::intersect_map(*scatra_field()->dof_block_maps()->map(iblock),
                  *meshtying_strategy_s2i_->coupling_adapter()->slave_dof_map());

          Coupling::Adapter::MatrixLogicalSplitAndTransform()(evaluate_iblock,
              *scatra_slave_block_mapi, *slave_map, 1.0, nullptr, &slave_slave_converter,
              slave_iblock, true, true);
        }
      }
      slavematrix->complete();
      break;
    }

    case Core::LinAlg::MatrixType::sparse:
    {
      evaluate_matrix->complete(
          *full_map_structure_, *meshtying_strategy_s2i_->coupling_adapter()->slave_dof_map());

      auto evaluate_matrix_sparse =
          Core::LinAlg::cast_to_const_sparse_matrix_and_check_success(evaluate_matrix);
      auto slavematrix_sparse = Core::LinAlg::cast_to_sparse_matrix_and_check_success(slavematrix);

      // "slave side" from scatra and from structure do not need to be the same nodes.
      // Linearization is evaluated on scatra slave side node --> Transformation needed
      for (const auto& meshtying : ssi_structure_meshtying_->mesh_tying_handlers())
      {
        auto slave_slave_transformation = meshtying->slave_slave_transformation();
        // converter between old slave dofs from input and actual slave dofs from current mesh tying
        // adapter
        auto slave_slave_converter =
            Coupling::Adapter::CouplingSlaveConverter(*slave_slave_transformation);

        // old slave dofs from input
        auto slave_map = slave_slave_transformation->slave_dof_map();

        Coupling::Adapter::MatrixLogicalSplitAndTransform()(*evaluate_matrix_sparse,
            *meshtying_strategy_s2i_->coupling_adapter()->slave_dof_map(), *slave_map, 1.0, nullptr,
            &slave_slave_converter, *slavematrix_sparse, true, true);
      }
      slavematrix->complete(
          *full_map_structure_, *meshtying_strategy_s2i_->coupling_adapter()->slave_dof_map());

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
SSI::ScatraStructureOffDiagCouplingSSTI::ScatraStructureOffDiagCouplingSSTI(
    std::shared_ptr<const Core::LinAlg::MultiMapExtractor> block_map_structure,
    std::shared_ptr<const Core::LinAlg::Map> full_map_scatra,
    std::shared_ptr<const Core::LinAlg::Map> full_map_structure,
    std::shared_ptr<const SSI::Utils::SSIMeshTying> ssi_structure_meshtying,
    std::shared_ptr<const ScaTra::MeshtyingStrategyS2I> meshtying_strategy_s2i,
    std::shared_ptr<ScaTra::ScaTraTimIntImpl> scatra,
    std::shared_ptr<Adapter::SSIStructureWrapper> structure)
    : ScatraStructureOffDiagCoupling(std::move(block_map_structure), std::move(full_map_structure),
          std::move(ssi_structure_meshtying), std::move(meshtying_strategy_s2i), std::move(scatra),
          std::move(structure)),
      full_map_scatra_(std::move(full_map_scatra))
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::ScatraStructureOffDiagCouplingSSTI::evaluate_off_diag_block_structure_scatra_domain(
    std::shared_ptr<Core::LinAlg::SparseOperator> structurescatradomain) const
{
  ScatraStructureOffDiagCoupling::evaluate_off_diag_block_structure_scatra_domain(
      structurescatradomain);

  // finalize structure-scatra matrix block
  switch (scatra_field()->matrix_type())
  {
    case Core::LinAlg::MatrixType::block_condition:
    case Core::LinAlg::MatrixType::block_condition_dof:
    {
      structurescatradomain->complete();
      break;
    }

    case Core::LinAlg::MatrixType::sparse:
    {
      structurescatradomain->complete(*full_map_scatra_, *full_map_structure());
      break;
    }

    default:
    {
      FOUR_C_THROW("Invalid matrix type associated with scalar transport field!");
      break;
    }
  }
}

FOUR_C_NAMESPACE_CLOSE

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_ssti_monolithic_evaluate_OffDiag.hpp"

#include "4C_adapter_scatra_base_algorithm.hpp"
#include "4C_adapter_str_ssiwrapper.hpp"
#include "4C_coupling_adapter.hpp"
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
SSTI::ThermoStructureOffDiagCoupling::ThermoStructureOffDiagCoupling(
    std::shared_ptr<const Core::LinAlg::MultiMapExtractor> blockmapstructure,
    std::shared_ptr<const Core::LinAlg::MultiMapExtractor> blockmapthermo,
    std::shared_ptr<const Core::LinAlg::Map> full_map_structure,
    std::shared_ptr<const Core::LinAlg::Map> full_map_thermo,
    std::shared_ptr<const SSI::Utils::SSIMeshTying> ssti_structure_meshtying,
    std::shared_ptr<const ScaTra::MeshtyingStrategyS2I> meshtying_strategy_thermo,
    std::shared_ptr<Adapter::SSIStructureWrapper> structure,
    std::shared_ptr<Adapter::ScaTraBaseAlgorithm> thermo)
    : blockmapstructure_(std::move(blockmapstructure)),
      blockmapthermo_(std::move(blockmapthermo)),
      full_map_structure_(std::move(full_map_structure)),
      full_map_thermo_(std::move(full_map_thermo)),
      meshtying_strategy_thermo_(std::move(meshtying_strategy_thermo)),
      ssti_structure_meshtying_(std::move(ssti_structure_meshtying)),
      structure_(std::move(structure)),
      thermo_(std::move(thermo))
{
}

/*-----------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------*/
void SSTI::ThermoStructureOffDiagCoupling::evaluate_off_diag_block_thermo_structure_domain(
    std::shared_ptr<Core::LinAlg::SparseOperator> thermostructuredomain)
{
  // initialize thermo-structure matrix block
  thermostructuredomain->zero();

  Teuchos::ParameterList eleparams;

  Core::Utils::add_enum_class_to_parameter_list<ScaTra::Action>(
      "action", ScaTra::Action::calc_scatra_mono_odblock_mesh, eleparams);

  // remove state vectors from thermo discretization
  thermo_->scatra_field()->discretization()->clear_state();

  // add state vectors to thermo discretization
  thermo_->scatra_field()->add_time_integration_specific_vectors();

  // create strategy for assembly of thermo-structure matrix block
  Core::FE::AssembleStrategy strategyscatrastructure(
      0,  // row assembly based on number of dofset associated with thermo dofs on thermo
          // discretization
      1,  // column assembly based on number of dofset associated with structural dofs on thermo
          // discretization
      thermostructuredomain,  // thermo-structure matrix block
      nullptr, nullptr, nullptr, nullptr);

  thermo_->scatra_field()->discretization()->evaluate(eleparams, strategyscatrastructure);

  thermo_->scatra_field()->discretization()->clear_state();
}
/*-----------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------*/
void SSTI::ThermoStructureOffDiagCoupling::evaluate_off_diag_block_thermo_structure_interface(
    Core::LinAlg::SparseOperator& thermostructureinterface)
{
  thermostructureinterface.zero();

  // slave and master matrix for evaluation of conditions
  std::shared_ptr<Core::LinAlg::SparseOperator> slavematrix(nullptr);
  std::shared_ptr<Core::LinAlg::SparseOperator> mastermatrix(nullptr);
  switch (thermo_->scatra_field()->matrix_type())
  {
    case Core::LinAlg::MatrixType::block_condition:
    {
      slavematrix = std::make_shared<
          Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(

          *blockmapstructure_, meshtying_strategy_thermo_->block_maps_slave(), 81, false, true);
      mastermatrix = std::make_shared<
          Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(

          *blockmapstructure_, meshtying_strategy_thermo_->block_maps_master(), 81, false, true);
      break;
    }
    case Core::LinAlg::MatrixType::sparse:
    {
      slavematrix = std::make_shared<Core::LinAlg::SparseMatrix>(
          *meshtying_strategy_thermo_->coupling_adapter()->slave_dof_map(), 27, false, true);
      mastermatrix = std::make_shared<Core::LinAlg::SparseMatrix>(
          *meshtying_strategy_thermo_->coupling_adapter()->master_dof_map(), 27, false, true);
      break;
    }
    default:
    {
      FOUR_C_THROW("Invalid matrix type associated with scalar transport field!");
      break;
    }
  }

  evaluate_thermo_structure_interface_slave_side(slavematrix);

  copy_slave_to_master_thermo_structure_interface(slavematrix, mastermatrix);

  thermostructureinterface.add(*slavematrix, false, 1.0, 1.0);
  thermostructureinterface.add(*mastermatrix, false, 1.0, 1.0);

  // finalize thermo-structure matrix block
  switch (thermo_->scatra_field()->matrix_type())
  {
    case Core::LinAlg::MatrixType::block_condition:
    {
      thermostructureinterface.complete();
      break;
    }

    case Core::LinAlg::MatrixType::sparse:
    {
      thermostructureinterface.complete(*full_map_structure_, *full_map_thermo_);
      break;
    }

    default:
    {
      FOUR_C_THROW("Invalid matrix type associated with scalar transport field!");
      break;
    }
  }

  thermo_->scatra_field()->discretization()->clear_state();
}

/*-----------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------*/
void SSTI::ThermoStructureOffDiagCoupling::evaluate_off_diag_block_structure_thermo_domain(
    std::shared_ptr<Core::LinAlg::SparseOperator> structurethermodomain)
{
  structurethermodomain->zero();

  Teuchos::ParameterList eleparams;

  eleparams.set("action", "calc_struct_stiffscalar");

  eleparams.set<int>("differentiationtype", static_cast<int>(Solid::DifferentiationType::temp));

  eleparams.set<double>("total time", structure_->time());

  structure_->discretization()->clear_state();

  structure_->discretization()->set_state("displacement", *structure_->dispnp());

  // create strategy for assembly of structure-thermo matrix block
  Core::FE::AssembleStrategy strategystructurescatra(
      0,  // row assembly based on number of dofset associated with structure dofs on structural
          // discretization
      2,  // column assembly based on number of dofset associated with thermo dofs on structural
          // discretization
      structurethermodomain,  // structure-thermo matrix block
      nullptr, nullptr, nullptr, nullptr);

  structure_->discretization()->evaluate(eleparams, strategystructurescatra);

  // need to scale structurethermoblock_ with 'timefac' to getcorrect implementation
  structurethermodomain->scale(1.0 - structure_->tim_int_param());

  structure_->discretization()->clear_state();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::ThermoStructureOffDiagCoupling::copy_slave_to_master_thermo_structure_interface(
    std::shared_ptr<const Core::LinAlg::SparseOperator> slavematrix,
    std::shared_ptr<Core::LinAlg::SparseOperator>& mastermatrix)
{
  mastermatrix->zero();

  switch (thermo_->scatra_field()->matrix_type())
  {
    case Core::LinAlg::MatrixType::block_condition:
    {
      const int numberthermoblocks = thermo_->scatra_field()->dof_block_maps()->num_maps();

      auto blockslavematrix =
          std::dynamic_pointer_cast<const Core::LinAlg::BlockSparseMatrixBase>(slavematrix);
      auto blockmastermatrix =
          std::dynamic_pointer_cast<Core::LinAlg::BlockSparseMatrixBase>(mastermatrix);

      // initialize auxiliary system matrix for linearizations of master-side scatra fluxes w.r.t.
      // master-side structural dofs
      Core::LinAlg::SparseMatrix mastermatrixsparse(
          *meshtying_strategy_thermo_->coupling_adapter()->master_dof_map(), 27, false, true);

      // derive linearizations of master-side scatra fluxes w.r.t. master-side structural dofs and
      // assemble into auxiliary system matrix
      for (int iblock = 0; iblock < numberthermoblocks; ++iblock)
      {
        for (const auto& meshtying : ssti_structure_meshtying_->mesh_tying_handlers())
        {
          auto slave_dof_map = meshtying->slave_master_coupling()->slave_dof_map();
          auto slave_side_converter_struct = meshtying->slave_side_converter();

          auto slave_side_converter_thermo = Coupling::Adapter::CouplingSlaveConverter(
              *meshtying_strategy_thermo_->coupling_adapter());

          Coupling::Adapter::MatrixLogicalSplitAndTransform()(blockslavematrix->matrix(iblock, 0),
              *meshtying_strategy_thermo_->coupling_adapter()->slave_dof_map(), *slave_dof_map,
              -1.0, &slave_side_converter_thermo, &(*slave_side_converter_struct),
              mastermatrixsparse, true, true);
        }
      }

      // finalize auxiliary system matrix
      mastermatrixsparse.complete(*full_map_structure_, *full_map_thermo_);

      // split sparse matrix to block matrix
      auto mastermatrix_split =
          Core::LinAlg::split_matrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
              mastermatrixsparse, *blockmapstructure_, *blockmapthermo_);
      mastermatrix_split->complete();
      blockmastermatrix->add(*mastermatrix_split, false, 1.0, 1.0);

      mastermatrix->complete();

      break;
    }
    case Core::LinAlg::MatrixType::sparse:
    {
      auto sparseslavematrix =
          std::dynamic_pointer_cast<const Core::LinAlg::SparseMatrix>(slavematrix);
      auto sparsemastermatrix = std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(mastermatrix);

      // derive linearizations of master-side scatra fluxes w.r.t. master-side structural dofs and
      // assemble into auxiliary system matrix
      for (const auto& meshtying : ssti_structure_meshtying_->mesh_tying_handlers())
      {
        auto slave_dof_map = meshtying->slave_master_coupling()->slave_dof_map();
        auto slave_side_converter_struct = meshtying->slave_side_converter();
        auto slave_side_converter_thermo = Coupling::Adapter::CouplingSlaveConverter(
            *meshtying_strategy_thermo_->coupling_adapter());

        Coupling::Adapter::MatrixLogicalSplitAndTransform()(*sparseslavematrix,
            *meshtying_strategy_thermo_->coupling_adapter()->slave_dof_map(), *slave_dof_map, -1.0,
            &slave_side_converter_thermo, &(*slave_side_converter_struct), *sparsemastermatrix,
            true, true);
      }

      mastermatrix->complete(
          *full_map_structure_, *meshtying_strategy_thermo_->coupling_adapter()->master_dof_map());
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
void SSTI::ThermoStructureOffDiagCoupling::evaluate_thermo_structure_interface_slave_side(
    std::shared_ptr<Core::LinAlg::SparseOperator> slavematrix)
{
  Teuchos::ParameterList condparams;

  Core::Utils::add_enum_class_to_parameter_list<ScaTra::BoundaryAction>(
      "action", ScaTra::BoundaryAction::calc_s2icoupling_od, condparams);

  Core::Utils::add_enum_class_to_parameter_list<ScaTra::DifferentiationType>(
      "differentiationtype", ScaTra::DifferentiationType::disp, condparams);

  thermo_->scatra_field()->discretization()->clear_state();

  thermo_->scatra_field()->add_time_integration_specific_vectors();

  std::shared_ptr<Core::LinAlg::SparseOperator> evaluate_matrix;
  if (thermo_->scatra_field()->matrix_type() == Core::LinAlg::MatrixType::sparse)
  {
    evaluate_matrix = std::make_shared<Core::LinAlg::SparseMatrix>(
        *meshtying_strategy_thermo_->coupling_adapter()->slave_dof_map(), 27, false, true);
  }
  else
  {
    evaluate_matrix =
        std::make_shared<Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(
            *blockmapstructure_, meshtying_strategy_thermo_->block_maps_slave(), 81, false, true);
  }

  // create strategy for assembly of auxiliary system matrix
  Core::FE::AssembleStrategy strategyscatrastructures2i(
      0,  // row assembly based on number of dofset associated with thermo dofs on
          // thermo discretization
      1,  // column assembly based on number of dofset associated with structural dofs on
          // thermo discretization
      evaluate_matrix, nullptr, nullptr, nullptr, nullptr);

  // evaluate interface coupling
  for (auto kinetics_slave_cond :
      meshtying_strategy_thermo_->kinetics_conditions_meshtying_slave_side())
  {
    if (kinetics_slave_cond.second->parameters().get<Inpar::S2I::KineticModels>("KINETIC_MODEL") !=
        static_cast<int>(Inpar::S2I::kinetics_nointerfaceflux))
    {
      // collect condition specific data and store to scatra boundary parameter class
      meshtying_strategy_thermo_->set_condition_specific_scatra_parameters(
          *kinetics_slave_cond.second);
      // evaluate the condition
      thermo_->scatra_field()->discretization()->evaluate_condition(
          condparams, strategyscatrastructures2i, "S2IKinetics", kinetics_slave_cond.first);
    }
  }

  // finalize thermo-structure matrix block
  switch (thermo_->scatra_field()->matrix_type())
  {
    case Core::LinAlg::MatrixType::block_condition:
    {
      evaluate_matrix->complete();

      auto evaluate_matrix_block =
          Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(evaluate_matrix);
      auto slavematrix_block =
          Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(slavematrix);

      // "slave side" from thermo and from structure do not need to be the same nodes.
      // Linearization is evaluated on scatra slave side node --> Transformation needed
      for (const auto& meshtying : ssti_structure_meshtying_->mesh_tying_handlers())
      {
        auto slave_slave_transformation = meshtying->slave_slave_transformation();
        // converter between old slave dofs from input and actual slave dofs from current mesh tying
        // adapter
        auto slave_slave_converter =
            Coupling::Adapter::CouplingSlaveConverter(*slave_slave_transformation);

        // old slave dofs from input
        auto slave_map = slave_slave_transformation->slave_dof_map();

        for (int iblock = 0; iblock < thermo_->scatra_field()->dof_block_maps()->num_maps();
            ++iblock)
        {
          auto evaluate_iblock = evaluate_matrix_block->matrix(iblock, 0);
          auto slave_iblock = slavematrix_block->matrix(iblock, 0);

          auto scatra_slave_block_mapi =
              Core::LinAlg::intersect_map(*thermo_->scatra_field()->dof_block_maps()->map(iblock),
                  *meshtying_strategy_thermo_->coupling_adapter()->slave_dof_map());

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
      auto evaluate_matrix_sparse =
          Core::LinAlg::cast_to_const_sparse_matrix_and_check_success(evaluate_matrix);
      auto slavematrix_sparse = Core::LinAlg::cast_to_sparse_matrix_and_check_success(slavematrix);

      evaluate_matrix->complete(
          *full_map_structure_, *meshtying_strategy_thermo_->coupling_adapter()->slave_dof_map());

      // "slave side" from thermo and from structure do not need to be the same nodes.
      // Linearization is evaluated on scatra slave side node --> Transformation needed
      for (const auto& meshtying : ssti_structure_meshtying_->mesh_tying_handlers())
      {
        auto slave_slave_transformation = meshtying->slave_slave_transformation();
        // converter between old slave dofs from input and actual slave dofs from current mesh tying
        // adapter
        auto slave_slave_converter =
            Coupling::Adapter::CouplingSlaveConverter(*slave_slave_transformation);

        // old slave dofs from input
        auto slave_map = slave_slave_transformation->slave_dof_map();

        Coupling::Adapter::MatrixLogicalSplitAndTransform()(*evaluate_matrix_sparse,
            *meshtying_strategy_thermo_->coupling_adapter()->slave_dof_map(), *slave_map, 1.0,
            nullptr, &slave_slave_converter, *slavematrix_sparse, true, true);
      }
      slavematrix->complete(
          *full_map_structure_, *meshtying_strategy_thermo_->coupling_adapter()->slave_dof_map());

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

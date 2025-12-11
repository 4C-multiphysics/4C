// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_ssi_contact_strategy.hpp"

#include "4C_contact_nitsche_strategy_ssi.hpp"
#include "4C_linalg_blocksparsematrix.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_ssi_utils.hpp"

FOUR_C_NAMESPACE_OPEN

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
SSI::ContactStrategyBase::ContactStrategyBase(
    std::shared_ptr<CONTACT::NitscheStrategySsi> contact_nitsche_strategy,
    std::shared_ptr<const SSI::Utils::SSIMaps> ssi_maps)
    : contact_strategy_nitsche_(std::move(contact_nitsche_strategy)), ssi_maps_(std::move(ssi_maps))
{
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
SSI::ContactStrategySparse::ContactStrategySparse(
    std::shared_ptr<CONTACT::NitscheStrategySsi> contact_nitsche_strategy,
    std::shared_ptr<const SSI::Utils::SSIMaps> ssi_maps)
    : ContactStrategyBase(contact_nitsche_strategy, ssi_maps)
{
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
SSI::ContactStrategyBlock::ContactStrategyBlock(
    std::shared_ptr<CONTACT::NitscheStrategySsi> contact_nitsche_strategy,
    std::shared_ptr<const SSI::Utils::SSIMaps> ssi_maps)
    : ContactStrategyBase(contact_nitsche_strategy, ssi_maps)
{
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
void SSI::ContactStrategyBase::apply_contact_to_scatra_residual(
    std::shared_ptr<Core::LinAlg::Vector<double>> scatra_residual)
{
  scatra_residual->update(
      1.0, *nitsche_strategy_ssi()->get_rhs_block_ptr(CONTACT::VecBlockType::scatra), 1.0);
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
void SSI::ContactStrategySparse::apply_contact_to_scatra_scatra(
    std::shared_ptr<Core::LinAlg::SparseOperator> scatra_scatra_matrix)
{
  auto scatra_scatra_matrix_sparse =
      Core::LinAlg::cast_to_sparse_matrix_and_check_success(scatra_scatra_matrix);

  const auto& scatra_scatra_sparsematrix =
      nitsche_strategy_ssi()->get_matrix_block_ptr(CONTACT::MatBlockType::scatra_scatra);

  Core::LinAlg::matrix_add(
      *scatra_scatra_sparsematrix, false, 1.0, *scatra_scatra_matrix_sparse, 1.0);
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
void SSI::ContactStrategyBlock::apply_contact_to_scatra_scatra(
    std::shared_ptr<Core::LinAlg::SparseOperator> scatra_scatra_matrix)
{
  auto scatra_scatra_matrix_block =
      Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(scatra_scatra_matrix);

  // get scatra-scatra block matrix and complete split matrix
  const auto& scatra_scatra_blockmatrix =

      Core::LinAlg::split_matrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
          *nitsche_strategy_ssi()->get_matrix_block_ptr(CONTACT::MatBlockType::scatra_scatra),
          *ssi_maps()->block_map_scatra(), *ssi_maps()->block_map_scatra());
  scatra_scatra_blockmatrix->complete();

  scatra_scatra_matrix_block->add(*scatra_scatra_blockmatrix, false, 1.0, 1.0);
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
void SSI::ContactStrategySparse::apply_contact_to_scatra_structure(
    std::shared_ptr<Core::LinAlg::SparseOperator> scatra_structure_matrix)
{
  auto scatra_structure_matrix_sparse =
      Core::LinAlg::cast_to_sparse_matrix_and_check_success(scatra_structure_matrix);
  scatra_structure_matrix_sparse->un_complete();

  const auto& scatra_struct_matrix =
      nitsche_strategy_ssi()->get_matrix_block_ptr(CONTACT::MatBlockType::scatra_displ);

  Core::LinAlg::matrix_add(*scatra_struct_matrix, false, 1.0, *scatra_structure_matrix_sparse, 1.0);
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
void SSI::ContactStrategyBlock::apply_contact_to_scatra_structure(
    std::shared_ptr<Core::LinAlg::SparseOperator> scatra_structure_matrix)
{
  auto scatra_structure_matrix_block =
      Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(scatra_structure_matrix);

  // get scatra-structure block matrix and complete split matrix
  const auto& scatra_struct_blockmatrix =

      Core::LinAlg::split_matrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
          *nitsche_strategy_ssi()->get_matrix_block_ptr(CONTACT::MatBlockType::scatra_displ),
          *ssi_maps()->block_map_structure(), *ssi_maps()->block_map_scatra());
  scatra_struct_blockmatrix->complete();

  scatra_structure_matrix_block->add(*scatra_struct_blockmatrix, false, 1.0, 1.0);
}


/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
void SSI::ContactStrategySparse::apply_contact_to_structure_scatra(
    std::shared_ptr<Core::LinAlg::SparseOperator> structure_scatra_matrix)
{
  auto structure_scatra_matrix_sparse =
      Core::LinAlg::cast_to_sparse_matrix_and_check_success(structure_scatra_matrix);
  structure_scatra_matrix_sparse->un_complete();

  const auto& struct_scatra_matrix =
      nitsche_strategy_ssi()->get_matrix_block_ptr(CONTACT::MatBlockType::displ_scatra);

  Core::LinAlg::matrix_add(*struct_scatra_matrix, false, 1.0, *structure_scatra_matrix_sparse, 1.0);
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
void SSI::ContactStrategyBlock::apply_contact_to_structure_scatra(
    std::shared_ptr<Core::LinAlg::SparseOperator> structure_scatra_matrix)
{
  auto structure_scatra_matrix_block =
      Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(structure_scatra_matrix);

  // get structure-scatra block matrix and complete split matrix
  const auto& struct_scatra_blockmatrix =

      Core::LinAlg::split_matrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
          *nitsche_strategy_ssi()->get_matrix_block_ptr(CONTACT::MatBlockType::displ_scatra),
          *ssi_maps()->block_map_scatra(), *ssi_maps()->block_map_structure());
  struct_scatra_blockmatrix->complete();

  structure_scatra_matrix_block->add(*struct_scatra_blockmatrix, false, 1.0, 1.0);
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
std::shared_ptr<SSI::ContactStrategyBase> SSI::build_contact_strategy(
    std::shared_ptr<CONTACT::NitscheStrategySsi> contact_nitsche_strategy,
    std::shared_ptr<const SSI::Utils::SSIMaps> ssi_maps, Core::LinAlg::MatrixType matrixtype_scatra)
{
  std::shared_ptr<SSI::ContactStrategyBase> contact_strategy(nullptr);

  switch (matrixtype_scatra)
  {
    case Core::LinAlg::MatrixType::block_condition:
    case Core::LinAlg::MatrixType::block_condition_dof:
    {
      contact_strategy =
          std::make_shared<SSI::ContactStrategyBlock>(contact_nitsche_strategy, ssi_maps);
      break;
    }
    case Core::LinAlg::MatrixType::sparse:
    {
      contact_strategy =
          std::make_shared<SSI::ContactStrategySparse>(contact_nitsche_strategy, ssi_maps);
      break;
    }

    default:
    {
      FOUR_C_THROW("unknown matrix type of ScaTra field");
      break;
    }
  }

  return contact_strategy;
}

FOUR_C_NAMESPACE_CLOSE

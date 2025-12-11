// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_ssi_monolithic_assemble_strategy.hpp"

#include "4C_adapter_str_ssiwrapper.hpp"
#include "4C_contact_nitsche_strategy_ssi.hpp"
#include "4C_coupling_adapter_converter.hpp"
#include "4C_fem_condition_locsys.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_scatra_timint_meshtying_strategy_s2i.hpp"
#include "4C_ssi_monolithic.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
SSI::AssembleStrategyBase::AssembleStrategyBase(
    std::shared_ptr<const SSI::Utils::SSIMaps> ssi_maps, const bool is_scatra_manifold)
    : is_scatra_manifold_(is_scatra_manifold), ssi_maps_(std::move(ssi_maps))
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
SSI::AssembleStrategyBlock::AssembleStrategyBlock(
    std::shared_ptr<const SSI::Utils::SSIMaps> ssi_maps, const bool is_scatra_manifold_value)
    : AssembleStrategyBase(ssi_maps, is_scatra_manifold_value),
      block_position_scatra_(ssi_maps->get_block_positions(SSI::Subproblem::scalar_transport)),
      position_structure_(ssi_maps->get_block_positions(SSI::Subproblem::structure).at(0))
{
  if (is_scatra_manifold())
    block_position_scatra_manifold_ = ssi_maps->get_block_positions(SSI::Subproblem::manifold);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
SSI::AssembleStrategyBlockBlock::AssembleStrategyBlockBlock(
    std::shared_ptr<const SSI::Utils::SSIMaps> ssi_maps, const bool is_scatra_manifold)
    : AssembleStrategyBlock(ssi_maps, is_scatra_manifold)
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
SSI::AssembleStrategyBlockSparse::AssembleStrategyBlockSparse(
    std::shared_ptr<const SSI::Utils::SSIMaps> ssi_maps, const bool is_scatra_manifold)
    : AssembleStrategyBlock(ssi_maps, is_scatra_manifold)
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
SSI::AssembleStrategySparse::AssembleStrategySparse(
    std::shared_ptr<const SSI::Utils::SSIMaps> ssi_maps, const bool is_scatra_manifold)
    : AssembleStrategyBase(ssi_maps, is_scatra_manifold)
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::AssembleStrategyBlockBlock::assemble_scatra_scatra(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
    std::shared_ptr<const Core::LinAlg::SparseOperator> scatra_scatra_matrix)
{
  auto systemmatrix_block =
      Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(systemmatrix);
  auto scatra_scatra_matrix_block =
      Core::LinAlg::cast_to_const_block_sparse_matrix_base_and_check_success(scatra_scatra_matrix);
  systemmatrix_block->un_complete();

  // assemble blocks of scalar transport system matrix into global system matrix
  for (int iblock = 0; iblock < static_cast<int>(block_position_scatra().size()); ++iblock)
  {
    for (int jblock = 0; jblock < static_cast<int>(block_position_scatra().size()); ++jblock)
    {
      auto& systemmatrix_block_iscatra_jscatra = systemmatrix_block->matrix(
          block_position_scatra().at(iblock), block_position_scatra().at(jblock));

      Core::LinAlg::matrix_add(scatra_scatra_matrix_block->matrix(iblock, jblock), false, 1.0,
          systemmatrix_block_iscatra_jscatra, 1.0);
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::AssembleStrategyBlockSparse::assemble_scatra_scatra(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
    std::shared_ptr<const Core::LinAlg::SparseOperator> scatra_scatra_matrix)
{
  auto systemmatrix_block =
      Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(systemmatrix);
  auto scatra_scatra_matrix_sparse =
      Core::LinAlg::cast_to_const_sparse_matrix_and_check_success(scatra_scatra_matrix);

  auto& systemmatrix_block_scatra_scatra =
      systemmatrix_block->matrix(block_position_scatra().at(0), block_position_scatra().at(0));

  Core::LinAlg::matrix_add(
      *scatra_scatra_matrix_sparse, false, 1.0, systemmatrix_block_scatra_scatra, 1.0);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::AssembleStrategySparse::assemble_scatra_scatra(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
    std::shared_ptr<const Core::LinAlg::SparseOperator> scatra_scatra_matrix)
{
  auto systemmatrix_sparse = Core::LinAlg::cast_to_sparse_matrix_and_check_success(systemmatrix);
  auto scatra_scatra_matrix_sparse =
      Core::LinAlg::cast_to_const_sparse_matrix_and_check_success(scatra_scatra_matrix);

  // add scalar transport system matrix to global system matrix
  Core::LinAlg::matrix_add(*scatra_scatra_matrix_sparse, false, 1.0, *systemmatrix_sparse, 1.0);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::AssembleStrategyBlockBlock::assemble_structure_structure(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
    std::shared_ptr<const Core::LinAlg::SparseMatrix> structure_structure_matrix)
{
  auto systemmatrix_block =
      Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(systemmatrix);
  auto& systemmatrix_block_struct_struct =
      systemmatrix_block->matrix(position_structure(), position_structure());

  Core::LinAlg::matrix_add(
      *structure_structure_matrix, false, 1.0, systemmatrix_block_struct_struct, 1.0);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::AssembleStrategyBlockSparse::assemble_structure_structure(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
    std::shared_ptr<const Core::LinAlg::SparseMatrix> structure_structure_matrix)
{
  auto systemmatrix_block =
      Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(systemmatrix);
  auto& systemmatrix_block_struct_struct =
      systemmatrix_block->matrix(position_structure(), position_structure());

  Core::LinAlg::matrix_add(
      *structure_structure_matrix, false, 1.0, systemmatrix_block_struct_struct, 1.0);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::AssembleStrategySparse::assemble_structure_structure(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
    std::shared_ptr<const Core::LinAlg::SparseMatrix> structure_structure_matrix)
{
  auto systemmatrix_sparse = Core::LinAlg::cast_to_sparse_matrix_and_check_success(systemmatrix);
  Core::LinAlg::matrix_add(*structure_structure_matrix, false, 1.0, *systemmatrix_sparse, 1.0);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::AssembleStrategyBlockBlock::assemble_scatra_structure(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
    std::shared_ptr<const Core::LinAlg::SparseOperator> scatra_structure_matrix)
{
  auto systemmatrix_block =
      Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(systemmatrix);
  auto scatra_structure_matrix_block =
      Core::LinAlg::cast_to_const_block_sparse_matrix_base_and_check_success(
          scatra_structure_matrix);

  // assemble blocks of scalar transport system matrix into global system matrix
  for (int iblock = 0; iblock < static_cast<int>(block_position_scatra().size()); ++iblock)
  {
    auto& systemmatrix_block_iscatra_struct =
        systemmatrix_block->matrix(block_position_scatra().at(iblock), position_structure());

    Core::LinAlg::matrix_add(scatra_structure_matrix_block->matrix(iblock, 0), false, 1.0,
        systemmatrix_block_iscatra_struct, 1.0);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::AssembleStrategyBlockSparse::assemble_scatra_structure(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
    std::shared_ptr<const Core::LinAlg::SparseOperator> scatra_structure_matrix)
{
  auto systemmatrix_block =
      Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(systemmatrix);
  auto scatra_structure_matrix_sparse =
      Core::LinAlg::cast_to_const_sparse_matrix_and_check_success(scatra_structure_matrix);

  auto& systemmatrix_block_scatra_struct =
      systemmatrix_block->matrix(block_position_scatra().at(0), position_structure());
  systemmatrix_block_scatra_struct.un_complete();

  Core::LinAlg::matrix_add(
      *scatra_structure_matrix_sparse, false, 1.0, systemmatrix_block_scatra_struct, 1.0);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::AssembleStrategySparse::assemble_scatra_structure(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
    std::shared_ptr<const Core::LinAlg::SparseOperator> scatra_structure_matrix)
{
  auto systemmatrix_sparse = Core::LinAlg::cast_to_sparse_matrix_and_check_success(systemmatrix);
  auto scatra_structure_matrix_sparse =
      Core::LinAlg::cast_to_const_sparse_matrix_and_check_success(scatra_structure_matrix);

  Core::LinAlg::matrix_add(*scatra_structure_matrix_sparse, false, 1.0, *systemmatrix_sparse, 1.0);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::AssembleStrategyBlockBlock::assemble_scatra_scatramanifold(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
    std::shared_ptr<const Core::LinAlg::SparseOperator> scatra_scatramanifold_matrix)
{
  auto systemmatrix_block =
      Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(systemmatrix);
  auto scatra_scatramanifold_matrix_block =
      Core::LinAlg::cast_to_const_block_sparse_matrix_base_and_check_success(
          scatra_scatramanifold_matrix);

  for (int iblock = 0; iblock < static_cast<int>(block_position_scatra().size()); ++iblock)
  {
    for (int jblock = 0; jblock < static_cast<int>(block_position_scatra_manifold().size());
        ++jblock)
    {
      Core::LinAlg::matrix_add(scatra_scatramanifold_matrix_block->matrix(iblock, jblock), false,
          1.0,
          systemmatrix_block->matrix(
              block_position_scatra().at(iblock), block_position_scatra_manifold().at(jblock)),
          1.0);
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::AssembleStrategyBlockSparse::assemble_scatra_scatramanifold(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
    std::shared_ptr<const Core::LinAlg::SparseOperator> scatra_scatramanifold_matrix)
{
  auto systemmatrix_block =
      Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(systemmatrix);
  auto scatra_scatramanifold_matrix_sparse =
      Core::LinAlg::cast_to_const_sparse_matrix_and_check_success(scatra_scatramanifold_matrix);

  Core::LinAlg::matrix_add(*scatra_scatramanifold_matrix_sparse, false, 1.0,
      systemmatrix_block->matrix(
          block_position_scatra().at(0), block_position_scatra_manifold().at(0)),
      1.0);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::AssembleStrategySparse::assemble_scatra_scatramanifold(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
    std::shared_ptr<const Core::LinAlg::SparseOperator> scatra_scatramanifold_matrix)
{
  auto systemmatrix_sparse = Core::LinAlg::cast_to_sparse_matrix_and_check_success(systemmatrix);
  auto scatra_scatramanifold_matrix_sparse =
      Core::LinAlg::cast_to_const_sparse_matrix_and_check_success(scatra_scatramanifold_matrix);

  Core::LinAlg::matrix_add(
      *scatra_scatramanifold_matrix_sparse, false, 1.0, *systemmatrix_sparse, 1.0);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::AssembleStrategyBlockBlock::assemble_structure_scatra(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
    std::shared_ptr<const Core::LinAlg::SparseOperator> structure_scatra_matrix)
{
  auto systemmatrix_block =
      Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(systemmatrix);
  auto structure_scatra_matrix_block =
      Core::LinAlg::cast_to_const_block_sparse_matrix_base_and_check_success(
          structure_scatra_matrix);

  // assemble blocks of scalar transport system matrix into global system matrix
  for (int iblock = 0; iblock < static_cast<int>(block_position_scatra().size()); ++iblock)
  {
    auto& systemmatrix_block_struct_iscatra =
        systemmatrix_block->matrix(position_structure(), block_position_scatra().at(iblock));
    Core::LinAlg::matrix_add(structure_scatra_matrix_block->matrix(0, iblock), false, 1.0,
        systemmatrix_block_struct_iscatra, 1.0);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::AssembleStrategyBlockSparse::assemble_structure_scatra(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
    std::shared_ptr<const Core::LinAlg::SparseOperator> structure_scatra_matrix)
{
  auto systemmatrix_block =
      Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(systemmatrix);
  auto structure_scatra_matrix_sparse =
      Core::LinAlg::cast_to_const_sparse_matrix_and_check_success(structure_scatra_matrix);

  auto& systemmatrix_block_struct_scatra =
      systemmatrix_block->matrix(position_structure(), block_position_scatra().at(0));
  systemmatrix_block_struct_scatra.un_complete();
  Core::LinAlg::matrix_add(
      *structure_scatra_matrix_sparse, false, 1.0, systemmatrix_block_struct_scatra, 1.0);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::AssembleStrategySparse::assemble_structure_scatra(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
    std::shared_ptr<const Core::LinAlg::SparseOperator> structure_scatra_matrix)
{
  auto systemmatrix_sparse = Core::LinAlg::cast_to_sparse_matrix_and_check_success(systemmatrix);
  auto structure_scatra_matrix_sparse =
      Core::LinAlg::cast_to_const_sparse_matrix_and_check_success(structure_scatra_matrix);

  Core::LinAlg::matrix_add(*structure_scatra_matrix_sparse, false, 1.0, *systemmatrix_sparse, 1.0);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::AssembleStrategyBlockBlock::assemble_scatramanifold_scatra(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
    std::shared_ptr<const Core::LinAlg::SparseOperator> scatramanifold_scatra_matrix)
{
  auto systemmatrix_block =
      Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(systemmatrix);
  auto scatramanifold_scatra_matrix_block =
      Core::LinAlg::cast_to_const_block_sparse_matrix_base_and_check_success(
          scatramanifold_scatra_matrix);

  for (int iblock = 0; iblock < static_cast<int>(block_position_scatra_manifold().size()); ++iblock)
  {
    for (int jblock = 0; jblock < static_cast<int>(block_position_scatra().size()); ++jblock)
    {
      Core::LinAlg::matrix_add(scatramanifold_scatra_matrix_block->matrix(iblock, jblock), false,
          1.0,
          systemmatrix_block->matrix(
              block_position_scatra_manifold().at(iblock), block_position_scatra().at(jblock)),
          1.0);
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::AssembleStrategyBlockSparse::assemble_scatramanifold_scatra(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
    std::shared_ptr<const Core::LinAlg::SparseOperator> scatramanifold_scatra_matrix)
{
  auto systemmatrix_block =
      Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(systemmatrix);
  auto scatramanifold_scatra_matrix_sparse =
      Core::LinAlg::cast_to_const_sparse_matrix_and_check_success(scatramanifold_scatra_matrix);

  Core::LinAlg::matrix_add(*scatramanifold_scatra_matrix_sparse, false, 1.0,
      systemmatrix_block->matrix(
          block_position_scatra_manifold().at(0), block_position_scatra().at(0)),
      1.0);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::AssembleStrategySparse::assemble_scatramanifold_scatra(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
    std::shared_ptr<const Core::LinAlg::SparseOperator> scatramanifold_scatra_matrix)
{
  auto systemmatrix_sparse = Core::LinAlg::cast_to_sparse_matrix_and_check_success(systemmatrix);
  auto scatramanifold_scatra_matrix_sparse =
      Core::LinAlg::cast_to_const_sparse_matrix_and_check_success(scatramanifold_scatra_matrix);

  Core::LinAlg::matrix_add(
      *scatramanifold_scatra_matrix_sparse, false, 1.0, *systemmatrix_sparse, 1.0);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::AssembleStrategyBlockBlock::assemble_scatramanifold_scatramanifold(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
    std::shared_ptr<const Core::LinAlg::SparseOperator> scatramanifold_scatramanifold_matrix)
{
  auto systemmatrix_block =
      Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(systemmatrix);
  auto scatramanifold_scatramanifold_matrix_block =
      Core::LinAlg::cast_to_const_block_sparse_matrix_base_and_check_success(
          scatramanifold_scatramanifold_matrix);

  // assemble blocks of scalar transport system matrix into global system matrix
  for (int iblock = 0; iblock < static_cast<int>(block_position_scatra_manifold().size()); ++iblock)
  {
    for (int jblock = 0; jblock < static_cast<int>(block_position_scatra_manifold().size());
        ++jblock)
    {
      auto& systemmatrix_block_iscatramanifold_jscatramanifold = systemmatrix_block->matrix(
          block_position_scatra_manifold().at(iblock), block_position_scatra_manifold().at(jblock));

      Core::LinAlg::matrix_add(scatramanifold_scatramanifold_matrix_block->matrix(iblock, jblock),
          false, 1.0, systemmatrix_block_iscatramanifold_jscatramanifold, 1.0);
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::AssembleStrategyBlockSparse::assemble_scatramanifold_scatramanifold(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
    std::shared_ptr<const Core::LinAlg::SparseOperator> scatramanifold_scatramanifold_matrix)
{
  auto systemmatrix_block =
      Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(systemmatrix);
  auto scatramanifold_scatramanifold_matrix_sparse =
      Core::LinAlg::cast_to_const_sparse_matrix_and_check_success(
          scatramanifold_scatramanifold_matrix);

  auto& systemmatrix_block_scatramanifold_scatramanifold = systemmatrix_block->matrix(
      block_position_scatra_manifold().at(0), block_position_scatra_manifold().at(0));

  Core::LinAlg::matrix_add(*scatramanifold_scatramanifold_matrix_sparse, false, 1.0,
      systemmatrix_block_scatramanifold_scatramanifold, 1.0);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::AssembleStrategySparse::assemble_scatramanifold_scatramanifold(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
    std::shared_ptr<const Core::LinAlg::SparseOperator> scatramanifold_scatramanifold_matrix)
{
  auto systemmatrix_sparse = Core::LinAlg::cast_to_sparse_matrix_and_check_success(systemmatrix);
  auto scatramanifold_scatramanifold_matrix_sparse =
      Core::LinAlg::cast_to_const_sparse_matrix_and_check_success(
          scatramanifold_scatramanifold_matrix);

  Core::LinAlg::matrix_add(
      *scatramanifold_scatramanifold_matrix_sparse, false, 1.0, *systemmatrix_sparse, 1.0);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::AssembleStrategyBlockBlock::assemble_scatramanifold_structure(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
    std::shared_ptr<const Core::LinAlg::SparseOperator> scatramanifold_structure_matrix)
{
  auto systemmatrix_block =
      Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(systemmatrix);
  auto scatramanifold_structure_matrix_block =
      Core::LinAlg::cast_to_const_block_sparse_matrix_base_and_check_success(
          scatramanifold_structure_matrix);

  for (int iblock = 0; iblock < static_cast<int>(block_position_scatra_manifold().size()); ++iblock)
  {
    auto& systemmatrix_block_iscatramanifold_struct = systemmatrix_block->matrix(
        block_position_scatra_manifold().at(iblock), position_structure());
    Core::LinAlg::matrix_add(scatramanifold_structure_matrix_block->matrix(iblock, 0), false, 1.0,
        systemmatrix_block_iscatramanifold_struct, 1.0);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::AssembleStrategyBlockSparse::assemble_scatramanifold_structure(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
    std::shared_ptr<const Core::LinAlg::SparseOperator> scatramanifold_structure_matrix)
{
  auto systemmatrix_block =
      Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(systemmatrix);
  auto scatramanifold_structure_matrix_sparse =
      Core::LinAlg::cast_to_const_sparse_matrix_and_check_success(scatramanifold_structure_matrix);

  auto& systemmatrix_block_scatramanifold_struct =
      systemmatrix_block->matrix(block_position_scatra_manifold().at(0), position_structure());
  Core::LinAlg::matrix_add(*scatramanifold_structure_matrix_sparse, false, 1.0,
      systemmatrix_block_scatramanifold_struct, 1.0);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::AssembleStrategySparse::assemble_scatramanifold_structure(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
    std::shared_ptr<const Core::LinAlg::SparseOperator> scatramanifold_structure_matrix)
{
  auto systemmatrix_sparse = Core::LinAlg::cast_to_sparse_matrix_and_check_success(systemmatrix);
  auto scatramanifold_structure_matrix_sparse =
      Core::LinAlg::cast_to_const_sparse_matrix_and_check_success(scatramanifold_structure_matrix);

  Core::LinAlg::matrix_add(
      *scatramanifold_structure_matrix_sparse, false, 1.0, *systemmatrix_sparse, 1.0);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::AssembleStrategyBase::assemble_rhs(std::shared_ptr<Core::LinAlg::Vector<double>> rhs,
    std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_scatra,
    std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_structure,
    const Core::LinAlg::Vector<double>* rhs_manifold)
{
  ssi_maps()->maps_sub_problems()->insert_vector(
      *rhs_scatra, Utils::SSIMaps::get_problem_position(SSI::Subproblem::scalar_transport), *rhs);

  if (is_scatra_manifold())
  {
    FOUR_C_ASSERT(
        rhs_manifold, "Internal error: rhs_manifold must not be null if ScaTra manifold is used");
    ssi_maps()->maps_sub_problems()->insert_vector(
        *rhs_manifold, Utils::SSIMaps::get_problem_position(SSI::Subproblem::manifold), *rhs);
  }

  ssi_maps()->maps_sub_problems()->add_vector(
      *rhs_structure, Utils::SSIMaps::get_problem_position(SSI::Subproblem::structure), *rhs, -1.0);
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
std::shared_ptr<SSI::AssembleStrategyBase> SSI::build_assemble_strategy(
    std::shared_ptr<const SSI::Utils::SSIMaps> ssi_maps, const bool is_scatra_manifold,
    Core::LinAlg::MatrixType matrixtype_ssi, Core::LinAlg::MatrixType matrixtype_scatra)
{
  std::shared_ptr<SSI::AssembleStrategyBase> assemblestrategy = nullptr;

  switch (matrixtype_ssi)
  {
    case Core::LinAlg::MatrixType::block_field:
    {
      switch (matrixtype_scatra)
      {
        case Core::LinAlg::MatrixType::block_condition:
        case Core::LinAlg::MatrixType::block_condition_dof:
        {
          assemblestrategy =
              std::make_shared<SSI::AssembleStrategyBlockBlock>(ssi_maps, is_scatra_manifold);
          break;
        }
        case Core::LinAlg::MatrixType::sparse:
        {
          assemblestrategy =
              std::make_shared<SSI::AssembleStrategyBlockSparse>(ssi_maps, is_scatra_manifold);
          break;
        }

        default:
        {
          FOUR_C_THROW("unknown matrix type of ScaTra field");
          break;
        }
      }
      break;
    }
    case Core::LinAlg::MatrixType::sparse:
    {
      assemblestrategy =
          std::make_shared<SSI::AssembleStrategySparse>(ssi_maps, is_scatra_manifold);
      break;
    }
    default:
    {
      FOUR_C_THROW("unknown matrix type of SSI problem");
      break;
    }
  }

  return assemblestrategy;
}
FOUR_C_NAMESPACE_CLOSE

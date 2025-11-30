// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_ssi_monolithic_meshtying_strategy.hpp"

#include "4C_adapter_str_ssiwrapper.hpp"
#include "4C_coupling_adapter.hpp"
#include "4C_coupling_adapter_converter.hpp"
#include "4C_linalg_blocksparsematrix.hpp"
#include "4C_linalg_map.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_scatra_timint_meshtying_strategy_s2i.hpp"
#include "4C_ssi_monolithic.hpp"
#include "4C_ssi_utils.hpp"

#include <utility>

FOUR_C_NAMESPACE_OPEN

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
SSI::MeshtyingStrategySparse::MeshtyingStrategySparse(
    const bool is_scatra_manifold, const SSI::Utils::SSIMaps& ssi_maps)
    : MeshtyingStrategyBase()
{
  temp_scatra_struct_mat_ =
      SSI::Utils::SSIMatrices::setup_sparse_matrix(*ssi_maps.scatra_dof_row_map());

  if (is_scatra_manifold)
  {
    temp_scatramanifold_struct_mat_ =
        SSI::Utils::SSIMatrices::setup_sparse_matrix(*ssi_maps.scatra_manifold_dof_row_map());
  }

  temp_struct_scatra_mat_ =
      SSI::Utils::SSIMatrices::setup_sparse_matrix(*ssi_maps.structure_dof_row_map());
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
SSI::MeshtyingStrategyBlock::MeshtyingStrategyBlock(
    const bool is_scatra_manifold, const SSI::Utils::SSIMaps& ssi_maps)
    : MeshtyingStrategyBase(),
      block_position_scatra_(ssi_maps.get_block_positions(SSI::Subproblem::scalar_transport)),
      position_structure_(ssi_maps.get_block_positions(SSI::Subproblem::structure).at(0))
{
  temp_scatra_struct_mat_ = SSI::Utils::SSIMatrices::setup_block_matrix(
      *ssi_maps.block_map_scatra(), *ssi_maps.block_map_structure());

  if (is_scatra_manifold)
  {
    temp_scatramanifold_struct_mat_ = SSI::Utils::SSIMatrices::setup_block_matrix(
        *ssi_maps.block_map_scatra_manifold(), *ssi_maps.block_map_structure());
  }

  temp_struct_scatra_mat_ = SSI::Utils::SSIMatrices::setup_block_matrix(
      *ssi_maps.block_map_structure(), *ssi_maps.block_map_scatra());

  if (is_scatra_manifold)
    block_position_scatra_manifold_ = ssi_maps.get_block_positions(SSI::Subproblem::manifold);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::MeshtyingStrategyBase::apply_meshtying_to_structure_matrix(
    Core::LinAlg::SparseMatrix& ssi_structure_matrix,
    const Core::LinAlg::SparseMatrix& structure_matrix,
    const SSI::Utils::SSIMeshTying& ssi_structure_meshtying, const bool do_uncomplete)
{
  /* Transform and assemble the structure matrix into the ssi structure matrix block by block:
   * S_i:  structure interior dofs
   * S_m:  structure master side dofs
   * S_s1: slave dofs
   * S_s2: other slave dofs
   *
   *      | S_i | S_m | S_s1| S_s2|
   *      |-----|-----|-----|-----|
   * S_i  |  a  |  b  |  c  |     |
   * S_m  |  e  |  f  |  g  |     |
   * S_s1 |  i  |  j  |  k  |  l  |
   * S_s2 |     |     |     |     |
   *      -------------------------
   */
  // uncomplete the ssi matrix if necessary
  if (do_uncomplete) ssi_structure_matrix.un_complete();

  auto map_structure_interior = ssi_structure_meshtying.interior_map();
  auto master_dof_map = ssi_structure_meshtying.full_master_side_map();

  // assemble derivatives of interior dofs w.r.t. interior dofs (block a)
  Coupling::Adapter::MatrixLogicalSplitAndTransform()(structure_matrix, *map_structure_interior,
      *map_structure_interior, 1.0, nullptr, nullptr, ssi_structure_matrix, true, true);

  // assemble derivatives of interior dofs w.r.t. master dofs (block b)
  Coupling::Adapter::MatrixLogicalSplitAndTransform()(structure_matrix, *map_structure_interior,
      *master_dof_map, 1.0, nullptr, nullptr, ssi_structure_matrix, true, true);

  // assemble derivatives of master dofs w.r.t. interior dofs (block e)
  Coupling::Adapter::MatrixLogicalSplitAndTransform()(structure_matrix, *master_dof_map,
      *map_structure_interior, 1.0, nullptr, nullptr, ssi_structure_matrix, true, true);

  // assemble derivatives of master dofs w.r.t. master dofs (block f)
  Coupling::Adapter::MatrixLogicalSplitAndTransform()(structure_matrix, *master_dof_map,
      *master_dof_map, 1.0, nullptr, nullptr, ssi_structure_matrix, true, true);

  for (const auto& meshtying : ssi_structure_meshtying.mesh_tying_handlers())
  {
    auto cond_slave_dof_map = meshtying->slave_master_coupling()->slave_dof_map();
    auto converter = meshtying->slave_side_converter();

    // assemble derivatives of slave dofs w.r.t. interior dofs (block i)
    Coupling::Adapter::MatrixLogicalSplitAndTransform()(structure_matrix, *cond_slave_dof_map,
        *map_structure_interior, 1.0, &(*converter), nullptr, ssi_structure_matrix, true, true);

    // assemble derivatives of slave dofs w.r.t. master dofs (block j)
    Coupling::Adapter::MatrixLogicalSplitAndTransform()(structure_matrix, *cond_slave_dof_map,
        *master_dof_map, 1.0, &(*converter), nullptr, ssi_structure_matrix, true, true);

    // assemble derivatives of interior w.r.t. slave dofs (block c)
    Coupling::Adapter::MatrixLogicalSplitAndTransform()(structure_matrix, *map_structure_interior,
        *cond_slave_dof_map, 1.0, nullptr, &(*converter), ssi_structure_matrix, true, true);

    // assemble derivatives of master w.r.t. slave dofs (block g)
    Coupling::Adapter::MatrixLogicalSplitAndTransform()(structure_matrix, *master_dof_map,
        *cond_slave_dof_map, 1.0, nullptr, &(*converter), ssi_structure_matrix, true, true);

    for (const auto& meshtying2 : ssi_structure_meshtying.mesh_tying_handlers())
    {
      auto cond_slave_dof_map2 = meshtying2->slave_master_coupling()->slave_dof_map();
      auto converter2 = meshtying2->slave_side_converter();

      // assemble derivatives of slave dofs w.r.t. other slave dofs (block l)
      Coupling::Adapter::MatrixLogicalSplitAndTransform()(structure_matrix, *cond_slave_dof_map,
          *cond_slave_dof_map2, 1.0, &(*converter), &(*converter2), ssi_structure_matrix, true,
          true);
    }
  }

  finalize_meshtying_structure_matrix(ssi_structure_matrix, ssi_structure_meshtying);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::MeshtyingStrategyBase::apply_meshtying_to_xxx_structure(
    Core::LinAlg::SparseMatrix& ssi_xxx_structure_matrix,
    const Core::LinAlg::SparseMatrix& xxx_structure_matrix,
    const SSI::Utils::SSIMeshTying& ssi_structure_meshtying)
{
  // uncomplete matrix first
  ssi_xxx_structure_matrix.un_complete();

  auto map_structure_interior = ssi_structure_meshtying.interior_map();
  auto master_dof_map = ssi_structure_meshtying.full_master_side_map();

  // assemble derivatives of x w.r.t. structure interior dofs
  Coupling::Adapter::MatrixLogicalSplitAndTransform()(xxx_structure_matrix,
      xxx_structure_matrix.range_map(), *map_structure_interior, 1.0, nullptr, nullptr,
      ssi_xxx_structure_matrix, true, true);

  // assemble derivatives of x w.r.t. structure master dofs
  Coupling::Adapter::MatrixLogicalSplitAndTransform()(xxx_structure_matrix,
      xxx_structure_matrix.range_map(), *master_dof_map, 1.0, nullptr, nullptr,
      ssi_xxx_structure_matrix, true, true);

  auto meshtying_handlers = ssi_structure_meshtying.mesh_tying_handlers();

  for (const auto& meshtying : meshtying_handlers)
  {
    auto cond_slave_dof_map = meshtying->slave_master_coupling()->slave_dof_map();
    auto converter = meshtying->slave_side_converter();

    // assemble derivatives of x w.r.t. structure slave dofs
    Coupling::Adapter::MatrixLogicalSplitAndTransform()(xxx_structure_matrix,
        xxx_structure_matrix.range_map(), *cond_slave_dof_map, 1.0, nullptr, &(*converter),
        ssi_xxx_structure_matrix, true, true);
  }
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
Core::LinAlg::Vector<double> SSI::MeshtyingStrategyBase::apply_meshtying_to_structure_rhs(
    const Core::LinAlg::Vector<double>& structure_rhs, const SSI::Utils::SSIMaps& ssi_maps,
    const SSI::Utils::SSIMeshTying& ssi_structure_meshtying)
{
  // make copy of structure right-hand side vector
  Core::LinAlg::Vector<double> rhs_structure(structure_rhs);

  auto rhs_structure_master =
      std::make_shared<Core::LinAlg::Vector<double>>(*ssi_maps.structure_dof_row_map(), true);

  for (const auto& meshtying : ssi_structure_meshtying.mesh_tying_handlers())
  {
    auto coupling_adapter = meshtying->slave_master_coupling();
    auto coupling_map_extractor = meshtying->slave_master_extractor();

    // transform slave-side part of structure right-hand side vector to master side
    const auto rhs_structure_only_slave_dofs =
        coupling_map_extractor->extract_vector(rhs_structure, 1);

    const auto rhs_structure_only_master_dofs =
        coupling_adapter->slave_to_master(*rhs_structure_only_slave_dofs);

    coupling_map_extractor->add_vector(*rhs_structure_only_master_dofs, 2, *rhs_structure_master);

    // zero out slave-side part of structure right-hand side vector
    coupling_map_extractor->put_scalar(rhs_structure, 1, 0.0);
  }

  // assemble master-side part of structure right-hand side vector
  rhs_structure.update(1.0, *rhs_structure_master, 1.0);

  return rhs_structure;
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
void SSI::MeshtyingStrategySparse::apply_meshtying_to_scatra_manifold_structure(
    std::shared_ptr<Core::LinAlg::SparseOperator> manifold_structure_matrix,
    const SSI::Utils::SSIMaps& ssi_maps, const SSI::Utils::SSIMeshTying& ssi_structure_meshtying,
    const bool do_uncomplete)
{
  temp_scatramanifold_struct_mat_->zero();
  auto temp_scatramanifold_struct_sparse_matrix =
      Core::LinAlg::cast_to_sparse_matrix_and_check_success(temp_scatramanifold_struct_mat_);
  auto manifold_structure_sparse_matrix =
      Core::LinAlg::cast_to_sparse_matrix_and_check_success(manifold_structure_matrix);

  // apply mesh tying contributions to temp matrix and complete it
  apply_meshtying_to_xxx_structure(*temp_scatramanifold_struct_sparse_matrix,
      *manifold_structure_sparse_matrix, ssi_structure_meshtying);
  temp_scatramanifold_struct_sparse_matrix->complete(
      *ssi_maps.structure_dof_row_map(), *ssi_maps.scatra_manifold_dof_row_map());

  if (do_uncomplete) manifold_structure_sparse_matrix->un_complete();
  manifold_structure_sparse_matrix->add(*temp_scatramanifold_struct_sparse_matrix, false, 1.0, 0.0);
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
void SSI::MeshtyingStrategyBlock::apply_meshtying_to_scatra_manifold_structure(
    std::shared_ptr<Core::LinAlg::SparseOperator> manifold_structure_matrix,
    const SSI::Utils::SSIMaps& ssi_maps, const SSI::Utils::SSIMeshTying& ssi_structure_meshtying,
    const bool do_uncomplete)
{
  temp_scatramanifold_struct_mat_->zero();
  auto temp_scatramanifold_struct_block_matrix =
      Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(
          temp_scatramanifold_struct_mat_);
  auto manifold_structure_matrix_block_matrix =
      Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(
          std::move(manifold_structure_matrix));

  // apply mesh tying contributions to temp matrix blocks and complete the resulting matrix
  for (int iblock = 0; iblock < static_cast<int>(block_position_scatra_manifold().size()); ++iblock)
  {
    apply_meshtying_to_xxx_structure(temp_scatramanifold_struct_block_matrix->matrix(iblock, 0),
        manifold_structure_matrix_block_matrix->matrix(iblock, 0), ssi_structure_meshtying);
  }
  temp_scatramanifold_struct_block_matrix->complete();

  if (do_uncomplete) manifold_structure_matrix_block_matrix->un_complete();
  manifold_structure_matrix_block_matrix->add(
      *temp_scatramanifold_struct_block_matrix, false, 1.0, 0.0);
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
void SSI::MeshtyingStrategySparse::apply_meshtying_to_scatra_structure(
    std::shared_ptr<Core::LinAlg::SparseOperator> scatra_structure_matrix,
    const SSI::Utils::SSIMaps& ssi_maps, const SSI::Utils::SSIMeshTying& ssi_structure_meshtying,
    const bool do_uncomplete)
{
  temp_scatra_struct_mat_->zero();
  auto temp_scatra_struct_domain_sparse_matrix =
      Core::LinAlg::cast_to_sparse_matrix_and_check_success(temp_scatra_struct_mat_);
  auto scatra_structure_matrix_sparse =
      Core::LinAlg::cast_to_sparse_matrix_and_check_success(scatra_structure_matrix);

  apply_meshtying_to_xxx_structure(*temp_scatra_struct_domain_sparse_matrix,
      *scatra_structure_matrix_sparse, ssi_structure_meshtying);
  temp_scatra_struct_domain_sparse_matrix->complete(
      *ssi_maps.structure_dof_row_map(), *ssi_maps.scatra_dof_row_map());

  if (do_uncomplete) scatra_structure_matrix_sparse->un_complete();
  scatra_structure_matrix_sparse->add(*temp_scatra_struct_domain_sparse_matrix, false, 1.0, 0.0);
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
void SSI::MeshtyingStrategyBlock::apply_meshtying_to_scatra_structure(
    std::shared_ptr<Core::LinAlg::SparseOperator> scatra_structure_matrix,
    const SSI::Utils::SSIMaps& ssi_maps, const SSI::Utils::SSIMeshTying& ssi_structure_meshtying,
    const bool do_uncomplete)
{
  temp_scatra_struct_mat_->zero();
  auto temp_scatra_struct_domain_block_sparse_matrix =
      Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(temp_scatra_struct_mat_);
  auto scatra_structure_matrix_block_sparse =
      Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(scatra_structure_matrix);

  // apply mesh tying for all blocks
  for (int iblock = 0; iblock < static_cast<int>(block_position_scatra().size()); ++iblock)
  {
    apply_meshtying_to_xxx_structure(
        temp_scatra_struct_domain_block_sparse_matrix->matrix(iblock, 0),
        scatra_structure_matrix_block_sparse->matrix(iblock, 0), ssi_structure_meshtying);
  }
  temp_scatra_struct_domain_block_sparse_matrix->complete();

  if (do_uncomplete) scatra_structure_matrix_block_sparse->un_complete();
  scatra_structure_matrix_block_sparse->add(
      *temp_scatra_struct_domain_block_sparse_matrix, false, 1.0, 0.0);
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
void SSI::MeshtyingStrategySparse::apply_meshtying_to_structure_scatra(
    std::shared_ptr<Core::LinAlg::SparseOperator> structure_scatra_matrix,
    const SSI::Utils::SSIMaps& ssi_maps, const SSI::Utils::SSIMeshTying& ssi_structure_meshtying,
    const bool do_uncomplete)
{
  temp_struct_scatra_mat_->zero();
  auto temp_struct_scatra_sparse_matrix =
      Core::LinAlg::cast_to_sparse_matrix_and_check_success(temp_struct_scatra_mat_);
  auto struct_scatra_sparse_matrix =
      Core::LinAlg::cast_to_sparse_matrix_and_check_success(structure_scatra_matrix);

  // apply mesh tying contributions to temp matrix and complete it
  if (do_uncomplete) temp_struct_scatra_sparse_matrix->un_complete();
  apply_meshtying_to_structure_xxx(
      *temp_struct_scatra_sparse_matrix, *struct_scatra_sparse_matrix, ssi_structure_meshtying);
  temp_struct_scatra_sparse_matrix->complete(
      *ssi_maps.scatra_dof_row_map(), *ssi_maps.structure_dof_row_map());

  if (do_uncomplete) struct_scatra_sparse_matrix->un_complete();
  struct_scatra_sparse_matrix->add(*temp_struct_scatra_sparse_matrix, false, 1.0, 0.0);
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
void SSI::MeshtyingStrategyBlock::apply_meshtying_to_structure_scatra(
    std::shared_ptr<Core::LinAlg::SparseOperator> structure_scatra_matrix,
    const SSI::Utils::SSIMaps& ssi_maps, const SSI::Utils::SSIMeshTying& ssi_structure_meshtying,
    const bool do_uncomplete)
{
  temp_struct_scatra_mat_->zero();
  auto temp_struct_scatra_block_matrix =
      Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(temp_struct_scatra_mat_);
  auto structure_scatra_block_matrix =
      Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(structure_scatra_matrix);

  // apply mesh tying contributions to temp matrix blocks and complete the resulting matrix
  if (do_uncomplete) temp_struct_scatra_block_matrix->un_complete();
  for (int iblock = 0; iblock < static_cast<int>(block_position_scatra().size()); ++iblock)
  {
    apply_meshtying_to_structure_xxx(temp_struct_scatra_block_matrix->matrix(0, iblock),
        structure_scatra_block_matrix->matrix(0, iblock), ssi_structure_meshtying);
  }
  temp_struct_scatra_block_matrix->complete();

  if (do_uncomplete) structure_scatra_block_matrix->un_complete();
  structure_scatra_block_matrix->add(*temp_struct_scatra_block_matrix, false, 1.0, 0.0);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::MeshtyingStrategyBase::apply_meshtying_to_structure_xxx(
    Core::LinAlg::SparseMatrix& ssi_structure_xxx_matrix,
    const Core::LinAlg::SparseMatrix& structure_xxx_matrix,
    const SSI::Utils::SSIMeshTying& ssi_structure_meshtying)
{
  auto map_structure_interior = ssi_structure_meshtying.interior_map();
  auto master_dof_map = ssi_structure_meshtying.full_master_side_map();

  // assemble derivatives of structure interior dofs w.r.t. scatra dofs
  Coupling::Adapter::MatrixLogicalSplitAndTransform()(structure_xxx_matrix, *map_structure_interior,
      structure_xxx_matrix.domain_map(), 1.0, nullptr, nullptr, ssi_structure_xxx_matrix, true,
      true);
  // assemble derivatives of structure master dofs w.r.t. scatra dofs
  Coupling::Adapter::MatrixLogicalSplitAndTransform()(structure_xxx_matrix, *master_dof_map,
      structure_xxx_matrix.domain_map(), 1.0, nullptr, nullptr, ssi_structure_xxx_matrix, true,
      true);

  for (const auto& meshtying : ssi_structure_meshtying.mesh_tying_handlers())
  {
    auto cond_slave_dof_map = meshtying->slave_master_coupling()->slave_dof_map();
    auto converter = meshtying->slave_side_converter();

    // assemble derivatives of structure slave dofs & interior dofs w.r.t. scatra dofs
    Coupling::Adapter::MatrixLogicalSplitAndTransform()(structure_xxx_matrix, *cond_slave_dof_map,
        structure_xxx_matrix.domain_map(), 1.0, &(*converter), nullptr, ssi_structure_xxx_matrix,
        true, true);
  }
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
void SSI::MeshtyingStrategyBase::finalize_meshtying_structure_matrix(
    Core::LinAlg::SparseMatrix& ssi_structure_matrix,
    const SSI::Utils::SSIMeshTying& ssi_structure_meshtying)
{
  // map for slave side structure degrees of freedom
  auto slavemaps = ssi_structure_meshtying.full_slave_side_map();

  // subject slave-side rows of structure system matrix to pseudo Dirichlet conditions to finalize
  // structure mesh tying
  const double one(1.0);
  for (int doflid_slave = 0; doflid_slave < slavemaps->num_my_elements(); ++doflid_slave)
  {
    // extract global ID of current slave-side row
    const int dofgid_slave = slavemaps->gid(doflid_slave);
    if (dofgid_slave < 0) FOUR_C_THROW("Local ID not found!");

    // apply pseudo Dirichlet conditions to filled matrix, i.e., to local row and column indices
    if (ssi_structure_matrix.filled())
    {
      const int rowlid_slave = ssi_structure_matrix.row_map().lid(dofgid_slave);
      if (rowlid_slave < 0) FOUR_C_THROW("Global ID not found!");
      ssi_structure_matrix.replace_my_values(rowlid_slave, 1, &one, &rowlid_slave);
    }

    // apply pseudo Dirichlet conditions to unfilled matrix, i.e., to global row and column indices
    else
    {
      ssi_structure_matrix.insert_global_values(dofgid_slave, 1, &one, &dofgid_slave);
    }
  }
}
/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
std::unique_ptr<SSI::MeshtyingStrategyBase> SSI::build_meshtying_strategy(
    const bool is_scatra_manifold, const Core::LinAlg::MatrixType matrixtype_scatra,
    const SSI::Utils::SSIMaps& ssi_maps)
{
  switch (matrixtype_scatra)
  {
    case Core::LinAlg::MatrixType::block_condition:
    case Core::LinAlg::MatrixType::block_condition_dof:
    {
      return std::make_unique<SSI::MeshtyingStrategyBlock>(is_scatra_manifold, ssi_maps);
    }
    case Core::LinAlg::MatrixType::sparse:
    {
      return std::make_unique<SSI::MeshtyingStrategySparse>(is_scatra_manifold, ssi_maps);
    }

    default:
    {
      FOUR_C_THROW("unknown matrix type of ScaTra field");
    }
  }
}
FOUR_C_NAMESPACE_CLOSE

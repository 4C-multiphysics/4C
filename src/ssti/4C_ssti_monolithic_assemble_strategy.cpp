// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_ssti_monolithic_assemble_strategy.hpp"

#include "4C_adapter_str_ssiwrapper.hpp"
#include "4C_coupling_adapter.hpp"
#include "4C_coupling_adapter_converter.hpp"
#include "4C_fem_condition_locsys.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_scatra_timint_meshtying_strategy_s2i.hpp"
#include "4C_ssi_utils.hpp"
#include "4C_ssti_monolithic.hpp"
#include "4C_ssti_utils.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
SSTI::AssembleStrategyBase::AssembleStrategyBase(std::shared_ptr<const SSTI::SSTIMono> ssti_mono)
    : ssti_mono_(std::move(ssti_mono))
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
SSTI::AssembleStrategyBlock::AssembleStrategyBlock(std::shared_ptr<const SSTI::SSTIMono> ssti_mono)
    : AssembleStrategyBase(ssti_mono),
      block_position_scatra_(ssti_mono->get_block_positions(SSTI::Subproblem::scalar_transport)),
      block_position_thermo_(ssti_mono->get_block_positions(SSTI::Subproblem::thermo)),
      position_structure_(ssti_mono->get_block_positions(SSTI::Subproblem::structure).at(0))
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
SSTI::AssembleStrategyBlockBlock::AssembleStrategyBlockBlock(
    std::shared_ptr<const SSTI::SSTIMono> ssti_mono)
    : AssembleStrategyBlock(std::move(ssti_mono))
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
SSTI::AssembleStrategyBlockSparse::AssembleStrategyBlockSparse(
    std::shared_ptr<const SSTI::SSTIMono> ssti_mono)
    : AssembleStrategyBlock(std::move(ssti_mono))
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
SSTI::AssembleStrategySparse::AssembleStrategySparse(
    std::shared_ptr<const SSTI::SSTIMono> ssti_mono)
    : AssembleStrategyBase(std::move(ssti_mono))
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBlockBlock::assemble_scatra(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
    std::shared_ptr<const Core::LinAlg::SparseOperator> scatradomain)
{
  auto systemmatrix_block =
      Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(systemmatrix);
  auto scatradomain_block =
      Core::LinAlg::cast_to_const_block_sparse_matrix_base_and_check_success(scatradomain);

  // assemble blocks of scalar transport system matrix into global system matrix
  for (int iblock = 0; iblock < static_cast<int>(block_position_scatra().size()); ++iblock)
  {
    for (int jblock = 0; jblock < static_cast<int>(block_position_scatra().size()); ++jblock)
    {
      auto& systemmatrix_block_iscatra_jscatra = systemmatrix_block->matrix(
          block_position_scatra().at(iblock), block_position_scatra().at(jblock));

      Core::LinAlg::matrix_add(scatradomain_block->matrix(iblock, jblock), false, 1.0,
          systemmatrix_block_iscatra_jscatra, 1.0);
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBlockSparse::assemble_scatra(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
    std::shared_ptr<const Core::LinAlg::SparseOperator> scatradomain)
{
  auto systemmatrix_block =
      Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(systemmatrix);
  auto scatradomain_sparse =
      Core::LinAlg::cast_to_const_sparse_matrix_and_check_success(scatradomain);

  auto& systemmatrix_block_scatra_scatra =
      systemmatrix_block->matrix(block_position_scatra().at(0), block_position_scatra().at(0));

  Core::LinAlg::matrix_add(*scatradomain_sparse, false, 1.0, systemmatrix_block_scatra_scatra, 1.0);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategySparse::assemble_scatra(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
    std::shared_ptr<const Core::LinAlg::SparseOperator> scatradomain)
{
  auto systemmatrix_sparse = Core::LinAlg::cast_to_sparse_matrix_and_check_success(systemmatrix);
  auto scatradomain_sparse =
      Core::LinAlg::cast_to_const_sparse_matrix_and_check_success(scatradomain);

  // add scalar transport system matrix to global system matrix
  Core::LinAlg::matrix_add(*scatradomain_sparse, false, 1.0, *systemmatrix_sparse, 1.0);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBlockBlock::assemble_structure(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
    std::shared_ptr<const Core::LinAlg::SparseMatrix> structuredomain)
{
  auto systemmatrix_block =
      Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(systemmatrix);

  // add entire block or assemble slave side to master side
  if (interface_meshtying())
  {
    assemble_structure_meshtying(
        systemmatrix_block->matrix(position_structure(), position_structure()), structuredomain);
  }
  else
  {
    auto& systemmatrix_block_struct_struct =
        systemmatrix_block->matrix(position_structure(), position_structure());

    Core::LinAlg::matrix_add(*structuredomain, false, 1.0, systemmatrix_block_struct_struct, 1.0);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBlockSparse::assemble_structure(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
    std::shared_ptr<const Core::LinAlg::SparseMatrix> structuredomain)
{
  auto systemmatrix_block =
      Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(systemmatrix);

  // add entire block or assemble slave side to master side
  if (interface_meshtying())
  {
    assemble_structure_meshtying(
        systemmatrix_block->matrix(position_structure(), position_structure()), structuredomain);
  }
  else
  {
    auto& systemmatrix_block_struct_struct =
        systemmatrix_block->matrix(position_structure(), position_structure());

    Core::LinAlg::matrix_add(*structuredomain, false, 1.0, systemmatrix_block_struct_struct, 1.0);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategySparse::assemble_structure(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
    std::shared_ptr<const Core::LinAlg::SparseMatrix> structuredomain)
{
  auto systemmatrix_sparse = Core::LinAlg::cast_to_sparse_matrix_and_check_success(systemmatrix);

  // add entire block or assemble slave side to master side
  if (interface_meshtying())
    assemble_structure_meshtying(*systemmatrix_sparse, structuredomain);
  else
    Core::LinAlg::matrix_add(*structuredomain, false, 1.0, *systemmatrix_sparse, 1.0);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBase::assemble_structure_meshtying(
    Core::LinAlg::SparseMatrix& systemmatrix_structure,
    std::shared_ptr<const Core::LinAlg::SparseMatrix> structuredomain)
{
  /* Transform and assemble the structure matrix into the global system matrix block by block:
   * S_i:  structure interior dofs
   * S_m:  structure master side dofs
   * S_ss: structure slave surface dofs
   * S_sl: structure slave line dofs
   *
   *      | S_i | S_m | S_ss| S_sl|
   *      |-----|-----|-----|-----|
   * S_i  |  a  |  b  |  c  |  d  |
   * S_m  |  e  |  f  |  g  |  -  |
   * S_ss |  h  |  i  |  j  |  k  |
   * S_sl |  l  |  -  |  m  |  n  |
   *      -------------------------
   */

  auto map_structure_interior = ssti_structure_meshtying()->interior_map();
  auto master_dof_map = ssti_structure_meshtying()->full_master_side_map();

  // assemble derivs. of interior dofs w.r.t. interior dofs (block a)
  Coupling::Adapter::MatrixLogicalSplitAndTransform()(*structuredomain, *map_structure_interior,
      *map_structure_interior, 1.0, nullptr, nullptr, systemmatrix_structure, true, true);

  // assemble derivs. of interior dofs w.r.t. master dofs (block b)
  Coupling::Adapter::MatrixLogicalSplitAndTransform()(*structuredomain, *map_structure_interior,
      *master_dof_map, 1.0, nullptr, nullptr, systemmatrix_structure, true, true);

  // assemble derivs. of master dofs w.r.t. interior dofs (block e)
  Coupling::Adapter::MatrixLogicalSplitAndTransform()(*structuredomain, *master_dof_map,
      *map_structure_interior, 1.0, nullptr, nullptr, systemmatrix_structure, true, true);

  // assemble derivs. of master dofs w.r.t. master dofs (block f)
  Coupling::Adapter::MatrixLogicalSplitAndTransform()(*structuredomain, *master_dof_map,
      *master_dof_map, 1.0, nullptr, nullptr, systemmatrix_structure, true, true);

  for (const auto& meshtying : ssti_structure_meshtying()->mesh_tying_handlers())
  {
    auto cond_slave_dof_map = meshtying->slave_master_coupling()->slave_dof_map();
    auto converter = meshtying->slave_side_converter();

    // assemble derivs. of surface slave dofs w.r.t. interior dofs (block h)
    Coupling::Adapter::MatrixLogicalSplitAndTransform()(*structuredomain, *cond_slave_dof_map,
        *map_structure_interior, 1.0, &(*converter), nullptr, systemmatrix_structure, true, true);

    // assemble derivs. of surface slave dofs w.r.t. master dofs (block i)
    Coupling::Adapter::MatrixLogicalSplitAndTransform()(*structuredomain, *cond_slave_dof_map,
        *master_dof_map, 1.0, &(*converter), nullptr, systemmatrix_structure, true, true);

    // assemble derivs. of interior dofs w.r.t. surface slave dofs (block c)
    Coupling::Adapter::MatrixLogicalSplitAndTransform()(*structuredomain, *map_structure_interior,
        *cond_slave_dof_map, 1.0, nullptr, &(*converter), systemmatrix_structure, true, true);

    // assemble derivs. of master dofs w.r.t. surface slave dofs (block g)
    Coupling::Adapter::MatrixLogicalSplitAndTransform()(*structuredomain, *master_dof_map,
        *cond_slave_dof_map, 1.0, nullptr, &(*converter), systemmatrix_structure, true, true);

    // assemble derivs. of surface slave dofs w.r.t. surface slave dofs (block j)
    Coupling::Adapter::MatrixLogicalSplitAndTransform()(*structuredomain, *cond_slave_dof_map,
        *cond_slave_dof_map, 1.0, &(*converter), &(*converter), systemmatrix_structure, true, true);

    for (const auto& meshtying2 : ssti_structure_meshtying()->mesh_tying_handlers())
    {
      if (meshtying2 != meshtying)
      {
        auto cond_slave_dof_map2 = meshtying2->slave_master_coupling()->slave_dof_map();
        auto converter2 = meshtying2->slave_side_converter();

        // assemble derivatives of surface slave dofs w.r.t. line slave dofs (block l)
        Coupling::Adapter::MatrixLogicalSplitAndTransform()(*structuredomain, *cond_slave_dof_map,
            *cond_slave_dof_map2, 1.0, &(*converter), &(*converter2), systemmatrix_structure, true,
            true);
      }
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBlockBlock::assemble_thermo(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
    std::shared_ptr<const Core::LinAlg::SparseOperator> thermodomain)
{
  auto systemmatrix_block =
      Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(systemmatrix);
  auto thermodomain_block =
      Core::LinAlg::cast_to_const_block_sparse_matrix_base_and_check_success(thermodomain);

  // assemble blocks of scalar transport system matrix into global system matrix
  for (int iblock = 0; iblock < static_cast<int>(block_position_thermo().size()); ++iblock)
  {
    for (int jblock = 0; jblock < static_cast<int>(block_position_thermo().size()); ++jblock)
    {
      auto& systemmatrix_block_ithermo_jthermo = systemmatrix_block->matrix(
          block_position_thermo().at(iblock), block_position_thermo().at(jblock));

      Core::LinAlg::matrix_add(thermodomain_block->matrix(iblock, jblock), false, 1.0,
          systemmatrix_block_ithermo_jthermo, 1.0);
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBlockSparse::assemble_thermo(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
    std::shared_ptr<const Core::LinAlg::SparseOperator> thermodomain)
{
  auto systemmatrix_block =
      Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(systemmatrix);
  auto thermodomain_sparse =
      Core::LinAlg::cast_to_const_sparse_matrix_and_check_success(thermodomain);

  auto& systemmatrix_block_thermo_thermo =
      systemmatrix_block->matrix(block_position_thermo().at(0), block_position_thermo().at(0));

  Core::LinAlg::matrix_add(*thermodomain_sparse, false, 1.0, systemmatrix_block_thermo_thermo, 1.0);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategySparse::assemble_thermo(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
    std::shared_ptr<const Core::LinAlg::SparseOperator> thermodomain)
{
  auto systemmatrix_sparse = Core::LinAlg::cast_to_sparse_matrix_and_check_success(systemmatrix);
  auto thermodomain_sparse =
      Core::LinAlg::cast_to_const_sparse_matrix_and_check_success(thermodomain);

  // add scalar transport system matrix to global system matrix
  Core::LinAlg::matrix_add(*thermodomain_sparse, false, 1.0, *systemmatrix_sparse, 1.0);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBlockBlock::assemble_scatra_structure(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
    std::shared_ptr<const Core::LinAlg::SparseOperator> scatrastructuredomain,
    std::shared_ptr<const Core::LinAlg::SparseOperator> scatrastructureinterface)
{
  auto systemmatrix_block =
      Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(systemmatrix);
  auto scatrastructuredomain_block =
      Core::LinAlg::cast_to_const_block_sparse_matrix_base_and_check_success(scatrastructuredomain);

  // assemble blocks of scalar transport system matrix into global system matrix
  for (int iblock = 0; iblock < static_cast<int>(block_position_scatra().size()); ++iblock)
  {
    const auto scatrastructuredomain_subblock = scatrastructuredomain_block->matrix(iblock, 0);

    // add entire block or assemble slave side to master side
    if (interface_meshtying())
    {
      assemble_xxx_structure_meshtying(
          systemmatrix_block->matrix(block_position_scatra().at(iblock), position_structure()),
          scatrastructuredomain_block->matrix(iblock, 0));

      auto scatrastructureinterface_block =
          Core::LinAlg::cast_to_const_block_sparse_matrix_base_and_check_success(
              scatrastructureinterface);

      assemble_xxx_structure_meshtying(
          systemmatrix_block->matrix(block_position_scatra().at(iblock), position_structure()),
          scatrastructureinterface_block->matrix(iblock, 0));
    }
    else
    {
      auto& systemmatrix_block_iscatra_struct =
          systemmatrix_block->matrix(block_position_scatra().at(iblock), position_structure());

      Core::LinAlg::matrix_add(
          scatrastructuredomain_subblock, false, 1.0, systemmatrix_block_iscatra_struct, 1.0);
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBlockSparse::assemble_scatra_structure(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
    std::shared_ptr<const Core::LinAlg::SparseOperator> scatrastructuredomain,
    std::shared_ptr<const Core::LinAlg::SparseOperator> scatrastructureinterface)
{
  auto systemmatrix_block =
      Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(systemmatrix);

  auto scatrastructuredomain_sparse =
      Core::LinAlg::cast_to_const_sparse_matrix_and_check_success(scatrastructuredomain);

  // add entire block or assemble slave side to master side
  if (interface_meshtying())
  {
    assemble_xxx_structure_meshtying(
        systemmatrix_block->matrix(block_position_scatra().at(0), position_structure()),
        *scatrastructuredomain_sparse);

    auto scatrastructureinterface_sparse =
        Core::LinAlg::cast_to_const_sparse_matrix_and_check_success(scatrastructureinterface);

    assemble_xxx_structure_meshtying(
        systemmatrix_block->matrix(block_position_scatra().at(0), position_structure()),
        *scatrastructureinterface_sparse);
  }
  else
  {
    auto& systemmatrix_block_scatra_struct =
        systemmatrix_block->matrix(block_position_scatra().at(0), position_structure());

    Core::LinAlg::matrix_add(
        *scatrastructuredomain_sparse, false, 1.0, systemmatrix_block_scatra_struct, 1.0);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategySparse::assemble_scatra_structure(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
    std::shared_ptr<const Core::LinAlg::SparseOperator> scatrastructuredomain,
    std::shared_ptr<const Core::LinAlg::SparseOperator> scatrastructureinterface)
{
  auto systemmatrix_sparse = Core::LinAlg::cast_to_sparse_matrix_and_check_success(systemmatrix);
  auto scatrastructuredomain_sparse =
      Core::LinAlg::cast_to_const_sparse_matrix_and_check_success(scatrastructuredomain);

  // add entire block or assemble slave side to master side
  if (interface_meshtying())
  {
    assemble_xxx_structure_meshtying(*systemmatrix_sparse, *scatrastructuredomain_sparse);

    auto scatrastructureinterface_sparse =
        Core::LinAlg::cast_to_const_sparse_matrix_and_check_success(scatrastructureinterface);

    assemble_xxx_structure_meshtying(*systemmatrix_sparse, *scatrastructureinterface_sparse);
  }

  else
    Core::LinAlg::matrix_add(*scatrastructuredomain_sparse, false, 1.0, *systemmatrix_sparse, 1.0);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBase::assemble_xxx_structure_meshtying(
    Core::LinAlg::SparseMatrix& systemmatrix_x_structure,
    const Core::LinAlg::SparseMatrix& x_structurematrix)
{
  auto map_structure_interior = ssti_structure_meshtying()->interior_map();
  auto master_dof_map = ssti_structure_meshtying()->full_master_side_map();

  // assemble derivs. of x w.r.t. structural interior dofs
  Coupling::Adapter::MatrixLogicalSplitAndTransform()(x_structurematrix,
      x_structurematrix.range_map(), *map_structure_interior, 1.0, nullptr, nullptr,
      systemmatrix_x_structure, true, true);

  // assemble derivs. of x w.r.t. structural master dofs
  Coupling::Adapter::MatrixLogicalSplitAndTransform()(x_structurematrix,
      x_structurematrix.range_map(), *master_dof_map, 1.0, nullptr, nullptr,
      systemmatrix_x_structure, true, true);

  for (const auto& meshtying : ssti_structure_meshtying()->mesh_tying_handlers())
  {
    auto cond_slave_dof_map = meshtying->slave_master_coupling()->slave_dof_map();
    auto converter = meshtying->slave_side_converter();

    // assemble derivs. of x w.r.t. structural surface slave dofs
    Coupling::Adapter::MatrixLogicalSplitAndTransform()(x_structurematrix,
        x_structurematrix.range_map(), *cond_slave_dof_map, 1.0, nullptr, &(*converter),
        systemmatrix_x_structure, true, true);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBlockBlock::assemble_scatra_thermo_domain(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
    std::shared_ptr<Core::LinAlg::SparseOperator> scatrathermodomain)
{
  auto systemmatrix_block =
      Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(systemmatrix);
  auto scatrathermodomain_block =
      Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(scatrathermodomain);

  // assemble blocks of scalar transport-thermo matrix into global system matrix
  for (int iblock = 0; iblock < static_cast<int>(block_position_scatra().size()); ++iblock)
  {
    for (int jblock = 0; jblock < static_cast<int>(block_position_thermo().size()); ++jblock)
    {
      auto scatrathermodomain_subblock = scatrathermodomain_block->matrix(iblock, jblock);
      auto systemmatrix_subblock = systemmatrix_block->matrix(
          block_position_scatra().at(iblock), block_position_thermo().at(jblock));
      systemmatrix_subblock.un_complete();
      Core::LinAlg::matrix_add(scatrathermodomain_subblock, false, 1.0, systemmatrix_subblock, 1.0);
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBlockSparse::assemble_scatra_thermo_domain(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
    std::shared_ptr<Core::LinAlg::SparseOperator> scatrathermodomain)
{
  auto systemmatrix_subblock =
      Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(systemmatrix)
          ->matrix(block_position_scatra().at(0), block_position_thermo().at(0));
  auto scatrathermodomain_sparse =
      Core::LinAlg::cast_to_sparse_matrix_and_check_success(scatrathermodomain);

  systemmatrix_subblock.un_complete();
  Core::LinAlg::matrix_add(*scatrathermodomain_sparse, false, 1.0, systemmatrix_subblock, 1.0);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategySparse::assemble_scatra_thermo_domain(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
    std::shared_ptr<Core::LinAlg::SparseOperator> scatrathermodomain)
{
  auto systemmatrix_sparse = Core::LinAlg::cast_to_sparse_matrix_and_check_success(systemmatrix);
  auto scatrathermodomain_sparse =
      Core::LinAlg::cast_to_sparse_matrix_and_check_success(scatrathermodomain);

  // add scalar transport-thermo matrix into global system matrix
  Core::LinAlg::matrix_add(*scatrathermodomain_sparse, false, 1.0, *systemmatrix_sparse, 1.0);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBlockBlock::assemble_scatra_thermo_interface(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
    std::shared_ptr<const Core::LinAlg::SparseOperator> scatrathermointerface)
{
  auto systemmatrix_block =
      Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(systemmatrix);
  auto scatrathermointerface_block =
      Core::LinAlg::cast_to_const_block_sparse_matrix_base_and_check_success(scatrathermointerface);

  Core::LinAlg::SparseMatrix masterderiv(
      *all_maps()->block_map_scatra()->full_map(), 27, false, true);

  for (int i = 0; i < static_cast<int>(block_position_scatra().size()); ++i)
  {
    for (int j = 0; j < static_cast<int>(block_position_thermo().size()); ++j)
    {
      const auto scatrathermointerface_subblock = scatrathermointerface_block->matrix(i, j);

      // assemble linearizations of slave- and master side scatra fluxes w.r.t. slave temperatures
      // into system matrix
      auto& systemmatrix_block_iscatra_jthermo =
          systemmatrix_block->matrix(block_position_scatra().at(i), block_position_thermo().at(j));
      Core::LinAlg::matrix_add(
          scatrathermointerface_subblock, false, 1.0, systemmatrix_block_iscatra_jthermo, 1.0);

      // assemble linearizations of slave- and master side scatra fluxes w.r.t. master temperatures
      // into system matrix
      Coupling::Adapter::CouplingSlaveConverter thermo_converter(
          *meshtying_thermo()->coupling_adapter());

      Coupling::Adapter::MatrixLogicalSplitAndTransform()(scatrathermointerface_subblock,
          scatrathermointerface_subblock.range_map(),
          *meshtying_thermo()->coupling_adapter()->master_dof_map(), 1.0, nullptr,
          &thermo_converter, masterderiv, true, true);
    }
  }

  masterderiv.complete(*meshtying_thermo()->coupling_adapter()->master_dof_map(),
      *all_maps()->block_map_scatra()->full_map());

  const auto blockmasterderiv =
      Core::LinAlg::split_matrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
          masterderiv, *all_maps()->block_map_thermo(), *all_maps()->block_map_scatra());

  blockmasterderiv->complete();

  for (int i = 0; i < static_cast<int>(block_position_scatra().size()); ++i)
  {
    for (int j = 0; j < static_cast<int>(block_position_thermo().size()); ++j)
    {
      // assemble linearizations of slave side scatra fluxes w.r.t. slave and master side elch
      // into system matrix
      auto& systemmatrix_block_iscatra_jthermo =
          systemmatrix_block->matrix(block_position_scatra().at(i), block_position_thermo().at(j));
      Core::LinAlg::matrix_add(
          blockmasterderiv->matrix(i, j), false, 1.0, systemmatrix_block_iscatra_jthermo, 1.0);
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBlockSparse::assemble_scatra_thermo_interface(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
    std::shared_ptr<const Core::LinAlg::SparseOperator> scatrathermointerface)
{
  auto systemmatrix_block =
      Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(systemmatrix);
  auto scatrathermointerface_sparse =
      Core::LinAlg::cast_to_const_sparse_matrix_and_check_success(scatrathermointerface);

  // assemble linearizations of slave- and master side scatra fluxes w.r.t. slave temperatures into
  // system matrix
  auto& systemmatrix_block_scatra_thermo =
      systemmatrix_block->matrix(block_position_scatra().at(0), block_position_thermo().at(0));
  Core::LinAlg::matrix_add(
      *scatrathermointerface_sparse, false, 1.0, systemmatrix_block_scatra_thermo, 1.0);

  // assemble linearizations of slave- and master side scatra fluxes w.r.t. master temperatures into
  // system matrix
  Coupling::Adapter::CouplingSlaveConverter thermo_converter(
      *meshtying_thermo()->coupling_adapter());

  Coupling::Adapter::MatrixLogicalSplitAndTransform()(*scatrathermointerface_sparse,
      scatrathermointerface_sparse->range_map(),
      *meshtying_thermo()->coupling_adapter()->master_dof_map(), 1.0, nullptr, &thermo_converter,
      systemmatrix_block->matrix(block_position_scatra().at(0), block_position_thermo().at(0)),
      true, true);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategySparse::assemble_scatra_thermo_interface(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
    std::shared_ptr<const Core::LinAlg::SparseOperator> scatrathermointerface)
{
  auto systemmatrix_sparse = Core::LinAlg::cast_to_sparse_matrix_and_check_success(systemmatrix);
  auto scatrathermointerface_sparse =
      Core::LinAlg::cast_to_const_sparse_matrix_and_check_success(scatrathermointerface);

  // assemble linearizations of slave- and master side scatra fluxes w.r.t. slave temperatures into
  // system matrix
  Core::LinAlg::matrix_add(*scatrathermointerface_sparse, false, 1.0, *systemmatrix_sparse, 1.0);

  // assemble linearizations of slave- and master side scatra fluxes w.r.t. master temperatures into
  // system matrix
  Coupling::Adapter::CouplingSlaveConverter thermo_converter(
      *meshtying_thermo()->coupling_adapter());

  Coupling::Adapter::MatrixLogicalSplitAndTransform()(*scatrathermointerface_sparse,
      scatrathermointerface_sparse->range_map(),
      *meshtying_thermo()->coupling_adapter()->master_dof_map(), 1.0, nullptr, &thermo_converter,
      *systemmatrix_sparse, true, true);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBlockBlock::assemble_structure_scatra(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
    std::shared_ptr<const Core::LinAlg::SparseOperator> structurescatradomain)
{
  auto systemmatrix_block =
      Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(systemmatrix);
  auto structurescatradomain_block =
      Core::LinAlg::cast_to_const_block_sparse_matrix_base_and_check_success(structurescatradomain);

  // assemble blocks of scalar transport system matrix into global system matrix
  for (int iblock = 0; iblock < static_cast<int>(block_position_scatra().size()); ++iblock)
  {
    const auto structurescatradomain_subblock = structurescatradomain_block->matrix(0, iblock);

    // add entire block or assemble slave side to master side
    if (interface_meshtying())
    {
      assemble_structure_xxx_meshtying(
          systemmatrix_block->matrix(position_structure(), block_position_scatra().at(iblock)),
          structurescatradomain_subblock);
    }
    else
    {
      auto& systemmatrix_block_struct_iscatra =
          systemmatrix_block->matrix(position_structure(), block_position_scatra().at(iblock));

      Core::LinAlg::matrix_add(
          structurescatradomain_subblock, false, 1.0, systemmatrix_block_struct_iscatra, 1.0);
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBlockSparse::assemble_structure_scatra(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
    std::shared_ptr<const Core::LinAlg::SparseOperator> structurescatradomain)
{
  auto systemmatrix_block =
      Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(systemmatrix);
  auto structurescatradomain_sparse =
      Core::LinAlg::cast_to_const_sparse_matrix_and_check_success(structurescatradomain);

  // add entire block or assemble slave side to master side
  if (interface_meshtying())
  {
    assemble_structure_xxx_meshtying(
        systemmatrix_block->matrix(position_structure(), block_position_scatra().at(0)),
        *structurescatradomain_sparse);
  }
  else
  {
    auto& systemmatrix_block_struct_scatra =
        systemmatrix_block->matrix(position_structure(), block_position_scatra().at(0));

    Core::LinAlg::matrix_add(
        *structurescatradomain_sparse, false, 1.0, systemmatrix_block_struct_scatra, 1.0);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategySparse::assemble_structure_scatra(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
    std::shared_ptr<const Core::LinAlg::SparseOperator> structurescatradomain)
{
  auto systemmatrix_sparse = Core::LinAlg::cast_to_sparse_matrix_and_check_success(systemmatrix);
  auto structurescatradomain_sparse =
      Core::LinAlg::cast_to_const_sparse_matrix_and_check_success(structurescatradomain);

  // add entire block or assemble slave side to master side
  if (interface_meshtying())
    assemble_structure_xxx_meshtying(*systemmatrix_sparse, *structurescatradomain_sparse);
  else
    Core::LinAlg::matrix_add(*structurescatradomain_sparse, false, 1.0, *systemmatrix_sparse, 1.0);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBase::assemble_structure_xxx_meshtying(
    Core::LinAlg::SparseMatrix& systemmatrix_structure_x,
    const Core::LinAlg::SparseMatrix& structures_x_matrix)
{
  auto map_structure_interior = ssti_structure_meshtying()->interior_map();
  auto master_dof_map = ssti_structure_meshtying()->full_master_side_map();

  // assemble derivs. of structural interior dofs w.r.t. scatra dofs
  Coupling::Adapter::MatrixLogicalSplitAndTransform()(structures_x_matrix, *map_structure_interior,
      structures_x_matrix.domain_map(), 1.0, nullptr, nullptr, systemmatrix_structure_x, true,
      true);

  // assemble derivs. of structural master dofs w.r.t. scatra dofs
  Coupling::Adapter::MatrixLogicalSplitAndTransform()(structures_x_matrix, *master_dof_map,
      structures_x_matrix.domain_map(), 1.0, nullptr, nullptr, systemmatrix_structure_x, true,
      true);

  for (const auto& meshtying : ssti_structure_meshtying()->mesh_tying_handlers())
  {
    auto cond_slave_dof_map = meshtying->slave_master_coupling()->slave_dof_map();
    auto converter = meshtying->slave_side_converter();

    // assemble derivs. of structural surface slave dofs w.r.t. scatra dofs
    Coupling::Adapter::MatrixLogicalSplitAndTransform()(structures_x_matrix, *cond_slave_dof_map,
        structures_x_matrix.domain_map(), 1.0, &(*converter), nullptr, systemmatrix_structure_x,
        true, true);
  }
}
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBlockBlock::assemble_thermo_scatra(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
    std::shared_ptr<const Core::LinAlg::SparseOperator> thermoscatradomain,
    std::shared_ptr<const Core::LinAlg::SparseOperator> thermoscatrainterface)
{
  auto systemmatrix_block =
      Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(systemmatrix);
  auto thermoscatradomain_block =
      Core::LinAlg::cast_to_const_block_sparse_matrix_base_and_check_success(thermoscatradomain);

  // assemble blocks of scalar transport system matrix into global system matrix
  for (int iblock = 0; iblock < static_cast<int>(block_position_thermo().size()); ++iblock)
  {
    for (int jblock = 0; jblock < static_cast<int>(block_position_scatra().size()); ++jblock)
    {
      auto systemmatrix_block_ithermo_jscatra = systemmatrix_block->matrix(
          block_position_thermo().at(iblock), block_position_scatra().at(jblock));
      systemmatrix_block_ithermo_jscatra.un_complete();
      Core::LinAlg::matrix_add(thermoscatradomain_block->matrix(iblock, jblock), false, 1.0,
          systemmatrix_block_ithermo_jscatra, 1.0);
    }
  }

  if (interface_meshtying()) assemble_thermo_scatra_interface(systemmatrix, thermoscatrainterface);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBlockSparse::assemble_thermo_scatra(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
    std::shared_ptr<const Core::LinAlg::SparseOperator> thermoscatradomain,
    std::shared_ptr<const Core::LinAlg::SparseOperator> thermoscatrainterface)
{
  auto systemmatrix_block =
      Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(systemmatrix);
  auto thermoscatradomain_sparse =
      Core::LinAlg::cast_to_const_sparse_matrix_and_check_success(thermoscatradomain);

  auto& systemmatrix_block_thermo_scatra =
      systemmatrix_block->matrix(block_position_thermo().at(0), block_position_scatra().at(0));
  systemmatrix_block_thermo_scatra.un_complete();
  Core::LinAlg::matrix_add(
      *thermoscatradomain_sparse, false, 1.0, systemmatrix_block_thermo_scatra, 1.0);

  if (interface_meshtying()) assemble_thermo_scatra_interface(systemmatrix, thermoscatrainterface);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategySparse::assemble_thermo_scatra(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
    std::shared_ptr<const Core::LinAlg::SparseOperator> thermoscatradomain,
    std::shared_ptr<const Core::LinAlg::SparseOperator> thermoscatrainterface)
{
  auto systemmatrix_sparse = Core::LinAlg::cast_to_sparse_matrix_and_check_success(systemmatrix);
  auto thermoscatradomain_sparse =
      Core::LinAlg::cast_to_const_sparse_matrix_and_check_success(thermoscatradomain);

  // add scalar transport system matrix to global system matrix
  Core::LinAlg::matrix_add(*thermoscatradomain_sparse, false, 1.0, *systemmatrix_sparse, 1.0);

  if (interface_meshtying()) assemble_thermo_scatra_interface(systemmatrix, thermoscatrainterface);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBlockBlock::assemble_thermo_scatra_interface(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
    std::shared_ptr<const Core::LinAlg::SparseOperator> thermoscatrainterface)
{
  auto systemmatrix_block =
      Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(systemmatrix);

  auto thermoscatrainterface_block =
      Core::LinAlg::cast_to_const_block_sparse_matrix_base_and_check_success(thermoscatrainterface);

  Core::LinAlg::SparseMatrix masterflux(
      *all_maps()->block_map_thermo()->full_map(), 27, false, true);

  for (int i = 0; i < static_cast<int>(block_position_thermo().size()); ++i)
  {
    for (int j = 0; j < static_cast<int>(block_position_scatra().size()); ++j)
    {
      auto thermoscatrainterface_subblock = thermoscatrainterface_block->matrix(i, j);

      // assemble linearizations of slave side scatra fluxes w.r.t. slave and master side elch
      // into system matrix
      auto& systemmatrix_block_ithermo_jscatra =
          systemmatrix_block->matrix(block_position_thermo().at(i), block_position_scatra().at(j));
      Core::LinAlg::matrix_add(
          thermoscatrainterface_subblock, false, 1.0, systemmatrix_block_ithermo_jscatra, 1.0);

      // assemble linearizations of master side thermo fluxes w.r.t. slave and master side elch
      // into system matrix
      Coupling::Adapter::CouplingSlaveConverter thermo_converter(
          *meshtying_thermo()->coupling_adapter());

      Core::LinAlg::SparseMatrix slaveflux(
          *all_maps()->block_map_thermo()->full_map(), 27, false, true);

      ScaTra::MeshtyingStrategyS2I::extract_matrix_rows(thermoscatrainterface_subblock, slaveflux,
          *meshtying_thermo()->block_maps_slave().map(i));

      slaveflux.complete(
          *all_maps()->block_map_scatra()->full_map(), *all_maps()->block_map_thermo()->full_map());

      Coupling::Adapter::MatrixLogicalSplitAndTransform()(thermoscatrainterface_subblock,
          *meshtying_thermo()->coupling_adapter()->master_dof_map(),
          thermoscatrainterface_subblock.domain_map(), -1.0, &thermo_converter, nullptr, masterflux,
          true, true);
    }
  }

  masterflux.complete(
      *all_maps()->block_map_scatra()->full_map(), *all_maps()->block_map_thermo()->full_map());

  const auto blockmasterflux = Core::LinAlg::split_matrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
      masterflux, *all_maps()->block_map_scatra(), *all_maps()->block_map_thermo());

  blockmasterflux->complete();

  // assemble linearizations of slave side thermo fluxes w.r.t. slave and master side elch
  // into system matrix
  for (int i = 0; i < static_cast<int>(block_position_thermo().size()); ++i)
  {
    for (int j = 0; j < static_cast<int>(block_position_scatra().size()); ++j)
    {
      auto& systemmatrix_block_ithermo_jscatra =
          systemmatrix_block->matrix(block_position_thermo().at(i), block_position_scatra().at(j));

      Core::LinAlg::matrix_add(
          blockmasterflux->matrix(i, j), false, 1.0, systemmatrix_block_ithermo_jscatra, 1.0);
    }
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBlockSparse::assemble_thermo_scatra_interface(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
    std::shared_ptr<const Core::LinAlg::SparseOperator> thermoscatrainterface)
{
  auto systemmatrix_block =
      Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(systemmatrix);
  auto thermoscatrainterface_sparse =
      Core::LinAlg::cast_to_const_sparse_matrix_and_check_success(thermoscatrainterface);

  // assemble linearizations of slave side scatra fluxes w.r.t. slave and master side elch
  // into system matrix
  auto& systemmatrix_block_thermo_scatra =
      systemmatrix_block->matrix(block_position_thermo().at(0), block_position_scatra().at(0));
  Core::LinAlg::matrix_add(
      *thermoscatrainterface_sparse, false, 1.0, systemmatrix_block_thermo_scatra, 1.0);

  // assemble linearizations of master side thermo fluxes w.r.t. slave and master side elch
  // into system matrix
  Coupling::Adapter::CouplingSlaveConverter thermo_converter(
      *meshtying_thermo()->coupling_adapter());

  Coupling::Adapter::MatrixLogicalSplitAndTransform()(*thermoscatrainterface_sparse,
      *meshtying_thermo()->coupling_adapter()->master_dof_map(),
      Core::LinAlg::Map(thermoscatrainterface_sparse->domain_map()), -1.0, &thermo_converter,
      nullptr,
      systemmatrix_block->matrix(block_position_thermo().at(0), block_position_scatra().at(0)),
      true, true);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategySparse::assemble_thermo_scatra_interface(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
    std::shared_ptr<const Core::LinAlg::SparseOperator> thermoscatrainterface)
{
  auto systemmatrix_sparse = Core::LinAlg::cast_to_sparse_matrix_and_check_success(systemmatrix);
  auto thermoscatrainterface_sparse =
      Core::LinAlg::cast_to_const_sparse_matrix_and_check_success(thermoscatrainterface);

  // assemble linearizations of slave side scatra fluxes w.r.t. slave and master side elch
  // into system matrix
  Core::LinAlg::matrix_add(*thermoscatrainterface_sparse, false, 1.0, *systemmatrix_sparse, 1.0);

  // assemble linearizations of master side scatra fluxes w.r.t. slave and master side elch
  // into system matrix
  Coupling::Adapter::CouplingSlaveConverter thermo_converter(
      *meshtying_thermo()->coupling_adapter());

  Coupling::Adapter::MatrixLogicalSplitAndTransform()(*thermoscatrainterface_sparse,
      *meshtying_thermo()->coupling_adapter()->master_dof_map(),
      Core::LinAlg::Map(thermoscatrainterface_sparse->domain_map()), -1.0, &thermo_converter,
      nullptr, *systemmatrix_sparse, true, true);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBlockBlock::assemble_thermo_structure(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
    std::shared_ptr<const Core::LinAlg::SparseOperator> thermostructuredomain,
    std::shared_ptr<const Core::LinAlg::SparseOperator> thermostructureinterface)
{
  auto systemmatrix_block =
      Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(systemmatrix);
  auto thermostructuredomain_block =
      Core::LinAlg::cast_to_const_block_sparse_matrix_base_and_check_success(thermostructuredomain);

  // assemble blocks of scalar transport system matrix into global system matrix
  for (int iblock = 0; iblock < static_cast<int>(block_position_thermo().size()); ++iblock)
  {
    const auto thermostructuredomain_subblock = thermostructuredomain_block->matrix(iblock, 0);

    // add entire block or assemble slave side to master side
    if (interface_meshtying())
    {
      assemble_xxx_structure_meshtying(
          systemmatrix_block->matrix(block_position_thermo().at(iblock), position_structure()),
          thermostructuredomain_subblock);

      auto thermostructureinterface_block =
          Core::LinAlg::cast_to_const_block_sparse_matrix_base_and_check_success(
              thermostructureinterface);

      assemble_xxx_structure_meshtying(
          systemmatrix_block->matrix(block_position_thermo().at(iblock), position_structure()),
          thermostructureinterface_block->matrix(iblock, 0));
    }
    else
    {
      auto& systemmatrix_block_ithermo_struct =
          systemmatrix_block->matrix(block_position_thermo().at(iblock), position_structure());

      Core::LinAlg::matrix_add(
          thermostructuredomain_subblock, false, 1.0, systemmatrix_block_ithermo_struct, 1.0);
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBlockSparse::assemble_thermo_structure(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
    std::shared_ptr<const Core::LinAlg::SparseOperator> thermostructuredomain,
    std::shared_ptr<const Core::LinAlg::SparseOperator> thermostructureinterface)
{
  auto systemmatrix_block =
      Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(systemmatrix);
  auto thermostructuredomain_sparse =
      Core::LinAlg::cast_to_const_sparse_matrix_and_check_success(thermostructuredomain);

  // add entire block or assemble slave side to master side
  if (interface_meshtying())
  {
    assemble_xxx_structure_meshtying(
        systemmatrix_block->matrix(block_position_thermo().at(0), position_structure()),
        *thermostructuredomain_sparse);

    auto thermostructureinterface_sparse =
        Core::LinAlg::cast_to_const_sparse_matrix_and_check_success(thermostructureinterface);

    assemble_xxx_structure_meshtying(
        systemmatrix_block->matrix(block_position_thermo().at(0), position_structure()),
        *thermostructureinterface_sparse);
  }
  else
  {
    auto& sytemmatrix_block_thermo_structure =
        systemmatrix_block->matrix(block_position_thermo().at(0), position_structure());

    Core::LinAlg::matrix_add(
        *thermostructuredomain_sparse, false, 1.0, sytemmatrix_block_thermo_structure, 1.0);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategySparse::assemble_thermo_structure(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
    std::shared_ptr<const Core::LinAlg::SparseOperator> thermostructuredomain,
    std::shared_ptr<const Core::LinAlg::SparseOperator> thermostructureinterface)
{
  auto systemmatrix_sparse = Core::LinAlg::cast_to_sparse_matrix_and_check_success(systemmatrix);
  auto thermostructuredomain_sparse =
      Core::LinAlg::cast_to_const_sparse_matrix_and_check_success(thermostructuredomain);

  if (interface_meshtying())
  {
    assemble_xxx_structure_meshtying(*systemmatrix_sparse, *thermostructuredomain_sparse);

    auto thermostructureinterface_sparse =
        Core::LinAlg::cast_to_const_sparse_matrix_and_check_success(thermostructureinterface);

    assemble_xxx_structure_meshtying(*systemmatrix_sparse, *thermostructureinterface_sparse);
  }
  else
    Core::LinAlg::matrix_add(*thermostructuredomain_sparse, false, 1.0, *systemmatrix_sparse, 1.0);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBlockBlock::assemble_structure_thermo(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
    std::shared_ptr<const Core::LinAlg::SparseOperator> structurethermodomain)
{
  auto systemmatrix_block =
      Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(systemmatrix);
  auto structurethermodomain_block =
      Core::LinAlg::cast_to_const_block_sparse_matrix_base_and_check_success(structurethermodomain);

  // assemble blocks of scalar transport system matrix into global system matrix
  for (int iblock = 0; iblock < static_cast<int>(block_position_thermo().size()); ++iblock)
  {
    const auto structurethermodomain_subblock = structurethermodomain_block->matrix(0, iblock);

    // add entire block or assemble slave side to master side
    if (interface_meshtying())
    {
      assemble_structure_xxx_meshtying(
          systemmatrix_block->matrix(position_structure(), block_position_thermo().at(iblock)),
          structurethermodomain_subblock);
    }
    else
    {
      auto& systemmatrix_block_struct_ithermo =
          systemmatrix_block->matrix(position_structure(), block_position_thermo().at(iblock));

      Core::LinAlg::matrix_add(
          structurethermodomain_subblock, false, 1.0, systemmatrix_block_struct_ithermo, 1.0);
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBlockSparse::assemble_structure_thermo(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
    std::shared_ptr<const Core::LinAlg::SparseOperator> structurethermodomain)
{
  auto systemmatrix_block =
      Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(systemmatrix);
  auto structurethermodomain_sparse =
      Core::LinAlg::cast_to_const_sparse_matrix_and_check_success(structurethermodomain);

  // add entire block or assemble slave side to master side
  if (interface_meshtying())
  {
    assemble_structure_xxx_meshtying(
        systemmatrix_block->matrix(position_structure(), block_position_thermo().at(0)),
        *structurethermodomain_sparse);
  }
  else
  {
    auto& systemmatrix_block_struct_thermo =
        systemmatrix_block->matrix(position_structure(), block_position_thermo().at(0));

    Core::LinAlg::matrix_add(
        *structurethermodomain_sparse, false, 1.0, systemmatrix_block_struct_thermo, 1.0);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategySparse::assemble_structure_thermo(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
    std::shared_ptr<const Core::LinAlg::SparseOperator> structurethermodomain)
{
  auto systemmatrix_sparse = Core::LinAlg::cast_to_sparse_matrix_and_check_success(systemmatrix);
  auto structurethermodomain_sparse =
      Core::LinAlg::cast_to_const_sparse_matrix_and_check_success(structurethermodomain);

  // add entire block or assemble slave side to master side
  if (interface_meshtying())
    assemble_structure_xxx_meshtying(*systemmatrix_sparse, *structurethermodomain_sparse);
  else
    Core::LinAlg::matrix_add(*structurethermodomain_sparse, false, 1.0, *systemmatrix_sparse, 1.0);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBlockBlock::apply_meshtying_system_matrix(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix)
{
  if (interface_meshtying())
  {
    auto systemmatrix_block =
        Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(systemmatrix);

    apply_meshtying_sys_mat(systemmatrix_block->matrix(position_structure(), position_structure()));
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBlockSparse::apply_meshtying_system_matrix(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix)
{
  if (interface_meshtying())
  {
    // cast systemmatrix
    auto systemmatrix_block =
        Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(systemmatrix);

    apply_meshtying_sys_mat(systemmatrix_block->matrix(position_structure(), position_structure()));
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategySparse::apply_meshtying_system_matrix(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix)
{
  if (interface_meshtying())
  {
    // cast systemmatrix
    auto systemmatrix_sparse = Core::LinAlg::cast_to_sparse_matrix_and_check_success(systemmatrix);

    apply_meshtying_sys_mat(*systemmatrix_sparse);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBase::apply_meshtying_sys_mat(
    Core::LinAlg::SparseMatrix& systemmatrix_structure)
{
  // map for slave side structural degrees of freedom
  auto slavemaps = ssti_structure_meshtying()->full_slave_side_map();

  // subject slave-side rows of structural system matrix to pseudo Dirichlet conditions to
  // finalize structural meshtying
  const double one(1.0);
  for (int doflid_slave = 0; doflid_slave < slavemaps->num_my_elements(); ++doflid_slave)
  {
    // extract global ID of current slave-side row
    const int dofgid_slave = slavemaps->gid(doflid_slave);
    if (dofgid_slave < 0) FOUR_C_THROW("Local ID not found!");

    // apply pseudo Dirichlet conditions to filled matrix, i.e., to local row and column
    // indices
    if (systemmatrix_structure.filled())
    {
      const int rowlid_slave = systemmatrix_structure.row_map().lid(dofgid_slave);
      if (rowlid_slave < 0) FOUR_C_THROW("Global ID not found!");
      systemmatrix_structure.replace_my_values(rowlid_slave, 1, &one, &rowlid_slave);
    }

    // apply pseudo Dirichlet conditions to unfilled matrix, i.e., to global row and
    // column indices
    else
    {
      systemmatrix_structure.insert_global_values(dofgid_slave, 1, &one, &dofgid_slave);
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBlock::apply_structural_dbc_system_matrix(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix)
{
  // locsys manager of structure
  const auto& locsysmanager_structure = structure_field()->locsys_manager();

  // map of structural Dirichlet BCs
  const auto dbcmap_structure = structure_field()->get_dbc_map_extractor()->cond_map();

  if (locsysmanager_structure == nullptr)
    systemmatrix->apply_dirichlet(*dbcmap_structure);
  else
  {
    auto systemmatrix_block =
        Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(systemmatrix);

    // apply structural Dirichlet conditions
    for (int iblock = 0; iblock < systemmatrix_block->cols(); ++iblock)
    {
      locsysmanager_structure->rotate_global_to_local(Core::Utils::shared_ptr_from_ref(
          systemmatrix_block->matrix(position_structure(), iblock)));
      systemmatrix_block->matrix(position_structure(), iblock)
          .apply_dirichlet_with_trafo(
              *locsysmanager_structure->trafo(), *dbcmap_structure, iblock == position_structure());
      locsysmanager_structure->rotate_local_to_global(
          systemmatrix_block->matrix(position_structure(), iblock));
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategySparse::apply_structural_dbc_system_matrix(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix)
{
  // locsys manager of structure
  const auto& locsysmanager_structure = structure_field()->locsys_manager();

  // map of structural Dirichlet BCs
  const auto& dbcmap_structure = structure_field()->get_dbc_map_extractor()->cond_map();

  // structural dof row map
  const auto& dofrowmap_structure = structure_field()->dof_row_map();

  if (locsysmanager_structure == nullptr)
    systemmatrix->apply_dirichlet(*dbcmap_structure);
  else
  {
    auto systemmatrix_sparse = Core::LinAlg::cast_to_sparse_matrix_and_check_success(systemmatrix);

    // extract structural rows of global system matrix
    auto systemmatrix_structure =
        std::make_shared<Core::LinAlg::SparseMatrix>(*dofrowmap_structure, 27, false, true);
    Coupling::Adapter::MatrixLogicalSplitAndTransform()(*systemmatrix_sparse, *dofrowmap_structure,
        Core::LinAlg::Map(systemmatrix->domain_map()), 1.0, nullptr, nullptr,
        *systemmatrix_structure);
    systemmatrix_structure->complete(systemmatrix->domain_map(), *dofrowmap_structure);

    // apply structural Dirichlet conditions
    locsysmanager_structure->rotate_global_to_local(systemmatrix_structure);
    systemmatrix_structure->apply_dirichlet_with_trafo(
        *locsysmanager_structure->trafo(), *dbcmap_structure);
    locsysmanager_structure->rotate_local_to_global(*systemmatrix_structure);

    // assemble structural rows of global system matrix back into global system matrix
    Core::LinAlg::matrix_put(
        *systemmatrix_structure, 1.0, dofrowmap_structure, *systemmatrix_sparse);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBase::assemble_rhs(std::shared_ptr<Core::LinAlg::Vector<double>> RHS,
    std::shared_ptr<const Core::LinAlg::Vector<double>> RHSscatra,
    const Core::LinAlg::Vector<double>& RHSstructure,
    std::shared_ptr<const Core::LinAlg::Vector<double>> RHSthermo)
{
  // zero out RHS
  RHS->put_scalar(0.0);

  // assemble scalar transport right-hand side vector into monolithic right-hand side vector
  all_maps()->maps_sub_problems()->insert_vector(
      *RHSscatra, ssti_mono_->get_problem_position(Subproblem::scalar_transport), *RHS);
  all_maps()->maps_sub_problems()->insert_vector(
      *RHSthermo, ssti_mono_->get_problem_position(Subproblem::thermo), *RHS);

  if (interface_meshtying())
  {
    // perform structural meshtying before assembling structural right-hand side vector into
    // monolithic right-hand side vector

    // make copy of structural right-hand side vector
    Core::LinAlg::Vector<double> residual_structure(RHSstructure);

    auto rhs_structure_master =
        std::make_shared<Core::LinAlg::Vector<double>>(*structure_field()->dof_row_map(), true);

    for (const auto& meshtying : ssti_structure_meshtying()->mesh_tying_handlers())
    {
      auto coupling_adapter = meshtying->slave_master_coupling();
      auto coupling_map_extractor = meshtying->slave_master_extractor();

      // transform slave-side part of structure right-hand side vector to master side
      const auto rhs_structure_only_slave_dofs =
          coupling_map_extractor->extract_vector(residual_structure, 1);

      const auto rhs_structure_only_master_dofs =
          coupling_adapter->slave_to_master(*rhs_structure_only_slave_dofs);

      coupling_map_extractor->add_vector(*rhs_structure_only_master_dofs, 2, *rhs_structure_master);

      // zero out slave-side part of structure right-hand side vector
      coupling_map_extractor->put_scalar(residual_structure, 1, 0.0);
    }

    // locsys manager of structure
    const auto& locsysmanager_structure = ssti_mono_->structure_field()->locsys_manager();

    // apply pseudo Dirichlet conditions to transformed slave-side part of structural right-hand
    // side vector
    Core::LinAlg::Vector<double> zeros(residual_structure.get_map());

    if (locsysmanager_structure != nullptr)
      locsysmanager_structure->rotate_global_to_local(*rhs_structure_master);
    Core::LinAlg::apply_dirichlet_to_system(*rhs_structure_master, zeros,
        *ssti_mono_->structure_field()->get_dbc_map_extractor()->cond_map());
    if (locsysmanager_structure != nullptr)
      locsysmanager_structure->rotate_local_to_global(*rhs_structure_master);

    // assemble master-side part of structure right-hand side vector
    residual_structure.update(1.0, *rhs_structure_master, 1.0);

    // assemble final structural right-hand side vector into monolithic right-hand side vector
    all_maps()->maps_sub_problems()->add_vector(
        residual_structure, ssti_mono_->get_problem_position(Subproblem::structure), *RHS, -1.0);
  }
  else
  {
    all_maps()->maps_sub_problems()->add_vector(
        RHSstructure, ssti_mono_->get_problem_position(Subproblem::structure), *RHS, -1.0);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<SSTI::AssembleStrategyBase> SSTI::build_assemble_strategy(
    std::shared_ptr<const SSTI::SSTIMono> ssti_mono, Core::LinAlg::MatrixType matrixtype_ssti,
    Core::LinAlg::MatrixType matrixtype_scatra)
{
  std::shared_ptr<SSTI::AssembleStrategyBase> assemblestrategy = nullptr;

  switch (matrixtype_ssti)
  {
    case Core::LinAlg::MatrixType::block_field:
    {
      switch (matrixtype_scatra)
      {
        case Core::LinAlg::MatrixType::block_condition:
        {
          assemblestrategy = std::make_shared<SSTI::AssembleStrategyBlockBlock>(ssti_mono);
          break;
        }
        case Core::LinAlg::MatrixType::sparse:
        {
          assemblestrategy = std::make_shared<SSTI::AssembleStrategyBlockSparse>(ssti_mono);
          break;
        }
        default:
        {
          FOUR_C_THROW("unknown matrix type");
          break;
        }
      }
      break;
    }
    case Core::LinAlg::MatrixType::sparse:
    {
      assemblestrategy = std::make_shared<SSTI::AssembleStrategySparse>(ssti_mono);
      break;
    }
    default:
    {
      FOUR_C_THROW("unknown matrix type");
      break;
    }
  }

  return assemblestrategy;
}

FOUR_C_NAMESPACE_CLOSE

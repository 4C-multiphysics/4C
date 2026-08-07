// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_IO_MESH_TEST_UTILS_TEST_HPP
#define FOUR_C_IO_MESH_TEST_UTILS_TEST_HPP

#include "4C_fem_general_cell_type.hpp"
#include "4C_io_mesh.hpp"
#include "4C_utils_exceptions.hpp"

namespace FourC::Core::IO::MeshInput::Test
{
  template <unsigned dim>
  std::vector<CellBlockReference<dim, true>> cell_blocks_by_group_id(
      RawMesh<dim>& mesh, const int group_id)
  {
    const auto lookup = Mesh<dim>::create_view(mesh).create_cell_block_lookup_tables();
    FOUR_C_ASSERT_ALWAYS(
        lookup.by_id.contains(group_id), "Expected a cell block with group id {}.", group_id);
    return lookup.by_id.at(group_id);
  }

  template <unsigned dim>
  std::vector<CellBlockReference<dim, true>> cell_blocks_by_group_name(
      RawMesh<dim>& mesh, const std::string& group_name)
  {
    const auto lookup = Mesh<dim>::create_view(mesh).create_cell_block_lookup_tables();
    FOUR_C_ASSERT_ALWAYS(lookup.by_name.contains(group_name),
        "Expected a cell block with group name '{}'.", group_name);
    return lookup.by_name.at(group_name);
  }

  template <unsigned dim>
  const CellBlock<dim>& unique_cell_block_by_group_id(RawMesh<dim>& mesh, const int group_id)
  {
    const auto cell_blocks = cell_blocks_by_group_id(mesh, group_id);
    FOUR_C_ASSERT_ALWAYS(cell_blocks.size() == 1,
        "Expected exactly one cell block with group id {}, but found {}.", group_id,
        cell_blocks.size());
    return cell_blocks.front().get();
  }

  template <unsigned dim>
  const CellBlock<dim>& unique_cell_block_by_group_id_and_cell_type(
      RawMesh<dim>& mesh, const int group_id, const FE::CellType cell_type)
  {
    const CellBlock<dim>* matching_cell_block = nullptr;
    for (const auto& cell_block : cell_blocks_by_group_id(mesh, group_id))
    {
      if (cell_block.cell_type() != cell_type) continue;

      FOUR_C_ASSERT_ALWAYS(matching_cell_block == nullptr,
          "Expected exactly one cell block with group id {} and cell type {}, but found more.",
          group_id, FE::cell_type_to_string(cell_type));
      matching_cell_block = &cell_block.get();
    }

    FOUR_C_ASSERT_ALWAYS(matching_cell_block != nullptr,
        "Expected a cell block with group id {} and cell type {}.", group_id,
        FE::cell_type_to_string(cell_type));
    return *matching_cell_block;
  }
}  // namespace FourC::Core::IO::MeshInput::Test

#endif

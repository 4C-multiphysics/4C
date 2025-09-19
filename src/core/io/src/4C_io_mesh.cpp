// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_io_mesh.hpp"

#include "4C_utils_enum.hpp"
#include "4C_utils_std23_unreachable.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>



FOUR_C_NAMESPACE_OPEN

std::string Core::IO::MeshInput::describe(VerbosityLevel level)
{
  switch (level)
  {
    case VerbosityLevel::none:
      return "no output";
    case VerbosityLevel::summary:
      return "output of summary for blocks and sets";
    case VerbosityLevel::detailed_summary:
      return "output of summary for each block and set";
    case VerbosityLevel::detailed:
      return "detailed output for each block and set";
    case VerbosityLevel::full:
      return "detailed output, even for nodes and element connectivities";
  }
  std23::unreachable();
}

template <unsigned dim>
void Core::IO::MeshInput::assert_valid(const Mesh<dim>& mesh)
{
  FOUR_C_ASSERT_ALWAYS(!mesh.points.empty(), "The mesh has no points.");

  if (!mesh.point_data.empty())
  {
    for (const auto& [key, data] : mesh.point_data)
    {
      // Avoid capturing structured binding for clang OpenMP
      const std::string_view field_name = key;
      std::visit(
          [&](const auto& vec)
          {
            FOUR_C_ASSERT_ALWAYS(vec.size() == mesh.points.size(),
                "Point data field '{}' has {} entries, but the mesh has {} points.", field_name,
                vec.size(), mesh.points.size());
          },
          data);
    }
  }

  FOUR_C_ASSERT_ALWAYS(!mesh.cell_blocks.empty(), "The mesh has no cell blocks.");

  for (const auto& [id, cell_block] : mesh.cell_blocks)
  {
    // Avoid capturing structured binding for clang OpenMP
    const CellBlock<dim>& cell_block_ref = cell_block;
    FOUR_C_ASSERT_ALWAYS(cell_block.size() > 0, "Cell block {} has no cells.", id);

    if (cell_block.external_ids_.has_value())
    {
      FOUR_C_ASSERT_ALWAYS(cell_block.external_ids_->size() == cell_block.size(),
          "Cell block {} has {} cells but {} external IDs.", id, cell_block.size(),
          cell_block.external_ids_->size());
    }

    for (const auto& connectivity : cell_block.cells())
    {
      for (const auto node_id : connectivity)
      {
        FOUR_C_ASSERT_ALWAYS(node_id >= 0 && static_cast<std::size_t>(node_id) < mesh.points.size(),
            "Cell block {} has a cell with invalid node ID {}.", id, node_id);
      }
    }

    if (!cell_block.cell_data.empty())
    {
      for (const auto& [key, data] : cell_block.cell_data)
      {
        // Avoid capturing structured binding for clang OpenMP
        const std::string_view field_name = key;
        std::visit(
            [&](const auto& vec)
            {
              FOUR_C_ASSERT_ALWAYS(vec.size() == cell_block_ref.size(),
                  "Cell data field '{}' has {} entries, but the block has {} cells.", field_name,
                  vec.size(), cell_block_ref.size());
            },
            data);
      }
    }
  }

  for (const auto& [id, point_set] : mesh.point_sets)
  {
    FOUR_C_ASSERT_ALWAYS(!point_set.point_ids.empty(), "Point set {} has no points.", id);

    for (const auto point_id : point_set.point_ids)
    {
      FOUR_C_ASSERT_ALWAYS(point_id >= 0 && static_cast<std::size_t>(point_id) < mesh.points.size(),
          "Point set {} has invalid point ID {}.", id, point_id);
    }
  }
}


template <unsigned dim>
void Core::IO::MeshInput::print(const Mesh<dim>& mesh, std::ostream& os, VerbosityLevel verbose)
{
  if (verbose >= VerbosityLevel::summary)
  {
    auto num_elements = std::accumulate(mesh.cell_blocks.begin(), mesh.cell_blocks.end(), 0,
        [](int sum, const auto& block) { return sum + block.second.size(); });

    os << "Mesh consists of " << mesh.points.size() << " points and " << num_elements
       << " cells organized in " << mesh.cell_blocks.size() << " cell-blocks and "
       << mesh.point_sets.size() << " point-sets.\n\n";
  }
  if (verbose >= VerbosityLevel::detailed_summary)
  {
    if (verbose == VerbosityLevel::full)
    {
      std::size_t i = 0;
      for (const auto& point : mesh.points)
      {
        if (mesh.external_ids.has_value())
        {
          os << "  " << mesh.external_ids->at(i) << ": [";
        }
        else
        {
          os << "  [";
        }
        for (const auto& coord : point)
        {
          os << std::format("{:10.6g},", coord);
        }
        os << "]\n";
        i++;
      }
      os << "\n";
    }
    os << "cell-blocks:\n";
    for (const auto& [id, cell_block] : mesh.cell_blocks)
    {
      os << "  cell-block " << id;
      if (cell_block.name.has_value()) os << " (" << *cell_block.name << ")";
      os << ": ";
      print(cell_block, os, verbose);
    }
    os << "\n";

    os << "point-sets:\n";
    for (const auto& [ps_id, ps] : mesh.point_sets)
    {
      os << "  point-set " << ps_id;
      if (ps.name.has_value()) os << " (" << *ps.name << ")";
      os << ": ";
      print(ps, os, verbose);
    }
  }
}

template <unsigned dim>
void Core::IO::MeshInput::print(
    const CellBlock<dim>& block, std::ostream& os, VerbosityLevel verbose)
{
  using EnumTools::operator<<;

  os << block.size() << " cells of type " << block.cell_type << "\n";

  if (verbose == VerbosityLevel::full)
  {
    std::size_t i = 0;
    for (const auto& connectivity : block.cells())
    {
      if (block.external_ids_.has_value())
      {
        const auto external_id = block.external_ids_->at(i);
        os << "    " << external_id << ": [";
      }
      else
      {
        os << "    [";
      }
      for (const auto& id : connectivity) os << id << ", ";
      os << "]\n";
      i++;
    }
    os << "\n";
  }
}

void Core::IO::MeshInput::print(const PointSet& point_set, std::ostream& os, VerbosityLevel verbose)
{
  os << point_set.point_ids.size() << " points";
  if (verbose == VerbosityLevel::full && point_set.point_ids.size() > 0)
  {
    int nline = 0;
    int nodelength = std::to_string(*std::ranges::max_element(point_set.point_ids)).size();
    for (const int id : point_set.point_ids)
    {
      if (nline++ % 12 == 0) os << "\n  ";
      os << " " << std::setw(nodelength) << id << ",";
    }
    if (nline % 12 != 0) os << "\n";
  }

  os << "\n";
}

template void Core::IO::MeshInput::assert_valid(const Mesh<3>& mesh);
template void Core::IO::MeshInput::print(
    const Mesh<3>& mesh, std::ostream& os, VerbosityLevel verbose);
template void Core::IO::MeshInput::print(
    const CellBlock<3>& block, std::ostream& os, VerbosityLevel verbose);

FOUR_C_NAMESPACE_CLOSE
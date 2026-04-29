// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_io_mesh.hpp"

#include "4C_utils_enum.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_std23_unreachable.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <set>



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
Core::IO::MeshInput::Mesh<dim>::Mesh() : raw_mesh_(Utils::make_owner<RawMesh<dim>>())
{
}


template <unsigned dim>
Core::IO::MeshInput::Mesh<dim>::Mesh(RawMesh<dim>&& raw_mesh)
    : raw_mesh_(Utils::make_owner<RawMesh<dim>>(std::move(raw_mesh)))
{
  default_fill_indices();
}


template <unsigned dim>
Core::IO::MeshInput::Mesh<dim> Core::IO::MeshInput::Mesh<dim>::create_view(RawMesh<dim>& raw_mesh)
{
  Mesh mesh;
  mesh.raw_mesh_ = Utils::make_view(&raw_mesh);
  mesh.default_fill_indices();
  return mesh;
}


template <unsigned dim>
bool Core::IO::MeshInput::Mesh<dim>::has_point_data(const std::string& field_name) const
{
  return raw_mesh_->point_data.find(field_name) != raw_mesh_->point_data.end();
}


template <unsigned dim>
Core::IO::MeshInput::CellBlockLookupTables<dim>
Core::IO::MeshInput::Mesh<dim>::create_cell_block_lookup_tables() const
{
  CellBlockLookupTables<dim> lookup_tables;

  for (const auto& cell_block : cell_blocks())
  {
    const auto cell_block_id = cell_block.id();

    bool inserted = lookup_tables.by_id.emplace(cell_block_id, cell_block).second;

    FOUR_C_ASSERT_ALWAYS(inserted,
        "Duplicate cell block ID {} found in the mesh. Cell block IDs must be unique.",
        cell_block_id);

    if (cell_block.name().has_value())
    {
      lookup_tables.by_name[cell_block.name().value()].emplace_back(cell_block);
    }
  }

  return lookup_tables;
}


template <unsigned dim>
Core::IO::MeshInput::Mesh<dim> Core::IO::MeshInput::Mesh<dim>::filter_by_cell_block_ids(
    const std::vector<ExternalIdType>& cell_block_ids) const
{
  Mesh filtered_mesh;
  filtered_mesh.raw_mesh_ = Utils::make_view(raw_mesh_.get());
  filtered_mesh.cell_blocks_ids_filter_ = cell_block_ids;

  // Check that the given cell-block IDs exist in the mesh, and collect all point IDs that are used
  // by the cell-blocks
  std::set<size_t> point_ids_set;
  for (const auto id : cell_block_ids)
  {
    FOUR_C_ASSERT_ALWAYS(raw_mesh_->cell_blocks.contains(id),
        "You are trying to filter the mesh by cell-block ID {}, but the mesh does not contain "
        "a cell-block with this ID.",
        id);

    for (const auto& cell : raw_mesh_->cell_blocks.at(id).cells())
    {
      point_ids_set.insert(cell.begin(), cell.end());
    }
  }
  filtered_mesh.point_ids_filter_.assign(point_ids_set.begin(), point_ids_set.end());

  // Filter the point sets depending on whether they contain points that are still in the filtered
  // mesh
  for (const auto& [id, data] : raw_mesh_->point_sets)
  {
    // Either all points of the point set are in the filtered mesh, or none
    bool has_relevant_points = false;
    bool has_irrelevant_points = false;

    for (const auto point_id : data.point_ids)
    {
      if (point_ids_set.contains(point_id))
        has_relevant_points = true;
      else
        has_irrelevant_points = true;
    }

    if (has_relevant_points && has_irrelevant_points)
    {
      FOUR_C_THROW(
          "While filtering the mesh by cell-block IDs, point-set {} contains some points "
          "that are used by the remaining cell-blocks, and some that are not. Point-sets "
          "must either contain all or none of the remaining points.",
          id);
    }

    if (has_relevant_points) filtered_mesh.point_sets_ids_filter_.emplace_back(id);
  }

  return filtered_mesh;
}


template <unsigned dim>
void Core::IO::MeshInput::Mesh<dim>::default_fill_indices()
{
  FOUR_C_ASSERT(raw_mesh_.get() != nullptr, "Internal error: RawMesh pointer must not be null.");

  cell_blocks_ids_filter_.resize(raw_mesh_->cell_blocks.size());
  std::ranges::copy(raw_mesh_->cell_blocks | std::views::keys, cell_blocks_ids_filter_.begin());

  point_sets_ids_filter_.resize(raw_mesh_->point_sets.size());
  std::ranges::copy(raw_mesh_->point_sets | std::views::keys, point_sets_ids_filter_.begin());

  point_ids_filter_.resize(raw_mesh_->points.size());
  std::iota(point_ids_filter_.begin(), point_ids_filter_.end(), 0);
}


template <unsigned dim>
void Core::IO::MeshInput::assert_valid(const RawMesh<dim>& mesh)
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
    const auto cell_blocks = mesh.cell_blocks();
    auto num_elements = std::accumulate(cell_blocks.begin(), cell_blocks.end(), 0,
        [](int sum, const auto& block) { return sum + block.size(); });

    os << "Mesh consists of " << mesh.points().size() << " points and " << num_elements
       << " cells organized in " << mesh.cell_blocks().size() << " cell-blocks and "
       << mesh.point_sets().size() << " point-sets.\n\n";
  }
  if (verbose >= VerbosityLevel::detailed_summary)
  {
    if (verbose == VerbosityLevel::full)
    {
      os << std::format(" {:^23} |{:^33}|\n", "node ID", "coordinates");
      os << std::format(
          "{:^12}|{:^12}|{:>10},{:>10},{:>10} |\n", "external", "internal(+1)", "X", "Y", "Z");
      os << std::format("{:-<12}|{:-<12}|{:-<33}|\n", "", "", "");
      for (const auto& point : mesh.points_with_data())
      {
        int internal_id = point.id() + 1;
        if (point.external_id() != invalid_external_id)
        {
          os << std::format(" {:10} | {:10} |", point.external_id(), internal_id);
        }
        else
        {
          os << std::format(" {:10} | {:10} |", " ", internal_id);
        }
        std::string coord_str;
        for (const auto& coord : point.coordinate())
        {
          if (!coord_str.empty()) coord_str += ",";
          coord_str += std::format("{:10.6g}", coord);
        }
        os << coord_str << " |\n";
      }
    }
    os << "cell-blocks:\n";
    for (const auto& cell_block : mesh.cell_blocks())
    {
      os << "  cell-block " << cell_block.id();
      if (cell_block.name().has_value()) os << " (" << *cell_block.name() << ")";
      os << ": ";
      print(cell_block.get(), os, verbose);
    }
    os << "\n";

    os << "point-sets (internal node numbers +1):\n";
    for (const auto& [ps_id, ps] : mesh.point_sets())
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
    os << "    (connectivity contains internal node numbers +1)\n";
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
      std::string connectivity_str;
      for (const auto& id : connectivity)
      {
        if (!connectivity_str.empty()) connectivity_str += ", ";
        connectivity_str += std::to_string(id + 1);
      }
      os << connectivity_str << "]\n";
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
      os << " " << std::setw(nodelength) << id + 1 << ",";
    }
    if (nline % 12 != 0) os << "\n";
  }

  os << "\n";
}

template void Core::IO::MeshInput::assert_valid(const RawMesh<3>& mesh);
template void Core::IO::MeshInput::print(
    const Mesh<3>& mesh, std::ostream& os, VerbosityLevel verbose);
template void Core::IO::MeshInput::print(
    const CellBlock<3>& block, std::ostream& os, VerbosityLevel verbose);

template class Core::IO::MeshInput::Mesh<3>;

FOUR_C_NAMESPACE_CLOSE
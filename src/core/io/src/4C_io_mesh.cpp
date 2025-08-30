// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_io_mesh.hpp"

#include "4C_utils_enum.hpp"
#include "4C_utils_std23_unreachable.hpp"



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

void Core::IO::MeshInput::print(const Mesh& mesh, std::ostream& os, VerbosityLevel verbose)
{
  if (verbose >= VerbosityLevel::summary)
  {
    auto num_elements = std::accumulate(mesh.cell_blocks.begin(), mesh.cell_blocks.end(), 0,
        [](int sum, const auto& block) { return sum + block.second.cell_connectivities.size(); });

    os << "  consists of " << mesh.points.size() << " points and " << num_elements << " cells; \n"
       << "  organized in " << mesh.cell_blocks.size() << " cell-blocks, ";
    os << mesh.point_sets.size() << " point-sets and ";
    os << mesh.side_sets.size() << " side-sets. ";
    os << std::endl << std::endl;
  }
  if (verbose >= VerbosityLevel::detailed_summary)
  {
    if (verbose == VerbosityLevel::full)
    {
      for (const auto& [point_id, coords] : mesh.points)
      {
        os << "Point " << point_id << ": ";
        for (const auto& coord : coords)
        {
          os << std::format("{:10.6g},", coord);
        }
        os << "\n";
      }
      os << std::endl;
    }
    os << "cell-blocks:\n";
    os << "------------\n";
    for (const auto& [id, connectivity] : mesh.cell_blocks)
    {
      os << "cell-block " << id << ": ";
      print(connectivity, os, verbose);
    }

    os << "\nside-sets:\n";
    os << "-----------\n";
    for (const auto& [ss_id, ss] : mesh.side_sets)
    {
      os << "side-set " << ss_id << ": ";
      print(ss, os, verbose);
    }

    os << "\nline-sets:\n";
    os << "----------\n";
    for (const auto& [ls_id, ls] : mesh.line_sets)
    {
      os << "line-set " << ls_id << ": ";
      print(ls, os, verbose);
    }

    os << "\npoint-sets:\n";
    os << "-------------\n";
    for (const auto& [ps_id, ps] : mesh.point_sets)
    {
      os << "point-set " << ps_id << ": ";
      print(ps, os, verbose);
    }
  }
}

void Core::IO::MeshInput::print(const CellBlock& block, std::ostream& os, VerbosityLevel verbose)
{
  using EnumTools::operator<<;

  os << block.cell_connectivities.size() << " Cells of shape " << block.cell_type;
  os << std::endl;

  if (verbose == VerbosityLevel::detailed)
  {
    int nline = 0;
    os << "cells:";
    for (const auto& [id, connectivity] : block.cell_connectivities)
    {
      if (nline++ % 12 == 0) os << "\n  ";
      os << id << ",";
    }
    if (nline % 12 != 0) os << std::endl;
    os << std::endl;
  }
  else if (verbose == VerbosityLevel::full)
  {
    for (const auto& [id, conn] : block.cell_connectivities)
    {
      os << "  cell " << id << ": ";
      for (const auto& point_id : conn)
      {
        os << point_id << ",";
      }
      os << std::endl;
    }
    os << std::endl;
  }
}

void Core::IO::MeshInput::print(const SideSet& side_set, std::ostream& os, VerbosityLevel verbose)
{
  os << " contains " << side_set.sides.size() << " sides";
  if (verbose == VerbosityLevel::full)
  {
    os << ": ";
    int nline = 0;
    for (const auto& [id, _] : side_set.sides)
    {
      if (nline++ % 12 == 0) os << "\n  ";
      os << std::setw(5) << id << ",";
    }
    if (nline % 12 != 0) os << std::endl;
  }
  else
    os << "\n";
}

void Core::IO::MeshInput::print(const LineSet& line_set, std::ostream& os, VerbosityLevel verbose)
{
  os << " contains " << line_set.lines.size() << " lines";
  if (verbose == VerbosityLevel::full)
  {
    os << ": ";
    int nline = 0;
    for (const auto& [id, _] : line_set.lines)
    {
      if (nline++ % 12 == 0) os << "\n  ";
      os << std::setw(5) << id << ",";
    }
    if (nline % 12 != 0) os << std::endl;
  }
  else
    os << "\n";
}

void Core::IO::MeshInput::print(const PointSet& point_set, std::ostream& os, VerbosityLevel verbose)
{
  os << " contains " << point_set.point_ids.size() << " points";
  if (verbose == VerbosityLevel::full)
  {
    os << ": ";
    int nline = 0;
    int nodelength =
        std::to_string(*std::max_element(point_set.point_ids.begin(), point_set.point_ids.end()))
            .size();
    for (const int id : point_set.point_ids)
    {
      if (nline++ % 12 == 0) os << "\n  ";
      os << std::setw(nodelength) << id << ",";
    }
    if (nline % 12 != 0) os << std::endl;
  }
  else
    os << "\n";
}

FOUR_C_NAMESPACE_CLOSE
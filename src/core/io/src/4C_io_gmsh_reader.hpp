// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_IO_GMSH_READER_HPP
#define FOUR_C_IO_GMSH_READER_HPP

#include "4C_config.hpp"

#include "4C_io_mesh.hpp"

#include <filesystem>

FOUR_C_NAMESPACE_OPEN

namespace Core::IO::Gmsh
{
  /**
   * @brief Read a Gmsh `.msh` file into the 4C intermediate mesh representation.
   *
   * The mesh is imported using Gmsh physical groups to define element blocks
   * and point sets. Each physical group is identified by a `(dim, tag)` pair. It is asserted that
   * physical tags are unique across dimensions.
   *
   * @subsection element_blocks Element blocks
   * For each physical group, an element block is created using the corresponding
   * (unique) tag as a key. If a physical group contains multiple element types, a warning is issued
   * and the group is excluded from the element blocks.
   *
   * @subsection point_sets Point sets
   * All physical groups (independent of their dimension) are converted into point
   * sets using the (unique) tag as a key. Point sets remain available even if the
   * corresponding physical group is excluded from the element blocks.
   *
   * @param msh_file Path to the Gmsh `.msh` file.
   * @return Core::IO::MeshInput::RawMesh<3> The intermediate mesh.
   */
  Core::IO::MeshInput::RawMesh<3> read_msh_file(const std::filesystem::path& msh_file);
}  // namespace Core::IO::Gmsh
FOUR_C_NAMESPACE_CLOSE

#endif

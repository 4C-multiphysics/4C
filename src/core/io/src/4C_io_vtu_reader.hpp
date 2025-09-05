// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_IO_VTU_READER_HPP
#define FOUR_C_IO_VTU_READER_HPP

#include "4C_config.hpp"

#include "4C_io_mesh.hpp"

#include <filesystem>

FOUR_C_NAMESPACE_OPEN

namespace Core::IO::VTU
{
  /*!
   * @brief Read a vtu-mesh into the 4C intermediate representation
   *
   * The vtu-format does not have a natural representation of cell- and point-sets. To define these
   * sets, we assume the following point-/cell-data arrays to be present:
   *
   *  - zero or more point-arrays named "point_set_#": Integer arrays telling whether a point is
   *    part of a point-set, where `#` is the id of the point-set
   *  - cell-array "block_id": The id of the block of the respective cell. Note that every block can
   *    only contain elements of the same cell-type.
   *
   * @param vtu_file (in) : Path to the file
   * @return Core::IO::MeshInput::Mesh
   */
  Core::IO::MeshInput::Mesh<3> read_vtu_file(const std::filesystem::path& vtu_file);
}  // namespace Core::IO::VTU

FOUR_C_NAMESPACE_CLOSE

#endif

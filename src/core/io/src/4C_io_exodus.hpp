// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_IO_EXODUS_HPP
#define FOUR_C_IO_EXODUS_HPP

#include "4C_config.hpp"

#include "4C_fem_general_element.hpp"
#include "4C_io_mesh.hpp"

#include <filesystem>
#include <string>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::IO::Exodus
{
  /**
   * Additional parameters that are used in the constructor.
   */
  struct MeshParameters
  {
    /**
     * The ID of the first node in the mesh. This defaults to 1, since this is the default in
     * the Exodus II mesh format.
     */
    int node_start_id{1};
  };

  MeshInput::Mesh read_exodus_file(
      const std::filesystem::path& exodus_file, MeshParameters mesh_parameters = {});
}  // namespace Core::IO::Exodus

FOUR_C_NAMESPACE_CLOSE

#endif

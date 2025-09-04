// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_IO_EXODUS_HPP
#define FOUR_C_IO_EXODUS_HPP

#include "4C_config.hpp"

#include "4C_io_mesh.hpp"

#include <filesystem>

FOUR_C_NAMESPACE_OPEN

namespace Core::IO::Exodus
{
  MeshInput::Mesh<3> read_exodus_file(const std::filesystem::path& exodus_file);
}  // namespace Core::IO::Exodus

FOUR_C_NAMESPACE_CLOSE

#endif

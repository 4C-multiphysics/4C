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

#ifdef FOUR_C_WITH_GMSH
#include <gmsh.h>
#endif

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
   * (unique) tag as a key.
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

#ifdef FOUR_C_WITH_GMSH
  /**
   * @brief RAII guard to manage a gmsh session.
   *
   * Initializes and opens the given msh file on construction and finalizes gmsh on destruction.
   * Explicitly non-copyable and non-movable to avoid multiple finalization calls.
   * @throws if a GmshSession is already active (nested sessions are not supported).
   *
   */
  class GmshSession
  {
   public:
    [[nodiscard]] explicit GmshSession(const std::filesystem::path& msh_file)
    {
      if (active_) FOUR_C_THROW("Gmsh Session already active");

      active_ = true;
      gmsh::initialize();
      try
      {
        gmsh::open(msh_file.string());
      }
      catch (...)
      {
        gmsh::finalize();
        active_ = false;
        FOUR_C_THROW("Failed to open Gmsh file {}\nIs the file corrupted?\nGmsh version: {}",
            msh_file.string(), GMSH_API_VERSION);
      }
    }

    // Non-copyable (Gmsh is a global singleton)
    GmshSession(const GmshSession&) = delete;
    GmshSession& operator=(const GmshSession&) = delete;

    // We don't need move operations
    GmshSession(GmshSession&&) = delete;
    GmshSession& operator=(GmshSession&&) = delete;

    ~GmshSession() noexcept
    {
      gmsh::finalize();
      active_ = false;
    }

   private:
    static inline bool active_ = false;
  };
#endif

}  // namespace Core::IO::Gmsh
FOUR_C_NAMESPACE_CLOSE

#endif

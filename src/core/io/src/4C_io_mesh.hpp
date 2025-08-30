// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_IO_MESH_HPP
#define FOUR_C_IO_MESH_HPP

#include "4C_config.hpp"

#include "4C_fem_general_element.hpp"

#include <filesystem>
#include <string>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::IO::MeshInput
{
  enum class VerbosityLevel : int
  {
    none = 0,              ///< no output,
    summary = 1,           ///< output of summary for blocks and sets,
    detailed_summary = 2,  ///< output of summary for each block and set,
    detailed = 3,          ///< detailed output for each block and set,
    full = 4               ///< detailed output, even for nodes and element connectivities
  };
  constexpr bool operator>(VerbosityLevel lhs, VerbosityLevel rhs)
  {
    return static_cast<int>(lhs) > static_cast<int>(rhs);
  }

  /**
   * Describe each of the VerbosityLevel options.
   */
  std::string describe(VerbosityLevel level);


  struct CellBlock;
  struct SideSet;
  struct LineSet;
  struct PointSet;

  /*!
   * @brief An intermediate representation of finite element meshes
   *
   * 4C will read meshes into this basic representation of the mesh and generate its internal
   * Discretization from it.
   *
   */
  struct Mesh
  {
    /**
     * The points in the mesh. The keys are the point IDs, and the values are the coordinates of the
     * points.
     */
    std::map<int, std::vector<double>> points;

    /**
     * The cell blocks in the mesh. The keys are the cell block IDs, and the values are the cell
     * blocks.
     *
     * The mesh is organized into cell blocks, each containing a collection of cells. Each
     * cell-block is required to have the same cell-type. 4C can solve different equations on each
     * block.
     */
    std::map<int, CellBlock> cell_blocks;

    /**
     * The sides in the mesh. The keys are the side-set IDs, and the values are the side-sets.
     */
    std::map<int, SideSet> side_sets;

    /**
     * The lines in the mesh. The keys are the line-set IDs, and the values are the line-sets.
     */
    std::map<int, LineSet> line_sets;

    /**
     * The points in the mesh. The keys are the point-set IDs, and the values are the point-sets.
     */
    std::map<int, PointSet> point_sets;
  };


  /**
   * A cell-block. This encodes a collection of cells of the same type.
   */
  struct CellBlock
  {
    /**
     * The type of the cells in the cell block.
     */
    FE::CellType cell_type;

    /**
     * Cells in this block. The keys are the cell-IDs, and the values are the IDs of the points
     * making up the cell. The ordering of the points are important for the cell connectivity.
     */
    std::map<int, std::vector<int>> cell_connectivities;
  };

  /**
   * A point set. This encodes a collection of points.
   */
  struct PointSet
  {
    /**
     *  The IDs of the points in the point set.
     */
    std::vector<int> point_ids;
  };

  /**
   * An side set. This encodes a collection of sides/faces of elements.
   */
  struct SideSet
  {
    /**
     * The IDs of the nodes making up the sides of the side set.
     */
    std::map<int, std::vector<int>> sides;
  };

  /**
   * An line set. This encodes a collection of lines/edges of elements.
   */
  struct LineSet
  {
    /**
     * The IDs of the nodes making up the lines of the line set.
     */
    std::map<int, std::vector<int>> lines;
  };

  /**
   * Print a summary of the mesh to the given output stream (details according to @p verbose )
   */
  void print(const Mesh& mesh, std::ostream& os, VerbosityLevel verbose);

  /**
   * Print a summary of the cell block to the given output stream (details according to @p verbose )
   */
  void print(const CellBlock& block, std::ostream& os, VerbosityLevel verbose);

  /**
   * Print a summary of the side set to the given output stream (details according to @p verbose )
   */
  void print(const SideSet& side_set, std::ostream& os, VerbosityLevel verbose);

  /**
   * Print a summary of the line set to the given output stream (details according to @p verbose )
   */
  void print(const LineSet& line_set, std::ostream& os, VerbosityLevel verbose);

  /**
   * Print a summary of the point set to the given output stream (details according to @p verbose )
   */
  void print(const PointSet& point_set, std::ostream& os, VerbosityLevel verbose);
}  // namespace Core::IO::MeshInput

FOUR_C_NAMESPACE_CLOSE

#endif

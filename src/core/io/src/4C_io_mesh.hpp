// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_IO_MESH_HPP
#define FOUR_C_IO_MESH_HPP

#include "4C_config.hpp"

#include "4C_fem_general_cell_type_traits.hpp"
#include "4C_utils_exceptions.hpp"

#include <cstddef>
#include <map>
#include <ranges>
#include <string>
#include <unordered_set>
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


  class CellBlock;
  struct PointSet;

  /*!
   * @brief A point in the mesh
   */
  template <unsigned dim>
  struct Point
  {
    /// External ID of the point as defined in the mesh
    int external_id{};

    /// Coordinates of the point
    std::array<double, dim> coords{};
  };

  /*!
   * @brief An intermediate representation of finite element meshes
   *
   * 4C will read meshes into this basic representation of the mesh and generate its internal
   * Discretization from it.
   *
   */
  template <unsigned dim>
  struct Mesh
  {
    /**
     * The points in the mesh.
     */
    std::vector<Point<dim>> points{};

    /**
     * The cell blocks in the mesh. The keys are the cell block IDs, and the values are the cell
     * blocks.
     *
     * The mesh is organized into cell blocks, each containing a collection of cells. Each
     * cell-block is required to have the same cell-type. 4C can solve different equations on each
     * block.
     */
    std::map<int, CellBlock> cell_blocks{};

    /**
     * The points in the mesh. The keys are the point-set IDs, and the values are the point-sets.
     */
    std::map<int, PointSet> point_sets{};
  };

  /*!
   * A cell in a cell-block.
   */
  struct CellView
  {
    /// External ID of the cell as defined in the mesh
    int external_id{};

    /// Connectivity of the cell (list of point IDs)
    std::span<const int> connectivity{};
  };

  /**
   * A cell-block. This encodes a collection of cells of the same type.
   */
  class CellBlock
  {
   public:
    /**
     * The type of the cells in the cell block.
     */
    FE::CellType cell_type;

    CellBlock(FE::CellType cell_type) : cell_type(cell_type) {}

    /*!
     * @brief Returns the number of cells in this block
     */
    [[nodiscard]] std::size_t size() const
    {
      return cells_connectivity_.size() / FE::num_nodes(cell_type);
    }

    /*!
     * @brief Add a cell to this block
     */
    void add_cell(int external_id, std::span<const int> connectivity)
    {
      FOUR_C_ASSERT_ALWAYS(
          connectivity.size() == static_cast<std::size_t>(FE::num_nodes(cell_type)),
          "You are adding a cell with {} points to a cell-block of type {} expecting {} points per "
          "cell.",
          connectivity.size(), FE::cell_type_to_string(cell_type), FE::num_nodes(cell_type));

      cells_connectivity_.insert(
          cells_connectivity_.end(), connectivity.begin(), connectivity.end());
      external_ids_.push_back(external_id);
    }

    /*!
     * @brief Returns a range for iterating over the cell connectivities in this block
     */
    [[nodiscard]] auto cell_connectivities() const
    {
      auto indices = std::views::iota(std::size_t(0), size());
      return indices | std::views::transform(
                           [this](std::size_t i)
                           {
                             return std::span<const int>(
                                 cells_connectivity_.data() + i * FE::num_nodes(cell_type),
                                 FE::num_nodes(cell_type));
                           });
    }

    /*!
     * @brief Returns a range for iterating over the cells in this block
     */
    [[nodiscard]] auto cells() const
    {
      auto indices = std::views::iota(std::size_t(0), size());
      return indices | std::views::transform(
                           [this](std::size_t i)
                           {
                             return CellView{
                                 .external_id = external_ids_[i],
                                 .connectivity = std::span<const int>(
                                     cells_connectivity_.data() + i * FE::num_nodes(cell_type),
                                     FE::num_nodes(cell_type)),
                             };
                           });
    }

    /*!
     * @brief Returns the connectivity of the i-th cell in this block
     */
    [[nodiscard]] CellView cell(std::size_t i) const
    {
      FOUR_C_ASSERT(
          i < size(), "You are trying to access cell {} in a block with {} cells.", i, size());

      return {.external_id = external_ids_[i],
          .connectivity = {cells_connectivity_.data() + i * FE::num_nodes(cell_type),
              static_cast<std::size_t>(FE::num_nodes(cell_type))}};
    }

    /*!
     * @brief Returns the external ID of the i-th cell in this block
     */
    [[nodiscard]] int external_id(std::size_t i) const
    {
      FOUR_C_ASSERT(i < size(),
          "You are trying to access external ID of cell {} in a block with {} cells.", i, size());
      return external_ids_[i];
    }

   private:
    /*!
     * Cells in this block. The cell connectivity is flattened to a 1D array.
     */
    std::vector<int> cells_connectivity_{};

    /*!
     * The external IDs of the cells in this block
     */
    std::vector<int> external_ids_{};
  };

  /*!
   * A point set. This encodes a collection of points.
   */
  struct PointSet
  {
    /**
     *  The IDs of the points in the point set.
     */
    std::unordered_set<int> point_ids;
  };

  /*!
   * Print a summary of the mesh to the given output stream (details according to @p verbose )
   */
  template <unsigned dim>
  void print(const Mesh<dim>& mesh, std::ostream& os, VerbosityLevel verbose);

  /*!
   * Print a summary of the cell block to the given output stream (details according to @p verbose
   * )
   */
  void print(const CellBlock& block, std::ostream& os, VerbosityLevel verbose);

  /*!
   * Print a summary of the point set to the given output stream (details according to @p verbose
   * )
   */
  void print(const PointSet& point_set, std::ostream& os, VerbosityLevel verbose);
}  // namespace Core::IO::MeshInput

FOUR_C_NAMESPACE_CLOSE

#endif

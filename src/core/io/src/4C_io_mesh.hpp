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
#include "4C_io_input_parameter_container.hpp"
#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_utils_demangle.hpp"
#include "4C_utils_exceptions.hpp"

#include <algorithm>
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


  template <unsigned dim>
  class CellBlock;
  struct PointSet;

  using InternalIdType = int;
  using ExternalIdType = int;


  /*!
   * @brief These are the supported field data (scalars, vectors, symmetric tensors and tensors)
   * with scalar types int and double.
   */
  template <unsigned dim>
  using EligibleFieldTypes = std::variant<int, double, Core::LinAlg::Tensor<int, dim>,
      Core::LinAlg::Tensor<double, dim>, Core::LinAlg::SymmetricTensor<int, dim, dim>,
      Core::LinAlg::SymmetricTensor<double, dim, dim>, Core::LinAlg::Tensor<int, dim, dim>,
      Core::LinAlg::Tensor<double, dim, dim>>;

  namespace Internal
  {
    template <typename T>
    struct EligibleFieldVariantTypeHelper;

    template <typename... Types>
    struct EligibleFieldVariantTypeHelper<std::variant<Types...>>
    {
      using type = std::variant<std::vector<Types>...>;
    };
  }  // namespace Internal

  /*!
   * @brief A variant holding a std::vector of all supported field data types
   */
  template <unsigned dim>
  using FieldDataVariantType =
      Internal::EligibleFieldVariantTypeHelper<EligibleFieldTypes<dim>>::type;

  /*!
   * @brief An intermediate representation of finite element meshes
   *
   * 4C will read meshes into this basic representation of the mesh and generate its internal
   * Discretization from it.
   */
  template <unsigned dim>
  struct Mesh
  {
    /**
     * The points in the mesh.
     */
    std::vector<std::array<double, dim>> points{};

    /**
     * Point data associated with each point in the mesh.
     *
     * The key refers to the name of the field and the value is a variant holding a std::vector of
     * one of the eligible data types, which are scalars, vectors, symmetric tensors or tensors
     * with data-types @p int or @p double.
     */
    std::unordered_map<std::string, FieldDataVariantType<dim>> point_data{};

    /*!
     * @brief Some mesh formats provide an ID for points in the mesh. If available, these IDs
     * are stored in this vector.
     */
    std::optional<std::vector<ExternalIdType>> external_ids;

    /**
     * The cell blocks in the mesh. The keys are the cell block IDs, and the values are the cell
     * blocks.
     *
     * The mesh is organized into cell blocks, each containing a collection of cells. Each
     * cell-block is required to have the same cell-type. 4C can solve different equations on each
     * block.
     */
    std::map<ExternalIdType, CellBlock<dim>> cell_blocks{};

    /**
     * The points in the mesh. The keys are the point-set IDs, and the values are the point-sets.
     */
    std::map<ExternalIdType, PointSet> point_sets{};
  };

  /**
   * A cell-block. This encodes a collection of cells of the same type.
   */
  template <unsigned dim>
  class CellBlock
  {
   public:
    /**
     * The type of the cells in the cell block.
     */
    FE::CellType cell_type;

    /*!
     * The external IDs of the cells in this block (if available).
     */
    std::optional<std::vector<ExternalIdType>> external_ids_{};

    /**
     * An optional name for the cell block.
     *
     * @note Not every file formats provides std::string-names for cell blocks.
     */
    std::optional<std::string> name{};

    /**
     * Cell data associated with each cell in the block.
     *
     * The key refers to the name of the field and the value is a variant holding a std::vector of
     * one of the eligible data types, which are scalars, vectors, symmetric tensors or tensors
     * with data-types @p int or @p double.
     */
    std::unordered_map<std::string, FieldDataVariantType<dim>> cell_data{};

    /**
     * Optional specific data for the cell block.
     *
     * This data can be used to construct specialized user elements for this block.
     */
    std::optional<Core::IO::InputParameterContainer> specific_data{};

    CellBlock(FE::CellType cell_type) : cell_type(cell_type) {}

    /*!
     * @brief Returns the number of cells in this block
     */
    [[nodiscard]] std::size_t size() const { return cells_.size() / FE::num_nodes(cell_type); }

    /*!
     * @brief Add a cell to this block
     */
    void add_cell(std::span<const InternalIdType> connectivity)
    {
      FOUR_C_ASSERT_ALWAYS(
          connectivity.size() == static_cast<std::size_t>(FE::num_nodes(cell_type)),
          "You are adding a cell with {} points to a cell-block of type {} expecting {} points per "
          "cell.",
          connectivity.size(), FE::cell_type_to_string(cell_type), FE::num_nodes(cell_type));

      cells_.insert(cells_.end(), connectivity.begin(), connectivity.end());
    }

    /*!
     * @brief Returns a range for iterating over the cell connectivities in this block
     */
    [[nodiscard]] auto cells() const
    {
      auto indices = std::views::iota(std::size_t(0), size());
      return indices |
             std::views::transform(
                 [this](std::size_t i)
                 {
                   return std::span<const InternalIdType>(
                       cells_.data() + i * FE::num_nodes(cell_type), FE::num_nodes(cell_type));
                 });
    }

    /*!
     * @brief Allocates memory for the given number of cells in this block
     */
    void reserve(const std::size_t num_cells)
    {
      cells_.reserve(num_cells * FE::num_nodes(cell_type));
      if (external_ids_.has_value()) external_ids_->reserve(num_cells);
    }

   private:
    /*!
     * Cells in this block. The cell connectivity is flattened to a 1D array.
     */
    std::vector<InternalIdType> cells_{};
  };

  /*!
   * A point set. This encodes a collection of points.
   */
  struct PointSet
  {
    /**
     *  The IDs of the points in the point set.
     */
    std::unordered_set<ExternalIdType> point_ids;

    /**
     * An optional name for the point set.
     *
     * @note Not every file formats provides std::string-names for point sets.
     */
    std::optional<std::string> name{};
  };

  /*!
   * @brief Asserts that the given mesh internals are consistent and valid.
   */
  template <unsigned dim>
  void assert_valid(const Mesh<dim>& mesh);

  /*!
   * Print a summary of the mesh to the given output stream (details according to @p verbose )
   */
  template <unsigned dim>
  void print(const Mesh<dim>& mesh, std::ostream& os, VerbosityLevel verbose);

  /*!
   * Print a summary of the cell block to the given output stream (details according to @p verbose
   * )
   */
  template <unsigned dim>
  void print(const CellBlock<dim>& block, std::ostream& os, VerbosityLevel verbose);

  /*!
   * Print a summary of the point set to the given output stream (details according to @p verbose
   * )
   */
  void print(const PointSet& point_set, std::ostream& os, VerbosityLevel verbose);


  /*!
   * @brief Reads in a map with value_type @p T from the element data of the given @p mesh.
   *
   * @throws if the element data is not found in at least one of the element blocks or if the type
   * of the element data is not implicitly convertible to the expected type.
   */
  template <typename IndexType, typename T, unsigned dim>
    requires(std::is_integral_v<IndexType>)
  void read_value_from_element_data(
      const Mesh<dim>& mesh, const std::string& key, std::unordered_map<IndexType, T>& value)
  {
    unsigned ele_id = 0;
    for (const auto& [block_id, block] : mesh.cell_blocks)
    {
      // This block is not part of any physics, hence ignore it
      if (!block.specific_data) continue;

      FOUR_C_ASSERT_ALWAYS(block.cell_data.contains(key),
          "The element block {} does not contain element data with the name '{}'.", block_id, key);

      if (!block.cell_data.contains(key)) continue;

      const FieldDataVariantType<dim>& data = block.cell_data.at(key);

      std::visit(
          [&](const auto& data)
          {
            using Type = std::decay_t<decltype(data)>::value_type;
            if constexpr (!std::is_convertible_v<Type, T>)
            {
              FOUR_C_THROW(
                  "Element data with name '{}' has type '{}' which is not convertible to "
                  "the target type '{}'.",
                  key, Core::Utils::try_demangle(typeid(Type).name()),
                  Core::Utils::try_demangle(typeid(T).name()));
            }
            else
            {
              for (const auto& v : data)
              {
                value[ele_id++] = static_cast<T>(v);
              }
            }
          },
          data);
    }
  }
}  // namespace Core::IO::MeshInput

FOUR_C_NAMESPACE_CLOSE

#endif

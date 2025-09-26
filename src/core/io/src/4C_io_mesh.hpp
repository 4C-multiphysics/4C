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
#include "4C_utils_exceptions.hpp"
#include "4C_utils_owner_or_view.hpp"

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

  /**
   * Constant representing an invalid external ID.
   */
  constexpr ExternalIdType invalid_external_id = -1;

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
  struct RawMesh
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
      auto indices = std::views::iota(0ull, size());
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

  namespace Internal
  {
    /**
     * A lightweight reference to a point in a Mesh used to provide a nicer interface.
     *
     * @note This class does not own any data and  only refers to data in the RawMesh. Thus, it can
     * only be used as long as the corresponding RawMesh is alive.
     */
    template <unsigned dim, bool is_const>
    struct PointReference
    {
      template <typename T>
      using MaybeConst = std::conditional_t<is_const, const T, T>;

      PointReference(MaybeConst<RawMesh<dim>>* raw_mesh, size_t index)
          : raw_mesh_(raw_mesh), index(index)
      {
        FOUR_C_ASSERT(raw_mesh_ != nullptr, "RawMesh pointer must not be null.");
        FOUR_C_ASSERT(index < raw_mesh_->points.size(), "Point index {} is out of bounds.", index);
      }

      /**
       * Get the spatial coordinates of the point.
       */
      MaybeConst<std::array<double, dim>>& coordinate() const { return raw_mesh_->points[index]; }

      /**
       * Get the external ID of the point (if available). If not available, returns
       * #invalid_external_id.
       */
      ExternalIdType external_id() const
      {
        return raw_mesh_->external_ids ? (*raw_mesh_->external_ids)[index] : invalid_external_id;
      }

     private:
      MaybeConst<RawMesh<dim>>* raw_mesh_;
      size_t index;
    };
  }  // namespace Internal

  /**
   * @brief An interface to a mesh.
   *
   * This class internally uses a RawMesh and exposes a reduced interface to it that is easier
   * to work with as it does not require knowledge of the internals. Also, this allows us to
   * implement filtering operations that return a new Mesh object with only a subset of
   * selected entities.
   */
  template <unsigned dim>
  class Mesh
  {
   public:
    /**
     * Default constructor creating an empty mesh.
     */
    Mesh() : raw_mesh_(std::make_unique<RawMesh<dim>>()) {}

    /**
     * Construct a mesh from a RawMesh. The Mesh takes ownership of the @p raw_mesh.
     */
    explicit Mesh(RawMesh<dim>&& raw_mesh)
        : raw_mesh_(Utils::make_owner<RawMesh<dim>>(std::move(raw_mesh)))
    {
      default_fill_indices();
    }

    /**
     * Construct a mesh that is a view on the given RawMesh. The Mesh does not take ownership of
     * the @p raw_mesh.
     */
    static Mesh create_view(RawMesh<dim>& raw_mesh)
    {
      Mesh mesh;
      mesh.raw_mesh_ = Utils::make_view(&raw_mesh);
      mesh.default_fill_indices();
      return mesh;
    }

    /**
     * Get a range of all cell blocks defined in this mesh.
     */
    [[nodiscard]] auto cell_blocks() const
    {
      return cell_blocks_ids_filter_ | std::views::transform([this](size_t id) -> decltype(auto)
                                           { return *raw_mesh_->cell_blocks.find(id); });
    }

    [[nodiscard]] auto cell_blocks()
    {
      return cell_blocks_ids_filter_ | std::views::transform([this](size_t id) -> decltype(auto)
                                           { return *raw_mesh_->cell_blocks.find(id); });
    }

    /**
     * Get a range of all points defined in this mesh.
     */
    [[nodiscard]] auto points() const { return std::views::all(raw_mesh_->points); }
    [[nodiscard]] auto points() { return std::views::all(raw_mesh_->points); }

    /**
     * Get a range of all points defined in this mesh along with their associated data (if any).
     * This method is likely slower than points(), so only use it if you actually need the
     * associated data.
     */
    [[nodiscard]] auto points_with_data() const
    {
      return std::views::iota(0ull, raw_mesh_->points.size()) |
             std::views::transform([this](std::size_t i)
                 { return Internal::PointReference<dim, true>(raw_mesh_.get(), i); });
    }

    [[nodiscard]] auto points_with_data()
    {
      return std::views::iota(0ull, raw_mesh_->points.size()) |
             std::views::transform([this](std::size_t i)
                 { return Internal::PointReference<dim, false>(raw_mesh_.get(), i); });
    }

    [[nodiscard]] auto point_sets() const { return std::views::all(raw_mesh_->point_sets); }
    [[nodiscard]] auto point_sets() { return std::views::all(raw_mesh_->point_sets); }

    /**
     * Filter the mesh to only contain cell blocks with the given IDs.
     */
    [[nodiscard]] Mesh filter_by_cell_block_ids(
        const std::vector<ExternalIdType>& cell_block_ids) const
    {
      Mesh filtered_mesh;
      filtered_mesh.raw_mesh_ = Utils::make_view(raw_mesh_.get());
      filtered_mesh.cell_blocks_ids_filter_ = cell_block_ids;
      for (const auto id : cell_block_ids)
      {
        FOUR_C_ASSERT_ALWAYS(raw_mesh_->cell_blocks.contains(id),
            "You are trying to filter the mesh by cell-block ID {}, but the mesh does not contain "
            "a cell-block with this ID.",
            id);
      }
      return filtered_mesh;
    }

   private:
    /**
     * Setup default indices including all entities in the mesh.
     */
    void default_fill_indices()
    {
      FOUR_C_ASSERT(
          raw_mesh_.get() != nullptr, "Internal error: RawMesh pointer must not be null.");

      cell_blocks_ids_filter_.resize(raw_mesh_->cell_blocks.size());
      std::ranges::copy(raw_mesh_->cell_blocks | std::views::keys, cell_blocks_ids_filter_.begin());
    }

    Utils::OwnerOrView<RawMesh<dim>> raw_mesh_;

    /**
     * A filter function that can be used to filter cell blocks when accessing them. By default,
     * nothing is filtered.
     */
    std::vector<ExternalIdType> cell_blocks_ids_filter_{};
  };


  /*!
   * @brief Asserts that the given mesh internals are consistent and valid.
   *
   * Mostly used for internal consistency checks and unit tests.
   */
  template <unsigned dim>
  void assert_valid(const RawMesh<dim>& mesh);

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
}  // namespace Core::IO::MeshInput

FOUR_C_NAMESPACE_CLOSE

#endif

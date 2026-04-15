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
#include "4C_utils_demangle.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_owner_or_view.hpp"
#include "4C_utils_vector2D.hpp"

#include <algorithm>
#include <any>
#include <cstddef>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <ranges>
#include <string>
#include <tuple>
#include <type_traits>
#include <typeindex>
#include <unordered_map>
#include <unordered_set>
#include <variant>
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
   * @brief These are the supported raw data types.
   */
  using EligibleRawFieldTypes = std::variant<bool, int, double>;

  namespace Internal
  {
    template <typename T>
    struct EligibleFieldVariantTypeHelper;

    template <typename... Types>
    struct EligibleFieldVariantTypeHelper<std::variant<Types...>>
    {
      using type = std::variant<Core::Utils::Vector2D<Types>...>;
    };

    template <typename T>
    struct EligibleDataSpanHelper;

    template <typename... Types>
    struct EligibleDataSpanHelper<std::variant<Types...>>
    {
      using type = std::variant<std::span<const Types>...>;
    };
  }  // namespace Internal

  /*!
   * @brief A variant holding a std::vector of all supported field data types
   */
  using FieldDataVariantType =
      Internal::EligibleFieldVariantTypeHelper<EligibleRawFieldTypes>::type;

  /*!
   * @brief A variant holding a std::span of all supported field data types
   */
  using FieldRawDataSpanVariantType = Internal::EligibleDataSpanHelper<EligibleRawFieldTypes>::type;


  /**
   * @brief A map of data converters that can convert EligibleFieldTypes to specific user types. The
   * converters need to be attached by the mesh reader to decide how to convert the raw data
   * to specific types (e.g., SymmetricTensors rely on specific internal ordering which might differ
   * between file formats).
   */
  using ConverterMapType = std::unordered_map<std::type_index, std::function<std::any(std::any)>>;

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
     * The key refers to the name of the field and the value is a variant holding a
     * Core::Utils::Vector2D of one of the eligible data types, which are @p bool,  @p int or @p
     * double.
     */
    std::unordered_map<std::string, FieldDataVariantType> point_data{};

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

    /**
     * A map of data converters that can convert EligibleFieldTypes to specific user types. The
     * converters need to be attached by the mesh reader to decide how to convert the raw data
     * to specific types (e.g., SymmetricTensors rely on specific internal ordering which might
     * differ between file formats).
     *
     * The key is the typeid of the target type, and the value is a function that takes an
     * eligible raw field type and converts it to the target type (type-erased via std::any).
     *
     * @note The converter should throw if the conversion is not possible from the given
     * eligible field type to the specified target type.
     */
    ConverterMapType converters{};
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
    std::unordered_map<std::string, FieldDataVariantType> cell_data{};

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
      auto indices = std::views::iota(size_t{0}, size());
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

  /**
   * @brief Creates a type index for a converter from SourceType to TargetType.
   */
  template <typename SourceType, typename TargetType>
  std::type_index make_converter_type_index()
  {
    return std::type_index(typeid(std::tuple<SourceType, TargetType>));
  }


  namespace Internal
  {
    /**
     * @brief Find an eligible converter from a list of explicit converters (or some implicit
     * converters).
     *
     * @throw If no eligible converter could be found from SrcType to TargetType
     *
     * @note An eligible converter might not guarantee successful conversion at runtime (e.g.,
     * number of components need to match the size of an array)
     */
    template <typename SrcType, typename TargetType>
    std::function<TargetType(std::span<const SrcType>)> get_eligible_converter(
        const ConverterMapType& explicit_converters)
    {
      // search for eligible converter
      std::optional<std::function<TargetType(std::span<const SrcType>)>> converter = std::nullopt;
      if (auto it = explicit_converters.find(make_converter_type_index<SrcType, TargetType>());
          it != explicit_converters.end())
      {
        const auto& eligible_converter_to_any = it->second;
        converter = [&](std::span<const SrcType> raw_data) -> TargetType
        { return std::any_cast<TargetType>(eligible_converter_to_any(raw_data)); };
      }

      // Let's also have some default converters (if we don't have an explicit one)
      if (!converter.has_value())
      {
        if constexpr (std::is_convertible_v<SrcType, TargetType>)
        {
          converter = [&](std::span<const SrcType> raw_data) -> TargetType
          {
            FOUR_C_ASSERT_ALWAYS(raw_data.size() == 1,
                "Expecting a scalar type, but got array of size {}.", raw_data.size());

            return static_cast<TargetType>(raw_data[0]);
          };
        }
        else if constexpr (std::ranges::sized_range<TargetType>)
        {
          if constexpr (std::is_convertible_v<SrcType, typename TargetType::value_type>)
          {
            converter = [&](std::span<const SrcType> raw_data) -> TargetType
            {
              TargetType target_value;

              // For dynamic containers like vector, we need to ensure they have the right size
              // first
              if constexpr (requires { target_value.resize(raw_data.size()); })
              {
                target_value.resize(raw_data.size());
              }

              // Check sizes match (especially important for std::array)
              FOUR_C_ASSERT_ALWAYS(target_value.size() == raw_data.size(),
                  "Extracted {} values, but expected {}.", raw_data.size(), target_value.size());

              // Copy the data in
              std::ranges::copy(raw_data, target_value.begin());

              return target_value;
            };
          }
        }
      }

      FOUR_C_ASSERT_ALWAYS(converter.has_value(),
          "Your mesh reader does not provide a converter from type '{}' to target type '{}'.",
          Core::Utils::try_demangle(typeid(SrcType).name()),
          Core::Utils::try_demangle(typeid(TargetType).name()));

      return *converter;
    }
  }  // namespace Internal

  /**
   * A lightweight reference to a point in a Mesh used to provide a nicer interface.
   *
   * @note This class does not own any data and only refers to data in the RawMesh. Thus, it can
   * only be used as long as the corresponding RawMesh is alive.
   */
  template <unsigned dim, bool is_const>
  struct PointReference
  {
    template <typename T>
    using MaybeConst = std::conditional_t<is_const, const T, T>;

    PointReference(MaybeConst<RawMesh<dim>>* raw_mesh, size_t index)
        : raw_mesh_(raw_mesh), index_(index)
    {
      FOUR_C_ASSERT(raw_mesh_ != nullptr, "RawMesh pointer must not be null.");
      FOUR_C_ASSERT(index_ < raw_mesh_->points.size(), "Point index {} is out of bounds.", index_);
    }

    /**
     * Get the spatial coordinates of the point.
     */
    [[nodiscard]] MaybeConst<std::array<double, dim>>& coordinate() const
    {
      return raw_mesh_->points[index_];
    }

    /**
     * Get the ID of the point in the mesh. This is the ID that cells use to form their
     * connectivity.
     */
    [[nodiscard]] size_t id() const { return index_; }

    /**
     * Get the external ID of the point (if available). If not available, returns
     * #invalid_external_id.
     */
    [[nodiscard]] ExternalIdType external_id() const
    {
      return raw_mesh_->external_ids ? (*raw_mesh_->external_ids)[index_] : invalid_external_id;
    }

    /**
     * @brief Returns the number of data-fields associated with this point
     */
    [[nodiscard]] std::size_t data_size() const { return raw_mesh_->point_data.size(); }

    [[nodiscard]] FieldRawDataSpanVariantType data(const std::string& field_name) const
    {
      return std::visit([&](const auto& data) -> FieldRawDataSpanVariantType
          { return data.at(index_); }, raw_mesh_->point_data.at(field_name));
    }

    /**
     * @brief Return the data as a specific type by making use of all known converters
     */
    template <typename T>
    [[nodiscard]] T data_as(const std::string& field_name) const
    {
      return std::visit(
          [&](const auto& data) -> T
          {
            using TargetType = std::decay_t<T>;
            using SourceType = std::decay_t<decltype(data)>::value_type;

            const auto converter =
                Internal::get_eligible_converter<SourceType, TargetType>(raw_mesh_->converters);

            return converter(data[index_]);
          },
          raw_mesh_->point_data.at(field_name));
    }

   private:
    MaybeConst<RawMesh<dim>>* raw_mesh_;
    size_t index_;
  };


  /**
   * A lightweight reference to a cell in a Mesh used to provide a nicer interface.
   *
   * @note This class does not own any data and only refers to data in the RawMesh. Thus, it can
   * only be used as long as the corresponding RawMesh is alive.
   */
  template <unsigned dim, bool is_const>
  struct CellReference
  {
    template <typename T>
    using MaybeConst = std::conditional_t<is_const, const T, T>;

    CellReference(
        MaybeConst<RawMesh<dim>>* raw_mesh, MaybeConst<CellBlock<dim>>* cell_block, size_t index)
        : raw_mesh_(raw_mesh), cell_block_(cell_block), index_(index)
    {
      FOUR_C_ASSERT(raw_mesh_ != nullptr, "RawMesh pointer must not be null.");
      FOUR_C_ASSERT(cell_block_ != nullptr, "CellBlock pointer must not be null.");
      FOUR_C_ASSERT(index_ < cell_block_->size(), "Cell index {} is out of bounds.", index_);
    }

    /**
     * Get the ID of the cell in the mesh.
     */
    [[nodiscard]] size_t id() const { return index_; }

    /**
     * Get the external ID of the cell (if available). If not available, returns
     * #invalid_external_id.
     */
    [[nodiscard]] ExternalIdType external_id() const
    {
      return cell_block_->external_ids ? (*cell_block_->external_ids)[index_] : invalid_external_id;
    }

    [[nodiscard]] FieldRawDataSpanVariantType data(const std::string& field_name) const
    {
      return std::visit([&](const auto& data) -> FieldRawDataSpanVariantType
          { return data.at(index_); }, cell_block_->cell_data.at(field_name));
    }

    template <typename T>
    [[nodiscard]] T data_as(const std::string& field_name) const
    {
      return std::visit(
          [&](const auto& data) -> T
          {
            using TargetType = std::decay_t<T>;
            using SourceType = std::decay_t<decltype(data)>::value_type;

            const auto converter =
                Internal::get_eligible_converter<SourceType, TargetType>(raw_mesh_->converters);

            return converter(data[index_]);
          },
          cell_block_->cell_data.at(field_name));
    }

   private:
    MaybeConst<RawMesh<dim>>* raw_mesh_;
    MaybeConst<CellBlock<dim>>* cell_block_;
    size_t index_;
  };


  /**
   * A lightweight reference to a cell-block in a Mesh used to provide a nicer interface.
   *
   * @note This class does not own any data and only refers to data in the RawMesh. Thus, it can
   * only be used as long as the corresponding RawMesh is alive.
   */
  template <unsigned dim, bool is_const>
  struct CellBlockReference
  {
    template <typename T>
    using MaybeConst = std::conditional_t<is_const, const T, T>;

    CellBlockReference(MaybeConst<RawMesh<dim>>* raw_mesh, size_t id)
        : raw_mesh_(raw_mesh), cell_block_(&raw_mesh_->cell_blocks.at(id)), id_(id)
    {
      FOUR_C_ASSERT(raw_mesh_ != nullptr, "RawMesh pointer must not be null.");
    }

    [[nodiscard]] MaybeConst<CellBlock<dim>>& get() { return *cell_block_; }

    [[nodiscard]] const CellBlock<dim>& get() const { return *cell_block_; }

    /**
     * Get the ID of the cell-block in the mesh.
     */
    [[nodiscard]] size_t id() const { return id_; }

    [[nodiscard]] const std::optional<std::string>& name() const { return cell_block_->name; }


    [[nodiscard]] std::size_t size() const { return cell_block_->size(); }

    [[nodiscard]] FE::CellType cell_type() const { return cell_block_->cell_type; }

    [[nodiscard]] auto cells() const { return cell_block_->cells(); }

    /**
     * Get a lightweight cells iterator in this block along with their associated data (if any).
     */
    [[nodiscard]] auto cells_with_data() const
    {
      auto indices = std::views::iota(size_t{0}, cell_block_->size());
      return indices | std::views::transform([this](std::size_t i)
                           { return CellReference<dim, true>(raw_mesh_, cell_block_, i); });
    }

    [[nodiscard]] const auto& cell_data() const { return cell_block_->cell_data; }

   private:
    MaybeConst<RawMesh<dim>>* raw_mesh_;
    MaybeConst<CellBlock<dim>>* cell_block_;
    size_t id_;
  };

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
    Mesh();

    /**
     * Construct a mesh from a RawMesh. The Mesh takes ownership of the @p raw_mesh.
     */
    explicit Mesh(RawMesh<dim>&& raw_mesh);

    /**
     * Construct a mesh that is a view on the given RawMesh. The Mesh does not take ownership of
     * the @p raw_mesh.
     */
    static Mesh create_view(RawMesh<dim>& raw_mesh);

    /**
     * Get a range of all cell blocks as a lightweight reference defined in this mesh.
     */
    [[nodiscard]] auto cell_blocks() const
    {
      return cell_blocks_ids_filter_ |
             std::views::transform(
                 [this](size_t id) { return CellBlockReference<dim, true>(raw_mesh_.get(), id); });
    }

    [[nodiscard]] auto cell_blocks()
    {
      return cell_blocks_ids_filter_ |
             std::views::transform(
                 [this](size_t id) { return CellBlockReference<dim, false>(raw_mesh_.get(), id); });
    }

    /**
     * Get a range of all points defined in this mesh. This only returns the coordinates of the
     * points. See points_with_data() if you also need other associated data.
     */
    [[nodiscard]] auto points() const
    {
      return point_ids_filter_ | std::views::transform([this](std::size_t i) -> decltype(auto)
                                     { return raw_mesh_->points[i]; });
    }

    [[nodiscard]] auto points()
    {
      return point_ids_filter_ | std::views::transform([this](std::size_t i) -> decltype(auto)
                                     { return raw_mesh_->points[i]; });
    }

    /**
     * Return true if the mesh has point data with the given @p field_name.
     *
     * @note If a field of the given name exists, point data is guaranteed to be available for all
     * points in the mesh.
     */
    [[nodiscard]] bool has_point_data(const std::string& field_name) const;

    /**
     * Get a range of all points defined in this mesh along with their associated data (if any).
     * This method is likely slower than points(), so only use it if you actually need the
     * associated data.
     */
    [[nodiscard]] auto points_with_data() const
    {
      return point_ids_filter_ | std::views::transform([this](std::size_t i)
                                     { return PointReference<dim, true>(raw_mesh_.get(), i); });
    }

    [[nodiscard]] auto points_with_data()
    {
      return point_ids_filter_ | std::views::transform([this](std::size_t i)
                                     { return PointReference<dim, false>(raw_mesh_.get(), i); });
    }

    /**
     * Get a range of all point sets defined in this mesh.
     */
    [[nodiscard]] auto point_sets() const
    {
      return point_sets_ids_filter_ | std::views::transform([this](std::size_t i) -> decltype(auto)
                                          { return *raw_mesh_->point_sets.find(i); });
    }

    [[nodiscard]] auto point_sets()
    {
      return point_sets_ids_filter_ | std::views::transform([this](std::size_t i) -> decltype(auto)
                                          { return *raw_mesh_->point_sets.find(i); });
    }

    /**
     * Filter the mesh to only contain cell blocks with the given IDs. The points are filtered to
     * only contain those that are used by the remaining cell blocks. Point sets must either
     * contain all the remaining points or none of them, otherwise an error is thrown. Only the
     * point sets that contain all the remaining points are kept.
     *
     * @note The returned filtered mesh is a view on the original mesh and does not own any data.
     */
    [[nodiscard]] Mesh filter_by_cell_block_ids(
        const std::vector<ExternalIdType>& cell_block_ids) const;

    /**
     * Check whether the mesh is empty, i.e., it contains no cell blocks and no points.
     *
     * @note A mesh is considered as not empty if it contains an empty cell-block (the mesh does not
     * have points or cells).
     */
    [[nodiscard]] bool empty() const
    {
      return point_ids_filter_.empty() && cell_blocks_ids_filter_.empty();
    }

    [[nodiscard]] const auto& converters() const { return raw_mesh_->converters; }

   private:
    /**
     * Setup default indices including all entities in the mesh.
     */
    void default_fill_indices();

    /**
     * Underlying raw mesh.
     */
    Utils::OwnerOrView<RawMesh<dim>> raw_mesh_;

    /**
     * A list of filtered indices to be used to filter cell blocks when accessing them. By default,
     * nothing is filtered.
     */
    std::vector<ExternalIdType> cell_blocks_ids_filter_{};

    /**
     * A list of filtered indices to be used to filter point sets when accessing them.
     */
    std::vector<ExternalIdType> point_sets_ids_filter_{};

    /**
     * A list of filtered indices to be used to filter points when accessing them.
     */
    std::vector<ExternalIdType> point_ids_filter_{};
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

  /*!
   * @brief Allows to pass a pointer to the block cell data and a block local cell index to the
   * element creation routine.
   */
  class ElementDataFromCellData
  {
    static constexpr unsigned dim = 3;
    using CellDataMap = std::unordered_map<std::string, FieldDataVariantType>;

   public:
    /*!
     * @brief Empyt constructor if no cell data is available.
     */
    ElementDataFromCellData() = default;

    /*!
     * @brief Constructor with given map and cell index.
     */
    ElementDataFromCellData(
        const CellDataMap& cell_data, std::size_t cell_index, const ConverterMapType& converters)
        : cell_data_(&cell_data), cell_index_(cell_index), converters_(&converters)
    {
      // Check that the given cell index is valid for all cell data fields (should all have the same
      // size).
      for (const auto& [key, variant] : *cell_data_)
      {
        std::visit(
            [&](const auto& vec)
            {
              if (cell_index_ >= vec.size())
              {
                FOUR_C_THROW("Cell index {} is out of bounds for cell data '{}' with size {}.",
                    cell_index_, key, vec.size());
              }
            },
            variant);
      }
    }

    /*!
     * @brief Get the value of the cell data with the given name for the cell corresponding to this
     * view.
     *
     * @throws if no cell data with the given name exists or if the type of the cell data is not
     * implicitly or explicitly convertible to the expected type.
     */
    template <typename T>
    T get(const std::string& cell_data_key) const
    {
      FOUR_C_ASSERT_ALWAYS(
          cell_data_ != nullptr, "Cell data pointer in ElementDataFromCellData is null.");

      FOUR_C_ASSERT_ALWAYS(cell_data_->contains(cell_data_key),
          "The cell data does not contain the key '{}'.", cell_data_key);

      return std::visit(
          [&](const auto& data) -> T
          {
            using TargetType = std::decay_t<T>;
            using SourceType = std::decay_t<decltype(data)>::value_type;

            const auto converter =
                Internal::get_eligible_converter<SourceType, TargetType>(*converters_);

            return converter(data[cell_index_]);
          },
          cell_data_->at(cell_data_key));
    }

   private:
    // Pointer to the cell data map.
    const CellDataMap* cell_data_ = nullptr;

    // Block local cell index for which data will be returned by this object.
    std::size_t cell_index_ = std::numeric_limits<std::size_t>::max();

    // A reference to the converters of the raw-mesh
    const ConverterMapType* converters_ = nullptr;
  };

  /*!
   * @brief Reads in a map with value_type @p T from the cell data of the given @p mesh.
   *
   * @throws if the cell data is not found in all cell blocks or if the type of the cell data is not
   * implicitly convertible to the expected type.
   */
  template <typename IndexType, typename TargetType, unsigned dim>
    requires(std::is_integral_v<IndexType>)
  void read_value_from_cell_data(const Mesh<dim>& mesh, const std::string& key,
      std::unordered_map<IndexType, TargetType>& value)
  {
    unsigned ele_id = 0;
    for (const auto& cell_block : mesh.cell_blocks())
    {
      FOUR_C_ASSERT_ALWAYS(cell_block.cell_data().contains(key),
          "The cell block {} does not contain cell data with the name '{}'.", cell_block.id(), key);

      const FieldDataVariantType& data = cell_block.cell_data().at(key);

      std::visit(
          [&](const auto& data)
          {
            using SourceType = std::decay_t<decltype(data)>::value_type;

            // Get for eligible converter
            std::function<TargetType(std::span<const SourceType>)> converter =
                Internal::get_eligible_converter<SourceType, TargetType>(mesh.converters());

            // We can safely convert each entry
            for (const auto& v : data.items())
            {
              value[ele_id++] = converter(v);
            }
          },
          data);
    }
  }

  /*!
   * @brief Reads in a map with value_type @p T from the point data of the given @p mesh.
   *
   * @throws if the point data is not found in all point sets or if the type of the point data is
   * not implicitly convertible to the expected type.
   */
  template <typename IndexType, typename TargetType, unsigned dim>
    requires(std::is_integral_v<IndexType>)
  void read_value_from_point_data(const Mesh<dim>& mesh, const std::string& key,
      std::unordered_map<IndexType, TargetType>& value)
  {
    if (mesh.empty())
    {
      // Nothing to do for an empty mesh
      return;
    }
    FOUR_C_ASSERT_ALWAYS(
        mesh.has_point_data(key), "The mesh does not contain point data with the name '{}'.", key);

    for (const auto& point_ref : mesh.points_with_data())
    {
      const FieldRawDataSpanVariantType& data = point_ref.data(key);

      std::visit(
          [&](const auto& data)
          {
            using SourceType = std::decay_t<decltype(data)>::value_type;

            // Get for eligible converter
            std::function<TargetType(std::span<const SourceType>)> converter =
                Internal::get_eligible_converter<SourceType, TargetType>(mesh.converters());

            // We can safely convert each entry
            value[point_ref.id()] = converter(data);
          },
          data);
    }
  }
}  // namespace Core::IO::MeshInput

FOUR_C_NAMESPACE_CLOSE

#endif

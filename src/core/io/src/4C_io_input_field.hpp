// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_IO_INPUT_FIELD_HPP
#define FOUR_C_IO_INPUT_FIELD_HPP

#include "4C_config.hpp"

#include "4C_comm_exporter.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_io_mesh.hpp"
#include "4C_io_yaml.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_std23_unreachable.hpp"

#include <filesystem>
#include <functional>
#include <numeric>
#include <ranges>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>

FOUR_C_NAMESPACE_OPEN

namespace Core::IO
{
  /** Source of the input field data. */
  enum class InputFieldSource : std::uint8_t
  {
    separate_file,
    from_mesh
  };

  /** The basis on which the field data is defined */
  enum class FieldDataBasis : std::uint8_t
  {
    cells,
    points
  };



  struct InputFieldRegistry;
  struct MeshDataInputFieldRegistry;

  /**
   * Refer to an input field by a name. This name is used to look up the input field in a registry
   * of known fields.
   */
  struct InputFieldReference
  {
    //! The name which is used to uniquely identify this input field.
    std::string ref_name;
    InputFieldRegistry* registry;
  };


  struct InputFieldRegistry
  {
    using InitFunction = std::function<void(const std::filesystem::path&, const std::string&)>;
    using RedistributeFunction = std::function<void(const Core::LinAlg::Map&)>;
    struct SetupFunctions
    {
      //! Functions to initialize the field with data from the discretization.
      //! These are type-erased but we know the InputField they operate on, so they can be
      //! unregistered again.
      std::unordered_map<void*, InitFunction> init_functions;
      std::unordered_map<void*, RedistributeFunction> redistribute_functions;
    };
    std::unordered_map<std::string, SetupFunctions> fields;

    /**
     * @brief Register a reference to a field with the given @p ref_name. Repeated calls with the
     * same @p ref_name will return the same reference.
     */
    [[nodiscard]] InputFieldReference register_field_reference(const std::string& ref_name);

    /**
     * Associate an InputField with a reference @p ref. The @p init function should later be called
     * with the target map to initialize the field with data from the discretization.
     *
     * @note There should not be any need to call this function directly, as it is used by the
     * InputField internally.
     */
    void attach_input_field(InputFieldReference ref, InitFunction init,
        RedistributeFunction redistribute, void* field_ptr);

    /**
     * Detach an InputField from a reference @p ref. This will remove the @p field_ptr from the
     * list of init functions for the given reference.
     *
     * @note There should not be any need to call this function directly, as it is used by the
     * InputField internally.
     */
    void detach_input_field(InputFieldReference ref, void* field_ptr);
  };

  /**
   * @brief Get the global InputFieldRegistry instance.
   *
   * The standard input mechanism of 4C will automatically register input fields in this registry.
   */
  InputFieldRegistry& global_input_field_registry();


  /**
   * Mesh data reference. This name is to used initialize the field from the mesh data.
   */
  struct MeshDataReference
  {
    //! The name which is used to uniquely identify this mesh data.
    std::string ref_name;
    MeshDataInputFieldRegistry* registry;
  };

  struct MeshDataInputFieldRegistry
  {
    using InitFunction = std::function<void(
        const FE::Discretization&, const MeshInput::Mesh<3>&, FieldDataBasis, const std::string&)>;
    using RedistributeFunction = std::function<void(const Core::LinAlg::Map&)>;
    struct SetupFunctions
    {
      //! Functions to initialize the field with data from the discretization.
      //! These are type-erased but we know the InputField they operate on, so they can be
      //! unregistered again.
      std::unordered_map<void*, InitFunction> init_functions;
      std::unordered_map<void*, RedistributeFunction> redistribute_functions;
    };
    std::unordered_map<std::string, SetupFunctions> fields;

    /**
     * @brief Register a reference to a mesh data input field with the given @p ref_name. Repeated
     * calls with the same @p ref_name will return the same reference.
     */
    [[nodiscard]] MeshDataReference register_field_reference(const std::string& ref_name);

    /**
     * Associate an InputField with a mesh data reference @p ref. The @p init function should later
     * be called to initialize the data, and the redistribute function should be called with the
     * target map to initialize the field with data from the discretization.
     *
     * @note There should not be any need to call this function directly, as it is used by the
     * InputField internally.
     */
    void attach_input_field(MeshDataReference ref, InitFunction init,
        RedistributeFunction redistribute, void* field_ptr);

    /**
     * Detach an InputField from a mesh data reference @p ref. This will remove the @p field_ptr
     * from the list of init functions for the given reference.
     *
     * @note There should not be any need to call this function directly, as it is used by the
     * InputField internally.
     */
    void detach_input_field(MeshDataReference ref, void* field_ptr);
  };

  /**
   * @brief Get the global MeshDataInputFieldRegistry instance.
   *
   * The standard input mechanism of 4C will automatically register mesh data input fields in this
   * registry.
   */
  MeshDataInputFieldRegistry& global_mesh_data_input_field_registry();


  /**
   * A structure to hold point-wise data along with a pointer to the discretization needed to obtain
   * the shape functions used to interpolate the data.
   */
  template <typename IndexType, typename T>
  struct PointDataMap
  {
    /// A map holding the point-wise data
    std::unordered_map<IndexType, T> map{};

    /// A pointer to the underlying discretization used to interpolate the data
    const Core::FE::Discretization* discretization = nullptr;
  };

  namespace Internal
  {
    struct NoInterpolation
    {
    };

    template <typename T, typename Interpolation>
    class GeneralizedInputField
    {
      static constexpr bool requires_interpolation =
          !std::is_same_v<Interpolation, NoInterpolation>;

     public:
      using IndexType = int;
      using MapType = std::unordered_map<IndexType, T>;
      using PointMapType = PointDataMap<IndexType, T>;
      using StorageType = std::conditional_t<requires_interpolation,
          std::variant<T, MapType, PointMapType>, std::variant<T, MapType>>;

      /**
       * Default constructor. This GeneralizedInputField will not hold any data and will throw an
       * error if any attempt is made to access its data. You need to assign a value to it before
       * using it.
       */
      GeneralizedInputField() = default;

      /**
       * Construct an GeneralizedInputField from a single @p const_data. This field will have the
       * same value for every element index.
       */
      explicit GeneralizedInputField(T const_data) : data_(std::move(const_data)) {}

      /**
       * Construct an GeneralizedInputField from a map of element-wise data. The @p data map
       * contains element indices as keys and the corresponding values of type T. Element indices
       * are expected to be one-based and are converted to zero-based indices, which is what the
       * Discretization expects internally.
       */
      explicit GeneralizedInputField(std::unordered_map<IndexType, T> data)
      {
        make_index_zero_based(data);
        data_ = std::move(data);
      }

      /**
       * Construct an GeneralizedInputField that refers to a centrally registered field. The
       * necessary @p ref may be obtained by calling the
       * InputFieldRegistry::register_field_reference() function. The resulting
       * GeneralizedInputField will not be usable until the reference is set up and redistributed
       * with the desired target map. When using the global input field registry and 4C's standard
       * main function this is done automatically.
       */
      explicit GeneralizedInputField(InputFieldReference ref) : ref_(ref)
      {
        // empty initialize the internal data
        data_.template emplace<std::unordered_map<IndexType, T>>();

        ref.registry->attach_input_field(ref,
            std::bind_front(&GeneralizedInputField::initialize_from_file, this),
            std::bind_front(&GeneralizedInputField::redistribute, this), this);
      }

      /**
       * Construct an GeneralizedInputField that refers to field data defined in the input mesh. The
       * provided
       * @p ref identifies the mesh data field to be used as a source for this
       * GeneralizedInputField.
       *
       * @note The resulting GeneralizedInputField will not be usable until the reference is
       * initialized with the mesh data using @p initialize_from_mesh_data and redistributed with
       * the desired target map using @p redistribute. When using the global mesh data input field
       * registry and 4C's standard main function this is done automatically.
       */
      explicit GeneralizedInputField(MeshDataReference ref) : ref_(ref)
      {
        // empty initialize the internal data
        data_.template emplace<std::unordered_map<IndexType, T>>();

        ref.registry->attach_input_field(ref,
            std::bind_front(&GeneralizedInputField::initialize_from_mesh_data, this),
            std::bind_front(&GeneralizedInputField::redistribute, this), this);
      }

      /**
       * @{
       * Special member functions.
       */
      ~GeneralizedInputField();
      GeneralizedInputField(const GeneralizedInputField& other);
      GeneralizedInputField& operator=(const GeneralizedInputField& other);
      GeneralizedInputField(GeneralizedInputField&& other) noexcept;
      GeneralizedInputField& operator=(GeneralizedInputField&& other) noexcept;
      /** @} */

      /**
       * Redistribute the GeneralizedInputField such that its data is distributed to the given @p
       * target_map. This is a collective operation and must be called on all ranks.
       */
      void redistribute(const Core::LinAlg::Map& target_map);

      /**
       * Access the value of the field for the given @p element index. The @p element_id
       * is not checked for validity and asking for an invalid index will lead to undefined
       * behavior. Use the `at()` function if you want to check for validity.
       *
       * @note Not available for interpolated fields
       */
      template <typename = void>
        requires(!requires_interpolation)
      [[nodiscard]] const T& operator[](IndexType element_id) const
      {
        return get(element_id, false);
      }

      /**
       * Access the value of the field for the given @p element index. If the @p element_id
       * is not a valid index, this function will throw an error which contains the optional @p
       *
       * @note Not available for interpolated fields
       */
      template <typename = void>
        requires(!requires_interpolation)
      [[nodiscard]] const T& at(
          IndexType element_id, std::string_view field_name = "unknown field") const
      {
        return get(element_id, true, field_name);
      }

      /**
       * Interpolate the value of the field to a given point inside an element. The point
       * coordinates
       * @p xi are given in the local coordinates of the element. The @p element_id is the index of
       * the element in which the point lies. If the field is not point-based, this function will
       * return the value associated with the element. If the field is point-based, the value will
       * be interpolated using the provided @p xi coordinates, the shape functions of the element
       * and the template parameter of the interpolator. The optional @p field_name is only used for
       * error messages.
       *
       * @note The @p xi coordinates are expected to be in the reference element of the element's
       * cell type.
       *
       * @note Only available for interpolated fields
       */
      template <typename = void>
        requires(requires_interpolation)
      [[nodiscard]] T interpolate(IndexType element_id, std::span<const double> xi,
          std::string_view field_name = "unknown field") const
      {
        FOUR_C_ASSERT(xi.size() == 3,
            "xi must have exactly 3 components (1d/2d elements use zero-padded coordinates)");

        if (const T* data = std::get_if<T>(&data_))
        {
          return *data;
        }
        if (const MapType* map = std::get_if<MapType>(&data_))
        {
#ifdef FOUR_C_ENABLE_ASSERTIONS
          validate_field_reference(*map);
#endif
          auto it = map->find(element_id);

          FOUR_C_ASSERT_ALWAYS(it != map->end(),
              "Element index {} not found in InterpolatedInputField '{}'.", element_id + 1,
              field_name);

          return it->second;
        }
        if (const PointMapType* map = std::get_if<PointMapType>(&data_))
        {
#ifdef FOUR_C_ENABLE_ASSERTIONS
          validate_field_reference(map->map);
#endif
          FOUR_C_ASSERT(map->discretization,
              "Discretization is not set for point-based InterpolatedInputField! Cannot "
              "interpolate data to "
              "specified point.");

          const Elements::Element& ele = *map->discretization->g_element(element_id);

          const auto values = std::span(ele.nodes(), ele.num_node()) |
                              std::views::transform(
                                  [&](const Nodes::Node* node)
                                  {
                                    auto it = map->map.find(node->id());
                                    FOUR_C_ASSERT_ALWAYS(it != map->map.end(),
                                        "Point index {} not found in InterpolatedInputField '{}'.",
                                        node->id(), field_name);
                                    return it->second;
                                  });

          // Avoid dynamic memory allocation for small elements by using a stack array (up to 27
          // points per element)
          constexpr std::size_t MAX_NODES_ON_STACK = 27;
          std::array<double, MAX_NODES_ON_STACK> weights_on_stack{};
          std::vector<double> weights_on_heap;
          double* weights_ptr = nullptr;
          if (std::cmp_less_equal(ele.num_node(), MAX_NODES_ON_STACK))
          {
            weights_ptr = weights_on_stack.data();
          }
          else
          {
            weights_on_heap.resize(ele.num_node());
            weights_ptr = weights_on_heap.data();
          }
          std::span<double> weights(weights_ptr, ele.num_node());
          switch (Core::FE::get_dimension(ele.shape()))
          {
            case 1:
              Core::FE::shape_function_1d(weights, xi[0], ele.shape());
              break;
            case 2:
              Core::FE::shape_function_2d(weights, xi[0], xi[1], ele.shape());
              break;
            case 3:
              Core::FE::shape_function_3d(weights, xi[0], xi[1], xi[2], ele.shape());
              break;
            default:
              FOUR_C_THROW("Element shape '{}' has invalid dimension for input field interpolation",
                  ele.shape());
          }

          return Interpolation{}(weights, values);
        }
        std23::unreachable();
      }

     private:
      //! Internal getter which can optionally check for the validity of the element index.
      template <typename = void>
        requires(!requires_interpolation)
      const T& get(
          IndexType element_id, bool check, std::string_view field_name = "unknown field") const
      {
        if (const T* data = std::get_if<T>(&data_))
        {
          return *data;
        }
        if (const MapType* map = std::get_if<MapType>(&data_))
        {
#ifdef FOUR_C_ENABLE_ASSERTIONS
          validate_field_reference(*map);
#endif
          auto it = map->find(element_id);
          if (check)
          {
            FOUR_C_ASSERT_ALWAYS(it != map->end(), "Element index {} not found in InputField '{}'.",
                element_id + 1, field_name);
          }
          return it->second;
        }
        std23::unreachable();
      }

      void make_index_zero_based(MapType& map)
      {
        MapType new_map;
        for (auto&& [index, value] : map)
        {
          if (index < 1)
            FOUR_C_THROW("InputField index {} is less than 1. All indices must be >= 1.", index);
          new_map[index - 1] = std::move(value);
        }
        map = std::move(new_map);
      }

      /*!
       * @brief Initialize the InputField from a file.
       *
       * @note The data is only read on rank 0. It relies on calling redistribute() later to
       * distribute the data to their respective ranks.
       */
      void initialize_from_file(const std::filesystem::path& source_file, const std::string& key)
      {
        MapType& map = std::get<MapType>(data_);
        IO::read_value_from_yaml(source_file, key, map);
        make_index_zero_based(map);
      }

      /*!
       * @brief Initialize the InputField from mesh data.
       *
       * @note The data is only read on rank 0. It relies on calling redistribute() later to
       * distribute the data to their respective ranks.
       */
      void initialize_from_mesh_data(const FE::Discretization& discretization,
          const MeshInput::Mesh<3>& mesh, FieldDataBasis basis, const std::string& key)
      {
        switch (basis)
        {
          case FieldDataBasis::cells:
          {
            MapType& map = data_.template emplace<MapType>();
            MeshInput::read_value_from_cell_data(mesh, key, map);
            break;
          }
          case FieldDataBasis::points:
          {
            if constexpr (!requires_interpolation)
            {
              FOUR_C_THROW(
                  "InputField can only hold constant or element-wise data. You are trying "
                  "to initialize it from point-based mesh data.");
            }
            else
            {
              PointMapType& map = data_.template emplace<PointMapType>(
                  PointMapType{.discretization = &discretization});
              MeshInput::read_value_from_point_data(mesh, key, map.map);
            }
            break;
          }
          default:
            FOUR_C_THROW("Unknown FieldDataBasis");
        }
      }

      /**
       * Validate whether the field reference is set up correctly on this rank.
       */
      void validate_field_reference(const std::unordered_map<IndexType, T>& map) const
      {
        if (map.empty())
        {
          std::visit(
              [](auto& ref)
              {
                if constexpr (!std::is_same_v<std::decay_t<decltype(ref)>, std::monostate>)
                {
                  if (ref.registry == nullptr)
                  {
                    FOUR_C_THROW("No registry assigned to this reference-type input field.");
                  }
                  else
                  {
                    FOUR_C_THROW(
                        "Accessing a value on an empty reference-type InputField on this "
                        "processor. Probably, InputField is not set up and distributed across "
                        "ranks. Initialize and redistribute it first.");
                  }
                }
              },
              ref_);
        }
      }

      StorageType data_;

      //! Reference to the input field registry, if this GeneralizedInputField is a field reference.
      std::variant<std::monostate, InputFieldReference, MeshDataReference> ref_{};
    };

    template <typename T, typename Interpolation>
    GeneralizedInputField<T, Interpolation>::~GeneralizedInputField()
    {
      // If this GeneralizedInputField is a reference, we need to detach it from the registry.
      if (auto* ref = std::get_if<InputFieldReference>(&ref_))
      {
        if (ref->registry) ref->registry->detach_input_field(*ref, this);
      }
      else if (auto* ref = std::get_if<MeshDataReference>(&ref_))
      {
        if (ref->registry) ref->registry->detach_input_field(*ref, this);
      }
    }

    template <typename T, typename Interpolation>
    GeneralizedInputField<T, Interpolation>::GeneralizedInputField(
        const GeneralizedInputField& other)
        : data_(other.data_), ref_(other.ref_)
    {
      // If this InputField is a reference, we need to reattach it to the registry.
      if (auto* ref = std::get_if<InputFieldReference>(&ref_))
      {
        if (ref->registry)
        {
          ref->registry->attach_input_field(*ref,
              std::bind_front(&GeneralizedInputField::initialize_from_file, this),
              std::bind_front(&GeneralizedInputField::redistribute, this), this);
        }
      }
      else if (auto* ref = std::get_if<MeshDataReference>(&ref_))
      {
        if (ref->registry)
        {
          ref->registry->attach_input_field(*ref,
              std::bind_front(&GeneralizedInputField::initialize_from_mesh_data, this),
              std::bind_front(&GeneralizedInputField::redistribute, this), this);
        }
      }
    }

    template <typename T, typename Interpolation>
    GeneralizedInputField<T, Interpolation>& GeneralizedInputField<T, Interpolation>::operator=(
        const GeneralizedInputField& other)
    {
      data_ = other.data_;
      ref_ = other.ref_;
      // If this GeneralizedInputField is a reference, we need to reattach it to the registry.
      if (auto* ref = std::get_if<InputFieldReference>(&ref_))
      {
        if (ref->registry)
        {
          ref->registry->attach_input_field(*ref,
              std::bind_front(&GeneralizedInputField::initialize_from_file, this),
              std::bind_front(&GeneralizedInputField::redistribute, this), this);
        }
      }
      else if (auto* ref = std::get_if<MeshDataReference>(&ref_))
      {
        if (ref->registry)
        {
          ref->registry->attach_input_field(*ref,
              std::bind_front(&GeneralizedInputField::initialize_from_mesh_data, this),
              std::bind_front(&GeneralizedInputField::redistribute, this), this);
        }
      }
      return *this;
    }

    template <typename T, typename Interpolation>
    GeneralizedInputField<T, Interpolation>::GeneralizedInputField(
        GeneralizedInputField&& other) noexcept
        : data_(std::move(other.data_)), ref_(std::move(other.ref_))
    {
      // If this GeneralizedInputField is a reference, we need to reattach it to the registry.
      if (auto* ref = std::get_if<InputFieldReference>(&ref_))
      {
        if (ref->registry)
        {
          ref->registry->detach_input_field(*ref, &other);
          ref->registry->attach_input_field(*ref,
              std::bind_front(&GeneralizedInputField::initialize_from_file, this),
              std::bind_front(&GeneralizedInputField::redistribute, this), this);
        }
      }
      else if (auto* ref = std::get_if<MeshDataReference>(&ref_))
      {
        if (ref->registry)
        {
          ref->registry->detach_input_field(*ref, &other);
          ref->registry->attach_input_field(*ref,
              std::bind_front(&GeneralizedInputField::initialize_from_mesh_data, this),
              std::bind_front(&GeneralizedInputField::redistribute, this), this);
        }
      }
    }


    template <typename T, typename Interpolation>
    GeneralizedInputField<T, Interpolation>& GeneralizedInputField<T, Interpolation>::operator=(
        GeneralizedInputField&& other) noexcept
    {
      data_ = std::move(other.data_);
      ref_ = std::move(other.ref_);
      // If this GeneralizedInputField is a reference, we need to reattach it to the registry.
      if (auto* ref = std::get_if<InputFieldReference>(&ref_))
      {
        if (ref->registry)
        {
          ref->registry->detach_input_field(*ref, &other);
          ref->registry->attach_input_field(*ref,
              std::bind_front(&GeneralizedInputField::initialize_from_file, this),
              std::bind_front(&GeneralizedInputField::redistribute, this), this);
        }
      }
      else if (auto* ref = std::get_if<MeshDataReference>(&ref_))
      {
        if (ref->registry)
        {
          ref->registry->detach_input_field(*ref, &other);
          ref->registry->attach_input_field(*ref,
              std::bind_front(&GeneralizedInputField::initialize_from_mesh_data, this),
              std::bind_front(&GeneralizedInputField::redistribute, this), this);
        }
      }
      return *this;
    }


    template <typename T, typename Interpolation>
    void GeneralizedInputField<T, Interpolation>::redistribute(const Core::LinAlg::Map& target_map)
    {
      auto& map = [&]() -> MapType&
      {
        if (std::holds_alternative<MapType>(data_))
        {
          return std::get<MapType>(data_);
        }

        if constexpr (requires_interpolation)
        {
          if (std::holds_alternative<PointMapType>(data_))
          {
            return std::get<PointMapType>(data_).map;
          }
        }
        FOUR_C_THROW("Internal error: We expect that this input field internally holds a map!");
      }();

      // Generate the source map from the stored map
      std::vector<int> local_indices;
      local_indices.reserve(map.size());
      for (const auto& [index, _] : map)
      {
        local_indices.push_back(index);
      }

      MPI_Comm comm = target_map.get_comm();
      Core::LinAlg::Map source_map(-1, local_indices.size(), local_indices.data(), 0, comm);
      Communication::Exporter exporter(source_map, target_map, comm);
      exporter.do_export(map);
    }
  }  // namespace Internal


  /**
   * @brief A class to represent an element-wise or constant input parameter field.
   *
   * In its current form, this class can either hold a single value of type T or a map of
   * element-wise values of type T. It can be accessed via the `operator[]` or `at()` functions with
   * global element ids. If you wish to interpolate the field with a cell, see
   * InterpolatedInputField.
   */
  template <typename T>
  using InputField = Internal::GeneralizedInputField<T, Internal::NoInterpolation>;

  /**
   * A concept to check whether a given Functional can be used to interpolate input field data.
   */
  template <typename Functional, typename T>
  concept InputFieldInterpolator = requires() {
    {
      // The functional needs to be
      //   * default constructible
      //   * callable with two ranges:
      //     * a range of double weights
      //     * a view-range of values of type T
      // and must result in a type convertible to T
      Functional{}(std::declval<std::vector<double>>(),
          std::declval<std::vector<T>>() | std::ranges::views::transform(std::identity()))
    } -> std::convertible_to<T>;
  };

  /*!
   * @brief A simple component-wise interpolator based on the shape-functions.
   *
   * @tparam T
   */
  template <typename T>
  struct ComponentInterpolator
  {
    template <typename WeightRange, typename ValueRange>
    T operator()(WeightRange weights, ValueRange values) const
    {
      FOUR_C_ASSERT(std::ranges::size(weights) == std::ranges::size(values),
          "Weights and values must have the same length.");

      return std::inner_product(weights.begin(), weights.end(), values.begin(), T{});
    }
  };

  /**
   * @brief A class to represent an interpolated input parameter field.
   *
   * In its current form, this class can either hold a single value of type T, a map of
   * element-wise values of type T, or point-wise data of type T. It can be accessed via the
   * @p interpolate() function.
   */
  template <typename T, InputFieldInterpolator<T> Interpolation = ComponentInterpolator<T>>
  using InterpolatedInputField = Internal::GeneralizedInputField<T, Interpolation>;
}  // namespace Core::IO

FOUR_C_NAMESPACE_CLOSE

#endif

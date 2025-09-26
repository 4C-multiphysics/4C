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
#include "4C_comm_mpi_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_io_mesh.hpp"
#include "4C_io_yaml.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_std23_unreachable.hpp"

#include <filesystem>
#include <unordered_map>
#include <utility>
#include <variant>

FOUR_C_NAMESPACE_OPEN

namespace Core::IO
{
  /** Source of the input field data. */
  enum class InputFieldSource
  {
    separate_file,
    from_mesh
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
   * Mesh data reference. This name is used initialize the field from the mesh data.
   */
  struct MeshDataReference
  {
    //! The name which is used to uniquely identify this mesh data.
    std::string ref_name;
    MeshDataInputFieldRegistry* registry;
  };

  struct MeshDataInputFieldRegistry
  {
    using InitFunction = std::function<void(const MeshInput::Mesh<3>&, const std::string&)>;
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
   * @brief A class to represent an input parameter field.
   *
   * In its current form, this class can either hold a single value of type T or a map of
   * element-wise values of type T.
   */
  template <typename T>
  class InputField
  {
   public:
    using IndexType = int;
    using MapType = std::unordered_map<IndexType, T>;
    using StorageType = std::variant<T, MapType>;

    /**
     * Default constructor. This InputField will not hold any data and will throw an error if
     * any attempt is made to access its data. You need to assign a value to it before using it.
     */
    InputField() = default;

    /**
     * Construct an InputField from a single @p const_data. This field will have the same value
     * for every element index.
     */
    explicit InputField(T const_data) : data_(std::move(const_data)) {}

    /**
     * Construct an InputField from a map of element-wise data. The @p data map contains
     * element indices as keys and the corresponding values of type T. Element indices are expected
     * to be one-based and are converted to zero-based indices, which is what the Discretization
     * expects internally.
     */
    explicit InputField(std::unordered_map<IndexType, T> data)
    {
      make_index_zero_based(data);
      data_ = std::move(data);
    }

    /**
     * Construct an InputField that refers to a centrally registered field. The necessary @p ref
     * may be obtained by calling the InputFieldRegistry::register_field_reference() function.
     * The resulting InputField will not be usable until the reference is set up and redistributed
     * with the desired target map. When using the global input field registry and 4C's standard
     * main function this is done automatically.
     */
    explicit InputField(InputFieldReference ref) : ref_(ref)
    {
      // empty initialize the internal data
      data_.template emplace<std::unordered_map<IndexType, T>>();

      ref.registry->attach_input_field(ref,
          std::bind_front(&InputField::initialize_from_file, this),
          std::bind_front(&InputField::redistribute, this), this);
    }

    /**
     * Construct an InputField that refers to field data defined in the input mesh. ...
     */
    explicit InputField(MeshDataReference ref) : ref_(ref)
    {
      // empty initialize the internal data
      data_.template emplace<std::unordered_map<IndexType, T>>();

      ref.registry->attach_input_field(ref,
          std::bind_front(&InputField::initialize_from_mesh_data, this),
          std::bind_front(&InputField::redistribute, this), this);
    }

    /**
     * @{
     * Special member functions.
     */
    ~InputField();
    InputField(const InputField& other);
    InputField& operator=(const InputField& other);
    InputField(InputField&& other) noexcept;
    InputField& operator=(InputField&& other) noexcept;
    /** @} */

    /**
     * Redistribute the InputField such that its data is distributed to the given @p target_map.
     * This is a collective operation and must be called on all ranks.
     */
    void redistribute(const Core::LinAlg::Map& target_map);

    /**
     * Access the value of the field for the given @p element index. The @p element_id
     * is not checked for validity and asking for an invalid index will lead to undefined behavior.
     * Use the `at()` function if you want to check for validity.
     */
    [[nodiscard]] const T& operator[](IndexType element_id) const { return get(element_id, false); }

    /**
     * Access the value of the field for the given @p element index. If the @p element_id
     * is not a valid index, this function will throw an error which contains the optional @p
     * field_name for informational purposes.
     */
    [[nodiscard]] const T& at(
        IndexType element_id, std::string_view field_name = "unknown field") const
    {
      return get(element_id, true, field_name);
    }

   private:
    //! Internal getter which can optionally check for the validity of the element index.
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
        if (map->empty())
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
    void initialize_from_mesh_data(const MeshInput::Mesh<3>& mesh, const std::string& key)
    {
      MapType& map = std::get<MapType>(data_);
      MeshInput::read_value_from_element_data(mesh, key, map);
    }

    StorageType data_;

    //! Reference to the input field registry, if this InputField is a field reference.
    std::variant<std::monostate, InputFieldReference, MeshDataReference> ref_{};
  };

  template <typename T>
  InputField<T>::~InputField()
  {
    // If this InputField is a reference, we need to detach it from the registry.
    if (auto* ref = std::get_if<InputFieldReference>(&ref_))
    {
      if (ref->registry) ref->registry->detach_input_field(*ref, this);
    }
    else if (auto* ref = std::get_if<MeshDataReference>(&ref_))
    {
      if (ref->registry) ref->registry->detach_input_field(*ref, this);
    }
  }

  template <typename T>
  InputField<T>::InputField(const InputField& other) : data_(other.data_), ref_(other.ref_)
  {
    // If this InputField is a reference, we need to reattach it to the registry.
    if (auto* ref = std::get_if<InputFieldReference>(&ref_))
    {
      if (ref->registry)
      {
        ref->registry->attach_input_field(*ref,
            std::bind_front(&InputField::initialize_from_file, this),
            std::bind_front(&InputField::redistribute, this), this);
      }
    }
    else if (auto* ref = std::get_if<MeshDataReference>(&ref_))
    {
      if (ref->registry)
      {
        ref->registry->attach_input_field(*ref,
            std::bind_front(&InputField::initialize_from_mesh_data, this),
            std::bind_front(&InputField::redistribute, this), this);
      }
    }
  }

  template <typename T>
  InputField<T>& InputField<T>::operator=(const InputField& other)
  {
    data_ = other.data_;
    ref_ = other.ref_;
    // If this InputField is a reference, we need to reattach it to the registry.
    if (auto* ref = std::get_if<InputFieldReference>(&ref_))
    {
      if (ref->registry)
      {
        ref->registry->attach_input_field(*ref,
            std::bind_front(&InputField::initialize_from_file, this),
            std::bind_front(&InputField::redistribute, this), this);
      }
    }
    else if (auto* ref = std::get_if<MeshDataReference>(&ref_))
    {
      if (ref->registry)
      {
        ref->registry->attach_input_field(*ref,
            std::bind_front(&InputField::initialize_from_mesh_data, this),
            std::bind_front(&InputField::redistribute, this), this);
      }
    }
    return *this;
  }

  template <typename T>
  InputField<T>::InputField(InputField&& other) noexcept
      : data_(std::move(other.data_)), ref_(std::move(other.ref_))
  {
    // If this InputField is a reference, we need to reattach it to the registry.
    if (auto* ref = std::get_if<InputFieldReference>(&ref_))
    {
      if (ref->registry)
      {
        ref->registry->detach_input_field(*ref, &other);
        ref->registry->attach_input_field(*ref,
            std::bind_front(&InputField::initialize_from_file, this),
            std::bind_front(&InputField::redistribute, this), this);
      }
    }
    else if (auto* ref = std::get_if<MeshDataReference>(&ref_))
    {
      if (ref->registry)
      {
        ref->registry->detach_input_field(*ref, &other);
        ref->registry->attach_input_field(*ref,
            std::bind_front(&InputField::initialize_from_mesh_data, this),
            std::bind_front(&InputField::redistribute, this), this);
      }
    }
  }


  template <typename T>
  InputField<T>& InputField<T>::operator=(InputField&& other) noexcept
  {
    data_ = std::move(other.data_);
    ref_ = std::move(other.ref_);
    // If this InputField is a reference, we need to reattach it to the registry.
    if (auto* ref = std::get_if<InputFieldReference>(&ref_))
    {
      if (ref->registry)
      {
        ref->registry->detach_input_field(*ref, &other);
        ref->registry->attach_input_field(*ref,
            std::bind_front(&InputField::initialize_from_file, this),
            std::bind_front(&InputField::redistribute, this), this);
      }
    }
    else if (auto* ref = std::get_if<MeshDataReference>(&ref_))
    {
      if (ref->registry)
      {
        ref->registry->detach_input_field(*ref, &other);
        ref->registry->attach_input_field(*ref,
            std::bind_front(&InputField::initialize_from_mesh_data, this),
            std::bind_front(&InputField::redistribute, this), this);
      }
    }
    return *this;
  }


  template <typename T>
  void InputField<T>::redistribute(const Core::LinAlg::Map& target_map)
  {
    FOUR_C_ASSERT((std::holds_alternative<std::unordered_map<IndexType, T>>(data_)),
        "Internal error: We expect that this input field internally holds a map!");
    auto& map = std::get<std::unordered_map<IndexType, T>>(data_);

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
}  // namespace Core::IO

FOUR_C_NAMESPACE_CLOSE

#endif

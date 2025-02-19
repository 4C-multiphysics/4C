// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_IO_VISUALIZATION_DATA_HPP
#define FOUR_C_IO_VISUALIZATION_DATA_HPP

#include "4C_config.hpp"

#include "4C_utils_exceptions.hpp"

#include <map>
#include <set>
#include <string>
#include <variant>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::IO
{
  //! Type to be used for global index and offset values
  using index_type = int32_t;

  //! Since the output data can come in different types, we use this variant type which allows us
  //! to put all data vectors in a single map
  using visualization_vector_type_variant = std::variant<std::vector<int>, std::vector<double>>;

  //! The polyhedral cell type requires special treatment. This cell has the vtk cell type 42. We
  //! define this variable here to be used in 4C when referring to that cell type.
  static constexpr int polyhedron_cell_type = 42;

  namespace Internal
  {
    /**
     * @brief Return a string representation of the supported scalar types
     *
     * @tparam scalar_type type of the scalar values
     * @return std::string representation of the type name
     */
    template <typename ScalarType>
    std::string scalar_type_to_string()
    {
      FOUR_C_THROW("The scalar type in scalar_type_to_string is unknown");
    }
    template <>
    inline std::string scalar_type_to_string<int>()
    {
      return "int";
    }
    template <>
    inline std::string scalar_type_to_string<double>()
    {
      return "double";
    }
  }  // namespace Internal


  /**
   * @brief This class holds data in the VTK Unstructured Grid (vtu) data format.
   *
   * The data format follows https://vtk.org/doc/nightly/html/classvtkUnstructuredGrid.html
   *
   * This implementation allows for convenient ways to define the connectivity data for cell and
   * face topologies. For more detail, have a look at the documentation of the methods
   * \sa complete_cell_connectivity() and \sa complete_face_connectivity().
   */
  class VisualizationData
  {
   public:
    /**
     * @brief Default constructor
     */
    VisualizationData() = default;

    /**
     * @brief Return a const reference to the point coordinates
     */
    [[nodiscard]] const std::vector<double>& get_point_coordinates() const
    {
      return point_coordinates_;
    }

    /**
     * @brief Return a mutable reference to the point coordinates
     *
     * @param n (in) If this is larger than 0, additional entries will be reserved
     */
    [[nodiscard]] std::vector<double>& get_point_coordinates(const size_t n = 0)
    {
      return additional_reserve(point_coordinates_, n);
    }

    /**
     * @brief Return the number of points in this container
     */
    [[nodiscard]] size_t get_point_coordinates_number_of_points() const;

    /**
     * @brief Return a const reference to the cell types
     */
    [[nodiscard]] const std::vector<uint8_t>& get_cell_types() const { return cell_types_; }

    /**
     * @brief Return a mutable reference to the cell types
     *
     * @param n (in) If this is larger than 0, additional entries will be reserved
     */
    [[nodiscard]] std::vector<uint8_t>& get_cell_types(const size_t n = 0)
    {
      return additional_reserve(cell_types_, n);
    }

    /**
     * @brief Return a const reference to the cell connectivity
     */
    [[nodiscard]] const std::vector<index_type>& get_cell_connectivity() const
    {
      return cell_connectivity_;
    }

    /**
     * @brief Return a mutable reference to the cell connectivity
     *
     * @param n (in) If this is larger than 0, additional entries will be reserved
     */
    [[nodiscard]] std::vector<index_type>& get_cell_connectivity(const size_t n = 0)
    {
      return additional_reserve(cell_connectivity_, n);
    }

    /**
     * @brief Return a const reference to the cell offsets
     */
    [[nodiscard]] const std::vector<index_type>& get_cell_offsets() const { return cell_offsets_; }

    /**
     * @brief Return a mutable reference to the cell offsets
     *
     * @param n (in) If this is larger than 0, additional entries will be reserved
     */
    [[nodiscard]] std::vector<index_type>& get_cell_offsets(const size_t n = 0)
    {
      return additional_reserve(cell_offsets_, n);
    }

    /**
     * @brief Return a const reference to the face connectivity
     */
    [[nodiscard]] const std::vector<index_type>& get_face_connectivity() const
    {
      return face_connectivity_;
    }

    /**
     * @brief Return a mutable reference to the face connectivity
     *
     * @param n (in) If this is larger than 0, additional entries will be reserved
     */
    [[nodiscard]] std::vector<index_type>& get_face_connectivity(const size_t n = 0)
    {
      return additional_reserve(face_connectivity_, n);
    }

    /**
     * @brief Return a const reference to the face offsets
     */
    [[nodiscard]] const std::vector<index_type>& get_face_offsets() const { return face_offsets_; }

    /**
     * @brief Return a mutable reference to the face offsets
     *
     * @param n (in) If this is larger than 0, additional entries will be reserved
     */
    [[nodiscard]] std::vector<index_type>& get_face_offsets(const size_t n = 0)
    {
      return additional_reserve(face_offsets_, n);
    }

    /**
     * @brief Return a const reference to a point data vector
     *
     * @tparam T Expected type of the data
     * @param data_name (in) Name of the data field
     */
    template <typename T>
    [[nodiscard]] const std::vector<T>& get_point_data(const std::string& data_name) const
    {
      return get_data<std::vector<T>>(point_data_, data_name);
    }

    /**
     * @brief Return a mutable reference to a point data vector
     *
     * @tparam T Expected type of the data
     * @param data_name (in) Name of the data field
     * @param n (in) If this is larger than 0, additional entries will be reserved
     */
    template <typename T>
    [[nodiscard]] std::vector<T>& get_point_data(const std::string& data_name, const size_t n = 0)
    {
      // The const_cast can be done here, since we know that the underlying data is not const
      return additional_reserve(
          const_cast<std::vector<T>&>(get_data<std::vector<T>>(point_data_, data_name)), n);
    }

    /**
     * @brief Return a const reference to the variant type of a point data vector
     *
     * @param data_name (in) Name of the data field
     */
    [[nodiscard]] const visualization_vector_type_variant& get_point_data_variant(
        const std::string& data_name) const
    {
      return get_data_vector_from_map_item(get_data_map_item(point_data_, data_name));
    }

    /**
     * @brief Return a const reference to a cell data vector
     *
     * @tparam T Expected type of the data
     * @param data_name (in) Name of the data field
     */
    template <typename T>
    [[nodiscard]] const std::vector<T>& get_cell_data(const std::string& data_name) const
    {
      return get_data<std::vector<T>>(cell_data_, data_name);
    }

    /**
     * @brief Return a mutable reference to a cell data vector
     *
     * @tparam T Expected type of the data
     * @param data_name (in) Name of the data field
     * @param n (in) If this is larger than 0, additional entries will be reserved
     */
    template <typename T>
    [[nodiscard]] std::vector<T>& get_cell_data(const std::string& data_name, const size_t n = 0)
    {
      // The const_cast can be done here, since we know that the underlying data is not const
      return additional_reserve(
          const_cast<std::vector<T>&>(get_data<std::vector<T>>(cell_data_, data_name)), n);
    }

    /**
     * @brief Return a const reference to the variant type of a cell data vector. This is mainly
     * used when the data is written to disk.
     *
     * @param data_name (in) Name of the data field
     */
    [[nodiscard]] const visualization_vector_type_variant& get_cell_data_variant(
        const std::string& data_name) const
    {
      return get_data_vector_from_map_item(get_data_map_item(cell_data_, data_name));
    }

    /**
     * @brief Return a const reference to a field data vector
     *
     * @tparam T Expected type of the data
     * @param data_name (in) Name of the data field
     */
    template <typename T>
    [[nodiscard]] const std::vector<T>& get_field_data(const std::string& data_name) const
    {
      return get_data<std::vector<T>>(field_data_, data_name);
    }

    /**
     * @brief Return a mutable reference to a field data vector
     *
     * @tparam T Expected type of the data
     * @param data_name (in) Name of the data field
     * @param n (in) If this is larger than 0, additional entries will be reserved
     */
    template <typename T>
    [[nodiscard]] std::vector<T>& get_field_data(const std::string& data_name, const size_t n = 0)
    {
      // The const_cast can be done here, since we know that the underlying data is not const
      return additional_reserve(
          const_cast<std::vector<T>&>(get_data<std::vector<T>>(field_data_, data_name)), n);
    }

    /**
     * @brief Return a const reference to the variant type of a field data vector
     *
     * @param data_name (in) Name of the data field
     */
    [[nodiscard]] const visualization_vector_type_variant& get_field_data_variant(
        const std::string& data_name) const
    {
      return get_data_vector_from_map_item(get_data_map_item(field_data_, data_name));
    }

    /**
     * @brief Return a const reference to the map containing the field data
     */
    [[nodiscard]] const std::map<std::string, visualization_vector_type_variant>&
    get_field_data_map() const
    {
      return field_data_;
    }

    /**
     * \brief Register a new empty point data vector to this object
     *
     * @param data_name (in) Name of the new data. This name may not already exist.
     * @param n_dim (in) Number of dimensions of the data. When writing the object, the
     * dimension of this vector should be n_dim * n_points.
     * @param reserve (in) If this is larger than 0, the created vector will allocate the number of
     * entries.
     */
    template <typename T>
    std::vector<T>& register_point_data(
        const std::string& data_name, const unsigned int n_dim, const size_t reserve = 0)
    {
      return register_data_vector<T>(point_data_, data_name, n_dim, reserve);
    }

    /**
     * \brief Register a new empty cell data vector to this object and return the created vector
     *
     * @param data_name (in) Name of the new data. This name may not already exist.
     * @param n_dim (in) Number of dimensions of the data. When writing the object, the
     * dimension of this vector should be n_dim * n_cells.
     * @param reserve (in) If this is larger than 0, the created vector will allocate the number of
     * entries.
     */
    template <typename T>
    std::vector<T>& register_cell_data(
        const std::string& data_name, const unsigned int n_dim, const size_t reserve = 0)
    {
      return register_data_vector<T>(cell_data_, data_name, n_dim, reserve);
    }

    /**
     * \brief Register a new empty field data vector to this object and return the created vector
     *
     * @param data_name (in) Name of the new data. This name may not already exist.
     * @param n_dim (in) Number of dimensions of the data. When writing the object, the
     * dimension of this vector should be n_dim.
     */
    template <typename T>
    std::vector<T>& register_field_data(const std::string& data_name, const unsigned int n_dim = 1)
    {
      if (field_data_.find(data_name) != field_data_.end())
        FOUR_C_THROW(
            "The field data vector with the name \"%s\" you want to register already exists.",
            data_name.c_str());

      std::vector<T> new_vector;
      field_data_[data_name] = new_vector;
      auto& vector = std::get<std::vector<T>>(field_data_[data_name]);
      vector.reserve(n_dim);
      return vector;
    }

    /**
     * @brief Take a given vector and set is as a point data vector
     *
     * @param data_name (in) Name of the point data field, this field can already exist and will be
     * overwritten
     * @param result (in) Data to be set
     * @param n_dim (in) Number of dimensions of the data, i.e., the
     * number of entries in the result vector has to be a multiplicity of n_dim
     */
    template <typename T>
    void set_point_data_vector(
        const std::string& data_name, const std::vector<T>& result, const unsigned int n_dim)
    {
      set_data_vector(point_data_, data_name, n_dim, result);
    }

    /**
     * @brief Take a given vector and set is as a cell data vector
     *
     * @param data_name (in) Name of the cell data field, this field can already exist and will be
     * overwritten
     * @param result (in) Data to be set
     * @param n_dim (in) Number of dimensions of the data, i.e., the
     * number of entries in the result vector has to be a multiplicity of n_dim
     */
    template <typename T>
    void set_cell_data_vector(
        const std::string& data_name, const std::vector<T>& result, const unsigned int n_dim)
    {
      set_data_vector(cell_data_, data_name, n_dim, result);
    }

    /**
     * @brief Take a given vector and set is as a field data vector
     *
     * @param data_name (in) Name of the field data, this field can already exist and will be
     * overwritten
     * @param result (in) Data to be set
     */
    template <typename T>
    void set_field_data_vector(const std::string& data_name, const std::vector<T>& result)
    {
      if (field_data_.find(data_name) != field_data_.end())
      {
        // Check that the dimensions match
        const auto& vector = std::get<std::vector<T>>(field_data_[data_name]);
        if (vector.size() != result.size())
        {
          FOUR_C_THROW(
              "The field array \"%s\" is registered with %d components, you are trying to set it "
              "to %d components, this is not supported.",
              data_name.c_str(), vector.size(), result.size());
        }
      }

      // Set the data
      field_data_[data_name] = result;
    }

    /**
     * @brief Return a std::set with the registered point data names
     */
    [[nodiscard]] std::set<std::string> get_point_data_names() const
    {
      return get_data_names(point_data_);
    }

    /**
     * @brief Return a std::set with the registered cell data names
     */
    [[nodiscard]] std::set<std::string> get_cell_data_names() const
    {
      return get_data_names(cell_data_);
    }

    /**
     * @brief Return a std::set with the registered field data names
     */
    [[nodiscard]] std::set<std::string> get_field_data_names() const
    {
      return get_data_names(field_data_);
    }

    /**
     * @brief Return the size of the point data
     *
     * @param data_name (in) Name of the data
     */
    [[nodiscard]] size_t get_point_data_size(const std::string& data_name) const
    {
      return get_data_vector_size(
          get_data_vector_from_map_item(get_data_map_item(point_data_, data_name)));
    }

    /**
     * @brief Return the size of the cell data
     *
     * @param data_name (in) Name of the data
     */
    [[nodiscard]] size_t get_cell_data_size(const std::string& data_name) const
    {
      return get_data_vector_size(
          get_data_vector_from_map_item(get_data_map_item(cell_data_, data_name)));
    }

    /**
     * @brief Return the dimension of a point data name
     *
     * @param data_name (in) Name of the data
     */
    [[nodiscard]] size_t get_point_data_dimension(const std::string& data_name) const
    {
      return get_data_dimension(get_data_map_item(point_data_, data_name));
    }

    /**
     * @brief Return the dimension of a cell data name
     *
     * @param data_name (in) Name of the data
     */
    [[nodiscard]] size_t get_cell_data_dimension(const std::string& data_name) const
    {
      return get_data_dimension(get_data_map_item(cell_data_, data_name));
    }

    /**
     * @brief Return the dimension of a field data name
     *
     * @param data_name (in) Name of the data
     */
    [[nodiscard]] size_t get_field_data_dimension(const std::string& data_name) const
    {
      return get_data_vector_size(
          get_data_vector_from_map_item(get_data_map_item(field_data_, data_name)));
    }

    /**
     * @brief Clear all data from this container while the registered data names will remain
     */
    void clear_data();

    /**
     * @brief Clear all data from this container, the registered data names will also be removed
     */
    void reset_container();

    /**
     * @brief Complete possibly missing data entries and perform a consistency check
     *
     * This has to be called before writing data to disk that follows the convention described in
     * \sa complete_cell_connectivity and \sa complete_face_connectivity. \sa ConsistencyCheck is
     * also called.
     */
    void consistency_check_and_complete_data();

    /**
     * @brief Perform consistency checks for this data container
     *
     * This check should be called before the data is written to disk to avoid basic inconsistencies
     * that can lead to non-readable visualization files. The following checks are performed:
     * - It is checked that the cell type, connectivity and offset arrays have the correct dimension
     * - It is checked that the face connectivity and offset arrays have the correct dimension
     * - It is checked that the point and cell data vectors have the correct dimension
     */
    void consistency_check() const;

    /**
     * @brief Check and complete the consistency of the cell connectivity and offsets
     *
     * The vtu format requires each cell to have entries in the connectivity vector, i.e., entries
     * that link the cell points to the points in point_coordinates_. The user has to add the
     * cell_offsets_ data, i.e., basically describing the number of points for each cell. The
     * cell_connectivity_ vector is optional, it shall either be completely defined by the user, or
     * left empty. In the case it is left empty, it is assumed that the cells don't share points and
     * that the points for each cell were added to the global points successively. This function
     * then fills up the entries required for the connectivity array.
     */
    void complete_cell_connectivity();

    /**
     * @brief Check and complete the consistency of the face connectivity and offsets
     *
     * Faces are a little bit tricky in vtu. We support polyhedron cells (type == 42). They require
     * face topology data (see https://vtk.org/doc/nightly/html/classvtkPolyhedron.html). If
     * polyhedrons are present, the face_offsets_ and face_connectivity_ vectors have to contain the
     * data describing the faces, in that case each cell needs an entry in the face_offsets_ vector,
     * for standard cells this entry should be -1. If no polyhedrons are present, both face_offsets_
     * and face_connectivity_ should be empty.
     *
     * We support two ways of dealing with polyhedrons:
     * - The user is responsible for adding the full data in face_offsets_ and face_connectivity_
     * - The previous way can be difficult if standard cells are added that are not aware, that
     *   polyhedrons will be in the same data set. In that case it is also supported, that the user
     *   only adds the face_offsets_ and face_connectivity_ for the polyhedrons. This function will
     *   then add the missing -1 entries in face_offsets_ for the standard cells.
     */
    void complete_face_connectivity();

   private:
    /**
     * @brief Return the name of the point or cell data.
     *
     * This function is mainly used for detailed error output.
     */
    [[nodiscard]] std::string get_data_type(
        const std::map<std::string, std::pair<visualization_vector_type_variant, uint8_t>>& data)
        const;

    /**
     * @brief Return the name of the field data.
     *
     * This function is mainly used for detailed error output.
     */
    [[nodiscard]] std::string get_data_type(
        const std::map<std::string, visualization_vector_type_variant>& data) const;

    /**
     * @brief Return the data vector for point and cell data as it is stored in the respective maps,
     * neglecting the number of components
     */
    [[nodiscard]] const visualization_vector_type_variant& get_data_vector_from_map_item(
        const std::pair<visualization_vector_type_variant, uint8_t>& map_item) const
    {
      return map_item.first;
    }

    /**
     * @brief Return the data vector for field data
     *
     * Since the field data vector is stored directly in the map, nothing has to be changed here
     */
    [[nodiscard]] const visualization_vector_type_variant& get_data_vector_from_map_item(
        const visualization_vector_type_variant& map_item) const
    {
      return map_item;
    }

    /**
     * @brief Return the item in a data map, if the name does not exist an error is thrown
     *
     * @param data (in) Data map to look for the item
     * @param data_name (in) Name of the data
     */
    template <typename V>
    [[nodiscard]] const V& get_data_map_item(
        const std::map<std::string, V>& data, const std::string& data_name) const
    {
      if (data.find(data_name) == data.end())
      {
        FOUR_C_THROW("The requested %s field \"%s\" does not exist", get_data_type(data).c_str(),
            data_name.c_str());
      }
      else
      {
        return data.at(data_name);
      }
    }

    /**
     * @brief Return a mutable data object from a given map
     *
     * @param data (in) Data map
     * @param data_name (in) Name of the data
     */
    template <typename T, typename V>
    [[nodiscard]] const T& get_data(const V& data, const std::string& data_name) const
    {
      auto& data_vector = get_data_vector_from_map_item(get_data_map_item(data, data_name));
      try
      {
        return std::get<T>(data_vector);
      }
      catch (const std::bad_variant_access& ex)
      {
        const std::string requested_type =
            "std::vector<" + Internal::scalar_type_to_string<typename T::value_type>() + ">";
        std::string allocated_type = "";
        if (std::holds_alternative<std::vector<double>>(data_vector))
          allocated_type = "std::vector<double>";
        else if (std::holds_alternative<std::vector<int>>(data_vector))
          allocated_type = "std::vector<int>";
        else
          allocated_type = "UNKNOWN";

        FOUR_C_THROW("Requested %s field \"%s\" with type %s, but the allocated type is %s",
            get_data_type(data).c_str(), data_name.c_str(), requested_type.c_str(),
            allocated_type.c_str());
      }
    }

    /**
     * @brief Get all existent names of the given data (keys in the map)
     */
    template <typename V>
    [[nodiscard]] std::set<std::string> get_data_names(const V& data) const
    {
      std::set<std::string> data_names;
      for (const auto& [key, _] : data)
      {
        data_names.insert(key);
      }
      return data_names;
    }

    /**
     * @brief Return the size of the given data vector variant
     */
    template <typename V>
    [[nodiscard]] size_t get_data_vector_size(const V& data_vector) const
    {
      auto size_visitor = [](const auto& data_vector) { return data_vector.size(); };
      return std::visit(size_visitor, data_vector);
    }

    /**
     * @brief Get the number of components for point or cell data
     *
     * @param data_item (in) Item from the point or cell data map
     */
    template <typename V>
    [[nodiscard]] size_t get_data_dimension(V& data_item) const
    {
      return data_item.second;
    }

    /**
     * @brief Register a data name in the corresponding data map and return the created vector
     *
     * @param data (in/out) Data map where the vector shall be registered
     * @param data_name (in) Name of the new data field
     * @param n_dim (in) Number of components
     * @param reserve (in) If this is larger than 0, entries in the created data vector will be
     * reserved
     */
    template <typename T, typename V>
    std::vector<T>& register_data_vector(
        V& data, const std::string& data_name, const unsigned int n_dim, const size_t reserve) const
    {
      if (data.find(data_name) != data.end())
      {
        FOUR_C_THROW("The %s vector with the name \"%s\" you want to add already exists.",
            get_data_type(data).c_str(), data_name.c_str());
      }
      if (n_dim == 0)
      {
        FOUR_C_THROW(
            "You are trying to set the %s vector with the name \"%s\" and n_dim==0, this is not "
            "possible.",
            get_data_type(data).c_str(), data_name.c_str());
      }

      std::vector<T> new_vector;
      data[data_name] = std::make_pair(new_vector, n_dim);
      if (reserve > 0)
      {
        auto reserve_visitor = [&reserve](auto& data_vector) { data_vector.reserve(reserve); };
        std::visit(reserve_visitor, data[data_name].first);
      }
      return std::get<std::vector<T>>(data[data_name].first);
    }

    /**
     * @brief Take a given vector and set is as a data vector
     *
     * @param data (in/out) Data map where the vector shall be set
     * @param data_name (in) Name of the data
     * @param n_dim (in) Number of components
     * @param result (in) Vector containing the data that will be copied to the data map
     */
    template <typename T, typename V>
    void set_data_vector(V& data, const std::string& data_name, const unsigned int n_dim,
        const std::vector<T>& result)
    {
      // Check if data name is already registered
      const auto data_names = get_data_names(data);
      if (data_names.find(data_name) == data_names.end())
      {
        register_data_vector<T>(data, data_name, n_dim, 0);
      }
      else
      {  // Safety check that the number of dimensions is consistent
        const unsigned int data_n_dim = get_data_dimension(get_data_map_item(data, data_name));
        if (data_n_dim != n_dim)
        {
          FOUR_C_THROW("Dimensions do not match");
        }
      }

      // Copy the result vector data to this data container
      // The const_cast can be done here, since we know that the underlying data is not const
      auto& vector = const_cast<std::vector<T>&>(get_data<std::vector<T>>(data, data_name));
      vector = result;
    }

    /**
     * @brief Clear an entry in a data map
     *
     * @param data (in/out) Map where an entry shall be cleared
     * @param data_name (in) Name of the data that shall be cleared
     */
    template <typename V>
    void clear_data(V& data, const std::string& data_name)
    {
      auto clear_visitor = [](auto& data_vector) { data_vector.clear(); };
      // The const_cast can be done here, since we know that the underlying data is not const
      std::visit(
          clear_visitor, const_cast<visualization_vector_type_variant&>(
                             get_data_vector_from_map_item(get_data_map_item(data, data_name))));
    }

    /**
     * @brief Take a vector and reserve additional entries
     *
     * @param vector (in) Data vector
     * @param n (in) If this is larger than 0, additional entries will be reserved
     * @return std::vector<T>& A reference to the vector with additionally reserved entries
     */
    template <typename T>
    [[nodiscard]] static std::vector<T>& additional_reserve(std::vector<T>& vector, const size_t n)
    {
      if (n > 0) vector.reserve(vector.size() + n);
      return vector;
    }

   private:
    // Number of spatial dimensions
    uint8_t n_dim_ = 3;

    //! Vector with point coordinates, the spatial components are stored successively in this
    //! container
    std::vector<double> point_coordinates_;

    //! VTK cell type for each cell
    std::vector<uint8_t> cell_types_;

    //! Point connectivity for the cells
    std::vector<index_type> cell_connectivity_;

    //! Offset for each cell in the cell connectivity array
    std::vector<index_type> cell_offsets_;

    //! Point connectivity for the faces
    std::vector<index_type> face_connectivity_;

    //! Offset for each face in the face connectivity array
    std::vector<index_type> face_offsets_;

    //! Point data, key: result name, entry: (solution data vector, num_components)
    std::map<std::string, std::pair<visualization_vector_type_variant, uint8_t>> point_data_;

    //! Cell data, key: result name, entry: (solution data vector, num_components)
    std::map<std::string, std::pair<visualization_vector_type_variant, uint8_t>> cell_data_;

    //! Field data, key: result name, entry: field data vector (the length of the vector is the
    //! number of components)
    std::map<std::string, visualization_vector_type_variant> field_data_;
  };

}  // namespace Core::IO


FOUR_C_NAMESPACE_CLOSE

#endif

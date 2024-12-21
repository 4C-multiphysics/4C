// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_IO_INPUT_PARAMETER_CONTAINER_HPP
#define FOUR_C_IO_INPUT_PARAMETER_CONTAINER_HPP


#include "4C_config.hpp"

#include "4C_utils_demangle.hpp"
#include "4C_utils_exceptions.hpp"

#include <any>
#include <map>
#include <string>
#include <vector>


FOUR_C_NAMESPACE_OPEN

namespace Core::IO
{
  namespace Internal
  {
    template <typename T>
    const T* try_get_any_data(const std::string& name, const std::any& data)
    {
      if (typeid(T) == data.type())
      {
        const T* any_ptr = std::any_cast<T>(&data);
        FOUR_C_ASSERT(any_ptr != nullptr, "Implementation error.");
        return any_ptr;
      }
      else
      {
        FOUR_C_THROW(
            "You tried to get the data named %s from the container as type '%s'.\n"
            "Actually, it has type '%s'.",
            name.c_str(), Core::Utils::get_type_name<T>().c_str(),
            Core::Utils::try_demangle(data.type().name()).c_str());
      }
    }
  }  // namespace Internal

  /**
   * A container to store dynamic input parameters. The container can store arbitrary types of
   * parameters. Parameters can be grouped in sub-containers.
   *
   * This class is a core part of the input mechanism, as it contains the parsed data from the input
   * file and grants access to it.
   */
  class InputParameterContainer
  {
   public:
    /*!
     * \brief Default constructor.
     */
    InputParameterContainer() = default;

    /*!
     * \brief Destructor.
     */
    virtual ~InputParameterContainer() = default;

    /*!
     * \brief Print this container.
     */
    virtual void print(std::ostream& os) const;


    /*!
     * \brief Add @data to the container at the given key @name.
     *
     * If a record with name name already exists, it will be overwritten.
     */
    template <typename T>
    void add(const std::string& name, const T& data)
    {
      if constexpr (std::is_same_v<T, int>)
        intdata_[name] = data;
      else if constexpr (std::is_same_v<T, double>)
        doubledata_[name] = data;
      else if constexpr (std::is_same_v<T, bool>)
        booldata_[name] = data;
      else if constexpr (std::is_same_v<T, std::vector<int>>)
        vecintdata_[name] = data;
      else if constexpr (std::is_same_v<T, std::vector<double>>)
        vecdoubledata_[name] = data;
      else if constexpr (std::is_same_v<T, std::map<int, std::vector<double>>>)
        mapdata_[name] = data;
      else if constexpr (std::is_same_v<T, std::string>)
        stringdata_[name] = data;
      else
        anydata_[name] = data;
    }

    /**
     * Access group @p name. If the group does not exist, it will be created.
     */
    InputParameterContainer& group(const std::string& name);

    /**
     * Access group @p name. This function throws an error if the group does not exist.
     */
    [[nodiscard]] const InputParameterContainer& group(const std::string& name) const;


    /**
     * Combine the data from another container with this one. Conflicting data will throw an error.
     */
    void merge(const InputParameterContainer& other);

    /*!
     * Get a const reference to the data stored at the key @p name from the container. An error
     * is thrown in case no value of specified type is stored under @p name in the container.
     */
    template <typename T>
    const T& get(const std::string& name) const
    {
      if (const T* p = get_if<T>(name))
        return *p;
      else
        FOUR_C_THROW("Key %s cannot be found in the container's map!", name.c_str());
    }

    /*!
     * Get the data stored at the key @p name from the container. Return the @p default_value if no
     * value of specified type is stored under @p name in the container.
     *
     * @note This function returns the value as a copy.
     */
    template <typename T>
    T get_or(const std::string& name, T default_value) const
    {
      if (const T* p = get_if<T>(name))
        return *p;
      else
        return default_value;
    }

    /*!
     * Get the const pointer to the data stored at the key @p name from the container. Return a
     * `nullptr` if no value of specified type is stored under @p name in the container.
     */
    template <typename T>
    const T* get_if(const std::string& name) const
    {
      // access function for known types
      [[maybe_unused]] const auto access = [](const auto& map, const std::string& name) -> const T*
      {
        const auto it = map.find(name);
        if (it != map.end())
        {
          return std::addressof(it->second);
        }
        else
        {
          return nullptr;
        }
      };

      // access function for any type
      [[maybe_unused]] const auto access_any = [](const auto& map,
                                                   const std::string& name) -> const T*
      {
        const auto it = map.find(name);
        if (it != map.end())
        {
          return Internal::try_get_any_data<T>(name, it->second);
        }
        else
        {
          return nullptr;
        }
      };

      // get data from the container
      if constexpr (std::is_same_v<T, int>)
        return access(intdata_, name);
      else if constexpr (std::is_same_v<T, double>)
        return access(doubledata_, name);
      else if constexpr (std::is_same_v<T, bool>)
        return access(booldata_, name);
      else if constexpr (std::is_same_v<T, std::vector<int>>)
        return access(vecintdata_, name);
      else if constexpr (std::is_same_v<T, std::vector<double>>)
        return access(vecdoubledata_, name);
      else if constexpr (std::is_same_v<T, std::map<int, std::vector<double>>>)
        return access(mapdata_, name);
      else if constexpr (std::is_same_v<T, std::string>)
        return access(stringdata_, name);
      else
        return access_any(anydata_, name);
    }


   private:
    //! a map to store integer data in
    std::map<std::string, int> intdata_;

    //! a map to store double data in
    std::map<std::string, double> doubledata_;

    //! a map to store bool data in
    std::map<std::string, bool> booldata_;

    //! a map to store vector integer data in
    std::map<std::string, std::vector<int>> vecintdata_;

    //! a map to store vector double data in
    std::map<std::string, std::vector<double>> vecdoubledata_;

    //! a map to store maps of stl vector data
    std::map<std::string, std::map<int, std::vector<double>>> mapdata_;

    //! a map to store string data in
    std::map<std::string, std::string> stringdata_;

    //! a map to store anything
    std::map<std::string, std::any> anydata_;

    //! Groups present in this container
    std::map<std::string, InputParameterContainer> groups_;
  };  // class InputParameterContainer
}  // namespace Core::IO

// << operator
std::ostream& operator<<(std::ostream& os, const Core::IO::InputParameterContainer& cont);


FOUR_C_NAMESPACE_CLOSE

#endif

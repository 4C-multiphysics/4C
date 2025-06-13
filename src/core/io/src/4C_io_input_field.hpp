// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_IO_INPUT_FIELD_HPP
#define FOUR_C_IO_INPUT_FIELD_HPP

#include "4C_config.hpp"

#include "4C_utils_exceptions.hpp"
#include "4C_utils_std23_unreachable.hpp"

#include <unordered_map>
#include <variant>

FOUR_C_NAMESPACE_OPEN

namespace Core::IO
{
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
    using StorageType = std::variant<T, std::unordered_map<IndexType, T>>;

    /**
     * Construct an InputField from a single @p const_data. This field will have the same value
     * for every element index.
     */
    explicit InputField(T const_data) : data_(std::move(const_data)) {}

    /**
     * Construct an InputField from a map of element-wise data. The @p data map contains
     * element indices as keys and the corresponding values of type T.
     */
    explicit InputField(std::unordered_map<IndexType, T> data) : data_(data) {}

    /**
     * Access the value of the field for the given @p element index. The @p element_id
     * is not checked for validity and asking for an invalid index will lead to undefined behavior.
     * Use the `at()` function if you want to check for validity.
     */
    [[nodiscard]] const T& operator[](IndexType element_id) const { return get(element_id, false); }

    /**
     * Access the value of the field for the given @p element index. If the @p element_id
     * is not a valid index, this function will throw an error.
     */
    [[nodiscard]] const T& at(IndexType element_id) const { return get(element_id, true); }

   private:
    //! Internal getter which can optionally check for the validity of the element index.
    const T& get(IndexType element_id, bool check) const
    {
      if (std::holds_alternative<T>(data_))
      {
        return std::get<T>(data_);
      }
      else if (std::holds_alternative<std::unordered_map<IndexType, T>>(data_))
      {
        const auto& map = std::get<std::unordered_map<IndexType, T>>(data_);
        auto it = map.find(element_id);
        if (check)
        {
          FOUR_C_ASSERT_ALWAYS(
              it != map.end(), "Element index {} not found in InputField.", element_id);
        }
        return it->second;
      }
      std23::unreachable();
    }

    std::variant<T, std::unordered_map<IndexType, T>> data_;
  };
}  // namespace Core::IO

FOUR_C_NAMESPACE_CLOSE

#endif

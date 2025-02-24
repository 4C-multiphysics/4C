// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_IO_INPUT_TYPES_HPP
#define FOUR_C_IO_INPUT_TYPES_HPP

#include "4C_config.hpp"

#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::IO
{
  /**
   * A type that either contains a value of type T or nothing. This type is useful to
   * represent the absence of a value marked by "none" in the input file. The type is just an alias
   * for std::optional<T> to more precisely convey the intended meaning.
   */
  template <typename T>
  using Noneable = std::optional<T>;

  /**
   * A constant to represent the absence of a value in the input file. The constant is templated
   * on the type of the value inside the Noneable to properly handle cases where a Noneable is
   * itself stored inside a std::optional type.
   */
  template <typename T>
  inline constexpr auto none = Noneable<T>{std::nullopt};

  /**
   * Concept to check if a type is a Noneable type.
   */
  template <typename T>
  concept IsNoneable = requires(T t) {
    { t.has_value() } -> std::convertible_to<bool>;
    { t.value() } -> std::convertible_to<typename T::value_type>;
  };

  namespace Internal
  {
    template <typename T>
    struct SupportedTypeHelper : std::false_type
    {
    };

    template <typename T>
    concept SupportedTypePrimitives =
        std::same_as<T, int> || std::same_as<T, double> || std::same_as<T, bool> ||
        std::same_as<T, std::string> || std::same_as<T, std::filesystem::path>;

    template <SupportedTypePrimitives T>
    struct SupportedTypeHelper<T> : std::true_type
    {
    };

    template <typename T>
    struct SupportedTypeHelper<std::vector<T>> : SupportedTypeHelper<T>
    {
    };

    template <typename U>
    struct SupportedTypeHelper<std::map<std::string, U>> : SupportedTypeHelper<U>
    {
    };

    template <typename T>
    struct SupportedTypeHelper<Noneable<T>> : SupportedTypeHelper<T>
    {
    };

    template <typename T>
    struct RankHelper
    {
      static constexpr std::size_t value = 0;
    };

    template <typename T>
    struct RankHelper<std::vector<T>>
    {
      static constexpr std::size_t value = 1 + RankHelper<T>::value;
    };

    template <typename T>
    struct RankHelper<std::map<std::string, T>>
    {
      static constexpr std::size_t value = 1 + RankHelper<T>::value;
    };

    template <typename T>
    concept IsStdArray = requires {
      typename T::value_type;
      std::tuple_size<T>::value;
    };

    template <typename T>
    struct RemoveNoneableHelper
    {
      using type = T;
    };

    template <typename T>
      requires IsNoneable<T>
    struct RemoveNoneableHelper<T>
    {
      using type = typename T::value_type;
    };
  }  // namespace Internal

  /**
   * We deliberately limit ourselves to a few generally useful types. While it would not be too
   * difficult to support all the fundamental and container types that C++ provides, this would
   * likely lead to more confusion for users than it would provide benefits. After all, when
   * consuming the parsed input, the user will have to use the exact type of the parameter. Also,
   * input file formats are often not able to distinguish fundamental types like `double` and
   * `float` and there is little benefit in supporting both in the input mechanism. Any conversion
   * between types can be done in the user code, which usually entails additional validation and
   * error handling anyway.
   *
   * The supported types are:
   * - `int`
   * - `double`
   * - `bool`
   * - `std::string`
   * - `std::filesystem::path`
   * - `Noneable<T>`, where `T` is one of the supported types
   * - `std::vector<T>`, where `T` is one of the supported types
   * - `std::map<std::string, T>`, where `T` is one of the supported types
   */
  template <typename T>
  concept SupportedType = Internal::SupportedTypeHelper<T>::value;

  /**
   * Determine the rank of a type, i.e., how many levels of nested containers are present.
   */
  template <SupportedType T>
  constexpr std::size_t rank()
  {
    return Internal::RankHelper<T>::value;
  }

  /**
   * Remove the Noneable wrapper from a type if it is a Noneable type. Otherwise, return the type
   * itself.
   */
  template <typename T>
  using RemoveNoneable = typename Internal::RemoveNoneableHelper<T>::type;

}  // namespace Core::IO

FOUR_C_NAMESPACE_CLOSE

#endif

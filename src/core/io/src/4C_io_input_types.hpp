// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_IO_INPUT_TYPES_HPP
#define FOUR_C_IO_INPUT_TYPES_HPP

#include "4C_config.hpp"

#include "4C_utils_enum.hpp"

#include <array>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::IO
{

  namespace Internal
  {
    template <typename T>
    struct SupportedTypeHelper : std::false_type
    {
    };

    // see https://yaml.org/spec/1.2.2/#1032-tag-resolution
    // note that `true`, `false`, `.inf`, `.Inf`, `.INF`, `.nan`, `.NaN`, `.NAN` and `~` cannot be
    // used as enum names anyways
    constexpr std::array illegal_enum_names = {
        "True",
        "TRUE",
        "False",
        "FALSE",
        "null",
        "Null",
        "NULL",
    };

    template <typename T>
    concept LegalEnum =
        std::is_enum_v<T> && std::invoke(
                                 []
                                 {
                                   for (std::string_view name : EnumTools::enum_names<T>())
                                   {
                                     for (auto reserved : illegal_enum_names)
                                     {
                                       if (name == reserved)
                                       {
                                         return false;
                                       }
                                     }
                                   }
                                   return true;
                                 });

    template <typename T>
    concept SupportedTypePrimitives =
        std::same_as<T, int> || std::same_as<T, double> || std::same_as<T, bool> ||
        std::same_as<T, std::string> || std::same_as<T, std::filesystem::path> || LegalEnum<T>;

    template <SupportedTypePrimitives T>
    struct SupportedTypeHelper<T> : std::true_type
    {
    };

    template <typename T>
    struct SupportedTypeHelper<std::vector<T>> : SupportedTypeHelper<T>
    {
    };

    template <typename T, std::size_t n>
    struct SupportedTypeHelper<std::array<T, n>> : SupportedTypeHelper<T>
    {
    };

    template <typename... Ts>
    struct SupportedTypeHelper<std::tuple<Ts...>>
        : std::bool_constant<(SupportedTypeHelper<Ts>::value && ...)>
    {
    };

    template <typename T1, typename T2>
    struct SupportedTypeHelper<std::pair<T1, T2>>
        : std::bool_constant<SupportedTypeHelper<T1>::value && SupportedTypeHelper<T2>::value>
    {
    };

    template <typename U>
    struct SupportedTypeHelper<std::map<std::string, U>> : SupportedTypeHelper<U>
    {
    };

    template <typename T>
    struct SupportedTypeHelper<std::optional<T>> : SupportedTypeHelper<T>
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

    template <typename T, std::size_t n>
    struct RankHelper<std::array<T, n>>
    {
      static constexpr std::size_t value = RankHelper<T>::value;
    };

    template <typename T>
    struct RankHelper<std::map<std::string, T>>
    {
      static constexpr std::size_t value = 1 + RankHelper<T>::value;
    };

    template <typename... Ts>
    struct RankHelper<std::tuple<Ts...>>
    {
      static constexpr std::size_t value = (RankHelper<Ts>::value + ...);
    };

    template <typename T1, typename T2>
    struct RankHelper<std::pair<T1, T2>>
    {
      static constexpr std::size_t value = RankHelper<T1>::value + RankHelper<T2>::value;
    };

    template <typename T>
    struct RankHelper<std::optional<T>>
    {
      // std::optional is not considered a container, so it does not increase the rank.
      static constexpr std::size_t value = RankHelper<T>::value;
    };

    template <typename T>
    concept IsStdArray = requires {
      typename T::value_type;
      std::tuple_size<T>::value;
    };

    template <typename T>
    struct OptionalHelper : std::false_type
    {
    };

    template <typename T>
    struct OptionalHelper<std::optional<T>> : std::true_type
    {
    };

    template <typename T>
    struct RemoveOptionalHelper
    {
      using type = T;
    };

    template <typename T>
      requires OptionalHelper<T>::value
    struct RemoveOptionalHelper<T>
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
   * - any enum type
   * - `std::optional<T>`, where `T` is one of the supported types
   * - `std::vector<T>`, where `T` is one of the supported types
   * -  std::array<T, n>, where `T` is one of the supported types and `n` is the array size
   * - `std::map<std::string, T>`, where `T` is one of the supported types
   * - `std::tuple<Ts...>`, where all `Ts` are one of the supported types
   * - `std::pair<T1, T2>`, where `T1` and `T2` are both one of the supported types
   */
  template <typename T>
  concept SupportedType = Internal::SupportedTypeHelper<T>::value;

  /**
   * Determine the dynamic rank of a type, i.e., how many dynamic extents a type has.
   *
   * Examples:
   *   std::vector<int>: 1
   *   std::vector<std::vector<int>>: 2
   *   std::array<int, 3>: 0
   */
  template <typename T>
    requires SupportedType<std::decay_t<T>>
  constexpr std::size_t dynamic_rank()
  {
    return Internal::RankHelper<std::decay_t<T>>::value;
  }

  /**
   * Concept to check if a type is a std::optional type.
   */
  template <typename T>
  concept OptionalType = Internal::OptionalHelper<T>::value;

  /**
   * Remove the std::optional wrapped around a type. If the type is not a std::optional, the type
   * itself is returned.
   */
  template <typename T>
  using RemoveOptional = typename Internal::RemoveOptionalHelper<T>::type;
}  // namespace Core::IO

FOUR_C_NAMESPACE_CLOSE

#endif

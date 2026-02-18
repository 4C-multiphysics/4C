// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FEM_GENERAL_CELL_TYPE_TRAITS_DETAILS_HPP
#define FOUR_C_FEM_GENERAL_CELL_TYPE_TRAITS_DETAILS_HPP

#include "4C_config.hpp"

#include "4C_fem_general_cell_type.hpp"
#include "4C_fem_general_cell_type_traits_def.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_std23_unreachable.hpp"

#include <array>
#include <tuple>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE::Internal
{
  //! @name helper functions for cell-type related type lists
  /// @{
  template <typename... T>
  struct Join
  {
    using type = decltype(std::tuple_cat(std::declval<T>()...));
  };


  template <template <typename...> typename TypeList, template <CellType> typename Base, typename T>
  struct apply_celltype_sequence;

  template <template <CellType> typename Base, template <typename...> typename TypeList,
      CellType... celltypes>
  struct apply_celltype_sequence<TypeList, Base, CelltypeSequence<celltypes...>>
  {
    using type = TypeList<Base<celltypes>...>;
  };

  template <typename T>
  struct celltype_sequence_to_string_array;

  template <CellType... celltypes>
  struct celltype_sequence_to_string_array<CelltypeSequence<celltypes...>>
  {
    static constexpr std::array value = {Internal::CellTypeInformation<celltypes>::name...};
  };

  template <typename T>
  struct celltype_sequence_to_array;

  template <CellType... celltypes>
  struct celltype_sequence_to_array<CelltypeSequence<celltypes...>>
  {
    static constexpr std::array value = {celltypes...};
  };

  template <std::size_t first, typename T>
  struct celltype_sequence_generator;

  template <std::size_t first, std::size_t... ints>
  struct celltype_sequence_generator<first, std::integer_sequence<std::size_t, ints...>>
  {
    using type = CelltypeSequence<static_cast<CellType>(ints + first)...>;
  };

  template <typename T>
  struct first_celltype_in_sequence;

  template <CellType celltype, CellType... celltypes>
  struct first_celltype_in_sequence<CelltypeSequence<celltype, celltypes...>>
  {
    static constexpr CellType value = celltype;
  };

  template <CellType... celltypes>
  constexpr bool is_cell_type_in_sequence(CellType celltype, CelltypeSequence<celltypes...>)
  {
    return ((celltype == celltypes) || ...);
  }

  template <CellType first, CellType last>
  using make_celltype_sequence =
      typename celltype_sequence_generator<static_cast<std::size_t>(first),
          std::make_integer_sequence<std::size_t,
              static_cast<std::size_t>(last) + 1 - static_cast<std::size_t>(first)>>::type;
  /// @}

  //! @name Helper functions for the celltype switch
  /// @{
  template <CellType celltype, typename CelltypeSequence, typename Function,
      typename UnsupportedCellTypeCallable>
  constexpr auto cell_type_switch_item([[maybe_unused]] Function fct,
      UnsupportedCellTypeCallable unsupported_celltype_callable) -> std::invoke_result_t<Function,
      std::integral_constant<CellType, first_celltype_in_sequence<CelltypeSequence>::value>>
  {
    if constexpr (is_cell_type_in_sequence(celltype, CelltypeSequence{}))
    {
      return fct(std::integral_constant<CellType, celltype>{});
    }

    // 1. Determine the actual return types
    using ErrorReturnType = std::invoke_result_t<UnsupportedCellTypeCallable,
        std::integral_constant<CellType, celltype>>;

    using SuccessReturnType = std::invoke_result_t<Function,
        std::integral_constant<CellType, first_celltype_in_sequence<CelltypeSequence>::value>>;

    // 2. Check for void safely
    constexpr bool is_void = std::is_void_v<ErrorReturnType>;

    // 3. Safely check for convertibility
    // If ErrorReturnType is void, we pass SuccessReturnType instead (Success is always convertible
    // to itself). This prevents the "reference to void" error.
    constexpr bool is_convertible =
        std::is_convertible_v<std::conditional_t<is_void, SuccessReturnType, ErrorReturnType>,
            SuccessReturnType>;

    // 4. Now the static_assert works as intended
    static_assert(is_void || is_convertible,
        "The unsupported celltype callable must either return void (and throw) "
        "or return a type convertible to the success path return type!");

    if constexpr (is_void)
    {
      // Call the handler (which throws).
      // No 'return' here because it returns void.
      unsupported_celltype_callable(std::integral_constant<CellType, celltype>{});

      // This is never reached but required to satisfy the compiler. The compiler may still assume
      // the void type is returned although it is blocked by the previous static_assert.
      std23::unreachable();
    }
    else if constexpr (is_convertible)
    {
      // This path is safe because we know it's not void and it's convertible.
      return unsupported_celltype_callable(std::integral_constant<CellType, celltype>{});
    }

    unsupported_celltype_callable(std::integral_constant<CellType, celltype>{});
    FOUR_C_THROW("Your unsupported celltype callable does not throw or return a compatible type!");
  }

  template <typename CelltypeSequence>
  struct ThrowUnsupportedCellTypeError
  {
    template <typename CellTypeConstant>
    [[noreturn]] void operator()(CellTypeConstant celltype_t) const
    {
      constexpr std::array celltypes_str =
          celltype_sequence_to_string_array<CelltypeSequence>::value;

      std::string celltypes_str_acc;

      for (const auto& item : celltypes_str)
      {
        if (!celltypes_str_acc.empty())
        {
          celltypes_str_acc += ", ";
        }
        celltypes_str_acc += item;
      }

      FOUR_C_THROW(
          "The function you are calling is not implemented for the cell type {}. Supported "
          "celltypes are {}",
          Internal::CellTypeInformation<celltype_t()>::name, celltypes_str_acc.c_str());
    }
  };
  /// @}
}  // namespace Core::FE::Internal

FOUR_C_NAMESPACE_CLOSE

#endif
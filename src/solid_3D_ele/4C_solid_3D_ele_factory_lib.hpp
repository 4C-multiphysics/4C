// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SOLID_3D_ELE_FACTORY_LIB_HPP
#define FOUR_C_SOLID_3D_ELE_FACTORY_LIB_HPP

#include "4C_config.hpp"

#include "4C_fem_general_cell_type_traits.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_utils_box.hpp"

#include <variant>

FOUR_C_NAMESPACE_OPEN

namespace Solid::Elements
{
  enum class EasType;
}
namespace Discret::Elements
{
  namespace Internal
  {
    template <typename Tuple>
    struct CreateVariant;

    template <typename... Ts>
    struct CreateVariant<Core::FE::BaseTypeList<Ts...>>
    {
      using type = std::variant<Core::Utils::Box<Ts>...>;
    };

  }  // namespace Internal

  /*!
   * @brief Meta function to create a std::variant of all provides types wrapped in a @p
   * Internal::VariantItem.
   */
  template <typename... Ts>
  using CreateVariantType = typename Internal::CreateVariant<Ts...>::type;

  namespace Internal
  {
    /*!
     * @brief A struct that determines whether @p T is a valid template type
     *
     * The member variable value is true if the first template parameter is a valid type
     *
     * @tparam typename : Template parameter that may be a valid type or not
     */
    template <typename, typename = void>
    struct IsValidTypeTrait : std::false_type
    {
    };

    template <typename T>
    struct IsValidTypeTrait<T, std::void_t<typename T::type>> : std::true_type
    {
    };
  }  // namespace Internal

  /*!
   * @brief Determines whether we have implemented a solid calculation formulation
   *
   * @tparam T typename: Template parameter that may be a valid type or not
   */
  template <typename T>
  constexpr bool is_valid_type = Internal::IsValidTypeTrait<T>::value;

}  // namespace Discret::Elements


FOUR_C_NAMESPACE_CLOSE

#endif

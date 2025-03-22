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

#include <memory>
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
    /*!
     * @brief A simple container type that holds a pointer to the value of type T providing value
     * semantics.
     *
     * It is used to store types of different size in a std::variant.
     *
     * @tparam T
     */
    template <typename T>
    class VariantItem
    {
     public:
      VariantItem() { value_ = std::make_unique<T>(); }
      VariantItem(T&& obj) : value_(new T(std::move(obj))) {}
      VariantItem(const T& obj) : value_(new T(obj)) {}

      ~VariantItem() = default;

      VariantItem(const VariantItem& other) : VariantItem(*other.value_) {}
      VariantItem& operator=(const VariantItem& other)
      {
        *value_ = *other.value_;
        return *this;
      }

      // allow access via dereferencing operator propagating constness
      T& operator*() { return *value_; }
      const T& operator*() const { return *value_; }

      // allow class member access propagating constness
      T* operator->() { return value_.get(); }
      const T* operator->() const { return value_.get(); }

     private:
      std::unique_ptr<T> value_;
    };

    template <typename Tuple>
    struct CreateVariant;

    template <typename... Ts>
    struct CreateVariant<Core::FE::BaseTypeList<Ts...>>
    {
      using type = std::variant<VariantItem<Ts>...>;
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

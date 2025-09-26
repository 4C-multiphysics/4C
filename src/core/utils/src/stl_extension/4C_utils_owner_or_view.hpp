// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_UTILS_OWNER_OR_VIEW_HPP
#define FOUR_C_UTILS_OWNER_OR_VIEW_HPP

#include "4C_config.hpp"

#include <functional>
#include <memory>


FOUR_C_NAMESPACE_OPEN

namespace Core::Utils
{
  /**
   * This class holds a pointer to an object of type T. It can either own the object or
   * views it.
   * This type is useful to hold wrapped objects in wrappers in combination with the LinAlg::View
   * mechanism.
   */
  template <typename T>
  class OwnerOrView
  {
   public:
    using Deleter = std::function<void(T*)>;

    /**
     * Default constructor. The constructed OwnerOrView will not allocate any object. Accessing the
     * content before copy or move assigning another OwnerOrView will fail at runtime.
     */
    OwnerOrView() = default;

    /**
     * Construct owning version from a unique_ptr.
     */
    template <typename D>
    explicit OwnerOrView(std::unique_ptr<T, D>&& obj_) : obj_(std::move(obj_))
    {
    }

    /**
     * Construct non-owning view on @p ptr.
     */
    explicit OwnerOrView(T* ptr) : obj_(ptr, [](T*) {}) {}

    /**
     * Const-propagating dereference operator.
     */
    const T& operator*() const { return *obj_; }

    /**
     * Non-const dereference operator.
     */
    T& operator*() { return *obj_; }

    /**
     * Const-propagating pointer operator.
     */
    const T* operator->() const { return obj_.get(); }

    /**
     * Non-const pointer operator.
     */
    T* operator->() { return obj_.get(); }

    /**
     * Get the raw pointer to the object.
     *
     * @note This function does not propagate constness and instead follows the constness of the
     * type T. This is consistent with std::unique_ptr<T>::get().
     */
    T* get() const { return obj_.get(); }

   private:
    /**
     * Pointer to the object which is either owned or viewed.
     */
    std::unique_ptr<T, Deleter> obj_;
  };


  /**
   * Construct an owning OwnerOrView.
   */
  template <typename T, typename... Args>
  OwnerOrView<T> make_owner(Args&&... args)
  {
    return OwnerOrView<T>(std::make_unique<T>(std::forward<Args>(args)...));
  }

  /**
   * Construct a viewing OwnerOrView.
   */
  template <typename T>
  OwnerOrView<T> make_view(T* obj)
  {
    return OwnerOrView<T>(obj);
  }
}  // namespace Core::Utils

FOUR_C_NAMESPACE_CLOSE

#endif
// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_UTILS_BOX_HPP
#define FOUR_C_UTILS_BOX_HPP

#include "4C_config.hpp"

#include "4C_utils_exceptions.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Core::Utils
{
  /**
   * @brief A simple container type that stores a value of type T on the heap with value semantics.
   */
  template <typename T>
  class Box
  {
   public:
    /**
     * @brief Default constructor. The Box contains a default-constructed T.
     */
    Box()
      requires std::default_initializable<T>
        : value_(std::make_unique<T>())
    {
    }

    Box(T&& obj) : value_(new T(std::move(obj))) {}
    Box(const T& obj) : value_(new T(obj)) {}

    ~Box() = default;

    Box(const Box& other) : Box(*other) {}
    Box& operator=(const Box& other)
    {
      *value_ = *other;
      return *this;
    }

    Box(Box&& other) noexcept = default;
    Box& operator=(Box&& other) noexcept = default;

    // Propagate constness in member access.
    T& operator*()
    {
      FOUR_C_ASSERT(value_, "Box is empty.");
      return *value_;
    }

    const T& operator*() const
    {
      FOUR_C_ASSERT(value_, "Box is empty.");
      return *value_;
    }

    // Propagate constness in member access.
    T* operator->() { return value_.get(); }
    const T* operator->() const { return value_.get(); }

   private:
    /// Store the value in a unique_ptr.
    std::unique_ptr<T> value_;
  };
}  // namespace Core::Utils

FOUR_C_NAMESPACE_CLOSE

#endif

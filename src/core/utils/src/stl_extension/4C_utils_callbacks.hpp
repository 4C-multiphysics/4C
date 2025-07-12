// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_UTILS_CALLBACKS_HPP
#define FOUR_C_UTILS_CALLBACKS_HPP

#include "4C_config.hpp"

#include <functional>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::Utils
{

  /**
   * @brief A list of callbacks that can be called with a specific set of arguments @p Args.
   *
   * This class allows you to register callbacks that can be called later with the specified
   * arguments.
   */
  template <typename... Args>
  class CallbackList
  {
    using Callback = std::function<void(Args...)>;
    std::vector<Callback> callbacks_;

   public:
    using IndexType = std::size_t;

    /**
     * @brief Add a callback to the list.
     */
    void add(Callback f) { callbacks_.emplace_back(std::move(f)); }

    /**
     * @brief Remove all callbacks from the list.
     */
    void clear() { callbacks_.clear(); }

    /**
     * @brief Invoke all registered callbacks with the given arguments.
     */
    void call_all(Args... args) const
    {
      for (auto& f : callbacks_) f(args...);
    }
  };

}  // namespace Core::Utils

FOUR_C_NAMESPACE_CLOSE

#endif

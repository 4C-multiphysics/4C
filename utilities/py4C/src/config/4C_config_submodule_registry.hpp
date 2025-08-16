// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONFIG_SUBMODULE_REGISTRY_HPP
#define FOUR_C_CONFIG_SUBMODULE_REGISTRY_HPP

#include "4C_config.hpp"

#include <pybind11/pybind11.h>

#include <functional>
#include <vector>

FOUR_C_NAMESPACE_OPEN


namespace Py4C
{
  struct SubmoduleRegistryEntry
  {
    std::string name;
    std::string doc;
    std::function<void(pybind11::module_&)> init_funct;
  };

  inline std::vector<SubmoduleRegistryEntry>& get_submodule_registry()
  {
    static std::vector<SubmoduleRegistryEntry> registry;
    return registry;
  }
}  // namespace Py4C

/*!
 * @brief Register a new submodule for 4C Python bindings
 *
 * This macro simplifies the registration of new submodules by automatically
 * registering the submodule at the global 4C Python bindings module upon initialization.
 *
 * @param name The name of the submodule
 * @param doc A brief documentation string for the submodule
 * @param init_func A functional with signature `void(pybind11::module_&)` that initializes all
 * bindings
 */
#define FOUR_C_PYBIND_REGISTER_SUBMODULE(name, doc, init_func)          \
  static bool _four_c_pybind_submodule_register_##name = []() -> bool   \
  {                                                                     \
    Py4C::get_submodule_registry().emplace_back(#name, doc, init_func); \
    return true;                                                        \
  }();

FOUR_C_NAMESPACE_CLOSE

#endif

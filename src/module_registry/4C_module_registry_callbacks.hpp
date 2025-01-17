// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MODULE_REGISTRY_CALLBACKS_HPP
#define FOUR_C_MODULE_REGISTRY_CALLBACKS_HPP

#include "4C_config.hpp"

#include "4C_io_input_spec.hpp"
#include "4C_utils_function_manager.hpp"

#include <functional>

FOUR_C_NAMESPACE_OPEN

/**
 * This struct holds callback functions that are populated by modules. The callbacks are invoked
 * by executables if the specific module is configured. All callbacks are optional and do not have
 * to be set inside the modules.
 */
struct ModuleCallbacks
{
  /**
   * Register all types defined in a module that are derived from Core::Communication::ParObject and
   * can be communicated over MPI.
   *
   * This call ensures that every ParObjectType listed within the corresponding source file
   * registers itself with ParObjectRegistry. This call is necessary at the beginning of any
   * executable that uses the parallel communication of data as documented in ParObject.
   *
   */
  std::function<void()> RegisterParObjectTypes;

  /**
   * Allow the module to attach any custom Function to the @p function_manager object. Inside this
   * callback, a module should call `FunctionManager::add_function_definition`.
   */
  std::function<void(Core::Utils::FunctionManager& function_manager)> AttachFunctionDefinitions;

  /**
   * A callback to return valid result description lines.
   */
  std::function<Core::IO::InputSpec()> valid_result_description_lines;
};

FOUR_C_NAMESPACE_CLOSE

#endif

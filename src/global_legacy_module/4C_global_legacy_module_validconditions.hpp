// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_GLOBAL_LEGACY_MODULE_VALIDCONDITIONS_HPP
#define FOUR_C_GLOBAL_LEGACY_MODULE_VALIDCONDITIONS_HPP

#include "4C_config.hpp"

#include "4C_fem_condition_definition.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Global
{
  /// construct list with all conditions and documentation
  std::vector<Core::Conditions::ConditionDefinition> valid_conditions();
}  // namespace Global

FOUR_C_NAMESPACE_CLOSE

#endif

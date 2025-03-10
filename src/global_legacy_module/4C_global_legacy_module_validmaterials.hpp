// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_GLOBAL_LEGACY_MODULE_VALIDMATERIALS_HPP
#define FOUR_C_GLOBAL_LEGACY_MODULE_VALIDMATERIALS_HPP

#include "4C_config.hpp"

#include "4C_io_input_spec.hpp"
#include "4C_legacy_enum_definitions_materials.hpp"

#include <Teuchos_Array.hpp>

#include <unordered_map>

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  class MaterialDefinition;
}

namespace Global
{
  std::unordered_map<Core::Materials::MaterialType, Core::IO::InputSpec> valid_materials();
}  // namespace Global


FOUR_C_NAMESPACE_CLOSE

#endif

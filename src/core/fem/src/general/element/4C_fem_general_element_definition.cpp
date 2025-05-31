// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_general_element_definition.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_io_input_parameter_container.hpp"
#include "4C_io_input_spec.hpp"

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::Elements::ElementDefinition::setup_valid_element_lines()
{
  Core::Communication::ParObjectFactory::instance().setup_element_definition(definitions_);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
const Core::IO::InputSpec& Core::Elements::ElementDefinition::element_lines(
    std::string name, std::string cell_type)
{
  FOUR_C_ASSERT(definitions_.contains(name), "Element type not found: {}", name);
  auto& defs = definitions_.at(name);
  FOUR_C_ASSERT(defs.contains(cell_type), "Cell type not found: {}", cell_type);
  return defs.at(cell_type);
}

FOUR_C_NAMESPACE_CLOSE

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_condition_definition.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_io_input_file.hpp"
#include "4C_io_input_file_utils.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_utils_exceptions.hpp"

#include <algorithm>
#include <iterator>
#include <utility>

FOUR_C_NAMESPACE_OPEN



/* -----------------------------------------------------------------------------------------------*
 | Class ConditionDefinition                                                                      |
 * -----------------------------------------------------------------------------------------------*/

Core::Conditions::ConditionDefinition::ConditionDefinition(std::string sectionname,
    std::string conditionname, std::string description, Core::Conditions::ConditionType condtype,
    bool buildgeometry, Core::Conditions::GeometryType gtype)
    : sectionname_(std::move(sectionname)),
      conditionname_(std::move(conditionname)),
      description_(std::move(description)),
      condtype_(condtype),
      buildgeometry_(buildgeometry),
      gtype_(gtype)
{
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::Conditions::ConditionDefinition::add_component(Core::IO::InputSpec&& spec)
{
  specs_.emplace_back(std::move(spec));
}


void Core::Conditions::ConditionDefinition::add_component(const Core::IO::InputSpec& spec)
{
  specs_.emplace_back(spec);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::Conditions::ConditionDefinition::read(Core::IO::InputFile& input,
    std::multimap<int, std::shared_ptr<Core::Conditions::Condition>>& cmap) const
{
  Core::IO::InputParameterContainer container;
  try
  {
    input.match_section(section_name(), container);
  }
  catch (const Core::Exception& e)
  {
    FOUR_C_THROW("Failed to match condition specification in section '{}'. The error was:\n{}.",
        section_name(), e.what());
  }


  for (const auto& condition_data :
      container.get_or<std::vector<Core::IO::InputParameterContainer>>(section_name(), {}))
  {
    auto entity_type = condition_data.get<EntityType>("ENTITY_TYPE");

    int id = condition_data.get<int>("E");
    // Legacy IDs are read as 1-based, but internally we use 0-based IDs.
    if (entity_type == EntityType::legacy_id) id -= 1;

    std::shared_ptr<Core::Conditions::Condition> condition =
        std::make_shared<Core::Conditions::Condition>(
            id, condtype_, buildgeometry_, gtype_, entity_type);
    condition->parameters() = condition_data;

    cmap.emplace(id, condition);
  }
}

FOUR_C_NAMESPACE_CLOSE

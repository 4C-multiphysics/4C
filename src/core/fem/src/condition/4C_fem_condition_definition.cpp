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
#include <optional>
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
  using namespace Core::IO::InputSpecBuilders;
  // Add common parameters to all conditions.

  add_component(parameter<std::optional<int>>(
      "E", {.description = "ID of the condition. This ID refers to the respective "
                           "topological entity of the condition. Not allowed if ENTITY_TYPE is "
                           "NODE_SET_NAME."}));
  add_component(parameter<std::optional<std::string>>("NODE_SET_NAME",
      {.description = "This refers to the respective node set name in the external mesh file. Only "
                      "allowed if ENTITY_TYPE is NODE_SET_NAME and no E: ID is given."}));
  add_component(parameter<Core::Conditions::EntityType>("ENTITY_TYPE",
      {.description = "The type of entity identifier being used. Refers to E for ID-based "
                      "identification or to NODE_SET_NAME for name-based identification.",
          .default_value = Core::Conditions::EntityType::legacy_id}));
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
    std::multimap<int, std::shared_ptr<Core::Conditions::Condition>>& cmap,
    const std::map<std::string, std::vector<int>>& node_sets_names) const
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
    // get entity_type, id, node_set_name from input
    auto entity_type = condition_data.get<EntityType>("ENTITY_TYPE");
    auto id = condition_data.get<std::optional<int>>("E");
    auto node_set_name = condition_data.get<std::optional<std::string>>("NODE_SET_NAME");

    int resolved_id = resolve_entity_id(entity_type, id, node_set_name, node_sets_names);

    auto condition = std::make_shared<Core::Conditions::Condition>(
        resolved_id, condtype_, buildgeometry_, gtype_, entity_type, node_set_name);

    condition->parameters() = condition_data;

    cmap.emplace(resolved_id, condition);
  }
}

int Core::Conditions::ConditionDefinition::resolve_entity_id(EntityType entity_type,
    std::optional<int> id, const std::optional<std::string>& node_set_name,
    const std::map<std::string, std::vector<int>>& node_sets_names) const
{
  if (entity_type == EntityType::node_set_name)
  {
    FOUR_C_ASSERT_ALWAYS(node_set_name.has_value(),
        "{} condition of entity type NODE_SET_NAME requires a non-empty NODE_SET_NAME.", condtype_);

    FOUR_C_ASSERT_ALWAYS(!id.has_value(),
        "{} condition of entity type NODE_SET_NAME must not specify an ID via E: ID.", condtype_);

    FOUR_C_ASSERT_ALWAYS(node_sets_names.contains(node_set_name.value()),
        "Cannot apply {} condition with external name '{}' which is not specified in the mesh "
        "file.",
        condtype_, node_set_name.value());

    const auto& ids = node_sets_names.at(node_set_name.value());
    FOUR_C_ASSERT_ALWAYS(ids.size() == 1,
        "Cannot apply {} condition with external name '{}' which is not unique in the mesh file "
        "({} occurrences found).",
        condtype_, node_set_name.value(), ids.size());
    return ids[0];
  }

  // Other entity types
  FOUR_C_ASSERT_ALWAYS(id.has_value() && id.value() >= 0,
      "{} condition of entity type {} requires a non-negative ID specified via E: ID.", condtype_,
      entity_type);

  FOUR_C_ASSERT_ALWAYS(!node_set_name.has_value(),
      "{} condition of entity type {} must not specify a NODE_SET_NAME.", condtype_, entity_type);

  // Legacy IDs are read as 1-based, but internally we use 0-based IDs.
  if (entity_type == EntityType::legacy_id)
  {
    FOUR_C_ASSERT_ALWAYS(id.value() > 0,
        "{} condition of entity type legacy_id requires a positive ID specified via E: ID.",
        condtype_);
    return id.value() - 1;
  }

  return id.value();
}



Core::IO::InputSpec Core::Conditions::ConditionDefinition::spec() const
{
  using namespace Core::IO::InputSpecBuilders;
  return list(section_name(), all_of(specs_), {.description = description_, .required = false});
}

FOUR_C_NAMESPACE_CLOSE

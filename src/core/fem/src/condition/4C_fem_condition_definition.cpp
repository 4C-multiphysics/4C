// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_condition_definition.hpp"

#include "4C_fem_condition.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_io_input_file.hpp"
#include "4C_io_input_file_utils.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_io_input_spec_validators.hpp"
#include "4C_utils_exceptions.hpp"

#include <algorithm>
#include <iterator>
#include <optional>
#include <utility>

FOUR_C_NAMESPACE_OPEN

namespace
{
  using namespace Core::Conditions;
  struct EntitySpec
  {
    EntityType type;
    // The resolved internal ID of the entity
    int id;
    std::optional<std::string> node_set_name;
  };

  /**
   * @brief Parse entity specification from input data and resolve internal ID if necessary.
   *
   * @param input_data Input parameters containing entity specification.
   * @param node_sets_names Mapping from external node set names to internal entity IDs.
   * @return validated and resolved EntitySpec
   */
  EntitySpec parse_entity(const Core::IO::InputParameterContainer& input_data,
      const std::map<std::string, std::vector<int>>& node_sets_names)
  {
    // get entity_type, id, node_set_name from input
    auto entity_type = input_data.get<std::optional<EntityType>>("ENTITY_TYPE");
    auto id = input_data.get<std::optional<int>>("E");
    auto node_set_name = input_data.get<std::optional<std::string>>("NODE_SET_NAME");

    // NODE_SET_NAME based identification
    if (node_set_name.has_value())
    {
      FOUR_C_ASSERT_ALWAYS(!entity_type.has_value() && !id.has_value(),
          "Condition with NODE_SET_NAME '{}' must not specify ENTITY_TYPE or E: ID.",
          node_set_name.value());

      FOUR_C_ASSERT_ALWAYS(node_sets_names.contains(node_set_name.value()),
          "NODE_SET_NAME '{}' could not be found in the meshfile.", node_set_name.value());

      const auto& ids = node_sets_names.at(node_set_name.value());
      FOUR_C_ASSERT_ALWAYS(ids.size() == 1,
          "NODE_SET_NAME '{}' is not unique in the meshfile ({} occurrences).",
          node_set_name.value(), ids.size());

      return {.type = EntityType::node_set_name, .id = ids[0], .node_set_name = node_set_name};
    }

    // ID based identfication
    FOUR_C_ASSERT_ALWAYS(id.has_value(),
        "A condition must specify either an ID via E or a node set name via NODE_SET_NAME.");

    // Legacy ID case (fallback for backwards compatibility)
    if (not entity_type.has_value() or entity_type.value() == EntityType::legacy_id)
    {
      FOUR_C_ASSERT_ALWAYS(id.value() > 0,
          "Conditions with ENTITY_TYPE: legacy_id require positive E: ID. (given: {})", id.value());

      return {.type = EntityType::legacy_id, .id = id.value() - 1, .node_set_name = std::nullopt};
    }

    // Other entity types

    FOUR_C_ASSERT_ALWAYS(
        id.value() >= 0, "Conditions require non-negative E: ID. (given: {})", id.value());
    return {.type = entity_type.value(), .id = id.value(), .node_set_name = std::nullopt};
  }

}  // namespace


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
                           "topological entity of the condition. Not allowed if "
                           "NODE_SET_NAME is given."}));
  add_component(parameter<std::optional<Core::Conditions::EntityType>>("ENTITY_TYPE",
      {.description = "The type of entity that E refers to. Not allowed if NODE_SET_NAME is given.",
          .validator = Validators::null_or(Validators::in_set<EntityType>(
              {EntityType::legacy_id, EntityType::element_block_id, EntityType::node_set_id}))}));
  add_component(parameter<std::optional<std::string>>("NODE_SET_NAME",
      {.description = "This refers to the respective node set name in the external mesh file. Only "
                      "allowed if neither ENTITY_TYPE nor E: ID is given."}));
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
    auto entity_spec = parse_entity(condition_data, node_sets_names);

    auto condition = std::make_shared<Core::Conditions::Condition>(entity_spec.id, condtype_,
        buildgeometry_, gtype_, entity_spec.type, entity_spec.node_set_name);

    condition->parameters() = condition_data;

    cmap.emplace(entity_spec.id, condition);
  }
}


Core::IO::InputSpec Core::Conditions::ConditionDefinition::spec() const
{
  using namespace Core::IO::InputSpecBuilders;
  return list(section_name(), all_of(specs_), {.description = description_, .required = false});
}

FOUR_C_NAMESPACE_CLOSE

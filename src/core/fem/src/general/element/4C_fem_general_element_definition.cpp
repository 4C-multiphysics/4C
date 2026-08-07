// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_general_element_definition.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_io_input_parameter_container.hpp"
#include "4C_io_input_spec_builders.hpp"

FOUR_C_NAMESPACE_OPEN


Core::Elements::ElementDefinition::ElementDefinition()
{
  Core::Communication::ParObjectFactory::instance().setup_element_definition(definitions);
}

const Core::IO::InputSpec& Core::Elements::ElementDefinition::get(
    const std::string& element_name, Core::FE::CellType cell_type) const
{
  auto it = definitions.find(element_name);
  if (it == definitions.end()) FOUR_C_THROW("No element '{}' found.", element_name);
  auto it2 = it->second.find(cell_type);
  if (it2 == it->second.end())
    FOUR_C_THROW("Element '{}' does not seem to know cell type '{}'.", element_name, cell_type);
  return it2->second;
}

std::pair<std::string, std::map<Core::FE::CellType, Core::IO::InputParameterContainer>>
Core::Elements::ElementDefinition::unpack_element_data(
    const Core::IO::InputParameterContainer& data) const
{
  const auto& [element_name, element_group] =
      data.exactly_one_group(definitions | std::views::keys);

  std::map<Core::FE::CellType, Core::IO::InputParameterContainer> data_by_cell_type;
  for (const auto& [cell_type, spec] : definitions.at(element_name))
  {
    const auto cell_type_name = Core::FE::cell_type_to_string(cell_type);
    if (element_group.has_group(cell_type_name))
      data_by_cell_type.emplace(cell_type, element_group.group(cell_type_name));
  }

  FOUR_C_ASSERT_ALWAYS(!data_by_cell_type.empty(),
      "Element definition for '{}' must contain at least one cell type group.", element_name);
  return {element_name, std::move(data_by_cell_type)};
}


Core::IO::InputSpec Core::Elements::ElementDefinition::element_data_spec(
    CellTypeGrouping cell_type_grouping) const
{
  using namespace Core::IO::InputSpecBuilders;
  std::vector<Core::IO::InputSpec> element_choices;
  for (const auto& [element_name, cell_specs] : definitions)
  {
    std::vector<Core::IO::InputSpec> cell_specs_choices;
    for (const auto& [cell_type, spec] : cell_specs)
    {
      cell_specs_choices.emplace_back(group(Core::FE::cell_type_to_string(cell_type), {spec},
          {.required = (cell_type_grouping == CellTypeGrouping::one_of)}));
    }
    element_choices.emplace_back(
        group(element_name, {(cell_type_grouping == CellTypeGrouping::one_of)
                                    ? one_of(std::move(cell_specs_choices))
                                    : all_of(std::move(cell_specs_choices))}));
  }
  return one_of(std::move(element_choices));
}

FOUR_C_NAMESPACE_CLOSE

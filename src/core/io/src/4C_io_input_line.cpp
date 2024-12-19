// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_io_input_line.hpp"

#include "4C_utils_exceptions.hpp"

#include <set>
#include <unordered_map>
#include <utility>

FOUR_C_NAMESPACE_OPEN

namespace
{
  void parse_in_arbitrary_order(Core::IO::ValueParser& parser,
      const std::vector<Core::IO::Component>& line, Core::IO::InputParameterContainer& container,
      const Core::IO::InputLineOptions& options)
  {
    std::unordered_map<std::string, const Core::IO::Component*> name_to_entry_map;
    for (const auto& entry : line)
    {
      name_to_entry_map[entry.name()] = &entry;
    }

    // Parse as long as there are tokens and we expect more entries.
    while (!parser.at_end() && !name_to_entry_map.empty())
    {
      // Only peek at the next value, do not consume it yet.
      const std::string next(parser.peek());

      const auto it = name_to_entry_map.find(next);
      if (it == name_to_entry_map.end())
      {
        if (next == "//")
        {
          parser.consume_comment(next);
          break;
        }
        else
        {
          FOUR_C_THROW("Unexpected value '%s' found in input line.", next.c_str());
        }
      }

      // Will consume the name as well as the value.
      it->second->parse(parser, container);

      // Drop the entry from the map: we do not want to parse the same value twice. This also
      // allows to check if all required values have been parsed.
      name_to_entry_map.erase(it);
    }

    // Consume a potential trailing comment
    {
      const auto next = parser.peek();
      if (next == "//") parser.consume_comment("//");
    }

    // Check if all required values have been parsed, i.e., any remaining component must be optional
    for (const auto& [name, entry] : name_to_entry_map)
    {
      if (entry->required())
      {
        FOUR_C_THROW("Required value '%s' not found in input line", name.c_str());
      }
      else if (options.store_default_values)
      {
        entry->set_default_value(container);
      }
    }
  }
}  // namespace


void Core::IO::InputLineBuilders::Internal::GroupComponent::parse_and_store_value(
    Core::IO::ValueParser& parser, Core::IO::InputParameterContainer& container) const
{
  // TODO store_default_values needs to be considered correctly

  auto& subcontainer = container.group(data.name);
  parse_in_arbitrary_order(parser, data.entries, subcontainer, {.store_default_values = true});
}


void Core::IO::InputLineBuilders::Internal::GroupComponent::set_default_value(
    Core::IO::InputParameterContainer& container) const
{
  auto& subcontainer = container.group(data.name);
  for (const auto& component : data.entries)
  {
    component.set_default_value(subcontainer);
  }
}


void Core::IO::InputLineBuilders::Internal::GroupComponent::print_value(
    std::ostream& stream, const Core::IO::InputParameterContainer& container) const
{
  for (const auto& component : data.entries)
  {
    component.print(stream, container);
  }
}


Core::IO::Component Core::IO::InputLineBuilders::tag(ScalarComponentData<bool> data)
{
  return user_defined<bool>(
      data,
      // If we encounter the tag, we set the value to true.
      [name = data.name](ValueParser& parser, InputParameterContainer& container)
      { container.add(name, true); },
      [](std::ostream&, const InputParameterContainer& container)
      {
        // Do not print anything for a tag.
      });
}


Core::IO::Component Core::IO::InputLineBuilders::group(
    Core::IO::InputLineBuilders::GroupComponentData data)
{
  auto sanitized_data = Internal::sanitize_user_input_data(std::move(data));

  FOUR_C_ASSERT_ALWAYS(!sanitized_data.entries.empty(), "A group must contain at least one entry.");

  Component::CommonData common_data{.name = sanitized_data.name,
      .description = sanitized_data.description,
      .required = sanitized_data.required};
  return Component(Internal::GroupComponent{.data = std::move(sanitized_data)}, common_data);
}


namespace
{
  void assert_unique_names(const std::vector<Core::IO::Component>& components)
  {
    std::set<std::string> names;
    for (const auto& component : components)
    {
      FOUR_C_ASSERT_ALWAYS(names.insert(component.name()).second,
          "Duplicate component name '%s' found in input line.", component.name().c_str());
    }
  }
}  // namespace


Core::IO::InputLine::InputLine(std::initializer_list<Component> components)
    : components_(components)
{
  assert_unique_names(components_);
}


Core::IO::InputLine::InputLine(std::vector<Component> components)
    : components_(std::move(components))
{
  assert_unique_names(components_);
}


bool Core::IO::InputLine::parse(std::string_view input,
    Core::IO::InputParameterContainer& container, InputLineOptions options,
    ValueParserContext context) const
{
  try
  {
    ValueParser parser(input, std::move(context));
    parse_in_arbitrary_order(parser, components_, container, options);
    FOUR_C_ASSERT_ALWAYS(parser.at_end(), "After parsing, the line still contains '%s'.",
        std::string(parser.get_unparsed_remainder()).c_str());
  }
  catch (const std::exception&)
  {
    if (options.throw_on_error)
    {
      throw;
    }
    return false;
  }
  return true;
}


void Core::IO::InputLine::print(
    std::ostream& stream, const Core::IO::InputParameterContainer& container) const
{
  for (const auto& component : components_)
  {
    component.print(stream, container);
  }
}


void Core::IO::InputLine::print_default(std::ostream& stream) const
{
  InputParameterContainer container;

  for (const auto& component : components_)
  {
    if (!component.required()) component.set_default_value(container);
    component.print(stream, container);
  }
}

FOUR_C_NAMESPACE_CLOSE

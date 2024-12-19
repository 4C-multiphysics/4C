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
    std::set<const Core::IO::Component*> unnamed_entries;
    for (const auto& entry : line)
    {
      const auto& name = entry.name();
      if (name.empty())
      {
        unnamed_entries.insert(&entry);
      }
      else
      {
        name_to_entry_map[name] = &entry;
      }
    }

    const auto are_entries_left = [&]()
    { return !name_to_entry_map.empty() || !unnamed_entries.empty(); };


    // Parse as long as there are tokens and we expect more entries.
    while (!parser.at_end() && are_entries_left())
    {
      // Only peek at the next value, do not consume it yet.
      const std::string next(parser.peek());

      if (next == "//")
      {
        parser.consume_comment(next);
        break;
      }

      // The typical case: peeking reveals the key of the next value.
      const auto it = name_to_entry_map.find(next);
      if (it != name_to_entry_map.end())
      {
        // Will consume the name as well as the value.
        it->second->parse(parser, container);

        // Drop the entry from the map: we do not want to parse the same value twice. This also
        // allows to check if all required values have been parsed.
        name_to_entry_map.erase(it);
        continue;
      }

      // We might find a parseable unnamed component.
      if (!unnamed_entries.empty())
      {
        bool parsed = false;
        Core::IO::ValueParser::BacktrackScope backtrack_scope(parser);
        for (const auto& entry : unnamed_entries)
        {
          try
          {
            entry->parse(parser, container);
          }
          catch (const Core::Exception&)
          {
            // Try the next component.
            parser.backtrack();
          }

          // Drop the entry from the set: we do not want to parse the same value twice.
          unnamed_entries.erase(entry);
          parsed = true;
          break;
        }
        // If we reach this point, none of the unnamed components fit the input.
        if (!parsed)
        {
          FOUR_C_THROW("None of the unnamed components fit the input.");
        }
        continue;
      }


      FOUR_C_THROW("Unexpected value '%s' found in input line.", next.c_str());
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


void Core::IO::InputLineBuilders::Internal::GroupComponent::parse(
    Core::IO::ValueParser& parser, Core::IO::InputParameterContainer& container) const
{
  parser.consume(data.name);

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


void Core::IO::InputLineBuilders::Internal::GroupComponent::print(
    std::ostream& stream, const Core::IO::InputParameterContainer& container) const
{
  for (const auto& component : data.entries)
  {
    component.print(stream, container);
  }
}

void Core::IO::InputLineBuilders::Internal::OneOfComponent::parse(
    Core::IO::ValueParser& parser, Core::IO::InputParameterContainer& container) const
{
  ValueParser::BacktrackScope backtrack_scope(parser);

  for (const auto& component : components)
  {
    try
    {
      component.parse(parser, container);
      return;
    }
    catch (const Core::Exception&)
    {
      // Try the next component.
      parser.backtrack();
    }
  }
  FOUR_C_THROW("None of the components fit the input.");
}


Core::IO::Component Core::IO::InputLineBuilders::tag(ScalarComponentData<bool> data)
{
  return user_defined<bool>(
      data,
      // If we encounter the tag, we set the value to true.
      [name = data.name](ValueParser& parser, InputParameterContainer& container)
      {
        parser.consume(name);
        container.add(name, true);
      },
      [name = data.name](std::ostream& stream, const InputParameterContainer& container)
      { stream << name; });
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


Core::IO::Component Core::IO::InputLineBuilders::one_of(std::vector<Component> components)
{
  return Component(Internal::OneOfComponent{.components = std::move(components)},
      {.name = "", .description = "One of the following options.", .required = true});
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

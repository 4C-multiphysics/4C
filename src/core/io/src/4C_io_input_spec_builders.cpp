// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_io_input_spec_builders.hpp"

#include <set>
#include <unordered_map>

FOUR_C_NAMESPACE_OPEN

namespace
{
  void parse_in_arbitrary_order(Core::IO::ValueParser& parser,
      const std::vector<Core::IO::InputSpec>& line, Core::IO::InputParameterContainer& container)
  {
    std::unordered_map<std::string, const Core::IO::InputSpec*> name_to_entry_map;
    std::set<const Core::IO::InputSpec*> unnamed_entries;
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

        if (parsed) continue;
      }

      // If we reach this point, we could not parse the next value. What remains must therefore
      // belong to other specs.
      break;
    }

    // Consume a potential trailing comment
    {
      const auto next = parser.peek();
      if (next == "//") parser.consume_comment("//");
    }

    for (const auto& entry : unnamed_entries)
    {
      // Unnamed entries contain a useful description, which indicates what is missing.
      FOUR_C_ASSERT_ALWAYS(!entry->required(), "Required '%s' not found in input line",
          entry->description().c_str());
    }

    // Check if all required values have been parsed, i.e., any remaining component must be optional
    for (const auto& [name, entry] : name_to_entry_map)
    {
      if (entry->required())
      {
        FOUR_C_THROW("Required value '%s' not found in input line", name.c_str());
      }
      else if (entry->has_default_value())
      {
        entry->set_default_value(container);
      }
    }
  }

  void assert_unique_or_empty_names(const std::vector<Core::IO::InputSpec>& specs)
  {
    std::set<std::string> names;
    for (const auto& component : specs)
    {
      if (!component.name().empty())
        FOUR_C_ASSERT_ALWAYS(names.insert(component.name()).second,
            "Duplicate component name '%s' found in input line.", component.name().c_str());
    }
  }

  bool all_have_default_values(const std::vector<Core::IO::InputSpec>& specs)
  {
    return std::all_of(specs.begin(), specs.end(),
        [](const Core::IO::InputSpec& component) { return component.has_default_value(); });
  }

  [[nodiscard]] const std::string& describe_for_error_handling(const Core::IO::InputSpec& spec)
  {
    return (spec.name().empty()) ? spec.description() : spec.name();
  }

  [[nodiscard]] std::string describe_for_error_handling(
      const std::vector<Core::IO::InputSpec>& specs)
  {
    if (specs.empty()) return "{}";

    std::string description = "{";
    for (const auto& spec : specs)
    {
      // Unnamed InputSpecs are created internally and will have a description that is useful for
      // error messages.
      description += describe_for_error_handling(spec);
      description += ", ";
    }
    description.pop_back();
    description.pop_back();
    description += "}";

    return description;
  }


}  // namespace

void Core::IO::InputSpecBuilders::Internal::GroupSpec::parse(
    Core::IO::ValueParser& parser, Core::IO::InputParameterContainer& container) const
{
  // Support the special case of unnamed groups.
  if (!name.empty()) parser.consume(name);

  // Parse into a separate container to avoid side effects if parsing fails.
  Core::IO::InputParameterContainer subcontainer;
  parse_in_arbitrary_order(parser, specs, subcontainer);

  if (name.empty())
    container.merge(subcontainer);
  else
    container.group(name) = subcontainer;
}


void Core::IO::InputSpecBuilders::Internal::GroupSpec::set_default_value(
    Core::IO::InputParameterContainer& container) const
{
  auto& subcontainer = (name.empty()) ? container : container.group(name);
  for (const auto& component : specs)
  {
    if (component.has_default_value()) component.set_default_value(subcontainer);
  }
}


void Core::IO::InputSpecBuilders::Internal::GroupSpec::print(
    std::ostream& stream, const Core::IO::InputParameterContainer& container) const
{
  if (!name.empty()) stream << name;

  const auto& subcontainer = (name.empty()) ? container : container.group(name);
  for (const auto& component : specs)
  {
    component.print(stream, subcontainer);
    stream << " ";
  }
}

void Core::IO::InputSpecBuilders::Internal::OneOfSpec::parse(
    Core::IO::ValueParser& parser, Core::IO::InputParameterContainer& container) const
{
  ValueParser::BacktrackScope backtrack_scope(parser);

  // Try to parse a component and backtrack if it fails.
  const auto try_parse = [&](const Core::IO::InputSpec& component) -> bool
  {
    try
    {
      component.parse(parser, container);
      return true;
    }
    catch (const Core::Exception&)
    {
      parser.backtrack();
      return false;
    }
  };

  auto component = specs.begin();
  for (; component != specs.end(); ++component)
  {
    const auto success = try_parse(*component);

    // Now check that no other component can be parsed.
    if (success)
    {
      ValueParser::BacktrackScope backtrack_scope_after_success(parser);
      for (auto other = specs.begin(); other != specs.end(); ++other)
      {
        if (other == component) continue;
        if (try_parse(*other))
        {
          // Backtrack to the original position.
          parser.backtrack();
          FOUR_C_THROW(
              "Ambiguous input: both '%s' and '%s' could be parsed, but only one of them is "
              "expected.",
              describe_for_error_handling(*component).c_str(),
              describe_for_error_handling(*other).c_str());
        }
      }

      // If we reach this point, we have successfully parsed the input.
      std::size_t index = std::distance(specs.begin(), component);
      if (on_parse_callback) on_parse_callback(parser, container, index);

      return;
    }
  }

  // If we reach this point, none of the specs could be parsed.
  FOUR_C_THROW("None of the specs fit the input. Expected %s", data.description.c_str());
}


void Core::IO::InputSpecBuilders::Internal::OneOfSpec::set_default_value(
    Core::IO::InputParameterContainer& container) const
{
  FOUR_C_THROW("Implementation error: OneOfSpec cannot have a default value.");
}


void Core::IO::InputSpecBuilders::Internal::OneOfSpec::print(
    std::ostream& stream, const Core::IO::InputParameterContainer& container) const
{
  stream << "<one_of {";
  for (const auto& component : specs)
  {
    component.print(stream, container);
    stream << ";";
  }
  stream << "}>";
}


Core::IO::InputSpec Core::IO::InputSpecBuilders::tag(std::string name, ScalarData<bool> data)
{
  return user_defined<bool>(
      name, data,
      // If we encounter the tag, we set the value to true.
      [name](ValueParser& parser, InputParameterContainer& container)
      {
        parser.consume(name);
        container.add(name, true);
      },
      [name](std::ostream& stream, const InputParameterContainer& container) { stream << name; });
}


Core::IO::InputSpec Core::IO::InputSpecBuilders::group(
    std::string name, std::vector<InputSpec> specs, Core::IO::InputSpecBuilders::GroupData data)
{
  assert_unique_or_empty_names(specs);

  InputSpec::CommonData common_data{
      .name = name,
      .description = data.description,
      .required = data.required,
      .has_default_value = all_have_default_values(specs),
  };

  return InputSpec(
      Internal::GroupSpec{.name = name, .data = std::move(data), .specs = std::move(specs)},
      common_data);
}


Core::IO::InputSpec Core::IO::InputSpecBuilders::group(std::vector<InputSpec> specs)
{
  assert_unique_or_empty_names(specs);

  // Generate a description of the form "group {a, b, c}".
  std::string description = "group " + describe_for_error_handling(specs);

  InputSpec::CommonData common_data{
      .name = "",
      .description = description,
      .required = true,
      .has_default_value = all_have_default_values(specs),
  };

  return InputSpec(Internal::GroupSpec{.name = "",
                       .data = {.description = description, .required = true},
                       .specs = std::move(specs)},
      common_data);
}


Core::IO::InputSpec Core::IO::InputSpecBuilders::one_of(std::vector<InputSpec> specs,
    std::function<void(ValueParser& parser, InputParameterContainer& container, std::size_t index)>
        on_parse_callback)
{
  FOUR_C_ASSERT_ALWAYS(!specs.empty(), "`one_of` must contain at least one entry.");

  assert_unique_or_empty_names(specs);

  std::string description = "one_of " + describe_for_error_handling(specs);
  InputSpec::CommonData common_data{
      .name = "",
      .description = description,
      .required = true,
      .has_default_value = false,
  };

  GroupData group_data{
      .description = description,
      .required = true,
  };

  return InputSpec(Internal::OneOfSpec{.data = std::move(group_data),
                       .specs = std::move(specs),
                       .on_parse_callback = on_parse_callback},
      std::move(common_data));
}


FOUR_C_NAMESPACE_CLOSE
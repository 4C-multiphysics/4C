// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_io_input_spec_builders.hpp"

#include "4C_utils_string.hpp"

#include <format>
#include <iostream>
#include <numeric>
#include <set>
#include <unordered_map>
#include <utility>

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
      const auto& name = entry.impl().name();
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
        it->second->impl().parse(parser, container);

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
            entry->impl().parse(parser, container);
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
      FOUR_C_ASSERT_ALWAYS(!entry->impl().required(), "Required '%s' not found in input line",
          entry->impl().description().c_str());
    }

    // Check if all required values have been parsed, i.e., any remaining component must be optional
    for (const auto& [name, entry] : name_to_entry_map)
    {
      if (entry->impl().required())
      {
        FOUR_C_THROW("Required value '%s' not found in input line", name.c_str());
      }
      else if (entry->impl().has_default_value())
      {
        entry->impl().set_default_value(container);
      }
    }
  }

  void assert_unique_or_empty_names(const std::vector<Core::IO::InputSpec>& specs)
  {
    std::set<std::string> names;
    for (const auto& spec : specs)
    {
      if (!spec.impl().name().empty())
        FOUR_C_ASSERT_ALWAYS(names.insert(spec.impl().name()).second,
            "Duplicate component name '%s' found in input line.", spec.impl().name().c_str());
    }
  }

  bool all_have_default_values(const std::vector<Core::IO::InputSpec>& specs)
  {
    return std::all_of(specs.begin(), specs.end(),
        [](const Core::IO::InputSpec& spec) { return spec.impl().has_default_value(); });
  }

  [[nodiscard]] const std::string& describe(const Core::IO::InputSpec& spec)
  {
    return (spec.impl().name().empty()) ? spec.impl().description() : spec.impl().name();
  }

  [[nodiscard]] std::string describe(const std::vector<Core::IO::InputSpec>& specs)
  {
    if (specs.empty()) return "{}";

    std::string description = "{";
    for (const auto& spec : specs)
    {
      // Unnamed InputSpecs are created internally and will have a description that is useful for
      // error messages.
      description += describe(spec);
      description += ", ";
    }
    description.pop_back();
    description.pop_back();
    description += "}";

    return description;
  }

  // Match a vector of specs against a node. Returns true when all specs could be matched.
  // Does not validate that all content of the node was matched as this differs depending on the
  // context.
  bool match_vector_of_specs(const std::vector<Core::IO::InputSpec>& specs,
      Core::IO::ConstYamlNodeRef node, Core::IO::InputParameterContainer& container,
      Core::IO::Internal::MatchEntry& vector_matches)
  {
    bool all_ok = true;
    for (const auto& spec : specs)
    {
      auto& spec_match = vector_matches.append_child(&spec);
      all_ok &= spec.impl().match(node, container, spec_match);
    }
    return all_ok;
  }

  /**
   * Helper class to remember the child nodes of a node and restore this state at a later point,
   * if emitting fails.
   */
  struct ChildNodeCheckpoint
  {
    explicit ChildNodeCheckpoint(Core::IO::YamlNodeRef node)
        : node(node), num_children_before(node.node.num_children())
    {
    }

    void restore()
    {
      for (auto i = node.node.num_children(); i > num_children_before; --i)
      {
        node.node.remove_child(i - 1);
      }
    }

    Core::IO::YamlNodeRef node;
    ryml::id_type num_children_before;
  };

}  // namespace


Core::IO::Internal::MatchTree::MatchTree(const Core::IO::InputSpec& root, ConstYamlNodeRef node)
    : node_(node)
{
  // The number of nodes in the tree is known in advance as it is exactly equal to the
  // number of specs. This is why we can reserve the space for the entries and do not need
  // to deal with reallocations, which would invalidate MatchEntry references.
  const auto total_nodes = root.impl().data.n_specs;
  entries_.reserve(total_nodes);
  entries_.emplace_back(this, &root);
}

namespace
{
  using namespace Core::IO;
  using namespace Core::IO::Internal;
  //! Helper function to dump the match tree for visual inspection during debugging.
  [[maybe_unused]] void dump_match_entry(std::ostream& stream, const MatchEntry& entry, int depth)
  {
    const auto id = std::invoke(
        [&entry]() -> int
        {
          // No special meaning for the negative values; this is just debug output.
          switch (entry.state)
          {
            case MatchEntry::State::defaulted:
              return -3;
            case MatchEntry::State::unmatched:
              return -2;
            case MatchEntry::State::partial:
              return -1;
            case MatchEntry::State::matched:
              return entry.matched_node;
          }
          FOUR_C_ASSERT_ALWAYS(false, "Internal error: unknown MatchEntry::State.");
        });

    std::string indent = std::string(depth, ' ');
    stream << indent << entry.spec->impl().name() << "(matched: " << std::boolalpha
           << (entry.state == MatchEntry::State::matched) << " " << id << ")"
           << ":\n";
    for (const auto* child : entry.children)
    {
      dump_match_entry(stream, *child, depth + 2);
    }
  }

  void recursively_print_match_entries(const MatchEntry& entry, std::ostream& out, int depth)
  {
    constexpr std::array state_symbol = {"[X]", "[ ]", "[!]", "[ ]"};
    constexpr std::array state_phrase = {
        "Expected", "Fully matched", "Partially matched", "Defaulted"};
    std::string indent = std::string(depth, ' ');

    switch (entry.type)
    {
      case MatchEntry::Type::parameter:
      {
        out << indent;
        out << std::format("{} {} parameter '{}'", state_symbol[static_cast<int>(entry.state)],
            state_phrase[static_cast<int>(entry.state)], entry.spec->impl().name());
        if (entry.state == MatchEntry::State::partial)
        {
          out << std::format(
              " (expected type '{}' did not validate)", entry.spec->impl().pretty_type_name());
        }
        out << '\n';
        break;
      }
      case MatchEntry::Type::group:
      {
        out << indent;
        out << std::format("{} {} group '{}'\n", state_symbol[static_cast<int>(entry.state)],
            state_phrase[static_cast<int>(entry.state)], entry.spec->impl().name());
        if (entry.state == MatchEntry::State::partial)
        {
          for (const auto* child : entry.children)
          {
            recursively_print_match_entries(*child, out, depth + 2);
          }
        }
        break;
      }
      case MatchEntry::Type::list:
      {
        out << indent;
        out << std::format("{} {} list '{}'\n", state_symbol[static_cast<int>(entry.state)],
            state_phrase[static_cast<int>(entry.state)], entry.spec->impl().name());
        if (entry.state == MatchEntry::State::partial)
        {
          FOUR_C_ASSERT(entry.children.size() == 1,
              "Internal error: lists should only store one MatchEntry.");

          auto partially_matched_list_node =
              entry.tree->node().node.tree()->ref(entry.matched_node);

          if (entry.children.front()->state == MatchEntry::State::matched)
          {
            auto* list_spec = dynamic_cast<
                const InputSpecTypeErasedImplementation<InputSpecBuilders::Internal::ListSpec>*>(
                &entry.spec->impl());
            FOUR_C_ASSERT(list_spec != nullptr, "Internal error: ListSpec expected.");

            const int n_actual_entries = partially_matched_list_node.num_children();
            const int n_expected_entries = list_spec->wrapped.data.size;
            // If the last entry was matched, this means that the list was too long.
            out << indent << "  ";
            out << std::format("{} Too many list entries encountered: expected {} but matched {}\n",
                state_symbol[static_cast<int>(MatchEntry::State::unmatched)], n_expected_entries,
                n_actual_entries);
          }
          else
          {
            out << indent << "  ";
            out << std::format("{} The following list entry did not match:\n",
                state_symbol[static_cast<int>(MatchEntry::State::partial)]);
            recursively_print_match_entries(*entry.children.front(), out, depth + 4);
          }
        }
        break;
      }
      case MatchEntry::Type::all_of:
      {
        out << indent << "{\n";
        for (const auto* child : entry.children)
        {
          recursively_print_match_entries(*child, out, depth + 2);
        }
        out << indent << "}\n";
        break;
      }
      case MatchEntry::Type::one_of:
      {
        // one_of can print way too much information, so we try to figure out what the best match
        // was.
        std::vector<const MatchEntry*> fully_matching_children;
        std::vector<const MatchEntry*> partially_matching_children;
        for (const auto* child : entry.children)
        {
          if (child->state == MatchEntry::State::matched)
          {
            fully_matching_children.push_back(child);
          }
          else if (child->state == MatchEntry::State::partial)
          {
            partially_matching_children.push_back(child);
          }
        }

        // The happy path with one exact match.
        if (fully_matching_children.size() == 1 && partially_matching_children.empty())
        {
          out << indent
              << std::format("{} Matched exactly one:\n",
                     state_symbol[static_cast<int>(MatchEntry::State::matched)]);
          recursively_print_match_entries(*fully_matching_children.front(), out, depth + 2);
          break;
        }

        if (fully_matching_children.empty() && partially_matching_children.empty())
        {
          // Nothing matched at all, so we just dump all children.
          out << indent
              << std::format("{} Expected one of the following:\n",
                     state_symbol[static_cast<int>(MatchEntry::State::unmatched)]);
          for (const auto* child : entry.children)
          {
            recursively_print_match_entries(*child, out, depth + 2);
          }
        }
        else if (fully_matching_children.size() > 1)
        {
          out << indent
              << std::format("{} Expected exactly one but matched multiple:\n",
                     state_symbol[static_cast<int>(MatchEntry::State::unmatched)]);
          for (const auto* child : fully_matching_children)
          {
            out << indent << "  "
                << std::format("{} Possible match:\n",
                       state_symbol[static_cast<int>(MatchEntry::State::partial)]);

            recursively_print_match_entries(*child, out, depth + 4);
          }
        }
        else
        {
          out << indent
              << std::format("{} Expected exactly one of a few choices but matched:\n",
                     state_symbol[static_cast<int>(MatchEntry::State::unmatched)]);
          for (const auto* child : fully_matching_children)
          {
            recursively_print_match_entries(*child, out, depth + 2);
          }
          for (const auto* child : partially_matching_children)
          {
            recursively_print_match_entries(*child, out, depth + 2);
          }
        }
        break;
      }
      default:
        FOUR_C_ASSERT(false, "Internal error: unknown MatchEntry::Type.");
    }
  }

  void recursively_find_unmatched_nodes(ryml::ConstNodeRef node,
      const std::vector<ryml::id_type>& matched_node_ids,
      std::vector<ryml::id_type>& unmatched_node_ids)
  {
    if (std::find(matched_node_ids.begin(), matched_node_ids.end(), node.id()) ==
        matched_node_ids.end())
    {
      unmatched_node_ids.push_back(node.id());
    }
    else if (node.has_children())
    {
      for (const auto& child : node.children())
      {
        recursively_find_unmatched_nodes(child, matched_node_ids, unmatched_node_ids);
      }
    }
  }

  void recursively_add_node_ids(ryml::ConstNodeRef node, std::vector<ryml::id_type>& child_ids)
  {
    child_ids.push_back(node.id());
    if (node.has_children())
    {
      for (const auto& child : node.children())
      {
        recursively_add_node_ids(child, child_ids);
      }
    }
  }

  std::vector<ryml::id_type> unmatched_input_nodes(
      const std::vector<MatchEntry> entries, ryml::ConstNodeRef node)
  {
    // We assume that the top-level node is always matched. If it is not, this must mean that the
    // InputSpec does not fit at all and this check is pointless because we throw before. Fixing
    // this up here allows to treat logical nodes (without associated input nodes) and real nodes
    // uniformly.
    std::vector<ryml::id_type> matched_node_ids{node.id()};
    for (const auto& entry : entries)
    {
      switch (entry.type)
      {
        case MatchEntry::Type::parameter:
        {
          if (entry.state == MatchEntry::State::matched)
          {
            recursively_add_node_ids(node.tree()->ref(entry.matched_node), matched_node_ids);
          }
          break;
        }
        case MatchEntry::Type::group:
        {
          if (entry.state != MatchEntry::State::unmatched)
          {
            // For groups, only add the grouping node, children will be treated by other
            // MatchEntries.
            matched_node_ids.push_back(entry.matched_node);
          }
          break;
        }
        case MatchEntry::Type::list:
        {
          if (entry.state == MatchEntry::State::matched)
          {
            // If the list is matched, just add all child nodes as well
            recursively_add_node_ids(node.tree()->ref(entry.matched_node), matched_node_ids);
          }
          break;
        }
        case MatchEntry::Type::selection:
        {
          if (entry.state == MatchEntry::State::matched)
          {
            matched_node_ids.push_back(entry.matched_node);
          }
        }
        default:
          break;
      }
    }

    // Now compare against all the nodes that are in the input yaml tree.
    std::vector<ryml::id_type> unmatched_node_ids;
    recursively_find_unmatched_nodes(node, matched_node_ids, unmatched_node_ids);
    return unmatched_node_ids;
  }
}  // namespace

Core::IO::Internal::MatchEntry& Core::IO::Internal::MatchTree::append_child(
    const Core::IO::InputSpec* spec)
{
  FOUR_C_ASSERT(
      entries_.size() < entries_.capacity(), "Internal error: too many entries in MatchTree.");
  return entries_.emplace_back(this, spec);
}

void Core::IO::Internal::MatchTree::dump(std::ostream& stream) const
{
  // Start at the root node and do depth-first traversal.
  dump_match_entry(stream, entries_.front(), 0);
}



void Core::IO::Internal::MatchTree::assert_match() const
{
  std::stringstream ss;
  ss << "Could not match this input\n\n";
  ss << node_.node << "\n\n";
  ss << "against the given input specification. ";

  if (entries_.front().state != MatchEntry::State::matched &&
      entries_.front().state != MatchEntry::State::defaulted)
  {
    ss << "This was the best attempt to match the input:\n\n";
    recursively_print_match_entries(entries_.front(), ss, 0);
    FOUR_C_THROW("%s", ss.str().c_str());
  }

  // Check that everything in the input was actually used.
  auto unmatched_node_ids = unmatched_input_nodes(entries_, node_.node);
  if (!unmatched_node_ids.empty())
  {
    ss << "The following parts of the input did not match:\n\n";

    for (const auto& id : unmatched_node_ids)
    {
      ss << node_.node.tree()->ref(id) << "\n";
    }
    FOUR_C_THROW("%s", ss.str().c_str());
  }
}

void Core::IO::Internal::MatchTree::erase_everything_after(const MatchEntry& entry)
{
  // We know that the entry must be stored in the memory of entries_. This means we can find the
  // entry by pointer comparison.
  auto it = std::find_if(entries_.begin(), entries_.end(),
      [&entry](const MatchEntry& stored_entry) { return &stored_entry == &entry; });
  FOUR_C_ASSERT(it != entries_.end(), "Internal error: entry not found in MatchTree.");

  // Erase everything that follows the entry.
  entries_.erase(it + 1, entries_.end());
}



Core::IO::Internal::MatchEntry& Core::IO::Internal::MatchEntry::append_child(
    const InputSpec* in_spec)
{
  auto& child = tree->append_child(in_spec);
  children.push_back(&child);
  return child;
}

void MatchEntry::reset()
{
  state = State::unmatched;
  type = Type::unknown;
  matched_node = ryml::npos;
  children.clear();
  tree->erase_everything_after(*this);
}


InputSpecTypeErasedBase::InputSpecTypeErasedBase(InputSpecTypeErasedBase::CommonData data)
    : data(std::move(data))
{
  const auto check_for_unprintable_chars =
      [](const std::string& check_string, const auto& field_type)
  {
    auto first_unprintable_char =
        std::ranges::find_if_not(check_string, [](char c) { return std::isprint(c) || c == '\n'; });
    if (first_unprintable_char != check_string.end())
    {
      std::stringstream escaped_string;
      for (const auto c : check_string)
      {
        if (std::isprint(c))
        {
          escaped_string << c;
        }
        else if (c == '\n')
        {
          escaped_string << "\\n";
        }
        else
        {
          escaped_string << '<' << "0x" << std::hex << static_cast<int>(c) << '>';
        }
      }
      FOUR_C_THROW(
          "A %s may only consist of printable characters. Here is the offending "
          "%s with unprintable characters marked in angle brackets:\n'%s'",
          field_type, field_type, escaped_string.str().c_str());
    }
  };

  check_for_unprintable_chars(this->data.description, "description");
  check_for_unprintable_chars(this->data.name, "name");
}

std::string Core::IO::Internal::InputSpecTypeErasedBase::description_one_line() const
{
  return Core::Utils::trim(data.description);
}


void Core::IO::InputSpecBuilders::Internal::GroupSpec::parse(
    Core::IO::ValueParser& parser, Core::IO::InputParameterContainer& container) const
{
  FOUR_C_ASSERT(!name.empty(), "Internal error: group name must not be empty.");

  parser.consume(name);

  // Parse into a separate container to avoid side effects if parsing fails.
  Core::IO::InputParameterContainer subcontainer;
  parse_in_arbitrary_order(parser, specs, subcontainer);

  container.group(name) = subcontainer;
}

bool Core::IO::InputSpecBuilders::Internal::GroupSpec::match(ConstYamlNodeRef node,
    Core::IO::InputParameterContainer& container, IO::Internal::MatchEntry& match_entry) const
{
  match_entry.type = IO::Internal::MatchEntry::Type::group;
  FOUR_C_ASSERT(!name.empty(), "Internal error: group name must not be empty.");
  const auto group_name = ryml::to_csubstr(name);

  const bool group_node_is_input = node.node.has_key() && (node.node.key() == group_name);
  const bool group_exists_nested =
      node.node.is_map() && node.node.has_child(group_name) && node.node[group_name].is_map();

  if (!group_exists_nested && !group_node_is_input)
  {
    if (data.defaultable)
    {
      match_entry.state = IO::Internal::MatchEntry::State::defaulted;
      set_default_value(container);
    }
    return !data.required.value();
  }

  auto group_node = group_node_is_input ? node : node.wrap(node.node[group_name]);

  // Matching the key of the group is at least a partial match.
  match_entry.state = IO::Internal::MatchEntry::State::partial;
  match_entry.matched_node = group_node.node.id();

  // Parse into a separate container to avoid side effects if parsing fails.
  InputParameterContainer subcontainer;
  bool all_matched = match_vector_of_specs(specs, group_node, subcontainer, match_entry);

  if (!all_matched)
  {
    // Match will stay a partial match.
    return false;
  }

  // Everything was correctly matched, so mark the group node as matched.
  match_entry.state = IO::Internal::MatchEntry::State::matched;

  container.group(name) = subcontainer;
  return true;
}


void Core::IO::InputSpecBuilders::Internal::GroupSpec::set_default_value(
    Core::IO::InputParameterContainer& container) const
{
  FOUR_C_ASSERT(!name.empty(), "Internal error: group name must not be empty.");
  for (const auto& spec : specs)
  {
    if (spec.impl().has_default_value()) spec.impl().set_default_value(container.group(name));
  }
}


void Core::IO::InputSpecBuilders::Internal::GroupSpec::print(
    std::ostream& stream, std::size_t indent) const
{
  stream << "// " << std::string(indent, ' ') << name << ":\n";
  indent += 2;

  for (const auto& spec : specs)
  {
    spec.impl().print(stream, indent);
  }
}

void Core::IO::InputSpecBuilders::Internal::GroupSpec::emit_metadata(ryml::NodeRef node) const
{
  node |= ryml::MAP;
  node["name"] << name;

  node["type"] = "group";
  if (!data.description.empty())
  {
    node["description"] << data.description;
  }
  emit_value_as_yaml(node["required"], data.required.value());
  emit_value_as_yaml(node["defaultable"], data.defaultable);
  node["specs"] |= ryml::SEQ;
  {
    for (const auto& spec : specs)
    {
      auto child = node["specs"].append_child();
      child |= ryml::MAP;
      spec.impl().emit_metadata(child);
    };
  }
}


bool InputSpecBuilders::Internal::GroupSpec::emit(YamlNodeRef node,
    const InputParameterContainer& container, const InputSpecEmitOptions& options) const
{
  node.node |= ryml::MAP;
  ChildNodeCheckpoint checkpoint(node);
  if (container.has_group(name))
  {
    auto group_node = node.node.append_child();
    group_node << ryml::key(name);
    group_node |= ryml::MAP;
    for (const auto& spec : specs)
    {
      if (!spec.impl().emit(node.wrap(group_node), container.group(name), options))
      {
        checkpoint.restore();
        return false;
      }
    }
    // If no children were added, remove the group node as well, unless it is required.
    if (!data.required.value() && group_node.num_children() == 0)
    {
      checkpoint.restore();
    }
    return true;
  }
  // If group is not present, success depends on whether it is required.
  return !data.required.value();
}


void Core::IO::InputSpecBuilders::Internal::AllOfSpec::parse(
    Core::IO::ValueParser& parser, Core::IO::InputParameterContainer& container) const
{
  // Parse into a separate container to avoid side effects if parsing fails.
  InputParameterContainer subcontainer;
  parse_in_arbitrary_order(parser, specs, subcontainer);
  container.merge(subcontainer);
}


bool Core::IO::InputSpecBuilders::Internal::AllOfSpec::match(ConstYamlNodeRef node,
    Core::IO::InputParameterContainer& container, IO::Internal::MatchEntry& match_entry) const
{
  match_entry.type = IO::Internal::MatchEntry::Type::all_of;
  std::vector<ryml::id_type> parsed_node_ids;

  // Parse into a separate container to avoid side effects if parsing fails.
  InputParameterContainer subcontainer;
  bool all_matched = match_vector_of_specs(specs, node, subcontainer, match_entry);

  if (!all_matched)
  {
    return false;
  }

  // N.B. an all_of does not constitute a full yaml object, so we cannot say anything about the
  // match state of the node.
  match_entry.state = IO::Internal::MatchEntry::State::matched;

  container.merge(subcontainer);

  return true;
}


void Core::IO::InputSpecBuilders::Internal::AllOfSpec::set_default_value(
    Core::IO::InputParameterContainer& container) const
{
  for (const auto& spec : specs)
  {
    if (spec.impl().has_default_value()) spec.impl().set_default_value(container);
  }
}


void Core::IO::InputSpecBuilders::Internal::AllOfSpec::print(
    std::ostream& stream, std::size_t indent) const
{
  // Only print an "<all_of>" header if it is not present at the top-level.
  if (indent > 0)
  {
    stream << "// " << std::string(indent, ' ') << "<all_of>:\n";
    indent += 2;
  }

  for (const auto& spec : specs)
  {
    spec.impl().print(stream, indent);
  }
}

void Core::IO::InputSpecBuilders::Internal::AllOfSpec::emit_metadata(ryml::NodeRef node) const
{
  node |= ryml::MAP;

  node["type"] = "all_of";
  if (!data.description.empty())
  {
    node["description"] << data.description;
  }
  emit_value_as_yaml(node["required"], data.required.value());
  node["specs"] |= ryml::SEQ;
  {
    for (const auto& spec : specs)
    {
      auto child = node["specs"].append_child();
      child |= ryml::MAP;
      spec.impl().emit_metadata(child);
    }
  }
}


bool InputSpecBuilders::Internal::AllOfSpec::emit(YamlNodeRef node,
    const InputParameterContainer& container, const InputSpecEmitOptions& options) const
{
  node.node |= ryml::MAP;
  ChildNodeCheckpoint checkpoint(node);
  for (const auto& spec : specs)
  {
    if (!spec.impl().emit(node, container, options))
    {
      checkpoint.restore();
      return false;
    }
  }
  return true;
}


void Core::IO::InputSpecBuilders::Internal::OneOfSpec::parse(
    Core::IO::ValueParser& parser, Core::IO::InputParameterContainer& container) const
{
  ValueParser::BacktrackScope backtrack_scope(parser);

  // Try to parse a component and backtrack if it fails.
  const auto try_parse = [&](const Core::IO::InputSpec& spec) -> bool
  {
    try
    {
      spec.impl().parse(parser, container);
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
              describe(*component).c_str(), describe(*other).c_str());
        }
      }

      // If we reach this point, we have successfully parsed the input.
      std::size_t index = std::distance(specs.begin(), component);
      if (on_parse_callback) on_parse_callback(container, index);

      return;
    }
  }

  // Convert remainder into a null-terminated string for the error message.
  std::string remainder(parser.get_unparsed_remainder());
  FOUR_C_THROW("While parsing '%s'.\nNone of the specs fit the input. Expected %s",
      remainder.c_str(), data.description.c_str());
}


bool Core::IO::InputSpecBuilders::Internal::OneOfSpec::match(ConstYamlNodeRef node,
    Core::IO::InputParameterContainer& container, IO::Internal::MatchEntry& match_entry) const
{
  match_entry.type = IO::Internal::MatchEntry::Type::one_of;
  InputParameterContainer subcontainer;

  std::size_t matched_index = 0;
  for (; matched_index < specs.size(); ++matched_index)
  {
    auto& spec_match = match_entry.append_child(&specs[matched_index]);
    subcontainer.clear();
    if (specs[matched_index].impl().match(node, subcontainer, spec_match))
    {
      break;
    }
  }

  // If we have reached the end of the loop, no spec could be matched.
  if (matched_index == specs.size())
  {
    return false;
  }

  // Check that no other spec can be matched.
  {
    InputParameterContainer dummy_container;
    for (std::size_t j = matched_index + 1; j < specs.size(); ++j)
    {
      auto& spec_match = match_entry.append_child(&specs[j]);
      dummy_container.clear();
      if (specs[j].impl().match(node, dummy_container, spec_match))
      {
        return false;
      }
    }
  }

  // Match succeeded.
  if (on_parse_callback) on_parse_callback(subcontainer, matched_index);
  match_entry.state = IO::Internal::MatchEntry::State::matched;
  container.merge(subcontainer);
  return true;
}


void Core::IO::InputSpecBuilders::Internal::OneOfSpec::set_default_value(
    Core::IO::InputParameterContainer& container) const
{
  FOUR_C_THROW("Implementation error: OneOfSpec cannot have a default value.");
}


void Core::IO::InputSpecBuilders::Internal::OneOfSpec::print(
    std::ostream& stream, std::size_t indent) const
{
  stream << "// " << std::string(indent, ' ') << "<one_of>:\n";
  for (const auto& spec : specs)
  {
    spec.impl().print(stream, indent + 2);
  }
}

void Core::IO::InputSpecBuilders::Internal::OneOfSpec::emit_metadata(ryml::NodeRef node) const
{
  node |= ryml::MAP;

  node["type"] << "one_of";
  if (!data.description.empty())
  {
    node["description"] << data.description;
  }
  node["specs"] |= ryml::SEQ;
  for (const auto& spec : specs)
  {
    auto child = node["specs"].append_child();
    spec.impl().emit_metadata(child);
  }
}


bool InputSpecBuilders::Internal::OneOfSpec::emit(YamlNodeRef node,
    const InputParameterContainer& container, const InputSpecEmitOptions& options) const
{
  node.node |= ryml::MAP;
  ChildNodeCheckpoint checkpoint(node);

  for (const auto& spec : specs)
  {
    if (spec.impl().emit(node, container, options))
    {
      return true;
    }
    else
    {
      checkpoint.restore();
    }
  }

  return false;
}


void Core::IO::InputSpecBuilders::Internal::ListSpec::parse(
    Core::IO::ValueParser& parser, Core::IO::InputParameterContainer& container) const
{
  FOUR_C_ASSERT_ALWAYS(
      data.size != dynamic_size, "ListSpec with dynamic size cannot be parsed from dat.");

  parser.consume(name);

  std::vector<InputParameterContainer> container_list(data.size);
  for (int i = 0; i < data.size; ++i)
  {
    spec.impl().parse(parser, container_list[i]);
  }
  container.add_list(name, std::move(container_list));
}


bool Core::IO::InputSpecBuilders::Internal::ListSpec::match(ConstYamlNodeRef node,
    Core::IO::InputParameterContainer& container, IO::Internal::MatchEntry& match_entry) const
{
  match_entry.type = IO::Internal::MatchEntry::Type::list;
  FOUR_C_ASSERT(!name.empty(), "Internal error: list name must not be empty.");
  const auto list_name = ryml::to_csubstr(name);

  const bool list_node_is_input =
      node.node.has_key() && (node.node.key() == list_name) && node.node.is_seq();
  const bool list_exists_nested =
      node.node.is_map() && node.node.has_child(list_name) && node.node[list_name].is_seq();
  if (!list_exists_nested && !list_node_is_input)
  {
    return !data.required;
  }

  auto list_node = list_node_is_input ? node : node.wrap(node.node[list_name]);
  // Matching the key is at least a partial match.
  match_entry.state = IO::Internal::MatchEntry::State::partial;
  match_entry.matched_node = list_node.node.id();

  std::vector<InputParameterContainer> container_list;

  const std::size_t max_entries = data.size > 0 ? data.size : std::numeric_limits<int>::max();

  // Parse all child nodes of the list node. Reuse the same MatchEntry.
  auto& child_match_entry = match_entry.append_child(&spec);
  for (const auto child : list_node.node.children())
  {
    // Always reset the match entry. We only store the last match attempt for lists.
    child_match_entry.reset();
    bool child_matched =
        spec.impl().match(list_node.wrap(child), container_list.emplace_back(), child_match_entry);
    if (!child_matched)
    {
      // Match will stay a partial match and the MatchEntry will have more details.
      return false;
    }
  }

  if (container_list.size() > max_entries)
  {
    // The list is too long. This is a partial match, and we can figure out the reason in the
    // error message constructed by the MatchTree.
    return false;
  }

  // Everything was correctly matched, so mark the list node as matched.
  container.add_list(name, std::move(container_list));
  match_entry.state = IO::Internal::MatchEntry::State::matched;

  return true;
}


void Core::IO::InputSpecBuilders::Internal::ListSpec::set_default_value(
    Core::IO::InputParameterContainer& container) const
{
  std::vector<InputParameterContainer> default_list(data.size);

  for (auto& subcontainer : default_list)
  {
    spec.impl().set_default_value(subcontainer);
  }

  container.add_list(name, std::move(default_list));
}


void Core::IO::InputSpecBuilders::Internal::ListSpec::print(
    std::ostream& stream, std::size_t indent) const
{
  stream << "// " << std::string(indent, ' ') << "list '" << name << "'"
         << " with entries:\n";
  spec.impl().print(stream, indent + 2);
}

void Core::IO::InputSpecBuilders::Internal::ListSpec::emit_metadata(ryml::NodeRef node) const
{
  node |= ryml::MAP;

  node["name"] << name;

  node["type"] << "list";
  if (!data.description.empty())
  {
    node["description"] << Core::Utils::trim(data.description);
  }
  emit_value_as_yaml(node["required"], data.required);
  if (data.size > 0) node["size"] << data.size;
  node["spec"] |= ryml::MAP;
  spec.impl().emit_metadata(node["spec"]);
}


bool InputSpecBuilders::Internal::ListSpec::emit(YamlNodeRef node,
    const InputParameterContainer& container, const InputSpecEmitOptions& options) const
{
  node.node |= ryml::MAP;
  ChildNodeCheckpoint checkpoint(node);
  if (const auto* list = container.get_if<InputParameterContainer::List>(name); list)
  {
    auto list_node = node.node.append_child();
    list_node << ryml::key(name);
    list_node |= ryml::SEQ;

    for (const auto& subcontainer : *list)
    {
      if (!spec.impl().emit(node.wrap(list_node.append_child()), subcontainer, options))
      {
        checkpoint.restore();
        return false;
      }
    }
    return true;
  }

  // If list is not present, success depends on whether it is required.
  return !data.required;
}


namespace
{
  // Nesting all_of or one_of within another spec of that same type is the same as pulling the
  // nested specs up into the parent spec. An additional predicate can be used to filter the nested
  // specs that may be flattened.
  template <typename SpecType>
  std::vector<Core::IO::InputSpec> flatten_nested(std::vector<Core::IO::InputSpec> specs,
      std::function<bool(const SpecType&)> predicate = nullptr)
  {
    std::vector<Core::IO::InputSpec> flattened_specs;
    for (auto&& spec : specs)
    {
      if (auto* nested =
              dynamic_cast<Core::IO::Internal::InputSpecTypeErasedImplementation<SpecType>*>(
                  &spec.impl());
          nested && (!predicate || predicate(nested->wrapped)))
      {
        for (auto&& sub_spec : nested->wrapped.specs)
        {
          flattened_specs.emplace_back(std::move(sub_spec));
        }
      }
      else
      {
        flattened_specs.emplace_back(std::move(spec));
      }
    }

    return flattened_specs;
  }

  std::size_t count_contained_specs(const std::vector<Core::IO::InputSpec>& specs)
  {
    return std::accumulate(specs.begin(), specs.end(), 0u,
        [](std::size_t acc, const auto& spec) { return acc + spec.impl().data.n_specs; });
  }
}  // namespace


Core::IO::InputSpec Core::IO::InputSpecBuilders::group(
    std::string name, std::vector<InputSpec> specs, Core::IO::InputSpecBuilders::GroupData data)
{
  auto flattened_specs = flatten_nested<Internal::AllOfSpec>(std::move(specs));

  assert_unique_or_empty_names(flattened_specs);

  if (!data.required.has_value())
  {
    data.required = !data.defaultable;
  }
  if (data.required.value() && data.defaultable)
  {
    FOUR_C_THROW("Group '%s': a group cannot be both required and defaultable.", name.c_str());
  }
  if (data.defaultable && !all_have_default_values(flattened_specs))
  {
    FOUR_C_THROW(
        "Group '%s': a group cannot be defaultable if not all of its child specs have default "
        "values.",
        name.c_str());
  }

  IO::Internal::InputSpecTypeErasedBase::CommonData common_data{
      .name = name,
      .description = data.description,
      .required = data.required.value(),
      .has_default_value = data.defaultable,
      .n_specs = count_contained_specs(flattened_specs) + 1,
  };

  return IO::Internal::make_spec(
      Internal::GroupSpec{
          .name = name, .data = std::move(data), .specs = std::move(flattened_specs)},
      common_data);
}


Core::IO::InputSpec Core::IO::InputSpecBuilders::all_of(std::vector<InputSpec> specs)
{
  auto flattened_specs = flatten_nested<Internal::AllOfSpec>(std::move(specs));

  if (flattened_specs.size() == 1)
  {
    return flattened_specs[0];
  }

  assert_unique_or_empty_names(flattened_specs);

  const bool any_required =
      std::ranges::any_of(flattened_specs, [](const auto& spec) { return spec.impl().required(); });

  // Generate a description of the form "group {a, b, c}".
  std::string description = "all_of " + describe(flattened_specs);

  IO::Internal::InputSpecTypeErasedBase::CommonData common_data{
      .name = "",
      .description = description,
      .required = any_required,
      .has_default_value = all_have_default_values(flattened_specs),
      .n_specs = count_contained_specs(flattened_specs) + 1,
  };

  return IO::Internal::make_spec(
      Internal::AllOfSpec{
          .data = {.description = description, .required = any_required},
          .specs = std::move(flattened_specs),
      },
      common_data);
}


Core::IO::InputSpec Core::IO::InputSpecBuilders::one_of(std::vector<InputSpec> specs,
    std::function<void(InputParameterContainer& container, std::size_t index)> on_parse_callback)
{
  // We can only flatten the specs if there is no custom on_parse_callback, either for this
  // one_of or any nested one_of.
  auto flattened_specs = on_parse_callback ? specs
                                           : flatten_nested<Internal::OneOfSpec>(std::move(specs),
                                                 [](const Internal::OneOfSpec& one_of_spec)
                                                 { return !one_of_spec.on_parse_callback; });

  FOUR_C_ASSERT_ALWAYS(!flattened_specs.empty(), "A `one_of` must contain entries.");

  if (flattened_specs.size() == 1)
  {
    return flattened_specs[0];
  }

  assert_unique_or_empty_names(flattened_specs);

  // Assert that all specs are required.
  const bool all_required =
      std::ranges::all_of(flattened_specs, [](const auto& spec) { return spec.impl().required(); });
  if (!all_required)
  {
    std::string non_required;
    for (const auto& spec : flattened_specs)
    {
      if (!spec.impl().required()) non_required += "    " + describe(spec) + "\n";
    }

    FOUR_C_THROW(
        "All specs in a 'one_of' must be required to avoid confusion. The following InputSpecs "
        "are not required:\n%s",
        non_required.c_str());
  }

  std::string description = "one_of " + describe(flattened_specs);
  IO::Internal::InputSpecTypeErasedBase::CommonData common_data{
      .name = "",
      .description = description,
      .required = true,
      .has_default_value = false,
      .n_specs = count_contained_specs(flattened_specs) + 1,
  };

  GroupData group_data{
      .description = description,
      .required = true,
  };


  return IO::Internal::make_spec(Internal::OneOfSpec{.data = std::move(group_data),
                                     .specs = std::move(flattened_specs),
                                     .on_parse_callback = std::move(on_parse_callback)},
      std::move(common_data));
}

Core::IO::InputSpec Core::IO::InputSpecBuilders::list(
    std::string name, Core::IO::InputSpec spec, ListData data)
{
  IO::Internal::InputSpecTypeErasedBase::CommonData common_data{
      .name = name,
      .description = data.description,
      .required = data.required,
      // We can only set a default value if the size is fixed and the contained spec has
      // a default value.
      .has_default_value = spec.impl().has_default_value() && data.size != dynamic_size,
      .n_specs = spec.impl().data.n_specs + 1,
  };

  return IO::Internal::make_spec(
      Internal::ListSpec{
          .name = name,
          .spec = std::move(spec),
          .data = std::move(data),
      },
      std::move(common_data));
}


FOUR_C_NAMESPACE_CLOSE

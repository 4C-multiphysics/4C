// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config_revision.hpp"

#include "4C_io_input_file_utils.hpp"

#include "4C_io_input_file.hpp"

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_StrUtils.hpp>
#include <yaml-cpp/yaml.h>

FOUR_C_NAMESPACE_OPEN

namespace
{
  void print_dat_impl(std::ostream& stream, const Teuchos::ParameterList& list,
      const std::string& parentname, bool comment);

  void print_documentation(std::ostream& stream, const Teuchos::ParameterEntry& entry)
  {
    // Helper function to print documentation
    std::string doc = entry.docString();
    if (!doc.empty())
    {
      Teuchos::StrUtils::printLines(stream, "// ", doc);
    }
  }


  void print_sublist(std::ostream& stream, const std::string& parentname, const std::string& name,
      const Teuchos::ParameterList& list, bool comment)
  {
    // Helper function to print a sublist
    std::string secname = parentname;
    if (!secname.empty()) secname += "/";
    secname += name;
    unsigned l = secname.length();
    stream << "--" << std::string(std::max<int>(65 - l, 0), '-');
    stream << secname << "\n";
    print_dat_impl(stream, list.sublist(name), secname, comment);
  }

  void print_parameter(std::ostream& stream, const Teuchos::ParameterEntry& entry,
      const std::string& name, const Teuchos::ParameterList& list, bool comment)
  {
    // Retrieve the parameter entry's validator (if any)
    Teuchos::RCP<const Teuchos::ParameterEntryValidator> validator = entry.validator();

    // Print comments if requested
    if (comment)
    {
      // Check if the validator has valid string values
      if (validator != Teuchos::null)
      {
        Teuchos::RCP<const Teuchos::Array<std::string>> validValues =
            validator->validStringValues();

        // If valid values exist, print them
        if (validValues != Teuchos::null)
        {
          unsigned totalLength = 0;
          // Calculate the total length of all valid values
          for (const auto& value : *validValues)
          {
            totalLength += value.length() + 1;  // Include space/comma
          }
          // Print valid values in a compact or expanded format based on total length
          if (totalLength < 74)
          {
            // Print all values in a single line, separated by commas
            stream << "//     ";
            for (auto it = validValues->begin(); it != validValues->end(); ++it)
            {
              stream << *it;
              if (std::next(it) != validValues->end())
              {
                stream << ",";  // Add a comma if it's not the last element
              }
            }
            stream << "\n";
          }
          else
          {
            // Print each value on a new line
            for (const auto& value : *validValues)
            {
              stream << "//     " << value << '\n';
            }
          }
        }
      }
    }

    // Print the parameter's name and value
    const Teuchos::any& value = entry.getAny(false);
    stream << name;
    unsigned nameLength = name.length();
    // Ensure proper spacing for alignment
    stream << std::string(std::max<int>(31 - nameLength, 0), ' ');

    // Optionally print an equal sign if needed
    if (Core::IO::InputFileUtils::need_to_print_equal_sign(list)) stream << " =";

    try
    {
      // print true/false for bool values to distinguish them from type int
      if (value.type() == typeid(bool))
      {
        stream << " " << (Teuchos::any_cast<bool>(value) ? "true" : "false") << "\n";
      }
      else
      {
        // For non-boolean types, print the value directly
        stream << " " << value << "\n";
      }
    }
    catch (const Teuchos::NonprintableTypeException&)
    {
      // Handle non-printable enum class types
      stream << value.typeName() << "\n";
    }
  }

  void print_dat_impl(std::ostream& stream, const Teuchos::ParameterList& list,
      const std::string& parentname, bool comment)
  {
    // Main loop over the parameter list that calls the helper functions to print
    // documentation, sublists or parameters:
    //
    // Iterate through the parameter list in two distinct phases to ensure proper ordering and
    // handling:
    // - **Phase 0**:
    //    Print all parameters that are not sublists. This ensures that top-level parameters
    //    are written to stream first, without any nested content interfering.
    // - **Phase 1**:
    //    Recursively handle and print all sublists. This phase is executed after all non-sublists
    //    have been processed, allowing sublists to be printed in their hierarchical order.
    //
    // By separating the iteration into these phases, we avoid issues with alphabetical
    // ordering that could cause invalid output sequences for nested lists.
    for (int iterationPhase = 0; iterationPhase < 2; ++iterationPhase)
    {
      for (auto paramIter = list.begin(); paramIter != list.end(); ++paramIter)
      {
        const Teuchos::ParameterEntry& entry = list.entry(paramIter);
        const std::string& name = list.name(paramIter);

        if ((entry.isList() && iterationPhase == 0) || (!entry.isList() && iterationPhase == 1))
        {
          continue;
        }
        if (comment)
        {
          stream << "//\n";
          print_documentation(stream, entry);
        }
        if (entry.isList())
        {
          print_sublist(stream, parentname, name, list, comment);
        }
        else
        {
          print_parameter(stream, entry, name, list, comment);
        }
      }
    }
    stream << std::endl;
  }


  void recursively_determine_sublists(const Teuchos::ParameterList& list,
      std::vector<std::pair<std::string, const Teuchos::ParameterList*>>& sublists,
      const std::string& parent_section_name = "")
  {
    for (const auto& key_value : list)
    {
      const Teuchos::ParameterEntry& entry = key_value.second;
      const std::string& name = key_value.first;
      if (entry.isList())
      {
        const std::string current_section_full_name =
            (parent_section_name == "") ? name : parent_section_name + "/" + name;

        sublists.emplace_back(current_section_full_name, &list.sublist(name));
        recursively_determine_sublists(list.sublist(name), sublists, current_section_full_name);
      }
    }
  }


  void print_metadata_yaml_impl(YAML::Emitter& yaml, const Teuchos::ParameterList& list,
      const std::string& parent_section_name)
  {
    // prevent invalid ordering of parameters caused by alphabetical output:
    // determine all sublists first to pull them out onto the same indentation level
    std::vector<std::pair<std::string, const Teuchos::ParameterList*>> sublists;
    recursively_determine_sublists(list, sublists);



    const auto print_key_value = [&](const std::string& key, const Teuchos::ParameterEntry& entry)
    {
      const auto to_string = [](const Teuchos::any& any)
      {
        std::stringstream s;
        s << any;
        return s.str();
      };

      yaml << YAML::Key << key;
      yaml << YAML::Value << YAML::BeginMap;

      const Teuchos::any& v = entry.getAny(false);
      yaml << YAML::Value << "type" << YAML::Value << v.typeName();

      yaml << YAML::Value << "default" << YAML::Value << to_string(v);

      std::string doc = entry.docString();
      if (doc != "")
      {
        yaml << YAML::Key << "description" << YAML::Value << doc;
      }

      Teuchos::RCP<const Teuchos::ParameterEntryValidator> validator = entry.validator();
      if (validator != Teuchos::null)
      {
        Teuchos::RCP<const Teuchos::Array<std::string>> values = validator->validStringValues();
        if (values != Teuchos::null)
        {
          yaml << YAML::Key << "valid options";
          yaml << YAML::Value << YAML::BeginSeq;
          for (int i = 0; i < (int)values->size(); ++i)
          {
            yaml << (*values)[i];
          }
          yaml << YAML::EndSeq;
        }
      }
      yaml << YAML::EndMap;
    };

    yaml << YAML::BeginMap;
    for (const auto& [name, sublist] : sublists)
    {
      yaml << YAML::Key << name;
      yaml << YAML::Value << YAML::BeginMap;
      for (const auto& key_value : *sublist)
      {
        if (!key_value.second.isList()) print_key_value(key_value.first, key_value.second);
      }
      yaml << YAML::EndMap;
    }
    yaml << YAML::EndMap;
  }
}  // namespace

void Core::IO::InputFileUtils::print_section_header(std::ostream& out, const std::string& header)
{
  constexpr std::size_t max_line_width = 65ul;
  FOUR_C_ASSERT_ALWAYS(header.length() <= max_line_width, "Header '%s' too long", header.c_str());

  out << "--";
  out << std::string(std::max(max_line_width - header.length(), 0ul), '-');
  out << header << '\n';
}



void Core::IO::InputFileUtils::print_section(std::ostream& out, const std::string& header,
    const std::vector<Input::LineDefinition>& possible_lines)
{
  print_section_header(out, header);

  for (const auto& line : possible_lines)
  {
    out << "// ";
    line.print(out);
    out << '\n';
  }
}


void Core::IO::InputFileUtils::print_dat(
    std::ostream& stream, const Teuchos::ParameterList& list, bool comment)
{
  print_dat_impl(stream, list, "", comment);
}


void Core::IO::InputFileUtils::print_metadata_yaml(
    std::ostream& stream, const Teuchos::ParameterList& list)
{
  YAML::Emitter yaml(stream);
  yaml << YAML::BeginMap;
  {
    // First write some metadata
    yaml << YAML::Key << "metadata" << YAML::Value;
    yaml << YAML::BeginMap;
    {
      yaml << YAML::Key << "commit_hash" << YAML::Value << VersionControl::git_hash;
    }
    yaml << YAML::EndMap;
  }

  {
    // Then write the key-value parameters.
    yaml << YAML::Key << "parameters" << YAML::Value;
    print_metadata_yaml_impl(yaml, list, "");
  }
  yaml << YAML::EndMap;
  yaml << YAML::Newline;
}


bool Core::IO::InputFileUtils::need_to_print_equal_sign(const Teuchos::ParameterList& list)
{
  // Helper function to check if string contains a space.
  const auto string_has_space = [](const std::string& s)
  { return std::any_of(s.begin(), s.end(), [](unsigned char c) { return std::isspace(c); }); };

  return std::any_of(list.begin(), list.end(),
      [&](const auto& it)
      {
        // skip entries that are lists: they are allowed to have spaces
        if (it.second.isList()) return false;

        const std::string& name = it.key;

        const Teuchos::RCP<const Teuchos::Array<std::string>>& values_ptr =
            it.second.validator()->validStringValues();

        const bool value_has_space =
            (values_ptr != Teuchos::null) &&
            std::any_of(values_ptr->begin(), values_ptr->end(), string_has_space);

        return value_has_space || string_has_space(name);
      });
}


std::vector<Input::LineDefinition> Core::IO::InputFileUtils::read_all_lines_in_section(
    Core::IO::InputFile& input, const std::string& section,
    const std::vector<Input::LineDefinition>& possible_lines)
{
  auto [parsed_lines, unparsed_lines] =
      read_matching_lines_in_section(input, section, possible_lines);

  // In this function, encountering unparsed lines is an error, so construct a nice message.
  if (unparsed_lines.size() > 0)
  {
    std::stringstream out;
    out << "Read failed in section " << std::quoted(section) << '\n';
    for (const auto& unparsed : unparsed_lines)
    {
      out << "  " << std::quoted(unparsed) << '\n';
    }
    out << "Valid lines are:\n";
    std::for_each(possible_lines.begin(), possible_lines.end(),
        [&](const Input::LineDefinition& def)
        {
          def.print(out);
          out << '\n';
        });
    FOUR_C_THROW(out.str().c_str());
  }

  return parsed_lines;
}


std::pair<std::vector<Input::LineDefinition>, std::vector<std::string>>
Core::IO::InputFileUtils::read_matching_lines_in_section(Core::IO::InputFile& input,
    const std::string& section, const std::vector<Input::LineDefinition>& possible_lines)
{
  std::vector<std::string> unparsed_lines;
  std::vector<Input::LineDefinition> parsed_lines;

  const auto process_line = [&](const std::string& input_line)
  {
    for (const auto& definition : possible_lines)
    {
      std::stringstream l{input_line};

      // Make a copy that potentially gets filled by the Read.
      auto parsed_definition = definition;
      if (parsed_definition.read(l))
      {
        parsed_lines.emplace_back(std::move(parsed_definition));
        return;
      }
    }
    // None of the possible lines matched.
    unparsed_lines.emplace_back(input_line);
  };

  for (const auto& input_line : input.lines_in_section(section))
  {
    process_line(std::string(input_line));
  }

  return {parsed_lines, unparsed_lines};
}
FOUR_C_NAMESPACE_CLOSE

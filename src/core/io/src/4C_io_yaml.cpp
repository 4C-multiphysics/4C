// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_io_yaml.hpp"

FOUR_C_NAMESPACE_OPEN

namespace
{
  [[noreturn]] void throw_on_yaml_parse_error(
      const char* msg, size_t msg_len, ryml::Location location, void* user_data)
  {
    throw Core::IO::YamlException(std::string(msg, msg_len));
  }
}  // namespace

Core::IO::YamlException::YamlException(const std::string& message) : Core::Exception(message) {}

ryml::Tree Core::IO::init_yaml_tree_with_exceptions()
{
  ryml::Callbacks cb{};
  cb.m_error = throw_on_yaml_parse_error;
  return ryml::Tree{cb};
}

void Core::IO::emit_value_as_yaml(YamlNodeRef node, const int& value) { node.node << value; }

void Core::IO::emit_value_as_yaml(YamlNodeRef node, const double& value) { node.node << value; }

void Core::IO::emit_value_as_yaml(YamlNodeRef node, const std::string& value)
{
  node.node |= ryml::VAL_DQUO;
  node.node << ryml::to_csubstr(value);
}

void Core::IO::emit_value_as_yaml(YamlNodeRef node, const std::string_view& value)
{
  node.node |= ryml::VAL_DQUO;
  node.node << ryml::csubstr(value.data(), value.size());
}

void Core::IO::emit_value_as_yaml(YamlNodeRef node, const bool& value)
{
  node.node << (value ? "true" : "false");
}

void Core::IO::emit_value_as_yaml(YamlNodeRef node, const std::filesystem::path& value)
{
  node.node |= ryml::VAL_DQUO;
  if (node.associated_file.empty() || value.is_absolute())
  {
    node.node << value.string();
  }
  // Make the relative path relative to the associated file.
  else
  {
    auto relative_path = std::filesystem::relative(value, node.associated_file.parent_path());
    node.node << relative_path.string();
  }
}

void Core::IO::read_value_from_yaml(Core::IO::ConstYamlNodeRef node, double& value)
{
  FOUR_C_ASSERT_ALWAYS(node.node.has_val(), "Expected a value node.");
  // Instead of relying on the ryml library to parse the double value, we do it ourselves to
  // ensure that we only accept data that fully parses as a double.

  if (node.node.is_val_quoted()) throw YamlException("Found a quoted value.");

  // Null-terminate the string.
  std::string string(node.node.val().data(), node.node.val().size());
  std::size_t end;
  try
  {
    value = std::stod(string.data(), &end);
  }
  catch (const std::logic_error&)
  {
    throw YamlException("Could not parse '" + string + "' as a double value.");
  }

  if (end != string.size())
  {
    throw YamlException("Could not parse '" + string + "' as a double value.");
  }
}

void Core::IO::read_value_from_yaml(Core::IO::ConstYamlNodeRef node, bool& value)
{
  FOUR_C_ASSERT_ALWAYS(node.node.has_val(), "Expected a value node.");

  if (node.node.is_val_quoted()) throw YamlException("Found a quoted value.");

  std::string token(node.node.val().data(), node.node.val().size());
  std::transform(token.begin(), token.end(), token.begin(), ::tolower);
  if (token == "true" || token == "yes" || token == "on" || token == "1")
    value = true;
  else if (token == "false" || token == "no" || token == "off" || token == "0")
    value = false;
  else
  {
    throw YamlException("Could not parse '" + token + "' as a boolean value.");
  }
}

void Core::IO::read_value_from_yaml(ConstYamlNodeRef node, std::filesystem::path& value)
{
  FOUR_C_ASSERT_ALWAYS(node.node.has_val(), "Expected a value node.");
  std::string path;
  read_value_from_yaml(node, path);

  value = std::filesystem::path(path);
  if (value.is_relative())
  {
    value = node.associated_file.parent_path() / value;
  }
}

FOUR_C_NAMESPACE_CLOSE

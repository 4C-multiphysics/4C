// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_io_yaml.hpp"

#include <format>

FOUR_C_NAMESPACE_OPEN

namespace
{
  [[noreturn]] void throw_on_yaml_parse_error(ryml::csubstr msg, const ryml::ErrorDataParse&, void*)
  {
    throw Core::IO::YamlException(std::string(msg.data(), msg.size()));
  }
}  // namespace

Core::IO::YamlException::YamlException(const std::string& message) : Core::Exception(message) {}

ryml::Tree Core::IO::init_yaml_tree_with_exceptions()
{
  ryml::Callbacks cb{};
  cb.set_error_parse(throw_on_yaml_parse_error);
  return ryml::Tree{cb};
}

void Core::IO::emit_value_as_yaml(YamlNodeRef node, const int& value)
{
  node.node.set_serialized(value);
}

void Core::IO::emit_value_as_yaml(YamlNodeRef node, const double& value)
{
  // ryml only formats doubles with std::to_chars if the feature-test macro __cpp_lib_to_chars is
  // defined. libc++ (macOS) does not define it since its std::from_chars for floating-point types
  // is incomplete, so ryml falls back to printf("%g"), which keeps only six significant digits.
  // std::format always writes the shortest representation that reads back exactly.
  node.node.set_serialized(std::format("{}", value));
}

void Core::IO::emit_value_as_yaml(YamlNodeRef node, const std::string& value)
{
  node.node.set_serialized(value, ryml::VAL_DQUO);
}

void Core::IO::emit_value_as_yaml(YamlNodeRef node, const std::string_view& value)
{
  node.node.set_serialized(value, ryml::VAL_DQUO);
}

void Core::IO::emit_value_as_yaml(YamlNodeRef node, const bool& value)
{
  node.node.set_serialized(value ? "true" : "false");
}

void Core::IO::emit_value_as_yaml(YamlNodeRef node, const std::filesystem::path& value)
{
  if (node.associated_file.empty() || value.is_absolute())
  {
    node.node.set_serialized(value.string(), ryml::VAL_DQUO);
  }
  // Make the relative path relative to the associated file.
  else
  {
    auto relative_path = std::filesystem::relative(value, node.associated_file.parent_path());
    node.node.set_serialized(relative_path.string(), ryml::VAL_DQUO);
  }
}

Core::IO::YamlReadStatus Core::IO::Internal::read_value_from_yaml(
    Core::IO::ConstYamlNodeRef node, double& value)
{
  if (!node.node.has_val()) return YamlReadStatus::wrong_type;
  // Instead of relying on the ryml library to parse the double value, we do it ourselves to
  // ensure that we only accept data that fully parses as a double.

  if (node.node.is_val_quoted()) return YamlReadStatus::wrong_type;

  // Null-terminate the string.
  std::string string(node.node.val().data(), node.node.val().size());
  std::size_t end;
  try
  {
    value = std::stod(string.data(), &end);
  }
  catch (const std::logic_error&)
  {
    return YamlReadStatus::wrong_type;
  }

  if (end != string.size()) return YamlReadStatus::wrong_type;
  return YamlReadStatus::success;
}

Core::IO::YamlReadStatus Core::IO::Internal::read_value_from_yaml(
    Core::IO::ConstYamlNodeRef node, bool& value)
{
  if (!node.node.has_val()) return YamlReadStatus::wrong_type;

  if (node.node.is_val_quoted()) return YamlReadStatus::wrong_type;

  std::string token(node.node.val().data(), node.node.val().size());
  std::transform(token.begin(), token.end(), token.begin(), ::tolower);
  if (token == "true" || token == "yes" || token == "on" || token == "1")
    value = true;
  else if (token == "false" || token == "no" || token == "off" || token == "0")
    value = false;
  else
    return YamlReadStatus::wrong_type;

  return YamlReadStatus::success;
}

Core::IO::YamlReadStatus Core::IO::Internal::read_value_from_yaml(
    ConstYamlNodeRef node, std::filesystem::path& value)
{
  if (!node.node.has_val()) return YamlReadStatus::wrong_type;
  std::string path;
  auto status = read_value_from_yaml(node, path);
  if (status != YamlReadStatus::success) return status;

  value = std::filesystem::path(path);
  if (value.is_relative())
  {
    value = node.associated_file.parent_path() / value;
  }
  return YamlReadStatus::success;
}

FOUR_C_NAMESPACE_CLOSE

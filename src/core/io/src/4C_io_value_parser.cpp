// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_io_value_parser.hpp"

#include <algorithm>

FOUR_C_NAMESPACE_OPEN

namespace
{
  void skip_whitespace(std::string_view& line, std::size_t& index)
  {
    while (index < line.size() && std::isspace(line[index])) ++index;
  }

  //! Helper to find the next token and update the given @p index into the line.
  std::string_view advance_token_impl(std::string_view line, std::size_t& index)
  {
    // Find the end of the token
    std::size_t start_of_token = index;
    while (index < line.size() && !std::isspace(line[index])) ++index;

    auto token = line.substr(start_of_token, index - start_of_token);

    skip_whitespace(line, index);

    return token;
  }
}  // namespace

void Core::IO::ValueParser::read_internal(bool& value)
{
  std::string token(advance_token());
  std::transform(token.begin(), token.end(), token.begin(), ::tolower);
  if (token == "true" || token == "yes" || token == "on" || token == "1")
    value = true;
  else if (token == "false" || token == "no" || token == "off" || token == "0")
    value = false;
  else
  {
    FOUR_C_THROW(
        "Could not parse '%s' as a boolean value.\nPossible values are (case insensitive): "
        "'true' (equivalent to 'yes', 'on', '1') or 'false' (equivalent to 'no', 'off', '0').",
        token.c_str());
  }
}

void Core::IO::ValueParser::read_internal(int& value)
{
  std::string token(advance_token());
  std::size_t end;
  try
  {
    value = std::stoi(token.data(), &end);
  }
  catch (const std::logic_error&)
  {
    FOUR_C_THROW("Could not parse '%s' as an integer value.", token.c_str());
  }

  if (end != token.size()) FOUR_C_THROW("Could not parse '%s' as an integer value.", token.c_str());
}

void Core::IO::ValueParser::read_internal(double& value)
{
  std::string token(advance_token());
  std::size_t end;
  try
  {
    value = std::stod(token.data(), &end);
  }
  catch (const std::logic_error&)
  {
    FOUR_C_THROW("Could not parse '%s' as a double value.", token.c_str());
  }

  if (end != token.size()) FOUR_C_THROW("Could not parse '%s' as a double value.", token.c_str());
}

void Core::IO::ValueParser::read_internal(std::string& value)
{
  value = std::string(advance_token());
}

void Core::IO::ValueParser::read_internal(std::filesystem::path& value)
{
  std::string token(advance_token());
  value = std::filesystem::path(token);
  if (!value.is_absolute()) value = context_.base_path / value;
}

Core::IO::ValueParser::ValueParser(std::string_view line, ValueParserContext context)
    : line_(line), context_(std::move(context))
{
  skip_whitespace(line_, current_index_);
}


void Core::IO::ValueParser::consume(const std::string& expected)
{
  std::string_view read_string = advance_token();
  if (read_string != std::string_view(expected))
    FOUR_C_THROW("%sCould not read expected string '%s'.", context_.user_scope_message.c_str(),
        expected.c_str());
}


void Core::IO::ValueParser::consume_comment(const std::string& comment_marker)
{
  std::string token(advance_token());
  if (token != comment_marker)
    FOUR_C_THROW("%sExpected comment marker '%s', but found '%s'.",
        context_.user_scope_message.c_str(), comment_marker.c_str(), token.c_str());

  // Consume the rest of the line
  current_index_ = line_.size();
}


std::string_view Core::IO::ValueParser::peek() const
{
  // Copy the current index to avoid modifying the parser state
  std::size_t temp_index = current_index_;
  return advance_token_impl(line_, temp_index);
}


bool Core::IO::ValueParser::at_end() const { return current_index_ == line_.size(); }


std::string_view Core::IO::ValueParser::get_unparsed_remainder() const
{
  return line_.substr(current_index_);
}


std::string_view Core::IO::ValueParser::advance_token()
{
  auto token = advance_token_impl(line_, current_index_);

  FOUR_C_ASSERT_ALWAYS(!token.empty(), "%sExpected more tokens, but reached the end of the line.",
      context_.user_scope_message.c_str());

  return token;
}

void Core::IO::ValueParser::backtrack()
{
  FOUR_C_ASSERT_ALWAYS(!backtrack_positions_.empty(), "No backtrack position to return to.");
  // Note: we do not pop the position here, as the BacktrackScope will take care of this.
  current_index_ = backtrack_positions_.top();
}


Core::IO::ValueParser::BacktrackScope::BacktrackScope(Core::IO::ValueParser& parser)
    : parser_(parser)
{
  parser_.backtrack_positions_.push(parser_.current_index_);
}

Core::IO::ValueParser::BacktrackScope::~BacktrackScope() { parser_.backtrack_positions_.pop(); }

FOUR_C_NAMESPACE_CLOSE

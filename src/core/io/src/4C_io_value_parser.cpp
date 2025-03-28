// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_io_value_parser.hpp"

#include <c4/charconv.hpp>

#include <algorithm>

FOUR_C_NAMESPACE_OPEN

namespace
{
  void skip_whitespace(std::string_view& line, std::size_t& index)
  {
    while (index < line.size() && std::isspace(line[index])) ++index;
  }

  //! Helper to find the next token and update the given @p index into the line.
  std::string_view advance_token_impl(std::string_view line, std::size_t& index, char delimiter)
  {
    if (index >= line.size()) return {};

    std::size_t start_of_token = index;
    if (delimiter != 0)
    {
      FOUR_C_ASSERT_ALWAYS(line[start_of_token] == delimiter,
          "Delimiter mismatch. Expected delimiter '{}' at position {}, but found '{}'.", delimiter,
          start_of_token, line[start_of_token]);
      ++index;
      while (index < line.size() && line[index] != delimiter) ++index;
      FOUR_C_ASSERT_ALWAYS(line[index] == delimiter,
          "Delimiter mismatch. Expected delimiter '{}' at position {}, but found '{}'.", delimiter,
          index, line[index]);

      // Skip the delimiter chars at start and end.
      auto token = line.substr(start_of_token + 1, index - start_of_token - 1);
      index++;

      skip_whitespace(line, index);

      return token;
    }
    else
    {
      while (index < line.size() && !std::isspace(line[index])) ++index;
      auto token = line.substr(start_of_token, index - start_of_token);

      skip_whitespace(line, index);

      return token;
    }
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
        "Could not parse '{}' as a boolean value.\nPossible values are (case insensitive): "
        "'true' (equivalent to 'yes', 'on', '1') or 'false' (equivalent to 'no', 'off', '0').",
        token.c_str());
  }
}

void Core::IO::ValueParser::read_internal(int& value)
{
  auto token(advance_token());
  if (token.front() == '+') [[unlikely]]
  {
    token = token.substr(1);
  }

  c4::csubstr token_cstr = c4::csubstr{token.data(), token.size()};
  // Note: this could use std::from_chars if clang fully supports it at some point.
  std::size_t n_chars_used = c4::from_chars_first(token_cstr, &value);
  if (n_chars_used != token.size())
  {
    FOUR_C_THROW("Could not parse '{}' as an integer value.", token);
  }
}

void Core::IO::ValueParser::read_internal(double& value)
{
  auto token(advance_token());
  if (token.front() == '+') [[unlikely]]
  {
    token = token.substr(1);
  }

  c4::csubstr token_cstr = c4::csubstr{token.data(), token.size()};
  // Note: this could use std::from_chars if clang fully supports it at some point.
  std::size_t n_chars_used = c4::from_chars_first(token_cstr, &value);
  if (n_chars_used != token.size())
  {
    FOUR_C_THROW("Could not parse '{}' as a double value.", token);
  }
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
    FOUR_C_THROW("{}Could not read expected string '{}'.", context_.user_scope_message.c_str(),
        expected.c_str());
}


void Core::IO::ValueParser::consume_comment(const std::string& comment_marker)
{
  std::string token(advance_token());
  if (token != comment_marker)
    FOUR_C_THROW("{}', but found '{}'.", context_.user_scope_message.c_str(),
        comment_marker.c_str(), token.c_str());

  // Consume the rest of the line
  current_index_ = line_.size();
}


std::string_view Core::IO::ValueParser::peek() const
{
  // Copy the current index to avoid modifying the parser state
  std::size_t temp_index = current_index_;
  return advance_token_impl(line_, temp_index, context_.token_delimiter);
}


bool Core::IO::ValueParser::at_end() const { return current_index_ == line_.size(); }


std::string_view Core::IO::ValueParser::get_unparsed_remainder() const
{
  return line_.substr(current_index_);
}


std::string_view Core::IO::ValueParser::advance_token()
{
  auto token = advance_token_impl(line_, current_index_, context_.token_delimiter);

  FOUR_C_ASSERT_ALWAYS(!token.empty(), "{}Expected more tokens, but reached the end of the line.",
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

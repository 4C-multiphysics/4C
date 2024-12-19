// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_IO_VALUE_PARSER_HPP
#define FOUR_C_IO_VALUE_PARSER_HPP

#include "4C_config.hpp"

#include "4C_utils_demangle.hpp"
#include "4C_utils_exceptions.hpp"

#include <array>
#include <filesystem>
#include <stack>
#include <string>
#include <string_view>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::IO
{
  struct ValueParserContext
  {
    /**
     * A message that is prepended to all error messages and allows to give additional context.
     */
    std::string user_scope_message{};

    /**
     * A path that is used as a base when parsing relative paths.
     */
    std::filesystem::path base_path{};
  };

  /**
   * A helper to parse values as defined in the .dat file format into C++ data. This
   * is a low-level class intended for use inside more user-friendly input mechanisms. Based on a
   * input string_view, it allows to read values of different types in sequence and validate that
   * the format matches. Note that the whole class works on string_views, so the original data must
   * outlive the parser.
   */
  class ValueParser
  {
   public:
    /**
     * A scope guard which creates a backtrack point in the parser. This allows to try parsing
     * complicated lines and backtrack if parsing fails. Note that the BacktrackScope alone
     * does not perform any backtracking. It only creates a point to which the parser can backtrack,
     * if the backtrack() function is called. When the scope is destroyed, the backtrack point is
     * removed.
     */
    class [[nodiscard]] BacktrackScope
    {
     public:
      explicit BacktrackScope(ValueParser& parser);
      ~BacktrackScope();

      // Delete copy and move operations.
      BacktrackScope(const BacktrackScope&) = delete;
      BacktrackScope& operator=(const BacktrackScope&) = delete;
      BacktrackScope(BacktrackScope&&) = delete;
      BacktrackScope& operator=(BacktrackScope&&) = delete;

     private:
      ValueParser& parser_;
    };

    /**
     * Set up the ValueParser and give an optional additional scope message. This information is
     * prepended to all error messages. Example:
     *
     * @code
     *   ValueParser parser(line, "While reading section MY PARAMETERS: ");
     * @endcode
     *
     * The input @p line is a string_view onto the data to parse. The parser will not copy the data,
     * so the original data must outlive the parser.
     */
    ValueParser(std::string_view line, ValueParserContext context = {});

    //! Read the next string and ensure it matches the expectation.
    void consume(const std::string& expected);

    /**
     * Consume the given @p comment_marker and anything that follows in the line.
     *
     * @post If no error occurred, then `at_end() == true`.
     */
    void consume_comment(const std::string& comment_marker = "//");

    /**
     * Read a value of type T from the input. For most types, this function does not take any
     * additional arguments. However, when reading into a dynamic container (e.g. std::vector), the
     * size of the container must be given as an additional argument. For example, to read two
     * nested vectors of integers, use:
     *
     * @code
     *  std::vector<std::vector<int>> nested_vector;
     *  parser.read(nested_vector, 2, 3);
     *  // This reads two vectors of size 3.
     * @endcode
     */
    template <typename T, typename... SizeInfo>
    T read(SizeInfo&&... size_info)
    {
      T val;
      // Dispatch to the correct parsing function.
      read_internal(val, std::forward<SizeInfo>(size_info)...);
      return val;
    }

    /**
     * Return the next token without consuming it. The return token may be empty if the parser has
     * reached the end of the input string.
     */
    [[nodiscard]] std::string_view peek() const;

    //! Check if this parser reached the end of the input string.
    [[nodiscard]] bool at_end() const;

    /**
     * Get anything that hasn't been parsed by previous calls to consume() or read(), not including
     * whitespace. Note that the returned string_view is a view into the original input string, so
     * it is only valid as long as the original string is valid. The returned string_view is empty
     * if the parser has reached the end of the input string.
     */
    [[nodiscard]] std::string_view get_unparsed_remainder() const;

    /**
     * Backtrack to the last position pushed by a BacktrackScope. This allows to try parsing rather
     * complicated lines. Before trying to parse a new value, create a BacktrackScope. If parsing
     * fails, call backtrack() to return to the position before the last parsing attempt. Note that
     * the BacktrackScope alone does not perform any backtracking! It only creates a point to which
     * the parser can backtrack when this function is called.
     */
    void backtrack();

   private:
    std::string_view advance_token();

    void read_internal(bool& value);
    void read_internal(int& value);
    void read_internal(double& value);
    void read_internal(std::string& value);
    void read_internal(std::filesystem::path& value);

    template <typename T, typename... SizeInfo>
    void read_internal(std::vector<T>& value, std::size_t size, SizeInfo&&... size_info)
    {
      value.resize(size);
      for (std::size_t i = 0; i < size; ++i)
      {
        read_internal(value[i], std::forward<SizeInfo>(size_info)...);
      }
    }

    template <typename T, std::size_t n, typename... SizeInfo>
    void read_internal(std::array<T, n>& value, SizeInfo&&... size_info)
    {
      for (std::size_t i = 0; i < n; ++i)
      {
        read_internal(value[i], std::forward<SizeInfo>(size_info)...);
      }
    }

    template <typename T, typename U, typename... SizeInfo>
    void read_internal(std::pair<T, U>& value, SizeInfo&&... size_info)
    {
      read_internal(value.first, std::forward<SizeInfo>(size_info)...);
      read_internal(value.second, std::forward<SizeInfo>(size_info)...);
    }

    //! The data to parse from.
    std::string_view line_;

    //! The current position in the line_.
    std::size_t current_index_{0};

    //! Additional context.
    ValueParserContext context_{};

    //! A stack of positions that the parser can backtrack to.
    std::stack<std::size_t> backtrack_positions_;
  };
}  // namespace Core::IO

FOUR_C_NAMESPACE_CLOSE

#endif

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_IO_VALUE_PARSER_HPP
#define FOUR_C_IO_VALUE_PARSER_HPP

#include "4C_config.hpp"

#include "4C_io_input_types.hpp"
#include "4C_utils_demangle.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_exceptions.hpp"

#include <array>
#include <filesystem>
#include <format>
#include <map>
#include <optional>
#include <stack>
#include <string>
#include <string_view>
#include <tuple>
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

    /**
     * An optional delimiter that is used to split tokens. If set, the parser will split tokens at
     * this delimiter. If not set, the parser will split tokens at whitespace. A useful choice
     * can be a quote character (") since it allows for tokens with spaces inside.
     */
    char token_delimiter = 0;
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
     *  parser.read(nested_vector, {2, 3});
     *  // This reads two vectors of size 3.
     * @endcode
     */
    template <typename T>
      requires(dynamic_rank<T>() == 0)
    T read()
    {
      T val;
      read_internal(val);
      return val;
    }

    template <typename T>
      requires(dynamic_rank<T>() == 1)
    T read(std::size_t size)
    {
      size_info_ = &size;
      T val;
      read_internal(val);
      size_info_ = nullptr;
      return val;
    }


    template <typename T>
      requires(dynamic_rank<T>() > 0)
    T read(const std::array<std::size_t, dynamic_rank<T>()>& size)
    {
      size_info_ = size.data();
      T val;
      read_internal(val);
      size_info_ = nullptr;
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

    template <typename Enum>
      requires(std::is_enum_v<Enum>)
    void read_internal(Enum& value)
    {
      std::string string;
      read_internal(string);
      auto val = EnumTools::enum_cast<Enum>(string);
      if (val)
      {
        value = *val;
      }
      else
      {
        FOUR_C_THROW("Could not parse value '{}' as an enum constant of type '{}'.", string,
            EnumTools::enum_type_name<Enum>());
      }
    }

    template <typename T>
    void read_internal(std::optional<T>& value)
    {
      auto next = peek();
      if (next == "none")
      {
        consume("none");
        value.reset();
      }
      else
      {
        read_internal(value.emplace());
      }
    }

    template <typename T>
    void read_internal(std::vector<T>& value)
    {
      const std::size_t my_size = size_info_[0];
      size_info_++;
      value.resize(my_size);
      for (std::size_t i = 0; i < my_size; ++i)
      {
        read_internal(value[i]);
      }
      size_info_--;
    }

    template <typename T, std::size_t n>
    void read_internal(std::array<T, n>& value)
    {
      for (std::size_t i = 0; i < n; ++i)
      {
        read_internal(value[i]);
      }
    }

    template <typename... Ts>
    void read_internal(std::tuple<Ts...>& value)
    {
      std::apply([this](auto&... val) { (this->read_internal(val), ...); }, value);
    }

    template <typename T1, typename T2>
    void read_internal(std::pair<T1, T2>& value)
    {
      read_internal(value.first);
      read_internal(value.second);
    }

    template <typename U>
    void read_internal(std::map<std::string, U>& value)
    {
      const std::size_t my_size = size_info_[0];
      size_info_++;
      for (std::size_t i = 0; i < my_size; ++i)
      {
        std::string key;
        read_internal(key);
        read_internal(value[key]);
      }
      size_info_--;
    }

    //! The data to parse from.
    std::string_view line_;

    //! The current position in the line_.
    std::size_t current_index_{0};

    //! Additional context.
    ValueParserContext context_{};

    //! A stack of positions that the parser can backtrack to.
    std::stack<std::size_t> backtrack_positions_;

    //! This is used to store the size information for the current read operation.
    const std::size_t* size_info_{nullptr};
  };
}  // namespace Core::IO

FOUR_C_NAMESPACE_CLOSE

#endif

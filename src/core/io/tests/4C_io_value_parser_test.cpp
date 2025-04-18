// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_io_value_parser.hpp"
namespace
{

  using namespace FourC;

  TEST(ValueParser, ConsumeSuccess)
  {
    std::string_view in("expected");
    Core::IO::ValueParser parser(
        in, {.user_scope_message = "While reading section MY PARAMETERS: "});
    parser.consume("expected");
  }

  TEST(ValueParser, ConsumeFail)
  {
    std::string_view in("unexpected");
    Core::IO::ValueParser parser(
        in, {.user_scope_message = "While reading section MY PARAMETERS: "});
    EXPECT_ANY_THROW(parser.consume("expected"));
  }

  TEST(ValueParser, ReadBoolTrueEquivalent)
  {
    std::string_view in("true on yes 1 True On Yes TRUE ON YES");
    Core::IO::ValueParser parser(in);

    for (int i = 0; i < 10; ++i) EXPECT_TRUE(parser.read<bool>());
    EXPECT_TRUE(parser.at_end());
  }

  TEST(ValueParser, ReadBoolFalseEquivalent)
  {
    std::string_view in("false off no 0 False Off No FALSE OFF NO");
    Core::IO::ValueParser parser(in);

    for (int i = 0; i < 10; ++i) EXPECT_FALSE(parser.read<bool>());
    EXPECT_TRUE(parser.at_end());
  }

  TEST(ValueParser, ReadDoubleSuccess)
  {
    std::string_view in("11.3");
    Core::IO::ValueParser parser(in);

    EXPECT_EQ(parser.read<double>(), 11.3);
  }

  TEST(ValueParser, ReadIntSuccess)
  {
    std::string_view in("42");
    Core::IO::ValueParser parser(in);

    EXPECT_EQ(parser.read<int>(), 42);
  }

  TEST(ValueParser, ReadDoubleFromIntSuccess)
  {
    std::string_view in("42");
    Core::IO::ValueParser parser(in);

    EXPECT_EQ(parser.read<double>(), 42);
  }

  TEST(ValueParser, ReadIntFromDoubleFail)
  {
    std::string_view in("11.3");
    Core::IO::ValueParser parser(in);

    EXPECT_ANY_THROW(parser.read<int>());
  }

  TEST(ValueParser, ReadFailWithExtraCharacters)
  {
    std::string_view in("11.3*2");
    Core::IO::ValueParser parser(in);

    EXPECT_ANY_THROW(parser.read<double>());
  }

  TEST(ValueParser, ReadExtraCharactersAfterWhitespaceCheckEOF)
  {
    std::string_view in("11.3 3");
    Core::IO::ValueParser parser(in);

    EXPECT_EQ(parser.read<double>(), 11.3);
    EXPECT_FALSE(parser.at_end());
  }

  TEST(ValueParser, ReadIntArraySuccess)
  {
    std::string_view in("1 2 3");
    Core::IO::ValueParser parser(in);

    const auto array = parser.read<std::array<int, 3>>();

    EXPECT_EQ(array[0], 1);
    EXPECT_EQ(array[1], 2);
    EXPECT_EQ(array[2], 3);
    EXPECT_TRUE(parser.at_end());
  }

  TEST(ValueParser, ReadIntArrayFailTooLong)
  {
    std::string_view in("1 2 3 4");
    Core::IO::ValueParser parser(in);

    parser.read<std::array<int, 3>>();
    EXPECT_FALSE(parser.at_end());
  }

  TEST(ValueParser, ReadIntArrayFailTooShort)
  {
    std::string_view in("1 2");
    Core::IO::ValueParser parser(in);

    EXPECT_ANY_THROW((parser.read<std::array<int, 3>>()));
  }

  TEST(ValueParser, ReadIntArrayFailOtherCharacters)
  {
    std::string_view in("1 2 a");
    Core::IO::ValueParser parser(in);

    EXPECT_ANY_THROW((parser.read<std::array<int, 3>>()));
  }

  TEST(ValueParser, ReadCombinedIntStringArraySuccess)
  {
    std::string_view in("1 2 3 a b c");
    Core::IO::ValueParser parser(in);

    const auto ints = parser.read<std::array<int, 3>>();
    const auto strings = parser.read<std::array<std::string, 3>>();

    EXPECT_EQ(ints[0], 1);
    EXPECT_EQ(ints[1], 2);
    EXPECT_EQ(ints[2], 3);
    EXPECT_EQ(strings[0], "a");
    EXPECT_EQ(strings[1], "b");
    EXPECT_EQ(strings[2], "c");
    EXPECT_TRUE(parser.at_end());
  }

  TEST(ValueParser, ReadDoubleVectorSuccess)
  {
    // read a vector of doubles with not specified size
    std::string_view in("0.1 0.2 0.3 0.4 0.5 0.6");
    Core::IO::ValueParser parser(in);

    std::vector<double> vec;

    while (!parser.at_end())
    {
      vec.push_back(parser.read<double>());
    }

    for (size_t i = 0; i < 6; ++i)
    {
      EXPECT_DOUBLE_EQ(vec[i], 0.1 * (i + 1));
    }
  }

  TEST(ValueParser, ReadPathNoContext)
  {
    std::string_view in("path/to/file");
    Core::IO::ValueParser parser(in);

    auto path = parser.read<std::filesystem::path>();

    EXPECT_EQ(path, std::filesystem::path("path/to/file"));
  }

  TEST(ValueParser, ReadPathWithContext)
  {
    std::string_view in("path/to/file");
    Core::IO::ValueParser parser(in, {.base_path = std::filesystem::path("base")});

    auto path = parser.read<std::filesystem::path>();

    EXPECT_EQ(path, std::filesystem::path("base/path/to/file"));
  }

  TEST(ValueParser, ReadPathAbsolute)
  {
    std::string_view in("/path/to/file");
    Core::IO::ValueParser parser(in, {.base_path = std::filesystem::path("base")});

    auto path = parser.read<std::filesystem::path>();

    EXPECT_EQ(path, std::filesystem::path("/path/to/file"));
  }

  TEST(ValueParser, ReadOptionalGiven)
  {
    std::string_view in("1 2 string");
    Core::IO::ValueParser parser(in);

    auto optional_int = parser.read<std::optional<int>>();
    auto optional_double = parser.read<std::optional<double>>();
    auto optional_string = parser.read<std::optional<std::string>>();

    EXPECT_EQ(optional_int, 1);
    EXPECT_EQ(optional_double, 2);
    EXPECT_EQ(optional_string, "string");
  }

  TEST(ValueParser, ReadOptionalNone)
  {
    std::string_view in("none none none 1 none");
    Core::IO::ValueParser parser(in);

    auto optional_int = parser.read<std::optional<int>>();
    auto optional_double = parser.read<std::optional<double>>();
    auto optional_string = parser.read<std::optional<std::string>>();
    auto optional_vector = parser.read<std::vector<std::optional<int>>>(2);

    EXPECT_FALSE(optional_int.has_value());
    EXPECT_FALSE(optional_double.has_value());
    EXPECT_FALSE(optional_string.has_value());
    EXPECT_EQ(optional_vector.size(), 2);
    EXPECT_TRUE(optional_vector[0].has_value());
    EXPECT_FALSE(optional_vector[1].has_value());
  }

  TEST(ValueParser, NestedVectors)
  {
    std::string_view in("1 2 3 4 5 6 7 8 9");
    Core::IO::ValueParser parser(in);

    auto nested_vectors = parser.read<std::vector<std::vector<int>>>({3, 3});

    EXPECT_EQ(nested_vectors.size(), 3);
    for (size_t i = 0; i < 3; ++i)
    {
      for (size_t j = 0; j < 3; ++j)
      {
        EXPECT_EQ(nested_vectors[i][j], 3 * i + j + 1);
      }
    }
  }

  TEST(ValueParser, Map)
  {
    std::string_view in("key1 1 2 key2 2 3 key3 3 4");
    Core::IO::ValueParser parser(in);

    auto map = parser.read<std::map<std::string, std::vector<int>>>({3, 2});

    EXPECT_EQ(map.size(), 3);
    EXPECT_EQ(map.at("key1"), (std::vector{1, 2}));
    EXPECT_EQ(map.at("key2"), (std::vector{2, 3}));
    EXPECT_EQ(map.at("key3"), (std::vector{3, 4}));
  }

  TEST(ValueParser, Unparsed)
  {
    std::string_view in("a 1 b 2 c 3");
    Core::IO::ValueParser parser(in);

    parser.consume("a");
    parser.read<int>();
    parser.consume("b");

    EXPECT_FALSE(parser.at_end());

    std::string unparsed = std::string(parser.get_unparsed_remainder());

    Core::IO::ValueParser parser2(unparsed);

    parser2.read<int>();
    parser2.consume("c");
    parser2.read<int>();

    EXPECT_TRUE(parser2.at_end());
  }

  TEST(ValueParser, Peek)
  {
    std::string_view in("a 1");
    Core::IO::ValueParser parser(in);

    EXPECT_EQ(parser.peek(), "a");
    parser.consume("a");
    EXPECT_EQ(parser.peek(), "1");

    // Peek should not consume the token.
    EXPECT_EQ(parser.get_unparsed_remainder(), "1");

    // Now consume it.
    EXPECT_EQ(parser.read<int>(), 1);
  }

  TEST(ValueParser, ParseEmpty)
  {
    std::string_view in("");
    Core::IO::ValueParser parser(in);

    EXPECT_EQ(parser.peek(), "");
    EXPECT_TRUE(parser.at_end());

    EXPECT_ANY_THROW(parser.consume("a"));
  }

  TEST(ValueParser, Backtrack)
  {
    std::string_view in("a 1 b");
    Core::IO::ValueParser parser(in);

    {
      Core::IO::ValueParser::BacktrackScope scope(parser);
      parser.consume("a");
      {
        Core::IO::ValueParser::BacktrackScope scope2(parser);
        parser.consume("1");
        parser.consume("b");
        parser.backtrack();

        parser.consume("1");
        parser.consume("b");
      }
      parser.backtrack();
    }

    EXPECT_ANY_THROW(parser.backtrack());

    parser.consume("a");
    parser.consume("1");
    parser.consume("b");

    EXPECT_TRUE(parser.at_end());
  }

  TEST(ValueParser, ReadQuoted)
  {
    std::string_view in(R"("string with spaces" "a" "1.0" "none 1.0 3" "consume this"
  )");  // trailing whitespace and newline are deliberate
    Core::IO::ValueParser parser(in, {.token_delimiter = '"'});

    EXPECT_EQ(parser.read<std::string>(), "string with spaces");
    EXPECT_EQ(parser.read<std::string>(), "a");
    EXPECT_EQ(parser.read<double>(), 1.0);
    EXPECT_EQ(parser.read<std::vector<std::optional<double>>>(3),
        (std::vector<std::optional<double>>{std::nullopt, 1.0, 3.0}));
    EXPECT_EQ(parser.peek(), "consume this");
    parser.consume("consume this");
    EXPECT_TRUE(parser.at_end());
    EXPECT_TRUE(parser.peek().empty());
  }


}  // namespace
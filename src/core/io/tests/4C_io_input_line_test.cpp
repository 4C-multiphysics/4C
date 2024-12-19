// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_io_input_line.hpp"

#include "4C_io_value_parser.hpp"
#include "4C_unittest_utils_assertions_test.hpp"


namespace
{
  using namespace FourC;
  using namespace FourC::Core::IO;
  using namespace FourC::Core::IO::InputLineBuilders;

  TEST(InputLineTest, Simple)
  {
    InputLine line{
        tag({.name = "marker"}),
        entry<int>({.name = "a"}),
        entry<double>({.name = "b"}),
        entry<std::string>({.name = "c"}),
        entry<bool>({.name = "d"}),
    };
    InputParameterContainer container;
    std::string stream("marker a 1 b 2.0 c string d true // trailing comment");
    line.parse(stream, container);
    EXPECT_EQ(container.get<int>("a"), 1);
    EXPECT_EQ(container.get<double>("b"), 2.0);
    EXPECT_EQ(container.get<std::string>("c"), "string");
    EXPECT_EQ(container.get<bool>("d"), true);
  }

  TEST(InputLineTest, OutofOrder)
  {
    InputLine line{
        entry<int>({.name = "a"}),
        entry<double>({.name = "b"}),
        entry<std::string>({.name = "c"}),
    };
    InputParameterContainer container;
    std::string stream("b 2.0 c string a 1");
    line.parse(stream, container);
    EXPECT_EQ(container.get<int>("a"), 1);
    EXPECT_EQ(container.get<double>("b"), 2.0);
    EXPECT_EQ(container.get<std::string>("c"), "string");
  }

  TEST(InputLineTest, OptionalLeftOut)
  {
    InputLine line{
        entry<int>({.name = "a"}),
        entry<double>({.name = "b"}),
        entry<std::string>({.name = "c", .default_value = "default"}),
    };

    {
      InputParameterContainer container;
      std::string stream("a 1 b 2.0 // c 1");
      line.parse(stream, container);
      EXPECT_EQ(container.get<int>("a"), 1);
      EXPECT_EQ(container.get<double>("b"), 2.0);
      EXPECT_EQ(container.get<std::string>("c"), "default");
    }
  }

  TEST(InputLineTest, RequiredLeftOut)
  {
    InputLine line{
        entry<int>({.name = "a"}),
        entry<double>({.name = "b"}),
        entry<std::string>({.name = "c"}),
    };

    {
      InputParameterContainer container;
      std::string stream("a 1 b 2.0");
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(line.parse(stream, container), Core::Exception,
          "Required value 'c' not found in input line");
    }
  }

  TEST(InputLineTest, Vector)
  {
    InputLine line{
        entry<int>({.name = "a"}),
        entry<std::vector<double>>({.name = "b", .size = 3}),
        entry<std::string>({.name = "c"}),
    };

    {
      InputParameterContainer container;
      std::string stream("a 1 b 1.0 2.0 3.0 c string");
      line.parse(stream, container);
      EXPECT_EQ(container.get<int>("a"), 1);
      const auto& b = container.get<std::vector<double>>("b");
      EXPECT_EQ(b.size(), 3);
      EXPECT_EQ(b[0], 1.0);
      EXPECT_EQ(b[1], 2.0);
      EXPECT_EQ(b[2], 3.0);
      EXPECT_EQ(container.get<std::string>("c"), "string");
    }

    {
      InputParameterContainer container;
      std::string stream("a 1 b 1.0 2.0 c string");
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(
          line.parse(stream, container), Core::Exception, "Could not parse 'c' as a double value");
    }
  }

  TEST(InputLineTest, VectorWithParsedLength)
  {
    InputLine line{
        entry<int>({.name = "a"}),
        entry<std::vector<double>>({.name = "b", .size = from_parameter<int>("a")}),
        entry<std::string>({.name = "c"}),
    };

    {
      InputParameterContainer container;
      std::string stream("a 3 b 1.0 2.0 3.0 c string");
      line.parse(stream, container);
      EXPECT_EQ(container.get<int>("a"), 3);
      const auto& b = container.get<std::vector<double>>("b");
      EXPECT_EQ(b.size(), 3);
      EXPECT_EQ(b[0], 1.0);
      EXPECT_EQ(b[1], 2.0);
      EXPECT_EQ(b[2], 3.0);
      EXPECT_EQ(container.get<std::string>("c"), "string");
    }

    {
      InputParameterContainer container;
      std::string stream("a 3 b 1.0 2.0 c string");
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(
          line.parse(stream, container), Core::Exception, "Could not parse 'c' as a double value");
    }
  }

  TEST(InputLineTest, UserDefinedComponent)
  {
    InputLine line{
        entry<int>({.name = "a"}),
        entry<double>({.name = "b"}),
        user_defined<std::string>(
            {.name = "c", .description = "A string", .default_value = "Not found"},
            [name = "c"](ValueParser& parser, InputParameterContainer& container)
            {
              parser.consume(name);
              // Some special parsing logic
              parser.consume("_");
              parser.consume("_");

              container.add<std::string>("c", "I found c");
            },
            [](std::ostream& out, const Core::IO::InputParameterContainer&) { out << "_ _"; }),
        entry<std::string>({.name = "s"}),
    };

    {
      InputParameterContainer container;
      std::string stream("a 1 b 2.0 c _ _ s hello");
      line.parse(stream, container);
      EXPECT_EQ(container.get<int>("a"), 1);
      EXPECT_EQ(container.get<double>("b"), 2.0);
      EXPECT_EQ(container.get<std::string>("c"), "I found c");
      EXPECT_EQ(container.get<std::string>("s"), "hello");
    }

    {
      InputParameterContainer container;
      std::string stream("a 1 b 2.0 c _ s hello");
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(
          line.parse(stream, container), Core::Exception, "expected string '_'");
    }
  }

  TEST(InputLineTest, Selection)
  {
    InputLine line{
        entry<int>({.name = "a"}),
        selection<int>({.name = "b", .default_value = 1, .choices = {{"b1", 1}, {"b2", 2}}}),
        selection<std::string>(
            {.name = "c", .default_value = "3", .choices = {{"c1", "1"}, {"c2", "2"}}}),
        entry<std::string>({.name = "d"}),
    };

    {
      InputParameterContainer container;
      std::string stream("a 1 b b2 d string c c1");
      line.parse(stream, container);
      EXPECT_EQ(container.get<int>("a"), 1);
      EXPECT_EQ(container.get<int>("b"), 2);
      EXPECT_EQ(container.get<std::string>("c"), "1");
      EXPECT_EQ(container.get<std::string>("d"), "string");
    }

    {
      InputParameterContainer container;
      std::string stream("a 1 b b4 c string");
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(
          line.parse(stream, container), Core::Exception, "Invalid value 'b4'");
    }
  }

  TEST(InputLineTest, Unparsed)
  {
    InputLine line{
        entry<int>({.name = "a"}),
        entry<int>({.name = "optional", .default_value = 42}),
        entry<double>({.name = "b"}),
        entry<std::string>({.name = "c"}),
    };
    InputParameterContainer container;
    std::string stream("a 1 b 2.0 c string unparsed unparsed");
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        line.parse(stream, container), Core::Exception, "Unexpected value 'unparsed'");
  }


  TEST(InputLineTest, Nested)
  {
    InputLine line{
        entry<int>({.name = "a"}),
        group({.name = "nested",
            .required = false,
            .entries =
                {
                    entry<double>({.name = "b", .default_value = 3.0}),
                }}),
        entry<std::string>({.name = "c"}),
    };

    {
      InputParameterContainer container;
      std::string stream("a 1 nested b 2.0 c string");
      line.parse(stream, container);
      EXPECT_EQ(container.get<int>("a"), 1);
      EXPECT_EQ(container.group("nested").get<double>("b"), 2.0);
      EXPECT_EQ(container.get<std::string>("c"), "string");
    }

    {
      InputParameterContainer container;
      std::string stream("a 1 nested c string");
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(
          line.parse(stream, container), Core::Exception, "Unexpected value 'c'");
    }

    {
      InputParameterContainer container;
      std::string stream("a 1 c string");
      line.parse(stream, container);
      EXPECT_EQ(container.get<int>("a"), 1);
      EXPECT_EQ(container.group("nested").get<double>("b"), 3.0);
      EXPECT_EQ(container.get<std::string>("c"), "string");
    }
  }

  TEST(InputLineTest, OneOf)
  {
    InputLine line{
        entry<int>({.name = "a", .default_value = 42}),
        one_of({
            one_of({
                entry<int>({.name = "x"}),
                entry<double>({.name = "y"}),
            }),
            entry<double>({.name = "b"}),
            group({.name = "nested",
                .entries =
                    {
                        entry<std::string>({.name = "c"}),
                        entry<double>({.name = "d"}),
                    }}),
        }),
    };

    {
      InputParameterContainer container;
      std::string stream("a 1 x 2");
      line.parse(stream, container);
      EXPECT_EQ(container.get<int>("a"), 1);
      EXPECT_EQ(container.get<int>("x"), 2);
    }
  }
}  // namespace

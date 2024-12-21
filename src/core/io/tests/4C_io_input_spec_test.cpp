// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_io_input_spec_builders.hpp"
#include "4C_io_value_parser.hpp"
#include "4C_unittest_utils_assertions_test.hpp"


namespace
{
  using namespace FourC;
  using namespace FourC::Core::IO;
  using namespace FourC::Core::IO::InputSpecBuilders;

  TEST(InputSpecTest, Simple)
  {
    auto line = group({
        tag("marker"),
        entry<int>("a", {.description = "An integer", .default_value = 1}),
        entry<double>("b", {.required = true}),
        entry<std::string>("c", {.required = false}),
        entry<bool>("d"),
    });
    InputParameterContainer container;
    std::string stream("marker b 2.0 d true // trailing comment");
    ValueParser parser(stream);
    fully_parse(parser, line, container);
    EXPECT_EQ(container.get<int>("a"), 1);
    EXPECT_EQ(container.get<double>("b"), 2.0);
    EXPECT_EQ(container.get_if<std::string>("c"), nullptr);
    EXPECT_EQ(container.get<bool>("d"), true);
  }

  TEST(InputSpecTest, OutofOrder)
  {
    auto line = group({
        entry<int>("a"),
        entry<double>("b"),
        entry<std::string>("c"),
    });
    InputParameterContainer container;
    std::string stream("b 2.0 c string a 1");
    ValueParser parser(stream);
    fully_parse(parser, line, container);
    EXPECT_EQ(container.get<int>("a"), 1);
    EXPECT_EQ(container.get<double>("b"), 2.0);
    EXPECT_EQ(container.get<std::string>("c"), "string");
  }

  TEST(InputSpecTest, OptionalLeftOut)
  {
    auto line = group({
        entry<int>("a"),
        entry<double>("b"),
        entry<std::string>("c", {.default_value = "default"}),
    });

    {
      InputParameterContainer container;
      std::string stream("a 1 b 2.0 // c 1");
      ValueParser parser(stream);
      fully_parse(parser, line, container);
      EXPECT_EQ(container.get<int>("a"), 1);
      EXPECT_EQ(container.get<double>("b"), 2.0);
      EXPECT_EQ(container.get<std::string>("c"), "default");
    }
  }

  TEST(InputSpecTest, RequiredLeftOut)
  {
    auto line = group({
        entry<int>("a"),
        entry<double>("b"),
        entry<std::string>("c"),
    });

    {
      InputParameterContainer container;
      std::string stream("a 1 b 2.0");
      ValueParser parser(stream);
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(fully_parse(parser, line, container), Core::Exception,
          "Required value 'c' not found in input line");
    }
  }

  TEST(InputSpecTest, Vector)
  {
    auto line = group({
        entry<int>("a"),
        entry<std::vector<double>>("b", {.size = 3}),
        entry<std::string>("c"),
    });

    {
      InputParameterContainer container;
      std::string stream("a 1 b 1.0 2.0 3.0 c string");
      ValueParser parser(stream);
      fully_parse(parser, line, container);
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
      ValueParser parser(stream);
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(fully_parse(parser, line, container), Core::Exception,
          "Could not parse 'c' as a double value");
    }
  }

  TEST(InputSpecTest, VectorWithParsedLength)
  {
    auto line = group({
        entry<int>("a"),
        entry<std::vector<double>>("b", {.size = from_parameter<int>("a")}),
        entry<std::string>("c"),
    });

    {
      InputParameterContainer container;
      std::string stream("a 3 b 1.0 2.0 3.0 c string");
      ValueParser parser(stream);
      fully_parse(parser, line, container);
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
      ValueParser parser(stream);
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(fully_parse(parser, line, container), Core::Exception,
          "Could not parse 'c' as a double value");
    }
  }

  TEST(InputSpecTest, UserDefined)
  {
    auto line = group({
        entry<int>("a"),
        entry<double>("b"),
        user_defined<std::string>(
            "c", {.description = "A string", .default_value = "Not found"},
            [name = "c"](ValueParser& parser, InputParameterContainer& container)
            {
              parser.consume(name);
              // Some special parsing logic
              parser.consume("_");
              parser.consume("_");

              container.add<std::string>("c", "I found c");
            },
            [](std::ostream& out, const Core::IO::InputParameterContainer&) { out << "_ _"; }),
        entry<std::string>("s"),
    });

    {
      InputParameterContainer container;
      std::string stream("a 1 b 2.0 c _ _ s hello");
      ValueParser parser(stream);
      fully_parse(parser, line, container);
      EXPECT_EQ(container.get<int>("a"), 1);
      EXPECT_EQ(container.get<double>("b"), 2.0);
      EXPECT_EQ(container.get<std::string>("c"), "I found c");
      EXPECT_EQ(container.get<std::string>("s"), "hello");
    }

    {
      InputParameterContainer container;
      std::string stream("a 1 b 2.0 c _ s hello");
      ValueParser parser(stream);
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(
          fully_parse(parser, line, container), Core::Exception, "expected string '_'");
    }
  }

  TEST(InputSpecTest, Selection)
  {
    auto line = group({
        entry<int>("a"),
        selection<int>("b", {.default_value = 1, .choices = {{"b1", 1}, {"b2", 2}}}),
        selection<std::string>("c", {.default_value = "2", .choices = {{"c1", "1"}, {"c2", "2"}}}),
        entry<std::string>("d"),
    });

    {
      InputParameterContainer container;
      std::string stream("a 1 b b2 d string c c1");
      ValueParser parser(stream);
      fully_parse(parser, line, container);
      EXPECT_EQ(container.get<int>("a"), 1);
      EXPECT_EQ(container.get<int>("b"), 2);
      EXPECT_EQ(container.get<std::string>("c"), "1");
      EXPECT_EQ(container.get<std::string>("d"), "string");
    }

    {
      InputParameterContainer container;
      std::string stream("a 1 b b4 c string");
      ValueParser parser(stream);
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(
          fully_parse(parser, line, container), Core::Exception, "Invalid value 'b4'");
    }
  }

  TEST(InputSpecTest, Unparsed)
  {
    auto line = group({
        entry<int>("a"),
        entry<int>("optional", {.default_value = 42}),
        entry<double>("b"),
        entry<std::string>("c"),
    });
    InputParameterContainer container;
    std::string stream("a 1 b 2.0 c string unparsed unparsed");
    ValueParser parser(stream);
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(fully_parse(parser, line, container), Core::Exception,
        "line still contains 'unparsed unparsed'");
  }


  TEST(InputSpecTest, Groups)
  {
    auto line = group({
        entry<int>("a"),
        group("group1",
            {
                .entries =
                    {
                        entry<double>("b"),
                    },
            }),
        group("group2",
            {
                .required = false,
                .entries =
                    {
                        entry<double>("b", {.default_value = 3.0}),
                        entry<std::string>("c"),
                    },
            }),
        group("group3",
            {
                .required = false,
                .entries =
                    {
                        entry<std::string>("c", {.default_value = "default"}),
                    },
            }),
        entry<std::string>("c"),
    });

    {
      InputParameterContainer container;
      std::string stream("a 1 group1 b 2.0 c string");
      ValueParser parser(stream);
      fully_parse(parser, line, container);
      const auto& const_container = container;
      EXPECT_EQ(const_container.get<int>("a"), 1);
      EXPECT_EQ(const_container.group("group1").get<double>("b"), 2.0);
      EXPECT_ANY_THROW([[maybe_unused]] const auto& c = const_container.group("group2"));
      // Group 3 only contains entries that have default values, so it implicitly has a default
      // value.
      EXPECT_EQ(const_container.group("group3").get<std::string>("c"), "default");
    }

    {
      InputParameterContainer container;
      std::string stream("a 1 group2 b 2.0 c string group1 b 4.0 c string");
      ValueParser parser(stream);
      fully_parse(parser, line, container);
      const auto& const_container = container;
      EXPECT_EQ(const_container.get<int>("a"), 1);
      EXPECT_EQ(const_container.group("group2").get<double>("b"), 2.0);
      EXPECT_EQ(const_container.group("group1").get<double>("b"), 4.0);
      EXPECT_EQ(const_container.get<std::string>("c"), "string");
    }

    {
      InputParameterContainer container;
      std::string stream("a 1 group1 b 4.0 c string");
      ValueParser parser(stream);
      fully_parse(parser, line, container);
      const auto& const_container = container;
      EXPECT_EQ(const_container.get<int>("a"), 1);
      EXPECT_ANY_THROW([[maybe_unused]] const auto& c = const_container.group("group2"));
      EXPECT_EQ(const_container.group("group1").get<double>("b"), 4.0);
      EXPECT_EQ(const_container.get<std::string>("c"), "string");
    }
  }

  TEST(InputSpecTest, OneOf)
  {
    auto line = group({
        entry<int>("a", {.default_value = 42}),
        one_of({
            entry<double>("b"),
            group("group", {.entries =
                                   {
                                       entry<std::string>("c"),
                                       entry<double>("d"),
                                   }}),
        }),
    });

    {
      InputParameterContainer container;
      std::string stream("a 1 b 2");
      ValueParser parser(stream);
      fully_parse(parser, line, container);
      EXPECT_EQ(container.get<int>("a"), 1);
      EXPECT_EQ(container.get<double>("b"), 2);
    }

    {
      InputParameterContainer container;
      std::string stream("a 1 group c string d 2.0");
      ValueParser parser(stream);
      fully_parse(parser, line, container);
      EXPECT_EQ(container.get<int>("a"), 1);
      EXPECT_EQ(container.group("group").get<std::string>("c"), "string");
      EXPECT_EQ(container.group("group").get<double>("d"), 2);
    }

    {
      InputParameterContainer container;
      std::string stream("a 1 group c string d 2.0 b 3.0");
      ValueParser parser(stream);
      // More than one of the one_of entries is present. Refuse to parse any of them.
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(fully_parse(parser, line, container), Core::Exception,
          "still contains 'group c string d 2.0 b 3.0'");
    }

    {
      InputParameterContainer container;
      std::string stream("a 1 group c string");
      // Note: we start to parse the group, but the entries are not complete, so we backtrack.
      // The result is that the parts of the group remain unparsed.
      ValueParser parser(stream);
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(
          fully_parse(parser, line, container), Core::Exception, "still contains 'group c string'");
    }

    {
      InputParameterContainer container;
      std::string stream("a 1");
      ValueParser parser(stream);
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(
          fully_parse(parser, line, container), Core::Exception, "Required 'one_of {b, group}'");
    }
  }

  TEST(InputSpecTest, OneOfTopLevel)
  {
    auto line = one_of({
        group({
            entry<int>("a"),
            entry<double>("b"),
        }),
        group({
            entry<std::string>("c"),
            entry<double>("d"),
        }),
    });

    {
      InputParameterContainer container;
      std::string stream("a 1 b 2");
      ValueParser parser(stream);
      fully_parse(parser, line, container);
      EXPECT_EQ(container.get<int>("a"), 1);
      EXPECT_EQ(container.get<double>("b"), 2);
    }

    {
      InputParameterContainer container;
      std::string stream("c string d 2.0");
      ValueParser parser(stream);
      fully_parse(parser, line, container);
      EXPECT_EQ(container.get<std::string>("c"), "string");
      EXPECT_EQ(container.get<double>("d"), 2);
    }

    {
      InputParameterContainer container;
      std::string stream("a 1 b 2 c string d 2.0");
      ValueParser parser(stream);
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(fully_parse(parser, line, container), Core::Exception,
          "both 'group {a, b}' and 'group {c, d}'");
    }

    {
      InputParameterContainer container;
      std::string stream("a 1 c string");
      ValueParser parser(stream);
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(fully_parse(parser, line, container), Core::Exception,
          "one_of {group {a, b}, group {c, d}}");
    }
  }

  TEST(InputSpecTest, Print)
  {
    auto line = group({
        entry<int>("a"),
        entry<std::string>("b"),
        selection<int>("c", {.default_value = 1, .choices = {{"c1", 1}, {"c2", 2}}}),
    });

    {
      InputParameterContainer container;
      container.add("a", 1);
      container.add("b", std::string("s"));
      std::ostringstream out;
      line.print(out, container);
      EXPECT_EQ(out.str(), "a 1 b s [c (c1|c2|)] ");
    }
  }
}  // namespace

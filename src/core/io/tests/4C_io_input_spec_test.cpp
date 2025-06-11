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
    auto spec = all_of({
        parameter<int>("a", {.description = "An integer", .default_value = 1}),
        parameter<double>("b"),
        parameter<bool>("d"),
    });
    InputParameterContainer container;
    std::string stream("b 2.0 d true // trailing comment");
    ValueParser parser(stream);
    spec.fully_parse(parser, container);
    EXPECT_EQ(container.get<int>("a"), 1);
    EXPECT_EQ(container.get<double>("b"), 2.0);
    EXPECT_EQ(container.get<bool>("d"), true);
  }

  TEST(InputSpecTest, OptionalLeftOut)
  {
    auto spec = all_of({
        parameter<int>("a"),
        parameter<double>("b"),
        parameter<std::string>("c", {.default_value = "default"}),
    });

    {
      InputParameterContainer container;
      std::string stream("a 1 b 2.0 // c 1");
      ValueParser parser(stream);
      spec.fully_parse(parser, container);
      EXPECT_EQ(container.get<int>("a"), 1);
      EXPECT_EQ(container.get<double>("b"), 2.0);
      EXPECT_EQ(container.get<std::string>("c"), "default");
    }
  }

  TEST(InputSpecTest, RequiredLeftOut)
  {
    auto spec = all_of({
        parameter<int>("a"),
        parameter<double>("b"),
        parameter<std::string>("c"),
    });

    {
      InputParameterContainer container;
      std::string stream("a 1 b 2.0");
      ValueParser parser(stream);
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(spec.fully_parse(parser, container), Core::Exception,
          "Required value 'c' not found in input line");
    }
  }

  TEST(InputSpecTest, EnumClassSelection)
  {
    enum class EnumClass
    {
      A,
      B,
      C,
    };

    auto spec = all_of({
        deprecated_selection<EnumClass>("enum",
            {
                {"A", EnumClass::A},
                {"B", EnumClass::B},
                {"C", EnumClass::C},
            }),
    });

    {
      InputParameterContainer container;
      std::string stream("enum A");
      ValueParser parser(stream);
      spec.fully_parse(parser, container);
      EXPECT_EQ(container.get<EnumClass>("enum"), EnumClass::A);
    }
  }

  TEST(InputSpecTest, MagicEnumParameter)
  {
    enum class EnumClass
    {
      A,
      B,
      C,
    };

    auto spec = parameter<EnumClass>("enum");

    {
      SCOPED_TRACE("Valid enum constant");
      InputParameterContainer container;
      std::string stream("enum A");
      ValueParser parser(stream);
      spec.fully_parse(parser, container);
      EXPECT_EQ(container.get<EnumClass>("enum"), EnumClass::A);
    }

    {
      SCOPED_TRACE("Invalid enum constant");
      InputParameterContainer container;
      std::string stream("enum XYZ");
      ValueParser parser(stream);
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(spec.fully_parse(parser, container), Core::Exception,
          "Could not parse value 'XYZ' as an enum constant of type 'EnumClass'");
    }

    {
      std::ostringstream out;
      ryml::Tree tree = init_yaml_tree_with_exceptions();
      ryml::NodeRef root = tree.rootref();
      YamlNodeRef yaml(root, "");
      spec.emit_metadata(yaml);
      out << tree;

      std::string expected = R"(name: enum
type: enum
choices:
  - name: A
  - name: B
  - name: C
required: true
)";
      EXPECT_EQ(out.str(), expected);
    }
  }

  TEST(InputSpecTest, ParseSingleDefaultedEntryDat)
  {
    // This used to be a bug where a single default dat parameter was not accepted.
    auto spec = all_of({
        parameter<double>("a", {.default_value = 1.0}),
    });

    {
      InputParameterContainer container;
      std::string stream("");
      ValueParser parser(stream);
      spec.fully_parse(parser, container);
      EXPECT_EQ(container.get<double>("a"), 1.0);
    }
  }

  TEST(InputSpecTest, Vector)
  {
    auto spec = all_of({
        parameter<std::vector<std::vector<int>>>("a", {.size = {2, 2}}),
        parameter<std::vector<double>>("b", {.size = 3}),
    });

    {
      InputParameterContainer container;
      std::string stream("a 1 2 3 4 b 1.0 2.0 3.0");
      ValueParser parser(stream);
      spec.fully_parse(parser, container);
      const auto& a = container.get<std::vector<std::vector<int>>>("a");
      EXPECT_EQ(a.size(), 2);
      EXPECT_EQ(a[0].size(), 2);
      EXPECT_EQ(a[0][0], 1);
      EXPECT_EQ(a[0][1], 2);
      EXPECT_EQ(a[1].size(), 2);
      EXPECT_EQ(a[1][0], 3);
      EXPECT_EQ(a[1][1], 4);
      const auto& b = container.get<std::vector<double>>("b");
      EXPECT_EQ(b.size(), 3);
      EXPECT_EQ(b[0], 1.0);
      EXPECT_EQ(b[1], 2.0);
      EXPECT_EQ(b[2], 3.0);
    }

    {
      InputParameterContainer container;
      std::string stream("a 1 2 3 4 b 1.0 2.0 c");
      ValueParser parser(stream);
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(spec.fully_parse(parser, container), Core::Exception,
          "Could not parse 'c' as a double value");
    }
  }

  TEST(InputSpecTest, Optional)
  {
    auto spec = all_of({
        parameter<int>("size"),
        parameter<std::vector<std::optional<int>>>("vector_none",
            {
                .default_value = std::vector<std::optional<int>>{std::nullopt, 1},
                .size = from_parameter<int>("size"),
            }),
        parameter<std::optional<std::vector<int>>>(
            "none_vector", {.size = from_parameter<int>("size")}),
        parameter<std::optional<std::string>>("b", {.description = "b"}),
        parameter<std::optional<int>>("e"),
    });

    {
      SCOPED_TRACE("All values");
      InputParameterContainer container;
      std::string stream("size 3 vector_none 1 2 3 b none none_vector 1 2 3");
      ValueParser parser(stream);
      spec.fully_parse(parser, container);
      const auto& vector_none = container.get<std::vector<std::optional<int>>>("vector_none");
      EXPECT_EQ(vector_none.size(), 3);
      EXPECT_EQ(vector_none[0].has_value(), true);
      EXPECT_EQ(vector_none[0].value(), 1);
      EXPECT_EQ(vector_none[1].has_value(), true);
      EXPECT_EQ(vector_none[1].value(), 2);
      EXPECT_EQ(vector_none[2].has_value(), true);
      EXPECT_EQ(vector_none[2].value(), 3);

      EXPECT_TRUE(container.get<std::optional<std::vector<int>>>("none_vector").has_value());

      const auto& b = container.get<std::optional<std::string>>("b");
      EXPECT_EQ(b.has_value(), false);

      const auto& e = container.get<std::optional<int>>("e");
      EXPECT_EQ(e.has_value(), false);
    }

    {
      SCOPED_TRACE("None values");
      InputParameterContainer container;
      std::string stream("size 3 vector_none 1 none 3 b none e none none_vector none");
      ValueParser parser(stream);
      spec.fully_parse(parser, container);
      const auto& a = container.get<std::vector<std::optional<int>>>("vector_none");
      EXPECT_EQ(a.size(), 3);
      EXPECT_EQ(a[0].has_value(), true);
      EXPECT_EQ(a[0].value(), 1);
      EXPECT_EQ(a[1].has_value(), false);
      EXPECT_EQ(a[2].has_value(), true);
      EXPECT_EQ(a[2].value(), 3);

      const auto& b = container.get<std::optional<std::string>>("b");
      EXPECT_EQ(b.has_value(), false);

      const auto& e = container.get<std::optional<int>>("e");
      EXPECT_EQ(e.has_value(), false);
    }

    {
      SCOPED_TRACE("Defaults");
      InputParameterContainer container;
      std::string stream("size 3 b string e 42");
      ValueParser parser(stream);
      spec.fully_parse(parser, container);

      const auto& vector_none = container.get<std::vector<std::optional<int>>>("vector_none");
      EXPECT_EQ(vector_none.size(), 2);
      EXPECT_EQ(vector_none[0].has_value(), false);
      EXPECT_EQ(vector_none[1].has_value(), true);

      const auto& b = container.get<std::optional<std::string>>("b");
      EXPECT_EQ(b.has_value(), true);
      EXPECT_EQ(b.value(), "string");

      const auto& e = container.get<std::optional<int>>("e");
      EXPECT_EQ(e.has_value(), true);
      EXPECT_EQ(e.value(), 42);
    }
  }

  TEST(InputSpecTest, VectorWithParsedLength)
  {
    auto spec = all_of({
        parameter<int>("a"),
        parameter<std::vector<double>>("b", {.size = from_parameter<int>("a")}),
        parameter<std::string>("c"),
    });

    {
      InputParameterContainer container;
      std::string stream("a 3 b 1.0 2.0 3.0 c string");
      ValueParser parser(stream);
      spec.fully_parse(parser, container);
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
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(spec.fully_parse(parser, container), Core::Exception,
          "Could not parse 'c' as a double value");
    }
  }

  TEST(InputSpecTest, EntryWithCallback)
  {
    auto spec = all_of({
        parameter<int>("a"),
        parameter<double>("b"),
        parameter<std::string>("c",
            {
                .description = "A string",
                .default_value = "Not found",
                .on_parse_callback = [](InputParameterContainer& container)
                { container.add<int>("c_as_int", std::stoi(container.get<std::string>("c"))); },
            }),
        parameter<std::string>("s"),
    });

    {
      InputParameterContainer container;
      std::string stream("a 1 b 2.0 c 10 s hello");
      ValueParser parser(stream);
      spec.fully_parse(parser, container);
      EXPECT_EQ(container.get<int>("a"), 1);
      EXPECT_EQ(container.get<double>("b"), 2.0);
      EXPECT_EQ(container.get<std::string>("c"), "10");
      EXPECT_EQ(container.get<int>("c_as_int"), 10);
      EXPECT_EQ(container.get<std::string>("s"), "hello");
    }

    {
      InputParameterContainer container;
      std::string stream("a 1 b 2.0 c _ hello");
      ValueParser parser(stream);
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(
          spec.fully_parse(parser, container), std::invalid_argument, "stoi");
    }
  }

  TEST(InputSpecTest, Unparsed)
  {
    auto spec = all_of({
        parameter<int>("a"),
        parameter<int>("optional", {.default_value = 42}),
        parameter<double>("b"),
        parameter<std::string>("c"),
    });
    InputParameterContainer container;
    std::string stream("a 1 b 2.0 c string unparsed unparsed");
    ValueParser parser(stream);
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(spec.fully_parse(parser, container), Core::Exception,
        "line still contains 'unparsed unparsed'");
  }


  TEST(InputSpecTest, Groups)
  {
    auto spec = all_of({
        parameter<int>("a"),
        group("group1",
            {
                parameter<double>("b"),
            }),
        group("group2",
            {
                parameter<double>("b", {.default_value = 3.0}),
                parameter<std::string>("c"),
            },
            {
                .required = false,
            }),
        group("group3",
            {
                parameter<std::string>("c", {.default_value = "default"}),
            },
            {
                .required = false,
                .defaultable = true,
            }),
        parameter<std::string>("c"),
    });

    {
      InputParameterContainer container;
      std::string stream("a 1 group1 b 2.0 c string");
      ValueParser parser(stream);
      spec.fully_parse(parser, container);
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
      spec.fully_parse(parser, container);
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
      spec.fully_parse(parser, container);
      const auto& const_container = container;
      EXPECT_EQ(const_container.get<int>("a"), 1);
      EXPECT_ANY_THROW([[maybe_unused]] const auto& c = const_container.group("group2"));
      EXPECT_EQ(const_container.group("group1").get<double>("b"), 4.0);
      EXPECT_EQ(const_container.get<std::string>("c"), "string");
    }
  }

  TEST(InputSpecTest, NestedAllOf)
  {
    auto spec = all_of({
        parameter<int>("a"),
        all_of({
            all_of({
                parameter<double>("b"),
            }),
            // Not useful but might happen in practice, so ensure this can be handled.
            all_of({}),
        }),
    });

    {
      InputParameterContainer container;
      std::string stream("a 1 b 2.0");
      ValueParser parser(stream);
      spec.fully_parse(parser, container);
      const auto& const_container = container;
      EXPECT_EQ(const_container.get<int>("a"), 1);
      EXPECT_EQ(const_container.get<double>("b"), 2.0);
    }
  }

  TEST(InputSpecTest, OneOf)
  {
    auto spec = all_of({
        parameter<int>("a", {.default_value = 42}),
        one_of({
            parameter<double>("b"),
            group("group",
                {
                    parameter<std::string>("c"),
                    parameter<double>("d"),
                }),
        }),
    });

    {
      InputParameterContainer container;
      std::string stream("a 1 b 2");
      ValueParser parser(stream);
      spec.fully_parse(parser, container);
      EXPECT_EQ(container.get<int>("a"), 1);
      EXPECT_EQ(container.get<double>("b"), 2);
    }

    {
      InputParameterContainer container;
      std::string stream("group c string d 2.0");
      ValueParser parser(stream);
      spec.fully_parse(parser, container);
      EXPECT_EQ(container.get<int>("a"), 42);
      EXPECT_EQ(container.group("group").get<std::string>("c"), "string");
      EXPECT_EQ(container.group("group").get<double>("d"), 2);
    }

    {
      InputParameterContainer container;
      std::string stream("a 1 group c string d 2.0 b 3.0");
      ValueParser parser(stream);
      // More than one of the one_of entries is present. Refuse to parse any of them.
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(
          spec.fully_parse(parser, container), Core::Exception, "still contains 'b 3.0'");
    }

    {
      InputParameterContainer container;
      std::string stream("a 1 group c string");
      // Note: we start to parse the group, but the entries are not complete, so we backtrack.
      // The result is that the parts of the group remain unparsed.
      ValueParser parser(stream);
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(spec.fully_parse(parser, container), Core::Exception,
          "Required 'one_of' not found in input line");
    }

    {
      InputParameterContainer container;
      std::string stream("a 1");
      ValueParser parser(stream);
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(spec.fully_parse(parser, container), Core::Exception,
          "Required 'one_of' not found in input line");
    }
  }

  TEST(InputSpecTest, OneOfTopLevel)
  {
    auto spec = one_of(
        {
            all_of({
                parameter<int>("a"),
                parameter<double>("b"),
            }),
            all_of({
                parameter<std::string>("c"),
                parameter<double>("d"),
            }),
        },
        // Additionally store the index of the parsed group but map it to a different value.
        store_index_as<int>("index", /*reindex*/ {1, 10}));

    {
      InputParameterContainer container;
      std::string stream("a 1 b 2");
      ValueParser parser(stream);
      spec.fully_parse(parser, container);
      EXPECT_EQ(container.get<int>("a"), 1);
      EXPECT_EQ(container.get<double>("b"), 2);
      EXPECT_EQ(container.get<int>("index"), 1);
    }

    {
      InputParameterContainer container;
      std::string stream("c string d 2.0");
      ValueParser parser(stream);
      spec.fully_parse(parser, container);
      EXPECT_EQ(container.get<std::string>("c"), "string");
      EXPECT_EQ(container.get<double>("d"), 2);
      EXPECT_EQ(container.get<int>("index"), 10);
    }

    {
      InputParameterContainer container;
      std::string stream("a 1 b 2 c string d 2.0");
      ValueParser parser(stream);
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(
          spec.fully_parse(parser, container), Core::Exception, "Ambiguous input in one_of.");
    }

    {
      InputParameterContainer container;
      std::string stream("a 1 c string");
      ValueParser parser(stream);
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(
          spec.fully_parse(parser, container), Core::Exception, "None of the specs fit the input");
    }
  }

  TEST(InputSpecTest, NestedOneOfs)
  {
    auto spec = one_of({
        one_of({
            parameter<int>("a"),
            parameter<double>("b"),
        }),
        one_of({
            parameter<std::string>("c"),
            parameter<double>("d"),
            one_of({
                parameter<int>("e"),
                parameter<std::string>("f"),
            }),
        }),
    });

    {
      // Verify that all entries got pulled to the highest level.
      std::ostringstream out;
      spec.print_as_dat(out);
      EXPECT_EQ(out.str(), R"(// <one_of>:
//   a <int>
//   b <double>
//   c <string>
//   d <double>
//   e <int>
//   f <string>
)");
    }
  }

  TEST(InputSpecTest, NestedOneOfsWithCallback)
  {
    auto spec = one_of({
        one_of({
            parameter<int>("a"),
            parameter<double>("b"),
        }),
        // This one_of has a callback and should not be flattened into the parent one_of.
        one_of(
            {
                parameter<std::string>("c"),
                // This one_of will not be flattened into the parent that has a callback.
                one_of({
                    parameter<double>("d"),
                    // This one_of can be flattened into the parent one_of.
                    one_of({
                        parameter<int>("e"),
                        parameter<std::string>("f"),
                    }),
                }),
            },
            [](InputParameterContainer& container, int index)
            { container.add<int>("index", index); }),
    });

    std::ostringstream out;
    ryml::Tree tree = init_yaml_tree_with_exceptions();
    ryml::NodeRef root = tree.rootref();
    YamlNodeRef yaml(root, "");
    spec.emit_metadata(yaml);
    out << tree;

    std::string expected = R"(type: one_of
specs:
  - type: all_of
    specs:
      - name: a
        type: int
        required: true
  - type: all_of
    specs:
      - name: b
        type: double
        required: true
  - type: all_of
    specs:
      - type: one_of
        specs:
          - type: all_of
            specs:
              - name: c
                type: string
                required: true
          - type: all_of
            specs:
              - type: one_of
                specs:
                  - type: all_of
                    specs:
                      - name: d
                        type: double
                        required: true
                  - type: all_of
                    specs:
                      - name: e
                        type: int
                        required: true
                  - type: all_of
                    specs:
                      - name: f
                        type: string
                        required: true
)";
    EXPECT_EQ(out.str(), expected);
  }

  TEST(InputSpecTest, PrintAsDat)
  {
    enum class Options
    {
      c1,
      c2,
    };
    auto spec = group("g",
        {
            // Note: the all_of entries will be pulled into the parent group.
            all_of({
                parameter<int>("a", {.description = "An integer"}),
                parameter<Options>("c", {.description = "Selection", .default_value = Options::c1}),
            }),
            parameter<int>("d", {.description = "Another\n integer ", .default_value = 42}),
        });

    {
      std::ostringstream out;
      spec.print_as_dat(out);
      EXPECT_EQ(out.str(), R"(// g:
// a <int> "An integer"
// c <Options> (default: c1) "Selection"
// d <int> (default: 42) "Another integer"
)");
    }
  }

  TEST(InputSpecTest, EmitMetadata)
  {
    enum class EnumClass
    {
      A,
      B,
      C,
    };

    auto spec = all_of({
        parameter<int>("a", {.default_value = 42}),
        parameter<std::vector<std::optional<double>>>("b",
            {.default_value = std::vector<std::optional<double>>{1., std::nullopt, 3.}, .size = 3}),
        one_of({
            all_of({
                parameter<std::map<std::string, std::string>>("string to string",
                    {.default_value = std::map<std::string, std::string>{{"key", "abc"}},
                        .size = 1}),
                parameter<std::string>("c"),
            }),
            parameter<std::vector<std::vector<std::vector<int>>>>(
                "triple_vector", {.size = {dynamic_size, 2, from_parameter<int>("a")}}),
            group("group",
                {
                    parameter<std::string>("c", {.description = "A string"}),
                    parameter<double>("d"),
                },
                {.description = "A group"}),
        }),
        parameter<EnumClass>("e", {.default_value = EnumClass::A}),
        parameter<std::optional<EnumClass>>("eo"),
        group("group2",
            {
                parameter<int>("g"),
            },
            {.required = false}),
        list("list",
            all_of({
                parameter<int>("l1"),
                parameter<double>("l2"),
            }),
            {.size = 2}),
        selection<EnumClass>("selection_group", {.selector = "type",
                                                    .choices =
                                                        {
                                                            {EnumClass::A, parameter<int>("a")},
                                                            {EnumClass::B, parameter<int>("b")},
                                                            {EnumClass::C, parameter<int>("c")},
                                                        }}),
    });


    {
      std::ostringstream out;
      ryml::Tree tree = init_yaml_tree_with_exceptions();
      ryml::NodeRef root = tree.rootref();
      YamlNodeRef yaml(root, "");
      spec.emit_metadata(yaml);
      out << tree;

      std::string expected = R"(type: all_of
specs:
  - type: one_of
    specs:
      - type: all_of
        specs:
          - name: a
            type: int
            required: false
            default: 42
          - name: b
            type: vector
            size: 3
            value_type:
              noneable: true
              type: double
            required: false
            default: [1,null,3]
          - name: string to string
            type: map
            size: 1
            value_type:
              type: string
            required: false
            default:
              key: "abc"
          - name: c
            type: string
            required: true
          - name: e
            type: enum
            choices:
              - name: A
              - name: B
              - name: C
            required: false
            default: A
          - name: eo
            noneable: true
            type: enum
            choices:
              - name: A
              - name: B
              - name: C
            required: false
            default: null
          - name: group2
            type: group
            required: false
            defaultable: false
            specs:
              - type: all_of
                specs:
                  - name: g
                    type: int
                    required: true
          - name: list
            type: list
            required: true
            size: 2
            spec:
              type: all_of
              specs:
                - name: l1
                  type: int
                  required: true
                - name: l2
                  type: double
                  required: true
          - name: selection_group
            type: selection
            required: true
            selector: type
            choices:
              - name: A
                spec:
                  name: a
                  type: int
                  required: true
              - name: B
                spec:
                  name: b
                  type: int
                  required: true
              - name: C
                spec:
                  name: c
                  type: int
                  required: true
      - type: all_of
        specs:
          - name: a
            type: int
            required: false
            default: 42
          - name: b
            type: vector
            size: 3
            value_type:
              noneable: true
              type: double
            required: false
            default: [1,null,3]
          - name: triple_vector
            type: vector
            value_type:
              type: vector
              size: 2
              value_type:
                type: vector
                value_type:
                  type: int
            required: true
          - name: e
            type: enum
            choices:
              - name: A
              - name: B
              - name: C
            required: false
            default: A
          - name: eo
            noneable: true
            type: enum
            choices:
              - name: A
              - name: B
              - name: C
            required: false
            default: null
          - name: group2
            type: group
            required: false
            defaultable: false
            specs:
              - type: all_of
                specs:
                  - name: g
                    type: int
                    required: true
          - name: list
            type: list
            required: true
            size: 2
            spec:
              type: all_of
              specs:
                - name: l1
                  type: int
                  required: true
                - name: l2
                  type: double
                  required: true
          - name: selection_group
            type: selection
            required: true
            selector: type
            choices:
              - name: A
                spec:
                  name: a
                  type: int
                  required: true
              - name: B
                spec:
                  name: b
                  type: int
                  required: true
              - name: C
                spec:
                  name: c
                  type: int
                  required: true
      - type: all_of
        specs:
          - name: a
            type: int
            required: false
            default: 42
          - name: b
            type: vector
            size: 3
            value_type:
              noneable: true
              type: double
            required: false
            default: [1,null,3]
          - name: group
            type: group
            description: A group
            required: true
            defaultable: false
            specs:
              - type: all_of
                specs:
                  - name: c
                    type: string
                    description: "A string"
                    required: true
                  - name: d
                    type: double
                    required: true
          - name: e
            type: enum
            choices:
              - name: A
              - name: B
              - name: C
            required: false
            default: A
          - name: eo
            noneable: true
            type: enum
            choices:
              - name: A
              - name: B
              - name: C
            required: false
            default: null
          - name: group2
            type: group
            required: false
            defaultable: false
            specs:
              - type: all_of
                specs:
                  - name: g
                    type: int
                    required: true
          - name: list
            type: list
            required: true
            size: 2
            spec:
              type: all_of
              specs:
                - name: l1
                  type: int
                  required: true
                - name: l2
                  type: double
                  required: true
          - name: selection_group
            type: selection
            required: true
            selector: type
            choices:
              - name: A
                spec:
                  name: a
                  type: int
                  required: true
              - name: B
                spec:
                  name: b
                  type: int
                  required: true
              - name: C
                spec:
                  name: c
                  type: int
                  required: true
)";
      EXPECT_EQ(out.str(), expected);
      std::cout << out.str() << std::endl;
    }
  }

  TEST(InputSpecTest, Copyable)
  {
    InputSpec spec;
    {
      auto tmp = all_of({
          parameter<int>("a"),
          parameter<std::string>("b"),
      });

      spec = all_of({
          tmp,
          parameter<int>("d"),
      });
    }

    {
      InputParameterContainer container;
      std::string stream("a 1 b string d 42");
      ValueParser parser(stream);
      spec.fully_parse(parser, container);
      EXPECT_EQ(container.get<int>("a"), 1);
      EXPECT_EQ(container.get<std::string>("b"), "string");
      EXPECT_EQ(container.get<int>("d"), 42);
    }
  }

  TEST(InputSpecTest, MatchYamlEntry)
  {
    auto spec = parameter<int>("a");

    {
      ryml::Tree tree = init_yaml_tree_with_exceptions();
      ryml::NodeRef root = tree.rootref();

      root |= ryml::MAP;
      root["a"] << 1;
      ConstYamlNodeRef node(root, "");

      InputParameterContainer container;
      spec.match(node, container);
      EXPECT_EQ(container.get<int>("a"), 1);
    }

    {
      SCOPED_TRACE("Error match against sequence node.");
      ryml::Tree tree = init_yaml_tree_with_exceptions();
      ryml::NodeRef root = tree.rootref();

      root |= ryml::SEQ;
      root.append_child() << 1;
      ConstYamlNodeRef node(root, "");

      InputParameterContainer container;
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(
          spec.match(node, container), Core::Exception, "Expected parameter 'a'");
    }

    {
      ryml::Tree tree = init_yaml_tree_with_exceptions();
      ryml::NodeRef root = tree.rootref();

      root |= ryml::MAP;
      root["b"] << 1;
      ConstYamlNodeRef node(root, "");

      InputParameterContainer container;
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(
          spec.match(node, container), Core::Exception, "Expected parameter 'a'");
    }

    {
      SCOPED_TRACE("Wrong type.");
      ryml::Tree tree = init_yaml_tree_with_exceptions();
      ryml::NodeRef root = tree.rootref();

      root |= ryml::MAP;
      root["a"] << "string";
      ConstYamlNodeRef node(root, "");

      InputParameterContainer container;
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(spec.match(node, container), Core::Exception,
          "Candidate parameter 'a' has wrong type, expected type: int");
    }
  }

  TEST(InputSpecTest, MatchYamlGroup)
  {
    auto spec = group("group", {
                                   parameter<int>("a"),
                                   parameter<std::string>("b"),
                               });

    {
      ryml::Tree tree = init_yaml_tree_with_exceptions();
      ryml::NodeRef root = tree.rootref();

      root |= ryml::MAP;
      root["group"] |= ryml::MAP;
      root["group"]["a"] << 1;
      root["group"]["b"] << "b";

      {
        SCOPED_TRACE("Match root node.");
        ConstYamlNodeRef node(root, "");
        InputParameterContainer container;
        spec.match(node, container);
        EXPECT_EQ(container.group("group").get<int>("a"), 1);
        EXPECT_EQ(container.group("group").get<std::string>("b"), "b");
      }

      {
        SCOPED_TRACE("Match group node.");
        ConstYamlNodeRef node(root["group"], "");
        InputParameterContainer container;
        spec.match(node, container);
        EXPECT_EQ(container.group("group").get<int>("a"), 1);
        EXPECT_EQ(container.group("group").get<std::string>("b"), "b");
      }
    }
  }


  TEST(InputSpecTest, MatchYamlSelectionEnum)
  {
    enum class Model
    {
      linear,
      quadratic,
    };

    auto spec = selection<Model>("model",
        {
            .choices =
                {
                    {Model::linear, parameter<double>("coefficient")},
                    {Model::quadratic, one_of({
                                           all_of({
                                               parameter<int>("a"),
                                               parameter<double>("b"),
                                           }),
                                           parameter<double>("c"),
                                       })},
                },
        },
        {.description = "", .required = false});
    {
      SCOPED_TRACE("First selection");
      ryml::Tree tree = init_yaml_tree_with_exceptions();
      ryml::NodeRef root = tree.rootref();
      ryml::parse_in_arena(R"(model:
  type: linear
  coefficient: 1.0
)",
          root);

      ConstYamlNodeRef node(root, "");
      InputParameterContainer container;
      spec.match(node, container);
      EXPECT_EQ(container.group("model").get<Model>("type"), Model::linear);
      EXPECT_EQ(container.group("model").get<double>("coefficient"), 1.0);
    }

    {
      SCOPED_TRACE("Second selection");
      ryml::Tree tree = init_yaml_tree_with_exceptions();
      ryml::NodeRef root = tree.rootref();
      ryml::parse_in_arena(R"(model:
  type: quadratic
  a: 1
  b: 2.0
)",
          root);

      ConstYamlNodeRef node(root, "");
      InputParameterContainer container;
      spec.match(node, container);
      EXPECT_EQ(container.group("model").get<Model>("type"), Model::quadratic);
      EXPECT_EQ(container.group("model").get<int>("a"), 1);
      EXPECT_EQ(container.group("model").get<double>("b"), 2.0);
    }

    {
      SCOPED_TRACE("Second selection, other one_of");
      ryml::Tree tree = init_yaml_tree_with_exceptions();
      ryml::NodeRef root = tree.rootref();
      ryml::parse_in_arena(R"(model:
  type: quadratic
  c: 3.0
)",
          root);

      ConstYamlNodeRef node(root, "");
      InputParameterContainer container;
      spec.match(node, container);
      EXPECT_EQ(container.group("model").get<Model>("type"), Model::quadratic);
      EXPECT_EQ(container.group("model").get<double>("c"), 3.0);
    }

    {
      SCOPED_TRACE("Type matched but wrong choice");
      ryml::Tree tree = init_yaml_tree_with_exceptions();
      ryml::NodeRef root = tree.rootref();
      ryml::parse_in_arena(R"(model:
  type: quadratic
  coefficient: 1
)",
          root);

      ConstYamlNodeRef node(root, "");
      InputParameterContainer container;
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(
          spec.match(node, container), Core::Exception, "Expected parameter 'a'");
    }

    {
      SCOPED_TRACE("Type matched but wrong choice v2");
      ryml::Tree tree = init_yaml_tree_with_exceptions();
      ryml::NodeRef root = tree.rootref();
      ryml::parse_in_arena(R"(model:
  type: quadratic
  a: 1
)",
          root);

      ConstYamlNodeRef node(root, "");
      InputParameterContainer container;
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(
          spec.match(node, container), Core::Exception, "Expected parameter 'b'");
    }
  }

  TEST(InputSpecTest, MatchYamlDeprecatedSelection)
  {
    enum class Enum
    {
      a,
      b,
    };

    auto spec = deprecated_selection<Enum>("enum", {{"A", Enum::a}, {"B", Enum::b}});

    {
      SCOPED_TRACE("Match");
      ryml::Tree tree = init_yaml_tree_with_exceptions();
      ryml::NodeRef root = tree.rootref();
      ryml::parse_in_arena(R"(enum: A)", root);

      ConstYamlNodeRef node(root, "");
      InputParameterContainer container;
      spec.match(node, container);
      EXPECT_EQ(container.get<Enum>("enum"), Enum::a);
    }

    {
      SCOPED_TRACE("No match: wrong key");
      ryml::Tree tree = init_yaml_tree_with_exceptions();
      ryml::NodeRef root = tree.rootref();
      ryml::parse_in_arena(R"(this_is_the_wrong_name: A)", root);

      ConstYamlNodeRef node(root, "");
      InputParameterContainer container;
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(
          spec.match(node, container), Core::Exception, "Expected deprecated_selection 'enum'");
    }

    {
      SCOPED_TRACE("No match: wrong value");
      ryml::Tree tree = init_yaml_tree_with_exceptions();
      ryml::NodeRef root = tree.rootref();
      ryml::parse_in_arena(R"(enum: wrong_value)", root);

      ConstYamlNodeRef node(root, "");
      InputParameterContainer container;
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(spec.match(node, container), Core::Exception,
          "Candidate deprecated_selection 'enum' has wrong value, possible values: A|B");
    }
  }


  TEST(InputSpecTest, MatchYamlAllOf)
  {
    auto spec = all_of({
        parameter<int>("a"),
        parameter<std::vector<std::string>>("b"),
        parameter<std::optional<int>>("c"),
        group("group",
            {
                parameter<int>("d"),
            },
            {.required = false}),
    });

    // Full parse
    {
      ryml::Tree tree = init_yaml_tree_with_exceptions();
      ryml::NodeRef root = tree.rootref();

      root |= ryml::MAP;
      root["a"] << 1;
      root["b"] |= ryml::SEQ;
      root["b"].append_child() << "b1";
      root["b"].append_child() << "b2";
      root["c"] << 2;
      root["group"] |= ryml::MAP;
      root["group"]["d"] << 42;

      ConstYamlNodeRef node(root, "");

      InputParameterContainer container;
      spec.match(node, container);
      EXPECT_EQ(container.get<int>("a"), 1);
      const auto& b = container.get<std::vector<std::string>>("b");
      EXPECT_EQ(b.size(), 2);
      EXPECT_EQ(b[0], "b1");
      EXPECT_EQ(b[1], "b2");
      EXPECT_EQ(container.get<std::optional<int>>("c").value(), 2);
      EXPECT_EQ(container.group("group").get<int>("d"), 42);
    }

    // default left out
    {
      ryml::Tree tree = init_yaml_tree_with_exceptions();
      ryml::NodeRef root = tree.rootref();

      ryml::parse_in_arena(R"(a: 1
b:
    - b1
    - b2
)",
          root);
      ConstYamlNodeRef node(root, "");

      InputParameterContainer container;
      spec.match(node, container);
      EXPECT_FALSE(container.get<std::optional<int>>("c").has_value());
      EXPECT_FALSE(container.has_group("group"));
    }

    {
      SCOPED_TRACE("Explicit null in selection");
      ryml::Tree tree = init_yaml_tree_with_exceptions();
      ryml::NodeRef root = tree.rootref();

      ryml::parse_in_arena(R"(a: 1
b:
    - b1
    - b2
c: null
group:
    d: 42
)",
          root);
      ConstYamlNodeRef node(root, "");

      InputParameterContainer container;
      spec.match(node, container);
      EXPECT_FALSE(container.get<std::optional<int>>("c").has_value());
      EXPECT_EQ(container.group("group").get<int>("d"), 42);
    }

    // too little input
    {
      ryml::Tree tree = init_yaml_tree_with_exceptions();
      ryml::NodeRef root = tree.rootref();

      root |= ryml::MAP;
      root["a"] << 1;
      ConstYamlNodeRef node(root, "");

      InputParameterContainer container;
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(
          spec.match(node, container), Core::Exception, "Expected parameter 'b'");
    }

    // too much input
    {
      ryml::Tree tree = init_yaml_tree_with_exceptions();
      ryml::NodeRef root = tree.rootref();

      ryml::parse_in_arena(R"(a: 1
b:
    - b1
    - b2
c: 2
d:
  d1: 42
  d2:
    d3: 43
)",
          root);

      ConstYamlNodeRef node(root, "");

      InputParameterContainer container;
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(spec.match(node, container), Core::Exception,
          R"({
  [ ] Matched parameter 'a'
  [ ] Matched parameter 'b'
  [ ] Matched parameter 'c'
  [ ] Skipped optional group 'group'
  [!] The following data remains unused:
    d:
      d1: 42
      d2:
        d3: 43
})");
    }
  }

  TEST(InputSpecTest, MatchYamlOneOf)
  {
    auto spec = one_of({
        all_of({
            parameter<int>("a"),
            parameter<std::string>("b"),
        }),
        all_of({
            parameter<std::string>("a"),
            parameter<double>("d"),
        }),
    });

    {
      ryml::Tree tree = init_yaml_tree_with_exceptions();
      ryml::NodeRef root = tree.rootref();

      root |= ryml::MAP;
      root["a"] << 1;
      root["b"] << "b";
      ConstYamlNodeRef node(root, "");

      InputParameterContainer container;
      spec.match(node, container);
      EXPECT_EQ(container.get<int>("a"), 1);
      EXPECT_EQ(container.get<std::string>("b"), "b");
    }

    {
      ryml::Tree tree = init_yaml_tree_with_exceptions();
      ryml::NodeRef root = tree.rootref();

      root |= ryml::MAP;
      root["a"] << "c";
      root["d"] << 2.0;
      ConstYamlNodeRef node(root, "");

      InputParameterContainer container;
      spec.match(node, container);
      EXPECT_EQ(container.get<std::string>("a"), "c");
      EXPECT_EQ(container.get<double>("d"), 2.0);
    };

    {
      SCOPED_TRACE("Multiple possible matches.");
      ryml::Tree tree = init_yaml_tree_with_exceptions();
      ryml::NodeRef root = tree.rootref();

      root |= ryml::MAP;
      root["a"] << 1;
      root["b"] << "b";
      root["d"] << 2.0;
      ConstYamlNodeRef node(root, "");

      InputParameterContainer container;
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(
          spec.match(node, container), Core::Exception, R"([X] Expected one of:
    {
      [ ] Matched parameter 'a'
      [ ] Matched parameter 'b'
      [!] The following data remains unused:
        d: 2
    }
    {
      [ ] Matched parameter 'a'
      [ ] Matched parameter 'd'
      [!] The following data remains unused:
        b: b
    })");
    }
  }

  TEST(InputSpecTest, MatchYamlList)
  {
    auto spec = list("list",
        all_of({
            parameter<int>("a"),
            parameter<std::string>("b"),
        }),
        {.size = 1});

    {
      ryml::Tree tree = init_yaml_tree_with_exceptions();
      ryml::NodeRef root = tree.rootref();

      root |= ryml::MAP;
      auto list_node = root.append_child();
      list_node << ryml::key("list");
      list_node |= ryml::SEQ;
      {
        auto first_entry = list_node.append_child();
        first_entry |= ryml::MAP;
        first_entry["a"] << 1;
        first_entry["b"] << "string";
      }

      {
        SCOPED_TRACE("Match root node.");
        ConstYamlNodeRef node(root, "");

        InputParameterContainer container;
        spec.match(node, container);
        const auto& list = container.get_list("list");
        EXPECT_EQ(list.size(), 1);
        EXPECT_EQ(list[0].get<int>("a"), 1);
        EXPECT_EQ(list[0].get<std::string>("b"), "string");
      }

      {
        SCOPED_TRACE("Match list node.");
        ConstYamlNodeRef node(root["list"], "");

        InputParameterContainer container;
        spec.match(node, container);
        const auto& list = container.get_list("list");
        EXPECT_EQ(list.size(), 1);
        EXPECT_EQ(list[0].get<int>("a"), 1);
        EXPECT_EQ(list[0].get<std::string>("b"), "string");
      }
    }

    // unmatched node
    {
      ryml::Tree tree = init_yaml_tree_with_exceptions();
      ryml::NodeRef root = tree.rootref();

      root |= ryml::MAP;
      auto list_node = root.append_child();
      list_node << ryml::key("list");
      list_node |= ryml::SEQ;
      {
        auto first_entry = list_node.append_child();
        first_entry |= ryml::MAP;
        first_entry["a"] << "wrong type";
        first_entry["b"] << "string";
      }
      {
        auto second_entry = list_node.append_child();
        second_entry |= ryml::MAP;
        second_entry["a"] << 2;
        second_entry["b"] << "string2";
      }
      ConstYamlNodeRef node(root, "");

      InputParameterContainer container;
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(
          spec.match(node, container), Core::Exception, "The following list entry did not match:");
    }

    // too many entries
    {
      ryml::Tree tree = init_yaml_tree_with_exceptions();
      ryml::NodeRef root = tree.rootref();

      root |= ryml::MAP;
      auto list_node = root.append_child();
      list_node << ryml::key("list");
      list_node |= ryml::SEQ;
      {
        auto first_entry = list_node.append_child();
        first_entry |= ryml::MAP;
        first_entry["a"] << 1;
        first_entry["b"] << "string";
      }
      {
        auto second_entry = list_node.append_child();
        second_entry |= ryml::MAP;
        second_entry["a"] << 2;
        second_entry["b"] << "string2";
      }
      ConstYamlNodeRef node(root, "");

      InputParameterContainer container;
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(spec.match(node, container), Core::Exception,
          "Too many list entries encountered: expected 1 but matched 2");
    }
  }

  TEST(InputSpecTest, MatchYamlPath)
  {
    auto spec = all_of({
        parameter<std::filesystem::path>("a"),
    });

    {
      ryml::Tree tree = init_yaml_tree_with_exceptions();
      ryml::NodeRef root = tree.rootref();

      root |= ryml::MAP;
      root["a"] << "dir/file.txt";
      ConstYamlNodeRef node(root, "path/to/input.yaml");

      InputParameterContainer container;
      spec.match(node, container);
      EXPECT_EQ(container.get<std::filesystem::path>("a"), "path/to/dir/file.txt");
    }

    {
      ryml::Tree tree = init_yaml_tree_with_exceptions();
      ryml::NodeRef root = tree.rootref();

      root |= ryml::MAP;
      root["a"] << "dir/file.txt";
      ConstYamlNodeRef node(root, "input.yaml");

      InputParameterContainer container;
      spec.match(node, container);
      EXPECT_EQ(container.get<std::filesystem::path>("a"), "dir/file.txt");
    }

    {
      ryml::Tree tree = init_yaml_tree_with_exceptions();
      ryml::NodeRef root = tree.rootref();

      root |= ryml::MAP;
      root["a"] << "/root/dir/file.txt";
      ConstYamlNodeRef node(root, "path/to/input.yaml");

      InputParameterContainer container;
      spec.match(node, container);
      EXPECT_EQ(container.get<std::filesystem::path>("a"), "/root/dir/file.txt");
    }
  }

  TEST(InputSpecTest, MatchYamlOptional)
  {
    auto spec = all_of({
        parameter<std::optional<int>>("i"),
        parameter<std::optional<std::string>>("s"),
        parameter<std::vector<std::optional<double>>>("v", {.size = 3}),
    });

    {
      ryml::Tree tree = init_yaml_tree_with_exceptions();
      ryml::parse_in_arena(R"(i : 1
s: string
v: [1.0, 2.0, 3.0]
)",
          &tree);
      ryml::NodeRef root = tree.rootref();
      ConstYamlNodeRef node(root, "");

      InputParameterContainer container;
      spec.match(node, container);
      EXPECT_EQ(container.get<std::optional<int>>("i"), 1);
      EXPECT_EQ(container.get<std::optional<std::string>>("s"), "string");
      const auto& v = container.get<std::vector<std::optional<double>>>("v");
      EXPECT_EQ(v.size(), 3);
      EXPECT_EQ(v[0], 1.0);
      EXPECT_EQ(v[1], 2.0);
      EXPECT_EQ(v[2], 3.0);
    }

    {
      ryml::Tree tree = init_yaml_tree_with_exceptions();
      ryml::parse_in_arena(R"(i : null
s: # Note: leaving the key out is the same as setting null
v: [Null, NULL, ~] # all the other spellings that YAML supports
)",
          &tree);
      ryml::NodeRef root = tree.rootref();
      ConstYamlNodeRef node(root, "");

      InputParameterContainer container;
      spec.match(node, container);
      EXPECT_EQ(container.get<std::optional<int>>("i"), std::nullopt);
      EXPECT_EQ(container.get<std::optional<std::string>>("s"), std::nullopt);
      const auto& v = container.get<std::vector<std::optional<double>>>("v");
      EXPECT_EQ(v.size(), 3);
      EXPECT_EQ(v[0], std::nullopt);
      EXPECT_EQ(v[1], std::nullopt);
      EXPECT_EQ(v[2], std::nullopt);
    }
  }

  TEST(InputSpecTest, MatchYamlSizes)
  {
    using ComplicatedType = std::vector<std::map<std::string, std::vector<int>>>;
    auto spec = all_of({
        parameter<int>("num"),
        parameter<ComplicatedType>("v", {.size = {2, dynamic_size, from_parameter<int>("num")}}),
    });

    {
      SCOPED_TRACE("Expected sizes");
      ryml::Tree tree = init_yaml_tree_with_exceptions();
      ryml::parse_in_arena(R"(num: 2
v:
  - key1: [1, 2]
    key2: [3, 4]
  - key1: [5, 6])",
          &tree);
      ryml::NodeRef root = tree.rootref();
      ConstYamlNodeRef node(root, "");

      InputParameterContainer container;
      spec.match(node, container);
      const auto& v = container.get<ComplicatedType>("v");
      EXPECT_EQ(v.size(), 2);
    }

    {
      SCOPED_TRACE("Wrong size from_parameter");
      ryml::Tree tree = init_yaml_tree_with_exceptions();
      ryml::parse_in_arena(R"(num: 2
v:
  - key1: [1, 2, 3]
    key2: [3, 4]
  - key1: [5, 6])",
          &tree);
      ryml::NodeRef root = tree.rootref();
      ConstYamlNodeRef node(root, "");

      InputParameterContainer container;
      EXPECT_THROW(spec.match(node, container), Core::Exception);
    }

    {
      SCOPED_TRACE("Wrong size explicitly set.");
      ryml::Tree tree = init_yaml_tree_with_exceptions();
      ryml::parse_in_arena(R"(num: 2
v:
  - key1: [1, 2]
  - key1: [5, 6]
  - key1: [7, 8])",
          &tree);
      ryml::NodeRef root = tree.rootref();
      ConstYamlNodeRef node(root, "");

      InputParameterContainer container;
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(
          spec.match(node, container), Core::Exception, "Candidate parameter 'v'");
    }
  }

  TEST(InputSpecTest, MaterialExample)
  {
    auto mat_spec = all_of({
        parameter<int>("MAT"),
        one_of({
            group("MAT_A",
                {
                    parameter<int>("a"),
                }),
            group("MAT_B",
                {
                    parameter<int>("b"),
                }),
        }),
    });

    {
      ryml::Tree tree = init_yaml_tree_with_exceptions();
      ryml::NodeRef root = tree.rootref();

      root |= ryml::MAP;
      root["MAT"] << 1;

      root["MAT_A"] |= ryml::MAP;
      root["MAT_A"]["a"] << 2;
      ConstYamlNodeRef node(root, "");

      InputParameterContainer container;
      mat_spec.match(node, container);
      EXPECT_EQ(container.get<int>("MAT"), 1);
      EXPECT_EQ(container.group("MAT_A").get<int>("a"), 2);
    }
  }

  TEST(InputSpecTest, EmptyMatchesAllDefaulted)
  {
    // This was a bug where a single defaulted parameter was incorrectly reported as not matching.
    auto spec = all_of({
        parameter<int>("a", {.default_value = 42}),
    });

    {
      ryml::Tree tree = init_yaml_tree_with_exceptions();
      ryml::NodeRef root = tree.rootref();
      root |= ryml::MAP;
      ConstYamlNodeRef node(root, "");

      InputParameterContainer container;
      spec.match(node, container);
      EXPECT_EQ(container.get<int>("a"), 42);
    }
  }

  TEST(InputSpecTest, SizedOptionalVector)
  {
    // This was a bug where an optional vector was not parsed correctly.
    auto spec = all_of({
        parameter<int>("num", {.default_value = 2}),
        parameter<std::optional<std::vector<double>>>("v", {.size = from_parameter<int>("num")}),
    });

    {
      SCOPED_TRACE("Optional has value");
      ryml::Tree tree = init_yaml_tree_with_exceptions();
      ryml::parse_in_arena(R"(num: 2
v: [1.0, 2.0])",
          &tree);
      ConstYamlNodeRef node(tree.rootref(), "");

      InputParameterContainer container;
      spec.match(node, container);
      EXPECT_EQ(container.get<int>("num"), 2);
      const auto& v = container.get<std::optional<std::vector<double>>>("v");
      EXPECT_TRUE(v.has_value());
      EXPECT_EQ(v->size(), 2);
      EXPECT_EQ((*v)[0], 1.0);
      EXPECT_EQ((*v)[1], 2.0);
    }

    {
      SCOPED_TRACE("Empty optional");
      ryml::Tree tree = init_yaml_tree_with_exceptions();
      ryml::parse_in_arena(R"(num: 2
v: null)",
          &tree);
      ConstYamlNodeRef node(tree.rootref(), "");

      InputParameterContainer container;
      spec.match(node, container);
      EXPECT_EQ(container.get<int>("num"), 2);
      const auto& v = container.get<std::optional<std::vector<double>>>("v");
      EXPECT_FALSE(v.has_value());
    }
  }

  TEST(InputSpecTest, DatToYaml)
  {
    auto spec = all_of({
        // awkward one_of where the first choice partially matches
        one_of({
            all_of({
                parameter<int>("a"),
                parameter<std::string>("b"),
            }),
            all_of({
                parameter<int>("a"),
                parameter<double>("d"),
            }),
        }),
        // group with all defaulted entries
        group("group",
            {
                parameter<double>("c", {.default_value = 1.0}),
            },
            {.required = false}),
        list("list",
            all_of({
                parameter<int>("l1"),
                parameter<double>("l2"),
            }),
            {.size = 2}),
        parameter<int>("i", {.default_value = 0}),
        parameter<std::vector<double>>("v", {.size = 3}),
    });

    std::string dat = "a 1 d 3.0 group c 1 i 42 v 1.0 2.0 3.0 list l1 1 l2 2.0 l1 3 l2 4.0";

    InputParameterContainer container;
    ValueParser parser(dat);
    spec.fully_parse(parser, container);

    {
      SCOPED_TRACE("Emit without default values");
      auto tree = init_yaml_tree_with_exceptions();
      YamlNodeRef yaml(tree.rootref(), "");
      spec.emit(yaml, container);

      std::ostringstream out;
      out << tree;
      std::string expected = R"(a: 1
d: 3
list:
  - l1: 1
    l2: 2
  - l1: 3
    l2: 4
i: 42
v: [1,2,3]
)";
      EXPECT_EQ(out.str(), expected);
    }

    {
      SCOPED_TRACE("Emit with defaulted values");
      auto tree = init_yaml_tree_with_exceptions();
      YamlNodeRef yaml(tree.rootref(), "");
      spec.emit(yaml, container, {.emit_defaulted_values = true});

      std::ostringstream out;
      out << tree;
      std::string expected = R"(a: 1
d: 3
group:
  c: 1
list:
  - l1: 1
    l2: 2
  - l1: 3
    l2: 4
i: 42
v: [1,2,3]
)";
      EXPECT_EQ(out.str(), expected);
    }
  }

  TEST(InputSpecTest, ComplexMatchError)
  {
    // Let this test look a little bit more like actual input so it doubles as documentation.
    enum class Type
    {
      user,
      gemm,
    };
    auto spec = group("parameters", {
                                        parameter<double>("start"),
                                        parameter<bool>("write_output", {.default_value = true}),
                                        group("TimeIntegration",
                                            {
                                                one_of({
                                                    group("OST",
                                                        {
                                                            parameter<double>("theta"),
                                                        }),
                                                    group("Special", {parameter<Type>("type")}),
                                                }),
                                            }),
                                    });

    {
      SCOPED_TRACE("Partial match in one_of");
      auto tree = init_yaml_tree_with_exceptions();
      ryml::NodeRef root = tree.rootref();
      ryml::parse_in_arena(R"(parameters:
  start: 0.0
  TimeIntegration:
    OST:
      theta: true # wrong type
    Special:
      type: invalid)",
          root);

      ConstYamlNodeRef node(root, "");
      InputParameterContainer container;
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(spec.match(node, container), Core::Exception,
          R"({
  [!] Candidate group 'parameters'
    {
      [ ] Matched parameter 'start'
      [ ] Defaulted parameter 'write_output'
      [!] Candidate group 'TimeIntegration'
        {
          [X] Expected one of:
            {
              [!] Candidate group 'OST'
                {
                  [!] Candidate parameter 'theta' has wrong type, expected type: double
                }
              [!] The following data remains unused:
                Special:
                  type: invalid
            }
            {
              [!] Candidate group 'Special'
                {
                  [!] Candidate parameter 'type' has wrong value, possible values: user|gemm
                }
              [!] The following data remains unused:
                OST:
                  theta: true
            }
          [!] The following data remains unused:
            Special:
              type: invalid
            OST:
              theta: true
        }
    }
})");
    }
    {
      SCOPED_TRACE("Unused parts.");
      auto tree = init_yaml_tree_with_exceptions();
      ryml::NodeRef root = tree.rootref();
      ryml::parse_in_arena(R"(data:
  a: 1
parameters:
  start: 0.0
  unused: "abc"
  TimeIntegration:
    OST:
      theta: 0.5
    Special:)",
          root);

      ConstYamlNodeRef node(root, "");
      InputParameterContainer container;
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(spec.match(node, container), Core::Exception,
          R"({
  [!] Candidate group 'parameters'
    {
      [ ] Matched parameter 'start'
      [ ] Defaulted parameter 'write_output'
      [!] Candidate group 'TimeIntegration'
        {
          [X] Expected one of:
            {
              [!] Candidate group 'OST'
                {
                  [ ] Matched parameter 'theta'
                }
              [!] The following data remains unused:
                Special: 
            }
          [!] The following data remains unused:
            Special: 
            OST:
              theta: 0.5
        }
      [!] The following data remains unused:
        unused: "abc"
    }
  [!] The following data remains unused:
    data:
      a: 1
})");
    }
  }

  TEST(InputSpecTest, ParameterValidation)
  {
    auto spec = all_of({
        parameter<int>("a", {.default_value = 42, .validator = Validators::in_range(0, 50)}),
        parameter<std::optional<double>>("b", {.validator = Validators::positive<double>()}),
    });

    {
      SCOPED_TRACE("Valid input");
      ryml::Tree tree = init_yaml_tree_with_exceptions();
      ryml::NodeRef root = tree.rootref();
      ryml::parse_in_arena(R"(a: 1
b: 2.0)",
          root);

      ConstYamlNodeRef node(root, "");
      InputParameterContainer container;
      spec.match(node, container);
      EXPECT_EQ(container.get<int>("a"), 1);
      EXPECT_EQ(*container.get<std::optional<double>>("b"), 2.0);
    }

    {
      SCOPED_TRACE("Valid input with defaulted parameter");
      ryml::Tree tree = init_yaml_tree_with_exceptions();
      ryml::NodeRef root = tree.rootref();
      ryml::parse_in_arena(R"(a: 1)", root);

      ConstYamlNodeRef node(root, "");
      InputParameterContainer container;
      spec.match(node, container);
      EXPECT_EQ(container.get<int>("a"), 1);
      EXPECT_FALSE(container.get<std::optional<double>>("b").has_value());
    }

    {
      SCOPED_TRACE("Validation failure");
      ryml::Tree tree = init_yaml_tree_with_exceptions();
      ryml::NodeRef root = tree.rootref();
      ryml::parse_in_arena(R"(a: -1
b: 0.0)",
          root);

      ConstYamlNodeRef node(root, "");
      InputParameterContainer container;
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(spec.match(node, container), Core::Exception, R"({
  [!] Candidate parameter 'a' does not pass validation: in_range[0,50]
  [!] Candidate parameter 'b' does not pass validation: in_range(0,1.7976931348623157e+308]
})");
    }
  }

  TEST(InputSpecTest, DefaultedParameterValidation)
  {
    using namespace Core::IO::InputSpecBuilders::Validators;
    const auto construct = []()
    {
      [[maybe_unused]] auto spec =
          parameter<int>("a", {.default_value = 42, .validator = in_range(excl(0), 10)});
    };
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(construct(), Core::Exception,
        "Default value '42' does not pass validation: in_range(0,10]");
  }

  TEST(InputSpecTest, OneOfOverlappingOptionsSingleParameter)
  {
    // This is a tricky case, where one_of the choices is a single parameter
    const auto spec = one_of({parameter<int>("a"), all_of({
                                                       parameter<int>("a"),
                                                       parameter<int>("b"),
                                                   })});

    {
      SCOPED_TRACE("Overlapping values.");
      ryml::Tree tree = init_yaml_tree_with_exceptions();
      ryml::parse_in_arena(R"(a: 1
b: 2)",
          &tree);
      const ConstYamlNodeRef node(tree.rootref(), "");

      InputParameterContainer container;
      spec.match(node, container);
      EXPECT_EQ(container.get<int>("a"), 1);
      EXPECT_EQ(container.get<int>("b"), 2);
    }
  }

  TEST(InputSpecTest, OneOfOverlappingOptionsSafeAllOf)
  {
    // This case is similar to the previous one, but already uses all_ofs.
    const auto spec = one_of({
        all_of({
            parameter<int>("a"),
            parameter<int>("b"),
        }),
        all_of({
            parameter<int>("a"),
            parameter<int>("b"),
            parameter<int>("c"),
        }),
    });
    ryml::Tree tree = init_yaml_tree_with_exceptions();
    ryml::parse_in_arena(R"(a: 1
b: 2
c: 3)",
        &tree);
    const ConstYamlNodeRef node(tree.rootref(), "");
    InputParameterContainer container;
    spec.match(node, container);
    EXPECT_EQ(container.get<int>("a"), 1);
    EXPECT_EQ(container.get<int>("b"), 2);
    EXPECT_EQ(container.get<int>("c"), 3);
  }
}  // namespace

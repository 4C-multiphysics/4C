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
        entry<int>("a", {.description = "An integer", .default_value = 1}),
        entry<double>("b", {.required = true}),
        entry<std::string>("c", {.required = false}),
        entry<bool>("d"),
    });
    InputParameterContainer container;
    std::string stream("b 2.0 d true // trailing comment");
    ValueParser parser(stream);
    spec.fully_parse(parser, container);
    EXPECT_EQ(container.get<int>("a"), 1);
    EXPECT_EQ(container.get<double>("b"), 2.0);
    EXPECT_EQ(container.get_if<std::string>("c"), nullptr);
    EXPECT_EQ(container.get<bool>("d"), true);
  }

  TEST(InputSpecTest, OptionalLeftOut)
  {
    auto spec = all_of({
        entry<int>("a"),
        entry<double>("b"),
        entry<std::string>("c", {.default_value = "default"}),
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
        entry<int>("a"),
        entry<double>("b"),
        entry<std::string>("c"),
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
        selection<EnumClass>("enum",
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

  TEST(InputSpecTest, ParseSingleDefaultedEntryDat)
  {
    // This used to be a bug where a single default dat parameter was not accepted.
    auto spec = all_of({
        entry<double>("a", {.default_value = 1.0}),
    });

    {
      InputParameterContainer container;
      std::string stream("");
      ValueParser parser(stream);
      spec.fully_parse(parser, container);
      EXPECT_EQ(container.get<double>("a"), 1.0);
    }
  }

  TEST(InputSpecTest, ParseSingleDefaultedSelectionDat)
  {
    // This used to be a bug where a single default selection parameter was not accepted.
    auto spec = all_of({
        selection<int>("a", {{"A", 1}, {"B", 2}}, {.default_value = 2}),
    });

    {
      InputParameterContainer container;
      std::string stream("");
      ValueParser parser(stream);
      spec.fully_parse(parser, container);
      EXPECT_EQ(container.get<int>("a"), 2);
    }
  }


  TEST(InputSpecTest, Vector)
  {
    auto spec = all_of({
        entry<std::vector<std::vector<int>>>("a", {.size = {2, 2}}),
        entry<std::vector<double>>("b", {.size = 3}),
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

  TEST(InputSpecTest, Noneable)
  {
    auto spec = all_of({
        entry<int>("size"),
        entry<std::vector<Noneable<int>>>("a", {.size = from_parameter<int>("size")}),
        entry<Noneable<std::string>>("b", {.description = "b"}),
        entry<Noneable<double>>("c", {.default_value = 1.0}),
        entry<Noneable<bool>>("d", {.required = false}),
        entry<Noneable<int>>("e", {.default_value = none<int>}),
    });

    {
      InputParameterContainer container;
      std::string stream("size 3 a 1 2 3 b none");
      ValueParser parser(stream);
      spec.fully_parse(parser, container);
      const auto& a = container.get<std::vector<Noneable<int>>>("a");
      EXPECT_EQ(a.size(), 3);
      EXPECT_EQ(a[0].has_value(), true);
      EXPECT_EQ(a[0].value(), 1);
      EXPECT_EQ(a[1].has_value(), true);
      EXPECT_EQ(a[1].value(), 2);
      EXPECT_EQ(a[2].has_value(), true);
      EXPECT_EQ(a[2].value(), 3);

      const auto& b = container.get<Noneable<std::string>>("b");
      EXPECT_EQ(b.has_value(), false);

      const auto& c = container.get<Noneable<double>>("c");
      EXPECT_EQ(c.has_value(), true);
      EXPECT_EQ(c.value(), 1.0);

      const auto& e = container.get<Noneable<int>>("e");
      EXPECT_EQ(e.has_value(), false);
    }

    {
      InputParameterContainer container;
      std::string stream("size 3 a 1 none 3 b none c none d none e none");
      ValueParser parser(stream);
      spec.fully_parse(parser, container);
      const auto& a = container.get<std::vector<Noneable<int>>>("a");
      EXPECT_EQ(a.size(), 3);
      EXPECT_EQ(a[0].has_value(), true);
      EXPECT_EQ(a[0].value(), 1);
      EXPECT_EQ(a[1].has_value(), false);
      EXPECT_EQ(a[2].has_value(), true);
      EXPECT_EQ(a[2].value(), 3);

      const auto& b = container.get<Noneable<std::string>>("b");
      EXPECT_EQ(b.has_value(), false);

      const auto& c = container.get<Noneable<double>>("c");
      EXPECT_EQ(c.has_value(), false);

      const auto& d = container.get<Noneable<bool>>("d");
      EXPECT_EQ(d.has_value(), false);

      const auto& e = container.get<Noneable<int>>("e");
      EXPECT_EQ(e.has_value(), false);
    }

    {
      InputParameterContainer container;
      std::string stream("size 3 a 1 2 3 b string c 2.0 d true e 42");
      ValueParser parser(stream);
      spec.fully_parse(parser, container);

      const auto& b = container.get<Noneable<std::string>>("b");
      EXPECT_EQ(b.has_value(), true);
      EXPECT_EQ(b.value(), "string");

      const auto& c = container.get<Noneable<double>>("c");
      EXPECT_EQ(c.has_value(), true);
      EXPECT_EQ(c.value(), 2.0);

      const auto& d = container.get<Noneable<bool>>("d");
      EXPECT_EQ(d.has_value(), true);
      EXPECT_EQ(d.value(), true);

      const auto& e = container.get<Noneable<int>>("e");
      EXPECT_EQ(e.has_value(), true);
      EXPECT_EQ(e.value(), 42);
    }
  }

  TEST(InputSpecTest, VectorWithParsedLength)
  {
    auto spec = all_of({
        entry<int>("a"),
        entry<std::vector<double>>("b", {.size = from_parameter<int>("a")}),
        entry<std::string>("c"),
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
        entry<int>("a"),
        entry<double>("b"),
        entry<std::string>("c",
            {
                .description = "A string",
                .default_value = "Not found",
                .on_parse_callback = [](InputParameterContainer& container)
                { container.add<int>("c_as_int", std::stoi(container.get<std::string>("c"))); },
            }),
        entry<std::string>("s"),
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

  TEST(InputSpecTest, Selection)
  {
    auto spec = all_of({
        entry<int>("a"),
        selection<int>("b", {{"b1", 1}, {"b2", 2}}, {.default_value = 1}),
        selection<std::string>("c", {"c1", "c2"}, {.default_value = "c2"}),
        entry<std::string>("d"),
    });

    {
      InputParameterContainer container;
      std::string stream("a 1 b b2 d string c c1");
      ValueParser parser(stream);
      spec.fully_parse(parser, container);
      EXPECT_EQ(container.get<int>("a"), 1);
      EXPECT_EQ(container.get<int>("b"), 2);
      EXPECT_EQ(container.get<std::string>("c"), "c1");
      EXPECT_EQ(container.get<std::string>("d"), "string");
    }

    {
      InputParameterContainer container;
      std::string stream("a 1 b b4 c string");
      ValueParser parser(stream);
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(
          spec.fully_parse(parser, container), Core::Exception, "Invalid value 'b4'");
    }
  }

  TEST(InputSpecTest, Unparsed)
  {
    auto spec = all_of({
        entry<int>("a"),
        entry<int>("optional", {.default_value = 42}),
        entry<double>("b"),
        entry<std::string>("c"),
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
        entry<int>("a"),
        group("group1",
            {
                entry<double>("b"),
            }),
        group("group2",
            {
                entry<double>("b", {.default_value = 3.0}),
                entry<std::string>("c"),
            },
            {
                .required = false,
            }),
        group("group3",
            {
                entry<std::string>("c", {.default_value = "default"}),
            },
            {
                .required = false,
                .defaultable = true,
            }),
        entry<std::string>("c"),
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
        entry<int>("a"),
        all_of({
            all_of({
                entry<double>("b"),
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
        entry<int>("a", {.default_value = 42}),
        one_of({
            entry<double>("b"),
            group("group",
                {
                    entry<std::string>("c"),
                    entry<double>("d"),
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
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(spec.fully_parse(parser, container), Core::Exception,
          "still contains 'group c string d 2.0 b 3.0'");
    }

    {
      InputParameterContainer container;
      std::string stream("a 1 group c string");
      // Note: we start to parse the group, but the entries are not complete, so we backtrack.
      // The result is that the parts of the group remain unparsed.
      ValueParser parser(stream);
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(
          spec.fully_parse(parser, container), Core::Exception, "still contains 'group c string'");
    }

    {
      InputParameterContainer container;
      std::string stream("a 1");
      ValueParser parser(stream);
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(
          spec.fully_parse(parser, container), Core::Exception, "Required 'one_of {b, group}'");
    }
  }

  TEST(InputSpecTest, OneOfTopLevel)
  {
    auto spec = one_of(
        {
            all_of({
                entry<int>("a"),
                entry<double>("b"),
            }),
            all_of({
                entry<std::string>("c"),
                entry<double>("d"),
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
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(spec.fully_parse(parser, container), Core::Exception,
          "both 'all_of {a, b}' and 'all_of {c, d}'");
    }

    {
      InputParameterContainer container;
      std::string stream("a 1 c string");
      ValueParser parser(stream);
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(spec.fully_parse(parser, container), Core::Exception,
          "one_of {all_of {a, b}, all_of {c, d}}");
    }
  }

  TEST(InputSpecTest, NestedOneOfs)
  {
    auto spec = one_of({
        one_of({
            entry<int>("a"),
            entry<double>("b"),
        }),
        one_of({
            entry<std::string>("c"),
            entry<double>("d"),
            one_of({
                entry<int>("e"),
                entry<std::string>("f"),
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
            entry<int>("a"),
            entry<double>("b"),
        }),
        // This one_of has a callback and should not be flattened into the parent one_of.
        one_of(
            {
                entry<std::string>("c"),
                // This one_of will not be flattened into the parent that has a callback.
                one_of({
                    entry<double>("d"),
                    // This one_of can be flattened into the parent one_of.
                    one_of({
                        entry<int>("e"),
                        entry<std::string>("f"),
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
description: 'one_of {a, b, one_of {c, one_of {d, e, f}}}'
specs:
  - name: a
    type: int
    required: true
  - name: b
    type: double
    required: true
  - type: one_of
    description: 'one_of {c, one_of {d, e, f}}'
    specs:
      - name: c
        type: string
        required: true
      - type: one_of
        description: 'one_of {d, e, f}'
        specs:
          - name: d
            type: double
            required: true
          - name: e
            type: int
            required: true
          - name: f
            type: string
            required: true
)";
    EXPECT_EQ(out.str(), expected);
  }

  TEST(InputSpecTest, PrintAsDat)
  {
    auto spec =
        group("g", {
                       // Note: the all_of entries will be pulled into the parent group.
                       all_of({
                           entry<int>("a", {.description = "An integer"}),
                           selection<int>("c", {{"c1", 1}, {"c2", 2}},
                               {.description = "Selection", .default_value = 1}),
                       }),
                       entry<int>("d", {.description = "Another\n integer ", .default_value = 42}),
                   });

    {
      std::ostringstream out;
      spec.print_as_dat(out);
      EXPECT_EQ(out.str(), R"(// g:
//   a <int> "An integer"
//   c (default: c1) (choices: c1|c2|) "Selection"
//   d <int> (optional) (default: 42) "Another integer"
)");
    }
  }

  TEST(InputSpecTest, EmitMetadata)
  {
    auto spec = all_of({
        entry<int>("a", {.default_value = 42}),
        entry<std::vector<Noneable<double>>>(
            "b", {.default_value = std::vector<Noneable<double>>{1., none<double>, 3.}, .size = 3}),
        one_of({
            all_of({
                entry<std::map<std::string, std::string>>(
                    "b", {.default_value = std::map<std::string, std::string>{{"key", "abc"}},
                             .size = 1}),
                entry<std::string>("c"),
            }),
            entry<std::vector<std::vector<std::vector<int>>>>(
                "triple_vector", {.size = {dynamic_size, 2, from_parameter<int>("a")}}),
            group("group",
                {
                    entry<std::string>("c", {.description = "A string"}),
                    entry<double>("d"),
                },
                {.description = "A group"}),
        }),
        selection<int>("e", {{"e1", 1}, {"e2", 2}}, {.default_value = 1}),
        group("group2",
            {
                entry<int>("g"),
            },
            {.required = false}),
        list("list",
            all_of({
                entry<int>("l1"),
                entry<double>("l2"),
            }),
            {.size = 2}),
    });


    {
      std::ostringstream out;
      ryml::Tree tree = init_yaml_tree_with_exceptions();
      ryml::NodeRef root = tree.rootref();
      YamlNodeRef yaml(root, "");
      spec.emit_metadata(yaml);
      out << tree;

      std::string expected = R"(type: all_of
description: 'all_of {a, b, one_of {all_of {b, c}, triple_vector, group}, e, group2, list}'
required: true
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
  - type: one_of
    description: 'one_of {all_of {b, c}, triple_vector, group}'
    specs:
      - type: all_of
        description: 'all_of {b, c}'
        required: true
        specs:
          - name: b
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
      - name: group
        type: group
        description: A group
        required: true
        defaultable: false
        specs:
          - name: c
            type: string
            description: "A string"
            required: true
          - name: d
            type: double
            required: true
  - name: e
    type: selection
    required: false
    default: "e1"
    choices:
      - name: "e1"
      - name: "e2"
  - name: group2
    type: group
    required: false
    defaultable: false
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
      description: 'all_of {l1, l2}'
      required: true
      specs:
        - name: l1
          type: int
          required: true
        - name: l2
          type: double
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
          entry<int>("a"),
          entry<std::string>("b"),
          selection<int>("c", {{"c1", 1}, {"c2", 2}}, {.default_value = 1}),
      });

      spec = all_of({
          tmp,
          entry<int>("d"),
      });
    }

    {
      InputParameterContainer container;
      std::string stream("a 1 b string c c2 d 42");
      ValueParser parser(stream);
      spec.fully_parse(parser, container);
      EXPECT_EQ(container.get<int>("a"), 1);
      EXPECT_EQ(container.get<std::string>("b"), "string");
      EXPECT_EQ(container.get<int>("c"), 2);
      EXPECT_EQ(container.get<int>("d"), 42);
    }
  }

  TEST(InputSpecTest, MatchYamlEntry)
  {
    auto spec = entry<int>("a");

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
          spec.match(node, container), Core::Exception, "Expected entry 'a'");
    }

    {
      ryml::Tree tree = init_yaml_tree_with_exceptions();
      ryml::NodeRef root = tree.rootref();

      root |= ryml::MAP;
      root["b"] << 1;
      ConstYamlNodeRef node(root, "");

      InputParameterContainer container;
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(
          spec.match(node, container), Core::Exception, "Expected entry 'a'");
    }
  }

  TEST(InputSpecTest, MatchYamlGroup)
  {
    auto spec = group("group", {
                                   entry<int>("a"),
                                   entry<std::string>("b"),
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


  TEST(InputSpecTest, MatchYamlAllOf)
  {
    auto spec = all_of({
        entry<int>("a"),
        entry<std::vector<std::string>>("b"),
        selection<int>("c", {{"c1", 1}, {"c2", 2}}, {.default_value = 1}),
        group("group",
            {
                entry<int>("d"),
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
      root["c"] << "c2";
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
      EXPECT_EQ(container.get<int>("c"), 2);
      EXPECT_EQ(container.group("group").get<int>("d"), 42);
    }

    // default left out
    {
      ryml::Tree tree = init_yaml_tree_with_exceptions();
      ryml::NodeRef root = tree.rootref();

      root |= ryml::MAP;
      root["a"] << 1;
      root["b"] |= ryml::SEQ;
      root["b"].append_child() << "b1";
      root["b"].append_child() << "b2";
      ConstYamlNodeRef node(root, "");

      InputParameterContainer container;
      spec.match(node, container);
      EXPECT_EQ(container.get<int>("c"), 1);
      EXPECT_FALSE(container.has_group("group"));
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
          spec.match(node, container), Core::Exception, "Expected entry 'b'");
    }

    // too much input
    {
      ryml::Tree tree = init_yaml_tree_with_exceptions();
      ryml::NodeRef root = tree.rootref();

      root |= ryml::MAP;
      root["a"] << 1;
      root["b"] |= ryml::SEQ;
      root["b"].append_child() << "b1";
      root["b"].append_child() << "b2";
      root["c"] << "c2";
      root["d"] << 42;
      ConstYamlNodeRef node(root, "");

      InputParameterContainer container;
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(spec.match(node, container), Core::Exception,
          "The following parts of the input did not match:\n\n"
          "d: 42");
    }
  }

  TEST(InputSpecTest, MatchYamlOneOf)
  {
    auto spec = one_of({
        all_of({
            entry<int>("a"),
            entry<std::string>("b"),
        }),
        all_of({
            entry<std::string>("a"),
            entry<double>("d"),
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
      ryml::Tree tree = init_yaml_tree_with_exceptions();
      ryml::NodeRef root = tree.rootref();

      root |= ryml::MAP;
      root["a"] << 1;
      root["b"] << "b";
      root["d"] << 2.0;
      ConstYamlNodeRef node(root, "");

      InputParameterContainer container;
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(spec.match(node, container), Core::Exception,
          "[X] Expected exactly one but matched multiple:\n"
          "  [!] Possible match:\n"
          "    {\n"
          "      [ ] Fully matched entry 'a'\n"
          "      [ ] Fully matched entry 'b'\n"
          "    }\n"
          "  [!] Possible match:\n"
          "    {\n"
          "      [ ] Fully matched entry 'a'\n"
          "      [ ] Fully matched entry 'd'\n"
          "    }");
    }
  }

  TEST(InputSpecTest, MatchYamlList)
  {
    auto spec = list("list",
        all_of({
            entry<int>("a"),
            entry<std::string>("b"),
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
        entry<std::filesystem::path>("a"),
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

  TEST(InputSpecTest, MatchYamlNoneable)
  {
    auto spec = all_of({
        entry<Noneable<int>>("i", {.default_value = none<int>}),
        entry<Noneable<std::string>>("s"),
        entry<std::vector<Noneable<double>>>("v", {.size = 3}),
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
      EXPECT_EQ(container.get<Noneable<int>>("i"), 1);
      EXPECT_EQ(container.get<Noneable<std::string>>("s"), "string");
      const auto& v = container.get<std::vector<Noneable<double>>>("v");
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
      EXPECT_EQ(container.get<Noneable<int>>("i"), none<int>);
      EXPECT_EQ(container.get<Noneable<std::string>>("s"), none<std::string>);
      const auto& v = container.get<std::vector<Noneable<double>>>("v");
      EXPECT_EQ(v.size(), 3);
      EXPECT_EQ(v[0], none<double>);
      EXPECT_EQ(v[1], none<double>);
      EXPECT_EQ(v[2], none<double>);
    }
  }

  TEST(InputSpecTest, MatchYamlSizes)
  {
    using ComplicatedType = std::vector<std::map<std::string, std::vector<int>>>;
    auto spec = all_of({
        entry<int>("num"),
        entry<ComplicatedType>("v", {.size = {2, dynamic_size, from_parameter<int>("num")}}),
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
      SCOPED_TRACE("Wrong size from_parameter not validated in yaml.");
      ryml::Tree tree = init_yaml_tree_with_exceptions();
      ryml::parse_in_arena(R"(num: 2
v:
  - key1: [1, 2, 3] # inner most vector is too long but we do not check for this in yaml
    key2: [3, 4]
  - key1: [5, 6])",
          &tree);
      ryml::NodeRef root = tree.rootref();
      ConstYamlNodeRef node(root, "");

      InputParameterContainer container;
      spec.match(node, container);
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
          spec.match(node, container), Core::Exception, "Partially matched entry 'v'");
    }
  }

  TEST(InputSpecTest, MaterialExample)
  {
    auto mat_spec = all_of({
        entry<int>("MAT"),
        one_of({
            group("MAT_A",
                {
                    entry<int>("a"),
                }),
            group("MAT_B",
                {
                    entry<int>("b"),
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
    // This was a bug where a single defaulted entry was incorrectly reported as not matching.
    auto spec = all_of({
        entry<int>("a", {.default_value = 42}),
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

  TEST(InputSpecTest, DatToYaml)
  {
    auto spec = all_of({
        // awkward one_of where the first choice partially matches
        one_of({
            all_of({
                entry<int>("a"),
                entry<std::string>("b"),
            }),
            all_of({
                entry<int>("a"),
                entry<double>("d"),
            }),
        }),
        // group with all defaulted entries
        group("group",
            {
                entry<double>("c", {.default_value = 1.0}),
                selection<int>("d", {{"d1", 1}, {"d2", 2}}, {.default_value = 2}),
            },
            {.required = false}),
        list("list",
            all_of({
                entry<int>("l1"),
                entry<double>("l2"),
            }),
            {.size = 2}),
        entry<int>("i", {.default_value = 0}),
        entry<std::vector<double>>("v", {.size = 3}),
    });

    std::string dat = "a 1 d 3.0 group c 1 d d2 i 42 v 1.0 2.0 3.0 list l1 1 l2 2.0 l1 3 l2 4.0";

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
  d: "d2"
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
}  // namespace

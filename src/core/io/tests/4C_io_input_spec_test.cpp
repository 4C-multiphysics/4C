// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_io_input_spec_builders.hpp"
#include "4C_io_input_spec_builders.templates.hpp"
#include "4C_unittest_utils_assertions_test.hpp"


namespace
{
  using namespace FourC;
  using namespace Core::IO;
  using namespace Core::IO::InputSpecBuilders;

  namespace Helpers
  {
    std::string emit_metadata(const InputSpec& spec, InputSpecEmitMetadataOptions options = {})
    {
      ryml::Tree tree = init_yaml_tree_with_exceptions();
      ryml::NodeRef root = tree.rootref();
      YamlNodeRef yaml(root, "");
      spec.emit_metadata(yaml, options);

      std::string metadata;
      ryml::emitrs_yaml(tree, &metadata);
      return metadata;
    }

    InputParameterContainer match(
        const InputSpec& spec, const std::string& yaml_str, const std::string& filepath = "")
    {
      InputParameterContainer container;
      ryml::Tree tree = init_yaml_tree_with_exceptions();
      ryml::NodeRef root = tree.rootref();
      ryml::parse_in_arena(ryml::to_csubstr(yaml_str), root);

      ConstYamlNodeRef node(root, filepath);
      spec.match(node, container);

      return container;
    }
  }  // namespace Helpers

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
      const std::string metadata = Helpers::emit_metadata(spec);
      EXPECT_EQ(metadata, R"(type: one_of
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
      - name: c
        type: string
        required: true
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

    const std::string metadata = Helpers::emit_metadata(spec);

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
    EXPECT_EQ(metadata, expected);
  }

  TEST(InputSpecTest, NestedOneOfsWithAllOfs)
  {
    auto spec = one_of({
        all_of({
            one_of({
                parameter<int>("a"),
                parameter<int>("b"),
            }),
            one_of({
                parameter<std::string>("c"),
            }),
        }),
        all_of({
            one_of({
                parameter<int>("c"),
                parameter<int>("d"),
            }),
        }),
    });

    const std::string metadata = Helpers::emit_metadata(spec);

    std::string expected = R"(type: one_of
specs:
  - type: all_of
    specs:
      - name: a
        type: int
        required: true
      - name: c
        type: string
        required: true
  - type: all_of
    specs:
      - name: b
        type: int
        required: true
      - name: c
        type: string
        required: true
  - type: all_of
    specs:
      - name: c
        type: int
        required: true
  - type: all_of
    specs:
      - name: d
        type: int
        required: true
)";
    EXPECT_EQ(metadata, expected);
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
        parameter<EnumClass>(
            "e", {.default_value = EnumClass::A,
                     .validator = Validators::in_set({EnumClass::A, EnumClass::C})}),
        parameter<std::optional<EnumClass>>("eo"),
        group("group2",
            {
                parameter<int>("g", {.validator = Validators::positive<int>()}),
            },
            {.required = false}),
        list("list",
            all_of({
                parameter<int>("l1"),
                parameter<double>("l2"),
            }),
            {.size = 2}),
        selection<EnumClass>("selection_group",
            {
                parameter<int>("A"),
                parameter<int>("B"),
                parameter<int>("C"),
            }),
    });


    {
      const std::string metadata = Helpers::emit_metadata(spec);

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
            required: false
            default: A
            choices:
              - name: A
              - name: C
          - name: eo
            noneable: true
            type: enum
            required: false
            default: null
            choices:
              - name: A
              - name: B
              - name: C
          - name: group2
            type: group
            required: false
            specs:
              - type: all_of
                specs:
                  - name: g
                    type: int
                    required: true
                    validator:
                      range:
                        minimum: 0
                        maximum: 2147483647
                        minimum_exclusive: true
                        maximum_exclusive: false
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
            choices:
              - name: A
                spec:
                  name: A
                  type: int
                  required: true
              - name: B
                spec:
                  name: B
                  type: int
                  required: true
              - name: C
                spec:
                  name: C
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
            required: false
            default: A
            choices:
              - name: A
              - name: C
          - name: eo
            noneable: true
            type: enum
            required: false
            default: null
            choices:
              - name: A
              - name: B
              - name: C
          - name: group2
            type: group
            required: false
            specs:
              - type: all_of
                specs:
                  - name: g
                    type: int
                    required: true
                    validator:
                      range:
                        minimum: 0
                        maximum: 2147483647
                        minimum_exclusive: true
                        maximum_exclusive: false
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
            choices:
              - name: A
                spec:
                  name: A
                  type: int
                  required: true
              - name: B
                spec:
                  name: B
                  type: int
                  required: true
              - name: C
                spec:
                  name: C
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
            required: false
            default: A
            choices:
              - name: A
              - name: C
          - name: eo
            noneable: true
            type: enum
            required: false
            default: null
            choices:
              - name: A
              - name: B
              - name: C
          - name: group2
            type: group
            required: false
            specs:
              - type: all_of
                specs:
                  - name: g
                    type: int
                    required: true
                    validator:
                      range:
                        minimum: 0
                        maximum: 2147483647
                        minimum_exclusive: true
                        maximum_exclusive: false
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
            choices:
              - name: A
                spec:
                  name: A
                  type: int
                  required: true
              - name: B
                spec:
                  name: B
                  type: int
                  required: true
              - name: C
                spec:
                  name: C
                  type: int
                  required: true
)";
      EXPECT_EQ(metadata, expected);

      std::cout << metadata << std::endl;
    }
  }


  TEST(InputSpecTest, EmitMetadataCondense)
  {
    InputSpec spec;

    // Let the various building blocks go out of scope
    {
      auto spec_list = list("list",
          all_of({
              parameter<int>("l1"),
              parameter<double>("l2"),
          }),
          {.size = 2});

      auto spec_inner = group("inner", {
                                           parameter<int>("a", {.default_value = 42}),
                                           parameter<double>("b"),
                                           spec_list,
                                       });
      auto spec_outer = group("outer", {spec_inner, spec_list});

      // The outer group is duplicated and should be shared. The inner group is also duplicated
      // but it will not be shared since it is part of the already shared outer group. The list
      // is duplicated and will be referenced on the outer level, but not inside the group, since
      // references may not be nested.
      spec = all_of({
          group("first", {spec_outer}),
          group("second", {spec_outer}),
          spec_list,
      });
    }


    const std::string metadata =
        Helpers::emit_metadata(spec, {.condense_duplicated_specs_threshold = 1});

    const std::string expected = R"(type: all_of
specs:
  - name: first
    type: group
    required: true
    specs:
      - type: all_of
        specs:
          - $ref: "0"
  - name: second
    type: group
    required: true
    specs:
      - type: all_of
        specs:
          - $ref: "0"
  - $ref: "1"
$references:
  "0":
    name: outer
    type: group
    required: true
    specs:
      - type: all_of
        specs:
          - name: inner
            type: group
            required: true
            specs:
              - type: all_of
                specs:
                  - name: a
                    type: int
                    required: false
                    default: 42
                  - name: b
                    type: double
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
  "1":
    name: list
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
)";

    std::cout << metadata << std::endl;
    EXPECT_EQ(metadata, expected);
  }

  TEST(InputSpecTest, Copyable)
  {
    InputSpec spec;
    {
      auto tmp = all_of({
          parameter<int>("a"),
          parameter<std::string>("b"),
      });

      spec = group("group", {
                                tmp,
                                parameter<int>("d"),
                            });
    }

    {
      InputParameterContainer container = Helpers::match(spec, R"(
group:
  a: 1
  b: string
  d: 42
)");

      const auto& group = container.group("group");
      EXPECT_EQ(group.get<int>("a"), 1);
      EXPECT_EQ(group.get<std::string>("b"), "string");
      EXPECT_EQ(group.get<int>("d"), 42);
    }
  }

  TEST(InputSpecTest, MatchYamlParameter)
  {
    auto spec = parameter<int>("a");

    {
      InputParameterContainer container = Helpers::match(spec, R"(
a: 1
)");
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
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(Helpers::match(spec, R"(
b: 1
)"),
          Core::Exception, "Expected parameter 'a'");
    }

    {
      SCOPED_TRACE("Wrong type.");
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(Helpers::match(spec, R"(
a: "string"
)"),
          Core::Exception, "Candidate parameter 'a' has wrong type, expected type: int");
    }
  }

  TEST(InputSpecTest, MatchYamlComplicatedParameterTuplePairMap)
  {
    using ComplicatedType =
        std::tuple<int, std::pair<std::vector<double>, std::map<std::string, bool>>,
            std::tuple<std::string, int, double>>;

    // vector and map are sized to 2 entries each
    const auto spec = parameter<ComplicatedType>("t", {.description = "", .size = {2, 2}});

    const InputParameterContainer container = Helpers::match(spec, R"(
t:
  - 42
  - [ [1.1, 2.2], {a: true, b: false} ]
  - ["hello", 7, 8.9]
)");

    const auto& tuple = container.get<ComplicatedType>("t");

    // first element: int
    EXPECT_EQ(std::get<0>(tuple), 42);

    // second element: pair<vector<double>, map<string,bool>>
    const auto& pair = std::get<1>(tuple);
    const auto& vec = pair.first;
    ASSERT_EQ(vec.size(), 2);
    EXPECT_DOUBLE_EQ(vec[0], 1.1);
    EXPECT_DOUBLE_EQ(vec[1], 2.2);

    const auto& map = pair.second;
    ASSERT_EQ(map.size(), 2);
    EXPECT_EQ(map.at("a"), true);
    EXPECT_EQ(map.at("b"), false);

    // third element: tuple<string,int>
    const auto& inner_tuple = std::get<2>(tuple);
    EXPECT_EQ(std::get<0>(inner_tuple), "hello");
    EXPECT_EQ(std::get<1>(inner_tuple), 7);
  }

  TEST(InputSpecTest, MatchYamlComplicatedParameterPairVectorMap)
  {
    using ComplicatedType =
        std::pair<std::vector<double>, std::pair<std::vector<double>, std::map<std::string, bool>>>;

    // outer vector is sized to 3 entries, inner vector to 2 entries, map to 4 entries
    auto spec = parameter<ComplicatedType>("c", {.description = "", .size = {2, 3, 4}});

    InputParameterContainer container = Helpers::match(spec, R"(
c: [[1.0, 2.0], [[1.0, 2.0, 8.0], {a: true, b: false, c: true, d: false}]]
)");

    const auto& pair = container.get<ComplicatedType>("c");
    ASSERT_EQ(pair.first.size(), 2);
    EXPECT_EQ(pair.first[0], 1.0);
    EXPECT_EQ(pair.first[1], 2.0);

    ASSERT_EQ(pair.second.first.size(), 3);
    EXPECT_EQ(pair.second.first[0], 1.0);
    EXPECT_EQ(pair.second.first[1], 2.0);
    EXPECT_EQ(pair.second.first[2], 8.0);

    const auto& map = pair.second.second;
    ASSERT_EQ(map.size(), 4);
    EXPECT_EQ(map.at("a"), true);
    EXPECT_EQ(map.at("b"), false);
    EXPECT_EQ(map.at("c"), true);
    EXPECT_EQ(map.at("d"), false);
  }

  TEST(InputSpecTest, MatchArray)
  {
    auto spec = parameter<std::array<int, 3>>("a", {});

    {
      SCOPED_TRACE("Matches");
      Helpers::match(spec, R"(
a: [1, 2, 3]
)");
    }

    {
      SCOPED_TRACE("Too few elements");
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(Helpers::match(spec, R"(
a: [1, 2 ]
)"),
          Core::Exception, "Candidate parameter 'a' has incorrect size");
    }

    {
      SCOPED_TRACE("Too many elements");
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(Helpers::match(spec, R"(
a: [1, 2, 3, 4]
)"),
          Core::Exception, "Candidate parameter 'a' has incorrect size");
    }
  }

  TEST(InputSpecTest, MatchYamlVectorOfArraysOfVectorsOfArrays)
  {
    using namespace Core::IO::InputSpecBuilders::Validators;
    using ComplicatedType = std::vector<std::array<std::array<std::vector<int>, 3>, 2>>;

    auto spec = parameter<ComplicatedType>("t",
        {
            .description = "",
            .validator = all_elements(all_elements(all_elements(all_elements(positive<int>())))),
            .size = {2, 3},
        });

    InputParameterContainer container = Helpers::match(spec, R"(
t: [[[[1,1,1],[1,1,2],[1,3,1]], [[2,2,2],[2,2,1],[2,2,9]]], [[[3,3,3],[3,2,3],[3,1,3]], [[4,4,4],[4,4,5],[4,4,6]]]]
)");

    const auto& outer_vector = container.get<ComplicatedType>("t");
    ASSERT_EQ(outer_vector.size(), 2);
    const auto& outer_array = outer_vector[0];
    ASSERT_EQ(outer_array.size(), 2);
    const auto& inner_array = outer_array[1];
    ASSERT_EQ(inner_array.size(), 3);
    const auto& inner_vector = inner_array[2];
    ASSERT_EQ(inner_vector.size(), 3);

    ASSERT_EQ(outer_vector[0][0][0][0], 1);
    ASSERT_EQ(outer_vector[0][1][2][1], 2);
    ASSERT_EQ(outer_vector[1][1][2][2], 6);
  }

  TEST(InputSpecTest, MatchTensor)
  {
    auto spec = parameter<Core::LinAlg::Tensor<int, 3>>("a", {});

    {
      SCOPED_TRACE("Matches");
      Helpers::match(spec, R"(
a: [1, 2, 3]
)");
    }

    {
      SCOPED_TRACE("Too few elements");
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(Helpers::match(spec, R"(
a: [1, 2 ]
)"),
          Core::Exception, "Candidate parameter 'a' has incorrect size");
    }

    {
      SCOPED_TRACE("Too many elements");
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(Helpers::match(spec, R"(
a: [1, 2, 3, 4]
)"),
          Core::Exception, "Candidate parameter 'a' has incorrect size");
    }
  }

  TEST(InputSpecTest, MatchSymmetricTensor)
  {
    auto spec = parameter<Core::LinAlg::SymmetricTensor<int, 2, 2>>("a", {});

    {
      SCOPED_TRACE("Matches");
      Helpers::match(spec, R"(
a: [[1, 2], [2, 3]]
)");
    }

    {
      SCOPED_TRACE("Invalid nonsymmetric match");
      EXPECT_THROW(Helpers::match(spec, R"(
a: [[1, 2], [3, 3]]
)"),
          Core::Exception);
    }
  }

  TEST(InputSpecTest, MatchYamlHigherOrderTensor)
  {
    using namespace Core::IO::InputSpecBuilders::Validators;

    auto spec = parameter<Core::LinAlg::Tensor<int, 2, 2, 3, 3>>("t", {.description = ""});

    InputParameterContainer container = Helpers::match(spec, R"(
t: [[[[1,1,1],[1,1,2],[1,3,1]], [[2,2,2],[2,2,1],[2,2,9]]], [[[3,3,3],[3,2,3],[3,1,3]], [[4,4,4],[4,4,5],[4,4,6]]]]
)");

    const auto& outer_vector = container.get<Core::LinAlg::Tensor<int, 2, 2, 3, 3>>("t");

    ASSERT_EQ(outer_vector(0, 0, 0, 0), 1);
    ASSERT_EQ(outer_vector(0, 1, 2, 1), 2);
    ASSERT_EQ(outer_vector(1, 1, 2, 2), 6);
  }

  TEST(InputSpecTest, MatchYamlSymmetricTensor)
  {
    using namespace Core::IO::InputSpecBuilders::Validators;

    auto spec = parameter<Core::LinAlg::SymmetricTensor<int, 2, 2>>("t", {.description = ""});

    {
      SCOPED_TRACE("Valid Match");
      InputParameterContainer container = Helpers::match(spec, R"(
t: [[1, 2], [2, 3]]
)");

      const auto& outer_vector = container.get<Core::LinAlg::SymmetricTensor<int, 2, 2>>("t");

      ASSERT_EQ(outer_vector(0, 0), 1);
      ASSERT_EQ(outer_vector(0, 1), 2);
      ASSERT_EQ(outer_vector(1, 0), 2);
      ASSERT_EQ(outer_vector(1, 1), 3);
    }
  }

  TEST(InputSpecTest, MatchYamlGroup)
  {
    auto spec = group("group", {
                                   parameter<int>("a"),
                                   parameter<std::string>("b"),
                               });

    {
      {
        SCOPED_TRACE("Match root node.");
        InputParameterContainer container = Helpers::match(spec, R"(
group:
  a: 1
  b: "b"
)");
        EXPECT_EQ(container.group("group").get<int>("a"), 1);
        EXPECT_EQ(container.group("group").get<std::string>("b"), "b");
      }

      {
        SCOPED_TRACE("Top-level match ignores unused.");
        InputParameterContainer container = Helpers::match(spec, R"(
group:
  a: 1
  b: "b"
dummy: 1
)");
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
            group("linear", {parameter<double>("coefficient")}),
            group("quadratic", {one_of({
                                   all_of({
                                       parameter<int>("a"),
                                       parameter<double>("b"),
                                   }),
                                   parameter<double>("c"),
                               })}),
        },
        {.description = "", .store_selector = in_container<Model>("type")});
    {
      SCOPED_TRACE("First selection");
      InputParameterContainer container = Helpers::match(spec, R"(
model:
  linear:
    coefficient: 1.0
)");
      EXPECT_EQ(container.group("model").get<Model>("type"), Model::linear);
      EXPECT_EQ(container.group("model").group("linear").get<double>("coefficient"), 1.0);
    }

    {
      SCOPED_TRACE("Second selection");
      InputParameterContainer container = Helpers::match(spec, R"(
model:
  quadratic:
    a: 1
    b: 2.0
)");
      EXPECT_EQ(container.group("model").get<Model>("type"), Model::quadratic);
      EXPECT_EQ(container.group("model").group("quadratic").get<int>("a"), 1);
      EXPECT_EQ(container.group("model").group("quadratic").get<double>("b"), 2.0);
    }

    {
      SCOPED_TRACE("Second selection, other one_of");
      InputParameterContainer container = Helpers::match(spec, R"(
model:
  quadratic:
    c: 3.0
)");
      EXPECT_EQ(container.group("model").get<Model>("type"), Model::quadratic);
      EXPECT_EQ(container.group("model").group("quadratic").get<double>("c"), 3.0);
    }

    {
      SCOPED_TRACE("Too many keys");
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(Helpers::match(spec, R"(
model:
  type: quadratic
  coefficient: 1
)"),
          Core::Exception, "'model' needs exactly one child with selector value as key");
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
      InputParameterContainer container = Helpers::match(spec, R"(
enum: A
)");
      EXPECT_EQ(container.get<Enum>("enum"), Enum::a);
    }

    {
      SCOPED_TRACE("No match: wrong key");
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(Helpers::match(spec, R"(
this_is_the_wrong_name: A
)"),
          Core::Exception, "Expected deprecated_selection 'enum'");
    }

    {
      SCOPED_TRACE("No match: wrong value");
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(Helpers::match(spec, R"(
enum: wrong_value
)"),
          Core::Exception,
          "Candidate deprecated_selection 'enum' has wrong value, possible values: A|B");
    }
  }


  TEST(InputSpecTest, MatchYamlOneOf)
  {
    auto spec = group("data", {
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
                              });

    {
      InputParameterContainer container = Helpers::match(spec, R"(
data:
  a: 1
  b: b
)");
      const auto& data = container.group("data");
      EXPECT_EQ(data.get<int>("a"), 1);
      EXPECT_EQ(data.get<std::string>("b"), "b");
    }

    {
      InputParameterContainer container = Helpers::match(spec, R"(
data:
  a: 1
  d: 2.0
)");
      const auto& data = container.group("data");
      EXPECT_EQ(data.get<int>("a"), 1);
      EXPECT_EQ(data.get<double>("d"), 2.0);
    };

    {
      SCOPED_TRACE("Multiple possible matches.");
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(Helpers::match(spec, R"(
data:
  a: 1
  b: b
  d: 2
)"),
          Core::Exception, R"([X] Expected one of:
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
      {
        SCOPED_TRACE("Match root node.");
        InputParameterContainer container = Helpers::match(spec, R"(
list:
  - a: 1
    b: "string"
)");
        const auto& list = container.get_list("list");
        EXPECT_EQ(list.size(), 1);
        EXPECT_EQ(list[0].get<int>("a"), 1);
        EXPECT_EQ(list[0].get<std::string>("b"), "string");
      }
    }

    // unmatched node
    {
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(Helpers::match(spec, R"(
list:
  - a: "wrong type"
    b: "string"
  - a: 2
    b: "string2"
)"),
          Core::Exception, "The following list entry did not match:");
    }

    // too many entries
    {
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(Helpers::match(spec, R"(
list:
  - a: 1
    b: "string"
  - a: 2
    b: "string2"
)"),
          Core::Exception, "Too many list entries encountered: expected 1 but matched 2");
    }
  }

  TEST(InputSpecTest, MatchYamlPath)
  {
    auto spec = parameter<std::filesystem::path>("a");

    {
      InputParameterContainer container = Helpers::match(spec, R"(
a: "dir/file.txt"
)",
          "path/to/input.yaml");

      EXPECT_EQ(container.get<std::filesystem::path>("a"), "path/to/dir/file.txt");
    }

    {
      InputParameterContainer container = Helpers::match(spec, R"(
a: "dir/file.txt"
)",
          "input.yaml");

      EXPECT_EQ(container.get<std::filesystem::path>("a"), "dir/file.txt");
    }

    {
      InputParameterContainer container = Helpers::match(spec, R"(
a: "/root/dir/file.txt"
)",
          "path/to/input.yaml");

      EXPECT_EQ(container.get<std::filesystem::path>("a"), "/root/dir/file.txt");
    }
  }

  TEST(InputSpecTest, MatchYamlOptional)
  {
    auto spec = group("data", {
                                  parameter<std::optional<int>>("i"),
                                  parameter<std::optional<std::string>>("s"),
                                  parameter<std::vector<std::optional<double>>>("v", {.size = 3}),
                              });

    {
      InputParameterContainer container = Helpers::match(spec, R"(
data:
  i : 1
  s: string
  v: [1.0, 2.0, 3.0]
)");
      const auto& data = container.group("data");
      EXPECT_EQ(data.get<std::optional<int>>("i"), 1);
      EXPECT_EQ(data.get<std::optional<std::string>>("s"), "string");
      const auto& v = data.get<std::vector<std::optional<double>>>("v");
      EXPECT_EQ(v.size(), 3);
      EXPECT_EQ(v[0], 1.0);
      EXPECT_EQ(v[1], 2.0);
      EXPECT_EQ(v[2], 3.0);
    }

    {
      InputParameterContainer container = Helpers::match(spec, R"(
data:
  i : null
  s: # Note: leaving the key out is the same as setting null
  v: [Null, NULL, ~] # all the other spellings that YAML supports
)");
      const auto& data = container.group("data");
      EXPECT_EQ(data.get<std::optional<int>>("i"), std::nullopt);
      EXPECT_EQ(data.get<std::optional<std::string>>("s"), std::nullopt);
      const auto& v = data.get<std::vector<std::optional<double>>>("v");
      EXPECT_EQ(v.size(), 3);
      EXPECT_EQ(v[0], std::nullopt);
      EXPECT_EQ(v[1], std::nullopt);
      EXPECT_EQ(v[2], std::nullopt);
    }
  }

  TEST(InputSpecTest, MatchYamlSizes)
  {
    using ComplicatedType = std::vector<std::map<std::string,
        std::tuple<std::vector<int>, std::vector<double>, std::pair<std::string, bool>>>>;
    auto spec = group("data", {
                                  parameter<int>("num"),
                                  parameter<ComplicatedType>("v",
                                      {.size = {2, dynamic_size, from_parameter<int>("num"), 1}}),
                              });

    {
      SCOPED_TRACE("Expected sizes");
      InputParameterContainer container = Helpers::match(spec, R"(
data:
  num: 3
  v:
    - key1: [[1, 2, 1], [9.876], [true, true]]
      key2: [[3, 4, 5], [9.876], [true, false]]
    - key1: [[5, 6, 9], [9.876], [false, false]]
)");
      const auto& v = container.group("data").get<ComplicatedType>("v");
      EXPECT_EQ(v.size(), 2);
      EXPECT_EQ(std::get<0>(v[0].at("key1")).size(), 3);
      EXPECT_EQ(std::get<1>(v[0].at("key1")).size(), 1);

      auto pair = std::get<2>(v[0].at("key1"));
      EXPECT_EQ(pair.first, "true");
      EXPECT_EQ(pair.second, true);
    }

    {
      SCOPED_TRACE("Wrong size from_parameter");
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(Helpers::match(spec, R"(
data:
  num: 3
  v:
    - key1: [[1, 2, 3], [9.876], [true, true]]
      key2: [[3, 4, 5], [9.876], [true, false]]
    - key1: [[5, 6], [9.876], [false, false]]
)"),
          Core::Exception, "");  // [5, 6] should be size 3
    }

    {
      SCOPED_TRACE("Wrong size explicitly set for outer vector.");
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(Helpers::match(spec, R"(
data:
  num: 3
  v:
    - key1: [[1, 2, 3], [9.876], [true, true]]
    - key1: [[5, 6, 5], [9.876], [true, false]]
    - key1: [[7, 8, 9], [9.876], [false, false]]
)"),  // v should only have 2 entries
          Core::Exception, "has incorrect size");
    }

    {
      SCOPED_TRACE("Wrong size explicitly set for inner vector.");
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(Helpers::match(spec, R"(
data:
  num: 3
  v:
    - key1: [[1, 2, 3], [9.876], [true, true]]
      key2: [[5, 6, 5], [9.876, 4.244], [true, false]]
    - key1: [[7, 8, 9], [9.876], [false, false]]
)"),  // [9.876, 4.244] should be size 1
          Core::Exception, "has incorrect size");
    }
  }

  TEST(InputSpecTest, MaterialExample)
  {
    auto mat_spec = group("material", {
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
      InputParameterContainer container = Helpers::match(mat_spec, R"(
material:
  MAT: 1
  MAT_A:
    a: 2
)");

      const auto& material = container.group("material");
      EXPECT_EQ(material.get<int>("MAT"), 1);
      EXPECT_EQ(material.group("MAT_A").get<int>("a"), 2);
    }
  }

  TEST(InputSpecTest, EmptyMatchesAllDefaulted)
  {
    // This was a bug where a single defaulted parameter was incorrectly reported as not matching.
    auto spec = parameter<int>("a", {.default_value = 42});

    {
      InputParameterContainer container = Helpers::match(spec, R"(
)");
      EXPECT_EQ(container.get<int>("a"), 42);
    }
  }

  TEST(InputSpecTest, SizedOptionalVector)
  {
    // This was a bug where an optional vector was not parsed correctly.
    auto spec = group("data", {
                                  parameter<int>("num", {.default_value = 2}),
                                  parameter<std::optional<std::vector<double>>>(
                                      "v", {.size = from_parameter<int>("num")}),
                              });

    {
      SCOPED_TRACE("Optional has value");
      InputParameterContainer container = Helpers::match(spec, R"(
data:
  num: 2
  v: [1.0, 2.0]
)");
      const auto& data = container.group("data");
      EXPECT_EQ(data.get<int>("num"), 2);
      const auto& v = data.get<std::optional<std::vector<double>>>("v");
      EXPECT_TRUE(v.has_value());
      EXPECT_EQ(v->size(), 2);
      EXPECT_EQ((*v)[0], 1.0);
      EXPECT_EQ((*v)[1], 2.0);
    }

    {
      SCOPED_TRACE("Empty optional");
      InputParameterContainer container = Helpers::match(spec, R"(
data:
  num: 2
  v: null
)");
      const auto& data = container.group("data");
      EXPECT_EQ(data.get<int>("num"), 2);
      const auto& v = data.get<std::optional<std::vector<double>>>("v");
      EXPECT_FALSE(v.has_value());
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
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(Helpers::match(spec, R"(
parameters:
  start: 0.0
  TimeIntegration:
    OST:
      theta: true # wrong type
    Special:
      type: invalid
)"),
          Core::Exception,
          R"([!] Candidate group 'parameters'
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
)");
    }
    {
      SCOPED_TRACE("Unused parts.");
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(Helpers::match(spec, R"(
data:
  a: 1
parameters:
  start: 0.0
  unused: "abc"
  TimeIntegration:
    OST:
      theta: 0.5
    Special:
)"),
          Core::Exception,
          R"([!] Candidate group 'parameters'
  {
    [ ] Matched parameter 'start'
    [ ] Defaulted parameter 'write_output'
    [!] Candidate group 'TimeIntegration'
      {
        [X] Expected one of:
          {
            [ ] Matched group 'OST'
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
)");
    }
  }

  TEST(InputSpecTest, ParameterValidation)
  {
    using namespace Core::IO::InputSpecBuilders::Validators;
    auto spec = group("parameters",
        {
            parameter<int>("a", {.default_value = 42, .validator = in_range(0, 50)}),
            parameter<std::optional<double>>("b", {.validator = null_or(positive<double>())}),
        });

    {
      SCOPED_TRACE("Valid input");
      InputParameterContainer container = Helpers::match(spec, R"(
parameters:
  a: 1
  b: 2.0
)");
      const auto& parameters = container.group("parameters");
      EXPECT_EQ(parameters.get<int>("a"), 1);
      EXPECT_EQ(*parameters.get<std::optional<double>>("b"), 2.0);
    }

    {
      SCOPED_TRACE("Valid input with defaulted parameter");
      InputParameterContainer container = Helpers::match(spec, R"(
parameters:
  a: 1
)");
      const auto& parameters = container.group("parameters");
      EXPECT_EQ(parameters.get<int>("a"), 1);
      EXPECT_FALSE(parameters.get<std::optional<double>>("b").has_value());
    }

    {
      SCOPED_TRACE("Validation failure");
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(Helpers::match(spec, R"(
parameters:
  a: -1
  b: 0.0
)"),
          Core::Exception, R"(
    [!] Candidate parameter 'a' does not pass validation: in_range[0,50]
    [!] Candidate parameter 'b' does not pass validation: null_or{in_range(0,1.7976931348623157e+308]}
)");
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
        "Parameter 'a' has a default value that does not pass the given validation: "
        "in_range(0,10]");
  }

  TEST(InputSpecTest, OptionalParameterValidation)
  {
    using namespace Core::IO::InputSpecBuilders::Validators;
    auto spec = parameter<std::optional<int>>("a", {.validator = null_or(in_range(0, 10))});

    {
      SCOPED_TRACE("Valid input");
      InputParameterContainer container = Helpers::match(spec, R"(
a: 5
)");
      EXPECT_EQ(container.get<std::optional<int>>("a"), 5);
    }

    {
      SCOPED_TRACE("Invalid input");
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(Helpers::match(spec, R"(
a: 15
)"),
          Core::Exception,
          "Candidate parameter 'a' does not pass validation: null_or{in_range[0,10]}");
    }
  }

  TEST(InputSpecTest, OptionalEnum)
  {
    enum class EnumClass
    {
      A,
      B,
    };

    auto spec = parameter<std::optional<EnumClass>>("e", {});
    {
      InputParameterContainer container = Helpers::match(spec, R"(
e: A
)");
      EXPECT_EQ(container.get<std::optional<EnumClass>>("e"), EnumClass::A);
    }
  }

  TEST(InputSpecTest, EnumValidatorAndDescriptionMetadata)
  {
    enum Enum
    {
      allowed,
      not_allowed,
    };

    const auto spec = parameter<Enum>("e",
        {
            .description = "This is an enum parameter",
            .enum_value_description = [](Enum e) -> std::string
            { return e == Enum::allowed ? "allowed value" : "not allowed value"; },
            .validator = Validators::in_set({Enum::allowed}),
        });

    {
      SCOPED_TRACE("Only print description for validator values");
      const auto metadata = Helpers::emit_metadata(spec);
      EXPECT_EQ(metadata, R"(name: e
type: enum
description: "This is an enum parameter"
required: true
choices:
  - name: allowed
    description: "allowed value"
)");
    }
  }

  TEST(InputSpecTest, OptionalParameterValidationComplex)
  {
    using namespace Core::IO::InputSpecBuilders::Validators;
    using Type = std::optional<std::vector<std::optional<int>>>;
    auto spec =
        parameter<Type>("v", {
                                 .validator = null_or(all_elements(null_or(in_range(0, 10)))),
                                 .size = 3,
                             });

    {
      SCOPED_TRACE("Valid input");
      InputParameterContainer container = Helpers::match(spec, R"(
v: [null, 2, 3]
)");
      const auto& v = container.get<Type>("v");
      EXPECT_TRUE(v.has_value());
      EXPECT_EQ(v->size(), 3);
      EXPECT_EQ((*v)[0], std::nullopt);
      EXPECT_EQ((*v)[1], 2);
      EXPECT_EQ((*v)[2], 3);
    }

    {
      SCOPED_TRACE("Invalid input");
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(Helpers::match(spec, R"(
v: [-1, null, 4]
)"),
          Core::Exception,
          "Candidate parameter 'v' does not pass validation: "
          "null_or{all_elements{null_or{in_range[0,10]}}}");
    }
  }

  TEST(InputSpecTest, OneOfOverlappingOptionsSingleParameter)
  {
    // This is a tricky case, where one_of the choices is a single parameter
    const auto spec = group("data", {
                                        one_of({parameter<int>("a"), all_of({
                                                                         parameter<int>("a"),
                                                                         parameter<int>("b"),
                                                                     })}),
                                    });

    {
      SCOPED_TRACE("Overlapping values.");
      InputParameterContainer container = Helpers::match(spec, R"(
data:
  a: 1
  b: 2
)");
      const auto& data = container.group("data");
      EXPECT_EQ(data.get<int>("a"), 1);
      EXPECT_EQ(data.get<int>("b"), 2);
    }
  }

  TEST(InputSpecTest, OneOfOverlappingOptionsSafeAllOf)
  {
    // This case is similar to the previous one, but already uses all_ofs.
    const auto spec = group("data", {
                                        one_of({
                                            all_of({
                                                parameter<int>("a"),
                                                parameter<int>("b"),
                                            }),
                                            all_of({
                                                parameter<int>("a"),
                                                parameter<int>("b"),
                                                parameter<int>("c"),
                                            }),
                                        }),
                                    });
    InputParameterContainer container = Helpers::match(spec, R"(
data:
  a: 1
  b: 2
  c: 3
)");
    const auto& data = container.group("data");
    EXPECT_EQ(data.get<int>("a"), 1);
    EXPECT_EQ(data.get<int>("b"), 2);
    EXPECT_EQ(data.get<int>("c"), 3);
  }



  TEST(InputSpecTest, StoreStruct)
  {
    enum class Option
    {
      a,
      b,
    };

    struct Inner
    {
      int a;
      Option option;
      std::string s;
      bool b_defaulted;
      std::vector<double> v;
    };

    struct Outer
    {
      double d;
      Inner inner;
    };

    auto spec = group<Outer>("outer",
        {
            parameter<double>("d", {.store = in_struct(&Outer::d)}),
            group<Inner>("inner",
                {
                    parameter<int>("a", {.store = in_struct(&Inner::a)}),
                    parameter<Option>("option", {.store = in_struct(&Inner::option)}),
                    deprecated_selection<std::string>(
                        "s", {"abc", "def"}, {.store = in_struct(&Inner::s)}),
                    parameter<bool>("b_defaulted",
                        {.default_value = true, .store = in_struct(&Inner::b_defaulted)}),
                    parameter<std::vector<double>>("v", {.store = in_struct(&Inner::v), .size = 3}),
                },
                {.store = in_struct(&Outer::inner)}),
        });

    const InputParameterContainer container = Helpers::match(spec, R"(
outer:
  d: 1.23
  inner:
    option: b
    a: 1
    s: "abc"
    v: [1.0, 2.0, 3.0])");

    const auto& outer = container.get<Outer>("outer");
    EXPECT_EQ(outer.d, 1.23);
    EXPECT_EQ(outer.inner.a, 1);
    EXPECT_EQ(outer.inner.option, Option::b);
    EXPECT_EQ(outer.inner.s, "abc");
    EXPECT_EQ(outer.inner.b_defaulted, true);
    EXPECT_EQ(outer.inner.v, (std::vector{1.0, 2.0, 3.0}));
  }

  TEST(InputSpecTest, StoreStructRejectInconsistent)
  {
    struct S
    {
      int a;
      int b;
    };

    struct Other
    {
    };

    {
      SCOPED_TRACE("Inconsistent fields in group");
      const auto construct = []()
      {
        auto spec = group<S>("inconsistent",
            {
                parameter<int>("a", {.store = in_struct(&S::a)}),
                // here we "forgot" to specify the .store for b and want to receive an error
                parameter<int>("b"),
            });
      };

      FOUR_C_EXPECT_THROW_WITH_MESSAGE(construct(), Core::Exception,
          "All specs in an all_of must store to the same destination type.");
      // There is more detailed output but the type names may not be stable across compilers.
    }

    {
      SCOPED_TRACE("Wrong struct in group");
      const auto construct = []()
      {
        auto spec =
            group<Other>("inconsistent", {
                                             parameter<int>("a", {.store = in_struct(&S::a)}),
                                             parameter<int>("b", {.store = in_struct(&S::b)}),
                                         });
      };

      FOUR_C_EXPECT_THROW_WITH_MESSAGE(
          construct(), Core::Exception, "contains specs that store to");
    }

    {
      SCOPED_TRACE("Wrong default storage in group");
      const auto construct = []()
      {
        auto spec = group("inconsistent", {
                                              parameter<int>("a", {.store = in_struct(&S::a)}),
                                              parameter<int>("b", {.store = in_struct(&S::b)}),
                                          });
      };
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(
          construct(), Core::Exception, "contains specs that store to");
    }

    {
      SCOPED_TRACE("Top-level group");

      // Construction of this spec is fine because one could continue to use this spec inside
      // a group, but it is not allowed to match it directly.
      auto spec = parameter<int>("a", {.store = in_struct(&S::a)});

      // But matching should not work because the spec does not store to InputParameterContainer
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(Helpers::match(spec, ""), Core::Exception,
          "the top-level InputSpec that is used for matching must store to the "
          "InputParameterContainer type");
    }
  }

  TEST(InputSpecTest, StoreStructWithDefaulted)
  {
    struct S
    {
      int a;
      int b;
      std::string s;
    };

    auto spec = group<S>("s",
        {
            parameter<int>("a", {.default_value = 1, .store = in_struct(&S::a)}),
            parameter<int>("b", {.default_value = 2, .store = in_struct(&S::b)}),
            parameter<std::string>("s", {.default_value = "default", .store = in_struct(&S::s)}),
        },
        {.required = false});

    ryml::Tree tree = init_yaml_tree_with_exceptions();
    ryml::NodeRef root = tree.rootref();

    ConstYamlNodeRef node(root, "");
    InputParameterContainer container;
    spec.match(node, container);

    const auto& s = container.get<S>("s");
    EXPECT_EQ(s.a, 1);
    EXPECT_EQ(s.b, 2);
    EXPECT_EQ(s.s, "default");
  }

  TEST(InputSpecTest, StoreSelectionStructsInContainer)
  {
    enum class Options
    {
      a,
      b,
    };

    struct A
    {
      int a;
      std::string s;
    };

    struct B
    {
      double b;
      bool flag;
    };

    auto spec = selection<Options>(
        "model", {
                     group<A>("a",
                         {
                             parameter<int>("a", {.store = in_struct(&A::a)}),
                             parameter<std::string>("s", {.store = in_struct(&A::s)}),
                         }),
                     group<B>("b",
                         {
                             parameter<double>("b", {.store = in_struct(&B::b)}),
                             parameter<bool>("flag", {.store = in_struct(&B::flag)}),
                         }),
                 });

    InputParameterContainer container = Helpers::match(spec, R"(model:
  a:
    a: 1
    s: abc)");

    const auto& model = container.group("model");
    EXPECT_EQ(model.get<Options>("_selector"), Options::a);
    const auto& a = model.get<A>("a");
    EXPECT_EQ(a.a, 1);
    EXPECT_EQ(a.s, "abc");
    EXPECT_FALSE(model.has_group("b"));
    EXPECT_FALSE(model.get_if<B>("b"));
  }

  TEST(InputSpecTest, ListWithExplicitStore)
  {
    struct Item
    {
      int value;
    };

    struct Parameters
    {
      std::vector<Item> items;
    };

    // A list that collects its entries in an InputParameterContainer::List, but post-processes
    // that list into a struct member via an explicit store function, instead of nesting it under
    // its own name.
    const auto store_items = StoreFunction<InputParameterContainer::List>(
        [](Storage& storage, InputParameterContainer::List&& value)
        {
          auto& target = std::any_cast<Parameters&>(storage).items;
          for (const auto& entry : value) target.push_back(Item{entry.get<int>("value")});
          return StoreStatus::ok();
        },
        typeid(Parameters));

    auto spec = group<Parameters>(
        "parameters", {
                          list("items", parameter<int>("value"), {.store = store_items}),
                      });

    const InputParameterContainer container = Helpers::match(spec, R"(
parameters:
  items:
    - value: 1
    - value: 2
    - value: 3)");

    const auto& items = container.get<Parameters>("parameters").items;
    ASSERT_EQ(items.size(), 3u);
    EXPECT_EQ(items[0].value, 1);
    EXPECT_EQ(items[1].value, 2);
    EXPECT_EQ(items[2].value, 3);
  }

  TEST(InputSpecTest, ListWithExplicitStoreUsesDefault)
  {
    struct Item
    {
      int value;
    };

    struct Parameters
    {
      std::vector<Item> items;
    };

    // A list is only defaultable if it has a fixed size and its entries have a default value.
    // Whenever the enclosing group is then absent from the input, the default list must reach
    // the struct member through the explicit store function, not just through
    // InputParameterContainer::add_list().
    const auto store_items = StoreFunction<InputParameterContainer::List>(
        [](Storage& storage, InputParameterContainer::List&& value)
        {
          auto& target = std::any_cast<Parameters&>(storage).items;
          for (const auto& entry : value) target.push_back(Item{entry.get<int>("value")});
          return StoreStatus::ok();
        },
        typeid(Parameters));

    auto spec = group<Parameters>("parameters",
        {
            list("items", parameter<int>("value", {.default_value = 42}),
                {.size = 2, .store = store_items}),
        },
        {.required = false});

    ryml::Tree tree = init_yaml_tree_with_exceptions();
    ryml::NodeRef root = tree.rootref();

    ConstYamlNodeRef node(root, "");
    InputParameterContainer container;
    spec.match(node, container);

    const auto& items = container.get<Parameters>("parameters").items;
    ASSERT_EQ(items.size(), 2u);
    EXPECT_EQ(items[0].value, 42);
    EXPECT_EQ(items[1].value, 42);
  }

  TEST(InputSpecTest, StoreSelectionStructsInStruct)
  {
    enum class Options
    {
      a,
      b,
    };

    struct A
    {
      int a;
      std::string s;
    };

    struct B
    {
      double b;
      bool flag;
    };

    struct Model
    {
      Options type;
      std::variant<A, B> model;
    };

    struct Parameters
    {
      Model model;
    };

    auto model_a = group<A>("a",
        {
            parameter<int>("a", {.store = in_struct(&A::a)}),
            parameter<std::string>("s", {.store = in_struct(&A::s)}),
        },
        {.store = as_variant<A>(&Model::model)});

    auto model_b = group<B>("b",
        {
            parameter<double>("b", {.store = in_struct(&B::b)}),
            parameter<bool>("flag", {.store = in_struct(&B::flag)}),
        },
        {.store = as_variant<B>(&Model::model)});

    auto spec =
        group<Parameters>("parameters", {
                                            selection<Options, Model>("model",
                                                {
                                                    model_a,
                                                    model_b,

                                                },
                                                {
                                                    .store = in_struct(&Parameters::model),
                                                    .store_selector = in_struct(&Model::type),
                                                }),
                                        });

    InputParameterContainer container = Helpers::match(spec, R"(
parameters:
  model:
    a:
      a: 1
      s: abc)");

    const auto& model = container.get<Parameters>("parameters").model;
    EXPECT_EQ(model.type, Options::a);
    const auto& a = std::get<A>(model.model);
    EXPECT_EQ(a.a, 1);
    EXPECT_EQ(a.s, "abc");
  }

  TEST(InputSpecTest, QuotedStrings)
  {
    auto spec = group("test", {
                                  parameter<std::string>("a"),
                                  parameter<std::string>("b"),
                                  parameter<std::string>("c"),
                                  parameter<std::string>("d"),
                                  parameter<std::string>("e"),
                                  parameter<std::string>("f"),
                              });

    Helpers::match(spec, R"(
test:
  a: "double-quoted string"
  b: 'single quoted string'
  c: "123"
  d: "null"
  e: '1.23'
  f: "true"
)");
  }

  TEST(InputSpecTest, QuoutedStringDoNotMatchOtherTypes)
  {
    auto spec = group("test", {
                                  parameter<bool>("a"),
                                  parameter<int>("b"),
                                  parameter<double>("c"),
                                  parameter<std::optional<int>>("d"),
                              });

    // Matching should fail because the quoted strings do not match the expected types.
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(Helpers::match(spec, R"(
test:
  a: "true"
  b: "123"
  c: "1.23"
  d: "null"
)"),
        Core::Exception, R"([!] Candidate group 'test'
  {
    [!] Candidate parameter 'a' has wrong type, expected type: bool
    [!] Candidate parameter 'b' has wrong type, expected type: int
    [!] Candidate parameter 'c' has wrong type, expected type: double
    [!] Candidate parameter 'd' has wrong type, expected type: std::optional<int>
  }
)");
  }

  TEST(InputSpecSymbolicExpression, StoreInContainer)
  {
    auto spec = group("test", {
                                  symbolic_expression<double, "x", "y">("expr"),
                              });

    {
      SCOPED_TRACE("OK");
      InputParameterContainer container = Helpers::match(spec, R"(
test:
  expr: "x + y * 2.0"
)");
      const auto& expr =
          container.group("test").get<Core::Utils::SymbolicExpression<double, "x", "y">>("expr");
      EXPECT_DOUBLE_EQ(expr.value(Core::Utils::var<"x">(1.0), Core::Utils::var<"y">(2.0)), 5.0);
    }

    {
      SCOPED_TRACE("Wrong variables.");
      FOUR_C_EXPECT_THROW_WITH_MESSAGE(Helpers::match(spec, R"(
test:
  expr: "x + y * 2.0 + z" # z is not defined in the expression
)"),
          Core::Exception, R"([!] Candidate group 'test'
  {
    [!] Candidate parameter 'expr' could not be parsed as symbolic expression with variables: "x" "y" 
  }
)");
    }
  }

  TEST(InputSpecSymbolicExpression, StoreInStruct)
  {
    struct S
    {
      Core::Utils::SymbolicExpression<double, "x", "y"> expr;
    };

    auto spec = group<S>(
        "test", {
                    symbolic_expression<double, "x", "y">("expr", {.store = in_struct(&S::expr)}),
                });

    InputParameterContainer container = Helpers::match(spec, R"(
test:
  expr: "x + y * 2.0"
)");
    const auto& expr = container.get<S>("test").expr;
    EXPECT_DOUBLE_EQ(expr.value(Core::Utils::var<"x">(1.0), Core::Utils::var<"y">(2.0)), 5.0);
  }
}  // namespace

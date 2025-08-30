// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_io_input_spec_builders.hpp"

#include <benchmark/benchmark.h>

namespace
{

  using namespace FourC;
  using namespace Core::IO;
  using namespace Core::IO::InputSpecBuilders;

  enum Selector
  {
    a,
    b,
    c,
  };

  static void match_spec(benchmark::State& state)
  {
    auto spec = group("TestGroup",
        {
            parameter<int>("int_param", {}),
            parameter<double>("double_param", {}),
            parameter<std::string>("string_param", {}),
            parameter<std::vector<int>>("vector_param", {.default_value = std::vector{1, 2, 3}}),
            group("NestedGroup",
                {
                    selection<Selector>("selection_param",
                        {
                            parameter<int>("a", {}),
                            parameter<double>("b", {}),
                            group("c",
                                {
                                    parameter<int>("e"),
                                    parameter<double>("f"),
                                }),
                        }),
                }),
        });

    auto tree = init_yaml_tree_with_exceptions();
    auto root = tree.rootref();
    ryml::parse_in_arena(R"(
TestGroup:
  int_param: 42
  double_param: 3.14
  string_param: "Hello, World!"
  NestedGroup:
    selection_param:
      c:
        e: 1
        f: 2.0
)",
        root);

    ConstYamlNodeRef yaml(root, "");

    for (auto _ : state)
    {
      InputParameterContainer container;
      spec.match(yaml, container);
    }
  }
  BENCHMARK(match_spec);

  // Same test but match into a struct
  static void match_spec_struct(benchmark::State& state)
  {
    struct C
    {
      int e;
      double f;
    };

    struct TestGroup
    {
      int int_param;
      double double_param;
      std::string string_param;
      std::vector<int> vector_param;
      struct NestedGroup
      {
        struct SelectionParam
        {
          std::variant<int, double, C> options;
        } selection_param;
      } nested_group;
    };
    auto spec = group<TestGroup>("TestGroup",
        {
            parameter<int>("int_param", {.store = in_struct(&TestGroup::int_param)}),
            parameter<double>("double_param", {.store = in_struct(&TestGroup::double_param)}),
            parameter<std::string>("string_param", {.store = in_struct(&TestGroup::string_param)}),
            parameter<std::vector<int>>(
                "vector_param", {.default_value = std::vector{1, 2, 3},
                                    .store = in_struct(&TestGroup::vector_param)}),
            group<TestGroup::NestedGroup>("NestedGroup",
                {
                    selection<Selector, TestGroup::NestedGroup::SelectionParam>("selection_param",
                        {
                            parameter<int>(
                                "a", {.store = as_variant<int>(
                                          &TestGroup::NestedGroup::SelectionParam::options)}),
                            parameter<double>(
                                "b", {.store = as_variant<double>(
                                          &TestGroup::NestedGroup::SelectionParam::options)}),
                            group<C>("c",
                                {
                                    parameter<int>("e", {.store = in_struct(&C::e)}),
                                    parameter<double>("f", {.store = in_struct(&C::f)}),
                                },
                                {.store = as_variant<C>(
                                     &TestGroup::NestedGroup::SelectionParam::options)}),
                        },
                        {.store = in_struct(&TestGroup::NestedGroup::selection_param)}),
                },
                {.store = in_struct(&TestGroup::nested_group)}),
        });

    auto tree = init_yaml_tree_with_exceptions();
    auto root = tree.rootref();
    ryml::parse_in_arena(R"(
TestGroup:
  int_param: 42
  double_param: 3.14
  string_param: "Hello, World!"
  NestedGroup:
    selection_param:
      c:
        e: 1
        f: 2.0
)",
        root);

    ConstYamlNodeRef yaml(root, "");

    for (auto _ : state)
    {
      InputParameterContainer container;
      spec.match(yaml, container);
    }
  }
  BENCHMARK(match_spec_struct);

}  // namespace

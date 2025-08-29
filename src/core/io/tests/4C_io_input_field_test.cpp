// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_io_input_field.hpp"

#include "4C_io_input_file.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_io_value_parser.hpp"
#include "4C_io_yaml.hpp"
#include "4C_legacy_enum_definitions_conditions.hpp"
#include "4C_unittest_utils_assertions_test.hpp"
#include "4C_unittest_utils_support_files_test.hpp"
#include "4C_utils_singleton_owner.hpp"

#include <iostream>
#include <map>
#include <sstream>

namespace
{
  using namespace FourC;
  using namespace Core::IO;
  using namespace Core::IO::InputSpecBuilders;

  TEST(InputField, ReadJsonScalarInputField)
  {
    const std::string input_field_file =
        TESTING::get_support_file_path("test_files/input_field/stiffness_input_field.json");
    std::unordered_map<int, double> stiffness_map;
    read_value_from_yaml(input_field_file, "stiffness", stiffness_map);
    std::unordered_map<int, double> expected_stiffness_map = {
        {1, 2.0}, {2, 3.5}, {3, 4.0}, {4, 5.5}};
    EXPECT_EQ(stiffness_map, expected_stiffness_map);
  }

  TEST(InputField, ReadSpecScalarInputField)
  {
    const std::string input_field_file =
        TESTING::get_support_file_path("test_files/input_field/stiffness_input_field.json");
    auto spec = input_field<double>("stiffness", {.description = "A stiffness field"});

    {
      SCOPED_TRACE("Constant scalar input field");
      ryml::Tree tree = init_yaml_tree_with_exceptions();
      ryml::NodeRef root = tree.rootref();
      ryml::parse_in_arena(R"(stiffness:
              constant: 1.0)",
          root);

      ConstYamlNodeRef node(root, "");
      InputParameterContainer container;
      spec.match(node, container);
      InputField<double> input_field_stiffness = container.get<InputField<double>>("stiffness");
      EXPECT_EQ(input_field_stiffness.at(1), 1.0);
    }

    {
      SCOPED_TRACE("Scalar input field from file");
      ryml::Tree tree = init_yaml_tree_with_exceptions();
      ryml::NodeRef root = tree.rootref();
      ryml::parse_in_arena(("stiffness:\n    from_file: " + input_field_file).c_str(), root);
      ConstYamlNodeRef node(root, "");
      InputParameterContainer container;
      spec.match(node, container);
      InputField<double> input_field_stiffness = container.get<InputField<double>>("stiffness");
      EXPECT_EQ(input_field_stiffness.at(0), 2.0);
      EXPECT_EQ(input_field_stiffness.at(1), 3.5);
      EXPECT_EQ(input_field_stiffness.at(2), 4.0);
      EXPECT_EQ(input_field_stiffness.at(3), 5.5);
    }
  }

  TEST(InputField, ReadJsonVectorInputField)
  {
    const std::string input_field_file =
        TESTING::get_support_file_path("test_files/input_field/conductivity_input_field.json");
    std::unordered_map<int, std::vector<double>> conductivity_map;
    read_value_from_yaml(input_field_file, "CONDUCT", conductivity_map);
    std::unordered_map<int, std::vector<double>> expected_conductivity_map = {
        {1, {1.0, 2.0, 3.0}}, {2, {3.0, 2.0, 1.0}}, {3, {1.0, 2.0, 3.0}}, {4, {3.0, 2.0, 1.0}}};
    EXPECT_EQ(conductivity_map, expected_conductivity_map);
  }

  TEST(InputField, ReadSpecVectorInputField)
  {
    const std::string input_field_file =
        TESTING::get_support_file_path("test_files/input_field/conductivity_input_field.json");
    auto spec =
        input_field<std::vector<double>>("CONDUCT", {.description = "A conductivity field"});

    {
      SCOPED_TRACE("Constant vector input field");
      ryml::Tree tree = init_yaml_tree_with_exceptions();
      ryml::NodeRef root = tree.rootref();
      ryml::parse_in_arena(R"(CONDUCT:
              constant: [1.0, 2.0, 3.0])",
          root);
      ConstYamlNodeRef node(root, "");
      InputParameterContainer container;
      spec.match(node, container);
      auto input_field_conductivity = container.get<InputField<std::vector<double>>>("CONDUCT");
      std::vector<double> expected_conductivity{1.0, 2.0, 3.0};
      EXPECT_EQ(input_field_conductivity.at(0), expected_conductivity);
    }

    {
      SCOPED_TRACE("Vector input field from file");
      ryml::Tree tree = init_yaml_tree_with_exceptions();
      ryml::NodeRef root = tree.rootref();
      ryml::parse_in_arena(("CONDUCT:\n    from_file: " + input_field_file).c_str(), root);
      ConstYamlNodeRef node(root, "");
      InputParameterContainer container;
      spec.match(node, container);
      auto input_field_conductivity = container.get<InputField<std::vector<double>>>("CONDUCT");
      std::vector<std::vector<double>> expected_conductivity{
          {1.0, 2.0, 3.0}, {3.0, 2.0, 1.0}, {1.0, 2.0, 3.0}, {3.0, 2.0, 1.0}};
      EXPECT_EQ(input_field_conductivity.at(0), expected_conductivity[0]);
      EXPECT_EQ(input_field_conductivity.at(1), expected_conductivity[1]);
      EXPECT_EQ(input_field_conductivity.at(2), expected_conductivity[2]);
      EXPECT_EQ(input_field_conductivity.at(3), expected_conductivity[3]);
    }
  }

  TEST(InputField, ReadSpecToStruct)
  {
    struct Data
    {
      InputField<double> stiffness;
    };

    auto spec = group<Data>(
        "data", {
                    input_field<double>("stiffness",
                        {.description = "A stiffness field", .store = in_struct(&Data::stiffness)}),
                });
    {
      SCOPED_TRACE("Constant input field");
      ryml::Tree tree = init_yaml_tree_with_exceptions();
      ryml::NodeRef root = tree.rootref();
      ryml::parse_in_arena(R"(data:
  stiffness:
    constant: 1.0)",
          root);

      ConstYamlNodeRef node(root, "");
      InputParameterContainer container;
      spec.match(node, container);
      const auto& data = container.get<Data>("data");
      EXPECT_EQ(data.stiffness.at(0), 1.0);
    }

    {
      SCOPED_TRACE("Input field from file");
      ryml::Tree tree = init_yaml_tree_with_exceptions();
      ryml::NodeRef root = tree.rootref();
      const std::string input_field_file =
          TESTING::get_support_file_path("test_files/input_field/stiffness_input_field.json");
      ryml::parse_in_arena(
          ("data:\n  stiffness:\n    from_file: " + input_field_file).c_str(), root);

      ConstYamlNodeRef node(root, "");
      InputParameterContainer container;
      spec.match(node, container);
      const auto& data = container.get<Data>("data");
      EXPECT_EQ(data.stiffness.at(0), 2.0);
      EXPECT_EQ(data.stiffness.at(1), 3.5);
      EXPECT_EQ(data.stiffness.at(2), 4.0);
      EXPECT_EQ(data.stiffness.at(3), 5.5);
    }
  }
}  // namespace

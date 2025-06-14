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

#include <iostream>
#include <map>
#include <sstream>
namespace
{
  using namespace FourC;
  using namespace FourC::Core::IO;
  using namespace FourC::Core::IO::InputSpecBuilders;

  TEST(InputFile, ReadJsonInputField)
  {
    const std::string input_field_file =
        TESTING::get_support_file_path("test_files/input_field/stiffness_input_field.json");
    std::unordered_map<int, double> stiffness_map;
    read_value_from_yaml(input_field_file, "stiffness", stiffness_map);
    std::unordered_map<int, double> expected_stiffness_map = {
        {1, 2.0}, {2, 3.5}, {3, 4.0}, {4, 5.5}};
    EXPECT_EQ(stiffness_map, expected_stiffness_map);
  }
  TEST(InputFile, ReadSpecInputField)
  {
    const std::string input_field_file =
        TESTING::get_support_file_path("test_files/input_field/stiffness_input_field.json");
    auto spec = input_field<double>("stiffness", {.description = "A stiffness field"});
    {
      SCOPED_TRACE("Constant input field");
      ryml::Tree tree = init_yaml_tree_with_exceptions();
      ryml::NodeRef root = tree.rootref();
      ryml::parse_in_arena(R"(stiffness:
              type: constant
              value: 1.0)",
          root);

      ConstYamlNodeRef node(root, "");
      InputParameterContainer container;
      spec.match(node, container);
      InputField<double> input_field_stiffness =
          container.group("stiffness").get<InputField<double>>("stiffness");
      EXPECT_EQ(input_field_stiffness.at(1), 1.0);
    }
    {
      SCOPED_TRACE("Input field from file");
      ryml::Tree tree = init_yaml_tree_with_exceptions();
      ryml::NodeRef root = tree.rootref();
      ryml::parse_in_arena(
          ("stiffness:\n    type: from_file\n    value: " + input_field_file).c_str(), root);
      std::flush(std::cout);
      ConstYamlNodeRef node(root, "");
      InputParameterContainer container;
      spec.match(node, container);
      InputField<double> input_field_stiffness =
          container.group("stiffness").get<InputField<double>>("stiffness");
      EXPECT_EQ(input_field_stiffness.at(1), 2.0);
      EXPECT_EQ(input_field_stiffness.at(2), 3.5);
      EXPECT_EQ(input_field_stiffness.at(3), 4.0);
      EXPECT_EQ(input_field_stiffness.at(4), 5.5);
    }
  }
}  // namespace

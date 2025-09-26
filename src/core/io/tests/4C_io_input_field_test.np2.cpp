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

  TEST(InputField, ReadFieldRegistry)
  {
    Core::Utils::SingletonOwnerRegistry::ScopeGuard guard;
    const std::string input_field_file =
        TESTING::get_support_file_path("test_files/input_field/conductivity_input_field.json");
    auto spec =
        input_field<std::vector<double>>("CONDUCT", {.description = "A conductivity field"});

    {
      SCOPED_TRACE("Vector input field from file");
      ryml::Tree tree = init_yaml_tree_with_exceptions();
      ryml::NodeRef root = tree.rootref();
      ryml::parse_in_arena("CONDUCT:\n    field_reference: conduct_ref_name", root);

      ConstYamlNodeRef node(root, "");
      InputParameterContainer container;
      spec.match(node, container);

      // After reading we need to post-process what is in the registry.
      auto& registry = global_input_field_registry();
      EXPECT_TRUE(registry.fields.contains("conduct_ref_name"));
      for (auto& [ref_name, data] : registry.fields)
      {
        if (ref_name == "conduct_ref_name")
        {
          EXPECT_FALSE(data.redistribute_functions.empty());

          // This needs to be done by some global input post-processing step.
          // The information can come from a section that defines all the fields.
          if (Core::Communication::my_mpi_rank(MPI_COMM_WORLD) == 0)
          {
            data.init_functions.begin()->second(input_field_file, "CONDUCT");
          }

          auto& redistribute_function = data.redistribute_functions.begin()->second;

          ASSERT_EQ(Core::Communication::num_mpi_ranks(MPI_COMM_WORLD), 2);
          Core::LinAlg::Map target_map(4, 2, 0, MPI_COMM_WORLD);
          redistribute_function(target_map);
        }
      }

      // Now the InputField can be retrieved.
      const InputField<std::vector<double>>& input_field_conductivity =
          container.get<InputField<std::vector<double>>>("CONDUCT");
      std::vector<std::vector<double>> expected_conductivity{
          {1.0, 2.0, 3.0}, {3.0, 2.0, 1.0}, {1.0, 2.0, 3.0}, {3.0, 2.0, 1.0}};

      if (Core::Communication::my_mpi_rank(MPI_COMM_WORLD) == 0)
      {
        EXPECT_EQ(input_field_conductivity.at(0), expected_conductivity[0]);
        EXPECT_EQ(input_field_conductivity.at(1), expected_conductivity[1]);
        EXPECT_ANY_THROW([[maybe_unused]] auto v = input_field_conductivity.at(3));
      }
      else
      {
        EXPECT_EQ(input_field_conductivity.at(2), expected_conductivity[2]);
        EXPECT_EQ(input_field_conductivity.at(3), expected_conductivity[3]);
        EXPECT_ANY_THROW([[maybe_unused]] auto v = input_field_conductivity.at(1));
      }
    }
  }
}  // namespace

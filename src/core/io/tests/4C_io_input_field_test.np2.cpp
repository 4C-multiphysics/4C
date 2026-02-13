// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_config.hpp"

#include "4C_io_input_field.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_io_input_file.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_io_mesh.hpp"
#include "4C_io_vtu_reader.hpp"
#include "4C_io_yaml.hpp"
#include "4C_rebalance.hpp"
#include "4C_unittest_utils_assertions_test.hpp"
#include "4C_unittest_utils_create_discretization_helper_test.hpp"
#include "4C_unittest_utils_support_files_test.hpp"
#include "4C_utils_singleton_owner.hpp"

#include <mpi.h>

#include <map>
#include <type_traits>

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

  TEST(InputField, ReadPointBasedFieldRegistry)
  {
#ifndef FOUR_C_WITH_VTK
    GTEST_SKIP() << "Skipping test: Reading point based input fields requires VTK support";
#endif
    Core::Utils::SingletonOwnerRegistry::ScopeGuard guard;
    // Read mesh from vtu file (only on rank 0)
    auto mesh = MeshInput::Mesh<3>(Core::Communication::my_mpi_rank(MPI_COMM_WORLD) == 0
                                       ? VTU::read_vtu_file(TESTING::get_support_file_path(
                                             "test_files/vtu/hex8_point_data.vtu"))
                                       : MeshInput::RawMesh<3>{});

    // Create discretization from mesh and redistribute
    Core::FE::Discretization dis{"test", MPI_COMM_WORLD, 3};
    TESTING::fill_discretization_from_mesh(dis, mesh);

    // Read input field from yaml spec
    auto spec = interpolated_input_field<double>("FIELD", {.description = "A field"});
    {
      SCOPED_TRACE("Vector input field from file");
      ryml::Tree tree = init_yaml_tree_with_exceptions();
      ryml::NodeRef root = tree.rootref();
      ryml::parse_in_arena("FIELD:\n from_mesh: scalar_float", root);

      ConstYamlNodeRef node(root, "");
      InputParameterContainer container;
      spec.match(node, container);

      // After reading we need to post-process what is in the registry.
      auto& registry = global_mesh_data_input_field_registry();
      EXPECT_TRUE(registry.fields.contains("scalar_float"));
      for (auto& [ref_name, data] : registry.fields)
      {
        if (ref_name == "scalar_float")
        {
          EXPECT_FALSE(data.init_functions.empty());
          data.init_functions.begin()->second(
              dis, mesh, Core::IO::FieldDataBasis::points, "scalar_float");

          EXPECT_FALSE(data.redistribute_functions.empty());
          auto& redistribute_function = data.redistribute_functions.begin()->second;

          ASSERT_EQ(Core::Communication::num_mpi_ranks(MPI_COMM_WORLD), 2);
          redistribute_function(*dis.node_col_map());
        }
      }

      // Now the InputField can be retrieved.
      const auto& input_field = container.get<InterpolatedInputField<double>>("FIELD");
      std::vector<double> expected_conductivity{3.5, 7.5, 11.5};

      if (Core::Communication::my_mpi_rank(MPI_COMM_WORLD) == 0)
      {
        EXPECT_NEAR(
            input_field.interpolate(0, std::array{0.0, 0.0, 0.0}), expected_conductivity[0], 1e-12);
        EXPECT_NEAR(
            input_field.interpolate(1, std::array{0.0, 0.0, 0.0}), expected_conductivity[1], 1e-12);
        EXPECT_NEAR(
            input_field.interpolate(2, std::array{0.0, 0.0, 0.0}), expected_conductivity[2], 1e-12);
      }
      else
      {
        EXPECT_NEAR(
            input_field.interpolate(0, std::array{0.0, 0.0, 0.0}), expected_conductivity[0], 1e-12);
        EXPECT_NEAR(
            input_field.interpolate(1, std::array{0.0, 0.0, 0.0}), expected_conductivity[1], 1e-12);
        EXPECT_NEAR(
            input_field.interpolate(2, std::array{0.0, 0.0, 0.0}), expected_conductivity[2], 1e-12);
      }
    }
  }

  TEST(InputField, ReadCustomInterpolatedPointBasedFieldRegistry)
  {
#ifndef FOUR_C_WITH_VTK
    GTEST_SKIP() << "Skipping test: Reading point based input fields requires VTK support";
#endif
    Core::Utils::SingletonOwnerRegistry::ScopeGuard guard;

    // Read mesh from vtu file (only on rank 0)
    auto mesh = MeshInput::Mesh<3>(Core::Communication::my_mpi_rank(MPI_COMM_WORLD) == 0
                                       ? VTU::read_vtu_file(TESTING::get_support_file_path(
                                             "test_files/vtu/hex8_point_data.vtu"))
                                       : MeshInput::RawMesh<3>{});

    // Create discretization from mesh and redistribute
    Core::FE::Discretization dis{"test", MPI_COMM_WORLD, 3};
    TESTING::fill_discretization_from_mesh(dis, mesh);

    using ScaledLinearInterpolator = decltype([](const auto& weights, const auto& values) {
      using T = std::remove_cvref_t<decltype(values[0])>;
      return 2*std::inner_product(weights.begin(), weights.end(), values.begin(), T{});
    });


    // Read input field from yaml spec
    auto spec = interpolated_input_field<double, ScaledLinearInterpolator>(
        "FIELD", {.description = "A field"});
    {
      SCOPED_TRACE("Vector input field from file");
      ryml::Tree tree = init_yaml_tree_with_exceptions();
      ryml::NodeRef root = tree.rootref();
      ryml::parse_in_arena("FIELD:\n from_mesh: scalar_float", root);

      ConstYamlNodeRef node(root, "");
      InputParameterContainer container;
      spec.match(node, container);

      // After reading we need to post-process what is in the registry.
      auto& registry = global_mesh_data_input_field_registry();
      EXPECT_TRUE(registry.fields.contains("scalar_float"));
      for (auto& [ref_name, data] : registry.fields)
      {
        if (ref_name == "scalar_float")
        {
          EXPECT_FALSE(data.init_functions.empty());
          data.init_functions.begin()->second(
              dis, mesh, Core::IO::FieldDataBasis::points, "scalar_float");

          EXPECT_FALSE(data.redistribute_functions.empty());
          auto& redistribute_function = data.redistribute_functions.begin()->second;

          ASSERT_EQ(Core::Communication::num_mpi_ranks(MPI_COMM_WORLD), 2);
          redistribute_function(*dis.node_col_map());
        }
      }

      // Now the InputField can be retrieved.
      const auto& input_field =
          container.get<InterpolatedInputField<double, ScaledLinearInterpolator>>("FIELD");
      std::vector<double> expected_conductivity{7, 15, 23};

      if (Core::Communication::my_mpi_rank(MPI_COMM_WORLD) == 0)
      {
        EXPECT_NEAR(
            input_field.interpolate(0, std::array{0.0, 0.0, 0.0}), expected_conductivity[0], 1e-12);
        EXPECT_NEAR(
            input_field.interpolate(1, std::array{0.0, 0.0, 0.0}), expected_conductivity[1], 1e-12);
        EXPECT_NEAR(
            input_field.interpolate(2, std::array{0.0, 0.0, 0.0}), expected_conductivity[2], 1e-12);
      }
      else
      {
        EXPECT_NEAR(
            input_field.interpolate(0, std::array{0.0, 0.0, 0.0}), expected_conductivity[0], 1e-12);
        EXPECT_NEAR(
            input_field.interpolate(1, std::array{0.0, 0.0, 0.0}), expected_conductivity[1], 1e-12);
        EXPECT_NEAR(
            input_field.interpolate(2, std::array{0.0, 0.0, 0.0}), expected_conductivity[2], 1e-12);
      }
    }
  }
}  // namespace

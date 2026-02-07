// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_comm_mpi_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_clement_interpolation.hpp"
#include "4C_io_gridgenerator.hpp"
#include "4C_io_input_parameter_container.templates.hpp"
#include "4C_io_pstream.hpp"
#include "4C_unittest_utils_create_discretization_helper_test.hpp"


namespace
{
  using namespace FourC;

  class BuildClementInterpolationTest : public testing::Test
  {
   public:
    BuildClementInterpolationTest()
    {
      comm_ = MPI_COMM_WORLD;

      Core::IO::cout.setup(false, false, false, Core::IO::standard, comm_, 0, 0, "dummyFilePrefix");
    }

    void TearDown() override { Core::IO::cout.close(); }

    void setup_uniform_hex_mesh(Core::FE::Discretization& discretization)
    {
      const auto add_node = [&](int id, std::array<double, 3> coords)
      { discretization.add_node(coords, id, nullptr); };

      const std::vector<std::array<double, 3>> coords{{0.0, 0.0, 0.0},  // 0
          {1.0, 0.0, 0.0},                                              // 1
          {2.0, 0.0, 0.0},                                              // 2
          {2.0, 1.0, 0.0},                                              // 3
          {1.0, 1.0, 0.0},                                              // 4
          {0.0, 1.0, 0.0},                                              // 5
          {0.0, 0.0, 1.0},                                              // 6
          {1.0, 0.0, 1.0},                                              // 7
          {2.0, 0.0, 1.0},                                              // 8
          {2.0, 1.0, 1.0},                                              // 9
          {1.0, 1.0, 1.0},                                              // 10
          {0.0, 1.0, 1.0}};                                             // 11

      int counter = 0;
      for (const auto& coord : coords) add_node(counter++, coord);

      // Add unit hex8 element
      {
        std::array<int, 8> nodeids{0, 1, 4, 5, 6, 7, 10, 11};

        TESTING::PureGeometryElementType::instance();
        auto ele = std::make_unique<TESTING::PureGeometryElement>(0, 0,
            TESTING::PureGeometryElement::Data{
                .cell_type = Core::FE::CellType::hex8, .num_dof_per_node = 1});
        ele->set_node_ids(8, nodeids.begin());

        discretization.add_element(std::move(ele));
      }

      // Add unit hex8 element
      {
        std::array<int, 8> nodeids{1, 2, 3, 4, 7, 8, 9, 10};

        TESTING::PureGeometryElementType::instance();
        auto ele = std::make_unique<TESTING::PureGeometryElement>(1, 0,
            TESTING::PureGeometryElement::Data{
                .cell_type = Core::FE::CellType::hex8, .num_dof_per_node = 1});
        ele->set_node_ids(8, nodeids.begin());

        discretization.add_element(std::move(ele));
      }

      discretization.fill_complete(Core::FE::OptionsFillComplete::none());
    }

    void setup_nonuniform_hex_mesh(Core::FE::Discretization& discretization)
    {
      const auto add_node = [&](int id, std::array<double, 3> coords)
      { discretization.add_node(coords, id, nullptr); };

      const std::vector<std::array<double, 3>> coords{{0.0, 0.0, 0.0},  // 0
          {1.0, 0.0, 0.0},                                              // 1
          {1.5, 0.0, 0.0},                                              // 2
          {1.5, 1.0, 0.0},                                              // 3
          {1.0, 1.0, 0.0},                                              // 4
          {0.0, 1.0, 0.0},                                              // 5
          {0.0, 0.0, 1.0},                                              // 6
          {1.0, 0.0, 1.0},                                              // 7
          {1.5, 0.0, 1.0},                                              // 8
          {1.5, 1.0, 1.0},                                              // 9
          {1.0, 1.0, 1.0},                                              // 10
          {0.0, 1.0, 1.0}};                                             // 11

      int counter = 0;
      for (const auto& coord : coords) add_node(counter++, coord);

      // Add unit hex8 element
      {
        std::array<int, 8> nodeids{0, 1, 4, 5, 6, 7, 10, 11};

        TESTING::PureGeometryElementType::instance();
        auto ele = std::make_unique<TESTING::PureGeometryElement>(0, 0,
            TESTING::PureGeometryElement::Data{
                .cell_type = Core::FE::CellType::hex8, .num_dof_per_node = 1});
        ele->set_node_ids(8, nodeids.begin());

        discretization.add_element(std::move(ele));
      }

      // Add half hex8 element
      {
        std::array<int, 8> nodeids{1, 2, 3, 4, 7, 8, 9, 10};

        TESTING::PureGeometryElementType::instance();
        auto ele = std::make_unique<TESTING::PureGeometryElement>(1, 0,
            TESTING::PureGeometryElement::Data{
                .cell_type = Core::FE::CellType::hex8, .num_dof_per_node = 1});
        ele->set_node_ids(8, nodeids.begin());

        discretization.add_element(std::move(ele));
      }

      discretization.fill_complete(Core::FE::OptionsFillComplete::none());
    }

   protected:
    MPI_Comm comm_;
  };


  TEST_F(BuildClementInterpolationTest, UniformMeshScalarElementValue)
  {
    auto test_discretization = Core::FE::Discretization("dummy", comm_, 3);
    setup_uniform_hex_mesh(test_discretization);

    // We initialize the element vector with constant values of 1. The Clement interpolant, due to
    // uniform mesh size, will produce values of 1 at all nodes.
    {
      Core::LinAlg::Vector<double> element_values(*test_discretization.element_col_map());
      element_values.put_scalar(1.0);

      auto nodal_values =
          Core::FE::compute_nodal_clement_interpolation(test_discretization, element_values);

      EXPECT_EQ(nodal_values->local_length(), test_discretization.num_my_row_nodes());
      EXPECT_EQ(nodal_values->num_vectors(), 1);

      double min_value;
      nodal_values->get_vector(0).min_value(&min_value);
      EXPECT_NEAR(min_value, 1.0, 1e-14);

      double max_value;
      nodal_values->get_vector(0).max_value(&max_value);
      EXPECT_NEAR(max_value, 1.0, 1e-14);
    }

    // We initialize the element vector with two different constant values. The Clement interpolant,
    // due to uniform mesh size, will produce the arithmetic average on each node.
    {
      Core::LinAlg::Vector<double> element_values(*test_discretization.element_col_map());
      element_values.replace_global_value(0, 10);
      element_values.replace_global_value(1, 20);

      auto nodal_values =
          Core::FE::compute_nodal_clement_interpolation(test_discretization, element_values);

      EXPECT_EQ(nodal_values->local_length(), test_discretization.num_my_row_nodes());
      EXPECT_EQ(nodal_values->num_vectors(), 1);

      // The minimal value has to be 10.0 as the first node is only attached to the element holding
      // 0
      double min_value;
      nodal_values->get_vector(0).min_value(&min_value);
      EXPECT_NEAR(min_value, 10.0, 1e-14);

      // The maximum value has to be 20.0 as the last node is only attached to the element holding 7
      double max_value;
      nodal_values->get_vector(0).max_value(&max_value);
      EXPECT_NEAR(max_value, 20.0, 1e-14);

      // The nodes in the middle have to hold the arithmetic average of all values, due to a uniform
      // mesh size
      EXPECT_NEAR(nodal_values->get_values()[1], 15.0, 1e-14);
    }
  }

  TEST_F(BuildClementInterpolationTest, UniformMeshVectorElementValue)
  {
    auto test_discretization = Core::FE::Discretization("dummy", comm_, 3);
    setup_uniform_hex_mesh(test_discretization);

    // We initialize the element vector columns with constant values of 1, 2 and 3.
    // The Clement interpolant, due to uniform mesh size, will produce values of 1,2 and 3 at all
    // nodes.
    {
      Core::LinAlg::MultiVector<double> element_values(*test_discretization.element_col_map(), 3);
      element_values.get_vector(0).put_scalar(1.0);
      element_values.get_vector(1).put_scalar(2.0);
      element_values.get_vector(2).put_scalar(3.0);

      auto nodal_values =
          Core::FE::compute_nodal_clement_interpolation(test_discretization, element_values);

      EXPECT_EQ(nodal_values->local_length(), test_discretization.num_my_row_nodes());
      EXPECT_EQ(nodal_values->num_vectors(), 3);

      double min_value;
      nodal_values->get_vector(0).min_value(&min_value);
      EXPECT_NEAR(min_value, 1.0, 1e-14);
      nodal_values->get_vector(1).min_value(&min_value);
      EXPECT_NEAR(min_value, 2.0, 1e-14);
      nodal_values->get_vector(2).min_value(&min_value);
      EXPECT_NEAR(min_value, 3.0, 1e-14);

      double max_value;
      nodal_values->get_vector(0).max_value(&max_value);
      EXPECT_NEAR(max_value, 1.0, 1e-14);
      nodal_values->get_vector(1).max_value(&max_value);
      EXPECT_NEAR(max_value, 2.0, 1e-14);
      nodal_values->get_vector(2).max_value(&max_value);
      EXPECT_NEAR(max_value, 3.0, 1e-14);
    }

    // We initialize the element vector columns with two different constant values each.
    // The Clement interpolant, due to uniform mesh size, will produce the arithmetic average on
    // each node.
    {
      Core::LinAlg::MultiVector<double> element_values(*test_discretization.element_col_map(), 3);
      element_values.get_vector(0).replace_global_value(0, 10);
      element_values.get_vector(0).replace_global_value(1, 20);
      element_values.get_vector(1).replace_global_value(0, 10);
      element_values.get_vector(1).replace_global_value(1, 100);
      element_values.get_vector(2).replace_global_value(0, 10);
      element_values.get_vector(2).replace_global_value(1, 1000);

      auto nodal_values =
          Core::FE::compute_nodal_clement_interpolation(test_discretization, element_values);

      EXPECT_EQ(nodal_values->local_length(), test_discretization.num_my_row_nodes());
      EXPECT_EQ(nodal_values->num_vectors(), 3);

      // The minimal value has to be 10.0 as the first node is only attached to the element holding
      // 0
      double min_value;
      nodal_values->get_vector(0).min_value(&min_value);
      EXPECT_NEAR(min_value, 10.0, 1e-14);
      nodal_values->get_vector(1).min_value(&min_value);
      EXPECT_NEAR(min_value, 10.0, 1e-14);
      nodal_values->get_vector(2).min_value(&min_value);
      EXPECT_NEAR(min_value, 10.0, 1e-14);

      // The maximum value has to be 20.0 as the last node is only attached to the element holding 7
      double max_value;
      nodal_values->get_vector(0).max_value(&max_value);
      EXPECT_NEAR(max_value, 20.0, 1e-14);
      nodal_values->get_vector(1).max_value(&max_value);
      EXPECT_NEAR(max_value, 100.0, 1e-14);
      nodal_values->get_vector(2).max_value(&max_value);
      EXPECT_NEAR(max_value, 1000.0, 1e-14);

      // The nodes in the middle have to hold the arithmetic average of all values, due to a uniform
      // mesh size
      EXPECT_NEAR(nodal_values->get_vector(0).get_values()[1], 15.0, 1e-14);
      EXPECT_NEAR(nodal_values->get_vector(1).get_values()[1], 55.0, 1e-14);
      EXPECT_NEAR(nodal_values->get_vector(2).get_values()[1], 505.0, 1e-14);
    }
  }


  TEST_F(BuildClementInterpolationTest, NonUniformMeshScalarElementValue)
  {
    auto test_discretization = Core::FE::Discretization("dummy", comm_, 3);
    setup_nonuniform_hex_mesh(test_discretization);

    // We initialize the element vector with constant values of 1. The Clement interpolant, due to
    // uniform mesh size, will produce values of 1 at all nodes.
    {
      Core::LinAlg::Vector<double> element_values(*test_discretization.element_col_map());
      element_values.put_scalar(1.0);

      auto nodal_values =
          Core::FE::compute_nodal_clement_interpolation(test_discretization, element_values);

      EXPECT_EQ(nodal_values->local_length(), test_discretization.num_my_row_nodes());
      EXPECT_EQ(nodal_values->num_vectors(), 1);

      double min_value;
      nodal_values->get_vector(0).min_value(&min_value);
      EXPECT_NEAR(min_value, 1.0, 1e-14);

      double max_value;
      nodal_values->get_vector(0).max_value(&max_value);
      EXPECT_NEAR(max_value, 1.0, 1e-14);
    }

    // We initialize the element vector with two different constant values. The Clement interpolant,
    // due to non-uniform mesh size, will produce the weighted average on each node.
    {
      Core::LinAlg::Vector<double> element_values(*test_discretization.element_col_map());
      element_values.replace_global_value(0, 10.0);
      element_values.replace_global_value(1, 20.0);

      auto nodal_values =
          Core::FE::compute_nodal_clement_interpolation(test_discretization, element_values);

      EXPECT_EQ(nodal_values->local_length(), test_discretization.num_my_row_nodes());
      EXPECT_EQ(nodal_values->num_vectors(), 1);

      // The minimal value has to be 10.0 as the first node is only attached to the element holding
      // 0
      double min_value;
      nodal_values->get_vector(0).min_value(&min_value);
      EXPECT_NEAR(min_value, 10.0, 1e-14);

      // The maximum value has to be 20.0 as the last node is only attached to the element holding 7
      double max_value;
      nodal_values->get_vector(0).max_value(&max_value);
      EXPECT_NEAR(max_value, 20.0, 1e-14);

      // The nodes in the middle have to hold the weighted average of all values, due to a
      // non-uniform mesh size. The weights are represented by the element volume.
      EXPECT_NEAR(nodal_values->get_values()[1], 13.33333333333333, 1e-14);
    }
  }


  TEST_F(BuildClementInterpolationTest, NonUniformMeshVectorElementValue)
  {
    auto test_discretization = Core::FE::Discretization("dummy", comm_, 3);
    setup_nonuniform_hex_mesh(test_discretization);

    // We initialize the element vector columns with constant values of 1, 2 and 3.
    // The Clement interpolant, due to uniform mesh size, will produce values of 1,2 and 3 at all
    // nodes.
    {
      Core::LinAlg::MultiVector<double> element_values(*test_discretization.element_col_map(), 3);
      element_values.get_vector(0).put_scalar(1.0);
      element_values.get_vector(1).put_scalar(2.0);
      element_values.get_vector(2).put_scalar(3.0);

      auto nodal_values =
          Core::FE::compute_nodal_clement_interpolation(test_discretization, element_values);

      EXPECT_EQ(nodal_values->local_length(), test_discretization.num_my_row_nodes());
      EXPECT_EQ(nodal_values->num_vectors(), 3);

      double min_value;
      nodal_values->get_vector(0).min_value(&min_value);
      EXPECT_NEAR(min_value, 1.0, 1e-14);
      nodal_values->get_vector(1).min_value(&min_value);
      EXPECT_NEAR(min_value, 2.0, 1e-14);
      nodal_values->get_vector(2).min_value(&min_value);
      EXPECT_NEAR(min_value, 3.0, 1e-14);

      double max_value;
      nodal_values->get_vector(0).max_value(&max_value);
      EXPECT_NEAR(max_value, 1.0, 1e-14);
      nodal_values->get_vector(1).max_value(&max_value);
      EXPECT_NEAR(max_value, 2.0, 1e-14);
      nodal_values->get_vector(2).max_value(&max_value);
      EXPECT_NEAR(max_value, 3.0, 1e-14);
    }

    // We initialize the element vector columns with two different constant values each.
    // The Clement interpolant, due to uniform mesh size, will produce the arithmetic average on
    // each node.
    {
      Core::LinAlg::MultiVector<double> element_values(*test_discretization.element_col_map(), 3);
      element_values.get_vector(0).replace_global_value(0, 10);
      element_values.get_vector(0).replace_global_value(1, 20);
      element_values.get_vector(1).replace_global_value(0, 10);
      element_values.get_vector(1).replace_global_value(1, 100);
      element_values.get_vector(2).replace_global_value(0, 10);
      element_values.get_vector(2).replace_global_value(1, 1000);

      auto nodal_values =
          Core::FE::compute_nodal_clement_interpolation(test_discretization, element_values);

      EXPECT_EQ(nodal_values->local_length(), test_discretization.num_my_row_nodes());
      EXPECT_EQ(nodal_values->num_vectors(), 3);

      // The minimal value has to be 10.0 as the first node is only attached to the element holding
      // 0
      double min_value;
      nodal_values->get_vector(0).min_value(&min_value);
      EXPECT_NEAR(min_value, 10.0, 1e-14);
      nodal_values->get_vector(1).min_value(&min_value);
      EXPECT_NEAR(min_value, 10.0, 1e-14);
      nodal_values->get_vector(2).min_value(&min_value);
      EXPECT_NEAR(min_value, 10.0, 1e-14);

      // The maximum value has to be 20.0 as the last node is only attached to the element holding 7
      // etc.
      double max_value;
      nodal_values->get_vector(0).max_value(&max_value);
      EXPECT_NEAR(max_value, 20.0, 1e-14);
      nodal_values->get_vector(1).max_value(&max_value);
      EXPECT_NEAR(max_value, 100.0, 1e-14);
      nodal_values->get_vector(2).max_value(&max_value);
      EXPECT_NEAR(max_value, 1000.0, 1e-14);

      // The nodes in the middle have to hold the weighted average of all values, due to a
      // non-uniform mesh size. The weights are represented by the element volume.
      EXPECT_NEAR(nodal_values->get_vector(0).get_values()[1], 13.33333333333333, 1e-14);
      EXPECT_NEAR(nodal_values->get_vector(1).get_values()[1], 40.0, 1e-14);
      EXPECT_NEAR(nodal_values->get_vector(2).get_values()[1], 340.0, 1e-14);
    }
  }
}  // namespace

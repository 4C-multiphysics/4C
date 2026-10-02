// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_fem_general_node.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_discretization_builder.hpp"
#include "4C_rebalance.hpp"
#include "4C_unittest_utils_assertions_test.hpp"
#include "4C_utils_exceptions.hpp"


namespace
{
  using namespace FourC;

  // General tests for FEM nodes
  class GeneralNodeTest : public testing::Test
  {
   public:
    /// setup a discretization consisting of two HEX8 elements stacked onto each other
    void set_up_two_stacked_hex8_discretization(Core::FE::Discretization& test_discretization)
    {
      Core::FE::DiscretizationBuilder<3> builder(test_discretization.get_comm());

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
      for (const auto& coord : coords) builder.add_node(coord, counter++, nullptr);

      // Add unit hex8 element
      {
        const int ele_id = 0;
        std::array<int, 8> node_ids{0, 1, 4, 5, 6, 7, 10, 11};

        builder.add_element(Core::FE::CellType::hex8, node_ids, ele_id,
            {.num_dof_per_node = 1, .num_dof_per_element = 0});
      }

      // Add unit hex8 element
      {
        const int ele_id = 1;
        std::array<int, 8> node_ids{1, 2, 3, 4, 7, 8, 9, 10};

        builder.add_element(Core::FE::CellType::hex8, node_ids, ele_id,
            {.num_dof_per_node = 1, .num_dof_per_element = 0});
      }

      Core::Rebalance::RebalanceParameters rebalance_parameters;
      builder.build(test_discretization, rebalance_parameters);
      test_discretization.fill_complete(Core::FE::OptionsFillComplete::none());
    }

   protected:
    const MPI_Comm comm_ = MPI_COMM_WORLD;
  };

  TEST_F(GeneralNodeTest, NodeDistanceToAdjacentNodes)
  {
    Core::FE::Discretization test_discretization_stacked_hex8 =
        Core::FE::Discretization("dummy", comm_, 3);
    set_up_two_stacked_hex8_discretization(test_discretization_stacked_hex8);
    for (int inode = 0; inode < test_discretization_stacked_hex8.num_my_row_nodes(); inode++)
    {
      Core::Nodes::Node* node = test_discretization_stacked_hex8.l_row_node(inode);
      EXPECT_EQ(node->minimum_distance_to_adjacent_nodes(), 1.0);
    }

    const auto isolated_node = Core::Nodes::Node(0, std::array<double, 3>{0.0, 0.0, 0.0}, 0);
    FOUR_C_EXPECT_THROW_WITH_MESSAGE((void)isolated_node.minimum_distance_to_adjacent_nodes(),
        Core::Exception, "This method should not be called for isolated nodes");
  }

}  // namespace

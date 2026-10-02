// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_fem_general_element.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_discretization_builder.hpp"
#include "4C_rebalance.hpp"
#include "4C_utils_exceptions.hpp"

namespace
{
  using namespace FourC;

  // General tests for FEM elements
  class GeneralElementTest : public testing::Test
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


    /// setup a discretization consisting of a single HEX8 element
    void set_up_single_hex8_discretization(Core::FE::Discretization& test_discretization)
    {
      Core::FE::DiscretizationBuilder<3> builder(test_discretization.get_comm());

      const std::vector<std::array<double, 3>> coords{{0.0, 0.0, 0.0},  // 0
          {1.0, 0.0, 0.0},                                              // 1
          {1.0, 1.0, 0.0},                                              // 2
          {0.0, 1.0, 0.0},                                              // 3
          {0.0, 0.0, 1.0},                                              // 4
          {1.0, 0.0, 1.0},                                              // 5
          {1.0, 1.0, 1.0},                                              // 6
          {0.0, 1.0, 1.0}};                                             // 7

      int counter = 0;
      for (const auto& coord : coords) builder.add_node(coord, counter++, nullptr);

      // Add unit hex8 element
      {
        const int ele_id = 0;
        std::array<int, 8> node_ids{0, 1, 2, 3, 4, 5, 6, 7};

        builder.add_element(Core::FE::CellType::hex8, node_ids, ele_id,
            {.num_dof_per_node = 1, .num_dof_per_element = 0});
      }

      Core::Rebalance::RebalanceParameters rebalance_parameters;
      builder.build(test_discretization, rebalance_parameters);
      test_discretization.fill_complete(Core::FE::OptionsFillComplete::none());
    }

   protected:
    MPI_Comm comm_ = MPI_COMM_WORLD;
  };

  TEST_F(GeneralElementTest, CentroidDistanceToAdjacentElements)
  {
    // for two stacked HEX8 elements, both elements are adjacent and should have a centroid distance
    // to each other
    Core::FE::Discretization test_discretization_stacked_hex8 =
        Core::FE::Discretization("dummy", comm_, 3);
    set_up_two_stacked_hex8_discretization(test_discretization_stacked_hex8);

    for (int i = 0; i < test_discretization_stacked_hex8.num_my_row_elements(); i++)
    {
      const Core::Elements::Element* ele = test_discretization_stacked_hex8.l_row_element(i);
      const std::optional<double> centroid_distance =
          ele->minimum_centroid_distance_to_adjacent_elements();
      FOUR_C_ASSERT_ALWAYS(centroid_distance.has_value(),
          "The element with local id {} should have adjacent elements and thus a centroid distance "
          "to them!",
          i);
      EXPECT_EQ(ele->minimum_centroid_distance_to_adjacent_elements(), 1.0);
    }

    // a single HEX8 element does not have any adjacent elements -> nothing is computed as the
    // centroid distance
    Core::FE::Discretization test_discretization_single_hex8 =
        Core::FE::Discretization("dummy", comm_, 3);
    set_up_single_hex8_discretization(test_discretization_single_hex8);
    FOUR_C_ASSERT_ALWAYS(test_discretization_single_hex8.num_my_row_elements() == 1,
        "The created discretization with a single HEX8 element does not only have a single "
        "element, but {}!",
        test_discretization_single_hex8.num_my_row_elements());
    const Core::Elements::Element* ele = test_discretization_single_hex8.l_row_element(0);
    FOUR_C_ASSERT_ALWAYS(not ele->minimum_centroid_distance_to_adjacent_elements().has_value(),
        "Nothing should be returned here, as the single HEX8 element does not have adjacent "
        "elements");
  }

}  // namespace

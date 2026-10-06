// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_config.hpp"

#include "4C_geometric_search_matchingoctree.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_unittest_utils_assertions_test.hpp"
#include "4C_unittest_utils_create_discretization_helper_test.hpp"

namespace
{
  using namespace FourC;

  class GeometricSearchMatchingOctree : public ::testing::Test
  {
   public:
    GeometricSearchMatchingOctree()
    {
      comm_ = MPI_COMM_WORLD;
      my_rank_ = Core::Communication::my_mpi_rank(comm_);
    }

    /**
     * @brief Set up a mesh consisting of two unit HEX8 elements.
     *
     * @param discretization The discretization to which the mesh is added.
     * @param mesh_scaling_factor Scaling factor applied to the element sides.
     */
    void setup_scaled_hex_mesh(
        Core::FE::Discretization& discretization, const double mesh_scaling_factor = 1.0)
    {
      Core::FE::DiscretizationBuilder<3> builder(discretization.get_comm());

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
      for (const auto& coord : coords)
        builder.add_node(std::array<double, 3>{coord[0] * mesh_scaling_factor,
                             coord[1] * mesh_scaling_factor, coord[2] * mesh_scaling_factor},
            counter++, nullptr);

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
      builder.build(discretization, rebalance_parameters);
      discretization.fill_complete(Core::FE::OptionsFillComplete::none());
    }

   protected:
    MPI_Comm comm_;
    int my_rank_;
  };

  TEST_F(GeometricSearchMatchingOctree, NodeMatchingUnderSmallMeshScale)
  {
    auto all_source_node_ids_matched = [](const std::vector<int>& source_node_ids,
                                           const std::map<int, std::pair<int, double>>& coupling)
    {
      return std::all_of(source_node_ids.begin(), source_node_ids.end(),
          [&](const int value)
          {
            return std::any_of(coupling.begin(), coupling.end(),
                [&](const auto& entry) { return entry.second.first == value; });
          });
    };



    // setup two identical meshes with an extremely small length scale
    const double mesh_scale = 1.0e-10;
    auto test_target_discretization = Core::FE::Discretization("dummy_target", comm_, 3);
    setup_scaled_hex_mesh(test_target_discretization, mesh_scale);
    auto test_source_discretization = Core::FE::Discretization("dummy_source", comm_, 3);
    setup_scaled_hex_mesh(test_source_discretization, mesh_scale);
    std::vector<int> source_and_target_node_ids;
    auto node_range = test_target_discretization.my_row_node_range();
    for (const auto& node : node_range) source_and_target_node_ids.push_back(node.global_id());

    // setup two node matching octrees; one using a tight tolerance, and one with a very loose one.
    // Then attempt to find the matching nodes from the source and target discretizations: this will
    // only work if the tolerance is not very loose.
    const int max_node_per_octree_leaf = 150;

    const double tight_tolerance =
        1.0e-3;  // note that the tolerance is still high compared to the mesh scale, but the node
                 // matching tree scales it further internally based on the mesh scale!
    auto tight_tolerance_node_matching_octree = Core::GeometricSearch::NodeMatchingOctree();
    tight_tolerance_node_matching_octree.init(test_target_discretization,
        source_and_target_node_ids, max_node_per_octree_leaf, tight_tolerance);
    tight_tolerance_node_matching_octree.setup();
    std::map<int, std::pair<int, double>> tight_tolerance_coupling;
    tight_tolerance_node_matching_octree.find_match(
        test_source_discretization, source_and_target_node_ids, tight_tolerance_coupling);
    EXPECT_TRUE(all_source_node_ids_matched(source_and_target_node_ids, tight_tolerance_coupling));


    const double loose_tolerance =
        1.0e2;  // note that this actually means 100x the distance from one node to the next one, so
                // we are spanning multiple elements with this loose tolerance!
    auto loose_tolerance_node_matching_octree = Core::GeometricSearch::NodeMatchingOctree();
    loose_tolerance_node_matching_octree.init(test_target_discretization,
        source_and_target_node_ids, max_node_per_octree_leaf, loose_tolerance);
    loose_tolerance_node_matching_octree.setup();
    std::map<int, std::pair<int, double>> loose_tolerance_coupling;
    loose_tolerance_node_matching_octree.find_match(
        test_source_discretization, source_and_target_node_ids, loose_tolerance_coupling);
    EXPECT_FALSE(all_source_node_ids_matched(source_and_target_node_ids,
        loose_tolerance_coupling));  // loose tolerances may lose source nodes along the way!
  }

}  // namespace

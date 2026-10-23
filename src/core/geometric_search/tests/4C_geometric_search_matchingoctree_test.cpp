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
#include "4C_fem_general_element.hpp"
#include "4C_unittest_utils_create_discretization_helper_test.hpp"
#include "4C_utils_exceptions.hpp"

#include <algorithm>
#include <utility>

namespace
{
  using namespace FourC;

  class GeometricSearchMatchingOctree : public ::testing::Test
  {
   public:
    GeometricSearchMatchingOctree() { comm_ = MPI_COMM_WORLD; }

    /**
     * @brief Set up a mesh consisting of two unit HEX8 elements.
     *
     * @param discretization The discretization to which the mesh is added.
     * @param mesh_scaling_factor Scaling factor applied to the element sides.
     */
    void setup_scaled_two_hex8_mesh(
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

    /**
     * @brief Set up a mesh consisting of a single HEX8 element with nodes shifted by \p node_shift
     * from original location (\f$x, y, z \in {0, 1}, {0, 1}, {0,1}\f$).
     *
     * @param discretization The discretization to which the mesh is added.
     * @param node_shift Shift applied to the nodes \f$ [x, y, z]_{\mathrm{shift}} \f$
     * @param mesh_scaling_factor Scaling factor applied to the element sides.
     */
    void setup_scaled_shifted_hex8_element(Core::FE::Discretization& discretization,
        const std::array<double, 3> node_shift, const double mesh_scaling_factor = 1.0)
    {
      Core::FE::DiscretizationBuilder<3> builder(discretization.get_comm());
      std::vector<std::array<double, 3>> coords{{0.0, 0.0, 0.0},  // 0
          {1.0, 0.0, 0.0},                                        // 1
          {1.0, 1.0, 0.0},                                        // 2
          {0.0, 1.0, 0.0},                                        // 3
          {0.0, 0.0, 1.0},                                        // 4
          {1.0, 0.0, 1.0},                                        // 5
          {1.0, 1.0, 1.0},                                        // 6
          {0.0, 1.0, 1.0}};                                       // 7
      for (auto& node_coords : coords)
      {
        for (unsigned int dim = 0; dim < 3; ++dim)
        {
          node_coords[dim] += node_shift[dim];
        }
      }

      int counter = 0;
      for (const auto& coord : coords)
        builder.add_node(std::array<double, 3>{coord[0] * mesh_scaling_factor,
                             coord[1] * mesh_scaling_factor, coord[2] * mesh_scaling_factor},
            counter++, nullptr);

      // Add unit hex8 element
      {
        const int ele_id = 0;
        std::array<int, 8> node_ids{0, 1, 2, 3, 4, 5, 6, 7};

        builder.add_element(Core::FE::CellType::hex8, node_ids, ele_id,
            {.num_dof_per_node = 1, .num_dof_per_element = 0});
      }

      Core::Rebalance::RebalanceParameters rebalance_parameters;
      builder.build(discretization, rebalance_parameters);
      discretization.fill_complete(Core::FE::OptionsFillComplete::none());
    }

   protected:
    MPI_Comm comm_;
  };

  TEST_F(GeometricSearchMatchingOctree, NodeMatching)
  {
    // utility function to run node-matching tests for a given characteristic mesh length
    auto perform_node_matching_tests = [this](const double characteristic_mesh_length)
    {
      /// returns whether all input source ids are matched exactly in the coupling map, used for
      /// \ref Core::GeometricSearch::MatchingOctree::find_match
      auto find_match_check = [](const std::vector<int>& source_ids,
                                  const std::map<int, std::pair<int, double>>& coupling)
      {
        if (coupling.size() != source_ids.size()) return false;

        return std::ranges::all_of(coupling,
            [](const auto& coupling_item)
            {
              return coupling_item.first == coupling_item.second.first &&
                     std::abs(coupling_item.second.second) <=
                         1.0e-15;  // check that the coupled nodes are the same and that they also
                                   // spatially coincide (distance numerically 0.0)
            });
      };

      // \overload of the previous function, used for
      // \ref Core::GeometricSearch::MatchingOctree::fill_source_to_target_gid_mapping
      auto fill_source_to_target_check =
          [](const std::vector<int>& source_ids, const std::map<int, std::vector<double>>& coupling)
      {
        if (coupling.size() != source_ids.size()) return false;

        return std::ranges::all_of(coupling,
            [](const auto& coupling_item)
            {
              return coupling_item.first == coupling_item.second[0] &&
                     std::abs(coupling_item.second[1]) <= 1.0e-15 &&
                     bool(coupling_item.second[2]);  // check that the coupled nodes are the same
                                                     // and that they also spatially coincide
                                                     // (distance numerically 0.0); also, the target
                                                     // node should be owned
            });
      };

      // settings for node matching
      const int max_nodes_per_octree_leaf = 150;
      const double tight_tolerance = 1.0e-2;  // this means 1% of the mesh length is used as a
                                              // tolerance for the octree procedures
      const double loose_tolerance =
          1.0e2;  // this means 100x the distance from one node to the next one,
                  // so we are spanning multiple elements with this loose tolerance!

      /// Utility function: Run node matching tests given the discretizations, the node octree
      /// matching tolerance, and whether to expect successful matching using find_match, and
      /// source-to-target coupling.
      auto run_node_matching_tests =
          [&](const Core::FE::Discretization& source_discretization,
              const Core::FE::Discretization& target_discretization, const double octree_tolerance,
              const bool expect_find_match, const bool expect_source_to_target)
      {
        std::vector<int> source_and_target_node_ids;
        auto node_range = target_discretization.my_row_node_range();
        for (const auto& node : node_range) source_and_target_node_ids.push_back(node.global_id());

        auto octree = Core::GeometricSearch::NodeMatchingOctree();
        octree.init(target_discretization, source_and_target_node_ids, max_nodes_per_octree_leaf,
            octree_tolerance);
        octree.setup();
        std::map<int, std::pair<int, double>> coupling;
        octree.find_match(source_discretization, source_and_target_node_ids, coupling);
        const bool found_all_matches = find_match_check(source_and_target_node_ids, coupling);
        if (expect_find_match)
        {
          EXPECT_TRUE(found_all_matches);
        }
        else
          EXPECT_FALSE(found_all_matches);
        std::map<int, std::vector<double>> source_to_target_coupling;
        octree.fill_source_to_target_gid_mapping(
            source_discretization, source_and_target_node_ids, source_to_target_coupling);
        const bool source_to_target_coupled =
            fill_source_to_target_check(source_and_target_node_ids, source_to_target_coupling);
        if (expect_source_to_target)
          EXPECT_TRUE(source_to_target_coupled);
        else
          EXPECT_FALSE(source_to_target_coupled);
      };

      /// run tests of discretizations with two HEX8 elements -> accurate matching only possible for
      /// the tight tolerance
      auto run_two_hex8_mesh_tests = [&, this](const double octree_tolerance,
                                         const bool expect_find_match,
                                         const bool expect_source_to_target)
      {
        auto target_discretization = Core::FE::Discretization("dummy_target_two_hex8", comm_, 3);
        setup_scaled_two_hex8_mesh(target_discretization, characteristic_mesh_length);
        auto source_discretization = Core::FE::Discretization("dummy_source_two_hex8", comm_, 3);
        setup_scaled_two_hex8_mesh(source_discretization, characteristic_mesh_length);

        run_node_matching_tests(source_discretization, target_discretization, octree_tolerance,
            expect_find_match, expect_source_to_target);
      };
      run_two_hex8_mesh_tests(tight_tolerance, true, true);
      run_two_hex8_mesh_tests(loose_tolerance, false, false);
    };

    // test for extremely small characteristic mesh length
    perform_node_matching_tests(1.0e-10);
    // test for moderate characteristic mesh length
    perform_node_matching_tests(1.0);
    // test for extremely large characteristic mesh length
    perform_node_matching_tests(1.0e10);
  }

  TEST_F(GeometricSearchMatchingOctree, ElementMatching)
  {
    // utility function to run element-matching tests for a given characteristic mesh length
    auto perform_element_matching_tests = [this](const double characteristic_mesh_length)
    {
      // settings for element matching
      const int max_elements_per_octree_leaf = 150;
      const double tight_tolerance = 1.0e-2;  // this means 1% of the mesh length is used as a
                                              // tolerance for the octree procedures
      const double loose_tolerance = 1.0e2;   // this means 10000% of the mesh length is used as a
                                              // tolerance for the octree procedures

      /// Utility function: Run element matching tests given the discretizations, the element octree
      /// matching tolerance, and whether to expect successful matching using find_match, and
      /// source-to-target coupling.
      auto run_element_matching_tests =
          [&](const Core::FE::Discretization& source_discretization,
              const Core::FE::Discretization& target_discretization, const double octree_tolerance,
              const bool expect_find_match, const bool expect_source_to_target)
      {
        std::vector<int> source_and_target_element_ids;
        auto element_range = target_discretization.my_row_element_range();
        for (const auto& element : element_range)
          source_and_target_element_ids.push_back(element.global_id());

        auto octree = Core::GeometricSearch::ElementMatchingOctree();
        octree.init(target_discretization, source_and_target_element_ids,
            max_elements_per_octree_leaf, octree_tolerance);
        octree.setup();
        std::map<int, std::pair<int, double>> coupling;
        octree.find_match(source_discretization, source_and_target_element_ids, coupling);
        const bool found_all_matches =
            coupling.size() == source_and_target_element_ids.size() &&
            std::ranges::all_of(coupling,
                [](const auto& coupling_item)
                {
                  return coupling_item.first == coupling_item.second.first;
                });  // check that the elements found during coupling match in regards to GIDs
        if (expect_find_match)
        {
          EXPECT_TRUE(found_all_matches);
        }
        else
          EXPECT_FALSE(found_all_matches);
        std::map<int, std::vector<double>> source_to_target_coupling;
        octree.fill_source_to_target_gid_mapping(
            source_discretization, source_and_target_element_ids, source_to_target_coupling);
        const bool source_to_target_coupled =
            source_to_target_coupling.size() == source_and_target_element_ids.size() &&
            std::ranges::all_of(source_to_target_coupling,
                [](const auto& coupling_item)
                {
                  return coupling_item.first == coupling_item.second[0] &&
                         bool(coupling_item.second[2]);  // check that the elements found during
                                                         // coupling match in regards to GIDs; also,
                                                         // the target element should be owned
                });
        if (expect_source_to_target)
          EXPECT_TRUE(source_to_target_coupled);
        else
          EXPECT_FALSE(source_to_target_coupled);
      };

      /// run tests of single coincident HEX8 elements (matching expected regardless of tolerance)
      auto run_coincident_hex8_mesh_tests = [&, this](const double octree_tolerance)
      {
        auto target_discretization =
            Core::FE::Discretization("dummy_target_coincident_hex8", comm_, 3);
        setup_scaled_shifted_hex8_element(
            target_discretization, {0.0, 0.0, 0.0}, characteristic_mesh_length);
        auto source_discretization =
            Core::FE::Discretization("dummy_source_coincident_hex8", comm_, 3);
        setup_scaled_shifted_hex8_element(
            source_discretization, {0.0, 0.0, 0.0}, characteristic_mesh_length);

        run_element_matching_tests(source_discretization, target_discretization, octree_tolerance,
            true, true);  // coincident discretizations must always match regardless of tolerance
      };
      run_coincident_hex8_mesh_tests(tight_tolerance);
      run_coincident_hex8_mesh_tests(loose_tolerance);

      /// run tests of shifted single HEX8 elements -> elements are shifted such that only the loose
      /// tolerance can lead to (inaccurate) matching
      auto run_shifted_hex8_mesh_tests = [&, this](const double octree_tolerance,
                                             const bool expect_find_match,
                                             const bool expect_source_to_target)
      {
        auto target_discretization =
            Core::FE::Discretization("dummy_target_shifted_hex8", comm_, 3);
        setup_scaled_shifted_hex8_element(
            target_discretization, {0.0, 0.0, 0.0}, characteristic_mesh_length);
        auto source_discretization =
            Core::FE::Discretization("dummy_source_shifted_hex8", comm_, 3);
        setup_scaled_shifted_hex8_element(source_discretization, {loose_tolerance * 0.1, 0.0, 0.0},
            characteristic_mesh_length);  // shift unit element using the specified loose tolerance

        run_element_matching_tests(source_discretization, target_discretization, octree_tolerance,
            expect_find_match, expect_source_to_target);
      };
      run_shifted_hex8_mesh_tests(tight_tolerance, false, false);
      run_shifted_hex8_mesh_tests(loose_tolerance, true, true);


      /// run tests of discretizations with two HEX8 elements -> accurate matching only possible for
      /// the tight tolerance
      auto run_two_hex8_mesh_tests = [&, this](const double octree_tolerance,
                                         const bool expect_find_match,
                                         const bool expect_source_to_target)
      {
        auto target_discretization = Core::FE::Discretization("dummy_target_two_hex8", comm_, 3);
        setup_scaled_two_hex8_mesh(target_discretization, characteristic_mesh_length);
        auto source_discretization = Core::FE::Discretization("dummy_source_two_hex8", comm_, 3);
        setup_scaled_two_hex8_mesh(source_discretization, characteristic_mesh_length);

        run_element_matching_tests(source_discretization, target_discretization, octree_tolerance,
            expect_find_match, expect_source_to_target);
      };
      run_two_hex8_mesh_tests(tight_tolerance, true, true);
      run_two_hex8_mesh_tests(loose_tolerance, false, false);
    };

    // test for extremely small characteristic mesh length
    perform_element_matching_tests(1.0e-10);
    // test for moderate characteristic mesh length
    perform_element_matching_tests(1.0);
    // test for extremely large characteristic mesh length
    perform_element_matching_tests(1.0e10);
  }



}  // namespace

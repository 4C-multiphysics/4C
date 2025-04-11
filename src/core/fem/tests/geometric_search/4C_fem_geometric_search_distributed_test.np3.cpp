// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_config.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_utils_exceptions.hpp"

#include <array>

#ifdef FOUR_C_WITH_ARBORX

#include "4C_fem_geometric_search_create_bounding_volumes_test.hpp"
#include "4C_fem_geometric_search_distributed_tree.hpp"


namespace
{
  using namespace FourC;

  class GeometricSearchDistributed : public ::testing::Test
  {
   public:
    GeometricSearchDistributed()
    {
      comm_ = MPI_COMM_WORLD;
      verbosity_ = Core::IO::minimal;
      my_rank_ = Core::Communication::my_mpi_rank(comm_);
    }

   protected:
    std::vector<std::pair<int, Core::GeometricSearch::BoundingVolume>> primitives_, predicates_;
    MPI_Comm comm_;
    int my_rank_;
    Core::IO::Verbositylevel verbosity_;
  };

  /**
   * Test setup is based on https://github.com/arborx/ArborX/issues/867
   * Checking collision search of kdop primitives against kdop predicates
   */
  TEST_F(GeometricSearchDistributed, CollisionSearchKDOPVsKDOP)
  {
    const auto volumes = create_kdop_bounding_volumes();

    // Add the volumes to the primitives
    if (my_rank_ == 0)
    {
      primitives_.emplace_back(std::pair{0, volumes[0]});
      primitives_.emplace_back(std::pair{1, volumes[2]});
      primitives_.emplace_back(std::pair{2, volumes[1]});
      predicates_.emplace_back(std::pair{3, volumes[0]});
      predicates_.emplace_back(std::pair{4, volumes[2]});
      predicates_.emplace_back(std::pair{5, volumes[1]});
    }
    else if (my_rank_ == 1)
    {
      primitives_.emplace_back(std::pair{6, volumes[0]});
      primitives_.emplace_back(std::pair{7, volumes[1]});
      predicates_.emplace_back(std::pair{8, volumes[2]});
    }
    else if (my_rank_ == 2)
    {
      primitives_.emplace_back(std::pair{9, volumes[0]});
      primitives_.emplace_back(std::pair{10, volumes[2]});
      predicates_.emplace_back(std::pair{11, volumes[1]});
    }

    const auto pairs =
        Core::GeometricSearch::global_collision_search(primitives_, predicates_, comm_, verbosity_);

    // The order of the results we get are is not deterministic. Therefore, we save the reference
    // results in a map, with the colliding GIDs being the keys. Thus the ordering of the pairs in
    // the collision search does not matter.
    auto compare_results = [](const auto& pairs_from_collision_search, const auto& reference_map)
    {
      EXPECT_EQ(pairs_from_collision_search.size(), reference_map.size());
      for (const auto& pair : pairs_from_collision_search)
      {
        if (reference_map.count({pair.gid_predicate, pair.gid_primitive}))
        {
          const auto& reference_pair = reference_map.at({pair.gid_predicate, pair.gid_primitive});
          EXPECT_EQ(pair.lid_predicate, std::get<0>(reference_pair));
          EXPECT_EQ(pair.pid_primitive, std::get<1>(reference_pair));
        }
        else
        {
          FOUR_C_THROW(
              "Pair <{}, {}> not found in reference map", pair.gid_predicate, pair.gid_primitive);
        }
      }
    };

    if (my_rank_ == 0)
    {
      std::map<std::pair<int, int>, std::array<int, 2>> reference_pairs{
          {{3, 0}, {0, 0}},   //
          {{3, 1}, {0, 0}},   //
          {{3, 6}, {0, 1}},   //
          {{3, 9}, {0, 2}},   //
          {{3, 10}, {0, 2}},  //
          {{4, 0}, {1, 0}},   //
          {{4, 1}, {1, 0}},   //
          {{4, 6}, {1, 1}},   //
          {{4, 9}, {1, 2}},   //
          {{4, 10}, {1, 2}},  //
          {{5, 2}, {2, 0}},   //
          {{5, 7}, {2, 1}}    //
      };
      compare_results(pairs, reference_pairs);
    }
    else if (my_rank_ == 1)
    {
      std::map<std::pair<int, int>, std::array<int, 2>> reference_pairs{
          {{8, 0}, {0, 0}},   //
          {{8, 1}, {0, 0}},   //
          {{8, 6}, {0, 1}},   //
          {{8, 9}, {0, 2}},   //
          {{8, 10}, {0, 2}},  //
      };
      compare_results(pairs, reference_pairs);
    }
    else if (my_rank_ == 2)
    {
      std::map<std::pair<int, int>, std::array<int, 2>> reference_pairs{
          {{11, 2}, {0, 0}},  //
          {{11, 7}, {0, 1}}   //
      };
      compare_results(pairs, reference_pairs);
    }
  }

  /**
   * Checking collision search with no predicates on rank 0 and 1 and no primitives on rank 1 and 2
   */
  TEST_F(GeometricSearchDistributed, CollisionSearchNoPrimitivesNoPredicatesMixed)
  {
    const auto volumes = create_kdop_bounding_volumes();

    // Add the volumes to the primitives and predicates for rank 0 and 1
    if (my_rank_ == 0)
    {
      primitives_.emplace_back(std::pair{0, volumes[0]});
      primitives_.emplace_back(std::pair{1, volumes[2]});
      primitives_.emplace_back(std::pair{2, volumes[1]});
    }
    if (my_rank_ == 1)
    {
      predicates_.emplace_back(std::pair{3, volumes[2]});
    }

    const auto pairs =
        Core::GeometricSearch::global_collision_search(primitives_, predicates_, comm_, verbosity_);

    if (my_rank_ == 1)
    {
      EXPECT_EQ(pairs.size(), 2);

      EXPECT_EQ(pairs[0].lid_predicate, 0);
      EXPECT_EQ(pairs[0].gid_predicate, 3);
      EXPECT_EQ(pairs[0].gid_primitive, 1);
      EXPECT_EQ(pairs[0].pid_primitive, 0);

      EXPECT_EQ(pairs[1].lid_predicate, 0);
      EXPECT_EQ(pairs[1].gid_predicate, 3);
      EXPECT_EQ(pairs[1].gid_primitive, 0);
      EXPECT_EQ(pairs[1].pid_primitive, 0);
    }
    else
    {
      EXPECT_EQ(pairs.size(), 0);
    }
  }

  /**
   * Checking collision search with no primitives and no predicates
   */
  TEST_F(GeometricSearchDistributed, CollisionSearchNoPrimitivesNoPredicatesAll)
  {
    EXPECT_EQ(primitives_.size(), 0);
    EXPECT_EQ(predicates_.size(), 0);

    const auto pairs =
        Core::GeometricSearch::global_collision_search(primitives_, predicates_, comm_, verbosity_);

    EXPECT_EQ(pairs.size(), 0);
  }
}  // namespace

#endif

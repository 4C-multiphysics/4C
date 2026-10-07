// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_linalg_map.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_linalg_multi_vector.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_unittest_utils_assertions_test.hpp"
#include "4C_utils_exceptions.hpp"

#include <memory>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace
{

  TEST(ExchangeMapTest, Vector)
  {
    // initialize a communicator and the number of elements
    MPI_Comm comm = MPI_COMM_WORLD;
    int NumGlobalElements = 10;

    // set up a map
    Core::LinAlg::Map starting_map(NumGlobalElements, 0, comm);

    // create a vector
    auto vector = Core::LinAlg::Vector<double>(starting_map, true);

    const Epetra_Map& expected_map = starting_map.get_epetra_map();
    const Epetra_BlockMap& actual_map = vector.get_map().get_epetra_block_map();

    // check if underlying maps are identical.
    EXPECT_TRUE(actual_map.SameAs(expected_map));

    // create a new map with block map and different index base
    Epetra_BlockMap new_epetra_block_map(
        NumGlobalElements, 1, 1, Core::Communication::as_epetra_comm(comm));
    auto new_map = std::make_shared<Core::LinAlg::Map>(new_epetra_block_map);

    // replace map with vector function
    vector.replace_map(*new_map);

    // Ensure that the underlying epetra maps are identical
    EXPECT_EQ(vector.get_map().same_as(*new_map), true);

    // create a new map with index base 2
    new_map = std::make_shared<Core::LinAlg::Map>(NumGlobalElements, 2, comm);

    // replace the map based on the map of epetra vector
    EXPECT_EQ(vector.get_ref_of_epetra_vector().ReplaceMap(new_map->get_epetra_map()), 0);

    // compare result with our map wrapper
    EXPECT_TRUE(vector.get_map().same_as(*new_map));
  }

  TEST(ExchangeMapTest, MultiVector)
  {
    // initialize a communicator and the number of elements
    MPI_Comm comm = MPI_COMM_WORLD;
    int NumGlobalElements = 10;

    Core::LinAlg::Map starting_map(NumGlobalElements, 0, comm);

    // create a multi vector
    auto vector = Core::LinAlg::MultiVector<double>(starting_map, true);

    // check that the vector has the same epetra map
    EXPECT_TRUE(vector.get_map().same_as(starting_map));

    // create a new map with different index base
    auto new_map = std::make_shared<Core::LinAlg::Map>(NumGlobalElements, 1, comm);

    // check if map replacement is successfully
    EXPECT_NO_THROW(vector.replace_map(*new_map));

    // compare if the wrapper returns the correct map
    EXPECT_TRUE(vector.get_map().same_as(*new_map));

    // check that the epetra maps are same
    EXPECT_TRUE(vector.get_map().same_as(*new_map));

    // create a new map with index base 2
    new_map = std::make_shared<Core::LinAlg::Map>(NumGlobalElements, 2, comm);

    // replace the map based on the epetra vector
    EXPECT_EQ(static_cast<Epetra_MultiVector&>(vector.get_epetra_multi_vector())
                  .ReplaceMap(new_map->get_epetra_block_map()),
        0);

    // compare result with our map wrapper
    EXPECT_TRUE(vector.get_map().same_as(*new_map));
  }

  /// Tests the utility to verify whether all global indices of a map are contained within another
  /// map across all processors. Performs several tests including empty maps, maps with offset index
  /// bases and maps with non-unique GIDs on the current processor.
  TEST(LinAlgMapTest, MapContainsGlobalIdsOfAnotherMap)
  {
    // initialize the communicator
    MPI_Comm comm = MPI_COMM_WORLD;

    // Helper function to create a map by (homogeneously) distributing a vector of global ids onto
    // the processors. The last processor may receive more elements to fully distribute the entire
    // global_ids vector.
    auto setup_map_with_given_global_ids = [&comm](const std::vector<int>& global_ids)
    {
      std::vector<int> global_ids_on_this_rank;
      const int mpi_rank = Core::Communication::my_mpi_rank(comm);
      const int num_mpi_ranks = Core::Communication::num_mpi_ranks(comm);
      const int num_global_ids_on_this_rank = global_ids.size() / num_mpi_ranks;
      const int starting_index_on_this_rank = num_global_ids_on_this_rank * mpi_rank;
      if (mpi_rank < num_mpi_ranks - 1)
      {
        global_ids_on_this_rank.insert(global_ids_on_this_rank.end(),
            global_ids.begin() + starting_index_on_this_rank,
            global_ids.begin() + starting_index_on_this_rank + num_global_ids_on_this_rank);
      }
      else
      {
        global_ids_on_this_rank.insert(global_ids_on_this_rank.end(),
            global_ids.begin() + starting_index_on_this_rank, global_ids.end());
      }

      return Core::LinAlg::Map(-1, static_cast<int>(global_ids_on_this_rank.size()),
          global_ids_on_this_rank.data(), 0, comm);
    };

    //// large contiguous map
    Core::LinAlg::Map large_map(20, 0, comm);
    EXPECT_TRUE(large_map.contains_all_global_ids_of_map(large_map));

    // empty map tests
    Core::LinAlg::Map empty_map(0, 0, comm);
    EXPECT_TRUE(large_map.contains_all_global_ids_of_map(empty_map));
    EXPECT_TRUE(empty_map.contains_all_global_ids_of_map(empty_map));
    EXPECT_FALSE(empty_map.contains_all_global_ids_of_map(large_map));

    // small contiguous map contained within the large map
    Core::LinAlg::Map small_map(5, 0, comm);
    EXPECT_TRUE(large_map.contains_all_global_ids_of_map(small_map));

    // small contiguous map but with permuted global ids
    Core::LinAlg::Map small_map_permuted_global_ids =
        setup_map_with_given_global_ids({4, 2, 3, 0, 1});
    EXPECT_TRUE(large_map.contains_all_global_ids_of_map(small_map_permuted_global_ids));

    // small contiguous map with an offset index base contained within the large map
    Core::LinAlg::Map small_offset_map(5, 10, comm);
    EXPECT_TRUE(large_map.contains_all_global_ids_of_map(small_offset_map));

    // small contiguous map with an offset index base exceeding the maximum bound of the large map
    Core::LinAlg::Map small_offset_map_not_contained(5, 21, comm);
    EXPECT_FALSE(large_map.contains_all_global_ids_of_map(small_offset_map_not_contained));

    // even larger contiguous map
    Core::LinAlg::Map even_larger_map(30, 0, comm);
    EXPECT_FALSE(large_map.contains_all_global_ids_of_map(even_larger_map));
    EXPECT_TRUE(even_larger_map.contains_all_global_ids_of_map(large_map));

    // test failing import for an item: two maps matching in all criteria except for a single
    // global id
    Core::LinAlg::Map map_global_ids = setup_map_with_given_global_ids({0, 10, 20, 30, 40});
    Core::LinAlg::Map map_global_ids_alternative =
        setup_map_with_given_global_ids({0, 10, 21, 30, 40});
    EXPECT_FALSE(map_global_ids_alternative.contains_all_global_ids_of_map(map_global_ids));
    EXPECT_FALSE(map_global_ids.contains_all_global_ids_of_map(map_global_ids_alternative));

    // test various cases of non-unique gids inside the maps; initially some tests for unique
    // GIDs on the current processor
    EXPECT_TRUE(setup_map_with_given_global_ids({4, 0, 1, 2, 3}).unique_my_gids());
    EXPECT_TRUE((Core::Communication::num_mpi_ranks(comm) == 1) or
                setup_map_with_given_global_ids({4, 0, 1, 2, 3, 4})
                    .unique_my_gids());  // under the assumption of np=2 true; for np=1 false
    EXPECT_FALSE(setup_map_with_given_global_ids({4, 4, 4, 4})
            .unique_my_gids());  // holds for np = 1 and np = 2, but would not hold e.g., for np = 3
                                 // on a processor with a single element

    EXPECT_TRUE(setup_map_with_given_global_ids({4, 0, 2, 3, 1, 4})
            .contains_all_global_ids_of_map(setup_map_with_given_global_ids(
                {0, 1, 2, 3, 4})));  // superset map has non-unique gids on different processors
    EXPECT_FALSE(setup_map_with_given_global_ids({4, 0, 2, 3, 1, 4})
            .contains_all_global_ids_of_map(setup_map_with_given_global_ids({0, 1, 2, 3, 5})));

    EXPECT_TRUE(setup_map_with_given_global_ids({4, 0, 2, 3, 1, 4})
            .contains_all_global_ids_of_map(setup_map_with_given_global_ids(
                {4, 0, 1, 2, 3, 4})));  // both maps have non-unique gids on different processors
    EXPECT_FALSE(setup_map_with_given_global_ids({4, 0, 2, 3, 1, 4})
            .contains_all_global_ids_of_map(
                setup_map_with_given_global_ids({4, 0, 1, 2, 3, 4, 5})));

    EXPECT_TRUE(setup_map_with_given_global_ids({0, 1, 2, 3, 4, 5})
            .contains_all_global_ids_of_map(setup_map_with_given_global_ids(
                {4, 0, 1, 2, 3, 4})));  // subset map has non-unique gids on different processors
    EXPECT_FALSE(setup_map_with_given_global_ids({0, 1, 2, 3, 4, 5})
            .contains_all_global_ids_of_map(
                setup_map_with_given_global_ids({4, 0, 1, 2, 3, 4, 6})));
  }

}  // namespace

FOUR_C_NAMESPACE_CLOSE

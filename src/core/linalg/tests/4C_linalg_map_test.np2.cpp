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

#include <memory>

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
    auto vector = Core::LinAlg::Vector<double>(starting_map.get_epetra_map(), true);

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

}  // namespace

FOUR_C_NAMESPACE_CLOSE

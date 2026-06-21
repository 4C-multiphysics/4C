// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_linalg_blocksparsematrix.hpp"

#include "4C_linalg_mapextractor.hpp"

#include <array>
#include <memory>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace
{
  TEST(BlockSparseMatrixTest, MergeReservesSpaceForAllColumnBlocks)
  {
    const MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = Core::Communication::my_mpi_rank(comm);
    const std::array first_gids = {0, 7};
    const std::array other_gids = {1, 2, 3, 4, 5, 6};

    auto first = std::make_shared<Core::LinAlg::Map>(
        -1, rank == 0 ? std::span<const int>(first_gids) : std::span<const int>{}, 0, comm);

    auto others = std::make_shared<Core::LinAlg::Map>(
        -1, rank == 1 ? std::span<const int>(other_gids) : std::span<const int>{}, 0, comm);
    std::vector<std::shared_ptr<const Core::LinAlg::Map>> maps = {first, others};
    auto full_map = Core::LinAlg::MultiMapExtractor::merge_maps(maps);
    Core::LinAlg::MultiMapExtractor extractor(*full_map, maps);
    Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy> blocks(
        extractor, extractor, 2);

    if (rank == 0)
    {
      const double diagonal = 1.0;
      for (int gid : first_gids) blocks.matrix(0, 0).insert_global_values(gid, 1, &diagonal, &gid);
      const std::array values = {2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
      blocks.matrix(0, 1).insert_global_values(
          0, static_cast<int>(values.size()), values.data(), other_gids.data());
    }
    else
    {
      for (int gid : other_gids)
      {
        const double diagonal = 1.0;
        blocks.matrix(1, 1).insert_global_values(gid, 1, &diagonal, &gid);
      }
    }
    blocks.complete();

    const auto merged = blocks.merge();
    EXPECT_EQ(merged->num_global_nonzeros(), 14);
    if (rank == 0)
    {
      EXPECT_EQ(merged->num_allocated_global_entries(0), 7);
      EXPECT_EQ(merged->num_allocated_global_entries(7), 1);
      int count = 0;
      std::array<double, 7> values{};
      std::array<int, 7> columns{};
      merged->extract_global_row_copy(0, 7, count, values.data(), columns.data());
      EXPECT_EQ(count, 7);
      for (int i = 0; i < count; ++i)
        EXPECT_DOUBLE_EQ(values[i], columns[i] == 0 ? 1.0 : columns[i] + 1.0);
    }
  }
}  // namespace

FOUR_C_NAMESPACE_CLOSE

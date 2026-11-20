// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_linalg_utils_sparse_algebra_static_profile_test.hpp"

#include <array>

FOUR_C_NAMESPACE_OPEN

namespace
{
  // Rank one starts with no rows, but must receive a row wider than its local row maximum.
  TEST_F(SparseAlgebraStaticProfileTest, RedistributeRowsOntoRankWithNoSourceRows)
  {
    const int rank = Core::Communication::my_mpi_rank(comm_);
    const std::array source_rows = {0, 1};
    auto source_map = std::make_shared<Core::LinAlg::Map>(
        -1, rank == 0 ? std::span<const int>(source_rows) : std::span<const int>{}, 0, comm_);

    const int destination_row = rank;

    auto destination_map = std::make_shared<Core::LinAlg::Map>(
        -1, std::span<const int>(&destination_row, 1), 0, comm_);

    const std::array columns = {10, 11, 12};
    Core::LinAlg::Map domain_map(
        -1, rank == 0 ? std::span<const int>(columns) : std::span<const int>{}, 0, comm_);

    Core::LinAlg::SparseMatrix source(*source_map, 3, StorageProfile::static_profile);
    if (rank == 0)
    {
      source.assemble(7.0, 0, 10);
      for (int i = 0; i < 3; ++i) source.assemble(1.0 + i, 1, columns[i]);
    }
    source.complete(domain_map, *source_map);
    EXPECT_EQ(source.max_num_entries(), rank == 0 ? 3 : 0);
    EXPECT_EQ(source.global_max_num_entries(), 3);

    auto result = Core::LinAlg::redistribute(source, *destination_map, domain_map);
    EXPECT_TRUE(result->row_map().same_as(*destination_map));
    EXPECT_TRUE(result->domain_map().same_as(domain_map));
    int count = 0;
    std::array<double, 3> values{};
    std::array<int, 3> indices{};
    result->extract_global_row_copy(destination_row, 3, count, values.data(), indices.data());
    ASSERT_EQ(count, rank == 0 ? 1 : 3);
    for (int i = 0; i < count; ++i)
    {
      EXPECT_GE(indices[i], 10);
      EXPECT_LE(indices[i], 12);
      EXPECT_DOUBLE_EQ(values[i], rank == 0 ? 7.0 : indices[i] - 9.0);
    }
  }
}  // namespace

FOUR_C_NAMESPACE_CLOSE

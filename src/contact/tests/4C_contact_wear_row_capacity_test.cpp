// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_contact_wear_row_capacity.hpp"

#include "4C_linalg_map.hpp"
#include "4C_linalg_sparsematrix.hpp"

#include <map>
#include <set>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace
{
  TEST(WearRowCapacityTest, ImplicitWearColumnsRequireUnionWithGapColumns)
  {
    // Pure-structure input rejects implicit wear, so exercise its S-row sparsity directly.
    const std::map<int, double> gap = {{10, 1.0}, {11, 1.0}, {12, 1.0}};
    const std::map<int, double> wear = {{12, 2.0}, {13, 2.0}, {14, 2.0}};
    const auto columns = Wear::Internal::s_derivative_columns(gap, wear);
    EXPECT_EQ(columns, (std::set<int>{10, 11, 12, 13, 14}));

    const Core::LinAlg::Map row_map(1, 0, MPI_COMM_SELF);
    std::vector<int> gap_only_capacity = {static_cast<int>(gap.size())};
    Core::LinAlg::SparseMatrix gap_only(
        row_map, gap_only_capacity, Core::LinAlg::SparseMatrix::StorageProfile::static_profile);
    for (const auto& [column, value] : gap) gap_only.assemble(value, 0, column);
#ifdef FOUR_C_ENABLE_ASSERTIONS
    EXPECT_ANY_THROW(gap_only.assemble(wear.at(13), 0, 13));
#else
    // In Release, the Epetra insertion error is not thrown because assertions are disabled.
    gap_only.assemble(wear.at(13), 0, 13);
#endif
    EXPECT_EQ(gap_only.num_my_entries(0), static_cast<int>(gap.size()));

    std::vector<int> union_capacity = {static_cast<int>(columns.size())};
    Core::LinAlg::SparseMatrix with_wear(
        row_map, union_capacity, Core::LinAlg::SparseMatrix::StorageProfile::static_profile);
    for (const auto& [column, value] : gap) with_wear.assemble(value, 0, column);
    for (const auto& [column, value] : wear) with_wear.assemble(value, 0, column);
    EXPECT_EQ(with_wear.num_my_entries(0), 5);
    EXPECT_EQ(with_wear.num_allocated_global_entries(0), 5);
  }
}  // namespace

FOUR_C_NAMESPACE_CLOSE

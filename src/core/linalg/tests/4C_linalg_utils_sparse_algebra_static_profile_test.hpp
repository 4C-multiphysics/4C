// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINALG_UTILS_SPARSE_ALGEBRA_STATIC_PROFILE_TEST_HPP
#define FOUR_C_LINALG_UTILS_SPARSE_ALGEBRA_STATIC_PROFILE_TEST_HPP

#include <gtest/gtest.h>

#include "4C_comm_mpi_utils.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"

#include <memory>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace
{
  using StorageProfile = Core::LinAlg::SparseMatrix::StorageProfile;

  // Included by rank-specific translation units to run identical regressions on 1, 2, 3 and 4
  // ranks.
  class SparseAlgebraStaticProfileTest : public testing::Test
  {
   protected:
    MPI_Comm comm_ = MPI_COMM_WORLD;
  };

  // Split a static-profile matrix into 2x2 and 3x3 blocks, with one local row per block.
  // Each row has a column on every rank in every block. Check all column indices and values,
  // including off-rank columns, rather than only entries in the rank-local diagonal blocks.
  TEST_F(SparseAlgebraStaticProfileTest, SplitMatrixReservesRowCapacity)
  {
    const int rank = Core::Communication::my_mpi_rank(comm_);
    const int num_ranks = Core::Communication::num_mpi_ranks(comm_);
    for (const int num_blocks : {2, 3})
    {
      SCOPED_TRACE(num_blocks);
      std::vector<int> gids(num_blocks);
      for (int i = 0; i < num_blocks; ++i) gids[i] = rank * num_blocks + i;
      Core::LinAlg::Map row_map(-1, num_blocks, gids.data(), 0, comm_);
      Core::LinAlg::SparseMatrix matrix(
          row_map, num_blocks * num_ranks, StorageProfile::static_profile);
      std::vector<std::shared_ptr<const Core::LinAlg::Map>> maps;
      for (int i = 0; i < num_blocks; ++i)
      {
        maps.push_back(std::make_shared<Core::LinAlg::Map>(-1, 1, &gids[i], 0, comm_));
        for (int owner = 0; owner < num_ranks; ++owner)
          for (int j = 0; j < num_blocks; ++j)
            matrix.assemble(100.0 * owner + 10.0 * i + j + 1.0, gids[i], owner * num_blocks + j);
      }
      matrix.complete();
      Core::LinAlg::MultiMapExtractor extractor(row_map, maps);
      auto blocks = Core::LinAlg::split_matrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
          matrix, extractor, extractor);
      blocks->complete();
      for (int i = 0; i < num_blocks; ++i)
      {
        for (int j = 0; j < num_blocks; ++j)
        {
          SCOPED_TRACE(testing::Message() << "block=(" << i << ", " << j << ")");
          int count = 0;
          std::vector<int> columns(num_ranks);
          std::vector<double> values(num_ranks);
          (*blocks)(i, j).extract_global_row_copy(
              gids[i], num_ranks, count, values.data(), columns.data());
          ASSERT_EQ(count, num_ranks);
          for (int entry = 0; entry < count; ++entry)
          {
            EXPECT_GE(columns[entry], 0);
            EXPECT_LT(columns[entry], num_blocks * num_ranks);
            EXPECT_EQ(columns[entry] % num_blocks, j);
            EXPECT_DOUBLE_EQ(
                values[entry], 100.0 * (columns[entry] / num_blocks) + 10.0 * i + j + 1.0);
          }
        }
      }
    }
  }

  // Square a static-profile diagonal matrix with local diagonal entries 2 and 3.
  // Check both multiplication overloads with immediate and deferred completion:
  // the completed product must have the same maps and local diagonal entries 4 and 9.
  TEST_F(SparseAlgebraStaticProfileTest, MatrixProductCompletionWithStaticInputs)
  {
    const int rank = Core::Communication::my_mpi_rank(comm_);
    const std::vector<int> gids{2 * rank, 2 * rank + 1};
    Core::LinAlg::Map row_map(-1, 2, gids.data(), 0, comm_);
    Core::LinAlg::SparseMatrix matrix(row_map, 1, StorageProfile::static_profile);
    matrix.assemble(2.0, gids[0], gids[0]);
    matrix.assemble(3.0, gids[1], gids[1]);
    matrix.complete();

    for (const auto profile : {StorageProfile::dynamic_profile, StorageProfile::static_profile})
      for (const bool trans_a : {false, true})
        for (const bool trans_b : {false, true})
          for (const bool explicit_flags : {false, true})
            for (const bool complete : {false, true})
            {
              SCOPED_TRACE(testing::Message()
                           << "static=" << (profile == StorageProfile::static_profile)
                           << ", trans_a=" << trans_a << ", trans_b=" << trans_b
                           << ", explicit_flags=" << explicit_flags << ", complete=" << complete);
              auto product = explicit_flags ? Core::LinAlg::matrix_multiply(matrix, trans_a, matrix,
                                                  trans_b, false, true, complete, profile)
                                            : Core::LinAlg::matrix_multiply(matrix, trans_a, matrix,
                                                  trans_b, complete, profile);
              EXPECT_EQ(product->filled(), complete);
              EXPECT_EQ(product->explicit_dirichlet(), !explicit_flags);
              EXPECT_EQ(product->save_graph(), explicit_flags);
              if (!complete) product->complete(row_map, row_map);
              EXPECT_TRUE(product->range_map().same_as(row_map));
              EXPECT_TRUE(product->domain_map().same_as(row_map));
              EXPECT_EQ(product->num_my_nonzeros(), 2);
              Core::LinAlg::Vector<double> diagonal(row_map, true);
              product->extract_diagonal_copy(diagonal);
              EXPECT_DOUBLE_EQ(diagonal.local_values_as_span()[0], 4.0);
              EXPECT_DOUBLE_EQ(diagonal.local_values_as_span()[1], 9.0);
            }
  }

  // A single column in each input row can produce a wide outer product. Reserve only
  // the columns actually present in each resulting row, including empty local rows.
  TEST_F(SparseAlgebraStaticProfileTest, StaticProductOfRankOneBasis)
  {
    const int rank = Core::Communication::my_mpi_rank(comm_);
    const int num_ranks = Core::Communication::num_mpi_ranks(comm_);
    std::vector<int> rows(20);
    for (int i = 0; i < 20; ++i) rows[i] = 20 * rank + i;
    Core::LinAlg::Map row_map(-1, 20, rows.data(), 0, comm_);
    Core::LinAlg::Map coarse_map(1, 0, comm_);
    Core::LinAlg::SparseMatrix basis(row_map, 1, StorageProfile::static_profile);
    for (int i = 0; i < 19; ++i) basis.assemble(1.0, rows[i], 0);
    basis.complete(coarse_map, row_map);

    for (const bool explicit_flags : {false, true})
      for (const bool complete : {false, true})
      {
        SCOPED_TRACE(
            testing::Message() << "explicit_flags=" << explicit_flags << ", complete=" << complete);
        auto product = explicit_flags ? Core::LinAlg::matrix_multiply(basis, false, basis, true,
                                            true, false, complete, StorageProfile::static_profile)
                                      : Core::LinAlg::matrix_multiply(basis, false, basis, true,
                                            complete, StorageProfile::static_profile);
        EXPECT_EQ(product->filled(), complete);
        const auto graph = product->get_current_graph();
        for (int row = 0; row < 20; ++row)
          EXPECT_EQ(graph.num_allocated_local_indices(row), row == 19 ? 0 : 19 * num_ranks);
        if (!complete) product->complete(row_map, row_map);
        EXPECT_TRUE(product->domain_map().same_as(row_map));
        EXPECT_TRUE(product->range_map().same_as(row_map));
        for (int row = 0; row < 20; ++row)
        {
          int count = 0;
          std::vector<int> columns(19 * num_ranks);
          std::vector<double> values(19 * num_ranks);
          product->extract_global_row_copy(
              rows[row], columns.size(), count, values.data(), columns.data());
          EXPECT_EQ(count, row == 19 ? 0 : 19 * num_ranks);
          std::vector<bool> seen(20 * num_ranks, false);
          for (int i = 0; i < count; ++i)
          {
            ASSERT_GE(columns[i], 0);
            ASSERT_LT(columns[i], 20 * num_ranks);
            EXPECT_NE(columns[i] % 20, 19);
            EXPECT_FALSE(seen[columns[i]]);
            seen[columns[i]] = true;
            EXPECT_DOUBLE_EQ(values[i], 1.0);
          }
        }
      }
  }

  // Rank zero owns both input rows; every rank owns one output row and one domain entry.
  // A^T B must contain contributions on all ranks, including those with no input rows.
  // Check every product entry and its action on an all-ones vector for both overloads
  // and both completion modes. The one-rank run also covers the serial case.
  TEST_F(SparseAlgebraStaticProfileTest, MatrixProductWithEmptyLocalInputs)
  {
    const int rank = Core::Communication::my_mpi_rank(comm_);
    const int num_ranks = Core::Communication::num_mpi_ranks(comm_);
    const std::vector<int> slave_gids = rank == 0 ? std::vector<int>{0, 1} : std::vector<int>{};
    Core::LinAlg::Map slave_map(
        -1, static_cast<int>(slave_gids.size()), slave_gids.data(), 0, comm_);
    const int master_gid = 10 + rank;
    Core::LinAlg::Map master_map(-1, 1, &master_gid, 0, comm_);
    const int domain_gid = 20 + rank;
    Core::LinAlg::Map domain_map(-1, 1, &domain_gid, 0, comm_);

    Core::LinAlg::SparseMatrix a(slave_map, num_ranks, StorageProfile::static_profile);
    Core::LinAlg::SparseMatrix b(slave_map, num_ranks, StorageProfile::static_profile);
    for (const int row : slave_gids)
      for (int owner = 0; owner < num_ranks; ++owner)
      {
        a.assemble(1.0 + 2.0 * row + owner, row, 10 + owner);
        b.assemble(5.0 + 2.0 * row + owner, row, 20 + owner);
      }
    a.complete(master_map, slave_map);
    b.complete(domain_map, slave_map);

    for (const auto profile : {StorageProfile::dynamic_profile, StorageProfile::static_profile})
      for (const bool explicit_flags : {false, true})
        for (const bool complete : {false, true})
        {
          SCOPED_TRACE(testing::Message()
                       << "static=" << (profile == StorageProfile::static_profile)
                       << ", explicit_flags=" << explicit_flags << ", complete=" << complete);
          auto product =
              explicit_flags
                  ? Core::LinAlg::matrix_multiply(a, true, b, false, true, false, complete, profile)
                  : Core::LinAlg::matrix_multiply(a, true, b, false, complete, profile);
          EXPECT_EQ(product->filled(), complete);
          if (!complete) product->complete(domain_map, master_map);
          EXPECT_TRUE(product->range_map().same_as(master_map));
          EXPECT_TRUE(product->domain_map().same_as(domain_map));
          EXPECT_EQ(product->num_my_nonzeros(), num_ranks);

          int count = 0;
          std::vector<int> columns(num_ranks);
          std::vector<double> values(num_ranks);
          product->extract_global_row_copy(
              master_gid, num_ranks, count, values.data(), columns.data());
          ASSERT_EQ(count, num_ranks);
          for (int entry = 0; entry < count; ++entry)
          {
            const int owner = columns[entry] - 20;
            EXPECT_GE(owner, 0);
            EXPECT_LT(owner, num_ranks);
            EXPECT_DOUBLE_EQ(
                values[entry], (1.0 + rank) * (5.0 + owner) + (3.0 + rank) * (7.0 + owner));
          }

          Core::LinAlg::Vector<double> x(domain_map, true);
          x.put_scalar(1.0);
          Core::LinAlg::Vector<double> y(master_map, true);
          product->multiply(false, x, y);
          double expected = 0.0;
          for (int column = 0; column < num_ranks; ++column)
            expected += (1.0 + rank) * (5.0 + column) + (3.0 + rank) * (7.0 + column);
          EXPECT_DOUBLE_EQ(y.local_values_as_span()[0], expected);
        }
  }
  // CRS matrices support both storage profiles; FE matrices support dynamic storage.
  // The storage-profile overload must also preserve the requested constructor flags.
  TEST_F(SparseAlgebraStaticProfileTest, StorageProfilesSupportMatrixTypes)
  {
    const int rank = Core::Communication::my_mpi_rank(comm_);
    const std::vector<int> gids{2 * rank, 2 * rank + 1};
    Core::LinAlg::Map row_map(-1, 2, gids.data(), 0, comm_);
    for (const auto matrix_type :
        {Core::LinAlg::SparseMatrix::CRS_MATRIX, Core::LinAlg::SparseMatrix::FE_MATRIX})
    {
      for (const auto profile : {StorageProfile::dynamic_profile, StorageProfile::static_profile})
      {
        if (matrix_type == Core::LinAlg::SparseMatrix::FE_MATRIX &&
            profile == StorageProfile::static_profile)
          continue;
        SCOPED_TRACE(testing::Message() << "matrix_type=" << matrix_type << ", static_profile="
                                        << (profile == StorageProfile::static_profile));
        const int capacity = profile == StorageProfile::static_profile ? 2 : 1;
        Core::LinAlg::SparseMatrix matrix(row_map, capacity, profile, false, true, matrix_type);
        EXPECT_FALSE(matrix.filled());
        EXPECT_FALSE(matrix.explicit_dirichlet());
        EXPECT_TRUE(matrix.save_graph());
        EXPECT_EQ(matrix.get_matrixtype(), matrix_type);
        matrix.set_value(2.0, gids[0], gids[0]);
        matrix.set_value(1.0, gids[0], gids[1]);
        matrix.set_value(4.0, gids[1], gids[0]);
        matrix.set_value(3.0, gids[1], gids[1]);
        matrix.complete();
        EXPECT_EQ(matrix.num_my_nonzeros(), 4);
        Core::LinAlg::Vector<double> x(row_map, true);
        x.put_scalar(1.0);
        Core::LinAlg::Vector<double> y(row_map, true);
        matrix.multiply(false, x, y);
        EXPECT_DOUBLE_EQ(y.local_values_as_span()[0], 3.0);
        EXPECT_DOUBLE_EQ(y.local_values_as_span()[1], 7.0);
      }
    }
  }

  // Epetra_FECrsMatrix's boolean constructor argument controls nonlocal assembly, not storage.
  TEST_F(SparseAlgebraStaticProfileTest, RejectStaticProfileForFEMatrix)
  {
    const int rank = Core::Communication::my_mpi_rank(comm_);
    Core::LinAlg::Map row_map(-1, 1, &rank, 0, comm_);
    EXPECT_ANY_THROW(Core::LinAlg::SparseMatrix(row_map, 1, StorageProfile::static_profile, true,
        false, Core::LinAlg::SparseMatrix::FE_MATRIX));
  }

  // Each rank contributes to its own row and the next rank's row; GlobalAssemble must retain
  // the off-rank contributions. On one rank, the contributions land in the same local row.
  TEST_F(SparseAlgebraStaticProfileTest, FEMatrixGathersOffRankContributions)
  {
    const int rank = Core::Communication::my_mpi_rank(comm_);
    const int num_ranks = Core::Communication::num_mpi_ranks(comm_);
    Core::LinAlg::Map row_map(-1, 1, &rank, 0, comm_);
    Core::LinAlg::SparseMatrix matrix(row_map, 1, StorageProfile::dynamic_profile, true, false,
        Core::LinAlg::SparseMatrix::FE_MATRIX);
    matrix.fe_assemble(2.0, rank, rank);
    const int next_row = (rank + 1) % num_ranks;
    matrix.fe_assemble(3.0, next_row, next_row);
    matrix.complete();

    EXPECT_EQ(matrix.num_my_nonzeros(), 1);
    Core::LinAlg::Vector<double> diagonal(row_map, true);
    matrix.extract_diagonal_copy(diagonal);
    EXPECT_DOUBLE_EQ(diagonal.local_values_as_span()[0], 5.0);
  }

  // The pre-existing constructor must still allow insertion beyond its initial row estimate.
  TEST_F(SparseAlgebraStaticProfileTest, LegacyConstructorAllowsRowGrowth)
  {
    const int rank = Core::Communication::my_mpi_rank(comm_);
    const std::vector<int> gids{2 * rank, 2 * rank + 1};
    Core::LinAlg::Map row_map(-1, 2, gids.data(), 0, comm_);
    for (const auto matrix_type :
        {Core::LinAlg::SparseMatrix::CRS_MATRIX, Core::LinAlg::SparseMatrix::FE_MATRIX})
    {
      SCOPED_TRACE(matrix_type);
      Core::LinAlg::SparseMatrix matrix(row_map, 1, true, false, matrix_type);
      matrix.set_value(2.0, gids[0], gids[0]);
      matrix.set_value(1.0, gids[0], gids[1]);
      matrix.set_value(3.0, gids[1], gids[1]);
      matrix.complete();
      EXPECT_TRUE(matrix.explicit_dirichlet());
      EXPECT_FALSE(matrix.save_graph());
      EXPECT_EQ(matrix.num_my_nonzeros(), 3);
    }
  }
}  // namespace

FOUR_C_NAMESPACE_CLOSE

#endif

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"

#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_unittest_utils_support_files_test.hpp"

#include <Epetra_MpiComm.h>
#include <EpetraExt_CrsMatrixIn.h>

FOUR_C_NAMESPACE_OPEN

namespace
{
  class SparseAlgebraManipulationTest : public testing::Test
  {
   public:
    //! Testing parameters
    MPI_Comm comm_;

   protected:
    SparseAlgebraManipulationTest() { comm_ = MPI_COMM_WORLD; }
  };

  /** The test setup is based on a simple 1d poisson problem with the given matrix "poisson1d.mm"
   * constructed with MATLAB, having the well known [1 -2 1] tri-diagonal entries.
   *
   * Due to the threshold tol=1.1, the off-diagonal values are dropped resulting in a diagonal
   * matrix with -2 values on the main diagonal.
   */
  TEST_F(SparseAlgebraManipulationTest, ThresholdMatrix1)
  {
    Epetra_CrsMatrix* A;

    int err = EpetraExt::MatrixMarketFileToCrsMatrix(
        TESTING::get_support_file_path("test_matrices/poisson1d.mm").c_str(),
        Core::Communication::as_epetra_comm(comm_), A);
    if (err != 0) FOUR_C_THROW("Matrix read failed.");
    std::shared_ptr<Epetra_CrsMatrix> A_crs = Core::Utils::shared_ptr_from_ref(*A);
    Core::LinAlg::SparseMatrix A_sparse(A_crs, Core::LinAlg::Copy);

    const double tol = 1.1;
    std::shared_ptr<Core::LinAlg::SparseMatrix> A_thresh =
        Core::LinAlg::threshold_matrix(A_sparse, tol);

    // Check for global entries
    const int A_thresh_nnz = A_thresh->epetra_matrix()->NumGlobalNonzeros();
    EXPECT_EQ(A_thresh_nnz, 20);

    // Check for overall norm of matrix
    EXPECT_NEAR(A_thresh->norm_frobenius(), 8.944271909999159, 1e-12);
  }

  /** The test setup is based on the simple matrix  "filter.mm" constructed with MATLAB.
   *
   * Due to the threshold all values smaller than 1e-5 are dropped.
   */
  TEST_F(SparseAlgebraManipulationTest, ThresholdMatrix2)
  {
    Epetra_CrsMatrix* A;

    int err = EpetraExt::MatrixMarketFileToCrsMatrix(
        TESTING::get_support_file_path("test_matrices/filter.mm").c_str(),
        Core::Communication::as_epetra_comm(comm_), A);
    if (err != 0) FOUR_C_THROW("Matrix read failed.");
    std::shared_ptr<Epetra_CrsMatrix> A_crs = Core::Utils::shared_ptr_from_ref(*A);
    Core::LinAlg::SparseMatrix A_sparse(A_crs, Core::LinAlg::Copy);

    const double tol = 1e-5;
    std::shared_ptr<Core::LinAlg::SparseMatrix> A_thresh =
        Core::LinAlg::threshold_matrix(A_sparse, tol);

    // Check for global entries
    const int A_thresh_nnz = A_thresh->epetra_matrix()->NumGlobalNonzeros();
    EXPECT_EQ(A_thresh_nnz, 13);

    // Check for overall norm of matrix
    EXPECT_NEAR(A_thresh->norm_frobenius(), 7.549834435270750, 1e-12);
  }

  /** The test setup is based on the simple matrix  "filter.mm" constructed with MATLAB.
   *
   * Due to the threshold all indices based on values smaller than 1e-5 are dropped from the graph.
   */
  TEST_F(SparseAlgebraManipulationTest, ThresholdMatrixGraph)
  {
    Epetra_CrsMatrix* A;

    int err = EpetraExt::MatrixMarketFileToCrsMatrix(
        TESTING::get_support_file_path("test_matrices/filter.mm").c_str(),
        Core::Communication::as_epetra_comm(comm_), A);
    if (err != 0) FOUR_C_THROW("Matrix read failed.");
    std::shared_ptr<Epetra_CrsMatrix> A_crs = Core::Utils::shared_ptr_from_ref(*A);
    Core::LinAlg::SparseMatrix A_sparse(A_crs, Core::LinAlg::Copy);

    const double tol = 1e-5;
    std::shared_ptr<Core::LinAlg::Graph> G = Core::LinAlg::threshold_matrix_graph(A_sparse, tol);

    // Check for global entries
    const int A_thresh_nnz = G->NumGlobalNonzeros();
    EXPECT_EQ(A_thresh_nnz, 13);
  }

  /** The test setup is based on the one-dimensional Poisson matrix "poisson1d.mm" constructed
   * with MATLAB.
   *
   * Taking the powers of a matrix and therefore enriching it's graph (a little bit related to
   * MATLAB's: B = mpower(A, power)
   */
  TEST_F(SparseAlgebraManipulationTest, EnrichMatrixGraph1)
  {
    Epetra_CrsMatrix* A;

    int err = EpetraExt::MatrixMarketFileToCrsMatrix(
        TESTING::get_support_file_path("test_matrices/poisson1d.mm").c_str(),
        Core::Communication::as_epetra_comm(comm_), A);
    if (err != 0) FOUR_C_THROW("Matrix read failed.");
    std::shared_ptr<Epetra_CrsMatrix> A_crs = Core::Utils::shared_ptr_from_ref(*A);
    Core::LinAlg::SparseMatrix A_sparse(A_crs, Core::LinAlg::Copy);

    {
      const int power = 0;
      std::shared_ptr<Core::LinAlg::Graph> graph_enriched =
          Core::LinAlg::enrich_matrix_graph(A_sparse, power);

      // Check for global entries
      EXPECT_EQ(graph_enriched->NumGlobalNonzeros(), 58);
    }

    {
      const int power = -3;
      std::shared_ptr<Core::LinAlg::Graph> graph_enriched =
          Core::LinAlg::enrich_matrix_graph(A_sparse, power);

      // Check for global entries
      EXPECT_EQ(graph_enriched->NumGlobalNonzeros(), 58);
    }

    {
      const int power = 1;
      std::shared_ptr<Core::LinAlg::Graph> graph_enriched =
          Core::LinAlg::enrich_matrix_graph(A_sparse, power);

      // Check for global entries
      EXPECT_EQ(graph_enriched->NumGlobalNonzeros(), 58);
    }

    {
      const int power = 2;
      std::shared_ptr<Core::LinAlg::Graph> graph_enriched =
          Core::LinAlg::enrich_matrix_graph(A_sparse, power);

      // Check for global entries
      EXPECT_EQ(graph_enriched->NumGlobalNonzeros(), 94);
    }

    {
      const int power = 3;
      std::shared_ptr<Core::LinAlg::Graph> graph_enriched =
          Core::LinAlg::enrich_matrix_graph(A_sparse, power);

      // Check for global entries
      EXPECT_EQ(graph_enriched->NumGlobalNonzeros(), 128);
    }

    {
      const int power = 4;
      std::shared_ptr<Core::LinAlg::Graph> graph_enriched =
          Core::LinAlg::enrich_matrix_graph(A_sparse, power);

      // Check for global entries
      EXPECT_EQ(graph_enriched->NumGlobalNonzeros(), 160);
    }
  }

  TEST_F(SparseAlgebraManipulationTest, EnrichMatrixGraph2)
  {
    Epetra_CrsMatrix* A;

    int err = EpetraExt::MatrixMarketFileToCrsMatrix(
        TESTING::get_support_file_path("test_matrices/beam.mm").c_str(),
        Core::Communication::as_epetra_comm(comm_), A);
    if (err != 0) FOUR_C_THROW("Matrix read failed.");
    std::shared_ptr<Epetra_CrsMatrix> A_crs = Core::Utils::shared_ptr_from_ref(*A);
    Core::LinAlg::SparseMatrix A_sparse(A_crs, Core::LinAlg::Copy);

    {
      const int power = 3;
      std::shared_ptr<Core::LinAlg::Graph> graph_enriched =
          Core::LinAlg::enrich_matrix_graph(A_sparse, power);

      // Check for global entries
      EXPECT_EQ(graph_enriched->NumGlobalNonzeros(), 228400);
    }
  }
}  // namespace

FOUR_C_NAMESPACE_CLOSE

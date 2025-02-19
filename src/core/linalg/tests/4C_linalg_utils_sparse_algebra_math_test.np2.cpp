// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_linalg_utils_sparse_algebra_math.hpp"

#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_unittest_utils_support_files_test.hpp"

#include <Epetra_MpiComm.h>
#include <EpetraExt_CrsMatrixIn.h>

FOUR_C_NAMESPACE_OPEN

namespace
{
  class SparseAlgebraMathTest : public testing::Test
  {
   public:
    //! Testing parameters
    MPI_Comm comm_;

   protected:
    SparseAlgebraMathTest() { comm_ = MPI_COMM_WORLD; }
  };

  /** The test setup is based on a simple 1d poisson problem with the given matrix "poisson1d.mm"
   * constructed with MATLAB, having the well known [1 -2 1] tri-diagonal entries.
   *
   * All test results can be reconstructed loading the matrix into MATLAB and using the command
   * A_inverse = matrix_sparse_inverse(A, A), by using the given MATLAB script.
   */
  TEST_F(SparseAlgebraMathTest, MatrixSparseInverse1)
  {
    Epetra_CrsMatrix* A;

    int err = EpetraExt::MatrixMarketFileToCrsMatrix(
        TESTING::get_support_file_path("test_matrices/poisson1d.mm").c_str(),
        Core::Communication::as_epetra_comm(comm_), A);
    if (err != 0) FOUR_C_THROW("Matrix read failed.");
    std::shared_ptr<Epetra_CrsMatrix> A_crs = Core::Utils::shared_ptr_from_ref(*A);
    Core::LinAlg::SparseMatrix A_sparse(A_crs, Core::LinAlg::Copy);

    std::shared_ptr<Core::LinAlg::SparseMatrix> A_inverse = Core::LinAlg::matrix_sparse_inverse(
        A_sparse, std::make_shared<Core::LinAlg::Graph>(A->Graph()));

    // Check for global entries
    const int A_sparse_nnz = A_sparse.epetra_matrix()->NumGlobalNonzeros();
    const int A_inverse_nnz = A_inverse->epetra_matrix()->NumGlobalNonzeros();
    EXPECT_EQ(A_sparse_nnz, A_inverse_nnz);

    // Check for overall norm of matrix inverse
    EXPECT_NEAR(A_inverse->norm_frobenius(), 3.037251711528645, 1e-12);
  }

  /** The test setup is based on the given nonsymmetric matrix "nonsym.mm" constructed with MATLAB.
   *  All test results can be reconstructed loading the matrix into MATLAB and using the command
   *  A_inverse = inv(A).
   *  A is given as:
   *    10     0     5     0     1
   *     0    20     0    10     0
   *     0     0    30     0    20
   *     0     0     0    40     0
   *     0     0     0     0    59
   *  With it's sparse inverse A_inverse:
   *  0.1000         0   -0.0167         0    0.0040
   *      0    0.0500         0   -0.0125         0
   *      0         0    0.0333         0   -0.0113
   *      0         0         0    0.0250         0
   *      0         0         0         0    0.0169
   */
  TEST_F(SparseAlgebraMathTest, MatrixSparseInverse2)
  {
    Epetra_CrsMatrix* A;

    int err = EpetraExt::MatrixMarketFileToCrsMatrix(
        TESTING::get_support_file_path("test_matrices/nonsym.mm").c_str(),
        Core::Communication::as_epetra_comm(comm_), A);
    if (err != 0) FOUR_C_THROW("Matrix read failed.");
    std::shared_ptr<Epetra_CrsMatrix> A_crs = Core::Utils::shared_ptr_from_ref(*A);
    Core::LinAlg::SparseMatrix A_sparse(A_crs, Core::LinAlg::Copy);

    std::shared_ptr<Core::LinAlg::SparseMatrix> A_inverse = Core::LinAlg::matrix_sparse_inverse(
        A_sparse, std::make_shared<Core::LinAlg::Graph>(A->Graph()));

    // Check for global entries
    const int A_sparse_nnz = A_sparse.epetra_matrix()->NumGlobalNonzeros();
    const int A_inverse_nnz = A_inverse->epetra_matrix()->NumGlobalNonzeros();
    EXPECT_EQ(A_sparse_nnz, A_inverse_nnz);

    // Check for overall norm of matrix inverse
    EXPECT_NEAR(A_inverse->norm_frobenius(), 0.1235706050986417, 1e-12);

    // Check fist matrix row of inverse
    if (Core::Communication::my_mpi_rank(comm_) == 0)
    {
      double* values;
      int length;
      A_inverse->epetra_matrix()->ExtractMyRowView(0, length, values);

      EXPECT_NEAR(values[0], 0.1, 1e-12);
      EXPECT_NEAR(values[1], -0.016666666666666673, 1e-12);
      EXPECT_NEAR(values[2], 0.0046666666666666688, 1e-12);
    }
  }

  /** The test setup is based on a beam discretization with the given block-diagonal matrix
   * "beam.mm", as they appear in beam-solid volume meshtying.
   *
   * In a first step the matrix graph of A is sparsified, then a sparse inverse of A is calculated
   * on that sparsity pattern and finally the inverse matrix is again filtered. The algorithmic
   * procedure is loosely based on ParaSails and the following publications:
   *
   * E. Chow: Parallel implementation and practical use of sparse approximate inverse
   * preconditioners with a priori sparsity patterns.
   * The International Journal of High Performance Computing Applications, 15(1):56-74, 2001,
   * https://doi.org/10.1177/109434200101500106
   *
   * E. Chow: A Priori Sparsity Patterns for Parallel Sparse Approximate Inverse Preconditioners.
   * SIAM Journal on Scientific Computing, 21(5):1804-1822, 2000,
   * https://doi.org/10.1137/S106482759833913X
   */
  TEST_F(SparseAlgebraMathTest, MatrixSparseInverse3)
  {
    Epetra_CrsMatrix* A;

    int err = EpetraExt::MatrixMarketFileToCrsMatrix(
        TESTING::get_support_file_path("test_matrices/beam.mm").c_str(),
        Core::Communication::as_epetra_comm(comm_), A);
    if (err != 0) FOUR_C_THROW("Matrix read failed.");
    std::shared_ptr<Epetra_CrsMatrix> A_crs = Core::Utils::shared_ptr_from_ref(*A);
    Core::LinAlg::SparseMatrix A_sparse(A_crs, Core::LinAlg::Copy);

    {
      const double tol = 1e-8;
      std::shared_ptr<Core::LinAlg::Graph> sparsity_pattern =
          Core::LinAlg::threshold_matrix_graph(A_sparse, tol);
      std::shared_ptr<Core::LinAlg::SparseMatrix> A_inverse =
          Core::LinAlg::matrix_sparse_inverse(A_sparse, sparsity_pattern);
      std::shared_ptr<Core::LinAlg::SparseMatrix> A_thresh =
          Core::LinAlg::threshold_matrix(*A_inverse, tol);

      // Check for global entries
      const int A_inverse_nnz = A_inverse->epetra_matrix()->NumGlobalNonzeros();
      // Note: the number of entries lower than a tolerance is not necessarily deterministic
      EXPECT_NEAR(A_inverse_nnz, 115760, 10);

      // Check for overall norm of matrix inverse
      constexpr double expected_frobenius_norm = 8.31688788510637e+06;
      EXPECT_NEAR(
          A_inverse->norm_frobenius(), expected_frobenius_norm, expected_frobenius_norm * 1e-12);
    }

    {
      const double tol = 1e-10;
      const int power = 3;

      std::shared_ptr<Core::LinAlg::Graph> sparsity_pattern =
          std::make_shared<Core::LinAlg::Graph>(A->Graph());

      std::shared_ptr<Core::LinAlg::SparseMatrix> A_thresh =
          Core::LinAlg::threshold_matrix(A_sparse, tol);
      std::shared_ptr<Core::LinAlg::Graph> sparsity_pattern_enriched =
          Core::LinAlg::enrich_matrix_graph(*A_thresh, power);
      std::shared_ptr<Core::LinAlg::SparseMatrix> A_inverse =
          Core::LinAlg::matrix_sparse_inverse(A_sparse, sparsity_pattern_enriched);
      A_thresh = Core::LinAlg::threshold_matrix(*A_inverse, tol);

      // Check for global entries
      const int A_inverse_nnz = A_thresh->epetra_matrix()->NumGlobalNonzeros();
      // Note: the number of entries lower than a tolerance is not necessarily deterministic
      EXPECT_NEAR(A_inverse_nnz, 228388, 10);

      // Check for overall norm of matrix inverse
      constexpr double expected_frobenius_norm = 1.1473820881252188e+07;
      EXPECT_NEAR(
          A_thresh->norm_frobenius(), expected_frobenius_norm, expected_frobenius_norm * 1e-12);
    }
  }
}  // namespace

FOUR_C_NAMESPACE_CLOSE

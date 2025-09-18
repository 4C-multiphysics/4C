// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_linalg_utils_sparse_algebra_math.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_unittest_utils_support_files_test.hpp"

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
    Core::LinAlg::SparseMatrix A_sparse(A_crs, Core::LinAlg::DataAccess::Copy);

    std::shared_ptr<Core::LinAlg::SparseMatrix> A_inverse = Core::LinAlg::matrix_sparse_inverse(
        A_sparse, std::make_shared<Core::LinAlg::Graph>(A->Graph()));

    // Check for global entries
    const int A_sparse_nnz = A_sparse.num_global_nonzeros();
    const int A_inverse_nnz = A_inverse->num_global_nonzeros();
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
    Core::LinAlg::SparseMatrix A_sparse(A_crs, Core::LinAlg::DataAccess::Copy);

    std::shared_ptr<Core::LinAlg::SparseMatrix> A_inverse = Core::LinAlg::matrix_sparse_inverse(
        A_sparse, std::make_shared<Core::LinAlg::Graph>(A->Graph()));

    // Check for global entries
    const int A_sparse_nnz = A_sparse.num_global_nonzeros();
    const int A_inverse_nnz = A_inverse->num_global_nonzeros();
    EXPECT_EQ(A_sparse_nnz, A_inverse_nnz);

    // Check for overall norm of matrix inverse
    EXPECT_NEAR(A_inverse->norm_frobenius(), 0.1235706050986417, 1e-12);

    // Check fist matrix row of inverse
    if (Core::Communication::my_mpi_rank(comm_) == 0)
    {
      double* values;
      int* indices;
      int length;
      A_inverse->extract_my_row_view(0, length, values, indices);

      EXPECT_NEAR(values[0], 0.1, 1e-12);
      EXPECT_NEAR(values[1], -0.016666666666666673, 1e-12);
      EXPECT_NEAR(values[2], 0.0046666666666666688, 1e-12);
    }
  }

  /** The test setup is based on a beam discretization with the given block-diagonal matrix
   * "beamI.mm", as they appear in beam-solid volume meshtying regularized by a penalty approach.
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
        TESTING::get_support_file_path("test_matrices/beamI.mm").c_str(),
        Core::Communication::as_epetra_comm(comm_), A);
    if (err != 0) FOUR_C_THROW("Matrix read failed.");
    std::shared_ptr<Epetra_CrsMatrix> A_crs = Core::Utils::shared_ptr_from_ref(*A);
    Core::LinAlg::SparseMatrix A_sparse(A_crs, Core::LinAlg::DataAccess::Copy);

    {
      const double tol = 1e-8;
      std::shared_ptr<Core::LinAlg::Graph> sparsity_pattern =
          Core::LinAlg::threshold_matrix_graph(A_sparse, tol);
      std::shared_ptr<Core::LinAlg::SparseMatrix> A_inverse =
          Core::LinAlg::matrix_sparse_inverse(A_sparse, sparsity_pattern);
      std::shared_ptr<Core::LinAlg::SparseMatrix> A_thresh =
          Core::LinAlg::threshold_matrix(*A_inverse, tol);

      // Check for global entries
      const int A_inverse_nnz = A_inverse->num_global_nonzeros();
      // Note: the number of entries lower than a tolerance is not necessarily deterministic
      EXPECT_NEAR(A_inverse_nnz, 115760, 10);

      // Check for overall norm of matrix inverse
      constexpr double expected_frobenius_norm = 8.31688788510637e+06;
      EXPECT_NEAR(
          A_inverse->norm_frobenius(), expected_frobenius_norm, expected_frobenius_norm * 1e-10);
    }

    {
      const double tol = 1e-10;
      const int power = 3;

      Core::LinAlg::Graph sparsity_pattern(A->Graph());

      std::shared_ptr<Core::LinAlg::SparseMatrix> A_thresh =
          Core::LinAlg::threshold_matrix(A_sparse, tol);
      std::shared_ptr<Core::LinAlg::Graph> sparsity_pattern_enriched =
          Core::LinAlg::enrich_matrix_graph(*A_thresh, power);
      std::shared_ptr<Core::LinAlg::SparseMatrix> A_inverse =
          Core::LinAlg::matrix_sparse_inverse(A_sparse, sparsity_pattern_enriched);
      A_thresh = Core::LinAlg::threshold_matrix(*A_inverse, tol);

      // Check for global entries
      const int A_inverse_nnz = A_thresh->num_global_nonzeros();
      // Note: the number of entries lower than a tolerance is not necessarily deterministic
      EXPECT_NEAR(A_inverse_nnz, 228388, 10);

      // Check for overall norm of matrix inverse
      constexpr double expected_frobenius_norm = 1.1473820881252188e+07;
      EXPECT_NEAR(
          A_thresh->norm_frobenius(), expected_frobenius_norm, expected_frobenius_norm * 1e-10);
    }
  }

  /** The test setup is based on a beam discretization with Euler-Bernoulli beam elements.
   * The underlying discretized operator is singular as it stemms from a pure Neumann problem.
   * A normal calculation of a sparse inverse on such kind of matrix is ill-defined and thus throws.
   * By projecting the operator in a space explicitly not containing the rigid body modes / null
   * space the problem can be shifted to be non-singular, but highly ill-conditioned. By introducing
   * an a-priori diagonal perturbation, the Eigenvalues are shifted minimally to provide better
   * conditioning and thus be able to calculate an inverse of the operator.
   */
  TEST_F(SparseAlgebraMathTest, MatrixSparseInverse4)
  {
    // Try to invert pure Neumann problem, this should fail as the matrix is singular.
    {
      Epetra_CrsMatrix* A;

      int err = EpetraExt::MatrixMarketFileToCrsMatrix(
          TESTING::get_support_file_path("test_matrices/beamII.mm").c_str(),
          Core::Communication::as_epetra_comm(comm_), A);
      if (err != 0) FOUR_C_THROW("Matrix read failed.");
      std::shared_ptr<Epetra_CrsMatrix> A_crs = Core::Utils::shared_ptr_from_ref(*A);
      Core::LinAlg::SparseMatrix A_sparse(A_crs, Core::LinAlg::DataAccess::Copy);

      const double tol = 1e-14;
      const int power = 4;

      Core::LinAlg::Graph sparsity_pattern(A->Graph());

      std::shared_ptr<Core::LinAlg::SparseMatrix> A_thresh =
          Core::LinAlg::threshold_matrix(A_sparse, tol);
      std::shared_ptr<Core::LinAlg::Graph> sparsity_pattern_enriched =
          Core::LinAlg::enrich_matrix_graph(*A_thresh, power);

      Core::LinAlg::OptionsSparseMatrixInverse options;
      options.alpha = 1e-5;
      options.rho = 1.01;

      EXPECT_ANY_THROW(
          Core::LinAlg::matrix_sparse_inverse(A_sparse, sparsity_pattern_enriched, options));
    }

    // Try to invert pure Neumann problem, this should succeed as we use a projected operator and
    // a-priori diagonal perturbation
    {
      Epetra_CrsMatrix* A;

      int err = EpetraExt::MatrixMarketFileToCrsMatrix(
          TESTING::get_support_file_path("test_matrices/beamII_projected.mm").c_str(),
          Core::Communication::as_epetra_comm(comm_), A);
      if (err != 0) FOUR_C_THROW("Matrix read failed.");
      std::shared_ptr<Epetra_CrsMatrix> A_crs = Core::Utils::shared_ptr_from_ref(*A);
      Core::LinAlg::SparseMatrix A_sparse(A_crs, Core::LinAlg::DataAccess::Copy);

      const double tol = 1e-14;
      const int power = 4;

      Core::LinAlg::Graph sparsity_pattern(A->Graph());

      std::shared_ptr<Core::LinAlg::SparseMatrix> A_thresh =
          Core::LinAlg::threshold_matrix(A_sparse, tol);
      std::shared_ptr<Core::LinAlg::Graph> sparsity_pattern_enriched =
          Core::LinAlg::enrich_matrix_graph(*A_thresh, power);

      Core::LinAlg::OptionsSparseMatrixInverse options;
      options.alpha = 1e-5;
      options.rho = 1.01;

      std::shared_ptr<Core::LinAlg::SparseMatrix> A_inverse = Core::LinAlg::matrix_sparse_inverse(
          A_sparse, std::make_shared<Core::LinAlg::Graph>(sparsity_pattern), options);

      A_thresh = Core::LinAlg::threshold_matrix(*A_inverse, tol);

      // Check for global entries
      const int A_inverse_nnz = A_thresh->num_global_nonzeros();
      // Note: the number of entries lower than a tolerance is not necessarily deterministic
      EXPECT_NEAR(A_inverse_nnz, 29139, 10);

      // Check for overall norm of matrix inverse
      constexpr double expected_frobenius_norm = 7.448506913184814e+03;
      EXPECT_NEAR(
          A_thresh->norm_frobenius(), expected_frobenius_norm, expected_frobenius_norm * 1e-10);
    }
  }
}  // namespace

FOUR_C_NAMESPACE_CLOSE

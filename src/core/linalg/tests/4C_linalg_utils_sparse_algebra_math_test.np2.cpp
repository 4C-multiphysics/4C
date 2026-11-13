// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_linalg_utils_sparse_algebra_math.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_comm_utils.hpp"
#include "4C_linalg_utils_sparse_algebra_io.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_unittest_utils_assertions_test.hpp"
#include "4C_unittest_utils_support_files_test.hpp"

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
    Core::LinAlg::SparseMatrix A = Core::LinAlg::read_matrix_market_file_as_sparse_matrix(
        TESTING::get_support_file_path("test_matrices/poisson1d.mm").c_str(), comm_);

    std::shared_ptr<Core::LinAlg::SparseMatrix> A_inverse = Core::LinAlg::matrix_sparse_inverse(
        A, std::make_shared<Core::LinAlg::Graph>(A.epetra_matrix().Graph()));

    // Check for global entries
    const int A_sparse_nnz = A.num_global_nonzeros();
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
    Core::LinAlg::SparseMatrix A = Core::LinAlg::read_matrix_market_file_as_sparse_matrix(
        TESTING::get_support_file_path("test_matrices/nonsym.mm").c_str(), comm_);

    std::shared_ptr<Core::LinAlg::SparseMatrix> A_inverse = Core::LinAlg::matrix_sparse_inverse(
        A, std::make_shared<Core::LinAlg::Graph>(A.epetra_matrix().Graph()));

    // Check for global entries
    const int A_sparse_nnz = A.num_global_nonzeros();
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
    Core::LinAlg::SparseMatrix A = Core::LinAlg::read_matrix_market_file_as_sparse_matrix(
        TESTING::get_support_file_path("test_matrices/beamI.mm").c_str(), comm_);

    {
      const double tol = 1e-8;
      std::shared_ptr<Core::LinAlg::Graph> sparsity_pattern =
          Core::LinAlg::threshold_matrix_graph(A, tol);
      std::shared_ptr<Core::LinAlg::SparseMatrix> A_inverse =
          Core::LinAlg::matrix_sparse_inverse(A, sparsity_pattern);
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

      Core::LinAlg::Graph sparsity_pattern(A.epetra_matrix().Graph());

      std::shared_ptr<Core::LinAlg::SparseMatrix> A_thresh = Core::LinAlg::threshold_matrix(A, tol);
      std::shared_ptr<Core::LinAlg::Graph> sparsity_pattern_enriched =
          Core::LinAlg::enrich_matrix_graph(*A_thresh, power);
      std::shared_ptr<Core::LinAlg::SparseMatrix> A_inverse =
          Core::LinAlg::matrix_sparse_inverse(A, sparsity_pattern_enriched);
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
      Core::LinAlg::SparseMatrix A = Core::LinAlg::read_matrix_market_file_as_sparse_matrix(
          TESTING::get_support_file_path("test_matrices/beamII.mm").c_str(), comm_);

      const double tol = 1e-14;
      const int power = 4;

      Core::LinAlg::Graph sparsity_pattern(A.epetra_matrix().Graph());

      std::shared_ptr<Core::LinAlg::SparseMatrix> A_thresh = Core::LinAlg::threshold_matrix(A, tol);
      std::shared_ptr<Core::LinAlg::Graph> sparsity_pattern_enriched =
          Core::LinAlg::enrich_matrix_graph(*A_thresh, power);

      Core::LinAlg::OptionsSparseMatrixInverse options;
      options.alpha = 1e-5;
      options.rho = 1.01;

      EXPECT_ANY_THROW(Core::LinAlg::matrix_sparse_inverse(A, sparsity_pattern_enriched, options));
    }

    // Try to invert pure Neumann problem, this should succeed as we use a projected operator and
    // a-priori diagonal perturbation
    {
      Core::LinAlg::SparseMatrix A = Core::LinAlg::read_matrix_market_file_as_sparse_matrix(
          TESTING::get_support_file_path("test_matrices/beamII_projected.mm").c_str(), comm_);

      const double tol = 1e-14;
      const int power = 4;

      Core::LinAlg::Graph sparsity_pattern(A.epetra_matrix().Graph());

      std::shared_ptr<Core::LinAlg::SparseMatrix> A_thresh = Core::LinAlg::threshold_matrix(A, tol);
      std::shared_ptr<Core::LinAlg::Graph> sparsity_pattern_enriched =
          Core::LinAlg::enrich_matrix_graph(*A_thresh, power);

      Core::LinAlg::OptionsSparseMatrixInverse options;
      options.alpha = 1e-5;
      options.rho = 1.01;

      std::shared_ptr<Core::LinAlg::SparseMatrix> A_inverse = Core::LinAlg::matrix_sparse_inverse(
          A, std::make_shared<Core::LinAlg::Graph>(sparsity_pattern), options);

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

  TEST_F(SparseAlgebraMathTest, MultiplyMultiVectorDenseMatrix)
  {
    auto map = Core::LinAlg::Map(10, 0, comm_);
    auto multi_vector = Core::LinAlg::MultiVector<double>(map, 3, true);
    multi_vector.put_scalar(1.0);

    // Test with a square dense matrix
    {
      auto matrix = Core::LinAlg::SerialDenseMatrix(3, 3, true);
      matrix(0, 0) = 1.0;
      matrix(1, 1) = 2.0;
      matrix(2, 2) = 3.0;

      auto result = Core::LinAlg::multiply_multi_vector_dense_matrix(multi_vector, matrix);

      for (int col = 0; col < result.num_vectors(); col++)
        for (int my_row = 0; my_row < result.local_length(); my_row++)
          EXPECT_NEAR(result.get_vector(col).get_values()[my_row], matrix(col, col), 1e-12);
    }

    // Test with a tall-and-skinny matrix
    {
      auto matrix = Core::LinAlg::SerialDenseMatrix(3, 4, true);
      matrix(0, 0) = 1.0;
      matrix(0, 3) = 1.0;
      matrix(1, 1) = 2.0;
      matrix(1, 3) = 2.0;
      matrix(2, 2) = 3.0;
      matrix(2, 3) = 3.0;

      std::array<double, 4> result_values = {1.0, 2.0, 3.0, 6.0};
      auto result = Core::LinAlg::multiply_multi_vector_dense_matrix(multi_vector, matrix);

      EXPECT_EQ(result.num_vectors(), 4);

      for (int col = 0; col < result.num_vectors(); col++)
        for (int my_row = 0; my_row < result.local_length(); my_row++)
          EXPECT_NEAR(result.get_vector(col).get_values()[my_row], result_values[col], 1e-12);
    }

#ifdef FOUR_C_ENABLE_ASSERTIONS
    // Test with a tall-and-skinny matrix (wrong dimensions)
    {
      auto matrix = Core::LinAlg::SerialDenseMatrix(4, 3, true);
      matrix(0, 0) = 1.0;
      matrix(1, 1) = 2.0;
      matrix(2, 2) = 3.0;
      matrix(3, 0) = 2.0;
      matrix(3, 1) = 1.0;
      matrix(3, 2) = 3.0;

      EXPECT_ANY_THROW(Core::LinAlg::multiply_multi_vector_dense_matrix(multi_vector, matrix));
    }
#endif
  }

  /**
   * This test generates a random multi vector with three basis vectors
   * and applies an orthonormalization procedure to it.
   *
   * After orthonormalization, the Gram matrix of the resulting vectors is computed. If the columns
   * are orthonormal, the Gram matrix must equal the identity matrix.
   */
  TEST_F(SparseAlgebraMathTest, MultiVectorOrthonormalization)
  {
    const int num_basis_vectors = 3;

    auto map = Core::LinAlg::Map(10, 0, comm_);
    auto multi_vector = Core::LinAlg::MultiVector<double>(map, num_basis_vectors, true);
    multi_vector.random();

    auto local_map = Core::LinAlg::Map(multi_vector.num_vectors(), 0, multi_vector.get_comm(),
        Core::LinAlg::LocalGlobal::locally_replicated);
    auto result = Core::LinAlg::MultiVector<double>(local_map, multi_vector.num_vectors());

    // The inner product is equal to an identity, thus the vector columns are orthonormal.
    {
      auto identity = Core::LinAlg::SerialDenseMatrix(num_basis_vectors, num_basis_vectors, true);
      identity(0, 0) = 1.0;
      identity(1, 1) = 1.0;
      identity(2, 2) = 1.0;

      multi_vector = Core::LinAlg::orthonormalize_multi_vector(multi_vector);

      result.multiply('T', 'N', 1.0, multi_vector, multi_vector, 0.0);
      auto gram_matrix = Core::LinAlg::SerialDenseMatrix(Teuchos::DataAccess::Copy,
          result.get_values(), result.stride(), result.num_vectors(), result.num_vectors());

      FOUR_C_EXPECT_NEAR(gram_matrix, identity, 1e-12);
    }
  }

  /**
   * Test rank-1 correction of a singular matrix.
   *
   * The test loads the periodic 1D Poisson finite-difference matrix, which is
   * singular with the constant vector as its nullspace. A basis vector spanning
   * this nullspace is constructed and orthonormalized. A rank-1 correction is
   * then applied with shift 1.0 along this direction.
   *
   * The test verifies that the corrected operator maps the normalized nullspace
   * vector to itself, confirming that the rank correction shifts the zero
   * eigenvalue to one and removes the singularity in this mode.
   */
  void check_rank_correction(
      MPI_Comm comm, Core::LinAlg::SparseMatrix::StorageProfile projection_profile)
  {
    Core::LinAlg::SparseMatrix A = Core::LinAlg::read_matrix_market_file_as_sparse_matrix(
        TESTING::get_support_file_path("test_matrices/poisson1d_periodic.mm").c_str(), comm);

    auto basis = Core::LinAlg::MultiVector<double>(A.row_map(), 1);
    basis.put_scalar(1.0);
    auto ortho_basis = Core::LinAlg::orthonormalize_multi_vector(basis);

    auto A_projected =
        projection_profile == Core::LinAlg::SparseMatrix::StorageProfile::static_profile
            ? Core::LinAlg::matrix_rank_correction(
                  A, basis, {1.0, true, Core::LinAlg::SparseMatrix::StorageProfile::static_profile})
            : Core::LinAlg::matrix_rank_correction(A, basis, {1.0, true});

    auto result = Core::LinAlg::MultiVector<double>(A.row_map(), 1);
    A_projected->multiply(false, ortho_basis, result);

    for (int my_row = 0; my_row < result.local_length(); my_row++)
    {
      const double result_value = result.get_values()[my_row];
      const double ortho_basis_value = ortho_basis.get_values()[my_row];

      EXPECT_NEAR(result_value, ortho_basis_value, 1e-12);
    }
  }

  TEST_F(SparseAlgebraMathTest, MatrixRankmCorrection)
  {
    check_rank_correction(comm_, Core::LinAlg::SparseMatrix::StorageProfile::dynamic_profile);
  }

  // The same rank-one projection needs more than npr entries per row when CRS is forced static.
  TEST_F(SparseAlgebraMathTest, MatrixRankmCorrectionStaticProfile)
  {
    check_rank_correction(comm_, Core::LinAlg::SparseMatrix::StorageProfile::static_profile);
  }

}  // namespace

FOUR_C_NAMESPACE_CLOSE

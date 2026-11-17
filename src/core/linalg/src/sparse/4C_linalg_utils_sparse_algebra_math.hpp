// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINALG_UTILS_SPARSE_ALGEBRA_MATH_HPP
#define FOUR_C_LINALG_UTILS_SPARSE_ALGEBRA_MATH_HPP

#include "4C_config.hpp"

#include "4C_linalg_blocksparsematrix.hpp"
#include "4C_linalg_graph.hpp"
#include "4C_linalg_map.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  /*!
   \brief Add a (transposed) Core::LinAlg::SparseMatrix to a Core::LinAlg::SparseMatrix:
   B = B*scalarB + A(^T)*scalarA

   Add one matrix to another.

   As opposed to the other Add() functions, this method can handle both the case where
   matrix B is fill-completed (for performance reasons) but does not have to.
   If B is completed and new matrix elements are detected, the matrix is un-completed and
   rebuild internally (expensive).

   The matrix B may or may not be completed. If B is completed, no new elements can be
   inserted and the addition only succeeds in case the sparsity pattern of B is a superset of
   the sparsity pattern of A (otherwise: FOUR_C_THROW).

   Performance characterization: If B is filled (completed), this function is pretty fast,
   typically on the order of two to four matrix-vector products with B. The case where B is
   un-filled runs much slower (on the order of up to 100 matrix-vector products).

   Sparsity patterns of A and B need not match and A and B can be
   nonsymmetric in value and pattern.

   Row map of A has to be a processor-local subset of the row map of B.

   Note that this is a true parallel add, even in the transposed case!

   \param A          (in)     : Matrix to add to B (must have Filled()==true)
   \param transposeA (in)     : flag indicating whether transposed of A should be used
   \param scalarA    (in)     : scaling factor for A
   \param B          (in/out) : Matrix to be added to (must have Filled()==false)
   \param scalarB    (in)     : scaling factor for B
   */
  void matrix_add(const Core::LinAlg::SparseMatrix& A, const bool transposeA, const double scalarA,
      Core::LinAlg::SparseMatrix& B, const double scalarB);

  /*!
   \brief Put a sparse matrix (partially) onto another: B(rowmap) = A(rowmap)*scalarA

  Put one matrix onto another. The matrix B to be added to must not be completed.
  Sparsity patterns of A and B need not match and A and B can be nonsymmetric in value and pattern.
  Row map of A has to be a processor-local subset of the row map of B.

  \param A          (in)     : Matrix to add to this (must have Filled()==true)
  \param scalarA    (in)     : scaling factor for #A
  \param rowmap     (in)     : to put selectively on rows in #rowmap (inactive if ==nullptr)
  \param B          (in/out) : Matrix to be added to (must have Filled()==false)
  */
  void matrix_put(const Core::LinAlg::SparseMatrix& A, const double scalarA,
      std::shared_ptr<const Core::LinAlg::Map> rowmap, Core::LinAlg::SparseMatrix& B);

  /*!
   \brief Multiply a (transposed) sparse matrix with another (transposed): C = A(^T)*B(^T)

   Multiply one matrix with another. Both matrices must be completed. Sparsity
   Respective Range, Row and Domain maps of A(^T) and B(^T) have to match.

   \note that this is a true parallel multiplication, even in the transposed case!

   \note Does call complete on C upon exit by default.

   \param A          (in) : Matrix to multiply with B (must have Filled()==true)
   \param transA     (in) : flag indicating whether transposed of A should be used
   \param B          (in) : Matrix to multiply with A (must have Filled()==true)
   \param transB     (in) : flag indicating whether transposed of B should be used
   \param complete   (in) : flag indicating whether fill_complete should be called on C upon
                            exit, (defaults to true)
   \param profile    (in) : output row-storage profile (defaults to dynamic); static output is
                            sized from actual product rows using a growable temporary
   \return Matrix product A(^T)*B(^T)
   */
  std::unique_ptr<SparseMatrix> matrix_multiply(const SparseMatrix& A, bool transA,
      const SparseMatrix& B, bool transB, bool complete = true,
      SparseMatrix::StorageProfile profile = SparseMatrix::StorageProfile::dynamic_profile);

  /*!
   \brief Multiply a (transposed) sparse matrix with another (transposed): C = A(^T)*B(^T)

   Multiply one matrix with another. Both matrices must be completed. Sparsity
   Respective Range, Row and Domain maps of A(^T) and B(^T) have to match.

   \note that this is a true parallel multiplication, even in the transposed case!

   \note Does call complete on C upon exit by default.

   \note In this version the flags explicitdirichlet and savegraph must be handed in.
   Thus, they can be defined explicitly, while in the standard version of MatrixMultiply()
   above, result matrix C automatically inherits these flags from input matrix A.

   \param A                 (in) : Matrix to multiply with B (must have Filled()==true)
   \param transA            (in) : flag indicating whether transposed of A should be used
   \param B                 (in) : Matrix to multiply with A (must have Filled()==true)
   \param transB            (in) : flag indicating whether transposed of B should be used
   \param explicitdirichlet (in) : flag deciding on explicitdirichlet flag of C
   \param savegraph         (in) : flag deciding on savegraph flag of C
   \param complete          (in) : flag indicating whether fill_complete should be called on C upon
                                   exit, (defaults to true)
   \param profile           (in) : output row-storage profile (defaults to dynamic); static output
                                   uses an intermediate and an exactly sized per-row copy
   \return Matrix product A(^T)*B(^T)
   */
  std::unique_ptr<SparseMatrix> matrix_multiply(const SparseMatrix& A, bool transA,
      const SparseMatrix& B, bool transB, bool explicitdirichlet, bool savegraph,
      bool complete = true,
      SparseMatrix::StorageProfile profile = SparseMatrix::StorageProfile::dynamic_profile);


  /*!
   \brief Compute transposed matrix of a sparse matrix explicitly

   \warning This is an expensive operation!

   \pre Matrix needs to be completed for this operation.

   \param A (in) : Matrix to transpose

   \return matrix_transpose of the input matrix A.
   */
  std::shared_ptr<SparseMatrix> matrix_transpose(const SparseMatrix& A);

  /**
   * \brief Options for sparse matrix inverse
   *
   * A well-known disadvantage of incomplete factorizations is that the two factors can be unstable.
   * This may occur, for instance, if matrix A is badly scaled, or if one of the pivotal elements
   * occurs to be very small. In this case, a-priori diagonal perturbations may be effective.
   *
   * A_perturbed(i,j) = A(i,j)  i~=j
   * A_perturbed(i,i) = alpha*sign(A(i,i))+rho*A(i,i)
   *
   */
  struct OptionsSparseMatrixInverse
  {
    //! Absolute diagonal scaling factor
    double alpha = 0.0;

    //! Relative diagonal scaling factor
    double rho = 1.0;
  };

  /**
   * \brief Compute sparse inverse matrix of a sparse matrix explicitly
   *
   * \warning This is an expensive operation depending on the density of the sparse operator!
   *
   * \pre Matrix needs to be completed for this operation.
   *
   * \param A                (in) : Matrix to invert
   * \param sparsity_pattern (in) : Sparsity pattern to calculate the sparse inverse of A on
   *
   * The implementation is loosely based on:
   * M. J. Grote and T. Huckle: Parallel preconditioning with sparse approximate inverses.
   * SIAM Journal on Scientific Computing, 18(3):838-853, 1997,
   * https://doi.org/10.1137/S1064827594276552
   *
   * \return Sparse inverse A^(-1) of the input matrix A.
   */
  std::shared_ptr<SparseMatrix> matrix_sparse_inverse(const SparseMatrix& A,
      std::shared_ptr<Core::LinAlg::Graph> sparsity_pattern,
      OptionsSparseMatrixInverse options = {});

  struct OptionsSparseMatrixRankCorrection
  {
    //! Absolute diagonal scaling factor
    double alpha = 1.0;

    //! Flag indicating if the basis vectors should be orthonormalized during rank correction
    bool orthonormalize = true;

    //! Storage profile of the intermediate projection; dynamic preserves existing behavior.
    SparseMatrix::StorageProfile projection_profile = SparseMatrix::StorageProfile::dynamic_profile;
  };

  /**
   * @brief Applies a rank correction to a sparse matrix using a given basis.
   *
   * This function constructs a low-rank projection matrix from the provided
   * basis vectors and adds a scaled version of this projection to the input
   * matrix. The correction has the form
   *
   * \f[
   * A' = A + \alpha \, (B B^T)
   * \f]
   *
   * where \f$A\f$ is the input matrix, \f$B\f$ is the matrix whose columns are
   * the provided basis vectors, and \f$\alpha\f$ is a scaling factor specified
   * in the options.
   *
   * Optionally, the basis vectors can be orthonormalized prior to constructing
   * the projection matrix. The projection matrix \f$B B^T\f$ is assembled using
   * sparse matrix operations.
   *
   * @param A       (in): Input sparse matrix to which the rank correction is applied.
   * @param basis   (in): Multi-vector whose columns define the basis used to construct the low-rank
   *                      correction.
   * @param options (in): Options controlling the rank correction, including the scaling factor
   * \f$\alpha\f$, whether the basis should be orthonormalized, and the optional
   * projection storage profile.
   *
   * @return A unique pointer to a new sparse matrix containing the corrected matrix
   */
  std::unique_ptr<SparseMatrix> matrix_rank_correction(const SparseMatrix& A,
      const MultiVector<double>& basis, OptionsSparseMatrixRankCorrection options = {});

  /**
   * \brief Computes multiplication of a MultiVector times a SerialDenseMatrix
   *
   * \param multi_vector (in): MultiVector to be used for multiplication
   * \param matrix (in): SerialDenseMatrix to be used for multiplication
   *
   * \return MultiVector containing multi_vector*matrix
   */
  MultiVector<double> multiply_multi_vector_dense_matrix(
      const Core::LinAlg::MultiVector<double>& multi_vector,
      const Core::LinAlg::SerialDenseMatrix& matrix);

  /**
   * \brief Computes the outer product of two MultiVectors
   *
   * \param mv1 (in): First MultiVector to be used for multiplication
   * \param mv2 (in): Second MultiVector to be used for multiplication
   * \param id (in): Flag indicating which MultiVector should be used for sparsity estimate
   * \param fill (in): Flag indicating whether complete should be called on result upon exit,
   * (defaults to true)
   *
   * \return SparseMatrix containing the outer-product MV*MV^T
   */
  Core::LinAlg::SparseMatrix multiply_multi_vector_multi_vector(
      const Core::LinAlg::MultiVector<double>& mv1, const Core::LinAlg::MultiVector<double>& mv2,
      const int id = 1, const bool fill = true);

  /**
   * \brief Orthonormalize the columns of a multi-vector.
   *
   * This function takes a multi-vector whose columns represent a set of vectors (e.g. nullspace
   * or near-nullspace basis vectors) and returns a new multi-vector whose columns form an
   * orthonormal basis for the same column space.
   *
   * \param multi_vector (in): Input multi-vector whose columns will be orthonormalized.
   *
   * \return A new multi-vector with orthonormal columns spanning the same column space as the
   * input.
   */
  Core::LinAlg::MultiVector<double> orthonormalize_multi_vector(
      const Core::LinAlg::MultiVector<double>& multi_vector);

}  // namespace Core::LinAlg

FOUR_C_NAMESPACE_CLOSE

#endif

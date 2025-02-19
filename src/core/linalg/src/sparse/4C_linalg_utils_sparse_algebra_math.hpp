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
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_graph.hpp"
#include "4C_linalg_serialdensematrix.hpp"

#include <Epetra_CrsMatrix.h>
#include <Epetra_Export.h>
#include <Epetra_Import.h>
#include <Epetra_Map.h>

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  /*!
   \brief Add a (transposed) Epetra_CrsMatrix to another: B = B*scalarB + A(^T)*scalarA

   Add one matrix to another.

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
  void add(const Epetra_CrsMatrix& A, const bool transposeA, const double scalarA,
      Epetra_CrsMatrix& B, const double scalarB);

  /*!
   \brief Add a (transposed) Epetra_CrsMatrix to a Core::LinAlg::SparseMatrix: B = B*scalarB +
   A(^T)*scalarA

   Add one matrix to another.

   As opposed to the other Add() functions, this method can handle both the case where
   matrix B is fill-completed (for performance reasons) but does not have to.
   If B is completed and new matrix elements are detected, the matrix is un-completed and
   rebuild internally (expensive).

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
  void add(const Epetra_CrsMatrix& A, const bool transposeA, const double scalarA,
      Core::LinAlg::SparseMatrixBase& B, const double scalarB);

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
      std::shared_ptr<const Epetra_Map> rowmap, Core::LinAlg::SparseMatrixBase& B);

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
   \return Matrix product A(^T)*B(^T)
   */
  std::unique_ptr<SparseMatrix> matrix_multiply(
      const SparseMatrix& A, bool transA, const SparseMatrix& B, bool transB, bool complete = true);

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
   \return Matrix product A(^T)*B(^T)
   */
  std::unique_ptr<SparseMatrix> matrix_multiply(const SparseMatrix& A, bool transA,
      const SparseMatrix& B, bool transB, bool explicitdirichlet, bool savegraph,
      bool complete = true);


  /*!
   \brief Compute transposed matrix of a sparse matrix explicitly

   \warning This is an expensive operation!

   \pre Matrix needs to be completed for this operation.

   \param A (in) : Matrix to transpose

   \return matrix_transpose of the input matrix A.
   */
  std::shared_ptr<SparseMatrix> matrix_transpose(const SparseMatrix& A);

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
  std::shared_ptr<SparseMatrix> matrix_sparse_inverse(
      const SparseMatrix& A, std::shared_ptr<Core::LinAlg::Graph> sparsity_pattern);

}  // namespace Core::LinAlg

FOUR_C_NAMESPACE_CLOSE

#endif

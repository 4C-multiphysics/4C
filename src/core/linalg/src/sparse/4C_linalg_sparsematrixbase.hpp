// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINALG_SPARSEMATRIXBASE_HPP
#define FOUR_C_LINALG_SPARSEMATRIXBASE_HPP

#include "4C_config.hpp"

#include "4C_linalg_sparseoperator.hpp"

#include <Epetra_CrsMatrix.h>
#include <Epetra_FECrsMatrix.h>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  /// Base class of sparse matrix that provides the simple functions
  class SparseMatrixBase : public SparseOperator
  {
   public:
    /*! \brief return the internal Epetra_Operator

      The internal Epetra_Operator here is the internal Epetra_CrsMatrix or Epetra_FECrsMatrix. This
      way the solver can down-cast to Epetra_CrsMatrix or Epetra_FECrsMatrix and access the matrix
      rows directly.

      \note This method is here for performance reasons.
     */
    std::shared_ptr<Epetra_Operator> epetra_operator() override { return sysmat_; }

    /// return the internal Epetra matrix as Epetra_Operator
    std::shared_ptr<Epetra_Operator> epetra_operator() const { return sysmat_; }

    /// return the internal Epetra_CrsMatrix or Epetra_FECrsMatrix
    /// (down-cast from Epetra_CrsMatrix !) (you should not need this!)
    std::shared_ptr<Epetra_CrsMatrix> epetra_matrix() { return sysmat_; }

    /// return the internal Epetra_CrsMatrix or Epetra_FECrsMatrix
    /// (down-cast from Epetra_CrsMatrix !) (you should not need this!)
    std::shared_ptr<Epetra_CrsMatrix> epetra_matrix() const { return sysmat_; }

    /** \name Attribute set methods */
    //@{

    /// If set true, transpose of this operator will be applied.
    int SetUseTranspose(bool UseTranspose) override;

    //@}

    /** \name Matrix Properties Query Methods */
    //@{

    /// If Complete() has been called, this query returns true, otherwise it returns false.
    bool filled() const override { return sysmat_->Filled(); }

    /** \brief Return TRUE if all Dirichlet boundary conditions have been applied
     *  to this matrix */
    /** Actual implementation of the check. If a local coordinate transformation
     *  has been considered, we do a point by point comparison of each DBC row
     *  of the diagonal block. Note that the number of entries in each row
     *  of the trafo matrix must not coincide with the number of entries in
     *  each corresponding DBC row of this matrix, if the DBCs are not applied
     *  explicitly.
     *  If no local transformation is involved, we are just looking for a 1.0
     *  on the diagonal of diagonal blocks and zeros everywhere else in the DBC
     *  rows.
     *
     *  \author hiermeier \date 01/2018 */
    bool is_dbc_applied(const Epetra_Map& dbcmap, bool diagonalblock = true,
        const Core::LinAlg::SparseMatrix* trafo = nullptr) const override;

    //@}

    /** \name Mathematical functions */
    //@{

    /// Returns the result of a Epetra_Operator applied to a Epetra_MultiVector X in Y.
    int Apply(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const override;

    /// Returns the result of a Epetra_Operator inverse applied to an Epetra_MultiVector X in Y.
    int ApplyInverse(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const override;

    /// Returns the infinity norm of the global matrix.
    double NormInf() const override;

    /// Returns the one norm of the global matrix.
    double norm_one() const;

    /// Returns the frobenius norm of the global matrix.
    double norm_frobenius() const;

    //@}

    /** \name Attribute access functions */
    //@{

    /// Returns the maximum number of nonzero entries across all rows on this processor.
    int max_num_entries() const;

    /// Returns the Epetra_Map object associated with the rows of this matrix.
    const Epetra_Map& row_map() const { return sysmat_->RowMap(); }

    /// Returns the Epetra_Map object that describes the set of column-indices that appear in each
    /// processor's locally owned matrix rows.
    const Epetra_Map& col_map() const { return sysmat_->ColMap(); }

    /// Returns the Epetra_Map object associated with the domain of this matrix operator.
    const Epetra_Map& domain_map() const override { return sysmat_->DomainMap(); }

    /// Returns the Epetra_Map object associated with the range of this matrix operator.
    const Epetra_Map& range_map() const { return sysmat_->RangeMap(); }

    /// Returns the current UseTranspose setting.
    bool UseTranspose() const override;

    /// Returns true if the this object can provide an approximate Inf-norm, false otherwise.
    bool HasNormInf() const override;

    /// Returns a pointer to the Epetra_Comm communicator associated with this operator.
    const Epetra_Comm& Comm() const override;

    /// Returns the Epetra_Map object associated with the domain of this operator.
    const Epetra_Map& OperatorDomainMap() const override;

    /// Returns the Epetra_Map object associated with the range of this operator.
    const Epetra_Map& OperatorRangeMap() const override;

    //@}

    /** \name Computational methods */
    //@{

    /// Returns the result of a matrix multiplied by a Core::LinAlg::Vector<double> x in y.
    int multiply(
        bool TransA, const Core::LinAlg::Vector<double>& x, Core::LinAlg::Vector<double>& y) const;

    /// Returns the result of a Epetra_CrsMatrix multiplied by a Epetra_MultiVector X in Y.
    int multiply(bool TransA, const Core::LinAlg::MultiVector<double>& X,
        Core::LinAlg::MultiVector<double>& Y) const override;

    /// Scales the Epetra_CrsMatrix on the left with a Core::LinAlg::Vector<double> x.
    int left_scale(const Core::LinAlg::Vector<double>& x);

    /// Scales the Epetra_CrsMatrix on the right with a Core::LinAlg::Vector<double> x.
    int right_scale(const Core::LinAlg::Vector<double>& x);

    //@}

    /** \name Insertion/Replace/SumInto methods */
    //@{

    /// Initialize all values in the matrix with constant value.
    int put_scalar(double ScalarConstant);

    /// Multiply all values in the matrix by a constant value (in place: A <- ScalarConstant * A).
    int scale(double ScalarConstant) override;

    /// Replaces diagonal values of the matrix with those in the user-provided vector.
    int replace_diagonal_values(const Core::LinAlg::Vector<double>& Diagonal);

    //@}

    /** \name Extraction methods */
    //@{

    /// Returns a copy of the main diagonal in a user-provided vector.
    int extract_diagonal_copy(Core::LinAlg::Vector<double>& Diagonal) const;

    //@}

    /// Add one operator to another
    void add(const Core::LinAlg::SparseOperator& A, const bool transposeA, const double scalarA,
        const double scalarB) override;

    /// Add one SparseMatrixBase to another
    void add_other(Core::LinAlg::SparseMatrixBase& B, const bool transposeA, const double scalarA,
        const double scalarB) const override;

    /// Add one BlockSparseMatrix to another
    void add_other(Core::LinAlg::BlockSparseMatrixBase& B, const bool transposeA,
        const double scalarA, const double scalarB) const override;


   protected:
    /// internal epetra matrix (Epetra_CrsMatrix or Epetra_FECrsMatrix)
    std::shared_ptr<Epetra_CrsMatrix> sysmat_;
  };

}  // namespace Core::LinAlg

FOUR_C_NAMESPACE_CLOSE

#endif

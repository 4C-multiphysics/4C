// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINALG_SPARSEMATRIX_HPP
#define FOUR_C_LINALG_SPARSEMATRIX_HPP

#include "4C_config.hpp"

#include "4C_linalg.hpp"
#include "4C_linalg_graph.hpp"
#include "4C_linalg_mapextractor.hpp"
#include "4C_linalg_sparseoperator.hpp"
#include "4C_linalg_utils_exceptions.hpp"

#include <Epetra_CrsMatrix.h>
#include <Epetra_FECrsMatrix.h>


FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  // forward declarations:
  class BlockSparseMatrixBase;

  template <class Strategy>
  class BlockSparseMatrix;

  /// A single sparse matrix enhanced with features for FE simulations
  /*!

    A single sparse matrix. Internally we have an Epetra_CrsMatrix
    (or the sub-class Epetra_FECrsMatrix). So we have
    all the glory of a fully parallel and fast sparse matrix. The added value
    is twofold. For one thing there are the FE specific operations. Assemble()
    adds an (element) matrix to the (global) sparse matrix and
    ApplyDirichlet() modifies the matrix to contain just ones on Dirichlet
    rows (the columns in other rows are not touched, so the matrix becomes
    unsymmetric).

    The second gain are the different states this matrix can be in. You can
    set explicitdirichlet==true in order to modify the matrix graph in each
    ApplyDirichlet() call to contain just the diagonal entries on those rows
    -- this essentially copies the matrix. (ML gains a lot from completely
    Dirichlet-constrained rows.) With explicitdirichlet==false the matrix
    graph is not touched, instead the Dirichlet rows are filled with zeros.

    With savegraph==true you specify that you want to keep the original matrix
    graph before you apply Dirichlet conditions. This way you can call Zero()
    and get an already Filled() matrix. You cannot alter its graph afterwards,
    but Assemble() is much faster if your matrix is already Filled(). Of course
    you can always reset() you matrix, that is throw away the matrix graph and
    start anew with an empty matrix.

    If FE_MATRIX is set as a flag, the implementation is based on an Epetra_FECrsMatrix.
    Nonlocal matrix values can be assembled by invoking FEAssemble()-methods
    instead of Assemble()-methods. Internally this will cause
    the GlobalAssemble()-method to distribute the nonlocal values to the owning
    procs before fill_complete is called on the matrix.
    Since Epetra_FECrsMatrix is a sub-class of Epetra_CrsMatrix,
    all other functionality can be used as described above.
    All SparseMatrix constructors create Epetra_CrsMatrices by default.

    \note A large part of the SparseMatrix interface consists of methods from
    the internal Epetra_CrsMatrix. If there are methods in Epetra_CrsMatrix
    and not in SparseMatrix that you would like to call (for legitimate
    reasons!) please add them to the SparseMatrix.

   */
  class SparseMatrix : public SparseOperator
  {
   public:
    /*!
      flag for the underlying Epetra matrix type; CRS_MATRIX means that the implementation of the
      linalg sparse matrix is based on an Epetra_CrsMatrix.
      If FE_MATRIX is chosen, the implementation is based on an Epetra_FECrsMatrix.
      This allows the assembly and handling of nonlocal values in addition to
      Epetra_CrsMatrix-functionality.
     */
    enum MatrixType
    {
      CRS_MATRIX,
      FE_MATRIX
    };

    /// The following dummy templated constructors will catch all attempts to call any constructor
    /// with arguments that require implicit casting. Therefore you have to instantiate this class
    /// with exactly the types matching one of the constructors
    template <typename... T>
    SparseMatrix(T...) = delete;

    /// construction of sparse matrix
    SparseMatrix(std::shared_ptr<Core::LinAlg::Graph> crsgraph,
        std::shared_ptr<Core::LinAlg::MultiMapExtractor> dbcmaps);

    /// construction of sparse matrix
    SparseMatrix(const Core::LinAlg::Map& rowmap, const int npr, bool explicitdirichlet = true,
        bool savegraph = false, MatrixType matrixtype = CRS_MATRIX);

    /// construction of sparse matrix
    SparseMatrix(const Core::LinAlg::Map& rowmap, const Core::LinAlg::Map& colmap, const int npr,
        bool explicitdirichlet = true, bool savegraph = false, MatrixType matrixtype = CRS_MATRIX);

    /// construction of sparse matrix using an individual estimate for number of non-zeros per row
    SparseMatrix(const Core::LinAlg::Map& rowmap, std::vector<int>& numentries,
        bool explicitdirichlet = true, bool savegraph = false, MatrixType matrixtype = CRS_MATRIX);

    /// construction of sparse matrix
    /*!
       Makes either a deep copy of the Epetra_CrsMatrix or Epetra_FECrsMatrix.
       (Note: \pre matrix.Filled()==true)
       or an implicit construction from a Epetra_CrsMatrix or Epetra_FECrsMatrix
       where the std::shared_ptr is copied internally leading to a new view on the
       Epetra_CrsMatrix or Epetra_FECrsMatrix.
     */
    SparseMatrix(std::shared_ptr<Epetra_CrsMatrix> matrix, DataAccess access,
        bool explicitdirichlet = true, bool savegraph = false, MatrixType matrixtype = CRS_MATRIX);

    /// construction of a diagonal matrix from a vector
    /*!
      Creates diagonal matrix with range and domain map equal to vector map.
      Sets diagonal values from vector and does NOT call Complete() on matrix

      Allocates new memory for the Epetra_CrsMatrix or Epetra_FECrsMatrix
     */
    SparseMatrix(const Core::LinAlg::Vector<double>& diag, bool explicitdirichlet = true,
        bool savegraph = false, MatrixType matrixtype = CRS_MATRIX);

    /// Copy constructor. Deep copy or view on matrix.
    /*!
      \warning A view assignment will have your matrix use the same internal
      data as the original matrix. Changes on one side will affect the
      other. However, some methods like Zero() or reset() can, depending on
      the SparseMatrix flags, cut the connection. Do not rely on the view if
      you change one of these matrices! The view assignment is meant to
      provide a slim copy operation that transfers ownership from one matrix
      to the other before the original matrix is destroyed. Do a deep copy if
      both matrices are meant to live on.

      \param mat matrix to assign from
      \param access how to treat this assignment: Copy or View
     */
    SparseMatrix(const SparseMatrix& mat, DataAccess access = DataAccess::Copy);

    /// Assignment operator. Makes a deep copy.
    SparseMatrix& operator=(const SparseMatrix& mat);

    /// Assignment method. Deep copy or view on matrix.
    /*!
      Explicit method for the assignment operator. You can make an explicit
      copy of the internal Epetra_CrsMatrix (or Epetra_FECrsMatrix)
      or have a second view on it.

      \warning A view assignment will have your matrix use the same internal
      data as the original matrix. Changes on one side will affect the
      other. However, some methods like Zero() or reset() can, depending on
      the SparseMatrix flags, cut the connection. Do not rely on the view if
      you change one of these matrices! The view assignment is meant to
      provide a slim copy operation that transfers ownership from one matrix
      to the other before the original matrix is destroyed. Do a deep copy if
      both matrices are meant to live on.

      \param access how to treat this assignment: Copy or View
      \param mat matrix to assign from
     */
    void assign(DataAccess access, const SparseMatrix& mat);

    /** \name FE methods */
    //@{

    /// set all matrix entries to zero
    void zero() override;

    /// throw away the matrix and its graph and start anew
    void reset() override;

    /// assemble method, if ONLY local values are assembled
    void assemble(int eid, const std::vector<int>& lmstride,
        const Core::LinAlg::SerialDenseMatrix& Aele, const std::vector<int>& lm,
        const std::vector<int>& lmowner) override
    {
      assemble(eid, lmstride, Aele, lm, lmowner, lm);
    }

    /// assemble method, if ONLY local values are assembled
    virtual void assemble(int eid, const Core::LinAlg::SerialDenseMatrix& Aele,
        const std::vector<int>& lm, const std::vector<int>& lmowner)
    {
      assemble(eid, Aele, lm, lmowner, lm);
    }

    /// assemble method, if ONLY local values are assembled
    void assemble(int eid, const std::vector<int>& lmstride,
        const Core::LinAlg::SerialDenseMatrix& Aele, const std::vector<int>& lmrow,
        const std::vector<int>& lmrowowner, const std::vector<int>& lmcol) override;

    /// assemble method, if ONLY local values are assembled
    void assemble(int eid, const Core::LinAlg::SerialDenseMatrix& Aele,
        const std::vector<int>& lmrow, const std::vector<int>& lmrowowner,
        const std::vector<int>& lmcol);

    /// single value assemble used by BlockSparseMatrix
    void assemble(double val, int rgid, int cgid) override;


    /*
     * \brief Set a single value
     *
     * This method inserts a new entry if it does not yet exist. If the entry
     * already exists, it is overwritten.
     *
     * \params[in] val Value to insert
     * \params[in] rgid row position
     * \params[in] cgid column position
     */
    void set_value(double val, int rgid, int cgid);

    /*!
     * \brief Assemble method for an FE-style sparse matrix
     *
     * This method is also able to handle the assembly of nonlocal values.
     * It sets the doGlobalAssemble-flag to true and causes the
     * GlobalAssemble() method to redistribute the non-local
     * values to their owning procs, such that fill_complete can be safely
     * called on this matrix.
     *
     * \note This methods checks if rowowner == myrank. Only in this case
     * values are set. This is needed if the method is called in a loop over
     * column elements (which is the standard in 4C) to avoid multiple same entries.
     */
    void fe_assemble(const Core::LinAlg::SerialDenseMatrix& Aele, const std::vector<int>& lmrow,
        const std::vector<int>& lmrowowner, const std::vector<int>& lmcol);

    /*!
     * \brief Assemble method for an FE-style sparse matrix
     *
     * This method is also able to handle the assembly of nonlocal values.
     * It sets the doGlobalAssemble-flag to true and causes the
     * GlobalAssemble() method to redistribute the non-local
     * values to their owning procs, such that fill_complete can be safely
     * called on this matrix.
     */
    void fe_assemble(const Core::LinAlg::SerialDenseMatrix& Aele, const std::vector<int>& lmrow,
        const std::vector<int>& lmcol);

    /*!
     * \brief Assemble method for an FE-style sparse matrix
     *
     * This method is also able
     * to handle the assembly of nonlocal values.
     * It sets the doGlobalAssemble-flag to true and causes the
     * GlobalAssemble -method() to redistribute the non-local
     * values to their owning procs, such that fill_complete can be safely
     * called on this matrix.
     */
    void fe_assemble(double val, int rgid, int cgid);

    /*!
      The GlobalAssembleMethod() distributes nonlocal values to their owning procs
      for Epetra_FECrsMatrices.
      Afterwards Fillcomplete is called such as for Epetra_CrsMatrices.
     */
    void complete(OptionsMatrixComplete options_matrix_complete = {}) override;

    /*!
      The GlobalAssembleMethod() distributes nonlocal values to their owning procs
      for Epetra_FECrsMatrices.
      Afterwards Fillcomplete is called such as for Epetra_CrsMatrices.
     */
    void complete(const Core::LinAlg::Map& domainmap, const Core::LinAlg::Map& rangemap,
        OptionsMatrixComplete options_matrix_complete = {}) override;

    void un_complete() override;

    void apply_dirichlet(
        const Core::LinAlg::Vector<double>& dbctoggle, bool diagonalblock = true) override;

    /// Apply dirichlet boundary condition to a matrix.
    ///
    ///  This method blanks the rows associated with Dirichlet DOFs
    ///  and puts a 1.0 at the diagonal entry if diagonlblock==true.
    ///  Only the rows are blanked, the columns are not touched.
    ///  We are left with a non-symmetric matrix, if the original
    ///  matrix was symmetric. However, the blanking of columns is computationally
    ///  quite expensive, because the matrix is stored in a sparse and distributed
    ///  manner.
    void apply_dirichlet(const Core::LinAlg::Map& dbctoggle, bool diagonalblock = true) override;

    /// Apply dirichlet boundary condition to a matrix using a #trafo matrix
    ///
    /// This method the same as the method #ApplyDirichlet, but instead of
    /// 1.0 on diagonal, the corresponding row of #trafo is inserted. This
    /// is needed to treat efficiently Dirichlet BCs with local co-ordinate systems.
    /// The transformation matrix #trafo basically holds rotation matrices
    /// for the DOFs of the nodes.
    void apply_dirichlet_with_trafo(const Core::LinAlg::SparseMatrix& trafo,
        const Core::LinAlg::Map& dbctoggle, bool diagonalblock = true, bool complete = true);

    /// create matrix that contains all Dirichlet lines from my
    std::shared_ptr<SparseMatrix> extract_dirichlet_rows(
        const Core::LinAlg::Vector<double>& dbctoggle);

    /// create matrix that contains all Dirichlet lines from my
    std::shared_ptr<SparseMatrix> extract_dirichlet_rows(const Core::LinAlg::Map& dbctoggle);

    //@}

    /** \name Matrix Properties Query Methods */
    //@{

    /// Whether Dirichlet conditions should result in a trimmed graph row
    /*!
      ML requires rows of length 1 to recognize Dirichlet lines. However it is
      an expensive operation to apply Dirichlet conditions in this case.
     */
    bool explicit_dirichlet() const { return explicitdirichlet_; }

    /// Whether the matrix graph should be saved when the matrix is zeroed
    /*!
      Saving the graph will result in constructing new matrices in Filled()
      state. This speeds up assembling but limits assembling to the current
      graph.
     */
    bool save_graph() const { return savegraph_; }

    /// Return matrix type
    MatrixType get_matrixtype() const { return matrixtype_; }

    //@}

    /** \name Attribute access functions */
    //@{

    /// Returns a character string describing the operator.
    const char* Label() const override;

    //@}

    /** \name Utility functions */
    //@{

    /// Add a (transposed) Epetra_CrsMatrix to another: (*this) = (*this)*scalarB + A(^T)*scalarA
    /*!

    Add one matrix to another. the matrix (*this) to be added to must not be
    completed. Sparsity patterns of A and (*this) need not match and A and (*this) can be
    nonsymmetric in value and pattern.  Row map of A has to be a
    processor-local subset of the row map of (*this).

    \note This is a true parallel add, even in the transposed case!

    \param A          (in)     : Matrix to add to B (must have Filled()==true)
    \param transposeA (in)     : flag indicating whether transposed of A should be used
    \param scalarA    (in)     : scaling factor for A
    \param scalarB    (in)     : scaling factor for B
    */
    // void add(const SparseMatrix& A, const bool transposeA, const double scalarA,
    //     const double scalarB);

    //@}

    /*! \brief return the internal Epetra_Operator

    The internal Epetra_Operator here is the internal Epetra_CrsMatrix or Epetra_FECrsMatrix. This
    way the solver can down-cast to Epetra_CrsMatrix or Epetra_FECrsMatrix and access the matrix
    rows directly.

    \note This method is here for performance reasons.
   */
    Epetra_Operator& epetra_operator() override { return *sysmat_; }

    /// return the internal Epetra matrix as Epetra_Operator
    const Epetra_Operator& epetra_operator() const { return *sysmat_; }

    /// return the internal Epetra_CrsMatrix or Epetra_FECrsMatrix
    /// (down-cast from Epetra_CrsMatrix !) (you should not need this!)
    Epetra_CrsMatrix& epetra_matrix() { return *sysmat_; }

    /// return the internal Epetra_CrsMatrix or Epetra_FECrsMatrix
    /// (down-cast from Epetra_CrsMatrix !) (you should not need this!)
    const Epetra_CrsMatrix& epetra_matrix() const { return *sysmat_; }

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
     *  */

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

    /// Returns the number of rows locally owned.
    int num_my_rows() const { return sysmat_->NumMyRows(); }

    /// Returns the current number of nonzero entries in specified local row on this processor.
    int num_my_entries(int my_row) const { return sysmat_->NumMyEntries(my_row); }

    /// Returns the number of global rows.
    int num_global_rows() const { return sysmat_->NumGlobalRows(); }

    /// Returns the number of global rows.
    int num_global_cols() const { return sysmat_->NumGlobalCols(); }

    /// Returns the number of nonzero entries in a global row.
    int num_global_entries(int global_row) const { return sysmat_->NumGlobalEntries(global_row); }

    /// Returns the maximum number of nonzero entries across all rows on this processor.
    int max_num_entries() const;

    /// Returns the maximum number of nonzero entries across all rows on all processors.
    int global_max_num_entries() const;

    /// Returns the global number of nonzeros.
    int num_my_nonzeros() const { return sysmat_->NumMyNonzeros(); }

    /// Returns the global number of nonzeros.
    int num_global_nonzeros() const { return sysmat_->NumGlobalNonzeros(); }

    /// Returns the number of allocated entries in a global row.
    int num_allocated_global_entries(int global_row) const
    {
      return sysmat_->NumAllocatedGlobalEntries(global_row);
    }

    /// Returns the number of global nonzero diagonal entries, based on global row/column index
    /// comparisons.
    int num_global_diagonals() const { return sysmat_->NumGlobalDiagonals(); };

    /// Returns the global row index for give local row index, returns IndexBase-1 if we don't have
    /// this local row.
    int global_row_index(int local_row_index) const { return sysmat_->GRID(local_row_index); }

    /// Returns the Core::LinAlg::Map object associated with the rows of this matrix.
    const Core::LinAlg::Map& row_map() const { return row_map_.sync(sysmat_->RowMap()); }

    /// Returns the Core::LinAlg::Map object that describes the set of column-indices that appear in
    /// each processor's locally owned matrix rows.
    const Core::LinAlg::Map& col_map() const { return column_map_.sync(sysmat_->ColMap()); }

    /// Returns the Core::LinAlg::Map object associated with the domain of this matrix operator.
    const Map& domain_map() const override { return domain_map_.sync(sysmat_->DomainMap()); }

    /// Returns the Core::LinAlg::Map object associated with the range of this matrix operator.
    const Core::LinAlg::Map& range_map() const { return range_map_.sync(sysmat_->RangeMap()); }


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
    void multiply(
        bool TransA, const Core::LinAlg::Vector<double>& x, Core::LinAlg::Vector<double>& y) const;

    /// Returns the result of a Epetra_CrsMatrix multiplied by a Epetra_MultiVector X in Y.
    int multiply(bool TransA, const Core::LinAlg::MultiVector<double>& X,
        Core::LinAlg::MultiVector<double>& Y) const override;

    /// Scales the Core::LinAlg::SparseMatrix on the left with a Core::LinAlg::Vector<double> x.
    void left_scale(const Core::LinAlg::Vector<double>& x);

    /// Scales the Core::LinAlg::SparseMatrix on the right with a Core::LinAlg::Vector<double> x.
    void right_scale(const Core::LinAlg::Vector<double>& x);

    // Computes the inverse of the sum of absolute values of the rows.
    void inv_row_sums(Core::LinAlg::Vector<double>& x) const;

    // Computes the inverse of the sum of absolute values of the columns.
    void inv_col_sums(Core::LinAlg::Vector<double>& x) const;

    //@}

    /** \name Insertion/Replace/SumInto methods */
    //@{

    /// Initialize all values in the matrix with constant value.
    void put_scalar(double ScalarConstant);

    /// Multiply all values in the matrix by a constant value (in place: A <- ScalarConstant * A).
    int scale(double ScalarConstant) override;

    /// Replaces diagonal values of the matrix with those in the user-provided vector.
    int replace_diagonal_values(const Core::LinAlg::Vector<double>& Diagonal);

    /// Inserts values into a local row.
    void insert_my_values(int my_row, int num_entries, const double* values, const int* indices);

    /// Sum values into a local row.
    void sum_into_my_values(int my_row, int num_entries, const double* values, const int* indices);

    /// Replaces values in a local row.
    void replace_my_values(int my_row, int num_entries, const double* values, const int* indices);

    /// Replaces values in a global row.
    void replace_global_values(
        int global_row, int num_entries, const double* values, const int* indices);

    /// Inserts values into a global row.
    void insert_global_values(
        int global_row, int num_entries, const double* values, const int* indices);

    /// Sum values into a global row.
    void sum_into_global_values(
        int global_row, int num_entries, const double* values, const int* indices);

    //@}

    /** \name Extraction methods */
    //@{

    /// Returns a copy of the main diagonal in a user-provided vector.
    void extract_diagonal_copy(Core::LinAlg::Vector<double>& Diagonal) const;

    /// Returns a copy of the values and indices of a local row.
    void extract_my_row_copy(
        int my_row, int length, int& num_entries, double* values, int* indices) const;

    /// Returns a copy of the values and indices of a global row.
    void extract_global_row_copy(
        int global_row, int length, int& num_entries, double* values, int* indices) const;

    /// Returns a view of the values and indices of a local row.
    void extract_my_row_view(int my_row, int& num_entries, double*& values, int*& indices) const;

    /// Returns a view of the values and indices of a global row.
    void extract_global_row_view(
        int global_row, int& num_entries, double*& values, int*& indices) const;

    //@}

    /** \name Import / Export methods */
    //@{
    void import(const SparseMatrix& A, const Core::LinAlg::Import& importer,
        Core::LinAlg::CombineMode combine_mode)
    {
      ASSERT_EPETRA_CALL(sysmat_->Import(A.epetra_matrix(), importer.get_epetra_import(),
          Core::LinAlg::to_epetra_combine_mode(combine_mode)));
    }

    void import(const SparseMatrix& A, const Core::LinAlg::Export& exporter,
        Core::LinAlg::CombineMode combine_mode)
    {
      ASSERT_EPETRA_CALL(sysmat_->Import(A.epetra_matrix(), exporter.get_epetra_export(),
          Core::LinAlg::to_epetra_combine_mode(combine_mode)));
    }

    //@}

    /// Add one operator to another
    void add(const Core::LinAlg::SparseOperator& A, const bool transposeA, const double scalarA,
        const double scalarB) override;

    /// Add one SparseMatrixBase to another
    void add_other(Core::LinAlg::SparseMatrix& B, const bool transposeA, const double scalarA,
        const double scalarB) const override;

    /// Add one BlockSparseMatrix to another
    void add_other(Core::LinAlg::BlockSparseMatrixBase& B, const bool transposeA,
        const double scalarA, const double scalarB) const override;

    //! Print to user-provided output stream
    void print(std::ostream& os) const { sysmat_->Print(os); }

   private:
    /*! \brief Accumulate values into a global row, inserting if summation fails.
     *
     * This helper centralizes the logic for handling Epetra's return codes when
     * adding values to a matrix. Depending on the error code, it will either
     * sum into an existing row or insert new entries.
     *
     * It exists primarily to work around cases where the matrix was not
     * preallocated with sufficient storage and should be removed in the future.
     */
    void sum_or_insert_global_values(
        int global_row, int num_entries, const double* values, const int* indices);

    /*! \brief Replace values of a global row, inserting if replacement fails.
     *
     * This helper centralizes the logic for handling Epetra's return codes when
     * replacing values of a matrix. Depending on the error code, it will either
     * replace an existing row or insert new entries.
     *
     * It exists primarily to work around cases where the matrix was not
     * preallocated with sufficient storage and should be removed in the future.
     */
    void replace_or_insert_global_values(
        int global_row, int num_entries, const double* values, const int* indices);

    /// internal epetra matrix (Epetra_CrsMatrix or Epetra_FECrsMatrix)
    std::shared_ptr<Epetra_CrsMatrix> sysmat_;

    mutable View<const Map> range_map_;
    mutable View<const Map> row_map_;
    mutable View<const Map> domain_map_;
    mutable View<const Map> column_map_;

    /// saved graph (if any)
    std::shared_ptr<Core::LinAlg::Graph> graph_;

    /// Dirichlet row map (if known)
    std::shared_ptr<Core::LinAlg::MultiMapExtractor> dbcmaps_;

    /// whether to modify the matrix graph on apply Dirichlet
    bool explicitdirichlet_;

    /// whether to save the graph and assemble to a filled matrix next time
    bool savegraph_;

    /// matrix type (Epetra_CrsMatrix or Epetra_FECrsMatrix)
    MatrixType matrixtype_;
  };

  //! Cast matrix of type SparseOperator to const SparseMatrix and check in debug mode if cast was
  //! successful
  std::shared_ptr<const Core::LinAlg::SparseMatrix> cast_to_const_sparse_matrix_and_check_success(
      std::shared_ptr<const Core::LinAlg::SparseOperator> input_matrix);

  //! Cast matrix of type SparseOperator to SparseMatrix and check in debug mode if cast was
  //! successful
  std::shared_ptr<Core::LinAlg::SparseMatrix> cast_to_sparse_matrix_and_check_success(
      std::shared_ptr<Core::LinAlg::SparseOperator> input_matrix);
}  // namespace Core::LinAlg



FOUR_C_NAMESPACE_CLOSE

#endif

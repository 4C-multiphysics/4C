// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINALG_SPARSEMATRIX_HPP
#define FOUR_C_LINALG_SPARSEMATRIX_HPP

#include "4C_config.hpp"

#include "4C_linalg_graph.hpp"
#include "4C_linalg_mapextractor.hpp"
#include "4C_linalg_sparsematrixbase.hpp"
#include "4C_utils_exceptions.hpp"

#include <Epetra_FECrsMatrix.h>

class Epetra_CrsMatrix;

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

    \author u.kue
    \date 02/08
   */
  class SparseMatrix : public SparseMatrixBase
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
    SparseMatrix(const Epetra_Map& rowmap, const int npr, bool explicitdirichlet = true,
        bool savegraph = false, MatrixType matrixtype = CRS_MATRIX);

    /// construction of sparse matrix using an individual estimate for number of non-zeros per row
    SparseMatrix(const Epetra_Map& rowmap, std::vector<int>& numentries,
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
    SparseMatrix(const SparseMatrix& mat, DataAccess access = Copy);

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

    /// destroy the underlying Epetra objects
    virtual bool destroy(bool throw_exception = true);

    /// assemble method for Epetra_CrsMatrices, if ONLY local values are assembled
    void assemble(int eid, const std::vector<int>& lmstride,
        const Core::LinAlg::SerialDenseMatrix& Aele, const std::vector<int>& lm,
        const std::vector<int>& lmowner) override
    {
      assemble(eid, lmstride, Aele, lm, lmowner, lm);
    }

    /// assemble method for Epetra_CrsMatrices, if ONLY local values are assembled
    virtual void assemble(int eid, const Core::LinAlg::SerialDenseMatrix& Aele,
        const std::vector<int>& lm, const std::vector<int>& lmowner)
    {
      assemble(eid, Aele, lm, lmowner, lm);
    }

    /// assemble method for Epetra_CrsMatrices, if ONLY local values are assembled
    void assemble(int eid, const std::vector<int>& lmstride,
        const Core::LinAlg::SerialDenseMatrix& Aele, const std::vector<int>& lmrow,
        const std::vector<int>& lmrowowner, const std::vector<int>& lmcol) override;

    /// assemble method for Epetra_CrsMatrices, if ONLY local values are assembled
    void assemble(int eid, const Core::LinAlg::SerialDenseMatrix& Aele,
        const std::vector<int>& lmrow, const std::vector<int>& lmrowowner,
        const std::vector<int>& lmcol);

    /// single value assemble used by BlockSparseMatrix
    void assemble(double val, int rgid, int cgid) override;


    /*
     * \brief Set a single value in a Epetra_FECrsMatrix
     *
     * This method inserts a new entry in a EpetraFECrsMatrix if it does not yet exist. If the entry
     * already exists, it is overwritten.
     *
     * \params[in] val Value to insert
     * \params[in] rgid row position
     * \params[in] cgid column position
     */
    void set_value(double val, int rgid, int cgid);

    /*!
      Assemble method for an Epetra_FECrsMatrix.
      This method is also able to handle the assembly of nonlocal values.
      It sets the doGlobalAssemble-flag to true and causes the
      GlobalAssemble() method to redistribute the non-local
      values to their owning procs, such that fill_complete can be safely
      called on this matrix.

      NOTE: This methods checks if rowowner == myrank. Only in this case
      values are set. This is needed if the method is called in a loop over
      column elements (which is the standard in 4C) to avoid multiple same entries.
     */
    void fe_assemble(const Core::LinAlg::SerialDenseMatrix& Aele, const std::vector<int>& lmrow,
        const std::vector<int>& lmrowowner, const std::vector<int>& lmcol);

    /*!
      Assemble method for an Epetra_FECrsMatrices.
      This method is also able to handle the assembly of nonlocal values.
      It sets the doGlobalAssemble-flag to true and causes the
      GlobalAssemble() method to redistribute the non-local
      values to their owning procs, such that fill_complete can be safely
      called on this matrix.
     */
    void fe_assemble(const Core::LinAlg::SerialDenseMatrix& Aele, const std::vector<int>& lmrow,
        const std::vector<int>& lmcol);

    /*!
      Assemble method for an Epetra_FECrsMatrices.
      This method is also able
      to handle the assembly of nonlocal values.
      It sets the doGlobalAssemble-flag to true and causes the
      GlobalAssemble -method() to redistribute the non-local
      values to their owning procs, such that fill_complete can be safely
      called on this matrix.
     */
    void fe_assemble(double val, int rgid, int cgid);


    /*!
      The GlobalAssembleMethod() distributes nonlocal values to their owning procs
      for Epetra_FECrsMatrices.
      Afterwards Fillcomplete is called such as for Epetra_CrsMatrices.

      @param enforce_complete Enforce fill_complete() even though the matrix might already be filled
     */
    void complete(bool enforce_complete = false) override;

    /*!
      The GlobalAssembleMethod() distributes nonlocal values to their owning procs
      for Epetra_FECrsMatrices.
      Afterwards Fillcomplete is called such as for Epetra_CrsMatrices.

      @param enforce_complete Enforce fill_complete() even though the matrix might already be filled
     */
    void complete(const Epetra_Map& domainmap, const Epetra_Map& rangemap,
        bool enforce_complete = false) override;

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
    void apply_dirichlet(const Epetra_Map& dbctoggle, bool diagonalblock = true) override;

    /// Apply dirichlet boundary condition to a matrix using a #trafo matrix
    ///
    /// This method the same as the method #ApplyDirichlet, but instead of
    /// 1.0 on diagonal, the corresponding row of #trafo is inserted. This
    /// is needed to treat efficiently Dirichlet BCs with local co-ordinate systems.
    /// The transformation matrix #trafo basically holds rotation matrices
    /// for the DOFs of the nodes.
    void apply_dirichlet_with_trafo(const Core::LinAlg::SparseMatrix& trafo,
        const Epetra_Map& dbctoggle, bool diagonalblock = true, bool complete = true);

    /// create matrix that contains all Dirichlet lines from my
    std::shared_ptr<SparseMatrix> extract_dirichlet_rows(
        const Core::LinAlg::Vector<double>& dbctoggle);

    /// create matrix that contains all Dirichlet lines from my
    std::shared_ptr<SparseMatrix> extract_dirichlet_rows(const Epetra_Map& dbctoggle);

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
    using SparseMatrixBase::add;

    void add(const SparseMatrixBase& A, const bool transposeA, const double scalarA,
        const double scalarB);

    //@}

   private:
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

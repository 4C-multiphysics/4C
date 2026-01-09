// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINALG_SPARSEOPERATOR_HPP
#define FOUR_C_LINALG_SPARSEOPERATOR_HPP

#include "4C_config.hpp"

#include "4C_linalg_map.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_utils_shared_ptr_from_ref.hpp"

#include <Epetra_Operator.h>

#include <memory>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  // forward declarations
  class BlockSparseMatrixBase;
  class SparseMatrix;

  //! type of global system matrix in global system of equations
  enum class MatrixType
  {
    undefined,           /*!< Type of system matrix is undefined. */
    sparse,              /*!< System matrix is a sparse matrix. */
    block_field,         /*!< System matrix is a block matrix that consists of NxN matrices.
                            In the simplest case, where each (physical) field is represented by just one
                            sparse matrix, N equals the number of (physical) fields of your problem.
                            However, it is also possible that the matrix of each (physical) field itself is
                            a block matrix, then of course N is the number of all sub matrix blocks*/
    block_condition,     /*!< System matrix is a block matrix that consists of NxN sparse matrices.
                            How the system matrix is divided has to be defined by a condition (e.g.
                            \link ::Core::Conditions::ScatraPartitioning ScatraPartitioning \endlink.)*/
    block_condition_dof, /*!< System matrix is a block matrix that consists of NxN sparse
                            matrices. Each of the blocks as created by block_condition is
                            further subdivided by the dofs, meaning e.g. for two dofs per node
                            each 'original' block is divided into 2 blocks. */
  };

  //! \brief Options for matrix completion
  struct OptionsMatrixComplete
  {
    //! enforce a lightweight fill_complete() even though the matrix might already have been filled
    bool enforce_complete = false;

    //! make consecutive row index sections contiguous, minimize internal storage used for
    //! constructing graph
    bool optimize_data_storage = true;
  };

  /// Linear operator interface enhanced for use in FE simulations
  /*!

    The point in FE simulations is that you have to assemble (element)
    contributions to the global matrix, apply Dirichlet conditions in some way
    and finally solve the completed system of equations.

    Here we have an interface that has different implementations. The obvious
    one is the SparseMatrix, a single Epetra_CrsMatrix in a box, another one
    is BlockSparseMatrix, a block matrix build from a list of SparseMatrix.

   */
  class SparseOperator
  {
   public:
    virtual ~SparseOperator() = default;

    /// return the internal Epetra_Operator
    //! \warning Only low level solver routines are interested in the internal Epetra_Operator.
    virtual Epetra_Operator& epetra_operator() = 0;

    /// set matrix to zero
    virtual void zero() = 0;

    /// throw away the matrix and its graph and start anew
    virtual void reset() = 0;

    virtual MPI_Comm get_comm() const = 0;

    /// Assemble a Core::LinAlg::SerialDenseMatrix into a matrix with striding
    /*!

    This is an individual call.  Will only assemble locally and will never
    do any communication.  All values that cannot be assembled locally will
    be ignored.  Will use the communicator and rowmap from matrix to
    determine ownerships.  Local matrix Aele has to be square.

    If matrix is Filled(), it stays so and you can only assemble to places
    already masked. An attempt to assemble into a non-existing place is a
    grave mistake.

    If matrix is not Filled(), the matrix is enlarged as required.

    \note Assembling to a non-Filled() matrix is much more expensive than to
    a Filled() matrix. If the sparse mask does not change it pays to keep
    the matrix around and assemble into the Filled() matrix.

    The first parameter \p eid is purely for performance enhancements. Plain
    sparse matrices do not know about finite elements and do not use the
    element id at all. However, BlockSparseMatrix might be created with
    specialized, problem specific assembling strategies. And these strategies
    might gain considerable performance advantages from knowing the element
    id.

    \param eid (in) : element gid
    \param Aele (in) : dense matrix to be assembled
    \param lm (in) : vector with gids
    \param lmowner (in) : vector with owner procs of gids
    */
    virtual void assemble(int eid, const std::vector<int>& lmstride,
        const Core::LinAlg::SerialDenseMatrix& Aele, const std::vector<int>& lm,
        const std::vector<int>& lmowner)
    {
      assemble(eid, lmstride, Aele, lm, lmowner, lm);
    }

    /// Assemble a Core::LinAlg::SerialDenseMatrix into a matrix with striding
    /*!

      This is an individual call.
      Will only assemble locally and will never do any communication.
      All values that can not be assembled locally will be ignored.
      Will use the communicator and rowmap from matrix A to determine ownerships.
      Local matrix Aele may be \b square or \b rectangular.

      If matrix is Filled(), it stays so and you can only assemble to places
      already masked. An attempt to assemble into a non-existing place is a
      grave mistake.

      If matrix is not Filled(), the matrix is enlarged as required.

      \note Assembling to a non-Filled() matrix is much more expensive than to
      a Filled() matrix. If the sparse mask does not change it pays to keep
      the matrix around and assemble into the Filled() matrix.

      \note The user must provide an \b additional input vector 'lmcol'
      containing the column gids for assembly separately!

      The first parameter \p eid is purely for performance enhancements. Plain
      sparse matrices do not know about finite elements and do not use the
      element id at all. However, BlockSparseMatrix might be created with
      specialized, problem specific assembling strategies. And these
      strategies might gain considerable performance advantages from knowing
      the element id.

      \param eid (in) : element gid
      \param Aele (in)       : dense matrix to be assembled
      \param lmrow (in)      : vector with row gids
      \param lmrowowner (in) : vector with owner procs of row gids
      \param lmcol (in)      : vector with column gids
    */
    virtual void assemble(int eid, const std::vector<int>& lmstride,
        const Core::LinAlg::SerialDenseMatrix& Aele, const std::vector<int>& lmrow,
        const std::vector<int>& lmrowowner, const std::vector<int>& lmcol) = 0;

    /// single value assemble using gids
    virtual void assemble(double val, int rgid, int cgid) = 0;

    /// If Complete() has been called, this query returns true, otherwise it returns false.
    virtual bool filled() const = 0;

    /// Call fill_complete on a matrix
    virtual void complete(OptionsMatrixComplete options_matrix_complete = {}) = 0;

    /// Call fill_complete on a matrix (for rectangular and square matrices)
    virtual void complete(const Core::LinAlg::Map& domainmap, const Core::LinAlg::Map& rangemap,
        OptionsMatrixComplete options_matrix_complete = {}) = 0;

    /// Undo a previous Complete() call
    virtual void un_complete() = 0;

    /// Apply dirichlet boundary condition to a matrix
    virtual void apply_dirichlet(
        const Core::LinAlg::Vector<double>& dbctoggle, bool diagonalblock = true) = 0;

    /// Apply dirichlet boundary condition to a matrix
    ///
    ///  This method blanks the rows associated with Dirichlet DOFs
    ///  and puts a 1.0 at the diagonal entry if diagonalblock==true.
    ///  Only the rows are blanked, the columns are not touched.
    ///  We are left with a non-symmetric matrix, if the original
    ///  matrix was symmetric. However, the blanking of columns is computationally
    ///  quite expensive, because the matrix is stored in a sparse and distributed
    ///  manner.
    virtual void apply_dirichlet(const Core::LinAlg::Map& dbcmap, bool diagonalblock = true) = 0;

    /// Returns the Epetra_Map object associated with the (full) domain of this operator.
    virtual const Map& domain_map() const = 0;

    /// Add one operator to another
    virtual void add(const Core::LinAlg::SparseOperator& A, const bool transposeA,
        const double scalarA, const double scalarB) = 0;

    /// Multiply all values by a constant value (in place: A <- ScalarConstant * A).
    virtual void scale(double ScalarConstant) = 0;

    /// Matrix-vector product
    virtual void multiply(bool TransA, const Core::LinAlg::MultiVector<double>& X,
        Core::LinAlg::MultiVector<double>& Y) const = 0;
  };


}  // namespace Core::LinAlg

FOUR_C_NAMESPACE_CLOSE

#endif

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINALG_UTILS_SPARSE_ALGEBRA_ASSEMBLE_HPP
#define FOUR_C_LINALG_UTILS_SPARSE_ALGEBRA_ASSEMBLE_HPP

#include "4C_config.hpp"

#include "4C_linalg_blocksparsematrix.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_graph.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_vector.hpp"

#include <Epetra_Map.h>

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  /*!
   \brief Assemble an Core::LinAlg::SerialDenseMatrix into an Epetra_CrsMatrix

   This is an individual call.
   Will only assemble locally and will never do any communication.
   All values that can not be assembled locally will be ignored.
   Will use the communicator and rowmap from matrix A to determine ownerships.
   Local matrix Aele may be \b square or \b rectangular.

   This version of 'Assemble' does not work for a matrix A that is already
   Filled()! If matrix A is not Filled(), it will be enlarged as required.

   \note The user must provide an \b additional input vector 'lmcol'
   containing the column gids for assembly separately!

   \param A (out)         : Sparse matrix to be assembled on
   \param Aele (in)       : dense matrix to be assembled
   \param lmrow (in)      : vector with row gids
   \param lmrowowner (in) : vector with owner procs of row gids
   \param lmcol (in)      : vector with column gids
   */
  void assemble(Epetra_CrsMatrix& A, const Core::LinAlg::SerialDenseMatrix& Aele,
      const std::vector<int>& lmrow, const std::vector<int>& lmrowowner,
      const std::vector<int>& lmcol);

  /*!
   \brief Assemble an Core::LinAlg::SerialDenseVector into an Core::LinAlg::Vector<double>

   This is an individual call.
   Will only assemble locally and will never do any communication.
   All values that can not be assembled locally will be ignored.
   Will use the communicator from vector V to determine ownerships.

   \param V (out)   : Vector to be assembled on
   \param Vele (in) : dense vector to be assembled
   \param lm (in) : vector with gids
   \param lmowner (in) : vector with owner procs of gids
   */
  void assemble(Core::LinAlg::Vector<double>& V, const Core::LinAlg::SerialDenseVector& Vele,
      const std::vector<int>& lm, const std::vector<int>& lmowner);

  /*!
   \brief Assemble a Core::LinAlg::SerialDenseVector into a Core::LinAlg::MultiVector<double>

   This is an individual call.
   Will only assemble locally and will never do any communication.
   All values that can not be assembled locally will be ignored.
   Will use the communicator from vector V to determine ownerships.

   \param V (out)   : Vector to be assembled on
   \param n (in)   : column index of MultiVector to be assembled on
   \param Vele (in) : dense vector to be assembled
   \param lm (in) : vector with gids
   \param lmowner (in) : vector with owner procs of gids
   */
  void assemble(Core::LinAlg::MultiVector<double>& V, const int n,
      const Core::LinAlg::SerialDenseVector& Vele, const std::vector<int>& lm,
      const std::vector<int>& lmowner);

  /*! \brief Assemble a source Core::LinAlg::Vector<double> into a target
   * Core::LinAlg::Vector<double>
   *
   *  The map of the source vector has to be a sub-map of the target vector and
   *  the maps must have the same processor distribution. This method does not
   *  build up any communication between different processors!
   *  The entries of the source vector are added to the target vector:
   *
   *      target_vector[GID] = source_scalar * source[GID] + target_scalar * target[GID]
   *
   *  \note The remaining GIDs, which are no part of the source map stay untouched!
   *
   *  \param scalar_target (in) : scale the target entries by this factor
   *  \param target        (out): target vector (part of the source)
   *  \param scalar_source (in) : scale the source entries by this factor
   *  \param source        (in) : source vector
   *
   *  \author hiermeier \date 03/17 */
  void assemble_my_vector(double scalar_target, Core::LinAlg::Vector<double>& target,
      double scalar_source, const Core::LinAlg::Vector<double>& source);

  /*!
   \brief Apply dirichlet boundary condition to a linear system of equations

   Modifies a system of equations such that dirichlet boundary conditions are enforced.
   Prescribed dirichlet BC values are supplied in dbcval and dbctoggle, where
   a prescribed value is dbcval[i] and dbctoggle[i] = 1.0. No BC is enforced in
   all places where dbctoggle[i] != 1.0.<br>
   Let us denote the \f$ A_{2 \times 2} \f$ blocks of \f$A\f$ by
   \f$A_{ff}, A_{fD}, A_{Df}, A_{DD}\f$, where \f$f\f$ stands for 'free' and
   \f$D\f$ stands for 'Dirichlet BC'. Then, after a call to this method<br>

   \f$ A_{ff} = A_{ff}, \f$<br>
   \f$ A_{fD} = A_{fD}, \f$<br>
   \f$ A_{Df} = 0_{Df}, \f$<br>
   \f$ A_{DD} = I_{DD}, \f$<br>
   \f$ x_{D} = dbcval_{D}, \f$<br>
   \f$ b_{D} = dbcval_{D} \f$<br>

   and

   \f$ A_{ff} x_f + A_{fD} x_D = b_f \f$<br>
   \f$ 0 x_f + I_{DD} x_D = x_D \f$.<br>

   \note The matrix is then nonsymmetric. When using iterative methods on this
   linear system of equations that depend on the symmetry of the matrix (such as e.g. CG),
   the initial guess supplied to the solver has to be exact at the
   Dirichlet BCs. This should be easy, as the values at the Dirichlet BCs
   are known.

   \note The mask of matrix \f$A\f$ is not modified. That is the
   entries in \f$A_{Df}\f$ and \f$A_{DD}\f$ are set to zero, not
   removed. This way the matrix can be reused in the next step.

   \param A         (in/out) : Matrix of Ax=b
   \param x         (in/out) : initial guess vector x of Ax=b
   \param b         (in/out) : rhs vector b of Ax=b
   \param dbcval    (in)     : vector holding prescribed dirichlet values
   \param dbctoggle (in)     : vector holding 1.0 where dirichlet should be applied
   and 0.0 everywhere else
     */
  void apply_dirichlet_to_system(Core::LinAlg::SparseOperator& A, Core::LinAlg::Vector<double>& x,
      Core::LinAlg::Vector<double>& b, const Core::LinAlg::Vector<double>& dbcval,
      const Core::LinAlg::Vector<double>& dbctoggle);

  /*!
   \brief Apply dirichlet boundary condition to a linear system of equations

   This is a flexible routine. The vectors x and dbcval might have different
   maps. The map does not need to contain all Dirichlet dofs.

   The purpose is to set Dirichlet values at a subset of all Dirichlet
   boundaries.

   \param A (in/out)         : Matrix of Ax=b
   \param x (in/out)         : vector x of Ax=b
   \param b (in/out)         : vector b of Ax=b
   \param dbcval (in)        : vector holding values that are supposed to be prescribed
   \param dbcmap (in)        : unique map of all dofs that should be constrained

   \pre The map dbcmap must be subset of the maps of the vectors.
   */
  void apply_dirichlet_to_system(Core::LinAlg::SparseOperator& A, Core::LinAlg::Vector<double>& x,
      Core::LinAlg::Vector<double>& b, const Core::LinAlg::Vector<double>& dbcval,
      const Epetra_Map& dbcmap);

  /*!
   \brief Apply dirichlet boundary condition to a linear system of equations

   This is a flexible routine. The vectors x and dbcval might have different
   maps. The map does not need to contain all Dirichlet dofs.

   The purpose is to set Dirichlet values at a subset of all Dirichlet
   boundaries.

   Special in this routine is the ability to insert rows of general rotation
   matrices (stored in #trafo) rather than simply put ones and zeros
   at the rows associated Dirichlet DOFs.

   \param A (in/out)         : Matrix of Ax=b
   \param x (in/out)         : vector x of Ax=b
   \param b (in/out)         : vector b of Ax=b
   \param trafo (in)         : global matrix holding rotation matrices to convert
   from global to local co-ordinate systems
   \param dbcval (in)        : vector holding values that are supposed to be prescribed
   \param dbcmap (in)        : unique map of all dofs that should be constrained

   \pre The map dbcmap must be subset of the maps of the vectors.
   */
  void apply_dirichlet_to_system(Core::LinAlg::SparseMatrix& A, Core::LinAlg::Vector<double>& x,
      Core::LinAlg::Vector<double>& b, const Core::LinAlg::SparseMatrix& trafo,
      const Core::LinAlg::Vector<double>& dbcval, const Epetra_Map& dbcmap);

  /*!
   \brief Apply dirichlet boundary condition to a linear system of equations


   \param x (in/out)         : vector x of Ax=b
   \param b (in/out)         : vector b of Ax=b
   \param dbcval (in)        : vector holding values that are supposed to be prescribed
   \param dbctoggle (in)     : vector holding 1.0 where dirichlet should be applied
   and 0.0 everywhere else
   */
  void apply_dirichlet_to_system(Core::LinAlg::Vector<double>& x, Core::LinAlg::Vector<double>& b,
      const Core::LinAlg::Vector<double>& dbcval, const Core::LinAlg::Vector<double>& dbctoggle);

  /*!
   \brief Apply dirichlet boundary condition to a linear system of equations

   This is a flexible routine. The vectors x and dbcval might have different
   maps. The dbcmap does not need to contain all Dirichlet dofs, but the vectors
   all dofs defined in it.

   The purpose is to set Dirichlet values at a subset of all Dirichlet
   boundaries.

   \param x (in/out)         : vector x of Ax=b
   \param b (in/out)         : vector b of Ax=b
   \param dbcval (in)        : vector holding values that are supposed to be prescribed
   \param dbcmap (in)        : unique map of all dofs that should be constrained

   \pre The map dbcmap must be subset of the maps of the vectors.
   */
  void apply_dirichlet_to_system(Core::LinAlg::Vector<double>& x, Core::LinAlg::Vector<double>& b,
      const Core::LinAlg::Vector<double>& dbcval, const Epetra_Map& dbcmap);

  /*!
   \brief Apply dirichlet boundary condition to a linear system of equations

   This is a flexible routine. The vectors x and dbcval might have different
   maps. The map does not need to contain all Dirichlet dofs.

   NOTE: Vector b does not need to contain all Dirichlet dofs defined in dbcmap

   The purpose is to set Dirichlet values at a subset of all Dirichlet
   boundaries.

   \param b (in/out)         : vector b of Ax=b
   \param dbcval (in)        : vector holding values that are supposed to be prescribed
   \param dbcmap (in)        : unique map of all dofs that should be constrained

   \pre The map dbcmap must be subset of the maps of the vectors.
   */
  void apply_dirichlet_to_system(Core::LinAlg::Vector<double>& b,
      const Core::LinAlg::Vector<double>& dbcval, const Epetra_Map& dbcmap);

  /*!
   \brief Convert a Dirichlet toggle vector in a Dirichlet map

   The purpose of the routine is a smooth transition from Dirichlet toggle vectors
   to Dirichlet condition maps. Eventually, this method should be removed.

   A Dirichlet toggle vector is a real vector which holds a 1.0 at DOF subjected
   to Dirichlet boundary conditions and a 0.0 at every remaining/free DOF.

   \param dbctoggle (in)     : the Dirichlet toggle vector
   \return MapExtractor object which stores the Dirichlet condition and remaining (other) DOF map

   \author bborn
   \date 10/08
   */
  std::shared_ptr<Core::LinAlg::MapExtractor> convert_dirichlet_toggle_vector_to_maps(
      const Core::LinAlg::Vector<double>& dbctoggle);

}  // namespace Core::LinAlg

FOUR_C_NAMESPACE_CLOSE

#endif

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINEAR_SOLVER_PRECONDITIONER_MUELU_HPP
#define FOUR_C_LINEAR_SOLVER_PRECONDITIONER_MUELU_HPP

#include "4C_config.hpp"

#include "4C_linalg_blocksparsematrix.hpp"
#include "4C_linear_solver_preconditioner_type.hpp"

#include <MueLu_Hierarchy.hpp>
#include <MueLu_UseDefaultTypes.hpp>
#include <Xpetra_MultiVector.hpp>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinearSolver
{
  /*! \brief General layout of the MueLu preconditioner working with sparse matrices
   * in CRS format.
   *
   * For more specialized versions (e.g. block systems) derive from this
   * class as it already owns all core components of a MueLu preconditioner.
   *
   * MueLu User's Guide: L. Berger-Vergiat, C. A. Glusa, J. J. Hu, M. Mayr,
   * A. Prokopenko, C. M. Siefert, R. S. Tuminaro, T. A. Wiesner:
   * MueLu User's Guide, Technical Report, Sandia National Laboratories, SAND2019-0537, 2019
   */
  class MueLuPreconditioner : public PreconditionerTypeBase
  {
   public:
    MueLuPreconditioner(Teuchos::ParameterList &muelulist);

    /*! \brief Create and compute the preconditioner
     *
     * The MueLu preconditioner only works for matrices of the type
     * Epetra_CrsMatrix. We check whether the input matrix is of proper type
     * and throw an error if not!
     *
     * This routine either re-creates the entire preconditioner from scratch or
     * it re-uses the existing preconditioner and only updates the fine level matrix
     * for the Krylov solver.
     *
     * It maintains backward compability to the ML interface!
     *
     * @param create Boolean flag to enforce (re-)creation of the preconditioner
     * @param matrix Epetra_CrsMatrix to be used as input for the preconditioner
     * @param x Solution of the linear system
     * @param b Right-hand side of the linear system
     */
    void setup(bool create, Epetra_Operator *matrix, Core::LinAlg::MultiVector<double> *x,
        Core::LinAlg::MultiVector<double> *b) override;

    //! linear operator used for preconditioning
    std::shared_ptr<Epetra_Operator> prec_operator() const final
    {
      return Core::Utils::shared_ptr_from_ref(*P_);
    }

   private:
    //! system of equations used for preconditioning used by P_ only
    Teuchos::RCP<Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>> pmatrix_;

   protected:
    //! MueLu parameter list
    Teuchos::ParameterList &muelulist_;

    //! preconditioner
    Teuchos::RCP<Epetra_Operator> P_;

    //! MueLu hierarchy
    Teuchos::RCP<MueLu::Hierarchy<Scalar, LocalOrdinal, GlobalOrdinal, Node>> H_;

  };  // class MueLuPreconditioner
}  // namespace Core::LinearSolver
FOUR_C_NAMESPACE_CLOSE

#endif

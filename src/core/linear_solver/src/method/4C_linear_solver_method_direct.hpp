// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINEAR_SOLVER_METHOD_DIRECT_HPP
#define FOUR_C_LINEAR_SOLVER_METHOD_DIRECT_HPP

#include "4C_config.hpp"

#include "4C_linalg_sparseoperator.hpp"
#include "4C_linear_solver_method.hpp"
#include "4C_linear_solver_method_projector.hpp"

#include <Amesos2.hpp>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinearSolver
{
  /// direct linear solver (using amesos)
  class DirectSolver : public SolverTypeBase
  {
   public:
    explicit DirectSolver(std::string solvertype);

    /*! \brief Setup the solver object
     *
     * @param A Matrix of the linear system
     * @param x Solution vector of the linear system
     * @param b Right-hand side vector of the linear system
     * @param refactor Boolean flag to enforce a refactorization of the matrix
     * @param reset Boolean flag to enforce a full reset of the solver object
     * @param projector Krylov projector
     */
    void setup(std::shared_ptr<Core::LinAlg::SparseOperator> matrix,
        std::shared_ptr<Core::LinAlg::MultiVector<double>> x,
        std::shared_ptr<Core::LinAlg::MultiVector<double>> b, const bool refactor, const bool reset,
        std::shared_ptr<Core::LinAlg::LinearSystemProjector> projector = nullptr) override;

    int solve() override;

    bool is_factored() { return factored_; }

   private:
    //! type/implementation of Amesos solver to be used
    const std::string solvertype_;

    //! flag indicating whether a valid factorization is stored
    bool factored_;

    //! initial guess and solution
    std::shared_ptr<Core::LinAlg::MultiVector<double>> x_;

    //! right hand side vector
    std::shared_ptr<Core::LinAlg::MultiVector<double>> b_;

    //! system of equations
    std::shared_ptr<Core::LinAlg::SparseMatrix> a_;

    //! an abstract Amesos2 solver that can be any of the concrete implementations
    Teuchos::RCP<Amesos2::Solver<Epetra_CrsMatrix, Epetra_MultiVector>> solver_;

    /*! \brief A projector applied before solving the linear systems
     *
     * Instead of solving Ax=b a projected system of the form P'APu=P'b is solved.
     * With P being the linear system projector.
     *
     * For example, for nearly singular systems, a Krylov projector can be used:
     *
     * P. Bochev and R. B. Lehoucq: On the Finite Element Solution of the Pure Neumann Problem,
     * SIAM Review, 47(1):50-66, 2005, http://dx.doi.org/10.1137/S0036144503426074
     *
     */
    std::shared_ptr<Core::LinAlg::LinearSystemProjector> projector_;
  };
}  // namespace Core::LinearSolver

FOUR_C_NAMESPACE_CLOSE

#endif

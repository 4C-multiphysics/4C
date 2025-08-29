// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINEAR_SOLVER_PRECONDITIONER_KRYLOVPROJECTION_HPP
#define FOUR_C_LINEAR_SOLVER_PRECONDITIONER_KRYLOVPROJECTION_HPP

#include "4C_config.hpp"

#include "4C_linalg_krylov_projector.hpp"
#include "4C_linear_solver_preconditioner_type.hpp"

#include <Epetra_Operator.h>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinearSolver
{
  /// krylov projection for undefined pressure value in incompressible fluids
  /*!
    This is not a preconditioner in a mathematical sense, but it fits the
    software framework nicely. A "real" preconditioner is wrapped.
  */
  class KrylovProjectionPreconditioner : public PreconditionerTypeBase
  {
   public:
    KrylovProjectionPreconditioner(std::shared_ptr<PreconditionerTypeBase> preconditioner,
        std::shared_ptr<Core::LinAlg::KrylovProjector> projector);

    void setup(Core::LinAlg::SparseOperator& matrix, const Core::LinAlg::MultiVector<double>& x,
        Core::LinAlg::MultiVector<double>& b) override;

    /// linear operator used for preconditioning
    std::shared_ptr<Epetra_Operator> prec_operator() const override { return p_; }

   private:
    std::shared_ptr<PreconditionerTypeBase> preconditioner_;

    /// Peter's projector object that does the actual work
    std::shared_ptr<Core::LinAlg::KrylovProjector> projector_;

    std::shared_ptr<Epetra_Operator> p_;
  };
}  // namespace Core::LinearSolver

FOUR_C_NAMESPACE_CLOSE

#endif

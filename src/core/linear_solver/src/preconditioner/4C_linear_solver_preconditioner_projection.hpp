// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINEAR_SOLVER_PRECONDITIONER_PROJECTION_HPP
#define FOUR_C_LINEAR_SOLVER_PRECONDITIONER_PROJECTION_HPP

#include "4C_config.hpp"

#include "4C_linalg_krylov_projector.hpp"
#include "4C_linear_solver_method_projector.hpp"
#include "4C_linear_solver_preconditioner_type.hpp"

#include <Epetra_Operator.h>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinearSolver
{
  /*!
   * A preconditioner that applies a linear projection first and then the usual preconditioner.
   */
  class ProjectionPreconditioner : public PreconditionerTypeBase
  {
   public:
    ProjectionPreconditioner(std::shared_ptr<PreconditionerTypeBase> preconditioner,
        std::shared_ptr<Core::LinAlg::LinearSystemProjector> projector);

    void setup(Core::LinAlg::SparseOperator& matrix, Core::LinAlg::MultiVector<double>& b) override;

    /// linear operator used for preconditioning
    std::shared_ptr<Epetra_Operator> prec_operator() const override { return p_; }

    Teuchos::RCP<const Thyra::LinearOpBase<double>> thyra_operator() const override
    {
      FOUR_C_THROW("One should not end up here.");
    }

   private:
    std::shared_ptr<PreconditionerTypeBase> preconditioner_;

    /// projector object that does the actual work
    std::shared_ptr<Core::LinAlg::LinearSystemProjector> projector_;

    std::shared_ptr<Epetra_Operator> p_;

    Teuchos::RCP<const Thyra::LinearOpBase<double>> p_thyra_;
  };
}  // namespace Core::LinearSolver

FOUR_C_NAMESPACE_CLOSE

#endif

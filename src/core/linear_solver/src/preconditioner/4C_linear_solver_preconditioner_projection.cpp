// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_linear_solver_preconditioner_projection.hpp"

#include "4C_linear_solver_method_projector.hpp"

#include <Thyra_EpetraThyraWrappers.hpp>

#include <utility>

FOUR_C_NAMESPACE_OPEN

//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
Core::LinearSolver::ProjectionPreconditioner::ProjectionPreconditioner(
    std::shared_ptr<Core::LinearSolver::PreconditionerTypeBase> preconditioner,
    std::shared_ptr<Core::LinAlg::LinearSystemProjector> projector)
    : preconditioner_(std::move(preconditioner)), projector_(std::move(projector))
{
}

//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
void Core::LinearSolver::ProjectionPreconditioner::setup(
    Core::LinAlg::SparseOperator& matrix, Core::LinAlg::MultiVector<double>& b)
{
  FOUR_C_ASSERT_ALWAYS(b.num_vectors() == 1,
      "Expecting only one solution vector during projector call! Got {} vectors.", b.num_vectors());
  b.get_vector(0) = projector_->to_reduced(b.get_vector(0));

  // setup wrapped preconditioner
  preconditioner_->setup(matrix, b);

  // Wrap the linear operator of the contained preconditioner. This way the
  // actual preconditioner is called first and the projection is done
  // afterwards.

  p_ = Teuchos::make_rcp<Core::LinearSolver::LinalgPrecondOperator>(
      preconditioner_->prec_operator(), true, projector_);
}


//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
Core::LinearSolver::LinalgPrecondOperator::LinalgPrecondOperator(
    Teuchos::RCP<const Thyra::LinearOpBase<double>> precond, bool project,
    std::shared_ptr<Core::LinAlg::LinearSystemProjector> projector)
    : project_(project), precond_(precond), projector_(projector)
{
  if (project_ && (projector == nullptr))
    FOUR_C_THROW("Kernel projection enabled but got no projector object");
}

//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
Teuchos::RCP<const Thyra::VectorSpaceBase<double>>
Core::LinearSolver::LinalgPrecondOperator::range() const
{
  return precond_->range();
}

//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
Teuchos::RCP<const Thyra::VectorSpaceBase<double>>
Core::LinearSolver::LinalgPrecondOperator::domain() const
{
  return precond_->domain();
}

//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
bool Core::LinearSolver::LinalgPrecondOperator::opSupportedImpl(
    const Thyra::EOpTransp M_trans) const
{
  return (M_trans == Thyra::NOTRANS);
}

//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
void Core::LinearSolver::LinalgPrecondOperator::applyImpl(const Thyra::EOpTransp M_trans,
    const Thyra::MultiVectorBase<double>& x, const Teuchos::Ptr<Thyra::MultiVectorBase<double>>& y,
    const double alpha, const double beta) const
{
  // Apply the inverse preconditioner to get new basis vector for the
  // Krylov space
  precond_->apply(::Thyra::NOTRANS, x, y, alpha, beta);

  // if necessary, project out matrix kernel to maintain well-posedness
  // of problem
  if (project_)
  {
    auto map = Thyra::get_Epetra_Map(range());
    auto Y = Thyra::get_Epetra_MultiVector(Teuchos::rcpFromPtr(y), map);
    Core::LinAlg::View Y_view(*Y);

    FOUR_C_ASSERT_ALWAYS(Y->NumVectors() == 1,
        "Expecting only one solution vector during projector call! Got {} vectors.",
        Y->NumVectors());

    Y_view.underlying().get_vector(0) = projector_->to_full(Y_view.underlying().get_vector(0));
  }
}

FOUR_C_NAMESPACE_CLOSE

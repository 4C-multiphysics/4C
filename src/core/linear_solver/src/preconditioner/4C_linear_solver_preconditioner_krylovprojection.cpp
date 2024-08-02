/*----------------------------------------------------------------------*/
/*! \file

\brief Declaration

 \level 1

*/
/*----------------------------------------------------------------------*/

#include "4C_linear_solver_preconditioner_krylovprojection.hpp"

#include "4C_linalg_projected_precond.hpp"

FOUR_C_NAMESPACE_OPEN

//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
Core::LinearSolver::KrylovProjectionPreconditioner::KrylovProjectionPreconditioner(
    Teuchos::RCP<Core::LinearSolver::PreconditionerTypeBase> preconditioner,
    Teuchos::RCP<Core::LinAlg::KrylovProjector> projector)
    : preconditioner_(preconditioner), projector_(projector)
{
}

//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
void Core::LinearSolver::KrylovProjectionPreconditioner::setup(
    bool create, Epetra_Operator* matrix, Epetra_MultiVector* x, Epetra_MultiVector* b)
{
  projector_->apply_pt(*b);

  // setup wrapped preconditioner
  preconditioner_->setup(create, matrix, x, b);

  // Wrap the linear operator of the contained preconditioner. This way the
  // actual preconditioner is called first and the projection is done
  // afterwards.

  p_ = Teuchos::rcp(
      new Core::LinAlg::LinalgPrecondOperator(preconditioner_->prec_operator(), true, projector_));
}

FOUR_C_NAMESPACE_CLOSE

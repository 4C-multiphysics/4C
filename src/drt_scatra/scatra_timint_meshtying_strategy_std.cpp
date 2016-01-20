/*!----------------------------------------------------------------------
\file scatra_timint_meshtying_strategy_std.cpp

\brief Standard solution strategy for standard scalar transport problems (without meshtying)

<pre>
Maintainer: Rui Fang
            fang@lnm.mw.tum.de
            http://www.lnm.mw.tum.de/
            089 - 289-15251
</pre>

*----------------------------------------------------------------------*/

#include "../drt_lib/drt_discret.H"

#include "../drt_scatra/scatra_timint_implicit.H"

#include "../linalg/linalg_solver.H"

#include "scatra_timint_meshtying_strategy_std.H"

/*----------------------------------------------------------------------*
 | constructor                                               fang 12/14 |
 *----------------------------------------------------------------------*/
SCATRA::MeshtyingStrategyStd::MeshtyingStrategyStd(
    SCATRA::ScaTraTimIntImpl* scatratimint
    ) :
MeshtyingStrategyBase(scatratimint)
{
  return;
} // SCATRA::MeshtyingStrategyStd::MeshtyingStrategyStd


/*----------------------------------------------------------------------*
 | dummy meshtying evaluate for standard scalar transport    fang 12/14 |
 *----------------------------------------------------------------------*/
void SCATRA::MeshtyingStrategyStd::EvaluateMeshtying()
{
  return;
} // SCATRA::MeshtyingStrategyStd::EvaluateMeshtying


/*----------------------------------------------------------------------*
 | initialize system matrix for standard scalar transport    fang 12/14 |
 *----------------------------------------------------------------------*/
Teuchos::RCP<LINALG::SparseOperator> SCATRA::MeshtyingStrategyStd::InitSystemMatrix() const
{
  // initialize standard (stabilized) system matrix (and save its graph)
  return Teuchos::rcp(new LINALG::SparseMatrix(*(scatratimint_->Discretization()->DofRowMap()),27,false,true));
} // SCATRA::MeshtyingStrategyStd::InitSystemMatrix


/*-----------------------------------------------------------------------------*
 | solve linear system of equations for standard scalar transport   fang 12/14 |
 *-----------------------------------------------------------------------------*/
void SCATRA::MeshtyingStrategyStd::Solve(
    const Teuchos::RCP<LINALG::Solver>&            solver,         //! solver
    const Teuchos::RCP<LINALG::SparseOperator>&    systemmatrix,   //! system matrix
    const Teuchos::RCP<Epetra_Vector>&             increment,      //! increment vector
    const Teuchos::RCP<Epetra_Vector>&             residual,       //! residual vector
    const Teuchos::RCP<Epetra_Vector>&             phinp,          //! state vector at time n+1
    const int&                                     iteration,      //! number of current Newton-Raphson iteration
    const Teuchos::RCP<LINALG::KrylovProjector>&   projector       //! Krylov projector
    ) const
{
  solver->Solve(systemmatrix->EpetraOperator(),increment,residual,true,iteration==1,projector);

  return;
} // SCATRA::MeshtyingStrategyStd::Solve

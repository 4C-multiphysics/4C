/*!----------------------------------------------------------------------
\file scatra_timint_meshtying_strategy_fluid.cpp

\brief Fluid-fluid meshtying strategy for standard scalar transport problems

<pre>
Maintainer: Rui Fang
            fang@lnm.mw.tum.de
            http://www.lnm.mw.tum.de/
            089 - 289-15251
</pre>

*----------------------------------------------------------------------*/

#include "../drt_fluid/fluid_meshtying.H"

#include "../drt_lib/drt_globalproblem.H"

#include "../drt_scatra/scatra_timint_implicit.H"

#include "../linalg/linalg_sparseoperator.H"

#include "scatra_timint_meshtying_strategy_fluid.H"

/*----------------------------------------------------------------------*
 | constructor                                               fang 12/14 |
 *----------------------------------------------------------------------*/
SCATRA::MeshtyingStrategyFluid::MeshtyingStrategyFluid(
    SCATRA::ScaTraTimIntImpl* scatratimint
    ) :
MeshtyingStrategyBase(scatratimint),
meshtying_(Teuchos::null),
type_(INPAR::FLUID::no_meshtying)
{
  return;
} // SCATRA::MeshtyingStrategyFluid::MeshtyingStrategyFluid


/*----------------------------------------------------------------------*
 | evaluate fluid-fluid meshtying                            fang 12/14 |
 *----------------------------------------------------------------------*/
void SCATRA::MeshtyingStrategyFluid::EvaluateMeshtying() const
{

  // need to complete system matrix due to subsequent matrix-matrix multiplications
  scatratimint_->SystemMatrixOperator()->Complete();

  // evaluate fluid-fluid meshtying
  meshtying_->PrepareMeshtyingSystem(scatratimint_->SystemMatrixOperator(),scatratimint_->Residual(),scatratimint_->Phinp());

  return;
} // SCATRA::MeshtyingStrategyFluid::EvaluateMeshtying


/*----------------------------------------------------------------------*
 | include Dirichlet conditions into condensation            fang 12/14 |
 *----------------------------------------------------------------------*/
void SCATRA::MeshtyingStrategyFluid::IncludeDirichletInCondensation() const
{
  meshtying_->IncludeDirichletInCondensation(scatratimint_->Phinp(),scatratimint_->Phin());

  return;
} // SCATRA::MeshtyingStrategyFluid::IncludeDirichletInCondensation()


/*----------------------------------------------------------------------*
 | perform setup of fluid-fluid meshtying                    fang 12/14 |
 *----------------------------------------------------------------------*/
void SCATRA::MeshtyingStrategyFluid::InitMeshtying()
{
  // Important: Meshtying for scalar transport is not well tested!
  // get meshtying type
  type_ = DRT::INPUT::IntegralValue<INPAR::FLUID::MeshTying>(*(scatratimint_->ScatraParameterList()),"MESHTYING");

  // safety checks
  if(type_ == INPAR::FLUID::condensed_bmat)
    dserror("The 2x2 block solver algorithm for a block matrix system has not been activated yet. Just do it!");

  // setup meshtying
  meshtying_ = Teuchos::rcp(new FLD::Meshtying(scatratimint_->Discretization(),*(scatratimint_->Solver()),type_,DRT::Problem::Instance()->NDim()));

  return;
} // SCATRA::MeshtyingStrategyFluid::InitMeshtying


/*----------------------------------------------------------------------*
 | initialize system matrix for fluid-fluid meshtying        fang 12/14 |
 *----------------------------------------------------------------------*/
Teuchos::RCP<LINALG::SparseOperator> SCATRA::MeshtyingStrategyFluid::InitSystemMatrix() const
{
  // safety check
  if(scatratimint_->NumScal() < 1)
    dserror("Number of transported scalars not correctly set!");

  // define coupling and initialize system matrix
  std::vector<int> coupleddof(scatratimint_->NumScal(),1);

  return meshtying_->Setup(coupleddof);
} // SCATRA::MeshtyingStrategyFluid::InitSystemMatrix


/*-------------------------------------------------------------------------*
 | solve linear system of equations for fluid-fluid meshtying   fang 12/14 |
 *-------------------------------------------------------------------------*/
void SCATRA::MeshtyingStrategyFluid::Solve(
    const Teuchos::RCP<LINALG::Solver>&            solver,         //! solver
    const Teuchos::RCP<LINALG::SparseOperator>&    systemmatrix,   //! system matrix
    const Teuchos::RCP<Epetra_Vector>&             increment,      //! increment vector
    const Teuchos::RCP<Epetra_Vector>&             residual,       //! residual vector
    const Teuchos::RCP<Epetra_Vector>&             phinp,          //! state vector at time n+1
    const int&                                     iteration,      //! number of current Newton-Raphson iteration
    const Teuchos::RCP<LINALG::KrylovProjector>&   projector       //! Krylov projector
    ) const
{
  meshtying_->SolveMeshtying(*solver,systemmatrix,increment,residual,phinp,iteration,projector);

  return;
} // SCATRA::MeshtyingStrategyFluid::Solve

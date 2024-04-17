/*----------------------------------------------------------------------*/
/*! \file
\brief standard case without mesh tying

\level 3

*----------------------------------------------------------------------*/

#include "baci_porofluidmultiphase_meshtying_strategy_std.hpp"

#include "baci_linear_solver_method_linalg.hpp"
#include "baci_porofluidmultiphase_utils.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | constructor                                (public) kremheller 04/18 |
 *----------------------------------------------------------------------*/
POROFLUIDMULTIPHASE::MeshtyingStrategyStd::MeshtyingStrategyStd(
    POROFLUIDMULTIPHASE::TimIntImpl* porofluidmultitimint, const Teuchos::ParameterList& probparams,
    const Teuchos::ParameterList& poroparams)
    : MeshtyingStrategyBase(porofluidmultitimint, probparams, poroparams)
{
  return;
}



/*----------------------------------------------------------------------*
 | prepare time loop                                   kremheller 04/18 |
 *----------------------------------------------------------------------*/
void POROFLUIDMULTIPHASE::MeshtyingStrategyStd::PrepareTimeLoop() { return; }

/*----------------------------------------------------------------------*
 | setup the variables to do a new time step  (public) kremheller 04/18 |
 *----------------------------------------------------------------------*/
void POROFLUIDMULTIPHASE::MeshtyingStrategyStd::PrepareTimeStep() { return; }

/*----------------------------------------------------------------------*
 | current solution becomes most recent solution of next timestep       |
 |                                                     kremheller 04/18 |
 *----------------------------------------------------------------------*/
void POROFLUIDMULTIPHASE::MeshtyingStrategyStd::Update() { return; }

/*--------------------------------------------------------------------------*
 | initialize the linear solver                            kremheller 07/20 |
 *--------------------------------------------------------------------------*/
void POROFLUIDMULTIPHASE::MeshtyingStrategyStd::InitializeLinearSolver(
    Teuchos::RCP<CORE::LINALG::Solver> solver)
{
  porofluidmultitimint_->Discretization()->ComputeNullSpaceIfNecessary(solver->Params());
}

/*--------------------------------------------------------------------------*
 | solve linear system of equations                        kremheller 04/18 |
 *--------------------------------------------------------------------------*/
void POROFLUIDMULTIPHASE::MeshtyingStrategyStd::LinearSolve(
    Teuchos::RCP<CORE::LINALG::Solver> solver, Teuchos::RCP<CORE::LINALG::SparseOperator> sysmat,
    Teuchos::RCP<Epetra_Vector> increment, Teuchos::RCP<Epetra_Vector> residual,
    CORE::LINALG::SolverParams& solver_params)
{
  solver_params.refactor = true;
  solver_params.reset = true;
  solver->Solve(sysmat->EpetraOperator(), increment, residual, solver_params);

  return;
}

/*----------------------------------------------------------------------*
 | Calculate problem specific norm                     kremheller 03/18 |
 *----------------------------------------------------------------------*/
void POROFLUIDMULTIPHASE::MeshtyingStrategyStd::CalculateNorms(std::vector<double>& preresnorm,
    std::vector<double>& incprenorm, std::vector<double>& prenorm,
    const Teuchos::RCP<const Epetra_Vector> increment)
{
  preresnorm.resize(1);
  incprenorm.resize(1);
  prenorm.resize(1);

  preresnorm[0] = UTILS::CalculateVectorNorm(vectornormfres_, porofluidmultitimint_->RHS());
  incprenorm[0] = UTILS::CalculateVectorNorm(vectornorminc_, increment);
  prenorm[0] = UTILS::CalculateVectorNorm(vectornorminc_, porofluidmultitimint_->Phinp());

  return;
}

/*----------------------------------------------------------------------*
 | create result test for this field                   kremheller 04/18 |
 *----------------------------------------------------------------------*/
void POROFLUIDMULTIPHASE::MeshtyingStrategyStd::CreateFieldTest() { return; }

/*----------------------------------------------------------------------*
 |  read restart data                                  kremheller 04/18 |
 -----------------------------------------------------------------------*/
void POROFLUIDMULTIPHASE::MeshtyingStrategyStd::ReadRestart(const int step) { return; }

/*----------------------------------------------------------------------*
 | output of solution vector to BINIO                  kremheller 04/18 |
 *----------------------------------------------------------------------*/
void POROFLUIDMULTIPHASE::MeshtyingStrategyStd::Output() { return; }

/*----------------------------------------------------------------------*
 | evaluate matrix and rhs                             kremheller 04/18 |
 *----------------------------------------------------------------------*/
void POROFLUIDMULTIPHASE::MeshtyingStrategyStd::Evaluate() { return; }

/*----------------------------------------------------------------------*
 | extract and update                                  kremheller 04/18 |
 *----------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Vector> POROFLUIDMULTIPHASE::MeshtyingStrategyStd::ExtractAndUpdateIter(
    const Teuchos::RCP<const Epetra_Vector> inc)
{
  return inc;
}

/*----------------------------------------------------------------------*
 | extract and update                                  kremheller 04/18 |
 *----------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Vector> POROFLUIDMULTIPHASE::MeshtyingStrategyStd::CombinedIncrement(
    const Teuchos::RCP<const Epetra_Vector> inc) const
{
  return inc;
}

/*----------------------------------------------------------------------*
 | check initial fields                                kremheller 06/18 |
 *----------------------------------------------------------------------*/
void POROFLUIDMULTIPHASE::MeshtyingStrategyStd::CheckInitialFields(
    Teuchos::RCP<const Epetra_Vector> vec_cont) const
{
  return;
}

/*-------------------------------------------------------------------------*
 | set element pairs that are close                       kremheller 03/19 |
 *------------------------------------------------------------------------ */
void POROFLUIDMULTIPHASE::MeshtyingStrategyStd::SetNearbyElePairs(
    const std::map<int, std::set<int>>* nearbyelepairs)
{
  return;
}

/*-------------------------------------------------------------------------*
 | setup the strategy                                     kremheller 03/19 |
 *------------------------------------------------------------------------ */
void POROFLUIDMULTIPHASE::MeshtyingStrategyStd::Setup() { return; }

/*----------------------------------------------------------------------*
 | apply mesh movement                                 kremheller 06/18 |
 *----------------------------------------------------------------------*/
void POROFLUIDMULTIPHASE::MeshtyingStrategyStd::ApplyMeshMovement() const { return; }

FOUR_C_NAMESPACE_CLOSE

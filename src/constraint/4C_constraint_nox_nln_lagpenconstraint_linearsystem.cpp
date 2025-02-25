// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_constraint_nox_nln_lagpenconstraint_linearsystem.hpp"

#include "4C_linalg_sparseoperator.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_solver_nonlin_nox_interface_jacobian.hpp"
#include "4C_solver_nonlin_nox_interface_required.hpp"

#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
NOX::Nln::LAGPENCONSTRAINT::LinearSystem::LinearSystem(Teuchos::ParameterList& printParams,
    Teuchos::ParameterList& linearSolverParams, const SolverMap& solvers,
    const Teuchos::RCP<::NOX::Epetra::Interface::Required>& iReq,
    const Teuchos::RCP<::NOX::Epetra::Interface::Jacobian>& iJac,
    const NOX::Nln::CONSTRAINT::ReqInterfaceMap& iConstr,
    const Teuchos::RCP<Core::LinAlg::SparseOperator>& J,
    const Teuchos::RCP<::NOX::Epetra::Interface::Preconditioner>& iPrec,
    const NOX::Nln::CONSTRAINT::PrecInterfaceMap& iConstrPrec,
    const Teuchos::RCP<Core::LinAlg::SparseOperator>& M, const ::NOX::Epetra::Vector& cloneVector,
    const Teuchos::RCP<::NOX::Epetra::Scaling> scalingObject)
    : NOX::Nln::LinearSystem(printParams, linearSolverParams, solvers, iReq, iJac, J, iPrec, M,
          cloneVector, scalingObject),
      i_constr_(iConstr),
      i_constr_prec_(iConstrPrec)
{
  // empty
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
NOX::Nln::LAGPENCONSTRAINT::LinearSystem::LinearSystem(Teuchos::ParameterList& printParams,
    Teuchos::ParameterList& linearSolverParams, const SolverMap& solvers,
    const Teuchos::RCP<::NOX::Epetra::Interface::Required>& iReq,
    const Teuchos::RCP<::NOX::Epetra::Interface::Jacobian>& iJac,
    const NOX::Nln::CONSTRAINT::ReqInterfaceMap& iConstr,
    const Teuchos::RCP<Core::LinAlg::SparseOperator>& J,
    const Teuchos::RCP<::NOX::Epetra::Interface::Preconditioner>& iPrec,
    const NOX::Nln::CONSTRAINT::PrecInterfaceMap& iConstrPrec,
    const Teuchos::RCP<Core::LinAlg::SparseOperator>& M, const ::NOX::Epetra::Vector& cloneVector)
    : NOX::Nln::LinearSystem(
          printParams, linearSolverParams, solvers, iReq, iJac, J, iPrec, M, cloneVector),
      i_constr_(iConstr),
      i_constr_prec_(iConstrPrec)
{
  // empty
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::LinAlg::SolverParams NOX::Nln::LAGPENCONSTRAINT::LinearSystem::set_solver_options(
    Teuchos::ParameterList& p, Teuchos::RCP<Core::LinAlg::Solver>& solverPtr,
    const NOX::Nln::SolutionType& solverType)
{
  Core::LinAlg::SolverParams solver_params;

  bool isAdaptiveControl = p.get<bool>("Adaptive Control");
  double adaptiveControlObjective = p.get<double>("Adaptive Control Objective");

  if (isAdaptiveControl)
  {
    // dynamic cast of the required/rhs interface
    Teuchos::RCP<NOX::Nln::Interface::Required> iNlnReq =
        Teuchos::rcp_dynamic_cast<NOX::Nln::Interface::Required>(reqInterfacePtr_);
    if (iNlnReq.is_null()) throw_error("setSolverOptions", "required interface cast failed");

    double worst = iNlnReq->calc_ref_norm_force();
    // This value has to be specified in the PrePostOperator object of
    // the non-linear solver (i.e. runPreSolve())
    double wanted = p.get<double>("Wanted Tolerance");
    solver_params.nonlin_tolerance = wanted;
    solver_params.nonlin_residual = worst;
    solver_params.lin_tol_better = adaptiveControlObjective;
  }
  return solver_params;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
NOX::Nln::SolutionType NOX::Nln::LAGPENCONSTRAINT::LinearSystem::get_active_lin_solver(
    const std::map<NOX::Nln::SolutionType, Teuchos::RCP<Core::LinAlg::Solver>>& solvers,
    Teuchos::RCP<Core::LinAlg::Solver>& currSolver)
{
  // check input
  if (solvers.size() > 2)
    throw_error("GetCurrentLinSolver",
        "There have to be exactly two Core::LinAlg::Solvers (structure + contact)!");
  // ---------------------------------------------------------------------
  // Solving a saddle point system
  // Lagrange multiplier constraints -> SaddlePoint
  // ---------------------------------------------------------------------
  NOX::Nln::CONSTRAINT::PrecInterfaceMap::const_iterator cit;
  bool issaddlepoint = false;
  for (cit = i_constr_prec_.begin(); cit != i_constr_prec_.end(); ++cit)
  {
    if (cit->second->is_saddle_point_system())
    {
      issaddlepoint = true;
      break;
    }
  }

  if (issaddlepoint)
  {
    currSolver = solvers.at(NOX::Nln::sol_lag_pen_constraint);
    return NOX::Nln::sol_lag_pen_constraint;
  }
  // ----------------------------------------------------------------------
  // Solving a standard displacement-based system
  // Penalty-enforced constraints
  // ----------------------------------------------------------------------
  if (!issaddlepoint)
  {
    currSolver = solvers.at(NOX::Nln::sol_structure);
    return NOX::Nln::sol_structure;
  }

  // default return
  currSolver = solvers.at(NOX::Nln::sol_lag_pen_constraint);
  return NOX::Nln::sol_lag_pen_constraint;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::Nln::LAGPENCONSTRAINT::LinearSystem::throw_error(
    const std::string& functionName, const std::string& errorMsg) const
{
  if (utils_.isPrintType(::NOX::Utils::Error))
  {
    utils_.out() << "NOX::LAGPENCONSTRAINT::LinearSystem::" << functionName << " - " << errorMsg
                 << std::endl;
  }
  throw "NOX Error";
}

FOUR_C_NAMESPACE_CLOSE

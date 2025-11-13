// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_cardiovascular0d_nox_nln_linearsystem.hpp"

#include "4C_linalg_sparseoperator.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_solver_nonlin_nox_interface_jacobian.hpp"
#include "4C_solver_nonlin_nox_interface_required.hpp"
#include "4C_solver_nonlin_nox_vector.hpp"

#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
NOX::Nln::Cardiovascular0D::LinearSystem::LinearSystem(Teuchos::ParameterList& printParams,
    Teuchos::ParameterList& linearSolverParams,
    const std::map<NOX::Nln::SolutionType, Teuchos::RCP<Core::LinAlg::Solver>>& solvers,
    const std::shared_ptr<NOX::Nln::Interface::RequiredBase> iReq,
    const Teuchos::RCP<::NOX::Epetra::Interface::Jacobian>& iJac,
    const Teuchos::RCP<Core::LinAlg::SparseOperator>& J,
    const Teuchos::RCP<Core::LinAlg::SparseOperator>& M, const NOX::Nln::Vector& cloneVector,
    const std::shared_ptr<NOX::Nln::Scaling> scalingObject)
    : NOX::Nln::LinearSystem(
          printParams, linearSolverParams, solvers, iReq, iJac, J, M, cloneVector, scalingObject)
{
  // empty constructor
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
NOX::Nln::Cardiovascular0D::LinearSystem::LinearSystem(Teuchos::ParameterList& printParams,
    Teuchos::ParameterList& linearSolverParams,
    const std::map<NOX::Nln::SolutionType, Teuchos::RCP<Core::LinAlg::Solver>>& solvers,
    const std::shared_ptr<NOX::Nln::Interface::RequiredBase> iReq,
    const Teuchos::RCP<::NOX::Epetra::Interface::Jacobian>& iJac,
    const Teuchos::RCP<Core::LinAlg::SparseOperator>& J,
    const Teuchos::RCP<Core::LinAlg::SparseOperator>& M, const NOX::Nln::Vector& cloneVector)
    : NOX::Nln::LinearSystem(
          printParams, linearSolverParams, solvers, iReq, iJac, J, M, cloneVector)
{
  // empty constructor
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
NOX::Nln::Cardiovascular0D::LinearSystem::LinearSystem(Teuchos::ParameterList& printParams,
    Teuchos::ParameterList& linearSolverParams,
    const std::map<NOX::Nln::SolutionType, Teuchos::RCP<Core::LinAlg::Solver>>& solvers,
    const std::shared_ptr<NOX::Nln::Interface::RequiredBase> iReq,
    const Teuchos::RCP<::NOX::Epetra::Interface::Jacobian>& iJac,
    const Teuchos::RCP<Core::LinAlg::SparseOperator>& J, const NOX::Nln::Vector& cloneVector,
    const std::shared_ptr<NOX::Nln::Scaling> scalingObject)
    : NOX::Nln::LinearSystem(
          printParams, linearSolverParams, solvers, iReq, iJac, J, cloneVector, scalingObject)
{
  // empty constructor
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
NOX::Nln::Cardiovascular0D::LinearSystem::LinearSystem(Teuchos::ParameterList& printParams,
    Teuchos::ParameterList& linearSolverParams,
    const std::map<NOX::Nln::SolutionType, Teuchos::RCP<Core::LinAlg::Solver>>& solvers,
    const std::shared_ptr<NOX::Nln::Interface::RequiredBase> iReq,
    const Teuchos::RCP<::NOX::Epetra::Interface::Jacobian>& iJac,
    const Teuchos::RCP<Core::LinAlg::SparseOperator>& J, const NOX::Nln::Vector& cloneVector)
    : NOX::Nln::LinearSystem(printParams, linearSolverParams, solvers, iReq, iJac, J, cloneVector)
{
  // empty constructor
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::LinAlg::SolverParams NOX::Nln::Cardiovascular0D::LinearSystem::set_solver_options(
    Teuchos::ParameterList& p, Teuchos::RCP<Core::LinAlg::Solver>& solverPtr,
    const NOX::Nln::SolutionType& solverType)
{
  Core::LinAlg::SolverParams solver_params;

  bool isAdaptiveControl = p.get<bool>("Adaptive Control");
  double adaptiveControlObjective = p.get<double>("Adaptive Control Objective");

  if (isAdaptiveControl)
  {
    // dynamic cast of the required/rhs interface
    const auto iNlnReq = std::dynamic_pointer_cast<NOX::Nln::Interface::Required>(reqInterfacePtr_);
    FOUR_C_ASSERT(iNlnReq,
        "NOX::Nln::Cardiovascular0D::LinearSystem::set_solver_options(): required interface cast "
        "failed.");

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
NOX::Nln::SolutionType NOX::Nln::Cardiovascular0D::LinearSystem::get_active_lin_solver(
    const std::map<NOX::Nln::SolutionType, Teuchos::RCP<Core::LinAlg::Solver>>& solvers,
    Teuchos::RCP<Core::LinAlg::Solver>& currSolver)
{
  //  // check input
  //  if (solvers.size()>1)
  //    throw_error("GetCurrentLinSolver","There has to be exactly one "
  //        "Core::LinAlg::Solver (structure)!");

  currSolver = solvers.at(NOX::Nln::sol_cardiovascular0d);
  return NOX::Nln::sol_cardiovascular0d;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::Nln::Cardiovascular0D::LinearSystem::throw_error(
    const std::string& functionName, const std::string& errorMsg) const
{
  if (utils_.isPrintType(::NOX::Utils::Error))
  {
    utils_.out() << "NOX::Nln::Cardiovascular0D::LinearSystem::" << functionName << " - "
                 << errorMsg << std::endl;
  }
  throw "NOX Error";
}

FOUR_C_NAMESPACE_CLOSE

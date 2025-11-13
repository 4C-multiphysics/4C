// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_structure_new_nox_nln_str_linearsystem.hpp"

#include "4C_linalg_sparseoperator.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_solver_nonlin_nox_interface_jacobian.hpp"
#include "4C_solver_nonlin_nox_interface_required.hpp"

#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
NOX::Nln::Solid::LinearSystem::LinearSystem(Teuchos::ParameterList& printParams,
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
NOX::Nln::Solid::LinearSystem::LinearSystem(Teuchos::ParameterList& printParams,
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
NOX::Nln::Solid::LinearSystem::LinearSystem(Teuchos::ParameterList& printParams,
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
NOX::Nln::Solid::LinearSystem::LinearSystem(Teuchos::ParameterList& printParams,
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
Core::LinAlg::SolverParams NOX::Nln::Solid::LinearSystem::set_solver_options(
    Teuchos::ParameterList& p, Teuchos::RCP<Core::LinAlg::Solver>& solverPtr,
    const NOX::Nln::SolutionType& solverType)
{
  Core::LinAlg::SolverParams solver_params;
  bool isAdaptiveControl = p.get<bool>("Adaptive Control");

  if (isAdaptiveControl)
  {
    // dynamic cast of the required/rhs interface
    const auto iNlnReq = std::dynamic_pointer_cast<NOX::Nln::Interface::Required>(reqInterfacePtr_);
    FOUR_C_ASSERT(iNlnReq,
        "NOX::Nln::Solid::LinearSystem::set_solver_options(): required interface cast "
        "failed");

    solver_params.nonlin_tolerance = p.get<double>("Wanted Tolerance");
    solver_params.nonlin_residual = iNlnReq->calc_ref_norm_force();
    solver_params.lin_tol_better = p.get<double>("Adaptive Control Objective");
  }

  return solver_params;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
NOX::Nln::SolutionType NOX::Nln::Solid::LinearSystem::get_active_lin_solver(
    const std::map<NOX::Nln::SolutionType, Teuchos::RCP<Core::LinAlg::Solver>>& solvers,
    Teuchos::RCP<Core::LinAlg::Solver>& currSolver)
{
  // check input
  if (solvers.size() > 1)
    FOUR_C_THROW("There has to be exactly one Core::LinAlg::Solver (structure)!");

  currSolver = solvers.at(NOX::Nln::sol_structure);
  return NOX::Nln::sol_structure;
}

FOUR_C_NAMESPACE_CLOSE

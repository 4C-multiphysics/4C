// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_solver_nonlin_nox_linearsystem_generic.hpp"

#include "4C_linear_solver_method_linalg.hpp"
#include "4C_solver_nonlin_nox_interface_jacobian.hpp"
#include "4C_solver_nonlin_nox_interface_required.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN

NOX::Nln::Generic::LinearSystem::LinearSystem(Teuchos::ParameterList& printParams,
    Teuchos::ParameterList& linearSolverParams,
    const std::map<NOX::Nln::SolutionType, Teuchos::RCP<Core::LinAlg::Solver>>& solvers,
    const std::shared_ptr<NOX::Nln::Interface::RequiredBase> iReq,
    const std::shared_ptr<NOX::Nln::Interface::JacobianBase> iJac,
    const std::shared_ptr<Core::LinAlg::SparseOperator>& J, const NOX::Nln::Vector& cloneVector)
    : NOX::Nln::LinearSystem(printParams, linearSolverParams, solvers, iReq, iJac, J, cloneVector)
{
}

NOX::Nln::Generic::LinearSystem::LinearSystem(Teuchos::ParameterList& printParams,
    Teuchos::ParameterList& linearSolverParams,
    const std::map<NOX::Nln::SolutionType, Teuchos::RCP<Core::LinAlg::Solver>>& solvers,
    const std::shared_ptr<NOX::Nln::Interface::RequiredBase> iReq,
    const std::shared_ptr<NOX::Nln::Interface::JacobianBase> iJac,
    const std::shared_ptr<Core::LinAlg::SparseOperator>& J,
    const std::shared_ptr<Core::LinAlg::SparseOperator>& M, const NOX::Nln::Vector& cloneVector,
    const std::shared_ptr<NOX::Nln::Scaling> scalingObject)
    : NOX::Nln::LinearSystem(
          printParams, linearSolverParams, solvers, iReq, iJac, J, M, cloneVector, scalingObject)
{
}

Core::LinAlg::SolverParams NOX::Nln::Generic::LinearSystem::set_solver_options(
    Teuchos::ParameterList& p, Teuchos::RCP<Core::LinAlg::Solver>& solverPtr,
    const NOX::Nln::SolutionType& solverType)
{
  (void)p;
  (void)solverPtr;
  (void)solverType;
  // Generic linear system currently uses solver defaults without additional NOX-side options.

  Core::LinAlg::SolverParams solver_params;
  return solver_params;
}

NOX::Nln::SolutionType NOX::Nln::Generic::LinearSystem::get_active_lin_solver(
    const std::map<NOX::Nln::SolutionType, Teuchos::RCP<Core::LinAlg::Solver>>& solvers,
    Teuchos::RCP<Core::LinAlg::Solver>& currSolver)
{
  if (solvers.size() != 1)
    FOUR_C_THROW("There has to be exactly one Core::LinAlg::Solver (generic).");

  auto it = solvers.begin();
  currSolver = it->second;
  return it->first;
}

FOUR_C_NAMESPACE_CLOSE

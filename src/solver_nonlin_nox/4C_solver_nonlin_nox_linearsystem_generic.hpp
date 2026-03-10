// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SOLVER_NONLIN_NOX_LINEARSYSTEM_GENERIC_HPP
#define FOUR_C_SOLVER_NONLIN_NOX_LINEARSYSTEM_GENERIC_HPP

#include "4C_config.hpp"

#include "4C_solver_nonlin_nox_linearsystem.hpp"

FOUR_C_NAMESPACE_OPEN

namespace NOX
{
  namespace Nln
  {
    namespace Generic
    {
      class LinearSystem : public NOX::Nln::LinearSystem
      {
       public:
        //! Standard constructor with full functionality.
        LinearSystem(Teuchos::ParameterList& printParams,
            Teuchos::ParameterList& linearSolverParams,
            const std::map<NOX::Nln::SolutionType, Teuchos::RCP<Core::LinAlg::Solver>>& solvers,
            const std::shared_ptr<NOX::Nln::Interface::RequiredBase> iReq,
            const std::shared_ptr<NOX::Nln::Interface::JacobianBase> iJac,
            const std::shared_ptr<Core::LinAlg::SparseOperator>& J,
            const std::shared_ptr<Core::LinAlg::SparseOperator>& M,
            const NOX::Nln::Vector& cloneVector,
            const std::shared_ptr<NOX::Nln::Scaling> scalingObject);

        //! Constructor without scaling object
        LinearSystem(Teuchos::ParameterList& printParams,
            Teuchos::ParameterList& linearSolverParams,
            const std::map<NOX::Nln::SolutionType, Teuchos::RCP<Core::LinAlg::Solver>>& solvers,
            const std::shared_ptr<NOX::Nln::Interface::RequiredBase> iReq,
            const std::shared_ptr<NOX::Nln::Interface::JacobianBase> iJac,
            const std::shared_ptr<Core::LinAlg::SparseOperator>& J,
            const NOX::Nln::Vector& cloneVector);

        //! sets the options of the underlying solver
        Core::LinAlg::SolverParams set_solver_options(Teuchos::ParameterList& p,
            Teuchos::RCP<Core::LinAlg::Solver>& solverPtr,
            const NOX::Nln::SolutionType& solverType) override;

        //! Returns a pointer to the linear solver, which has to be used
        NOX::Nln::SolutionType get_active_lin_solver(
            const std::map<NOX::Nln::SolutionType, Teuchos::RCP<Core::LinAlg::Solver>>& solvers,
            Teuchos::RCP<Core::LinAlg::Solver>& currSolver) override;
      };
    }  // namespace Generic
  }  // namespace Nln
}  // namespace NOX

FOUR_C_NAMESPACE_CLOSE

#endif

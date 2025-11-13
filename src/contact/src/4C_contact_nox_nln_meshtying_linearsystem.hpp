// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONTACT_NOX_NLN_MESHTYING_LINEARSYSTEM_HPP
#define FOUR_C_CONTACT_NOX_NLN_MESHTYING_LINEARSYSTEM_HPP

#include "4C_config.hpp"

#include "4C_solver_nonlin_nox_constraint_interface_preconditioner.hpp"
#include "4C_solver_nonlin_nox_constraint_interface_required.hpp"
#include "4C_solver_nonlin_nox_linearsystem.hpp"

FOUR_C_NAMESPACE_OPEN

// forward declaration
namespace Mortar
{
  class StrategyBase;
}

namespace NOX
{
  namespace Nln
  {
    namespace MeshTying
    {
      class LinearSystem : public NOX::Nln::LinearSystem
      {
       public:
        //! Standard constructor with full functionality.
        LinearSystem(Teuchos::ParameterList& printParams,
            Teuchos::ParameterList& linearSolverParams, const SolverMap& solvers,
            const std::shared_ptr<NOX::Nln::Interface::RequiredBase> iReq,
            const Teuchos::RCP<::NOX::Epetra::Interface::Jacobian>& iJac,
            const NOX::Nln::CONSTRAINT::ReqInterfaceMap& iConstr,
            const Teuchos::RCP<Core::LinAlg::SparseOperator>& J,
            const NOX::Nln::CONSTRAINT::PrecInterfaceMap& iConstrPrec,
            const Teuchos::RCP<Core::LinAlg::SparseOperator>& M,
            const NOX::Nln::Vector& cloneVector,
            const std::shared_ptr<NOX::Nln::Scaling> scalingObject);

        //! Constructor without scaling object
        LinearSystem(Teuchos::ParameterList& printParams,
            Teuchos::ParameterList& linearSolverParams, const SolverMap& solvers,
            const std::shared_ptr<NOX::Nln::Interface::RequiredBase> iReq,
            const Teuchos::RCP<::NOX::Epetra::Interface::Jacobian>& iJac,
            const NOX::Nln::CONSTRAINT::ReqInterfaceMap& iConstr,
            const Teuchos::RCP<Core::LinAlg::SparseOperator>& J,
            const NOX::Nln::CONSTRAINT::PrecInterfaceMap& iConstrPrec,
            const Teuchos::RCP<Core::LinAlg::SparseOperator>& M,
            const NOX::Nln::Vector& cloneVector);

        //! Sets the options of the underlying solver
        Core::LinAlg::SolverParams set_solver_options(Teuchos::ParameterList& p,
            Teuchos::RCP<Core::LinAlg::Solver>& solverPtr,
            const NOX::Nln::SolutionType& solverType) override;

        //! Returns a pointer to linear solver, which has to be used
        NOX::Nln::SolutionType get_active_lin_solver(
            const std::map<NOX::Nln::SolutionType, Teuchos::RCP<Core::LinAlg::Solver>>& solvers,
            Teuchos::RCP<Core::LinAlg::Solver>& currSolver) override;

       private:
        //! throws an error message
        void throw_error(const std::string& functionName, const std::string& errorMsg) const;

       private:
        //! map of NOX::Nln::CONSTRAINT::Interface::Required objects
        NOX::Nln::CONSTRAINT::ReqInterfaceMap i_constr_;

        //! map of NOX::Nln::CONSTRAINT::Interface::Preconditioner objects
        NOX::Nln::CONSTRAINT::PrecInterfaceMap i_constr_prec_;

      };  // class LinearSystem
    }  // namespace MeshTying
  }  // namespace Nln
}  // namespace NOX


FOUR_C_NAMESPACE_CLOSE

#endif

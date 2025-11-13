// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_contact_nox_nln_meshtying_linearsystem.hpp"  // base class

#include "4C_contact_abstract_strategy.hpp"
#include "4C_contact_input.hpp"
#include "4C_linalg_blocksparsematrix.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_mortar_strategy_base.hpp"
#include "4C_solver_nonlin_nox_aux.hpp"
#include "4C_solver_nonlin_nox_interface_jacobian.hpp"
#include "4C_solver_nonlin_nox_interface_required.hpp"
#include "4C_solver_nonlin_nox_vector.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
NOX::Nln::MeshTying::LinearSystem::LinearSystem(Teuchos::ParameterList& printParams,
    Teuchos::ParameterList& linearSolverParams, const SolverMap& solvers,
    const std::shared_ptr<NOX::Nln::Interface::RequiredBase> iReq,
    const Teuchos::RCP<::NOX::Epetra::Interface::Jacobian>& iJac,
    const NOX::Nln::CONSTRAINT::ReqInterfaceMap& iConstr,
    const Teuchos::RCP<Core::LinAlg::SparseOperator>& J,
    const NOX::Nln::CONSTRAINT::PrecInterfaceMap& iConstrPrec,
    const Teuchos::RCP<Core::LinAlg::SparseOperator>& M, const NOX::Nln::Vector& cloneVector,
    const std::shared_ptr<NOX::Nln::Scaling> scalingObject)
    : NOX::Nln::LinearSystem(
          printParams, linearSolverParams, solvers, iReq, iJac, J, M, cloneVector, scalingObject),
      i_constr_(iConstr),
      i_constr_prec_(iConstrPrec)
{
  // empty
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
NOX::Nln::MeshTying::LinearSystem::LinearSystem(Teuchos::ParameterList& printParams,
    Teuchos::ParameterList& linearSolverParams, const SolverMap& solvers,
    const std::shared_ptr<NOX::Nln::Interface::RequiredBase> iReq,
    const Teuchos::RCP<::NOX::Epetra::Interface::Jacobian>& iJac,
    const NOX::Nln::CONSTRAINT::ReqInterfaceMap& iConstr,
    const Teuchos::RCP<Core::LinAlg::SparseOperator>& J,
    const NOX::Nln::CONSTRAINT::PrecInterfaceMap& iConstrPrec,
    const Teuchos::RCP<Core::LinAlg::SparseOperator>& M, const NOX::Nln::Vector& cloneVector)
    : NOX::Nln::LinearSystem(
          printParams, linearSolverParams, solvers, iReq, iJac, J, M, cloneVector),
      i_constr_(iConstr),
      i_constr_prec_(iConstrPrec)
{
  // empty
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::LinAlg::SolverParams NOX::Nln::MeshTying::LinearSystem::set_solver_options(
    Teuchos::ParameterList& p, Teuchos::RCP<Core::LinAlg::Solver>& solverPtr,
    const NOX::Nln::SolutionType& solverType)
{
  Core::LinAlg::SolverParams solver_params;

  bool isAdaptiveControl = p.get<bool>("Adaptive Control");
  double adaptiveControlObjective = p.get<double>("Adaptive Control Objective");
  // This value is specified in the underlying time integrator
  // (i.e. RunPreNoxNlnSolve())
  int step = p.get<int>("Current Time Step");
  // This value is specified in the PrePostOperator object of
  // the non-linear solver (i.e. runPreIterate())
  int nlnIter = p.get<int>("Number of Nonlinear Iterations");

  if (isAdaptiveControl)
  {
    // dynamic cast of the required/rhs interface
    const auto iNlnReq = std::dynamic_pointer_cast<NOX::Nln::Interface::Required>(reqInterfacePtr_);
    FOUR_C_ASSERT(iNlnReq,
        "NOX::Nln::MeshTying::LinearSystem::set_solver_options(): required interface cast "
        "failed");

    double worst = iNlnReq->calc_ref_norm_force();
    // This value has to be specified in the PrePostOperator object of
    // the non-linear solver (i.e. runPreSolve())
    double wanted = p.get<double>("Wanted Tolerance");
    solver_params.nonlin_tolerance = wanted;
    solver_params.nonlin_residual = worst;
    solver_params.lin_tol_better = adaptiveControlObjective;
  }

  // nothing more to do for a pure structural solver
  if (solverType == NOX::Nln::sol_structure) return solver_params;

  // update information about active slave dofs
  // ---------------------------------------------------------------------
  // feed solver/preconditioner with additional information about the
  // contact/meshtying problem
  // ---------------------------------------------------------------------
  {
    // TODO: maps for merged meshtying and contact problem !!!
    // feed Belos based solvers with contact information
    if (solverPtr->params().isSublist("Belos Parameters"))
    {
      if (i_constr_prec_.size() > 1)
        FOUR_C_THROW(
            "Currently only one constraint preconditioner interface can be handled! \n "
            "Needs to be extended!");

      Teuchos::ParameterList& mueluParams = solverPtr->params().sublist("Belos Parameters");

      // vector entries:
      // (0) masterDofMap
      // (1) slaveDofMap
      // (2) innerDofMap
      // (3) activeDofMap
      std::vector<Teuchos::RCP<Core::LinAlg::Map>> prec_maps(4, Teuchos::null);
      i_constr_prec_.begin()->second->fill_maps_for_preconditioner(prec_maps);
      mueluParams.set<Teuchos::RCP<Epetra_Map>>(
          "contact masterDofMap", Teuchos::rcpFromRef((prec_maps[0]->get_epetra_map())));
      mueluParams.set<Teuchos::RCP<Epetra_Map>>(
          "contact slaveDofMap", Teuchos::rcpFromRef(prec_maps[1]->get_epetra_map()));
      mueluParams.set<Teuchos::RCP<Epetra_Map>>(
          "contact innerDofMap", Teuchos::rcpFromRef(prec_maps[2]->get_epetra_map()));
      mueluParams.set<Teuchos::RCP<Epetra_Map>>(
          "contact activeDofMap", Teuchos::rcpFromRef(prec_maps[3]->get_epetra_map()));
      // contact or contact/meshtying
      if (i_constr_prec_.begin()->first == NOX::Nln::sol_contact)
        mueluParams.set<std::string>("Core::ProblemType", "contact");
      // only meshtying
      else if (i_constr_prec_.begin()->first == NOX::Nln::sol_meshtying)
        mueluParams.set<std::string>("Core::ProblemType", "meshtying");
      else
        FOUR_C_THROW("Currently we support only a pure meshtying OR a pure contact problem!");

      mueluParams.set<int>("time step", step);
      // increase counter by one (historical reasons)
      mueluParams.set<int>("iter", nlnIter + 1);
    }
  }  // end: feed solver with contact/meshtying information

  return solver_params;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
NOX::Nln::SolutionType NOX::Nln::MeshTying::LinearSystem::get_active_lin_solver(
    const std::map<NOX::Nln::SolutionType, Teuchos::RCP<Core::LinAlg::Solver>>& solvers,
    Teuchos::RCP<Core::LinAlg::Solver>& currSolver)
{
  currSolver = solvers.at(NOX::Nln::sol_meshtying);
  return NOX::Nln::sol_meshtying;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::Nln::MeshTying::LinearSystem::throw_error(
    const std::string& functionName, const std::string& errorMsg) const
{
  if (utils_.isPrintType(::NOX::Utils::Error))
  {
    utils_.out() << "NOX::CONTACT::LinearSystem::" << functionName << " - " << errorMsg
                 << std::endl;
  }
  throw "NOX Error";
}

FOUR_C_NAMESPACE_CLOSE

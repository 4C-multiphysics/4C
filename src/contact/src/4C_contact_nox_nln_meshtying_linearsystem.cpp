// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_contact_nox_nln_meshtying_linearsystem.hpp"  // base class

#include "4C_contact_abstract_strategy.hpp"
#include "4C_contact_input.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
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
    const std::shared_ptr<NOX::Nln::Interface::JacobianBase> iJac,
    const NOX::Nln::CONSTRAINT::ReqInterfaceMap& iConstr,
    const std::shared_ptr<Core::LinAlg::SparseOperator>& J,
    const NOX::Nln::CONSTRAINT::PrecInterfaceMap& iConstrPrec,
    const std::shared_ptr<Core::LinAlg::SparseOperator>& M, const NOX::Nln::Vector& cloneVector,
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
    const std::shared_ptr<NOX::Nln::Interface::JacobianBase> iJac,
    const NOX::Nln::CONSTRAINT::ReqInterfaceMap& iConstr,
    const std::shared_ptr<Core::LinAlg::SparseOperator>& J,
    const NOX::Nln::CONSTRAINT::PrecInterfaceMap& iConstrPrec,
    const std::shared_ptr<Core::LinAlg::SparseOperator>& M, const NOX::Nln::Vector& cloneVector)
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

      auto precInterface = i_constr_prec_.begin();

      const auto& strategy = dynamic_cast<const Mortar::StrategyBase&>(*precInterface->second);

      std::shared_ptr<Core::LinAlg::Map> masterDofMap, slaveDofMap, innerDofMap, activeDofMap;
      strategy.collect_maps_for_preconditioner(
          masterDofMap, slaveDofMap, innerDofMap, activeDofMap);
      mueluParams.set<std::shared_ptr<Core::LinAlg::Map>>("contact masterDofMap", masterDofMap);
      mueluParams.set<std::shared_ptr<Core::LinAlg::Map>>("contact slaveDofMap", slaveDofMap);
      mueluParams.set<std::shared_ptr<Core::LinAlg::Map>>("contact innerDofMap", innerDofMap);
      mueluParams.set<std::shared_ptr<Core::LinAlg::Map>>("contact activeDofMap", activeDofMap);
      // contact or contact/meshtying
      if (precInterface->first == NOX::Nln::sol_contact)
        mueluParams.set<std::string>("Core::ProblemType", "contact");
      // only meshtying
      else if (precInterface->first == NOX::Nln::sol_meshtying)
        mueluParams.set<std::string>("Core::ProblemType", "meshtying");
      else
        FOUR_C_THROW("Currently we support only a pure meshtying OR a pure contact problem!");

      // construct the mapping of the dual node IDs to primal node IDs for Lagrange multiplier
      // aggregation
      std::map<int, int> dual2primal_map;

      const auto& slave_node_row_map = strategy.slave_row_nodes_ptr();

      const auto& discret = Global::Problem::instance()->get_dis("structure");
      const auto* solid_node_map = discret->node_row_map();

      for (int dual_lid = 0; dual_lid < slave_node_row_map->num_my_elements(); dual_lid++)
      {
        const int dual_gid = slave_node_row_map->gid(dual_lid);
        if (discret->have_global_node(dual_gid))
          (dual2primal_map)[dual_lid] = solid_node_map->lid(dual_gid);
      }

      mueluParams.set<std::shared_ptr<std::map<int, int>>>("Interface DualNodeID to PrimalNodeID",
          std::make_shared<std::map<int, int>>(dual2primal_map));

      if (precInterface->second->is_saddle_point_system())
      {
        const auto& sol_type = Teuchos::getIntegralValue<FourC::CONTACT::SolvingStrategy>(
            strategy.params(), "STRATEGY");
        if (sol_type == FourC::CONTACT::SolvingStrategy::lagmult)
        {
          // compute the nullspace vectors for the Lagrange multiplier field for MueLu
          if (solverPtr->params().isSublist("MueLu Parameters"))
          {
            const int dim_nullspace = discret->n_dim();

            auto block_mat_blocked_operator =
                std::dynamic_pointer_cast<const Core::LinAlg::BlockSparseMatrixBase>(
                    get_jacobian_operator());
            FOUR_C_ASSERT(block_mat_blocked_operator != nullptr,
                "Failed to cast blockMat to BlockSparseMatrixBase");
            const auto& mat11 = block_mat_blocked_operator->matrix(1, 1);
            const auto& dofmap = mat11.domain_map();

            // set the nullspace
            Core::LinAlg::MultiVector<double> nullspace(dofmap, dim_nullspace, true);
            for (int ldof = 0; ldof < dofmap.num_my_elements(); ++ldof)
            {
              nullspace.replace_local_value(ldof, ldof % dim_nullspace, 1.0);
            }

            // add the nullspace to the parameter list
            solverPtr->params()
                .sublist("Inverse2")
                .sublist("MueLu Parameters")
                .set<std::shared_ptr<Core::LinAlg::MultiVector<double>>>(
                    "nullspace", std::make_shared<Core::LinAlg::MultiVector<double>>(nullspace));
          }
        }
      }

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

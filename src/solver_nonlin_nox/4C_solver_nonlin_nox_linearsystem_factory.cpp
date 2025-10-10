// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_solver_nonlin_nox_linearsystem_factory.hpp"

#include "4C_cardiovascular0d_nox_nln_linearsystem.hpp"
#include "4C_constraint_nox_nln_lagpenconstraint_linearsystem.hpp"
#include "4C_contact_nox_nln_contact_linearsystem.hpp"
#include "4C_contact_nox_nln_meshtying_linearsystem.hpp"
#include "4C_linalg_sparseoperator.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_solver_nonlin_nox_globaldata.hpp"
#include "4C_solver_nonlin_nox_scaling.hpp"
#include "4C_structure_new_nox_nln_str_linearsystem.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_exceptions.hpp"

#include <NOX_Epetra_Interface_Jacobian.H>
#include <NOX_Epetra_Interface_Required.H>
#include <NOX_Epetra_Vector.H>
#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
NOX::Nln::LinSystem::Factory::Factory()
{
  // empty constructor
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<NOX::Nln::LinearSystemBase> NOX::Nln::LinSystem::Factory::build_linear_system(
    const NOX::Nln::LinSystem::LinearSystemType& linsystype, NOX::Nln::GlobalData& noxNlnGlobalData,
    const Teuchos::RCP<Core::LinAlg::SparseOperator>& jac, ::NOX::Epetra::Vector& cloneVector,
    const Teuchos::RCP<Core::LinAlg::SparseOperator>& precMat,
    const std::shared_ptr<NOX::Nln::Scaling>& scalingObject) const
{
  Teuchos::RCP<NOX::Nln::LinearSystemBase> linSys = Teuchos::null;

  // extract some stuff from the NOX::Nln::GlobalData object
  const NOX::Nln::LinearSystem::SolverMap& linSolvers = noxNlnGlobalData.get_linear_solvers();
  const Teuchos::RCP<::NOX::Epetra::Interface::Required>& iReq =
      noxNlnGlobalData.get_required_interface();
  const Teuchos::RCP<::NOX::Epetra::Interface::Jacobian>& iJac =
      noxNlnGlobalData.get_jacobian_interface();

  Teuchos::ParameterList& params = noxNlnGlobalData.get_nln_parameter_list();
  // printing parameters
  Teuchos::ParameterList& printParams = params.sublist("Printing", true);
  // linear solver parameters
  Teuchos::ParameterList& lsParams =
      params.sublist("Direction", true).sublist("Newton", true).sublist("Linear Solver", true);

  switch (linsystype)
  {
    // pure structural case
    case NOX::Nln::LinSystem::linear_system_structure:
    {
      linSys = Teuchos::make_rcp<NOX::Nln::Solid::LinearSystem>(
          printParams, lsParams, linSolvers, iReq, iJac, jac, precMat, cloneVector, scalingObject);
      break;
    }
    // structural/contact case
    case NOX::Nln::LinSystem::linear_system_structure_contact:
    {
      const NOX::Nln::CONSTRAINT::ReqInterfaceMap& iConstr =
          noxNlnGlobalData.get_constraint_interfaces();
      const NOX::Nln::CONSTRAINT::PrecInterfaceMap& iConstrPrec =
          noxNlnGlobalData.get_constraint_prec_interfaces();

      linSys = Teuchos::make_rcp<NOX::Nln::CONTACT::LinearSystem>(printParams, lsParams, linSolvers,
          iReq, iJac, iConstr, jac, iConstrPrec, precMat, cloneVector, scalingObject);
      break;
    }
    case NOX::Nln::LinSystem::linear_system_structure_meshtying:
    {
      const NOX::Nln::CONSTRAINT::ReqInterfaceMap& iConstr =
          noxNlnGlobalData.get_constraint_interfaces();
      const NOX::Nln::CONSTRAINT::PrecInterfaceMap& iConstrPrec =
          noxNlnGlobalData.get_constraint_prec_interfaces();

      linSys = Teuchos::make_rcp<NOX::Nln::MeshTying::LinearSystem>(printParams, lsParams,
          linSolvers, iReq, iJac, iConstr, jac, iConstrPrec, precMat, cloneVector, scalingObject);
      break;
    }
    // structural/cardiovascular0d case
    case NOX::Nln::LinSystem::linear_system_structure_cardiovascular0d:
    {
      linSys = Teuchos::make_rcp<NOX::Nln::Cardiovascular0D::LinearSystem>(
          printParams, lsParams, linSolvers, iReq, iJac, jac, precMat, cloneVector, scalingObject);
      break;
    }
    // structural/constraint case
    case NOX::Nln::LinSystem::linear_system_structure_lag_pen_constraint:
    {
      const NOX::Nln::CONSTRAINT::ReqInterfaceMap& iConstr =
          noxNlnGlobalData.get_constraint_interfaces();
      const NOX::Nln::CONSTRAINT::PrecInterfaceMap& iConstrPrec =
          noxNlnGlobalData.get_constraint_prec_interfaces();

      linSys = Teuchos::make_rcp<NOX::Nln::LAGPENCONSTRAINT::LinearSystem>(printParams, lsParams,
          linSolvers, iReq, iJac, iConstr, jac, iConstrPrec, precMat, cloneVector, scalingObject);

      break;
    }

    // default case
    default:
    {
      FOUR_C_THROW(
          "ERROR - NOX::Nln::LinSystem::Factory::BuildLinearSystem - "
          "No capable LinearSystem constructor was found (enum = {})!",
          linsystype);
      break;
    }
  }  // end switch

  // return the linear system
  return linSys;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<NOX::Nln::LinearSystemBase> NOX::Nln::LinSystem::build_linear_system(
    const NOX::Nln::LinSystem::LinearSystemType& linsystype, NOX::Nln::GlobalData& noxNlnGlobalData,
    const Teuchos::RCP<Core::LinAlg::SparseOperator>& jac, ::NOX::Epetra::Vector& cloneVector,
    const Teuchos::RCP<Core::LinAlg::SparseOperator>& precMat,
    const std::shared_ptr<NOX::Nln::Scaling>& scalingObject)
{
  Factory factory;
  return factory.build_linear_system(
      linsystype, noxNlnGlobalData, jac, cloneVector, precMat, scalingObject);
}

FOUR_C_NAMESPACE_CLOSE

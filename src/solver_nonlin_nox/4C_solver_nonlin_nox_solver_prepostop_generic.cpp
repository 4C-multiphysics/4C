// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_solver_nonlin_nox_solver_prepostop_generic.hpp"  // class definition

#include "4C_solver_nonlin_nox_aux.hpp"
#include "4C_solver_nonlin_nox_solver_linesearchbased.hpp"
#include "4C_solver_nonlin_nox_statustest_normf.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
NOX::Nln::Solver::PrePostOp::Generic::Generic()
{
  // empty constructor
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::Nln::Solver::PrePostOp::Generic::runPreIterate(const ::NOX::Solver::Generic& nlnSolver)
{
  // Set the current number of nonlinear iterations
  // this is necessary for the linear solver in some cases (e.g. contact)
  const Teuchos::RCP<Teuchos::ParameterList>& params =
      Teuchos::rcp_const_cast<Teuchos::ParameterList>(nlnSolver.getListPtr());

  const std::string dir_method_str(NOX::Nln::Aux::get_direction_method_list_name(*params));
  if (params->sublist("Direction").isSublist(dir_method_str))
  {
    if (params->sublist("Direction").sublist(dir_method_str).isSublist("Linear Solver"))
    {
      Teuchos::ParameterList& linearSolverParams =
          params->sublist("Direction").sublist(dir_method_str).sublist("Linear Solver");
      linearSolverParams.set<int>("Number of Nonlinear Iterations", nlnSolver.getNumIterations());
    }
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::Nln::Solver::PrePostOp::Generic::runPreSolve(const ::NOX::Solver::Generic& nlnSolver)
{
  const NOX::Nln::Solver::LineSearchBased* lsSolver =
      dynamic_cast<const NOX::Nln::Solver::LineSearchBased*>(&nlnSolver);

  if (lsSolver != nullptr)
  {
    // set the wanted tolerance for the linear solver
    const Teuchos::RCP<Teuchos::ParameterList>& params =
        Teuchos::rcp_const_cast<Teuchos::ParameterList>(nlnSolver.getListPtr());

    const std::string dir_method_str(NOX::Nln::Aux::get_direction_method_list_name(*params));
    if (params->sublist("Direction").isSublist(dir_method_str))
    {
      if (params->sublist("Direction").sublist(dir_method_str).isSublist("Linear Solver"))
      {
        Teuchos::ParameterList& linearSolverParams =
            params->sublist("Direction").sublist(dir_method_str).sublist("Linear Solver");

        // Find and get the "specified tolerance" of the structural normF test in the statusTest
        // object
        const ::NOX::StatusTest::Generic& statusTest = lsSolver->get_outer_status_test();
        double specified_tol = NOX::Nln::Aux::get_norm_f_class_variable(
            statusTest, NOX::Nln::StatusTest::quantity_structure, "SpecifiedTolerance");
        if (specified_tol == -1.0)
          specified_tol = NOX::Nln::Aux::get_norm_f_class_variable(
              statusTest, NOX::Nln::StatusTest::quantity_levelset_reinit, "SpecifiedTolerance");

        if (specified_tol == -1.0)
        {
          if (lsSolver->get_utils().isPrintType(::NOX::Utils::Warning))
          {
            lsSolver->get_utils().out()
                << "\n*** WARNING ***\n"
                << "There is no NOX::Nln::StatusTest::NormF test for the primal field \n"
                << "components. The \"Wanted Tolerance\" for the sublist\n"
                << "\"Linear Solver\" was set to its default value 1.0e-6!\n";
          }
          linearSolverParams.set<double>("Wanted Tolerance", 1.0e-6);
        }
        else
          linearSolverParams.set<double>("Wanted Tolerance", specified_tol);
      }
    }
  }
}

FOUR_C_NAMESPACE_CLOSE

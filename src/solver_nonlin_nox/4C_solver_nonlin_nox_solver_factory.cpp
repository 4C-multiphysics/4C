/*-----------------------------------------------------------*/
/*! \file

\brief Factory to create the desired non-linear solver object.



\level 3

*/
/*-----------------------------------------------------------*/

#include "4C_solver_nonlin_nox_solver_factory.hpp"

#include "4C_solver_nonlin_nox_globaldata.hpp"
#include "4C_solver_nonlin_nox_solver_linesearchbased.hpp"
#include "4C_solver_nonlin_nox_solver_ptc.hpp"
#include "4C_solver_nonlin_nox_solver_singlestep.hpp"

#include <NOX_Solver_Factory.H>
#include <NOX_Solver_Generic.H>
#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
NOX::NLN::Solver::Factory::Factory()
{
  // empty constructor

  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<::NOX::Solver::Generic> NOX::NLN::Solver::Factory::BuildSolver(
    const Teuchos::RCP<::NOX::Abstract::Group>& grp,
    const Teuchos::RCP<::NOX::StatusTest::Generic>& outerTests,
    const Teuchos::RCP<NOX::NLN::INNER::StatusTest::Generic>& innerTests,
    const Teuchos::RCP<NOX::NLN::GlobalData>& nlnGlobalData)
{
  Teuchos::RCP<::NOX::Solver::Generic> solver;
  Teuchos::RCP<Teuchos::ParameterList> params = nlnGlobalData->get_nln_parameter_list_ptr();

  std::string method = params->get<std::string>("Nonlinear Solver", "Line Search Based");

  if ((method == "Newton") or (method == "Line Search Based"))
  {
    solver =
        Teuchos::rcp(new NOX::NLN::Solver::LineSearchBased(grp, outerTests, innerTests, params));
  }
  else if (method == "Pseudo Transient")
  {
    solver =
        Teuchos::rcp(new NOX::NLN::Solver::PseudoTransient(grp, outerTests, innerTests, params));
  }
  else if (method == "Single Step")
  {
    solver = Teuchos::rcp(new NOX::NLN::Solver::SingleStep(grp, innerTests, params));
  }
  else if (not nlnGlobalData->GetIsConstrained())
  {
    // unconstrained problems are able to call the standard nox factory
    solver = ::NOX::Solver::buildSolver(grp, outerTests, params);
  }
  else
  {
    std::ostringstream msg;
    msg << "Error - NOX::NLN::Solver::Factory::buildSolver() - The \"Nonlinear Solver\" parameter\n"
        << "\"" << method
        << "\" is not a valid solver option for CONSTRAINED optimization problems.\n"
        << "Please fix your parameter list!\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, msg.str());
  }

  return solver;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<::NOX::Solver::Generic> NOX::NLN::Solver::BuildSolver(
    const Teuchos::RCP<::NOX::Abstract::Group>& grp,
    const Teuchos::RCP<::NOX::StatusTest::Generic>& outerTests,
    const Teuchos::RCP<NOX::NLN::INNER::StatusTest::Generic>& innerTests,
    const Teuchos::RCP<NOX::NLN::GlobalData>& nlnGlobalData)
{
  Factory factory;
  return factory.BuildSolver(grp, outerTests, innerTests, nlnGlobalData);
}

FOUR_C_NAMESPACE_CLOSE

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
NOX::Nln::Solver::Factory::Factory()
{
  // empty constructor

  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<::NOX::Solver::Generic> NOX::Nln::Solver::Factory::build_solver(
    const Teuchos::RCP<::NOX::Abstract::Group>& grp,
    const Teuchos::RCP<::NOX::StatusTest::Generic>& outerTests,
    const Teuchos::RCP<NOX::Nln::Inner::StatusTest::Generic>& innerTests,
    NOX::Nln::GlobalData& nlnGlobalData)
{
  Teuchos::RCP<::NOX::Solver::Generic> solver;
  Teuchos::RCP<Teuchos::ParameterList> params = nlnGlobalData.get_nln_parameter_list_ptr();

  std::string method = params->get<std::string>("Nonlinear Solver", "Line Search Based");

  if ((method == "Newton") or (method == "Line Search Based"))
  {
    solver =
        Teuchos::make_rcp<NOX::Nln::Solver::LineSearchBased>(grp, outerTests, innerTests, params);
  }
  else if (method == "Pseudo Transient")
  {
    solver =
        Teuchos::make_rcp<NOX::Nln::Solver::PseudoTransient>(grp, outerTests, innerTests, params);
  }
  else if (method == "Single Step")
  {
    solver = Teuchos::make_rcp<NOX::Nln::Solver::SingleStep>(grp, innerTests, params);
  }
  else if (not nlnGlobalData.is_constrained())
  {
    // unconstrained problems are able to call the standard nox factory
    solver = ::NOX::Solver::buildSolver(grp, outerTests, params);
  }
  else
  {
    std::ostringstream msg;
    msg << "Error - NOX::Nln::Solver::Factory::buildSolver() - The \"Nonlinear Solver\" parameter\n"
        << "\"" << method
        << "\" is not a valid solver option for CONSTRAINED optimization problems.\n"
        << "Please fix your parameter list!\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, msg.str());
  }

  return solver;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<::NOX::Solver::Generic> NOX::Nln::Solver::build_solver(
    const Teuchos::RCP<::NOX::Abstract::Group>& grp,
    const Teuchos::RCP<::NOX::StatusTest::Generic>& outerTests,
    const Teuchos::RCP<NOX::Nln::Inner::StatusTest::Generic>& innerTests,
    const Teuchos::RCP<NOX::Nln::GlobalData>& nlnGlobalData)
{
  Factory factory;
  return factory.build_solver(grp, outerTests, innerTests, *nlnGlobalData);
}

FOUR_C_NAMESPACE_CLOSE

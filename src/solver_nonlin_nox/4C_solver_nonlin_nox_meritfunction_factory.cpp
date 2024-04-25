/*-----------------------------------------------------------*/
/*! \file

\brief Factory to create a merit function evaluation object.



\level 3

*/
/*-----------------------------------------------------------*/

#include "4C_solver_nonlin_nox_meritfunction_factory.hpp"

#include "4C_solver_nonlin_nox_meritfunction_lagrangian.hpp"

#include <NOX_Utils.H>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
NOX::NLN::MeritFunction::Factory::Factory()
{
  // empty;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<::NOX::MeritFunction::Generic> NOX::NLN::MeritFunction::Factory::BuildMeritFunction(
    const NOX::NLN::GlobalData& noxNlnGlobalData) const
{
  Teuchos::RCP<::NOX::MeritFunction::Generic> mrtFctPtr;

  const Teuchos::ParameterList& solverOptionsList =
      noxNlnGlobalData.GetNlnParameterList().sublist("Solver Options");

  const std::string& mftype = solverOptionsList.get<std::string>("Merit Function");

  if (noxNlnGlobalData.GetIsConstrained())
    mrtFctPtr = BuildConstrainedMeritFunction(mftype, noxNlnGlobalData);
  else
    mrtFctPtr = BuildUnconstrainedMeritFunction(mftype, noxNlnGlobalData);

  return mrtFctPtr;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<::NOX::MeritFunction::Generic>
NOX::NLN::MeritFunction::Factory::BuildUnconstrainedMeritFunction(
    const std::string& mftype, const NOX::NLN::GlobalData& noxNlnGlobalData) const
{
  Teuchos::RCP<::NOX::MeritFunction::Generic> mrtFctPtr = Teuchos::null;

  if (mftype == "Sum of Squares")
  {
    // default NOX case, no pointer necessary
    mrtFctPtr = Teuchos::null;
  }
  else
  {
    std::ostringstream msg;
    msg << "Error - NOX::NLN::MeritFunction::buildUnconstraintedMeritFunction() - \n"
           "The \"Solver Options > Merit Function\" parameter \""
        << mftype
        << "\" is not a valid UNCONSTRAINTED merit function name.  Please "
           "fix your parameter list!";
    FOUR_C_THROW(msg.str().c_str());
  }

  return mrtFctPtr;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<::NOX::MeritFunction::Generic>
NOX::NLN::MeritFunction::Factory::BuildConstrainedMeritFunction(
    const std::string& mftype, const NOX::NLN::GlobalData& noxNlnGlobalData) const
{
  Teuchos::RCP<::NOX::MeritFunction::Generic> mrtFctPtr = Teuchos::null;

  if (mftype == "Sum of Squares")
  {
    // default NOX case, no pointer necessary
    mrtFctPtr = Teuchos::null;
  }
  else if (mftype.substr(0, 10) == "Lagrangian")
  {
    mrtFctPtr = Teuchos::rcp(new Lagrangian(mftype, noxNlnGlobalData.GetNoxUtilsPtr()));
  }
  else
  {
    std::ostringstream msg;
    msg << "Error - NOX::NLN::MeritFunction::buildConstrainedMeritFunction() - \n"
           "The \"Solver Options > Merit Function\" parameter \""
        << mftype
        << "\" is not a valid CONSTRAINT merit function name.  Please "
           "fix your parameter list!";
    FOUR_C_THROW(msg.str().c_str());
  }

  return mrtFctPtr;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<::NOX::MeritFunction::Generic> NOX::NLN::MeritFunction::BuildMeritFunction(
    const NOX::NLN::GlobalData& noxNlnGlobalData)
{
  const Factory factory;
  return factory.BuildMeritFunction(noxNlnGlobalData);
}

FOUR_C_NAMESPACE_CLOSE

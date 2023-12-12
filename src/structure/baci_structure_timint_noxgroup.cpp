/*----------------------------------------------------------------------------*/
/*! \file
\brief Definitions of NOX group for non-linear solution techniques
       used within implicit structural time integration
\level 1
*/

/*----------------------------------------------------------------------------*/
/* definitions */

/*----------------------------------------------------------------------------*/
/* headers */
#include "baci_structure_timint_noxgroup.H"

#include "baci_linalg_utils_sparse_algebra_math.H"

BACI_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
NOX::STR::Group::Group(BACI::STR::TimIntImpl& sti, Teuchos::ParameterList& printParams,
    const Teuchos::RCP<::NOX::Epetra::Interface::Required>& i, const ::NOX::Epetra::Vector& x,
    const Teuchos::RCP<::NOX::Epetra::LinearSystem>& linSys)
    : ::NOX::Epetra::Group(printParams, i, x, linSys)
{
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
::NOX::Abstract::Group::ReturnType NOX::STR::Group::computeF()
{
  ::NOX::Abstract::Group::ReturnType ret = ::NOX::Epetra::Group::computeF();

  // Not sure why we call this (historical reasons???)
  sharedLinearSystem.getObject(this);

  // We just evaluated the residual, so its linearization is not valid anymore.
  isValidJacobian = false;

  return ret;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
::NOX::Abstract::Group::ReturnType NOX::STR::Group::computeJacobian()
{
  ::NOX::Abstract::Group::ReturnType ret = ::NOX::Epetra::Group::computeJacobian();
  if (ret == ::NOX::Abstract::Group::Ok)
  {
    isValidJacobian = true;

    if (not isValidRHS) isValidRHS = true;
  }

  return ret;
}

BACI_NAMESPACE_CLOSE

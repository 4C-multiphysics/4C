#include "4C_structure_timint_noxgroup.hpp"

#include "4C_linalg_utils_sparse_algebra_math.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
NOX::Solid::Group::Group(FourC::Solid::TimIntImpl& sti, Teuchos::ParameterList& printParams,
    const Teuchos::RCP<::NOX::Epetra::Interface::Required>& i, const ::NOX::Epetra::Vector& x,
    const Teuchos::RCP<::NOX::Epetra::LinearSystem>& linSys)
    : ::NOX::Epetra::Group(printParams, i, x, linSys)
{
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
::NOX::Abstract::Group::ReturnType NOX::Solid::Group::computeF()
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
::NOX::Abstract::Group::ReturnType NOX::Solid::Group::computeJacobian()
{
  ::NOX::Abstract::Group::ReturnType ret = ::NOX::Epetra::Group::computeJacobian();
  if (ret == ::NOX::Abstract::Group::Ok)
  {
    isValidJacobian = true;

    if (not isValidRHS) isValidRHS = true;
  }

  return ret;
}

FOUR_C_NAMESPACE_CLOSE

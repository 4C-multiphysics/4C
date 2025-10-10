// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fsi_nox_group.hpp"

#include "4C_fsi_monolithicinterface.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
NOX::FSI::Group::Group(FourC::FSI::MonolithicInterface& mfsi, Teuchos::ParameterList& printParams,
    const Teuchos::RCP<::NOX::Epetra::Interface::Required>& i, const ::NOX::Epetra::Vector& x,
    const Teuchos::RCP<NOX::Nln::LinearSystemBase>& linSys)
    : NOX::Nln::GroupBase(printParams, i, x, linSys), mfsi_(mfsi)
{
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::FSI::Group::capture_system_state()
{
  // we know we already have the first linear system calculated

  Core::LinAlg::View rhs_view(RHSVector.getEpetraVector());
  mfsi_.setup_rhs(rhs_view, true);
  mfsi_.setup_system_matrix();

  isValidJacobian = true;
  isValidRHS = true;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
::NOX::Abstract::Group::ReturnType NOX::FSI::Group::computeF()
{
  ::NOX::Abstract::Group::ReturnType ret = NOX::Nln::GroupBase::computeF();
  if (ret == ::NOX::Abstract::Group::Ok)
  {
    if (not isValidJacobian)
    {
      mfsi_.setup_system_matrix();
      isValidJacobian = true;
    }
  }
  return ret;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
::NOX::Abstract::Group::ReturnType NOX::FSI::Group::computeJacobian()
{
  ::NOX::Abstract::Group::ReturnType ret = NOX::Nln::GroupBase::computeJacobian();
  if (ret == ::NOX::Abstract::Group::Ok)
  {
    if (not isValidRHS)
    {
      Core::LinAlg::View rhs_view(RHSVector.getEpetraVector());
      mfsi_.setup_rhs(rhs_view);
      isValidRHS = true;
    }
  }
  return ret;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
::NOX::Abstract::Group::ReturnType NOX::FSI::Group::computeNewton(Teuchos::ParameterList& p)
{
  Core::LinAlg::View rhs_view(RHSVector.getEpetraVector());
  mfsi_.scale_system(rhs_view);

  ::NOX::Abstract::Group::ReturnType status = NOX::Nln::GroupBase::computeNewton(p);
  Core::LinAlg::View newton_vector_view(NewtonVector.getEpetraVector());

  mfsi_.unscale_solution(newton_vector_view, rhs_view);

  // check return value of computeNewton call
  if (status == ::NOX::Abstract::Group::NotConverged || status == ::NOX::Abstract::Group::Failed)
    FOUR_C_THROW("NOX::FSI::Group::computeNewton: linear solver not converged...");

  return status;
}

FOUR_C_NAMESPACE_CLOSE

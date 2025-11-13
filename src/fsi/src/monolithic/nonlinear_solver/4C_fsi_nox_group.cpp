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
    const std::shared_ptr<NOX::Nln::Interface::RequiredBase> i, const NOX::Nln::Vector& x,
    const Teuchos::RCP<NOX::Nln::LinearSystemBase>& linSys)
    : NOX::Nln::GroupBase(printParams, i, x, linSys), mfsi_(mfsi)
{
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::FSI::Group::capture_system_state()
{
  // we know we already have the first linear system calculated

  mfsi_.setup_rhs(RHSVector.get_linalg_vector(), true);
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
      mfsi_.setup_rhs(RHSVector.get_linalg_vector());
      isValidRHS = true;
    }
  }
  return ret;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
::NOX::Abstract::Group::ReturnType NOX::FSI::Group::computeNewton(Teuchos::ParameterList& p)
{
  mfsi_.scale_system(RHSVector.get_linalg_vector());

  ::NOX::Abstract::Group::ReturnType status = NOX::Nln::GroupBase::computeNewton(p);

  mfsi_.unscale_solution(NewtonVector.get_linalg_vector(), RHSVector.get_linalg_vector());

  // check return value of computeNewton call
  if (status == ::NOX::Abstract::Group::NotConverged || status == ::NOX::Abstract::Group::Failed)
    FOUR_C_THROW("NOX::FSI::Group::computeNewton: linear solver not converged...");

  return status;
}

FOUR_C_NAMESPACE_CLOSE

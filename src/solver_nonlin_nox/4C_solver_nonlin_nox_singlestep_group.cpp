// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_solver_nonlin_nox_singlestep_group.hpp"

#include "4C_solver_nonlin_nox_group_prepostoperator.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
NOX::Nln::SINGLESTEP::Group::Group(Teuchos::ParameterList& printParams,
    Teuchos::ParameterList& grpOptionParams,
    const std::shared_ptr<NOX::Nln::Interface::RequiredBase> i, const NOX::Nln::Vector& x,
    const Teuchos::RCP<NOX::Nln::LinearSystemBase>& linSys)
    : NOX::Nln::Group(printParams, grpOptionParams, i, x, linSys)
{
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
NOX::Nln::SINGLESTEP::Group::Group(const NOX::Nln::SINGLESTEP::Group& source, ::NOX::CopyType type)
    : NOX::Nln::Group(source, type)
{
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<::NOX::Abstract::Group> NOX::Nln::SINGLESTEP::Group::clone(::NOX::CopyType type) const
{
  Teuchos::RCP<::NOX::Abstract::Group> newgrp =
      Teuchos::make_rcp<NOX::Nln::SINGLESTEP::Group>(*this, type);
  return newgrp;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::Nln::SINGLESTEP::Group::computeX(
    const ::NOX::Abstract::Group& grp, const ::NOX::Abstract::Vector& d, double step)
{
  // Cast to appropriate type, then call the "native" computeX
  const NOX::Nln::SINGLESTEP::Group* nlngrp =
      dynamic_cast<const NOX::Nln::SINGLESTEP::Group*>(&grp);
  if (nlngrp == nullptr) throw_error("computeX", "dyn_cast to nox_nln_group failed!");
  const auto& epetrad = dynamic_cast<const NOX::Nln::Vector&>(d);

  computeX(*nlngrp, epetrad, step);
  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::Nln::SINGLESTEP::Group::computeX(
    const NOX::Nln::SINGLESTEP::Group& grp, const NOX::Nln::Vector& d, double step)
{
  prePostOperatorPtr_->run_pre_compute_x(grp, d.get_linalg_vector(), step, *this);

  reset_is_valid();

  step = 1.0;
  xVector.update(-1.0, d, step, grp.xVector);

  prePostOperatorPtr_->run_post_compute_x(grp, d.get_linalg_vector(), step, *this);

  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::Nln::SINGLESTEP::Group::throw_error(
    const std::string& functionName, const std::string& errorMsg) const
{
  std::ostringstream msg;
  msg << "ERROR - NOX::Nln::SINGLESTEP::Group::" << functionName << " - " << errorMsg << std::endl;

  FOUR_C_THROW("{}", msg.str());
}

FOUR_C_NAMESPACE_CLOSE

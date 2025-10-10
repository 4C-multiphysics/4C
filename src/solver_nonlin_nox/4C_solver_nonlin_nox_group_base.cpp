// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_solver_nonlin_nox_group_base.hpp"

#include "4C_utils_exceptions.hpp"

#include <NOX_SolverStats.hpp>
#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN

NOX::Nln::GroupBase::GroupBase(Teuchos::ParameterList& printParams,
    const Teuchos::RCP<::NOX::Epetra::Interface::Required>& i, const ::NOX::Epetra::Vector& x,
    const Teuchos::RCP<NOX::Nln::LinearSystemBase>& linSys)
    : utils(printParams),
      xVector(x, ::NOX::DeepCopy),
      RHSVector(x, ::NOX::ShapeCopy),
      gradVector(x, ::NOX::ShapeCopy),
      NewtonVector(x, ::NOX::ShapeCopy),
      linearSystemPtr(linSys),
      userInterfacePtr(i),
      lastLinearSolveConverged(false),
      lastNumIterations(-1),
      lastAchievedTol(-1.0)
{
  reset_is_valid();
}

NOX::Nln::GroupBase::GroupBase(const GroupBase& source, ::NOX::CopyType type)
    : utils(source.utils),
      xVector(source.xVector, type),
      RHSVector(source.RHSVector, type),
      gradVector(source.gradVector, type),
      NewtonVector(source.NewtonVector, type),
      linearSystemPtr(source.linearSystemPtr),
      userInterfacePtr(source.userInterfacePtr),
      lastLinearSolveConverged(source.lastLinearSolveConverged),
      lastNumIterations(source.lastNumIterations),
      lastAchievedTol(source.lastAchievedTol)
{
  FOUR_C_ASSERT(type == ::NOX::DeepCopy || type == ::NOX::ShapeCopy,
      "NOX::Nln::GroupBase::GroupBase() - invalid copy type provided");

  switch (type)
  {
    case ::NOX::DeepCopy:
      isValidRHS = source.isValidRHS;
      isValidJacobian = source.isValidJacobian;
      isValidGrad = source.isValidGrad;
      isValidNewton = source.isValidNewton;
      break;

    case ::NOX::ShapeCopy:
      reset_is_valid();
      break;
  }
}

void NOX::Nln::GroupBase::reset_is_valid()
{
  isValidRHS = false;
  isValidJacobian = false;
  isValidGrad = false;
  isValidNewton = false;
}

Teuchos::RCP<::NOX::Abstract::Group> NOX::Nln::GroupBase::clone(::NOX::CopyType type) const
{
  Teuchos::RCP<::NOX::Abstract::Group> newgrp = Teuchos::rcp(new GroupBase(*this, type));
  return newgrp;
}

::NOX::Abstract::Group& NOX::Nln::GroupBase::operator=(const Group& source_abs)
{
  const auto& source = dynamic_cast<const GroupBase&>(source_abs);

  // Copy the xVector
  xVector = source.xVector;

  // Update the isValid flags
  isValidRHS = source.isValidRHS;
  isValidGrad = source.isValidGrad;
  isValidNewton = source.isValidNewton;
  isValidJacobian = source.isValidJacobian;

  // Only copy vectors that are valid
  if (isValidRHS) RHSVector = source.RHSVector;
  if (isValidGrad) gradVector = source.gradVector;
  if (isValidNewton) NewtonVector = source.NewtonVector;

  lastLinearSolveConverged = source.lastLinearSolveConverged;
  lastNumIterations = source.lastNumIterations;
  lastAchievedTol = source.lastAchievedTol;

  return *this;
}

Teuchos::RCP<const ::NOX::Abstract::Vector> NOX::Nln::GroupBase::getXPtr() const
{
  FOUR_C_THROW("NOX::Nln::GroupBase::getXPtr() - Not implemented.");
}

Teuchos::RCP<const ::NOX::Abstract::Vector> NOX::Nln::GroupBase::getFPtr() const
{
  FOUR_C_THROW("NOX::Nln::GroupBase::getFPtr() - Not implemented.");
}

Teuchos::RCP<const ::NOX::Abstract::Vector> NOX::Nln::GroupBase::getGradientPtr() const
{
  FOUR_C_THROW("NOX::Nln::GroupBase::getGradientPtr() - Not implemented.");
}

Teuchos::RCP<const ::NOX::Abstract::Vector> NOX::Nln::GroupBase::getNewtonPtr() const
{
#if (FOUR_C_TRILINOS_INTERNAL_VERSION_GE(2025, 4))
  FOUR_C_THROW("NOX::Nln::GroupBase::getNewtonPtr() - Not implemented.");
#else
  return Teuchos::rcpFromRef(NewtonVector);
#endif
}

void NOX::Nln::GroupBase::setX(const ::NOX::Abstract::Vector& y)
{
  reset_is_valid();
  xVector = dynamic_cast<const ::NOX::Epetra::Vector&>(y);
}

void NOX::Nln::GroupBase::computeX(
    const ::NOX::Abstract::Group& grp, const ::NOX::Abstract::Vector& d, double step)
{
  const auto& grp_base = dynamic_cast<const GroupBase&>(grp);
  const auto& epetra_d = dynamic_cast<const ::NOX::Epetra::Vector&>(d);

  reset_is_valid();
  xVector.update(1.0, grp_base.xVector, step, epetra_d);
}

::NOX::Abstract::Group::ReturnType NOX::Nln::GroupBase::computeF()
{
  if (isF()) return ::NOX::Abstract::Group::Ok;

  isValidRHS = userInterfacePtr->computeF(xVector.getEpetraVector(), RHSVector.getEpetraVector(),
      ::NOX::Epetra::Interface::Required::Residual);

  FOUR_C_ASSERT(isValidRHS, "NOX::Nln::GroupBase::computeF() - failed");

  return ::NOX::Abstract::Group::Ok;
}

::NOX::Abstract::Group::ReturnType NOX::Nln::GroupBase::computeJacobian()
{
  if (isJacobian()) return ::NOX::Abstract::Group::Ok;

  isValidJacobian = linearSystemPtr->compute_jacobian(xVector);

  FOUR_C_ASSERT(isValidJacobian, "NOX::Nln::GroupBase::computeJacobian() - failed");

  return ::NOX::Abstract::Group::Ok;
}

::NOX::Abstract::Group::ReturnType NOX::Nln::GroupBase::computeGradient()
{
  if (isGradient()) return ::NOX::Abstract::Group::Ok;

  FOUR_C_ASSERT(isF(), "NOX::Nln::GroupBase::computeGradient() - RHS is out of date wrt X!");
  FOUR_C_ASSERT(
      isJacobian(), "NOX::Nln::GroupBase::computeGradient() - Jacobian is out of date wrt X!");

  isValidGrad = linearSystemPtr->apply_jacobian_transpose(RHSVector, gradVector);

  return ::NOX::Abstract::Group::Ok;
}

::NOX::Abstract::Group::ReturnType NOX::Nln::GroupBase::computeNewton(Teuchos::ParameterList& p)
{
  if (isNewton()) return ::NOX::Abstract::Group::Ok;

  FOUR_C_ASSERT(isF(), "NOX::Nln::GroupBase::computeNewton() - RHS is out of date wrt X!");
  FOUR_C_ASSERT(
      isJacobian(), "NOX::Nln::GroupBase::computeNewton() - Jacobian is out of date wrt X!");

  ::NOX::Abstract::Group::ReturnType status;

  NewtonVector.init(0.0);

  status = applyJacobianInverse(p, RHSVector, NewtonVector);

  // Scale by -1
  NewtonVector.scale(-1.0);

  // Update state even if linear solve failed since we may still want top use the vector
  isValidNewton = true;

  return status;
}

::NOX::Abstract::Group::ReturnType NOX::Nln::GroupBase::applyJacobian(
    const ::NOX::Abstract::Vector& input, ::NOX::Abstract::Vector& result) const
{
  const auto& epetra_input = dynamic_cast<const ::NOX::Epetra::Vector&>(input);
  auto& epetra_result = dynamic_cast<::NOX::Epetra::Vector&>(result);

  if (!isJacobian()) return ::NOX::Abstract::Group::BadDependency;

  const bool status = linearSystemPtr->apply_jacobian(epetra_input, epetra_result);

  return status ? ::NOX::Abstract::Group::Ok : ::NOX::Abstract::Group::Failed;
}

::NOX::Abstract::Group::ReturnType NOX::Nln::GroupBase::applyJacobianInverse(
    Teuchos::ParameterList& p, const ::NOX::Abstract::Vector& input,
    ::NOX::Abstract::Vector& result) const
{
  const auto& epetra_input = dynamic_cast<const ::NOX::Epetra::Vector&>(input);
  auto& epetra_result = dynamic_cast<::NOX::Epetra::Vector&>(result);

  if (!isJacobian()) return ::NOX::Abstract::Group::BadDependency;

  // Save linear solve stats
  lastLinearSolveConverged =
      linearSystemPtr->apply_jacobian_inverse(p, epetra_input, epetra_result);
  lastNumIterations = p.sublist("Output").get("Number of Linear Iterations", 0);
  lastAchievedTol = p.sublist("Output").get("Achieved Tolerance", 0.0);

  return lastLinearSolveConverged ? ::NOX::Abstract::Group::Ok
                                  : ::NOX::Abstract::Group::NotConverged;
}

::NOX::Abstract::Group::ReturnType NOX::Nln::GroupBase::applyJacobianTranspose(
    const ::NOX::Abstract::Vector& input, ::NOX::Abstract::Vector& result) const
{
  const auto& epetra_input = dynamic_cast<const ::NOX::Epetra::Vector&>(input);
  auto& epetra_result = dynamic_cast<::NOX::Epetra::Vector&>(result);

  if (!isJacobian()) return ::NOX::Abstract::Group::BadDependency;

  const bool status = linearSystemPtr->apply_jacobian_transpose(epetra_input, epetra_result);

  return status ? ::NOX::Abstract::Group::Ok : ::NOX::Abstract::Group::Failed;
}

bool NOX::Nln::GroupBase::isF() const { return isValidRHS; }

bool NOX::Nln::GroupBase::isJacobian() const { return isValidJacobian; }

bool NOX::Nln::GroupBase::isGradient() const { return isValidGrad; }

bool NOX::Nln::GroupBase::isNewton() const { return isValidNewton; }

const ::NOX::Abstract::Vector& NOX::Nln::GroupBase::getX() const { return xVector; }

const ::NOX::Abstract::Vector& NOX::Nln::GroupBase::getF() const { return RHSVector; }

const ::NOX::Abstract::Vector& NOX::Nln::GroupBase::getGradient() const { return gradVector; }

const ::NOX::Abstract::Vector& NOX::Nln::GroupBase::getNewton() const { return NewtonVector; }

double NOX::Nln::GroupBase::getNormF() const
{
  FOUR_C_ASSERT(isF(), "NOX::Nln::GroupBase::getNormF() - invalid RHS");

  return RHSVector.norm();
}

Teuchos::RCP<::NOX::Epetra::Interface::Required> NOX::Nln::GroupBase::get_required_interface()
{
  return userInterfacePtr;
}

Teuchos::RCP<const NOX::Nln::LinearSystemBase> NOX::Nln::GroupBase::get_linear_system() const
{
  return linearSystemPtr;
}

Teuchos::RCP<NOX::Nln::LinearSystemBase> NOX::Nln::GroupBase::get_linear_system()
{
  return linearSystemPtr;
}

void NOX::Nln::GroupBase::logLastLinearSolveStats(::NOX::SolverStats& stats) const
{
  stats.linearSolve.logLinearSolve(
      lastLinearSolveConverged, lastNumIterations, lastAchievedTol, 0.0, 0.0);
}

FOUR_C_NAMESPACE_CLOSE

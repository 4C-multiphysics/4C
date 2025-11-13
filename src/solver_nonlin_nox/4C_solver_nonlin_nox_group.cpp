// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_solver_nonlin_nox_group.hpp"

#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_solver_nonlin_nox_group_prepostoperator.hpp"
#include "4C_solver_nonlin_nox_interface_jacobian.hpp"
#include "4C_solver_nonlin_nox_interface_required.hpp"
#include "4C_solver_nonlin_nox_linearsystem.hpp"
#include "4C_solver_nonlin_nox_solver_ptc.hpp"

#include <NOX_StatusTest_NormF.H>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
NOX::Nln::Group::Group(Teuchos::ParameterList& printParams, Teuchos::ParameterList& grpOptionParams,
    const std::shared_ptr<NOX::Nln::Interface::RequiredBase> i, const NOX::Nln::Vector& x,
    const Teuchos::RCP<NOX::Nln::LinearSystemBase>& linSys)
    : GroupBase(printParams, i, x, linSys),
      skipUpdateX_(false),
      prePostOperatorPtr_(Teuchos::make_rcp<NOX::Nln::GROUP::PrePostOperator>(grpOptionParams))
{
  // empty constructor
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
NOX::Nln::Group::Group(const NOX::Nln::Group& source, ::NOX::CopyType type)
    : GroupBase(source, type), skipUpdateX_(false), prePostOperatorPtr_(source.prePostOperatorPtr_)
{
  switch (type)
  {
    case ::NOX::DeepCopy:
    {
      skipUpdateX_ = source.skipUpdateX_;
      break;
    }
    default:
      break;
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<::NOX::Abstract::Group> NOX::Nln::Group::clone(::NOX::CopyType type) const
{
  Teuchos::RCP<::NOX::Abstract::Group> newgrp = Teuchos::make_rcp<NOX::Nln::Group>(*this, type);
  return newgrp;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
::NOX::Abstract::Group& NOX::Nln::Group::operator=(const ::NOX::Abstract::Group& source)
{
  NOX::Nln::GroupBase::operator=(source);
  const NOX::Nln::Group& nln_src = dynamic_cast<const NOX::Nln::Group&>(source);

  this->skipUpdateX_ = nln_src.skipUpdateX_;

  return *this;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::Nln::Group::computeX(
    const ::NOX::Abstract::Group& grp, const ::NOX::Abstract::Vector& d, double step)
{
  // Cast to appropriate type, then call the "native" computeX
  const NOX::Nln::Group* nlngrp = dynamic_cast<const NOX::Nln::Group*>(&grp);
  if (nlngrp == nullptr) throw_error("computeX", "dyn_cast to nox_nln_group failed!");
  const auto& epetrad = dynamic_cast<const NOX::Nln::Vector&>(d);

  computeX(*nlngrp, epetrad, step);
  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::Nln::Group::computeX(const NOX::Nln::Group& grp, const NOX::Nln::Vector& d, double step)
{
  skipUpdateX_ = false;

  // Some call further down will perform a const-cast on d. fixme
  prePostOperatorPtr_->run_pre_compute_x(grp, d.get_linalg_vector(), step, *this);

  reset_is_valid();

  if (not skipUpdateX_) xVector.update(1.0, grp.xVector, step, d);

  prePostOperatorPtr_->run_post_compute_x(grp, d.get_linalg_vector(), step, *this);

  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
::NOX::Abstract::Group::ReturnType NOX::Nln::Group::set_f(Teuchos::RCP<NOX::Nln::Vector> Fptr)
{
  if (Fptr == Teuchos::null or Fptr->get_linalg_vector().get_map().num_global_elements() == 0)
    return ::NOX::Abstract::Group::BadDependency;

  RHSVector = *Fptr;
  isValidRHS = true;

  return ::NOX::Abstract::Group::Ok;
}

#if !(FOUR_C_TRILINOS_INTERNAL_VERSION_GE(2025, 4))
/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::Nln::Group::reset_x() { xVector.init(0.0); }
#endif

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::Nln::Group::set_skip_update_x(bool skipUpdateX) { skipUpdateX_ = skipUpdateX; }

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
::NOX::Abstract::Group::ReturnType NOX::Nln::Group::computeF()
{
  prePostOperatorPtr_->run_pre_compute_f(RHSVector.get_linalg_vector(), *this);

  if (isF()) return ::NOX::Abstract::Group::Ok;

  const bool success = userInterfacePtr->computeF(xVector.get_linalg_vector(),
      RHSVector.get_linalg_vector(), ::NOX::Epetra::Interface::Required::Residual);

  if (not success)
  {
    throw "NOX::Nln::Group::computeF() - fill failed";
  }

  isValidRHS = true;

  prePostOperatorPtr_->run_post_compute_f(RHSVector.get_linalg_vector(), *this);

  return ::NOX::Abstract::Group::Ok;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
::NOX::Abstract::Group::ReturnType NOX::Nln::Group::compute_f_and_jacobian()
{
  // initialize the return type
  ::NOX::Abstract::Group::ReturnType ret = ::NOX::Abstract::Group::Failed;

  // update right hand side vector
  if (!isF() and isJacobian())
  {
    ret = computeF();
  }
  // update right hand side vector and jacobian
  else if (!isJacobian())
  {
    isValidRHS = false;
    prePostOperatorPtr_->run_pre_compute_f(RHSVector.get_linalg_vector(), *this);

    bool status = false;
    Teuchos::RCP<NOX::Nln::LinearSystem> nlnSharedLinearSystem =
        Teuchos::rcp_dynamic_cast<NOX::Nln::LinearSystem>(linearSystemPtr);

    if (nlnSharedLinearSystem.is_null())
      throw_error("compute_f_and_jacobian", "Dynamic cast of the shared linear system failed!");

    status = nlnSharedLinearSystem->compute_f_and_jacobian(xVector, RHSVector);
    if (!status) throw_error("compute_f_and_jacobian", "evaluation failed!");

    isValidRHS = true;
    isValidJacobian = true;

    ret = ::NOX::Abstract::Group::Ok;

    prePostOperatorPtr_->run_post_compute_f(RHSVector.get_linalg_vector(), *this);
  }
  // nothing to do, because all quantities are up-to-date
  else
  {
    prePostOperatorPtr_->run_pre_compute_f(RHSVector.get_linalg_vector(), *this);

    ret = ::NOX::Abstract::Group::Ok;
  }

  return ret;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
::NOX::Abstract::Group::ReturnType NOX::Nln::Group::applyJacobianInverse(Teuchos::ParameterList& p,
    const ::NOX::Abstract::Vector& input, ::NOX::Abstract::Vector& result) const
{
  prePostOperatorPtr_->run_pre_apply_jacobian_inverse(input, result, xVector, *this);

  ::NOX::Abstract::Group::ReturnType status =
      NOX::Nln::GroupBase::applyJacobianInverse(p, input, result);

  prePostOperatorPtr_->run_post_apply_jacobian_inverse(input, result, xVector, *this);

  return status;
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<const NOX::Nln::Interface::Required> NOX::Nln::Group::get_nln_req_interface_ptr()
    const
{
  const auto userInterfaceNlnPtr =
      std::dynamic_pointer_cast<NOX::Nln::Interface::Required>(userInterfacePtr);
  FOUR_C_ASSERT(userInterfaceNlnPtr,
      "NOX::Nln::Group::get_nln_req_interface_ptr(): required interface cast failed");

  return userInterfaceNlnPtr;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<const std::vector<double>> NOX::Nln::Group::get_rhs_norms(
    const std::vector<::NOX::Abstract::Vector::NormType>& type,
    const std::vector<NOX::Nln::StatusTest::QuantityType>& chQ,
    Teuchos::RCP<const std::vector<::NOX::StatusTest::NormF::ScaleType>> scale) const
{
  if (scale.is_null())
    scale = Teuchos::make_rcp<std::vector<::NOX::StatusTest::NormF::ScaleType>>(
        chQ.size(), ::NOX::StatusTest::NormF::Unscaled);

  Teuchos::RCP<std::vector<double>> norms = Teuchos::make_rcp<std::vector<double>>(0);

  double rval = -1.0;
  for (std::size_t i = 0; i < chQ.size(); ++i)
  {
    rval = get_nln_req_interface_ptr()->get_primary_rhs_norms(RHSVector.get_linalg_vector(), chQ[i],
        type[i], (*scale)[i] == ::NOX::StatusTest::NormF::Scaled);
    if (rval >= 0.0)
    {
      norms->push_back(rval);
    }
    else
    {
      std::ostringstream msg;
      msg << "The desired quantity"
             " for the \"NormF\" Status Test could not be found! (enum="
          << chQ[i] << " | " << NOX::Nln::StatusTest::quantity_type_to_string(chQ[i]) << ")"
          << std::endl;
      throw_error("get_rhs_norms", msg.str());
    }
  }

  return norms;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<std::vector<double>> NOX::Nln::Group::get_solution_update_rms(
    const ::NOX::Abstract::Vector& xOld, const std::vector<double>& aTol,
    const std::vector<double>& rTol, const std::vector<NOX::Nln::StatusTest::QuantityType>& chQ,
    const std::vector<bool>& disable_implicit_weighting) const
{
  const auto& xOldNox = dynamic_cast<const NOX::Nln::Vector&>(xOld);
  Teuchos::RCP<std::vector<double>> rms = Teuchos::make_rcp<std::vector<double>>(0);

  double rval = -1.0;
  for (std::size_t i = 0; i < chQ.size(); ++i)
  {
    rval = get_nln_req_interface_ptr()->get_primary_solution_update_rms(xVector.get_linalg_vector(),
        xOldNox.get_linalg_vector(), aTol[i], rTol[i], chQ[i], disable_implicit_weighting[i]);
    if (rval >= 0.0)
    {
      rms->push_back(rval);
    }
    else
    {
      std::ostringstream msg;
      msg << "The desired quantity"
             " for the \"NormWRMS\" Status Test could not be found! (enum="
          << chQ[i] << " | " << NOX::Nln::StatusTest::quantity_type_to_string(chQ[i]) << ")"
          << std::endl;
      throw_error("get_solution_update_rms", msg.str());
    }
  }

  return rms;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double NOX::Nln::Group::get_trial_update_norm(const ::NOX::Abstract::Vector& dir,
    const ::NOX::Abstract::Vector::NormType normtype, const StatusTest::QuantityType quantity,
    const StatusTest::NormUpdate::ScaleType scale) const
{
  const std::vector<::NOX::Abstract::Vector::NormType> normtypes(1, normtype);
  const std::vector<StatusTest::QuantityType> quantities(1, quantity);
  const std::vector<StatusTest::NormUpdate::ScaleType> scales(1, scale);

  if (!tmp_vector_ptr_ or
      !tmp_vector_ptr_->get_map().same_as(xVector.get_linalg_vector().get_map()) or
      tmp_vector_ptr_.get() == &xVector.get_linalg_vector())
    tmp_vector_ptr_ = std::make_shared<Core::LinAlg::Vector<double>>(xVector.get_linalg_vector());
  else
    tmp_vector_ptr_->scale(1.0, xVector.get_linalg_vector());

  // change the internally stored x-vector for the norm evaluation
  auto& x_mutable = const_cast<NOX::Nln::Vector&>(xVector);
  x_mutable.update(1.0, dir, 1.0);

  NOX::Nln::Vector xold(tmp_vector_ptr_, NOX::Nln::Vector::MemoryType::View);

  const double rval =
      get_solution_update_norms(xold, normtypes, quantities, Teuchos::rcpFromRef(scales))->at(0);

  // un-do the changes to the x-vector
  x_mutable.get_linalg_vector().scale(1.0, *tmp_vector_ptr_);

  return rval;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<std::vector<double>> NOX::Nln::Group::get_solution_update_norms(
    const ::NOX::Abstract::Vector& xOld, const std::vector<::NOX::Abstract::Vector::NormType>& type,
    const std::vector<StatusTest::QuantityType>& chQ,
    Teuchos::RCP<const std::vector<StatusTest::NormUpdate::ScaleType>> scale) const
{
  const auto& xOldNox = dynamic_cast<const NOX::Nln::Vector&>(xOld);
  if (scale.is_null())
    scale = Teuchos::make_rcp<std::vector<StatusTest::NormUpdate::ScaleType>>(
        chQ.size(), StatusTest::NormUpdate::Unscaled);

  Teuchos::RCP<std::vector<double>> norms = Teuchos::make_rcp<std::vector<double>>(0);

  double rval = -1.0;
  for (std::size_t i = 0; i < chQ.size(); ++i)
  {
    rval = get_nln_req_interface_ptr()->get_primary_solution_update_norms(
        xVector.get_linalg_vector(), xOldNox.get_linalg_vector(), chQ[i], type[i],
        (*scale)[i] == StatusTest::NormUpdate::Scaled);
    if (rval >= 0.0)
    {
      norms->push_back(rval);
    }
    else
    {
      std::ostringstream msg;
      msg << "The desired quantity"
             " for the \"NormIncr\" Status Test could not be found! (enum="
          << chQ[i] << " | " << NOX::Nln::StatusTest::quantity_type_to_string(chQ[i]) << ")"
          << std::endl;
      throw_error("get_solution_update_norms", msg.str());
    }
  }

  return norms;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<std::vector<double>> NOX::Nln::Group::get_previous_solution_norms(
    const ::NOX::Abstract::Vector& xOld, const std::vector<::NOX::Abstract::Vector::NormType>& type,
    const std::vector<StatusTest::QuantityType>& chQ,
    Teuchos::RCP<const std::vector<StatusTest::NormUpdate::ScaleType>> scale) const
{
  const auto& xOldNox = dynamic_cast<const NOX::Nln::Vector&>(xOld);
  if (scale.is_null())
    scale = Teuchos::make_rcp<std::vector<StatusTest::NormUpdate::ScaleType>>(
        chQ.size(), StatusTest::NormUpdate::Unscaled);

  Teuchos::RCP<std::vector<double>> norms = Teuchos::make_rcp<std::vector<double>>(0);

  double rval = -1.0;
  for (std::size_t i = 0; i < chQ.size(); ++i)
  {
    rval = get_nln_req_interface_ptr()->get_previous_primary_solution_norms(
        xOldNox.get_linalg_vector(), chQ[i], type[i],
        (*scale)[i] == StatusTest::NormUpdate::Scaled);
    if (rval >= 0.0)
    {
      norms->push_back(rval);
    }
    else
    {
      std::ostringstream msg;
      msg << "The desired quantity"
             " for the \"NormUpdate\" Status Test could not be found! (enum="
          << chQ[i] << " | " << NOX::Nln::StatusTest::quantity_type_to_string(chQ[i]) << ")"
          << std::endl;
      throw_error("get_previous_solution_norms", msg.str());
    }
  }

  return norms;
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::Nln::Group::reset_pre_post_operator(
    Teuchos::ParameterList& grpOptionParams, const bool& resetIsValidFlags)
{
  if (resetIsValidFlags) reset_is_valid();

  prePostOperatorPtr_->reset(grpOptionParams);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::Nln::Group::reset_lin_sys_pre_post_operator(
    Teuchos::ParameterList& linearSolverParams, const bool& resetIsValidFlags)
{
  if (resetIsValidFlags) reset_is_valid();

  Teuchos::RCP<NOX::Nln::LinearSystem> nlnLinsysPtr =
      Teuchos::rcp_dynamic_cast<NOX::Nln::LinearSystem>(get_linear_system());

  if (nlnLinsysPtr.is_null())
    throw_error("reset_lin_sys_pre_post_operator", "The linear system cast failed!");

  nlnLinsysPtr->reset_pre_post_operator(linearSolverParams);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::Nln::Group::adjust_pseudo_time_step(double& delta, const double& stepSize,
    const ::NOX::Abstract::Vector& dir, const NOX::Nln::Solver::PseudoTransient& ptcsolver)
{
  const auto& dirEpetra = dynamic_cast<const NOX::Nln::Vector&>(dir);
  adjust_pseudo_time_step(delta, stepSize, dirEpetra, ptcsolver);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::Nln::Group::adjust_pseudo_time_step(double& delta, const double& stepSize,
    const NOX::Nln::Vector& dir, const NOX::Nln::Solver::PseudoTransient& ptcsolver)
{
  if (!isF() or !isJacobian())
    throw_error("AdjustPseudoTimeStep", "F and/or the jacobian are not evaluated!");

  Teuchos::RCP<NOX::Nln::LinearSystem> nlnSharedLinearSystem =
      Teuchos::rcp_dynamic_cast<NOX::Nln::LinearSystem>(linearSystemPtr);

  if (nlnSharedLinearSystem.is_null())
    throw_error("AdjustPseudoTimeStep()", "Dynamic cast of the shared linear system failed!");

  nlnSharedLinearSystem->adjust_pseudo_time_step(delta, stepSize, dir, RHSVector, ptcsolver);
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<Core::LinAlg::SparseMatrix> NOX::Nln::Group::get_contributions_from_element_level()
{
  return std::dynamic_pointer_cast<NOX::Nln::Interface::Jacobian>(userInterfacePtr)
      ->calc_jacobian_contributions_from_element_level_for_ptc();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool NOX::Nln::Group::isJacobian() const { return NOX::Nln::GroupBase::isJacobian(); }

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::Nln::Group::throw_error(
    const std::string& functionName, const std::string& errorMsg) const
{
  std::ostringstream msg;
  msg << "ERROR - NOX::Nln::Group::" << functionName << " - " << errorMsg << std::endl;

  FOUR_C_THROW("{}", msg.str());
}

FOUR_C_NAMESPACE_CLOSE

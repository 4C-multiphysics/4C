// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fsi_nox_linearsystem.hpp"

#include "4C_global_data.hpp"
#include "4C_linalg_blocksparsematrix.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_linear_solver_method_linalg.hpp"

#include <Epetra_CrsMatrix.h>
#include <Epetra_Operator.h>
#include <Epetra_RowMatrix.h>
#include <Epetra_VbrMatrix.h>

#include <vector>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
NOX::FSI::LinearSystem::LinearSystem(Teuchos::ParameterList& printParams,
    Teuchos::ParameterList& linearSolverParams,
    const std::shared_ptr<::NOX::Epetra::Interface::Jacobian>& iJac,
    const std::shared_ptr<Epetra_Operator>& J, const ::NOX::Epetra::Vector& cloneVector,
    std::shared_ptr<Core::LinAlg::Solver> solver, const Teuchos::RCP<::NOX::Epetra::Scaling> s)
    : utils_(printParams),
      jac_interface_ptr_(iJac),
      jac_type_(EpetraOperator),
      jac_ptr_(J),
      scaling_(s),
      callcount_(0),
      solver_(solver),
      timer_("", true)
{
  tmp_vector_ptr_ = std::make_shared<::NOX::Epetra::Vector>(cloneVector);

  jac_type_ = get_operator_type(*jac_ptr_);

  reset(linearSolverParams);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
NOX::FSI::LinearSystem::OperatorType NOX::FSI::LinearSystem::get_operator_type(
    const Epetra_Operator& Op)
{
  // check via dynamic cast, which type of Jacobian was broadcast
  const Epetra_Operator* testOperator = nullptr;

  testOperator = dynamic_cast<
      const Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>*>(&Op);
  if (testOperator != nullptr) return BlockSparseMatrix;

  testOperator = dynamic_cast<const Core::LinAlg::SparseMatrix*>(&Op);
  if (testOperator != nullptr) return SparseMatrix;

  testOperator = dynamic_cast<const Epetra_CrsMatrix*>(&Op);
  if (testOperator != nullptr) return EpetraCrsMatrix;

  testOperator = dynamic_cast<const Epetra_VbrMatrix*>(&Op);
  if (testOperator != nullptr) return EpetraVbrMatrix;

  testOperator = dynamic_cast<const Epetra_RowMatrix*>(&Op);
  if (testOperator != nullptr) return EpetraRowMatrix;

  return EpetraOperator;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::FSI::LinearSystem::reset(Teuchos::ParameterList& linearSolverParams)
{
  zero_initial_guess_ = linearSolverParams.get("Zero Initial Guess", false);
  manual_scaling_ = linearSolverParams.get("Compute Scaling Manually", true);
  output_solve_details_ = linearSolverParams.get("Output Solver Details", true);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool NOX::FSI::LinearSystem::applyJacobian(
    const ::NOX::Epetra::Vector& input, ::NOX::Epetra::Vector& result) const
{
  jac_ptr_->SetUseTranspose(false);
  int status = jac_ptr_->Apply(input.getEpetraVector(), result.getEpetraVector());

  return status == 0;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool NOX::FSI::LinearSystem::applyJacobianTranspose(
    const ::NOX::Epetra::Vector& input, ::NOX::Epetra::Vector& result) const
{
  jac_ptr_->SetUseTranspose(true);
  int status = jac_ptr_->Apply(input.getEpetraVector(), result.getEpetraVector());
  jac_ptr_->SetUseTranspose(false);

  return status == 0;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool NOX::FSI::LinearSystem::applyJacobianInverse(
    Teuchos::ParameterList& p, const ::NOX::Epetra::Vector& input, ::NOX::Epetra::Vector& result)
{
  // Zero out the delta X of the linear problem if requested by user.
  if (zero_initial_guess_) result.init(0.0);

  const int maxit = p.get("Max Iterations", 30);
  const double tol = p.get("Tolerance", 1.0e-10);

  std::shared_ptr<Core::LinAlg::Vector<double>> fres =
      std::make_shared<Core::LinAlg::Vector<double>>(input.getEpetraVector());
  Core::LinAlg::View disi = Core::LinAlg::View(result.getEpetraVector());

  // get the hopefully adaptive linear solver convergence tolerance
  solver_->params()
      .sublist("Belos Parameters")
      .set("Convergence Tolerance", p.get("Tolerance", 1.0e-10));

  Core::LinAlg::SolverParams solver_params;
  solver_params.refactor = true;
  solver_params.reset = callcount_ == 0;
  solver_->solve(
      jac_ptr_, Core::Utils::shared_ptr_from_ref(disi.underlying()), fres, solver_params);

  callcount_ += 1;

  // Set the output parameters in the "Output" sublist
  if (output_solve_details_)
  {
    Teuchos::ParameterList& outputList = p.sublist("Output");
    const int prevLinIters = outputList.get("Total Number of Linear Iterations", 0);
    const int curLinIters = maxit;
    const double achievedTol = tol;

    outputList.set("Number of Linear Iterations", curLinIters);
    outputList.set("Total Number of Linear Iterations", (prevLinIters + curLinIters));
    outputList.set("Achieved Tolerance", achievedTol);
  }

  return true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool NOX::FSI::LinearSystem::applyRightPreconditioning(bool useTranspose,
    Teuchos::ParameterList& params, const ::NOX::Epetra::Vector& input,
    ::NOX::Epetra::Vector& result) const
{
  if (&result != &input) result = input;

  return true;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<::NOX::Epetra::Scaling> NOX::FSI::LinearSystem::getScaling() { return scaling_; }


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::FSI::LinearSystem::resetScaling(const Teuchos::RCP<::NOX::Epetra::Scaling>& scalingObject)
{
  scaling_ = scalingObject;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool NOX::FSI::LinearSystem::computeJacobian(const ::NOX::Epetra::Vector& x)
{
  bool success = jac_interface_ptr_->computeJacobian(x.getEpetraVector(), *jac_ptr_);
  return success;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool NOX::FSI::LinearSystem::createPreconditioner(
    const ::NOX::Epetra::Vector& x, Teuchos::ParameterList& p, bool recomputeGraph) const
{
  return false;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool NOX::FSI::LinearSystem::destroyPreconditioner() const { return false; }


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool NOX::FSI::LinearSystem::recomputePreconditioner(
    const ::NOX::Epetra::Vector& x, Teuchos::ParameterList& linearSolverParams) const
{
  return false;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
NOX::FSI::LinearSystem::PreconditionerReusePolicyType
NOX::FSI::LinearSystem::getPreconditionerPolicy(bool advanceReuseCounter)
{
  return PRPT_REBUILD;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool NOX::FSI::LinearSystem::isPreconditionerConstructed() const { return false; }


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool NOX::FSI::LinearSystem::hasPreconditioner() const { return false; }


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Operator> NOX::FSI::LinearSystem::getJacobianOperator() const
{
  return Teuchos::rcpFromRef(*jac_ptr_);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Operator> NOX::FSI::LinearSystem::getJacobianOperator()
{
  return Teuchos::rcpFromRef(*jac_ptr_);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Operator> NOX::FSI::LinearSystem::getGeneratedPrecOperator() const
{
  return Teuchos::null;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Operator> NOX::FSI::LinearSystem::getGeneratedPrecOperator()
{
  return Teuchos::null;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::FSI::LinearSystem::setJacobianOperatorForSolve(
    const Teuchos::RCP<const Epetra_Operator>& solveJacOp)
{
  jac_ptr_ =
      Core::Utils::shared_ptr_from_ref(*Teuchos::rcp_const_cast<Epetra_Operator>(solveJacOp));
  jac_type_ = get_operator_type(*solveJacOp);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::FSI::LinearSystem::setPrecOperatorForSolve(
    const Teuchos::RCP<const Epetra_Operator>& solvePrecOp)
{
  throw_error("setPrecOperatorForSolve", "no preconditioner supported");
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::FSI::LinearSystem::throw_error(
    const std::string& functionName, const std::string& errorMsg) const
{
  if (utils_.isPrintType(::NOX::Utils::Error))
    utils_.out() << "NOX::FSI::LinearSystem::" << functionName << " - " << errorMsg << std::endl;

  throw "NOX Error";
}

FOUR_C_NAMESPACE_CLOSE

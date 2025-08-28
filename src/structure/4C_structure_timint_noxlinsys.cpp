// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_structure_timint_noxlinsys.hpp"

#include "4C_global_data.hpp"
#include "4C_linalg_blocksparsematrix.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_utils_enum.hpp"

#include <Epetra_CrsMatrix.h>
#include <Epetra_Operator.h>
#include <Epetra_RowMatrix.h>
#include <Epetra_VbrMatrix.h>

#include <vector>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
NOX::Solid::LinearSystem::LinearSystem(Teuchos::ParameterList& printParams,
    Teuchos::ParameterList& linearSolverParams,
    const std::shared_ptr<::NOX::Epetra::Interface::Jacobian>& iJac,
    const std::shared_ptr<Epetra_Operator>& J, const ::NOX::Epetra::Vector& cloneVector,
    std::shared_ptr<Core::LinAlg::Solver> structure_solver,
    const std::shared_ptr<NOX::Nln::Scaling> s)
    : utils_(printParams),
      jacInterfacePtr_(iJac),
      jacPtr_(J),
      scaling_(s),
      conditionNumberEstimate_(0.0),
      callcount_(0),
      structureSolver_(structure_solver),
      timer_("", true),
      timeApplyJacbianInverse_(0.0)
{
  tmpVectorPtr_ = std::make_shared<::NOX::Epetra::Vector>(cloneVector);

  // std::cout << "STRUCTURE SOLVER: " << *structureSolver_ << " " << structureSolver_ << std::endl;

  reset(linearSolverParams);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::Solid::LinearSystem::reset(Teuchos::ParameterList& linearSolverParams)
{
  zeroInitialGuess_ = linearSolverParams.get("Zero Initial Guess", false);
  manualScaling_ = linearSolverParams.get("Compute Scaling Manually", true);
  outputSolveDetails_ = linearSolverParams.get("Output Solver Details", true);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool NOX::Solid::LinearSystem::applyJacobian(
    const ::NOX::Epetra::Vector& input, ::NOX::Epetra::Vector& result) const
{
  jacPtr_->SetUseTranspose(false);
  int status = jacPtr_->Apply(input.getEpetraVector(), result.getEpetraVector());
  return (status == 0);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool NOX::Solid::LinearSystem::applyJacobianTranspose(
    const ::NOX::Epetra::Vector& input, ::NOX::Epetra::Vector& result) const
{
  jacPtr_->SetUseTranspose(true);
  int status = jacPtr_->Apply(input.getEpetraVector(), result.getEpetraVector());
  jacPtr_->SetUseTranspose(false);
  return (status == 0);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool NOX::Solid::LinearSystem::applyJacobianInverse(
    Teuchos::ParameterList& p, const ::NOX::Epetra::Vector& input, ::NOX::Epetra::Vector& result)
{
  double startTime = timer_.wallTime();

  // Zero out the delta X of the linear problem if requested by user.
  if (zeroInitialGuess_) result.init(0.0);

  int maxit = p.get("Max Iterations", 30);
  double tol = p.get("Tolerance", 1.0e-10);

  // Structure
  std::shared_ptr<Core::LinAlg::Vector<double>> fres =
      std::make_shared<Core::LinAlg::Vector<double>>(input.getEpetraVector());
  Core::LinAlg::View result_view(result.getEpetraVector());
  Core::LinAlg::SparseOperator* J = dynamic_cast<Core::LinAlg::SparseOperator*>(jacPtr_.get());

  FOUR_C_ASSERT(J, "NOX::Solid::LinearSystem works only with Core::LinAlg::SparseOperator");

  Core::LinAlg::SolverParams solver_params;
  solver_params.refactor = true;
  solver_params.reset = callcount_ == 0;
  structureSolver_->solve(Core::Utils::shared_ptr_from_ref(*J),
      Core::Utils::shared_ptr_from_ref(result_view.underlying()), fres, solver_params);
  callcount_ += 1;

  // Set the output parameters in the "Output" sublist
  if (outputSolveDetails_)
  {
    Teuchos::ParameterList& outputList = p.sublist("Output");
    int prevLinIters = outputList.get("Total Number of Linear Iterations", 0);
    int curLinIters = maxit;
    double achievedTol = tol;

    outputList.set("Number of Linear Iterations", curLinIters);
    outputList.set("Total Number of Linear Iterations", (prevLinIters + curLinIters));
    outputList.set("Achieved Tolerance", achievedTol);
  }

  double endTime = timer_.wallTime();
  timeApplyJacbianInverse_ += (endTime - startTime);

  return true;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool NOX::Solid::LinearSystem::computeJacobian(const ::NOX::Epetra::Vector& x)
{
  bool success = jacInterfacePtr_->computeJacobian(x.getEpetraVector(), *jacPtr_);
  return success;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Operator> NOX::Solid::LinearSystem::getJacobianOperator() const
{
  return Teuchos::rcpFromRef(*jacPtr_);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Operator> NOX::Solid::LinearSystem::getJacobianOperator()
{
  return Teuchos::rcpFromRef(*jacPtr_);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::Solid::LinearSystem::throw_error(
    const std::string& functionName, const std::string& errorMsg) const
{
  if (utils_.isPrintType(::NOX::Utils::Error))

  {
    utils_.out() << "NOX::Solid::LinearSystem::" << functionName << " - " << errorMsg << std::endl;
  }
  throw "NOX Error";
}

/*----------------------------------------------------------------------*/

FOUR_C_NAMESPACE_CLOSE

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fsi_nox_linearsystem.hpp"

#include "4C_global_data.hpp"
#include "4C_linalg_sparseoperator.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_solver_nonlin_nox_vector.hpp"

#include <Epetra_Operator.h>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
NOX::FSI::LinearSystem::LinearSystem(Teuchos::ParameterList& printParams,
    Teuchos::ParameterList& linearSolverParams,
    const std::shared_ptr<::NOX::Epetra::Interface::Jacobian>& iJac,
    const std::shared_ptr<Core::LinAlg::SparseOperator>& J, const NOX::Nln::Vector& cloneVector,
    std::shared_ptr<Core::LinAlg::Solver> solver, const std::shared_ptr<NOX::Nln::Scaling> s)
    : utils_(printParams),
      jac_interface_ptr_(iJac),
      // jac_ptr_(Teuchos::rcp:_(J->epetra_operator())),
      operator_(J),
      scaling_(s),
      callcount_(0),
      solver_(solver),
      timer_("", true)
{
  tmp_vector_ptr_ = std::make_shared<NOX::Nln::Vector>(cloneVector);
  // TODO: Potential error here?
  jac_ptr_ = std::shared_ptr<Epetra_Operator>(J, &J->epetra_operator());
  reset(linearSolverParams);
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
bool NOX::FSI::LinearSystem::apply_jacobian(
    const NOX::Nln::Vector& input, NOX::Nln::Vector& result) const
{
  jac_ptr_->SetUseTranspose(false);
  int status = jac_ptr_->Apply(input.get_linalg_vector(), result.get_linalg_vector());

  return status == 0;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool NOX::FSI::LinearSystem::apply_jacobian_transpose(
    const NOX::Nln::Vector& input, NOX::Nln::Vector& result) const
{
  jac_ptr_->SetUseTranspose(true);
  int status = jac_ptr_->Apply(input.get_linalg_vector(), result.get_linalg_vector());
  jac_ptr_->SetUseTranspose(false);

  return status == 0;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool NOX::FSI::LinearSystem::apply_jacobian_inverse(
    Teuchos::ParameterList& p, const NOX::Nln::Vector& input, NOX::Nln::Vector& result)
{
  // Zero out the delta X of the linear problem if requested by user.
  if (zero_initial_guess_) result.init(0.0);

  const int maxit = p.get("Max Iterations", 30);
  const double tol = p.get("Tolerance", 1.0e-10);

  // get the hopefully adaptive linear solver convergence tolerance
  solver_->params()
      .sublist("Belos Parameters")
      .set("Convergence Tolerance", p.get("Tolerance", 1.0e-10));

  Core::LinAlg::SolverParams solver_params;
  solver_params.refactor = true;
  solver_params.reset = callcount_ == 0;

  // There is a const_cast introduced - should be removed
  solver_->solve(operator_, Core::Utils::shared_ptr_from_ref(result.get_linalg_vector()),
      Core::Utils::shared_ptr_from_ref(
          const_cast<Core::LinAlg::Vector<double>&>(input.get_linalg_vector())),
      solver_params);

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
bool NOX::FSI::LinearSystem::compute_jacobian(const NOX::Nln::Vector& x)
{
  bool success = jac_interface_ptr_->computeJacobian(x.get_linalg_vector(), *jac_ptr_);
  return success;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Operator> NOX::FSI::LinearSystem::get_jacobian_operator() const
{
  return Teuchos::rcpFromRef(*jac_ptr_);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Operator> NOX::FSI::LinearSystem::get_jacobian_operator()
{
  return Teuchos::rcpFromRef(*jac_ptr_);
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

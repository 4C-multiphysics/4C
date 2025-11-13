// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fsi_nox_linearsystem_gcr.hpp"

#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_solver_nonlin_nox_vector.hpp"
#include "4C_utils_shared_ptr_from_ref.hpp"

#include <Epetra_CrsMatrix.h>
#include <Epetra_LinearProblem.h>
#include <Epetra_Operator.h>
#include <Epetra_RowMatrix.h>
#include <Epetra_VbrMatrix.h>
#include <Teuchos_ParameterList.hpp>

#include <vector>

FOUR_C_NAMESPACE_OPEN


NOX::FSI::LinearSystemGCR::LinearSystemGCR(Teuchos::ParameterList& printParams,
    Teuchos::ParameterList& linearSolverParams,
    const std::shared_ptr<NOX::Nln::Interface::RequiredBase> iReq,
    const Teuchos::RCP<::NOX::Epetra::Interface::Jacobian>& iJac,
    const Teuchos::RCP<Epetra_Operator>& jacobian, const NOX::Nln::Vector& cloneVector,
    const Teuchos::RCP<::NOX::Epetra::Scaling> s)
    : utils(printParams),
      jacInterfacePtr(iJac),
      jacType(EpetraOperator),
      jacPtr(jacobian),
      scaling(s),
      conditionNumberEstimate(0.0),
      timer("fsi_nox_LinearSystemGCR", true),
      timeApplyJacbianInverse(0.0)
{
  // Allocate solver
  tmpVectorPtr = std::make_shared<NOX::Nln::Vector>(cloneVector);

  reset(linearSolverParams);
}


void NOX::FSI::LinearSystemGCR::reset(Teuchos::ParameterList& linearSolverParams)
{
  zeroInitialGuess = linearSolverParams.get("Zero Initial Guess", false);

  manualScaling = linearSolverParams.get("Compute Scaling Manually", true);

  // Place linear solver details in the "Output" sublist of the
  // "Linear Solver" parameter list
  outputSolveDetails = linearSolverParams.get("Output Solver Details", true);

  // so we have a new time step and start anew
  u_.clear();
  c_.clear();
}


bool NOX::FSI::LinearSystemGCR::apply_jacobian(
    const NOX::Nln::Vector& input, NOX::Nln::Vector& result) const
{
  jacPtr->SetUseTranspose(false);
  int status = jacPtr->Apply(input.get_linalg_vector(), result.get_linalg_vector());
  return status == 0;
}


bool NOX::FSI::LinearSystemGCR::apply_jacobian_transpose(
    const NOX::Nln::Vector& input, NOX::Nln::Vector& result) const
{
  jacPtr->SetUseTranspose(true);
  int status = jacPtr->Apply(input.get_linalg_vector(), result.get_linalg_vector());
  jacPtr->SetUseTranspose(false);

  return status == 0;
}


bool NOX::FSI::LinearSystemGCR::apply_jacobian_inverse(
    Teuchos::ParameterList& p, const NOX::Nln::Vector& input, NOX::Nln::Vector& result)
{
  double startTime = timer.wallTime();

  // Need non-const version of the input vector
  // Epetra_LinearProblem requires non-const versions so we can perform
  // scaling of the linear problem.
  NOX::Nln::Vector& nonConstInput = const_cast<NOX::Nln::Vector&>(input);

  // Zero out the delta X of the linear problem if requested by user.
  if (zeroInitialGuess) result.init(0.0);

  // Create Epetra linear problem object for the linear solve
  Epetra_LinearProblem Problem(jacPtr.get(),
      &(result.get_linalg_vector().get_ref_of_epetra_vector()),
      &(nonConstInput.get_linalg_vector().get_ref_of_epetra_vector()));

  // ************* Begin linear system scaling *******************
  if (scaling)
  {
    if (!manualScaling) scaling->computeScaling(Problem);

    scaling->scaleLinearSystem(Problem);

    if (utils.isPrintType(::NOX::Utils::Details))
    {
      utils.out() << *scaling << std::endl;
    }
  }
  // ************* End linear system scaling *******************

  // Get linear solver convergence parameters
  int maxit = p.get("Max Iterations", 400);
  double tol = p.get("Tolerance", 1.0e-6);

  int status = -1;

  // solve using GCR
  std::string linearSolver = p.get("Solver", "GMRES");
  if (linearSolver == "GMRES")
    status = solve_gmres(input, result, maxit, tol, p.get("Size of Krylov Subspace", 300));
  else if (linearSolver == "GCR")
    status = solve_gcr(input, result, maxit, tol);
  else
  {
    utils.out() << "ERROR: NOX::FSI::LinearSystemGCR::apply_jacobian_inverse()" << std::endl
                << "\"Solver\" parameter \"" << linearSolver << "\" is invalid!" << std::endl;
    throw "NOX Error";
  }

  // Unscale the linear system
  if (scaling) scaling->unscaleLinearSystem(Problem);

  // Set the output parameters in the "Output" sublist
  if (outputSolveDetails)
  {
    Teuchos::ParameterList& outputList = p.sublist("Output");
    int prevLinIters = outputList.get("Total Number of Linear Iterations", 0);
    int curLinIters = maxit;
    double achievedTol = tol;

    outputList.set("Number of Linear Iterations", curLinIters);
    outputList.set("Total Number of Linear Iterations", (prevLinIters + curLinIters));
    outputList.set("Achieved Tolerance", achievedTol);
  }

  double endTime = timer.wallTime();
  timeApplyJacbianInverse += (endTime - startTime);

  if (status != 0) return false;

  return true;
}


int NOX::FSI::LinearSystemGCR::solve_gcr(
    const NOX::Nln::Vector& b, NOX::Nln::Vector& x, int& maxit, double& tol)
{
  NOX::Nln::Vector r(x, ::NOX::ShapeCopy);
  NOX::Nln::Vector tmp(x, ::NOX::ShapeCopy);
  if (not zeroInitialGuess)
  {
    // calculate initial residual
    if (not apply_jacobian(x, r)) throw_error("SolveGCR", "apply_jacobian() failed");
    r.update(1., b, -1.);
  }
  else
  {
    r = b;
  }

  double normb = b.norm();
  double error0 = r.norm() / normb;

  std::vector<std::shared_ptr<NOX::Nln::Vector>>& u = u_;
  std::vector<std::shared_ptr<NOX::Nln::Vector>>& c = c_;

  // reset krylov space
  u.clear();
  c.clear();

  int k = static_cast<int>(u.size());

  // use the available vectors
  for (int i = 0; i < k; ++i)
  {
    double alpha = c[i]->innerProduct(r);
    x.update(alpha, *u[i], 1.);
    r.update(-alpha, *c[i], 1.);
  }

  double error = 1. * normb;
  // while (error>=tol*error0)
  while (error / normb >= tol)
  {
    // this is GCR, not GMRESR
    u.push_back(std::make_shared<NOX::Nln::Vector>(r));
    if (not apply_jacobian(r, tmp)) throw_error("SolveGCR", "apply_jacobian() failed");
    c.push_back(std::make_shared<NOX::Nln::Vector>(tmp));

    for (int i = 0; i < k; ++i)
    {
      double beta = c.back()->innerProduct(*c[i]);

      c.back()->update(-beta, *c[i], 1.);
      u.back()->update(-beta, *u[i], 1.);
    }

    double nc = c.back()->norm();
    c.back()->scale(1. / nc);
    u.back()->scale(1. / nc);

    double alpha = c.back()->innerProduct(r);
    x.update(alpha, *u.back(), 1.);
    r.update(-alpha, *c.back(), 1.);
    error = r.norm();

    k += 1;

    utils.out() << "gcr |r|=" << error << " |r0|=" << error0 << " |dx|=" << u.back()->norm() * alpha
                << " |b|=" << normb << " tol=" << tol << std::endl;
  }

  maxit = k;
  tol = error;

  return 0;
}


int NOX::FSI::LinearSystemGCR::solve_gmres(
    const NOX::Nln::Vector& b, NOX::Nln::Vector& x, int& max_iter, double& tol, int m)
{
  double resid = 0;
  Core::LinAlg::SerialDenseVector s(m + 1, true);
  Core::LinAlg::SerialDenseVector cs(m + 1, true);
  Core::LinAlg::SerialDenseVector sn(m + 1, true);
  Core::LinAlg::SerialDenseMatrix H(m + 1, m, true);

  NOX::Nln::Vector r(x, ::NOX::ShapeCopy);
  NOX::Nln::Vector w(x, ::NOX::ShapeCopy);
  if (not zeroInitialGuess)
  {
    // calculate initial residual
    if (not apply_jacobian(x, r)) throw_error("SolveGMRES", "apply_jacobian() failed");
    r.update(1., b, -1.);
  }
  else
  {
    r = b;
  }

  double normb = b.norm();
  double beta = r.norm();

  if (normb == 0.0) normb = 1;

  if ((resid = beta / normb) <= tol)
  {
    tol = resid;
    max_iter = 0;
    return 0;
  }

  std::vector<std::shared_ptr<NOX::Nln::Vector>> v;
  v.reserve(m + 1);

  int j = 1;
  while (j <= max_iter)
  {
    v.clear();
    v.push_back(std::make_shared<NOX::Nln::Vector>(r, ::NOX::ShapeCopy));
    v[0]->update(1. / beta, r, 0.);
    s.putScalar(0.0);
    s(0) = beta;

    for (int i = 0; i < m and j <= max_iter; i++, j++)
    {
      Teuchos::Time t("GMRES", true);
      // w = M.solve(A * v[i]);
      if (not apply_jacobian(*v[i], w)) throw_error("SolveGMRES", "apply_jacobian() failed");
      for (int k = 0; k <= i; k++)
      {
        H(k, i) = w.innerProduct(*v[k]);
        w.update(H(k, i), *v[k], -1.);
      }
      H(i + 1, i) = w.norm();
      v.push_back(std::make_shared<NOX::Nln::Vector>(w));
      v.back()->scale(1.0 / H(i + 1, i));

      for (int k = 0; k < i; k++) apply_plane_rotation(H(k, i), H(k + 1, i), cs(k), sn(k));

      generate_plane_rotation(H(i, i), H(i + 1, i), cs(i), sn(i));
      apply_plane_rotation(H(i, i), H(i + 1, i), cs(i), sn(i));
      apply_plane_rotation(s(i), s(i + 1), cs(i), sn(i));

      utils.out() << "gmres |r|=" << std::scientific << fabs(s(i + 1))
                  << "   |b|=" << std::scientific << normb << "   tol=" << std::scientific << tol
                  << "   time=" << std::scientific << t.totalElapsedTime(true) << std::endl;

      if ((resid = fabs(s(i + 1)) / normb) < tol)
      {
        // Update(x, i, H, s, v);
        Core::LinAlg::SerialDenseVector y(s);

        // Backsolve:
        for (int l = i; l >= 0; l--)
        {
          y(l) /= H(l, l);
          for (int k = l - 1; k >= 0; k--) y(k) -= H(k, l) * y(l);
        }

        for (int k = 0; k <= i; k++) x.update(y(k), *v[k], 1.);

        tol = resid;
        max_iter = j;
        return 0;
      }
    }

    // Update(x, m - 1, H, s, v);
    Core::LinAlg::SerialDenseVector y(s);

    // Backsolve:
    for (int i = m - 1; i >= 0; i--)
    {
      y(i) /= H(i, i);
      for (int k = i - 1; k >= 0; k--) y(k) -= H(k, i) * y(i);
    }

    for (int k = 0; k <= m - 1; k++) x.update(y(k), *v[k], 1.);

    // Isn't there a cheaper way to calculate that?
    if (not apply_jacobian(x, r)) throw_error("SolveGMRES", "apply_jacobian() failed");
    r.update(1., b, -1.);
    beta = r.norm();
    if ((resid = beta / normb) < tol)
    {
      tol = resid;
      max_iter = j;
      return 0;
    }
  }

  tol = resid;
  return 1;
}


void NOX::FSI::LinearSystemGCR::generate_plane_rotation(
    double& dx, double& dy, double& cs, double& sn)
{
  if (dy == 0.0)
  {
    cs = 1.0;
    sn = 0.0;
  }
  else if (fabs(dy) > fabs(dx))
  {
    double temp = dx / dy;
    sn = 1.0 / sqrt(1.0 + temp * temp);
    cs = temp * sn;
  }
  else
  {
    double temp = dy / dx;
    cs = 1.0 / sqrt(1.0 + temp * temp);
    sn = temp * cs;
  }
}


void NOX::FSI::LinearSystemGCR::apply_plane_rotation(double& dx, double& dy, double& cs, double& sn)
{
  double temp = cs * dx + sn * dy;
  dy = -sn * dx + cs * dy;
  dx = temp;
}


bool NOX::FSI::LinearSystemGCR::compute_jacobian(const NOX::Nln::Vector& x)
{
  bool success = jacInterfacePtr->computeJacobian(x.get_linalg_vector(), *jacPtr);
  return success;
}


Teuchos::RCP<const Epetra_Operator> NOX::FSI::LinearSystemGCR::get_jacobian_operator() const
{
  return Teuchos::rcpFromRef(*jacPtr);
}


Teuchos::RCP<Epetra_Operator> NOX::FSI::LinearSystemGCR::get_jacobian_operator()
{
  return Teuchos::rcpFromRef(*jacPtr);
}


void NOX::FSI::LinearSystemGCR::throw_error(
    const std::string& functionName, const std::string& errorMsg) const
{
  if (utils.isPrintType(::NOX::Utils::Error))
  {
    utils.out() << "NOX::FSI::LinearSystemGCR::" << functionName << " - " << errorMsg << std::endl;
  }
  throw "NOX Error";
}


FOUR_C_NAMESPACE_CLOSE

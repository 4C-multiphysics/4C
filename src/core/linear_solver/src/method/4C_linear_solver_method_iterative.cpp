// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_linear_solver_method_iterative.hpp"

#include "4C_linear_solver_amgnxn_preconditioner.hpp"
#include "4C_linear_solver_preconditioner_ifpack.hpp"
#include "4C_linear_solver_preconditioner_krylovprojection.hpp"
#include "4C_linear_solver_preconditioner_muelu.hpp"
#include "4C_linear_solver_preconditioner_teko.hpp"
#include "4C_utils_exceptions.hpp"

#include <BelosBiCGStabSolMgr.hpp>
#include <BelosBlockCGSolMgr.hpp>
#include <BelosBlockGmresSolMgr.hpp>
#include <BelosEpetraAdapter.hpp>
#include <BelosLinearProblem.hpp>
#include <BelosPseudoBlockCGSolMgr.hpp>
#include <BelosPseudoBlockGmresSolMgr.hpp>
#include <Epetra_CrsMatrix.h>
#include <Epetra_Map.h>
#include <Teuchos_RCPStdSharedPtrConversions.hpp>
#include <Teuchos_TimeMonitor.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>

FOUR_C_NAMESPACE_OPEN

using BelosVectorType = Epetra_MultiVector;

//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
template <class MatrixType, class VectorType>
Core::LinearSolver::IterativeSolver<MatrixType, VectorType>::IterativeSolver(
    MPI_Comm comm, Teuchos::ParameterList& params)
    : comm_(comm), params_(params)
{
}

//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
template <class MatrixType, class VectorType>
void Core::LinearSolver::IterativeSolver<MatrixType, VectorType>::setup(
    std::shared_ptr<MatrixType> A, std::shared_ptr<VectorType> x, std::shared_ptr<VectorType> b,
    const bool refactor, const bool reset, std::shared_ptr<Core::LinAlg::KrylovProjector> projector)
{
  if (!params().isSublist("Belos Parameters")) FOUR_C_THROW("Do not have belos parameter list");
  Teuchos::ParameterList& belist = params().sublist("Belos Parameters");

  const int reuse = belist.get("reuse", 0);
  const bool create = !allow_reuse_preconditioner(reuse, reset);
  if (create)
  {
    ncall_ = 0;
    preconditioner_ = create_preconditioner(belist, projector);
  }

  a_ = A;
  x_ = x;
  b_ = b;

  preconditioner_->setup(create, a_.get(), x_.get(), b_.get());
}

//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
template <class MatrixType, class VectorType>
int Core::LinearSolver::IterativeSolver<MatrixType, VectorType>::solve()
{
  Teuchos::ParameterList& belist = params().sublist("Belos Parameters");

  auto problem = Teuchos::make_rcp<Belos::LinearProblem<double, BelosVectorType, MatrixType>>(
      Teuchos::rcp(a_), Teuchos::rcp(x_->get_ptr_of_Epetra_MultiVector()),
      Teuchos::rcp(b_->get_ptr_of_Epetra_MultiVector()));

  if (preconditioner_ != nullptr)
  {
    auto belosPrec =
        Teuchos::make_rcp<Belos::EpetraPrecOp>(Teuchos::rcp(preconditioner_->prec_operator()));
    problem->setRightPrec(belosPrec);
  }

  const bool set = problem->setProblem();
  if (set == false)
    FOUR_C_THROW("Core::LinearSolver::BelosSolver: Iterative solver failed to set up correctly.");

  std::shared_ptr<Belos::SolverManager<double, BelosVectorType, MatrixType>> newSolver;

  if (belist.isParameter("SOLVER_XML_FILE"))
  {
    const std::string xmlFileName = belist.get<std::string>("SOLVER_XML_FILE");
    Teuchos::ParameterList belosParams;
    Teuchos::updateParametersFromXmlFileAndBroadcast(xmlFileName,
        Teuchos::Ptr<Teuchos::ParameterList>(&belosParams),
        *Xpetra::toXpetra(Core::Communication::as_epetra_comm(comm_)));

    if (belosParams.isSublist("GMRES"))
    {
      auto belosSolverList = rcpFromRef(belosParams.sublist("GMRES"));
      if (belist.isParameter("Convergence Tolerance"))
      {
        belosSolverList->set("Convergence Tolerance", belist.get<double>("Convergence Tolerance"));
      }

      newSolver =
          std::make_shared<Belos::PseudoBlockGmresSolMgr<double, BelosVectorType, MatrixType>>(
              problem, belosSolverList);
    }
    else if (belosParams.isSublist("CG"))
    {
      auto belosSolverList = rcpFromRef(belosParams.sublist("CG"));
      if (belist.isParameter("Convergence Tolerance"))
      {
        belosSolverList->set("Convergence Tolerance", belist.get<double>("Convergence Tolerance"));
      }

      newSolver = std::make_shared<Belos::PseudoBlockCGSolMgr<double, BelosVectorType, MatrixType>>(
          problem, belosSolverList);
    }
    else if (belosParams.isSublist("BiCGSTAB"))
    {
      auto belosSolverList = rcpFromRef(belosParams.sublist("BiCGSTAB"));
      if (belist.isParameter("Convergence Tolerance"))
      {
        belosSolverList->set("Convergence Tolerance", belist.get<double>("Convergence Tolerance"));
      }

      newSolver = std::make_shared<Belos::BiCGStabSolMgr<double, BelosVectorType, MatrixType>>(
          problem, belosSolverList);
    }
    else
      FOUR_C_THROW("Core::LinearSolver::BelosSolver: Unknown iterative solver solver type chosen.");
  }
  else
  {
    if (Core::Communication::my_mpi_rank(comm_) == 0)
      std::cout << "WARNING: The linear solver input parameters from the .dat file will be "
                   "depreciated soon. Switch to an appropriate xml-file version."
                << std::endl;

    std::string solverType = belist.get<std::string>("Solver Type");
    if (solverType == "GMRES")
      newSolver = std::make_shared<Belos::BlockGmresSolMgr<double, BelosVectorType, MatrixType>>(
          problem, Teuchos::rcpFromRef(belist));
    else if (solverType == "CG")
      newSolver = std::make_shared<Belos::BlockCGSolMgr<double, BelosVectorType, MatrixType>>(
          problem, Teuchos::rcpFromRef(belist));
    else if (solverType == "BiCGSTAB")
      newSolver = std::make_shared<Belos::BiCGStabSolMgr<double, BelosVectorType, MatrixType>>(
          problem, Teuchos::rcpFromRef(belist));
    else
      FOUR_C_THROW("Core::LinearSolver::BelosSolver: Unknown iterative solver solver type chosen.");
  }

  Belos::ReturnType ret = newSolver->solve();

  int my_error = 0;
  if (ret != Belos::Converged) my_error = 1;
  int glob_error = 0;
  Core::Communication::sum_all(&my_error, &glob_error, 1, comm_);

  if (glob_error > 0 and Core::Communication::my_mpi_rank(this->comm_) == 0)
    std::cout << std::endl
              << "Core::LinearSolver::BelosSolver: WARNING: Iterative solver did not converge!"
              << std::endl;

  numiters_ = newSolver->getNumIters();

  ncall_ += 1;

  return 0;
}

//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
template <class MatrixType, class VectorType>
bool Core::LinearSolver::IterativeSolver<MatrixType, VectorType>::allow_reuse_preconditioner(
    const int reuse, const bool reset)
{
  // first, check some parameters with information that has to be updated
  Teuchos::ParameterList& linSysParams = params().sublist("Belos Parameters");

  bool bAllowReuse = linSysParams.get<bool>("reuse preconditioner", true);

  const bool create = reset or not ncall() or not reuse or (ncall() % reuse) == 0;
  if (create) bAllowReuse = false;

  // here, each processor has its own local decision made
  // bAllowReuse = true -> preconditioner can be reused
  // bAllowReuse = false -> preconditioner has to be recomputed
  // If one or more processors decide that the preconditioner has to be recomputed
  // all of the processors have to recompute it

  // synchronize results of all processors
  // all processors have to do the same (either recompute preconditioner or allow reusing it)
  int nProc = Core::Communication::num_mpi_ranks(comm_);
  int lAllowReuse = bAllowReuse == true ? 1 : 0;
  int gAllowReuse = 0;
  Core::Communication::sum_all(&lAllowReuse, &gAllowReuse, 1, comm_);

  return gAllowReuse == nProc;
}


//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
template <class MatrixType, class VectorType>
std::shared_ptr<Core::LinearSolver::PreconditionerTypeBase>
Core::LinearSolver::IterativeSolver<MatrixType, VectorType>::create_preconditioner(
    Teuchos::ParameterList& solverlist, std::shared_ptr<Core::LinAlg::KrylovProjector> projector)
{
  TEUCHOS_FUNC_TIME_MONITOR("Core::LinAlg::Solver:  1.1)   create_preconditioner");

  std::shared_ptr<Core::LinearSolver::PreconditionerTypeBase> preconditioner;

  if (params().isSublist("IFPACK Parameters"))
  {
    preconditioner = std::make_shared<Core::LinearSolver::IFPACKPreconditioner>(
        params().sublist("IFPACK Parameters"), solverlist);
  }
  else if (params().isSublist("MueLu Parameters"))
  {
    preconditioner = std::make_shared<Core::LinearSolver::MueLuPreconditioner>(params());
  }
  else if (params().isSublist("Teko Parameters"))
  {
    preconditioner = std::make_shared<Core::LinearSolver::TekoPreconditioner>(params());
  }
  else if (params().isSublist("AMGnxn Parameters"))
  {
    preconditioner = std::make_shared<Core::LinearSolver::AmGnxnPreconditioner>(params());
  }
  else
    FOUR_C_THROW("Unknown preconditioner chosen for iterative linear solver.");

  if (projector != nullptr)
  {
    preconditioner = std::make_shared<Core::LinearSolver::KrylovProjectionPreconditioner>(
        preconditioner, projector);
  }

  return preconditioner;
}

//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
// explicit initialization
template class Core::LinearSolver::IterativeSolver<Epetra_Operator,
    Core::LinAlg::MultiVector<double>>;

FOUR_C_NAMESPACE_CLOSE

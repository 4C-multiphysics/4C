// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_linear_solver_method_iterative.hpp"

#include "4C_linalg_map.hpp"
#include "4C_linear_solver_amgnxn_preconditioner.hpp"
#include "4C_linear_solver_method_projector.hpp"
#include "4C_linear_solver_preconditioner_ifpack.hpp"
#include "4C_linear_solver_preconditioner_muelu.hpp"
#include "4C_linear_solver_preconditioner_projection.hpp"
#include "4C_linear_solver_preconditioner_teko.hpp"
#include "4C_utils_exceptions.hpp"

#include <BelosBiCGStabSolMgr.hpp>
#include <BelosBlockCGSolMgr.hpp>
#include <BelosBlockGmresSolMgr.hpp>
#include <BelosEpetraAdapter.hpp>
#include <BelosLinearProblem.hpp>
#include <BelosPseudoBlockCGSolMgr.hpp>
#include <BelosPseudoBlockGmresSolMgr.hpp>
#include <Teuchos_RCPDecl.hpp>
#include <Teuchos_RCPStdSharedPtrConversions.hpp>
#include <Teuchos_TimeMonitor.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>

FOUR_C_NAMESPACE_OPEN

using BelosVectorType = Epetra_MultiVector;
using BelosMatrixType = Epetra_Operator;

//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
Core::LinearSolver::IterativeSolver::IterativeSolver(MPI_Comm comm, Teuchos::ParameterList& params)
    : comm_(comm), params_(params)
{
}

//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
void Core::LinearSolver::IterativeSolver::setup(std::shared_ptr<Core::LinAlg::SparseOperator> A,
    std::shared_ptr<Core::LinAlg::MultiVector<double>> x,
    std::shared_ptr<Core::LinAlg::MultiVector<double>> b, const bool refactor, const bool reset,
    std::shared_ptr<Core::LinAlg::LinearSystemProjector> projector)
{
  if (!params().isSublist("Belos Parameters")) FOUR_C_THROW("Do not have belos parameter list");
  Teuchos::ParameterList& belist = params().sublist("Belos Parameters");

  if (Core::Communication::my_mpi_rank(comm_) == 0)
    std::cout << "*******************************************************" << std::endl;

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

  if (create)
  {
    if (Core::Communication::my_mpi_rank(comm_) == 0)
    {
      std::cout << "Compute preconditioner " << std::endl;
      std::cout << "*******************************************************" << std::endl;
    }

    preconditioner_->setup(*a_, *x_, *b_);
  }
  else
  {
    if (Core::Communication::my_mpi_rank(comm_) == 0)
    {
      std::cout << "Reuse preconditioner (reused for " << ncall() << " steps)" << std::endl;
      std::cout << "*******************************************************" << std::endl;
    }
  }
}

//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
int Core::LinearSolver::IterativeSolver::solve()
{
  Teuchos::ParameterList& belist = params().sublist("Belos Parameters");

  auto problem = Teuchos::make_rcp<Belos::LinearProblem<double, BelosVectorType, BelosMatrixType>>(
      Teuchos::rcpFromRef(a_->epetra_operator()),
      Teuchos::rcpFromRef(x_->get_epetra_multi_vector()),
      Teuchos::rcpFromRef(b_->get_epetra_multi_vector()));

  if (preconditioner_ != nullptr)
  {
    auto belosPrec =
        Teuchos::make_rcp<Belos::EpetraPrecOp>(Teuchos::rcp(preconditioner_->prec_operator()));
    problem->setRightPrec(belosPrec);
  }

  const bool set = problem->setProblem();
  if (set == false)
    FOUR_C_THROW("Core::LinearSolver::BelosSolver: Iterative solver failed to set up correctly.");

  std::shared_ptr<Belos::SolverManager<double, BelosVectorType, BelosMatrixType>> newSolver;

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
          std::make_shared<Belos::PseudoBlockGmresSolMgr<double, BelosVectorType, BelosMatrixType>>(
              problem, belosSolverList);
    }
    else if (belosParams.isSublist("CG"))
    {
      auto belosSolverList = rcpFromRef(belosParams.sublist("CG"));
      if (belist.isParameter("Convergence Tolerance"))
      {
        belosSolverList->set("Convergence Tolerance", belist.get<double>("Convergence Tolerance"));
      }

      newSolver =
          std::make_shared<Belos::PseudoBlockCGSolMgr<double, BelosVectorType, BelosMatrixType>>(
              problem, belosSolverList);
    }
    else if (belosParams.isSublist("BiCGSTAB"))
    {
      auto belosSolverList = rcpFromRef(belosParams.sublist("BiCGSTAB"));
      if (belist.isParameter("Convergence Tolerance"))
      {
        belosSolverList->set("Convergence Tolerance", belist.get<double>("Convergence Tolerance"));
      }

      newSolver = std::make_shared<Belos::BiCGStabSolMgr<double, BelosVectorType, BelosMatrixType>>(
          problem, belosSolverList);
    }
    else
      FOUR_C_THROW("Core::LinearSolver::BelosSolver: Unknown iterative solver solver type chosen.");
  }
  else
  {
    if (Core::Communication::my_mpi_rank(comm_) == 0)
      std::cout << "WARNING: The linear solver input parameters from the input file will be "
                   "removed soon. Switch to an appropriate xml-file version."
                << std::endl;

    std::string solverType = belist.get<std::string>("Solver Type");
    if (solverType == "GMRES")
      newSolver =
          std::make_shared<Belos::BlockGmresSolMgr<double, BelosVectorType, BelosMatrixType>>(
              problem, Teuchos::rcpFromRef(belist));
    else if (solverType == "CG")
      newSolver = std::make_shared<Belos::BlockCGSolMgr<double, BelosVectorType, BelosMatrixType>>(
          problem, Teuchos::rcpFromRef(belist));
    else if (solverType == "BiCGSTAB")
      newSolver = std::make_shared<Belos::BiCGStabSolMgr<double, BelosVectorType, BelosMatrixType>>(
          problem, Teuchos::rcpFromRef(belist));
    else
      FOUR_C_THROW("Core::LinearSolver::BelosSolver: Unknown iterative solver solver type chosen.");
  }

  Belos::ReturnType ret = newSolver->solve();

  numiters_ = newSolver->getNumIters();
  ncall_ += 1;

  // Check and communicate failed convergence to user
  {
    int my_error = 0;
    if (ret != Belos::Converged) my_error = 1;
    int glob_error = 0;
    glob_error = Core::Communication::sum_all(my_error, comm_);

    if (glob_error > 0)
    {
      if (belist.get<bool>("THROW_IF_UNCONVERGED"))
      {
        FOUR_C_THROW("Core::LinearSolver::BelosSolver: Iterative solver did not converge.");
      }
      else
      {
        if (Core::Communication::my_mpi_rank(comm_) == 0)
        {
          std::cout
              << "Core::LinearSolver::BelosSolver: WARNING: Iterative solver did not converge."
              << std::endl;
        }
      }
    }
  }

  return 0;
}

//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
bool Core::LinearSolver::IterativeSolver::allow_reuse_preconditioner(
    const int reuse, const bool reset)
{
  // first, check some parameters with information that has to be updated
  Teuchos::ParameterList& linSysParams = params().sublist("Belos Parameters");

  bool bAllowReuse = linSysParams.get<bool>("reuse preconditioner", true);
  int max_stall_iter = linSysParams.get<int>("max linear iterations for stall");

  // 1. check if we are allowed to reuse the preconditioner over several nonlinear solves
  if (not ncall() or not reuse or ncall() % reuse == 0)
  {
    if (Core::Communication::my_mpi_rank(comm_) == 0)
      std::cout << "Recomputation due to reaching " << ncall() << " nonlinear steps." << std::endl;

    bAllowReuse = false;
  }

  // 2. check if the number of iterations exceeds the given one for stalling of the solver
  if (bAllowReuse and get_num_iters() > max_stall_iter)
  {
    if (Core::Communication::my_mpi_rank(comm_) == 0)
      std::cout << "Recomputation due to linear solver stalling." << std::endl;

    bAllowReuse = false;
  }

  // 3. check if there's an external reset
  if (reset)
  {
    if (Core::Communication::my_mpi_rank(comm_) == 0)
      std::cout << "Recomputation due to reset." << std::endl;

    bAllowReuse = false;
  }

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
  gAllowReuse = Core::Communication::sum_all(lAllowReuse, comm_);

  return gAllowReuse == nProc;
}


//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
std::shared_ptr<Core::LinearSolver::PreconditionerTypeBase>
Core::LinearSolver::IterativeSolver::create_preconditioner(Teuchos::ParameterList& solverlist,
    std::shared_ptr<Core::LinAlg::LinearSystemProjector> projector)
{
  TEUCHOS_FUNC_TIME_MONITOR("Core::LinAlg::Solver:  1.1)   create_preconditioner");

  std::shared_ptr<Core::LinearSolver::PreconditionerTypeBase> preconditioner;

  if (params().isSublist("IFPACK Parameters"))
  {
    preconditioner = std::make_shared<Core::LinearSolver::IFPACKPreconditioner>(params());
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
    preconditioner =
        std::make_shared<Core::LinearSolver::ProjectionPreconditioner>(preconditioner, projector);
  }

  return preconditioner;
}

FOUR_C_NAMESPACE_CLOSE

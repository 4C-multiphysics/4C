// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_linear_solver_method_linalg.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_io_pstream.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linear_solver_method_direct.hpp"
#include "4C_linear_solver_method_iterative.hpp"
#include "4C_utils_parameter_list.hpp"

#include <BelosTypes.hpp>  // for Belos verbosity codes
#include <Epetra_LinearProblem.h>
#include <Epetra_MpiComm.h>
#include <ml_MultiLevelPreconditioner.h>  // includes for ML parameter list validation
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::LinAlg::Solver::Solver(const Teuchos::ParameterList &inparams, MPI_Comm comm,
    const std::function<const Teuchos::ParameterList &(int)> &get_solver_params,
    Core::IO::Verbositylevel verbosity, const bool translate_params_to_belos)
    : comm_(comm), params_(std::make_shared<Teuchos::ParameterList>())
{
  if (translate_params_to_belos)
    *params_ = translate_solver_parameters(inparams, get_solver_params, verbosity);
  else
    *params_ = inparams;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::LinAlg::Solver::~Solver() { reset(); }

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::Solver::reset() { solver_ = nullptr; }

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
int Core::LinAlg::Solver::get_num_iters() const { return solver_->get_num_iters(); }

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::Solver::adapt_tolerance(
    const double desirednlnres, const double currentnlnres, const double better)
{
  if (!params().isSublist("Belos Parameters")) FOUR_C_THROW("Adaptive tolerance only for Belos.");

  Teuchos::ParameterList &solver_params = params().sublist("Belos Parameters");

  if (!solver_params.isParameter("Convergence Tolerance"))
    FOUR_C_THROW("No iterative solver tolerance in ParameterList");

  const bool do_output = solver_params.get<int>("Output Frequency", 1) and
                         !Core::Communication::my_mpi_rank(get_comm());

  const std::string conv_test_strategy = solver_params.get<std::string>(
      "Implicit Residual Scaling", Belos::convertScaleTypeToString(Belos::ScaleType::None));

  if (conv_test_strategy != Belos::convertScaleTypeToString(Belos::ScaleType::NormOfInitRes))
  {
    FOUR_C_THROW(
        "You are using an adaptive tolerance for the linear solver. Therefore, the iterative "
        "solver needs to work with a relative residual norm. This can be achieved by setting "
        "'AZCONV' to 'AZ_r0' in the input file.");
  }

  // save original value of convergence
  const bool have_saved_value = solver_params.isParameter("Convergence Tolerance Saved");
  if (!have_saved_value)
  {
    solver_params.set<double>(
        "Convergence Tolerance Saved", solver_params.get<double>("Convergence Tolerance"));
  }

  const double input_tolerance = solver_params.get<double>("Convergence Tolerance Saved");

  if (do_output)
    std::cout << "                --- Solver input relative tolerance " << input_tolerance << "\n";
  if (currentnlnres * input_tolerance < desirednlnres)
  {
    double adapted_tolerance = desirednlnres * better / currentnlnres;
    if (adapted_tolerance > 1.0)
    {
      adapted_tolerance = 1.0;
      if (do_output)
      {
        std::cout << "WARNING:  Computed adapted relative tolerance bigger than 1\n";
        std::cout << "          Value constrained to 1, but consider adapting Parameter "
                     "ADAPTCONV_BETTER\n";
      }
    }
    if (adapted_tolerance < input_tolerance) adapted_tolerance = input_tolerance;
    if (do_output && adapted_tolerance > input_tolerance)
    {
      std::cout << "                *** Solver adapted relative tolerance " << adapted_tolerance
                << "\n";
    }

    solver_params.set<double>("Convergence Tolerance", adapted_tolerance);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::Solver::set_tolerance(const double tolerance)
{
  if (!params().isSublist("Belos Parameters"))
    FOUR_C_THROW("Set tolerance of linear solver only for Belos solver.");

  Teuchos::ParameterList &solver_params = params().sublist("Belos Parameters");

  const bool have_saved_value = solver_params.isParameter("Convergence Tolerance Saved");
  if (!have_saved_value)
  {
    solver_params.set<double>(
        "Convergence Tolerance Saved", solver_params.get<double>("Convergence Tolerance", 1.0e-8));
  }

  solver_params.set<double>("Convergence Tolerance", tolerance);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::Solver::reset_tolerance()
{
  if (!params().isSublist("Belos Parameters")) return;

  Teuchos::ParameterList &solver_params = params().sublist("Belos Parameters");

  const bool have_saved_value = solver_params.isParameter("Convergence Tolerance Saved");
  if (!have_saved_value) return;

  solver_params.set<double>(
      "Convergence Tolerance", solver_params.get<double>("Convergence Tolerance Saved"));
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::Solver::setup(std::shared_ptr<Epetra_Operator> matrix,
    std::shared_ptr<Core::LinAlg::MultiVector<double>> x,
    std::shared_ptr<Core::LinAlg::MultiVector<double>> b, const SolverParams &params)
{
  TEUCHOS_FUNC_TIME_MONITOR("Core::LinAlg::Solver:  1)   Setup");

  FOUR_C_ASSERT(!(params.lin_tol_better > -1.0 and params.tolerance > 0.0),
      "Do not set tolerance and adaptive tolerance to the linear solver.");

  if (params.lin_tol_better > -1.0)
  {
    adapt_tolerance(params.nonlin_tolerance, params.nonlin_residual, params.lin_tol_better);
  }

  if (params.tolerance > 0.0)
  {
    set_tolerance(params.tolerance);
  }

  // reset data flags on demand
  bool refactor = params.refactor;
  if (params.reset)
  {
    reset();
    refactor = true;
  }

  if (solver_ == nullptr)
  {
    // decide what solver to use
    std::string solvertype = Solver::params().get("solver", "none");

    if ("belos" == solvertype)
    {
      solver_ = std::make_shared<
          Core::LinearSolver::IterativeSolver<Epetra_Operator, Core::LinAlg::MultiVector<double>>>(
          comm_, Solver::params());
    }
    else if ("umfpack" == solvertype or "superlu" == solvertype)
    {
      solver_ = std::make_shared<
          Core::LinearSolver::DirectSolver<Epetra_Operator, Core::LinAlg::MultiVector<double>>>(
          solvertype);
    }
    else
      FOUR_C_THROW("Unknown type of solver");
  }

  solver_->setup(matrix, x, b, refactor, params.reset, params.projector);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/

int Core::LinAlg::Solver::solve_with_multi_vector(std::shared_ptr<Epetra_Operator> matrix,
    std::shared_ptr<Core::LinAlg::MultiVector<double>> x,
    std::shared_ptr<Core::LinAlg::MultiVector<double>> b, const Core::LinAlg::SolverParams &params)
{
  setup(matrix, x, b, params);

  int error_value = 0;
  {
    TEUCHOS_FUNC_TIME_MONITOR("Core::LinAlg::Solver:  2)   Solve");
    error_value = solver_->solve();
  }

  return error_value;
}

int Core::LinAlg::Solver::solve(std::shared_ptr<Epetra_Operator> matrix,
    std::shared_ptr<Core::LinAlg::Vector<double>> x,
    std::shared_ptr<Core::LinAlg::Vector<double>> b, const SolverParams &params)
{
  setup(matrix, x->get_ptr_of_MultiVector(), b->get_ptr_of_MultiVector(), params);

  int error_value = 0;
  {
    TEUCHOS_FUNC_TIME_MONITOR("Core::LinAlg::Solver:  2)   Solve");
    error_value = solver_->solve();
  }

  return error_value;
}

/*------------------------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------------------------*/
Teuchos::ParameterList translate_four_c_to_ifpack(const Teuchos::ParameterList &inparams)
{
  Teuchos::ParameterList ifpacklist;

  ifpacklist.set("fact: level-of-fill", inparams.get<int>("IFPACKGFILL"));
  ifpacklist.set("partitioner: overlap", inparams.get<int>("IFPACKOVERLAP"));
  ifpacklist.set("schwarz: combine mode",
      inparams.get<std::string>("IFPACKCOMBINE"));    // can be "Zero", "Add", "Insert"
  ifpacklist.set("schwarz: reordering type", "rcm");  // "rcm" or "metis" or "amd"

  return ifpacklist;
}

/*------------------------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------------------------*/
Teuchos::ParameterList translate_four_c_to_muelu(
    const Teuchos::ParameterList &inparams, Teuchos::ParameterList *azlist)
{
  Teuchos::ParameterList muelulist;

  std::string xmlfile = inparams.get<std::string>("MUELU_XML_FILE");
  if (xmlfile != "none") muelulist.set("MUELU_XML_FILE", xmlfile);

  return muelulist;
}

/*------------------------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------------------------*/
Teuchos::ParameterList translate_four_c_to_teko(
    const Teuchos::ParameterList &inparams, Teuchos::ParameterList *azlist)
{
  Teuchos::ParameterList tekolist;

  std::string xmlfile = inparams.get<std::string>("TEKO_XML_FILE");
  if (xmlfile != "none") tekolist.set("TEKO_XML_FILE", xmlfile);

  return tekolist;
}

/*------------------------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------------------------*/
Teuchos::ParameterList translate_four_c_to_belos(const Teuchos::ParameterList &inparams,
    const std::function<const Teuchos::ParameterList &(int)> &get_solver_params,
    Core::IO::Verbositylevel verbosity)
{
  Teuchos::ParameterList outparams;
  outparams.set("solver", "belos");
  Teuchos::ParameterList &beloslist = outparams.sublist("Belos Parameters");

  beloslist.set("reuse", inparams.get<int>("AZREUSE"));
  beloslist.set("ncall", 0);

  // try to get an xml file if possible
  std::string xmlfile = inparams.get<std::string>("SOLVER_XML_FILE");
  if (xmlfile != "none")
  {
    beloslist.set("SOLVER_XML_FILE", xmlfile);
  }
  else
  {
    switch (verbosity)
    {
      case Core::IO::minimal:
        beloslist.set("Output Style", Belos::OutputType::Brief);
        beloslist.set("Verbosity", Belos::MsgType::Warnings);
        break;
      case Core::IO::standard:
        beloslist.set("Output Style", Belos::OutputType::Brief);
        beloslist.set("Verbosity", Belos::MsgType::Warnings + Belos::MsgType::StatusTestDetails);
        break;
      case Core::IO::verbose:
        beloslist.set("Output Style", Belos::OutputType::General);
        beloslist.set("Verbosity", Belos::MsgType::Warnings + Belos::MsgType::StatusTestDetails +
                                       Belos::MsgType::FinalSummary);
        break;
      case Core::IO::debug:
        beloslist.set("Output Style", Belos::OutputType::General);
        beloslist.set("Verbosity", Belos::MsgType::Debug);
      default:
        break;
    }
    beloslist.set("Output Frequency", inparams.get<int>("AZOUTPUT"));

    // set tolerances and iterations
    beloslist.set("Maximum Iterations", inparams.get<int>("AZITER"));
    beloslist.set("Convergence Tolerance", inparams.get<double>("AZTOL"));
    beloslist.set("Implicit Residual Scaling",
        Belos::convertScaleTypeToString(
            Teuchos::getIntegralValue<Belos::ScaleType>(inparams, "AZCONV")));

    // set type of solver
    switch (Teuchos::getIntegralValue<Core::LinearSolver::IterativeSolverType>(inparams, "AZSOLVE"))
    {
      case Core::LinearSolver::IterativeSolverType::cg:
        beloslist.set("Solver Type", "CG");
        break;
      case Core::LinearSolver::IterativeSolverType::bicgstab:
        beloslist.set("Solver Type", "BiCGSTAB");
        break;
      case Core::LinearSolver::IterativeSolverType::gmres:
        beloslist.set("Solver Type", "GMRES");
        beloslist.set("Num Blocks", inparams.get<int>("AZSUB"));
        break;
      default:
      {
        FOUR_C_THROW("Flag '%s'! \nUnknown solver for Belos.",
            Teuchos::getIntegralValue<Core::LinearSolver::IterativeSolverType>(
                inparams, "AZSOLVE"));
        break;
      }
    }
  }

  // set type of preconditioner
  const auto azprectype =
      Teuchos::getIntegralValue<Core::LinearSolver::PreconditionerType>(inparams, "AZPREC");

  switch (azprectype)
  {
    case Core::LinearSolver::PreconditionerType::ilu:
      beloslist.set("Preconditioner Type", "ILU");
      break;
    case Core::LinearSolver::PreconditionerType::multigrid_muelu:
      beloslist.set("Preconditioner Type", "ML");
      break;
    case Core::LinearSolver::PreconditionerType::multigrid_nxn:
      beloslist.set("Preconditioner Type", "AMGnxn");
      break;
    case Core::LinearSolver::PreconditionerType::block_teko:
      beloslist.set("Preconditioner Type", "Teko");
      break;
    default:
      FOUR_C_THROW("Unknown preconditioner for Belos");
      break;
  }

  // set parameters for Ifpack if used
  if (azprectype == Core::LinearSolver::PreconditionerType::ilu)
  {
    Teuchos::ParameterList &ifpacklist = outparams.sublist("IFPACK Parameters");
    ifpacklist = translate_four_c_to_ifpack(inparams);
  }

  // set parameters for ML if used
  if (azprectype == Core::LinearSolver::PreconditionerType::multigrid_muelu)
  {
    Teuchos::ParameterList &muelulist = outparams.sublist("MueLu Parameters");
    muelulist = translate_four_c_to_muelu(inparams, &beloslist);
  }
  if (azprectype == Core::LinearSolver::PreconditionerType::block_teko)
  {
    Teuchos::ParameterList &tekolist = outparams.sublist("Teko Parameters");
    tekolist = translate_four_c_to_teko(inparams, &beloslist);
  }
  if (azprectype == Core::LinearSolver::PreconditionerType::multigrid_nxn)
  {
    Teuchos::ParameterList &amgnxnlist = outparams.sublist("AMGnxn Parameters");
    std::string amgnxn_xml = inparams.get<std::string>("AMGNXN_XML_FILE");
    amgnxnlist.set<std::string>("AMGNXN_XML_FILE", amgnxn_xml);
    std::string amgnxn_type = inparams.get<std::string>("AMGNXN_TYPE");
    amgnxnlist.set<std::string>("AMGNXN_TYPE", amgnxn_type);
  }

  return outparams;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::ParameterList Core::LinAlg::Solver::translate_solver_parameters(
    const Teuchos::ParameterList &inparams,
    const std::function<const Teuchos::ParameterList &(int)> &get_solver_params,
    Core::IO::Verbositylevel verbosity)
{
  TEUCHOS_FUNC_TIME_MONITOR("Core::LinAlg::Solver:  0)   translate_solver_parameters");

  Teuchos::ParameterList outparams;
  if (inparams.isParameter("NAME"))
    outparams.set<std::string>("name", inparams.get<std::string>("NAME"));

  switch (Teuchos::getIntegralValue<Core::LinearSolver::SolverType>(inparams, "SOLVER"))
  {
    case Core::LinearSolver::SolverType::undefined:
      std::cout << "undefined solver! Set " << inparams.name() << "  in your dat file!"
                << std::endl;
      FOUR_C_THROW("fix your dat file");
      break;
    case Core::LinearSolver::SolverType::umfpack:
      outparams.set("solver", "umfpack");
      break;
    case Core::LinearSolver::SolverType::superlu:
      outparams.set("solver", "superlu");
      break;
    case Core::LinearSolver::SolverType::belos:
      outparams = translate_four_c_to_belos(inparams, get_solver_params, verbosity);
      break;
    default:
      FOUR_C_THROW("Unsupported type of solver");
      break;
  }

  return outparams;
}

FOUR_C_NAMESPACE_CLOSE

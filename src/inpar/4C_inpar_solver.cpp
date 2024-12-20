// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_solver.hpp"

#include "4C_linear_solver_method.hpp"
#include "4C_utils_parameter_list.hpp"

#include <BelosTypes.hpp>

FOUR_C_NAMESPACE_OPEN

namespace Inpar::SOLVER
{
  void set_valid_solver_parameters(Teuchos::ParameterList& list)
  {
    // Solver options
    {
      Teuchos::setStringToIntegralParameter<Core::LinearSolver::SolverType>("SOLVER", "undefined",
          "The solver to attack the system of linear equations arising of FE approach with.",
          Teuchos::tuple<std::string>("UMFPACK", "Superlu", "Belos", "undefined"),
          Teuchos::tuple<Core::LinearSolver::SolverType>(Core::LinearSolver::SolverType::umfpack,
              Core::LinearSolver::SolverType::superlu, Core::LinearSolver::SolverType::belos,
              Core::LinearSolver::SolverType::undefined),
          &list);
    }

    // Iterative solver options
    {
      Teuchos::setStringToIntegralParameter<Core::LinearSolver::IterativeSolverType>("AZSOLVE",
          "GMRES", "Type of linear solver algorithm to use.",
          Teuchos::tuple<std::string>("CG", "GMRES", "BiCGSTAB"),
          Teuchos::tuple<Core::LinearSolver::IterativeSolverType>(
              Core::LinearSolver::IterativeSolverType::cg,
              Core::LinearSolver::IterativeSolverType::gmres,
              Core::LinearSolver::IterativeSolverType::bicgstab),
          &list);
    }

    // Preconditioner options
    {
      Teuchos::setStringToIntegralParameter<Core::LinearSolver::PreconditionerType>("AZPREC", "ILU",
          "Type of internal preconditioner to use.\n"
          "Note! this preconditioner will only be used if the input operator\n"
          "supports the Epetra_RowMatrix interface and the client does not pass\n"
          "in an external preconditioner!",
          Teuchos::tuple<std::string>("ILU", "MueLu", "AMGnxn", "Teko"),
          Teuchos::tuple<Core::LinearSolver::PreconditionerType>(
              Core::LinearSolver::PreconditionerType::ilu,
              Core::LinearSolver::PreconditionerType::multigrid_muelu,
              Core::LinearSolver::PreconditionerType::multigrid_nxn,
              Core::LinearSolver::PreconditionerType::block_teko),
          &list);
    }

    // Ifpack options
    {
      Core::Utils::int_parameter("IFPACKOVERLAP", 0,
          "The amount of overlap used for the ifpack \"ilu\" preconditioner.", &list);

      Core::Utils::int_parameter("IFPACKGFILL", 0,
          "The amount of fill allowed for an internal \"ilu\" preconditioner.", &list);

      std::vector<std::string> ifpack_combine_valid_input = {"Add", "Insert", "Zero"};
      Core::Utils::string_parameter("IFPACKCOMBINE", "Add",
          "Combine mode for Ifpack Additive Schwarz", &list, ifpack_combine_valid_input);
    }

    // Iterative solver options
    {
      Core::Utils::int_parameter("AZITER", 1000,
          "The maximum number of iterations the underlying iterative solver is allowed to "
          "perform",
          &list);

      Core::Utils::double_parameter("AZTOL", 1e-8,
          "The level the residual norms must reach to decide about successful convergence", &list);

      Teuchos::setStringToIntegralParameter<Belos::ScaleType>("AZCONV", "AZ_r0",
          "The implicit residual norm scaling type to use for terminating the iterative solver.",
          Teuchos::tuple<std::string>("AZ_r0", "AZ_noscaled"),
          Teuchos::tuple<Belos::ScaleType>(Belos::ScaleType::NormOfInitRes, Belos::ScaleType::None),
          &list);

      Core::Utils::int_parameter("AZOUTPUT", 0,
          "The number of iterations between each output of the solver's progress is written to "
          "screen",
          &list);
      Core::Utils::int_parameter(
          "AZREUSE", 0, "The number specifying how often to recompute some preconditioners", &list);

      Core::Utils::int_parameter("AZSUB", 50,
          "The maximum size of the Krylov subspace used with \"GMRES\" before\n"
          "a restart is performed.",
          &list);

      Core::Utils::string_parameter(
          "SOLVER_XML_FILE", "none", "xml file defining any linear solver", &list);
    }

    // MueLu options
    {
      Core::Utils::string_parameter(
          "MUELU_XML_FILE", "none", "xml file defining any MueLu preconditioner", &list);
    }

    // Teko options
    {
      Core::Utils::string_parameter(
          "TEKO_XML_FILE", "none", "xml file defining any Teko preconditioner", &list);
    }

    // user-given name of solver block (just for beauty)
    Core::Utils::string_parameter("NAME", "No_name", "User specified name for solver block", &list);

    // Parameters for AMGnxn Preconditioner
    {
      Core::Utils::string_parameter("AMGNXN_TYPE", "AMG(BGS)",
          "Name of the pre-built preconditioner to be used. If set to\"XML\" the preconditioner "
          "is defined using a xml file",
          &list);
      Core::Utils::string_parameter(
          "AMGNXN_XML_FILE", "none", "xml file defining the AMGnxn preconditioner", &list);
    }
  }


  void set_valid_parameters(Teuchos::ParameterList& list)
  {
    // set valid parameters for solver blocks

    // Note: the maximum number of solver blocks is hardwired here. If you change this,
    // don't forget to edit the corresponding parts in globalproblems.cpp, too.
    for (int i = 1; i < 10; i++)
    {
      std::stringstream ss;
      ss << "SOLVER " << i;
      std::stringstream ss_description;
      ss_description << "solver parameters for solver block " << i;
      Teuchos::ParameterList& solverlist = list.sublist(ss.str(), false, ss_description.str());
      set_valid_solver_parameters(solverlist);
    }
  }

}  // namespace Inpar::SOLVER

FOUR_C_NAMESPACE_CLOSE

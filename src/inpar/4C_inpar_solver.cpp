// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_solver.hpp"

#include "4C_io_input_spec_builders.hpp"
#include "4C_linear_solver_method.hpp"

#include <BelosTypes.hpp>

FOUR_C_NAMESPACE_OPEN

namespace Inpar::SOLVER
{
  Core::IO::InputSpec make_valid_solver_parameters()
  {
    using namespace Core::IO::InputSpecBuilders;
    return all_of({
        // Solver options
        deprecated_selection<Core::LinearSolver::SolverType>("SOLVER",
            {
                {"UMFPACK", Core::LinearSolver::SolverType::umfpack},
                {"Superlu", Core::LinearSolver::SolverType::superlu},
                {"Belos", Core::LinearSolver::SolverType::belos},
                {"undefined", Core::LinearSolver::SolverType::undefined},
            },
            {.description = "The solver to attack the system of linear equations arising of FE "
                            "approach with.",
                .default_value = Core::LinearSolver::SolverType::undefined}),

        // Iterative solver options
        deprecated_selection<Core::LinearSolver::IterativeSolverType>("AZSOLVE",
            {
                {"CG", Core::LinearSolver::IterativeSolverType::cg},
                {"GMRES", Core::LinearSolver::IterativeSolverType::gmres},
                {"BiCGSTAB", Core::LinearSolver::IterativeSolverType::bicgstab},
            },
            {.description = "Type of linear solver algorithm to use.",
                .default_value = Core::LinearSolver::IterativeSolverType::gmres}),

        // Preconditioner options
        deprecated_selection<Core::LinearSolver::PreconditionerType>("AZPREC",
            {
                {"ILU", Core::LinearSolver::PreconditionerType::ilu},
                {"MueLu", Core::LinearSolver::PreconditionerType::multigrid_muelu},
                {"AMGnxn", Core::LinearSolver::PreconditionerType::multigrid_nxn},
                {"Teko", Core::LinearSolver::PreconditionerType::block_teko},
            },
            {.description =
                    "Type of internal preconditioner to use.\nNote! this preconditioner will "
                    "only be used if the input operator\nsupports the Epetra "
                    "interface and the client does not pass\nin an external preconditioner!",
                .default_value = Core::LinearSolver::PreconditionerType::ilu}),

        // Ifpack options
        Core::IO::InputSpecBuilders::parameter<std::optional<std::filesystem::path>>(
            "IFPACK_XML_FILE",
            {.description = "This parameter describes the absolute or relative path to an xml file "
                            "containing the configuration of a Trilinos/Ifpack preconditioner. The "
                            "content of this xml file needs to follow Ifpack guidelines. Consult "
                            "the Trilinos/Ifpack documentation and user guide for more information "
                            "on valid Ifpack parameters.."}),

        // Iterative solver options
        parameter<int>(
            "AZITER", {.description = "The maximum number of iterations the underlying iterative "
                                      "solver is allowed to perform",
                          .default_value = 1000}),

        parameter<double>("AZTOL", {.description = "The level the residual norms must reach to "
                                                   "decide about successful convergence",
                                       .default_value = 1e-8}),

        deprecated_selection<Belos::ScaleType>("AZCONV",
            {
                {"AZ_r0", Belos::ScaleType::NormOfInitRes},
                {"AZ_noscaled", Belos::ScaleType::None},
            },
            {.description = "The implicit residual norm scaling type to use for terminating the "
                            "iterative solver.",
                .default_value = Belos::ScaleType::NormOfInitRes}),

        parameter<int>(
            "AZOUTPUT", {.description = "The number of iterations between each output of the "
                                        "solver's progress is written to "
                                        "screen",
                            .default_value = 0}),

        parameter<int>("AZREUSE",
            {.description = "The number specifying how often to recompute some preconditioners",
                .default_value = 0}),

        parameter<int>(
            "AZSUB", {.description = "The maximum size of the Krylov subspace used with \"GMRES\" "
                                     "before\n a restart is performed.",
                         .default_value = 50}),

        parameter<bool>("THROW_IF_UNCONVERGED",
            {.description =
                    "If set to true, the iterative linear solver "
                    "will throw an exception if it does not "
                    "converge. To only issue a warning without stopping the simulation, set "
                    "this parameter to false.",
                .default_value = true}),

        parameter<std::optional<std::filesystem::path>>(
            "SOLVER_XML_FILE", {.description = "xml file defining any iterative solver"}),


        // MueLu options
        Core::IO::InputSpecBuilders::parameter<std::optional<std::filesystem::path>>(
            "MUELU_XML_FILE", {.description = "xml file defining any MueLu preconditioner"}),

        // Teko options
        Core::IO::InputSpecBuilders::parameter<std::optional<std::filesystem::path>>(
            "TEKO_XML_FILE", {.description = "xml file defining any Teko preconditioner"}),

        // user-given name of solver block (just for beauty)
        parameter<std::string>("NAME",
            {.description = "User specified name for solver block", .default_value = "No_name"}),

        // Parameters for AMGnxn Preconditioner
        parameter<std::string>("AMGNXN_TYPE",
            {.description = "Name of the pre-built preconditioner to be used. If set "
                            "to\"XML\" the preconditioner is defined using a xml file",
                .default_value = "AMG(BGS)"}),

        Core::IO::InputSpecBuilders::parameter<std::optional<std::filesystem::path>>(
            "AMGNXN_XML_FILE", {.description = "xml file defining the AMGnxn preconditioner"}),
    });
  }


  void set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list)
  {
    // set valid parameters for solver blocks

    // Note: the maximum number of solver blocks is hardwired here. If you change this,
    // don't forget to edit the corresponding parts in globalproblems.cpp, too.
    auto spec_solver = make_valid_solver_parameters();
    for (int i = 1; i < 10; i++)
    {
      std::stringstream ss;
      ss << "SOLVER " << i;
      std::stringstream ss_description;
      ss_description << "solver parameters for solver block " << i;
      list[ss.str()] = Core::IO::InputSpecBuilders::group(
          ss.str(), {spec_solver}, {.description = ss_description.str(), .defaultable = true});
    }
  }

}  // namespace Inpar::SOLVER

FOUR_C_NAMESPACE_CLOSE

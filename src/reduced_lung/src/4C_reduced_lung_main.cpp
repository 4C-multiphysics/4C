// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_reduced_lung_main.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_comm_utils.hpp"
#include "4C_geometric_search_input.hpp"
#include "4C_global_data.hpp"
#include "4C_rebalance.hpp"
#include "4C_reduced_lung_input.hpp"
#include "4C_reduced_lung_model.hpp"

#include <Teuchos_TimeMonitor.hpp>

#include <iostream>


FOUR_C_NAMESPACE_OPEN

namespace ReducedLung
{
  namespace
  {
    /**
     * Collect reduced-lung input, communicator, IO, and solver configuration from the 4C problem.
     */
    ModelContext make_model_context_from_problem(Global::Problem& problem)
    {
      const ReducedLungParameters parameters =
          problem.parameters().get<ReducedLungParameters>("reduced_dimensional_lung");

      return ModelContext{
          .parameters = parameters,
          .comm = problem.get_communicators().local_comm(),
          .rebalance_parameters =
              Core::Rebalance::RebalanceParameters{
                  .mesh_partitioning_parameters =
                      problem.parameters().get<Core::Rebalance::MeshPartitioningParameters>(
                          "MESH PARTITIONING"),
                  .geometric_search_parameters =
                      Core::GeometricSearch::geometric_search_params_factory(problem.parameters())},
          .io_parameters = problem.io_params(),
          .linear_solver_parameters = problem.solver_params(parameters.dynamics.linear_solver),
          .solver_params_callback = problem.solver_params_callback(),
          .output_control_file = problem.output_control_file(),
          .function_manager = problem.function_manager(),
      };
    }

    /**
     * Return a printable name for the selected reduced-lung nonlinear solver workflow.
     */
    const char* nonlinear_solver_name(ReducedLungParameters::NonlinearSolverType solver)
    {
      using SolverType = ReducedLungParameters::NonlinearSolverType;
      switch (solver)
      {
        case SolverType::NewtonSparse:
          return "NewtonSparse";
        case SolverType::NewtonTree:
          return "NewtonTree";
      }

      return "Unknown";
    }
  }  // namespace

  /**
   * Run the reduced-lung simulation for an explicit 4C problem instance.
   */
  void reduced_lung_main(Global::Problem& problem)
  {
    const ModelContext context = make_model_context_from_problem(problem);
    const auto& dynamics = context.parameters.dynamics;

    Model model(context);
    const MPI_Comm comm = context.comm;
    const bool is_rank_zero = Core::Communication::my_mpi_rank(comm) == 0;

    if (is_rank_zero)
    {
      std::cout << "-------- Start Time Integration --------\n"
                << "Reduced lung nonlinear solver: "
                << nonlinear_solver_name(dynamics.nonlinear_solver) << "\n"
                << "----------------------------------------\n"
                << std::flush;
    }

    {
      TEUCHOS_FUNC_TIME_MONITOR("TimeLoop");
      for (int step = 1; step <= dynamics.number_of_steps; ++step)
      {
        if (is_rank_zero)
        {
          std::cout << "Timestep: " << step << "/" << dynamics.number_of_steps
                    << "\n----------------------------------------\n"
                    << std::flush;
        }

        model.solve_step(dynamics.time_increment);
        model.accept_step();

        if (step % dynamics.results_every == 0)
        {
          model.write_output(step);
        }
      }
    }

    problem.add_field_test(model.create_result_test());
    problem.test_all(comm);
  }

  /**
   * Run the reduced-lung simulation for the global 4C problem instance.
   */
  void reduced_lung_main() { reduced_lung_main(*Global::Problem::instance()); }
}  // namespace ReducedLung

FOUR_C_NAMESPACE_CLOSE

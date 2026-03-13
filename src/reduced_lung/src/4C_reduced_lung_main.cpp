// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_reduced_lung_main.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_comm_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_io_discretization_visualization_writer_mesh.hpp"
#include "4C_io_input_field.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_rebalance.hpp"
#include "4C_reduced_lung_airways.hpp"
#include "4C_reduced_lung_boundary_conditions.hpp"
#include "4C_reduced_lung_helpers.hpp"
#include "4C_reduced_lung_input.hpp"
#include "4C_reduced_lung_junctions.hpp"
#include "4C_reduced_lung_terminal_unit.hpp"
#include "4C_solver_nonlin_nox_adapter.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

#include <functional>
#include <iostream>
#include <map>
#include <memory>


FOUR_C_NAMESPACE_OPEN

namespace ReducedLung
{
  namespace
  {
    struct ReducedLungContext
    {
      ReducedLungParameters parameters;
      MPI_Comm local_comm;
      Core::Rebalance::RebalanceParameters rebalance_parameters;
      const Teuchos::ParameterList& io_parameters;
      const Teuchos::ParameterList& linear_solver_parameters;
      std::function<const Teuchos::ParameterList&(int)> solver_params_callback;
      std::shared_ptr<Core::IO::OutputControl> output_control_file;
      const Core::Utils::FunctionManager& function_manager;
    };

    ReducedLungContext make_reduced_lung_context_from_problem(Global::Problem& problem)
    {
      const ReducedLungParameters parameters =
          problem.parameters().get<ReducedLungParameters>("reduced_dimensional_lung");

      return ReducedLungContext{
          .parameters = parameters,
          .local_comm = problem.get_communicators().local_comm(),
          .rebalance_parameters =
              Core::Rebalance::RebalanceParameters{
                  .mesh_partitioning_parameters =
                      problem.parameters().get<Core::Rebalance::MeshPartitioningParameters>(
                          "MESH PARTITIONING"),
                  .geometric_search_parameters = problem.geometric_search_params(),
                  .io_parameters = problem.io_params(),
              },
          .io_parameters = problem.io_params(),
          .linear_solver_parameters = problem.solver_params(parameters.dynamics.linear_solver),
          .solver_params_callback = problem.solver_params_callback(),
          .output_control_file = problem.output_control_file(),
          .function_manager = problem.function_manager(),
      };
    }

    void run_reduced_lung(const ReducedLungContext& context)
    {
      auto actdis =
          std::make_shared<Core::FE::Discretization>("reduced_lung", context.local_comm, 3);

      build_discretization_from_topology(
          *actdis, context.parameters.lung_tree.topology, context.rebalance_parameters);
      actdis->fill_complete();

      auto solver = std::make_shared<Core::LinAlg::Solver>(context.linear_solver_parameters,
          actdis->get_comm(), context.solver_params_callback,
          Teuchos::getIntegralValue<Core::IO::Verbositylevel>(context.io_parameters, "VERBOSITY"));

      // Create runtime output writer
      Core::IO::DiscretizationVisualizationWriterMesh visualization_writer(
          actdis, Core::IO::visualization_parameters_factory(
                      context.io_parameters.sublist("RUNTIME VTK OUTPUT"),
                      *context.output_control_file, 0));

      // The existing mpi communicator is recycled for the new data layout.
      const auto& comm = actdis->get_comm();

      // "Elements" of the lung tree introducing the dofs.
      Airways::AirwayContainer airways;
      TerminalUnits::TerminalUnitContainer terminal_units;
      std::map<int, int> dof_per_ele;  // Map global element id -> dof.
      int n_airways = 0;
      int n_terminal_units = 0;
      create_local_element_models(*actdis, context.parameters, airways, terminal_units, dof_per_ele,
          n_airways, n_terminal_units);

      // Create global dof numbering (done on every processor simultaneously).
      std::map<int, int> first_global_dof_of_ele;
      std::map<int, int> global_dof_per_ele;
      create_global_dof_maps(dof_per_ele, comm, global_dof_per_ele, first_global_dof_of_ele);
      assign_global_dof_ids_to_models(first_global_dof_of_ele, airways, terminal_units);

      // Create evaluator functions (assembly, residual, updates) for the different models.
      TerminalUnits::create_evaluators(terminal_units);
      Airways::create_evaluators(airways);

      // Mapping node -> elements to differentiate between boundary conditions and junctions.
      auto global_ele_ids_per_node = create_global_ele_ids_per_node(*actdis, comm);

      // Create entities with equations connecting elements (acting on "nodes" of the lung tree).
      BoundaryConditions::BoundaryConditionContainer boundary_conditions;
      Junctions::ConnectionData connections;
      Junctions::BifurcationData bifurcations;

      BoundaryConditions::create_boundary_conditions(*actdis, context.parameters,
          global_ele_ids_per_node, global_dof_per_ele, first_global_dof_of_ele,
          context.function_manager, boundary_conditions);
      BoundaryConditions::create_evaluators(boundary_conditions);

      Junctions::create_junctions(*actdis, global_ele_ids_per_node, global_dof_per_ele,
          first_global_dof_of_ele, connections, bifurcations);
      int n_connections = static_cast<int>(connections.size());
      int n_bifurcations = static_cast<int>(bifurcations.size());
      int n_boundary_conditions =
          BoundaryConditions::count_boundary_conditions(boundary_conditions);

      print_instantiated_object_counts(
          comm, n_airways, n_terminal_units, n_connections, n_bifurcations, n_boundary_conditions);

      // Calculate local and global number of "element" equations and assign local row IDs to
      // define the structure of the system of equations (for the row map).
      int n_local_equations = 0;
      Airways::assign_local_equation_ids(airways, n_local_equations);
      TerminalUnits::assign_local_equation_ids(terminal_units, n_local_equations);
      // Assign local equation ids to connections, bifurcations, and boundary conditions.
      Junctions::assign_junction_local_equation_ids(connections, bifurcations, n_local_equations);
      BoundaryConditions::assign_local_equation_ids(boundary_conditions, n_local_equations);

      // Create all necessary maps for matrix, rhs, and dof-vector.
      // Map with all dof ids belonging to the local elements (airways and terminal units).
      const Core::LinAlg::Map locally_owned_dof_map =
          create_domain_map(comm, airways, terminal_units);
      // Map with row ids for the equations of local elements, connections, bifurcations, and
      // boundary conditions.
      const Core::LinAlg::Map row_map = create_row_map(
          comm, airways, terminal_units, connections, bifurcations, boundary_conditions);
      // Map with all relevant dof ids for the local equations.
      const Core::LinAlg::Map locally_relevant_dof_map =
          create_column_map(comm, airways, terminal_units, global_dof_per_ele,
              first_global_dof_of_ele, connections, bifurcations, boundary_conditions);

      // Assign global equation ids to connections, bifurcations, and boundary conditions based on
      // the row map. Maybe not necessary, but helps with debugging.
      Junctions::assign_junction_global_equation_ids(row_map, connections, bifurcations);
      BoundaryConditions::assign_global_equation_ids(row_map, boundary_conditions);

      // Save locally relevant dof ids of every entity. Needed for local assembly.
      Airways::assign_local_dof_ids(locally_relevant_dof_map, airways);
      TerminalUnits::assign_local_dof_ids(locally_relevant_dof_map, terminal_units);
      Junctions::assign_junction_local_dof_ids(locally_relevant_dof_map, connections, bifurcations);
      BoundaryConditions::assign_local_dof_ids(locally_relevant_dof_map, boundary_conditions);

      // Create system matrix and vectors:
      // Vector with all degrees of freedom (p1, p2, q, ...) associated to the elements.
      auto dofs = Core::LinAlg::Vector<double>(locally_owned_dof_map, true);
      // Vector with all degrees of freedom (p1, p2, q, ...) at the last timestep.
      auto dofs_n = Core::LinAlg::Vector<double>(locally_owned_dof_map, true);
      // Vector with locally relevant degrees of freedom, needs to import data from dofs vector.
      auto locally_relevant_dofs = Core::LinAlg::Vector<double>(locally_relevant_dof_map, true);
      // Solution vector (on row_map) used as the NOX initial guess each time step.
      auto x = Core::LinAlg::Vector<double>(row_map, true);
      // Right hand side vector with residuals of the system equations.
      auto rhs = Core::LinAlg::Vector<double>(row_map, true);
      // Jacobian of the system equations.
      auto sysmat = Core::LinAlg::SparseMatrix(row_map, locally_relevant_dof_map, 3);

      // Time integration parameters.
      const double dt = context.parameters.dynamics.time_increment;
      const int n_timesteps = context.parameters.dynamics.number_of_steps;
      const int max_nonlinear_iterations = context.parameters.dynamics.max_nonlinear_iterations;
      const double nonlinear_residual_tolerance =
          context.parameters.dynamics.nonlinear_residual_tolerance;
      const double nonlinear_increment_tolerance =
          context.parameters.dynamics.nonlinear_increment_tolerance;
      Airways::update_internal_state_vectors(airways, locally_relevant_dofs, dt);
      TerminalUnits::update_internal_state_vectors(terminal_units, locally_relevant_dofs, dt);

      Teuchos::ParameterList nox_params = create_nox_parameter_list(
          nonlinear_residual_tolerance, nonlinear_increment_tolerance, max_nonlinear_iterations);

      FourC::NOX::Nln::Adapter::SolverMap solver_map;
      solver_map[FourC::NOX::Nln::sol_generic] = solver;
      FourC::NOX::Nln::Adapter nox_adapter(comm, nox_params, std::move(solver_map));
      auto nox_callbacks = create_nox_adapter_callbacks(dofs, locally_relevant_dofs, airways,
          terminal_units, connections, bifurcations, boundary_conditions, dt);
      nox_adapter.residual = nox_callbacks.residual;
      nox_adapter.jacobian = nox_callbacks.jacobian;

      // Time loop
      if (Core::Communication::my_mpi_rank(comm) == 0)
      {
        std::cout << "-------- Start Time Integration --------\n"
                  << "----------------------------------------\n"
                  << std::flush;
      }
      for (int n = 1; n <= n_timesteps; n++)
      {
        if (Core::Communication::my_mpi_rank(comm) == 0)
        {
          std::cout << "Timestep: " << n << "/" << n_timesteps
                    << "\n----------------------------------------\n"
                    << std::flush;
        }
        dofs_n.update(1.0, dofs, 0.0);

        const double current_time = n * dt;
        nox_callbacks.set_time(current_time);

        // Solve the nonlinear system and map converged equation solution back to DOFs.
        nox_adapter.solve(x, rhs, sysmat);
        export_to(x, dofs);
        export_to(dofs, locally_relevant_dofs);

        // End-of-timestep routines (history updates, not per-iteration).
        TerminalUnits::end_of_timestep_routine(terminal_units, locally_relevant_dofs, dt);
        Airways::end_of_timestep_routine(airways, locally_relevant_dofs, dt);

        // Runtime output
        if (n % context.parameters.dynamics.results_every == 0)
        {
          visualization_writer.reset();
          collect_runtime_output_data(visualization_writer, airways, terminal_units,
              locally_relevant_dofs, actdis->element_row_map());
          visualization_writer.write_to_disk(current_time, n);
        }
      }
    }

  }  // namespace

  void reduced_lung_main()
  {
    const ReducedLungContext context =
        make_reduced_lung_context_from_problem(*Global::Problem::instance());
    run_reduced_lung(context);
  }
}  // namespace ReducedLung

FOUR_C_NAMESPACE_CLOSE

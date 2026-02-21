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

#include <Teuchos_StandardParameterEntryValidators.hpp>

#include <iostream>
#include <map>


FOUR_C_NAMESPACE_OPEN

namespace ReducedLung
{
  void reduced_lung_main()
  {
    // Access given 4C infrastructure.
    auto* problem = Global::Problem::instance();
    ReducedLungParameters parameters =
        problem->parameters().get<ReducedLungParameters>("reduced_dimensional_lung");
    MPI_Comm local_comm = problem->get_communicators().local_comm();
    auto actdis = std::make_shared<Core::FE::Discretization>("reduced_lung", local_comm, 3);

    Core::Rebalance::RebalanceParameters rebalance_parameters{
        .mesh_partitioning_parameters =
            problem->parameters().get<Core::Rebalance::MeshPartitioningParameters>(
                "MESH PARTITIONING"),
        .geometric_search_parameters = problem->geometric_search_params(),
        .io_parameters = problem->io_params(),
    };
    build_discretization_from_topology(
        *actdis, parameters.lung_tree.topology, rebalance_parameters);
    actdis->fill_complete();
    Core::LinAlg::Solver solver(problem->solver_params(parameters.dynamics.linear_solver),
        actdis->get_comm(), problem->solver_params_callback(),
        Teuchos::getIntegralValue<Core::IO::Verbositylevel>(problem->io_params(), "VERBOSITY"));
    // Create runtime output writer
    Core::IO::DiscretizationVisualizationWriterMesh visualization_writer(
        actdis, Core::IO::visualization_parameters_factory(
                    problem->io_params().sublist("RUNTIME VTK OUTPUT"),
                    *problem->output_control_file(), 0));
    // The existing mpi communicator is recycled for the new data layout.
    const auto& comm = actdis->get_comm();

    // "Elements" of the lung tree introducing the dofs.
    Airways::AirwayContainer airways;
    TerminalUnits::TerminalUnitContainer terminal_units;
    std::map<int, int> dof_per_ele;  // Map global element id -> dof.
    int n_airways = 0;
    int n_terminal_units = 0;
    create_local_element_models(
        *actdis, parameters, airways, terminal_units, dof_per_ele, n_airways, n_terminal_units);

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

    BoundaryConditions::create_boundary_conditions(*actdis, parameters, global_ele_ids_per_node,
        global_dof_per_ele, first_global_dof_of_ele, boundary_conditions);
    BoundaryConditions::create_evaluators(boundary_conditions);

    Junctions::create_junctions(*actdis, global_ele_ids_per_node, global_dof_per_ele,
        first_global_dof_of_ele, connections, bifurcations);
    int n_connections = static_cast<int>(connections.size());
    int n_bifurcations = static_cast<int>(bifurcations.size());
    int n_boundary_conditions = BoundaryConditions::count_boundary_conditions(boundary_conditions);

    print_instantiated_object_counts(
        comm, n_airways, n_terminal_units, n_connections, n_bifurcations, n_boundary_conditions);

    // Calculate local and global number of "element" equations and assign local row IDs to define
    // the structure of the system of equations (for the row map).
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
    // Map with row ids for the equations of local elements, connections, bifurcations, and boundary
    // conditions.
    const Core::LinAlg::Map row_map = create_row_map(
        comm, airways, terminal_units, connections, bifurcations, boundary_conditions);
    // Map with all relevant dof ids for the local equations.
    const Core::LinAlg::Map locally_relevant_dof_map =
        create_column_map(comm, airways, terminal_units, global_dof_per_ele,
            first_global_dof_of_ele, connections, bifurcations, boundary_conditions);

    // Assign global equation ids to connections, bifurcations, and boundary conditions based on the
    // row map. Maybe not necessary, but helps with debugging.
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
    // Solution vector of the system of equations with increments of all dofs calculated per
    // iteration.
    auto x = Core::LinAlg::Vector<double>(row_map, true);
    // Exported solution that can be directly added to dofs.
    auto x_mapped_to_dofs = Core::LinAlg::Vector<double>(locally_owned_dof_map, true);
    // Right hand side vector with residuals of the system equations.
    auto rhs = Core::LinAlg::Vector<double>(row_map, true);
    // Jacobian of the system equations.
    auto sysmat = Core::LinAlg::SparseMatrix(row_map, locally_relevant_dof_map, 3);

    // Time integration parameters.
    const double dt = parameters.dynamics.time_increment;
    const int n_timesteps = parameters.dynamics.number_of_steps;
    Airways::update_internal_state_vectors(airways, locally_relevant_dofs, dt);
    TerminalUnits::update_internal_state_vectors(terminal_units, locally_relevant_dofs, dt);

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

      Airways::update_negative_residual_vector(rhs, airways, locally_relevant_dofs, dt);
      Airways::update_jacobian(sysmat, airways, locally_relevant_dofs, dt);

      TerminalUnits::update_negative_residual_vector(
          rhs, terminal_units, locally_relevant_dofs, dt);
      TerminalUnits::update_jacobian(sysmat, terminal_units, locally_relevant_dofs, dt);

      Junctions::update_negative_residual_vector(
          rhs, connections, bifurcations, locally_relevant_dofs);
      Junctions::update_jacobian(sysmat, connections, bifurcations);

      BoundaryConditions::update_negative_residual_vector(
          rhs, boundary_conditions, locally_relevant_dofs, n * dt);
      BoundaryConditions::update_jacobian(sysmat, boundary_conditions);

      // Fix sparsity pattern after the first assembly process.
      if (!sysmat.filled())
      {
        sysmat.complete();
      }

      // Solve.
      solver.solve(Core::Utils::shared_ptr_from_ref(sysmat), Core::Utils::shared_ptr_from_ref(x),
          Core::Utils::shared_ptr_from_ref(rhs), {});

      // Update dofs with solution vector.
      export_to(x, x_mapped_to_dofs);
      dofs.update(1.0, x_mapped_to_dofs, 1.0);
      export_to(dofs, locally_relevant_dofs);

      // To be done at end of each nonlinear loop iteration
      TerminalUnits::update_internal_state_vectors(terminal_units, locally_relevant_dofs, dt);
      Airways::update_internal_state_vectors(airways, locally_relevant_dofs, dt);

      // To be done at end of each timestep
      TerminalUnits::end_of_timestep_routine(terminal_units, locally_relevant_dofs, dt);
      Airways::end_of_timestep_routine(airways, locally_relevant_dofs, dt);

      // Runtime output
      if (n % parameters.dynamics.results_every == 0)
      {
        visualization_writer.reset();
        collect_runtime_output_data(visualization_writer, airways, terminal_units,
            locally_relevant_dofs, actdis->element_row_map());
        visualization_writer.write_to_disk(dt * n, n);
      }
    }
  }
}  // namespace ReducedLung

FOUR_C_NAMESPACE_CLOSE

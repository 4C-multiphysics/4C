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
#include "4C_reduced_lung_helpers.hpp"
#include "4C_reduced_lung_input.hpp"
#include "4C_reduced_lung_junctions.hpp"
#include "4C_reduced_lung_terminal_unit.hpp"
#include "4C_utils_function_of_time.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

#include <cmath>

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

    // Create vectors of local entities (equations acting on the dofs).
    // Physical "elements" of the lung tree introducing the dofs.
    AirwayContainer airways;
    TerminalUnitContainer terminal_units;
    std::map<int, int> dof_per_ele;  // Map global element id -> dof.
    int n_airways = 0;
    int n_terminal_units = 0;
    // Loop over all elements in actdis and create the new element data layout (for airways and
    // terminal units). Adds all information directly given in the row element range.
    for (auto ele : actdis->my_row_element_range())
    {
      int global_element_id = ele.global_id();
      int local_element_id = actdis->element_row_map()->lid(global_element_id);
      FOUR_C_ASSERT_ALWAYS(local_element_id >= 0,
          "Element {} not found in element row map while iterating row elements.",
          global_element_id + 1);
      const auto element_type =
          parameters.lung_tree.element_type.at(global_element_id, "element_type");
      if (element_type == ReducedLungParameters::LungTree::ElementType::Airway)
      {
        ReducedLungParameters::LungTree::Airways::FlowModel::ResistanceType flow_model_name =
            parameters.lung_tree.airways.flow_model.resistance_type.at(
                global_element_id, "resistance_type");
        ReducedLungParameters::LungTree::Airways::WallModelType wall_model_type =
            parameters.lung_tree.airways.wall_model_type.at(global_element_id, "wall_model_type");

        add_airway_with_model_selection(airways, global_element_id, local_element_id, parameters,
            flow_model_name, wall_model_type);

        dof_per_ele[global_element_id] = 2 + airways.models.back().data.n_state_equations;
        n_airways++;
      }
      else if (element_type == ReducedLungParameters::LungTree::ElementType::TerminalUnit)
      {
        ReducedLungParameters::LungTree::TerminalUnits::RheologicalModel::RheologicalModelType
            rheological_model_name =
                parameters.lung_tree.terminal_units.rheological_model.rheological_model_type.at(
                    global_element_id, "rheological_model_type");

        ReducedLungParameters::LungTree::TerminalUnits::ElasticityModel::ElasticityModelType
            elasticity_model_name =
                parameters.lung_tree.terminal_units.elasticity_model.elasticity_model_type.at(
                    global_element_id, "elasticity_model_type");

        add_terminal_unit_with_model_selection(terminal_units, global_element_id, local_element_id,
            parameters, rheological_model_name, elasticity_model_name);
        dof_per_ele[global_element_id] = 3;
        n_terminal_units++;
      }
      else
      {
        FOUR_C_THROW("Unknown reduced lung element type.");
      }
    }

    /* Create global dof numbering (done on every processor simultaneously)
       Logic: global dof ids are created from element global ids by expanding them with their
       associated dofs. Example: 3 resistive airway elements:
           ele ids         dof ids
                             {0       |\
             {0               1       |   dofs associated with element 0
                              2       |/
                              3       |\
              1       ->      4       |   dofs associated with element 1
                              5       |/
                              6       |\
              2}              7       |   dofs associated with element 2
                              8}      |/
    */
    auto global_dof_per_ele = Core::Communication::all_reduce(dof_per_ele, comm);
    std::map<int, int> first_global_dof_of_ele;
    int acc = 0;
    for (auto ele_dof : global_dof_per_ele)
    {
      first_global_dof_of_ele[ele_dof.first] = acc;
      acc += ele_dof.second;
    }
    // Assign every local element its associated global dof ids.
    for (auto& model : airways.models)
    {
      for (size_t i = 0; i < model.data.number_of_elements(); i++)
      {
        int first_dof_gid = first_global_dof_of_ele[model.data.global_element_id[i]];
        model.data.gid_p1.push_back(first_dof_gid);
        model.data.gid_p2.push_back(first_dof_gid + 1);
        model.data.gid_q1.push_back(first_dof_gid + 2);
        if (model.data.n_state_equations == 2)
        {
          model.data.gid_q2.push_back(first_dof_gid + 3);
        }
        else if (model.data.n_state_equations == 1)
        {
          // rigid airways -> only 3 unknowns
        }
        else
        {
          FOUR_C_THROW("Number of state equations not implemented.");
        }
      }
    }
    for (auto& model : terminal_units.models)
    {
      for (size_t i = 0; i < model.data.number_of_elements(); i++)
      {
        int first_dof_gid = first_global_dof_of_ele[model.data.global_element_id[i]];
        model.data.gid_p1.push_back(first_dof_gid);
        model.data.gid_p2.push_back(first_dof_gid + 1);
        model.data.gid_q.push_back(first_dof_gid + 2);
      }
    }

    TerminalUnits::create_evaluators(terminal_units);
    Airways::create_evaluators(airways);

    // Build local map node id -> adjacent element id and distribute to all processors.
    std::map<int, std::vector<int>> ele_ids_per_node;
    for (const auto& node : actdis->my_row_node_range())
    {
      for (auto ele : node.adjacent_elements())
      {
        ele_ids_per_node[node.global_id()].push_back(ele.global_id());
      }
    }
    auto merge_maps =
        [](const std::map<int, std::vector<int>>& map1, const std::map<int, std::vector<int>>& map2)
    {
      std::map<int, std::vector<int>> result = map1;
      for (const auto& [key, values] : map2)
      {
        result[key].insert(result[key].end(), values.begin(), values.end());
      }
      return result;
    };
    auto global_ele_ids_per_node = Core::Communication::all_reduce<std::map<int, std::vector<int>>>(
        ele_ids_per_node, merge_maps, comm);

    // Create entities with equations connecting elements (acting on "nodes" of the lung tree).
    std::vector<BoundaryCondition> boundary_conditions;
    Junctions::ConnectionData connections;
    Junctions::BifurcationData bifurcations;
    int n_boundary_conditions = 0;
    const auto& bc_parameters = parameters.boundary_conditions;
    if (bc_parameters.num_conditions < 0)
    {
      FOUR_C_THROW("Number of boundary conditions must be non-negative, got {}.",
          bc_parameters.num_conditions);
    }
    boundary_conditions.reserve(bc_parameters.num_conditions);
    for (int bc_id = 0; bc_id < bc_parameters.num_conditions; ++bc_id)
    {
      const int node_id_one_based = bc_parameters.node_id.at(bc_id, "bc_node_id");
      if (node_id_one_based < 1 || node_id_one_based > parameters.lung_tree.topology.num_nodes)
      {
        FOUR_C_THROW("Boundary condition bc_node_id {} is outside the valid range [1, {}].",
            node_id_one_based, parameters.lung_tree.topology.num_nodes);
      }
      const int node_id = node_id_one_based - 1;
      auto node_it = global_ele_ids_per_node.find(node_id);
      if (node_it == global_ele_ids_per_node.end())
      {
        FOUR_C_THROW(
            "Boundary condition bc_node_id {} is not part of the topology.", node_id_one_based);
      }
      const auto& adjacent_elements = node_it->second;
      if (adjacent_elements.size() != 1u)
      {
        FOUR_C_THROW(
            "Boundary condition bc_node_id {} must connect to exactly one element, but connects "
            "to {} elements.",
            node_id_one_based, adjacent_elements.size());
      }

      const int element_id = adjacent_elements.front();
      const int local_element_id = actdis->element_row_map()->lid(element_id);
      if (local_element_id == -1)
      {
        continue;
      }
      auto* ele = actdis->l_row_element(local_element_id);
      const auto node_ids = ele->node_ids();
      const bool is_inlet = node_ids[0] == node_id;
      const bool is_outlet = node_ids[1] == node_id;
      if (!is_inlet && !is_outlet)
      {
        FOUR_C_THROW(
            "Boundary condition bc_node_id {} is not attached to element {} as inlet or outlet.",
            node_id_one_based, element_id + 1);
      }

      const auto bc_kind = bc_parameters.bc_type.at(bc_id, "bc_type");
      BoundaryConditionType bc_type;
      int dof_offset = 0;
      if (bc_kind == ReducedLungParameters::BoundaryConditions::Type::Pressure)
      {
        bc_type =
            is_inlet ? BoundaryConditionType::pressure_in : BoundaryConditionType::pressure_out;
        dof_offset = is_inlet ? 0 : 1;
      }
      else if (bc_kind == ReducedLungParameters::BoundaryConditions::Type::Flow)
      {
        bc_type = is_inlet ? BoundaryConditionType::flow_in : BoundaryConditionType::flow_out;
        if (is_inlet)
        {
          dof_offset = 2;
        }
        else
        {
          const auto dof_it = global_dof_per_ele.find(element_id);
          FOUR_C_ASSERT_ALWAYS(dof_it != global_dof_per_ele.end(),
              "Missing dof count for element {}.", element_id + 1);
          dof_offset = dof_it->second - 1;
        }
      }
      else
      {
        FOUR_C_THROW(
            "Boundary condition type '{}' not implemented. Supported types are Pressure and Flow.",
            static_cast<int>(bc_kind));
      }

      const auto first_dof_it = first_global_dof_of_ele.find(element_id);
      FOUR_C_ASSERT_ALWAYS(first_dof_it != first_global_dof_of_ele.end(),
          "Missing dof offset for element {}.", element_id + 1);
      const int global_dof_id = first_dof_it->second + dof_offset;
      int function_id = 0;
      double value = 0.0;
      if (bc_parameters.value_source ==
          ReducedLungParameters::BoundaryConditions::ValueSource::bc_function_id)
      {
        function_id = bc_parameters.function_id.at(bc_id, "bc_function_id");
        if (function_id <= 0)
        {
          FOUR_C_THROW("Boundary condition bc_function_id must be positive, got {}.", function_id);
        }
      }
      else if (bc_parameters.value_source ==
               ReducedLungParameters::BoundaryConditions::ValueSource::bc_value)
      {
        value = bc_parameters.value.at(bc_id, "bc_value");
      }
      else
      {
        FOUR_C_THROW("Boundary condition value source not implemented.");
      }

      boundary_conditions.push_back(BoundaryCondition{
          element_id, 0, 0, n_boundary_conditions, bc_type, global_dof_id, function_id, value});
      n_boundary_conditions++;
    }

    Junctions::create_junctions(*actdis, global_ele_ids_per_node, global_dof_per_ele,
        first_global_dof_of_ele, connections, bifurcations);
    int n_connections = static_cast<int>(connections.size());
    int n_bifurcations = static_cast<int>(bifurcations.size());

    // Print info on instantiated objects.
    {
      int n_total_airways, n_total_terminal_units, n_total_connections, n_total_bifurcations,
          n_total_boundary_conditions;
      n_total_airways = Core::Communication::sum_all(n_airways, comm);
      n_total_terminal_units = Core::Communication::sum_all(n_terminal_units, comm);
      n_total_connections = Core::Communication::sum_all(n_connections, comm);
      n_total_bifurcations = Core::Communication::sum_all(n_bifurcations, comm);
      n_total_boundary_conditions = Core::Communication::sum_all(n_boundary_conditions, comm);
      if (Core::Communication::my_mpi_rank(comm) == 0)
      {
        // clang-format off
        std::cout << "--------- Instantiated objects ---------" 
                  << "\nAirways:              |  " << n_total_airways
                  << "\nTerminal Units:       |  " << n_total_terminal_units
                  << "\nConnections:          |  " << n_total_connections
                  << "\nBifurcations:         |  " << n_total_bifurcations
                  << "\nBoundary Conditions:  |  " << n_total_boundary_conditions << "\n\n"
                  << std::flush;
        // clang-format on
      }
    }

    // Calculate local and global number of "element" equations and assign local row IDs to define
    // the structure of the system of equations.
    int n_local_equations = 0;
    for (auto& model : airways.models)
    {
      for (size_t i = 0; i < model.data.number_of_elements(); i++)
      {
        model.data.local_row_id.push_back(n_local_equations);
        n_local_equations += model.data.n_state_equations;
      }
    }
    for (auto& tu_model : terminal_units.models)
    {
      for (size_t i = 0; i < tu_model.data.number_of_elements(); i++)
      {
        tu_model.data.local_row_id.push_back(n_local_equations);
        n_local_equations++;
      }
    }
    // Assign local equation ids to connections, bifurcations, and boundary conditions.
    Junctions::assign_junction_local_equation_ids(connections, bifurcations, n_local_equations);
    for (BoundaryCondition& bc : boundary_conditions)
    {
      // Each boundaary condition adds 1 equation enforcing it at the respective dof.
      bc.local_equation_id = n_local_equations;
      n_local_equations++;
    }

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
    for (BoundaryCondition& bc : boundary_conditions)
    {
      bc.global_equation_id = row_map.gid(bc.local_equation_id);
    }

    // Save locally relevant dof ids of every entity. Needed for local assembly.
    for (auto& model : airways.models)
    {
      for (size_t i = 0; i < model.data.number_of_elements(); i++)
      {
        model.data.lid_p1.push_back(locally_relevant_dof_map.lid(model.data.gid_p1[i]));
        model.data.lid_p2.push_back(locally_relevant_dof_map.lid(model.data.gid_p2[i]));
        model.data.lid_q1.push_back(locally_relevant_dof_map.lid(model.data.gid_q1[i]));
        if (model.data.n_state_equations == 2)
        {
          model.data.lid_q2.push_back(locally_relevant_dof_map.lid(model.data.gid_q2[i]));
        }
      }
    }
    for (auto& tu_model : terminal_units.models)
    {
      for (size_t i = 0; i < tu_model.data.number_of_elements(); i++)
      {
        tu_model.data.lid_p1.push_back(locally_relevant_dof_map.lid(tu_model.data.gid_p1[i]));
        tu_model.data.lid_p2.push_back(locally_relevant_dof_map.lid(tu_model.data.gid_p2[i]));
        tu_model.data.lid_q.push_back(locally_relevant_dof_map.lid(tu_model.data.gid_q[i]));
      }
    }
    Junctions::assign_junction_local_dof_ids(locally_relevant_dof_map, connections, bifurcations);
    for (BoundaryCondition& bc : boundary_conditions)
    {
      bc.local_dof_id = locally_relevant_dof_map.lid(bc.global_dof_id);
    }

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
      [[maybe_unused]] int err;  // Saves error code of trilinos functions.

      // Assemble system of equations.
      // Assemble airway equations in system matrix and rhs.
      Airways::update_negative_residual_vector(rhs, airways, locally_relevant_dofs, dt);
      Airways::update_jacobian(sysmat, airways, locally_relevant_dofs, dt);

      // Assemble terminal unit equations.
      TerminalUnits::update_negative_residual_vector(
          rhs, terminal_units, locally_relevant_dofs, dt);
      TerminalUnits::update_jacobian(sysmat, terminal_units, locally_relevant_dofs, dt);

      Junctions::update_negative_residual_vector(
          rhs, connections, bifurcations, locally_relevant_dofs);
      Junctions::update_jacobian(sysmat, connections, bifurcations);

      // Assemble boundary conditions (equation: dof_value - bc_value = 0).
      for (const BoundaryCondition& bc : boundary_conditions)
      {
        const double val = 1.0;
        double res;
        double bc_value = bc.value;
        if (bc.function_id > 0)
        {
          bc_value = Global::Problem::instance()
                         ->function_by_id<Core::Utils::FunctionOfTime>(bc.function_id)
                         .evaluate(n * dt);
        }
        int local_dof_id = bc.local_dof_id;
        if (!sysmat.filled())
        {
          sysmat.insert_my_values(bc.local_equation_id, 1, &val, &local_dof_id);
        }
        else
        {
          sysmat.replace_my_values(bc.local_equation_id, 1, &val, &local_dof_id);
        }
        res = -locally_relevant_dofs.local_values_as_span()[local_dof_id] + bc_value;
        rhs.replace_local_value(bc.local_equation_id, res);
      }

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

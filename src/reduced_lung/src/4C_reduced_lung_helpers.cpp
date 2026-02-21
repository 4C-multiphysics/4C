// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_reduced_lung_helpers.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_fem_condition.hpp"
#include "4C_fem_discretization_builder.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_rebalance.hpp"
#include "4C_reduced_lung_airways.hpp"
#include "4C_reduced_lung_input.hpp"
#include "4C_reduced_lung_terminal_unit.hpp"
#include "4C_utils_exceptions.hpp"

#include <algorithm>
#include <array>
#include <iostream>

FOUR_C_NAMESPACE_OPEN

namespace ReducedLung
{
  using namespace TerminalUnits;
  using namespace Airways;

  void build_discretization_from_topology(Core::FE::Discretization& discretization,
      const ReducedLungParameters::LungTree::Topology& topology,
      const Core::Rebalance::RebalanceParameters& rebalance_parameters)
  {
    Core::FE::DiscretizationBuilder<3> builder(discretization.get_comm());

    const int my_rank = Core::Communication::my_mpi_rank(discretization.get_comm());
    if (my_rank == 0)
    {
      if (topology.num_nodes <= 0)
      {
        FOUR_C_THROW("Topology num_nodes must be positive, got {}.", topology.num_nodes);
      }
      if (topology.num_elements <= 0)
      {
        FOUR_C_THROW("Topology num_elements must be positive, got {}.", topology.num_elements);
      }

      for (int node_id = 0; node_id < topology.num_nodes; ++node_id)
      {
        const auto coords = topology.node_coordinates.at(node_id, "node_coordinates");
        if (coords.size() != 3u)
        {
          FOUR_C_THROW("Topology node_coordinates entry {} must have 3 components, got {}.",
              node_id + 1, coords.size());
        }
        const std::array<double, 3> coord_array{coords[0], coords[1], coords[2]};
        builder.add_node(coord_array, node_id, nullptr);
      }

      for (int element_id = 0; element_id < topology.num_elements; ++element_id)
      {
        const auto nodes = topology.element_nodes.at(element_id, "element_nodes");
        if (nodes.size() != 2u)
        {
          FOUR_C_THROW("Topology element_nodes entry {} must have 2 entries, got {}.",
              element_id + 1, nodes.size());
        }
        if (nodes[0] < 1 || nodes[1] < 1)
        {
          FOUR_C_THROW(
              "Topology element_nodes entry {} must use 1-based node ids.", element_id + 1);
        }

        const int node_in = nodes[0] - 1;
        const int node_out = nodes[1] - 1;
        if (node_in >= topology.num_nodes || node_out >= topology.num_nodes)
        {
          FOUR_C_THROW("Topology element_nodes entry {} references node ids outside [1, {}].",
              element_id + 1, topology.num_nodes);
        }
        if (node_in == node_out)
        {
          FOUR_C_THROW(
              "Topology element_nodes entry {} uses identical in/out node ids.", element_id + 1);
        }

        const std::array<int, 2> node_ids{node_in, node_out};
        builder.add_element(Core::FE::CellType::line2, node_ids, element_id,
            Core::FE::DiscretizationBuilder<3>::DofInfo{
                .num_dof_per_node = 1,
                .num_dof_per_element = 0,
            });
      }
    }

    builder.build(discretization, rebalance_parameters);
  }

  void create_local_element_models(const Core::FE::Discretization& discretization,
      const ReducedLungParameters& parameters, AirwayContainer& airways,
      TerminalUnitContainer& terminal_units, std::map<int, int>& dof_per_ele, int& n_airways,
      int& n_terminal_units)
  {
    dof_per_ele.clear();
    n_airways = 0;
    n_terminal_units = 0;

    for (auto ele : discretization.my_row_element_range())
    {
      const int global_element_id = ele.global_id();
      const int local_element_id = discretization.element_row_map()->lid(global_element_id);
      FOUR_C_ASSERT_ALWAYS(local_element_id >= 0,
          "Element {} not found in element row map while iterating row elements.",
          global_element_id + 1);

      const auto element_type =
          parameters.lung_tree.element_type.at(global_element_id, "element_type");
      if (element_type == ReducedLungParameters::LungTree::ElementType::Airway)
      {
        auto flow_model_name = parameters.lung_tree.airways.flow_model.resistance_type.at(
            global_element_id, "resistance_type");
        auto wall_model_type =
            parameters.lung_tree.airways.wall_model_type.at(global_element_id, "wall_model_type");

        add_airway_with_model_selection(airways, global_element_id, local_element_id, parameters,
            flow_model_name, wall_model_type);

        // 3 dofs with rigid walls, 4 dofs with compliant walls.
        dof_per_ele[global_element_id] = 2 + airways.models.back().data.n_state_equations;
        n_airways++;
      }
      else if (element_type == ReducedLungParameters::LungTree::ElementType::TerminalUnit)
      {
        auto rheological_model_name =
            parameters.lung_tree.terminal_units.rheological_model.rheological_model_type.at(
                global_element_id, "rheological_model_type");
        auto elasticity_model_name =
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
  }

  void create_global_dof_maps(const std::map<int, int>& local_dof_per_ele, const MPI_Comm& comm,
      std::map<int, int>& global_dof_per_ele, std::map<int, int>& first_global_dof_of_ele)
  {
    global_dof_per_ele = Core::Communication::all_reduce(local_dof_per_ele, comm);

    first_global_dof_of_ele.clear();
    int acc = 0;
    for (const auto& ele_dof : global_dof_per_ele)
    {
      first_global_dof_of_ele[ele_dof.first] = acc;
      acc += ele_dof.second;
    }
  }

  void assign_global_dof_ids_to_models(const std::map<int, int>& first_global_dof_of_ele,
      AirwayContainer& airways, TerminalUnitContainer& terminal_units)
  {
    for (auto& model : airways.models)
    {
      for (size_t i = 0; i < model.data.number_of_elements(); i++)
      {
        const int first_dof_gid = first_global_dof_of_ele.at(model.data.global_element_id[i]);
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
        const int first_dof_gid = first_global_dof_of_ele.at(model.data.global_element_id[i]);
        model.data.gid_p1.push_back(first_dof_gid);
        model.data.gid_p2.push_back(first_dof_gid + 1);
        model.data.gid_q.push_back(first_dof_gid + 2);
      }
    }
  }

  std::map<int, std::vector<int>> create_global_ele_ids_per_node(
      const Core::FE::Discretization& discretization, const MPI_Comm& comm)
  {
    std::map<int, std::vector<int>> ele_ids_per_node;
    for (const auto& node : discretization.my_row_node_range())
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

    return Core::Communication::all_reduce<std::map<int, std::vector<int>>>(
        ele_ids_per_node, merge_maps, comm);
  }

  void print_instantiated_object_counts(const MPI_Comm& comm, int n_airways, int n_terminal_units,
      int n_connections, int n_bifurcations, int n_boundary_conditions)
  {
    int n_total_airways = Core::Communication::sum_all(n_airways, comm);
    int n_total_terminal_units = Core::Communication::sum_all(n_terminal_units, comm);
    int n_total_connections = Core::Communication::sum_all(n_connections, comm);
    int n_total_bifurcations = Core::Communication::sum_all(n_bifurcations, comm);
    int n_total_boundary_conditions = Core::Communication::sum_all(n_boundary_conditions, comm);

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

  Core::LinAlg::Map create_domain_map(const MPI_Comm& comm, const AirwayContainer& airways,
      const TerminalUnitContainer& terminal_units)
  {
    std::vector<int> locally_owned_dof_indices;
    for (const auto& airway : airways.models)
    {
      locally_owned_dof_indices.insert(
          locally_owned_dof_indices.end(), airway.data.gid_p1.begin(), airway.data.gid_p1.end());
      locally_owned_dof_indices.insert(
          locally_owned_dof_indices.end(), airway.data.gid_p2.begin(), airway.data.gid_p2.end());
      locally_owned_dof_indices.insert(
          locally_owned_dof_indices.end(), airway.data.gid_q1.begin(), airway.data.gid_q1.end());
      locally_owned_dof_indices.insert(locally_owned_dof_indices.end(), airway.data.gid_q2.begin(),
          airway.data.gid_q2
              .end());  // does nothing if the model has only 1 state equation, i.e. gid_q2 is empty
    }
    for (const auto& tu_model : terminal_units.models)
    {
      locally_owned_dof_indices.insert(locally_owned_dof_indices.end(),
          tu_model.data.gid_p1.begin(), tu_model.data.gid_p1.end());
      locally_owned_dof_indices.insert(locally_owned_dof_indices.end(),
          tu_model.data.gid_p2.begin(), tu_model.data.gid_p2.end());
      locally_owned_dof_indices.insert(
          locally_owned_dof_indices.end(), tu_model.data.gid_q.begin(), tu_model.data.gid_q.end());
    }
    const Core::LinAlg::Map domain_map(
        -1, locally_owned_dof_indices.size(), locally_owned_dof_indices.data(), 0, comm);

    return domain_map;
  }

  Core::LinAlg::Map create_row_map(const MPI_Comm& comm, const AirwayContainer& airways,
      const TerminalUnitContainer& terminal_units, const Junctions::ConnectionData& connections,
      const Junctions::BifurcationData& bifurcations,
      const BoundaryConditions::BoundaryConditionContainer& boundary_conditions)
  {
    int n_local_state_equations = 0;
    for (const auto& airway : airways.models)
    {
      n_local_state_equations += airway.data.n_state_equations * airway.data.number_of_elements();
    }
    for (const auto& tu_model : terminal_units.models)
    {
      n_local_state_equations += tu_model.data.number_of_elements();
    }
    // Intermediate maps for the different equation types
    const Core::LinAlg::Map state_equations(-1, n_local_state_equations, 0, comm);
    const Core::LinAlg::Map couplings(-1, connections.size() * 2 + bifurcations.size() * 3,
        state_equations.num_global_elements(), comm);
    const Core::LinAlg::Map boundaries(-1,
        BoundaryConditions::count_boundary_conditions(boundary_conditions),
        couplings.num_global_elements() + state_equations.num_global_elements(), comm);

    //  Merge all maps to the full local matrix row map
    std::vector<int> global_row_indices;
    int n_local_indices = state_equations.num_my_elements() + couplings.num_my_elements() +
                          boundaries.num_my_elements();
    global_row_indices.reserve(n_local_indices);
    global_row_indices.insert(global_row_indices.end(), state_equations.my_global_elements(),
        state_equations.my_global_elements() + state_equations.num_my_elements());
    global_row_indices.insert(global_row_indices.end(), couplings.my_global_elements(),
        couplings.my_global_elements() + couplings.num_my_elements());
    global_row_indices.insert(global_row_indices.end(), boundaries.my_global_elements(),
        boundaries.my_global_elements() + boundaries.num_my_elements());

    const Core::LinAlg::Map row_map(-1, n_local_indices, global_row_indices.data(), 0, comm);

    return row_map;
  }

  Core::LinAlg::Map create_column_map(const MPI_Comm& comm, const AirwayContainer& airways,
      const TerminalUnitContainer& terminal_units, const std::map<int, int>& global_dof_per_ele,
      const std::map<int, int>& first_global_dof_of_ele,
      const Junctions::ConnectionData& connections, const Junctions::BifurcationData& bifurcations,
      const BoundaryConditions::BoundaryConditionContainer& boundary_conditions)
  {
    // Vector for intermediate storage of necessary dof ids
    std::vector<int> locally_relevant_dof_indices;

    // Loop over all elements and add their global dof ids
    for (const auto& airway : airways.models)
    {
      locally_relevant_dof_indices.insert(
          locally_relevant_dof_indices.end(), airway.data.gid_p1.begin(), airway.data.gid_p1.end());
      locally_relevant_dof_indices.insert(
          locally_relevant_dof_indices.end(), airway.data.gid_p2.begin(), airway.data.gid_p2.end());
      locally_relevant_dof_indices.insert(
          locally_relevant_dof_indices.end(), airway.data.gid_q1.begin(), airway.data.gid_q1.end());
      locally_relevant_dof_indices.insert(locally_relevant_dof_indices.end(),
          airway.data.gid_q2.begin(),
          airway.data.gid_q2.end());  // does nothing if the model has only 1 state equation, i.e.
                                      // gid_q2 is empty
    }
    for (const auto& tu_model : terminal_units.models)
    {
      locally_relevant_dof_indices.insert(locally_relevant_dof_indices.end(),
          tu_model.data.gid_p1.begin(), tu_model.data.gid_p1.end());
      locally_relevant_dof_indices.insert(locally_relevant_dof_indices.end(),
          tu_model.data.gid_p2.begin(), tu_model.data.gid_p2.end());
      locally_relevant_dof_indices.insert(locally_relevant_dof_indices.end(),
          tu_model.data.gid_q.begin(), tu_model.data.gid_q.end());
    }

    // Loop over all connections of two elements and add relevant dof ids (p and q associated with
    // the end of the parent element and p and q associated with the start of the child element)
    for (size_t i = 0; i < connections.size(); ++i)
    {
      int n_dofs_parent = global_dof_per_ele.find(connections.global_parent_element_id[i])->second;
      // p2 always second dof of element
      locally_relevant_dof_indices.insert(locally_relevant_dof_indices.end(),
          first_global_dof_of_ele.find(connections.global_parent_element_id[i])->second + 1);
      // q_out always last dof of element
      locally_relevant_dof_indices.insert(locally_relevant_dof_indices.end(),
          first_global_dof_of_ele.find(connections.global_parent_element_id[i])->second +
              n_dofs_parent - 1);
      // p1 always first dof of element
      locally_relevant_dof_indices.insert(locally_relevant_dof_indices.end(),
          first_global_dof_of_ele.find(connections.global_child_element_id[i])->second);
      // q_in always third dof of element
      locally_relevant_dof_indices.insert(locally_relevant_dof_indices.end(),
          first_global_dof_of_ele.find(connections.global_child_element_id[i])->second + 2);
    }

    // Loop over all bifurcations and add relevant dof ids (p and q associated with the
    // end of the parent element and p and q associated with the start of the child elements)
    for (size_t i = 0; i < bifurcations.size(); ++i)
    {
      int n_dofs_parent = global_dof_per_ele.find(bifurcations.global_parent_element_id[i])->second;
      // p2 always second dof of element
      locally_relevant_dof_indices.insert(locally_relevant_dof_indices.end(),
          first_global_dof_of_ele.find(bifurcations.global_parent_element_id[i])->second + 1);
      // q_out always last dof of element
      locally_relevant_dof_indices.insert(locally_relevant_dof_indices.end(),
          first_global_dof_of_ele.find(bifurcations.global_parent_element_id[i])->second +
              n_dofs_parent - 1);
      // p1 always first dof of element
      locally_relevant_dof_indices.insert(locally_relevant_dof_indices.end(),
          first_global_dof_of_ele.find(bifurcations.global_child_1_element_id[i])->second);
      locally_relevant_dof_indices.insert(locally_relevant_dof_indices.end(),
          first_global_dof_of_ele.find(bifurcations.global_child_2_element_id[i])->second);
      // q_in always third dof of element
      locally_relevant_dof_indices.insert(locally_relevant_dof_indices.end(),
          first_global_dof_of_ele.find(bifurcations.global_child_1_element_id[i])->second + 2);
      locally_relevant_dof_indices.insert(locally_relevant_dof_indices.end(),
          first_global_dof_of_ele.find(bifurcations.global_child_2_element_id[i])->second + 2);
    }
    // Loop over all boundary conditions and add relevant dof ids (dof where bc is applied)
    for (const auto& model : boundary_conditions.models)
    {
      for (size_t i = 0; i < model.data.size(); ++i)
      {
        locally_relevant_dof_indices.push_back(model.data.global_dof_id[i]);
      }
    }

    // Erase duplicate dof indices and sort the remaining ids
    std::sort(locally_relevant_dof_indices.begin(), locally_relevant_dof_indices.end());
    locally_relevant_dof_indices.erase(
        std::unique(locally_relevant_dof_indices.begin(), locally_relevant_dof_indices.end()),
        locally_relevant_dof_indices.end());

    const Core::LinAlg::Map column_map(
        -1, locally_relevant_dof_indices.size(), locally_relevant_dof_indices.data(), 0, comm);

    return column_map;
  }

  void add_airway_with_model_selection(AirwayContainer& airways, int global_element_id,
      int local_element_id, const ReducedLungParameters& parameters,
      ReducedLungParameters::LungTree::Airways::FlowModel::ResistanceType flow_model_type,
      ReducedLungParameters::LungTree::Airways::WallModelType wall_model_type)
  {
    using ResistanceType = ReducedLungParameters::LungTree::Airways::FlowModel::ResistanceType;
    using WallModelType = ReducedLungParameters::LungTree::Airways::WallModelType;

    if (flow_model_type == ResistanceType::Linear)
    {
      if (wall_model_type == WallModelType::Rigid)
      {
        add_airway_ele<LinearResistive, RigidWall>(
            airways, global_element_id, local_element_id, parameters);
      }
      else if (wall_model_type == WallModelType::KelvinVoigt)
      {
        add_airway_ele<LinearResistive, KelvinVoigtWall>(
            airways, global_element_id, local_element_id, parameters);
      }
      else
      {
        FOUR_C_THROW("Wall model not implemented.");
      }
    }
    else if (flow_model_type == ResistanceType::NonLinear)
    {
      if (wall_model_type == WallModelType::Rigid)
      {
        add_airway_ele<NonLinearResistive, RigidWall>(
            airways, global_element_id, local_element_id, parameters);
      }
      else if (wall_model_type == WallModelType::KelvinVoigt)
      {
        add_airway_ele<NonLinearResistive, KelvinVoigtWall>(
            airways, global_element_id, local_element_id, parameters);
      }
      else
      {
        FOUR_C_THROW("Wall model not implemented.");
      }
    }
    else
    {
      FOUR_C_THROW("Flow model not implemented.");
    }
  }

  void add_terminal_unit_with_model_selection(TerminalUnitContainer& terminal_units,
      int global_element_id, int local_element_id, const ReducedLungParameters& parameters,
      ReducedLungParameters::LungTree::TerminalUnits::RheologicalModel::RheologicalModelType
          rheological_model_type,
      ReducedLungParameters::LungTree::TerminalUnits::ElasticityModel::ElasticityModelType
          elasticity_model_type)
  {
    using RheologicalModelType =
        ReducedLungParameters::LungTree::TerminalUnits::RheologicalModel::RheologicalModelType;
    using ElasticityModelType =
        ReducedLungParameters::LungTree::TerminalUnits::ElasticityModel::ElasticityModelType;

    if (rheological_model_type == RheologicalModelType::KelvinVoigt)
    {
      if (elasticity_model_type == ElasticityModelType::Linear)
      {
        add_terminal_unit_ele<KelvinVoigt, LinearElasticity>(
            terminal_units, global_element_id, local_element_id, parameters);
      }
      else if (elasticity_model_type == ElasticityModelType::Ogden)
      {
        add_terminal_unit_ele<KelvinVoigt, OgdenHyperelasticity>(
            terminal_units, global_element_id, local_element_id, parameters);
      }
      else
      {
        FOUR_C_THROW("Elasticity model not implemented.");
      }
    }
    else if (rheological_model_type == RheologicalModelType::FourElementMaxwell)
    {
      if (elasticity_model_type == ElasticityModelType::Linear)
      {
        add_terminal_unit_ele<FourElementMaxwell, LinearElasticity>(
            terminal_units, global_element_id, local_element_id, parameters);
      }
      else if (elasticity_model_type == ElasticityModelType::Ogden)
      {
        add_terminal_unit_ele<FourElementMaxwell, OgdenHyperelasticity>(
            terminal_units, global_element_id, local_element_id, parameters);
      }
      else
      {
        FOUR_C_THROW("Elasticity model not implemented.");
      }
    }
    else
    {
      FOUR_C_THROW("Rheological model not implemented.");
    }
  }

  void collect_runtime_output_data(
      Core::IO::DiscretizationVisualizationWriterMesh& visualization_writer,
      const AirwayContainer& airways, const TerminalUnitContainer& terminal_units,
      const Core::LinAlg::Vector<double>& locally_relevant_dofs,
      const Core::LinAlg::Map* element_row_map)
  {
    Core::LinAlg::Vector<double> pressure_in(*element_row_map, true);
    Core::LinAlg::Vector<double> pressure_out(*element_row_map, true);
    Core::LinAlg::Vector<double> flow_in(*element_row_map, true);
    Core::LinAlg::Vector<double> flow_out(*element_row_map, true);
    for (const auto& model : airways.models)
    {
      const bool has_q_out = model.data.n_state_equations == 2;
      for (size_t i = 0; i < model.data.number_of_elements(); i++)
      {
        pressure_in.replace_local_value(model.data.local_element_id[i],
            locally_relevant_dofs.local_values_as_span()[model.data.lid_p1[i]]);
        pressure_out.replace_local_value(model.data.local_element_id[i],
            locally_relevant_dofs.local_values_as_span()[model.data.lid_p2[i]]);
        flow_in.replace_local_value(model.data.local_element_id[i],
            locally_relevant_dofs.local_values_as_span()[model.data.lid_q1[i]]);
        flow_out.replace_local_value(model.data.local_element_id[i],
            locally_relevant_dofs
                .local_values_as_span()[has_q_out ? model.data.lid_q2[i] : model.data.lid_q1[i]]);
      }
    }
    for (const auto& model : terminal_units.models)
    {
      for (size_t i = 0; i < model.data.number_of_elements(); i++)
      {
        pressure_in.replace_local_value(model.data.local_element_id[i],
            locally_relevant_dofs.local_values_as_span()[model.data.lid_p1[i]]);
        pressure_out.replace_local_value(model.data.local_element_id[i],
            locally_relevant_dofs.local_values_as_span()[model.data.lid_p2[i]]);
        flow_in.replace_local_value(model.data.local_element_id[i],
            locally_relevant_dofs.local_values_as_span()[model.data.lid_q[i]]);
        flow_out.replace_local_value(model.data.local_element_id[i],
            locally_relevant_dofs.local_values_as_span()[model.data.lid_q[i]]);
      }
    }
    visualization_writer.append_result_data_vector_with_context(
        pressure_in, Core::IO::OutputEntity::element, {"p_1"});
    visualization_writer.append_result_data_vector_with_context(
        pressure_out, Core::IO::OutputEntity::element, {"p_2"});
    visualization_writer.append_result_data_vector_with_context(
        flow_in, Core::IO::OutputEntity::element, {"q_in"});
    visualization_writer.append_result_data_vector_with_context(
        flow_out, Core::IO::OutputEntity::element, {"q_out"});
  }
}  // namespace ReducedLung

FOUR_C_NAMESPACE_CLOSE

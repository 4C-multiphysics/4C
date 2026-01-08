// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_reduced_lung_helpers.hpp"

#include "4C_fem_condition.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_red_airways_elementbase.hpp"
#include "4C_reduced_lung_airways.hpp"
#include "4C_reduced_lung_input.hpp"
#include "4C_reduced_lung_terminal_unit.hpp"


FOUR_C_NAMESPACE_OPEN

namespace ReducedLung
{
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
      const TerminalUnitContainer& terminal_units, const std::vector<Connection>& connections,
      const std::vector<Bifurcation>& bifurcations,
      const std::vector<BoundaryCondition>& boundary_conditions)
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
    const Core::LinAlg::Map boundaries(-1, boundary_conditions.size(),
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
      const std::map<int, int>& first_global_dof_of_ele, const std::vector<Connection>& connections,
      const std::vector<Bifurcation>& bifurcations,
      const std::vector<BoundaryCondition>& boundary_conditions)
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
    for (const auto& conn : connections)
    {
      int n_dofs_parent = global_dof_per_ele.find(conn.global_parent_element_id)->second;
      // p2 always second dof of element
      locally_relevant_dof_indices.insert(locally_relevant_dof_indices.end(),
          first_global_dof_of_ele.find(conn.global_parent_element_id)->second + 1);
      // q_out always last dof of element
      locally_relevant_dof_indices.insert(locally_relevant_dof_indices.end(),
          first_global_dof_of_ele.find(conn.global_parent_element_id)->second + n_dofs_parent - 1);
      // p1 always first dof of element
      locally_relevant_dof_indices.insert(locally_relevant_dof_indices.end(),
          first_global_dof_of_ele.find(conn.global_child_element_id)->second);
      // q_in always third dof of element
      locally_relevant_dof_indices.insert(locally_relevant_dof_indices.end(),
          first_global_dof_of_ele.find(conn.global_child_element_id)->second + 2);
    }

    // Loop over all bifurcations and add relevant dof ids (p and q associated with the
    // end of the parent element and p and q associated with the start of the child elements)
    for (const auto& bif : bifurcations)
    {
      int n_dofs_parent = global_dof_per_ele.find(bif.global_parent_element_id)->second;
      // p2 always second dof of element
      locally_relevant_dof_indices.insert(locally_relevant_dof_indices.end(),
          first_global_dof_of_ele.find(bif.global_parent_element_id)->second + 1);
      // q_out always last dof of element
      locally_relevant_dof_indices.insert(locally_relevant_dof_indices.end(),
          first_global_dof_of_ele.find(bif.global_parent_element_id)->second + n_dofs_parent - 1);
      // p1 always first dof of element
      locally_relevant_dof_indices.insert(locally_relevant_dof_indices.end(),
          first_global_dof_of_ele.find(bif.global_child_1_element_id)->second);
      locally_relevant_dof_indices.insert(locally_relevant_dof_indices.end(),
          first_global_dof_of_ele.find(bif.global_child_2_element_id)->second);
      // q_in always third dof of element
      locally_relevant_dof_indices.insert(locally_relevant_dof_indices.end(),
          first_global_dof_of_ele.find(bif.global_child_1_element_id)->second + 2);
      locally_relevant_dof_indices.insert(locally_relevant_dof_indices.end(),
          first_global_dof_of_ele.find(bif.global_child_2_element_id)->second + 2);
    }
    // Loop over all boundary conditions and add relevant dof ids (dof where bc is applied)
    for (const auto& bc : boundary_conditions)
    {
      switch (bc.bc_type)
      {
        case BoundaryConditionType::pressure_in:
          locally_relevant_dof_indices.insert(locally_relevant_dof_indices.end(),
              first_global_dof_of_ele.find(bc.global_element_id)->second);
          break;
        case BoundaryConditionType::pressure_out:
          locally_relevant_dof_indices.insert(locally_relevant_dof_indices.end(),
              first_global_dof_of_ele.find(bc.global_element_id)->second + 1);
          break;
        case BoundaryConditionType::flow_in:
          locally_relevant_dof_indices.insert(locally_relevant_dof_indices.end(),
              first_global_dof_of_ele.find(bc.global_element_id)->second + 2);
          break;
        case BoundaryConditionType::flow_out:
          locally_relevant_dof_indices.insert(locally_relevant_dof_indices.end(),
              first_global_dof_of_ele.find(bc.local_bc_id)->second +
                  global_dof_per_ele.find(bc.global_element_id)->second - 1);
          break;
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

  void add_airway_with_model_selection(AirwayContainer& airways, Core::Elements::Element* ele,
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
        add_airway_ele<LinearResistive, RigidWall>(airways, ele, local_element_id, parameters);
      }
      else if (wall_model_type == WallModelType::KelvinVoigt)
      {
        add_airway_ele<LinearResistive, KelvinVoigtWall>(
            airways, ele, local_element_id, parameters);
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
        add_airway_ele<NonLinearResistive, RigidWall>(airways, ele, local_element_id, parameters);
      }
      else if (wall_model_type == WallModelType::KelvinVoigt)
      {
        add_airway_ele<NonLinearResistive, KelvinVoigtWall>(
            airways, ele, local_element_id, parameters);
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
      Core::Elements::Element* ele, int local_element_id,
      const ReducedLungParameters::LungTree::TerminalUnits& tu_parameters,
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
            terminal_units, ele, local_element_id, tu_parameters);
      }
      else if (elasticity_model_type == ElasticityModelType::Ogden)
      {
        add_terminal_unit_ele<KelvinVoigt, OgdenHyperelasticity>(
            terminal_units, ele, local_element_id, tu_parameters);
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
            terminal_units, ele, local_element_id, tu_parameters);
      }
      else if (elasticity_model_type == ElasticityModelType::Ogden)
      {
        add_terminal_unit_ele<FourElementMaxwell, OgdenHyperelasticity>(
            terminal_units, ele, local_element_id, tu_parameters);
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
      for (size_t i = 0; i < model.data.number_of_elements(); i++)
      {
        pressure_in.replace_local_value(model.data.local_element_id[i],
            locally_relevant_dofs.local_values_as_span()[model.data.lid_p1[i]]);
        pressure_out.replace_local_value(model.data.local_element_id[i],
            locally_relevant_dofs.local_values_as_span()[model.data.lid_p2[i]]);
        flow_in.replace_local_value(model.data.local_element_id[i],
            locally_relevant_dofs.local_values_as_span()[model.data.lid_q1[i]]);
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
      }
    }
    visualization_writer.append_result_data_vector_with_context(
        pressure_in, Core::IO::OutputEntity::element, {"p_1"});
    visualization_writer.append_result_data_vector_with_context(
        pressure_out, Core::IO::OutputEntity::element, {"p_2"});
    visualization_writer.append_result_data_vector_with_context(
        flow_in, Core::IO::OutputEntity::element, {"q_in"});
  }
}  // namespace ReducedLung

FOUR_C_NAMESPACE_CLOSE

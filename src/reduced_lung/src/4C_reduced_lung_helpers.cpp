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


FOUR_C_NAMESPACE_OPEN

namespace ReducedLung
{

  Core::LinAlg::Map create_domain_map(
      const MPI_Comm& comm, const std::vector<Airway>& airways, const TerminalUnits& terminal_units)
  {
    std::vector<int> locally_owned_dof_indices;
    for (const auto& airway : airways)
    {
      locally_owned_dof_indices.insert(locally_owned_dof_indices.end(),
          airway.global_dof_ids.begin(), airway.global_dof_ids.end());
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

  Core::LinAlg::Map create_row_map(const MPI_Comm& comm, const std::vector<Airway>& airways,
      const TerminalUnits& terminal_units, const std::vector<Connection>& connections,
      const std::vector<Bifurcation>& bifurcations,
      const std::vector<BoundaryCondition>& boundary_conditions)
  {
    int n_local_state_equations = 0;
    for (const auto& airway : airways)
    {
      n_local_state_equations += airway.n_state_equations;
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

  Core::LinAlg::Map create_column_map(const MPI_Comm& comm, const std::vector<Airway>& airways,
      const TerminalUnits& terminal_units, const std::map<int, int>& global_dof_per_ele,
      const std::map<int, int>& first_global_dof_of_ele, const std::vector<Connection>& connections,
      const std::vector<Bifurcation>& bifurcations,
      const std::vector<BoundaryCondition>& boundary_conditions)
  {
    // Vector for intermediate storage of necessary dof ids
    std::vector<int> locally_relevant_dof_indices;

    // Loop over all elements and add their global dof ids
    for (const auto& airway : airways)
    {
      locally_relevant_dof_indices.insert(locally_relevant_dof_indices.end(),
          airway.global_dof_ids.begin(), airway.global_dof_ids.end());
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

  void collect_runtime_output_data(
      Core::IO::DiscretizationVisualizationWriterMesh& visualization_writer,
      const std::vector<Airway>& airways, const TerminalUnits& terminal_units,
      const Core::LinAlg::Vector<double>& locally_relevant_dofs,
      const Core::LinAlg::Map* element_row_map)
  {
    Core::LinAlg::Vector<double> pressure_in(*element_row_map, true);
    Core::LinAlg::Vector<double> pressure_out(*element_row_map, true);
    Core::LinAlg::Vector<double> flow_in(*element_row_map, true);
    for (const auto& airway : airways)
    {
      [[maybe_unused]] int err = pressure_in.replace_local_value(
          airway.local_element_id, locally_relevant_dofs[airway.local_dof_ids[p_in]]);
      FOUR_C_ASSERT(err == 0,
          "Internal error: replace_local_value for runtime output (p_in from airways) did not "
          "work.");
      err = pressure_out.replace_local_value(
          airway.local_element_id, locally_relevant_dofs[airway.local_dof_ids[p_out]]);
      FOUR_C_ASSERT(err == 0,
          "Internal error: replace_local_value for runtime output (p_out from airways) did not "
          "work.");
      err = flow_in.replace_local_value(
          airway.local_element_id, locally_relevant_dofs[airway.local_dof_ids[q_in]]);
      FOUR_C_ASSERT(err == 0,
          "Internal error: replace_local_value for runtime output (q_in from airways) did not "
          "work.");
    }
    for (const auto& model : terminal_units.models)
    {
      for (size_t i = 0; i < model.data.number_of_elements(); i++)
      {
        [[maybe_unused]] int err = pressure_in.replace_local_value(
            model.data.local_element_id[i], locally_relevant_dofs[model.data.lid_p1[i]]);
        FOUR_C_ASSERT(err == 0,
            "Internal error: replace_local_value for runtime output (p_in from terminal units) did "
            "not work.");
        err = pressure_out.replace_local_value(
            model.data.local_element_id[i], locally_relevant_dofs[model.data.lid_p2[i]]);
        FOUR_C_ASSERT(err == 0,
            "Internal error: replace_local_value for runtime output (p_out from terminal units) "
            "did not work.");
        err = flow_in.replace_local_value(
            model.data.local_element_id[i], locally_relevant_dofs[model.data.lid_q[i]]);
        FOUR_C_ASSERT(err == 0,
            "Internal error: replace_local_value for runtime output (q_in from terminal units) did "
            "not work.");
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

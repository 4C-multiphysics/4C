// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_reduced_lung_junctions.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_linalg_map.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_reduced_lung_tree_linearization.hpp"
#include "4C_utils_exceptions.hpp"

#include <array>

FOUR_C_NAMESPACE_OPEN

namespace ReducedLung
{
  namespace Junctions
  {
    namespace
    {
      void validate_connection_candidate(const ConnectionData& connections,
          const BifurcationData& bifurcations, const std::array<int, 2>& global_ele_ids)
      {
        for (size_t i = 0; i < connections.size(); ++i)
        {
          FOUR_C_ASSERT_ALWAYS((connections.global_parent_element_id[i] != global_ele_ids[1] ||
                                   connections.global_child_element_id[i] != global_ele_ids[0]),
              "Connection instantiated twice! Check node ordering in input file.");
          FOUR_C_ASSERT_ALWAYS(connections.global_parent_element_id[i] != global_ele_ids[0],
              "Second connection entity at parent element! Check input file.");
        }
        for (size_t i = 0; i < bifurcations.size(); ++i)
        {
          FOUR_C_ASSERT_ALWAYS(
              (((bifurcations.global_parent_element_id[i] != global_ele_ids[1]) ||
                   (bifurcations.global_child_1_element_id[i] != global_ele_ids[0] &&
                       bifurcations.global_child_2_element_id[i] != global_ele_ids[0])) &&
                  (bifurcations.global_parent_element_id[i] != global_ele_ids[0])),
              "Bifurcation and connection instantiated at the same node! Check input file.");
        }
      }

      void validate_bifurcation_candidate(const ConnectionData& connections,
          const BifurcationData& bifurcations, const std::array<int, 3>& global_ele_ids)
      {
        for (size_t i = 0; i < bifurcations.size(); ++i)
        {
          FOUR_C_ASSERT_ALWAYS(
              ((bifurcations.global_parent_element_id[i] != global_ele_ids[1] &&
                   bifurcations.global_parent_element_id[i] != global_ele_ids[2]) ||
                  (bifurcations.global_child_1_element_id[i] != global_ele_ids[0] &&
                      bifurcations.global_child_2_element_id[i] != global_ele_ids[0])),
              "Bifurcation instantiated twice! Check node ordering in input file.");
          FOUR_C_ASSERT_ALWAYS(bifurcations.global_parent_element_id[i] != global_ele_ids[0],
              "Second bifurcation entity at parent element! Check input file.");
        }
        for (size_t i = 0; i < connections.size(); ++i)
        {
          FOUR_C_ASSERT_ALWAYS(((connections.global_parent_element_id[i] != global_ele_ids[1] ||
                                    connections.global_child_element_id[i] != global_ele_ids[0]) &&
                                   connections.global_parent_element_id[i] != global_ele_ids[0]),
              "Connection and bifurcation instantiated at the same node! Check input file.");
        }
      }
    }  // namespace

    void ConnectionData::clear()
    {
      first_global_equation_id.clear();
      first_local_equation_id.clear();
      local_connection_id.clear();
      global_parent_element_id.clear();
      global_child_element_id.clear();
      global_dof_ids.clear();
      local_dof_ids.clear();
      first_row.clear();
      p_out_parent_lid.clear();
      p_in_child_lid.clear();
      q_out_parent_lid.clear();
      q_in_child_lid.clear();
    }

    void ConnectionData::reserve(size_t count)
    {
      first_global_equation_id.reserve(count);
      first_local_equation_id.reserve(count);
      local_connection_id.reserve(count);
      global_parent_element_id.reserve(count);
      global_child_element_id.reserve(count);
      global_dof_ids.reserve(count);
      local_dof_ids.reserve(count);
      first_row.reserve(count);
      p_out_parent_lid.reserve(count);
      p_in_child_lid.reserve(count);
      q_out_parent_lid.reserve(count);
      q_in_child_lid.reserve(count);
    }

    void ConnectionData::add_connection(
        int local_id, int global_parent_id, int global_child_id, const std::array<int, 4>& dof_ids)
    {
      first_global_equation_id.push_back(0);
      first_local_equation_id.push_back(0);
      local_connection_id.push_back(local_id);
      global_parent_element_id.push_back(global_parent_id);
      global_child_element_id.push_back(global_child_id);
      global_dof_ids.push_back(dof_ids);
      local_dof_ids.push_back(std::array<int, 4>{});
      first_row.push_back(0);
      p_out_parent_lid.push_back(0);
      p_in_child_lid.push_back(0);
      q_out_parent_lid.push_back(0);
      q_in_child_lid.push_back(0);
    }

    void BifurcationData::clear()
    {
      first_global_equation_id.clear();
      first_local_equation_id.clear();
      local_bifurcation_id.clear();
      global_parent_element_id.clear();
      global_child_1_element_id.clear();
      global_child_2_element_id.clear();
      global_dof_ids.clear();
      local_dof_ids.clear();
      first_row.clear();
      p_out_parent_lid.clear();
      p_in_child_1_lid.clear();
      p_in_child_2_lid.clear();
      q_out_parent_lid.clear();
      q_in_child_1_lid.clear();
      q_in_child_2_lid.clear();
    }

    void BifurcationData::reserve(size_t count)
    {
      first_global_equation_id.reserve(count);
      first_local_equation_id.reserve(count);
      local_bifurcation_id.reserve(count);
      global_parent_element_id.reserve(count);
      global_child_1_element_id.reserve(count);
      global_child_2_element_id.reserve(count);
      global_dof_ids.reserve(count);
      local_dof_ids.reserve(count);
      first_row.reserve(count);
      p_out_parent_lid.reserve(count);
      p_in_child_1_lid.reserve(count);
      p_in_child_2_lid.reserve(count);
      q_out_parent_lid.reserve(count);
      q_in_child_1_lid.reserve(count);
      q_in_child_2_lid.reserve(count);
    }

    void BifurcationData::add_bifurcation(int local_id, int global_parent_id, int global_child_1_id,
        int global_child_2_id, const std::array<int, 6>& dof_ids)
    {
      first_global_equation_id.push_back(0);
      first_local_equation_id.push_back(0);
      local_bifurcation_id.push_back(local_id);
      global_parent_element_id.push_back(global_parent_id);
      global_child_1_element_id.push_back(global_child_1_id);
      global_child_2_element_id.push_back(global_child_2_id);
      global_dof_ids.push_back(dof_ids);
      local_dof_ids.push_back(std::array<int, 6>{});
      first_row.push_back(0);
      p_out_parent_lid.push_back(0);
      p_in_child_1_lid.push_back(0);
      p_in_child_2_lid.push_back(0);
      q_out_parent_lid.push_back(0);
      q_in_child_1_lid.push_back(0);
      q_in_child_2_lid.push_back(0);
    }

    void create_junctions(const Core::FE::Discretization& discretization,
        const std::map<int, std::vector<int>>& global_ele_ids_per_node,
        const std::map<int, int>& global_dof_per_ele,
        const std::map<int, int>& first_global_dof_of_ele, ConnectionData& connections,
        BifurcationData& bifurcations)
    {
      connections.clear();
      bifurcations.clear();
      int local_connection_id = 0;
      int local_bifurcation_id = 0;

      for (auto ele : discretization.my_row_element_range())
      {
        const auto nodes = ele.nodes();
        auto node_out = nodes[1];
        const auto node_out_it = global_ele_ids_per_node.find(node_out.global_id());
        FOUR_C_ASSERT_ALWAYS(node_out_it != global_ele_ids_per_node.end(),
            "Node {} missing from element adjacency map.", node_out.global_id());

        const auto& node_out_elements = node_out_it->second;
        const auto node_out_n_eles = node_out_elements.size();

        if (node_out_n_eles == 2)
        {
          std::array<int, 2> global_ele_ids{node_out_elements[0], node_out_elements[1]};
          validate_connection_candidate(connections, bifurcations, global_ele_ids);
          // dofs: {p2_parent, p1_child, q2_parent (q for non-compliant airways), q1_child}
          std::array<int, 4> global_dof_ids{first_global_dof_of_ele.at(global_ele_ids[0]) + 1,
              first_global_dof_of_ele.at(global_ele_ids[1]),
              first_global_dof_of_ele.at(global_ele_ids[0]) +
                  global_dof_per_ele.at(global_ele_ids[0]) - 1,
              first_global_dof_of_ele.at(global_ele_ids[1]) + 2};
          connections.add_connection(
              local_connection_id, global_ele_ids[0], global_ele_ids[1], global_dof_ids);
          local_connection_id++;
        }
        else if (node_out_n_eles == 3)
        {
          std::array<int, 3> global_ele_ids{
              node_out_elements[0], node_out_elements[1], node_out_elements[2]};
          validate_bifurcation_candidate(connections, bifurcations, global_ele_ids);
          // dofs: {p2_parent, p1_child_1, p1_child_2, q2_parent (q for non-compliant airways),
          // q1_child_1, q1_child_2}
          std::array<int, 6> global_dof_ids{first_global_dof_of_ele.at(global_ele_ids[0]) + 1,
              first_global_dof_of_ele.at(global_ele_ids[1]),
              first_global_dof_of_ele.at(global_ele_ids[2]),
              first_global_dof_of_ele.at(global_ele_ids[0]) +
                  global_dof_per_ele.at(global_ele_ids[0]) - 1,
              first_global_dof_of_ele.at(global_ele_ids[1]) + 2,
              first_global_dof_of_ele.at(global_ele_ids[2]) + 2};
          bifurcations.add_bifurcation(local_bifurcation_id, global_ele_ids[0], global_ele_ids[1],
              global_ele_ids[2], global_dof_ids);
          local_bifurcation_id++;
        }
        else if (node_out_n_eles > 3)
        {
          FOUR_C_THROW("Too many elements at junction.");
        }
      }
    }

    void assign_junction_local_equation_ids(
        ConnectionData& connections, BifurcationData& bifurcations, int& n_local_equations)
    {
      for (size_t i = 0; i < connections.size(); ++i)
      {
        // Every connection adds 1 momentum and 1 mass balance equation.
        connections.first_local_equation_id[i] = n_local_equations;
        connections.first_row[i] = n_local_equations;
        n_local_equations += 2;
      }
      for (size_t i = 0; i < bifurcations.size(); ++i)
      {
        // Every bifurcation adds 2 momentum balance equations and 1 mass balance equation.
        bifurcations.first_local_equation_id[i] = n_local_equations;
        bifurcations.first_row[i] = n_local_equations;
        n_local_equations += 3;
      }
    }

    void assign_junction_global_equation_ids(const Core::LinAlg::Map& row_map,
        ConnectionData& connections, BifurcationData& bifurcations)
    {
      for (size_t i = 0; i < connections.size(); ++i)
      {
        connections.first_global_equation_id[i] =
            row_map.gid(connections.first_local_equation_id[i]);
      }
      for (size_t i = 0; i < bifurcations.size(); ++i)
      {
        bifurcations.first_global_equation_id[i] =
            row_map.gid(bifurcations.first_local_equation_id[i]);
      }
    }

    void assign_junction_local_dof_ids(const Core::LinAlg::Map& locally_relevant_dof_map,
        ConnectionData& connections, BifurcationData& bifurcations)
    {
      for (size_t i = 0; i < connections.size(); ++i)
      {
        for (size_t j = 0; j < connections.global_dof_ids[i].size(); ++j)
        {
          connections.local_dof_ids[i][j] =
              locally_relevant_dof_map.lid(connections.global_dof_ids[i][j]);
        }
        connections.p_out_parent_lid[i] =
            connections.local_dof_ids[i][ConnectionData::p_out_parent];
        connections.p_in_child_lid[i] = connections.local_dof_ids[i][ConnectionData::p_in_child];
        connections.q_out_parent_lid[i] =
            connections.local_dof_ids[i][ConnectionData::q_out_parent];
        connections.q_in_child_lid[i] = connections.local_dof_ids[i][ConnectionData::q_in_child];
      }
      for (size_t i = 0; i < bifurcations.size(); ++i)
      {
        for (size_t j = 0; j < bifurcations.global_dof_ids[i].size(); ++j)
        {
          bifurcations.local_dof_ids[i][j] =
              locally_relevant_dof_map.lid(bifurcations.global_dof_ids[i][j]);
        }
        bifurcations.p_out_parent_lid[i] =
            bifurcations.local_dof_ids[i][BifurcationData::p_out_parent];
        bifurcations.p_in_child_1_lid[i] =
            bifurcations.local_dof_ids[i][BifurcationData::p_in_child_1];
        bifurcations.p_in_child_2_lid[i] =
            bifurcations.local_dof_ids[i][BifurcationData::p_in_child_2];
        bifurcations.q_out_parent_lid[i] =
            bifurcations.local_dof_ids[i][BifurcationData::q_out_parent];
        bifurcations.q_in_child_1_lid[i] =
            bifurcations.local_dof_ids[i][BifurcationData::q_in_child_1];
        bifurcations.q_in_child_2_lid[i] =
            bifurcations.local_dof_ids[i][BifurcationData::q_in_child_2];
      }
    }

    void update_residual_vector(Core::LinAlg::Vector<double>& rhs,
        const ConnectionData& connections, const BifurcationData& bifurcations,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs)
    {
      auto residual_values = rhs.local_values_as_span();
      const auto dof_values = locally_relevant_dofs.local_values_as_span();
      const auto& connection_first_row = connections.first_row;
      const auto& connection_p_out_parent_lid = connections.p_out_parent_lid;
      const auto& connection_p_in_child_lid = connections.p_in_child_lid;
      const auto& connection_q_out_parent_lid = connections.q_out_parent_lid;
      const auto& connection_q_in_child_lid = connections.q_in_child_lid;
      for (size_t i = 0; i < connections.size(); ++i)
      {
        const int first_row = connection_first_row[i];
        const double pressure_res =
            dof_values[connection_p_out_parent_lid[i]] - dof_values[connection_p_in_child_lid[i]];
        residual_values[static_cast<std::size_t>(first_row)] = pressure_res;

        const double flow_res =
            dof_values[connection_q_out_parent_lid[i]] - dof_values[connection_q_in_child_lid[i]];
        residual_values[static_cast<std::size_t>(first_row + 1)] = flow_res;
      }

      const auto& bifurcation_first_row = bifurcations.first_row;
      const auto& bifurcation_p_out_parent_lid = bifurcations.p_out_parent_lid;
      const auto& bifurcation_p_in_child_1_lid = bifurcations.p_in_child_1_lid;
      const auto& bifurcation_p_in_child_2_lid = bifurcations.p_in_child_2_lid;
      const auto& bifurcation_q_out_parent_lid = bifurcations.q_out_parent_lid;
      const auto& bifurcation_q_in_child_1_lid = bifurcations.q_in_child_1_lid;
      const auto& bifurcation_q_in_child_2_lid = bifurcations.q_in_child_2_lid;
      for (size_t i = 0; i < bifurcations.size(); ++i)
      {
        const int first_row = bifurcation_first_row[i];
        const double child_1_pressure_res = dof_values[bifurcation_p_out_parent_lid[i]] -
                                            dof_values[bifurcation_p_in_child_1_lid[i]];
        residual_values[static_cast<std::size_t>(first_row)] = child_1_pressure_res;

        const double child_2_pressure_res = dof_values[bifurcation_p_out_parent_lid[i]] -
                                            dof_values[bifurcation_p_in_child_2_lid[i]];
        residual_values[static_cast<std::size_t>(first_row + 1)] = child_2_pressure_res;

        const double flow_res = dof_values[bifurcation_q_out_parent_lid[i]] -
                                dof_values[bifurcation_q_in_child_1_lid[i]] -
                                dof_values[bifurcation_q_in_child_2_lid[i]];
        residual_values[static_cast<std::size_t>(first_row + 2)] = flow_res;
      }
    }

    void update_jacobian(Core::LinAlg::SparseMatrix& sysmat, const ConnectionData& connections,
        const BifurcationData& bifurcations)
    {
      if (sysmat.filled())
      {
        return;
      }

      std::array<double, 2> vals_momentum{1.0, -1.0};
      std::array<double, 3> vals_mass{1.0, -1.0, -1.0};

      for (size_t i = 0; i < connections.size(); ++i)
      {
        const auto& local_dof_ids = connections.local_dof_ids[i];
        std::array<int, 2> local_ids{
            local_dof_ids[ConnectionData::p_out_parent], local_dof_ids[ConnectionData::p_in_child]};
        sysmat.insert_my_values(connections.first_local_equation_id[i], vals_momentum.size(),
            vals_momentum.data(), local_ids.data());

        local_ids = {
            local_dof_ids[ConnectionData::q_out_parent], local_dof_ids[ConnectionData::q_in_child]};
        sysmat.insert_my_values(connections.first_local_equation_id[i] + 1,
            vals_momentum.size() /*mass balance*/, vals_momentum.data() /*mass balance*/,
            local_ids.data());
      }

      for (size_t i = 0; i < bifurcations.size(); ++i)
      {
        const auto& local_dof_ids = bifurcations.local_dof_ids[i];
        std::array<int, 2> local_ids_mom_balance{local_dof_ids[BifurcationData::p_out_parent],
            local_dof_ids[BifurcationData::p_in_child_1]};
        sysmat.insert_my_values(bifurcations.first_local_equation_id[i], vals_momentum.size(),
            vals_momentum.data(), local_ids_mom_balance.data());

        local_ids_mom_balance = {local_dof_ids[BifurcationData::p_out_parent],
            local_dof_ids[BifurcationData::p_in_child_2]};
        sysmat.insert_my_values(bifurcations.first_local_equation_id[i] + 1, vals_momentum.size(),
            vals_momentum.data(), local_ids_mom_balance.data());

        std::array<int, 3> local_ids_mass_balance = {local_dof_ids[BifurcationData::q_out_parent],
            local_dof_ids[BifurcationData::q_in_child_1],
            local_dof_ids[BifurcationData::q_in_child_2]};
        sysmat.insert_my_values(bifurcations.first_local_equation_id[i] + 2, vals_mass.size(),
            vals_mass.data(), local_ids_mass_balance.data());
      }
    }

    void update_tree_linearization(TreeCoefficientAssemblyTarget& target,
        const ConnectionData& connections, const BifurcationData& bifurcations)
    {
      /* Junction equations are linear and state-independent; only constant coefficients are stored.
       */
      for (size_t i = 0; i < connections.size(); ++i)
      {
        const auto& local_dof_ids = connections.local_dof_ids[i];
        const int pressure_row = connections.first_local_equation_id[i];
        const int flow_row = pressure_row + 1;

        target.append_value(pressure_row, local_dof_ids[ConnectionData::p_out_parent], 1.0);
        target.append_value(pressure_row, local_dof_ids[ConnectionData::p_in_child], -1.0);
        target.append_value(flow_row, local_dof_ids[ConnectionData::q_out_parent], 1.0);
        target.append_value(flow_row, local_dof_ids[ConnectionData::q_in_child], -1.0);
      }

      for (size_t i = 0; i < bifurcations.size(); ++i)
      {
        const auto& local_dof_ids = bifurcations.local_dof_ids[i];
        const int child_1_pressure_row = bifurcations.first_local_equation_id[i];
        const int child_2_pressure_row = child_1_pressure_row + 1;
        const int flow_row = child_1_pressure_row + 2;

        target.append_value(
            child_1_pressure_row, local_dof_ids[BifurcationData::p_out_parent], 1.0);
        target.append_value(
            child_1_pressure_row, local_dof_ids[BifurcationData::p_in_child_1], -1.0);
        target.append_value(
            child_2_pressure_row, local_dof_ids[BifurcationData::p_out_parent], 1.0);
        target.append_value(
            child_2_pressure_row, local_dof_ids[BifurcationData::p_in_child_2], -1.0);
        target.append_value(flow_row, local_dof_ids[BifurcationData::q_out_parent], 1.0);
        target.append_value(flow_row, local_dof_ids[BifurcationData::q_in_child_1], -1.0);
        target.append_value(flow_row, local_dof_ids[BifurcationData::q_in_child_2], -1.0);
      }
    }
  }  // namespace Junctions
}  // namespace ReducedLung

FOUR_C_NAMESPACE_CLOSE

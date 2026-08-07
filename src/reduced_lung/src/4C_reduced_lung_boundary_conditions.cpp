// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_reduced_lung_boundary_conditions.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_linalg_map.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_function_manager.hpp"
#include "4C_utils_function_of_time.hpp"

#include <map>
#include <ranges>
#include <tuple>
#include <utility>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace ReducedLung
{
  namespace BoundaryConditions
  {
    namespace
    {
      struct ModelKey
      {
        Type type;
        int function_id;

        bool operator<(const ModelKey& other) const
        {
          return std::tie(type, function_id) < std::tie(other.type, other.function_id);
        }
      };

      const char* bc_type_label(Type type)
      {
        switch (type)
        {
          case Type::Pressure:
            return "pressure";
          case Type::Flow:
            return "flow";
        }
        return "unknown";
      }

      void evaluate_function_value(const BoundaryConditionModel& model,
          Core::LinAlg::Vector<double>& rhs, const Core::LinAlg::Vector<double>& dofs, double time)
      {
        if (model.function == nullptr)
        {
          FOUR_C_THROW(
              "Boundary condition function not set for bc_function_id {}.", model.function_id);
        }
        const double bc_value = model.function->evaluate(time);
        for (size_t i = 0; i < model.data.size(); ++i)
        {
          const int local_dof_id = model.data.local_dof_id[i];
          const double res = dofs.local_values_as_span()[local_dof_id] - bc_value;
          rhs.replace_local_value(model.data.local_equation_id[i], res);
        }
      }

      /**
       * The definitions of the input file and the `bc_id`s of the mesh must describe the same set
       * of boundary conditions: every id used by the mesh needs a definition, and every definition
       * needs at least one node. Definition ids must be unique and positive across all condition
       * types combined, since 0 is reserved for unconstrained nodes.
       *
       * @return The definitions, indexed by their id.
       */
      std::map<int, std::pair<Type, const ReducedLungParameters::BoundaryConditions::Definition*>>
      index_boundary_condition_definitions(
          const ReducedLungParameters::BoundaryConditions& bc_parameters,
          const std::map<int, std::vector<int>>& bc_nodes)
      {
        std::map<int, std::pair<Type, const ReducedLungParameters::BoundaryConditions::Definition*>>
            definitions;

        const auto index_definitions_of_type =
            [&](Type type, const std::vector<ReducedLungParameters::BoundaryConditions::Definition>&
                               type_definitions)
        {
          for (const auto& definition : type_definitions)
          {
            if (definition.id <= 0)
            {
              FOUR_C_THROW(
                  "Boundary condition definition ids must be positive, got {}. The id 0 marks "
                  "nodes without a boundary condition and cannot be defined.",
                  definition.id);
            }
            if (!definitions.emplace(definition.id, std::pair{type, &definition}).second)
            {
              FOUR_C_THROW(
                  "Boundary condition definition id {} is defined more than once.", definition.id);
            }
            if (definition.function_id <= 0)
            {
              FOUR_C_THROW(
                  "The function_id of boundary condition definition {} must be positive, got {}.",
                  definition.id, definition.function_id);
            }
            if (!bc_nodes.contains(definition.id))
            {
              FOUR_C_THROW(
                  "Boundary condition definition {} is not used by any node of the mesh. Assign "
                  "it by setting 'bc_id' to {} on the respective nodes.",
                  definition.id, definition.id);
            }
          }
        };

        index_definitions_of_type(Type::Pressure, bc_parameters.pressure);
        index_definitions_of_type(Type::Flow, bc_parameters.flow);

        for (const int bc_id : bc_nodes | std::views::keys)
        {
          if (!definitions.contains(bc_id))
          {
            FOUR_C_THROW(
                "The mesh assigns the boundary condition id {} to some of its nodes, but no "
                "definition with this id exists in the input file.",
                bc_id);
          }
        }

        return definitions;
      }

      void assemble_diagonal_jacobian(
          const BoundaryConditionModel& model, Core::LinAlg::SparseMatrix& sysmat)
      {
        const double val = 1.0;
        for (size_t i = 0; i < model.data.size(); ++i)
        {
          const int local_dof_id = model.data.local_dof_id[i];
          sysmat.insert_my_values(model.data.local_equation_id[i], 1, &val, &local_dof_id);
        }
      }
    }  // namespace

    void BoundaryConditionData::clear()
    {
      node_id.clear();
      global_element_id.clear();
      input_bc_id.clear();
      local_bc_id.clear();
      local_equation_id.clear();
      global_equation_id.clear();
      global_dof_id.clear();
      local_dof_id.clear();
    }

    void BoundaryConditionData::reserve(size_t count)
    {
      node_id.reserve(count);
      global_element_id.reserve(count);
      input_bc_id.reserve(count);
      local_bc_id.reserve(count);
      local_equation_id.reserve(count);
      global_equation_id.reserve(count);
      global_dof_id.reserve(count);
      local_dof_id.reserve(count);
    }

    void BoundaryConditionData::add_entry(int node_id_value, int element_id_value,
        int local_bc_id_value, int global_dof_id_value, int input_bc_id_value)
    {
      node_id.push_back(node_id_value);
      global_element_id.push_back(element_id_value);
      input_bc_id.push_back(input_bc_id_value);
      local_bc_id.push_back(local_bc_id_value);
      local_equation_id.push_back(0);
      global_equation_id.push_back(0);
      global_dof_id.push_back(global_dof_id_value);
      local_dof_id.push_back(0);
    }

    void BoundaryConditionModel::add_condition(int node_id_value, int element_id_value,
        int local_bc_id_value, int global_dof_id_value, int input_bc_id_value)
    {
      data.add_entry(node_id_value, element_id_value, local_bc_id_value, global_dof_id_value,
          input_bc_id_value);
    }

    void create_boundary_conditions(const Core::FE::Discretization& discretization,
        const ReducedLungParameters& parameters, const std::map<int, std::vector<int>>& bc_nodes,
        const std::map<int, std::vector<int>>& global_ele_ids_per_node,
        const std::map<int, int>& global_dof_per_ele,
        const std::map<int, int>& first_global_dof_of_ele,
        const Core::Utils::FunctionManager& function_manager,
        BoundaryConditionContainer& boundary_conditions)
    {
      boundary_conditions.models.clear();

      const auto& bc_parameters = parameters.boundary_conditions;
      const auto definitions = index_boundary_condition_definitions(bc_parameters, bc_nodes);

      std::map<ModelKey, size_t> model_indices;
      std::map<std::pair<int, Type>, int> bc_per_node_and_type;
      int local_bc_id = 0;

      // Walk the constrained nodes grouped by the definition they refer to. Since a node carries
      // exactly one `bc_id`, it can never appear in more than one group.
      for (const auto& [bc_id, nodes] : bc_nodes)
      {
        const auto& [bc_type, definition] = definitions.at(bc_id);

        const int function_id = definition->function_id;
        const ModelKey key{.type = bc_type, .function_id = function_id};

        for (const int node_id : nodes)
        {
          auto node_it = global_ele_ids_per_node.find(node_id);
          if (node_it == global_ele_ids_per_node.end())
          {
            FOUR_C_THROW(
                "Node {} with boundary condition id {} is not part of the tree.", node_id, bc_id);
          }
          const auto& adjacent_elements = node_it->second;
          if (adjacent_elements.size() != 1u)
          {
            FOUR_C_THROW(
                "Node {} with boundary condition id {} must connect to exactly one element, but "
                "connects to {} elements.",
                node_id, bc_id, adjacent_elements.size());
          }

          const int element_id = adjacent_elements.front();
          const int local_element_id = discretization.element_row_map()->lid(element_id);
          if (local_element_id == -1)
          {
            // Only own boundary conditions when the attached element is owned by this rank.
            continue;
          }
          auto* ele = discretization.l_row_element(local_element_id);
          const auto node_ids = ele->node_ids();
          const bool is_inlet = node_ids[0] == node_id;
          const bool is_outlet = node_ids[1] == node_id;
          if (!is_inlet && !is_outlet)
          {
            FOUR_C_THROW(
                "Node {} with boundary condition id {} is not attached to element {} as inlet or "
                "outlet.",
                node_id, bc_id, element_id);
          }

          // Enforce at most one boundary condition per (node, type). For a `bc_nodes` derived from
          // the mesh this can never trigger, since a node carries exactly one `bc_id` and therefore
          // appears in exactly one group. It guards callers that assemble `bc_nodes` themselves,
          // which the unit tests do.
          const auto duplicate_key = std::make_pair(node_id, bc_type);
          auto duplicate_it = bc_per_node_and_type.find(duplicate_key);
          if (duplicate_it != bc_per_node_and_type.end())
          {
            FOUR_C_THROW(
                "Multiple {} boundary conditions assigned to node {} (definitions {} and {}).",
                bc_type_label(bc_type), node_id, duplicate_it->second, bc_id);
          }
          bc_per_node_and_type.emplace(duplicate_key, bc_id);
          int dof_offset = 0;
          if (bc_type == Type::Pressure)
          {
            // Pressure uses inlet/outlet pressure dofs on the attached element.
            dof_offset = is_inlet ? 0 : 1;
          }
          else if (bc_type == Type::Flow)
          {
            if (is_inlet)
            {
              // Inlet flow dof is stored at the fixed offset 2.
              dof_offset = 2;
            }
            else
            {
              // Outlet flow dof is always the last dof of the element.
              const auto dof_it = global_dof_per_ele.find(element_id);
              FOUR_C_ASSERT_ALWAYS(dof_it != global_dof_per_ele.end(),
                  "Missing dof count for element {}.", element_id + 1);
              dof_offset = dof_it->second - 1;
            }
          }
          else
          {
            FOUR_C_THROW("Boundary condition type not implemented.");
          }

          const auto first_dof_it = first_global_dof_of_ele.find(element_id);
          FOUR_C_ASSERT_ALWAYS(first_dof_it != first_global_dof_of_ele.end(),
              "Missing dof offset for element {}.", element_id + 1);
          const int global_dof_id = first_dof_it->second + dof_offset;

          auto model_it = model_indices.find(key);
          if (model_it == model_indices.end())
          {
            // Create a new model block for this (type, function) group.
            BoundaryConditionModel model;
            model.type = bc_type;
            model.function_id = function_id;
            model.function =
                &function_manager.function_by_id<Core::Utils::FunctionOfTime>(function_id);
            boundary_conditions.models.push_back(std::move(model));
            const size_t new_index = boundary_conditions.models.size() - 1;
            model_indices.emplace(key, new_index);
            model_it = model_indices.find(key);
          }

          auto& model = boundary_conditions.models[model_it->second];
          model.add_condition(node_id, element_id, local_bc_id, global_dof_id, bc_id);
          local_bc_id++;
        }
      }
    }

    int count_boundary_conditions(const BoundaryConditionContainer& boundary_conditions)
    {
      int count = 0;
      for (const auto& model : boundary_conditions.models)
      {
        count += static_cast<int>(model.data.size());
      }
      return count;
    }

    void assign_local_equation_ids(
        BoundaryConditionContainer& boundary_conditions, int& n_local_equations)
    {
      for (auto& model : boundary_conditions.models)
      {
        for (size_t i = 0; i < model.data.size(); ++i)
        {
          model.data.local_equation_id[i] = n_local_equations;
          n_local_equations++;
        }
      }
    }

    void assign_global_equation_ids(
        const Core::LinAlg::Map& row_map, BoundaryConditionContainer& boundary_conditions)
    {
      for (auto& model : boundary_conditions.models)
      {
        for (size_t i = 0; i < model.data.size(); ++i)
        {
          model.data.global_equation_id[i] = row_map.gid(model.data.local_equation_id[i]);
        }
      }
    }

    void assign_local_dof_ids(const Core::LinAlg::Map& locally_relevant_dof_map,
        BoundaryConditionContainer& boundary_conditions)
    {
      for (auto& model : boundary_conditions.models)
      {
        for (size_t i = 0; i < model.data.size(); ++i)
        {
          model.data.local_dof_id[i] = locally_relevant_dof_map.lid(model.data.global_dof_id[i]);
        }
      }
    }

    void create_evaluators(BoundaryConditionContainer& boundary_conditions)
    {
      for (auto& model : boundary_conditions.models)
      {
        model.residual_evaluator = evaluate_function_value;
        model.jacobian_evaluator = assemble_diagonal_jacobian;
      }
    }

    void update_residual_vector(Core::LinAlg::Vector<double>& rhs,
        const BoundaryConditionContainer& boundary_conditions,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs, double time)
    {
      for (const auto& model : boundary_conditions.models)
      {
        model.residual_evaluator(model, rhs, locally_relevant_dofs, time);
      }
    }

    void update_jacobian(
        Core::LinAlg::SparseMatrix& sysmat, const BoundaryConditionContainer& boundary_conditions)
    {
      if (sysmat.filled())
      {
        return;
      }

      for (const auto& model : boundary_conditions.models)
      {
        model.jacobian_evaluator(model, sysmat);
      }
    }
  }  // namespace BoundaryConditions
}  // namespace ReducedLung

FOUR_C_NAMESPACE_CLOSE

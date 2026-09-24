// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_reduced_lung_boundary_conditions.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_linalg_map.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_reduced_lung_terminal_unit.hpp"
#include "4C_reduced_lung_tree_linearization.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_function_manager.hpp"
#include "4C_utils_function_of_time.hpp"

#include <cmath>
#include <map>
#include <ranges>
#include <set>
#include <utility>
#include <variant>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace ReducedLung
{
  namespace BoundaryConditions
  {
    namespace
    {
      using InputFromFunction = ReducedLungParameters::BoundaryConditions::FromFunctionDefinition;

      /**
       * Human-readable name of a constrained variable, for error messages.
       */
      [[nodiscard]] const char* constrained_variable_label(ConstrainedVariable variable)
      {
        switch (variable)
        {
          case ConstrainedVariable::Pressure:
            return "pressure";
          case ConstrainedVariable::Flow:
            return "flow";
        }
        return "unknown";
      }

      /**
       * Constrain the boundary dof of every entry of @p model to the value @p bc_value_at
       * returns for that entry's index.
       */
      template <typename ValueAt>
      void apply_dirichlet_residual(const BoundaryConditionModel& model,
          Core::LinAlg::Vector<double>& rhs, const Core::LinAlg::Vector<double>& dofs,
          const ValueAt& bc_value_at)
      {
        auto residual_values = rhs.local_values_as_span();
        const auto dof_values = dofs.local_values_as_span();
        const auto& local_dof_id = model.data.local_dof_id;
        const auto& local_equation_id = model.data.local_equation_id;
        for (size_t i = 0; i < model.data.size(); ++i)
        {
          const double res = dof_values[local_dof_id[i]] - bc_value_at(i);
          residual_values[static_cast<std::size_t>(local_equation_id[i])] = res;
        }
      }

      /**
       * Pleural pressure over the normalized terminal unit volume
       * xi = (V - residual_volume) / (total_lung_capacity - residual_volume), with the
       * pressure_offset looked up for @p global_element_id, which allows it to vary spatially.
       */
      [[nodiscard]] double evaluate_volume_dependent_pleural_pressure(
          const VolumeDependentPleuralPressure& pleural_pressure, double total_terminal_unit_volume,
          int global_element_id)
      {
        const auto& curve = pleural_pressure.normalized_linear_exponential;
        const double xi = (total_terminal_unit_volume - pleural_pressure.residual_volume) /
                          (pleural_pressure.total_lung_capacity - pleural_pressure.residual_volume);
        const double pressure_offset =
            curve.pressure_offset.at(global_element_id, "pressure_offset");
        return pressure_offset + curve.linear_coefficient * xi +
               curve.exponential_coefficient * (std::exp(curve.exponential_rate * xi) - 1.0);
      }

      /**
       * The value a boundary condition prescribes in the state of @p assembly_context for the
       * entry attached to @p global_element_id. One overload per alternative of ValueModel.
       */
      [[nodiscard]] double prescribed_value(const TimeFunction& time_function,
          const AssemblyContext& assembly_context, int /*global_element_id*/)
      {
        FOUR_C_ASSERT(time_function.function != nullptr,
            "Implementation error: function {} of a boundary condition was not resolved.",
            time_function.function_id);
        return time_function.function->evaluate(assembly_context.time);
      }

      [[nodiscard]] double prescribed_value(const VolumeDependentPleuralPressure& pleural_pressure,
          const AssemblyContext& assembly_context, int global_element_id)
      {
        switch (pleural_pressure.coupling)
        {
          case VolumeDependentPleuralPressure::Coupling::Frozen:
            // The context carries the volume of the last converged timestep, so the value stays
            // constant over the Newton solve.
            return evaluate_volume_dependent_pleural_pressure(
                pleural_pressure, assembly_context.total_terminal_unit_volume, global_element_id);
        }
        FOUR_C_THROW("Unknown coupling of volume-dependent pleural pressure.");
      }

      /**
       * Bind a value model to the callable that assembles its residual. The concrete alternative
       * is captured here, so the assembly loop never inspects the variant again. Each entry is
       * evaluated separately by its own global_element_id, since the value model may vary
       * spatially even within a single model.
       */
      [[nodiscard]] ResidualEvaluator make_residual_evaluator(const ValueModel& value_model)
      {
        return std::visit(
            [](const auto& value) -> ResidualEvaluator
            {
              return [value](const BoundaryConditionModel& model, Core::LinAlg::Vector<double>& rhs,
                         const Core::LinAlg::Vector<double>& dofs,
                         const AssemblyContext& assembly_context)
              {
                apply_dirichlet_residual(model, rhs, dofs,
                    [&](size_t i)
                    {
                      return prescribed_value(
                          value, assembly_context, model.data.global_element_id[i]);
                    });
              };
            },
            value_model);
      }

      /**
       * A value that does not depend on the solution contributes a single unit entry on the
       * diagonal. Uses insert_my_values(), which needs an unfilled matrix.
       */
      void assemble_unit_diagonal_jacobian(const BoundaryConditionModel& model,
          Core::LinAlg::SparseMatrix& sysmat, const Core::LinAlg::Vector<double>& /*dofs*/,
          const AssemblyContext& /*assembly_context*/)
      {
        const double val = 1.0;
        for (size_t i = 0; i < model.data.size(); ++i)
        {
          const int local_dof_id = model.data.local_dof_id[i];
          sysmat.insert_my_values(model.data.local_equation_id[i], 1, &val, &local_dof_id);
        }
      }

      /**
       * Build the value a function-valued definition prescribes.
       */
      [[nodiscard]] ValueModel make_value_model(
          const InputFromFunction& definition, const Core::Utils::FunctionManager& function_manager)
      {
        return TimeFunction{.function_id = definition.function_id,
            .function = &function_manager.function_by_id<Core::Utils::FunctionOfTime>(
                definition.function_id)};
      }

      /**
       * Build the value a pleural pressure definition prescribes.
       */
      [[nodiscard]] ValueModel make_value_model(const VolumeDependentPleuralPressure& definition)
      {
        // Relates two parameters, so the input specs cannot check it.
        if (definition.total_lung_capacity <= definition.residual_volume)
        {
          FOUR_C_THROW(
              "The volume-dependent pleural pressure definition {} requires total_lung_capacity > "
              "residual_volume, got {} <= {}.",
              definition.id, definition.total_lung_capacity, definition.residual_volume);
        }

        return definition;
      }

      /**
       * The definitions of the input file and the `bc_id`s of the mesh must describe the same set
       * of boundary conditions: every id used by the mesh needs a definition, and every definition
       * needs at least one node. Definition ids must be unique across all condition types
       * combined; that they are positive is checked by the input specs.
       *
       * Runs before the definitions are resolved, so an input/mesh mismatch is reported first.
       */
      void check_definition_ids(const ReducedLungParameters::BoundaryConditions& bc_parameters,
          const std::map<int, std::vector<int>>& bc_nodes)
      {
        std::set<int> definition_ids;

        const auto check_id = [&](int id)
        {
          if (!definition_ids.insert(id).second)
          {
            FOUR_C_THROW("Boundary condition definition id {} is defined more than once.", id);
          }
          if (!bc_nodes.contains(id))
          {
            FOUR_C_THROW(
                "Boundary condition definition {} is not used by any node of the mesh. Assign "
                "it by setting 'bc_id' to {} on the respective nodes.",
                id, id);
          }
        };

        for (const auto& definition : bc_parameters.pressure) check_id(definition.id);
        for (const auto& definition : bc_parameters.flow) check_id(definition.id);
        for (const auto& definition : bc_parameters.volume_dependent_pleural_pressure)
          check_id(definition.id);

        for (const int bc_id : bc_nodes | std::views::keys)
        {
          if (!definition_ids.contains(bc_id))
          {
            FOUR_C_THROW(
                "The mesh assigns the boundary condition id {} to some of its nodes, but no "
                "definition with this id exists in the input file.",
                bc_id);
          }
        }
      }

      /**
       * Turn every definition of the input into an empty model, indexed by definition id. The
       * caller fills in the entries of the nodes it owns. Every rank resolves every definition,
       * so a broken one fails identically everywhere.
       */
      std::map<int, BoundaryConditionModel> resolve_boundary_condition_definitions(
          const ReducedLungParameters::BoundaryConditions& bc_parameters,
          const Core::Utils::FunctionManager& function_manager)
      {
        std::map<int, BoundaryConditionModel> definitions;

        const auto add_model = [&](int id, ConstrainedVariable variable, ValueModel value_model)
        {
          BoundaryConditionModel model;
          model.definition_id = id;
          model.constrained_variable = variable;
          model.value_model = std::move(value_model);
          definitions.emplace(id, std::move(model));
        };

        const auto add_from_function_definitions =
            [&](ConstrainedVariable variable, const std::vector<InputFromFunction>& list)
        {
          for (const auto& definition : list)
          {
            add_model(definition.id, variable, make_value_model(definition, function_manager));
          }
        };

        add_from_function_definitions(ConstrainedVariable::Pressure, bc_parameters.pressure);
        add_from_function_definitions(ConstrainedVariable::Flow, bc_parameters.flow);
        for (const auto& definition : bc_parameters.volume_dependent_pleural_pressure)
        {
          add_model(definition.id, ConstrainedVariable::Pressure, make_value_model(definition));
        }

        return definitions;
      }

      /**
       * Sum the volume of all terminal units across all ranks. Collective.
       */
      [[nodiscard]] double compute_total_terminal_unit_volume(
          const TerminalUnits::TerminalUnitContainer& terminal_units, MPI_Comm comm)
      {
        double local_volume = 0.0;
        for (const auto& model : terminal_units.models)
        {
          for (const double volume : model.data.volume_v)
          {
            local_volume += volume;
          }
        }
        return Core::Communication::sum_all(local_volume, comm);
      }
    }  // namespace

    void BoundaryConditionData::clear()
    {
      node_id.clear();
      global_element_id.clear();
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
      local_bc_id.reserve(count);
      local_equation_id.reserve(count);
      global_equation_id.reserve(count);
      global_dof_id.reserve(count);
      local_dof_id.reserve(count);
    }

    void BoundaryConditionData::add_entry(
        int node_id_value, int element_id_value, int local_bc_id_value, int global_dof_id_value)
    {
      node_id.push_back(node_id_value);
      global_element_id.push_back(element_id_value);
      local_bc_id.push_back(local_bc_id_value);
      local_equation_id.push_back(0);
      global_equation_id.push_back(0);
      global_dof_id.push_back(global_dof_id_value);
      local_dof_id.push_back(0);
    }

    void BoundaryConditionModel::add_condition(
        int node_id_value, int element_id_value, int local_bc_id_value, int global_dof_id_value)
    {
      data.add_entry(node_id_value, element_id_value, local_bc_id_value, global_dof_id_value);
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
      check_definition_ids(bc_parameters, bc_nodes);
      const auto definitions =
          resolve_boundary_condition_definitions(bc_parameters, function_manager);

      // Derived from the input, not from the models: the reduction is collective, but a model
      // only exists on the rank owning its element.
      boundary_conditions.requires_total_terminal_unit_volume =
          !bc_parameters.volume_dependent_pleural_pressure.empty();

      std::map<int, size_t> model_indices;
      std::map<std::pair<int, ConstrainedVariable>, int> bc_per_node_and_variable;
      int local_bc_id = 0;

      // Walk the constrained nodes grouped by the definition they refer to. Since a node carries
      // exactly one `bc_id`, it can never appear in more than one group.
      for (const auto& [bc_id, nodes] : bc_nodes)
      {
        const auto& definition = definitions.at(bc_id);
        const auto variable = definition.constrained_variable;

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

          // Unreachable for a `bc_nodes` built from the mesh; guards callers that assemble it
          // themselves.
          const auto duplicate_key = std::make_pair(node_id, variable);
          auto duplicate_it = bc_per_node_and_variable.find(duplicate_key);
          if (duplicate_it != bc_per_node_and_variable.end())
          {
            FOUR_C_THROW(
                "Multiple {} boundary conditions assigned to node {} (definitions {} and {}).",
                constrained_variable_label(variable), node_id, duplicate_it->second, bc_id);
          }
          bc_per_node_and_variable.emplace(duplicate_key, bc_id);

          int dof_offset = 0;
          switch (variable)
          {
            case ConstrainedVariable::Pressure:
              // Pressure uses inlet/outlet pressure dofs on the attached element.
              dof_offset = is_inlet ? 0 : 1;
              break;
            case ConstrainedVariable::Flow:
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
              break;
          }

          const auto first_dof_it = first_global_dof_of_ele.find(element_id);
          FOUR_C_ASSERT_ALWAYS(first_dof_it != first_global_dof_of_ele.end(),
              "Missing dof offset for element {}.", element_id + 1);
          const int global_dof_id = first_dof_it->second + dof_offset;

          auto model_it = model_indices.find(bc_id);
          if (model_it == model_indices.end())
          {
            boundary_conditions.models.push_back(definition);
            model_it = model_indices.emplace(bc_id, boundary_conditions.models.size() - 1).first;
          }

          auto& model = boundary_conditions.models[model_it->second];
          model.add_condition(node_id, element_id, local_bc_id, global_dof_id);
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
        model.residual_evaluator = make_residual_evaluator(model.value_model);
        // Every value model implemented so far prescribes a value that is constant over the
        // Newton solve, so the equation stays a Dirichlet row inserted once. A value model that
        // depends on the current iterate must instead assemble its own row and clear the flag.
        model.jacobian_evaluator = assemble_unit_diagonal_jacobian;
        model.has_constant_jacobian = true;
      }
    }

    void update_residual_vector(Core::LinAlg::Vector<double>& rhs,
        const BoundaryConditionContainer& boundary_conditions,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs, double time)
    {
      const AssemblyContext assembly_context{.time = time,
          .total_terminal_unit_volume = boundary_conditions.total_terminal_unit_volume};

      for (const auto& model : boundary_conditions.models)
      {
        model.residual_evaluator(model, rhs, locally_relevant_dofs, assembly_context);
      }
    }

    void update_jacobian(Core::LinAlg::SparseMatrix& sysmat,
        const BoundaryConditionContainer& boundary_conditions,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs, double time)
    {
      const AssemblyContext assembly_context{.time = time,
          .total_terminal_unit_volume = boundary_conditions.total_terminal_unit_volume};

      for (const auto& model : boundary_conditions.models)
      {
        // A constant row only needs to be inserted while the matrix graph is still being built.
        if (model.has_constant_jacobian && sysmat.filled())
        {
          continue;
        }
        model.jacobian_evaluator(model, sysmat, locally_relevant_dofs, assembly_context);
      }
    }

    void refresh_total_terminal_unit_volume(BoundaryConditionContainer& boundary_conditions,
        const TerminalUnits::TerminalUnitContainer& terminal_units, MPI_Comm comm)
    {
      if (!boundary_conditions.requires_total_terminal_unit_volume)
      {
        return;
      }

      boundary_conditions.total_terminal_unit_volume =
          compute_total_terminal_unit_volume(terminal_units, comm);
    }

    void update_tree_linearization(TreeCoefficientAssemblyTarget& target,
        const BoundaryConditionContainer& boundary_conditions)
    {
      /* Boundary constraints contribute constant identity rows to the structured linearization. */
      for (const auto& model : boundary_conditions.models)
      {
        for (size_t i = 0; i < model.data.size(); ++i)
        {
          target.append_value(model.data.local_equation_id[i], model.data.local_dof_id[i], 1.0);
        }
      }
    }
  }  // namespace BoundaryConditions
}  // namespace ReducedLung

FOUR_C_NAMESPACE_CLOSE

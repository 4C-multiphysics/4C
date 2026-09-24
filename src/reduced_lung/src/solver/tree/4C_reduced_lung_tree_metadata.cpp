// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_reduced_lung_tree_metadata.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_linalg_map.hpp"
#include "4C_utils_exceptions.hpp"

#include <algorithm>
#include <array>
#include <numeric>
#include <utility>

FOUR_C_NAMESPACE_OPEN

namespace ReducedLung
{
  namespace
  {
    /**
     * Element state-equation layout used to build serial tree metadata.
     */
    struct ElementEquationMetadata
    {
      int first_local_equation_id = -1;
      int first_global_equation_id = -1;
      int num_equations = 0;
    };

    TreeElementKind map_element_kind(ReducedLungParameters::LungTree::ElementType type)
    {
      switch (type)
      {
        case ReducedLungParameters::LungTree::ElementType::Airway:
          return TreeElementKind::Airway;
        case ReducedLungParameters::LungTree::ElementType::TerminalUnit:
          return TreeElementKind::TerminalUnit;
      }
      FOUR_C_THROW("Unknown reduced-lung element type while building tree metadata.");
    }

    std::vector<int> consecutive_ids(int first_id, int count)
    {
      std::vector<int> ids(static_cast<std::size_t>(count));
      std::iota(ids.begin(), ids.end(), first_id);
      return ids;
    }

    /**
     * Map global ids to local ids on the provided map, preserving input order.
     */
    std::vector<int> local_ids_for_global_ids(
        const Core::LinAlg::Map& map, const std::vector<int>& global_ids)
    {
      std::vector<int> local_ids;
      local_ids.reserve(global_ids.size());
      for (const int global_id : global_ids)
      {
        local_ids.push_back(map.lid(global_id));
      }
      return local_ids;
    }

    void insert_element_equation_metadata(std::map<int, ElementEquationMetadata>& equation_metadata,
        int global_element_id, int first_local_equation_id, int num_equations,
        const Core::LinAlg::Map& row_map)
    {
      FOUR_C_ASSERT_ALWAYS(first_local_equation_id >= 0,
          "Missing local state-equation id for reduced-lung element {}.", global_element_id + 1);
      const ElementEquationMetadata metadata{
          .first_local_equation_id = first_local_equation_id,
          .first_global_equation_id = row_map.gid(first_local_equation_id),
          .num_equations = num_equations,
      };
      const auto insert_result = equation_metadata.emplace(global_element_id, metadata);
      FOUR_C_ASSERT_ALWAYS(insert_result.second,
          "Duplicate reduced-lung model data for element {}.", global_element_id + 1);
    }

    /**
     * Collect element state-equation row ranges from locally owned model blocks.
     */
    std::map<int, ElementEquationMetadata> collect_element_equation_metadata(
        const Airways::AirwayContainer& airways,
        const TerminalUnits::TerminalUnitContainer& terminal_units,
        const Core::LinAlg::Map& row_map)
    {
      std::map<int, ElementEquationMetadata> equation_metadata;

      for (const auto& model : airways.models)
      {
        const auto& data = model.data;
        for (std::size_t i = 0; i < data.global_element_id.size(); ++i)
        {
          FOUR_C_ASSERT_ALWAYS(i < data.local_row_id.size(),
              "Missing airway local row id for element {}.", data.global_element_id[i] + 1);
          insert_element_equation_metadata(equation_metadata, data.global_element_id[i],
              data.local_row_id[i], data.n_state_equations, row_map);
        }
      }

      for (const auto& model : terminal_units.models)
      {
        const auto& data = model.data;
        for (std::size_t i = 0; i < data.global_element_id.size(); ++i)
        {
          FOUR_C_ASSERT_ALWAYS(i < data.local_row_id.size(),
              "Missing terminal-unit local row id for element {}.", data.global_element_id[i] + 1);
          insert_element_equation_metadata(
              equation_metadata, data.global_element_id[i], data.local_row_id[i], 1, row_map);
        }
      }

      return equation_metadata;
    }

    int element_index(const ReducedLungTreeMetadata& metadata, int global_element_id)
    {
      const auto it = metadata.element_index_by_global_id.find(global_element_id);
      FOUR_C_ASSERT_ALWAYS(it != metadata.element_index_by_global_id.end(),
          "Reduced-lung tree metadata references unknown element {}.", global_element_id + 1);
      return it->second;
    }

    bool has_child(const TreeElementMetadata& parent, int child_index)
    {
      for (int i = 0; i < parent.child_count; ++i)
      {
        if (parent.child_element_indices[static_cast<std::size_t>(i)] == child_index)
        {
          return true;
        }
      }
      return false;
    }

    /**
     * Depth-first cycle check using temporary and final visitation colors.
     */
    void validate_acyclic_from_element(
        const ReducedLungTreeMetadata& metadata, int element_index, std::vector<int>& color)
    {
      if (color[static_cast<std::size_t>(element_index)] == 1)
      {
        FOUR_C_THROW("Reduced-lung tree topology contains a directed cycle.");
      }
      if (color[static_cast<std::size_t>(element_index)] == 2)
      {
        return;
      }

      color[static_cast<std::size_t>(element_index)] = 1;
      const auto& element = metadata.elements[static_cast<std::size_t>(element_index)];
      for (int i = 0; i < element.child_count; ++i)
      {
        validate_acyclic_from_element(
            metadata, element.child_element_indices[static_cast<std::size_t>(i)], color);
      }
      color[static_cast<std::size_t>(element_index)] = 2;
    }

    void validate_acyclic(const ReducedLungTreeMetadata& metadata)
    {
      std::vector<int> color(metadata.elements.size(), 0);
      for (std::size_t i = 0; i < metadata.elements.size(); ++i)
      {
        validate_acyclic_from_element(metadata, static_cast<int>(i), color);
      }
    }

    /**
     * Check that every element can be reached from the unique directed root element.
     */
    void validate_connected_from_root(const ReducedLungTreeMetadata& metadata)
    {
      std::vector<int> stack{metadata.root_element_index};
      std::vector<bool> visited(metadata.elements.size(), false);

      while (!stack.empty())
      {
        const int current = stack.back();
        stack.pop_back();
        if (visited[static_cast<std::size_t>(current)])
        {
          continue;
        }
        visited[static_cast<std::size_t>(current)] = true;

        const auto& element = metadata.elements[static_cast<std::size_t>(current)];
        for (int i = 0; i < element.child_count; ++i)
        {
          stack.push_back(element.child_element_indices[static_cast<std::size_t>(i)]);
        }
      }

      const bool all_visited =
          std::all_of(visited.begin(), visited.end(), [](bool value) { return value; });
      FOUR_C_ASSERT_ALWAYS(
          all_visited, "Reduced-lung tree topology is not connected from the root element.");
    }

    /**
     * Build parent/child relations from the directed element-node topology.
     */
    void build_tree_relations(ReducedLungTreeMetadata& metadata)
    {
      std::map<int, std::vector<int>> elements_starting_at_node;
      for (const auto& element : metadata.elements)
      {
        elements_starting_at_node[element.inlet_node_id].push_back(element.global_element_id);
      }

      for (std::size_t parent_index = 0; parent_index < metadata.elements.size(); ++parent_index)
      {
        auto& parent = metadata.elements[parent_index];
        const auto child_it = elements_starting_at_node.find(parent.outlet_node_id);
        if (child_it == elements_starting_at_node.end())
        {
          continue;
        }

        const auto& child_global_ids = child_it->second;
        FOUR_C_ASSERT_ALWAYS(child_global_ids.size() <= 2u,
            "Reduced-lung tree element {} has unsupported branch degree {}.",
            parent.global_element_id + 1, child_global_ids.size());

        parent.child_count = static_cast<int>(child_global_ids.size());
        for (std::size_t child_slot = 0; child_slot < child_global_ids.size(); ++child_slot)
        {
          const int child_index_value = element_index(metadata, child_global_ids[child_slot]);
          parent.child_element_indices[child_slot] = child_index_value;

          auto& child = metadata.elements[static_cast<std::size_t>(child_index_value)];
          FOUR_C_ASSERT_ALWAYS(child.parent_element_index == -1,
              "Reduced-lung tree element {} has more than one parent element.",
              child.global_element_id + 1);
          child.parent_element_index = static_cast<int>(parent_index);
        }
      }

      validate_acyclic(metadata);

      std::vector<int> root_candidates;
      for (std::size_t i = 0; i < metadata.elements.size(); ++i)
      {
        if (metadata.elements[i].parent_element_index == -1)
        {
          root_candidates.push_back(static_cast<int>(i));
        }
      }
      FOUR_C_ASSERT_ALWAYS(root_candidates.size() == 1u,
          "Expected exactly one reduced-lung tree root element, found {}.", root_candidates.size());

      metadata.root_element_index = root_candidates.front();
      metadata.root_node_id =
          metadata.elements[static_cast<std::size_t>(metadata.root_element_index)].inlet_node_id;

      for (const auto& element : metadata.elements)
      {
        FOUR_C_ASSERT_ALWAYS(element.kind != TreeElementKind::TerminalUnit || element.is_leaf(),
            "Terminal-unit element {} has child elements, which is unsupported for tree metadata.",
            element.global_element_id + 1);
      }

      validate_connected_from_root(metadata);
    }

    /**
     * Build root-to-leaf and leaf-to-root traversal layers for tree solves.
     */
    void build_layers(ReducedLungTreeMetadata& metadata)
    {
      std::vector<int> current_layer{metadata.root_element_index};
      while (!current_layer.empty())
      {
        metadata.top_down_layers.push_back(current_layer);

        std::vector<int> next_layer;
        for (const int element_index_value : current_layer)
        {
          const auto& element = metadata.elements[static_cast<std::size_t>(element_index_value)];
          for (int i = 0; i < element.child_count; ++i)
          {
            next_layer.push_back(element.child_element_indices[static_cast<std::size_t>(i)]);
          }
        }
        current_layer = std::move(next_layer);
      }

      metadata.bottom_up_layers = metadata.top_down_layers;
      std::reverse(metadata.bottom_up_layers.begin(), metadata.bottom_up_layers.end());
    }

    /**
     * Add metadata for one-child junction equations.
     */
    void add_connection_metadata(ReducedLungTreeMetadata& metadata,
        const std::map<int, int>& child_by_parent,
        const std::map<int, int>& first_global_equation_by_parent, const Core::LinAlg::Map& row_map,
        const Core::LinAlg::Map& locally_relevant_dof_map,
        std::map<int, TreeJunctionKind>& junction_kind)
    {
      for (const auto& [parent_global_id, child_global_id] : child_by_parent)
      {
        const int parent_index = element_index(metadata, parent_global_id);
        const int child_index = element_index(metadata, child_global_id);
        FOUR_C_ASSERT_ALWAYS(
            has_child(metadata.elements[static_cast<std::size_t>(parent_index)], child_index),
            "Connection metadata for parent element {} and child element {} does not match the "
            "directed topology.",
            parent_global_id + 1, child_global_id + 1);

        const auto first_global_equation_it =
            first_global_equation_by_parent.find(parent_global_id);
        FOUR_C_ASSERT_ALWAYS(first_global_equation_it != first_global_equation_by_parent.end(),
            "Missing global equation id for connection at parent element {}.",
            parent_global_id + 1);

        const auto& parent = metadata.elements[static_cast<std::size_t>(parent_index)];
        const auto& child = metadata.elements[static_cast<std::size_t>(child_index)];
        const std::vector<int> global_dof_ids{parent.first_global_dof + 1, child.first_global_dof,
            parent.first_global_dof + parent.num_dofs - 1, child.first_global_dof + 2};

        TreeJunctionMetadata junction;
        junction.kind = TreeJunctionKind::Connection;
        junction.parent_element_index = parent_index;
        junction.child_element_indices[0] = child_index;
        junction.child_count = 1;
        junction.first_global_equation_id = first_global_equation_it->second;
        junction.first_local_equation_id = row_map.lid(junction.first_global_equation_id);
        junction.num_equations = 2;
        junction.global_dof_ids = global_dof_ids;
        junction.local_dof_ids = local_ids_for_global_ids(locally_relevant_dof_map, global_dof_ids);
        metadata.junctions.push_back(junction);

        const auto insert_result =
            junction_kind.emplace(parent_index, TreeJunctionKind::Connection);
        FOUR_C_ASSERT_ALWAYS(insert_result.second,
            "Duplicate junction metadata for parent element {}.", parent_global_id + 1);
      }
    }

    /**
     * Check whether the two bifurcation children match the directed topology of the parent.
     */
    bool bifurcation_children_match(
        const TreeElementMetadata& parent, int child_1_index, int child_2_index)
    {
      return parent.child_count == 2 && has_child(parent, child_1_index) &&
             has_child(parent, child_2_index) && child_1_index != child_2_index;
    }

    /**
     * Add metadata for two-child junction equations.
     */
    void add_bifurcation_metadata(ReducedLungTreeMetadata& metadata,
        const std::map<int, int>& child_1_by_parent, const std::map<int, int>& child_2_by_parent,
        const std::map<int, int>& first_global_equation_by_parent, const Core::LinAlg::Map& row_map,
        const Core::LinAlg::Map& locally_relevant_dof_map,
        std::map<int, TreeJunctionKind>& junction_kind)
    {
      for (const auto& [parent_global_id, child_1_global_id] : child_1_by_parent)
      {
        const auto child_2_it = child_2_by_parent.find(parent_global_id);
        FOUR_C_ASSERT_ALWAYS(child_2_it != child_2_by_parent.end(),
            "Missing second child for bifurcation at parent element {}.", parent_global_id + 1);
        const int child_2_global_id = child_2_it->second;
        const int parent_index = element_index(metadata, parent_global_id);
        const int child_1_index = element_index(metadata, child_1_global_id);
        const int child_2_index = element_index(metadata, child_2_global_id);
        FOUR_C_ASSERT_ALWAYS(
            bifurcation_children_match(metadata.elements[static_cast<std::size_t>(parent_index)],
                child_1_index, child_2_index),
            "Bifurcation metadata for parent element {} does not match the directed topology.",
            parent_global_id + 1);

        const auto first_global_equation_it =
            first_global_equation_by_parent.find(parent_global_id);
        FOUR_C_ASSERT_ALWAYS(first_global_equation_it != first_global_equation_by_parent.end(),
            "Missing global equation id for bifurcation at parent element {}.",
            parent_global_id + 1);

        const auto& parent = metadata.elements[static_cast<std::size_t>(parent_index)];
        const auto& child_1 = metadata.elements[static_cast<std::size_t>(child_1_index)];
        const auto& child_2 = metadata.elements[static_cast<std::size_t>(child_2_index)];
        const std::vector<int> global_dof_ids{parent.first_global_dof + 1, child_1.first_global_dof,
            child_2.first_global_dof, parent.first_global_dof + parent.num_dofs - 1,
            child_1.first_global_dof + 2, child_2.first_global_dof + 2};

        TreeJunctionMetadata junction;
        junction.kind = TreeJunctionKind::Bifurcation;
        junction.parent_element_index = parent_index;
        junction.child_element_indices[0] = child_1_index;
        junction.child_element_indices[1] = child_2_index;
        junction.child_count = 2;
        junction.first_global_equation_id = first_global_equation_it->second;
        junction.first_local_equation_id = row_map.lid(junction.first_global_equation_id);
        junction.num_equations = 3;
        junction.global_dof_ids = global_dof_ids;
        junction.local_dof_ids = local_ids_for_global_ids(locally_relevant_dof_map, global_dof_ids);
        metadata.junctions.push_back(junction);

        const auto insert_result =
            junction_kind.emplace(parent_index, TreeJunctionKind::Bifurcation);
        FOUR_C_ASSERT_ALWAYS(insert_result.second,
            "Duplicate junction metadata for parent element {}.", parent_global_id + 1);
      }
    }

    /**
     * Ensure every non-leaf element has matching connection or bifurcation metadata.
     */
    void validate_junction_coverage(const ReducedLungTreeMetadata& metadata,
        const std::map<int, TreeJunctionKind>& junction_kind)
    {
      for (std::size_t i = 0; i < metadata.elements.size(); ++i)
      {
        const auto& element = metadata.elements[i];
        if (element.child_count == 0)
        {
          continue;
        }

        const auto junction_it = junction_kind.find(static_cast<int>(i));
        FOUR_C_ASSERT_ALWAYS(junction_it != junction_kind.end(),
            "Missing junction metadata for parent element {}.", element.global_element_id + 1);

        if (element.child_count == 1)
        {
          FOUR_C_ASSERT_ALWAYS(junction_it->second == TreeJunctionKind::Connection,
              "Parent element {} has one child but is not represented by a connection.",
              element.global_element_id + 1);
        }
        else if (element.child_count == 2)
        {
          FOUR_C_ASSERT_ALWAYS(junction_it->second == TreeJunctionKind::Bifurcation,
              "Parent element {} has two children but is not represented by a bifurcation.",
              element.global_element_id + 1);
        }
      }
    }

    /**
     * Build junction metadata from the serial connection and bifurcation rows.
     */
    void build_junction_metadata(ReducedLungTreeMetadata& metadata,
        const Junctions::ConnectionData& connections,
        const Junctions::BifurcationData& bifurcations, const Core::LinAlg::Map& row_map,
        const Core::LinAlg::Map& locally_relevant_dof_map)
    {
      std::map<int, int> connection_child_by_parent;
      std::map<int, int> connection_first_equation_by_parent;
      for (std::size_t i = 0; i < connections.size(); ++i)
      {
        const int parent_global_id = connections.global_parent_element_id[i];
        connection_child_by_parent[parent_global_id] = connections.global_child_element_id[i];
        connection_first_equation_by_parent[parent_global_id] =
            connections.first_global_equation_id[i];
      }

      std::map<int, int> bifurcation_child_1_by_parent;
      std::map<int, int> bifurcation_child_2_by_parent;
      std::map<int, int> bifurcation_first_equation_by_parent;
      for (std::size_t i = 0; i < bifurcations.size(); ++i)
      {
        const int parent_global_id = bifurcations.global_parent_element_id[i];
        bifurcation_child_1_by_parent[parent_global_id] = bifurcations.global_child_1_element_id[i];
        bifurcation_child_2_by_parent[parent_global_id] = bifurcations.global_child_2_element_id[i];
        bifurcation_first_equation_by_parent[parent_global_id] =
            bifurcations.first_global_equation_id[i];
      }

      std::map<int, TreeJunctionKind> junction_kind;
      add_connection_metadata(metadata, connection_child_by_parent,
          connection_first_equation_by_parent, row_map, locally_relevant_dof_map, junction_kind);
      add_bifurcation_metadata(metadata, bifurcation_child_1_by_parent,
          bifurcation_child_2_by_parent, bifurcation_first_equation_by_parent, row_map,
          locally_relevant_dof_map, junction_kind);
      validate_junction_coverage(metadata, junction_kind);
    }

    /**
     * Return the dof constrained by a boundary condition under the reduced-lung element layout.
     */
    int expected_boundary_dof_id(const TreeElementMetadata& element, TreeBoundarySide side,
        BoundaryConditions::ConstrainedVariable constrained_variable)
    {
      if (constrained_variable == BoundaryConditions::ConstrainedVariable::Pressure)
      {
        return element.first_global_dof + (side == TreeBoundarySide::Inlet ? 0 : 1);
      }

      if (constrained_variable == BoundaryConditions::ConstrainedVariable::Flow)
      {
        return element.first_global_dof +
               (side == TreeBoundarySide::Inlet ? 2 : element.num_dofs - 1);
      }

      FOUR_C_THROW("Unsupported reduced-lung boundary-condition type in tree metadata.");
    }

    /**
     * Determine whether a boundary node lies on the inlet or outlet side of an element.
     */
    TreeBoundarySide determine_boundary_side(const TreeElementMetadata& element, int node_id)
    {
      if (node_id == element.inlet_node_id)
      {
        return TreeBoundarySide::Inlet;
      }
      if (node_id == element.outlet_node_id)
      {
        return TreeBoundarySide::Outlet;
      }
      FOUR_C_THROW("Boundary condition node {} is not attached to element {}.", node_id + 1,
          element.global_element_id + 1);
    }

    /**
     * Build boundary-condition metadata and validate constrained dofs against topology.
     */
    void build_boundary_condition_metadata(ReducedLungTreeMetadata& metadata,
        const BoundaryConditions::BoundaryConditionContainer& boundary_conditions,
        const Core::LinAlg::Map& row_map, const Core::LinAlg::Map& locally_relevant_dof_map)
    {
      std::map<int, int> type_by_equation;
      std::map<int, int> node_by_equation;
      std::map<int, int> element_by_equation;
      std::map<int, int> dof_by_equation;
      for (const auto& model : boundary_conditions.models)
      {
        const auto& data = model.data;
        for (std::size_t i = 0; i < data.size(); ++i)
        {
          const int global_equation_id = data.global_equation_id[i];
          type_by_equation[global_equation_id] = static_cast<int>(model.constrained_variable);
          node_by_equation[global_equation_id] = data.node_id[i];
          element_by_equation[global_equation_id] = data.global_element_id[i];
          dof_by_equation[global_equation_id] = data.global_dof_id[i];
        }
      }

      for (const auto& [global_equation_id, element_global_id] : element_by_equation)
      {
        const auto type_it = type_by_equation.find(global_equation_id);
        const auto node_it = node_by_equation.find(global_equation_id);
        const auto dof_it = dof_by_equation.find(global_equation_id);
        FOUR_C_ASSERT_ALWAYS(type_it != type_by_equation.end() &&
                                 node_it != node_by_equation.end() &&
                                 dof_it != dof_by_equation.end(),
            "Incomplete boundary-condition metadata for equation {}.", global_equation_id);

        const int element_index_value = element_index(metadata, element_global_id);
        const auto& element = metadata.elements[static_cast<std::size_t>(element_index_value)];
        const auto constrained_variable =
            static_cast<BoundaryConditions::ConstrainedVariable>(type_it->second);
        const TreeBoundarySide side = determine_boundary_side(element, node_it->second);
        const int expected_dof_id = expected_boundary_dof_id(element, side, constrained_variable);
        FOUR_C_ASSERT_ALWAYS(dof_it->second == expected_dof_id,
            "Boundary condition at equation {} constrains global dof {}, but element {} side "
            "expects dof {}.",
            global_equation_id, dof_it->second, element.global_element_id + 1, expected_dof_id);

        metadata.boundary_conditions.push_back(TreeBoundaryConditionMetadata{
            .constrained_variable = constrained_variable,
            .side = side,
            .node_id = node_it->second,
            .element_index = element_index_value,
            .local_equation_id = row_map.lid(global_equation_id),
            .global_equation_id = global_equation_id,
            .global_dof_id = dof_it->second,
            .local_dof_id = locally_relevant_dof_map.lid(dof_it->second),
        });
      }
    }

    /**
     * Check whether an element side has a boundary condition attached to it.
     */
    bool has_boundary_on_element_side(
        const ReducedLungTreeMetadata& metadata, int element_index_value, TreeBoundarySide side)
    {
      return std::any_of(metadata.boundary_conditions.begin(), metadata.boundary_conditions.end(),
          [element_index_value, side](const TreeBoundaryConditionMetadata& boundary)
          { return boundary.element_index == element_index_value && boundary.side == side; });
    }

    /**
     * Validate that the root inlet and all leaf outlets close the tree system.
     */
    void validate_boundary_closure(const ReducedLungTreeMetadata& metadata)
    {
      FOUR_C_ASSERT_ALWAYS(has_boundary_on_element_side(
                               metadata, metadata.root_element_index, TreeBoundarySide::Inlet),
          "Reduced-lung tree root inlet node {} has no boundary condition.",
          metadata.root_node_id + 1);

      for (std::size_t i = 0; i < metadata.elements.size(); ++i)
      {
        const auto& element = metadata.elements[i];
        if (!element.is_leaf())
        {
          continue;
        }

        FOUR_C_ASSERT_ALWAYS(
            has_boundary_on_element_side(metadata, static_cast<int>(i), TreeBoundarySide::Outlet),
            "Reduced-lung tree leaf element {} outlet node {} has no boundary condition.",
            element.global_element_id + 1, element.outlet_node_id + 1);
      }
    }
  }  // namespace

  /**
   * Build topology, equation, dof, junction, boundary, and traversal metadata for NewtonTree.
   */
  ReducedLungTreeMetadata build_reduced_lung_tree_metadata(
      const ReducedLungTreeMetadataContext& context)
  {
    ReducedLungTreeMetadata metadata;
    FOUR_C_ASSERT_ALWAYS(Core::Communication::num_mpi_ranks(context.row_map.get_comm()) == 1,
        "Reduced-lung tree metadata is only supported for serial NewtonTree solves.");
    const int num_elements = context.discretization.num_global_elements();
    FOUR_C_ASSERT_ALWAYS(
        num_elements > 0, "Reduced-lung tree metadata requires at least one element.");
    FOUR_C_ASSERT_ALWAYS(context.element_types.size() == static_cast<std::size_t>(num_elements),
        "Reduced-lung tree metadata received {} element types for {} elements.",
        context.element_types.size(), num_elements);

    const auto equation_metadata =
        collect_element_equation_metadata(context.airways, context.terminal_units, context.row_map);

    metadata.elements.reserve(static_cast<std::size_t>(num_elements));
    for (int global_element_id = 0; global_element_id < num_elements; ++global_element_id)
    {
      const auto* discretization_element = context.discretization.g_element(global_element_id);
      FOUR_C_ASSERT_ALWAYS(discretization_element != nullptr,
          "Reduced-lung discretization is missing global element {}.", global_element_id + 1);
      const auto topology_nodes = discretization_element->node_ids();
      FOUR_C_ASSERT_ALWAYS(discretization_element->num_node() == 2,
          "Reduced-lung element {} must have 2 nodes, got {}.", global_element_id + 1,
          discretization_element->num_node());
      FOUR_C_ASSERT_ALWAYS(topology_nodes[0] != topology_nodes[1],
          "Reduced-lung element {} uses identical inlet and outlet node ids.",
          global_element_id + 1);

      const auto first_dof_it = context.first_global_dof_of_ele.find(global_element_id);
      FOUR_C_ASSERT_ALWAYS(first_dof_it != context.first_global_dof_of_ele.end(),
          "Missing first global dof id for reduced-lung element {}.", global_element_id + 1);
      const auto dof_count_it = context.global_dof_per_ele.find(global_element_id);
      FOUR_C_ASSERT_ALWAYS(dof_count_it != context.global_dof_per_ele.end(),
          "Missing global dof count for reduced-lung element {}.", global_element_id + 1);
      FOUR_C_ASSERT_ALWAYS(dof_count_it->second >= 3,
          "Reduced-lung element {} has unsupported dof count {}.", global_element_id + 1,
          dof_count_it->second);

      const auto equation_metadata_it = equation_metadata.find(global_element_id);
      FOUR_C_ASSERT_ALWAYS(equation_metadata_it != equation_metadata.end(),
          "Missing model equation metadata for reduced-lung element {}.", global_element_id + 1);
      const auto& element_equation_metadata = equation_metadata_it->second;

      const auto global_dof_ids = consecutive_ids(first_dof_it->second, dof_count_it->second);
      TreeElementMetadata element{
          .global_element_id = global_element_id,
          .kind = map_element_kind(context.element_types[global_element_id]),
          .inlet_node_id = topology_nodes[0],
          .outlet_node_id = topology_nodes[1],
          .parent_element_index = -1,
          .child_element_indices = {-1, -1},
          .child_count = 0,
          .first_global_dof = first_dof_it->second,
          .num_dofs = dof_count_it->second,
          .global_dof_ids = global_dof_ids,
          .local_dof_ids =
              local_ids_for_global_ids(context.locally_relevant_dof_map, global_dof_ids),
          .first_local_state_equation_id = element_equation_metadata.first_local_equation_id,
          .first_global_state_equation_id = element_equation_metadata.first_global_equation_id,
          .num_state_equations = element_equation_metadata.num_equations,
      };

      if (element.kind == TreeElementKind::Airway)
      {
        metadata.airway_element_indices.push_back(static_cast<int>(metadata.elements.size()));
      }
      else if (element.kind == TreeElementKind::TerminalUnit)
      {
        metadata.terminal_unit_element_indices.push_back(
            static_cast<int>(metadata.elements.size()));
      }

      metadata.element_index_by_global_id.emplace(
          global_element_id, static_cast<int>(metadata.elements.size()));
      metadata.elements.push_back(std::move(element));
    }

    metadata.num_global_dofs = 0;
    for (const auto& dof_entry : context.global_dof_per_ele)
    {
      metadata.num_global_dofs += dof_entry.second;
    }
    metadata.num_global_equations = context.row_map.num_global_elements();
    metadata.num_locally_relevant_dofs = context.locally_relevant_dof_map.num_my_elements();
    FOUR_C_ASSERT_ALWAYS(metadata.num_global_equations == metadata.num_global_dofs,
        "Reduced-lung tree metadata requires a square Newton system, got {} equation rows and {} "
        "dofs.",
        metadata.num_global_equations, metadata.num_global_dofs);

    build_tree_relations(metadata);
    build_junction_metadata(metadata, context.connections, context.bifurcations, context.row_map,
        context.locally_relevant_dof_map);
    build_boundary_condition_metadata(
        metadata, context.boundary_conditions, context.row_map, context.locally_relevant_dof_map);
    validate_boundary_closure(metadata);
    build_layers(metadata);

    return metadata;
  }
}  // namespace ReducedLung

FOUR_C_NAMESPACE_CLOSE

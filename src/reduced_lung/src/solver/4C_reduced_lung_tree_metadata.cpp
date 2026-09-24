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
#include <utility>

FOUR_C_NAMESPACE_OPEN

namespace ReducedLung
{
  namespace
  {
    /**
     * Element equation and dof layout copied from existing airway and terminal-unit data.
     */
    struct ElementModelMetadata
    {
      int first_local_equation_id = -1;
      int first_global_equation_id = -1;
      int num_equations = 0;
      std::vector<int> global_dof_ids;
      std::vector<int> local_dof_ids;
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

    /**
     * Insert model-owned metadata for one reduced-lung element and catch inconsistent setup data.
     */
    void insert_element_model_metadata(std::map<int, ElementModelMetadata>& model_metadata,
        int global_element_id, int first_local_equation_id, int num_equations,
        const std::vector<int>& global_dof_ids, const std::vector<int>& local_dof_ids,
        const Core::LinAlg::Map& row_map)
    {
      FOUR_C_ASSERT_ALWAYS(first_local_equation_id >= 0,
          "Missing local state-equation id for reduced-lung element {}.", global_element_id + 1);
      FOUR_C_ASSERT_ALWAYS(global_dof_ids.size() == local_dof_ids.size(),
          "Reduced-lung element {} has {} global dofs but {} local dofs.", global_element_id + 1,
          global_dof_ids.size(), local_dof_ids.size());
      FOUR_C_ASSERT_ALWAYS(global_dof_ids.size() >= 3,
          "Reduced-lung element {} has unsupported dof count {}.", global_element_id + 1,
          global_dof_ids.size());
      const ElementModelMetadata metadata{
          .first_local_equation_id = first_local_equation_id,
          .first_global_equation_id = row_map.gid(first_local_equation_id),
          .num_equations = num_equations,
          .global_dof_ids = global_dof_ids,
          .local_dof_ids = local_dof_ids,
      };
      const auto insert_result = model_metadata.emplace(global_element_id, metadata);
      FOUR_C_ASSERT_ALWAYS(insert_result.second,
          "Duplicate reduced-lung model data for element {}.", global_element_id + 1);
    }

    /**
     * Collect element row and dof layouts from locally owned airway and terminal-unit blocks.
     */
    std::map<int, ElementModelMetadata> collect_element_model_metadata(
        const Airways::AirwayContainer& airways,
        const TerminalUnits::TerminalUnitContainer& terminal_units,
        const Core::LinAlg::Map& row_map)
    {
      std::map<int, ElementModelMetadata> model_metadata;

      for (const auto& model : airways.models)
      {
        const auto& data = model.data;
        for (std::size_t i = 0; i < data.global_element_id.size(); ++i)
        {
          FOUR_C_ASSERT_ALWAYS(i < data.local_row_id.size(),
              "Missing airway local row id for element {}.", data.global_element_id[i] + 1);
          std::vector<int> global_dof_ids{data.gid_p1[i], data.gid_p2[i], data.gid_q1[i]};
          std::vector<int> local_dof_ids{data.lid_p1[i], data.lid_p2[i], data.lid_q1[i]};
          if (data.n_state_equations == 2)
          {
            global_dof_ids.push_back(data.gid_q2[i]);
            local_dof_ids.push_back(data.lid_q2[i]);
          }
          insert_element_model_metadata(model_metadata, data.global_element_id[i],
              data.local_row_id[i], data.n_state_equations, global_dof_ids, local_dof_ids, row_map);
        }
      }

      for (const auto& model : terminal_units.models)
      {
        const auto& data = model.data;
        for (std::size_t i = 0; i < data.global_element_id.size(); ++i)
        {
          FOUR_C_ASSERT_ALWAYS(i < data.local_row_id.size(),
              "Missing terminal-unit local row id for element {}.", data.global_element_id[i] + 1);
          const std::vector<int> global_dof_ids{data.gid_p1[i], data.gid_p2[i], data.gid_q[i]};
          const std::vector<int> local_dof_ids{data.lid_p1[i], data.lid_p2[i], data.lid_q[i]};
          insert_element_model_metadata(model_metadata, data.global_element_id[i],
              data.local_row_id[i], 1, global_dof_ids, local_dof_ids, row_map);
        }
      }

      return model_metadata;
    }

    int element_index(const ReducedLungTreeMetadata& metadata, int global_element_id)
    {
      const auto it = metadata.element_index_by_global_id.find(global_element_id);
      FOUR_C_ASSERT_ALWAYS(it != metadata.element_index_by_global_id.end(),
          "Reduced-lung tree metadata references unknown element {}.", global_element_id + 1);
      return it->second;
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
     * Convert one junction parent-child relation into solver-local element indices.
     */
    void add_child_relation(
        ReducedLungTreeMetadata& metadata, int parent_index, int child_index, int parent_global_id)
    {
      auto& parent = metadata.elements[static_cast<std::size_t>(parent_index)];
      FOUR_C_ASSERT_ALWAYS(parent.child_count < 2,
          "Reduced-lung tree element {} has unsupported branch degree {}.", parent_global_id + 1,
          parent.child_count + 1);
      for (int i = 0; i < parent.child_count; ++i)
      {
        FOUR_C_ASSERT_ALWAYS(
            parent.child_element_indices[static_cast<std::size_t>(i)] != child_index,
            "Duplicate junction child metadata for parent element {} and child element {}.",
            parent_global_id + 1,
            metadata.elements[static_cast<std::size_t>(child_index)].global_element_id + 1);
      }

      parent.child_element_indices[static_cast<std::size_t>(parent.child_count)] = child_index;
      parent.child_count++;

      auto& child = metadata.elements[static_cast<std::size_t>(child_index)];
      FOUR_C_ASSERT_ALWAYS(child.parent_element_index == -1,
          "Reduced-lung tree element {} has more than one parent element.",
          child.global_element_id + 1);
      child.parent_element_index = parent_index;
    }

    /**
     * Validate the junction-derived topology assumptions required by the tree solver.
     */
    void validate_tree_relations(ReducedLungTreeMetadata& metadata)
    {
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

    std::vector<int> copy_dof_ids(const std::array<int, 4>& dof_ids)
    {
      return std::vector<int>(dof_ids.begin(), dof_ids.end());
    }

    std::vector<int> copy_dof_ids(const std::array<int, 6>& dof_ids)
    {
      return std::vector<int>(dof_ids.begin(), dof_ids.end());
    }

    /**
     * Copy one-child junction equation metadata from existing connection data.
     */
    void add_connection_metadata(ReducedLungTreeMetadata& metadata,
        const Junctions::ConnectionData& connections,
        std::map<int, TreeJunctionKind>& junction_kind)
    {
      for (std::size_t i = 0; i < connections.size(); ++i)
      {
        const int parent_global_id = connections.global_parent_element_id[i];
        const int child_global_id = connections.global_child_element_id[i];
        const int parent_index = element_index(metadata, parent_global_id);
        const int child_index = element_index(metadata, child_global_id);
        add_child_relation(metadata, parent_index, child_index, parent_global_id);

        TreeJunctionMetadata junction;
        junction.kind = TreeJunctionKind::Connection;
        junction.parent_element_index = parent_index;
        junction.child_element_indices[0] = child_index;
        junction.child_count = 1;
        junction.first_global_equation_id = connections.first_global_equation_id[i];
        junction.first_local_equation_id = connections.first_local_equation_id[i];
        junction.num_equations = 2;
        junction.global_dof_ids = copy_dof_ids(connections.global_dof_ids[i]);
        junction.local_dof_ids = copy_dof_ids(connections.local_dof_ids[i]);
        metadata.junctions.push_back(junction);

        const auto insert_result =
            junction_kind.emplace(parent_index, TreeJunctionKind::Connection);
        FOUR_C_ASSERT_ALWAYS(insert_result.second,
            "Duplicate junction metadata for parent element {}.", parent_global_id + 1);
      }
    }

    /**
     * Copy two-child junction equation metadata from existing bifurcation data.
     */
    void add_bifurcation_metadata(ReducedLungTreeMetadata& metadata,
        const Junctions::BifurcationData& bifurcations,
        std::map<int, TreeJunctionKind>& junction_kind)
    {
      for (std::size_t i = 0; i < bifurcations.size(); ++i)
      {
        const int parent_global_id = bifurcations.global_parent_element_id[i];
        const int child_1_global_id = bifurcations.global_child_1_element_id[i];
        const int child_2_global_id = bifurcations.global_child_2_element_id[i];
        const int parent_index = element_index(metadata, parent_global_id);
        const int child_1_index = element_index(metadata, child_1_global_id);
        const int child_2_index = element_index(metadata, child_2_global_id);
        add_child_relation(metadata, parent_index, child_1_index, parent_global_id);
        add_child_relation(metadata, parent_index, child_2_index, parent_global_id);

        TreeJunctionMetadata junction;
        junction.kind = TreeJunctionKind::Bifurcation;
        junction.parent_element_index = parent_index;
        junction.child_element_indices[0] = child_1_index;
        junction.child_element_indices[1] = child_2_index;
        junction.child_count = 2;
        junction.first_global_equation_id = bifurcations.first_global_equation_id[i];
        junction.first_local_equation_id = bifurcations.first_local_equation_id[i];
        junction.num_equations = 3;
        junction.global_dof_ids = copy_dof_ids(bifurcations.global_dof_ids[i]);
        junction.local_dof_ids = copy_dof_ids(bifurcations.local_dof_ids[i]);
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
     * Build solver topology and junction metadata from the existing junction setup.
     */
    void build_junction_topology_and_metadata(ReducedLungTreeMetadata& metadata,
        const Junctions::ConnectionData& connections,
        const Junctions::BifurcationData& bifurcations)
    {
      std::map<int, TreeJunctionKind> junction_kind;
      add_connection_metadata(metadata, connections, junction_kind);
      add_bifurcation_metadata(metadata, bifurcations, junction_kind);
      validate_junction_coverage(metadata, junction_kind);
      validate_tree_relations(metadata);
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
     * Build boundary-condition metadata and classify each boundary by tree side.
     */
    void build_boundary_condition_metadata(ReducedLungTreeMetadata& metadata,
        const BoundaryConditions::BoundaryConditionContainer& boundary_conditions)
    {
      for (const auto& model : boundary_conditions.models)
      {
        const auto& data = model.data;
        for (std::size_t i = 0; i < data.size(); ++i)
        {
          const int element_index_value = element_index(metadata, data.global_element_id[i]);
          const auto& element = metadata.elements[static_cast<std::size_t>(element_index_value)];
          const TreeBoundarySide side = determine_boundary_side(element, data.node_id[i]);

          metadata.boundary_conditions.push_back(TreeBoundaryConditionMetadata{
              .constrained_variable = model.constrained_variable,
              .side = side,
              .node_id = data.node_id[i],
              .element_index = element_index_value,
              .local_equation_id = data.local_equation_id[i],
              .global_equation_id = data.global_equation_id[i],
              .global_dof_id = data.global_dof_id[i],
              .local_dof_id = data.local_dof_id[i],
          });
        }
      }
    }

    /**
     * Count boundary conditions attached to one element side.
     */
    int count_boundaries_on_element_side(
        const ReducedLungTreeMetadata& metadata, int element_index_value, TreeBoundarySide side)
    {
      return static_cast<int>(
          std::count_if(metadata.boundary_conditions.begin(), metadata.boundary_conditions.end(),
              [element_index_value, side](const TreeBoundaryConditionMetadata& boundary)
              { return boundary.element_index == element_index_value && boundary.side == side; }));
    }

    /**
     * Validate that the root inlet and all leaf outlets close the tree system.
     */
    void validate_boundary_closure(const ReducedLungTreeMetadata& metadata)
    {
      const int root_inlet_boundary_count = count_boundaries_on_element_side(
          metadata, metadata.root_element_index, TreeBoundarySide::Inlet);
      FOUR_C_ASSERT_ALWAYS(root_inlet_boundary_count == 1,
          "Reduced-lung tree root inlet node {} must have exactly one boundary condition, found "
          "{}.",
          metadata.root_node_id + 1, root_inlet_boundary_count);

      for (std::size_t i = 0; i < metadata.elements.size(); ++i)
      {
        const auto& element = metadata.elements[i];
        if (!element.is_leaf())
        {
          continue;
        }

        const int outlet_boundary_count = count_boundaries_on_element_side(
            metadata, static_cast<int>(i), TreeBoundarySide::Outlet);
        FOUR_C_ASSERT_ALWAYS(outlet_boundary_count == 1,
            "Reduced-lung tree leaf element {} outlet node {} must have exactly one boundary "
            "condition, found {}.",
            element.global_element_id + 1, element.outlet_node_id + 1, outlet_boundary_count);
      }
    }
  }  // namespace

  const TreeJunctionMetadata& tree_junction_for_parent(
      const ReducedLungTreeMetadata& metadata, int parent_element_index)
  {
    const auto it = std::ranges::find_if(metadata.junctions,
        [parent_element_index](const TreeJunctionMetadata& junction)
        { return junction.parent_element_index == parent_element_index; });
    FOUR_C_ASSERT(it != metadata.junctions.end(),
        "Reduced-lung tree metadata has no junction for parent element index {}.",
        parent_element_index);
    return *it;
  }

  const TreeBoundaryConditionMetadata& tree_outlet_boundary_for_element(
      const ReducedLungTreeMetadata& metadata, int element_index)
  {
    const auto it = std::ranges::find_if(metadata.boundary_conditions,
        [element_index](const TreeBoundaryConditionMetadata& boundary)
        {
          return boundary.element_index == element_index &&
                 boundary.side == TreeBoundarySide::Outlet;
        });
    FOUR_C_ASSERT(it != metadata.boundary_conditions.end(),
        "Reduced-lung tree metadata has no outlet boundary for element index {}.", element_index);
    return *it;
  }

  const TreeBoundaryConditionMetadata& tree_root_inlet_boundary(
      const ReducedLungTreeMetadata& metadata)
  {
    const auto it = std::ranges::find_if(metadata.boundary_conditions,
        [&metadata](const TreeBoundaryConditionMetadata& boundary)
        {
          return boundary.element_index == metadata.root_element_index &&
                 boundary.side == TreeBoundarySide::Inlet;
        });
    FOUR_C_ASSERT(it != metadata.boundary_conditions.end(),
        "Reduced-lung tree metadata has no root inlet boundary.");
    return *it;
  }

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

    const auto model_metadata =
        collect_element_model_metadata(context.airways, context.terminal_units, context.row_map);

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

      const auto model_metadata_it = model_metadata.find(global_element_id);
      FOUR_C_ASSERT_ALWAYS(model_metadata_it != model_metadata.end(),
          "Missing model equation metadata for reduced-lung element {}.", global_element_id + 1);
      const auto& element_model_metadata = model_metadata_it->second;

      TreeElementMetadata element{
          .global_element_id = global_element_id,
          .kind = map_element_kind(context.element_types[global_element_id]),
          .inlet_node_id = topology_nodes[0],
          .outlet_node_id = topology_nodes[1],
          .parent_element_index = -1,
          .child_element_indices = {-1, -1},
          .child_count = 0,
          .first_global_dof = element_model_metadata.global_dof_ids.front(),
          .num_dofs = static_cast<int>(element_model_metadata.global_dof_ids.size()),
          .global_dof_ids = element_model_metadata.global_dof_ids,
          .local_dof_ids = element_model_metadata.local_dof_ids,
          .first_local_state_equation_id = element_model_metadata.first_local_equation_id,
          .first_global_state_equation_id = element_model_metadata.first_global_equation_id,
          .num_state_equations = element_model_metadata.num_equations,
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

    metadata.num_global_dofs = context.locally_relevant_dof_map.num_global_elements();
    metadata.num_global_equations = context.row_map.num_global_elements();
    metadata.num_locally_relevant_dofs = context.locally_relevant_dof_map.num_my_elements();
    FOUR_C_ASSERT_ALWAYS(metadata.num_global_equations == metadata.num_global_dofs,
        "Reduced-lung tree metadata requires a square Newton system, got {} equation rows and {} "
        "dofs.",
        metadata.num_global_equations, metadata.num_global_dofs);

    build_junction_topology_and_metadata(metadata, context.connections, context.bifurcations);
    build_boundary_condition_metadata(metadata, context.boundary_conditions);
    validate_boundary_closure(metadata);
    build_layers(metadata);

    return metadata;
  }
}  // namespace ReducedLung

FOUR_C_NAMESPACE_CLOSE

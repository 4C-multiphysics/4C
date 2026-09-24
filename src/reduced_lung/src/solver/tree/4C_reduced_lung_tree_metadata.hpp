// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_REDUCED_LUNG_TREE_METADATA_HPP
#define FOUR_C_REDUCED_LUNG_TREE_METADATA_HPP

#include "4C_config.hpp"

#include "4C_reduced_lung_airways.hpp"
#include "4C_reduced_lung_boundary_conditions.hpp"
#include "4C_reduced_lung_input.hpp"
#include "4C_reduced_lung_junctions.hpp"
#include "4C_reduced_lung_terminal_unit.hpp"

#include <array>
#include <cstdint>
#include <map>
#include <span>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core
{
  namespace FE
  {
    class Discretization;
  }

  namespace LinAlg
  {
    class Map;
  }
}  // namespace Core

namespace ReducedLung
{
  /**
   * @brief Reduced-lung element category used by tree metadata.
   */
  enum class TreeElementKind : std::uint8_t
  {
    Airway,       ///< Airway element.
    TerminalUnit  ///< Terminal-unit element.
  };

  /**
   * @brief Junction category used by tree metadata.
   */
  enum class TreeJunctionKind : std::uint8_t
  {
    Connection,  ///< One parent element connected to one child element.
    Bifurcation  ///< One parent element connected to two child elements.
  };

  /**
   * @brief Boundary side relative to the directed reduced-lung tree.
   */
  enum class TreeBoundarySide : std::uint8_t
  {
    Inlet,  ///< Parent/root side of an element.
    Outlet  ///< Child/leaf side of an element.
  };

  /**
   * @brief Per-element topology, dof, and state-equation layout for tree solvers.
   */
  struct TreeElementMetadata
  {
    int global_element_id = -1;                      ///< Global reduced-lung element id.
    TreeElementKind kind = TreeElementKind::Airway;  ///< Element category.

    int inlet_node_id = -1;   ///< Global inlet node id.
    int outlet_node_id = -1;  ///< Global outlet node id.

    int parent_element_index = -1;  ///< Parent element index, or -1 for the root element.
    std::array<int, 2> child_element_indices{-1, -1};  ///< Child element indices.
    int child_count = 0;  ///< Number of valid entries in @ref child_element_indices.

    int first_global_dof = -1;        ///< First global dof id of the element.
    int num_dofs = 0;                 ///< Number of dofs associated with the element.
    std::vector<int> global_dof_ids;  ///< Global dof ids associated with the element.
    std::vector<int> local_dof_ids;   ///< Local dof ids on the locally relevant dof map.

    int first_local_state_equation_id = -1;   ///< First local state-equation row id.
    int first_global_state_equation_id = -1;  ///< First global state-equation row id.
    int num_state_equations = 0;              ///< Number of element state equations.

    /**
     * @brief Check whether the element has no child elements.
     *
     * @return True if this element is a leaf element.
     */
    [[nodiscard]] bool is_leaf() const { return child_count == 0; }
  };

  /**
   * @brief Junction-equation metadata connecting one parent element to one or two children.
   */
  struct TreeJunctionMetadata
  {
    TreeJunctionKind kind = TreeJunctionKind::Connection;  ///< Junction category.
    int parent_element_index = -1;                         ///< Parent element index.
    std::array<int, 2> child_element_indices{-1, -1};      ///< Child element indices.
    int child_count = 0;  ///< Number of valid entries in @ref child_element_indices.

    int first_local_equation_id = -1;   ///< First local junction-equation row id.
    int first_global_equation_id = -1;  ///< First global junction-equation row id.
    int num_equations = 0;              ///< Number of junction equations.

    std::vector<int> global_dof_ids;  ///< Global dof ids used by the junction equations.
    std::vector<int> local_dof_ids;   ///< Local dof ids used by the junction equations.
  };

  /**
   * @brief Boundary-condition equation metadata attached to an inlet or outlet element side.
   */
  struct TreeBoundaryConditionMetadata
  {
    BoundaryConditions::ConstrainedVariable constrained_variable =
        BoundaryConditions::ConstrainedVariable::Pressure;  ///< Constrained boundary variable.
    TreeBoundarySide side = TreeBoundarySide::Inlet;  ///< Element side constrained by the boundary.

    int node_id = -1;        ///< Global node id of the boundary condition.
    int element_index = -1;  ///< Attached element index.

    int local_equation_id = -1;   ///< Local boundary-equation row id.
    int global_equation_id = -1;  ///< Global boundary-equation row id.

    int global_dof_id = -1;  ///< Global constrained dof id.
    int local_dof_id = -1;   ///< Local constrained dof id.
  };

  /**
   * @brief Immutable tree metadata consumed by tree-based Newton linear solvers.
   */
  struct ReducedLungTreeMetadata
  {
    std::vector<TreeElementMetadata> elements;    ///< Metadata for all directed tree elements.
    std::vector<TreeJunctionMetadata> junctions;  ///< Metadata for connection/bifurcation rows.
    std::vector<TreeBoundaryConditionMetadata>
        boundary_conditions;  ///< Metadata for inlet/outlet boundary rows.

    std::map<int, int> element_index_by_global_id;  ///< Global element id to metadata index.

    int root_element_index = -1;  ///< Metadata index of the root element.
    int root_node_id = -1;        ///< Global node id of the root inlet.

    std::vector<int> airway_element_indices;         ///< Metadata indices of airway elements.
    std::vector<int> terminal_unit_element_indices;  ///< Metadata indices of terminal units.

    std::vector<std::vector<int>> bottom_up_layers;  ///< Leaf-to-root element traversal layers.
    std::vector<std::vector<int>> top_down_layers;   ///< Root-to-leaf element traversal layers.

    int num_global_dofs = 0;            ///< Total number of global dofs.
    int num_global_equations = 0;       ///< Total number of global equations.
    int num_locally_relevant_dofs = 0;  ///< Number of locally relevant dofs on this rank.
  };

  /**
   * @brief Inputs required to build reduced-lung tree metadata from the existing model setup.
   */
  struct ReducedLungTreeMetadataContext
  {
    const Core::FE::Discretization& discretization;  ///< Initialized reduced-lung discretization.
    std::span<const ReducedLungParameters::LungTree::ElementType>
        element_types;  ///< Resolved element category by global element id.
    const std::map<int, int>& first_global_dof_of_ele;  ///< First global dof by element id.
    const std::map<int, int>& global_dof_per_ele;       ///< Number of dofs by element id.

    const Airways::AirwayContainer& airways;                     ///< Local airway model blocks.
    const TerminalUnits::TerminalUnitContainer& terminal_units;  ///< Local terminal-unit blocks.

    const Junctions::ConnectionData& connections;    ///< Connection equation data.
    const Junctions::BifurcationData& bifurcations;  ///< Bifurcation equation data.

    const BoundaryConditions::BoundaryConditionContainer&
        boundary_conditions;  ///< Boundary-condition equation data.

    const Core::LinAlg::Map& row_map;                   ///< Row map for reduced-lung equations.
    const Core::LinAlg::Map& locally_relevant_dof_map;  ///< Locally relevant dof map.
  };

  /**
   * @brief Build and validate tree metadata without changing solver state or assembly behavior.
   *
   * @param context Existing reduced-lung setup data used to construct the tree metadata.
   * @return Directed tree metadata with topology, dof, equation, and traversal information.
   */
  ReducedLungTreeMetadata build_reduced_lung_tree_metadata(
      const ReducedLungTreeMetadataContext& context);
}  // namespace ReducedLung

FOUR_C_NAMESPACE_CLOSE

#endif

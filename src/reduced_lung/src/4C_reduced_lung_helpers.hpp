// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_REDUCED_LUNG_HELPERS_HPP
#define FOUR_C_REDUCED_LUNG_HELPERS_HPP

#include "4C_config.hpp"

#include "4C_io_discretization_visualization_writer_mesh.hpp"
#include "4C_linalg_map.hpp"
#include "4C_reduced_lung_airways.hpp"
#include "4C_reduced_lung_boundary_conditions.hpp"
#include "4C_reduced_lung_junctions.hpp"
#include "4C_reduced_lung_terminal_unit.hpp"

#include <mpi.h>

#include <map>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::Nodes
{
  class Node;
}

namespace Core::FE
{
  class Discretization;
}

namespace Core::Rebalance
{
  struct RebalanceParameters;
}

namespace ReducedLung
{
  /*!
   * @brief Build a minimal discretization from the reduced lung topology.
   *
   * Creates line2 elements with a lightweight geometry-only element type and registers
   * node coordinates/element connectivity for distribution, rebalancing, and output.
   *
   * @param discretization Target 4C discretization to populate.
   * @param topology Reduced lung topology description (nodes and element connectivity).
   * @param rebalance_parameters Parameters for mesh rebalancing/partitioning.
   */
  void build_discretization_from_topology(Core::FE::Discretization& discretization,
      const ReducedLungParameters::LungTree::Topology& topology,
      const Core::Rebalance::RebalanceParameters& rebalance_parameters);

  /**
   * @brief Create locally owned airway/terminal-unit entities from the row elements.
   *
   * Populates model containers and the local element->dof-count map used for global dof numbering.
   */
  void create_local_element_models(const Core::FE::Discretization& discretization,
      const ReducedLungParameters& parameters, Airways::AirwayContainer& airways,
      TerminalUnits::TerminalUnitContainer& terminal_units, std::map<int, int>& dof_per_ele,
      int& n_airways, int& n_terminal_units);

  /**
   * @brief Build global dof layout maps from local element dof counts.
   *
   * @param local_dof_per_ele Local map global element id -> number of dofs.
   * @param comm Communicator used for all-reduce.
   * @param global_dof_per_ele Output global map global element id -> number of dofs.
   * @param first_global_dof_of_ele Output global map global element id -> first global dof id.
   */
  void create_global_dof_maps(const std::map<int, int>& local_dof_per_ele, const MPI_Comm& comm,
      std::map<int, int>& global_dof_per_ele, std::map<int, int>& first_global_dof_of_ele);

  /**
   * @brief Assign global dof ids to airway and terminal-unit model data.
   */
  void assign_global_dof_ids_to_models(const std::map<int, int>& first_global_dof_of_ele,
      Airways::AirwayContainer& airways, TerminalUnits::TerminalUnitContainer& terminal_units);

  /**
   * @brief Build global map node id -> adjacent global element ids.
   */
  std::map<int, std::vector<int>> create_global_ele_ids_per_node(
      const Core::FE::Discretization& discretization, const MPI_Comm& comm);

  /**
   * @brief Print globally reduced counts of instantiated reduced-lung entities.
   */
  void print_instantiated_object_counts(const MPI_Comm& comm, int n_airways, int n_terminal_units,
      int n_connections, int n_bifurcations, int n_boundary_conditions);

  /*!
   * @brief Create the map with the locally owned dofs spanning the computation domain that
   * are necessary for the solution vector.
   *
   * The 4C discretization gives a mpi distribution of the Reduced Lung elements. From this
   * distribution and the knowledge about the number of dofs in every element, the new map
   * mapping the local dofs to their global ids is created. Example: The dofs of element k have
   * global ids in the range first_global_dof_of_ele[k] to first_global_dof_of_ele[k] + dofs of
   * element k.
   *
   * @param comm Communicator of the 4C discretization.
   * @param airways Vector of locally owned airways.
   * @param terminal_units Locally owned terminal units.
   * @return map specifying the dof-distribution over all ranks.
   */
  Core::LinAlg::Map create_domain_map(const MPI_Comm& comm, const Airways::AirwayContainer& airways,
      const TerminalUnits::TerminalUnitContainer& terminal_units);

  /*!
   * @brief Create the map with the locally owned row indices of the system matrix, i.e. the
   * distribution of the system's equation.
   *
   * The row indices are uniquely tied to the system equations. Given the locally owned
   * elements and nodes in the 4C discretization, the related equation ids are created and stored in
   * this map. Every element provides state equations, every node provides information about
   * its elements. Depending on the number of connected elements at one node, different sets and
   * numbers of equations are needed.
   * Per owned element: one or two equations and row ids.
   * Node contained by 1 element: Boundary condition -> 1 equation and row id.
   * Node contained by 2 elements: Connection -> 2 equations and row ids.
   * Node contained by 3 elements: Bifurcation -> 3 equations and row ids.
   *
   * @param comm Communicator of the 4C discretization.
   * @param airways Vector of locally owned airways.
   * @param terminal_units Locally owned terminal units.
   * @param connections Connection data (parent element id and child element id).
   * @param bifurcations Bifurcation data (parent element id and two child element ids).
   * @param boundary_conditions Boundary condition container with locally owned conditions.
   * @return map with locally owned rows.
   */
  Core::LinAlg::Map create_row_map(const MPI_Comm& comm, const Airways::AirwayContainer& airways,
      const TerminalUnits::TerminalUnitContainer& terminal_units,
      const Junctions::ConnectionData& connections, const Junctions::BifurcationData& bifurcations,
      const BoundaryConditions::BoundaryConditionContainer& boundary_conditions);

  /*!
   * @brief Create the map with the dof indices relevant for the locally owned equations/rows.
   *
   * This map connects the equations (rows of matrix and rhs vector) with their relevant dofs.
   * Therefore, it needs explicit knowledge of the different equation types in the row map and the
   * dof ordering in the domain map. This function loops over every local equation type and extracts
   * the relevant dof ids for every present element. From these ids, the column map is created.
   *
   * @param comm Communicator of the 4C discretization.
   * @param airways Vector of locally owned airways.
   * @param terminal_units Locally owned terminal units
   * @param global_dof_per_ele Map from global element id to associated dofs (over all processors).
   * @param first_global_dof_of_ele Map from global element id to its first global dof id.
   * @param connections Connection data (parent element id and child element id).
   * @param bifurcations Bifurcation data (parent element id and two child element ids).
   * @param boundary_conditions Boundary condition container with locally owned conditions.
   * @return map with distribution of column indices for the system matrix.
   */
  Core::LinAlg::Map create_column_map(const MPI_Comm& comm, const Airways::AirwayContainer& airways,
      const TerminalUnits::TerminalUnitContainer& terminal_units,
      const std::map<int, int>& global_dof_per_ele,
      const std::map<int, int>& first_global_dof_of_ele,
      const Junctions::ConnectionData& connections, const Junctions::BifurcationData& bifurcations,
      const BoundaryConditions::BoundaryConditionContainer& boundary_conditions);

  /*!
   * @brief Add an airway element with the appropriate template instantiation based on model types.
   *
   * This helper function encapsulates the template instantiation logic for adding airway elements
   * based on the flow model (Linear/NonLinear) and wall model (Rigid/KelvinVoigt) types.
   *
   * @param airways Container for all airway models.
   * @param global_element_id Unique global identifier for the element.
   * @param local_element_id Local element id for the row map.
   * @param parameters Reduced lung parameters containing model and geometry information.
   * @param flow_model_type The flow model type.
   * @param wall_model_type The wall model type.
   */
  void add_airway_with_model_selection(Airways::AirwayContainer& airways, int global_element_id,
      int local_element_id, const ReducedLungParameters& parameters,
      ReducedLungParameters::LungTree::Airways::FlowModel::ResistanceType flow_model_type,
      ReducedLungParameters::LungTree::Airways::WallModelType wall_model_type);

  /*!
   * @brief Add a terminal unit element with the appropriate template instantiation based on model
   * types.
   *
   * This helper function encapsulates the template instantiation logic for adding terminal unit
   * elements based on the rheological model (KelvinVoigt/FourElementMaxwell) and elasticity model
   * (Linear/Ogden) types.
   *
   * @param terminal_units Container for all terminal unit models.
   * @param global_element_id Unique global identifier of the element.
   * @param local_element_id Local element id for the row map.
   * @param parameters Terminal unit parameters containing model information.
   * @param rheological_model_type The rheological model type.
   * @param elasticity_model_type The elasticity model type.
   */
  void add_terminal_unit_with_model_selection(TerminalUnits::TerminalUnitContainer& terminal_units,
      int global_element_id, int local_element_id, const ReducedLungParameters& parameters,
      ReducedLungParameters::LungTree::TerminalUnits::RheologicalModel::RheologicalModelType
          rheological_model_type,
      ReducedLungParameters::LungTree::TerminalUnits::ElasticityModel::ElasticityModelType
          elasticity_model_type);

  void collect_runtime_output_data(
      Core::IO::DiscretizationVisualizationWriterMesh& visualization_writer,
      const Airways::AirwayContainer& airways,
      const TerminalUnits::TerminalUnitContainer& terminal_units,
      const Core::LinAlg::Vector<double>& locally_relevant_dofs,
      const Core::LinAlg::Map* element_row_map);
}  // namespace ReducedLung

FOUR_C_NAMESPACE_CLOSE

#endif

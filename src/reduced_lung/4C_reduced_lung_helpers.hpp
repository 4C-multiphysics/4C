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
#include "4C_reduced_lung_terminal_unit.hpp"

#include <mpi.h>

FOUR_C_NAMESPACE_OPEN

namespace Core::Nodes
{
  class Node;
}

namespace ReducedLung
{
  enum class BoundaryConditionType
  {
    pressure_in,
    pressure_out,
    flow_in,
    flow_out
  };

  struct BoundaryCondition
  {
    int global_element_id;
    int global_equation_id;
    int local_equation_id;
    int local_bc_id;
    BoundaryConditionType bc_type;
    int global_dof_id;
    int funct_num;
    int local_dof_id = 0;
  };

  struct Connection
  {
    int first_global_equation_id;
    int first_local_equation_id;
    int local_connection_id;
    int global_parent_element_id;
    int global_child_element_id;
    enum DofNumbering
    {
      p_out_parent = 0,
      p_in_child = 1,
      q_out_parent = 2,
      q_in_child = 3
    };
    std::array<int, 4> global_dof_ids;
    std::array<int, 4> local_dof_ids{};
  };

  struct Bifurcation
  {
    int first_global_equation_id;
    int first_local_equation_id;
    int local_bifurcation_id;
    int global_parent_element_id;
    int global_child_1_element_id;
    int global_child_2_element_id;
    enum DofNumbering
    {
      p_out_parent = 0,
      p_in_child_1 = 1,
      p_in_child_2 = 2,
      q_out_parent = 3,
      q_in_child_1 = 4,
      q_in_child_2 = 5
    };
    std::array<int, 6> global_dof_ids;
    std::array<int, 6> local_dof_ids{};
  };

  enum ElementDofNumbering
  {
    p_in = 0,
    p_out = 1,
    q_in = 2,
    q_out = 3
  };

  enum class AirwayType
  {
    resistive,
    viscoelastic_RLC
  };

  struct Airway
  {
    int global_element_id;
    int local_element_id;
    int local_airway_id;
    AirwayType airway_type;
    // dofs: {p1, p2, q} for resistive airways; {p1, p2, q1, q2} for compliant airways
    std::vector<int> global_dof_ids{};
    int n_state_equations = 1;
    // local dof ids in locally relevant dof map!
    std::vector<int> local_dof_ids{};
  };

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
  Core::LinAlg::Map create_domain_map(const MPI_Comm& comm, const std::vector<Airway>& airways,
      const TerminalUnits& terminal_units);

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
   * @param connections Vector with Connection type entries (parent element id and child
   * element id).
   * @param bifurcations Vector with Bifurcation type entries (parent element id and two
   * child element ids).
   * @param boundary_conditions Vector with boundary condition information. Here, only the boundary
   * element ids are needed.
   * @return map with locally owned rows.
   */
  Core::LinAlg::Map create_row_map(const MPI_Comm& comm, const std::vector<Airway>& airways,
      const TerminalUnits& terminal_units, const std::vector<Connection>& connections,
      const std::vector<Bifurcation>& bifurcations,
      const std::vector<BoundaryCondition>& boundary_conditions);

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
   * @param connections Vector with Connection type entries (parent element id and child
   * element id).
   * @param bifurcations Vector with Bifurcation type entries (parent element id and two
   * child element ids).
   * @param boundary_conditions Vector with boundary condition information. Here, only the boundary
   * element ids are needed.
   * @return map with distribution of column indices for the system matrix.
   */
  Core::LinAlg::Map create_column_map(const MPI_Comm& comm, const std::vector<Airway>& airways,
      const TerminalUnits& terminal_units, const std::map<int, int>& global_dof_per_ele,
      const std::map<int, int>& first_global_dof_of_ele, const std::vector<Connection>& connections,
      const std::vector<Bifurcation>& bifurcations,
      const std::vector<BoundaryCondition>& boundary_conditions);

  void collect_runtime_output_data(
      Core::IO::DiscretizationVisualizationWriterMesh& visualization_writer,
      const std::vector<Airway>& airways, const TerminalUnits& terminal_units,
      const Core::LinAlg::Vector<double>& locally_relevant_dofs,
      const Core::LinAlg::Map* element_row_map);
}  // namespace ReducedLung

FOUR_C_NAMESPACE_CLOSE

#endif

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_REDUCED_LUNG_JUNCTIONS_HPP
#define FOUR_C_REDUCED_LUNG_JUNCTIONS_HPP

#include "4C_config.hpp"

#include <array>
#include <map>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}

namespace Core::LinAlg
{
  class Map;
  class SparseMatrix;
  template <typename T>
  class Vector;
}  // namespace Core::LinAlg

namespace ReducedLung
{
  class TreeCoefficientAssemblyTarget;
  class TreeLinearization;

  namespace Junctions
  {
    /**
     * @brief Shared data container for all connections of two elements.
     *
     * Stores identifiers and dof mappings for connection equations.
     */
    struct ConnectionData
    {
      /**
       * @brief Dof ordering for connection equations and structured coefficient assembly.
       */
      enum DofNumbering
      {
        p_out_parent = 0,  ///< Outlet pressure of the parent element.
        p_in_child = 1,    ///< Inlet pressure of the child element.
        q_out_parent = 2,  ///< Outlet flow of the parent element.
        q_in_child = 3     ///< Inlet flow of the child element.
      };

      std::vector<int> first_global_equation_id;
      std::vector<int> first_local_equation_id;
      std::vector<int> local_connection_id;
      std::vector<int> global_parent_element_id;
      std::vector<int> global_child_element_id;
      std::vector<std::array<int, 4>> global_dof_ids;
      std::vector<std::array<int, 4>> local_dof_ids;  ///< Local dof ids in DofNumbering order.

      std::vector<int> first_row;         ///< First local row of each connection equation pair.
      std::vector<int> p_out_parent_lid;  ///< Cached local id for parent outlet pressure.
      std::vector<int> p_in_child_lid;    ///< Cached local id for child inlet pressure.
      std::vector<int> q_out_parent_lid;  ///< Cached local id for parent outlet flow.
      std::vector<int> q_in_child_lid;    ///< Cached local id for child inlet flow.

      [[nodiscard]] size_t size() const { return global_parent_element_id.size(); }
      void clear();
      void reserve(size_t count);
      void add_connection(int local_id, int global_parent_id, int global_child_id,
          const std::array<int, 4>& dof_ids);
    };

    /**
     * @brief Shared data container for all bifurcations with one parent element splitting into two
     * child elements.
     *
     * Stores identifiers and dof mappings for bifurcation equations.
     */
    struct BifurcationData
    {
      /**
       * @brief Dof ordering for bifurcation equations and structured coefficient assembly.
       */
      enum DofNumbering
      {
        p_out_parent = 0,  ///< Outlet pressure of the parent element.
        p_in_child_1 = 1,  ///< Inlet pressure of the first child element.
        p_in_child_2 = 2,  ///< Inlet pressure of the second child element.
        q_out_parent = 3,  ///< Outlet flow of the parent element.
        q_in_child_1 = 4,  ///< Inlet flow of the first child element.
        q_in_child_2 = 5   ///< Inlet flow of the second child element.
      };

      std::vector<int> first_global_equation_id;
      std::vector<int> first_local_equation_id;
      std::vector<int> local_bifurcation_id;
      std::vector<int> global_parent_element_id;
      std::vector<int> global_child_1_element_id;
      std::vector<int> global_child_2_element_id;
      std::vector<std::array<int, 6>> global_dof_ids;
      std::vector<std::array<int, 6>> local_dof_ids;  ///< Local dof ids in DofNumbering order.

      std::vector<int> first_row;         ///< First local row of each bifurcation equation triple.
      std::vector<int> p_out_parent_lid;  ///< Cached local id for parent outlet pressure.
      std::vector<int> p_in_child_1_lid;  ///< Cached local id for first child inlet pressure.
      std::vector<int> p_in_child_2_lid;  ///< Cached local id for second child inlet pressure.
      std::vector<int> q_out_parent_lid;  ///< Cached local id for parent outlet flow.
      std::vector<int> q_in_child_1_lid;  ///< Cached local id for first child inlet flow.
      std::vector<int> q_in_child_2_lid;  ///< Cached local id for second child inlet flow.

      [[nodiscard]] size_t size() const { return global_parent_element_id.size(); }
      void clear();
      void reserve(size_t count);
      void add_bifurcation(int local_id, int global_parent_id, int global_child_1_id,
          int global_child_2_id, const std::array<int, 6>& dof_ids);
    };

    /*!
     * @brief Create connections and bifurcations based on the node adjacency of the lung tree.
     *
     * The node ordering associated with an element is assumed to be top-down: nodes[0] is the
     * inlet/parent, nodes[1] the outlet/child side. This is required for correct classification and
     * dof layout.
     */
    void create_junctions(const Core::FE::Discretization& discretization,
        const std::map<int, std::vector<int>>& global_ele_ids_per_node,
        const std::map<int, int>& global_dof_per_ele,
        const std::map<int, int>& first_global_dof_of_ele, ConnectionData& connections,
        BifurcationData& bifurcations);

    void assign_junction_local_equation_ids(
        ConnectionData& connections, BifurcationData& bifurcations, int& n_local_equations);

    void assign_junction_global_equation_ids(const Core::LinAlg::Map& row_map,
        ConnectionData& connections, BifurcationData& bifurcations);

    /**
     * @brief Assign local dof ids and cache named dof handles for junction assembly.
     */
    void assign_junction_local_dof_ids(const Core::LinAlg::Map& locally_relevant_dof_map,
        ConnectionData& connections, BifurcationData& bifurcations);

    void update_residual_vector(Core::LinAlg::Vector<double>& rhs,
        const ConnectionData& connections, const BifurcationData& bifurcations,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs);

    void update_jacobian(Core::LinAlg::SparseMatrix& sysmat, const ConnectionData& connections,
        const BifurcationData& bifurcations);

    /**
     * @brief Assemble connection and bifurcation coefficients into structured tree storage.
     */
    void update_tree_linearization(TreeCoefficientAssemblyTarget& target,
        const ConnectionData& connections, const BifurcationData& bifurcations);
  }  // namespace Junctions
}  // namespace ReducedLung

FOUR_C_NAMESPACE_CLOSE

#endif

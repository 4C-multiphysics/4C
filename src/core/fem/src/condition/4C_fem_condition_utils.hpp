// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FEM_CONDITION_UTILS_HPP
#define FOUR_C_FEM_CONDITION_UTILS_HPP

#include "4C_config.hpp"

#include "4C_fem_condition.hpp"
#include "4C_linalg_map.hpp"

#include <memory>
#include <set>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  class MapExtractor;
}

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::Nodes
{
  class Node;
}

namespace Core::Elements
{
  class Element;
}

namespace Core::Conditions
{
  // forward declaration
  class Selector;

  /**
   * A functor that returns true if the given global id is owned by the emap.
   */
  struct MyGID
  {
    const Core::LinAlg::Map* emap_;
    MyGID(const Core::LinAlg::Map* emap) : emap_(emap) {}
    bool operator()(int gid) const { return emap_->my_gid(gid); }
  };

  /**
   * An enum to specify whether to look for locally owned nodes or locally owned and ghosted
   * nodes in the various finder functions below.
   */
  enum class LookFor
  {
    locally_owned,
    locally_owned_and_ghosted,
  };

  /**
   * Loop all conditions of the given Discretization @p dis and find the ones with the
   * specified name @p condition_name. Depending on @p look_for, return a set of node IDs
   * that are locally owned or locally owned and ghosted.
   */
  [[nodiscard]] std::set<int> find_conditioned_node_ids(
      const Core::FE::Discretization& dis, const std::string& condition_name, LookFor look_for);

  /**
   * Loop over all @p conditions and extract their node IDs, if they are locally owned by the
   * discretization @p dis. Depending on @p look_for, return a set of node IDs
   * that are locally owned or locally owned and ghosted.
   */
  [[nodiscard]] std::set<int> find_conditioned_node_ids(const Core::FE::Discretization& dis,
      std::span<const Condition*> conditions, LookFor look_for);

  /**
   * Loop all conditions of the given Discretization @p dis, find the ones with the given name
   * and return a map of all conditioned node IDs and their corresponding conditions. A node
   * may be subject to multiple conditions. Depending on @p look_for, return node IDs
   * that are locally owned or locally owned and ghosted.
   */
  [[nodiscard]] std::multimap<int, const Condition*> find_conditioned_node_ids_and_conditions(
      const Core::FE::Discretization& dis, std::span<const Condition*> conditions,
      LookFor look_for);


  /// find all local nodes from discretization marked with condition and
  /// put them into a vector indexed by Id of the condition
  void find_conditioned_nodes(const Core::FE::Discretization& dis,
      std::span<const Condition*> conditions,
      std::map<int, std::shared_ptr<std::vector<int>>>& nodes, bool use_coupling_id = true);


  /// collect all nodes (in- and excluding 'ghosts') and
  /// elements (including ghosts) in a condition
  /*!
    \param dis discretization
    \param nodes unique map of nodes
    \param ghostnodes overlapping map of nodes
    \param elements overlapping map of elements
    \param condition_name name of condition
   */
  void find_condition_objects(const Core::FE::Discretization& dis,
      std::map<int, Core::Nodes::Node*>& nodes, std::map<int, Core::Nodes::Node*>& gnodes,
      std::map<int, std::shared_ptr<Core::Elements::Element>>& elements,
      std::span<const Condition*> conditions);

  /// collect all nodes (in- and excluding 'ghosts') and
  /// elements (including ghosts) in a condition
  /*!
    \param dis discretization
    \param nodes unique map of nodes
    \param ghostnodes overlapping map of nodes
    \param elements overlapping map of elements
    \param condition_name name of condition
   */
  void find_condition_objects(const Core::FE::Discretization& dis,
      std::map<int, Core::Nodes::Node*>& nodes, std::map<int, Core::Nodes::Node*>& gnodes,
      std::map<int, std::shared_ptr<Core::Elements::Element>>& elements,
      const std::string& condition_name);

  /// collect all nodes (in- and excluding 'ghosts') and
  /// elements (including ghosts) in a condition
  /*!
    \param dis discretization
    \param nodes unique map of nodes
    \param ghostnodes overlapping map of nodes
    \param elements overlapping map of elements
    \param condition_name name of condition
   */
  void find_condition_objects(const Core::FE::Discretization& dis,
      std::map<int, std::map<int, Core::Nodes::Node*>>& nodes,
      std::map<int, std::map<int, Core::Nodes::Node*>>& gnodes,
      std::map<int, std::map<int, std::shared_ptr<Core::Elements::Element>>>& elements,
      const std::string& condition_name);

  /// collect all elements in a condition including ghosts
  /*!
    \param dis discretization
    \param elements overlapping map of elements
    \param condition_name name of condition
   */
  void find_condition_objects(const Core::FE::Discretization& dis,
      std::map<int, std::shared_ptr<Core::Elements::Element>>& elements,
      const std::string& condition_name, const int label = -1);

  /// Find all conditions with given name that all nodes of the element have in common
  /*!
    \param ele (in) the element
    \param condition_name (in) name of the condition to look for
    \param condition (out) all conditions that cover all element nodes
  */
  void find_element_conditions(const Core::Elements::Element* ele,
      const std::string& condition_name, std::vector<Core::Conditions::Condition*>& condition);

  /// row map with nodes from condition
  std::shared_ptr<Core::LinAlg::Map> condition_node_row_map(
      const Core::FE::Discretization& dis, const std::string& condition_name);

  /// col map with nodes from condition
  std::shared_ptr<Core::LinAlg::Map> condition_node_col_map(
      const Core::FE::Discretization& dis, const std::string& condition_name);

  /// create the set of column element gids that have conditioned nodes
  /*!
    \note These are not elements from the condition geometry. Rather the
    gids of actual discretization elements are listed.
   */
  std::shared_ptr<std::set<int>> conditioned_element_map(
      const Core::FE::Discretization& dis, const std::string& condition_name);

}  // namespace Core::Conditions

FOUR_C_NAMESPACE_CLOSE

#endif

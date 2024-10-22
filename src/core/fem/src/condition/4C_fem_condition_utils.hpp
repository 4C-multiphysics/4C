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

#include <Epetra_Map.h>
#include <Teuchos_RCP.hpp>

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
  class ConditionSelector;

  /**
   * A functor that returns true if the given global id is owned by the emap.
   */
  struct MyGID
  {
    const Epetra_Map* emap_;
    MyGID(const Epetra_Map* emap) : emap_(emap) {}
    bool operator()(int gid) const { return emap_->MyGID(gid); }
  };

  /// find all local nodes from discretization marked with condition
  /*!
    Loop all conditions of the given discretization, find the ones with the
    specified name and collect the locally owned node ids in the supplied
    set. The nodes vector is unique and ordered on output.

    \param dis : (in) discretization
    \param condname : (in) name of condition in question
    \param nodes : (out) empty set on input, filled with nodal gids of local nodes
  */
  void find_conditioned_nodes(
      const Core::FE::Discretization& dis, const std::string& condname, std::vector<int>& nodes);

  /// find all local nodes from discretization marked with condition
  void find_conditioned_nodes(const Core::FE::Discretization& dis, const std::string& condname,
      std::map<int, Core::Nodes::Node*>& nodes);

  /// find all local nodes from discretization marked with condition
  void find_conditioned_nodes(
      const Core::FE::Discretization& dis, const std::string& condname, std::set<int>& nodeset);

  /// find all local nodes from discretization marked with condition
  /*!
    Loop all conditions of the given discretization, find the ones with the
    specified name and collect the locally owned node ids in the suppied
    set. The nodes vector is unique and ordered on output.

    \param dis : (in) discretization
    \param conds : (in) conditions in question
    \param nodes : (out) empty set on input, filled with nodal gids of local nodes
  */
  void find_conditioned_nodes(const Core::FE::Discretization& dis,
      const std::vector<Core::Conditions::Condition*>& conds, std::vector<int>& nodes);

  /// find all local nodes from discretization marked with condition
  void find_conditioned_nodes(const Core::FE::Discretization& dis,
      const std::vector<Core::Conditions::Condition*>& conds,
      std::map<int, Core::Nodes::Node*>& nodes);

  /// find all local nodes from discretization marked with condition and
  /// put them into a map indexed by Id of the condition
  void find_conditioned_nodes(const Core::FE::Discretization& dis,
      const std::vector<Core::Conditions::Condition*>& conds,
      std::map<int, std::map<int, Core::Nodes::Node*>>& nodes);

  /// find all local nodes from discretization marked with condition and
  /// put them into a vector indexed by Id of the condition
  void find_conditioned_nodes(const Core::FE::Discretization& dis,
      const std::vector<Core::Conditions::Condition*>& conds,
      std::map<int, Teuchos::RCP<std::vector<int>>>& nodes, bool use_coupling_id = true);

  /// find all local nodes from discretization marked with condition
  void find_conditioned_nodes(const Core::FE::Discretization& dis,
      const std::vector<Core::Conditions::Condition*>& conds, std::set<int>& nodeset);


  /// collect all local nodes and elements in a condition
  /*!
    \param dis discretization
    \param nodes unique map of nodes
    \param elements unique map of elements
    \param condname name of condition
   */
  void find_condition_objects(const Core::FE::Discretization& dis,
      std::map<int, Core::Nodes::Node*>& nodes,
      std::map<int, Teuchos::RCP<Core::Elements::Element>>& elements, const std::string& condname);

  /// collect all nodes (in- and excluding 'ghosts') and
  /// elements (including ghosts) in a condition
  /*!
    \param dis discretization
    \param nodes unique map of nodes
    \param ghostnodes overlapping map of nodes
    \param elements overlapping map of elements
    \param condname name of condition
   */
  void find_condition_objects(const Core::FE::Discretization& dis,
      std::map<int, Core::Nodes::Node*>& nodes, std::map<int, Core::Nodes::Node*>& gnodes,
      std::map<int, Teuchos::RCP<Core::Elements::Element>>& elements,
      const std::vector<Core::Conditions::Condition*>& conds);

  /// collect all elements in a condition including ghosts
  /*!
    \param elements overlapping map of elements
    \param vector containing condition pointers
   */
  void find_condition_objects(std::map<int, Teuchos::RCP<Core::Elements::Element>>& elements,
      const std::vector<Core::Conditions::Condition*>& conds);

  /// collect all nodes (in- and excluding 'ghosts') and
  /// elements (including ghosts) in a condition
  /*!
    \param dis discretization
    \param nodes unique map of nodes
    \param ghostnodes overlapping map of nodes
    \param elements overlapping map of elements
    \param condname name of condition
   */
  void find_condition_objects(const Core::FE::Discretization& dis,
      std::map<int, Core::Nodes::Node*>& nodes, std::map<int, Core::Nodes::Node*>& gnodes,
      std::map<int, Teuchos::RCP<Core::Elements::Element>>& elements, const std::string& condname);

  /// collect all nodes (in- and excluding 'ghosts') and
  /// elements (including ghosts) in a condition
  /*!
    \param dis discretization
    \param nodes unique map of nodes
    \param ghostnodes overlapping map of nodes
    \param elements overlapping map of elements
    \param condname name of condition
   */
  void find_condition_objects(const Core::FE::Discretization& dis,
      std::map<int, std::map<int, Core::Nodes::Node*>>& nodes,
      std::map<int, std::map<int, Core::Nodes::Node*>>& gnodes,
      std::map<int, std::map<int, Teuchos::RCP<Core::Elements::Element>>>& elements,
      const std::string& condname);

  /// collect all elements in a condition including ghosts
  /*!
    \param dis discretization
    \param elements overlapping map of elements
    \param condname name of condition
   */
  void find_condition_objects(const Core::FE::Discretization& dis,
      std::map<int, Teuchos::RCP<Core::Elements::Element>>& elements, const std::string& condname,
      const int label = -1);

  /// Find all conditions with given name that all nodes of the element have in common
  /*!
    \param ele (in) the element
    \param condname (in) name of the condition to look for
    \param condition (out) all conditions that cover all element nodes
  */
  void find_element_conditions(const Core::Elements::Element* ele, const std::string& condname,
      std::vector<Core::Conditions::Condition*>& condition);

  /// row map with nodes from condition
  Teuchos::RCP<Epetra_Map> condition_node_row_map(
      const Core::FE::Discretization& dis, const std::string& condname);

  /// col map with nodes from condition
  Teuchos::RCP<Epetra_Map> condition_node_col_map(
      const Core::FE::Discretization& dis, const std::string& condname);

  /// create the set of column element gids that have conditioned nodes
  /*!
    \note These are not elements from the condition geometry. Rather the
    gids of actual discretization elements are listed.
   */
  Teuchos::RCP<std::set<int>> conditioned_element_map(
      const Core::FE::Discretization& dis, const std::string& condname);

  /*!
   * \brief This method checks whether handed in conditions are defined on the same set of nodes
   *
   * @param[in] condition1  first condition to be tested
   * @param[in] condition2  second condition to be tested
   * @param[in] mustmatch   both conditions must match
   * @return flag indicating if both conditions are defined on the same set of nodes
   */
  bool have_same_nodes(const Core::Conditions::Condition* const condition1,
      const Core::Conditions::Condition* const condition2, bool mustmatch);

}  // namespace Core::Conditions

FOUR_C_NAMESPACE_CLOSE

#endif

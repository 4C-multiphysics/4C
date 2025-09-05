// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_condition_utils.hpp"

#include "4C_fem_condition_selector.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_linalg_utils_densematrix_communication.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace
{
  template <typename Range>
  void fill_conditioned_node_set(const Range& node_range,
      const Core::Conditions::Condition* condition, std::set<int>& node_set)
  {
    for (const Core::FE::ConstNodeRef node : node_range)
    {
      if (condition->contains_node(node.global_id()))
      {
        node_set.insert(node.global_id());
      }
    }
  }

  template <typename Range>
  std::shared_ptr<Core::LinAlg::Map> fill_condition_map(const Core::FE::Discretization& dis,
      const Range& nodeRange, const std::string& condition_name)
  {
    std::set<int> condnodeset;
    std::vector<const Core::Conditions::Condition*> conditions;
    dis.get_condition(condition_name, conditions);
    for (const auto& condition : conditions)
      fill_conditioned_node_set(nodeRange, condition, condnodeset);

    std::shared_ptr<Core::LinAlg::Map> condnodemap =
        Core::LinAlg::create_map(condnodeset, dis.get_comm());
    return condnodemap;
  }


  void find_conditioned_nodes_with_gid(const Core::FE::Discretization& dis,
      std::span<const Core::Conditions::Condition*> conditions,
      std::map<int, Core::Nodes::Node*>& nodes)
  {
    const int my_rank = Core::Communication::my_mpi_rank(dis.get_comm());
    for (auto cond : conditions)
    {
      for (int gid : *cond->get_nodes())
      {
        if (dis.have_global_node(gid) and dis.g_node(gid)->owner() == my_rank)
        {
          nodes[gid] = dis.g_node(gid);
        }
      }
    }
  }

  void find_conditioned_nodes_with_coupling_id(const Core::FE::Discretization& dis,
      std::span<const Core::Conditions::Condition*> conditions,
      std::map<int, std::map<int, Core::Nodes::Node*>>& nodes)
  {
    const int my_rank = Core::Communication::my_mpi_rank(dis.get_comm());
    for (auto* cond : conditions)
    {
      int id = cond->parameters().get<int>("coupling_id");
      for (int gid : *cond->get_nodes())
      {
        if (dis.have_global_node(gid) and dis.g_node(gid)->owner() == my_rank)
        {
          (nodes[id])[gid] = dis.g_node(gid);
        }
      }
    }
  }
}  // namespace



std::set<int> Core::Conditions::find_conditioned_node_ids(
    const Core::FE::Discretization& dis, const std::string& condition_name, LookFor look_for)
{
  std::vector<const Condition*> conditions;
  dis.get_condition(condition_name, conditions);
  return find_conditioned_node_ids(dis, conditions, look_for);
}



std::set<int> Core::Conditions::find_conditioned_node_ids(
    const Core::FE::Discretization& dis, std::span<const Condition*> conditions, LookFor look_for)
{
  std::set<int> node_set;
  const auto node_range =
      (look_for == LookFor::locally_owned) ? dis.my_row_node_range() : dis.my_col_node_range();

  for (const auto& cond : conditions)
  {
    fill_conditioned_node_set(node_range, cond, node_set);
  }

  return node_set;
}


std::multimap<int, const Core::Conditions::Condition*>
Core::Conditions::find_conditioned_node_ids_and_conditions(
    const Core::FE::Discretization& dis, std::span<const Condition*> conditions, LookFor look_for)
{
  std::multimap<int, const Condition*> node_set;
  const auto node_range =
      (look_for == LookFor::locally_owned) ? dis.my_row_node_range() : dis.my_col_node_range();

  for (const auto& cond : conditions)
  {
    std::set<int> nodes_of_current_condition;
    fill_conditioned_node_set(node_range, cond, nodes_of_current_condition);
    for (const auto& node_id : nodes_of_current_condition)
    {
      node_set.emplace(node_id, cond);
    }
  }

  return node_set;
}


std::multimap<int, const Core::Conditions::Condition*>
Core::Conditions::find_conditioned_node_ids_and_conditions(
    const Core::FE::Discretization& dis, const std::string& condition_name, LookFor look_for)
{
  std::vector<const Condition*> conditions;
  dis.get_condition(condition_name, conditions);
  return find_conditioned_node_ids_and_conditions(dis, conditions, look_for);
}



void Core::Conditions::find_conditioned_nodes(const Core::FE::Discretization& dis,
    std::span<const Condition*> conditions, std::map<int, std::shared_ptr<std::vector<int>>>& nodes,
    bool use_coupling_id)
{
  std::map<int, std::set<int>> node_set;
  const int my_rank = Core::Communication::my_mpi_rank(dis.get_comm());
  for (const auto& cond : conditions)
  {
    int id = use_coupling_id ? cond->parameters().get<int>("coupling_id") : 0;
    for (int gid : *cond->get_nodes())
    {
      if (dis.have_global_node(gid) and dis.g_node(gid)->owner() == my_rank)
      {
        node_set[id].insert(gid);
      }
    }
  }

  for (const auto& [id, gids] : node_set)
  {
    nodes[id] = std::make_shared<std::vector<int>>(gids.begin(), gids.end());
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::Conditions::find_condition_objects(const Core::FE::Discretization& dis,
    std::map<int, Core::Nodes::Node*>& nodes, std::map<int, Core::Nodes::Node*>& gnodes,
    std::map<int, std::shared_ptr<Core::Elements::Element>>& elements,
    std::span<const Condition*> conditions)
{
  find_conditioned_nodes_with_gid(dis, conditions, nodes);

  for (const auto& cond : conditions)
  {
    // get this condition's elements
    const std::map<int, std::shared_ptr<Core::Elements::Element>>& geo = cond->geometry();
    std::map<int, std::shared_ptr<Core::Elements::Element>>::const_iterator iter, pos;
    pos = elements.begin();
    for (iter = geo.begin(); iter != geo.end(); ++iter)
    {
      // get all elements locally known, including ghost elements
      pos = elements.insert(pos, *iter);
      const int* n = iter->second->node_ids();
      for (int j = 0; j < iter->second->num_node(); ++j)
      {
        const int gid = n[j];
        if (dis.have_global_node(gid))
        {
          gnodes[gid] = dis.g_node(gid);
        }
        else
          FOUR_C_THROW("All nodes of known elements must be known. Panic.");
      }
    }
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::Conditions::find_condition_objects(const Core::FE::Discretization& dis,
    std::map<int, Core::Nodes::Node*>& nodes, std::map<int, Core::Nodes::Node*>& gnodes,
    std::map<int, std::shared_ptr<Core::Elements::Element>>& elements,
    const std::string& condition_name)
{
  std::vector<const Condition*> conditions;
  dis.get_condition(condition_name, conditions);

  find_conditioned_nodes_with_gid(dis, conditions, nodes);

  for (const auto& cond : conditions)
  {
    // get this condition's elements
    const std::map<int, std::shared_ptr<Core::Elements::Element>>& geo = cond->geometry();
    std::map<int, std::shared_ptr<Core::Elements::Element>>::const_iterator iter, pos;
    pos = elements.begin();
    for (iter = geo.begin(); iter != geo.end(); ++iter)
    {
      // get all elements locally known, including ghost elements
      pos = elements.insert(pos, *iter);
      const int* n = iter->second->node_ids();
      for (int j = 0; j < iter->second->num_node(); ++j)
      {
        const int gid = n[j];
        if (dis.have_global_node(gid))
        {
          gnodes[gid] = dis.g_node(gid);
        }
        else
          FOUR_C_THROW("All nodes of known elements must be known. Panic.");
      }
    }
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::Conditions::find_condition_objects(const Core::FE::Discretization& dis,
    std::map<int, std::map<int, Core::Nodes::Node*>>& nodes,
    std::map<int, std::map<int, Core::Nodes::Node*>>& gnodes,
    std::map<int, std::map<int, std::shared_ptr<Core::Elements::Element>>>& elements,
    const std::string& condition_name)
{
  std::vector<const Condition*> conditions;
  dis.get_condition(condition_name, conditions);

  find_conditioned_nodes_with_coupling_id(dis, conditions, nodes);

  for (auto& cond : conditions)
  {
    int id = cond->parameters().get<int>("coupling_id");
    // get this condition's elements
    const std::map<int, std::shared_ptr<Core::Elements::Element>>& geo = cond->geometry();
    std::map<int, std::shared_ptr<Core::Elements::Element>>::const_iterator iter, pos;
    pos = elements[id].begin();
    for (iter = geo.begin(); iter != geo.end(); ++iter)
    {
      // get all elements locally known, including ghost elements
      pos = elements[id].insert(pos, *iter);
      const int* n = iter->second->node_ids();
      for (int j = 0; j < iter->second->num_node(); ++j)
      {
        const int gid = n[j];
        if (dis.have_global_node(gid))
        {
          gnodes[id][gid] = dis.g_node(gid);
        }
        else
          FOUR_C_THROW("All nodes of known elements must be known. Panic.");
      }
    }
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::Conditions::find_condition_objects(const Core::FE::Discretization& dis,
    std::map<int, std::shared_ptr<Core::Elements::Element>>& elements,
    const std::string& condition_name, const int label)
{
  std::vector<const Condition*> conditions;
  dis.get_condition(condition_name, conditions);

  bool checklabel = (label >= 0);

  for (auto& cond : conditions)
  {
    if (checklabel)
    {
      const int condlabel = cond->parameters().get<int>("COUPLINGID");

      if (condlabel != label) continue;  // do not consider conditions with wrong label
    }

    // get this condition's elements
    const std::map<int, std::shared_ptr<Core::Elements::Element>>& geo = cond->geometry();
    std::map<int, std::shared_ptr<Core::Elements::Element>>::const_iterator iter, pos;
    pos = elements.begin();
    for (iter = geo.begin(); iter != geo.end(); ++iter)
    {
      // get all elements locally known, including ghost elements
      pos = elements.insert(pos, *iter);
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::Conditions::find_element_conditions(const Core::Elements::Element* ele,
    const std::string& condition_name, std::vector<Condition*>& condition)
{
  const Core::Nodes::Node* const* nodes = ele->nodes();

  // We assume the conditions have unique ids. The framework has to provide
  // those.

  // the final set of conditions all nodes of this elements have in common
  std::set<Condition*> fcond;

  // we assume to always have at least one node
  // the first vector of conditions
  std::vector<Condition*> neumcond0;
  nodes[0]->get_condition(condition_name, neumcond0);

  // the first set of conditions (copy vector to set)
  std::set<Condition*> cond0;
  std::copy(neumcond0.begin(), neumcond0.end(), std::inserter(cond0, cond0.begin()));


  // loop all remaining nodes
  int iel = ele->num_node();
  for (int inode = 1; inode < iel; ++inode)
  {
    std::vector<Condition*> neumcondn;
    nodes[inode]->get_condition(condition_name, neumcondn);

    // the current set of conditions (copy vector to set)
    std::set<Condition*> condn;
    std::copy(neumcondn.begin(), neumcondn.end(), std::inserter(condn, condn.begin()));

    // intersect the first and the current conditions
    std::set_intersection(
        cond0.begin(), cond0.end(), condn.begin(), condn.end(), inserter(fcond, fcond.begin()));

    // make intersection to new starting condition
    cond0.clear();  // ensures that fcond is cleared in the next iteration
    std::swap(cond0, fcond);

    if (cond0.size() == 0)
    {
      // No intersections. Done. empty set is copied into condition-vector
      break;
    }
  }

  condition.clear();
  std::copy(cond0.begin(), cond0.end(), back_inserter(condition));
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Map> Core::Conditions::condition_node_row_map(
    const Core::FE::Discretization& dis, const std::string& condition_name)
{
  return fill_condition_map(dis, dis.my_row_node_range(), condition_name);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Map> Core::Conditions::condition_node_col_map(
    const Core::FE::Discretization& dis, const std::string& condition_name)
{
  return fill_condition_map(dis, dis.my_col_node_range(), condition_name);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<std::set<int>> Core::Conditions::conditioned_element_map(
    const Core::FE::Discretization& dis, const std::string& condition_name)
{
  std::vector<const Core::Conditions::Condition*> conditions;
  dis.get_condition(condition_name, conditions);

  std::shared_ptr<std::set<int>> condelementmap = std::make_shared<std::set<int>>();
  const int nummyelements = dis.num_my_col_elements();
  for (int i = 0; i < nummyelements; ++i)
  {
    const Core::Elements::Element* actele = dis.l_col_element(i);

    const size_t numnodes = actele->num_node();
    const Core::Nodes::Node* const* nodes = actele->nodes();
    for (size_t n = 0; n < numnodes; ++n)
    {
      const Core::Nodes::Node* actnode = nodes[n];
      const bool contains_node = std::ranges::any_of(
          conditions, [gid = actnode->id()](const auto* cond) { return cond->contains_node(gid); });
      if (contains_node) condelementmap->insert(actele->id());
    }
  }

  return condelementmap;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/


FOUR_C_NAMESPACE_CLOSE

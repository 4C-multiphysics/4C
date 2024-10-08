/*----------------------------------------------------------------------*/
/*! \file

\brief Implementation of utils on conditions

\level 1


*/
/*----------------------------------------------------------------------*/

#include "4C_fem_condition_utils.hpp"

#include "4C_fem_condition_selector.hpp"
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
  Teuchos::RCP<Epetra_Map> fill_condition_map(
      const Core::FE::Discretization& dis, const Range& nodeRange, const std::string& condname)
  {
    std::set<int> condnodeset;

    Core::Conditions::ConditionSelector conds(dis, condname);

    for (const Core::Nodes::Node* node : nodeRange)
    {
      if (conds.contains_node(node->id()))
      {
        condnodeset.insert(node->id());
      }
    }

    Teuchos::RCP<Epetra_Map> condnodemap = Core::LinAlg::create_map(condnodeset, dis.get_comm());
    return condnodemap;
  }
}  // namespace

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::Conditions::find_conditioned_nodes(
    const Core::FE::Discretization& dis, const std::string& condname, std::vector<int>& nodes)
{
  std::vector<Condition*> conds;
  dis.get_condition(condname, conds);
  find_conditioned_nodes(dis, conds, nodes);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::Conditions::find_conditioned_nodes(
    const Core::FE::Discretization& dis, const std::string& condname, std::set<int>& nodeset)
{
  std::vector<Condition*> conds;
  dis.get_condition(condname, conds);
  find_conditioned_nodes(dis, conds, nodeset);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::Conditions::find_conditioned_nodes(const Core::FE::Discretization& dis,
    const std::string& condname, std::map<int, Core::Nodes::Node*>& nodes)
{
  std::vector<Condition*> conds;
  dis.get_condition(condname, conds);
  find_conditioned_nodes(dis, conds, nodes);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::Conditions::find_conditioned_nodes(const Core::FE::Discretization& dis,
    const std::vector<Condition*>& conds, std::vector<int>& nodes)
{
  std::set<int> nodeset;
  const int myrank = dis.get_comm().MyPID();
  for (const auto& cond : conds)
  {
    for (const auto node : *cond->get_nodes())
    {
      const int gid = node;
      if (dis.have_global_node(gid) and dis.g_node(gid)->owner() == myrank)
      {
        nodeset.insert(gid);
      }
    }
  }

  nodes.reserve(nodeset.size());
  nodes.assign(nodeset.begin(), nodeset.end());
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::Conditions::find_conditioned_nodes(const Core::FE::Discretization& dis,
    const std::vector<Condition*>& conds, std::set<int>& nodeset)
{
  const int myrank = dis.get_comm().MyPID();
  for (auto cond : conds)
  {
    for (int gid : *cond->get_nodes())
    {
      if (dis.have_global_node(gid) and dis.g_node(gid)->owner() == myrank)
      {
        nodeset.insert(gid);
      }
    }
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::Conditions::find_conditioned_nodes(const Core::FE::Discretization& dis,
    const std::vector<Condition*>& conds, std::map<int, Core::Nodes::Node*>& nodes)
{
  const int myrank = dis.get_comm().MyPID();
  for (auto cond : conds)
  {
    for (int gid : *cond->get_nodes())
    {
      if (dis.have_global_node(gid) and dis.g_node(gid)->owner() == myrank)
      {
        nodes[gid] = dis.g_node(gid);
      }
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::Conditions::find_conditioned_nodes(const Core::FE::Discretization& dis,
    const std::vector<Condition*>& conds, std::map<int, Teuchos::RCP<std::vector<int>>>& nodes,
    bool use_coupling_id)
{
  std::map<int, std::set<int>> nodeset;
  const int myrank = dis.get_comm().MyPID();
  for (const auto& cond : conds)
  {
    int id = use_coupling_id ? cond->parameters().get<int>("coupling id") : 0;
    for (int gid : *cond->get_nodes())
    {
      if (dis.have_global_node(gid) and dis.g_node(gid)->owner() == myrank)
      {
        nodeset[id].insert(gid);
      }
    }
  }

  for (const auto& [id, gids] : nodeset)
  {
    nodes[id] = Teuchos::make_rcp<std::vector<int>>(gids.size());
    nodes[id]->assign(gids.begin(), gids.end());
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::Conditions::find_conditioned_nodes(const Core::FE::Discretization& dis,
    const std::vector<Condition*>& conds, std::map<int, std::map<int, Core::Nodes::Node*>>& nodes)
{
  const int myrank = dis.get_comm().MyPID();
  for (auto* cond : conds)
  {
    int id = cond->parameters().get<int>("coupling id");
    for (int gid : *cond->get_nodes())
    {
      if (dis.have_global_node(gid) and dis.g_node(gid)->owner() == myrank)
      {
        (nodes[id])[gid] = dis.g_node(gid);
      }
    }
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::Conditions::find_condition_objects(const Core::FE::Discretization& dis,
    std::map<int, Core::Nodes::Node*>& nodes,
    std::map<int, Teuchos::RCP<Core::Elements::Element>>& elements, const std::string& condname)
{
  int myrank = dis.get_comm().MyPID();
  std::vector<Condition*> conds;
  dis.get_condition(condname, conds);

  find_conditioned_nodes(dis, conds, nodes);

  for (auto& cond : conds)
  {
    // get this condition's elements
    std::map<int, Teuchos::RCP<Core::Elements::Element>>& geo = cond->geometry();
    std::map<int, Teuchos::RCP<Core::Elements::Element>>::iterator iter, pos;
    pos = elements.begin();
    for (iter = geo.begin(); iter != geo.end(); ++iter)
    {
      if (iter->second->owner() == myrank)
      {
        pos = elements.insert(pos, *iter);
      }
    }
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::Conditions::find_condition_objects(const Core::FE::Discretization& dis,
    std::map<int, Core::Nodes::Node*>& nodes, std::map<int, Core::Nodes::Node*>& gnodes,
    std::map<int, Teuchos::RCP<Core::Elements::Element>>& elements,
    const std::vector<Condition*>& conds)
{
  find_conditioned_nodes(dis, conds, nodes);

  for (const auto& cond : conds)
  {
    // get this condition's elements
    std::map<int, Teuchos::RCP<Core::Elements::Element>>& geo = cond->geometry();
    std::map<int, Teuchos::RCP<Core::Elements::Element>>::iterator iter, pos;
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
void Core::Conditions::find_condition_objects(
    std::map<int, Teuchos::RCP<Core::Elements::Element>>& elements,
    const std::vector<Condition*>& conds)
{
  for (auto cond : conds)
  {
    // get this condition's elements
    std::map<int, Teuchos::RCP<Core::Elements::Element>>& geo = cond->geometry();
    std::map<int, Teuchos::RCP<Core::Elements::Element>>::iterator iter, pos;
    pos = elements.begin();
    for (iter = geo.begin(); iter != geo.end(); ++iter)
    {
      // get all elements locally known, including ghost elements
      pos = elements.insert(pos, *iter);
    }
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::Conditions::find_condition_objects(const Core::FE::Discretization& dis,
    std::map<int, Core::Nodes::Node*>& nodes, std::map<int, Core::Nodes::Node*>& gnodes,
    std::map<int, Teuchos::RCP<Core::Elements::Element>>& elements, const std::string& condname)
{
  std::vector<Condition*> conds;
  dis.get_condition(condname, conds);

  find_conditioned_nodes(dis, conds, nodes);

  for (const auto& cond : conds)
  {
    // get this condition's elements
    std::map<int, Teuchos::RCP<Core::Elements::Element>>& geo = cond->geometry();
    std::map<int, Teuchos::RCP<Core::Elements::Element>>::iterator iter, pos;
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
    std::map<int, std::map<int, Teuchos::RCP<Core::Elements::Element>>>& elements,
    const std::string& condname)
{
  std::vector<Condition*> conds;
  dis.get_condition(condname, conds);

  find_conditioned_nodes(dis, conds, nodes);

  for (auto& cond : conds)
  {
    int id = cond->parameters().get<int>("coupling id");
    // get this condition's elements
    std::map<int, Teuchos::RCP<Core::Elements::Element>>& geo = cond->geometry();
    std::map<int, Teuchos::RCP<Core::Elements::Element>>::iterator iter, pos;
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
    std::map<int, Teuchos::RCP<Core::Elements::Element>>& elements, const std::string& condname,
    const int label)
{
  std::vector<Condition*> conds;
  dis.get_condition(condname, conds);

  bool checklabel = (label >= 0);

  for (auto& cond : conds)
  {
    if (checklabel)
    {
      const int condlabel = cond->parameters().get<int>("COUPLINGID");

      if (condlabel != label) continue;  // do not consider conditions with wrong label
    }

    // get this condition's elements
    std::map<int, Teuchos::RCP<Core::Elements::Element>>& geo = cond->geometry();
    std::map<int, Teuchos::RCP<Core::Elements::Element>>::iterator iter, pos;
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
    const std::string& condname, std::vector<Condition*>& condition)
{
  const Core::Nodes::Node* const* nodes = ele->nodes();

  // We assume the conditions have unique ids. The framework has to provide
  // those.

  // the final set of conditions all nodes of this elements have in common
  std::set<Condition*> fcond;

  // we assume to always have at least one node
  // the first vector of conditions
  std::vector<Condition*> neumcond0;
  nodes[0]->get_condition(condname, neumcond0);

  // the first set of conditions (copy vector to set)
  std::set<Condition*> cond0;
  std::copy(neumcond0.begin(), neumcond0.end(), std::inserter(cond0, cond0.begin()));


  // loop all remaining nodes
  int iel = ele->num_node();
  for (int inode = 1; inode < iel; ++inode)
  {
    std::vector<Condition*> neumcondn;
    nodes[inode]->get_condition(condname, neumcondn);

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
Teuchos::RCP<Epetra_Map> Core::Conditions::condition_node_row_map(
    const Core::FE::Discretization& dis, const std::string& condname)
{
  return fill_condition_map(dis, dis.my_row_node_range(), condname);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Map> Core::Conditions::condition_node_col_map(
    const Core::FE::Discretization& dis, const std::string& condname)
{
  return fill_condition_map(dis, dis.my_col_node_range(), condname);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<std::set<int>> Core::Conditions::conditioned_element_map(
    const Core::FE::Discretization& dis, const std::string& condname)
{
  ConditionSelector conds(dis, condname);

  Teuchos::RCP<std::set<int>> condelementmap = Teuchos::make_rcp<std::set<int>>();
  const int nummyelements = dis.num_my_col_elements();
  for (int i = 0; i < nummyelements; ++i)
  {
    const Core::Elements::Element* actele = dis.l_col_element(i);

    const size_t numnodes = actele->num_node();
    const Core::Nodes::Node* const* nodes = actele->nodes();
    for (size_t n = 0; n < numnodes; ++n)
    {
      const Core::Nodes::Node* actnode = nodes[n];

      // test if node is covered by condition
      if (conds.contains_node(actnode->id()))
      {
        condelementmap->insert(actele->id());
      }
    }
  }

  return condelementmap;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool Core::Conditions::have_same_nodes(
    const Condition* const condition1, const Condition* const condition2, const bool mustmatch)
{
  // indicates, if both conditions match
  bool matching_conditions = true;

  // get nodes of conditions
  const auto* condition1nodes = condition1->get_nodes();
  const auto* condition2nodes = condition2->get_nodes();

  // simple first check just checks the size
  if (condition1nodes->size() != condition2nodes->size())
  {
    matching_conditions = false;
    if (mustmatch)
    {
      FOUR_C_THROW(
          "Number of nodes that are defined for both conditions do not match! Did you define the "
          "conditions for the same nodesets?");
    }
  }

  // loop over all node global IDs belonging to condition1
  for (auto condition1nodegid : *condition1nodes)
  {
    bool found_node = false;
    // loop over all node global IDs belonging to condition2
    for (auto condition2nodegid : *condition2nodes)
    {
      if (condition1nodegid == condition2nodegid)
      {
        // we found the node, so set foundit to true and continue with next condition1node
        found_node = true;
        continue;
      }
    }
    // throw error if node global ID is not found in condition2
    if (!found_node)
    {
      matching_conditions = false;
      if (mustmatch)
      {
        std::cout << "Node with global ID: " << condition1nodegid
                  << "  which is part of condition: ";
        condition1->print(std::cout);
        std::cout << " is not part of condition: ";
        condition2->print(std::cout);
        FOUR_C_THROW(
            "Did you assign those conditions to the same nodeset? Please check your input file and "
            "fix this inconsistency!");
      }
    }
  }

  // when we get here everything is fine
  return matching_conditions;
}

FOUR_C_NAMESPACE_CLOSE

/*----------------------------------------------------------------------*/
/*! \file

\brief Implementation of utils on conditions

\level 1


*/
/*----------------------------------------------------------------------*/

#include <map>
#include <set>
#include <string>
#include <vector>
#include <algorithm>

#include "lib_condition_utils.H"
#include "lib_condition_selector.H"
#include "lib_discret_iterator.H"
#include "lib_globalproblem.H"

#include "linalg_utils_sparse_algebra_create.H"
#include "linalg_utils_densematrix_communication.H"
#include "io_control.H"

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void DRT::UTILS::FindConditionedNodes(
    const DRT::Discretization& dis, const std::string& condname, std::vector<int>& nodes)
{
  std::vector<DRT::Condition*> conds;
  dis.GetCondition(condname, conds);
  FindConditionedNodes(dis, conds, nodes);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void DRT::UTILS::FindConditionedNodes(
    const DRT::Discretization& dis, const std::string& condname, std::set<int>& nodeset)
{
  std::vector<DRT::Condition*> conds;
  dis.GetCondition(condname, conds);
  FindConditionedNodes(dis, conds, nodeset);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void DRT::UTILS::FindConditionedNodes(
    const DRT::Discretization& dis, const std::string& condname, std::map<int, DRT::Node*>& nodes)
{
  std::vector<DRT::Condition*> conds;
  dis.GetCondition(condname, conds);
  FindConditionedNodes(dis, conds, nodes);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void DRT::UTILS::FindConditionedNodes(const DRT::Discretization& dis,
    const std::vector<DRT::Condition*>& conds, std::vector<int>& nodes)
{
  std::set<int> nodeset;
  const int myrank = dis.Comm().MyPID();
  for (const auto& cond : conds)
  {
    for (const auto node : *cond->Nodes())
    {
      const int gid = node;
      if (dis.HaveGlobalNode(gid) and dis.gNode(gid)->Owner() == myrank)
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
void DRT::UTILS::FindConditionedNodes(const DRT::Discretization& dis,
    const std::vector<DRT::Condition*>& conds, std::set<int>& nodeset)
{
  const int myrank = dis.Comm().MyPID();
  for (auto cond : conds)
  {
    for (int gid : *cond->Nodes())
    {
      if (dis.HaveGlobalNode(gid) and dis.gNode(gid)->Owner() == myrank)
      {
        nodeset.insert(gid);
      }
    }
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void DRT::UTILS::FindConditionedNodes(const DRT::Discretization& dis,
    const std::vector<DRT::Condition*>& conds, std::map<int, DRT::Node*>& nodes)
{
  const int myrank = dis.Comm().MyPID();
  for (auto cond : conds)
  {
    for (int gid : *cond->Nodes())
    {
      if (dis.HaveGlobalNode(gid) and dis.gNode(gid)->Owner() == myrank)
      {
        nodes[gid] = dis.gNode(gid);
      }
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void DRT::UTILS::FindConditionedNodes(const DRT::Discretization& dis,
    const std::vector<DRT::Condition*>& conds, std::map<int, Teuchos::RCP<std::vector<int>>>& nodes,
    bool use_coupling_id)
{
  std::map<int, std::set<int>> nodeset;
  const int myrank = dis.Comm().MyPID();
  for (const auto& cond : conds)
  {
    int id = use_coupling_id ? cond->GetInt("coupling id") : 0;
    for (int gid : *cond->Nodes())
    {
      if (dis.HaveGlobalNode(gid) and dis.gNode(gid)->Owner() == myrank)
      {
        nodeset[id].insert(gid);
      }
    }
  }

  for (const auto& [id, gids] : nodeset)
  {
    nodes[id] = Teuchos::rcp(new std::vector<int>(gids.size()));
    nodes[id]->assign(gids.begin(), gids.end());
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void DRT::UTILS::FindConditionedNodes(const DRT::Discretization& dis,
    const std::vector<DRT::Condition*>& conds, std::map<int, std::map<int, DRT::Node*>>& nodes)
{
  const int myrank = dis.Comm().MyPID();
  for (auto* cond : conds)
  {
    int id = cond->GetInt("coupling id");
    for (int gid : *cond->Nodes())
    {
      if (dis.HaveGlobalNode(gid) and dis.gNode(gid)->Owner() == myrank)
      {
        (nodes[id])[gid] = dis.gNode(gid);
      }
    }
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void DRT::UTILS::FindConditionObjects(const DRT::Discretization& dis,
    std::map<int, DRT::Node*>& nodes, std::map<int, Teuchos::RCP<DRT::Element>>& elements,
    const std::string& condname)
{
  int myrank = dis.Comm().MyPID();
  std::vector<DRT::Condition*> conds;
  dis.GetCondition(condname, conds);

  FindConditionedNodes(dis, conds, nodes);

  for (auto& cond : conds)
  {
    // get this condition's elements
    std::map<int, Teuchos::RCP<DRT::Element>>& geo = cond->Geometry();
    std::map<int, Teuchos::RCP<DRT::Element>>::iterator iter, pos;
    pos = elements.begin();
    for (iter = geo.begin(); iter != geo.end(); ++iter)
    {
      if (iter->second->Owner() == myrank)
      {
        pos = elements.insert(pos, *iter);
      }
    }
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void DRT::UTILS::FindConditionObjects(const DRT::Discretization& dis,
    std::map<int, DRT::Node*>& nodes, std::map<int, DRT::Node*>& gnodes,
    std::map<int, Teuchos::RCP<DRT::Element>>& elements, const std::vector<DRT::Condition*>& conds)
{
  FindConditionedNodes(dis, conds, nodes);

  for (const auto& cond : conds)
  {
    // get this condition's elements
    std::map<int, Teuchos::RCP<DRT::Element>>& geo = cond->Geometry();
    std::map<int, Teuchos::RCP<DRT::Element>>::iterator iter, pos;
    pos = elements.begin();
    for (iter = geo.begin(); iter != geo.end(); ++iter)
    {
      // get all elements locally known, including ghost elements
      pos = elements.insert(pos, *iter);
      const int* n = iter->second->NodeIds();
      for (int j = 0; j < iter->second->NumNode(); ++j)
      {
        const int gid = n[j];
        if (dis.HaveGlobalNode(gid))
        {
          gnodes[gid] = dis.gNode(gid);
        }
        else
          dserror("All nodes of known elements must be known. Panic.");
      }
    }
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void DRT::UTILS::FindConditionObjects(
    std::map<int, Teuchos::RCP<DRT::Element>>& elements, const std::vector<DRT::Condition*>& conds)
{
  for (auto cond : conds)
  {
    // get this condition's elements
    std::map<int, Teuchos::RCP<DRT::Element>>& geo = cond->Geometry();
    std::map<int, Teuchos::RCP<DRT::Element>>::iterator iter, pos;
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
void DRT::UTILS::FindConditionObjects(const DRT::Discretization& dis,
    std::map<int, DRT::Node*>& nodes, std::map<int, DRT::Node*>& gnodes,
    std::map<int, Teuchos::RCP<DRT::Element>>& elements, const std::string& condname)
{
  std::vector<DRT::Condition*> conds;
  dis.GetCondition(condname, conds);

  FindConditionedNodes(dis, conds, nodes);

  for (const auto& cond : conds)
  {
    // get this condition's elements
    std::map<int, Teuchos::RCP<DRT::Element>>& geo = cond->Geometry();
    std::map<int, Teuchos::RCP<DRT::Element>>::iterator iter, pos;
    pos = elements.begin();
    for (iter = geo.begin(); iter != geo.end(); ++iter)
    {
      // get all elements locally known, including ghost elements
      pos = elements.insert(pos, *iter);
      const int* n = iter->second->NodeIds();
      for (int j = 0; j < iter->second->NumNode(); ++j)
      {
        const int gid = n[j];
        if (dis.HaveGlobalNode(gid))
        {
          gnodes[gid] = dis.gNode(gid);
        }
        else
          dserror("All nodes of known elements must be known. Panic.");
      }
    }
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void DRT::UTILS::FindConditionObjects(const DRT::Discretization& dis,
    std::map<int, std::map<int, DRT::Node*>>& nodes,
    std::map<int, std::map<int, DRT::Node*>>& gnodes,
    std::map<int, std::map<int, Teuchos::RCP<DRT::Element>>>& elements, const std::string& condname)
{
  std::vector<DRT::Condition*> conds;
  dis.GetCondition(condname, conds);

  FindConditionedNodes(dis, conds, nodes);

  for (auto& cond : conds)
  {
    int id = cond->GetInt("coupling id");
    // get this condition's elements
    std::map<int, Teuchos::RCP<DRT::Element>>& geo = cond->Geometry();
    std::map<int, Teuchos::RCP<DRT::Element>>::iterator iter, pos;
    pos = elements[id].begin();
    for (iter = geo.begin(); iter != geo.end(); ++iter)
    {
      // get all elements locally known, including ghost elements
      pos = elements[id].insert(pos, *iter);
      const int* n = iter->second->NodeIds();
      for (int j = 0; j < iter->second->NumNode(); ++j)
      {
        const int gid = n[j];
        if (dis.HaveGlobalNode(gid))
        {
          gnodes[id][gid] = dis.gNode(gid);
        }
        else
          dserror("All nodes of known elements must be known. Panic.");
      }
    }
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void DRT::UTILS::FindConditionObjects(const DRT::Discretization& dis,
    std::map<int, Teuchos::RCP<DRT::Element>>& elements, const std::string& condname,
    const int label)
{
  std::vector<DRT::Condition*> conds;
  dis.GetCondition(condname, conds);

  bool checklabel = (label >= 0);

  for (auto& cond : conds)
  {
    if (checklabel)
    {
      const int condlabel = cond->GetInt("label");

      if (condlabel != label) continue;  // do not consider conditions with wrong label
    }

    // get this condition's elements
    std::map<int, Teuchos::RCP<DRT::Element>>& geo = cond->Geometry();
    std::map<int, Teuchos::RCP<DRT::Element>>::iterator iter, pos;
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
Teuchos::RCP<Epetra_Map> DRT::UTILS::ConditionElementMap(
    const DRT::Discretization& dis, const std::string& condname, bool colmap)
{
  std::vector<DRT::Condition*> conds;
  dis.GetCondition(condname, conds);
  std::set<int> elementset;

  if (colmap)
  {
    for (auto& cond : conds)
    {
      std::map<int, Teuchos::RCP<DRT::Element>>& geometry = cond->Geometry();
      std::transform(geometry.begin(), geometry.end(),
          std::inserter(elementset, elementset.begin()),
          LINALG::select1st<std::pair<int, Teuchos::RCP<DRT::Element>>>());
    }
  }
  else
  {
    int myrank = dis.Comm().MyPID();
    for (auto& cond : conds)
    {
      for (const auto& [ele_id, ele] : cond->Geometry())
      {
        if (ele->Owner() == myrank)
        {
          elementset.insert(ele_id);
        }
      }
    }
  }

  return LINALG::CreateMap(elementset, dis.Comm());
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void DRT::UTILS::FindElementConditions(
    const DRT::Element* ele, const std::string& condname, std::vector<DRT::Condition*>& condition)
{
  const DRT::Node* const* nodes = ele->Nodes();

  // We assume the conditions have unique ids. The framework has to provide
  // those.

  // the final set of conditions all nodes of this elements have in common
  std::set<DRT::Condition*> fcond;

  // we assume to always have at least one node
  // the first vector of conditions
  std::vector<DRT::Condition*> neumcond0;
  nodes[0]->GetCondition(condname, neumcond0);

  // the first set of conditions (copy vector to set)
  std::set<DRT::Condition*> cond0;
  std::copy(neumcond0.begin(), neumcond0.end(), std::inserter(cond0, cond0.begin()));


  // loop all remaining nodes
  int iel = ele->NumNode();
  for (int inode = 1; inode < iel; ++inode)
  {
    std::vector<DRT::Condition*> neumcondn;
    nodes[inode]->GetCondition(condname, neumcondn);

    // the current set of conditions (copy vector to set)
    std::set<DRT::Condition*> condn;
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
Teuchos::RCP<Epetra_Map> DRT::UTILS::ConditionNodeRowMap(
    const DRT::Discretization& dis, const std::string& condname)
{
  RowNodeIterator iter(dis);
  return ConditionMap(dis, iter, condname);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Map> DRT::UTILS::ConditionNodeColMap(
    const DRT::Discretization& dis, const std::string& condname)
{
  ColNodeIterator iter(dis);
  return ConditionMap(dis, iter, condname);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Map> DRT::UTILS::ConditionMap(const DRT::Discretization& dis,
    const DiscretizationNodeIterator& iter, const std::string& condname)
{
  std::set<int> condnodeset;

  ConditionSelector conds(dis, condname);

  const int numnodes = iter.NumEntries();
  for (int i = 0; i < numnodes; ++i)
  {
    const DRT::Node* node = iter.Entry(i);
    if (conds.ContainsNode(node->Id()))
    {
      condnodeset.insert(node->Id());
    }
  }

  Teuchos::RCP<Epetra_Map> condnodemap = LINALG::CreateMap(condnodeset, dis.Comm());
  return condnodemap;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<std::set<int>> DRT::UTILS::ConditionedElementMap(
    const DRT::Discretization& dis, const std::string& condname)
{
  ConditionSelector conds(dis, condname);

  Teuchos::RCP<std::set<int>> condelementmap = Teuchos::rcp(new std::set<int>());
  const int nummyelements = dis.NumMyColElements();
  for (int i = 0; i < nummyelements; ++i)
  {
    const DRT::Element* actele = dis.lColElement(i);

    const size_t numnodes = actele->NumNode();
    const DRT::Node* const* nodes = actele->Nodes();
    for (size_t n = 0; n < numnodes; ++n)
    {
      const DRT::Node* actnode = nodes[n];

      // test if node is covered by condition
      if (conds.ContainsNode(actnode->Id()))
      {
        condelementmap->insert(actele->Id());
      }
    }
  }

  return condelementmap;
}


/*-----------------------------------------------------------------------*
 * Writes boundary surfaces of a volumetrically coupled problem to file  *
 * 'boundarysurfaces.log' storing the condition-Id as surface-Id. For    *
 * visualisation in gmsh and checking for tetrahedra whose four surfaces *
 * are wrongly contained in the boundary surface of the volumetric       *
 * coupling this file can be used.                         (croth 01/15) *
 *-----------------------------------------------------------------------*/
void DRT::UTILS::WriteBoundarySurfacesVolumeCoupling(
    std::map<std::vector<int>, Teuchos::RCP<DRT::Element>> surfmap, int condID, int numproc,
    int mypid)
{
  if (numproc == 1)
  {
    // Get output prefix
    std::string outputprefix = DRT::Problem::Instance()->OutputControlFile()->NewOutputFileName();
    // Create boundary surface file
    std::ostringstream sf;
    sf << outputprefix << "_boundarysurfaces.log";
    std::string boundarysurffilename;
    boundarysurffilename = sf.str();

    std::ofstream myfile;
    myfile.open(boundarysurffilename.c_str(), std::ios_base::app);
    myfile << "Surfaces in Surfmap for Coupling Condition No. " << condID + 1 << ":\n";
    myfile << "Format: [Node1, Node2, Node3, CondID] \n";
    for (std::map<std::vector<int>, Teuchos::RCP<DRT::Element>>::const_iterator iterat =
             surfmap.begin();
         iterat != surfmap.end(); ++iterat)
    {
      myfile << iterat->first[0] << " " << iterat->first[1] << " " << iterat->first[2] << " "
             << condID + 1 << "\n";
    }
    myfile << "End \n";
    myfile.close();
    std::cout << " Condition " << condID + 1 << " checked and written to file." << std::endl;
  }
  else if (mypid == 0)
  {
    std::cout << " No 'boundarysurfaces.log' written as number of procs = " << numproc
              << " is bigger than 1." << std::endl;
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool DRT::UTILS::HaveSameNodes(const DRT::Condition* const condition1,
    const DRT::Condition* const condition2, const bool mustmatch)
{
  // indicates, if both conditions match
  bool matching_conditions = true;

  // get nodes of conditions
  const auto* condition1nodes = condition1->Nodes();
  const auto* condition2nodes = condition2->Nodes();

  // simple first check just checks the size
  if (condition1nodes->size() != condition2nodes->size())
  {
    matching_conditions = false;
    if (mustmatch)
    {
      dserror(
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
        condition1->Print(std::cout);
        std::cout << " is not part of condition: ";
        condition2->Print(std::cout);
        dserror(
            "Did you assign those conditions to the same nodeset? Please check your input file and "
            "fix this inconsistency!");
      }
    }
  }

  // when we get here everything is fine
  return matching_conditions;
}

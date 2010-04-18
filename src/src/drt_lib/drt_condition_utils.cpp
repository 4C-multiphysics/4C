/*----------------------------------------------------------------------*/
/*!
\file drt_condition_utils.cpp

\brief

<pre>
Maintainer: Axel Gerstenberger
            gerstenberger@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15236
</pre>
*/
/*----------------------------------------------------------------------*/
#ifdef CCADISCRET

#include "drt_condition_utils.H"
#include "drt_condition_selector.H"
#include "standardtypes_cpp.H"
//#include "adapter_coupling_mortar.H"

#include "drt_globalproblem.H"
#include "drt_utils.H"
#include "linalg_utils.H"

#include <map>
#include <set>
#include <string>
#include <vector>
#include <algorithm>

/*----------------------------------------------------------------------*
 |                                                       m.gee 06/01    |
 | general problem data                                                 |
 | global variable GENPROB genprob is defined in global_control.c       |
 *----------------------------------------------------------------------*/
extern struct _GENPROB     genprob;


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void DRT::UTILS::FindConditionedNodes(const DRT::Discretization& dis,
                                      std::string condname, std::vector<int>& nodes)
{
  std::vector<DRT::Condition*> conds;
  dis.GetCondition(condname, conds);
  FindConditionedNodes(dis,conds,nodes);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void DRT::UTILS::FindConditionedNodes(const DRT::Discretization& dis, std::string condname, std::set<int>& nodeset)
{
  std::vector<DRT::Condition*> conds;
  dis.GetCondition(condname, conds);
  FindConditionedNodes(dis,conds,nodeset);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void DRT::UTILS::FindConditionedNodes(const DRT::Discretization& dis, std::string condname, map<int, DRT::Node*>& nodes)
{
  std::vector<DRT::Condition*> conds;
  dis.GetCondition(condname, conds);
  FindConditionedNodes(dis,conds,nodes);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void DRT::UTILS::FindConditionedNodes(const DRT::Discretization& dis,
                                      const std::vector<DRT::Condition*>& conds,
                                      std::vector<int>& nodes)
{
  std::set<int> nodeset;
  const int myrank = dis.Comm().MyPID();
  for (unsigned i=0; i<conds.size(); ++i)
  {
    const std::vector<int>* n = conds[i]->Nodes();
    for (unsigned j=0; j<n->size(); ++j)
    {
      const int gid = (*n)[j];
      if (dis.HaveGlobalNode(gid) and dis.gNode(gid)->Owner()==myrank)
      {
        nodeset.insert(gid);
      }
    }
  }

  nodes.reserve(nodeset.size());
  nodes.assign(nodeset.begin(),nodeset.end());
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void DRT::UTILS::FindConditionedNodes(const DRT::Discretization& dis,
                                      const std::vector<DRT::Condition*>& conds,
                                      std::set<int>& nodeset)
{
  const int myrank = dis.Comm().MyPID();
  for (unsigned i=0; i<conds.size(); ++i)
  {
    const std::vector<int>* n = conds[i]->Nodes();
    for (unsigned j=0; j<n->size(); ++j)
    {
      const int gid = (*n)[j];
      if (dis.HaveGlobalNode(gid) and dis.gNode(gid)->Owner()==myrank)
      {
        nodeset.insert(gid);
      }
    }
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void DRT::UTILS::FindConditionedNodes(const DRT::Discretization& dis,
                                      const std::vector<DRT::Condition*>& conds,
                                      map<int, DRT::Node*>& nodes)
{
  const int myrank = dis.Comm().MyPID();
  for (unsigned i=0; i<conds.size(); ++i)
  {
    const std::vector<int>* n = conds[i]->Nodes();
    for (unsigned j=0; j<n->size(); ++j)
    {
      const int gid = (*n)[j];
      if (dis.HaveGlobalNode(gid) and dis.gNode(gid)->Owner()==myrank)
      {
        nodes[gid] = dis.gNode(gid);
      }
    }
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void DRT::UTILS::FindConditionObjects(const DRT::Discretization& dis,
                                      map<int, DRT::Node*>& nodes,
                                      map<int, RCP<DRT::Element> >& elements,
                                      const string& condname)
{
  int myrank = dis.Comm().MyPID();
  vector<DRT::Condition*> conds;
  dis.GetCondition(condname, conds);

  FindConditionedNodes(dis, conds, nodes);

  for (unsigned i = 0; i < conds.size(); ++i)
  {
    // get this condition's elements
    map< int, RCP< DRT::Element > >& geo = conds[i]->Geometry();
    map< int, RCP< DRT::Element > >::iterator iter, pos;
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
                                      map<int, DRT::Node*>& nodes,
                                      map<int, DRT::Node*>& gnodes,
                                      map<int, RCP<DRT::Element> >& elements,
                                      const string& condname)
{
  vector<DRT::Condition*> conds;
  dis.GetCondition(condname, conds);

  FindConditionedNodes(dis, conds, nodes);

  for (unsigned i = 0; i < conds.size(); ++i)
  {
    // get this condition's elements
    map< int, RCP< DRT::Element > >& geo = conds[i]->Geometry();
    map< int, RCP< DRT::Element > >::iterator iter, pos;
    pos = elements.begin();
    for (iter = geo.begin(); iter != geo.end(); ++iter)
    {
      // get all elements locally known, including ghost elements
      pos = elements.insert(pos, *iter);
      const int* n = iter->second->NodeIds();
      for (int j=0; j < iter->second->NumNode(); ++j)
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
                                      map<int, RCP<DRT::Element> >& elements,
                                      const string& condname)
{
  vector<DRT::Condition*> conds;
  dis.GetCondition(condname, conds);
  for (unsigned i = 0; i < conds.size(); ++i)
  {
    // get this condition's elements
    map< int, RCP< DRT::Element > >& geo = conds[i]->Geometry();
    map< int, RCP< DRT::Element > >::iterator iter, pos;
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
Teuchos::RCP<Epetra_Map> DRT::UTILS::ConditionElementMap(const DRT::Discretization& dis,
                                                         std::string condname,
                                                         bool colmap)
{
  std::vector<DRT::Condition*> conds;
  dis.GetCondition(condname, conds);
  std::set<int> elementset;

  if (colmap)
  {
    for (unsigned i=0; i<conds.size(); ++i)
    {
      std::map<int,RCP<DRT::Element> >& geometry = conds[i]->Geometry();
      std::transform(geometry.begin(),
                     geometry.end(),
                     std::inserter(elementset,elementset.begin()),
                     LINALG::select1st<std::pair<int,RCP<DRT::Element> > >());
    }
  }
  else
  {
    int myrank = dis.Comm().MyPID();
    for (unsigned i=0; i<conds.size(); ++i)
    {
      std::map<int,RCP<DRT::Element> >& geometry = conds[i]->Geometry();
      for (std::map<int,RCP<DRT::Element> >::const_iterator iter=geometry.begin();
           iter!=geometry.end();
           ++iter)
      {
        if (iter->second->Owner()==myrank)
        {
          elementset.insert(iter->first);
        }
      }
    }
  }

  return LINALG::CreateMap(elementset, dis.Comm());
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void DRT::UTILS::FindElementConditions(const DRT::Element* ele, const std::string& condname, std::vector<DRT::Condition*>& condition)
{
  const DRT::Node* const* nodes = ele->Nodes();

  // We assume the conditions have unique ids. The framework has to provide
  // those.

  // we assume to always have at least one node
  std::vector<DRT::Condition*> neumcond0;
  nodes[0]->GetCondition(condname,neumcond0);

  // the first set of conditions
  std::set<DRT::Condition*> cond0;
  std::copy(neumcond0.begin(),
            neumcond0.end(),
            std::inserter(cond0,cond0.begin()));

  // the final set
  std::set<DRT::Condition*> fcond;

  // loop all remaining nodes

  int iel = ele->NumNode();
  for (int inode=1; inode<iel; ++inode)
  {
    std::vector<DRT::Condition*> neumcondn;
    nodes[inode]->GetCondition(condname,neumcondn);

    std::set<DRT::Condition*> condn;
    std::copy(neumcondn.begin(),
              neumcondn.end(),
              std::inserter(condn,condn.begin()));

    // intersect the first and the current conditions
    std::set_intersection(cond0.begin(),cond0.end(),
                          condn.begin(),condn.end(),
                          inserter(fcond,fcond.begin()));

    // make intersection to new starting condition
    cond0.clear();
    std::swap(cond0,fcond);

    if (fcond.size()==0)
      // No intersections. Done.
      break;
  }

  condition.clear();
  std::copy(cond0.begin(),cond0.end(),back_inserter(condition));
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Map> DRT::UTILS::ConditionNodeRowMap(const DRT::Discretization& dis,
                                                         const std::string& condname)
{
  RowNodeIterator iter(dis);
  return ConditionMap(dis,iter,condname);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Map> DRT::UTILS::ConditionNodeColMap(const DRT::Discretization& dis,
                                                         const std::string& condname)
{
  ColNodeIterator iter(dis);
  return ConditionMap(dis,iter,condname);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Map> DRT::UTILS::ConditionMap(const DRT::Discretization& dis,
                                                  const DiscretizationNodeIterator& iter,
                                                  const std::string& condname)
{
  std::set<int> condnodeset;

  ConditionSelector conds(dis, condname);

  const int numnodes = iter.NumEntries();
  for (int i=0; i<numnodes; ++i)
  {
    const DRT::Node* node = iter.Entry(i);
    if (conds.ContainsNode(node->Id()))
    {
      condnodeset.insert(node->Id());
    }
  }

  Teuchos::RCP<Epetra_Map> condnodemap =
    LINALG::CreateMap(condnodeset, dis.Comm());
  return condnodemap;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<std::set<int> > DRT::UTILS::ConditionedElementMap(const DRT::Discretization& dis,
                                                               const std::string& condname)
{
  ConditionSelector conds(dis, condname);

  Teuchos::RCP<std::set<int> > condelementmap = Teuchos::rcp(new set<int>());
  const int nummyelements = dis.NumMyColElements();
  for (int i=0; i<nummyelements; ++i)
  {
    const DRT::Element* actele = dis.lColElement(i);

    const size_t numnodes = actele->NumNode();
    const DRT::Node*const* nodes = actele->Nodes();
    for (size_t n=0; n<numnodes; ++n)
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


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void DRT::UTILS::SetupNDimExtractor(const DRT::Discretization& dis,
                                    std::string condname,
                                    LINALG::MapExtractor& extractor)
{
  SetupExtractor(dis,condname,0,genprob.ndim,rcp(new Epetra_Map(*(dis.DofRowMap()))),extractor);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void DRT::UTILS::SetupNDimExtractor(const DRT::Discretization& dis,
                                    std::string condname,
                                    Teuchos::RCP<Epetra_Map> fullmap,
                                    LINALG::MapExtractor& extractor)
{
  SetupExtractor(dis,condname,0,genprob.ndim,fullmap,extractor);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void DRT::UTILS::SetupExtractor(const DRT::Discretization& dis,
                                std::string condname,
                                unsigned startdim,
                                unsigned enddim,
                                LINALG::MapExtractor& extractor)
{
  SetupExtractor(dis,condname,0,genprob.ndim,rcp(new Epetra_Map(*(dis.DofRowMap()))),extractor);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void DRT::UTILS::SetupExtractor(const DRT::Discretization& dis,
                                std::string condname,
                                unsigned startdim,
                                unsigned enddim,
                                Teuchos::RCP<Epetra_Map> fullmap,
                                LINALG::MapExtractor& extractor)
{
  MultiConditionSelector mcs;
  mcs.AddSelector(rcp(new NDimConditionSelector(dis,condname,startdim,enddim)));
  mcs.SetupExtractor(dis,*fullmap,extractor);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<DRT::Discretization> DRT::UTILS::CreateDiscretizationFromCondition(
    Teuchos::RCP<DRT::Discretization>  sourcedis,
        const string&                  condname,
        const string&                  discret_name,
        const string&                  element_name,
        const vector<string>&          conditions_to_copy
        )
{
  RCP<Epetra_Comm> com = rcp(sourcedis->Comm().Clone());
  RCP<DRT::Discretization> conditiondis = rcp(new DRT::Discretization(discret_name,com));

  // make sure connectivity is all set
  // we don't care, whether dofs exist or not
  if (!sourcedis->Filled())
    dserror("sourcedis is not filled");

  const int myrank = conditiondis->Comm().MyPID();
  const Epetra_Map* sourcenoderowmap = sourcedis->NodeRowMap();

  // We need to test for all elements (including ghosted ones) to
  // catch all nodes
  map<int, RCP<DRT::Element> >  sourceelements;
  DRT::UTILS::FindConditionObjects(*sourcedis, sourceelements, condname);

  set<int> rownodeset;
  set<int> colnodeset;

  // construct new elements
  for (map<int, RCP<DRT::Element> >::const_iterator sourceele_iter = sourceelements.begin();
       sourceele_iter != sourceelements.end();
       ++sourceele_iter)
  {
    const RCP<DRT::Element> sourceele = sourceele_iter->second;

    // get global node ids
    vector<int> nids;
    nids.reserve(sourceele->NumNode());
    transform(sourceele->Nodes(), sourceele->Nodes()+sourceele->NumNode(),
              back_inserter(nids), mem_fun(&DRT::Node::Id));

    if (std::count_if(nids.begin(), nids.end(), DRT::UTILS::MyGID(sourcenoderowmap))==0)
    {
      dserror("no own node in element %d", sourceele->Id());
    }

    if (std::count_if(nids.begin(), nids.end(),
                      DRT::UTILS::MyGID(sourcedis->NodeColMap())) < static_cast<int>(nids.size()))
    {
      dserror("element %d has remote non-ghost nodes",sourceele->Id());
    }

    copy(nids.begin(), nids.end(),
         inserter(colnodeset, colnodeset.begin()));

    // copy node ids of condition ele to rownodeset but leave those that do
    // not belong to this processor
    remove_copy_if(nids.begin(), nids.end(),
                   inserter(rownodeset, rownodeset.begin()),
                   not1(DRT::UTILS::MyGID(sourcenoderowmap)));

    // Do not clone ghost elements here! Those will be handled by the
    // discretization itself.
    if (sourceele->Owner()==myrank)
    {
      // create an element with the same global element id
      RCP<DRT::Element> condele = DRT::UTILS::Factory(element_name, "Polynomial", sourceele->Id(), myrank);

      // set the same global node ids to the new element
      condele->SetNodeIds(nids.size(), &nids[0]);

      // add element
      conditiondis->AddElement(condele);
    }
  }

  // construct new nodes, which use the same global id as the source nodes
  for (int i=0; i<sourcenoderowmap->NumMyElements(); ++i)
  {
    const int gid = sourcenoderowmap->GID(i);
    if (rownodeset.find(gid)!=rownodeset.end())
    {
      const DRT::Node* sourcenode = sourcedis->lRowNode(i);
      conditiondis->AddNode(rcp(new DRT::Node(gid, sourcenode->X(), myrank)));
    }
  }

  // we get the node maps almost for free
  vector<int> condnoderowvec(rownodeset.begin(), rownodeset.end());
  rownodeset.clear();
  RCP<Epetra_Map> condnoderowmap = rcp(new Epetra_Map(-1,
                                                      condnoderowvec.size(),
                                                      &condnoderowvec[0],
                                                      0,
                                                      conditiondis->Comm()));
  condnoderowvec.clear();

  vector<int> condnodecolvec(colnodeset.begin(), colnodeset.end());
  colnodeset.clear();
  RCP<Epetra_Map> condnodecolmap = rcp(new Epetra_Map(-1,
                                                      condnodecolvec.size(),
                                                      &condnodecolvec[0],
                                                      0,
                                                      conditiondis->Comm()));
  condnodecolvec.clear();

  // copy selected conditions to the new discretization
  for (vector<string>::const_iterator conditername = conditions_to_copy.begin();
       conditername != conditions_to_copy.end();
       ++conditername)
  {
    vector<DRT::Condition*> conds;
    sourcedis->GetCondition(*conditername, conds);
    for (unsigned i=0; i<conds.size(); ++i)
    {
      // We use the same nodal ids and therefore we can just copy the conditions.
      conditiondis->SetCondition(*conditername, rcp(new DRT::Condition(*conds[i])));
    }
  }

  // redistribute nodes to column (ghost) map
  RedistributeWithNewNodalDistribution(*conditiondis, *condnoderowmap, *condnodecolmap);
  conditiondis->FillComplete();

  return conditiondis;
}


/*----------------------------------------------------------------------
 * collects elements by labels (have to be implemented in the           *
 * corresponding condition)                                             *
 *----------------------------------------------------------------------*/
void DRT::UTILS::CollectElementsByConditionLabel(
    const DRT::Discretization&           discret,
    std::map<int,std::set<int> >&        elementsByLabel,
    const string&                        name)
{
  // Reset
  elementsByLabel.clear();
  // get condition
  vector< DRT::Condition* >  conditions;
  discret.GetCondition (name, conditions);

  // collect elements by xfem coupling label
  for(vector<DRT::Condition*>::const_iterator conditer = conditions.begin(); conditer!= conditions.end(); ++conditer)
  {
    const DRT::Condition* condition = *conditer;
    const int label = condition->GetInt("label");
    for (int iele=0;iele < discret.NumMyColElements(); ++iele)
    {
      // for each element, check, whether all nodes belong to same condition label
      const DRT::Element* element = discret.lColElement(iele);
      int nodecounter = 0;
      for (int inode=0; inode < element->NumNode(); ++inode)
      {
        const DRT::Node* node = element->Nodes()[inode];
        if (condition->ContainsNode(node->Id()))
          nodecounter++;
      }
      // if all nodes belong to label, then this element gets a label entry
      if (nodecounter == element->NumNode())
        elementsByLabel[label].insert(element->Id());
    }
  }
  int numOfCollectedIds = 0;
  for (std::map<int,std::set<int> >::const_iterator entry = elementsByLabel.begin();
      entry != elementsByLabel.end();
      ++entry)
  {
    numOfCollectedIds += entry->second.size();
  }

  if(discret.NumMyColElements() != numOfCollectedIds)
    dserror("not all elements collected.");
}

#endif

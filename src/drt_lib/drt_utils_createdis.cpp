/*----------------------------------------------------------------------*/
/*!
\file drt_utils_createdis.cpp

\brief utility functions for automatic creation of a discretization
       from an existing one (e.g. ALE from Fluid)

<pre>
Maintainer: Georg Bauer
            bauer@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15252
</pre>
*/
/*----------------------------------------------------------------------*/

#include "drt_utils_createdis.H"
#include "../drt_lib/drt_utils_parallel.H"


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void DRT::UTILS::DiscretizationCreatorBase::InitialChecks(
    Teuchos::RCP<DRT::Discretization> sourcedis,
    Teuchos::RCP<DRT::Discretization> targetdis)
{
  // is the source discretization ready?
  if (!sourcedis->Filled()) sourcedis->FillComplete(false,false,false);

  // is the target discretization really empty?
  if (targetdis->NumGlobalElements() or targetdis->NumGlobalNodes())
  {
    dserror("There are %d elements and %d nodes in target discretization. Panic.",
        targetdis->NumGlobalElements(), targetdis->NumGlobalNodes());
  }
  // Ok. Let's go on
  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void DRT::UTILS::DiscretizationCreatorBase::CreateNodes(
    Teuchos::RCP<DRT::Discretization> sourcedis,
    Teuchos::RCP<DRT::Discretization> targetdis,
    set<int>& rownodeset,
    set<int>& colnodeset,
    const bool isnurbsdis
    )
{
  // prepare some variables we need
  int myrank = targetdis->Comm().MyPID();
  const Epetra_Map* sourcenoderowmap = sourcedis->NodeRowMap();

  // construct nodes / control points in the new discretization
  if (isnurbsdis==false)
  {
    for (int i=0; i<sourcenoderowmap->NumMyElements(); ++i)
    {
      int gid = sourcenoderowmap->GID(i);
      if (rownodeset.find(gid)!=rownodeset.end())
      {
        DRT::Node* fluidnode = sourcedis->lRowNode(i);
        targetdis->AddNode(rcp(new DRT::Node(gid, fluidnode->X(), myrank)));
      }
    }
  }
  else
  {
    for (int i=0; i<sourcenoderowmap->NumMyElements(); ++i)
    {
      const int gid = sourcenoderowmap->GID(i);
      if (rownodeset.find(gid)!=rownodeset.end())
      {
        DRT::NURBS::ControlPoint* fluidnode
        =
          dynamic_cast<DRT::NURBS::ControlPoint* >(sourcedis->lRowNode(i));
          targetdis->AddNode(rcp(new DRT::NURBS::ControlPoint(gid, fluidnode->X(),fluidnode->W(),myrank)));
      }
    }
  }

  // ensure Reset() is called on targetdis on all procs (including procs without rownodes)
  targetdis->CheckFilledGlobally();

  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
RCP<Epetra_Map> DRT::UTILS::DiscretizationCreatorBase::CreateNodeRowMap(
    set<int>& rownodeset,Teuchos::RCP<DRT::Discretization> targetdis)
{
  // we get the node maps almost for free
  vector<int> targetnoderowvec(rownodeset.begin(), rownodeset.end());
  rownodeset.clear();

  RCP<Epetra_Map> targetnoderowmap = rcp(new Epetra_Map(-1,
      targetnoderowvec.size(),
      &targetnoderowvec[0],
      0,
      targetdis->Comm()));
  targetnoderowvec.clear();

  return targetnoderowmap ;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
RCP<Epetra_Map> DRT::UTILS::DiscretizationCreatorBase::CreateNodeColMap(
    set<int>& colnodeset,Teuchos::RCP<DRT::Discretization> targetdis)
{
  // we get the node maps almost for free
  vector<int> targetnodecolvec(colnodeset.begin(), colnodeset.end());
  colnodeset.clear();
  RCP<Epetra_Map> targetnodecolmap = rcp(new Epetra_Map(-1,
      targetnodecolvec.size(),
      &targetnodecolvec[0],
      0,
      targetdis->Comm()));
  targetnodecolvec.clear();

  return targetnodecolmap;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void DRT::UTILS::DiscretizationCreatorBase::CopyConditions(
    const Teuchos::RCP<DRT::Discretization> sourcedis,
    Teuchos::RCP<DRT::Discretization> targetdis,
    const map<string,string>& conditions_to_copy)
{
  // copy selected conditions to the new discretization (and rename them if desired)
  for (map<string,string>::const_iterator conditername = conditions_to_copy.begin();
  conditername != conditions_to_copy.end();
  ++conditername)
  {
    vector<DRT::Condition*> conds;
    sourcedis->GetCondition((*conditername).first, conds);
    for (unsigned i=0; i<conds.size(); ++i)
    {
      // We use the same nodal ids and therefore we can just copy the conditions.
      // The string-map gives the new condition names
      // (e.g. renaming from TransportDirichlet to Dirichlet)
      targetdis->SetCondition((*conditername).second, rcp(new DRT::Condition(*conds[i])));
    }
    conds.clear();
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void DRT::UTILS::DiscretizationCreatorBase::Finalize(
    const Teuchos::RCP<DRT::Discretization> sourcedis,
    Teuchos::RCP<DRT::Discretization> targetdis)
{
  // redistribute nodes to column (ghost) map
  DRT::UTILS::RedistributeWithNewNodalDistribution(*targetdis, *targetnoderowmap_, *targetnodecolmap_);
  targetdis->FillComplete();

  // extra work for NURBS discretizations

  // try to cast sourcedis to NurbsDiscretisation
  DRT::NURBS::NurbsDiscretization* nurbsdis
    =
    dynamic_cast<DRT::NURBS::NurbsDiscretization*>(&(*(sourcedis)));

  if(nurbsdis!=NULL)
  {
    DRT::NURBS::NurbsDiscretization* targetnurbsdis
      =
      dynamic_cast<DRT::NURBS::NurbsDiscretization*>(&(*(targetdis)));

    if(targetnurbsdis==NULL)
    {
      dserror("Nurbs source discretization but no nurbs target discretization\n");
    }

    Teuchos::RCP<DRT::NURBS::Knotvector> knots
      =
      Teuchos::rcp(new DRT::NURBS::Knotvector(*(nurbsdis->GetKnotVector())));

    // reset offsets
    int smallest_gid_in_dis=targetnurbsdis->ElementRowMap()->MinAllGID();
    knots->FinishKnots(smallest_gid_in_dis);

    targetnurbsdis->SetKnotVector(knots);
  }

  // at the end, we do several checks to ensure that we really have identical
  // distributions of elements and nodes over processors (as expected!)
  // We should not perform this check if the new discretization is only a subset of the
  // source discretization. But currently, this option is not used.
  if (not sourcedis->NodeRowMap()->SameAs(*(targetdis->NodeRowMap())))
    dserror("NodeRowMaps of source and target discretization are different!");
  if (not sourcedis->NodeColMap()->SameAs(*(targetdis->NodeColMap())))
    dserror("NodeColMaps of source and target discretization are different!");
  if (not sourcedis->ElementRowMap()->SameAs(*(targetdis->ElementRowMap())))
    dserror("ElementRowMaps of source and target discretization are different!");
  if (not sourcedis->ElementRowMap()->SameAs(*(targetdis->ElementRowMap())))
    dserror("ElementRowMaps of source and target discretization are different!");

  // all done ;-)
  return;
}




/*----------------------------------------------------------------------*/
/*!
\file adapter_coupling.cpp

\brief

<pre>
Maintainer: Matthias Mayr
            mayr@mhpc.mw.tum.de
            089 - 289-10362
</pre>
*/
/*----------------------------------------------------------------------*/


#include <algorithm>

#include "adapter_coupling.H"
#include "../drt_lib/drt_nodematchingoctree.H"
#include "../linalg/linalg_utils.H"
#include "../drt_lib/drt_condition_utils.H"
#include "../drt_lib/drt_discret.H"


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
ADAPTER::Coupling::Coupling()
{
  masterdofmap_ = Teuchos::null;
  slavedofmap_ = Teuchos::null;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ADAPTER::Coupling::SetupConditionCoupling(const DRT::Discretization& masterdis,
                                               Teuchos::RCP<const Epetra_Map> mastercondmap,
                                               const DRT::Discretization& slavedis,
                                               Teuchos::RCP<const Epetra_Map> slavecondmap,
                                               const std::string& condname,
                                               const int numdof,
                                               bool matchall)
{
  std::vector<int> masternodes;
  DRT::UTILS::FindConditionedNodes(masterdis,condname,masternodes);
  std::vector<int> slavenodes;
  DRT::UTILS::FindConditionedNodes(slavedis,condname,slavenodes);

  int localmastercount = static_cast<int>(masternodes.size());
  int mastercount;
  int localslavecount = static_cast<int>(slavenodes.size());
  int slavecount;

  masterdis.Comm().SumAll(&localmastercount,&mastercount,1);
  slavedis.Comm().SumAll(&localslavecount,&slavecount,1);

  if (mastercount != slavecount)
    dserror("got %d master nodes but %d slave nodes for coupling",
            mastercount,slavecount);

  SetupCoupling(masterdis, slavedis, masternodes, slavenodes, numdof, matchall);

  // test for completeness
  if (static_cast<int>(masternodes.size())*numdof != masterdofmap_->NumMyElements())
    dserror("failed to setup master nodes properly");
  if (static_cast<int>(slavenodes.size())*numdof != slavedofmap_->NumMyElements())
    dserror("failed to setup slave nodes properly");

  // Now swap in the maps we already had.
  // So we did a little more work than required. But there are cases
  // where we have to do that work (fluid-ale coupling) and we want to
  // use just one setup implementation.
  //
  // The point is to make sure there is only one map for each
  // interface.

  if (not masterdofmap_->PointSameAs(*mastercondmap))
    dserror("master dof map mismatch");

  if (not slavedofmap_->PointSameAs(*slavecondmap))
  {
    dserror("slave dof map mismatch");
  }

  masterdofmap_ = mastercondmap;
  masterexport_ = Teuchos::rcp(new Epetra_Export(*permmasterdofmap_, *masterdofmap_));

  slavedofmap_ = slavecondmap;
  slaveexport_ = Teuchos::rcp(new Epetra_Export(*permslavedofmap_, *slavedofmap_));
}



/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ADAPTER::Coupling::SetupConstrainedConditionCoupling(const DRT::Discretization& masterdis,
                                                          Teuchos::RCP<const Epetra_Map> mastercondmap,
                                                          const DRT::Discretization& slavedis,
                                                          Teuchos::RCP<const Epetra_Map> slavecondmap,
                                                          const std::string& condname1,
                                                          const std::string& condname2,
                                                          const int numdof,
                                                          bool matchall)
{
  std::vector<int> masternodes1;
  DRT::UTILS::FindConditionedNodes(masterdis,condname1,masternodes1);
  std::vector<int> slavenodes1;
  DRT::UTILS::FindConditionedNodes(slavedis,condname1,slavenodes1);

  std::set<int> masternodes2;
  DRT::UTILS::FindConditionedNodes(masterdis,condname2,masternodes2);
  std::set<int> slavenodes2;
  DRT::UTILS::FindConditionedNodes(slavedis,condname2,slavenodes2);

  // now find all those elements of slavenodes1 and masternodes1 that
  // do not belong to slavenodes2 and masternodes2 at the same time

  std::vector<int> masternodes;
  std::vector<int> slavenodes;

  for (unsigned int i=0; i<masternodes1.size(); ++i)
  {
    if (masternodes2.find(masternodes1[i]) == masternodes2.end())
      masternodes.push_back(masternodes1[i]);
  }

  for (unsigned int i=0; i<slavenodes1.size(); ++i)
  {
    if (slavenodes2.find(slavenodes1[i]) == slavenodes2.end())
      slavenodes.push_back(slavenodes1[i]);
  }

  int localmastercount = static_cast<int>(masternodes.size());
  int mastercount;
  int localslavecount = static_cast<int>(slavenodes.size());
  int slavecount;

  masterdis.Comm().SumAll(&localmastercount,&mastercount,1);
  slavedis.Comm().SumAll(&localslavecount,&slavecount,1);

  if (mastercount != slavecount and matchall)
    dserror("got %d master nodes but %d slave nodes for coupling",
            mastercount,slavecount);

  SetupCoupling(masterdis, slavedis, masternodes, slavenodes, numdof, matchall);

  // test for completeness
  if (static_cast<int>(masternodes.size())*numdof != masterdofmap_->NumMyElements())
    dserror("failed to setup master nodes properly");
  if (static_cast<int>(slavenodes.size())*numdof != slavedofmap_->NumMyElements())
    dserror("failed to setup slave nodes properly");

  // Now swap in the maps we already had.
  // So we did a little more work than required. But there are cases
  // where we have to do that work (fluid-ale coupling) and we want to
  // use just one setup implementation.
  //
  // The point is to make sure there is only one map for each
  // interface.

  if (not masterdofmap_->PointSameAs(*mastercondmap))
    dserror("master dof map mismatch");

  if (not slavedofmap_->PointSameAs(*slavecondmap))
    dserror("slave dof map mismatch");

  masterdofmap_ = mastercondmap;
  masterexport_ = Teuchos::rcp(new Epetra_Export(*permmasterdofmap_, *masterdofmap_));

  slavedofmap_ = slavecondmap;
  slaveexport_ = Teuchos::rcp(new Epetra_Export(*permslavedofmap_, *slavedofmap_));
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ADAPTER::Coupling::SetupCoupling(const DRT::Discretization& masterdis,
                                      const DRT::Discretization& slavedis,
                                      const std::vector<int>& masternodes,
                                      const std::vector<int>& slavenodes,
                                      const int numdof,
                                      const bool matchall,
                                      const double tolerance)
{
  std::vector<int> patchedmasternodes(masternodes);
  std::vector<int> permslavenodes;
  MatchNodes(masterdis, slavedis, patchedmasternodes, permslavenodes, slavenodes, matchall, tolerance);

  // Epetra maps in original distribution

  Teuchos::RCP<Epetra_Map> masternodemap =
    Teuchos::rcp(new Epetra_Map(-1, patchedmasternodes.size(), &patchedmasternodes[0], 0, masterdis.Comm()));

  Teuchos::RCP<Epetra_Map> slavenodemap =
    Teuchos::rcp(new Epetra_Map(-1, slavenodes.size(), &slavenodes[0], 0, slavedis.Comm()));

  Teuchos::RCP<Epetra_Map> permslavenodemap =
    Teuchos::rcp(new Epetra_Map(-1, permslavenodes.size(), &permslavenodes[0], 0, slavedis.Comm()));

  FinishCoupling(masterdis, slavedis, masternodemap, slavenodemap, permslavenodemap, numdof);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ADAPTER::Coupling::SetupCoupling(const DRT::Discretization& masterdis,
                                      const DRT::Discretization& slavedis,
                                      const Epetra_Map& masternodes,
                                      const Epetra_Map& slavenodes,
                                      const int numdof,
                                      const bool matchall,
                                      const double tolerance)
{
  if (masternodes.NumGlobalElements()!=slavenodes.NumGlobalElements() and matchall)
    dserror("got %d master nodes but %d slave nodes for coupling",
            masternodes.NumGlobalElements(),
            slavenodes.NumGlobalElements());

  std::vector<int> mastervect(masternodes.MyGlobalElements(),
                         masternodes.MyGlobalElements() + masternodes.NumMyElements());
  std::vector<int> slavevect(slavenodes.MyGlobalElements(),
                        slavenodes.MyGlobalElements() + slavenodes.NumMyElements());
  std::vector<int> permslavenodes;

  MatchNodes(masterdis, slavedis, mastervect, permslavenodes, slavevect, matchall, tolerance);

  // Epetra maps in original distribution

  Teuchos::RCP<Epetra_Map> masternodemap =
    Teuchos::rcp(new Epetra_Map(-1, mastervect.size(), &mastervect[0], 0, masterdis.Comm()));

  Teuchos::RCP<Epetra_Map> slavenodemap =
    Teuchos::rcp(new Epetra_Map(slavenodes));

  Teuchos::RCP<Epetra_Map> permslavenodemap =
    Teuchos::rcp(new Epetra_Map(-1, permslavenodes.size(), &permslavenodes[0], 0, slavedis.Comm()));

  FinishCoupling(masterdis, slavedis, masternodemap, slavenodemap, permslavenodemap, numdof);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ADAPTER::Coupling::MatchNodes(const DRT::Discretization& masterdis,
                                   const DRT::Discretization& slavedis,
                                   std::vector<int>& masternodes,
                                   std::vector<int>& permslavenodes,
                                   const std::vector<int>& slavenodes,
                                   const bool matchall,
                                   const double tolerance)
{
  // match master and slave nodes using Peter's octtree
  DRT::UTILS::NodeMatchingOctree tree(masterdis, masternodes, 150, tolerance);

  std::map<int,std::pair<int,double> > coupling;
  tree.FindMatch(slavedis, slavenodes, coupling);

  if (masternodes.size() != coupling.size() and matchall)
    dserror("Did not get 1:1 correspondence. \nmasternodes.size()=%d (%s), coupling.size()=%d (%s)",
            masternodes.size(),masterdis.Name().c_str(), coupling.size(),slavedis.Name().c_str());

  // extract permutation

  std::vector<int> patchedmasternodes;
  patchedmasternodes.reserve(coupling.size());
  permslavenodes.reserve(slavenodes.size());

  for (unsigned i=0; i<masternodes.size(); ++i)
  {
    int gid = masternodes[i];

    // We allow to hand in master nodes that do not take part in the
    // coupling. If this is undesired behaviour the user has to make
    // sure all nodes were used.
    if (coupling.find(gid) != coupling.end())
    {
      std::pair<int,double>& coupled = coupling[gid];
#if 0
      if (coupled.second > 1e-7)
        dserror("Coupled nodes (%d,%d) do not match. difference=%e", gid, coupled.first, coupled.second);
#endif
      patchedmasternodes.push_back(gid);
      permslavenodes.push_back(coupled.first);
    }
  }

  // return new list of master nodes via reference
  swap(masternodes,patchedmasternodes);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ADAPTER::Coupling::FinishCoupling(const DRT::Discretization& masterdis,
                                       const DRT::Discretization& slavedis,
                                       Teuchos::RCP<Epetra_Map> masternodemap,
                                       Teuchos::RCP<Epetra_Map> slavenodemap,
                                       Teuchos::RCP<Epetra_Map> permslavenodemap,
                                       const int numdof)
{
  // we expect to get maps of exactly the same shape
  if (not masternodemap->PointSameAs(*permslavenodemap))
    dserror("master and permuted slave node maps do not match");

  // export master nodes to slave node distribution

  // To do so we create vectors that contain the values of the master
  // maps, assigned to the slave maps. On the master side we actually
  // create just a view on the map! This vector must not be changed!
  Teuchos::RCP<Epetra_IntVector> masternodevec =
    Teuchos::rcp(new Epetra_IntVector(View, *permslavenodemap, masternodemap->MyGlobalElements()));

  Teuchos::RCP<Epetra_IntVector> permmasternodevec =
    Teuchos::rcp(new Epetra_IntVector(*slavenodemap));

  Epetra_Export masternodeexport(*permslavenodemap, *slavenodemap);
  const int err = permmasternodevec->Export(*masternodevec, masternodeexport, Insert);
  if (err)
    dserror("failed to export master nodes");

  Teuchos::RCP<const Epetra_Map> permmasternodemap =
    Teuchos::rcp(new Epetra_Map(-1, permmasternodevec->MyLength(), permmasternodevec->Values(), 0, masterdis.Comm()));

  if (not slavenodemap->PointSameAs(*permmasternodemap))
    dserror("slave and permuted master node maps do not match");

  masternodevec = Teuchos::null;
  permmasternodevec = Teuchos::null;

  BuildDofMaps(masterdis, masternodemap, permmasternodemap, masterdofmap_, permmasterdofmap_, masterexport_, numdof);
  BuildDofMaps(slavedis,  slavenodemap,  permslavenodemap,  slavedofmap_,  permslavedofmap_,  slaveexport_, numdof);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ADAPTER::Coupling::BuildDofMaps(const DRT::Discretization& dis,
                                     Teuchos::RCP<const Epetra_Map> nodemap,
                                     Teuchos::RCP<const Epetra_Map> permnodemap,
                                     Teuchos::RCP<const Epetra_Map>& dofmap,
                                     Teuchos::RCP<const Epetra_Map>& permdofmap,
                                     Teuchos::RCP<Epetra_Export>& exporter,
                                     const int numdof)
{
  // communicate dofs

  std::vector<int> dofmapvec;
  std::map<int, std::vector<int> > dofs;

  const int* nodes = nodemap->MyGlobalElements();
  const int numnode = nodemap->NumMyElements();

  for (int i=0; i<numnode; ++i)
  {
    const DRT::Node* actnode = dis.gNode(nodes[i]);

    // ----------------------------------------------------------------
    // get all periodic boundary conditions on this node
    // slave nodes do not contribute dofs, we skip them
    // ----------------------------------------------------------------
    std::vector<DRT::Condition*> thiscond;
    actnode->GetCondition("SurfacePeriodic",thiscond);

    if(thiscond.empty())
    {
      actnode->GetCondition("LinePeriodic",thiscond);
    }

    if(!thiscond.empty())
    {
      // loop them and check, whether this is a pbc pure master node
      // for all previous conditions
      unsigned ntimesmaster = 0;
      for (unsigned numcond=0;numcond<thiscond.size();++numcond)
      {
        const std::string* mymasterslavetoggle
          = thiscond[numcond]->Get<std::string>("Is slave periodic boundary condition");

        if(*mymasterslavetoggle=="Master")
        {
          ++ntimesmaster;
        } // end is slave?
      } // end loop this conditions

      if(ntimesmaster<thiscond.size())
      {
        // this node is not a master and does not own its own dofs
        continue;
      }
    }

    const std::vector<int> dof = dis.Dof(0,actnode);
    if (numdof > static_cast<int>(dof.size()))
      dserror("got just %d dofs at node %d (lid=%d) but expected %d",dof.size(),nodes[i],i,numdof);
    copy(&dof[0], &dof[0]+numdof, back_inserter(dofs[nodes[i]]));
    copy(&dof[0], &dof[0]+numdof, back_inserter(dofmapvec));
  }

  std::vector<int>::const_iterator pos = std::min_element(dofmapvec.begin(), dofmapvec.end());
  if (pos!=dofmapvec.end() and *pos < 0)
    dserror("illegal dof number %d", *pos);

  // dof map is the original, unpermuted distribution of dofs
  dofmap = Teuchos::rcp(new Epetra_Map(-1, dofmapvec.size(), &dofmapvec[0], 0, dis.Comm()));

  dofmapvec.clear();

  DRT::Exporter exportdofs(*nodemap,*permnodemap,dis.Comm());
  exportdofs.Export(dofs);

  const int* permnodes = permnodemap->MyGlobalElements();
  const int permnumnode = permnodemap->NumMyElements();

  for (int i=0; i<permnumnode; ++i)
  {
    const std::vector<int>& dof = dofs[permnodes[i]];
    copy(dof.begin(), dof.end(), back_inserter(dofmapvec));
  }

  dofs.clear();

  // permuted dof map according to a given permuted node map
  permdofmap = Teuchos::rcp(new Epetra_Map(-1, dofmapvec.size(), &dofmapvec[0], 0, dis.Comm()));

  // prepare communication plan to create a dofmap out of a permuted
  // dof map
  exporter = Teuchos::rcp(new Epetra_Export(*permdofmap, *dofmap));
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector> ADAPTER::Coupling::MasterToSlave(Teuchos::RCP<const Epetra_Vector> mv) const
{
  Teuchos::RCP<Epetra_Vector> sv =
    Teuchos::rcp(new Epetra_Vector(*slavedofmap_));

  MasterToSlave(mv,sv);

  return sv;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector> ADAPTER::Coupling::SlaveToMaster(Teuchos::RCP<const Epetra_Vector> sv) const
{
  Teuchos::RCP<Epetra_Vector> mv =
    Teuchos::rcp(new Epetra_Vector(*masterdofmap_));

  SlaveToMaster(sv,mv);

  return mv;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_MultiVector> ADAPTER::Coupling::MasterToSlave(Teuchos::RCP<const Epetra_MultiVector> mv) const
{
  Teuchos::RCP<Epetra_MultiVector> sv =
    Teuchos::rcp(new Epetra_MultiVector(*slavedofmap_,mv->NumVectors()));

  MasterToSlave(mv,sv);

  return sv;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_MultiVector> ADAPTER::Coupling::SlaveToMaster(Teuchos::RCP<const Epetra_MultiVector> sv) const
{
  Teuchos::RCP<Epetra_MultiVector> mv =
    Teuchos::rcp(new Epetra_MultiVector(*masterdofmap_,sv->NumVectors()));

  SlaveToMaster(sv,mv);

  return mv;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ADAPTER::Coupling::MasterToSlave(Teuchos::RCP<const Epetra_MultiVector> mv, Teuchos::RCP<Epetra_MultiVector> sv) const
{
#ifdef DEBUG
  if (not mv->Map().PointSameAs(*masterdofmap_))
    dserror("master dof map vector expected");
  if (not sv->Map().PointSameAs(*slavedofmap_))
    dserror("slave dof map vector expected");
  if (sv->NumVectors()!=mv->NumVectors())
    dserror("column number mismatch %d!=%d",sv->NumVectors(),mv->NumVectors());
#endif

  Epetra_MultiVector perm(*permslavedofmap_,mv->NumVectors());
  std::copy(mv->Values(), mv->Values()+(mv->MyLength()*mv->NumVectors()), perm.Values());

  const int err = sv->Export(perm,*slaveexport_,Insert);
  if (err)
    dserror("Export to slave distribution returned err=%d",err);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ADAPTER::Coupling::SlaveToMaster(Teuchos::RCP<const Epetra_MultiVector> sv, Teuchos::RCP<Epetra_MultiVector> mv) const
{
#ifdef DEBUG
  if (not mv->Map().PointSameAs(*masterdofmap_))
    dserror("master dof map vector expected");
  if (not sv->Map().PointSameAs(*slavedofmap_))
  {
    std::cout << "slavedofmap_" << std::endl;
    std::cout << *slavedofmap_ << std::endl;
    std::cout << "sv" << std::endl;
    std::cout << sv->Map() << std::endl;
    dserror("slave dof map vector expected");
  }
  if (sv->NumVectors()!=mv->NumVectors())
    dserror("column number mismatch %d!=%d",sv->NumVectors(),mv->NumVectors());
#endif

  Epetra_MultiVector perm(*permmasterdofmap_,sv->NumVectors());
  std::copy(sv->Values(), sv->Values()+(sv->MyLength()*sv->NumVectors()), perm.Values());

  const int err = mv->Export(perm,*masterexport_,Insert);
  if (err)
    dserror("Export to master distribution returned err=%d",err);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ADAPTER::Coupling::FillMasterToSlaveMap(std::map<int,int>& rowmap) const
{
  for (int i=0; i<masterdofmap_->NumMyElements(); ++i)
  {
    rowmap[masterdofmap_->GID(i)] = permslavedofmap_->GID(i);
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ADAPTER::Coupling::FillSlaveToMasterMap(std::map<int,int>& rowmap) const
{
  for (int i=0; i<slavedofmap_->NumMyElements(); ++i)
  {
    rowmap[slavedofmap_->GID(i)] = permmasterdofmap_->GID(i);
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Map> ADAPTER::Coupling::SlaveToMasterMap(Teuchos::RCP<Epetra_Map> slave)
{
  int nummyele = 0;
  std::vector<int> globalelements;
  const Teuchos::RCP<Epetra_Map> slavemap = LINALG::AllreduceEMap(*slave);
  for (int i = 0; i < slavemap->NumMyElements(); ++i)
  {
    int lid = permslavedofmap_->LID(slavemap->GID(i));
    if (lid != -1)
    {
      globalelements.push_back(masterdofmap_->GID(lid));
      nummyele++;
    }
  }

  return Teuchos::rcp<Epetra_Map>(new Epetra_Map(-1,nummyele,&globalelements[0],0,slave->Comm()));
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Map> ADAPTER::Coupling::MasterToSlaveMap(Teuchos::RCP<Epetra_Map> master)
{
  int nummyele = 0;
  std::vector<int> globalelements;
  const Teuchos::RCP<Epetra_Map> mastermap = LINALG::AllreduceEMap(*master);
  for (int i = 0; i < mastermap->NumMyElements(); ++i)
  {
    int lid = permmasterdofmap_->LID(mastermap->GID(i));
    if (lid != -1)
    {
      globalelements.push_back(slavedofmap_->GID(lid));
      nummyele++;
    }
  }

  return Teuchos::rcp<Epetra_Map>(new Epetra_Map(-1,nummyele,&globalelements[0],0,master->Comm()));
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<LINALG::SparseMatrix> ADAPTER::Coupling::MasterToPermMaster(const LINALG::SparseMatrix& sm) const
{
  Teuchos::RCP<Epetra_CrsMatrix> permsm = Teuchos::rcp(new Epetra_CrsMatrix(Copy,*permmasterdofmap_,sm.MaxNumEntries()));

  // OK. You cannot use the same exporter for different matrices. So we
  // recreate one all the time... This has to be optimized later on.

#if 0
  int err = permsm->Import(*sm.EpetraMatrix(),*masterexport_,Insert);
#else
  Teuchos::RCP<Epetra_Export> exporter = Teuchos::rcp(new Epetra_Export(*permmasterdofmap_, *masterdofmap_));
  int err = permsm->Import(*sm.EpetraMatrix(),*exporter,Insert);
#endif

  if (err)
    dserror("Import failed with err=%d",err);

  permsm->FillComplete(sm.DomainMap(),*permmasterdofmap_);

  // create a SparseMatrix that wraps the new CrsMatrix.
  return Teuchos::rcp(new LINALG::SparseMatrix(permsm,View,sm.ExplicitDirichlet(),sm.SaveGraph()));
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<LINALG::SparseMatrix> ADAPTER::Coupling::SlaveToPermSlave(const LINALG::SparseMatrix& sm) const
{
#ifdef DEBUG
  if (not sm.RowMap().PointSameAs(*slavedofmap_))
    dserror("slave dof map vector expected");
  if (not sm.Filled())
    dserror("matrix must be filled");
#endif

  Teuchos::RCP<Epetra_CrsMatrix> permsm = Teuchos::rcp(new Epetra_CrsMatrix(Copy,*permslavedofmap_,sm.MaxNumEntries()));

  // OK. You cannot use the same exporter for different matrices. So we
  // recreate one all the time... This has to be optimized later on.

#if 0
  int err = permsm->Import(*sm.EpetraMatrix(),*slaveexport_,Insert);
#else
  Teuchos::RCP<Epetra_Export> exporter = Teuchos::rcp(new Epetra_Export(*permslavedofmap_, *slavedofmap_));
  int err = permsm->Import(*sm.EpetraMatrix(),*exporter,Insert);
#endif

  if (err)
    dserror("Import failed with err=%d",err);

  permsm->FillComplete(sm.DomainMap(),*permslavedofmap_);

  // create a SparseMatrix that wraps the new CrsMatrix.
  return Teuchos::rcp(new LINALG::SparseMatrix(permsm,View,sm.ExplicitDirichlet(),sm.SaveGraph()));
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ADAPTER::Coupling::SetupCouplingMatrices(const Epetra_Map& shiftedmastermap,
                                              const Epetra_Map& masterdomainmap,
                                              const Epetra_Map& slavedomainmap)
{
  // we always use the masterdofmap for the domain
  matmm_ = Teuchos::rcp(new Epetra_CrsMatrix(Copy,shiftedmastermap,1,true));
  matsm_ = Teuchos::rcp(new Epetra_CrsMatrix(Copy,shiftedmastermap,1,true));

  matmm_trans_ = Teuchos::rcp(new Epetra_CrsMatrix(Copy,masterdomainmap,1,true));
  matsm_trans_ = Teuchos::rcp(new Epetra_CrsMatrix(Copy,*PermSlaveDofMap(),1,true));

  int length = shiftedmastermap.NumMyElements();
  double one = 1.;
  for (int i=0; i<length; ++i)
  {
    int sgid = PermSlaveDofMap()->GID(i);
    int mgid = MasterDofMap()->GID(i);
    int shiftedmgid = shiftedmastermap.GID(i);

    int err = matmm_->InsertGlobalValues(shiftedmgid, 1, &one, &mgid);
    if (err!=0)
      dserror("InsertGlobalValues for entry (%d,%d) failed with err=%d",shiftedmgid,mgid,err);

    err = matsm_->InsertGlobalValues(shiftedmgid, 1, &one, &sgid);
    if (err!=0)
      dserror("InsertGlobalValues for entry (%d,%d) failed with err=%d",shiftedmgid,sgid,err);

    err = matmm_trans_->InsertGlobalValues(mgid, 1, &one, &shiftedmgid);
    if (err!=0)
      dserror("InsertGlobalValues for entry (%d,%d) failed with err=%d",mgid,shiftedmgid,err);

    err = matsm_trans_->InsertGlobalValues(sgid, 1, &one, &shiftedmgid);
    if (err!=0)
      dserror("InsertGlobalValues for entry (%d,%d) failed with err=%d",sgid,shiftedmgid,err);
  }

  matmm_->FillComplete(masterdomainmap,shiftedmastermap);
  matsm_->FillComplete(slavedomainmap,shiftedmastermap);

  matmm_trans_->FillComplete(shiftedmastermap,masterdomainmap);
  matsm_trans_->FillComplete(shiftedmastermap,*PermSlaveDofMap());

  // communicate slave to master matrix

  Teuchos::RCP<Epetra_CrsMatrix> tmp = Teuchos::rcp(new Epetra_CrsMatrix(Copy,slavedomainmap,1));

  Teuchos::RCP<Epetra_Import> exporter = Teuchos::rcp(new Epetra_Import(slavedomainmap, *PermSlaveDofMap()));
  int err = tmp->Import(*matsm_trans_,*exporter,Insert);
  if (err)
    dserror("Import failed with err=%d",err);

  tmp->FillComplete(shiftedmastermap,slavedomainmap);
  matsm_trans_ = tmp;
}


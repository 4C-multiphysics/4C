
#ifdef CCADISCRET

#include "fsi_coupling.H"
#include "../drt_lib/drt_nodematchingoctree.H"
#include "../drt_lib/linalg_utils.H"


/*----------------------------------------------------------------------*
 |                                                       m.gee 06/01    |
 | general problem data                                                 |
 | global variable GENPROB genprob is defined in global_control.c       |
 *----------------------------------------------------------------------*/
extern struct _GENPROB     genprob;


using namespace std;


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
FSI::Coupling::Coupling()
{
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::Coupling::SetupConditionCoupling(const DRT::Discretization& masterdis,
                                           const LINALG::MapExtractor& master,
                                           const DRT::Discretization& slavedis,
                                           const LINALG::MapExtractor& slave,
                                           const std::string& condname)
{
  std::vector<int> masternodes;
  DRT::UTILS::FindConditionedNodes(masterdis,condname,masternodes);
  std::vector<int> slavenodes;
  DRT::UTILS::FindConditionedNodes(slavedis,condname,slavenodes);
  SetupCoupling(masterdis, slavedis, masternodes, slavenodes);

  // test for completeness
  if (static_cast<int>(masternodes.size())*genprob.ndim != masterdofmap_->NumMyElements())
    dserror("failed to setup master nodes properly");
  if (static_cast<int>(slavenodes.size())*genprob.ndim != slavedofmap_->NumMyElements())
    dserror("failed to setup slave nodes properly");

  // Now swap in the maps we already had.
  // So we did a little more work than required. But there are cases
  // where we have to do that work (fluid-ale coupling) and we want to
  // use just one setup implementation.
  //
  // The point is to make sure there is only one map for each
  // interface.

  if (not masterdofmap_->SameAs(*master.CondMap()))
    dserror("master dof map mismatch");

  if (not slavedofmap_->SameAs(*slave.CondMap()))
    dserror("slave dof map mismatch");

  masterdofmap_ = master.CondMap();
  masterexport_ = rcp(new Epetra_Export(*permmasterdofmap_, *masterdofmap_));

  slavedofmap_ = slave.CondMap();
  slaveexport_ = rcp(new Epetra_Export(*permslavedofmap_, *slavedofmap_));
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::Coupling::SetupCoupling(const DRT::Discretization& masterdis,
                                  const DRT::Discretization& slavedis,
                                  const std::vector<int>& masternodes,
                                  const std::vector<int>& slavenodes)
{
  std::vector<int> patchedmasternodes(masternodes);
  std::vector<int> permslavenodes;
  MatchNodes(masterdis, slavedis, patchedmasternodes, permslavenodes, slavenodes);

  // Epetra maps in original distribution

  Teuchos::RCP<Epetra_Map> masternodemap =
    rcp(new Epetra_Map(-1, patchedmasternodes.size(), &patchedmasternodes[0], 0, masterdis.Comm()));

  Teuchos::RCP<Epetra_Map> slavenodemap =
    rcp(new Epetra_Map(-1, slavenodes.size(), &slavenodes[0], 0, slavedis.Comm()));

  Teuchos::RCP<Epetra_Map> permslavenodemap =
    rcp(new Epetra_Map(-1, permslavenodes.size(), &permslavenodes[0], 0, slavedis.Comm()));

  FinishCoupling(masterdis, slavedis, masternodemap, slavenodemap, permslavenodemap);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::Coupling::SetupCoupling(const DRT::Discretization& masterdis,
                                  const DRT::Discretization& slavedis,
                                  const Epetra_Map& masternodes,
                                  const Epetra_Map& slavenodes)
{
  vector<int> mastervect(masternodes.MyGlobalElements(),
                         masternodes.MyGlobalElements() + masternodes.NumMyElements());
  vector<int> slavevect(slavenodes.MyGlobalElements(),
                        slavenodes.MyGlobalElements() + slavenodes.NumMyElements());
  vector<int> permslavenodes;

  MatchNodes(masterdis, slavedis, mastervect, permslavenodes, slavevect);

  // Epetra maps in original distribution

  Teuchos::RCP<Epetra_Map> masternodemap =
    rcp(new Epetra_Map(-1, mastervect.size(), &mastervect[0], 0, masterdis.Comm()));

  Teuchos::RCP<Epetra_Map> slavenodemap =
    rcp(new Epetra_Map(slavenodes));

  Teuchos::RCP<Epetra_Map> permslavenodemap =
    rcp(new Epetra_Map(-1, permslavenodes.size(), &permslavenodes[0], 0, slavedis.Comm()));

  FinishCoupling(masterdis, slavedis, masternodemap, slavenodemap, permslavenodemap);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::Coupling::MatchNodes(const DRT::Discretization& masterdis,
                               const DRT::Discretization& slavedis,
                               std::vector<int>& masternodes,
                               std::vector<int>& permslavenodes,
                               const std::vector<int>& slavenodes)
{
  // match master and slave nodes using Peter's octtree

  DRT::UTILS::NodeMatchingOctree tree(masterdis, masternodes);

  map<int,pair<int,double> > coupling;
  tree.FindMatch(slavedis, slavenodes, coupling);

  if (masternodes.size() != coupling.size())
    dserror("Did not get 1:1 correspondence. masternodes.size()=%d, coupling.size()=%d",
            masternodes.size(), coupling.size());

  // extract permutation

  vector<int> patchedmasternodes;
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
      pair<int,double>& coupled = coupling[gid];
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
void FSI::Coupling::FinishCoupling(const DRT::Discretization& masterdis,
                                   const DRT::Discretization& slavedis,
                                   Teuchos::RCP<Epetra_Map> masternodemap,
                                   Teuchos::RCP<Epetra_Map> slavenodemap,
                                   Teuchos::RCP<Epetra_Map> permslavenodemap)
{
  // we expect to get maps of exactly the same shape
  if (not masternodemap->PointSameAs(*permslavenodemap))
    dserror("master and permutated slave node maps do not match");

  // export master nodes to slave node distribution

  // To do so we create vectors that contain the values of the master
  // maps, assigned to the slave maps. On the master side we actually
  // create just a view on the map! This vector must not be changed!
  Teuchos::RCP<Epetra_IntVector> masternodevec =
    rcp(new Epetra_IntVector(View, *permslavenodemap, masternodemap->MyGlobalElements()));

  Teuchos::RCP<Epetra_IntVector> permmasternodevec =
    rcp(new Epetra_IntVector(*slavenodemap));

  Epetra_Export masternodeexport(*permslavenodemap, *slavenodemap);
  const int err = permmasternodevec->Export(*masternodevec, masternodeexport, Insert);
  if (err)
    dserror("failed to export master nodes");

  Teuchos::RCP<const Epetra_Map> permmasternodemap =
    rcp(new Epetra_Map(-1, permmasternodevec->MyLength(), permmasternodevec->Values(), 0, masterdis.Comm()));

  if (not slavenodemap->PointSameAs(*permmasternodemap))
    dserror("slave and permutated master node maps do not match");

  masternodevec = Teuchos::null;
  permmasternodevec = Teuchos::null;

  BuildDofMaps(masterdis, masternodemap, permmasternodemap, masterdofmap_, permmasterdofmap_, masterexport_);
  BuildDofMaps(slavedis,  slavenodemap,  permslavenodemap,  slavedofmap_,  permslavedofmap_,  slaveexport_);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::Coupling::BuildDofMaps(const DRT::Discretization& dis,
                                 RCP<const Epetra_Map> nodemap,
                                 RCP<const Epetra_Map> permnodemap,
                                 RCP<const Epetra_Map>& dofmap,
                                 RCP<const Epetra_Map>& permdofmap,
                                 RCP<Epetra_Export>& exporter)
{
  // communicate dofs

  vector<int> dofmapvec;
  map<int, vector<int> > dofs;

  const int* nodes = nodemap->MyGlobalElements();
  const int numnode = nodemap->NumMyElements();

  for (int i=0; i<numnode; ++i)
  {
    const DRT::Node* actnode = dis.gNode(nodes[i]);
    const vector<int> dof = dis.Dof(actnode);
    copy(&dof[0], &dof[0]+genprob.ndim, back_inserter(dofs[nodes[i]]));
    copy(&dof[0], &dof[0]+genprob.ndim, back_inserter(dofmapvec));
  }

  // dof map is the original, unpermuted distribution of dofs
  dofmap = rcp(new Epetra_Map(-1, dofmapvec.size(), &dofmapvec[0], 0, dis.Comm()));

  dofmapvec.clear();

  DRT::Exporter exportdofs(*nodemap,*permnodemap,dis.Comm());
  exportdofs.Export(dofs);

  const int* permnodes = permnodemap->MyGlobalElements();
  const int permnumnode = permnodemap->NumMyElements();

  for (int i=0; i<permnumnode; ++i)
  {
    const vector<int>& dof = dofs[permnodes[i]];
    copy(dof.begin(), dof.end(), back_inserter(dofmapvec));
  }

  dofs.clear();

  // permutated dof map according to a given permutated node map
  permdofmap = rcp(new Epetra_Map(-1, dofmapvec.size(), &dofmapvec[0], 0, dis.Comm()));

  // prepare communication plan to create a dofmap out of a permutated
  // dof map
  exporter = rcp(new Epetra_Export(*permdofmap, *dofmap));
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector> FSI::Coupling::MasterToSlave(Teuchos::RCP<const Epetra_Vector> mv) const
{
  Teuchos::RCP<Epetra_Vector> sv =
    Teuchos::rcp(new Epetra_Vector(*slavedofmap_));

  MasterToSlave(mv,sv);

  return sv;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector> FSI::Coupling::SlaveToMaster(Teuchos::RCP<const Epetra_Vector> sv) const
{
  Teuchos::RCP<Epetra_Vector> mv =
    Teuchos::rcp(new Epetra_Vector(*masterdofmap_));

  SlaveToMaster(sv,mv);

  return mv;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_MultiVector> FSI::Coupling::MasterToSlave(Teuchos::RCP<const Epetra_MultiVector> mv) const
{
  Teuchos::RCP<Epetra_MultiVector> sv =
    Teuchos::rcp(new Epetra_MultiVector(*slavedofmap_,mv->NumVectors()));

  MasterToSlave(mv,sv);

  return sv;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_MultiVector> FSI::Coupling::SlaveToMaster(Teuchos::RCP<const Epetra_MultiVector> sv) const
{
  Teuchos::RCP<Epetra_MultiVector> mv =
    Teuchos::rcp(new Epetra_MultiVector(*masterdofmap_,sv->NumVectors()));

  SlaveToMaster(sv,mv);

  return mv;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::Coupling::MasterToSlave(Teuchos::RCP<const Epetra_MultiVector> mv, Teuchos::RCP<Epetra_MultiVector> sv) const
{
#ifdef DEBUG
  if (not mv->Map().SameAs(*masterdofmap_))
    dserror("master dof map vector expected");
  if (not sv->Map().SameAs(*slavedofmap_))
    dserror("slave dof map vector expected");
  if (sv->NumVectors()!=mv->NumVectors())
    dserror("column number mismatch %d!=%d",sv->NumVectors(),mv->NumVectors());
#endif

  Epetra_MultiVector perm(*permslavedofmap_,mv->NumVectors());
  copy(mv->Values(), mv->Values()+(mv->MyLength()*mv->NumVectors()), perm.Values());

  const int err = sv->Export(perm,*slaveexport_,Insert);
  if (err)
    dserror("Export to slave distribution returned err=%d",err);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::Coupling::SlaveToMaster(Teuchos::RCP<const Epetra_MultiVector> sv, Teuchos::RCP<Epetra_MultiVector> mv) const
{
#ifdef DEBUG
  if (not mv->Map().SameAs(*masterdofmap_))
    dserror("master dof map vector expected");
  if (not sv->Map().SameAs(*slavedofmap_))
    dserror("slave dof map vector expected");
  if (sv->NumVectors()!=mv->NumVectors())
    dserror("column number mismatch %d!=%d",sv->NumVectors(),mv->NumVectors());
#endif

  Epetra_MultiVector perm(*permmasterdofmap_,sv->NumVectors());
  copy(sv->Values(), sv->Values()+(sv->MyLength()*sv->NumVectors()), perm.Values());

  const int err = mv->Export(perm,*masterexport_,Insert);
  if (err)
    dserror("Export to master distribution returned err=%d",err);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::Coupling::FillSlaveToMasterMap(std::map<int,int>& rowmap) const
{
  for (int i=0; i<slavedofmap_->NumMyElements(); ++i)
  {
    rowmap[slavedofmap_->GID(i)] = permmasterdofmap_->GID(i);
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<LINALG::SparseMatrix> FSI::Coupling::MasterToPermMaster(const LINALG::SparseMatrix& sm) const
{
  Teuchos::RCP<Epetra_CrsMatrix> permsm = Teuchos::rcp(new Epetra_CrsMatrix(Copy,*permmasterdofmap_,sm.MaxNumEntries()));

  // OK. You cannot use the same exporter for different matrices. So we
  // recreate one all the time... This has to be optimized later on.

#if 0
  int err = permsm->Import(*sm.EpetraMatrix(),*masterexport_,Insert);
#else
  Teuchos::RCP<Epetra_Export> exporter = rcp(new Epetra_Export(*permmasterdofmap_, *masterdofmap_));
  int err = permsm->Import(*sm.EpetraMatrix(),*exporter,Insert);
#endif

  if (err)
    dserror("Import failed with err=%d",err);

  permsm->FillComplete(sm.DomainMap(),*permmasterdofmap_);

  // create a SparseMatrix that wraps the new CrsMatrix.
  return Teuchos::rcp(new LINALG::SparseMatrix(permsm,sm.ExplicitDirichlet(),sm.SaveGraph()));
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<LINALG::SparseMatrix> FSI::Coupling::SlaveToPermSlave(const LINALG::SparseMatrix& sm) const
{
#ifdef DEBUG
  if (not sm.RowMap().SameAs(*slavedofmap_))
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
  Teuchos::RCP<Epetra_Export> exporter = rcp(new Epetra_Export(*permslavedofmap_, *slavedofmap_));
  int err = permsm->Import(*sm.EpetraMatrix(),*exporter,Insert);
#endif

  if (err)
    dserror("Import failed with err=%d",err);

  permsm->FillComplete(sm.DomainMap(),*permslavedofmap_);

  // create a SparseMatrix that wraps the new CrsMatrix.
  return Teuchos::rcp(new LINALG::SparseMatrix(permsm,sm.ExplicitDirichlet(),sm.SaveGraph()));
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::Coupling::SetupCouplingMatrices(const Epetra_Map& masterdomainmap, const Epetra_Map& slavedomainmap)
{
  // we always use the masterdofmap for the domain
  matmm_ = Teuchos::rcp(new Epetra_CrsMatrix(Copy,*MasterDofMap(),1,true));
  matsm_ = Teuchos::rcp(new Epetra_CrsMatrix(Copy,*MasterDofMap(),1,true));

  matmm_trans_ = Teuchos::rcp(new Epetra_CrsMatrix(Copy,masterdomainmap,1,true));
  matsm_trans_ = Teuchos::rcp(new Epetra_CrsMatrix(Copy,slavedomainmap,1,true));

  int length = MasterDofMap()->NumMyElements();
  double one = 1.;
  for (int i=0; i<length; ++i)
  {
    int sgid = SlaveDofMap()->GID(i);
    int mgid = MasterDofMap()->GID(i);

    int err = matmm_->InsertGlobalValues(mgid, 1, &one, &mgid);
    if (err!=0)
      dserror("InsertGlobalValues for entry (%d,%d) failed with err=%d",mgid,mgid,err);

    err = matsm_->InsertGlobalValues(mgid, 1, &one, &sgid);
    if (err!=0)
      dserror("InsertGlobalValues for entry (%d,%d) failed with err=%d",mgid,sgid,err);

    err = matmm_trans_->InsertGlobalValues(mgid, 1, &one, &mgid);
    if (err!=0)
      dserror("InsertGlobalValues for entry (%d,%d) failed with err=%d",mgid,mgid,err);

    err = matsm_trans_->InsertGlobalValues(sgid, 1, &one, &mgid);
    if (err!=0)
      dserror("InsertGlobalValues for entry (%d,%d) failed with err=%d",sgid,mgid,err);
  }

  matmm_->FillComplete(masterdomainmap,*MasterDofMap());
  matsm_->FillComplete(slavedomainmap,*MasterDofMap());

  matmm_trans_->FillComplete(*MasterDofMap(),masterdomainmap);
  matsm_trans_->FillComplete(*MasterDofMap(),slavedomainmap);

}

#endif

/*---------------------------------------------------------------------*/
/*! \file

\brief A set of degrees of freedom

\level 0


*/
/*---------------------------------------------------------------------*/

#include "4C_fem_dofset.hpp"

#include "4C_comm_exporter.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_discretization_hdg.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"

#include <Epetra_FECrsGraph.h>

#include <algorithm>
#include <iostream>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  ctor (public)                                             ukue 04/07|
 *----------------------------------------------------------------------*/
Core::DOFSets::DofSet::DofSet()
    : Core::DOFSets::DofSetBase(), filled_(false), dspos_(0), pccdofhandling_(false)
{
  return;
}



/*----------------------------------------------------------------------*
 |  << operator                                               ukue 04/07|
 *----------------------------------------------------------------------*/
std::ostream& operator<<(std::ostream& os, const Core::DOFSets::DofSet& dofset)
{
  dofset.print(os);
  return os;
}


/*----------------------------------------------------------------------*
 |  print this  (public)                                      ukue 04/07|
 *----------------------------------------------------------------------*/
void Core::DOFSets::DofSet::print(std::ostream& os) const
{
  for (int proc = 0; proc < numdfcolelements_->Comm().NumProc(); ++proc)
  {
    if (proc == numdfcolelements_->Comm().MyPID())
    {
      if (numdfcolelements_->MyLength()) os << "-------------------------- Proc " << proc << " :\n";
      for (int i = 0; i < numdfcolelements_->MyLength(); ++i)
      {
        int numdf = (*numdfcolelements_)[i];
        int idx = (*idxcolelements_)[i];
        os << i << ": ";
        for (int j = 0; j < numdf; ++j) os << (idx + j) << " ";
        os << "\n";
      }
      os << std::endl;
    }
    numdfcolelements_->Comm().Barrier();
  }
  for (int proc = 0; proc < numdfcolnodes_->Comm().NumProc(); ++proc)
  {
    if (proc == numdfcolnodes_->Comm().MyPID())
    {
      if (numdfcolnodes_->MyLength()) os << "-------------------------- Proc " << proc << " :\n";
      for (int i = 0; i < numdfcolnodes_->MyLength(); ++i)
      {
        int numdf = (*numdfcolnodes_)[i];
        int idx = (*idxcolnodes_)[i];
        os << i << ": ";
        for (int j = 0; j < numdf; ++j) os << (idx + j) << " ";
        os << "\n";
      }
      os << std::endl;
    }
    numdfcolnodes_->Comm().Barrier();
  }
  for (int proc = 0; proc < numdfcolfaces_->Comm().NumProc(); ++proc)
  {
    if (proc == numdfcolfaces_->Comm().MyPID())
    {
      if (numdfcolfaces_->MyLength()) os << "-------------------------- Proc " << proc << " :\n";
      for (int i = 0; i < numdfcolfaces_->MyLength(); ++i)
      {
        int numdf = (*numdfcolfaces_)[i];
        int idx = (*idxcolfaces_)[i];
        os << i << ": ";
        for (int j = 0; j < numdf; ++j) os << (idx + j) << " ";
        os << "\n";
      }
      os << std::endl;
    }
    numdfcolfaces_->Comm().Barrier();
  }
}


/*----------------------------------------------------------------------*
 |  reset everything  (public)                                ukue 04/07|
 *----------------------------------------------------------------------*/
void Core::DOFSets::DofSet::reset()
{
  dofrowmap_ = Teuchos::null;
  dofcolmap_ = Teuchos::null;
  numdfcolnodes_ = Teuchos::null;
  numdfcolelements_ = Teuchos::null;
  idxcolnodes_ = Teuchos::null;
  idxcolelements_ = Teuchos::null;
  shiftcolnodes_ = Teuchos::null;
  dofscolnodes_ = Teuchos::null;

  filled_ = false;

  // tell all proxies
  notify_reset();
}

/*----------------------------------------------------------------------*
 |  setup everything  (public)                                ukue 04/07|
 *----------------------------------------------------------------------*/
int Core::DOFSets::DofSet::assign_degrees_of_freedom(
    const Core::FE::Discretization& dis, const unsigned dspos, const int start)
{
  if (!dis.filled()) FOUR_C_THROW("discretization Filled()==false");
  if (!dis.node_row_map()->UniqueGIDs()) FOUR_C_THROW("Nodal row map is not unique");
  if (!dis.element_row_map()->UniqueGIDs()) FOUR_C_THROW("Element row map is not unique");

  // A definite offset is currently not supported.
  // TODO (kronbichler) find a better solution for this
  // if (start!=0)
  //  FOUR_C_THROW("right now user specified dof offsets are not supported");

  dspos_ = dspos;

  // Add DofSets in order of assignment to list. Once it is there it has its
  // place and will get its starting id from the previous DofSet.
  add_dof_setto_list();

  // We assume that all dof sets before this one have been set up. Otherwise
  // we'd have to reorder the list.
  //
  // There is no test anymore to make sure that all prior dof sets have been
  // assigned. It seems people like to manipulate dof sets. People do create
  // dof sets that do not contain any dofs (on its first assignment), people
  // even shift dof set numbers to create overlapping dof sets. This is
  // perfectly fine.
  //
  // However if you rely on non-overlapping dof sets, you have to
  // fill_complete() your discretizations in the order of their creation. This
  // is guaranteed for all discretizations read from the input file since the
  // input reader calls fill_complete(). If you create your own discretizations
  // try to understand what you do.

  // Get highest GID used so far and add one
  int count = get_first_gid_number_to_be_used(dis);

  // Check if we have a face discretization which supports degrees of freedom on faces
  Teuchos::RCP<const Core::FE::DiscretizationHDG> facedis =
      Teuchos::rcp_dynamic_cast<const Core::FE::DiscretizationHDG>(Teuchos::RCP(&dis, false));

  // set count to 0 in case of dofset 2 in HDG discretizations
  if (facedis != Teuchos::null && dspos_ == 2) count = 0;

  // Now this is tricky. We have to care for nodes, faces, and elements, both
  // row and column maps. In general both nodes, faces, and elements can have
  // dofs. In all cases these dofs might be shared with other nodes, faces,
  // or elements. (The very general case. For elements we'd probably
  // don't need that.)
  //
  // The point is that we have to make sure the dof numbering of a
  // mesh is independent of its parallel distribution. Otherwise we
  // could not redistribute a mesh. We would not be able to use old
  // distributed vectors afterwards.
  //
  // Each object (node or element) could have a different number of
  // dofs. The parallel distribution is arbitrary. So we fall back to
  // two redundant vectors here to gather the number of dofs per node
  // or element.

  // numdf for all nodes and elements
  numdfcolnodes_ = Teuchos::make_rcp<Core::LinAlg::Vector<int>>(*dis.node_col_map());
  numdfcolelements_ = Teuchos::make_rcp<Core::LinAlg::Vector<int>>(*dis.element_col_map());
  if (facedis != Teuchos::null && facedis->face_col_map() != nullptr)
    numdfcolfaces_ = Teuchos::make_rcp<Core::LinAlg::Vector<int>>(*facedis->face_col_map());

  // index of first dof for all nodes and elements
  idxcolnodes_ = Teuchos::make_rcp<Core::LinAlg::Vector<int>>(*dis.node_col_map());
  idxcolelements_ = Teuchos::make_rcp<Core::LinAlg::Vector<int>>(*dis.element_col_map());
  if (facedis != Teuchos::null && facedis->face_col_map() != nullptr)
    idxcolfaces_ = Teuchos::make_rcp<Core::LinAlg::Vector<int>>(*facedis->face_col_map());

  //////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////
  int maxnodenumdf = 0;
  int maxelementnumdf = 0;
  std::map<int, std::vector<int>> nodedofset;
  std::map<int, std::vector<int>> nodeduplicatedofset;
  std::map<int, std::vector<int>> elementdofset;
  std::map<int, std::vector<int>> facedofset;

  {
    // get DoF coupling conditions
    std::vector<Core::Conditions::Condition*> couplingconditions(0);
    dis.get_condition("PointCoupling", couplingconditions);
    if ((int)couplingconditions.size() > 0) pccdofhandling_ = true;

    // do the nodes first
    Core::LinAlg::Vector<int> numdfrownodes(*dis.node_row_map());
    Core::LinAlg::Vector<int> idxrownodes(*dis.node_row_map());

    int numrownodes = dis.num_my_row_nodes();
    for (int i = 0; i < numrownodes; ++i)
    {
      Core::Nodes::Node* actnode = dis.l_row_node(i);
      numdfrownodes[i] = num_dof_per_node(*actnode);
    }

    int minnodegid = get_minimal_node_gid_if_relevant(dis);
    maxnodenumdf = numdfrownodes.MaxValue();
    get_reserved_max_num_dofper_node(maxnodenumdf);  // XFEM::XFEMDofSet set to const number!

    for (int i = 0; i < numrownodes; ++i)
    {
      Core::Nodes::Node* actnode = dis.l_row_node(i);
      const int gid = actnode->id();

      // **********************************************************************
      // **********************************************************************
      // check for DoF coupling conditions                         popp 02/2016
      // **********************************************************************
      // **********************************************************************
      int relevantcondid = -1;
      if (dspos_ == 0)
      {
        for (int k = 0; k < (int)couplingconditions.size(); ++k)
        {
          if (couplingconditions[k]->contains_node(gid))
          {
            if (relevantcondid != -1) FOUR_C_THROW("ERROR: Two coupling conditions on one node");
            relevantcondid = k;
          }
        }
      }

      // check for node coupling condition and slave/master status
      bool specialtreatment = false;
      if (relevantcondid >= 0)
      {
        const std::vector<int>* nodeids = couplingconditions[relevantcondid]->get_nodes();
        if (!nodeids) FOUR_C_THROW("ERROR: Condition does not have Node Ids");

        // check if all nodes in this condition are on same processor
        // (otherwise throw a FOUR_C_THROW for now - not yet implemented)
        bool allononeproc = true;
        for (int k = 0; k < (int)(nodeids->size()); ++k)
        {
          int checkgid = (*nodeids)[k];
          if (!dis.node_row_map()->MyGID(checkgid)) allononeproc = false;
        }
        if (!allononeproc)
          FOUR_C_THROW(
              "ERRROR: Nodes in point coupling condition must all be on same processsor (for now)");

        // do nothing for first (master) node in coupling condition
        // do something for second, third, ... (slave) node
        if ((*nodeids)[0] != gid)
        {
          // critical case
          specialtreatment = true;

          // check total number of dofs and determine which dofs are to be coupled
          if (couplingconditions[relevantcondid]->parameters().get<int>("NUMDOF") !=
              numdfrownodes[i])
            FOUR_C_THROW(
                "ERROR: Number of DoFs in coupling condition (%i) does not match node (%i)",
                couplingconditions[relevantcondid]->parameters().get<int>("NUMDOF"),
                numdfrownodes[i]);
          const std::vector<int>& onoffcond =
              couplingconditions[relevantcondid]->parameters().get<std::vector<int>>("ONOFF");

          // get master node of this condition
          int mgid = (*nodeids)[0];
          std::vector<int>& mdofs = nodedofset[mgid];
          if ((int)(mdofs.size()) == 0)
            FOUR_C_THROW("ERROR: Master node has not yet been initialized with DoFs");

          // special treatment
          int numdf = numdfrownodes[i];
          int dof = count + (gid - minnodegid) * maxnodenumdf;
          idxrownodes[i] = dof;
          std::vector<int>& dofs = nodedofset[gid];
          std::vector<int>& duplicatedofs = nodeduplicatedofset[gid];
          dofs.reserve(numdf);
          duplicatedofs.reserve(numdf);
          for (int j = 0; j < numdf; ++j)
          {
            // push back master node DoF ID if coupled
            if (onoffcond[j] == 1)
            {
              dofs.push_back(mdofs[j]);
              duplicatedofs.push_back(1);
            }
            // push back new DoF ID if not coupled
            else
            {
              dofs.push_back(dof + j);
              duplicatedofs.push_back(0);
            }
          }
        }
      }

      // standard treatment for non-coupling nodes and master coupling nodes
      if (!specialtreatment)
      {
        int numdf = numdfrownodes[i];
        int dof = count + (gid - minnodegid) * maxnodenumdf;
        idxrownodes[i] = dof;
        std::vector<int>& dofs = nodedofset[gid];
        std::vector<int>& duplicatedofs = nodeduplicatedofset[gid];
        dofs.reserve(numdf);
        duplicatedofs.reserve(numdf);
        for (int j = 0; j < numdf; ++j)
        {
          dofs.push_back(dof + j);
          duplicatedofs.push_back(0);
        }
      }
      // **********************************************************************
      // **********************************************************************
      // **********************************************************************
      // **********************************************************************
    }

    Epetra_Import nodeimporter(numdfcolnodes_->Map(), numdfrownodes.Map());
    int err = numdfcolnodes_->Import(numdfrownodes, nodeimporter, Insert);
    if (err) FOUR_C_THROW("Import using importer returned err=%d", err);
    err = idxcolnodes_->Import(idxrownodes, nodeimporter, Insert);
    if (err) FOUR_C_THROW("Import using importer returned err=%d", err);

    count = maxnodenumdf > 0 ? idxrownodes.MaxValue() + maxnodenumdf : 0;

    //////////////////////////////////////////////////////////////////

    // Now do it again for the faces
    if (facedis != Teuchos::null && facedis->face_row_map() != nullptr)
    {
      Core::LinAlg::Vector<int> numdfrowfaces(*facedis->face_row_map());
      Core::LinAlg::Vector<int> idxrowfaces(*facedis->face_row_map());
      int numcolelements = dis.num_my_col_elements();

      const int mypid = dis.get_comm().MyPID();
      for (int i = 0; i < numcolelements; ++i)
      {
        Teuchos::RCP<Core::Elements::FaceElement>* faces = dis.l_col_element(i)->faces();
        // If no faces are found, continue...
        if (faces == nullptr) continue;
        for (int face = 0; face < dis.l_col_element(i)->num_face(); ++face)
          if (faces[face]->owner() == mypid)
          {
            const int mylid = facedis->face_row_map()->LID(faces[face]->id());
            numdfrowfaces[mylid] = num_dof_per_face(*(dis.l_col_element(i)), face);
          }
      }

      int minfacegid = facedis->face_row_map()->MinAllGID();
      int maxfacenumdf = numdfrowfaces.MaxValue();

      for (int i = 0; i < numcolelements; ++i)
      {
        Teuchos::RCP<Core::Elements::FaceElement>* faces = dis.l_col_element(i)->faces();
        if (faces == nullptr) continue;
        for (int face = 0; face < dis.l_col_element(i)->num_face(); ++face)
          if (faces[face]->owner() == mypid)
          {
            const int gid = faces[face]->id();
            const int mylid = facedis->face_row_map()->LID(gid);
            int numdf = numdfrowfaces[mylid];
            int dof = count + (gid - minfacegid) * maxfacenumdf;
            idxrowfaces[mylid] = dof;
            std::vector<int>& dofs = facedofset[gid];
            // do not visit the same face more than once
            if (dofs.empty())
            {
              dofs.reserve(numdf);
              for (int j = 0; j < numdf; ++j)
              {
                dofs.push_back(dof + j);
              }
            }
          }
      }

      Epetra_Import faceimporter(numdfcolfaces_->Map(), numdfrowfaces.Map());
      err = numdfcolfaces_->Import(numdfrowfaces, faceimporter, Insert);
      if (err) FOUR_C_THROW("Import using importer returned err=%d", err);
      err = idxcolfaces_->Import(idxrowfaces, faceimporter, Insert);
      if (err) FOUR_C_THROW("Import using importer returned err=%d", err);

      count = idxrowfaces.MaxValue() + maxfacenumdf;
    }

    //////////////////////////////////////////////////////////////////

    // Now do it again for the elements
    Core::LinAlg::Vector<int> numdfrowelements(*dis.element_row_map());
    Core::LinAlg::Vector<int> idxrowelements(*dis.element_row_map());

    int numrowelements = dis.num_my_row_elements();
    for (int i = 0; i < numrowelements; ++i)
    {
      Core::Elements::Element* actele = dis.l_row_element(i);
      // const int gid = actele->Id();
      int numdf = num_dof_per_element(*actele);
      numdfrowelements[i] = numdf;
    }

    int minelementgid = dis.element_row_map()->MinAllGID();
    maxelementnumdf = numdfrowelements.MaxValue();

    for (int i = 0; i < numrowelements; ++i)
    {
      Core::Elements::Element* actelement = dis.l_row_element(i);
      const int gid = actelement->id();
      int numdf = numdfrowelements[i];
      int dof = count + (gid - minelementgid) * maxelementnumdf;
      idxrowelements[i] = dof;
      std::vector<int>& dofs = elementdofset[gid];
      dofs.reserve(numdf);
      for (int j = 0; j < numdf; ++j)
      {
        dofs.push_back(dof + j);
      }
    }

    Epetra_Import elementimporter(numdfcolelements_->Map(), numdfrowelements.Map());
    err = numdfcolelements_->Import(numdfrowelements, elementimporter, Insert);
    if (err) FOUR_C_THROW("Import using importer returned err=%d", err);
    err = idxcolelements_->Import(idxrowelements, elementimporter, Insert);
    if (err) FOUR_C_THROW("Import using importer returned err=%d", err);
  }

  // Now finally we have everything in place to build the maps.
  int numrownodes = dis.num_my_row_nodes();
  int numrowelements = dis.num_my_row_elements();

  std::vector<int> localrowdofs;
  std::vector<int> localcoldofs;
  localrowdofs.reserve(numrownodes * maxnodenumdf + numrowelements * maxelementnumdf);
  localcoldofs.reserve(numrownodes * maxnodenumdf + numrowelements * maxelementnumdf);

  std::vector<int> allnodelocalcoldofs;
  allnodelocalcoldofs.reserve(numrownodes * maxnodenumdf);

  for (std::map<int, std::vector<int>>::iterator i = nodedofset.begin(); i != nodedofset.end(); ++i)
  {
    std::vector<int>& dofs = i->second;
    std::vector<int>& duplicatedofs = nodeduplicatedofset[i->first];
    std::vector<int> cleandofs;
    for (unsigned j = 0; j < dofs.size(); ++j)
    {
      if (duplicatedofs[j] == 0) cleandofs.push_back(dofs[j]);
    }
    std::copy(cleandofs.begin(), cleandofs.end(), std::back_inserter(localrowdofs));
    // printf("Proc %d nodal gid %d ndofs %d\n",proc,i->first,(int)dofs.size());
    // for (unsigned j=0; j<dofs.size(); ++j) printf(" %d ",dofs[j]);
    // printf("\n");
  }
  for (std::map<int, std::vector<int>>::iterator i = facedofset.begin(); i != facedofset.end(); ++i)
  {
    std::vector<int>& dofs = i->second;
    std::copy(dofs.begin(), dofs.end(), std::back_inserter(localrowdofs));
    // printf("Proc %d ele gid %d ndofs %d\n",dis.Comm().MyPID(),i->first,(int)dofs.size());
    // for (unsigned j=0; j<dofs.size(); ++j) printf(" %d ",dofs[j]);
    // printf("\n");
  }
  for (std::map<int, std::vector<int>>::iterator i = elementdofset.begin();
       i != elementdofset.end(); ++i)
  {
    std::vector<int>& dofs = i->second;
    std::copy(dofs.begin(), dofs.end(), std::back_inserter(localrowdofs));
    // printf("Proc %d ele gid %d ndofs %d\n",dis.Comm().MyPID(),i->first,(int)dofs.size());
    // for (unsigned j=0; j<dofs.size(); ++j) printf(" %d ",dofs[j]);
    // printf("\n");
  }

  Core::Communication::Exporter nodeexporter(
      *dis.node_row_map(), *dis.node_col_map(), dis.get_comm());
  nodeexporter.do_export(nodedofset);

  Core::Communication::Exporter elementexporter(
      *dis.element_row_map(), *dis.element_col_map(), dis.get_comm());
  elementexporter.do_export(elementdofset);

  if (facedis != Teuchos::null && facedis->face_row_map() != nullptr)
  {
    Core::Communication::Exporter faceexporter(
        *facedis->face_row_map(), *facedis->face_col_map(), dis.get_comm());
    faceexporter.do_export(facedofset);
  }

  for (std::map<int, std::vector<int>>::iterator i = nodedofset.begin(); i != nodedofset.end(); ++i)
  {
    std::vector<int>& dofs = i->second;
    std::vector<int> cleandofs;
    for (unsigned j = 0; j < dofs.size(); ++j)
    {
      if (std::find(localcoldofs.begin(), localcoldofs.end(), dofs[j]) == localcoldofs.end())
        cleandofs.push_back(dofs[j]);
    }
    std::copy(cleandofs.begin(), cleandofs.end(), std::back_inserter(localcoldofs));
    std::copy(dofs.begin(), dofs.end(), std::back_inserter(allnodelocalcoldofs));
  }
  for (std::map<int, std::vector<int>>::iterator i = facedofset.begin(); i != facedofset.end(); ++i)
  {
    std::vector<int>& dofs = i->second;
    std::copy(dofs.begin(), dofs.end(), std::back_inserter(localcoldofs));
  }
  for (std::map<int, std::vector<int>>::iterator i = elementdofset.begin();
       i != elementdofset.end(); ++i)
  {
    std::vector<int>& dofs = i->second;
    std::copy(dofs.begin(), dofs.end(), std::back_inserter(localcoldofs));
  }

  dofrowmap_ = Teuchos::make_rcp<Epetra_Map>(
      -1, localrowdofs.size(), localrowdofs.data(), 0, dis.get_comm());
  if (!dofrowmap_->UniqueGIDs()) FOUR_C_THROW("Dof row map is not unique");
  dofcolmap_ = Teuchos::make_rcp<Epetra_Map>(
      -1, localcoldofs.size(), localcoldofs.data(), 0, dis.get_comm());

  // **********************************************************************
  // **********************************************************************
  // build map of all (non-unique) column DoFs
  dofscolnodes_ = Teuchos::make_rcp<Epetra_Map>(
      -1, allnodelocalcoldofs.size(), allnodelocalcoldofs.data(), 0, dis.get_comm());

  // build shift vector
  shiftcolnodes_ = Teuchos::make_rcp<Core::LinAlg::Vector<int>>(*dis.node_col_map());
  int numcolnodes = dis.num_my_col_nodes();
  for (int i = 0; i < numcolnodes; ++i)
  {
    if (i == 0)
    {
      (*shiftcolnodes_)[i] = 0;
    }
    else
    {
      Core::Nodes::Node* lastnode = dis.l_col_node(i - 1);
      (*shiftcolnodes_)[i] = (*shiftcolnodes_)[i - 1] + num_dof_per_node(*lastnode);
    }
  }
  // **********************************************************************
  // **********************************************************************

  // degrees of freedom have now been assigned
  filled_ = true;

  // tell all proxies
  notify_assigned();

  // return maximum dof number of this dofset (+1)
  count = dofrowmap_->MaxAllGID() + 1;
  return count;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Core::DOFSets::DofSet::initialized() const
{
  if (dofcolmap_ == Teuchos::null or dofrowmap_ == Teuchos::null)
    return false;
  else
    return true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const Epetra_Map* Core::DOFSets::DofSet::dof_row_map() const
{
  if (dofrowmap_ == Teuchos::null)
    FOUR_C_THROW("Core::DOFSets::DofSet::dof_row_map(): dofrowmap_ not initialized, yet");
  return dofrowmap_.get();
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const Epetra_Map* Core::DOFSets::DofSet::dof_col_map() const
{
  if (dofcolmap_ == Teuchos::null)
    FOUR_C_THROW("Core::DOFSets::DofSet::DofColMap(): dofcolmap_ not initialized, yet");
  return dofcolmap_.get();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
int Core::DOFSets::DofSet::num_global_elements() const
{
  if (dofrowmap_ == Teuchos::null)
    FOUR_C_THROW("Core::DOFSets::DofSet::NumGlobalElements(): dofrowmap_ not initialized, yet");
  return dofrowmap_->NumGlobalElements();
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
int Core::DOFSets::DofSet::max_all_gid() const
{
  if (dofrowmap_ == Teuchos::null)
    FOUR_C_THROW("Core::DOFSets::DofSet::MaxAllGID(): dofrowmap_ not initialized, yet");
  return dofrowmap_->MaxAllGID();
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
int Core::DOFSets::DofSet::min_all_gid() const
{
  if (dofrowmap_ == Teuchos::null)
    FOUR_C_THROW("Core::DOFSets::DofSet::MinAllGID(): dofrowmap_ not initialized, yet");
  return dofrowmap_->MinAllGID();
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
int Core::DOFSets::DofSet::get_first_gid_number_to_be_used(
    const Core::FE::Discretization& dis) const
{
  return max_gi_din_list(dis.get_comm()) + 1;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
int Core::DOFSets::DofSet::get_minimal_node_gid_if_relevant(
    const Core::FE::Discretization& dis) const
{
  return dis.node_row_map()->MinAllGID();
}

FOUR_C_NAMESPACE_CLOSE

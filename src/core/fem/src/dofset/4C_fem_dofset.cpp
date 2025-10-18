// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_dofset.hpp"

#include "4C_comm_exporter.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_discretization_hdg.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"

#include <algorithm>
#include <format>

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
  for (int proc = 0; proc < Core::Communication::num_mpi_ranks(numdfcolelements_->get_comm());
      ++proc)
  {
    if (proc == Core::Communication::my_mpi_rank(numdfcolelements_->get_comm()))
    {
      if (numdfcolelements_->local_length())
        os << "-------------------------- Proc " << proc << " :\n";
      for (int i = 0; i < numdfcolelements_->local_length(); ++i)
      {
        int numdf = (numdfcolelements_->get_local_values())[i];
        int idx = (idxcolelements_->get_local_values())[i];
        os << i << ": ";
        for (int j = 0; j < numdf; ++j) os << (idx + j) << " ";
        os << "\n";
      }
      os << std::endl;
    }
    Core::Communication::barrier(numdfcolelements_->get_comm());
  }
  for (int proc = 0; proc < Core::Communication::num_mpi_ranks(numdfcolnodes_->get_comm()); ++proc)
  {
    if (proc == Core::Communication::my_mpi_rank(numdfcolnodes_->get_comm()))
    {
      if (numdfcolnodes_->local_length())
        os << "-------------------------- Proc " << proc << " :\n";
      for (int i = 0; i < numdfcolnodes_->local_length(); ++i)
      {
        int numdf = (numdfcolnodes_->get_local_values())[i];
        int idx = (idxcolnodes_->get_local_values())[i];

        os << i << ": ";
        for (int j = 0; j < numdf; ++j) os << (idx + j) << " ";
        os << "\n";
      }
      os << std::endl;
    }
    Core::Communication::barrier(numdfcolnodes_->get_comm());
  }
  for (int proc = 0; proc < Core::Communication::num_mpi_ranks(numdfcolfaces_->get_comm()); ++proc)
  {
    if (proc == Core::Communication::my_mpi_rank(numdfcolfaces_->get_comm()))
    {
      if (numdfcolfaces_->local_length())
        os << "-------------------------- Proc " << proc << " :\n";
      for (int i = 0; i < numdfcolfaces_->local_length(); ++i)
      {
        int numdf = (numdfcolfaces_->get_local_values())[i];
        int idx = (idxcolfaces_->get_local_values())[i];
        os << i << ": ";
        for (int j = 0; j < numdf; ++j) os << (idx + j) << " ";
        os << "\n";
      }
      os << std::endl;
    }
    Core::Communication::barrier(numdfcolfaces_->get_comm());
  }
}


/*----------------------------------------------------------------------*
 |  reset everything  (public)                                ukue 04/07|
 *----------------------------------------------------------------------*/
void Core::DOFSets::DofSet::reset()
{
  dofrowmap_ = nullptr;
  dofcolmap_ = nullptr;
  numdfcolnodes_ = nullptr;
  numdfcolelements_ = nullptr;
  idxcolnodes_ = nullptr;
  idxcolelements_ = nullptr;
  shiftcolnodes_ = nullptr;
  dofscolnodes_ = nullptr;

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
  if (!dis.node_row_map()->unique_gids()) FOUR_C_THROW("Nodal row map is not unique");
  if (!dis.element_row_map()->unique_gids()) FOUR_C_THROW("Element row map is not unique");

  // A definite offset is currently not supported.
  // TODO (kronbichler) find a better solution for this
  // if (start!=0)
  //  FOUR_C_THROW("right now user specified dof offsets are not supported");

  dspos_ = dspos;

  // Add DofSets in order of assignment to list. Once it is there it has its
  // place and will get its starting id from the previous DofSet.
  add_dof_set_to_list();

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
  std::shared_ptr<const Core::FE::DiscretizationHDG> facedis =
      std::dynamic_pointer_cast<const Core::FE::DiscretizationHDG>(
          Core::Utils::shared_ptr_from_ref(dis));

  // set count to 0 in case of dofset 2 in HDG discretizations
  if (facedis != nullptr && dspos_ == 2) count = 0;

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
  numdfcolnodes_ = std::make_shared<Core::LinAlg::Vector<int>>(*dis.node_col_map());
  numdfcolelements_ = std::make_shared<Core::LinAlg::Vector<int>>(*dis.element_col_map());
  if (facedis != nullptr && facedis->face_col_map() != nullptr)
    numdfcolfaces_ = std::make_shared<Core::LinAlg::Vector<int>>(*facedis->face_col_map());

  // index of first dof for all nodes and elements
  idxcolnodes_ = std::make_shared<Core::LinAlg::Vector<int>>(*dis.node_col_map());
  idxcolelements_ = std::make_shared<Core::LinAlg::Vector<int>>(*dis.element_col_map());
  if (facedis != nullptr && facedis->face_col_map() != nullptr)
    idxcolfaces_ = std::make_shared<Core::LinAlg::Vector<int>>(*facedis->face_col_map());

  //////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////
  int maxnodenumdf = 0;
  int maxelementnumdf = 0;
  int numrownodes = dis.num_my_row_nodes();
  int numrowelements = dis.num_my_row_elements();
  std::map<int, std::vector<int>> nodedofset;
  std::map<int, std::vector<int>> elementdofset;
  std::map<int, std::vector<int>> facedofset;
  std::vector<int> localrowdofs;

  {
    // get DoF coupling conditions
    std::vector<const Core::Conditions::Condition*> couplingconditions;
    dis.get_condition("PointCoupling", couplingconditions);
    if (!couplingconditions.empty()) pccdofhandling_ = true;

    // do the nodes first
    Core::LinAlg::Vector<int> num_dof_rownodes(*dis.node_row_map());
    Core::LinAlg::Vector<int> idxrownodes(*dis.node_row_map());

    for (int i = 0; i < numrownodes; ++i)
    {
      Core::Nodes::Node* actnode = dis.l_row_node(i);
      num_dof_rownodes.get_local_values()[i] = num_dof_per_node(*actnode);
    }

    int minnodegid = get_minimal_node_gid_if_relevant(dis);
    maxnodenumdf = num_dof_rownodes.max_value();
    get_reserved_max_num_dofper_node(maxnodenumdf);  // XFEM::XFEMDofSet set to const number!
    localrowdofs.reserve(numrownodes * maxnodenumdf);

    std::vector<std::vector<int>> onoffcond;  // vector of onoff status for each condition
    std::vector<int> numdofcond;              // vector of NUMDOF for each condition
    std::vector<std::vector<int>> nodeids;    // vector of node IDs for each condition
    std::vector<int> masterIds;               // vector of master node IDs for each condition

    for (const auto* condition : couplingconditions)
    {
      onoffcond.push_back(condition->parameters().get<std::vector<int>>("ONOFF"));
      numdofcond.push_back(condition->parameters().get<int>("NUMDOF"));
      const auto conditioned_node_ids = condition->get_nodes();
      nodeids.push_back(*conditioned_node_ids);
      // The first node ID of each condition serves as the master node:
      masterIds.push_back(*conditioned_node_ids->begin());
    }

    int allononeproc = 1;
    for (int i = 0; i < numrownodes; ++i)
    {
      const int gid = dis.l_row_node(i)->id();

      // **********************************************************************
      // **********************************************************************
      // check for DoF coupling conditions                         popp 02/2016
      // **********************************************************************
      // **********************************************************************
      std::vector<int> applied_condition(maxnodenumdf, -1);
      if (dspos_ == 0)
      {
        for (size_t condition_id = 0; condition_id < couplingconditions.size(); ++condition_id)
        {
          // check if the node gid is contained in the coupling condition
          if (couplingconditions[condition_id]->contains_node(gid))
          {
            for (size_t k_on = 0; k_on < onoffcond[condition_id].size(); ++k_on)
            {
              if (onoffcond[condition_id][k_on] == 0) continue;
              FOUR_C_ASSERT(applied_condition[k_on] == -1,
                  "ERROR: Two coupling conditions on the same degree of freedom");
              applied_condition[k_on] = condition_id;
            }
          }
        }
      }

      // check if all nodes in this condition are on the same processor
      for (int k_on = 0; k_on < maxnodenumdf; ++k_on)
      {
        if (applied_condition[k_on] == -1) continue;
        const int condition_id = applied_condition[k_on];
        const std::vector<int>* ndvec = couplingconditions[condition_id]->get_nodes();
        for (const auto nd : *ndvec)
        {
          if (!dis.node_row_map()->my_gid(nd))
          {
            allononeproc = 0;
            std::cout << "Node " << nd << " (LID " << dis.node_row_map()->lid(nd)
                      << ") in condition " << condition_id << "(" << k_on
                      << ") is not in the current row map!\n";
          }
        }
      }
      // leave loop here in case of corrupt node distribution and inform remaining procs
      if (allononeproc == 0) break;

      // check for node coupling condition and slave/master status

      // do nothing if the node is the master in all of the applied conditions
      // do something for second, third, ... (slave) node
      // also if the node is a master in one condition, but a slave in another
      bool is_slave = false;
      for (auto condition_id : applied_condition)
      {
        if (condition_id == -1) continue;
        // check total number of dofs and determine which dofs are to be coupled
        if (numdofcond[condition_id] != num_dof_rownodes.get_local_values()[i])
          FOUR_C_THROW("ERROR: Number of DoFs in coupling condition {} does not match node {}",
              numdofcond[condition_id], num_dof_rownodes.get_local_values()[i]);
        if (masterIds[condition_id] != gid) is_slave = true;
      }

      if (is_slave)
      {  // in case it is a slave node (in some dof): the dof is cancelled, replaced by the master
         // dof
        int numdf = num_dof_rownodes.get_local_values()[i];
        int dof = count + (gid - minnodegid) * maxnodenumdf;
        idxrownodes.get_local_values()[i] = dof;
        std::vector<int>& dofs = nodedofset[gid];
        dofs.reserve(numdf);
        for (int j = 0; j < numdf; ++j)
        {
          // push back master node DoF ID if the id is coupled
          // it might be that the node is the master in one direction and a slave in another
          if (applied_condition[j] >= 0 && masterIds[applied_condition[j]] != gid)
          {
            std::vector<int>& mdofs = nodedofset[masterIds[applied_condition[j]]];
            dofs.push_back(mdofs[j]);
          }
          // push back new DoF ID if not coupled
          else
          {
            dofs.push_back(dof + j);
            localrowdofs.push_back(dof + j);
          }
        }
      }
      else
      // standard treatment for non-coupling nodes and master coupling nodes
      {
        // now treat only the nodes, which are master and thus did not get the special treatment
        int numdf = num_dof_rownodes.get_local_values()[i];
        int dof = count + (gid - minnodegid) * maxnodenumdf;
        idxrownodes.get_local_values()[i] = dof;
        std::vector<int>& dofs = nodedofset[gid];
        dofs.reserve(numdf);
        for (int j = 0; j < numdf; ++j)
        {
          dofs.push_back(dof + j);
          localrowdofs.push_back(dof + j);
        }
      }
      // **********************************************************************
      // **********************************************************************
      // **********************************************************************
      // **********************************************************************
    }

    // communication to ensure that all procs know if an error occurred on certain procs
    int allononeproc_global;
    allononeproc_global = Core::Communication::min_all(allononeproc, dis.get_comm());
    if (allononeproc_global == 0)
    {
      throw NodalDistributionException(
          "ERROR: Nodes in point coupling condition must all be on same processor.");
    }

    // fill ghost node entries
    Core::LinAlg::Import nodeimporter(numdfcolnodes_->get_map(), num_dof_rownodes.get_map());
    numdfcolnodes_->import(num_dof_rownodes, nodeimporter, Insert);
    idxcolnodes_->import(idxrownodes, nodeimporter, Insert);

    count = maxnodenumdf > 0 ? idxrownodes.max_value() + maxnodenumdf : 0;

    //////////////////////////////////////////////////////////////////

    // Now do it again for the faces
    if (facedis != nullptr && facedis->face_row_map() != nullptr)
    {
      Core::LinAlg::Vector<int> numdfrowfaces(*facedis->face_row_map());
      Core::LinAlg::Vector<int> idxrowfaces(*facedis->face_row_map());
      int numcolelements = dis.num_my_col_elements();

      const int mypid = Core::Communication::my_mpi_rank(dis.get_comm());
      for (int i = 0; i < numcolelements; ++i)
      {
        std::shared_ptr<Core::Elements::FaceElement>* faces = dis.l_col_element(i)->faces();
        // If no faces are found, continue...
        if (faces == nullptr) continue;
        for (int face = 0; face < dis.l_col_element(i)->num_face(); ++face)
          if (faces[face]->owner() == mypid)
          {
            const int mylid = facedis->face_row_map()->lid(faces[face]->id());
            numdfrowfaces.get_local_values()[mylid] =
                num_dof_per_face(*(dis.l_col_element(i)), face);
          }
      }

      int minfacegid = facedis->face_row_map()->min_all_gid();
      int maxfacenumdf = numdfrowfaces.max_value();

      for (int i = 0; i < numcolelements; ++i)
      {
        std::shared_ptr<Core::Elements::FaceElement>* faces = dis.l_col_element(i)->faces();
        if (faces == nullptr) continue;
        for (int face = 0; face < dis.l_col_element(i)->num_face(); ++face)
          if (faces[face]->owner() == mypid)
          {
            const int gid = faces[face]->id();
            const int mylid = facedis->face_row_map()->lid(gid);
            int numdf = numdfrowfaces.get_local_values()[mylid];
            int dof = count + (gid - minfacegid) * maxfacenumdf;
            idxrowfaces.get_local_values()[mylid] = dof;
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

      Core::LinAlg::Import faceimporter(numdfcolfaces_->get_map(), numdfrowfaces.get_map());
      numdfcolfaces_->import(numdfrowfaces, faceimporter, Insert);
      idxcolfaces_->import(idxrowfaces, faceimporter, Insert);

      count = idxrowfaces.max_value() + maxfacenumdf;
    }

    //////////////////////////////////////////////////////////////////

    // Now do it again for the elements
    Core::LinAlg::Vector<int> numdfrowelements(*dis.element_row_map());
    Core::LinAlg::Vector<int> idxrowelements(*dis.element_row_map());

    for (int i = 0; i < numrowelements; ++i)
    {
      Core::Elements::Element* actele = dis.l_row_element(i);
      // const int gid = actele->Id();
      int numdf = num_dof_per_element(*actele);
      numdfrowelements.get_local_values()[i] = numdf;
    }

    int minelementgid = dis.element_row_map()->min_all_gid();
    maxelementnumdf = numdfrowelements.max_value();
    localrowdofs.reserve(numrownodes * maxnodenumdf + numrowelements * maxelementnumdf);

    for (int i = 0; i < numrowelements; ++i)
    {
      Core::Elements::Element* actelement = dis.l_row_element(i);
      const int gid = actelement->id();
      int numdf = numdfrowelements.get_local_values()[i];
      int dof = count + (gid - minelementgid) * maxelementnumdf;
      idxrowelements.get_local_values()[i] = dof;
      std::vector<int>& dofs = elementdofset[gid];
      dofs.reserve(numdf);
      for (int j = 0; j < numdf; ++j)
      {
        dofs.push_back(dof + j);
      }
    }

    Core::LinAlg::Import elementimporter(numdfcolelements_->get_map(), numdfrowelements.get_map());
    numdfcolelements_->import(numdfrowelements, elementimporter, Insert);
    idxcolelements_->import(idxrowelements, elementimporter, Insert);
  }

  // Now finally we have everything in place to build the maps.
  for (std::map<int, std::vector<int>>::iterator i = facedofset.begin(); i != facedofset.end(); ++i)
  {
    std::vector<int>& dofs = i->second;
    std::copy(dofs.begin(), dofs.end(), std::back_inserter(localrowdofs));
  }
  for (std::map<int, std::vector<int>>::iterator i = elementdofset.begin();
      i != elementdofset.end(); ++i)
  {
    std::vector<int>& dofs = i->second;
    std::copy(dofs.begin(), dofs.end(), std::back_inserter(localrowdofs));
  }

  Core::Communication::Exporter nodeexporter(
      *dis.node_row_map(), *dis.node_col_map(), dis.get_comm());
  nodeexporter.do_export(nodedofset);

  if (facedis != nullptr && facedis->face_row_map() != nullptr)
  {
    Core::Communication::Exporter faceexporter(
        *facedis->face_row_map(), *facedis->face_col_map(), dis.get_comm());
    faceexporter.do_export(facedofset);
  }

  Core::Communication::Exporter elementexporter(
      *dis.element_row_map(), *dis.element_col_map(), dis.get_comm());
  elementexporter.do_export(elementdofset);

  std::vector<int> localcoldofs;
  localcoldofs.reserve(numrownodes * maxnodenumdf + numrowelements * maxelementnumdf);
  std::vector<int> allnodelocalcoldofs;
  allnodelocalcoldofs.reserve(numrownodes * maxnodenumdf);

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

  dofrowmap_ = std::make_shared<Core::LinAlg::Map>(
      -1, localrowdofs.size(), localrowdofs.data(), 0, dis.get_comm());
  if (!dofrowmap_->unique_gids()) FOUR_C_THROW("Dof row map is not unique");
  dofcolmap_ = std::make_shared<Core::LinAlg::Map>(
      -1, localcoldofs.size(), localcoldofs.data(), 0, dis.get_comm());

  // **********************************************************************
  // **********************************************************************
  // build map of all (non-unique) column DoFs
  dofscolnodes_ = std::make_shared<Core::LinAlg::Map>(
      -1, allnodelocalcoldofs.size(), allnodelocalcoldofs.data(), 0, dis.get_comm());

  // build shift vector
  shiftcolnodes_ = std::make_shared<Core::LinAlg::Vector<int>>(*dis.node_col_map());
  int numcolnodes = dis.num_my_col_nodes();
  for (int i = 0; i < numcolnodes; ++i)
  {
    if (i == 0)
    {
      (*shiftcolnodes_).get_local_values()[i] = 0;
    }
    else
    {
      Core::Nodes::Node* lastnode = dis.l_col_node(i - 1);
      (*shiftcolnodes_).get_local_values()[i] =
          (shiftcolnodes_->get_local_values())[i - 1] + num_dof_per_node(*lastnode);
    }
  }
  // **********************************************************************
  // **********************************************************************

  // degrees of freedom have now been assigned
  filled_ = true;

  // tell all proxies
  notify_assigned();

  // return maximum dof number of this dofset (+1)
  count = dofrowmap_->max_all_gid() + 1;
  return count;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Core::DOFSets::DofSet::initialized() const
{
  if (dofcolmap_ == nullptr or dofrowmap_ == nullptr)
    return false;
  else
    return true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const Core::LinAlg::Map* Core::DOFSets::DofSet::dof_row_map() const
{
  if (dofrowmap_ == nullptr)
    FOUR_C_THROW("Core::DOFSets::DofSet::dof_row_map(): dofrowmap_ not initialized, yet");
  return dofrowmap_.get();
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const Core::LinAlg::Map* Core::DOFSets::DofSet::dof_col_map() const
{
  if (dofcolmap_ == nullptr)
    FOUR_C_THROW("Core::DOFSets::DofSet::DofColMap(): dofcolmap_ not initialized, yet");
  return dofcolmap_.get();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
int Core::DOFSets::DofSet::num_global_elements() const
{
  if (dofrowmap_ == nullptr)
    FOUR_C_THROW("Core::DOFSets::DofSet::NumGlobalElements(): dofrowmap_ not initialized, yet");
  return dofrowmap_->num_global_elements();
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
int Core::DOFSets::DofSet::max_all_gid() const
{
  if (dofrowmap_ == nullptr)
    FOUR_C_THROW("Core::DOFSets::DofSet::MaxAllGID(): dofrowmap_ not initialized, yet");
  return dofrowmap_->max_all_gid();
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
int Core::DOFSets::DofSet::min_all_gid() const
{
  if (dofrowmap_ == nullptr)
    FOUR_C_THROW("Core::DOFSets::DofSet::MinAllGID(): dofrowmap_ not initialized, yet");
  return dofrowmap_->min_all_gid();
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
int Core::DOFSets::DofSet::get_first_gid_number_to_be_used(
    const Core::FE::Discretization& dis) const
{
  return max_gid_in_list(dis.get_comm()) + 1;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
int Core::DOFSets::DofSet::get_minimal_node_gid_if_relevant(
    const Core::FE::Discretization& dis) const
{
  return dis.node_row_map()->min_all_gid();
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::DOFSets::NodalDistributionException::NodalDistributionException(const std::string& message)
    : Core::Exception(message)
{
}

FOUR_C_NAMESPACE_CLOSE

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_dofset_merged_wrapper.hpp"

#include "4C_fem_condition_utils.hpp"
#include "4C_geometric_search_matchingoctree.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"


FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::DOFSets::DofSetMergedWrapper::DofSetMergedWrapper(
    std::shared_ptr<DofSetInterface> sourcedofset,
    std::shared_ptr<const Core::FE::Discretization> sourcedis,
    const std::string& couplingcond_master, const std::string& couplingcond_slave)
    : DofSetBase(),
      master_nodegids_col_layout_(nullptr),
      sourcedofset_(sourcedofset),
      sourcedis_(sourcedis),
      couplingcond_master_(couplingcond_master),
      couplingcond_slave_(couplingcond_slave),
      filled_(false)
{
  if (sourcedofset_ == nullptr) FOUR_C_THROW("Source dof set is null pointer.");
  if (sourcedis_ == nullptr) FOUR_C_THROW("Source discretization is null pointer.");

  sourcedofset_->register_proxy(this);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::DOFSets::DofSetMergedWrapper::~DofSetMergedWrapper()
{
  if (sourcedofset_ != nullptr) sourcedofset_->unregister(this);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const Core::LinAlg::Map* Core::DOFSets::DofSetMergedWrapper::dof_row_map() const
{
  // the merged dofset does not add new dofs. So we can just return the
  // original dof map here.
  return sourcedofset_->dof_row_map();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const Core::LinAlg::Map* Core::DOFSets::DofSetMergedWrapper::dof_col_map() const
{
  // the merged dofset does not add new dofs. So we can just return the
  // original dof map here.
  return sourcedofset_->dof_col_map();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
int Core::DOFSets::DofSetMergedWrapper::assign_degrees_of_freedom(
    const Core::FE::Discretization& dis, const unsigned dspos, const int start)
{
  if (sourcedofset_ == nullptr) FOUR_C_THROW("No source dof set assigned to merged dof set!");
  if (sourcedis_ == nullptr) FOUR_C_THROW("No source discretization assigned to mapping dof set!");

  // get nodes to be coupled
  auto masternodes_set = Core::Conditions::find_conditioned_node_ids(
      *sourcedis_, couplingcond_master_, Core::Conditions::LookFor::locally_owned);
  std::vector<int> masternodes(masternodes_set.begin(), masternodes_set.end());
  std::set<int> slavenodes_set = Core::Conditions::find_conditioned_node_ids(
      dis, couplingcond_slave_, Core::Conditions::LookFor::locally_owned);
  std::vector<int> slavenodes(slavenodes_set.begin(), slavenodes_set.end());


  // initialize search tree
  auto tree = Core::GeometricSearch::NodeMatchingOctree();
  tree.init(*sourcedis_, masternodes, 150);
  tree.setup();

  // match master and slave nodes using octtree
  // master id -> slave id, distance
  std::map<int, std::pair<int, double>> coupling;
  tree.find_match(dis, slavenodes, coupling);

  // all nodes should be coupled
  if (masternodes.size() != coupling.size())
    FOUR_C_THROW(
        "Did not get 1:1 correspondence. \nmasternodes.size()={}, coupling.size()={}."
        "DofSetMergedWrapper requires matching slave and master meshes!",
        masternodes.size(), coupling.size());

  // initialize final mapping
  Core::LinAlg::Vector<int> my_master_nodegids_row_layout(*dis.node_row_map());

  // loop over all coupled nodes
  for (unsigned i = 0; i < masternodes.size(); ++i)
  {
    // get master gid
    int gid = masternodes[i];

    // We allow to hand in master nodes that do not take part in the
    // coupling. If this is undesired behaviour the user has to make
    // sure all nodes were used.
    if (coupling.find(gid) != coupling.end())
    {
      std::pair<int, double>& coupled = coupling[gid];
      int slavegid = coupled.first;
      int slavelid = dis.node_row_map()->lid(slavegid);
      if (slavelid == -1) FOUR_C_THROW("slave gid {} was not found on this proc", slavegid);

      // save master gid at col lid of corresponding slave node
      (my_master_nodegids_row_layout.get_local_values())[slavelid] = gid;
    }
  }

  // initialize final mapping
  master_nodegids_col_layout_ = std::make_shared<Core::LinAlg::Vector<int>>(*dis.node_col_map());

  // export to column map
  Core::LinAlg::export_to(my_master_nodegids_row_layout, *master_nodegids_col_layout_);


  ////////////////////////////////////////////////////
  // now we match source slave and aux dis
  ////////////////////////////////////////////////////

  // get nodes to be coupled
  masternodes_set = Core::Conditions::find_conditioned_node_ids(
      dis, couplingcond_slave_, Core::Conditions::LookFor::locally_owned);
  masternodes = std::vector<int>(masternodes_set.begin(), masternodes_set.end());
  slavenodes_set = Core::Conditions::find_conditioned_node_ids(
      *sourcedis_, couplingcond_slave_, Core::Conditions::LookFor::locally_owned);
  slavenodes = std::vector<int>(slavenodes_set.begin(), slavenodes_set.end());


  // initialize search tree
  tree.init(*sourcedis_, masternodes, 150);
  tree.setup();

  // match master and slave nodes using octtree
  // master id -> slave id, distance
  coupling.clear();
  tree.find_match(dis, slavenodes, coupling);

  // all nodes should be coupled
  if (masternodes.size() != coupling.size())
    FOUR_C_THROW(
        "Did not get 1:1 correspondence. \nmasternodes.size()={}, coupling.size()={}."
        "DofSetMergedWrapper requires matching slave and master meshes!",
        masternodes.size(), coupling.size());

  // initialize final mapping
  Core::LinAlg::Vector<int> my_slave_nodegids_row_layout(*dis.node_row_map());

  // loop over all coupled nodes
  for (unsigned i = 0; i < masternodes.size(); ++i)
  {
    // get master gid
    int gid = masternodes[i];

    // We allow to hand in master nodes that do not take part in the
    // coupling. If this is undesired behaviour the user has to make
    // sure all nodes were used.
    if (coupling.find(gid) != coupling.end())
    {
      std::pair<int, double>& coupled = coupling[gid];
      int slavegid = coupled.first;
      int slavelid = dis.node_row_map()->lid(slavegid);
      if (slavelid == -1) FOUR_C_THROW("slave gid {} was not found on this proc", slavegid);

      // save master gid at col lid of corresponding slave node
      (my_slave_nodegids_row_layout.get_local_values())[slavelid] = gid;
    }
  }

  // initialize final mapping
  slave_nodegids_col_layout_ = std::make_shared<Core::LinAlg::Vector<int>>(*dis.node_col_map());

  // export to column map
  Core::LinAlg::export_to(my_slave_nodegids_row_layout, *slave_nodegids_col_layout_);

  // set filled flag true
  filled_ = true;

  // tell the proxies
  notify_assigned();

  return start;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::DOFSets::DofSetMergedWrapper::reset()
{
  master_nodegids_col_layout_ = nullptr;
  slave_nodegids_col_layout_ = nullptr;

  // set filled flag
  filled_ = false;

  notify_reset();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::DOFSets::DofSetMergedWrapper::disconnect(DofSetInterface* dofset)
{
  if (dofset == sourcedofset_.get())
  {
    sourcedofset_ = nullptr;
    sourcedis_ = nullptr;
  }
  else
    FOUR_C_THROW("cannot disconnect from non-connected DofSet");

  // clear my Teuchos::rcps.
  reset();
}

FOUR_C_NAMESPACE_CLOSE

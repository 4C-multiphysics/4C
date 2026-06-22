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
    const std::string& coupling_cond_target, const std::string& coupling_cond_source)
    : DofSetBase(),
      target_nodegids_col_layout_(nullptr),
      source_dofset_(sourcedofset),
      source_dis_(sourcedis),
      coupling_cond_target_(coupling_cond_target),
      coupling_cond_source_(coupling_cond_source),
      filled_(false)
{
  if (source_dofset_ == nullptr) FOUR_C_THROW("Source dof set is null pointer.");
  if (source_dis_ == nullptr) FOUR_C_THROW("Source discretization is null pointer.");

  source_dofset_->register_proxy(this);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::DOFSets::DofSetMergedWrapper::~DofSetMergedWrapper()
{
  if (source_dofset_ != nullptr) source_dofset_->unregister(this);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const Core::LinAlg::Map* Core::DOFSets::DofSetMergedWrapper::dof_row_map() const
{
  // the merged dofset does not add new dofs. So we can just return the
  // original dof map here.
  return source_dofset_->dof_row_map();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const Core::LinAlg::Map* Core::DOFSets::DofSetMergedWrapper::dof_col_map() const
{
  // the merged dofset does not add new dofs. So we can just return the
  // original dof map here.
  return source_dofset_->dof_col_map();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
int Core::DOFSets::DofSetMergedWrapper::assign_degrees_of_freedom(
    const Core::FE::Discretization& dis, const unsigned dspos, const int start)
{
  if (source_dofset_ == nullptr) FOUR_C_THROW("No source dof set assigned to merged dof set!");
  if (source_dis_ == nullptr) FOUR_C_THROW("No source discretization assigned to mapping dof set!");

  // get nodes to be coupled
  auto target_nodes_set = Core::Conditions::find_conditioned_node_ids(
      *source_dis_, coupling_cond_target_, Core::Conditions::LookFor::locally_owned);
  std::vector<int> target_nodes(target_nodes_set.begin(), target_nodes_set.end());
  std::set<int> source_nodes_set = Core::Conditions::find_conditioned_node_ids(
      dis, coupling_cond_source_, Core::Conditions::LookFor::locally_owned);
  std::vector<int> source_nodes(source_nodes_set.begin(), source_nodes_set.end());


  // initialize search tree
  auto tree = Core::GeometricSearch::NodeMatchingOctree();
  tree.init(*source_dis_, target_nodes, 150);
  tree.setup();

  // match target and source nodes using octtree
  // target id -> source id, distance
  std::map<int, std::pair<int, double>> coupling;
  tree.find_match(dis, source_nodes, coupling);

  // all nodes should be coupled
  if (target_nodes.size() != coupling.size())
    FOUR_C_THROW(
        "Did not get 1:1 correspondence. \ntarget_nodes.size()={}, coupling.size()={}."
        "DofSetMergedWrapper requires matching source and target meshes!",
        target_nodes.size(), coupling.size());

  // initialize final mapping
  Core::LinAlg::Vector<int> my_target_nodegids_row_layout(*dis.node_row_map());

  // loop over all coupled nodes
  for (unsigned i = 0; i < target_nodes.size(); ++i)
  {
    // get target gid
    int gid = target_nodes[i];

    // We allow to hand in target nodes that do not take part in the
    // coupling. If this is undesired behaviour the user has to make
    // sure all nodes were used.
    if (coupling.find(gid) != coupling.end())
    {
      std::pair<int, double>& coupled = coupling[gid];
      int source_gid = coupled.first;
      int source_lid = dis.node_row_map()->lid(source_gid);
      if (source_lid == -1) FOUR_C_THROW("source gid {} was not found on this proc", source_gid);

      // save target gid at col lid of corresponding source node
      (my_target_nodegids_row_layout.get_local_values())[source_lid] = gid;
    }
  }

  // initialize final mapping
  target_nodegids_col_layout_ = std::make_shared<Core::LinAlg::Vector<int>>(*dis.node_col_map());

  // export to column map
  Core::LinAlg::export_to(my_target_nodegids_row_layout, *target_nodegids_col_layout_);


  ////////////////////////////////////////////////////
  // now we match source and aux dis
  ////////////////////////////////////////////////////

  // get nodes to be coupled
  target_nodes_set = Core::Conditions::find_conditioned_node_ids(
      dis, coupling_cond_source_, Core::Conditions::LookFor::locally_owned);
  target_nodes = std::vector<int>(target_nodes_set.begin(), target_nodes_set.end());
  source_nodes_set = Core::Conditions::find_conditioned_node_ids(
      *source_dis_, coupling_cond_source_, Core::Conditions::LookFor::locally_owned);
  source_nodes = std::vector<int>(source_nodes_set.begin(), source_nodes_set.end());


  // initialize search tree
  tree.init(*source_dis_, target_nodes, 150);
  tree.setup();

  // match target and source nodes using octtree
  // target id -> source id, distance
  coupling.clear();
  tree.find_match(dis, source_nodes, coupling);

  // all nodes should be coupled
  if (target_nodes.size() != coupling.size())
    FOUR_C_THROW(
        "Did not get 1:1 correspondence. \ntarget_nodes.size()={}, coupling.size()={}."
        "DofSetMergedWrapper requires matching source and target meshes!",
        target_nodes.size(), coupling.size());

  // initialize final mapping
  Core::LinAlg::Vector<int> my_source_nodegids_row_layout(*dis.node_row_map());

  // loop over all coupled nodes
  for (unsigned i = 0; i < target_nodes.size(); ++i)
  {
    // get target gid
    int gid = target_nodes[i];

    // We allow to hand in target nodes that do not take part in the
    // coupling. If this is undesired behaviour the user has to make
    // sure all nodes were used.
    if (coupling.find(gid) != coupling.end())
    {
      std::pair<int, double>& coupled = coupling[gid];
      int source_gid = coupled.first;
      int source_lid = dis.node_row_map()->lid(source_gid);
      if (source_lid == -1) FOUR_C_THROW("source gid {} was not found on this proc", source_gid);

      // save target gid at col lid of corresponding source node
      (my_source_nodegids_row_layout.get_local_values())[source_lid] = gid;
    }
  }

  // initialize final mapping
  source_nodegids_col_layout_ = std::make_shared<Core::LinAlg::Vector<int>>(*dis.node_col_map());

  // export to column map
  Core::LinAlg::export_to(my_source_nodegids_row_layout, *source_nodegids_col_layout_);

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
  target_nodegids_col_layout_ = nullptr;
  source_nodegids_col_layout_ = nullptr;

  // set filled flag
  filled_ = false;

  notify_reset();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::DOFSets::DofSetMergedWrapper::disconnect(DofSetInterface* dofset)
{
  if (dofset == source_dofset_.get())
  {
    source_dofset_ = nullptr;
    source_dis_ = nullptr;
  }
  else
    FOUR_C_THROW("cannot disconnect from non-connected DofSet");

  // clear my Teuchos::rcps.
  reset();
}

FOUR_C_NAMESPACE_CLOSE

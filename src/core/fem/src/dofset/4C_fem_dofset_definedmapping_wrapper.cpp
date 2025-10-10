// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_dofset_definedmapping_wrapper.hpp"

#include "4C_fem_condition_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_dofset_base.hpp"
#include "4C_geometric_search_matchingoctree.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::DOFSets::DofSetDefinedMappingWrapper::DofSetDefinedMappingWrapper(
    std::shared_ptr<DofSetInterface> sourcedofset,
    std::shared_ptr<const Core::FE::Discretization> sourcedis, const std::string& couplingcond,
    const std::set<int> condids)
    : DofSetBase(),
      sourcedofset_(sourcedofset),
      targetlidtosourcegidmapping_(nullptr),
      sourcedis_(sourcedis),
      couplingcond_(couplingcond),
      condids_(condids),
      filled_(false)
{
  if (sourcedofset_ == nullptr) FOUR_C_THROW("Source dof set is null pointer.");
  if (sourcedis_ == nullptr) FOUR_C_THROW("Source discretization is null pointer.");

  sourcedofset_->register_proxy(this);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::DOFSets::DofSetDefinedMappingWrapper::~DofSetDefinedMappingWrapper()
{
  if (sourcedofset_ != nullptr) sourcedofset_->unregister(this);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
int Core::DOFSets::DofSetDefinedMappingWrapper::assign_degrees_of_freedom(
    const Core::FE::Discretization& dis, const unsigned dspos, const int start)
{
  if (sourcedofset_ == nullptr) FOUR_C_THROW("No source dof set assigned to mapping dof set!");
  if (sourcedis_ == nullptr) FOUR_C_THROW("No source discretization assigned to mapping dof set!");

  // get condition which defines the coupling on target discretization
  std::vector<const Core::Conditions::Condition*> conds;
  dis.get_condition(couplingcond_, conds);

  // get condition which defines the coupling on source discretization
  std::vector<const Core::Conditions::Condition*> conds_source;
  sourcedis_->get_condition(couplingcond_, conds_source);

  // get the respective nodes which are in the condition
  const bool use_coupling_id = condids_.size() != 1;
  std::map<int, std::shared_ptr<std::vector<int>>> nodes;
  Core::Conditions::find_conditioned_nodes(dis, conds, nodes, use_coupling_id);
  std::map<int, std::shared_ptr<std::vector<int>>> nodes_source;
  Core::Conditions::find_conditioned_nodes(
      *sourcedis_, conds_source, nodes_source, use_coupling_id);

  // map that will be filled with coupled nodes
  // mapping: target node gid to (source node gid, distance)
  std::map<int, std::pair<int, double>> coupling;

  // define iterators
  std::map<int, std::shared_ptr<std::vector<int>>>::iterator iter_target;
  std::map<int, std::shared_ptr<std::vector<int>>>::iterator iter_source;

  for (std::set<int>::iterator it = condids_.begin(); it != condids_.end(); ++it)
  {
    // find corresponding condition on source discretization
    iter_target = nodes.find(*it);
    // find corresponding condition on source discretization
    iter_source = nodes_source.find(*it);

    // get the nodes
    std::vector<int> sourcenodes;
    std::vector<int> targetnodes;
    if (iter_source != nodes_source.end()) sourcenodes = *iter_source->second;
    if (iter_target != nodes.end()) targetnodes = *iter_target->second;

    // initialize search tree for search
    Core::GeometricSearch::NodeMatchingOctree nodematchingtree;
    nodematchingtree.init(dis, targetnodes, 150, 1e-08);
    nodematchingtree.setup();

    // map that will be filled with coupled nodes for this condition
    // mapping: target node gid to (source node gid, distance)
    // note: FindMatch loops over all SOURCE (i.e. slave) nodes
    //       and finds corresponding target nodes.
    std::map<int, std::pair<int, double>> condcoupling;
    // match target and source nodes using octtree
    nodematchingtree.find_match(*sourcedis_, sourcenodes, condcoupling);

    // check if all nodes where matched for this condition ID
    if (targetnodes.size() != condcoupling.size())
      FOUR_C_THROW(
          "Did not get unique target to source spatial node coordinate mapping.\n"
          "targetnodes.size()={}, coupling.size()={}.\n"
          "The heterogeneous reaction strategy requires matching source and target meshes!",
          targetnodes.size(), condcoupling.size());

    // insert found coupling of this condition ID into map of all match nodes
    coupling.insert(condcoupling.begin(), condcoupling.end());

  }  // loop over all condition ids

  MPI_Comm com = dis.get_comm();

  // extract permutation
  std::vector<int> targetnodes(dis.node_row_map()->my_global_elements(),
      dis.node_row_map()->my_global_elements() + dis.node_row_map()->num_my_elements());

  std::vector<int> patchedtargetnodes;
  patchedtargetnodes.reserve(coupling.size());
  std::vector<int> permsourcenodes;
  permsourcenodes.reserve(coupling.size());

  for (unsigned i = 0; i < targetnodes.size(); ++i)
  {
    const int gid = targetnodes[i];

    // We allow to hand in target nodes that do not take part in the
    // coupling. If this is undesired behaviour the user has to make
    // sure all nodes were used.
    if (coupling.find(gid) != coupling.end())
    {
      std::pair<int, double>& coupled = coupling[gid];
      patchedtargetnodes.push_back(gid);
      permsourcenodes.push_back(coupled.first);
    }
  }

  // Epetra maps
  Core::LinAlg::Map targetnodemap(-1, patchedtargetnodes.size(), patchedtargetnodes.data(), 0, com);

  Core::LinAlg::Map permsourcenodemap(-1, permsourcenodes.size(), permsourcenodes.data(), 0, com);

  // we expect to get maps of exactly the same shape
  if (not targetnodemap.point_same_as(permsourcenodemap))
    FOUR_C_THROW("target and permuted source node maps do not match");

  // export target nodes to source node distribution
  Core::LinAlg::Vector<int> permsourcenodevec(
      targetnodemap, permsourcenodemap.my_global_elements());

  // initialize the final mapping
  targetlidtosourcegidmapping_ = std::make_shared<Core::LinAlg::Vector<int>>(*dis.node_col_map());

  // default value -1
  targetlidtosourcegidmapping_->put_value(-1);

  // export to column map
  Core::LinAlg::export_to(permsourcenodevec, *targetlidtosourcegidmapping_);

  // filled.
  filled_ = true;

  // tell the proxies
  notify_assigned();

  return start;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::DOFSets::DofSetDefinedMappingWrapper::reset()
{
  targetlidtosourcegidmapping_ = nullptr;
  filled_ = false;

  // tell the proxies
  notify_reset();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::DOFSets::DofSetDefinedMappingWrapper::disconnect(DofSetInterface* dofset)
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

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const Core::Nodes::Node* Core::DOFSets::DofSetDefinedMappingWrapper::get_source_node(
    int targetLid) const
{
  // check
  FOUR_C_ASSERT(
      targetLid <= targetlidtosourcegidmapping_->local_length(), "Target Lid out of range!");

  // get the gid of the source node
  int sourcegid = (targetlidtosourcegidmapping_->get_local_values())[targetLid];

  // the target is not mapped -> return null pointer
  if (sourcegid == -1) return nullptr;
  // get the node from the source discretization
  return sourcedis_->g_node(sourcegid);
}

FOUR_C_NAMESPACE_CLOSE

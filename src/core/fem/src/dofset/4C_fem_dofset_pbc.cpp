// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_dofset_pbc.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  ctor (public)                                            gammi 05/07|
 *----------------------------------------------------------------------*/
Core::DOFSets::PBCDofSet::PBCDofSet(std::shared_ptr<std::map<int, std::vector<int>>> couplednodes)
    : DofSet(), perbndcouples_(nullptr), myMaxGID_(-1)
{
  set_coupled_nodes(couplednodes);
}



/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
int Core::DOFSets::PBCDofSet::max_all_gid() const { return myMaxGID_; }


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
int Core::DOFSets::PBCDofSet::min_all_gid() const { return myMinGID_; }


int Core::DOFSets::PBCDofSet::assign_degrees_of_freedom(
    const Core::FE::Discretization& dis, const unsigned dspos, const int start)
{
  // temporarily store the source node set
  std::shared_ptr<std::set<int>> tempest = source_node_ids_;
  source_node_ids_ = std::make_shared<std::set<int>>();

  // assign dofs using the empty source node set. This way the dofrowmap_
  // contains exactly the entries as in a regular dofset
  DofSet::assign_degrees_of_freedom(dis, dspos, start);
  if (pccdofhandling_)
    FOUR_C_THROW("ERROR: Point coupling cinditions not yet implemented for PBCDofSet");

  myMaxGID_ = DofSet::max_all_gid();
  myMinGID_ = DofSet::min_all_gid();

  // restore the source node set
  source_node_ids_ = tempest;

  // assign dofs for the standard dofset, that is without periodic boundary
  // conditions and with the source node set back in place
  int count = DofSet::assign_degrees_of_freedom(dis, dspos, start);


  // loop all target nodes and set the dofs of the sources to the dofs of the target
  // remark: the previously assigned dofs of source nodes are overwritten here
  for (std::map<int, std::vector<int>>::iterator target = perbndcouples_->begin();
      target != perbndcouples_->end(); ++target)
  {
    int target_lid = dis.node_col_map()->lid(target->first);

    if (target_lid < 0)
    {
      FOUR_C_THROW("target gid {} not on proc {}, but required by source {}", target->first,
          Core::Communication::my_mpi_rank(dis.get_comm()), target->second[0]);
    }

    for (std::vector<int>::iterator source = target->second.begin(); source != target->second.end();
        ++source)
    {
      int source_lid = dis.node_col_map()->lid(*source);

      if (source_lid > -1)
      {
        (numdfcolnodes_->get_local_values())[source_lid] =
            (numdfcolnodes_->get_local_values())[target_lid];
        (idxcolnodes_->get_local_values())[source_lid] =
            (idxcolnodes_->get_local_values())[target_lid];
      }
      else
      {
#ifdef FOUR_C_ENABLE_ASSERTIONS
        if (dis.node_row_map()->my_gid(target->first))
        {
          FOUR_C_THROW("source not on proc but target owned by proc\n");
        }
#endif
      }
    }
  }

  return count;
}


/*----------------------------------------------------------------------*
 |  update coupled nodes map                             rasthofer 07/11|
 |                                                       DA wichmann    |
 *----------------------------------------------------------------------*/
void Core::DOFSets::PBCDofSet::set_coupled_nodes(
    std::shared_ptr<std::map<int, std::vector<int>>> couplednodes)
{
  perbndcouples_ = couplednodes;
  source_node_ids_ = std::make_shared<std::set<int>>();

  for (std::map<int, std::vector<int>>::iterator curr = perbndcouples_->begin();
      curr != perbndcouples_->end(); ++curr)
  {
    std::vector<int>& sids = curr->second;
    std::copy(
        sids.begin(), sids.end(), std::inserter(*source_node_ids_, source_node_ids_->begin()));
  }

  /// Build the connectivity between source node and its target node
  build_source_to_target_node_connectivity();

  return;
}

/*----------------------------------------------------------------------*
 |  Build the connectivity between source node and its target node       |
 |                                                       schott 05/15   |
 *----------------------------------------------------------------------*/
void Core::DOFSets::PBCDofSet::build_source_to_target_node_connectivity()
{
  perbnd_source_to_target_ = std::make_shared<std::map<int, int>>();

  for (std::map<int, std::vector<int>>::const_iterator target_source_pair = perbndcouples_->begin();
      target_source_pair != perbndcouples_->end(); ++target_source_pair)
  {
    // loop source nodes associated with target
    for (std::vector<int>::const_iterator iter = target_source_pair->second.begin();
        iter != target_source_pair->second.end(); ++iter)
    {
      const int source_gid = *iter;
      (*perbnd_source_to_target_)[source_gid] = target_source_pair->first;
    }
  }
}

FOUR_C_NAMESPACE_CLOSE

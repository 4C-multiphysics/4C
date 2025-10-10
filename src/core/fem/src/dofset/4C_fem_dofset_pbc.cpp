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
  // temporarily store the slave node set
  std::shared_ptr<std::set<int>> tempest = slavenodeids_;
  slavenodeids_ = std::make_shared<std::set<int>>();

  // assign dofs using the empty slave node set. This way the dofrowmap_
  // contains exactly the entries as in a regular dofset
  DofSet::assign_degrees_of_freedom(dis, dspos, start);
  if (pccdofhandling_)
    FOUR_C_THROW("ERROR: Point coupling cinditions not yet implemented for PBCDofSet");

  myMaxGID_ = DofSet::max_all_gid();
  myMinGID_ = DofSet::min_all_gid();

  // restore the slave node set
  slavenodeids_ = tempest;

  // assign dofs for the standard dofset, that is without periodic boundary
  // conditions and with the slave node set back in place
  int count = DofSet::assign_degrees_of_freedom(dis, dspos, start);


  // loop all master nodes and set the dofs of the slaves to the dofs of the master
  // remark: the previously assigned dofs of slave nodes are overwritten here
  for (std::map<int, std::vector<int>>::iterator master = perbndcouples_->begin();
      master != perbndcouples_->end(); ++master)
  {
    int master_lid = dis.node_col_map()->lid(master->first);

    if (master_lid < 0)
    {
      FOUR_C_THROW("master gid {} not on proc {}, but required by slave {}", master->first,
          Core::Communication::my_mpi_rank(dis.get_comm()), master->second[0]);
    }

    for (std::vector<int>::iterator slave = master->second.begin(); slave != master->second.end();
        ++slave)
    {
      int slave_lid = dis.node_col_map()->lid(*slave);

      if (slave_lid > -1)
      {
        (numdfcolnodes_->get_local_values())[slave_lid] =
            (numdfcolnodes_->get_local_values())[master_lid];
        (idxcolnodes_->get_local_values())[slave_lid] =
            (idxcolnodes_->get_local_values())[master_lid];
      }
      else
      {
#ifdef FOUR_C_ENABLE_ASSERTIONS
        if (dis.node_row_map()->my_gid(master->first))
        {
          FOUR_C_THROW("slave not on proc but master owned by proc\n");
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
  slavenodeids_ = std::make_shared<std::set<int>>();

  for (std::map<int, std::vector<int>>::iterator curr = perbndcouples_->begin();
      curr != perbndcouples_->end(); ++curr)
  {
    std::vector<int>& sids = curr->second;
    std::copy(sids.begin(), sids.end(), std::inserter(*slavenodeids_, slavenodeids_->begin()));
  }

  /// Build the connectivity between slave node and its master node
  build_slave_to_master_node_connectivity();

  return;
}

/*----------------------------------------------------------------------*
 |  Build the connectivity between slave node and its master node       |
 |                                                       schott 05/15   |
 *----------------------------------------------------------------------*/
void Core::DOFSets::PBCDofSet::build_slave_to_master_node_connectivity()
{
  perbnd_slavetomaster_ = std::make_shared<std::map<int, int>>();

  for (std::map<int, std::vector<int>>::const_iterator masterslavepair = perbndcouples_->begin();
      masterslavepair != perbndcouples_->end(); ++masterslavepair)
  {
    // loop slave nodes associated with master
    for (std::vector<int>::const_iterator iter = masterslavepair->second.begin();
        iter != masterslavepair->second.end(); ++iter)
    {
      const int slavegid = *iter;
      (*perbnd_slavetomaster_)[slavegid] = masterslavepair->first;
    }
  }
}

FOUR_C_NAMESPACE_CLOSE

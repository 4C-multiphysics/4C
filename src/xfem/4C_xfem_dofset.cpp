// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_xfem_dofset.hpp"

#include "4C_cut_cutwizard.hpp"
#include "4C_cut_node.hpp"

FOUR_C_NAMESPACE_OPEN



/*----------------------------------------------------------------------*
 |  Get the gid of all dofs of a node                      schott 12/14 |
 *----------------------------------------------------------------------*/
void XFEM::XFEMDofSet::dof(
    std::vector<int>& dofs, const Core::Nodes::Node* node, unsigned nodal_dofset_id) const
{
  const int lid = node->lid();
  if (lid == -1) return;
  int numdf = Core::DOFSets::DofSet::num_dof_per_node(*node);
  const int idx = idxcolnodes_->get_local_values()[lid] + nodal_dofset_id * numdf;
  dofs.reserve(numdf);
  for (int i = 0; i < numdf; ++i)
  {
    dofs.push_back(idx + i);
  }
}

/*----------------------------------------------------------------------*
 |  Get the gid of all dofs of a node                      schott 12/14 |
 *----------------------------------------------------------------------*/
int XFEM::XFEMDofSet::num_dof_per_node(const Core::Nodes::Node& node) const
{
  Cut::Node* n = wizard_.get_node(node.id());
  if (n != nullptr)
  {
    int numdofpernode = Core::DOFSets::DofSet::num_dof_per_node(node);
    return numdofpernode * n->num_dof_sets();
  }
  return Core::DOFSets::DofSet::num_dof_per_node(node);
}

FOUR_C_NAMESPACE_CLOSE

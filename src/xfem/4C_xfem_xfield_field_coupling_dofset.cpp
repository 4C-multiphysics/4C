// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_xfem_xfield_field_coupling_dofset.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
XFEM::XFieldField::CouplingDofSet::CouplingDofSet(const int& my_num_reserve_dof_per_node,
    const int& g_node_index_range, const int& g_num_std_dof_per_node,
    const std::map<int, int>& my_num_dofs_per_node)
    : Core::DOFSets::FixedSizeDofSet(my_num_reserve_dof_per_node, g_node_index_range),
      my_num_dof_per_node_(my_num_dofs_per_node),
      g_num_std_dof_per_node_(g_num_std_dof_per_node)
{
  // empty
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void XFEM::XFieldField::CouplingDofSet::dof(
    std::vector<int>& dofs, const Core::Nodes::Node* node, unsigned nodal_dofset_id) const
{
  const int lid = node->lid();
  if (lid == -1) return;
  const int num_dof = num_standard_dof_per_node();
  const int idx = idxcolnodes_->get_local_values()[lid] + nodal_dofset_id * num_dof;
  dofs.resize(num_dof, 0);
  for (int i = 0; i < num_dof; ++i) dofs[i] = idx + i;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
int XFEM::XFieldField::CouplingDofSet::num_dof_per_node(const Core::Nodes::Node& node) const
{
  return my_num_dof_per_node(node.id());
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
int XFEM::XFieldField::CouplingDofSet::my_num_dof_per_node(const int& node_gid) const
{
  std::map<int, int>::const_iterator pos = my_num_dof_per_node_.find(node_gid);
  if (pos == my_num_dof_per_node_.end())
    FOUR_C_THROW("The given node GID {} is no coupling interface node!", node_gid);

  return pos->second;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
int XFEM::XFieldField::CouplingDofSet::num_standard_dof_per_node() const
{
  return g_num_std_dof_per_node_;
}

FOUR_C_NAMESPACE_CLOSE

#ifndef FOUR_C_XFEM_DOFSET_HPP
#define FOUR_C_XFEM_DOFSET_HPP


#include "4C_config.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_dofset_fixed_size.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE


namespace Cut
{
  class CutWizard;
}

namespace XFEM
{
  class XFEMDofSet : public Core::DOFSets::FixedSizeDofSet
  {
   public:
    /// constructor
    XFEMDofSet(Cut::CutWizard& wizard, int numMyReservedDofsperNode, Core::FE::Discretization& dis)
        : FixedSizeDofSet(numMyReservedDofsperNode,
              dis.node_row_map()->MaxAllGID() - dis.node_row_map()->MinAllGID() + 1),
          wizard_(wizard),
          dis_(dis)
    {
    }

    /// equality relational operator for two XFEM::DofSets based on the number of nodal dofsets for
    /// all nodes
    bool operator==(XFEMDofSet const& other) const
    {
      const int numnode = dis_.num_my_row_nodes();
      for (int lid = 0; lid < numnode; lid++)
      {
        int gid = dis_.node_row_map()->GID(lid);
        Core::Nodes::Node* node = dis_.g_node(gid);
        if (num_dof_per_node(*node) != other.num_dof_per_node(*node))
          return false;  // dofsets not equal if at least one node has a different number of nodal
                         // dofsets
      }

      return true;
    }

    /// equality relational operator for two XFEM::DofSets based on the number of dofsets for all
    /// nodes
    bool operator!=(XFEMDofSet const& other) const { return !(*this == other); }

    /*!
    \brief Get the gid of all dofs of a node.

    Ask the current DofSet for the gids of the dofs of this node. The
    required vector is created and filled on the fly. So better keep it
    if you need more than one dof gid.
    - HaveDofs()==true prerequisite (produced by call to assign_degrees_of_freedom()))

    Additional input nodal_dof_set_id: If the node contains more than one set of dofs, which can be
    evaluated, the ID of the set needs to be given. Currently only the case for XFEM.

    \param dof             (out): vector of dof gids (to be filled)
    \param node            (in) : the node
    \param nodal_dofset_id (in) : id of the nodal dofset
    */
    void dof(std::vector<int>& dofs, const Core::Nodes::Node* node,
        unsigned nodal_dofset_id) const override;

   protected:
    /// get number of nodal dofs for this element at this node
    int num_dof_per_node(const Core::Nodes::Node& node) const override;

   private:
    /// the cut wizard, holds information about the number of XFEM dofsets per node
    Cut::CutWizard& wizard_;

    /// background discretization
    Core::FE::Discretization&
        dis_;  ///< use reference instead of RCP to avoid Ringschluss in discretization
  };
}  // namespace XFEM

FOUR_C_NAMESPACE_CLOSE

#endif

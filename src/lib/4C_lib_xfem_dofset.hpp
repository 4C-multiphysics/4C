/*----------------------------------------------------------------------*/
/*! \file

\brief provides a general XFEM dofset which uses the information from the cut-library to determine
the number of dofs per node when multiple sets of degrees of freedom per node have to be used


\level 1

*/
/*----------------------------------------------------------------------*/


#ifndef FOUR_C_LIB_XFEM_DOFSET_HPP
#define FOUR_C_LIB_XFEM_DOFSET_HPP


#include "4C_config.hpp"

#include "4C_lib_discret.hpp"
#include "4C_lib_dofset_fixed_size.hpp"

FOUR_C_NAMESPACE_OPEN

namespace DRT
{
  class Discretization;
}

namespace CORE::GEO
{
  class CutWizard;
}

namespace XFEM
{
  class XFEMDofSet : public DRT::FixedSizeDofSet
  {
   public:
    /// constructor
    XFEMDofSet(CORE::GEO::CutWizard& wizard, int numMyReservedDofsperNode, DRT::Discretization& dis)
        : FixedSizeDofSet(numMyReservedDofsperNode,
              dis.NodeRowMap()->MaxAllGID() - dis.NodeRowMap()->MinAllGID() + 1),
          wizard_(wizard),
          dis_(dis)
    {
    }

    /// equality relational operator for two XFEM::DofSets based on the number of nodal dofsets for
    /// all nodes
    bool operator==(XFEMDofSet const& other) const
    {
      const int numnode = dis_.NumMyRowNodes();
      for (int lid = 0; lid < numnode; lid++)
      {
        int gid = dis_.NodeRowMap()->GID(lid);
        DRT::Node* node = dis_.gNode(gid);
        if (NumDofPerNode(*node) != other.NumDofPerNode(*node))
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
    - HaveDofs()==true prerequisite (produced by call to AssignDegreesOfFreedom()))

    Additional input nodal_dof_set_id: If the node contains more than one set of dofs, which can be
    evaluated, the ID of the set needs to be given. Currently only the case for XFEM.

    \param dof             (out): vector of dof gids (to be filled)
    \param node            (in) : the node
    \param nodal_dofset_id (in) : id of the nodal dofset
    */
    void Dof(
        std::vector<int>& dofs, const DRT::Node* node, unsigned nodal_dofset_id) const override;

   protected:
    /// get number of nodal dofs for this element at this node
    int NumDofPerNode(const DRT::Node& node) const override;

   private:
    /// the cut wizard, holds information about the number of XFEM dofsets per node
    CORE::GEO::CutWizard& wizard_;

    /// background discretization
    DRT::Discretization&
        dis_;  ///< use reference instead of RCP to avoid Ringschluss in discretization
  };
}  // namespace XFEM

FOUR_C_NAMESPACE_CLOSE

#endif

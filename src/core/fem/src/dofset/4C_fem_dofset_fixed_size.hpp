// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FEM_DOFSET_FIXED_SIZE_HPP
#define FOUR_C_FEM_DOFSET_FIXED_SIZE_HPP

#include "4C_config.hpp"

#include "4C_fem_dofset.hpp"

#include <Epetra_Map.h>

#include <map>
#include <memory>
#include <vector>

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::DOFSets
{
  /*!
  \brief A set of degrees of freedom

  \author
  */
  class FixedSizeDofSet : virtual public DofSet
  {
   public:
    /*!
    \brief Standard Constructor



    create a dofset that reserves a certain amount of dofGIDs


    \return void

    */
    FixedSizeDofSet(int numreservedofpernode, int nodeindexrange)
        : DofSet(),
          numMyReservedDofs_(numreservedofpernode * nodeindexrange),
          numMyReservedDofsperNode_(numreservedofpernode)
    {
      minGID_ = -1;
      return;
    }



    /// create a copy of this object
    std::shared_ptr<DofSet> clone() override { return std::make_shared<FixedSizeDofSet>(*this); }

    /// Get maximum GID of degree of freedom row map
    int max_all_gid() const override { return min_all_gid() + numMyReservedDofs_; }

    /// Get minimum GID of degree of freedom row map
    int min_all_gid() const override
    {
      int mymindof;
      if (minGID_ == -1)
        mymindof = dofrowmap_->MinAllGID();
      else
      {
        int minall = dofrowmap_->MinAllGID();
        mymindof = (minGID_ <= minall) ? minGID_ : minall;
      }
      return mymindof;
    }

    /// set the minimal global id
    virtual void set_min_gid(int mingid) { minGID_ = mingid; }

    /// Get Reserved Max Number Dofs per Node
    void get_reserved_max_num_dofper_node(int& maxnodenumdf) override
    {
      if (maxnodenumdf > numMyReservedDofsperNode_)
      {
        FOUR_C_THROW(
            "FixedSizeDofSet::get_reserved_max_num_dofper_node: Not enough Dofs reserved!!!");
        return;
      }
      maxnodenumdf = numMyReservedDofsperNode_;

      return;
    }

    /// get the number of reserved DoF's (see also num_my_reserved_dofs_per_node())
    int num_my_reserved_dofs() const { return numMyReservedDofs_; }

    /// get the number of reserved DoF's per node
    int num_my_reserved_dofs_per_node() const { return numMyReservedDofsperNode_; }

    /// minimal global dof id
    int minGID_;

   protected:
    /// Number of reserved dofs
    int numMyReservedDofs_;

    /// Number of reserved dofs per node
    int numMyReservedDofsperNode_;

   private:
  };  // class FixedSizeDofSet

}  // namespace Core::DOFSets

FOUR_C_NAMESPACE_CLOSE

#endif

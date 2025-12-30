// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FEM_DOFSET_PBC_HPP
#define FOUR_C_FEM_DOFSET_PBC_HPP

#include "4C_config.hpp"

#include "4C_fem_dofset.hpp"

#include <map>
#include <memory>
#include <set>
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
  */
  class PBCDofSet : virtual public DofSet
  {
   public:
    /*!
    \brief Standard Constructor



    create a dofset that allows coupled nodes for periodic boundary
    conditions                                         gammi 05/07


    \param couplednodes (i) list of coupled nodes


    \return void

    */
    PBCDofSet(std::shared_ptr<std::map<int, std::vector<int>>> couplednodes);


    /// Get maximum GID of degree of freedom row map
    int max_all_gid() const override;

    /// Get minimum GID of degree of freedom row map
    int min_all_gid() const override;

    /// create a copy of this object
    std::shared_ptr<DofSet> clone() override { return std::make_shared<PBCDofSet>(*this); }

    /// Assign dof numbers using all elements and nodes of the discretization.
    int assign_degrees_of_freedom(
        const Core::FE::Discretization& dis, const unsigned dspos, const int start) override;

    /// Update the coupled nodes map of dofset
    virtual void set_coupled_nodes(std::shared_ptr<std::map<int, std::vector<int>>> couplednodes);

    /// Get coupled nodes map (corresponding col format)
    [[nodiscard]] const std::map<int, std::vector<int>>* get_coupled_nodes() const
    {
      return perbndcouples_.get();
    }

    /// Get connectivity map between slave node and its master node
    [[nodiscard]] virtual std::shared_ptr<const std::map<int, int>>
    get_slave_to_master_node_connectivity() const
    {
      return perbnd_slavetomaster_;
    };

   protected:
    /// get number of nodal dofs for this element at this node
    int num_dof_per_node(const Core::Nodes::Node& node) const override
    {
      if (slavenodeids_->count(node.id()) == 0)
      {
        return DofSet::num_dof_per_node(node);
      }
      return 0;
    }

    //!\brief master and slave node connectivity for periodic boundary conditions
    std::shared_ptr<std::map<int, std::vector<int>>> perbndcouples_;

    //!\brief the largest original GID, to stop the dofset from 'shrinking'
    int myMaxGID_;

    //!\brief the smallest original GID, to stop the dofset from 'shrinking'
    int myMinGID_;

   private:
    /// Build the connectivity between slave node and its master node
    void build_slave_to_master_node_connectivity();

    std::shared_ptr<std::set<int>> slavenodeids_;

    //!\brief slave node to master node connectivity for periodic boundary conditions (key=slave
    //! nid, value=master nid)
    std::shared_ptr<std::map<int, int>> perbnd_slavetomaster_;

  };  // class PBCDofSet

}  // namespace Core::DOFSets

FOUR_C_NAMESPACE_CLOSE

#endif

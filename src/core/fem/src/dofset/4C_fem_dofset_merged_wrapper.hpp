// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FEM_DOFSET_MERGED_WRAPPER_HPP
#define FOUR_C_FEM_DOFSET_MERGED_WRAPPER_HPP


#include "4C_config.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_dofset_proxy.hpp"
#include "4C_fem_general_node.hpp"


FOUR_C_NAMESPACE_OPEN

namespace Core::DOFSets
{
  /*!
  \brief A dofset that adds additional, existing degrees of freedom from the same
         discretization to nodes.

  \warning not implemented for element DOFs

    The Dofs of the nodes to be merged are defined by master and slave side conditions given as
  input. Overlapping nodes are identified using a search tree and this dofset will handle the dofs
  of one node as if there were one in the Dof(..) and NumDof(..) methods (see the implementation of
  these methods).

  \warning For the Dof(..) methods providing the full dof vector an ordering of the nodes is
  assumed. That is, first the Dofs from the slave node are filled into the dof vector followed by
  the Dofs of the master node.

  */

  class DofSetMergedWrapper : public DofSetBase
  {
   public:
    //! Standard Constructor
    DofSetMergedWrapper(std::shared_ptr<DofSetInterface> dofset,
        std::shared_ptr<const Core::FE::Discretization> sourcedis,
        const std::string& couplingcond_master, const std::string& couplingcond_slave);

    //! Destructor
    ~DofSetMergedWrapper() override;

    /// Returns true if filled
    bool filled() const override { return filled_ and sourcedofset_->filled(); };

    /// Assign dof numbers using all elements and nodes of the discretization.
    int assign_degrees_of_freedom(
        const Core::FE::Discretization& dis, const unsigned dspos, const int start) override;

    /// reset all internal variables
    void reset() override;

    //! @name Proxy management
    /// Proxies need to know about changes to the DofSet.

    /// our original DofSet dies
    void disconnect(DofSetInterface* dofset) override;

    //@}

    /// Get degree of freedom row map
    const Core::LinAlg::Map* dof_row_map() const override;

    /// Get degree of freedom column map
    const Core::LinAlg::Map* dof_col_map() const override;

    //! @name Access methods

    /// Get number of dofs for given node
    int num_dof(const Core::Nodes::Node* node) const override
    {
      const Core::Nodes::Node* masternode = get_master_node(node->lid());
      const Core::Nodes::Node* slavenode = get_slave_node(node->lid());
      return sourcedofset_->num_dof(slavenode) + sourcedofset_->num_dof(masternode);
    }

    /// Get number of dofs for given element
    int num_dof(const Core::Elements::Element* element) const override
    {
      return sourcedofset_->num_dof(element);
    }

    /// get number of nodal dofs
    int num_dof_per_node(
        const Core::Nodes::Node& node  ///< node, for which you want to know the number of dofs
    ) const override
    {
      const Core::Nodes::Node* masternode = get_master_node(node.lid());
      const Core::Nodes::Node* slavenode = get_slave_node(node.lid());
      return sourcedofset_->num_dof_per_node(*masternode) +
             sourcedofset_->num_dof_per_node(*slavenode);
    };

    /*! \brief Get the gid of all dofs of a node
     *
     \note convention: First the slave dofs and then the master dofs are inserted into full dof
     vector! Thus all definitions in the input file concerning dof numbering have to be set
     accordingly (e.g. for reactions in MAT_scatra_reaction and MAT_scatra_reaction, see test case
     'ssi_3D_tet4_tet4_tri3.4C.yaml')  */
    std::vector<int> dof(const Core::Nodes::Node* node) const override
    {
      const Core::Nodes::Node* slavenode = get_slave_node(node->lid());
      std::vector<int> slavedof = sourcedofset_->dof(slavenode);
      const Core::Nodes::Node* masternode = get_master_node(node->lid());
      std::vector<int> masterdof = sourcedofset_->dof(masternode);

      std::vector<int> dof;
      dof.reserve(slavedof.size() + masterdof.size());  // preallocate memory
      dof.insert(dof.end(), slavedof.begin(), slavedof.end());
      dof.insert(dof.end(), masterdof.begin(), masterdof.end());

      return dof;
    }

    /// Get the gid of all dofs of a node
    void dof(std::vector<int>& dof,     ///< vector of dof gids (to be filled)
        const Core::Nodes::Node* node,  ///< node, for which you want the dof positions
        unsigned nodaldofset  ///< number of nodal dof set of the node (currently !=0 only for XFEM)
    ) const override
    {
      const Core::Nodes::Node* masternode = get_master_node(node->lid());
      const Core::Nodes::Node* slavenode = get_slave_node(node->lid());
      std::vector<int> slavedof;
      sourcedofset_->dof(slavedof, slavenode, nodaldofset);
      std::vector<int> masterdof;
      sourcedofset_->dof(masterdof, masternode, nodaldofset);

      dof.reserve(slavedof.size() + masterdof.size());  // preallocate memory
      dof.insert(dof.end(), slavedof.begin(), slavedof.end());
      dof.insert(dof.end(), masterdof.begin(), masterdof.end());
    }

    /// Get the gid of all dofs of a element
    std::vector<int> dof(const Core::Elements::Element* element) const override
    {
      return sourcedofset_->dof(element);
    }

    /// Get the gid of a dof for given node
    int dof(const Core::Nodes::Node* node, int dof) const override
    {
      const Core::Nodes::Node* slavenode = get_slave_node(node->lid());
      const int numslavedofs = sourcedofset_->num_dof(slavenode);
      if (dof < numslavedofs)
        return sourcedofset_->dof(slavenode, dof);
      else
      {
        const Core::Nodes::Node* masternode = get_master_node(node->lid());
        return sourcedofset_->dof(masternode, dof - numslavedofs);
      }
    }

    /// Get the gid of a dof for given element
    int dof(const Core::Elements::Element* element, int dof) const override
    {
      return sourcedofset_->dof(element, dof);
    }

    /// Get the gid of all dofs of a node
    void dof(const Core::Nodes::Node* node, std::vector<int>& lm) const override
    {
      const Core::Nodes::Node* masternode = get_master_node(node->lid());
      const Core::Nodes::Node* slavenode = get_slave_node(node->lid());
      sourcedofset_->dof(slavenode, lm);
      sourcedofset_->dof(masternode, lm);

      return;
    }

    /// Get the gid of all dofs of a node
    void dof(const Core::Nodes::Node* node,  ///< node, for which you want the dof positions
        const unsigned startindex,  ///< first index of vector at which will be written to end
        std::vector<int>& lm        ///< already allocated vector to be filled with dof positions
    ) const override
    {
      const Core::Nodes::Node* slavenode = get_slave_node(node->lid());
      const int numslavedofs = sourcedofset_->num_dof(slavenode);
      sourcedofset_->dof(slavenode, startindex, lm);

      const Core::Nodes::Node* masternode = get_master_node(node->lid());
      sourcedofset_->dof(masternode, startindex + numslavedofs, lm);
    }

    /// Get the gid of all dofs of a element
    void dof(const Core::Elements::Element* element, std::vector<int>& lm) const override
    {
      sourcedofset_->dof(element, lm);
    }

    /// Get the GIDs of the first DOFs of a node of which the associated element is interested in
    void dof(const Core::Elements::Element*
                 element,  ///< element which provides its expected number of DOFs per node
        const Core::Nodes::Node* node,  ///< node, for which you want the DOF positions
        std::vector<int>& lm  ///< already allocated vector to be filled with DOF positions
    ) const override
    {
      const Core::Nodes::Node* slavenode = get_slave_node(node->lid());
      sourcedofset_->dof(element, slavenode, lm);

      const Core::Nodes::Node* masternode = get_master_node(node->lid());
      sourcedofset_->dof(element, masternode, lm);
    }

    /// Get maximum GID of degree of freedom row map
    int max_all_gid() const override { return sourcedofset_->max_all_gid(); };

    /// Get minimum GID of degree of freedom row map
    int min_all_gid() const override { return sourcedofset_->min_all_gid(); };

    /// Get Max of all GID assigned in the DofSets in front of current one in the list
    /// #static_dofsets_
    int max_gid_in_list(MPI_Comm comm) const override
    {
      return sourcedofset_->max_gid_in_list(comm);
    };

    /// are the dof maps already initialized?
    bool initialized() const override { return sourcedofset_->initialized(); };

    /// Get Number of Global Elements of degree of freedom row map
    int num_global_elements() const override { return sourcedofset_->num_global_elements(); };

   private:
    //! get the master node to a corresponding slave node (given by LID)
    const Core::Nodes::Node* get_master_node(int slaveLid) const
    {
      FOUR_C_ASSERT(
          slaveLid < master_nodegids_col_layout_->local_length(), "Slave node Lid out of range!");
      int mastergid = (master_nodegids_col_layout_->get_local_values())[slaveLid];
      // std::cout<<"master gid = "<<mastergid<<" <-> slave lid ="<<slaveLid<<"  size of map =
      // "<<master_nodegids_col_layout_->MyLength()<<std::endl;
      return sourcedis_->g_node(mastergid);
    }

    //! get the slave node to a corresponding master node (given by LID)
    const Core::Nodes::Node* get_slave_node(int masterLid) const
    {
      FOUR_C_ASSERT(
          masterLid < slave_nodegids_col_layout_->local_length(), "Master node Lid out of range!");
      int slavegid = (slave_nodegids_col_layout_->get_local_values())[masterLid];
      return sourcedis_->g_node(slavegid);
    }

    //! master node gids in col layout matching conditioned slave nodes
    std::shared_ptr<Core::LinAlg::Vector<int>> master_nodegids_col_layout_;

    //! slave node gids in col layout matching conditioned master nodes
    std::shared_ptr<Core::LinAlg::Vector<int>> slave_nodegids_col_layout_;

    //! underlying actual dofset
    std::shared_ptr<DofSetInterface> sourcedofset_;

    //! source discretization
    std::shared_ptr<const Core::FE::Discretization> sourcedis_;

    //! condition strings defining the coupling
    const std::string couplingcond_master_;
    const std::string couplingcond_slave_;

    /// filled flag
    bool filled_;
  };
}  // namespace Core::DOFSets


FOUR_C_NAMESPACE_CLOSE

#endif

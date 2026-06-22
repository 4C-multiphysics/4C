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

    The Dofs of the nodes to be merged are defined by target and source side conditions given as
  input. Overlapping nodes are identified using a search tree and this dofset will handle the dofs
  of one node as if there were one in the Dof(..) and NumDof(..) methods (see the implementation of
  these methods).

  \warning For the Dof(..) methods providing the full dof vector an ordering of the nodes is
  assumed. That is, first the Dofs from the source node are filled into the dof vector followed by
  the Dofs of the target node.

  */

  class DofSetMergedWrapper : public DofSetBase
  {
   public:
    //! Standard Constructor
    DofSetMergedWrapper(std::shared_ptr<DofSetInterface> dofset,
        std::shared_ptr<const Core::FE::Discretization> sourcedis,
        const std::string& coupling_cond_target, const std::string& coupling_cond_source);

    //! Destructor
    ~DofSetMergedWrapper() override;

    /// Returns true if filled
    bool filled() const override { return filled_ and source_dofset_->filled(); };

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
      const Core::Nodes::Node* target_node = get_target_node(node->lid());
      const Core::Nodes::Node* source_node = get_source_node(node->lid());
      return source_dofset_->num_dof(source_node) + source_dofset_->num_dof(target_node);
    }

    /// Get number of dofs for given element
    int num_dof(const Core::Elements::Element* element) const override
    {
      return source_dofset_->num_dof(element);
    }

    /// get number of nodal dofs
    int num_dof_per_node(
        const Core::Nodes::Node& node  ///< node, for which you want to know the number of dofs
    ) const override
    {
      const Core::Nodes::Node* target_node = get_target_node(node.lid());
      const Core::Nodes::Node* source_node = get_source_node(node.lid());
      return source_dofset_->num_dof_per_node(*target_node) +
             source_dofset_->num_dof_per_node(*source_node);
    };

    /*! \brief Get the gid of all dofs of a node
     *
     \note convention: First the source dofs and then the target dofs are inserted into full dof
     vector! Thus all definitions in the input file concerning dof numbering have to be set
     accordingly (e.g. for reactions in MAT_scatra_reaction and MAT_scatra_reaction, see test case
     'ssi_3D_tet4_tet4_tri3.4C.yaml')  */
    std::vector<int> dof(const Core::Nodes::Node* node) const override
    {
      const Core::Nodes::Node* source_node = get_source_node(node->lid());
      std::vector<int> source_dof = source_dofset_->dof(source_node);
      const Core::Nodes::Node* target_node = get_target_node(node->lid());
      std::vector<int> target_dof = source_dofset_->dof(target_node);

      std::vector<int> dof;
      dof.reserve(source_dof.size() + target_dof.size());  // preallocate memory
      dof.insert(dof.end(), source_dof.begin(), source_dof.end());
      dof.insert(dof.end(), target_dof.begin(), target_dof.end());

      return dof;
    }

    /// Get the gid of all dofs of a node
    void dof(std::vector<int>& dof,     ///< vector of dof gids (to be filled)
        const Core::Nodes::Node* node,  ///< node, for which you want the dof positions
        unsigned nodaldofset  ///< number of nodal dof set of the node (currently !=0 only for XFEM)
    ) const override
    {
      const Core::Nodes::Node* target_node = get_target_node(node->lid());
      const Core::Nodes::Node* source_node = get_source_node(node->lid());
      std::vector<int> source_dof;
      source_dofset_->dof(source_dof, source_node, nodaldofset);
      std::vector<int> target_dof;
      source_dofset_->dof(target_dof, target_node, nodaldofset);

      dof.reserve(source_dof.size() + target_dof.size());  // preallocate memory
      dof.insert(dof.end(), source_dof.begin(), source_dof.end());
      dof.insert(dof.end(), target_dof.begin(), target_dof.end());
    }

    /// Get the gid of all dofs of a element
    std::vector<int> dof(const Core::Elements::Element* element) const override
    {
      return source_dofset_->dof(element);
    }

    /// Get the gid of a dof for given node
    int dof(const Core::Nodes::Node* node, int dof) const override
    {
      const Core::Nodes::Node* source_node = get_source_node(node->lid());
      const int num_source_dofs = source_dofset_->num_dof(source_node);
      if (dof < num_source_dofs)
        return source_dofset_->dof(source_node, dof);
      else
      {
        const Core::Nodes::Node* target_node = get_target_node(node->lid());
        return source_dofset_->dof(target_node, dof - num_source_dofs);
      }
    }

    /// Get the gid of a dof for given element
    int dof(const Core::Elements::Element* element, int dof) const override
    {
      return source_dofset_->dof(element, dof);
    }

    /// Get the gid of all dofs of a node
    void dof(const Core::Nodes::Node* node, std::vector<int>& lm) const override
    {
      const Core::Nodes::Node* target_node = get_target_node(node->lid());
      const Core::Nodes::Node* source_node = get_source_node(node->lid());
      source_dofset_->dof(source_node, lm);
      source_dofset_->dof(target_node, lm);
    }

    /// Get the gid of all dofs of a node
    void dof(const Core::Nodes::Node* node,  ///< node, for which you want the dof positions
        const unsigned startindex,  ///< first index of vector at which will be written to end
        std::vector<int>& lm        ///< already allocated vector to be filled with dof positions
    ) const override
    {
      const Core::Nodes::Node* source_node = get_source_node(node->lid());
      const int num_source_dofs = source_dofset_->num_dof(source_node);
      source_dofset_->dof(source_node, startindex, lm);

      const Core::Nodes::Node* target_node = get_target_node(node->lid());
      source_dofset_->dof(target_node, startindex + num_source_dofs, lm);
    }

    /// Get the gid of all dofs of a element
    void dof(const Core::Elements::Element* element, std::vector<int>& lm) const override
    {
      source_dofset_->dof(element, lm);
    }

    /// Get the GIDs of the first DOFs of a node of which the associated element is interested in
    void dof(const Core::Elements::Element*
                 element,  ///< element which provides its expected number of DOFs per node
        const Core::Nodes::Node* node,  ///< node, for which you want the DOF positions
        std::vector<int>& lm  ///< already allocated vector to be filled with DOF positions
    ) const override
    {
      const Core::Nodes::Node* source_node = get_source_node(node->lid());
      source_dofset_->dof(element, source_node, lm);

      const Core::Nodes::Node* target_node = get_target_node(node->lid());
      source_dofset_->dof(element, target_node, lm);
    }

    /// Get maximum GID of degree of freedom row map
    int max_all_gid() const override { return source_dofset_->max_all_gid(); };

    /// Get minimum GID of degree of freedom row map
    int min_all_gid() const override { return source_dofset_->min_all_gid(); };

    /// Get Max of all GID assigned in the DofSets in front of current one in the list
    /// #static_dofsets_
    int max_gid_in_list(MPI_Comm comm) const override
    {
      return source_dofset_->max_gid_in_list(comm);
    };

    /// are the dof maps already initialized?
    bool initialized() const override { return source_dofset_->initialized(); };

    /// Get Number of Global Elements of degree of freedom row map
    int num_global_elements() const override { return source_dofset_->num_global_elements(); };

   private:
    //! get the target node to a corresponding source node (given by LID)
    const Core::Nodes::Node* get_target_node(int source_lid) const
    {
      FOUR_C_ASSERT(source_lid < target_nodegids_col_layout_->local_length(),
          "Source node local id out of range!");
      int target_gid = (target_nodegids_col_layout_->get_local_values())[source_lid];
      return source_dis_->g_node(target_gid);
    }

    //! get the source node to a corresponding target node (given by LID)
    const Core::Nodes::Node* get_source_node(int target_lid) const
    {
      FOUR_C_ASSERT(target_lid < source_nodegids_col_layout_->local_length(),
          "Target node local id out of range!");
      int source_gid = (source_nodegids_col_layout_->get_local_values())[target_lid];
      return source_dis_->g_node(source_gid);
    }

    //! target node gids in col layout matching conditioned source nodes
    std::shared_ptr<Core::LinAlg::Vector<int>> target_nodegids_col_layout_;

    //! source node gids in col layout matching conditioned target nodes
    std::shared_ptr<Core::LinAlg::Vector<int>> source_nodegids_col_layout_;

    //! underlying actual dofset
    std::shared_ptr<DofSetInterface> source_dofset_;

    //! source discretization
    std::shared_ptr<const Core::FE::Discretization> source_dis_;

    //! condition strings defining the coupling
    const std::string coupling_cond_target_;
    const std::string coupling_cond_source_;

    /// filled flag
    bool filled_;
  };
}  // namespace Core::DOFSets


FOUR_C_NAMESPACE_CLOSE

#endif

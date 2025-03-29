// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FEM_DOFSET_DEFINEDMAPPING_WRAPPER_HPP
#define FOUR_C_FEM_DOFSET_DEFINEDMAPPING_WRAPPER_HPP

#include "4C_config.hpp"

#include "4C_fem_dofset_base.hpp"
#include "4C_fem_general_node.hpp"

#include <set>


FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::LinAlg
{
  template <typename T>
  class Vector;
}

namespace Core::DOFSets
{
  /*!
  \brief A dofset that does not rely on same GID/LID numbers but uses
         a defined node mapping instead

  \warning not implemented for element DOFs

  This dofset was originally written to handle heterogeneous reactions :
  \ref ScaTra::HeterogeneousReactionStrategy::HeterogeneousReactionStrategy .
  This means reactions between volume-bound scalars and surface-bound scalars in the
  context of scalar transport simulations.
  It is also used in solid-scatra interaction \ref SSI::SSI_Base::SSI_Base .


  <h3>assign_degrees_of_freedom</h3>

  The important work is done in \ref assign_degrees_of_freedom . If you want to use
  or extend this class, please be aware of the documentation of this method. Also
  have in mind the needs of those models and methods in 4C this class was originally
  intended to support (see above).


  <h3>Usage</h3>

  After construction the newly created dofset has to be added to the target discretization.
  The target discretization then has to be filled. During \ref fill_complete() the method
  \ref assign_degrees_of_freedom is called.

  One example for the proper use of this class:

  \code

    std::shared_ptr<Core::DOFSets::DofSet> newsourcedofset_for_target_dis =
        Teuchos::rcp(new
  Core::DOFSets::DofSetMappedProxy(sourcedis->GetDofSetProxy(),sourcedis,"CouplingSourceToTarget",setofcouplingids));

    target_dis->AddDofSet(newsourcedofset_for_target_dis);

    target_dis->fill_complete();

  \endcode

  See also the example input file 'immersed_cell_biochemo_mechano_pureProtrusion_h8' in
  the 4C 'Input' folder.



  */
  class DofSetDefinedMappingWrapper : public DofSetBase
  {
   public:
    /*! \brief Standard Constructor

    \param dofset       (in) : dofset proxy of source discretization \ref sourcedis
    \param sourcedis    (in) : source discretization for the new dofset
    \param couplingcond (in) : condition to be coupled
    \param condids      (in) : set of condition ids to merge into one dofset    */
    DofSetDefinedMappingWrapper(std::shared_ptr<DofSetInterface> dofset,
        std::shared_ptr<const Core::FE::Discretization> sourcedis, const std::string& couplingcond,
        const std::set<int> condids);

    //! destructor
    ~DofSetDefinedMappingWrapper() override;


    /*! \brief Assign dof numbers using all elements and nodes of the discretization.

           This method is called during \ref fill_complete() on the target
           discretization.


          <h3>Usage</h3>

          An important feature of this dofset proxy is, that portions of source and
          target discretizations, which hare supposed to be coupled by this dofset do
          not necessarily need to have the same number of nodes, as long as the mapping
          is still unique. We give an example to show what this means:

          Imagine, we have given the following conditions in our .dat-file:

          -----DESIGN SSI COUPLING SOLIDTOSCATRA VOL CONDITIONS
          // scatra volume matching to struct volume
          E 2 - coupling_id 1
          // struct volume matching to scatra volume
          E 1 - coupling_id 1
          ----DESIGN SSI COUPLING SOLIDTOSCATRA SURF CONDITIONS
         // separate surface discretization matching struct volume boundary
         E 1 - coupling_id 2
         // struct volume boundary
         E 2 - coupling_id 2

         We have two matching volume discretizations for structure and scatra, and
         we have a scatra surface discretization matching the boundary of the struct
         volume. Thus, we have more scatra nodes, than structure nodes, because we
         have NumNodeScatraVolume + NumNodeScatraSurface scatra nodes, but only
         NumNodeStructVolume (= NumNodeScatraVolume) structure nodes.
         However, since we are matching, each scatra node can be mapped uniquiely to
         a structure node. Note that we could not map each structure node uniquely
         to a scatra node because at the boundary two scatra nodes reside at the same
         location (volume + surface)!
         So in the first case, if we want to add the structure dofset to the scatradis,
         the structure is the source and the scatra is the target.
         We can now create the dofset proxy from the struct dis via

         \code
          std::shared_ptr<Core::DOFSets::DofSet> newdofset =
              Teuchos::rcp(new
       Core::DOFSets::DofSetMappedProxy(structdis->GetDofSetProxy(), structdis,
                                                      "SSICouplingSolidToScatra",
                                                      couplingids));
         \endcode

         The std::set 'couplingids' contains the ids from both, VOL and SURF conditions
         given above. This way, one unique struct dofset for all scatra nodes is created.    */
    int assign_degrees_of_freedom(
        const Core::FE::Discretization& dis, const unsigned dspos, const int start) override;

    /// reset all internal variables
    void reset() override;

    //! @name Proxy management
    /// Proxies need to know about changes to the DofSet.

    /// our original DofSet dies
    void disconnect(DofSetInterface* dofset) override;

    //@}

    /// Returns true if filled
    bool filled() const override { return filled_ and sourcedofset_->filled(); };


    //! @name Access methods

    /// Get degree of freedom row map
    const Core::LinAlg::Map* dof_row_map() const override { return sourcedofset_->dof_row_map(); };

    /// Get degree of freedom column map
    const Core::LinAlg::Map* dof_col_map() const override { return sourcedofset_->dof_col_map(); };

    /// Get number of dofs for given node
    int num_dof(
        const Core::Nodes::Node* node  ///< node, for which you want to know the number of dofs
    ) const override
    {
      const Core::Nodes::Node* sourcenode = get_source_node(node->lid());
      if (sourcenode == nullptr)
        return sourcedofset_->num_dof(node);
      else
        return sourcedofset_->num_dof(sourcenode);
    }

    /// Get number of dofs for given element
    int num_dof(const Core::Elements::Element*
            element  ///< element, for which you want to know the number of dofs
    ) const override
    {
      // element dofs not yet supported
      return 0;
    }

    /// get number of nodal dofs
    int num_dof_per_node(
        const Core::Nodes::Node& node  ///< node, for which you want to know the number of dofs
    ) const override
    {
      const Core::Nodes::Node* sourcenode = get_source_node(node.lid());
      if (sourcenode == nullptr)
        return sourcedofset_->num_dof_per_node(node);
      else
        return sourcedofset_->num_dof_per_node(*sourcenode);
    }

    /// Get the gid of all dofs of a node
    std::vector<int> dof(
        const Core::Nodes::Node* node  ///< node, for which you want to know the number of dofs
    ) const override
    {
      const Core::Nodes::Node* sourcenode = get_source_node(node->lid());
      if (sourcenode == nullptr)
        return sourcedofset_->dof(node);
      else
        return sourcedofset_->dof(sourcenode);
    }

    /// Get the gid of all dofs of a node
    void dof(std::vector<int>& dof,     ///< vector of dof gids (to be filled)
        const Core::Nodes::Node* node,  ///< node, for which you want the dof positions
        unsigned nodaldofset  ///< number of nodal dof set of the node (currently !=0 only for XFEM)
    ) const override
    {
      const Core::Nodes::Node* sourcenode = get_source_node(node->lid());
      if (sourcenode == nullptr)
        sourcedofset_->dof(dof, node, nodaldofset);
      else
        sourcedofset_->dof(dof, sourcenode, nodaldofset);
    };

    /// Get the gid of all dofs of a element
    std::vector<int> dof(const Core::Elements::Element* element) const override
    {
      return std::vector<int>();
    }

    /// Get the gid of a dof for given node
    int dof(const Core::Nodes::Node* node, int dof) const override
    {
      const Core::Nodes::Node* sourcenode = get_source_node(node->lid());
      if (sourcenode == nullptr)
        return sourcedofset_->dof(node, dof);
      else
        return sourcedofset_->dof(sourcenode, dof);
    }

    /// Get the gid of a dof for given element
    int dof(const Core::Elements::Element* element, int dof) const override
    {
      // element dofs not yet supported
      return 0;
    }

    /// Get the gid of all dofs of a node
    void dof(const Core::Nodes::Node* node, std::vector<int>& lm) const override
    {
      const Core::Nodes::Node* sourcenode = get_source_node(node->lid());
      if (sourcenode == nullptr)
        return sourcedofset_->dof(node, lm);
      else
        return sourcedofset_->dof(sourcenode, lm);
    }

    /// Get the gid of all dofs of a node
    void dof(const Core::Nodes::Node* node,  ///< node, for which you want the dof positions
        const unsigned startindex,  ///< first index of vector at which will be written to end
        std::vector<int>& lm        ///< already allocated vector to be filled with dof positions
    ) const override
    {
      const Core::Nodes::Node* sourcenode = get_source_node(node->lid());
      if (sourcenode == nullptr)
        sourcedofset_->dof(node, startindex, lm);
      else
        sourcedofset_->dof(sourcenode, startindex, lm);
    }

    /// Get the gid of all dofs of a element
    void dof(const Core::Elements::Element* element, std::vector<int>& lm) const override
    {
      // element dofs not yet supported
      return;
    }

    /// Get the GIDs of the first DOFs of a node of which the associated element is interested in
    void dof(const Core::Elements::Element*
                 element,  ///< element which provides its expected number of DOFs per node
        const Core::Nodes::Node* node,  ///< node, for which you want the DOF positions
        std::vector<int>& lm  ///< already allocated vector to be filled with DOF positions
    ) const override
    {
      const Core::Nodes::Node* sourcenode = get_source_node(node->lid());
      if (sourcenode == nullptr)
        sourcedofset_->dof(element, node, lm);
      else
        sourcedofset_->dof(element, sourcenode, lm);
    }

    /// Get maximum GID of degree of freedom row map
    int max_all_gid() const override { return sourcedofset_->max_all_gid(); };

    /// Get minimum GID of degree of freedom row map
    int min_all_gid() const override { return sourcedofset_->min_all_gid(); };

    /// Get Max of all GID assigned in the DofSets in front of current one in the list
    /// #static_dofsets_
    int max_gi_din_list(MPI_Comm comm) const override
    {
      return sourcedofset_->max_gi_din_list(comm);
    };

    /// are the dof maps already initialized?
    bool initialized() const override { return sourcedofset_->initialized(); };

    /// Get Number of Global Elements of degree of freedom row map
    int num_global_elements() const override { return sourcedofset_->num_global_elements(); };

   private:
    //! get corresponding source node from source discretization
    const Core::Nodes::Node* get_source_node(int targetLid) const;

    // dofset
    std::shared_ptr<DofSetInterface> sourcedofset_;

    //! map containing the mapping of the target node GID to the corresponding source node GID
    //! (value)
    std::shared_ptr<Core::LinAlg::Vector<int>> targetlidtosourcegidmapping_;

    //! source discretization
    std::shared_ptr<const Core::FE::Discretization> sourcedis_;

    //! condition string defining the coupling
    const std::string couplingcond_;

    //! ID of condition the dofset is build from
    const std::set<int> condids_;

    /// filled flag
    bool filled_;
  };  // DofSetDefinedMappingWrapper
}  // namespace Core::DOFSets



FOUR_C_NAMESPACE_CLOSE

#endif

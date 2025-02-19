// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_XFEM_XFLUID_TIMEINT_HPP
#define FOUR_C_XFEM_XFLUID_TIMEINT_HPP


#include "4C_config.hpp"

#include "4C_cut_node.hpp"
#include "4C_cut_utils.hpp"
#include "4C_inpar_xfem.hpp"
#include "4C_linalg_fixedsizematrix.hpp"

class Epetra_Map;

FOUR_C_NAMESPACE_OPEN

namespace Core::Communication
{
  class PackBuffer;
}

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::Nodes
{
  class Node;
}

namespace Core::Elements
{
  class Element;
}



namespace Cut
{
  class CutWizard;
  class SideHandle;
}  // namespace Cut


namespace Core::LinAlg
{
  class MapExtractor;
}

namespace XFEM
{
  class XFEMDofSet;
  class ConditionManager;

  /*!
  \brief this class is the basic TIMEINT class for the projection, adaption or
         something else in XFEM-problems between consecutive time steps
   */
  class XFluidTimeInt
  {
   public:
    //! constructor
    explicit XFluidTimeInt(
        const bool is_newton_increment_transfer,  /// monolithic newton increment transfer or time
                                                  /// step transfer?
        const std::shared_ptr<Core::FE::Discretization>& dis,              /// discretization
        const std::shared_ptr<XFEM::ConditionManager>& condition_manager,  /// condition manager
        const std::shared_ptr<Cut::CutWizard>& wizard_old,                 /// cut wizard at t^n
        const std::shared_ptr<Cut::CutWizard>& wizard_new,                 /// cut wizard at t^(n+1)
        const std::shared_ptr<XFEM::XFEMDofSet>& dofset_old,               /// XFEM dofset at t^n
        const std::shared_ptr<XFEM::XFEMDofSet>& dofset_new,           /// XFEM dofset at t^(n+1)
        const Inpar::XFEM::XFluidTimeIntScheme xfluid_timintapproach,  /// xfluid_timintapproch
        std::map<int, std::vector<Inpar::XFEM::XFluidTimeInt>>&
            node_to_reconstr_method,  /// reconstruction map for nodes and its dofsets
        std::map<Inpar::XFEM::XFluidTimeInt, std::map<int, std::set<int>>>&
            reconstr_method_to_node,  /// inverse reconstruction map for nodes and its dofsets
        const int step,               /// global time step
        const bool xfluid_timint_check_interfacetips,      /// check interfacetips?
        const bool xfluid_timint_check_sliding_on_surface  /// check sliding on surface?
    );

    /// set and print reconstruction status for nodes
    void set_and_print_status(const bool screenout);


    /// transfer standard and ghost dofs to new map as far as possible and mark dofs for
    /// reconstruction
    void transfer_dofs_to_new_map(
        const std::vector<std::shared_ptr<const Core::LinAlg::Vector<double>>>&
            oldRowStateVectors,  /// row map based vectors w.r.t old interface position
        const std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>>&
            newRowStateVectors,  /// row map based vectors w.r.t new interface position
        const std::shared_ptr<std::set<int>>
            dbcgids  /// set of dof gids that must not be changed by ghost penalty reconstruction
    );

    /// transfer standard and ghost dofs to new map as far as possible and mark dofs for
    /// reconstruction for given vector of node gids
    void transfer_dofs_to_new_map(
        const std::vector<std::shared_ptr<const Core::LinAlg::Vector<double>>>&
            oldRowStateVectors,  /// row map based vectors w.r.t old interface position
        const std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>>&
            newRowStateVectors,  /// row map based vectors w.r.t new interface position
        const std::shared_ptr<std::set<int>>
            dbcgids,  /// set of dof gids that must not be changed by ghost penalty reconstruction
        const std::vector<int>& node_gids  /// vector of node gids
    );

    /// get for each type of reconstruction method the number of nodes for that this method has to
    /// be applied on this proc
    std::map<Inpar::XFEM::XFluidTimeInt, int>& get_reconstr_counts() { return reconstr_counts_; };

    /// get for each type of reconstruction method the node ids with corresponding dof its for
    /// that this method has to be applied on this proc
    std::map<int, std::set<int>>& get_node_to_dof_map_for_reconstr(
        Inpar::XFEM::XFluidTimeInt reconstr);

    /// get permutation map for ghost dofs
    std::shared_ptr<std::map<int, int>> get_permutation_map() { return permutation_map_; };

   private:
    /// transfer standard and ghost dofs to new map as far as possible and mark dofs for
    /// reconstruction for a given node gid
    void transfer_nodal_dofs_to_new_map(
        const std::vector<std::shared_ptr<const Core::LinAlg::Vector<double>>>&
            oldRowStateVectors,  /// row map based vectors w.r.t old interface position
        const std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>>&
            newRowStateVectors,  /// row map based vectors w.r.t new interface position
        const std::shared_ptr<std::set<int>>
            dbcgids,  /// set of dof gids that must not be changed by ghost penalty reconstruction
        int gid       /// nodal gid
    );

    /// returns matching std::string for each reconstruction method
    std::string map_method_enum_to_string(const enum Inpar::XFEM::XFluidTimeInt term);

    /// all surrounding elements non-intersected?
    bool non_intersected_elements(Core::Nodes::Node* n, Cut::CutWizard& wizard);

    /// find all ghost dofsets around this node and its std-dofset
    void find_surrounding_ghost_dofsets(
        std::map<int, std::set<int>>&
            ghostDofsets,  /// surrounding ghost dofsets to be filled, map of ghost nodes and
                           /// corresponding ghost nds index w.r.t given std nodal dofset
        const Core::Nodes::Node* node,  /// node
        const int nds_new  /// dofset of node used for finding the surrounding ghost dofs
    );

    /// copy dofs from old vectors to new vector for all row vectors
    void copy_dofs(const Core::Nodes::Node* node,  /// drt node
        const int nds_new,                         /// nodal dofset at t^(n+1)
        const int nds_old,                         /// nodal dofset at t^n
        const Inpar::XFEM::XFluidTimeInt method,   /// reconstruction method
        const std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>>&
            newRowStateVectors,  /// row map based state vectors at t^(n+1)
        const std::vector<std::shared_ptr<const Core::LinAlg::Vector<double>>>&
            oldRowStateVectors,                       /// row map based state vectors at t^n
        const std::shared_ptr<std::set<int>> dbcgids  /// set of DBC global ids
    );

    /// mark nodal dofs of vector w.r.t new interface position for reconstruction
    void mark_dofs(const Core::Nodes::Node* node,  /// drt node
        const int nds_new,                         /// nodal dofset at t^(n+1)
        const std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>>&
            newRowStateVectors,                   /// row map based state vectors at t^(n+1)
        const Inpar::XFEM::XFluidTimeInt method,  /// reconstruction method
        const std::shared_ptr<std::set<int>>
            dbcgids  /// set of dof gids that must not be changed by ghost penalty reconstruction
    );

    /// mark one specific nodal dofset with used for export
    void mark_dofs_for_export(const int nid,  /// node id
        const int dofset,                     /// ghost dofset number
        const Inpar::XFEM::XFluidTimeInt
            method  /// reconstruction method used for marking the nodal dofset
    );

    /// set the reconstruction method for current nodal dofset, return if set
    bool set_reconstr_method(const Core::Nodes::Node* node,  /// drt node
        const int nds_new,                                   /// nodal dofset at t^(n+1)
        const Inpar::XFEM::XFluidTimeInt method              /// which type of reconstruction method
    );

    /// identify cellsets at time t^n with cellsets at time t^(n+1)
    int identify_old_sets(const Cut::Node* n_old,  /// node w.r.t to old wizard
        const Cut::Node* n_new,                    /// node w.r.t to new wizard
        const std::vector<std::shared_ptr<Cut::NodalDofSet>>&
            dof_cellsets_old,                 /// all dofcellsets at t^n
        const Cut::NodalDofSet* cell_set_new  /// dofcellset at t^(n+1) which has to be identified
    );

    /// special check if the node slides along the cut surface
    bool special_check_sliding_on_surface(bool& changed_side,
        const Cut::Node* n_old,  /// node w.r.t to old wizard
        const Cut::Node* n_new   /// node w.r.t to new wizard
    );

    /// check if the node has changed the side w.r.t identified sides at t^n and t^(n+1), return
    /// if check was successful
    bool special_check_interface_tips(bool& changed_side,  /// did the node change the side ?
        std::vector<int>& identified_sides,                /// side Id of identified side
        const Cut::Node* n_old,                            /// node w.r.t to old wizard
        const Cut::Node* n_new                             /// node w.r.t to new wizard
    );

    /// special check for level-set based interface tips, note: currently empty, can be filled if
    /// necessary
    bool special_check_interface_tips_levelset(
        bool& changed_side  /// did the node change the side ?
    );

    /// special check for mesh based interface tips
    bool special_check_interface_tips_space_time(
        bool& changed_side,  /// did the node change the side ?
        Core::Elements::Element* side, const int coup_sid,
        const Core::LinAlg::Matrix<3, 1>& n_coord  /// node coordinates
    );

    /// check if the node is within the space time side
    template <Core::FE::CellType side_distype,
        Core::FE::CellType space_time_distype>
    bool within_space_time_side(bool& within_space_time_side,  /// within the space time side
        Core::Elements::Element* side, const int coup_sid,
        const Core::LinAlg::Matrix<3, 1>& n_coord  /// node coordinates
    );

    /// check the volume of the space time side, distorted space-time side ?
    template <Core::FE::CellType space_time_distype, const int numnode_space_time>
    bool check_st_side_volume(const Core::LinAlg::Matrix<3, numnode_space_time>& xyze_st);

    /// export data about reconstruction method to neighbor proc and receive data from previous
    /// proc
    void export_methods(
        const std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>>&
            newRowStateVectors,  /// row map based vectors w.r.t new interface position
        const std::shared_ptr<std::set<int>>
            dbcgids  /// set of dof gids that must not be changed by ghost penalty reconstruction
    );

    /*!
    \brief Basic function sending data to destination and receiving data from source
     */
    void send_data(Core::Communication::PackBuffer& dataSend,  //!< pack buffer
        int& dest,                                             //!< destination proc
        int& source,                                           //!< source proc
        std::vector<char>& dataRecv                            //!< received data
    ) const;

    const bool is_newton_increment_transfer_;  /// monolithic newton increment transfer or time
                                               /// step transfer?

    std::shared_ptr<Core::FE::Discretization> dis_;  /// background  discretization

    std::shared_ptr<XFEM::ConditionManager> condition_manager_;  /// condition manager

    std::shared_ptr<Cut::CutWizard> wizard_old_;  /// old cut wizard w.r.t old interface position
    std::shared_ptr<Cut::CutWizard> wizard_new_;  /// new cut wizard w.r.t new interface position

    std::shared_ptr<XFEM::XFEMDofSet> dofset_old_;  /// old XFEM dofset w.r.t old interface position
    std::shared_ptr<XFEM::XFEMDofSet> dofset_new_;  /// new XFEM dofset w.r.t new interface position

    // current processor id and number of procs
    int myrank_;
    int numproc_;

    Inpar::XFEM::XFluidTimeIntScheme timeint_scheme_;  /// which type of time integration scheme
                                                       /// shall be used for reconstructing values

    std::map<int, std::vector<Inpar::XFEM::XFluidTimeInt>>&
        node_to_reconstr_method_;  /// map of nodeIds and reconstruction method for all nodal
                                   /// dofsets

    std::map<Inpar::XFEM::XFluidTimeInt, std::map<int, std::set<int>>>&
        reconstr_method_to_node_;  /// map of reconstruction method to map of node-id and nodal
                                   /// dofset-id (inverse to node_to_reconstr_method)

    std::map<Inpar::XFEM::XFluidTimeInt, int>
        reconstr_counts_;  /// counts the number of dofsets that have to be reconstructed using
                           /// each method

    std::map<int, std::map<int, int>>
        dofset_marker_export_;  /// std::map<nid, std::map<dofset_number,method> > that contains
                                /// marker for reconstruction method of single nodal dofsets used
                                /// for export in parallel

    // global timestep
    int step_;

    // name check interfacetips in timeintegration
    bool xfluid_timint_check_interfacetips_;

    //! @name check sliding on surface in timeintegration
    bool xfluid_timint_check_sliding_on_surface_;

    std::shared_ptr<std::map<int, int>> permutation_map_;
  };

}  // namespace XFEM

FOUR_C_NAMESPACE_CLOSE

#endif

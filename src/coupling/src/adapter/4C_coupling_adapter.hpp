// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_COUPLING_ADAPTER_HPP
#define FOUR_C_COUPLING_ADAPTER_HPP

#include "4C_config.hpp"

#include "4C_coupling_adapter_base.hpp"
#include "4C_linalg_fevector.hpp"
#include "4C_linalg_map.hpp"

#include <map>
#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  class SparseMatrix;

}  // namespace Core::LinAlg

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Coupling::Adapter
{
  /*! \class Coupling
   *  \brief Management of coupling between two (matching) sets of nodes from
   *  two discretizations
   *
   *  Interface coupled problems with matching meshes need to transfer
   *  results between their interfaces. That is values which belong to
   *  the dofs of one side have to be accessed by the other side as
   *  well. In a parallel setting, of course, these sides do not in
   *  general reside on the same processors.
   *
   *  This class handles the transfer of dof values at the interface
   *  (Core::LinAlg::Vectors build on the interface dof map of either side) to
   *  the other side. To distinguish both sides lets speak of target and
   *  source, even though no side really dominates the other. On the
   *  contrary we provide perfect symmetry once the setup is done.
   *
   *  The idea is simple: We have a target dof map that describes the
   *  distribution of the target's interface dofs. And we have a source
   *  dof map that describes the distribution of the source's interface
   *  dofs. Both maps, however, live in their respective
   *  communicator. We cannot transfer data between them. Furthermore
   *  the dof gids are totally independent of each other.
   *
   *  So we build a mapping between the nodes from both meshes during
   *  the setup phase. This establishes the connection. From that
   *  connection we deduce (in a somewhat painful way) a permuted
   *  target dof map and a permuted source dof map. These permuted
   *  maps are bound to have the same layout as the normal maps from the
   *  other side. So we can exchange dof values between fields by simply
   *  copying from a normal Core::LinAlg::Vector<double> to the permuted
   *  Core::LinAlg::Vector<double> from the other side without actually looking at the
   *  respective maps. Afterwards the communication happens within one
   *  field in the usual fashion. So the transfer functions
   *  target_to_source() and source_to_target() are quite simple. The hard
   *  work happens (once) during setup.
   *

   */
  class Coupling : public CouplingBase
  {
   public:
    /// empty constructor
    Coupling();

    /** \name Setup */
    //@{

    /// setup coupling of nodes marked with condition
    /*!
      Setup the whole thing. Find matching nodes via octtree and build
      appropriate dof maps.

      first variant will couple the first numdof DOFs of both discretizations

      second variant accepts two vectors to couple the DOFs specified in these
      vectors, e.g. target_dofs = [0 3 5] and source_dofs = [1 2 4] will couple
      DOF 0 (target) with DOF 1 source, DOF 3 (target) with DOF 2 source, etc.

     */
    void setup_condition_coupling(const Core::FE::Discretization& target_dis,
        std::shared_ptr<const Core::LinAlg::Map> target_cond_map,
        const Core::FE::Discretization& source_dis,
        std::shared_ptr<const Core::LinAlg::Map> source_cond_map, const std::string& condname,
        const int numdof, bool matchall = true, const int target_dofset_number = 0,
        const int source_dofset_number = 0);

    void setup_condition_coupling(const Core::FE::Discretization& target_dis,
        std::shared_ptr<const Core::LinAlg::Map> target_cond_map,
        const Core::FE::Discretization& source_dis,
        std::shared_ptr<const Core::LinAlg::Map> source_cond_map, const std::string& condname,
        const std::vector<int>& target_dofs, const std::vector<int>& source_dofs,
        bool matchall = true, const int target_dofset_number = 0,
        const int source_dofset_number = 0);

    /*! \brief Setup coupling of given nodes
     *
     *  Setup the whole thing. Find matching nodes via octtree and build
     *  appropriate dof maps.
     *
     *  \note All source nodes are required to find a match. In contrast
     *  target nodes do not need to have a match. So it is legal to hand
     *  in more target nodes that source nodes as long as there is one
     *  target node for each source node.
     *
     *  We need some way to guess the tolerance for the octree. It must not be
     *  too small, otherwise we won't find matching nodes. Too large a tolerance
     *  will not hurt that much. It just means we will have to test more nodes.
     *  (But it could hurt, if the geometry is very small. It should be
     *   around the size of the smallest element)
     *
     *  \param target_dis   (i) target side mesh
     *  \param source_dis    (i) source side mesh
     *  \param target_nodes (i) gids of nodes on target side to be coupled
     *  \param source_nodes  (i) gids of nodes on source side to be coupled
     *  \param target_dofs  (i) vector with DOF ids of target side to be coupled
     *  \param source_dofs   (i) vector with DOF ids of side side to be coupled
     *  \param matchall    (i) flag indicating matching source and target nodes
     *  \param tolerance   (i) tolerance for octree for node matching
     */
    void setup_coupling(const Core::FE::Discretization& target_dis,
        const Core::FE::Discretization& source_dis, const std::vector<int>& target_nodes,
        const std::vector<int>& source_nodes, const std::vector<int>& target_dofs,
        const std::vector<int>& source_dofs, const bool matchall = true,
        const double tolerance = 1.e-3, const int target_dofset_number = 0,
        const int source_dofset_number = 0);

    /*! \brief Setup coupling of given nodes
     *
     *  only difference to the variant above is that the first numdof
     *  DOFs on both source and target side will be coupled
     *
     *  \param numdof      (i) number of dofs per node to be coupled
     */
    void setup_coupling(const Core::FE::Discretization& target_dis,
        const Core::FE::Discretization& source_dis, const std::vector<int>& target_nodes,
        const std::vector<int>& source_nodes, const int numdof, const bool matchall = true,
        const double tolerance = 1.e-3, const int target_dofset_number = 0,
        const int source_dofset_number = 0);

    /*! \brief Setup coupling of given nodes
     *
     *  Setup the whole thing. Find matching nodes via octtree and build
     *  appropriate dof maps.
     *
     *  \note All source nodes are required to find a match. In contrast
     *  target nodes do not need to have a match. So it is legal to hand
     *  in more target nodes that source nodes as long as there is one
     *  target node for each source node.
     *
     *  We need some way to guess the tolerance for the octree. It must not be
     *  too small, otherwise we won't find matching nodes. Too large a tolerance
     *  will not hurt that much. It just means we will have to test more nodes.
     *
     *  \param target_dis   (i) target side mesh
     *  \param source_dis    (i) source side mesh
     *  \param target_nodes (i) gids of nodes on target side to be coupled
     *  \param source_nodes  (i) gids of nodes on source side to be coupled
     *  \param numdof      (i) number of dofs per node to be coupled
     *  \param matchall    (i) flag indicating matching source and target nodes
     *  \param tolerance   (i) tolerance for octree for node matching
     */
    void setup_coupling(const Core::FE::Discretization& target_dis,
        const Core::FE::Discretization& source_dis, const Core::LinAlg::Map& target_nodes,
        const Core::LinAlg::Map& source_nodes, const int numdof, const bool matchall = true,
        const double tolerance = 1.e-3, const int target_dofset_number = 0,
        const int source_dofset_number = 0);

    /*! \brief Setup coupling of given nodes and node maps
     *
     *  Setup the whole thing. Build appropriate dof maps (for given, matching nodes on both
     * sides).
     *
     *  \note All target and source nodes are required to match.
     *  The permuted source node map is given as input, thus no search tree to find matching nodes
     * is used. Only the corresponding dof maps are build here (+ some checks). This setup method
     * can be used, if one discretization was cloned from the others discretization, and thus
     * matching node GIDs are assured (for Fluid-ALE or poro, for instance). In this case, this
     * setup gives the same results as the standard version above, but without the need to perform
     * the matching node search.
     *
     *  \param target_dis         (i) target side mesh
     *  \param source_dis          (i) source side mesh
     *  \param target_node_map     (i) gids of nodes on target side to be coupled
     *  \param source_node_map      (i) gids of nodes on source side to be coupled
     *  \param permuted_source_node_map  (i) permuted node map of the source side
     *  \param numdof            (i) number of dofs per node to be coupled
     */
    void setup_coupling(const Core::FE::Discretization& target_dis,
        const Core::FE::Discretization& source_dis, const Core::LinAlg::Map& target_node_map,
        const Core::LinAlg::Map& source_node_map, const Core::LinAlg::Map& permuted_source_node_map,
        const int numdof);

    /*! \brief Setup coupling of given target and source discretization
     *
     *  Setup the whole thing for matching dofs on both sides
     *
     *  \note All target and source dofs are required to match.
     *  This setup method can be used, if one discretization was cloned from the other
     * discretization, and thus matching dof GIDs are assured. All dofs of all nodes are coupled
     * this way. In this case, this setup gives the same results as the standard version above,
     * but neither a matching node search nor building dof maps is necessary.
     *
     *  \param target_dis         (i) target side mesh
     *  \param source_dis          (i) source side mesh
     */
    void setup_coupling(
        const Core::FE::Discretization& target_dis, const Core::FE::Discretization& source_dis);

    /*! \brief Setup coupling of given nodes
     *
     *  Setup the whole thing. Find matching nodes via octtree and build
     *  appropriate dof maps.
     *
     *  \note All source nodes are required to find a match. In contrast
     *  target nodes do not need to have a match.
     *
     *  Nodes are clustered, such that coupling is performed between clusters
     *  (e.g. target_nodes_vec.at(0) <-> source_nodes_vec.at(0), target_nodes_vec.at(1) <->
     * source_nodes_vec.at(1), ...)
     *
     *  \param target_dis       (i) target side mesh
     *  \param source_dis        (i) source side mesh
     *  \param target_nodes_vec (i) gids of nodes on target side to be coupled
     *  \param source_nodes_vec  (i) gids of nodes on source side to be coupled
     *  \param numdof          (i) number of dofs per node to be coupled
     *  \param matchall        (i) flag indicating matching source and target nodes
     *  \param tolerance       (i) tolerance for octree for node matching
     *  \param target_dofset_number      (i) dofset ID of target side
     *  \param source_dofset_number       (i) dofset ID of source side
     */
    void setup_coupling(const Core::FE::Discretization& target_dis,
        const Core::FE::Discretization& source_dis,
        const std::vector<std::vector<int>>& target_nodes_vec,
        const std::vector<std::vector<int>>& source_nodes_vec, const int numdof,
        const bool matchall = true, const double tolerance = 1.0e-3,
        const int target_dofset_number = 0, const int source_dofset_number = 0);

    /*! Setup coupling based on dof maps. This can be used if no search is required anymore and
     * all dof maps are known already
     *
     * \param source_dof_map       (i) source side dof map
     * \param permuted_source_dof_map   (i) permuted dofs on source side to match dofs on target
     * side
     * \param target_dof_map      (i) target side dof map
     * \param permuted_target_dof_map  (i) permuted dofs on target side to match dofs on source side
     */
    void setup_coupling(std::shared_ptr<const Core::LinAlg::Map> source_dof_map,
        std::shared_ptr<const Core::LinAlg::Map> permuted_source_dof_map,
        std::shared_ptr<const Core::LinAlg::Map> target_dof_map,
        std::shared_ptr<const Core::LinAlg::Map> permuted_target_dof_map);
    //@}

    /** \name Conversion between target and source */
    //@{
    /// There are different versions to satisfy all needs. The basic
    /// idea is the same for all of them.

    /// transfer a dof vector from target to source
    std::shared_ptr<Core::LinAlg::Vector<double>> target_to_source(
        const Core::LinAlg::Vector<double>& tv  ///< target vector (to be transferred)
    ) const override;

    /// transfer a dof vector from source to target
    std::shared_ptr<Core::LinAlg::Vector<double>> source_to_target(
        const Core::LinAlg::Vector<double>& sv  ///< source vector (to be transferred)
    ) const override;

    /// transfer a dof vector from target to source
    std::shared_ptr<Core::LinAlg::FEVector<double>> target_to_source(
        const Core::LinAlg::FEVector<double>& tv  ///< target vector (to be transferred)
    ) const;

    /// transfer a dof vector from source to target
    std::shared_ptr<Core::LinAlg::FEVector<double>> source_to_target(
        const Core::LinAlg::FEVector<double>& sv  ///< source vector (to be transferred)
    ) const;

    /// transfer a dof vector from target to source
    std::shared_ptr<Core::LinAlg::MultiVector<double>> target_to_source(
        const Core::LinAlg::MultiVector<double>& tv  ///< target vector (to be transferred)
    ) const override;

    /// transfer a dof vector from source to target
    std::shared_ptr<Core::LinAlg::MultiVector<double>> source_to_target(
        const Core::LinAlg::MultiVector<double>& sv  ///< source vector (to be transferred)
    ) const override;

    /// transfer a dof vector from target to source
    void target_to_source(
        const Core::LinAlg::MultiVector<double>& tv,  ///< target vector (to be transferred)
        Core::LinAlg::MultiVector<double>& sv         ///< source vector (containing result)
    ) const override;

    /// transfer a dof vector from source to target
    void source_to_target(
        const Core::LinAlg::MultiVector<double>& sv,  ///< source vector (to be transferred)
        Core::LinAlg::MultiVector<double>& tv         ///< target vector (containing result)
    ) const override;

    /// transfer a dof vector from target to source
    void target_to_source(
        const Core::LinAlg::Vector<int>& tv,  ///< target vector (to be transferred)
        Core::LinAlg::Vector<int>& sv         ///< source vector (containing result)
    ) const;

    /// transfer a dof vector from source to target
    void source_to_target(
        const Core::LinAlg::Vector<int>& sv,  ///< source vector (to be transferred)
        Core::LinAlg::Vector<int>& tv         ///< target vector (containing result)
    ) const;

    //@}

    //! @name Access to coupled maps
    //@{

    /// the interface dof map of the target side
    std::shared_ptr<const Core::LinAlg::Map> target_dof_map() const override
    {
      return target_dof_map_;
    }

    /// the interface dof map of the source side
    std::shared_ptr<const Core::LinAlg::Map> source_dof_map() const override
    {
      return source_dof_map_;
    }

    /// the permuted interface dof map of the target side
    std::shared_ptr<const Core::LinAlg::Map> permuted_target_dof_map() const
    {
      return permuted_target_dof_map_;
    }

    /// the permuted interface dof map of the source side
    std::shared_ptr<const Core::LinAlg::Map> permuted_source_dof_map() const
    {
      return permuted_source_dof_map_;
    }

    //@}

    //! @name Matrix transform
    //@{

    /// fill rowmap with target -> source pairs
    void fill_target_to_source_map(std::map<int, int>& rowmap) const;

    /// fill rowmap with source -> target pairs
    void fill_source_to_target_map(std::map<int, int>& rowmap) const;

    /// fill partial target map with gid of partial source map
    std::shared_ptr<Core::LinAlg::Map> source_to_target_map(Core::LinAlg::Map& source);

    /// fill partial source map with gid of partial target map
    std::shared_ptr<Core::LinAlg::Map> target_to_source_map(Core::LinAlg::Map& target);

    //@}

    /// \name Lagrangian coupling helpers

    /// create coupling matrices for Lagrangian coupling conditions
    void setup_coupling_matrices(const Core::LinAlg::Map& shifted_target_map,
        const Core::LinAlg::Map& target_domain_map, const Core::LinAlg::Map& source_domain_map);

    //@}

   protected:
    virtual void build_dof_maps(const Core::FE::Discretization& target_dis,
        const Core::FE::Discretization& source_dis,
        const std::shared_ptr<const Core::LinAlg::Map>& target_node_map,
        const std::shared_ptr<const Core::LinAlg::Map>& source_node_map,
        const std::shared_ptr<const Core::LinAlg::Map>& permuted_target_node_map,
        const std::shared_ptr<const Core::LinAlg::Map>& permuted_source_node_map,
        const std::vector<int>& target_dofs, const std::vector<int>& source_dofs,
        const int target_dofset_number = 0, const int source_dofset_number = 0);

    /*! \brief Helper function for creating vector from numdof
     *
     *  will return just an ascending vector [0, 1, ..., numdof-1]
     *  if called with numdof > 0, otherwise will return [-1],
     *  which is used inside xfield_field_coupling to determine
     *  if base class is called
     */
    virtual std::vector<int> build_dof_vector_from_num_dof(const int numdof);

   private:
    /*! \brief Do the actual matching of the target <-> source pairs
     *
     *  Here the octree is used. Afterwards all not paired target nodes
     *  are removed.
     *
     *  \param target_dis   (i) target side mesh
     *  \param source_dis    (i) source side mesh
     *  \param target_nodes (i/o) all target node gids. on output those that have a match
     *  \param permuted_source_nodes (o) source node gids permuted to match target node gids
     *  \param source_nodes (i) source node gids
     *  \param matchall (i) bool indicating match of all source and target nodes
     *  \param tolerance   (i) tolerance for octree for node matching
     */
    void match_nodes(const Core::FE::Discretization& target_dis,
        const Core::FE::Discretization& source_dis, std::vector<int>& target_nodes,
        std::vector<int>& permuted_source_nodes, const std::vector<int>& source_nodes,
        const bool matchall, const double tolerance);

    /// build source to target permutation and dof all maps
    void finish_coupling(const Core::FE::Discretization& target_dis,
        const Core::FE::Discretization& source_dis,
        std::shared_ptr<Core::LinAlg::Map> target_node_map,
        std::shared_ptr<Core::LinAlg::Map> source_node_map,
        std::shared_ptr<Core::LinAlg::Map> permuted_source_node_map,
        const std::vector<int>& target_dofs, const std::vector<int>& source_dofs,
        const int target_dofset_number = 0, const int source_dofset_number = 0);

    /// build dof maps from node maps
    /*!
      \note It is assumed that the first numdof dofs of each
      node as specified in vector coupled_dofs are of interest.
     */
    void build_dof_maps(const Core::FE::Discretization& dis, const Core::LinAlg::Map& nodemap,
        const Core::LinAlg::Map& permnodemap, std::shared_ptr<const Core::LinAlg::Map>& dofmap,
        std::shared_ptr<const Core::LinAlg::Map>& permdofmap,
        std::shared_ptr<Core::LinAlg::Export>& exporter, const std::vector<int>& coupled_dofs,
        const int nds = 0) const;

   protected:
    /// @name accessors to the private class members for derived classes
    /// @{

    /// access the interface DoF map of the target side
    std::shared_ptr<const Core::LinAlg::Map>& target_dof_map_ptr();

    /// access the permuted interface DoF map of the target side
    std::shared_ptr<const Core::LinAlg::Map>& permuted_target_dof_map_ptr();

    /// access the interface DoF map of the source side
    std::shared_ptr<const Core::LinAlg::Map>& source_dof_map_ptr();

    /// access the permuted interface DoF map of the source side
    std::shared_ptr<const Core::LinAlg::Map>& permuted_source_dof_map_ptr();

    /// access the permuted target dof map to target dof map exporter
    std::shared_ptr<Core::LinAlg::Export>& target_exporter_ptr();

    /// access the permuted source dof map to source dof map exporter
    std::shared_ptr<Core::LinAlg::Export>& source_exporter_ptr();

    /// @}


   private:
    /*! @name Fundamental dof maps
     *
     *  We keep the target and source dof map as well as permuted versions
     *  that match the respective other side.
     */
    //@{

    //! the interface dof map of the target side
    std::shared_ptr<const Core::LinAlg::Map> target_dof_map_;

    //! the permuted interface dof map of the target side
    std::shared_ptr<const Core::LinAlg::Map> permuted_target_dof_map_;

    //! the interface dof map of the source side
    std::shared_ptr<const Core::LinAlg::Map> source_dof_map_;

    //! the permuted interface dof map of the source side
    std::shared_ptr<const Core::LinAlg::Map> permuted_source_dof_map_;

    //@}

    //! @name Communication object
    //@{

    //! permuted target dof map to target dof map exporter
    std::shared_ptr<Core::LinAlg::Export> target_export_;

    //! permuted source dof map to source dof map exporter
    std::shared_ptr<Core::LinAlg::Export> source_export_;

    //@}

    //! @name coupling matrices for Lagrangian multiplier coupling
    //@{

    std::shared_ptr<Core::LinAlg::SparseMatrix> matmm_;
    std::shared_ptr<Core::LinAlg::SparseMatrix> matsm_;
    std::shared_ptr<Core::LinAlg::SparseMatrix> matmm_trans_;
    std::shared_ptr<Core::LinAlg::SparseMatrix> matsm_trans_;

    //@}
  };
}  // namespace Coupling::Adapter

FOUR_C_NAMESPACE_CLOSE

#endif

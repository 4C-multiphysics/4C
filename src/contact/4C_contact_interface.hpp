// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONTACT_INTERFACE_HPP
#define FOUR_C_CONTACT_INTERFACE_HPP

#include "4C_config.hpp"

#include "4C_contact_input.hpp"
#include "4C_coupling_adapter.hpp"
#include "4C_linalg_fevector.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_mortar_interface.hpp"
#include "4C_mortar_strategy_base.hpp"

// forward declarations

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  class SerialDenseMatrix;
}  // namespace Core::LinAlg


namespace Core::Nodes
{
  class Node;
}

namespace Core::LinAlg
{
  class SparseMatrix;
}

namespace CONTACT
{
  class Interface;
  class Node;
  class Element;
  class SelfBinaryTree;

  /*----------------------------------------------------------------------------*/
  /** \brief Contact interface data container
   *
   *  This class is supposed to contain all relevant members for the contact
   *  interfaces. The external storage in this object, instead of the actual
   *  interface class itself, makes it possible to share the interface
   *  data between different interface objects w/o the need of copying them.
   *
   * */
  class InterfaceDataContainer : public Mortar::InterfaceDataContainer
  {
   public:
    /// constructor
    InterfaceDataContainer();

    /// @name Accessors
    /// @{

    inline bool& is_self_contact() { return selfcontact_; }

    [[nodiscard]] inline bool is_self_contact() const { return selfcontact_; }

    inline bool& is_friction() { return friction_; }

    [[nodiscard]] inline bool is_friction() const { return friction_; }

    inline bool& is_non_smooth_contact() { return non_smooth_contact_; }

    [[nodiscard]] inline bool is_non_smooth_contact() const { return non_smooth_contact_; }

    [[nodiscard]] inline bool is_two_half_pass() const { return two_half_pass_; }

    inline bool& is_two_half_pass() { return two_half_pass_; }

    inline enum CONTACT::ConstraintDirection& constraint_direction() { return constr_direction_; }

    [[nodiscard]] inline enum CONTACT::ConstraintDirection constraint_direction() const
    {
      return constr_direction_;
    }

    inline std::shared_ptr<Core::LinAlg::Map>& active_nodes() { return activenodes_; }

    [[nodiscard]] inline const std::shared_ptr<Core::LinAlg::Map>& active_nodes() const
    {
      return activenodes_;
    }

    inline std::shared_ptr<Core::LinAlg::Map>& active_dofs() { return activedofs_; }

    [[nodiscard]] inline std::shared_ptr<const Core::LinAlg::Map> active_dofs() const
    {
      return activedofs_;
    }

    inline std::shared_ptr<Core::LinAlg::Map>& inactive_nodes() { return inactivenodes_; }

    [[nodiscard]] inline const std::shared_ptr<Core::LinAlg::Map>& inactive_nodes() const
    {
      return inactivenodes_;
    }

    inline std::shared_ptr<Core::LinAlg::Map>& inactive_dofs() { return inactivedofs_; }

    [[nodiscard]] inline std::shared_ptr<const Core::LinAlg::Map> inactive_dofs() const
    {
      return inactivedofs_;
    }

    inline std::shared_ptr<Core::LinAlg::Map>& active_n() { return activen_; }

    [[nodiscard]] inline std::shared_ptr<const Core::LinAlg::Map> active_n() const
    {
      return activen_;
    }

    inline std::shared_ptr<Core::LinAlg::Map>& active_t() { return activet_; }

    [[nodiscard]] inline std::shared_ptr<const Core::LinAlg::Map> active_t() const
    {
      return activet_;
    }

    inline std::shared_ptr<Core::LinAlg::Map>& slip_nodes() { return slipnodes_; }

    [[nodiscard]] inline std::shared_ptr<const Core::LinAlg::Map> slip_nodes() const
    {
      return slipnodes_;
    }

    inline std::shared_ptr<Core::LinAlg::Map>& slip_dofs() { return slipdofs_; }

    [[nodiscard]] inline std::shared_ptr<const Core::LinAlg::Map> slip_dofs() const
    {
      return slipdofs_;
    }

    inline std::shared_ptr<Core::LinAlg::Map>& slip_t() { return slipt_; }

    [[nodiscard]] inline std::shared_ptr<const Core::LinAlg::Map> slip_t() const { return slipt_; }

    inline std::shared_ptr<Core::LinAlg::Map>& non_smooth_nodes() { return nonsmoothnodes_; }

    [[nodiscard]] inline std::shared_ptr<const Core::LinAlg::Map> non_smooth_nodes() const
    {
      return nonsmoothnodes_;
    }

    inline std::shared_ptr<Core::LinAlg::Map>& smooth_nodes() { return smoothnodes_; }

    [[nodiscard]] inline std::shared_ptr<const Core::LinAlg::Map> smooth_nodes() const
    {
      return smoothnodes_;
    }

    [[nodiscard]] inline std::shared_ptr<const Core::LinAlg::Map> sdof_vertex_rowmap() const
    {
      return sdof_vertex_rowmap_;
    }

    inline std::shared_ptr<Core::LinAlg::Map>& sdof_vertex_rowmap() { return sdof_vertex_rowmap_; }

    [[nodiscard]] inline std::shared_ptr<const Core::LinAlg::Map> sdof_vertex_colmap() const
    {
      return sdof_vertex_colmap_;
    }

    inline std::shared_ptr<Core::LinAlg::Map>& sdof_vertex_colmap() { return sdof_vertex_colmap_; }

    [[nodiscard]] inline std::shared_ptr<const Core::LinAlg::Map> sdof_edge_rowmap() const
    {
      return sdof_edge_rowmap_;
    }

    inline std::shared_ptr<Core::LinAlg::Map>& sdof_edge_rowmap() { return sdof_edge_rowmap_; }

    [[nodiscard]] inline std::shared_ptr<const Core::LinAlg::Map> sdof_edge_colmap() const
    {
      return sdof_edge_colmap_;
    }

    inline std::shared_ptr<Core::LinAlg::Map>& sdof_edge_colmap() { return sdof_edge_colmap_; }

    [[nodiscard]] inline std::shared_ptr<const Core::LinAlg::Map> sdof_surf_rowmap() const
    {
      return sdof_surf_rowmap_;
    }

    inline std::shared_ptr<Core::LinAlg::Map>& sdof_surf_rowmap() { return sdof_surf_rowmap_; }

    [[nodiscard]] inline std::shared_ptr<const Core::LinAlg::Map> sdof_surf_colmap() const
    {
      return sdof_surf_colmap_;
    }

    inline std::shared_ptr<Core::LinAlg::Map>& sdof_surf_colmap() { return sdof_surf_colmap_; }

    inline std::shared_ptr<Core::LinAlg::Map>& n_extended_ghosting() { return nextendedghosting_; }

    [[nodiscard]] inline std::shared_ptr<const Core::LinAlg::Map> n_extended_ghosting() const
    {
      return nextendedghosting_;
    }

    inline std::shared_ptr<Core::LinAlg::Map>& e_extended_ghosting() { return eextendedghosting_; }

    [[nodiscard]] inline std::shared_ptr<const Core::LinAlg::Map> e_extended_ghosting() const
    {
      return eextendedghosting_;
    }

    inline std::shared_ptr<SelfBinaryTree>& binary_tree_self() { return binarytreeself_; }

    [[nodiscard]] inline std::shared_ptr<const SelfBinaryTree> binary_tree_self() const
    {
      return binarytreeself_;
    }

    inline std::shared_ptr<Core::LinAlg::Vector<double>>& cn_values() { return cn_values_; }

    [[nodiscard]] inline std::shared_ptr<const Core::LinAlg::Vector<double>> cn_values() const
    {
      return cn_values_;
    }

    inline std::shared_ptr<Core::LinAlg::Vector<double>>& ct_values() { return ct_values_; }

    [[nodiscard]] inline std::shared_ptr<const Core::LinAlg::Vector<double>> ct_values() const
    {
      return ct_values_;
    }

    inline int& sm_pairs() { return smpairs_; }

    [[nodiscard]] inline int sm_pairs() const { return smpairs_; }

    inline int& sm_int_pairs() { return smintpairs_; }

    [[nodiscard]] inline int sm_int_pairs() const { return smintpairs_; }

    inline int& int_cells() { return intcells_; }

    [[nodiscard]] inline int int_cells() const { return intcells_; }

    /// @}

   private:
    //! flag indicating if this is a self contact interface
    bool selfcontact_;

    //! flag for frictional contact
    bool friction_;

    //! flag for non-smooth contact algorithm
    bool non_smooth_contact_;

    //! flag for two half pass contact algorithm
    bool two_half_pass_;

    //! direction in which the contact constraints are formulated
    CONTACT::ConstraintDirection constr_direction_;

    //! @name Maps
    //! @{

    //! row map of all active slave nodes
    std::shared_ptr<Core::LinAlg::Map> activenodes_;

    //! row map of all active slave dofs
    std::shared_ptr<Core::LinAlg::Map> activedofs_;

    //! row map of all inactive slave nodes
    std::shared_ptr<Core::LinAlg::Map> inactivenodes_;

    //! row map of all inactive slave dofs
    std::shared_ptr<Core::LinAlg::Map> inactivedofs_;

    //! row map of global N-matrix
    std::shared_ptr<Core::LinAlg::Map> activen_;

    //! row map of global T-matrix
    std::shared_ptr<Core::LinAlg::Map> activet_;

    //! row map of all slip slave nodes
    std::shared_ptr<Core::LinAlg::Map> slipnodes_;

    //! row map of all slip slave dofs
    std::shared_ptr<Core::LinAlg::Map> slipdofs_;

    //! row map of part of T-matrix (slip nodes)
    std::shared_ptr<Core::LinAlg::Map> slipt_;

    //! row map of all nonsmooth slave nodes
    std::shared_ptr<Core::LinAlg::Map> nonsmoothnodes_;

    //! row map of all smooth slave nodes
    std::shared_ptr<Core::LinAlg::Map> smoothnodes_;

    //! row map of all nonsmooth slave nodes
    std::shared_ptr<Core::LinAlg::Map> sdof_vertex_rowmap_;

    //! row map of all smooth slave nodes
    std::shared_ptr<Core::LinAlg::Map> sdof_vertex_colmap_;

    //! row map of all nonsmooth slave nodes
    std::shared_ptr<Core::LinAlg::Map> sdof_edge_rowmap_;

    //! row map of all smooth slave nodes
    std::shared_ptr<Core::LinAlg::Map> sdof_edge_colmap_;

    //! row map of all nonsmooth slave nodes
    std::shared_ptr<Core::LinAlg::Map> sdof_surf_rowmap_;

    //! row map of all smooth slave nodes
    std::shared_ptr<Core::LinAlg::Map> sdof_surf_colmap_;

    std::shared_ptr<Core::LinAlg::Map> nextendedghosting_;
    std::shared_ptr<Core::LinAlg::Map> eextendedghosting_;

    //! @}

    //! binary tree for self contact search
    std::shared_ptr<SelfBinaryTree> binarytreeself_;

    //! cn-values of each node
    std::shared_ptr<Core::LinAlg::Vector<double>> cn_values_;

    //! ct-values of each node
    std::shared_ptr<Core::LinAlg::Vector<double>> ct_values_;

    //! proc local number of slave/master pairs
    int smpairs_;

    //! proc local number of slave/master integration pairs
    int smintpairs_;

    ///< proc local number of integration cells
    int intcells_;

  };  // class CONTACT::InterfaceDataContainer

  /*----------------------------------------------------------------------------*/
  /*!
  \brief One contact interface

  */
  class Interface : public Mortar::Interface
  {
   protected:
    /// constructor ( only for derived classes )
    Interface(const std::shared_ptr<CONTACT::InterfaceDataContainer>& interfaceData_ptr);

   public:
    /** \brief Create a new contact interface object
     *
     *  This method creates first a new interface data object and subsequently
     *  a new interface object.
     *
     *  \param id (in) : unique interface ID
     *  \param comm (in) : communicator object
     *  \param spatialDim (in) : spatial dimension of the problem
     *  \param icontact (in) : global contact parameter-list
     *  \param selfcontact (in): Boolean flag to indicate self-contact
     *
     *  */
    static std::shared_ptr<Interface> create(const int id, MPI_Comm comm, const int spatialDim,
        const Teuchos::ParameterList& icontact, const bool selfcontact);

    /*!
    \brief Standard constructor creating empty contact interface

    This initializes the employed shape function set for Lagrange multipliers
    to a specific setting. Throughout the evaluation process, this set will be employed
    for the field of Lagrange multipliers.

    \param interfaceData_ptr (in): data container
    \param id (in): Unique interface id
    \param comm (in): A communicator object
    \param spatialDim (in): spatial dimension of the problem
    \param icontact (in): Global contact parameter list
    \param selfcontact (in): Flag for self contact status

    */
    Interface(const std::shared_ptr<Mortar::InterfaceDataContainer>& interfaceData_ptr,
        const int id, MPI_Comm comm, const int spatialDim, const Teuchos::ParameterList& icontact,
        bool selfcontact);

    // don't want = operator and cctor
    Interface operator=(const Interface& old) = delete;
    Interface(const Interface& old) = delete;

    /*!
    \brief Print this Interface

    \param[in] os Output stream used for printing
    */
    void print(std::ostream& os) const final;

    //! @name Access methods

    /*!
    \brief Get self contact status of this interface

    \return Boolean flag to indicate self contact status of this interface
    */
    [[nodiscard]] bool self_contact() const { return selfcontact_; }

    /*!
    \brief Get two half pass status of this interface

    \return Boolean flag to indicate if two half pass algorithm shall be used
    */
    [[nodiscard]] bool two_half_pass() const { return two_half_pass_; }

    /*!
    \brief Get row map of active nodes

    \pre Filled() == true is prerequisite

    */
    [[nodiscard]] std::shared_ptr<const Core::LinAlg::Map> active_nodes() const
    {
      if (not filled()) FOUR_C_THROW("CONTACT::Interface::fill_complete was not called");

      return activenodes_;
    }

    /*!
    \brief Get row map of active dofs

    \pre Filled() == true is prerequisite

    */
    [[nodiscard]] std::shared_ptr<const Core::LinAlg::Map> active_dofs() const
    {
      if (not filled()) FOUR_C_THROW("CONTACT::Interface::fill_complete was not called");

      return activedofs_;
    }

    [[nodiscard]] std::shared_ptr<const Core::LinAlg::Map> inactive_nodes() const
    {
      if (not filled()) FOUR_C_THROW("CONTACT::Interface::fill_complete was not called");

      return inactivenodes_;
    }

    /*!
    \brief Get row map of active dofs

    \pre Filled() == true is prerequisite

    */
    [[nodiscard]] virtual std::shared_ptr<const Core::LinAlg::Map> inactive_dofs() const
    {
      if (not filled()) FOUR_C_THROW("CONTACT::Interface::fill_complete was not called");

      return inactivedofs_;
    }

    /*!
    \brief Get row map of matrix N

    \pre Filled() == true is prerequisite

    */
    [[nodiscard]] virtual std::shared_ptr<const Core::LinAlg::Map> active_n_dofs() const
    {
      if (not filled()) FOUR_C_THROW("CONTACT::Interface::fill_complete was not called");

      return activen_;
    }

    /*!
    \brief Get row map of matrix T

    \pre Filled() == true is prerequisite

    */
    [[nodiscard]] virtual std::shared_ptr<const Core::LinAlg::Map> active_t_dofs() const
    {
      if (not filled()) FOUR_C_THROW("CONTACT::Interface::fill_complete was not called");

      return activet_;
    }

    /*!
    \brief Get row map of slip nodes

    \pre Filled() == true is prerequisite

    */
    [[nodiscard]] virtual std::shared_ptr<const Core::LinAlg::Map> slip_nodes() const
    {
      if (filled())
      {
        return slipnodes_;
      }
      else
      {
        FOUR_C_THROW("CONTACT::Interface::fill_complete was not called");
      }
    }

    /*!
    \brief Get row map of slip node dofs

    \pre Filled() == true is prerequisite

    */
    [[nodiscard]] virtual std::shared_ptr<const Core::LinAlg::Map> slip_dofs() const
    {
      if (filled())
      {
        return slipdofs_;
      }
      else
      {
        FOUR_C_THROW("CONTACT::Interface::fill_complete was not called");
      }
    }

    /*!
    \brief Get row map of matrix T for slip nodes

    \pre Filled() == true is prerequisite

    */
    [[nodiscard]] virtual std::shared_ptr<const Core::LinAlg::Map> slip_t_dofs() const
    {
      if (filled())
      {
        return slipt_;
      }
      else
      {
        FOUR_C_THROW("CONTACT::Interface::fill_complete was not called");
      }
    }

    /*!
    \brief Get row map of nonsmooth node

    \pre Filled() == true is prerequisite

    */
    [[nodiscard]] virtual std::shared_ptr<const Core::LinAlg::Map> non_smooth_nodes() const
    {
      if (filled())
      {
        return nonsmoothnodes_;
      }
      else
      {
        FOUR_C_THROW("CONTACT::Interface::fill_complete was not called");
      }
    }

    /*!
    \brief Get row map of smooth nodes

    \pre Filled() == true is prerequisite

    */
    [[nodiscard]] virtual std::shared_ptr<const Core::LinAlg::Map> smooth_nodes() const
    {
      if (filled())
      {
        return smoothnodes_;
      }
      else
      {
        FOUR_C_THROW("CONTACT::Interface::fill_complete was not called");
      }
    }

    /*!
    \brief Get row map of smooth nodes

    \pre Filled() == true is prerequisite

    */
    [[nodiscard]] virtual std::shared_ptr<const Core::LinAlg::Map> sdof_vertex_rowmap() const
    {
      if (filled())
      {
        return sdofVertexRowmap_;
      }
      else
      {
        FOUR_C_THROW("CONTACT::Interface::fill_complete was not called");
      }
    }
    /*!
    \brief Get row map of smooth nodes

    \pre Filled() == true is prerequisite

    */
    [[nodiscard]] virtual std::shared_ptr<const Core::LinAlg::Map> sdof_vertex_colmap() const
    {
      if (filled())
      {
        return sdofVertexColmap_;
      }
      else
      {
        FOUR_C_THROW("CONTACT::Interface::fill_complete was not called");
      }
    }
    [[nodiscard]] virtual std::shared_ptr<const Core::LinAlg::Map> sdof_edge_rowmap() const
    {
      if (filled())
      {
        return sdofEdgeRowmap_;
      }
      else
      {
        FOUR_C_THROW("CONTACT::Interface::fill_complete was not called");
      }
    }
    [[nodiscard]] virtual std::shared_ptr<const Core::LinAlg::Map> sdof_edge_colmap() const
    {
      if (filled())
      {
        return sdofEdgeColmap_;
      }
      else
      {
        FOUR_C_THROW("CONTACT::Interface::fill_complete was not called");
      }
    }
    [[nodiscard]] virtual std::shared_ptr<const Core::LinAlg::Map> sdof_surf_rowmap() const
    {
      if (filled())
      {
        return sdofSurfRowmap_;
      }
      else
      {
        FOUR_C_THROW("CONTACT::Interface::fill_complete was not called");
      }
    }
    [[nodiscard]] virtual std::shared_ptr<const Core::LinAlg::Map> sdof_surf_colmap() const
    {
      if (filled())
      {
        return sdofSurfColmap_;
      }
      else
      {
        FOUR_C_THROW("CONTACT::Interface::fill_complete was not called");
      }
    }

    /*!
    \brief Get number of slave / master pairs of this interface (proc local)

    */
    [[nodiscard]] virtual int slave_master_pairs() const { return smpairs_; }

    /*!
    \brief Get number of slave / master integration pairs of this interface (proc local)

    */
    [[nodiscard]] virtual int slave_master_int_pairs() const { return smintpairs_; }

    /*!
    \brief Get number of integration cells of this interface (proc local)

    */
    [[nodiscard]] virtual int integration_cells() const { return intcells_; }

    //@}

    //! @name Evlauation methods

    /*!
    \brief Add a CONTACT::Node to the interface (Filled()==true NOT prerequisite)

    \param cnode (in): Teuchos::rcp to a contact node

    \return Filled()==false

    */
    virtual void add_node(std::shared_ptr<CONTACT::Node> cnode);

    /*!
    \brief Add a CONTACT::Element to the interface

    \pre Filled() == true is prerequisite

    \param cele (in): Teuchos::rcp to a contact element

    \return Filled()==false

    */
    virtual void add_element(std::shared_ptr<CONTACT::Element> cele);

    //! @name Parallel distribution and ghosting
    /* References
    ==========

    - M. Mayr, A. Popp: Scalable computational kernels for mortar finite element methods,
    Engineering with Computers, 2023, https://doi.org/10.1007/s00366-022-01779-3
    */
    //! @{

    /*!
    \brief Update the parallel layout, distribution, and related data structures

    1. If required by \c perform_rebalancing, let's rebalance the interface discretizations.
    1. If required by \c enforce_ghosting_update, let's update the ghosting of the master-sided
    interface.
    1. fill_complete to update all relevant maps on all procs.
    1. Re-create search tree, if ghosting has changed.

    @param perform_rebalancing Flag to enforce rebalancing of interface discretizations
    @param enforce_ghosting_update Flag to enforce an update of the interface ghosting
    @param maxdof Largest GID of underlying solid discretization
    @param meanVelocity Mean velocity of this interface
    */
    void update_parallel_layout_and_data_structures(const bool perform_rebalancing,
        const bool enforce_ghosting_update, const int maxdof, const double meanVelocity) final;

    /*!
    \brief Redistribute contact interface among all procs

    Derived version!

    When first creating a contact interface, its parallel distribution
    is simply copied from the underlying problem discretization. This,
    of course, is not the optimal parallel distribution for evaluating
    the contact coupling terms, as the interface ownership might be
    restricted to only very few processors. Moreover, no parallel
    scalability can be achieved with this procedure, because adding
    processors to the problem discretization does not automatically
    mean adding processors to the interface discretization.

    Thus, an independent parallel distribution of the interface is
    desirable, which divides the interface among all available
    processors. redistribute() is the method to achieve this.
    Moreover, for contact problems we have to account for the fact
    that only parts of the slave surface actually need to evaluate
    contact terms (those parts that are "close" to the master side).

    Internally, we call ZOLTAN to re-partition the contact interfaces
    in three independent parts: (1) close slave part, (2) non-close
    slave part, (3) master part. This results in new "optimal" node/element
    maps of the interface discretization. Note that after redistribute(),
    we must call fill_complete() again. Note also that for contact
    simulations redistribute() might be called dynamically again and
    again to account for changes of the contact zone.

    Two special cases are treated separately: First, if ALL slave
    elements of the interface have some "close" neighbors, we do not
    need to distinguish the two different slave parts. Thus, we
    simply call the base class method redistribute() also used for
    meshtying. Second, if NO slave element of the interface has any
    "close" neighbors, we do not need to redistribute at all. This
    is indicated by returning with a boolean return value FALSE.

    References
    ==========

    - M. Mayr, A. Popp: Scalable computational kernels for mortar finite element methods,
    Engineering with Computers, 2023, https://doi.org/10.1007/s00366-022-01779-3
    */
    void redistribute() final;

    void round_robin_change_ownership();

    void round_robin_detect_ghosting();

    void round_robin_extend_ghosting(bool firstevaluation);

    /*!
    \brief Collect data concerning load balance and parallel distribution

    Check all slave elements and count
    - possibly active elements in the column map
    - possibly active elements in the row map

    \param[out] numColElements Number of column elements that are potentially in contact
    \param[out] numRowElements Number of row elements that are potentially in contact

    \note We are only interested in slave elements here, since they have to do (almost) all the work
    during evaluation. Master elements don't require any computations and, hence, can be neglected
    here.
    */
    void collect_distribution_data(int& numColElements, int& numRowElements);

    //! @}

    /*!
    \brief Create binary search tree

    The method creates a binary tree object for efficient search. This is
    an overloaded method specific for contact, as in this case we have to
    consider the possibility of SELF-contact.

    Derived version!

    */
    void create_search_tree() final;

    /*!
    \brief Initialize / reset interface for contact

    Derived version!

    */
    void initialize() override;

    /*!
    \brief Set element areas

    Derived version!

    */
    void set_element_areas() override;

    /*!
    \brief Export nodal normals

    This method exports / communicates the nodal normal vector and all
    associated information (nodal tangent vectors, normal and tangent
    linearizations) from row to column map layout.

    Derived version!

    */
    void export_nodal_normals() const override;

    /*!
    \brief Binary tree search algorithm for potentially coupling slave /
    master pairs (element-based algorithm) including self-contact

    Derived version!

    */
    bool evaluate_search_binarytree() final;

    /*!
    \brief Integrate Mortar matrix M and gap g on slave/master overlaps

    Derived version!

    */
    bool mortar_coupling(Mortar::Element* sele, std::vector<Mortar::Element*> mele,
        const std::shared_ptr<Mortar::ParamsInterface>& mparams_ptr) final;

    /*!
    \brief evaluate coupling terms for nts coupling + lin

    */
    void evaluate_nts() final;

    /*!
    \brief evaluate coupling terms for lts coupling + lin

    */
    void evaluate_lts() final;

    /*!
    \brief evaluate coupling terms for ltl coupling + lin

    */
    void evaluate_ltl() final;

    /*!
    \brief evaluate coupling terms for stl coupling + lin

    */
    void evaluate_stl() final;

    /*!
    \brief Integrate penalty scaling factor \f$\kappa\f$ on slave element

    This method is only called, if a penalty strategy is applied. It is
    called ONCE at the beginning of the simulation and evaluates the
    penalty scaling factor kappa_j = int_{slave} (N_j) dslave. The
    correct interpolation N_j is chosen for any case (2D, 3D, linear
    quadratic, piecewise linear...)

    \todo maybe update kappa each time step?

    */
    virtual bool integrate_kappa_penalty(CONTACT::Element& sele);

    /*!
    \brief Evaluate relative movement (jump) of slave nodes

    In the case of frictional contact, an important geometric measure is
    the relative movement (jump) of the contacting bodies. Here, this is evaluated
    over change of mortar projection. Also, the directional derivatives are
    evaluated here.

    */
    virtual void evaluate_relative_movement(
        const std::shared_ptr<Core::LinAlg::Vector<double>> xsmod,
        const std::shared_ptr<Core::LinAlg::SparseMatrix> dmatrixmod,
        const std::shared_ptr<Core::LinAlg::SparseMatrix> doldmod);

    /*!
      \brief Evaluate nodal distances and linearization

    */
    virtual void evaluate_distances(const std::shared_ptr<const Core::LinAlg::Vector<double>>& vec,
        std::map<int, std::vector<double>>& mynormals,
        std::map<int, std::vector<Core::Gen::Pairedvector<int, double>>>& dmynormals,
        std::map<int, double>& mygap, std::map<int, std::map<int, double>>& dmygap);

    /*!
    \brief Assemble slave coordinates (xs)

    */
    virtual void assemble_slave_coord(std::shared_ptr<Core::LinAlg::Vector<double>>& xsmod);

    /*!
    \brief Evaluate L2 Norm of tangential contact conditions

    */
    virtual void evaluate_tangent_norm(double& cnormtan);

    /*!
    \brief Assemble gap-computed lagrange multipliers and nodal linlambda derivatives into nodal
    quantities using the Macauley bracket

    When dealing with penalty methods, the lagrange multipliers are not independent variables
    anymore. Instead, they can be computed in terms of the weighted gap and the penalty parameter.
    This is done here so every node stores the correct lm and thus we integrate smoothly into the
    overlaying algorithm.

    Additionally, we use the performed loop over all nodes to store the nodal derivlambda_j matrix
    right there.

    As a result, the function notifies the calling routine if any negative gap was detected
    and thus whether the interface is in contact or not. In consequence, after calling this routine
    from within the penalty strategy object, the contact status is known at a global level.

    Note: To be able to perform this computation, weighted gaps and normals have to be available
    within every node! Since this computation is done via Interface::evaluate() in the Integrator
    class, these corresponding methods have to be called before AssembleMacauley()!

    */
    virtual void assemble_reg_normal_forces(bool& localisincontact, bool& localactivesetchange);

    /*!
    \brief Assemble geometry dependent, tangential lagrange multipliers
    and their derivatives in the penalty case
    */
    virtual void assemble_reg_tangent_forces_penalty();

    /*!
    \brief Assemble geometry dependent, tangential lagrange multipliers
    and their derivatives in the Uzawa augmented lagrange case
    */
    virtual void assemble_reg_tangent_forces_uzawa();

    /*!
    \brief Assemble LM derivatives into global matrix (penalty strategy)

    \param[in/out] lambdaglobal Matrix to be assembled into
    */
    virtual void assemble_lin_z(Core::LinAlg::SparseMatrix& linzglobal);

    /*!
    \brief Assemble matrix T containing nodal tangents and/or matrix N containing nodal normals!

    */
    virtual void assemble_tn(std::shared_ptr<Core::LinAlg::SparseMatrix> tglobal = nullptr,
        std::shared_ptr<Core::LinAlg::SparseMatrix> nglobal = nullptr);

    /*!
    \brief Assemble matrix S containing linearizations

    This method builds an algebraic form of the FULL linearization
    of the normal contact condition g~ = 0. Concretely, this
    includes assembling the linearizations of the slave side
    nodal normals and of the Mortar matrices D  and M.

    */
    virtual void assemble_s(Core::LinAlg::SparseMatrix& sglobal);

    /*!
    \brief Assemble matrix Tderiv and Nderiv containing linearizations

    This method builds an algebraic form of the FULL linearization
    of the tangential contact condition (frictionless) and/or normal
    condition (tractionlss).
    Concretely, this means assembling the linearization of the slave side
    nodal tangents / nodal normals and the current Lagrange multipliers.

      usePoroLM: linearisation will be multiplied with ...
     - true ->  poro no penetration lagrange multiplier!
     - false -> standard contact lagrange multiplier!

    */
    virtual void assemble_t_nderiv(
        std::shared_ptr<Core::LinAlg::SparseMatrix> tderivglobal = nullptr,
        std::shared_ptr<Core::LinAlg::SparseMatrix> nderivglobal = nullptr, bool usePoroLM = false);

    /*!
    \brief Assemble matrices LinD, LinM containing linearizations

    This method builds an algebraic form of the FULL linearization
    of the contact force vector. Concretely, this means assembling
    the linearization of the Mortar matrices D and M and the
    current Lagrange multipliers.

    usePoroLM: linearisation will be multiplied with ...
     - true ->  poro no penetration lagrange multiplier!
     - false -> standard contact lagrange multiplier!

    */
    virtual void assemble_lin_dm(Core::LinAlg::SparseMatrix& lindglobal,
        Core::LinAlg::SparseMatrix& linmglobal, bool usePoroLM = false);

    /*!
    \brief subroutine assemble lin d
    */
    virtual void assemble_lin_d(Core::LinAlg::SparseMatrix& lindglobal, bool usePoroLM = false);

    /*!
    \brief subroutine assemble lin m
    */
    virtual void assemble_lin_m(Core::LinAlg::SparseMatrix& linmglobal, bool usePoroLM = false);


    /*!
    \brief Assemble weighted gap g

    Derived version! It is very important to note that g has a different
    meaning here in contact than in standard mortar meshtying applications,
    thus we need a derived method. Referring to Mortar::Interface::AssembleG(),
    we notice that g is a vector-quantity at each node there. Yet, in
    (frictionless) we are only interested in the normal part, which makes
    g a scalar quantity here. Compare also the different definitions of g_
    in CONTACT::MtAbstractStrategy::mortar_coupling() -> gsdofrowmap_ and
    in CONTACT::AbstractStrategy::initialize_mortar()/assemble_mortar() -> gsnoderowmap_!!!

    */
    virtual void assemble_g(Core::LinAlg::Vector<double>& gglobal);

    /*!
    \brief Assemble inactive rhs (incremental delta_z_)
    */
    virtual void assemble_inactiverhs(Core::LinAlg::Vector<double>& inactiverhs);

    /*!
    \brief Assemble tangential rhs (incremental delta_z_)
    */
    virtual void assemble_tangrhs(Core::LinAlg::Vector<double>& tangrhs);

    /*!
    \brief Assemble matrix LinStick containing linearizations

    This method builds an algebraic form of the FULL linearization
    of the tangential stick condition delta tg = 0. Concretely, this
    includes assembling the linearizations of the slave side
    nodal tangents and of the Mortar matrices D  and M.

    */
    virtual void assemble_lin_stick(Core::LinAlg::SparseMatrix& linstickLMglobal,
        Core::LinAlg::SparseMatrix& linstickDISglobal,
        Core::LinAlg::Vector<double>& linstickRHSglobal);
    /*!
    \brief Assemble matrix LinSlip containing linearizations

    This method builds an algebraic form of the FULL linearization
    of the tangential slip condition. Concretely, this
    includes assembling the linearizations of the slave side
    nodal tangents and of the Mortar matrices D  and M.

    */
    virtual void assemble_lin_slip(Core::LinAlg::SparseMatrix& linslipLMglobal,
        Core::LinAlg::SparseMatrix& linslipDISglobal,
        Core::LinAlg::Vector<double>& linslipRHSglobal);

    /*!
      \brief Assemble linearization of regularized normal constraint
    */
    virtual void assemble_normal_contact_regularization(Core::LinAlg::SparseMatrix& d_disp,
        Core::LinAlg::SparseMatrix& d_lm, Core::LinAlg::Vector<double>& f);

    /*!
      \brief Assemble linearization of slip condition with regularized normal constraint
    */
    virtual void assemble_lin_slip_normal_regularization(
        Core::LinAlg::SparseMatrix& linslipLMglobal, Core::LinAlg::SparseMatrix& linslipDISglobal,
        Core::LinAlg::Vector<double>& linslipRHSglobal);


    /*!
    \brief Update active set and check for convergence

    In this function we loop over all  slave nodes to check, whether the
    assumption of them being active or inactive respectively has been correct.
    If a single node changes state, the active set is adapted accordingly and the convergence
    flag is kept on false.

    Here we have the semi-smooth Newton case
    with one combined iteration loop for active set search and large
    deformations. As a consequence this method is called AFTER each
    (not yet converged) Newton step. If there is a change in the active
    set or the residual and disp norm are still above their limits,
    another Newton step has to be performed.

    \return Boolean flag indicating convergence of active set
    */
    virtual bool update_active_set_semi_smooth();

    /*!
    \brief Update active set to conform with given active set from input file

    In this function we loop over all  slave nodes to check, whether the
    current active set decision for each node still conforms with the prescribed value
    from the input file. If not, the input file overrules the current value.

    Since this just enforces prescribed information from the input file,
    this is not to be considered as a change in the active set and, thus,
    does not affect the convergence of the active set. So, not convergence flag
    on return.
    */
    virtual void update_active_set_initial_status() const;

    /*!
    \brief Build active set (nodes / dofs) of this interface

    If the flag init==true, the active set is initialized (for t=0)
    according to the contact initialization defined in the input file.

    \param[in] init Boolean flag to enforce initialization of the active set

    \return true (hard-coded)
    */
    virtual bool build_active_set(bool init = false);

    /*!
    \brief Split active dofs into N- and T-part

    */
    virtual bool split_active_dofs();

    /*!
    \brief Update the lagrange multiplier sets for self contact

    \param(in) gref_lmmap: global lagrange multiplier reference map
    \param(in) gref_smmap: global merged slave/master reference map
    */
    void update_self_contact_lag_mult_set(
        const Core::LinAlg::Map& gref_lmmap, const Core::LinAlg::Map& gref_smmap);

    /*!
    \brief Assemble normal coupling weighted condition. It is useful for poro contact

    */
    virtual void assemble_normal_coupling(Core::LinAlg::Vector<double>& gglobal);

    /*!
    \brief Assemble linearisation of normal coupling weighted condition for poro contact

    */
    virtual void assemble_normal_coupling_linearisation(
        Core::LinAlg::SparseMatrix& sglobal, Coupling::Adapter::Coupling& coupfs,
        bool AssembleVelocityLin = false  // if true velocity linearisation will be assembled into
                                          // sglobal, otherwise lin. w.r.t. displacements!
    );

    /*!
    \brief Derivative of D-matrix multiplied with a slave dof vector

    \todo Complete documentation of input parameters.

    @param CoupLin ??
    @param x ??
    */
    virtual void assemble_coup_lin_d(
        Core::LinAlg::SparseMatrix& CoupLin, const std::shared_ptr<Core::LinAlg::Vector<double>> x);

    /*! \brief Derivative of (transposed) M-matrix multiplied with a slave dof vector

    \todo Complete documentation of input parameters.

    @param CoupLin ??
    @param x ??
    */
    virtual void assemble_coup_lin_m(
        Core::LinAlg::SparseMatrix& CoupLin, const std::shared_ptr<Core::LinAlg::Vector<double>> x);

    /*!
    \brief Store current (contact) nodal entries to old ones

    \todo Complete documentation of input parameters.

    Contact nodes own their current entries and old ones (last converged
    state) from. p.e. the mortar matrices D and M. This function writes the
    current ones to the old ones.

    \param type ??
    */
    virtual void store_to_old(Mortar::StrategyBase::QuantityType type);

    //! @name Finite difference checks
    //!@{

    /*!
    \brief Check normal/tangent derivatives with finite differences

    */
    void fd_check_normal_deriv();

    /*!
    \brief Check normal/tangent derivatives with finite differences

    */
    void fd_check_normal_cpp_deriv();

    /*!
    \brief Check Mortar matrix D derivatives with finite differences

    */
    void fd_check_mortar_d_deriv();

    /*!
    \brief Check Mortar matrix M derivatives with finite differences

    */
    void fd_check_mortar_m_deriv();

    /*!
    \brief Check weighted gap g derivatives with finite differences

    */
    void fd_check_gap_deriv();

    /*!
    \brief Check gap g derivatives with finite differences LTL

    */
    void fd_check_gap_deriv_ltl();

    /*!
    \brief Check jump derivatives with finite differences LTL

    */
    void fd_check_jump_deriv_ltl();

    /*!
    \brief Check alpha derivatives with finite differences (for hybrid formulation)

    */
    void fd_check_alpha_deriv();


    /*!
    \brief Check weighted slip increment derivatives with finite differences (gp-wise calculated)

    */
    void fd_check_slip_incr_deriv_txi();   //- TXI
    void fd_check_slip_incr_deriv_teta();  //- TETA

    /*!
    \brief Check tangential LM derivatives with finite differences

    */
    void fd_check_tang_lm_deriv();

    /*!
    \brief Check stick condition derivatives with finite differences

    */
    virtual void fd_check_stick_deriv(Core::LinAlg::SparseMatrix& linstickLMglobal,
        Core::LinAlg::SparseMatrix& linstickDISglobal);

    /*!
    \brief Check slip condition derivatives with finite differences

    */
    virtual void fd_check_slip_deriv(
        Core::LinAlg::SparseMatrix& linslipLMglobal, Core::LinAlg::SparseMatrix& linslipDISglobal);

    /*!
    \brief Check penalty approach with finite differences

    */
    void fd_check_penalty_trac_nor();

    /*!
    \brief Check frictional penalty traction with finite differences

    */
    virtual void fd_check_penalty_trac_fric();

    //!@}

    void write_nodal_coordinates_to_file(const int interfacel_id,
        const Core::LinAlg::Map& nodal_map, const std::string& full_path) const;

    /*!
    \brief Add line to line penalty forces

    */
    void add_ltl_forces(Core::LinAlg::FEVector<double>& feff);

    /*!
    \brief Add line to line penalty forces

    */
    void add_lts_forces_master(Core::LinAlg::FEVector<double>& feff);

    /*!
    \brief Add line to line penalty forces

    */
    void add_nts_forces_master(Core::LinAlg::FEVector<double>& feff);

    /*!
    \brief Add line to line penalty forces - friction

    */
    void add_ltl_forces_friction(Core::LinAlg::FEVector<double>& feff);

    /*!
    \brief Add line to line penalty stiffness contribution

    */
    void add_ltl_stiffness(Core::LinAlg::SparseMatrix& kteff);

    /*!
    \brief Add line to segment penalty stiffness contribution master side

    */
    void add_lts_stiffness_master(Core::LinAlg::SparseMatrix& kteff);

    /*!
    \brief Add node to segment penalty stiffness contribution master side

    */
    void add_nts_stiffness_master(Core::LinAlg::SparseMatrix& kteff);

    /*!
    \brief Add line to line penalty stiffness contribution

    */
    void add_ltl_stiffness_friction(Core::LinAlg::SparseMatrix& kteff);

    [[nodiscard]] inline bool is_friction() const { return friction_; }

    [[nodiscard]] const Interface& get_ma_sharing_ref_interface() const;

    //! @name Output
    //! @{

    //! [derived]
    void postprocess_quantities(const Teuchos::ParameterList& outputParams) const final;

    //! @}
   public:
    [[nodiscard]] std::shared_ptr<const Core::LinAlg::Vector<double>> cn() const
    {
      return cnValues_;
    };

    [[nodiscard]] const Core::LinAlg::Vector<double>& cn_ref() const
    {
      if (!cnValues_) FOUR_C_THROW("The cnValues_ is not initialized!");
      return *cnValues_;
    }

    [[nodiscard]] std::shared_ptr<const Core::LinAlg::Vector<double>> ct() const
    {
      return ctValues_;
    };

    [[nodiscard]] const Core::LinAlg::Vector<double>& ct_ref() const
    {
      if (!ctValues_) FOUR_C_THROW("The ctValues_ is not initialized!");
      return *ctValues_;
    }

   private:
    std::shared_ptr<Core::LinAlg::Vector<double>>& get_cn() { return cnValues_; };

    Core::LinAlg::Vector<double>& get_cn_ref()
    {
      if (!cnValues_) FOUR_C_THROW("The cnValues_ is not initialized!");
      return *cnValues_;
    }

    std::shared_ptr<Core::LinAlg::Vector<double>>& get_ct() { return ctValues_; };

    Core::LinAlg::Vector<double>& get_ct_ref()
    {
      if (!ctValues_) FOUR_C_THROW("The ctValues_ is not initialized!");
      return *ctValues_;
    }

   protected:
    /** \brief split the interface elements into a far and a close set
     *
     *  This version of the method performs the split closely bound to the
     *  information collected during the contact search. See the derived
     *  version(s) for alternatives.
     *
     *  \note The here collected information mainly decides over the
     *  distribution after the parallel redistribution.
     *
     *  \note Splitting into close/non-close elements/nodes can be suppressed via the input file. If
     *  done so, then all elements/nodes are considered to be far nodes. The list of close
     *  element/nodes is left empty.
     *
     *  \param closeele (out)     (slave) interface element GIDs of the close set
     *  \param noncloseele (out)  (slave) interface element GIDs of the far set
     *  \param localcns (out)     (slave) node GIDs of the elements in the close set
     *  \param localfns (out)     (slave) node GIDs of the elements in the far set
     *
     *  All sets are restricted to the current/local processor. */
    virtual void split_into_far_and_close_sets(std::vector<int>& closeele,
        std::vector<int>& noncloseele, std::vector<int>& localcns,
        std::vector<int>& localfns) const;

    /*!
    \brief initialize node and element data container

    Derived version!

    */
    void initialize_data_container() override;

    /*!
    \brief initialize slave/master node status for corner/edge modification

    Derived version!

    */
    void initialize_corner_edge() final;

    /*!
    \brief Set closest-point-projection normal and tangent to data container

    Derived version!

    */
    virtual void set_cpp_normal(Mortar::Node& snode, double* normal,
        std::vector<Core::Gen::Pairedvector<int, double>>& normallin);

    /*!
    \brief do calculations which are required for contact term evaluation:
           for example: nodal normal calculation

    Derived version!

    */
    void pre_evaluate(const int& step, const int& iter) final;

    /*!
    \brief Routine to control contact term evaluation. Here, we decide if mortar, nts
           etc. is evaluated

    Derived version!

    */
    void evaluate_coupling(const Core::LinAlg::Map& selecolmap,
        const Core::LinAlg::Map* snoderowmap,
        const std::shared_ptr<Mortar::ParamsInterface>& mparams_ptr) final;

    /*!
    \brief Evaluate segment-to-segment coupling (mortar...)

    */
    void evaluate_sts(const Core::LinAlg::Map& selecolmap,
        const std::shared_ptr<Mortar::ParamsInterface>& mparams_ptr) final;

    /*!
    \brief export master nodal normals for cpp calculation

    */
    virtual void export_master_nodal_normals() const;

    /*!
    \brief evaluate cpp normals on slave side based on averaged normal field on master side

    */
    virtual void evaluate_cpp_normals();

    /*!
    \brief do calculations which are required after contact term evaluation:
           for example: scale nodal entries

    Derived version!

    */
    void post_evaluate(const int step, const int iter) override;

    /*!
    \brief Compute cpp normal based on averaged nodal normal field on master side.

    */
    virtual double compute_cpp_normal(Mortar::Node& mrtrnode, std::vector<Mortar::Element*> meles,
        double* normal, std::vector<Core::Gen::Pairedvector<int, double>>& normaltolineLin);

    /*!
    \brief 2D routine for cpp normal

    */
    virtual double compute_cpp_normal_2d(const Mortar::Node& mrtrnode,
        std::vector<Mortar::Element*> meles, double* normal,
        std::vector<Core::Gen::Pairedvector<int, double>>& normaltolineLin) const;

    /*!
    \brief 3D routine for cpp normal

    */
    virtual double compute_cpp_normal_3d(Mortar::Node& mrtrnode,
        std::vector<Mortar::Element*> meles, double* normal,
        std::vector<Core::Gen::Pairedvector<int, double>>& normaltolineLin);

    /*!
    \brief Compute normal between slave and master node

    */
    virtual double compute_normal_node_to_node(const Mortar::Node& snode, const Mortar::Node& mnode,
        double* normal, std::vector<Core::Gen::Pairedvector<int, double>>& normaltonodelin) const;

    /*!
    \brief Compute normal between slave node and master edge ele

    */
    virtual double compute_normal_node_to_edge(const Mortar::Node& snode,
        const Mortar::Element& mele, double* normal,
        std::vector<Core::Gen::Pairedvector<int, double>>& normaltonodelin) const;

    /*!
    \brief Set new cn and ct values (global interface vector)

    */
    virtual void set_cn_ct_values(const int& iter);  // newton step


    /*!
    \brief Routine which stores entries from nts algorithm into mortar nodes to reuse
           standard assemble functions
    */
    virtual void store_nt_svalues();

    /*!
    \brief Routine which stores entries from lts algorithm into mortar nodes to reuse
           standard assemble functions
    */
    virtual void store_lt_svalues();

    /*!
    \brief Update interface master and slave sets

    This update is usually only done ONCE in the initialization phase
    and sets up the slave and master sets (elements, nodes, dofs) for
    the whole simulation. Yet, in the case of self contact the sets
    need to be updated again and again during simulation time, as the
    slave/master status is assigned dynamically.

    */
    void update_master_slave_sets() override;

   private:
    //! @name Parallel distribution and ghosting
    //! @{

    /*!
    \brief fill_complete the mortar interface

    The methods completes construction phase of a mortar interface. It creates all row/column maps
    of the mortar interface discretization. Extension of the interface ghosting is done
    separately.herefore, we also have to extend the interface
    ghosting.

    If we have arrived at the final parallel distribution, we have to ask the underlying
    Core::FE::Discretization to assign degrees of freedom. Since this is very expensive,
    let's do this only if requested by the user/algorithm.

    \sa extend_interface_ghosting_safely()

    @param[in] isFinalParallelDistribution Is this the final parallel distribution?
    @param[in] maxdof Largest GID of underlying solid discretization
    */
    void fill_complete_new(const bool isFinalParallelDistribution, const int maxdof = 0) final;

    /*!
    \brief Extend the interface ghosting while guaranteeing sufficient extension

    \note The argument \c meanVelocity is just needed for contact problems that extend the
    master-sided interface ghosting via binning.

    @param meanVelocity Mean velocity of this interface

    References
    ==========

    - M. Mayr, A. Popp: Scalable computational kernels for mortar finite element methods,
    Engineering with Computers, 2023, https://doi.org/10.1007/s00366-022-01779-3

    */
    void extend_interface_ghosting_safely(const double meanVelocity = 0.0) final;

    //! @}

    /*! \brief Set node active if it is active in input file
     *
     * A given node \c cnode is set to be active if it has been specified
     * as active in the input file.
     *
     * @param[in/out] cnode A single contact node
     */
    virtual void set_node_initially_active(CONTACT::Node& cnode) const;

    /*! \brief Check if node \c cnode is set active by gap on input
     *
     * Check if the contact node \c cnode is set active using the INITCONTACTGAPVALUE
     * mechanism. If yes, set the status of the contact node \c cnode to active.
     *
     * @param[in/out] cnode A single contact node
     */
    void set_node_initially_active_by_gap(Node& cnode) const;

    /*!
     * \brief Set condition specific parameters such that the correct parameters are available for
     * the actual evaluation process
     */
    void set_condition_specific_parameters();

    /// pointer to the interface data object
    std::shared_ptr<CONTACT::InterfaceDataContainer> interface_data_;

   protected:
    /** @name References to the interface data container content
     *
     * \remark Please add no new member variables to this class and use the
     *  corresponding data container, instead! If you have any questions
     *  concerning this, do not hesitate and ask me.
     *                                                          hiermeier 03/17 */
    // \todo As already noted by Michael Hiermeier above, the contact interface should not store
    // references to all member variables of the IDataContainer as it is currently implemented.
    // Instead, the contact interface should only store a reference to the data container and then
    // directly access the member variables of the data container using suitable set and get
    // methods! Please also refer to GitLab Issue 165 for more details.
    /// @{

    bool& selfcontact_;       ///< ref. to flag indicating if this is a self contact interface
    bool& friction_;          ///< ref. to flag for frictional contact
    bool& nonSmoothContact_;  ///< ref. to flag for non-smooth contact algorithm
    bool& two_half_pass_;     ///< ref. to flag for two half pass contact algorithm
    CONTACT::ConstraintDirection&
        constr_direction_;  ///< ref. to direction in which the contact constraints are formulated

    std::shared_ptr<Core::LinAlg::Map>&
        activenodes_;                                 ///< ref. to row map of all active slave nodes
    std::shared_ptr<Core::LinAlg::Map>& activedofs_;  ///< ref. to row map of all active slave dofs
    std::shared_ptr<Core::LinAlg::Map>&
        inactivenodes_;  ///< ref. to row map of all active slave nodes
    std::shared_ptr<Core::LinAlg::Map>&
        inactivedofs_;                               ///< ref. to row map of all active slave dofs
    std::shared_ptr<Core::LinAlg::Map>& activen_;    ///< ref. to row map of global N-matrix
    std::shared_ptr<Core::LinAlg::Map>& activet_;    ///< ref. to row map of global T-matrix
    std::shared_ptr<Core::LinAlg::Map>& slipnodes_;  ///< ref. to row map of all slip slave nodes
    std::shared_ptr<Core::LinAlg::Map>& slipdofs_;   ///< ref. to row map of all slip slave dofs
    std::shared_ptr<Core::LinAlg::Map>&
        slipt_;  ///< ref. to row map of part of T-matrix (slip nodes)

    std::shared_ptr<Core::LinAlg::Map>&
        nonsmoothnodes_;  ///< ref. to row map of all nonsmooth slave nodes
    std::shared_ptr<Core::LinAlg::Map>&
        smoothnodes_;  ///< ref. to row map of all smooth slave nodes
    std::shared_ptr<Core::LinAlg::Map>&
        sdofVertexRowmap_;  ///< ref. to row map of all nonsmooth slave nodes
    std::shared_ptr<Core::LinAlg::Map>&
        sdofVertexColmap_;  ///< ref. to row map of all smooth slave nodes
    std::shared_ptr<Core::LinAlg::Map>&
        sdofEdgeRowmap_;  ///< ref. to row map of all nonsmooth slave nodes
    std::shared_ptr<Core::LinAlg::Map>&
        sdofEdgeColmap_;  ///< ref. to row map of all smooth slave nodes
    std::shared_ptr<Core::LinAlg::Map>&
        sdofSurfRowmap_;  ///< ref. to row map of all nonsmooth slave nodes
    std::shared_ptr<Core::LinAlg::Map>&
        sdofSurfColmap_;  ///< ref. to row map of all smooth slave nodes

    std::shared_ptr<Core::LinAlg::Map>& nextendedghosting_;
    std::shared_ptr<Core::LinAlg::Map>& eextendedghosting_;


    std::shared_ptr<SelfBinaryTree>&
        binarytreeself_;  ///< ref. to binary tree for self contact search

    //! cn-values of each node
    std::shared_ptr<Core::LinAlg::Vector<double>>& cnValues_;  ///< ref. to cn
    std::shared_ptr<Core::LinAlg::Vector<double>>& ctValues_;  ///< ref. to ct

    int& smpairs_;     ///< ref. to proc local number of slave/master pairs
    int& smintpairs_;  ///< ref. to proc local number of slave/master integration pairs
    int& intcells_;    ///< ref. to proc local number of integration cells

    /// @}
   private:
    static bool abs_compare(int a, int b) { return (std::abs(a) < std::abs(b)); }
  };  // class Interface
}  // namespace CONTACT

//! << operator
std::ostream& operator<<(std::ostream& os, const CONTACT::Interface& interface);

FOUR_C_NAMESPACE_CLOSE

#endif

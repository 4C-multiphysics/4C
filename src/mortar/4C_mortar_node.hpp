// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MORTAR_NODE_HPP
#define FOUR_C_MORTAR_NODE_HPP

#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_linalg_map.hpp"
#include "4C_utils_pairedvector.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Mortar
{
  /*!
  \brief A subclass of Core::Communication::ParObjectType that adds mortar element type specific
  methods

  */
  class NodeType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "Mortar::NodeType"; }

    static NodeType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static NodeType instance_;
  };

  /*!
   \brief A class containing additional data from mortar nodes

   This class contains additional information from mortar nodes which are
   are not needed for contact search and therefore are only available on the
   node's processor (ColMap). The class NodeDataContainer must be declared
   before the Node itself.

   */
  class NodeDataContainer
  {
   public:
    //! @name Constructors and destructors and related methods
    //! @{

    /*!
     \brief Standard Constructor

     */
    NodeDataContainer();

    //! @}

    //! @name Pack/Unpack
    //! @{

    /*!
     \brief Pack this class so that it can be communicated

     This function packs the data container. This is only called
     when the class has been initialized and the pointer to this
     class exists.

     */
    void pack(Core::Communication::PackBuffer& data) const;

    /*!
     \brief Unpack data from a vector into this class

     This function unpacks the data container. This is only called
     when the class has been initialized and the pointer to this
     class exists.

     */
    void unpack(Core::Communication::UnpackBuffer& buffer);

    //!@}

    //! @name Access methods (general stuff)
    //! @{

    /*!
     \brief Return current nodal normal (only for slave side!) (length 3)
     */
    double* n() { return n_; }

    /*!
     \brief Return const version of current nodal normal (only for slave side!) (length 3)
     */
    const double* n() const { return n_; }

    /*!
     \brief Return current edge tangent(length 3)
     */
    double* edge_tangent() { return edgeTangent_; }

    /*!
     \brief Return const version of current edge tangent(length 3)
     */
    const double* edge_tangent() const { return edgeTangent_; }

    //! @}

    //! @name Access methods for Lagrange multipliers
    //! @{

    /*!
     \brief Return current Lagrange multiplier in step n+1 (only for slave side!) (length 3)
     */
    double* lm() { return lm_; }

    /*!
     \brief Return const ptr to current Lagrange multiplier in step n+1
     (only for slave side!) (length 3)
     */
    const double* lm() const { return lm_; }

    /*!
     \brief Return old Lagrange multiplier from step n (only for slave side!) (length 3)
     */
    double* lmold() { return lmold_; }

    /*!
     \brief Return Lagrange multiplier from last Uzawa step (only for slave side!) (length 3)
     */
    double* lmuzawa() { return lmuzawa_; }

    //! @}

    //! @name Access methods for segment-to-segment data
    //! @{

    /*!
     \brief Return the 'D' map (vector) of this node
     */
    Core::Gen::Pairedvector<int, double>& get_d() { return drows_; }

    /*!
     \brief Return the 'M' map of this node
     */
    std::map<int, double>& get_m() { return mrows_; }

    //! @}

    //! @name Access methods for node-to-segment data
    //! @{

    /*!
     \brief Return the 'D' map (vector) of this node for node-to-segment
     */
    Core::Gen::Pairedvector<int, double>& get_dnts() { return drows_nts_; }

    /*!
     \brief Return the 'M' map of this node for node-to-segment
     */
    std::map<int, double>& get_mnts() { return mrows_nts_; }

    //! @}

    //! @name Access methods for line-to-segment data
    //! @{

    /*!
     \brief Return the 'D' map (vector) of this node for line-to-segment
     */
    Core::Gen::Pairedvector<int, double>& get_dlts() { return drows_lts_; }

    /*!
     \brief Return the 'M' map of this node for line-to-segment
     */
    std::map<int, double>& get_mlts() { return mrows_lts_; }

    //! @}

    //! @name Access methods for line-to-line data
    //! @{

    /*!
     \brief Return the 'D' map (vector) of this node for line-to-line
     */
    Core::Gen::Pairedvector<int, double>& get_dltl() { return drows_ltl_; }

    /*!
     \brief Return the 'M' map of this node for line-to-line
     */
    std::map<int, double>& get_mltl() { return mrows_ltl_; }

    //! @}

    /*!
     \brief Return the 'Mmod' map of this node
     */
    std::map<int, double>& get_mmod() { return mmodrows_; }

    /*!
     \brief Return the d matrix scale factor of this node
     */
    double& get_dscale() { return d_nonsmooth_; }

   private:
    //! don't want = operator
    NodeDataContainer operator=(const NodeDataContainer& old);

    //! don't want copy constructor
    NodeDataContainer(const NodeDataContainer& old);

   protected:
    //! @name Geometry-related vectors
    //! @{

    //! Nodal normal for contact methods
    double n_[3];

    //! Current edge tangent
    double edgeTangent_[3];

    //! @}

    //! @name Lagrange multipliers
    //! @{

    //! Current Lagrange multiplier value (n+1)
    double lm_[3];

    //! Old Lagrange multiplier value (last converged state n)
    double lmold_[3];

    //! Uzawa Lagrange multiplier value (last Uzawa step k)
    double lmuzawa_[3];

    //! @}
    double d_nonsmooth_;

    //! @name Segment-to-segment (sts) specifics
    //! @{

    //! Nodal rows of D matrix for segment-to-segment
    Core::Gen::Pairedvector<int, double> drows_;

    //! Nodal rows of M matrix for segment-to-segment
    std::map<int, double> mrows_;

    //! Nodal rows of Mmod matrix (segment-to-segment)
    std::map<int, double> mmodrows_;

    //! @}

    //! @name Node-to-segment (nts) specifics
    //! @{

    //! Nodal rows of D matrix for node-to-segment
    Core::Gen::Pairedvector<int, double> drows_nts_;

    //! Nodal rows of M matrix for node-to-segment
    std::map<int, double> mrows_nts_;

    //! @}

    //! @name Line-to-segment (lts) specifics
    //! @{

    //! Nodal rows of D matrix for line-to-segment
    Core::Gen::Pairedvector<int, double> drows_lts_;

    //! Nodal rows of M matrix for line-to-segment
    std::map<int, double> mrows_lts_;

    //! @}

    //! @name Line-to-line (ltl) specifics
    //! @{

    //! Nodal rows of D matrix for lts
    Core::Gen::Pairedvector<int, double> drows_ltl_;

    //! Nodal rows of M matrix for lts
    std::map<int, double> mrows_ltl_;

    //! @}
  };  // class NodeDataContainer

  /*!
   \brief A class for a mortar node derived from Core::Nodes::Node

   This class represents a finite element node capable of mortar coupling.

   */
  class Node : public Core::Nodes::Node
  {
   public:
    //! @name Enums and Friends
    //! @{

    /*!
     \brief The discretization is a friend of Node
     */
    friend class Core::FE::Discretization;

    //! @}

    //! @name Constructors and destructors and related methods

    /*!
     \brief Standard Constructor

     \param id     (in): A globally unique node id
     \param coords (in): vector of nodal coordinates
     \param owner  (in): Owner of this node.
     \param dofs   (in): list of global degrees of freedom
     \param isslave(in): flag indicating whether node is slave or master

     */
    Node(int id, const std::vector<double>& coords, const int owner, const std::vector<int>& dofs,
        const bool isslave);

    /*!
     \brief Copy Constructor

     Makes a deep copy of a Node

     */
    Node(const Mortar::Node& old);

    /*!
     \brief Deep copy the derived class and return pointer to it

     */
    Mortar::Node* clone() const override;



    /*!
     \brief Return unique ParObject id

     every class implementing ParObject needs a unique id defined at the
     top of lib/parobject.H.

     */
    int unique_par_object_id() const override
    {
      return NodeType::instance().unique_par_object_id();
    }

    /*!
     \brief Pack this class so it can be communicated

     \ref pack and \ref unpack are used to communicate this node

     */
    void pack(Core::Communication::PackBuffer& data) const override;

    /*!
     \brief Unpack data from a char vector into this class

     \ref pack and \ref unpack are used to communicate this node

     */
    void unpack(Core::Communication::UnpackBuffer& buffer) override;

    //@}

    //! @name Access methods

    /*!
     \brief Print this Node
     */
    void print(std::ostream& os) const override;

    /*!
     \brief Is node on slave or master side of mortar interface
     */
    bool is_slave() const { return isslave_; }

    /*!
     \brief Modify slave / master status of current node

     This changing of topology becomes necessary for self contact
     simulations, where slave and master status are assigned dynamically

     This belated modification is also necessary to be able to deal with
     boundary nodes on the slave side of the interface. Their status is
     changed to master, thy do NOT carry Lagrange multipliers and their
     neighbors' dual shape fct. are modified!
     */
    bool& set_slave() { return isslave_; }

    /*!
     \brief Return attached status

     */
    bool is_detected() const { return detected_; }

    /*!
     \brief Return attached status

     */
    bool& set_detected() { return detected_; }

    /*!
     \brief Is slave node tied or untied
     */
    bool is_tied_slave() const { return istiedslave_; }

    /*!
     \brief Modify tying status of current slave node

     This changing of status becomes necessary for meshtying simulations
     where the given slave surface only partially overlaps with the master
     surface. Then the flag istiedslave_ needs to be initialized according
     to the actual tying status. True means that the node is participating
     in meshtying and thus carries mortar contributions. False means that
     the node is not involved in meshtying and does not need to carry
     Lagrange multipliers anyway. There is some similarity with an active
     set definition in contact mechanics, yet the set is static here in the
     meshtying case, of course.

     During problem initialization this flag is first set to the same value
     as isslave_, i.e. we assume a complete projection of the slave surface.
     Then, the actual meshtying zone is identified and this flag is adapted
     for each node accordingly.

     */
    bool& set_tied_slave() { return istiedslave_; }

    /*!
     \brief Is node on boundary of slave side of mortar interface
     */
    bool is_on_bound() const { return isonbound_; }

    /*!
     \brief Set slave side boundary status of current node
     */
    bool& set_bound() { return isonbound_; }

    /*!
     \brief Is this node on a boundary, edge, or corner?
     */
    bool is_on_boundor_ce() const { return ((isonedge_ or isoncorner_) or isonbound_); }

    /*!
     \brief Is this node on an edge or corner?
     */
    bool is_on_corner_edge() const { return (isonedge_ or isoncorner_); }

    /*!
     \brief Is this nnode on a geometrical edge
     */
    bool is_on_edge() const { return isonedge_; }

    /*!
     \brief Set node is on geometrical edge
     */
    bool& set_on_edge() { return isonedge_; }

    /*!
     \brief Is this node on geometrical corner?
     */
    bool is_on_corner() const { return isoncorner_; }

    /*!
     \brief Is this node on a boundary or a corner?
     */
    bool is_on_corneror_bound() const { return (isoncorner_ or isonbound_); }

    /*!
     \brief Set node is on geometrical corner
     */
    bool& set_on_corner() { return isoncorner_; }

    /*!
     \brief Return D.B.C. status of this node (true if at least one dof with D.B.C)
     */
    bool is_dbc() const { return isdbc_; }

    /*!
     \brief Set D.B.C. status of current node
     */
    bool& set_dbc() { return isdbc_; }

    /*!
     \brief Get number of degrees of freedom
     */
    [[nodiscard]] int num_dof() const { return static_cast<int>(dofs_.size()); }

    /*!
     \brief Get predefined degrees of freedom
     */
    [[nodiscard]] const std::vector<int>& dofs() const { return dofs_; };

    /*!
     \brief Return current configuration (length 3)
     */
    double* xspatial() { return xspatial_; }

    /*!
     \brief Return const ptr to current configuration (length 3)
     */
    const double* xspatial() const { return xspatial_; }

    /*!
     \brief Return old displacement (length 3)
     */
    double* uold() { return uold_; }
    const double* uold() const { return uold_; }

    /*!
     \brief Return projection status of this node (only for slave side!)
     */
    bool& has_proj() { return hasproj_; }

    /*!
     \brief Return segmentation / cell status of this node (only for slave side!)
     */
    bool& has_segment() { return hassegment_; }

    /// get the upper bound for the number of D-entries
    int get_num_dentries() const { return dentries_; };

    /*!
     \brief Return of data container of this node

     This method returns the data container of this mortar node where additional
     contact specific quantities/information are stored.

     */
    inline Mortar::NodeDataContainer& mo_data()
    {
      if (modata_ == nullptr) FOUR_C_THROW("No mortar data attached. (node-id = {})", id());
      return *modata_;
    }
    inline const Mortar::NodeDataContainer& mo_data() const
    {
      if (modata_ == nullptr) FOUR_C_THROW("No mortar data attached. (node-id = {})", id());
      return *modata_;
    }

    //@}

    //! @name Evaluation methods

    /*!
     \brief Add a value to the 'D' map of this node

     The 'D' map is later assembled to the D matrix.
     Note that drows_ here is a vector.

     \param row : local dof row id to add to (rowwise)
     \param val : value to be added
     \param col : global dof column id of the value added

     */
    void add_d_value(const int& colnode, const double& val);
    void add_dnts_value(const int& colnode, const double& val);
    void add_dlts_value(const int& colnode, const double& val);
    void add_dltl_value(const int& colnode, const double& val);

    /*!
     \brief Add a value to the 'M' map of this node

     The 'M' map is later assembled to the M matrix.
     Note that mrows_ here is a vector.

     \param row : local dof row id to add to (rowwise)
     \param val : value to be added
     \param col : global dof column id of the value added

     */
    void add_m_value(const int& colnode, const double& val);
    void add_mnts_value(const int& colnode, const double& val);
    void add_mlts_value(const int& colnode, const double& val);
    void add_mltl_value(const int& colnode, const double& val);

    /*!
     \brief Add a value to the 'Mmod' map of this node

     The 'Mmod' map is later assembled to the M matrix.
     Note that mmodrows_ here is a vector.

     \param row : local dof row id to add to (rowwise)
     \param val : value to be added
     \param col : global dof column id of the value added

     */
    void add_mmod_value(const int& colnode, const double& val);

    /*!
     \brief Build nodal normal
     */
    virtual void build_averaged_normal();

    /*!
     \brief Find closest node from given node set and return pointer

     This method will compute the distance of the active node to all
     nodes of the given Core::LinAlg::Map on the given Core::FE::Discretization

     \param intdis (in):         Master Node to project
     \param nodesearchmap (in):  Slave Celement to project on
     \param mindist (out):       Distance to closest node

     */
    virtual Mortar::Node* find_closest_node(const std::shared_ptr<Core::FE::Discretization> intdis,
        const std::shared_ptr<Core::LinAlg::Map> nodesearchmap, double& mindist);

    /*!
     \brief Check if mesh re-initialization for this node was feasible

     This method checks whether mesh distortion due to the applied relocation
     of the current node stays below a certain limit. This check is very
     empirical, of course!

     The mesh distortion status is returned via a boolean parameter.
     (TRUE indicates that mesh distortion is acceptable!)

     \param relocation (in):    relocation distance
     \param limit      (in):    limit in percent of edge length

     */
    virtual bool check_mesh_distortion(double& relocation, const double& limit);

    /*!
     \brief Initializes the data container of the node

     With this function, the container with contact specific quantities/information
     is initialized.

     */
    virtual void initialize_data_container();

    virtual void initialize_poro_data_container()
    {
      FOUR_C_THROW("Shouldn't land here...");
      return;
    };
    virtual void initialize_ehl_data_container()
    {
      FOUR_C_THROW("Shouldn't land here...");
      return;
    };

    /*!
     \brief Resets the data container of the node

     With this function, the container with contact specific quantities/information
     is deleted / reset to nullptr pointer

     */
    virtual void reset_data_container();

    // return dirichlet status at node
    bool* dbc_dofs() { return dbcdofs_; }

    /*!
     \brief Return weighting for nurbs control point (node)

     */
    inline double& nurbs_w() { return nurbsw_; }
    inline double nurbs_w() const { return nurbsw_; }

    //@}

   protected:
    //! true if this node is on slave side of mortar interface
    bool isslave_;

    //! true if this node is on slave side and actually tied
    bool istiedslave_;

    //! true if this node is on slave side boundary
    bool isonbound_;

    //! true if this node is on geometrical edge
    bool isonedge_;

    //! true if this node is on geometrical corner
    bool isoncorner_;

    //! true if this node has a Dirichlet boundary condition applied to at least one of its DOFs
    bool isdbc_;

    //! true at each node that has a Dirichlet boundary condition applied at i-th dof
    bool dbcdofs_[3];

    //! degrees of freedom of this node
    std::vector<int> dofs_;

    //! position in current configuration
    double xspatial_[3];

    //! old displacement (last converged state)
    double uold_[3];

    //! true if feasible projection within search set (slave nodes only)
    bool hasproj_;

    //! true if some integration segment / cell attached (slave nodes only)
    bool hassegment_;

    //! true if this node is detected in active(slip) maps (only for master side)
    bool detected_;

    //! max number of d matrix entries for this node
    int dentries_;

    //! additional information of proc's mortar nodes
    std::shared_ptr<Mortar::NodeDataContainer> modata_;

    //! @name Nurbs specifics
    //! @{

    //! Nurbs control point weighting
    double nurbsw_;

    //! @}

  };  // class Node
}  // namespace Mortar


//! << operator
std::ostream& operator<<(std::ostream& os, const Mortar::Node& mrtrnode);


FOUR_C_NAMESPACE_CLOSE

#endif

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONTACT_FRICTION_NODE_HPP
#define FOUR_C_CONTACT_FRICTION_NODE_HPP

#include "4C_config.hpp"

#include "4C_contact_node.hpp"

#include <set>

FOUR_C_NAMESPACE_OPEN

namespace CONTACT
{
  class FriNodeType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const final { return "FriNodeType"; }

    static FriNodeType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static FriNodeType instance_;
  };

  /*!
   \brief A class containing additional data from frictional contact nodes

   This class contains additional information from frictional contact nodes which are
   are not needed for contact search and therefore are only available on the
   node's processor (ColMap). The class FriNodeDataContainer must be declared
   before the FriNode itself.

   */
  class FriNodeDataContainer
  {
   public:
    //! @name Constructors and destructors and related methods

    /*!
     \brief Standard Constructor

     */
    FriNodeDataContainer();

    /*!
     \brief Destructor

     */
    virtual ~FriNodeDataContainer() = default;
    /*!
     \brief Pack this class so that it can be communicated

     This function packs the datacontainer. This is only called
     when the class has been initialized and the pointer to this
     class exists.

     */
    virtual void pack(Core::Communication::PackBuffer& data) const;

    /*!
     \brief Unpack data from a vector into this class

     This function unpacks the datacontainer. This is only called
     when the class has been initialized and the pointer to this
     class exists.

     */
    virtual void unpack(Core::Communication::UnpackBuffer& buffer);

    //@}

    //! @name Access methods

    /*!
     \brief Return jump per time step (only for slave side!) (length 3)
     */
    virtual inline double* jump() { return jump_; }

    /*!
     \brief Return jump per time step (only for slave side!) (max length 2)
     */
    virtual inline double* jump_var() { return jumpvar_; }

    /*!
     \brief Return friction status of this node (slip=true)
     */
    virtual inline bool& slip() { return slip_; }

    /*!
     \brief Return the old 'D' map (vector) of this node (last converged state)
     */
    virtual inline Core::Gen::Pairedvector<int, double>& get_d_old() { return drowsold_; }

    /*!
     \brief Return the old 'M' map (vector) of this node (last converged state)
     */
    virtual inline std::map<int, double>& get_m_old() { return mrowsold_; }

    /*!
     \brief Return the old 'D' map (vector) of this node (last converged state)
     */
    virtual inline Core::Gen::Pairedvector<int, double>& get_d_old_ltl() { return drowsoldLTL_; }

    /*!
     \brief Return the old 'M' map (vector) of this node (last converged state)
     */
    virtual inline std::map<int, double>& get_m_old_ltl() { return mrowsoldLTL_; }

    /*!
     \brief Return the map with according slave nodes
     */
    virtual inline std::set<int>& get_s_nodes() { return snodes_; }

    /*!
     \brief Return the map with according master nodes
     */
    virtual inline std::set<int>& get_m_nodes() { return mnodes_; }

    /*!
     \brief Return the old map with according master nodes
     */
    virtual inline std::set<int>& get_m_nodes_old() { return mnodesold_; }

    /*!
     \brief Return the 'DerivJump' map (vector) of this node

     These maps contain the directional derivatives of the node's
     relative movement (jump).
     A vector is used because the jump itself is a vector (2 or 3 components).

     */
    virtual inline std::vector<std::map<int, double>>& get_deriv_jump() { return derivjump_; }

    /*!
     \brief Return the 'DerivVarJump' map (vector) of this node

     These maps contain the directional derivatives of the node's
     relative -- object variant-- movement (jump). This Jump is already
     multiplied and linearized with the tangent vectors txi and teta.
     Therefore, the vector has 1/2 components instead of 2/3

     */
    virtual inline std::vector<std::map<int, double>>& get_deriv_var_jump()
    {
      return derivvarjump_;
    }
    /*!
     \brief Return current penalty traction (length 3)
     */
    virtual inline double* traction() { return traction_; }
    /*!
     \brief Return current penalty traction (length 3)
     */
    virtual inline double* traction_ltl() { return tractionLTL_; }
    /*!
     \brief Return contact status of last converged state n (active=true)
     */
    virtual inline bool& slip_old() { return slipold_; }

    /*!
     \brief Return old penalty traction (length 3)
     */
    virtual inline double* tractionold() { return tractionold_; }
    /*!
     \brief Return old penalty traction (length 3)
     */
    virtual inline double* tractionold_ltl() { return tractionoldLTL_; }

    //@}

   protected:
    // don't want = operator and cctor
    FriNodeDataContainer operator=(const FriNodeDataContainer& old) = delete;
    FriNodeDataContainer(const FriNodeDataContainer& old) = delete;

    //! Jump per time step
    double jump_[3];

    //! Variant jump per time step
    double jumpvar_[2];

    /*!
    \brief Flag set to \c true if contact node is slipping

    Node is slipping if contact force reaches friction bound.
    */
    bool slip_;

    /*!
    \brief Flag set to \c true if contact node was slipping at last converged state

    Node is slipping if contact force reaches friction bound.
    */
    bool slipold_;

    //! Nodal rows of old D matrix
    Core::Gen::Pairedvector<int, double> drowsold_;

    //! Nodal rows of old M matrix
    std::map<int, double> mrowsold_;

    //! Nodal rows of old D matrix for line-to-line contact
    Core::Gen::Pairedvector<int, double> drowsoldLTL_;

    //! Nodal rows of old M matrix for line-to-line contact
    std::map<int, double> mrowsoldLTL_;

    //! Nodal set of according slave nodes
    std::set<int> snodes_;

    //! Nodal set of according master nodes
    std::set<int> mnodes_;

    //! nodal set of old according master nodes
    std::set<int> mnodesold_;

    //! Directional derivative of nodal weighted jump vector
    std::vector<std::map<int, double>> derivjump_;

    //! Directional derivative of obj.-variant. nodal weighted jump vector
    std::vector<std::map<int, double>> derivvarjump_;

    //! @name Penalty related quantities
    //!@{

    //! Traction vector of current time step
    double traction_[3];

    //! Traction vector of previous time step
    double tractionold_[3];

    //! Traction vector of current time step for line-to-line contact
    double tractionLTL_[3];

    //! Traction vector of previous time step for line-to-line contact
    double tractionoldLTL_[3];

    //!@}
  };
  // class FriNodeDataContainer

  /*!
   \brief An additional container for wear and TSI with contact specific data

   This class contains additional information to the data container of the
   frictional node. These additional data are needed only for contact
   problems with wear and thermo-structure-interaction problems with contact.

   */
  class FriNodeWearDataContainer
  {
   public:
    //! @name Constructors and destructors and related methods

    /*!
     \brief Standard Constructor

     */
    FriNodeWearDataContainer();

    /*!
     \brief Destructor

     */
    virtual ~FriNodeWearDataContainer() = default;
    //@}

    //! @name Access methods

    virtual void pack(Core::Communication::PackBuffer& data) const;
    virtual void unpack(Core::Communication::UnpackBuffer& buffer);

    /*!
     \brief Return the weighted wear per node (length 1)
     */
    virtual double& weighted_wear() { return weightedwear_; }

    /*!
     \brief Return the delta of the weighted wear per node without wear coefficient (length 1)
     */
    virtual double& delta_weighted_wear() { return deltaweightedwear_; }

    /*!
     \brief Return the 'T' map (vector) of this node
     */
    virtual std::vector<std::map<int, double>>& get_t() { return trows_; }

    /*!
     \brief Return the 'E' map (vector) of this node
     */
    virtual std::vector<std::map<int, double>>& get_e() { return erows_; }

    /*!
     \brief Deriv. w.r.t. displ. of E and T matrix entries for
     this node. This matrices are required for the
     discrete wear condition.
     */
    virtual std::map<int, std::map<int, double>>& get_deriv_tw() { return derivt_; }
    virtual std::map<int, std::map<int, double>>& get_deriv_e() { return derive_; }

    /*!
     \brief Return the 'D2' map (vector) of this node
     */
    virtual std::vector<std::map<int, double>>& get_d2() { return d2rows_; }

    /*!
     \brief Return current discrete wear in step n+1 (only for slave side!) (length 1)
     */
    virtual double* wcurr() { return wcurr_; }

    /*!
     \brief Return accumulated wear for different time scales (only for slave side!) (length 1)
     */
    virtual double* waccu() { return waccu_; }

    /*!
     \brief Return old discrete wear in time step i (only for slave side!) (length 1)
     */
    virtual double* wold() { return wold_; }
    //@}

   protected:
    // don't want = operator and cctor
    FriNodeWearDataContainer operator=(const FriNodeWearDataContainer& old) = delete;
    FriNodeWearDataContainer(const FriNodeWearDataContainer& old) = delete;

    //! @name Wear related quantities
    //!@{

    //! Weighted wear
    double weightedwear_;

    //! delta weighted wear
    double deltaweightedwear_;

    //! Current pv wear value (n+1)
    double wcurr_[1];

    //! Old pv wear value (i) - for paritioned solution scheme
    double wold_[1];

    //! Accumulated pv wear value (i) - for different time scales
    double waccu_[1];

    //!@}

    //! Nodal rows of T matrix
    std::vector<std::map<int, double>> trows_;

    //! Nodal rows of E matrix
    std::vector<std::map<int, double>> erows_;

    //! Nodal rows of master sided D matrix
    std::vector<std::map<int, double>> d2rows_;

    //! Directional derivative of nodal Tw-matrix values
    std::map<int, std::map<int, double>> derivt_;

    //! Directional derivative of nodal E-matrix values
    std::map<int, std::map<int, double>> derive_;
  };
  // class FriNodeWearDataContainer

  /*!
   \brief A class for a frictional contact node derived from CONTACT::Node

   This class represents a finite element node capable of frictional contact.

   */
  class FriNode : public Node
  {
   public:
    //! @name Enums and Friends

    /*!
     \brief The discretization is a friend of FriNode
     */
    friend class Core::FE::Discretization;

    //@}

    //! @name Constructors and destructors and related methods

    /*!
     \brief Standard Constructor

     \param id     (in): A globally unique node id
     \param coords (in): vector of nodal coordinates, length 3
     \param owner  (in): Owner of this node.
     \param dofs   (in): list of global degrees of freedom
     \param isslave(in): flag indicating whether node is slave or master
     \param initactive (in): flag indicating whether initially set to active
     \param friplus (in): ?? (looks like its related to wear)

     */
    FriNode(int id, const std::vector<double>& coords, const int owner,
        const std::vector<int>& dofs, const bool isslave, const bool initactive,
        const bool friplus);

    /*!
     \brief Copy Constructor

     Makes a deep copy of a FriNode

     */
    FriNode(const CONTACT::FriNode& old);

    /*!
     \brief Deep copy the derived class and return pointer to it

     */
    CONTACT::FriNode* clone() const override;


    /*!
     \brief Return unique ParObject id

     every class implementing ParObject needs a unique id defined at the
     top of lib/parobject.H.

     */
    int unique_par_object_id() const override
    {
      return FriNodeType::instance().unique_par_object_id();
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

    //! @name access methods

    /*!
     \brief Print this cnode
     */
    void print(std::ostream& os) const override;

    /*!
     \brief Return of data container of this node

     This method returns the data container of this node where additional
     contact specific quantities/information are stored.

     */
    inline CONTACT::FriNodeDataContainer& fri_data() { return *fridata_; }

    /*!
     \brief Return of additional data container of this node

     This method returns the additional data container of this node where additional
     wear specific quantities/information are stored.

     */
    inline CONTACT::FriNodeWearDataContainer& wear_data() { return *weardata_; }

    /*!
     \brief calculate the apparent coefficient of friction

     in a TSI problem, this is dependent on the slave and master
     temperatures. Since the master temperature is displacement
     dependent (due to the involved projections), it requires
     a consistent linearization (see function below).
     */
    double fr_coeff(const double frcoeff_in) const;

    /*!
     \brief calculate the derivative of apparent coefficient of friction

     in a TSI problem, this is dependent on the slave and master
     temperatures. Since the master temperature is displacement
     dependent (due to the involved projections), it requires
     a consistent linearization (see function below).
     */
    void deriv_fr_coeff_temp(const double frcoeff_in, std::map<int, double>& derivT,
        std::map<int, double>& derivDisp) const;

    //@}

    //! @name Evaluation methods

    /*!
     \brief Add a value to the SNode set of this node

     */
    void add_s_node(int node) override;

    /*!
     \brief Add a value to the 'T' map of this node

     The 'T' map is later assembled to the T matrix.
     Note that drows_ here is a vector.

     \param row : local dof row id to add to (rowwise)
     \param val : value to be added
     \param col : global dof column id of the value added

     */
    void add_t_value(int row, int col, double val);

    /*!
     \brief Add a value to the 'E' map of this node

     The 'E' map is later assembled to the E matrix.
     Note that drows_ here is a vector.

     \param row : local dof row id to add to (rowwise)
     \param val : value to be added
     \param col : global dof column id of the value added

     */
    void add_e_value(int row, int col, double val);

    /*!
     \brief Add a value to the 'WS' map of this node

     The 'WS' map is later assembled to the WS matrix.
     Note that drows_ here is a vector.

     \param row : local dof row id to add to (rowwise)
     \param val : value to be added
     \param col : global dof column id of the value added

     */
    void add_ws_value(int row, int col, double val);
    /*!
     \brief Add a value to the MNode set of this node

     */
    void add_m_node(int node) override;

    /*!
     \brief Add a value to the map of Jump derivatives of this node

     Note that derivjump_ here is a vector.

     \param row : local dof row id to add to (rowise)
     \param val : value to be added
     \param col : global dof column id of the value added

     */
    void add_deriv_jump_value(int row, int col, double val);

    void add_jump_value(double val, int k);

    void add_d2_value(int row, int col, double val);

    /*!
     \brief Set deltawear to value

     */
    void add_delta_weighted_wear_value(double val);

    /*!
     \brief Write nodal entries of D and M to Dold and Mold

     At the end of a time step the nodal entries (vector) of the Mortar
     Matrices D and M are stored to the old ones.

     */
    void store_dm_old();

    /*!
     \brief Write nodal entries of Penalty tractions to old ones

     At the end of a time step the nodal entries (vector) of the Penalty
     tractions are stored to the old ones.

     */
    void store_trac_old();

    /*!
     \brief Initializes the data container of the node

     With this function, the container with contact specific quantities/information
     is initialized.

     */
    void initialize_data_container() override;

    /*!
     \brief Resets the data container of the node

     With this function, the container with contact specific quantities/information
     is deleted / reset to nullptr pointer

     */
    void reset_data_container() override;

    //@}

   protected:
    //! Additional information of proc's friction nodes
    std::shared_ptr<CONTACT::FriNodeDataContainer> fridata_;

    //! Additional information to proc's data container
    std::shared_ptr<CONTACT::FriNodeWearDataContainer> weardata_;

    //! bool for wear
    bool wear_;
  };
  // class FriNode

}  // namespace CONTACT

//// << operator
// std::ostream& operator << (std::ostream& os, const CONTACT::FriNode& cnode);

FOUR_C_NAMESPACE_CLOSE

#endif

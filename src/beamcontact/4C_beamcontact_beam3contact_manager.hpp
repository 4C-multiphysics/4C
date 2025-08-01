// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAMCONTACT_BEAM3CONTACT_MANAGER_HPP
#define FOUR_C_BEAMCONTACT_BEAM3CONTACT_MANAGER_HPP

#include "4C_config.hpp"

#include "4C_beamcontact_beam3contact.hpp"
#include "4C_beamcontact_beam3contactnew.hpp"
#include "4C_beamcontact_beam3tosolidcontact.hpp"
#include "4C_beaminteraction_beam_to_beam_contact_defines.hpp"
#include "4C_contact_element.hpp"
#include "4C_contact_node.hpp"
#include "4C_io.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_vector.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

#include <memory>

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::LinAlg
{
  class SparseMatrix;
}

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::Elements
{
  class Element;
}

class Beam3ContactOctTree;

namespace CONTACT
{
  class Beam3cmanager
  {
   public:
    //! @name Friends

    // no fried classes defined

    //@}

    //! @name Constructors and destructors and related methods

    /*!
    \brief Standard Constructor

    \param discret (in): A discretization containing beam elements

    */
    Beam3cmanager(Core::FE::Discretization& discret, double alphaf);

    /*!
    \brief Destructor

    */
    virtual ~Beam3cmanager() = default;

    //@}

    //! @name Access methods

    /*!
    \brief Print this beam3 contact manager

    */
    virtual void print(std::ostream& os) const;

    /*!
    \brief Get problem discretization

    */
    inline const Core::FE::Discretization& problem_discret() const { return pdiscret_; }

    /*!
    \brief Get beam to solid contact discretization

    */
    inline Core::FE::Discretization& bt_sol_discret() { return *btsoldiscret_; }

    /*!
    \brief Get communicator

    */
    virtual MPI_Comm get_comm() const { return pdiscomm_; }

    /*!
    \brief Get different node or element maps

    */
    inline std::shared_ptr<Core::LinAlg::Map> row_nodes() const { return noderowmap_; }
    inline std::shared_ptr<Core::LinAlg::Map> col_nodes() const { return nodecolmap_; }
    inline std::shared_ptr<Core::LinAlg::Map> full_nodes() const { return nodefullmap_; }
    inline std::shared_ptr<Core::LinAlg::Map> row_elements() const { return elerowmap_; }
    inline std::shared_ptr<Core::LinAlg::Map> col_elements() const { return elecolmap_; }
    inline std::shared_ptr<Core::LinAlg::Map> full_elements() const { return elefullmap_; }
    // template<int numnodes, int numnodalvalues>
    inline const std::vector<std::shared_ptr<Beam3contactinterface>>& pairs() const
    {
      return oldpairs_;
    }

    inline std::shared_ptr<Beam3ContactOctTree> oc_tree() const { return tree_; }

    /*!
    \brief Get list of beam contact input parameters
    */
    inline const Teuchos::ParameterList& beam_contact_parameters() { return sbeamcontact_; }

    /*!
    \brief Get list of general contact input parameters
    */
    inline const Teuchos::ParameterList& general_contact_parameters() { return scontact_; }

    /*!
    \brief Get current constraint norm
    */
    double get_constr_norm() { return constrnorm_; }

    // \brief Get current penalty parameter
    double get_currentpp() { return currentpp_; }

    // \brief Get minimal beam/sphere element radius of discretization
    double get_min_ele_radius() { return mineleradius_; }

    //@}

    //! @name Public evaluation methods

    /*!
    \brief Evaluate beam contact

    First, we search for potential beam element pairs coming into contact.
    For each pair, a temporary Beam3contact object is generated, which handles
    penalty force and stiffness computation. Then, this method calls each beam
    contact pair to compute its contact forces and stiffness. Finally, all entries
    are assembles into global force resdiual and global stiffness matrix.

    */
    void evaluate(Core::LinAlg::SparseMatrix& stiffmatrix, Core::LinAlg::Vector<double>& fres,
        const Core::LinAlg::Vector<double>& disrow, Teuchos::ParameterList timeintparams,
        bool newsti = false, double time = 0.0);

    /*!
    \brief Update beam contact

    Stores fc_ into fcold_ and clears fc_ as needed for generalized alpha time

    */
    void update(
        const Core::LinAlg::Vector<double>& disrow, const int& timestep, const int& newtonstep);

    /*!
    \brief Update constraint norm

    Calculate and print gap values and constraint norm.

    */
    void update_constr_norm();

    /*!
    \brief Shift current normal "normal_" vector to old normal vector "normal_old_"

    The converged normal vector of the last time step is stored as "normal_old_" to serve
    as a reference for the modified gap function definition

    */
    void update_all_pairs();

    /*!
    \brief Print active set

    Print some output data to screen at the end of each time step.
    Interesting values are:
      a) IDs of current pairs and their elements
      b) the residual gap of this pair
      c) the current (augmented part) Lagrange multiplier of this pair
      d) the current element coordinates of the contact point

    NOTE: This method can also be called after each newton-step (e.g. if you want to check
    convergence problems).

    */
    void console_output();

    /*!
    \brief Get total potential energy of penalty approach
    */
    double get_tot_energy() { return totpenaltyenergy_; };

    /*!
    \brief Get total contact work of penalty approach
    */
    double get_tot_work() { return totpenaltywork_; };

    /*!
    \brief Read restart
    */
    void read_restart(Core::IO::DiscretizationReader& reader);

    /*!
    \brief Write restart
    */
    void write_restart(Core::IO::DiscretizationWriter& output);

    //@}

   private:
    // don't want = operator and cctor
    Beam3cmanager operator=(const Beam3cmanager& old);
    Beam3cmanager(const Beam3cmanager& old);

    //! @name member variables

    //! Flag from input file indicating if beam-to-solid mehstying is applied or not (default:
    //! false)
    bool btsolmt_;

    //! Flag from input file indicating if beam-to-solid contact is applied or not (default: false)
    bool btsol_;

    //! number of nodes of applied element type
    int numnodes_;

    //! number of values per node for the applied element type (Reissner beam: numnodalvalues_=1,
    //! Kirchhoff beam: numnodalvalues_=2)
    int numnodalvalues_;

    //! problem discretizaton
    Core::FE::Discretization& pdiscret_;

    //! contact discretization (basically a copy)
    std::shared_ptr<Core::FE::Discretization> btsoldiscret_;

    //! the Comm interface of the problem discretization
    MPI_Comm pdiscomm_;

    //! general map that describes arbitrary dof offset between pdicsret and cdiscret
    std::map<int, int> dofoffsetmap_;

    //! node and element maps
    std::shared_ptr<Core::LinAlg::Map> noderowmap_;
    std::shared_ptr<Core::LinAlg::Map> nodecolmap_;
    std::shared_ptr<Core::LinAlg::Map> nodefullmap_;
    std::shared_ptr<Core::LinAlg::Map> elerowmap_;
    std::shared_ptr<Core::LinAlg::Map> elecolmap_;
    std::shared_ptr<Core::LinAlg::Map> elefullmap_;

    //! occtree for contact search
    std::shared_ptr<Beam3ContactOctTree> tree_;

    //! vector of contact pairs (pairs of elements, which might get in contact)
    std::vector<std::shared_ptr<Beam3contactinterface>> pairs_;
    //! vector of contact pairs of last time step. After update() oldpairs_ is identical with pairs_
    //! until a new time
    // step starts. Therefore oldpairs_ can be used for output at the end of a time step after
    // Update() is called.
    std::vector<std::shared_ptr<Beam3contactinterface>> oldpairs_;

    //! vector of close beam to solid contact pairs (pairs of elements, which might get in contact)
    std::vector<std::shared_ptr<Beam3tosolidcontactinterface>> btsolpairs_;
    //! vector of beam to solid contact pairs of last time step. After update() oldpairs_ is
    //! identical with btsolpairs_ until a
    // new time step starts. Therefore oldbtsolpairs_ can be used for output at the end of a time
    // step after Update() is called.
    std::vector<std::shared_ptr<Beam3tosolidcontactinterface>> oldbtsolpairs_;
    //! total vector of solid contact elements
    std::vector<std::shared_ptr<CONTACT::Element>> solcontacteles_;
    //! total vector of solid contact nodes
    std::vector<std::shared_ptr<CONTACT::Node>> solcontactnodes_;

    //! total vector of solid meshtying elements
    std::vector<std::shared_ptr<Mortar::Element>> solmeshtyingeles_;
    //! total vector of solid meyhtying nodes
    std::vector<std::shared_ptr<Mortar::Node>> solmeshtyingnodes_;

    //! 2D-map with pointers on the contact pairs_. This map is necessary, to call a contact pair
    //! directly by the two element-iD's of the pair.
    // It is not needed at the moment due to the direct neighbour determination in the constructor
    // but may be useful for future operations
    // beam-to-beam pair map
    std::map<std::pair<int, int>, std::shared_ptr<Beam3contactinterface>> contactpairmap_;

    // beam-to-beam pair map of last time step
    std::map<std::pair<int, int>, std::shared_ptr<Beam3contactinterface>> oldcontactpairmap_;

    // beam-to-solid contact pair map
    std::map<std::pair<int, int>, std::shared_ptr<Beam3tosolidcontactinterface>> btsolpairmap_;

    // beam-to-solid pair map of last time step
    std::map<std::pair<int, int>, std::shared_ptr<Beam3tosolidcontactinterface>> oldbtsolpairmap_;

    //! parameter list for beam contact options
    Teuchos::ParameterList sbeamcontact_;

    //! parameter list for general contact options
    Teuchos::ParameterList scontact_;

    //! parameter list for structural dynamic options
    Teuchos::ParameterList sstructdynamic_;

    //! search radius
    double searchradius_;

    //! search radius for spherical intersection
    double sphericalsearchradius_;

    //! additive searchbox increment prescribed in input file
    double searchboxinc_;

    //! minimal beam/sphere radius appearing in discretization
    double mineleradius_;

    //! maximal beam/shpere radius appearing in discretization
    double maxeleradius_;

    //! contact forces of current time step
    std::shared_ptr<Core::LinAlg::Vector<double>> fc_;

    //! contact forces of previous time step (for generalized alpha)
    std::shared_ptr<Core::LinAlg::Vector<double>> fcold_;

    //! contact stiffness matrix of current time step
    std::shared_ptr<Core::LinAlg::SparseMatrix> stiffc_;

    //! time integration parameter (0.0 for statics)
    double alphaf_;

    //! current constraint norm (violation of non-penetration condition)
    double constrnorm_;

    //! current constraint norm (violation of non-penetration condition) of beam-to-solid contact
    //! pairs
    double btsolconstrnorm_;

    //! current BTB penalty parameter (might be modified within augmented Lagrange strategy)
    double currentpp_;

    //! beam-to-solid contact penalty parameter
    double btspp_;

    //! maximal converged absolute gap during the simulation
    double maxtotalsimgap_;
    //! maximal converged absolute gap during the simulation (for individual contact types)
    double maxtotalsimgap_cp_;
    double maxtotalsimgap_gp_;
    double maxtotalsimgap_ep_;

    //! maximal converged relative gap during the simulation
    double maxtotalsimrelgap_;

    //! minimal converged absolute gap during the simulation
    double mintotalsimgap_;
    //! minimal converged absolute gap during the simulation (for individual contact types)
    double mintotalsimgap_cp_;
    double mintotalsimgap_gp_;
    double mintotalsimgap_ep_;

    //! minimal converged relative gap during the simulation
    double mintotalsimrelgap_;

    //! minimal unconverged absolute gap during the simulation
    double mintotalsimunconvgap_;

    //! total contact energy (of elastic penalty forces)
    double totpenaltyenergy_;

    //! total contact work (of penalty forces) -> does not work for restart up to now!
    double totpenaltywork_;

    //! current displacement vector
    std::shared_ptr<Core::LinAlg::Vector<double>> dis_;

    //! displacement vector of last time step
    std::shared_ptr<Core::LinAlg::Vector<double>> dis_old_;

    //! inf-norm of dis_ - dis_old_
    double maxdeltadisp_;

    double totalmaxdeltadisp_;

    //! line charge conditions
    std::vector<const Core::Conditions::Condition*> linechargeconds_;

    //! point charge conditions (rigid sphere)
    std::vector<const Core::Conditions::Condition*> pointchargeconds_;

    // bool indicating if we are in the first time step of a simulation
    bool firststep_;

    // bool indicating if the element type has already been set (only necessary in the first time
    // step with contact)
    bool elementtypeset_;

    // counts the number of output files already written
    int outputcounter_;

    // end time of current time step
    double timen_;

    // accumulated evaluation time of all contact pairs of total simulation time
    double contactevaluationtime_;

    // maximum curvature occurring in one of the potential contact elements
    double global_kappa_max_;

    // output file counter needed for PRINTGAPSOVERLENGTHFILE
    int step_;

    //@}

    //! @name Private evaluation methods

    /*!
    \brief Search contact pairs

    We search pairs of elements that might get in contact. Pairs of elements that are direct
    neighbours, i.e. share one node, will be rejected.
    */
    std::vector<std::vector<Core::Elements::Element*>> brute_force_search(
        std::map<int, Core::LinAlg::Matrix<3, 1>>& currentpositions, const double searchradius,
        const double sphericalsearchradius);

    /*!
    \brief Compute the search radius

    The search radius will be computed by analyzing the chracteristic length of each
    element. To guarantee, that each possible contact pair will be detected some
    empiric criterion will define the search radius, taking into account:

      a) the maximum element radius
      b) the maximum element length

    These two characteric lengths will be compared, the larger one is the characteristic
    length for this processor. Then via a communication among all procs the largest
    characteristic length in the whole discretization is found. Using this global
    characteristic length, we can compute a searchradius by multiplying with a constant factor.
    This method is called only once at the beginning of the simulation. If axial deformation
    of beam elements was high, it would have to be called more often!

    */
    void compute_search_radius();

    /*!
    \brief Get maximum element radius

    Finds minimum and maximum element radius in the whole discretization for circular cross
    sections. Stores the values in corresponding class variables.

    */
    void set_min_max_ele_radius();

    /*
    \brief Test if element midpoints are close (spherical bounding box intersection)
    */
    bool close_midpoint_distance(const Core::Elements::Element* ele1,
        const Core::Elements::Element* ele2,
        std::map<int, Core::LinAlg::Matrix<3, 1>>& currentpositions,
        const double sphericalsearchradius);

    /*!
    \brief Set the member variables numnodes_ and numnodalvalues depending on the element type
    handed in!
    */
    void set_element_type_and_distype(Core::Elements::Element* ele1);

    /*!
    \brief Check, if pair with given element IDs is already existing in the vector pairs_!
    */
    bool pair_already_existing(int currid1, int currid2);

    /*!
    \brief Get maximum element length

    Finds maximum element radius in the whole discretization for circular cross
    sections. Stores the maximum radius to 'max_ele_length'. For higher-order-elements
    an approximation of the true element length is introduced, as only the direct distance
    of the two end nodes is computed. Yet, this is assumed to be accurate enough.

    */
    void get_max_ele_length(double& maxelelength);

    /*!
    \brief Compute rotation matrix R from given angle theta in 3D

    This function computes from a three dimensional rotation angle theta
    (which gives rotation axis and absolute value of rotation angle) the related
    rotation matrix R. Note that theta is given in radiant.

    */
    void transform_angle_to_triad(
        Core::LinAlg::SerialDenseVector& theta, Core::LinAlg::SerialDenseMatrix& R);

    /*!
    \brief Compute spin

    Compute spin matrix according to Crisfield Vol. 2, equation (16.8)

    */
    void compute_spin(
        Core::LinAlg::SerialDenseMatrix& spin, Core::LinAlg::SerialDenseVector& rotationangle);

    /*!
    \brief Shift map of displacement vector

    */
    void shift_dis_map(
        const Core::LinAlg::Vector<double>& disrow, Core::LinAlg::Vector<double>& disccol);


    /** \brief set up the discretization btsoldiscret_ to be used within beam contact manager
     *
     */
    void init_beam_contact_discret();

    /*!
    \brief Store current displacement state in currentpositions

    */
    void set_current_positions(std::map<int, Core::LinAlg::Matrix<3, 1>>& currentpositions,
        const Core::LinAlg::Vector<double>& disccol);

    /*!
    \brief Set displacement state on contact element pair level

    The contact element pairs are updated with these current positions and also with
    the current tangent vectors in case of Kirchhoff beam elements
    */
    void set_state(std::map<int, Core::LinAlg::Matrix<3, 1>>& currentpositions,
        const Core::LinAlg::Vector<double>& disccol);

    /*!
    \brief Evaluate all pairs stored in the different pairs vectors (BTB, BTSPH, BTSOL; contact)

    */
    void evaluate_all_pairs(Teuchos::ParameterList timeintparams);

    /*!
    \brief Sort found element pairs and fill vectors of contact pairs (BTB, BTSOL and BTSPH)

    */
    void fill_contact_pairs_vectors(
        const std::vector<std::vector<Core::Elements::Element*>> elementpairs);

  };  // class Beam3cmanager
}  // namespace CONTACT

FOUR_C_NAMESPACE_CLOSE

#endif

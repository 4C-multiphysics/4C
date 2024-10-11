/*----------------------------------------------------------------------*/
/*! \file

\brief contact manager for contact in a beam3 discretization

\level 2

*/
/*----------------------------------------------------------------------*/

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

#include <Epetra_Comm.h>
#include <Teuchos_RCP.hpp>
#include <Teuchos_StandardParameterEntryValidators.hpp>

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
    virtual const Epetra_Comm& get_comm() const { return pdiscomm_; }

    /*!
    \brief Get different node or element maps

    */
    inline Teuchos::RCP<Epetra_Map> row_nodes() const { return noderowmap_; }
    inline Teuchos::RCP<Epetra_Map> col_nodes() const { return nodecolmap_; }
    inline Teuchos::RCP<Epetra_Map> full_nodes() const { return nodefullmap_; }
    inline Teuchos::RCP<Epetra_Map> row_elements() const { return elerowmap_; }
    inline Teuchos::RCP<Epetra_Map> col_elements() const { return elecolmap_; }
    inline Teuchos::RCP<Epetra_Map> full_elements() const { return elefullmap_; }
    // template<int numnodes, int numnodalvalues>
    inline const std::vector<Teuchos::RCP<Beam3contactinterface>>& pairs() const
    {
      return oldpairs_;
    }

    inline Teuchos::RCP<Beam3ContactOctTree> oc_tree() const { return tree_; }

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
    integration scheme at the end of each time step. ASCII output files for
    visualization in GMSH will be written. Also some output to screen is done.

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
    \brief Create output files for GMSH visualization

    Create ASCII-files to visualize beams with GMSH. The circular cross section will
    be approximated by prisms, which are rotated around the element's axis. This output
    method only works safely for the serial case, the parallel case is not yet implemented!

    */
    void gmsh_output(const Core::LinAlg::Vector<double>& disrow, const int& timestep,
        const int& newtonstep, bool endoftimestep = false);

    /*!
    \brief Print active set

    Print some output data to screen at the end of each time step.
    Interesting values are:
      a) IDs of current pairs and their elements
      b) the residual gap of this pair
      c) the current (augmented part) Lagrange multiplier of this pair
      d) the current element coordinates of the contact point

    NOTE: This method can also be called after each newton-step (e.g. if you want to check
    convergence problems). In this case, you have to uncomment the GMSHNEWTONSTEP preprocessor
    flag in 'beam3contact_defines.h'.

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

    //! Flag from input file indicating if beam-to-solid potential-based interaction is applied or
    //! not (default: false)
    bool potbtsol_;

    //! Flag from input file indicating if beam-to-sphere potential-based interaction is applied or
    //! not (default: false)
    bool potbtsph_;

    //! number of nodes of applied element type
    int numnodes_;

    //! number of values per node for the applied element type (Reissner beam: numnodalvalues_=1,
    //! Kirchhoff beam: numnodalvalues_=2)
    int numnodalvalues_;

    //! problem discretizaton
    Core::FE::Discretization& pdiscret_;

    //! contact discretization (basically a copy)
    Teuchos::RCP<Core::FE::Discretization> btsoldiscret_;

    //! the Comm interface of the problem discretization
    const Epetra_Comm& pdiscomm_;

    //! general map that describes arbitrary dof offset between pdicsret and cdiscret
    std::map<int, int> dofoffsetmap_;

    //! node and element maps
    Teuchos::RCP<Epetra_Map> noderowmap_;
    Teuchos::RCP<Epetra_Map> nodecolmap_;
    Teuchos::RCP<Epetra_Map> nodefullmap_;
    Teuchos::RCP<Epetra_Map> elerowmap_;
    Teuchos::RCP<Epetra_Map> elecolmap_;
    Teuchos::RCP<Epetra_Map> elefullmap_;

    //! occtree for contact search
    Teuchos::RCP<Beam3ContactOctTree> tree_;

    //! occtree for search of potential-based interaction pairs
    Teuchos::RCP<Beam3ContactOctTree> pottree_;

    //! vector of contact pairs (pairs of elements, which might get in contact)
    std::vector<Teuchos::RCP<Beam3contactinterface>> pairs_;
    //! vector of contact pairs of last time step. After update() oldpairs_ is identical with pairs_
    //! until a new time
    // step starts. Therefore oldpairs_ can be used for output at the end of a time step after
    // Upadte() is called.
    std::vector<Teuchos::RCP<Beam3contactinterface>> oldpairs_;

    //! vector of close beam to solid contact pairs (pairs of elements, which might get in contact)
    std::vector<Teuchos::RCP<Beam3tosolidcontactinterface>> btsolpairs_;
    //! vector of beam to solid contact pairs of last time step. After update() oldpairs_ is
    //! identical with btsolpairs_ until a
    // new time step starts. Therefore oldbtsolpairs_ can be used for output at the end of a time
    // step after Upadte() is called.
    std::vector<Teuchos::RCP<Beam3tosolidcontactinterface>> oldbtsolpairs_;
    //! total vector of solid contact elements
    std::vector<Teuchos::RCP<CONTACT::Element>> solcontacteles_;
    //! total vector of solid contact nodes
    std::vector<Teuchos::RCP<CONTACT::Node>> solcontactnodes_;

    //! total vector of solid meshtying elements
    std::vector<Teuchos::RCP<Mortar::Element>> solmeshtyingeles_;
    //! total vector of solid meyhtying nodes
    std::vector<Teuchos::RCP<Mortar::Node>> solmeshtyingnodes_;

    //! 2D-map with pointers on the contact pairs_. This map is necessary, to call a contact pair
    //! directly by the two element-iD's of the pair.
    // It is not needed at the moment due to the direct neigbour determination in the constructor
    // but may be useful for future operations
    // beam-to-beam pair map
    std::map<std::pair<int, int>, Teuchos::RCP<Beam3contactinterface>> contactpairmap_;

    // beam-to-beam pair map of last time step
    std::map<std::pair<int, int>, Teuchos::RCP<Beam3contactinterface>> oldcontactpairmap_;

    // beam-to-solid contact pair map
    std::map<std::pair<int, int>, Teuchos::RCP<Beam3tosolidcontactinterface>> btsolpairmap_;

    // beam-to-solid pair map of last time step
    std::map<std::pair<int, int>, Teuchos::RCP<Beam3tosolidcontactinterface>> oldbtsolpairmap_;

    //! parameter list for beam contact options
    Teuchos::ParameterList sbeamcontact_;

    //! parameter list for beam potential interaction options
    Teuchos::ParameterList sbeampotential_;

    //! parameter list for general contact options
    Teuchos::ParameterList scontact_;

    //! parameter list for structural dynamic options
    Teuchos::ParameterList sstructdynamic_;

    //! search radius
    double searchradius_;

    //! search radius for spherical intersection
    double sphericalsearchradius_;

    //! search radius for potential-based interactions
    double searchradiuspot_;

    //! additive searchbox increment prescribed in input file
    double searchboxinc_;

    //! minimal beam/sphere radius appearing in discretization
    double mineleradius_;

    //! maximal beam/shpere radius appearing in discretization
    double maxeleradius_;

    //! contact forces of current time step
    Teuchos::RCP<Core::LinAlg::Vector<double>> fc_;

    //! contact forces of previous time step (for generalized alpha)
    Teuchos::RCP<Core::LinAlg::Vector<double>> fcold_;

    //! contact stiffness matrix of current time step
    Teuchos::RCP<Core::LinAlg::SparseMatrix> stiffc_;

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
    Teuchos::RCP<Core::LinAlg::Vector<double>> dis_;

    //! displacement vector of last time step
    Teuchos::RCP<Core::LinAlg::Vector<double>> dis_old_;

    //! inf-norm of dis_ - dis_old_
    double maxdeltadisp_;

    double totalmaxdeltadisp_;

    //! parameters of the potential law to be applied: Phi(r)~ \sum_i (k_i * r^(-m_i))
    Teuchos::RCP<std::vector<double>> ki_;
    Teuchos::RCP<std::vector<double>> mi_;

    //! line charge conditions
    std::vector<Core::Conditions::Condition*> linechargeconds_;

    //! point charge conditions (rigid sphere)
    std::vector<Core::Conditions::Condition*> pointchargeconds_;

    // bool indicating if we are in the first time step of a simulation
    bool firststep_;

    // bool indicating if the element type has already been set (only necessary in the first time
    // step with contact)
    bool elementtypeset_;

    // counts the number of gmsh-output files already written
    int outputcounter_;

    // end time of current time step
    double timen_;

    // accumulated evaluation time of all contact pairs of total simulation time
    double contactevaluationtime_;

    // maximum curvature occuring in one of the potential contact elements
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
    sections. Stores the values in corresponing class variables.

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
    \brief Check, if pair with given element IDs is allready existing in the vector pairs_!
    */
    bool pair_allready_existing(int currid1, int currid2);

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
     *  \author grill
     *  \date 05/16 */
    void init_beam_contact_discret();

    /*!
    \brief Store current displacment state in currentpositions

    */
    void set_current_positions(std::map<int, Core::LinAlg::Matrix<3, 1>>& currentpositions,
        const Core::LinAlg::Vector<double>& disccol);

    /*!
    \brief Set displacment state on contact element pair level

    The contact element pairs are updated with these current positions and also with
    the current tangent vectors in case of Kirchhoff beam elements
    */
    void set_state(std::map<int, Core::LinAlg::Matrix<3, 1>>& currentpositions,
        const Core::LinAlg::Vector<double>& disccol);

    /*!
    \brief Evaluate all pairs stored in the different pairs vectors (BTB, BTSPH, BTSOL; contact and
    potential)

    */
    void evaluate_all_pairs(Teuchos::ParameterList timeintparams);

    /*!
    \brief Sort found element pairs and fill vectors of contact pairs (BTB, BTSOL and BTSPH)

    */
    void fill_contact_pairs_vectors(
        const std::vector<std::vector<Core::Elements::Element*>> elementpairs);

    /*!
    \brief Sort found element pairs and fill vectors of potential pairs (BTB, BTSOL and BTSPH)

    */
    void fill_potential_pairs_vectors(
        const std::vector<std::vector<Core::Elements::Element*>> elementpairs);

    /*!
    \brief Compute coordinates for GMSH-Output for two-noded-elements

    */
    void gmsh_2_noded(const int& n, const Core::LinAlg::SerialDenseMatrix& coord,
        const Core::Elements::Element* thisele, std::stringstream& gmshfilecontent);

    /*!
    \brief Compute coordinates for GMSH-Output for three-noded-elements

    */
    void gmsh_3_noded(const int& n, const Core::LinAlg::SerialDenseMatrix& allcoord,
        const Core::Elements::Element* thisele, std::stringstream& gmshfilecontent);

    /*!
    \brief Compute coordinates for GMSH-Output for four-noded-elements

    */
    void gmsh_4_noded(const int& n, const Core::LinAlg::SerialDenseMatrix& allcoord,
        const Core::Elements::Element* thisele, std::stringstream& gmshfilecontent);

    /*!
    \brief Compute coordinates for GMSH-Output for N-noded-elements
    */
    void gmsh_n_noded(const int& n, int& n_axial, const Core::LinAlg::SerialDenseMatrix& allcoord,
        const Core::Elements::Element* thisele, std::stringstream& gmshfilecontent);

    /*!
    \brief Compute coordinates for GMSH-Line-Output for N-noded-elements
    */
    void gmsh_n_noded_line(const int& n, const int& n_axial,
        const Core::LinAlg::SerialDenseMatrix& allcoord, const Core::Elements::Element* thisele,
        std::stringstream& gmshfilecontent);

    /*!
    \brief Compute coordinates for GMSH-Output of rigid sphere

    */
    void gmsh_sphere(const Core::LinAlg::SerialDenseMatrix& coord,
        const Core::Elements::Element* thisele, std::stringstream& gmshfilecontent);

    /*!
    \brief Print Gmsh Triangle to stringstream by specifying the vertices

    */
    void print_gmsh_triangle_to_stream(std::stringstream& gmshfilecontent,
        const std::vector<std::vector<double>>& vertexlist, int i, int j, int k, double color,
        const double centercoord[]);
    /*!
    \brief Refine the icosphere by subdivision of each face in four new triangles

    */
    void gmsh_refine_icosphere(std::vector<std::vector<double>>& vertexlist,
        std::vector<std::vector<int>>& facelist, double radius);

    //**********************Begin: Output-Methods for BTS-Contact****************************
    /*!
    \brief GMSH-Surface-Output for solid elements
    */
    void gmsh_solid(const Core::Elements::Element* element,
        const Core::LinAlg::Vector<double>& disrow, std::stringstream& gmshfilecontent);

    /*!
    \brief GMSH-Surface-Output for solid surfaces
    */
    void gmsh_solid_surface_element_numbers(const Core::Elements::Element* element,
        const Core::LinAlg::Vector<double>& disrow, std::stringstream& gmshfilecontent);

    /*!
    \brief Get color of solid element surfaces for GMSH-Output
    */
    void gmsh_get_surf_color(const Core::Elements::Element* element, const int& n_surfNodes,
        const int surfNodes[6][9], double surfColor[6]);

    /*!
    \brief GMSH-Surface-Output for 4-noded quadrangle (SQ)
    */
    void gmsh_sq(
        const double coords[3][4], const double color[4], std::stringstream& gmshfilecontent);

    /*!
    \brief GMSH-Surface-Output for 3-noded triangle (ST)
    */
    void gmsh_st(
        const double coords[3][3], const double color[3], std::stringstream& gmshfilecontent);
    //**********************End: Output-Methods for BTS-Contact****************************

    //@}

  };  // class Beam3cmanager
}  // namespace CONTACT

FOUR_C_NAMESPACE_CLOSE

#endif

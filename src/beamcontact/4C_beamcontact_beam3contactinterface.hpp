/*----------------------------------------------------------------------*/
/*! \file

\brief interface class for templated classes beam3contact and beam3contactnew

\level 2

*/
/*----------------------------------------------------------------------*/
#ifndef FOUR_C_BEAMCONTACT_BEAM3CONTACTINTERFACE_HPP
#define FOUR_C_BEAMCONTACT_BEAM3CONTACTINTERFACE_HPP

#include "4C_config.hpp"

#include "4C_beaminteraction_beam_to_beam_contact_defines.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_sparsematrix.hpp"

FOUR_C_NAMESPACE_OPEN


namespace CONTACT
{
  /*!
   \brief interface class for templated classes beam3contact and beam3contactnew

   */

  class Beam3contactinterface
  {
   public:
    //! @name Friends
    // no friend classes defined
    //@}

    //! @name Constructors and destructors and related methods
    /*!
    \brief Standard Constructor
    \param pdiscret (in): the problem discretization
    \param cdiscret (in): the beam contact discretization
    \param dofoffset (in): offset of dof between pdiscret and cdiscret
    \param element1 (in): first element of contact pair
    \param element2 (in): second element of contact pair
    \param ele1pos (in): nodal coordinates of first element
    \param ele2pos (in): nodal coordinates of second element
    */
    Beam3contactinterface() {}

    /*!
    \brief Destructor
    */
    virtual ~Beam3contactinterface() = default;
    //@}

    //! @name Access methods
    /*!
    \brief Get problem discretization
    */
    virtual const Core::FE::Discretization& problem_discret() const = 0;

    /*!
    \brief Get beam contact discretization
    */
    virtual const Core::FE::Discretization& contact_discret() const = 0;

    /*!
    \brief Get offset of dofs between cdiscret and pdiscret
    */
    virtual const std::map<int, int>& dof_offset() const = 0;

    /*!
    \brief Get first element
    */
    virtual const Core::Elements::Element* element1() = 0;
    // inline const Core::Elements::Element* Element1() { return element1_;};

    /*!
    \brief Get first element
    */
    virtual const Core::Elements::Element* element2() = 0;

    /*!
    \brief Get number of contact points on this element pair
    */
    virtual int get_num_cps() = 0;
    virtual int get_num_gps() = 0;
    virtual int get_num_eps() = 0;

    /*!
    \brief Get vector of type declarations (0=closest point contact, 1=gauss point contact, 2= end
    point contact) of all contact pairs
    */
    virtual std::vector<int> get_contact_type() = 0;

    /*!
    \brief Get vector of all gaps of this contact pair
    */
    virtual std::vector<double> get_gap() = 0;

    /*!
    \brief Get vector of all contact forces of this contact pair
    */
    virtual std::vector<double> get_contact_force() = 0;

    /*!
    \brief Get vector of all contact angles of this contact pair
    */
    virtual std::vector<double> get_contact_angle() = 0;

    /*!
    \brief Get vector of all closest points of this contact pair
    */
    virtual std::vector<std::pair<double, double>> get_closest_point() = 0;

    /*!
    \brief Return number of individual contact segments on element pair
    */
    virtual std::pair<int, int> get_num_segments() = 0;

    /*!
    \brief Return ids of active segments
    */
    virtual std::vector<std::pair<int, int>> get_segment_ids() = 0;

    /*!
    \brief Get flag ndicating whether contact is active (true) or inactive (false)
    */
    virtual bool get_contact_flag() = 0;

    /*!
    \brief Get coordinates of contact point of element1 and element2
    */
    virtual std::vector<Core::LinAlg::Matrix<3, 1>> get_x1() = 0;

    virtual std::vector<Core::LinAlg::Matrix<3, 1>> get_x2() = 0;

    virtual Core::LinAlg::SerialDenseVector get_normal() = 0;

    virtual Core::LinAlg::Matrix<3, 1, TYPE>* get_normal_old() = 0;

    /*!
      \Check, if there is a difference between the result of the new and old gap definition, i.e. if
      the beams centerlines have already crossed or not.
    */
    virtual bool get_new_gap_status() = 0;

    /*!
      \Get energy of penalty contact.
    */
    virtual double get_energy() = 0;

    /*!
      \Get energy of perp penalty contact without transition factor contribution.
    */
    virtual double get_unscaled_perp_energy() = 0;

    /*!
      \Get energy of parallel penalty contact without transition factor contribution.
    */
    virtual double get_unscaled_parallel_energy() = 0;

    virtual bool first_time_step() = 0;

    /*!
    \brief Get flag indicating whether the nodal values of one element had been shifted due to r1=r2
    */
    virtual bool get_shift_status() = 0;
    //@}

    /** \brief print this beam contact element pair to screen
     *
     *  \author grill
     *  \date 05/16 */
    virtual void print() const = 0;


    //! @name Public evaluation methods
    /*!
    \brief Evaluate this contact element pair
    */
    virtual bool evaluate(Core::LinAlg::SparseMatrix& stiffmatrix, Core::LinAlg::Vector& fint,
        const double& pp,
        std::map<std::pair<int, int>, Teuchos::RCP<Beam3contactinterface>>& contactpairmap,
        Teuchos::ParameterList& timeintparams, bool fdcheck = false) = 0;

    //! return appropriate internal implementation class (acts as a simple factory)
    static Teuchos::RCP<Beam3contactinterface> impl(const int numnodes, const int numnodalvalues,
        const Core::FE::Discretization& pdiscret, const Core::FE::Discretization& cdiscret,
        const std::map<int, int>& dofoffsetmap, Core::Elements::Element* element1,
        Core::Elements::Element* element2, Teuchos::ParameterList& beamcontactparams);

    /*!
    \brief Change the sign of the normal vector: This has to be done at the end of a time step when
    the remainig penetration is larger that the sum of the beam radii (R1+R2). Otherwise, the beams
    could cross in the next time step when the new gap function definition (ngf_=true) for slender
    beams is applied!
    */
    virtual void invert_normal() = 0;

    /*!
      \brief Update of class variables at the end of a time step
    */
    virtual void update_class_variables_step() = 0;

    /*
    \brief Update nodal coordinates of both elements at the beginning of a new time step!
    */
    virtual void update_ele_pos(Core::LinAlg::SerialDenseMatrix& newele1pos,
        Core::LinAlg::SerialDenseMatrix& newele2pos) = 0;

    /*
    \brief Update interpolated nodal tangents for tangent smoothing
    */
    virtual void update_ele_smooth_tangents(
        std::map<int, Core::LinAlg::Matrix<3, 1>>& currentpositions) = 0;

  };  // class Beam3contactinterface
}  // namespace CONTACT

FOUR_C_NAMESPACE_CLOSE

#endif

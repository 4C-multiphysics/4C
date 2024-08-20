/*----------------------------------------------------------------------------*/
/*! \file

\brief One beam and solid contact pair (two elements)

\level 3

*/
/*----------------------------------------------------------------------------*/

#ifndef FOUR_C_BEAMCONTACT_BEAM3TOSOLIDCONTACT_HPP
#define FOUR_C_BEAMCONTACT_BEAM3TOSOLIDCONTACT_HPP

#include "4C_config.hpp"

#include "4C_beaminteraction_beam_to_beam_contact_defines.hpp"
#include "4C_beaminteraction_beam_to_beam_contact_tangentsmoothing.hpp"
#include "4C_beaminteraction_beam_to_beam_contact_utils.hpp"
#include "4C_fem_general_utils_local_connectivity_matrices.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_utils_fad.hpp"

#include <Sacado.hpp>

FOUR_C_NAMESPACE_OPEN

namespace CONTACT
{
  /*!
   \brief contact element for contact between a 3D beam end a 2D surface (belonging to a 3D solid)
   element

   */

  class Beam3tosolidcontactinterface
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
    Beam3tosolidcontactinterface() {}

    /*!
    \brief Destructor
    */
    virtual ~Beam3tosolidcontactinterface() = default;
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
    \brief Get gap of this contact pair
    */
    virtual double get_gap() = 0;

    /*!
    \brief Get flag ndicating whether contact is active (true) or inactive (false)
    */
    virtual bool get_contact_flag() = 0;

    /*!
    \brief Get coordinates of contact point of element1 and element2
    */
    virtual Core::LinAlg::SerialDenseVector get_x1() = 0;

    virtual Core::LinAlg::SerialDenseVector get_x2() = 0;

    /*!
      \Check, if there is a difference between the result of the new and old gap definition, i.e. if
      the beams centerlines have already crossed or not.
    */
    virtual bool get_new_gap_status() = 0;

    /*!
    \brief Get flag indicating whether the nodal values of one element had been shifted due to r1=r2
    */
    virtual bool get_shift_status() = 0;
    //@}


    //! @name Public evaluation methods
    /*!
    \brief Evaluate this contact element pair
    */
    virtual bool evaluate(
        Core::LinAlg::SparseMatrix& stiffmatrix, Epetra_Vector& fint, const double& pp) = 0;

    //! return appropriate internal implementation class (acts as a simple factory)
    static Teuchos::RCP<Beam3tosolidcontactinterface> impl(const int numnodessol,
        const int numnodes, const int numnodalvalues, const Core::FE::Discretization& pdiscret,
        const Core::FE::Discretization& cdiscret, const std::map<int, int>& dofoffsetmap,
        Core::Elements::Element* element1, Core::Elements::Element* element2,
        Teuchos::ParameterList beamcontactparams);

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

    /*!
      \brief Shift current normal vector to old normal vector at the end of a time step. This is
      necessary when the new gap function definition (ngf_=true) for slender beams is applied!
    */
    virtual void shift_normal() = 0;

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

    //! brief Struct for debug data in Gmsh
    struct GmshDebugPoint
    {
      Core::LinAlg::Matrix<3, 1, double> r1;
      Core::LinAlg::Matrix<3, 1, double> x2;
      Core::LinAlg::Matrix<3, 1, double> n2;
      double gap;
      double fp;
      int type;
    };

    /*
    \ brief Get debug data for Gmsh
     */
    virtual std::vector<GmshDebugPoint> get_gmsh_debug_points() = 0;


  };  // class Beam3tosolidcontactinterface



  template <const int numnodessol, const int numnodes, const int numnodalvalues>
  class Beam3tosolidcontact : public Beam3tosolidcontactinterface
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
    Beam3tosolidcontact(const Core::FE::Discretization& pdiscret,
        const Core::FE::Discretization& cdiscret, const std::map<int, int>& dofoffsetmap,
        Core::Elements::Element* element1, Core::Elements::Element* element2,
        Teuchos::ParameterList beamcontactparams);

    /*!
    \brief Copy Constructor
    Makes a deep copy of this contact element pair
    */
    Beam3tosolidcontact(const Beam3tosolidcontact& old);


    //@}


    //! @name Access methods
    /*!
    \brief Get problem discretization
    */
    inline const Core::FE::Discretization& problem_discret() const override { return pdiscret_; };

    /*!
    \brief Get beam contact discretization
    */
    inline const Core::FE::Discretization& contact_discret() const override { return cdiscret_; };

    /*!
    \brief Get offset of dofs between cdiscret and pdiscret
    */
    inline const std::map<int, int>& dof_offset() const override { return dofoffsetmap_; };

    /*!
    \brief Get first element
    */
    inline const Core::Elements::Element* element1() override { return element1_; };

    /*!
    \brief Get first element
    */
    inline const Core::Elements::Element* element2() override { return element2_; };

    /*!
    \brief Get gap of this contact pair
    */
    double get_gap() override { return Core::FADUtils::cast_to_double(gap_); };

    /*!
    \brief Get flag indicating whether contact is active (true) or inactive (false)
    */
    bool get_contact_flag() override { return contactflag_; };

    /*!
    \brief Get coordinates of contact point of element1 and element2
    */
    Core::LinAlg::SerialDenseVector get_x1() override
    {
      Core::LinAlg::SerialDenseVector r1;
      r1.resize(3);
      for (int i = 0; i < 3; i++) r1(i) = Core::FADUtils::cast_to_double(r1_(i));

      return r1;
    };

    Core::LinAlg::SerialDenseVector get_x2() override
    {
      Core::LinAlg::SerialDenseVector r2;
      r2.resize(3);
      for (int i = 0; i < 3; i++) r2(i) = Core::FADUtils::cast_to_double(r2_(i));

      return r2;
    };

    /*!
    \brief Get flag indicating whether the nodal values of one element had been shifted due to r1=r2
    */
    bool get_shift_status() override { return shiftnodalvalues_; };

    /*!
      \Check, if there is a difference between the result of the new and old gap definition, i.e. if
      the beams centerlines have already crossed or not.
    */
    bool get_new_gap_status() override;
    //@}


    //! @name Public evaluation methods
    /*!
    \brief Evaluate this contact element pair
    */
    bool evaluate(
        Core::LinAlg::SparseMatrix& stiffmatrix, Epetra_Vector& fint, const double& pp) override;

    /*!
    \brief Change the sign of the normal vector: This has to be done at the end of a time step when
    the remainig penetration is larger that the sum of the beam radii (R1+R2). Otherwise, the beams
    could cross in the next time step when the new gap function definition (ngf_=true) for slender
    beams is applied!
    */
    void invert_normal() override;

    /*!
      \brief Update of class variables at the end of a time step
    */
    void update_class_variables_step() override;

    /*!
      \brief Shift current normal vector to old normal vector at the end of a time step. This is
      necessary when the new gap function definition (ngf_=true) for slender beams is applied!
    */
    void shift_normal() override;

    /*
    \brief Update nodal coordinates of both elements at the beginning of a new time step!
    */
    void update_ele_pos(Core::LinAlg::SerialDenseMatrix& newele1pos,
        Core::LinAlg::SerialDenseMatrix& newele2pos) override;

    /*
    \brief Update interpolated nodal tangents for tangent smoothing
    */
    void update_ele_smooth_tangents(
        std::map<int, Core::LinAlg::Matrix<3, 1>>& currentpositions) override;

    /*!
    \brief Get debugging data at Gauss points for Gmsh
    */
    std::vector<GmshDebugPoint> get_gmsh_debug_points() override { return gmsh_debug_points_; };


    //@}
   private:
    //! @name member variables

    //! reference to problem discretization
    const Core::FE::Discretization& pdiscret_;

    //! reference to beam contact discretization
    const Core::FE::Discretization& cdiscret_;

    //! dof offset between pdiscret and cdiscret
    const std::map<int, int>& dofoffsetmap_;

    //! first element of contact pair
    Core::Elements::Element* element1_;

    //! second element of contact pair
    Core::Elements::Element* element2_;

    //! current node coordinates of the two elements
    Core::LinAlg::Matrix<3 * numnodes * numnodalvalues, 1, TYPEBTS> ele1pos_;
    Core::LinAlg::Matrix<3 * numnodessol, 1, TYPEBTS> ele2pos_;

    //! variable to check if old or modified gap function
    bool ngf_;

    //! variable to check which smoothing type should be applied
    int smoothing_;

    //! sgn(normal*normal_old)
    double sgn_;

    //! variable to identify first call of a pair (for initializing)
    bool firstcall_;

    //! gap function according to original (ngf_==false) or modified (ngf_==true) definition
    TYPEBTS gap_;

    //! gap function according to original definition
    TYPEBTS gap_original_;

    //! flag indicating contact (active/inactive)
    bool contactflag_;

    //! flag indicating if elements are collinear or not
    bool elementscolinear_;

    //! flag indicating if elements share the same contact point, i.e. r1_=r2_ --> evaluation not
    //! possible
    bool elementscrossing_;

    //! flag indicating if the element nodal positions have been shifted in order to avoid r1_=r2_
    bool shiftnodalvalues_;

    //! coordinates of contact points
    Core::LinAlg::Matrix<3, 1, TYPEBTS> r1_;
    Core::LinAlg::Matrix<3, 1, TYPEBTS> r2_;

    //! parameter values of contact point
    TYPEBTS xi1_;
    TYPEBTS xi2_;

    //! Vector containing pairs of unit distance vector nD and beam parameter eta of current time
    //! step
    std::vector<std::pair<TYPEBTS, Core::LinAlg::Matrix<3, 1, TYPEBTS>>> normalsets_;

    //! Vector containing pairs of unit distance vector nD and beam parameter eta of last time step
    std::vector<std::pair<TYPEBTS, Core::LinAlg::Matrix<3, 1, TYPEBTS>>> normalsets_old_;

    //! neighbor elements of element 1
    Teuchos::RCP<BEAMINTERACTION::B3CNeighbor> neighbors1_;

    //! neighbor elements of element 2
    Teuchos::RCP<BEAMINTERACTION::B3CNeighbor> neighbors2_;

    //! averaged nodal tangents, necessary for smoothed tangent fields of C^0 Reissner beams
    Core::LinAlg::Matrix<3 * numnodes, 1> nodaltangentssmooth1_;
    Core::LinAlg::Matrix<3 * numnodes, 1> nodaltangentssmooth2_;

    //! Comparator for comparing the beam parameter of two parameter sets
    static bool compare_parsets(
        const std::pair<Core::LinAlg::Matrix<3, 1, TYPEBTS>, Core::LinAlg::Matrix<2, 1, int>>& lhs,
        const std::pair<Core::LinAlg::Matrix<3, 1, TYPEBTS>, Core::LinAlg::Matrix<2, 1, int>>& rhs)
    {
      // Compare eta
      return lhs.first(2) < rhs.first(2);
    }

    //! Comparator for comparing the beam parameter of two normal sets
    static bool compare_normalsets(
        const std::pair<TYPEBTS, Core::LinAlg::Matrix<3, 1, TYPEBTS>>& lhs, const TYPEBTS& rhs)
    {
      // Compare eta
      return lhs.first < rhs;
    }

    //! Vector containing structs for Gmsh debug
    std::vector<GmshDebugPoint> gmsh_debug_points_;


    //@}

    //! @name Private evaluation methods

    /*!
    \brief Evaluate contact forces and stiffness for one contact interval
    */
    void evaluate_contact_interval(const double& pp,
        const std::pair<Core::LinAlg::Matrix<3, 1, TYPEBTS>, int>& parset_a,
        const std::pair<Core::LinAlg::Matrix<3, 1, TYPEBTS>, int>& parset_b,
        Core::LinAlg::Matrix<3 * numnodes * numnodalvalues, 1, TYPEBTS>& fc1,
        Core::LinAlg::Matrix<3 * numnodessol, 1, TYPEBTS>& fc2,
        Core::LinAlg::Matrix<3 * numnodes * numnodalvalues,
            3 * numnodes * numnodalvalues + 3 * numnodessol, TYPEBTS>& stiffc1,
        Core::LinAlg::Matrix<3 * numnodessol, 3 * numnodes * numnodalvalues + 3 * numnodessol,
            TYPEBTS>& stiffc2,
        bool& doAssembleContactInterval,
        Core::LinAlg::Matrix<3 * numnodes * numnodalvalues,
            3 * numnodes * numnodalvalues + 3 * numnodessol, TYPEBTS>& stiffc1_FAD,
        Core::LinAlg::Matrix<3 * numnodessol, 3 * numnodes * numnodalvalues + 3 * numnodessol,
            TYPEBTS>& stiffc2_FAD);

    /*!
    \brief Evaluate penalty force law for different regularizations
    */
    void evaluate_penalty_force_law(
        const double& pp, const TYPEBTS& gap, TYPEBTS& fp, TYPEBTS& dfp);

    /*!
    \brief Evaluate contact forces
    */
    void evaluate_fc_contact(const TYPEBTS& fp,
        Core::LinAlg::Matrix<3 * numnodes * numnodalvalues, 1, TYPEBTS>& fc1,
        Core::LinAlg::Matrix<3 * numnodessol, 1, TYPEBTS>& fc2, const TYPEBTS& eta_a,
        const TYPEBTS& eta_b, const double& w_gp, const double& sgn,
        const Core::LinAlg::Matrix<3, 1, TYPEBTS>& nD,
        const Core::LinAlg::Matrix<3, 1, TYPEBTS>& n2,
        const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPEBTS>& N1,
        const Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N2, const double& jacobi);

    /*!
    \brief Evaluate contact stiffness
    */
    void evaluate_stiffc_contact(const TYPEBTS& fp, const TYPEBTS& dfp, const TYPEBTS& gap,
        Core::LinAlg::Matrix<3 * numnodes * numnodalvalues,
            3 * numnodes * numnodalvalues + 3 * numnodessol, TYPEBTS>& stiffc1,
        Core::LinAlg::Matrix<3 * numnodessol, 3 * numnodes * numnodalvalues + 3 * numnodessol,
            TYPEBTS>& stiffc2,
        const double& sgn, const TYPEBTS& eta_a, const TYPEBTS& eta_b,
        Core::LinAlg::Matrix<3 * numnodes * numnodalvalues + 3 * numnodessol, 1, TYPEBTS>& eta_d,
        Core::LinAlg::Matrix<3 * numnodes * numnodalvalues + 3 * numnodessol, 1, TYPEBTS>& eta_a_d,
        Core::LinAlg::Matrix<3 * numnodes * numnodalvalues + 3 * numnodessol, 1, TYPEBTS>& eta_b_d,
        const double& w_gp, const Core::LinAlg::Matrix<3, 1, TYPEBTS>& rD, const TYPEBTS& norm_rD,
        const Core::LinAlg::Matrix<3, 1, TYPEBTS>& nD,
        const Core::LinAlg::Matrix<3, 1, TYPEBTS>& a2, const TYPEBTS& norm_a2,
        const Core::LinAlg::Matrix<3, 1, TYPEBTS>& n2,
        const Core::LinAlg::Matrix<3, 1, TYPEBTS>& r1_eta,
        const Core::LinAlg::Matrix<3, 1, TYPEBTS>& x2_xi1,
        const Core::LinAlg::Matrix<3, 1, TYPEBTS>& x2_xi2,
        const Core::LinAlg::Matrix<3, 1, TYPEBTS>& x2_xi1xi1,
        const Core::LinAlg::Matrix<3, 1, TYPEBTS>& x2_xi2xi2,
        const Core::LinAlg::Matrix<3, 1, TYPEBTS>& x2_xi1xi2,
        const Core::LinAlg::Matrix<3, 1, TYPEBTS>& x2_xi2xi1,
        const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPEBTS>& N1,
        const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPEBTS>& N1_eta,
        const Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N2,
        const Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N2_xi1,
        const Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N2_xi2, const double& jacobi,
        Core::LinAlg::Matrix<3 * numnodes * numnodalvalues,
            3 * numnodes * numnodalvalues + 3 * numnodessol, TYPEBTS>& stiffc1_FAD,
        Core::LinAlg::Matrix<3 * numnodessol, 3 * numnodes * numnodalvalues + 3 * numnodessol,
            TYPEBTS>& stiffc2_FAD);

    /*!
    \brief Compute linearizations of element parameters xi1, xi2 and eta
    */
    void compute_lin_parameter(const int& fixed_par,
        Core::LinAlg::Matrix<3 * numnodes * numnodalvalues + 3 * numnodessol, 1, TYPEBTS>& xi1_d,
        Core::LinAlg::Matrix<3 * numnodes * numnodalvalues + 3 * numnodessol, 1, TYPEBTS>& xi2_d,
        Core::LinAlg::Matrix<3 * numnodes * numnodalvalues + 3 * numnodessol, 1, TYPEBTS>& eta_d,
        const Core::LinAlg::Matrix<3, 1, TYPEBTS>& rD,
        const Core::LinAlg::Matrix<3, 1, TYPEBTS>& r1_eta,
        const Core::LinAlg::Matrix<3, 1, TYPEBTS>& x2_xi1,
        const Core::LinAlg::Matrix<3, 1, TYPEBTS>& x2_xi2,
        const Core::LinAlg::Matrix<3, 1, TYPEBTS>& x2_xi1xi1,
        const Core::LinAlg::Matrix<3, 1, TYPEBTS>& x2_xi2xi2,
        const Core::LinAlg::Matrix<3, 1, TYPEBTS>& x2_xi1xi2,
        const Core::LinAlg::Matrix<3, 1, TYPEBTS>& x2_xi2xi1,
        const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPEBTS>& N1,
        const Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N2,
        const Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N2_xi1,
        const Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N2_xi2);

    /*!
    \brief Compute linearization of gap
    */
    void compute_lin_gap(
        Core::LinAlg::Matrix<3 * numnodes * numnodalvalues + 3 * numnodessol, 1, TYPEBTS>& gap_d,
        const Core::LinAlg::Matrix<3 * numnodes * numnodalvalues + 3 * numnodessol, 1, TYPEBTS>&
            xi1_d,
        const Core::LinAlg::Matrix<3 * numnodes * numnodalvalues + 3 * numnodessol, 1, TYPEBTS>&
            xi2_d,
        const Core::LinAlg::Matrix<3 * numnodes * numnodalvalues + 3 * numnodessol, 1, TYPEBTS>&
            eta_d,
        const double sgn, const Core::LinAlg::Matrix<3, 1, TYPEBTS>& rD,
        const Core::LinAlg::Matrix<3, 1, TYPEBTS>& nD,
        const Core::LinAlg::Matrix<3, 1, TYPEBTS>& n2, const TYPEBTS& norm_rD,
        const Core::LinAlg::Matrix<3, 1, TYPEBTS>& r1_eta,
        const Core::LinAlg::Matrix<3, 1, TYPEBTS>& x2_xi1,
        const Core::LinAlg::Matrix<3, 1, TYPEBTS>& x2_xi2,
        const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPEBTS>& N1,
        const Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N2,
        Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues + 3 * numnodessol, TYPEBTS>& rD_d);

    /*!
    \brief Compute linearization of unit distance vector nD and surface unit normal vector n2
    */
    void compute_lin_normal(
        Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues + 3 * numnodessol, TYPEBTS>& nD_d,
        const Core::LinAlg::Matrix<3, 1, TYPEBTS>& nD, const TYPEBTS& norm_rD,
        Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues + 3 * numnodessol, TYPEBTS>& n2_d,
        const Core::LinAlg::Matrix<3, 1, TYPEBTS>& n2, const TYPEBTS& norm_a2,
        const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues + 3 * numnodessol, TYPEBTS>&
            rD_d,
        const Core::LinAlg::Matrix<3 * numnodes * numnodalvalues + 3 * numnodessol, 1, TYPEBTS>&
            xi1_d,
        const Core::LinAlg::Matrix<3 * numnodes * numnodalvalues + 3 * numnodessol, 1, TYPEBTS>&
            xi2_d,
        const Core::LinAlg::Matrix<3, 1, TYPEBTS>& x2_xi1,
        const Core::LinAlg::Matrix<3, 1, TYPEBTS>& x2_xi2,
        const Core::LinAlg::Matrix<3, 1, TYPEBTS>& x2_xi1xi1,
        const Core::LinAlg::Matrix<3, 1, TYPEBTS>& x2_xi2xi2,
        const Core::LinAlg::Matrix<3, 1, TYPEBTS>& x2_xi1xi2,
        const Core::LinAlg::Matrix<3, 1, TYPEBTS>& x2_xi2xi1,
        const Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N2_xi1,
        const Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N2_xi2);

    /*!
    \brief Assemble contact forces and stiffness
    */
    void assemble_fc_and_stiffc_contact(
        const Core::LinAlg::Matrix<3 * numnodes * numnodalvalues, 1, TYPEBTS> fc1,
        const Core::LinAlg::Matrix<3 * numnodessol, 1, TYPEBTS> fc2, Epetra_Vector* fint,
        const Core::LinAlg::Matrix<3 * numnodes * numnodalvalues,
            3 * numnodes * numnodalvalues + 3 * numnodessol, TYPEBTS>
            stiffc1,
        const Core::LinAlg::Matrix<3 * numnodessol, 3 * numnodes * numnodalvalues + 3 * numnodessol,
            TYPEBTS>
            stiffc2,
        Core::LinAlg::SparseMatrix& stiffmatrix);

    /*!
    \brief Find projection of surface edges on beam and projection of beam center line on surface
    (CPP)
    */
    void projection(
        const int& fixed_par, TYPEBTS& xi1, TYPEBTS& xi2, TYPEBTS& eta, bool& proj_allowed);

    /*!
    \brief Find contact interval borders
    */
    void get_contact_interval_borders(
        std::vector<std::pair<Core::LinAlg::Matrix<3, 1, TYPEBTS>, int>>& parsets);

    /*!
    \brief Calculate beam shape function values for given parameter value eta
    */
    void get_beam_shape_functions(
        Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPEBTS>& N,
        Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPEBTS>& N_eta,
        Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPEBTS>& N_etaeta,
        const TYPEBTS& eta);

    /*!
    \brief Calculate solid surface shape function values for given parameter values xi1 and xi2
    */
    void get_surf_shape_functions(Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N,
        Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N_xi1,
        Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N_xi2,
        Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N_xi1xi1,
        Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N_xi2xi2,
        Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N_xi1xi2,
        Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N_xi2xi1, const TYPEBTS& xi1,
        const TYPEBTS& xi2);

    /*!
    \brief Assemble beam shape functions into corresponding matrices
    */
    void assemble_beam_shapefunctions(
        const Core::LinAlg::Matrix<1, numnodes * numnodalvalues, TYPEBTS>& N_i,
        const Core::LinAlg::Matrix<1, numnodes * numnodalvalues, TYPEBTS>& N_i_eta,
        const Core::LinAlg::Matrix<1, numnodes * numnodalvalues, TYPEBTS>& N_i_etaeta,
        Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPEBTS>& N,
        Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPEBTS>& N_eta,
        Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPEBTS>& N_etaeta);

    /*!
    \brief Assemble solid surface shape functions into corresponding matrices
    */
    void assemble_surf_shapefunctions(const Core::LinAlg::Matrix<1, numnodessol, TYPEBTS>& N_i,
        const Core::LinAlg::Matrix<2, numnodessol, TYPEBTS>& N_i_xi,
        const Core::LinAlg::Matrix<3, numnodessol, TYPEBTS>& N_i_xixi,
        Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N,
        Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N_xi1,
        Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N_xi2,
        Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N_xi1xi1,
        Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N_xi2xi2,
        Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N_xi1xi2,
        Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N_xi2xi1);

    /*!
    \brief Compute beam coordinates and their derivatives from the discretization
    */
    void compute_beam_coords_and_derivs(Core::LinAlg::Matrix<3, 1, TYPEBTS>& r,
        Core::LinAlg::Matrix<3, 1, TYPEBTS>& r_eta, Core::LinAlg::Matrix<3, 1, TYPEBTS>& r_etaeta,
        const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPEBTS>& N,
        const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPEBTS>& N_eta,
        const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPEBTS>& N_etaeta);

    /*!
    \brief Compute solid surface coordinates and their derivatives from the discretization
    */
    void compute_surf_coords_and_derivs(Core::LinAlg::Matrix<3, 1, TYPEBTS>& r,
        Core::LinAlg::Matrix<3, 1, TYPEBTS>& r_xi1, Core::LinAlg::Matrix<3, 1, TYPEBTS>& r_xi2,
        Core::LinAlg::Matrix<3, 1, TYPEBTS>& r_xi1xi1,
        Core::LinAlg::Matrix<3, 1, TYPEBTS>& r_xi2xi2,
        Core::LinAlg::Matrix<3, 1, TYPEBTS>& r_xi1xi2,
        Core::LinAlg::Matrix<3, 1, TYPEBTS>& r_xi2xi1,
        Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N,
        Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N_xi1,
        Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N_xi2,
        Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N_xi1xi1,
        Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N_xi2xi2,
        Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N_xi1xi2,
        Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N_xi2xi1);

    /*!
    \brief Compute distance vector rD, its norm norm_rD and unit distance vector nD
    */
    void compute_distance_normal(const Core::LinAlg::Matrix<3, 1, TYPEBTS>& r1,
        const Core::LinAlg::Matrix<3, 1, TYPEBTS>& x2, Core::LinAlg::Matrix<3, 1, TYPEBTS>& rD,
        TYPEBTS& norm_rD, Core::LinAlg::Matrix<3, 1, TYPEBTS>& nD);

    /*!
    \brief Compute tangent cross product a2, its norm norm_a2 and surface unit normal vector n2
    */
    void compute_surface_normal(const Core::LinAlg::Matrix<3, 1, TYPEBTS>& x2_xi1,
        const Core::LinAlg::Matrix<3, 1, TYPEBTS>& x2_xi2, Core::LinAlg::Matrix<3, 1, TYPEBTS>& a2,
        TYPEBTS& norm_a2, Core::LinAlg::Matrix<3, 1, TYPEBTS>& n2);

    /*!
    \brief Utility method for CPP (evaluate nonlinear function f)
    */
    void evaluate_orthogonality_condition(Core::LinAlg::Matrix<2, 1, TYPEBTS>& f,
        const Core::LinAlg::Matrix<3, 1, TYPEBTS>& delta_r, const double norm_delta_r,
        const Core::LinAlg::Matrix<3, 1, TYPEBTS>& r1_xi,
        const Core::LinAlg::Matrix<3, 1, TYPEBTS>& x2_xi,
        const Core::LinAlg::Matrix<3, 1, TYPEBTS>& t1,
        const Core::LinAlg::Matrix<3, 1, TYPEBTS>& t2);

    /*!
    \brief Utility method for CPP (evaluate Jacobian of nonlinear function f)
    */
    void evaluate_lin_orthogonality_condition(Core::LinAlg::Matrix<2, 2, TYPEBTS>& df,
        Core::LinAlg::Matrix<2, 2, TYPEBTS>& dfinv,
        const Core::LinAlg::Matrix<3, 1, TYPEBTS>& delta_r, const double norm_delta_r,
        const Core::LinAlg::Matrix<3, 1, TYPEBTS>& r1_xi,
        const Core::LinAlg::Matrix<3, 1, TYPEBTS>& x2_xi,
        const Core::LinAlg::Matrix<3, 1, TYPEBTS>& r1_xixi,
        const Core::LinAlg::Matrix<3, 1, TYPEBTS>& x2_xixi,
        const Core::LinAlg::Matrix<3, 1, TYPEBTS>& t1,
        const Core::LinAlg::Matrix<3, 1, TYPEBTS>& t2,
        const Core::LinAlg::Matrix<3, 1, TYPEBTS>& t1_xi,
        const Core::LinAlg::Matrix<3, 1, TYPEBTS>& t2_xi);

    /*!
    \brief Check, if we have contact or not
    */
    void check_contact_status(const double& pp, const TYPEBTS& gap, bool& contactflag);

    /*!
    \brief These method shifts the nodal positions applied within the beam contact framework py a
    small pre-defined amount in order to enable contact evaluation in the case of two identical
    contact points, i.e r1=r2
    */
    void shift_nodal_positions();

    void fad_check_lin_parameter(const int& fixed_par,
        const Core::LinAlg::Matrix<3, 1, TYPEBTS>& rD,
        const Core::LinAlg::Matrix<3, 1, TYPEBTS>& x2_xi1,
        const Core::LinAlg::Matrix<3, 1, TYPEBTS>& x2_xi2,
        Core::LinAlg::Matrix<3 * numnodes * numnodalvalues + 3 * numnodessol, 1, TYPEBTS>&
            xi_1d_FAD,
        Core::LinAlg::Matrix<3 * numnodes * numnodalvalues + 3 * numnodessol, 1, TYPEBTS>&
            xi_2d_FAD,
        Core::LinAlg::Matrix<3 * numnodes * numnodalvalues + 3 * numnodessol, 1, TYPEBTS>&
            eta_d_FAD,
        const Core::LinAlg::Matrix<3 * numnodes * numnodalvalues + 3 * numnodessol, 1, TYPEBTS>&
            xi1_d,
        const Core::LinAlg::Matrix<3 * numnodes * numnodalvalues + 3 * numnodessol, 1, TYPEBTS>&
            xi2_d,
        const Core::LinAlg::Matrix<3 * numnodes * numnodalvalues + 3 * numnodessol, 1, TYPEBTS>&
            eta_d);

    void fad_check_lin_orthogonality_condition(const int& fixed_par,
        const Core::LinAlg::Matrix<3, 1, TYPEBTS>& rD,
        const Core::LinAlg::Matrix<3, 1, TYPEBTS>& x2_xi1,
        const Core::LinAlg::Matrix<3, 1, TYPEBTS>& x2_xi2,
        Core::LinAlg::Matrix<2, 2, TYPEBTS>& J_FAD, const Core::LinAlg::Matrix<2, 2, TYPEBTS>& J);

    void fad_check_lin_gap_and_distance_vector(const TYPEBTS& gap,
        const Core::LinAlg::Matrix<3, 1, TYPEBTS>& rD,
        const Core::LinAlg::Matrix<3 * numnodes * numnodalvalues + 3 * numnodessol, 1, TYPEBTS>&
            xi1_d,
        const Core::LinAlg::Matrix<3 * numnodes * numnodalvalues + 3 * numnodessol, 1, TYPEBTS>&
            xi2_d,
        const Core::LinAlg::Matrix<3 * numnodes * numnodalvalues + 3 * numnodessol, 1, TYPEBTS>&
            eta_d,
        Core::LinAlg::Matrix<3 * numnodes * numnodalvalues + 3 * numnodessol, 1, TYPEBTS>&
            gap_d_FAD,
        Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues + 3 * numnodessol, TYPEBTS>& rD_d_FAD,
        const Core::LinAlg::Matrix<3 * numnodes * numnodalvalues + 3 * numnodessol, 1, TYPEBTS>&
            gap_d,
        const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues + 3 * numnodessol, TYPEBTS>&
            rD_d);

    void fad_check_lin_normal(const Core::LinAlg::Matrix<3, 1, TYPEBTS>& nD,
        const Core::LinAlg::Matrix<3, 1, TYPEBTS>& n2,
        const Core::LinAlg::Matrix<3 * numnodes * numnodalvalues + 3 * numnodessol, 1, TYPEBTS>&
            xi1_d,
        const Core::LinAlg::Matrix<3 * numnodes * numnodalvalues + 3 * numnodessol, 1, TYPEBTS>&
            xi2_d,
        const Core::LinAlg::Matrix<3 * numnodes * numnodalvalues + 3 * numnodessol, 1, TYPEBTS>&
            eta_d,
        Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues + 3 * numnodessol, TYPEBTS>& nD_d_FAD,
        Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues + 3 * numnodessol, TYPEBTS>& n_2d_FAD,
        const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues + 3 * numnodessol, TYPEBTS>&
            nD_d,
        const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues + 3 * numnodessol, TYPEBTS>&
            n2_d);

    void fd_check_stiffness(const double& pp,
        const Core::LinAlg::Matrix<3 * numnodes * numnodalvalues, 1, TYPEBTS>& fc1,
        const Core::LinAlg::Matrix<3 * numnodessol, 1, TYPEBTS>& fc2,
        Core::LinAlg::Matrix<3 * numnodes * numnodalvalues,
            3 * numnodes * numnodalvalues + 3 * numnodessol, TYPEBTS>& stiffc1,
        Core::LinAlg::Matrix<3 * numnodessol, 3 * numnodes * numnodalvalues + 3 * numnodessol,
            TYPEBTS>& stiffc2);

    void fad_check_stiffness(const Core::LinAlg::Matrix<3 * numnodes * numnodalvalues,
                                 3 * numnodes * numnodalvalues + 3 * numnodessol, TYPEBTS>& stiffc1,
        const Core::LinAlg::Matrix<3 * numnodessol, 3 * numnodes * numnodalvalues + 3 * numnodessol,
            TYPEBTS>& stiffc2,
        const Core::LinAlg::Matrix<3 * numnodes * numnodalvalues,
            3 * numnodes * numnodalvalues + 3 * numnodessol, TYPEBTS>& stiffc1_FAD,
        const Core::LinAlg::Matrix<3 * numnodessol, 3 * numnodes * numnodalvalues + 3 * numnodessol,
            TYPEBTS>& stiffc2_FAD);


    /*!
    \brief Get global dofs of a node

    Internally this method first extracts the dofs of the given node
    in the beam contact discretization (which has its own dofs) and
    then transfers these dofs to their actual GIDs in the underlying
    problem discretization by applying the pre-computed dofoffset_.
    */
    std::vector<int> get_global_dofs(const Core::Nodes::Node* node);

    //@}

  };  // class Beam3tosolidcontact
}  // namespace CONTACT

FOUR_C_NAMESPACE_CLOSE

#endif

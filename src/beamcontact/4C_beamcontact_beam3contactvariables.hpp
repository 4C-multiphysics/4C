#ifndef FOUR_C_BEAMCONTACT_BEAM3CONTACTVARIABLES_HPP
#define FOUR_C_BEAMCONTACT_BEAM3CONTACTVARIABLES_HPP

#include "4C_config.hpp"

#include "4C_beamcontact_beam3contactinterface.hpp"
#include "4C_beaminteraction_beam_to_beam_contact_defines.hpp"
#include "4C_beaminteraction_beam_to_beam_contact_tangentsmoothing.hpp"
#include "4C_beaminteraction_beam_to_beam_contact_utils.hpp"
#include "4C_fem_general_utils_local_connectivity_matrices.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_sparsematrix.hpp"

#include <Sacado.hpp>

FOUR_C_NAMESPACE_OPEN


namespace CONTACT
{
  /*!
   \brief contact element for contact between two 3D beam elements

   Refer also to the Semesterarbeit of Matthias Mayr, 2010

   */

  template <const int numnodes, const int numnodalvalues>
  class Beam3contactvariables
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
    Beam3contactvariables(std::pair<TYPE, TYPE>& closestpoint, std::pair<int, int>& segids,
        std::pair<int, int>& intids, const double& pp, TYPE jacobi);

    /*!
    \brief Destructor
    */
    virtual ~Beam3contactvariables() = default;
    //@}


    //! @name Access methods

    /*!
    \brief Set closest point
    */

    void set_cp(std::pair<TYPE, TYPE> cp) { closestpoint_ = cp; };

    /*!
    \brief Get closest point
    */
    std::pair<TYPE, TYPE> get_cp() { return closestpoint_; };

    /*!
    \brief Get Segment Ids
    */
    std::pair<int, int> get_seg_ids() { return segids_; };

    /*!
    \brief Get Integration Ids
    */
    std::pair<int, int> get_int_ids() { return intids_; };

    /*!
    \brief Get jacobi factor
    */
    TYPE get_jacobi() { return jacobi_; };

    /*!
    \brief Set gap
    */
    void set_gap(TYPE gap) { gap_ = gap; };

    /*!
    \brief Get gap
    */
    TYPE get_gap() { return gap_; };

    /*!
    \brief Set gap
    */
    void set_normal(Core::LinAlg::Matrix<3, 1, TYPE> normal) { normal_ = normal; };

    /*!
    \brief Get gap
    */
    Core::LinAlg::Matrix<3, 1, TYPE> get_normal() { return normal_; };

    /*!
    \brief Get penalty parameter
    */
    double get_pp() { return pp_; };

    /*!
    \brief Set penalty force
    */
    void setfp(TYPE fp) { fp_ = fp; };

    /*!
    \brief Get penalty force
    */
    TYPE getfp() { return fp_; };

    /*!
    \brief Set derivative of penalty force
    */
    void setdfp(TYPE dfp) { dfp_ = dfp; };

    /*!
    \brief Get pre-factor for penalty parameter
    */
    TYPE get_p_pfac() { return ppfac_; };

    /*!
    \brief Set pre-factor for penalty parameter
    */
    void set_p_pfac(TYPE ppfac) { ppfac_ = ppfac; };

    /*!
    \brief Get linearization of pre-factor for penalty parameter
    */
    TYPE get_dp_pfac() { return dppfac_; };

    /*!
    \brief Set linearization of pre-factor for penalty parameter
    */
    void set_dp_pfac(TYPE dppfac) { dppfac_ = dppfac; };

    /*!
    \brief Get derivative of penalty force
    */
    TYPE getdfp() { return dfp_; };

    /*!
    \brief Set penalty energy
    */
    void set_energy(TYPE e) { energy_ = e; };

    /*!
    \brief Get penalty energy
    */
    TYPE get_energy() { return energy_; };

    /*!
    \brief Set length integrated penalty energy
    */
    void set_integrated_energy(double inte) { integratedenergy_ = inte; };

    /*!
    \brief Get length integrated penalty energy
    */
    double get_integrated_energy() { return integratedenergy_; };

    /*!
    \brief Set contact angle
    */
    void set_angle(double angle) { angle_ = angle; };

    /*!
    \brief Get contact angle
    */
    double get_angle() { return angle_; };

    //@}


    //@}
   private:
    // closest point coordinates
    std::pair<TYPE, TYPE> closestpoint_;

    // element local Ids of considered segments
    std::pair<int, int> segids_;

    // stores numgp and number of integration interval (only necessary for small-angle contact)
    std::pair<int, int> intids_;

    // jacobi factor for integration (only necessary for line contact)
    TYPE jacobi_;

    // gap function
    TYPE gap_;

    // normal vector
    Core::LinAlg::Matrix<3, 1, TYPE> normal_;

    // penalty parameter
    double pp_;

    // penalty parameter
    TYPE ppfac_;

    // linearization of penalty parameter
    TYPE dppfac_;

    // penalty force
    TYPE fp_;

    // derivative of penalty force with respect to gap: d(fp_)/d(gap_)
    TYPE dfp_;

    // penalty energy of collocation point / Gauss point
    TYPE energy_;

    // length integrated penalty energy of collocation point (in this case identical to energy_) /
    // Gauss point
    double integratedenergy_;

    // contact angle
    double angle_;
    //@}

  };  // class Beam3contactvariables
}  // namespace CONTACT

FOUR_C_NAMESPACE_CLOSE

#endif

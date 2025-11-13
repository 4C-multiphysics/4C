// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAMINTERACTION_BEAM_TO_BEAM_CONTACT_VARIABLES_HPP
#define FOUR_C_BEAMINTERACTION_BEAM_TO_BEAM_CONTACT_VARIABLES_HPP


#include "4C_config.hpp"

#include "4C_beaminteraction_beam_to_beam_contact_defines.hpp"
#include "4C_beaminteraction_beam_to_beam_contact_tangentsmoothing.hpp"
#include "4C_beaminteraction_beam_to_beam_contact_utils.hpp"
#include "4C_beaminteraction_contact_pair.hpp"
#include "4C_fem_general_utils_local_connectivity_matrices.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_utils_fad.hpp"

#include <Sacado.hpp>

FOUR_C_NAMESPACE_OPEN

namespace BeamInteraction
{
  /*!
   \brief Todo
   */

  template <unsigned int numnodes, unsigned int numnodalvalues>
  class BeamToBeamContactVariables
  {
   public:
    //! @name Friends
    // no friend classes defined
    //@}

    //! @name Constructors and destructors and related methods
    /*!
    \brief Standard Constructor
    */
    BeamToBeamContactVariables(std::pair<TYPE, TYPE>& closestpoint, std::pair<int, int>& segids,
        std::pair<int, int>& intids, const double& pp, TYPE jacobi);

    /*!
    \brief Destructor
    */
    virtual ~BeamToBeamContactVariables() = default;
    //@}


    //! @name Access methods

    /*!
    \brief Set closest point
    */

    void set_cp(std::pair<TYPE, TYPE> cp) { closestpoint_ = cp; };

    /*!
    \brief Get closest point
    */
    std::pair<TYPE, TYPE> get_cp() const { return closestpoint_; };

    /*!
    \brief Get Segment Ids
    */
    std::pair<int, int> get_seg_ids() const { return segids_; };

    /*!
    \brief Get Integration Ids
    */
    std::pair<int, int> get_int_ids() const { return intids_; };

    /*!
    \brief Get jacobi factor
    */
    TYPE get_jacobi() const { return jacobi_; };

    /*!
    \brief Set gap
    */
    void set_gap(TYPE gap) { gap_ = gap; };

    /*!
    \brief Get gap
    */
    TYPE get_gap() const { return gap_; };

    /*!
    \brief Set normal
    */
    void set_normal(Core::LinAlg::Matrix<3, 1, TYPE> normal) { normal_ = normal; };

    /*!
    \brief Set old normal
    */
    void set_old_normal(Core::LinAlg::Matrix<3, 1, TYPE> old_normal) { old_normal_ = old_normal; };

    /*!
    \brief Set sign
    */
    void set_sign(double sign) { sign_ = sign; };

    /*!
    \brief Get sign
    */
    double get_sign() const { return sign_; };


    /*!
    \brief Get normal
    */
    Core::LinAlg::Matrix<3, 1, TYPE> get_normal() const { return normal_; };

    /*!
    \brief Get previous normal
    */
    Core::LinAlg::Matrix<3, 1, TYPE> get_old_normal() const { return old_normal_; };

    /*!
    \brief Get penalty parameter
    */
    double get_pp() const { return pp_; };

    /*!
    \brief Set penalty force
    */
    void setfp(TYPE fp) { fp_ = fp; };

    /*!
    \brief Get penalty force
    */
    TYPE getfp() const { return fp_; };

    /*!
    \brief Set derivative of penalty force
    */
    void setdfp(TYPE dfp) { dfp_ = dfp; };

    /*!
    \brief Get pre-factor for penalty parameter
    */
    TYPE get_p_pfac() const { return ppfac_; };

    /*!
    \brief Set pre-factor for penalty parameter
    */
    void set_p_pfac(TYPE ppfac) { ppfac_ = ppfac; };

    /*!
    \brief Get linearization of pre-factor for penalty parameter
    */
    TYPE get_dp_pfac() const { return dppfac_; };

    /*!
    \brief Set linearization of pre-factor for penalty parameter
    */
    void set_dp_pfac(TYPE dppfac) { dppfac_ = dppfac; };

    /*!
    \brief Get derivative of penalty force
    */
    TYPE getdfp() const { return dfp_; };

    /*!
    \brief Set penalty energy
    */
    void set_energy(TYPE e) { energy_ = e; };

    /*!
    \brief Get penalty energy
    */
    TYPE get_energy() const { return energy_; };

    /*!
    \brief Set length integrated penalty energy
    */
    void set_integrated_energy(double inte) { integratedenergy_ = inte; };

    /*!
    \brief Get length integrated penalty energy
    */
    double get_integrated_energy() const { return integratedenergy_; };

    /*!
    \brief Set contact angle
    */
    void set_angle(double angle) { angle_ = angle; };

    /*!
    \brief Get contact angle
    */
    double get_angle() const { return angle_; };

    //@}


    //! @name Print methods

    /** \brief print information to screen
     *
     */
    inline void print(std::ostream& out) const
    {
      out << "\nInstance of BeamToBeamContactVariables (SegmentIds " << segids_.first << " & "
          << segids_.second << "):";
      out << "\ngap= " << Core::FADUtils::cast_to_double(gap_);
      out << "\nangle= " << angle_ / std::numbers::pi * 180.0 << " degree";
      out << "\nclosest point coords: " << Core::FADUtils::cast_to_double(closestpoint_.first)
          << " " << Core::FADUtils::cast_to_double(closestpoint_.second);

      out << "\n";
    }

    /** \brief print information to screen
     *
     */
    inline void print_summary_in_one_line(std::ostream& out) const
    {
      out << std::setw(9) << std::left << std::setprecision(2) << closestpoint_.first
          << std::setw(9) << std::left << std::setprecision(2) << closestpoint_.second
          << std::setw(9) << std::left << std::setprecision(3) << angle_ / std::numbers::pi * 180.0
          << std::setw(12) << std::left << std::scientific << gap_ << std::setw(12) << std::left
          << std::scientific << fp_ << std::setprecision(6)
          << std::resetiosflags(std::ios::scientific) << std::right;
    }

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

    // sign of gap
    double sign_;

    // normal vector
    Core::LinAlg::Matrix<3, 1, TYPE> normal_;

    // normal previous vector
    Core::LinAlg::Matrix<3, 1, TYPE> old_normal_;

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
  };
}  // namespace BeamInteraction

FOUR_C_NAMESPACE_CLOSE

#endif

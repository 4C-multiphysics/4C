// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAMINTERACTION_POTENTIAL_PAIR_BASE_HPP
#define FOUR_C_BEAMINTERACTION_POTENTIAL_PAIR_BASE_HPP

#include "4C_config.hpp"

#include "4C_beaminteraction_potential_input.hpp"
#include "4C_fem_condition.hpp"
#include "4C_fem_general_utils_integration.hpp"
#include "4C_linalg_fixedsizematrix.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

// forward declaration ...
namespace Core::LinAlg
{
  class SerialDenseVector;
  class SerialDenseMatrix;
}  // namespace Core::LinAlg

namespace Core::Elements
{
  class Element;
}

namespace BeamInteraction
{
  // forward declaration ...
  struct BeamPotentialParameters;


  /*!
   \brief
   */
  class BeamPotentialPair
  {
   public:
    //! @name Friends
    // no friend classes defined
    //@}

    //! @name Constructors and destructors and related methods

    /*!
    \brief Constructor
    */
    BeamPotentialPair();

    /*!
    \brief Destructor
    */
    virtual ~BeamPotentialPair() = default;
    //! Initialization
    void init(const BeamInteraction::Potential::BeamPotentialParameters* params_ptr,
        Core::Elements::Element const* element1, Core::Elements::Element const* element2);

    //! Setup
    virtual void setup();

    //! return appropriate derived (templated) class (acts as a simple factory)
    static std::shared_ptr<BeamPotentialPair> create(
        std::vector<Core::Elements::Element const*> const& ele_ptrs,
        BeamInteraction::Potential::BeamPotentialParameters const& beam_potential_params);

    //@}


    //! @name Public evaluation methods
    /*!
    \brief Evaluate this contact element pair, return value indicates whether pair is active,
           i.e. non-zero values for force and stiffmat are returned
    */
    virtual bool evaluate(Core::LinAlg::SerialDenseVector* forcevec1,
        Core::LinAlg::SerialDenseVector* forcevec2, Core::LinAlg::SerialDenseMatrix* stiffmat11,
        Core::LinAlg::SerialDenseMatrix* stiffmat12, Core::LinAlg::SerialDenseMatrix* stiffmat21,
        Core::LinAlg::SerialDenseMatrix* stiffmat22,
        const std::vector<Core::Conditions::Condition*> linechargeconds, const double k,
        const double m) = 0;

    /*
    \brief Update state of translational nodal DoFs (absolute positions and tangents) of both
    elements
    */
    virtual void reset_state(double time, std::vector<double> const& centerline_dofvec_ele1,
        std::vector<double> const& centerline_dofvec_ele2) = 0;

    //@}

    //! @name Access methods

    inline BeamInteraction::Potential::BeamPotentialParameters const* params() const
    {
      return beam_potential_parameters_;
    }

    /*!
    \brief Get first element
    */
    inline Core::Elements::Element const* element1() const { return element1_; };

    /*!
    \brief Get second element
    */
    inline Core::Elements::Element const* element2() const { return element2_; };

    /*!
    \brief Get coordinates of all interacting points on element1 and element2
    */
    virtual void get_all_interacting_point_coords_element1(
        std::vector<Core::LinAlg::Matrix<3, 1, double>>& coords) const = 0;

    virtual void get_all_interacting_point_coords_element2(
        std::vector<Core::LinAlg::Matrix<3, 1, double>>& coords) const = 0;

    /*!
    \brief Get forces at all interacting points on element1 and element2
    */
    virtual void get_forces_at_all_interacting_points_element1(
        std::vector<Core::LinAlg::Matrix<3, 1, double>>& forces) const = 0;

    virtual void get_forces_at_all_interacting_points_element2(
        std::vector<Core::LinAlg::Matrix<3, 1, double>>& forces) const = 0;

    /*!
    \brief Get moments at all interacting points on element1 and element2
    */
    virtual void get_moments_at_all_interacting_points_element1(
        std::vector<Core::LinAlg::Matrix<3, 1, double>>& moments) const = 0;

    virtual void get_moments_at_all_interacting_points_element2(
        std::vector<Core::LinAlg::Matrix<3, 1, double>>& moments) const = 0;

    /*!
    \brief Get interaction free energy / potential
    */
    virtual double get_energy() const = 0;

    /** \brief print this beam potential-based element pair to screen
     *
     *  */
    virtual void print(std::ostream& out) const = 0;

    /** \brief print this beam potential element pair to screen
     *
     *  */
    virtual void print_summary_one_line_per_active_segment_pair(std::ostream& out) const = 0;

   protected:
    //! returns init state
    inline bool const& is_init() const { return isinit_; };

    //! returns setup state
    inline bool const& is_setup() const { return issetup_; };

    //! Check the init state
    void check_init() const;

    //! Check the init and setup state
    void check_init_setup() const;

    //! get Gauss rule to be used
    Core::FE::GaussRule1D get_gauss_rule() const;

    /*!
    \brief Set first element
    */
    inline void set_element1(Core::Elements::Element const* element1) { element1_ = element1; };

    /*!
    \brief Set second element
    */
    inline void set_element2(Core::Elements::Element const* element2) { element2_ = element2; };

    //@}

   protected:
    //! @name member variables

    //! indicates if the init() function has been called
    bool isinit_;

    //! indicates if the setup() function has been called
    bool issetup_;

   private:
    //! beam potential parameter data container
    const BeamInteraction::Potential::BeamPotentialParameters* beam_potential_parameters_;

    //! first element of interacting pair
    Core::Elements::Element const* element1_;

    //! second element of interacting pair
    Core::Elements::Element const* element2_;
    //@}
  };
}  // namespace BeamInteraction

FOUR_C_NAMESPACE_CLOSE

#endif

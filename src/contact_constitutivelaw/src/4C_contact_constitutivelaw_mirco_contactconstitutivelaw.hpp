// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONTACT_CONSTITUTIVELAW_MIRCO_CONTACTCONSTITUTIVELAW_HPP
#define FOUR_C_CONTACT_CONSTITUTIVELAW_MIRCO_CONTACTCONSTITUTIVELAW_HPP

#include "4C_config.hpp"

#include "4C_contact_constitutivelaw_contactconstitutivelaw.hpp"
#include "4C_contact_constitutivelaw_contactconstitutivelaw_parameter.hpp"
#include "4C_linalg_serialdensematrix.hpp"

#ifdef FOUR_C_WITH_MIRCO
#include <mirco_kokkostypes.h>
#endif

FOUR_C_NAMESPACE_OPEN

namespace CONTACT
{
  namespace CONSTITUTIVELAW
  {
    /*----------------------------------------------------------------------*/
    /** \brief constitutive law parameters for a mirco contact law to the contact pressure
     *
     */
    class MircoConstitutiveLawParams : public Parameter
    {
     public:
      /** \brief standard constructor
       * \param[in] container containing the law parameter from the input file
       */
      MircoConstitutiveLawParams(const Core::IO::InputParameterContainer& container);

      /// @name get-functions for the Constitutive Law parameters of a mirco function
      //@{

      int get_first_mat_id() const { return firstmatid_; };
      int get_second_mat_id() const { return secondmatid_; };
      double get_lateral_length() const { return lateral_length_; };
      bool get_pressure_green_fun_flag() const { return pressure_green_fun_flag_; };
      double get_tolerance() const { return tolerance_; };
      double get_max_iteration() const { return max_iteration_; };
      bool get_warm_starting_flag() const { return warm_starting_flag_; };
      double get_composite_youngs() const { return composite_youngs_; };
      double get_grid_size() const { return grid_size_; };
      double get_compliance_correction() const { return elastic_compliance_correction_; };
      double get_finite_difference_fraction() const { return finite_difference_fraction_; };
      double get_active_gap_tolerance() const { return active_gap_tolerance_; };
#ifdef FOUR_C_WITH_MIRCO
      const MIRCO::ViewVector_d* get_mesh_grid() const { return &meshgrid_; }
#endif

      void set_parameters();

     private:
      /// @name Constitutive Law parameters of a mirco function
      //@{

      int firstmatid_;
      int secondmatid_;
      double lateral_length_;
      int resolution_;
      bool pressure_green_fun_flag_;
      bool random_topology_flag_;
      bool random_seed_flag_;
      int random_generator_seed_;
      double tolerance_;
      int max_iteration_;
      bool warm_starting_flag_;
      double composite_youngs_;
      double grid_size_;
      double elastic_compliance_correction_;
#ifdef FOUR_C_WITH_MIRCO
      MIRCO::ViewVector_d meshgrid_;
#endif
      double finite_difference_fraction_;
      double active_gap_tolerance_;
      std::string topology_file_path_;
      //@}
    };  // class

    /*----------------------------------------------------------------------*/
    /** \brief implements a mirco contact constitutive law relating the gap to the
     * contact pressure
     */
    class MircoConstitutiveLaw : public ConstitutiveLaw
    {
     public:
      /// construct the constitutive law object given a set of parameters
      explicit MircoConstitutiveLaw(CONTACT::CONSTITUTIVELAW::MircoConstitutiveLawParams params);

      //! @name Access methods
      //@{

      /// Return quick accessible contact constitutive law parameter data
      const CONTACT::CONSTITUTIVELAW::Parameter* parameter() const override { return &params_; }

      //@}

      //! @name Evaluation methods
      //@{
      /** \brief Evaluate the constitutive law
       *
       * The pressure response for a gap is calucated using MIRCO, which uses BEM for solving
       * contact between a rigid rough surface and a linear elastic half space.
       *
       * \param gap contact gap at the mortar node
       * \return The pressure response from MIRCO
       */
      double evaluate(const double gap, CONTACT::Node* cnode) override;

      /** \brief Evaluate derivative of the constitutive law
       *
       * The derivative of the pressure response is approximated using a finite difference approach
       * by calling MIRCO twice at two different gap values and doing a backward difference
       * approximation for the linearization.
       *
       * \param gap contact gap at the mortar node
       * \return Derivative of the pressure responses from MIRCO
       */
      double evaluate_derivative(const double gap, CONTACT::Node* cnode) override;
      //@}

     private:
      /// my constitutive law parameters
      CONTACT::CONSTITUTIVELAW::MircoConstitutiveLawParams params_;
    };
  }  // namespace CONSTITUTIVELAW
}  // namespace CONTACT

FOUR_C_NAMESPACE_CLOSE

#endif

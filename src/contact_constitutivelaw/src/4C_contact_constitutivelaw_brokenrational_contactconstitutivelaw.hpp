// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONTACT_CONSTITUTIVELAW_BROKENRATIONAL_CONTACTCONSTITUTIVELAW_HPP
#define FOUR_C_CONTACT_CONSTITUTIVELAW_BROKENRATIONAL_CONTACTCONSTITUTIVELAW_HPP


#include "4C_config.hpp"

#include "4C_contact_constitutivelaw_contactconstitutivelaw.hpp"
#include "4C_contact_constitutivelaw_contactconstitutivelaw_parameter.hpp"

FOUR_C_NAMESPACE_OPEN

namespace CONTACT
{
  namespace CONSTITUTIVELAW
  {
    /*----------------------------------------------------------------------*/
    /** \brief Constitutive law parameters for a broken rational contact law \f$ A/(x-B)+C \f$
     * relating the gap to the contact pressure
     *
     */
    class BrokenRationalConstitutiveLawParams : public Parameter
    {
     public:
      /// standard constructor
      BrokenRationalConstitutiveLawParams(const Core::IO::InputParameterContainer& container);

      /// @name get-functions for the Constitutive Law parameters of a broken rational function
      //@{
      /// Get the scaling factor
      double getdata() { return a_; };
      /// Get the asymptote
      double get_b() { return b_; };
      /// get the y intercept
      double get_c() { return c_; };
      //@}

     private:
      /// @name Constitutive Law parameters of a broken rational function
      //@{
      /// scaling
      const double a_;
      /// asymptote
      const double b_;
      /// y intercept
      const double c_;
      //@}
    };  // class
    /*----------------------------------------------------------------------*/
    /**
     * \brief implements a broken rational function \f$ A/(x-B)+C \f$ as contact constitutive law
     * relating the gap to the contact pressure
     */
    class BrokenRationalConstitutiveLaw : public ConstitutiveLaw
    {
     public:
      /// construct the constitutive law object given a set of parameters
      explicit BrokenRationalConstitutiveLaw(
          CONTACT::CONSTITUTIVELAW::BrokenRationalConstitutiveLawParams params);

      //! @name Access methods
      //@{

      /// Get scaling factor of the broken rational function
      double getdata() { return params_.getdata(); }
      /// Get asymptote of the broken rational function
      double get_b() { return params_.get_b(); }
      /// Get Y intercept of the broken rational function
      double get_c() { return params_.get_c(); }

      /// Return quick accessible mcontact constitutive law parameter data
      const CONTACT::CONSTITUTIVELAW::Parameter* parameter() const override { return &params_; }

      //@}

      //! @name Evaluation methods
      //@{

      /// evaluate the constitutive law
      double evaluate(const double gap, CONTACT::Node* cnode) override;
      /// Evaluate derivative of the constitutive law
      double evaluate_derivative(const double gap, CONTACT::Node* cnode) override;
      //@}

     private:
      /// my constitutive law parameters
      CONTACT::CONSTITUTIVELAW::BrokenRationalConstitutiveLawParams params_;
    };
  }  // namespace CONSTITUTIVELAW
}  // namespace CONTACT

FOUR_C_NAMESPACE_CLOSE

#endif

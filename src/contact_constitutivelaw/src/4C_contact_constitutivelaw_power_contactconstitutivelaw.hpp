// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONTACT_CONSTITUTIVELAW_POWER_CONTACTCONSTITUTIVELAW_HPP
#define FOUR_C_CONTACT_CONSTITUTIVELAW_POWER_CONTACTCONSTITUTIVELAW_HPP


#include "4C_config.hpp"

#include "4C_contact_constitutivelaw_contactconstitutivelaw.hpp"
#include "4C_contact_constitutivelaw_contactconstitutivelaw_parameter.hpp"

FOUR_C_NAMESPACE_OPEN


namespace CONTACT
{
  namespace CONSTITUTIVELAW
  {
    /*----------------------------------------------------------------------*/
    /** \brief constitutive law parameters for a power contact law \f$ Ax^B \f$ relating the gap to
     * the contact pressure
     *
     */
    class PowerConstitutiveLawParams : public Parameter
    {
     public:
      /** \brief standard constructor
       * \param[in] container containing the law parameter from the input file
       */
      PowerConstitutiveLawParams(const Core::IO::InputParameterContainer& container);

      /// @name get-functions for the Constitutive Law parameters of a power law function
      //@{
      /// Get the scaling factor
      double getdata() const { return a_; };
      /// Get the power coefficient
      double get_b() const { return b_; };
      //@}

     private:
      /// @name Constitutive Law parameters of a power function
      //@{
      /// scaling factor
      const double a_;
      /// power coefficient
      const double b_;
      //@}
    };  // class

    /*----------------------------------------------------------------------*/
    /** \brief implements a power contact constitutive law \f$ Ax^B \f$ relating the gap to the
     * contact pressure
     *
     */
    class PowerConstitutiveLaw : public ConstitutiveLaw
    {
     public:
      /// construct the constitutive law object given a set of parameters
      explicit PowerConstitutiveLaw(CONTACT::CONSTITUTIVELAW::PowerConstitutiveLawParams params);

      //! @name Access methods
      //@{

      /// Get scaling factor of power law
      double getdata() { return params_.getdata(); }
      /// Get power coefficient of power law
      double get_b() { return params_.get_b(); }

      /// Return quick accessible contact constitutive law parameter data
      const CONTACT::CONSTITUTIVELAW::Parameter* parameter() const override { return &params_; }
      //@}

      //! @name Evaluation methods
      //@{
      /// Evaluate the constitutive law
      double evaluate(double gap, CONTACT::Node* cnode) override;
      /// Evaluate derivative of the constitutive law
      double evaluate_derivative(double gap, CONTACT::Node* cnode) override;
      //@}

     private:
      /// my constitutive law parameters
      CONTACT::CONSTITUTIVELAW::PowerConstitutiveLawParams params_;
    };
  }  // namespace CONSTITUTIVELAW
}  // namespace CONTACT

FOUR_C_NAMESPACE_CLOSE

#endif

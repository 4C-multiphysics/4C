/*----------------------------------------------------------------------*/
/*! \file
\brief implements a broken rational function \f$ A/(x-B)+C \f$ as contact constitutive law

\level 1

*/
/*----------------------------------------------------------------------*/
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
      BrokenRationalConstitutiveLawParams(
          const Teuchos::RCP<const CONTACT::CONSTITUTIVELAW::Container> container);


      /// create constitutive law instance of matching type with my parameters
      Teuchos::RCP<CONTACT::CONSTITUTIVELAW::ConstitutiveLaw> CreateConstitutiveLaw() override;

      /// @name get-functions for the Constitutive Law parameters of a broken rational function
      //@{
      /// Get the scaling factor
      double GetA() { return a_; };
      /// Get the asymptote
      double GetB() { return b_; };
      /// get the y intercept
      double GetC() { return c_; };
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
          CONTACT::CONSTITUTIVELAW::BrokenRationalConstitutiveLawParams* params);

      //! @name Access methods

      /// return contact constitutive law type
      INPAR::CONTACT::ConstitutiveLawType GetConstitutiveLawType() const override
      {
        return INPAR::CONTACT::ConstitutiveLawType::colaw_brokenrational;
      }

      /// Get scaling factor of the broken rational function
      double GetA() { return params_->GetA(); }
      /// Get asymptote of the broken rational function
      double GetB() { return params_->GetB(); }
      /// Get Y intercept of the broken rational function
      double GetC() { return params_->GetC(); }

      /// Return quick accessible mcontact constitutive law parameter data
      CONTACT::CONSTITUTIVELAW::Parameter* Parameter() const override { return params_; }

      //@}

      //! @name Evaluation methods

      /// evaluate the constitutive law
      double Evaluate(double gap, CONTACT::Node* cnode) override;
      /// Evaluate derivative of the constitutive law
      double EvaluateDeriv(double gap, CONTACT::Node* cnode) override;
      //@}

     private:
      /// my constitutive law parameters
      CONTACT::CONSTITUTIVELAW::BrokenRationalConstitutiveLawParams* params_;
    };
  }  // namespace CONSTITUTIVELAW
}  // namespace CONTACT

FOUR_C_NAMESPACE_CLOSE

#endif

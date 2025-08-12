// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONTACT_CONSTITUTIVELAW_PYTHON_SURROGATE_CONTACTCONSTITUTIVELAW_HPP
#define FOUR_C_CONTACT_CONSTITUTIVELAW_PYTHON_SURROGATE_CONTACTCONSTITUTIVELAW_HPP

#include "4C_config.hpp"

#include "4C_contact_constitutivelaw_contactconstitutivelaw.hpp"
#include "4C_contact_constitutivelaw_contactconstitutivelaw_parameter.hpp"

#include <filesystem>

#ifdef FOUR_C_WITH_PYBIND11

#include <pybind11/embed.h>
#include <pybind11/numpy.h>

FOUR_C_NAMESPACE_OPEN

namespace CONTACT
{
  namespace CONSTITUTIVELAW
  {
    /*----------------------------------------------------------------------*/
    /** \brief constitutive law parameters for Python-based surrogate of the contact law to the
     * contact pressure
     *
     */
    class PythonSurrogateConstitutiveLawParams : public Parameter
    {
     public:
      /** \brief standard constructor
       * \param[in] container Contains the law parameters from the input file
       */
      PythonSurrogateConstitutiveLawParams(const Core::IO::InputParameterContainer& container);

      /// Get the filename of the Python script
      std::filesystem::path get_python_filepath() const { return python_filename_; }

     private:
      /// File name of python file defining the surrogate model
      std::filesystem::path python_filename_;

    };  // class

    /*----------------------------------------------------------------------*/
    /** \brief implements a Python-based surrogate for the contact constitutive law relating the gap
     * to the contact pressure
     */
    class PythonSurrogateConstitutiveLaw : public ConstitutiveLaw
    {
     public:
      /// construct the constitutive law object given a set of parameters
      explicit PythonSurrogateConstitutiveLaw(
          CONTACT::CONSTITUTIVELAW::PythonSurrogateConstitutiveLawParams params);

      //! @name Access methods
      //@{

      /// Get the filename of the Python script
      std::filesystem::path get_python_filepath() { return params_.get_python_filepath(); }

      /// Return quick accessible contact constitutive law parameter data
      const CONTACT::CONSTITUTIVELAW::Parameter* parameter() const override { return &params_; }

      //@}

      //! @name Evaluation methods
      //@{
      /** \brief Evaluate the constitutive law
       *
       * The pressure response for a given gap is evaluated in a Python-based model specified in the
       * input file.
       *
       * \param[in] gap Contact gap at the mortar node
       * \return The pressure response from the Python surrogate model
       */
      double evaluate(double gap, CONTACT::Node* cnode) override;

      /** \brief Evaluate derivative of the constitutive law
       *
       * The derivative of the pressure response is evaluated in a Python-based model specified in
       * the input file.
       *
       * \param[in] gap Contact gap at the mortar node
       * \return Derivative of the pressure response from the Python surrogate model
       */
      double evaluate_derivative(double gap, CONTACT::Node* cnode) override;
      //@}

     private:
      std::unique_ptr<pybind11::scoped_interpreter> guard_ = nullptr;

      pybind11::object evaluate_;
      pybind11::object evaluate_derivative_;

      /// my constitutive law parameters
      CONTACT::CONSTITUTIVELAW::PythonSurrogateConstitutiveLawParams params_;
    };
  }  // namespace CONSTITUTIVELAW
}  // namespace CONTACT

FOUR_C_NAMESPACE_CLOSE

#endif
#endif

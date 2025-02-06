// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_UTILS_PARAMETER_LIST_HPP
#define FOUR_C_UTILS_PARAMETER_LIST_HPP

#include "4C_config.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

namespace Core
{
  namespace Utils
  {
    //! add entry as item of enum class @p value to @p list with name @p parameter_name
    template <class EnumType>
    void add_enum_class_to_parameter_list(
        const std::string& parameter_name, const EnumType value, Teuchos::ParameterList& list)
    {
      const std::string docu = "";
      const std::string value_name = "val";
      Teuchos::setStringToIntegralParameter<EnumType>(parameter_name, value_name, docu,
          Teuchos::tuple<std::string>(value_name), Teuchos::tuple<EnumType>(value), &list);
    }

    /// local wrapper to test multiple versions of "Yes", "YES", etc
    void bool_parameter(std::string const& paramName, std::string const& value,
        std::string const& docString, Teuchos::ParameterList* paramList);

    /// local wrapper for Teuchos::setIntParameter() that allows only integers
    void int_parameter(std::string const& paramName, int const value, std::string const& docString,
        Teuchos::ParameterList* paramList);

    /// local wrapper for Teuchos::setDoubleParameter() that allows only doubles
    void double_parameter(std::string const& paramName, double const& value,
        std::string const& docString, Teuchos::ParameterList* paramList);

    /*!
    \brief Sets a string parameter in a Teuchos::ParameterList with optional validation.

    This function adds a string parameter to a given Teuchos::ParameterList with a
     Teuchos::Stringalidator.
    Optionally, a list of valid string values can be provided for validation.

    @param[in] paramName Name of the parameter to be added to the parameter list.
    @param[in] value The string value of the parameter.
    @param[in] docString Documentation string describing the parameter.
    @param[in/out] paramList The parameter list that will be updated with the new parameter.
    @param[in] validParams (Optional) A list of valid string values for the parameter.
    */
    void string_parameter(std::string const& paramName, std::string const& value,
        std::string const& docString, Teuchos::ParameterList* paramList,
        std::vector<std::string> const& validParams = {});


    /**
     * Add an integral parameter to a parameter list with a list of valid strings.
     */
    template <typename T>
    void string_to_integral_parameter(const std::string& paramName, const std::string& defaultValue,
        const std::string& docString, const Teuchos::ArrayView<const std::string>& strings,
        const Teuchos::ArrayView<const T>& integrals, Teuchos::ParameterList* paramList)
    {
      Teuchos::setStringToIntegralParameter<T>(
          paramName, defaultValue, docString, strings, integrals, paramList);
    }

  }  // namespace Utils
}  // namespace Core


FOUR_C_NAMESPACE_CLOSE

#endif

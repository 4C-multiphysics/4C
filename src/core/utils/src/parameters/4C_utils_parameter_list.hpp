// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_UTILS_PARAMETER_LIST_HPP
#define FOUR_C_UTILS_PARAMETER_LIST_HPP

#include "4C_config.hpp"

#include "4C_io_input_spec_builders.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

namespace Core
{
  namespace Utils
  {
    //! A wrapper to gather InputSpecs for a section. This type is used during migration
    //! to InputSpec.
    struct SectionSpecs
    {
      SectionSpecs(const std::string& section_name) : section_name(section_name) {}
      //! Create the concatenated section name
      SectionSpecs(const SectionSpecs& parent, const std::string& section_name)
          : section_name(parent.section_name + "/" + section_name)
      {
      }

      ~SectionSpecs()
      {
        FOUR_C_ASSERT(specs.empty(),
            "SectionSpecs of '%s' must be moved into a collection before destruction.",
            section_name.c_str());
      }

      std::string section_name;
      std::vector<Core::IO::InputSpec> specs;

      void move_into_collection(std::map<std::string, Core::IO::InputSpec>& map)
      {
        FOUR_C_ASSERT(map.contains(section_name) == false,
            "SectionSpecs of '%s' already exists in the collection.", section_name.c_str());
        map[section_name] = Core::IO::InputSpecBuilders::all_of(std::move(specs));
      }
    };

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

    /// temporary helper around InputSpecBuilders::parameter<bool>
    void bool_parameter(std::string const& paramName, bool default_value,
        std::string const& docString, SectionSpecs& section_specs);
    // fail if users provide a string literal as default value
    void bool_parameter(std::string const& paramName, const char* default_value,
        std::string const& docString, SectionSpecs& section_specs) = delete;

    /// local wrapper for Teuchos::setIntParameter() that allows only integers
    void int_parameter(std::string const& paramName, int const value, std::string const& docString,
        SectionSpecs& section_specs);

    /// local wrapper for Teuchos::setDoubleParameter() that allows only doubles
    void double_parameter(std::string const& paramName, double const& value,
        std::string const& docString, SectionSpecs& section_specs);

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
        std::string const& docString, SectionSpecs& section_specs,
        std::vector<std::string> const& validParams = {});


    /**
     * Add an integral parameter to a parameter list with a list of valid strings.
     */
    template <typename T>
    void string_to_integral_parameter(const std::string& paramName, const std::string& defaultValue,
        const std::string& docString, const Teuchos::ArrayView<const std::string>& strings,
        const Teuchos::ArrayView<const T>& integrals, SectionSpecs& section_specs)
    {
      FOUR_C_ASSERT(
          strings.size() == integrals.size(), "The number of strings and integrals must match.");
      std::map<std::string, T> choices;
      for (int i = 0; i < strings.size(); ++i)
      {
        choices.emplace(strings[i], integrals[i]);
      }

      auto default_it = std::find(strings.begin(), strings.end(), defaultValue);
      FOUR_C_ASSERT(default_it != strings.end(), "Default value not found in choices.");

      section_specs.specs.emplace_back(
          Core::IO::InputSpecBuilders::deprecated_selection<T>(paramName, choices,
              {
                  .description = docString,
                  .default_value = integrals[std::distance(strings.begin(), default_it)],
              }));
    }

  }  // namespace Utils
}  // namespace Core


FOUR_C_NAMESPACE_CLOSE

#endif

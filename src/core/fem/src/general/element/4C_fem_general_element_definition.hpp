// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FEM_GENERAL_ELEMENT_DEFINITION_HPP
#define FOUR_C_FEM_GENERAL_ELEMENT_DEFINITION_HPP

#include "4C_config.hpp"

#include "4C_io_input_spec.hpp"

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

FOUR_C_NAMESPACE_OPEN


namespace Core::Elements
{
  /// Collection of valid element input file line definitions
  /*!
    The actual definition is done by each element's type class.
   */
  class ElementDefinition
  {
   public:
    /// Setup of
    void setup_valid_element_lines();

    /// print all valid element definitions to stream
    void print_element_dat_header_to_stream(std::ostream& stream);

    /// Print section header to stream
    void print_section_header(std::ostream& stream, std::string name);

    /// Print valid input lines for given element type
    void print_element_lines(std::ostream& stream, std::string name);

    /// return line definitions for given element type
    const Core::IO::InputSpec& element_lines(std::string name, std::string cell_type);

    const auto& definitions() const { return definitions_; }

   private:
    /// input line definitions per element type
    std::map<std::string, std::map<std::string, Core::IO::InputSpec>> definitions_;
  };

}  // namespace Core::Elements


void print_element_dat_header();


FOUR_C_NAMESPACE_CLOSE

#endif

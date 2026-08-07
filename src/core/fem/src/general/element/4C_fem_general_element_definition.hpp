// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FEM_GENERAL_ELEMENT_DEFINITION_HPP
#define FOUR_C_FEM_GENERAL_ELEMENT_DEFINITION_HPP

#include "4C_config.hpp"

#include "4C_fem_general_cell_type.hpp"
#include "4C_io_input_spec.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <utility>

FOUR_C_NAMESPACE_OPEN


namespace Core::Elements
{
  /**
   * Collection of valid element input.
   *
   * This glass gathers all valid element definitions from the global state. This is possible
   * since the ElementType subclass register themselves in the global Factory.
   */
  struct ElementDefinition
  {
    enum class CellTypeGrouping : std::uint8_t
    {
      one_of,  //< only one cell type group is allowed (and required) in the input
      all_of,  //< all cell type groups are allowed in the input, but none are required.
    };

    //! Gather all valid element definitions from global state
    ElementDefinition();

    /**
     * Convenience access with check for existence of element definition.
     */
    const Core::IO::InputSpec& get(
        const std::string& element_name, Core::FE::CellType cell_type) const;

    /**
     * Get an InputSpec that describes all valid element definitions.
     *
     * @param cell_type_grouping Whether the cell type groups are wrapped in a @c one_of or @c
     * all_of group.
     */
    [[nodiscard]] Core::IO::InputSpec element_data_spec(
        CellTypeGrouping cell_type_grouping = CellTypeGrouping::all_of) const;

    /**
     * Given a @p data container with exactly one element definition group, unpack the the element
     * name and a map from all present cell type definitions to their corresponding input data.
     * Asserts that at least one cell type group is defined.
     */
    [[nodiscard]] std::pair<std::string,
        std::map<Core::FE::CellType, Core::IO::InputParameterContainer>>
    unpack_element_data(const Core::IO::InputParameterContainer& data) const;

    //! Map from element name to cell type to InputSpec.
    std::map<std::string, std::map<Core::FE::CellType, Core::IO::InputSpec>> definitions;
  };

}  // namespace Core::Elements


FOUR_C_NAMESPACE_CLOSE

#endif

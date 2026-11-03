// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_STRUCTURE_NEW_ELEMENT_EROSION_HPP
#define FOUR_C_STRUCTURE_NEW_ELEMENT_EROSION_HPP

#include "4C_config.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}

namespace Core::Utils
{
  class FunctionOfSpaceTime;
}

namespace Solid
{
  struct ElementErosionParameters;

  //! Result of eroding structural elements
  struct ErosionResult
  {
    //! Global IDs of the deactivated row elements owned by this rank
    std::vector<int> row_element_ids;

    //! Number of deactivated elements across all ranks
    int global_count = 0;
  };

  /*!
   * @brief Erode elements according to the @p parameters at @p time
   *
   * Active elements are deactivated for good if the erosion function at their center in the
   * reference configuration reaches the threshold or if their material failed. Inactive elements
   * stay inactive. This is a collective call. @p erosion_function is null when no function is
   * configured.
   */
  ErosionResult erode_elements(Core::FE::Discretization& discretization,
      const ElementErosionParameters& parameters,
      const Core::Utils::FunctionOfSpaceTime* erosion_function, double time);
}  // namespace Solid

FOUR_C_NAMESPACE_CLOSE

#endif

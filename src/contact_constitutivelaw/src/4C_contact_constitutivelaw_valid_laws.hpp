// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONTACT_CONSTITUTIVELAW_VALID_LAWS_HPP
#define FOUR_C_CONTACT_CONSTITUTIVELAW_VALID_LAWS_HPP

/*----------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_io_input_spec.hpp"

#include <iostream>
#include <memory>
#include <vector>

FOUR_C_NAMESPACE_OPEN


namespace CONTACT::CONSTITUTIVELAW
{
  /**
   * Defines how a contact constitutive law should look like in the input file.
   */
  [[nodiscard]] Core::IO::InputSpec valid_contact_constitutive_laws();

  /**
   * After reading input, this function creates the contact constitutive law.
   */
  void create_contact_constitutive_law_from_input(
      const Core::IO::InputParameterContainer& container);
}  // namespace CONTACT::CONSTITUTIVELAW



FOUR_C_NAMESPACE_CLOSE

#endif

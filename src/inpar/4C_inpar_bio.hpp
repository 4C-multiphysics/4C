// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_INPAR_BIO_HPP
#define FOUR_C_INPAR_BIO_HPP

#include "4C_config.hpp"

#include "4C_io_input_spec.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN

// forward declaration
namespace Core::Conditions
{
  class ConditionDefinition;
}

namespace Inpar
{
  namespace BioFilm
  {
    /// set the biofilm parameters
    Core::IO::InputSpec set_valid_parameters();

    /// set specific biofilm conditions
    void set_valid_conditions(std::vector<Core::Conditions::ConditionDefinition>& condlist);
  }  // namespace BioFilm

}  // namespace Inpar

/*----------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif

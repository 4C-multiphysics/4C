// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAMINTERACTION_INPUT_HPP
#define FOUR_C_BEAMINTERACTION_INPUT_HPP

#include "4C_config.hpp"

#include "4C_fem_general_utils_integration.hpp"
#include "4C_io_input_spec.hpp"
#include "4C_utils_exceptions.hpp"

#include <vector>


FOUR_C_NAMESPACE_OPEN

// forward declaration
namespace Core::Conditions
{
  class ConditionDefinition;
}

namespace BeamInteraction
{
  enum RepartitionStrategy
  {
    repstr_adaptive,  ///< only do repartitioning in case physically necessary
    repstr_everydt    ///< do repartitioning every time step
  };

  /// beam interaction parameters
  std::vector<Core::IO::InputSpec> valid_parameters();

  /// set beam interaction specific conditions
  void set_valid_conditions(std::vector<Core::Conditions::ConditionDefinition>& condlist);

}  // namespace BeamInteraction


FOUR_C_NAMESPACE_CLOSE

#endif

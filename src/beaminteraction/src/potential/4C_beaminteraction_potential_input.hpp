// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAMINTERACTION_POTENTIAL_INPUT_HPP
#define FOUR_C_BEAMINTERACTION_POTENTIAL_INPUT_HPP

#include "4C_config.hpp"

#include "4C_io_input_spec.hpp"

#include <map>
#include <vector>

FOUR_C_NAMESPACE_OPEN

// forward declaration
namespace Core::Conditions
{
  class ConditionDefinition;
}  // namespace Core::Conditions

namespace BeamPotential
{
  /// type of potential interaction
  enum class Type
  {
    surface,  ///< surface potential
    volume,   ///< volume potential
    vague
  };

  /// available strategies/methods to evaluate potential interaction
  enum class Strategy
  {
    double_length_specific_large_separations,         ///< double length specific potential, large
                                                      ///< separations
    double_length_specific_small_separations,         ///< double length specific potential, small
                                                      ///< separations
    single_length_specific_small_separations,         ///< single length specific potential, small
                                                      ///< separations
    single_length_specific_small_separations_simple,  ///< single length specific potential, small
                                                      ///< separations, reduced variant
    vague
  };

  /// available types to regularize the force law for separations smaller than the specified
  /// regularization separation
  enum class RegularizationType
  {
    linear,    ///< linear extrapolation
    constant,  ///< constant extrapolation, i.e. f(r)=f(r_reg) for all r<r_reg
    none       ///< no regularization
  };

  /// rule for how to assign the role of slave and master to beam elements
  enum class MasterSlaveChoice
  {
    smaller_eleGID_is_slave,
    higher_eleGID_is_slave,
    vague
  };

  /// set the beam potential parameters
  void set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list);

  /// set beam potential specific conditions
  void set_valid_conditions(std::vector<Core::Conditions::ConditionDefinition>& condlist);

}  // namespace BeamPotential

/*----------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif

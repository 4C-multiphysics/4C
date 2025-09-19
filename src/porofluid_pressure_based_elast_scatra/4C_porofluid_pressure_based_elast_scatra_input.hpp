// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_POROFLUID_PRESSURE_BASED_ELAST_SCATRA_INPUT_HPP
#define FOUR_C_POROFLUID_PRESSURE_BASED_ELAST_SCATRA_INPUT_HPP

#include "4C_config.hpp"

#include "4C_io_input_spec.hpp"
#include "4C_utils_exceptions.hpp"

#include <map>
#include <vector>

FOUR_C_NAMESPACE_OPEN

// forward declaration
namespace Core::Conditions
{
  class ConditionDefinition;
}
/*----------------------------------------------------------------------*
 |                                                                      |
 *----------------------------------------------------------------------*/
namespace PoroPressureBased
{
  /// type of coupling strategy for porofluid-elasticity with scalar transport problems
  enum class SolutionSchemePorofluidElastScatra
  {
    twoway_partitioned_nested,
    twoway_partitioned_sequential,
    twoway_monolithic
  };

  enum class volfrac_closing_relation
  {
    blood_lung,
    homogenized_vasculature_tumor
  };

  enum class fluidporo_reaction_coupling
  {
    scalar_by_function,
    no_coupling
  };

  /// set valid parameters for porofluid-elasticity with scalar transport problems
  void set_valid_parameters_porofluid_elast_scatra(
      std::map<std::string, Core::IO::InputSpec>& list);

  /// set valid conditions for porofluid-elasticity with scalar transport problems
  void set_valid_conditions_porofluid_elast_scatra(
      std::vector<Core::Conditions::ConditionDefinition>& condlist);

}  // namespace PoroPressureBased



FOUR_C_NAMESPACE_CLOSE

#endif

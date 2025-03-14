// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FBI_INPUT_HPP
#define FOUR_C_FBI_INPUT_HPP

#include "4C_config.hpp"

#include "4C_fem_general_utils_integration.hpp"
#include "4C_io_input_spec.hpp"
#include "4C_utils_exceptions.hpp"

#include <map>

FOUR_C_NAMESPACE_OPEN

// forward declaration
namespace Core::Conditions
{
  class ConditionDefinition;
}  // namespace Core::Conditions

namespace FBI
{
  /// Coupling of the Fluid and the beam problems
  enum class BeamToFluidCoupling
  {
    fluid,  //< Coupling on the fluid partition, while the beam is not influenced
    solid,  //< Coupling on the structure partition, while the fluid is not influenced
    twoway  //< Full two-way FBI coupling
  };

  /// Parallel presorting strategy to be used for the beam mesh
  enum class BeamToFluidPreSortStrategy
  {
    bruteforce,  //< each processor searches for each beam if it is near one of its fluid elements
    binning      //< each processor only searches for beam elements which lie in or around its bins
  };

  /// Constraint enforcement for beam to fluid meshtying.
  enum class BeamToFluidConstraintEnforcement
  {
    //! Default value.
    none,
    //! Penalty method.
    penalty
  };

  /// discretization approach for beam to fluid meshtying.
  enum class BeamToFluidDiscretization
  {
    none,                    //< Default value
    gauss_point_to_segment,  //< Gauss point to segment approach
    mortar                   //< mortar-type segment to segment approach
  };

  /**
   * \brief Shape function for the mortar Lagrange-multiplicators
   */
  enum class BeamToFluidMeshtingMortarShapefunctions
  {
    //! Default value.
    none,
    //! Linear.
    line2,
    //! Quadratic.
    line3,
    //! Cubic.
    line4
  };

  /// set the beam interaction parameters
  void set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list);

  /// set beam interaction specific conditions
  void set_valid_conditions(std::vector<Core::Conditions::ConditionDefinition>& condlist);

}  // namespace FBI

/*----------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_ART_NET_INPUT_HPP
#define FOUR_C_ART_NET_INPUT_HPP

#include "4C_config.hpp"

#include "4C_io_input_spec.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN

// forward declaration
namespace Core::Conditions
{
  class ConditionDefinition;
}


namespace ArtDyn
{
  enum TimeIntegrationScheme
  {
    tay_gal,
    stationary
  };

  /// initial field for artery problem
  enum InitialField
  {
    initfield_zero_field,
    initfield_field_by_function,
    initfield_field_by_condition
  };

  //! element implementation type
  enum ImplType
  {
    impltype_undefined,
    impltype_lin_exp,
    impltype_pressure_based
  };

  /// arterial dynamic parameters
  Core::IO::InputSpec valid_parameters();
}  // namespace ArtDyn

namespace ArteryNetwork
{
  //! coupling between artery network and pressure-based porofluid-elasticity-scatra problem
  enum class ArteryPorofluidElastScatraCouplingMethod
  {
    none,                    // none
    nodal,                   // nodal
    gauss_point_to_segment,  // Gauss-point-to-segment
    mortar_penalty,          // mortar-penalty
    node_to_point            // 1D node-to-point in 2D/3D
  };

  /// set specific artnet conditions
  void set_valid_conditions(std::vector<Core::Conditions::ConditionDefinition>& condlist);

}  // namespace ArteryNetwork

/*----------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif

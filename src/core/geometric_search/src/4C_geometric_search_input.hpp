// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_GEOMETRIC_SEARCH_INPUT_HPP
#define FOUR_C_GEOMETRIC_SEARCH_INPUT_HPP

#include "4C_config.hpp"

#include "4C_io_input_spec.hpp"
#include "4C_io_pstream.hpp"

#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN

namespace Core::GeometricSearch
{
  //! Parameters for beam bounding volumes
  enum class BeamBoundingVolumeType
  {
    centerline_kdop,
    integration_points_surface
  };
  struct BeamBoundingVolume
  {
    BeamBoundingVolumeType bounding_volume_type;
    double radius_extension_factor;
    std::optional<int> integration_points_axial{std::nullopt};
    std::optional<int> integration_points_circumference{std::nullopt};
  };

  //! Define the valid parameters for the geometric search strategy
  Core::IO::InputSpec valid_parameters();

  //! Parameters for geometric search
  struct GeometricSearchParams
  {
    BeamBoundingVolume beam_bounding_volume;
    double sphere_radius_extension_factor;
    double bounding_volume_inflation_factor;
    double point_tolerance;
    bool use_auxiliary_position;
    bool write_geometric_search_visualization;
    Core::IO::Verbositylevel verbosity;
  };

  //! Factory function for the GeometricSearchParams from the global parameter list
  GeometricSearchParams geometric_search_params_factory(
      const Teuchos::ParameterList& global_parameters);

}  // namespace Core::GeometricSearch

FOUR_C_NAMESPACE_CLOSE

#endif

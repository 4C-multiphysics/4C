// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_geometric_search_input.hpp"

#include "4C_io_input_spec_builders.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN

Core::IO::InputSpec Core::GeometricSearch::valid_parameters()
{
  using namespace Core::IO::InputSpecBuilders;

  auto beam_bounding_volume = group<BeamBoundingVolume>("BEAM_BOUNDING_VOLUME",
      {
          parameter<BeamBoundingVolumeType>("BOUNDING_VOLUME_TYPE",
              {.description = "Type of bounding volume for beams.",
                  .default_value = BeamBoundingVolumeType::centerline_kdop,
                  .store = in_struct(&BeamBoundingVolume::bounding_volume_type)}),

          parameter<double>("RADIUS_EXTENSION_FACTOR",
              {.description = "Beams radius is multiplied with the factor and then the "
                              "bounding volume only depending on the beam centerline is "
                              "extended in all directions (+ and -) by that value.",
                  .default_value = 2.0,
                  .store = in_struct(&BeamBoundingVolume::radius_extension_factor)}),

          parameter<std::optional<int>>("INTEGRATION_POINTS_AXIAL",
              {.description = "Number of axial Gauss points for the bounding volume of beams.",
                  .store = in_struct(&BeamBoundingVolume::integration_points_axial)}),
          parameter<std::optional<int>>("INTEGRATION_POINTS_CIRCUMFERENCE",
              {.description =
                      "Number of circumferential Gauss points for the bounding volume of beams.",
                  .store = in_struct(&BeamBoundingVolume::integration_points_circumference)}),
      },
      {.required = false, .store = in_struct(&GeometricSearchParams::beam_bounding_volume)});

  Core::IO::InputSpec spec = group<GeometricSearchParams>("BOUNDINGVOLUME STRATEGY",
      {beam_bounding_volume,
          parameter<double>("SPHERE_RADIUS_EXTENSION_FACTOR",
              {.description =
                      "Bounding volume of the sphere is the sphere center extended by this factor "
                      "times the sphere radius in all directions (+ and -).",
                  .default_value = 2.0,
                  .store = in_struct(&GeometricSearchParams::sphere_radius_extension_factor)}),
          parameter<double>("CONTACT_EXTENSION_FACTOR",
              {.description = "Extension factor for the bounding volumes used in the contact "
                              "search. The bounding volumes are enlarged by this factor times the "
                              "minimum edge length of all contact pair elements.",
                  .default_value = 0.3,
                  .store = in_struct(&GeometricSearchParams::contact_extension_factor)}),
          parameter<bool>("USE_AUXILIARY_POSITION",
              {.description = "Auxiliary position is used for computing bounding volumes.",
                  .default_value = true,
                  .store = in_struct(&GeometricSearchParams::use_auxiliary_position)}),
          parameter<double>(
              "POINT_TOLERANCE", {.description = "Tolerance for point objects",
                                     .default_value = 1e-4,
                                     .store = in_struct(&GeometricSearchParams::point_tolerance)}),
          parameter<bool>("WRITE_GEOMETRIC_SEARCH_VISUALIZATION",
              {.description = "If visualization output for the geometric search should be written",
                  .default_value = false,
                  .store =
                      in_struct(&GeometricSearchParams::write_geometric_search_visualization)})},
      {.required = false});

  return spec;
}

Core::GeometricSearch::GeometricSearchParams Core::GeometricSearch::geometric_search_params_factory(
    const Teuchos::ParameterList& global_parameters)
{
  auto geometric_search_params =
      global_parameters.get<Core::GeometricSearch::GeometricSearchParams>(
          "BOUNDINGVOLUME STRATEGY");

  geometric_search_params.verbosity = Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
      global_parameters.sublist("IO"), "VERBOSITY");

  // Do some consistency checks.
  if (geometric_search_params.beam_bounding_volume.bounding_volume_type ==
      BeamBoundingVolumeType::integration_points_surface)
  {
    if (!(geometric_search_params.beam_bounding_volume.integration_points_axial.has_value() and
            geometric_search_params.beam_bounding_volume.integration_points_circumference
                .has_value()))
      FOUR_C_THROW(
          "For beam surface based bounding volumes, both axial and circumferential integration "
          "points must be specified!");
  }

  return geometric_search_params;
}


FOUR_C_NAMESPACE_CLOSE
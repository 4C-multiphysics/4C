// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_beam_to_solid_surface_visualization_output_params.hpp"

#include "4C_global_data.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN


/**
 *
 */
BEAMINTERACTION::BeamToSolidSurfaceVisualizationOutputParams::
    BeamToSolidSurfaceVisualizationOutputParams()
    : isinit_(false),
      issetup_(false),
      output_interval_steps_(-1),
      output_every_iteration_(false),
      output_flag_(false),
      nodal_forces_(false),
      averaged_normals_(false),
      mortar_lambda_discret_(false),
      mortar_lambda_continuous_(false),
      mortar_lambda_continuous_segments_(0),
      segmentation_(false),
      integration_points_(false),
      write_unique_ids_(false)
{
  // empty constructor
}

/**
 *
 */
void BEAMINTERACTION::BeamToSolidSurfaceVisualizationOutputParams::init()
{
  issetup_ = false;
  isinit_ = true;
}

/**
 *
 */
void BEAMINTERACTION::BeamToSolidSurfaceVisualizationOutputParams::setup()
{
  check_init();

  // Teuchos parameter lists from input file.
  const Teuchos::ParameterList& beam_to_solid_volume_meshtying_visualization_output_paramslist =
      Global::Problem::instance()
          ->beam_interaction_params()
          .sublist("BEAM TO SOLID SURFACE")
          .sublist("RUNTIME VTK OUTPUT");
  const Teuchos::ParameterList& global_visualization_output_paramslist =
      Global::Problem::instance()->io_params().sublist("RUNTIME VTK OUTPUT");

  // Get global parameters.
  output_interval_steps_ = global_visualization_output_paramslist.get<int>("INTERVAL_STEPS");
  output_every_iteration_ = global_visualization_output_paramslist.get<bool>("EVERY_ITERATION");

  // Get beam to solid surface specific parameters.
  output_flag_ =
      beam_to_solid_volume_meshtying_visualization_output_paramslist.get<bool>("WRITE_OUTPUT");

  nodal_forces_ =
      beam_to_solid_volume_meshtying_visualization_output_paramslist.get<bool>("NODAL_FORCES");

  averaged_normals_ =
      beam_to_solid_volume_meshtying_visualization_output_paramslist.get<bool>("AVERAGED_NORMALS");

  mortar_lambda_discret_ = beam_to_solid_volume_meshtying_visualization_output_paramslist.get<bool>(
      "MORTAR_LAMBDA_DISCRET");

  mortar_lambda_continuous_ =
      beam_to_solid_volume_meshtying_visualization_output_paramslist.get<bool>(
          "MORTAR_LAMBDA_CONTINUOUS");

  mortar_lambda_continuous_segments_ =
      beam_to_solid_volume_meshtying_visualization_output_paramslist.get<int>(
          "MORTAR_LAMBDA_CONTINUOUS_SEGMENTS");

  segmentation_ =
      beam_to_solid_volume_meshtying_visualization_output_paramslist.get<bool>("SEGMENTATION");

  integration_points_ = beam_to_solid_volume_meshtying_visualization_output_paramslist.get<bool>(
      "INTEGRATION_POINTS");

  write_unique_ids_ =
      beam_to_solid_volume_meshtying_visualization_output_paramslist.get<bool>("UNIQUE_IDS");

  // Set the setup flag.
  issetup_ = true;
}

FOUR_C_NAMESPACE_CLOSE

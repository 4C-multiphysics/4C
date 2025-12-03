// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_beam_to_beam_contact_params.hpp"

#include "4C_beaminteraction_contact_beam_to_beam_input.hpp"
#include "4C_global_data.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
BeamInteraction::BeamToBeamContactParams::BeamToBeamContactParams()
    : isinit_(false),
      issetup_(false),
      penalty_law_(BeamInteraction::Contact::BeamToBeam::PenaltyLaw::pl_lp),
      btb_penalty_law_regularization_g0_(-1.0),
      btb_penalty_law_regularization_f0_(-1.0),
      btb_penalty_law_regularization_c0_(-1.0),
      gap_shift_(0.0),
      btb_point_penalty_param_(-1.0),
      btb_function_id_for_point_penalty_(-1),
      btb_line_penalty_param_(-1.0),
      btb_function_id_for_line_penalty_(-1),
      btb_perp_shifting_angle1_(-1.0),
      btb_perp_shifting_angle2_(-1.0),
      btb_parallel_shifting_angle1_(-1.0),
      btb_parallel_shifting_angle2_(-1.0),
      segangle_(-1.0),
      num_integration_intervals_(0),
      btb_basicstiff_gap_(-1.0),
      btb_endpoint_penalty_(false),
      btb_use_new_gap_function_(true)
{
  // empty constructor
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::BeamToBeamContactParams::init()
{
  issetup_ = false;

  // Teuchos parameter list for beam contact
  const Teuchos::ParameterList& beam_contact_params_list =
      Global::Problem::instance()->beam_interaction_params().sublist("BEAM TO BEAM CONTACT");

  /****************************************************************************/
  // get and check required parameters
  /****************************************************************************/

  /****************************************************************************/
  penalty_law_ = Teuchos::getIntegralValue<BeamInteraction::Contact::BeamToBeam::PenaltyLaw>(
      beam_contact_params_list, "BEAMS_PENALTYLAW");

  /****************************************************************************/
  btb_penalty_law_regularization_g0_ = beam_contact_params_list.get<double>("BEAMS_PENREGPARAM_G0");
  btb_penalty_law_regularization_f0_ = beam_contact_params_list.get<double>("BEAMS_PENREGPARAM_F0");
  btb_penalty_law_regularization_c0_ = beam_contact_params_list.get<double>("BEAMS_PENREGPARAM_C0");

  // Todo check and refine these safety checks
  if (penalty_law_ != BeamInteraction::Contact::BeamToBeam::pl_lp and
      penalty_law_ != BeamInteraction::Contact::BeamToBeam::pl_qp)
  {
    if (btb_penalty_law_regularization_g0_ == -1.0 or btb_penalty_law_regularization_f0_ == -1.0 or
        btb_penalty_law_regularization_c0_ == -1.0)
      FOUR_C_THROW(
          "Regularized penalty law chosen, but not all regularization parameters are set!");
  }

  /****************************************************************************/
  // Todo check this parameter
  gap_shift_ = beam_contact_params_list.get<double>("BEAMS_GAPSHIFTPARAM");

  if (gap_shift_ != 0.0 and penalty_law_ != BeamInteraction::Contact::BeamToBeam::pl_lpqp)
    FOUR_C_THROW("BEAMS_GAPSHIFTPARAM only possible for penalty law LinPosQuadPen!");

  /****************************************************************************/
  btb_point_penalty_param_ = beam_contact_params_list.get<double>("BEAMS_BTBPENALTYPARAM");

  btb_function_id_for_point_penalty_ =
      beam_contact_params_list.get<int>("BEAMS_BTBPENALTYPARAM_BY_FUNCT");

  if (btb_point_penalty_param_ < 0.0 and btb_function_id_for_point_penalty_ < 0.0)
    FOUR_C_THROW(
        "beam-to-beam point penalty parameter or beam-to-beam function id for point penalty must "
        "be specified!");

  if (btb_point_penalty_param_ > 0.0 and btb_function_id_for_point_penalty_ > 0.0)
    FOUR_C_THROW(
        "Please specify the beam-to-beam point penalty parameter either with BEAMS_BTBPENALTYPARAM "
        "or BEAMS_BTBPENALTYPARAM_BY_FUNCT. Not both!");


  // input parameters required for all-angle-beam contact formulation ...
  if (beam_contact_params_list.get<bool>("BEAMS_SEGCON"))
  {
    /****************************************************************************/


    btb_function_id_for_line_penalty_ =
        beam_contact_params_list.get<int>("BEAMS_BTBLINEPENALTYPARAM_BY_FUNCT");

    btb_line_penalty_param_ = beam_contact_params_list.get<double>("BEAMS_BTBLINEPENALTYPARAM");

    if (btb_line_penalty_param_ < 0.0 and btb_function_id_for_line_penalty_ < 0.0)
      FOUR_C_THROW(
          " beam-to-beam line penalty parameter or beam-to-beam function id for line penalty must "
          "be specified non negative!");

    if (btb_line_penalty_param_ > 0.0 and btb_function_id_for_line_penalty_ > 0.0)
      FOUR_C_THROW(
          "Please specify the beam-to-beam line penalty parameter either with "
          "BEAMS_BTBLINEPENALTYPARAM or BEAMS_BTBLINEPENALTYPARAM_BY_FUNCT. Not both!");


    /****************************************************************************/
    // Todo find more verbose and expressive naming
    // note: conversion from degrees (input parameter) to radians (class variable) done here!
    btb_perp_shifting_angle1_ =
        beam_contact_params_list.get<double>("BEAMS_PERPSHIFTANGLE1") / 180.0 * std::numbers::pi;
    btb_perp_shifting_angle2_ =
        beam_contact_params_list.get<double>("BEAMS_PERPSHIFTANGLE2") / 180.0 * std::numbers::pi;

    btb_parallel_shifting_angle1_ =
        beam_contact_params_list.get<double>("BEAMS_PARSHIFTANGLE1") / 180.0 * std::numbers::pi;
    btb_parallel_shifting_angle2_ =
        beam_contact_params_list.get<double>("BEAMS_PARSHIFTANGLE2") / 180.0 * std::numbers::pi;

    if (btb_perp_shifting_angle1_ < 0.0 or btb_perp_shifting_angle2_ < 0.0 or
        btb_parallel_shifting_angle1_ < 0.0 or btb_parallel_shifting_angle2_ < 0.0)
      FOUR_C_THROW(
          "You chose all-angle-beam contact algorithm: thus, shifting angles for"
          " beam-to-beam contact fade must be >= 0 degrees");

    if (btb_perp_shifting_angle1_ > 0.5 * std::numbers::pi or
        btb_perp_shifting_angle2_ > 0.5 * std::numbers::pi or
        btb_parallel_shifting_angle1_ > 0.5 * std::numbers::pi or
        btb_parallel_shifting_angle2_ > 0.5 * std::numbers::pi)
      FOUR_C_THROW(
          "You chose all-angle-beam contact algorithm: thus, Shifting angles for"
          " beam-to-beam contact fade must be <= 90 degrees");

    if (btb_parallel_shifting_angle2_ <= btb_perp_shifting_angle1_)
      FOUR_C_THROW("No angle overlap between large-angle and small-angle contact!");

    /****************************************************************************/
    // note: conversion from degrees (input parameter) to radians (class variable) done here!
    segangle_ = beam_contact_params_list.get<double>("BEAMS_SEGANGLE") / 180.0 * std::numbers::pi;

    if (segangle_ <= 0.0) FOUR_C_THROW("Segmentation angle must be greater than zero!");

    /****************************************************************************/
    num_integration_intervals_ = beam_contact_params_list.get<int>("BEAMS_NUMINTEGRATIONINTERVAL");

    if (num_integration_intervals_ <= 0)
      FOUR_C_THROW("Number of integration intervals must be greater than zero!");
  }

  /****************************************************************************/
  // Todo check need and usage of this parameter
  btb_basicstiff_gap_ = beam_contact_params_list.get<double>("BEAMS_BASICSTIFFGAP");

  /****************************************************************************/
  btb_endpoint_penalty_ = beam_contact_params_list.get<bool>("BEAMS_ENDPOINTPENALTY");

  /****************************************************************************/
  // safety checks for currently unsupported parameter settings
  /****************************************************************************/
  // if (beam_contact_params_list.get<bool>("BEAMS_NEWGAP"))
  btb_use_new_gap_function_ = beam_contact_params_list.get<bool>("BEAMS_NEWGAP");

  /****************************************************************************/
  // for the time being only allow all-angle-beam contact formulation ...
  if (not beam_contact_params_list.get<bool>("BEAMS_SEGCON"))
    FOUR_C_THROW(
        "only all-angle-beam contact (BEAMS_SEGCON) formulation tested yet"
        " in new beam interaction framework!");

  /****************************************************************************/
  if (btb_basicstiff_gap_ != -1.0) FOUR_C_THROW("BEAMS_BASICSTIFFGAP currently not supported!");

  isinit_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::BeamToBeamContactParams::setup()
{
  check_init();

  // empty for now

  issetup_ = true;
}

FOUR_C_NAMESPACE_CLOSE

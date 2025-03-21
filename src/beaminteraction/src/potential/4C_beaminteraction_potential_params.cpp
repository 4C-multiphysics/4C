// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_potential_params.hpp"

#include "4C_beamcontact_input.hpp"
#include "4C_beaminteraction_potential_input.hpp"
#include "4C_beaminteraction_potential_runtime_visualization_output_params.hpp"
#include "4C_global_data.hpp"
#include "4C_io_value_parser.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
BeamInteraction::BeamPotentialParams::BeamPotentialParams()
    : isinit_(false),
      issetup_(false),
      pot_law_exponents_(nullptr),
      pot_law_prefactors_(nullptr),
      potential_type_(BeamPotential::Type::vague),
      strategy_(BeamPotential::Strategy::vague),
      cutoff_radius_(0.0),
      regularization_type_(BeamPotential::RegularizationType::none),
      regularization_separation_(0.0),
      num_integration_segments_(-1),
      num_gp_s_(-1),
      use_fad_(false),
      choice_master_slave_(BeamPotential::MasterSlaveChoice::vague),
      visualization_output_(false),
      params_runtime_visualization_output_btb_potential_(nullptr),
      potential_reduction_length_(0.0)
{
  // empty constructor
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void BeamInteraction::BeamPotentialParams::init(const double restart_time)
{
  issetup_ = false;

  // Teuchos parameter list for beam potential-based interactions
  const Teuchos::ParameterList& beam_potential_params_list =
      Global::Problem::instance()->beam_potential_params();

  /****************************************************************************/
  // get and check required parameters
  /****************************************************************************/

  pot_law_prefactors_ = std::make_shared<std::vector<double>>();
  pot_law_exponents_ = std::make_shared<std::vector<double>>();
  pot_law_prefactors_->clear();
  pot_law_exponents_->clear();
  // read potential law parameters from input and check
  {
    std::string pot_law_exponents_in =
        Teuchos::getNumericStringParameter(beam_potential_params_list, "POT_LAW_EXPONENT");

    Core::IO::ValueParser pot_law_exponents_parser(
        pot_law_exponents_in, {.user_scope_message = "While reading potential law exponents: "});

    while (!pot_law_exponents_parser.at_end())
    {
      pot_law_exponents_->push_back(pot_law_exponents_parser.read<double>());
    }
  }
  {
    std::string pot_law_prefactors_in =
        Teuchos::getNumericStringParameter(beam_potential_params_list, "POT_LAW_PREFACTOR");

    Core::IO::ValueParser pot_law_prefactors_parser(
        pot_law_prefactors_in, {.user_scope_message = "While reading potential law prefactors: "});

    while (!pot_law_prefactors_parser.at_end())
    {
      pot_law_prefactors_->push_back(pot_law_prefactors_parser.read<double>());
    }
  }
  if (!pot_law_prefactors_->empty())
  {
    if (pot_law_prefactors_->size() != pot_law_exponents_->size())
      FOUR_C_THROW(
          "Number of potential law prefactors does not match number of potential law exponents."
          " Check your input file!");

    for (unsigned int i = 0; i < pot_law_exponents_->size(); ++i)
      if (pot_law_exponents_->at(i) <= 0)
        FOUR_C_THROW(
            "Only positive values are allowed for potential law exponent."
            " Check your input file");
  }

  /****************************************************************************/
  strategy_ =
      Teuchos::getIntegralValue<BeamPotential::Strategy>(beam_potential_params_list, "STRATEGY");

  if (strategy_ == BeamPotential::Strategy::vague)
    FOUR_C_THROW("You must specify a strategy to be used to evaluate beam interaction potential!");

  /****************************************************************************/
  potential_type_ =
      Teuchos::getIntegralValue<BeamPotential::Type>(beam_potential_params_list, "TYPE");

  if (potential_type_ == BeamPotential::Type::vague)
    FOUR_C_THROW("You must specify the type of the specified beam interaction potential!");

  if (potential_type_ == BeamPotential::Type::surface and
      strategy_ != BeamPotential::Strategy::double_length_specific_large_separations)
  {
    FOUR_C_THROW("Surface interaction is not implemented for this strategy yet!");
  }

  /****************************************************************************/
  cutoff_radius_ = beam_potential_params_list.get<double>("CUTOFF_RADIUS");

  if (cutoff_radius_ != -1.0 and cutoff_radius_ <= 0.0)
    FOUR_C_THROW("Invalid cutoff radius! Must be positive value or -1 to deactivate.");

  /****************************************************************************/
  regularization_type_ = Teuchos::getIntegralValue<BeamPotential::RegularizationType>(
      beam_potential_params_list, "REGULARIZATION_TYPE");

  if ((regularization_type_ != BeamPotential::RegularizationType::none and
          strategy_ == BeamPotential::Strategy::double_length_specific_large_separations) or
      (regularization_type_ == BeamPotential::RegularizationType::constant and
          strategy_ == BeamPotential::Strategy::single_length_specific_small_separations))
  {
    FOUR_C_THROW(
        "This kind of regularization of the force law is not implemented for this strategy yet!");
  }

  /****************************************************************************/
  regularization_separation_ = beam_potential_params_list.get<double>("REGULARIZATION_SEPARATION");

  if (regularization_type_ != BeamPotential::RegularizationType::none and
      regularization_separation_ <= 0.0)
  {
    FOUR_C_THROW(
        "Invalid regularization separation! Must be a positive value since force law "
        "is not defined for separations <= 0!");
  }

  /****************************************************************************/
  num_integration_segments_ = beam_potential_params_list.get<int>("NUM_INTEGRATION_SEGMENTS");

  if (num_integration_segments_ <= 0)
    FOUR_C_THROW("Invalid number of integration segments per element!");

  /****************************************************************************/
  num_gp_s_ = beam_potential_params_list.get<int>("NUM_GAUSSPOINTS");

  if (num_gp_s_ <= 0) FOUR_C_THROW("Invalid number of Gauss points per integration segment!");

  /****************************************************************************/
  use_fad_ = beam_potential_params_list.get<bool>("AUTOMATIC_DIFFERENTIATION");

  /****************************************************************************/
  choice_master_slave_ = Teuchos::getIntegralValue<BeamPotential::MasterSlaveChoice>(
      beam_potential_params_list, "CHOICE_MASTER_SLAVE");

  if (choice_master_slave_ == BeamPotential::MasterSlaveChoice::vague)
  {
    FOUR_C_THROW("Invalid choice of master and slave!");
  }

  /****************************************************************************/
  // check for vtk output which is to be handled by an own writer object
  visualization_output_ = beam_potential_params_list.sublist("RUNTIME VTK OUTPUT")
                              .get<bool>("VTK_OUTPUT_BEAM_POTENTIAL");

  // create and initialize parameter container object for runtime output
  if (visualization_output_)
  {
    params_runtime_visualization_output_btb_potential_ =
        std::make_shared<BeamInteraction::BeamToBeamPotentialRuntimeOutputParams>(restart_time);

    params_runtime_visualization_output_btb_potential_->init(
        beam_potential_params_list.sublist("RUNTIME VTK OUTPUT"));
    params_runtime_visualization_output_btb_potential_->setup();
  }

  /****************************************************************************/
  potential_reduction_length_ =
      beam_potential_params_list.get<double>("POTENTIAL_REDUCTION_LENGTH");

  if (potential_reduction_length_ != -1.0 and potential_reduction_length_ <= 0.0)
    FOUR_C_THROW("Invalid potential reduction length! Must be positive value or -1 to deactivate.");

  isinit_ = true;
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void BeamInteraction::BeamPotentialParams::setup()
{
  throw_error_if_not_init();

  // empty for now

  issetup_ = true;
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void BeamInteraction::BeamPotentialParams::throw_error_if_not_init_and_setup() const
{
  if (!is_init() or !is_setup()) FOUR_C_THROW("Call init() and setup() first!");
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void BeamInteraction::BeamPotentialParams::throw_error_if_not_init() const
{
  if (!is_init()) FOUR_C_THROW("init() has not been called, yet!");
}

FOUR_C_NAMESPACE_CLOSE

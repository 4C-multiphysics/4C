// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_spherebeamlinking_params.hpp"

#include "4C_global_data.hpp"
#include "4C_inpar_beaminteraction.hpp"
#include "4C_io_value_parser.hpp"
#include "4C_mat_crosslinkermat.hpp"
#include "4C_structure_new_timint_basedataglobalstate.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
BeamInteraction::SphereBeamLinkingParams::SphereBeamLinkingParams()
    : isinit_(false), issetup_(false), deltatime_(-1.0), own_deltatime_(true)
{
  mat_.clear();
  contractionrate_.clear();
  maxnumlinkerpertype_.clear();
  matlinkerpertype_.clear();
  linkertypes_.clear();
  filamentbspotintervalglobal_.clear();
  filamentbspotintervallocal_.clear();
  filamentbspotrangeglobal_.clear();
  filamentbspotrangelocal_.clear();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::SphereBeamLinkingParams::init(
    Solid::TimeInt::BaseDataGlobalState const& gstate)
{
  issetup_ = false;

  const Teuchos::ParameterList& spherebeamlink_params_list =
      Global::Problem::instance()->beam_interaction_params().sublist("SPHERE BEAM LINK");

  // time step for stochastic events concering crosslinking
  deltatime_ = spherebeamlink_params_list.get<double>("TIMESTEP");

  // safety check
  // todo: maybe make input of time step obligatory
  if (deltatime_ < 0.0)
  {
    own_deltatime_ = false;
    deltatime_ = (*gstate.get_delta_time())[0];
    if (gstate.get_my_rank() == 0)
    {
      std::cout << " Time step " << (*gstate.get_delta_time())[0]
                << " from Structural Dynamic section "
                   "used for sphere beam link.\n"
                   "Force dependent unbinding of beam-sphere linker is activated for dt > 0"
                << std::endl;
    }
  }

  // number of linker in simulation volume
  {
    maxnumlinkerpertype_.clear();
    std::string max_num_linker_per_type_in(
        Teuchos::getNumericStringParameter(spherebeamlink_params_list, "MAXNUMLINKERPERTYPE"));

    Core::IO::ValueParser max_num_linker_per_type_parser(max_num_linker_per_type_in,
        {.user_scope_message = "While reading max number of linker per type: "});

    while (!max_num_linker_per_type_parser.at_end())
    {
      maxnumlinkerpertype_.push_back(max_num_linker_per_type_parser.read<int>());
    }

    // safety check
    for (unsigned int i = 0; i < maxnumlinkerpertype_.size(); ++i)
      if (maxnumlinkerpertype_[i] < 0)
        FOUR_C_THROW("Negative number of linker does not make sense!");
  }

  // material numbers for crosslinker types
  {
    matlinkerpertype_.clear();
    std::string material_linker_per_type_in(
        Teuchos::getNumericStringParameter(spherebeamlink_params_list, "MATLINKERPERTYPE"));

    Core::IO::ValueParser material_linker_per_type_parser(material_linker_per_type_in,
        {.user_scope_message = "While reading material linker per type: "});

    while (!material_linker_per_type_parser.at_end())
    {
      matlinkerpertype_.push_back(material_linker_per_type_parser.read<int>());
    }

    for (unsigned int i = 0; i < matlinkerpertype_.size(); ++i)
    {
      // safety check
      if (matlinkerpertype_[i] < 0) FOUR_C_THROW("Negative material number does not make sense!");

      // store materials
      mat_.push_back(
          std::dynamic_pointer_cast<Mat::CrosslinkerMat>(Mat::factory(matlinkerpertype_[i])));
      if (mat_.back() == nullptr) FOUR_C_THROW("Invalid material given for beam sphere link. \n");
    }
  }

  // safety check
  if (maxnumlinkerpertype_.size() != matlinkerpertype_.size())
    FOUR_C_THROW("Number of crosslinker types does not fit number of assigned materials");

  // store number of different linker types
  linkertypes_.clear();
  for (unsigned int type_i = 0; type_i < matlinkerpertype_.size(); ++type_i)
  {
    if (not(std::find(linkertypes_.begin(), linkertypes_.end(), matlinkerpertype_[type_i]) !=
            linkertypes_.end()))
    {
      linkertypes_.push_back(
          std::dynamic_pointer_cast<Mat::CrosslinkerMat>(Mat::factory(matlinkerpertype_[type_i]))
              ->linker_type());
    }
  }

  // store contraction rate, each linker type (not material) can have its own
  {
    contractionrate_.clear();
    std::string contraction_rate_in(
        Teuchos::getNumericStringParameter(spherebeamlink_params_list, "CONTRACTIONRATE"));

    Core::IO::ValueParser contraction_rate_parser(
        contraction_rate_in, {.user_scope_message = "While reading contraction rate: "});

    int count = 0;
    while (!contraction_rate_parser.at_end())
    {
      contractionrate_[linkertypes_[count]] = contraction_rate_parser.read<double>();
      ++count;
    }
  }

  if (contractionrate_.size() != linkertypes_.size())
    FOUR_C_THROW("You need to specify a contractionrate for each linker type. Exciting ... ");

  // distance between the two binding spots on each filament the same
  {
    filamentbspotintervalglobal_.clear();
    std::string filament_bspot_interval_global_in(Teuchos::getNumericStringParameter(
        spherebeamlink_params_list, "FILAMENTBSPOTINTERVALGLOBAL"));

    Core::IO::ValueParser filament_bspot_interval_global_parser(filament_bspot_interval_global_in,
        {.user_scope_message = "While reading filament binding spot interval global: "});

    int count = 0;
    while (!filament_bspot_interval_global_parser.at_end())
    {
      filamentbspotintervalglobal_[linkertypes_[count]] =
          filament_bspot_interval_global_parser.read<double>();
      ++count;
    }
  }

  // distance between the two binding spots on a filament as percentage of current filament
  // reference length
  {
    filamentbspotintervallocal_.clear();
    std::string filament_bspot_interval_local_in(Teuchos::getNumericStringParameter(
        spherebeamlink_params_list, "FILAMENTBSPOTINTERVALLOCAL"));

    Core::IO::ValueParser filament_bspot_interval_local_parser(filament_bspot_interval_local_in,
        {.user_scope_message = "While reading filament binding spot interval local: "});

    int count = 0;
    while (!filament_bspot_interval_local_parser.at_end())
    {
      filamentbspotintervallocal_[linkertypes_[count]] =
          filament_bspot_interval_local_parser.read<double>();
      ++count;
    }
  }

  if (linkertypes_.size() != filamentbspotintervalglobal_.size() and
      linkertypes_.size() != filamentbspotintervallocal_.size())
    FOUR_C_THROW("You need to specify filament binding spots for all your linker types!");

  // safety checks for feasibility of input
  if (filamentbspotintervalglobal_.size() == filamentbspotintervallocal_.size())
  {
    for (auto const& iter : filamentbspotintervalglobal_)
    {
      // safety feasibility checks
      if (iter.second <= 0.0 and not(filamentbspotintervallocal_.at(iter.first) > 0.0 and
                                     filamentbspotintervallocal_.at(iter.first) <= 1.0))
        FOUR_C_THROW(
            " Choose realistic value for FILAMENTBSPOTINTERVAL (i.e. distance between "
            "two binding spots on a filament) in input file. ");
      if (iter.second > 0.0 and filamentbspotintervallocal_.at(iter.first) > 0.0)
        FOUR_C_THROW(" You can only set either a global or a local filament binding spot interval");
    }
  }

  // start and end arc parameter for binding spots on a filament
  {
    filamentbspotrangeglobal_.clear();
    std::string filament_bspot_range_global_in(
        Teuchos::getNumericStringParameter(spherebeamlink_params_list, "FILAMENTBSPOTRANGEGLOBAL"));

    Core::IO::ValueParser filament_bspot_range_global_parser(filament_bspot_range_global_in,
        {.user_scope_message = "While reading filament binding spot range global: "});

    int count = 0;
    while (!filament_bspot_range_global_parser.at_end())
    {
      std::pair<double, double> pair;
      pair.first = filament_bspot_range_global_parser.read<double>();
      if (!filament_bspot_range_global_parser.at_end())
        pair.second = filament_bspot_range_global_parser.read<double>();
      else
        FOUR_C_THROW("Filament binding spot range needs to be specified via two values");

      // store global range
      filamentbspotrangeglobal_[linkertypes_[count]] = pair;

      if (pair.first > 0.0 and pair.second > 0.0 and (pair.first > pair.second))
        FOUR_C_THROW(" lower bound > upper bound, fix FILAMENTBSPOTRANGEGLOBAL in input file ");

      ++count;
    }
  }

  // start and end arc parameter for binding spots on a filament
  {
    filamentbspotrangelocal_.clear();
    std::string filament_bspot_range_local_in(
        Teuchos::getNumericStringParameter(spherebeamlink_params_list, "FILAMENTBSPOTRANGELOCAL"));

    Core::IO::ValueParser filament_bspot_range_local_parser(filament_bspot_range_local_in,
        {.user_scope_message = "While reading filament binding spot range local: "});

    int count = 0;
    while (!filament_bspot_range_local_parser.at_end())
    {
      std::pair<double, double> pair;
      pair.first = filament_bspot_range_local_parser.read<double>();
      if (!filament_bspot_range_local_parser.at_end())
        pair.second = filament_bspot_range_local_parser.read<double>();
      else
        FOUR_C_THROW("Filament binding spot range needs to be specified via two values");

      // store local range
      filamentbspotrangelocal_[linkertypes_[count]] = pair;

      if (pair.first > 0.0 and pair.second > 0.0 and (pair.first > pair.second))
        FOUR_C_THROW(" lower bound > upper bound, fix FILAMENTBSPOTRANGEGLOCAL in input file ");
      if (pair.first > 1.0 or pair.second > 1.0)
        FOUR_C_THROW("values > 1.0 do not make sense for local filament binding spot range");

      ++count;
    }
  }

  if (linkertypes_.size() != filamentbspotrangeglobal_.size() and
      linkertypes_.size() != filamentbspotrangelocal_.size())
    FOUR_C_THROW("You need to specify filament binding spots for all your linker types");

  // safety checks for feasibility of input
  if (filamentbspotrangeglobal_.size() == filamentbspotrangelocal_.size())
  {
    for (auto const& iter : filamentbspotrangeglobal_)
    {
      if (filamentbspotrangelocal_.at(iter.first).first > 0.0 and
          filamentbspotrangelocal_.at(iter.first).second > 0.0 and
          (iter.second.first > 0.0 or iter.second.second > 0.0))
        FOUR_C_THROW("either local or global binding spot range can be specified");
    }
  }


  isinit_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::SphereBeamLinkingParams::setup()
{
  check_init();

  // empty for now

  issetup_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::SphereBeamLinkingParams::reset_time_step(double structure_delta_time)
{
  check_init_setup();

  if (not own_deltatime_) deltatime_ = structure_delta_time;
}

FOUR_C_NAMESPACE_CLOSE

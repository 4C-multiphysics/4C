// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_crosslinking_params.hpp"

#include "4C_global_data.hpp"
#include "4C_io_value_parser.hpp"
#include "4C_mat_crosslinkermat.hpp"
#include "4C_structure_new_timint_basedataglobalstate.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
BeamInteraction::CrosslinkingParams::CrosslinkingParams()
    : isinit_(false), issetup_(false), viscosity_(0.0), kt_(0.0), deltatime_(0.0), init_box_(true)
{
  maxnum_init_crosslinker_pertype_.clear();
  numcrosslinkerpertype_.clear();
  matcrosslinkerpertype_.clear();
  linkertypes_.clear();
  max_num_bonds_per_filament_bspot_.clear();
  filamentbspotintervalglobal_.clear();
  filamentbspotintervallocal_.clear();
  filamentbspotrangeglobal_.clear();
  filamentbspotrangelocal_.clear();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::CrosslinkingParams::init(Solid::TimeInt::BaseDataGlobalState const& gstate)
{
  issetup_ = false;

  const Teuchos::ParameterList& crosslinking_params_list =
      Global::Problem::instance()->beam_interaction_params().sublist("CROSSLINKING");

  // viscosity
  viscosity_ = crosslinking_params_list.get<double>("VISCOSITY");

  // thermal energy
  kt_ = crosslinking_params_list.get<double>("KT");

  // time step for stochastic events concering crosslinking
  deltatime_ = crosslinking_params_list.get<double>("TIMESTEP");

  init_box_.put_scalar(1.0e12);
  std::istringstream init_box_stream(
      Teuchos::getNumericStringParameter(crosslinking_params_list, "INIT_LINKER_BOUNDINGBOX"));
  for (int col = 0; col < 2; ++col)
  {
    for (int row = 0; row < 3; ++row)
    {
      double value = 1.0e12;
      if (init_box_stream >> value)
        init_box_(row, col) = value;
      else
      {
        FOUR_C_THROW(
            " Specify six values for bounding box in three dimensional problem."
            " Fix your input file.");
      }
    }
  }

  bool feasibleboxinput = true;
  for (int col = 0; col < 2; ++col)
    for (int row = 0; row < 3; ++row)
      if (init_box_(row, col) > 1.0e11) feasibleboxinput = false;

  if (not feasibleboxinput)
  {
    std::istringstream pbb_stream(Teuchos::getNumericStringParameter(
        Global::Problem::instance()->binning_strategy_params(), "DOMAINBOUNDINGBOX"));
    for (int col = 0; col < 2; ++col)
    {
      for (int row = 0; row < 3; ++row)
      {
        double value = 1.0e12;
        if (pbb_stream >> value)
          init_box_(row, col) = value;
        else
        {
          FOUR_C_THROW(
              " Specify six values for bounding box in three dimensional problem."
              " Fix your input file.");
        }
      }
    }
  }

  // safety check
  // todo: maybe make input of time step obligatory
  if (deltatime_ < 0.0)
  {
    deltatime_ = (*gstate.get_delta_time())[0];
    if (gstate.get_my_rank() == 0)
    {
      std::cout << " Time step " << (*gstate.get_delta_time())[0]
                << " form Structural Dynamic section "
                   "used for crosslinking.\n"
                << std::endl;
    }
  }

  // number of linker in simulation volume
  {
    numcrosslinkerpertype_.clear();
    std::string crosslinker_type_in(
        Teuchos::getNumericStringParameter(crosslinking_params_list, "NUMCROSSLINKERPERTYPE"));

    Core::IO::ValueParser crosslinker_type_parser(
        crosslinker_type_in, {.user_scope_message = "While reading crosslinker type: "});

    while (!crosslinker_type_parser.at_end())
    {
      numcrosslinkerpertype_.push_back(crosslinker_type_parser.read<int>());
    }

    // safety check
    for (int i = 0; i < static_cast<int>(numcrosslinkerpertype_.size()); ++i)
      if (numcrosslinkerpertype_[i] < 0)
        FOUR_C_THROW(" negative number of crosslinker does not make sense.");
  }

  // material numbers for crosslinker types
  {
    matcrosslinkerpertype_.clear();
    std::string crosslinker_material_in(
        Teuchos::getNumericStringParameter(crosslinking_params_list, "MATCROSSLINKERPERTYPE"));

    Core::IO::ValueParser crosslinker_material_parser(
        crosslinker_material_in, {.user_scope_message = "While reading crosslinker material: "});

    while (!crosslinker_material_parser.at_end())
    {
      matcrosslinkerpertype_.push_back(crosslinker_material_parser.read<int>());
    }

    // safety check
    for (int i = 0; i < static_cast<int>(matcrosslinkerpertype_.size()); ++i)
      if (matcrosslinkerpertype_[i] < 0)
        FOUR_C_THROW(" negative material number does not make sense.");
  }

  // safety check
  if (numcrosslinkerpertype_.size() != matcrosslinkerpertype_.size())
    FOUR_C_THROW("number of crosslinker types does not fit number of assigned materials");

  // compute number of linker types
  linkertypes_.clear();
  for (unsigned int type_i = 0; type_i < matcrosslinkerpertype_.size(); ++type_i)
  {
    if (not(std::find(linkertypes_.begin(), linkertypes_.end(), matcrosslinkerpertype_[type_i]) !=
            linkertypes_.end()))
    {
      linkertypes_.push_back(std::dynamic_pointer_cast<Mat::CrosslinkerMat>(
          Mat::factory(matcrosslinkerpertype_[type_i]))
                                 ->linker_type());
    }
  }

  // number of initially set linker
  {
    maxnum_init_crosslinker_pertype_.clear();
    std::vector<int> maxnuminitcrosslinkerpertype;
    std::string num_crosslinker_per_type_in(Teuchos::getNumericStringParameter(
        crosslinking_params_list, "MAXNUMINITCROSSLINKERPERTYPE"));

    Core::IO::ValueParser num_crosslinker_per_type_parser(num_crosslinker_per_type_in,
        {.user_scope_message = "While reading number of initial crosslinker: "});

    while (!num_crosslinker_per_type_parser.at_end())
    {
      maxnuminitcrosslinkerpertype.push_back(num_crosslinker_per_type_parser.read<int>());
    }

    if (maxnuminitcrosslinkerpertype.size() > 1 or maxnuminitcrosslinkerpertype[0] != 0)
    {
      // safety checks
      for (int i = 0; i < static_cast<int>(maxnuminitcrosslinkerpertype.size()); ++i)
        if (maxnuminitcrosslinkerpertype[i] < 0)
          FOUR_C_THROW(" negative number of crosslinker does not make sense.");
      if (maxnuminitcrosslinkerpertype.size() != numcrosslinkerpertype_.size())
        FOUR_C_THROW(
            "number of initial set crosslinker types does not fit number of crosslinker types");

      for (int i = 0; i < static_cast<int>(maxnuminitcrosslinkerpertype.size()); ++i)
        maxnum_init_crosslinker_pertype_[matcrosslinkerpertype_[i]] =
            maxnuminitcrosslinkerpertype[i];
    }
  }

  // maximal number of bonds per filament binding spot
  {
    max_num_bonds_per_filament_bspot_.clear();
    std::string max_num_bonds_per_filament_bspot_in(Teuchos::getNumericStringParameter(
        crosslinking_params_list, "MAXNUMBONDSPERFILAMENTBSPOT"));

    Core::IO::ValueParser max_num_bonds_per_filament_bspot_parser(
        max_num_bonds_per_filament_bspot_in,
        {.user_scope_message = "While reading max number of bonds per filament binding spot: "});

    int count = 0;
    while (!max_num_bonds_per_filament_bspot_parser.at_end())
    {
      max_num_bonds_per_filament_bspot_[linkertypes_[count]] =
          max_num_bonds_per_filament_bspot_parser.read<int>();
      if (max_num_bonds_per_filament_bspot_.at(linkertypes_[count]) < 0)
        FOUR_C_THROW(" Choose a number of bonds per filament binding spot type >= 0. ");
      ++count;
    }

    if (max_num_bonds_per_filament_bspot_.size() != linkertypes_.size())
    {
      FOUR_C_THROW(
          " Num linker types %i does not match num input for MAXNUMBONDSPERFILAMENTBSPOT %i. ",
          linkertypes_.size(), max_num_bonds_per_filament_bspot_.size());
    }
  }

  // distance between the two binding spots on each filament the same
  {
    filamentbspotintervalglobal_.clear();
    std::string filament_interval_bspot_in(Teuchos::getNumericStringParameter(
        crosslinking_params_list, "FILAMENTBSPOTINTERVALGLOBAL"));

    Core::IO::ValueParser filament_interval_bspot_parser(filament_interval_bspot_in,
        {.user_scope_message = "While reading filament binding spot interval global: "});

    int count = 0;
    while (!filament_interval_bspot_parser.at_end())
    {
      filamentbspotintervalglobal_[linkertypes_[count]] =
          filament_interval_bspot_parser.read<double>();
      ++count;
    }
  }

  // distance between the two binding spots on a filament as percentage of current filament
  // reference length
  {
    filamentbspotintervallocal_.clear();
    std::string filament_bspot_interval_local_in(
        Teuchos::getNumericStringParameter(crosslinking_params_list, "FILAMENTBSPOTINTERVALLOCAL"));

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
    FOUR_C_THROW("You need to specify filament binding spots for all your linker types");

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
        Teuchos::getNumericStringParameter(crosslinking_params_list, "FILAMENTBSPOTRANGEGLOBAL"));

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
        Teuchos::getNumericStringParameter(crosslinking_params_list, "FILAMENTBSPOTRANGELOCAL"));

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
void BeamInteraction::CrosslinkingParams::setup()
{
  check_init();

  // empty for now

  issetup_ = true;
}

FOUR_C_NAMESPACE_CLOSE

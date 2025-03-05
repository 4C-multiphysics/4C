// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_beaminteraction.hpp"

#include "4C_beamcontact_input.hpp"
#include "4C_fem_condition_definition.hpp"
#include "4C_inpar_beam_to_solid.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN


void Inpar::BeamInteraction::beam_interaction_conditions_get_all(
    std::vector<Inpar::BeamInteraction::BeamInteractionConditions>& interactions)
{
  interactions = {Inpar::BeamInteraction::BeamInteractionConditions::beam_to_beam_contact,
      Inpar::BeamInteraction::BeamInteractionConditions::beam_to_beam_point_coupling,
      Inpar::BeamInteraction::BeamInteractionConditions::beam_to_solid_volume_meshtying,
      Inpar::BeamInteraction::BeamInteractionConditions::beam_to_solid_surface_meshtying,
      Inpar::BeamInteraction::BeamInteractionConditions::beam_to_solid_surface_contact};
}

void Inpar::BeamInteraction::set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list)
{
  using Teuchos::tuple;
  using namespace Core::IO::InputSpecBuilders;

  Core::Utils::SectionSpecs beaminteraction{"BEAM INTERACTION"};

  Core::Utils::string_to_integral_parameter<Inpar::BeamInteraction::RepartitionStrategy>(
      "REPARTITIONSTRATEGY", "Adaptive", "Type of employed repartitioning strategy",
      tuple<std::string>("Adaptive", "adaptive", "Everydt", "everydt"),
      tuple<Inpar::BeamInteraction::RepartitionStrategy>(
          repstr_adaptive, repstr_adaptive, repstr_everydt, repstr_everydt),
      beaminteraction);

  Core::Utils::string_to_integral_parameter<SearchStrategy>("SEARCH_STRATEGY",
      "bruteforce_with_binning", "Type of search strategy used for finding coupling pairs",
      tuple<std::string>("bruteforce_with_binning", "bounding_volume_hierarchy"),
      tuple<SearchStrategy>(
          SearchStrategy::bruteforce_with_binning, SearchStrategy::bounding_volume_hierarchy),
      beaminteraction);

  beaminteraction.move_into_collection(list);

  /*----------------------------------------------------------------------*/
  /* parameters for crosslinking submodel */

  Core::Utils::SectionSpecs crosslinking{beaminteraction, "CROSSLINKING"};

  // remove this some day
  crosslinking.specs.emplace_back(parameter<bool>(
      "CROSSLINKER", {.description = "Crosslinker in problem", .default_value = false}));

  // bounding box for initial random crosslinker position
  crosslinking.specs.emplace_back(parameter<std::string>("INIT_LINKER_BOUNDINGBOX",
      {.description = "Linker are initially set randomly within this bounding box",
          .default_value = "1e12 1e12 1e12 1e12 1e12 1e12"}));

  // time step for stochastic events concerning crosslinking
  crosslinking.specs.emplace_back(parameter<double>(
      "TIMESTEP", {.description = "time step for stochastic events concerning crosslinking (e.g. "
                                  "diffusion, p_link, p_unlink) ",
                      .default_value = -1.0}));
  // Reading double parameter for viscosity of background fluid
  crosslinking.specs.emplace_back(
      parameter<double>("VISCOSITY", {.description = "viscosity", .default_value = 0.0}));
  // Reading double parameter for thermal energy in background fluid (temperature * Boltzmann
  // constant)
  crosslinking.specs.emplace_back(
      parameter<double>("KT", {.description = "thermal energy", .default_value = 0.0}));
  // number of initial (are set right in the beginning) crosslinker of certain type
  crosslinking.specs.emplace_back(parameter<std::string>(
      "MAXNUMINITCROSSLINKERPERTYPE", {.description = "number of initial crosslinker of certain "
                                                      "type (additional to NUMCROSSLINKERPERTYPE) ",
                                          .default_value = "0"}));
  // number of crosslinker of certain type
  crosslinking.specs.emplace_back(parameter<std::string>("NUMCROSSLINKERPERTYPE",
      {.description = "number of crosslinker of certain type ", .default_value = "0"}));
  // material number characterizing crosslinker type
  crosslinking.specs.emplace_back(parameter<std::string>("MATCROSSLINKERPERTYPE",
      {.description = "material number characterizing crosslinker type ", .default_value = "-1"}));
  // maximal number of binding partner per filament binding spot for each binding spot type
  crosslinking.specs.emplace_back(parameter<std::string>("MAXNUMBONDSPERFILAMENTBSPOT",
      {.description = "maximal number of bonds per filament binding spot", .default_value = "1"}));
  // distance between two binding spots on a filament (same on all filaments)
  crosslinking.specs.emplace_back(parameter<std::string>("FILAMENTBSPOTINTERVALGLOBAL",
      {.description = "distance between two binding spots on all filaments",
          .default_value = "-1.0"}));
  // distance between two binding spots on a filament (as percentage of current filament length)
  crosslinking.specs.emplace_back(parameter<std::string>("FILAMENTBSPOTINTERVALLOCAL",
      {.description = "distance between two binding spots on current filament",
          .default_value = "-1.0"}));
  // start and end for bspots on a filament in arc parameter (same on each filament independent of
  // their length)
  crosslinking.specs.emplace_back(parameter<std::string>("FILAMENTBSPOTRANGEGLOBAL",
      {.description = "Lower and upper arc parameter bound for binding spots on a filament",
          .default_value = "-1.0 -1.0"}));
  // start and end for bspots on a filament in percent of reference filament length
  crosslinking.specs.emplace_back(parameter<std::string>("FILAMENTBSPOTRANGELOCAL",
      {.description = "Lower and upper arc parameter bound for binding spots on a filament",
          .default_value = "0.0 1.0"}));

  crosslinking.move_into_collection(list);


  /*----------------------------------------------------------------------*/
  /* parameters for sphere beam link submodel */

  Core::Utils::SectionSpecs spherebeamlink{beaminteraction, "SPHERE BEAM LINK"};

  spherebeamlink.specs.emplace_back(parameter<bool>(
      "SPHEREBEAMLINKING", {.description = "Integrins in problem", .default_value = false}));

  // Reading double parameter for contraction rate for active linker
  spherebeamlink.specs.emplace_back(parameter<double>(
      "CONTRACTIONRATE", {.description = "contraction rate of cell (integrin linker) in [microm/s]",
                             .default_value = 0.0}));
  // time step for stochastic events concerning sphere beam linking
  spherebeamlink.specs.emplace_back(parameter<double>(
      "TIMESTEP", {.description = "time step for stochastic events concerning sphere beam linking "
                                  "(e.g. catch-slip-bond behavior) ",
                      .default_value = -1.0}));
  spherebeamlink.specs.emplace_back(parameter<std::string>("MAXNUMLINKERPERTYPE",
      {.description = "number of crosslinker of certain type ", .default_value = "0"}));
  // material number characterizing crosslinker type
  spherebeamlink.specs.emplace_back(parameter<std::string>("MATLINKERPERTYPE",
      {.description = "material number characterizing crosslinker type ", .default_value = "-1"}));
  // distance between two binding spots on a filament (same on all filaments)
  spherebeamlink.specs.emplace_back(parameter<std::string>("FILAMENTBSPOTINTERVALGLOBAL",
      {.description = "distance between two binding spots on all filaments",
          .default_value = "-1.0"}));
  // distance between two binding spots on a filament (as percentage of current filament length)
  spherebeamlink.specs.emplace_back(parameter<std::string>("FILAMENTBSPOTINTERVALLOCAL",
      {.description = "distance between two binding spots on current filament",
          .default_value = "-1.0"}));
  // start and end for bspots on a filament in arc parameter (same on each filament independent of
  // their length)
  spherebeamlink.specs.emplace_back(parameter<std::string>("FILAMENTBSPOTRANGEGLOBAL",
      {.description = "Lower and upper arc parameter bound for binding spots on a filament",
          .default_value = "-1.0 -1.0"}));
  // start and end for bspots on a filament in percent of reference filament length
  spherebeamlink.specs.emplace_back(parameter<std::string>("FILAMENTBSPOTRANGELOCAL",
      {.description = "Lower and upper arc parameter bound for binding spots on a filament",
          .default_value = "0.0 1.0"}));

  spherebeamlink.move_into_collection(list);

  /*----------------------------------------------------------------------*/
  /* parameters for beam to ? contact submodel*/
  /*----------------------------------------------------------------------*/

  /*----------------------------------------------------------------------*/
  /* parameters for beam to beam contact */

  Core::Utils::SectionSpecs beamtobeamcontact{beaminteraction, "BEAM TO BEAM CONTACT"};

  Core::Utils::string_to_integral_parameter<Inpar::BeamInteraction::Strategy>("STRATEGY", "None",
      "Type of employed solving strategy", tuple<std::string>("None", "none", "Penalty", "penalty"),
      tuple<Inpar::BeamInteraction::Strategy>(bstr_none, bstr_none, bstr_penalty, bstr_penalty),
      beamtobeamcontact);

  beamtobeamcontact.move_into_collection(list);

  // ...

  /*----------------------------------------------------------------------*/
  /* parameters for beam to sphere contact */

  Core::Utils::SectionSpecs beamtospherecontact{beaminteraction, "BEAM TO SPHERE CONTACT"};

  Core::Utils::string_to_integral_parameter<Inpar::BeamInteraction::Strategy>("STRATEGY", "None",
      "Type of employed solving strategy", tuple<std::string>("None", "none", "Penalty", "penalty"),
      tuple<Inpar::BeamInteraction::Strategy>(bstr_none, bstr_none, bstr_penalty, bstr_penalty),
      beamtospherecontact);

  beamtospherecontact.specs.emplace_back(parameter<double>("PENALTY_PARAMETER",
      {.description = "Penalty parameter for beam-to-rigidsphere contact", .default_value = 0.0}));

  beamtospherecontact.move_into_collection(list);

  // ...

  /*----------------------------------------------------------------------*/
  /* parameters for beam to solid contact */
  BeamToSolid::set_valid_parameters(list);
}

void Inpar::BeamInteraction::set_valid_conditions(
    std::vector<Core::Conditions::ConditionDefinition>& condlist)
{
  using namespace Core::IO::InputSpecBuilders;

  /*-------------------------------------------------------------------*/
  // beam potential interaction: atom/charge density per unit length on LINE
  Core::Conditions::ConditionDefinition beam_filament_condition(
      "DESIGN LINE BEAM FILAMENT CONDITIONS", "BeamLineFilamentCondition",
      "Beam_Line_Filament_Condition", Core::Conditions::FilamentBeamLineCondition, false,
      Core::Conditions::geometry_type_line);

  beam_filament_condition.add_component(parameter<int>("ID", {.description = "filament id"}));
  beam_filament_condition.add_component(selection<std::string>("TYPE",
      {"Arbitrary", "arbitrary", "Actin", "actin", "Collagen", "collagen"},
      {.description = "", .default_value = "Arbitrary"}));

  condlist.push_back(beam_filament_condition);

  /*-------------------------------------------------------------------*/
  Core::Conditions::ConditionDefinition penalty_coupling_condition(
      "DESIGN POINT PENALTY COUPLING CONDITIONS", "PenaltyPointCouplingCondition",
      "Couples beam nodes that lie on the same position",
      Core::Conditions::PenaltyPointCouplingCondition, false,
      Core::Conditions::geometry_type_point);

  penalty_coupling_condition.add_component(parameter<double>("POSITIONAL_PENALTY_PARAMETER"));
  penalty_coupling_condition.add_component(parameter<double>("ROTATIONAL_PENALTY_PARAMETER"));

  condlist.push_back(penalty_coupling_condition);

  // beam-to-beam interactions
  BeamContact::set_valid_conditions(condlist);

  // beam-to-solid interactions
  Inpar::BeamToSolid::set_valid_conditions(condlist);
}

FOUR_C_NAMESPACE_CLOSE

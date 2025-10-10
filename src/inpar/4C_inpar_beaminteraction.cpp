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
FOUR_C_NAMESPACE_OPEN


void Inpar::BeamInteraction::beam_interaction_conditions_get_all(
    std::vector<Inpar::BeamInteraction::BeamInteractionConditions>& interactions)
{
  interactions = {Inpar::BeamInteraction::BeamInteractionConditions::beam_to_beam_contact,
      Inpar::BeamInteraction::BeamInteractionConditions::beam_to_beam_point_coupling,
      Inpar::BeamInteraction::BeamInteractionConditions::beam_to_solid_volume_meshtying,
      Inpar::BeamInteraction::BeamInteractionConditions::beam_to_solid_surface_meshtying,
      Inpar::BeamInteraction::BeamInteractionConditions::beam_to_solid_surface_contact,
      Inpar::BeamInteraction::BeamInteractionConditions::beam_to_solid_edge_contact};
}

std::vector<Core::IO::InputSpec> Inpar::BeamInteraction::valid_parameters()
{
  using namespace Core::IO::InputSpecBuilders;

  std::vector<Core::IO::InputSpec> specs;
  specs.push_back(group("BEAM INTERACTION",
      {

          deprecated_selection<Inpar::BeamInteraction::RepartitionStrategy>("REPARTITIONSTRATEGY",
              {
                  {"Adaptive", repstr_adaptive},
                  {"adaptive", repstr_adaptive},
                  {"Everydt", repstr_everydt},
                  {"everydt", repstr_everydt},
              },
              {.description = "Type of employed repartitioning strategy",
                  .default_value = repstr_adaptive}),

          parameter<SearchStrategy>("SEARCH_STRATEGY",
              {.description = "Type of search strategy used for finding coupling pairs",
                  .default_value = SearchStrategy::bruteforce_with_binning})},
      {.required = false}));
  /*----------------------------------------------------------------------*/
  /* parameters for crosslinking submodel */
  specs.push_back(group("BEAM INTERACTION/CROSSLINKING",
      {

          // remove this some day
          parameter<bool>(
              "CROSSLINKER", {.description = "Crosslinker in problem", .default_value = false}),

          // bounding box for initial random crosslinker position
          parameter<std::string>("INIT_LINKER_BOUNDINGBOX",
              {.description = "Linker are initially set randomly within this bounding box",
                  .default_value = "1e12 1e12 1e12 1e12 1e12 1e12"}),

          // time step for stochastic events concerning crosslinking
          parameter<double>("TIMESTEP",
              {.description = "time step for stochastic events concerning crosslinking (e.g. "
                              "diffusion, p_link, p_unlink) ",
                  .default_value = -1.0}),
          // Reading double parameter for viscosity of background fluid
          parameter<double>("VISCOSITY", {.description = "viscosity", .default_value = 0.0}),
          // Reading double parameter for thermal energy in background fluid (temperature *
          // Boltzmann
          // constant)
          parameter<double>("KT", {.description = "thermal energy", .default_value = 0.0}),
          // number of initial (are set right in the beginning) crosslinker of certain type
          parameter<std::string>("MAXNUMINITCROSSLINKERPERTYPE",
              {.description = "number of initial crosslinker of certain "
                              "type (additional to NUMCROSSLINKERPERTYPE) ",
                  .default_value = "0"}),
          // number of crosslinker of certain type
          parameter<std::string>("NUMCROSSLINKERPERTYPE",
              {.description = "number of crosslinker of certain type ", .default_value = "0"}),
          // material number characterizing crosslinker type
          parameter<std::string>("MATCROSSLINKERPERTYPE",
              {.description = "material number characterizing crosslinker type ",
                  .default_value = "-1"}),
          // maximal number of binding partner per filament binding spot for each binding spot type
          parameter<std::string>("MAXNUMBONDSPERFILAMENTBSPOT",
              {.description = "maximal number of bonds per filament binding spot",
                  .default_value = "1"}),
          // distance between two binding spots on a filament (same on all filaments)
          parameter<std::string>("FILAMENTBSPOTINTERVALGLOBAL",
              {.description = "distance between two binding spots on all filaments",
                  .default_value = "-1.0"}),
          // distance between two binding spots on a filament (as percentage of current filament
          // length)
          parameter<std::string>("FILAMENTBSPOTINTERVALLOCAL",
              {.description = "distance between two binding spots on current filament",
                  .default_value = "-1.0"}),
          // start and end for bspots on a filament in arc parameter (same on each filament
          // independent
          // of
          // their length)
          parameter<std::string>("FILAMENTBSPOTRANGEGLOBAL",
              {.description = "Lower and upper arc parameter bound for binding spots on a filament",
                  .default_value = "-1.0 -1.0"}),
          // start and end for bspots on a filament in percent of reference filament length
          parameter<std::string>("FILAMENTBSPOTRANGELOCAL",
              {.description = "Lower and upper arc parameter bound for binding spots on a filament",
                  .default_value = "0.0 1.0"})},
      {.required = false}));


  /*----------------------------------------------------------------------*/
  /* parameters for sphere beam link submodel */
  specs.push_back(group("BEAM INTERACTION/SPHERE BEAM LINK",
      {

          parameter<bool>(
              "SPHEREBEAMLINKING", {.description = "Integrins in problem", .default_value = false}),

          // Reading double parameter for contraction rate for active linker
          parameter<double>("CONTRACTIONRATE",
              {.description = "contraction rate of cell (integrin linker) in [microm/s]",
                  .default_value = 0.0}),
          // time step for stochastic events concerning sphere beam linking
          parameter<double>("TIMESTEP",
              {.description = "time step for stochastic events concerning sphere beam linking "
                              "(e.g. catch-slip-bond behavior) ",
                  .default_value = -1.0}),
          parameter<std::string>("MAXNUMLINKERPERTYPE",
              {.description = "number of crosslinker of certain type ", .default_value = "0"}),
          // material number characterizing crosslinker type
          parameter<std::string>("MATLINKERPERTYPE",
              {.description = "material number characterizing crosslinker type ",
                  .default_value = "-1"}),
          // distance between two binding spots on a filament (same on all filaments)
          parameter<std::string>("FILAMENTBSPOTINTERVALGLOBAL",
              {.description = "distance between two binding spots on all filaments",
                  .default_value = "-1.0"}),
          // distance between two binding spots on a filament (as percentage of current filament
          // length)
          parameter<std::string>("FILAMENTBSPOTINTERVALLOCAL",
              {.description = "distance between two binding spots on current filament",
                  .default_value = "-1.0"}),
          // start and end for bspots on a filament in arc parameter (same on each filament
          // independent of their length)
          parameter<std::string>("FILAMENTBSPOTRANGEGLOBAL",
              {.description = "Lower and upper arc parameter bound for binding spots on a filament",
                  .default_value = "-1.0 -1.0"}),
          // start and end for bspots on a filament in percent of reference filament length
          parameter<std::string>("FILAMENTBSPOTRANGELOCAL",
              {.description = "Lower and upper arc parameter bound for binding spots on a filament",
                  .default_value = "0.0 1.0"})},
      {.required = false}));

  /*----------------------------------------------------------------------*/
  /* parameters for beam to ? contact submodel*/
  /*----------------------------------------------------------------------*/

  /*----------------------------------------------------------------------*/
  /* parameters for beam to beam contact */
  specs.push_back(group("BEAM INTERACTION/BEAM TO BEAM CONTACT",
      {

          deprecated_selection<Inpar::BeamInteraction::Strategy>("STRATEGY",
              {
                  {"None", bstr_none},
                  {"none", bstr_none},
                  {"Penalty", bstr_penalty},
                  {"penalty", bstr_penalty},
              },
              {.description = "Type of employed solving strategy", .default_value = bstr_none})},
      {.required = false}));

  // ...

  /*----------------------------------------------------------------------*/
  /* parameters for beam to sphere contact */
  specs.push_back(group("BEAM INTERACTION/BEAM TO SPHERE CONTACT",
      {

          deprecated_selection<Inpar::BeamInteraction::Strategy>("STRATEGY",
              {
                  {"None", bstr_none},
                  {"none", bstr_none},
                  {"Penalty", bstr_penalty},
                  {"penalty", bstr_penalty},
              },
              {.description = "Type of employed solving strategy", .default_value = bstr_none}),

          parameter<double>("PENALTY_PARAMETER",
              {.description = "Penalty parameter for beam-to-rigidsphere contact",
                  .default_value = 0.0})},
      {.required = false}));

  /*----------------------------------------------------------------------*/
  /* parameters for beam to solid contact */
  std::vector<Core::IO::InputSpec> beam_to_solid_contact = BeamToSolid::valid_parameters();
  specs.insert(specs.end(), beam_to_solid_contact.begin(), beam_to_solid_contact.end());
  return specs;
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
  beam_filament_condition.add_component(deprecated_selection<std::string>("TYPE",
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
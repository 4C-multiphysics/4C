// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_art_net_input.hpp"

#include "4C_fem_condition_definition.hpp"
#include "4C_io_input_spec_builders.hpp"
FOUR_C_NAMESPACE_OPEN


Core::IO::InputSpec ArtDyn::valid_parameters()
{
  using namespace Core::IO::InputSpecBuilders;
  Core::IO::InputSpec spec = group("ARTERIAL DYNAMIC",
      {

          deprecated_selection<TimeIntegrationScheme>("DYNAMICTYPE",
              {
                  {"ExpTaylorGalerkin", tay_gal},
                  {"Stationary", stationary},
              },
              {.description = "Explicit Taylor Galerkin Scheme", .default_value = tay_gal}),


          parameter<double>(
              "TIMESTEP", {.description = "Time increment dt", .default_value = 0.01}),

          parameter<int>("NUMSTEP", {.description = "Number of Time Steps", .default_value = 0}),
          parameter<double>(
              "MAXTIME", {.description = "total simulation time", .default_value = 1000.0}),
          parameter<int>(
              "RESTARTEVERY", {.description = "Increment for writing restart", .default_value = 1}),
          parameter<int>("RESULTSEVERY",
              {.description = "Increment for writing solution", .default_value = 1}),

          parameter<bool>("SOLVESCATRA",
              {.description = "Flag to (de)activate solving scalar transport in blood",
                  .default_value = false}),

          // number of linear solver used for arterial dynamics
          parameter<int>(
              "LINEAR_SOLVER", {.description = "number of linear solver used for arterial dynamics",
                                   .default_value = -1}),

          // initial function number
          parameter<int>("INITFUNCNO",
              {.description = "function number for artery initial field", .default_value = -1}),

          // type of initial field
          deprecated_selection<InitialField>("INITIALFIELD",
              {
                  {"zero_field", initfield_zero_field},
                  {"field_by_function", initfield_field_by_function},
                  {"field_by_condition", initfield_field_by_condition},
              },
              {.description = "Initial Field for artery problem",
                  .default_value = initfield_zero_field})},
      {.required = false});
  return spec;
}



void ArteryNetwork::set_valid_conditions(
    std::vector<Core::Conditions::ConditionDefinition>& condlist)
{
  using namespace Core::IO::InputSpecBuilders;

  /*--------------------------------------------------------------------*/
  // 1D-Artery connector condition

  Core::Conditions::ConditionDefinition art_connection_bc(
      "DESIGN NODE 1D ARTERY JUNCTION CONDITIONS", "ArtJunctionCond",
      "Artery junction boundary condition", Core::Conditions::ArtJunctionCond, true,
      Core::Conditions::geometry_type_point);

  art_connection_bc.add_component(parameter<int>("ConditionID"));
  art_connection_bc.add_component(parameter<double>("Kr"));

  condlist.push_back(art_connection_bc);

  /*--------------------------------------------------------------------*/
  // 1D artery prescribed BC

  Core::Conditions::ConditionDefinition art_in_bc("DESIGN NODE 1D ARTERY PRESCRIBED CONDITIONS",
      "ArtPrescribedCond", "Artery prescribed boundary condition",
      Core::Conditions::ArtPrescribedCond, true, Core::Conditions::geometry_type_point);
  art_in_bc.add_component(deprecated_selection<std::string>("boundarycond",
      {"flow", "pressure", "velocity", "area", "characteristicWave"},
      {.description = "boundarycond", .default_value = "flow"}));
  art_in_bc.add_component(deprecated_selection<std::string>(
      "type", {"forced", "absorbing"}, {.description = "type", .default_value = "forced"}));

  art_in_bc.add_component(
      parameter<std::vector<double>>("VAL", {.description = "values", .size = 2}));
  art_in_bc.add_component(
      parameter<std::vector<std::optional<int>>>("curve", {.description = "curve ids", .size = 2}));

  condlist.push_back(art_in_bc);

  /*--------------------------------------------------------------------*/
  // 1D artery reflective BC
  Core::Conditions::ConditionDefinition art_rf_bc("DESIGN NODE 1D ARTERY REFLECTIVE CONDITIONS",
      "ArtRfCond", "Artery reflection condition", Core::Conditions::ArtRfCond, true,
      Core::Conditions::geometry_type_point);

  art_rf_bc.add_component(
      parameter<std::vector<double>>("VAL", {.description = "value", .size = 1}));
  art_rf_bc.add_component(
      parameter<std::vector<std::optional<int>>>("curve", {.description = "curve", .size = 1}));
  condlist.push_back(art_rf_bc);


  /*--------------------------------------------------------------------*/
  // 1D artery in/out condition

  Core::Conditions::ConditionDefinition art_in_outlet_bc(
      "DESIGN NODE 1D ARTERY IN_OUTLET CONDITIONS", "ArtInOutCond",
      "Artery terminal in_outlet condition", Core::Conditions::ArtInOutletCond, true,
      Core::Conditions::geometry_type_point);

  art_in_outlet_bc.add_component(deprecated_selection<std::string>("terminaltype",
      {"inlet", "outlet"}, {.description = "terminaltype", .default_value = "inlet"}));

  condlist.push_back(art_in_outlet_bc);

  /*--------------------------------------------------------------------*/
  // 1D artery-to-porofluid coupling BC

  Core::Conditions::ConditionDefinition artcoup(
      "DESIGN NODE 1D ARTERY TO POROFLUID COUPLING CONDITIONS", "ArtPorofluidCouplConNodebased",
      "Artery coupling with porofluid", Core::Conditions::ArtPorofluidCouplingCondNodebased, true,
      Core::Conditions::geometry_type_point);

  artcoup.add_component(parameter<int>("COUPID"));

  condlist.push_back(artcoup);

  /*--------------------------------------------------------------------*/
  // 1D artery-to-scatra coupling BC

  Core::Conditions::ConditionDefinition artscatracoup(
      "DESIGN NODE 1D ARTERY TO SCATRA COUPLING CONDITIONS", "ArtScatraCouplConNodebased",
      "Artery coupling with porofluid", Core::Conditions::ArtScatraCouplingCondNodebased, true,
      Core::Conditions::geometry_type_point);

  artscatracoup.add_component(parameter<int>("COUPID"));

  condlist.push_back(artscatracoup);

  /*--------------------------------------------------------------------*/
  // 1D artery-to-porofluid coupling BC Node-To-Point
  Core::Conditions::ConditionDefinition artcoup_ntp(
      "DESIGN 1D ARTERY/AIRWAY TO POROFLUID NONCONF COUPLING CONDITIONS",
      "ArtPorofluidCouplConNodeToPoint", "Artery coupling with porofluid nonconf",
      Core::Conditions::ArtPorofluidCouplingCondNodeToPoint, true,
      Core::Conditions::geometry_type_point);

  artcoup_ntp.add_component(deprecated_selection<std::string>("COUPLING_TYPE", {"ARTERY", "AIRWAY"},
      {.description = "coupling type", .default_value = "ARTERY"}));
  artcoup_ntp.add_component(parameter<int>("NUMDOF"));
  artcoup_ntp.add_component(parameter<std::vector<int>>(
      "COUPLEDDOF_REDUCED", {.description = "coupling dofs of reduced airways or arteries",
                                .size = from_parameter<int>("NUMDOF")}));
  artcoup_ntp.add_component(parameter<std::vector<int>>("COUPLEDDOF_PORO",
      {.description = "coupling dofs in porous domain", .size = from_parameter<int>("NUMDOF")}));
  artcoup_ntp.add_component(parameter<double>("PENALTY"));

  condlist.push_back(artcoup_ntp);

  /*--------------------------------------------------------------------*/
  // 1D artery-to-scatra coupling BC Node-To-Point
  Core::Conditions::ConditionDefinition artscatracoup_ntp(
      "DESIGN 1D ARTERY/AIRWAY TO SCATRA NONCONF COUPLING CONDITIONS",
      "ArtScatraCouplConNodeToPoint", "Artery coupling with scatra nonconf",
      Core::Conditions::ArtScatraCouplingCondNodeToPoint, true,
      Core::Conditions::geometry_type_point);

  artscatracoup_ntp.add_component(deprecated_selection<std::string>("COUPLING_TYPE",
      {"ARTERY", "AIRWAY"}, {.description = "coupling type", .default_value = "ARTERY"}));
  artscatracoup_ntp.add_component(parameter<int>("NUMDOF"));
  artscatracoup_ntp.add_component(parameter<std::vector<int>>(
      "COUPLEDDOF_REDUCED", {.description = "coupling dofs of reduced airways or arteries",
                                .size = from_parameter<int>("NUMDOF")}));
  artscatracoup_ntp.add_component(parameter<std::vector<int>>("COUPLEDDOF_PORO",
      {.description = "coupling dofs in porous domain", .size = from_parameter<int>("NUMDOF")}));
  artscatracoup_ntp.add_component(parameter<double>("PENALTY"));

  condlist.push_back(artscatracoup_ntp);

  /*--------------------------------------------------------------------*/
  // 3-D/reduced-D coupling boundary condition
  Core::Conditions::ConditionDefinition art_red_to_3d_bc(
      "DESIGN NODE REDUCED D To 3D FLOW COUPLING CONDITIONS", "Art_redD_3D_CouplingCond",
      "Artery reduced D 3D coupling condition", Core::Conditions::ArtRedTo3DCouplingCond, true,
      Core::Conditions::geometry_type_point);

  art_red_to_3d_bc.add_component(parameter<int>("ConditionID"));
  art_red_to_3d_bc.add_component(deprecated_selection<std::string>("CouplingType",
      {"forced", "absorbing"}, {.description = "coupling type", .default_value = "forced"}));
  art_red_to_3d_bc.add_component(deprecated_selection<std::string>("ReturnedVariable",
      {"pressure", "flow"}, {.description = "returned variable", .default_value = "pressure"}));
  art_red_to_3d_bc.add_component(parameter<double>("Tolerance"));
  art_red_to_3d_bc.add_component(parameter<int>("MaximumIterations"));

  condlist.push_back(art_red_to_3d_bc);

  /*--------------------------------------------------------------------*/
  // 3-D/reduced-D coupling boundary condition
  Core::Conditions::ConditionDefinition art_3d_to_red_bc(
      "DESIGN SURF 3D To REDUCED D FLOW COUPLING CONDITIONS", "Art_3D_redD_CouplingCond",
      "Artery 3D reduced D coupling condition", Core::Conditions::Art3DToRedCouplingCond, true,
      Core::Conditions::geometry_type_surface);

  art_3d_to_red_bc.add_component(parameter<int>("ConditionID"));
  art_3d_to_red_bc.add_component(deprecated_selection<std::string>("ReturnedVariable",
      {"pressure", "flow"}, {.description = "returned variable", .default_value = "flow"}));
  art_3d_to_red_bc.add_component(parameter<double>("Tolerance"));
  art_3d_to_red_bc.add_component(parameter<int>("MaximumIterations"));

  condlist.push_back(art_3d_to_red_bc);
}

FOUR_C_NAMESPACE_CLOSE
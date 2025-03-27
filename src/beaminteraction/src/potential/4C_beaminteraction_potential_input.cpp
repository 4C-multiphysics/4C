// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_potential_input.hpp"

#include "4C_beamcontact_input.hpp"
#include "4C_fem_condition_definition.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN



void BeamPotential::set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list)
{
  using Teuchos::tuple;
  using namespace Core::IO::InputSpecBuilders;

  /* parameters for potential-based beam interaction */
  Core::Utils::SectionSpecs beampotential{"BEAM POTENTIAL"};

  // TODO change to vector and remove default value
  beampotential.specs.emplace_back(parameter<std::string>(
      "POT_LAW_EXPONENT", {.description = "negative(!) exponent(s)  $m_i$ of potential law  "
                                          "$\\Phi(r) = \\sum_i (k_i * r^{-m_i}).$",
                              .default_value = "1.0"}));

  // TODO change to vector and remove default value
  beampotential.specs.emplace_back(parameter<std::string>("POT_LAW_PREFACTOR",
      {.description = "prefactor(s) $k_i$ of potential law $\\Phi(r) = \\sum_i (k_i * r^{-m_i})$.",
          .default_value = "0.0"}));

  // TODO remove default value
  beampotential.specs.emplace_back(parameter<BeamPotential::Type>("TYPE",
      {.description = "Type of potential interaction: surface (default) or volume potential",
          .default_value = BeamPotential::Type::surface}));

  // TODO remove default value
  beampotential.specs.emplace_back(parameter<BeamPotential::Strategy>("STRATEGY",
      {.description = "strategy to evaluate interaction potential: double/single length specific, "
                      "small/large separation approximation, ...",
          .default_value = BeamPotential::Strategy::double_length_specific_large_separations}));

  beampotential.specs.emplace_back(parameter<std::optional<double>>("CUTOFF_RADIUS",
      {.description =
              "Neglect all potential contributions at separation largerthan this cutoff radius"}));

  // TODO subgroup regularization
  beampotential.specs.emplace_back(parameter<BeamPotential::RegularizationType>(
      "REGULARIZATION_TYPE", {.description = "Type of regularization applied to the force law",
                                 .default_value = BeamPotential::RegularizationType::none}));

  beampotential.specs.emplace_back(parameter<double>("REGULARIZATION_SEPARATION",
      {.description = "Use regularization of force law at separations smaller than this separation",
          .default_value = -1.0}));

  beampotential.specs.emplace_back(parameter<int>("N_INTEGRATION_SEGMENTS",
      {.description = "Number of integration segments used per beam element", .default_value = 1}));

  beampotential.specs.emplace_back(parameter<int>("N_GAUSS_POINTS",
      {.description = "Number of Gauss points used per integration segment", .default_value = 10}));

  beampotential.specs.emplace_back(parameter<bool>("AUTOMATIC_DIFFERENTIATION",
      {.description = "apply automatic differentiation via FAD?", .default_value = false}));

  beampotential.specs.emplace_back(parameter<MasterSlaveChoice>(
      "CHOICE_MASTER_SLAVE", {.description = "According to which rule shall the role of master and "
                                             "slave be assigned to beam elements?",
                                 .default_value = MasterSlaveChoice::smaller_eleGID_is_slave}));

  beampotential.specs.emplace_back(parameter<std::optional<double>>("POTENTIAL_REDUCTION_LENGTH",
      {.description =
              "Within this length of the master beam end point the potential is smoothly "
              "reduced to one half to account for infinitely long master beam surrogates."}));

  beampotential.move_into_collection(list);

  /*------------------------------------------------------------------------*/
  /* parameters for visualization of potential-based beam interactions via output at runtime */

  Core::Utils::SectionSpecs beampotential_output_sublist{beampotential, "RUNTIME VTK OUTPUT"};


  // whether to write visualization output for beam contact
  beampotential_output_sublist.specs.emplace_back(parameter<bool>("VTK_OUTPUT_BEAM_POTENTIAL",
      {.description = "write visualization output for potential-based beam interactions",
          .default_value = false}));

  // output interval regarding steps: write output every INTERVAL_STEPS steps
  beampotential_output_sublist.specs.emplace_back(parameter<int>("INTERVAL_STEPS",
      {.description = "write output at runtime every INTERVAL_STEPS steps", .default_value = 1}));

  // whether to write output in every iteration of the nonlinear solver
  beampotential_output_sublist.specs.emplace_back(parameter<bool>(
      "EVERY_ITERATION", {.description = "write output in every iteration of the nonlinear solver",
                             .default_value = false}));

  // whether to write visualization output for forces
  beampotential_output_sublist.specs.emplace_back(parameter<bool>(
      "FORCES", {.description = "write visualization output for forces", .default_value = false}));

  // whether to write visualization output for moments
  beampotential_output_sublist.specs.emplace_back(parameter<bool>("MOMENTS",
      {.description = "write visualization output for moments", .default_value = false}));

  // whether to write visualization output for forces/moments separately for each element pair
  beampotential_output_sublist.specs.emplace_back(
      parameter<bool>("WRITE_FORCE_MOMENT_PER_ELEMENTPAIR",
          {.description =
                  "write visualization output for forces/moments separately for each element pair",
              .default_value = false}));

  // whether to write out the UIDs (uid_0_beam_1_gid, uid_1_beam_2_gid, uid_2_gp_id)
  beampotential_output_sublist.specs.emplace_back(parameter<bool>(
      "WRITE_UIDS", {.description = "write out the unique ID's for each visualization point,i.e., "
                                    "master and slave beam element global ID (uid_0_beam_1_gid, "
                                    "uid_1_beam_2_gid) and local Gauss point ID (uid_2_gp_id)",
                        .default_value = false}));

  beampotential_output_sublist.move_into_collection(list);
}

void BeamPotential::set_valid_conditions(
    std::vector<Core::Conditions::ConditionDefinition>& condlist)
{
  using namespace Core::IO::InputSpecBuilders;

  // beam potential interaction: atom/charge density per unit length on LINE

  Core::Conditions::ConditionDefinition beam_potential_line_charge(
      "DESIGN LINE BEAM POTENTIAL CHARGE CONDITIONS", "BeamPotentialLineCharge",
      "Beam_Potential_Line_Charge_Density", Core::Conditions::BeamPotential_LineChargeDensity,
      false, Core::Conditions::geometry_type_line);

  beam_potential_line_charge.add_component(parameter<int>("POTLAW"));
  beam_potential_line_charge.add_component(parameter<double>("VAL"));
  beam_potential_line_charge.add_component(
      parameter<std::optional<int>>("FUNCT", {.description = ""}));

  condlist.push_back(beam_potential_line_charge);

  Core::Conditions::ConditionDefinition rigidsphere_potential_charge(
      "DESIGN POINT RIGIDSPHERE POTENTIAL CHARGE CONDITIONS", "RigidspherePotentialPointCharge",
      "Rigidsphere_Potential_Point_Charge", Core::Conditions::RigidspherePotential_PointCharge,
      false, Core::Conditions::geometry_type_point);

  rigidsphere_potential_charge.add_component(parameter<int>("POTLAW"));
  rigidsphere_potential_charge.add_component(parameter<double>("VAL"));
  rigidsphere_potential_charge.add_component(
      parameter<std::optional<int>>("FUNCT", {.description = ""}));

  condlist.push_back(rigidsphere_potential_charge);
}

FOUR_C_NAMESPACE_CLOSE

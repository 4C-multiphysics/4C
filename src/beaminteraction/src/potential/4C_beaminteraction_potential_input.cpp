// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_potential_input.hpp"

#include "4C_beamcontact_input.hpp"
#include "4C_fem_condition_definition.hpp"
#include "4C_global_data.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_io_input_spec_validators.hpp"
#include "4C_utils_exceptions.hpp"


FOUR_C_NAMESPACE_OPEN



void BeamInteraction::Potential::set_valid_parameters(
    std::map<std::string, Core::IO::InputSpec>& list)
{
  using namespace Core::IO::InputSpecBuilders;
  using namespace Core::IO::InputSpecBuilders::Validators;

  list["beam_potential"] = group<BeamPotentialParameters>("beam_potential",
      {
          parameter<std::vector<double>>("potential_law_exponents",
              {.description = "Negative(!) exponent(s)  $m_i$ of potential law $\\Phi(r) = \\sum_i "
                              "(k_i * r^{-m_i}).$",
                  .validator = all_elements(positive<double>()),
                  .store = in_struct(&BeamPotentialParameters::potential_law_exponents)}),

          parameter<std::vector<double>>("potential_law_prefactors",
              {.description = "Prefactor(s) $k_i$ of potential law $\\Phi(r) = \\sum_i (k_i * "
                              "r^{-m_i})$.",
                  .store = in_struct(&BeamPotentialParameters::potential_law_prefactors)}),

          parameter<Potential::Type>("type",
              {.description = "Type of potential interaction, i.e., surface or volume interaction.",
                  .store = in_struct(&BeamPotentialParameters::type)}),

          parameter<Potential::Strategy>("strategy",
              {.description = "Strategy to evaluate interaction potential: double/single length "
                              "small/large separation approximation, ...",
                  .store = in_struct(&BeamPotentialParameters::strategy)}),

          parameter<std::optional<double>>(
              "cutoff_radius", {.description = "Neglect all potential contributions at separation "
                                               "larger than this cutoff radius",
                                   .validator = null_or(positive<double>()),
                                   .store = in_struct(&BeamPotentialParameters::cutoff_radius)}),

          parameter<int>("n_integration_segments",
              {.description = "Number of integration segments used per beam element",
                  .default_value = 1,
                  .validator = positive<int>(),
                  .store = in_struct(&BeamPotentialParameters::n_integration_segments)}),

          parameter<int>("n_gauss_points",
              {.description = "Number of Gauss points used per integration segment",
                  .default_value = 10,
                  .validator = positive<int>(),
                  .store = in_struct(&BeamPotentialParameters::n_gauss_points)}),

          parameter<bool>("automatic_differentiation",
              {.description = "Option to apply automatic differentiation via FAD",
                  .default_value = false,
                  .store = in_struct(&BeamPotentialParameters::automatic_differentiation)}),

          parameter<MasterSlaveChoice>("choice_master_slave",
              {.description =
                      "Rule to which the role of master and slave is assigned to beam elements",
                  .default_value = MasterSlaveChoice::smaller_eleGID_is_slave,
                  .store = in_struct(&BeamPotentialParameters::choice_master_slave)}),

          parameter<std::optional<double>>("potential_reduction_length",
              {.description =
                      "Length in which the potential is reduced to 0 at master beam end points "
                      "(only applicable for 'single_length_specific_small_separations' strategy).",
                  .validator = null_or(positive<double>()),
                  .store = in_struct(&BeamPotentialParameters::potential_reduction_length)}),

          group<BeamPotentialRegularizationParameters>("regularization",
              {
                  parameter<Potential::RegularizationType>("type",
                      {.description = "Type of regularization applied to the force law",
                          .default_value = Potential::RegularizationType::none,
                          .store = in_struct(&BeamPotentialRegularizationParameters::type)}),

                  parameter<double>("separation",
                      {.description = "Activate regularization of force law at separations smaller "
                                      "than this separation",
                          .default_value = 0.0,
                          .store = in_struct(&BeamPotentialRegularizationParameters::separation)}),
              },
              {.description = "Regularization of the potential at small separations",
                  .required = false,
                  .store = in_struct(&BeamPotentialParameters::regularization)}),

          // Visualization output parameters
          group<BeamPotentialVisualizationParameters>("runtime_output",
              {
                  parameter<std::optional<int>>("interval_steps",
                      {.description = "Frequency at which the output is written.",
                          .validator = null_or(positive<int>()),
                          .store =
                              in_struct(&BeamPotentialVisualizationParameters::output_interval)}),
                  parameter<bool>("every_iteration",
                      {.description = "Write output in every iteration of the nonlinear solver",
                          .default_value = false,
                          .store = in_struct(
                              &BeamPotentialVisualizationParameters::write_every_iteration)}),
                  parameter<bool>("force",
                      {.description = "Write visualization output for forces",
                          .default_value = false,
                          .store = in_struct(&BeamPotentialVisualizationParameters::write_forces)}),
                  parameter<bool>("moment",
                      {.description = "Write visualization output for moments",
                          .default_value = false,
                          .store =
                              in_struct(&BeamPotentialVisualizationParameters::write_moments)}),
                  parameter<bool>("write_force_moment_per_elementpair",
                      {.description = "Write visualization output for forces/moments "
                                      "separately for each element pair",
                          .default_value = false,
                          .store = in_struct(&BeamPotentialVisualizationParameters::
                                  write_forces_moments_per_pair)}),
                  parameter<bool>("write_uids",
                      {.description =
                              "Write out the unique ID's for each visualization point,i.e., "
                              "master and slave beam element global ID (uid_0_beam_1_gid, "
                              "uid_1_beam_2_gid) and local Gauss point ID (uid_2_gp_id)",
                          .default_value = false,
                          .store = in_struct(&BeamPotentialVisualizationParameters::write_uids)}),
              },
              {.description = "Runtime output parameters for beam potential-based "
                              "interactions",
                  .required = false,
                  .store = in_struct(&BeamPotentialParameters::runtime_output_params)}),
      },
      {.description = "Parameters for beam interactions based on potentials. Beam-to-beam and "
                      "beam-to-sphere interactions are available.",
          .required = false});
};

void BeamInteraction::Potential::set_valid_conditions(
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

void BeamInteraction::Potential::initialize_validate_beam_potential_params(
    BeamInteraction::Potential::BeamPotentialParameters& params, double restart_time)
{
  params = Global::Problem::instance()
               ->parameters()
               .get<BeamInteraction::Potential::BeamPotentialParameters>("beam_potential");


  FOUR_C_ASSERT_ALWAYS(
      params.potential_law_prefactors.size() == params.potential_law_exponents.size(),
      "Number of potential law prefactors must match number of potential law exponents. "
      "Check your input file!");

  // Check correct combination of potential type, strategy and regularization
  if (params.type == BeamInteraction::Potential::Type::surface and
      params.strategy !=
          BeamInteraction::Potential::Strategy::double_length_specific_large_separations)
  {
    FOUR_C_THROW("Surface interaction is not implemented for this strategy yet!");
  }

  if ((params.regularization.type != BeamInteraction::Potential::RegularizationType::none and
          params.strategy ==
              BeamInteraction::Potential::Strategy::double_length_specific_large_separations) or
      (params.regularization.type == BeamInteraction::Potential::RegularizationType::constant and
          params.strategy ==
              BeamInteraction::Potential::Strategy::single_length_specific_small_separations))
  {
    FOUR_C_THROW(
        "This kind of regularization of the force law is not implemented for this strategy "
        "yet!");
  }

  // create and initialize parameter container object for runtime output
  if (params.runtime_output_params.output_interval.has_value())
  {
    params.runtime_output_params.visualization_parameters =
        Core::IO::visualization_parameters_factory(
            Global::Problem::instance()->io_params().sublist("RUNTIME VTK OUTPUT"),
            *Global::Problem::instance()->output_control_file(), restart_time);

    // overwrite global option to write visualization output of every iteration
    params.runtime_output_params.visualization_parameters.every_iteration_ =
        params.runtime_output_params.write_every_iteration;
  }
};

FOUR_C_NAMESPACE_CLOSE
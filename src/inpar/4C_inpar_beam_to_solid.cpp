// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_beam_to_solid.hpp"

#include "4C_fem_condition_definition.hpp"
#include "4C_geometry_pair_input.hpp"
#include "4C_inpar_beaminteraction.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN


/**
 *
 */
void Inpar::BeamToSolid::beam_to_solid_interaction_get_string(
    const Inpar::BeamInteraction::BeamInteractionConditions& interaction,
    std::array<std::string, 2>& condition_names)
{
  if (interaction ==
      Inpar::BeamInteraction::BeamInteractionConditions::beam_to_solid_volume_meshtying)
  {
    condition_names[0] = "BeamToSolidVolumeMeshtyingLine";
    condition_names[1] = "BeamToSolidVolumeMeshtyingVolume";
  }
  else if (interaction ==
           Inpar::BeamInteraction::BeamInteractionConditions::beam_to_solid_surface_meshtying)
  {
    condition_names[0] = "BeamToSolidSurfaceMeshtyingLine";
    condition_names[1] = "BeamToSolidSurfaceMeshtyingSurface";
  }
  else if (interaction ==
           Inpar::BeamInteraction::BeamInteractionConditions::beam_to_solid_surface_contact)
  {
    condition_names[0] = "BeamToSolidSurfaceContactLine";
    condition_names[1] = "BeamToSolidSurfaceContactSurface";
  }
  else if (interaction ==
           Inpar::BeamInteraction::BeamInteractionConditions::beam_to_solid_edge_contact)
  {
    condition_names[0] = "BeamToSolidEdgeContactBeam";
    condition_names[1] = "BeamToSolidEdgeContactEdge";
  }
  else
    FOUR_C_THROW("Got unexpected beam-to-solid interaction type.");
}

/**
 *
 */
std::vector<Core::IO::InputSpec> Inpar::BeamToSolid::valid_parameters()
{
  using namespace Core::IO::InputSpecBuilders;

  // Beam to solid volume mesh tying parameters.
  std::vector<Core::IO::InputSpec> beam_to_solid_volume_mestying = {
      parameter<BeamToSolidContactDiscretization>(
          "CONTACT_DISCRETIZATION", {.description = "Type of employed contact discretization",
                                        .default_value = BeamToSolidContactDiscretization::none}),

      parameter<BeamToSolidConstraintEnforcement>(
          "CONSTRAINT_STRATEGY", {.description = "Type of employed constraint enforcement strategy",
                                     .default_value = BeamToSolidConstraintEnforcement::none}),

      parameter<BeamToSolidMortarShapefunctions>("MORTAR_SHAPE_FUNCTION",
          {.description = "Shape function for the mortar Lagrange-multipliers",
              .default_value = BeamToSolidMortarShapefunctions::none}),

      parameter<BeamToSolidLagrangeFormulation>("LAGRANGE_FORMULATION",
          {.description = "Type of employed Lagrange Formulation while using Lagrange Multipliers "
                          "for constraint enforcement ",
              .default_value = BeamToSolidLagrangeFormulation::none}),


      parameter<int>("MORTAR_FOURIER_MODES",
          {.description = "Number of fourier modes to be used for cross-section mortar coupling",
              .default_value = -1}),

      parameter<double>("PENALTY_PARAMETER",
          {.description = "Penalty parameter for beam-to-solid volume meshtying",
              .default_value = 0.0}),


      // This option only has an effect during a restart simulation.
      // - No:  (default) The coupling is treated the same way as during a non restart
      // simulation,
      //        i.e. the initial configurations (zero displacement) of the beams and solids are
      //        coupled.
      // - Yes: The beam and solid states at the restart configuration are coupled. This allows
      // to
      //        pre-deform the structures and then couple them.
      parameter<bool>("COUPLE_RESTART_STATE",
          {.description = "Enable / disable the coupling of the restart configuration.",
              .default_value = false}),

      parameter<BeamToSolidRotationCoupling>(
          "ROTATION_COUPLING", {.description = "Type of rotational coupling",
                                   .default_value = BeamToSolidRotationCoupling::none}),


      parameter<BeamToSolidMortarShapefunctions>("ROTATION_COUPLING_MORTAR_SHAPE_FUNCTION",
          {.description = "Shape function for the mortar Lagrange-multipliers",
              .default_value = BeamToSolidMortarShapefunctions::none}),


      parameter<double>("ROTATION_COUPLING_PENALTY_PARAMETER",
          {.description = "Penalty parameter for rotational coupling in beam-to-solid volume "
                          "mesh tying",
              .default_value = 0.0}),
  };
  // Add the geometry pair input parameters.
  GeometryPair::valid_parameters_line_to_3d(beam_to_solid_volume_mestying);

  std::vector<Core::IO::InputSpec> specs;
  specs.push_back(group("BEAM INTERACTION/BEAM TO SOLID VOLUME MESHTYING",
      beam_to_solid_volume_mestying, {.required = false}));


  // Beam to solid volume mesh tying output parameters.
  specs.push_back(group("BEAM INTERACTION/BEAM TO SOLID VOLUME MESHTYING/RUNTIME VTK OUTPUT",
      {
          // Whether to write visualization output at all for btsvmt.
          parameter<bool>("WRITE_OUTPUT",
              {.description = "Enable / disable beam-to-solid volume mesh tying output.",
                  .default_value = false}),

          parameter<bool>("NODAL_FORCES",
              {.description = "Enable / disable output of the resulting nodal forces due "
                              "to beam to solid interaction.",
                  .default_value = false}),

          parameter<bool>("MORTAR_LAMBDA_DISCRET",
              {.description =
                      "Enable / disable output of the discrete Lagrange multipliers at the node "
                      "of the Lagrange multiplier shape functions.",
                  .default_value = false}),

          parameter<bool>("MORTAR_LAMBDA_CONTINUOUS",
              {.description = "Enable / disable output of the continuous "
                              "Lagrange multipliers function along the beam.",
                  .default_value = false}),

          parameter<int>("MORTAR_LAMBDA_CONTINUOUS_SEGMENTS",
              {.description = "Number of segments for continuous mortar output",
                  .default_value = 5}),

          parameter<int>("MORTAR_LAMBDA_CONTINUOUS_SEGMENTS_CIRCUMFERENCE",
              {.description = "Number of segments for continuous mortar output along the beam "
                              "cross-section circumference",
                  .default_value = 8}),

          parameter<bool>(
              "SEGMENTATION", {.description = "Enable / disable output of segmentation points.",
                                  .default_value = false}),

          parameter<bool>("INTEGRATION_POINTS",
              {.description =
                      "Enable / disable output of used integration points. If the contact method "
                      "has 'forces' at the integration point, they will also be output.",
                  .default_value = false}),

          parameter<bool>(
              "UNIQUE_IDS", {.description = "Enable / disable output of unique IDs (mainly "
                                            "for testing of created VTK files).",
                                .default_value = false}),
      },
      {.required = false}));


  // Beam to solid surface mesh tying parameters.
  std::vector<Core::IO::InputSpec> beam_to_solid_surface_meshtying = {
      parameter<BeamToSolidContactDiscretization>(
          "CONTACT_DISCRETIZATION", {.description = "Type of employed contact discretization",
                                        .default_value = BeamToSolidContactDiscretization::none}),

      parameter<BeamToSolidConstraintEnforcement>(
          "CONSTRAINT_STRATEGY", {.description = "Type of employed constraint enforcement strategy",
                                     .default_value = BeamToSolidConstraintEnforcement::none}),

      parameter<BeamToSolidSurfaceCoupling>(
          "COUPLING_TYPE", {.description = "How the coupling constraints are formulated/",
                               .default_value = BeamToSolidSurfaceCoupling::none}),


      parameter<BeamToSolidMortarShapefunctions>("MORTAR_SHAPE_FUNCTION",
          {.description = "Shape function for the mortar Lagrange-multipliers",
              .default_value = BeamToSolidMortarShapefunctions::none}),

      parameter<double>("PENALTY_PARAMETER",
          {.description = "Penalty parameter for beam-to-solid surface meshtying",
              .default_value = 0.0}),

      // Parameters for rotational coupling.
      parameter<bool>("ROTATIONAL_COUPLING",
          {.description = "Enable / disable rotational coupling", .default_value = false}),

      parameter<double>("ROTATIONAL_COUPLING_PENALTY_PARAMETER",
          {.description = "Penalty parameter for beam-to-solid surface rotational meshtying",
              .default_value = 0.0}),

      parameter<BeamToSolidSurfaceRotationCoupling>("ROTATIONAL_COUPLING_SURFACE_TRIAD",
          {.description = "Construction method for surface triad",
              .default_value = BeamToSolidSurfaceRotationCoupling::none}),
  };
  // Add the geometry pair input parameters.
  GeometryPair::valid_parameters_line_to_3d(beam_to_solid_surface_meshtying);

  // Add the surface options.
  GeometryPair::valid_parameters_line_to_surface(beam_to_solid_surface_meshtying);
  specs.push_back(group("BEAM INTERACTION/BEAM TO SOLID SURFACE MESHTYING",
      beam_to_solid_surface_meshtying, {.required = false}));


  // Beam to solid surface contact parameters.
  std::vector<Core::IO::InputSpec> beam_to_solid_surface_contact = {
      parameter<BeamToSolidContactDiscretization>(
          "CONTACT_DISCRETIZATION", {.description = "Type of employed contact discretization",
                                        .default_value = BeamToSolidContactDiscretization::none}),

      parameter<BeamToSolidConstraintEnforcement>(
          "CONSTRAINT_STRATEGY", {.description = "Type of employed constraint enforcement strategy",
                                     .default_value = BeamToSolidConstraintEnforcement::none}),

      parameter<double>("PENALTY_PARAMETER",
          {.description = "Penalty parameter for beam-to-solid surface contact",
              .default_value = 0.0}),

      parameter<BeamToSolidSurfaceContact>(
          "CONTACT_TYPE", {.description = "How the contact constraints are formulated",
                              .default_value = BeamToSolidSurfaceContact::none}),

      parameter<BeamToSolidSurfaceContactPenaltyLaw>(
          "PENALTY_LAW", {.description = "Type of penalty law",
                             .default_value = BeamToSolidSurfaceContactPenaltyLaw::none}),

      parameter<double>("PENALTY_PARAMETER_G0",
          {.description =
                  "First penalty regularization parameter G0 >=0: For gap<G0 contact is active",
              .default_value = 0.0}),


      parameter<BeamToSolidSurfaceContactMortarDefinedIn>("MORTAR_CONTACT_DEFINED_IN",
          {.description = "Configuration where the mortar contact is defined",
              .default_value = BeamToSolidSurfaceContactMortarDefinedIn::none}),

      // Define the mortar shape functions for contact

      parameter<BeamToSolidMortarShapefunctions>("MORTAR_SHAPE_FUNCTION",
          {.description = "Shape function for the mortar Lagrange-multipliers",
              .default_value = BeamToSolidMortarShapefunctions::none}),
  };
  // Add the geometry pair input parameters.
  GeometryPair::valid_parameters_line_to_3d(beam_to_solid_surface_contact);

  // Add the surface options.
  GeometryPair::valid_parameters_line_to_surface(beam_to_solid_surface_contact);
  specs.push_back(group("BEAM INTERACTION/BEAM TO SOLID SURFACE CONTACT",
      beam_to_solid_surface_contact, {.required = false}));


  // Beam to solid surface output parameters.
  specs.push_back(group("BEAM INTERACTION/BEAM TO SOLID SURFACE/RUNTIME VTK OUTPUT",
      {
          // Whether to write visualization output at all.
          parameter<bool>("WRITE_OUTPUT",
              {.description = "Enable / disable beam-to-solid volume mesh tying output.",
                  .default_value = false}),

          parameter<bool>("NODAL_FORCES",
              {.description = "Enable / disable output of the resulting nodal forces due "
                              "to beam to solid interaction.",
                  .default_value = false}),

          parameter<bool>("AVERAGED_NORMALS",
              {.description = "Enable / disable output of averaged nodal normals on the surface.",
                  .default_value = false}),

          parameter<bool>("MORTAR_LAMBDA_DISCRET",
              {.description =
                      "Enable / disable output of the discrete Lagrange multipliers at the node "
                      "of the Lagrange multiplier shape functions.",
                  .default_value = false}),

          parameter<bool>("MORTAR_LAMBDA_CONTINUOUS",
              {.description = "Enable / disable output of the continuous "
                              "Lagrange multipliers function along the beam.",
                  .default_value = false}),

          parameter<int>("MORTAR_LAMBDA_CONTINUOUS_SEGMENTS",
              {.description = "Number of segments for continuous mortar output",
                  .default_value = 5}),

          parameter<bool>(
              "SEGMENTATION", {.description = "Enable / disable output of segmentation points.",
                                  .default_value = false}),

          parameter<bool>("INTEGRATION_POINTS",
              {.description =
                      "Enable / disable output of used integration points. If the contact method "
                      "has 'forces' at the integration point, they will also be output.",
                  .default_value = false}),

          parameter<bool>(
              "UNIQUE_IDS", {.description = "Enable / disable output of unique IDs (mainly "
                                            "for testing of created VTK files).",
                                .default_value = false}),
      },
      {.required = false}));
  return specs;
}

/**
 *
 */
void Inpar::BeamToSolid::set_valid_conditions(
    std::vector<Core::Conditions::ConditionDefinition>& condlist)
{
  using namespace Core::IO::InputSpecBuilders;

  // Beam-to-volume mesh tying conditions.
  {
    std::array<std::string, 2> condition_names;
    beam_to_solid_interaction_get_string(
        Inpar::BeamInteraction::BeamInteractionConditions::beam_to_solid_volume_meshtying,
        condition_names);

    Core::Conditions::ConditionDefinition beam_to_solid_volume_meshtying_condition(
        "BEAM INTERACTION/BEAM TO SOLID VOLUME MESHTYING VOLUME", condition_names[1],
        "Beam-to-volume mesh tying conditions - volume definition",
        Core::Conditions::BeamToSolidVolumeMeshtyingVolume, true,
        Core::Conditions::geometry_type_volume);
    beam_to_solid_volume_meshtying_condition.add_component(parameter<int>("COUPLING_ID"));
    condlist.push_back(beam_to_solid_volume_meshtying_condition);

    beam_to_solid_volume_meshtying_condition = Core::Conditions::ConditionDefinition(
        "BEAM INTERACTION/BEAM TO SOLID VOLUME MESHTYING LINE", condition_names[0],
        "Beam-to-volume mesh tying conditions - line definition",
        Core::Conditions::BeamToSolidVolumeMeshtyingLine, true,
        Core::Conditions::geometry_type_line);
    beam_to_solid_volume_meshtying_condition.add_component(parameter<int>("COUPLING_ID"));
    condlist.push_back(beam_to_solid_volume_meshtying_condition);
  }

  // Beam-to-surface mesh tying conditions.
  {
    std::array<std::string, 2> condition_names;
    beam_to_solid_interaction_get_string(
        Inpar::BeamInteraction::BeamInteractionConditions::beam_to_solid_surface_meshtying,
        condition_names);

    Core::Conditions::ConditionDefinition beam_to_solid_surface_meshtying_condition(
        "BEAM INTERACTION/BEAM TO SOLID SURFACE MESHTYING SURFACE", condition_names[1],
        "Beam-to-surface mesh tying conditions - surface definition",
        Core::Conditions::BeamToSolidSurfaceMeshtyingSurface, true,
        Core::Conditions::geometry_type_surface);
    beam_to_solid_surface_meshtying_condition.add_component(parameter<int>("COUPLING_ID"));
    condlist.push_back(beam_to_solid_surface_meshtying_condition);

    beam_to_solid_surface_meshtying_condition = Core::Conditions::ConditionDefinition(
        "BEAM INTERACTION/BEAM TO SOLID SURFACE MESHTYING LINE", condition_names[0],
        "Beam-to-surface mesh tying conditions - line definition",
        Core::Conditions::BeamToSolidSurfaceMeshtyingLine, true,
        Core::Conditions::geometry_type_line);
    beam_to_solid_surface_meshtying_condition.add_component(parameter<int>("COUPLING_ID"));
    condlist.push_back(beam_to_solid_surface_meshtying_condition);
  }

  // Beam-to-surface contact conditions.
  {
    std::array<std::string, 2> condition_names;
    beam_to_solid_interaction_get_string(
        Inpar::BeamInteraction::BeamInteractionConditions::beam_to_solid_surface_contact,
        condition_names);

    Core::Conditions::ConditionDefinition beam_to_solid_surface_contact_condition(
        "BEAM INTERACTION/BEAM TO SOLID SURFACE CONTACT SURFACE", condition_names[1],
        "Beam-to-surface contact conditions - surface definition",
        Core::Conditions::BeamToSolidSurfaceContactSurface, true,
        Core::Conditions::geometry_type_surface);
    beam_to_solid_surface_contact_condition.add_component(parameter<int>("COUPLING_ID"));
    condlist.push_back(beam_to_solid_surface_contact_condition);

    beam_to_solid_surface_contact_condition =
        Core::Conditions::ConditionDefinition("BEAM INTERACTION/BEAM TO SOLID SURFACE CONTACT LINE",
            condition_names[0], "Beam-to-surface contact conditions - line definition",
            Core::Conditions::BeamToSolidSurfaceContactLine, true,
            Core::Conditions::geometry_type_line);
    beam_to_solid_surface_contact_condition.add_component(parameter<int>("COUPLING_ID"));
    condlist.push_back(beam_to_solid_surface_contact_condition);
  }
}

FOUR_C_NAMESPACE_CLOSE

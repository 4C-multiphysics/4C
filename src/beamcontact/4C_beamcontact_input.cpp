// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beamcontact_input.hpp"

#include "4C_fem_condition_definition.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN



std::vector<Core::IO::InputSpec> BeamContact::valid_parameters()
{
  using namespace Core::IO::InputSpecBuilders;

  std::vector<Core::IO::InputSpec> specs;
  specs.push_back(group("BEAM CONTACT",
      {

          deprecated_selection<BeamContact::Strategy>("BEAMS_STRATEGY",
              {
                  {"None", bstr_none},
                  {"none", bstr_none},
                  {"Penalty", bstr_penalty},
                  {"penalty", bstr_penalty},
              },
              {.description = "Type of employed solving strategy", .default_value = bstr_none}),

          deprecated_selection<BeamContact::Modelevaluator>("MODELEVALUATOR",
              {
                  {"Old", bstr_old},
                  {"old", bstr_old},
                  {"Standard", bstr_standard},
                  {"standard", bstr_standard},
              },
              {.description = "Type of model evaluator", .default_value = bstr_old}),

          parameter<bool>(
              "BEAMS_NEWGAP", {.description = "choose between original or enhanced gapfunction",
                                  .default_value = true}),

          parameter<bool>("BEAMS_SEGCON",
              {.description = "choose between beam contact with and without subsegment generation",
                  .default_value = false}),

          parameter<bool>("BEAMS_DEBUG",
              {.description = "This flag can be used for testing purposes. When it is "
                              "switched on, some sanity checks are not performed!",
                  .default_value = false}),

          parameter<bool>("BEAMS_INACTIVESTIFF",
              {.description = "Always apply contact stiffness in first Newton step "
                              "for pairs which have active in last time step",
                  .default_value = false}),

          parameter<bool>("BEAMS_BTSOL",
              {.description = "decide, if also the contact between beams and solids is possible",
                  .default_value = false}),

          parameter<bool>("BEAMS_ENDPOINTPENALTY",
              {.description =
                      "Additional consideration of endpoint-line and endpoint-endpoint contacts",
                  .default_value = false}),

          deprecated_selection<BeamContact::Smoothing>("BEAMS_SMOOTHING",
              {
                  {"None", bsm_none},
                  {"none", bsm_none},
                  {"Cpp", bsm_cpp},
                  {"cpp", bsm_cpp},
              },
              {.description = "Application of smoothed tangent field", .default_value = bsm_none}),

          parameter<bool>("BEAMS_DAMPING",
              {.description = "Application of a contact damping force", .default_value = false}),

          parameter<double>("BEAMS_BTBPENALTYPARAM",
              {.description = "Penalty parameter for beam-to-beam point contact",
                  .default_value = 0.0}),
          parameter<double>("BEAMS_BTBLINEPENALTYPARAM",
              {.description = "Penalty parameter per unit length for beam-to-beam line contact",
                  .default_value = -1.0}),
          parameter<double>("BEAMS_BTSPENALTYPARAM",
              {.description = "Penalty parameter for beam-to-solid contact", .default_value = 0.0}),
          parameter<double>(
              "BEAMS_DAMPINGPARAM", {.description = "Damping parameter for contact damping force",
                                        .default_value = -1000.0}),
          parameter<double>("BEAMS_DAMPREGPARAM1",
              {.description = "First (at gap1, with gap1>gap2) regularization "
                              "parameter for contact damping force",
                  .default_value = -1000.0}),
          parameter<double>("BEAMS_DAMPREGPARAM2",
              {.description = "Second (at gap2, with gap1>gap2) regularization "
                              "parameter for contact damping force",
                  .default_value = -1000.0}),
          parameter<double>("BEAMS_MAXDISISCALEFAC",
              {.description = "Scale factor in order to limit maximal iterative "
                              "displacement increment (resiudal displacement)",
                  .default_value = -1.0}),
          parameter<double>("BEAMS_MAXDELTADISSCALEFAC",
              {.description = "Scale factor in order to limit maximal displacement per time step",
                  .default_value = 1.0}),

          parameter<double>("BEAMS_PERPSHIFTANGLE1",
              {.description =
                      "Lower shift angle (in degrees) for penalty scaling of large-angle-contact",
                  .default_value = -1.0}),
          parameter<double>("BEAMS_PERPSHIFTANGLE2",
              {.description =
                      "Upper shift angle (in degrees) for penalty scaling of large-angle-contact",
                  .default_value = -1.0}),
          parameter<double>("BEAMS_PARSHIFTANGLE1",
              {.description =
                      "Lower shift angle (in degrees) for penalty scaling of small-angle-contact",
                  .default_value = -1.0}),
          parameter<double>("BEAMS_PARSHIFTANGLE2",
              {.description =
                      "Upper shift angle (in degrees) for penalty scaling of small-angle-contact",
                  .default_value = -1.0}),
          parameter<double>("BEAMS_SEGANGLE",
              {.description = "Maximal angle deviation allowed for contact search segmentation",
                  .default_value = -1.0}),
          parameter<int>("BEAMS_NUMINTEGRATIONINTERVAL",
              {.description = "Number of integration intervals per element", .default_value = 1}),

          deprecated_selection<BeamContact::PenaltyLaw>("BEAMS_PENALTYLAW",
              {
                  {"LinPen", pl_lp},
                  {"QuadPen", pl_qp},
                  {"LinNegQuadPen", pl_lnqp},
                  {"LinPosQuadPen", pl_lpqp},
                  {"LinPosCubPen", pl_lpcp},
                  {"LinPosDoubleQuadPen", pl_lpdqp},
                  {"LinPosExpPen", pl_lpep},
              },
              {.description = "Applied Penalty Law", .default_value = pl_lp}),

          parameter<double>(
              "BEAMS_PENREGPARAM_G0", {.description = "First penalty regularization parameter G0 "
                                                      ">=0: For gap<G0 contact is active!",
                                          .default_value = -1.0}),
          parameter<double>("BEAMS_PENREGPARAM_F0",
              {.description =
                      "Second penalty regularization parameter F0 >=0: F0 represents the force at "
                      "the transition point between regularized and linear force law!",
                  .default_value = -1.0}),
          parameter<double>("BEAMS_PENREGPARAM_C0",
              {.description =
                      "Third penalty regularization parameter C0 >=0: C0 has different physical "
                      "meanings for the different penalty laws!",
                  .default_value = -1.0}),
          parameter<double>("BEAMS_GAPSHIFTPARAM",
              {.description = "Parameter to shift penalty law!", .default_value = 0.0}),
          parameter<double>("BEAMS_BASICSTIFFGAP",
              {.description = "For gaps > -BEAMS_BASICSTIFFGAP, only the basic part "
                              "of the contact linearization is applied!",
                  .default_value = -1.0}),

          // enable octree search and determine type of bounding box (aabb = axis aligned, cobb =
          // cylindrical oriented)
          deprecated_selection<BeamContact::OctreeType>("BEAMS_OCTREE",
              {
                  {"None", boct_none},
                  {"none", boct_none},
                  {"octree_axisaligned", boct_aabb},
                  {"octree_cylorient", boct_cobb},
                  {"octree_spherical", boct_spbb},
              },
              {.description = "octree and bounding box type for octree search routine",
                  .default_value = boct_none}),

          parameter<bool>("BEAMS_ADDITEXT",
              {.description = "Switch between No==multiplicative extrusion factor and "
                              "Yes==additive extrusion factor",
                  .default_value = true}),
          parameter<std::string>("BEAMS_EXTVAL",
              {.description =
                      "extrusion value(s) of the bounding box, Depending on BEAMS_ADDITIVEEXTFAC "
                      "is either  additive or multiplicative. Give one or two values.",
                  .default_value = "-1.0"}),

          parameter<int>("BEAMS_TREEDEPTH",
              {.description = "max. tree depth of the octree", .default_value = 6}),
          parameter<int>(
              "BEAMS_BOXESINOCT", {.description = "max number of bounding boxes in any leaf octant",
                                      .default_value = 8})},
      {.required = false}));
  /* parameters for visualization of beam contact via output at runtime */
  specs.push_back(group("BEAM CONTACT/RUNTIME VTK OUTPUT",
      {

          // whether to write visualization output for beam contact
          parameter<bool>("VTK_OUTPUT_BEAM_CONTACT",
              {.description = "write visualization output for beam contact",
                  .default_value = false}),

          // output interval regarding steps: write output every INTERVAL_STEPS steps
          parameter<int>("INTERVAL_STEPS",
              {.description = "write visualization output at runtime every INTERVAL_STEPS steps",
                  .default_value = -1}),

          // whether to write output in every iteration of the nonlinear solver
          parameter<bool>("EVERY_ITERATION",
              {.description = "write output in every iteration of the nonlinear solver",
                  .default_value = false}),

          // whether to write visualization output for contact forces
          parameter<bool>(
              "CONTACT_FORCES", {.description = "write visualization output for contact forces",
                                    .default_value = false}),

          // whether to write visualization output for gaps
          parameter<bool>(
              "GAPS", {.description = "write visualization output for gap, i.e. penetration",
                          .default_value = false}),
          // whether to write visualization output for the contact angle of beams
          parameter<bool>(
              "CONTACT_ANGLE", {.description = "write visualization output for contact angle",
                                   .default_value = false}),
          // whether to write visualization output for the type of contact formulation used
          parameter<bool>("CONTACT_TYPE",
              {.description =
                      "write visualization output to indicate which different penalty values are "
                      "active."
                      "Explanation: 0 corresponds to btb point contact;"
                      "1 corresponds to active btb line contact;"
                      "2 indicates active end point contact ",
                  .default_value = false}),
      },
      {.required = false}));
  return specs;
}

/**
 *
 */
void BeamContact::set_valid_conditions(std::vector<Core::Conditions::ConditionDefinition>& condlist)
{
  using namespace Core::IO::InputSpecBuilders;

  // Beam-to-beam conditions.
  {
    std::string condition_name = "BeamToBeamContact";

    Core::Conditions::ConditionDefinition beam_to_beam_contact_condition(
        "BEAM INTERACTION/BEAM TO BEAM CONTACT CONDITIONS", condition_name,
        "Beam-to-beam contact conditions", Core::Conditions::BeamToBeamContact, true,
        Core::Conditions::geometry_type_line);
    beam_to_beam_contact_condition.add_component(parameter<int>("COUPLING_ID"));
    condlist.push_back(beam_to_beam_contact_condition);
  }
}

FOUR_C_NAMESPACE_CLOSE
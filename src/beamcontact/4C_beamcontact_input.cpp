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



void BeamContact::set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list)
{
  using Teuchos::tuple;
  using namespace Core::IO::InputSpecBuilders;

  Core::Utils::SectionSpecs beamcontact{"BEAM CONTACT"};

  Core::Utils::string_to_integral_parameter<BeamContact::Strategy>("BEAMS_STRATEGY", "None",
      "Type of employed solving strategy",
      tuple<std::string>("None", "none", "Penalty", "penalty", "Gmshonly", "gmshonly"),
      tuple<BeamContact::Strategy>(
          bstr_none, bstr_none, bstr_penalty, bstr_penalty, bstr_gmshonly, bstr_gmshonly),
      beamcontact);

  Core::Utils::string_to_integral_parameter<BeamContact::Modelevaluator>("MODELEVALUATOR", "old",
      "Type of model evaluator", tuple<std::string>("Old", "old", "Standard", "standard"),
      tuple<BeamContact::Modelevaluator>(bstr_old, bstr_old, bstr_standard, bstr_standard),
      beamcontact);

  beamcontact.specs.emplace_back(parameter<bool>("BEAMS_NEWGAP",
      {.description = "choose between original or enhanced gapfunction", .default_value = false}));

  beamcontact.specs.emplace_back(parameter<bool>("BEAMS_SEGCON",
      {.description = "choose between beam contact with and without subsegment generation",
          .default_value = false}));

  beamcontact.specs.emplace_back(parameter<bool>(
      "BEAMS_DEBUG", {.description = "This flag can be used for testing purposes. When it is "
                                     "switched on, some sanity checks are not performed!",
                         .default_value = false}));

  beamcontact.specs.emplace_back(parameter<bool>(
      "BEAMS_INACTIVESTIFF", {.description = "Always apply contact stiffness in first Newton step "
                                             "for pairs which have active in last time step",
                                 .default_value = false}));

  beamcontact.specs.emplace_back(parameter<bool>("BEAMS_BTSOL",
      {.description = "decide, if also the contact between beams and solids is possible",
          .default_value = false}));

  beamcontact.specs.emplace_back(parameter<bool>("BEAMS_ENDPOINTPENALTY",
      {.description = "Additional consideration of endpoint-line and endpoint-endpoint contacts",
          .default_value = false}));

  Core::Utils::string_to_integral_parameter<BeamContact::Smoothing>("BEAMS_SMOOTHING", "None",
      "Application of smoothed tangent field", tuple<std::string>("None", "none", "Cpp", "cpp"),
      tuple<BeamContact::Smoothing>(bsm_none, bsm_none, bsm_cpp, bsm_cpp), beamcontact);

  beamcontact.specs.emplace_back(parameter<bool>("BEAMS_DAMPING",
      {.description = "Application of a contact damping force", .default_value = false}));

  beamcontact.specs.emplace_back(parameter<double>("BEAMS_BTBPENALTYPARAM",
      {.description = "Penalty parameter for beam-to-beam point contact", .default_value = 0.0}));
  beamcontact.specs.emplace_back(parameter<double>("BEAMS_BTBLINEPENALTYPARAM",
      {.description = "Penalty parameter per unit length for beam-to-beam line contact",
          .default_value = -1.0}));
  beamcontact.specs.emplace_back(parameter<double>("BEAMS_BTSPENALTYPARAM",
      {.description = "Penalty parameter for beam-to-solid contact", .default_value = 0.0}));
  beamcontact.specs.emplace_back(parameter<double>("BEAMS_DAMPINGPARAM",
      {.description = "Damping parameter for contact damping force", .default_value = -1000.0}));
  beamcontact.specs.emplace_back(parameter<double>("BEAMS_DAMPREGPARAM1",
      {.description =
              "First (at gap1, with gap1>gap2) regularization parameter for contact damping force",
          .default_value = -1000.0}));
  beamcontact.specs.emplace_back(parameter<double>("BEAMS_DAMPREGPARAM2",
      {.description =
              "Second (at gap2, with gap1>gap2) regularization parameter for contact damping force",
          .default_value = -1000.0}));
  beamcontact.specs.emplace_back(parameter<double>(
      "BEAMS_MAXDISISCALEFAC", {.description = "Scale factor in order to limit maximal iterative "
                                               "displacement increment (resiudal displacement)",
                                   .default_value = -1.0}));
  beamcontact.specs.emplace_back(parameter<double>("BEAMS_MAXDELTADISSCALEFAC",
      {.description = "Scale factor in order to limit maximal displacement per time step",
          .default_value = 1.0}));

  beamcontact.specs.emplace_back(parameter<double>("BEAMS_PERPSHIFTANGLE1",
      {.description = "Lower shift angle (in degrees) for penalty scaling of large-angle-contact",
          .default_value = -1.0}));
  beamcontact.specs.emplace_back(parameter<double>("BEAMS_PERPSHIFTANGLE2",
      {.description = "Upper shift angle (in degrees) for penalty scaling of large-angle-contact",
          .default_value = -1.0}));
  beamcontact.specs.emplace_back(parameter<double>("BEAMS_PARSHIFTANGLE1",
      {.description = "Lower shift angle (in degrees) for penalty scaling of small-angle-contact",
          .default_value = -1.0}));
  beamcontact.specs.emplace_back(parameter<double>("BEAMS_PARSHIFTANGLE2",
      {.description = "Upper shift angle (in degrees) for penalty scaling of small-angle-contact",
          .default_value = -1.0}));
  beamcontact.specs.emplace_back(parameter<double>("BEAMS_SEGANGLE",
      {.description = "Maximal angle deviation allowed for contact search segmentation",
          .default_value = -1.0}));
  Core::Utils::int_parameter("BEAMS_NUMINTEGRATIONINTERVAL", 1,
      "Number of integration intervals per element", beamcontact);

  Core::Utils::string_to_integral_parameter<BeamContact::PenaltyLaw>("BEAMS_PENALTYLAW", "LinPen",
      "Applied Penalty Law",
      tuple<std::string>("LinPen", "QuadPen", "LinNegQuadPen", "LinPosQuadPen", "LinPosCubPen",
          "LinPosDoubleQuadPen", "LinPosExpPen"),
      tuple<BeamContact::PenaltyLaw>(pl_lp, pl_qp, pl_lnqp, pl_lpqp, pl_lpcp, pl_lpdqp, pl_lpep),
      beamcontact);

  beamcontact.specs.emplace_back(parameter<double>("BEAMS_PENREGPARAM_G0",
      {.description =
              "First penalty regularization parameter G0 >=0: For gap<G0 contact is active!",
          .default_value = -1.0}));
  beamcontact.specs.emplace_back(parameter<double>("BEAMS_PENREGPARAM_F0",
      {.description = "Second penalty regularization parameter F0 >=0: F0 represents the force at "
                      "the transition point between regularized and linear force law!",
          .default_value = -1.0}));
  beamcontact.specs.emplace_back(parameter<double>("BEAMS_PENREGPARAM_C0",
      {.description = "Third penalty regularization parameter C0 >=0: C0 has different physical "
                      "meanings for the different penalty laws!",
          .default_value = -1.0}));
  beamcontact.specs.emplace_back(parameter<double>("BEAMS_GAPSHIFTPARAM",
      {.description = "Parameter to shift penalty law!", .default_value = 0.0}));
  beamcontact.specs.emplace_back(parameter<double>(
      "BEAMS_BASICSTIFFGAP", {.description = "For gaps > -BEAMS_BASICSTIFFGAP, only the basic part "
                                             "of the contact linearization is applied!",
                                 .default_value = -1.0}));

  // enable octree search and determine type of bounding box (aabb = axis aligned, cobb =
  // cylindrical oriented)
  Core::Utils::string_to_integral_parameter<BeamContact::OctreeType>("BEAMS_OCTREE", "None",
      "octree and bounding box type for octree search routine",
      tuple<std::string>(
          "None", "none", "octree_axisaligned", "octree_cylorient", "octree_spherical"),
      tuple<BeamContact::OctreeType>(boct_none, boct_none, boct_aabb, boct_cobb, boct_spbb),
      beamcontact);

  beamcontact.specs.emplace_back(parameter<bool>(
      "BEAMS_ADDITEXT", {.description = "Switch between No==multiplicative extrusion factor and "
                                        "Yes==additive extrusion factor",
                            .default_value = true}));
  beamcontact.specs.emplace_back(parameter<std::string>("BEAMS_EXTVAL",
      {.description = "extrusion value(s) of the bounding box, Depending on BEAMS_ADDITIVEEXTFAC "
                      "is either  additive or multiplicative. Give one or two values.",
          .default_value = "-1.0"}));

  Core::Utils::int_parameter("BEAMS_TREEDEPTH", 6, "max, tree depth of the octree", beamcontact);
  Core::Utils::int_parameter(
      "BEAMS_BOXESINOCT", 8, "max number of bounding boxes in any leaf octant", beamcontact);

  beamcontact.move_into_collection(list);

  /*------------------------------------------------------------------------*/
  /* parameters for visualization of beam contact via output at runtime */

  Core::Utils::SectionSpecs beamcontact_vtk_sublist{beamcontact, "RUNTIME VTK OUTPUT"};

  // whether to write visualization output for beam contact
  beamcontact_vtk_sublist.specs.emplace_back(parameter<bool>("VTK_OUTPUT_BEAM_CONTACT",
      {.description = "write visualization output for beam contact", .default_value = false}));

  // output interval regarding steps: write output every INTERVAL_STEPS steps
  Core::Utils::int_parameter("INTERVAL_STEPS", -1,
      "write visualization output at runtime every INTERVAL_STEPS steps", beamcontact_vtk_sublist);

  // whether to write output in every iteration of the nonlinear solver
  beamcontact_vtk_sublist.specs.emplace_back(parameter<bool>(
      "EVERY_ITERATION", {.description = "write output in every iteration of the nonlinear solver",
                             .default_value = false}));

  // whether to write visualization output for contact forces
  beamcontact_vtk_sublist.specs.emplace_back(parameter<bool>("CONTACT_FORCES",
      {.description = "write visualization output for contact forces", .default_value = false}));

  // whether to write visualization output for gaps
  beamcontact_vtk_sublist.specs.emplace_back(parameter<bool>(
      "GAPS", {.description = "write visualization output for gap, i.e. penetration",
                  .default_value = false}));

  beamcontact_vtk_sublist.move_into_collection(list);
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

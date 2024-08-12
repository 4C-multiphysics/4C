/*----------------------------------------------------------------------*/
/*! \file

\brief input parameter definitions for beam contact

\level 2

*/
/*----------------------------------------------------------------------*/



#include "4C_inpar_beamcontact.hpp"

#include "4C_fem_condition_definition.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN



void Inpar::BEAMCONTACT::SetValidParameters(Teuchos::RCP<Teuchos::ParameterList> list)
{
  using namespace Input;
  using Teuchos::setStringToIntegralParameter;
  using Teuchos::tuple;

  Teuchos::ParameterList& beamcontact = list->sublist("BEAM CONTACT", false, "");

  setStringToIntegralParameter<int>("BEAMS_STRATEGY", "None", "Type of employed solving strategy",
      tuple<std::string>("None", "none", "Penalty", "penalty", "Gmshonly", "gmshonly"),
      tuple<int>(bstr_none, bstr_none, bstr_penalty, bstr_penalty, bstr_gmshonly, bstr_gmshonly),
      &beamcontact);

  setStringToIntegralParameter<int>("MODELEVALUATOR", "old", "Type of model evaluator",
      tuple<std::string>("Old", "old", "Standard", "standard"),
      tuple<int>(bstr_old, bstr_old, bstr_standard, bstr_standard), &beamcontact);

  Core::UTILS::BoolParameter(
      "BEAMS_NEWGAP", "No", "choose between original or enhanced gapfunction", &beamcontact);

  Core::UTILS::BoolParameter("BEAMS_SEGCON", "No",
      "choose between beam contact with and without subsegment generation", &beamcontact);

  Core::UTILS::BoolParameter("BEAMS_DEBUG", "No",
      "This flag can be used for testing purposes. When it is switched on, some sanity checks are "
      "not performed!",
      &beamcontact);

  Core::UTILS::BoolParameter("BEAMS_INACTIVESTIFF", "No",
      "Always apply contact stiffness in first Newton step for pairs which have active in last "
      "time step",
      &beamcontact);

  Core::UTILS::BoolParameter("BEAMS_BTSOL", "No",
      "decide, if also the contact between beams and solids is possible", &beamcontact);

  Core::UTILS::BoolParameter("BEAMS_ENDPOINTPENALTY", "No",
      "Additional consideration of endpoint-line and endpoint-endpoint contacts", &beamcontact);

  setStringToIntegralParameter<int>("BEAMS_SMOOTHING", "None",
      "Application of smoothed tangent field", tuple<std::string>("None", "none", "Cpp", "cpp"),
      tuple<int>(bsm_none, bsm_none, bsm_cpp, bsm_cpp), &beamcontact);

  setStringToIntegralParameter<int>("BEAMS_DAMPING", "No", "Application of a contact damping force",
      tuple<std::string>("No", "no", "Yes", "yes"), tuple<int>(bd_no, bd_no, bd_yes, bd_yes),
      &beamcontact);

  Core::UTILS::DoubleParameter("BEAMS_BTBPENALTYPARAM", 0.0,
      "Penalty parameter for beam-to-beam point contact", &beamcontact);
  Core::UTILS::DoubleParameter("BEAMS_BTBLINEPENALTYPARAM", -1.0,
      "Penalty parameter per unit length for beam-to-beam line contact", &beamcontact);
  Core::UTILS::DoubleParameter(
      "BEAMS_BTSPENALTYPARAM", 0.0, "Penalty parameter for beam-to-solid contact", &beamcontact);
  Core::UTILS::DoubleParameter(
      "BEAMS_DAMPINGPARAM", -1000.0, "Damping parameter for contact damping force", &beamcontact);
  Core::UTILS::DoubleParameter("BEAMS_DAMPREGPARAM1", -1000.0,
      "First (at gap1, with gap1>gap2) regularization parameter for contact damping force",
      &beamcontact);
  Core::UTILS::DoubleParameter("BEAMS_DAMPREGPARAM2", -1000.0,
      "Second (at gap2, with gap1>gap2) regularization parameter for contact damping force",
      &beamcontact);
  Core::UTILS::DoubleParameter("BEAMS_MAXDISISCALEFAC", -1.0,
      "Scale factor in order to limit maximal iterative displacement increment (resiudal "
      "displacement)",
      &beamcontact);
  Core::UTILS::DoubleParameter("BEAMS_MAXDELTADISSCALEFAC", 1.0,
      "Scale factor in order to limit maximal displacement per time step", &beamcontact);

  Core::UTILS::DoubleParameter("BEAMS_PERPSHIFTANGLE1", -1.0,
      "Lower shift angle (in degrees) for penalty scaling of large-angle-contact", &beamcontact);
  Core::UTILS::DoubleParameter("BEAMS_PERPSHIFTANGLE2", -1.0,
      "Upper shift angle (in degrees) for penalty scaling of large-angle-contact", &beamcontact);
  Core::UTILS::DoubleParameter("BEAMS_PARSHIFTANGLE1", -1.0,
      "Lower shift angle (in degrees) for penalty scaling of small-angle-contact", &beamcontact);
  Core::UTILS::DoubleParameter("BEAMS_PARSHIFTANGLE2", -1.0,
      "Upper shift angle (in degrees) for penalty scaling of small-angle-contact", &beamcontact);
  Core::UTILS::DoubleParameter("BEAMS_SEGANGLE", -1.0,
      "Maximal angle deviation allowed for contact search segmentation", &beamcontact);
  Core::UTILS::IntParameter("BEAMS_NUMINTEGRATIONINTERVAL", 1,
      "Number of integration intervals per element", &beamcontact);

  setStringToIntegralParameter<int>("BEAMS_PENALTYLAW", "LinPen", "Applied Penalty Law",
      tuple<std::string>("LinPen", "QuadPen", "LinNegQuadPen", "LinPosQuadPen", "LinPosCubPen",
          "LinPosDoubleQuadPen", "LinPosExpPen"),
      tuple<int>(pl_lp, pl_qp, pl_lnqp, pl_lpqp, pl_lpcp, pl_lpdqp, pl_lpep), &beamcontact);

  Core::UTILS::DoubleParameter("BEAMS_PENREGPARAM_G0", -1.0,
      "First penalty regularization parameter G0 >=0: For gap<G0 contact is active!", &beamcontact);
  Core::UTILS::DoubleParameter("BEAMS_PENREGPARAM_F0", -1.0,
      "Second penalty regularization parameter F0 >=0: F0 represents the force at the transition "
      "point between regularized and linear force law!",
      &beamcontact);
  Core::UTILS::DoubleParameter("BEAMS_PENREGPARAM_C0", -1.0,
      "Third penalty regularization parameter C0 >=0: C0 has different physical meanings for the "
      "different penalty laws!",
      &beamcontact);
  Core::UTILS::DoubleParameter(
      "BEAMS_GAPSHIFTPARAM", 0.0, "Parameter to shift penalty law!", &beamcontact);
  Core::UTILS::DoubleParameter("BEAMS_BASICSTIFFGAP", -1.0,
      "For gaps > -BEAMS_BASICSTIFFGAP, only the basic part of the contact linearization is "
      "applied!",
      &beamcontact);

  // enable octree search and determine type of bounding box (aabb = axis aligned, cobb =
  // cylindrical oriented)
  setStringToIntegralParameter<int>("BEAMS_OCTREE", "None",
      "octree and bounding box type for octree search routine",
      tuple<std::string>(
          "None", "none", "octree_axisaligned", "octree_cylorient", "octree_spherical"),
      tuple<int>(boct_none, boct_none, boct_aabb, boct_cobb, boct_spbb), &beamcontact);

  Core::UTILS::BoolParameter("BEAMS_ADDITEXT", "Yes",
      "Switch between No==multiplicative extrusion factor and Yes==additive extrusion factor",
      &beamcontact);
  setNumericStringParameter("BEAMS_EXTVAL", "-1.0",
      "extrusion value(s) of the bounding box, Depending on BEAMS_ADDITIVEEXTFAC is either "
      "additive or multiplicative. Give one or two values.",
      &beamcontact);
  Core::UTILS::IntParameter("BEAMS_TREEDEPTH", 6, "max, tree depth of the octree", &beamcontact);
  Core::UTILS::IntParameter(
      "BEAMS_BOXESINOCT", 8, "max number of bounding boxes in any leaf octant", &beamcontact);


  /*------------------------------------------------------------------------*/
  /* parameters for visualization of beam contact via output at runtime */

  Teuchos::ParameterList& beamcontact_vtk_sublist =
      beamcontact.sublist("RUNTIME VTK OUTPUT", false, "");


  // whether to write visualization output for beam contact
  Core::UTILS::BoolParameter("VTK_OUTPUT_BEAM_CONTACT", "No",
      "write visualization output for beam contact", &beamcontact_vtk_sublist);

  // output interval regarding steps: write output every INTERVAL_STEPS steps
  Core::UTILS::IntParameter("INTERVAL_STEPS", -1,
      "write visualization output at runtime every INTERVAL_STEPS steps", &beamcontact_vtk_sublist);

  // whether to write output in every iteration of the nonlinear solver
  Core::UTILS::BoolParameter("EVERY_ITERATION", "No",
      "write output in every iteration of the nonlinear solver", &beamcontact_vtk_sublist);

  // whether to write visualization output for contact forces
  Core::UTILS::BoolParameter("CONTACT_FORCES", "No",
      "write visualization output for contact forces", &beamcontact_vtk_sublist);

  // whether to write visualization output for gaps
  Core::UTILS::BoolParameter("GAPS", "No", "write visualization output for gap, i.e. penetration",
      &beamcontact_vtk_sublist);
}

/**
 *
 */
void Inpar::BEAMCONTACT::SetValidConditions(
    std::vector<Teuchos::RCP<Core::Conditions::ConditionDefinition>>& condlist)
{
  using namespace Input;

  // Beam-to-beam conditions.
  {
    std::string condition_name = "BeamToBeamContact";

    Teuchos::RCP<Core::Conditions::ConditionDefinition> beam_to_beam_contact_condition =
        Teuchos::rcp(new Core::Conditions::ConditionDefinition(
            "BEAM INTERACTION/BEAM TO BEAM CONTACT CONDITIONS", condition_name,
            "Beam-to-beam contact conditions", Core::Conditions::BeamToBeamContact, true,
            Core::Conditions::geometry_type_line));
    add_named_int(beam_to_beam_contact_condition, "COUPLING_ID");
    condlist.push_back(beam_to_beam_contact_condition);
  }
}

FOUR_C_NAMESPACE_CLOSE

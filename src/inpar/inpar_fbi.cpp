/*-----------------------------------------------------------*/
/*! \file
\file inpar_fbi.cpp

\brief input parameter for Fluid-Beam Interaction


\level 3

*/
/*-----------------------------------------------------------*/


#include "inpar_fbi.H"
#include "validparameters.H"
#include "conditiondefinition.H"
#include "inpar_geometry_pair.H"

void INPAR::FBI::SetValidParameters(Teuchos::RCP<Teuchos::ParameterList> list)
{
  using namespace DRT::INPUT;
  using Teuchos::setStringToIntegralParameter;
  using Teuchos::tuple;

  Teuchos::ParameterList& fbi = list->sublist("FLUID BEAM INTERACTION", false, "");

  /*----------------------------------------------------------------------*/
  /* parameters for beam to fluid meshtying */

  setStringToIntegralParameter<BeamToFluidCoupling>("COUPLING", "two-way", "Type of FBI coupling",
      tuple<std::string>("two-way", "fluid", "solid"),
      tuple<BeamToFluidCoupling>(
          BeamToFluidCoupling::twoway, BeamToFluidCoupling::fluid, BeamToFluidCoupling::solid),
      &fbi);

  IntParameter("STARTSTEP", 0,
      "Time Step at which to begin the fluid beam coupling. Usually this will be the first step.",
      &fbi);

  setStringToIntegralParameter<BeamToFluidPreSortStrategy>("PRESORT_STRATEGY", "bruteforce",
      "Presort strategy for the beam elements", tuple<std::string>("bruteforce", "binning"),
      tuple<BeamToFluidPreSortStrategy>(
          BeamToFluidPreSortStrategy::bruteforce, BeamToFluidPreSortStrategy::binning),
      &fbi);

  /*----------------------------------------------------------------------*/

  Teuchos::ParameterList& beam_to_fluid_meshtying =
      fbi.sublist("BEAM TO FLUID MESHTYING", false, "");

  setStringToIntegralParameter<BeamToFluidDiscretization>("MESHTYING_DISCRETIZATION", "none",
      "Type of employed meshtying discretization",
      tuple<std::string>("none", "gauss_point_to_segment", "mortar"),
      tuple<FBI::BeamToFluidDiscretization>(BeamToFluidDiscretization::none,
          BeamToFluidDiscretization::gauss_point_to_segment, BeamToFluidDiscretization::mortar),
      &beam_to_fluid_meshtying);

  setStringToIntegralParameter<BeamToFluidConstraintEnforcement>("CONSTRAINT_STRATEGY", "none",
      "Type of employed constraint enforcement strategy", tuple<std::string>("none", "penalty"),
      tuple<BeamToFluidConstraintEnforcement>(
          BeamToFluidConstraintEnforcement::none, BeamToFluidConstraintEnforcement::penalty),
      &beam_to_fluid_meshtying);

  DoubleParameter("PENALTY_PARAMETER", 0.0, "Penalty parameter for beam-to-Fluid volume meshtying",
      &beam_to_fluid_meshtying);

  DoubleParameter("SEARCH_RADIUS", 1000,
      "Absolute Search radius for beam-to-fluid volume meshtying. Choose carefully to not blow up "
      "memory demand but to still find all interaction pairs!",
      &beam_to_fluid_meshtying);

  setStringToIntegralParameter<INPAR::FBI::BeamToFluidMeshtingMortarShapefunctions>(
      "MORTAR_SHAPE_FUNCTION", "none", "Shape function for the mortar Lagrange-multipliers",
      tuple<std::string>("none", "line2", "line3", "line4"),
      tuple<INPAR::FBI::BeamToFluidMeshtingMortarShapefunctions>(
          INPAR::FBI::BeamToFluidMeshtingMortarShapefunctions::none,
          INPAR::FBI::BeamToFluidMeshtingMortarShapefunctions::line2,
          INPAR::FBI::BeamToFluidMeshtingMortarShapefunctions::line3,
          INPAR::FBI::BeamToFluidMeshtingMortarShapefunctions::line4),
      &beam_to_fluid_meshtying);

  // Add the geometry pair input parameters.
  INPAR::GEOMETRYPAIR::SetValidParametersLineTo3D(beam_to_fluid_meshtying);

  /*----------------------------------------------------------------------*/

  // Create subsection for runtime vtk output.
  Teuchos::ParameterList& beam_to_fluid_meshtying_vtk =
      beam_to_fluid_meshtying.sublist("RUNTIME VTK OUTPUT", false, "");

  // Whether to write vtp output at all for beam to fluid meshtying.
  BoolParameter("WRITE_OUTPUT", "No", "Enable / disable beam-to-fluid mesh tying output.",
      &beam_to_fluid_meshtying_vtk);

  BoolParameter("NODAL_FORCES", "No",
      "Enable / disable output of the resulting nodal forces due to beam to Fluid interaction.",
      &beam_to_fluid_meshtying_vtk);

  BoolParameter("SEGMENTATION", "No", "Enable / disable output of segmentation points.",
      &beam_to_fluid_meshtying_vtk);

  BoolParameter("INTEGRATION_POINTS", "No",
      "Enable / disable output of used integration points. If the meshtying method has 'forces' at "
      "the integration point, they will also be output.",
      &beam_to_fluid_meshtying_vtk);

  BoolParameter("CONSTRAINT_VIOLATION", "No",
      "Enable / disable output of the constraint violation into a output_name.penalty csv file.",
      &beam_to_fluid_meshtying_vtk);

  BoolParameter("MORTAR_LAMBDA_DISCRET", "No",
      "Enable / disable output of the discrete Lagrange multipliers at the node of the Lagrange "
      "multiplier shape functions.",
      &beam_to_fluid_meshtying_vtk);

  BoolParameter("MORTAR_LAMBDA_CONTINUOUS", "No",
      "Enable / disable output of the continuous Lagrange multipliers function along the beam.",
      &beam_to_fluid_meshtying_vtk);

  DRT::INPUT::IntParameter("MORTAR_LAMBDA_CONTINUOUS_SEGMENTS", 5,
      "Number of segments for continuous mortar output", &beam_to_fluid_meshtying_vtk);
}

void INPAR::FBI::SetValidConditions(
    std::vector<Teuchos::RCP<DRT::INPUT::ConditionDefinition>>& condlist)
{
  using namespace DRT::INPUT;

  /*-------------------------------------------------------------------*/
}

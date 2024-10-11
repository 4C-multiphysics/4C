/*-----------------------------------------------------------*/
/*! \file
\brief input parameter for Brownian dynamics simulation


\level 2

*/
/*-----------------------------------------------------------*/


#include "4C_inpar_browniandyn.hpp"

#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN


void Inpar::BrownianDynamics::set_valid_parameters(Teuchos::ParameterList& list)
{
  using Teuchos::setStringToIntegralParameter;
  using Teuchos::tuple;

  Teuchos::ParameterList& browniandyn_list = list.sublist("BROWNIAN DYNAMICS", false, "");

  Core::UTILS::bool_parameter(
      "BROWNDYNPROB", "No", "switch Brownian dynamics on/off", &browniandyn_list);

  // Reading double parameter for viscosity of background fluid
  Core::UTILS::double_parameter("VISCOSITY", 0.0, "viscosity", &browniandyn_list);

  // Reading double parameter for thermal energy in background fluid (temperature * Boltzmann
  // constant)
  Core::UTILS::double_parameter("KT", 0.0, "thermal energy", &browniandyn_list);

  // cutoff for random forces, which determines the maximal value
  Core::UTILS::double_parameter("MAXRANDFORCE", -1.0,
      "Any random force beyond MAXRANDFORCE*(standard dev.) will be omitted and redrawn. "
      "-1.0 means no bounds.'",
      &browniandyn_list);

  // time interval in which random numbers are constant
  Core::UTILS::double_parameter("TIMESTEP", -1.0,
      "Within this time interval the random numbers remain constant. -1.0 ", &browniandyn_list);

  // the way how damping coefficient values for beams are specified
  setStringToIntegralParameter<BeamDampingCoefficientSpecificationType>(
      "BEAMS_DAMPING_COEFF_SPECIFIED_VIA", "cylinder_geometry_approx",
      "In which way are damping coefficient values for beams specified?",
      tuple<std::string>(
          "cylinder_geometry_approx", "Cylinder_geometry_approx", "input_file", "Input_file"),
      tuple<BeamDampingCoefficientSpecificationType>(
          Inpar::BrownianDynamics::cylinder_geometry_approx,
          Inpar::BrownianDynamics::cylinder_geometry_approx, Inpar::BrownianDynamics::input_file,
          Inpar::BrownianDynamics::input_file),
      &browniandyn_list);

  // values for damping coefficients of beams if they are specified via input file
  // (per unit length, NOT yet multiplied by fluid viscosity)
  Core::UTILS::string_parameter("BEAMS_DAMPING_COEFF_PER_UNITLENGTH", "0.0 0.0 0.0",
      "values for beam damping coefficients (per unit length and NOT yet multiplied by fluid "
      "viscosity): "
      "translational perpendicular/parallel to beam axis, rotational around axis",
      &browniandyn_list);
}

FOUR_C_NAMESPACE_CLOSE

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_browniandyn_input.hpp"

#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN


void BrownianDynamics::set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list)
{
  using Teuchos::tuple;
  using namespace Core::IO::InputSpecBuilders;

  Core::Utils::SectionSpecs browniandyn_list{"BROWNIAN DYNAMICS"};

  browniandyn_list.specs.emplace_back(parameter<bool>(
      "BROWNDYNPROB", {.description = "switch Brownian dynamics on/off", .default_value = false}));

  // Reading double parameter for viscosity of background fluid
  Core::Utils::double_parameter("VISCOSITY", 0.0, "viscosity", browniandyn_list);

  // Reading double parameter for thermal energy in background fluid (temperature * Boltzmann
  // constant)
  Core::Utils::double_parameter("KT", 0.0, "thermal energy", browniandyn_list);

  // cutoff for random forces, which determines the maximal value
  Core::Utils::double_parameter("MAXRANDFORCE", -1.0,
      "Any random force beyond MAXRANDFORCE*(standard dev.) will be omitted and redrawn. "
      "-1.0 means no bounds.'",
      browniandyn_list);

  // time interval in which random numbers are constant
  Core::Utils::double_parameter("TIMESTEP", -1.0,
      "Within this time interval the random numbers remain constant. -1.0 ", browniandyn_list);

  // the way how damping coefficient values for beams are specified
  Core::Utils::string_to_integral_parameter<BeamDampingCoefficientSpecificationType>(
      "BEAMS_DAMPING_COEFF_SPECIFIED_VIA", "cylinder_geometry_approx",
      "In which way are damping coefficient values for beams specified?",
      tuple<std::string>(
          "cylinder_geometry_approx", "Cylinder_geometry_approx", "input_file", "Input_file"),
      tuple<BeamDampingCoefficientSpecificationType>(BrownianDynamics::cylinder_geometry_approx,
          BrownianDynamics::cylinder_geometry_approx, BrownianDynamics::input_file,
          BrownianDynamics::input_file),
      browniandyn_list);

  // values for damping coefficients of beams if they are specified via input file
  // (per unit length, NOT yet multiplied by fluid viscosity)
  Core::Utils::string_parameter("BEAMS_DAMPING_COEFF_PER_UNITLENGTH", "0.0 0.0 0.0",
      "values for beam damping coefficients (per unit length and NOT yet multiplied by fluid "
      "viscosity): "
      "translational perpendicular/parallel to beam axis, rotational around axis",
      browniandyn_list);

  browniandyn_list.move_into_collection(list);
}

FOUR_C_NAMESPACE_CLOSE

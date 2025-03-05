// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_lubrication_input.hpp"

#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN

void Lubrication::set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list)
{
  using Teuchos::tuple;
  using namespace Core::IO::InputSpecBuilders;

  Core::Utils::SectionSpecs lubricationdyn("LUBRICATION DYNAMIC");

  lubricationdyn.specs.emplace_back(parameter<double>(
      "MAXTIME", {.description = "Total simulation time", .default_value = 1000.0}));
  Core::Utils::int_parameter("NUMSTEP", 20, "Total number of time steps", lubricationdyn);
  lubricationdyn.specs.emplace_back(
      parameter<double>("TIMESTEP", {.description = "Time increment dt", .default_value = 0.1}));
  Core::Utils::int_parameter("RESULTSEVERY", 1, "Increment for writing solution", lubricationdyn);
  Core::Utils::int_parameter("RESTARTEVERY", 1, "Increment for writing restart", lubricationdyn);

  Core::Utils::string_to_integral_parameter<Lubrication::CalcError>("CALCERROR", "No",
      "compute error compared to analytical solution",
      tuple<std::string>("No", "error_by_function"),
      tuple<Lubrication::CalcError>(calcerror_no, calcerror_byfunction), lubricationdyn);

  Core::Utils::int_parameter(
      "CALCERRORNO", -1, "function number for lubrication error computation", lubricationdyn);

  Core::Utils::string_to_integral_parameter<Lubrication::VelocityField>("VELOCITYFIELD", "zero",
      "type of velocity field used for lubrication problems",
      tuple<std::string>("zero", "function", "EHL"),
      tuple<Lubrication::VelocityField>(velocity_zero, velocity_function, velocity_EHL),
      lubricationdyn);

  Core::Utils::int_parameter(
      "VELFUNCNO", -1, "function number for lubrication velocity field", lubricationdyn);

  Core::Utils::string_to_integral_parameter<Lubrication::HeightField>("HEIGHTFEILD", "zero",
      "type of height field used for lubrication problems",
      tuple<std::string>("zero", "function", "EHL"),
      tuple<Lubrication::HeightField>(height_zero, height_function, height_EHL), lubricationdyn);

  Core::Utils::int_parameter(
      "HFUNCNO", -1, "function number for lubrication height field", lubricationdyn);

  lubricationdyn.specs.emplace_back(parameter<bool>("OUTMEAN",
      {.description = "Output of mean values for scalars and density", .default_value = false}));

  lubricationdyn.specs.emplace_back(parameter<bool>("OUTPUT_GMSH",
      {.description = "Do you want to write Gmsh postprocessing files?", .default_value = false}));

  lubricationdyn.specs.emplace_back(parameter<bool>("MATLAB_STATE_OUTPUT",
      {.description = "Do you want to write the state solution to Matlab file?",
          .default_value = false}));

  /// linear solver id used for lubrication problems
  Core::Utils::int_parameter("LINEAR_SOLVER", -1,
      "number of linear solver used for the Lubrication problem", lubricationdyn);

  Core::Utils::int_parameter("ITEMAX", 10, "max. number of nonlin. iterations", lubricationdyn);
  lubricationdyn.specs.emplace_back(parameter<double>("ABSTOLRES",
      {.description =
              "Absolute tolerance for deciding if residual of nonlinear problem is already zero",
          .default_value = 1e-14}));
  lubricationdyn.specs.emplace_back(parameter<double>(
      "CONVTOL", {.description = "Tolerance for convergence check", .default_value = 1e-13}));

  // convergence criteria adaptivity
  lubricationdyn.specs.emplace_back(parameter<bool>("ADAPTCONV",
      {.description =
              "Switch on adaptive control of linear solver tolerance for nonlinear solution",
          .default_value = false}));
  lubricationdyn.specs.emplace_back(parameter<double>("ADAPTCONV_BETTER",
      {.description = "The linear solver shall be this much better than the current nonlinear "
                      "residual in the nonlinear convergence limit",
          .default_value = 0.1}));

  Core::Utils::string_to_integral_parameter<ConvNorm>("NORM_PRE", "Abs",
      "type of norm for temperature convergence check", tuple<std::string>("Abs", "Rel", "Mix"),
      tuple<ConvNorm>(convnorm_abs, convnorm_rel, convnorm_mix), lubricationdyn);

  Core::Utils::string_to_integral_parameter<ConvNorm>("NORM_RESF", "Abs",
      "type of norm for residual convergence check", tuple<std::string>("Abs", "Rel", "Mix"),
      tuple<ConvNorm>(convnorm_abs, convnorm_rel, convnorm_mix), lubricationdyn);

  Core::Utils::string_to_integral_parameter<VectorNorm>("ITERNORM", "L2",
      "type of norm to be applied to residuals", tuple<std::string>("L1", "L2", "Rms", "Inf"),
      tuple<VectorNorm>(norm_l1, norm_l2, norm_rms, norm_inf), lubricationdyn);

  /// Iterationparameters
  lubricationdyn.specs.emplace_back(parameter<double>(
      "TOLPRE", {.description = "tolerance in the temperature norm of the Newton iteration",
                    .default_value = 1.0E-06}));

  lubricationdyn.specs.emplace_back(parameter<double>(
      "TOLRES", {.description = "tolerance in the residual norm for the Newton iteration",
                    .default_value = 1.0E-06}));

  lubricationdyn.specs.emplace_back(parameter<double>("PENALTY_CAVITATION",
      {.description = "penalty parameter for regularized cavitation", .default_value = 0.}));

  lubricationdyn.specs.emplace_back(parameter<double>(
      "GAP_OFFSET", {.description = "Additional offset to the fluid gap", .default_value = 0.}));

  lubricationdyn.specs.emplace_back(parameter<double>("ROUGHNESS_STD_DEVIATION",
      {.description = "standard deviation of surface roughness", .default_value = 0.}));

  /// use modified reynolds equ.
  lubricationdyn.specs.emplace_back(parameter<bool>("MODIFIED_REYNOLDS_EQU",
      {.description = "the lubrication problem will use the modified reynolds equ. in order to "
                      "consider surface roughness",
          .default_value = false}));

  /// Flag for considering the Squeeze term in Reynolds Equation
  lubricationdyn.specs.emplace_back(parameter<bool>("ADD_SQUEEZE_TERM",
      {.description = "the squeeze term will also be considered in the Reynolds Equation",
          .default_value = false}));

  /// Flag for considering the pure Reynolds Equation
  lubricationdyn.specs.emplace_back(parameter<bool>(
      "PURE_LUB", {.description = "the problem is pure lubrication", .default_value = false}));

  lubricationdyn.move_into_collection(list);
}

FOUR_C_NAMESPACE_CLOSE

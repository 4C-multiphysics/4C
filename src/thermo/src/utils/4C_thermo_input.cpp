// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_thermo_input.hpp"

#include "4C_fem_condition_definition.hpp"
#include "4C_io_geometry_type.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN

void Thermo::set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list)
{
  using Teuchos::tuple;
  using namespace Core::IO::InputSpecBuilders;

  Core::Utils::SectionSpecs tdyn{"THERMAL DYNAMIC"};

  Core::Utils::string_to_integral_parameter<DynamicType>("DYNAMICTYPE", "OneStepTheta",
      "type of time integration control", tuple<std::string>("Statics", "OneStepTheta", "GenAlpha"),
      tuple<DynamicType>(dyna_statics, dyna_onesteptheta, dyna_genalpha), tdyn);

  // output type
  tdyn.specs.emplace_back(parameter<int>("RESULTSEVERY",
      {.description = "save temperature and other global quantities every RESULTSEVERY steps",
          .default_value = 1}));

  tdyn.specs.emplace_back(parameter<int>("RESTARTEVERY",
      {.description = "write restart possibility every RESTARTEVERY steps", .default_value = 1}));


  Core::Utils::string_to_integral_parameter<InitialField>("INITIALFIELD", "zero_field",
      "Initial Field for thermal problem",
      tuple<std::string>("zero_field", "field_by_function", "field_by_condition"),
      tuple<InitialField>(
          initfield_zero_field, initfield_field_by_function, initfield_field_by_condition),
      tdyn);

  tdyn.specs.emplace_back(parameter<int>("INITFUNCNO",
      {.description = "function number for thermal initial field", .default_value = -1}));

  // Time loop control
  tdyn.specs.emplace_back(
      parameter<double>("TIMESTEP", {.description = "time step size", .default_value = 0.05}));
  tdyn.specs.emplace_back(
      parameter<int>("NUMSTEP", {.description = "maximum number of steps", .default_value = 200}));
  tdyn.specs.emplace_back(
      parameter<double>("MAXTIME", {.description = "maximum time", .default_value = 5.0}));

  // Iterationparameters
  tdyn.specs.emplace_back(parameter<double>(
      "TOLTEMP", {.description = "tolerance in the temperature norm of the Newton iteration",
                     .default_value = 1.0E-10}));

  Core::Utils::string_to_integral_parameter<ConvNorm>("NORM_TEMP", "Abs",
      "type of norm for temperature convergence check", tuple<std::string>("Abs", "Rel", "Mix"),
      tuple<ConvNorm>(convnorm_abs, convnorm_rel, convnorm_mix), tdyn);

  tdyn.specs.emplace_back(parameter<double>(
      "TOLRES", {.description = "tolerance in the residual norm for the Newton iteration",
                    .default_value = 1.0E-08}));

  Core::Utils::string_to_integral_parameter<ConvNorm>("NORM_RESF", "Abs",
      "type of norm for residual convergence check", tuple<std::string>("Abs", "Rel", "Mix"),
      tuple<ConvNorm>(convnorm_abs, convnorm_rel, convnorm_mix), tdyn);

  Core::Utils::string_to_integral_parameter<BinaryOp>("NORMCOMBI_RESFTEMP", "And",
      "binary operator to combine temperature and residual force values",
      tuple<std::string>("And", "Or"), tuple<BinaryOp>(bop_and, bop_or), tdyn);

  tdyn.specs.emplace_back(parameter<int>("MAXITER",
      {.description =
              "maximum number of iterations allowed for Newton-Raphson iteration before failure",
          .default_value = 50}));

  tdyn.specs.emplace_back(parameter<int>("MINITER",
      {.description = "minimum number of iterations to be done within Newton-Raphson loop",
          .default_value = 0}));

  Core::Utils::string_to_integral_parameter<VectorNorm>("ITERNORM", "L2",
      "type of norm to be applied to residuals", tuple<std::string>("L1", "L2", "Rms", "Inf"),
      tuple<VectorNorm>(norm_l1, norm_l2, norm_rms, norm_inf), tdyn);

  Core::Utils::string_to_integral_parameter<DivContAct>("DIVERCONT", "stop",
      "What to do with time integration when Newton-Raphson iteration failed",
      tuple<std::string>("stop", "continue", "halve_step", "repeat_step", "repeat_simulation"),
      tuple<DivContAct>(divcont_stop, divcont_continue, divcont_halve_step, divcont_repeat_step,
          divcont_repeat_simulation),
      tdyn);

  tdyn.specs.emplace_back(parameter<int>("MAXDIVCONREFINEMENTLEVEL",
      {.description = "number of times timestep is halved in case nonlinear solver diverges",
          .default_value = 10}));

  Core::Utils::string_to_integral_parameter<NonlinSolTech>("NLNSOL", "fullnewton",
      "Nonlinear solution technique", tuple<std::string>("vague", "fullnewton"),
      tuple<NonlinSolTech>(soltech_vague, soltech_newtonfull), tdyn);

  Core::Utils::string_to_integral_parameter<PredEnum>("PREDICT", "ConstTemp",
      "Predictor of iterative solution techniques",
      tuple<std::string>("Vague", "ConstTemp", "ConstTempRate", "TangTemp"),
      tuple<PredEnum>(pred_vague, pred_consttemp, pred_consttemprate, pred_tangtemp), tdyn);

  // convergence criteria solver adaptivity
  tdyn.specs.emplace_back(parameter<bool>("ADAPTCONV",
      {.description =
              "Switch on adaptive control of linear solver tolerance for nonlinear solution",
          .default_value = false}));
  tdyn.specs.emplace_back(parameter<double>("ADAPTCONV_BETTER",
      {.description = "The linear solver shall be this much better than the current nonlinear "
                      "residual in the nonlinear convergence limit",
          .default_value = 0.1}));

  tdyn.specs.emplace_back(parameter<bool>(
      "LUMPCAPA", {.description = "Lump the capacity matrix for explicit time integration",
                      .default_value = false}));

  // number of linear solver used for thermal problems
  tdyn.specs.emplace_back(parameter<int>("LINEAR_SOLVER",
      {.description = "number of linear solver used for thermal problems", .default_value = -1}));

  // where the geometry comes from
  Core::Utils::string_to_integral_parameter<Core::IO::GeometryType>("GEOMETRY", "full",
      "How the geometry is specified", tuple<std::string>("full", "box", "file"),
      tuple<Core::IO::GeometryType>(
          Core::IO::geometry_full, Core::IO::geometry_box, Core::IO::geometry_file),
      tdyn);

  Core::Utils::string_to_integral_parameter<CalcError>("CALCERROR", "No",
      "compute error compared to analytical solution", tuple<std::string>("No", "byfunct"),
      tuple<CalcError>(no_error_calculation, calcerror_byfunct), tdyn);
  tdyn.specs.emplace_back(parameter<int>(
      "CALCERRORFUNCNO", {.description = "Function for Error Calculation", .default_value = -1}));

  tdyn.move_into_collection(list);

  /*----------------------------------------------------------------------*/
  /* parameters for generalised-alpha thermal integrator */
  Core::Utils::SectionSpecs tgenalpha{tdyn, "GENALPHA"};

  Core::Utils::string_to_integral_parameter<MidAverageEnum>("GENAVG", "TrLike",
      "mid-average type of internal forces", tuple<std::string>("Vague", "ImrLike", "TrLike"),
      tuple<MidAverageEnum>(midavg_vague, midavg_imrlike, midavg_trlike), tgenalpha);

  // default values correspond to midpoint-rule
  tgenalpha.specs.emplace_back(parameter<double>(
      "GAMMA", {.description = "Generalised-alpha factor in (0,1]", .default_value = 0.5}));
  tgenalpha.specs.emplace_back(parameter<double>(
      "ALPHA_M", {.description = "Generalised-alpha factor in [0.5,1)", .default_value = 0.5}));
  tgenalpha.specs.emplace_back(parameter<double>(
      "ALPHA_F", {.description = "Generalised-alpha factor in [0.5,1)", .default_value = 0.5}));
  tgenalpha.specs.emplace_back(parameter<double>(
      "RHO_INF", {.description = "Generalised-alpha factor in [0,1]", .default_value = -1.0}));

  tgenalpha.move_into_collection(list);

  /*----------------------------------------------------------------------*/
  /* parameters for one-step-theta thermal integrator */
  Core::Utils::SectionSpecs tonesteptheta{tdyn, "ONESTEPTHETA"};

  tonesteptheta.specs.emplace_back(parameter<double>(
      "THETA", {.description = "One-step-theta factor in (0,1]", .default_value = 0.5}));

  tonesteptheta.move_into_collection(list);

  // vtk runtime output
  {
    Core::Utils::SectionSpecs sublist_vtk_output{tdyn, "RUNTIME VTK OUTPUT"};

    // whether to write output for thermo
    sublist_vtk_output.specs.emplace_back(parameter<bool>(
        "OUTPUT_THERMO", {.description = "write thermo output", .default_value = false}));

    // whether to write temperature state
    sublist_vtk_output.specs.emplace_back(parameter<bool>(
        "TEMPERATURE", {.description = "write temperature output", .default_value = false}));

    // whether to write heatflux state
    sublist_vtk_output.specs.emplace_back(parameter<bool>(
        "HEATFLUX", {.description = "write heatflux output", .default_value = false}));

    // whether to write temperature gradient state
    sublist_vtk_output.specs.emplace_back(parameter<bool>(
        "TEMPGRAD", {.description = "write temperature gradient output", .default_value = false}));

    // whether to write element owner
    sublist_vtk_output.specs.emplace_back(parameter<bool>(
        "ELEMENT_OWNER", {.description = "write element owner", .default_value = false}));

    // whether to write element GIDs
    sublist_vtk_output.specs.emplace_back(parameter<bool>(
        "ELEMENT_GID", {.description = "write 4C internal element GIDs", .default_value = false}));

    // whether to write node GIDs
    sublist_vtk_output.specs.emplace_back(parameter<bool>(
        "NODE_GID", {.description = "write 4C internal node GIDs", .default_value = false}));

    sublist_vtk_output.move_into_collection(list);
  }

  // csv runtime output
  {
    Core::Utils::SectionSpecs sublist_csv_output{tdyn, "RUNTIME CSV OUTPUT"};

    // whether to write csv output for thermo
    sublist_csv_output.specs.emplace_back(parameter<bool>(
        "OUTPUT_THERMO", {.description = "write thermo output", .default_value = false}));

    // whether to write energy state
    sublist_csv_output.specs.emplace_back(
        parameter<bool>("ENERGY", {.description = "write energy output", .default_value = false}));

    sublist_csv_output.move_into_collection(list);
  }
}



void Thermo::set_valid_conditions(std::vector<Core::Conditions::ConditionDefinition>& condlist)
{
  using namespace Core::IO::InputSpecBuilders;

  /*--------------------------------------------------------------------*/
  // Convective heat transfer (Newton's law of heat transfer)

  Core::Conditions::ConditionDefinition linethermoconvect(
      "DESIGN THERMO CONVECTION LINE CONDITIONS", "ThermoConvections", "Line Thermo Convections",
      Core::Conditions::ThermoConvections, true, Core::Conditions::geometry_type_line);
  Core::Conditions::ConditionDefinition surfthermoconvect(
      "DESIGN THERMO CONVECTION SURF CONDITIONS", "ThermoConvections", "Surface Thermo Convections",
      Core::Conditions::ThermoConvections, true, Core::Conditions::geometry_type_surface);

  const auto make_thermoconvect = [&condlist](Core::Conditions::ConditionDefinition& cond)
  {
    // decide here if approximation is sufficient
    // --> Tempn (old temperature T_n)
    // or if the exact solution is needed
    // --> Tempnp (current temperature solution T_n+1) with linearisation
    cond.add_component(deprecated_selection<std::string>(
        "temperature_state", {"Tempnp", "Tempn"}, {.description = "temperature state"}));
    cond.add_component(parameter<double>("coeff", {.description = "heat transfer coefficient h"}));
    cond.add_component(
        parameter<double>("surtemp", {.description = "surrounding (fluid) temperature T_oo"}));
    cond.add_component(parameter<std::optional<int>>("surtempfunct",
        {.description =
                "time curve to increase the surrounding (fluid) temperature T_oo in time"}));
    cond.add_component(parameter<std::optional<int>>("funct",
        {.description =
                "time curve to increase the complete boundary condition, i.e., the heat flux"}));
    condlist.push_back(cond);
  };

  make_thermoconvect(linethermoconvect);
  make_thermoconvect(surfthermoconvect);

  /*--------------------------------------------------------------------*/
  // Robin boundary conditions for heat transfer
  // NOTE: this condition must be
  Core::Conditions::ConditionDefinition thermorobinline("DESIGN THERMO ROBIN LINE CONDITIONS",
      "ThermoRobin", "Thermo Robin boundary condition", Core::Conditions::ThermoRobin, true,
      Core::Conditions::geometry_type_line);
  Core::Conditions::ConditionDefinition thermorobinsurf("DESIGN THERMO ROBIN SURF CONDITIONS",
      "ThermoRobin", "Thermo Robin boundary condition", Core::Conditions::ThermoRobin, true,
      Core::Conditions::geometry_type_surface);

  const auto make_thermorobin = [&condlist](Core::Conditions::ConditionDefinition& cond)
  {
    cond.add_component(parameter<int>("NUMSCAL"));
    cond.add_component(parameter<std::vector<int>>(
        "ONOFF", {.description = "", .size = from_parameter<int>("NUMSCAL")}));
    cond.add_component(parameter<double>("PREFACTOR"));
    cond.add_component(parameter<double>("REFVALUE"));

    condlist.push_back(cond);
  };

  make_thermorobin(thermorobinline);
  make_thermorobin(thermorobinsurf);
}

FOUR_C_NAMESPACE_CLOSE

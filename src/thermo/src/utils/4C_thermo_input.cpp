// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_thermo_input.hpp"

#include "4C_fem_condition_definition.hpp"
#include "4C_io_input_spec_builders.hpp"

FOUR_C_NAMESPACE_OPEN

std::vector<Core::IO::InputSpec> Thermo::valid_parameters()
{
  using namespace Core::IO::InputSpecBuilders;

  std::vector<Core::IO::InputSpec> specs;
  specs.push_back(group("THERMAL DYNAMIC",
      {

          parameter<Thermo::DynamicType>(
              "DYNAMICTYPE", {.description = "type of time integration control",
                                 .default_value = Thermo::DynamicType::OneStepTheta}),

          // output type
          parameter<int>("RESULTSEVERY",
              {.description =
                      "save temperature and other global quantities every RESULTSEVERY steps",
                  .default_value = 1}),

          parameter<int>(
              "RESTARTEVERY", {.description = "write restart possibility every RESTARTEVERY steps",
                                  .default_value = 1}),


          deprecated_selection<InitialField>("INITIALFIELD",
              {
                  {"zero_field", initfield_zero_field},
                  {"field_by_function", initfield_field_by_function},
                  {"field_by_condition", initfield_field_by_condition},
              },
              {.description = "Initial Field for thermal problem",
                  .default_value = initfield_zero_field}),

          parameter<int>("INITFUNCNO",
              {.description = "function number for thermal initial field", .default_value = -1}),

          // Time loop control
          parameter<double>("TIMESTEP", {.description = "time step size", .default_value = 0.05}),

          parameter<int>(
              "NUMSTEP", {.description = "maximum number of steps", .default_value = 200}),

          parameter<double>("MAXTIME", {.description = "maximum time", .default_value = 5.0}),

          // Iterationparameters
          parameter<double>("TOLTEMP",
              {.description = "tolerance in the temperature norm of the Newton iteration",
                  .default_value = 1.0E-10}),

          deprecated_selection<ConvNorm>("NORM_TEMP",
              {
                  {"Abs", convnorm_abs},
                  {"Rel", convnorm_rel},
                  {"Mix", convnorm_mix},
              },
              {.description = "type of norm for temperature convergence check",
                  .default_value = convnorm_abs}),

          parameter<double>(
              "TOLRES", {.description = "tolerance in the residual norm for the Newton iteration",
                            .default_value = 1.0E-08}),

          deprecated_selection<ConvNorm>("NORM_RESF",
              {
                  {"Abs", convnorm_abs},
                  {"Rel", convnorm_rel},
                  {"Mix", convnorm_mix},
              },
              {.description = "type of norm for residual convergence check",
                  .default_value = convnorm_abs}),

          deprecated_selection<BinaryOp>("NORMCOMBI_RESFTEMP",
              {
                  {"And", bop_and},
                  {"Or", bop_or},
              },
              {.description = "binary operator to combine temperature and residual force values",
                  .default_value = bop_and}),

          parameter<int>("MAXITER", {.description = "maximum number of iterations allowed for "
                                                    "Newton-Raphson iteration before failure",
                                        .default_value = 50}),

          parameter<int>("MINITER",
              {.description = "minimum number of iterations to be done within Newton-Raphson loop",
                  .default_value = 0}),

          deprecated_selection<VectorNorm>("ITERNORM",
              {
                  {"L1", norm_l1},
                  {"L2", norm_l2},
                  {"Rms", norm_rms},
                  {"Inf", norm_inf},
              },
              {.description = "type of norm to be applied to residuals", .default_value = norm_l2}),

          deprecated_selection<DivContAct>("DIVERCONT",
              {
                  {"stop", divcont_stop},
                  {"continue", divcont_continue},
                  {"halve_step", divcont_halve_step},
                  {"repeat_step", divcont_repeat_step},
                  {"repeat_simulation", divcont_repeat_simulation},
              },
              {.description =
                      "What to do with time integration when Newton-Raphson iteration failed",
                  .default_value = divcont_stop}),

          parameter<int>("MAXDIVCONREFINEMENTLEVEL",
              {.description =
                      "number of times timestep is halved in case nonlinear solver diverges",
                  .default_value = 10}),

          deprecated_selection<NonlinSolTech>("NLNSOL",
              {
                  {"vague", soltech_vague},
                  {"fullnewton", soltech_newtonfull},
              },
              {.description = "Nonlinear solution technique", .default_value = soltech_newtonfull}),

          deprecated_selection<PredEnum>("PREDICT",
              {
                  {"Vague", pred_vague},
                  {"ConstTemp", pred_consttemp},
                  {"ConstTempRate", pred_consttemprate},
                  {"TangTemp", pred_tangtemp},
              },
              {.description = "Predictor of iterative solution techniques",
                  .default_value = pred_consttemp}),

          // convergence criteria solver adaptivity
          parameter<bool>("ADAPTCONV", {.description = "Switch on adaptive control of linear "
                                                       "solver tolerance for nonlinear solution",
                                           .default_value = false}),
          parameter<double>("ADAPTCONV_BETTER",
              {.description =
                      "The linear solver shall be this much better than the current nonlinear "
                      "residual in the nonlinear convergence limit",
                  .default_value = 0.1}),

          parameter<bool>(
              "LUMPCAPA", {.description = "Lump the capacity matrix for explicit time integration",
                              .default_value = false}),

          // number of linear solver used for thermal problems
          parameter<int>(
              "LINEAR_SOLVER", {.description = "number of linear solver used for thermal problems",
                                   .default_value = -1}),

          deprecated_selection<CalcError>("CALCERROR",
              {
                  {"No", no_error_calculation},
                  {"byfunct", calcerror_byfunct},
              },
              {.description = "compute error compared to analytical solution",
                  .default_value = no_error_calculation}),
          parameter<int>("CALCERRORFUNCNO",
              {.description = "Function for Error Calculation", .default_value = -1})},
      {.required = false}));
  /*----------------------------------------------------------------------*/
  /* parameters for generalised-alpha thermal integrator */
  specs.push_back(group("THERMAL DYNAMIC/GENALPHA",
      {

          deprecated_selection<MidAverageEnum>("GENAVG",
              {
                  {"Vague", midavg_vague},
                  {"ImrLike", midavg_imrlike},
                  {"TrLike", midavg_trlike},
              },
              {.description = "mid-average type of internal forces",
                  .default_value = midavg_trlike}),

          // default values correspond to midpoint-rule
          parameter<double>(
              "GAMMA", {.description = "Generalised-alpha factor in (0,1]", .default_value = 0.5}),
          parameter<double>("ALPHA_M",
              {.description = "Generalised-alpha factor in [0.5,1)", .default_value = 0.5}),
          parameter<double>("ALPHA_F",
              {.description = "Generalised-alpha factor in [0.5,1)", .default_value = 0.5}),
          parameter<double>("RHO_INF",
              {.description = "Generalised-alpha factor in [0,1]", .default_value = -1.0})},
      {.required = false}));

  /*----------------------------------------------------------------------*/
  /* parameters for one-step-theta thermal integrator */
  specs.push_back(group("THERMAL DYNAMIC/ONESTEPTHETA",
      {

          parameter<double>(
              "THETA", {.description = "One-step-theta factor in (0,1]", .default_value = 0.5})},
      {.required = false}));

  // vtk runtime output
  {
    specs.push_back(group("THERMAL DYNAMIC/RUNTIME VTK OUTPUT",
        {

            // whether to write output for thermo
            parameter<bool>(
                "OUTPUT_THERMO", {.description = "write thermo output", .default_value = false}),

            // whether to write temperature state
            parameter<bool>(
                "TEMPERATURE", {.description = "write temperature output", .default_value = false}),

            // whether to write temperature rate state
            parameter<bool>("TEMPERATURE_RATE",
                {.description = "write temperature rate output", .default_value = false}),

            // whether to write conductivity state
            parameter<bool>("CONDUCTIVITY",
                {.description = "write conductivity output", .default_value = false}),
            // whether to write heatflux state
            parameter<Thermo::HeatFluxType>("HEATFLUX",
                {.description = "write heatflux output", .default_value = HeatFluxType::None}),

            // whether to write tempgrad state
            parameter<Thermo::TempGradType>("TEMPGRAD",
                {.description = "write tempgrad output", .default_value = TempGradType::None}),

            // whether to write element owner
            parameter<bool>(
                "ELEMENT_OWNER", {.description = "write element owner", .default_value = false}),

            // whether to write element GIDs
            parameter<bool>("ELEMENT_GID",
                {.description = "write 4C internal element GIDs", .default_value = false}),

            // whether to write node GIDs
            parameter<bool>("NODE_GID",
                {.description = "write 4C internal node GIDs", .default_value = false})},
        {.required = false}));
  }

  // csv runtime output
  {
    specs.push_back(group("THERMAL DYNAMIC/RUNTIME CSV OUTPUT",
        {

            // whether to write csv output for thermo
            parameter<bool>(
                "OUTPUT_THERMO", {.description = "write thermo output", .default_value = false}),

            // whether to write energy state
            parameter<bool>(
                "ENERGY", {.description = "write energy output", .default_value = false})},
        {.required = false}));
  }
  return specs;
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
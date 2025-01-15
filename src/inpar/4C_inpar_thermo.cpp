// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_thermo.hpp"

#include "4C_fem_condition_definition.hpp"
#include "4C_io_geometry_type.hpp"
#include "4C_io_linecomponent.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN



void Inpar::Thermo::set_valid_parameters(Teuchos::ParameterList& list)
{
  using namespace Input;
  using Teuchos::setStringToIntegralParameter;
  using Teuchos::tuple;

  Teuchos::ParameterList& tdyn = list.sublist("THERMAL DYNAMIC", false, "");

  setStringToIntegralParameter<DynamicType>("DYNAMICTYPE", "OneStepTheta",
      "type of time integration control",
      tuple<std::string>("Statics", "OneStepTheta", "GenAlpha", "ExplicitEuler"),
      tuple<DynamicType>(dyna_statics, dyna_onesteptheta, dyna_genalpha, dyna_expleuler), &tdyn);

  // output type
  Core::Utils::int_parameter("RESULTSEVERY", 1,
      "save temperature and other global quantities every RESULTSEVERY steps", &tdyn);
  Core::Utils::int_parameter(
      "RESEVERYERGY", 0, "write system energies every requested step", &tdyn);
  Core::Utils::int_parameter(
      "RESTARTEVERY", 1, "write restart possibility every RESTARTEVERY steps", &tdyn);

  setStringToIntegralParameter<InitialField>("INITIALFIELD", "zero_field",
      "Initial Field for thermal problem",
      tuple<std::string>("zero_field", "field_by_function", "field_by_condition"),
      tuple<InitialField>(
          initfield_zero_field, initfield_field_by_function, initfield_field_by_condition),
      &tdyn);

  Core::Utils::int_parameter("INITFUNCNO", -1, "function number for thermal initial field", &tdyn);

  // Time loop control
  Core::Utils::double_parameter("TIMESTEP", 0.05, "time step size", &tdyn);
  Core::Utils::int_parameter("NUMSTEP", 200, "maximum number of steps", &tdyn);
  Core::Utils::double_parameter("MAXTIME", 5.0, "maximum time", &tdyn);

  // Iterationparameters
  Core::Utils::double_parameter(
      "TOLTEMP", 1.0E-10, "tolerance in the temperature norm of the Newton iteration", &tdyn);

  setStringToIntegralParameter<ConvNorm>("NORM_TEMP", "Abs",
      "type of norm for temperature convergence check", tuple<std::string>("Abs", "Rel", "Mix"),
      tuple<ConvNorm>(convnorm_abs, convnorm_rel, convnorm_mix), &tdyn);

  Core::Utils::double_parameter(
      "TOLRES", 1.0E-08, "tolerance in the residual norm for the Newton iteration", &tdyn);

  setStringToIntegralParameter<ConvNorm>("NORM_RESF", "Abs",
      "type of norm for residual convergence check", tuple<std::string>("Abs", "Rel", "Mix"),
      tuple<ConvNorm>(convnorm_abs, convnorm_rel, convnorm_mix), &tdyn);

  setStringToIntegralParameter<BinaryOp>("NORMCOMBI_RESFTEMP", "And",
      "binary operator to combine temperature and residual force values",
      tuple<std::string>("And", "Or"), tuple<BinaryOp>(bop_and, bop_or), &tdyn);

  Core::Utils::int_parameter("MAXITER", 50,
      "maximum number of iterations allowed for Newton-Raphson iteration before failure", &tdyn);

  Core::Utils::int_parameter(
      "MINITER", 0, "minimum number of iterations to be done within Newton-Raphson loop", &tdyn);

  setStringToIntegralParameter<VectorNorm>("ITERNORM", "L2",
      "type of norm to be applied to residuals", tuple<std::string>("L1", "L2", "Rms", "Inf"),
      tuple<VectorNorm>(norm_l1, norm_l2, norm_rms, norm_inf), &tdyn);

  setStringToIntegralParameter<DivContAct>("DIVERCONT", "stop",
      "What to do with time integration when Newton-Raphson iteration failed",
      tuple<std::string>("stop", "continue", "halve_step", "repeat_step", "repeat_simulation"),
      tuple<DivContAct>(divcont_stop, divcont_continue, divcont_halve_step, divcont_repeat_step,
          divcont_repeat_simulation),
      &tdyn);

  Core::Utils::int_parameter("MAXDIVCONREFINEMENTLEVEL", 10,
      "number of times timestep is halved in case nonlinear solver diverges", &tdyn);

  setStringToIntegralParameter<NonlinSolTech>("NLNSOL", "fullnewton",
      "Nonlinear solution technique", tuple<std::string>("vague", "fullnewton"),
      tuple<NonlinSolTech>(soltech_vague, soltech_newtonfull), &tdyn);

  setStringToIntegralParameter<PredEnum>("PREDICT", "ConstTemp",
      "Predictor of iterative solution techniques",
      tuple<std::string>("Vague", "ConstTemp", "ConstTempRate", "TangTemp"),
      tuple<PredEnum>(pred_vague, pred_consttemp, pred_consttemprate, pred_tangtemp), &tdyn);

  // convergence criteria solver adaptivity
  Core::Utils::bool_parameter("ADAPTCONV", "No",
      "Switch on adaptive control of linear solver tolerance for nonlinear solution", &tdyn);
  Core::Utils::double_parameter("ADAPTCONV_BETTER", 0.1,
      "The linear solver shall be this much better than the current nonlinear residual in the "
      "nonlinear convergence limit",
      &tdyn);

  Core::Utils::bool_parameter(
      "LUMPCAPA", "No", "Lump the capacity matrix for explicit time integration", &tdyn);

  // number of linear solver used for thermal problems
  Core::Utils::int_parameter(
      "LINEAR_SOLVER", -1, "number of linear solver used for thermal problems", &tdyn);

  // where the geometry comes from
  setStringToIntegralParameter<Core::IO::GeometryType>("GEOMETRY", "full",
      "How the geometry is specified", tuple<std::string>("full", "box", "file"),
      tuple<Core::IO::GeometryType>(
          Core::IO::geometry_full, Core::IO::geometry_box, Core::IO::geometry_file),
      &tdyn);

  setStringToIntegralParameter<CalcError>("CALCERROR", "No",
      "compute error compared to analytical solution", tuple<std::string>("No", "byfunct"),
      tuple<CalcError>(no_error_calculation, calcerror_byfunct), &tdyn);
  Core::Utils::int_parameter("CALCERRORFUNCNO", -1, "Function for Error Calculation", &tdyn);

  /*----------------------------------------------------------------------*/
  /* parameters for generalised-alpha thermal integrator */
  Teuchos::ParameterList& tgenalpha = tdyn.sublist("GENALPHA", false, "");

  setStringToIntegralParameter<MidAverageEnum>("GENAVG", "TrLike",
      "mid-average type of internal forces", tuple<std::string>("Vague", "ImrLike", "TrLike"),
      tuple<MidAverageEnum>(midavg_vague, midavg_imrlike, midavg_trlike), &tgenalpha);

  // default values correspond to midpoint-rule
  Core::Utils::double_parameter("GAMMA", 0.5, "Generalised-alpha factor in (0,1]", &tgenalpha);
  Core::Utils::double_parameter("ALPHA_M", 0.5, "Generalised-alpha factor in [0.5,1)", &tgenalpha);
  Core::Utils::double_parameter("ALPHA_F", 0.5, "Generalised-alpha factor in [0.5,1)", &tgenalpha);
  Core::Utils::double_parameter("RHO_INF", -1.0, "Generalised-alpha factor in [0,1]", &tgenalpha);

  /*----------------------------------------------------------------------*/
  /* parameters for one-step-theta thermal integrator */
  Teuchos::ParameterList& tonesteptheta = tdyn.sublist("ONESTEPTHETA", false, "");

  Core::Utils::double_parameter("THETA", 0.5, "One-step-theta factor in (0,1]", &tonesteptheta);
}



void Inpar::Thermo::set_valid_conditions(
    std::vector<std::shared_ptr<Core::Conditions::ConditionDefinition>>& condlist)
{
  using namespace Input;

  /*--------------------------------------------------------------------*/
  // Convective heat transfer (Newton's law of heat transfer)

  std::shared_ptr<Core::Conditions::ConditionDefinition> linethermoconvect =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN THERMO CONVECTION LINE CONDITIONS", "ThermoConvections",
          "Line Thermo Convections", Core::Conditions::ThermoConvections, true,
          Core::Conditions::geometry_type_line);
  std::shared_ptr<Core::Conditions::ConditionDefinition> surfthermoconvect =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN THERMO CONVECTION SURF CONDITIONS", "ThermoConvections",
          "Surface Thermo Convections", Core::Conditions::ThermoConvections, true,
          Core::Conditions::geometry_type_surface);

  for (const auto& cond : {linethermoconvect, surfthermoconvect})
  {
    // decide here if approximation is sufficient
    // --> Tempn (old temperature T_n)
    // or if the exact solution is needed
    // --> Tempnp (current temperature solution T_n+1) with linearisation
    add_named_selection_component(cond, "temperature_state", "temperature state", "Tempnp",
        Teuchos::tuple<std::string>("Tempnp", "Tempn"),
        Teuchos::tuple<std::string>("Tempnp", "Tempn"));
    add_named_real(cond, "coeff", "heat transfer coefficient h");
    add_named_real(cond, "surtemp", "surrounding (fluid) temperature T_oo");
    add_named_int(cond, "surtempfunct",
        "time curve to increase the surrounding (fluid) temperature T_oo in time", 0, false);
    add_named_int(cond, "funct",
        "time curve to increase the complete boundary condition, i.e., the heat flux", 0, false);

    condlist.push_back(cond);
  }

  /*--------------------------------------------------------------------*/
  // Robin boundary conditions for heat transfer
  // NOTE: this condition must be
  std::shared_ptr<Core::Conditions::ConditionDefinition> thermorobinline =
      std::make_shared<Core::Conditions::ConditionDefinition>("DESIGN THERMO ROBIN LINE CONDITIONS",
          "ThermoRobin", "Thermo Robin boundary condition", Core::Conditions::ThermoRobin, true,
          Core::Conditions::geometry_type_line);
  std::shared_ptr<Core::Conditions::ConditionDefinition> thermorobinsurf =
      std::make_shared<Core::Conditions::ConditionDefinition>("DESIGN THERMO ROBIN SURF CONDITIONS",
          "ThermoRobin", "Thermo Robin boundary condition", Core::Conditions::ThermoRobin, true,
          Core::Conditions::geometry_type_surface);

  for (const auto& cond : {thermorobinline, thermorobinsurf})
  {
    add_named_int(cond, "NUMSCAL");
    add_named_int_vector(cond, "ONOFF", "", "NUMSCAL");
    add_named_real(cond, "PREFACTOR");
    add_named_real(cond, "REFVALUE");

    condlist.emplace_back(cond);
  }
}

FOUR_C_NAMESPACE_CLOSE

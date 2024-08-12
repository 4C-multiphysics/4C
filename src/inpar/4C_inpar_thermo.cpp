/*----------------------------------------------------------------------*/
/*! \file
\brief Input parameters for thermo problems

\level 1


*/

/*----------------------------------------------------------------------*/



#include "4C_inpar_thermo.hpp"

#include "4C_fem_condition_definition.hpp"
#include "4C_io_geometry_type.hpp"
#include "4C_io_linecomponent.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN



void Inpar::THR::SetValidParameters(Teuchos::RCP<Teuchos::ParameterList> list)
{
  using namespace Input;
  using Teuchos::setStringToIntegralParameter;
  using Teuchos::tuple;

  Teuchos::ParameterList& tdyn = list->sublist("THERMAL DYNAMIC", false, "");

  setStringToIntegralParameter<int>("DYNAMICTYP", "OneStepTheta",
      "type of time integration control",
      tuple<std::string>("Statics", "OneStepTheta", "GenAlpha", "ExplicitEuler"),
      tuple<int>(dyna_statics, dyna_onesteptheta, dyna_genalpha, dyna_expleuler), &tdyn);

  // output type
  Core::UTILS::IntParameter("RESULTSEVRY", 1,
      "save temperature and other global quantities every RESULTSEVRY steps", &tdyn);
  Core::UTILS::IntParameter("RESEVRYERGY", 0, "write system energies every requested step", &tdyn);
  Core::UTILS::IntParameter(
      "RESTARTEVRY", 1, "write restart possibility every RESTARTEVRY steps", &tdyn);

  setStringToIntegralParameter<int>("INITIALFIELD", "zero_field",
      "Initial Field for thermal problem",
      tuple<std::string>("zero_field", "field_by_function", "field_by_condition"),
      tuple<int>(initfield_zero_field, initfield_field_by_function, initfield_field_by_condition),
      &tdyn);

  Core::UTILS::IntParameter("INITFUNCNO", -1, "function number for thermal initial field", &tdyn);

  // Time loop control
  Core::UTILS::DoubleParameter("TIMESTEP", 0.05, "time step size", &tdyn);
  Core::UTILS::IntParameter("NUMSTEP", 200, "maximum number of steps", &tdyn);
  Core::UTILS::DoubleParameter("MAXTIME", 5.0, "maximum time", &tdyn);

  // Iterationparameters
  Core::UTILS::DoubleParameter(
      "TOLTEMP", 1.0E-10, "tolerance in the temperature norm of the Newton iteration", &tdyn);

  setStringToIntegralParameter<int>("NORM_TEMP", "Abs",
      "type of norm for temperature convergence check", tuple<std::string>("Abs", "Rel", "Mix"),
      tuple<int>(convnorm_abs, convnorm_rel, convnorm_mix), &tdyn);

  Core::UTILS::DoubleParameter(
      "TOLRES", 1.0E-08, "tolerance in the residual norm for the Newton iteration", &tdyn);

  setStringToIntegralParameter<int>("NORM_RESF", "Abs",
      "type of norm for residual convergence check", tuple<std::string>("Abs", "Rel", "Mix"),
      tuple<int>(convnorm_abs, convnorm_rel, convnorm_mix), &tdyn);

  setStringToIntegralParameter<int>("NORMCOMBI_RESFTEMP", "And",
      "binary operator to combine temperature and residual force values",
      tuple<std::string>("And", "Or"), tuple<int>(bop_and, bop_or), &tdyn);

  Core::UTILS::IntParameter("MAXITER", 50,
      "maximum number of iterations allowed for Newton-Raphson iteration before failure", &tdyn);

  Core::UTILS::IntParameter(
      "MINITER", 0, "minimum number of iterations to be done within Newton-Raphson loop", &tdyn);

  setStringToIntegralParameter<int>("ITERNORM", "L2", "type of norm to be applied to residuals",
      tuple<std::string>("L1", "L2", "Rms", "Inf"),
      tuple<int>(norm_l1, norm_l2, norm_rms, norm_inf), &tdyn);

  setStringToIntegralParameter<int>("DIVERCONT", "stop",
      "What to do with time integration when Newton-Raphson iteration failed",
      tuple<std::string>("stop", "continue", "halve_step", "repeat_step", "repeat_simulation"),
      tuple<int>(divcont_stop, divcont_continue, divcont_halve_step, divcont_repeat_step,
          divcont_repeat_simulation),
      &tdyn);

  Core::UTILS::IntParameter("MAXDIVCONREFINEMENTLEVEL", 10,
      "number of times timestep is halved in case nonlinear solver diverges", &tdyn);

  setStringToIntegralParameter<int>("NLNSOL", "fullnewton", "Nonlinear solution technique",
      tuple<std::string>("vague", "fullnewton"), tuple<int>(soltech_vague, soltech_newtonfull),
      &tdyn);

  setStringToIntegralParameter<int>("PREDICT", "ConstTemp",
      "Predictor of iterative solution techniques",
      tuple<std::string>("Vague", "ConstTemp", "ConstTempRate", "TangTemp"),
      tuple<int>(pred_vague, pred_consttemp, pred_consttemprate, pred_tangtemp), &tdyn);

  // convergence criteria solver adaptivity
  Core::UTILS::BoolParameter("ADAPTCONV", "No",
      "Switch on adaptive control of linear solver tolerance for nonlinear solution", &tdyn);
  Core::UTILS::DoubleParameter("ADAPTCONV_BETTER", 0.1,
      "The linear solver shall be this much better than the current nonlinear residual in the "
      "nonlinear convergence limit",
      &tdyn);

  Core::UTILS::BoolParameter(
      "LUMPCAPA", "No", "Lump the capacity matrix for explicit time integration", &tdyn);

  // number of linear solver used for thermal problems
  Core::UTILS::IntParameter(
      "LINEAR_SOLVER", -1, "number of linear solver used for thermal problems", &tdyn);

  // where the geometry comes from
  setStringToIntegralParameter<int>("GEOMETRY", "full", "How the geometry is specified",
      tuple<std::string>("full", "box", "file"),
      tuple<int>(Core::IO::geometry_full, Core::IO::geometry_box, Core::IO::geometry_file), &tdyn);

  setStringToIntegralParameter<int>("CALCERROR", "No",
      "compute error compared to analytical solution", tuple<std::string>("No", "byfunct"),
      tuple<int>(no_error_calculation, calcerror_byfunct), &tdyn);
  Core::UTILS::IntParameter("CALCERRORFUNCNO", -1, "Function for Error Calculation", &tdyn);

  /*----------------------------------------------------------------------*/
  /* parameters for generalised-alpha thermal integrator */
  Teuchos::ParameterList& tgenalpha = tdyn.sublist("GENALPHA", false, "");

  setStringToIntegralParameter<int>("GENAVG", "TrLike", "mid-average type of internal forces",
      tuple<std::string>("Vague", "ImrLike", "TrLike"),
      tuple<int>(midavg_vague, midavg_imrlike, midavg_trlike), &tgenalpha);

  // default values correspond to midpoint-rule
  Core::UTILS::DoubleParameter("GAMMA", 0.5, "Generalised-alpha factor in (0,1]", &tgenalpha);
  Core::UTILS::DoubleParameter("ALPHA_M", 0.5, "Generalised-alpha factor in [0.5,1)", &tgenalpha);
  Core::UTILS::DoubleParameter("ALPHA_F", 0.5, "Generalised-alpha factor in [0.5,1)", &tgenalpha);
  Core::UTILS::DoubleParameter("RHO_INF", -1.0, "Generalised-alpha factor in [0,1]", &tgenalpha);

  /*----------------------------------------------------------------------*/
  /* parameters for one-step-theta thermal integrator */
  Teuchos::ParameterList& tonesteptheta = tdyn.sublist("ONESTEPTHETA", false, "");

  Core::UTILS::DoubleParameter("THETA", 0.5, "One-step-theta factor in (0,1]", &tonesteptheta);
}



void Inpar::THR::SetValidConditions(
    std::vector<Teuchos::RCP<Core::Conditions::ConditionDefinition>>& condlist)
{
  using namespace Input;

  /*--------------------------------------------------------------------*/
  // Convective heat transfer (Newton's law of heat transfer)

  Teuchos::RCP<Core::Conditions::ConditionDefinition> linethermoconvect = Teuchos::rcp(
      new Core::Conditions::ConditionDefinition("DESIGN THERMO CONVECTION LINE CONDITIONS",
          "ThermoConvections", "Line Thermo Convections", Core::Conditions::ThermoConvections, true,
          Core::Conditions::geometry_type_line));
  Teuchos::RCP<Core::Conditions::ConditionDefinition> surfthermoconvect = Teuchos::rcp(
      new Core::Conditions::ConditionDefinition("DESIGN THERMO CONVECTION SURF CONDITIONS",
          "ThermoConvections", "Surface Thermo Convections", Core::Conditions::ThermoConvections,
          true, Core::Conditions::geometry_type_surface));

  for (const auto& cond : {linethermoconvect, surfthermoconvect})
  {
    // decide here if approximation is sufficient
    // --> Tempn (old temperature T_n)
    // or if the exact solution is needed
    // --> Tempnp (current temperature solution T_n+1) with linearisation
    cond->add_component(Teuchos::rcp(new Input::SelectionComponent("temperature state", "Tempnp",
        Teuchos::tuple<std::string>("Tempnp", "Tempn"),
        Teuchos::tuple<std::string>("Tempnp", "Tempn"))));
    add_named_real(cond, "coeff", "heat transfer coefficient h");
    add_named_real(cond, "surtemp", "surrounding (fluid) temperature T_oo");
    add_named_int(cond, "surtempfunct",
        "time curve to increase the surrounding (fluid) temperature T_oo in time", 0, false, true,
        true);
    add_named_int(cond, "funct",
        "time curve to increase the complete boundary condition, i.e., the heat flux", 0, false,
        true, true);

    condlist.push_back(cond);
  }

  /*--------------------------------------------------------------------*/
  // Robin boundary conditions for heat transfer
  // NOTE: this condition must be
  Teuchos::RCP<Core::Conditions::ConditionDefinition> thermorobinline =
      Teuchos::rcp(new Core::Conditions::ConditionDefinition("DESIGN THERMO ROBIN LINE CONDITIONS",
          "ThermoRobin", "Thermo Robin boundary condition", Core::Conditions::ThermoRobin, true,
          Core::Conditions::geometry_type_line));
  Teuchos::RCP<Core::Conditions::ConditionDefinition> thermorobinsurf =
      Teuchos::rcp(new Core::Conditions::ConditionDefinition("DESIGN THERMO ROBIN SURF CONDITIONS",
          "ThermoRobin", "Thermo Robin boundary condition", Core::Conditions::ThermoRobin, true,
          Core::Conditions::geometry_type_surface));

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

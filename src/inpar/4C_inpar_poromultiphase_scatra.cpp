/*----------------------------------------------------------------------*/
/*! \file
 \brief input parameters for porous multiphase problem with scalar transport

   \level 3

 *----------------------------------------------------------------------*/

#include "4C_inpar_poromultiphase_scatra.hpp"

#include "4C_fem_condition_definition.hpp"
#include "4C_linalg_equilibrate.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN


void Inpar::PoroMultiPhaseScaTra::set_valid_parameters(Teuchos::RCP<Teuchos::ParameterList> list)
{
  using namespace Input;
  using Teuchos::setStringToIntegralParameter;
  using Teuchos::tuple;

  // ----------------------------------------------------------------------
  // (1) general control parameters
  Teuchos::ParameterList& poromultiphasescatradyn = list->sublist("POROMULTIPHASESCATRA DYNAMIC",
      false, "Control paramters for scatra porous multiphase media coupling");

  // Output type
  Core::UTILS::int_parameter("RESTARTEVRY", 1, "write restart possibility every RESTARTEVRY steps",
      &poromultiphasescatradyn);
  // Time loop control
  Core::UTILS::int_parameter(
      "NUMSTEP", 200, "maximum number of Timesteps", &poromultiphasescatradyn);
  Core::UTILS::double_parameter(
      "MAXTIME", 1000.0, "total simulation time", &poromultiphasescatradyn);
  Core::UTILS::double_parameter("TIMESTEP", 0.05, "time step size dt", &poromultiphasescatradyn);
  Core::UTILS::int_parameter(
      "RESULTSEVRY", 1, "increment for writing solution", &poromultiphasescatradyn);
  Core::UTILS::int_parameter(
      "ITEMAX", 10, "maximum number of iterations over fields", &poromultiphasescatradyn);
  Core::UTILS::int_parameter(
      "ITEMIN", 1, "minimal number of iterations over fields", &poromultiphasescatradyn);

  // Coupling strategy for poroscatra solvers
  setStringToIntegralParameter<int>("COUPALGO", "twoway_partitioned_nested",
      "Coupling strategies for poroscatra solvers",
      tuple<std::string>(
          "twoway_partitioned_nested", "twoway_partitioned_sequential", "twoway_monolithic"),
      tuple<int>(solscheme_twoway_partitioned_nested, solscheme_twoway_partitioned_sequential,
          solscheme_twoway_monolithic),
      &poromultiphasescatradyn);

  // coupling with 1D artery network active
  Core::UTILS::bool_parameter(
      "ARTERY_COUPLING", "No", "Coupling with 1D blood vessels.", &poromultiphasescatradyn);

  // no convergence of coupling scheme
  setStringToIntegralParameter<int>("DIVERCONT", "stop",
      "What to do with time integration when Poromultiphase-Scatra iteration failed",
      tuple<std::string>("stop", "continue"), tuple<int>(divcont_stop, divcont_continue),
      &poromultiphasescatradyn);

  // ----------------------------------------------------------------------
  // (2) monolithic parameters
  Teuchos::ParameterList& poromultiphasescatradynmono = poromultiphasescatradyn.sublist(
      "MONOLITHIC", false, "Parameters for monolithic Poro-Multiphase-Scatra Interaction");

  setStringToIntegralParameter<int>("VECTORNORM_RESF", "L2",
      "type of norm to be applied to residuals",
      tuple<std::string>("L1", "L1_Scaled", "L2", "Rms", "Inf"),
      tuple<int>(Inpar::PoroMultiPhaseScaTra::norm_l1, Inpar::PoroMultiPhaseScaTra::norm_l1_scaled,
          Inpar::PoroMultiPhaseScaTra::norm_l2, Inpar::PoroMultiPhaseScaTra::norm_rms,
          Inpar::PoroMultiPhaseScaTra::norm_inf),
      &poromultiphasescatradynmono);

  setStringToIntegralParameter<int>("VECTORNORM_INC", "L2",
      "type of norm to be applied to residuals",
      tuple<std::string>("L1", "L1_Scaled", "L2", "Rms", "Inf"),
      tuple<int>(Inpar::PoroMultiPhaseScaTra::norm_l1, Inpar::PoroMultiPhaseScaTra::norm_l1_scaled,
          Inpar::PoroMultiPhaseScaTra::norm_l2, Inpar::PoroMultiPhaseScaTra::norm_rms,
          Inpar::PoroMultiPhaseScaTra::norm_inf),
      &poromultiphasescatradynmono);

  // convergence criteria adaptivity --> note ADAPTCONV_BETTER set pretty small
  Core::UTILS::bool_parameter("ADAPTCONV", "No",
      "Switch on adaptive control of linear solver tolerance for nonlinear solution",
      &poromultiphasescatradynmono);
  Core::UTILS::double_parameter("ADAPTCONV_BETTER", 0.001,
      "The linear solver shall be this much better "
      "than the current nonlinear residual in the nonlinear convergence limit",
      &poromultiphasescatradynmono);

  // Iterationparameters
  Core::UTILS::double_parameter("TOLRES_GLOBAL", 1e-8,
      "tolerance in the residual norm for the Newton iteration", &poromultiphasescatradynmono);
  Core::UTILS::double_parameter("TOLINC_GLOBAL", 1e-8,
      "tolerance in the increment norm for the Newton iteration", &poromultiphasescatradynmono);

  // number of linear solver used for poroelasticity
  Core::UTILS::int_parameter("LINEAR_SOLVER", -1,
      "number of linear solver used for monolithic poroscatra problems",
      &poromultiphasescatradynmono);

  // parameters for finite difference check
  setStringToIntegralParameter<int>("FDCHECK", "none",
      "flag for finite difference check: none or global",
      tuple<std::string>("none",
          "global"),  // perform finite difference check on time integrator level
      tuple<int>(fdcheck_none, fdcheck_global), &poromultiphasescatradynmono);

  // flag for equilibration of global system of equations
  setStringToIntegralParameter<Core::LinAlg::EquilibrationMethod>("EQUILIBRATION", "none",
      "flag for equilibration of global system of equations",
      tuple<std::string>("none", "rows_full", "rows_maindiag", "columns_full", "columns_maindiag",
          "rowsandcolumns_full", "rowsandcolumns_maindiag"),
      tuple<Core::LinAlg::EquilibrationMethod>(Core::LinAlg::EquilibrationMethod::none,
          Core::LinAlg::EquilibrationMethod::rows_full,
          Core::LinAlg::EquilibrationMethod::rows_maindiag,
          Core::LinAlg::EquilibrationMethod::columns_full,
          Core::LinAlg::EquilibrationMethod::columns_maindiag,
          Core::LinAlg::EquilibrationMethod::rowsandcolumns_full,
          Core::LinAlg::EquilibrationMethod::rowsandcolumns_maindiag),
      &poromultiphasescatradynmono);

  // ----------------------------------------------------------------------
  // (3) partitioned parameters
  Teuchos::ParameterList& poromultiphasescatradynpart = poromultiphasescatradyn.sublist(
      "PARTITIONED", false, "Parameters for partitioned Poro-Multiphase-Scatra Interaction");

  // convergence tolerance of outer iteration loop
  Core::UTILS::double_parameter("CONVTOL", 1e-6,
      "tolerance for convergence check of outer iteration", &poromultiphasescatradynpart);
}

void Inpar::PoroMultiPhaseScaTra::set_valid_conditions(
    std::vector<Teuchos::RCP<Core::Conditions::ConditionDefinition>>& condlist)
{
  using namespace Input;

  /*--------------------------------------------------------------------*/
  // oxygen partial pressure calculation condition
  {
    // definition of oxygen partial pressure calculation condition
    Teuchos::RCP<Core::Conditions::ConditionDefinition> oxypartpressline =
        Teuchos::rcp(new Core::Conditions::ConditionDefinition(
            "DESIGN OXYGEN PARTIAL PRESSURE CALCULATION LINE CONDITIONS",
            "PoroMultiphaseScatraOxyPartPressCalcCond",
            "PoroMultiphaseScatra Oxygen Partial Pressure Calculation line condition",
            Core::Conditions::PoroMultiphaseScatraOxyPartPressCalcCond, true,
            Core::Conditions::geometry_type_line));
    Teuchos::RCP<Core::Conditions::ConditionDefinition> oxypartpresssurf =
        Teuchos::rcp(new Core::Conditions::ConditionDefinition(
            "DESIGN OXYGEN PARTIAL PRESSURE CALCULATION SURF CONDITIONS",
            "PoroMultiphaseScatraOxyPartPressCalcCond",
            "PoroMultiphaseScatra Oxygen Partial Pressure Calculation surface condition",
            Core::Conditions::PoroMultiphaseScatraOxyPartPressCalcCond, true,
            Core::Conditions::geometry_type_surface));
    Teuchos::RCP<Core::Conditions::ConditionDefinition> oxypartpressvol =
        Teuchos::rcp(new Core::Conditions::ConditionDefinition(
            "DESIGN OXYGEN PARTIAL PRESSURE CALCULATION VOL CONDITIONS",
            "PoroMultiphaseScatraOxyPartPressCalcCond",
            "PoroMultiphaseScatra Oxygen Partial Pressure Calculation volume condition",
            Core::Conditions::PoroMultiphaseScatraOxyPartPressCalcCond, true,
            Core::Conditions::geometry_type_volume));

    for (const auto& cond : {oxypartpressline, oxypartpresssurf, oxypartpressvol})
    {
      // insert input file line components into condition definitions
      add_named_int(cond, "SCALARID");
      add_named_real(cond, "n");
      add_named_real(cond, "Pb50");
      add_named_real(cond, "CaO2_max");
      add_named_real(cond, "alpha_bl_eff");
      add_named_real(cond, "rho_oxy");
      add_named_real(cond, "rho_bl");

      // insert condition definitions into global list of valid condition definitions
      condlist.push_back(cond);
    }
  }
}

FOUR_C_NAMESPACE_CLOSE

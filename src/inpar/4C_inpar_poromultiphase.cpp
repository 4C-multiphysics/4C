/*----------------------------------------------------------------------*/
/*! \file
 \brief input parameters for porous multiphase problem

   \level 3

 *----------------------------------------------------------------------*/



#include "4C_inpar_poromultiphase.hpp"

#include "4C_discretization_condition_definition.hpp"
#include "4C_linalg_equilibrate.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN


void INPAR::POROMULTIPHASE::SetValidParameters(Teuchos::RCP<Teuchos::ParameterList> list)
{
  using namespace INPUT;
  using Teuchos::setStringToIntegralParameter;
  using Teuchos::tuple;

  // ----------------------------------------------------------------------
  // (1) general control parameters
  Teuchos::ParameterList& poromultiphasedyn = list->sublist(
      "POROMULTIPHASE DYNAMIC", false, "Control paramters for multiphase porous medium");

  // Output type
  CORE::UTILS::IntParameter(
      "RESTARTEVRY", 1, "write restart possibility every RESTARTEVRY steps", &poromultiphasedyn);
  // Time loop control
  CORE::UTILS::IntParameter("NUMSTEP", 200, "maximum number of Timesteps", &poromultiphasedyn);
  CORE::UTILS::DoubleParameter("MAXTIME", 1000.0, "total simulation time", &poromultiphasedyn);
  CORE::UTILS::DoubleParameter("TIMESTEP", -1, "time step size dt", &poromultiphasedyn);
  CORE::UTILS::IntParameter("RESULTSEVRY", 1, "increment for writing solution", &poromultiphasedyn);
  CORE::UTILS::IntParameter(
      "ITEMAX", 10, "maximum number of iterations over fields", &poromultiphasedyn);

  // here the computation of the structure can be skipped, this is helpful if only fluid-scatra
  // coupling should be calculated
  CORE::UTILS::BoolParameter(
      "SOLVE_STRUCTURE", "yes", "Flag to skip computation of structural field", &poromultiphasedyn);


  // Coupling strategy for solvers
  setStringToIntegralParameter<int>("COUPALGO", "twoway_partitioned",
      "Coupling strategies for poro multiphase solvers",
      tuple<std::string>("twoway_partitioned", "twoway_monolithic"),
      tuple<int>(solscheme_twoway_partitioned, solscheme_twoway_monolithic), &poromultiphasedyn);

  // coupling with 1D artery network active
  CORE::UTILS::BoolParameter(
      "ARTERY_COUPLING", "No", "Coupling with 1D blood vessels.", &poromultiphasedyn);

  // ----------------------------------------------------------------------
  // (2) monolithic parameters
  Teuchos::ParameterList& poromultiphasedynmono = poromultiphasedyn.sublist(
      "MONOLITHIC", false, "Parameters for monolithic Poro-Multiphase-Scatra Interaction");

  // convergence tolerances for monolithic coupling
  CORE::UTILS::DoubleParameter("TOLRES_GLOBAL", 1e-8,
      "tolerance in the residual norm for the Newton iteration", &poromultiphasedynmono);
  CORE::UTILS::DoubleParameter("TOLINC_GLOBAL", 1e-8,
      "tolerance in the increment norm for the Newton iteration", &poromultiphasedynmono);

  // number of linear solver used for poroelasticity
  CORE::UTILS::IntParameter("LINEAR_SOLVER", -1,
      "number of linear solver used for poroelasticity problems", &poromultiphasedynmono);

  // parameters for finite difference check
  setStringToIntegralParameter<int>("FDCHECK", "none",
      "flag for finite difference check: none or global",
      tuple<std::string>("none",
          "global"),  // perform finite difference check on time integrator level
      tuple<int>(fdcheck_none, fdcheck_global), &poromultiphasedynmono);

  setStringToIntegralParameter<int>("VECTORNORM_RESF", "L2",
      "type of norm to be applied to residuals",
      tuple<std::string>("L1", "L1_Scaled", "L2", "Rms", "Inf"),
      tuple<int>(INPAR::POROMULTIPHASE::norm_l1, INPAR::POROMULTIPHASE::norm_l1_scaled,
          INPAR::POROMULTIPHASE::norm_l2, INPAR::POROMULTIPHASE::norm_rms,
          INPAR::POROMULTIPHASE::norm_inf),
      &poromultiphasedynmono);

  setStringToIntegralParameter<int>("VECTORNORM_INC", "L2",
      "type of norm to be applied to residuals",
      tuple<std::string>("L1", "L1_Scaled", "L2", "Rms", "Inf"),
      tuple<int>(INPAR::POROMULTIPHASE::norm_l1, INPAR::POROMULTIPHASE::norm_l1_scaled,
          INPAR::POROMULTIPHASE::norm_l2, INPAR::POROMULTIPHASE::norm_rms,
          INPAR::POROMULTIPHASE::norm_inf),
      &poromultiphasedynmono);

  // flag for equilibration of global system of equations
  setStringToIntegralParameter<CORE::LINALG::EquilibrationMethod>("EQUILIBRATION", "none",
      "flag for equilibration of global system of equations",
      tuple<std::string>("none", "rows_full", "rows_maindiag", "columns_full", "columns_maindiag",
          "rowsandcolumns_full", "rowsandcolumns_maindiag"),
      tuple<CORE::LINALG::EquilibrationMethod>(CORE::LINALG::EquilibrationMethod::none,
          CORE::LINALG::EquilibrationMethod::rows_full,
          CORE::LINALG::EquilibrationMethod::rows_maindiag,
          CORE::LINALG::EquilibrationMethod::columns_full,
          CORE::LINALG::EquilibrationMethod::columns_maindiag,
          CORE::LINALG::EquilibrationMethod::rowsandcolumns_full,
          CORE::LINALG::EquilibrationMethod::rowsandcolumns_maindiag),
      &poromultiphasedynmono);

  // convergence criteria adaptivity --> note ADAPTCONV_BETTER set pretty small
  CORE::UTILS::BoolParameter("ADAPTCONV", "No",
      "Switch on adaptive control of linear solver tolerance for nonlinear solution",
      &poromultiphasedynmono);
  CORE::UTILS::DoubleParameter("ADAPTCONV_BETTER", 0.001,
      "The linear solver shall be this much better "
      "than the current nonlinear residual in the nonlinear convergence limit",
      &poromultiphasedynmono);

  // ----------------------------------------------------------------------
  // (3) partitioned parameters
  Teuchos::ParameterList& poromultiphasedynpart = poromultiphasedyn.sublist(
      "PARTITIONED", false, "Parameters for partitioned Poro-Multiphase-Scatra Interaction");

  // convergence tolerance of outer iteration loop
  CORE::UTILS::DoubleParameter("CONVTOL", 1e-6,
      "tolerance for convergence check of outer iteration", &poromultiphasedynpart);

  // flag for relaxation of partitioned scheme
  setStringToIntegralParameter<int>("RELAXATION", "none",
      "flag for relaxation of partitioned scheme", tuple<std::string>("none", "Constant", "Aitken"),
      tuple<int>(relaxation_none, relaxation_constant, relaxation_aitken), &poromultiphasedynpart);

  // parameters for relaxation of partitioned coupling
  CORE::UTILS::DoubleParameter(
      "STARTOMEGA", 1.0, "fixed relaxation parameter", &poromultiphasedynpart);
  CORE::UTILS::DoubleParameter(
      "MINOMEGA", 0.1, "smallest omega allowed for Aitken relaxation", &poromultiphasedynpart);
  CORE::UTILS::DoubleParameter(
      "MAXOMEGA", 10.0, "largest omega allowed for Aitken relaxation", &poromultiphasedynpart);
}

FOUR_C_NAMESPACE_CLOSE

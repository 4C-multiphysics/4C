/*----------------------------------------------------------------------*/
/*! \file
\brief input parameters for solid-scatra-thermo-interaction

\level 2

*/

/*----------------------------------------------------------------------*/

#include "4C_inpar_ssti.hpp"

#include "4C_fem_condition_definition.hpp"
#include "4C_inpar_s2i.hpp"
#include "4C_inpar_scatra.hpp"
#include "4C_io_linecomponent.hpp"
#include "4C_linalg_equilibrate.hpp"
#include "4C_linalg_sparseoperator.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN

void Inpar::SSTI::set_valid_parameters(Teuchos::ParameterList& list)
{
  using namespace Input;
  using Teuchos::setStringToIntegralParameter;
  using Teuchos::tuple;

  Teuchos::ParameterList& sstidyn = list.sublist(
      "SSTI CONTROL", false, "Control paramters for scatra structure thermo interaction");

  Core::Utils::double_parameter(
      "RESTARTEVRYTIME", 0, "write restart possibility every RESTARTEVRY steps", &sstidyn);
  Core::Utils::int_parameter(
      "RESTARTEVRY", 1, "write restart possibility every RESTARTEVRY steps", &sstidyn);
  Core::Utils::int_parameter("NUMSTEP", 200, "maximum number of Timesteps", &sstidyn);
  Core::Utils::double_parameter("MAXTIME", 1000.0, "total simulation time", &sstidyn);
  Core::Utils::double_parameter("TIMESTEP", -1, "time step size dt", &sstidyn);
  Core::Utils::double_parameter("RESULTSEVRYTIME", 0, "increment for writing solution", &sstidyn);
  Core::Utils::int_parameter("RESULTSEVRY", 1, "increment for writing solution", &sstidyn);
  Core::Utils::int_parameter("ITEMAX", 10, "maximum number of iterations over fields", &sstidyn);
  Core::Utils::bool_parameter("SCATRA_FROM_RESTART_FILE", "No",
      "read scatra result from restart files (use option 'restartfromfile' during execution of "
      "4C)",
      &sstidyn);
  Core::Utils::string_parameter(
      "SCATRA_FILENAME", "nil", "Control-file name for reading scatra results in SSTI", &sstidyn);
  setStringToIntegralParameter<SolutionScheme>("COUPALGO", "ssti_Monolithic",
      "Coupling strategies for SSTI solvers", tuple<std::string>("ssti_Monolithic"),
      tuple<Inpar::SSTI::SolutionScheme>(SolutionScheme::monolithic), &sstidyn);
  setStringToIntegralParameter<ScaTraTimIntType>("SCATRATIMINTTYPE", "Elch",
      "scalar transport time integration type is needed to instantiate correct scalar transport "
      "time integration scheme for ssi problems",
      tuple<std::string>("Elch"), tuple<ScaTraTimIntType>(ScaTraTimIntType::elch), &sstidyn);
  Core::Utils::bool_parameter(
      "ADAPTIVE_TIMESTEPPING", "no", "flag for adaptive time stepping", &sstidyn);

  /*----------------------------------------------------------------------*/
  /* parameters for monolithic SSTI                                       */
  /*----------------------------------------------------------------------*/
  Teuchos::ParameterList& sstidynmono = sstidyn.sublist("MONOLITHIC", false,
      "Monolithic Structure Scalar Interaction\n"
      "Control section for monolithic SSI");
  Core::Utils::double_parameter("ABSTOLRES", 1.0e-14,
      "absolute tolerance for deciding if global residual of nonlinear problem is already zero",
      &sstidynmono);
  Core::Utils::double_parameter("CONVTOL", 1.0e-6,
      "tolerance for convergence check of Newton-Raphson iteration within monolithic SSI",
      &sstidynmono);
  Core::Utils::int_parameter(
      "LINEAR_SOLVER", -1, "ID of linear solver for global system of equations", &sstidynmono);
  setStringToIntegralParameter<Core::LinAlg::MatrixType>("MATRIXTYPE", "undefined",
      "type of global system matrix in global system of equations",
      tuple<std::string>("undefined", "block", "sparse"),
      tuple<Core::LinAlg::MatrixType>(Core::LinAlg::MatrixType::undefined,
          Core::LinAlg::MatrixType::block_field, Core::LinAlg::MatrixType::sparse),
      &sstidynmono);
  setStringToIntegralParameter<Core::LinAlg::EquilibrationMethod>("EQUILIBRATION", "none",
      "flag for equilibration of global system of equations",
      tuple<std::string>("none", "rows_full", "rows_maindiag", "rowsandcolumns_full",
          "rowsandcolumns_maindiag", "local"),
      tuple<Core::LinAlg::EquilibrationMethod>(Core::LinAlg::EquilibrationMethod::none,
          Core::LinAlg::EquilibrationMethod::rows_full,
          Core::LinAlg::EquilibrationMethod::rows_maindiag,
          Core::LinAlg::EquilibrationMethod::rowsandcolumns_full,
          Core::LinAlg::EquilibrationMethod::rowsandcolumns_maindiag,
          Core::LinAlg::EquilibrationMethod::local),
      &sstidynmono);
  setStringToIntegralParameter<Core::LinAlg::EquilibrationMethod>("EQUILIBRATION_STRUCTURE", "none",
      "flag for equilibration of structural equations",
      tuple<std::string>(
          "none", "rows_maindiag", "columns_maindiag", "rowsandcolumns_maindiag", "symmetry"),
      tuple<Core::LinAlg::EquilibrationMethod>(Core::LinAlg::EquilibrationMethod::none,
          Core::LinAlg::EquilibrationMethod::rows_maindiag,
          Core::LinAlg::EquilibrationMethod::columns_maindiag,
          Core::LinAlg::EquilibrationMethod::rowsandcolumns_maindiag,
          Core::LinAlg::EquilibrationMethod::symmetry),
      &sstidynmono);
  setStringToIntegralParameter<Core::LinAlg::EquilibrationMethod>("EQUILIBRATION_SCATRA", "none",
      "flag for equilibration of scatra equations",
      tuple<std::string>(
          "none", "rows_maindiag", "columns_maindiag", "rowsandcolumns_maindiag", "symmetry"),
      tuple<Core::LinAlg::EquilibrationMethod>(Core::LinAlg::EquilibrationMethod::none,
          Core::LinAlg::EquilibrationMethod::rows_maindiag,
          Core::LinAlg::EquilibrationMethod::columns_maindiag,
          Core::LinAlg::EquilibrationMethod::rowsandcolumns_maindiag,
          Core::LinAlg::EquilibrationMethod::symmetry),
      &sstidynmono);
  setStringToIntegralParameter<Core::LinAlg::EquilibrationMethod>("EQUILIBRATION_THERMO", "none",
      "flag for equilibration of scatra equations",
      tuple<std::string>(
          "none", "rows_maindiag", "columns_maindiag", "rowsandcolumns_maindiag", "symmetry"),
      tuple<Core::LinAlg::EquilibrationMethod>(Core::LinAlg::EquilibrationMethod::none,
          Core::LinAlg::EquilibrationMethod::rows_maindiag,
          Core::LinAlg::EquilibrationMethod::columns_maindiag,
          Core::LinAlg::EquilibrationMethod::rowsandcolumns_maindiag,
          Core::LinAlg::EquilibrationMethod::symmetry),
      &sstidynmono);
  Core::Utils::bool_parameter("EQUILIBRATION_INIT_SCATRA", "no",
      "use equilibration method of ScaTra to equilibrate initial calculation of potential",
      &sstidynmono);

  /*----------------------------------------------------------------------*/
  /* parameters for thermo                                                */
  /*----------------------------------------------------------------------*/
  Teuchos::ParameterList& thermodyn =
      sstidyn.sublist("THERMO", false, "Parameters for thermo subproblem");
  Core::Utils::int_parameter(
      "INITTHERMOFUNCT", -1, "initial function for thermo field", &thermodyn);
  Core::Utils::int_parameter("LINEAR_SOLVER", -1, "linear solver for thermo field", &thermodyn);
  setStringToIntegralParameter<Inpar::ScaTra::InitialField>("INITIALFIELD", "field_by_function",
      "defines, how to set the initial field",
      tuple<std::string>("field_by_function", "field_by_condition"),
      tuple<Inpar::ScaTra::InitialField>(Inpar::ScaTra::InitialField::initfield_field_by_function,
          Inpar::ScaTra::InitialField::initfield_field_by_condition),
      &thermodyn);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Inpar::SSTI::set_valid_conditions(
    std::vector<Teuchos::RCP<Core::Conditions::ConditionDefinition>>& condlist)
{
  using namespace Input;

  /*--------------------------------------------------------------------*/
  // set Scalar-Structure-Thermo interaction interface meshtying condition
  Teuchos::RCP<Core::Conditions::ConditionDefinition> linesstiinterfacemeshtying =
      Teuchos::make_rcp<Core::Conditions::ConditionDefinition>(
          "DESIGN SSTI INTERFACE MESHTYING LINE CONDITIONS", "SSTIInterfaceMeshtying",
          "SSTI Interface Meshtying", Core::Conditions::SSTIInterfaceMeshtying, true,
          Core::Conditions::geometry_type_line);
  Teuchos::RCP<Core::Conditions::ConditionDefinition> surfsstiinterfacemeshtying =
      Teuchos::make_rcp<Core::Conditions::ConditionDefinition>(
          "DESIGN SSTI INTERFACE MESHTYING SURF CONDITIONS", "SSTIInterfaceMeshtying",
          "SSTI Interface Meshtying", Core::Conditions::SSTIInterfaceMeshtying, true,
          Core::Conditions::geometry_type_surface);

  // insert input file line components into condition definitions
  for (const auto& cond : {linesstiinterfacemeshtying, surfsstiinterfacemeshtying})
  {
    cond->add_component(Teuchos::make_rcp<Input::IntComponent>("ConditionID"));
    cond->add_component(Teuchos::make_rcp<Input::SelectionComponent>("interface side", "Undefined",
        Teuchos::tuple<std::string>("Undefined", "Slave", "Master"),
        Teuchos::tuple<int>(
            Inpar::S2I::side_undefined, Inpar::S2I::side_slave, Inpar::S2I::side_master)));
    add_named_int(cond, "S2I_KINETICS_ID");
    condlist.push_back(cond);
  }
}

FOUR_C_NAMESPACE_CLOSE

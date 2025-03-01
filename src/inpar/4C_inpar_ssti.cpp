// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_ssti.hpp"

#include "4C_fem_condition_definition.hpp"
#include "4C_inpar_s2i.hpp"
#include "4C_inpar_scatra.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_linalg_equilibrate.hpp"
#include "4C_linalg_sparseoperator.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN

void Inpar::SSTI::set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list)
{
  using Teuchos::tuple;
  using namespace Core::IO::InputSpecBuilders;

  Core::Utils::SectionSpecs sstidyn{"SSTI CONTROL"};

  sstidyn.specs.emplace_back(parameter<double>("RESTARTEVERYTIME",
      {.description = "write restart possibility every RESTARTEVERY steps", .default_value = 0.0}));
  sstidyn.specs.emplace_back(parameter<int>("RESTARTEVERY",
      {.description = "write restart possibility every RESTARTEVERY steps", .default_value = 1}));
  sstidyn.specs.emplace_back(parameter<int>(
      "NUMSTEP", {.description = "maximum number of Timesteps", .default_value = 200}));

  sstidyn.specs.emplace_back(parameter<double>(
      "MAXTIME", {.description = "total simulation time", .default_value = 1000.0}));
  sstidyn.specs.emplace_back(
      parameter<double>("TIMESTEP", {.description = "time step size dt", .default_value = -1.0}));
  sstidyn.specs.emplace_back(parameter<double>(
      "RESULTSEVERYTIME", {.description = "increment for writing solution", .default_value = 0.0}));
  sstidyn.specs.emplace_back(parameter<int>(
      "RESULTSEVERY", {.description = "increment for writing solution", .default_value = 1}));
  sstidyn.specs.emplace_back(parameter<int>(
      "ITEMAX", {.description = "maximum number of iterations over fields", .default_value = 10}));
  sstidyn.specs.emplace_back(parameter<bool>("SCATRA_FROM_RESTART_FILE",
      {.description = "read scatra result from restart files (use option 'restartfromfile' during "
                      "execution of 4C)",
          .default_value = false}));
  sstidyn.specs.emplace_back(parameter<std::string>(
      "SCATRA_FILENAME", {.description = "Control-file name for reading scatra results in SSTI",
                             .default_value = "nil"}));
  Core::Utils::string_to_integral_parameter<SolutionScheme>("COUPALGO", "ssti_Monolithic",
      "Coupling strategies for SSTI solvers", tuple<std::string>("ssti_Monolithic"),
      tuple<Inpar::SSTI::SolutionScheme>(SolutionScheme::monolithic), sstidyn);
  Core::Utils::string_to_integral_parameter<ScaTraTimIntType>("SCATRATIMINTTYPE", "Elch",
      "scalar transport time integration type is needed to instantiate correct scalar transport "
      "time integration scheme for ssi problems",
      tuple<std::string>("Elch"), tuple<ScaTraTimIntType>(ScaTraTimIntType::elch), sstidyn);
  sstidyn.specs.emplace_back(parameter<bool>("ADAPTIVE_TIMESTEPPING",
      {.description = "flag for adaptive time stepping", .default_value = false}));

  sstidyn.move_into_collection(list);

  /*----------------------------------------------------------------------*/
  /* parameters for monolithic SSTI                                       */
  /*----------------------------------------------------------------------*/
  Core::Utils::SectionSpecs sstidynmono{sstidyn, "MONOLITHIC"};
  sstidynmono.specs.emplace_back(parameter<double>(
      "ABSTOLRES", {.description = "absolute tolerance for deciding if global residual of "
                                   "nonlinear problem is already zero",
                       .default_value = 1.0e-14}));
  sstidynmono.specs.emplace_back(parameter<double>("CONVTOL",
      {.description =
              "tolerance for convergence check of Newton-Raphson iteration within monolithic SSI",
          .default_value = 1.0e-6}));
  sstidynmono.specs.emplace_back(parameter<int>("LINEAR_SOLVER",
      {.description = "ID of linear solver for global system of equations", .default_value = -1}));
  Core::Utils::string_to_integral_parameter<Core::LinAlg::MatrixType>("MATRIXTYPE", "undefined",
      "type of global system matrix in global system of equations",
      tuple<std::string>("undefined", "block", "sparse"),
      tuple<Core::LinAlg::MatrixType>(Core::LinAlg::MatrixType::undefined,
          Core::LinAlg::MatrixType::block_field, Core::LinAlg::MatrixType::sparse),
      sstidynmono);
  Core::Utils::string_to_integral_parameter<Core::LinAlg::EquilibrationMethod>("EQUILIBRATION",
      "none", "flag for equilibration of global system of equations",
      tuple<std::string>("none", "rows_full", "rows_maindiag", "rowsandcolumns_full",
          "rowsandcolumns_maindiag", "local"),
      tuple<Core::LinAlg::EquilibrationMethod>(Core::LinAlg::EquilibrationMethod::none,
          Core::LinAlg::EquilibrationMethod::rows_full,
          Core::LinAlg::EquilibrationMethod::rows_maindiag,
          Core::LinAlg::EquilibrationMethod::rowsandcolumns_full,
          Core::LinAlg::EquilibrationMethod::rowsandcolumns_maindiag,
          Core::LinAlg::EquilibrationMethod::local),
      sstidynmono);
  Core::Utils::string_to_integral_parameter<Core::LinAlg::EquilibrationMethod>(
      "EQUILIBRATION_STRUCTURE", "none", "flag for equilibration of structural equations",
      tuple<std::string>(
          "none", "rows_maindiag", "columns_maindiag", "rowsandcolumns_maindiag", "symmetry"),
      tuple<Core::LinAlg::EquilibrationMethod>(Core::LinAlg::EquilibrationMethod::none,
          Core::LinAlg::EquilibrationMethod::rows_maindiag,
          Core::LinAlg::EquilibrationMethod::columns_maindiag,
          Core::LinAlg::EquilibrationMethod::rowsandcolumns_maindiag,
          Core::LinAlg::EquilibrationMethod::symmetry),
      sstidynmono);
  Core::Utils::string_to_integral_parameter<Core::LinAlg::EquilibrationMethod>(
      "EQUILIBRATION_SCATRA", "none", "flag for equilibration of scatra equations",
      tuple<std::string>(
          "none", "rows_maindiag", "columns_maindiag", "rowsandcolumns_maindiag", "symmetry"),
      tuple<Core::LinAlg::EquilibrationMethod>(Core::LinAlg::EquilibrationMethod::none,
          Core::LinAlg::EquilibrationMethod::rows_maindiag,
          Core::LinAlg::EquilibrationMethod::columns_maindiag,
          Core::LinAlg::EquilibrationMethod::rowsandcolumns_maindiag,
          Core::LinAlg::EquilibrationMethod::symmetry),
      sstidynmono);
  Core::Utils::string_to_integral_parameter<Core::LinAlg::EquilibrationMethod>(
      "EQUILIBRATION_THERMO", "none", "flag for equilibration of scatra equations",
      tuple<std::string>(
          "none", "rows_maindiag", "columns_maindiag", "rowsandcolumns_maindiag", "symmetry"),
      tuple<Core::LinAlg::EquilibrationMethod>(Core::LinAlg::EquilibrationMethod::none,
          Core::LinAlg::EquilibrationMethod::rows_maindiag,
          Core::LinAlg::EquilibrationMethod::columns_maindiag,
          Core::LinAlg::EquilibrationMethod::rowsandcolumns_maindiag,
          Core::LinAlg::EquilibrationMethod::symmetry),
      sstidynmono);
  sstidynmono.specs.emplace_back(parameter<bool>("EQUILIBRATION_INIT_SCATRA",
      {.description =
              "use equilibration method of ScaTra to equilibrate initial calculation of potential",
          .default_value = false}));

  sstidynmono.move_into_collection(list);

  /*----------------------------------------------------------------------*/
  /* parameters for thermo                                                */
  /*----------------------------------------------------------------------*/
  Core::Utils::SectionSpecs thermodyn{sstidyn, "THERMO"};
  thermodyn.specs.emplace_back(parameter<int>("INITTHERMOFUNCT",
      {.description = "initial function for thermo field", .default_value = -1}));
  thermodyn.specs.emplace_back(parameter<int>(
      "LINEAR_SOLVER", {.description = "linear solver for thermo field", .default_value = -1}));
  Core::Utils::string_to_integral_parameter<Inpar::ScaTra::InitialField>("INITIALFIELD",
      "field_by_function", "defines, how to set the initial field",
      tuple<std::string>("field_by_function", "field_by_condition"),
      tuple<Inpar::ScaTra::InitialField>(Inpar::ScaTra::InitialField::initfield_field_by_function,
          Inpar::ScaTra::InitialField::initfield_field_by_condition),
      thermodyn);

  thermodyn.move_into_collection(list);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Inpar::SSTI::set_valid_conditions(std::vector<Core::Conditions::ConditionDefinition>& condlist)
{
  using namespace Core::IO::InputSpecBuilders;

  /*--------------------------------------------------------------------*/
  // set Scalar-Structure-Thermo interaction interface meshtying condition
  Core::Conditions::ConditionDefinition linesstiinterfacemeshtying(
      "DESIGN SSTI INTERFACE MESHTYING LINE CONDITIONS", "SSTIInterfaceMeshtying",
      "SSTI Interface Meshtying", Core::Conditions::SSTIInterfaceMeshtying, true,
      Core::Conditions::geometry_type_line);
  Core::Conditions::ConditionDefinition surfsstiinterfacemeshtying(
      "DESIGN SSTI INTERFACE MESHTYING SURF CONDITIONS", "SSTIInterfaceMeshtying",
      "SSTI Interface Meshtying", Core::Conditions::SSTIInterfaceMeshtying, true,
      Core::Conditions::geometry_type_surface);

  const auto make_sstiinterfacemeshtying = [&condlist](Core::Conditions::ConditionDefinition& cond)
  {
    cond.add_component(parameter<int>("ConditionID"));
    cond.add_component(deprecated_selection<S2I::InterfaceSides>("INTERFACE_SIDE",
        {{"Undefined", Inpar::S2I::side_undefined}, {"Slave", Inpar::S2I::side_slave},
            {"Master", Inpar::S2I::side_master}},
        {.description = "interface side"}));
    cond.add_component(parameter<int>("S2I_KINETICS_ID"));
    condlist.push_back(cond);
  };

  make_sstiinterfacemeshtying(linesstiinterfacemeshtying);
  make_sstiinterfacemeshtying(surfsstiinterfacemeshtying);
}

FOUR_C_NAMESPACE_CLOSE

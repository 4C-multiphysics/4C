// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_ssti_input.hpp"

#include "4C_fem_condition_definition.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_linalg_equilibrate.hpp"
#include "4C_linalg_sparseoperator.hpp"
#include "4C_scatra_input.hpp"
#include "4C_scatra_s2i_input.hpp"

FOUR_C_NAMESPACE_OPEN

std::vector<Core::IO::InputSpec> SSTI::valid_parameters()
{
  using namespace Core::IO::InputSpecBuilders;

  std::vector<Core::IO::InputSpec> specs;
  specs.push_back(group("SSTI CONTROL",
      {

          parameter<double>("RESTARTEVERYTIME",
              {.description = "write restart possibility every RESTARTEVERY steps",
                  .default_value = 0.0}),
          parameter<int>(
              "RESTARTEVERY", {.description = "write restart possibility every RESTARTEVERY steps",
                                  .default_value = 1}),
          parameter<int>(
              "NUMSTEP", {.description = "maximum number of Timesteps", .default_value = 200}),

          parameter<double>(
              "MAXTIME", {.description = "total simulation time", .default_value = 1000.0}),

          parameter<double>(
              "TIMESTEP", {.description = "time step size dt", .default_value = -1.0}),
          parameter<double>("RESULTSEVERYTIME",
              {.description = "increment for writing solution", .default_value = 0.0}),
          parameter<int>("RESULTSEVERY",
              {.description = "increment for writing solution", .default_value = 1}),
          parameter<int>("ITEMAX",
              {.description = "maximum number of iterations over fields", .default_value = 10}),
          parameter<bool>("SCATRA_FROM_RESTART_FILE",
              {.description =
                      "read scatra result from restart files (use option 'restartfromfile' during "
                      "execution of 4C)",
                  .default_value = false}),
          parameter<std::string>("SCATRA_FILENAME",
              {.description = "Control-file name for reading scatra results in SSTI",
                  .default_value = "nil"}),
          deprecated_selection<SolutionScheme>("COUPALGO",
              {
                  {"ssti_Monolithic", SolutionScheme::monolithic},
              },
              {.description = "Coupling strategies for SSTI solvers",
                  .default_value = SolutionScheme::monolithic}),
          deprecated_selection<ScaTraTimIntType>("SCATRATIMINTTYPE",
              {
                  {"Elch", ScaTraTimIntType::elch},
              },
              {.description =
                      "scalar transport time integration type is needed to instantiate correct "
                      "scalar transport time integration scheme for ssi problems",
                  .default_value = ScaTraTimIntType::elch}),
          parameter<bool>("ADAPTIVE_TIMESTEPPING",
              {.description = "flag for adaptive time stepping", .default_value = false})},
      {.required = false}));
  /*----------------------------------------------------------------------*/
  /* parameters for monolithic SSTI                                       */
  /*----------------------------------------------------------------------*/
  specs.push_back(group("SSTI CONTROL/MONOLITHIC",
      {

          parameter<double>(
              "ABSTOLRES", {.description = "absolute tolerance for deciding if global residual of "
                                           "nonlinear problem is already zero",
                               .default_value = 1.0e-14}),
          parameter<double>(
              "CONVTOL", {.description = "tolerance for convergence check of "
                                         "Newton-Raphson iteration within monolithic SSI",
                             .default_value = 1.0e-6}),
          parameter<int>(
              "LINEAR_SOLVER", {.description = "ID of linear solver for global system of equations",
                                   .default_value = -1}),
          parameter<Core::LinAlg::MatrixType>("MATRIXTYPE",
              {.description = "type of global system matrix in global system of equations"}),
          parameter<Core::LinAlg::EquilibrationMethod>("EQUILIBRATION",
              {.description = "flag for equilibration of global system of equations",
                  .default_value = Core::LinAlg::EquilibrationMethod::none}),
          parameter<Core::LinAlg::EquilibrationMethod>("EQUILIBRATION_STRUCTURE",
              {.description = "flag for equilibration of structural equations",
                  .default_value = Core::LinAlg::EquilibrationMethod::none}),
          parameter<Core::LinAlg::EquilibrationMethod>("EQUILIBRATION_SCATRA",
              {.description = "flag for equilibration of scatra equations",
                  .default_value = Core::LinAlg::EquilibrationMethod::none}),
          parameter<Core::LinAlg::EquilibrationMethod>("EQUILIBRATION_THERMO",
              {.description = "flag for equilibration of scatra equations",
                  .default_value = Core::LinAlg::EquilibrationMethod::none}),
          parameter<bool>("EQUILIBRATION_INIT_SCATRA",
              {.description =
                      "use equilibration method of ScaTra to equilibrate initial calculation "
                      "of potential",
                  .default_value = false})},
      {.required = false}));

  /*----------------------------------------------------------------------*/
  /* parameters for thermo                                                */
  /*----------------------------------------------------------------------*/
  specs.push_back(group("SSTI CONTROL/THERMO",
      {

          parameter<int>("INITTHERMOFUNCT",
              {.description = "initial function for thermo field", .default_value = -1}),
          parameter<int>("LINEAR_SOLVER",
              {.description = "linear solver for thermo field", .default_value = -1}),
          deprecated_selection<ScaTra::InitialField>("INITIALFIELD",
              {
                  {"field_by_function", ScaTra::InitialField::initfield_field_by_function},
                  {"field_by_condition", ScaTra::InitialField::initfield_field_by_condition},
              },
              {.description = "defines, how to set the initial field",
                  .default_value = ScaTra::InitialField::initfield_field_by_function})},
      {.required = false}));
  return specs;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSTI::set_valid_conditions(std::vector<Core::Conditions::ConditionDefinition>& condlist)
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
        {{"Undefined", S2I::side_undefined}, {"Slave", S2I::side_source},
            {"Master", S2I::side_master}},
        {.description = "interface side"}));
    cond.add_component(parameter<int>("S2I_KINETICS_ID"));
    condlist.push_back(cond);
  };

  make_sstiinterfacemeshtying(linesstiinterfacemeshtying);
  make_sstiinterfacemeshtying(surfsstiinterfacemeshtying);
}

FOUR_C_NAMESPACE_CLOSE
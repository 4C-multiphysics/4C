// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_elemag.hpp"

#include "4C_fem_condition_definition.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_linalg_equilibrate.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN

void Inpar::EleMag::set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list)
{
  using Teuchos::tuple;
  using namespace Core::IO::InputSpecBuilders;

  Core::Utils::SectionSpecs electromagneticdyn{"ELECTROMAGNETIC DYNAMIC"};

  // general settings for time-integration scheme
  electromagneticdyn.specs.emplace_back(
      parameter<double>("TIMESTEP", {.description = "Time-step length dt", .default_value = 0.01}));
  electromagneticdyn.specs.emplace_back(
      parameter<double>("TAU", {.description = "Stabilization parameter", .default_value = 1.0}));
  Core::Utils::int_parameter("NUMSTEP", 100, "Number of time steps", electromagneticdyn);
  electromagneticdyn.specs.emplace_back(
      parameter<double>("MAXTIME", {.description = "Total simulation time", .default_value = 1.0}));

  // additional parameters
  Core::Utils::int_parameter(
      "RESULTSEVERY", 1, "Increment for writing solution", electromagneticdyn);
  Core::Utils::int_parameter(
      "RESTARTEVERY", 1, "Increment for writing restart", electromagneticdyn);
  Core::Utils::int_parameter("LINEAR_SOLVER", -1,
      "Number of linear solver used for electromagnetic problem", electromagneticdyn);
  Core::Utils::int_parameter("STARTFUNCNO", -1, "Function for initial field", electromagneticdyn);
  Core::Utils::int_parameter(
      "SOURCEFUNCNO", -1, "Function for source term in volume", electromagneticdyn);

  {
    // time integration

    Teuchos::Tuple<std::string, 8> name;
    Teuchos::Tuple<Inpar::EleMag::DynamicType, 8> label;
    name[0] = "One_Step_Theta";
    label[0] = elemag_ost;
    name[1] = "BDF1";
    label[1] = elemag_bdf1;
    name[2] = "BDF2";
    label[2] = elemag_bdf2;
    name[3] = "BDF4";
    label[3] = elemag_bdf4;
    name[4] = "GenAlpha";
    label[4] = elemag_genAlpha;
    name[5] = "Explicit_Euler";
    label[5] = elemag_explicit_euler;
    name[6] = "Runge_Kutta";
    label[6] = elemag_rk;
    name[7] = "Crank_Nicolson";
    label[7] = elemag_cn;

    Core::Utils::string_to_integral_parameter<Inpar::EleMag::DynamicType>("TIMEINT",
        "One_Step_Theta", "Type of time integration scheme", name, label, electromagneticdyn);
  }

  {
    // a standard Teuchos::tuple can have at maximum 10 entries! We have to circumvent this here.
    Teuchos::Tuple<std::string, 4> name;
    Teuchos::Tuple<Inpar::EleMag::InitialField, 4> label;
    name[0] = "zero_field";
    label[0] = initfield_zero_field;
    name[1] = "field_by_function";
    label[1] = initfield_field_by_function;
    name[2] = "field_by_steady_state";
    label[2] = initfield_scatra;
    name[3] = "field_by_steady_state_hdg";
    label[3] = initfield_scatra_hdg;

    Core::Utils::string_to_integral_parameter<Inpar::EleMag::InitialField>("INITIALFIELD",
        "zero_field", "Initial field for ele problem", name, label, electromagneticdyn);

    // Error calculation
    electromagneticdyn.specs.emplace_back(parameter<bool>(
        "CALCERR", {.description = "Calc the error wrt ERRORFUNCNO?", .default_value = false}));

    // Post process solution?
    electromagneticdyn.specs.emplace_back(parameter<bool>("POSTPROCESS",
        {.description = "Postprocess solution? (very slow)", .default_value = false}));
  }

  Core::Utils::int_parameter(
      "ERRORFUNCNO", -1, "Function for error calculation", electromagneticdyn);

  // flag for equilibration of global system of equations
  Core::Utils::string_to_integral_parameter<Core::LinAlg::EquilibrationMethod>("EQUILIBRATION",
      "none", "flag for equilibration of global system of equations",
      tuple<std::string>("none", "rows_full", "rows_maindiag", "columns_full", "columns_maindiag",
          "rowsandcolumns_full", "rowsandcolumns_maindiag"),
      tuple<Core::LinAlg::EquilibrationMethod>(Core::LinAlg::EquilibrationMethod::none,
          Core::LinAlg::EquilibrationMethod::rows_full,
          Core::LinAlg::EquilibrationMethod::rows_maindiag,
          Core::LinAlg::EquilibrationMethod::columns_full,
          Core::LinAlg::EquilibrationMethod::columns_maindiag,
          Core::LinAlg::EquilibrationMethod::rowsandcolumns_full,
          Core::LinAlg::EquilibrationMethod::rowsandcolumns_maindiag),
      electromagneticdyn);

  electromagneticdyn.move_into_collection(list);
}

/// set specific electromagnetic conditions
void Inpar::EleMag::set_valid_conditions(
    std::vector<Core::Conditions::ConditionDefinition>& condlist)
{
  using namespace Core::IO::InputSpecBuilders;

  //*--------------------------------------------------------------------* /
  // absorbing boundary condition for electromagnetic problems
  // line
  Core::Conditions::ConditionDefinition silvermueller_line("DESIGN LINE SILVER-MUELLER CONDITIONS",
      "Silver-Mueller", "Absorbing-emitting line for electromagnetics",
      Core::Conditions::SilverMueller, true, Core::Conditions::geometry_type_line);

  // surface
  Core::Conditions::ConditionDefinition silvermueller_surface(
      "DESIGN SURF SILVER-MUELLER CONDITIONS", "Silver-Mueller",
      "Absorbing-emitting surface for electromagnetics", Core::Conditions::SilverMueller, true,
      Core::Conditions::geometry_type_surface);

  const auto make_silvermueller = [&condlist](Core::Conditions::ConditionDefinition& cond)
  {
    cond.add_component(parameter<int>("NUMDOF"));
    cond.add_component(parameter<std::vector<int>>(
        "ONOFF", {.description = "", .size = from_parameter<int>("NUMDOF")}));
    cond.add_component(parameter<std::vector<std::optional<int>>>(
        "FUNCT", {.description = "", .size = from_parameter<int>("NUMDOF")}));
    cond.add_component(parameter<std::vector<double>>(
        "VAL", {.description = "", .size = from_parameter<int>("NUMDOF")}));

    condlist.push_back(cond);
  };

  make_silvermueller(silvermueller_line);
  make_silvermueller(silvermueller_surface);
}

FOUR_C_NAMESPACE_CLOSE

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_elemag.hpp"

#include "4C_fem_condition_definition.hpp"
#include "4C_linalg_equilibrate.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN

void Inpar::EleMag::set_valid_parameters(Teuchos::ParameterList& list)
{
  using namespace Input;
  using Teuchos::setStringToIntegralParameter;
  using Teuchos::tuple;

  Teuchos::ParameterList& electromagneticdyn = list.sublist(
      "ELECTROMAGNETIC DYNAMIC", false, "control parameters for electromagnetic problems\n");

  // general settings for time-integration scheme
  Core::Utils::double_parameter("TIMESTEP", 0.01, "Time-step length dt", &electromagneticdyn);
  Core::Utils::double_parameter("TAU", 1, "Stabilization parameter", &electromagneticdyn);
  Core::Utils::int_parameter("NUMSTEP", 100, "Number of time steps", &electromagneticdyn);
  Core::Utils::double_parameter("MAXTIME", 1.0, "Total simulation time", &electromagneticdyn);

  // additional parameters
  Core::Utils::int_parameter(
      "RESULTSEVERY", 1, "Increment for writing solution", &electromagneticdyn);
  Core::Utils::int_parameter(
      "RESTARTEVERY", 1, "Increment for writing restart", &electromagneticdyn);
  Core::Utils::int_parameter("LINEAR_SOLVER", -1,
      "Number of linear solver used for electromagnetic problem", &electromagneticdyn);
  Core::Utils::int_parameter("STARTFUNCNO", -1, "Function for initial field", &electromagneticdyn);
  Core::Utils::int_parameter(
      "SOURCEFUNCNO", -1, "Function for source term in volume", &electromagneticdyn);

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

    setStringToIntegralParameter<Inpar::EleMag::DynamicType>("TIMEINT", "One_Step_Theta",
        "Type of time integration scheme", name, label, &electromagneticdyn);
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

    setStringToIntegralParameter<Inpar::EleMag::InitialField>("INITIALFIELD", "zero_field",
        "Initial field for ele problem", name, label, &electromagneticdyn);

    // Error calculation
    Core::Utils::bool_parameter(
        "CALCERR", "No", "Calc the error wrt ERRORFUNCNO?", &electromagneticdyn);

    // Post process solution?
    Core::Utils::bool_parameter(
        "POSTPROCESS", "No", "Postprocess solution? (very slow)", &electromagneticdyn);
  }

  Core::Utils::int_parameter(
      "ERRORFUNCNO", -1, "Function for error calculation", &electromagneticdyn);

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
      &electromagneticdyn);
}

/// set specific electromagnetic conditions
void Inpar::EleMag::set_valid_conditions(
    std::vector<std::shared_ptr<Core::Conditions::ConditionDefinition>>& condlist)
{
  using namespace Input;

  //*--------------------------------------------------------------------* /
  // absorbing boundary condition for electromagnetic problems
  // line
  std::shared_ptr<Core::Conditions::ConditionDefinition> silvermueller_line =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN LINE SILVER-MUELLER CONDITIONS", "Silver-Mueller",
          "Absorbing-emitting line for electromagnetics", Core::Conditions::SilverMueller, true,
          Core::Conditions::geometry_type_line);

  // surface
  std::shared_ptr<Core::Conditions::ConditionDefinition> silvermueller_surface =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN SURF SILVER-MUELLER CONDITIONS", "Silver-Mueller",
          "Absorbing-emitting surface for electromagnetics", Core::Conditions::SilverMueller, true,
          Core::Conditions::geometry_type_surface);

  for (const auto& cond : {silvermueller_line, silvermueller_surface})
  {
    add_named_int(cond, "NUMDOF");
    add_named_int_vector(cond, "ONOFF", "", "NUMDOF");
    add_named_int_vector(cond, "FUNCT", "", "NUMDOF", 0, true);
    add_named_real_vector(cond, "VAL", "", "NUMDOF");

    condlist.push_back(cond);
  }
}

FOUR_C_NAMESPACE_CLOSE

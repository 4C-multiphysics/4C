// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_porofluid_pressure_based_elast_scatra_input.hpp"

#include "4C_fem_condition_definition.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_linalg_equilibrate.hpp"
#include "4C_porofluid_pressure_based_input.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN


void PoroPressureBased::set_valid_parameters_porofluid_elast_scatra(
    std::map<std::string, Core::IO::InputSpec>& list)
{
  using namespace Core::IO::InputSpecBuilders;

  // general control parameters
  list["porofluid_elasticity_scatra_dynamic"] = group("porofluid_elasticity_scatra_dynamic",
      {
          parameter<double>("total_simulation_time",
              {.description = "total simulation time", .default_value = -1.0}),

          // output
          group("output",
              {
                  parameter<int>("results_every",
                      {.description = "increment for writing solution", .default_value = 1}),
                  parameter<int>("restart_data_every",
                      {.description = "write restart data every nth steps", .default_value = 1}),
              },
              {.required = false}),

          // time integration
          group("time_integration",
              {
                  parameter<int>("number_of_time_steps",
                      {.description = "maximum number of time steps", .default_value = -1}),
                  parameter<double>(
                      "theta", {.description = "One-step-theta time integration factor",
                                   .default_value = -1.0}),
                  parameter<double>(
                      "time_step", {.description = "time step size dt", .default_value = 0.5}),
              },
              {.required = false}),

          // nonlinear solver
          group("nonlinear_solver",
              {
                  parameter<int>("maximum_number_of_iterations",
                      {.description = "maximum number of iterations over fields",
                          .default_value = 10}),
                  parameter<int>("minimum_number_of_iterations",
                      {.description = "minimum number of iterations over fields",
                          .default_value = 1}),
                  parameter<int>("linear_solver_id",
                      {.description = "ID of linear solver", .default_value = 1}),
              },
              {.required = false}),

          // coupling scheme for porofluid-elasticity to scalar transport coupling
          parameter<SolutionSchemePorofluidElastScatra>("coupling_scheme",
              {.description =
                      "Coupling scheme for porofluid-elasticity to scalar transport coupling",
                  .default_value =
                      SolutionSchemePorofluidElastScatra::twoway_partitioned_sequential}),

          // coupling with 1D artery network active
          parameter<bool>("artery_coupling_active",
              {.description = "Coupling with 1D blood vessels.", .default_value = false}),

          // what to do when the nonlinear solver does not converge
          parameter<DivergenceAction>("divergence_action",
              {.description =
                      "What to do with time integration when the nonlinear solver did not converge",
                  .default_value = DivergenceAction::stop}),
      },
      {.required = false});

  // monolithic parameters
  list["porofluid_elasticity_scatra_dynamic/monolithic"] = group(
      "porofluid_elasticity_scatra_dynamic/monolithic",
      {
          // nonlinear solver
          group("nonlinear_solver",
              {
                  parameter<int>(
                      "linear_solver_id", {.description = "number of linear solver used"}),
                  // flag for equilibration of global system of equations
                  parameter<Core::LinAlg::EquilibrationMethod>("equilibration",
                      {.description = "flag for equilibration of global system of equations",
                          .default_value = Core::LinAlg::EquilibrationMethod::none}),

                  group("residual",
                      {
                          parameter<double>("global_tolerance",
                              {.description = "Tolerance for residual norm of the nonlinear solver",
                                  .default_value = 1e-8}),
                          parameter<VectorNorm>("vector_norm",
                              {.description = "type of norm to be applied to residuals",
                                  .default_value = VectorNorm::l2}),
                      },
                      {.required = false}),

                  group("increment",
                      {
                          parameter<double>("global_tolerance",
                              {.description =
                                      "Tolerance for increment norm of the nonlinear solver",
                                  .default_value = 1e-8}),
                          parameter<VectorNorm>("vector_norm",
                              {.description = "type of norm to be applied to residuals",
                                  .default_value = VectorNorm::l2}),
                      },
                      {.required = false}),

                  // convergence criteria adaptivity
                  group("convergence_criteria_adaptivity",
                      {
                          parameter<bool>(
                              "active", {.description = "Activate adaptive control of linear "
                                                        "solver tolerance for nonlinear solution",
                                            .default_value = false}),
                          parameter<double>("nonlinear_to_linear_tolerance_ratio",
                              {.description = "The linear solver shall be this much better than "
                                              "the current nonlinear residual in the nonlinear "
                                              "convergence limit",
                                  .default_value = 0.1}),
                      },
                      {.required = false}),
              }),

          // finite difference check
          parameter<bool>("fd_check", {.description = "FD check active", .default_value = false}),
      },
      {.required = false});

  // partitioned parameters
  list["porofluid_elasticity_scatra_dynamic/partitioned"] =
      group("porofluid_elasticity_scatra_dynamic/partitioned",
          {
              parameter<double>("convergence_tolerance",
                  {.description = "tolerance for convergence check of outer iteration",
                      .default_value = 1e-6}),
          },
          {.required = false});
}

void PoroPressureBased::set_valid_conditions_porofluid_elast_scatra(
    std::vector<Core::Conditions::ConditionDefinition>& condlist)
{
  using namespace Core::IO::InputSpecBuilders;

  /*--------------------------------------------------------------------*/
  // oxygen partial pressure calculation condition
  {
    // definition of oxygen partial pressure calculation condition
    Core::Conditions::ConditionDefinition oxypartpressline(
        "DESIGN OXYGEN PARTIAL PRESSURE CALCULATION LINE CONDITIONS",
        "PoroMultiphaseScatraOxyPartPressCalcCond",
        "PoroMultiphaseScatra Oxygen Partial Pressure Calculation line condition",
        Core::Conditions::PoroMultiphaseScatraOxyPartPressCalcCond, true,
        Core::Conditions::geometry_type_line);
    Core::Conditions::ConditionDefinition oxypartpresssurf(
        "DESIGN OXYGEN PARTIAL PRESSURE CALCULATION SURF CONDITIONS",
        "PoroMultiphaseScatraOxyPartPressCalcCond",
        "PoroMultiphaseScatra Oxygen Partial Pressure Calculation surface condition",
        Core::Conditions::PoroMultiphaseScatraOxyPartPressCalcCond, true,
        Core::Conditions::geometry_type_surface);
    Core::Conditions::ConditionDefinition oxypartpressvol(
        "DESIGN OXYGEN PARTIAL PRESSURE CALCULATION VOL CONDITIONS",
        "PoroMultiphaseScatraOxyPartPressCalcCond",
        "PoroMultiphaseScatra Oxygen Partial Pressure Calculation volume condition",
        Core::Conditions::PoroMultiphaseScatraOxyPartPressCalcCond, true,
        Core::Conditions::geometry_type_volume);

    const auto make_oxypartpress = [&condlist](Core::Conditions::ConditionDefinition& cond)
    {
      cond.add_component(
          parameter<int>("SCALARID", {.description = "scalar id of oxygen partial pressure"}));
      cond.add_component(parameter<double>("n", {.description = "n"}));
      cond.add_component(parameter<double>("Pb50", {.description = "Pb50"}));
      cond.add_component(parameter<double>("CaO2_max", {.description = "CaO2_max"}));
      cond.add_component(parameter<double>("alpha_bl_eff", {.description = "alpha_bl_eff"}));
      cond.add_component(parameter<double>("rho_oxy", {.description = "rho_oxy"}));
      cond.add_component(parameter<double>("rho_bl", {.description = "rho_bl"}));

      condlist.push_back(cond);
    };

    make_oxypartpress(oxypartpressline);
    make_oxypartpress(oxypartpresssurf);
    make_oxypartpress(oxypartpressvol);
  }
}

FOUR_C_NAMESPACE_CLOSE
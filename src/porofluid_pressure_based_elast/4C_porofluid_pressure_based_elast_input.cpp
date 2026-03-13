// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "4C_porofluid_pressure_based_elast_input.hpp"

#include "4C_fem_condition_definition.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_linalg_equilibrate.hpp"
#include "4C_porofluid_pressure_based_input.hpp"

FOUR_C_NAMESPACE_OPEN


std::vector<Core::IO::InputSpec> PoroPressureBased::valid_parameters_porofluid_elast()
{
  using namespace Core::IO::InputSpecBuilders;
  std::vector<Core::IO::InputSpec> specs;
  // general control parameters
  specs.push_back(group("porofluid_elasticity_dynamic",
      {
          parameter<double>("total_simulation_time",
              {.description = "total simulation time", .default_value = -1.0}),

          // output
          group("output",
              {
                  parameter<int>("result_data_every",
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
                                   .default_value = 0.5}),
                  parameter<double>("time_step_size",
                      {.description = "time step size dt", .default_value = -1.0}),
              },
              {.required = false}),

          // body force contribution for solid part
          parameter<std::optional<std::vector<double>>>("body_force",
              {.description = "External body force contribution vector. The vector must "
                              "have the same dimension as the problem (2D/3D)."}),

          // nonlinear solver
          group("nonlinear_solver",
              {
                  parameter<int>("maximum_number_of_iterations",
                      {.description = "maximum number of iterations over fields",
                          .default_value = 10}),
                  parameter<int>("linear_solver_id",
                      {.description = "ID of linear solver", .default_value = 1}),
              },
              {.required = false}),

          // solve or not structure
          // this is helpful if only porofluid-scatra coupling should be calculated
          parameter<bool>(
              "solve_structure", {.description = "Whether or not to solve the structure problem",
                                     .default_value = true}),

          // Coupling strategy for solvers
          parameter<SolutionSchemePorofluidElast>("coupling_scheme",
              {.description = "Coupling strategies for porofluid-elasticity solvers",
                  .default_value = SolutionSchemePorofluidElast::twoway_partitioned}),

          // coupling with 1D artery network active
          parameter<bool>("artery_coupling_active",
              {.description = "Coupling with 1D blood vessels.", .default_value = false}),
      },
      {.required = false}));

  // monolithic parameters
  specs.push_back(group("porofluid_elasticity_dynamic/monolithic",
      {
          // nonlinear solver
          group("nonlinear_solver",
              {
                  parameter<int>("linear_solver_id",
                      {.description = "number of linear solver used for poroelasticity problems",
                          .default_value = -1}),

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
              },
              {.required = false}),

          // finite difference check
          parameter<bool>("fd_check", {.description = "FD check active", .default_value = false}),
      },
      {.required = false}));

  // partitioned parameters
  specs.push_back(group("porofluid_elasticity_dynamic/partitioned",
      {
          // convergence tolerance of outer iteration loop
          parameter<double>("convergence_tolerance",
              {.description = "tolerance for convergence check of outer iteration",
                  .default_value = 1e-6}),

          // relaxation of partitioned scheme
          group("relaxation",
              {
                  parameter<RelaxationMethods>(
                      "type", {.description = "type relaxation of partitioned scheme",
                                  .default_value = RelaxationMethods::none}),

                  // parameters for relaxation of partitioned coupling
                  parameter<double>("start_omega",
                      {.description = "fixed relaxation parameter", .default_value = 1.0}),
                  parameter<double>("minimum_omega",
                      {.description = "smallest omega allowed for Aitken relaxation",
                          .default_value = 0.1}),
                  parameter<double>("maximum_omega",
                      {.description = "largest omega allowed for Aitken relaxation",
                          .default_value = 10.0}),
              },
              {.required = false}),
      },
      {.required = false}));
  return specs;
}

FOUR_C_NAMESPACE_CLOSE
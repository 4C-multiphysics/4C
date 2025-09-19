// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_porofluid_pressure_based_input.hpp"

#include "4C_art_net_input.hpp"
#include "4C_io_input_spec_builders.hpp"

#include <Teuchos_Time.hpp>

FOUR_C_NAMESPACE_OPEN

std::vector<Core::IO::InputSpec> PoroPressureBased::valid_parameters_porofluid()
{
  using namespace Core::IO::InputSpecBuilders;

  std::vector<Core::IO::InputSpec> specs;
  specs.push_back(group("porofluid_dynamic",
      {
          parameter<double>("total_simulation_time",
              {.description = "Total simulation time", .default_value = -1.0}),

          // time integration
          group("time_integration",
              {
                  parameter<int>("number_of_time_steps",
                      {.description = "Number of time steps", .default_value = -1}),
                  parameter<double>(
                      "time_step_size", {.description = "Time step size", .default_value = -1.0}),
                  parameter<double>(
                      "theta", {.description = "One-step-theta time integration factor",
                                   .default_value = 0.5}),
                  parameter<TimeIntegrationScheme>(
                      "scheme", {.description = "Time integration scheme",
                                    .default_value = TimeIntegrationScheme::one_step_theta}),
              },
              {.required = false}),

          // nonlinear solver settings
          group("nonlinear_solver",
              {
                  parameter<int>("maximum_number_of_iterations",
                      {.description = "Maximum number of nonlinear iterations",
                          .default_value = 10}),
                  parameter<double>("absolute_tolerance_residual",
                      {.description = "Absolute tolerance for deciding if residual "
                                      "of nonlinear problem is already zero",
                          .default_value = 1e-14}),
                  parameter<int>("linear_solver_id",
                      {.description = "ID of linear solver", .default_value = 1}),

                  group("residual",
                      {
                          parameter<VectorNorm>(
                              "vector_norm", {.description = "Type of norm to be applied to "
                                                             "residuals in the nonlinear solver",
                                                 .default_value = VectorNorm::l2}),
                          parameter<double>("tolerance",
                              {.description = "Tolerance for residual norm of the nonlinear solver",
                                  .default_value = 1e-6}),
                      },
                      {.required = false}),

                  group("increment",
                      {
                          parameter<VectorNorm>(
                              "vector_norm", {.description = "Type of norm to be applied to "
                                                             "increments in the nonlinear solver",
                                                 .default_value = VectorNorm::l2}),
                          parameter<double>("tolerance",
                              {.description =
                                      "Tolerance for the increment norm of the nonlinear solver",
                                  .default_value = 1e-6}),
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

          // initial field of the FE solution
          group("initial_condition",
              {
                  parameter<InitialField>(
                      "type", {.description = "Initial Field for the porofluid problem"}),
                  parameter<int>("function_id",
                      {.description = "function number for scalar transport initial field",
                          .default_value = -1}),
              }),

          // output of the simulation
          group("output",
              {
                  parameter<int>("result_data_every",
                      {.description = "Increment for writing solution", .default_value = 1}),
                  parameter<int>("restart_data_every",
                      {.description = "Increment for writing restart data", .default_value = 1}),
                  parameter<bool>("saturation_and_pressure",
                      {.description = "Flag if output of saturations and pressures should be "
                                      "calculated",
                          .default_value = true}),
                  parameter<bool>("solid_pressure",
                      {.description = "Flag if output of solid pressure should be calculated",
                          .default_value = true}),
                  parameter<bool>(
                      "porosity", {.description = "Flag if output of porosity should be calculated",
                                      .default_value = true}),
                  parameter<bool>("phase_velocities",
                      {.description = "Flag if output of phase velocities should be calculated",
                          .default_value = true}),
                  parameter<bool>("volfrac_blood_lung",
                      {.description = "Flag if output of volfrac blood lung should be calculated",
                          .default_value = false}),
                  parameter<bool>("determinant_of_deformation_gradient",
                      {.description = "Flag if output of determinant_of_deformation_gradient "
                                      "should be calculated",
                          .default_value = false}),
              },
              {.required = false}),

          // finite difference check (to check correct linearization)
          group("fd_check",
              {
                  parameter<bool>("active",
                      {.description = "Activate finite difference check", .default_value = false}),
                  parameter<double>("epsilon",
                      {.description =
                              "Dof perturbation magnitude for finite difference check (1.e-6 "
                              "seems to work very well, whereas smaller values don't)",
                          .default_value = 1.e-6}),
                  parameter<double>(
                      "tolerance", {.description = "Relative tolerance for finite difference check",
                                       .default_value = 1.e-6}),
              },
              {.required = false}),

          // calculate error of the FE solution to the analytical solution
          group("calculate_error_to_analytical_solution",
              {
                  parameter<bool>(
                      "active", {.description = "Compute error compared to analytical solution",
                                    .default_value = false}),
                  parameter<int>("function_id",
                      {.description = "Function ID of analytical solution for error computation",
                          .default_value = -1}),
              },
              {.required = false}),

          // skip the computation of the initial time derivative
          parameter<bool>("skip_initial_time_derivative",
              {.description = "Flag to skip computation of initial time derivative",
                  .default_value = true}),

          // Biot stabilization
          group("biot_stabilization",
              {
                  parameter<bool>(
                      "active", {.description = "Flag to (de)activate BIOT stabilization",
                                    .default_value = false}),
                  parameter<double>("scaling_factor",
                      {.description = "Scaling factor for stabilization parameter for "
                                      "biot stabilization",
                          .default_value = 1.0}),
              },
              {.required = false}),

          // what to do when the nonlinear solver does not converge
          parameter<DivergenceAction>("divergence_action",
              {.description =
                      "What to do with time integration when the nonlinear solver did not converge",
                  .default_value = DivergenceAction::stop}),

          // flux reconstruction
          group("flux_reconstruction",
              {
                  parameter<bool>("active",
                      {.description = "Activate flux reconstruction", .default_value = false}),
                  parameter<int>("solver_id",
                      {.description = "Number of linear solver used for flux reconstruction",
                          .default_value = -1}),
              },
              {.required = false}),

          // functions used for domain integrals
          parameter<std::string>("domain_integrals_function_ids",
              {.description = "functions used for domain integrals", .default_value = "-1.0"}),

          // coupling with 1D artery network
          parameter<bool>("artery_coupling_active",
              {.description = "Coupling with 1D blood vessels.", .default_value = false}),

          // Dirichlet boundary condition only applied to the first time steps
          group("starting_DBC",
              {
                  parameter<std::string>(
                      "active", {.description = "Switching the starting Dirichlet BC on or off.",
                                    .default_value = "0"}),
                  parameter<double>(
                      "time_end", {.description = "End time for the starting Dirichlet BC.",
                                      .default_value = -1.0}),
                  parameter<std::string>("function_ids",
                      {.description = "Function prescribing the starting Dirichlet BC.",
                          .default_value = "0"}),
              },
              {.required = false}),
      },
      {.required = false}));

  // artery meshtying
  specs.push_back(group("porofluid_dynamic/artery_coupling",
      {
          parameter<ArteryNetwork::ArteryPorofluidElastScatraCouplingMethod>(
              "coupling_method", {.description = "Coupling method for artery coupling."}),

          // coupled dofs for meshtying
          group("coupled_dofs",
              {
                  parameter<std::string>(
                      "artery", {.description = "coupled artery dofs for meshtying",
                                    .default_value = "-1.0"}),
                  parameter<std::string>(
                      "homogenized", {.description = "coupled homogenized dofs for meshtying",
                                         .default_value = "-1.0"}),
              },
              {.required = false}),

          // reactions for transfer from artery to homogenized part and vice versa
          group("reaction_terms",
              {
                  // functions for coupling (artery part)
                  parameter<std::string>(
                      "artery_function_ids", {.description = "functions for coupling (artery part)",
                                                 .default_value = "-1"}),
                  // scale for coupling (artery part)
                  parameter<std::string>("artery_scaling",
                      {.description = "scale for coupling (artery part)", .default_value = "1"}),

                  // functions for coupling (porofluid part)
                  parameter<std::string>("homogenized_function_ids",
                      {.description = "functions for coupling (homogenized part)",
                          .default_value = "-1"}),
                  // scale for coupling (porofluid part)
                  parameter<std::string>("homogenized_scaling",
                      {.description = "scale for coupling (homogenized part)",
                          .default_value = "1"}),
              },
              {.required = false}),

          // maximum number of segments per artery element for 1D-3D artery coupling
          parameter<int>("maximum_number_of_segments_per_artery_element",
              {.description =
                      "maximum number of segments per artery element for 1D-3D artery coupling",
                  .default_value = 5}),

          // penalty parameter
          parameter<double>(
              "penalty_parameter", {.description = "Penalty parameter for line-based coupling",
                                       .default_value = 1000.0}),

          // Flag if artery elements are evaluated in reference or current configuration
          parameter<bool>("evaluate_in_reference_configuration",
              {.description = "Flag if artery elements are evaluated in reference or current "
                              "configuration"}),

          // Flag if 1D-3D coupling should be evaluated on lateral (cylinder) surface of embedded
          // artery elements
          parameter<bool>("lateral_surface_coupling",
              {.description = "Flag if 1D-3D coupling should be evaluated on lateral (cylinder) "
                              "surface of embedded artery elements",
                  .default_value = false}),

          // Integration patches for lateral surface coupling
          group("integration_patches",
              {
                  // Number of integration patches per 1D element in axial direction
                  parameter<int>("number_of_patches_axial",
                      {.description =
                              "Number of integration patches per 1D element in axial direction for "
                              "lateral surface coupling",
                          .default_value = 1}),

                  // Number of integration patches per 1D element in radial direction
                  parameter<int>("number_of_patches_radial",
                      {.description = "Number of integration patches per 1D element in radial "
                                      "direction for lateral surface coupling",
                          .default_value = 1}),
              },
              {.required = false}),

          // Flag if blood vessel volume fraction should be output
          parameter<bool>("output_blood_vessel_volume_fraction",
              {.description = "Flag if output of blood vessel volume fraction should be calculated",
                  .default_value = false}),

          // Flag if summary of coupling-pairs should be printed
          parameter<bool>("print_coupling_pairs_summary",
              {.description = "Flag if summary of coupling-pairs should be printed",
                  .default_value = false}),

          // Flag if free-hanging elements (after blood vessel collapse) should be deleted
          parameter<bool>("delete_free_hanging_elements",
              {.description = "Flag if free-hanging elements (after blood vessel collapse) should "
                              "be deleted",
                  .default_value = false}),

          // components whose size is smaller than this fraction of the total network size are
          // also deleted
          parameter<double>("delete_small_components_fraction",
              {.description =
                      "Small connected components whose size is smaller than this fraction of "
                      "the overall network size are additionally deleted (a valid choice of this "
                      "parameter should lie between 0 and 1)",
                  .default_value = -1.0}),
      },
      {.required = false}));
  return specs;
}

FOUR_C_NAMESPACE_CLOSE
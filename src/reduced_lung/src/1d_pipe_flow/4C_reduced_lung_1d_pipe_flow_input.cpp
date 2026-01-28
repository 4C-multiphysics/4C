// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_reduced_lung_1d_pipe_flow_input.hpp"

#include "4C_io_input_spec_builders.hpp"
#include "4C_utils_function.hpp"

FOUR_C_NAMESPACE_OPEN

Core::IO::InputSpec ReducedLung1dPipeFlow::valid_parameters()
{
  using namespace Core::IO::InputSpecBuilders;

  Core::IO::InputSpec spec = group("reduced_lung",
      {
          group<Parameters>("general",
              {
                  parameter<double>(
                      "final_time", {.description = "Final time when the simulation ends",
                                        .store = in_struct(&Parameters::final_time)}),
                  parameter<int>("steps", {.description = "Number of time steps in the simulation",
                                              .store = in_struct(&Parameters::n_steps)}),
                  parameter<int>(
                      "result_every", {.description = "Write result every n timestep",
                                          .validator = Validators::positive<int>(),
                                          .store = in_struct(&Parameters::result_every)}),

                  group<Parameters::Fluid>("fluid_properties",
                      {parameter<double>(
                           "density_rho", {.description = "Density of blood",
                                              .validator = Validators::positive<double>(),
                                              .store = in_struct(&Parameters::Fluid::density_rho)}),
                          parameter<double>("viscosity_mu",
                              {.description = "Viscosity of blood",
                                  .store = in_struct(&Parameters::Fluid::viscosity_mu)})},
                      {.description = "Parameters of blood",
                          .store = in_struct(&Parameters::fluid)}),

                  group<Parameters::Material>("material_parameters",
                      {input_field<double>("Young_E",
                           {.description =
                                   "Young's modulus depending on generation and vessel type.",
                               .store = in_struct(&Parameters::Material::youngs_modulus_E)}),
                          parameter<double>("Poisson_ratio_nue",
                              {.description = "Poisson ratio of material",
                                  .store = in_struct(&Parameters::Material::poisson_ratio_nu)})},
                      {.description = "Parameters of wall mechanics",
                          .store = in_struct(&Parameters::material)}),

                  group<Parameters::Geometry>("geometry_parameters",
                      {
                          input_field<double>("reference_area_A0",
                              {.description = "Reference area of element",
                                  .store = in_struct(&Parameters::Geometry::reference_area_A0)}),
                          input_field<double>("thickness_th",
                              {.description = "Wall thickness of element",
                                  .store = in_struct(&Parameters::Geometry::thickness_th)}),
                      },
                      {.description = "Geometry of pulmonary vasculature",
                          .store = in_struct(&Parameters::geometry)}),

                  group<Parameters::BoundaryConditions>("boundary_conditions",
                      {
                          deprecated_selection<std::string>("input",
                              {"area", "velocity", "flow", "pressure"},
                              {.description = "Choice of prescribed parameter at inflow boundary",
                                  .store = in_struct(&Parameters::BoundaryConditions::input)}),
                          deprecated_selection<std::string>("output", {"reflection", "pressure"},
                              {.description = "Choice of prescribed parameter at outflow boundary",
                                  .store = in_struct(&Parameters::BoundaryConditions::output)}),
                          parameter<int>("function_ID",
                              {.description = "ID of function describing the boundary condition",
                                  .store = in_struct(&ReducedLung1dPipeFlow::Parameters::
                                          BoundaryConditions::function_id_inflow)}),
                          parameter<double>("condition_outflow",
                              {.description = "ID of function describing the boundary condition",
                                  .store = in_struct(
                                      &Parameters::BoundaryConditions::condition_outflow)}),
                          parameter<std::optional<double>>("cycle_period",
                              {.description = "Duration of one cycle until next one starts, to "
                                              "implement pulsatility (systolic vs diastolic).",
                                  .store =
                                      in_struct(&Parameters::BoundaryConditions::cycle_period)}),
                          parameter<std::optional<double>>("pulse_width",
                              {.description = "Duration of input pulse in the cycle - Heaviside "
                                              "function = 1.",
                                  .store =
                                      in_struct(&Parameters::BoundaryConditions::pulse_width)}),
                      },
                      {.description = "Parameters of blood",
                          .store = in_struct(&Parameters::boundary_conditions)}),
              }),
      },  // content of group reduced_lung
      {.description = "Reduced-dimensional fluid dynamics in a 1D network", .required = false});
  return spec;
}

FOUR_C_NAMESPACE_CLOSE

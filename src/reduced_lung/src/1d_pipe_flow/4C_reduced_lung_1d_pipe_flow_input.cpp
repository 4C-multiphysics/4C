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

  // rheology model spec
  Core::IO::InputSpec rheological_model_spec_terminal_unit = group<
      Parameters::TerminalUnits::RheologicalModel>("rheological_model",
      {
          input_field<Parameters::TerminalUnits::RheologicalModel::RheologicalModelType>(
              "rheological_model_type",
              {
                  .description = "Type of the rheological model.",
                  .store = in_struct(
                      &Parameters::TerminalUnits::RheologicalModel::rheological_model_type),
              }),
          group<Parameters::TerminalUnits::RheologicalModel::KelvinVoigt>("kelvin_voigt",
              {
                  input_field<double>("viscosity_kelvin_voigt_eta",
                      {
                          .description = "Viscosity parameter (dashpot) of the terminal unit.",
                          .store = in_struct(&Parameters::TerminalUnits::RheologicalModel::
                                  KelvinVoigt::viscosity_kelvin_voigt_eta),
                      }),
              },
              {
                  .description = "Kelvin-Voigt model of the terminal unit.",
                  .required = false,
                  .store = in_struct(&Parameters::TerminalUnits::RheologicalModel::kelvin_voigt),
              }),
          group<Parameters::TerminalUnits::RheologicalModel::FourElementMaxwell>(
              "4_element_maxwell",
              {
                  input_field<double>("viscosity_kelvin_voigt_eta",
                      {
                          .description =
                              "Dashpot viscosity of the Kelvin-Voigt body of the terminal unit.",
                          .store = in_struct(&Parameters::TerminalUnits::RheologicalModel::
                                  FourElementMaxwell::viscosity_kelvin_voigt_eta),
                      }),
                  input_field<double>("viscosity_maxwell_eta_m",
                      {
                          .description =
                              "Dashpot viscosity of the Maxwell body of the terminal unit.",
                          .store = in_struct(&Parameters::TerminalUnits::RheologicalModel::
                                  FourElementMaxwell::viscosity_maxwell_eta_m),
                      }),
                  input_field<double>("elasticity_maxwell_e_m",
                      {
                          .description = "Spring stiffness of the Maxwell "
                                         "body of the terminal unit.",
                          .store = in_struct(&Parameters::TerminalUnits::RheologicalModel::
                                  FourElementMaxwell::elasticity_maxwell_e_m),
                      }),
              },
              {
                  .description = "4-element Maxwell model of the "
                                 "terminal unit.",
                  .required = false,
                  .store =
                      in_struct(&Parameters::TerminalUnits::RheologicalModel::four_element_maxwell),
              }),
      },
      {
          .description = "Rheological model of the terminal unit.",
          .required = false,
          .store = in_struct(&Parameters::TerminalUnits::rheological_model),
      });

  // elasticity model spec
  Core::IO::InputSpec elasticity_model_spec_terminal_units = group<
      Parameters::TerminalUnits::ElasticityModel>("elasticity_model",
      {
          input_field<Parameters::TerminalUnits::ElasticityModel::ElasticityModelType>(
              "elasticity_model_type",
              {
                  .description = "Type of the elastic model.",
                  .store =
                      in_struct(&Parameters::TerminalUnits::ElasticityModel::elasticity_model_type),
              }),
          input_field<double>("acinar_reference_volume_v0",
              {.description = "Reference volume of acinus at terminal unit.",
                  .store =
                      in_struct(&Parameters::TerminalUnits::ElasticityModel::acinar_volume_v)}),
          group<Parameters::TerminalUnits::ElasticityModel::Linear>("linear",
              {
                  input_field<double>("elasticity_e",
                      {
                          .description = "Linear elastic stiffness of the terminal unit.",
                          .store = in_struct(
                              &Parameters::TerminalUnits::ElasticityModel::Linear::elasticity_e),
                      }),
              },
              {
                  .description =
                      "Linear elastic model in the rheological model of the terminal unit.",
                  .required = false,
                  .store = in_struct(&Parameters::TerminalUnits::ElasticityModel::linear),
              }),
          group<Parameters::TerminalUnits::ElasticityModel::Ogden>("ogden",
              {
                  input_field<double>("ogden_parameter_kappa",
                      {
                          .description = "Parameter Kappa in volumetric Ogden law.",
                          .store = in_struct(&Parameters::TerminalUnits::ElasticityModel::Ogden::
                                  ogden_parameter_kappa),
                      }),
                  input_field<double>("ogden_parameter_beta",
                      {
                          .description = "Parameter Beta in volumetric Ogden law.",
                          .store = in_struct(&Parameters::TerminalUnits::ElasticityModel::Ogden::
                                  ogden_parameter_beta),
                      }),
              },
              {
                  .description = "Ogden type spring in the rheological model of the terminal unit.",
                  .required = false,
                  .store = in_struct(&Parameters::TerminalUnits::ElasticityModel::ogden),
              }),
      },
      {
          .description = "Elasticity model for the customizable spring of the rheological model.",
          .required = false,
          .store = in_struct(&Parameters::TerminalUnits::elasticity_model),
      });

  Core::IO::InputSpec spec = group("reduced_lung",
      {
          group<Parameters>("general",
              {
                  parameter<double>(
                      "final_time", {.description = "Final time when the simulation ends",
                                        .store = in_struct(&Parameters::final_time)}),
                  parameter<int>("steps", {.description = "Number of time steps in the simulation",
                                              .validator = Validators::positive<int>(),
                                              .store = in_struct(&Parameters::n_steps)}),
                  parameter<int>(
                      "result_every", {.description = "Write result every n timestep",
                                          .validator = Validators::positive<int>(),
                                          .store = in_struct(&Parameters::result_every)}),

                  group<Parameters::Fluid>("fluid_properties",
                      {parameter<double>(
                           "density_rho", {.description = "Density of fluid",
                                              .validator = Validators::positive<double>(),
                                              .store = in_struct(&Parameters::Fluid::density_rho)}),
                          parameter<double>("viscosity_mu",
                              {.description = "Viscosity of fluid",
                                  .store = in_struct(&Parameters::Fluid::viscosity_mu)})},
                      {.description = "Parameters of fluid",
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
                          deprecated_selection<std::string>("output",
                              {"reflection", "pressure", "terminal_unit"},
                              {.description = "Choice of prescribed parameter at outflow boundary",
                                  .store = in_struct(&Parameters::BoundaryConditions::output)}),
                          parameter<int>("function_ID",
                              {.description = "ID of function describing the boundary condition",
                                  .store = in_struct(&ReducedLung1dPipeFlow::Parameters::
                                          BoundaryConditions::function_id_inflow)}),
                          parameter<double>("condition_outflow",
                              {.description = "Condition applied at the outlet, either pressure "
                                              "or reflection.",
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
                      {.description = "Parameters of fluid",
                          .store = in_struct(&Parameters::boundary_conditions)}),

                  group<Parameters::TerminalUnits>("terminal_units",
                      {
                          input_field<Parameters::TerminalUnits::TerminalUnitType>(
                              "terminal_unit_type",
                              {
                                  .description = "Type of terminal unit model.",
                                  .store =
                                      in_struct(&Parameters::TerminalUnits::terminal_unit_type),
                              }),
                          rheological_model_spec_terminal_unit,
                          elasticity_model_spec_terminal_units,
                          group<Parameters::TerminalUnits::WindkesselModel>("windkessel_model",
                              {
                                  input_field<double>("proximal_resistance_R_p",
                                      {.description = "Proximal resistance of Windkessel model.",
                                          .store = in_struct(&Parameters::TerminalUnits::
                                                  WindkesselModel::proximal_resistance_R_p)}),
                                  input_field<double>("compliance_C",
                                      {.description = "Compliance of Windkessel model.",
                                          .store = in_struct(&Parameters::TerminalUnits::
                                                  WindkesselModel::compliance_C)}),
                                  input_field<double>("distal_resistance_R_d",
                                      {.description = "Distal resistance of Windkessel model.",
                                          .store = in_struct(&Parameters::TerminalUnits::
                                                  WindkesselModel::distal_resistance_R_d)}),
                                  input_field<double>("pressure_peripheral",
                                      {.description = "Peripheral pressure after Windkessel model.",
                                          .store = in_struct(&Parameters::TerminalUnits::
                                                  WindkesselModel::pressure_peripheral)}),
                              },
                              {.description = "Windkessel (RCR) model for terminal units.",
                                  .required = false,
                                  .store =
                                      in_struct(&Parameters::TerminalUnits::windkessel_model)}),
                      },
                      {.description = "Terminal units.",
                          .required = false,
                          .store = in_struct(&Parameters::terminal_units)}),
              },  // content of group reduced_lung
              {.description = "Information needed for 1D modeling of fluid flow."}),
      },
      {.description = "Reduced-dimensional fluid dynamics in a 1D network", .required = false});
  return spec;
}

FOUR_C_NAMESPACE_CLOSE

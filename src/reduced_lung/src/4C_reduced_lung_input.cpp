// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_reduced_lung_input.hpp"

#include "4C_fem_condition_definition.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_red_airways_input.hpp"
#include "4C_reduced_lung_input.hpp"
#include "4C_reduced_lung_terminal_unit.hpp"

#include <KokkosKernels_Utils.hpp>


FOUR_C_NAMESPACE_OPEN


Core::IO::InputSpec ReducedLung::valid_parameters()
{
  using namespace Core::IO::InputSpecBuilders;

  Core::IO::InputSpec flow_model_spec_airway = group<
      ReducedLungParameters::LungTree::Airways::FlowModel>("flow_model",
      {
          input_field<ReducedLungParameters::LungTree::Airways::FlowModel::ResistanceType>(
              "resistance_type",
              {
                  .description = "Type of resistance model for the airway.",
                  .store = in_struct(
                      &ReducedLungParameters::LungTree::Airways::FlowModel::resistance_type),
              }),
          group<ReducedLungParameters::LungTree::Airways::FlowModel::ResistanceModel>(
              "resistance_model",
              {group<
                  ReducedLungParameters::LungTree::Airways::FlowModel::ResistanceModel::NonLinear>(
                  "non_linear",
                  {input_field<double>("turbulence_factor_gamma",
                      {
                          .description = "Van Ertbruggen's generation dependent turbulence factor "
                                         "defining turbulent onset.",
                          .store = in_struct(&ReducedLungParameters::LungTree::Airways::FlowModel::
                                  ResistanceModel::NonLinear::turbulence_factor_gamma),
                      })},
                  {
                      .description = "Definition of the non-linear airway "
                                     "resistance model.",
                      .required = false,
                      .store = in_struct(&ReducedLungParameters::LungTree::Airways::FlowModel::
                              ResistanceModel::non_linear),
                  })},
              {
                  .description = "Definition of the airway resistance model",
                  .required = false,
                  .store = in_struct(
                      &ReducedLungParameters::LungTree::Airways::FlowModel::resistance_model),
              }),
          input_field<bool>("include_inertia",
              {
                  .description = "Include inertial effects in the airway flow model.",
                  .store = in_struct(
                      &ReducedLungParameters::LungTree::Airways::FlowModel::include_inertia),
              }),
      },
      {
          .description = "Flow model of the airway.",
          .store = in_struct(&ReducedLungParameters::LungTree::Airways::flow_model),
      });
  Core::IO::InputSpec wall_model_spec_airway = group<
      ReducedLungParameters::LungTree::Airways::WallModel>("wall_model",
      {
          group<ReducedLungParameters::LungTree::Airways::WallModel::KelvinVoigt>("kelvin_voigt",
              {group<ReducedLungParameters::LungTree::Airways::WallModel::KelvinVoigt::Elasticity>(
                   "elasticity",
                   {
                       input_field<double>("wall_poisson_ratio",
                           {
                               .description = "Poisson's ratio of the airway wall.",
                               .store = in_struct(&ReducedLungParameters::LungTree::Airways::
                                       WallModel::KelvinVoigt::Elasticity::wall_poisson_ratio),
                           }),
                       input_field<double>("wall_elasticity",
                           {
                               .description = "Elasticity of the airway wall.",
                               .store = in_struct(&ReducedLungParameters::LungTree::Airways::
                                       WallModel::KelvinVoigt::Elasticity::wall_elasticity),
                           }),
                       input_field<double>("wall_thickness",
                           {
                               .description = "Airway wall thickness.",
                               .store = in_struct(&ReducedLungParameters::LungTree::Airways::
                                       WallModel::KelvinVoigt::Elasticity::wall_thickness),
                           }),
                   },
                   {
                       .description = "Elasticity parameters of the airway wall.",
                       .required = false,
                       .store = in_struct(&ReducedLungParameters::LungTree::Airways::WallModel::
                               KelvinVoigt::elasticity),
                   }),
                  group<
                      ReducedLungParameters::LungTree::Airways::WallModel::KelvinVoigt::Viscosity>(
                      "viscosity",
                      {
                          input_field<double>("viscous_time_constant",
                              {
                                  .description = "Viscous time constant.",
                                  .store = in_struct(&ReducedLungParameters::LungTree::Airways::
                                          WallModel::KelvinVoigt::Viscosity::viscous_time_constant),
                              }),
                          input_field<double>("viscous_phase_shift",
                              {
                                  .description = "Viscous phase shift.",
                                  .store = in_struct(&ReducedLungParameters::LungTree::Airways::
                                          WallModel::KelvinVoigt::Viscosity::viscous_phase_shift),
                              }),
                      },
                      {
                          .description = "Viscous parameters of the airway wall.",
                          .required = false,
                          .store = in_struct(&ReducedLungParameters::LungTree::Airways::WallModel::
                                  KelvinVoigt::viscosity),
                      })},
              {
                  .description = "Kelvin-Voigt type airway wall model.",
                  .required = false,
                  .store =
                      in_struct(&ReducedLungParameters::LungTree::Airways::WallModel::kelvin_voigt),
              }),
      },
      {
          .description = "Wall model of the airway.",
          .required = false,
          .store = in_struct(&ReducedLungParameters::LungTree::Airways::wall_model),
      });

  Core::IO::InputSpec rheological_model_spec_terminal_unit = group<
      ReducedLungParameters::LungTree::TerminalUnits::RheologicalModel>("rheological_model",
      {
          input_field<ReducedLungParameters::LungTree::TerminalUnits::RheologicalModel::
                  RheologicalModelType>("rheological_model_type",
              {
                  .description = "Type of the rheological model.",
                  .store = in_struct(&ReducedLungParameters::LungTree::TerminalUnits::
                          RheologicalModel::rheological_model_type),
              }),
          group<ReducedLungParameters::LungTree::TerminalUnits::RheologicalModel::KelvinVoigt>(
              "kelvin_voigt",
              {
                  input_field<double>("viscosity_kelvin_voigt_eta",
                      {
                          .description = "Viscosity parameter (dashpot) of the terminal unit.",
                          .store = in_struct(&ReducedLungParameters::LungTree::TerminalUnits::
                                  RheologicalModel::KelvinVoigt::viscosity_kelvin_voigt_eta),
                      }),
              },
              {
                  .description = "Kelvin-Voigt model of the terminal unit.",
                  .required = false,
                  .store = in_struct(&ReducedLungParameters::LungTree::TerminalUnits::
                          RheologicalModel::kelvin_voigt),
              }),
          group<
              ReducedLungParameters::LungTree::TerminalUnits::RheologicalModel::FourElementMaxwell>(
              "4_element_maxwell",
              {
                  input_field<double>("viscosity_kelvin_voigt_eta",
                      {
                          .description =
                              "Dashpot viscosity of the Kelvin-Voigt body of the terminal unit.",
                          .store = in_struct(&ReducedLungParameters::LungTree::TerminalUnits::
                                  RheologicalModel::FourElementMaxwell::viscosity_kelvin_voigt_eta),
                      }),
                  input_field<double>("viscosity_maxwell_eta_m",
                      {
                          .description =
                              "Dashpot viscosity of the Maxwell body of the terminal unit.",
                          .store = in_struct(&ReducedLungParameters::LungTree::TerminalUnits::
                                  RheologicalModel::FourElementMaxwell::viscosity_maxwell_eta_m),
                      }),
                  input_field<double>("elasticity_maxwell_e_m",
                      {
                          .description = "Spring stiffness of the Maxwell "
                                         "body of the terminal unit.",
                          .store = in_struct(&ReducedLungParameters::LungTree::TerminalUnits::
                                  RheologicalModel::FourElementMaxwell::elasticity_maxwell_e_m),
                      }),
              },
              {
                  .description = "4-element Maxwell model of the "
                                 "terminal unit.",
                  .required = false,
                  .store = in_struct(&ReducedLungParameters::LungTree::TerminalUnits::
                          RheologicalModel::four_element_maxwell),
              }),
      },
      {
          .description = "Rheological model of the terminal unit.",
          .store = in_struct(&ReducedLungParameters::LungTree::TerminalUnits::rheological_model),
      });


  Core::IO::InputSpec elasticity_model_spec_terminal_units = group<
      ReducedLungParameters::LungTree::TerminalUnits::ElasticityModel>("elasticity_model",
      {
          input_field<
              ReducedLungParameters::LungTree::TerminalUnits::ElasticityModel::ElasticityModelType>(
              "elasticity_model_type",
              {
                  .description = "Type of the elastic model.",
                  .store = in_struct(&ReducedLungParameters::LungTree::TerminalUnits::
                          ElasticityModel::elasticity_model_type),
              }),
          group<ReducedLungParameters::LungTree::TerminalUnits::ElasticityModel::Linear>("linear",
              {
                  input_field<double>("elasticity_e",
                      {
                          .description = "Linear elastic stiffness of the terminal unit.",
                          .store = in_struct(&ReducedLungParameters::LungTree::TerminalUnits::
                                  ElasticityModel::Linear::elasticity_e),
                      }),
              },
              {
                  .description =
                      "Linear elastic model in the rheological model of the terminal unit.",
                  .required = false,
                  .store = in_struct(
                      &ReducedLungParameters::LungTree::TerminalUnits::ElasticityModel::linear),
              }),
          group<ReducedLungParameters::LungTree::TerminalUnits::ElasticityModel::Ogden>("ogden",
              {
                  input_field<double>("ogden_parameter_kappa",
                      {
                          .description = "Parameter Kappa in volumetric Ogden law.",
                          .store = in_struct(&ReducedLungParameters::LungTree::TerminalUnits::
                                  ElasticityModel::Ogden::ogden_parameter_kappa),
                      }),
                  input_field<double>("ogden_parameter_beta",
                      {
                          .description = "Parameter Beta in volumetric Ogden law.",
                          .store = in_struct(&ReducedLungParameters::LungTree::TerminalUnits::
                                  ElasticityModel::Ogden::ogden_parameter_beta),
                      }),
              },
              {
                  .description = "Ogden type spring in the rheological model of the terminal unit.",
                  .required = false,
                  .store = in_struct(
                      &ReducedLungParameters::LungTree::TerminalUnits::ElasticityModel::ogden),
              }),
      },
      {
          .description = "Elasticity model for the customizable spring of the rheological model.",
          .store = in_struct(&ReducedLungParameters::LungTree::TerminalUnits::elasticity_model),
      });
  Core::IO::InputSpec spec = group<ReducedLungParameters>("reduced_dimensional_lung",
      {
          group<ReducedLungParameters::Dynamics>("dynamics",
              {
                  parameter<double>("time_increment",
                      {
                          .description = "Time increment dt.",
                          .store = in_struct(&ReducedLungParameters::Dynamics::time_increment),
                      }),
                  parameter<int>("number_of_steps",
                      {
                          .description = "Number of time steps.",
                          .store = in_struct(&ReducedLungParameters::Dynamics::number_of_steps),
                      }),
                  parameter<int>("restart_every",
                      {
                          .description = "Increment for writing restart.",
                          .default_value = 1,
                          .store = in_struct(&ReducedLungParameters::Dynamics::restart_every),
                      }),
                  parameter<int>("results_every",
                      {
                          .description = "Increment for writing solution.",
                          .default_value = 1,
                          .store = in_struct(&ReducedLungParameters::Dynamics::results_every),
                      }),
                  parameter<int>("linear_solver",
                      {
                          .description = "Number of linear solver used for reduced "
                                         "dimensional lung simulation.",
                          .store = in_struct(&ReducedLungParameters::Dynamics::linear_solver),
                      }),
              },
              {
                  .required = true,
                  .store = in_struct(&ReducedLungParameters::dynamics),
              }),

          group<ReducedLungParameters::LungTree>("lung_tree",
              {
                  input_field<ReducedLungParameters::LungTree::ElementType>("element_type",
                      {
                          .description = "Type of reduced lung elements.",
                          .store = in_struct(&ReducedLungParameters::LungTree::element_type),
                      }),
                  input_field<int>("generation",
                      {
                          .description = "Generation of the airway elements.",
                          .store = in_struct(&ReducedLungParameters::LungTree::generation),
                      }),
                  group<ReducedLungParameters::LungTree::Airways>("airways",
                      {input_field<double>("radius",
                           {
                               .description = "Radius of the Airway.",
                               .store =
                                   in_struct(&ReducedLungParameters::LungTree::Airways::radius),
                           }),
                          flow_model_spec_airway,
                          input_field<ReducedLungParameters::LungTree::Airways::WallModelType>(
                              "wall_model_type",
                              {
                                  .description = "Type of wall model of the airway.",
                                  .store = in_struct(
                                      &ReducedLungParameters::LungTree::Airways::wall_model_type),
                              }),
                          wall_model_spec_airway},
                      {
                          .description = "Definition of the airway model.",
                          .required = true,
                          .store = in_struct(&ReducedLungParameters::LungTree::airways),
                      }),
                  group<ReducedLungParameters::LungTree::TerminalUnits>("terminal_units",
                      {rheological_model_spec_terminal_unit, elasticity_model_spec_terminal_units},
                      {
                          .description = "Terminal units.",
                          .store = in_struct(&ReducedLungParameters::LungTree::terminal_units),
                      }),
              },
              {
                  .description = "Definition of the reduced dimensional lung tree including model "
                                 "definitions and parameters",
                  .store = in_struct(&ReducedLungParameters::lung_tree),
              }),
          group<ReducedLungParameters::AirProperties>("air_properties",
              {
                  parameter<double>("dynamic_viscosity",
                      {
                          .description = "Dynamic viscosity of air in the reduced dimensional lung "
                                         "simulation.",
                          .store =
                              in_struct(&ReducedLungParameters::AirProperties::dynamic_viscosity),
                      }),

                  parameter<double>("density",
                      {
                          .description =
                              "Density of air in the reduced dimensional lung simulation.",
                          .store = in_struct(&ReducedLungParameters::AirProperties::density),
                      }),
              },
              {
                  .description = "Air properties for the reduced dimensional lung simulation",
                  .store = in_struct(&ReducedLungParameters::air_properties),
              }),
      },
      {
          .required = false,
      });
  return spec;
}

FOUR_C_NAMESPACE_CLOSE

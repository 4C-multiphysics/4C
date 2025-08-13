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
#include "4C_reduced_lung_terminal_unit.hpp"

#include <KokkosKernels_Utils.hpp>


FOUR_C_NAMESPACE_OPEN


void ReducedLung::set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list)
{
  using namespace Core::IO::InputSpecBuilders;

  list["REDUCED DIMENSIONAL LUNG"] = group("REDUCED DIMENSIONAL LUNG",
      {
          group("DYNAMICS",
              {
                  parameter<IO::ReducedLungDyn>(
                      "DYNAMIC_TYPE", {.description = "Time integration scheme.",
                                          .default_value = IO::ReducedLungDyn::BackwardEuler}),
                  parameter<double>("TIMESTEP", {.description = "Time increment dt."}),
                  parameter<int>("NUM_STEP", {.description = "Number of time steps."}),
                  parameter<int>("RESTART_EVERY",
                      {.description = "Increment for writing restart.", .default_value = 1}),
                  parameter<int>("RESULTS_EVERY",
                      {.description = "Increment for writing solution.", .default_value = 1}),
                  parameter<int>(
                      "LINEAR_SOLVER", {.description = "Number of linear solver used for reduced "
                                                       "dimensional lung simulation."}),
              },
              {.required = true}),

          group("LUNG_TREE",
              {
                  input_field<IO::ElementType>(
                      "ELEMENT_TYPE", {.description = "Type of reduced lung elements."}),
                  input_field<int>(
                      "GENERATION", {.description = "Generation of the airway elements"}),
                  group("AIRWAYS",
                      {input_field<double>("RADIUS", {.description = "Radius of the Airway."}),
                          input_field<IO::AirwayModel>("AIRWAY_MODEL"),
                          group("RESISTIVE",
                              {
                                  input_field<double>("GAMMA",
                                      {.description =
                                              "Van Ertbruggen's generation dependent turbulence "
                                              "factor defining turbulent onset."}),
                              },
                              {.description = "Resistive airway model", .required = false}),
                          group("VISCOELASTIC_RLC",
                              {input_field<double>("GAMMA",
                                   {.description = "Van Ertbruggen's generation dependent "
                                                   "turbulence factor defining turbulent onset."}),
                                  group("WALL_MODEL",
                                      {
                                          input_field<IO::WallModel>("WALL_MODEL_TYPE",
                                              {.description = "Type of wall model for the "
                                                              "viscoelastic airway."}),
                                          group("STANDARD_WALL_MODEL",
                                              {
                                                  input_field<double>("WALL_POISSONS_RATIO",
                                                      {.description = "Poisson's ratio of the "
                                                                      "airway wall."}),
                                                  input_field<double>("WALL_ELASTICITY",
                                                      {.description =
                                                              "Elasticity of the airway wall."}),
                                                  input_field<double>(
                                                      "DIAMETER_OVER_WALL_THICKNESS",
                                                      {.description = "Ratio of diameter over wall "
                                                                      "thickness"}),
                                                  input_field<double>("VISCOUS_TIME_CONSTANT",
                                                      {.description = "Viscous time constant"}),
                                                  input_field<double>("VISCOUS_PHASE_SHIFT",
                                                      {.description = "Viscous phase shift"}),
                                              },
                                              {.description = "TODO find a name and description",
                                                  .required = false}),
                                      })},
                              {.description = "Wall model of the viscoelasitic airway",
                                  .required = false})}),
                  group("TERMINAL_UNITS",
                      {group("RHEOLOGICAL_MODEL",
                           {input_field<IO::RheologicalModel>("RHEOLOGICAL_MODEL_TYPE",
                                {.description = "Type of the rheological model."}),
                               group("KELVIN_VOIGT",
                                   {input_field<double>(
                                       "ETA", {.description = "Viscosity parameter (dashpot) of "
                                                              "the terminal unit."})},
                                   {.description = "Kelvin-Voigt model of the terminal unit",
                                       .required = false}),
                               group("4_ELEMENT_MAXWELL",
                                   {input_field<double>("ETA",
                                        {.description = "Dashpot viscosity of the Kelvin-Voigt "
                                                        "body of the terminal unit."}),
                                       input_field<double>("ETA_M",
                                           {.description = "Dashpot viscosity of the Maxwell body "
                                                           "of the terminal unit."}),
                                       input_field<double>("ELASTICITY_M",
                                           {.description = "Spring stiffness of the Maxwell body "
                                                           "of the terminal unit."})},
                                   {.description = "4-element Maxwell model of the terminal unit",
                                       .required = false})},
                           {.description = "Rheological model of the terminal unit"}),
                          group("ELASTICITY_MODEL",
                              {input_field<IO::ElasticityModel>("ELASTICITY_MODEL_TYPE",
                                   {.description = "Type of the elastic model."}),
                                  group("LINEAR",
                                      {input_field<double>("ELASTICITY",
                                          {.description = "Linear elastic stiffness."})},
                                      {.description = "Linear elastic model in the rheological "
                                                      "model of the terminal unit.",
                                          .required = false}),
                                  group("OGDEN",
                                      {input_field<double>("KAPPA", {.description = "Kappa"}),
                                          input_field<double>("BETA", {.description = "Beta"})},
                                      {.description =
                                              "Ogden type spring in the rheological model of the "
                                              "terminal unit.",
                                          .required = false})},
                              {.description = "Elasticity model for the customizable spring of the "
                                              "rheological model."})},
                      {.description = "Terminal units", .required = true}),
              },
              {.description = "Description of the reduced dimensional lung tree."}),

          group("AIR_PROPERTIES",
              {parameter<double>("DYNAMIC_VISCOSITY",
                   {.description =
                           "Dynamic viscosity of air in the reduced dimensional lung simulation."}),
                  parameter<double>("DENSITY",
                      {.description =
                              "Density of air in the reduced dimensional lung simulation."})},
              {.required = true}),
      },
      {.required = false});
}


FOUR_C_NAMESPACE_CLOSE

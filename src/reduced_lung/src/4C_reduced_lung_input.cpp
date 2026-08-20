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

  Core::IO::InputSpec geometry_spec = group<ReducedLungParameters::Geometry>("geometry",
      {
          parameter<std::filesystem::path>("file",
              {
                  .description =
                      "Path to the VTU mesh file describing the reduced lung tree. Either "
                      "absolute or relative to the input file. The mesh provides the nodes and "
                      "the line2 cells of the tree and is also the source of all input fields "
                      "that are given as `from_mesh`. Such fields are read from the cell data of "
                      "the mesh; point data is not supported.",
                  .store = in_struct(&ReducedLungParameters::Geometry::file),
              }),
      },
      {
          .description = "Geometry of the reduced lung tree.",
          .required = true,
          .store = in_struct(&ReducedLungParameters::geometry),
      });

  using BoundaryConditions = ReducedLungParameters::BoundaryConditions;
  using FromFunctionDefinition = BoundaryConditions::FromFunctionDefinition;
  using VolumeDependentPleuralPressureDefinition =
      BoundaryConditions::VolumeDependentPleuralPressureDefinition;

  // Every definition is identified the same way, whichever value it prescribes, so share one spec.
  const Core::IO::InputSpec definition_id_spec = parameter<int>("id",
      {
          .description = "Unique positive id of this definition. It is referenced by the `bc_id` "
                         "point data of the mesh: every node with this `bc_id` carries this "
                         "boundary condition. The id 0 marks nodes without a boundary condition "
                         "and cannot be defined.",
          .validator = Validators::positive<int>(),
      });

  // Entries stay flat: a named group would force an extra nesting level in the yaml.
  const Core::IO::InputSpec from_function_definition_spec = all_of({
      definition_id_spec,
      parameter<int>("function_id",
          {
              .description = "Id of the function of time prescribing the boundary value.",
              .validator = Validators::positive<int>(),
          }),
  });

  // Converts a list of function-valued definitions into @p member.
  const auto store_from_function_definitions =
      [](std::vector<FromFunctionDefinition> BoundaryConditions::* member)
  {
    return StoreFunction<Core::IO::InputParameterContainer::List>(
        [member](Storage& storage, Core::IO::InputParameterContainer::List&& value)
        {
          FOUR_C_ASSERT(storage.type() == typeid(BoundaryConditions),
              "Implementation error: expected BoundaryConditions storage.");

          auto& target = std::any_cast<BoundaryConditions&>(storage).*member;
          // The store also runs from set_default_value(), so it may see the same target twice.
          target.clear();
          for (const auto& entry : value)
          {
            target.push_back(FromFunctionDefinition{
                .id = entry.get<int>("id"), .function_id = entry.get<int>("function_id")});
          }

          return StoreStatus::ok();
        },
        typeid(BoundaryConditions));
  };

  // The curve sits behind a one_of so that further curve shapes can be added later.
  const Core::IO::InputSpec pleural_pressure_definition_spec = all_of({
      definition_id_spec,
      parameter<VolumeDependentPleuralPressureDefinition::Coupling>("coupling",
          {
              .description = "How the pleural pressure follows the terminal unit volume. 'Frozen' "
                             "evaluates it from the total volume of the last converged timestep, "
                             "which makes it an explicit source term without Jacobian "
                             "contribution.",
              .default_value = VolumeDependentPleuralPressureDefinition::Coupling::Frozen,
          }),
      parameter<double>("residual_volume",
          {
              .description = "Total terminal unit volume at which the normalized volume is zero. "
                             "Must be smaller than total_lung_capacity.",
              .validator = Validators::positive_or_zero<double>(),
          }),
      parameter<double>("total_lung_capacity",
          {
              .description = "Total terminal unit volume at which the normalized volume is one. "
                             "Must be larger than residual_volume.",
              .validator = Validators::positive<double>(),
          }),
      one_of({
          group("normalized_linear_exponential",
              {
                  parameter<double>("pressure_offset",
                      {.description = "Pleural pressure at the residual volume, where the linear "
                                      "and the exponential term both vanish."}),
                  parameter<double>(
                      "linear_coefficient", {.description = "Factor of the linear term."}),
                  parameter<double>("exponential_coefficient",
                      {.description = "Factor of the exponential term."}),
                  parameter<double>(
                      "exponential_rate", {.description = "Rate inside the exponential term."}),
              },
              {
                  .description =
                      "p_pl(xi) = pressure_offset + linear_coefficient * xi + "
                      "exponential_coefficient * (exp(exponential_rate * xi) - 1), with the "
                      "normalized volume xi = (V - residual_volume) / (total_lung_capacity - "
                      "residual_volume).",
              }),
      }),
  });

  const auto store_pleural_pressure_definitions =
      StoreFunction<Core::IO::InputParameterContainer::List>(
          [](Storage& storage, Core::IO::InputParameterContainer::List&& value)
          {
            FOUR_C_ASSERT(storage.type() == typeid(BoundaryConditions),
                "Implementation error: expected BoundaryConditions storage.");

            auto& target =
                std::any_cast<BoundaryConditions&>(storage).volume_dependent_pleural_pressure;
            // The store also runs from set_default_value(), so it may see the same target twice.
            target.clear();
            for (const auto& entry : value)
            {
              const auto& curve = entry.group("normalized_linear_exponential");
              target.push_back(VolumeDependentPleuralPressureDefinition{.id = entry.get<int>("id"),
                  .coupling =
                      entry.get<VolumeDependentPleuralPressureDefinition::Coupling>("coupling"),
                  .residual_volume = entry.get<double>("residual_volume"),
                  .total_lung_capacity = entry.get<double>("total_lung_capacity"),
                  .normalized_linear_exponential =
                      VolumeDependentPleuralPressureDefinition::NormalizedLinearExponential{
                          .pressure_offset = curve.get<double>("pressure_offset"),
                          .linear_coefficient = curve.get<double>("linear_coefficient"),
                          .exponential_coefficient = curve.get<double>("exponential_coefficient"),
                          .exponential_rate = curve.get<double>("exponential_rate")}});
            }

            return StoreStatus::ok();
          },
          typeid(BoundaryConditions));

  Core::IO::InputSpec boundary_conditions_spec =
      group<ReducedLungParameters::BoundaryConditions>("boundary_conditions",
          {
              list("pressure", from_function_definition_spec,
                  {
                      .description = "Reusable pressure boundary condition definitions.",
                      .required = false,
                      .store = store_from_function_definitions(&BoundaryConditions::pressure),
                  }),
              list("flow", from_function_definition_spec,
                  {
                      .description = "Reusable volumetric flow boundary condition definitions.",
                      .required = false,
                      .store = store_from_function_definitions(&BoundaryConditions::flow),
                  }),
              list("volume_dependent_pleural_pressure", pleural_pressure_definition_spec,
                  {
                      .description =
                          "Reusable definitions of a pleural pressure that follows the total "
                          "volume of all terminal units. Every node carrying one of these ids "
                          "has its pressure dof constrained to the pleural pressure, evaluated "
                          "from the curve given below at the total terminal unit volume.",
                      .required = false,
                      .store = store_pleural_pressure_definitions,
                  }),
          },
          {
              .description = "Boundary conditions for the reduced lung tree.",
              .required = true,
              .store = in_struct(&ReducedLungParameters::boundary_conditions),
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
                  parameter<int>("max_nonlinear_iterations",
                      {
                          .description = "Maximum number of nonlinear iterations.",
                          .default_value = 10,
                          .store =
                              in_struct(&ReducedLungParameters::Dynamics::max_nonlinear_iterations),
                      }),
                  parameter<double>("nonlinear_residual_tolerance",
                      {
                          .description =
                              "Absolute residual norm tolerance for nonlinear convergence.",
                          .default_value = 1.0e-8,
                          .store = in_struct(
                              &ReducedLungParameters::Dynamics::nonlinear_residual_tolerance),
                      }),
                  parameter<double>("nonlinear_increment_tolerance",
                      {
                          .description =
                              "Absolute increment norm tolerance for nonlinear convergence.",
                          .default_value = 1.0e-10,
                          .store = in_struct(
                              &ReducedLungParameters::Dynamics::nonlinear_increment_tolerance),
                      }),
                  parameter<ReducedLungParameters::OutputVerbosity>("output_verbosity",
                      {
                          .description = "Output verbosity level.",
                          .default_value = ReducedLungParameters::OutputVerbosity::minimal,
                          .store = in_struct(&ReducedLungParameters::Dynamics::output_verbosity),
                      }),
              },
              {
                  .required = true,
                  .store = in_struct(&ReducedLungParameters::dynamics),
              }),

          geometry_spec,
          group<ReducedLungParameters::LungTree>("lung_tree",
              {
                  group<ReducedLungParameters::LungTree::Airways>("airways",
                      {parameter<std::vector<int>>("element_blocks",
                           {
                               .description =
                                   "Ids of the cell blocks of the mesh that hold the airway "
                                   "elements. Every cell block of the mesh must be claimed by "
                                   "exactly one element type.",
                               .default_value = std::vector<int>{},
                               .store = in_struct(
                                   &ReducedLungParameters::LungTree::Airways::element_blocks),
                           }),
                          input_field<double>("radius",
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
                      {parameter<std::vector<int>>("element_blocks",
                           {
                               .description =
                                   "Ids of the cell blocks of the mesh that hold the terminal unit "
                                   "elements. Every cell block of the mesh must be claimed by "
                                   "exactly one element type.",
                               .default_value = std::vector<int>{},
                               .store = in_struct(
                                   &ReducedLungParameters::LungTree::TerminalUnits::element_blocks),
                           }),
                          rheological_model_spec_terminal_unit,
                          elasticity_model_spec_terminal_units},
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
          boundary_conditions_spec,
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

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_reduced_lung_airways_wall_mechanics.hpp"

#include "4C_reduced_lung_helpers.hpp"

#include <array>
#include <cmath>
#include <type_traits>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace ReducedLung::Airways::WallMechanics
{
  void evaluate_rigid_wall_residual(Core::LinAlg::Vector<double>& target, const AirwayData& data,
      const Core::LinAlg::Vector<double>& locally_relevant_dofs,
      const std::vector<double>& resistance, const std::vector<double>& inertia, double dt)
  {
    for (size_t i = 0; i < data.number_of_elements(); i++)
    {
      double rigid_wall_residual =
          (locally_relevant_dofs.local_values_as_span()[data.lid_p1[i]] -
              locally_relevant_dofs.local_values_as_span()[data.lid_p2[i]] -
              resistance[i] * locally_relevant_dofs.local_values_as_span()[data.lid_q1[i]] -
              inertia[i] / dt *
                  (locally_relevant_dofs.local_values_as_span()[data.lid_q1[i]] - data.q1_n[i]));
      target.replace_local_value(data.local_row_id[i], rigid_wall_residual);
    }
  }

  void evaluate_kelvin_voigt_wall_residual(Core::LinAlg::Vector<double>& target,
      const KelvinVoigtWall& kelvin_voigt_wall_model, const AirwayData& data,
      const Core::LinAlg::Vector<double>& locally_relevant_dofs,
      const std::vector<double>& resistance, const std::vector<double>& inertia, double dt)
  {
    for (size_t i = 0; i < data.number_of_elements(); i++)
    {
      const int momentum_row = data.local_row_id[i];
      const int mass_row = data.local_row_id[i] + 1;

      double res_momentum = (locally_relevant_dofs.local_values_as_span()[data.lid_p1[i]] -
                             locally_relevant_dofs.local_values_as_span()[data.lid_p2[i]] -
                             (resistance[i] / 2 + inertia[i] / (2 * dt)) *
                                 (locally_relevant_dofs.local_values_as_span()[data.lid_q1[i]] +
                                     locally_relevant_dofs.local_values_as_span()[data.lid_q2[i]]) +
                             inertia[i] / (2 * dt) * (data.q1_n[i] + data.q2_n[i]));
      double res_mass = (locally_relevant_dofs.local_values_as_span()[data.lid_p1[i]] +
                         locally_relevant_dofs.local_values_as_span()[data.lid_p2[i]] -
                         data.p1_n[i] - data.p2_n[i] -
                         2 *
                             (kelvin_voigt_wall_model.viscous_resistance_Rvisc[i] +
                                 dt / kelvin_voigt_wall_model.compliance_C[i]) *
                             (locally_relevant_dofs.local_values_as_span()[data.lid_q1[i]] -
                                 locally_relevant_dofs.local_values_as_span()[data.lid_q2[i]]) +
                         2 * kelvin_voigt_wall_model.viscous_resistance_Rvisc[i] *
                             (data.q1_n[i] - data.q2_n[i]));
      target.replace_local_value(momentum_row, res_momentum);
      target.replace_local_value(mass_row, res_mass);
    }
  }

  void evaluate_jacobian_rigid_wall(Core::LinAlg::SparseMatrix& target, AirwayData const& data,
      const std::vector<double>& resistance_derivative,
      const std::vector<double>& inertia_derivative, double dt)
  {
    [[maybe_unused]] int err;
    std::array<int, 3> column_indices;
    std::array<double, 3> values;
    for (size_t i = 0; i < data.number_of_elements(); i++)
    {
      if (!target.filled())
      {
        column_indices = {data.lid_p1[i], data.lid_p2[i], data.lid_q1[i]};
        values = {1.0, -1.0, -resistance_derivative[i] - inertia_derivative[i]};
        target.insert_my_values(data.local_row_id[i], 3, values.data(), column_indices.data());
      }
      else
      {
        auto grad_q = -resistance_derivative[i] - inertia_derivative[i];
        target.replace_my_values(data.local_row_id[i], 1, &grad_q, &data.lid_q1[i]);
      }
    }
  }

  void evaluate_jacobian_kelvin_voigt_wall(Core::LinAlg::SparseMatrix& target,
      AirwayData const& data, const KelvinVoigtWall& kelvin_voigt_wall_model,
      const std::pair<std::vector<double>, std::vector<double>>& resistance_derivative,
      const std::pair<std::vector<double>, std::vector<double>>& inertia_derivative,
      const std::pair<std::vector<double>, std::vector<double>>& viscous_wall_resistance_derivative,
      double dt)
  {
    [[maybe_unused]] int err;
    for (size_t i = 0; i < data.number_of_elements(); i++)
    {
      const int momentum_row = data.local_row_id[i];
      const int mass_row = data.local_row_id[i] + 1;

      if (!target.filled())
      {
        std::array<int, 4> column_indices;
        std::array<double, 4> values;
        column_indices = {data.lid_p1[i], data.lid_p2[i], data.lid_q1[i], data.lid_q2[i]};
        values = {1.0, -1.0, resistance_derivative.first[i] + inertia_derivative.first[i],
            resistance_derivative.second[i] + inertia_derivative.second[i]};
        target.insert_my_values(momentum_row, 4, values.data(), column_indices.data());
        values = {1.0, 1.0, viscous_wall_resistance_derivative.first[i],
            viscous_wall_resistance_derivative.second[i]};
        target.insert_my_values(mass_row, 4, values.data(), column_indices.data());
      }
      else
      {
        std::array<int, 2> q_column_indices{data.lid_q1[i], data.lid_q2[i]};
        std::array<double, 2> grad_q{resistance_derivative.first[i] + inertia_derivative.first[i],
            resistance_derivative.second[i] + inertia_derivative.second[i]};
        target.replace_my_values(momentum_row, 2, grad_q.data(), q_column_indices.data());
        grad_q = {viscous_wall_resistance_derivative.first[i],
            viscous_wall_resistance_derivative.second[i]};
        target.replace_my_values(mass_row, 2, grad_q.data(), q_column_indices.data());
      }
    }
  }

  std::pair<std::vector<double>, std::vector<double>>
  evaluate_viscous_wall_resistance_derivative_kelvin_voigt(const KelvinVoigtWall& model,
      const AirwayData& data, const Core::LinAlg::Vector<double>& dofs, double dt)
  {
    std::vector<double> viscous_resistance_derivative_q1(data.number_of_elements());
    std::vector<double> viscous_resistance_derivative_q2(data.number_of_elements());

    for (size_t i = 0; i < data.number_of_elements(); i++)
    {
      double dRvisc_da =
          -model.gamma_w[i] / (2.0 * model.area[i] * std::sqrt(model.area[i]) * data.ref_length[i]);
      double da_dq1 = dt / data.ref_length[i];
      double da_dq2 = -dt / data.ref_length[i];
      double dCinv_da =
          -model.beta_w[i] / (4.0 * model.area[i] * std::sqrt(model.area[i]) * data.ref_length[i]);
      viscous_resistance_derivative_q1[i] =
          -2 * ((dRvisc_da * da_dq1 + dt * dCinv_da * da_dq1) *
                       (dofs.local_values_as_span()[data.lid_q1[i]] -
                           dofs.local_values_as_span()[data.lid_q2[i]]) +
                   (model.viscous_resistance_Rvisc[i] + dt / model.compliance_C[i]) -
                   dRvisc_da * da_dq1 * (data.q1_n[i] - data.q2_n[i]));
      viscous_resistance_derivative_q2[i] =
          -2 * ((dRvisc_da * da_dq2 + dt * dCinv_da * da_dq2) *
                       (dofs.local_values_as_span()[data.lid_q1[i]] -
                           dofs.local_values_as_span()[data.lid_q2[i]]) -
                   (model.viscous_resistance_Rvisc[i] + dt / model.compliance_C[i]) -
                   dRvisc_da * da_dq2 * (data.q1_n[i] - data.q2_n[i]));
    }
    return {viscous_resistance_derivative_q1, viscous_resistance_derivative_q2};
  }

  ResidualEvaluator make_residual_evaluator(WallModel& wall_model, FlowModel& flow_model)
  {
    return std::visit(
        [&flow_model](auto& wall_model_data) -> ResidualEvaluator
        {
          using WallModelType = std::decay_t<decltype(wall_model_data)>;
          if constexpr (std::is_same_v<WallModelType, RigidWall>)
          {
            auto resistance_evaluator =
                FlowResistance::make_flow_resistance_evaluator_rigid(flow_model);
            auto inertia_evaluator = FlowResistance::make_inertia_evaluator(flow_model);
            return [resistance_evaluator, inertia_evaluator](const AirwayData& airway_data,
                       Core::LinAlg::Vector<double>& target_vector,
                       const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt)
            {
              auto resistance =
                  resistance_evaluator(airway_data, locally_relevant_dofs, airway_data.ref_area);
              auto inertia = inertia_evaluator(airway_data, airway_data.ref_area);
              evaluate_rigid_wall_residual(
                  target_vector, airway_data, locally_relevant_dofs, resistance, inertia, dt);
            };
          }
          else if constexpr (std::is_same_v<WallModelType, KelvinVoigtWall>)
          {
            auto resistance_evaluator =
                FlowResistance::make_flow_resistance_evaluator_kelvin_voigt(flow_model);
            auto inertia_evaluator = FlowResistance::make_inertia_evaluator(flow_model);
            return [resistance_evaluator, inertia_evaluator, &wall_model_data](
                       const AirwayData& airway_data, Core::LinAlg::Vector<double>& target_vector,
                       const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt)
            {
              auto resistance =
                  resistance_evaluator(airway_data, locally_relevant_dofs, wall_model_data.area);
              auto inertia = inertia_evaluator(airway_data, wall_model_data.area);
              evaluate_kelvin_voigt_wall_residual(target_vector, wall_model_data, airway_data,
                  locally_relevant_dofs, resistance, inertia, dt);
            };
          }
          else
          {
            FOUR_C_THROW("Unknown airway wall model.");
          }
        },
        wall_model);
  }

  JacobianEvaluator make_jacobian_evaluator(WallModel& wall_model, FlowModel& flow_model)
  {
    return std::visit(
        [&flow_model](auto& wall_model_data) -> JacobianEvaluator
        {
          using WallModelType = std::decay_t<decltype(wall_model_data)>;
          if constexpr (std::is_same_v<WallModelType, RigidWall>)
          {
            auto resistance_derivative_evaluator =
                FlowResistance::make_flow_resistance_derivative_evaluator_rigid(flow_model);
            auto inertia_evaluator = FlowResistance::make_inertia_evaluator(flow_model);
            return [resistance_derivative_evaluator, inertia_evaluator](
                       const AirwayData& airway_data, Core::LinAlg::SparseMatrix& target,
                       const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt)
            {
              auto resistance_derivative =
                  resistance_derivative_evaluator(airway_data, locally_relevant_dofs, dt);
              auto inertia_derivative = [dt, &evaluator = inertia_evaluator, &data = airway_data]()
              {
                auto vals = evaluator(data, data.ref_area);
                for (auto& v : vals) v /= dt;
                return vals;
              }();
              evaluate_jacobian_rigid_wall(
                  target, airway_data, resistance_derivative, inertia_derivative, dt);
            };
          }
          else if constexpr (std::is_same_v<WallModelType, KelvinVoigtWall>)
          {
            auto resistance_derivative_evaluator =
                FlowResistance::make_flow_resistance_derivative_evaluator_kelvin_voigt(
                    flow_model, wall_model_data);
            auto inertia_derivative_evaluator =
                FlowResistance::make_inertia_derivative_evaluator_kelvin_voigt(
                    flow_model, wall_model_data);

            return
                [resistance_derivative_evaluator, inertia_derivative_evaluator, &wall_model_data](
                    const AirwayData& airway_data, Core::LinAlg::SparseMatrix& target,
                    const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt)
            {
              auto resistance_derivative =
                  resistance_derivative_evaluator(airway_data, locally_relevant_dofs, dt);
              auto inertia_derivative =
                  inertia_derivative_evaluator(airway_data, locally_relevant_dofs, dt);
              auto viscous_wall_resistance_derivative =
                  evaluate_viscous_wall_resistance_derivative_kelvin_voigt(
                      wall_model_data, airway_data, locally_relevant_dofs, dt);

              evaluate_jacobian_kelvin_voigt_wall(target, airway_data, wall_model_data,
                  resistance_derivative, inertia_derivative, viscous_wall_resistance_derivative,
                  dt);
            };
          }
          else
          {
            FOUR_C_THROW("Unknown airway wall model.");
          }
        },
        wall_model);
  }

  InternalStateUpdater make_internal_state_updater(
      WallModel& wall_model, FlowModelInternalStateUpdater flow_state_updater)
  {
    return std::visit(
        [flow_state_updater](auto& wall_model_data) -> InternalStateUpdater
        {
          using WallModelType = std::decay_t<decltype(wall_model_data)>;

          if constexpr (std::is_same_v<WallModelType, RigidWall>)
          {
            return [flow_state_updater](AirwayData& data,
                       const Core::LinAlg::Vector<double>& locally_relevant_dofs, double /*dt*/)
            { flow_state_updater(data, locally_relevant_dofs); };
          }
          else if constexpr (std::is_same_v<WallModelType, KelvinVoigtWall>)
          {
            return [flow_state_updater, &wall_model_data](AirwayData& data,
                       const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt)
            {
              flow_state_updater(data, locally_relevant_dofs);
              for (size_t i = 0; i < data.number_of_elements(); i++)
              {
                wall_model_data.area[i] =
                    wall_model_data.area_n[i] +
                    dt / data.ref_length[i] *
                        (locally_relevant_dofs.local_values_as_span()[data.lid_q1[i]] -
                            locally_relevant_dofs.local_values_as_span()[data.lid_q2[i]]);
                wall_model_data.beta_w[i] = std::sqrt(M_PI) * wall_model_data.wall_thickness[i] *
                                            wall_model_data.wall_elasticity[i] /
                                            ((1 - wall_model_data.wall_poisson_ratio[i] *
                                                      wall_model_data.wall_poisson_ratio[i]) *
                                                data.ref_area[i]);
                wall_model_data.gamma_w[i] =
                    wall_model_data.beta_w[i] * wall_model_data.viscous_time_constant[i] *
                    std::tan(wall_model_data.viscous_phase_shift[i]) / (4.0 * M_PI);
                wall_model_data.compliance_C[i] = 2 * std::sqrt(wall_model_data.area[i]) *
                                                  data.ref_length[i] / wall_model_data.beta_w[i];
                wall_model_data.viscous_resistance_Rvisc[i] =
                    wall_model_data.gamma_w[i] /
                    (std::sqrt(wall_model_data.area[i]) * data.ref_length[i]);
              }
            };
          }
          else
          {
            FOUR_C_THROW("Unknown airway wall model.");
          }
        },
        wall_model);
  }

  void append_wall_output(const RigidWall& /*wall_model_data*/, const AirwayData& /*data*/,
      RuntimeOutputCollector& /*collector*/, ReducedLungParameters::OutputVerbosity /*verbosity*/)
  {
  }

  void append_wall_output(const KelvinVoigtWall& wall_model_data, const AirwayData& data,
      RuntimeOutputCollector& collector, ReducedLungParameters::OutputVerbosity verbosity)
  {
    if (verbosity >= ReducedLungParameters::OutputVerbosity::medium)
    {
      auto& area = collector.get_or_create_vector("area");
      for (size_t i = 0; i < data.number_of_elements(); i++)
      {
        area.replace_local_value(data.local_element_id[i], wall_model_data.area[i]);
      }
    }
  }

  OutputEvaluator make_output_evaluator(WallModel& wall_model)
  {
    return std::visit(
        [](auto& wall_model_data) -> OutputEvaluator
        {
          return [&wall_model_data](const AirwayData& data, RuntimeOutputCollector& collector,
                     ReducedLungParameters::OutputVerbosity verbosity)
          { append_wall_output(wall_model_data, data, collector, verbosity); };
        },
        wall_model);
  }

  EndOfTimestepRoutine make_end_of_timestep_routine(WallModel& wall_model)
  {
    return std::visit(
        [](auto& wall_model_data) -> EndOfTimestepRoutine
        {
          using WallModelType = std::decay_t<decltype(wall_model_data)>;
          if constexpr (std::is_same_v<WallModelType, RigidWall>)
          {
            return [](AirwayData&, const Core::LinAlg::Vector<double>&, double) {};
          }
          else if constexpr (std::is_same_v<WallModelType, KelvinVoigtWall>)
          {
            return [&wall_model_data](AirwayData& data,
                       const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt)
            {
              for (size_t i = 0; i < data.number_of_elements(); i++)
              {
                data.q2_n[i] = locally_relevant_dofs.local_values_as_span()[data.lid_q2[i]];
                wall_model_data.area_n[i] =
                    wall_model_data.area_n[i] +
                    dt / data.ref_length[i] *
                        (locally_relevant_dofs.local_values_as_span()[data.lid_q1[i]] -
                            locally_relevant_dofs.local_values_as_span()[data.lid_q2[i]]);
              }
            };
          }
          else
          {
            FOUR_C_THROW("Unknown airway wall model.");
          }
        },
        wall_model);
  }

  void append_model_parameters(WallModel& wall_model, int global_element_id,
      const ReducedLungParameters::LungTree::Airways::WallModel& parameters, double ref_area)
  {
    std::visit(
        [&](auto& model)
        {
          using ModelType = std::decay_t<decltype(model)>;
          if constexpr (std::is_same_v<ModelType, RigidWall>)
          {
            return;
          }
          else if constexpr (std::is_same_v<ModelType, KelvinVoigtWall>)
          {
            model.wall_poisson_ratio.push_back(
                parameters.kelvin_voigt.elasticity.wall_poisson_ratio.at(
                    global_element_id, "wall_poisson_ratio"));
            model.wall_elasticity.push_back(parameters.kelvin_voigt.elasticity.wall_elasticity.at(
                global_element_id, "wall_elasticity"));
            model.wall_thickness.push_back(parameters.kelvin_voigt.elasticity.wall_thickness.at(
                global_element_id, "wall_thickness"));
            model.viscous_time_constant.push_back(
                parameters.kelvin_voigt.viscosity.viscous_time_constant.at(
                    global_element_id, "viscous_time_constant"));
            model.viscous_phase_shift.push_back(
                parameters.kelvin_voigt.viscosity.viscous_phase_shift.at(
                    global_element_id, "viscous_phase_shift"));
            model.area_n.push_back(ref_area);

            model.area.push_back(ref_area);
            model.beta_w.push_back(0.0);
            model.gamma_w.push_back(0.0);
            model.compliance_C.push_back(0.0);
            model.viscous_resistance_Rvisc.push_back(0.0);
          }
          else
          {
            FOUR_C_THROW("Unknown airway wall model.");
          }
        },
        wall_model);
  }
}  // namespace ReducedLung::Airways::WallMechanics

FOUR_C_NAMESPACE_CLOSE

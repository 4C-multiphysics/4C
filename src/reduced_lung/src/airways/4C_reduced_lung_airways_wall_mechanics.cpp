// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_reduced_lung_airways_wall_mechanics.hpp"

#include "4C_reduced_lung_helpers.hpp"
#include "4C_reduced_lung_tree_linear_solver.hpp"

#include <array>
#include <cmath>
#include <span>
#include <type_traits>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace ReducedLung::Airways::WallMechanics
{
  namespace
  {
    std::span<double> resize_scratch(std::vector<double>& scratch, size_t size)
    {
      scratch.resize(size);
      return std::span<double>(scratch.data(), scratch.size());
    }

    std::span<int> resize_scratch(std::vector<int>& scratch, size_t size)
    {
      scratch.resize(size);
      return std::span<int>(scratch.data(), scratch.size());
    }

    std::span<const double> as_const_span(const std::vector<double>& values)
    {
      return std::span<const double>(values.data(), values.size());
    }

  }  // namespace

  void evaluate_rigid_wall_residual(Core::LinAlg::Vector<double>& target, const AirwayData& data,
      const Core::LinAlg::Vector<double>& locally_relevant_dofs, std::span<const double> resistance,
      std::span<const double> inertia, double dt)
  {
    auto residual_values = target.local_values_as_span();
    const auto dof_values = locally_relevant_dofs.local_values_as_span();
    const auto& local_row_id = data.local_row_id;
    const auto& lid_p1 = data.lid_p1;
    const auto& lid_p2 = data.lid_p2;
    const auto& lid_q1 = data.lid_q1;
    const auto& q1_n = data.q1_n;
    for (size_t i = 0; i < data.number_of_elements(); i++)
    {
      const double q1 = dof_values[lid_q1[i]];
      double rigid_wall_residual = (dof_values[lid_p1[i]] - dof_values[lid_p2[i]] -
                                    resistance[i] * q1 - inertia[i] / dt * (q1 - q1_n[i]));
      residual_values[static_cast<std::size_t>(local_row_id[i])] = rigid_wall_residual;
    }
  }

  void evaluate_kelvin_voigt_wall_residual(Core::LinAlg::Vector<double>& target,
      const KelvinVoigtWall& kelvin_voigt_wall_model, const AirwayData& data,
      const Core::LinAlg::Vector<double>& locally_relevant_dofs, std::span<const double> resistance,
      std::span<const double> inertia, double dt)
  {
    auto residual_values = target.local_values_as_span();
    const auto dof_values = locally_relevant_dofs.local_values_as_span();
    const auto& local_row_id = data.local_row_id;
    const auto& lid_p1 = data.lid_p1;
    const auto& lid_p2 = data.lid_p2;
    const auto& lid_q1 = data.lid_q1;
    const auto& lid_q2 = data.lid_q2;
    const auto& p1_n = data.p1_n;
    const auto& p2_n = data.p2_n;
    const auto& q1_n = data.q1_n;
    const auto& q2_n = data.q2_n;
    const auto& viscous_resistance = kelvin_voigt_wall_model.viscous_resistance_Rvisc;
    const auto& compliance = kelvin_voigt_wall_model.compliance_C;
    for (size_t i = 0; i < data.number_of_elements(); i++)
    {
      const int momentum_row = local_row_id[i];
      const int mass_row = local_row_id[i] + 1;
      const double p1 = dof_values[lid_p1[i]];
      const double p2 = dof_values[lid_p2[i]];
      const double q1 = dof_values[lid_q1[i]];
      const double q2 = dof_values[lid_q2[i]];

      double res_momentum = (p1 - p2 - (resistance[i] / 2 + inertia[i] / (2 * dt)) * (q1 + q2) +
                             inertia[i] / (2 * dt) * (q1_n[i] + q2_n[i]));
      double res_mass = (p1 + p2 - p1_n[i] - p2_n[i] -
                         2 * (viscous_resistance[i] + dt / compliance[i]) * (q1 - q2) +
                         2 * viscous_resistance[i] * (q1_n[i] - q2_n[i]));
      residual_values[static_cast<std::size_t>(momentum_row)] = res_momentum;
      residual_values[static_cast<std::size_t>(mass_row)] = res_mass;
    }
  }

  void evaluate_jacobian_rigid_wall(Core::LinAlg::SparseMatrix& target, AirwayData const& data,
      std::span<const double> resistance_derivative, std::span<const double> inertia_derivative,
      double dt)
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
      std::span<const double> resistance_derivative_q1,
      std::span<const double> resistance_derivative_q2,
      std::span<const double> inertia_derivative_q1, std::span<const double> inertia_derivative_q2,
      std::span<const double> viscous_wall_resistance_derivative_q1,
      std::span<const double> viscous_wall_resistance_derivative_q2, double dt)
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
        values = {1.0, -1.0, resistance_derivative_q1[i] + inertia_derivative_q1[i],
            resistance_derivative_q2[i] + inertia_derivative_q2[i]};
        target.insert_my_values(momentum_row, 4, values.data(), column_indices.data());
        values = {1.0, 1.0, viscous_wall_resistance_derivative_q1[i],
            viscous_wall_resistance_derivative_q2[i]};
        target.insert_my_values(mass_row, 4, values.data(), column_indices.data());
      }
      else
      {
        std::array<int, 2> q_column_indices{data.lid_q1[i], data.lid_q2[i]};
        std::array<double, 2> grad_q{resistance_derivative_q1[i] + inertia_derivative_q1[i],
            resistance_derivative_q2[i] + inertia_derivative_q2[i]};
        target.replace_my_values(momentum_row, 2, grad_q.data(), q_column_indices.data());
        grad_q = {
            viscous_wall_resistance_derivative_q1[i], viscous_wall_resistance_derivative_q2[i]};
        target.replace_my_values(mass_row, 2, grad_q.data(), q_column_indices.data());
      }
    }
  }

  void evaluate_tree_linearization_rigid_wall(TreeNewtonLinearSolver& target,
      AirwayData const& data, std::span<double> resistance_derivative,
      std::span<const double> inertia_derivative)
  {
    /* Static setup inserted pressure entries and a q placeholder; only q changes with state. */
    FOUR_C_ASSERT_ALWAYS(resistance_derivative.size() == data.number_of_elements(),
        "Rigid airway tree coefficient buffer has {} entries but expected {}.",
        resistance_derivative.size(), data.number_of_elements());
    FOUR_C_ASSERT_ALWAYS(inertia_derivative.size() == data.number_of_elements(),
        "Rigid airway inertia derivative buffer has {} entries but expected {}.",
        inertia_derivative.size(), data.number_of_elements());
    for (size_t i = 0; i < data.number_of_elements(); i++)
    {
      resistance_derivative[i] = -resistance_derivative[i] - inertia_derivative[i];
    }
    target.set_values(data.local_row_id, data.lid_q1, resistance_derivative);
  }

  void evaluate_tree_linearization_kelvin_voigt_wall(TreeNewtonLinearSolver& target,
      AirwayData const& data, std::span<double> resistance_derivative_q1,
      std::span<double> resistance_derivative_q2, std::span<const double> inertia_derivative_q1,
      std::span<const double> inertia_derivative_q2,
      std::span<const double> viscous_wall_resistance_derivative_q1,
      std::span<const double> viscous_wall_resistance_derivative_q2, std::span<int> mass_row_id)
  {
    /* Combine derivative terms in scratch and set all dynamic q-coefficients in batches. */
    const size_t element_count = data.number_of_elements();
    FOUR_C_ASSERT_ALWAYS(resistance_derivative_q1.size() == element_count &&
                             resistance_derivative_q2.size() == element_count &&
                             inertia_derivative_q1.size() == element_count &&
                             inertia_derivative_q2.size() == element_count &&
                             viscous_wall_resistance_derivative_q1.size() == element_count &&
                             viscous_wall_resistance_derivative_q2.size() == element_count &&
                             mass_row_id.size() == element_count,
        "Kelvin-Voigt airway tree coefficient buffers must all have {} entries.", element_count);

    for (size_t i = 0; i < element_count; i++)
    {
      mass_row_id[i] = data.local_row_id[i] + 1;
      resistance_derivative_q1[i] += inertia_derivative_q1[i];
      resistance_derivative_q2[i] += inertia_derivative_q2[i];
    }

    target.set_values(data.local_row_id, data.lid_q1, resistance_derivative_q1);
    target.set_values(data.local_row_id, data.lid_q2, resistance_derivative_q2);
    target.set_values(mass_row_id, data.lid_q1, viscous_wall_resistance_derivative_q1);
    target.set_values(mass_row_id, data.lid_q2, viscous_wall_resistance_derivative_q2);
  }

  void initialize_rigid_wall_tree_linearization(
      TreeNewtonLinearSolver& target, const AirwayData& data)
  {
    /* Set the fixed rigid-airway row pattern once. */
    for (size_t i = 0; i < data.number_of_elements(); ++i)
    {
      target.set_value(data.local_row_id[i], data.lid_p1[i], 1.0);
      target.set_value(data.local_row_id[i], data.lid_p2[i], -1.0);
      target.set_value(data.local_row_id[i], data.lid_q1[i], 0.0);
    }
  }

  void initialize_kelvin_voigt_wall_tree_linearization(
      TreeNewtonLinearSolver& target, const AirwayData& data)
  {
    /* Kelvin-Voigt airways use two fixed rows; q entries are dynamic placeholders. */
    for (size_t i = 0; i < data.number_of_elements(); ++i)
    {
      const int momentum_row = data.local_row_id[i];
      const int mass_row = momentum_row + 1;

      target.set_value(momentum_row, data.lid_p1[i], 1.0);
      target.set_value(momentum_row, data.lid_p2[i], -1.0);
      target.set_value(momentum_row, data.lid_q1[i], 0.0);
      target.set_value(momentum_row, data.lid_q2[i], 0.0);

      target.set_value(mass_row, data.lid_p1[i], 1.0);
      target.set_value(mass_row, data.lid_p2[i], 1.0);
      target.set_value(mass_row, data.lid_q1[i], 0.0);
      target.set_value(mass_row, data.lid_q2[i], 0.0);
    }
  }

  void evaluate_viscous_wall_resistance_derivative_kelvin_voigt(const KelvinVoigtWall& model,
      const AirwayData& data, const Core::LinAlg::Vector<double>& dofs, double dt,
      std::span<double> viscous_resistance_derivative_q1,
      std::span<double> viscous_resistance_derivative_q2)
  {
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
            return [resistance_evaluator, inertia_evaluator, resistance = std::vector<double>{},
                       inertia = std::vector<double>{}](const AirwayData& airway_data,
                       Core::LinAlg::Vector<double>& target_vector,
                       const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt) mutable
            {
              const size_t element_count = airway_data.number_of_elements();
              resistance_evaluator(airway_data, locally_relevant_dofs, airway_data.ref_area,
                  resize_scratch(resistance, element_count));
              inertia_evaluator(
                  airway_data, airway_data.ref_area, resize_scratch(inertia, element_count));
              evaluate_rigid_wall_residual(target_vector, airway_data, locally_relevant_dofs,
                  as_const_span(resistance), as_const_span(inertia), dt);
            };
          }
          else if constexpr (std::is_same_v<WallModelType, KelvinVoigtWall>)
          {
            auto resistance_evaluator =
                FlowResistance::make_flow_resistance_evaluator_kelvin_voigt(flow_model);
            auto inertia_evaluator = FlowResistance::make_inertia_evaluator(flow_model);
            return [resistance_evaluator, inertia_evaluator, &wall_model_data,
                       resistance = std::vector<double>{}, inertia = std::vector<double>{}](
                       const AirwayData& airway_data, Core::LinAlg::Vector<double>& target_vector,
                       const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt) mutable
            {
              const size_t element_count = airway_data.number_of_elements();
              resistance_evaluator(airway_data, locally_relevant_dofs, wall_model_data.area,
                  resize_scratch(resistance, element_count));
              inertia_evaluator(
                  airway_data, wall_model_data.area, resize_scratch(inertia, element_count));
              evaluate_kelvin_voigt_wall_residual(target_vector, wall_model_data, airway_data,
                  locally_relevant_dofs, as_const_span(resistance), as_const_span(inertia), dt);
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
            return [resistance_derivative_evaluator, inertia_evaluator,
                       resistance_derivative = std::vector<double>{},
                       inertia_derivative = std::vector<double>{}](const AirwayData& airway_data,
                       Core::LinAlg::SparseMatrix& target,
                       const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt) mutable
            {
              const size_t element_count = airway_data.number_of_elements();
              resistance_derivative_evaluator(airway_data, locally_relevant_dofs, dt,
                  resize_scratch(resistance_derivative, element_count));
              inertia_evaluator(airway_data, airway_data.ref_area,
                  resize_scratch(inertia_derivative, element_count));
              for (auto& value : inertia_derivative)
              {
                value /= dt;
              }
              evaluate_jacobian_rigid_wall(target, airway_data,
                  as_const_span(resistance_derivative), as_const_span(inertia_derivative), dt);
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

            return [resistance_derivative_evaluator, inertia_derivative_evaluator, &wall_model_data,
                       resistance_derivative_q1 = std::vector<double>{},
                       resistance_derivative_q2 = std::vector<double>{},
                       inertia_derivative_q1 = std::vector<double>{},
                       inertia_derivative_q2 = std::vector<double>{},
                       viscous_wall_resistance_derivative_q1 = std::vector<double>{},
                       viscous_wall_resistance_derivative_q2 = std::vector<double>{}](
                       const AirwayData& airway_data, Core::LinAlg::SparseMatrix& target,
                       const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt) mutable
            {
              const size_t element_count = airway_data.number_of_elements();
              resistance_derivative_evaluator(airway_data, locally_relevant_dofs, dt,
                  resize_scratch(resistance_derivative_q1, element_count),
                  resize_scratch(resistance_derivative_q2, element_count));
              inertia_derivative_evaluator(airway_data, locally_relevant_dofs, dt,
                  resize_scratch(inertia_derivative_q1, element_count),
                  resize_scratch(inertia_derivative_q2, element_count));
              evaluate_viscous_wall_resistance_derivative_kelvin_voigt(wall_model_data, airway_data,
                  locally_relevant_dofs, dt,
                  resize_scratch(viscous_wall_resistance_derivative_q1, element_count),
                  resize_scratch(viscous_wall_resistance_derivative_q2, element_count));

              evaluate_jacobian_kelvin_voigt_wall(target, airway_data, wall_model_data,
                  as_const_span(resistance_derivative_q1), as_const_span(resistance_derivative_q2),
                  as_const_span(inertia_derivative_q1), as_const_span(inertia_derivative_q2),
                  as_const_span(viscous_wall_resistance_derivative_q1),
                  as_const_span(viscous_wall_resistance_derivative_q2), dt);
            };
          }
          else
          {
            FOUR_C_THROW("Unknown airway wall model.");
          }
        },
        wall_model);
  }

  StaticTreeLinearizationEvaluator make_static_tree_linearization_evaluator(WallModel& wall_model)
  {
    return std::visit(
        [](auto& wall_model_data) -> StaticTreeLinearizationEvaluator
        {
          using WallModelType = std::decay_t<decltype(wall_model_data)>;
          if constexpr (std::is_same_v<WallModelType, RigidWall>)
          {
            return [](const AirwayData& airway_data, TreeNewtonLinearSolver& target)
            { initialize_rigid_wall_tree_linearization(target, airway_data); };
          }
          else if constexpr (std::is_same_v<WallModelType, KelvinVoigtWall>)
          {
            return [](const AirwayData& airway_data, TreeNewtonLinearSolver& target)
            { initialize_kelvin_voigt_wall_tree_linearization(target, airway_data); };
          }
          else
          {
            FOUR_C_THROW("Unknown airway wall model.");
          }
        },
        wall_model);
  }

  TreeLinearizationEvaluator make_tree_linearization_evaluator(
      WallModel& wall_model, FlowModel& flow_model)
  {
    return std::visit(
        [&flow_model](auto& wall_model_data) -> TreeLinearizationEvaluator
        {
          using WallModelType = std::decay_t<decltype(wall_model_data)>;
          if constexpr (std::is_same_v<WallModelType, RigidWall>)
          {
            auto resistance_derivative_evaluator =
                FlowResistance::make_flow_resistance_derivative_evaluator_rigid(flow_model);
            auto inertia_evaluator = FlowResistance::make_inertia_evaluator(flow_model);
            return [resistance_derivative_evaluator, inertia_evaluator,
                       resistance_derivative = std::vector<double>{},
                       inertia_derivative = std::vector<double>{}](const AirwayData& airway_data,
                       TreeNewtonLinearSolver& target,
                       const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt) mutable
            {
              const size_t element_count = airway_data.number_of_elements();
              resistance_derivative_evaluator(airway_data, locally_relevant_dofs, dt,
                  resize_scratch(resistance_derivative, element_count));
              inertia_evaluator(airway_data, airway_data.ref_area,
                  resize_scratch(inertia_derivative, element_count));
              for (auto& value : inertia_derivative)
              {
                value /= dt;
              }
              evaluate_tree_linearization_rigid_wall(target, airway_data,
                  std::span<double>(resistance_derivative.data(), resistance_derivative.size()),
                  as_const_span(inertia_derivative));
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

            return [resistance_derivative_evaluator, inertia_derivative_evaluator, &wall_model_data,
                       resistance_derivative_q1 = std::vector<double>{},
                       resistance_derivative_q2 = std::vector<double>{},
                       inertia_derivative_q1 = std::vector<double>{},
                       inertia_derivative_q2 = std::vector<double>{},
                       viscous_wall_resistance_derivative_q1 = std::vector<double>{},
                       viscous_wall_resistance_derivative_q2 = std::vector<double>{},
                       mass_row_id = std::vector<int>{}](const AirwayData& airway_data,
                       TreeNewtonLinearSolver& target,
                       const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt) mutable
            {
              const size_t element_count = airway_data.number_of_elements();
              resistance_derivative_evaluator(airway_data, locally_relevant_dofs, dt,
                  resize_scratch(resistance_derivative_q1, element_count),
                  resize_scratch(resistance_derivative_q2, element_count));
              inertia_derivative_evaluator(airway_data, locally_relevant_dofs, dt,
                  resize_scratch(inertia_derivative_q1, element_count),
                  resize_scratch(inertia_derivative_q2, element_count));
              evaluate_viscous_wall_resistance_derivative_kelvin_voigt(wall_model_data, airway_data,
                  locally_relevant_dofs, dt,
                  resize_scratch(viscous_wall_resistance_derivative_q1, element_count),
                  resize_scratch(viscous_wall_resistance_derivative_q2, element_count));

              evaluate_tree_linearization_kelvin_voigt_wall(target, airway_data,
                  std::span<double>(
                      resistance_derivative_q1.data(), resistance_derivative_q1.size()),
                  std::span<double>(
                      resistance_derivative_q2.data(), resistance_derivative_q2.size()),
                  as_const_span(inertia_derivative_q1), as_const_span(inertia_derivative_q2),
                  as_const_span(viscous_wall_resistance_derivative_q1),
                  as_const_span(viscous_wall_resistance_derivative_q2),
                  resize_scratch(mass_row_id, element_count));
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

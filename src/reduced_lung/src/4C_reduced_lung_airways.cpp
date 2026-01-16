// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_reduced_lung_airways.hpp"

#include <cmath>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace ReducedLung
{
  namespace Airways
  {
    void evaluate_negative_rigid_wall_residual(Core::LinAlg::Vector<double>& target,
        const AirwayData& data, const Core::LinAlg::Vector<double>& locally_relevant_dofs,
        const std::vector<double>& resistance, const std::vector<double>& inertia, double dt)
    {
      for (size_t i = 0; i < data.number_of_elements(); i++)
      {
        double negative_rigid_wall_residual =
            -1 *
            (locally_relevant_dofs.local_values_as_span()[data.lid_p1[i]] -
                locally_relevant_dofs.local_values_as_span()[data.lid_p2[i]] -
                resistance[i] * locally_relevant_dofs.local_values_as_span()[data.lid_q1[i]] -
                inertia[i] / dt *
                    (locally_relevant_dofs.local_values_as_span()[data.lid_q1[i]] - data.q1_n[i]));
        target.replace_local_value(data.local_row_id[i], negative_rigid_wall_residual);
      }
    }

    void evaluate_negative_kelvin_voigt_wall_residual(Core::LinAlg::Vector<double>& target,
        const KelvinVoigtWall& kelvin_voigt_wall_model, const AirwayData& data,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs,
        const std::vector<double>& resistance, const std::vector<double>& inertia, double dt)
    {
      for (size_t i = 0; i < data.number_of_elements(); i++)
      {
        double neg_res_momentum =
            -1 * (locally_relevant_dofs.local_values_as_span()[data.lid_p1[i]] -
                     locally_relevant_dofs.local_values_as_span()[data.lid_p2[i]] -
                     (resistance[i] / 2 + inertia[i] / (2 * dt)) *
                         (locally_relevant_dofs.local_values_as_span()[data.lid_q1[i]] +
                             locally_relevant_dofs.local_values_as_span()[data.lid_q2[i]]) +
                     inertia[i] / (2 * dt) * (data.q1_n[i] + data.q2_n[i]));
        double neg_res_mass =
            -1 * (locally_relevant_dofs.local_values_as_span()[data.lid_p1[i]] +
                     locally_relevant_dofs.local_values_as_span()[data.lid_p2[i]] - data.p1_n[i] -
                     data.p2_n[i] -
                     2 *
                         (kelvin_voigt_wall_model.viscous_resistance_Rvisc[i] +
                             dt / kelvin_voigt_wall_model.compliance_C[i]) *
                         (locally_relevant_dofs.local_values_as_span()[data.lid_q1[i]] -
                             locally_relevant_dofs.local_values_as_span()[data.lid_q2[i]]) +
                     2 * kelvin_voigt_wall_model.viscous_resistance_Rvisc[i] *
                         (data.q1_n[i] - data.q2_n[i]));  // Pext component not implemented yet.
        target.replace_local_value(data.local_row_id[i], neg_res_momentum);
        target.replace_local_value(
            static_cast<int>(data.local_row_id[i] + data.number_of_elements()), neg_res_mass);
      }
    }

    void evaluate_jacobian_rigid_wall(Core::LinAlg::SparseMatrix& target, AirwayData const& data,
        const std::vector<double>& resistance_derivative,
        const std::vector<double>& inertia_derivative, double dt)
    {
      [[maybe_unused]] int err;  // Saves error code of trilinos functions.
      std::array<int, 3> column_indices;
      std::array<double, 3> values;
      for (size_t i = 0; i < data.number_of_elements(); i++)
      {
        column_indices = {data.lid_p1[i], data.lid_p2[i], data.lid_q1[i]};
        values = {1.0, -1.0, -resistance_derivative[i] - inertia_derivative[i]};
        target.insert_my_values(data.local_row_id[i], 3, values.data(), column_indices.data());
      }
    }

    std::vector<double> evaluate_nonlinear_flow_resistance_derivative_rigid(
        const NonLinearResistive& model, const AirwayData& data,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt)
    {
      auto poiseuille_resistance = ComputePoiseuilleResistance{}(data, data.ref_area);
      std::vector<double> resistance_derivative(data.number_of_elements());
      for (size_t i = 0; i < data.number_of_elements(); i++)
      {
        double R = poiseuille_resistance[i] * model.k_turb[i];

        double dk_dq1;
        if (model.k_turb[i] > 1.0)
        {
          dk_dq1 =
              model.turbulence_factor_gamma[i] *
              std::sqrt(data.air_properties.density /
                        (M_PI * data.air_properties.dynamic_viscosity * data.ref_length[i])) *
              1 / std::sqrt(std::abs(locally_relevant_dofs.local_values_as_span()[data.lid_q1[i]]));
        }
        else
        {
          dk_dq1 = 0.0;
        }
        resistance_derivative[i] =
            (poiseuille_resistance[i] * dk_dq1) *
                locally_relevant_dofs.local_values_as_span()[data.lid_q1[i]] +
            R;
      }
      return resistance_derivative;
    }

    void evaluate_jacobian_kelvin_voigt_wall(Core::LinAlg::SparseMatrix& target,
        AirwayData const& data, const KelvinVoigtWall& kelvin_voigt_wall_model,
        const std::pair<std::vector<double>, std::vector<double>>& resistance_derivative,
        const std::pair<std::vector<double>, std::vector<double>>& inertia_derivative,
        const std::pair<std::vector<double>, std::vector<double>>&
            viscous_wall_resistance_derivative,
        double dt)
    {
      [[maybe_unused]] int err;  // Saves error code of trilinos functions.
      std::array<int, 4> column_indices;
      std::array<double, 4> values;
      for (size_t i = 0; i < data.number_of_elements(); i++)
      {
        column_indices = {data.lid_p1[i], data.lid_p2[i], data.lid_q1[i], data.lid_q2[i]};
        values = {1.0, -1.0, resistance_derivative.first[i] + inertia_derivative.first[i],
            resistance_derivative.second[i] + inertia_derivative.second[i]};
        target.insert_my_values(data.local_row_id[i], 4, values.data(), column_indices.data());
        values = {1.0, 1.0, viscous_wall_resistance_derivative.first[i],
            viscous_wall_resistance_derivative.second[i]};
        target.insert_my_values(static_cast<int>(data.local_row_id[i] + data.number_of_elements()),
            4, values.data(), column_indices.data());
      }
    }

    std::pair<std::vector<double>, std::vector<double>>
    evaluate_linear_flow_resistance_derivative_kelvin_voigt(const LinearResistive& model,
        const AirwayData& data, const Core::LinAlg::Vector<double>& dofs, std::vector<double>& area,
        double dt)
    {
      auto poiseuille_resistance = ComputePoiseuilleResistance{}(data, area);
      std::vector<double> resistance_derivative_q1(data.number_of_elements());
      std::vector<double> resistance_derivative_q2(data.number_of_elements());
      for (size_t i = 0; i < data.number_of_elements(); i++)
      {
        double dRp_da = -16 * M_PI * data.air_properties.dynamic_viscosity * data.ref_length[i] /
                        (area[i] * area[i] * area[i]);
        double da_dq1 = dt / data.ref_length[i];
        double da_dq2 = -dt / data.ref_length[i];
        resistance_derivative_q1[i] =
            -0.5 * (dRp_da * da_dq1 *
                           (dofs.local_values_as_span()[data.lid_q1[i]] +
                               dofs.local_values_as_span()[data.lid_q2[i]]) +
                       poiseuille_resistance[i]);
        resistance_derivative_q2[i] =
            -0.5 * (dRp_da * da_dq2 *
                           (dofs.local_values_as_span()[data.lid_q1[i]] +
                               dofs.local_values_as_span()[data.lid_q2[i]]) +
                       poiseuille_resistance[i]);
      }
      return {resistance_derivative_q1, resistance_derivative_q2};
    }

    std::pair<std::vector<double>, std::vector<double>>
    evaluate_nonlinear_flow_resistance_derivative_kelvin_voigt(const NonLinearResistive& model,
        const AirwayData& data, const Core::LinAlg::Vector<double>& dofs, std::vector<double>& area,
        double dt)
    {
      auto poiseuille_resistance = ComputePoiseuilleResistance{}(data, area);
      std::vector<double> resistance_derivative_q1(data.number_of_elements());
      std::vector<double> resistance_derivative_q2(data.number_of_elements());
      for (size_t i = 0; i < data.number_of_elements(); i++)
      {
        double dRp_da = -16 * M_PI * data.air_properties.dynamic_viscosity * data.ref_length[i] /
                        (area[i] * area[i] * area[i]);
        double da_dq1 = dt / data.ref_length[i];
        double da_dq2 = -dt / data.ref_length[i];
        double dk_dq1;
        if (model.k_turb[i] > 1.0)
        {
          dk_dq1 =
              model.turbulence_factor_gamma[i] *
              std::sqrt(data.air_properties.density /
                        (2 * M_PI * data.air_properties.dynamic_viscosity * data.ref_length[i])) *
              (dofs.local_values_as_span()[data.lid_q1[i]] +
                  dofs.local_values_as_span()[data.lid_q2[i]]) /
              (std::abs(dofs.local_values_as_span()[data.lid_q1[i]] +
                        dofs.local_values_as_span()[data.lid_q2[i]]) *
                  std::sqrt(std::abs(
                      dofs.local_values_as_span()[data.lid_q1[i]] +
                      dofs.local_values_as_span()[data.lid_q2[i]])));  // (|q1+q2|^1.5) not using
                                                                       // std::pow for performance
        }
        else
        {
          dk_dq1 = 0.0;
        }
        double dk_dq2 = dk_dq1;
        double dalpha_dk = -4.0 / ((4.0 * model.k_turb[i] - 1.0) * (4.0 * model.k_turb[i] - 1.0));
        double alpha = 4.0 * model.k_turb[i] / (4.0 * model.k_turb[i] - 1.0);

        double resistance = poiseuille_resistance[i] * model.k_turb[i] +
                            2 * data.air_properties.density * alpha / (area[i] * area[i]) *
                                (dofs.local_values_as_span()[data.lid_q2[i]] -
                                    dofs.local_values_as_span()[data.lid_q1[i]]);
        resistance_derivative_q1[i] =
            -0.5 *
            ((dRp_da * da_dq1 * model.k_turb[i] + poiseuille_resistance[i] * dk_dq1 +
                 2 * data.air_properties.density *
                     (dofs.local_values_as_span()[data.lid_q2[i]] -
                         dofs.local_values_as_span()[data.lid_q1[i]]) /
                     (area[i] * area[i]) * dalpha_dk * dk_dq1 -  // <-- careful with sign here
                 2 * data.air_properties.density * alpha / (area[i] * area[i]) -
                 4 * data.air_properties.density * alpha / (area[i] * area[i] * area[i]) * da_dq1 *
                     (dofs.local_values_as_span()[data.lid_q2[i]] -
                         dofs.local_values_as_span()[data.lid_q1[i]])) *
                    (dofs.local_values_as_span()[data.lid_q1[i]] +
                        dofs.local_values_as_span()[data.lid_q2[i]]) +
                resistance);
        resistance_derivative_q2[i] =
            -0.5 *
            ((dRp_da * da_dq2 * model.k_turb[i] + poiseuille_resistance[i] * dk_dq2 +
                 2 * data.air_properties.density *
                     (dofs.local_values_as_span()[data.lid_q2[i]] -
                         dofs.local_values_as_span()[data.lid_q1[i]]) /
                     (area[i] * area[i]) * dalpha_dk * dk_dq2 +  // <-- careful with sign here
                 2 * data.air_properties.density * alpha / (area[i] * area[i]) -
                 4 * data.air_properties.density * alpha / (area[i] * area[i] * area[i]) * da_dq2 *
                     (dofs.local_values_as_span()[data.lid_q2[i]] -
                         dofs.local_values_as_span()[data.lid_q1[i]])) *
                    (dofs.local_values_as_span()[data.lid_q1[i]] +
                        dofs.local_values_as_span()[data.lid_q2[i]]) +
                resistance);
      }
      return {resistance_derivative_q1, resistance_derivative_q2};
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
            model.viscous_time_constant[i] * std::tan(model.viscous_phase_shift[i]) /
            (8.0 * M_PI * model.area[i] * std::sqrt(model.area[i]) * data.ref_length[i]) *
            model.beta_w[i];
        double da_dq1 = dt / data.ref_length[i];
        double da_dq2 = -dt / data.ref_length[i];
        double dCinv_da = -model.beta_w[i] / (4.0 * data.ref_area[i] * std::sqrt(data.ref_area[i]) *
                                                 data.ref_length[i]);
        viscous_resistance_derivative_q1[i] =
            -2 * ((dRvisc_da * da_dq1 + dt * dCinv_da * da_dq1) *
                         (dofs.local_values_as_span()[data.lid_q1[i]] -
                             dofs.local_values_as_span()[data.lid_q2[i]]) +  // <-- careful with
                                                                             // sign here
                     (model.viscous_resistance_Rvisc[i] + dt / model.compliance_C[i]) -
                     dRvisc_da * da_dq1 * (data.q1_n[i] - data.q2_n[i]));
        viscous_resistance_derivative_q2[i] =
            -2 * ((dRvisc_da * da_dq2 + dt * dCinv_da * da_dq2) *
                         (dofs.local_values_as_span()[data.lid_q1[i]] -
                             dofs.local_values_as_span()[data.lid_q2[i]]) -  // <-- careful with
                                                                             // sign here
                     (model.viscous_resistance_Rvisc[i] + dt / model.compliance_C[i]) -
                     dRvisc_da * da_dq2 * (data.q1_n[i] - data.q2_n[i]));
      }
      return {viscous_resistance_derivative_q1, viscous_resistance_derivative_q2};
    }


    std::pair<std::vector<double>, std::vector<double>> evaluate_inertia_derivative_kelvin_voigt(
        const std::vector<bool>& has_inertia, const AirwayData& data,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs, const KelvinVoigtWall& wall,
        double dt)
    {
      std::vector<double> inertia_q1(data.number_of_elements(), 0.0);
      std::vector<double> inertia_q2(data.number_of_elements(), 0.0);
      for (size_t i = 0; i < data.number_of_elements(); ++i)
      {
        if (i < has_inertia.size() && has_inertia[i])
        {
          double dI_da =
              -data.air_properties.density * data.ref_length[i] / (wall.area[i] * wall.area[i]);
          double da_dq1 = dt / data.ref_length[i];
          double da_dq2 = -dt / data.ref_length[i];
          inertia_q1[i] = -0.5 / dt *
                          (data.air_properties.density * data.ref_length[i] / wall.area[i] +
                              dI_da * da_dq1 *
                                  (locally_relevant_dofs.local_values_as_span()[data.lid_q1[i]] +
                                      locally_relevant_dofs.local_values_as_span()[data.lid_q2[i]] -
                                      data.q1_n[i] - data.q2_n[i]));

          inertia_q2[i] = -0.5 / dt *
                          (data.air_properties.density * data.ref_length[i] / wall.area[i] +
                              dI_da * da_dq2 *
                                  (locally_relevant_dofs.local_values_as_span()[data.lid_q1[i]] +
                                      locally_relevant_dofs.local_values_as_span()[data.lid_q2[i]] -
                                      data.q1_n[i] - data.q2_n[i]));
        }
      }
      return {inertia_q1, inertia_q2};
    }
    void update_jacobian(Core::LinAlg::SparseMatrix& jac, AirwayContainer& airways,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt)
    {
      for (auto& model : airways.models)
      {
        model.jacobian_evaluator(model.data, jac, locally_relevant_dofs, dt);
      }
    }

    void update_negative_residual_vector(Core::LinAlg::Vector<double>& res_vector,
        AirwayContainer& airways, const Core::LinAlg::Vector<double>& locally_relevant_dofs,
        double dt)
    {
      for (auto& model : airways.models)
      {
        model.negative_residual_evaluator(model.data, res_vector, locally_relevant_dofs, dt);
      }
    }


    void update_internal_state_vectors(AirwayContainer& airways,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt)
    {
      for (auto& model : airways.models)
      {
        // Model specific updates
        model.internal_state_updater(model.data, locally_relevant_dofs, dt);
      }
    }

    void end_of_timestep_routine(AirwayContainer& airways,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt)
    {
      for (auto& model : airways.models)
      {
        for (size_t i = 0; i < model.data.number_of_elements(); i++)
        {
          model.data.p1_n[i] = locally_relevant_dofs.local_values_as_span()[model.data.lid_p1[i]];
          model.data.p2_n[i] = locally_relevant_dofs.local_values_as_span()[model.data.lid_p2[i]];
          model.data.q1_n[i] = locally_relevant_dofs.local_values_as_span()[model.data.lid_q1[i]];
          if (model.data.n_state_equations == 2)
          {
            model.data.q2_n[i] = locally_relevant_dofs.local_values_as_span()[model.data.lid_q2[i]];
          }

          model.end_of_timestep_routine(model.data, locally_relevant_dofs, dt);
        }
      }
    }

    void create_evaluators(AirwayContainer& airways)
    {
      for (auto& model : airways.models)
      {
        model.negative_residual_evaluator =
            std::visit(MakeNegativeResidualEvaluator{model.flow_model}, model.wall_model);
        model.jacobian_evaluator =
            std::visit(MakeJacobianEvaluator{model.flow_model}, model.wall_model);
        model.internal_state_updater =
            std::visit(MakeInternalStateUpdater{model.flow_model}, model.wall_model);
        model.end_of_timestep_routine = std::visit(MakeEndOfTimestepRoutine{}, model.wall_model);
      }
    }
  }  // namespace Airways
}  // namespace ReducedLung

FOUR_C_NAMESPACE_CLOSE
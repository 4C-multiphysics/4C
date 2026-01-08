// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_reduced_lung_terminal_unit.hpp"

FOUR_C_NAMESPACE_OPEN

namespace ReducedLung
{
  namespace TerminalUnits
  {
    std::vector<double>& evaluate_linear_elastic_pressure(LinearElasticity& linear_elastic_model,
        TerminalUnitData& data, const Core::LinAlg::Vector<double>& locally_relevant_dofs,
        double dt)
    {
      for (size_t i = 0; i < data.number_of_elements(); i++)
      {
        linear_elastic_model.elastic_pressure_p_el[i] =
            linear_elastic_model.elasticity_E[i] *
            ((data.volume_v[i] + dt * locally_relevant_dofs.local_values_as_span()[data.lid_q[i]]) /
                    data.reference_volume_v0[i] -
                1);
      }
      return linear_elastic_model.elastic_pressure_p_el;
    }

    std::vector<double>& evaluate_ogden_hyperelastic_pressure(
        OgdenHyperelasticity& ogden_hyperelastic_model, TerminalUnitData& data,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt)
    {
      for (size_t i = 0; i < data.number_of_elements(); i++)
      {
        double v0_over_vi =
            data.reference_volume_v0[i] /
            (data.volume_v[i] + dt * locally_relevant_dofs.local_values_as_span()[data.lid_q[i]]);
        ogden_hyperelastic_model.elastic_pressure_p_el[i] =
            ogden_hyperelastic_model.bulk_modulus_kappa[i] /
            ogden_hyperelastic_model.nonlinear_stiffening_beta[i] * v0_over_vi *
            (1 - std::pow(v0_over_vi, ogden_hyperelastic_model.nonlinear_stiffening_beta[i]));
      }
      return ogden_hyperelastic_model.elastic_pressure_p_el;
    }

    void evaluate_negative_kelvin_voigt_residual(Core::LinAlg::Vector<double>& target,
        const KelvinVoigt& kelvin_voigt_model, const TerminalUnitData& data,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs,
        const std::vector<double>& elastic_pressure_p_el)
    {
      for (size_t i = 0; i < data.number_of_elements(); i++)
      {
        double negative_kelvin_voigt_residual =
            -1 * (locally_relevant_dofs.local_values_as_span()[data.lid_p1[i]] -
                     locally_relevant_dofs.local_values_as_span()[data.lid_p2[i]] -
                     elastic_pressure_p_el[i] -
                     kelvin_voigt_model.viscosity_eta[i] *
                         locally_relevant_dofs.local_values_as_span()[data.lid_q[i]] /
                         data.reference_volume_v0[i]);
        target.replace_local_value(data.local_row_id[i], negative_kelvin_voigt_residual);
      }
    }

    void evaluate_negative_four_element_maxwell_residual(Core::LinAlg::Vector<double>& target,
        const FourElementMaxwell& four_element_maxwell_model, const TerminalUnitData& data,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs,
        const std::vector<double>& elastic_pressure_p_el, double dt)
    {
      for (size_t i = 0; i < data.number_of_elements(); i++)
      {
        double negative_four_element_maxwell_residual =
            -1 * (locally_relevant_dofs.local_values_as_span()[data.lid_p1[i]] -
                     locally_relevant_dofs.local_values_as_span()[data.lid_p2[i]] -
                     elastic_pressure_p_el[i] -
                     (four_element_maxwell_model.viscosity_eta[i] +
                         (four_element_maxwell_model.elasticity_E_m[i] * dt *
                             four_element_maxwell_model.viscosity_eta_m[i]) /
                             (four_element_maxwell_model.elasticity_E_m[i] * dt +
                                 four_element_maxwell_model.viscosity_eta_m[i])) /
                         data.reference_volume_v0[i] *
                         locally_relevant_dofs.local_values_as_span()[data.lid_q[i]] -
                     four_element_maxwell_model.viscosity_eta_m[i] /
                         (four_element_maxwell_model.elasticity_E_m[i] * dt +
                             four_element_maxwell_model.viscosity_eta_m[i]) *
                         four_element_maxwell_model.maxwell_pressure_p_m[i]);
        target.replace_local_value(data.local_row_id[i], negative_four_element_maxwell_residual);
      }
    }

    std::vector<double>& linear_elastic_pressure_gradient(
        LinearElasticity& linear_elastic_model, TerminalUnitData& data, double dt)
    {
      for (size_t i = 0; i < data.number_of_elements(); i++)
      {
        linear_elastic_model.elastic_pressure_grad_dp_el[i] =
            linear_elastic_model.elasticity_E[i] * dt / data.reference_volume_v0[i];
      }
      return linear_elastic_model.elastic_pressure_grad_dp_el;
    }

    std::vector<double>& ogden_hyperelastic_pressure_gradient(
        OgdenHyperelasticity& ogden_hyperelastic_model, TerminalUnitData& data,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt)
    {
      for (size_t i = 0; i < data.number_of_elements(); i++)
      {
        double v0_over_vi =
            data.reference_volume_v0[i] /
            (data.volume_v[i] + dt * locally_relevant_dofs.local_values_as_span()[data.lid_q[i]]);
        ogden_hyperelastic_model.elastic_pressure_grad_dp_el[i] =
            ogden_hyperelastic_model.bulk_modulus_kappa[i] * dt /
            (ogden_hyperelastic_model.nonlinear_stiffening_beta[i] * data.reference_volume_v0[i]) *
            v0_over_vi * v0_over_vi *
            ((ogden_hyperelastic_model.nonlinear_stiffening_beta[i] + 1) *
                    std::pow(v0_over_vi, ogden_hyperelastic_model.nonlinear_stiffening_beta[i]) -
                1);
      }
      return ogden_hyperelastic_model.elastic_pressure_grad_dp_el;
    }

    void evaluate_kelvin_voigt_jacobian(Core::LinAlg::SparseMatrix& target,
        const KelvinVoigt& kelvin_voigt_model, TerminalUnitData& data,
        const std::vector<double>& elastic_pressure_grad_dp_el)
    {
      if (!target.filled())
      {
        for (size_t i = 0; i < data.number_of_elements(); i++)
        {
          std::array<int, 3> column_indices{data.lid_p1[i], data.lid_p2[i], data.lid_q[i]};
          std::array<double, 3> values{1.0, -1.0,
              -elastic_pressure_grad_dp_el[i] -
                  kelvin_voigt_model.viscosity_eta[i] / data.reference_volume_v0[i]};
          target.insert_my_values(data.local_row_id[i], 3, values.data(), column_indices.data());
        }
      }
      else
      {
        for (size_t i = 0; i < data.number_of_elements(); i++)
        {
          double grad_q = -elastic_pressure_grad_dp_el[i] -
                          kelvin_voigt_model.viscosity_eta[i] / data.reference_volume_v0[i];
          target.replace_my_values(data.local_row_id[i], 1, &grad_q, &data.lid_q[i]);
        }
      }
    }

    void evaluate_four_element_maxwell_jacobian(Core::LinAlg::SparseMatrix& target,
        const FourElementMaxwell& four_element_maxwell_model, TerminalUnitData& data,
        const std::vector<double>& elastic_pressure_grad_dp_el, double dt)
    {
      if (!target.filled())
      {
        for (size_t i = 0; i < data.number_of_elements(); i++)
        {
          std::array<int, 3> column_indices{data.lid_p1[i], data.lid_p2[i], data.lid_q[i]};
          std::array<double, 3> values{1.0, -1.0,
              -elastic_pressure_grad_dp_el[i] -
                  (four_element_maxwell_model.viscosity_eta[i] +
                      four_element_maxwell_model.elasticity_E_m[i] * dt *
                          four_element_maxwell_model.viscosity_eta_m[i] /
                          (four_element_maxwell_model.elasticity_E_m[i] * dt +
                              four_element_maxwell_model.viscosity_eta_m[i])) /
                      data.reference_volume_v0[i]};
          target.insert_my_values(data.local_row_id[i], 3, values.data(), column_indices.data());
        }
      }
      else
      {
        for (size_t i = 0; i < data.number_of_elements(); i++)
        {
          double grad_q = -elastic_pressure_grad_dp_el[i] -
                          (four_element_maxwell_model.viscosity_eta[i] +
                              four_element_maxwell_model.elasticity_E_m[i] * dt *
                                  four_element_maxwell_model.viscosity_eta_m[i] /
                                  (four_element_maxwell_model.elasticity_E_m[i] * dt +
                                      four_element_maxwell_model.viscosity_eta_m[i])) /
                              data.reference_volume_v0[i];
          target.replace_my_values(data.local_row_id[i], 1, &grad_q, &data.lid_q[i]);
        }
      }
    }

    void update_negative_residual_vector(Core::LinAlg::Vector<double>& res_vector,
        TerminalUnitContainer& terminal_units,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt)
    {
      for (auto& model : terminal_units.models)
      {
        model.negative_residual_evaluator(model.data, res_vector, locally_relevant_dofs, dt);
      }
    }

    void update_jacobian(Core::LinAlg::SparseMatrix& jac, TerminalUnitContainer& terminal_units,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt)
    {
      for (auto& model : terminal_units.models)
      {
        model.jacobian_evaluator(model.data, jac, locally_relevant_dofs, dt);
      }
    }

    void update_internal_state_vectors(TerminalUnitContainer& terminal_units,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt)
    {
      for (auto& model : terminal_units.models)
      {
        model.internal_state_updater(model.data, locally_relevant_dofs, dt);
      }
    }

    void end_of_timestep_routine(TerminalUnitContainer& terminal_units,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt)
    {
      for (auto& model : terminal_units.models)
      {
        // Update volume
        for (size_t i = 0; i < model.data.number_of_elements(); i++)
        {
          model.data.volume_v[i] +=
              locally_relevant_dofs.local_values_as_span()[model.data.lid_q[i]] * dt;
        }
        // Model specific updates
        model.end_of_timestep_routine(model.data, locally_relevant_dofs, dt);
      }
    }

    void create_evaluators(TerminalUnitContainer& terminal_units)
    {
      for (auto& model : terminal_units.models)
      {
        model.negative_residual_evaluator = std::visit(
            MakeNegativeResidualEvaluator{model.elasticity_model}, model.rheological_model);
        model.jacobian_evaluator =
            std::visit(MakeJacobianEvaluator{model.elasticity_model}, model.rheological_model);
        model.internal_state_updater =
            std::visit(MakeInternalStateUpdater{}, model.rheological_model);
        model.end_of_timestep_routine =
            std::visit(MakeEndOfTimestepRoutine{}, model.rheological_model);
      }
    }
  }  // namespace TerminalUnits
}  // namespace ReducedLung


FOUR_C_NAMESPACE_CLOSE
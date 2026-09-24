// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_reduced_lung_terminal_unit_rheology.hpp"

#include "4C_reduced_lung_helpers.hpp"
#include "4C_reduced_lung_tree_linearization.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <span>
#include <utility>

FOUR_C_NAMESPACE_OPEN

namespace ReducedLung::TerminalUnits::Rheology
{
  namespace
  {
    std::span<double> resize_scratch(std::vector<double>& scratch, size_t size)
    {
      scratch.resize(size);
      return std::span<double>(scratch.data(), scratch.size());
    }

    bool all_viscosity_zero(const KelvinVoigt& kelvin_voigt_model)
    {
      for (const double viscosity : kelvin_voigt_model.viscosity_eta)
      {
        if (viscosity != 0.0) return false;
      }
      return true;
    }

    struct FourElementMaxwellResidualCoefficients
    {
      std::vector<double> flow_coeff;
      std::vector<double> history_coeff;
      double dt = std::numeric_limits<double>::quiet_NaN();
    };

    FourElementMaxwellResidualCoefficients make_four_element_maxwell_residual_coefficients(
        const TerminalUnitData& data)
    {
      return FourElementMaxwellResidualCoefficients{
          .flow_coeff = std::vector<double>(data.number_of_elements()),
          .history_coeff = std::vector<double>(data.number_of_elements()),
          .dt = std::numeric_limits<double>::quiet_NaN()};
    }

    void update_four_element_maxwell_residual_coefficients(
        FourElementMaxwellResidualCoefficients& coefficients,
        const FourElementMaxwell& four_element_maxwell_model, const TerminalUnitData& data,
        double dt)
    {
      if (coefficients.dt == dt) return;

      FOUR_C_ASSERT_ALWAYS(coefficients.flow_coeff.size() == data.number_of_elements() &&
                               coefficients.history_coeff.size() == data.number_of_elements(),
          "Four-element Maxwell residual coefficient buffers must have {} entries.",
          data.number_of_elements());

      for (size_t i = 0; i < data.number_of_elements(); i++)
      {
        const double denominator = four_element_maxwell_model.elasticity_E_m[i] * dt +
                                   four_element_maxwell_model.viscosity_eta_m[i];
        const double branch_viscosity = four_element_maxwell_model.elasticity_E_m[i] * dt *
                                        four_element_maxwell_model.viscosity_eta_m[i] / denominator;
        coefficients.flow_coeff[i] = four_element_maxwell_model.viscosity_eta[i] + branch_viscosity;
        coefficients.history_coeff[i] = four_element_maxwell_model.viscosity_eta_m[i] / denominator;
      }
      coefficients.dt = dt;
    }

    double evaluate_linear_elastic_pressure(const LinearElasticity& linear_elastic_model,
        const TerminalUnitData& data, std::span<const double> dof_values, double dt, size_t i)
    {
      const double q = dof_values[data.lid_q[i]];
      return linear_elastic_model.elasticity_E[i] *
             ((data.volume_v[i] + dt * q) * data.reference_volume_context[i].inv_v0_eff - 1.0);
    }

    double evaluate_ogden_elastic_pressure(const OgdenHyperelasticity& ogden_hyperelastic_model,
        const TerminalUnitData& data, std::span<const double> dof_values, double dt, size_t i)
    {
      const double q = dof_values[data.lid_q[i]];
      const double v0_over_vi =
          data.reference_volume_context[i].v0_eff / (data.volume_v[i] + dt * q);
      return ogden_hyperelastic_model.bulk_modulus_kappa[i] /
             ogden_hyperelastic_model.nonlinear_stiffening_beta[i] * v0_over_vi *
             (1.0 - std::pow(v0_over_vi, ogden_hyperelastic_model.nonlinear_stiffening_beta[i]));
    }

    void evaluate_linear_kelvin_voigt_zero_viscosity_residual(Core::LinAlg::Vector<double>& target,
        LinearElasticity& linear_elastic_model, const TerminalUnitData& data,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt)
    {
      auto residual_values = target.local_values_as_span();
      const auto dof_values = locally_relevant_dofs.local_values_as_span();
      const auto& local_row_id = data.local_row_id;
      const auto& lid_p1 = data.lid_p1;
      const auto& lid_p2 = data.lid_p2;
      const auto& lid_q = data.lid_q;
      const auto& volume = data.volume_v;
      const auto& reference_volume = data.reference_volume_context;
      const auto& elasticity = linear_elastic_model.elasticity_E;
      auto& elastic_pressure = linear_elastic_model.elastic_pressure_p_el;
      for (size_t i = 0; i < data.number_of_elements(); i++)
      {
        const double pressure =
            elasticity[i] *
            ((volume[i] + dt * dof_values[lid_q[i]]) * reference_volume[i].inv_v0_eff - 1.0);
        elastic_pressure[i] = pressure;
        residual_values[static_cast<std::size_t>(local_row_id[i])] =
            dof_values[lid_p1[i]] - dof_values[lid_p2[i]] - pressure;
      }
    }

    void evaluate_linear_kelvin_voigt_residual(Core::LinAlg::Vector<double>& target,
        const KelvinVoigt& kelvin_voigt_model, LinearElasticity& linear_elastic_model,
        const TerminalUnitData& data, const Core::LinAlg::Vector<double>& locally_relevant_dofs,
        double dt)
    {
      auto residual_values = target.local_values_as_span();
      const auto dof_values = locally_relevant_dofs.local_values_as_span();
      const auto& local_row_id = data.local_row_id;
      const auto& lid_p1 = data.lid_p1;
      const auto& lid_p2 = data.lid_p2;
      const auto& lid_q = data.lid_q;
      const auto& volume = data.volume_v;
      const auto& reference_volume = data.reference_volume_context;
      const auto& elasticity = linear_elastic_model.elasticity_E;
      auto& elastic_pressure = linear_elastic_model.elastic_pressure_p_el;
      const auto& viscosity = kelvin_voigt_model.viscosity_eta;
      for (size_t i = 0; i < data.number_of_elements(); i++)
      {
        const double q = dof_values[lid_q[i]];
        const double pressure =
            elasticity[i] * ((volume[i] + dt * q) * reference_volume[i].inv_v0_eff - 1.0);
        elastic_pressure[i] = pressure;
        residual_values[static_cast<std::size_t>(local_row_id[i])] =
            dof_values[lid_p1[i]] - dof_values[lid_p2[i]] - pressure -
            viscosity[i] * q * reference_volume[i].inv_v0_eff;
      }
    }

    void evaluate_linear_four_element_maxwell_residual(Core::LinAlg::Vector<double>& target,
        const FourElementMaxwell& four_element_maxwell_model,
        FourElementMaxwellResidualCoefficients& coefficients,
        LinearElasticity& linear_elastic_model, TerminalUnitData& data,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt)
    {
      update_four_element_maxwell_residual_coefficients(
          coefficients, four_element_maxwell_model, data, dt);

      auto residual_values = target.local_values_as_span();
      const auto dof_values = locally_relevant_dofs.local_values_as_span();
      const auto& local_row_id = data.local_row_id;
      const auto& lid_p1 = data.lid_p1;
      const auto& lid_p2 = data.lid_p2;
      const auto& lid_q = data.lid_q;
      auto& elastic_pressure = linear_elastic_model.elastic_pressure_p_el;
      const auto& maxwell_pressure = four_element_maxwell_model.maxwell_pressure_p_m;
      const auto& flow_coeff = coefficients.flow_coeff;
      const auto& history_coeff = coefficients.history_coeff;
      for (size_t i = 0; i < data.number_of_elements(); i++)
      {
        const double q = dof_values[lid_q[i]];
        const double pressure =
            evaluate_linear_elastic_pressure(linear_elastic_model, data, dof_values, dt, i);
        elastic_pressure[i] = pressure;
        residual_values[static_cast<std::size_t>(local_row_id[i])] =
            dof_values[lid_p1[i]] - dof_values[lid_p2[i]] - pressure -
            flow_coeff[i] * data.reference_volume_context[i].inv_v0_eff * q -
            history_coeff[i] * maxwell_pressure[i];
      }
    }

    void evaluate_ogden_kelvin_voigt_zero_viscosity_residual(Core::LinAlg::Vector<double>& target,
        OgdenHyperelasticity& ogden_hyperelastic_model, TerminalUnitData& data,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt)
    {
      auto residual_values = target.local_values_as_span();
      const auto dof_values = locally_relevant_dofs.local_values_as_span();
      const auto& local_row_id = data.local_row_id;
      const auto& lid_p1 = data.lid_p1;
      const auto& lid_p2 = data.lid_p2;
      auto& elastic_pressure = ogden_hyperelastic_model.elastic_pressure_p_el;
      for (size_t i = 0; i < data.number_of_elements(); i++)
      {
        const double pressure =
            evaluate_ogden_elastic_pressure(ogden_hyperelastic_model, data, dof_values, dt, i);
        elastic_pressure[i] = pressure;
        residual_values[static_cast<std::size_t>(local_row_id[i])] =
            dof_values[lid_p1[i]] - dof_values[lid_p2[i]] - pressure;
      }
    }

    void evaluate_ogden_kelvin_voigt_residual(Core::LinAlg::Vector<double>& target,
        const KelvinVoigt& kelvin_voigt_model, OgdenHyperelasticity& ogden_hyperelastic_model,
        TerminalUnitData& data, const Core::LinAlg::Vector<double>& locally_relevant_dofs,
        double dt)
    {
      auto residual_values = target.local_values_as_span();
      const auto dof_values = locally_relevant_dofs.local_values_as_span();
      const auto& local_row_id = data.local_row_id;
      const auto& lid_p1 = data.lid_p1;
      const auto& lid_p2 = data.lid_p2;
      const auto& lid_q = data.lid_q;
      const auto& reference_volume = data.reference_volume_context;
      const auto& viscosity = kelvin_voigt_model.viscosity_eta;
      auto& elastic_pressure = ogden_hyperelastic_model.elastic_pressure_p_el;
      for (size_t i = 0; i < data.number_of_elements(); i++)
      {
        const double q = dof_values[lid_q[i]];
        const double pressure =
            evaluate_ogden_elastic_pressure(ogden_hyperelastic_model, data, dof_values, dt, i);
        elastic_pressure[i] = pressure;
        residual_values[static_cast<std::size_t>(local_row_id[i])] =
            dof_values[lid_p1[i]] - dof_values[lid_p2[i]] - pressure -
            viscosity[i] * q * reference_volume[i].inv_v0_eff;
      }
    }

    void evaluate_ogden_four_element_maxwell_residual(Core::LinAlg::Vector<double>& target,
        const FourElementMaxwell& four_element_maxwell_model,
        FourElementMaxwellResidualCoefficients& coefficients,
        OgdenHyperelasticity& ogden_hyperelastic_model, TerminalUnitData& data,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt)
    {
      update_four_element_maxwell_residual_coefficients(
          coefficients, four_element_maxwell_model, data, dt);

      auto residual_values = target.local_values_as_span();
      const auto dof_values = locally_relevant_dofs.local_values_as_span();
      const auto& local_row_id = data.local_row_id;
      const auto& lid_p1 = data.lid_p1;
      const auto& lid_p2 = data.lid_p2;
      const auto& lid_q = data.lid_q;
      auto& elastic_pressure = ogden_hyperelastic_model.elastic_pressure_p_el;
      const auto& maxwell_pressure = four_element_maxwell_model.maxwell_pressure_p_m;
      const auto& flow_coeff = coefficients.flow_coeff;
      const auto& history_coeff = coefficients.history_coeff;
      for (size_t i = 0; i < data.number_of_elements(); i++)
      {
        const double q = dof_values[lid_q[i]];
        const double pressure =
            evaluate_ogden_elastic_pressure(ogden_hyperelastic_model, data, dof_values, dt, i);
        elastic_pressure[i] = pressure;
        residual_values[static_cast<std::size_t>(local_row_id[i])] =
            dof_values[lid_p1[i]] - dof_values[lid_p2[i]] - pressure -
            flow_coeff[i] * data.reference_volume_context[i].inv_v0_eff * q -
            history_coeff[i] * maxwell_pressure[i];
      }
    }

    /**
     * Assemble Kelvin-Voigt residual entries for one model block.
     */
    void evaluate_kelvin_voigt_residual(Core::LinAlg::Vector<double>& target,
        const KelvinVoigt& kelvin_voigt_model, const TerminalUnitData& data,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs,
        const std::vector<double>& elastic_pressure_p_el)
    {
      auto residual_values = target.local_values_as_span();
      const auto dof_values = locally_relevant_dofs.local_values_as_span();
      const auto& local_row_id = data.local_row_id;
      const auto& lid_p1 = data.lid_p1;
      const auto& lid_p2 = data.lid_p2;
      const auto& lid_q = data.lid_q;
      const auto& viscosity = kelvin_voigt_model.viscosity_eta;
      for (size_t i = 0; i < data.number_of_elements(); i++)
      {
        const double inv_v0 = data.reference_volume_context[i].inv_v0_eff;
        const double kelvin_voigt_residual =
            (dof_values[lid_p1[i]] - dof_values[lid_p2[i]] - elastic_pressure_p_el[i] -
                viscosity[i] * dof_values[lid_q[i]] * inv_v0);
        residual_values[static_cast<std::size_t>(local_row_id[i])] = kelvin_voigt_residual;
      }
    }

    /**
     * Assemble Four-element-Maxwell residual entries for one model block.
     */
    void evaluate_four_element_maxwell_residual(Core::LinAlg::Vector<double>& target,
        const FourElementMaxwell& four_element_maxwell_model, const TerminalUnitData& data,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs,
        const std::vector<double>& elastic_pressure_p_el, double dt)
    {
      auto residual_values = target.local_values_as_span();
      const auto dof_values = locally_relevant_dofs.local_values_as_span();
      const auto& local_row_id = data.local_row_id;
      const auto& lid_p1 = data.lid_p1;
      const auto& lid_p2 = data.lid_p2;
      const auto& lid_q = data.lid_q;
      const auto& viscosity = four_element_maxwell_model.viscosity_eta;
      const auto& maxwell_elasticity = four_element_maxwell_model.elasticity_E_m;
      const auto& maxwell_viscosity = four_element_maxwell_model.viscosity_eta_m;
      const auto& maxwell_pressure = four_element_maxwell_model.maxwell_pressure_p_m;
      for (size_t i = 0; i < data.number_of_elements(); i++)
      {
        const double inv_v0 = data.reference_volume_context[i].inv_v0_eff;
        const double four_element_maxwell_residual =
            (dof_values[lid_p1[i]] - dof_values[lid_p2[i]] - elastic_pressure_p_el[i] -
                (viscosity[i] + (maxwell_elasticity[i] * dt * maxwell_viscosity[i]) /
                                    (maxwell_elasticity[i] * dt + maxwell_viscosity[i])) *
                    inv_v0 * dof_values[lid_q[i]] -
                maxwell_viscosity[i] / (maxwell_elasticity[i] * dt + maxwell_viscosity[i]) *
                    maxwell_pressure[i]);
        residual_values[static_cast<std::size_t>(local_row_id[i])] = four_element_maxwell_residual;
      }
    }

    /**
     * Assemble Kelvin-Voigt Jacobian entries for one model block.
     */
    void evaluate_kelvin_voigt_jacobian(Core::LinAlg::SparseMatrix& target,
        const KelvinVoigt& kelvin_voigt_model, TerminalUnitData& data,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs,
        const Elasticity::ElasticPressurePartialsView& elastic_pressure_partials, const double dt)
    {
      if (!target.filled())
      {
        for (size_t i = 0; i < data.number_of_elements(); i++)
        {
          const auto& context = data.reference_volume_context[i];
          const double q = locally_relevant_dofs.local_values_as_span()[data.lid_q[i]];
          const double alpha = (-elastic_pressure_partials.dp_el_dv0[i] +
                                   kelvin_voigt_model.viscosity_eta[i] * q * context.inv_v0_eff *
                                       context.inv_v0_eff) *
                               context.dv0_dp;
          std::array<int, 3> column_indices{data.lid_p1[i], data.lid_p2[i], data.lid_q[i]};
          std::array<double, 3> values{1.0 + alpha, -1.0 - alpha,
              -elastic_pressure_partials.dp_el_dq[i] -
                  kelvin_voigt_model.viscosity_eta[i] * context.inv_v0_eff};
          target.insert_my_values(data.local_row_id[i], 3, values.data(), column_indices.data());
        }
      }
      else
      {
        for (size_t i = 0; i < data.number_of_elements(); i++)
        {
          const auto& context = data.reference_volume_context[i];
          const double q = locally_relevant_dofs.local_values_as_span()[data.lid_q[i]];
          const double alpha = (-elastic_pressure_partials.dp_el_dv0[i] +
                                   kelvin_voigt_model.viscosity_eta[i] * q * context.inv_v0_eff *
                                       context.inv_v0_eff) *
                               context.dv0_dp;
          const std::array<int, 3> column_indices{data.lid_p1[i], data.lid_p2[i], data.lid_q[i]};
          const std::array<double, 3> values{1.0 + alpha, -1.0 - alpha,
              -elastic_pressure_partials.dp_el_dq[i] -
                  kelvin_voigt_model.viscosity_eta[i] * context.inv_v0_eff};
          target.replace_my_values(data.local_row_id[i], 3, values.data(), column_indices.data());
        }
      }
    }

    /**
     * Assemble Four-element-Maxwell Jacobian entries for one model block.
     */
    void evaluate_four_element_maxwell_jacobian(Core::LinAlg::SparseMatrix& target,
        const FourElementMaxwell& four_element_maxwell_model, TerminalUnitData& data,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs,
        const Elasticity::ElasticPressurePartialsView& elastic_pressure_partials, double dt)
    {
      const auto damping_factor = [&](const size_t i)
      {
        return four_element_maxwell_model.viscosity_eta[i] +
               four_element_maxwell_model.elasticity_E_m[i] * dt *
                   four_element_maxwell_model.viscosity_eta_m[i] /
                   (four_element_maxwell_model.elasticity_E_m[i] * dt +
                       four_element_maxwell_model.viscosity_eta_m[i]);
      };
      if (!target.filled())
      {
        for (size_t i = 0; i < data.number_of_elements(); i++)
        {
          const auto& context = data.reference_volume_context[i];
          const double alpha =
              (-elastic_pressure_partials.dp_el_dv0[i] +
                  damping_factor(i) * locally_relevant_dofs.local_values_as_span()[data.lid_q[i]] *
                      context.inv_v0_eff * context.inv_v0_eff) *
              context.dv0_dp;
          std::array<int, 3> column_indices{data.lid_p1[i], data.lid_p2[i], data.lid_q[i]};
          std::array<double, 3> values{1.0 + alpha, -1.0 - alpha,
              -elastic_pressure_partials.dp_el_dq[i] - damping_factor(i) * context.inv_v0_eff};
          target.insert_my_values(data.local_row_id[i], 3, values.data(), column_indices.data());
        }
      }
      else
      {
        for (size_t i = 0; i < data.number_of_elements(); i++)
        {
          const auto& context = data.reference_volume_context[i];
          const double alpha =
              (-elastic_pressure_partials.dp_el_dv0[i] +
                  damping_factor(i) * locally_relevant_dofs.local_values_as_span()[data.lid_q[i]] *
                      context.inv_v0_eff * context.inv_v0_eff) *
              context.dv0_dp;
          const std::array<int, 3> column_indices{data.lid_p1[i], data.lid_p2[i], data.lid_q[i]};
          const std::array<double, 3> values{1.0 + alpha, -1.0 - alpha,
              -elastic_pressure_partials.dp_el_dq[i] - damping_factor(i) * context.inv_v0_eff};
          target.replace_my_values(data.local_row_id[i], 3, values.data(), column_indices.data());
        }
      }
    }

    void evaluate_kelvin_voigt_tree_linearization(TreeCoefficientAssemblyTarget& target,
        KelvinVoigt& kelvin_voigt_model, TerminalUnitData& data,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs,
        const Elasticity::ElasticPressurePartialsView& elastic_pressure_partials)
    {
      /* Static setup inserted the row pattern; recruitment can make every value state-dependent. */
      const size_t element_count = data.number_of_elements();
      const auto& viscosity = kelvin_voigt_model.viscosity_eta;
      std::span<double> grad_q =
          resize_scratch(kelvin_voigt_model.tree_linearization_grad_q, element_count);
      for (size_t i = 0; i < element_count; i++)
      {
        const auto& context = data.reference_volume_context[i];
        const double q = locally_relevant_dofs.local_values_as_span()[data.lid_q[i]];
        const double damping = viscosity[i];
        const double alpha = (-elastic_pressure_partials.dp_el_dv0[i] +
                                 damping * q * context.inv_v0_eff * context.inv_v0_eff) *
                             context.dv0_dp;
        target.replace_value(data.local_row_id[i], data.lid_p1[i], 1.0 + alpha);
        target.replace_value(data.local_row_id[i], data.lid_p2[i], -1.0 - alpha);
        grad_q[i] = -elastic_pressure_partials.dp_el_dq[i] - damping * context.inv_v0_eff;
      }
      target.replace_values(data.local_row_id, data.lid_q, grad_q);
    }

    void evaluate_four_element_maxwell_tree_linearization(TreeCoefficientAssemblyTarget& target,
        FourElementMaxwell& four_element_maxwell_model, TerminalUnitData& data,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs,
        const Elasticity::ElasticPressurePartialsView& elastic_pressure_partials, double dt)
    {
      /* Batch Four-element Maxwell q coefficients into persistent scratch before replacement. */
      const size_t element_count = data.number_of_elements();
      const auto& viscosity = four_element_maxwell_model.viscosity_eta;
      const auto& maxwell_elasticity = four_element_maxwell_model.elasticity_E_m;
      const auto& maxwell_viscosity = four_element_maxwell_model.viscosity_eta_m;
      std::span<double> grad_q =
          resize_scratch(four_element_maxwell_model.tree_linearization_grad_q, element_count);
      for (size_t i = 0; i < element_count; i++)
      {
        const auto& context = data.reference_volume_context[i];
        const double q = locally_relevant_dofs.local_values_as_span()[data.lid_q[i]];
        const double maxwell_elasticity_dt = maxwell_elasticity[i] * dt;
        const double maxwell_branch_viscosity = maxwell_elasticity_dt * maxwell_viscosity[i] /
                                                (maxwell_elasticity_dt + maxwell_viscosity[i]);
        const double damping = viscosity[i] + maxwell_branch_viscosity;
        const double alpha = (-elastic_pressure_partials.dp_el_dv0[i] +
                                 damping * q * context.inv_v0_eff * context.inv_v0_eff) *
                             context.dv0_dp;
        target.replace_value(data.local_row_id[i], data.lid_p1[i], 1.0 + alpha);
        target.replace_value(data.local_row_id[i], data.lid_p2[i], -1.0 - alpha);
        grad_q[i] = -elastic_pressure_partials.dp_el_dq[i] - damping * context.inv_v0_eff;
      }
      target.replace_values(data.local_row_id, data.lid_q, grad_q);
    }

    void initialize_tree_linearization(
        TreeCoefficientAssemblyTarget& target, TerminalUnitData& data)
    {
      /* Append the terminal-unit row pattern once for either generic or direct targets. */
      for (size_t i = 0; i < data.number_of_elements(); ++i)
      {
        target.append_value(data.local_row_id[i], data.lid_p1[i], 1.0);
        target.append_value(data.local_row_id[i], data.lid_p2[i], -1.0);
        target.append_value(data.local_row_id[i], data.lid_q[i], 0.0);
      }
    }

    void append_rheology_output(const KelvinVoigt& /*model*/, const TerminalUnitData& /*data*/,
        RuntimeOutputCollector& /*collector*/, ReducedLungParameters::OutputVerbosity /*verbosity*/)
    {
    }

    void append_rheology_output(const FourElementMaxwell& model, const TerminalUnitData& data,
        RuntimeOutputCollector& collector, ReducedLungParameters::OutputVerbosity verbosity)
    {
      if (verbosity >= ReducedLungParameters::OutputVerbosity::high)
      {
        auto& maxwell_pressure_vec = collector.get_or_create_vector("maxwell_pressure");
        for (size_t i = 0; i < data.number_of_elements(); i++)
        {
          maxwell_pressure_vec.replace_local_value(
              data.local_element_id[i], model.maxwell_pressure_p_m[i]);
        }
      }
    }
  }  // namespace

  /**
   * Resolve variant-based residual evaluator.
   */
  ResidualEvaluator make_residual_evaluator(RheologicalModel& rheological_model,
      ElasticityModel& elasticity_model, Elasticity::ElasticPressureEvaluator pressure_evaluator,
      const TerminalUnitData& model_data)
  {
    return std::visit(
        [&](auto& model) -> ResidualEvaluator
        {
          using ModelType = std::decay_t<decltype(model)>;
          if constexpr (std::is_same_v<ModelType, KelvinVoigt>)
          {
            if (auto* linear_elastic_model = std::get_if<LinearElasticity>(&elasticity_model);
                linear_elastic_model != nullptr)
            {
              if (all_viscosity_zero(model))
              {
                return [linear_elastic_model](TerminalUnitData& data,
                           Core::LinAlg::Vector<double>& target,
                           const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt)
                {
                  evaluate_linear_kelvin_voigt_zero_viscosity_residual(
                      target, *linear_elastic_model, data, locally_relevant_dofs, dt);
                };
              }

              return [&model, linear_elastic_model](TerminalUnitData& data,
                         Core::LinAlg::Vector<double>& target,
                         const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt)
              {
                evaluate_linear_kelvin_voigt_residual(
                    target, model, *linear_elastic_model, data, locally_relevant_dofs, dt);
              };
            }

            if (auto* ogden_hyperelastic_model =
                    std::get_if<OgdenHyperelasticity>(&elasticity_model);
                ogden_hyperelastic_model != nullptr)
            {
              if (all_viscosity_zero(model))
              {
                return [ogden_hyperelastic_model](TerminalUnitData& data,
                           Core::LinAlg::Vector<double>& target,
                           const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt)
                {
                  evaluate_ogden_kelvin_voigt_zero_viscosity_residual(
                      target, *ogden_hyperelastic_model, data, locally_relevant_dofs, dt);
                };
              }

              return [&model, ogden_hyperelastic_model](TerminalUnitData& data,
                         Core::LinAlg::Vector<double>& target,
                         const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt)
              {
                evaluate_ogden_kelvin_voigt_residual(
                    target, model, *ogden_hyperelastic_model, data, locally_relevant_dofs, dt);
              };
            }

            return [&model, pressure_evaluator](TerminalUnitData& data,
                       Core::LinAlg::Vector<double>& target,
                       const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt)
            {
              auto& pressure = pressure_evaluator(data, locally_relevant_dofs, dt);
              evaluate_kelvin_voigt_residual(target, model, data, locally_relevant_dofs, pressure);
            };
          }
          else if constexpr (std::is_same_v<ModelType, FourElementMaxwell>)
          {
            if (auto* linear_elastic_model = std::get_if<LinearElasticity>(&elasticity_model);
                linear_elastic_model != nullptr)
            {
              return
                  [&model, linear_elastic_model,
                      coefficients = make_four_element_maxwell_residual_coefficients(model_data)](
                      TerminalUnitData& data, Core::LinAlg::Vector<double>& target,
                      const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt) mutable
              {
                evaluate_linear_four_element_maxwell_residual(target, model, coefficients,
                    *linear_elastic_model, data, locally_relevant_dofs, dt);
              };
            }

            if (auto* ogden_hyperelastic_model =
                    std::get_if<OgdenHyperelasticity>(&elasticity_model);
                ogden_hyperelastic_model != nullptr)
            {
              return
                  [&model, ogden_hyperelastic_model,
                      coefficients = make_four_element_maxwell_residual_coefficients(model_data)](
                      TerminalUnitData& data, Core::LinAlg::Vector<double>& target,
                      const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt) mutable
              {
                evaluate_ogden_four_element_maxwell_residual(target, model, coefficients,
                    *ogden_hyperelastic_model, data, locally_relevant_dofs, dt);
              };
            }

            return [&model, pressure_evaluator](TerminalUnitData& data,
                       Core::LinAlg::Vector<double>& target,
                       const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt)
            {
              auto& pressure = pressure_evaluator(data, locally_relevant_dofs, dt);
              evaluate_four_element_maxwell_residual(
                  target, model, data, locally_relevant_dofs, pressure, dt);
            };
          }
          else
          {
            FOUR_C_THROW("Unknown terminal-unit rheological model.");
          }
        },
        rheological_model);
  }

  /**
   * Resolve variant-based Jacobian evaluator.
   */
  JacobianEvaluator make_jacobian_evaluator(RheologicalModel& rheological_model,
      Elasticity::ElasticPressurePartialsEvaluator elastic_pressure_partials_evaluator)
  {
    return std::visit(
        [&](auto& model) -> JacobianEvaluator
        {
          using ModelType = std::decay_t<decltype(model)>;
          if constexpr (std::is_same_v<ModelType, KelvinVoigt>)
          {
            return [&model, elastic_pressure_partials_evaluator](TerminalUnitData& data,
                       Core::LinAlg::SparseMatrix& target,
                       const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt)
            {
              const auto elastic_pressure_partials =
                  elastic_pressure_partials_evaluator(data, locally_relevant_dofs, dt);
              evaluate_kelvin_voigt_jacobian(
                  target, model, data, locally_relevant_dofs, elastic_pressure_partials, dt);
            };
          }
          else if constexpr (std::is_same_v<ModelType, FourElementMaxwell>)
          {
            return [&model, elastic_pressure_partials_evaluator](TerminalUnitData& data,
                       Core::LinAlg::SparseMatrix& target,
                       const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt)
            {
              const auto elastic_pressure_partials =
                  elastic_pressure_partials_evaluator(data, locally_relevant_dofs, dt);
              evaluate_four_element_maxwell_jacobian(
                  target, model, data, locally_relevant_dofs, elastic_pressure_partials, dt);
            };
          }
          else
          {
            FOUR_C_THROW("Unknown terminal-unit rheological model.");
          }
        },
        rheological_model);
  }

  StaticTreeLinearizationEvaluator make_static_tree_linearization_evaluator(
      RheologicalModel& rheological_model)
  {
    return std::visit(
        [](auto& /*model*/) -> StaticTreeLinearizationEvaluator
        {
          return [](TerminalUnitData& data, TreeCoefficientAssemblyTarget& target)
          { initialize_tree_linearization(target, data); };
        },
        rheological_model);
  }

  TreeLinearizationEvaluator make_tree_linearization_evaluator(RheologicalModel& rheological_model,
      Elasticity::ElasticPressurePartialsEvaluator elastic_pressure_partials_evaluator)
  {
    return std::visit(
        [&](auto& model) -> TreeLinearizationEvaluator
        {
          using ModelType = std::decay_t<decltype(model)>;
          if constexpr (std::is_same_v<ModelType, KelvinVoigt>)
          {
            return [&model, elastic_pressure_partials_evaluator](TerminalUnitData& data,
                       TreeCoefficientAssemblyTarget& target,
                       const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt)
            {
              const auto elastic_pressure_partials =
                  elastic_pressure_partials_evaluator(data, locally_relevant_dofs, dt);
              evaluate_kelvin_voigt_tree_linearization(
                  target, model, data, locally_relevant_dofs, elastic_pressure_partials);
            };
          }
          else if constexpr (std::is_same_v<ModelType, FourElementMaxwell>)
          {
            return [&model, elastic_pressure_partials_evaluator](TerminalUnitData& data,
                       TreeCoefficientAssemblyTarget& target,
                       const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt)
            {
              const auto elastic_pressure_partials =
                  elastic_pressure_partials_evaluator(data, locally_relevant_dofs, dt);
              evaluate_four_element_maxwell_tree_linearization(
                  target, model, data, locally_relevant_dofs, elastic_pressure_partials, dt);
            };
          }
          else
          {
            FOUR_C_THROW("Unknown terminal-unit rheological model.");
          }
        },
        rheological_model);
  }

  /**
   * Resolve variant-based nonlinear-state updater.
   */
  InternalStateUpdater make_internal_state_updater(RheologicalModel& rheological_model)
  {
    return std::visit(
        [&](auto& model) -> InternalStateUpdater
        {
          using ModelType = std::decay_t<decltype(model)>;
          if constexpr (std::is_same_v<ModelType, KelvinVoigt>)
          {
            return [](TerminalUnitData& /*data*/,
                       const Core::LinAlg::Vector<double>& /*locally_relevant_dofs*/,
                       double /*dt*/) {};
          }
          else if constexpr (std::is_same_v<ModelType, FourElementMaxwell>)
          {
            return [](TerminalUnitData& /*data*/,
                       const Core::LinAlg::Vector<double>& /*locally_relevant_dofs*/,
                       double /*dt*/) {};
          }
          else
          {
            FOUR_C_THROW("Unknown terminal-unit rheological model.");
          }
        },
        rheological_model);
  }

  /**
   * Resolve variant-based end-of-step history update routine.
   */
  EndOfTimestepRoutine make_end_of_timestep_routine(RheologicalModel& rheological_model)
  {
    return std::visit(
        [&](auto& model) -> EndOfTimestepRoutine
        {
          using ModelType = std::decay_t<decltype(model)>;
          if constexpr (std::is_same_v<ModelType, KelvinVoigt>)
          {
            return [](TerminalUnitData& /*data*/,
                       const Core::LinAlg::Vector<double>& /*locally_relevant_dofs*/,
                       double /*dt*/) {};
          }
          else if constexpr (std::is_same_v<ModelType, FourElementMaxwell>)
          {
            return [&model](TerminalUnitData& data,
                       const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt)
            {
              for (size_t i = 0; i < data.number_of_elements(); i++)
              {
                const double inv_v0 = data.reference_volume_context[i].inv_v0_eff;
                model.maxwell_pressure_p_m[i] *=
                    model.viscosity_eta_m[i] /
                    (model.elasticity_E_m[i] * dt + model.viscosity_eta_m[i]);
                model.maxwell_pressure_p_m[i] +=
                    model.elasticity_E_m[i] * dt * model.viscosity_eta_m[i] /
                    (model.elasticity_E_m[i] * dt + model.viscosity_eta_m[i]) * inv_v0 *
                    locally_relevant_dofs.local_values_as_span()[data.lid_q[i]];
              }
            };
          }
          else
          {
            FOUR_C_THROW("Unknown terminal-unit rheological model.");
          }
        },
        rheological_model);
  }

  /**
   * Resolve variant-based output evaluator.
   */
  OutputEvaluator make_output_evaluator(RheologicalModel& rheological_model)
  {
    return std::visit(
        [&](auto& model) -> OutputEvaluator
        {
          return [&model](const TerminalUnitData& data, RuntimeOutputCollector& collector,
                     ReducedLungParameters::OutputVerbosity verbosity)
          { append_rheology_output(model, data, collector, verbosity); };
        },
        rheological_model);
  }

  /**
   * Append input parameters and initialize internal rheology state vectors.
   */
  void append_model_parameters(RheologicalModel& rheological_model, const int global_element_id,
      const ReducedLungParameters::LungTree::TerminalUnits::RheologicalModel& parameters)
  {
    std::visit(
        [&](auto& model)
        {
          using ModelType = std::decay_t<decltype(model)>;
          if constexpr (std::is_same_v<ModelType, KelvinVoigt>)
          {
            model.viscosity_eta.push_back(parameters.kelvin_voigt.viscosity_kelvin_voigt_eta.at(
                global_element_id, "viscosity_kelvin_voigt_eta"));
            model.tree_linearization_grad_q.push_back(0.0);
          }
          else if constexpr (std::is_same_v<ModelType, FourElementMaxwell>)
          {
            model.elasticity_E_m.push_back(
                parameters.four_element_maxwell.elasticity_maxwell_e_m.at(
                    global_element_id, "elasticity_maxwell_e_m"));
            model.viscosity_eta.push_back(
                parameters.four_element_maxwell.viscosity_kelvin_voigt_eta.at(
                    global_element_id, "viscosity_kelvin_voigt_eta"));
            model.viscosity_eta_m.push_back(
                parameters.four_element_maxwell.viscosity_maxwell_eta_m.at(
                    global_element_id, "viscosity_maxwell_eta_m"));
            model.maxwell_pressure_p_m.push_back(0.0);
            model.tree_linearization_grad_q.push_back(0.0);
          }
          else
          {
            FOUR_C_THROW("Unknown terminal-unit rheological model.");
          }
        },
        rheological_model);
  }
}  // namespace ReducedLung::TerminalUnits::Rheology

FOUR_C_NAMESPACE_CLOSE

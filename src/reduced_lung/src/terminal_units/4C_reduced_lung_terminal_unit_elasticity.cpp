// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_reduced_lung_terminal_unit_elasticity.hpp"

#include "4C_reduced_lung_helpers.hpp"

#include <cmath>

FOUR_C_NAMESPACE_OPEN

namespace ReducedLung::TerminalUnits::Elasticity
{
  namespace
  {
    /**
     * Compute linear elastic pressure for all elements in one model block.
     */
    std::vector<double>& evaluate_linear_elastic_pressure(LinearElasticity& linear_elastic_model,
        TerminalUnitData& data, const Core::LinAlg::Vector<double>& locally_relevant_dofs,
        double dt)
    {
      for (size_t i = 0; i < data.number_of_elements(); i++)
      {
        const double inv_v0 = data.reference_volume_context[i].inv_v0_eff;
        linear_elastic_model.elastic_pressure_p_el[i] =
            linear_elastic_model.elasticity_E[i] *
            ((data.volume_v[i] + dt * locally_relevant_dofs.local_values_as_span()[data.lid_q[i]]) *
                    inv_v0 -
                1);
      }
      return linear_elastic_model.elastic_pressure_p_el;
    }

    /**
     * Compute Ogden hyperelastic pressure for all elements in one model block.
     */
    std::vector<double>& evaluate_ogden_hyperelastic_pressure(
        OgdenHyperelasticity& ogden_hyperelastic_model, TerminalUnitData& data,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt)
    {
      for (size_t i = 0; i < data.number_of_elements(); i++)
      {
        const double v0 = data.reference_volume_context[i].v0_eff;
        const double v0_over_vi =
            v0 /
            (data.volume_v[i] + dt * locally_relevant_dofs.local_values_as_span()[data.lid_q[i]]);
        ogden_hyperelastic_model.elastic_pressure_p_el[i] =
            ogden_hyperelastic_model.bulk_modulus_kappa[i] /
            ogden_hyperelastic_model.nonlinear_stiffening_beta[i] * v0_over_vi *
            (1 - std::pow(v0_over_vi, ogden_hyperelastic_model.nonlinear_stiffening_beta[i]));
      }
      return ogden_hyperelastic_model.elastic_pressure_p_el;
    }

    /**
     * Compute d(p_el)/dq and d(p_el)/dV0 for the linear elasticity model.
     */
    ElasticPressurePartialsView linear_elastic_pressure_partials(
        LinearElasticity& linear_elastic_model, TerminalUnitData& data,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt)
    {
      for (size_t i = 0; i < data.number_of_elements(); i++)
      {
        const auto& context = data.reference_volume_context[i];
        linear_elastic_model.dp_el_dq[i] =
            linear_elastic_model.elasticity_E[i] * dt * context.inv_v0_eff;
        linear_elastic_model.dp_el_dv0[i] =
            -linear_elastic_model.elasticity_E[i] *
            (data.volume_v[i] + dt * locally_relevant_dofs.local_values_as_span()[data.lid_q[i]]) *
            context.inv_v0_eff * context.inv_v0_eff;
      }
      return {
          .dp_el_dq = linear_elastic_model.dp_el_dq, .dp_el_dv0 = linear_elastic_model.dp_el_dv0};
    }

    /**
     * Compute d(p_el)/dq and d(p_el)/dV0 for the Ogden elasticity model.
     */
    ElasticPressurePartialsView ogden_hyperelastic_pressure_partials(
        OgdenHyperelasticity& ogden_hyperelastic_model, TerminalUnitData& data,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt)
    {
      for (size_t i = 0; i < data.number_of_elements(); i++)
      {
        const auto& context = data.reference_volume_context[i];
        const double v0 = context.v0_eff;
        const double current_volume =
            data.volume_v[i] + dt * locally_relevant_dofs.local_values_as_span()[data.lid_q[i]];
        const double v0_over_vi = v0 / current_volume;
        ogden_hyperelastic_model.dp_el_dq[i] =
            ogden_hyperelastic_model.bulk_modulus_kappa[i] * dt /
            (ogden_hyperelastic_model.nonlinear_stiffening_beta[i] * v0) * v0_over_vi * v0_over_vi *
            ((ogden_hyperelastic_model.nonlinear_stiffening_beta[i] + 1) *
                    std::pow(v0_over_vi, ogden_hyperelastic_model.nonlinear_stiffening_beta[i]) -
                1);
        ogden_hyperelastic_model.dp_el_dv0[i] =
            ogden_hyperelastic_model.bulk_modulus_kappa[i] /
            (ogden_hyperelastic_model.nonlinear_stiffening_beta[i] * current_volume) *
            (1.0 - (ogden_hyperelastic_model.nonlinear_stiffening_beta[i] + 1.0) *
                       std::pow(v0_over_vi, ogden_hyperelastic_model.nonlinear_stiffening_beta[i]));
      }
      return {.dp_el_dq = ogden_hyperelastic_model.dp_el_dq,
          .dp_el_dv0 = ogden_hyperelastic_model.dp_el_dv0};
    }

    template <typename Model>
    void append_elastic_pressure_output(const Model& model, const TerminalUnitData& data,
        RuntimeOutputCollector& collector, ReducedLungParameters::OutputVerbosity verbosity)
    {
      if (verbosity >= ReducedLungParameters::OutputVerbosity::high)
      {
        auto& elastic_pressure_vec = collector.get_or_create_vector("elastic_pressure");
        for (size_t i = 0; i < data.number_of_elements(); i++)
        {
          elastic_pressure_vec.replace_local_value(
              data.local_element_id[i], model.elastic_pressure_p_el[i]);
        }
      }
    }
  }  // namespace

  /**
   * Resolve variant-based pressure evaluator.
   */
  ElasticPressureEvaluator make_elastic_pressure_evaluator(ElasticityModel& elasticity_model)
  {
    return std::visit(
        [&](auto& model) -> ElasticPressureEvaluator
        {
          using ModelType = std::decay_t<decltype(model)>;
          if constexpr (std::is_same_v<ModelType, LinearElasticity>)
          {
            return [&model](TerminalUnitData& data,
                       const Core::LinAlg::Vector<double>& locally_relevant_dofs,
                       double dt) -> std::vector<double>&
            { return evaluate_linear_elastic_pressure(model, data, locally_relevant_dofs, dt); };
          }
          else if constexpr (std::is_same_v<ModelType, OgdenHyperelasticity>)
          {
            return [&model](TerminalUnitData& data,
                       const Core::LinAlg::Vector<double>& locally_relevant_dofs,
                       double dt) -> std::vector<double>&
            {
              return evaluate_ogden_hyperelastic_pressure(model, data, locally_relevant_dofs, dt);
            };
          }
          else
          {
            FOUR_C_THROW("Unknown terminal-unit elasticity model.");
          }
        },
        elasticity_model);
  }

  ElasticPressurePartialsEvaluator make_elastic_pressure_partials_evaluator(
      ElasticityModel& elasticity_model)
  {
    return std::visit(
        [&](auto& model) -> ElasticPressurePartialsEvaluator
        {
          using ModelType = std::decay_t<decltype(model)>;
          if constexpr (std::is_same_v<ModelType, LinearElasticity>)
          {
            return [&model](TerminalUnitData& data,
                       const Core::LinAlg::Vector<double>& locally_relevant_dofs,
                       double dt) -> ElasticPressurePartialsView
            { return linear_elastic_pressure_partials(model, data, locally_relevant_dofs, dt); };
          }
          else if constexpr (std::is_same_v<ModelType, OgdenHyperelasticity>)
          {
            return [&model](TerminalUnitData& data,
                       const Core::LinAlg::Vector<double>& locally_relevant_dofs,
                       double dt) -> ElasticPressurePartialsView
            {
              return ogden_hyperelastic_pressure_partials(model, data, locally_relevant_dofs, dt);
            };
          }
          else
          {
            FOUR_C_THROW("Unknown terminal-unit elasticity model.");
          }
        },
        elasticity_model);
  }

  /**
   * Resolve variant-based output evaluator.
   */
  OutputEvaluator make_output_evaluator(ElasticityModel& elasticity_model)
  {
    return std::visit(
        [&](auto& model) -> OutputEvaluator
        {
          return [&model](const TerminalUnitData& data, RuntimeOutputCollector& collector,
                     ReducedLungParameters::OutputVerbosity verbosity)
          { append_elastic_pressure_output(model, data, collector, verbosity); };
        },
        elasticity_model);
  }

  /**
   * Append input parameters and initialize internal elasticity state vectors.
   */
  void append_model_parameters(ElasticityModel& elasticity_model, const int global_element_id,
      const ReducedLungParameters::LungTree::TerminalUnits::ElasticityModel& parameters)
  {
    std::visit(
        [&](auto& model)
        {
          using ModelType = std::decay_t<decltype(model)>;
          if constexpr (std::is_same_v<ModelType, LinearElasticity>)
          {
            model.elasticity_E.push_back(
                parameters.linear.elasticity_e.at(global_element_id, "elasticity_e"));
            model.elastic_pressure_p_el.push_back(0.0);
            model.dp_el_dq.push_back(0.0);
            model.dp_el_dv0.push_back(0.0);
          }
          else if constexpr (std::is_same_v<ModelType, OgdenHyperelasticity>)
          {
            model.bulk_modulus_kappa.push_back(parameters.ogden.ogden_parameter_kappa.at(
                global_element_id, "ogden_parameter_kappa"));
            model.nonlinear_stiffening_beta.push_back(parameters.ogden.ogden_parameter_beta.at(
                global_element_id, "ogden_parameter_beta"));
            model.elastic_pressure_p_el.push_back(0.0);
            model.dp_el_dq.push_back(0.0);
            model.dp_el_dv0.push_back(0.0);
          }
          else
          {
            FOUR_C_THROW("Unknown terminal-unit elasticity model.");
          }
        },
        elasticity_model);
  }
}  // namespace ReducedLung::TerminalUnits::Elasticity

FOUR_C_NAMESPACE_CLOSE

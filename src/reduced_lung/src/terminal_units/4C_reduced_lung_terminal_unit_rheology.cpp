// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_reduced_lung_terminal_unit_rheology.hpp"

#include "4C_reduced_lung_helpers.hpp"

#include <array>

FOUR_C_NAMESPACE_OPEN

namespace ReducedLung::TerminalUnits::Rheology
{
  namespace
  {
    /**
     * Assemble Kelvin-Voigt residual entries for one model block.
     */
    void evaluate_kelvin_voigt_residual(Core::LinAlg::Vector<double>& target,
        const KelvinVoigt& kelvin_voigt_model, const TerminalUnitData& data,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs,
        const std::vector<double>& elastic_pressure_p_el)
    {
      for (size_t i = 0; i < data.number_of_elements(); i++)
      {
        const double kelvin_voigt_residual =
            (locally_relevant_dofs.local_values_as_span()[data.lid_p1[i]] -
                locally_relevant_dofs.local_values_as_span()[data.lid_p2[i]] -
                elastic_pressure_p_el[i] -
                kelvin_voigt_model.viscosity_eta[i] *
                    locally_relevant_dofs.local_values_as_span()[data.lid_q[i]] /
                    data.reference_volume_v0[i]);
        target.replace_local_value(data.local_row_id[i], kelvin_voigt_residual);
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
      for (size_t i = 0; i < data.number_of_elements(); i++)
      {
        const double four_element_maxwell_residual =
            (locally_relevant_dofs.local_values_as_span()[data.lid_p1[i]] -
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
        target.replace_local_value(data.local_row_id[i], four_element_maxwell_residual);
      }
    }

    /**
     * Assemble Kelvin-Voigt Jacobian entries for one model block.
     */
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
          const double grad_q = -elastic_pressure_grad_dp_el[i] -
                                kelvin_voigt_model.viscosity_eta[i] / data.reference_volume_v0[i];
          target.replace_my_values(data.local_row_id[i], 1, &grad_q, &data.lid_q[i]);
        }
      }
    }

    /**
     * Assemble Four-element-Maxwell Jacobian entries for one model block.
     */
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
          const double grad_q = -elastic_pressure_grad_dp_el[i] -
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
  ResidualEvaluator make_residual_evaluator(
      RheologicalModel& rheological_model, Elasticity::ElasticPressureEvaluator pressure_evaluator)
  {
    return std::visit(
        [&](auto& model) -> ResidualEvaluator
        {
          using ModelType = std::decay_t<decltype(model)>;
          if constexpr (std::is_same_v<ModelType, KelvinVoigt>)
          {
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
      Elasticity::ElasticPressureGradientEvaluator pressure_gradient_evaluator)
  {
    return std::visit(
        [&](auto& model) -> JacobianEvaluator
        {
          using ModelType = std::decay_t<decltype(model)>;
          if constexpr (std::is_same_v<ModelType, KelvinVoigt>)
          {
            return [&model, pressure_gradient_evaluator](TerminalUnitData& data,
                       Core::LinAlg::SparseMatrix& target,
                       const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt)
            {
              auto& pressure_gradient =
                  pressure_gradient_evaluator(data, locally_relevant_dofs, dt);
              evaluate_kelvin_voigt_jacobian(target, model, data, pressure_gradient);
            };
          }
          else if constexpr (std::is_same_v<ModelType, FourElementMaxwell>)
          {
            return [&model, pressure_gradient_evaluator](TerminalUnitData& data,
                       Core::LinAlg::SparseMatrix& target,
                       const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt)
            {
              auto& pressure_gradient =
                  pressure_gradient_evaluator(data, locally_relevant_dofs, dt);
              evaluate_four_element_maxwell_jacobian(target, model, data, pressure_gradient, dt);
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
                model.maxwell_pressure_p_m[i] *=
                    model.viscosity_eta_m[i] /
                    (model.elasticity_E_m[i] * dt + model.viscosity_eta_m[i]);
                model.maxwell_pressure_p_m[i] +=
                    model.elasticity_E_m[i] * dt * model.viscosity_eta_m[i] /
                    (model.elasticity_E_m[i] * dt + model.viscosity_eta_m[i]) /
                    data.reference_volume_v0[i] *
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

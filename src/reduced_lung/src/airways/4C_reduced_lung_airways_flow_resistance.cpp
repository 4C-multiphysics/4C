// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_reduced_lung_airways_flow_resistance.hpp"

#include "4C_reduced_lung_airways_wall_mechanics.hpp"
#include "4C_reduced_lung_helpers.hpp"

#include <cmath>
#include <numbers>
#include <type_traits>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace ReducedLung::Airways::FlowResistance
{
  namespace
  {
    struct ComputePoiseuilleResistance
    {
      std::vector<double> operator()(const AirwayData& data, const std::vector<double>& area) const
      {
        std::vector<double> poiseuille(data.number_of_elements());
        for (size_t i = 0; i < data.number_of_elements(); ++i)
        {
          poiseuille[i] = 8 * std::numbers::pi * data.air_properties.dynamic_viscosity *
                          data.ref_length[i] / (area[i] * area[i]);
        }
        return poiseuille;
      }
    };

    template <typename Model>
    InertiaEvaluator make_inertia_evaluator_impl(const Model& model)
    {
      using HasInertiaT = decltype(std::declval<Model>().has_inertia);
      static_assert(std::is_same_v<std::remove_reference_t<HasInertiaT>, std::vector<bool>>,
          "Model must have member 'has_inertia' of type std::vector<bool>");

      return [&model](const AirwayData& data, const std::vector<double>& area)
      {
        std::vector<double> inertia(data.number_of_elements(), 0.0);
        for (size_t i = 0; i < data.number_of_elements(); ++i)
        {
          if (i < model.has_inertia.size() && model.has_inertia[i])
          {
            inertia[i] = data.air_properties.density * data.ref_length[i] / area[i];
          }
        }
        return inertia;
      };
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

    std::pair<std::vector<double>, std::vector<double>>
    evaluate_linear_flow_resistance_derivative_kelvin_voigt(const LinearResistive& model,
        const AirwayData& data, const Core::LinAlg::Vector<double>& dofs,
        const std::vector<double>& area, double dt)
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
        const AirwayData& data, const Core::LinAlg::Vector<double>& dofs,
        const std::vector<double>& area, double dt)
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
                  std::sqrt(std::abs(dofs.local_values_as_span()[data.lid_q1[i]] +
                                     dofs.local_values_as_span()[data.lid_q2[i]])));
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
                     (area[i] * area[i]) * dalpha_dk * dk_dq1 -
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
                     (area[i] * area[i]) * dalpha_dk * dk_dq2 +
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

    template <typename Model>
    InertiaDerivativeEvaluatorKelvinVoigt make_inertia_derivative_evaluator_kelvin_voigt_impl(
        const Model& model, KelvinVoigtWall& kelvin_voigt_wall_model)
    {
      using HasInertiaT = decltype(std::declval<Model>().has_inertia);
      static_assert(std::is_same_v<std::remove_reference_t<HasInertiaT>, std::vector<bool>>,
          "Model must have member 'has_inertia' of type std::vector<bool>");

      KelvinVoigtWall* wall = &kelvin_voigt_wall_model;
      return [wall, &model](
                 const AirwayData& data, const Core::LinAlg::Vector<double>& dofs, double dt)
      {
        return evaluate_inertia_derivative_kelvin_voigt(model.has_inertia, data, dofs, *wall, dt);
      };
    }
  }  // namespace

  InertiaEvaluator make_inertia_evaluator(FlowModel& flow_model)
  {
    return std::visit([](const auto& model) -> InertiaEvaluator
        { return make_inertia_evaluator_impl(model); }, flow_model);
  }

  FlowResistanceEvaluator make_flow_resistance_evaluator_rigid(FlowModel& flow_model)
  {
    return std::visit(
        [](const auto& model) -> FlowResistanceEvaluator
        {
          using ModelType = std::decay_t<decltype(model)>;
          if constexpr (std::is_same_v<ModelType, LinearResistive>)
          {
            return [](const AirwayData& data, const Core::LinAlg::Vector<double>&,
                       const std::vector<double>& area)
            { return ComputePoiseuilleResistance{}(data, area); };
          }
          else if constexpr (std::is_same_v<ModelType, NonLinearResistive>)
          {
            return [&model](const AirwayData& data, const Core::LinAlg::Vector<double>&,
                       const std::vector<double>& area)
            {
              auto poiseuille = ComputePoiseuilleResistance{}(data, area);
              std::vector<double> resistance(data.number_of_elements());
              for (size_t i = 0; i < resistance.size(); ++i)
              {
                resistance[i] = poiseuille[i] * model.k_turb[i];
              }
              return resistance;
            };
          }
          else
          {
            FOUR_C_THROW("Unknown airway flow model.");
          }
        },
        flow_model);
  }

  FlowResistanceEvaluator make_flow_resistance_evaluator_kelvin_voigt(FlowModel& flow_model)
  {
    return std::visit(
        [](const auto& model) -> FlowResistanceEvaluator
        {
          using ModelType = std::decay_t<decltype(model)>;
          if constexpr (std::is_same_v<ModelType, LinearResistive>)
          {
            return [](const AirwayData& data, const Core::LinAlg::Vector<double>&,
                       const std::vector<double>& area)
            { return ComputePoiseuilleResistance{}(data, area); };
          }
          else if constexpr (std::is_same_v<ModelType, NonLinearResistive>)
          {
            return [&model](const AirwayData& data, const Core::LinAlg::Vector<double>& dofs,
                       const std::vector<double>& area)
            {
              auto poiseuille = ComputePoiseuilleResistance{}(data, area);
              std::vector<double> resistance(area.size());
              for (size_t i = 0; i < resistance.size(); ++i)
              {
                const double alpha = 4.0 * model.k_turb[i] / (4.0 * model.k_turb[i] - 1.0);
                resistance[i] = poiseuille[i] * model.k_turb[i] +
                                2 * data.air_properties.density * alpha / (area[i] * area[i]) *
                                    (dofs.local_values_as_span()[data.lid_q2[i]] -
                                        dofs.local_values_as_span()[data.lid_q1[i]]);
              }
              return resistance;
            };
          }
          else
          {
            FOUR_C_THROW("Unknown airway flow model.");
          }
        },
        flow_model);
  }

  FlowResistanceDerivativeEvaluatorRigid make_flow_resistance_derivative_evaluator_rigid(
      FlowModel& flow_model)
  {
    return std::visit(
        [](const auto& model) -> FlowResistanceDerivativeEvaluatorRigid
        {
          using ModelType = std::decay_t<decltype(model)>;
          if constexpr (std::is_same_v<ModelType, LinearResistive>)
          {
            return [](const AirwayData& data, const Core::LinAlg::Vector<double>&, double)
            { return ComputePoiseuilleResistance{}(data, data.ref_area); };
          }
          else if constexpr (std::is_same_v<ModelType, NonLinearResistive>)
          {
            return [&model](
                       const AirwayData& data, const Core::LinAlg::Vector<double>& dofs, double dt)
            { return evaluate_nonlinear_flow_resistance_derivative_rigid(model, data, dofs, dt); };
          }
          else
          {
            FOUR_C_THROW("Unknown airway flow model.");
          }
        },
        flow_model);
  }

  FlowResistanceDerivativeEvaluatorKelvinVoigt
  make_flow_resistance_derivative_evaluator_kelvin_voigt(
      FlowModel& flow_model, KelvinVoigtWall& kelvin_voigt_wall_model)
  {
    KelvinVoigtWall* wall = &kelvin_voigt_wall_model;
    return std::visit(
        [wall](const auto& model) -> FlowResistanceDerivativeEvaluatorKelvinVoigt
        {
          using ModelType = std::decay_t<decltype(model)>;
          if constexpr (std::is_same_v<ModelType, LinearResistive>)
          {
            return [wall, &model](
                       const AirwayData& data, const Core::LinAlg::Vector<double>& dofs, double dt)
            {
              return evaluate_linear_flow_resistance_derivative_kelvin_voigt(
                  model, data, dofs, wall->area, dt);
            };
          }
          else if constexpr (std::is_same_v<ModelType, NonLinearResistive>)
          {
            return [wall, &model](
                       const AirwayData& data, const Core::LinAlg::Vector<double>& dofs, double dt)
            {
              return evaluate_nonlinear_flow_resistance_derivative_kelvin_voigt(
                  model, data, dofs, wall->area, dt);
            };
          }
          else
          {
            FOUR_C_THROW("Unknown airway flow model.");
          }
        },
        flow_model);
  }

  InertiaDerivativeEvaluatorKelvinVoigt make_inertia_derivative_evaluator_kelvin_voigt(
      FlowModel& flow_model, KelvinVoigtWall& kelvin_voigt_wall_model)
  {
    return std::visit(
        [&kelvin_voigt_wall_model](const auto& model) -> InertiaDerivativeEvaluatorKelvinVoigt
        {
          return make_inertia_derivative_evaluator_kelvin_voigt_impl(
              model, kelvin_voigt_wall_model);
        },
        flow_model);
  }

  InternalStateUpdaterFlowModel make_internal_state_updater(FlowModel& flow_model)
  {
    return std::visit(
        [](auto& model) -> InternalStateUpdaterFlowModel
        {
          using ModelType = std::decay_t<decltype(model)>;
          if constexpr (std::is_same_v<ModelType, LinearResistive>)
          {
            return [](AirwayData&, const Core::LinAlg::Vector<double>&) {};
          }
          else if constexpr (std::is_same_v<ModelType, NonLinearResistive>)
          {
            return [&model](AirwayData& data, const Core::LinAlg::Vector<double>& dofs)
            {
              for (size_t i = 0; i < data.number_of_elements(); i++)
              {
                double q_characteristic;
                if (data.n_state_equations == 1)
                {
                  q_characteristic = std::abs(dofs.local_values_as_span()[data.lid_q1[i]]);
                }
                else if (data.n_state_equations == 2)
                {
                  q_characteristic = 0.5 * std::abs(dofs.local_values_as_span()[data.lid_q1[i]] +
                                                    dofs.local_values_as_span()[data.lid_q2[i]]);
                }
                else
                {
                  FOUR_C_THROW("Number of state equations not supported.");
                }
                model.k_turb[i] =
                    model.turbulence_factor_gamma[i] *
                    std::sqrt((4 * data.air_properties.density) /
                              (M_PI * data.air_properties.dynamic_viscosity * data.ref_length[i]) *
                              q_characteristic);
                if (model.k_turb[i] < 1.0) model.k_turb[i] = 1.0;
              }
            };
          }
          else
          {
            FOUR_C_THROW("Unknown airway flow model.");
          }
        },
        flow_model);
  }

  void append_flow_output(const LinearResistive& /*model*/, const AirwayData& /*data*/,
      RuntimeOutputCollector& /*collector*/, ReducedLungParameters::OutputVerbosity /*verbosity*/)
  {
  }

  void append_flow_output(const NonLinearResistive& model, const AirwayData& data,
      RuntimeOutputCollector& collector, ReducedLungParameters::OutputVerbosity verbosity)
  {
    if (verbosity >= ReducedLungParameters::OutputVerbosity::high)
    {
      auto& k_turb = collector.get_or_create_vector("flow_k_turb");
      for (size_t i = 0; i < data.number_of_elements(); ++i)
      {
        k_turb.replace_local_value(data.local_element_id[i], model.k_turb[i]);
      }
    }
  }

  OutputEvaluator make_output_evaluator(FlowModel& flow_model)
  {
    return std::visit(
        [](auto& model) -> OutputEvaluator
        {
          return [&model](const AirwayData& data, RuntimeOutputCollector& collector,
                     ReducedLungParameters::OutputVerbosity verbosity)
          { append_flow_output(model, data, collector, verbosity); };
        },
        flow_model);
  }

  void append_model_parameters(FlowModel& flow_model, int global_element_id,
      const ReducedLungParameters::LungTree::Airways::FlowModel& parameters)
  {
    std::visit(
        [&](auto& model)
        {
          using ModelType = std::decay_t<decltype(model)>;
          if constexpr (std::is_same_v<ModelType, LinearResistive>)
          {
            model.has_inertia.push_back(
                parameters.include_inertia.at(global_element_id, "include_inertia"));
          }
          else if constexpr (std::is_same_v<ModelType, NonLinearResistive>)
          {
            model.turbulence_factor_gamma.push_back(
                parameters.resistance_model.non_linear.turbulence_factor_gamma.at(
                    global_element_id, "turbulence_factor_gamma"));
            model.has_inertia.push_back(
                parameters.include_inertia.at(global_element_id, "include_inertia"));
            model.k_turb.push_back(1.0);
          }
          else
          {
            FOUR_C_THROW("Unknown airway flow model.");
          }
        },
        flow_model);
  }
}  // namespace ReducedLung::Airways::FlowResistance

FOUR_C_NAMESPACE_CLOSE

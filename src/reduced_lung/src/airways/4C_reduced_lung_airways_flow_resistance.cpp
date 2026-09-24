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
#include <span>
#include <type_traits>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace ReducedLung::Airways::FlowResistance
{
  namespace
  {
    struct ComputePoiseuilleResistance
    {
      void operator()(
          const AirwayData& data, const std::vector<double>& area, std::span<double> result) const
      {
        FOUR_C_ASSERT_ALWAYS(area.size() == data.number_of_elements(),
            "Airway area vector has {} entries but expected {}.", area.size(),
            data.number_of_elements());
        FOUR_C_ASSERT_ALWAYS(result.size() == data.number_of_elements(),
            "Poiseuille resistance output has {} entries but expected {}.", result.size(),
            data.number_of_elements());

        const double resistance_factor =
            8.0 * std::numbers::pi * data.air_properties.dynamic_viscosity;
        for (size_t i = 0; i < data.number_of_elements(); ++i)
        {
          const double area_i = area[i];
          result[i] = resistance_factor * data.ref_length[i] / (area_i * area_i);
        }
      }
    };

    double evaluate_poiseuille_resistance(
        const AirwayData& data, const std::vector<double>& area, size_t i)
    {
      return 8 * std::numbers::pi * data.air_properties.dynamic_viscosity * data.ref_length[i] /
             (area[i] * area[i]);
    }

    void assert_output_size(std::span<double> output, size_t expected_size, const char* name)
    {
      FOUR_C_ASSERT_ALWAYS(output.size() == expected_size,
          "{} output has {} entries but expected {}.", name, output.size(), expected_size);
    }

    template <typename Model>
    InertiaEvaluator make_inertia_evaluator_impl(const Model& model)
    {
      using HasInertiaT = decltype(std::declval<Model>().has_inertia);
      static_assert(std::is_same_v<std::remove_reference_t<HasInertiaT>, std::vector<bool>>,
          "Model must have member 'has_inertia' of type std::vector<bool>");

      return [&model](
                 const AirwayData& data, const std::vector<double>& area, std::span<double> inertia)
      {
        FOUR_C_ASSERT_ALWAYS(area.size() == data.number_of_elements(),
            "Airway area vector has {} entries but expected {}.", area.size(),
            data.number_of_elements());
        assert_output_size(inertia, data.number_of_elements(), "Inertia");

        for (size_t i = 0; i < data.number_of_elements(); ++i)
        {
          inertia[i] = 0.0;
          if (i < model.has_inertia.size() && model.has_inertia[i])
          {
            inertia[i] = data.air_properties.density * data.ref_length[i] / area[i];
          }
        }
      };
    }

    void evaluate_nonlinear_flow_resistance_derivative_rigid(const NonLinearResistive& model,
        const AirwayData& data, const Core::LinAlg::Vector<double>& locally_relevant_dofs,
        double dt, std::span<double> resistance_derivative)
    {
      /* Fill caller-owned scratch directly; these values become rigid-airway q coefficients. */
      (void)dt;
      assert_output_size(
          resistance_derivative, data.number_of_elements(), "Rigid flow-resistance derivative");
      const auto dof_values = locally_relevant_dofs.local_values_as_span();
      const double dynamic_viscosity = data.air_properties.dynamic_viscosity;
      const double density = data.air_properties.density;
      const double poiseuille_factor = 8.0 * std::numbers::pi * dynamic_viscosity;
      const double turbulence_factor_base = density / (std::numbers::pi * dynamic_viscosity);
      for (size_t i = 0; i < data.number_of_elements(); i++)
      {
        const double ref_length = data.ref_length[i];
        const double ref_area = data.ref_area[i];
        const double k_turb = model.k_turb[i];
        const double q1 = dof_values[data.lid_q1[i]];
        const double poiseuille_resistance = poiseuille_factor * ref_length / (ref_area * ref_area);
        const double R = poiseuille_resistance * k_turb;

        double dk_dq1;
        if (k_turb > 1.0)
        {
          dk_dq1 = model.turbulence_factor_gamma[i] *
                   std::sqrt(turbulence_factor_base / ref_length) / std::sqrt(std::abs(q1));
        }
        else
        {
          dk_dq1 = 0.0;
        }
        resistance_derivative[i] = (poiseuille_resistance * dk_dq1) * q1 + R;
      }
    }

    void evaluate_linear_flow_resistance_derivative_kelvin_voigt(const LinearResistive& model,
        const AirwayData& data, const Core::LinAlg::Vector<double>& dofs,
        const std::vector<double>& area, double dt, std::span<double> resistance_derivative_q1,
        std::span<double> resistance_derivative_q2)
    {
      /* Kelvin-Voigt derivatives are produced in q1/q2 batches for structured replacement. */
      (void)model;
      FOUR_C_ASSERT_ALWAYS(area.size() == data.number_of_elements(),
          "Airway area vector has {} entries but expected {}.", area.size(),
          data.number_of_elements());
      assert_output_size(resistance_derivative_q1, data.number_of_elements(),
          "Kelvin-Voigt q1 flow-resistance derivative");
      assert_output_size(resistance_derivative_q2, data.number_of_elements(),
          "Kelvin-Voigt q2 flow-resistance derivative");
      const auto dof_values = dofs.local_values_as_span();
      const double dynamic_viscosity = data.air_properties.dynamic_viscosity;
      const double poiseuille_factor = 8.0 * std::numbers::pi * dynamic_viscosity;
      const double poiseuille_area_derivative_factor = -16.0 * std::numbers::pi * dynamic_viscosity;
      for (size_t i = 0; i < data.number_of_elements(); i++)
      {
        const double ref_length = data.ref_length[i];
        const double area_i = area[i];
        const double area_squared = area_i * area_i;
        const double area_cubed = area_squared * area_i;
        const double poiseuille_resistance = poiseuille_factor * ref_length / area_squared;
        const double dRp_da = poiseuille_area_derivative_factor * ref_length / area_cubed;
        const double da_dq1 = dt / ref_length;
        const double da_dq2 = -dt / ref_length;
        const double q_sum = dof_values[data.lid_q1[i]] + dof_values[data.lid_q2[i]];
        resistance_derivative_q1[i] = -0.5 * (dRp_da * da_dq1 * q_sum + poiseuille_resistance);
        resistance_derivative_q2[i] = -0.5 * (dRp_da * da_dq2 * q_sum + poiseuille_resistance);
      }
    }

    void evaluate_nonlinear_flow_resistance_derivative_kelvin_voigt(const NonLinearResistive& model,
        const AirwayData& data, const Core::LinAlg::Vector<double>& dofs,
        const std::vector<double>& area, double dt, std::span<double> resistance_derivative_q1,
        std::span<double> resistance_derivative_q2)
    {
      FOUR_C_ASSERT_ALWAYS(area.size() == data.number_of_elements(),
          "Airway area vector has {} entries but expected {}.", area.size(),
          data.number_of_elements());
      assert_output_size(resistance_derivative_q1, data.number_of_elements(),
          "Kelvin-Voigt q1 flow-resistance derivative");
      assert_output_size(resistance_derivative_q2, data.number_of_elements(),
          "Kelvin-Voigt q2 flow-resistance derivative");
      const auto dof_values = dofs.local_values_as_span();
      const double dynamic_viscosity = data.air_properties.dynamic_viscosity;
      const double density = data.air_properties.density;
      const double poiseuille_factor = 8.0 * std::numbers::pi * dynamic_viscosity;
      const double poiseuille_area_derivative_factor = -16.0 * std::numbers::pi * dynamic_viscosity;
      const double turbulence_factor_base = density / (2.0 * std::numbers::pi * dynamic_viscosity);
      for (size_t i = 0; i < data.number_of_elements(); i++)
      {
        const double ref_length = data.ref_length[i];
        const double area_i = area[i];
        const double area_squared = area_i * area_i;
        const double area_cubed = area_squared * area_i;
        const double k_turb = model.k_turb[i];
        const double q1 = dof_values[data.lid_q1[i]];
        const double q2 = dof_values[data.lid_q2[i]];
        const double q_sum = q1 + q2;
        const double q_difference = q2 - q1;
        const double poiseuille_resistance = poiseuille_factor * ref_length / area_squared;
        const double dRp_da = poiseuille_area_derivative_factor * ref_length / area_cubed;
        const double da_dq1 = dt / ref_length;
        const double da_dq2 = -dt / ref_length;
        double dk_dq1;
        if (k_turb > 1.0)
        {
          const double abs_q_sum = std::abs(q_sum);
          dk_dq1 = model.turbulence_factor_gamma[i] *
                   std::sqrt(turbulence_factor_base / ref_length) * q_sum /
                   (abs_q_sum * std::sqrt(abs_q_sum));
        }
        else
        {
          dk_dq1 = 0.0;
        }
        const double dk_dq2 = dk_dq1;
        const double alpha_denominator = 4.0 * k_turb - 1.0;
        const double dalpha_dk = -4.0 / (alpha_denominator * alpha_denominator);
        const double alpha = 4.0 * k_turb / alpha_denominator;
        const double density_alpha_over_area_squared = density * alpha / area_squared;
        const double density_q_difference_over_area_squared = density * q_difference / area_squared;
        const double resistance =
            poiseuille_resistance * k_turb + 2.0 * density_alpha_over_area_squared * q_difference;
        resistance_derivative_q1[i] =
            -0.5 * ((dRp_da * da_dq1 * k_turb + poiseuille_resistance * dk_dq1 +
                        2.0 * density_q_difference_over_area_squared * dalpha_dk * dk_dq1 -
                        2.0 * density_alpha_over_area_squared -
                        4.0 * density * alpha / area_cubed * da_dq1 * q_difference) *
                           q_sum +
                       resistance);
        resistance_derivative_q2[i] =
            -0.5 * ((dRp_da * da_dq2 * k_turb + poiseuille_resistance * dk_dq2 +
                        2.0 * density_q_difference_over_area_squared * dalpha_dk * dk_dq2 +
                        2.0 * density_alpha_over_area_squared -
                        4.0 * density * alpha / area_cubed * da_dq2 * q_difference) *
                           q_sum +
                       resistance);
      }
    }

    void evaluate_inertia_derivative_kelvin_voigt(const std::vector<bool>& has_inertia,
        const AirwayData& data, const Core::LinAlg::Vector<double>& locally_relevant_dofs,
        const KelvinVoigtWall& wall, double dt, std::span<double> inertia_q1,
        std::span<double> inertia_q2)
    {
      assert_output_size(
          inertia_q1, data.number_of_elements(), "Kelvin-Voigt q1 inertia derivative");
      assert_output_size(
          inertia_q2, data.number_of_elements(), "Kelvin-Voigt q2 inertia derivative");
      const auto dof_values = locally_relevant_dofs.local_values_as_span();
      const double density = data.air_properties.density;
      const double inertia_factor = -0.5 / dt;
      for (size_t i = 0; i < data.number_of_elements(); ++i)
      {
        inertia_q1[i] = 0.0;
        inertia_q2[i] = 0.0;
        if (i < has_inertia.size() && has_inertia[i])
        {
          const double ref_length = data.ref_length[i];
          const double area_i = wall.area[i];
          const double area_squared = area_i * area_i;
          const double dI_da = -density * ref_length / area_squared;
          const double da_dq1 = dt / ref_length;
          const double da_dq2 = -dt / ref_length;
          const double inertia = density * ref_length / area_i;
          const double q_history_difference =
              dof_values[data.lid_q1[i]] + dof_values[data.lid_q2[i]] - data.q1_n[i] - data.q2_n[i];
          inertia_q1[i] = inertia_factor * (inertia + dI_da * da_dq1 * q_history_difference);
          inertia_q2[i] = inertia_factor * (inertia + dI_da * da_dq2 * q_history_difference);
        }
      }
    }

    template <typename Model>
    InertiaDerivativeEvaluatorKelvinVoigt make_inertia_derivative_evaluator_kelvin_voigt_impl(
        const Model& model, KelvinVoigtWall& kelvin_voigt_wall_model)
    {
      using HasInertiaT = decltype(std::declval<Model>().has_inertia);
      static_assert(std::is_same_v<std::remove_reference_t<HasInertiaT>, std::vector<bool>>,
          "Model must have member 'has_inertia' of type std::vector<bool>");

      KelvinVoigtWall* wall = &kelvin_voigt_wall_model;
      return [wall, &model](const AirwayData& data, const Core::LinAlg::Vector<double>& dofs,
                 double dt, std::span<double> inertia_q1, std::span<double> inertia_q2)
      {
        evaluate_inertia_derivative_kelvin_voigt(
            model.has_inertia, data, dofs, *wall, dt, inertia_q1, inertia_q2);
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
                       const std::vector<double>& area, std::span<double> resistance)
            { ComputePoiseuilleResistance{}(data, area, resistance); };
          }
          else if constexpr (std::is_same_v<ModelType, NonLinearResistive>)
          {
            return [&model](const AirwayData& data, const Core::LinAlg::Vector<double>&,
                       const std::vector<double>& area, std::span<double> resistance)
            {
              FOUR_C_ASSERT_ALWAYS(area.size() == data.number_of_elements(),
                  "Airway area vector has {} entries but expected {}.", area.size(),
                  data.number_of_elements());
              assert_output_size(resistance, data.number_of_elements(), "Flow resistance");
              for (size_t i = 0; i < data.number_of_elements(); ++i)
              {
                resistance[i] = evaluate_poiseuille_resistance(data, area, i) * model.k_turb[i];
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

  FlowResistanceEvaluator make_flow_resistance_evaluator_kelvin_voigt(FlowModel& flow_model)
  {
    return std::visit(
        [](const auto& model) -> FlowResistanceEvaluator
        {
          using ModelType = std::decay_t<decltype(model)>;
          if constexpr (std::is_same_v<ModelType, LinearResistive>)
          {
            return [](const AirwayData& data, const Core::LinAlg::Vector<double>&,
                       const std::vector<double>& area, std::span<double> resistance)
            { ComputePoiseuilleResistance{}(data, area, resistance); };
          }
          else if constexpr (std::is_same_v<ModelType, NonLinearResistive>)
          {
            return [&model](const AirwayData& data, const Core::LinAlg::Vector<double>& dofs,
                       const std::vector<double>& area, std::span<double> resistance)
            {
              FOUR_C_ASSERT_ALWAYS(area.size() == data.number_of_elements(),
                  "Airway area vector has {} entries but expected {}.", area.size(),
                  data.number_of_elements());
              assert_output_size(resistance, data.number_of_elements(), "Flow resistance");
              for (size_t i = 0; i < data.number_of_elements(); ++i)
              {
                const double alpha = 4.0 * model.k_turb[i] / (4.0 * model.k_turb[i] - 1.0);
                resistance[i] = evaluate_poiseuille_resistance(data, area, i) * model.k_turb[i] +
                                2 * data.air_properties.density * alpha / (area[i] * area[i]) *
                                    (dofs.local_values_as_span()[data.lid_q2[i]] -
                                        dofs.local_values_as_span()[data.lid_q1[i]]);
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

  FlowResistanceDerivativeEvaluatorRigid make_flow_resistance_derivative_evaluator_rigid(
      FlowModel& flow_model)
  {
    return std::visit(
        [](const auto& model) -> FlowResistanceDerivativeEvaluatorRigid
        {
          using ModelType = std::decay_t<decltype(model)>;
          if constexpr (std::is_same_v<ModelType, LinearResistive>)
          {
            return [](const AirwayData& data, const Core::LinAlg::Vector<double>&, double,
                       std::span<double> resistance_derivative)
            { ComputePoiseuilleResistance{}(data, data.ref_area, resistance_derivative); };
          }
          else if constexpr (std::is_same_v<ModelType, NonLinearResistive>)
          {
            return [&model](const AirwayData& data, const Core::LinAlg::Vector<double>& dofs,
                       double dt, std::span<double> resistance_derivative)
            {
              evaluate_nonlinear_flow_resistance_derivative_rigid(
                  model, data, dofs, dt, resistance_derivative);
            };
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
            return [wall, &model](const AirwayData& data, const Core::LinAlg::Vector<double>& dofs,
                       double dt, std::span<double> resistance_derivative_q1,
                       std::span<double> resistance_derivative_q2)
            {
              evaluate_linear_flow_resistance_derivative_kelvin_voigt(model, data, dofs, wall->area,
                  dt, resistance_derivative_q1, resistance_derivative_q2);
            };
          }
          else if constexpr (std::is_same_v<ModelType, NonLinearResistive>)
          {
            return [wall, &model](const AirwayData& data, const Core::LinAlg::Vector<double>& dofs,
                       double dt, std::span<double> resistance_derivative_q1,
                       std::span<double> resistance_derivative_q2)
            {
              evaluate_nonlinear_flow_resistance_derivative_kelvin_voigt(model, data, dofs,
                  wall->area, dt, resistance_derivative_q1, resistance_derivative_q2);
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

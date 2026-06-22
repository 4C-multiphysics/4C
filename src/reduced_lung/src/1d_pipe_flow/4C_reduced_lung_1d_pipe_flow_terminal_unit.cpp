// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_reduced_lung_1d_pipe_flow_terminal_unit.hpp"

#include "4C_reduced_lung_1d_pipe_flow_input.hpp"
#include "4C_utils_exceptions.hpp"

#include <cmath>

FOUR_C_NAMESPACE_OPEN
namespace ReducedLung1DPipe
{
  namespace TerminalUnit
  {

    double evaluate_linear_elastic_pressure(LinearElasticity& linear_elastic_model,
        const RheologicalElasticityModel& terminal_unit_model, const double Q, const double dt)
    {
      linear_elastic_model.elastic_pressure_p_el =
          linear_elastic_model.elasticity_E *
          ((terminal_unit_model.volume_v + dt * Q) / terminal_unit_model.reference_volume_v0 - 1);
      return linear_elastic_model.elastic_pressure_p_el;
    }

    double evaluate_linear_elastic_pressure_gradient(LinearElasticity& linear_elastic_model,
        const RheologicalElasticityModel& terminal_unit_model, const double dQdA, const double dt)
    {
      linear_elastic_model.elastic_pressure_grad_dp_el =
          linear_elastic_model.elasticity_E * dt / terminal_unit_model.reference_volume_v0 * (dQdA);
      return linear_elastic_model.elastic_pressure_grad_dp_el;
    }

    double evaluate_ogden_hyperelastic_pressure(OgdenHyperelasticity& ogden_hyperelastic_model,
        const double Q, const RheologicalElasticityModel& terminal_unit_model, const double dt)
    {
      const double v0_over_v =
          terminal_unit_model.reference_volume_v0 / (terminal_unit_model.volume_v + dt * Q);
      ogden_hyperelastic_model.elastic_pressure_p_el =
          ogden_hyperelastic_model.bulk_modulus_kappa /
          ogden_hyperelastic_model.nonlinear_stiffening_beta * v0_over_v *
          (1 - std::pow(v0_over_v, ogden_hyperelastic_model.nonlinear_stiffening_beta));
      return ogden_hyperelastic_model.elastic_pressure_p_el;
    }

    double evaluate_ogden_hyperelastic_pressure_gradient(
        OgdenHyperelasticity& ogden_hyperelastic_model,
        const RheologicalElasticityModel& terminal_unit_model, const double Q, const double dQdA,
        const double dt)
    {
      const double v0_over_v =
          terminal_unit_model.reference_volume_v0 / (terminal_unit_model.volume_v + dt * Q);
      ogden_hyperelastic_model.elastic_pressure_grad_dp_el =
          ogden_hyperelastic_model.bulk_modulus_kappa * dt /
          ogden_hyperelastic_model.nonlinear_stiffening_beta *
          (1 - std::pow(v0_over_v, ogden_hyperelastic_model.nonlinear_stiffening_beta) *
                   (1 + ogden_hyperelastic_model.nonlinear_stiffening_beta)) *
          (-v0_over_v / (terminal_unit_model.volume_v + dt * Q)) * dQdA;
      return ogden_hyperelastic_model.elastic_pressure_grad_dp_el;
    }

    double evaluate_kelvin_voigt_residual(double area_A, double velocity_u,
        const double reference_area_A0, const double beta, const double Pext,
        const KelvinVoigt& kelvin_voigt_model, const double elastic_pressure_p_el,
        const RheologicalElasticityModel& terminal_unit_model)
    {
      // p_1D == p_0D && Q_1D == Q_0D
      const double p_1D = beta * (std::sqrt(area_A) - std::sqrt(reference_area_A0)) + Pext;
      const double flow_Q = area_A * velocity_u;

      const double p_0D = elastic_pressure_p_el + kelvin_voigt_model.viscosity_eta * flow_Q /
                                                      terminal_unit_model.reference_volume_v0;
      return (p_1D - p_0D);
    }

    double evaluate_kelvin_voigt_jacobian(const double area_A, const double dQdA, const double beta,
        const KelvinVoigt& kelvin_voigt_model, const double dp_el_dA,
        const RheologicalElasticityModel& terminal_unit_model)
    {
      // compute (par f) / (par A)
      const double dpdA_1D = 0.5 * beta / std::sqrt(area_A);

      const double dpdA_0D = dp_el_dA + kelvin_voigt_model.viscosity_eta /
                                            terminal_unit_model.reference_volume_v0 * dQdA;
      return dpdA_1D - dpdA_0D;
    }

    double evaluate_four_element_maxwell_residual(const double area_A, const double velocity_u,
        const double reference_area_A0, const double beta, const double Pext,
        const FourElementMaxwell& four_element_maxwell_model, const double elastic_pressure_p_el,
        const double dt, const RheologicalElasticityModel& terminal_unit_model)
    {
      // p_1D == p_0D && Q_1D == Q_0D
      const double p_1D = beta * (std::sqrt(area_A) - std::sqrt(reference_area_A0)) + Pext;
      const double flow_Q = area_A * velocity_u;

      const double p_0D = elastic_pressure_p_el +
                          (four_element_maxwell_model.viscosity_eta +
                              (four_element_maxwell_model.elasticity_E_m * dt *
                                  four_element_maxwell_model.viscosity_eta_m) /
                                  (four_element_maxwell_model.elasticity_E_m * dt +
                                      four_element_maxwell_model.viscosity_eta_m)) *
                              flow_Q / terminal_unit_model.reference_volume_v0 +
                          four_element_maxwell_model.viscosity_eta_m /
                              (four_element_maxwell_model.elasticity_E_m * dt +
                                  four_element_maxwell_model.viscosity_eta_m) *
                              four_element_maxwell_model.maxwell_pressure_p_m;
      return p_1D - p_0D;
    }

    double evaluate_four_element_maxwell_jacobian(const double area_A, const double dQdA,
        const double beta, const FourElementMaxwell& four_element_maxwell_model,
        const double dp_el_dA, const double dt,
        const RheologicalElasticityModel& terminal_unit_model)
    {
      // compute (par f) / (par A)
      const double dpdA_1D = 0.5 * beta / std::sqrt(area_A);

      const double dpdA_0D = dp_el_dA + (four_element_maxwell_model.viscosity_eta +
                                            (four_element_maxwell_model.elasticity_E_m * dt *
                                                four_element_maxwell_model.viscosity_eta_m) /
                                                (four_element_maxwell_model.elasticity_E_m * dt +
                                                    four_element_maxwell_model.viscosity_eta_m)) *
                                            dQdA / terminal_unit_model.reference_volume_v0;
      return dpdA_1D - dpdA_0D;
    }

    ElasticityModel create_elasticity_model(
        const ReducedLung1dPipeFlow::Parameters::TerminalUnits::ElasticityModel& elasticity_model,
        const int global_id)
    {
      if (elasticity_model.elasticity_model_type.at(global_id) ==
          ReducedLung1dPipeFlow::Parameters::TerminalUnits::ElasticityModel::ElasticityModelType::
              Linear)
      {
        return LinearElasticity{.elasticity_E = elasticity_model.linear.elasticity_e.at(global_id),
            .elastic_pressure_p_el = 0.0,
            .elastic_pressure_grad_dp_el = 0.0};
      }
      else if (elasticity_model.elasticity_model_type.at(global_id) ==
               ReducedLung1dPipeFlow::Parameters::TerminalUnits::ElasticityModel::
                   ElasticityModelType::Ogden)
      {
        return OgdenHyperelasticity{
            .bulk_modulus_kappa = elasticity_model.ogden.ogden_parameter_kappa.at(global_id),
            .nonlinear_stiffening_beta = elasticity_model.ogden.ogden_parameter_beta.at(global_id),
            .elastic_pressure_p_el = 0.0,
            .elastic_pressure_grad_dp_el = 0.0};
      }
      FOUR_C_THROW(
          "Unsupported terminal unit elasticity model type for global_id = %d.", global_id);
    }

    RheologicalModel create_rheological_model(
        const ReducedLung1dPipeFlow::Parameters::TerminalUnits::RheologicalModel& rheology_model,
        const int global_id)
    {
      if (rheology_model.rheological_model_type.at(global_id) ==
          ReducedLung1dPipeFlow::Parameters::TerminalUnits::RheologicalModel::RheologicalModelType::
              KelvinVoigt)
      {
        return KelvinVoigt{
            .viscosity_eta = rheology_model.kelvin_voigt.viscosity_kelvin_voigt_eta.at(global_id)};
      }
      else if (rheology_model.rheological_model_type.at(global_id) ==
               ReducedLung1dPipeFlow::Parameters::TerminalUnits::RheologicalModel::
                   RheologicalModelType::FourElementMaxwell)
      {
        return FourElementMaxwell{
            .viscosity_eta =
                rheology_model.four_element_maxwell.viscosity_kelvin_voigt_eta.at(global_id),
            .elasticity_E_m =
                rheology_model.four_element_maxwell.elasticity_maxwell_e_m.at(global_id),
            .viscosity_eta_m =
                rheology_model.four_element_maxwell.viscosity_maxwell_eta_m.at(global_id),
            .maxwell_pressure_p_m = 0.0};
      }
      FOUR_C_THROW(
          "Unsupported terminal unit rheological model type for global_id = %d.", global_id);
    }

    /**
     * Creates a Windkessel model from input parameters.
     */
    WindkesselModel create_windkessel_model(
        const ReducedLung1dPipeFlow::Parameters::TerminalUnits::WindkesselModel& windkessel_model,
        const int global_id)
    {
      return WindkesselModel{
          .proximal_resistance_R_p = windkessel_model.proximal_resistance_R_p.at(global_id),
          .compliance_C = windkessel_model.compliance_C.at(global_id),
          .distal_resistance_R_d = windkessel_model.distal_resistance_R_d.at(global_id),
          .pressure_peripheral = windkessel_model.pressure_peripheral.at(global_id),
          .windkessel_pressure_wk = windkessel_model.pressure_peripheral.at(global_id),
      };
    }

    /**
     * Evaluates the Windkessel model residual.
     * Residual: F = Q - (p_1D - PC^n) / R1 = 0
     */
    double evaluate_windkessel_residual(const double area_A, const double velocity_u,
        const double reference_area_A0, const double beta, const double Pext,
        const WindkesselModel& wk_model)
    {
      // Use windkessel element pressure from previous time step
      const double p_c_n = wk_model.windkessel_pressure_wk;

      // Flow Q = A*u extrapolated from 1D domain
      const double flow_Q = area_A * velocity_u;

      // p_1D from 1D domain
      const double p_1D = beta * (std::sqrt(area_A) - std::sqrt(reference_area_A0)) + Pext;

      // Residual: Q - (p_1D - PC^n) / R1 = 0
      return flow_Q - (p_1D - p_c_n) / wk_model.proximal_resistance_R_p;
    }

    /**
     * Evaluates the Windkessel model jacobian df/dA.
     *
     * Residual: F = Q - (p_1D - PC^n) / R1
     * Since PC^n = const during solve, dPC/dA = 0
     * dF/dA = dQ/dA - (1/R1) * dp_1D/dA
     */
    double evaluate_windkessel_jacobian(
        const double area_A, const double dQdA, const double beta, const WindkesselModel& wk_model)
    {
      // dp_1D/dA
      const double dpdA_1D = 0.5 * beta / std::sqrt(area_A);

      // dF/dA = dQ/dA - (1/R1) * dp_1D/dA (PC^n is frozen, so dPC/dA = 0)
      return dQdA - dpdA_1D / wk_model.proximal_resistance_R_p;
    }

    double TerminalUnitModel::evaluate_elastic_pressure(const double area_A, const double beta,
        const double density_rho, const double characteristic_W_outgoing, double dt)
    {
      // Only applicable for RheologicalElasticityModels
      auto* rheol_elastic = std::get_if<RheologicalElasticityModel>(&model);
      if (!rheol_elastic)
      {
        FOUR_C_ASSERT(false, "Model is not RheologicalElasticityModel");
      }

      const double velocity_u =
          characteristic_W_outgoing - 4 * std::pow(area_A, 0.25) * sqrt(0.5 * beta / density_rho);
      double flow_Q = velocity_u * area_A;
      // compute elastic pressure depending on ElasticityModel
      return std::visit(
          [&](auto& elasticity) -> double
          {
            using T = std::decay_t<decltype(elasticity)>;

            if constexpr (std::is_same_v<T, LinearElasticity>)
            {
              return evaluate_linear_elastic_pressure(elasticity, *rheol_elastic, flow_Q, dt);
            }
            else if constexpr (std::is_same_v<T, OgdenHyperelasticity>)
            {
              return evaluate_ogden_hyperelastic_pressure(elasticity, flow_Q, *rheol_elastic, dt);
            }
          },
          rheol_elastic->elasticity_model);
    }

    std::pair<double, double> TerminalUnitModel::evaluate_residual_jacobian(double area_A,
        const double reference_area_A0, const double beta, double Pext, const double density_rho,
        const double characteristic_W_outgoing, double dt)
    {
      const double velocity_u =
          characteristic_W_outgoing - 4 * std::pow(area_A, 0.25) * sqrt(0.5 * beta / density_rho);
      double flow_Q = velocity_u * area_A;
      double dQdA = velocity_u - std::pow(area_A, 0.25) * sqrt(0.5 * beta / density_rho);

      // Handle Windkessel model
      auto* windkessel = std::get_if<WindkesselModel>(&model);
      if (windkessel)
      {
        double residual = evaluate_windkessel_residual(
            area_A, velocity_u, reference_area_A0, beta, Pext, *windkessel);
        double jacobian = evaluate_windkessel_jacobian(area_A, dQdA, beta, *windkessel);
        return std::make_pair(residual, jacobian);
      }

      // Handle RheologicalElasticityModels
      auto* rheol_elastic = std::get_if<RheologicalElasticityModel>(&model);
      if (!rheol_elastic)
      {
        FOUR_C_THROW("Unknown terminal unit model type.");
      }

      double p_el = 0.0;
      double dp_eldA = 0.0;
      // compute elastic pressure depending on ElasticityModel
      std::visit(
          [&](auto& elasticity)
          {
            using T = std::decay_t<decltype(elasticity)>;

            if constexpr (std::is_same_v<T, LinearElasticity>)
            {
              p_el = evaluate_linear_elastic_pressure(elasticity, *rheol_elastic, flow_Q, dt);
              dp_eldA =
                  evaluate_linear_elastic_pressure_gradient(elasticity, *rheol_elastic, dQdA, dt);
            }
            else if constexpr (std::is_same_v<T, OgdenHyperelasticity>)
            {
              p_el = evaluate_ogden_hyperelastic_pressure(elasticity, flow_Q, *rheol_elastic, dt);
              dp_eldA = evaluate_ogden_hyperelastic_pressure_gradient(
                  elasticity, *rheol_elastic, flow_Q, dQdA, dt);
            }
          },
          rheol_elastic->elasticity_model);

      // compute residual and jacobian depending on rheological model
      return std::visit(
          [&](auto& rheology) -> std::pair<double, double>
          {
            using T = std::decay_t<decltype(rheology)>;

            if constexpr (std::is_same_v<T, KelvinVoigt>)
            {
              double residual = evaluate_kelvin_voigt_residual(area_A, velocity_u,
                  reference_area_A0, beta, Pext, rheology, p_el, *rheol_elastic);
              double jacobian = evaluate_kelvin_voigt_jacobian(
                  area_A, dQdA, beta, rheology, dp_eldA, *rheol_elastic);
              return std::make_pair(residual, jacobian);
            }
            else if constexpr (std::is_same_v<T, FourElementMaxwell>)
            {
              double residual = evaluate_four_element_maxwell_residual(area_A, velocity_u,
                  reference_area_A0, beta, Pext, rheology, p_el, dt, *rheol_elastic);
              double jacobian = evaluate_four_element_maxwell_jacobian(
                  area_A, dQdA, beta, rheology, dp_eldA, dt, *rheol_elastic);
              return std::make_pair(residual, jacobian);
            }
            else
            {
              FOUR_C_ASSERT(false, "Wrongly defined rheological model.");
            }
          },
          rheol_elastic->rheological_model);
    }

    void TerminalUnitModel::update_terminal_unit_data(const double flow_Q, const double dt,
        const double area_A, const double beta, const double density_rho,
        const double characteristic_W_outgoing)
    {
      // Handle Windkessel model
      auto* windkessel = std::get_if<WindkesselModel>(&model);
      if (windkessel)
      {
        // Update PC after convergence using explicit Euler
        const double old_pc = windkessel->windkessel_pressure_wk;  // PC^n (frozen value)
        const double dt_over_C = dt / windkessel->compliance_C;
        windkessel->windkessel_pressure_wk =
            old_pc + dt_over_C * (flow_Q - (old_pc - windkessel->pressure_peripheral) /
                                               windkessel->distal_resistance_R_d);

        return;
      }

      // Handle RheologicalElasticityModels
      auto* rheol_elastic = std::get_if<RheologicalElasticityModel>(&model);
      if (rheol_elastic)
      {
        rheol_elastic->volume_v += dt * flow_Q;
        // update p_m in FourElementMaxwellElement
        auto* four_element_maxwell =
            std::get_if<FourElementMaxwell>(&rheol_elastic->rheological_model);
        if (four_element_maxwell)
        {
          double p_m = four_element_maxwell->maxwell_pressure_p_m;
          double p_el =
              evaluate_elastic_pressure(area_A, beta, density_rho, characteristic_W_outgoing, dt);
          four_element_maxwell->maxwell_pressure_p_m =
              four_element_maxwell->viscosity_eta_m /
                  (four_element_maxwell->elasticity_E_m * dt +
                      four_element_maxwell->viscosity_eta_m) *
                  p_m +
              four_element_maxwell->elasticity_E_m * dt /
                  (four_element_maxwell->elasticity_E_m * dt +
                      four_element_maxwell->viscosity_eta_m) *
                  p_el;
        }
      }
    }

    TerminalUnitModel create_terminal_unit_model(
        const ReducedLung1dPipeFlow::Parameters& params, int global_id, int node_owner)
    {
      TerminalUnitModel terminal_unit_model;
      // TerminalUnitData
      terminal_unit_model.data.global_node_id = global_id;
      terminal_unit_model.data.node_owner = node_owner;

      // Check terminal unit type
      const auto& tu_params = params.terminal_units;
      auto tu_type = tu_params.terminal_unit_type.at(global_id);

      if (tu_type == ReducedLung1dPipeFlow::Parameters::TerminalUnits::TerminalUnitType::Windkessel)
      {
        // Create Windkessel model
        WindkesselModel wk = create_windkessel_model(tu_params.windkessel_model, global_id);
        terminal_unit_model.model = wk;
      }
      else if (tu_type == ReducedLung1dPipeFlow::Parameters::TerminalUnits::TerminalUnitType::
                              RheologicalElasticity)
      {
        // Create rheological-elasticity model
        RheologicalElasticityModel rheol_elastic;
        rheol_elastic.volume_v =
            params.terminal_units.elasticity_model.acinar_volume_v.at(global_id);
        rheol_elastic.reference_volume_v0 = rheol_elastic.volume_v;

        rheol_elastic.rheological_model =
            create_rheological_model(tu_params.rheological_model, global_id);
        rheol_elastic.elasticity_model =
            create_elasticity_model(tu_params.elasticity_model, global_id);
        terminal_unit_model.model = rheol_elastic;
      }
      else
      {
        FOUR_C_THROW("Unknown terminal unit for global_id %d: parsed terminal_unit_type=%d",
            global_id, static_cast<int>(tu_type));
      }

      return terminal_unit_model;
    }

  }  // namespace TerminalUnit
}  // namespace ReducedLung1DPipe
FOUR_C_NAMESPACE_CLOSE
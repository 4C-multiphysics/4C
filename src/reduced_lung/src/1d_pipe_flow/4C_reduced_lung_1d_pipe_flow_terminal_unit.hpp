// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_REDUCED_LUNG_1D_PIPE_FLOW_TERMINAL_UNIT_HPP
#define FOUR_C_REDUCED_LUNG_1D_PIPE_FLOW_TERMINAL_UNIT_HPP

#include "4C_config.hpp"

#include "4C_reduced_lung_1d_pipe_flow_input.hpp"

#include <variant>

FOUR_C_NAMESPACE_OPEN
namespace ReducedLung1DPipe
{
  namespace TerminalUnit
  {
    /**
     * @brief Container for all terminal unit elements.
     *
     * Stores identifiers for each terminal unit,
     * which are used across all rheological and elasticity models.
     */
    struct TerminalUnitData
    {
      // nodal ID
      int global_node_id;
      // Owner
      int node_owner;
    };

    // ----- Rheological models of viscoelasticity -----

    /**
     * @brief Rheological Kelvin-Voigt model (spring and dashpot in parallel).
     */
    struct KelvinVoigt
    {
      double viscosity_eta;
    };

    /**
     * @brief Rheological four-element Maxwell model (Kelvin-Voigt body and Maxwell body in
     * parallel).
     *
     * The Maxwell body introduces an additional internal pressure state p_m, which is stored in
     * this model and updated after every time step.
     */
    struct FourElementMaxwell
    {
      double viscosity_eta;
      double elasticity_E_m;
      double viscosity_eta_m;
      // Internal pressure state of the maxwell body
      double maxwell_pressure_p_m;
    };

    // ----- Specialized Elasticity models -----

    /**
     * @brief Linear elasticity model. Substitutes a spring in a rheological model.
     *
     * Computes elastic pressure as linear function of volume.
     * Includes internal variables for pressure and its gradient.
     */
    struct LinearElasticity
    {
      double elasticity_E;
      // Internal elastic pressure state
      double elastic_pressure_p_el;
      double elastic_pressure_grad_dp_el;
    };

    /**
     * @brief Ogden-type hyperelastic model. Substitutes a spring in a rheological model.
     *
     * Nonlinear stiffening behavior based on a bulk modulus kappa and shape parameter beta.
     * Includes internal variables for pressure and pressure gradient.
     *
     * @note Ill-posed for beta = 0!
     */
    struct OgdenHyperelasticity
    {
      double bulk_modulus_kappa;
      double nonlinear_stiffening_beta;
      // Internal elastic pressure state
      double elastic_pressure_p_el;
      double elastic_pressure_grad_dp_el;
    };

    // Model Types
    using RheologicalModel = std::variant<KelvinVoigt, FourElementMaxwell>;
    using ElasticityModel = std::variant<LinearElasticity, OgdenHyperelasticity>;

    /**
     * @brief Windkessel model (3-element RCR model).
     *
     * The model consists of a proximal resistance R_p, a compliance C, and a distal resistance R_d.
     * The pressure at the periphery (after R_d) is given by pressure_peripheral.
     * The internal state variable windkessel_pressure_wk represents the pressure in the
     * 0D windkessel model.
     */
    struct WindkesselModel
    {
      double proximal_resistance_R_p;
      double compliance_C;
      double distal_resistance_R_d;
      double pressure_peripheral;
      double windkessel_pressure_wk;
    };

    /**
     * @brief Combination of rheological and elasticity models for terminal units.
     */
    struct RheologicalElasticityModel
    {
      RheologicalModel rheological_model;
      ElasticityModel elasticity_model;
      double volume_v;
      double reference_volume_v0;
    };

    /**
     * @brief Variant type for terminal unit models.
     */
    using TerminalUnitModelVariant = std::variant<RheologicalElasticityModel, WindkesselModel>;

    /**
     * @brief Encapsulates a terminal unit model.
     *
     * Stores the element-specific data and associated evaluation functions
     * (residuals, jacobians, internal state updates, end-of-timestep routines) for a given model
     * combination. Every terminal-unit specific information or action can be found here.
     */
    struct TerminalUnitModel
    {
      TerminalUnitData data;
      TerminalUnitModelVariant model;

      /**
       * Evaluates the residual and the jacobian on a terminal unit depending on area A.
       * @param area_A Updated through Newton iterations.
       * @param reference_area_A0 Reference area of 1D terminal unit.
       * @param beta wall stiffness parameter
       * @param Pext External pressure at the terminal 1D node.
       * @param density_rho Density of the fluid.
       * @param characteristic_W_outgoing Characteristic traveling outside the 1D domain.
       * @param dt Time step size.
       * @return <residual, jacobian>
       */
      std::pair<double, double> evaluate_residual_jacobian(double area_A, double reference_area_A0,
          double beta, double Pext, double density_rho, double characteristic_W_outgoing,
          double dt);

      /**
       * Evaluates the elastic pressure depending on the chosen ElasticityModel.
       * @param area_A Updated through Newton iterations.
       * @param beta Wall stiffness parameter.
       * @param density_rho Density of the fluid.
       * @param characteristic_W_outgoing Characteristic wave traveling outside the 1D domain.
       * @param dt Time step size.
       * @return p_el
       */
      double evaluate_elastic_pressure(double area_A, double beta, double density_rho,
          double characteristic_W_outgoing, double dt);

      /**
       * Updates the data stored in the TerminalUnit container, depending on the respective
       * rheological model.
       * @param flow_Q Flow condition applied to the boundary node.
       * @param dt Time step size.
       * @param area_A Area value the NewtonRaphson converged to.
       * @param beta Wall stiffness parameter.
       * @param density_rho Density of the fluid
       * @param characteristic_W_outgoing Characteristic wave traveling outside the 1D domain.
       */
      void update_terminal_unit_data(double flow_Q, double dt, double area_A, double beta,
          double density_rho, double characteristic_W_outgoing);
    };

    TerminalUnitModel create_terminal_unit_model(
        const ReducedLung1dPipeFlow::Parameters& params, int global_id, int node_owner);

    /**
     * Evaluates the linear elastic pressure if ElasticityModel == LinearElasticity.
     * @param linear_elastic_model Chosen model: LinearElasticity
     * @param data TerminalUnitData storing geometrical information of terminal node.
     * @param Q Flow depending on area: Q = u(A) * A
     * @param dt Time step size.
     * @return p_el
     */
    double evaluate_linear_elastic_pressure(LinearElasticity& linear_elastic_model,
        const RheologicalElasticityModel& terminal_unit_model, double Q, double dt);

    /**
     * Evaluates the linear elastic pressure gradient with respect to area A if ElasticityModel ==
     * LinearElasticity.
     * @param linear_elastic_model Chosen model: LinearElasticity.
     * @param data TerminalUnitData storing geometrical information of terminal node.
     * @param dQdA Derivative of flow with respect  to area A.
     * @param dt Time step size.
     * @return dpel_dA
     */
    double evaluate_linear_elastic_pressure_gradient(LinearElasticity& linear_elastic_model,
        const RheologicalElasticityModel& terminal_unit_model, double dQdA, double dt);

    /**
     * Evaluates the Ogden Hyperelastic pressure p_el.
     * @param ogden_hyperelastic_model  Chosen model: OgdenHyperelasticity
     * @param Q Flow as function of area A: Q = A * u
     * @param terminal_unit_model TerminalUnitData storing geometrical information of terminal node.
     * @param dt Time step size.
     * @return p_el
     */
    double evaluate_ogden_hyperelastic_pressure(OgdenHyperelasticity& ogden_hyperelastic_model,
        const double Q, const RheologicalElasticityModel& terminal_unit_model, double dt);

    /**
     * Evaluate Ogden Hyperelastic pressure gradient, derivative of p_el w.r.t. A.
     * @param ogden_hyperelastic_model Chosen model: OgdenHyperelasticity
     * @param data TerminalUnitData storing geometrical information of terminal node.
     * @param Q
     * @param dQdA
     * @param dt
     * @return derivative of elastic pressure w.r.t A
     */
    double evaluate_ogden_hyperelastic_pressure_gradient(
        OgdenHyperelasticity& ogden_hyperelastic_model,
        const RheologicalElasticityModel& terminal_unit_model, double Q, double dQdA, double dt);

    /**
     * Evaluate residual when RheologyModel is Kelvin Voigt.
     * @param area_A Updated area in Newton iterations.
     * @param velocity_u Velocity, computed from updated A.
     * @param reference_area_A0 Reference area of terminal 1D node.
     * @param beta Wall stiffness parameter in 1D.
     * @param Pext External pressure at terminal 1D node.
     * @param kelvin_voigt_model Chosen model: KelvinVoigt
     * @param elastic_pressure_p_el Elastic pressure computed depending on chosen ElasticityModel
     * @param data TerminalUnitData storing geometrical information of terminal node.
     * @return
     */
    double evaluate_kelvin_voigt_residual(double area_A, double velocity_u,
        double reference_area_A0, double beta, double Pext, const KelvinVoigt& kelvin_voigt_model,
        double elastic_pressure_p_el, const RheologicalElasticityModel& terminal_unit_model);

    /**
     * Evaluate residual when RheologyModel is Four Element Maxwell.
     * @param area_A Updated area in Newton iterations.
     * @param velocity_u Velocity, computed from updated A.
     * @param reference_area_A0 Reference area of terminal 1D node.
     * @param beta Wall stiffness parameter in 1D.
     * @param Pext External pressure at terminal 1D node.
     * @param four_element_maxwell_model Chosen model: FourElementMaxwell
     * @param elastic_pressure_p_el Elastic pressure computed depending on chosen ElasticityModel
     * @param dt Time step size.
     * @param data TerminalUnitData storing geometrical information of terminal node.
     * @return
     */
    double evaluate_four_element_maxwell_residual(double area_A, double velocity_u,
        double reference_area_A0, double beta, double Pext,
        const FourElementMaxwell& four_element_maxwell_model, double elastic_pressure_p_el,
        double dt, const RheologicalElasticityModel& terminal_unit_model);

    /**
     * Calculates jacobian df/dA for KelvinVoigt model.
     * @param area_A Area calculated from last Newton step.
     * @param beta Wall stiffness parameter in 1D.
     * @param kelvin_voigt_model Model containing information of material behavior.
     * @param dp_el_dA Derivative of elastic pressure w.r.t Newton-Raphson variable A.
     * @param data Struct containing information of terminal unit.
     * @return jacobian
     */
    double evaluate_kelvin_voigt_jacobian(double area_A, double dQdA, double beta,
        const KelvinVoigt& kelvin_voigt_model, double dp_el_dA,
        const RheologicalElasticityModel& terminal_unit_model);

    /**
     * Calculates jacobian df/dA for FourElementMaxwell model.
     * @param area_A Area calculated from last Newton step.
     * @param dQdA Derivative of flow Q(A) w.r.t. area A.
     * @param beta Wall stiffness parameter in 1D.
     * @param four_element_maxwell_model Model containing information of material behavior.
     * @param dp_el_dA Derivative of elastic pressure w.r.t Newton-Raphson variable A.
     * @param dt Time step size.
     * @param data TerminalUnitData storing geometrical information of terminal node.
     * @return jacobian
     */
    double evaluate_four_element_maxwell_jacobian(double area_A, double dQdA, double beta,
        const FourElementMaxwell& four_element_maxwell_model, double dp_el_dA, double dt,
        const RheologicalElasticityModel& terminal_unit_model);

    /**
     * Initializes the corresponding rheological model.
     * @param rheology_model Struct RheologicalModel from input file
     * @param global_id global node ID of terminal unit node
     * @return RheologicalModel for terminal unit
     */
    RheologicalModel create_rheological_model(
        const ReducedLung1dPipeFlow::Parameters::TerminalUnits::RheologicalModel& rheology_model,
        int global_id);

    /**
     * Initializes the corresponding elasticity model.
     * @param elasticity_model Struct ElasticityModel from input file
     * @param global_id global node ID of terminal unit node
     * @return ElasticityModel for terminal unit
     */
    ElasticityModel create_elasticity_model(
        const ReducedLung1dPipeFlow::Parameters::TerminalUnits::ElasticityModel& elasticity_model,
        int global_id);

    WindkesselModel create_windkessel_model(
        const ReducedLung1dPipeFlow::Parameters::TerminalUnits::WindkesselModel& windkessel_model,
        int global_id);

    double evaluate_windkessel_residual(double area_A, double velocity_u, double reference_area_A0,
        double beta, double Pext, const WindkesselModel& wk_model);

    double evaluate_windkessel_jacobian(
        double area_A, double dQdA, double beta, const WindkesselModel& wk_model);

  }  // namespace TerminalUnit
}  // namespace ReducedLung1DPipe
FOUR_C_NAMESPACE_CLOSE
#endif  // FOUR_C_REDUCED_LUNG_1D_PIPE_FLOW_TERMINAL_UNIT_HPP

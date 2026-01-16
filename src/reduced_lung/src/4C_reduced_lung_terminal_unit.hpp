// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later


#ifndef FOUR_C_REDUCED_LUNG_TERMINAL_UNIT_HPP
#define FOUR_C_REDUCED_LUNG_TERMINAL_UNIT_HPP

#include "4C_config.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_io_input_field.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_mat_maxwell_0d_acinus.hpp"
#include "4C_reduced_lung_input.hpp"

#include <Teuchos_ParameterList.hpp>

#include <variant>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace ReducedLung
{
  namespace TerminalUnits
  {
    /**
     * @brief Shared data container for all terminal unit elements.
     *
     * Stores identifiers and physical parameters (volume, pressure, etc.) for each terminal unit,
     * which are used across all rheological and elasticity models.
     *
     * @note This data is independent of specific models and is shared across evaluators.
     */
    struct TerminalUnitData
    {
      // Global element IDs.
      std::vector<int> global_element_id;
      // Local element IDs in row map of the old discretization. May become obsolete with new input.
      std::vector<int> local_element_id;
      // IDs of the local rows in the row map. This defines the place for these terminal units in
      // the system of equations.
      std::vector<int> local_row_id;
      // Global dof IDs
      std::vector<int> gid_p1;
      std::vector<int> gid_p2;
      std::vector<int> gid_q;
      // Dof IDs from locally relevant / column map. Used for lookup in dof-vector during assembly.
      std::vector<int> lid_p1;
      std::vector<int> lid_p2;
      std::vector<int> lid_q;
      // Physical quantities of each terminal unit.
      std::vector<double> volume_v;
      std::vector<double> reference_volume_v0;

      /**
       * Access this function when looping over elements.
       */
      [[nodiscard]] size_t number_of_elements() const { return global_element_id.size(); }
    };

    // ----- Rheological models of viscoelasticity -----

    /**
     * @brief Rheological Kelvin-Voigt model (spring and dashpot in parallel).
     *
     * @note The stress component from the spring is calculated with a given elasticity model
     * (potentially nonlinear).
     */
    struct KelvinVoigt
    {
      std::vector<double> viscosity_eta;
    };

    /**
     * @brief Rheological four-element Maxwell model (Kelvin-Voigt body and Maxwell body in
     * parallel).
     *
     * The Maxwell body introduces an additional internal pressure state p_m, which is stored in
     * this model and updated after every time step.
     *
     * @note The stress component from the spring in the Kelvin-Voigt body is calculated with a
     * given elasticity model (potentially nonlinear).
     */
    struct FourElementMaxwell
    {
      std::vector<double> viscosity_eta;
      std::vector<double> elasticity_E_m;
      std::vector<double> viscosity_eta_m;
      // Internal pressure state of the maxwell body
      std::vector<double> maxwell_pressure_p_m;
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
      std::vector<double> elasticity_E;
      // Internal elastic pressure state
      std::vector<double> elastic_pressure_p_el;
      std::vector<double> elastic_pressure_grad_dp_el;
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
      std::vector<double> bulk_modulus_kappa;
      std::vector<double> nonlinear_stiffening_beta;
      // Internal elastic pressure state
      std::vector<double> elastic_pressure_p_el;
      std::vector<double> elastic_pressure_grad_dp_el;
    };

    // Function handle for evaluating the negative model residuals.
    using NegativeResidualEvaluator = std::function<void(TerminalUnitData& model_data,
        Core::LinAlg::Vector<double>& target_vector,
        const Core::LinAlg::Vector<double>& locally_relevant_dof_vector, double time_step_size_dt)>;

    // Function handle for evaluating the model gradients w.r.t. the primary variables.
    using JacobianEvaluator = std::function<void(TerminalUnitData& model_data,
        Core::LinAlg::SparseMatrix& target_matrix,
        const Core::LinAlg::Vector<double>& locally_relevant_dof_vector, double time_step_size_dt)>;

    // Function handle for updating the internal state vectors
    using InternalStateUpdater = std::function<void(TerminalUnitData& model_data,
        const Core::LinAlg::Vector<double>& locally_relevant_dof_vector, double time_step_size_dt)>;

    // Function handle to run at the end of time step. Stores history variables
    using EndOfTimestepRoutine = std::function<void(TerminalUnitData& model_data,
        const Core::LinAlg::Vector<double>& locally_relevant_dof_vector, double time_step_size_dt)>;

    // Model types
    using RheologicalModel = std::variant<KelvinVoigt, FourElementMaxwell>;
    using ElasticityModel = std::variant<LinearElasticity, OgdenHyperelasticity>;

    /**
     * @brief Encapsulates a unique combination of rheological and elasticity models.
     *
     * Stores the element-specific data and associated evaluation functions
     * (residuals, Jacobians, internal state updates, end-of-timestep routines) for a given model
     * combination. Every terminal-unit specific information or action can be found here.
     */
    struct TerminalUnitModel
    {
      TerminalUnitData data;
      RheologicalModel rheological_model;
      ElasticityModel elasticity_model;
      NegativeResidualEvaluator negative_residual_evaluator;
      JacobianEvaluator jacobian_evaluator;
      InternalStateUpdater internal_state_updater;
      EndOfTimestepRoutine end_of_timestep_routine;
    };

    /**
     * @brief All terminal unit models together and interface for access.
     *
     * Stores all different terminal unit models as distinct building blocks. Acts as interface for
     * distributing terminal units to the correct models in the input phase and allows access to all
     * terminal unit model blocks for assembly, output, etc..
     */
    struct TerminalUnitContainer
    {
      std::vector<TerminalUnitModel> models;
    };

    /**
     * @brief Evaluates the linear elastic pressure for all terminal units in one model.
     *
     * @param[in,out] linear_elastic_model Elastic model containing parameters and internal pressure
     * state vector. Must belong to the same TerminalUnitModel as data.
     * @param[in] data Terminal unit metadata (IDs, volume, etc.).
     * @param[in] locally_relevant_dofs DOF vector containing current inflow and pressure values in
     * column map layout.
     * @param[in] dt Current time step size.
     * @return Reference to internal elastic pressure vector, updated in-place with current values.
     *
     * @note This function updates and returns the internal vector `elastic_pressure_p_el` stored
     * in the linear_elastic_model. It must not be used concurrently across models.
     */
    std::vector<double>& evaluate_linear_elastic_pressure(LinearElasticity& linear_elastic_model,
        TerminalUnitData& data, const Core::LinAlg::Vector<double>& locally_relevant_dofs,
        double dt);

    /**
     * @brief Evaluates the nonlinear Ogden hyperelastic pressure for each terminal unit in one
     * model.
     *
     * @param[in,out] ogden_hyperelastic_model Model parameters and internal pressure state vector.
     * Must belong to the same TerminalUnitModel as data.
     * @param[in] data Metadata and volume of each terminal unit.
     * @param[in] locally_relevant_dofs DOF vector containing current inflow and pressure values in
     * column map layout.
     * @param[in] dt Time step size.
     * @return Reference to internal elastic pressure vector, updated in-place with current values.
     *
     * @note This function updates and returns the internal vector `elastic_pressure_p_el` stored
     * in the ogden_hyperelastic_model. It must not be used concurrently across models.
     */
    std::vector<double>& evaluate_ogden_hyperelastic_pressure(
        OgdenHyperelasticity& ogden_hyperelastic_model, TerminalUnitData& data,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt);

    /**
     * @brief Assembles the negative residual for a Kelvin-Voigt viscoelastic model.
     *
     * @param[out] target Target vector to which the residual is written.
     * @param[in] kelvin_voigt_model Model containing viscosity parameters. Must belong to the same
     * TerminalUnitModel as data.
     * @param[in] data Terminal unit metadata.
     * @param[in] locally_relevant_dofs DOF vector containing current inflow and pressure values in
     * column map layout.
     * @param[in] elastic_pressure_p_el Current elastic pressure values from the spring model.
     *
     * @note The function writes directly into the `target` vector using the row IDs from `data`.
     */
    void evaluate_negative_kelvin_voigt_residual(Core::LinAlg::Vector<double>& target,
        const KelvinVoigt& kelvin_voigt_model, const TerminalUnitData& data,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs,
        const std::vector<double>& elastic_pressure_p_el);

    /**
     * @brief Assembles the negative residual for a four-element Maxwell viscoelastic model.
     *
     * @param[out] target Target vector to which the residual is written.
     * @param[in] four_element_maxwell_model Model parameters and internal Maxwell body pressure
     * state. Must belong to the same TerminalUnitModel as data.
     * @param[in] data Terminal unit metadata.
     * @param[in] locally_relevant_dofs DOF vector containing current inflow and pressure values in
     * column map layout.
     * @param[in] elastic_pressure_p_el Current elastic pressure values from the spring model in the
     * Kelvin-Voigt body.
     * @param[in] dt Current time step size.
     *
     * @note The function writes directly into the `target` vector using the row IDs from `data`. It
     * relies on an up-to-date internal Maxwell pressure state (p_m updated with DOFs from the last
     * time step).
     */
    void evaluate_negative_four_element_maxwell_residual(Core::LinAlg::Vector<double>& target,
        const FourElementMaxwell& four_element_maxwell_model, const TerminalUnitData& data,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs,
        const std::vector<double>& elastic_pressure_p_el, double dt);

    /**
     * @brief Computes the gradient of the linear elastic pressure  model w.r.t. the inflow q.
     *
     * The elastic pressure is independent of the pressure state (p1 and p2).
     *
     * @param[in,out] linear_elastic_model Model with stiffness parameters and gradient vector. Must
     * belong to the same TerminalUnitModel as data.
     * @param[in] data Terminal unit metadata.
     * @param[in] dt Current time step size.
     * @return Reference to the internal gradient vector dp_el, updated in-place.
     */
    std::vector<double>& linear_elastic_pressure_gradient(
        LinearElasticity& linear_elastic_model, TerminalUnitData& data, double dt);

    /**
     * @brief Computes the gradient of the nonlinear Ogden hyperelastic pressure w.r.t. inflow q.
     *
     * The elastic pressure is independent of the pressure state (p1 and p2). The elastic pressure
     * gradient depends on the current volume state and inflow.
     *
     * @param[in,out] ogden_hyperelastic_model Ogden parameters and gradient vector. Must belong
     * to the same TerminalUnitModel as data.
     * @param[in] data Terminal unit metadata.
     * @param[in] locally_relevant_dofs DOF vector containing current inflow and pressure values in
     * column map layout.
     * @param[in] dt Current time step size.
     * @return Reference to the internal gradient vector dp_el, updated in-place.
     */
    std::vector<double>& ogden_hyperelastic_pressure_gradient(
        OgdenHyperelasticity& ogden_hyperelastic_model, TerminalUnitData& data,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt);

    /**
     * @brief Assembles the Jacobian matrix entries for the Kelvin-Voigt viscoelastic model.
     *
     * @param[out] target Sparse matrix to which the Jacobian entries are written.
     * @param[in] kelvin_voigt_model Rheological model with viscosity parameters. Must belong
     * to the same TerminalUnitModel as data.
     * @param[in] data Terminal unit metadata.
     * @param[in] elastic_pressure_grad_dp_el Gradient of the elastic pressure from the spring model
     * w.r.t. inflow.
     *
     * @note The function writes directly into the `target` matrix using the row IDs and column
     * (DOF) IDs from `data`.
     */
    void evaluate_kelvin_voigt_jacobian(Core::LinAlg::SparseMatrix& target,
        const KelvinVoigt& kelvin_voigt_model, TerminalUnitData& data,
        const std::vector<double>& elastic_pressure_grad_dp_el);

    /**
     * @brief Assembles the Jacobian matrix for the four-element Maxwell viscoelastic model.
     *
     * @param[out] target Sparse matrix to which the Jacobian entries are written.
     * @param[in] four_element_maxwell_model Four-element Maxwell parameters and internal Maxwell
     * pressure state. Must belong to the same TerminalUnitModel as data.
     * @param[in] data Terminal unit metadata.
     * @param[in] elastic_pressure_grad_dp_el Gradient of the elastic pressure from the spring model
     * w.r.t. inflow.
     * @param[in] dt Current time step size.
     *
     * @note The function writes directly into the `target` matrix using the row IDs and column
     * (DOF) IDs from `data`.
     */
    void evaluate_four_element_maxwell_jacobian(Core::LinAlg::SparseMatrix& target,
        const FourElementMaxwell& four_element_maxwell_model, TerminalUnitData& data,
        const std::vector<double>& elastic_pressure_grad_dp_el, double dt);


    /**
     * Creates the model-specific evaluator function for assembling the negative model residual.
     */
    struct MakeNegativeResidualEvaluator
    {
      /**
       * Creates a function for evaluating the elastic pressure given a specific elasticity model.
       * The elastic pressure is one additive component of the rheological model's residual
       * function.
       */
      struct MakeElasticPressureEvaluator
      {
        using ElasticPressureEvaluator = std::function<std::vector<double>&(
            TerminalUnitData&, const Core::LinAlg::Vector<double>&, double)>;

        ElasticPressureEvaluator operator()(LinearElasticity& linear_elasticity_model)
        {
          return std::bind_front(
              evaluate_linear_elastic_pressure, std::ref(linear_elasticity_model));
        }
        ElasticPressureEvaluator operator()(OgdenHyperelasticity& ogden_model)
        {
          return std::bind_front(evaluate_ogden_hyperelastic_pressure, std::ref(ogden_model));
        }
      };

      NegativeResidualEvaluator operator()(KelvinVoigt& kelvin_voigt_model)
      {
        auto elastic_pressure_evaluator =
            std::visit(MakeElasticPressureEvaluator{}, elasticity_model);
        return [elastic_pressure_evaluator, &kelvin_voigt_model](TerminalUnitData& data,
                   Core::LinAlg::Vector<double>& target, const Core::LinAlg::Vector<double>& dofs,
                   double dt)
        {
          auto& p_el = elastic_pressure_evaluator(data, dofs, dt);
          evaluate_negative_kelvin_voigt_residual(target, kelvin_voigt_model, data, dofs, p_el);
        };
      }
      NegativeResidualEvaluator operator()(FourElementMaxwell& four_element_maxwell_model)
      {
        auto elastic_pressure_evaluator =
            std::visit(MakeElasticPressureEvaluator{}, elasticity_model);
        return [elastic_pressure_evaluator, &four_element_maxwell_model](TerminalUnitData& data,
                   Core::LinAlg::Vector<double>& target, const Core::LinAlg::Vector<double>& dofs,
                   double dt)
        {
          auto& p_el = elastic_pressure_evaluator(data, dofs, dt);
          evaluate_negative_four_element_maxwell_residual(
              target, four_element_maxwell_model, data, dofs, p_el, dt);
        };
      }

      ElasticityModel& elasticity_model;
    };

    /**
     * Creates the model-specific evaluator function for assembling the Jacobian.
     */
    struct MakeJacobianEvaluator
    {
      /**
       * Creates a function for evaluating the elastic pressure gradient w.r.t. inflow q given a
       * specific elasticity model. The elastic pressure gradient is one additive component of the
       * rheological model's derivative w.r.t. inflow q.
       */
      struct MakeElasticPressureGradientEvaluator
      {
        using ElasticPressureGradientEvaluator = std::function<std::vector<double>&(
            TerminalUnitData&, const Core::LinAlg::Vector<double>&, double)>;

        ElasticPressureGradientEvaluator operator()(LinearElasticity& linear_elasticity_model)
        {
          return [&linear_elasticity_model](TerminalUnitData& data,
                     const Core::LinAlg::Vector<double>& dofs, double dt) -> std::vector<double>&
          { return linear_elastic_pressure_gradient(linear_elasticity_model, data, dt); };
        }
        ElasticPressureGradientEvaluator operator()(OgdenHyperelasticity& ogden_model)
        {
          return std::bind_front(ogden_hyperelastic_pressure_gradient, std::ref(ogden_model));
        }
      };

      JacobianEvaluator operator()(KelvinVoigt& kelvin_voigt_model)
      {
        auto elastic_pressure_gradient_evaluator =
            std::visit(MakeElasticPressureGradientEvaluator{}, elasticity_model);
        return [elastic_pressure_gradient_evaluator, &kelvin_voigt_model](TerminalUnitData& data,
                   Core::LinAlg::SparseMatrix& target, const Core::LinAlg::Vector<double>& dofs,
                   double dt)
        {
          auto& dp_el = elastic_pressure_gradient_evaluator(data, dofs, dt);
          evaluate_kelvin_voigt_jacobian(target, kelvin_voigt_model, data, dp_el);
        };
      }
      JacobianEvaluator operator()(FourElementMaxwell& four_element_maxwell_model)
      {
        auto elastic_pressure_gradient_evaluator =
            std::visit(MakeElasticPressureGradientEvaluator{}, elasticity_model);
        return [elastic_pressure_gradient_evaluator, &four_element_maxwell_model](
                   TerminalUnitData& data, Core::LinAlg::SparseMatrix& target,
                   const Core::LinAlg::Vector<double>& dofs, double dt)
        {
          auto& dp_el = elastic_pressure_gradient_evaluator(data, dofs, dt);
          evaluate_four_element_maxwell_jacobian(
              target, four_element_maxwell_model, data, dp_el, dt);
        };
      }

      ElasticityModel& elasticity_model;
    };

    /**
     * Creates the model-specific routine that updates internal state variables during the nonlinear
     * loop
     */
    struct MakeInternalStateUpdater
    {
      InternalStateUpdater operator()(KelvinVoigt& kelvin_voigt_model)
      {
        return [&](TerminalUnitData& data,
                   const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt) {};
      }
      InternalStateUpdater operator()(FourElementMaxwell& four_element_maxwell)
      {
        return [&](TerminalUnitData& data,
                   const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt) {};
      }
    };

    /**
     * Creates the model-specific routine at the end of the timestep that updates history variables
     * that need to be tracked over time for model evaluation
     */
    struct MakeEndOfTimestepRoutine
    {
      EndOfTimestepRoutine operator()(KelvinVoigt& kelvin_voigt_model)
      {
        return [&](TerminalUnitData& data,
                   const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt) {};
      }
      EndOfTimestepRoutine operator()(FourElementMaxwell& four_element_maxwell)
      {
        return [&](TerminalUnitData& data,
                   const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt)
        {
          for (size_t i = 0; i < data.number_of_elements(); i++)
          {
            four_element_maxwell.maxwell_pressure_p_m[i] *=
                four_element_maxwell.viscosity_eta_m[i] /
                (four_element_maxwell.elasticity_E_m[i] * dt +
                    four_element_maxwell.viscosity_eta_m[i]);
            four_element_maxwell.maxwell_pressure_p_m[i] +=
                four_element_maxwell.elasticity_E_m[i] * dt *
                four_element_maxwell.viscosity_eta_m[i] /
                (four_element_maxwell.elasticity_E_m[i] * dt +
                    four_element_maxwell.viscosity_eta_m[i]) /
                data.reference_volume_v0[i] *
                locally_relevant_dofs.local_values_as_span()[data.lid_q[i]];
          }
        };
      }
    };

    /**
     * @brief Returns an existing or newly created model for a given rheological/elasticity pair.
     *
     * May become obsolete with the new input
     *
     * @tparam R Rheological model type.
     * @tparam E Elasticity model type.
     * @param terminal_units Global container for all terminal unit models.
     * @return Reference to the matching TerminalUnitModel.
     *
     * @note Ensures reuse of model instances for shared material parameter sets.
     */
    template <typename R, typename E>
    TerminalUnitModel& register_or_access_terminal_unit_model(TerminalUnitContainer& terminal_units)
    {
      for (auto& model : terminal_units.models)
      {
        if (std::holds_alternative<R>(model.rheological_model) &&
            std::holds_alternative<E>(model.elasticity_model))
        {
          return model;
        }
      }

      // Create instance of the TerminalUnitModel
      terminal_units.models.emplace_back();
      TerminalUnitModel& new_model = terminal_units.models.back();

      // Create models
      new_model.rheological_model = R{};
      new_model.elasticity_model = E{};

      // Evaluators are assigned later to avoid capturing references into `terminal_units.models`
      // while the vector may still reallocate during element registration.

      return new_model;
    }

    struct AddRheologicalModelParameter
    {
      void operator()(KelvinVoigt& kelvin_voigt_model) const
      {
        kelvin_voigt_model.viscosity_eta.push_back(
            params.kelvin_voigt.viscosity_kelvin_voigt_eta.at(
                ele->id(), "viscosity_kelvin_voigt_eta"));
      }
      void operator()(FourElementMaxwell& model) const
      {
        model.elasticity_E_m.push_back(params.four_element_maxwell.elasticity_maxwell_e_m.at(
            ele->id(), "elasticity_maxwell_e_m"));
        model.viscosity_eta.push_back(params.four_element_maxwell.viscosity_kelvin_voigt_eta.at(
            ele->id(), "viscosity_kelvin_voigt_eta"));
        model.viscosity_eta_m.push_back(params.four_element_maxwell.viscosity_maxwell_eta_m.at(
            ele->id(), "viscosity_maxwell_eta_m"));
        model.maxwell_pressure_p_m.push_back(0.0);
      }

      Core::Elements::Element* ele;
      ReducedLungParameters::LungTree::TerminalUnits::RheologicalModel params;
    };

    struct AddElasticityModelParameter
    {
      void operator()(LinearElasticity& linear_elasticity_model) const
      {
        linear_elasticity_model.elasticity_E.push_back(
            params.linear.elasticity_e.at(ele->id(), "elasticity_e"));
        // Vector initialization for internal pressure state.
        linear_elasticity_model.elastic_pressure_p_el.push_back(0.0);
        linear_elasticity_model.elastic_pressure_grad_dp_el.push_back(0.0);
      }
      void operator()(OgdenHyperelasticity& ogden_model) const {}

      Core::Elements::Element* ele;
      ReducedLungParameters::LungTree::TerminalUnits::ElasticityModel params;
    };

    /**
     * @brief Adds a new element to the appropriate TerminalUnitModel group.
     *
     * Computes volume, assigns material parameters, and initializes internal state.
     *
     * @tparam R Rheological model.
     * @tparam E Elasticity model.
     * @param terminal_units Global container for all terminal unit models.
     * @param ele Pointer to element to be added.
     * @param local_element_id Local identifier in 4C discretization.
     */
    template <typename R, typename E>
    void add_terminal_unit_ele(TerminalUnitContainer& terminal_units, Core::Elements::Element* ele,
        int local_element_id, ReducedLungParameters::LungTree::TerminalUnits tu_params)
    {
      TerminalUnitModel& model = register_or_access_terminal_unit_model<R, E>(terminal_units);

      model.data.global_element_id.push_back(ele->id());
      model.data.local_element_id.push_back(local_element_id);
      const auto& coords_node_1 = ele->nodes()[0]->x();
      const auto& coords_node_2 = ele->nodes()[1]->x();
      const double radius =
          std::sqrt((coords_node_1[0] - coords_node_2[0]) * (coords_node_1[0] - coords_node_2[0]) +
                    (coords_node_1[1] - coords_node_2[1]) * (coords_node_1[1] - coords_node_2[1]) +
                    (coords_node_1[2] - coords_node_2[2]) * (coords_node_1[2] - coords_node_2[2]));
      const double volume = (4.0 / 3.0) * std::numbers::pi * radius * radius * radius;
      model.data.volume_v.push_back(volume);
      model.data.reference_volume_v0.push_back(volume);
      std::visit(AddRheologicalModelParameter{.ele = ele, .params = tu_params.rheological_model},
          model.rheological_model);
      std::visit(AddElasticityModelParameter{.ele = ele, .params = tu_params.elasticity_model},
          model.elasticity_model);
    }

    /**
     * @brief Assembles the negative residual vector of all terminal units.
     *
     * Applies model-specific logic to compute viscoelastic responses of all terminal units and
     * store them in the residual.
     *
     * @param res_vector Residual vector in row map layout
     * @param terminal_units All grouped terminal unit models.
     * @param locally_relevant_dofs DOF vector containing current inflow and pressure values in
     * column map layout.
     * @param dt Current time step size.
     */
    void update_negative_residual_vector(Core::LinAlg::Vector<double>& res_vector,
        TerminalUnitContainer& terminal_units,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt);

    /**
     * @brief Assembles the Jacobian matrix contributions from all terminal units.
     *
     * Derivatives w.r.t. pressure and flow variables are computed using model-specific logic. They
     * are directly stored in the given Jacobian.
     *
     * @param jac Jacobian matrix in (row, column) map layout.
     * @param terminal_units All grouped terminal unit models.
     * @param locally_relevant_dofs DOF vector containing current inflow and pressure values in
     * column map layout.
     * @param dt Current time step size.
     */
    void update_jacobian(Core::LinAlg::SparseMatrix& jac, TerminalUnitContainer& terminal_units,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt);

    /**
     * @brief Creates evaluator functions for all terminal unit models.
     *
     * This function must be called after all terminal unit elements have been added.
     * It initializes the evaluator function objects (residual, Jacobian, internal state updater,
     * end-of-timestep routine) for each model, ensuring that references captured by these
     * evaluators remain valid.
     */
    void create_evaluators(TerminalUnitContainer& terminal_units);

    /**
     * @brief Updates the internal state memory of each terminal unit model.
     *
     * Used to store history variables (e.g. volume, Maxwell pressure) from previous time steps.
     *
     * @param terminal_units All grouped terminal unit models.
     * @param locally_relevant_dofs DOF vector containing inflow and pressure values in
     * column map layout. The DOFs have to be in a converged state.
     * @param dt Time step size.
     *
     * @note Needs to be executed once between time steps to update the internal state with the
     * converged DOFs.
     */
    void update_internal_state_vectors(TerminalUnitContainer& terminal_units,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt);

    /**
     * @brief Routine to be executed at the end of each time step.
     *
     * Used to store history variables from previous time steps.
     *
     * @param terminal_units All grouped terminal unit models.
     * @param locally_relevant_dofs DOF vector containing inflow and pressure values in
     * column map layout. The DOFs have to be in a converged state.
     * @param dt Time step size.
     *
     * @note Needs to be executed once between time steps to update the internal state with the
     * converged DOFs.
     */
    void end_of_timestep_routine(TerminalUnitContainer& terminal_units,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt);
  }  // namespace TerminalUnits
}  // namespace ReducedLung

FOUR_C_NAMESPACE_CLOSE

#endif

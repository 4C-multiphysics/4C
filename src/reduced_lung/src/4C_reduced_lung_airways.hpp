// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later



#ifndef FOUR_C_REDUCED_LUNG_AIRWAYS_HPP
#define FOUR_C_REDUCED_LUNG_AIRWAYS_HPP

#include "4C_config.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_linalg_map.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_reduced_lung_input.hpp"
#include "4C_utils_exceptions.hpp"

#include <cstddef>
#include <cstdlib>
#include <functional>
#include <type_traits>
#include <variant>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace ReducedLung
{
  namespace Airways
  {
    /**
     * @brief Shared data container for all airway elements.
     *
     * Stores identifiers and physical parameters (length, area, etc.) for each
     * airway, which are used across all flow and wall models.
     *
     * @note This data is independent of specific models and is shared across evaluators.
     */
    struct AirwayData
    {
      // Global element IDs.
      std::vector<int> global_element_id;
      // Local element IDs.
      std::vector<int> local_element_id;
      // IDs of the local rows in the row map. This defines the place for these airways in the
      // system of equations.
      std::vector<int> local_row_id;
      // Global dof IDs
      std::vector<int> gid_p1, gid_p2, gid_q1, gid_q2;
      // Dof IDs from locally relevant / column map. Used for lookup in dof-vector during assembly.
      std::vector<int> lid_p1, lid_p2, lid_q1, lid_q2;

      // Physical quantities of each airway.
      std::vector<double> ref_length;
      std::vector<double> ref_area;
      struct AirProperties
      {
        double dynamic_viscosity;
        double density;
      } air_properties;

      // Dof values at previous time step
      std::vector<double> q1_n;
      std::vector<double> q2_n;
      std::vector<double> p1_n;
      std::vector<double> p2_n;

      int n_state_equations;  // number of state equations per airway
      // convenience
      [[nodiscard]] size_t number_of_elements() const { return global_element_id.size(); }
    };

    // ----- Flow resistance models -----

    /**
     * @brief Linear resistive model for airway flow.
     *
     * Considers Poiseuille resistance \f$R_p= 8 \pi \mu L / A^2\f$ and optional inertial effects.
     */
    struct LinearResistive
    {
      std::vector<bool> has_inertia;
    };

    /**
     * @brief Nonlinear resistive model considering turbulence effects and convective resistance
     * from momentum flux.
     *
     * Airway resistance is defined as \f$R = R_p \, k_{turb} + \frac{2 \alpha
     * \rho}{A^2}(Q_2-Q_1)\f$, where \f$_p\f$ is the Poiseuille resistance and \f$k_{turb}\f$ is a
     * turbulence correction factor depending on the Reynolds number. The second term represents
     * convective resistance due to momentum flux, with \f$\alpha\f$ being a correction factor
     * \f$\alpha = \frac{4 k_{turb}}{4 k_{turb}-1}\f$.
     */
    struct NonLinearResistive
    {
      std::vector<double> turbulence_factor_gamma;
      std::vector<bool> has_inertia;

      // internal state variables
      std::vector<double> k_turb;
    };

    // ----- Wall models -----

    /**
     * @brief Rigid wall model for airways.
     */
    struct RigidWall
    {
      // empty for now
    };

    /**
     * @brief Kelvin-Voigt wall model for airways.
     *
     * Models airway wall mechanics of a Kelvin-Voigt type wall by the equation \f$ \frac{\dot{P}_1
     * -
     * \dot{P}_2}{2} - \dot{P}_{ext} = R_{visc} \cdot (Q_1 - Q_2) + \frac{1}{C} \cdot \frac{(Q_1 -
     * Q_2)}{dt} \f$, where \f$R_{visc}\f$ is the viscous resistance and \f$C\f$ the compliance of
     * the airway wall.
     */
    struct KelvinVoigtWall
    {
      std::vector<double> wall_poisson_ratio;
      std::vector<double> wall_elasticity;
      std::vector<double> wall_thickness;
      std::vector<double> viscous_time_constant;
      std::vector<double> viscous_phase_shift;
      std::vector<double> area_n;

      // internal state variables
      std::vector<double> area;
      std::vector<double> viscous_resistance_Rvisc;
      std::vector<double> compliance_C;
      std::vector<double> gamma_w;
      std::vector<double> beta_w;
    };

    // ----- Evaluator function types -----

    // Function handle for evaluating the negative model residuals.
    using NegativeResidualEvaluator =
        std::function<void(const AirwayData& data, Core::LinAlg::Vector<double>& target_vector,
            const Core::LinAlg::Vector<double>& locally_relevant_dofs, double time_step_size_dt)>;

    // Function handle for evaluating Jacobian (the model gradients w.r.t the primary variables).
    using JacobianEvaluatorAW =
        std::function<void(const AirwayData& data, Core::LinAlg::SparseMatrix& target_mat,
            const Core::LinAlg::Vector<double>& locally_relevant_dofs, double time_step_size_dt)>;

    // Function handle to update internal state vector.
    using InternalStateUpdaterAW = std::function<void(AirwayData& data,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs, double time_step_size_dt)>;

    // Function handle to run at the end of time step. Stores history variables
    using EndOfTimestepRoutine = std::function<void(AirwayData& data,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs, double time_step_size_dt)>;

    // Variant type to hold different flow and wall models.
    using FlowModel = std::variant<LinearResistive, NonLinearResistive>;
    using WallModel = std::variant<RigidWall, KelvinVoigtWall>;

    /**
     * @brief Holds all elements with a unique combination of flow and wall model.
     *
     * Stores the element-specific data and associated evaluation functions
     * (residuals, Jacobians, internal state updates) for a given model combination. Every
     * airway-specific information or action can be found here.
     */
    struct AirwayModel
    {
      AirwayData data;
      FlowModel flow_model;
      WallModel wall_model;
      NegativeResidualEvaluator negative_residual_evaluator;
      JacobianEvaluatorAW jacobian_evaluator;
      InternalStateUpdaterAW internal_state_updater;
      EndOfTimestepRoutine end_of_timestep_routine;
    };

    /**
     * @brief All airway models together and interface for access.
     *
     * Stores all different airway models as distinct building blocks. Acts as interface for
     * distributing airways to the correct models in the input phase and allows access to all
     * airway model blocks for assembly, output, etc..
     */
    struct AirwayContainer
    {
      std::vector<AirwayModel> models;
    };

    // compile-time traits for number of state equations contributed by each model type
    template <typename F>
    struct FlowModelStateCount
    {
    };
    template <>
    struct FlowModelStateCount<LinearResistive>
    {
      static constexpr int value = 1;
    };
    template <>
    struct FlowModelStateCount<NonLinearResistive>
    {
      static constexpr int value = 1;
    };

    template <typename W>
    struct WallModelStateCount
    {
    };
    template <>
    struct WallModelStateCount<RigidWall>
    {
      static constexpr int value = 0;
    };
    template <>
    struct WallModelStateCount<KelvinVoigtWall>
    {
      static constexpr int value = 1;
    };

    /**
     * @brief Calculates the Poiseuille resistance for airway elements.
     *
     *
     * @param data The airway data containing air properties and element reference lengths
     * @param area Vector of cross-sectional areas for each airway element
     *
     * @return std::vector<double> Vector of Poiseuille resistance values for each element,
     *         computed as: \f$ R_p = \frac{8 \pi \mu L}{A^2} \f$
     */
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

    /**
    * @brief Assembles the negative residual vector of airways with rigid walls.
    * Each airway is described by one state equation (momentum conservation).
    *
    * \f$ -f_1 = -(P_1 - P_2 - R\,Q_1 - \frac{I}{\Delta t} \, (Q_1 - Q_1^n)) = -(P_1 - P_2  +
    f_{1,QR}
    * + f_{1,QI})\f$ \n
    * with \f$ f_{1,QR} = - R\,Q_1 \f$ and \f$ f_{1,QI} = - \frac{I}{\Delta t} \, (Q_1 - Q_1^n) \f$

    * @param target The target RHS vector to store the negative residuals.
    * @param data The airway data containing element information.
    * @param locally_relevant_dofs The DOF vector containing current flows and pressures in column
    map layout.
    * @param resistance The flow resistance \f$ R \f$ for each airway element.
    * @param inertia The inertia \f$ I \f$ for each airway element.
    * @param dt The time step size.
    */
    void evaluate_negative_rigid_wall_residual(Core::LinAlg::Vector<double>& target,
        const AirwayData& data, const Core::LinAlg::Vector<double>& locally_relevant_dofs,
        const std::vector<double>& resistance, const std::vector<double>& inertia, double dt);

    /**
     * @brief Assembles the negative residual vector of airways with Kelvin-Voigt walls.
     * Each airway is described by two state equations (momentum and mass conservation).
     *
     * \f$ -f_1 = -(P_1 - P_2 - ( \frac{R}{2} + \frac{I}{2 \Delta t}) (Q_1 + Q_2) + \frac{I}{2
     * \Delta t} (Q_1^n + Q_2^n)) = -(P_1 -P_2 + f_{1,QR} + f_{1,QI}) \f$
     *
     * \f$ -f_2 = -(P_1 + P_2 - P_1^n - P_2^n - 2 (R_{visc} + \frac{\Delta t}{C}) (Q_1 - Q_2) + 2
     * R_{visc} (Q_1^n - Q_2^n)) = -(P_1 + P_2 - P_1^n - P_2^n + f_{2,QR_{visc}})\f$
     *
     * with \f$ f_{1,QR} = - \frac{R}{2} (Q_1 + Q_2) \f$, \f$ f_{1,QI} = - \frac{I}{2 \Delta t} (Q_1
     * + Q_2 - Q_1^n - Q_2^n) \f$
     * and \f$ f_{2,QR_{visc}} = - 2 (R_{visc} + \frac{\Delta t}{C}) (Q_1 - Q_2) + 2 R_{visc} (Q_1^n
     * - Q_2^n) \f$
     *
     * @param target The target RHS vector to store the negative residuals.
     * @param kelvin_voigt_wall_model The Kelvin-Voigt wall model containing wall parameters and
     * internal states.
     * @param data The airway data containing element information.
     * @param locally_relevant_dofs The DOF vector containing current flows and pressures in column
     * map layout.
     * @param resistance The flow resistance \f$ R \f$ for each airway element.
     * @param inertia The inertia \f$ I \f$ for each airway element.
     * @param dt The time step size.
     */
    void evaluate_negative_kelvin_voigt_wall_residual(Core::LinAlg::Vector<double>& target,
        const KelvinVoigtWall& kelvin_voigt_wall_model, const AirwayData& data,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs,
        const std::vector<double>& resistance, const std::vector<double>& inertia, double dt);

    /**
     * @brief Assembles the Jacobian matrix of airways with rigid walls.
     *
     * @param target The target sparse matrix to store the Jacobian entries.
     * @param data The airway data containing element information.
     * @param resistance_derivative The derivative of the flow resistance term w.r.t. \f$ Q_1 \f$
     * for each airway element.
     * @param inertia_derivative The derivative of the inertia term w.r.t. \f$ Q_1 \f$ for each
     * airway element.
     * @param dt The time step size.
     */
    void evaluate_jacobian_rigid_wall(Core::LinAlg::SparseMatrix& target, AirwayData const& data,
        const std::vector<double>& resistance_derivative,
        const std::vector<double>& inertia_derivative, double dt);

    /**
     * @brief Calculates the derivative of the nonlinear flow term for rigid walls.
     *
     * @param model The nonlinear resistive flow model containing turbulence parameters.
     * @param data The airway data containing element information.
     * @param locally_relevant_dofs The DOF vector containing current flows and pressures in column
     * map layout.
     * @param dt The time step size.
     * @return std::vector<double> Vector of flow resistance derivatives for each airway element.
     *
     * The derivative is given by:
     * \f$ \frac{\partial f_{1,QR}}{\partial Q_1} = R_p \frac{\partial k_{turb}}{\partial Q_1} \cdot
     * Q_1 + R_p \, k_{turb} \f$
     *
     * with \f$ \frac{\partial k_{turb}}{\partial Q_1} = \gamma \cdot \sqrt{\frac{\rho}{\pi \mu L}}
     * \, \frac{1}{\sqrt{Q_1}}\f$
     */
    std::vector<double> evaluate_nonlinear_flow_resistance_derivative_rigid(
        const NonLinearResistive& model, const AirwayData& data,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt);

    /**
     * @brief Assembles the Jacobian matrix of airways with Kelvin-Voigt walls.
     *
     * @param target The target sparse matrix to store the Jacobian entries.
     * @param data The airway data containing element information.
     * @param resistance_derivative The derivative of the flow resistance term w.r.t. \f$ Q_1 \f$
     * and
     * \f$ Q_2 \f$ for each airway element.
     * @param inertia_derivative The derivative of the inertia term w.r.t. \f$ Q_1 \f$ and \f$ Q_2
     * \f$ for each airway element.
     * @param viscous_wall_resistance_derivative The derivative of the viscous wall resistance term
     * w.r.t. \f$ Q_1 \f$ and \f$ Q_2 \f$ for each airway element.
     * @param dt The time step size.
     */
    void evaluate_jacobian_kelvin_voigt_wall(Core::LinAlg::SparseMatrix& target,
        AirwayData const& data, const KelvinVoigtWall& kelvin_voigt_wall_model,
        const std::pair<std::vector<double>, std::vector<double>>& resistance_derivative,
        const std::pair<std::vector<double>, std::vector<double>>& inertia_derivative,
        const std::pair<std::vector<double>, std::vector<double>>&
            viscous_wall_resistance_derivative,
        double dt);


    /**
     * @brief Derivatives of the linear flow term for Kelvin-Voigt walls.
     *
     * @param model The linear resistive flow model.
     * @param data The airway data containing element information.
     * @param locally_relevant_dofs The DOF vector containing current flows and pressures in column
     * map layout.
     * @param area The current cross-sectional areas of the airway elements.
     * @param dt The time step size.
     * @return std::pair<std::vector<double>, std::vector<double>> Pair of vectors containing the
     * flow resistance term derivatives for each airway element w.r.t. \f$ Q_1 \f$ and \f$ Q_2 \f$.
     *
     * The derivatives are given by:
     * \f$ \frac{\partial f_{1,QR}}{\partial Q_1} = -\frac{1}{2}\left(\frac{\partial R_p}{\partial
     * A}\frac{\partial A}{\partial Q_1}(Q_1+Q_2) + R_p\right) \f$,
     *
     * \f$ \frac{\partial f_{1,QR}}{\partial Q_2} = -\frac{1}{2}\left(\frac{\partial R_p}{\partial
     * A}\frac{\partial A}{\partial Q_2}(Q_1+Q_2) + R_p\right) \f$
     *
     * with \f$ \frac{\partial R_p}{\partial A} = -\frac{16 \pi \mu
     * L}{A^3},\; \frac{\partial A}{\partial Q_1} = \frac{\Delta t}{L},\; \frac{\partial A}{\partial
     * Q_2} = -\frac{\Delta t}{L} \f$.
     * @note Unlike in rigid walls, the derivative of the linear flow resistance is not \f$ R_p \f$,
     * because of the change in cross-sectional area.
     */
    std::pair<std::vector<double>, std::vector<double>>
    evaluate_linear_flow_resistance_derivative_kelvin_voigt(const LinearResistive& model,
        const AirwayData& data, const Core::LinAlg::Vector<double>& locally_relevant_dofs,
        std::vector<double>& area, double dt);


    /**
     * @brief Calculates the derivative of the nonlinear flow term for Kelvin Voigt walls.
     *
     * @param data The airway data containing element information.
     * @param locally_relevant_dofs The DOF vector containing current flows and pressures in column
     * map layout.
     * @param area The current cross-sectional areas of the airway elements.
     * @param dt The time step size.
     * @return std::pair<std::vector<double>, std::vector<double>> Pair of vectors containing the
     * flow resistance term derivatives for each airway element w.r.t. \f$ Q_1 \f$ and \f$ Q_2 \f$.
     *
     * The derivatives are given by:
     *
     * \f$ \frac{\partial f_{1,QR}}{\partial Q_1} = -\frac{1}{2}\left[\left(\frac{\partial R_p}
     * {\partial A}\frac{\partial A}{\partial Q_1} k_{turb} + R_p \frac{\partial k_{turb}}
     * {\partial Q_1} + \frac{2\rho (Q_2-Q_1)}{A^2} \frac{\partial \alpha}{\partial k_{turb}}
     * \frac{\partial k_{turb}}{\partial Q_1} - \frac{2\rho \alpha}{A^2} - \frac{4\rho \alpha}
     * {A^3}\frac{\partial A}{\partial Q_1}(Q_2-Q_1)\right)(Q_1+Q_2) + R\right] \f$
     *
     * \f$ \frac{\partial f_{1,QR}}{\partial Q_2} = -\frac{1}{2}\left[\left(\frac{\partial
     * R_p}{\partial A}\frac{\partial A}{\partial Q_2} k_{turb} + R_p \frac{\partial k_{turb}}
     * {\partial Q_2} + \frac{2\rho (Q_2-Q_1)}{A^2} \frac{\partial \alpha}{\partial k_{turb}}
     * \frac{\partial k_{turb}}{\partial Q_2} + \frac{2\rho \alpha}{A^2} - \frac{4\rho \alpha}{A^3}
     * \frac{\partial A}{\partial Q_2}(Q_2-Q_1)\right)(Q_1+Q_2) + R\right] \f$
     *
     * where \f$ R = R_p k_{turb} + \frac{2 \alpha \rho}{A^2}(Q_2-Q_1) \f$,
     *
     * \f$ \frac{\partial R_p}{\partial A} = -\frac{16 \pi \mu L}{A^3}\f$,
     *
     * \f$\frac{\partial A}{\partial Q_1} = \frac{\Delta t}{L} = -\frac{\partial A}{\partial
     * Q_1} \f$,
     *
     * \f$ \frac{\partial k_{turb}}{\partial Q_1} = \gamma \cdot \sqrt{\frac{\rho}{2\pi \mu L}} \,
     * \frac{Q_1+Q_2}{|Q_1+Q_2|^{3/2}} =  \frac{\partial k_{turb}}{\partial Q_2} \f$,
     *
     * \f$ \alpha = \frac{4 k_{turb}}{4 k_{turb}-1} \f$,
     *
     * and \f$ \frac{\partial \alpha}{\partial k_{turb}} = -\frac{4}{(4k_{turb}-1)^2} \f$.
     */
    std::pair<std::vector<double>, std::vector<double>>
    evaluate_nonlinear_flow_resistance_derivative_kelvin_voigt(const NonLinearResistive& model,
        const AirwayData& data, const Core::LinAlg::Vector<double>& locally_relevant_dofs,
        std::vector<double>& area, double dt);

    /**
     * @brief Calculates the derivative of the viscous wall resistance term for Kelvin-Voigt walls.
     *
     * @param kelvin_voigt_wall_model The Kelvin-Voigt wall model containing wall parameters and
     * internal states.
     * @param data The airway data containing element information.
     * @param locally_relevant_dofs The DOF vector containing current flows and pressures in column
     * map layout.
     * @param dt The time step size.
     * @return std::pair<std::vector<double>, std::vector<double>> Pair of vectors containing the
     * viscous wall resistance term derivatives for each airway element w.r.t. \f$ Q_1 \f$ and \f$
     * Q_2
     * \f$.
     *
     * The derivatives are given by:
     *
     * \f$ \frac{\partial f_{2,QR_{visc}}}{\partial Q_1} = -2\left[\left(\frac{\partial
     * R_{visc}}{\partial A}\frac{\partial A}{\partial Q_1} + \Delta t \,\frac{\partial}{\partial
     * A}(\frac{1}{C})\frac{\partial A}{\partial Q_1}\right)(Q_1-Q_2) + \left(R_{visc} +
     * \frac{\Delta t}{C}\right) - \frac{\partial R_{visc}}{\partial A}\frac{\partial A}{\partial
     * Q_1}(Q_1^n-Q_2^n)\right] \f$
     *
     * \f$ \frac{\partial f_{2,QR_{visc}}}{\partial Q_2} = -2\left[\left(\frac{\partial
     * R_{visc}}{\partial A}\frac{\partial A}{\partial Q_2} + \Delta t \,\frac{\partial}{\partial
     * A}(\frac{1}{C})\frac{\partial A}{\partial Q_1}\right)(Q_1-Q_2) - \left(R_{visc} +
     * \frac{\Delta t}{C}\right) - \frac{\partial R_{visc}}{\partial A}\frac{\partial A}{\partial
     * Q_2}(Q_1^n-Q_2^n)\right] \f$
     *
     * where \f$ \frac{\partial R_{visc}}{\partial A} = \frac{\tau_w \tan(\phi_w)}{8\pi\sqrt{A^3}L}
     * \beta_w \f$, \n
     * \f$ \frac{\partial A}{\partial Q_1} = \frac{\Delta t}{L} = -\frac{\partial A}{\partial Q_2}
     * \f$,
     * \n and \f$\frac{\partial}{\partial A}(\frac{1}{C}) = -\frac{\beta_w}{4 L\, \sqrt{A^3}} \f$.
     */
    std::pair<std::vector<double>, std::vector<double>>
    evaluate_viscous_wall_resistance_derivative_kelvin_voigt(
        const KelvinVoigtWall& kelvin_voigt_wall_model, const AirwayData& data,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt);

    /**
     * @brief Calculates the derivative of the inertia term for Kelvin-Voigt walls.
     *
     * Computes the derivatives of the inertia contribution to the residual with respect to the
     * flow rates Q1 and Q2. The inertia term accounts for the mass of air in the airway and its
     * acceleration effects.
     *
     * @param has_inertia Vector of boolean flags indicating whether inertia is included for each
     * element.
     * @param data The airway data containing element information and properties.
     * @param locally_relevant_dofs The DOF vector containing current flows and pressures in column
     * map layout.
     * @param wall The Kelvin-Voigt wall model containing area information.
     * @param dt The time step size.
     *
     * @return std::pair<std::vector<double>, std::vector<double>> Pair of vectors containing the
     * inertia term derivatives for each airway element with respect to Q1 and Q2, respectively.
     * Elements where inertia is disabled (has_inertia[i] = false) will have zero derivatives.
     *
     * The derivatives are computed as:
     *
     * \f$ \frac{\partial f_{1,QI}}{\partial Q_1} = -\frac{1}{2\,\Delta t}\left[\frac{\rho L}{A} +
     * \frac{\partial I}{\partial A}\frac{\partial A}{\partial Q_1}(Q_1+Q_2-Q_1^n-Q_2^n)\right] \f$
     *
     * \f$ \frac{\partial f_{1,QI}}{\partial Q_2} = -\frac{1}{2\,\Delta t}\left[\frac{\rho L}{A} +
     * \frac{\partial I}{\partial A}\frac{\partial A}{\partial Q_2}(Q_1+Q_2-Q_1^n-Q_2^n)\right] \f$
     *
     * where \f$ I = \frac{\rho L}{A} \f$, \f$ \frac{\partial I}{\partial A} = -\frac{\rho L}{A^2}
     * \f$, and \f$ \frac{\partial A}{\partial Q_1} = \frac{\Delta t}{L} \f$,
     * \f$ \frac{\partial A}{\partial Q_2} = -\frac{\Delta t}{L} \f$.
     */
    std::pair<std::vector<double>, std::vector<double>> evaluate_inertia_derivative_kelvin_voigt(
        const std::vector<bool>& has_inertia, const AirwayData& data,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs, const KelvinVoigtWall& wall,
        double dt);

    /**
     * Creates the model-specific inertia evaluator function.
     */
    struct MakeInertiaEvaluator
    {
      using InertiaEvaluator =
          std::function<std::vector<double>(const AirwayData&, const std::vector<double>& area)>;

      template <typename M>
      InertiaEvaluator operator()(const M& model) const
      {
        using HasInertiaT = decltype(std::declval<M>().has_inertia);
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
    };

    /**
     * Creates the model-specific negative residual evaluator function for airways.
     */
    struct MakeNegativeResidualEvaluator
    {
      /**
       * Creates the model-specific flow resistance evaluator.
       */
      struct MakeFlowResistanceEvaluator
      {
        struct RigidWallTag
        {
        };
        struct KelvinVoigtWallTag
        {
        };

        // Evaluator type for both wall models: (AirwayData, DOF vector, area) -> flow resistance
        // vector
        using FlowResistanceEvaluator = std::function<std::vector<double>(
            const AirwayData&, const Core::LinAlg::Vector<double>&, const std::vector<double>&)>;

        FlowResistanceEvaluator operator()(
            RigidWallTag /*tag*/, const LinearResistive& /*model*/) const
        {
          return [](const AirwayData& data, const Core::LinAlg::Vector<double>& /*dofs*/,
                     const std::vector<double>& area)
          { return ComputePoiseuilleResistance{}(data, area); };
        }

        FlowResistanceEvaluator operator()(
            RigidWallTag /*tag*/, const NonLinearResistive& model) const
        {
          return [&model](const AirwayData& data, const Core::LinAlg::Vector<double>& dofs,
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

        FlowResistanceEvaluator operator()(
            KelvinVoigtWallTag /*tag*/, const LinearResistive& /*model*/) const
        {
          return [](const AirwayData& data, const Core::LinAlg::Vector<double>& /*dofs*/,
                     const std::vector<double>& area)
          { return ComputePoiseuilleResistance{}(data, area); };
        }

        FlowResistanceEvaluator operator()(
            KelvinVoigtWallTag /*tag*/, const NonLinearResistive& model) const
        {
          return [&model](const AirwayData& data, const Core::LinAlg::Vector<double>& dofs,
                     const std::vector<double>& area)
          {
            auto poiseuille = ComputePoiseuilleResistance{}(data, area);
            std::vector<double> resistance(area.size());
            for (size_t i = 0; i < resistance.size(); ++i)
            {
              double alpha = 4.0 * model.k_turb[i] / (4.0 * model.k_turb[i] - 1.0);
              resistance[i] = poiseuille[i] * model.k_turb[i] +
                              2 * data.air_properties.density * alpha / (area[i] * area[i]) *
                                  (dofs.local_values_as_span()[data.lid_q2[i]] -
                                      dofs.local_values_as_span()[data.lid_q1[i]]);
            }
            return resistance;
          };
        }
      };

      NegativeResidualEvaluator operator()(RigidWall& rigid_wall_model)
      {
        auto resistance_factory = MakeFlowResistanceEvaluator{};
        auto resistance_evaluator = std::visit([&resistance_factory](const auto& flow_model)
            { return resistance_factory(MakeFlowResistanceEvaluator::RigidWallTag{}, flow_model); },
            flow_model);
        auto inertia_evaluator = std::visit(MakeInertiaEvaluator{}, flow_model);
        return [resistance_evaluator, inertia_evaluator](const AirwayData& airway_data,
                   Core::LinAlg::Vector<double>& target_vector,
                   const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt)
        {
          std::vector<double> resistance =
              resistance_evaluator(airway_data, locally_relevant_dofs, airway_data.ref_area);

          std::vector<double> inertia = inertia_evaluator(airway_data, airway_data.ref_area);

          evaluate_negative_rigid_wall_residual(
              target_vector, airway_data, locally_relevant_dofs, resistance, inertia, dt);
        };
      }

      NegativeResidualEvaluator operator()(KelvinVoigtWall& kelvin_voigt_wall_model)
      {
        auto resistance_factory = MakeFlowResistanceEvaluator{};
        auto resistance_evaluator = std::visit(
            [&resistance_factory](const auto& flow_model)
            {
              return resistance_factory(
                  MakeFlowResistanceEvaluator::KelvinVoigtWallTag{}, flow_model);
            },
            flow_model);
        auto inertia_evaluator = std::visit(MakeInertiaEvaluator{}, flow_model);
        return [resistance_evaluator, inertia_evaluator, &kelvin_voigt_wall_model](
                   const AirwayData& airway_data, Core::LinAlg::Vector<double>& target_vector,
                   const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt)
        {
          std::vector<double> resistance = resistance_evaluator(
              airway_data, locally_relevant_dofs, kelvin_voigt_wall_model.area);

          std::vector<double> inertia =
              inertia_evaluator(airway_data, kelvin_voigt_wall_model.area);

          evaluate_negative_kelvin_voigt_wall_residual(target_vector, kelvin_voigt_wall_model,
              airway_data, locally_relevant_dofs, resistance, inertia, dt);
        };
      }
      FlowModel& flow_model;
    };

    // Creates the model-specific Jacobian evaluator function for airways.
    struct MakeJacobianEvaluator
    {
      /**
       * Creates the model-specific derivative evaluator of the flow resistance term.
       * The evaluators return the derivatives of the flow resistance term w.r.t. the flow DOFs.
       * \f$ \frac{\partial f_{1,QR}}{\partial Q_1} \f$ for rigid walls
       * and \f$ \frac{\partial f_{1,QR}}{\partial Q_1} \f$, \f$ \frac{\partial f_{1,QR}}{\partial
       * Q_2}
       * \f$ for Kelvin-Voigt walls.
       */
      struct MakeFlowResistanceDerivativeEvaluator
      {
        struct RigidWallTag
        {
        };
        struct KelvinVoigtWallTag
        {
        };

        KelvinVoigtWall* kv_wall_model = nullptr;

        // Rigid wall evaluators return single vector for the derivatives w.r.t. Q1
        using FlowResistanceDerivativeEvaluatorRigid = std::function<std::vector<double>(
            const AirwayData&, const Core::LinAlg::Vector<double>&, double)>;

        // Kelvin-Voigt wall evaluators return pair of vectors for the derivatives w.r.t. Q1 and Q2
        using FlowResistanceDerivativeEvaluatorKelvinVoigt =
            std::function<std::pair<std::vector<double>, std::vector<double>>(
                const AirwayData&, const Core::LinAlg::Vector<double>&, double)>;

        FlowResistanceDerivativeEvaluatorRigid operator()(
            RigidWallTag /*tag*/, const LinearResistive& /*model*/) const
        {
          return [](const AirwayData& data, const Core::LinAlg::Vector<double>&, double)
          { return ComputePoiseuilleResistance{}(data, data.ref_area); };
        }

        FlowResistanceDerivativeEvaluatorRigid operator()(
            RigidWallTag /*tag*/, const NonLinearResistive& model) const
        {
          return
              [&model](const AirwayData& data, const Core::LinAlg::Vector<double>& dofs, double dt)
          { return evaluate_nonlinear_flow_resistance_derivative_rigid(model, data, dofs, dt); };
        }

        FlowResistanceDerivativeEvaluatorKelvinVoigt operator()(
            KelvinVoigtWallTag /*tag*/, const LinearResistive& flow_resistance_model) const
        {
          KelvinVoigtWall* wall = kv_wall_model;
          return [wall, &flow_resistance_model](
                     const AirwayData& data, const Core::LinAlg::Vector<double>& dofs, double dt)
          {
            return evaluate_linear_flow_resistance_derivative_kelvin_voigt(
                flow_resistance_model, data, dofs, wall->area, dt);
          };
        }

        FlowResistanceDerivativeEvaluatorKelvinVoigt operator()(
            KelvinVoigtWallTag /*tag*/, const NonLinearResistive& flow_resistance_model) const
        {
          KelvinVoigtWall* wall = kv_wall_model;
          return [wall, &flow_resistance_model](
                     const AirwayData& data, const Core::LinAlg::Vector<double>& dofs, double dt)
          {
            return evaluate_nonlinear_flow_resistance_derivative_kelvin_voigt(
                flow_resistance_model, data, dofs, wall->area, dt);
          };
        }
      };

      // Creates the derivative evaluator for the interia term in Kelvin-Voigt walls.
      struct MakeInertiaDerivativeEvaluatorKelvinVoigt
      {
        KelvinVoigtWall* kv_wall_model;
        using InertiaDerivativeEvaluatorKelvinVoigt =
            std::function<std::pair<std::vector<double>, std::vector<double>>(
                const AirwayData&, const Core::LinAlg::Vector<double>&, double)>;
        template <typename M>
        InertiaDerivativeEvaluatorKelvinVoigt operator()(const M& model) const
        {
          using HasInertiaT = decltype(std::declval<M>().has_inertia);
          static_assert(std::is_same_v<std::remove_reference_t<HasInertiaT>, std::vector<bool>>,
              "Model must have member 'has_inertia' of type std::vector<bool>");

          KelvinVoigtWall* wall = kv_wall_model;
          return [&model, wall](
                     const AirwayData& data, const Core::LinAlg::Vector<double>& dofs, double dt)
          {
            return evaluate_inertia_derivative_kelvin_voigt(
                model.has_inertia, data, dofs, *wall, dt);
          };
        }
      };

      FlowModel& flow_model;

      JacobianEvaluatorAW operator()(RigidWall& rigid_wall_model)
      {
        auto resistance_derivative_factory = MakeFlowResistanceDerivativeEvaluator{};
        auto resistance_derivative_evaluator = std::visit(
            [&resistance_derivative_factory](const auto& flow_model)
            {
              return resistance_derivative_factory(
                  MakeFlowResistanceDerivativeEvaluator::RigidWallTag{}, flow_model);
            },
            flow_model);
        auto inertia_evaluator = std::visit(MakeInertiaEvaluator{}, flow_model);
        return [resistance_derivative_evaluator, inertia_evaluator](const AirwayData& airway_data,
                   Core::LinAlg::SparseMatrix& target,
                   const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt)
        {
          std::vector<double> resistance_derivative =
              resistance_derivative_evaluator(airway_data, locally_relevant_dofs, dt);
          std::vector<double> inertia_derivative =
              [dt, &evaluator = inertia_evaluator, &data = airway_data]()
          {
            auto vals = evaluator(data, data.ref_area);
            for (auto& v : vals) v /= dt;
            return vals;
          }();
          evaluate_jacobian_rigid_wall(
              target, airway_data, resistance_derivative, inertia_derivative, dt);
        };
      }

      JacobianEvaluatorAW operator()(KelvinVoigtWall& kelvin_voigt_wall_model)
      {
        auto resistance_derivative_factory =
            MakeFlowResistanceDerivativeEvaluator{&kelvin_voigt_wall_model};
        auto resistance_derivative_evaluator = std::visit(
            [&resistance_derivative_factory](const auto& flow_model)
            {
              return resistance_derivative_factory(
                  MakeFlowResistanceDerivativeEvaluator::KelvinVoigtWallTag{}, flow_model);
            },
            flow_model);
        auto inertia_derivative_evaluator = std::visit(
            MakeInertiaDerivativeEvaluatorKelvinVoigt{&kelvin_voigt_wall_model}, flow_model);
        return [resistance_derivative_evaluator, inertia_derivative_evaluator,
                   &kelvin_voigt_wall_model](const AirwayData& airway_data,
                   Core::LinAlg::SparseMatrix& target,
                   const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt)
        {
          auto resistance_derivative =
              resistance_derivative_evaluator(airway_data, locally_relevant_dofs, dt);
          auto inertia_derivative =
              inertia_derivative_evaluator(airway_data, locally_relevant_dofs, dt);
          auto viscous_wall_resistance_derivative =
              evaluate_viscous_wall_resistance_derivative_kelvin_voigt(
                  kelvin_voigt_wall_model, airway_data, locally_relevant_dofs, dt);

          evaluate_jacobian_kelvin_voigt_wall(target, airway_data, kelvin_voigt_wall_model,
              resistance_derivative, inertia_derivative, viscous_wall_resistance_derivative, dt);
        };
      }
    };

    // Creates the model-specific internal state updater function for airways.
    struct MakeInternalStateUpdater
    {
      struct MakeInternalStateUpdaterFlowModel
      {
        using InternalStateUpdaterFlowModel =
            std::function<void(AirwayData&, const Core::LinAlg::Vector<double>&)>;
        InternalStateUpdaterFlowModel operator()(LinearResistive& /*model*/)
        {
          return [](AirwayData& /*data*/,
                     const Core::LinAlg::Vector<double>& /*locally_relevant_dofs*/)
          {
            // No internal state to update for linear resistive model
          };
        }
        InternalStateUpdaterFlowModel operator()(NonLinearResistive& model)
        {
          return [&model](AirwayData& data, const Core::LinAlg::Vector<double>& dofs)
          {
            // Update internal state variable 'resistance' based on current flow rates
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
        };
      };
      InternalStateUpdaterAW operator()(RigidWall& /*rigid_wall_model*/)
      {
        auto internal_state_updater_flow_model =
            std::visit(MakeInternalStateUpdaterFlowModel{}, flow_model);
        return [internal_state_updater_flow_model](AirwayData& data,
                   const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt)
        { internal_state_updater_flow_model(data, locally_relevant_dofs); };
      }
      InternalStateUpdaterAW operator()(KelvinVoigtWall& kv_model)
      {
        auto internal_state_updater_flow_model =
            std::visit(MakeInternalStateUpdaterFlowModel{}, flow_model);
        return [internal_state_updater_flow_model, &kv_model](AirwayData& data,
                   const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt)
        {
          internal_state_updater_flow_model(data, locally_relevant_dofs);
          for (size_t i = 0; i < data.number_of_elements(); i++)
          {
            kv_model.beta_w[i] =
                std::sqrt(M_PI) * kv_model.wall_thickness[i] * kv_model.wall_elasticity[i] /
                ((1 - kv_model.wall_poisson_ratio[i] * kv_model.wall_poisson_ratio[i]) *
                    data.ref_area[i]);
            kv_model.gamma_w[i] = kv_model.beta_w[i] * kv_model.viscous_time_constant[i] *
                                  std::tan(kv_model.viscous_phase_shift[i]) / (4.0 * M_PI);
            kv_model.compliance_C[i] =
                2 * std::sqrt(data.ref_area[i]) * data.ref_length[i] / kv_model.beta_w[i];
            kv_model.viscous_resistance_Rvisc[i] =
                kv_model.gamma_w[i] / (std::sqrt(data.ref_area[i]) * data.ref_length[i]);
            kv_model.area[i] =
                kv_model.area_n[i] +
                dt / data.ref_length[i] *
                    (locally_relevant_dofs.local_values_as_span()[data.lid_q1[i]] -
                        locally_relevant_dofs.local_values_as_span()[data.lid_q2[i]]);
          }
        };
      }
      FlowModel& flow_model;
    };

    /**
     * Creates the model-specific routine at the end of the timestep that updates history variables
     * that need to be tracked over time for model evaluation
     */
    struct MakeEndOfTimestepRoutine
    {
      EndOfTimestepRoutine operator()(RigidWall& rigid_wall_model)
      {
        return [&](AirwayData& data, const Core::LinAlg::Vector<double>& locally_relevant_dofs,
                   double dt) {};
      }
      EndOfTimestepRoutine operator()(KelvinVoigtWall& kelvin_voigt_wall_model)
      {
        return [&](AirwayData& data, const Core::LinAlg::Vector<double>& locally_relevant_dofs,
                   double dt)
        {
          for (size_t i = 0; i < data.number_of_elements(); i++)
          {
            // only in Kelvin-Voigt walls q2_n is stored as history variable
            data.q2_n[i] = locally_relevant_dofs.local_values_as_span()[data.lid_q2[i]];

            kelvin_voigt_wall_model.area_n[i] =
                kelvin_voigt_wall_model.area_n[i] +
                dt / data.ref_length[i] *
                    (locally_relevant_dofs.local_values_as_span()[data.lid_q1[i]] -
                        locally_relevant_dofs.local_values_as_span()[data.lid_q2[i]]);
          }
        };
      }
    };

    /**
     * @brief Returns an existing AirwayModel with specified flow and wall models
     *
     * @tparam F The flow model type.
     * @tparam W The wall model type.
     * @param airways Global container for all airway models.
     * @return Reference to the matching AirwayModel.
     */
    template <typename F, typename W>
    AirwayModel& register_or_access_airway_model(AirwayContainer& airways)
    {
      for (auto& model : airways.models)
      {
        if (std::holds_alternative<F>(model.flow_model) &&
            std::holds_alternative<W>(model.wall_model))
        {
          return model;
        }
      }

      // Create instance of the AirwayModel
      airways.models.emplace_back();
      AirwayModel& new_model = airways.models.back();

      // Create models
      new_model.flow_model = F{};
      new_model.wall_model = W{};

      // replace the previous if/constexpr chain with a compile-time sum
      int n_state_eq = FlowModelStateCount<F>::value + WallModelStateCount<W>::value;
      new_model.data.n_state_equations = n_state_eq;

      return new_model;
    }


    struct AddFlowModelParameter
    {
      void operator()(LinearResistive& model)
      {
        model.has_inertia.push_back(params->lung_tree.airways.flow_model.include_inertia.at(
            global_element_id, "include_inertia"));
      }
      void operator()(NonLinearResistive& model)
      {
        model.turbulence_factor_gamma.push_back(
            params->lung_tree.airways.flow_model.resistance_model.non_linear.turbulence_factor_gamma
                .at(global_element_id, "turbulence_factor_gamma"));
        model.has_inertia.push_back(params->lung_tree.airways.flow_model.include_inertia.at(
            global_element_id, "include_inertia"));
      }
      int global_element_id;
      const ReducedLungParameters* params;
    };


    struct AddWallModelParameter
    {
      void operator()(RigidWall& model)
      {
        // nothing to do here
      }
      void operator()(KelvinVoigtWall& model)
      {
        model.wall_poisson_ratio.push_back(params->kelvin_voigt.elasticity.wall_poisson_ratio.at(
            global_element_id, "wall_poisson_ratio"));
        model.wall_elasticity.push_back(params->kelvin_voigt.elasticity.wall_elasticity.at(
            global_element_id, "wall_elasticity"));
        model.wall_thickness.push_back(
            params->kelvin_voigt.elasticity.wall_thickness.at(global_element_id, "wall_thickness"));
        model.viscous_time_constant.push_back(
            params->kelvin_voigt.viscosity.viscous_time_constant.at(
                global_element_id, "viscous_time_constant"));
        model.viscous_phase_shift.push_back(params->kelvin_voigt.viscosity.viscous_phase_shift.at(
            global_element_id, "viscous_phase_shift"));
        model.area_n.push_back(ref_area);
      }
      int global_element_id;
      const ReducedLungParameters::LungTree::Airways::WallModel* params;
      double ref_area;
    };


    /**
     * @brief Adds a new element to the appropriate AirwayModel group.
     *
     * Computes length and area, assigns material parameters, and initializes internal state.
     *
     * @tparam F Flow model.
     * @tparam W Wall model.
     * @param airways Global container for all airway models.
     * @param ele Pointer to element to be added.
     * @param local_element_id Local identifier in 4C discretization.
     */
    template <typename F, typename W>
    void add_airway_ele(AirwayContainer& airways, int global_element_id, int local_element_id,
        const ReducedLungParameters& params)
    {
      AirwayModel& model = register_or_access_airway_model<F, W>(airways);

      model.data.global_element_id.push_back(global_element_id);
      model.data.local_element_id.push_back(local_element_id);
      const auto& node_ids =
          params.lung_tree.topology.element_nodes.at(global_element_id, "element_nodes");
      const int node_in = node_ids[0] - 1;
      const int node_out = node_ids[1] - 1;
      const auto& coords_node_1 =
          params.lung_tree.topology.node_coordinates.at(node_in, "node_coordinates");
      const auto& coords_node_2 =
          params.lung_tree.topology.node_coordinates.at(node_out, "node_coordinates");
      const double length =
          std::sqrt((coords_node_1[0] - coords_node_2[0]) * (coords_node_1[0] - coords_node_2[0]) +
                    (coords_node_1[1] - coords_node_2[1]) * (coords_node_1[1] - coords_node_2[1]) +
                    (coords_node_1[2] - coords_node_2[2]) * (coords_node_1[2] - coords_node_2[2]));
      model.data.ref_length.push_back(length);
      const double radius = params.lung_tree.airways.radius.at(global_element_id, "radius");
      const double area = radius * radius * M_PI;
      model.data.air_properties.dynamic_viscosity = params.air_properties.dynamic_viscosity;
      model.data.air_properties.density = params.air_properties.density;
      model.data.ref_area.push_back(area);

      // Initialize history vectors to zero
      model.data.q1_n.push_back(0.0);
      model.data.q2_n.push_back(0.0);
      model.data.p1_n.push_back(0.0);
      model.data.p2_n.push_back(0.0);

      std::visit(AddFlowModelParameter{.global_element_id = global_element_id, .params = &params},
          model.flow_model);
      std::visit(AddWallModelParameter{.global_element_id = global_element_id,
                     .params = &params.lung_tree.airways.wall_model,
                     .ref_area = area},
          model.wall_model);
    }

    /**
     * @brief Assembles the negative residual vector of all airways.
     *
     * Applies model-specific logic and calculates the residuals of each airway element and stores
     * them.
     *
     * @param res_vector Residual vector in row map layout
     * @param airways Global container for all airway models.
     * @param locally_relevant_dofs DOF vector containing current inflow and pressure values in
     * column map layout.
     * @param dt Current time step size.
     */
    void update_negative_residual_vector(Core::LinAlg::Vector<double>& res_vector,
        AirwayContainer& airways, const Core::LinAlg::Vector<double>& locally_relevant_dofs,
        double dt);


    /**
     * @brief Assembles the Jacobian matrix contributions of all airways.
     *
     * Derivatives w.r.t. pressure and flow variables are computed using model-specific logic. They
     * are directly stored in the given Jacobian.
     *
     * @param jac Jacobian matrix in (row, column) map layout.
     * @param airways Global container for all airway models.
     * @param locally_relevant_dofs DOF vector containing current inflow and pressure values in
     * column map layout.
     * @param dt Current time step size.
     */
    void update_jacobian(Core::LinAlg::SparseMatrix& jac, AirwayContainer& airways,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt);

    /**
     * @brief Creates evaluator functions for all airway models.
     *
     * This function must be called after all airway elements have been added.
     * It initializes the evaluator function objects for each model, ensuring that references
     * captured by these evaluators remain valid.
     */
    void create_evaluators(AirwayContainer& airways);

    /**
     * @brief Assign local equation ids to airway state equations.
     */
    void assign_local_equation_ids(AirwayContainer& airways, int& n_local_equations);

    /**
     * @brief Assign local dof ids from the locally relevant dof map.
     */
    void assign_local_dof_ids(
        const Core::LinAlg::Map& locally_relevant_dof_map, AirwayContainer& airways);

    /**
     * @brief Updates the internal state memory of each airway model.
     *
     * To be executed inside the nonlinear loop.
     *
     * @param airways All grouped airway models.
     * @param locally_relevant_dofs DOF vector containing inflow and pressure values in
     * column map layout.
     * @param dt Time step size.
     */
    void update_internal_state_vectors(AirwayContainer& airways,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt);

    /**
     * @brief Routine to be executed at the end of each time step.
     *
     * Used to store history variables (e.g. flow, pressure, area) from previous time steps.
     *
     * @param airways All grouped airway models.
     * @param locally_relevant_dofs DOF vector containing inflow and pressure values in
     * column map layout. The DOFs have to be in a converged state.
     * @param dt Time step size.
     *
     * @note Needs to be executed once between time steps to update the internal state with the
     * converged DOFs.
     */
    void end_of_timestep_routine(AirwayContainer& airways,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt);
  }  // namespace Airways
}  // namespace ReducedLung

FOUR_C_NAMESPACE_CLOSE

#endif

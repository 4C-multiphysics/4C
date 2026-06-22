// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MIXTURE_REMODELFIBER_INTERNAL_HPP
#define FOUR_C_MIXTURE_REMODELFIBER_INTERNAL_HPP

#include "4C_config.hpp"

#include "4C_linalg_fixedsizematrix.hpp"

#include <Sacado_tradvec.hpp>

#include <memory>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::Communication
{
  class PackBuffer;
  class UnpackBuffer;
}  // namespace Core::Communication

namespace Mixture
{
  template <typename T>
  class RemodelFiberMaterial;
  template <typename T>
  class LinearCauchyGrowthWithPoissonTurnoverGrowthEvolution;


  namespace Implementation
  {
    // Definition of the time integration routine
    template <int numstates, typename T>
    struct IntegrationState
    {
      std::array<T, numstates> x;
      std::array<T, numstates> f;
    };

    template <int numstates, typename T>
    class ImplicitIntegration;

    // Corresponds to a one-step-theta method with theta=0.5 (trapezoidal rule)
    template <typename T>
    class ImplicitIntegration<2, T>
    {
     public:
      static constexpr double theta = 0.5;
      static inline T get_residuum(
          const Mixture::Implementation::IntegrationState<2, T>& state, const T dt)
      {
        return state.x[1] - state.x[0] - dt * ((1.0 - theta) * state.f[0] + theta * state.f[1]);
      }

      static inline T get_partial_derivative_xnp(
          const Mixture::Implementation::IntegrationState<2, T>& state, T dt)
      {
        return 1.0;
      }

      static inline T get_partial_derivative_fnp(
          const Mixture::Implementation::IntegrationState<2, T>& state, T dt)
      {
        return -dt * theta;
      }
    };

    template <int numstates, typename T>
    class RemodelFiberImplementation
    {
      struct GRState
      {
        T growth_scalar = 1.0;
        T lambda_r = 1.0;
        T lambda_f = 1.0;
        T lambda_ext = 1.0;
      };

     public:
      RemodelFiberImplementation(std::shared_ptr<const RemodelFiberMaterial<T>> material,
          LinearCauchyGrowthWithPoissonTurnoverGrowthEvolution<T> growth_evolution, T lambda_pre);

      /*!
       * @brief Pack all internal data into the #data
       *
       * @param data (out) : buffer to serialize data to.
       */
      void pack(Core::Communication::PackBuffer& data) const;

      /*!
       * @brief Unpack all internal data that was previously packed by
       * #pack(Core::Communication::PackBuffer&)
       *
       * @param position (in/out) : Position, where to start reading
       * @param data (in) : Vector of chars to extract data from
       */
      void unpack(Core::Communication::UnpackBuffer& buffer);

      /// @brief Updates previous history data
      void update();

      /*!
       * @brief Sets the deposition (homeostatic) stretch.
       *
       * @param lambda_pre
       */
      void update_deposition_stretch(T lambda_pre);

      /*!
       * @brief Set deformation state of the fiber
       *
       * @note This method has to be called before any Evaluation or local integration
       *
       * @param lambda_f (in) : total stretch in fiber direction
       * @param lambda_ext (in) : external inelastic stretch in fiber direction
       */
      void set_state(T lambda_f, T lambda_ext);
      void set_growth_scalar(T growth_scalar);
      void set_lambda_r(T lambda_r);

      [[nodiscard]] T evaluate_growth_reaction_coefficient(
          T lambda_f, T lambda_r, T lambda_ext) const;
      [[nodiscard]] T evaluate_growth_evolution_equation_dt_with_nonlocal_stimulus(
          T psi, T lambda_f, T lambda_r, T lambda_ext, T growth_scalar) const;
      [[nodiscard]] T evaluate_growth_evolution_equation_dt(
          T lambda_f, T lambda_r, T lambda_ext, T growth_scalar) const;
      [[nodiscard]] T evaluate_d_growth_evolution_equation_dt_d_sig(
          T lambda_f, T lambda_r, T lambda_ext, T growth_scalar) const;
      [[nodiscard]] T evaluate_d_growth_evolution_equation_dt_partial_dgrowth(
          T lambda_f, T lambda_r, T lambda_ext, T growth_scalar) const;
      [[nodiscard]] T evaluate_d_growth_evolution_equation_dt_partial_d_remodel(
          T lambda_f, T lambda_r, T lambda_ext, T growth_scalar) const;
      [[nodiscard]] T evaluate_d_growth_evolution_equation_dt_d_growth(
          T lambda_f, T lambda_r, T lambda_ext, T growth_scalar) const;
      [[nodiscard]] T evaluate_d_growth_evolution_equation_dt_d_remodel(
          T lambda_f, T lambda_r, T lambda_ext, T growth_scalar) const;
      [[nodiscard]] T evaluate_d_growth_evolution_equation_dt_d_lambda_ext(
          T lambda_f, T lambda_r, T lambda_ext, T growth_scalar) const;

      [[nodiscard]] T evaluate_remodel_evolution_equation_dt(
          T lambda_f, T lambda_r, T lambda_ext) const;
      [[nodiscard]] T evaluate_remodeling_reaction_coefficient(
          T lambda_f, T lambda_r, T lambda_ext) const;
      [[nodiscard]] T evaluate_d_remodel_evolution_equation_dt_d_sig(
          T lambda_f, T lambda_r, T lambda_ext) const;
      [[nodiscard]] T evaluate_d_remodel_evolution_equation_dt_d_i4(
          T lambda_f, T lambda_r, T lambda_ext) const;
      [[nodiscard]] T evaluate_d_remodel_evolution_equation_dt_partial_d_growth(
          T lambda_f, T lambda_r, T lambda_ext) const;
      [[nodiscard]] T evaluate_d_remodel_evolution_equation_dt_partial_d_remodel(
          T lambda_f, T lambda_r, T lambda_ext) const;
      [[nodiscard]] T evaluate_d_remodel_evolution_equation_dt_d_growth(
          T lambda_f, T lambda_r, T lambda_ext) const;
      [[nodiscard]] T evaluate_d_remodel_evolution_equation_dt_d_remodel(
          T lambda_f, T lambda_r, T lambda_ext) const;
      [[nodiscard]] T evaluate_d_remodel_evolution_equation_dt_d_lambda_ext(
          T lambda_f, T lambda_r, T lambda_ext) const;

      [[nodiscard]] T evaluate_fiber_cauchy_stress(T lambda_f, T lambda_r, T lambda_ext) const;
      [[nodiscard]] T evaluate_d_fiber_cauchy_stress_partial_d_i4(
          T lambda_f, T lambda_r, T lambda_ext) const;
      [[nodiscard]] T evaluate_d_fiber_cauchy_stress_partial_d_i4_d_i4(
          T lambda_f, T lambda_r, T lambda_ext) const;
      [[nodiscard]] T evaluate_d_fiber_cauchy_stress_d_remodel(
          T lambda_f, T lambda_r, T lambda_ext) const;

      [[nodiscard]] IntegrationState<numstates, T> get_integration_state_growth_scalar() const;

      [[nodiscard]] IntegrationState<numstates, T> get_integration_state_lambda_r() const;

      [[nodiscard]] IntegrationState<numstates, T>
      get_integration_state_growth_scalar_with_nonlocal_stimulus(T psi) const;

      /// @name Methods for doing explicit or implicit time integration
      /// @{
      /*!
       * @brief Integrate the local evolution equation with an implicit time integration scheme.
       *
       * @param dt (in) : timestep
       */
      void integrate_local_evolution_equations_implicit(T dt);

      /*!
       * @brief Integrate the local evolution equation with an explicit time integration scheme.
       *
       * @param dt (in) : timestep
       */
      void integrate_local_evolution_equations_explicit(T dt);

      /*!
       * @brief Integrate the local evolution equations explicitly, driven by a non-local stimulus
       * \psi.
       *
       * @param psi (in) : non-local stimulus \psi
       * @param dt (in) : timestep
       */
      void integrate_local_evolution_equations_explicit_with_nonlocal_stimulus(T psi, T dt);

      /*!
       * @brief Integrate the local evolution equations implicitly, driven by a non-local stimulus
       * \psi.
       *
       * @param psi (in) : non-local stimulus \psi
       * @param dt (in) : timestep
       */
      void integrate_local_evolution_equations_implicit_with_nonlocal_stimulus(
          const T psi, const T dt);

      /// @}
      /// @brief Evaluation methods
      ///
      /// @note It is important to call #set_state(#T) first.
      ///
      /// @{
      [[nodiscard]] T evaluate_current_homeostatic_fiber_cauchy_stress() const;
      [[nodiscard]] T evaluate_current_fiber_cauchy_stress() const;
      [[nodiscard]] T evaluate_current_fiber_pk2_stress() const;
      [[nodiscard]] T evaluate_d_current_fiber_pk2_stress_d_lambda_f_sq() const;
      [[nodiscard]] T evaluate_d_current_fiber_pk2_stress_d_lambda_r() const;
      [[nodiscard]] T evaluate_d_current_fiber_pk2_stress_d_lambda_ext() const;
      void set_d_lambda_ext_d_growth_scalar(T value);
      [[nodiscard]] T get_d_lambda_ext_d_growth_scalar() const;
      [[nodiscard]] T
      evaluate_d_current_growth_evolution_implicit_time_integration_residuum_d_lambda_f_sq(
          T dt) const;
      [[nodiscard]] T
      evaluate_d_current_remodel_evolution_implicit_time_integration_residuum_d_lambda_f_sq(
          T dt) const;
      [[nodiscard]] T evaluate_current_growth_scalar() const;
      [[nodiscard]] T evaluate_current_lambda_r() const;
      [[nodiscard]] T evaluate_growth_reaction_coefficient() const;
      [[nodiscard]] T evaluate_remodeling_reaction_coefficient() const;

      [[nodiscard]] T evaluate_d_current_growth_scalar_d_lambda_f_sq() const;
      [[nodiscard]] T evaluate_d_current_lambda_r_d_lambda_f_sq() const;
      [[nodiscard]] T evaluate_d_current_cauchy_stress_d_lambda_f_sq() const;
      [[nodiscard]] T evaluate_d_growth_scalar_d_nonlocal_stimulus() const;


      /// @}

      /// array of G&R states (the last state in the array is the current state)
      std::array<GRState, numstates> states_;

      /// current derivatives w.r.t. squared lambda_f
      /// @{
      T d_growth_scalar_d_lambda_f_sq_ = 0.0;
      T d_lambda_r_d_lambda_f_sq_ = 0.0;
      /// @}

      /// d_growth_scalar^{n+1}_d_psi - stored during integrate_..._with_nonlocal_stimulus calls.
      /// Used for K_u_psi assembly without recomputing from scratch.
      T d_growth_scalar_d_stimulus_ = 0.0;

      /// d_lambda_ext_d_growth_scalar_ - set from outside (growth strategy via mixture rule).
      /// Non-zero only for inelastic growth strategies where Fg depends on growth_scalar.
      T d_lambda_ext_d_growth_scalar_ = 0.0;

      /// homeostatic quantities
      /// @{
      T sig_h_ = 0.0;
      T lambda_pre_ = 1.0;
      /// @}

      /// Strain energy function of the fiber
      const std::shared_ptr<const RemodelFiberMaterial<T>> fiber_material_;

      /// Growth evolution equation
      const LinearCauchyGrowthWithPoissonTurnoverGrowthEvolution<T> growth_evolution_;

#ifdef FOUR_C_ENABLE_ASSERTIONS
      bool state_is_set_ = false;
#endif
    };
  }  // namespace Implementation
}  // namespace Mixture

FOUR_C_NAMESPACE_CLOSE

#endif
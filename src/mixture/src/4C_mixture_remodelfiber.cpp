// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mixture_remodelfiber.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_comm_parobject.hpp"
#include "4C_mixture_constituent_remodelfiber_material.hpp"
#include "4C_mixture_growth_evolution_linear_cauchy_poisson_turnover.hpp"
#include "4C_mixture_remodelfiber-internal.hpp"
#include "4C_utils_fad.hpp"

#include <Sacado.hpp>

#include <algorithm>
#include <memory>
#include <type_traits>

FOUR_C_NAMESPACE_OPEN

// anonymous namespace for helper functions and classes
namespace
{
  template <int numstates, typename T>
  class ExplicitIntegration;

  // Corresponds to the explicit euler method
  template <typename T>
  class ExplicitIntegration<2, T>
  {
   public:
    static T integrate(const Mixture::Implementation::IntegrationState<2, T>& state, const T dt)
    {
      return state.x[0] + dt * state.f[0];
    }
  };

  template <typename T>
  [[nodiscard]] T evaluate_i4(T lambda_f, T lambda_r, T lambda_ext)
  {
    return std::pow(lambda_f, 2) / std::pow(lambda_r * lambda_ext, 2);
  }

  template <typename T>
  [[nodiscard]] T evaluate_di4_dlambda_r(T lambda_f, T lambda_r, T lambda_ext)
  {
    return -2.0 * std::pow(lambda_f, 2) / (std::pow(lambda_r * lambda_ext, 2) * lambda_r);
  }

  template <typename T>
  [[nodiscard]] T evaluate_di4_dlambda_f_sq(T lambda_f, T lambda_r, T lambda_ext)
  {
    return 1.0 / std::pow(lambda_r * lambda_ext, 2);
  }
}  // namespace

template <int numstates, typename T>
Mixture::Implementation::RemodelFiberImplementation<numstates, T>::RemodelFiberImplementation(
    std::shared_ptr<const Mixture::RemodelFiberMaterial<T>> material,
    Mixture::LinearCauchyGrowthWithPoissonTurnoverGrowthEvolution<T> growth_evolution, T lambda_pre)
    : lambda_pre_(lambda_pre),
      fiber_material_(std::move(material)),
      growth_evolution_(std::move(growth_evolution))
{
  std::for_each(
      states_.begin(), states_.end(), [&](auto& state) { state.lambda_r = 1.0 / lambda_pre; });
  sig_h_ = evaluate_fiber_cauchy_stress(1.0, 1.0 / lambda_pre_, 1.0);
}

template <int numstates, typename T>
void Mixture::Implementation::RemodelFiberImplementation<numstates, T>::pack(
    Core::Communication::PackBuffer& data) const
{
  if constexpr (!std::is_floating_point_v<T>)
  {
    FOUR_C_THROW(
        "Pack and Unpack is only available for floating point types. You are probably using a "
        "FAD-type.");
    return;
  }
  else
  {
    data.add_to_pack(lambda_pre_);

    for (const auto& state : states_)
    {
      data.add_to_pack(state.growth_scalar);
      data.add_to_pack(state.lambda_r);
      data.add_to_pack(state.lambda_f);
      data.add_to_pack(state.lambda_ext);
    }
  }
}

template <int numstates, typename T>
void Mixture::Implementation::RemodelFiberImplementation<numstates, T>::unpack(
    Core::Communication::UnpackBuffer& buffer)
{
  if constexpr (!std::is_floating_point_v<T>)
  {
    FOUR_C_THROW(
        "Pack and Unpack is only available for floating point types. You are probably using a "
        "FAD-type.");
    return;
  }
  else
  {
    extract_from_pack(buffer, lambda_pre_);
    sig_h_ = evaluate_fiber_cauchy_stress(1.0, 1.0 / lambda_pre_, 1.0);


    for (auto& state : states_)
    {
      extract_from_pack(buffer, state.growth_scalar);
      extract_from_pack(buffer, state.lambda_r);
      extract_from_pack(buffer, state.lambda_f);
      extract_from_pack(buffer, state.lambda_ext);
    }
  }
}

template <int numstates, typename T>
void Mixture::Implementation::RemodelFiberImplementation<numstates, T>::update()
{
  for (std::size_t i = 1; i < numstates; ++i)
  {
    std::swap(states_[i], states_[i - 1]);
  }

  // predictor: start from previous solution
  states_.back() = states_[states_.size() - 2];
#ifdef FOUR_C_ENABLE_ASSERTIONS
  state_is_set_ = false;
#endif
}



template <int numstates, typename T>
void Mixture::Implementation::RemodelFiberImplementation<numstates, T>::update_deposition_stretch(
    const T lambda_pre)
{
  std::for_each(states_.begin(), states_.end(),
      [&](auto& state) { state.lambda_r = lambda_pre_ / lambda_pre * state.lambda_r; });

  lambda_pre_ = lambda_pre;
  sig_h_ = evaluate_fiber_cauchy_stress(1.0, 1.0 / lambda_pre_, 1.0);
}

template <int numstates, typename T>
void Mixture::Implementation::RemodelFiberImplementation<numstates, T>::set_state(
    const T lambda_f, const T lambda_ext)
{
  states_.back().lambda_f = lambda_f;
  states_.back().lambda_ext = lambda_ext;
#ifdef FOUR_C_ENABLE_ASSERTIONS
  state_is_set_ = true;
#endif
}

template <int numstates, typename T>
Mixture::Implementation::IntegrationState<numstates, T>
Mixture::Implementation::RemodelFiberImplementation<numstates,
    T>::get_integration_state_growth_scalar() const
{
  return std::invoke(
      [&]
      {
        IntegrationState<numstates, T> growth_state;
        std::transform(states_.begin(), states_.end(), growth_state.x.begin(),
            [&](const GRState& state) { return state.growth_scalar; });
        std::transform(states_.begin(), states_.end(), growth_state.f.begin(),
            [&](const GRState& state)
            {
              return evaluate_growth_evolution_equation_dt(
                  state.lambda_f, state.lambda_r, state.lambda_ext, state.growth_scalar);
            });
        return growth_state;
      });
}

template <int numstates, typename T>
Mixture::Implementation::IntegrationState<numstates, T>
Mixture::Implementation::RemodelFiberImplementation<numstates, T>::get_integration_state_lambda_r()
    const
{
  return std::invoke(
      [&]
      {
        IntegrationState<numstates, T> remodel_state;
        std::transform(states_.begin(), states_.end(), remodel_state.x.begin(),
            [&](const GRState& state) { return state.lambda_r; });
        std::transform(states_.begin(), states_.end(), remodel_state.f.begin(),
            [&](const GRState& state)
            {
              return evaluate_remodel_evolution_equation_dt(
                  state.lambda_f, state.lambda_r, state.lambda_ext);
            });
        return remodel_state;
      });
}

template <int numstates, typename T>
Core::LinAlg::Matrix<2, 2, T> Mixture::Implementation::RemodelFiberImplementation<numstates,
    T>::integrate_local_evolution_equations_implicit(const T dt)
{
  FOUR_C_ASSERT(state_is_set_, "You have to call set_state() before!");
  const T lambda_f = states_.back().lambda_f;
  const T lambda_ext = states_.back().lambda_ext;
  const auto EvaluateLocalNewtonLinearSystem = [&]()
  {
    const IntegrationState<numstates, T> growth_state = get_integration_state_growth_scalar();

    const IntegrationState<numstates, T> remodel_state = get_integration_state_lambda_r();

    T residuum_growth = ImplicitIntegration<numstates, T>::get_residuum(growth_state, dt);
    T residuum_remodel = ImplicitIntegration<numstates, T>::get_residuum(remodel_state, dt);

    Core::LinAlg::Matrix<2, 1, T> residuum(Core::LinAlg::Initialization::uninitialized);
    residuum(0, 0) = residuum_growth;
    residuum(1, 0) = residuum_remodel;

    const T growth_scalar_np = states_.back().growth_scalar;
    const T lambda_r_np = states_.back().lambda_r;

    // evaluate growth and remodel matrices
    Core::LinAlg::Matrix<2, 2, T> drdx(Core::LinAlg::Initialization::uninitialized);
    drdx(0, 0) = ImplicitIntegration<numstates, T>::get_partial_derivative_xnp(growth_state, dt) +
                 ImplicitIntegration<numstates, T>::get_partial_derivative_fnp(growth_state, dt) *
                     evaluate_d_growth_evolution_equation_dt_d_growth(
                         lambda_f, lambda_r_np, lambda_ext, growth_scalar_np);
    drdx(0, 1) = ImplicitIntegration<numstates, T>::get_partial_derivative_fnp(growth_state, dt) *
                 evaluate_d_growth_evolution_equation_dt_d_remodel(
                     lambda_f, lambda_r_np, lambda_ext, growth_scalar_np);
    drdx(1, 0) =
        ImplicitIntegration<numstates, T>::get_partial_derivative_fnp(remodel_state, dt) *
        evaluate_d_remodel_evolution_equation_dt_d_growth(lambda_f, lambda_r_np, lambda_ext);
    drdx(1, 1) =
        ImplicitIntegration<numstates, T>::get_partial_derivative_xnp(remodel_state, dt) +
        ImplicitIntegration<numstates, T>::get_partial_derivative_fnp(remodel_state, dt) *
            evaluate_d_remodel_evolution_equation_dt_d_remodel(lambda_f, lambda_r_np, lambda_ext);

    return std::make_tuple(drdx, residuum);
  };

  Core::LinAlg::Matrix<2, 1, T> x_np(Core::LinAlg::Initialization::uninitialized);
  x_np(0) = states_.back().growth_scalar;
  x_np(1) = states_.back().lambda_r;

  Core::LinAlg::Matrix<2, 2, T> K(Core::LinAlg::Initialization::uninitialized);
  Core::LinAlg::Matrix<2, 1, T> b(Core::LinAlg::Initialization::uninitialized);
  std::tie(K, b) = EvaluateLocalNewtonLinearSystem();

  unsigned iteration = 0;
  while (Core::FADUtils::vector_norm(b) > 1e-10)
  {
    if (iteration >= 500)
    {
      FOUR_C_THROW(
          "The local newton didn't converge within 500 iterations. Residuum is {:.3e} > {:.3e}",
          Core::FADUtils::cast_to_double(Core::FADUtils::vector_norm(b)), 1e-10);
    }
    K.invert();
    x_np.multiply_nn(-1, K, b, 1.0);
    states_.back().growth_scalar = x_np(0);
    states_.back().lambda_r = x_np(1);
    std::tie(K, b) = EvaluateLocalNewtonLinearSystem();
    iteration += 1;
  }


  // compute increment w.r.t. lambda_f_sq
  const Core::LinAlg::Matrix<2, 1, T> d_residuum_d_lambda_f_sq = std::invoke(
      [&]()
      {
        Core::LinAlg::Matrix<2, 1, T> d_residuum_d_lambfa_f_sq;

        d_residuum_d_lambfa_f_sq(0, 0) =
            evaluate_d_current_growth_evolution_implicit_time_integration_residuum_d_lambda_f_sq(
                dt);
        d_residuum_d_lambfa_f_sq(1, 0) =
            evaluate_d_current_remodel_evolution_implicit_time_integration_residuum_d_lambda_f_sq(
                dt);

        return d_residuum_d_lambfa_f_sq;
      });



  K.invert();
  Core::LinAlg::Matrix<2, 1, T> d_growth_scalar_d_residuum(
      Core::LinAlg::Initialization::uninitialized);
  Core::LinAlg::Matrix<2, 1, T> d_lambda_r_d_residuum(Core::LinAlg::Initialization::uninitialized);

  for (std::size_t i = 0; i < 2; ++i)
  {
    d_growth_scalar_d_residuum(i) = K(0, i);
    d_lambda_r_d_residuum(i) = K(1, i);
  }

  // store current derivative of lambda_r and growth_scalar w.r.t. lambda_f
  d_growth_scalar_d_lambda_f_sq_ = -d_growth_scalar_d_residuum.dot(d_residuum_d_lambda_f_sq);
  d_lambda_r_d_lambda_f_sq_ = -d_lambda_r_d_residuum.dot(d_residuum_d_lambda_f_sq);

  return K;
}



template <int numstates, typename T>
void Mixture::Implementation::RemodelFiberImplementation<numstates,
    T>::integrate_local_evolution_equations_explicit(const T dt)
{
  const IntegrationState<numstates, T> growth_state = get_integration_state_growth_scalar();

  const IntegrationState<numstates, T> remodel_state = get_integration_state_lambda_r();


  // Update state
  states_.back().growth_scalar = ExplicitIntegration<numstates, T>::integrate(growth_state, dt);
  states_.back().lambda_r = ExplicitIntegration<numstates, T>::integrate(remodel_state, dt);

  // reset current derivative of lambda_r and growth_scalar w.r.t. lambda_f (zero in explicit time
  // integration)
  d_growth_scalar_d_lambda_f_sq_ = 0.0;
  d_lambda_r_d_lambda_f_sq_ = 0.0;
}



template <int numstates, typename T>
T Mixture::Implementation::RemodelFiberImplementation<numstates,
    T>::evaluate_growth_evolution_equation_dt(const T lambda_f, const T lambda_r,
    const T lambda_ext, const T growth_scalar) const
{
  const T dsig = (evaluate_fiber_cauchy_stress(lambda_f, lambda_r, lambda_ext) - sig_h_) / sig_h_;
  return (growth_evolution_.evaluate_true_mass_production_rate(dsig) +
             growth_evolution_.evaluate_true_mass_removal_rate(dsig)) *
         growth_scalar;
}

template <int numstates, typename T>
T Mixture::Implementation::RemodelFiberImplementation<numstates,
    T>::evaluate_d_growth_evolution_equation_dt_d_sig(const T lambda_f, const T lambda_r,
    const T lambda_ext, const T growth_scalar) const
{
  const T dsig = (evaluate_fiber_cauchy_stress(lambda_f, lambda_r, lambda_ext) - sig_h_) / sig_h_;
  return (growth_evolution_.evaluate_d_true_mass_production_rate_d_sig(dsig) +
             growth_evolution_.evaluate_d_true_mass_removal_rate_d_sig(dsig)) /
         sig_h_ * growth_scalar;
}

template <int numstates, typename T>
T Mixture::Implementation::RemodelFiberImplementation<numstates,
    T>::evaluate_d_growth_evolution_equation_dt_partial_dgrowth(const T lambda_f, const T lambda_r,
    const T lambda_ext, const T growth_scalar) const
{
  const T dsig = (evaluate_fiber_cauchy_stress(lambda_f, lambda_r, lambda_ext) - sig_h_) / sig_h_;
  return (growth_evolution_.evaluate_true_mass_production_rate(dsig) +
          growth_evolution_.evaluate_true_mass_removal_rate(dsig));
}

template <int numstates, typename T>
T Mixture::Implementation::RemodelFiberImplementation<numstates,
    T>::evaluate_d_growth_evolution_equation_dt_partial_d_remodel(const T lambda_f,
    const T lambda_r, const T lambda_ext, const T growth_scalar) const
{
  if constexpr (!std::is_floating_point_v<T>)
    return T(0.0);
  else
    return 0.0;
}

template <int numstates, typename T>
T Mixture::Implementation::RemodelFiberImplementation<numstates,
    T>::evaluate_d_growth_evolution_equation_dt_d_growth(const T lambda_f, const T lambda_r,
    const T lambda_ext, const T growth_scalar) const
{
  return evaluate_d_growth_evolution_equation_dt_partial_dgrowth(
      lambda_f, lambda_r, lambda_ext, growth_scalar);
}

template <int numstates, typename T>
T Mixture::Implementation::RemodelFiberImplementation<numstates,
    T>::evaluate_d_growth_evolution_equation_dt_d_remodel(const T lambda_f, const T lambda_r,
    const T lambda_ext, const T growth_scalar) const
{
  const T dsigdremodel = evaluate_d_fiber_cauchy_stress_d_remodel(lambda_f, lambda_r, lambda_ext);
  return evaluate_d_growth_evolution_equation_dt_partial_d_remodel(
             lambda_f, lambda_r, lambda_ext, growth_scalar) +
         evaluate_d_growth_evolution_equation_dt_d_sig(
             lambda_f, lambda_r, lambda_ext, growth_scalar) *
             dsigdremodel;
}

template <int numstates, typename T>
T Mixture::Implementation::RemodelFiberImplementation<numstates,
    T>::evaluate_remodel_evolution_equation_dt(const T lambda_f, const T lambda_r,
    const T lambda_ext) const
{
  const T I4 = evaluate_i4<T>(lambda_f, lambda_r, lambda_ext);
  const T delta_sig = evaluate_fiber_cauchy_stress(lambda_f, lambda_r, lambda_ext) - sig_h_;
  const T dsigdI4 = evaluate_d_fiber_cauchy_stress_partial_d_i4(lambda_f, lambda_r, lambda_ext);

  T dlambdardt = (growth_evolution_.evaluate_true_mass_production_rate(delta_sig / sig_h_)) *
                 lambda_r * delta_sig / (2.0 * dsigdI4 * I4);


  return dlambdardt;
}

template <int numstates, typename T>
T Mixture::Implementation::RemodelFiberImplementation<numstates,
    T>::evaluate_d_remodel_evolution_equation_dt_d_sig(const T lambda_f, const T lambda_r,
    const T lambda_ext) const
{
  const T I4 = evaluate_i4<T>(lambda_f, lambda_r, lambda_ext);
  const T delta_sig = evaluate_fiber_cauchy_stress(lambda_f, lambda_r, lambda_ext) - sig_h_;
  const T dsigdI4 = evaluate_d_fiber_cauchy_stress_partial_d_i4(lambda_f, lambda_r, lambda_ext);

  return growth_evolution_.evaluate_d_true_mass_production_rate_d_sig(delta_sig / sig_h_) / sig_h_ *
             lambda_r * delta_sig / (2.0 * dsigdI4 * I4) +
         (growth_evolution_.evaluate_true_mass_production_rate(delta_sig / sig_h_)) * lambda_r /
             (2.0 * dsigdI4 * I4);
}

template <int numstates, typename T>
T Mixture::Implementation::RemodelFiberImplementation<numstates,
    T>::evaluate_d_remodel_evolution_equation_dt_d_i4(const T lambda_f, const T lambda_r,
    const T lambda_ext) const
{
  const T I4 = evaluate_i4<T>(lambda_f, lambda_r, lambda_ext);
  const T delta_sig = evaluate_fiber_cauchy_stress(lambda_f, lambda_r, lambda_ext) - sig_h_;
  const T dsigdI4 = evaluate_d_fiber_cauchy_stress_partial_d_i4(lambda_f, lambda_r, lambda_ext);
  const T dsigdI4dI4 =
      evaluate_d_fiber_cauchy_stress_partial_d_i4_d_i4(lambda_f, lambda_r, lambda_ext);

  return growth_evolution_.evaluate_d_true_mass_production_rate_d_sig(delta_sig / sig_h_) / sig_h_ *
             dsigdI4 * lambda_r * delta_sig / (2 * dsigdI4 * I4) +
         (growth_evolution_.evaluate_true_mass_production_rate(delta_sig / sig_h_)) * lambda_r *
             (1.0 / (2 * I4) -
                 delta_sig * (dsigdI4dI4 * I4 + dsigdI4) / (2 * (dsigdI4 * I4) * (dsigdI4 * I4)));
}

template <int numstates, typename T>
T Mixture::Implementation::RemodelFiberImplementation<numstates,
    T>::evaluate_d_remodel_evolution_equation_dt_partial_d_growth(const T lambda_f,
    const T lambda_r, const T lambda_ext) const
{
  if constexpr (!std::is_floating_point_v<T>)
    return T(0.0);
  else
    return 0.0;
}

template <int numstates, typename T>
T Mixture::Implementation::RemodelFiberImplementation<numstates,
    T>::evaluate_d_remodel_evolution_equation_dt_partial_d_remodel(const T lambda_f,
    const T lambda_r, const T lambda_ext) const
{
  const T I4 = evaluate_i4<T>(lambda_f, lambda_r, lambda_ext);
  const T dsigdI4 = evaluate_d_fiber_cauchy_stress_partial_d_i4(lambda_f, lambda_r, lambda_ext);
  const T delta_sig = evaluate_fiber_cauchy_stress(lambda_f, lambda_r, lambda_ext) - sig_h_;

  return (growth_evolution_.evaluate_true_mass_production_rate(delta_sig / sig_h_)) * delta_sig /
         (2.0 * dsigdI4 * I4);
}

template <int numstates, typename T>
T Mixture::Implementation::RemodelFiberImplementation<numstates,
    T>::evaluate_d_remodel_evolution_equation_dt_d_growth(const T lambda_f, const T lambda_r,
    const T lambda_ext) const
{
  return evaluate_d_remodel_evolution_equation_dt_partial_d_growth(lambda_f, lambda_r, lambda_ext);
}

template <int numstates, typename T>
T Mixture::Implementation::RemodelFiberImplementation<numstates,
    T>::evaluate_d_remodel_evolution_equation_dt_d_remodel(const T lambda_f, const T lambda_r,
    const T lambda_ext) const
{
  return evaluate_d_remodel_evolution_equation_dt_partial_d_remodel(
             lambda_f, lambda_r, lambda_ext) +
         evaluate_d_remodel_evolution_equation_dt_d_i4(lambda_f, lambda_r, lambda_ext) *
             evaluate_di4_dlambda_r(lambda_f, lambda_r, lambda_ext);
}

template <int numstates, typename T>
T Mixture::Implementation::RemodelFiberImplementation<numstates, T>::evaluate_fiber_cauchy_stress(
    const T lambda_f, const T lambda_r, const T lambda_ext) const
{
  const T I4 = evaluate_i4<T>(lambda_f, lambda_r, lambda_ext);
  return fiber_material_->get_cauchy_stress(I4);
}

template <int numstates, typename T>
T Mixture::Implementation::RemodelFiberImplementation<numstates,
    T>::evaluate_current_homeostatic_fiber_cauchy_stress() const
{
  return sig_h_;
}

template <int numstates, typename T>
T Mixture::Implementation::RemodelFiberImplementation<numstates,
    T>::evaluate_current_fiber_cauchy_stress() const
{
  FOUR_C_ASSERT(state_is_set_, "You have to call set_state() before!");
  const T lambda_f = states_.back().lambda_f;
  const T lambda_r = states_.back().lambda_r;
  const T lambda_ext = states_.back().lambda_ext;

  return evaluate_fiber_cauchy_stress(lambda_f, lambda_r, lambda_ext);
}

template <int numstates, typename T>
T Mixture::Implementation::RemodelFiberImplementation<numstates,
    T>::evaluate_current_fiber_pk2_stress() const
{
  FOUR_C_ASSERT(state_is_set_, "You have to call set_state() before!");
  const T lambda_f = states_.back().lambda_f;
  const T lambda_r = states_.back().lambda_r;
  const T lambda_ext = states_.back().lambda_ext;
  const T I4 = evaluate_i4<T>(lambda_f, lambda_r, lambda_ext);

  return fiber_material_->get_cauchy_stress(I4) / std::pow(lambda_f, 2);
}

template <int numstates, typename T>
T Mixture::Implementation::RemodelFiberImplementation<numstates,
    T>::evaluate_d_current_fiber_pk2_stress_d_lambda_f_sq() const
{
  FOUR_C_ASSERT(state_is_set_, "You have to call set_state() before!");
  const T lambda_f = states_.back().lambda_f;
  const T lambda_r = states_.back().lambda_r;
  const T lambda_ext = states_.back().lambda_ext;
  const T I4 = evaluate_i4<T>(lambda_f, lambda_r, lambda_ext);

  return (fiber_material_->get_d_cauchy_stress_d_i4(I4) * I4 -
             fiber_material_->get_cauchy_stress(I4)) /
         std::pow(lambda_f, 4);
}

template <int numstates, typename T>
T Mixture::Implementation::RemodelFiberImplementation<numstates,
    T>::evaluate_d_current_fiber_pk2_stress_d_lambda_r() const
{
  FOUR_C_ASSERT(state_is_set_, "You have to call set_state() before!");
  const T lambda_f = states_.back().lambda_f;
  const T lambda_r = states_.back().lambda_r;
  const T lambda_ext = states_.back().lambda_ext;
  const T I4 = evaluate_i4<T>(lambda_f, lambda_r, lambda_ext);

  const T dI4dlambdar = evaluate_di4_dlambda_r(lambda_f, lambda_r, lambda_ext);

  return fiber_material_->get_d_cauchy_stress_d_i4(I4) * dI4dlambdar / std::pow(lambda_f, 2);
}

template <int numstates, typename T>
T Mixture::Implementation::RemodelFiberImplementation<numstates,
    T>::evaluate_d_current_growth_evolution_implicit_time_integration_residuum_d_lambda_f_sq(T dt)
    const
{
  FOUR_C_ASSERT(state_is_set_, "You have to call set_state() before!");
  const IntegrationState<numstates, T> growth_state = get_integration_state_growth_scalar();

  const T lambda_f = states_.back().lambda_f;
  const T lambda_r = states_.back().lambda_r;
  const T lambda_ext = states_.back().lambda_ext;
  const T growth_scalar = states_.back().growth_scalar;
  const T dRgrowthdF =
      ImplicitIntegration<numstates, T>::get_partial_derivative_fnp(growth_state, dt);
  return dRgrowthdF *
         evaluate_d_growth_evolution_equation_dt_d_sig(
             lambda_f, lambda_r, lambda_ext, growth_scalar) *
         evaluate_d_fiber_cauchy_stress_partial_d_i4(lambda_f, lambda_r, lambda_ext) *
         evaluate_di4_dlambda_f_sq<T>(lambda_f, lambda_r, lambda_ext);
}

template <int numstates, typename T>
T Mixture::Implementation::RemodelFiberImplementation<numstates,
    T>::evaluate_d_current_remodel_evolution_implicit_time_integration_residuum_d_lambda_f_sq(T dt)
    const
{
  FOUR_C_ASSERT(state_is_set_, "You have to call set_state() before!");
  const IntegrationState<numstates, T> remodel_state = get_integration_state_lambda_r();

  const T dRremodeldF =
      ImplicitIntegration<numstates, T>::get_partial_derivative_fnp(remodel_state, dt);

  const T lambda_f = states_.back().lambda_f;
  const T lambda_r = states_.back().lambda_r;
  const T lambda_ext = states_.back().lambda_ext;
  return dRremodeldF *
         evaluate_d_remodel_evolution_equation_dt_d_i4(lambda_f, lambda_r, lambda_ext) *
         evaluate_di4_dlambda_f_sq<T>(lambda_f, lambda_r, lambda_ext);
}

template <int numstates, typename T>
T Mixture::Implementation::RemodelFiberImplementation<numstates,
    T>::evaluate_d_fiber_cauchy_stress_partial_d_i4(const T lambda_f, const T lambda_r,
    const T lambda_ext) const
{
  const T I4 = evaluate_i4<T>(lambda_f, lambda_r, lambda_ext);
  return fiber_material_->get_d_cauchy_stress_d_i4(I4);
}

template <int numstates, typename T>
T Mixture::Implementation::RemodelFiberImplementation<numstates,
    T>::evaluate_d_fiber_cauchy_stress_partial_d_i4_d_i4(const T lambda_f, const T lambda_r,
    const T lambda_ext) const
{
  const T I4 = evaluate_i4<T>(lambda_f, lambda_r, lambda_ext);
  return fiber_material_->get_d_cauchy_stress_d_i4_d_i4(I4);
}

template <int numstates, typename T>
T Mixture::Implementation::RemodelFiberImplementation<numstates,
    T>::evaluate_d_fiber_cauchy_stress_d_remodel(const T lambda_f, const T lambda_r,
    const T lambda_ext) const
{
  const T dI4dremodel = evaluate_di4_dlambda_r(lambda_f, lambda_r, lambda_ext);
  return evaluate_d_fiber_cauchy_stress_partial_d_i4(lambda_f, lambda_r, lambda_ext) * dI4dremodel;
}

template <int numstates, typename T>
T Mixture::Implementation::RemodelFiberImplementation<numstates,
    T>::evaluate_current_growth_scalar() const
{
  return states_.back().growth_scalar;
}

template <int numstates, typename T>
T Mixture::Implementation::RemodelFiberImplementation<numstates, T>::evaluate_current_lambda_r()
    const
{
  return states_.back().lambda_r;
}

template <int numstates, typename T>
T Mixture::Implementation::RemodelFiberImplementation<numstates,
    T>::evaluate_d_current_growth_scalar_d_lambda_f_sq() const
{
  return d_growth_scalar_d_lambda_f_sq_;
}

template <int numstates, typename T>
T Mixture::Implementation::RemodelFiberImplementation<numstates,
    T>::evaluate_d_current_lambda_r_d_lambda_f_sq() const
{
  return d_lambda_r_d_lambda_f_sq_;
}

template <int numstates, typename T>
T Mixture::Implementation::RemodelFiberImplementation<numstates,
    T>::evaluate_d_current_cauchy_stress_d_lambda_f_sq() const
{
  FOUR_C_ASSERT(state_is_set_, "You have to call set_state() before!");
  const T lambda_f = states_.back().lambda_f;
  const T lambda_r = states_.back().lambda_r;
  const T lambda_ext = states_.back().lambda_ext;
  const T I4 = evaluate_i4<T>(lambda_f, lambda_r, lambda_ext);

  const T d_i4_d_lambda_f_sq = evaluate_di4_dlambda_f_sq(lambda_f, lambda_r, lambda_ext);
  const T d_i4_d_lambda_r = evaluate_di4_dlambda_r(lambda_f, lambda_r, lambda_ext);

  return fiber_material_->get_d_cauchy_stress_d_i4(I4) *
         (d_i4_d_lambda_f_sq + d_i4_d_lambda_r * d_lambda_r_d_lambda_f_sq_);
}


//---- REMODELFIBER
template <int numstates>
Mixture::RemodelFiber<numstates>::RemodelFiber(
    std::shared_ptr<const RemodelFiberMaterial<double>> material,
    const Mixture::LinearCauchyGrowthWithPoissonTurnoverGrowthEvolution<double> growth_evolution,
    double lambda_pre)
    : impl_(std::make_unique<Implementation::RemodelFiberImplementation<numstates, double>>(
          material, growth_evolution, lambda_pre))
{
}

template <int numstates>
void Mixture::RemodelFiber<numstates>::pack(Core::Communication::PackBuffer& data) const
{
  impl_->pack(data);
}

template <int numstates>
void Mixture::RemodelFiber<numstates>::unpack(Core::Communication::UnpackBuffer& buffer)
{
  impl_->unpack(buffer);
}

template <int numstates>
void Mixture::RemodelFiber<numstates>::update()
{
  impl_->update();
}

template <int numstates>
void Mixture::RemodelFiber<numstates>::update_deposition_stretch(const double lambda_pre)
{
  impl_->update_deposition_stretch(lambda_pre);
}

template <int numstates>
void Mixture::RemodelFiber<numstates>::set_state(const double lambda_f, const double lambda_ext)
{
  impl_->set_state(lambda_f, lambda_ext);
}

template <int numstates>
Core::LinAlg::Matrix<2, 2>
Mixture::RemodelFiber<numstates>::integrate_local_evolution_equations_implicit(const double dt)
{
  return impl_->integrate_local_evolution_equations_implicit(dt);
};

template <int numstates>
void Mixture::RemodelFiber<numstates>::integrate_local_evolution_equations_explicit(const double dt)
{
  impl_->integrate_local_evolution_equations_explicit(dt);
}

template <int numstates>
double Mixture::RemodelFiber<numstates>::evaluate_current_homeostatic_fiber_cauchy_stress() const
{
  return impl_->evaluate_current_homeostatic_fiber_cauchy_stress();
}

template <int numstates>
double Mixture::RemodelFiber<numstates>::evaluate_current_fiber_cauchy_stress() const
{
  return impl_->evaluate_current_fiber_cauchy_stress();
}

template <int numstates>
double Mixture::RemodelFiber<numstates>::evaluate_current_fiber_pk2_stress() const
{
  return impl_->evaluate_current_fiber_pk2_stress();
}

template <int numstates>
double Mixture::RemodelFiber<numstates>::evaluate_d_current_fiber_pk2_stress_d_lambda_f_sq() const
{
  return impl_->evaluate_d_current_fiber_pk2_stress_d_lambda_f_sq();
};

template <int numstates>
double Mixture::RemodelFiber<numstates>::evaluate_d_current_fiber_pk2_stress_d_lambda_r() const
{
  return impl_->evaluate_d_current_fiber_pk2_stress_d_lambda_r();
};

template <int numstates>
double Mixture::RemodelFiber<numstates>::
    evaluate_d_current_growth_evolution_implicit_time_integration_residuum_d_lambda_f_sq(
        double dt) const
{
  return impl_
      ->evaluate_d_current_growth_evolution_implicit_time_integration_residuum_d_lambda_f_sq(dt);
}

template <int numstates>
double Mixture::RemodelFiber<numstates>::
    evaluate_d_current_remodel_evolution_implicit_time_integration_residuum_d_lambda_f_sq(
        double dt) const
{
  return impl_
      ->evaluate_d_current_remodel_evolution_implicit_time_integration_residuum_d_lambda_f_sq(dt);
}

template <int numstates>
double Mixture::RemodelFiber<numstates>::evaluate_current_growth_scalar() const
{
  return impl_->evaluate_current_growth_scalar();
}

template <int numstates>
double Mixture::RemodelFiber<numstates>::evaluate_current_lambda_r() const
{
  return impl_->evaluate_current_lambda_r();
}

template <int numstates>
double Mixture::RemodelFiber<numstates>::evaluate_d_current_growth_scalar_d_lambda_f_sq() const
{
  return impl_->evaluate_d_current_growth_scalar_d_lambda_f_sq();
}

template <int numstates>
double Mixture::RemodelFiber<numstates>::evaluate_d_current_lambda_r_d_lambda_f_sq() const
{
  return impl_->evaluate_d_current_lambda_r_d_lambda_f_sq();
}

template <int numstates>
double Mixture::RemodelFiber<numstates>::evaluate_d_current_cauchy_stress_d_lambda_f_sq() const
{
  return impl_->evaluate_d_current_cauchy_stress_d_lambda_f_sq();
}

template class Mixture::RemodelFiber<2>;
template class Mixture::Implementation::RemodelFiberImplementation<2, Sacado::Fad::DFad<double>>;
template class Mixture::Implementation::RemodelFiberImplementation<2, double>;

FOUR_C_NAMESPACE_CLOSE

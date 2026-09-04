// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_mat_viscoelast_fsls.hpp"
#include "4C_utils_exceptions.hpp"

#include <array>
#include <cmath>
#include <vector>

namespace
{
  using namespace FourC;
  using Mat::ViscoElast::FslsL1StepCache;
  using Mat::ViscoElast::FslsL1WeightCache;
  namespace Kernels = Mat::ViscoElast::Kernels;
  using Kernels::FslsHistory;
  using Kernels::FslsKernelInput;
  using Kernels::FslsStressVector;
  using Kernels::FslsTangentMatrix;
  using Mat::ViscoElast::FslsSolveKind;

  /// Median FSLS parameters identified for viable rat lung parenchyma by Birzle, A. M. and Wall,
  /// W. A. (2019): "A viscoelastic nonlinear compressible material model of lung parenchyma -
  /// Experiments and numerical identification", Journal of the Mechanical Behavior of Biomedical
  /// Materials, 94:164-175. Used throughout merely as a physically meaningful default parameter
  /// set; the tests below are not otherwise tied to lung mechanics.
  constexpr double kAlpha = 0.5378;
  constexpr double kTau = 0.06454;
  constexpr double kBeta = 1.856;

  /// Builds the FslsKernelInput shared by every L1-kernel test below; only stress, cmat, dt, and
  /// history vary per test.
  FslsKernelInput make_l1_input(const double dt, const FslsHistory& history,
      FslsL1WeightCache& weight_cache, FslsL1StepCache& step_cache)
  {
    FslsKernelInput input;
    input.dt = dt;
    input.gp = 0;
    input.tau = kTau;
    input.alpha = kAlpha;
    input.beta = kBeta;
    input.solve_kind = FslsSolveKind::l1;
    input.previous_history = &history;
    input.l1 = {&weight_cache, &step_cache};
    return input;
  }

  FslsStressVector make_stress_vector(const std::array<double, 6>& values)
  {
    FslsStressVector v(Core::LinAlg::Initialization::zero);
    for (int i = 0; i < 6; ++i) v(i) = values[i];
    return v;
  }

  FslsTangentMatrix make_identity_matrix()
  {
    FslsTangentMatrix m(Core::LinAlg::Initialization::zero);
    for (int i = 0; i < 6; ++i) m(i, i) = 1.0;
    return m;
  }

  /// Independent high-precision reference for K_k^h, evaluated with long double arithmetic and
  /// the direct (non-robustified) bracket subtraction. This is deliberately NOT the same code
  /// path as Kernels::evaluate_l1_weight, so agreement is a genuine cross-check rather than a
  /// tautology.
  long double reference_l1_weight(std::size_t k, double alpha_in, double tau_in, double dt_in)
  {
    const long double alpha = alpha_in;
    const long double p = 1.0L - alpha;
    const long double log_c =
        alpha * logl(static_cast<long double>(tau_in) / static_cast<long double>(dt_in)) -
        lgammal(2.0L - alpha);
    const long double c = expl(log_c);
    if (k == 0) return c;

    const long double kd = static_cast<long double>(k);
    return c * (powl(kd + 1.0L, p) - powl(kd, p));
  }

  /// Naive double-precision bracket evaluation (no expm1/log1p robustification), used to
  /// demonstrate that Kernels::evaluate_l1_weight is measurably more accurate for large k.
  double naive_l1_weight(std::size_t k, double alpha, double tau, double dt)
  {
    const double p = 1.0 - alpha;
    const double c = std::pow(tau / dt, alpha) / std::tgamma(2.0 - alpha);
    if (k == 0) return c;

    const double kd = static_cast<double>(k);
    return c * (std::pow(kd + 1.0, p) - std::pow(kd, p));
  }

  TEST(FslsL1WeightTest, MatchesHighPrecisionReferenceAcrossKRange)
  {
    const std::vector<double> alphas = {0.1, 0.5378, 0.9};
    const std::vector<std::size_t> ks = {0, 1, 2, 10, 1000, 100000};

    for (const double alpha : alphas)
    {
      for (const std::size_t k : ks)
      {
        const double actual = Kernels::evaluate_l1_weight(k, alpha, kTau, 0.01);
        const long double reference = reference_l1_weight(k, alpha, kTau, 0.01);

        const long double relative_error =
            std::abs(static_cast<long double>(actual) - reference) / std::abs(reference);
        EXPECT_LT(relative_error, 1.0e-7) << "alpha=" << alpha << " k=" << k << " actual=" << actual
                                          << " reference=" << static_cast<double>(reference);
      }
    }
  }

  TEST(FslsL1WeightTest, RobustFormulaBeatsNaiveSubtractionForLargeK)
  {
    // p = 1 - alpha close to 1 maximizes the cancellation in (k+1)^p - k^p for large k.
    constexpr double alpha = 0.05;
    constexpr std::size_t k = 100000000;  // 1e8

    const long double reference = reference_l1_weight(k, alpha, kTau, 0.01);
    const double robust = Kernels::evaluate_l1_weight(k, alpha, kTau, 0.01);
    const double naive = naive_l1_weight(k, alpha, kTau, 0.01);

    const long double robust_error =
        std::abs(static_cast<long double>(robust) - reference) / std::abs(reference);
    const long double naive_error =
        std::abs(static_cast<long double>(naive) - reference) / std::abs(reference);

    EXPECT_LT(robust_error, 1.0e-9);
    EXPECT_GT(naive_error, robust_error * 10.0)
        << "expected naive subtraction to lose measurably more precision than the robust "
           "expm1/log1p formula at k="
        << k;
  }

  TEST(FslsL1WeightTest, ZeroAlphaGivesUnitWeightsIndependentOfExtremeTimeStepRatio)
  {
    // alpha == 0 is special-cased to avoid 0 * log(tau/dt) producing NaN when tau/dt is extreme.
    EXPECT_DOUBLE_EQ(Kernels::evaluate_l1_weight(0, 0.0, kTau, 1.0e-12), 1.0);
    EXPECT_DOUBLE_EQ(Kernels::evaluate_l1_weight(5, 0.0, kTau, 1.0e-12), 1.0);
    EXPECT_DOUBLE_EQ(Kernels::evaluate_l1_weight(5, 0.0, kTau, 1.0e12), 1.0);
  }

  TEST(FslsL1WeightTest, RemainsFiniteForExtremeTimeStepRatios)
  {
    for (const double dt : {1.0e-10, 1.0e10})
    {
      for (const double alpha : {1.0e-6, 0.5, 1.0 - 1.0e-6})
      {
        const double weight = Kernels::evaluate_l1_weight(0, alpha, kTau, dt);
        EXPECT_TRUE(std::isfinite(weight)) << "alpha=" << alpha << " dt=" << dt;
        EXPECT_GT(weight, 0.0);
      }
    }
  }

  /// Builds a small synthetic FSLS history Q_0..Q_n (deterministic, not physically meaningful)
  /// used to exercise the L1 kernel independently of any finite-element context.
  FslsHistory make_synthetic_history(const std::size_t n)
  {
    FslsHistory history(1);
    auto& at_gp = history[0];
    for (std::size_t i = 0; i <= n; ++i)
    {
      std::array<double, 6> values{};
      for (int c = 0; c < 6; ++c)
        values[static_cast<std::size_t>(c)] = 0.1 * static_cast<double>(i + 1) * (c + 1);
      at_gp.push_back(make_stress_vector(values));
    }
    return history;
  }

  TEST(FslsL1KernelTest, ClosedFormRoundTripsExactTargetArtificialStress)
  {
    constexpr std::size_t n = 5;
    constexpr double dt = 0.01;
    FslsHistory history = make_synthetic_history(n);
    const auto& at_gp = history[0];

    // Independently compute r_n^Q = -K_0*Q_n + sum_{k=1}^n K_k*(Q_{n-k+1}-Q_{n-k}) using the pure
    // weight function, deliberately NOT going through the kernel's internal cache.
    const double k0 = Kernels::evaluate_l1_weight(0, kAlpha, kTau, dt);
    FslsStressVector r_q(at_gp[n]);
    r_q.scale(-k0);
    for (std::size_t k = 1; k <= n; ++k)
    {
      const double kk = Kernels::evaluate_l1_weight(k, kAlpha, kTau, dt);
      FslsStressVector delta(Core::LinAlg::Initialization::zero);
      delta.update(1.0, at_gp[n - k + 1], -1.0, at_gp[n - k]);
      r_q.update(kk, delta, 1.0);
    }

    const FslsStressVector q_target = make_stress_vector({2.0, 2.3, 2.6, 2.9, 3.2, 3.5});

    // Invert eq. (14): S_inf = ((1+K0)*Q_target + r_n^Q) / beta.
    FslsStressVector stress(Core::LinAlg::Initialization::zero);
    stress.update(1.0 + k0, q_target, 1.0, r_q);
    stress.scale(1.0 / kBeta);

    FslsTangentMatrix cmat = make_identity_matrix();
    FslsStressVector q_current(Core::LinAlg::Initialization::zero);
    FslsStressVector q_additive(Core::LinAlg::Initialization::zero);
    FslsTangentMatrix cmatq_additive(Core::LinAlg::Initialization::zero);

    FslsL1WeightCache cache;
    FslsL1StepCache step_cache;
    const FslsKernelInput input = make_l1_input(dt, history, cache, step_cache);

    Kernels::evaluate_fsls_kernel(stress, cmat, q_current, q_additive, cmatq_additive, input);

    for (int i = 0; i < 6; ++i) EXPECT_NEAR(q_current(i), q_target(i), 1.0e-9);

    // S_eff_additive = beta * S_inf - Q, and the additive tangent factor is beta*K0/(1+K0).
    for (int i = 0; i < 6; ++i) EXPECT_NEAR(q_additive(i), kBeta * stress(i) - q_target(i), 1.0e-9);

    const double expected_tangent_factor = kBeta * k0 / (1.0 + k0);
    for (int i = 0; i < 6; ++i)
      for (int j = 0; j < 6; ++j)
        EXPECT_NEAR(cmatq_additive(i, j), (i == j ? expected_tangent_factor : 0.0), 1.0e-9);
  }

  TEST(FslsL1KernelTest, StepCacheReusesHistoryRemainderWhenHistorySizeIsUnchanged)
  {
    constexpr std::size_t n = 5;
    constexpr double dt = 0.01;
    FslsHistory history = make_synthetic_history(n);

    const FslsStressVector stress = make_stress_vector({1.0, 0.5, -0.2, 0.3, 0.1, -0.4});
    const FslsTangentMatrix cmat = make_identity_matrix();

    const auto evaluate =
        [&](FslsHistory& hist, FslsL1WeightCache& weight_cache, FslsL1StepCache& step_cache)
    {
      const FslsKernelInput input = make_l1_input(dt, hist, weight_cache, step_cache);

      FslsStressVector q_current(Core::LinAlg::Initialization::zero);
      FslsStressVector q_additive(Core::LinAlg::Initialization::zero);
      FslsTangentMatrix cmatq_additive(Core::LinAlg::Initialization::zero);
      Kernels::evaluate_fsls_kernel(stress, cmat, q_current, q_additive, cmatq_additive, input);
      return q_current;
    };

    FslsL1WeightCache weight_cache;
    FslsL1StepCache step_cache;
    const FslsStressVector q_first = evaluate(history, weight_cache, step_cache);

    // Corrupt an already-committed history entry in place, without changing the history size.
    // If r_n^Q were reused from cache rather than recomputed, this must not affect the result.
    FslsHistory corrupted_history = history;
    corrupted_history[0][0](0) += 1000.0;

    const FslsStressVector q_second = evaluate(corrupted_history, weight_cache, step_cache);
    for (int i = 0; i < 6; ++i) EXPECT_DOUBLE_EQ(q_second(i), q_first(i));

    // Sanity check: evaluating the corrupted history fresh (no reused cache) does give a
    // different answer, confirming the equality above is genuinely due to caching and not a
    // coincidence (e.g. the perturbed entry not mattering).
    FslsL1WeightCache fresh_weight_cache;
    FslsL1StepCache fresh_step_cache;
    const FslsStressVector q_fresh_from_corrupted =
        evaluate(corrupted_history, fresh_weight_cache, fresh_step_cache);
    bool any_component_differs = false;
    for (int i = 0; i < 6; ++i)
      if (std::abs(q_fresh_from_corrupted(i) - q_first(i)) > 1.0e-6) any_component_differs = true;
    EXPECT_TRUE(any_component_differs);
  }

  TEST(FslsL1KernelTest, StepCacheRecomputesAfterHistoryGrows)
  {
    constexpr std::size_t n = 4;
    constexpr double dt = 0.01;
    FslsHistory history = make_synthetic_history(n);

    const FslsStressVector stress = make_stress_vector({0.7, -0.3, 0.2, 0.1, -0.5, 0.4});
    const FslsTangentMatrix cmat = make_identity_matrix();

    const auto evaluate = [&](FslsL1WeightCache& weight_cache, FslsL1StepCache& step_cache)
    {
      const FslsKernelInput input = make_l1_input(dt, history, weight_cache, step_cache);

      FslsStressVector q_current(Core::LinAlg::Initialization::zero);
      FslsStressVector q_additive(Core::LinAlg::Initialization::zero);
      FslsTangentMatrix cmatq_additive(Core::LinAlg::Initialization::zero);
      Kernels::evaluate_fsls_kernel(stress, cmat, q_current, q_additive, cmatq_additive, input);
      return q_current;
    };

    FslsL1WeightCache weight_cache;
    FslsL1StepCache step_cache;
    const FslsStressVector q_before_growth = evaluate(weight_cache, step_cache);
    (void)q_before_growth;

    // Grow the history by exactly one committed entry, as a real time step would.
    history[0].push_back(make_stress_vector({4.1, 4.2, 4.3, 4.4, 4.5, 4.6}));

    // Reuse the now-stale step cache: it must detect the size change and recompute.
    const FslsStressVector q_after_growth_reused_cache = evaluate(weight_cache, step_cache);

    // Independently evaluate the grown history with a completely fresh step cache.
    FslsL1WeightCache independent_weight_cache;
    FslsL1StepCache independent_step_cache;
    const FslsStressVector q_after_growth_fresh_cache =
        evaluate(independent_weight_cache, independent_step_cache);

    for (int i = 0; i < 6; ++i)
      EXPECT_DOUBLE_EQ(q_after_growth_reused_cache(i), q_after_growth_fresh_cache(i));
  }

  TEST(FslsL1KernelTest, ConsistentTangentMatchesCentralFiniteDifference)
  {
    constexpr std::size_t n = 4;
    constexpr double dt = 0.02;
    FslsHistory history = make_synthetic_history(n);

    const FslsStressVector stress0 = make_stress_vector({1.0, -0.5, 0.3, 0.2, -0.1, 0.4});
    const FslsTangentMatrix cmat = make_identity_matrix();

    FslsL1WeightCache cache;
    FslsL1StepCache step_cache;
    auto evaluate = [&](const FslsStressVector& stress)
    {
      const FslsKernelInput input = make_l1_input(dt, history, cache, step_cache);

      FslsStressVector q_current(Core::LinAlg::Initialization::zero);
      FslsStressVector q_additive(Core::LinAlg::Initialization::zero);
      FslsTangentMatrix cmatq_additive(Core::LinAlg::Initialization::zero);
      Kernels::evaluate_fsls_kernel(stress, cmat, q_current, q_additive, cmatq_additive, input);
      return q_additive;
    };

    const FslsKernelInput baseline_input = make_l1_input(dt, history, cache, step_cache);
    FslsStressVector q_current(Core::LinAlg::Initialization::zero);
    FslsStressVector q_additive0(Core::LinAlg::Initialization::zero);
    FslsTangentMatrix cmatq_additive(Core::LinAlg::Initialization::zero);
    Kernels::evaluate_fsls_kernel(
        stress0, cmat, q_current, q_additive0, cmatq_additive, baseline_input);

    constexpr double step = 1.0e-6;
    for (int i = 0; i < 6; ++i)
    {
      FslsStressVector stress_plus(stress0);
      FslsStressVector stress_minus(stress0);
      stress_plus(i) += step;
      stress_minus(i) -= step;

      const FslsStressVector q_additive_plus = evaluate(stress_plus);
      const FslsStressVector q_additive_minus = evaluate(stress_minus);

      for (int j = 0; j < 6; ++j)
      {
        const double numerical_derivative =
            (q_additive_plus(j) - q_additive_minus(j)) / (2.0 * step);
        EXPECT_NEAR(numerical_derivative, cmatq_additive(j, i), 1.0e-8)
            << "row=" << j << " col=" << i;
      }
    }
  }

  /// Analytic Mittag-Leffler relaxation solution for a constant equilibrium stress applied at
  /// t=0+ with zero initial memory: Qhat(t) = beta*S0*(1 - E_alpha(-(t/tau)^alpha)).
  double mittag_leffler_e_alpha(const double x, const double alpha)
  {
    double sum = 0.0;
    double term = 1.0;  // x^0 / Gamma(1) for k = 0
    for (int k = 0; k < 200; ++k)
    {
      sum += term;
      term *= x / std::tgamma(alpha * (k + 1) + 1.0) * std::tgamma(alpha * k + 1.0);
      if (std::abs(term) < 1.0e-18 && k > 5) break;
    }
    return sum;
  }

  TEST(FslsL1KernelTest, RelaxationConvergesTowardMittagLefflerSolutionAsStepShrinks)
  {
    constexpr double s0 = 1.0;
    constexpr double target_time = 0.1;  // ~1.5*tau
    const double analytic_q =
        kBeta * s0 * (1.0 - mittag_leffler_e_alpha(-std::pow(target_time / kTau, kAlpha), kAlpha));

    const auto run_to_target_time = [&](const int num_steps)
    {
      const double dt = target_time / num_steps;
      FslsHistory history(1);
      history[0].push_back(FslsStressVector(Core::LinAlg::Initialization::zero));  // Q_0 = 0

      FslsL1WeightCache cache;
      FslsL1StepCache step_cache;
      const FslsStressVector stress = make_stress_vector({s0, 0, 0, 0, 0, 0});
      const FslsTangentMatrix cmat = make_identity_matrix();

      FslsStressVector q_current(Core::LinAlg::Initialization::zero);
      for (int step = 0; step < num_steps; ++step)
      {
        const FslsKernelInput input = make_l1_input(dt, history, cache, step_cache);

        FslsStressVector q_additive(Core::LinAlg::Initialization::zero);
        FslsTangentMatrix cmatq_additive(Core::LinAlg::Initialization::zero);
        Kernels::evaluate_fsls_kernel(stress, cmat, q_current, q_additive, cmatq_additive, input);
        history[0].push_back(q_current);
      }
      return q_current(0);
    };

    const double error_coarse = std::abs(run_to_target_time(20) - analytic_q);
    const double error_fine = std::abs(run_to_target_time(200) - analytic_q);
    const double error_finer = std::abs(run_to_target_time(2000) - analytic_q);

    EXPECT_LT(error_fine, error_coarse);
    EXPECT_LT(error_finer, error_fine);
    EXPECT_LT(error_finer, 1.0e-3 * std::abs(analytic_q));
  }

  TEST(FslsL1KernelTest, LongHistoryUnderOscillatingLoadStaysBounded)
  {
    constexpr double dt = 0.01;
    constexpr int num_steps = 3000;

    FslsHistory history(1);
    history[0].push_back(FslsStressVector(Core::LinAlg::Initialization::zero));

    FslsL1WeightCache cache;
    FslsL1StepCache step_cache;
    const FslsTangentMatrix cmat = make_identity_matrix();

    FslsStressVector q_current(Core::LinAlg::Initialization::zero);
    for (int step = 0; step < num_steps; ++step)
    {
      const double amplitude = std::sin(0.3 * step);
      const FslsStressVector stress = make_stress_vector({amplitude, 0, 0, 0, 0, 0});

      const FslsKernelInput input = make_l1_input(dt, history, cache, step_cache);

      FslsStressVector q_additive(Core::LinAlg::Initialization::zero);
      FslsTangentMatrix cmatq_additive(Core::LinAlg::Initialization::zero);
      Kernels::evaluate_fsls_kernel(stress, cmat, q_current, q_additive, cmatq_additive, input);

      ASSERT_TRUE(std::isfinite(q_current(0))) << "step=" << step;
      // Q is a fractionally-damped response to a unit-amplitude oscillating drive; it must stay
      // within a small bounded multiple of beta*|S_inf|_max, never grow unboundedly.
      ASSERT_LT(std::abs(q_current(0)), 10.0 * kBeta) << "step=" << step;

      history[0].push_back(q_current);
    }
  }

  TEST(FslsL1KernelTest, RejectsTimeStepChangeAfterCacheIsEstablished)
  {
    FslsHistory history = make_synthetic_history(2);
    FslsL1WeightCache cache;
    FslsL1StepCache step_cache;
    const FslsStressVector stress = make_stress_vector({1, 0, 0, 0, 0, 0});
    const FslsTangentMatrix cmat = make_identity_matrix();

    auto run_with_dt = [&](const double dt)
    {
      const FslsKernelInput input = make_l1_input(dt, history, cache, step_cache);

      FslsStressVector q_current(Core::LinAlg::Initialization::zero);
      FslsStressVector q_additive(Core::LinAlg::Initialization::zero);
      FslsTangentMatrix cmatq_additive(Core::LinAlg::Initialization::zero);
      Kernels::evaluate_fsls_kernel(stress, cmat, q_current, q_additive, cmatq_additive, input);
    };

    run_with_dt(0.01);
    EXPECT_THROW(run_with_dt(0.02), Core::Exception);
    // Re-evaluating with the original, still-consistent dt must keep working.
    EXPECT_NO_THROW(run_with_dt(0.01));
  }
}  // namespace

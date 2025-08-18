// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_mixture_full_constrained_mixture_fiber.hpp"

#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_material_parameter_base.hpp"
#include "4C_mixture_constituent_remodelfiber_material_exponential.hpp"
#include "4C_mixture_growth_evolution_linear_cauchy_poisson_turnover.hpp"
#include "4C_mixture_remodelfiber-internal.hpp"

#include <Sacado.hpp>

#include <memory>

namespace
{
  using namespace FourC;
  using FADdouble = Sacado::Fad::DFad<double>;

  class FullConstrainedMixtureFiberTest : public ::testing::Test
  {
   protected:
    template <typename Number>
    const Mixture::PAR::RemodelFiberMaterialExponential<Number>* create_material()
    {
      Core::IO::InputParameterContainer container;
      container.add("K1", 1.3);
      container.add("K2", 1.3);
      container.add("COMPRESSION", true);
      static auto params = std::make_shared<Mixture::PAR::RemodelFiberMaterialExponential<Number>>(
          Core::Mat::PAR::Parameter::Data{.parameters = container});

      return params.get();
    }

    template <typename Number>
    Mixture::FullConstrainedMixtureFiber<Number> generate_fiber(const double decay_time = 12.0,
        const double growth_constant = 0.1, const Number lambda_pre = 1.1,
        const bool enable_growth = true,
        const Mixture::HistoryAdaptionStrategy adaptive_strategy =
            Mixture::HistoryAdaptionStrategy::none)
    {
      return {std::make_shared<Mixture::RemodelFiberMaterialExponential<Number>>(
                  create_material<Number>()),
          {growth_constant, decay_time, true}, lambda_pre, adaptive_strategy, enable_growth};
    }

    Mixture::Implementation::RemodelFiberImplementation<2, double> generate_remodel_fiber(
        const double decay_time = 12.0, const double growth_constant = 0.1,
        const double lambda_pre = 1.1)
    {
      return {std::make_unique<const Mixture::RemodelFiberMaterialExponential<double>>(
                  create_material<double>()),
          {growth_constant, decay_time, true}, lambda_pre};
    }
  };

  TEST_F(FullConstrainedMixtureFiberTest, TestThrowsIfHistoryIsEmpty)
  {
#ifndef FOUR_C_ENABLE_ASSERTIONS
    GTEST_SKIP() << "Skip debug assertion tests in release mode.";
#endif
    Mixture::FullConstrainedMixtureFiber fiber = generate_fiber<double>();

    const double lambda_f = 1.0;

    ASSERT_ANY_THROW(fiber.recompute_state(lambda_f, 0.0, 1.0));
  }

  TEST_F(FullConstrainedMixtureFiberTest, TestDoesNotThrowIfHistoryIsEmptyButGrowthIsDisabled)
  {
    Mixture::FullConstrainedMixtureFiber fiber = generate_fiber<double>(12.0, 0.1, 1.1, false);

    const double lambda_f = 1.0;

    fiber.recompute_state(lambda_f, 0.0, 1.0);
  }

  TEST_F(FullConstrainedMixtureFiberTest, TestEvaluateDGrowthEvolutionEquationDtDGrowth)
  {
    Mixture::FullConstrainedMixtureFiber fiber = generate_fiber<double>();
    const double lambda_f = 1.0;
    fiber.reinitialize_history(lambda_f, 0.0);

    fiber.recompute_state(lambda_f, 0.0, 1.0);

    ASSERT_NEAR(fiber.computed_sigma_, fiber.sig_h_, 1e-18);
  }

  TEST_F(FullConstrainedMixtureFiberTest, TestFiberMaintenanceAfterOneStep)
  {
    Mixture::FullConstrainedMixtureFiber fiber = generate_fiber<double>();
    const double lambda_f = 1.0;
    fiber.reinitialize_history(lambda_f, 0.0);

    fiber.recompute_state(lambda_f, 0.01, 0.01);

    EXPECT_NEAR(fiber.computed_growth_scalar_, 1.0, 1e-9);
    EXPECT_NEAR(fiber.computed_sigma_, fiber.sig_h_, 1e-9);
  }

  TEST_F(FullConstrainedMixtureFiberTest, TestFiberMaintenanceTwoSteps)
  {
    Mixture::FullConstrainedMixtureFiber fiber = generate_fiber<double>();
    const double lambda_f = 1.0;
    fiber.reinitialize_history(lambda_f, 0.0);

    for (unsigned timestep = 1; timestep <= 2; ++timestep)
    {
      fiber.recompute_state(lambda_f, timestep * 0.01, 0.01);
      fiber.update();
    }

    fiber.recompute_state(lambda_f, 0.03, 0.01);

    EXPECT_NEAR(fiber.computed_growth_scalar_, 1.0, 1e-9);
    EXPECT_NEAR(fiber.computed_sigma_, fiber.sig_h_, 1e-9);
  }

  TEST_F(FullConstrainedMixtureFiberTest, TestFiberMaintenanceNSteps)
  {
    Mixture::FullConstrainedMixtureFiber fiber = generate_fiber<double>();
    const double lambda_f = 1.0;
    fiber.reinitialize_history(lambda_f, 0.0);

    for (unsigned timestep = 1; timestep <= 3; ++timestep)
    {
      fiber.recompute_state(lambda_f, timestep * 0.01, 0.01);
      fiber.update();
    }

    fiber.recompute_state(lambda_f, 0.04, 0.01);

    EXPECT_NEAR(fiber.computed_growth_scalar_, 1.0, 1e-9);
    EXPECT_NEAR(fiber.computed_sigma_, fiber.sig_h_, 1e-9);
  }

  TEST_F(FullConstrainedMixtureFiberTest, CheckDerivativeGrowthScalarIntegrand)
  {
    Mixture::FullConstrainedMixtureFiber cm_fiber = generate_fiber<FADdouble>();
    const Mixture::MassIncrement<FADdouble> mass_increment{
        1.01, FADdouble(2, 0, 1.12), FADdouble(2, 1, 1.12), 1.0};

    const double time = 1.4;
    const FADdouble growth_scalar_integrand =
        cm_fiber.growth_scalar_integrand(mass_increment, time);

    EXPECT_FLOAT_EQ(growth_scalar_integrand.dx(0),
        cm_fiber.d_growth_scalar_integrand_d_growth_scalar(mass_increment, time).val());

    EXPECT_FLOAT_EQ(growth_scalar_integrand.dx(1),
        cm_fiber.d_growth_scalar_integrand_d_production_rate(mass_increment, time).val());
  }

  TEST_F(FullConstrainedMixtureFiberTest, CheckDerivativeScaledCauchyStressIntegrand)
  {
    Mixture::FullConstrainedMixtureFiber cm_fiber = generate_fiber<FADdouble>();

    const double time = 1.4;
    const FADdouble lambda_f = 1.012;
    cm_fiber.reinitialize_history(lambda_f, time);
    cm_fiber.recompute_state(lambda_f, time, 0.4);
    const Mixture::MassIncrement<FADdouble> mass_increment{
        1.01, FADdouble(2, 0, 1.12), FADdouble(2, 1, 1.12), 1.0};

    const FADdouble scaled_cauchy_stress_integrand =
        cm_fiber.scaled_cauchy_stress_integrand(mass_increment, time, lambda_f);

    EXPECT_FLOAT_EQ(scaled_cauchy_stress_integrand.dx(0),
        cm_fiber.d_scaled_cauchy_stress_integrand_d_growth_scalar(mass_increment, time, lambda_f)
            .val());

    EXPECT_FLOAT_EQ(scaled_cauchy_stress_integrand.dx(1),
        cm_fiber.d_scaled_cauchy_stress_integrand_d_production_rate(mass_increment, time, lambda_f)
            .val());
  }

  TEST_F(FullConstrainedMixtureFiberTest, RemodelFiberComparisonCauchyStress)
  {
    // compare the results of the full constrained mixture model with the homogenized constrained
    // mixture model
    Mixture::FullConstrainedMixtureFiber cm_fiber = generate_fiber<double>();
    Mixture::Implementation::RemodelFiberImplementation<2, double> remodel_fiber =
        generate_remodel_fiber();

    const double lambda_f = 1.05;
    cm_fiber.reinitialize_history(lambda_f, 0.0);

    const double dt = 0.1;
    for (unsigned timestep = 1; timestep <= 1000; ++timestep)
    {
      const double time = timestep * dt;
      cm_fiber.recompute_state(lambda_f, time, dt);
      remodel_fiber.set_state(lambda_f, 1.0);

      remodel_fiber.integrate_local_evolution_equations_implicit(dt);

      // compare
      ASSERT_NEAR(
          cm_fiber.sig_h_, remodel_fiber.evaluate_current_homeostatic_fiber_cauchy_stress(), 1e-2);
      ASSERT_NEAR(cm_fiber.computed_sigma_ / cm_fiber.sig_h_,
          remodel_fiber.evaluate_current_fiber_cauchy_stress() /
              remodel_fiber.evaluate_current_homeostatic_fiber_cauchy_stress(),
          1e-2);
      ASSERT_NEAR(
          cm_fiber.computed_growth_scalar_, remodel_fiber.evaluate_current_growth_scalar(), 1e-2);

      cm_fiber.update();
      remodel_fiber.update();
    }
  }

  TEST_F(FullConstrainedMixtureFiberTest, RemodelFiberComparison2PKStress)
  {
    // compare the results of the full constrained mixture model with the homogenized constrained
    // mixture model
    Mixture::FullConstrainedMixtureFiber cm_fiber = generate_fiber<double>();
    Mixture::Implementation::RemodelFiberImplementation<2, double> remodel_fiber =
        generate_remodel_fiber();

    const double lambda_f = 1.05;
    cm_fiber.reinitialize_history(lambda_f, 0.0);

    const double dt = 0.1;
    for (unsigned timestep = 1; timestep <= 1000; ++timestep)
    {
      const double time = timestep * dt;
      cm_fiber.recompute_state(lambda_f, time, dt);

      remodel_fiber.set_state(lambda_f, 1.0);
      remodel_fiber.integrate_local_evolution_equations_implicit(dt);

      // compare
      ASSERT_NEAR(cm_fiber.evaluate_current_second_pk_stress(),
          remodel_fiber.evaluate_current_fiber_pk2_stress(), 1e-2);

      cm_fiber.update();
      remodel_fiber.update();
    }
  }

  TEST_F(FullConstrainedMixtureFiberTest, AdaptiveHistoryToleranceComparisonModelEquation)
  {
    // compare the results of the fully integrated constrained mixture fiber with the adaptive
    // integrated fiber
    Mixture::FullConstrainedMixtureFiber cm_fiber = generate_fiber<double>();
    Mixture::FullConstrainedMixtureFiber cm_fiber_adaptive = generate_fiber<double>(
        12.0, 0.1, 1.1, true, Mixture::HistoryAdaptionStrategy::model_equation);
    cm_fiber_adaptive.adaptive_tolerance_ = 1e-7;

    const double lambda_f = 1.05;
    cm_fiber.reinitialize_history(lambda_f, 0.0);
    cm_fiber_adaptive.reinitialize_history(lambda_f, 0.0);

    const double dt = 0.1;
    for (unsigned timestep = 1; timestep <= 1000; ++timestep)
    {
      const double time = timestep * dt;
      cm_fiber.recompute_state(lambda_f, time, dt);
      cm_fiber_adaptive.recompute_state(lambda_f, time, dt);

      // compare
      EXPECT_NEAR(cm_fiber.evaluate_current_second_pk_stress(),
          cm_fiber_adaptive.evaluate_current_second_pk_stress(), 1e-7);

      cm_fiber.update();
      cm_fiber_adaptive.update();
    }

    EXPECT_LE(cm_fiber_adaptive.history_[0].timesteps.size(),
        0.25 * cm_fiber.history_[0].timesteps.size());
  }

  TEST_F(FullConstrainedMixtureFiberTest, AdaptiveHistoryToleranceComparisonHigherOrder)
  {
    // compare the results of the fully integrated constrained mixture fiber with the adaptive
    // integrated fiber
    Mixture::FullConstrainedMixtureFiber cm_fiber = generate_fiber<double>();
    Mixture::FullConstrainedMixtureFiber cm_fiber_adaptive = generate_fiber<double>(
        12.0, 0.1, 1.1, true, Mixture::HistoryAdaptionStrategy::higher_order_integration);
    cm_fiber_adaptive.adaptive_tolerance_ = 1e-7;

    const double lambda_f = 1.05;
    cm_fiber.reinitialize_history(lambda_f, 0.0);
    cm_fiber_adaptive.reinitialize_history(lambda_f, 0.0);

    const double dt = 0.1;
    for (unsigned timestep = 1; timestep <= 1000; ++timestep)
    {
      const double time = timestep * dt;
      cm_fiber.recompute_state(lambda_f, time, dt);
      cm_fiber_adaptive.recompute_state(lambda_f, time, dt);

      // compare
      EXPECT_NEAR(cm_fiber.evaluate_current_second_pk_stress(),
          cm_fiber_adaptive.evaluate_current_second_pk_stress(), 1e-7);

      cm_fiber.update();
      cm_fiber_adaptive.update();
    }

    EXPECT_LE(cm_fiber_adaptive.history_[0].timesteps.size(),
        0.25 * cm_fiber.history_[0].timesteps.size());
  }

  TEST_F(FullConstrainedMixtureFiberTest, AdaptiveHistoryToleranceComparisonWindow)
  {
    // compare the results of the fully integrated constrained mixture fiber with the adaptive
    // integrated fiber
    Mixture::FullConstrainedMixtureFiber cm_fiber = generate_fiber<double>(1.2, 0.1, 1.1);
    Mixture::FullConstrainedMixtureFiber cm_fiber_adaptive =
        generate_fiber<double>(1.2, 0.1, 1.1, true, Mixture::HistoryAdaptionStrategy::window);
    cm_fiber_adaptive.window_size = 500;

    const double lambda_f = 1.05;
    cm_fiber.reinitialize_history(lambda_f, 0.0);
    cm_fiber_adaptive.reinitialize_history(lambda_f, 0.0);

    const double dt = 0.1;
    for (unsigned timestep = 1; timestep <= 1000; ++timestep)
    {
      const double time = timestep * dt;
      cm_fiber.recompute_state(lambda_f, time, dt);
      cm_fiber_adaptive.recompute_state(lambda_f, time, dt);

      // compare
      EXPECT_NEAR(cm_fiber.computed_sigma_, cm_fiber_adaptive.computed_sigma_, 1e-9);
      EXPECT_NEAR(
          cm_fiber.computed_growth_scalar_, cm_fiber_adaptive.computed_growth_scalar_, 1e-4);

      cm_fiber.update();
      cm_fiber_adaptive.update();
    }

    EXPECT_EQ(cm_fiber_adaptive.history_[0].timesteps.size(), cm_fiber_adaptive.window_size);
  }

  TEST_F(FullConstrainedMixtureFiberTest, LocalNewtonHasAnalyticalDerivativeInFirstTimestep)
  {
    Mixture::FullConstrainedMixtureFiber cm_fiber = generate_fiber<FADdouble>();

    const double lambda_f = 1.05;
    cm_fiber.reinitialize_history(lambda_f, 0.0);
    cm_fiber.recompute_state(1.06, 0.01, 0.01);

    // evaluate local newton
    const auto local_newton_evaluator = cm_fiber.get_local_newton_evaluator();

    Core::LinAlg::Matrix<2, 1, FADdouble> x;
    x(0) = FADdouble(2, 0, 1.5);
    x(1) = FADdouble(2, 1, 0.453);

    const auto [residuum, derivative] = local_newton_evaluator(x);

    EXPECT_NEAR(residuum(0).dx(0), derivative(0, 0).val(), 1e-9);
    EXPECT_NEAR(residuum(0).dx(1), derivative(0, 1).val(), 1e-9);
    EXPECT_NEAR(residuum(1).dx(0), derivative(1, 0).val(), 1e-9);
    EXPECT_NEAR(residuum(1).dx(1), derivative(1, 1).val(), 1e-9);
  }

  TEST_F(FullConstrainedMixtureFiberTest, LocalNewtonHasAnalyticalDerivativeInFirstThreeTimesteps)
  {
    Mixture::FullConstrainedMixtureFiber cm_fiber = generate_fiber<FADdouble>();

    const double lambda_f = 1.05;
    cm_fiber.reinitialize_history(lambda_f, 0.0);
    const double dt = 0.1;
    for (unsigned timestep = 1; timestep <= 3; ++timestep)
    {
      const double time = timestep * dt;
      cm_fiber.recompute_state(lambda_f, time, dt);

      // evaluate local newton
      const auto local_newton_evaluator = cm_fiber.get_local_newton_evaluator();

      Core::LinAlg::Matrix<2, 1, FADdouble> x;
      x(0) = FADdouble(2, 0, cm_fiber.computed_growth_scalar_.val());
      x(1) = FADdouble(2, 1, cm_fiber.computed_sigma_.val());

      const auto [residuum, derivative] = local_newton_evaluator(x);

      EXPECT_NEAR(residuum(0).dx(0), derivative(0, 0).val(), 1e-9);
      EXPECT_NEAR(residuum(0).dx(1), derivative(0, 1).val(), 1e-9);
      EXPECT_NEAR(residuum(1).dx(0), derivative(1, 0).val(), 1e-9);
      EXPECT_NEAR(residuum(1).dx(1), derivative(1, 1).val(), 1e-9);

      cm_fiber.update();
    }
  }

  TEST_F(FullConstrainedMixtureFiberTest, d_scaled_cauchy_stress_integrand_d_lambda_f_sq)
  {
    Mixture::FullConstrainedMixtureFiber cm_fiber = generate_fiber<FADdouble>();

    const FADdouble lambda_f = FADdouble(1, 0, 1.2);
    const double time = 1.0;
    cm_fiber.reinitialize_history(lambda_f.val(), 0.0);

    cm_fiber.recompute_state(lambda_f, time, 1.0);
    Mixture::MassIncrement<FADdouble> current_increment = {1.1, 1.4, 0.9, 1.22};

    FADdouble scaled_cauchy_stress_integrand =
        cm_fiber.scaled_cauchy_stress_integrand(current_increment, time, lambda_f);

    FADdouble dscaled_cauchy_stress_integrandDLambdaFSq =
        cm_fiber.d_scaled_cauchy_stress_integrand_d_lambda_f_sq(current_increment, time, lambda_f);

    EXPECT_DOUBLE_EQ(dscaled_cauchy_stress_integrandDLambdaFSq.val() * 2 * lambda_f.val(),
        scaled_cauchy_stress_integrand.dx(0));
  }



  TEST_F(FullConstrainedMixtureFiberTest, DerivativeOfResiduumWithRespectToLambdaF)
  {
    Mixture::FullConstrainedMixtureFiber cm_fiber = generate_fiber<FADdouble>(800.0, 0.1, 1.1);

    const double lambda_f_0 = 1.05;
    cm_fiber.reinitialize_history(lambda_f_0, 0.0);

    const double lambda_f = 1.06;
    cm_fiber.recompute_state(FADdouble(1, 0, lambda_f), 0.01, 0.01);

    // evaluate local newton
    const auto local_newton_evaluator = cm_fiber.get_local_newton_evaluator();

    cm_fiber.computed_growth_scalar_ = cm_fiber.computed_growth_scalar_.val();
    cm_fiber.computed_sigma_ = cm_fiber.computed_sigma_.val();

    Core::LinAlg::Matrix<2, 1, FADdouble> x;
    x(0) = cm_fiber.computed_growth_scalar_;
    x(1) = cm_fiber.computed_sigma_;

    const auto [residuum, derivative] = local_newton_evaluator(x);

    EXPECT_NEAR(residuum(0).dx(0),
        cm_fiber.evaluate_d_residuum_growth_scalar_d_lambda_f_sq().val() * 2 * lambda_f, 1e-8);
    EXPECT_NEAR(residuum(1).dx(0),
        cm_fiber.evaluate_d_residuum_cauchy_stress_d_lambda_f_sq().val() * 2 * lambda_f, 1e-8);
  }

  TEST_F(FullConstrainedMixtureFiberTest,
      StressResponseLinearizationDuringHomeostasisIsAnalyticalSolution)
  {
    Mixture::FullConstrainedMixtureFiber cm_fiber = generate_fiber<FADdouble>();

    const double lambda_f_0 = 1.1;
    cm_fiber.reinitialize_history(lambda_f_0, 0.0);
    const double dt = 0.1;
    for (unsigned timestep = 1; timestep <= 3; ++timestep)
    {
      const double time = timestep * dt;
      const double lambda_f = lambda_f_0;
      cm_fiber.recompute_state(FADdouble(1, 0, lambda_f), time, dt);

      EXPECT_NEAR(cm_fiber.evaluate_d_current_fiber_pk2_stress_d_lambda_f_sq().val() * 2 * lambda_f,
          cm_fiber.evaluate_current_second_pk_stress().dx(0), 1e-7);

      // remove automatic differentiation type from history data
      cm_fiber.computed_growth_scalar_ = cm_fiber.computed_growth_scalar_.val();
      cm_fiber.computed_sigma_ = cm_fiber.computed_sigma_.val();
      cm_fiber.computed_dgrowth_scalar_dlambda_f_sq_ =
          cm_fiber.computed_dgrowth_scalar_dlambda_f_sq_.val();
      cm_fiber.computed_dsigma_dlambda_f_sq_ = cm_fiber.computed_dsigma_dlambda_f_sq_.val();
      cm_fiber.current_state_.lambda_f = cm_fiber.current_state_.lambda_f.val();
      cm_fiber.update();
    }
  }

  TEST_F(FullConstrainedMixtureFiberTest, StressResponseLinearizationIsAnalyticalSolution)
  {
    Mixture::FullConstrainedMixtureFiber cm_fiber = generate_fiber<FADdouble>();

    const double lambda_f_0 = 1.05;
    cm_fiber.reinitialize_history(lambda_f_0, 0.0);
    const double dt = 0.1;
    for (unsigned timestep = 1; timestep <= 3; ++timestep)
    {
      const double time = timestep * dt;
      const double lambda_f = lambda_f_0 + 0.001 * timestep;
      cm_fiber.recompute_state(FADdouble(1, 0, lambda_f), time, dt);

      EXPECT_NEAR(cm_fiber.evaluate_d_current_fiber_pk2_stress_d_lambda_f_sq().val() * 2 * lambda_f,
          cm_fiber.evaluate_current_second_pk_stress().dx(0), 1e-8);

      // remove automatic differentiation type from history data
      cm_fiber.computed_growth_scalar_ = cm_fiber.computed_growth_scalar_.val();
      cm_fiber.computed_sigma_ = cm_fiber.computed_sigma_.val();
      cm_fiber.computed_dgrowth_scalar_dlambda_f_sq_ =
          cm_fiber.computed_dgrowth_scalar_dlambda_f_sq_.val();
      cm_fiber.computed_dsigma_dlambda_f_sq_ = cm_fiber.computed_dsigma_dlambda_f_sq_.val();
      cm_fiber.current_state_.lambda_f = cm_fiber.current_state_.lambda_f.val();
      cm_fiber.update();
    }
  }

  TEST_F(FullConstrainedMixtureFiberTest,
      StressResponseLinearizationIsAnalyticalSolutionInInitialGrowthFreePeriod)
  {
    Mixture::FullConstrainedMixtureFiber fiber = generate_fiber<FADdouble>(12.0, 0.1, 1.1, false);

    double lambda_f = 1.2;
    fiber.recompute_state(FADdouble(1, 0, lambda_f), 1.1, 1.0);

    EXPECT_NEAR(fiber.evaluate_d_current_fiber_pk2_stress_d_lambda_f_sq().val() * 2 * lambda_f,
        fiber.evaluate_current_second_pk_stress().dx(0), 1e-8);
  }

  TEST_F(FullConstrainedMixtureFiberTest, GrowthScalarLinearizationIsAnalyticalSolution)
  {
    Mixture::FullConstrainedMixtureFiber cm_fiber = generate_fiber<FADdouble>();

    const double lambda_f_0 = 1.05;
    cm_fiber.reinitialize_history(lambda_f_0, 0.0);
    const double dt = 0.1;
    for (unsigned timestep = 1; timestep <= 3; ++timestep)
    {
      const double time = timestep * dt;
      const double lambda_f = lambda_f_0 + 0.001 * timestep;
      cm_fiber.recompute_state(FADdouble(1, 0, lambda_f), time, dt);

      EXPECT_NEAR(cm_fiber.computed_dgrowth_scalar_dlambda_f_sq_.val() * 2 * lambda_f,
          cm_fiber.computed_growth_scalar_.dx(0), 1e-8);

      // remove automatic differentiation type from history data
      cm_fiber.computed_growth_scalar_ = cm_fiber.computed_growth_scalar_.val();
      cm_fiber.computed_sigma_ = cm_fiber.computed_sigma_.val();
      cm_fiber.computed_dgrowth_scalar_dlambda_f_sq_ =
          cm_fiber.computed_dgrowth_scalar_dlambda_f_sq_.val();
      cm_fiber.computed_dsigma_dlambda_f_sq_ = cm_fiber.computed_dsigma_dlambda_f_sq_.val();
      cm_fiber.current_state_.lambda_f = cm_fiber.current_state_.lambda_f.val();
      cm_fiber.update();
    }
  }

  TEST_F(FullConstrainedMixtureFiberTest, TestReinitializeIdenticalHistory)
  {
    Mixture::FullConstrainedMixtureFiber cm_fiber = generate_fiber<double>();

    cm_fiber.reinitialize_history(1.0, 0.0);

    ASSERT_EQ(cm_fiber.history_.size(), 1);
    ASSERT_EQ(cm_fiber.history_.back().timesteps.size(), 1);

    cm_fiber.reinitialize_history(1.0, 0.0);
    ASSERT_EQ(cm_fiber.history_.size(), 1);
    ASSERT_EQ(cm_fiber.history_.back().timesteps.size(), 1);
  }

  TEST_F(FullConstrainedMixtureFiberTest, TestReinitializeThrowsIfTimeIsNotIdentical)
  {
    Mixture::FullConstrainedMixtureFiber cm_fiber = generate_fiber<double>();

    cm_fiber.reinitialize_history(1.0, 0.0);
    ASSERT_EQ(cm_fiber.history_.size(), 1);
    ASSERT_EQ(cm_fiber.history_.back().timesteps.size(), 1);

    ASSERT_ANY_THROW(cm_fiber.reinitialize_history(1.0, 1.0));
  }

  TEST_F(FullConstrainedMixtureFiberTest, TestReinitializeAddNewBlockOverwritesIfOldIsJustTimestep)
  {
    Mixture::FullConstrainedMixtureFiber cm_fiber = generate_fiber<double>();

    cm_fiber.reinitialize_history(1.0, 0.0);
    ASSERT_EQ(cm_fiber.history_.size(), 1);
    ASSERT_EQ(cm_fiber.history_.back().timesteps.size(), 1);

    cm_fiber.reinitialize_history(1.2, 0.0);

    ASSERT_EQ(cm_fiber.history_.size(), 1);
    ASSERT_EQ(cm_fiber.history_.back().timesteps.size(), 1);
  }

  TEST_F(FullConstrainedMixtureFiberTest, ThrowsIfTimestepsAreNotIdentical)
  {
    Mixture::FullConstrainedMixtureFiber cm_fiber = generate_fiber<double>();
    cm_fiber.reinitialize_history(1.0, 0.0);
    cm_fiber.recompute_state(1.1, 1.0, 1.0);

    cm_fiber.update();
    EXPECT_ANY_THROW(cm_fiber.recompute_state(1.1, 3.0, 2.0));
  }

  TEST_F(FullConstrainedMixtureFiberTest, TestReinitializeAddNewBlock)
  {
    Mixture::FullConstrainedMixtureFiber cm_fiber = generate_fiber<double>();

    cm_fiber.reinitialize_history(1.0, 0.0);
    ASSERT_EQ(cm_fiber.history_.size(), 1);
    ASSERT_EQ(cm_fiber.history_.back().timesteps.size(), 1);

    cm_fiber.recompute_state(1.1, 1.0, 1.0);
    cm_fiber.update();
    ASSERT_EQ(cm_fiber.history_.size(), 1);
    ASSERT_EQ(cm_fiber.history_.back().timesteps.size(), 2);

    cm_fiber.reinitialize_history(1.2, 1.0);
    ASSERT_EQ(cm_fiber.history_.size(), 2);
    ASSERT_EQ(cm_fiber.history_[0].timesteps.size(), 2);
    ASSERT_EQ(cm_fiber.history_[1].timesteps.size(), 1);
  }

  TEST_F(FullConstrainedMixtureFiberTest, ComparisonWithMultipleIntervals)
  {
    // compare the results of the full constrained mixture model with the homogenized constrained
    // mixture model
    Mixture::FullConstrainedMixtureFiber cm_fiber = generate_fiber<double>();
    Mixture::Implementation::RemodelFiberImplementation<2, double> remodel_fiber =
        generate_remodel_fiber();

    const std::array lambda_f = {1.05, 1.1, 1.09999};
    double time = 0.0;
    unsigned timestep = 1;

    for (unsigned interval_id = 0; interval_id < lambda_f.size(); ++interval_id)
    {
      cm_fiber.reinitialize_history(lambda_f[interval_id], time);

      const double dt = 0.1;
      for (; timestep <= 1000 * (interval_id + 1); ++timestep)
      {
        time = timestep * dt;
        cm_fiber.recompute_state(lambda_f[interval_id], time, dt);
        remodel_fiber.set_state(lambda_f[interval_id], 1.0);

        remodel_fiber.integrate_local_evolution_equations_implicit(dt);

        // compare
        ASSERT_NEAR(cm_fiber.computed_sigma_ / cm_fiber.sig_h_,
            remodel_fiber.evaluate_current_fiber_cauchy_stress() /
                remodel_fiber.evaluate_current_homeostatic_fiber_cauchy_stress(),
            1e-2);
        ASSERT_NEAR(
            cm_fiber.computed_growth_scalar_, remodel_fiber.evaluate_current_growth_scalar(), 3e-2);

        cm_fiber.update();
        remodel_fiber.update();
      }
    }
  }


  TEST_F(FullConstrainedMixtureFiberTest, FiberWithJumpInTimeWithSingleInterval)
  {
    Mixture::FullConstrainedMixtureFiber cm_fiber_timeshift = generate_fiber<double>();

    cm_fiber_timeshift.reinitialize_history(1.0, 0.0);

    for (int timestep = 1; timestep <= 4; ++timestep)
    {
      cm_fiber_timeshift.recompute_state(1.1 + 0.001 * timestep, 0.1 * timestep, 0.1);

      ASSERT_DOUBLE_EQ(
          cm_fiber_timeshift.computed_growth_scalar_, cm_fiber_timeshift.computed_growth_scalar_);

      cm_fiber_timeshift.update();
    }

    ASSERT_DOUBLE_EQ(cm_fiber_timeshift.reference_time_, 0.0);
    ASSERT_DOUBLE_EQ(cm_fiber_timeshift.history_.back().timesteps[0].deposition_time, 0.0);
    ASSERT_DOUBLE_EQ(cm_fiber_timeshift.history_.back().timesteps[1].deposition_time, 0.1);
    ASSERT_DOUBLE_EQ(cm_fiber_timeshift.history_.back().timesteps[2].deposition_time, 0.2);
    ASSERT_DOUBLE_EQ(cm_fiber_timeshift.history_.back().timesteps[3].deposition_time, 0.3);
    ASSERT_DOUBLE_EQ(cm_fiber_timeshift.history_.back().timesteps[4].deposition_time, 0.4);

    cm_fiber_timeshift.add_time(1.0);

    ASSERT_DOUBLE_EQ(cm_fiber_timeshift.reference_time_, 1.0);
    ASSERT_DOUBLE_EQ(cm_fiber_timeshift.history_.back().timesteps[0].deposition_time, 1.0);
    ASSERT_DOUBLE_EQ(cm_fiber_timeshift.history_.back().timesteps[1].deposition_time, 1.1);
    ASSERT_DOUBLE_EQ(cm_fiber_timeshift.history_.back().timesteps[2].deposition_time, 1.2);
    ASSERT_DOUBLE_EQ(cm_fiber_timeshift.history_.back().timesteps[3].deposition_time, 1.3);
    ASSERT_DOUBLE_EQ(cm_fiber_timeshift.history_.back().timesteps[4].deposition_time, 1.4);
  }

  TEST_F(FullConstrainedMixtureFiberTest, FiberWithJumpInTimeWithMultipleIntervals)
  {
    Mixture::FullConstrainedMixtureFiber cm_fiber_timeshift = generate_fiber<double>();

    cm_fiber_timeshift.reinitialize_history(1.0, 0.0);

    for (int timestep = 1; timestep <= 3; ++timestep)
    {
      cm_fiber_timeshift.recompute_state(1.1 + 0.001 * timestep, 0.1 * timestep, 0.1);

      ASSERT_DOUBLE_EQ(
          cm_fiber_timeshift.computed_growth_scalar_, cm_fiber_timeshift.computed_growth_scalar_);

      cm_fiber_timeshift.update();
    }

    cm_fiber_timeshift.reinitialize_history(1.2, 0.3);

    for (int timestep = 1; timestep <= 3; ++timestep)
    {
      cm_fiber_timeshift.recompute_state(1.2 + 0.001 * timestep, 0.3 + 0.1 * timestep, 0.1);

      ASSERT_DOUBLE_EQ(
          cm_fiber_timeshift.computed_growth_scalar_, cm_fiber_timeshift.computed_growth_scalar_);

      cm_fiber_timeshift.update();
    }

    ASSERT_DOUBLE_EQ(cm_fiber_timeshift.reference_time_, 0.0);
    ASSERT_DOUBLE_EQ(cm_fiber_timeshift.history_[0].timesteps[0].deposition_time, 0.0);
    ASSERT_DOUBLE_EQ(cm_fiber_timeshift.history_[0].timesteps[1].deposition_time, 0.1);
    ASSERT_DOUBLE_EQ(cm_fiber_timeshift.history_[0].timesteps[2].deposition_time, 0.2);
    ASSERT_DOUBLE_EQ(cm_fiber_timeshift.history_[0].timesteps[3].deposition_time, 0.3);
    ASSERT_DOUBLE_EQ(cm_fiber_timeshift.history_[1].timesteps[0].deposition_time, 0.3);
    ASSERT_DOUBLE_EQ(cm_fiber_timeshift.history_[1].timesteps[1].deposition_time, 0.4);
    ASSERT_DOUBLE_EQ(cm_fiber_timeshift.history_[1].timesteps[2].deposition_time, 0.5);
    ASSERT_DOUBLE_EQ(cm_fiber_timeshift.history_[1].timesteps[3].deposition_time, 0.6);

    cm_fiber_timeshift.add_time(1.0);

    ASSERT_DOUBLE_EQ(cm_fiber_timeshift.reference_time_, 1.0);
    ASSERT_DOUBLE_EQ(cm_fiber_timeshift.history_[0].timesteps[0].deposition_time, 1.0);
    ASSERT_DOUBLE_EQ(cm_fiber_timeshift.history_[0].timesteps[1].deposition_time, 1.1);
    ASSERT_DOUBLE_EQ(cm_fiber_timeshift.history_[0].timesteps[2].deposition_time, 1.2);
    ASSERT_DOUBLE_EQ(cm_fiber_timeshift.history_[0].timesteps[3].deposition_time, 1.3);
    ASSERT_DOUBLE_EQ(cm_fiber_timeshift.history_[1].timesteps[0].deposition_time, 1.3);
    ASSERT_DOUBLE_EQ(cm_fiber_timeshift.history_[1].timesteps[1].deposition_time, 1.4);
    ASSERT_DOUBLE_EQ(cm_fiber_timeshift.history_[1].timesteps[2].deposition_time, 1.5);
    ASSERT_DOUBLE_EQ(cm_fiber_timeshift.history_[1].timesteps[3].deposition_time, 1.6);
  }

  TEST_F(FullConstrainedMixtureFiberTest, FiberWithJumpInTimeEqualsWithoutJumpIntime)
  {
    Mixture::FullConstrainedMixtureFiber cm_fiber_default = generate_fiber<double>();
    Mixture::FullConstrainedMixtureFiber cm_fiber_timeshift = generate_fiber<double>();
    std::array all_lambda_f = {1.1, 1.2, 1.3};


    int timestep = 1;
    for (double lambda_f : all_lambda_f)
    {
      cm_fiber_default.reinitialize_history(lambda_f, 0.1 * (timestep - 1));
      cm_fiber_timeshift.reinitialize_history(lambda_f, 0.1 * (timestep - 1));

      for (int i = 0; i <= 3; ++i)
      {
        cm_fiber_default.recompute_state(lambda_f + 0.001 * timestep, 0.1 * timestep, 0.1);
        cm_fiber_timeshift.recompute_state(lambda_f + 0.001 * timestep, 0.1 * timestep, 0.1);

        ASSERT_DOUBLE_EQ(cm_fiber_default.computed_sigma_, cm_fiber_default.computed_sigma_);
        ASSERT_DOUBLE_EQ(
            cm_fiber_timeshift.computed_growth_scalar_, cm_fiber_timeshift.computed_growth_scalar_);

        cm_fiber_default.update();
        cm_fiber_timeshift.update();

        ++timestep;
      }
    }

    cm_fiber_timeshift.add_time(1.0);

    cm_fiber_default.recompute_state(1.3 + 0.001 * timestep, 0.1 * timestep, 0.1);
    cm_fiber_timeshift.recompute_state(1.3 + 0.001 * timestep, 0.1 * timestep + 1.0, 0.1);

    ASSERT_DOUBLE_EQ(cm_fiber_default.computed_sigma_, cm_fiber_default.computed_sigma_);
    ASSERT_DOUBLE_EQ(
        cm_fiber_timeshift.computed_growth_scalar_, cm_fiber_timeshift.computed_growth_scalar_);
  }


  TEST_F(FullConstrainedMixtureFiberTest, GetLastTimeInHistoryIfHistoryIsEmpty)
  {
    Mixture::FullConstrainedMixtureFiber cm_fiber = generate_fiber<double>();

    ASSERT_DOUBLE_EQ(cm_fiber.get_last_time_in_history(), 0.0);
  }


  TEST_F(FullConstrainedMixtureFiberTest, GetLastTimeInHistoryIfHistoryIsEmptyAndAfterAddTime)
  {
    Mixture::FullConstrainedMixtureFiber cm_fiber = generate_fiber<double>();

    cm_fiber.add_time(1.0);
    ASSERT_DOUBLE_EQ(cm_fiber.get_last_time_in_history(), 1.0);
  }

  TEST_F(FullConstrainedMixtureFiberTest, GetLastTimeInHistoryWithMultipleIntervals)
  {
    Mixture::FullConstrainedMixtureFiber cm_fiber = generate_fiber<double>();

    cm_fiber.reinitialize_history(1.0, 0.0);

    for (int timestep = 1; timestep <= 3; ++timestep)
    {
      cm_fiber.recompute_state(1.1 + 0.001 * timestep, 0.1 * timestep, 0.1);

      ASSERT_DOUBLE_EQ(cm_fiber.computed_growth_scalar_, cm_fiber.computed_growth_scalar_);

      cm_fiber.update();
    }

    cm_fiber.reinitialize_history(1.2, 0.3);

    for (int timestep = 1; timestep <= 3; ++timestep)
    {
      cm_fiber.recompute_state(1.2 + 0.001 * timestep, 0.3 + 0.1 * timestep, 0.1);

      ASSERT_DOUBLE_EQ(cm_fiber.computed_growth_scalar_, cm_fiber.computed_growth_scalar_);

      cm_fiber.update();
    }

    ASSERT_DOUBLE_EQ(cm_fiber.get_last_time_in_history(), 0.6);
  }

  TEST_F(FullConstrainedMixtureFiberTest, UpdateCallsAddTimeIfGrowthIsDisabled)
  {
    Mixture::FullConstrainedMixtureFiber fiber = generate_fiber<double>(12.0, 0.1, 1.1, false);

    for (int timestep = 1; timestep <= 3; ++timestep)
    {
      const double time = 0.3 + 0.1 * timestep;
      fiber.recompute_state(1.2 + 0.001 * timestep, time, 0.1);

      fiber.update();

      EXPECT_DOUBLE_EQ(fiber.reference_time_, time);
    }
  }

  TEST_F(FullConstrainedMixtureFiberTest,
      CauchyStressAndGrowthScalarIsNormalIfHistoryIsEmptyAndGrowthIsDisabled)
  {
    Mixture::FullConstrainedMixtureFiber fiber = generate_fiber<double>(12.0, 0.1, 1.1, false);
    Mixture::FullConstrainedMixtureFiber fiber_nogrowth =
        generate_fiber<double>(1e100, 0.0, 1.1, true);

    fiber_nogrowth.reinitialize_history(1.2, 0.0);

    for (int timestep = 1; timestep <= 3; ++timestep)
    {
      fiber.recompute_state(1.2 + 0.001 * timestep, 0.3 + 0.1 * timestep, 0.1);
      fiber_nogrowth.recompute_state(1.2 + 0.001 * timestep, 0.3 + 0.1 * timestep, 0.1);

      EXPECT_DOUBLE_EQ(fiber.computed_sigma_, fiber_nogrowth.computed_sigma_);
      EXPECT_DOUBLE_EQ(fiber.computed_growth_scalar_, 1.0);
      EXPECT_DOUBLE_EQ(fiber_nogrowth.computed_growth_scalar_, 1.0);

      fiber.update();
      fiber_nogrowth.update();
    }
  }

  TEST_F(FullConstrainedMixtureFiberTest,
      CauchyStressAndGrowthScalarRemainNormalIfGrowthIsDisabledAfterGrowthPeriod)
  {
    Mixture::FullConstrainedMixtureFiber fiber = generate_fiber<double>(12.0, 0.1, 1.1, true);

    double lambda_f = 1.2;
    fiber.reinitialize_history(lambda_f, 0.0);
    for (int timestep = 1; timestep <= 3; ++timestep)
    {
      lambda_f = 1.2 + 0.001 * timestep;
      fiber.recompute_state(lambda_f, 0.1 + 0.1 * timestep, 0.1);

      fiber.update();
    }

    fiber.enable_growth_ = false;

    const double growth_scalar = fiber.computed_growth_scalar_;
    const double sigma = fiber.computed_sigma_;

    for (int timestep = 4; timestep <= 600; ++timestep)
    {
      fiber.recompute_state(lambda_f, 0.1 + 0.1 * timestep, 0.1);

      EXPECT_NEAR(fiber.computed_growth_scalar_, growth_scalar, 1e-5);
      EXPECT_NEAR(fiber.computed_sigma_, sigma, 1e-5);
      fiber.update();
    }
  }

  TEST_F(FullConstrainedMixtureFiberTest,
      CauchyStressAndGrowthScalarRemainNormalIfGrowthIsDisabledAfterGrowthPeriodAndInitialGrowthFreePeriod)
  {
    Mixture::FullConstrainedMixtureFiber fiber = generate_fiber<double>(12.0, 0.1, 1.1, false);

    double lambda_f = 1.2;

    for (int timestep = 1; timestep <= 3; ++timestep)
    {
      lambda_f = 1.2 + 0.001 * timestep;
      fiber.recompute_state(lambda_f, 0.1 + 0.1 * timestep, 0.1);

      EXPECT_DOUBLE_EQ(fiber.computed_growth_scalar_, 1.0);

      fiber.update();
    }
    fiber.enable_growth_ = true;

    EXPECT_DOUBLE_EQ(fiber.computed_growth_scalar_, 1.0);

    fiber.reinitialize_history(lambda_f, 0.4);
    for (int timestep = 4; timestep <= 6; ++timestep)
    {
      lambda_f = 1.2 + 0.001 * timestep;
      fiber.recompute_state(lambda_f, 0.1 + 0.1 * timestep, 0.1);

      fiber.update();
    }

    fiber.enable_growth_ = false;

    const double growth_scalar = fiber.computed_growth_scalar_;
    const double sigma = fiber.computed_sigma_;

    for (int timestep = 7; timestep <= 9; ++timestep)
    {
      fiber.recompute_state(lambda_f, 0.1 + 0.1 * timestep, 0.1);

      EXPECT_NEAR(fiber.computed_growth_scalar_, growth_scalar, 1e-12);
      EXPECT_NEAR(fiber.computed_sigma_, sigma, 1e-12);
      fiber.update();
    }
  }

}  // namespace
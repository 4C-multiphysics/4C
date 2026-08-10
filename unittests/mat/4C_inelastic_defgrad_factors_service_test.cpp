// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_mat_inelastic_defgrad_factors_service.hpp"
#include "4C_unittest_utils_assertions_test.hpp"
#include "4C_utils_singleton_owner.hpp"


namespace
{
  using namespace FourC;

  namespace ViscoplastUtils = Mat::InelasticDefgradTransvIsotropElastViscoplastUtils;

  class InelasticDefgradFactorsServiceTest : public ::testing::Test
  {
   protected:
    void SetUp() override {}


    Core::Utils::SingletonOwnerRegistry::ScopeGuard guard;
  };

  /// tests the LocalIntegrationInput of
  /// InelasticDefgradTransvIsotropElastViscoplast
  TEST_F(InelasticDefgradFactorsServiceTest, TestLocalIntegrationInput)
  {
    // setup input
    Core::LinAlg::Matrix<3, 3> defgrad{Core::LinAlg::Initialization::zero};
    defgrad(0, 0) = 0.2513819028974873;
    defgrad(0, 1) = 0.957511195526664;
    defgrad(0, 2) = 0.8703229224151933;
    defgrad(1, 0) = 0.675673714544612;
    defgrad(1, 1) = 0.040444301498430923;
    defgrad(1, 2) = 0.10298502801901921;
    defgrad(2, 0) = 0.20079631315327318;
    defgrad(2, 1) = 0.6901106554801166;
    defgrad(2, 2) = 0.1769124998126297;


    Core::LinAlg::Matrix<3, 3> last_iFin{Core::LinAlg::Initialization::zero};
    last_iFin(0, 0) = 0.35530729350748047;
    last_iFin(0, 1) = 0.5829896953147952;
    last_iFin(0, 2) = 0.9336918888091672;
    last_iFin(1, 0) = 0.3099852313939162;
    last_iFin(1, 1) = 0.7243285059889488;
    last_iFin(1, 2) = 0.43375156919140767;
    last_iFin(2, 0) = 0.41454463288433163;
    last_iFin(2, 1) = 0.6433884759079006;
    last_iFin(2, 2) = 0.23890433987101256;


    // reference tensors based on input
    Core::LinAlg::Matrix<3, 3> inv_defgrad_ref{Core::LinAlg::Initialization::zero};
    inv_defgrad_ref(0, 0) = -0.22190619146746082;
    inv_defgrad_ref(0, 1) = 1.4971400487822994;
    inv_defgrad_ref(0, 2) = 0.2201485775679747;
    inv_defgrad_ref(1, 0) = -0.34321290611467375;
    inv_defgrad_ref(1, 1) = -0.4523291882409134;
    inv_defgrad_ref(1, 2) = 1.951751255286342;
    inv_defgrad_ref(2, 0) = 1.590689346533634;
    inv_defgrad_ref(2, 1) = 0.06521297552376205;
    inv_defgrad_ref(2, 2) = -2.2108633434926004;


    Core::LinAlg::Matrix<3, 3> right_cg_ref{Core::LinAlg::Initialization::zero};
    right_cg_ref(0, 0) = 0.5600469890068229;
    right_cg_ref(0, 1) = 0.40659981309094395;
    right_cg_ref(0, 2) = 0.3238910865092303;
    right_cg_ref(1, 0) = 0.40659981309094395;
    right_cg_ref(1, 1) = 1.3947161478897936;
    right_cg_ref(1, 2) = 0.9595983006673772;
    right_cg_ref(2, 0) = 0.3238910865092303;
    right_cg_ref(2, 1) = 0.9595983006673772;
    right_cg_ref(2, 2) = 0.7993659378673543;


    Core::LinAlg::Matrix<3, 3> elastic_predictor_inverse_plastic_defgrad_ref{
        Core::LinAlg::Initialization::zero};
    elastic_predictor_inverse_plastic_defgrad_ref(0, 0) = 0.35530729350748047;
    elastic_predictor_inverse_plastic_defgrad_ref(0, 1) = 0.5829896953147952;
    elastic_predictor_inverse_plastic_defgrad_ref(0, 2) = 0.9336918888091672;
    elastic_predictor_inverse_plastic_defgrad_ref(1, 0) = 0.3099852313939162;
    elastic_predictor_inverse_plastic_defgrad_ref(1, 1) = 0.7243285059889488;
    elastic_predictor_inverse_plastic_defgrad_ref(1, 2) = 0.43375156919140767;
    elastic_predictor_inverse_plastic_defgrad_ref(2, 0) = 0.41454463288433163;
    elastic_predictor_inverse_plastic_defgrad_ref(2, 1) = 0.6433884759079006;
    elastic_predictor_inverse_plastic_defgrad_ref(2, 2) = 0.23890433987101256;


    Core::LinAlg::Matrix<3, 3> elastic_predictor_elastic_defgrad_ref{
        Core::LinAlg::Initialization::zero};
    elastic_predictor_elastic_defgrad_ref(0, 0) = 0.7469198494262896;
    elastic_predictor_elastic_defgrad_ref(0, 1) = 1.4000614513018017;
    elastic_predictor_elastic_defgrad_ref(0, 2) = 0.8579591505610411;
    elastic_predictor_elastic_defgrad_ref(1, 0) = 0.2953008256002754;
    elastic_predictor_elastic_defgrad_ref(1, 1) = 0.48946515367319354;
    elastic_predictor_elastic_defgrad_ref(1, 2) = 0.6730174161271413;
    elastic_predictor_elastic_defgrad_ref(2, 0) = 0.3586066330866571;
    elastic_predictor_elastic_defgrad_ref(2, 1) = 0.7307524651000326;
    elastic_predictor_elastic_defgrad_ref(2, 2) = 0.5290836326068751;

    const double temperature = 293.15;
    const double last_plastic_strain = 0.0;
    const double timestep = 0.1;


    // initialize LocalIntegrationInput and perform checks for the saved quantities
    ViscoplastUtils::LocalIntegrationInput local_integration_input{
        ViscoplastUtils::LocalIntegrationInputConfig{.defgrad = defgrad,
            .temperature = temperature,
            .last_inv_inelastic_defgrad = last_iFin,
            .last_plastic_strain = last_plastic_strain,
            .step = timestep}};
    FOUR_C_EXPECT_NEAR(local_integration_input.defgrad, defgrad, 1.0e-15);
    FOUR_C_EXPECT_NEAR(local_integration_input.inv_defgrad, inv_defgrad_ref, 1.0e-15);
    FOUR_C_EXPECT_NEAR(local_integration_input.right_cg, right_cg_ref, 1.0e-15);
    FOUR_C_EXPECT_NEAR(local_integration_input.elastic_predictor_elastic_defgrad,
        elastic_predictor_elastic_defgrad_ref, 1.0e-15);
    FOUR_C_EXPECT_NEAR(local_integration_input.elastic_predictor_inverse_plastic_defgrad,
        elastic_predictor_inverse_plastic_defgrad_ref, 1.0e-15);
    EXPECT_EQ(local_integration_input.temperature, temperature);
    EXPECT_EQ(local_integration_input.last_plastic_strain, last_plastic_strain);
    EXPECT_EQ(local_integration_input.step, timestep);
  }


  /// tests the bookkeeping of iterations within the LocalNewtonManager of
  /// InelasticDefgradTransvIsotropElastViscoplast
  TEST_F(InelasticDefgradFactorsServiceTest, TestLocalNewtonManagerIterBookkeeping)
  {
    auto local_newton_params =
        Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonParams{
            .res_tol = 1.0e-8,
            .incr_tol = 1.0e-8,
            .conv_check = ViscoplastUtils::LocalNewtonConvCheck::residual_and_increment_ratio,
            .diver_cont = ViscoplastUtils::LocalNewtonDiverCont::stop,
            .max_iter = 5,
            .max_exceedance_fact_res_tol = 1.0e1,
            .max_exceedance_fact_incr_tol = 1.0e1,

        };
    Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonManager local_newton_manager(
        local_newton_params);

    EXPECT_EQ(local_newton_manager.iter(), 0);
    EXPECT_EQ(local_newton_manager.curr_num_iters().size(), 1);
    EXPECT_EQ(local_newton_manager.curr_num_iters()[0], 0);

    local_newton_manager.resize(3);
    EXPECT_EQ(local_newton_manager.curr_num_iters().size(), 3);
    EXPECT_EQ(local_newton_manager.curr_num_iters()[0], 0);
    EXPECT_EQ(local_newton_manager.curr_num_iters()[1], 0);
    EXPECT_EQ(local_newton_manager.curr_num_iters()[2], 0);

    Core::LinAlg::Matrix<10, 1> one_10x1{Core::LinAlg::Initialization::zero};
    for (unsigned int i = 0; i < 10; ++i) one_10x1(i) = 1.0;

    local_newton_manager.reset_iter();
    local_newton_manager.save_init_estimate_and_reset_convergence_quantities(one_10x1);
    local_newton_manager.increment_solution_vector(one_10x1);
    local_newton_manager.increment_iter();
    local_newton_manager.increment_solution_vector(one_10x1);
    local_newton_manager.increment_iter();
    local_newton_manager.increment_solution_vector(one_10x1);
    local_newton_manager.increment_iter();
    EXPECT_EQ(local_newton_manager.iter(), 3);
    local_newton_manager.update_after_local_newton(1);
    EXPECT_EQ(local_newton_manager.curr_num_iters()[1], 3);

    local_newton_manager.save_init_estimate_and_reset_convergence_quantities(one_10x1);
    local_newton_manager.increment_solution_vector(one_10x1);
    local_newton_manager.increment_iter();
    EXPECT_EQ(local_newton_manager.iter(), 4);
    local_newton_manager.update_after_local_newton(1);
    EXPECT_EQ(local_newton_manager.curr_num_iters()[1], 7);

    local_newton_manager.reset_curr_num_iters(0);
    EXPECT_EQ(local_newton_manager.curr_num_iters()[0], 0);
    local_newton_manager.reset_curr_num_iters(1);
    EXPECT_EQ(local_newton_manager.curr_num_iters()[1], 0);
    local_newton_manager.reset_curr_num_iters(2);
    EXPECT_EQ(local_newton_manager.curr_num_iters()[2], 0);


    // test whether the maximum number of iterations was exceeded
    EXPECT_FALSE(local_newton_manager.is_max_iter_reached());
    local_newton_manager.increment_solution_vector(one_10x1);
    local_newton_manager.increment_iter();
    local_newton_manager.increment_solution_vector(one_10x1);
    local_newton_manager.increment_iter();
    EXPECT_TRUE(local_newton_manager.is_max_iter_reached());
  }


  /// tests the basic functionality of the LocalNewtonManager (initialization, incrementation,
  /// convergence and "stuckness" verification) used within
  /// InelasticDefgradTransvIsotropElastViscoplast
  TEST_F(InelasticDefgradFactorsServiceTest, TestLocalNewtonManagerBasicFunctionality)
  {
    auto local_newton_params =
        Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonParams{
            .res_tol = 1.0e-8,
            .incr_tol = 1.0e-8,
            .conv_check = ViscoplastUtils::LocalNewtonConvCheck::residual_and_increment_ratio,
            .diver_cont = ViscoplastUtils::LocalNewtonDiverCont::stop,
            .max_iter = 100,
            .max_exceedance_fact_res_tol = 0.0,
            .max_exceedance_fact_incr_tol = 0.0,
        };
    Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonManager local_newton_manager(
        local_newton_params);

    // auxiliaries
    Core::LinAlg::Matrix<10, 1> one_10x1{Core::LinAlg::Initialization::zero};
    for (unsigned int i = 0; i < 10; ++i) one_10x1(i) = 1.0;


    // --> test initialization with and without iteration counter reset

    // with iteration counter reset
    local_newton_manager.reset_iter();
    local_newton_manager.save_init_estimate_and_reset_convergence_quantities(one_10x1);
    FOUR_C_EXPECT_NEAR(local_newton_manager.sol(), one_10x1, 1.0e-15);
    EXPECT_EQ(local_newton_manager.iter(), 0);

    // without iteration counter reset
    local_newton_manager.increment_solution_vector(one_10x1);
    local_newton_manager.increment_iter();  // increment the iteration counter
    local_newton_manager.save_init_estimate_and_reset_convergence_quantities(one_10x1);
    FOUR_C_EXPECT_NEAR(local_newton_manager.sol(), one_10x1, 1.0e-15);
    EXPECT_EQ(local_newton_manager.iter(), 1);


    // --> test workflow within the Local Newton: increment the solution vector (save the
    // increment), then set the residual norm, and perform the convergence check
    Core::LinAlg::Matrix<10, 1> vector_under_tol{
        Core::LinAlg::Initialization::zero};  // the 2-norm of this vector is smaller than the set
                                              // value for residual and increment tolerance
    vector_under_tol(0) = 1.0e-9;
    Core::LinAlg::Matrix<10, 1> vector_over_tol{Core::LinAlg::Initialization::zero};
    vector_over_tol(0) = 1.0e-7;  // the 2-norm of this vector is smaller than the set value for
                                  // residual and increment tolerance

    Core::LinAlg::Matrix<10, 1> updated_sol_ref(
        Core::LinAlg::Initialization::zero);  // reference: updated solution vector, used for the
                                              // solution vector checks

    // try out increment and residual vector exceeding the tolerance: no convergence!
    local_newton_manager.reset_iter();
    local_newton_manager.save_init_estimate_and_reset_convergence_quantities(one_10x1);
    local_newton_manager.increment_solution_vector(vector_over_tol);
    local_newton_manager.increment_iter();
    updated_sol_ref.update(1.0, one_10x1, 1.0, vector_over_tol, 0.0);
    FOUR_C_EXPECT_NEAR(local_newton_manager.sol(), updated_sol_ref, 1.0e-15);
    EXPECT_EQ(local_newton_manager.convergence_quantities().increment_norm,
        vector_over_tol(0) / updated_sol_ref.norm2());
    EXPECT_EQ(local_newton_manager.iter(), 1);
    local_newton_manager.set_residual_norm(vector_over_tol);
    EXPECT_EQ(local_newton_manager.convergence_quantities().residual_norm, vector_over_tol(0));
    EXPECT_FALSE(local_newton_manager.is_local_newton_converged());

    // try out increment exceeding the tolerance, and residual vector under the tolerance: no
    // convergence!
    local_newton_manager.increment_solution_vector(vector_over_tol);
    local_newton_manager.increment_iter();
    updated_sol_ref.update(1.0, vector_over_tol, 1.0);
    FOUR_C_EXPECT_NEAR(local_newton_manager.sol(), updated_sol_ref, 1.0e-15);
    EXPECT_EQ(local_newton_manager.convergence_quantities().increment_norm,
        vector_over_tol(0) / updated_sol_ref.norm2());
    EXPECT_EQ(local_newton_manager.iter(), 2);
    local_newton_manager.set_residual_norm(vector_under_tol);
    EXPECT_EQ(local_newton_manager.convergence_quantities().residual_norm, vector_under_tol(0));
    EXPECT_FALSE(local_newton_manager.is_local_newton_converged());

    // now the other way around: no convergence!
    local_newton_manager.increment_solution_vector(vector_under_tol);
    local_newton_manager.increment_iter();
    updated_sol_ref.update(1.0, vector_under_tol, 1.0);
    FOUR_C_EXPECT_NEAR(local_newton_manager.sol(), updated_sol_ref, 1.0e-15);
    EXPECT_EQ(local_newton_manager.convergence_quantities().increment_norm,
        vector_under_tol(0) / updated_sol_ref.norm2());
    EXPECT_EQ(local_newton_manager.iter(), 3);
    local_newton_manager.set_residual_norm(vector_over_tol);
    EXPECT_EQ(local_newton_manager.convergence_quantities().residual_norm, vector_over_tol(0));
    EXPECT_FALSE(local_newton_manager.is_local_newton_converged());

    // now, both increment and residual are under the tolerance: convergence!
    local_newton_manager.increment_solution_vector(vector_under_tol);
    local_newton_manager.increment_iter();
    updated_sol_ref.update(1.0, vector_under_tol, 1.0);
    FOUR_C_EXPECT_NEAR(local_newton_manager.sol(), updated_sol_ref, 1.0e-15);
    EXPECT_EQ(local_newton_manager.convergence_quantities().increment_norm,
        vector_under_tol(0) / updated_sol_ref.norm2());
    EXPECT_EQ(local_newton_manager.iter(), 4);
    local_newton_manager.set_residual_norm(vector_under_tol);
    EXPECT_EQ(local_newton_manager.convergence_quantities().residual_norm, vector_under_tol(0));
    EXPECT_TRUE(local_newton_manager.is_local_newton_converged());

    // --> test whether the Local Newton becomes stuck: increment is exactly 0.0, but the residual
    // is still over the set tolerance
    EXPECT_FALSE(local_newton_manager.is_local_newton_stuck());  // for the previous settings, the
                                                                 // Local Newton should not be stuck
    Core::LinAlg::Matrix<10, 1> zero_10x1{Core::LinAlg::Initialization::zero};
    local_newton_manager.increment_solution_vector(zero_10x1);
    local_newton_manager.increment_iter();
    FOUR_C_EXPECT_NEAR(local_newton_manager.sol(), updated_sol_ref, 1.0e-15);
    EXPECT_EQ(local_newton_manager.convergence_quantities().increment_norm, 0.0);
    EXPECT_EQ(local_newton_manager.iter(), 5);
    local_newton_manager.set_residual_norm(vector_over_tol);
    EXPECT_EQ(local_newton_manager.convergence_quantities().residual_norm, vector_over_tol(0));
    EXPECT_TRUE(local_newton_manager.is_local_newton_stuck());
  }


  /// tests the convergence of the LocalNewtonManager (for various settings) used within
  /// InelasticDefgradTransvIsotropElastViscoplast
  TEST_F(InelasticDefgradFactorsServiceTest, TestLocalNewtonManagerConvergenceVerification)
  {
    // framework for setting up multiple LocalNewtonManager objects with varied parameters
    auto local_newton_base_params =
        Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonParams{
            .res_tol = 1.0e-8,
            .incr_tol = 1.0e-10,
            .conv_check = ViscoplastUtils::LocalNewtonConvCheck::residual_and_increment_ratio,
            .diver_cont = ViscoplastUtils::LocalNewtonDiverCont::stop,
            .max_iter = 100,
            .max_exceedance_fact_res_tol = 0.0,
            .max_exceedance_fact_incr_tol = 0.0,
        };
    auto set_up_local_newton_manager = [local_newton_base_params](
                                           const ViscoplastUtils::LocalNewtonConvCheck conv_check)
    {
      Core::LinAlg::Matrix<10, 1> one_10x1{Core::LinAlg::Initialization::zero};
      for (unsigned int i = 0; i < 10; ++i) one_10x1(i) = 1.0;

      auto manager = ViscoplastUtils::LocalNewtonManager({
          .res_tol = local_newton_base_params.res_tol,
          .incr_tol = local_newton_base_params.incr_tol,
          .conv_check = conv_check,  // override
          .diver_cont = local_newton_base_params.diver_cont,
          .max_iter = local_newton_base_params.max_iter,
          .max_exceedance_fact_res_tol = local_newton_base_params.max_exceedance_fact_res_tol,
          .max_exceedance_fact_incr_tol = local_newton_base_params.max_exceedance_fact_incr_tol,
      });
      manager.reset_iter();
      manager.save_init_estimate_and_reset_convergence_quantities(one_10x1);

      return manager;
    };

    // setup several Local Newton managers
    ViscoplastUtils::LocalNewtonManager manager_res_and_incr = set_up_local_newton_manager(
        ViscoplastUtils::LocalNewtonConvCheck::residual_and_increment_ratio);
    ViscoplastUtils::LocalNewtonManager manager_res =
        set_up_local_newton_manager(ViscoplastUtils::LocalNewtonConvCheck::residual);
    ViscoplastUtils::LocalNewtonManager manager_incr =
        set_up_local_newton_manager(ViscoplastUtils::LocalNewtonConvCheck::increment_ratio);

    // setup vectors (residual / increment) to be used for convergence checks
    auto vector_from_tol = [](const double first_value)
    {
      Core::LinAlg::Matrix<10, 1> out{
          Core::LinAlg::Initialization::zero};  // the 2-norm of this vector is smaller than the set
                                                // value for residual and increment tolerance
      out(0) = first_value;

      return out;
    };

    // setup numerical values smaller than, or exceeding the set tolerances
    const double exceeds_incr_tol{1.0e-9};
    const double exceeds_res_tol{1.0e-7};
    const double smaller_than_incr_tol{1.0e-10};
    const double smaller_than_res_tol{1.0e-9};


    // try out residual and increment exceeding the set tolerances
    manager_res_and_incr.increment_solution_vector(vector_from_tol(exceeds_incr_tol));
    manager_res_and_incr.increment_iter();
    manager_res.increment_solution_vector(vector_from_tol(exceeds_incr_tol));
    manager_res.increment_iter();
    manager_incr.increment_solution_vector(vector_from_tol(exceeds_incr_tol));
    manager_incr.increment_iter();

    manager_res_and_incr.set_residual_norm(vector_from_tol(exceeds_res_tol));
    manager_res.set_residual_norm(vector_from_tol(exceeds_res_tol));
    manager_incr.set_residual_norm(vector_from_tol(exceeds_res_tol));

    EXPECT_FALSE(manager_res_and_incr.is_local_newton_converged());
    EXPECT_FALSE(manager_res.is_local_newton_converged());
    EXPECT_FALSE(manager_incr.is_local_newton_converged());

    // try out residual smaller than the set tolerance, with increment exceeding its set tolerance
    manager_res_and_incr.increment_solution_vector(vector_from_tol(exceeds_incr_tol));
    manager_res_and_incr.increment_iter();
    manager_res.increment_solution_vector(vector_from_tol(exceeds_incr_tol));
    manager_res.increment_iter();
    manager_incr.increment_solution_vector(vector_from_tol(exceeds_incr_tol));
    manager_incr.increment_iter();

    manager_res_and_incr.set_residual_norm(vector_from_tol(smaller_than_res_tol));
    manager_res.set_residual_norm(vector_from_tol(smaller_than_res_tol));
    manager_incr.set_residual_norm(vector_from_tol(smaller_than_res_tol));

    EXPECT_FALSE(manager_res_and_incr.is_local_newton_converged());
    EXPECT_TRUE(manager_res.is_local_newton_converged());
    EXPECT_FALSE(manager_incr.is_local_newton_converged());

    // try out residual exceeding its set tolerance, with increment smaller than its set tolerance
    manager_res_and_incr.increment_solution_vector(vector_from_tol(smaller_than_incr_tol));
    manager_res_and_incr.increment_iter();
    manager_res.increment_solution_vector(vector_from_tol(smaller_than_incr_tol));
    manager_res.increment_iter();
    manager_incr.increment_solution_vector(vector_from_tol(smaller_than_incr_tol));
    manager_incr.increment_iter();

    manager_res_and_incr.set_residual_norm(vector_from_tol(exceeds_res_tol));
    manager_res.set_residual_norm(vector_from_tol(exceeds_res_tol));
    manager_incr.set_residual_norm(vector_from_tol(exceeds_res_tol));

    EXPECT_FALSE(manager_res_and_incr.is_local_newton_converged());
    EXPECT_FALSE(manager_res.is_local_newton_converged());
    EXPECT_TRUE(manager_incr.is_local_newton_converged());
  }
}  // namespace

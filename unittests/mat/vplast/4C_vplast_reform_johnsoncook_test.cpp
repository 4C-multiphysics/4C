// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_global_data.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_mat_inelastic_defgrad_factors.hpp"
#include "4C_mat_inelastic_defgrad_factors_service.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_mat_vplast_law.hpp"
#include "4C_mat_vplast_reform_johnsoncook.hpp"
#include "4C_unittest_utils_assertions_test.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_singleton_owner.hpp"

#include <memory>
namespace
{
  using namespace FourC;

  struct ReformulatedJohnsonCookLaw
  {
    //! parameters
    std::shared_ptr<Mat::Viscoplastic::PAR::ReformulatedJohnsonCook> params;

    //! material
    std::shared_ptr<Mat::Viscoplastic::ReformulatedJohnsonCook> material;
  };

  int make_unique_viscoplastic_law_id()
  {
    static int viscoplastic_law_id = 1;
    return viscoplastic_law_id;
  }


  ReformulatedJohnsonCookLaw set_up_reformulated_johnson_cook_law(
      const Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorRegistrationSettings
          error_registration_settings)
  {
    const int viscoplastic_law_id = make_unique_viscoplastic_law_id();

    Core::IO::InputParameterContainer vplast_law_reformulated_JC_data;
    vplast_law_reformulated_JC_data.add("STRAIN_RATE_PREFAC", 1.0);
    vplast_law_reformulated_JC_data.add("STRAIN_RATE_EXP_FAC", 0.014);
    vplast_law_reformulated_JC_data.add("INIT_YIELD_STRENGTH", 792.0);
    vplast_law_reformulated_JC_data.add("ISOTROP_HARDEN_PREFAC", 510.0);
    vplast_law_reformulated_JC_data.add("ISOTROP_HARDEN_EXP", 0.26);
    vplast_law_reformulated_JC_data.add("REF_TEMPERATURE", 293.0);
    vplast_law_reformulated_JC_data.add("MELT_TEMPERATURE", 1793.0);
    vplast_law_reformulated_JC_data.add("TEMPERATURE_SENS", 1.03);
    auto params = std::dynamic_pointer_cast<Mat::Viscoplastic::PAR::ReformulatedJohnsonCook>(
        std::shared_ptr(Mat::make_parameter(viscoplastic_law_id,
            Core::Materials::MaterialType::mvl_reformulated_Johnson_Cook,
            vplast_law_reformulated_JC_data)));

    auto material = std::make_shared<Mat::Viscoplastic::ReformulatedJohnsonCook>(
        params.get(), error_registration_settings);

    // call setup method for ReformulatedJohnsonCook
    int numgp = 1;
    material->setup(numgp, {}, {});
    // pre_evaluate
    Teuchos::ParameterList param_list{};
    param_list.set<double>("temperature", 313.0);
    material->pre_evaluate(param_list, 0);

    return {.params = params, .material = material};
  }



  class ReformJohnsonCookTest : public ::testing::Test
  {
   protected:
    void SetUp() override
    {
      // set up equivalent stress and plastic strain
      equiv_stress_ = 1000.0;
      equiv_plastic_strain_ = 0.001;

      // manually create viscoplastic law (ReformulatedJohnsonCook)
      const double max_plastic_strain_incr_and_deriv_incr = std::exp(30.0);
      const auto reformulated_johnson_cook_law =
          set_up_reformulated_johnson_cook_law({.register_plastic_strain_incr_overflow = true,
              .max_plastic_strain_incr = max_plastic_strain_incr_and_deriv_incr,
              .register_plastic_strain_deriv_incr_overflow = true,
              .max_plastic_strain_deriv_incr = max_plastic_strain_incr_and_deriv_incr});
      params_vplast_law_reformulated_JC_ = reformulated_johnson_cook_law.params;
      vplast_law_reformulated_JC_ = reformulated_johnson_cook_law.material;
    }

    // equivalent stress
    double equiv_stress_;
    // equivalent stress
    double equiv_plastic_strain_;
    // reference solution for the stress ratio (ReformulatedJohnsonCook)
    double stress_ratio_reformulated_JC_solution_;
    // reference solution for the plastic strain rate (ReformulatedJohnsonCook)
    double plastic_strain_rate_reformulated_JC_solution_;
    // reference solution for the plastic strain rate derivatives, w.r.t. equivalent stress and
    // plastic strain (ReformulatedJohnsonCook)
    Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::PlasticStrainRateDerivs
        deriv_plastic_strain_rate_reformulated_JC_solution_;
    // pointer to ReformulatedJohnsonCook
    std::shared_ptr<Mat::Viscoplastic::ReformulatedJohnsonCook> vplast_law_reformulated_JC_;
    // pointer to parameters of ReformulatedJohnsonCook
    std::shared_ptr<Mat::Viscoplastic::PAR::ReformulatedJohnsonCook>
        params_vplast_law_reformulated_JC_;

    Core::Utils::SingletonOwnerRegistry::ScopeGuard guard;
  };

  TEST_F(ReformJohnsonCookTest, TestEvaluateStressRatio)
  {
    // set reference solution
    stress_ratio_reformulated_JC_solution_ = 1.1556126603348709;

    // compute solution from the viscoplasticity law
    auto err_status = Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType::no_errors;
    const double yield_stress = vplast_law_reformulated_JC_->compute_flow_resistance(
        equiv_stress_, equiv_plastic_strain_, err_status);
    ASSERT_EQ(
        err_status, Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType::no_errors);
    double stress_ratio_reformulated_JC = equiv_stress_ / yield_stress;

    // compare solutions
    EXPECT_NEAR(stress_ratio_reformulated_JC_solution_, stress_ratio_reformulated_JC, 1.0e-8);
  }

  TEST_F(ReformJohnsonCookTest, TestEvaluatePlasticStrainRate)
  {
    // set reference solution
    plastic_strain_rate_reformulated_JC_solution_ = 67182.9745369580195984;


    // declare error status
    Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType err_status =
        Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType::no_errors;

    // compute solution from the viscoplasticity law
    double plastic_strain_rate_reformulated_JC =
        vplast_law_reformulated_JC_->evaluate_plastic_strain_rate(
            equiv_stress_, equiv_plastic_strain_, 1.0, err_status, false);

    if (err_status != Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType::no_errors)
      FOUR_C_THROW("Error encountered during testing of TestEvaluatePlasticStrainRate");
    EXPECT_NEAR(
        plastic_strain_rate_reformulated_JC_solution_, plastic_strain_rate_reformulated_JC, 1.0e-8);

    // test registering of the overflow error
    const auto ref_jc_register_both_with_zero_incr =
        set_up_reformulated_johnson_cook_law({.register_plastic_strain_incr_overflow = true,
            .max_plastic_strain_incr =
                1.0e-16,  // effectively 0-tolerance for plastic strain increments -> results in
                          // overflow error regardless of the computed plastic strain increment
                          // value
            .register_plastic_strain_deriv_incr_overflow = true,
            .max_plastic_strain_deriv_incr = 1.0e-16});
    plastic_strain_rate_reformulated_JC =
        ref_jc_register_both_with_zero_incr.material->evaluate_plastic_strain_rate(
            equiv_stress_, equiv_plastic_strain_, 1.0, err_status, false);
    EXPECT_EQ(err_status,
        Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType::overflow_error);

    const auto ref_jc_register_none_with_zero_incr =
        set_up_reformulated_johnson_cook_law({.register_plastic_strain_incr_overflow = false,
            .max_plastic_strain_incr =
                1.0e-16,  // same test as above, but now without registering the error
            .register_plastic_strain_deriv_incr_overflow = false,
            .max_plastic_strain_deriv_incr = 1.0e-16});
    plastic_strain_rate_reformulated_JC =
        ref_jc_register_none_with_zero_incr.material->evaluate_plastic_strain_rate(
            equiv_stress_, equiv_plastic_strain_, 1.0, err_status, false);
    EXPECT_EQ(
        err_status, Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType::no_errors);
  }

  TEST_F(ReformJohnsonCookTest, TestEvaluatePlasticStrainRateDerivatives)
  {
    // set reference solution
    deriv_plastic_strain_rate_reformulated_JC_solution_.deriv_equiv_stress = 5545.6179676088768247;
    deriv_plastic_strain_rate_reformulated_JC_solution_.deriv_plastic_strain =
        -139210732.3144087791442871;
    deriv_plastic_strain_rate_reformulated_JC_solution_.deriv_temperature = 3623.4626950498809492;


    // declare error status
    Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType err_status =
        Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType::no_errors;

    // compute solution from the viscoplasticity law
    const double max_plastic_strain_incr_and_deriv_incr = std::exp(30.0);
    const auto ref_jc_register_both_with_set_incr =
        set_up_reformulated_johnson_cook_law({.register_plastic_strain_incr_overflow = true,
            .max_plastic_strain_incr = max_plastic_strain_incr_and_deriv_incr,
            .register_plastic_strain_deriv_incr_overflow = true,
            .max_plastic_strain_deriv_incr = max_plastic_strain_incr_and_deriv_incr});
    Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::PlasticStrainRateDerivs
        deriv_plastic_strain_rate_reformulated_JC =
            ref_jc_register_both_with_set_incr.material
                ->evaluate_derivatives_of_plastic_strain_rate(
                    equiv_stress_, equiv_plastic_strain_, 1.0, err_status);

    if (err_status != Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType::no_errors)
      FOUR_C_THROW("Error encountered during testing of TestEvaluatePlasticStrainRateDerivatives");

    // compare solutions
    EXPECT_NEAR(deriv_plastic_strain_rate_reformulated_JC_solution_.deriv_equiv_stress,
        deriv_plastic_strain_rate_reformulated_JC.deriv_equiv_stress, 1.0e-6);
    EXPECT_NEAR(deriv_plastic_strain_rate_reformulated_JC_solution_.deriv_plastic_strain,
        deriv_plastic_strain_rate_reformulated_JC.deriv_plastic_strain, 1.0e-6);
    EXPECT_NEAR(deriv_plastic_strain_rate_reformulated_JC_solution_.deriv_temperature,
        deriv_plastic_strain_rate_reformulated_JC.deriv_temperature, 1.0e-6);



    const auto ref_jc_register_both_with_zero_incr =
        set_up_reformulated_johnson_cook_law({.register_plastic_strain_incr_overflow = true,
            .max_plastic_strain_incr =
                1.0e-16,  // effectively 0-tolerance for plastic strain increments -> results
                          // in overflow error regardless of the computed plastic strain
                          // increment value
            .register_plastic_strain_deriv_incr_overflow = true,
            .max_plastic_strain_deriv_incr = 1.0e-16});
    deriv_plastic_strain_rate_reformulated_JC =
        ref_jc_register_both_with_zero_incr.material->evaluate_derivatives_of_plastic_strain_rate(
            equiv_stress_, equiv_plastic_strain_, 1.0, err_status);

    EXPECT_EQ(err_status, Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType::
                              failed_computation_flow_resistance_derivs);


    const auto ref_jc_register_none_with_zero_incr =
        set_up_reformulated_johnson_cook_law({.register_plastic_strain_incr_overflow = false,
            .max_plastic_strain_incr =
                1.0e-16,  // same test as above, but now without registering the error
            .register_plastic_strain_deriv_incr_overflow = false,
            .max_plastic_strain_deriv_incr = 1.0e-16});
    deriv_plastic_strain_rate_reformulated_JC =
        ref_jc_register_none_with_zero_incr.material->evaluate_derivatives_of_plastic_strain_rate(
            equiv_stress_, equiv_plastic_strain_, 1.0, err_status);
    EXPECT_EQ(
        err_status, Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType::no_errors);
  }

}  // namespace

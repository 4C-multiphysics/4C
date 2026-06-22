// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_global_data.hpp"
#include "4C_io_input_parameter_container.templates.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_fixedsizematrix_generators.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_linalg_four_tensor.hpp"
#include "4C_linalg_utils_densematrix_funct.hpp"
#include "4C_mat_elast_couptransverselyisotropic.hpp"
#include "4C_mat_electrode.hpp"
#include "4C_mat_inelastic_defgrad_factors.hpp"
#include "4C_mat_inelastic_defgrad_factors_service.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_mat_multiplicative_split_defgrad_elasthyper_service.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_mat_vplast_reform_johnsoncook.hpp"
#include "4C_solid_ele_fibers.hpp"
#include "4C_unittest_utils_assertions_test.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_singleton_owner.hpp"

#include <Teuchos_ParameterList.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <memory>
#include <optional>



namespace
{
  using namespace FourC;
  namespace ViscoplastUtils = Mat::InelasticDefgradTransvIsotropElastViscoplastUtils;

  struct ReformulatedJohnsonCookParameters
  {
    double strain_rate_prefac = 1.0;
    double strain_rate_exp_fac = 0.1;
    double init_yield_strength = 2.0e4;
    double isotrop_harden_prefac = 5.0e3;
    double isotrop_harden_exp = 0.2;
    double ref_temperature = 293.0;
    double melt_temperature = 1793.0;
    double temperature_sens = 1.03;
  };

  struct ViscoplasticMaterialSetup
  {
    ViscoplastUtils::MatBehavior mat_behavior = ViscoplastUtils::MatBehavior::isotropic;
    ViscoplastUtils::LocalNewtonParams local_newton_params{
        .res_tol = 1.0e-8,
        .incr_tol = 1.0e-8,
        .conv_check = ViscoplastUtils::LocalNewtonConvCheck::residual_and_increment_ratio,
        .diver_cont = ViscoplastUtils::LocalNewtonDiverCont::stop,
        .max_iter = 100,
        .max_exceedance_fact_res_tol = 1.0e1,
        .max_exceedance_fact_incr_tol = 1.0e1,
    };
    bool use_substepping = false;
    unsigned int max_substepping_halve_num = 0;
    ViscoplastUtils::LinearizationType linearization_type =
        ViscoplastUtils::LinearizationType::analytic;
    std::optional<double> yield_cond_a = 1.0;
    std::optional<double> yield_cond_b = 2.0;
    std::optional<double> yield_cond_f = 2.5;
    int numgp = 1;
    double taylor_quinney_coefficient = 0.0;
    double thermal_expansion_coefficient = 0.1;
    double ref_temperature = 293.0;
    ReformulatedJohnsonCookParameters viscoplastic_law_params{};
  };

  struct ViscoplasticTestMaterial
  {
    std::shared_ptr<Mat::InelasticDefgradTransvIsotropElastViscoplast> material;
    // Keep these params alive because InelasticDefgradTransvIsotropElastViscoplast stores them as
    // a raw pointer.
    std::shared_ptr<Mat::PAR::InelasticDefgradTransvIsotropElastViscoplast> params;
  };

  static constexpr int problem_id = 0;

  int make_unique_viscoplastic_law_id()
  {
    static int next_viscoplastic_law_id = 401;
    return next_viscoplastic_law_id++;
  }

  int make_unique_structural_tensor_strategy_id()
  {
    static int next_structural_tensor_strategy_id = 100;
    return next_structural_tensor_strategy_id++;
  }

  static void expect_near_relative(
      const double actual, const double expected, const double rel_tol, const double abs_floor)
  {
    const double scale = std::max(std::abs(actual), std::abs(expected));
    EXPECT_LE(std::abs(actual - expected), abs_floor + rel_tol * scale)
        << "actual: " << actual << ", expected: " << expected;
  }

  static void expect_near_relative(const Core::LinAlg::Matrix<1, 6>& actual,
      const Core::LinAlg::Matrix<1, 6>& expected, const double rel_tol, const double abs_floor)
  {
    for (int i = 0; i < 6; ++i)
    {
      const double scale = std::max(std::abs(actual(0, i)), std::abs(expected(0, i)));
      EXPECT_LE(std::abs(actual(0, i) - expected(0, i)), abs_floor + rel_tol * scale)
          << "entry (" << i << ",0), actual: " << actual(0, i) << ", expected: " << expected(0, i);
    }
  }

  template <unsigned int rows, unsigned int cols>
  static void expect_near_relative(const Core::LinAlg::Matrix<rows, cols>& actual,
      const Core::LinAlg::Matrix<rows, cols>& expected, const double rel_tol,
      const double abs_floor)
  {
    for (unsigned int row = 0; row < rows; ++row)
    {
      for (unsigned int col = 0; col < cols; ++col)
      {
        const double scale = std::max(std::abs(actual(row, col)), std::abs(expected(row, col)));
        EXPECT_LE(std::abs(actual(row, col) - expected(row, col)), abs_floor + rel_tol * scale)
            << "entry (" << row << "," << col << "), actual: " << actual(row, col)
            << ", expected: " << expected(row, col);
      }
    }
  }

  ViscoplasticTestMaterial set_up_viscoplastic_material(const ViscoplasticMaterialSetup& setup = {})
  {
    Global::Problem& problem = (*Global::Problem::instance(problem_id));
    const int viscoplastic_law_id = make_unique_viscoplastic_law_id();
    const int structural_tensor_strategy_id = make_unique_structural_tensor_strategy_id();

    Core::IO::InputParameterContainer material_data;
    material_data.add("FIBER_READER_ID", 5);
    material_data.add("TAYLOR_QUINNEY_COEFFICIENT", setup.taylor_quinney_coefficient);
    material_data.add("LINEARIZATION", setup.linearization_type);
    material_data.add("MATRIX_EXP_CALC_METHOD", Core::LinAlg::MatrixExpCalcMethod::automatic);
    material_data.add(
        "MATRIX_EXP_DERIV_CALC_METHOD", Core::LinAlg::GenMatrixExpFirstDerivCalcMethod::automatic);
    material_data.add("MATRIX_LOG_CALC_METHOD", Core::LinAlg::MatrixLogCalcMethod::inv_scal_square);
    material_data.add("MATRIX_LOG_DERIV_CALC_METHOD",
        Core::LinAlg::GenMatrixLogFirstDerivCalcMethod::pade_part_fract);
    material_data.add("MAT_BEHAVIOR", setup.mat_behavior);
    material_data.add("MAX_PLASTIC_STRAIN_DERIV_INCR", std::exp(30.0));
    material_data.add("MAX_PLASTIC_STRAIN_INCR", std::exp(30.0));
    material_data.add("TIME_INTEGRATION_HIST_VARS",
        Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::TimIntType::logarithmic);
    material_data.add("VISCOPLAST_LAW_ID", viscoplastic_law_id);
    material_data.add<std::optional<double>>("YIELD_COND_A", setup.yield_cond_a);
    material_data.add<std::optional<double>>("YIELD_COND_B", setup.yield_cond_b);
    material_data.add<std::optional<double>>("YIELD_COND_F", setup.yield_cond_f);
    material_data.group("LOCAL_SUBSTEPPING").add("USE_SUBSTEPPING", setup.use_substepping);
    material_data.group("LOCAL_SUBSTEPPING")
        .add("MAX_SUBSTEPPING_HALVE_NUM", static_cast<int>(setup.max_substepping_halve_num));
    material_data.group("LOCAL_NEWTON").add("CONV_CHECK", setup.local_newton_params.conv_check);
    material_data.group("LOCAL_NEWTON").add("DIVER_CONT", setup.local_newton_params.diver_cont);
    material_data.group("LOCAL_NEWTON").add("INCR_TOL", setup.local_newton_params.incr_tol);
    material_data.group("LOCAL_NEWTON").add("RES_TOL", setup.local_newton_params.res_tol);
    material_data.group("LOCAL_NEWTON")
        .add("MAX_ITER", static_cast<int>(setup.local_newton_params.max_iter));
    material_data.group("LOCAL_NEWTON")
        .add("MAX_EXCEEDANCE_FACT_RES_TOL", setup.local_newton_params.max_exceedance_fact_res_tol);
    material_data.group("LOCAL_NEWTON")
        .add(
            "MAX_EXCEEDANCE_FACT_INCR_TOL", setup.local_newton_params.max_exceedance_fact_incr_tol);

    auto material_params =
        std::dynamic_pointer_cast<Mat::PAR::InelasticDefgradTransvIsotropElastViscoplast>(
            std::shared_ptr(Mat::make_parameter(1,
                Core::Materials::MaterialType::mfi_transv_isotrop_elast_viscoplast,
                material_data)));

    Core::IO::InputParameterContainer fiber_reader_data;
    fiber_reader_data.add("ALPHA", 1.0);
    fiber_reader_data.add("BETA", 1.0);
    fiber_reader_data.add("GAMMA", 1.0);
    fiber_reader_data.add("ANGLE", 0.0);
    fiber_reader_data.add("STR_TENS_ID", structural_tensor_strategy_id);
    fiber_reader_data.add<std::string>("STRATEGY", "Standard");
    fiber_reader_data.add<std::string>("DISTR", "none");
    fiber_reader_data.add("C1", 1.0);
    fiber_reader_data.add("C2", 0.0);
    fiber_reader_data.add("C3", 0.0);
    fiber_reader_data.add("C4", 1.0e16);
    fiber_reader_data.add("FIBER", 1);
    fiber_reader_data.add("INIT", 1);
    problem.materials()->insert(structural_tensor_strategy_id,
        Mat::make_parameter(structural_tensor_strategy_id,
            Core::Materials::MaterialType::mes_structuraltensorstratgy, fiber_reader_data));

    auto fiber_reader_params =
        std::dynamic_pointer_cast<Mat::Elastic::PAR::CoupTransverselyIsotropic>(
            std::shared_ptr(Mat::make_parameter(1,
                Core::Materials::MaterialType::mes_couptransverselyisotropic, fiber_reader_data)));
    Mat::Elastic::CoupTransverselyIsotropic fiber_reader{fiber_reader_params.get()};

    Core::IO::InputParameterContainer viscoplastic_law_data;
    viscoplastic_law_data.add(
        "STRAIN_RATE_PREFAC", setup.viscoplastic_law_params.strain_rate_prefac);
    viscoplastic_law_data.add(
        "STRAIN_RATE_EXP_FAC", setup.viscoplastic_law_params.strain_rate_exp_fac);
    viscoplastic_law_data.add(
        "INIT_YIELD_STRENGTH", setup.viscoplastic_law_params.init_yield_strength);
    viscoplastic_law_data.add(
        "ISOTROP_HARDEN_PREFAC", setup.viscoplastic_law_params.isotrop_harden_prefac);
    viscoplastic_law_data.add(
        "ISOTROP_HARDEN_EXP", setup.viscoplastic_law_params.isotrop_harden_exp);
    viscoplastic_law_data.add("REF_TEMPERATURE", setup.viscoplastic_law_params.ref_temperature);
    viscoplastic_law_data.add("MELT_TEMPERATURE", setup.viscoplastic_law_params.melt_temperature);
    viscoplastic_law_data.add("TEMPERATURE_SENS", setup.viscoplastic_law_params.temperature_sens);
    problem.materials()->insert(viscoplastic_law_id,
        Mat::make_parameter(viscoplastic_law_id,
            Core::Materials::MaterialType::mvl_reformulated_Johnson_Cook, viscoplastic_law_data));

    auto viscoplastic_law = std::make_shared<Mat::Viscoplastic::ReformulatedJohnsonCook>(
        problem.materials()->parameter_by_id(viscoplastic_law_id));

    std::vector<std::shared_ptr<Mat::Elastic::Summand>> pot_sum_el;
    pot_sum_el.emplace_back(Mat::Elastic::Summand::factory(200));
    std::vector<std::shared_ptr<Mat::Elastic::CoupTransverselyIsotropic>> pot_sum_el_transv_iso;

    // The material stores its parameter object as a raw pointer. Capture the shared parameter
    // object in the deleter so callers may use the returned material directly when they do not
    // need to inspect the params.
    auto viscoplastic_material = std::shared_ptr<Mat::InelasticDefgradTransvIsotropElastViscoplast>(
        new Mat::InelasticDefgradTransvIsotropElastViscoplast(material_params.get(),
            viscoplastic_law, fiber_reader, pot_sum_el, pot_sum_el_transv_iso,
            setup.thermal_expansion_coefficient, setup.ref_temperature),
        [material_params](Mat::InelasticDefgradTransvIsotropElastViscoplast* material)
        { delete material; });

    Discret::Elements::Fibers fibers;
    fibers.element_fibers.emplace_back(Core::LinAlg::Tensor<double, 3>{{0.0, 0.0, 1.0}});
    viscoplastic_material->setup(setup.numgp, fibers, {});
    return {.material = viscoplastic_material, .params = material_params};
  }

  struct ThermoViscoplastLinearizationComparison
  {
    ViscoplasticTestMaterial analytic_material;
    ViscoplasticTestMaterial fd_material;
    double temperature = 313.0;
    double total_time = 1.0e-6;
    double time_step_size = 1.0e-6;
    Mat::EvaluationContext<3> context{.total_time = &total_time,
        .time_step_size = &time_step_size,
        .xi = {},
        .ref_coords = nullptr};
    Core::LinAlg::Matrix<3, 3> FM{Core::LinAlg::Initialization::zero};
    Core::LinAlg::Matrix<3, 3> iFin_analytic{Core::LinAlg::Initialization::zero};
    Core::LinAlg::Matrix<3, 3> iFin_fd{Core::LinAlg::Initialization::zero};
    Core::LinAlg::Matrix<6, 1> iCV{Core::LinAlg::Initialization::zero};
  };

  ThermoViscoplastLinearizationComparison set_up_thermo_viscoplast_linearization_comparison()
  {
    const ReformulatedJohnsonCookParameters viscoplastic_law_params{
        .strain_rate_prefac = 1.0,
        .strain_rate_exp_fac = 0.014,
        .init_yield_strength = 792.0,
        .isotrop_harden_prefac = 510.0,
        .isotrop_harden_exp = 0.26,
        .ref_temperature = 293.0,
        .melt_temperature = 1793.0,
        .temperature_sens = 1.03,
    };
    const ViscoplastUtils::LocalNewtonParams local_newton_params{
        .res_tol = 1.0e-14,
        .incr_tol = 1.0e-14,
        .conv_check = ViscoplastUtils::LocalNewtonConvCheck::residual_and_increment_ratio,
        .diver_cont = ViscoplastUtils::LocalNewtonDiverCont::stop,
        .max_iter = 100,
        .max_exceedance_fact_res_tol = 1.0e1,
        .max_exceedance_fact_incr_tol = 1.0e1,
    };

    ThermoViscoplastLinearizationComparison comparison{
        .analytic_material =
            set_up_viscoplastic_material({.local_newton_params = local_newton_params,
                .linearization_type = ViscoplastUtils::LinearizationType::analytic,
                .taylor_quinney_coefficient = 0.85,
                .viscoplastic_law_params = viscoplastic_law_params}),
        .fd_material = set_up_viscoplastic_material({.local_newton_params = local_newton_params,
            .linearization_type = ViscoplastUtils::LinearizationType::perturbation_based,
            .taylor_quinney_coefficient = 0.85,
            .viscoplastic_law_params = viscoplastic_law_params})};

    Teuchos::ParameterList param_list;
    param_list.set<double>("temperature", comparison.temperature);
    comparison.analytic_material.material->pre_evaluate(param_list, comparison.context, 0, 0);
    comparison.fd_material.material->pre_evaluate(param_list, comparison.context, 0, 0);

    comparison.FM(0, 0) = 1.55;
    comparison.FM(1, 1) = 1.0;
    comparison.FM(2, 2) = 1.0;
    comparison.FM(0, 1) = 0.15;

    comparison.analytic_material.material->evaluate_inverse_inelastic_def_grad(
        &comparison.FM, Core::LinAlg::identity_matrix<3>(), comparison.iFin_analytic);
    comparison.fd_material.material->evaluate_inverse_inelastic_def_grad(
        &comparison.FM, Core::LinAlg::identity_matrix<3>(), comparison.iFin_fd);

    Core::LinAlg::Matrix<3, 3> CM(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 3> iCM(Core::LinAlg::Initialization::zero);
    CM.multiply_tn(1.0, comparison.FM, comparison.FM, 0.0);
    iCM.invert(CM);
    Core::LinAlg::Voigt::Stresses::matrix_to_vector(iCM, comparison.iCV);

    return comparison;
  }


  class InelasticDefgradFactorsTest : public ::testing::Test
  {
   protected:
    void SetUp() override
    {
      Global::Problem& problem = (*Global::Problem::instance(problem_id));
      Global::Problem::instance()->materials()->set_read_from_problem(problem_id);

      // clang-format off
      // set up the deformation gradient
      FM_(0, 0) = 1.1;  FM_(0, 1) = 0.01; FM_(0, 2) = 0.03;
      FM_(1, 0) = 0.04; FM_(1, 1) = 1.2;  FM_(1, 2) = 0.02;
      FM_(2, 0) = 0.06; FM_(2, 1) = 0.05; FM_(2, 2) = 1.3;

      // set up the derivative of the second Piola-Kirchhoff stress tensor
      dSdiFin_(0, 0) =  9.83e+03; dSdiFin_(0, 1) =  2.15e+03; dSdiFin_(0, 2) =  2.14e+03; dSdiFin_(0, 3) = -5.49e+00; dSdiFin_(0, 4) =  6.38e-01; dSdiFin_(0, 5) = -3.29e+00; dSdiFin_(0, 6) =  2.13e+00; dSdiFin_(0, 7) =  6.38e-01; dSdiFin_(0, 8) =  1.28e+00;
      dSdiFin_(1, 0) =  1.80e+03; dSdiFin_(1, 1) =  9.49e+03; dSdiFin_(1, 2) =  1.80e+03; dSdiFin_(1, 3) =  1.79e+00; dSdiFin_(1, 4) = -1.75e+00; dSdiFin_(1, 5) =  1.07e+00; dSdiFin_(1, 6) = -5.83e+00; dSdiFin_(1, 7) =  5.36e-01; dSdiFin_(1, 8) =  1.07e+00;
      dSdiFin_(2, 0) =  1.55e+03; dSdiFin_(2, 1) =  1.54e+03; dSdiFin_(2, 2) =  9.23e+03; dSdiFin_(2, 3) =  1.53e+00; dSdiFin_(2, 4) =  4.59e-01; dSdiFin_(2, 5) =  9.18e-01; dSdiFin_(2, 6) =  1.53e+00; dSdiFin_(2, 7) = -1.82e+00; dSdiFin_(2, 8) = -3.65e+00;
      dSdiFin_(3, 0) = -8.76e+01; dSdiFin_(3, 1) = -8.75e+01; dSdiFin_(3, 2) = -8.37e+01; dSdiFin_(3, 3) =  3.84e+03; dSdiFin_(3, 4) = -2.31e+00; dSdiFin_(3, 5) = -1.19e+00; dSdiFin_(3, 6) =  3.84e+03; dSdiFin_(3, 7) = -2.49e-02; dSdiFin_(3, 8) = -4.98e-02;
      dSdiFin_(4, 0) = -8.97e+01; dSdiFin_(4, 1) = -9.08e+01; dSdiFin_(4, 2) = -9.07e+01; dSdiFin_(4, 3) = -8.89e-02; dSdiFin_(4, 4) =  3.85e+03; dSdiFin_(4, 5) = -5.33e-02; dSdiFin_(4, 6) = -2.37e+00; dSdiFin_(4, 7) =  3.84e+03; dSdiFin_(4, 8) = -3.86e+00;
      dSdiFin_(5, 0) = -1.40e+02; dSdiFin_(5, 1) = -1.37e+02; dSdiFin_(5, 2) = -1.40e+02; dSdiFin_(5, 3) = -1.28e+00; dSdiFin_(5, 4) = -4.09e-02; dSdiFin_(5, 5) =  3.85e+03; dSdiFin_(5, 6) = -1.36e-01; dSdiFin_(5, 7) = -3.85e+00; dSdiFin_(5, 8) =  3.84e+03;

      // set up reference solution of inverse inelastic deformation gradient of InelasticDefgradLinScalarIso
      iFin_lin_scalar_iso_solution_.clear();
      iFin_lin_scalar_iso_solution_(0, 0) = iFin_lin_scalar_iso_solution_(1, 1) = iFin_lin_scalar_iso_solution_(2, 2) = 0.9948803003804159;

      // set up reference solution of inverse inelastic deformation gradient of InelasticDefgradLinScalarAniso
      iFin_lin_scalar_aniso_solution_(0, 0) =  0.9885965678505413; iFin_lin_scalar_aniso_solution_(0, 1) = -0.0057017160747293; iFin_lin_scalar_aniso_solution_(0, 2) = -0.0034210296448376;
      iFin_lin_scalar_aniso_solution_(1, 0) = -0.0057017160747293; iFin_lin_scalar_aniso_solution_(1, 1) =  0.9971491419626352; iFin_lin_scalar_aniso_solution_(1, 2) = -0.0017105148224188;
      iFin_lin_scalar_aniso_solution_(2, 0) = -0.0034210296448376; iFin_lin_scalar_aniso_solution_(2, 1) = -0.0017105148224188; iFin_lin_scalar_aniso_solution_(2, 2) =  0.9989736911065487;

      // set up reference solution of inverse inelastic deformation gradient of InelasticDefgradPolyIntercalFracIso
      iFin_poly_intercal_frac_iso_solution_.clear();
      iFin_poly_intercal_frac_iso_solution_(0, 0) = iFin_poly_intercal_frac_iso_solution_(1, 1) = iFin_poly_intercal_frac_iso_solution_(2, 2) = 0.9991151119224016;

      // set up reference solution of inverse inelastic deformation gradient of InelasticDefgradPolyIntercalFracAniso
      iFin_poly_intercal_frac_aniso_solution_(0, 0) =  0.9980206598171963; iFin_poly_intercal_frac_aniso_solution_(0, 1) = -0.0009896700914018; iFin_poly_intercal_frac_aniso_solution_(0, 2) = -0.0005938020548410;
      iFin_poly_intercal_frac_aniso_solution_(1, 0) = -0.0009896700914018; iFin_poly_intercal_frac_aniso_solution_(1, 1) =  0.9995051649542991; iFin_poly_intercal_frac_aniso_solution_(1, 2) = -0.0002969010274205;
      iFin_poly_intercal_frac_aniso_solution_(2, 0) = -0.0005938020548410; iFin_poly_intercal_frac_aniso_solution_(2, 1) = -0.0002969010274205; iFin_poly_intercal_frac_aniso_solution_(2, 2) =  0.9998218593835476;

      // set up reference solution of inverse inelastic deformation gradient of InelasticDefgradLinTemIso
      iFin_lin_temp_iso_solution_.clear();
      iFin_lin_temp_iso_solution_(0, 0) = iFin_lin_temp_iso_solution_(1, 1) = iFin_lin_temp_iso_solution_(2, 2) = 1.006073023359708;

      // set up reference solution of inverse inelastic deformation gradient and plastic strain of InelasticDefgradTransvIsotropElastViscoplast
      iFin_transv_isotrop_vplast_refJC_solution_.clear();
      iFin_transv_isotrop_vplast_refJC_solution_(0,0)= 0.988; iFin_transv_isotrop_vplast_refJC_solution_(0, 1) = 0.006; iFin_transv_isotrop_vplast_refJC_solution_(0, 2) = 0.008;
      iFin_transv_isotrop_vplast_refJC_solution_(1,0)= -0.005; iFin_transv_isotrop_vplast_refJC_solution_(1, 1) = 0.997; iFin_transv_isotrop_vplast_refJC_solution_(1, 2) = -0.004;
      iFin_transv_isotrop_vplast_refJC_solution_(2,0)= -0.003; iFin_transv_isotrop_vplast_refJC_solution_(2, 1) = -0.002; iFin_transv_isotrop_vplast_refJC_solution_(2, 2) = 0.999;
      plastic_strain_transv_isotrop_vplast_refJC_solution_ = 1.1;


      // clang-format on

      // prepare variables needed to instantiate linear shape evaluation object
      const double growth_fac(5.27e-7);
      const double ref_conc(46456.0);

      // create linear shape evaluation object
      linear_shape_ = std::make_shared<Mat::InelasticDefgradLinearShape>(growth_fac, ref_conc);

      // prepare variables needed to instantiate polynomial shape evaluation object
      std::vector<double> poly_coeffs{0.1051717305, -3.9012322937, 31.9658107225, -122.8624633232,
          258.6769103514, -306.7800791732, 192.5096604774, -49.7490196448};
      const double x_min(0.152);
      const double x_max(0.887);

      // create polynomial shape evaluation object
      polynomial_shape_ =
          std::make_shared<Mat::InelasticDefgradPolynomialShape>(poly_coeffs, x_min, x_max);

      // parameter list to be passed to pre_evaluate
      Teuchos::ParameterList params_lin;
      // set up a dummy concentration vector and store it to the parameter list
      auto gpconc_lin = std::make_shared<std::vector<double>>(std::vector<double>({44327.362}));
      params_lin.set<std::shared_ptr<std::vector<double>>>("scalars", gpconc_lin);

      // create InelasticDefgradLinScalarIso object initialize container for material parameters
      Core::IO::InputParameterContainer inelastic_defgrad_scalar_data;
      inelastic_defgrad_scalar_data.add("SCALAR1", 1);
      inelastic_defgrad_scalar_data.add("SCALAR1_MolarGrowthFac", growth_fac);
      inelastic_defgrad_scalar_data.add("SCALAR1_RefConc", ref_conc);

      params_inelastic_defgrad_lin_scalar_iso_ =
          std::dynamic_pointer_cast<Mat::PAR::InelasticDefgradLinScalar>(std::shared_ptr(
              Mat::make_parameter(1, Core::Materials::MaterialType::mfi_lin_scalar_iso,
                  inelastic_defgrad_scalar_data)));

      // setup pointer to InelasticDefgradLinScalarIso object
      lin_scalar_iso_ = std::make_shared<Mat::InelasticDefgradLinScalarIso>(
          params_inelastic_defgrad_lin_scalar_iso_.get());

      // call pre_evaluate to set the concentration value
      lin_scalar_iso_->pre_evaluate(params_lin, {}, 0, 0);

      // create InelasticDefgradLinScalarAniso object initialize container for material parameters
      Core::IO::InputParameterContainer inelastic_defgrad_lin_scalar_aniso_data;
      inelastic_defgrad_lin_scalar_aniso_data.add("SCALAR1", 1);
      inelastic_defgrad_lin_scalar_aniso_data.add("SCALAR1_MolarGrowthFac", growth_fac);
      inelastic_defgrad_lin_scalar_aniso_data.add("SCALAR1_RefConc", ref_conc);

      // vector to instantiate the deformation direction object
      const std::vector<double> growthdir{1.0, 0.5, 0.3};
      inelastic_defgrad_lin_scalar_aniso_data.add("GrowthDirection", growthdir);

      params_inelastic_defgrad_lin_scalar_aniso_ =
          std::dynamic_pointer_cast<Mat::PAR::InelasticDefgradLinScalarAniso>(std::shared_ptr(
              Mat::make_parameter(1, Core::Materials::MaterialType::mfi_lin_scalar_aniso,
                  inelastic_defgrad_lin_scalar_aniso_data)));

      // set up pointer to InelasticDefgradLinScalarAniso object
      lin_scalar_aniso_ = std::make_shared<Mat::InelasticDefgradLinScalarAniso>(
          params_inelastic_defgrad_lin_scalar_aniso_.get());

      // call pre_evaluate to set the concentration value
      lin_scalar_aniso_->pre_evaluate(params_lin, {}, 0, 0);

      // InelasticDefgradPolyIntercalFracIso object initialize container for required electrode
      // material parameters
      // set up material to be added to problem instance
      const int matid(1);

      Core::IO::InputParameterContainer electrode_data;
      // add dummy parameters to electrode material
      electrode_data.add("DIFF_COEF", 1.0);
      electrode_data.add("DIFF_COEF_CONC_SCALE_FUNCT", std::optional<int>(std::nullopt));
      electrode_data.add("DIFF_COEF_TEMP_SCALE_FUNCT", std::optional<int>(std::nullopt));
      electrode_data.add("COND", 1.0);
      electrode_data.add("COND_CONC_SCALE_FUNCT", std::optional<int>(std::nullopt));
      electrode_data.add("COND_TEMP_SCALE_FUNCT", std::optional<int>(std::nullopt));
      // create and add the ocp model
      auto& ocp_model = electrode_data.group("OCP_MODEL");
      // choose the ocp model
      ocp_model.add("OCP_MODEL", Mat::PAR::OCPModels::function);
      // create the chosen ocp model 'Function'
      auto& ocp_function = ocp_model.group("Function");
      ocp_function.add("OCP_FUNCT_NUM", 1);
      // add the ocp model
      electrode_data.add("OCP_MODEL", ocp_model);

      // Set up parameters in global problem instance
      auto params = std::make_shared<Teuchos::ParameterList>();
      params->sublist("ELCH CONTROL").set("GAS_CONSTANT", 8.314472);
      problem.set_parameter_list(params);

      // add actually required parameters to electrode material
      const double c_max(4.91375e4);
      const double chi_max(1.0);
      electrode_data.add("C_MAX", c_max);
      electrode_data.add("CHI_MAX", chi_max);

      // add material to problem instance
      problem.materials()->insert(matid,
          Mat::make_parameter(1, Core::Materials::MaterialType::m_electrode, electrode_data));

      // parameter list to be passed to pre_evaluate
      Teuchos::ParameterList params_poly;
      // set up a dummy concentration vector and store it to the parameter list
      auto gpconc_poly = std::make_shared<std::vector<double>>(std::vector<double>({22641.893}));
      params_poly.set<std::shared_ptr<std::vector<double>>>("scalars", gpconc_poly);

      // initialize container for material parameters
      Core::IO::InputParameterContainer inelastic_defgrad_poly_intercal_frac_data;

      inelastic_defgrad_poly_intercal_frac_data.add("MATID", matid);
      inelastic_defgrad_poly_intercal_frac_data.add("SCALAR1", 1);
      inelastic_defgrad_poly_intercal_frac_data.add("SCALAR1_RefConc", ref_conc);
      inelastic_defgrad_poly_intercal_frac_data.add("POLY_PARAMS", poly_coeffs);
      inelastic_defgrad_poly_intercal_frac_data.add("X_max", x_max);
      inelastic_defgrad_poly_intercal_frac_data.add("X_min", x_min);
      inelastic_defgrad_poly_intercal_frac_data.add(
          "POLY_PARA_NUM", static_cast<int>(poly_coeffs.size()));

      // get pointer to parameter class
      params_inelastic_defgrad_poly_intercal_frac_ =
          std::dynamic_pointer_cast<Mat::PAR::InelasticDefgradPolyIntercalFrac>(std::shared_ptr(
              Mat::make_parameter(1, Core::Materials::MaterialType::mfi_poly_intercal_frac_iso,
                  inelastic_defgrad_poly_intercal_frac_data)));

      // get reference intercalation fraction
      const double x_ref =
          Mat::Electrode::compute_intercalation_fraction(ref_conc, chi_max, c_max, 1.0);

      // set the value of the reference polynomial
      params_inelastic_defgrad_poly_intercal_frac_->set_polynom_reference_value(
          polynomial_shape_->compute_polynomial(x_ref));

      // set up pointer to InelasticDefgradPolyIntercalFracIso object
      poly_intercal_frac_iso_ = std::make_shared<Mat::InelasticDefgradPolyIntercalFracIso>(
          params_inelastic_defgrad_poly_intercal_frac_.get());

      // call pre_evaluate to set the concentration value
      poly_intercal_frac_iso_->pre_evaluate(params_poly, {}, 0, 0);

      // create InelasticDefgradPolyIntercalFracAniso object initialize container for material
      // parameters
      Core::IO::InputParameterContainer inelastic_defgrad_poly_intercal_frac_aniso_data;
      inelastic_defgrad_poly_intercal_frac_aniso_data.add("MATID", matid);
      inelastic_defgrad_poly_intercal_frac_aniso_data.add("SCALAR1", 1);
      inelastic_defgrad_poly_intercal_frac_aniso_data.add("SCALAR1_RefConc", ref_conc);
      inelastic_defgrad_poly_intercal_frac_aniso_data.add("GrowthDirection", growthdir);
      inelastic_defgrad_poly_intercal_frac_aniso_data.add("POLY_PARAMS", poly_coeffs);
      inelastic_defgrad_poly_intercal_frac_aniso_data.add("X_max", x_max);
      inelastic_defgrad_poly_intercal_frac_aniso_data.add("X_min", x_min);
      inelastic_defgrad_poly_intercal_frac_aniso_data.add(
          "POLY_PARA_NUM", static_cast<int>(poly_coeffs.size()));

      // get pointer to parameter class
      params_inelastic_defgrad_poly_intercal_frac_aniso_ =
          std::dynamic_pointer_cast<Mat::PAR::InelasticDefgradPolyIntercalFracAniso>(
              std::shared_ptr(Mat::make_parameter(1,
                  Core::Materials::MaterialType::mfi_poly_intercal_frac_aniso,
                  inelastic_defgrad_poly_intercal_frac_aniso_data)));

      // set the value of the reference polynomial
      params_inelastic_defgrad_poly_intercal_frac_aniso_->set_polynom_reference_value(
          polynomial_shape_->compute_polynomial(x_ref));

      // set up pointer to InelasticDefgradPolyIntercalFracIso object
      poly_intercal_frac_aniso_ = std::make_shared<Mat::InelasticDefgradPolyIntercalFracAniso>(
          params_inelastic_defgrad_poly_intercal_frac_aniso_.get());

      // call pre_evaluate to set the concentration value
      poly_intercal_frac_aniso_->pre_evaluate(params_poly, {}, 0, 0);

      // create InelasticDefgradLinTempIso object initialize container for material parameters
      Core::IO::InputParameterContainer inelastic_defgrad_temp_iso_data;
      inelastic_defgrad_temp_iso_data.add("MATID", matid);
      inelastic_defgrad_temp_iso_data.add("RefTemp", 298.0);
      inelastic_defgrad_temp_iso_data.add("Temp_GrowthFac", 1.0e-3);

      // get pointer to parameter class
      params_lin_temp_iso_ = std::dynamic_pointer_cast<Mat::PAR::InelasticDefgradLinTempIso>(
          std::shared_ptr(Mat::make_parameter(1, Core::Materials::MaterialType::mfi_lin_temp_iso,
              inelastic_defgrad_temp_iso_data)));

      // setup pointer to InelasticDefgradLinScalarIso object
      lin_temp_iso_ = std::make_shared<Mat::InelasticDefgradLinTempIso>(params_lin_temp_iso_.get());

      // parameter list for pre_evaluate call with gp temperature
      Teuchos::ParameterList params_temp{};
      params_temp.set<double>("temperature", 280.0);
      // call pre_evaluate to set the temperature
      lin_temp_iso_->pre_evaluate(params_temp, {}, 0, 0);


      Core::IO::InputParameterContainer elast_pot_coup_neo_hooke_data;
      elast_pot_coup_neo_hooke_data.add("YOUNG", 200.0e3);
      elast_pot_coup_neo_hooke_data.add("NUE", 0.29);
      problem.materials()->insert(
          200, Mat::make_parameter(200, Core::Materials::MaterialType::mes_coupneohooke,
                   elast_pot_coup_neo_hooke_data));
    }

    void set_up_state_quantities_solution()
    {
      // clang-format on
      Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::StateQuantities& transv =
          state_quantities_solution_transv_isotrop_;
      Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::StateQuantities& iso =
          state_quantities_solution_isotrop_;

      // clang-format off
      // curr_CeM_
      transv.curr_CeM(0, 0) = 1.1849888441; transv.curr_CeM(0, 1) = 0.0606037319; transv.curr_CeM(0, 2) = 0.1142177392;
      transv.curr_CeM(1, 0) = 0.0606037319; transv.curr_CeM(1, 1) = 1.4343908522; transv.curr_CeM(1, 2) = 0.0810311701;
      transv.curr_CeM(2, 0) = 0.1142177392; transv.curr_CeM(2, 1) = 0.0810311701; transv.curr_CeM(2, 2) = 1.6890893033;

      iso.curr_CeM = transv.curr_CeM;

      // curr_gamma_, curr_delta_
      transv.curr_gamma(0) = 77519.37984496124; transv.curr_gamma(1) = 0; transv.curr_gamma(2) = -37710.707742398015;
      iso.curr_gamma = transv.curr_gamma;
      transv.curr_delta(0, 0) = 0.0000000000; transv.curr_delta(1, 0) = 0.0000000000; transv.curr_delta(2, 0) = 0.0000000000;
      transv.curr_delta(3, 0) = 0.0000000000; transv.curr_delta(4, 0) = 0.0000000000;  transv.curr_delta(5, 0) = 52076.6916442639;
      transv.curr_delta(6, 0) = 75421.4154847960; transv.curr_delta(7, 0) = 0.0000000000;
      iso.curr_delta = transv.curr_delta;

      // curr_Mtheta_dev_sym_M_
      transv.curr_Mtheta_dev_sym_M(0, 0) = -19470.3479922481; transv.curr_Mtheta_dev_sym_M(0, 1) = 4697.9637131783; transv.curr_Mtheta_dev_sym_M(0, 2) = 8854.0883100775;
      transv.curr_Mtheta_dev_sym_M(1, 0) = 4697.9637131783; transv.curr_Mtheta_dev_sym_M(1, 1) = -136.8589922481; transv.curr_Mtheta_dev_sym_M(1, 2) = 6281.4860542636;
      transv.curr_Mtheta_dev_sym_M(2, 0) = 8854.0883100775; transv.curr_Mtheta_dev_sym_M(2, 1) = 6281.4860542636; transv.curr_Mtheta_dev_sym_M(2, 2) = 19607.2069844961;
      iso.curr_Mtheta_dev_sym_M = transv.curr_Mtheta_dev_sym_M;

      // curr_equiv_stress_
      transv.curr_equiv_stress = 58944.20584626558;
      iso.curr_equiv_stress = 39561.370682681154;

      // curr_equiv_plastic_strain_rate_
      transv.curr_equiv_plastic_strain_rate =  720196.224582188;
      iso.curr_equiv_plastic_strain_rate =  317.579965736033;

      // curr_NpM_
      transv.curr_NpM(0, 0) = -1.3189512330; transv.curr_NpM(0, 1) = 0.3985093739; transv.curr_NpM(0, 2) = 0.3755283570;
      transv.curr_NpM(1, 0) = 0.3985093739; transv.curr_NpM(1, 1) = 0.3210309097; transv.curr_NpM(1, 2) = 0.2664166038;
      transv.curr_NpM(2, 0) = 0.3755283570; transv.curr_NpM(2, 1) = 0.2664166038; transv.curr_NpM(2, 2) = 0.9979203233;

      iso.curr_NpM(0, 0) = -0.7382333191; iso.curr_NpM(0, 1) = 0.1781269316; iso.curr_NpM(0, 2) = 0.3357096136;
      iso.curr_NpM(1, 0) = 0.1781269316; iso.curr_NpM(1, 1) = -0.0051891147; iso.curr_NpM(1, 2) = 0.2381674072;
      iso.curr_NpM(2, 0) = 0.3357096136; iso.curr_NpM(2, 1) = 0.2381674072; iso.curr_NpM(2, 2) = 0.7434224338;

      // curr_dpM_
      transv.curr_dpM(0, 0) =  -949903.69838536996; transv.curr_dpM(0, 1) = 287004.94653194677; transv.curr_dpM(0, 2) = 270454.10492365930;
      transv.curr_dpM(1, 0) = 287004.94653194677; transv.curr_dpM(1, 1) = 231205.24912650112; transv.curr_dpM(1, 2) = 191872.23222776141;
      transv.curr_dpM(2, 0) = 270454.10492365930; transv.curr_dpM(2, 1) = 191872.23222776141; transv.curr_dpM(2, 2) = 718698.44925886812;

      iso.curr_dpM(0, 0) = -234.4481121942; iso.curr_dpM(0, 1) = 56.5695448356; iso.curr_dpM(0, 2) = 106.6146475823;
      iso.curr_dpM(1, 0) = 56.5695448356; iso.curr_dpM(1, 1) = -1.6479588543; iso.curr_dpM(1, 2) = 75.6371970230;
      iso.curr_dpM(2, 0) = 106.6146475823; iso.curr_dpM(2, 1) = 75.6371970230; iso.curr_dpM(2, 2) = 236.0960710485;

      // curr_lpM_
      transv.curr_lpM(0, 0) = -949903.69838536996; transv.curr_lpM(0, 1) = 287004.94653194677; transv.curr_lpM(0, 2) = 0.0000000000;
      transv.curr_lpM(1, 0) = 287004.94653194677; transv.curr_lpM(1, 1) = 231205.24912650112; transv.curr_lpM(1, 2) = 0.0000000000;
      transv.curr_lpM(2, 0) = 540908.20984731859; transv.curr_lpM(2, 1) = 383744.46445552283; transv.curr_lpM(2, 2) = 718698.44925886812;

      iso.curr_lpM = iso.curr_dpM;
    }

    void set_up_state_quantity_derivatives_solution(){
      // clang-format on
      Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::StateQuantityDerivatives& transv =
          state_quantity_derivatives_solution_transv_isotrop_;
      Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::StateQuantityDerivatives& iso =
          state_quantity_derivatives_solution_isotrop_;

      // curr_dCediFin_
      transv.curr_dCediFin(0, 0) = 2.3999444000;
      transv.curr_dCediFin(0, 1) = 0.0000000000;
      transv.curr_dCediFin(0, 2) = 0.0000000000;
      transv.curr_dCediFin(0, 3) = 0.0000000000;
      transv.curr_dCediFin(0, 4) = 0.0000000000;
      transv.curr_dCediFin(0, 5) = 0.0000000000;
      transv.curr_dCediFin(0, 6) = 0.1075502000;
      transv.curr_dCediFin(0, 7) = 0.0000000000;
      transv.curr_dCediFin(0, 8) = 0.2098760000;
      transv.curr_dCediFin(1, 0) = 0.0000000000;
      transv.curr_dCediFin(1, 1) = 2.8769312000;
      transv.curr_dCediFin(1, 2) = 0.0000000000;
      transv.curr_dCediFin(1, 3) = 0.1377632000;
      transv.curr_dCediFin(1, 4) = 0.0000000000;
      transv.curr_dCediFin(1, 5) = 0.0000000000;
      transv.curr_dCediFin(1, 6) = 0.0000000000;
      transv.curr_dCediFin(1, 7) = 0.1726406000;
      transv.curr_dCediFin(1, 8) = 0.0000000000;
      transv.curr_dCediFin(2, 0) = 0.0000000000;
      transv.curr_dCediFin(2, 1) = 0.0000000000;
      transv.curr_dCediFin(2, 2) = 3.3802918000;
      transv.curr_dCediFin(2, 3) = 0.0000000000;
      transv.curr_dCediFin(2, 4) = 0.1678726000;
      transv.curr_dCediFin(2, 5) = 0.2423236000;
      transv.curr_dCediFin(2, 6) = 0.0000000000;
      transv.curr_dCediFin(2, 7) = 0.0000000000;
      transv.curr_dCediFin(2, 8) = 0.0000000000;
      transv.curr_dCediFin(3, 0) = 0.0688816000;
      transv.curr_dCediFin(3, 1) = 0.0537751000;
      transv.curr_dCediFin(3, 2) = 0.0000000000;
      transv.curr_dCediFin(3, 3) = 1.1999722000;
      transv.curr_dCediFin(3, 4) = 0.0000000000;
      transv.curr_dCediFin(3, 5) = 0.0000000000;
      transv.curr_dCediFin(3, 6) = 1.4384656000;
      transv.curr_dCediFin(3, 7) = 0.1049380000;
      transv.curr_dCediFin(3, 8) = 0.0863203000;
      transv.curr_dCediFin(4, 0) = 0.0000000000;
      transv.curr_dCediFin(4, 1) = 0.0839363000;
      transv.curr_dCediFin(4, 2) = 0.0863203000;
      transv.curr_dCediFin(4, 3) = 0.1211618000;
      transv.curr_dCediFin(4, 4) = 1.4384656000;
      transv.curr_dCediFin(4, 5) = 0.0688816000;
      transv.curr_dCediFin(4, 6) = 0.0000000000;
      transv.curr_dCediFin(4, 7) = 1.6901459000;
      transv.curr_dCediFin(4, 8) = 0.0000000000;
      transv.curr_dCediFin(5, 0) = 0.1211618000;
      transv.curr_dCediFin(5, 1) = 0.0000000000;
      transv.curr_dCediFin(5, 2) = 0.1049380000;
      transv.curr_dCediFin(5, 3) = 0.0000000000;
      transv.curr_dCediFin(5, 4) = 0.0537751000;
      transv.curr_dCediFin(5, 5) = 1.1999722000;
      transv.curr_dCediFin(5, 6) = 0.0839363000;
      transv.curr_dCediFin(5, 7) = 0.0000000000;
      transv.curr_dCediFin(5, 8) = 1.6901459000;

      iso.curr_dCediFin = transv.curr_dCediFin;

      // curr_dCedC_
      transv.curr_dCedC(0, 0) = 0.9761440000;
      transv.curr_dCedC(0, 1) = 0.0000250000;
      transv.curr_dCedC(0, 2) = 0.0000090000;
      transv.curr_dCedC(0, 3) = -0.0049400000;
      transv.curr_dCedC(0, 4) = 0.0000150000;
      transv.curr_dCedC(0, 5) = -0.0029640000;
      transv.curr_dCedC(1, 0) = 0.0000360000;
      transv.curr_dCedC(1, 1) = 0.9940090000;
      transv.curr_dCedC(1, 2) = 0.0000040000;
      transv.curr_dCedC(1, 3) = 0.0059820000;
      transv.curr_dCedC(1, 4) = -0.0019940000;
      transv.curr_dCedC(1, 5) = -0.0000120000;
      transv.curr_dCedC(2, 0) = 0.0000640000;
      transv.curr_dCedC(2, 1) = 0.0000160000;
      transv.curr_dCedC(2, 2) = 0.9980010000;
      transv.curr_dCedC(2, 3) = -0.0000320000;
      transv.curr_dCedC(2, 4) = -0.0039960000;
      transv.curr_dCedC(2, 5) = 0.0079920000;
      transv.curr_dCedC(3, 0) = 0.0059280000;
      transv.curr_dCedC(3, 1) = -0.0049850000;
      transv.curr_dCedC(3, 2) = 0.0000060000;
      transv.curr_dCedC(3, 3) = 0.4925030000;
      transv.curr_dCedC(3, 4) = -0.0014905000;
      transv.curr_dCedC(3, 5) = -0.0009970000;
      transv.curr_dCedC(4, 0) = 0.0000480000;
      transv.curr_dCedC(4, 1) = -0.0039880000;
      transv.curr_dCedC(4, 2) = -0.0019980000;
      transv.curr_dCedC(4, 3) = 0.0039760000;
      transv.curr_dCedC(4, 4) = 0.4980055000;
      transv.curr_dCedC(4, 5) = 0.0029890000;
      transv.curr_dCedC(5, 0) = 0.0079040000;
      transv.curr_dCedC(5, 1) = 0.0000200000;
      transv.curr_dCedC(5, 2) = -0.0029970000;
      transv.curr_dCedC(5, 3) = -0.0019960000;
      transv.curr_dCedC(5, 4) = -0.0024915000;
      transv.curr_dCedC(5, 5) = 0.4934940000;

      iso.curr_dCedC = transv.curr_dCedC;



      // curr_dMtheta_dev_sym_diFin_
      transv.curr_dMtheta_dev_sym_diFin(0, 0) = 124028.1343669251;
      transv.curr_dMtheta_dev_sym_diFin(0, 1) = -74339.3074935401;
      transv.curr_dMtheta_dev_sym_diFin(0, 2) = -87346.0413436692;
      transv.curr_dMtheta_dev_sym_diFin(0, 3) = -3559.7726098191;
      transv.curr_dMtheta_dev_sym_diFin(0, 4) = -4337.7932816537;
      transv.curr_dMtheta_dev_sym_diFin(0, 5) = -6261.5917312662;
      transv.curr_dMtheta_dev_sym_diFin(0, 6) = 5558.1498708010;
      transv.curr_dMtheta_dev_sym_diFin(0, 7) = -4460.9974160207;
      transv.curr_dMtheta_dev_sym_diFin(0, 8) = 10846.3049095607;
      transv.curr_dMtheta_dev_sym_diFin(1, 0) = -62014.0671834625;
      transv.curr_dMtheta_dev_sym_diFin(1, 1) = 148678.6149870801;
      transv.curr_dMtheta_dev_sym_diFin(1, 2) = -87346.0413436692;
      transv.curr_dMtheta_dev_sym_diFin(1, 3) = 7119.5452196382;
      transv.curr_dMtheta_dev_sym_diFin(1, 4) = -4337.7932816537;
      transv.curr_dMtheta_dev_sym_diFin(1, 5) = -6261.5917312662;
      transv.curr_dMtheta_dev_sym_diFin(1, 6) = -2779.0749354005;
      transv.curr_dMtheta_dev_sym_diFin(1, 7) = 8921.9948320413;
      transv.curr_dMtheta_dev_sym_diFin(1, 8) = -5423.1524547804;
      transv.curr_dMtheta_dev_sym_diFin(2, 0) = -62014.0671834625;
      transv.curr_dMtheta_dev_sym_diFin(2, 1) = -74339.3074935401;
      transv.curr_dMtheta_dev_sym_diFin(2, 2) = 174692.0826873385;
      transv.curr_dMtheta_dev_sym_diFin(2, 3) = -3559.7726098191;
      transv.curr_dMtheta_dev_sym_diFin(2, 4) = 8675.5865633075;
      transv.curr_dMtheta_dev_sym_diFin(2, 5) = 12523.1834625323;
      transv.curr_dMtheta_dev_sym_diFin(2, 6) = -2779.0749354005;
      transv.curr_dMtheta_dev_sym_diFin(2, 7) = -4460.9974160207;
      transv.curr_dMtheta_dev_sym_diFin(2, 8) = -5423.1524547804;
      transv.curr_dMtheta_dev_sym_diFin(3, 0) = 5339.6589147287;
      transv.curr_dMtheta_dev_sym_diFin(3, 1) = 4168.6124031008;
      transv.curr_dMtheta_dev_sym_diFin(3, 2) = 0.0000000000;
      transv.curr_dMtheta_dev_sym_diFin(3, 3) = 93021.1007751938;
      transv.curr_dMtheta_dev_sym_diFin(3, 4) = 0.0000000000;
      transv.curr_dMtheta_dev_sym_diFin(3, 5) = 0.0000000000;
      transv.curr_dMtheta_dev_sym_diFin(3, 6) = 111508.9612403101;
      transv.curr_dMtheta_dev_sym_diFin(3, 7) = 8134.7286821705;
      transv.curr_dMtheta_dev_sym_diFin(3, 8) = 6691.4961240310;
      transv.curr_dMtheta_dev_sym_diFin(4, 0) = 0.0000000000;
      transv.curr_dMtheta_dev_sym_diFin(4, 1) = 6506.6899224806;
      transv.curr_dMtheta_dev_sym_diFin(4, 2) = 6691.4961240310;
      transv.curr_dMtheta_dev_sym_diFin(4, 3) = 9392.3875968992;
      transv.curr_dMtheta_dev_sym_diFin(4, 4) = 111508.9612403101;
      transv.curr_dMtheta_dev_sym_diFin(4, 5) = 5339.6589147287;
      transv.curr_dMtheta_dev_sym_diFin(4, 6) = 0.0000000000;
      transv.curr_dMtheta_dev_sym_diFin(4, 7) = 131019.0620155039;
      transv.curr_dMtheta_dev_sym_diFin(4, 8) = 0.0000000000;
      transv.curr_dMtheta_dev_sym_diFin(5, 0) = 9392.3875968992;
      transv.curr_dMtheta_dev_sym_diFin(5, 1) = 0.0000000000;
      transv.curr_dMtheta_dev_sym_diFin(5, 2) = 8134.7286821705;
      transv.curr_dMtheta_dev_sym_diFin(5, 3) = 0.0000000000;
      transv.curr_dMtheta_dev_sym_diFin(5, 4) = 4168.6124031008;
      transv.curr_dMtheta_dev_sym_diFin(5, 5) = 93021.1007751938;
      transv.curr_dMtheta_dev_sym_diFin(5, 6) = 6506.6899224806;
      transv.curr_dMtheta_dev_sym_diFin(5, 7) = 0.0000000000;
      transv.curr_dMtheta_dev_sym_diFin(5, 8) = 131019.0620155039;

      iso.curr_dMtheta_dev_sym_diFin = transv.curr_dMtheta_dev_sym_diFin;


      // curr_dMtheta_dev_sym_dC_
      transv.curr_dMtheta_dev_sym_dC(0, 0) = 50444.1343669251;
      transv.curr_dMtheta_dev_sym_dC(0, 1) = -25684.1085271318;
      transv.curr_dMtheta_dev_sym_dC(0, 2) = -25787.7777777778;
      transv.curr_dMtheta_dev_sym_dC(0, 3) = -409.0439276486;
      transv.curr_dMtheta_dev_sym_dC(0, 4) = 155.5555555556;
      transv.curr_dMtheta_dev_sym_dC(0, 5) = -359.3798449612;
      transv.curr_dMtheta_dev_sym_dC(1, 0) = -25223.1524547804;
      transv.curr_dMtheta_dev_sym_dC(1, 1) = 51368.9147286822;
      transv.curr_dMtheta_dev_sym_dC(1, 2) = -25788.1653746770;
      transv.curr_dMtheta_dev_sym_dC(1, 3) = 437.6227390181;
      transv.curr_dMtheta_dev_sym_dC(1, 4) = -0.1808785530;
      transv.curr_dMtheta_dev_sym_dC(1, 5) = -130.5426356589;
      transv.curr_dMtheta_dev_sym_dC(2, 0) = -25220.9819121447;
      transv.curr_dMtheta_dev_sym_dC(2, 1) = -25684.8062015504;
      transv.curr_dMtheta_dev_sym_dC(2, 2) = 51575.9431524548;
      transv.curr_dMtheta_dev_sym_dC(2, 3) = -28.5788113695;
      transv.curr_dMtheta_dev_sym_dC(2, 4) = -155.3746770026;
      transv.curr_dMtheta_dev_sym_dC(2, 5) = 489.9224806202;
      transv.curr_dMtheta_dev_sym_dC(3, 0) = 459.5348837209;
      transv.curr_dMtheta_dev_sym_dC(3, 1) = -386.4341085271;
      transv.curr_dMtheta_dev_sym_dC(3, 2) = 0.4651162791;
      transv.curr_dMtheta_dev_sym_dC(3, 3) = 38178.5271317829;
      transv.curr_dMtheta_dev_sym_dC(3, 4) = -115.5426356589;
      transv.curr_dMtheta_dev_sym_dC(3, 5) = -77.2868217054;
      transv.curr_dMtheta_dev_sym_dC(4, 0) = 3.7209302326;
      transv.curr_dMtheta_dev_sym_dC(4, 1) = -309.1472868217;
      transv.curr_dMtheta_dev_sym_dC(4, 2) = -154.8837209302;
      transv.curr_dMtheta_dev_sym_dC(4, 3) = 308.2170542636;
      transv.curr_dMtheta_dev_sym_dC(4, 4) = 38605.0775193798;
      transv.curr_dMtheta_dev_sym_dC(4, 5) = 231.7054263566;
      transv.curr_dMtheta_dev_sym_dC(5, 0) = 612.7131782946;
      transv.curr_dMtheta_dev_sym_dC(5, 1) = 1.5503875969;
      transv.curr_dMtheta_dev_sym_dC(5, 2) = -232.3255813953;
      transv.curr_dMtheta_dev_sym_dC(5, 3) = -154.7286821705;
      transv.curr_dMtheta_dev_sym_dC(5, 4) = -193.1395348837;
      transv.curr_dMtheta_dev_sym_dC(5, 5) = 38255.3488372093;

      iso.curr_dMtheta_dev_sym_dC = transv.curr_dMtheta_dev_sym_dC;


      // curr_dequiv_stress_diFin_ and curr_dequiv_stress_dC_
      transv.curr_dequiv_stress_diFin(0, 0) = -234070.5670907900;
      transv.curr_dequiv_stress_diFin(0, 3) = 82572.5283838393;
      transv.curr_dequiv_stress_diFin(0, 5) = 91454.9787991677;
      transv.curr_dequiv_stress_diFin(0, 6) = 82765.2328670792;
      transv.curr_dequiv_stress_diFin(0, 1) = 78385.0892289339;
      transv.curr_dequiv_stress_diFin(0, 4) = 75532.8580504739;
      transv.curr_dequiv_stress_diFin(0, 8) = 82277.3731958116;
      transv.curr_dequiv_stress_diFin(0, 7) = 80591.1925156294;
      transv.curr_dequiv_stress_diFin(0, 2) = 271168.26338082500;

      iso.curr_dequiv_stress_diFin(0, 0) = -129134.0482102849;
      iso.curr_dequiv_stress_diFin(0, 3) = 37557.6315094706;
      iso.curr_dequiv_stress_diFin(0, 5) = 78964.6443255154;
      iso.curr_dequiv_stress_diFin(0, 6) = 37939.3978025361;
      iso.curr_dequiv_stress_diFin(0, 1) = 3427.1816397928;
      iso.curr_dequiv_stress_diFin(0, 4) = 65588.9253940802;
      iso.curr_dequiv_stress_diFin(0, 8) = 78341.9331995679;
      iso.curr_dequiv_stress_diFin(0, 7) = 65237.5232301227;
      iso.curr_dequiv_stress_diFin(0, 2) = 203454.2257209583;

      transv.curr_dequiv_stress_dC(0, 0) = -98970.8723205062;
      transv.curr_dequiv_stress_dC(0, 3) = 31128.5001943156;
      transv.curr_dequiv_stress_dC(0, 5) = 29714.7977578675;
      transv.curr_dequiv_stress_dC(0, 1) = 24264.1511852888;
      transv.curr_dequiv_stress_dC(0, 4) = 19972.6390863355;
      transv.curr_dequiv_stress_dC(0, 2) = 76946.0587989554;

      iso.curr_dequiv_stress_dC(0, 0) = -55281.6277749173;
      iso.curr_dequiv_stress_dC(0, 3) = 13922.6273357612;
      iso.curr_dequiv_stress_dC(0, 5) = 26398.4153239006;
      iso.curr_dequiv_stress_dC(0, 1) = -684.2409384814;
      iso.curr_dequiv_stress_dC(0, 4) = 17987.7579527249;
      iso.curr_dequiv_stress_dC(0, 2) = 57284.3290405007;

      // curr_dpsr_dequiv_stress_ and curr_dpsr_depsp_
      transv.curr_dpsr_dequiv_stress = 286.974332715012;
      transv.curr_dpsr_depsp = -624541.984266900;
      iso.curr_dpsr_dequiv_stress = 0.126943384343792;
      iso.curr_dpsr_depsp = -185.420977353392;

      // curr_ddp_...
      transv.curr_ddpdiFin(0, 0) = 91644138.7155280411;
      transv.curr_ddpdiFin(0, 1) = -33855746.9447646737;
      transv.curr_ddpdiFin(0, 2) = -101470278.2079664171;
      transv.curr_ddpdiFin(0, 3) = -30184411.5570825525;
      transv.curr_ddpdiFin(0, 4) = -27531341.9085373469;
      transv.curr_ddpdiFin(0, 5) = -33371873.2873882316;
      transv.curr_ddpdiFin(0, 6) = -29687680.7348305024;
      transv.curr_ddpdiFin(0, 7) = -29532464.0020302981;
      transv.curr_ddpdiFin(0, 8) = -29220130.8645477481;
      transv.curr_ddpdiFin(1, 0) = -25192466.2650080249;
      transv.curr_ddpdiFin(1, 1) = 15088651.1946296263;
      transv.curr_ddpdiFin(1, 2) = 20716795.2903412357;
      transv.curr_ddpdiFin(1, 3) = 7674773.8181884130;
      transv.curr_ddpdiFin(1, 4) = 6503389.0112429662;
      transv.curr_ddpdiFin(1, 5) = 7837286.2101101996;
      transv.curr_ddpdiFin(1, 6) = 7096590.0579067646;
      transv.curr_ddpdiFin(1, 7) = 7599112.3864524541;
      transv.curr_ddpdiFin(1, 8) = 6859721.9846235774;
      transv.curr_ddpdiFin(2, 0) = -66451672.4505201131;
      transv.curr_ddpdiFin(2, 1) = 18767095.7501350977;
      transv.curr_ddpdiFin(2, 2) = 80753482.9176253378;
      transv.curr_ddpdiFin(2, 3) = 22509637.7388941906;
      transv.curr_ddpdiFin(2, 4) = 21027952.8972944245;
      transv.curr_ddpdiFin(2, 5) = 25534587.0772780851;
      transv.curr_ddpdiFin(2, 6) = 22591090.6769237891;
      transv.curr_ddpdiFin(2, 7) = 21933351.6155778915;
      transv.curr_ddpdiFin(2, 8) = 22360408.8799242154;
      transv.curr_ddpdiFin(3, 0) = -25302850.3825668655;
      transv.curr_ddpdiFin(3, 1) = 8837274.2712409757;
      transv.curr_ddpdiFin(3, 2) = 29690990.4629474208;
      transv.curr_ddpdiFin(3, 3) = 14723887.4728642069;
      transv.curr_ddpdiFin(3, 4) = 8270309.1433168277;
      transv.curr_ddpdiFin(3, 5) = 10013667.7849416826;
      transv.curr_ddpdiFin(3, 6) = 15874435.5976522509;
      transv.curr_ddpdiFin(3, 7) = 9321122.2425347324;
      transv.curr_ddpdiFin(3, 8) = 9417577.9729933925;
      transv.curr_ddpdiFin(4, 0) = -17133866.5207469314;
      transv.curr_ddpdiFin(4, 1) = 5936506.6183022317;
      transv.curr_ddpdiFin(4, 2) = 20053798.4668786749;
      transv.curr_ddpdiFin(4, 3) = 6331170.9771714164;
      transv.curr_ddpdiFin(4, 4) = 8935089.7678608019;
      transv.curr_ddpdiFin(4, 5) = 6857569.2574761631;
      transv.curr_ddpdiFin(4, 6) = 6058380.0437969314;
      transv.curr_ddpdiFin(4, 7) = 9901306.9111622106;
      transv.curr_ddpdiFin(4, 8) = 6022668.9221798619;
      transv.curr_ddpdiFin(5, 0) = -23864198.3120955862;
      transv.curr_ddpdiFin(5, 1) = 8087671.0405692905;
      transv.curr_ddpdiFin(5, 2) = 28227267.7201730609;
      transv.curr_ddpdiFin(5, 3) = 8519725.5386940818;
      transv.curr_ddpdiFin(5, 4) = 7920714.6361805722;
      transv.curr_ddpdiFin(5, 5) = 12277597.4840316363;
      transv.curr_ddpdiFin(5, 6) = 8738359.7717888728;
      transv.curr_ddpdiFin(5, 7) = 8315293.8938418152;
      transv.curr_ddpdiFin(5, 8) = 12491337.5138968322;

      transv.curr_ddpdC(0, 0) = 38639582.8507589623;
      transv.curr_ddpdC(0, 1) = -10675982.8181045111;
      transv.curr_ddpdC(0, 2) = -28829711.8857412674;
      transv.curr_ddpdC(0, 3) = -11305990.6716092415;
      transv.curr_ddpdC(0, 4) = -7230276.9459540909;
      transv.curr_ddpdC(0, 5) = -10784310.0264084935;
      transv.curr_ddpdC(1, 0) = -10578817.6808521971;
      transv.curr_ddpdC(1, 1) = 4964596.4969077781;
      transv.curr_ddpdC(1, 2) = 5841776.5978700034;
      transv.curr_ddpdC(1, 3) = 2772081.0151980445;
      transv.curr_ddpdC(1, 4) = 1759780.9403602553;
      transv.curr_ddpdC(1, 5) = 2619010.3238850934;
      transv.curr_ddpdC(2, 0) = -28060765.1699068062;
      transv.curr_ddpdC(2, 1) = 5711386.3211967424;
      transv.curr_ddpdC(2, 2) = 22987935.2878713086;
      transv.curr_ddpdC(2, 3) = 8533909.6564112119;
      transv.curr_ddpdC(2, 4) = 5470496.0055938475;
      transv.curr_ddpdC(2, 5) = 8165299.7025234178;
      transv.curr_ddpdC(3, 0) = -10808530.9044242743;
      transv.curr_ddpdC(3, 1) = 2633143.6851439341;
      transv.curr_ddpdC(3, 2) = 8425072.9587976150;
      transv.curr_ddpdC(3, 3) = 5740726.4882670790;
      transv.curr_ddpdC(3, 4) = 2179802.8496266622;
      transv.curr_ddpdC(3, 5) = 3248836.8278546161;
      transv.curr_ddpdC(4, 0) = -7244512.3399683675;
      transv.curr_ddpdC(4, 1) = 1766682.4926764967;
      transv.curr_ddpdC(4, 2) = 5627687.9351656092;
      transv.curr_ddpdC(4, 3) = 2288007.7477375367;
      transv.curr_ddpdC(4, 4) = 2641206.8900739593;
      transv.curr_ddpdC(4, 5) = 2182188.2279272475;
      transv.curr_ddpdC(5, 0) = -10192969.4922571108;
      transv.curr_ddpdC(5, 1) = 2503590.7585001062;
      transv.curr_ddpdC(5, 2) = 7932097.2201953027;
      transv.curr_ddpdC(5, 3) = 3207071.6663878239;
      transv.curr_ddpdC(5, 4) = 2054851.2149345700;
      transv.curr_ddpdC(5, 5) = 4234469.4459014749;

      transv.curr_ddpdepsp(0, 0) = 823740.4201837323;
      transv.curr_ddpdepsp(3, 0) = -248885.8351145428;
      transv.curr_ddpdepsp(5, 0) = -234533.2252194774;
      transv.curr_ddpdepsp(1, 0) = -200497.2813432335;
      transv.curr_ddpdepsp(4, 0) = -166388.3543832313;
      transv.curr_ddpdepsp(2, 0) = -623243.1388405000;

      iso.curr_ddpdiFin(0, 0) = 12829.8331606621;
      iso.curr_ddpdiFin(0, 1) = -1196.0053973834;
      iso.curr_ddpdiFin(0, 2) = -18912.5270252701;
      iso.curr_ddpdiFin(0, 3) = -3339.9603963346;
      iso.curr_ddpdiFin(0, 4) = -5810.1301343318;
      iso.curr_ddpdiFin(0, 5) = -7007.5180574678;
      iso.curr_ddpdiFin(0, 6) = -3263.6832099348;
      iso.curr_ddpdiFin(0, 7) = -5780.7649015435;
      iso.curr_ddpdiFin(0, 8) = -6746.8505625536;
      iso.curr_ddpdiFin(1, 0) = -667.0448936479;
      iso.curr_ddpdiFin(1, 1) = 1788.1675585782;
      iso.curr_ddpdiFin(1, 2) = -1177.3041353505;
      iso.curr_ddpdiFin(1, 3) = 62.5528974475;
      iso.curr_ddpdiFin(1, 4) = -92.7054563616;
      iso.curr_ddpdiFin(1, 5) = -124.1242003306;
      iso.curr_ddpdiFin(1, 6) = -56.8748370657;
      iso.curr_ddpdiFin(1, 7) = 67.1763405175;
      iso.curr_ddpdiFin(1, 8) = -113.6440544923;
      iso.curr_ddpdiFin(2, 0) = -12162.7882670142;
      iso.curr_ddpdiFin(2, 1) = -592.1621611948;
      iso.curr_ddpdiFin(2, 2) = 20089.8311606207;
      iso.curr_ddpdiFin(2, 3) = 3277.4074988871;
      iso.curr_ddpdiFin(2, 4) = 5902.8355906934;
      iso.curr_ddpdiFin(2, 5) = 7131.6422577984;
      iso.curr_ddpdiFin(2, 6) = 3320.5580470005;
      iso.curr_ddpdiFin(2, 7) = 5713.5885610260;
      iso.curr_ddpdiFin(2, 8) = 6860.4946170460;
      iso.curr_ddpdiFin(3, 0) = -2671.0361106729;
      iso.curr_ddpdiFin(3, 1) = 122.7904338792;
      iso.curr_ddpdiFin(3, 2) = 4309.5911752903;
      iso.curr_ddpdiFin(3, 3) = 1915.6442152446;
      iso.curr_ddpdiFin(3, 4) = 1389.3122793270;
      iso.curr_ddpdiFin(3, 5) = 1672.6383201886;
      iso.curr_ddpdiFin(3, 6) = 2146.3485281346;
      iso.curr_ddpdiFin(3, 7) = 1479.8214671226;
      iso.curr_ddpdiFin(3, 8) = 1740.0222276724;
      iso.curr_ddpdiFin(4, 0) = -3657.3192050841;
      iso.curr_ddpdiFin(4, 1) = 175.4131742978;
      iso.curr_ddpdiFin(4, 2) = 5842.7807094743;
      iso.curr_ddpdiFin(4, 3) = 1176.7992528605;
      iso.curr_ddpdiFin(4, 4) = 3200.3135417769;
      iso.curr_ddpdiFin(4, 5) = 2300.7236786771;
      iso.curr_ddpdiFin(4, 6) = 1074.5151269988;
      iso.curr_ddpdiFin(4, 7) = 3425.2879583401;
      iso.curr_ddpdiFin(4, 8) = 2218.7909449537;
      iso.curr_ddpdiFin(5, 0) = -5042.0892564549;
      iso.curr_ddpdiFin(5, 1) = 136.8171919108;
      iso.curr_ddpdiFin(5, 2) = 8220.0887667428;
      iso.curr_ddpdiFin(5, 3) = 1499.3455900563;
      iso.curr_ddpdiFin(5, 4) = 2668.5838552794;
      iso.curr_ddpdiFin(5, 5) = 4272.4571653080;
      iso.curr_ddpdiFin(5, 6) = 1592.9350994585;
      iso.curr_ddpdiFin(5, 7) = 2604.3599883719;
      iso.curr_ddpdiFin(5, 8) = 4705.1422737136;

      iso.curr_ddpdC(0, 0) = 5460.4565545292;
      iso.curr_ddpdC(0, 1) = -249.2019038089;
      iso.curr_ddpdC(0, 2) = -5339.3746414694;
      iso.curr_ddpdC(0, 3) = -1227.1601268002;
      iso.curr_ddpdC(0, 4) = -1577.2298768853;
      iso.curr_ddpdC(0, 5) = -2321.7821936855;
      iso.curr_ddpdC(1, 0) = -269.6067761762;
      iso.curr_ddpdC(1, 1) = 618.9702372573;
      iso.curr_ddpdC(1, 2) = -345.8711127131;
      iso.curr_ddpdC(1, 3) = -3.3216656286;
      iso.curr_ddpdC(1, 4) = -11.1018481730;
      iso.curr_ddpdC(1, 5) = -17.8615196133;
      iso.curr_ddpdC(2, 0) = -5190.8497783530;
      iso.curr_ddpdC(2, 1) = -369.7683334485;
      iso.curr_ddpdC(2, 2) = 5685.2457541825;
      iso.curr_ddpdC(2, 3) = 1230.4817924288;
      iso.curr_ddpdC(2, 4) = 1588.3317250582;
      iso.curr_ddpdC(2, 5) = 2339.6437132988;
      iso.curr_ddpdC(3, 0) = -1165.4485042004;
      iso.curr_ddpdC(3, 1) = -19.1468367652;
      iso.curr_ddpdC(3, 2) = 1213.4089498820;
      iso.curr_ddpdC(3, 3) = 754.6294465519;
      iso.curr_ddpdC(3, 4) = 379.6275077393;
      iso.curr_ddpdC(3, 5) = 558.2436840095;
      iso.curr_ddpdC(4, 0) = -1565.6349036231;
      iso.curr_ddpdC(4, 1) = -23.1015229682;
      iso.curr_ddpdC(4, 2) = 1620.5349764111;
      iso.curr_ddpdC(4, 3) = 398.0263319536;
      iso.curr_ddpdC(4, 4) = 974.3020933837;
      iso.curr_ddpdC(4, 5) = 750.4428216766;
      iso.curr_ddpdC(5, 0) = -2199.5308716671;
      iso.curr_ddpdC(5, 1) = -27.2970483368;
      iso.curr_ddpdC(5, 2) = 2284.0614600306;
      iso.curr_ddpdC(5, 3) = 553.9448096130;
      iso.curr_ddpdC(5, 4) = 715.7671718130;
      iso.curr_ddpdC(5, 5) = 1514.5000876252;

      iso.curr_ddpdepsp(0, 0) = 136.8839435478;
      iso.curr_ddpdepsp(3, 0) = -33.0284697510;
      iso.curr_ddpdepsp(5, 0) = -62.2476046595;
      iso.curr_ddpdepsp(1, 0) = 0.9621707109;
      iso.curr_ddpdepsp(4, 0) = -44.1612334197;
      iso.curr_ddpdepsp(2, 0) = -137.8461142586;

      // curr_dlp_...
      transv.curr_dlpdiFin(0, 0) = 91644138.7155280411;
      transv.curr_dlpdiFin(0, 1) = -33855746.9447646737;
      transv.curr_dlpdiFin(0, 2) = -101470278.2079664171;
      transv.curr_dlpdiFin(0, 3) = -30184411.5570825525;
      transv.curr_dlpdiFin(0, 4) = -27531341.9085373469;
      transv.curr_dlpdiFin(0, 5) = -33371873.2873882316;
      transv.curr_dlpdiFin(0, 6) = -29687680.7348305024;
      transv.curr_dlpdiFin(0, 7) = -29532464.0020302981;
      transv.curr_dlpdiFin(0, 8) = -29220130.8645477481;
      transv.curr_dlpdiFin(1, 0) = -25192466.2650080249;
      transv.curr_dlpdiFin(1, 1) = 15088651.1946296263;
      transv.curr_dlpdiFin(1, 2) = 20716795.2903412357;
      transv.curr_dlpdiFin(1, 3) = 7674773.8181884130;
      transv.curr_dlpdiFin(1, 4) = 6503389.0112429662;
      transv.curr_dlpdiFin(1, 5) = 7837286.2101101996;
      transv.curr_dlpdiFin(1, 6) = 7096590.0579067646;
      transv.curr_dlpdiFin(1, 7) = 7599112.3864524541;
      transv.curr_dlpdiFin(1, 8) = 6859721.9846235774;
      transv.curr_dlpdiFin(2, 0) = -66451672.4505201131;
      transv.curr_dlpdiFin(2, 1) = 18767095.7501350977;
      transv.curr_dlpdiFin(2, 2) = 80753482.9176253378;
      transv.curr_dlpdiFin(2, 3) = 22509637.7388941906;
      transv.curr_dlpdiFin(2, 4) = 21027952.8972944245;
      transv.curr_dlpdiFin(2, 5) = 25534587.0772780851;
      transv.curr_dlpdiFin(2, 6) = 22591090.6769237891;
      transv.curr_dlpdiFin(2, 7) = 21933351.6155778915;
      transv.curr_dlpdiFin(2, 8) = 22360408.8799242154;
      transv.curr_dlpdiFin(3, 0) = -25302850.3825668655;
      transv.curr_dlpdiFin(3, 1) = 8837274.2712409757;
      transv.curr_dlpdiFin(3, 2) = 29690990.4629474208;
      transv.curr_dlpdiFin(3, 3) = 14723887.4728642069;
      transv.curr_dlpdiFin(3, 4) = 8270309.1433168277;
      transv.curr_dlpdiFin(3, 5) = 10013667.7849416826;
      transv.curr_dlpdiFin(3, 6) = 15874435.5976522509;
      transv.curr_dlpdiFin(3, 7) = 9321122.2425347324;
      transv.curr_dlpdiFin(3, 8) = 9417577.9729933925;
      transv.curr_dlpdiFin(4, 0) = 0.0000000000;
      transv.curr_dlpdiFin(4, 1) = 0.0000000000;
      transv.curr_dlpdiFin(4, 2) = 0.0000000000;
      transv.curr_dlpdiFin(4, 3) = 0.0000000000;
      transv.curr_dlpdiFin(4, 4) = 0.0000000000;
      transv.curr_dlpdiFin(4, 5) = 0.0000000000;
      transv.curr_dlpdiFin(4, 6) = 0.0000000000;
      transv.curr_dlpdiFin(4, 7) = 0.0000000000;
      transv.curr_dlpdiFin(4, 8) = 0.0000000000;
      transv.curr_dlpdiFin(5, 0) = 0.0000000000;
      transv.curr_dlpdiFin(5, 1) = 0.0000000000;
      transv.curr_dlpdiFin(5, 2) = 0.0000000000;
      transv.curr_dlpdiFin(5, 3) = 0.0000000000;
      transv.curr_dlpdiFin(5, 4) = 0.0000000000;
      transv.curr_dlpdiFin(5, 5) = 0.0000000000;
      transv.curr_dlpdiFin(5, 6) = 0.0000000000;
      transv.curr_dlpdiFin(5, 7) = 0.0000000000;
      transv.curr_dlpdiFin(5, 8) = 0.0000000000;
      transv.curr_dlpdiFin(6, 0) = -25302850.3825668655;
      transv.curr_dlpdiFin(6, 1) = 8837274.2712409757;
      transv.curr_dlpdiFin(6, 2) = 29690990.4629474208;
      transv.curr_dlpdiFin(6, 3) = 14723887.4728642069;
      transv.curr_dlpdiFin(6, 4) = 8270309.1433168277;
      transv.curr_dlpdiFin(6, 5) = 10013667.7849416826;
      transv.curr_dlpdiFin(6, 6) = 15874435.5976522509;
      transv.curr_dlpdiFin(6, 7) = 9321122.2425347324;
      transv.curr_dlpdiFin(6, 8) = 9417577.9729933925;
      transv.curr_dlpdiFin(7, 0) = -34267733.0414938629;
      transv.curr_dlpdiFin(7, 1) = 11873013.2366044633;
      transv.curr_dlpdiFin(7, 2) = 40107596.9337573498;
      transv.curr_dlpdiFin(7, 3) = 12662341.9543428328;
      transv.curr_dlpdiFin(7, 4) = 17870179.5357216038;
      transv.curr_dlpdiFin(7, 5) = 13715138.5149523262;
      transv.curr_dlpdiFin(7, 6) = 12116760.0875938628;
      transv.curr_dlpdiFin(7, 7) = 19802613.8223244213;
      transv.curr_dlpdiFin(7, 8) = 12045337.8443597239;
      transv.curr_dlpdiFin(8, 0) = -47728396.6241911724;
      transv.curr_dlpdiFin(8, 1) = 16175342.0811385810;
      transv.curr_dlpdiFin(8, 2) = 56454535.4403461218;
      transv.curr_dlpdiFin(8, 3) = 17039451.0773881637;
      transv.curr_dlpdiFin(8, 4) = 15841429.2723611444;
      transv.curr_dlpdiFin(8, 5) = 24555194.9680632725;
      transv.curr_dlpdiFin(8, 6) = 17476719.5435777456;
      transv.curr_dlpdiFin(8, 7) = 16630587.7876836304;
      transv.curr_dlpdiFin(8, 8) = 24982675.0277936645;

      transv.curr_dlpdC(0, 0) = 38639582.8507589623;
      transv.curr_dlpdC(0, 1) = -10675982.8181045111;
      transv.curr_dlpdC(0, 2) = -28829711.8857412674;
      transv.curr_dlpdC(0, 3) = -11305990.6716092415;
      transv.curr_dlpdC(0, 4) = -7230276.9459540909;
      transv.curr_dlpdC(0, 5) = -10784310.0264084935;
      transv.curr_dlpdC(1, 0) = -10578817.6808521971;
      transv.curr_dlpdC(1, 1) = 4964596.4969077781;
      transv.curr_dlpdC(1, 2) = 5841776.5978700034;
      transv.curr_dlpdC(1, 3) = 2772081.0151980445;
      transv.curr_dlpdC(1, 4) = 1759780.9403602553;
      transv.curr_dlpdC(1, 5) = 2619010.3238850934;
      transv.curr_dlpdC(2, 0) = -28060765.1699068062;
      transv.curr_dlpdC(2, 1) = 5711386.3211967424;
      transv.curr_dlpdC(2, 2) = 22987935.2878713086;
      transv.curr_dlpdC(2, 3) = 8533909.6564112119;
      transv.curr_dlpdC(2, 4) = 5470496.0055938475;
      transv.curr_dlpdC(2, 5) = 8165299.7025234178;
      transv.curr_dlpdC(3, 0) = -10808530.9044242743;
      transv.curr_dlpdC(3, 1) = 2633143.6851439341;
      transv.curr_dlpdC(3, 2) = 8425072.9587976150;
      transv.curr_dlpdC(3, 3) = 5740726.4882670790;
      transv.curr_dlpdC(3, 4) = 2179802.8496266622;
      transv.curr_dlpdC(3, 5) = 3248836.8278546161;
      transv.curr_dlpdC(4, 0) = 0.0000000000;
      transv.curr_dlpdC(4, 1) = 0.0000000000;
      transv.curr_dlpdC(4, 2) = 0.0000000000;
      transv.curr_dlpdC(4, 3) = 0.0000000000;
      transv.curr_dlpdC(4, 4) = 0.0000000000;
      transv.curr_dlpdC(4, 5) = 0.0000000000;
      transv.curr_dlpdC(5, 0) = 0.0000000000;
      transv.curr_dlpdC(5, 1) = 0.0000000000;
      transv.curr_dlpdC(5, 2) = 0.0000000000;
      transv.curr_dlpdC(5, 3) = 0.0000000000;
      transv.curr_dlpdC(5, 4) = 0.0000000000;
      transv.curr_dlpdC(5, 5) = 0.0000000000;
      transv.curr_dlpdC(6, 0) = -10808530.9044242743;
      transv.curr_dlpdC(6, 1) = 2633143.6851439341;
      transv.curr_dlpdC(6, 2) = 8425072.9587976150;
      transv.curr_dlpdC(6, 3) = 5740726.4882670790;
      transv.curr_dlpdC(6, 4) = 2179802.8496266622;
      transv.curr_dlpdC(6, 5) = 3248836.8278546161;
      transv.curr_dlpdC(7, 0) = -14489024.6799367350;
      transv.curr_dlpdC(7, 1) = 3533364.9853529935;
      transv.curr_dlpdC(7, 2) = 11255375.8703312185;
      transv.curr_dlpdC(7, 3) = 4576015.4954750733;
      transv.curr_dlpdC(7, 4) = 5282413.7801479185;
      transv.curr_dlpdC(7, 5) = 4364376.4558544951;
      transv.curr_dlpdC(8, 0) = -20385938.9845142215;
      transv.curr_dlpdC(8, 1) = 5007181.5170002123;
      transv.curr_dlpdC(8, 2) = 15864194.4403906055;
      transv.curr_dlpdC(8, 3) = 6414143.3327756478;
      transv.curr_dlpdC(8, 4) = 4109702.4298691400;
      transv.curr_dlpdC(8, 5) = 8468938.8918029498;

      transv.curr_dlpdepsp(0, 0) = 823740.4201837323;
      transv.curr_dlpdepsp(3, 0) = -248885.8351145428;
      transv.curr_dlpdepsp(5, 0) = 0.0000000000;
      transv.curr_dlpdepsp(6, 0) = -248885.8351145428;
      transv.curr_dlpdepsp(1, 0) = -200497.2813432335;
      transv.curr_dlpdepsp(4, 0) = 0.0000000000;
      transv.curr_dlpdepsp(8, 0) = -469066.4504389548;
      transv.curr_dlpdepsp(7, 0) = -332776.7087664626;
      transv.curr_dlpdepsp(2, 0) = -623243.1388405000;

      Core::LinAlg::FourTensor<3> tempFourTensor(true);
      Core::LinAlg::Voigt::setup_four_tensor_from_6x9_voigt_matrix(
          tempFourTensor, iso.curr_ddpdiFin);
      Core::LinAlg::Voigt::setup_9x9_voigt_matrix_from_four_tensor(
          iso.curr_dlpdiFin, tempFourTensor);

      tempFourTensor.clear();
      Core::LinAlg::Voigt::setup_four_tensor_from_6x6_voigt_matrix(tempFourTensor, iso.curr_ddpdC);
      Core::LinAlg::Voigt::setup_9x6_voigt_matrix_from_four_tensor(iso.curr_dlpdC, tempFourTensor);

      iso.curr_dlpdepsp(0, 0) = iso.curr_ddpdepsp(0, 0);
      iso.curr_dlpdepsp(3, 0) = iso.curr_ddpdepsp(3, 0);
      iso.curr_dlpdepsp(5, 0) = iso.curr_ddpdepsp(5, 0);
      iso.curr_dlpdepsp(6, 0) = iso.curr_ddpdepsp(3, 0);
      iso.curr_dlpdepsp(1, 0) = iso.curr_ddpdepsp(1, 0);
      iso.curr_dlpdepsp(4, 0) = iso.curr_ddpdepsp(4, 0);
      iso.curr_dlpdepsp(8, 0) = iso.curr_ddpdepsp(5, 0);
      iso.curr_dlpdepsp(7, 0) = iso.curr_ddpdepsp(4, 0);
      iso.curr_dlpdepsp(2, 0) = iso.curr_ddpdepsp(2, 0);
    }


    // deformation gradient
    Core::LinAlg::Matrix<3, 3> FM_;
    // derivative of second Piola-Kirchhoff stress tensor w.r.t. inverse inelastic deformation
    // gradient
    Core::LinAlg::Matrix<6, 9> dSdiFin_;
    // reference solution of inverse inelastic deformation gradient using
    // InelasticDefgradLinScalarIso
    Core::LinAlg::Matrix<3, 3> iFin_lin_scalar_iso_solution_;
    // reference solution of inverse inelastic deformation gradient using
    // InelasticDefgradLinScalarAniso
    Core::LinAlg::Matrix<3, 3> iFin_lin_scalar_aniso_solution_;
    // reference solution of inverse inelastic deformation gradient using
    // InelasticDefgradPolyIntercalFracIso
    Core::LinAlg::Matrix<3, 3> iFin_poly_intercal_frac_iso_solution_;
    // reference solution of inverse inelastic deformation gradient using
    // InelasticDefgradPolyIntercalFracAniso
    Core::LinAlg::Matrix<3, 3> iFin_poly_intercal_frac_aniso_solution_;
    // reference solution of inverse inelastic deformation gradient using InelasticDefgradLinTempIso
    Core::LinAlg::Matrix<3, 3> iFin_lin_temp_iso_solution_;
    // reference solution of inverse inelastic deformation gradient using
    // InelasticDefgradTransvIsotropElastViscoplast
    Core::LinAlg::Matrix<3, 3> iFin_transv_isotrop_vplast_refJC_solution_;
    // reference solution of plastic strain using InelasticDefgradTransvIsotropElastViscoplast
    double plastic_strain_transv_isotrop_vplast_refJC_solution_;
    // pointer to object that evaluates a linear shape
    std::shared_ptr<Mat::InelasticDefgradLinearShape> linear_shape_;
    // pointer to object that evaluates a polynomial shape
    std::shared_ptr<Mat::InelasticDefgradPolynomialShape> polynomial_shape_;
    // pointer to InelasticDefgradLinScalarIso object
    std::shared_ptr<Mat::InelasticDefgradLinScalarIso> lin_scalar_iso_;
    // pointer to parameters of InelasticDefgradScalar
    std::shared_ptr<Mat::PAR::InelasticDefgradLinScalar> params_inelastic_defgrad_lin_scalar_iso_;
    std::shared_ptr<Mat::PAR::InelasticDefgradScalar> params_inelastic_defgrad_scalar_;
    // pointer to InelasticDefgradLinScalarAniso object
    std::shared_ptr<Mat::InelasticDefgradLinScalarAniso> lin_scalar_aniso_;
    // pointer to parameters of InelasticDefgradScalar
    std::shared_ptr<Mat::PAR::InelasticDefgradLinScalarAniso>
        params_inelastic_defgrad_lin_scalar_aniso_;
    // pointer to InelasticDefgradPolyIntercalFracIso object
    std::shared_ptr<Mat::InelasticDefgradPolyIntercalFracIso> poly_intercal_frac_iso_;
    // pointer to parameters of InelasticDefgradIntercalFrac
    std::shared_ptr<Mat::PAR::InelasticDefgradPolyIntercalFrac>
        params_inelastic_defgrad_poly_intercal_frac_;
    // pointer to InelasticDefgradPolyIntercalFracAniso object
    std::shared_ptr<Mat::InelasticDefgradPolyIntercalFracAniso> poly_intercal_frac_aniso_;
    // pointer to parameters of InelasticDefgradPolyIntercalFracAniso
    std::shared_ptr<Mat::PAR::InelasticDefgradPolyIntercalFracAniso>
        params_inelastic_defgrad_poly_intercal_frac_aniso_;
    // pointer to InelasticDefgradLinTempIso
    std::shared_ptr<Mat::InelasticDefgradLinTempIso> lin_temp_iso_;
    // pointer to parameters of InelasticDefgradLinTempIso
    std::shared_ptr<Mat::PAR::InelasticDefgradLinTempIso> params_lin_temp_iso_;
    // reference StateQuantities struct of InelasticDefgradTransvIsotropElastViscoplast
    // (transversely isotropic, logarithmic substepping, Reformulated Johnson-Cook viscoplasticity)
    Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::StateQuantities
        state_quantities_solution_transv_isotrop_;
    // reference StateQuantities struct of InelasticDefgradTransvIsotropElastViscoplast (isotropic,
    // logarithmic substepping, Reformulated Johnson-Cook viscoplasticity)
    Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::StateQuantities
        state_quantities_solution_isotrop_;
    // reference StateQuantityDerivatives struct of InelasticDefgradTransvIsotropElastViscoplast
    // (transversely isotropic, logarithmic substepping, Reformulated Johnson-Cook viscoplasticity)
    Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::StateQuantityDerivatives
        state_quantity_derivatives_solution_transv_isotrop_;
    // reference StateQuantityDerivatives struct of InelasticDefgradTransvIsotropElastViscoplast
    // (isotropic, logarithmic substepping, Reformulated Johnson-Cook viscoplasticity)
    Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::StateQuantityDerivatives
        state_quantity_derivatives_solution_isotrop_;
    Core::Utils::SingletonOwnerRegistry::ScopeGuard guard;
  };

  TEST_F(InelasticDefgradFactorsTest, TestEvaluateLinearGrowth)
  {
    const std::vector<double> test_values{952834.0233, 44327.362, 12823.902};
    const std::vector<double> results{0.4776612182790999, -0.0011217922259999, -0.0177241156459999};

    // loop over test values and check whether the result is correct
    for (auto i = 0U; i < test_values.size(); ++i)
      EXPECT_NEAR(linear_shape_->evaluate_linear_growth(test_values[i]), results[i], 1.0e-12);
  }

  TEST_F(InelasticDefgradFactorsTest, TestInelasticDeformationDirection)
  {
    // growth directions to be tested
    const std::vector<std::vector<double>> growth_directions{
        {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}, {1.0, 0.5, 0.3}};

    // result of first growth direction
    Core::LinAlg::SymmetricTensor<double, 3, 3> growth_dir_1{};
    growth_dir_1(0, 0) = 1.0;

    // result of second growth direction
    Core::LinAlg::SymmetricTensor<double, 3, 3> growth_dir_2{};
    growth_dir_2(1, 1) = 1.0;

    // result of third growth direction
    Core::LinAlg::SymmetricTensor<double, 3, 3> growth_dir_3{};
    growth_dir_3(2, 2) = 1.0;

    // result of fourth growth direction
    Core::LinAlg::SymmetricTensor<double, 3, 3> growth_dir_4{};
    // clang-format off
    growth_dir_4(0, 0) = 0.74626865671641791044776119402985; growth_dir_4(0, 1) = 0.37313432835820895522388059701493; growth_dir_4(0, 2) = 0.22388059701492537313432835820896;
    growth_dir_4(1, 0) = growth_dir_4(0,1); growth_dir_4(1, 1) = 0.18656716417910447761194029850746; growth_dir_4(1, 2) = 0.11194029850746268656716417910448;
    growth_dir_4(2, 0) = growth_dir_4(0, 2); growth_dir_4(2, 1) = growth_dir_4(1, 2); growth_dir_4(2, 2) = 0.067164179104477611940298507462687;
    // clang-format on

    // put all results together
    const std::vector<Core::LinAlg::SymmetricTensor<double, 3, 3>> growth_direction_solutions{
        growth_dir_1, growth_dir_2, growth_dir_3, growth_dir_4};

    // loop over all growth directions to be tested and perform the test
    for (auto i = 0U; i < growth_directions.size(); ++i)
    {
      // create object
      auto growth_direction = Mat::PAR::InelasticDeformationDirection(growth_directions[i]);
      // check the results
      FOUR_C_EXPECT_NEAR(
          growth_direction.growth_dir_tensor(), growth_direction_solutions[i], 1.0e-12);
    }
  }

  TEST_F(InelasticDefgradFactorsTest, TestComputePolynomial)
  {
    const std::vector<double> TestValues{0.215, 0.462, 0.675, 0.802};
    const std::vector<double> Results{
        -0.0472760218320868, -0.0136018625081755, -0.0054533973886232, -0.0027181447027916};

    // loop over test values and check whether the result is correct
    for (auto i = 0U; i < TestValues.size(); ++i)
      EXPECT_NEAR(polynomial_shape_->compute_polynomial(TestValues[i]), Results[i], 1.0e-12);
  }

  TEST_F(InelasticDefgradFactorsTest, TestComputePolynomialDerivative)
  {
    const std::vector<double> TestValues{0.215, 0.462, 0.675, 0.802};
    const std::vector<double> Results{
        0.3081029423168080, 0.0393764326640364, 0.0224486538440280, 0.0329592765802879};

    // loop over test values and check whether the result is correct
    for (auto i = 0U; i < TestValues.size(); ++i)
      EXPECT_NEAR(
          polynomial_shape_->compute_polynomial_derivative(TestValues[i]), Results[i], 1.0e-12);
  }

  TEST_F(InelasticDefgradFactorsTest, TestEvaluateInelasticDefGradDerivative)
  {
    const double detF = FM_.determinant();
    Core::LinAlg::Tensor<double, 3, 3> DFinDx{};
    Core::LinAlg::Tensor<double, 3, 3> DFinDx_ref{};

    lin_scalar_iso_->evaluate_inelastic_def_grad_derivative(detF, DFinDx);
    DFinDx_ref(0, 0) = DFinDx_ref(1, 1) = DFinDx_ref(2, 2) = 2.977205763668e-07;
    FOUR_C_EXPECT_NEAR(DFinDx, DFinDx_ref, 1.0e-10);

    // clear variables and next test
    DFinDx = {};
    DFinDx_ref = {};
    lin_scalar_aniso_->evaluate_inelastic_def_grad_derivative(detF, DFinDx);
    DFinDx_ref(0, 0) = 6.734163313433e-07;
    DFinDx_ref(1, 1) = 1.683540828358e-07;
    DFinDx_ref(2, 2) = 6.060746982090e-08;
    DFinDx_ref(0, 1) = DFinDx_ref(1, 0) = 3.367081656716e-07;
    DFinDx_ref(1, 2) = DFinDx_ref(2, 1) = 1.010124497015e-07;
    DFinDx_ref(0, 2) = DFinDx_ref(2, 0) = 2.020248994030e-07;
    FOUR_C_EXPECT_NEAR(DFinDx, DFinDx_ref, 1.0e-10);

    // clear variables and next test
    DFinDx = {};
    DFinDx_ref = {};
    poly_intercal_frac_iso_->evaluate_inelastic_def_grad_derivative(detF, DFinDx);
    DFinDx_ref(0, 0) = DFinDx_ref(1, 1) = DFinDx_ref(2, 2) = 3.399216373729e-07;
    FOUR_C_EXPECT_NEAR(DFinDx, DFinDx_ref, 1.0e-10);

    // clear variables and next test
    DFinDx = {};
    DFinDx_ref = {};
    poly_intercal_frac_aniso_->evaluate_inelastic_def_grad_derivative(detF, DFinDx);
    DFinDx_ref(0, 0) = 7.623672134952e-07;
    DFinDx_ref(1, 1) = 1.905918033738e-07;
    DFinDx_ref(2, 2) = 6.861304921457e-08;
    DFinDx_ref(0, 1) = DFinDx_ref(1, 0) = 3.811836067476e-07;
    DFinDx_ref(1, 2) = DFinDx_ref(2, 1) = 1.143550820243e-07;
    DFinDx_ref(0, 2) = DFinDx_ref(2, 0) = 2.287101640486e-07;
    FOUR_C_EXPECT_NEAR(DFinDx, DFinDx_ref, 1.0e-10);

    // clear variables and next test
    DFinDx = {};
    DFinDx_ref = {};
    lin_temp_iso_->evaluate_inelastic_def_grad_derivative(detF, DFinDx);
    DFinDx_ref(0, 0) = DFinDx_ref(1, 1) = DFinDx_ref(2, 2) = 3.373943094440e-04;
    FOUR_C_EXPECT_NEAR(DFinDx, DFinDx_ref, 1.0e-10);
  }

  TEST_F(InelasticDefgradFactorsTest, TestEvaluateInverseInelasticDefGrad)
  {
    // set up the "other" inverse inelastic deformation gradient (for each inelastic factor: the
    // product of all inverse inelastic deformation gradients) here we assume a single inelastic
    // deformation gradient, therefore, this is the unit tensor!
    Core::LinAlg::Matrix<3, 3> iFin_other(Core::LinAlg::Initialization::zero);
    iFin_other(0, 0) = 1.0;
    iFin_other(1, 1) = 1.0;
    iFin_other(2, 2) = 1.0;

    // create matrix to be filled by tested methods
    Core::LinAlg::Matrix<3, 3> iFin(Core::LinAlg::Initialization::zero);

    // test InelasticDefgradLinScalarIso evaluate the method
    lin_scalar_iso_->evaluate_inverse_inelastic_def_grad(&FM_, iFin_other, iFin);

    // compare the results
    FOUR_C_EXPECT_NEAR(iFin, iFin_lin_scalar_iso_solution_, 1.0e-10);

    // clear matrix to be filled again
    iFin.clear();

    // test InelasticDefgradLinScalarAniso evaluate the method
    lin_scalar_aniso_->evaluate_inverse_inelastic_def_grad(&FM_, iFin_other, iFin);

    // compare the results
    FOUR_C_EXPECT_NEAR(iFin, iFin_lin_scalar_aniso_solution_, 1.0e-10);

    // clear matrix to be filled again
    iFin.clear();

    // test InelasticDefgradPolyIntercalFracIso evaluate the method
    poly_intercal_frac_iso_->evaluate_inverse_inelastic_def_grad(&FM_, iFin_other, iFin);

    // compare the results
    FOUR_C_EXPECT_NEAR(iFin, iFin_poly_intercal_frac_iso_solution_, 1.0e-10);

    // clear matrix to be filled again
    iFin.clear();

    // test InelasticDefgradPolyIntercalFracAniso evaluate the method
    poly_intercal_frac_aniso_->evaluate_inverse_inelastic_def_grad(&FM_, iFin_other, iFin);

    // compare the results
    FOUR_C_EXPECT_NEAR(iFin, iFin_poly_intercal_frac_aniso_solution_, 1.0e-10);

    // clear matrix to be filled again
    iFin.clear();

    // test InelasticDefgradLinTempIso: evaluate the method
    lin_temp_iso_->evaluate_inverse_inelastic_def_grad(&FM_, iFin_other, iFin);

    // compare the results
    FOUR_C_EXPECT_NEAR(iFin, iFin_lin_temp_iso_solution_, 1.0e-15);
  }

  TEST_F(InelasticDefgradFactorsTest, TestEvaluateAdditionalCmat)
  {
    // calculate the inverse Cauchy-Green tensor in voigt notation
    Core::LinAlg::Matrix<3, 3> CM;
    Core::LinAlg::Matrix<3, 3> iCM;
    Core::LinAlg::Matrix<6, 1> iCV;
    CM.multiply_tn(1.0, FM_, FM_, 0.0);
    iCM.invert(CM);
    Core::LinAlg::Voigt::Stresses::matrix_to_vector(iCM, iCV);

    // set up the "other" inverse inelastic deformation gradient (for each inelastic factor: the
    // product of all inverse inelastic deformation gradients) here we assume a single inelastic
    // deformation gradient, therefore, this is the unit tensor!
    Core::LinAlg::Matrix<3, 3> iFin_other(Core::LinAlg::Initialization::zero);
    iFin_other(0, 0) = 1.0;
    iFin_other(1, 1) = 1.0;
    iFin_other(2, 2) = 1.0;

    // matrix to be filled by the methods
    Core::LinAlg::Matrix<6, 6> CMatAdd(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<6, 6> CMatAdd_ref_solution(Core::LinAlg::Initialization::zero);

    // test InelasticDefgradLinScalarIso set up reference solution
    // clang-format off
    CMatAdd_ref_solution(0, 0) = -1.5298408106321838e+02; CMatAdd_ref_solution(0, 1) = -1.2850516942246250e+02; CMatAdd_ref_solution(0, 2) = -1.1003777104023889e+02; CMatAdd_ref_solution(0, 3) =  5.9684525433019955e+00; CMatAdd_ref_solution(0, 4) =  6.3904917135249439e+00; CMatAdd_ref_solution(0, 5) =  9.7975743219718243e+00;
    CMatAdd_ref_solution(1, 0) = -1.4182447741625558e+02; CMatAdd_ref_solution(1, 1) = -1.1913120876345852e+02; CMatAdd_ref_solution(1, 2) = -1.0201093646719031e+02; CMatAdd_ref_solution(1, 3) =  5.5330767557948386e+00; CMatAdd_ref_solution(1, 4) =  5.9243297825808430e+00; CMatAdd_ref_solution(1, 5) =  9.0828787446608494e+00;
    CMatAdd_ref_solution(2, 0) = -1.3348186109765231e+02; CMatAdd_ref_solution(2, 1) = -1.1212349060090213e+02; CMatAdd_ref_solution(2, 2) = -9.6010293145590879e+01; CMatAdd_ref_solution(2, 3) =  5.2076016525127891e+00; CMatAdd_ref_solution(2, 4) =  5.5758397953702055e+00; CMatAdd_ref_solution(2, 5) =  8.5485917596808001e+00;
    CMatAdd_ref_solution(3, 0) =  2.8039858483825015e+00; CMatAdd_ref_solution(3, 1) =  2.3553213772332362e+00; CMatAdd_ref_solution(3, 2) =  2.0168395995193928e+00; CMatAdd_ref_solution(3, 3) = -1.0939345029791475e-01; CMatAdd_ref_solution(3, 4) = -1.1712884245469231e-01; CMatAdd_ref_solution(3, 5) = -1.7957593728939861e-01;
    CMatAdd_ref_solution(4, 0) =  2.9383344748119566e+00; CMatAdd_ref_solution(4, 1) =  2.4681729424484300e+00; CMatAdd_ref_solution(4, 2) =  2.1134733361269680e+00; CMatAdd_ref_solution(4, 3) = -1.1463486754557373e-01; CMatAdd_ref_solution(4, 4) = -1.2274088900198049e-01; CMatAdd_ref_solution(4, 5) = -1.8818003938518124e-01;
    CMatAdd_ref_solution(5, 0) =  4.5180142920228095e+00; CMatAdd_ref_solution(5, 1) =  3.7950889269948203e+00; CMatAdd_ref_solution(5, 2) =  3.2496990455934576e+00; CMatAdd_ref_solution(5, 3) = -1.7626378969949943e-01; CMatAdd_ref_solution(5, 4) = -1.8872769437251424e-01; CMatAdd_ref_solution(5, 5) = -2.8934762693075433e-01;
    // clang-format on

    // evaluate the method
    lin_scalar_iso_->evaluate_additional_cmat(
        &FM_, iFin_other, iFin_lin_scalar_iso_solution_, iCV, dSdiFin_, CMatAdd);

    // compare the results
    FOUR_C_EXPECT_NEAR(CMatAdd, CMatAdd_ref_solution, 1.0e-10);

    // clear matrix before test of next method
    CMatAdd.clear();

    // test InelasticDefgradLinScalarAniso set up reference solution
    // clang-format off
    CMatAdd_ref_solution(0, 0) = -2.5348465151755443e+02; CMatAdd_ref_solution(0, 1) = -2.1292469035256323e+02; CMatAdd_ref_solution(0, 2) = -1.8232541485396175e+02; CMatAdd_ref_solution(0, 3) =  9.8893368677541016e+00; CMatAdd_ref_solution(0, 4) =  1.0588628266226561e+01; CMatAdd_ref_solution(0, 5) =  1.6233942090328238e+01;
    CMatAdd_ref_solution(1, 0) = -1.0402849846169919e+02; CMatAdd_ref_solution(1, 1) = -8.7382946818243298e+01; CMatAdd_ref_solution(1, 2) = -7.4825197601167204e+01; CMatAdd_ref_solution(1, 3) =  4.0585213304843331e+00; CMatAdd_ref_solution(1, 4) =  4.3455060995216570e+00; CMatAdd_ref_solution(1, 5) =  6.6623072034563453e+00;
    CMatAdd_ref_solution(2, 0) = -6.6413171948881100e+01; CMatAdd_ref_solution(2, 1) = -5.5786431201605694e+01; CMatAdd_ref_solution(2, 2) = -4.7769397692738316e+01; CMatAdd_ref_solution(2, 3) =  2.5910137987706867e+00; CMatAdd_ref_solution(2, 4) =  2.7742286782952910e+00; CMatAdd_ref_solution(2, 5) =  4.2533052040765602e+00;
    CMatAdd_ref_solution(3, 0) = -8.9367069403395547e+01; CMatAdd_ref_solution(3, 1) = -7.5067486202872757e+01; CMatAdd_ref_solution(3, 2) = -6.4279584210361918e+01; CMatAdd_ref_solution(3, 3) =  3.4865268919563563e+00; CMatAdd_ref_solution(3, 4) =  3.7330649863393925e+00; CMatAdd_ref_solution(3, 5) =  5.7233438821308598e+00;
    CMatAdd_ref_solution(4, 0) = -2.4741829181888583e+01; CMatAdd_ref_solution(4, 1) = -2.0782900604712967e+01; CMatAdd_ref_solution(4, 2) = -1.7796202818699221e+01; CMatAdd_ref_solution(4, 3) =  9.6526666225856417e-01; CMatAdd_ref_solution(4, 4) =  1.0335222675813636e+00; CMatAdd_ref_solution(4, 5) =  1.5845433628542860e+00;
    CMatAdd_ref_solution(5, 0) = -5.0871154285791278e+01; CMatAdd_ref_solution(5, 1) = -4.2731284554439576e+01; CMatAdd_ref_solution(5, 2) = -3.6590398092069350e+01; CMatAdd_ref_solution(5, 3) =  1.9846644701043878e+00; CMatAdd_ref_solution(5, 4) =  2.1250033837602929e+00; CMatAdd_ref_solution(5, 5) =  3.2579462614386228e+00;
    // clang-format on

    // evaluate the method
    lin_scalar_aniso_->evaluate_additional_cmat(
        &FM_, iFin_other, iFin_lin_scalar_aniso_solution_, iCV, dSdiFin_, CMatAdd);

    // compare the results
    FOUR_C_EXPECT_NEAR(CMatAdd, CMatAdd_ref_solution, 1.0e-10);

    // clear matrix before test of next method
    CMatAdd.clear();

    // test InelasticDefgradPolyIntercalFracIso set up reference solution
    // clang-format off
    CMatAdd_ref_solution(0, 0) = -8.9980093707125818e+01; CMatAdd_ref_solution(0, 1) = -7.5582420772949916e+01; CMatAdd_ref_solution(0, 2) = -6.4720517852000711e+01; CMatAdd_ref_solution(0, 3) =  3.5104431480744864e+00; CMatAdd_ref_solution(0, 4) =  3.7586724005615206e+00; CMatAdd_ref_solution(0, 5) =  5.7626038569938007e+00;
    CMatAdd_ref_solution(1, 0) = -8.3416389987696661e+01; CMatAdd_ref_solution(1, 1) = -7.0068972232146919e+01; CMatAdd_ref_solution(1, 2) = -5.9999403589425583e+01; CMatAdd_ref_solution(1, 3) =  3.2543697456299596e+00; CMatAdd_ref_solution(1, 4) =  3.4844916234667358e+00; CMatAdd_ref_solution(1, 5) =  5.3422439439128073e+00;
    CMatAdd_ref_solution(2, 0) = -7.8509543517832142e+01; CMatAdd_ref_solution(2, 1) = -6.5947267983197094e+01; CMatAdd_ref_solution(2, 2) = -5.6470026907694674e+01; CMatAdd_ref_solution(2, 3) =  3.0629362311811383e+00; CMatAdd_ref_solution(2, 4) =  3.2795215279686927e+00; CMatAdd_ref_solution(2, 5) =  5.0279943001532299e+00;
    CMatAdd_ref_solution(3, 0) =  1.6492102161051103e+00; CMatAdd_ref_solution(3, 1) =  1.3853208566600168e+00; CMatAdd_ref_solution(3, 2) =  1.1862372535480017e+00; CMatAdd_ref_solution(3, 3) = -6.4341550051110280e-02; CMatAdd_ref_solution(3, 4) = -6.8891247681680007e-02; CMatAdd_ref_solution(3, 5) = -1.0562052961685520e-01;
    CMatAdd_ref_solution(4, 0) =  1.7282295618535779e+00; CMatAdd_ref_solution(4, 1) =  1.4516963536560918e+00; CMatAdd_ref_solution(4, 2) =  1.2430739689421100e+00; CMatAdd_ref_solution(4, 3) = -6.7424375478597795e-02; CMatAdd_ref_solution(4, 4) = -7.2192064803986156e-02; CMatAdd_ref_solution(4, 5) = -1.1068117323064577e-01;
    CMatAdd_ref_solution(5, 0) =  2.6573441271863643e+00; CMatAdd_ref_solution(5, 1) =  2.2321437296260704e+00; CMatAdd_ref_solution(5, 2) =  1.9113637354308992e+00; CMatAdd_ref_solution(5, 3) = -1.0367243574695899e-01; CMatAdd_ref_solution(5, 4) = -1.1100328548400526e-01; CMatAdd_ref_solution(5, 5) = -1.7018454733473193e-01;
    // clang-format on

    // evaluate the method
    poly_intercal_frac_iso_->evaluate_additional_cmat(
        &FM_, iFin_other, iFin_poly_intercal_frac_iso_solution_, iCV, dSdiFin_, CMatAdd);

    // compare the results
    FOUR_C_EXPECT_NEAR(CMatAdd, CMatAdd_ref_solution, 1.0e-10);

    // clear matrix before test of next method
    CMatAdd.clear();

    // test InelasticDefgradPolyIntercalFracAniso set up reference solution
    // clang-format off
    CMatAdd_ref_solution(0, 0) = -1.5036309611574805e+02; CMatAdd_ref_solution(0, 1) = -1.2630356705712084e+02; CMatAdd_ref_solution(0, 2) = -1.0815255958852929e+02; CMatAdd_ref_solution(0, 3) =  5.8661986083372124e+00; CMatAdd_ref_solution(0, 4) =  6.2810072333641545e+00; CMatAdd_ref_solution(0, 5) =  9.6297183291054971e+00;
    CMatAdd_ref_solution(1, 0) = -6.1708064055666021e+01; CMatAdd_ref_solution(1, 1) = -5.1834185433507010e+01; CMatAdd_ref_solution(1, 2) = -4.4385126718431692e+01; CMatAdd_ref_solution(1, 3) =  2.4074508229590710e+00; CMatAdd_ref_solution(1, 4) =  2.5776856602645020e+00; CMatAdd_ref_solution(1, 5) =  3.9519755235222713e+00;
    CMatAdd_ref_solution(2, 0) = -3.9395245815937521e+01; CMatAdd_ref_solution(2, 1) = -3.3091630860106335e+01; CMatAdd_ref_solution(2, 2) = -2.8336053065395109e+01; CMatAdd_ref_solution(2, 3) =  1.5369485076488196e+00; CMatAdd_ref_solution(2, 4) =  1.6456286836471103e+00; CMatAdd_ref_solution(2, 5) =  2.5229935437171265e+00;
    CMatAdd_ref_solution(3, 0) = -5.3011135648009507e+01; CMatAdd_ref_solution(3, 1) = -4.4528848494436232e+01; CMatAdd_ref_solution(3, 2) = -3.8129635230532394e+01; CMatAdd_ref_solution(3, 3) =  2.0681527462386371e+00; CMatAdd_ref_solution(3, 4) =  2.2143952542562975e+00; CMatAdd_ref_solution(3, 5) =  3.3949972951033844e+00;
    CMatAdd_ref_solution(4, 0) = -1.4676462724995005e+01; CMatAdd_ref_solution(4, 1) = -1.2328088752048501e+01; CMatAdd_ref_solution(4, 2) = -1.0556426745775207e+01; CMatAdd_ref_solution(4, 3) =  5.7258095527910324e-01; CMatAdd_ref_solution(4, 4) =  6.1306910350483479e-01; CMatAdd_ref_solution(4, 5) =  9.3992612389760388e-01;
    CMatAdd_ref_solution(5, 0) = -3.0175966140749807e+01; CMatAdd_ref_solution(5, 1) = -2.5347523836817480e+01; CMatAdd_ref_solution(5, 2) = -2.1704846870581804e+01; CMatAdd_ref_solution(5, 3) =  1.1772716521069135e+00; CMatAdd_ref_solution(5, 4) =  1.2605184815953689e+00; CMatAdd_ref_solution(5, 5) =  1.9325623224754220e+00;
    // clang-format on

    // evaluate the method
    poly_intercal_frac_aniso_->evaluate_additional_cmat(
        &FM_, iFin_other, iFin_poly_intercal_frac_aniso_solution_, iCV, dSdiFin_, CMatAdd);

    // compare the results
    FOUR_C_EXPECT_NEAR(CMatAdd, CMatAdd_ref_solution, 1.0e-10);

    // clear matrix before test of next method
    CMatAdd.clear();

    // test InelasticDefgradLinTempIso: set up reference solution
    CMatAdd_ref_solution.put_scalar(0.0);

    // evaluate the method
    lin_temp_iso_->evaluate_additional_cmat(
        &FM_, iFin_other, iFin_lin_temp_iso_solution_, iCV, dSdiFin_, CMatAdd);

    // compare the results
    FOUR_C_EXPECT_NEAR(CMatAdd, CMatAdd_ref_solution, 1.0e-16);
  }

  TEST_F(InelasticDefgradFactorsTest, TestEvaluateODStiffMat)
  {
    Core::LinAlg::Matrix<6, 1> dSdc(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<6, 1> dSdc_ref_solution(Core::LinAlg::Initialization::zero);

    // test InelasticDefgradLinScalarIso set up reference solution
    // clang-format off
    dSdc_ref_solution(0) = -4.1608801904598307e-03; dSdc_ref_solution(1) = -3.8573598932803953e-03; dSdc_ref_solution(2) = -3.6304563701462541e-03; dSdc_ref_solution(3) = 7.6263158165085275e-05; dSdc_ref_solution(4) = 7.9917188927245458e-05; dSdc_ref_solution(5) = 1.2288151837264513e-04;
    // clang-format on

    // evaluate the method
    lin_scalar_iso_->evaluate_od_stiff_mat(
        &FM_, Core::LinAlg::identity_matrix<3>(), iFin_lin_scalar_iso_solution_, dSdiFin_, dSdc);

    // compare the results
    FOUR_C_EXPECT_NEAR(dSdc, dSdc_ref_solution, 1.0e-10);

    // clear the derivative to be refilled again by next method to be tested
    dSdc.clear();

    // test InelasticDefgradLinScalarAniso set up reference solution
    // clang-format off
    dSdc_ref_solution(0) = -6.8943072884109979e-03; dSdc_ref_solution(1) = -2.8293801255942189e-03; dSdc_ref_solution(2) = -1.8063137656362403e-03; dSdc_ref_solution(3) = -2.4306167424464095e-03; dSdc_ref_solution(4) = -6.7293136778145881e-04; dSdc_ref_solution(5) = -1.3836000233652130e-03;
    // clang-format on

    // evaluate the method
    lin_scalar_aniso_->evaluate_od_stiff_mat(
        &FM_, Core::LinAlg::identity_matrix<3>(), iFin_lin_scalar_aniso_solution_, dSdiFin_, dSdc);

    // compare the results
    FOUR_C_EXPECT_NEAR(dSdc, dSdc_ref_solution, 1.0e-10);

    // clear the derivative to be refilled again by next method to be tested
    dSdc.clear();

    // test InelasticDefgradPolyIntercalFracIso set up reference solution
    // clang-format off
    dSdc_ref_solution(0) = -4.7912028948510080e-03; dSdc_ref_solution(1) = -4.4417029669688171e-03; dSdc_ref_solution(2) = -4.1804263218530044e-03; dSdc_ref_solution(3) = 8.7816098384379664e-05; dSdc_ref_solution(4) = 9.2023670331699243e-05; dSdc_ref_solution(5) = 1.4149657274453755e-04;
    // clang-format on

    // evaluate the method
    poly_intercal_frac_iso_->evaluate_od_stiff_mat(&FM_, Core::LinAlg::identity_matrix<3>(),
        iFin_poly_intercal_frac_iso_solution_, dSdiFin_, dSdc);

    // compare the results
    FOUR_C_EXPECT_NEAR(dSdc, dSdc_ref_solution, 1.0e-10);

    // clear the derivative to be refilled again by next method to be tested
    dSdc.clear();

    // test InelasticDefgradPolyIntercalFracAniso set up reference solution
    // clang-format off
    dSdc_ref_solution(0) = -8.0064386655720986e-03; dSdc_ref_solution(1) = -3.2857918119254472e-03; dSdc_ref_solution(2) = -2.0976930343176458e-03; dSdc_ref_solution(3) = -2.8227032903830871e-03; dSdc_ref_solution(4) = -7.8148296803340908e-04; dSdc_ref_solution(5) = -1.6067906841603652e-03;
    // clang-format on

    // evaluate the method
    poly_intercal_frac_aniso_->evaluate_od_stiff_mat(&FM_, Core::LinAlg::identity_matrix<3>(),
        iFin_poly_intercal_frac_aniso_solution_, dSdiFin_, dSdc);

    // compare the results
    FOUR_C_EXPECT_NEAR(dSdc, dSdc_ref_solution, 1.0e-10);

    // clear the derivative to be refilled again by next method to be tested
    dSdc.clear();

    // test InelasticDefgradTempIso: set up reference solution
    // clang-format off
    dSdc_ref_solution(0) = -4.822047213115778; dSdc_ref_solution(1) = -4.470297310176029; dSdc_ref_solution(2) = -4.207338644871557; dSdc_ref_solution(3) = 0.08838143192311355; dSdc_ref_solution(4) = 0.09261609094879596; dSdc_ref_solution(5) = 0.1424074849765778;
    // clang-format on

    // evaluate the method
    lin_temp_iso_->evaluate_od_stiff_mat(
        &FM_, Core::LinAlg::identity_matrix<3>(), iFin_lin_temp_iso_solution_, dSdiFin_, dSdc);

    // compare the results
    FOUR_C_EXPECT_NEAR(dSdc, dSdc_ref_solution, 1.0e-15);
  }

  TEST_F(InelasticDefgradFactorsTest, TestEvaluateStateQuantities)
  {
    auto transv_isotropic_material = set_up_viscoplastic_material(
        {.mat_behavior = ViscoplastUtils::MatBehavior::transv_isotropic})
                                         .material;
    auto isotropic_material =
        set_up_viscoplastic_material({.mat_behavior = ViscoplastUtils::MatBehavior::isotropic})
            .material;

    // reference temperature with which the material was constructed.
    auto default_setup = ViscoplasticMaterialSetup{};
    const double ref_temperature = default_setup.ref_temperature;

    Teuchos::ParameterList params_list{};
    double total_time = 1.0e-6;
    double time_step_size = 1.0e-6;
    Mat::EvaluationContext<3> context{.total_time = &total_time,
        .time_step_size = &time_step_size,
        .xi = {},
        .ref_coords = nullptr};

    transv_isotropic_material->pre_evaluate(params_list, context, 0, 0);
    isotropic_material->pre_evaluate(params_list, context, 0, 0);

    // set reference solution for the state quantities
    set_up_state_quantities_solution();

    // compute right Cauchy-Green deformation tensor
    Core::LinAlg::Matrix<3, 3> CM(Core::LinAlg::Initialization::zero);
    CM.multiply_tn(1.0, FM_, FM_, 0.0);

    // declare error status
    Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType err_status =
        Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType::no_errors;

    // compute StateQuantities objects
    Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::StateQuantities
        computed_state_quantities_transv_isotrop =
            transv_isotropic_material->evaluate_state_quantities(CM, ref_temperature,
                iFin_transv_isotrop_vplast_refJC_solution_,
                plastic_strain_transv_isotrop_vplast_refJC_solution_, err_status, 1.0,
                Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::StateQuantityEvalType::
                    full_eval);
    Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::StateQuantities
        computed_state_quantities_isotrop = isotropic_material->evaluate_state_quantities(CM,
            ref_temperature, iFin_transv_isotrop_vplast_refJC_solution_,
            plastic_strain_transv_isotrop_vplast_refJC_solution_, err_status, 1.0,
            Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::StateQuantityEvalType::
                full_eval);

    if (err_status != Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType::no_errors)
    {
      FOUR_C_THROW("Error encountered during testing of TestEvaluateStateQuantities");
    }


    // compare the results
    FOUR_C_EXPECT_NEAR(state_quantities_solution_transv_isotrop_.curr_CeM,
        computed_state_quantities_transv_isotrop.curr_CeM, 1.0e-10);
    FOUR_C_EXPECT_NEAR(state_quantities_solution_isotrop_.curr_CeM,
        computed_state_quantities_isotrop.curr_CeM, 1.0e-10);
    FOUR_C_EXPECT_NEAR(state_quantities_solution_transv_isotrop_.curr_gamma,
        computed_state_quantities_transv_isotrop.curr_gamma, 1.0e-10);
    FOUR_C_EXPECT_NEAR(state_quantities_solution_isotrop_.curr_gamma,
        computed_state_quantities_isotrop.curr_gamma, 1.0e-10);
    FOUR_C_EXPECT_NEAR(state_quantities_solution_transv_isotrop_.curr_delta,
        computed_state_quantities_transv_isotrop.curr_delta, 1.0e-10);
    FOUR_C_EXPECT_NEAR(state_quantities_solution_isotrop_.curr_delta,
        computed_state_quantities_isotrop.curr_delta, 1.0e-10);
    FOUR_C_EXPECT_NEAR(state_quantities_solution_transv_isotrop_.curr_Mtheta_dev_sym_M,
        computed_state_quantities_transv_isotrop.curr_Mtheta_dev_sym_M, 1.0e-10);
    FOUR_C_EXPECT_NEAR(state_quantities_solution_isotrop_.curr_Mtheta_dev_sym_M,
        computed_state_quantities_isotrop.curr_Mtheta_dev_sym_M, 1.0e-10);
    EXPECT_NEAR(state_quantities_solution_transv_isotrop_.curr_equiv_stress,
        computed_state_quantities_transv_isotrop.curr_equiv_stress, 1.0e-8);
    EXPECT_NEAR(state_quantities_solution_isotrop_.curr_equiv_stress,
        computed_state_quantities_isotrop.curr_equiv_stress, 1.0e-8);
    EXPECT_NEAR(state_quantities_solution_transv_isotrop_.curr_equiv_plastic_strain_rate,
        computed_state_quantities_transv_isotrop.curr_equiv_plastic_strain_rate, 1.0e-8);
    EXPECT_NEAR(state_quantities_solution_isotrop_.curr_equiv_plastic_strain_rate,
        computed_state_quantities_isotrop.curr_equiv_plastic_strain_rate, 1.0e-8);
    FOUR_C_EXPECT_NEAR(state_quantities_solution_transv_isotrop_.curr_NpM,
        computed_state_quantities_transv_isotrop.curr_NpM, 1.0e-10);
    FOUR_C_EXPECT_NEAR(state_quantities_solution_isotrop_.curr_NpM,
        computed_state_quantities_isotrop.curr_NpM, 1.0e-10);
    FOUR_C_EXPECT_NEAR(state_quantities_solution_transv_isotrop_.curr_dpM,
        computed_state_quantities_transv_isotrop.curr_dpM, 1.0e-10);
    FOUR_C_EXPECT_NEAR(state_quantities_solution_isotrop_.curr_dpM,
        computed_state_quantities_isotrop.curr_dpM, 1.0e-10);
    FOUR_C_EXPECT_NEAR(state_quantities_solution_transv_isotrop_.curr_lpM,
        computed_state_quantities_transv_isotrop.curr_lpM, 1.0e-10);
    FOUR_C_EXPECT_NEAR(state_quantities_solution_isotrop_.curr_lpM,
        computed_state_quantities_isotrop.curr_lpM, 1.0e-10);
  }

  TEST_F(InelasticDefgradFactorsTest, TestLocalNewtonParametersParsing)
  {
    const auto local_newton_params = set_up_viscoplastic_material().params->local_newton_params();

    EXPECT_EQ(local_newton_params.conv_check,
        Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonConvCheck::
            residual_and_increment_ratio);
    EXPECT_EQ(local_newton_params.diver_cont,
        Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonDiverCont::stop);
    EXPECT_EQ(local_newton_params.max_iter, 100);
    EXPECT_DOUBLE_EQ(local_newton_params.res_tol, 1.0e-8);
    EXPECT_DOUBLE_EQ(local_newton_params.incr_tol, 1.0e-8);
    EXPECT_DOUBLE_EQ(local_newton_params.max_exceedance_fact_res_tol, 1.0e1);
    EXPECT_DOUBLE_EQ(local_newton_params.max_exceedance_fact_incr_tol, 1.0e1);
  }

  TEST_F(InelasticDefgradFactorsTest, TestLocalNewtonManagerBookkeeping)
  {
    const auto material = set_up_viscoplastic_material();
    Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonManager local_newton_manager(
        material.params->local_newton_params());

    EXPECT_EQ(local_newton_manager.iter(), 0);
    EXPECT_EQ(local_newton_manager.curr_num_iters().size(), 1);
    EXPECT_EQ(local_newton_manager.curr_num_iters()[0], 0);

    local_newton_manager.resize(3);
    EXPECT_EQ(local_newton_manager.curr_num_iters().size(), 3);
    EXPECT_EQ(local_newton_manager.curr_num_iters()[0], 0);
    EXPECT_EQ(local_newton_manager.curr_num_iters()[1], 0);
    EXPECT_EQ(local_newton_manager.curr_num_iters()[2], 0);

    local_newton_manager.set_iteration_count(4);
    local_newton_manager.update_after_local_newton(1);
    EXPECT_EQ(local_newton_manager.curr_num_iters()[1], 4);

    local_newton_manager.set_iteration_count(2);
    local_newton_manager.update_after_local_newton(1);
    EXPECT_EQ(local_newton_manager.curr_num_iters()[1], 6);

    local_newton_manager.reset();
    EXPECT_EQ(local_newton_manager.curr_num_iters()[0], 0);
    EXPECT_EQ(local_newton_manager.curr_num_iters()[1], 0);
    EXPECT_EQ(local_newton_manager.curr_num_iters()[2], 0);
  }

  TEST_F(InelasticDefgradFactorsTest, TestLocalNewtonDivergenceHandlingStop)
  {
    Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonParams local_newton_params{
        .res_tol = 1.0e-8,
        .incr_tol = 1.0e-8,
        .conv_check = FourC::Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::
            LocalNewtonConvCheck::residual_and_increment_ratio,
        .diver_cont = FourC::Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::
            LocalNewtonDiverCont::stop,
        .max_iter = 0,
        .max_exceedance_fact_res_tol = 1.0e1,
        .max_exceedance_fact_incr_tol = 1.0e1,
    };


    auto material =
        set_up_viscoplastic_material({.local_newton_params = local_newton_params}).material;

    Teuchos::ParameterList params_list;
    double total_time = 1.0e-6;
    double time_step_size = 1.0e-6;
    Mat::EvaluationContext<3> context{.total_time = &total_time,
        .time_step_size = &time_step_size,
        .xi = {},
        .ref_coords = nullptr};
    material->pre_evaluate(params_list, context, 0, 0);


    Core::LinAlg::Matrix<3, 3> unit_3x3(Core::LinAlg::Initialization::zero);
    unit_3x3(0, 0) = 1.0;
    unit_3x3(1, 1) = 1.0;
    unit_3x3(2, 2) = 1.0;
    Core::LinAlg::Matrix<3, 3> iFin_other(unit_3x3);

    Core::LinAlg::Matrix<3, 3> iFin_result(Core::LinAlg::Initialization::zero);
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        material->evaluate_inverse_inelastic_def_grad(&FM_, iFin_other, iFin_result),
        Core::Exception, "Local Newton Loop did not converge");
  }

  TEST_F(InelasticDefgradFactorsTest, TestLocalNewtonDivergenceHandlingContinue)
  {
    Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonParams local_newton_params{
        .res_tol = 1.0e-8,
        .incr_tol = 1.0e-8,
        .conv_check = FourC::Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::
            LocalNewtonConvCheck::residual_and_increment_ratio,
        .diver_cont = FourC::Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::
            LocalNewtonDiverCont::continue_sim,
        .max_iter = 0,
        .max_exceedance_fact_res_tol = 1.0e1,
        .max_exceedance_fact_incr_tol = 1.0e1,
    };

    auto material =
        set_up_viscoplastic_material({.local_newton_params = local_newton_params}).material;

    Teuchos::ParameterList params_list;
    double total_time = 1.0e-6;
    double time_step_size = 1.0e-6;
    Mat::EvaluationContext<3> context{.total_time = &total_time,
        .time_step_size = &time_step_size,
        .xi = {},
        .ref_coords = nullptr};
    material->pre_evaluate(params_list, context, 0, 0);



    Core::LinAlg::Matrix<3, 3> unit_3x3(Core::LinAlg::Initialization::zero);
    unit_3x3(0, 0) = 1.0;
    unit_3x3(1, 1) = 1.0;
    unit_3x3(2, 2) = 1.0;
    Core::LinAlg::Matrix<3, 3> iFin_other(unit_3x3);

    Core::LinAlg::Matrix<3, 3> iFin_result(Core::LinAlg::Initialization::zero);
    material->evaluate_inverse_inelastic_def_grad(&FM_, iFin_other, iFin_result);

    FOUR_C_EXPECT_NEAR(iFin_result, unit_3x3,
        1.0e-15);  // since we continue with 0 iterations, the elastic predictor must be the
                   // solution
  }

  TEST_F(InelasticDefgradFactorsTest, TestLocalNewtonDivergenceHandlingContinueWithSafeguard)
  {
    Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonParams local_newton_params{
        .res_tol = 1.0e-3,    // solution for the following settings: 0.013683468721433008
        .incr_tol = 1.0e-12,  // solution for the following settings: 5.7240307737697736e-11,
        .conv_check = FourC::Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::
            LocalNewtonConvCheck::residual_and_increment_ratio,
        .diver_cont = FourC::Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::
            LocalNewtonDiverCont::continue_sim_with_safeguard,
        .max_iter = 2,
        .max_exceedance_fact_res_tol = 1.0e2,
        .max_exceedance_fact_incr_tol = 1.0e2,
    };

    auto material =
        set_up_viscoplastic_material({.local_newton_params = local_newton_params}).material;

    Teuchos::ParameterList params_list;
    double total_time = 1.0e-6;
    double time_step_size = 1.0e-6;
    Mat::EvaluationContext<3> context{.total_time = &total_time,
        .time_step_size = &time_step_size,
        .xi = {},
        .ref_coords = nullptr};
    material->pre_evaluate(params_list, context, 0, 0);

    Core::LinAlg::Matrix<3, 3> unit_3x3(Core::LinAlg::Initialization::zero);
    unit_3x3(0, 0) = 1.0;
    unit_3x3(1, 1) = 1.0;
    unit_3x3(2, 2) = 1.0;
    Core::LinAlg::Matrix<3, 3> iFin_other(unit_3x3);

    Core::LinAlg::Matrix<3, 3> iFin_result(Core::LinAlg::Initialization::zero);
    material->evaluate_inverse_inelastic_def_grad(&FM_, iFin_other, iFin_result);


    Core::LinAlg::Matrix<3, 3> iFin_result_ref(Core::LinAlg::Initialization::zero);
    iFin_result_ref(0, 0) = 1.0000000000451326;
    iFin_result_ref(0, 1) = -0.0000000000119367;
    iFin_result_ref(0, 2) = -0.0000000000213770;
    iFin_result_ref(1, 0) = -0.0000000000119367;
    iFin_result_ref(1, 1) = 1.0000000000009559;
    iFin_result_ref(1, 2) = -0.0000000000168552;
    iFin_result_ref(2, 0) = -0.0000000000213770;
    iFin_result_ref(2, 1) = -0.0000000000168552;
    iFin_result_ref(2, 2) = 0.9999999999539116;

    FOUR_C_EXPECT_NEAR(iFin_result, iFin_result_ref, 1.0e-10);
  }


  TEST_F(InelasticDefgradFactorsTest, TestLocalNewtonDivergenceHandlingStopWithSafeguard)
  {
    Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonParams local_newton_params{
        .res_tol = 1.0e-3,    // solution for the following settings: 0.013683468721433008
        .incr_tol = 1.0e-12,  // solution for the following settings: 5.7240307737697736e-11,
        .conv_check = FourC::Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::
            LocalNewtonConvCheck::residual_and_increment_ratio,
        .diver_cont = FourC::Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::
            LocalNewtonDiverCont::continue_sim_with_safeguard,
        .max_iter = 2,
        .max_exceedance_fact_res_tol = 0.0,
        .max_exceedance_fact_incr_tol = 0.0,
    };

    auto material =
        set_up_viscoplastic_material({.local_newton_params = local_newton_params}).material;

    Teuchos::ParameterList params_list;
    double total_time = 1.0e-6;
    double time_step_size = 1.0e-6;
    Mat::EvaluationContext<3> context{.total_time = &total_time,
        .time_step_size = &time_step_size,
        .xi = {},
        .ref_coords = nullptr};
    material->pre_evaluate(params_list, context, 0, 0);

    Core::LinAlg::Matrix<3, 3> unit_3x3(Core::LinAlg::Initialization::zero);
    unit_3x3(0, 0) = 1.0;
    unit_3x3(1, 1) = 1.0;
    unit_3x3(2, 2) = 1.0;
    Core::LinAlg::Matrix<3, 3> iFin_other(unit_3x3);

    Core::LinAlg::Matrix<3, 3> iFin_result(Core::LinAlg::Initialization::zero);
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        material->evaluate_inverse_inelastic_def_grad(&FM_, iFin_other, iFin_result),
        Core::Exception, "by more than the set exceedance tolerance");
  }


  TEST_F(InelasticDefgradFactorsTest, TestLocalNewtonResidualDivergence)
  {
    Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonParams local_newton_params{
        .res_tol = 1.0e-3,  // solution for the following settings: 0.013683468721433008
        .incr_tol = 0.0,    // solution for the following settings: 5.7240307737697736e-11,
        .conv_check = FourC::Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::
            LocalNewtonConvCheck::residual,
        .diver_cont = FourC::Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::
            LocalNewtonDiverCont::stop,
        .max_iter = 2,
        .max_exceedance_fact_res_tol = 0.0,
        .max_exceedance_fact_incr_tol = 0.0,
    };

    auto material =
        set_up_viscoplastic_material({.local_newton_params = local_newton_params}).material;

    Teuchos::ParameterList params_list;
    double total_time = 1.0e-6;
    double time_step_size = 1.0e-6;
    Mat::EvaluationContext<3> context{.total_time = &total_time,
        .time_step_size = &time_step_size,
        .xi = {},
        .ref_coords = nullptr};
    material->pre_evaluate(params_list, context, 0, 0);

    Core::LinAlg::Matrix<3, 3> unit_3x3(Core::LinAlg::Initialization::zero);
    unit_3x3(0, 0) = 1.0;
    unit_3x3(1, 1) = 1.0;
    unit_3x3(2, 2) = 1.0;
    Core::LinAlg::Matrix<3, 3> iFin_other(unit_3x3);

    Core::LinAlg::Matrix<3, 3> iFin_result(Core::LinAlg::Initialization::zero);
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        material->evaluate_inverse_inelastic_def_grad(&FM_, iFin_other, iFin_result),
        Core::Exception, "Local Newton Loop did not converge");
  }

  TEST_F(InelasticDefgradFactorsTest, TestLocalNewtonResidualConvergence)
  {
    Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonParams local_newton_params{
        .res_tol = 1.0e-1,  // solution for the following settings: 0.013683468721433008
        .incr_tol = 0.0,    // solution for the following settings: 5.7240307737697736e-11,
        .conv_check = FourC::Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::
            LocalNewtonConvCheck::residual,
        .diver_cont = FourC::Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::
            LocalNewtonDiverCont::stop,
        .max_iter = 2,
        .max_exceedance_fact_res_tol = 0.0,
        .max_exceedance_fact_incr_tol = 0.0,
    };

    auto material =
        set_up_viscoplastic_material({.local_newton_params = local_newton_params}).material;

    Teuchos::ParameterList params_list;
    double total_time = 1.0e-6;
    double time_step_size = 1.0e-6;
    Mat::EvaluationContext<3> context{.total_time = &total_time,
        .time_step_size = &time_step_size,
        .xi = {},
        .ref_coords = nullptr};
    material->pre_evaluate(params_list, context, 0, 0);

    Core::LinAlg::Matrix<3, 3> unit_3x3(Core::LinAlg::Initialization::zero);
    unit_3x3(0, 0) = 1.0;
    unit_3x3(1, 1) = 1.0;
    unit_3x3(2, 2) = 1.0;
    Core::LinAlg::Matrix<3, 3> iFin_other(unit_3x3);

    Core::LinAlg::Matrix<3, 3> iFin_result(Core::LinAlg::Initialization::zero);
    material->evaluate_inverse_inelastic_def_grad(&FM_, iFin_other, iFin_result);

    FOUR_C_EXPECT_NEAR(iFin_result, unit_3x3,
        1.0e-15);  // elastic predictor directly suffices for the posed conditions
  }

  TEST_F(InelasticDefgradFactorsTest, TestLocalNewtonIncrementRatioDivergence)
  {
    Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonParams local_newton_params{
        .res_tol = 0.0,       // solution for the following settings: 0.013683468721433008
        .incr_tol = 1.0e-15,  // solution for the following settings: 5.7240307737697736e-11,
        .conv_check = FourC::Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::
            LocalNewtonConvCheck::increment_ratio,
        .diver_cont = FourC::Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::
            LocalNewtonDiverCont::stop,
        .max_iter = 2,
        .max_exceedance_fact_res_tol = 0.0,
        .max_exceedance_fact_incr_tol = 0.0,
    };

    auto material =
        set_up_viscoplastic_material({.local_newton_params = local_newton_params}).material;

    Teuchos::ParameterList params_list;
    double total_time = 1.0e-6;
    double time_step_size = 1.0e-6;
    Mat::EvaluationContext<3> context{.total_time = &total_time,
        .time_step_size = &time_step_size,
        .xi = {},
        .ref_coords = nullptr};
    material->pre_evaluate(params_list, context, 0, 0);

    Core::LinAlg::Matrix<3, 3> unit_3x3(Core::LinAlg::Initialization::zero);
    unit_3x3(0, 0) = 1.0;
    unit_3x3(1, 1) = 1.0;
    unit_3x3(2, 2) = 1.0;
    Core::LinAlg::Matrix<3, 3> iFin_other(unit_3x3);

    Core::LinAlg::Matrix<3, 3> iFin_result(Core::LinAlg::Initialization::zero);
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        material->evaluate_inverse_inelastic_def_grad(&FM_, iFin_other, iFin_result),
        Core::Exception, "Local Newton Loop did not converge");
  }

  TEST_F(InelasticDefgradFactorsTest, TestLocalNewtonIncrementRatioConvergence)
  {
    Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonParams local_newton_params{
        .res_tol = 0.0,       // solution for the following settings: 0.013683468721433008
        .incr_tol = 1.0e-10,  // solution for the following settings: 5.7240307737697736e-11,
        .conv_check = FourC::Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::
            LocalNewtonConvCheck::increment_ratio,
        .diver_cont = FourC::Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::
            LocalNewtonDiverCont::stop,
        .max_iter = 2,
        .max_exceedance_fact_res_tol = 0.0,
        .max_exceedance_fact_incr_tol = 0.0,
    };

    auto material =
        set_up_viscoplastic_material({.local_newton_params = local_newton_params}).material;

    Teuchos::ParameterList params_list;
    double total_time = 1.0e-6;
    double time_step_size = 1.0e-6;
    Mat::EvaluationContext<3> context{.total_time = &total_time,
        .time_step_size = &time_step_size,
        .xi = {},
        .ref_coords = nullptr};
    material->pre_evaluate(params_list, context, 0, 0);

    Core::LinAlg::Matrix<3, 3> unit_3x3(Core::LinAlg::Initialization::zero);
    unit_3x3(0, 0) = 1.0;
    unit_3x3(1, 1) = 1.0;
    unit_3x3(2, 2) = 1.0;
    Core::LinAlg::Matrix<3, 3> iFin_other(unit_3x3);

    Core::LinAlg::Matrix<3, 3> iFin_result(Core::LinAlg::Initialization::zero);
    material->evaluate_inverse_inelastic_def_grad(&FM_, iFin_other, iFin_result);


    Core::LinAlg::Matrix<3, 3> iFin_result_ref{Core::LinAlg::Initialization::zero};
    iFin_result_ref(0, 0) = 1.0000000000001195;
    iFin_result_ref(0, 1) = -0.0000000000000316;
    iFin_result_ref(0, 2) = -0.0000000000000566;
    iFin_result_ref(1, 0) = -0.0000000000000316;
    iFin_result_ref(1, 1) = 1.0000000000000024;
    iFin_result_ref(1, 2) = -0.0000000000000446;
    iFin_result_ref(2, 0) = -0.0000000000000566;
    iFin_result_ref(2, 1) = -0.0000000000000446;
    iFin_result_ref(2, 2) = 0.9999999999998780;

    FOUR_C_EXPECT_NEAR(iFin_result, iFin_result_ref, 1.0e-10);
  }

  TEST_F(InelasticDefgradFactorsTest, TestLocalNewtonResidualAndIncrementRatioDivergence)
  {
    Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonParams local_newton_params{
        .res_tol = 1.0e-15,   // solution for the following settings: 0.013683468721433008
        .incr_tol = 1.0e-15,  // solution for the following settings: 5.7240307737697736e-11,
        .conv_check = FourC::Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::
            LocalNewtonConvCheck::residual_and_increment_ratio,
        .diver_cont = FourC::Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::
            LocalNewtonDiverCont::stop,
        .max_iter = 2,
        .max_exceedance_fact_res_tol = 0.0,
        .max_exceedance_fact_incr_tol = 0.0,
    };

    auto material =
        set_up_viscoplastic_material({.local_newton_params = local_newton_params}).material;

    Teuchos::ParameterList params_list;
    double total_time = 1.0e-6;
    double time_step_size = 1.0e-6;
    Mat::EvaluationContext<3> context{.total_time = &total_time,
        .time_step_size = &time_step_size,
        .xi = {},
        .ref_coords = nullptr};
    material->pre_evaluate(params_list, context, 0, 0);

    Core::LinAlg::Matrix<3, 3> unit_3x3(Core::LinAlg::Initialization::zero);
    unit_3x3(0, 0) = 1.0;
    unit_3x3(1, 1) = 1.0;
    unit_3x3(2, 2) = 1.0;
    Core::LinAlg::Matrix<3, 3> iFin_other(unit_3x3);

    Core::LinAlg::Matrix<3, 3> iFin_result(Core::LinAlg::Initialization::zero);
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        material->evaluate_inverse_inelastic_def_grad(&FM_, iFin_other, iFin_result),
        Core::Exception, "Local Newton Loop did not converge");
  }

  TEST_F(InelasticDefgradFactorsTest, TestLocalNewtonResidualAndIncrementRatioConvergence)
  {
    Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonParams local_newton_params{
        .res_tol = 1.0e-1,    // solution for the following settings: 0.013683468721433008
        .incr_tol = 1.0e-10,  // solution for the following settings: 5.7240307737697736e-11,
        .conv_check = FourC::Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::
            LocalNewtonConvCheck::residual_and_increment_ratio,
        .diver_cont = FourC::Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::
            LocalNewtonDiverCont::stop,
        .max_iter = 2,
        .max_exceedance_fact_res_tol = 0.0,
        .max_exceedance_fact_incr_tol = 0.0,
    };

    auto material =
        set_up_viscoplastic_material({.local_newton_params = local_newton_params}).material;

    Teuchos::ParameterList params_list;
    double total_time = 1.0e-6;
    double time_step_size = 1.0e-6;
    Mat::EvaluationContext<3> context{.total_time = &total_time,
        .time_step_size = &time_step_size,
        .xi = {},
        .ref_coords = nullptr};
    material->pre_evaluate(params_list, context, 0, 0);

    Core::LinAlg::Matrix<3, 3> unit_3x3(Core::LinAlg::Initialization::zero);
    unit_3x3(0, 0) = 1.0;
    unit_3x3(1, 1) = 1.0;
    unit_3x3(2, 2) = 1.0;
    Core::LinAlg::Matrix<3, 3> iFin_other(unit_3x3);

    Core::LinAlg::Matrix<3, 3> iFin_result(Core::LinAlg::Initialization::zero);
    material->evaluate_inverse_inelastic_def_grad(&FM_, iFin_other, iFin_result);


    Core::LinAlg::Matrix<3, 3> iFin_result_ref{Core::LinAlg::Initialization::zero};
    iFin_result_ref(0, 0) = 1.0000000000001195;
    iFin_result_ref(0, 1) = -0.0000000000000316;
    iFin_result_ref(0, 2) = -0.0000000000000566;
    iFin_result_ref(1, 0) = -0.0000000000000316;
    iFin_result_ref(1, 1) = 1.0000000000000024;
    iFin_result_ref(1, 2) = -0.0000000000000446;
    iFin_result_ref(2, 0) = -0.0000000000000566;
    iFin_result_ref(2, 1) = -0.0000000000000446;
    iFin_result_ref(2, 2) = 0.9999999999998780;

    FOUR_C_EXPECT_NEAR(iFin_result, iFin_result_ref, 1.0e-10);
  }

  TEST_F(InelasticDefgradFactorsTest, TestViscoplasticCorrectionSubstepping)
  {
    // tests a challenging scenario, where the material formulation without substepping (one single
    // step delta_t) does not converge, while the material formulation using local substepping and
    // the same parameters does
    Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonParams local_newton_params{
        .res_tol = 1.0e-8,
        .incr_tol = 1.0e-8,
        .conv_check = FourC::Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::
            LocalNewtonConvCheck::residual_and_increment_ratio,
        .diver_cont = FourC::Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::
            LocalNewtonDiverCont::stop,
        .max_iter = 50,
        .max_exceedance_fact_res_tol = 0.0,
        .max_exceedance_fact_incr_tol = 0.0,
    };

    auto material_one_step = set_up_viscoplastic_material(
        {.local_newton_params = local_newton_params, .use_substepping = false})
                                 .material;

    auto material_substepping =
        set_up_viscoplastic_material({.local_newton_params = local_newton_params,
                                         .use_substepping = true,
                                         .max_substepping_halve_num = 10})
            .material;


    Teuchos::ParameterList params_list;
    double total_time = 1.0e-6;
    double time_step_size = 1.0e-6;
    Mat::EvaluationContext<3> context{.total_time = &total_time,
        .time_step_size = &time_step_size,
        .xi = {},
        .ref_coords = nullptr};
    material_one_step->pre_evaluate(params_list, context, 0, 0);
    material_substepping->pre_evaluate(params_list, context, 0, 0);

    Core::LinAlg::Matrix<3, 3> unit_3x3(Core::LinAlg::Initialization::zero);
    unit_3x3(0, 0) = 1.0;
    unit_3x3(1, 1) = 1.0;
    unit_3x3(2, 2) = 1.0;
    Core::LinAlg::Matrix<3, 3> iFin_other(unit_3x3);

    Core::LinAlg::Matrix<3, 3> FM(Core::LinAlg::Initialization::zero);
    FM(0, 0) = 2.0;
    FM(1, 1) = 1.0;
    FM(2, 2) = 1.0;

    Core::LinAlg::Matrix<3, 3> iFin_result(Core::LinAlg::Initialization::zero);

    // the one-step formulation fails to converge
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        material_one_step->evaluate_inverse_inelastic_def_grad(&FM, iFin_other, iFin_result),
        Core::Exception, "Local Newton evaluation has failed with err status overflow_error");


    // the local substepping formulation converges
    material_substepping->evaluate_inverse_inelastic_def_grad(&FM, iFin_other, iFin_result);
    Core::LinAlg::Matrix<3, 3> iFin_result_ref{Core::LinAlg::Initialization::zero};
    iFin_result_ref(0, 0) = 0.71055164642;
    iFin_result_ref(1, 1) = 1.18632088172;
    iFin_result_ref(2, 2) = 1.18632088172;
    FOUR_C_EXPECT_NEAR(iFin_result, iFin_result_ref, 1.0e-10);
  }



  TEST_F(InelasticDefgradFactorsTest, TestEvaluateStateQuantityDerivatives)
  {
    auto transv_isotropic_material = set_up_viscoplastic_material(
        {.mat_behavior = ViscoplastUtils::MatBehavior::transv_isotropic})
                                         .material;
    auto isotropic_material =
        set_up_viscoplastic_material({.mat_behavior = ViscoplastUtils::MatBehavior::isotropic})
            .material;

    // reference temperature with which the material was constructed.
    auto default_setup = ViscoplasticMaterialSetup{};
    const double ref_temperature = default_setup.ref_temperature;

    Teuchos::ParameterList params_list;
    double total_time = 1.0e-6;
    double time_step_size = 1.0e-6;
    Mat::EvaluationContext<3> context{.total_time = &total_time,
        .time_step_size = &time_step_size,
        .xi = {},
        .ref_coords = nullptr};

    transv_isotropic_material->pre_evaluate(params_list, context, 0, 0);
    isotropic_material->pre_evaluate(params_list, context, 0, 0);

    set_up_state_quantity_derivatives_solution();

    // compute right Cauchy-Green deformation tensor
    Core::LinAlg::Matrix<3, 3> CM(Core::LinAlg::Initialization::zero);
    CM.multiply_tn(1.0, FM_, FM_, 0.0);

    Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType err_status =
        Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType::no_errors;

    // compute StateQuantityDerivatives objects
    Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::StateQuantityDerivatives
        computed_state_quantity_derivatives_transv_isotrop =
            transv_isotropic_material->evaluate_state_quantity_derivatives(CM, ref_temperature,
                iFin_transv_isotrop_vplast_refJC_solution_,
                plastic_strain_transv_isotrop_vplast_refJC_solution_, err_status, 1.0,
                Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::StateQuantityDerivEvalType::
                    full_eval,
                true);

    Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::StateQuantityDerivatives
        computed_state_quantity_derivatives_isotrop =
            isotropic_material->evaluate_state_quantity_derivatives(CM, ref_temperature,
                iFin_transv_isotrop_vplast_refJC_solution_,
                plastic_strain_transv_isotrop_vplast_refJC_solution_, err_status, 1.0,
                Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::StateQuantityDerivEvalType::
                    full_eval,
                true);

    if (err_status != Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType::no_errors)
    {
      FOUR_C_THROW("Error encountered during testing of TestEvaluateStateQuantityDerivatives");
    }



    // compare the results
    FOUR_C_EXPECT_NEAR(state_quantity_derivatives_solution_transv_isotrop_.curr_dCedC,
        computed_state_quantity_derivatives_transv_isotrop.curr_dCedC, 1.0e-10);
    FOUR_C_EXPECT_NEAR(state_quantity_derivatives_solution_isotrop_.curr_dCedC,
        computed_state_quantity_derivatives_isotrop.curr_dCedC, 1.0e-10);
    FOUR_C_EXPECT_NEAR(state_quantity_derivatives_solution_transv_isotrop_.curr_dCediFin,
        computed_state_quantity_derivatives_transv_isotrop.curr_dCediFin, 1.0e-10);
    FOUR_C_EXPECT_NEAR(state_quantity_derivatives_solution_isotrop_.curr_dCediFin,
        computed_state_quantity_derivatives_isotrop.curr_dCediFin, 1.0e-10);
    FOUR_C_EXPECT_NEAR(state_quantity_derivatives_solution_transv_isotrop_.curr_dMtheta_dev_sym_dC,
        computed_state_quantity_derivatives_transv_isotrop.curr_dMtheta_dev_sym_dC, 1.0e-10);
    FOUR_C_EXPECT_NEAR(state_quantity_derivatives_solution_isotrop_.curr_dMtheta_dev_sym_dC,
        computed_state_quantity_derivatives_isotrop.curr_dMtheta_dev_sym_dC, 1.0e-10);
    FOUR_C_EXPECT_NEAR(
        state_quantity_derivatives_solution_transv_isotrop_.curr_dMtheta_dev_sym_diFin,
        computed_state_quantity_derivatives_transv_isotrop.curr_dMtheta_dev_sym_diFin, 1.0e-10);
    FOUR_C_EXPECT_NEAR(state_quantity_derivatives_solution_isotrop_.curr_dMtheta_dev_sym_diFin,
        computed_state_quantity_derivatives_isotrop.curr_dMtheta_dev_sym_diFin, 1.0e-10);
    FOUR_C_EXPECT_NEAR(state_quantity_derivatives_solution_transv_isotrop_.curr_dequiv_stress_diFin,
        computed_state_quantity_derivatives_transv_isotrop.curr_dequiv_stress_diFin, 1.0e-10);
    FOUR_C_EXPECT_NEAR(state_quantity_derivatives_solution_isotrop_.curr_dequiv_stress_diFin,
        computed_state_quantity_derivatives_isotrop.curr_dequiv_stress_diFin, 1.0e-10);
    FOUR_C_EXPECT_NEAR(state_quantity_derivatives_solution_transv_isotrop_.curr_dequiv_stress_dC,
        computed_state_quantity_derivatives_transv_isotrop.curr_dequiv_stress_dC, 1.0e-10);
    FOUR_C_EXPECT_NEAR(state_quantity_derivatives_solution_isotrop_.curr_dequiv_stress_dC,
        computed_state_quantity_derivatives_isotrop.curr_dequiv_stress_dC, 1.0e-10);
    EXPECT_NEAR(state_quantity_derivatives_solution_transv_isotrop_.curr_dpsr_dequiv_stress,
        computed_state_quantity_derivatives_transv_isotrop.curr_dpsr_dequiv_stress, 1.0e-8);
    EXPECT_NEAR(state_quantity_derivatives_solution_isotrop_.curr_dpsr_dequiv_stress,
        computed_state_quantity_derivatives_isotrop.curr_dpsr_dequiv_stress, 1.0e-8);
    EXPECT_NEAR(state_quantity_derivatives_solution_transv_isotrop_.curr_dpsr_depsp,
        computed_state_quantity_derivatives_transv_isotrop.curr_dpsr_depsp, 1.0e-8);
    EXPECT_NEAR(state_quantity_derivatives_solution_isotrop_.curr_dpsr_depsp,
        computed_state_quantity_derivatives_isotrop.curr_dpsr_depsp, 1.0e-8);
    FOUR_C_EXPECT_NEAR(state_quantity_derivatives_solution_transv_isotrop_.curr_ddpdC,
        computed_state_quantity_derivatives_transv_isotrop.curr_ddpdC, 1.0e-6);
    FOUR_C_EXPECT_NEAR(state_quantity_derivatives_solution_isotrop_.curr_ddpdC,
        computed_state_quantity_derivatives_isotrop.curr_ddpdC, 1.0e-6);
    FOUR_C_EXPECT_NEAR(state_quantity_derivatives_solution_transv_isotrop_.curr_ddpdiFin,
        computed_state_quantity_derivatives_transv_isotrop.curr_ddpdiFin, 1.0e-6);
    FOUR_C_EXPECT_NEAR(state_quantity_derivatives_solution_isotrop_.curr_ddpdiFin,
        computed_state_quantity_derivatives_isotrop.curr_ddpdiFin, 1.0e-6);
    FOUR_C_EXPECT_NEAR(state_quantity_derivatives_solution_transv_isotrop_.curr_ddpdepsp,
        computed_state_quantity_derivatives_transv_isotrop.curr_ddpdepsp, 1.0e-6);
    FOUR_C_EXPECT_NEAR(state_quantity_derivatives_solution_isotrop_.curr_ddpdepsp,
        computed_state_quantity_derivatives_isotrop.curr_ddpdepsp, 1.0e-6);
    FOUR_C_EXPECT_NEAR(state_quantity_derivatives_solution_transv_isotrop_.curr_dlpdC,
        computed_state_quantity_derivatives_transv_isotrop.curr_dlpdC, 1.0e-6);
    FOUR_C_EXPECT_NEAR(state_quantity_derivatives_solution_isotrop_.curr_dlpdC,
        computed_state_quantity_derivatives_isotrop.curr_dlpdC, 1.0e-6);
    FOUR_C_EXPECT_NEAR(state_quantity_derivatives_solution_transv_isotrop_.curr_dlpdiFin,
        computed_state_quantity_derivatives_transv_isotrop.curr_dlpdiFin, 1.0e-6);
    FOUR_C_EXPECT_NEAR(state_quantity_derivatives_solution_isotrop_.curr_dlpdiFin,
        computed_state_quantity_derivatives_isotrop.curr_dlpdiFin, 1.0e-6);
    FOUR_C_EXPECT_NEAR(state_quantity_derivatives_solution_transv_isotrop_.curr_dlpdepsp,
        computed_state_quantity_derivatives_transv_isotrop.curr_dlpdepsp, 1.0e-6);
    FOUR_C_EXPECT_NEAR(state_quantity_derivatives_solution_isotrop_.curr_dlpdepsp,
        computed_state_quantity_derivatives_isotrop.curr_dlpdepsp, 1.0e-6);
  }

  TEST_F(InelasticDefgradFactorsTest, ThermoViscoplastStateQuantitiesAndDerivativesTest)
  {
    // This tests evaluate_state_quantities and evaluate_state_quantity_derivatives
    // if there are temperature contributions, i.e. the temperature is not the reference
    // temperature.

    //**************************************************
    const double current_temperature = 313.0;
    //**************************************************
    Core::LinAlg::Matrix<3, 3> CM{Core::LinAlg::Initialization::zero};
    CM(0, 0) = 2.0000000000000000;
    CM(0, 1) = 0.0000000000000000;
    CM(0, 2) = 0.0000000000000000;
    CM(1, 0) = 0.0000000000000000;
    CM(1, 1) = 1.0000000000000000;
    CM(1, 2) = 0.0000000000000000;
    CM(2, 0) = 0.0000000000000000;
    CM(2, 1) = 0.0000000000000000;
    CM(2, 2) = 1.0000000000000000;
    //**************************************************
    Core::LinAlg::Matrix<3, 3> iFinM{Core::LinAlg::Initialization::zero};
    iFinM(0, 0) = 1.2771823873225885;
    iFinM(0, 1) = 3.8315471619677655;
    iFinM(0, 2) = 1.6603371035193650;
    iFinM(1, 0) = 0.3831547161967765;
    iFinM(1, 1) = 1.9157735809838827;
    iFinM(1, 2) = 0.0000000000000000;
    iFinM(2, 0) = 0.0000000000000000;
    iFinM(2, 1) = 0.0000000000000000;
    iFinM(2, 2) = 1.0217459098580708;
    //**************************************************
    double plastic_strain = 0.0100000000000000;
    //**************************************************
    Core::LinAlg::Matrix<3, 3> CeM_ref{Core::LinAlg::Initialization::zero};
    CeM_ref(0, 0) = 3.4091972375178852;
    CeM_ref(0, 1) = 10.5212067856413203;
    CeM_ref(0, 2) = 4.2411066112662690;
    CeM_ref(1, 0) = 10.5212067856413203;
    CeM_ref(1, 1) = 33.0316957223622865;
    CeM_ref(1, 2) = 12.7233198337988060;
    CeM_ref(2, 0) = 4.2411066112662690;
    CeM_ref(2, 1) = 12.7233198337988060;
    CeM_ref(2, 2) = 6.5574032989578459;
    //**************************************************
    Core::LinAlg::Matrix<3, 3> M_theta_dev_ref_{Core::LinAlg::Initialization::zero};
    M_theta_dev_ref_(0, 0) = -6038.6293233280339336;
    M_theta_dev_ref_(0, 1) = 5816.2009659724717494;
    M_theta_dev_ref_(0, 2) = 2344.5151180664397543;
    M_theta_dev_ref_(1, 0) = 5816.2009659724717494;
    M_theta_dev_ref_(1, 1) = 10336.9070397830182628;
    M_theta_dev_ref_(1, 2) = 7033.5453541996248532;
    M_theta_dev_ref_(2, 0) = 2344.5151180664397543;
    M_theta_dev_ref_(2, 1) = 7033.5453541996248532;
    M_theta_dev_ref_(2, 2) = -4298.2777164549988811;
    //**************************************************
    Core::LinAlg::Matrix<3, 3> dM_theta_dev_dT_ref_{Core::LinAlg::Initialization::zero};
    dM_theta_dev_dT_ref_(0, 0) = 741.8887454374042818;
    dM_theta_dev_dT_ref_(0, 1) = -714.5618329623580394;
    dM_theta_dev_dT_ref_(0, 2) = -288.0404287910280345;
    dM_theta_dev_dT_ref_(1, 0) = -714.5618329623580394;
    dM_theta_dev_dT_ref_(1, 1) = -1269.9628648876227999;
    dM_theta_dev_dT_ref_(1, 2) = -864.1212863730841036;
    dM_theta_dev_dT_ref_(2, 0) = -288.0404287910280345;
    dM_theta_dev_dT_ref_(2, 1) = -864.1212863730841036;
    dM_theta_dev_dT_ref_(2, 2) = 528.0741194502180633;
    //**************************************************
    Core::LinAlg::Matrix<6, 6> dM_theta_dev_dC_Voigt_ref_{Core::LinAlg::Initialization::zero};
    dM_theta_dev_dC_Voigt_ref_(0, 0) = -2612.0303046296012326;
    dM_theta_dev_dC_Voigt_ref_(0, 1) = -622.1982428715837159;
    dM_theta_dev_dC_Voigt_ref_(0, 2) = -192.3704712255548657;
    dM_theta_dev_dC_Voigt_ref_(0, 3) = -1172.2575590411943267;
    dM_theta_dev_dC_Voigt_ref_(0, 4) = 0.0000000001800800;
    dM_theta_dev_dC_Voigt_ref_(0, 5) = -312.6020157432431006;
    dM_theta_dev_dC_Voigt_ref_(1, 0) = 4601.8623663998041593;
    dM_theta_dev_dC_Voigt_ref_(1, 1) = 1325.5527782973986177;
    dM_theta_dev_dC_Voigt_ref_(1, 2) = -192.3704712263679539;
    dM_theta_dev_dC_Voigt_ref_(1, 3) = 2615.0360932529874844;
    dM_theta_dev_dC_Voigt_ref_(1, 4) = -0.0000000007476046;
    dM_theta_dev_dC_Voigt_ref_(1, 5) = -312.6020157437633316;
    dM_theta_dev_dC_Voigt_ref_(2, 0) = -1989.8320617701956508;
    dM_theta_dev_dC_Voigt_ref_(2, 1) = -703.3545354258058069;
    dM_theta_dev_dC_Voigt_ref_(2, 2) = 384.7409424519319145;
    dM_theta_dev_dC_Voigt_ref_(2, 3) = -1442.7785342117936125;
    dM_theta_dev_dC_Voigt_ref_(2, 4) = 0.0000000005676366;
    dM_theta_dev_dC_Voigt_ref_(2, 5) = 625.2040314869955182;
    dM_theta_dev_dC_Voigt_ref_(3, 0) = 2705.2097516246067244;
    dM_theta_dev_dC_Voigt_ref_(3, 1) = 405.7814627492407453;
    dM_theta_dev_dC_Voigt_ref_(3, 2) = 0.0000000004220055;
    dM_theta_dev_dC_Voigt_ref_(3, 3) = 1082.0839006570604397;
    dM_theta_dev_dC_Voigt_ref_(3, 4) = -0.0000000010331860;
    dM_theta_dev_dC_Voigt_ref_(3, 5) = -0.0000000065338099;
    dM_theta_dev_dC_Voigt_ref_(4, 0) = 3516.7726770963417948;
    dM_theta_dev_dC_Voigt_ref_(4, 1) = -0.0000000009968062;
    dM_theta_dev_dC_Voigt_ref_(4, 2) = -0.0000000001327862;
    dM_theta_dev_dC_Voigt_ref_(4, 3) = 879.1931692749094509;
    dM_theta_dev_dC_Voigt_ref_(4, 4) = 541.0419503227003588;
    dM_theta_dev_dC_Voigt_ref_(4, 5) = 1082.0839006451242312;
    dM_theta_dev_dC_Voigt_ref_(5, 0) = 1172.2575590275155264;
    dM_theta_dev_dC_Voigt_ref_(5, 1) = 0.0000000009713403;
    dM_theta_dev_dC_Voigt_ref_(5, 2) = 0.0000000003201421;
    dM_theta_dev_dC_Voigt_ref_(5, 3) = 175.8386338541968144;
    dM_theta_dev_dC_Voigt_ref_(5, 4) = 108.2083900641773653;
    dM_theta_dev_dC_Voigt_ref_(5, 5) = 360.6946335464308504;
    //**************************************************
    Core::LinAlg::Matrix<6, 9> dM_theta_dev_diFp_Voigt_ref_{Core::LinAlg::Initialization::zero};
    dM_theta_dev_diFp_Voigt_ref_(0, 0) = 1882.7623322118306533;
    dM_theta_dev_diFp_Voigt_ref_(0, 1) = -706.0358746152960521;
    dM_theta_dev_diFp_Voigt_ref_(0, 2) = -376.5524664578551892;
    dM_theta_dev_diFp_Voigt_ref_(0, 3) = -2824.1434984737352352;
    dM_theta_dev_diFp_Voigt_ref_(0, 4) = 0.0000000007004775;
    dM_theta_dev_diFp_Voigt_ref_(0, 5) = -1223.7955159961129539;
    dM_theta_dev_diFp_Voigt_ref_(0, 6) = 282.4143498440244002;
    dM_theta_dev_diFp_Voigt_ref_(0, 7) = 0.0000000014206307;
    dM_theta_dev_diFp_Voigt_ref_(0, 8) = -0.0000000027066562;
    dM_theta_dev_diFp_Voigt_ref_(1, 0) = -941.3811661213476327;
    dM_theta_dev_diFp_Voigt_ref_(1, 1) = 1412.0717492313815455;
    dM_theta_dev_diFp_Voigt_ref_(1, 2) = -376.5524664604199643;
    dM_theta_dev_diFp_Voigt_ref_(1, 3) = 5648.2869969418461551;
    dM_theta_dev_diFp_Voigt_ref_(1, 4) = -0.0000000021225941;
    dM_theta_dev_diFp_Voigt_ref_(1, 5) = -1223.7955159952107351;
    dM_theta_dev_diFp_Voigt_ref_(1, 6) = -141.2071749401075067;
    dM_theta_dev_diFp_Voigt_ref_(1, 7) = -0.0000000013951649;
    dM_theta_dev_diFp_Voigt_ref_(1, 8) = -0.0000000009167707;
    dM_theta_dev_diFp_Voigt_ref_(2, 0) = -941.3811660904320888;
    dM_theta_dev_diFp_Voigt_ref_(2, 1) = -706.0358746160563896;
    dM_theta_dev_diFp_Voigt_ref_(2, 2) = 753.1049329183006193;
    dM_theta_dev_diFp_Voigt_ref_(2, 3) = -2824.1434984681200149;
    dM_theta_dev_diFp_Voigt_ref_(2, 4) = 0.0000000014221166;
    dM_theta_dev_diFp_Voigt_ref_(2, 5) = 2447.5910319913241437;
    dM_theta_dev_diFp_Voigt_ref_(2, 6) = -141.2071749040042050;
    dM_theta_dev_diFp_Voigt_ref_(2, 7) = -0.0000000000145519;
    dM_theta_dev_diFp_Voigt_ref_(2, 8) = 0.0000000035215635;
    dM_theta_dev_diFp_Voigt_ref_(3, 0) = 4236.2152473528403789;
    dM_theta_dev_diFp_Voigt_ref_(3, 1) = 211.8107623872019758;
    dM_theta_dev_diFp_Voigt_ref_(3, 2) = 0.0000000032305252;
    dM_theta_dev_diFp_Voigt_ref_(3, 3) = 1412.0717492136172950;
    dM_theta_dev_diFp_Voigt_ref_(3, 4) = -0.0000000032487151;
    dM_theta_dev_diFp_Voigt_ref_(3, 5) = -0.0000000162544893;
    dM_theta_dev_diFp_Voigt_ref_(3, 6) = 1059.0538118852709886;
    dM_theta_dev_diFp_Voigt_ref_(3, 7) = 0.0000000021245796;
    dM_theta_dev_diFp_Voigt_ref_(3, 8) = -0.0000000133986759;
    dM_theta_dev_diFp_Voigt_ref_(4, 0) = -0.0000000818399712;
    dM_theta_dev_diFp_Voigt_ref_(4, 1) = 0.0000000022919266;
    dM_theta_dev_diFp_Voigt_ref_(4, 2) = 0.0000000003856258;
    dM_theta_dev_diFp_Voigt_ref_(4, 3) = 1835.6932739952317206;
    dM_theta_dev_diFp_Voigt_ref_(4, 4) = 1059.0538119160482893;
    dM_theta_dev_diFp_Voigt_ref_(4, 5) = 4236.2152476652117912;
    dM_theta_dev_diFp_Voigt_ref_(4, 6) = -0.0000000073341653;
    dM_theta_dev_diFp_Voigt_ref_(4, 7) = 564.8286996881288360;
    dM_theta_dev_diFp_Voigt_ref_(4, 8) = -0.0000000002328306;
    dM_theta_dev_diFp_Voigt_ref_(5, 0) = 1835.6932738409377635;
    dM_theta_dev_diFp_Voigt_ref_(5, 1) = 0.0000000021545929;
    dM_theta_dev_diFp_Voigt_ref_(5, 2) = 0.0000000008731149;
    dM_theta_dev_diFp_Voigt_ref_(5, 3) = -0.0000000091458787;
    dM_theta_dev_diFp_Voigt_ref_(5, 4) = 211.8107623823962058;
    dM_theta_dev_diFp_Voigt_ref_(5, 5) = 1412.0717492167095770;
    dM_theta_dev_diFp_Voigt_ref_(5, 6) = -0.0000000102991180;
    dM_theta_dev_diFp_Voigt_ref_(5, 7) = 0.0000000004365575;
    dM_theta_dev_diFp_Voigt_ref_(5, 8) = 564.8286996873503085;
    //**************************************************
    double equiv_stress_ref_ = 22562.6890921091871860;
    //**************************************************
    Core::LinAlg::Matrix<3, 3> Np_ref_{Core::LinAlg::Initialization::zero};
    Np_ref_(0, 0) = -0.4014567566841964;
    Np_ref_(0, 1) = 0.3866693997928572;
    Np_ref_(0, 2) = 0.1558667348002227;
    Np_ref_(1, 0) = 0.3866693997928572;
    Np_ref_(1, 1) = 0.6872124371512610;
    Np_ref_(1, 2) = 0.4676002044006883;
    Np_ref_(2, 0) = 0.1558667348002227;
    Np_ref_(2, 1) = 0.4676002044006883;
    Np_ref_(2, 2) = -0.2857556804670655;
    //**************************************************
    double plastic_strain_rate_ref_ = 0.4830914865309823;
    //**************************************************
    Core::LinAlg::Matrix<3, 3> lp_ref_{Core::LinAlg::Initialization::zero};
    lp_ref_(0, 0) = -0.1939403413644753;
    lp_ref_(0, 1) = 0.1867966951419741;
    lp_ref_(0, 2) = 0.0752978926153700;
    lp_ref_(1, 0) = 0.1867966951419741;
    lp_ref_(1, 1) = 0.3319864778259820;
    lp_ref_(1, 2) = 0.2258936778461197;
    lp_ref_(2, 0) = 0.0752978926153700;
    lp_ref_(2, 1) = 0.2258936778461197;
    lp_ref_(2, 2) = -0.1380461364615071;
    //**************************************************
    Core::LinAlg::Matrix<9, 1> dlp_dT_Voigt_ref_{Core::LinAlg::Initialization::zero};
    dlp_dT_Voigt_ref_(0) = 0.7562742988211517;
    dlp_dT_Voigt_ref_(1) = -1.2945880107744234;
    dlp_dT_Voigt_ref_(2) = 0.5383137119532734;
    dlp_dT_Voigt_ref_(3) = -0.7284175053353915;
    dlp_dT_Voigt_ref_(4) = -0.8808769831963305;
    dlp_dT_Voigt_ref_(5) = -0.2936256610654312;
    dlp_dT_Voigt_ref_(6) = -0.7284175053353914;
    dlp_dT_Voigt_ref_(7) = -0.8808769831963305;
    dlp_dT_Voigt_ref_(8) = -0.2936256610654312;


    FOUR_C_ASSERT_ALWAYS(
        std::abs(iFinM.determinant() - 1) < 1.0e-12, "iFinM is not volume preserving");

    Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonParams local_newton_params{
        .res_tol = 1.0e-8,
        .incr_tol = 1.0e-8,
        .conv_check = FourC::Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::
            LocalNewtonConvCheck::residual_and_increment_ratio,
        .diver_cont = FourC::Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::
            LocalNewtonDiverCont::stop,
        .max_iter = 50,
        .max_exceedance_fact_res_tol = 1.0e1,
        .max_exceedance_fact_incr_tol = 1.0e1,
    };


    std::shared_ptr<Mat::PAR::InelasticDefgradTransvIsotropElastViscoplast> material_params;
    auto material =
        set_up_viscoplastic_material({.local_newton_params = local_newton_params}).material;

    // parameter list for InelasticDefGradTransvIsotropElastViscoplast
    Teuchos::ParameterList param_list_thermo_vplast{};
    param_list_thermo_vplast.set<double>("temperature", current_temperature);


    double total_time = 4e-5;
    double time_step_size = 4e-5;
    Mat::EvaluationContext<3> context{.total_time = &total_time,
        .time_step_size = &time_step_size,
        .xi = {},
        .ref_coords = nullptr};
    // call pre_evaluate
    material->pre_evaluate(param_list_thermo_vplast, context, 0, 0);


    Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType err_status{
        FourC::Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType::no_errors};

    // compute StateQuantities objects
    Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::StateQuantities
        computed_state_quantities = material->evaluate_state_quantities(CM, current_temperature,
            iFinM, plastic_strain, err_status, 1.0,
            Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::StateQuantityEvalType::
                full_eval);

    EXPECT_NEAR(
        computed_state_quantities.curr_equiv_plastic_strain_rate, plastic_strain_rate_ref_, 1e-8);
    EXPECT_NEAR(computed_state_quantities.curr_equiv_stress, equiv_stress_ref_, 1e-8);
    FOUR_C_EXPECT_NEAR(computed_state_quantities.curr_CeM, CeM_ref, 1e-8);
    FOUR_C_EXPECT_NEAR(computed_state_quantities.curr_lpM, lp_ref_, 1e-8);
    FOUR_C_EXPECT_NEAR(computed_state_quantities.curr_Mtheta_dev_sym_M, M_theta_dev_ref_, 1e-8);
    FOUR_C_EXPECT_NEAR(computed_state_quantities.curr_NpM, Np_ref_, 1e-8);


    // compute StateQuantityDerivatives objects
    Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::StateQuantityDerivatives
        computed_state_quantity_derivatives = material->evaluate_state_quantity_derivatives(CM,
            current_temperature, iFinM, plastic_strain, err_status, 1.0,
            Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::StateQuantityDerivEvalType::
                full_eval,
            true);

    // postprocessing for comparison
    Core::LinAlg::Matrix<3, 3> curr_dMe_dev_sym_dT{Core::LinAlg::Initialization::zero};
    Core::LinAlg::Voigt::Stresses::vector_to_matrix(
        computed_state_quantity_derivatives.curr_dMtheta_dev_sym_dT, curr_dMe_dev_sym_dT);

    // assert equality
    FOUR_C_EXPECT_NEAR(curr_dMe_dev_sym_dT, dM_theta_dev_dT_ref_, 1e-6);
    FOUR_C_EXPECT_NEAR(computed_state_quantity_derivatives.curr_dMtheta_dev_sym_dC,
        dM_theta_dev_dC_Voigt_ref_, 1e-6);
    FOUR_C_EXPECT_NEAR(computed_state_quantity_derivatives.curr_dMtheta_dev_sym_diFin,
        dM_theta_dev_diFp_Voigt_ref_, 1e-6);
    FOUR_C_EXPECT_NEAR(computed_state_quantity_derivatives.curr_dlpdT, dlp_dT_Voigt_ref_, 1e-6);
  }

  TEST_F(InelasticDefgradFactorsTest, ThermoViscoplastAdditionalCmatLinearizationMatchesFD)
  {
    auto comparison = set_up_thermo_viscoplast_linearization_comparison();
    FOUR_C_EXPECT_NEAR(comparison.iFin_analytic, comparison.iFin_fd, 1.0e-10);

    Core::LinAlg::Matrix<6, 6> cmatadd_analytic(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<6, 6> cmatadd_fd(Core::LinAlg::Initialization::zero);
    comparison.analytic_material.material->evaluate_additional_cmat(&comparison.FM,
        Core::LinAlg::identity_matrix<3>(), comparison.iFin_analytic, comparison.iCV, dSdiFin_,
        cmatadd_analytic);
    comparison.fd_material.material->evaluate_additional_cmat(&comparison.FM,
        Core::LinAlg::identity_matrix<3>(), comparison.iFin_fd, comparison.iCV, dSdiFin_,
        cmatadd_fd);

    ASSERT_GT(cmatadd_analytic.norm2(), 0.0);
    expect_near_relative(cmatadd_analytic, cmatadd_fd, 1.0e-6, 1.0e-9);
  }

  TEST_F(InelasticDefgradFactorsTest, ThermoViscoplastODStiffMatLinearizationMatchesFD)
  {
    auto comparison = set_up_thermo_viscoplast_linearization_comparison();
    FOUR_C_EXPECT_NEAR(comparison.iFin_analytic, comparison.iFin_fd, 1.0e-10);

    Core::LinAlg::Matrix<6, 1> dstressdT_analytic(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<6, 1> dstressdT_fd(Core::LinAlg::Initialization::zero);
    comparison.analytic_material.material->evaluate_od_stiff_mat(&comparison.FM,
        Core::LinAlg::identity_matrix<3>(), comparison.iFin_analytic, dSdiFin_, dstressdT_analytic);
    comparison.fd_material.material->evaluate_od_stiff_mat(&comparison.FM,
        Core::LinAlg::identity_matrix<3>(), comparison.iFin_fd, dSdiFin_, dstressdT_fd);

    ASSERT_GT(dstressdT_analytic.norm2(), 0.0);
    expect_near_relative(dstressdT_analytic, dstressdT_fd, 1.0e-6, 1.0e-9);
  }

  TEST_F(InelasticDefgradFactorsTest, ThermoViscoplastTaylorQuinneyHeatSourceLinearizationMatchesFD)
  {
    auto comparison = set_up_thermo_viscoplast_linearization_comparison();
    FOUR_C_EXPECT_NEAR(comparison.iFin_analytic, comparison.iFin_fd, 1.0e-10);

    const Mat::HeatSource tq_heat_analytic =
        comparison.analytic_material.material->evaluate_taylor_quinney_heat_source(
            comparison.context, 0, 0, &comparison.FM, Core::LinAlg::identity_matrix<3>(),
            comparison.temperature);
    const Mat::HeatSource tq_heat_fd =
        comparison.fd_material.material->evaluate_taylor_quinney_heat_source(comparison.context, 0,
            0, &comparison.FM, Core::LinAlg::identity_matrix<3>(), comparison.temperature);

    ASSERT_GT(std::abs(tq_heat_analytic.value), 0.0);
    ASSERT_GT(tq_heat_analytic.derivative_wrt_cauchy_green.norm2(), 0.0);
    ASSERT_GT(std::abs(tq_heat_analytic.derivative_wrt_temperature), 0.0);

    ASSERT_DOUBLE_EQ(tq_heat_analytic.value, tq_heat_fd.value);
    expect_near_relative(tq_heat_analytic.derivative_wrt_cauchy_green,
        tq_heat_fd.derivative_wrt_cauchy_green, 1.0e-7, 1.0e-16);
    expect_near_relative(tq_heat_analytic.derivative_wrt_temperature,
        tq_heat_fd.derivative_wrt_temperature, 1.0e-8, 1.0e-16);
  }

  TEST_F(InelasticDefgradFactorsTest, ThermoViscoplastPublicEvaluationsCanBeCalledInAnyOrder)
  {
    // The viscoplastic factor has two public evaluation paths: one from the solid field (through
    // pre_evaluate, evaluate_inverse_inelastic_def_grad, evaluate_additional_cmat and (for
    // monolithic solvers) evaluate_od_stiff_mat) and one from the thermo field (through
    // evaluate_mechanical_dissipation).

    // This test tests that both evaluation paths are independent of each other, i.e. they can be
    // called in arbitrary order.

    // Since internally, both evaluation paths are coupled via Gauss point caches, we must also
    // ensure that either evaluation path uses wrong caches. This is done by letting the second
    // public evaluation request results at a different gauss point, temperature or deformation
    // gradient respectively.

    // requested_* values are the values we actually test on. stale_* values are used to show that
    // those values are not leaking into the evaluations with the requested_* values.

    const double requested_temperature = 313.0;
    const double stale_temperature = 312.0;

    Core::LinAlg::Matrix<3, 3> requested_FM(Core::LinAlg::Initialization::zero);
    requested_FM(0, 0) = 1.55;
    requested_FM(0, 1) = 0.15;
    requested_FM(1, 1) = 1.0;
    requested_FM(2, 2) = 1.0;

    Core::LinAlg::Matrix<3, 3> stale_FM(Core::LinAlg::Initialization::zero);
    stale_FM(0, 0) = 1.50;
    stale_FM(0, 1) = 0.15;
    stale_FM(1, 1) = 1.3;
    stale_FM(2, 2) = 0.8;

    const int requested_gp = 0;
    const int stale_gp = 1;
    const int numgp = 2;

    double total_time = 1.0e-6;
    double time_step_size = 1.0e-6;
    const Mat::EvaluationContext<3> context{.total_time = &total_time,
        .time_step_size = &time_step_size,
        .xi = {},
        .ref_coords = nullptr};

    // auxiliaries
    const Core::LinAlg::Matrix<3, 3> id3x3 = Core::LinAlg::identity_matrix<3>();
    using ViscoPlastMaterial = std::shared_ptr<Mat::InelasticDefgradTransvIsotropElastViscoplast>;

    const ReformulatedJohnsonCookParameters viscoplastic_law_params{.strain_rate_prefac = 1.0,
        .strain_rate_exp_fac = 0.014,
        .init_yield_strength = 792.0,
        .isotrop_harden_prefac = 510.0,
        .isotrop_harden_exp = 0.26,
        .ref_temperature = 293.0,
        .melt_temperature = 1793.0,
        .temperature_sens = 1.03};

    // create a new clean material instance
    auto make_material = [&viscoplastic_law_params]()
    {
      return set_up_viscoplastic_material({.numgp = numgp,
                                              .taylor_quinney_coefficient = 0.85,
                                              .viscoplastic_law_params = viscoplastic_law_params})
          .material;
    };

    struct SolidEvaluationResults
    {
      Core::LinAlg::Matrix<3, 3> iFin;
      Core::LinAlg::Matrix<6, 6> cmatadd;
      Core::LinAlg::Matrix<6, 1> dstressdT;
      static void assert_equal(
          const SolidEvaluationResults& actual, const SolidEvaluationResults& expected)
      {
        FOUR_C_EXPECT_NEAR(actual.iFin, expected.iFin, 1.0e-16);
        FOUR_C_EXPECT_NEAR(actual.cmatadd, expected.cmatadd, 1.0e-16);
        FOUR_C_EXPECT_NEAR(actual.dstressdT, expected.dstressdT, 1.0e-16);
      }
    };

    struct ThermoEvaluationResults
    {
      Mat::HeatSource heat_source;
      static void assert_equal(
          const ThermoEvaluationResults& actual, const ThermoEvaluationResults& expected)
      {
        ASSERT_DOUBLE_EQ(actual.heat_source.value, expected.heat_source.value);
        FOUR_C_EXPECT_NEAR(actual.heat_source.derivative_wrt_cauchy_green,
            expected.heat_source.derivative_wrt_cauchy_green, 1.0e-16);
        ASSERT_DOUBLE_EQ(actual.heat_source.derivative_wrt_temperature,
            expected.heat_source.derivative_wrt_temperature);
      }
    };

    // helper to store a full evaluation result
    struct EvaluationResults
    {
      SolidEvaluationResults solid_results;
      ThermoEvaluationResults thermo_results;

      // assert that two EvaluationResults are near each other (used for comparing results from
      // different evaluation orders)
      static void assert_equal(const EvaluationResults& actual, const EvaluationResults& expected)
      {
        SolidEvaluationResults::assert_equal(actual.solid_results, expected.solid_results);
        ThermoEvaluationResults::assert_equal(actual.thermo_results, expected.thermo_results);
      };
    };

    // perform a full solid evaluation of the material
    const auto solid_evaluation = [&context, this](const ViscoPlastMaterial& material, const int gp,
                                      const Core::LinAlg::Matrix<3, 3>& defgrad,
                                      const double temperature) -> SolidEvaluationResults
    {
      // pre-evaluate (sets the gauss point index and temperature)
      Teuchos::ParameterList param_list;
      param_list.set<double>("temperature", temperature);
      material->pre_evaluate(param_list, context, gp, 0);

      SolidEvaluationResults results;

      try
      {
        // evaluate and return the inverse inelastic defgrad (relies on pre_evaluate being called
        // first)
        material->evaluate_inverse_inelastic_def_grad(
            &defgrad, Core::LinAlg::identity_matrix<3>(), results.iFin);

        // evaluate the additional cmat contribution (relies on
        // evaluate_inverse_inelastic_defgrad being called first).
        // We simply use dSiFin_ from the SetUp(), it does not matter here
        Core::LinAlg::Matrix<3, 3> CM(Core::LinAlg::Initialization::zero);
        Core::LinAlg::Matrix<3, 3> iCM(Core::LinAlg::Initialization::zero);
        Core::LinAlg::Matrix<6, 1> iCV(Core::LinAlg::Initialization::zero);
        CM.multiply_tn(1.0, defgrad, defgrad, 0.0);
        iCM.invert(CM);
        Core::LinAlg::Voigt::Stresses::matrix_to_vector(iCM, iCV);
        material->evaluate_additional_cmat(&defgrad, Core::LinAlg::identity_matrix<3>(),
            results.iFin, iCV, dSdiFin_, results.cmatadd);
        EXPECT_GT(results.cmatadd.norm2(), 0.0);

        // evaluate the off-diagonal stiffness matrix contribution (relies on
        // evaluate_inverse_inelastic_defgrad being called first)
        material->evaluate_od_stiff_mat(&defgrad, Core::LinAlg::identity_matrix<3>(), results.iFin,
            dSdiFin_, results.dstressdT);
        EXPECT_GT(results.dstressdT.norm2(), 0.0);
      }
      catch (const std::exception& exception)
      {
        ADD_FAILURE() << "Solid evaluation failed:\n" << exception.what();
      }

      return results;
    };


    //************************* Actual test starts here *************************

    // container to hold the reference evaluation result obtained from solid first
    EvaluationResults reference;

    {
      SCOPED_TRACE("Solid evaluation first, thermo evaluation second.");

      {
        SCOPED_TRACE(
            "Thermo evaluation requests same temperature and deformation gradient as the solid "
            "evaluation.");
        auto material = make_material();
        // 1.: solid evaluation
        reference.solid_results =
            solid_evaluation(material, requested_gp, requested_FM, requested_temperature);
        // 2.: thermo evaluation
        reference.thermo_results.heat_source = material->evaluate_taylor_quinney_heat_source(
            context, requested_gp, 0, &requested_FM, id3x3, requested_temperature);

        // Since we are using this result as reference,assert that all values are non-zero to ensure
        // full evaluation paths (no early returns due to no plastic strain)
        ASSERT_GT(reference.solid_results.cmatadd.norm2(), 0.0);
        ASSERT_GT(reference.solid_results.dstressdT.norm2(), 0.0);
        ASSERT_GT(std::abs(reference.thermo_results.heat_source.value), 0.0);
        ASSERT_GT(reference.thermo_results.heat_source.derivative_wrt_cauchy_green.norm2(), 0.0);
        ASSERT_GT(std::abs(reference.thermo_results.heat_source.derivative_wrt_temperature), 0.0);
      }
      {
        SCOPED_TRACE(
            "Thermo evaluation requests a different temperature than the previous solid "
            "evaluation.");
        auto material = make_material();
        ThermoEvaluationResults current;
        // 1.: solid evaluation with stale temperature
        solid_evaluation(material, requested_gp, requested_FM, stale_temperature);
        // 2.: thermo evaluation with the requested temperature
        current.heat_source = material->evaluate_taylor_quinney_heat_source(
            context, requested_gp, 0, &requested_FM, id3x3, requested_temperature);

        ThermoEvaluationResults::assert_equal(current, reference.thermo_results);
      }
      {
        SCOPED_TRACE(
            "Thermo evaluation requests a different deformation gradient than the previous solid "
            "evaluation.");
        auto material = make_material();
        ThermoEvaluationResults current;
        // 1.: solid evaluation with stale deformation gradient
        solid_evaluation(material, requested_gp, stale_FM, requested_temperature);
        // 2.: thermo evaluation with the requested deformation gradient
        current.heat_source = material->evaluate_taylor_quinney_heat_source(
            context, requested_gp, 0, &requested_FM, id3x3, requested_temperature);
        ThermoEvaluationResults::assert_equal(current, reference.thermo_results);
      }
      {
        SCOPED_TRACE(
            "Thermo evaluation requests a different gauss point than the previous solid "
            "evaluation.");
        auto material = make_material();
        EvaluationResults current;
        // 1.: solid evaluation at two different gauss points, the most previous one with stale
        // values
        current.solid_results =
            solid_evaluation(material, requested_gp, requested_FM, requested_temperature);
        solid_evaluation(material, stale_gp, stale_FM,
            stale_temperature);  // evaluate the next gauss point with different values to test that
                                 // gauss point caches are not leaking into each other
        // 2.: thermo evaluation at the requested gauss point (should use the cache filled by the
        // first solid evaluation)
        current.thermo_results.heat_source = material->evaluate_taylor_quinney_heat_source(
            context, requested_gp, 0, &requested_FM, id3x3, requested_temperature);
        EvaluationResults::assert_equal(current, reference);
      }
    }

    {
      SCOPED_TRACE("Thermo evaluation first, solid evaluation second.");

      {
        SCOPED_TRACE(
            "Thermo evaluation requests same temperature and deformation gradient as the solid "
            "evaluation.");
        auto material = make_material();
        EvaluationResults current;
        // 1.: thermo evaluation
        current.thermo_results.heat_source = material->evaluate_taylor_quinney_heat_source(
            context, requested_gp, 0, &requested_FM, id3x3, requested_temperature);
        // 2.: solid evaluation
        current.solid_results =
            solid_evaluation(material, requested_gp, requested_FM, requested_temperature);

        EvaluationResults::assert_equal(current, reference);
      }
      {
        SCOPED_TRACE(
            "Solid evaluation requests a different temperature than the previous thermo "
            "evaluation.");
        auto material = make_material();
        // 1.: thermo evaluation with stale temperature
        (void)material->evaluate_taylor_quinney_heat_source(
            context, requested_gp, 0, &requested_FM, id3x3, stale_temperature);
        // 2.: solid evaluation with the requested temperature
        const auto current =
            solid_evaluation(material, requested_gp, requested_FM, requested_temperature);

        SolidEvaluationResults::assert_equal(current, reference.solid_results);
      }
      {
        SCOPED_TRACE(
            "Solid evaluation requests a different deformation gradient than the previous thermo "
            "evaluation.");
        auto material = make_material();
        // 1.: thermo evaluation with stale deformation gradient
        (void)material->evaluate_taylor_quinney_heat_source(
            context, requested_gp, 0, &stale_FM, id3x3, requested_temperature);
        // 2.: solid evaluation with the requested deformation gradient
        const auto current =
            solid_evaluation(material, requested_gp, requested_FM, requested_temperature);
        SolidEvaluationResults::assert_equal(current, reference.solid_results);
      }
      {
        SCOPED_TRACE(
            "Solid evaluation requests a different gauss point than the previous thermo "
            "evaluation.");
        auto material = make_material();
        EvaluationResults current;
        // 1.: thermo evaluation at two different gauss points, the most previous one with stale
        // values
        current.thermo_results.heat_source = material->evaluate_taylor_quinney_heat_source(
            context, requested_gp, 0, &requested_FM, id3x3, requested_temperature);
        (void)material->evaluate_taylor_quinney_heat_source(context, stale_gp, 0, &stale_FM, id3x3,
            stale_temperature);  // evaluate the next gauss point with different values to test that
                                 // gauss point caches are not leaking into each other
        // 2.: solid evaluation at the requested gauss point (should use the cache filled by the
        // first thermo evaluation)
        current.solid_results =
            solid_evaluation(material, requested_gp, requested_FM, requested_temperature);
        EvaluationResults::assert_equal(current, reference);
      }
    }
  }
}  // namespace

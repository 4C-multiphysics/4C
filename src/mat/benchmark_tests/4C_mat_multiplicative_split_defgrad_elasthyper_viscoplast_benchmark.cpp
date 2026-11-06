// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_global_data.hpp"
#include "4C_io_input_parameter_container.templates.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_linalg_tensor.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_linalg_utils_densematrix_funct.hpp"
#include "4C_mat_inelastic_defgrad_factors_service.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_mat_multiplicative_split_defgrad_elasthyper.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_solid_ele_fibers.hpp"
#include "4C_ssi_input.hpp"
#include "4C_structure_new_enum_lists.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_singleton_owner.hpp"

#include <benchmark/benchmark.h>
#include <Teuchos_ParameterList.hpp>

#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace
{
  namespace ViscoplastUtils = Mat::InelasticDefgradTransvIsotropElastViscoplastUtils;
  namespace AEI = ViscoplastUtils::AdaptiveEstimateInterpolation;

  /// state of the Gauss point
  enum class PlasticState
  {
    /// stress below the yield stress, no inelastic deformation
    elastic,
    /// viscoplastic flow
    plastic
  };

  /// coupling of the material to the temperature
  enum class TemperatureCoupling
  {
    /// no temperature in the parameter list and no thermal expansion (like solid-only simulations)
    off,
    /// temperature equal to the reference temperature and thermal expansion, i.e., the same
    /// mechanical state as without coupling, but including the cost of the coupling
    on
  };

  /// evaluated material functions
  enum class EvaluationType
  {
    /// stress and stiffness tensor
    stress_stiffness,
    /// stress and stiffness tensor followed by the derivative of the stress wrt. the
    /// temperature at the same state, (like in a monolithic TSI iteration)
    stress_stiffness_temperature_derivative
  };

  constexpr int matid_coup_neo_hooke = 1;
  constexpr int matid_fiber_reader = 2;
  constexpr int matid_structural_tensor_strategy = 3;
  constexpr int matid_viscoplastic_law = 4;
  constexpr int matid_viscoplastic_factor = 5;
  constexpr int matid_multiplicative_split = 6;

  constexpr double ref_temperature = 293.0;
  constexpr double time_step_size = 1.0e-6;

  constexpr int gp = 0;
  constexpr int ele_gid = 0;

  /// set up all required materials in the global problem instance.
  /// The parameters are chosen to match the respective benchmark test case.
  /// AEI is always enabled.
  void set_up_viscoplastic_material(const ViscoplastUtils::MatBehavior mat_behavior,
      const TemperatureCoupling temperature_coupling)
  {
    auto& problem = *Global::Problem::instance();

    Core::IO::InputParameterContainer coup_neo_hooke_data;
    coup_neo_hooke_data.add("YOUNG", 2.0e5);
    coup_neo_hooke_data.add("NUE", 0.29);
    problem.materials()->insert(matid_coup_neo_hooke,
        Mat::make_parameter(matid_coup_neo_hooke, Core::Materials::MaterialType::mes_coupneohooke,
            coup_neo_hooke_data));

    Core::IO::InputParameterContainer fiber_reader_data;
    fiber_reader_data.add("ALPHA", 1.0);
    fiber_reader_data.add("BETA", 1.0);
    fiber_reader_data.add("GAMMA", 1.0);
    fiber_reader_data.add("ANGLE", 0.0);
    fiber_reader_data.add("STR_TENS_ID", matid_structural_tensor_strategy);
    fiber_reader_data.add<std::string>("STRATEGY", "Standard");
    fiber_reader_data.add<std::string>("DISTR", "none");
    fiber_reader_data.add("C1", 1.0);
    fiber_reader_data.add("C2", 0.0);
    fiber_reader_data.add("C3", 0.0);
    fiber_reader_data.add("C4", 1.0e16);
    fiber_reader_data.add("FIBER", 1);
    fiber_reader_data.add("INIT", 1);
    problem.materials()->insert(matid_structural_tensor_strategy,
        Mat::make_parameter(matid_structural_tensor_strategy,
            Core::Materials::MaterialType::mes_structuraltensorstratgy, fiber_reader_data));
    problem.materials()->insert(matid_fiber_reader,
        Mat::make_parameter(matid_fiber_reader,
            Core::Materials::MaterialType::mes_couptransverselyisotropic, fiber_reader_data));

    Core::IO::InputParameterContainer viscoplastic_law_data;
    viscoplastic_law_data.add("STRAIN_RATE_PREFAC", 1.0);
    viscoplastic_law_data.add("STRAIN_RATE_EXP_FAC", 0.014);
    viscoplastic_law_data.add("INIT_YIELD_STRENGTH", 792.0);
    viscoplastic_law_data.add("ISOTROP_HARDEN_PREFAC", 510.0);
    viscoplastic_law_data.add("ISOTROP_HARDEN_EXP", 0.26);
    viscoplastic_law_data.add("REF_TEMPERATURE", ref_temperature);
    viscoplastic_law_data.add("MELT_TEMPERATURE", 1793.0);
    viscoplastic_law_data.add("TEMPERATURE_SENS", 1.03);
    problem.materials()->insert(matid_viscoplastic_law,
        Mat::make_parameter(matid_viscoplastic_law,
            Core::Materials::MaterialType::mvl_reformulated_Johnson_Cook, viscoplastic_law_data));

    Core::IO::InputParameterContainer viscoplastic_factor_data;
    viscoplastic_factor_data.add("FIBER_READER_ID", matid_fiber_reader);
    viscoplastic_factor_data.add("TAYLOR_QUINNEY_COEFFICIENT", 0.85);
    viscoplastic_factor_data.add("LINEARIZATION", ViscoplastUtils::LinearizationType::analytic);
    viscoplastic_factor_data.add(
        "MATRIX_EXP_CALC_METHOD", Core::LinAlg::MatrixExpCalcMethod::automatic);
    viscoplastic_factor_data.add(
        "MATRIX_EXP_DERIV_CALC_METHOD", Core::LinAlg::GenMatrixExpFirstDerivCalcMethod::automatic);
    viscoplastic_factor_data.add(
        "MATRIX_LOG_CALC_METHOD", Core::LinAlg::MatrixLogCalcMethod::inv_scal_square);
    viscoplastic_factor_data.add("MATRIX_LOG_DERIV_CALC_METHOD",
        Core::LinAlg::GenMatrixLogFirstDerivCalcMethod::pade_part_fract);
    viscoplastic_factor_data.add("MAT_BEHAVIOR", mat_behavior);
    viscoplastic_factor_data.add(
        "TIME_INTEGRATION_HIST_VARS", ViscoplastUtils::TimIntType::logarithmic);
    viscoplastic_factor_data.add("VISCOPLAST_LAW_ID", matid_viscoplastic_law);
    viscoplastic_factor_data.add<std::optional<double>>("YIELD_COND_A", 1.0);
    viscoplastic_factor_data.add<std::optional<double>>("YIELD_COND_B", 2.0);
    viscoplastic_factor_data.add<std::optional<double>>("YIELD_COND_F", 2.5);
    viscoplastic_factor_data.group("LOCAL_SUBSTEPPING").add("USE_SUBSTEPPING", false);
    viscoplastic_factor_data.group("LOCAL_SUBSTEPPING").add("MAX_SUBSTEPPING_HALVE_NUM", 0);
    viscoplastic_factor_data.add("LOCAL_NEWTON",
        ViscoplastUtils::LocalNewtonParams{
            .res_tol = 1.0e-14,
            .incr_tol = 1.0e-14,
            .conv_check = ViscoplastUtils::LocalNewtonConvCheck::residual_and_increment_ratio,
            .diver_cont = ViscoplastUtils::LocalNewtonDiverCont::stop_and_request_step_reduction,
            .max_iter = 100,
            .max_exceedance_fact_res_tol = 1.0e1,
            .max_exceedance_fact_incr_tol = 1.0e1,
        });
    viscoplastic_factor_data.add("ERROR_REGISTRATION_SETTINGS",
        ViscoplastUtils::ErrorRegistrationSettings{.register_plastic_strain_incr_overflow = true,
            .max_plastic_strain_incr = std::exp(30.0),
            .register_plastic_strain_deriv_incr_overflow = false,
            .max_plastic_strain_deriv_incr = std::exp(30.0)});
    viscoplastic_factor_data.add("ADAPTIVE_ESTIMATE_INTERPOLATION",
        AEI::AEIParams{
            .use_adaptive_estimate_interpolation = true,
            .elastic_predictor_zero_component_threshold = 1.0e-13,
            .plastic_predictor_construction =
                {
                    .elastic_stretch_eigenval_type =
                        AEI::PrelimPlasticPredictor::ElasticStretchEigenvalType::scale_unit,
                    .elastic_stretch_eigenvect_type = AEI::PrelimPlasticPredictor::
                        ElasticStretchEigenvectType::from_elastic_predictor,
                    .elastic_rotation_type =
                        AEI::PrelimPlasticPredictor::ElasticRotationType::from_elastic_predictor,
                    .max_iter = 50,
                    .relative_understress_tol = 1.0e-6,
                    .interval_scanning_param = 0.5,
                },
            .estimate_interpolation =
                {
                    .starting_point_type = AEI::StartingPointType::equiv_stress_history,
                    .user_set_starting_point = 0.5,
                    .max_iter = 50,
                    .interval_scanning_param = 0.5,
                },
            .hardening =
                {
                    .method = AEI::HardeningManagementMethod::integrate_via_evolution_equations,
                    .max_iter_integration = 50,
                    .tol_integration = 1.0e-8,
                },
            .reestimation =
                {
                    .max_num_reestimations = 10,
                    .interval_scanning_param = 0.5,
                },
        });
    problem.materials()->insert(matid_viscoplastic_factor,
        Mat::make_parameter(matid_viscoplastic_factor,
            Core::Materials::MaterialType::mfi_transv_isotrop_elast_viscoplast,
            viscoplastic_factor_data));

    Core::IO::InputParameterContainer multiplicative_split_data;
    multiplicative_split_data.add("NUMMATEL", 1);
    multiplicative_split_data.add("MATIDSEL", std::vector<int>{matid_coup_neo_hooke});
    multiplicative_split_data.add("NUMFACINEL", 1);
    multiplicative_split_data.add("INELDEFGRADFACIDS", std::vector<int>{matid_viscoplastic_factor});
    multiplicative_split_data.add("DENS", 1.0);
    multiplicative_split_data.add("REF_TEMPERATURE", ref_temperature);
    multiplicative_split_data.add("THERMAL_EXPANSION_COEFFICIENT",
        temperature_coupling == TemperatureCoupling::on ? 1.0e-5 : 0.0);
    problem.materials()->insert(matid_multiplicative_split,
        Mat::make_parameter(matid_multiplicative_split,
            Core::Materials::MaterialType::m_multiplicative_split_defgrad_elasthyper,
            multiplicative_split_data));

    // Some generic parameters required for safety checks in the material setup
    auto parameter_list = std::make_shared<Teuchos::ParameterList>();
    parameter_list->sublist("STRUCTURAL DYNAMIC", false).set("MASSLIN", Solid::MassLin::ml_none);
    parameter_list->sublist("SSI CONTROL")
        .set("COUPALGO", SSI::SolutionSchemeOverFields::ssi_IterStagg);
    problem.set_parameter_list(parameter_list);
  }

  /// create @p num_materials independent instances, each with a single Gauss point
  std::vector<std::shared_ptr<Mat::MultiplicativeSplitDefgradElastHyper>> create_materials(
      const std::size_t num_materials)
  {
    Discret::Elements::Fibers fibers;
    fibers.element_fibers.push_back(
        Core::LinAlg::Tensor<double, 3>{{1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0}});

    std::vector<std::shared_ptr<Mat::MultiplicativeSplitDefgradElastHyper>> materials;
    materials.reserve(num_materials);
    for (std::size_t i = 0; i < num_materials; ++i)
    {
      auto material = std::dynamic_pointer_cast<Mat::MultiplicativeSplitDefgradElastHyper>(
          Mat::factory(matid_multiplicative_split));
      material->setup(1, fibers, std::nullopt);
      materials.push_back(std::move(material));
    }
    return materials;
  }

  /// deformation gradient and Green-Lagrange strain of the benchmarked state
  struct DeformationState
  {
    Core::LinAlg::Tensor<double, 3, 3> deformation_gradient;
    Core::LinAlg::SymmetricTensor<double, 3, 3> green_lagrange_strain;
  };

  /// Get the deformation gradient and Green-Lagrange strain for the given state and material
  /// behavior.
  DeformationState get_deformation(
      const PlasticState state, const ViscoplastUtils::MatBehavior mat_behavior)
  {
    Core::LinAlg::Tensor<double, 3, 3> F{};
    switch (state)
    {
      case PlasticState::elastic:
        F = Core::LinAlg::Tensor<double, 3, 3>{
            {{1.0005, 0.00005, 0.00015}, {0.0002, 1.001, 0.0001}, {0.0003, 0.00025, 0.9985}}};
        break;
      case PlasticState::plastic:
      {
        // shear entry such that the equivalent stress is moderately above the yield stress
        const double shear_entry =
            mat_behavior == ViscoplastUtils::MatBehavior::isotropic ? 0.0065 : 0.004;
        F = Core::LinAlg::Tensor<double, 3, 3>{
            {{1.0, shear_entry, 0.0005}, {0.0002, 1.0003, 0.0001}, {0.0003, 0.0002, 0.9998}}};
        break;
      }
    }

    const Core::LinAlg::SymmetricTensor<double, 3, 3> E =
        0.5 * (Core::LinAlg::assume_symmetry(Core::LinAlg::dot(Core::LinAlg::transpose(F), F)) -
                  Core::LinAlg::TensorGenerators::identity<double, 3, 3>);
    return {.deformation_gradient = F, .green_lagrange_strain = E};
  }

  /// Get the expected number of local Newton iterations for the given state and material behavior.
  int expected_local_newton_iterations(
      const PlasticState state, const ViscoplastUtils::MatBehavior mat_behavior)
  {
    switch (state)
    {
      case PlasticState::elastic:
        return 0;
      case PlasticState::plastic:
        switch (mat_behavior)
        {
          case ViscoplastUtils::MatBehavior::isotropic:
            return 7;
          case ViscoplastUtils::MatBehavior::transv_isotropic:
            return 6;
          default:
            FOUR_C_THROW("Unexpected material behavior.");
        }
      default:
        FOUR_C_THROW("Unexpected plastic state.");
    }
  }

  /*!
   * Evaluates the multiplicative split with the viscoplastic factor at a fixed state.
   *
   * Each evaluation needs a fresh material instance, since the evaluation modifies the internal
   * state of the material. In particular, a repeated evaluation of the same instance at the same
   * state would only hit the internal caches. The fresh instances are created and set up in small
   * batches, and the setup time is not included in the benchmark.
   *
   * Before the benchmark, the number of local Newton iterations is checked to ensure that the state
   * is as expected, and large changes in the evaluation time don't stem from changes in the number
   * of iterations.
   */
  template <PlasticState plastic_state, TemperatureCoupling temperature_coupling,
      EvaluationType evaluation = EvaluationType::stress_stiffness,
      ViscoplastUtils::MatBehavior mat_behavior = ViscoplastUtils::MatBehavior::isotropic>
  void evaluate_multiplicative_split_with_viscoplastic_factor(benchmark::State& state)
  {
    Core::Utils::SingletonOwnerRegistry::ScopeGuard guard;

    set_up_viscoplastic_material(mat_behavior, temperature_coupling);

    const Mat::EvaluationContext<3> context{.total_time = &time_step_size,
        .time_step_size = &time_step_size,
        .xi = nullptr,
        .ref_coords = nullptr};

    Teuchos::ParameterList params;
    if (temperature_coupling == TemperatureCoupling::on)
    {
      params.set<double>("temperature", ref_temperature);
      // only read by evaluate_d_stress_d_scalar()
      params.set<int>("differentiationtype", static_cast<int>(Solid::DifferentiationType::temp));
    }

    const auto [F, E] = get_deformation(plastic_state, mat_behavior);
    Core::LinAlg::SymmetricTensor<double, 3, 3> stress{};
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> cmat{};

    // check the number of local Newton iterations once upfront on a separate instance
    {
      const auto material = create_materials(1).front();
      material->evaluate(&F, E, params, context, stress, cmat, gp, ele_gid);
      Core::LinAlg::SerialDenseMatrix local_newton_iterations(1, 1);
      material->evaluate_output_data("local_newton_iters", local_newton_iterations);
      const int iterations = static_cast<int>(local_newton_iterations(0, 0));
      const int expected_iterations = expected_local_newton_iterations(plastic_state, mat_behavior);
      // fail the whole benchmark immediately, even in dry runs
      FOUR_C_ASSERT_ALWAYS(iterations == expected_iterations,
          "Expected {} local Newton iterations, got {}.", expected_iterations, iterations);
    }

    // Create the fresh material instances in batches of pool_size:
    // - Creating one instance per iteration would require pausing the timing in every iteration,
    //   which should be avoided since PauseTiming() and ResumeTiming() are relatively heavyweight
    //   (see their documentation in Google Benchmark).
    // - Creating all instances upfront would let the memory grow with the number of iterations,
    //   which is chosen adaptively by Google Benchmark and can be very large for the fast cases.
    constexpr std::size_t pool_size = 64;
    auto materials = create_materials(pool_size);
    auto material = materials.begin();

    // benchmark loop
    for (auto _ : state)
    {
      if (material == materials.end())
      {
        // if no fresh materials are available, pause timing and create a new batch of materials
        state.PauseTiming();
        // replace materials with a fresh batch
        materials = create_materials(pool_size);
        // continue with the first material of the new batch
        material = materials.begin();
        state.ResumeTiming();
      }

      (*material)->evaluate(&F, E, params, context, stress, cmat, gp, ele_gid);
      // mark the results as used to prevent compiler optimizations
      benchmark::DoNotOptimize(stress);
      benchmark::DoNotOptimize(cmat);
      if constexpr (evaluation == EvaluationType::stress_stiffness_temperature_derivative)
      {
        Core::LinAlg::SymmetricTensor<double, 3, 3> dstress_dT =
            (*material)->evaluate_d_stress_d_scalar(F, E, params, context, gp, ele_gid);
        benchmark::DoNotOptimize(dstress_dT);
      }
      // increment material iterator
      material++;
    }
  }
}  // namespace

// Elastic
BENCHMARK(evaluate_multiplicative_split_with_viscoplastic_factor<PlasticState::elastic,
    TemperatureCoupling::off>);
BENCHMARK(evaluate_multiplicative_split_with_viscoplastic_factor<PlasticState::elastic,
    TemperatureCoupling::on>);
BENCHMARK(evaluate_multiplicative_split_with_viscoplastic_factor<PlasticState::elastic,
    TemperatureCoupling::on, EvaluationType::stress_stiffness_temperature_derivative>);

// Plastic
BENCHMARK(evaluate_multiplicative_split_with_viscoplastic_factor<PlasticState::plastic,
    TemperatureCoupling::off>);
BENCHMARK(evaluate_multiplicative_split_with_viscoplastic_factor<PlasticState::plastic,
    TemperatureCoupling::on>);
BENCHMARK(evaluate_multiplicative_split_with_viscoplastic_factor<PlasticState::plastic,
    TemperatureCoupling::on, EvaluationType::stress_stiffness_temperature_derivative>);

// Plastic + transversely isotropic
BENCHMARK(evaluate_multiplicative_split_with_viscoplastic_factor<PlasticState::plastic,
    TemperatureCoupling::off, EvaluationType::stress_stiffness,
    ViscoplastUtils::MatBehavior::transv_isotropic>);
BENCHMARK(evaluate_multiplicative_split_with_viscoplastic_factor<PlasticState::plastic,
    TemperatureCoupling::on, EvaluationType::stress_stiffness,
    ViscoplastUtils::MatBehavior::transv_isotropic>);

FOUR_C_NAMESPACE_CLOSE

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_comm_pack_buffer.hpp"
#include "4C_global_data.hpp"
#include "4C_io_input_parameter_container.templates.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_plasticmohrcoulomb.hpp"
#include "4C_material_parameter_base.hpp"
#include "4C_unittest_utils_assertions_test.hpp"
#include "4C_utils_singleton_owner.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <numbers>

namespace
{
  using namespace FourC;

  struct EvaluationResult
  {
    std::array<double, 3> kirchhoff_stress{};
    Core::LinAlg::SymmetricTensor<double, 3, 3> pk2_stress{};
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> tangent{};
    Core::LinAlg::SymmetricTensor<double, 3, 3> green_lagrange_strain{};
  };

  class MohrCoulombTest : public ::testing::Test
  {
   protected:
    void SetUp() override { create_material(); }

    void create_material(double dilatancy_angle = 20.0 * std::numbers::pi / 180.0,
        double linear_hardening = 0.0, double saturation_hardening = 0.0,
        double hardening_exponent = 0.0, int max_iterations = 100)
    {
      Core::IO::InputParameterContainer container;
      container.add("YOUNG_MODULUS", 1000.0);
      container.add("POISSON_RATIO", 0.25);
      container.add("DENSITY", 0.0);
      container.add("COHESION", 10.0);
      container.add("FRICTION_ANGLE", 30.0 * std::numbers::pi / 180.0);
      container.add("DILATANCY_ANGLE", dilatancy_angle);
      container.add("LINEAR_HARDENING", linear_hardening);
      container.add("SATURATION_HARDENING", saturation_hardening);
      container.add("HARDENING_EXP", hardening_exponent);
      container.add("TOLERANCE", 1.0e-12);
      container.add("MAX_ITERATIONS", max_iterations);

      const int material_id = next_material_id_++;
      parameter_ = std::shared_ptr(
          Mat::make_parameter(material_id, Core::Materials::m_plmohrcoulomb, container));
      Global::Problem& problem = *Global::Problem::instance();
      problem.materials()->set_read_from_problem(0);
      problem.materials()->insert(material_id, parameter_);
      material_ = std::make_shared<Mat::PlasticMohrCoulomb>(
          dynamic_cast<Mat::PAR::PlasticMohrCoulomb*>(parameter_.get()));
      material_->setup(1, {}, {});
    }

    EvaluationResult evaluate(const std::array<double, 3>& logarithmic_strain)
    {
      Core::LinAlg::Tensor<double, 3, 3> deformation_gradient{};
      EvaluationResult result;
      for (int i = 0; i < 3; ++i)
      {
        deformation_gradient(i, i) = std::exp(logarithmic_strain[i]);
        result.green_lagrange_strain(i, i) = 0.5 * (std::exp(2.0 * logarithmic_strain[i]) - 1.0);
      }

      Teuchos::ParameterList parameters;
      double total_time = 0.0;
      double time_step_size = 1.0;
      Mat::EvaluationContext<3> context{.total_time = &total_time,
          .time_step_size = &time_step_size,
          .xi = {},
          .ref_coords = nullptr};
      material_->evaluate(&deformation_gradient, result.green_lagrange_strain, parameters, context,
          result.pk2_stress, result.tangent, 0, 0);
      for (int i = 0; i < 3; ++i)
        result.kirchhoff_stress[i] =
            result.pk2_stress(i, i) * std::exp(2.0 * logarithmic_strain[i]);
      std::ranges::sort(result.kirchhoff_stress, std::greater<>());
      return result;
    }

    double scalar_output(const std::string& name)
    {
      Core::LinAlg::SerialDenseMatrix data(1, 1);
      EXPECT_TRUE(material_->evaluate_output_data(name, data));
      return data(0, 0);
    }

    static double yield_function(const std::array<double, 3>& stress, double cohesion_value)
    {
      constexpr double friction_angle = 30.0 * std::numbers::pi / 180.0;
      const double k = (1.0 + std::sin(friction_angle)) / (1.0 - std::sin(friction_angle));
      return k * stress[0] - stress[2] - 2.0 * std::sqrt(k) * cohesion_value;
    }

    std::shared_ptr<Core::Mat::PAR::Parameter> parameter_;
    std::shared_ptr<Mat::PlasticMohrCoulomb> material_;
    int next_material_id_ = 1;
    Core::Utils::SingletonOwnerRegistry::ScopeGuard guard_;
  };

  TEST_F(MohrCoulombTest, ElasticHenckyResponse)
  {
    const std::array<double, 3> logarithmic_strain{0.001, -0.0005, -0.001};
    const EvaluationResult result = evaluate(logarithmic_strain);

    constexpr double shear_modulus = 400.0;
    constexpr double lame_parameter = 400.0;
    std::array<double, 3> expected{};
    const double trace_logarithmic_strain =
        logarithmic_strain[0] + logarithmic_strain[1] + logarithmic_strain[2];
    for (int i = 0; i < 3; ++i)
      expected[i] =
          2.0 * shear_modulus * logarithmic_strain[i] + lame_parameter * trace_logarithmic_strain;
    std::ranges::sort(expected, std::greater<>());

    for (int i = 0; i < 3; ++i) EXPECT_NEAR(result.kirchhoff_stress[i], expected[i], 1.0e-12);
    EXPECT_DOUBLE_EQ(scalar_output("accumulated_plastic_strain"), 0.0);
  }

  TEST_F(MohrCoulombTest, ReturnsToSmoothFace)
  {
    const EvaluationResult result = evaluate({0.025, -0.0453125, -0.2});
    EXPECT_NEAR(yield_function(result.kirchhoff_stress, 10.0), 0.0, 1.0e-9);
    EXPECT_GT(result.kirchhoff_stress[0] - result.kirchhoff_stress[1], 1.0);
    EXPECT_GT(result.kirchhoff_stress[1] - result.kirchhoff_stress[2], 1.0);
  }

  TEST_F(MohrCoulombTest, ReturnsToCompressionEdge)
  {
    const EvaluationResult result = evaluate({0.025, 0.0109375, -0.2});
    EXPECT_NEAR(yield_function(result.kirchhoff_stress, 10.0), 0.0, 1.0e-9);
    EXPECT_NEAR(result.kirchhoff_stress[0], result.kirchhoff_stress[1], 1.0e-9);
  }

  TEST_F(MohrCoulombTest, ReturnsToTensionEdge)
  {
    const EvaluationResult result = evaluate({0.025, -0.0453125, -0.0453125});
    EXPECT_NEAR(yield_function(result.kirchhoff_stress, 10.0), 0.0, 1.0e-9);
    EXPECT_NEAR(result.kirchhoff_stress[1], result.kirchhoff_stress[2], 1.0e-9);
  }

  TEST_F(MohrCoulombTest, ReturnsToApex)
  {
    const EvaluationResult result = evaluate({0.025, 0.0109375, 0.0109375});
    EXPECT_NEAR(result.kirchhoff_stress[0], result.kirchhoff_stress[1], 1.0e-9);
    EXPECT_NEAR(result.kirchhoff_stress[1], result.kirchhoff_stress[2], 1.0e-9);
    EXPECT_NEAR(yield_function(result.kirchhoff_stress, 10.0), 0.0, 1.0e-9);
  }

  TEST_F(MohrCoulombTest, NonAssociatedFlowAndOutputs)
  {
    evaluate({0.025, -0.0453125, -0.2});

    EXPECT_GT(scalar_output("accumulated_plastic_strain"), 0.0);
    EXPECT_GT(scalar_output("accumulated_plastic_volumetric_strain"), 0.0);
    EXPECT_GT(scalar_output("local_dissipated_energy"), 0.0);

    Core::LinAlg::SerialDenseMatrix plastic_strain(1, 6);
    ASSERT_TRUE(material_->evaluate_output_data("plastic_strain", plastic_strain));
    double norm = 0.0;
    for (int i = 0; i < 6; ++i) norm += plastic_strain(0, i) * plastic_strain(0, i);
    EXPECT_GT(norm, 0.0);
  }

  TEST_F(MohrCoulombTest, AssociativeFlow)
  {
    create_material(30.0 * std::numbers::pi / 180.0);
    const EvaluationResult result = evaluate({0.025, -0.0453125, -0.2});

    EXPECT_NEAR(yield_function(result.kirchhoff_stress, 10.0), 0.0, 1.0e-9);
    EXPECT_GT(scalar_output("accumulated_plastic_volumetric_strain"), 0.0);
  }

  TEST_F(MohrCoulombTest, RejectsZeroDilatation) { EXPECT_ANY_THROW(create_material(0.0)); }

  TEST_F(MohrCoulombTest, ReportsLocalSolverFailure)
  {
    create_material(20.0 * std::numbers::pi / 180.0, 20.0, 30.0, 5.0, 1);
    EXPECT_ANY_THROW(evaluate({0.025, -0.0453125, -0.2}));
  }

  TEST_F(MohrCoulombTest, LinearAndVoceHardening)
  {
    create_material(20.0 * std::numbers::pi / 180.0, 20.0, 30.0, 5.0);
    const EvaluationResult result = evaluate({0.025, -0.0453125, -0.2});
    const double plastic_strain = scalar_output("accumulated_plastic_strain");
    const double current_cohesion =
        10.0 + 20.0 * plastic_strain + 20.0 * (1.0 - std::exp(-5.0 * plastic_strain));

    EXPECT_GT(plastic_strain, 0.0);
    EXPECT_NEAR(yield_function(result.kirchhoff_stress, current_cohesion), 0.0, 1.0e-8);
  }

  TEST_F(MohrCoulombTest, AnalyticalYieldSurfaceMeridians)
  {
    const char* output_path = std::getenv("FOUR_C_MOHR_COULOMB_YIELD_CURVE_OUTPUT");
    std::ofstream output;
    if (output_path != nullptr)
    {
      output.open(output_path);
      ASSERT_TRUE(output.good());
      output << "branch,pressure,analytical_von_mises,calculated_von_mises\n";
    }

    constexpr double bulk_modulus = 2000.0 / 3.0;
    constexpr double shear_modulus = 400.0;
    constexpr double friction_angle = 30.0 * std::numbers::pi / 180.0;
    const double k = (1.0 + std::sin(friction_angle)) / (1.0 - std::sin(friction_angle));
    const double q0 = 2.0 * std::sqrt(k) * 10.0;

    const auto evaluate_meridian = [&](double trial_pressure, bool compression_meridian)
    {
      const double denominator = compression_meridian ? k + 2.0 : 2.0 * k + 1.0;
      const double analytical_von_mises = 3.0 * (q0 + (k - 1.0) * trial_pressure) / denominator;
      const double trial_von_mises = 1.25 * analytical_von_mises;
      const std::array<double, 3> trial_stress =
          compression_meridian
              ? std::array<double, 3>{-trial_pressure + trial_von_mises / 3.0,
                    -trial_pressure + trial_von_mises / 3.0,
                    -trial_pressure - 2.0 * trial_von_mises / 3.0}
              : std::array<double, 3>{-trial_pressure + 2.0 * trial_von_mises / 3.0,
                    -trial_pressure - trial_von_mises / 3.0,
                    -trial_pressure - trial_von_mises / 3.0};

      const double mean_trial_stress = (trial_stress[0] + trial_stress[1] + trial_stress[2]) / 3.0;
      std::array<double, 3> logarithmic_strain{};
      for (int i = 0; i < 3; ++i)
        logarithmic_strain[i] = (trial_stress[i] - mean_trial_stress) / (2.0 * shear_modulus) +
                                mean_trial_stress / (3.0 * bulk_modulus);

      const EvaluationResult result = evaluate(logarithmic_strain);
      const double calculated_pressure =
          -(result.kirchhoff_stress[0] + result.kirchhoff_stress[1] + result.kirchhoff_stress[2]) /
          3.0;
      const double calculated_von_mises = result.kirchhoff_stress[0] - result.kirchhoff_stress[2];
      const double analytical_at_calculated_pressure =
          3.0 * (q0 + (k - 1.0) * calculated_pressure) / denominator;
      EXPECT_NEAR(calculated_von_mises, analytical_at_calculated_pressure, 1.0e-8);

      if (output.is_open())
        output << (compression_meridian ? "triaxial_compression" : "triaxial_tension") << ','
               << calculated_pressure << ',' << analytical_at_calculated_pressure << ','
               << calculated_von_mises << '\n';
    };

    for (int point = 0; point <= 20; ++point)
    {
      const double pressure = 5.0 * point;
      evaluate_meridian(pressure, true);
      evaluate_meridian(pressure, false);
    }
  }

  TEST_F(MohrCoulombTest, SmoothFaceConsistentTangent)
  {
    const std::array<double, 3> logarithmic_strain{0.025, -0.0453125, -0.2};
    const EvaluationResult base = evaluate(logarithmic_strain);
    constexpr double perturbation = 1.0e-7;

    for (int j = 0; j < 3; ++j)
    {
      auto perturbed_green_lagrange_strain = base.green_lagrange_strain;
      perturbed_green_lagrange_strain(j, j) += perturbation;
      Core::LinAlg::Tensor<double, 3, 3> perturbed_deformation_gradient{};
      for (int i = 0; i < 3; ++i)
        perturbed_deformation_gradient(i, i) =
            std::sqrt(1.0 + 2.0 * perturbed_green_lagrange_strain(i, i));

      Core::LinAlg::SymmetricTensor<double, 3, 3> perturbed_stress{};
      Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> unused_tangent{};
      Teuchos::ParameterList parameters;
      double total_time = 0.0;
      double time_step_size = 1.0;
      Mat::EvaluationContext<3> context{.total_time = &total_time,
          .time_step_size = &time_step_size,
          .xi = {},
          .ref_coords = nullptr};
      material_->evaluate(&perturbed_deformation_gradient, perturbed_green_lagrange_strain,
          parameters, context, perturbed_stress, unused_tangent, 0, 0);

      for (int i = 0; i < 3; ++i)
      {
        const double finite_difference =
            (perturbed_stress(i, i) - base.pk2_stress(i, i)) / perturbation;
        EXPECT_NEAR(base.tangent(i, i, j, j), finite_difference, 2.0e-3);
      }
    }
  }

  TEST_F(MohrCoulombTest, PackAndUnpackPreservesState)
  {
    evaluate({0.025, -0.0453125, -0.2});
    const double expected_plastic_strain = scalar_output("accumulated_plastic_strain");

    Core::Communication::PackBuffer pack_buffer;
    material_->pack(pack_buffer);
    std::vector<char> packed_data;
    swap(packed_data, pack_buffer());
    Core::Communication::UnpackBuffer unpack_buffer(packed_data);
    Mat::PlasticMohrCoulomb unpacked_material;
    unpacked_material.unpack(unpack_buffer);

    Core::LinAlg::SerialDenseMatrix data(1, 1);
    ASSERT_TRUE(unpacked_material.evaluate_output_data("accumulated_plastic_strain", data));
    EXPECT_DOUBLE_EQ(data(0, 0), expected_plastic_strain);
  }
}  // namespace

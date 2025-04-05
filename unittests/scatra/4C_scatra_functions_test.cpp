// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_scatra_functions.hpp"

namespace
{
  using namespace FourC;

  TEST(ScatraFunctionsMagnet, TestEvaluateMagneticForceCylindricalMagnetLinearSaturation)
  {
    constexpr auto parameters = ScaTra::CylinderMagnetParameters{
        .magnet_radius = 1.2,
        .magnet_length = 2.0,
        .magnetic_permeability = 12.234,
        .magnet_magnetization = 0.234,
        .magnet_position = {.x = -1.0, .y = 0.0, .z = 2.3},
        .dynamic_viscosity_fluid = 2.0,
        .magnet_rotation = {.x_axis = 0.0, .y_axis = 0.0},
        .particle_radius = 1.0,
        .particle_magnetization_model_type =
            ScaTra::ParticleMagnetizationModelType::linear_with_saturation,
        .particle_magnetization_parameters = {.saturation_magnetization = 10,
            .susceptibility = 3.0},
    };
    const auto cylindrical_magnet = ScaTra::CylinderMagnetFunction(parameters);

    std::vector<double> result_1(3);
    cylindrical_magnet.evaluate_vector(
        std::span<const double, 3>({-1.906, 0.0, 1.243}), 0.0, std::span<double, 3>(result_1));

    const std::vector test_result_1 = {-0.0088747501861973048, 0.0, 0.035341855488173306};
    for (std::size_t i = 0; i < result_1.size(); ++i)
    {
      EXPECT_NEAR(result_1[i], test_result_1[i], 1e-12) << "Results differ at index " << i;
    }

    std::vector<double> result_2(3);
    cylindrical_magnet.evaluate_vector(
        std::span<const double, 3>({1.21, 0.0, -2.34}), 0.0, std::span<double, 3>(result_2));
    const std::vector test_result_2 = {-3.2717442048087439e-06, 0.0, 5.3522708863354522e-06};
    for (std::size_t i = 0; i < result_2.size(); ++i)
    {
      EXPECT_NEAR(result_2[i], test_result_2[i], 1e-12) << "Results differ at index " << i;
    }
  }

  TEST(ScatraFunctionsMagnet, TestEvaluateMagneticForceCylindricalMagnetLinear)
  {
    constexpr auto parameters = ScaTra::CylinderMagnetParameters{
        .magnet_radius = 1.2,
        .magnet_length = 2.0,
        .magnetic_permeability = 12.234,
        .magnet_magnetization = 0.234,
        .magnet_position = {.x = -1.0, .y = 0.0, .z = 2.3},
        .dynamic_viscosity_fluid = 2.0,
        .magnet_rotation = {.x_axis = 0.0, .y_axis = 0.0},
        .particle_radius = 1.0,
        .particle_magnetization_model_type = ScaTra::ParticleMagnetizationModelType::linear,
        .particle_magnetization_parameters = {.susceptibility = 10.0},
    };
    const auto cylindrical_magnet = ScaTra::CylinderMagnetFunction(parameters);

    std::vector<double> result_1(3);
    cylindrical_magnet.evaluate_vector(
        std::span<const double, 3>({-1.906, 0.0, 1.243}), 0.0, std::span<double, 3>(result_1));

    const std::vector test_result_1 = {-0.013653461824920086, 0.0, 0.054372085366405504};
    for (std::size_t i = 0; i < result_1.size(); ++i)
    {
      EXPECT_NEAR(result_1[i], test_result_1[i], 1e-12) << "Results differ at index " << i;
    }

    std::vector<double> result_2(3);
    cylindrical_magnet.evaluate_vector(
        std::span<const double, 3>({1.21, 0.0, -2.34}), 0.0, std::span<double, 3>(result_2));
    const std::vector test_result_2 = {-5.0334526227827343e-06, 0.0, 8.2342629020545756e-06};
    for (std::size_t i = 0; i < result_2.size(); ++i)
    {
      EXPECT_NEAR(result_2[i], test_result_2[i], 1e-12) << "Results differ at index " << i;
    }
  }

  TEST(ScatraFunctionsMagnet, TestEvaluateMagneticForceCylindricalMagnetSuperparamagnetic)
  {
    constexpr auto parameters = ScaTra::CylinderMagnetParameters{
        .magnet_radius = 1.2,
        .magnet_length = 2.0,
        .magnetic_permeability = 12.234,
        .magnet_magnetization = 0.234,
        .magnet_position = {.x = -1.0, .y = 0.0, .z = 2.3},
        .dynamic_viscosity_fluid = 2.0,
        .magnet_rotation = {.x_axis = 0.0, .y_axis = 0.0},
        .particle_radius = 1.0,
        .particle_magnetization_model_type =
            ScaTra::ParticleMagnetizationModelType::superparamagnetic,
        .particle_magnetization_parameters =
            {
                .saturation_magnetization = 10,
            },
    };
    const auto cylindrical_magnet = ScaTra::CylinderMagnetFunction(parameters);

    std::vector<double> result_1(3);
    cylindrical_magnet.evaluate_vector(
        std::span<const double, 3>({-1.906, 0.0, 1.243}), 0.0, std::span<double, 3>(result_1));

    const std::vector test_result_1 = {-0.017749500372396115, 0.0, 0.070683710976327155};
    for (std::size_t i = 0; i < result_1.size(); ++i)
    {
      EXPECT_NEAR(result_1[i], test_result_1[i], 1e-12) << "Results differ at index " << i;
    }

    std::vector<double> result_2(3);
    cylindrical_magnet.evaluate_vector(
        std::span<const double, 3>({1.21, 0.0, -2.34}), 0.0, std::span<double, 3>(result_2));
    const std::vector test_result_2 = {-6.5434884096175547e-06, 0.0, 1.0704541772670948e-05};
    for (std::size_t i = 0; i < result_2.size(); ++i)
    {
      EXPECT_NEAR(result_2[i], test_result_2[i], 1e-12) << "Results differ at index " << i;
    }
  }

  TEST(ScatraFunctionsMagnet, TestRealValues)
  {
    constexpr auto parameters = ScaTra::CylinderMagnetParameters{
        .magnet_radius = 2.5,
        .magnet_length = 5.0,
        .magnetic_permeability = 1.25663706212,
        .magnet_magnetization = 1e3,
        .magnet_position = {.x = 0.0, .y = 0.0, .z = -4.5},
        .dynamic_viscosity_fluid = 0.001,
        .magnet_rotation = {.x_axis = 0.0, .y_axis = 0.0},
        .particle_radius = 100e-6,
        .particle_magnetization_model_type =
            ScaTra::ParticleMagnetizationModelType::linear_with_saturation,
        .particle_magnetization_parameters = {.saturation_magnetization = 4.78e2,
            .susceptibility = 3.0},

    };
    const auto cylindrical_magnet = ScaTra::CylinderMagnetFunction(parameters);

    std::vector<double> result_1(3);
    cylindrical_magnet.evaluate_vector(
        std::span<const double, 3>({-2.0, 0.0, -1.0}), 0.0, std::span<double, 3>(result_1));

    std::vector test_result_1 = {0.056881033313996132, 0.0, -0.16855012500303279};
    for (std::size_t i = 0; i < result_1.size(); ++i)
    {
      EXPECT_NEAR(result_1[i], test_result_1[i], 1e-12) << "Results differ at index " << i;
    }

    std::vector<double> result_2(3);
    cylindrical_magnet.evaluate_vector(
        std::span<const double, 3>({2.0, 0.0, -1.0}), 0.0, std::span<double, 3>(result_2));
    std::vector test_result_2 = {-0.056881033313996132, 0.0, -0.1685501250030326};
    for (std::size_t i = 0; i < result_2.size(); ++i)
    {
      EXPECT_NEAR(result_2[i], test_result_2[i], 1e-12) << "Results differ at index " << i;
    }
  }
}  // namespace
// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_solid_3D_ele_utils.hpp"

#include "4C_unittest_utils_assertions_test.hpp"

namespace
{
  using namespace FourC;

  Core::LinAlg::Matrix<3, 3> get_f()
  {
    Core::LinAlg::Matrix<3, 3> F(Core::LinAlg::Initialization::zero);
    F(0, 0) = 1.1;
    F(0, 1) = 0.2;
    F(0, 2) = 0.5;
    F(1, 0) = 0.14;
    F(1, 1) = 1.2;
    F(1, 2) = 0.3;
    F(2, 0) = 0.05;
    F(2, 1) = 0.2;
    F(2, 2) = 1.3;

    return F;
  }

  TEST(TestStressStrainMeasures, green_lagrange_to_euler_almansi)
  {
    Core::LinAlg::Matrix<6, 1> green_lagrange_strain(
        std::array<double, 6>{0.11605, 0.26, 0.515, 0.398, 0.72, 0.657}.data());

    Core::LinAlg::Matrix<6, 1> euler_almansi_strain =
        Solid::Utils::green_lagrange_to_euler_almansi(green_lagrange_strain, get_f());

    Core::LinAlg::Matrix<6, 1> euler_almansi_strain_ref(
        std::array<double, 6>{0.055233442151184, 0.101134166403205, 0.104112596224498,
            0.182642289473823, 0.214768580862521, 0.315358749090858}
            .data());

    FOUR_C_EXPECT_NEAR(euler_almansi_strain, euler_almansi_strain_ref, 1e-13);
  }

  TEST(TestStressStrainMeasures, green_lagrange_to_log_strain)
  {
    Core::LinAlg::Matrix<6, 1> green_lagrange_strain(
        std::array<double, 6>{0.11605, 0.26, 0.515, 0.398, 0.72, 0.657}.data());

    Core::LinAlg::Matrix<6, 1> log_strain =
        Solid::Utils::green_lagrange_to_log_strain(green_lagrange_strain);

    Core::LinAlg::Matrix<6, 1> log_strain_ref(
        std::array<double, 6>{0.039139830823291, 0.150129540734586, 0.281109187392933,
            0.218832208837098, 0.400808067245772, 0.400940161591198}
            .data());

    FOUR_C_EXPECT_NEAR(log_strain, log_strain_ref, 1e-13);
  }

  TEST(TestStressStrainMeasures, SecondPiolaKirchhoffToCauchy)
  {
    Core::LinAlg::Matrix<6, 1> pk2(std::array<double, 6>{283.6946919505318, 195.86721709838096,
        202.01904686970775, 142.72731871521245, 182.86374040756576, 278.020938548381}
            .data());

    Core::LinAlg::Matrix<6, 1> cauchy(Core::LinAlg::Initialization::zero);
    Solid::Utils::pk2_to_cauchy(pk2, get_f(), cauchy);

    Core::LinAlg::Matrix<6, 1> cauchy_ref(
        std::array<double, 6>{504.0646185061422, 317.85764952017706, 302.4131750725638,
            340.6815203116966, 306.97914008976466, 411.0514636046741}
            .data());

    FOUR_C_EXPECT_NEAR(cauchy, cauchy_ref, 1e-12);
  }
}  // namespace
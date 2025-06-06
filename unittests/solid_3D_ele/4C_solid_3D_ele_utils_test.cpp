// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_solid_3D_ele_utils.hpp"

#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_unittest_utils_assertions_test.hpp"

namespace
{
  using namespace FourC;

  Core::LinAlg::Tensor<double, 3, 3> get_f()
  {
    Core::LinAlg::Tensor<double, 3, 3> F{{
        {1.1, 0.2, 0.5},
        {0.14, 1.2, 0.3},
        {0.05, 0.2, 1.3},
    }};

    return F;
  }

  TEST(TestStressStrainMeasures, green_lagrange_to_euler_almansi)
  {
    Core::LinAlg::SymmetricTensor<double, 3, 3> green_lagrange_strain =
        Core::LinAlg::assume_symmetry(Core::LinAlg::Tensor<double, 3, 3>{{
            {0.11605, 0.199, 0.3285},
            {0.199, 0.26, 0.36},
            {0.3285, 0.36, 0.515},
        }});

    Core::LinAlg::SymmetricTensor<double, 3, 3> euler_almansi_strain =
        Solid::Utils::green_lagrange_to_euler_almansi(green_lagrange_strain, get_f());

    Core::LinAlg::SymmetricTensor<double, 3, 3> euler_almansi_strain_ref =
        Core::LinAlg::assume_symmetry(Core::LinAlg::Tensor<double, 3, 3>{
            {{0.055233442151184, 0.0913211447369115, 0.157679374545429},
                {0.0913211447369115, 0.101134166403205, 0.1073842904312605},
                {0.157679374545429, 0.1073842904312605, 0.104112596224498}}});

    FOUR_C_EXPECT_NEAR(euler_almansi_strain, euler_almansi_strain_ref, 1e-13);
  }

  TEST(TestStressStrainMeasures, green_lagrange_to_log_strain)
  {
    Core::LinAlg::SymmetricTensor<double, 3, 3> green_lagrange_strain =
        Core::LinAlg::assume_symmetry(Core::LinAlg::Tensor<double, 3, 3>{{
            {0.11605, 0.199, 0.3285},
            {0.199, 0.26, 0.36},
            {0.3285, 0.36, 0.515},
        }});

    Core::LinAlg::SymmetricTensor<double, 3, 3> log_strain =
        Solid::Utils::green_lagrange_to_log_strain(green_lagrange_strain);

    Core::LinAlg::SymmetricTensor<double, 3, 3> log_strain_ref = Core::LinAlg::assume_symmetry(
        Core::LinAlg::Tensor<double, 3, 3>{{{0.039139830823291, 0.10941610441855, 0.20047008079560},
            {0.10941610441855, 0.150129540734586, 0.20040403362289},
            {0.20047008079560, 0.20040403362289, 0.281109187392933}}});

    FOUR_C_EXPECT_NEAR(log_strain, log_strain_ref, 1e-13);
  }

  TEST(TestStressStrainMeasures, SecondPiolaKirchhoffToCauchy)
  {
    Core::LinAlg::SymmetricTensor<double, 3, 3> pk2 =
        Core::LinAlg::assume_symmetry(Core::LinAlg::Tensor<double, 3, 3>{
            {{283.6946919505318, 142.72731871521245, 278.020938548381},
                {142.72731871521245, 195.86721709838096, 182.86374040756576},
                {278.020938548381, 182.86374040756576, 202.01904686970775}}});

    Core::LinAlg::SymmetricTensor<double, 3, 3> cauchy = Solid::Utils::pk2_to_cauchy(pk2, get_f());

    Core::LinAlg::SymmetricTensor<double, 3, 3> cauchy_ref =
        Core::LinAlg::assume_symmetry(Core::LinAlg::Tensor<double, 3, 3>{
            {{504.0646185061422, 340.6815203116966, 411.0514636046741},
                {340.6815203116966, 317.85764952017706, 306.97914008976466},
                {411.0514636046741, 306.97914008976466, 302.4131750725638}}});

    FOUR_C_EXPECT_NEAR(cauchy, cauchy_ref, 1e-12);
  }
}  // namespace
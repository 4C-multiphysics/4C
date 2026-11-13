// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_mat_anisotropy.hpp"
#include "4C_mat_elast_aniso_structuraltensor_strategy.hpp"
#include "4C_mat_elast_coupanisoexpo.hpp"
#include "4C_unittest_utils_assertions_test.hpp"


namespace
{
  using namespace FourC;

  class CoupAnisoExpoAnisotropyExtensionGaussPointFiberTest
      : public ::testing::TestWithParam<std::tuple<int, int>>
  {
   protected:
    CoupAnisoExpoAnisotropyExtensionGaussPointFiberTest()
        : anisotropy_(),
          gpFibers_(2, std::vector<Core::LinAlg::Tensor<double, 3>>(2)),
          gpTensors_(2, std::vector<Core::LinAlg::SymmetricTensor<double, 3, 3>>(2))
    {
      /// initialize dummy fibers
      // gp 0
      gpFibers_[0][0](0) = 0.469809238649817;
      gpFibers_[0][0](1) = 0.872502871778232;
      gpFibers_[0][0](2) = 0.134231211042805;

      gpFibers_[0][1](0) = 0.071428571428571;
      gpFibers_[0][1](1) = 0.142857142857143;
      gpFibers_[0][1](2) = 0.214285714285714;

      // gp 1
      gpFibers_[1][0](0) = 0.245358246032859;
      gpFibers_[1][0](1) = 0.858753861115007;
      gpFibers_[1][0](2) = 0.449823451060242;

      gpFibers_[1][1](0) = 0.068965517241379;
      gpFibers_[1][1](1) = 0.103448275862069;
      gpFibers_[1][1](2) = 0.137931034482759;

      for (std::size_t gp = 0; gp < 2; ++gp)
      {
        for (std::size_t i = 0; i < 2; ++i)
        {
          gpTensors_[gp][i] = Core::LinAlg::self_dyadic(gpFibers_[gp][i]);
        }
      }

      setup_anisotropy_extension();
    }

    void setup_anisotropy_extension()
    {
      int fiber_id = std::get<0>(GetParam());
      auto strategy = std::make_shared<Mat::Elastic::StructuralTensorStrategyStandard>(nullptr);
      anisotropyExtension_ = std::make_unique<Mat::Elastic::CoupAnisoExpoAnisotropyExtension>(
          3, 0.0, false, strategy, fiber_id);
      anisotropyExtension_->register_needed_tensors(
          Mat::FiberAnisotropyExtension<1>::FIBER_VECTORS |
          Mat::FiberAnisotropyExtension<1>::STRUCTURAL_TENSOR);
      anisotropy_.register_anisotropy_extension(*anisotropyExtension_);
      anisotropy_.set_number_of_gauss_points(2);

      // Setup Gauss point fibers
      anisotropy_.set_gauss_point_fibers(gpFibers_);
    }

    [[nodiscard]] int get_gauss_point() const { return std::get<1>(GetParam()); }

    [[nodiscard]] int get_fiber_id() const { return std::get<0>(GetParam()); }

    Mat::Anisotropy anisotropy_;
    std::unique_ptr<Mat::Elastic::CoupAnisoExpoAnisotropyExtension> anisotropyExtension_;

    std::vector<std::vector<Core::LinAlg::Tensor<double, 3>>> gpFibers_;
    std::vector<std::vector<Core::LinAlg::SymmetricTensor<double, 3, 3>>> gpTensors_;
  };

  TEST_P(CoupAnisoExpoAnisotropyExtensionGaussPointFiberTest, GetScalarProduct)
  {
    EXPECT_NEAR(anisotropyExtension_->get_scalar_product(get_gauss_point()), 1.0, 1e-10);
  }

  TEST_P(CoupAnisoExpoAnisotropyExtensionGaussPointFiberTest, get_fiber)
  {
    FOUR_C_EXPECT_NEAR(anisotropyExtension_->get_fiber(get_gauss_point()),
        gpFibers_[get_gauss_point()][get_fiber_id() - 1], 1e-10);
  }

  TEST_P(CoupAnisoExpoAnisotropyExtensionGaussPointFiberTest, get_structural_tensor)
  {
    FOUR_C_EXPECT_NEAR(anisotropyExtension_->get_structural_tensor(get_gauss_point()),
        gpTensors_[get_gauss_point()][get_fiber_id() - 1], 1e-10);
  }

  INSTANTIATE_TEST_SUITE_P(GaussPoints, CoupAnisoExpoAnisotropyExtensionGaussPointFiberTest,
      ::testing::Combine(::testing::Values(1, 2), ::testing::Values(0, 1)));
}  // namespace
// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_mat_anisotropy.hpp"
#include "4C_mat_elast_aniso_structuraltensor_strategy.hpp"
#include "4C_mat_elast_coupanisoexpo.hpp"
#include "4C_unittest_utils_assertions_test.hpp"

#include <tuple>


namespace
{
  using namespace FourC;

  class CoupAnisoExpoAnisotropyExtensionElementFiberTest
      : public ::testing::TestWithParam<std::tuple<int, int>>
  {
   public:
    CoupAnisoExpoAnisotropyExtensionElementFiberTest()
        : anisotropy_(), eleFibers_(2), eleTensors_(2)
    {
      /// initialize dummy fibers
      // Element fibers
      eleFibers_[0](0) = 0.858753861115007;
      eleFibers_[0](1) = 0.449823451060242;
      eleFibers_[0](2) = 0.245358246032859;

      eleFibers_[1](0) = 0.103448275862069;
      eleFibers_[1](1) = 0.137931034482759;
      eleFibers_[1](2) = 0.068965517241379;
      for (std::size_t i = 0; i < 2; ++i)
      {
        eleTensors_[i] = Core::LinAlg::self_dyadic(eleFibers_[i]);
      }

      setup_anisotropy_extension();
    }

    void setup_anisotropy_extension()
    {
      int fiber_id = std::get<0>(GetParam());
      auto strategy = std::make_shared<Mat::Elastic::StructuralTensorStrategyStandard>(nullptr);
      anisotropyExtension_ = std::make_unique<Mat::Elastic::CoupAnisoExpoAnisotropyExtension>(
          1, 0.0, false, strategy, fiber_id);
      anisotropyExtension_->register_needed_tensors(
          Mat::FiberAnisotropyExtension<1>::FIBER_VECTORS |
          Mat::FiberAnisotropyExtension<1>::STRUCTURAL_TENSOR);
      anisotropy_.register_anisotropy_extension(*anisotropyExtension_);
      anisotropy_.set_number_of_gauss_points(2);

      // Setup element fibers
      anisotropy_.set_element_fibers(eleFibers_);
    }

    [[nodiscard]] int get_gauss_point() const { return std::get<1>(GetParam()); }

    [[nodiscard]] int get_fiber_id() const { return std::get<0>(GetParam()); }

    Mat::Anisotropy anisotropy_;
    std::unique_ptr<Mat::Elastic::CoupAnisoExpoAnisotropyExtension> anisotropyExtension_;

    std::vector<Core::LinAlg::Tensor<double, 3>> eleFibers_;
    std::vector<Core::LinAlg::SymmetricTensor<double, 3, 3>> eleTensors_;
  };

  TEST_P(CoupAnisoExpoAnisotropyExtensionElementFiberTest, GetScalarProduct)
  {
    EXPECT_NEAR(anisotropyExtension_->get_scalar_product(get_gauss_point()), 1.0, 1e-10);
  }

  TEST_P(CoupAnisoExpoAnisotropyExtensionElementFiberTest, get_fiber)
  {
    FOUR_C_EXPECT_NEAR(anisotropyExtension_->get_fiber(get_gauss_point()),
        eleFibers_.at(get_fiber_id() - 1), 1e-10);
  }

  TEST_P(CoupAnisoExpoAnisotropyExtensionElementFiberTest, get_structural_tensor)
  {
    FOUR_C_EXPECT_NEAR(anisotropyExtension_->get_structural_tensor(get_gauss_point()),
        eleTensors_.at(get_fiber_id() - 1), 1e-10);
  }

  INSTANTIATE_TEST_SUITE_P(GaussPoints, CoupAnisoExpoAnisotropyExtensionElementFiberTest,
      ::testing::Combine(::testing::Values(1, 2), ::testing::Values(0, 1)));
}  // namespace

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_mat_anisotropy.hpp"
#include "4C_mat_elast_coupanisoexpo.hpp"
#include "4C_mat_elast_coupanisoexposhear.hpp"
#include "4C_unittest_utils_assertions_test.hpp"

namespace
{
  using namespace FourC;

  void setup_single_structural_tensor(const Core::LinAlg::Tensor<double, 3>& fiber1,
      const Core::LinAlg::Tensor<double, 3>& fiber2,
      Core::LinAlg::SymmetricTensor<double, 3, 3>& structuralTensor)
  {
    Core::LinAlg::Tensor<double, 3, 3> fiber1fiber2T = Core::LinAlg::dyadic(fiber1, fiber2);

    structuralTensor =
        0.5 * Core::LinAlg::assume_symmetry(fiber1fiber2T + Core::LinAlg::transpose(fiber1fiber2T));
  }

  class CoupAnisoExpoShearElementFibersTest
      : public ::testing::TestWithParam<std::tuple<std::array<int, 2>, int>>
  {
   public:
    CoupAnisoExpoShearElementFibersTest()
        : anisotropy_(), eleFibers_(3), eleTensors_(), eleScalarProducts_(0.0)
    {
      // setup fibers fibers
      eleFibers_[0](0) = 0.858753861115007;
      eleFibers_[0](1) = 0.449823451060242;
      eleFibers_[0](2) = 0.245358246032859;

      eleFibers_[1](0) = 0.103448275862069;
      eleFibers_[1](1) = 0.137931034482759;
      eleFibers_[1](2) = 0.068965517241379;

      eleFibers_[2](0) = 0.872502871778232;
      eleFibers_[2](1) = 0.134231211042805;
      eleFibers_[2](2) = 0.469809238649817;

      // setup structural tensor
      setup_single_structural_tensor(
          eleFibers_[get_fiber_ids()[0]], eleFibers_[get_fiber_ids()[1]], eleTensors_);

      // setup scalar product
      eleScalarProducts_ =
          Core::LinAlg::dot(eleFibers_[get_fiber_ids()[0]], eleFibers_[get_fiber_ids()[1]]);

      setup_anisotropy_extension(get_fiber_ids());
    }

    void setup_anisotropy_extension(std::array<int, 2> fiber_ids)
    {
      anisotropyExtension_ =
          std::make_unique<Mat::Elastic::CoupAnisoExpoShearAnisotropyExtension>(1, fiber_ids);

      anisotropy_.register_anisotropy_extension(*anisotropyExtension_);

      anisotropy_.set_number_of_gauss_points(2);

      // Setup element fibers
      anisotropy_.set_element_fibers(eleFibers_);
    }

    [[nodiscard]] int get_gauss_point() const { return std::get<1>(GetParam()); }

    [[nodiscard]] std::array<int, 2> get_fiber_ids() const { return std::get<0>(GetParam()); }

    Mat::Anisotropy anisotropy_;
    std::unique_ptr<Mat::Elastic::CoupAnisoExpoShearAnisotropyExtension> anisotropyExtension_;

    std::vector<Core::LinAlg::Tensor<double, 3>> eleFibers_;
    Core::LinAlg::SymmetricTensor<double, 3, 3> eleTensors_;
    double eleScalarProducts_;
  };

  TEST_P(CoupAnisoExpoShearElementFibersTest, GetScalarProduct)
  {
    EXPECT_NEAR(
        anisotropyExtension_->get_scalar_product(get_gauss_point()), eleScalarProducts_, 1e-10);
  }

  TEST_P(CoupAnisoExpoShearElementFibersTest, get_structural_tensor)
  {
    FOUR_C_EXPECT_NEAR(
        anisotropyExtension_->get_structural_tensor(get_gauss_point()), eleTensors_, 1e-10);
  }

  INSTANTIATE_TEST_SUITE_P(GaussPoints, CoupAnisoExpoShearElementFibersTest,
      ::testing::Combine(::testing::Values(std::array<int, 2>({0, 1}), std::array<int, 2>({0, 2}),
                             std::array<int, 2>({1, 2})),
          ::testing::Values(0, 1)));
}  // namespace

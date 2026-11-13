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
    Core::LinAlg::Matrix<3, 3> fiber1fiber2T(Core::LinAlg::Initialization::uninitialized);

    Core::LinAlg::Tensor<double, 3, 3> f1f2 = Core::LinAlg::dyadic(fiber1, fiber2);

    structuralTensor = 0.5 * Core::LinAlg::assume_symmetry(f1f2 + Core::LinAlg::transpose(f1f2));
  }

  class CoupAnisoExpoShearGaussPointFibersTest
      : public ::testing::TestWithParam<std::tuple<std::array<int, 2>, int>>
  {
   public:
    CoupAnisoExpoShearGaussPointFibersTest()
        : anisotropy_(),
          gpFibers_(2, std::vector<Core::LinAlg::Tensor<double, 3>>(3)),
          gpTensors_(2),
          gpScalarProducts_(2)
    {
      /// initialize dummy fibers
      // gp 0
      gpFibers_[0][0](0) = 0.469809238649817;
      gpFibers_[0][0](1) = 0.872502871778232;
      gpFibers_[0][0](2) = 0.134231211042805;

      gpFibers_[0][1](0) = 0.071428571428571;
      gpFibers_[0][1](1) = 0.142857142857143;
      gpFibers_[0][1](2) = 0.214285714285714;

      gpFibers_[0][2](0) = 0.142857142857143;
      gpFibers_[0][2](1) = 0.214285714285714;
      gpFibers_[0][2](2) = 0.071428571428571;

      // gp 1
      gpFibers_[1][0](0) = 0.245358246032859;
      gpFibers_[1][0](1) = 0.858753861115007;
      gpFibers_[1][0](2) = 0.449823451060242;

      gpFibers_[1][1](0) = 0.068965517241379;
      gpFibers_[1][1](1) = 0.103448275862069;
      gpFibers_[1][1](2) = 0.137931034482759;

      gpFibers_[1][2](0) = 0.137931034482759;
      gpFibers_[1][2](1) = 0.068965517241379;
      gpFibers_[1][2](2) = 0.103448275862069;

      for (std::size_t gp = 0; gp < 2; ++gp)
      {
        // setup structural tensor
        setup_single_structural_tensor(
            gpFibers_[gp][get_fiber_ids()[0]], gpFibers_[gp][get_fiber_ids()[1]], gpTensors_[gp]);

        // setup scalar product
        gpScalarProducts_[gp] =
            gpFibers_[gp][get_fiber_ids()[0]] * gpFibers_[gp][get_fiber_ids()[1]];
      }

      setup_anisotropy_extension(get_fiber_ids());
    }

    void setup_anisotropy_extension(std::array<int, 2> fiber_ids)
    {
      anisotropyExtension_ =
          std::make_unique<Mat::Elastic::CoupAnisoExpoShearAnisotropyExtension>(3, fiber_ids);

      anisotropy_.register_anisotropy_extension(*anisotropyExtension_);

      anisotropy_.set_number_of_gauss_points(2);

      // Setup element fibers
      anisotropy_.set_gauss_point_fibers(gpFibers_);
    }

    [[nodiscard]] int get_gauss_point() const { return std::get<1>(GetParam()); }

    [[nodiscard]] std::array<int, 2> get_fiber_ids() const { return std::get<0>(GetParam()); }

    Mat::Anisotropy anisotropy_;
    std::unique_ptr<Mat::Elastic::CoupAnisoExpoShearAnisotropyExtension> anisotropyExtension_;

    std::vector<std::vector<Core::LinAlg::Tensor<double, 3>>> gpFibers_;
    std::vector<Core::LinAlg::SymmetricTensor<double, 3, 3>> gpTensors_;
    std::vector<double> gpScalarProducts_;
  };

  TEST_P(CoupAnisoExpoShearGaussPointFibersTest, GetScalarProduct)
  {
    EXPECT_NEAR(anisotropyExtension_->get_scalar_product(get_gauss_point()),
        gpScalarProducts_[get_gauss_point()], 1e-10);
  }

  TEST_P(CoupAnisoExpoShearGaussPointFibersTest, get_structural_tensor)
  {
    FOUR_C_EXPECT_NEAR(anisotropyExtension_->get_structural_tensor(get_gauss_point()),
        gpTensors_[get_gauss_point()], 1e-10);
  }

  INSTANTIATE_TEST_SUITE_P(GaussPoints, CoupAnisoExpoShearGaussPointFibersTest,
      ::testing::Combine(::testing::Values(std::array<int, 2>({0, 1}), std::array<int, 2>({0, 2}),
                             std::array<int, 2>({1, 2})),
          ::testing::Values(0, 1)));
}  // namespace

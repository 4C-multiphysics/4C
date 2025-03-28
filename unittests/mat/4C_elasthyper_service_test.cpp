// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_fixedsizematrix_tensor_products.hpp"
#include "4C_mat_elast_coupanisoexpo.hpp"
#include "4C_mat_elast_isoneohooke.hpp"
#include "4C_mat_elasthyper_service.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_material_parameter_base.hpp"
#include "4C_unittest_utils_assertions_test.hpp"

namespace
{
  using namespace FourC;

  class ElastHyperServiceTest : public ::testing::Test
  {
   protected:
    void SetUp() override
    {
      prinv_(0) = 3.1;
      prinv_(1) = 3.2;
      prinv_(2) = 1.05;
    }

    Core::LinAlg::Matrix<3, 1> prinv_;
  };

  TEST_F(ElastHyperServiceTest, TestCalculateGammaDelta)
  {
    // required inputs to test the method
    // first derivative of strain energy function w.r.t. principle invariants
    Core::LinAlg::Matrix<3, 1> dPI;
    dPI(0) = 0.536;
    dPI(1) = 0.681;
    dPI(2) = 3.981;

    // second derivative of strain energy function w.r.t. principle invariants
    Core::LinAlg::Matrix<6, 1> ddPII;
    ddPII(0) = 0.153;
    ddPII(1) = 0.861;
    ddPII(2) = 2.964;
    ddPII(3) = 1.549;
    ddPII(4) = 0.682;
    ddPII(5) = 3.584;

    // factors for stress calculation as in Holzapfel: Nonlinear Solid Mechanics, 2000:
    // p. 216 (6.32)
    Core::LinAlg::Matrix<3, 1> gamma(Core::LinAlg::Initialization::zero);
    // reference solutions
    Core::LinAlg::Matrix<3, 1> gamma_ref;
    gamma_ref(0) = 5.294200000000001;
    gamma_ref(1) = -1.362;
    gamma_ref(2) = 8.360099999999999;

    // factors for calculation of elasticity tensor as in Holzapfel: Nonlinear Solid Mechanics,
    // 2000: p. 261 (6.194)
    Core::LinAlg::Matrix<8, 1> delta(Core::LinAlg::Initialization::zero);
    // reference solutions
    Core::LinAlg::Matrix<8, 1> delta_ref;
    delta_ref(0) = 125.31604;
    delta_ref(1) = -25.0124;
    delta_ref(2) = 23.03238;
    delta_ref(3) = 3.444;
    delta_ref(4) = -6.5058;
    delta_ref(5) = 29.79144;
    delta_ref(6) = -16.7202;
    delta_ref(7) = -2.724;

    Mat::calculate_gamma_delta(gamma, delta, prinv_, dPI, ddPII);

    FOUR_C_EXPECT_NEAR(gamma, gamma_ref, 1.0e-10);
    FOUR_C_EXPECT_NEAR(delta, delta_ref, 1.0e-10);
  }

  TEST_F(ElastHyperServiceTest, TestEvaluateRightCauchyGreenStrainLikeVoigt)
  {
    // Green-Lagrange tensor in strain-like voigt notation as input
    Core::LinAlg::Matrix<6, 1> E_VoigtStrain;
    E_VoigtStrain(0, 0) = 0.1;
    E_VoigtStrain(1, 0) = 0.2;
    E_VoigtStrain(2, 0) = 0.3;
    E_VoigtStrain(3, 0) = 0.01;
    E_VoigtStrain(4, 0) = 0.02;
    E_VoigtStrain(5, 0) = 0.03;

    Core::LinAlg::Matrix<6, 1> C_VoigtStrain(Core::LinAlg::Initialization::zero);
    // set up reference solution for Cauchy-Green tensor in strain-like voigt notation
    Core::LinAlg::Matrix<6, 1> C_VoigtStrain_ref(Core::LinAlg::Initialization::zero);
    C_VoigtStrain_ref(0) = 1.2;
    C_VoigtStrain_ref(1) = 1.4;
    C_VoigtStrain_ref(2) = 1.6;
    C_VoigtStrain_ref(3) = 0.02;
    C_VoigtStrain_ref(4) = 0.04;
    C_VoigtStrain_ref(5) = 0.06;

    Mat::evaluate_right_cauchy_green_strain_like_voigt(E_VoigtStrain, C_VoigtStrain);

    FOUR_C_EXPECT_NEAR(C_VoigtStrain, C_VoigtStrain_ref, 1.0e-10);
  }

  TEST_F(ElastHyperServiceTest, TestEvaluateInvariantDerivatives)
  {
    // Create summands to test the elast hyper functions
    // Currently, only an isotropic summand is supported. Anisotropic summands are not easy to be
    // added here as the setup of the anisotropy is not trivial.

    // Create parameter of IsoNeoHooke material
    Core::IO::InputParameterContainer iso_neo_hooke_data;
    iso_neo_hooke_data.add("MUE", 1.3);

    auto iso_neo_hooke_params =
        Mat::make_parameter(1, Core::Materials::MaterialType::mes_isoneohooke, iso_neo_hooke_data);

    // Create summand vector
    std::vector<std::shared_ptr<Mat::Elastic::Summand>> potsum(0);
    potsum.emplace_back(std::make_shared<Mat::Elastic::IsoNeoHooke>(
        dynamic_cast<Mat::Elastic::PAR::IsoNeoHooke*>(iso_neo_hooke_params.get())));

    // Read summand properties
    Mat::SummandProperties properties;
    Mat::elast_hyper_properties(potsum, properties);

    // first derivative of strain energy function w.r.t. principle invariants
    Core::LinAlg::Matrix<3, 1> dPI(Core::LinAlg::Initialization::zero);
    // reference solutions
    Core::LinAlg::Matrix<3, 1> dPI_ref;
    dPI_ref(0) = 0.639514;
    dPI_ref(1) = 0.0;
    dPI_ref(2) = -0.629363;

    // second derivative of strain energy function w.r.t. principle invariants
    Core::LinAlg::Matrix<6, 1> ddPII(Core::LinAlg::Initialization::zero);
    // reference solutions
    Core::LinAlg::Matrix<6, 1> ddPII_ref;
    ddPII_ref(0) = 0.0;
    ddPII_ref(1) = 0.0;
    ddPII_ref(2) = 0.799191;
    ddPII_ref(3) = 0.0;
    ddPII_ref(4) = -0.20302;
    ddPII_ref(5) = 0.0;

    // Compute derivatives of the strain energy function w.r.t. the principal invariants of the
    // strain energy function
    Mat::elast_hyper_evaluate_invariant_derivatives(prinv_, dPI, ddPII, potsum, properties, 0, 0);

    FOUR_C_EXPECT_NEAR(dPI, dPI_ref, 1.0e-4);
    FOUR_C_EXPECT_NEAR(ddPII, ddPII_ref, 1.0e-4);
  }
}  // namespace

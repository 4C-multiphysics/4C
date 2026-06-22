// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_mat_elast_coupneohooke.hpp"
#include "4C_mat_elast_isoneohooke.hpp"
#include "4C_mat_elasthyper_service.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_mat_multiplicative_split_defgrad_elasthyper_service.hpp"
#include "4C_unittest_utils_assertions_test.hpp"

namespace
{
  using namespace FourC;

  class MultiplicativeSplitDefgradElastHyperServiceTest : public ::testing::Test
  {
   protected:
    void SetUp() override
    {
      FM_(0, 0) = 1.1;
      FM_(1, 1) = 1.2;
      FM_(2, 2) = 1.3;
      FM_(0, 1) = FM_(1, 0) = 0.01;
      FM_(1, 2) = FM_(2, 1) = 0.02;
      FM_(0, 2) = FM_(2, 0) = 0.03;

      iFinM_(0, 0) = 1.04;
      iFinM_(1, 1) = 1.03;
      iFinM_(2, 2) = 1.02;
      iFinM_(0, 1) = iFinM_(1, 0) = 0.003;
      iFinM_(1, 2) = iFinM_(2, 1) = 0.001;
      iFinM_(0, 2) = iFinM_(2, 0) = 0.005;

      CM_.multiply_tn(FM_, FM_);
      iCinM_.multiply_nt(iFinM_, iFinM_);
    }

    Core::LinAlg::Matrix<3, 3> FM_;
    Core::LinAlg::Matrix<3, 3> iFinM_;

    Core::LinAlg::Matrix<3, 3> CM_;
    Core::LinAlg::Matrix<3, 3> iCinM_;
  };

  TEST_F(MultiplicativeSplitDefgradElastHyperServiceTest, TestEvaluateCe)
  {
    Core::LinAlg::Matrix<3, 3> CeM_target(Core::LinAlg::Initialization::uninitialized);
    CeM_target(0, 0) = 1.3107725000000006;
    CeM_target(1, 1) = 1.5284889394999996;
    CeM_target(2, 2) = 1.7604995235000003;
    CeM_target(0, 1) = CeM_target(1, 0) = 0.03385382080000001;
    CeM_target(0, 2) = CeM_target(2, 0) = 0.091697784;
    CeM_target(1, 2) = CeM_target(2, 1) = 0.0564151401;

    Core::LinAlg::Matrix<3, 3> CeM(Core::LinAlg::Initialization::uninitialized);
    Mat::evaluate_ce(FM_, iFinM_, CeM);

    FOUR_C_EXPECT_NEAR(CeM, CeM_target, 1.0e-10);
  }

  TEST_F(MultiplicativeSplitDefgradElastHyperServiceTest, TestEvaluateiCinCiCin)
  {
    Core::LinAlg::Matrix<3, 3> iCinCiCinM_target(Core::LinAlg::Initialization::uninitialized);
    iCinCiCinM_target(0, 0) = 1.418955902138138;
    iCinCiCinM_target(1, 1) = 1.6219134553554275;
    iCinCiCinM_target(2, 2) = 1.832708744871652;
    iCinCiCinM_target(0, 1) = iCinCiCinM_target(1, 0) = 0.045473409425074995;
    iCinCiCinM_target(0, 2) = iCinCiCinM_target(2, 0) = 0.113283079933819;
    iCinCiCinM_target(1, 2) = iCinCiCinM_target(2, 1) = 0.0631150197598975;

    Core::LinAlg::Matrix<3, 3> iCinCiCinM(Core::LinAlg::Initialization::uninitialized);
    Mat::evaluatei_cin_ci_cin(CM_, iCinM_, iCinCiCinM);

    FOUR_C_EXPECT_NEAR(iCinCiCinM, iCinCiCinM_target, 1.0e-10);
  }


  TEST_F(MultiplicativeSplitDefgradElastHyperServiceTest, TestElastHyperEvaluateElasticPart)
  {
    Core::LinAlg::Matrix<6, 1> S_stress;
    Core::LinAlg::Matrix<6, 6> cmat;

    // Create parameter of IsoNeoHooke material
    Core::IO::InputParameterContainer iso_neo_hooke_data;
    iso_neo_hooke_data.add("MUE", Core::IO::InputField<double>(1.3));

    auto iso_neo_hooke_params =
        Mat::make_parameter(1, Core::Materials::MaterialType::mes_isoneohooke, iso_neo_hooke_data);

    // Create summand vector
    std::vector<std::shared_ptr<Mat::Elastic::Summand>> potsum;
    potsum.emplace_back(std::make_shared<Mat::Elastic::IsoNeoHooke>(
        dynamic_cast<Mat::Elastic::PAR::IsoNeoHooke*>(iso_neo_hooke_params.get())));

    // Read summand properties
    Mat::SummandProperties properties;
    Mat::elast_hyper_properties(potsum, properties);

    // Evaluate method to test
    Mat::elast_hyper_evaluate_elastic_part(FM_, iFinM_, S_stress, cmat, potsum, properties, 0, 0);

    // Build matrices with the correct solution
    Core::LinAlg::Matrix<6, 1> S_stress_target;
    Core::LinAlg::Matrix<6, 6> cmat_target;

    S_stress_target(0) = -0.16087311035295149;
    S_stress_target(1) = -0.0041652481745947378;
    S_stress_target(2) = 0.11178774018437565;
    S_stress_target(3) = 0.021510980954630672;
    S_stress_target(4) = 0.028193006189638534;
    S_stress_target(5) = 0.054703856893060218;
    cmat_target(0, 0) = 1.3769715241022054;
    cmat_target(0, 1) = -0.42612503282604719;
    cmat_target(0, 2) = -0.42455257184204381;
    cmat_target(0, 3) = -0.031079302592187129;
    cmat_target(0, 4) = -0.0020029032555127484;
    cmat_target(0, 5) = -0.084619011701144234;
    cmat_target(1, 0) = -0.42612503282604719;
    cmat_target(1, 1) = 0.84893700719517029;
    cmat_target(1, 2) = -0.40977461472058613;
    cmat_target(1, 3) = -0.025011243966307363;
    cmat_target(1, 4) = -0.037612994171778887;
    cmat_target(1, 5) = -0.0035293902859759667;
    cmat_target(2, 0) = -0.42455257184204381;
    cmat_target(2, 1) = -0.40977461472058613;
    cmat_target(2, 2) = 0.52732888530268807;
    cmat_target(2, 3) = 0.0006656868830938736;
    cmat_target(2, 4) = -0.030557790648953649;
    cmat_target(2, 5) = -0.055336798871663345;
    cmat_target(3, 0) = -0.031079302592187129;
    cmat_target(3, 1) = -0.025011243966307363;
    cmat_target(3, 2) = 0.0006656868830938736;
    cmat_target(3, 3) = 0.75548157994492326;
    cmat_target(3, 4) = -0.031268379573823023;
    cmat_target(3, 5) = -0.020742640930106387;
    cmat_target(4, 0) = -0.0020029032555127484;
    cmat_target(4, 1) = -0.037612994171778887;
    cmat_target(4, 2) = -0.030557790648953649;
    cmat_target(4, 3) = -0.031268379573823023;
    cmat_target(4, 4) = 0.54196060823048886;
    cmat_target(4, 5) = -0.0079082879496605533;
    cmat_target(5, 0) = -0.084619011701144234;
    cmat_target(5, 1) = -0.0035293902859759667;
    cmat_target(5, 2) = -0.055336798871663345;
    cmat_target(5, 3) = -0.020742640930106387;
    cmat_target(5, 4) = -0.0079082879496605533;
    cmat_target(5, 5) = 0.64761600029220823;

    FOUR_C_EXPECT_NEAR(S_stress, S_stress_target, 1.0e-9);
    FOUR_C_EXPECT_NEAR(cmat, cmat_target, 1.0e-9);
  }

  TEST_F(MultiplicativeSplitDefgradElastHyperServiceTest,
      TestEvaluateThermalQuantitiesStressAndStiffness)
  {
    //**************************************************
    double delta_temperature = 100.0000000000000000;
    //**************************************************
    Core::LinAlg::Matrix<3, 3> iFinM{Core::LinAlg::Initialization::zero};
    iFinM(0, 0) = 1.0026809779767947;
    iFinM(0, 1) = 0.2005361955953590;
    iFinM(0, 2) = 0.0000000000000000;
    iFinM(1, 0) = 0.3008042933930384;
    iFinM(1, 1) = 1.3034852713698331;
    iFinM(1, 2) = 0.0000000000000000;
    iFinM(2, 0) = 0.0000000000000000;
    iFinM(2, 1) = 0.0000000000000000;
    iFinM(2, 2) = 0.8021447823814358;
    //**************************************************
    Core::LinAlg::Matrix<3, 3> CTM_ref_{Core::LinAlg::Initialization::zero};
    CTM_ref_(0, 0) = 21.0000000000000000;
    CTM_ref_(0, 1) = 0.0000000000000000;
    CTM_ref_(0, 2) = 0.0000000000000000;
    CTM_ref_(1, 0) = 0.0000000000000000;
    CTM_ref_(1, 1) = 21.0000000000000000;
    CTM_ref_(1, 2) = 0.0000000000000000;
    CTM_ref_(2, 0) = 0.0000000000000000;
    CTM_ref_(2, 1) = 0.0000000000000000;
    CTM_ref_(2, 2) = 21.0000000000000000;
    //**************************************************
    Core::LinAlg::Matrix<3, 3> dCTM_dT_ref_{Core::LinAlg::Initialization::zero};
    dCTM_dT_ref_(0, 0) = 0.2000000000000000;
    dCTM_dT_ref_(0, 1) = 0.0000000000000000;
    dCTM_dT_ref_(0, 2) = 0.0000000000000000;
    dCTM_dT_ref_(1, 0) = 0.0000000000000000;
    dCTM_dT_ref_(1, 1) = 0.2000000000000000;
    dCTM_dT_ref_(1, 2) = 0.0000000000000000;
    dCTM_dT_ref_(2, 0) = 0.0000000000000000;
    dCTM_dT_ref_(2, 1) = 0.0000000000000000;
    dCTM_dT_ref_(2, 2) = 0.2000000000000000;
    //**************************************************
    Core::LinAlg::Matrix<3, 3> S_ref{Core::LinAlg::Initialization::zero};
    S_ref(0, 0) = -60.3191058810404357;
    S_ref(0, 1) = -32.4795185513294626;
    S_ref(0, 2) = 0.0000000000000000;
    S_ref(1, 0) = -32.4795185513294626;
    S_ref(1, 1) = -103.2384696810115088;
    S_ref(1, 2) = 0.0000000000000000;
    S_ref(2, 0) = 0.0000000000000000;
    S_ref(2, 1) = 0.0000000000000000;
    S_ref(2, 2) = -37.1194497729479664;
    //**************************************************
    Core::LinAlg::Matrix<3, 3> pS_pT_ref_{Core::LinAlg::Initialization::zero};
    pS_pT_ref_(0, 0) = -0.0000941798851085;
    pS_pT_ref_(0, 1) = -0.0000507122458277;
    pS_pT_ref_(0, 2) = -0.0000000000000000;
    pS_pT_ref_(1, 0) = -0.0000507122458277;
    pS_pT_ref_(1, 1) = -0.0001611924956665;
    pS_pT_ref_(1, 2) = -0.0000000000000000;
    pS_pT_ref_(2, 0) = -0.0000000000000000;
    pS_pT_ref_(2, 1) = -0.0000000000000000;
    pS_pT_ref_(2, 2) = -0.0000579568523745;

    // set thermal info
    const double thermal_expansion_fac = 0.1;

    Core::LinAlg::Matrix<3, 3> iCinM{Core::LinAlg::Initialization::zero};
    iCinM.multiply_nt(1.0, iFinM, iFinM, 0.0);
    Core::LinAlg::Matrix<6, 1> iCinV{Core::LinAlg::Initialization::zero};
    Core::LinAlg::Voigt::Stresses::matrix_to_vector(iCinM, iCinV);

    // Create summand vector
    std::vector<std::shared_ptr<Mat::Elastic::Summand>> potsum;
    Core::IO::InputParameterContainer elast_pot_coup_neo_hooke_data;
    elast_pot_coup_neo_hooke_data.add("YOUNG", 1.5e2);
    elast_pot_coup_neo_hooke_data.add("NUE", 0.3);
    auto coup_neo_hooke_params = Mat::make_parameter(
        200, Core::Materials::MaterialType::mes_coupneohooke, elast_pot_coup_neo_hooke_data);
    potsum.emplace_back(std::make_shared<Mat::Elastic::CoupNeoHooke>(
        dynamic_cast<Mat::Elastic::PAR::CoupNeoHooke*>(coup_neo_hooke_params.get())));


    // evaluate thermal quantities
    Mat::ThermalQuantities thermal_quantities = Mat::evaluate_thermal_quantities(
        delta_temperature, thermal_expansion_fac, iFinM, 0, 0, potsum);

    // evaluate thermal stress factors
    Mat::StressFactors thermal_stress_factors;
    Mat::calculate_gamma_delta(thermal_stress_factors.gamma, thermal_stress_factors.delta,
        thermal_quantities.prinv, thermal_quantities.dPI, thermal_quantities.ddPII);

    // evaluate thermal contribution to second Piola-Kirchhoff stress
    Core::LinAlg::Matrix<6, 1> S_V{Core::LinAlg::Initialization::zero};
    Mat::add_thermal_stress_contribution(
        S_V, thermal_quantities, thermal_stress_factors, iCinV, 1.0 / iFinM.determinant());
    Core::LinAlg::Matrix<3, 3> S_M{Core::LinAlg::Initialization::zero};
    Core::LinAlg::Voigt::Stresses::vector_to_matrix(S_V, S_M);
    // evaluate partial derivative of 2nd Piola-Kirchhoff stress wrt temperature
    Core::LinAlg::Matrix<6, 1> pS_pT_V =
        Mat::compute_partial_d_stress_d_temperature(iFinM, thermal_quantities);
    Core::LinAlg::Matrix<3, 3> pS_pT_M{Core::LinAlg::Initialization::zero};
    Core::LinAlg::Voigt::Stresses::vector_to_matrix(pS_pT_V, pS_pT_M);


    // postprocessed data for assertions
    Core::LinAlg::Matrix<3, 3> CTM{Core::LinAlg::Initialization::zero};
    Core::LinAlg::Voigt::Stresses::vector_to_matrix(thermal_quantities.CTV, CTM);
    Core::LinAlg::Matrix<3, 3> dCTM_dT{Core::LinAlg::Initialization::zero};
    Core::LinAlg::Voigt::Strains::vector_to_matrix(thermal_quantities.dCTdTV, dCTM_dT);

    FOUR_C_EXPECT_NEAR(CTM, CTM_ref_, 1.0e-10);
    FOUR_C_EXPECT_NEAR(dCTM_dT, dCTM_dT_ref_, 1.0e-10);
    FOUR_C_EXPECT_NEAR(S_M, S_ref, 1.0e-10);
    FOUR_C_EXPECT_NEAR(pS_pT_M, pS_pT_ref_, 1.0e-10);
  }
}  // namespace

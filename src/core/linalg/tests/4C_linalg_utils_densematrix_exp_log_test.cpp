// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_linalg_utils_densematrix_exp_log.hpp"

#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_unittest_utils_assertions_test.hpp"

#include <complex>

FOUR_C_NAMESPACE_OPEN


namespace
{
  /*
   * \note The values for the matrix used in tests below are generated with python/numpy
   */
  TEST(LinalgDenseMatrixExpLogTest, 2x2MatrixExp)
  {
    Core::LinAlg::Matrix<2, 2> A(Core::LinAlg::Initialization::zero);
    A(0, 0) = 0.1320698936;
    A(0, 1) = 0.7174840819;
    A(1, 0) = 0.9846543275;
    A(1, 1) = 0.0134355791;

    Core::LinAlg::Matrix<2, 2> exp_A_ref(Core::LinAlg::Initialization::zero);
    exp_A_ref(0, 0) = 1.5519954316;
    exp_A_ref(0, 1) = 0.8662338814;
    exp_A_ref(1, 0) = 1.1887942346;
    exp_A_ref(1, 1) = 1.4087656857;

    Core::LinAlg::Matrix<2, 2> exp_A = Core::LinAlg::matrix_exp(A);

    FOUR_C_EXPECT_NEAR(exp_A, exp_A_ref, 1.0e-9);
  }

  TEST(LinalgDenseMatrixExpLogTest, 2x2MatrixLog)
  {
    Core::LinAlg::Matrix<2, 2> A(Core::LinAlg::Initialization::zero);
    A(0, 0) = 0.8710528572;
    A(0, 1) = 0.4665807292;
    A(1, 0) = 0.0007588202;
    A(1, 1) = 0.7447870648;

    Core::LinAlg::Matrix<2, 2> log_A_ref(Core::LinAlg::Initialization::zero);
    log_A_ref(0, 0) = -0.1383113276;
    log_A_ref(0, 1) = 0.5787936215;
    log_A_ref(1, 0) = 0.0009413169;
    log_A_ref(1, 1) = -0.2949441044;

    Core::LinAlg::Matrix<2, 2> log_A = Core::LinAlg::matrix_log(A);

    FOUR_C_EXPECT_NEAR(log_A, log_A_ref, 1.0e-9);
  }

  TEST(LinalgDenseMatrixExpLogTest, 3x3MatrixExp)
  {
    Core::LinAlg::Matrix<3, 3> A(Core::LinAlg::Initialization::zero);
    A(0, 0) = 0.4908570193;
    A(0, 1) = 0.0227448063;
    A(0, 2) = 0.8823601080;
    A(1, 0) = 0.3080134014;
    A(1, 1) = 0.8620162322;
    A(1, 2) = 0.1386338379;
    A(2, 0) = 0.5582587798;
    A(2, 1) = 0.3123498741;
    A(2, 2) = 0.9434957224;

    Core::LinAlg::Matrix<3, 3> exp_A_ref(Core::LinAlg::Initialization::zero);
    exp_A_ref(0, 0) = 2.1642950359;
    exp_A_ref(0, 1) = 0.3611066893;
    exp_A_ref(0, 2) = 2.0038269622;
    exp_A_ref(1, 0) = 0.7573840134;
    exp_A_ref(1, 1) = 2.4634686195;
    exp_A_ref(1, 2) = 0.6797727284;
    exp_A_ref(2, 0) = 1.3743290645;
    exp_A_ref(2, 1) = 0.8535273999;
    exp_A_ref(2, 2) = 3.2314443053;

    Core::LinAlg::Matrix<3, 3> exp_A = Core::LinAlg::matrix_exp(A);

    FOUR_C_EXPECT_NEAR(exp_A, exp_A_ref, 1.0e-9);
  }
  TEST(LinalgDenseMatrixExpLogTest, MatrixExp1stDeriv)
  {
    Core::LinAlg::Matrix<3, 3> A(Core::LinAlg::Initialization::zero);
    A(0, 0) = 0.0825983659;
    A(0, 1) = 0.4025712945;
    A(0, 2) = 0.8812204459;
    A(1, 0) = 0.2195202517;
    A(1, 1) = 0.3325625152;
    A(1, 2) = 0.3090307594;
    A(2, 0) = 0.6577750730;
    A(2, 1) = 0.4929807721;
    A(2, 2) = 0.2853229892;

    Core::LinAlg::Matrix<9, 9> dexp_dA_ref(Core::LinAlg::Initialization::zero);
    dexp_dA_ref(0, 0) = 1.3754913371;
    dexp_dA_ref(0, 1) = 0.0304833776;
    dexp_dA_ref(0, 2) = 0.1363342698;
    dexp_dA_ref(0, 3) = 0.1921557120;
    dexp_dA_ref(0, 4) = 0.0770010890;
    dexp_dA_ref(0, 5) = 0.4591972517;
    dexp_dA_ref(0, 6) = 0.3658995170;
    dexp_dA_ref(0, 7) = 0.0542854547;
    dexp_dA_ref(0, 8) = 0.6106768758;
    dexp_dA_ref(1, 0) = 0.0304833776;
    dexp_dA_ref(1, 1) = 1.5285331987;
    dexp_dA_ref(1, 2) = 0.0482389806;
    dexp_dA_ref(1, 3) = 0.2010938640;
    dexp_dA_ref(1, 4) = 0.4306567768;
    dexp_dA_ref(1, 5) = 0.0349324549;
    dexp_dA_ref(1, 6) = 0.3828518059;
    dexp_dA_ref(1, 7) = 0.2762284172;
    dexp_dA_ref(1, 8) = 0.0421026685;
    dexp_dA_ref(2, 0) = 0.1363342698;
    dexp_dA_ref(2, 1) = 0.0482389806;
    dexp_dA_ref(2, 2) = 1.6823911395;
    dexp_dA_ref(2, 3) = 0.0562372279;
    dexp_dA_ref(2, 4) = 0.4395219110;
    dexp_dA_ref(2, 5) = 0.4918729857;
    dexp_dA_ref(2, 6) = 0.1173687055;
    dexp_dA_ref(2, 7) = 0.2818590743;
    dexp_dA_ref(2, 8) = 0.6542102844;
    dexp_dA_ref(3, 0) = 0.3658995170;
    dexp_dA_ref(3, 1) = 0.3828518059;
    dexp_dA_ref(3, 2) = 0.1173687055;
    dexp_dA_ref(3, 3) = 1.4499201522;
    dexp_dA_ref(3, 4) = 0.0659836642;
    dexp_dA_ref(3, 5) = 0.4114189775;
    dexp_dA_ref(3, 6) = 0.0575644245;
    dexp_dA_ref(3, 7) = 0.6398863904;
    dexp_dA_ref(3, 8) = 0.1026036294;
    dexp_dA_ref(4, 0) = 0.0542854547;
    dexp_dA_ref(4, 1) = 0.2762284172;
    dexp_dA_ref(4, 2) = 0.2818590743;
    dexp_dA_ref(4, 3) = 0.0222912411;
    dexp_dA_ref(4, 4) = 1.6051958053;
    dexp_dA_ref(4, 5) = 0.2051507417;
    dexp_dA_ref(4, 6) = 0.6398863904;
    dexp_dA_ref(4, 7) = 0.0307839662;
    dexp_dA_ref(4, 8) = 0.0749326364;
    dexp_dA_ref(5, 0) = 0.6106768758;
    dexp_dA_ref(5, 1) = 0.0421026685;
    dexp_dA_ref(5, 2) = 0.6542102844;
    dexp_dA_ref(5, 3) = 0.2639192170;
    dexp_dA_ref(5, 4) = 0.3904565212;
    dexp_dA_ref(5, 5) = 1.5237256921;
    dexp_dA_ref(5, 6) = 0.1026036294;
    dexp_dA_ref(5, 7) = 0.0749326364;
    dexp_dA_ref(5, 8) = 0.1816285568;
    dexp_dA_ref(6, 0) = 0.1921557120;
    dexp_dA_ref(6, 1) = 0.2010938640;
    dexp_dA_ref(6, 2) = 0.0562372279;
    dexp_dA_ref(6, 3) = 0.0161406654;
    dexp_dA_ref(6, 4) = 0.4811400221;
    dexp_dA_ref(6, 5) = 0.0407407355;
    dexp_dA_ref(6, 6) = 1.4499201522;
    dexp_dA_ref(6, 7) = 0.0222912411;
    dexp_dA_ref(6, 8) = 0.2639192170;
    dexp_dA_ref(7, 0) = 0.0770010890;
    dexp_dA_ref(7, 1) = 0.4306567768;
    dexp_dA_ref(7, 2) = 0.4395219110;
    dexp_dA_ref(7, 3) = 0.4811400221;
    dexp_dA_ref(7, 4) = 0.0755877602;
    dexp_dA_ref(7, 5) = 0.0880871026;
    dexp_dA_ref(7, 6) = 0.0659836642;
    dexp_dA_ref(7, 7) = 1.6051958053;
    dexp_dA_ref(7, 8) = 0.3904565212;
    dexp_dA_ref(8, 0) = 0.4591972517;
    dexp_dA_ref(8, 1) = 0.0349324549;
    dexp_dA_ref(8, 2) = 0.4918729857;
    dexp_dA_ref(8, 3) = 0.0407407355;
    dexp_dA_ref(8, 4) = 0.0880871026;
    dexp_dA_ref(8, 5) = 0.1023348159;
    dexp_dA_ref(8, 6) = 0.4114189775;
    dexp_dA_ref(8, 7) = 0.2051507417;
    dexp_dA_ref(8, 8) = 1.5237256921;
    Core::LinAlg::Matrix<9, 9> dexp_dA = Core::LinAlg::matrix_3x3_exp_1st_deriv(A);

    FOUR_C_EXPECT_NEAR(dexp_dA, dexp_dA_ref, 1.0e-9);
  }

  TEST(LinalgDenseMatrixExpLogTest, SymMatrix3x3Exp1stDeriv)
  {
    Core::LinAlg::Matrix<3, 3> A(Core::LinAlg::Initialization::zero);
    A(0, 0) = 0.8690903698;
    A(0, 1) = 0.1617986658;
    A(0, 2) = 0.2280178825;
    A(1, 0) = 0.1617986658;
    A(1, 1) = 0.8441051613;
    A(1, 2) = 0.4866432170;
    A(2, 0) = 0.2280178825;
    A(2, 1) = 0.4866432170;
    A(2, 2) = 0.1288095746;

    Core::LinAlg::Matrix<6, 6> dexp_dA_ref(Core::LinAlg::Initialization::zero);
    dexp_dA_ref(0, 0) = 2.4472876345;
    dexp_dA_ref(0, 1) = 0.0138990186;
    dexp_dA_ref(0, 2) = 0.0177623391;
    dexp_dA_ref(0, 3) = 0.2346451204;
    dexp_dA_ref(0, 4) = 0.0157716820;
    dexp_dA_ref(0, 5) = 0.2493333924;
    dexp_dA_ref(1, 0) = 0.0138990186;
    dexp_dA_ref(1, 1) = 2.5112905815;
    dexp_dA_ref(1, 2) = 0.0698865971;
    dexp_dA_ref(1, 3) = 0.2357000018;
    dexp_dA_ref(1, 4) = 0.4864009671;
    dexp_dA_ref(1, 5) = 0.0313889144;
    dexp_dA_ref(2, 0) = 0.0177623391;
    dexp_dA_ref(2, 1) = 0.0698865971;
    dexp_dA_ref(2, 2) = 1.2779285237;
    dexp_dA_ref(2, 3) = 0.0352514306;
    dexp_dA_ref(2, 4) = 0.3866647057;
    dexp_dA_ref(2, 5) = 0.2003128036;
    dexp_dA_ref(3, 0) = 0.2346451204;
    dexp_dA_ref(3, 1) = 0.2357000018;
    dexp_dA_ref(3, 2) = 0.0352514306;
    dexp_dA_ref(3, 3) = 1.2465976814;
    dexp_dA_ref(3, 4) = 0.1410159996;
    dexp_dA_ref(3, 5) = 0.2497413902;
    dexp_dA_ref(4, 0) = 0.0157716820;
    dexp_dA_ref(4, 1) = 0.4864009671;
    dexp_dA_ref(4, 2) = 0.3866647057;
    dexp_dA_ref(4, 3) = 0.1410159996;
    dexp_dA_ref(4, 4) = 0.9466709564;
    dexp_dA_ref(4, 5) = 0.1130414528;
    dexp_dA_ref(5, 0) = 0.2493333924;
    dexp_dA_ref(5, 1) = 0.0313889144;
    dexp_dA_ref(5, 2) = 0.2003128036;
    dexp_dA_ref(5, 3) = 0.2497413902;
    dexp_dA_ref(5, 4) = 0.1130414528;
    dexp_dA_ref(5, 5) = 0.9065591579;


    Core::LinAlg::Matrix<6, 6> dexp_dA = Core::LinAlg::sym_matrix_3x3_exp_1st_deriv(A);

    FOUR_C_EXPECT_NEAR(dexp_dA, dexp_dA_ref, 1.0e-9);
  }

  TEST(LinalgDenseMatrixExpLogTest, 3x3MatrixLog)
  {
    Core::LinAlg::Matrix<3, 3> A(Core::LinAlg::Initialization::zero);
    A(0, 0) = 0.4908570193;
    A(0, 1) = 0.0227448063;
    A(0, 2) = 0.8823601080;
    A(1, 0) = 0.3080134014;
    A(1, 1) = 0.8620162322;
    A(1, 2) = 0.1386338379;
    A(2, 0) = 0.5582587798;
    A(2, 1) = 0.3123498741;
    A(2, 2) = 0.9434957224;

    Core::LinAlg::Matrix<3, 3> log_A_ref(Core::LinAlg::Initialization::zero);
    log_A_ref(0, 0) = -2.3112341006;
    log_A_ref(0, 1) = -0.6036784103;
    log_A_ref(0, 2) = 2.6135476834;
    log_A_ref(1, 0) = 0.7868003639;
    log_A_ref(1, 1) = -0.0327999795;
    log_A_ref(1, 2) = -0.3807316790;
    log_A_ref(2, 0) = 1.4225476275;
    log_A_ref(2, 1) = 0.6125140275;
    log_A_ref(2, 2) = -1.0555537527;

    Core::LinAlg::Matrix<3, 3> log_A = Core::LinAlg::matrix_log(A);

    FOUR_C_EXPECT_NEAR(log_A, log_A_ref, 1.0e-9);
  }
  TEST(LinalgDenseMatrixExpLogTest, MatrixLog1stDeriv)
  {
    Core::LinAlg::Matrix<3, 3> A(Core::LinAlg::Initialization::zero);
    A(0, 0) = 0.9023044993;
    A(0, 1) = 0.5012335252;
    A(0, 2) = 0.9168173456;
    A(1, 0) = 0.5703668482;
    A(1, 1) = 0.4167101809;
    A(1, 2) = 0.0985904954;
    A(2, 0) = 0.1978501877;
    A(2, 1) = 0.2844070513;
    A(2, 2) = 0.8815705957;

    Core::LinAlg::Matrix<9, 9> dlog_dA_ref(Core::LinAlg::Initialization::zero);
    dlog_dA_ref(0, 0) = 1.9623864897;
    dlog_dA_ref(0, 1) = 0.6058626691;
    dlog_dA_ref(0, 2) = -0.0924008485;
    dlog_dA_ref(0, 3) = -1.4044032085;
    dlog_dA_ref(0, 4) = -0.0503297819;
    dlog_dA_ref(0, 5) = 0.0845579572;
    dlog_dA_ref(0, 6) = -0.6712728624;
    dlog_dA_ref(0, 7) = 1.1114356856;
    dlog_dA_ref(0, 8) = -1.2309211590;
    dlog_dA_ref(1, 0) = 0.6058626691;
    dlog_dA_ref(1, 1) = 4.2270072667;
    dlog_dA_ref(1, 2) = -0.3524298942;
    dlog_dA_ref(1, 3) = -2.2324686299;
    dlog_dA_ref(1, 4) = -0.7981058836;
    dlog_dA_ref(1, 5) = 0.4765952875;
    dlog_dA_ref(1, 6) = -1.0431180703;
    dlog_dA_ref(1, 7) = 1.5658630780;
    dlog_dA_ref(1, 8) = -0.4401174939;
    dlog_dA_ref(2, 0) = -0.0924008485;
    dlog_dA_ref(2, 1) = -0.3524298942;
    dlog_dA_ref(2, 2) = 1.1023372271;
    dlog_dA_ref(2, 3) = 0.1103782745;
    dlog_dA_ref(2, 4) = -0.3004186267;
    dlog_dA_ref(2, 5) = 0.0208904232;
    dlog_dA_ref(2, 6) = 0.3928224214;
    dlog_dA_ref(2, 7) = 0.4891199881;
    dlog_dA_ref(2, 8) = -0.7560007608;
    dlog_dA_ref(3, 0) = -0.6712728624;
    dlog_dA_ref(3, 1) = -1.0431180703;
    dlog_dA_ref(3, 2) = 0.3928224214;
    dlog_dA_ref(3, 3) = 2.8420772308;
    dlog_dA_ref(3, 4) = 0.2141475844;
    dlog_dA_ref(3, 5) = -0.5072404690;
    dlog_dA_ref(3, 6) = 0.2759509022;
    dlog_dA_ref(3, 7) = -1.9130192431;
    dlog_dA_ref(3, 8) = 0.5061517466;
    dlog_dA_ref(4, 0) = 1.1114356856;
    dlog_dA_ref(4, 1) = 1.5658630780;
    dlog_dA_ref(4, 2) = 0.4891199881;
    dlog_dA_ref(4, 3) = -1.0214713980;
    dlog_dA_ref(4, 4) = 1.9170944680;
    dlog_dA_ref(4, 5) = -0.8059376443;
    dlog_dA_ref(4, 6) = -1.9130192431;
    dlog_dA_ref(4, 7) = 0.7954129230;
    dlog_dA_ref(4, 8) = -0.8075330486;
    dlog_dA_ref(5, 0) = -1.2309211590;
    dlog_dA_ref(5, 1) = -0.4401174939;
    dlog_dA_ref(5, 2) = -0.7560007608;
    dlog_dA_ref(5, 3) = 0.9530463250;
    dlog_dA_ref(5, 4) = -0.4124281986;
    dlog_dA_ref(5, 5) = 1.3969978711;
    dlog_dA_ref(5, 6) = 0.5061517466;
    dlog_dA_ref(5, 7) = -0.8075330486;
    dlog_dA_ref(5, 8) = 0.9283890595;
    dlog_dA_ref(6, 0) = -1.4044032085;
    dlog_dA_ref(6, 1) = -2.2324686299;
    dlog_dA_ref(6, 2) = 0.1103782745;
    dlog_dA_ref(6, 3) = 1.3627688103;
    dlog_dA_ref(6, 4) = 0.1587414762;
    dlog_dA_ref(6, 5) = -0.1282738327;
    dlog_dA_ref(6, 6) = 2.8420772308;
    dlog_dA_ref(6, 7) = -1.0214713980;
    dlog_dA_ref(6, 8) = 0.9530463250;
    dlog_dA_ref(7, 0) = -0.0503297819;
    dlog_dA_ref(7, 1) = -0.7981058836;
    dlog_dA_ref(7, 2) = -0.3004186267;
    dlog_dA_ref(7, 3) = 0.1587414762;
    dlog_dA_ref(7, 4) = 0.1674533255;
    dlog_dA_ref(7, 5) = -0.0425634001;
    dlog_dA_ref(7, 6) = 0.2141475844;
    dlog_dA_ref(7, 7) = 1.9170944680;
    dlog_dA_ref(7, 8) = -0.4124281986;
    dlog_dA_ref(8, 0) = 0.0845579572;
    dlog_dA_ref(8, 1) = 0.4765952875;
    dlog_dA_ref(8, 2) = 0.0208904232;
    dlog_dA_ref(8, 3) = -0.1282738327;
    dlog_dA_ref(8, 4) = -0.0425634001;
    dlog_dA_ref(8, 5) = 0.0188774581;
    dlog_dA_ref(8, 6) = -0.5072404690;
    dlog_dA_ref(8, 7) = -0.8059376443;
    dlog_dA_ref(8, 8) = 1.3969978711;
    Core::LinAlg::Matrix<9, 9> dlog_dA = Core::LinAlg::matrix_3x3_log_1st_deriv(A);

    FOUR_C_EXPECT_NEAR(dlog_dA, dlog_dA_ref, 1.0e-9);
  }
}  // namespace

FOUR_C_NAMESPACE_CLOSE

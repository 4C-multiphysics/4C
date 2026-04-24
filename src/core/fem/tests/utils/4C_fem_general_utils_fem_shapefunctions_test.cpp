// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_fem_general_utils_fem_shapefunctions.hpp"

namespace
{
  using namespace FourC;

  TEST(ElementShapeFunctionsTest, TestLine5)
  {
    const double r = 0.123456789;
    Core::LinAlg::Matrix<5, 1> N(Core::LinAlg::Initialization::zero);
    Core::FE::shape_function_1d(N, r, Core::FE::CellType::line5);
    std::vector<double> N_ref = {0.016936287942307646, -0.021707073228640142, -0.12207542209790175,
        0.92472132914024041, 0.20212487824399381};
    for (std::size_t i = 0; i < N_ref.size(); ++i) EXPECT_NEAR(N(i), N_ref[i], 1e-10);

    Core::LinAlg::Matrix<1, 5> dN(Core::LinAlg::Initialization::zero);
    Core::FE::shape_function_1d_deriv1(dN, r, Core::FE::CellType::line5);
    std::vector<double> dN_ref = {0.10004904982439003, -0.17231796850818118, -0.63400202496498892,
        -1.2044610680513734, 1.9107320117001536};
    for (std::size_t i = 0; i < dN_ref.size(); ++i) EXPECT_NEAR(dN(i), dN_ref[i], 1e-10);

    Core::LinAlg::Matrix<1, 5> ddN(Core::LinAlg::Initialization::zero);
    Core::FE::shape_function_1d_deriv2(ddN, r, Core::FE::CellType::line5);
    std::vector<double> ddN_ref = {-0.70522785933180909, 0.28242645266819083, 5.8332571253272363,
        -9.2684042199908543, 3.8579485013272361};
    for (std::size_t i = 0; i < ddN_ref.size(); ++i) EXPECT_NEAR(ddN(i), ddN_ref[i], 1e-10);
  }

  TEST(ElementShapeFunctionsTest, TestDualHermiteLine2)
  {
    const double r = 0.123456789;
    const double L = 1.0;

    Core::LinAlg::Matrix<4, 1> Phi(Core::LinAlg::Initialization::zero);
    Core::FE::shape_function_dual_hermite_1d(Phi, r, L, Core::FE::CellType::line2);

    std::vector<double> Phi_ref = {
        -0.24634578918994066, -0.05276684284142519, -1.1393423701836305, -14.366895799082851};

    for (std::size_t i = 0; i < Phi_ref.size(); ++i) EXPECT_NEAR(Phi(i), Phi_ref[i], 1e-10);

    Core::LinAlg::Matrix<1, 4> dPhi(Core::LinAlg::Initialization::zero);
    Core::FE::shape_function_dual_hermite_1d_deriv1(dPhi, r, L, Core::FE::CellType::line2);

    std::vector<double> dPhi_ref = {
        4.275834475307499, -60.75445819868999, -2.423982640307499, -49.64334718868999};

    for (std::size_t i = 0; i < dPhi_ref.size(); ++i) EXPECT_NEAR(dPhi(i), dPhi_ref[i], 1e-10);

    Core::LinAlg::Matrix<1, 4> ddPhi(Core::LinAlg::Initialization::zero);
    Core::FE::shape_function_dual_hermite_1d_deriv2(ddPhi, r, L, Core::FE::CellType::line2);

    std::vector<double> ddPhi_ref = {1.0185185775, 32.77777707, 13.9814814225, 122.77777707};

    for (std::size_t i = 0; i < ddPhi_ref.size(); ++i) EXPECT_NEAR(ddPhi(i), ddPhi_ref[i], 1e-10);
  }

  TEST(ElementShapeFunctionsTest, TestDualHermiteLine2Length03)
  {
    const double r = 0.123456789;
    const double L = 0.3;

    Core::LinAlg::Matrix<4, 1> Phi(Core::LinAlg::Initialization::zero);
    Core::FE::shape_function_dual_hermite_1d(Phi, r, L, Core::FE::CellType::line2);

    std::vector<double> Phi_ref = {
        -0.24634578918994066, -0.17588947613808398, -1.1393423701836305, -47.8896526636095};

    for (std::size_t i = 0; i < Phi_ref.size(); ++i) EXPECT_NEAR(Phi(i), Phi_ref[i], 1e-10);

    Core::LinAlg::Matrix<1, 4> dPhi(Core::LinAlg::Initialization::zero);
    Core::FE::shape_function_dual_hermite_1d_deriv1(dPhi, r, L, Core::FE::CellType::line2);

    std::vector<double> dPhi_ref = {
        4.275834475307499, -202.51486066229998, -2.423982640307499, -165.47782396229996};

    for (std::size_t i = 0; i < dPhi_ref.size(); ++i) EXPECT_NEAR(dPhi(i), dPhi_ref[i], 1e-10);

    Core::LinAlg::Matrix<1, 4> ddPhi(Core::LinAlg::Initialization::zero);
    Core::FE::shape_function_dual_hermite_1d_deriv2(ddPhi, r, L, Core::FE::CellType::line2);

    std::vector<double> ddPhi_ref = {1.0185185775, 109.2592569, 13.9814814225, 409.2592569};

    for (std::size_t i = 0; i < ddPhi_ref.size(); ++i) EXPECT_NEAR(ddPhi(i), ddPhi_ref[i], 1e-10);
  }

}  // namespace

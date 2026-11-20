// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_fem_geometry_element_volume.hpp"

#include <numbers>

namespace
{
  using namespace FourC;

  class ElementVolumeTest : public ::testing::Test
  {
   protected:
    void SetUp() override {}

    //! testing parameters
    static constexpr double TOL = 1.0e-15;
  };

  TEST_F(ElementVolumeTest, TestElementLength_line2)
  {
    Core::LinAlg::Matrix<3, 2> xyz(Core::LinAlg::Initialization::zero);
    xyz(0, 0) = 0.0;
    xyz(1, 0) = 0.0;
    xyz(2, 0) = 0.0;  // node 1
    xyz(0, 1) = 2.0;
    xyz(1, 1) = 1.0;
    xyz(2, 1) = 2.0;  // node 2

    double length = Core::Geo::element_length<Core::FE::CellType::line2>(xyz);

    double correct_length = 3.0;
    EXPECT_NEAR(length, correct_length, ElementVolumeTest::TOL);
  }

  TEST_F(ElementVolumeTest, TestElementLength_line3)
  {
    Core::LinAlg::Matrix<3, 3> xyz(Core::LinAlg::Initialization::zero);
    xyz(0, 0) = 0.0;
    xyz(1, 0) = 0.0;
    xyz(2, 0) = 0.0;  // node 1
    xyz(0, 1) = 2.0;
    xyz(1, 1) = 1.0;
    xyz(2, 1) = 2.0;  // node 2
    xyz(0, 2) = 1.0;
    xyz(1, 2) = 0.5;
    xyz(2, 2) = 1.0;  // node 3

    double length = Core::Geo::element_length<Core::FE::CellType::line3>(xyz);

    double correct_length = 3.0;
    EXPECT_NEAR(length, correct_length, ElementVolumeTest::TOL);
  }

  TEST_F(ElementVolumeTest, TestElementLength_nurbs2)
  {
    Core::LinAlg::Matrix<3, 2> xyz(Core::LinAlg::Initialization::zero);
    xyz(0, 0) = 0.0;
    xyz(1, 0) = 0.0;
    xyz(2, 0) = 0.0;  // node 1
    xyz(0, 1) = 2.0;
    xyz(1, 1) = 1.0;
    xyz(2, 1) = 2.0;  // node 2

    std::vector<Core::LinAlg::SerialDenseVector> knots(1);
    knots[0].resize(4);
    knots[0][0] = 0;
    knots[0][1] = 0;
    knots[0][2] = 1;
    knots[0][3] = 1;

    Core::LinAlg::SerialDenseVector weights(2);
    weights(0) = 1.0;
    weights(1) = 1.0;

    double length = Core::Geo::element_length<Core::FE::CellType::nurbs2>(xyz, knots, weights);

    double correct_length = 3.0;
    EXPECT_NEAR(length, correct_length, ElementVolumeTest::TOL);
  }

  TEST_F(ElementVolumeTest, TestElementLength_nurbs3)
  {
    // straight line
    {
      Core::LinAlg::Matrix<3, 3> xyz(Core::LinAlg::Initialization::zero);
      xyz(0, 0) = 0.0;
      xyz(1, 0) = 0.0;
      xyz(2, 0) = 0.0;  // node 1
      xyz(0, 1) = 1.0;
      xyz(1, 1) = 0.5;
      xyz(2, 1) = 1.0;  // node 2
      xyz(0, 2) = 2.0;
      xyz(1, 2) = 1.0;
      xyz(2, 2) = 2.0;  // node 3

      std::vector<Core::LinAlg::SerialDenseVector> knots(1);
      knots[0].resize(6);
      knots[0][0] = 0;
      knots[0][1] = 0;
      knots[0][2] = 0;
      knots[0][3] = 1;
      knots[0][4] = 1;
      knots[0][5] = 1;

      Core::LinAlg::SerialDenseVector weights(3);
      weights(0) = 1.0;
      weights(1) = 1.0;
      weights(2) = 1.0;

      double length = Core::Geo::element_length<Core::FE::CellType::nurbs3>(xyz, knots, weights);

      double correct_length = 3.0;
      EXPECT_NEAR(length, correct_length, ElementVolumeTest::TOL);
    }

    // quarter circle
    {
      Core::LinAlg::Matrix<3, 3> xyz(Core::LinAlg::Initialization::zero);
      xyz(0, 0) = 0.0;
      xyz(1, 0) = 0.0;
      xyz(2, 0) = 0.0;  // node 1
      xyz(0, 1) = 0.0;
      xyz(1, 1) = 0.0;
      xyz(2, 1) = 1.0;  // node 2
      xyz(0, 2) = 0.5 * std::numbers::sqrt2;
      xyz(1, 2) = 0.5 * std::numbers::sqrt2;
      xyz(2, 2) = 1.0;  // node 3

      std::vector<Core::LinAlg::SerialDenseVector> knots(1);
      knots[0].resize(6);
      knots[0][0] = 0;
      knots[0][1] = 0;
      knots[0][2] = 0;
      knots[0][3] = 1;
      knots[0][4] = 1;
      knots[0][5] = 1;

      Core::LinAlg::SerialDenseVector weights(3);
      weights(0) = 1.0;
      weights(1) = 0.5 * std::numbers::sqrt2;
      weights(2) = 1.0;

      double length = Core::Geo::element_length<Core::FE::CellType::nurbs3>(xyz, knots, weights);

      double correct_length = 0.5 * std::numbers::pi;
      // The quadrature rule defines the accuracy, we use the default ones, so a rather coarse
      // approximation of the true length, which is still sufficient.
      EXPECT_NEAR(length, correct_length, 1e-3);
    }
  }

  TEST_F(ElementVolumeTest, TestElementArea_tri3)
  {
    Core::LinAlg::Matrix<3, 3> xyz(Core::LinAlg::Initialization::zero);
    xyz(0, 0) = 0.0;
    xyz(1, 0) = 0.0;
    xyz(2, 0) = 0.0;  // node 1
    xyz(0, 1) = 1.0;
    xyz(1, 1) = 0.0;
    xyz(2, 1) = 0.0;  // node 2
    xyz(0, 2) = 1.0;
    xyz(1, 2) = 1.0;
    xyz(2, 2) = 0.0;  // node 3

    double area = Core::Geo::element_area<Core::FE::CellType::tri3>(xyz);

    double correct_area = 0.5;
    EXPECT_NEAR(area, correct_area, ElementVolumeTest::TOL);
  }

  TEST_F(ElementVolumeTest, TestElementArea_tri6)
  {
    Core::LinAlg::Matrix<3, 6> xyz(Core::LinAlg::Initialization::zero);
    xyz(0, 0) = 0.0;
    xyz(1, 0) = 0.0;
    xyz(2, 0) = 0.0;  // node 1
    xyz(0, 1) = 1.0;
    xyz(1, 1) = 0.0;
    xyz(2, 1) = 0.0;  // node 2
    xyz(0, 2) = 1.0;
    xyz(1, 2) = 1.0;
    xyz(2, 2) = 0.0;  // node 3
    xyz(0, 3) = 0.5;
    xyz(1, 3) = 0.0;
    xyz(2, 3) = 0.0;  // node 4
    xyz(0, 4) = 1.0;
    xyz(1, 4) = 0.5;
    xyz(2, 4) = 0.0;  // node 5
    xyz(0, 5) = 0.5;
    xyz(1, 5) = 0.5;
    xyz(2, 5) = 0.0;  // node 6

    double area = Core::Geo::element_area<Core::FE::CellType::tri6>(xyz);

    double correct_area = 0.5;
    EXPECT_NEAR(area, correct_area, ElementVolumeTest::TOL);
  }

  TEST_F(ElementVolumeTest, TestElementArea_quad4)
  {
    Core::LinAlg::Matrix<3, 4> xyz(Core::LinAlg::Initialization::zero);
    xyz(0, 0) = 0.0;
    xyz(1, 0) = 0.0;
    xyz(2, 0) = 0.0;  // node 1
    xyz(0, 1) = 2.0;
    xyz(1, 1) = 0.0;
    xyz(2, 1) = 0.0;  // node 2
    xyz(0, 2) = 1.0;
    xyz(1, 2) = 1.0;
    xyz(2, 2) = 0.0;  // node 3
    xyz(0, 3) = 0.0;
    xyz(1, 3) = 1.0;
    xyz(2, 3) = 0.0;  // node 4

    double area = Core::Geo::element_area<Core::FE::CellType::quad4>(xyz);

    double correct_area = 1.5;
    EXPECT_NEAR(area, correct_area, ElementVolumeTest::TOL);
  }

  TEST_F(ElementVolumeTest, TestElementArea_quad8)
  {
    Core::LinAlg::Matrix<3, 8> xyz(Core::LinAlg::Initialization::zero);
    xyz(0, 0) = 0.0;
    xyz(1, 0) = 0.0;
    xyz(2, 0) = 0.0;  // node 1
    xyz(0, 1) = 2.0;
    xyz(1, 1) = 0.0;
    xyz(2, 1) = 0.0;  // node 2
    xyz(0, 2) = 1.0;
    xyz(1, 2) = 1.0;
    xyz(2, 2) = 0.0;  // node 3
    xyz(0, 3) = 0.0;
    xyz(1, 3) = 1.0;
    xyz(2, 3) = 0.0;  // node 4
    xyz(0, 4) = 1.0;
    xyz(1, 4) = 0.0;
    xyz(2, 4) = 0.0;  // node 5
    xyz(0, 5) = 1.5;
    xyz(1, 5) = 0.5;
    xyz(2, 5) = 0.0;  // node 6
    xyz(0, 6) = 0.5;
    xyz(1, 6) = 1.0;
    xyz(2, 6) = 0.0;  // node 7
    xyz(0, 7) = 0.0;
    xyz(1, 7) = 0.5;
    xyz(2, 7) = 0.0;  // node 8

    double area = Core::Geo::element_area<Core::FE::CellType::quad8>(xyz);

    double correct_area = 1.5;
    EXPECT_NEAR(area, correct_area, ElementVolumeTest::TOL);
  }

  TEST_F(ElementVolumeTest, TestElementArea_quad9)
  {
    Core::LinAlg::Matrix<3, 9> xyz(Core::LinAlg::Initialization::zero);
    xyz(0, 0) = 0.0;
    xyz(1, 0) = 0.0;
    xyz(2, 0) = 0.0;  // node 1
    xyz(0, 1) = 2.0;
    xyz(1, 1) = 0.0;
    xyz(2, 1) = 0.0;  // node 2
    xyz(0, 2) = 1.0;
    xyz(1, 2) = 1.0;
    xyz(2, 2) = 0.0;  // node 3
    xyz(0, 3) = 0.0;
    xyz(1, 3) = 1.0;
    xyz(2, 3) = 0.0;  // node 4
    xyz(0, 4) = 1.0;
    xyz(1, 4) = 0.0;
    xyz(2, 4) = 0.0;  // node 5
    xyz(0, 5) = 1.5;
    xyz(1, 5) = 0.5;
    xyz(2, 5) = 0.0;  // node 6
    xyz(0, 6) = 0.5;
    xyz(1, 6) = 1.0;
    xyz(2, 6) = 0.0;  // node 7
    xyz(0, 7) = 0.0;
    xyz(1, 7) = 0.5;
    xyz(2, 7) = 0.0;  // node 8
    xyz(0, 8) = 0.75;
    xyz(1, 8) = 0.5;
    xyz(2, 8) = 0.0;  // node 9

    double area = Core::Geo::element_area<Core::FE::CellType::quad9>(xyz);

    double correct_area = 1.5;
    EXPECT_NEAR(area, correct_area, ElementVolumeTest::TOL);
  }

  TEST_F(ElementVolumeTest, TestElementArea_nurbs4)
  {
    Core::LinAlg::Matrix<3, 4> xyz(Core::LinAlg::Initialization::zero);
    xyz(0, 0) = 0.0;
    xyz(1, 0) = 0.0;
    xyz(2, 0) = 0.0;  // node 1
    xyz(0, 1) = 2.0;
    xyz(1, 1) = 0.0;
    xyz(2, 1) = 0.0;  // node 2
    xyz(0, 2) = 0.0;
    xyz(1, 2) = 1.0;
    xyz(2, 2) = 0.0;  // node 3
    xyz(0, 3) = 1.0;
    xyz(1, 3) = 1.0;
    xyz(2, 3) = 0.0;  // node 4

    std::vector<Core::LinAlg::SerialDenseVector> knots(2);
    knots[0].resize(4);
    knots[0][0] = 0;
    knots[0][1] = 0;
    knots[0][2] = 1;
    knots[0][3] = 1;
    knots[1].resize(4);
    knots[1][0] = 0;
    knots[1][1] = 0;
    knots[1][2] = 1;
    knots[1][3] = 1;

    Core::LinAlg::SerialDenseVector weights(4);
    weights(0) = 1.0;
    weights(1) = 1.0;
    weights(2) = 1.0;
    weights(3) = 1.0;

    double area = Core::Geo::element_area<Core::FE::CellType::nurbs4>(xyz, knots, weights);

    double correct_area = 1.5;
    EXPECT_NEAR(area, correct_area, ElementVolumeTest::TOL);
  }

  TEST_F(ElementVolumeTest, TestElementArea_nurbs9)
  {
    // square surface
    {
      Core::LinAlg::Matrix<3, 9> xyz(Core::LinAlg::Initialization::zero);
      xyz(0, 0) = 0.0;
      xyz(1, 0) = 0.0;
      xyz(2, 0) = 0.0;  // node 1
      xyz(0, 1) = 1.0;
      xyz(1, 1) = 0.0;
      xyz(2, 1) = 0.0;  // node 2
      xyz(0, 2) = 2.0;
      xyz(1, 2) = 0.0;
      xyz(2, 2) = 0.0;  // node 3
      xyz(0, 3) = 0.0;
      xyz(1, 3) = 1.0;
      xyz(2, 3) = 0.0;  // node 4
      xyz(0, 4) = 1.0;
      xyz(1, 4) = 1.0;
      xyz(2, 4) = 0.0;  // node 5
      xyz(0, 5) = 2.0;
      xyz(1, 5) = 1.0;
      xyz(2, 5) = 0.0;  // node 6
      xyz(0, 6) = 0.0;
      xyz(1, 6) = 2.0;
      xyz(2, 6) = 0.0;  // node 7
      xyz(0, 7) = 1.0;
      xyz(1, 7) = 2.0;
      xyz(2, 7) = 0.0;  // node 8
      xyz(0, 8) = 2.0;
      xyz(1, 8) = 2.0;
      xyz(2, 8) = 0.0;  // node 9

      std::vector<Core::LinAlg::SerialDenseVector> knots(2);
      knots[0].resize(6);
      knots[0][0] = 0;
      knots[0][1] = 0;
      knots[0][2] = 0;
      knots[0][3] = 1;
      knots[0][4] = 1;
      knots[0][5] = 1;
      knots[1].resize(6);
      knots[1][0] = 0;
      knots[1][1] = 0;
      knots[1][2] = 0;
      knots[1][3] = 1;
      knots[1][4] = 1;
      knots[1][5] = 1;

      Core::LinAlg::SerialDenseVector weights(9);
      weights(0) = 1.0;
      weights(1) = 1.0;
      weights(2) = 1.0;
      weights(3) = 1.0;
      weights(4) = 1.0;
      weights(5) = 1.0;
      weights(6) = 1.0;
      weights(7) = 1.0;
      weights(8) = 1.0;

      double area = Core::Geo::element_area<Core::FE::CellType::nurbs9>(xyz, knots, weights);

      double correct_area = 4.0;
      EXPECT_NEAR(area, correct_area, ElementVolumeTest::TOL);
    }

    // octant surface of a sphere
    {
      Core::LinAlg::Matrix<3, 9> xyz(Core::LinAlg::Initialization::zero);
      xyz(0, 0) = 1.0;
      xyz(1, 0) = 0.0;
      xyz(2, 0) = 0.0;  // node 1
      xyz(0, 1) = 1.0;
      xyz(1, 1) = 1.0;
      xyz(2, 1) = 0.0;  // node 2
      xyz(0, 2) = 0.0;
      xyz(1, 2) = 1.0;
      xyz(2, 2) = 0.0;  // node 3
      xyz(0, 3) = 1.0;
      xyz(1, 3) = 0.0;
      xyz(2, 3) = 1.0;  // node 4
      xyz(0, 4) = 1.0;
      xyz(1, 4) = 1.0;
      xyz(2, 4) = 1.0;  // node 5
      xyz(0, 5) = 0.0;
      xyz(1, 5) = 1.0;
      xyz(2, 5) = 1.0;  // node 6
      xyz(0, 6) = 0.0;
      xyz(1, 6) = 0.0;
      xyz(2, 6) = 1.0;  // node 7
      xyz(0, 7) = 0.0;
      xyz(1, 7) = 0.0;
      xyz(2, 7) = 1.0;  // node 8
      xyz(0, 8) = 0.0;
      xyz(1, 8) = 0.0;
      xyz(2, 8) = 1.0;  // node 9

      std::vector<Core::LinAlg::SerialDenseVector> knots(2);
      knots[0].resize(6);
      knots[0][0] = 0;
      knots[0][1] = 0;
      knots[0][2] = 0;
      knots[0][3] = 1;
      knots[0][4] = 1;
      knots[0][5] = 1;
      knots[1].resize(6);
      knots[1][0] = 0;
      knots[1][1] = 0;
      knots[1][2] = 0;
      knots[1][3] = 1;
      knots[1][4] = 1;
      knots[1][5] = 1;

      Core::LinAlg::SerialDenseVector weights(9);
      weights(0) = 1.0;
      weights(1) = 0.5 * std::numbers::sqrt2;
      weights(2) = 1.0;
      weights(3) = 0.5 * std::numbers::sqrt2;
      weights(4) = 0.5;
      weights(5) = 0.5 * std::numbers::sqrt2;
      weights(6) = 1.0;
      weights(7) = 0.5 * std::numbers::sqrt2;
      weights(8) = 1.0;

      double area = Core::Geo::element_area<Core::FE::CellType::nurbs9>(xyz, knots, weights);

      double correct_area = 0.5 * std::numbers::pi;
      // The quadrature rule defines the accuracy, we use the default ones, so a rather coarse
      // approximation of the true length, which is still sufficient.
      EXPECT_NEAR(area, correct_area, 1e-2);
    }
  }

  TEST_F(ElementVolumeTest, TestElementVolume_tet4)
  {
    Core::LinAlg::Matrix<3, 4> xyz(Core::LinAlg::Initialization::zero);
    xyz(0, 0) = 0.0;
    xyz(1, 0) = 0.0;
    xyz(2, 0) = 0.0;  // node 1
    xyz(0, 1) = 1.0;
    xyz(1, 1) = 0.0;
    xyz(2, 1) = 0.0;  // node 2
    xyz(0, 2) = 1.0;
    xyz(1, 2) = 1.0;
    xyz(2, 2) = 0.0;  // node 3
    xyz(0, 3) = 0.0;
    xyz(1, 3) = 0.0;
    xyz(2, 3) = 1.0;  // node 4

    double volume = Core::Geo::element_volume<Core::FE::CellType::tet4>(xyz);

    double correct_volume = 0.5 / 3;
    EXPECT_NEAR(volume, correct_volume, ElementVolumeTest::TOL);
  }

  TEST_F(ElementVolumeTest, TestElementVolume_tet10)
  {
    Core::LinAlg::Matrix<3, 10> xyz(Core::LinAlg::Initialization::zero);
    xyz(0, 0) = 0.0;
    xyz(1, 0) = 0.0;
    xyz(2, 0) = 0.0;  // node 1
    xyz(0, 1) = 1.0;
    xyz(1, 1) = 0.0;
    xyz(2, 1) = 0.0;  // node 2
    xyz(0, 2) = 1.0;
    xyz(1, 2) = 1.0;
    xyz(2, 2) = 0.0;  // node 3
    xyz(0, 3) = 0.0;
    xyz(1, 3) = 0.0;
    xyz(2, 3) = 1.0;  // node 4
    xyz(0, 4) = 0.5;
    xyz(1, 4) = 0.0;
    xyz(2, 4) = 0.0;  // node 5
    xyz(0, 5) = 1.0;
    xyz(1, 5) = 0.5;
    xyz(2, 5) = 0.0;  // node 6
    xyz(0, 6) = 0.5;
    xyz(1, 6) = 0.5;
    xyz(2, 6) = 0.0;  // node 7
    xyz(0, 7) = 0.0;
    xyz(1, 7) = 0.0;
    xyz(2, 7) = 0.5;  // node 8
    xyz(0, 8) = 0.5;
    xyz(1, 8) = 0.0;
    xyz(2, 8) = 0.5;  // node 9
    xyz(0, 9) = 0.5;
    xyz(1, 9) = 0.5;
    xyz(2, 9) = 0.5;  // node 10

    double volume = Core::Geo::element_volume<Core::FE::CellType::tet10>(xyz);

    double correct_volume = 0.5 / 3;
    EXPECT_NEAR(volume, correct_volume, ElementVolumeTest::TOL);
  }

  TEST_F(ElementVolumeTest, TestElementVolume_hex8)
  {
    Core::LinAlg::Matrix<3, 8> xyz(Core::LinAlg::Initialization::zero);
    xyz(0, 0) = 0.0;
    xyz(1, 0) = 0.0;
    xyz(2, 0) = 0.0;  // node 1
    xyz(0, 1) = 1.5;
    xyz(1, 1) = 0.0;
    xyz(2, 1) = 0.0;  // node 2
    xyz(0, 2) = 1.5;
    xyz(1, 2) = 1.0;
    xyz(2, 2) = 0.0;  // node 3
    xyz(0, 3) = 0.0;
    xyz(1, 3) = 1.0;
    xyz(2, 3) = 0.0;  // node 4
    xyz(0, 4) = 0.0;
    xyz(1, 4) = 0.0;
    xyz(2, 4) = 1.0;  // node 5
    xyz(0, 5) = 1.0;
    xyz(1, 5) = 0.0;
    xyz(2, 5) = 1.0;  // node 6
    xyz(0, 6) = 1.0;
    xyz(1, 6) = 1.0;
    xyz(2, 6) = 1.0;  // node 7
    xyz(0, 7) = 0.0;
    xyz(1, 7) = 1.0;
    xyz(2, 7) = 1.0;  // node 8

    double volume = Core::Geo::element_volume<Core::FE::CellType::hex8>(xyz);

    double correct_volume = 1.25;
    EXPECT_NEAR(volume, correct_volume, ElementVolumeTest::TOL);
  }

  TEST_F(ElementVolumeTest, TestElementVolume_hex20)
  {
    Core::LinAlg::Matrix<3, 20> xyz(Core::LinAlg::Initialization::zero);

    double x1 = xyz(0, 0) = 0.0;
    double y1 = xyz(1, 0) = 0.0;
    double z1 = xyz(2, 0) = 0.0;  // node 1
    double x2 = xyz(0, 1) = 1.5;
    double y2 = xyz(1, 1) = 0.0;
    double z2 = xyz(2, 1) = 0.0;  // node 2
    double x3 = xyz(0, 2) = 1.5;
    double y3 = xyz(1, 2) = 1.0;
    double z3 = xyz(2, 2) = 0.0;  // node 3
    double x4 = xyz(0, 3) = 0.0;
    double y4 = xyz(1, 3) = 1.0;
    double z4 = xyz(2, 3) = 0.0;  // node 4

    double x5 = xyz(0, 4) = 0.0;
    double y5 = xyz(1, 4) = 0.0;
    double z5 = xyz(2, 4) = 1.0;  // node 5
    double x6 = xyz(0, 5) = 1.0;
    double y6 = xyz(1, 5) = 0.0;
    double z6 = xyz(2, 5) = 1.0;  // node 6
    double x7 = xyz(0, 6) = 1.0;
    double y7 = xyz(1, 6) = 1.0;
    double z7 = xyz(2, 6) = 1.0;  // node 7
    double x8 = xyz(0, 7) = 0.0;
    double y8 = xyz(1, 7) = 1.0;
    double z8 = xyz(2, 7) = 1.0;  // node 8

    xyz(0, 8) = 0.5 * (x1 + x2);
    xyz(1, 8) = 0.5 * (y1 + y2);
    xyz(2, 8) = 0.5 * (z1 + z2);  // node 9
    xyz(0, 9) = 0.5 * (x2 + x3);
    xyz(1, 9) = 0.5 * (y2 + y3);
    xyz(2, 9) = 0.5 * (z2 + z3);  // node 10
    xyz(0, 10) = 0.5 * (x3 + x4);
    xyz(1, 10) = 0.5 * (y3 + y4);
    xyz(2, 10) = 0.5 * (z3 + z4);  // node 11
    xyz(0, 11) = 0.5 * (x1 + x4);
    xyz(1, 11) = 0.5 * (y1 + y4);
    xyz(2, 11) = 0.5 * (z1 + z4);  // node 12

    xyz(0, 12) = 0.5 * (x1 + x5);
    xyz(1, 12) = 0.5 * (y1 + y5);
    xyz(2, 12) = 0.5 * (z1 + z5);  // node 13
    xyz(0, 13) = 0.5 * (x2 + x6);
    xyz(1, 13) = 0.5 * (y2 + y6);
    xyz(2, 13) = 0.5 * (z2 + z6);  // node 14
    xyz(0, 14) = 0.5 * (x3 + x7);
    xyz(1, 14) = 0.5 * (y3 + y7);
    xyz(2, 14) = 0.5 * (z3 + z7);  // node 15
    xyz(0, 15) = 0.5 * (x4 + x8);
    xyz(1, 15) = 0.5 * (y4 + y8);
    xyz(2, 15) = 0.5 * (z4 + z8);  // node 16

    xyz(0, 16) = 0.5 * (x5 + x6);
    xyz(1, 16) = 0.5 * (y5 + y6);
    xyz(2, 16) = 0.5 * (z5 + z6);  // node 17
    xyz(0, 17) = 0.5 * (x6 + x7);
    xyz(1, 17) = 0.5 * (y6 + y7);
    xyz(2, 17) = 0.5 * (z6 + z7);  // node 18
    xyz(0, 18) = 0.5 * (x7 + x8);
    xyz(1, 18) = 0.5 * (y7 + y8);
    xyz(2, 18) = 0.5 * (z7 + z8);  // node 19
    xyz(0, 19) = 0.5 * (x8 + x5);
    xyz(1, 19) = 0.5 * (y8 + y5);
    xyz(2, 19) = 0.5 * (z8 + z5);  // node 20

    double volume = Core::Geo::element_volume<Core::FE::CellType::hex20>(xyz);

    double correct_volume = 1.25;
    EXPECT_NEAR(volume, correct_volume, ElementVolumeTest::TOL);
  }

  TEST_F(ElementVolumeTest, TestElementVolume_hex27)
  {
    Core::LinAlg::Matrix<3, 27> xyz(Core::LinAlg::Initialization::zero);

    double x1 = xyz(0, 0) = 0.0;
    double y1 = xyz(1, 0) = 0.0;
    double z1 = xyz(2, 0) = 0.0;  // node 1
    double x2 = xyz(0, 1) = 1.5;
    double y2 = xyz(1, 1) = 0.0;
    double z2 = xyz(2, 1) = 0.0;  // node 2
    double x3 = xyz(0, 2) = 1.5;
    double y3 = xyz(1, 2) = 1.0;
    double z3 = xyz(2, 2) = 0.0;  // node 3
    double x4 = xyz(0, 3) = 0.0;
    double y4 = xyz(1, 3) = 1.0;
    double z4 = xyz(2, 3) = 0.0;  // node 4

    double x5 = xyz(0, 4) = 0.0;
    double y5 = xyz(1, 4) = 0.0;
    double z5 = xyz(2, 4) = 1.0;  // node 5
    double x6 = xyz(0, 5) = 1.0;
    double y6 = xyz(1, 5) = 0.0;
    double z6 = xyz(2, 5) = 1.0;  // node 6
    double x7 = xyz(0, 6) = 1.0;
    double y7 = xyz(1, 6) = 1.0;
    double z7 = xyz(2, 6) = 1.0;  // node 7
    double x8 = xyz(0, 7) = 0.0;
    double y8 = xyz(1, 7) = 1.0;
    double z8 = xyz(2, 7) = 1.0;  // node 8

    xyz(0, 8) = 0.5 * (x1 + x2);
    xyz(1, 8) = 0.5 * (y1 + y2);
    xyz(2, 8) = 0.5 * (z1 + z2);  // node 9
    xyz(0, 9) = 0.5 * (x2 + x3);
    xyz(1, 9) = 0.5 * (y2 + y3);
    xyz(2, 9) = 0.5 * (z2 + z3);  // node 10
    xyz(0, 10) = 0.5 * (x3 + x4);
    xyz(1, 10) = 0.5 * (y3 + y4);
    xyz(2, 10) = 0.5 * (z3 + z4);  // node 11
    xyz(0, 11) = 0.5 * (x1 + x4);
    xyz(1, 11) = 0.5 * (y1 + y4);
    xyz(2, 11) = 0.5 * (z1 + z4);  // node 12

    xyz(0, 12) = 0.5 * (x1 + x5);
    xyz(1, 12) = 0.5 * (y1 + y5);
    xyz(2, 12) = 0.5 * (z1 + z5);  // node 13
    xyz(0, 13) = 0.5 * (x2 + x6);
    xyz(1, 13) = 0.5 * (y2 + y6);
    xyz(2, 13) = 0.5 * (z2 + z6);  // node 14
    xyz(0, 14) = 0.5 * (x3 + x7);
    xyz(1, 14) = 0.5 * (y3 + y7);
    xyz(2, 14) = 0.5 * (z3 + z7);  // node 15
    xyz(0, 15) = 0.5 * (x4 + x8);
    xyz(1, 15) = 0.5 * (y4 + y8);
    xyz(2, 15) = 0.5 * (z4 + z8);  // node 16

    xyz(0, 16) = 0.5 * (x5 + x6);
    xyz(1, 16) = 0.5 * (y5 + y6);
    xyz(2, 16) = 0.5 * (z5 + z6);  // node 17
    xyz(0, 17) = 0.5 * (x6 + x7);
    xyz(1, 17) = 0.5 * (y6 + y7);
    xyz(2, 17) = 0.5 * (z6 + z7);  // node 18
    xyz(0, 18) = 0.5 * (x7 + x8);
    xyz(1, 18) = 0.5 * (y7 + y8);
    xyz(2, 18) = 0.5 * (z7 + z8);  // node 19
    xyz(0, 19) = 0.5 * (x8 + x5);
    xyz(1, 19) = 0.5 * (y8 + y5);
    xyz(2, 19) = 0.5 * (z8 + z5);  // node 20

    xyz(0, 20) = 0.25 * (x1 + x2 + x3 + x4);
    xyz(1, 20) = 0.25 * (y1 + y2 + y3 + y4);
    xyz(2, 20) = 0.25 * (z1 + z2 + z3 + z4);  // node 21
    xyz(0, 21) = 0.25 * (x1 + x2 + x5 + x6);
    xyz(1, 21) = 0.25 * (y1 + y2 + y5 + y6);
    xyz(2, 21) = 0.25 * (z1 + z2 + z5 + z6);  // node 22
    xyz(0, 22) = 0.25 * (x2 + x3 + x6 + x7);
    xyz(1, 22) = 0.25 * (y2 + y3 + y6 + y7);
    xyz(2, 22) = 0.25 * (z2 + z3 + z6 + z7);  // node 23
    xyz(0, 23) = 0.25 * (x3 + x4 + x7 + x8);
    xyz(1, 23) = 0.25 * (y3 + y4 + y7 + y8);
    xyz(2, 23) = 0.25 * (z3 + z4 + z7 + z8);  // node 24
    xyz(0, 24) = 0.25 * (x1 + x4 + x5 + x8);
    xyz(1, 24) = 0.25 * (y1 + y4 + y5 + y8);
    xyz(2, 24) = 0.25 * (z1 + z4 + z5 + z8);  // node 25
    xyz(0, 25) = 0.25 * (x5 + x6 + x7 + x8);
    xyz(1, 25) = 0.25 * (y5 + y6 + y7 + y8);
    xyz(2, 25) = 0.25 * (z5 + z6 + z7 + z8);  // node 26

    xyz(0, 26) = 0.125 * (x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8);
    xyz(1, 26) = 0.125 * (y1 + y2 + y3 + y4 + y5 + y6 + y7 + y8);
    xyz(2, 26) = 0.125 * (z1 + z2 + z3 + z4 + z5 + z6 + z7 + z8);  // node 27

    double volume = Core::Geo::element_volume<Core::FE::CellType::hex27>(xyz);

    double correct_volume = 1.25;
    EXPECT_NEAR(volume, correct_volume, ElementVolumeTest::TOL);
  }

  TEST_F(ElementVolumeTest, TestElementVolume_nurbs8)
  {
    Core::LinAlg::Matrix<3, 8> xyz(Core::LinAlg::Initialization::zero);
    xyz(0, 0) = 0.0;
    xyz(1, 0) = 0.0;
    xyz(2, 0) = 0.0;  // node 1
    xyz(0, 1) = 1.5;
    xyz(1, 1) = 0.0;
    xyz(2, 1) = 0.0;  // node 2
    xyz(0, 2) = 0.0;
    xyz(1, 2) = 1.0;
    xyz(2, 2) = 0.0;  // node 3
    xyz(0, 3) = 1.5;
    xyz(1, 3) = 1.0;
    xyz(2, 3) = 0.0;  // node 4
    xyz(0, 4) = 0.0;
    xyz(1, 4) = 0.0;
    xyz(2, 4) = 1.0;  // node 5
    xyz(0, 5) = 1.0;
    xyz(1, 5) = 0.0;
    xyz(2, 5) = 1.0;  // node 6
    xyz(0, 6) = 0.0;
    xyz(1, 6) = 1.0;
    xyz(2, 6) = 1.0;  // node 7
    xyz(0, 7) = 1.0;
    xyz(1, 7) = 1.0;
    xyz(2, 7) = 1.0;  // node 8

    std::vector<Core::LinAlg::SerialDenseVector> knots(3);
    knots[0].resize(4);
    knots[0][0] = 0;
    knots[0][1] = 0;
    knots[0][2] = 1;
    knots[0][3] = 1;
    knots[1].resize(4);
    knots[1][0] = 0;
    knots[1][1] = 0;
    knots[1][2] = 1;
    knots[1][3] = 1;
    knots[2].resize(4);
    knots[2][0] = 0;
    knots[2][1] = 0;
    knots[2][2] = 1;
    knots[2][3] = 1;

    Core::LinAlg::SerialDenseVector weights(8);
    weights(0) = 1.0;
    weights(1) = 1.0;
    weights(2) = 1.0;
    weights(3) = 1.0;
    weights(4) = 1.0;
    weights(5) = 1.0;
    weights(6) = 1.0;
    weights(7) = 1.0;

    double volume = Core::Geo::element_volume<Core::FE::CellType::nurbs8>(xyz, knots, weights);

    double correct_volume = 1.25;
    EXPECT_NEAR(volume, correct_volume, ElementVolumeTest::TOL);
  }

  TEST_F(ElementVolumeTest, TestElementVolume_nurbs27)
  {
    // unit-cube volume
    {
      Core::LinAlg::Matrix<3, 27> xyz(Core::LinAlg::Initialization::zero);
      xyz(0, 0) = 0.0;
      xyz(1, 0) = 0.0;
      xyz(2, 0) = 0.0;
      xyz(0, 1) = 0.5;
      xyz(1, 1) = 0.0;
      xyz(2, 1) = 0.0;
      xyz(0, 2) = 1.0;
      xyz(1, 2) = 0.0;
      xyz(2, 2) = 0.0;
      xyz(0, 3) = 0.0;
      xyz(1, 3) = 0.5;
      xyz(2, 3) = 0.0;
      xyz(0, 4) = 0.5;
      xyz(1, 4) = 0.5;
      xyz(2, 4) = 0.0;
      xyz(0, 5) = 1.0;
      xyz(1, 5) = 0.5;
      xyz(2, 5) = 0.0;
      xyz(0, 6) = 0.0;
      xyz(1, 6) = 1.0;
      xyz(2, 6) = 0.0;
      xyz(0, 7) = 0.5;
      xyz(1, 7) = 1.0;
      xyz(2, 7) = 0.0;
      xyz(0, 8) = 1.0;
      xyz(1, 8) = 1.0;
      xyz(2, 8) = 0.0;
      xyz(0, 9) = 0.0;
      xyz(1, 9) = 0.0;
      xyz(2, 9) = 0.5;
      xyz(0, 10) = 0.5;
      xyz(1, 10) = 0.0;
      xyz(2, 10) = 0.5;
      xyz(0, 11) = 1.0;
      xyz(1, 11) = 0.0;
      xyz(2, 11) = 0.5;
      xyz(0, 12) = 0.0;
      xyz(1, 12) = 0.5;
      xyz(2, 12) = 0.5;
      xyz(0, 13) = 0.5;
      xyz(1, 13) = 0.5;
      xyz(2, 13) = 0.5;
      xyz(0, 14) = 1.0;
      xyz(1, 14) = 0.5;
      xyz(2, 14) = 0.5;
      xyz(0, 15) = 0.0;
      xyz(1, 15) = 1.0;
      xyz(2, 15) = 0.5;
      xyz(0, 16) = 0.5;
      xyz(1, 16) = 1.0;
      xyz(2, 16) = 0.5;
      xyz(0, 17) = 1.0;
      xyz(1, 17) = 1.0;
      xyz(2, 17) = 0.5;
      xyz(0, 18) = 0.0;
      xyz(1, 18) = 0.0;
      xyz(2, 18) = 1.0;
      xyz(0, 19) = 0.5;
      xyz(1, 19) = 0.0;
      xyz(2, 19) = 1.0;
      xyz(0, 20) = 1.0;
      xyz(1, 20) = 0.0;
      xyz(2, 20) = 1.0;
      xyz(0, 21) = 0.0;
      xyz(1, 21) = 0.5;
      xyz(2, 21) = 1.0;
      xyz(0, 22) = 0.5;
      xyz(1, 22) = 0.5;
      xyz(2, 22) = 1.0;
      xyz(0, 23) = 1.0;
      xyz(1, 23) = 0.5;
      xyz(2, 23) = 1.0;
      xyz(0, 24) = 0.0;
      xyz(1, 24) = 1.0;
      xyz(2, 24) = 1.0;
      xyz(0, 25) = 0.5;
      xyz(1, 25) = 1.0;
      xyz(2, 25) = 1.0;
      xyz(0, 26) = 1.0;
      xyz(1, 26) = 1.0;
      xyz(2, 26) = 1.0;

      std::vector<Core::LinAlg::SerialDenseVector> knots(3);
      for (int idir = 0; idir < 3; ++idir)
      {
        knots[idir].resize(6);
        knots[idir][0] = 0.0;
        knots[idir][1] = 0.0;
        knots[idir][2] = 0.0;
        knots[idir][3] = 1.0;
        knots[idir][4] = 1.0;
        knots[idir][5] = 1.0;
      }

      Core::LinAlg::SerialDenseVector weights(27);
      weights.putScalar(1.0);

      double volume = Core::Geo::element_volume<Core::FE::CellType::nurbs27>(xyz, knots, weights);

      double correct_volume = 1.0;
      EXPECT_NEAR(volume, correct_volume, ElementVolumeTest::TOL);
    }

    // octant volume of a sphere
    {
      Core::LinAlg::Matrix<3, 27> xyz(Core::LinAlg::Initialization::zero);
      xyz(0, 0) = 0.0;
      xyz(1, 0) = 0.0;
      xyz(2, 0) = 1.0;
      xyz(0, 1) = 0.0;
      xyz(1, 1) = 0.0;
      xyz(2, 1) = 1.0;
      xyz(0, 2) = 0.0;
      xyz(1, 2) = 0.0;
      xyz(2, 2) = 1.0;
      xyz(0, 3) = 1.0;
      xyz(1, 3) = 0.0;
      xyz(2, 3) = 1.0;
      xyz(0, 4) = 1.0;
      xyz(1, 4) = 1.0;
      xyz(2, 4) = 1.0;
      xyz(0, 5) = 0.0;
      xyz(1, 5) = 1.0;
      xyz(2, 5) = 1.0;
      xyz(0, 6) = 1.0;
      xyz(1, 6) = 0.0;
      xyz(2, 6) = 0.0;
      xyz(0, 7) = 1.0;
      xyz(1, 7) = 1.0;
      xyz(2, 7) = 0.0;
      xyz(0, 8) = 0.0;
      xyz(1, 8) = 1.0;
      xyz(2, 8) = 0.0;
      xyz(0, 9) = 0.0;
      xyz(1, 9) = 0.0;
      xyz(2, 9) = 0.5;
      xyz(0, 10) = 0.0;
      xyz(1, 10) = 0.0;
      xyz(2, 10) = 0.5;
      xyz(0, 11) = 0.0;
      xyz(1, 11) = 0.0;
      xyz(2, 11) = 0.5;
      xyz(0, 12) = 0.5;
      xyz(1, 12) = 0.0;
      xyz(2, 12) = 0.5;
      xyz(0, 13) = 0.5;
      xyz(1, 13) = 0.5;
      xyz(2, 13) = 0.5;
      xyz(0, 14) = 0.0;
      xyz(1, 14) = 0.5;
      xyz(2, 14) = 0.5;
      xyz(0, 15) = 0.5;
      xyz(1, 15) = 0.0;
      xyz(2, 15) = 0.0;
      xyz(0, 16) = 0.5;
      xyz(1, 16) = 0.5;
      xyz(2, 16) = 0.0;
      xyz(0, 17) = 0.0;
      xyz(1, 17) = 0.5;
      xyz(2, 17) = 0.0;
      xyz(0, 18) = 0.0;
      xyz(1, 18) = 0.0;
      xyz(2, 18) = 0.0;
      xyz(0, 19) = 0.0;
      xyz(1, 19) = 0.0;
      xyz(2, 19) = 0.0;
      xyz(0, 20) = 0.0;
      xyz(1, 20) = 0.0;
      xyz(2, 20) = 0.0;
      xyz(0, 21) = 0.0;
      xyz(1, 21) = 0.0;
      xyz(2, 21) = 0.0;
      xyz(0, 22) = 0.0;
      xyz(1, 22) = 0.0;
      xyz(2, 22) = 0.0;
      xyz(0, 23) = 0.0;
      xyz(1, 23) = 0.0;
      xyz(2, 23) = 0.0;
      xyz(0, 24) = 0.0;
      xyz(1, 24) = 0.0;
      xyz(2, 24) = 0.0;
      xyz(0, 25) = 0.0;
      xyz(1, 25) = 0.0;
      xyz(2, 25) = 0.0;
      xyz(0, 26) = 0.0;
      xyz(1, 26) = 0.0;
      xyz(2, 26) = 0.0;

      std::vector<Core::LinAlg::SerialDenseVector> knots(3);
      for (int idir = 0; idir < 3; ++idir)
      {
        knots[idir].resize(6);
        knots[idir][0] = 0.0;
        knots[idir][1] = 0.0;
        knots[idir][2] = 0.0;
        knots[idir][3] = 1.0;
        knots[idir][4] = 1.0;
        knots[idir][5] = 1.0;
      }

      const double w = 0.5 * std::numbers::sqrt2;

      Core::LinAlg::SerialDenseVector weights(27);
      weights(0) = 1.0;
      weights(1) = w;
      weights(2) = 1.0;
      weights(3) = w;
      weights(4) = w * w;
      weights(5) = w;
      weights(6) = 1.0;
      weights(7) = w;
      weights(8) = 1.0;
      weights(9) = 1.0;
      weights(10) = w;
      weights(11) = 1.0;
      weights(12) = w;
      weights(13) = w * w;
      weights(14) = w;
      weights(15) = 1.0;
      weights(16) = w;
      weights(17) = 1.0;
      weights(18) = 1.0;
      weights(19) = w;
      weights(20) = 1.0;
      weights(21) = w;
      weights(22) = w * w;
      weights(23) = w;
      weights(24) = 1.0;
      weights(25) = w;
      weights(26) = 1.0;

      double volume = Core::Geo::element_volume<Core::FE::CellType::nurbs27>(xyz, knots, weights);

      double correct_volume = std::numbers::pi / 6.0;
      // The quadrature rule defines the accuracy, we use the default ones, so a rather coarse
      // approximation of the true length, which is still sufficient.
      EXPECT_NEAR(volume, correct_volume, 1e-3);
    }
  }

}  // namespace

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_solid_3D_ele_calc_lib.hpp"

namespace
{
  using namespace FourC;

  TEST(EvaluateParameterCoordinateCentroid, DisTypeHex)
  {
    // only tested for hex8, but equivalent for hex18, hex27, ...
    const auto distype = Core::FE::CellType::hex8;

    Core::LinAlg::Matrix<Discret::Elements::Internal::num_dim<distype>, 1> xi_centroid_ref(
        Core::LinAlg::Initialization::zero);

    Core::LinAlg::Matrix<Discret::Elements::Internal::num_dim<distype>, 1> xi_centroid =
        Discret::Elements::evaluate_parameter_coordinate_centroid<distype>();

    EXPECT_EQ(xi_centroid, xi_centroid_ref);
  }

  TEST(EvaluateParameterCoordinateCentroid, DisTypeTet)
  {
    // only tested for tet4, but equivalent for tet10
    const auto distype = Core::FE::CellType::tet4;

    Core::LinAlg::Matrix<Discret::Elements::Internal::num_dim<distype>, 1> xi_centroid_ref(
        Core::LinAlg::Initialization::zero);
    xi_centroid_ref(0) = 0.25;
    xi_centroid_ref(1) = 0.25;
    xi_centroid_ref(2) = 0.25;

    Core::LinAlg::Matrix<Discret::Elements::Internal::num_dim<distype>, 1> xi_centroid =
        Discret::Elements::evaluate_parameter_coordinate_centroid<distype>();

    EXPECT_EQ(xi_centroid, xi_centroid_ref);
  }

  TEST(EvaluateParameterCoordinateCentroid, DisTypePyramid)
  {
    const auto distype = Core::FE::CellType::pyramid5;

    Core::LinAlg::Matrix<Discret::Elements::Internal::num_dim<distype>, 1> xi_centroid_ref(
        Core::LinAlg::Initialization::zero);
    xi_centroid_ref(2) = 0.25;

    Core::LinAlg::Matrix<Discret::Elements::Internal::num_dim<distype>, 1> xi_centroid =
        Discret::Elements::evaluate_parameter_coordinate_centroid<distype>();

    EXPECT_EQ(xi_centroid, xi_centroid_ref);
  }

  TEST(EvaluateParameterCoordinateCentroid, DisTypeWedge)
  {
    const auto distype = Core::FE::CellType::wedge6;

    Core::LinAlg::Matrix<Discret::Elements::Internal::num_dim<distype>, 1> xi_centroid_ref(
        Core::LinAlg::Initialization::zero);
    xi_centroid_ref(0) = 1.0 / 3.0;
    xi_centroid_ref(1) = 1.0 / 3.0;

    Core::LinAlg::Matrix<Discret::Elements::Internal::num_dim<distype>, 1> xi_centroid =
        Discret::Elements::evaluate_parameter_coordinate_centroid<distype>();

    EXPECT_EQ(xi_centroid, xi_centroid_ref);
  }

  TEST(EvaluateReferenceCoordinateCentroid, DisTypeHex)
  {
    const auto distype = Core::FE::CellType::hex8;

    Core::LinAlg::Matrix<Discret::Elements::Internal::num_dim<distype>, 1>
        reference_coords_centroid_ref(Core::LinAlg::Initialization::zero);

    Discret::Elements::ElementNodes<distype> nodal_coordinates;

    nodal_coordinates.reference_coordinates(0, 0) = 0;
    nodal_coordinates.reference_coordinates(1, 0) = 0;
    nodal_coordinates.reference_coordinates(2, 0) = 0;

    nodal_coordinates.reference_coordinates(0, 1) = 4;
    nodal_coordinates.reference_coordinates(1, 1) = 0;
    nodal_coordinates.reference_coordinates(2, 1) = 0;

    nodal_coordinates.reference_coordinates(0, 2) = 4;
    nodal_coordinates.reference_coordinates(1, 2) = 1;
    nodal_coordinates.reference_coordinates(2, 2) = 0;

    nodal_coordinates.reference_coordinates(0, 3) = 0;
    nodal_coordinates.reference_coordinates(1, 3) = 1;
    nodal_coordinates.reference_coordinates(2, 3) = 0;

    nodal_coordinates.reference_coordinates(0, 4) = 0;
    nodal_coordinates.reference_coordinates(1, 4) = 0;
    nodal_coordinates.reference_coordinates(2, 4) = 2;

    nodal_coordinates.reference_coordinates(0, 5) = 4;
    nodal_coordinates.reference_coordinates(1, 5) = 0;
    nodal_coordinates.reference_coordinates(2, 5) = 2;

    nodal_coordinates.reference_coordinates(0, 6) = 4;
    nodal_coordinates.reference_coordinates(1, 6) = 1;
    nodal_coordinates.reference_coordinates(2, 6) = 2;

    nodal_coordinates.reference_coordinates(0, 7) = 0;
    nodal_coordinates.reference_coordinates(1, 7) = 1;
    nodal_coordinates.reference_coordinates(2, 7) = 2;

    Core::LinAlg::Matrix<Discret::Elements::Internal::num_dim<distype>, 1> x_centroid_ref(
        Core::LinAlg::Initialization::zero);
    x_centroid_ref(0) = 2;
    x_centroid_ref(1) = 0.5;
    x_centroid_ref(2) = 1.0;

    Core::LinAlg::Matrix<Discret::Elements::Internal::num_dim<distype>, 1> x_centroid =
        Discret::Elements::evaluate_reference_coordinate_centroid<distype>(nodal_coordinates);

    EXPECT_EQ(x_centroid, x_centroid_ref);
  }

  TEST(EvaluateReferenceCoordinateCentroid, DisTypeTet)
  {
    const auto distype = Core::FE::CellType::tet4;

    Core::LinAlg::Matrix<Discret::Elements::Internal::num_dim<distype>, 1>
        reference_coords_centroid_ref(Core::LinAlg::Initialization::zero);

    Discret::Elements::ElementNodes<distype> nodal_coordinates;

    nodal_coordinates.reference_coordinates(0, 0) = 0;
    nodal_coordinates.reference_coordinates(1, 0) = 0;
    nodal_coordinates.reference_coordinates(2, 0) = 0;

    nodal_coordinates.reference_coordinates(0, 1) = 1;
    nodal_coordinates.reference_coordinates(1, 1) = 0;
    nodal_coordinates.reference_coordinates(2, 1) = 0;

    nodal_coordinates.reference_coordinates(0, 2) = 0;
    nodal_coordinates.reference_coordinates(1, 2) = 2;
    nodal_coordinates.reference_coordinates(2, 2) = 0;

    nodal_coordinates.reference_coordinates(0, 3) = 0;
    nodal_coordinates.reference_coordinates(1, 3) = 0;
    nodal_coordinates.reference_coordinates(2, 3) = 4;

    Core::LinAlg::Matrix<Discret::Elements::Internal::num_dim<distype>, 1> x_centroid_ref(
        Core::LinAlg::Initialization::zero);
    x_centroid_ref(0) = 0.25;
    x_centroid_ref(1) = 0.5;
    x_centroid_ref(2) = 1.0;

    Core::LinAlg::Matrix<Discret::Elements::Internal::num_dim<distype>, 1> x_centroid =
        Discret::Elements::evaluate_reference_coordinate_centroid<distype>(nodal_coordinates);

    EXPECT_EQ(x_centroid, x_centroid_ref);
  }

  TEST(EvaluateReferenceCoordinateCentroid, DisTypePyramid)
  {
    const auto distype = Core::FE::CellType::pyramid5;

    Core::LinAlg::Matrix<Discret::Elements::Internal::num_dim<distype>, 1>
        reference_coords_centroid_ref(Core::LinAlg::Initialization::zero);

    Discret::Elements::ElementNodes<distype> nodal_coordinates;

    nodal_coordinates.reference_coordinates(0, 0) = -2;
    nodal_coordinates.reference_coordinates(1, 0) = -1;
    nodal_coordinates.reference_coordinates(2, 0) = 0;

    nodal_coordinates.reference_coordinates(0, 1) = 4;
    nodal_coordinates.reference_coordinates(1, 1) = -1;
    nodal_coordinates.reference_coordinates(2, 1) = 0;

    nodal_coordinates.reference_coordinates(0, 2) = 4;
    nodal_coordinates.reference_coordinates(1, 2) = 1;
    nodal_coordinates.reference_coordinates(2, 2) = 0;

    nodal_coordinates.reference_coordinates(0, 3) = -2;
    nodal_coordinates.reference_coordinates(1, 3) = 1;
    nodal_coordinates.reference_coordinates(2, 3) = 0;

    nodal_coordinates.reference_coordinates(0, 4) = 1;
    nodal_coordinates.reference_coordinates(1, 4) = 0;
    nodal_coordinates.reference_coordinates(2, 4) = 4;

    Core::LinAlg::Matrix<Discret::Elements::Internal::num_dim<distype>, 1> x_centroid_ref(
        Core::LinAlg::Initialization::zero);
    x_centroid_ref(0) = 1.0;
    x_centroid_ref(1) = 0.0;
    x_centroid_ref(2) = 1.0;

    Core::LinAlg::Matrix<Discret::Elements::Internal::num_dim<distype>, 1> x_centroid =
        Discret::Elements::evaluate_reference_coordinate_centroid<distype>(nodal_coordinates);

    EXPECT_EQ(x_centroid, x_centroid_ref);
  }

  TEST(EvaluateReferenceCoordinateCentroid, DisTypeWedge)
  {
    const auto distype = Core::FE::CellType::wedge6;

    Core::LinAlg::Matrix<Discret::Elements::Internal::num_dim<distype>, 1>
        reference_coords_centroid_ref(Core::LinAlg::Initialization::zero);

    Discret::Elements::ElementNodes<distype> nodal_coordinates;

    nodal_coordinates.reference_coordinates(0, 0) = 0;
    nodal_coordinates.reference_coordinates(1, 0) = 0;
    nodal_coordinates.reference_coordinates(2, 0) = 0;

    nodal_coordinates.reference_coordinates(0, 1) = 3;
    nodal_coordinates.reference_coordinates(1, 1) = 0;
    nodal_coordinates.reference_coordinates(2, 1) = 0;

    nodal_coordinates.reference_coordinates(0, 2) = 0;
    nodal_coordinates.reference_coordinates(1, 2) = 6;
    nodal_coordinates.reference_coordinates(2, 2) = 0;

    nodal_coordinates.reference_coordinates(0, 3) = 0;
    nodal_coordinates.reference_coordinates(1, 3) = 0;
    nodal_coordinates.reference_coordinates(2, 3) = 1;

    nodal_coordinates.reference_coordinates(0, 4) = 3;
    nodal_coordinates.reference_coordinates(1, 4) = 0;
    nodal_coordinates.reference_coordinates(2, 4) = 1;

    nodal_coordinates.reference_coordinates(0, 5) = 0;
    nodal_coordinates.reference_coordinates(1, 5) = 6;
    nodal_coordinates.reference_coordinates(2, 5) = 1;

    Core::LinAlg::Matrix<Discret::Elements::Internal::num_dim<distype>, 1> x_centroid_ref(
        Core::LinAlg::Initialization::zero);
    x_centroid_ref(0) = 1;
    x_centroid_ref(1) = 2;
    x_centroid_ref(2) = 0.5;

    Core::LinAlg::Matrix<Discret::Elements::Internal::num_dim<distype>, 1> x_centroid =
        Discret::Elements::evaluate_reference_coordinate_centroid<distype>(nodal_coordinates);

    for (int j = 0; j < Discret::Elements::Internal::num_dim<distype>; j++)
    {
      EXPECT_NEAR(x_centroid(j), x_centroid_ref(j), 1e-14);
    }
  }
}  // namespace
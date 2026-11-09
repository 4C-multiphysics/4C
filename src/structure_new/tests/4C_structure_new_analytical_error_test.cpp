// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_solid_ele_calc_lib.hpp"

#include <array>
#include <cmath>

FOUR_C_NAMESPACE_OPEN

namespace
{
  using namespace FourC;

  class DisplacementFunction : public Core::Utils::FunctionOfSpaceTime
  {
   public:
    double evaluate(std::span<const double> x, double, std::size_t component) const override
    {
      return component == 0 ? x[0] * x[0] : 0.0;
    }

    std::vector<double> evaluate_spatial_derivative(
        std::span<const double> x, double, std::size_t component) const override
    {
      std::vector<double> gradient(3, 0.0);
      if (component == 0) gradient[0] = 2.0 * x[0];

      return gradient;
    }

    std::size_t number_components() const override { return 3; }
  };


  TEST(SolidAnalyticalErrorTest, discretizaiton_error)
  {
    constexpr auto celltype = Core::FE::CellType::hex8;
    constexpr int dim = Core::FE::dim<celltype>;
    Discret::Elements::ElementNodes<celltype> element_nodes;

    constexpr std::array<std::array<double, 3>, 8> coordinates{{
        {{0.0, 0.0, 0.0}},
        {{1.0, 0.0, 0.0}},
        {{1.0, 1.0, 0.0}},
        {{0.0, 1.0, 0.0}},
        {{0.0, 0.0, 1.0}},
        {{1.0, 0.0, 1.0}},
        {{1.0, 1.0, 1.0}},
        {{0.0, 1.0, 1.0}},
    }};

    for (int node = 0; node < 8; ++node)
    {
      for (int direction = 0; direction < dim; ++direction)
        element_nodes.reference_coordinates(direction, node) = coordinates[node][direction];

      const double x = coordinates[node][0];
      element_nodes.displacements(0, node) = x * x;
      element_nodes.displacements(1, node) = 0.0;
      element_nodes.displacements(2, node) = 0.0;
    }

    element_nodes.current_coordinates = element_nodes.reference_coordinates;
    element_nodes.current_coordinates.update(1.0, element_nodes.displacements, 1.0);

    const auto integration =
        Core::FE::create_gauss_integration<celltype>(Core::FE::GaussRule3D::hex_27point);
    const DisplacementFunction analytical_function;

    // displacement error in the L2 norm
    {
      double squared_l2_norm_error = 0.0;

      Discret::Elements::for_each_gauss_point<celltype>(element_nodes,
          Discret::Elements::ElementProperties<celltype>{}, integration,
          [&](const auto&, const auto& shape_functions, const auto& jacobian_mapping,
              double integration_factor, int)
          {
            Core::LinAlg::Matrix<dim, 1> reference_coordinates;

            reference_coordinates.multiply_nn(
                element_nodes.reference_coordinates, shape_functions.shapefunctions_);

            squared_l2_norm_error +=
                Discret::Elements::compute_squared_l2_norm_error_at_gauss_point<celltype>(
                    element_nodes, shape_functions, analytical_function,
                    reference_coordinates.as_span(), 0.0) *
                integration_factor;
          });

      EXPECT_NEAR(squared_l2_norm_error, 1.0 / 30.0, 1.0e-12);
      EXPECT_NEAR(std::sqrt(squared_l2_norm_error), 1.0 / std::sqrt(30.0), 1.0e-12);
    }

    // displacement error in the H1 semi norm
    {
      double squared_h1_norm_error = 0.0;

      Discret::Elements::for_each_gauss_point<celltype>(element_nodes,
          Discret::Elements::ElementProperties<celltype>{}, integration,
          [&](const auto&, const auto& shape_functions, const auto& jacobian_mapping,
              double integration_factor, int)
          {
            Core::LinAlg::Matrix<dim, 1> reference_coordinates;

            reference_coordinates.multiply_nn(
                element_nodes.reference_coordinates, shape_functions.shapefunctions_);

            squared_h1_norm_error +=
                Discret::Elements::compute_squared_h1_semi_norm_error_at_gauss_point<celltype>(
                    element_nodes, jacobian_mapping, analytical_function,
                    reference_coordinates.as_span(), 0.0) *
                integration_factor;
          });

      EXPECT_NEAR(squared_h1_norm_error, 1.0 / 3.0, 1.0e-12);
      EXPECT_NEAR(std::sqrt(squared_h1_norm_error), 1.0 / std::sqrt(3.0), 1.0e-12);
    }
  }
}  // namespace

FOUR_C_NAMESPACE_CLOSE
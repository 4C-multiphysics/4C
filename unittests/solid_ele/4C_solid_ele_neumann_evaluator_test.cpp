// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_solid_ele_neumann_evaluator.hpp"

#include "4C_fem_general_element_integration.hpp"
#include "4C_solid_ele_calc_lib_integration.hpp"

namespace
{
  using namespace FourC;

  TEST(PlaneNormalPressure, ReferenceThicknessAndConsistentLinearization)
  {
    constexpr auto celltype = Core::FE::CellType::line2;
    constexpr double pressure = 3.2;
    constexpr double thickness = 2.75;
    Core::LinAlg::Matrix<2, 2> coordinates;
    coordinates(0, 0) = 2.0;
    coordinates(0, 1) = 0.0;
    coordinates(1, 0) = 3.0;
    coordinates(1, 1) = 1.0;

    auto evaluate = [&](const Core::LinAlg::Matrix<2, 2>& evaluated_coordinates,
                        Core::LinAlg::SerialDenseMatrix* load_linearization)
    {
      Core::LinAlg::SerialDenseVector force(4);
      const auto integration = Core::FE::create_gauss_integration<celltype>(
          Discret::Elements::get_gauss_rule_stiffness_matrix<celltype>());
      for (int gp = 0; gp < integration.num_points(); ++gp)
      {
        const auto xi = Core::Elements::evaluate_parameter_coordinate<celltype>(integration, gp);
        const Core::Elements::ElementNodes<celltype, 2> nodes{.coordinates = evaluated_coordinates};
        const auto shape = Core::Elements::evaluate_shape_functions_and_derivs<celltype>(xi, nodes);
        Discret::Elements::add_normal_pressure_load<celltype>(shape, evaluated_coordinates,
            pressure * integration.weight(gp), thickness, 2, force, load_linearization);
      }
      return force;
    };

    Core::LinAlg::SerialDenseMatrix load_linearization(4, 4);
    const auto force = evaluate(coordinates, &load_linearization);
    EXPECT_NEAR(force[0] + force[2], pressure * thickness, 1.0e-13);
    EXPECT_NEAR(force[1] + force[3], -pressure * thickness, 1.0e-13);

    constexpr double perturbation = 1.0e-7;
    for (int column = 0; column < 4; ++column)
    {
      auto coordinates_plus = coordinates;
      auto coordinates_minus = coordinates;
      coordinates_plus(column / 2, column % 2) += perturbation;
      coordinates_minus(column / 2, column % 2) -= perturbation;
      const auto force_plus = evaluate(coordinates_plus, nullptr);
      const auto force_minus = evaluate(coordinates_minus, nullptr);
      for (int row = 0; row < 4; ++row)
      {
        const double finite_difference =
            (force_plus[row] - force_minus[row]) / (2.0 * perturbation);
        EXPECT_NEAR(-load_linearization(row, column), finite_difference, 1.0e-7);
      }
    }
  }
}  // namespace

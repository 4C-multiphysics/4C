// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_mortar_element.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

namespace
{
  using namespace FourC;

  class TestMortarNode : public Mortar::Node
  {
   public:
    using Mortar::Node::Node;

    void initialize_mortar_data() { modata_ = std::make_shared<Mortar::NodeDataContainer>(); }
  };

  class MortarElementNormal : public testing::Test
  {
   protected:
    void SetUp() override
    {
      const std::array<std::array<double, 3>, 3> coordinates{{
          {1.0, 2.0, 3.0},
          {4.0, 2.0, 4.0},
          {1.0, 6.0, 5.0},
      }};

      const std::array<int, 3> node_ids{0, 1, 2};

      for (std::size_t node = 0; node < nodes_.size(); ++node)
      {
        nodes_[node] = std::make_unique<TestMortarNode>(node_ids[node],
            std::span<const double>(coordinates[node]), 0,
            std::vector<int>{
                static_cast<int>(3 * node),
                static_cast<int>(3 * node + 1),
                static_cast<int>(3 * node + 2),
            },
            true);

        nodes_[node]->initialize_mortar_data();
      }

      element_ = std::make_unique<Mortar::Element>(
          0, 0, Core::FE::CellType::tri3, 3, node_ids.data(), true);

      std::array<Core::Nodes::Node*, 3> node_pointers{
          nodes_[0].get(),
          nodes_[1].get(),
          nodes_[2].get(),
      };

      element_->build_nodal_pointers(node_pointers.data());
    }

    std::array<std::unique_ptr<TestMortarNode>, 3> nodes_;
    std::unique_ptr<Mortar::Element> element_;
  };

  // Test that the element normal is the normalized cross product of its parametric tangent vectors.
  TEST_F(MortarElementNormal, ComputesUnitNormalAndSurfaceJacobian)
  {
    const std::array<double, 2> xi = {1.0 / 3.0, 1.0 / 3.0};
    std::array<double, 3> normal = {0.0, 0.0, 0.0};

    const double surface_jacobian = element_->compute_unit_normal_at_xi(xi.data(), normal.data());

    EXPECT_NEAR(surface_jacobian, 14.0, 1.0e-14);
    EXPECT_NEAR(normal[0], -2.0 / 7.0, 1.0e-14);
    EXPECT_NEAR(normal[1], -3.0 / 7.0, 1.0e-14);
    EXPECT_NEAR(normal[2], 6.0 / 7.0, 1.0e-14);
  }

  // Test that the normal factor reverses the normal direction but does not change its length.
  TEST_F(MortarElementNormal, NormalFactorReversesOrientation)
  {
    element_->normal_fac() = -1.0;

    const std::array<double, 2> xi = {1.0 / 3.0, 1.0 / 3.0};
    std::array<double, 3> normal = {0.0, 0.0, 0.0};

    const double surface_jacobian = element_->compute_unit_normal_at_xi(xi.data(), normal.data());

    EXPECT_NEAR(surface_jacobian, 14.0, 1.0e-14);
    EXPECT_NEAR(normal[0], 2.0 / 7.0, 1.0e-14);
    EXPECT_NEAR(normal[1], 3.0 / 7.0, 1.0e-14);
    EXPECT_NEAR(normal[2], -6.0 / 7.0, 1.0e-14);
  }


  // Test that the analytical normal linearization agrees with central finite differences.
  TEST_F(MortarElementNormal, UnitNormalLinearization)
  {
    // positive normal
    {
      element_->normal_fac() = 1.0;

      const std::array<double, 2> xi = {0.23, 0.31};

      std::vector<Core::Gen::Pairedvector<int, double>> analytical_derivative;
      element_->deriv_unit_normal_at_xi(xi.data(), analytical_derivative);

      constexpr double perturbation = 1.0e-7;
      constexpr double tolerance = 5.0e-8;

      for (std::size_t node = 0; node < nodes_.size(); ++node)
      {
        for (int dim = 0; dim < 3; ++dim)
        {
          double& coordinate = nodes_[node]->xspatial()[dim];
          const double original_coordinate = coordinate;

          coordinate = original_coordinate + perturbation;
          double normal_plus[3];
          element_->compute_unit_normal_at_xi(xi.data(), normal_plus);

          coordinate = original_coordinate - perturbation;
          double normal_minus[3];
          element_->compute_unit_normal_at_xi(xi.data(), normal_minus);

          coordinate = original_coordinate;
          const int dof = nodes_[node]->dofs()[dim];

          for (int component = 0; component < 3; ++component)
          {
            const double finite_difference =
                (normal_plus[component] - normal_minus[component]) / (2.0 * perturbation);
            EXPECT_NEAR(analytical_derivative[component][dof], finite_difference, tolerance);
          }
        }
      }
    }

    // negative normal
    {
      element_->normal_fac() = -1.0;

      const std::array<double, 2> xi = {0.49, 0.81};

      std::vector<Core::Gen::Pairedvector<int, double>> analytical_derivative;
      element_->deriv_unit_normal_at_xi(xi.data(), analytical_derivative);

      constexpr double perturbation = 1.0e-7;
      constexpr double tolerance = 5.0e-8;

      for (std::size_t node = 0; node < nodes_.size(); ++node)
      {
        for (int dim = 0; dim < 3; ++dim)
        {
          double& coordinate = nodes_[node]->xspatial()[dim];
          const double original_coordinate = coordinate;

          coordinate = original_coordinate + perturbation;
          double normal_plus[3];
          element_->compute_unit_normal_at_xi(xi.data(), normal_plus);

          coordinate = original_coordinate - perturbation;
          double normal_minus[3];
          element_->compute_unit_normal_at_xi(xi.data(), normal_minus);

          coordinate = original_coordinate;
          const int dof = nodes_[node]->dofs()[dim];

          for (int component = 0; component < 3; ++component)
          {
            const double finite_difference =
                (normal_plus[component] - normal_minus[component]) / (2.0 * perturbation);
            EXPECT_NEAR(analytical_derivative[component][dof], finite_difference, tolerance);
          }
        }
      }
    }
  }
}  // namespace
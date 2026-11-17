// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_contact_element.hpp"
#include "4C_contact_input.hpp"
#include "4C_contact_integrator.hpp"
#include "4C_contact_node.hpp"
#include "4C_contact_wear_input.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_mortar_input.hpp"
#include "4C_utils_parameter_list.hpp"

#include <array>
#include <memory>
#include <numeric>
#include <vector>

namespace
{
  using namespace FourC;

  class TestableContactIntegrator : public CONTACT::Integrator
  {
   public:
    using CONTACT::Integrator::gp_dm;
    using CONTACT::Integrator::Integrator;
  };

  Teuchos::ParameterList create_integrator_parameters(const Mortar::ShapeFcn multiplier_shape)
  {
    Teuchos::ParameterList params;

    params.set("DIMENSION", 3);

    Core::Utils::add_enum_class_to_parameter_list("LM_SHAPEFCN", Mortar::shape_standard, params);
    Core::Utils::add_enum_class_to_parameter_list("LM_QUAD", Mortar::lagmult_quad, params);
    Core::Utils::add_enum_class_to_parameter_list("ALGORITHM", Mortar::algorithm_mortar, params);
    Core::Utils::add_enum_class_to_parameter_list(
        "STRATEGY", CONTACT::SolvingStrategy::lagmult, params);
    Core::Utils::add_enum_class_to_parameter_list("WEARLAW", Wear::wear_none, params);
    Core::Utils::add_enum_class_to_parameter_list("INTTYPE", Mortar::inttype_segments, params);

    params.set("GP_SLIP_INCR", false);
    params.set("CPP_NORMALS", false);
    params.set("SSWEAR", false);
    params.set("SSSLIP", 0.0);
    params.set("NONSMOOTH_CONTACT_SURFACE", false);
    params.set("NONSMOOTH_GEOMETRIES", false);
    params.set("NUMGP_PER_DIM", 0);

    return params;
  }

  void integrate_interface(const Mortar::ShapeFcn multiplier_shape,
      CONTACT::Element& source_element, CONTACT::Element& target_element)
  {
    Teuchos::ParameterList params = create_integrator_parameters(multiplier_shape);
    TestableContactIntegrator integrator(params, Core::FE::CellType::quad4, MPI_COMM_WORLD);

    constexpr int number_of_nodes = 4;

    Core::LinAlg::SerialDenseVector source_values(number_of_nodes);
    Core::LinAlg::SerialDenseVector target_values(number_of_nodes);
    Core::LinAlg::SerialDenseVector multiplier_values(number_of_nodes);

    Core::LinAlg::SerialDenseMatrix source_derivatives(number_of_nodes, 2);
    Core::LinAlg::SerialDenseMatrix target_derivatives(number_of_nodes, 2);
    Core::LinAlg::SerialDenseMatrix multiplier_derivatives(number_of_nodes, 2);

    bool boundary_modification = false;

    for (int gp = 0; gp < integrator.n_gp(); ++gp)
    {
      double xi[2] = {integrator.coordinate(gp, 0), integrator.coordinate(gp, 1)};

      source_element.evaluate_shape(xi, source_values, source_derivatives, number_of_nodes);
      target_element.evaluate_shape(xi, target_values, target_derivatives, number_of_nodes);
      source_element.evaluate_shape_lag_mult(
          multiplier_shape, xi, multiplier_values, multiplier_derivatives, number_of_nodes);

      double jacobian = source_element.jacobian(xi);
      double quadrature_weight = integrator.weight(gp);

      integrator.gp_dm(source_element, target_element, multiplier_values, source_values,
          target_values, jacobian, quadrature_weight, boundary_modification);
    }
  }


  class InterfaceScalingTest : public testing::Test
  {
   protected:
    static constexpr int number_of_nodes_per_element = 4;

    std::shared_ptr<Core::FE::Discretization> discretization_;

    std::array<std::shared_ptr<CONTACT::Node>, number_of_nodes_per_element> source_nodes_;
    std::array<std::shared_ptr<CONTACT::Node>, number_of_nodes_per_element> target_nodes_;

    std::shared_ptr<CONTACT::Element> source_element_;
    std::shared_ptr<CONTACT::Element> target_element_;

    void SetUp() override
    {
      discretization_ =
          std::make_shared<Core::FE::Discretization>("interface_scaling_test", MPI_COMM_WORLD, 3);

      // Counter-clockwise quad4 ordering.
      const std::array<std::array<double, 3>, number_of_nodes_per_element> coordinates = {
          std::array<double, 3>{0.0, 0.0, 0.0}, std::array<double, 3>{1.0, 0.0, 0.0},
          std::array<double, 3>{1.0, 1.0, 0.0}, std::array<double, 3>{0.0, 1.0, 0.0}};

      const int owner = 0;

      for (int i = 0; i < number_of_nodes_per_element; ++i)
      {
        const int source_node_id = i;
        const int target_node_id = number_of_nodes_per_element + i;

        const std::vector<int> source_dofs = {
            3 * source_node_id, 3 * source_node_id + 1, 3 * source_node_id + 2};
        const std::vector<int> target_dofs = {
            3 * target_node_id, 3 * target_node_id + 1, 3 * target_node_id + 2};

        source_nodes_[i] = std::make_shared<CONTACT::Node>(
            source_node_id, coordinates[i], owner, source_dofs, true, false);
        target_nodes_[i] = std::make_shared<CONTACT::Node>(
            target_node_id, coordinates[i], owner, target_dofs, false, false);

        discretization_->add_node(coordinates[i], source_node_id, source_nodes_[i]);
        discretization_->add_node(coordinates[i], target_node_id, target_nodes_[i]);
      }

      const std::array<int, number_of_nodes_per_element> source_node_ids = {0, 1, 2, 3};
      const std::array<int, number_of_nodes_per_element> target_node_ids = {4, 5, 6, 7};

      source_element_ = std::make_shared<CONTACT::Element>(0, owner, Core::FE::CellType::quad4,
          number_of_nodes_per_element, source_node_ids.data(), true, false);
      target_element_ = std::make_shared<CONTACT::Element>(1, owner, Core::FE::CellType::quad4,
          number_of_nodes_per_element, target_node_ids.data(), false, false);

      discretization_->add_element(source_element_);
      discretization_->add_element(target_element_);
      discretization_->fill_complete(Core::FE::OptionsFillComplete::none());

      for (auto& node : source_nodes_) node->initialize_data_container();
      for (auto& node : target_nodes_) node->initialize_data_container();

      source_element_->initialize_data_container();
      target_element_->initialize_data_container();
    }
  };


  TEST_F(InterfaceScalingTest, IntegratesUnitQuadWithDualMultipliers)
  {
    integrate_interface(Mortar::shape_dual, *source_element_, *target_element_);

    constexpr double tolerance = 1.0e-12;
    constexpr double expected_nodal_weight = 1.0 / 4.0;

    const std::array<std::array<double, 4>, 4> expected_d = {{
        {{0.25, 0.00, 0.00, 0.00}},
        {{0.00, 0.25, 0.00, 0.00}},
        {{0.00, 0.00, 0.25, 0.00}},
        {{0.00, 0.00, 0.00, 0.25}},
    }};

    const auto get_d_entry = [](CONTACT::Node& row_node, const int column_node_id)
    {
      double entry = 0.0;
      for (const auto& [stored_column_id, value] : row_node.mo_data().get_d())
        if (stored_column_id == column_node_id) entry += value;

      return entry;
    };

    double total_interface_weight = 0.0;
    for (std::size_t row = 0; row < source_nodes_.size(); ++row)
    {
      CONTACT::Node& source_node = *source_nodes_[row];
      const double interface_weight = source_node.mo_data().interface_weight();

      double d_row_sum = 0.0;
      for (std::size_t column = 0; column < source_nodes_.size(); ++column)
      {
        const double actual_d_entry = get_d_entry(source_node, source_nodes_[column]->id());
        EXPECT_NEAR(actual_d_entry, expected_d[row][column], tolerance)
            << "Unexpected D entry in row " << row << ", column " << column;
        d_row_sum += actual_d_entry;
      }

      EXPECT_NEAR(interface_weight, expected_nodal_weight, tolerance);
      EXPECT_NEAR(interface_weight, d_row_sum, tolerance);
      EXPECT_NEAR(interface_weight, expected_d[row][row], tolerance);

      total_interface_weight += interface_weight;
    }

    EXPECT_NEAR(total_interface_weight, 1.0, tolerance);
  }


  TEST_F(InterfaceScalingTest, IntegratesUnitQuadWithStandardMultipliers)
  {
    integrate_interface(Mortar::shape_standard, *source_element_, *target_element_);

    constexpr double tolerance = 1.0e-12;
    constexpr double expected_nodal_weight = 1.0 / 4.0;

    const std::array<std::array<double, 4>, 4> expected_d = {{
        {{4.0 / 36.0, 2.0 / 36.0, 1.0 / 36.0, 2.0 / 36.0}},
        {{2.0 / 36.0, 4.0 / 36.0, 2.0 / 36.0, 1.0 / 36.0}},
        {{1.0 / 36.0, 2.0 / 36.0, 4.0 / 36.0, 2.0 / 36.0}},
        {{2.0 / 36.0, 1.0 / 36.0, 2.0 / 36.0, 4.0 / 36.0}},
    }};

    const auto get_d_entry = [](CONTACT::Node& row_node, const int column_node_id)
    {
      double entry = 0.0;

      for (const auto& [stored_column_id, value] : row_node.mo_data().get_d())
        if (stored_column_id == column_node_id) entry += value;

      return entry;
    };

    double total_interface_weight = 0.0;
    for (std::size_t row = 0; row < source_nodes_.size(); ++row)
    {
      CONTACT::Node& source_node = *source_nodes_[row];
      const double interface_weight = source_node.mo_data().interface_weight();

      double d_row_sum = 0.0;
      for (std::size_t column = 0; column < source_nodes_.size(); ++column)
      {
        const double actual_d_entry = get_d_entry(source_node, source_nodes_[column]->id());
        EXPECT_NEAR(actual_d_entry, expected_d[row][column], tolerance)
            << "Unexpected D entry in row " << row << ", column " << column;
        d_row_sum += actual_d_entry;
      }

      EXPECT_NEAR(interface_weight, expected_nodal_weight, tolerance);
      EXPECT_NEAR(interface_weight, d_row_sum, tolerance);
      EXPECT_NE(interface_weight, expected_d[row][row]);

      total_interface_weight += interface_weight;
    }

    EXPECT_NEAR(total_interface_weight, 1.0, tolerance);
  }

}  // namespace
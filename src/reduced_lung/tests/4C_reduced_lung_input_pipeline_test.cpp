// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_fem_discretization.hpp"
#include "4C_io_input_file.hpp"
#include "4C_io_input_parameter_container.hpp"
#include "4C_rebalance.hpp"
#include "4C_reduced_lung_discretization_helpers.hpp"
#include "4C_reduced_lung_helpers.hpp"
#include "4C_reduced_lung_input.hpp"
#include "4C_unittest_utils_support_files_test.hpp"

#include <mpi.h>

#include <filesystem>
#include <numbers>
#include <string>
#include <vector>

namespace
{
  using namespace FourC;
  using namespace FourC::ReducedLung;

  ReducedLungParameters load_parameters_from_file(const std::filesystem::path& input_path)
  {
    std::vector<Core::IO::InputSpec> specs{ReducedLung::valid_parameters()};
    Core::IO::InputFile input_file(specs, {}, MPI_COMM_WORLD);
    input_file.read(input_path);

    Core::IO::InputParameterContainer container;
    input_file.match_section("reduced_dimensional_lung", container);

    return container.get<ReducedLungParameters>("reduced_dimensional_lung");
  }

  TEST(ReducedLungInputPipelineTest, ReadsInputBuildsDiscretizationAndModels)
  {
    const auto input_path =
        TESTING::get_support_file_path("4C_reduced_lung_3_aw_2_tu_topology.4C.yaml");
    const auto params = load_parameters_from_file(input_path);

    {
      SCOPED_TRACE("Input parsing");
      EXPECT_EQ(params.lung_tree.topology.num_nodes, 6);
      EXPECT_EQ(params.lung_tree.topology.num_elements, 5);

      const auto coords_0 = params.lung_tree.topology.node_coordinates.at(0, "node_coordinates");
      ASSERT_EQ(coords_0.size(), 3u);
      EXPECT_DOUBLE_EQ(coords_0[0], 0.0);
      EXPECT_DOUBLE_EQ(coords_0[1], 0.0);
      EXPECT_DOUBLE_EQ(coords_0[2], 0.0);

      const auto nodes_0 = params.lung_tree.topology.element_nodes.at(0, "element_nodes");
      EXPECT_EQ(nodes_0, (std::vector<int>{1, 2}));

      EXPECT_EQ(params.lung_tree.element_type.at(0, "element_type"),
          ReducedLungParameters::LungTree::ElementType::Airway);
      EXPECT_EQ(params.lung_tree.element_type.at(3, "element_type"),
          ReducedLungParameters::LungTree::ElementType::TerminalUnit);
    }

    Core::FE::Discretization discretization("reduced_lung_pipeline_test", MPI_COMM_WORLD, 3);
    Core::Rebalance::RebalanceParameters rebalance_parameters;

    build_discretization_from_topology(
        discretization, params.lung_tree.topology, rebalance_parameters);
    discretization.fill_complete(Core::FE::OptionsFillComplete{
        .assign_degrees_of_freedom = true,
        .init_elements = true,
        .do_boundary_conditions = false,
    });

    {
      SCOPED_TRACE("Discretization build");
      EXPECT_EQ(discretization.num_global_nodes(), params.lung_tree.topology.num_nodes);
      EXPECT_EQ(discretization.num_global_elements(), params.lung_tree.topology.num_elements);

      for (const auto& element : discretization.my_row_element_range())
      {
        EXPECT_EQ(&element.user_element()->element_type(),
            &Discret::Elements::ReducedLungLineType::instance());
      }

      if (discretization.element_row_map()->lid(0) != -1)
      {
        auto* ele = discretization.l_row_element(discretization.element_row_map()->lid(0));
        ASSERT_EQ(ele->num_node(), 2);
        EXPECT_EQ(ele->node_ids()[0], 0);
        EXPECT_EQ(ele->node_ids()[1], 1);
      }
    }

    AirwayContainer airways;
    TerminalUnitContainer terminal_units;
    for (const auto& element : discretization.my_row_element_range())
    {
      auto* user_ele = element.user_element();
      const int element_id = element.global_id();
      const int local_element_id = discretization.element_row_map()->lid(element_id);
      const auto element_kind = params.lung_tree.element_type.at(element_id, "element_type");

      if (element_kind == ReducedLungParameters::LungTree::ElementType::Airway)
      {
        const auto flow_model_type =
            params.lung_tree.airways.flow_model.resistance_type.at(element_id, "resistance_type");
        const auto wall_model_type =
            params.lung_tree.airways.wall_model_type.at(element_id, "wall_model_type");
        add_airway_with_model_selection(
            airways, user_ele, local_element_id, params, flow_model_type, wall_model_type);
      }
      else
      {
        const auto rheological_model_type =
            params.lung_tree.terminal_units.rheological_model.rheological_model_type.at(
                element_id, "rheological_model_type");
        const auto elasticity_model_type =
            params.lung_tree.terminal_units.elasticity_model.elasticity_model_type.at(
                element_id, "elasticity_model_type");
        add_terminal_unit_with_model_selection(terminal_units, user_ele, local_element_id,
            params.lung_tree.terminal_units, rheological_model_type, elasticity_model_type);
      }
    }

    {
      SCOPED_TRACE("Model creation");
      ASSERT_EQ(airways.models.size(), 1u);
      ASSERT_EQ(terminal_units.models.size(), 1u);

      const auto& airway_model = airways.models.front();
      EXPECT_TRUE(std::holds_alternative<Airways::LinearResistive>(airway_model.flow_model));
      EXPECT_TRUE(std::holds_alternative<Airways::RigidWall>(airway_model.wall_model));
      EXPECT_EQ(airway_model.data.n_state_equations, 1);
      EXPECT_EQ(airway_model.data.global_element_id.size(), 3u);
      ASSERT_EQ(airway_model.data.ref_length.size(), 3u);
      ASSERT_EQ(airway_model.data.ref_area.size(), 3u);
      const auto& linear_flow = std::get<Airways::LinearResistive>(airway_model.flow_model);
      EXPECT_EQ(linear_flow.has_inertia, (std::vector<bool>{false, false, false}));
      for (size_t i = 0; i < airway_model.data.global_element_id.size(); ++i)
      {
        EXPECT_DOUBLE_EQ(airway_model.data.ref_length[i], 1.0);
        const int element_id = airway_model.data.global_element_id[i];
        const double radius = params.lung_tree.airways.radius.at(element_id, "radius");
        const double expected_area = radius * radius * std::numbers::pi;
        EXPECT_NEAR(airway_model.data.ref_area[i], expected_area, 1e-12);
      }

      const auto& terminal_model = terminal_units.models.front();
      EXPECT_TRUE(
          std::holds_alternative<TerminalUnits::KelvinVoigt>(terminal_model.rheological_model));
      EXPECT_TRUE(
          std::holds_alternative<TerminalUnits::LinearElasticity>(terminal_model.elasticity_model));
      EXPECT_EQ(terminal_model.data.global_element_id.size(), 2u);
      ASSERT_EQ(terminal_model.data.volume_v.size(), 2u);
      const double expected_volume = (4.0 / 3.0) * std::numbers::pi;
      EXPECT_NEAR(terminal_model.data.volume_v[0], expected_volume, 1e-12);
      EXPECT_NEAR(terminal_model.data.volume_v[1], expected_volume, 1e-12);
      const auto& linear_elasticity =
          std::get<TerminalUnits::LinearElasticity>(terminal_model.elasticity_model);
      EXPECT_EQ(linear_elasticity.elasticity_E, (std::vector<double>{1.0, 1.0}));
      const auto& kelvin_voigt =
          std::get<TerminalUnits::KelvinVoigt>(terminal_model.rheological_model);
      EXPECT_EQ(kelvin_voigt.viscosity_eta, (std::vector<double>{0.0, 0.0}));
    }
  }
}  // namespace

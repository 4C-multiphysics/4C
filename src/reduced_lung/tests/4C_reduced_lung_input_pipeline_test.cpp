// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_elementtype.hpp"
#include "4C_rebalance.hpp"
#include "4C_reduced_lung_helpers.hpp"
#include "4C_reduced_lung_input.hpp"

#include <mpi.h>

#include <numbers>
#include <unordered_map>
#include <vector>

namespace
{
  using namespace FourC;
  using namespace FourC::ReducedLung;

  ReducedLungParameters make_parameters()
  {
    ReducedLungParameters params{};
    params.air_properties = {
        .density = 1.176e-06,
        .dynamic_viscosity = 1.79105e-05,
    };
    params.dynamics = {
        .time_increment = 1.0,
        .number_of_steps = 1,
        .restart_every = 1,
        .results_every = 1,
        .linear_solver = 1,
    };

    params.lung_tree.topology.num_nodes = 6;
    params.lung_tree.topology.num_elements = 5;
    params.lung_tree.topology.node_coordinates =
        Core::IO::InputField<std::vector<double>>(std::unordered_map<int, std::vector<double>>{
            {1, {0.0, 0.0, 0.0}},
            {2, {1.0, 0.0, 0.0}},
            {3, {1.0, 1.0, 0.0}},
            {4, {1.0, -1.0, 0.0}},
            {5, {2.0, 1.0, 0.0}},
            {6, {2.0, -1.0, 0.0}},
        });
    params.lung_tree.topology.element_nodes =
        Core::IO::InputField<std::vector<int>>(std::unordered_map<int, std::vector<int>>{
            {1, {1, 2}},
            {2, {2, 3}},
            {3, {2, 4}},
            {4, {3, 5}},
            {5, {4, 6}},
        });
    params.lung_tree.element_type =
        Core::IO::InputField<ReducedLungParameters::LungTree::ElementType>(
            std::unordered_map<int, ReducedLungParameters::LungTree::ElementType>{
                {1, ReducedLungParameters::LungTree::ElementType::Airway},
                {2, ReducedLungParameters::LungTree::ElementType::Airway},
                {3, ReducedLungParameters::LungTree::ElementType::Airway},
                {4, ReducedLungParameters::LungTree::ElementType::TerminalUnit},
                {5, ReducedLungParameters::LungTree::ElementType::TerminalUnit},
            });
    params.lung_tree.generation = Core::IO::InputField<int>(std::unordered_map<int, int>{
        {1, 0},
        {2, 1},
        {3, 1},
        {4, -1},
        {5, -1},
    });

    params.lung_tree.airways.radius =
        Core::IO::InputField<double>(std::unordered_map<int, double>{{1, 1.0}, {2, 0.8}, {3, 0.6}});
    params.lung_tree.airways.flow_model.resistance_type =
        Core::IO::InputField<ReducedLungParameters::LungTree::Airways::FlowModel::ResistanceType>(
            ReducedLungParameters::LungTree::Airways::FlowModel::ResistanceType::Linear);
    params.lung_tree.airways.flow_model.include_inertia = Core::IO::InputField<bool>(false);
    params.lung_tree.airways.wall_model_type =
        Core::IO::InputField<ReducedLungParameters::LungTree::Airways::WallModelType>(
            ReducedLungParameters::LungTree::Airways::WallModelType::Rigid);

    params.lung_tree.terminal_units.rheological_model.rheological_model_type = Core::IO::InputField<
        ReducedLungParameters::LungTree::TerminalUnits::RheologicalModel::RheologicalModelType>(
        ReducedLungParameters::LungTree::TerminalUnits::RheologicalModel::RheologicalModelType::
            KelvinVoigt);
    params.lung_tree.terminal_units.rheological_model.kelvin_voigt.viscosity_kelvin_voigt_eta =
        Core::IO::InputField<double>(0.0);
    params.lung_tree.terminal_units.elasticity_model.elasticity_model_type = Core::IO::InputField<
        ReducedLungParameters::LungTree::TerminalUnits::ElasticityModel::ElasticityModelType>(
        ReducedLungParameters::LungTree::TerminalUnits::ElasticityModel::ElasticityModelType::
            Linear);
    params.lung_tree.terminal_units.elasticity_model.linear.elasticity_e =
        Core::IO::InputField<double>(1.0);

    params.boundary_conditions.num_conditions = 3;
    params.boundary_conditions.bc_type =
        Core::IO::InputField<ReducedLungParameters::BoundaryConditions::Type>(
            std::unordered_map<int, ReducedLungParameters::BoundaryConditions::Type>{
                {1, ReducedLungParameters::BoundaryConditions::Type::Pressure},
                {2, ReducedLungParameters::BoundaryConditions::Type::Flow},
                {3, ReducedLungParameters::BoundaryConditions::Type::Flow},
            });
    params.boundary_conditions.node_id = Core::IO::InputField<int>(std::unordered_map<int, int>{
        {1, 1},
        {2, 5},
        {3, 9},
    });
    params.boundary_conditions.value_source =
        ReducedLungParameters::BoundaryConditions::ValueSource::bc_value;
    params.boundary_conditions.value = Core::IO::InputField<double>(
        std::unordered_map<int, double>{{1, 1.0}, {2, 0.0}, {3, -0.5}});

    return params;
  }

  TEST(ReducedLungInputPipelineTest, BuildsDiscretizationAndModels)
  {
    const auto params = make_parameters();

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
        ASSERT_NE(element.user_element(), nullptr);
        EXPECT_EQ(element.user_element()->element_type().name(), "PureGeometryElementType");
        EXPECT_EQ(element.user_element()->shape(), Core::FE::CellType::line2);
      }

      if (discretization.element_row_map()->lid(0) != -1)
      {
        auto* ele = discretization.l_row_element(discretization.element_row_map()->lid(0));
        ASSERT_EQ(ele->num_node(), 2);
        EXPECT_EQ(ele->node_ids()[0], 0);
        EXPECT_EQ(ele->node_ids()[1], 1);
      }
    }

    Airways::AirwayContainer airways;
    TerminalUnits::TerminalUnitContainer terminal_units;
    for (const auto& element : discretization.my_row_element_range())
    {
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
            airways, element_id, local_element_id, params, flow_model_type, wall_model_type);
      }
      else
      {
        const auto rheological_model_type =
            params.lung_tree.terminal_units.rheological_model.rheological_model_type.at(
                element_id, "rheological_model_type");
        const auto elasticity_model_type =
            params.lung_tree.terminal_units.elasticity_model.elasticity_model_type.at(
                element_id, "elasticity_model_type");
        add_terminal_unit_with_model_selection(terminal_units, element_id, local_element_id, params,
            rheological_model_type, elasticity_model_type);
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

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
#include "4C_reduced_lung_airways_model_registry.hpp"
#include "4C_reduced_lung_helpers.hpp"
#include "4C_reduced_lung_input.hpp"
#include "4C_reduced_lung_terminal_unit_model_registry.hpp"

#include <mpi.h>

#include <array>
#include <cmath>
#include <numbers>
#include <unordered_map>
#include <vector>

namespace
{
  using namespace FourC;
  using namespace FourC::ReducedLung;

  //! Two generations of airways, each ending in a terminal unit.
  const std::vector<std::array<double, 3>> node_coordinates{{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
      {1.0, 1.0, 0.0}, {1.0, -1.0, 0.0}, {2.0, 1.0, 0.0}, {2.0, -1.0, 0.0}};
  const std::vector<std::array<int, 2>> element_nodes{{0, 1}, {1, 2}, {1, 3}, {2, 4}, {3, 5}};

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
        .max_nonlinear_iterations = 10,
        .nonlinear_residual_tolerance = 1e-8,
        .nonlinear_increment_tolerance = 1e-10,
    };

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

    using InputBc = ReducedLungParameters::BoundaryConditions;
    const auto condition = [](int id, int function_id)
    { return InputBc::Definition{.id = id, .function_id = function_id}; };
    params.boundary_conditions.pressure = {condition(1, 1)};
    params.boundary_conditions.flow = {condition(2, 2), condition(3, 3)};

    return params;
  }

  TEST(ReducedLungInputPipelineTest, BuildsDiscretizationAndModels)
  {
    const auto params = make_parameters();

    Core::FE::Discretization discretization("reduced_lung_pipeline_test", MPI_COMM_WORLD, 3);
    Core::Rebalance::RebalanceParameters rebalance_parameters;

    build_discretization_from_nodes_and_elements(
        discretization, node_coordinates, element_nodes, rebalance_parameters);
    discretization.fill_complete(Core::FE::OptionsFillComplete{
        .assign_degrees_of_freedom = true,
        .init_elements = true,
        .do_boundary_conditions = false,
    });

    {
      SCOPED_TRACE("Discretization build");
      EXPECT_EQ(discretization.num_global_nodes(), static_cast<int>(node_coordinates.size()));
      EXPECT_EQ(discretization.num_global_elements(), static_cast<int>(element_nodes.size()));

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

    using ElementType = ReducedLungParameters::LungTree::ElementType;
    const std::vector element_types{ElementType::Airway, ElementType::Airway, ElementType::Airway,
        ElementType::TerminalUnit, ElementType::TerminalUnit};

    Airways::AirwayContainer airways;
    TerminalUnits::TerminalUnitContainer terminal_units;
    for (const auto& element : discretization.my_row_element_range())
    {
      const int element_id = element.global_id();
      const int local_element_id = discretization.element_row_map()->lid(element_id);
      const auto element_kind = element_types[element_id];
      const auto& nodes = element_nodes[element_id];
      const auto& x_in = node_coordinates[nodes[0]];
      const auto& x_out = node_coordinates[nodes[1]];
      const double ref_length =
          std::hypot(x_in[0] - x_out[0], x_in[1] - x_out[1], x_in[2] - x_out[2]);

      if (element_kind == ReducedLungParameters::LungTree::ElementType::Airway)
      {
        const auto flow_model_type =
            params.lung_tree.airways.flow_model.resistance_type.at(element_id, "resistance_type");
        const auto wall_model_type =
            params.lung_tree.airways.wall_model_type.at(element_id, "wall_model_type");
        Airways::ModelRegistry::add_airway_with_model_selection(airways, element_id,
            local_element_id, ref_length, params, flow_model_type, wall_model_type);
      }
      else
      {
        const auto rheological_model_type =
            params.lung_tree.terminal_units.rheological_model.rheological_model_type.at(
                element_id, "rheological_model_type");
        const auto elasticity_model_type =
            params.lung_tree.terminal_units.elasticity_model.elasticity_model_type.at(
                element_id, "elasticity_model_type");
        TerminalUnits::ModelRegistry::add_terminal_unit_with_model_selection(terminal_units,
            element_id, local_element_id, ref_length, params, rheological_model_type,
            elasticity_model_type);
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

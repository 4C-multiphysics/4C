// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_reduced_lung_tree_metadata.hpp"

#include "4C_io_input_field.hpp"
#include "4C_linalg_map.hpp"
#include "4C_reduced_lung_test_utils_test.hpp"
#include "4C_utils_exceptions.hpp"

#include <mpi.h>

#include <array>
#include <map>
#include <memory>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace
{
  using namespace FourC;
  using namespace FourC::ReducedLung;

  using ElementType = ReducedLungParameters::LungTree::ElementType;

  struct TreeMetadataFixture
  {
    std::unique_ptr<Core::FE::Discretization> discretization;
    std::vector<ElementType> element_types;
    std::map<int, int> first_global_dof_of_ele;
    std::map<int, int> global_dof_per_ele;
    Airways::AirwayContainer airways;
    TerminalUnits::TerminalUnitContainer terminal_units;
    Junctions::ConnectionData connections;
    Junctions::BifurcationData bifurcations;
    BoundaryConditions::BoundaryConditionContainer boundary_conditions;
    std::unique_ptr<Core::LinAlg::Map> row_map;
    std::unique_ptr<Core::LinAlg::Map> locally_relevant_dof_map;

    [[nodiscard]] ReducedLungTreeMetadata build() const
    {
      return build_reduced_lung_tree_metadata(ReducedLungTreeMetadataContext{
          .discretization = *discretization,
          .element_types = element_types,
          .first_global_dof_of_ele = first_global_dof_of_ele,
          .global_dof_per_ele = global_dof_per_ele,
          .airways = airways,
          .terminal_units = terminal_units,
          .connections = connections,
          .bifurcations = bifurcations,
          .boundary_conditions = boundary_conditions,
          .row_map = *row_map,
          .locally_relevant_dof_map = *locally_relevant_dof_map,
      });
    }
  };

  void set_identity_maps(TreeMetadataFixture& fixture, int size)
  {
    fixture.row_map = std::make_unique<Core::LinAlg::Map>(-1, size, 0, MPI_COMM_WORLD);

    std::vector<int> global_ids(static_cast<std::size_t>(size));
    std::iota(global_ids.begin(), global_ids.end(), 0);
    fixture.locally_relevant_dof_map = std::make_unique<Core::LinAlg::Map>(
        -1, global_ids.size(), global_ids.data(), 0, MPI_COMM_WORLD);
  }

  void add_pressure_boundary_conditions(
      TreeMetadataFixture& fixture, const std::vector<std::array<int, 5>>& entries)
  {
    BoundaryConditions::BoundaryConditionModel model;
    model.constrained_variable = BoundaryConditions::ConstrainedVariable::Pressure;

    for (std::size_t i = 0; i < entries.size(); ++i)
    {
      const auto& entry = entries[i];
      const int node_id = entry[0];
      const int element_id = entry[1];
      const int global_dof_id = entry[2];
      const int local_equation_id = entry[3];
      model.add_condition(node_id, element_id, static_cast<int>(i), global_dof_id);
      model.data.local_equation_id.back() = local_equation_id;
      model.data.global_equation_id.back() = local_equation_id;
      model.data.local_dof_id.back() = global_dof_id;
    }

    fixture.boundary_conditions.models.push_back(model);
  }

  void set_uniform_dof_maps(TreeMetadataFixture& fixture, int num_elements)
  {
    for (int element_id = 0; element_id < num_elements; ++element_id)
    {
      fixture.first_global_dof_of_ele[element_id] = 3 * element_id;
      fixture.global_dof_per_ele[element_id] = 3;
    }
  }

  TreeMetadataFixture make_connection_fixture()
  {
    TreeMetadataFixture fixture;
    fixture.discretization = TestUtils::make_chain_discretization("tree_metadata_connection", 2);
    fixture.element_types = {ElementType::Airway, ElementType::TerminalUnit};

    set_uniform_dof_maps(fixture, 2);

    Airways::AirwayModel airway_model;
    airway_model.data.global_element_id = {0};
    airway_model.data.local_row_id = {0};
    airway_model.data.n_state_equations = 1;
    fixture.airways.models.push_back(airway_model);

    TerminalUnits::TerminalUnitModel terminal_unit_model;
    terminal_unit_model.data.global_element_id = {1};
    terminal_unit_model.data.local_row_id = {1};
    fixture.terminal_units.models.push_back(terminal_unit_model);

    fixture.connections.add_connection(0, 0, 1, {1, 3, 2, 5});
    fixture.connections.first_local_equation_id[0] = 2;
    fixture.connections.first_global_equation_id[0] = 2;
    fixture.connections.local_dof_ids[0] = {1, 3, 2, 5};

    add_pressure_boundary_conditions(fixture, std::vector<std::array<int, 5>>{
                                                  {0, 0, 0, 4, 0},
                                                  {2, 1, 4, 5, 1},
                                              });

    set_identity_maps(fixture, 6);
    return fixture;
  }

  TreeMetadataFixture make_bifurcation_fixture()
  {
    TreeMetadataFixture fixture;
    fixture.discretization =
        TestUtils::make_bifurcation_discretization("tree_metadata_bifurcation");
    fixture.element_types = {
        ElementType::Airway, ElementType::TerminalUnit, ElementType::TerminalUnit};

    set_uniform_dof_maps(fixture, 3);

    Airways::AirwayModel airway_model;
    airway_model.data.global_element_id = {0};
    airway_model.data.local_row_id = {0};
    airway_model.data.n_state_equations = 1;
    fixture.airways.models.push_back(airway_model);

    TerminalUnits::TerminalUnitModel terminal_unit_model;
    terminal_unit_model.data.global_element_id = {1, 2};
    terminal_unit_model.data.local_row_id = {1, 2};
    fixture.terminal_units.models.push_back(terminal_unit_model);

    fixture.bifurcations.add_bifurcation(0, 0, 1, 2, {1, 3, 6, 2, 5, 8});
    fixture.bifurcations.first_local_equation_id[0] = 3;
    fixture.bifurcations.first_global_equation_id[0] = 3;
    fixture.bifurcations.local_dof_ids[0] = {1, 3, 6, 2, 5, 8};

    add_pressure_boundary_conditions(fixture, std::vector<std::array<int, 5>>{
                                                  {0, 0, 0, 6, 0},
                                                  {2, 1, 4, 7, 1},
                                                  {3, 2, 7, 8, 2},
                                              });

    set_identity_maps(fixture, 9);
    return fixture;
  }

  TreeMetadataFixture make_cycle_fixture()
  {
    TreeMetadataFixture fixture;
    fixture.discretization = TestUtils::make_line2_discretization("tree_metadata_cycle",
        {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {3.0, 0.0, 0.0}},
        {{0, 1}, {2, 3}, {3, 2}});
    fixture.element_types.assign(3, ElementType::Airway);

    set_uniform_dof_maps(fixture, 3);

    Airways::AirwayModel airway_model;
    airway_model.data.global_element_id = {0, 1, 2};
    airway_model.data.local_row_id = {0, 1, 2};
    airway_model.data.n_state_equations = 1;
    fixture.airways.models.push_back(airway_model);

    set_identity_maps(fixture, 9);
    return fixture;
  }

  TreeMetadataFixture make_unsupported_branch_degree_fixture()
  {
    TreeMetadataFixture fixture;
    fixture.discretization = TestUtils::make_line2_discretization("tree_metadata_branch_degree",
        {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {2.0, 1.0, 0.0}, {2.0, -1.0, 0.0}},
        {{0, 1}, {1, 2}, {1, 3}, {1, 4}});
    fixture.element_types.assign(4, ElementType::Airway);

    set_uniform_dof_maps(fixture, 4);

    Airways::AirwayModel airway_model;
    airway_model.data.global_element_id = {0, 1, 2, 3};
    airway_model.data.local_row_id = {0, 1, 2, 3};
    airway_model.data.n_state_equations = 1;
    fixture.airways.models.push_back(airway_model);

    set_identity_maps(fixture, 12);
    return fixture;
  }

  TEST(ReducedLungTreeMetadataTests, BuildsConnectionMetadata)
  {
    const auto fixture = make_connection_fixture();
    const auto metadata = fixture.build();

    ASSERT_EQ(metadata.elements.size(), 2u);
    EXPECT_EQ(metadata.root_element_index, 0);
    EXPECT_EQ(metadata.root_node_id, 0);
    EXPECT_EQ(metadata.num_global_dofs, 6);
    EXPECT_EQ(metadata.num_global_equations, 6);
    EXPECT_EQ(metadata.airway_element_indices, (std::vector<int>{0}));
    EXPECT_EQ(metadata.terminal_unit_element_indices, (std::vector<int>{1}));

    EXPECT_EQ(metadata.elements[0].kind, TreeElementKind::Airway);
    EXPECT_EQ(metadata.elements[0].child_count, 1);
    EXPECT_EQ(metadata.elements[0].child_element_indices[0], 1);
    EXPECT_EQ(metadata.elements[1].kind, TreeElementKind::TerminalUnit);
    EXPECT_TRUE(metadata.elements[1].is_leaf());
    EXPECT_EQ(metadata.elements[1].global_dof_ids, (std::vector<int>{3, 4, 5}));

    ASSERT_EQ(metadata.junctions.size(), 1u);
    EXPECT_EQ(metadata.junctions[0].kind, TreeJunctionKind::Connection);
    EXPECT_EQ(metadata.junctions[0].parent_element_index, 0);
    EXPECT_EQ(metadata.junctions[0].child_element_indices[0], 1);
    EXPECT_EQ(metadata.junctions[0].num_equations, 2);
    EXPECT_EQ(metadata.junctions[0].global_dof_ids, (std::vector<int>{1, 3, 2, 5}));

    EXPECT_EQ(metadata.top_down_layers, (std::vector<std::vector<int>>{{0}, {1}}));
    EXPECT_EQ(metadata.bottom_up_layers, (std::vector<std::vector<int>>{{1}, {0}}));
  }

  TEST(ReducedLungTreeMetadataTests, BuildsBifurcationMetadataWithTerminalUnitLeaves)
  {
    const auto fixture = make_bifurcation_fixture();
    const auto metadata = fixture.build();

    ASSERT_EQ(metadata.elements.size(), 3u);
    EXPECT_EQ(metadata.airway_element_indices, (std::vector<int>{0}));
    EXPECT_EQ(metadata.terminal_unit_element_indices, (std::vector<int>{1, 2}));

    EXPECT_EQ(metadata.elements[0].child_count, 2);
    EXPECT_EQ(metadata.elements[0].child_element_indices, (std::array<int, 2>{1, 2}));
    EXPECT_EQ(metadata.elements[1].parent_element_index, 0);
    EXPECT_EQ(metadata.elements[2].parent_element_index, 0);

    ASSERT_EQ(metadata.junctions.size(), 1u);
    EXPECT_EQ(metadata.junctions[0].kind, TreeJunctionKind::Bifurcation);
    EXPECT_EQ(metadata.junctions[0].num_equations, 3);
    EXPECT_EQ(metadata.junctions[0].global_dof_ids, (std::vector<int>{1, 3, 6, 2, 5, 8}));

    ASSERT_EQ(metadata.boundary_conditions.size(), 3u);
    EXPECT_EQ(metadata.boundary_conditions[0].side, TreeBoundarySide::Inlet);
    EXPECT_EQ(metadata.boundary_conditions[1].side, TreeBoundarySide::Outlet);
    EXPECT_EQ(metadata.boundary_conditions[2].side, TreeBoundarySide::Outlet);

    EXPECT_EQ(metadata.top_down_layers, (std::vector<std::vector<int>>{{0}, {1, 2}}));
    EXPECT_EQ(metadata.bottom_up_layers, (std::vector<std::vector<int>>{{1, 2}, {0}}));
  }

  TEST(ReducedLungTreeMetadataTests, ThrowsForDirectedCycle)
  {
    const auto fixture = make_cycle_fixture();
    EXPECT_THROW(fixture.build(), Core::Exception);
  }

  TEST(ReducedLungTreeMetadataTests, ThrowsForUnsupportedBranchDegree)
  {
    const auto fixture = make_unsupported_branch_degree_fixture();
    EXPECT_THROW(fixture.build(), Core::Exception);
  }

  TEST(ReducedLungTreeMetadataTests, ThrowsForMissingRootBoundary)
  {
    auto fixture = make_connection_fixture();
    auto& data = fixture.boundary_conditions.models.front().data;
    data.node_id[0] = 2;
    data.global_element_id[0] = 1;
    data.global_dof_id[0] = 4;
    data.local_dof_id[0] = 4;

    EXPECT_THROW(fixture.build(), Core::Exception);
  }
}  // namespace

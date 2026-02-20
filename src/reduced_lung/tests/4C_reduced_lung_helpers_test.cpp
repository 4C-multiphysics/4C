// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_reduced_lung_helpers.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_io_input_field.hpp"
#include "4C_linalg_map.hpp"
#include "4C_rebalance.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <map>
#include <memory>
#include <unordered_map>
#include <variant>
#include <vector>

namespace
{
  using namespace FourC;
  using namespace FourC::ReducedLung;

  void skip_if_parallel()
  {
    int comm_size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    if (comm_size != 1)
    {
      GTEST_SKIP() << "This test currently assumes serial map ownership.";
    }
  }

  ReducedLungParameters make_setup_model_parameters()
  {
    ReducedLungParameters params{};
    params.air_properties = {
        .density = 1.176e-06,
        .dynamic_viscosity = 1.79105e-05,
    };

    params.lung_tree.topology.num_nodes = 4;
    params.lung_tree.topology.num_elements = 3;
    params.lung_tree.topology.node_coordinates =
        Core::IO::InputField<std::vector<double>>(std::unordered_map<int, std::vector<double>>{
            {1, {0.0, 0.0, 0.0}},
            {2, {1.0, 0.0, 0.0}},
            {3, {2.0, 0.0, 0.0}},
            {4, {3.0, 0.0, 0.0}},
        });
    params.lung_tree.topology.element_nodes =
        Core::IO::InputField<std::vector<int>>(std::unordered_map<int, std::vector<int>>{
            {1, {1, 2}},
            {2, {2, 3}},
            {3, {3, 4}},
        });

    params.lung_tree.element_type =
        Core::IO::InputField<ReducedLungParameters::LungTree::ElementType>(
            std::unordered_map<int, ReducedLungParameters::LungTree::ElementType>{
                {1, ReducedLungParameters::LungTree::ElementType::Airway},
                {2, ReducedLungParameters::LungTree::ElementType::Airway},
                {3, ReducedLungParameters::LungTree::ElementType::TerminalUnit},
            });

    params.lung_tree.airways.radius =
        Core::IO::InputField<double>(std::unordered_map<int, double>{{1, 1.0}, {2, 0.9}});
    params.lung_tree.airways.flow_model.resistance_type =
        Core::IO::InputField<ReducedLungParameters::LungTree::Airways::FlowModel::ResistanceType>(
            std::unordered_map<int,
                ReducedLungParameters::LungTree::Airways::FlowModel::ResistanceType>{
                {1, ReducedLungParameters::LungTree::Airways::FlowModel::ResistanceType::Linear},
                {2, ReducedLungParameters::LungTree::Airways::FlowModel::ResistanceType::Linear},
            });
    params.lung_tree.airways.flow_model.include_inertia =
        Core::IO::InputField<bool>(std::unordered_map<int, bool>{{1, false}, {2, true}});
    params.lung_tree.airways.wall_model_type =
        Core::IO::InputField<ReducedLungParameters::LungTree::Airways::WallModelType>(
            std::unordered_map<int, ReducedLungParameters::LungTree::Airways::WallModelType>{
                {1, ReducedLungParameters::LungTree::Airways::WallModelType::Rigid},
                {2, ReducedLungParameters::LungTree::Airways::WallModelType::KelvinVoigt},
            });
    params.lung_tree.airways.wall_model.kelvin_voigt.elasticity.wall_poisson_ratio =
        Core::IO::InputField<double>(std::unordered_map<int, double>{{2, 0.3}});
    params.lung_tree.airways.wall_model.kelvin_voigt.elasticity.wall_elasticity =
        Core::IO::InputField<double>(std::unordered_map<int, double>{{2, 50000.0}});
    params.lung_tree.airways.wall_model.kelvin_voigt.elasticity.wall_thickness =
        Core::IO::InputField<double>(std::unordered_map<int, double>{{2, 0.001}});
    params.lung_tree.airways.wall_model.kelvin_voigt.viscosity.viscous_time_constant =
        Core::IO::InputField<double>(std::unordered_map<int, double>{{2, 0.01}});
    params.lung_tree.airways.wall_model.kelvin_voigt.viscosity.viscous_phase_shift =
        Core::IO::InputField<double>(std::unordered_map<int, double>{{2, 0.0}});

    params.lung_tree.terminal_units.rheological_model.rheological_model_type = Core::IO::InputField<
        ReducedLungParameters::LungTree::TerminalUnits::RheologicalModel::RheologicalModelType>(
        std::unordered_map<int,
            ReducedLungParameters::LungTree::TerminalUnits::RheologicalModel::RheologicalModelType>{
            {3, ReducedLungParameters::LungTree::TerminalUnits::RheologicalModel::
                    RheologicalModelType::KelvinVoigt}});
    params.lung_tree.terminal_units.rheological_model.kelvin_voigt.viscosity_kelvin_voigt_eta =
        Core::IO::InputField<double>(std::unordered_map<int, double>{{3, 0.0}});
    params.lung_tree.terminal_units.elasticity_model.elasticity_model_type = Core::IO::InputField<
        ReducedLungParameters::LungTree::TerminalUnits::ElasticityModel::ElasticityModelType>(
        std::unordered_map<int,
            ReducedLungParameters::LungTree::TerminalUnits::ElasticityModel::ElasticityModelType>{
            {3, ReducedLungParameters::LungTree::TerminalUnits::ElasticityModel::
                    ElasticityModelType::Linear}});
    params.lung_tree.terminal_units.elasticity_model.linear.elasticity_e =
        Core::IO::InputField<double>(std::unordered_map<int, double>{{3, 2.0}});

    return params;
  }

  ReducedLungParameters::LungTree::Topology make_bifurcation_topology()
  {
    ReducedLungParameters::LungTree::Topology topology{};
    topology.num_nodes = 4;
    topology.num_elements = 3;
    topology.node_coordinates =
        Core::IO::InputField<std::vector<double>>(std::unordered_map<int, std::vector<double>>{
            {1, {0.0, 0.0, 0.0}},
            {2, {1.0, 0.0, 0.0}},
            {3, {2.0, 1.0, 0.0}},
            {4, {2.0, -1.0, 0.0}},
        });
    topology.element_nodes =
        Core::IO::InputField<std::vector<int>>(std::unordered_map<int, std::vector<int>>{
            {1, {1, 2}},
            {2, {2, 3}},
            {3, {2, 4}},
        });
    return topology;
  }

  std::unique_ptr<Core::FE::Discretization> make_discretization_from_topology(
      const ReducedLungParameters::LungTree::Topology& topology, const char* name)
  {
    auto discretization = std::make_unique<Core::FE::Discretization>(name, MPI_COMM_WORLD, 3);
    Core::Rebalance::RebalanceParameters rebalance_parameters;
    build_discretization_from_topology(*discretization, topology, rebalance_parameters);
    discretization->fill_complete(Core::FE::OptionsFillComplete{
        .assign_degrees_of_freedom = true,
        .init_elements = true,
        .do_boundary_conditions = false,
    });
    return discretization;
  }

  TEST(ReducedLungHelpersTests, CreateGlobalDofMapsBuildsOffsetsInElementOrder)
  {
    skip_if_parallel();

    const std::map<int, int> local_dof_per_ele{{7, 3}, {1, 4}, {5, 2}};
    std::map<int, int> global_dof_per_ele;
    std::map<int, int> first_global_dof_of_ele;

    create_global_dof_maps(
        local_dof_per_ele, MPI_COMM_WORLD, global_dof_per_ele, first_global_dof_of_ele);

    EXPECT_EQ(global_dof_per_ele, local_dof_per_ele);
    EXPECT_EQ(first_global_dof_of_ele, (std::map<int, int>{{1, 0}, {5, 4}, {7, 6}}));
  }

  TEST(ReducedLungHelpersTests, AssignGlobalDofIdsToModelsAssignsExpectedLayouts)
  {
    Airways::AirwayContainer airways;
    Airways::AirwayModel rigid_airway_model;
    rigid_airway_model.data.global_element_id = {10, 30};
    rigid_airway_model.data.n_state_equations = 1;
    airways.models.push_back(rigid_airway_model);

    Airways::AirwayModel compliant_airway_model;
    compliant_airway_model.data.global_element_id = {20};
    compliant_airway_model.data.n_state_equations = 2;
    airways.models.push_back(compliant_airway_model);

    TerminalUnits::TerminalUnitContainer terminal_units;
    TerminalUnits::TerminalUnitModel terminal_model;
    terminal_model.data.global_element_id = {40, 50};
    terminal_units.models.push_back(terminal_model);

    const std::map<int, int> first_global_dof_of_ele{{10, 0}, {20, 3}, {30, 7}, {40, 10}, {50, 13}};

    assign_global_dof_ids_to_models(first_global_dof_of_ele, airways, terminal_units);

    ASSERT_EQ(airways.models.size(), 2u);
    ASSERT_EQ(terminal_units.models.size(), 1u);

    const auto& rigid_data = airways.models[0].data;
    EXPECT_EQ(rigid_data.gid_p1, (std::vector<int>{0, 7}));
    EXPECT_EQ(rigid_data.gid_p2, (std::vector<int>{1, 8}));
    EXPECT_EQ(rigid_data.gid_q1, (std::vector<int>{2, 9}));
    EXPECT_TRUE(rigid_data.gid_q2.empty());

    const auto& compliant_data = airways.models[1].data;
    EXPECT_EQ(compliant_data.gid_p1, (std::vector<int>{3}));
    EXPECT_EQ(compliant_data.gid_p2, (std::vector<int>{4}));
    EXPECT_EQ(compliant_data.gid_q1, (std::vector<int>{5}));
    EXPECT_EQ(compliant_data.gid_q2, (std::vector<int>{6}));

    const auto& terminal_data = terminal_units.models[0].data;
    EXPECT_EQ(terminal_data.gid_p1, (std::vector<int>{10, 13}));
    EXPECT_EQ(terminal_data.gid_p2, (std::vector<int>{11, 14}));
    EXPECT_EQ(terminal_data.gid_q, (std::vector<int>{12, 15}));
  }

  TEST(ReducedLungHelpersTests, AirwaysAssignLocalEquationAndDofIds)
  {
    Airways::AirwayContainer airways;

    Airways::AirwayModel rigid_airway_model;
    rigid_airway_model.data.global_element_id = {10, 30};
    rigid_airway_model.data.n_state_equations = 1;
    rigid_airway_model.data.gid_p1 = {10, 30};
    rigid_airway_model.data.gid_p2 = {11, 31};
    rigid_airway_model.data.gid_q1 = {12, 32};
    airways.models.push_back(rigid_airway_model);

    Airways::AirwayModel compliant_airway_model;
    compliant_airway_model.data.global_element_id = {20};
    compliant_airway_model.data.n_state_equations = 2;
    compliant_airway_model.data.gid_p1 = {20};
    compliant_airway_model.data.gid_p2 = {21};
    compliant_airway_model.data.gid_q1 = {22};
    compliant_airway_model.data.gid_q2 = {23};
    airways.models.push_back(compliant_airway_model);

    int n_local_equations = 5;
    Airways::assign_local_equation_ids(airways, n_local_equations);

    EXPECT_EQ(airways.models[0].data.local_row_id, (std::vector<int>{5, 6}));
    EXPECT_EQ(airways.models[1].data.local_row_id, (std::vector<int>{7}));
    EXPECT_EQ(n_local_equations, 9);

    const std::array<int, 10> global_dofs{20, 21, 22, 23, 10, 11, 12, 30, 31, 32};
    const Core::LinAlg::Map locally_relevant_dof_map(
        -1, global_dofs.size(), global_dofs.data(), 0, MPI_COMM_WORLD);
    Airways::assign_local_dof_ids(locally_relevant_dof_map, airways);

    EXPECT_EQ(airways.models[0].data.lid_p1, (std::vector<int>{4, 7}));
    EXPECT_EQ(airways.models[0].data.lid_p2, (std::vector<int>{5, 8}));
    EXPECT_EQ(airways.models[0].data.lid_q1, (std::vector<int>{6, 9}));
    EXPECT_TRUE(airways.models[0].data.lid_q2.empty());

    EXPECT_EQ(airways.models[1].data.lid_p1, (std::vector<int>{0}));
    EXPECT_EQ(airways.models[1].data.lid_p2, (std::vector<int>{1}));
    EXPECT_EQ(airways.models[1].data.lid_q1, (std::vector<int>{2}));
    EXPECT_EQ(airways.models[1].data.lid_q2, (std::vector<int>{3}));
  }

  TEST(ReducedLungHelpersTests, TerminalUnitsAssignLocalEquationAndDofIds)
  {
    TerminalUnits::TerminalUnitContainer terminal_units;

    TerminalUnits::TerminalUnitModel terminal_model_a;
    terminal_model_a.data.global_element_id = {100, 200};
    terminal_model_a.data.gid_p1 = {1000, 2000};
    terminal_model_a.data.gid_p2 = {1001, 2001};
    terminal_model_a.data.gid_q = {1002, 2002};
    terminal_units.models.push_back(terminal_model_a);

    TerminalUnits::TerminalUnitModel terminal_model_b;
    terminal_model_b.data.global_element_id = {300};
    terminal_model_b.data.gid_p1 = {3000};
    terminal_model_b.data.gid_p2 = {3001};
    terminal_model_b.data.gid_q = {3002};
    terminal_units.models.push_back(terminal_model_b);

    int n_local_equations = 3;
    TerminalUnits::assign_local_equation_ids(terminal_units, n_local_equations);

    EXPECT_EQ(terminal_units.models[0].data.local_row_id, (std::vector<int>{3, 4}));
    EXPECT_EQ(terminal_units.models[1].data.local_row_id, (std::vector<int>{5}));
    EXPECT_EQ(n_local_equations, 6);

    const std::array<int, 9> global_dofs{3000, 3001, 3002, 1000, 1001, 1002, 2000, 2001, 2002};
    const Core::LinAlg::Map locally_relevant_dof_map(
        -1, global_dofs.size(), global_dofs.data(), 0, MPI_COMM_WORLD);
    TerminalUnits::assign_local_dof_ids(locally_relevant_dof_map, terminal_units);

    EXPECT_EQ(terminal_units.models[0].data.lid_p1, (std::vector<int>{3, 6}));
    EXPECT_EQ(terminal_units.models[0].data.lid_p2, (std::vector<int>{4, 7}));
    EXPECT_EQ(terminal_units.models[0].data.lid_q, (std::vector<int>{5, 8}));

    EXPECT_EQ(terminal_units.models[1].data.lid_p1, (std::vector<int>{0}));
    EXPECT_EQ(terminal_units.models[1].data.lid_p2, (std::vector<int>{1}));
    EXPECT_EQ(terminal_units.models[1].data.lid_q, (std::vector<int>{2}));
  }

  TEST(ReducedLungHelpersTests, CreateLocalElementModelsBuildsModelContainers)
  {
    skip_if_parallel();

    const auto params = make_setup_model_parameters();
    auto discretization =
        make_discretization_from_topology(params.lung_tree.topology, "local_element_models_test");

    Airways::AirwayContainer airways;
    TerminalUnits::TerminalUnitContainer terminal_units;
    std::map<int, int> dof_per_ele;
    int n_airways = -1;
    int n_terminal_units = -1;

    create_local_element_models(
        *discretization, params, airways, terminal_units, dof_per_ele, n_airways, n_terminal_units);

    EXPECT_EQ(n_airways, 2);
    EXPECT_EQ(n_terminal_units, 1);
    EXPECT_EQ(dof_per_ele, (std::map<int, int>{{0, 3}, {1, 4}, {2, 3}}));

    ASSERT_EQ(airways.models.size(), 2u);
    bool found_rigid_model = false;
    bool found_kelvin_voigt_model = false;
    for (const auto& model : airways.models)
    {
      ASSERT_EQ(model.data.number_of_elements(), 1u);
      if (std::holds_alternative<Airways::RigidWall>(model.wall_model))
      {
        found_rigid_model = true;
        EXPECT_EQ(model.data.n_state_equations, 1);
        EXPECT_EQ(model.data.global_element_id[0], 0);
      }
      if (std::holds_alternative<Airways::KelvinVoigtWall>(model.wall_model))
      {
        found_kelvin_voigt_model = true;
        EXPECT_EQ(model.data.n_state_equations, 2);
        EXPECT_EQ(model.data.global_element_id[0], 1);
      }
    }
    EXPECT_TRUE(found_rigid_model);
    EXPECT_TRUE(found_kelvin_voigt_model);

    ASSERT_EQ(terminal_units.models.size(), 1u);
    EXPECT_EQ(terminal_units.models[0].data.global_element_id, (std::vector<int>{2}));
  }

  TEST(ReducedLungHelpersTests, CreateGlobalEleIdsPerNodeCollectsAdjacency)
  {
    skip_if_parallel();

    auto discretization =
        make_discretization_from_topology(make_bifurcation_topology(), "global_ele_ids_test");
    auto global_ele_ids_per_node = create_global_ele_ids_per_node(*discretization, MPI_COMM_WORLD);

    ASSERT_EQ(global_ele_ids_per_node.size(), 4u);

    auto sorted_unique = [](std::vector<int> ids)
    {
      std::sort(ids.begin(), ids.end());
      ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
      return ids;
    };

    EXPECT_EQ(sorted_unique(global_ele_ids_per_node.at(0)), (std::vector<int>{0}));
    EXPECT_EQ(sorted_unique(global_ele_ids_per_node.at(1)), (std::vector<int>{0, 1, 2}));
    EXPECT_EQ(sorted_unique(global_ele_ids_per_node.at(2)), (std::vector<int>{1}));
    EXPECT_EQ(sorted_unique(global_ele_ids_per_node.at(3)), (std::vector<int>{2}));
  }
}  // namespace

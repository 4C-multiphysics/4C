// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_reduced_lung_tree_linear_solver.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_linear_solver_method.hpp"
#include "4C_rebalance.hpp"
#include "4C_reduced_lung_boundary_conditions.hpp"
#include "4C_reduced_lung_helpers.hpp"
#include "4C_reduced_lung_junctions.hpp"
#include "4C_reduced_lung_newton_linear_solver.hpp"
#include "4C_reduced_lung_newton_solver.hpp"
#include "4C_reduced_lung_solver_profiles.hpp"
#include "4C_reduced_lung_terminal_unit.hpp"
#include "4C_reduced_lung_test_utils_test.hpp"
#include "4C_reduced_lung_tree_linearization.hpp"
#include "4C_reduced_lung_tree_metadata.hpp"
#include "4C_utils_function_manager.hpp"
#include "4C_utils_function_of_time.hpp"

#include <mpi.h>
#include <Teuchos_ParameterList.hpp>

#include <algorithm>
#include <any>
#include <array>
#include <cstdlib>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{
  using namespace FourC;
  using namespace FourC::ReducedLung;

  using ElementType = ReducedLungParameters::LungTree::ElementType;
  using ResistanceType = ReducedLungParameters::LungTree::Airways::FlowModel::ResistanceType;
  using WallModelType = ReducedLungParameters::LungTree::Airways::WallModelType;
  using RheologyType =
      ReducedLungParameters::LungTree::TerminalUnits::RheologicalModel::RheologicalModelType;
  using ElasticityType =
      ReducedLungParameters::LungTree::TerminalUnits::ElasticityModel::ElasticityModelType;

  bool require_tree_solver_simd_from_environment()
  {
    const char* const value = std::getenv("FOUR_C_REDUCED_LUNG_REQUIRE_SIMD_TREE_SOLVER");
    return value != nullptr && std::string(value) != "" && std::string(value) != "0";
  }

  struct TestProblem
  {
    ReducedLungParameters parameters;
    std::vector<std::array<double, 3>> node_coordinates;
    std::vector<std::array<int, 2>> element_nodes;
    std::vector<ElementType> element_types;
    std::map<int, std::vector<int>> bc_nodes;
  };

  void set_pressure_boundaries(
      TestProblem& problem, const std::vector<int>& outlet_nodes, bool root_flow = false)
  {
    using InputBc = ReducedLungParameters::BoundaryConditions;
    const auto condition = [](int id, int function_id)
    { return InputBc::FromFunctionDefinition{.id = id, .function_id = function_id}; };

    problem.parameters.boundary_conditions.pressure = {condition(2, 2)};
    problem.parameters.boundary_conditions.flow.clear();
    if (root_flow)
    {
      problem.parameters.boundary_conditions.flow = {condition(1, 1)};
    }
    else
    {
      problem.parameters.boundary_conditions.pressure.insert(
          problem.parameters.boundary_conditions.pressure.begin(), condition(1, 1));
    }
    problem.bc_nodes = {{1, {0}}, {2, outlet_nodes}};
  }

  Core::Utils::FunctionManager make_function_manager(
      const std::vector<std::string>& function_definitions)
  {
    Core::Utils::FunctionManager function_manager;
    std::vector<std::any> functions;
    functions.reserve(function_definitions.size());
    for (const auto& definition : function_definitions)
    {
      functions.emplace_back(std::shared_ptr<Core::Utils::FunctionOfTime>(
          std::make_shared<Core::Utils::SymbolicFunctionOfTime>(
              std::vector<std::string>{definition},
              std::vector<std::shared_ptr<Core::Utils::FunctionVariable>>{})));
    }
    function_manager.set_functions(functions);
    return function_manager;
  }

  ReducedLungParameters::Dynamics make_dynamics(double dt)
  {
    return ReducedLungParameters::Dynamics{
        .time_increment = dt,
        .number_of_steps = 1,
        .restart_every = 1,
        .results_every = 1,
        .linear_solver = 1,
        .max_nonlinear_iterations = 10,
        .nonlinear_residual_tolerance = 1.0e-8,
        .nonlinear_increment_tolerance = 1.0e-10,
    };
  }

  void set_common_air_properties(ReducedLungParameters& params)
  {
    params.air_properties = {
        .density = 1.176e-06,
        .dynamic_viscosity = 1.79105e-05,
    };
  }

  void set_airway_model(ReducedLungParameters& params, const std::unordered_map<int, double>& radii,
      ResistanceType resistance_type, WallModelType wall_model_type)
  {
    params.lung_tree.airways.radius = Core::IO::InputField<double>(radii);
    params.lung_tree.airways.flow_model.resistance_type =
        Core::IO::InputField<ResistanceType>(resistance_type);
    params.lung_tree.airways.flow_model.resistance_model.non_linear.turbulence_factor_gamma =
        Core::IO::InputField<double>(0.6);
    params.lung_tree.airways.flow_model.include_inertia = Core::IO::InputField<bool>(false);
    params.lung_tree.airways.wall_model_type = Core::IO::InputField<WallModelType>(wall_model_type);
    params.lung_tree.airways.wall_model.kelvin_voigt.elasticity.wall_poisson_ratio =
        Core::IO::InputField<double>(0.3);
    params.lung_tree.airways.wall_model.kelvin_voigt.elasticity.wall_elasticity =
        Core::IO::InputField<double>(50000.0);
    params.lung_tree.airways.wall_model.kelvin_voigt.elasticity.wall_thickness =
        Core::IO::InputField<double>(0.001);
    params.lung_tree.airways.wall_model.kelvin_voigt.viscosity.viscous_time_constant =
        Core::IO::InputField<double>(0.01);
    params.lung_tree.airways.wall_model.kelvin_voigt.viscosity.viscous_phase_shift =
        Core::IO::InputField<double>(0.1);
  }

  void set_linear_rigid_airway_model(
      ReducedLungParameters& params, const std::unordered_map<int, double>& radii)
  {
    set_airway_model(params, radii, ResistanceType::Linear, WallModelType::Rigid);
  }

  void set_terminal_unit_model(
      ReducedLungParameters& params, RheologyType rheology_type, ElasticityType elasticity_type)
  {
    params.lung_tree.terminal_units.v0 = Core::IO::InputField<double>(1.0);
    params.lung_tree.terminal_units.rheological_model.rheological_model_type =
        Core::IO::InputField<RheologyType>(rheology_type);
    params.lung_tree.terminal_units.rheological_model.kelvin_voigt.viscosity_kelvin_voigt_eta =
        Core::IO::InputField<double>(1.0);
    params.lung_tree.terminal_units.rheological_model.four_element_maxwell
        .viscosity_kelvin_voigt_eta = Core::IO::InputField<double>(0.1);
    params.lung_tree.terminal_units.rheological_model.four_element_maxwell.viscosity_maxwell_eta_m =
        Core::IO::InputField<double>(0.5);
    params.lung_tree.terminal_units.rheological_model.four_element_maxwell.elasticity_maxwell_e_m =
        Core::IO::InputField<double>(1.5);
    params.lung_tree.terminal_units.elasticity_model.elasticity_model_type =
        Core::IO::InputField<ElasticityType>(elasticity_type);
    params.lung_tree.terminal_units.elasticity_model.linear.elasticity_e =
        Core::IO::InputField<double>(1.0);
    params.lung_tree.terminal_units.elasticity_model.ogden.ogden_parameter_kappa =
        Core::IO::InputField<double>(1.0);
    params.lung_tree.terminal_units.elasticity_model.ogden.ogden_parameter_beta =
        Core::IO::InputField<double>(-8.0);
  }

  void set_linear_terminal_unit_model(ReducedLungParameters& params)
  {
    set_terminal_unit_model(params, RheologyType::KelvinVoigt, ElasticityType::Linear);
  }

  TestProblem make_single_terminal_unit_parameters(double dt)
  {
    TestProblem problem;
    auto& params = problem.parameters;
    set_common_air_properties(params);
    params.dynamics = make_dynamics(dt);

    problem.node_coordinates = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
    problem.element_nodes = {{0, 1}};
    problem.element_types = {ElementType::TerminalUnit};
    set_linear_terminal_unit_model(params);
    set_pressure_boundaries(problem, {1});
    return problem;
  }

  TestProblem make_serial_airway_parameters(double dt)
  {
    TestProblem problem;
    auto& params = problem.parameters;
    set_common_air_properties(params);
    params.dynamics = make_dynamics(dt);

    problem.node_coordinates = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {3.0, 0.0, 0.0}};
    problem.element_nodes = {{0, 1}, {1, 2}, {2, 3}};
    problem.element_types.assign(3, ElementType::Airway);
    set_linear_rigid_airway_model(params, {{1, 1.0}, {2, 0.9}, {3, 0.8}});
    set_linear_terminal_unit_model(params);
    set_pressure_boundaries(problem, {3});
    return problem;
  }

  TestProblem make_serial_airway_root_flow_parameters(double dt)
  {
    auto problem = make_serial_airway_parameters(dt);
    set_pressure_boundaries(problem, {3}, true);
    return problem;
  }

  TestProblem make_bifurcation_parameters(double dt)
  {
    TestProblem problem;
    auto& params = problem.parameters;
    set_common_air_properties(params);
    params.dynamics = make_dynamics(dt);

    problem.node_coordinates = {
        {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {2.0, 1.0, 0.0}, {2.0, -1.0, 0.0}};
    problem.element_nodes = {{0, 1}, {1, 2}, {1, 3}};
    problem.element_types.assign(3, ElementType::Airway);
    set_linear_rigid_airway_model(params, {{1, 1.0}, {2, 0.85}, {3, 0.7}});
    set_linear_terminal_unit_model(params);
    set_pressure_boundaries(problem, {2, 3});
    return problem;
  }

  TestProblem make_kelvin_voigt_airway_parameters(double dt)
  {
    auto problem = make_serial_airway_parameters(dt);
    set_airway_model(problem.parameters, {{1, 1.0}, {2, 0.9}, {3, 0.8}}, ResistanceType::Linear,
        WallModelType::KelvinVoigt);
    return problem;
  }

  TestProblem make_nonlinear_airway_parameters(double dt)
  {
    auto problem = make_serial_airway_parameters(dt);
    set_airway_model(problem.parameters, {{1, 1.0}, {2, 0.9}, {3, 0.8}}, ResistanceType::NonLinear,
        WallModelType::Rigid);
    return problem;
  }

  TestProblem make_nonlinear_kelvin_voigt_airway_parameters(double dt)
  {
    auto problem = make_serial_airway_parameters(dt);
    set_airway_model(problem.parameters, {{1, 1.0}, {2, 0.9}, {3, 0.8}}, ResistanceType::NonLinear,
        WallModelType::KelvinVoigt);
    return problem;
  }

  TestProblem make_ogden_terminal_unit_parameters(double dt)
  {
    auto problem = make_single_terminal_unit_parameters(dt);
    set_terminal_unit_model(problem.parameters, RheologyType::KelvinVoigt, ElasticityType::Ogden);
    return problem;
  }

  TestProblem make_four_element_maxwell_terminal_unit_parameters(double dt)
  {
    auto problem = make_single_terminal_unit_parameters(dt);
    set_terminal_unit_model(
        problem.parameters, RheologyType::FourElementMaxwell, ElasticityType::Linear);
    return problem;
  }

  TestProblem make_four_element_maxwell_ogden_terminal_unit_parameters(double dt)
  {
    auto problem = make_single_terminal_unit_parameters(dt);
    set_terminal_unit_model(
        problem.parameters, RheologyType::FourElementMaxwell, ElasticityType::Ogden);
    return problem;
  }

  TestProblem make_coupled_recruitment_terminal_unit_parameters(double dt)
  {
    auto problem = make_single_terminal_unit_parameters(dt);
    auto& recruitment = problem.parameters.lung_tree.terminal_units.recruitment_model;
    using RecruitmentModel = ReducedLungParameters::LungTree::TerminalUnits::RecruitmentModel;
    recruitment.pressure_law_type = Core::IO::InputField<RecruitmentModel::PressureLawType>(
        RecruitmentModel::PressureLawType::LinearPressure);
    recruitment.time_law_type =
        Core::IO::InputField<RecruitmentModel::TimeLawType>(RecruitmentModel::TimeLawType::None);
    recruitment.reference_volume_linearization =
        Core::IO::InputField<RecruitmentModel::ReferenceVolumeLinearization>(
            RecruitmentModel::ReferenceVolumeLinearization::Coupled);
    recruitment.linear_pressure.v0_min = Core::IO::InputField<double>(0.4);
    recruitment.linear_pressure.v0_max = Core::IO::InputField<double>(1.4);
    recruitment.linear_pressure.p_closing_min = Core::IO::InputField<double>(-0.2);
    recruitment.linear_pressure.p_opening_min = Core::IO::InputField<double>(0.0);
    recruitment.linear_pressure.delta_p_minmax = Core::IO::InputField<double>(1.0);
    recruitment.linear_pressure.epsilon_v0_switch = Core::IO::InputField<double>(0.01);
    problem.parameters.lung_tree.terminal_units.v0 = Core::IO::InputField<double>(0.4);
    recruitment.linear_pressure.initial_path =
        Core::IO::InputField<RecruitmentModel::HysteresisPath>(
            RecruitmentModel::HysteresisPath::Opening);
    return problem;
  }

  TestProblem make_mixed_airway_terminal_unit_parameters(double dt)
  {
    TestProblem problem;
    auto& params = problem.parameters;
    set_common_air_properties(params);
    params.dynamics = make_dynamics(dt);

    problem.node_coordinates = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {2.0, 1.0, 0.0}, {2.0, -1.0, 0.0},
        {3.0, 1.0, 0.0}, {3.0, -1.0, 0.0}};
    problem.element_nodes = {{0, 1}, {1, 2}, {1, 3}, {2, 4}, {3, 5}};
    problem.element_types = {ElementType::Airway, ElementType::Airway, ElementType::Airway,
        ElementType::TerminalUnit, ElementType::TerminalUnit};

    params.lung_tree.airways.radius = Core::IO::InputField<double>(
        std::unordered_map<int, double>{{1, 1.0}, {2, 0.85}, {3, 0.7}});
    params.lung_tree.airways.flow_model.resistance_type = Core::IO::InputField<ResistanceType>(
        std::unordered_map<int, ResistanceType>{{1, ResistanceType::Linear},
            {2, ResistanceType::NonLinear}, {3, ResistanceType::NonLinear}});
    params.lung_tree.airways.flow_model.resistance_model.non_linear.turbulence_factor_gamma =
        Core::IO::InputField<double>(0.6);
    params.lung_tree.airways.flow_model.include_inertia = Core::IO::InputField<bool>(false);
    params.lung_tree.airways.wall_model_type =
        Core::IO::InputField<WallModelType>(std::unordered_map<int, WallModelType>{
            {1, WallModelType::KelvinVoigt}, {2, WallModelType::Rigid}, {3, WallModelType::Rigid}});
    params.lung_tree.airways.wall_model.kelvin_voigt.elasticity.wall_poisson_ratio =
        Core::IO::InputField<double>(0.3);
    params.lung_tree.airways.wall_model.kelvin_voigt.elasticity.wall_elasticity =
        Core::IO::InputField<double>(50000.0);
    params.lung_tree.airways.wall_model.kelvin_voigt.elasticity.wall_thickness =
        Core::IO::InputField<double>(0.001);
    params.lung_tree.airways.wall_model.kelvin_voigt.viscosity.viscous_time_constant =
        Core::IO::InputField<double>(0.01);
    params.lung_tree.airways.wall_model.kelvin_voigt.viscosity.viscous_phase_shift =
        Core::IO::InputField<double>(0.1);

    params.lung_tree.terminal_units.v0 = Core::IO::InputField<double>(1.0);
    params.lung_tree.terminal_units.rheological_model.rheological_model_type =
        Core::IO::InputField<RheologyType>(std::unordered_map<int, RheologyType>{
            {4, RheologyType::FourElementMaxwell}, {5, RheologyType::KelvinVoigt}});
    params.lung_tree.terminal_units.rheological_model.kelvin_voigt.viscosity_kelvin_voigt_eta =
        Core::IO::InputField<double>(1.0);
    params.lung_tree.terminal_units.rheological_model.four_element_maxwell
        .viscosity_kelvin_voigt_eta = Core::IO::InputField<double>(0.1);
    params.lung_tree.terminal_units.rheological_model.four_element_maxwell.viscosity_maxwell_eta_m =
        Core::IO::InputField<double>(0.5);
    params.lung_tree.terminal_units.rheological_model.four_element_maxwell.elasticity_maxwell_e_m =
        Core::IO::InputField<double>(1.5);
    params.lung_tree.terminal_units.elasticity_model.elasticity_model_type =
        Core::IO::InputField<ElasticityType>(ElasticityType::Linear);
    params.lung_tree.terminal_units.elasticity_model.linear.elasticity_e =
        Core::IO::InputField<double>(1.0);

    set_pressure_boundaries(problem, {4, 5});
    return problem;
  }

  struct GeneratedTreeTopology
  {
    std::unordered_map<int, std::vector<double>> coordinates;
    std::unordered_map<int, std::vector<int>> element_nodes;
    std::unordered_map<int, int> generation;
    std::unordered_map<int, double> radii;
    std::vector<int> leaf_elements;
    std::vector<int> leaf_nodes;
    int num_nodes = 0;
    int num_elements = 0;
  };

  GeneratedTreeTopology make_asymmetric_test_tree_topology(int leaf_count)
  {
    GeneratedTreeTopology topology;
    topology.coordinates[1] = {0.0, 0.0, 0.0};
    int next_node = 1;
    int next_element = 1;

    const auto add_element = [&](int inlet_node, int generation, double x, double y)
    {
      const int element = next_element++;
      const int outlet_node = ++next_node;
      topology.coordinates[outlet_node] = {x, y, 0.0};
      topology.element_nodes[element] = {inlet_node, outlet_node};
      topology.generation[element] = generation;
      topology.radii[element] = 1.0 / (1.0 + 0.02 * static_cast<double>(element - 1));
      return std::pair<int, int>{element, outlet_node};
    };

    const auto [root_element, root_outlet_node] = add_element(1, 0, 1.0, 0.0);
    (void)root_element;

    std::function<void(int, int, int, double, double)> add_subtree =
        [&](int inlet_node, int depth, int leaves, double y, double span)
    {
      const auto [element, outlet_node] =
          add_element(inlet_node, depth, static_cast<double>(depth + 1), y);
      if (leaves == 1)
      {
        topology.leaf_elements.push_back(element);
        topology.leaf_nodes.push_back(outlet_node);
        return;
      }

      const int left_leaves = leaves / 2;
      const int right_leaves = leaves - left_leaves;
      add_subtree(outlet_node, depth + 1, left_leaves, y + span, 0.5 * span);
      add_subtree(outlet_node, depth + 1, right_leaves, y - span, 0.5 * span);
    };
    add_subtree(root_outlet_node, 1, leaf_count, 0.0, 1.0);

    topology.num_nodes = next_node;
    topology.num_elements = next_element - 1;
    return topology;
  }

  void set_generated_tree_geometry(TestProblem& problem, const GeneratedTreeTopology& topology)
  {
    problem.node_coordinates.reserve(static_cast<std::size_t>(topology.num_nodes));
    for (int node_id = 1; node_id <= topology.num_nodes; ++node_id)
    {
      const auto& coordinates = topology.coordinates.at(node_id);
      problem.node_coordinates.push_back({coordinates[0], coordinates[1], coordinates[2]});
    }

    problem.element_nodes.reserve(static_cast<std::size_t>(topology.num_elements));
    for (int element_id = 1; element_id <= topology.num_elements; ++element_id)
    {
      const auto& nodes = topology.element_nodes.at(element_id);
      problem.element_nodes.push_back({nodes[0] - 1, nodes[1] - 1});
    }
  }

  void set_pressure_boundaries_for_test_tree(
      TestProblem& problem, const std::vector<int>& leaf_nodes)
  {
    std::vector<int> outlet_nodes;
    outlet_nodes.reserve(leaf_nodes.size());
    for (const int node_id : leaf_nodes) outlet_nodes.push_back(node_id - 1);
    set_pressure_boundaries(problem, outlet_nodes);
  }

  TestProblem make_large_asymmetric_airway_parameters(double dt, WallModelType wall_model_type)
  {
    constexpr int leaf_count = 13;
    const auto topology = make_asymmetric_test_tree_topology(leaf_count);

    TestProblem problem;
    auto& params = problem.parameters;
    set_common_air_properties(params);
    params.dynamics = make_dynamics(dt);
    set_generated_tree_geometry(problem, topology);
    problem.element_types.assign(
        static_cast<std::size_t>(topology.num_elements), ElementType::Airway);
    set_airway_model(params, topology.radii, ResistanceType::Linear, wall_model_type);
    set_linear_terminal_unit_model(params);
    set_pressure_boundaries_for_test_tree(problem, topology.leaf_nodes);
    return problem;
  }

  TestProblem make_large_mixed_airway_terminal_unit_parameters(double dt)
  {
    constexpr int leaf_count = 13;
    const auto topology = make_asymmetric_test_tree_topology(leaf_count);

    TestProblem problem;
    auto& params = problem.parameters;
    set_common_air_properties(params);
    params.dynamics = make_dynamics(dt);
    set_generated_tree_geometry(problem, topology);
    problem.element_types.resize(static_cast<std::size_t>(topology.num_elements));

    std::unordered_map<int, double> airway_radii;
    std::unordered_map<int, WallModelType> wall_models;
    for (const auto& [element, nodes] : topology.element_nodes)
    {
      (void)nodes;
      const bool is_leaf = std::find(topology.leaf_elements.begin(), topology.leaf_elements.end(),
                               element) != topology.leaf_elements.end();
      if (is_leaf)
      {
        problem.element_types[static_cast<std::size_t>(element - 1)] = ElementType::TerminalUnit;
      }
      else
      {
        problem.element_types[static_cast<std::size_t>(element - 1)] = ElementType::Airway;
        airway_radii[element] = topology.radii.at(element);
        wall_models[element] = element % 3 == 0 ? WallModelType::KelvinVoigt : WallModelType::Rigid;
      }
    }

    params.lung_tree.airways.radius = Core::IO::InputField<double>(airway_radii);
    params.lung_tree.airways.flow_model.resistance_type =
        Core::IO::InputField<ResistanceType>(ResistanceType::Linear);
    params.lung_tree.airways.flow_model.include_inertia = Core::IO::InputField<bool>(false);
    params.lung_tree.airways.wall_model_type = Core::IO::InputField<WallModelType>(wall_models);
    params.lung_tree.airways.wall_model.kelvin_voigt.elasticity.wall_poisson_ratio =
        Core::IO::InputField<double>(0.3);
    params.lung_tree.airways.wall_model.kelvin_voigt.elasticity.wall_elasticity =
        Core::IO::InputField<double>(50000.0);
    params.lung_tree.airways.wall_model.kelvin_voigt.elasticity.wall_thickness =
        Core::IO::InputField<double>(0.001);
    params.lung_tree.airways.wall_model.kelvin_voigt.viscosity.viscous_time_constant =
        Core::IO::InputField<double>(0.01);
    params.lung_tree.airways.wall_model.kelvin_voigt.viscosity.viscous_phase_shift =
        Core::IO::InputField<double>(0.1);
    set_linear_terminal_unit_model(params);
    set_pressure_boundaries_for_test_tree(problem, topology.leaf_nodes);
    return problem;
  }

  struct LinearSolverFixture
  {
    ReducedLungParameters params;
    std::vector<std::array<double, 3>> node_coordinates;
    std::vector<std::array<int, 2>> element_nodes;
    std::vector<ElementType> element_types;
    std::map<int, std::vector<int>> bc_nodes;
    Core::Utils::FunctionManager function_manager;
    Core::FE::Discretization discretization;
    Airways::AirwayContainer airways;
    TerminalUnits::TerminalUnitContainer terminal_units;
    std::map<int, int> dof_per_ele;
    int n_airways = 0;
    int n_terminal_units = 0;
    std::map<int, int> first_global_dof_of_ele;
    std::map<int, int> global_dof_per_ele;
    std::map<int, std::vector<int>> global_ele_ids_per_node;
    BoundaryConditions::BoundaryConditionContainer boundary_conditions;
    Junctions::ConnectionData connections;
    Junctions::BifurcationData bifurcations;
    std::unique_ptr<Core::LinAlg::Map> locally_owned_dof_map;
    std::unique_ptr<Core::LinAlg::Map> row_map;
    std::unique_ptr<Core::LinAlg::Map> locally_relevant_dof_map;
    std::unique_ptr<Core::LinAlg::Vector<double>> dofs;
    std::unique_ptr<Core::LinAlg::Vector<double>> locally_relevant_dofs;
    std::unique_ptr<Core::LinAlg::Vector<double>> x;
    std::unique_ptr<Core::LinAlg::SparseMatrix> sysmat;
    ReducedLungAssemblyPipeline assembly_pipeline;
    std::optional<ReducedLungTreeMetadata> tree_metadata;
    Teuchos::ParameterList solver_params;
    std::function<const Teuchos::ParameterList&(int)> solver_params_callback;

    LinearSolverFixture(const std::string& name, const TestProblem& problem,
        const std::vector<std::string>& function_definitions)
        : params(problem.parameters),
          node_coordinates(problem.node_coordinates),
          element_nodes(problem.element_nodes),
          element_types(problem.element_types),
          bc_nodes(problem.bc_nodes),
          function_manager(make_function_manager(function_definitions)),
          discretization(name, MPI_COMM_WORLD, 3)
    {
      initialize();
    }

    void initialize()
    {
      Core::Rebalance::RebalanceParameters rebalance_parameters;
      build_discretization_from_nodes_and_elements(
          discretization, node_coordinates, element_nodes, rebalance_parameters);
      discretization.fill_complete();

      create_local_element_models(discretization, params, element_types, airways, terminal_units,
          dof_per_ele, n_airways, n_terminal_units);
      create_global_dof_maps(
          dof_per_ele, MPI_COMM_WORLD, global_dof_per_ele, first_global_dof_of_ele);
      assign_global_dof_ids_to_models(first_global_dof_of_ele, airways, terminal_units);
      TerminalUnits::create_evaluators(terminal_units);
      Airways::create_evaluators(airways);

      global_ele_ids_per_node = create_global_ele_ids_per_node(discretization, MPI_COMM_WORLD);
      BoundaryConditions::create_boundary_conditions(discretization, params, bc_nodes,
          global_ele_ids_per_node, global_dof_per_ele, first_global_dof_of_ele, function_manager,
          boundary_conditions);
      BoundaryConditions::create_evaluators(boundary_conditions);
      Junctions::create_junctions(discretization, global_ele_ids_per_node, global_dof_per_ele,
          first_global_dof_of_ele, connections, bifurcations);

      int n_local_equations = 0;
      Airways::assign_local_equation_ids(airways, n_local_equations);
      TerminalUnits::assign_local_equation_ids(terminal_units, n_local_equations);
      Junctions::assign_junction_local_equation_ids(connections, bifurcations, n_local_equations);
      BoundaryConditions::assign_local_equation_ids(boundary_conditions, n_local_equations);

      locally_owned_dof_map = std::make_unique<Core::LinAlg::Map>(
          create_domain_map(MPI_COMM_WORLD, airways, terminal_units));
      row_map = std::make_unique<Core::LinAlg::Map>(create_row_map(
          MPI_COMM_WORLD, airways, terminal_units, connections, bifurcations, boundary_conditions));
      locally_relevant_dof_map = std::make_unique<Core::LinAlg::Map>(
          create_column_map(MPI_COMM_WORLD, airways, terminal_units, global_dof_per_ele,
              first_global_dof_of_ele, connections, bifurcations, boundary_conditions));

      Junctions::assign_junction_global_equation_ids(*row_map, connections, bifurcations);
      BoundaryConditions::assign_global_equation_ids(*row_map, boundary_conditions);
      Airways::assign_local_dof_ids(*locally_relevant_dof_map, airways);
      TerminalUnits::assign_local_dof_ids(*locally_relevant_dof_map, terminal_units);
      Junctions::assign_junction_local_dof_ids(
          *locally_relevant_dof_map, connections, bifurcations);
      BoundaryConditions::assign_local_dof_ids(*locally_relevant_dof_map, boundary_conditions);

      dofs = std::make_unique<Core::LinAlg::Vector<double>>(*locally_owned_dof_map, true);
      locally_relevant_dofs =
          std::make_unique<Core::LinAlg::Vector<double>>(*locally_relevant_dof_map, true);
      x = std::make_unique<Core::LinAlg::Vector<double>>(*row_map, true);
      sysmat = std::make_unique<Core::LinAlg::SparseMatrix>(*row_map, *locally_relevant_dof_map, 4);

      solver_params.set("SOLVER", Core::LinearSolver::SolverType::UMFPACK);
      solver_params.set("NAME", "Reduced_lung_solver");
      solver_params_callback = [this](int) -> const Teuchos::ParameterList&
      { return solver_params; };
      assembly_pipeline = create_default_reduced_lung_assembly_pipeline(
          airways, terminal_units, connections, bifurcations, boundary_conditions);
    }

    void sync_state_from_x()
    {
      Core::LinAlg::export_to(*x, *dofs);
      Core::LinAlg::export_to(*dofs, *locally_relevant_dofs);
      for (const auto& update_state : assembly_pipeline.state_updaters)
      {
        update_state(*locally_relevant_dofs, params.dynamics.time_increment);
      }
    }

    Core::LinAlg::Vector<double> assemble_residual(double current_time)
    {
      Core::LinAlg::Vector<double> residual(*row_map, true);
      residual.put_scalar(0.0);
      for (const auto& assemble_residual_callback : assembly_pipeline.residual_assemblers)
      {
        assemble_residual_callback(
            residual, *locally_relevant_dofs, current_time, params.dynamics.time_increment);
      }
      return residual;
    }

    void assemble_jacobian(double current_time)
    {
      for (const auto& assemble_jacobian_callback : assembly_pipeline.jacobian_assemblers)
      {
        assemble_jacobian_callback(
            *sysmat, *locally_relevant_dofs, current_time, params.dynamics.time_increment);
      }
      if (!sysmat->filled()) sysmat->complete();
    }

    TreeLinearization assemble_tree_linearization(double current_time)
    {
      TreeLinearization linearization(
          row_map->num_my_elements(), locally_relevant_dof_map->num_my_elements());
      for (const auto& initialize_capacity :
          assembly_pipeline.tree_linearization_capacity_initializers)
      {
        initialize_capacity(linearization);
      }
      assemble_tree_coefficients(linearization, current_time);
      return linearization;
    }

    void assemble_tree_coefficients(TreeCoefficientAssemblyTarget& target, double current_time)
    {
      for (const auto& tree_linearization_static_assembler :
          assembly_pipeline.tree_linearization_static_assemblers)
      {
        tree_linearization_static_assembler.callback(target);
      }
      for (const auto& tree_linearization_assembler :
          assembly_pipeline.tree_linearization_assemblers)
      {
        tree_linearization_assembler.callback(
            target, *locally_relevant_dofs, current_time, params.dynamics.time_increment);
      }
    }

    ReducedLungTreeMetadata build_tree_metadata() const
    {
      return build_reduced_lung_tree_metadata(ReducedLungTreeMetadataContext{
          .discretization = discretization,
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

    std::unique_ptr<NoxSolver> create_nox_solver()
    {
      const NoxSolverContext context{
          .comm = MPI_COMM_WORLD,
          .dynamics = params.dynamics,
          .linear_solver_parameters = solver_params,
          .solver_params_callback = solver_params_callback,
          .assembly_pipeline = assembly_pipeline,
          .dofs = *dofs,
          .locally_relevant_dofs = *locally_relevant_dofs,
          .x = *x,
          .jacobian = *sysmat,
      };
      return std::make_unique<NoxSolver>(context);
    }

    std::unique_ptr<NewtonSolver> create_newton_solver(
        const std::shared_ptr<NewtonLinearSolver>& linear_solver,
        const ReducedLungAssemblyPipeline& solver_assembly_pipeline)
    {
      const NewtonSolverContext context{
          .dynamics = params.dynamics,
          .linear_solver = linear_solver,
          .assembly_pipeline = solver_assembly_pipeline,
          .dofs = *dofs,
          .locally_relevant_dofs = *locally_relevant_dofs,
          .x = *x,
      };
      return std::make_unique<NewtonSolver>(context);
    }

    std::unique_ptr<NewtonSolver> create_sparse_newton_solver()
    {
      auto linear_solver =
          std::make_shared<SparseNewtonLinearSolver>(SparseNewtonLinearSolverContext{
              .comm = MPI_COMM_WORLD,
              .linear_solver_parameters = solver_params,
              .solver_params_callback = solver_params_callback,
              .correction_map = *row_map,
              .jacobian = *sysmat,
          });
      return create_newton_solver(linear_solver, assembly_pipeline);
    }

    std::unique_ptr<NewtonSolver> create_tree_newton_solver()
    {
      tree_metadata = build_tree_metadata();
      auto linear_solver = std::make_shared<TreeNewtonLinearSolver>(
          TreeNewtonLinearSolverContext{.tree_metadata = *tree_metadata,
              .coefficient_source = TreeNewtonLinearSolverCoefficientSource::StructuredTreeBlocks});
      const auto tree_assembly_pipeline = create_default_reduced_lung_assembly_pipeline(
          airways, terminal_units, connections, bifurcations, boundary_conditions, false);
      EXPECT_TRUE(tree_assembly_pipeline.jacobian_assemblers.empty());
      return create_newton_solver(linear_solver, tree_assembly_pipeline);
    }

    double residual_norm(double current_time)
    {
      auto residual = assemble_residual(current_time);
      double norm = 0.0;
      residual.norm_2(&norm);
      return norm;
    }

    void advance_end_of_timestep()
    {
      TerminalUnits::end_of_timestep_routine(
          terminal_units, *locally_relevant_dofs, params.dynamics.time_increment);
      Airways::end_of_timestep_routine(
          airways, *locally_relevant_dofs, params.dynamics.time_increment);
    }
  };

  void expect_vectors_near(const Core::LinAlg::Vector<double>& expected,
      const Core::LinAlg::Vector<double>& actual, double tolerance)
  {
    ASSERT_EQ(expected.global_length(), actual.global_length());
    ASSERT_EQ(expected.local_length(), actual.local_length());
    const auto expected_values = expected.local_values_as_span();
    const auto actual_values = actual.local_values_as_span();
    for (std::size_t i = 0; i < expected_values.size(); ++i)
    {
      EXPECT_NEAR(expected_values[i], actual_values[i], tolerance) << "local vector entry " << i;
    }
  }

  void expect_structured_coefficients_near(
      const std::vector<TreeStructuredCoefficientValue>& expected,
      const std::vector<TreeStructuredCoefficientValue>& actual, double tolerance)
  {
    ASSERT_EQ(expected.size(), actual.size());
    for (std::size_t i = 0; i < expected.size(); ++i)
    {
      SCOPED_TRACE(i);
      EXPECT_EQ(expected[i].local_row, actual[i].local_row);
      EXPECT_EQ(expected[i].local_dof, actual[i].local_dof);
      EXPECT_STREQ(expected[i].context, actual[i].context);
      EXPECT_NEAR(expected[i].value, actual[i].value, tolerance);
    }
  }

  void seed_nonzero_initial_state(LinearSolverFixture& fixture, double scale = 1.0)
  {
    for (const auto& airway_model : fixture.airways.models)
    {
      const auto& data = airway_model.data;
      for (std::size_t i = 0; i < data.number_of_elements(); ++i)
      {
        fixture.x->replace_global_value(
            data.gid_p1[i], scale * (1.0 + 0.1 * static_cast<double>(i)));
        fixture.x->replace_global_value(
            data.gid_p2[i], scale * (0.5 + 0.05 * static_cast<double>(i)));
        fixture.x->replace_global_value(
            data.gid_q1[i], scale * (80.0 + 10.0 * static_cast<double>(i)));
        if (i < data.gid_q2.size())
        {
          fixture.x->replace_global_value(
              data.gid_q2[i], scale * (60.0 + 8.0 * static_cast<double>(i)));
        }
      }
    }

    for (const auto& terminal_unit_model : fixture.terminal_units.models)
    {
      const auto& data = terminal_unit_model.data;
      for (std::size_t i = 0; i < data.number_of_elements(); ++i)
      {
        fixture.x->replace_global_value(
            data.gid_p1[i], scale * (0.8 + 0.1 * static_cast<double>(i)));
        fixture.x->replace_global_value(
            data.gid_p2[i], scale * (0.2 + 0.05 * static_cast<double>(i)));
        fixture.x->replace_global_value(
            data.gid_q[i], scale * (0.05 + 0.01 * static_cast<double>(i)));
      }
    }
  }

  void compare_tree_and_sparse_corrections(
      const std::string& name, const TestProblem& problem, bool seed_nonzero_state = false)
  {
    const auto& params = problem.parameters;
    LinearSolverFixture fixture(name, problem, {"1.0 * t", "0.0"});
    const double current_time = params.dynamics.time_increment;

    if (seed_nonzero_state)
    {
      seed_nonzero_initial_state(fixture);
    }
    fixture.sync_state_from_x();
    auto residual = fixture.assemble_residual(current_time);
    fixture.assemble_jacobian(current_time);

    Core::LinAlg::Vector<double> sparse_delta(*fixture.row_map, true);
    Core::LinAlg::Vector<double> tree_delta(*fixture.row_map, true);
    Core::LinAlg::Vector<double> structured_tree_delta(*fixture.row_map, true);

    SparseNewtonLinearSolver sparse_solver(SparseNewtonLinearSolverContext{
        .comm = MPI_COMM_WORLD,
        .linear_solver_parameters = fixture.solver_params,
        .solver_params_callback = fixture.solver_params_callback,
        .correction_map = *fixture.row_map,
        .jacobian = *fixture.sysmat,
    });
    sparse_solver.solve(residual, *fixture.x,
        NewtonLinearSystemMetadata{.current_time = current_time,
            .time_step_size_dt = params.dynamics.time_increment,
            .nonlinear_iteration = 0},
        sparse_delta);

    const auto tree_metadata = fixture.build_tree_metadata();
    TreeNewtonLinearSolver tree_solver(TreeNewtonLinearSolverContext{.tree_metadata = tree_metadata,
        .pivot_tolerance = 1.0e-12,
        .validation_jacobian = fixture.sysmat.get()});
    tree_solver.solve(residual, *fixture.x,
        NewtonLinearSystemMetadata{.current_time = current_time,
            .time_step_size_dt = params.dynamics.time_increment,
            .nonlinear_iteration = 0},
        tree_delta);

    auto tree_linearization = fixture.assemble_tree_linearization(current_time);
    TreeNewtonLinearSolver structured_tree_solver(
        TreeNewtonLinearSolverContext{.tree_metadata = tree_metadata,
            .pivot_tolerance = 1.0e-12,
            .coefficient_source = TreeNewtonLinearSolverCoefficientSource::StructuredTreeBlocks});
    structured_tree_solver.set_tree_linearization(tree_linearization);
    structured_tree_solver.solve(residual, *fixture.x,
        NewtonLinearSystemMetadata{.current_time = current_time,
            .time_step_size_dt = params.dynamics.time_increment,
            .nonlinear_iteration = 0},
        structured_tree_delta);

    expect_vectors_near(sparse_delta, tree_delta, 1.0e-9);
    expect_vectors_near(tree_delta, structured_tree_delta, 1.0e-9);
  }

  void compare_reused_structured_tree_solver_corrections(
      const std::string& name, const TestProblem& problem)
  {
    const auto& params = problem.parameters;
    LinearSolverFixture fixture(name, problem, {"0.25 + 0.5 * t", "0.0"});
    const auto tree_metadata = fixture.build_tree_metadata();
    TreeNewtonLinearSolver tree_solver(TreeNewtonLinearSolverContext{.tree_metadata = tree_metadata,
        .pivot_tolerance = 1.0e-12,
        .coefficient_source = TreeNewtonLinearSolverCoefficientSource::StructuredTreeBlocks});

    const std::vector<double> state_scales{0.75, 1.0, 1.25};
    for (std::size_t step = 0; step < state_scales.size(); ++step)
    {
      seed_nonzero_initial_state(fixture, state_scales[step]);
      fixture.sync_state_from_x();
      const double current_time = params.dynamics.time_increment * static_cast<double>(step + 1);
      auto residual = fixture.assemble_residual(current_time);
      fixture.sysmat = std::make_unique<Core::LinAlg::SparseMatrix>(
          *fixture.row_map, *fixture.locally_relevant_dof_map, 4);
      fixture.assemble_jacobian(current_time);
      SparseNewtonLinearSolver sparse_solver(SparseNewtonLinearSolverContext{
          .comm = MPI_COMM_WORLD,
          .linear_solver_parameters = fixture.solver_params,
          .solver_params_callback = fixture.solver_params_callback,
          .correction_map = *fixture.row_map,
          .jacobian = *fixture.sysmat,
      });
      auto tree_linearization = fixture.assemble_tree_linearization(current_time);
      tree_solver.set_tree_linearization(tree_linearization);

      Core::LinAlg::Vector<double> sparse_delta(*fixture.row_map, true);
      Core::LinAlg::Vector<double> tree_delta(*fixture.row_map, true);
      const NewtonLinearSystemMetadata metadata{.current_time = current_time,
          .time_step_size_dt = params.dynamics.time_increment,
          .nonlinear_iteration = static_cast<unsigned int>(step)};
      sparse_solver.solve(residual, *fixture.x, metadata, sparse_delta);
      tree_solver.solve(residual, *fixture.x, metadata, tree_delta);

      expect_vectors_near(sparse_delta, tree_delta, 1.0e-9);
    }
  }

  void compare_direct_and_generic_structured_coefficients(const std::string& name,
      const TestProblem& problem, bool seed_nonzero_state,
      bool expect_recruitment_derivative = false)
  {
    const auto& params = problem.parameters;
    LinearSolverFixture fixture(name, problem, {"1.0 * t", "0.0"});
    const double current_time = params.dynamics.time_increment;
    if (seed_nonzero_state)
    {
      seed_nonzero_initial_state(fixture);
    }
    fixture.sync_state_from_x();
    if (expect_recruitment_derivative)
    {
      bool found_nonzero_derivative = false;
      for (const auto& model : fixture.terminal_units.models)
      {
        for (const auto& context : model.data.reference_volume_context)
        {
          found_nonzero_derivative = found_nonzero_derivative || context.dv0_dp != 0.0;
        }
      }
      EXPECT_TRUE(found_nonzero_derivative);
    }
    auto residual = fixture.assemble_residual(current_time);
    fixture.assemble_jacobian(current_time);

    const auto tree_metadata = fixture.build_tree_metadata();
    const auto tree_linearization = fixture.assemble_tree_linearization(current_time);
    TreeNewtonLinearSolver generic_tree_solver(
        TreeNewtonLinearSolverContext{.tree_metadata = tree_metadata,
            .pivot_tolerance = 1.0e-12,
            .coefficient_source = TreeNewtonLinearSolverCoefficientSource::StructuredTreeBlocks});
    generic_tree_solver.set_tree_linearization(tree_linearization);

    TreeNewtonLinearSolverProfile direct_profile;
    TreeNewtonLinearSolver direct_tree_solver(
        TreeNewtonLinearSolverContext{.tree_metadata = tree_metadata,
            .pivot_tolerance = 1.0e-12,
            .coefficient_source = TreeNewtonLinearSolverCoefficientSource::StructuredTreeBlocks,
            .profile = &direct_profile});
    TreeCoefficientAssemblyTarget* direct_target =
        direct_tree_solver.direct_tree_coefficient_target();
    ASSERT_NE(direct_target, nullptr);
    fixture.assemble_tree_coefficients(*direct_target, current_time);

    expect_structured_coefficients_near(generic_tree_solver.structured_coefficient_values(),
        direct_tree_solver.structured_coefficient_values(), 1.0e-12);

    Core::LinAlg::Vector<double> sparse_delta(*fixture.row_map, true);
    Core::LinAlg::Vector<double> direct_tree_delta(*fixture.row_map, true);
    SparseNewtonLinearSolver sparse_solver(SparseNewtonLinearSolverContext{
        .comm = MPI_COMM_WORLD,
        .linear_solver_parameters = fixture.solver_params,
        .solver_params_callback = fixture.solver_params_callback,
        .correction_map = *fixture.row_map,
        .jacobian = *fixture.sysmat,
    });
    const NewtonLinearSystemMetadata metadata{.current_time = current_time,
        .time_step_size_dt = params.dynamics.time_increment,
        .nonlinear_iteration = 0};
    sparse_solver.solve(residual, *fixture.x, metadata, sparse_delta);
    direct_tree_solver.solve(residual, *fixture.x, metadata, direct_tree_delta);

    expect_vectors_near(sparse_delta, direct_tree_delta, 1.0e-9);
    EXPECT_EQ(direct_profile.coefficient_lookup_count, 0u);
  }

  void expect_forced_batch_tree_profile(const TreeNewtonLinearSolverProfile& profile,
      std::size_t element_count, std::size_t solve_count = 1)
  {
    EXPECT_EQ(profile.dense_solve_count, static_cast<std::uint64_t>(element_count * solve_count));
    EXPECT_EQ(profile.dense_fallback_count, 0u);
    EXPECT_EQ(profile.unsupported_block_fallback_count, 0u);
    if (require_tree_solver_simd_from_environment())
    {
      EXPECT_GT(profile.simd_group_count, 0u);
      EXPECT_GT(profile.simd_lane_count, 0u);
      EXPECT_EQ(profile.scalar_group_count, 0u);
      EXPECT_EQ(profile.scalar_tail_lane_count, 0u);
      return;
    }
    if (profile.simd_group_count > 0)
    {
      EXPECT_GT(profile.simd_lane_count, 0u);
      EXPECT_EQ(profile.scalar_group_count, 0u);
      EXPECT_EQ(profile.scalar_tail_lane_count, 0u);
    }
  }

  void compare_forced_batch_structured_tree_and_sparse_corrections(
      const std::string& name, const TestProblem& problem, bool seed_nonzero_state)
  {
    const auto& params = problem.parameters;
    LinearSolverFixture fixture(name, problem, {"1.0 * t", "0.0"});
    const double current_time = params.dynamics.time_increment;

    if (seed_nonzero_state)
    {
      seed_nonzero_initial_state(fixture);
    }
    fixture.sync_state_from_x();
    auto residual = fixture.assemble_residual(current_time);
    fixture.assemble_jacobian(current_time);

    Core::LinAlg::Vector<double> sparse_delta(*fixture.row_map, true);
    Core::LinAlg::Vector<double> tree_delta(*fixture.row_map, true);
    SparseNewtonLinearSolver sparse_solver(SparseNewtonLinearSolverContext{
        .comm = MPI_COMM_WORLD,
        .linear_solver_parameters = fixture.solver_params,
        .solver_params_callback = fixture.solver_params_callback,
        .correction_map = *fixture.row_map,
        .jacobian = *fixture.sysmat,
    });
    const NewtonLinearSystemMetadata metadata{.current_time = current_time,
        .time_step_size_dt = params.dynamics.time_increment,
        .nonlinear_iteration = 0};
    sparse_solver.solve(residual, *fixture.x, metadata, sparse_delta);

    const auto tree_metadata = fixture.build_tree_metadata();
    auto tree_linearization = fixture.assemble_tree_linearization(current_time);
    TreeNewtonLinearSolverProfile profile;
    TreeNewtonLinearSolver tree_solver(TreeNewtonLinearSolverContext{.tree_metadata = tree_metadata,
        .pivot_tolerance = 1.0e-12,
        .coefficient_source = TreeNewtonLinearSolverCoefficientSource::StructuredTreeBlocks,
        .profile = &profile,
        .force_batch_tree_solve = true,
        .error_on_dense_fallback = true});
    tree_solver.set_tree_linearization(tree_linearization);
    tree_solver.solve(residual, *fixture.x, metadata, tree_delta);

    expect_vectors_near(sparse_delta, tree_delta, 1.0e-9);
    expect_forced_batch_tree_profile(profile, tree_metadata.elements.size());
  }

  void compare_reused_forced_batch_structured_tree_solver_corrections(
      const std::string& name, const TestProblem& problem)
  {
    const auto& params = problem.parameters;
    LinearSolverFixture fixture(name, problem, {"0.25 + 0.5 * t", "0.0"});
    const auto tree_metadata = fixture.build_tree_metadata();
    TreeNewtonLinearSolverProfile profile;
    TreeNewtonLinearSolver tree_solver(TreeNewtonLinearSolverContext{.tree_metadata = tree_metadata,
        .pivot_tolerance = 1.0e-12,
        .coefficient_source = TreeNewtonLinearSolverCoefficientSource::StructuredTreeBlocks,
        .profile = &profile,
        .force_batch_tree_solve = true,
        .error_on_dense_fallback = true});
    const std::vector<double> state_scales{0.75, 1.0, 1.25};
    for (std::size_t step = 0; step < state_scales.size(); ++step)
    {
      seed_nonzero_initial_state(fixture, state_scales[step]);
      fixture.sync_state_from_x();
      const double current_time = params.dynamics.time_increment * static_cast<double>(step + 1);
      auto residual = fixture.assemble_residual(current_time);
      fixture.sysmat = std::make_unique<Core::LinAlg::SparseMatrix>(
          *fixture.row_map, *fixture.locally_relevant_dof_map, 4);
      fixture.assemble_jacobian(current_time);
      SparseNewtonLinearSolver sparse_solver(SparseNewtonLinearSolverContext{
          .comm = MPI_COMM_WORLD,
          .linear_solver_parameters = fixture.solver_params,
          .solver_params_callback = fixture.solver_params_callback,
          .correction_map = *fixture.row_map,
          .jacobian = *fixture.sysmat,
      });
      auto tree_linearization = fixture.assemble_tree_linearization(current_time);
      tree_solver.set_tree_linearization(tree_linearization);

      Core::LinAlg::Vector<double> sparse_delta(*fixture.row_map, true);
      Core::LinAlg::Vector<double> tree_delta(*fixture.row_map, true);
      const NewtonLinearSystemMetadata metadata{.current_time = current_time,
          .time_step_size_dt = params.dynamics.time_increment,
          .nonlinear_iteration = static_cast<unsigned int>(step)};
      sparse_solver.solve(residual, *fixture.x, metadata, sparse_delta);
      tree_solver.solve(residual, *fixture.x, metadata, tree_delta);

      expect_vectors_near(sparse_delta, tree_delta, 1.0e-9);
    }

    expect_forced_batch_tree_profile(profile, tree_metadata.elements.size(), state_scales.size());
  }

  struct ComparisonChecks
  {
    bool terminal_unit_volumes = false;
    bool connection_flow_balance = false;
    bool bifurcation_flow_balance = false;
  };

  void expect_terminal_unit_volumes_near(
      const LinearSolverFixture& expected, const LinearSolverFixture& actual, double tolerance)
  {
    ASSERT_EQ(expected.terminal_units.models.size(), actual.terminal_units.models.size());
    for (std::size_t model_index = 0; model_index < expected.terminal_units.models.size();
        ++model_index)
    {
      const auto& expected_data = expected.terminal_units.models[model_index].data;
      const auto& actual_data = actual.terminal_units.models[model_index].data;
      ASSERT_EQ(expected_data.volume_v.size(), actual_data.volume_v.size());
      for (std::size_t i = 0; i < expected_data.volume_v.size(); ++i)
      {
        EXPECT_NEAR(expected_data.volume_v[i], actual_data.volume_v[i], tolerance)
            << "terminal-unit model " << model_index << ", entry " << i;
      }
    }
  }

  void expect_connection_flow_balance(const LinearSolverFixture& fixture, double tolerance)
  {
    const auto dofs = fixture.locally_relevant_dofs->local_values_as_span();
    for (std::size_t i = 0; i < fixture.connections.size(); ++i)
    {
      const auto& local_dof_ids = fixture.connections.local_dof_ids[i];
      const double flow_balance = dofs[local_dof_ids[Junctions::ConnectionData::q_out_parent]] -
                                  dofs[local_dof_ids[Junctions::ConnectionData::q_in_child]];
      EXPECT_NEAR(flow_balance, 0.0, tolerance) << "connection " << i;
    }
  }

  void expect_bifurcation_flow_balance(const LinearSolverFixture& fixture, double tolerance)
  {
    const auto dofs = fixture.locally_relevant_dofs->local_values_as_span();
    for (std::size_t i = 0; i < fixture.bifurcations.size(); ++i)
    {
      const auto& local_dof_ids = fixture.bifurcations.local_dof_ids[i];
      const double flow_balance = dofs[local_dof_ids[Junctions::BifurcationData::q_out_parent]] -
                                  dofs[local_dof_ids[Junctions::BifurcationData::q_in_child_1]] -
                                  dofs[local_dof_ids[Junctions::BifurcationData::q_in_child_2]];
      EXPECT_NEAR(flow_balance, 0.0, tolerance) << "bifurcation " << i;
    }
  }

  void compare_all_solver_workflows(const std::string& name, TestProblem problem,
      const std::vector<std::string>& function_definitions, ComparisonChecks checks)
  {
    const auto& params = problem.parameters;
    LinearSolverFixture nox_fixture(name + "_nox", problem, function_definitions);
    LinearSolverFixture sparse_fixture(name + "_sparse", problem, function_definitions);
    LinearSolverFixture tree_fixture(name + "_tree", problem, function_definitions);

    auto nox_solver = nox_fixture.create_nox_solver();
    auto sparse_solver = sparse_fixture.create_sparse_newton_solver();
    auto tree_solver = tree_fixture.create_tree_newton_solver();

    const double solution_tolerance = 1.0e-6;
    const double residual_tolerance =
        std::max(1.0e-7, 100.0 * params.dynamics.nonlinear_residual_tolerance);

    for (int step = 1; step <= params.dynamics.number_of_steps; ++step)
    {
      const double current_time = step * params.dynamics.time_increment;
      const unsigned int nox_iterations = nox_solver->solve(current_time);
      const unsigned int sparse_iterations = sparse_solver->solve(current_time);
      const unsigned int tree_iterations = tree_solver->solve(current_time);

      EXPECT_LE(
          nox_iterations, static_cast<unsigned int>(params.dynamics.max_nonlinear_iterations));
      EXPECT_LE(
          sparse_iterations, static_cast<unsigned int>(params.dynamics.max_nonlinear_iterations));
      EXPECT_LE(
          tree_iterations, static_cast<unsigned int>(params.dynamics.max_nonlinear_iterations));

      expect_vectors_near(*nox_fixture.x, *sparse_fixture.x, solution_tolerance);
      expect_vectors_near(*nox_fixture.x, *tree_fixture.x, solution_tolerance);
      expect_vectors_near(*nox_fixture.dofs, *sparse_fixture.dofs, solution_tolerance);
      expect_vectors_near(*nox_fixture.dofs, *tree_fixture.dofs, solution_tolerance);
      EXPECT_LE(nox_fixture.residual_norm(current_time), residual_tolerance);
      EXPECT_LE(sparse_fixture.residual_norm(current_time), residual_tolerance);
      EXPECT_LE(tree_fixture.residual_norm(current_time), residual_tolerance);

      if (checks.connection_flow_balance)
      {
        expect_connection_flow_balance(nox_fixture, solution_tolerance);
        expect_connection_flow_balance(sparse_fixture, solution_tolerance);
        expect_connection_flow_balance(tree_fixture, solution_tolerance);
      }
      if (checks.bifurcation_flow_balance)
      {
        expect_bifurcation_flow_balance(nox_fixture, solution_tolerance);
        expect_bifurcation_flow_balance(sparse_fixture, solution_tolerance);
        expect_bifurcation_flow_balance(tree_fixture, solution_tolerance);
      }

      nox_fixture.advance_end_of_timestep();
      sparse_fixture.advance_end_of_timestep();
      tree_fixture.advance_end_of_timestep();
      if (checks.terminal_unit_volumes)
      {
        expect_terminal_unit_volumes_near(nox_fixture, sparse_fixture, solution_tolerance);
        expect_terminal_unit_volumes_near(nox_fixture, tree_fixture, solution_tolerance);
      }
    }
  }

  TEST(ReducedLungTreeLinearSolverTests, SingleTerminalUnitMatchesSparseSolver)
  {
    compare_tree_and_sparse_corrections(
        "tree_linear_single_terminal", make_single_terminal_unit_parameters(0.1));
  }

  TEST(ReducedLungTreeLinearSolverTests, SerialAirwaysMatchSparseSolver)
  {
    compare_tree_and_sparse_corrections(
        "tree_linear_serial_airways", make_serial_airway_parameters(0.1));
  }

  TEST(ReducedLungTreeLinearSolverTests, RootInletFlowBoundaryMatchesSparseSolver)
  {
    compare_tree_and_sparse_corrections(
        "tree_linear_root_inlet_flow", make_serial_airway_root_flow_parameters(0.1));
  }

  TEST(ReducedLungTreeLinearSolverTests, BifurcationAirwaysMatchSparseSolver)
  {
    compare_tree_and_sparse_corrections(
        "tree_linear_bifurcation_airways", make_bifurcation_parameters(0.1));
  }

  TEST(ReducedLungTreeLinearSolverTests, KelvinVoigtAirwaysMatchSparseSolver)
  {
    compare_tree_and_sparse_corrections(
        "tree_linear_kelvin_voigt_airways", make_kelvin_voigt_airway_parameters(0.1), true);
  }

  TEST(ReducedLungTreeLinearSolverTests, NonlinearAirwaysMatchSparseSolver)
  {
    compare_tree_and_sparse_corrections(
        "tree_linear_nonlinear_airways", make_nonlinear_airway_parameters(0.1), true);
  }

  TEST(ReducedLungTreeLinearSolverTests, FourElementMaxwellTerminalUnitMatchesSparseSolver)
  {
    compare_tree_and_sparse_corrections("tree_linear_four_element_maxwell_terminal",
        make_four_element_maxwell_terminal_unit_parameters(0.1), true);
  }

  TEST(ReducedLungTreeLinearSolverTests, MixedAirwaysAndTerminalUnitsMatchSparseSolver)
  {
    compare_tree_and_sparse_corrections("tree_linear_mixed_airways_terminal_units",
        make_mixed_airway_terminal_unit_parameters(0.1), true);
  }

  TEST(ReducedLungTreeLinearSolverTests, ReusedStructuredTreeSolverMatchesSparseSolver)
  {
    compare_reused_structured_tree_solver_corrections(
        "tree_linear_reused_structured_solver", make_mixed_airway_terminal_unit_parameters(0.1));
  }

  TEST(ReducedLungTreeLinearSolverTests, DirectStructuredAssemblyRigidAirwaysMatchesGenericPath)
  {
    compare_direct_and_generic_structured_coefficients(
        "tree_linear_direct_structured_rigid_airways", make_serial_airway_parameters(0.1), false);
  }

  TEST(ReducedLungTreeLinearSolverTests,
      DirectStructuredAssemblyRootInletFlowBoundaryMatchesGenericPath)
  {
    compare_direct_and_generic_structured_coefficients(
        "tree_linear_direct_structured_root_inlet_flow",
        make_serial_airway_root_flow_parameters(0.1), false);
  }

  TEST(ReducedLungTreeLinearSolverTests,
      DirectStructuredAssemblyKelvinVoigtAirwaysMatchesGenericPath)
  {
    compare_direct_and_generic_structured_coefficients(
        "tree_linear_direct_structured_kelvin_voigt_airways",
        make_kelvin_voigt_airway_parameters(0.1), true);
  }

  TEST(ReducedLungTreeLinearSolverTests,
      DirectStructuredAssemblyMixedTerminalUnitsMatchesGenericPath)
  {
    compare_direct_and_generic_structured_coefficients(
        "tree_linear_direct_structured_mixed_terminal_units",
        make_mixed_airway_terminal_unit_parameters(0.1), true);
  }

  TEST(ReducedLungTreeLinearSolverTests,
      DirectStructuredAssemblyNonlinearRigidAirwaysMatchesGenericPath)
  {
    compare_direct_and_generic_structured_coefficients(
        "tree_linear_direct_structured_nonlinear_rigid_airways",
        make_nonlinear_airway_parameters(0.1), true);
  }

  TEST(ReducedLungTreeLinearSolverTests,
      DirectStructuredAssemblyNonlinearKelvinVoigtAirwaysMatchesGenericPath)
  {
    compare_direct_and_generic_structured_coefficients(
        "tree_linear_direct_structured_nonlinear_kelvin_voigt_airways",
        make_nonlinear_kelvin_voigt_airway_parameters(0.1), true);
  }

  TEST(
      ReducedLungTreeLinearSolverTests, DirectStructuredAssemblyOgdenTerminalUnitMatchesGenericPath)
  {
    compare_direct_and_generic_structured_coefficients(
        "tree_linear_direct_structured_ogden_terminal_unit",
        make_ogden_terminal_unit_parameters(0.1), true);
  }

  TEST(ReducedLungTreeLinearSolverTests,
      DirectStructuredAssemblyFourElementMaxwellTerminalUnitMatchesGenericPath)
  {
    compare_direct_and_generic_structured_coefficients(
        "tree_linear_direct_structured_four_element_maxwell_terminal_unit",
        make_four_element_maxwell_terminal_unit_parameters(0.1), true);
  }

  TEST(ReducedLungTreeLinearSolverTests,
      DirectStructuredAssemblyCoupledRecruitmentMatchesSparseSolver)
  {
    compare_direct_and_generic_structured_coefficients(
        "tree_linear_direct_structured_coupled_recruitment",
        make_coupled_recruitment_terminal_unit_parameters(0.1), true, true);
  }

  TEST(ReducedLungTreeLinearSolverTests,
      DirectStructuredAssemblyFourElementMaxwellOgdenTerminalUnitMatchesGenericPath)
  {
    compare_direct_and_generic_structured_coefficients(
        "tree_linear_direct_structured_four_element_maxwell_ogden_terminal_unit",
        make_four_element_maxwell_ogden_terminal_unit_parameters(0.1), true);
  }

  TEST(ReducedLungTreeLinearSolverTests, ForcedBatchLargeRigidAirwaysMatchSparseSolver)
  {
    compare_forced_batch_structured_tree_and_sparse_corrections(
        "tree_linear_forced_batch_large_rigid_airways",
        make_large_asymmetric_airway_parameters(0.1, WallModelType::Rigid), false);
  }

  TEST(ReducedLungTreeLinearSolverTests, ForcedBatchLargeKelvinVoigtAirwaysMatchSparseSolver)
  {
    compare_forced_batch_structured_tree_and_sparse_corrections(
        "tree_linear_forced_batch_large_kelvin_voigt_airways",
        make_large_asymmetric_airway_parameters(0.1, WallModelType::KelvinVoigt), true);
  }

  TEST(ReducedLungTreeLinearSolverTests, ReusedForcedBatchMixedTreeSolverMatchesSparseSolver)
  {
    compare_reused_forced_batch_structured_tree_solver_corrections(
        "tree_linear_reused_forced_batch_mixed_solver",
        make_large_mixed_airway_terminal_unit_parameters(0.1));
  }

  TEST(ReducedLungTreeWorkflowTests, SingleTerminalUnitMatchesNoxAndNewtonSparse)
  {
    auto problem = make_single_terminal_unit_parameters(0.25);
    problem.parameters.dynamics.number_of_steps = 3;
    compare_all_solver_workflows("tree_workflow_single_terminal", problem, {"0.5*t", "0.0"},
        {.terminal_unit_volumes = true});
  }

  TEST(ReducedLungTreeWorkflowTests, SerialRigidAirwaysMatchNoxAndNewtonSparse)
  {
    auto problem = make_serial_airway_parameters(0.5);
    problem.parameters.dynamics.number_of_steps = 3;
    compare_all_solver_workflows(
        "tree_workflow_serial_airways", problem, {"t", "0.0"}, {.connection_flow_balance = true});
  }

  TEST(ReducedLungTreeWorkflowTests, RootInletFlowBoundaryMatchesNoxAndNewtonSparse)
  {
    auto problem = make_serial_airway_root_flow_parameters(0.5);
    problem.parameters.dynamics.number_of_steps = 3;
    compare_all_solver_workflows("tree_workflow_root_inlet_flow", problem, {"0.25 + 0.1*t", "0.0"},
        {.connection_flow_balance = true});
  }

  TEST(ReducedLungTreeWorkflowTests, BifurcationRigidAirwaysMatchNoxAndNewtonSparse)
  {
    auto problem = make_bifurcation_parameters(0.5);
    problem.parameters.dynamics.number_of_steps = 3;
    compare_all_solver_workflows("tree_workflow_bifurcation_airways", problem, {"t", "0.0"},
        {.bifurcation_flow_balance = true});
  }

  TEST(ReducedLungTreeWorkflowTests, MixedAirwaysAndTerminalUnitsMatchNoxAndNewtonSparse)
  {
    auto problem = make_mixed_airway_terminal_unit_parameters(0.25);
    problem.parameters.dynamics.number_of_steps = 2;
    compare_all_solver_workflows("tree_workflow_mixed_airways_terminal_units", problem,
        {"0.5*t", "0.0"}, {.terminal_unit_volumes = true, .bifurcation_flow_balance = true});
  }
}  // namespace

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_fem_discretization.hpp"
#include "4C_io_pstream.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linear_solver_method.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_rebalance.hpp"
#include "4C_reduced_lung_boundary_conditions.hpp"
#include "4C_reduced_lung_helpers.hpp"
#include "4C_reduced_lung_junctions.hpp"
#include "4C_reduced_lung_terminal_unit.hpp"
#include "4C_solver_nonlin_nox_adapter.hpp"
#include "4C_utils_function_manager.hpp"
#include "4C_utils_function_of_time.hpp"

#include <mpi.h>
#include <Teuchos_ParameterList.hpp>

#include <any>
#include <cmath>
#include <map>
#include <numbers>
#include <unordered_map>
#include <vector>

// Test for the NOX solver of the reduced lung model. The test simulates a single terminal unit with
// an Ogden elasticity model and verifies that the computed volume matches the analytical solution
// V(t)=1+t for the given parameters.
namespace
{
  using namespace FourC;
  using namespace FourC::ReducedLung;

  ReducedLungParameters make_single_tu_parameters(double dt, int steps, double radius)
  {
    ReducedLungParameters params{};
    params.air_properties = {
        .density = 1.176e-06,
        .dynamic_viscosity = 1.79105e-05,
    };
    params.dynamics = {
        .time_increment = dt,
        .number_of_steps = steps,
        .restart_every = 1,
        .results_every = 1,
        .linear_solver = 1,
        .max_nonlinear_iterations = 10,
        .nonlinear_residual_tolerance = 1e-8,
        .nonlinear_increment_tolerance = 1e-10,
    };

    params.lung_tree.topology.num_nodes = 2;
    params.lung_tree.topology.num_elements = 1;
    params.lung_tree.topology.node_coordinates =
        Core::IO::InputField<std::vector<double>>(std::unordered_map<int, std::vector<double>>{
            {1, {0.0, 0.0, 0.0}},
            {2, {radius, 0.0, 0.0}},
        });
    params.lung_tree.topology.element_nodes = Core::IO::InputField<std::vector<int>>(
        std::unordered_map<int, std::vector<int>>{{1, {1, 2}}});

    params.lung_tree.element_type =
        Core::IO::InputField<ReducedLungParameters::LungTree::ElementType>(
            std::unordered_map<int, ReducedLungParameters::LungTree::ElementType>{
                {1, ReducedLungParameters::LungTree::ElementType::TerminalUnit},
            });
    params.lung_tree.generation = Core::IO::InputField<int>(std::unordered_map<int, int>{{1, -1}});

    params.lung_tree.terminal_units.rheological_model.rheological_model_type = Core::IO::InputField<
        ReducedLungParameters::LungTree::TerminalUnits::RheologicalModel::RheologicalModelType>(
        std::unordered_map<int,
            ReducedLungParameters::LungTree::TerminalUnits::RheologicalModel::RheologicalModelType>{
            {1, ReducedLungParameters::LungTree::TerminalUnits::RheologicalModel::
                    RheologicalModelType::KelvinVoigt},
        });
    params.lung_tree.terminal_units.rheological_model.kelvin_voigt.viscosity_kelvin_voigt_eta =
        Core::IO::InputField<double>(std::unordered_map<int, double>{{1, 1.0}});

    params.lung_tree.terminal_units.elasticity_model.elasticity_model_type = Core::IO::InputField<
        ReducedLungParameters::LungTree::TerminalUnits::ElasticityModel::ElasticityModelType>(
        std::unordered_map<int,
            ReducedLungParameters::LungTree::TerminalUnits::ElasticityModel::ElasticityModelType>{
            {1, ReducedLungParameters::LungTree::TerminalUnits::ElasticityModel::
                    ElasticityModelType::Ogden},
        });
    params.lung_tree.terminal_units.elasticity_model.ogden.ogden_parameter_kappa =
        Core::IO::InputField<double>(std::unordered_map<int, double>{{1, 1.0}});
    params.lung_tree.terminal_units.elasticity_model.ogden.ogden_parameter_beta =
        Core::IO::InputField<double>(std::unordered_map<int, double>{{1, -8.0}});

    params.boundary_conditions.num_conditions = 2;
    params.boundary_conditions.bc_type =
        Core::IO::InputField<ReducedLungParameters::BoundaryConditions::Type>(
            std::unordered_map<int, ReducedLungParameters::BoundaryConditions::Type>{
                {1, ReducedLungParameters::BoundaryConditions::Type::Pressure},
                {2, ReducedLungParameters::BoundaryConditions::Type::Pressure},
            });
    params.boundary_conditions.node_id =
        Core::IO::InputField<int>(std::unordered_map<int, int>{{1, 1}, {2, 2}});
    params.boundary_conditions.value_source =
        ReducedLungParameters::BoundaryConditions::ValueSource::bc_function_id;
    params.boundary_conditions.function_id =
        Core::IO::InputField<int>(std::unordered_map<int, int>{{1, 1}, {2, 2}});

    return params;
  }

  Core::Utils::FunctionManager make_function_manager()
  {
    Core::Utils::FunctionManager function_manager;
    std::vector<std::any> functions;
    auto pressure_function = std::shared_ptr<Core::Utils::FunctionOfTime>(
        std::make_shared<Core::Utils::SymbolicFunctionOfTime>(
            std::vector<std::string>{"-1/8*((1/(1+t)) - (1/(1+t))^(-7)) + 1"},
            std::vector<std::shared_ptr<Core::Utils::FunctionVariable>>{}));
    auto zero_function = std::shared_ptr<Core::Utils::FunctionOfTime>(
        std::make_shared<Core::Utils::SymbolicFunctionOfTime>(std::vector<std::string>{"0"},
            std::vector<std::shared_ptr<Core::Utils::FunctionVariable>>{}));
    functions.emplace_back(pressure_function);
    functions.emplace_back(zero_function);
    function_manager.set_functions(functions);
    return function_manager;
  }

  TEST(ReducedLungNoxSolverTest, SingleTerminalUnitOgdenMatchesAnalyticalVolume)
  {
    const double radius = std::cbrt(3.0 / (4.0 * std::numbers::pi));
    const double dt = 0.4;
    const int steps = 5;
    const auto params = make_single_tu_parameters(dt, steps, radius);

    Core::FE::Discretization discretization("reduced_lung_nox_test", MPI_COMM_WORLD, 3);
    Core::Rebalance::RebalanceParameters rebalance_parameters;
    build_discretization_from_topology(
        discretization, params.lung_tree.topology, rebalance_parameters);
    discretization.fill_complete();

    Airways::AirwayContainer airways;
    TerminalUnits::TerminalUnitContainer terminal_units;
    std::map<int, int> dof_per_ele;
    int n_airways = 0;
    int n_terminal_units = 0;
    create_local_element_models(
        discretization, params, airways, terminal_units, dof_per_ele, n_airways, n_terminal_units);

    std::map<int, int> first_global_dof_of_ele;
    std::map<int, int> global_dof_per_ele;
    create_global_dof_maps(
        dof_per_ele, MPI_COMM_WORLD, global_dof_per_ele, first_global_dof_of_ele);
    assign_global_dof_ids_to_models(first_global_dof_of_ele, airways, terminal_units);

    TerminalUnits::create_evaluators(terminal_units);
    Airways::create_evaluators(airways);

    auto global_ele_ids_per_node = create_global_ele_ids_per_node(discretization, MPI_COMM_WORLD);

    BoundaryConditions::BoundaryConditionContainer boundary_conditions;
    Junctions::ConnectionData connections;
    Junctions::BifurcationData bifurcations;
    const auto function_manager = make_function_manager();

    BoundaryConditions::create_boundary_conditions(discretization, params, global_ele_ids_per_node,
        global_dof_per_ele, first_global_dof_of_ele, function_manager, boundary_conditions);
    BoundaryConditions::create_evaluators(boundary_conditions);

    Junctions::create_junctions(discretization, global_ele_ids_per_node, global_dof_per_ele,
        first_global_dof_of_ele, connections, bifurcations);

    int n_local_equations = 0;
    Airways::assign_local_equation_ids(airways, n_local_equations);
    TerminalUnits::assign_local_equation_ids(terminal_units, n_local_equations);
    Junctions::assign_junction_local_equation_ids(connections, bifurcations, n_local_equations);
    BoundaryConditions::assign_local_equation_ids(boundary_conditions, n_local_equations);

    const Core::LinAlg::Map locally_owned_dof_map =
        create_domain_map(MPI_COMM_WORLD, airways, terminal_units);
    const Core::LinAlg::Map row_map = create_row_map(
        MPI_COMM_WORLD, airways, terminal_units, connections, bifurcations, boundary_conditions);
    const Core::LinAlg::Map locally_relevant_dof_map =
        create_column_map(MPI_COMM_WORLD, airways, terminal_units, global_dof_per_ele,
            first_global_dof_of_ele, connections, bifurcations, boundary_conditions);

    Junctions::assign_junction_global_equation_ids(row_map, connections, bifurcations);
    BoundaryConditions::assign_global_equation_ids(row_map, boundary_conditions);

    Airways::assign_local_dof_ids(locally_relevant_dof_map, airways);
    TerminalUnits::assign_local_dof_ids(locally_relevant_dof_map, terminal_units);
    Junctions::assign_junction_local_dof_ids(locally_relevant_dof_map, connections, bifurcations);
    BoundaryConditions::assign_local_dof_ids(locally_relevant_dof_map, boundary_conditions);

    Core::LinAlg::Vector<double> dofs(locally_owned_dof_map, true);
    Core::LinAlg::Vector<double> dofs_n(locally_owned_dof_map, true);
    Core::LinAlg::Vector<double> locally_relevant_dofs(locally_relevant_dof_map, true);
    Core::LinAlg::Vector<double> x(row_map, true);
    Core::LinAlg::Vector<double> rhs(row_map, true);
    Core::LinAlg::SparseMatrix sysmat(row_map, locally_relevant_dof_map, 3);

    TerminalUnits::update_internal_state_vectors(terminal_units, locally_relevant_dofs, dt);
    Airways::update_internal_state_vectors(airways, locally_relevant_dofs, dt);

    Teuchos::ParameterList solver_params;
    solver_params.set("SOLVER", Core::LinearSolver::SolverType::UMFPACK);
    solver_params.set("NAME", "Reduced_lung_solver");
    const auto get_solver_params = [&](int) -> const Teuchos::ParameterList&
    { return solver_params; };

    auto linear_solver = std::make_shared<Core::LinAlg::Solver>(
        solver_params, MPI_COMM_WORLD, get_solver_params, Core::IO::Verbositylevel::minimal);

    auto nox_params = create_nox_parameter_list(params.dynamics.nonlinear_residual_tolerance,
        params.dynamics.nonlinear_increment_tolerance, params.dynamics.max_nonlinear_iterations);

    FourC::NOX::Nln::Adapter::SolverMap solver_map;
    solver_map[FourC::NOX::Nln::sol_generic] = linear_solver;
    FourC::NOX::Nln::Adapter nox_adapter(MPI_COMM_WORLD, nox_params, std::move(solver_map));
    auto nox_callbacks = create_nox_adapter_callbacks(dofs, locally_relevant_dofs, airways,
        terminal_units, connections, bifurcations, boundary_conditions, dt);
    nox_adapter.residual = nox_callbacks.residual;
    nox_adapter.jacobian = nox_callbacks.jacobian;

    const int owns_terminal_unit = terminal_units.models.empty() ? 0 : 1;
    int owner_count = 0;
    MPI_Allreduce(&owns_terminal_unit, &owner_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    ASSERT_EQ(owner_count, 1);

    TerminalUnits::TerminalUnitModel* model = nullptr;
    if (owns_terminal_unit)
    {
      ASSERT_EQ(terminal_units.models.size(), 1u);
      model = &terminal_units.models.front();
      ASSERT_EQ(model->data.volume_v.size(), 1u);
    }

    for (int n = 1; n <= steps; ++n)
    {
      dofs_n.update(1.0, dofs, 0.0);
      const double current_time = n * dt;
      nox_callbacks.set_time(current_time);
      nox_adapter.solve(x, rhs, sysmat);
      export_to(x, dofs);
      export_to(dofs, locally_relevant_dofs);

      TerminalUnits::end_of_timestep_routine(terminal_units, locally_relevant_dofs, dt);
      Airways::end_of_timestep_routine(airways, locally_relevant_dofs, dt);

      if (owns_terminal_unit)
      {
        const double expected_volume = 1.0 + current_time;
        EXPECT_NEAR(model->data.volume_v[0], expected_volume, 1e-6);
      }
    }
  }
}  // namespace

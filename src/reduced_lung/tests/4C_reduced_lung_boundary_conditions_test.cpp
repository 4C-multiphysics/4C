// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_reduced_lung_boundary_conditions.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_linalg_map.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_red_airways_elementbase.hpp"
#include "4C_unittest_utils_assertions_test.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_function_manager.hpp"
#include "4C_utils_function_of_time.hpp"

#include <mpi.h>

#include <array>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{
  using namespace FourC;
  using namespace FourC::ReducedLung;
  using namespace FourC::ReducedLung::BoundaryConditions;

  std::unique_ptr<Core::FE::Discretization> make_airway_discretization(
      const std::vector<int>& node_ids, const std::vector<std::array<int, 2>>& element_nodes)
  {
    auto dis =
        std::make_unique<Core::FE::Discretization>("boundary_conditions_test", MPI_COMM_WORLD, 3);

    for (int node_id : node_ids)
    {
      std::array<double, 3> coords{static_cast<double>(node_id), 0.0, 0.0};
      dis->add_node(coords, node_id, nullptr);
    }

    for (size_t i = 0; i < element_nodes.size(); ++i)
    {
      auto ele = std::make_shared<Discret::Elements::RedAirway>(static_cast<int>(i), 0);
      ele->set_node_ids(2, element_nodes[i].data());
      dis->add_element(ele);
    }

    dis->fill_complete(Core::FE::OptionsFillComplete::none());
    return dis;
  }

  using InputBc = ReducedLungParameters::BoundaryConditions;

  //! Boundary condition input together with the constrained nodes of the mesh it refers to.
  struct BcInput
  {
    ReducedLungParameters parameters;
    std::map<int, std::vector<int>> bc_nodes;
  };

  //! The vector of definitions of the given type, so tests can build inputs generically.
  std::vector<InputBc::Definition>& definitions_of_type(InputBc& boundary_conditions, Type type)
  {
    return type == Type::Pressure ? boundary_conditions.pressure : boundary_conditions.flow;
  }

  InputBc::Definition make_definition(int id, int function_id)
  {
    return InputBc::Definition{.id = id, .function_id = function_id};
  }

  //! A FunctionManager with one constant-valued SymbolicFunctionOfTime per entry of @p values,
  //! registered as function ids 1, 2, ... in order.
  Core::Utils::FunctionManager make_function_manager(const std::vector<double>& values)
  {
    Core::Utils::FunctionManager function_manager;
    std::vector<std::any> functions;
    for (const double value : values)
    {
      functions.emplace_back(std::shared_ptr<Core::Utils::FunctionOfTime>(
          std::make_shared<Core::Utils::SymbolicFunctionOfTime>(
              std::vector<std::string>{std::to_string(value)},
              std::vector<std::shared_ptr<Core::Utils::FunctionVariable>>{})));
    }
    function_manager.set_functions(functions);
    return function_manager;
  }

  BcInput make_constant_parameters()
  {
    BcInput input{};
    // Node 0 carries both a pressure and a flow condition. The mesh cannot express this
    // through its `bc_id` array, but the container API can, so the grouping is still tested here.
    // Both pressure definitions share function id 1, so they group into a single model.
    input.bc_nodes = {{1, {0}}, {2, {2}}, {3, {0}}};
    input.parameters.boundary_conditions.pressure = {make_definition(1, 1), make_definition(2, 1)};
    input.parameters.boundary_conditions.flow = {make_definition(3, 2)};
    return input;
  }

  BcInput make_single_bc_parameters(int node_id, Type type)
  {
    BcInput input{};
    input.bc_nodes = {{1, {node_id}}};
    definitions_of_type(input.parameters.boundary_conditions, type) = {make_definition(1, 1)};
    return input;
  }

  BcInput make_function_bc_parameters(int node_id, Type type, int function_id)
  {
    BcInput input{};
    input.bc_nodes = {{1, {node_id}}};
    definitions_of_type(input.parameters.boundary_conditions, type) = {
        make_definition(1, function_id)};
    return input;
  }

  BcInput make_duplicate_type_parameters()
  {
    BcInput input{};
    // Two definitions claim the same node with the same condition type.
    input.bc_nodes = {{1, {0}}, {2, {0}}};
    input.parameters.boundary_conditions.pressure = {make_definition(1, 1), make_definition(2, 1)};
    return input;
  }

  BoundaryConditionModel* find_model(BoundaryConditionContainer& container, Type type)
  {
    for (auto& model : container.models)
    {
      if (model.type == type)
      {
        return &model;
      }
    }
    return nullptr;
  }

  void expect_row_entry(Core::LinAlg::SparseMatrix& mat, int row, int col, double expected_value)
  {
    int n_entries = 0;
    double* values = nullptr;
    int* cols = nullptr;
    mat.extract_my_row_view(row, n_entries, values, cols);

    ASSERT_EQ(n_entries, 1);
    EXPECT_EQ(cols[0], col);
    EXPECT_DOUBLE_EQ(values[0], expected_value);
  }

  struct BoundaryConditionFixture
  {
    std::unique_ptr<Core::FE::Discretization> discretization;
    ReducedLungParameters parameters;
    std::map<int, std::vector<int>> bc_nodes;
    std::map<int, std::vector<int>> ele_ids_per_node;
    std::map<int, int> global_dof_per_ele;
    std::map<int, int> first_global_dof_of_ele;

    void set_bc_input(BcInput&& input)
    {
      parameters = std::move(input.parameters);
      bc_nodes = std::move(input.bc_nodes);
    }
  };

  BoundaryConditionFixture make_fixture()
  {
    BoundaryConditionFixture fixture;
    fixture.discretization = make_airway_discretization({0, 1, 2}, {{0, 1}, {1, 2}});
    fixture.set_bc_input(make_constant_parameters());
    fixture.ele_ids_per_node = {{0, {0}}, {1, {0, 1}}, {2, {1}}};
    fixture.global_dof_per_ele = {{0, 3}, {1, 3}};
    fixture.first_global_dof_of_ele = {{0, 0}, {1, 3}};
    return fixture;
  }

  BoundaryConditionContainer create_boundary_conditions_from_fixture(
      const BoundaryConditionFixture& fixture, const Core::Utils::FunctionManager& function_manager)
  {
    BoundaryConditionContainer boundary_conditions;
    create_boundary_conditions(*fixture.discretization, fixture.parameters, fixture.bc_nodes,
        fixture.ele_ids_per_node, fixture.global_dof_per_ele, fixture.first_global_dof_of_ele,
        function_manager, boundary_conditions);
    return boundary_conditions;
  }

  void skip_if_parallel()
  {
    int comm_size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    if (comm_size != 1)
    {
      GTEST_SKIP() << "Boundary condition creation tests require a serial communicator.";
    }
  }

  TEST(BoundaryConditionsTests, CreateBoundaryConditionsGroupsByType)
  {
    skip_if_parallel();

    auto fixture = make_fixture();
    auto function_manager = make_function_manager({2.5, -1.0});
    auto boundary_conditions = create_boundary_conditions_from_fixture(fixture, function_manager);

    ASSERT_EQ(boundary_conditions.models.size(), 2u);

    auto* pressure_model = find_model(boundary_conditions, Type::Pressure);
    auto* flow_model = find_model(boundary_conditions, Type::Flow);
    ASSERT_NE(pressure_model, nullptr);
    ASSERT_NE(flow_model, nullptr);

    EXPECT_EQ(pressure_model->function_id, 1);
    EXPECT_EQ(flow_model->function_id, 2);

    EXPECT_EQ(pressure_model->data.size(), 2u);
    EXPECT_EQ(pressure_model->data.node_id, (std::vector<int>{0, 2}));
    EXPECT_EQ(pressure_model->data.global_element_id, (std::vector<int>{0, 1}));
    EXPECT_EQ(pressure_model->data.global_dof_id, (std::vector<int>{0, 4}));
    EXPECT_EQ(pressure_model->data.local_bc_id, (std::vector<int>{0, 1}));

    EXPECT_EQ(flow_model->data.size(), 1u);
    EXPECT_EQ(flow_model->data.node_id, (std::vector<int>{0}));
    EXPECT_EQ(flow_model->data.global_element_id, (std::vector<int>{0}));
    EXPECT_EQ(flow_model->data.global_dof_id, (std::vector<int>{2}));
    EXPECT_EQ(flow_model->data.local_bc_id, (std::vector<int>{2}));
  }

  TEST(BoundaryConditionsTests, ResidualAssemblyConstant)
  {
    skip_if_parallel();

    auto fixture = make_fixture();
    // A constant is expressed as a constant function, since the input has no separate constant
    // option: function 1 (pressure) is 2.5, function 2 (flow) is -1.0.
    auto function_manager = make_function_manager({2.5, -1.0});
    auto boundary_conditions = create_boundary_conditions_from_fixture(fixture, function_manager);

    int n_local_equations = 0;
    assign_local_equation_ids(boundary_conditions, n_local_equations);

    std::array<int, 3> global_dofs{0, 2, 4};
    Core::LinAlg::Map col_map(-1, std::span<const int>(global_dofs), 0, MPI_COMM_WORLD);
    assign_local_dof_ids(col_map, boundary_conditions);
    create_evaluators(boundary_conditions);

    Core::LinAlg::Map row_map(-1, n_local_equations, 0, MPI_COMM_WORLD);
    Core::LinAlg::Vector<double> rhs(row_map, true);
    Core::LinAlg::Vector<double> locally_relevant_dofs(col_map, true);

    auto dof_values = locally_relevant_dofs.get_values();
    dof_values[0] = 10.0;  // global dof 0
    dof_values[1] = 4.0;   // global dof 2
    dof_values[2] = 7.0;   // global dof 4

    update_residual_vector(rhs, boundary_conditions, locally_relevant_dofs, 0.0);

    for (const auto& model : boundary_conditions.models)
    {
      const double bc_value = model.function->evaluate(0.0);
      for (size_t i = 0; i < model.data.size(); ++i)
      {
        const int eq = model.data.local_equation_id[i];
        const int ldof = model.data.local_dof_id[i];
        const double expected = locally_relevant_dofs.local_values_as_span()[ldof] - bc_value;
        EXPECT_DOUBLE_EQ(rhs.local_values_as_span()[eq], expected);
      }
    }
  }

  TEST(BoundaryConditionsTests, JacobianAssembledOnce)
  {
    skip_if_parallel();

    auto fixture = make_fixture();
    auto function_manager = make_function_manager({2.5, -1.0});
    auto boundary_conditions = create_boundary_conditions_from_fixture(fixture, function_manager);

    int n_local_equations = 0;
    assign_local_equation_ids(boundary_conditions, n_local_equations);

    std::array<int, 3> global_dofs{0, 2, 4};
    Core::LinAlg::Map col_map(-1, std::span<const int>(global_dofs), 0, MPI_COMM_WORLD);
    assign_local_dof_ids(col_map, boundary_conditions);
    create_evaluators(boundary_conditions);

    Core::LinAlg::Map row_map(-1, n_local_equations, 0, MPI_COMM_WORLD);
    Core::LinAlg::SparseMatrix jac(row_map, col_map, 1);

    update_jacobian(jac, boundary_conditions);
    jac.complete();

    for (const auto& model : boundary_conditions.models)
    {
      for (size_t i = 0; i < model.data.size(); ++i)
      {
        expect_row_entry(jac, model.data.local_equation_id[i], model.data.local_dof_id[i], 1.0);
      }
    }

    update_jacobian(jac, boundary_conditions);

    for (const auto& model : boundary_conditions.models)
    {
      for (size_t i = 0; i < model.data.size(); ++i)
      {
        expect_row_entry(jac, model.data.local_equation_id[i], model.data.local_dof_id[i], 1.0);
      }
    }
  }

  TEST(BoundaryConditionsTests, CreateBoundaryConditionsMissingAdjacencyThrows)
  {
    skip_if_parallel();

    auto fixture = make_fixture();
    fixture.set_bc_input(make_single_bc_parameters(0, Type::Pressure));
    fixture.ele_ids_per_node.erase(0);

    BoundaryConditionContainer boundary_conditions;
    Core::Utils::FunctionManager function_manager;
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        create_boundary_conditions(*fixture.discretization, fixture.parameters, fixture.bc_nodes,
            fixture.ele_ids_per_node, fixture.global_dof_per_ele, fixture.first_global_dof_of_ele,
            function_manager, boundary_conditions),
        Core::Exception, "is not part of the tree");
  }

  TEST(BoundaryConditionsTests, CreateBoundaryConditionsMultipleAdjacencyThrows)
  {
    skip_if_parallel();

    auto fixture = make_fixture();
    fixture.set_bc_input(make_single_bc_parameters(1, Type::Pressure));

    BoundaryConditionContainer boundary_conditions;
    Core::Utils::FunctionManager function_manager;
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        create_boundary_conditions(*fixture.discretization, fixture.parameters, fixture.bc_nodes,
            fixture.ele_ids_per_node, fixture.global_dof_per_ele, fixture.first_global_dof_of_ele,
            function_manager, boundary_conditions),
        Core::Exception, "must connect to exactly one element");
  }

  TEST(BoundaryConditionsTests, ResidualAssemblyFunctionValue)
  {
    skip_if_parallel();

    Core::Utils::FunctionManager function_manager;
    std::vector<std::any> functions;
    auto function = std::shared_ptr<Core::Utils::FunctionOfTime>(
        std::make_shared<Core::Utils::SymbolicFunctionOfTime>(std::vector<std::string>{"2.0 * t"},
            std::vector<std::shared_ptr<Core::Utils::FunctionVariable>>{}));
    functions.emplace_back(function);
    function_manager.set_functions(functions);

    auto fixture = make_fixture();
    fixture.set_bc_input(make_function_bc_parameters(0, Type::Pressure, 1));

    auto boundary_conditions = create_boundary_conditions_from_fixture(fixture, function_manager);

    int n_local_equations = 0;
    assign_local_equation_ids(boundary_conditions, n_local_equations);

    std::array<int, 1> global_dofs{0};
    Core::LinAlg::Map col_map(-1, std::span<const int>(global_dofs), 0, MPI_COMM_WORLD);
    assign_local_dof_ids(col_map, boundary_conditions);
    create_evaluators(boundary_conditions);

    Core::LinAlg::Map row_map(-1, n_local_equations, 0, MPI_COMM_WORLD);
    Core::LinAlg::Vector<double> rhs(row_map, true);
    Core::LinAlg::Vector<double> locally_relevant_dofs(col_map, true);
    locally_relevant_dofs.get_values()[0] = 1.0;

    const double time = 1.5;
    update_residual_vector(rhs, boundary_conditions, locally_relevant_dofs, time);

    ASSERT_EQ(boundary_conditions.models.size(), 1u);
    const auto& model = boundary_conditions.models.front();
    ASSERT_EQ(model.data.size(), 1u);
    EXPECT_DOUBLE_EQ(rhs.local_values_as_span()[model.data.local_equation_id[0]], 1.0 - 2.0 * time);
  }

  TEST(BoundaryConditionsTests, CreateBoundaryConditionsDuplicateTypeThrows)
  {
    skip_if_parallel();

    auto fixture = make_fixture();
    fixture.set_bc_input(make_duplicate_type_parameters());

    BoundaryConditionContainer boundary_conditions;
    // Both duplicate definitions share function id 1, so the first node's model can be created
    // before the duplicate-assignment check on the second definition throws.
    auto function_manager = make_function_manager({1.0});
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        create_boundary_conditions(*fixture.discretization, fixture.parameters, fixture.bc_nodes,
            fixture.ele_ids_per_node, fixture.global_dof_per_ele, fixture.first_global_dof_of_ele,
            function_manager, boundary_conditions),
        Core::Exception, "Multiple pressure boundary conditions assigned to node");
  }

  TEST(BoundaryConditionsTests, CreateBoundaryConditionsUndefinedMeshIdThrows)
  {
    skip_if_parallel();

    auto fixture = make_fixture();
    fixture.set_bc_input(make_single_bc_parameters(0, Type::Pressure));
    // The mesh refers to a definition that the input file does not provide.
    fixture.bc_nodes[7] = {2};

    BoundaryConditionContainer boundary_conditions;
    Core::Utils::FunctionManager function_manager;
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        create_boundary_conditions(*fixture.discretization, fixture.parameters, fixture.bc_nodes,
            fixture.ele_ids_per_node, fixture.global_dof_per_ele, fixture.first_global_dof_of_ele,
            function_manager, boundary_conditions),
        Core::Exception, "no definition with this id exists in the input file");
  }

  TEST(BoundaryConditionsTests, CreateBoundaryConditionsUnusedDefinitionThrows)
  {
    skip_if_parallel();

    auto fixture = make_fixture();
    auto input = make_single_bc_parameters(0, Type::Pressure);
    // The input file defines a condition that no node of the mesh refers to.
    input.parameters.boundary_conditions.pressure.push_back(make_definition(2, 1));
    fixture.set_bc_input(std::move(input));

    BoundaryConditionContainer boundary_conditions;
    Core::Utils::FunctionManager function_manager;
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        create_boundary_conditions(*fixture.discretization, fixture.parameters, fixture.bc_nodes,
            fixture.ele_ids_per_node, fixture.global_dof_per_ele, fixture.first_global_dof_of_ele,
            function_manager, boundary_conditions),
        Core::Exception, "is not used by any node of the mesh");
  }

  TEST(BoundaryConditionsTests, CreateBoundaryConditionsDuplicateDefinitionIdThrows)
  {
    skip_if_parallel();

    auto fixture = make_fixture();
    auto input = make_single_bc_parameters(0, Type::Pressure);
    input.parameters.boundary_conditions.flow.push_back(make_definition(1, 1));
    fixture.set_bc_input(std::move(input));

    BoundaryConditionContainer boundary_conditions;
    Core::Utils::FunctionManager function_manager;
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        create_boundary_conditions(*fixture.discretization, fixture.parameters, fixture.bc_nodes,
            fixture.ele_ids_per_node, fixture.global_dof_per_ele, fixture.first_global_dof_of_ele,
            function_manager, boundary_conditions),
        Core::Exception, "is defined more than once");
  }
}  // namespace

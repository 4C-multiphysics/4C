// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_reduced_lung_boundary_conditions.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
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

  ReducedLungParameters make_constant_parameters()
  {
    ReducedLungParameters params{};
    params.lung_tree.topology.num_nodes = 3;

    params.boundary_conditions.num_conditions = 3;
    params.boundary_conditions.bc_type =
        Core::IO::InputField<ReducedLungParameters::BoundaryConditions::Type>(
            std::unordered_map<int, ReducedLungParameters::BoundaryConditions::Type>{
                {1, ReducedLungParameters::BoundaryConditions::Type::Pressure},
                {2, ReducedLungParameters::BoundaryConditions::Type::Pressure},
                {3, ReducedLungParameters::BoundaryConditions::Type::Flow},
            });
    params.boundary_conditions.node_id =
        Core::IO::InputField<int>(std::unordered_map<int, int>{{1, 1}, {2, 3}, {3, 1}});
    params.boundary_conditions.value_source =
        ReducedLungParameters::BoundaryConditions::ValueSource::bc_value;
    params.boundary_conditions.value = Core::IO::InputField<double>(
        std::unordered_map<int, double>{{1, 2.5}, {2, 4.0}, {3, -1.0}});

    return params;
  }

  ReducedLungParameters make_single_bc_parameters(
      int node_id_one_based, ReducedLungParameters::BoundaryConditions::Type type)
  {
    ReducedLungParameters params{};
    params.lung_tree.topology.num_nodes = 3;
    params.boundary_conditions.num_conditions = 1;
    params.boundary_conditions.bc_type =
        Core::IO::InputField<ReducedLungParameters::BoundaryConditions::Type>(
            std::unordered_map<int, ReducedLungParameters::BoundaryConditions::Type>{
                {1, type},
            });
    params.boundary_conditions.node_id =
        Core::IO::InputField<int>(std::unordered_map<int, int>{{1, node_id_one_based}});
    params.boundary_conditions.value_source =
        ReducedLungParameters::BoundaryConditions::ValueSource::bc_value;
    params.boundary_conditions.value =
        Core::IO::InputField<double>(std::unordered_map<int, double>{{1, 0.0}});
    return params;
  }

  ReducedLungParameters make_function_bc_parameters(
      int node_id_one_based, ReducedLungParameters::BoundaryConditions::Type type, int function_id)
  {
    ReducedLungParameters params{};
    params.lung_tree.topology.num_nodes = 3;
    params.boundary_conditions.num_conditions = 1;
    params.boundary_conditions.bc_type =
        Core::IO::InputField<ReducedLungParameters::BoundaryConditions::Type>(
            std::unordered_map<int, ReducedLungParameters::BoundaryConditions::Type>{
                {1, type},
            });
    params.boundary_conditions.node_id =
        Core::IO::InputField<int>(std::unordered_map<int, int>{{1, node_id_one_based}});
    params.boundary_conditions.value_source =
        ReducedLungParameters::BoundaryConditions::ValueSource::bc_function_id;
    params.boundary_conditions.function_id =
        Core::IO::InputField<int>(std::unordered_map<int, int>{{1, function_id}});
    params.boundary_conditions.value =
        Core::IO::InputField<double>(std::unordered_map<int, double>{{1, 0.0}});
    return params;
  }

  ReducedLungParameters make_duplicate_type_parameters()
  {
    ReducedLungParameters params{};
    params.lung_tree.topology.num_nodes = 3;
    params.boundary_conditions.num_conditions = 2;
    params.boundary_conditions.bc_type =
        Core::IO::InputField<ReducedLungParameters::BoundaryConditions::Type>(
            std::unordered_map<int, ReducedLungParameters::BoundaryConditions::Type>{
                {1, ReducedLungParameters::BoundaryConditions::Type::Pressure},
                {2, ReducedLungParameters::BoundaryConditions::Type::Pressure},
            });
    params.boundary_conditions.node_id =
        Core::IO::InputField<int>(std::unordered_map<int, int>{{1, 1}, {2, 1}});
    params.boundary_conditions.value_source =
        ReducedLungParameters::BoundaryConditions::ValueSource::bc_value;
    params.boundary_conditions.value =
        Core::IO::InputField<double>(std::unordered_map<int, double>{{1, 1.0}, {2, 2.0}});
    return params;
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
    std::map<int, std::vector<int>> ele_ids_per_node;
    std::map<int, int> global_dof_per_ele;
    std::map<int, int> first_global_dof_of_ele;
  };

  BoundaryConditionFixture make_fixture()
  {
    BoundaryConditionFixture fixture;
    fixture.discretization = make_airway_discretization({0, 1, 2}, {{0, 1}, {1, 2}});
    fixture.parameters = make_constant_parameters();
    fixture.ele_ids_per_node = {{0, {0}}, {1, {0, 1}}, {2, {1}}};
    fixture.global_dof_per_ele = {{0, 3}, {1, 3}};
    fixture.first_global_dof_of_ele = {{0, 0}, {1, 3}};
    return fixture;
  }

  BoundaryConditionContainer create_boundary_conditions_from_fixture(
      const BoundaryConditionFixture& fixture)
  {
    BoundaryConditionContainer boundary_conditions;
    create_boundary_conditions(*fixture.discretization, fixture.parameters,
        fixture.ele_ids_per_node, fixture.global_dof_per_ele, fixture.first_global_dof_of_ele,
        boundary_conditions);
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

  struct FunctionManagerGuard
  {
    Core::Utils::FunctionManager original;

    FunctionManagerGuard() : original(Global::Problem::instance()->function_manager()) {}

    ~FunctionManagerGuard()
    {
      Global::Problem::instance()->set_function_manager(std::move(original));
    }
  };

  TEST(BoundaryConditionsTests, CreateBoundaryConditionsGroupsByType)
  {
    skip_if_parallel();

    auto fixture = make_fixture();
    auto boundary_conditions = create_boundary_conditions_from_fixture(fixture);

    ASSERT_EQ(boundary_conditions.models.size(), 2u);

    auto* pressure_model = find_model(boundary_conditions, Type::Pressure);
    auto* flow_model = find_model(boundary_conditions, Type::Flow);
    ASSERT_NE(pressure_model, nullptr);
    ASSERT_NE(flow_model, nullptr);

    EXPECT_EQ(pressure_model->value_source, ValueSource::constant_value);
    EXPECT_EQ(flow_model->value_source, ValueSource::constant_value);

    EXPECT_EQ(pressure_model->data.size(), 2u);
    EXPECT_EQ(pressure_model->values, (std::vector<double>{2.5, 4.0}));
    EXPECT_EQ(pressure_model->data.node_id, (std::vector<int>{0, 2}));
    EXPECT_EQ(pressure_model->data.global_element_id, (std::vector<int>{0, 1}));
    EXPECT_EQ(pressure_model->data.global_dof_id, (std::vector<int>{0, 4}));
    EXPECT_EQ(pressure_model->data.local_bc_id, (std::vector<int>{0, 1}));

    EXPECT_EQ(flow_model->data.size(), 1u);
    EXPECT_EQ(flow_model->values, (std::vector<double>{-1.0}));
    EXPECT_EQ(flow_model->data.node_id, (std::vector<int>{0}));
    EXPECT_EQ(flow_model->data.global_element_id, (std::vector<int>{0}));
    EXPECT_EQ(flow_model->data.global_dof_id, (std::vector<int>{2}));
    EXPECT_EQ(flow_model->data.local_bc_id, (std::vector<int>{2}));
  }

  TEST(BoundaryConditionsTests, ResidualAssemblyConstant)
  {
    skip_if_parallel();

    auto fixture = make_fixture();
    auto boundary_conditions = create_boundary_conditions_from_fixture(fixture);

    int n_local_equations = 0;
    assign_local_equation_ids(boundary_conditions, n_local_equations);

    std::array<int, 3> global_dofs{0, 2, 4};
    Core::LinAlg::Map col_map(-1, global_dofs.size(), global_dofs.data(), 0, MPI_COMM_WORLD);
    assign_local_dof_ids(col_map, boundary_conditions);
    create_evaluators(boundary_conditions);

    Core::LinAlg::Map row_map(-1, n_local_equations, 0, MPI_COMM_WORLD);
    Core::LinAlg::Vector<double> rhs(row_map, true);
    Core::LinAlg::Vector<double> locally_relevant_dofs(col_map, true);

    auto dof_values = locally_relevant_dofs.get_values();
    dof_values[0] = 10.0;  // global dof 0
    dof_values[1] = 4.0;   // global dof 2
    dof_values[2] = 7.0;   // global dof 4

    update_negative_residual_vector(rhs, boundary_conditions, locally_relevant_dofs, 0.0);

    for (const auto& model : boundary_conditions.models)
    {
      for (size_t i = 0; i < model.data.size(); ++i)
      {
        const int eq = model.data.local_equation_id[i];
        const int ldof = model.data.local_dof_id[i];
        const double expected =
            -locally_relevant_dofs.local_values_as_span()[ldof] + model.values[i];
        EXPECT_DOUBLE_EQ(rhs.local_values_as_span()[eq], expected);
      }
    }
  }

  TEST(BoundaryConditionsTests, JacobianAssembledOnce)
  {
    skip_if_parallel();

    auto fixture = make_fixture();
    auto boundary_conditions = create_boundary_conditions_from_fixture(fixture);

    int n_local_equations = 0;
    assign_local_equation_ids(boundary_conditions, n_local_equations);

    std::array<int, 3> global_dofs{0, 2, 4};
    Core::LinAlg::Map col_map(-1, global_dofs.size(), global_dofs.data(), 0, MPI_COMM_WORLD);
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
    fixture.parameters =
        make_single_bc_parameters(1, ReducedLungParameters::BoundaryConditions::Type::Pressure);
    fixture.ele_ids_per_node.erase(0);

    BoundaryConditionContainer boundary_conditions;
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        create_boundary_conditions(*fixture.discretization, fixture.parameters,
            fixture.ele_ids_per_node, fixture.global_dof_per_ele, fixture.first_global_dof_of_ele,
            boundary_conditions),
        Core::Exception, "not part of the topology");
  }

  TEST(BoundaryConditionsTests, CreateBoundaryConditionsMultipleAdjacencyThrows)
  {
    skip_if_parallel();

    auto fixture = make_fixture();
    fixture.parameters =
        make_single_bc_parameters(2, ReducedLungParameters::BoundaryConditions::Type::Pressure);

    BoundaryConditionContainer boundary_conditions;
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        create_boundary_conditions(*fixture.discretization, fixture.parameters,
            fixture.ele_ids_per_node, fixture.global_dof_per_ele, fixture.first_global_dof_of_ele,
            boundary_conditions),
        Core::Exception, "must connect to exactly one element");
  }

  TEST(BoundaryConditionsTests, ResidualAssemblyFunctionValue)
  {
    skip_if_parallel();

    FunctionManagerGuard guard;

    Core::Utils::FunctionManager function_manager;
    std::vector<std::any> functions;
    auto function = std::shared_ptr<Core::Utils::FunctionOfTime>(
        std::make_shared<Core::Utils::SymbolicFunctionOfTime>(std::vector<std::string>{"2.0 * t"},
            std::vector<std::shared_ptr<Core::Utils::FunctionVariable>>{}));
    functions.emplace_back(function);
    function_manager.set_functions(functions);
    Global::Problem::instance()->set_function_manager(std::move(function_manager));

    auto fixture = make_fixture();
    fixture.parameters = make_function_bc_parameters(
        1, ReducedLungParameters::BoundaryConditions::Type::Pressure, 1);

    auto boundary_conditions = create_boundary_conditions_from_fixture(fixture);

    int n_local_equations = 0;
    assign_local_equation_ids(boundary_conditions, n_local_equations);

    std::array<int, 1> global_dofs{0};
    Core::LinAlg::Map col_map(-1, global_dofs.size(), global_dofs.data(), 0, MPI_COMM_WORLD);
    assign_local_dof_ids(col_map, boundary_conditions);
    create_evaluators(boundary_conditions);

    Core::LinAlg::Map row_map(-1, n_local_equations, 0, MPI_COMM_WORLD);
    Core::LinAlg::Vector<double> rhs(row_map, true);
    Core::LinAlg::Vector<double> locally_relevant_dofs(col_map, true);
    locally_relevant_dofs.get_values()[0] = 1.0;

    const double time = 1.5;
    update_negative_residual_vector(rhs, boundary_conditions, locally_relevant_dofs, time);

    ASSERT_EQ(boundary_conditions.models.size(), 1u);
    const auto& model = boundary_conditions.models.front();
    ASSERT_EQ(model.data.size(), 1u);
    EXPECT_DOUBLE_EQ(
        rhs.local_values_as_span()[model.data.local_equation_id[0]], -1.0 + 2.0 * time);
  }

  TEST(BoundaryConditionsTests, CreateBoundaryConditionsDuplicateTypeThrows)
  {
    skip_if_parallel();

    auto fixture = make_fixture();
    fixture.parameters = make_duplicate_type_parameters();

    BoundaryConditionContainer boundary_conditions;
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        create_boundary_conditions(*fixture.discretization, fixture.parameters,
            fixture.ele_ids_per_node, fixture.global_dof_per_ele, fixture.first_global_dof_of_ele,
            boundary_conditions),
        Core::Exception, "Multiple pressure boundary conditions assigned to node");
  }
}  // namespace

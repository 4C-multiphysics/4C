// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_reduced_lung_1d_pipe_flow_main.hpp"

#include "4C_art_net_impl_stationary.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_assemblestrategy.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_fem_general_utils_gausspoints.hpp"
#include "4C_global_data.hpp"
#include "4C_io_discretization_visualization_writer_mesh.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_io_pstream.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_mat_vplast_law.hpp"
#include "4C_reduced_lung_1d_pipe_flow_input.hpp"
#include "4C_reduced_lung_1d_pipe_flow_resulttest.hpp"
#include "4C_reduced_lung_1d_pipe_flow_terminal_unit.hpp"
#include "4C_structure_new_timint_base.hpp"
#include "4C_utils_function_of_time.hpp"
#include "4C_utils_local_newton.hpp"

#include <boost/graph/subgraph.hpp>
#include <Teuchos_StandardParameterEntryValidators.hpp>

#include <functional>

FOUR_C_NAMESPACE_OPEN
namespace ReducedLung1dPipeFlow
{
  namespace
  {
    struct ReducedLung1dPipeFlowContext
    {
      std::shared_ptr<Core::FE::Discretization> artery_discretization;
      const Teuchos::ParameterList* reduced_lung_parameters;
      std::function<const Core::Utils::FunctionOfTime&(int)> function_of_time_by_id;
      const Teuchos::ParameterList* io_parameters;
      std::shared_ptr<Core::IO::OutputControl> output_control_file;
      const Teuchos::ParameterList* linear_solver_parameters;
      std::function<const Teuchos::ParameterList&(int)> solver_params_callback;
      Core::IO::Verbositylevel verbosity;
      std::function<void(std::shared_ptr<Core::Utils::ResultTest>)> add_field_test;
      std::function<void(MPI_Comm)> test_all;
    };

    ReducedLung1dPipeFlowContext make_reduced_lung_1d_pipe_flow_context_from_problem(
        Global::Problem& problem)
    {
      return ReducedLung1dPipeFlowContext{
          .artery_discretization = problem.get_dis("artery"),
          .reduced_lung_parameters = &problem.reduced_lung_parameters(),
          .function_of_time_by_id = [&problem](
                                        int function_id) -> const Core::Utils::FunctionOfTime&
          {
            return problem.function_manager().function_by_id<Core::Utils::FunctionOfTime>(
                function_id);
          },
          .io_parameters = &problem.io_params(),
          .output_control_file = problem.output_control_file(),
          .linear_solver_parameters = &problem.solver_params(1),
          .solver_params_callback = problem.solver_params_callback(),
          .verbosity =
              Teuchos::getIntegralValue<Core::IO::Verbositylevel>(problem.io_params(), "VERBOSITY"),
          .add_field_test = [&problem](std::shared_ptr<Core::Utils::ResultTest> result_test)
          { problem.add_field_test(result_test); },
          .test_all = [&problem](MPI_Comm comm) { problem.test_all(comm); },
      };
    }
  }  // namespace

  void fill_parameters(Parameters& parameters, Core::LinAlg::Vector<double>& solution,
      Core::LinAlg::Vector<double>& reference_area, Core::LinAlg::Vector<double>& thickness,
      Core::LinAlg::Vector<double>& Young, Core::LinAlg::Vector<double>& beta,
      Core::LinAlg::Vector<double>& radius, const Core::FE::Discretization& discretization)
  {
    // constants to be computed from the input parameters
    parameters.fluid.viscous_resistance_K_R =
        8.0 * M_PI * parameters.fluid.viscosity_mu / parameters.fluid.density_rho;

    // elemental vectors
    for (const auto& node : discretization.my_row_node_range())
    {
      const int node_gid = node.global_id();
      const int element_global_id = node.adjacent_elements()[0].global_id();

      const double Young_value = parameters.material.youngs_modulus_E.at(element_global_id);
      const double A0_value = parameters.geometry.reference_area_A0.at(element_global_id);
      const double r0_value = std::sqrt(A0_value / M_PI);
      const double th_value = parameters.geometry.thickness_th.at(element_global_id);
      const double beta_value =
          (sqrt(M_PI) * th_value * Young_value) /
          ((1 - std::pow(parameters.material.poisson_ratio_nu, 2)) * A0_value);

      Young.replace_global_value(node_gid, Young_value);
      reference_area.replace_global_value(node_gid, A0_value);
      radius.replace_global_value(node_gid, r0_value);
      thickness.replace_global_value(node_gid, th_value);
      beta.replace_global_value(node_gid, beta_value);

      // initial state
      solution.replace_global_value(2 * node_gid, A0_value);
      solution.replace_global_value(2 * node_gid + 1, 0.0);
    }
  }

  double compute_length(const Core::Elements::Element& element)
  {
    const auto& p1 = element.nodes()[0]->x();
    const auto& p2 = element.nodes()[1]->x();

    // Calculate the length of artery element
    const double sum_of_squares = (p1[0] - p2[0]) * (p1[0] - p2[0]) +
                                  (p1[1] - p2[1]) * (p1[1] - p2[1]) +
                                  (p1[2] - p2[2]) * (p1[2] - p2[2]);
    const double L = std::sqrt(sum_of_squares);
    return L;
  }

  void compute_psi_matrix(Core::LinAlg::Matrix<2, 4>& Psi_matrix,
      const Core::LinAlg::Matrix<2, 4>& N_matrix, const Core::LinAlg::Matrix<2, 4>& dNdxi_matrix,
      const Core::LinAlg::Matrix<2, 2>& flux_jacobian, const double L, const double delta)
  {
    const double inv_det = 2.0 / L;
    Psi_matrix = N_matrix;
    for (int i = 0; i < 2; i++)
    {
      for (int j = 0; j < 4; j++)
      {
        for (int k = 0; k < 2; k++)
        {
          {
            Psi_matrix(i, j) += delta * flux_jacobian(k, i) * dNdxi_matrix(k, j) * inv_det;
          }
        }
      }
    }
  }

  void conditions_from_newton_raphson(const Parameters& input, const double& Q_condition,
      const double& boundary_A0, const double& characteristic_W_outgoing, const double& beta,
      double& A_condition, double& u_condition)
  {
    const double rho_beta = input.fluid.density_rho / beta;
    const double residual_constant = rho_beta * rho_beta * 0.5 / 1024.0;

    auto residuum_and_jacobian = [&](double W_in) -> std::tuple<double, double>
    {
      double f = residual_constant * pow(W_in - characteristic_W_outgoing, 4) *
                     (W_in + characteristic_W_outgoing) -
                 Q_condition;
      double dfdW1 = rho_beta * rho_beta * (1.0 / 1024.0) *
                     pow(W_in - characteristic_W_outgoing, 3) *
                     (2.5 * W_in + 1.5 * characteristic_W_outgoing);
      return {f, dfdW1};
    };

    double W_in = Core::Utils::solve_local_newton(residuum_and_jacobian,
        2.0 * Q_condition / boundary_A0 - characteristic_W_outgoing, 1e-6, 100);

    A_condition = rho_beta * rho_beta * pow(W_in - characteristic_W_outgoing, 4) / 1024.0;
    u_condition = 0.5 * (W_in + characteristic_W_outgoing);
  }

  void compute_residual(Core::LinAlg::SerialDenseVector& f, const int N_connected_nodes,
      const Core::LinAlg::SerialDenseVector& x, const std::vector<double>& junction_normal,
      const std::vector<double>& junction_ref_area_A0,
      const std::vector<double>& junction_characteristic_out,
      const std::vector<double>& junction_beta, const double density_rho)
  {
    // define vectors with primary variables at junction
    std::vector<double> junction_velocity_u(N_connected_nodes);
    std::vector<double> junction_area_A(N_connected_nodes);
    for (int i = 0; i < N_connected_nodes; i++)
    {
      junction_velocity_u[i] = x(i);
      junction_area_A[i] = x(N_connected_nodes + i);
    }

    // initialize the residual
    f.putScalar(0.0);

    // fill the entities that have to do with forward characteristic speeds
    for (int i = 0; i < N_connected_nodes; i++)
    {
      f[i] = junction_velocity_u[i] +
             junction_normal[i] * 4.0 * pow(junction_area_A[i], 0.25) *
                 sqrt(0.5 * junction_beta[i] / density_rho) -
             junction_characteristic_out[i];
    }

    // fill the entities that have to do with the mass conservation
    f[N_connected_nodes] = 0.0;
    for (int i = 0; i < N_connected_nodes; i++)
    {
      f[N_connected_nodes] += junction_normal[i] * junction_area_A[i] * junction_velocity_u[i];
    }

    // fill the entities that have to do with the pressure conservation
    // reference pressure
    const double P0 = 0.5 * density_rho * pow(junction_velocity_u[0], 2) +
                      junction_beta[0] * (sqrt(junction_area_A[0]) - sqrt(junction_ref_area_A0[0]));
    for (int i = 1; i < N_connected_nodes; i++)
    {
      f[N_connected_nodes + i] =
          P0 - (0.5 * density_rho * pow(junction_velocity_u[i], 2) +
                   junction_beta[i] * (sqrt(junction_area_A[i]) - sqrt(junction_ref_area_A0[i])));
    }
  }

  void compute_jacobian(Core::LinAlg::SerialDenseMatrix& jacobian, const int N_connected_nodes,
      const Core::LinAlg::SerialDenseVector& x, const std::vector<double>& junction_normal,
      const std::vector<double>& junction_beta, const double density_rho)
  {
    std::vector<double> junction_velocity_u(N_connected_nodes);
    std::vector<double> junction_area_A(N_connected_nodes);
    for (int i = 0; i < N_connected_nodes; i++)
    {
      junction_velocity_u[i] = x(i);
      junction_area_A[i] = x(N_connected_nodes + i);
    }
    jacobian.putScalar(0.0);

    // fill the entities that have to do with forward characteristic speeds
    for (int i = 0; i < N_connected_nodes; i++)
    {
      jacobian(i, i) = 1.0;
      jacobian(i, i + N_connected_nodes) = junction_normal[i] * pow(junction_area_A[i], -0.75) *
                                           sqrt(0.5 * junction_beta[i] / density_rho);
    }

    // fill the entities that have to do with the mass conservation
    for (int i = 0; i < N_connected_nodes; i++)
    {
      jacobian(N_connected_nodes, i) = junction_normal[i] * junction_area_A[i];
      jacobian(N_connected_nodes, N_connected_nodes + i) =
          junction_normal[i] * junction_velocity_u[i];
    }

    // fill the entities that have to do with the pressure conservation
    // reference pressure of first node in junction
    const double P_u = density_rho * junction_velocity_u[0];
    const double P_A = 0.5 * junction_beta[0] / (sqrt(junction_area_A[0]));
    // pressure conservation
    for (int i = 1; i < N_connected_nodes; i++)
    {
      jacobian(N_connected_nodes + i, 0) = P_u;
      jacobian(N_connected_nodes + i, i) = -density_rho * junction_velocity_u[i];
      jacobian(N_connected_nodes + i, N_connected_nodes) = P_A;
      jacobian(N_connected_nodes + i, N_connected_nodes + i) =
          -0.5 * junction_beta[i] / (sqrt(junction_area_A[i]));
    }
  }

  void get_conditions_at_junctions(const double density_rho,
      const Core::LinAlg::Vector<double>& characteristics_for_junction,
      Core::LinAlg::Vector<double>& solution_for_junction,
      const Core::LinAlg::Vector<double>& normals_for_junction,
      const Core::LinAlg::Vector<double>& area0_for_junction,
      const Core::LinAlg::Vector<double>& beta_for_junction,
      const std::vector<JunctionInfo>& all_junctions, Core::LinAlg::Vector<double>& dof_update)
  {
    for (const auto& junction : all_junctions)
    {
      // Get number of connected nodes
      const int N_connected_nodes = static_cast<int>(junction.node_ids.size());
      // create vectors for node data
      // the junction conditions are computed on every rank that participates in the junction
      std::vector<double> junction_area_A(N_connected_nodes);
      std::vector<double> junction_velocity_u(N_connected_nodes);
      std::vector<double> junction_normal(N_connected_nodes);
      std::vector<double> junction_beta(N_connected_nodes);
      std::vector<double> junction_characteristic_out(N_connected_nodes);
      std::vector<double> junction_ref_area_A0(N_connected_nodes);

      for (int i = 0; i < N_connected_nodes; ++i)
      {
        int global_node_id = junction.node_ids[i];

        // get local ID for solution
        const int local_A_id = solution_for_junction.get_map().lid(global_node_id * 2);
        const int local_u_id = solution_for_junction.get_map().lid(global_node_id * 2 + 1);
        const int local_node_id = characteristics_for_junction.get_map().lid(global_node_id);

        // get data from node
        junction_area_A[i] = solution_for_junction.get_values()[local_A_id];
        junction_velocity_u[i] = solution_for_junction.get_values()[local_u_id];
        junction_characteristic_out[i] = characteristics_for_junction.get_values()[local_node_id];
        junction_normal[i] = normals_for_junction.get_values()[local_node_id];

        FOUR_C_ASSERT(junction_normal[i] == -1.0 || junction_normal[i] == 1.0,
            "Junction normals need to be in/ outlet");

        junction_ref_area_A0[i] = area0_for_junction.get_values()[local_node_id];
        junction_beta[i] = beta_for_junction.get_values()[local_node_id];
      }

      /**************************************************************/
      // Newton-Raphson-method

      // first guess from old values
      // x = [u_0 u_1 ... u_N A_0 A_1 ... A_N]^T
      Core::LinAlg::SerialDenseVector x(2 * N_connected_nodes, true);
      for (int i = 0; i < N_connected_nodes; ++i)
      {
        x(i) = junction_velocity_u[i];
        x(N_connected_nodes + i) = junction_area_A[i];
      }

      auto residuum_and_jacobian_evaluator = [&](Core::LinAlg::SerialDenseVector& x_eval)
          -> std::tuple<Core::LinAlg::SerialDenseVector, Core::LinAlg::SerialDenseMatrix>
      {
        Core::LinAlg::SerialDenseVector f(2 * N_connected_nodes, true);
        compute_residual(f, N_connected_nodes, x_eval, junction_normal, junction_ref_area_A0,
            junction_characteristic_out, junction_beta, density_rho);

        Core::LinAlg::SerialDenseMatrix Jacobian(
            2 * N_connected_nodes, 2 * N_connected_nodes, true);
        compute_jacobian(
            Jacobian, N_connected_nodes, x_eval, junction_normal, junction_beta, density_rho);

        return {f, Jacobian};
      };

      x = Core::Utils::solve_local_newton(residuum_and_jacobian_evaluator, x, 1e-8, 50);

      for (int i = 0; i < N_connected_nodes; ++i)
      {
        // set A_np and u_np for application of fluxes at junctions
        dof_update.replace_global_value(2 * junction.node_ids[i], x(i + N_connected_nodes));
        dof_update.replace_global_value(2 * junction.node_ids[i] + 1, x(i));
      }
    }
  }

  void update_rhs_with_junction_properties(const double density_rho,
      Core::LinAlg::Vector<double>& rhs_junction,
      const Core::LinAlg::Vector<double>& solution_update_junction,
      const Core::LinAlg::Vector<double>& beta, const Core::LinAlg::Vector<double>& reference_area,
      const Core::LinAlg::Vector<double>& normals, const Core::LinAlg::Vector<double>& solution,
      const std::vector<JunctionInfo>& all_junctions)
  {
    for (const auto& [node_ids, node_owners] : all_junctions)
    {
      for (const auto global_node_id : node_ids)
      {
        // node found on this rank
        if (const int local_node_id = normals.get_map().lid(global_node_id); local_node_id != -1)
        {
          const int A_id = 2 * local_node_id;
          const int junction_dof_id_A = solution_update_junction.get_map().lid(2 * global_node_id);
          const int junction_dof_id_u =
              solution_update_junction.get_map().lid(2 * global_node_id + 1);
          const double A_condition = solution_update_junction.get_values()[junction_dof_id_A];
          const double u_condition = solution_update_junction.get_values()[junction_dof_id_u];

          const double A_n = solution.get_values()[A_id];
          const double u_n = solution.get_values()[A_id + 1];
          const double beta_node = beta.get_values()[local_node_id];
          const double A0_node = reference_area.get_values()[local_node_id];
          const double normal_in_out = normals.get_values()[local_node_id];
          // derive flux over boundaries
          const double F1 = A_condition * u_condition;
          const double F2 = 0.5 * pow(u_condition, 2) +
                            (beta_node * (sqrt(A_condition) - sqrt(A0_node))) / density_rho;
          const double F1_h = A_n * u_n;
          const double F2_h =
              0.5 * pow(u_n, 2) + (beta_node * (sqrt(A_n) - sqrt(A0_node))) / density_rho;

          // set values in rhs_junction
          rhs_junction.replace_local_value(2 * local_node_id, normal_in_out * (F1_h - F1));
          rhs_junction.replace_local_value(2 * local_node_id + 1, normal_in_out * (F2_h - F2));
        }
      }
    }
  }

  void run_reduced_lung_1d_pipe_flow(const ReducedLung1dPipeFlowContext& context)
  {
    FOUR_C_ASSERT_ALWAYS(
        context.artery_discretization != nullptr, "Artery discretization is not initialized.");
    FOUR_C_ASSERT_ALWAYS(
        context.reduced_lung_parameters != nullptr, "Reduced lung parameters are not initialized.");
    FOUR_C_ASSERT_ALWAYS(
        context.function_of_time_by_id, "Function lookup callback is not initialized.");
    FOUR_C_ASSERT_ALWAYS(context.io_parameters != nullptr, "I/O parameters are not initialized.");
    FOUR_C_ASSERT_ALWAYS(
        context.output_control_file != nullptr, "Output control file is not initialized.");
    FOUR_C_ASSERT_ALWAYS(context.linear_solver_parameters != nullptr,
        "Linear solver parameters are not initialized.");
    FOUR_C_ASSERT_ALWAYS(
        context.solver_params_callback, "Solver parameter callback is not initialized.");
    FOUR_C_ASSERT_ALWAYS(context.add_field_test, "Result test callback is not initialized.");
    FOUR_C_ASSERT_ALWAYS(context.test_all, "Result test runner callback is not initialized.");

    std::shared_ptr<Core::FE::Discretization> discretization = context.artery_discretization;
    if (!discretization->filled() || !discretization->have_dofs())
    {
      discretization->fill_complete({.assign_degrees_of_freedom = true,
          .init_elements = true,
          .do_boundary_conditions = true});
    }

    // parameters for time and blood from input file
    Parameters input = context.reduced_lung_parameters->get<Parameters>("general");
    input.boundary_conditions.bc_fct =
        &context.function_of_time_by_id(input.boundary_conditions.function_id_inflow);

    // vectors with nodal properties (changing over time)
    Core::LinAlg::Vector<double> solution(*discretization->dof_row_map());
    Core::LinAlg::Vector<double> pressure_solution(*discretization->node_row_map());
    Core::LinAlg::Vector<double> flow_solution(*discretization->node_row_map());
    Core::LinAlg::Vector<double> radius_solution(*discretization->node_row_map());
    Core::LinAlg::Vector<double> solution_for_evaluation(*discretization->dof_col_map());

    // Vectors with elemental geometry and material parameters on nodal vectors
    Core::LinAlg::Vector<double> reference_area_0(*discretization->node_row_map());
    Core::LinAlg::Vector<double> reference_area_0_evaluation(*discretization->node_col_map());
    Core::LinAlg::Vector<double> thickness_th(*discretization->node_row_map());
    Core::LinAlg::Vector<double> thickness_evaluation(*discretization->node_col_map());
    Core::LinAlg::Vector<double> Young(*discretization->node_row_map());
    Core::LinAlg::Vector<double> Young_evaluation(*discretization->node_col_map());
    Core::LinAlg::Vector<double> beta(*discretization->node_row_map());
    Core::LinAlg::Vector<double> beta_evaluation(*discretization->node_col_map());

    fill_parameters(input, solution, reference_area_0, thickness_th, Young, beta, radius_solution,
        *discretization);

    //  export to column vectors for parallel evaluation
    FourC::Core::LinAlg::export_to(solution, solution_for_evaluation);
    FourC::Core::LinAlg::export_to(reference_area_0, reference_area_0_evaluation);
    FourC::Core::LinAlg::export_to(thickness_th, thickness_evaluation);
    FourC::Core::LinAlg::export_to(Young, Young_evaluation);
    FourC::Core::LinAlg::export_to(beta, beta_evaluation);

    pressure_solution.put_scalar(0.0);
    flow_solution.put_scalar(0.0);

    auto mass_matrix =
        std::make_shared<Core::LinAlg::SparseMatrix>(*discretization->dof_row_map(), 4);
    auto rhs = std::make_shared<Core::LinAlg::Vector<double>>(*discretization->dof_row_map());
    auto rhs_junction =
        std::make_shared<Core::LinAlg::Vector<double>>(*discretization->dof_row_map());

    // Normals definition: Iteration over all elements and summing -1 for the first node, and +1
    // for the second node of the element -> Domain nodes sum up to 0, inflow nodes to -1 ,and
    // outflow nodes to +1
    Core::LinAlg::Vector<double> normals(*discretization->node_row_map());
    Core::LinAlg::Vector<double> normals_evaluation(*discretization->node_col_map());

    for (const auto& element : discretization->my_col_element_range())
    {
      auto global_in = element.nodes()[0].global_id();
      auto global_out = element.nodes()[1].global_id();
      normals_evaluation.sum_into_global_value(global_in, -1);
      normals_evaluation.sum_into_global_value(global_out, 1);
    }

    Core::LinAlg::export_to(normals_evaluation, normals);
    Core::LinAlg::export_to(normals, normals_evaluation);

    // Get local rank
    int mpi_rank = Core::Communication::my_mpi_rank(discretization->get_comm());
    auto* comm = discretization->get_comm();

    /**************************
     *Geometry check: go through geometry and check for junctions: same coordinates of nodes
     ************************/
    std::vector<NodeInfo> local_in_out_nodes;
    local_in_out_nodes.reserve(discretization->num_my_row_nodes());
    for (const auto& node : discretization->my_row_node_range())
    {
      // only add them to nodes if they are an inner or outer boundary node
      if (normals.get_values()[discretization->node_row_map()->lid(node.global_id())] != 0.0)
      {
        local_in_out_nodes.push_back(NodeInfo{
            .id = node.global_id(),
            .owner = mpi_rank,
            .x = node.x()[0],
            .y = node.x()[1],
            .z = node.x()[2],
        });
      }
    }

    auto all_in_out_nodes = Core::Communication::all_reduce(
        local_in_out_nodes, comm);  // contains all nodes with normal_in_out = +/-1
    std::unordered_set<int> visited_nodes;
    std::vector<JunctionInfo> all_junctions;
    std::vector<int> boundary_nodes;
    std::vector<ReducedLung1DPipe::TerminalUnit::TerminalUnitModel> all_terminal_units;
    std::unordered_map<int, std::size_t> global_tu_id_to_index;

    for (auto it = all_in_out_nodes.begin(); it != all_in_out_nodes.end(); ++it)
    {
      const auto& [id1, owner1, x1, y1, z1] = *it;
      if (visited_nodes.contains(id1)) continue;

      JunctionInfo junction_info;
      junction_info.node_ids.push_back(id1);
      junction_info.node_owners.push_back(owner1);

      visited_nodes.insert(id1);
      for (auto it_other = it + 1; it_other != all_in_out_nodes.end(); ++it_other)
      {
        const auto& [id2, owner2, x2, y2, z2] = *it_other;
        // junctions
        if (constexpr double tol = 1e-10;
            std::abs(x1 - x2) < tol && std::abs(y1 - y2) < tol && std::abs(z1 - z2) < tol)
        {
          visited_nodes.insert(id2);
          junction_info.node_ids.push_back(id2);
          junction_info.node_owners.push_back(owner2);
        }
      }

      if (junction_info.node_ids.size() > 1)
      {
        bool does_this_rank_participate = std::ranges::find(junction_info.node_owners, mpi_rank) !=
                                          junction_info.node_owners.end();
        if (does_this_rank_participate) all_junctions.emplace_back(junction_info);
      }
      // treat terminal units and create TU model per boundary node if node has information on
      // terminal unit
      else if (junction_info.node_ids.size() == 1 &&
               input.boundary_conditions.output == "terminal_unit" &&
               normals_evaluation.get_map().lid(junction_info.node_ids.front()) != -1 &&
               normals_evaluation.get_values()[normals_evaluation.get_map().lid(
                   junction_info.node_ids.front())] == 1)
      {
        // Create terminal units
        ReducedLung1DPipe::TerminalUnit::TerminalUnitModel tu_model =
            ReducedLung1DPipe::TerminalUnit::create_terminal_unit_model(
                input, junction_info.node_ids.front(), junction_info.node_owners.front());

        // create map to access terminal units
        global_tu_id_to_index[tu_model.data.global_node_id] = all_terminal_units.size();
        // add to vector
        all_terminal_units.push_back(tu_model);
      }
    }
    // Now we have a list of all_junctions that are relevant on a rank. This means every rank know
    // which nodes/dof are required to evaluate the junction.
    std::vector<int> locally_relevant_dof_for_junctions;
    std::vector<int> locally_relevant_nodes_for_junctions;
    std::vector<int> owned_nodes_at_junctions;

    for (const auto& [node_ids, node_owners] : all_junctions)
    {
      // Remap nodes to dof
      for (std::size_t i = 0; i < node_ids.size(); ++i)
      {
        int node_id = node_ids[i];
        // Locally relevant for junction
        // A
        locally_relevant_dof_for_junctions.push_back(node_id * 2);
        // u
        locally_relevant_dof_for_junctions.push_back(node_id * 2 + 1);
        // nodes and elements
        locally_relevant_nodes_for_junctions.push_back(node_id);
        // owned nodes
        if (node_owners[i] == mpi_rank)
        {
          owned_nodes_at_junctions.push_back(node_id);
        }
      }
    }

    // Define maps for junction mapping
    Core::LinAlg::Map locally_relevant_junction_dof_map(-1,
        static_cast<int>(locally_relevant_dof_for_junctions.size()),
        locally_relevant_dof_for_junctions.data(), 0, comm);
    Core::LinAlg::Map locally_relevant_junction_node_map(-1,
        static_cast<int>(locally_relevant_nodes_for_junctions.size()),
        locally_relevant_nodes_for_junctions.data(), 0, comm);
    Core::LinAlg::Map owned_junction_node_map(-1, static_cast<int>(owned_nodes_at_junctions.size()),
        owned_nodes_at_junctions.data(), 0, comm);

    Core::LinAlg::Vector<double> solution_for_junction(locally_relevant_junction_dof_map);
    Core::LinAlg::export_to(solution, solution_for_junction);

    Core::LinAlg::Vector<double> area0_for_junctions(locally_relevant_junction_node_map);
    Core::LinAlg::export_to(reference_area_0, area0_for_junctions);

    Core::LinAlg::Vector<double> beta_for_junctions(locally_relevant_junction_node_map);
    Core::LinAlg::export_to(beta, beta_for_junctions);

    Core::LinAlg::Vector<double> characteristics_for_junctions(locally_relevant_junction_node_map);
    Core::LinAlg::Vector<double> owned_characteristics_at_junction(owned_junction_node_map);

    Core::LinAlg::Vector<double> normals_for_junctions(locally_relevant_junction_node_map);
    Core::LinAlg::export_to(normals, normals_for_junctions);

    Core::LinAlg::Vector<double> updated_solutions_on_junctions(locally_relevant_junction_dof_map);

    /************/

    double time_n = 0.0;
    double dt = input.final_time / input.n_steps;
    double min_time_step_size = 1.0;

    Core::FE::AssembleStrategy strategy(0, 0, mass_matrix, nullptr, rhs, nullptr, nullptr);
    Core::FE::AssembleStrategy strategy_junction(
        0, 0, nullptr, nullptr, rhs_junction, nullptr, nullptr);

    const auto compute_local_contributions =
        [&](Core::Elements::Element& element, Core::Elements::LocationArray& la,
            Core::LinAlg::SerialDenseMatrix& element_matrix, Core::LinAlg::SerialDenseMatrix&,
            Core::LinAlg::SerialDenseVector& element_rhs, Core::LinAlg::SerialDenseVector&,
            Core::LinAlg::SerialDenseVector&) -> void
    {
      // Values on element (constant over element)
      int in_id = reference_area_0_evaluation.get_map().lid(element.node_ids()[0]);
      const double beta_element = beta_evaluation.get_values()[in_id];
      const double Young_element = Young_evaluation.get_values()[in_id];
      const double thickness_element = thickness_evaluation.get_values()[in_id];
      const double reference_area_element = reference_area_0_evaluation.get_values()[in_id];

      Core::LinAlg::Matrix<2, 1> Pext_node;
      Pext_node(0, 0) = 0;
      Pext_node(1, 0) = 0;

      // check size of element_matrix
      FOUR_C_ASSERT(element_matrix.num_cols() == 4, "Internal error.");
      FOUR_C_ASSERT(element_matrix.num_rows() == 4, "Internal error.");
      Core::FE::GaussIntegration gauss_integration(element.shape());
      // get solution at time n for local element
      std::vector<double> local_solution =
          Core::FE::extract_values(solution_for_evaluation, la[0].lm_);

      // check if element is part of a boundary or junction
      std::array<bool, 2> is_junction = {false, false};
      std::array<bool, 2> is_boundary = {false, false};
      std::array<int, 2> normal_in_out = {0, 0};

      for (int inode = 0; inode < element.num_node(); ++inode)
      {
        int global_node_id = element.nodes()[inode]->id();
        // Check that node is on rank
        if (auto normal_id = normals_evaluation.get_map().lid(global_node_id); normal_id != -1)
        {
          normal_in_out[inode] = static_cast<int>(normals_evaluation.get_values()[normal_id]);

          FOUR_C_ASSERT(
              normal_in_out[inode] == 0 || normal_in_out[inode] == 1 || normal_in_out[inode] == -1,
              "Normals should be 0, 1 or -1");
        }

        // check if junction or outer boundary

        // set property if node is junction, capillary or boundary
        if (normals_for_junctions.get_map().lid(global_node_id) != -1)
        {
          is_junction[inode] = true;
        }
        else if (normal_in_out[inode] == -1 || normal_in_out[inode] == 1)
        {
          is_boundary[inode] = true;
        }
        FOUR_C_ASSERT(is_junction[inode] + is_boundary[inode] == 0 ||
                          is_junction[inode] + is_boundary[inode] == 1,
            "Node can only be part of boundary OR junction. Or be part of the "
            "domain.");
      }


      // Computation of domain
      for (int gp = 0; gp < gauss_integration.num_points(); ++gp)
      {
        const double* xi = gauss_integration.point(gp);
        double weight = gauss_integration.weight(gp);

        // get dNdxi and N at xi
        Core::LinAlg::Matrix<1, 2> dN_dxi_1dof;
        shape_function_1d_deriv1(dN_dxi_1dof, *xi, element.shape());
        Core::LinAlg::Matrix<1, 2> N_1dof;
        shape_function_1d(N_1dof, *xi, element.shape());

        // derivatives on elements
        constexpr double dA0_dxi_gp =
            0;  // since A0 is assumed as elemental value, no spatial derivative
        constexpr double dYoung_dxi_gp =
            0;  // since Young is assumed as elemental value, no spatial derivative

        // Defining the shape function matrices
        Core::LinAlg::Matrix<2, 2 * 2> N_matrix;
        //  Defining the matrix : derivative of shape functions
        Core::LinAlg::Matrix<2, 2 * 2> dNdxi_matrix;
        // Define parameters based on gp, fill N and dNdxi
        N_matrix.clear();
        dNdxi_matrix.clear();
        // define shape function matrices
        for (int i = 0; i < 2; i++)
        {
          dNdxi_matrix(0, 2 * i) = dN_dxi_1dof(0, i);
          dNdxi_matrix(1, 2 * i + 1) = dN_dxi_1dof(0, i);

          N_matrix(0, 2 * i) = N_1dof(0, i);
          N_matrix(1, 2 * i + 1) = N_1dof(0, i);
        }

        // Calculate the length of artery element
        const double L = compute_length(element);
        FOUR_C_ASSERT(L != 0, "Length of element is zero on element {}", element.id());
        // Calculate further helpers
        const double det = L * 0.5;
        const double inv_det = 1.0 / det;

        // U = [A u]^T
        Core::LinAlg::Matrix<2, 1> U;
        U.put_scalar(0.0);
        for (int i = 0; i < 2; i++)
        {
          for (int j = 0; j < 4; j++)
          {
            U(i, 0) += N_matrix(i, j) * local_solution[j];
          }
        }

        double area_A = U(0);
        FOUR_C_ASSERT(area_A > 0, "area < 0");
        double velocity_u = U(1);

        // flux dF/dU
        Core::LinAlg::Matrix<2, 2> H;
        H(0, 0) = velocity_u;
        H(0, 1) = area_A;
        H(1, 0) = 0.5 * beta_element / (input.fluid.density_rho * sqrt(area_A));
        H(1, 1) = velocity_u;

        // Define helpers for test functions Psi
        // characteristic speed lambda = |u +/- c|
        double lambda_max = std::max(std::abs(velocity_u + sqrt(0.5 * beta_element * sqrt(area_A) /
                                                                (input.fluid.density_rho))),
            std::abs(
                velocity_u - sqrt(0.5 * beta_element * sqrt(area_A) / (input.fluid.density_rho))));
        // compute min timestep size for CFL condition
        if (L / lambda_max < min_time_step_size)
        {
          min_time_step_size = L / lambda_max;
        }

        double delta = std::abs(0.5 * L / lambda_max);
        // Define the test functions according to Petrov Galerkin Psi = N + delta * H^T * dNdxi
        // * dxidx
        Core::LinAlg::Matrix<2, 2 * 2> Psi_matrix;
        compute_psi_matrix(Psi_matrix, N_matrix, dNdxi_matrix, H, L, delta);

        //--------------------------------------------------------------------------------------
        // compute element_matrix
        for (int i = 0; i < 4; ++i)
        {
          for (int j = 0; j < 4; ++j)
          {
            for (int k = 0; k < 2; ++k)
            {
              element_matrix(i, j) += weight * det * Psi_matrix(k, i) * N_matrix(k, j);
            }
          }
        }

        //----------------------------------------------------------------------------------
        // Compute right hand side
        //(Psi^T, F(U)) - (Psi^T, H * dU/dx)

        //  Psi * H * Nxi * local_solution
        for (int i = 0; i < 4; ++i)
        {
          for (int j = 0; j < 2; ++j)
          {
            for (int k = 0; k < 2; ++k)
            {
              for (int l = 0; l < 4; ++l)
              {
                element_rhs(i) += (-1.0) *
                                  (Psi_matrix(j, i) * H(j, k) * dNdxi_matrix(k, l) * inv_det *
                                      local_solution[l]) *
                                  weight * det;
              }
            }
          }
        }

        // source term:F(U) = [0, f]
        // f = K_R * u - dp_dbeta dbeta_dx + dp_dA0 * dA0_dx
        const double dp_dbeta = sqrt(area_A) - sqrt(reference_area_element);
        //(A_0 (h_0 * dE/dx + dh_0/dx * E) - h_0 * E * dA_0/dx ) / A_0^2
        // assumption: thickness = const
        const double dbeta_dx =
            (sqrt(M_PI) * ((thickness_element * dYoung_dxi_gp * inv_det) * reference_area_element) -
                thickness_element * Young_element * dA0_dxi_gp * inv_det) /
            (pow(reference_area_element * (1 - pow(input.material.poisson_ratio_nu, 2)), 2));
        const double dp_dA0 = ((-0.5) * beta_element) / sqrt(reference_area_element);

        Core::LinAlg::Matrix<2, 1> S;
        S.put_scalar(0.0);
        S(1, 0) +=
            (-1 / input.fluid.density_rho) *
            (input.fluid.viscous_resistance_K_R * input.fluid.density_rho * velocity_u / area_A +
                dp_dA0 * dA0_dxi_gp * inv_det + dp_dbeta * dbeta_dx);

        // Apply source term
        for (int i = 0; i < 4; ++i)
        {
          for (int j = 0; j < 2; ++j)
          {
            element_rhs(i) += Psi_matrix(j, i) * S(j, 0) * det * weight;
          }
        }
      }

      //-------------------------------------------------------------------------------------------
      // BOUNDARY CONDITIONS
      //----------------------------------------------------------------------------------------
      // iterate over both nodes in element
      for (int boundary_local_index = 0; boundary_local_index < element.num_node();
          ++boundary_local_index)
      {
        if (is_boundary[boundary_local_index] || is_junction[boundary_local_index])
        {
          double xi = normal_in_out[boundary_local_index];
          Core::LinAlg::Matrix<2, 2 * 2> N_matrix;
          //  Defining the matrix : derivative of shape functions
          Core::LinAlg::Matrix<2, 2 * 2> dNdxi_matrix;
          // get dNdxi and N at xi
          Core::LinAlg::Matrix<1, 2> dN_dxi_1dof;
          shape_function_1d_deriv1(dN_dxi_1dof, xi, element.shape());
          Core::LinAlg::Matrix<1, 2> N_1dof;
          shape_function_1d(N_1dof, xi, element.shape());

          N_matrix.clear();
          dNdxi_matrix.clear();
          // define shape function matrices
          for (int i = 0; i < 2; i++)
          {
            dNdxi_matrix(0, 2 * i) = dN_dxi_1dof(0, i);
            dNdxi_matrix(1, 2 * i + 1) = dN_dxi_1dof(0, i);

            N_matrix(0, 2 * i) = N_1dof(0, i);
            N_matrix(1, 2 * i + 1) = N_1dof(0, i);
          }

          // get values at boundary node
          double boundary_Pext = Pext_node(boundary_local_index, 0);
          // get values at other node
          int inner_local_index = 1 - boundary_local_index;
          int boundary_A_index = 2 * boundary_local_index;
          int inner_A_index = 2 * inner_local_index;

          // read in current boundary values from solution vector
          double boundary_A = local_solution[boundary_A_index];
          double boundary_u = local_solution[boundary_A_index + 1];

          // flux dF/dU
          Core::LinAlg::Matrix<2, 2> H;
          H(0, 0) = boundary_u;
          H(0, 1) = boundary_A;
          H(1, 0) = 0.5 * beta_element / (input.fluid.density_rho * sqrt(boundary_A));
          H(1, 1) = boundary_u;

          // Define the test functions according to Petrov Galerkin Psi = N + delta * H^T *
          // dNdxi
          // * dxidx
          double lambda_max = std::max(
              std::abs(boundary_u +
                       sqrt(0.5 * beta_element * sqrt(boundary_A) / (input.fluid.density_rho))),
              std::abs(boundary_u -
                       sqrt(0.5 * beta_element * sqrt(boundary_A) / (input.fluid.density_rho))));
          double L = compute_length(element);
          double delta = std::abs(0.5 * L / lambda_max);
          Core::LinAlg::Matrix<2, 2 * 2> Psi_matrix;
          compute_psi_matrix(Psi_matrix, N_matrix, dNdxi_matrix, H, L, delta);
          //++++++++++++++++++

          // read in current element values from solution vector
          double inner_A = local_solution[inner_A_index];
          double inner_u = local_solution[inner_A_index + 1];

          // sound speed at boundary node
          double sound_speed_c_boundary =
              sqrt(0.5 * beta_element * (sqrt(boundary_A) / input.fluid.density_rho));

          double lambda_out =
              boundary_u + (normal_in_out[boundary_local_index]) * sound_speed_c_boundary;
          // lambda_1 > 0, lambda_2 < 0
          if (boundary_u > sound_speed_c_boundary)
          {
            FOUR_C_ASSERT(
                false, "Flow not subcritical in element {} at time {}", element.id(), time_n);
          }

          FOUR_C_ASSERT(normal_in_out[boundary_local_index] * lambda_out * dt < L,
              "Characteristic computation out of element {}.", element.id());

          // extrapolation: computation of outgoing characteristic extrapolated from last time
          // step (x = 0 + lambda_2 * dt)
          double N1 = 1 - normal_in_out[boundary_local_index] * (dt * lambda_out / L);
          double N2 = normal_in_out[boundary_local_index] * dt * lambda_out / L;

          // extrapolation of variables at x = lambda * dt
          double A_l = N1 * boundary_A + N2 * inner_A;
          double u_l = N1 * boundary_u + N2 * inner_u;
          double c_l = sqrt(beta_element * sqrt(A_l) / (2 * input.fluid.density_rho));

          // defining outgoing characteristic at dt*lambda
          double characteristic_W_outgoing = u_l + normal_in_out[boundary_local_index] * 4 * c_l;
          // characteristics in reference state
          double c_0 =
              sqrt(beta_element * sqrt(reference_area_element) / (2 * input.fluid.density_rho));

          // if junction, update outgoing characteristics
          if (is_junction[boundary_local_index])
          {
            // update characteristics for later junction computation
            if (int global_id = element.node_ids()[boundary_local_index];
                owned_characteristics_at_junction.get_map().lid(global_id) >= 0)
            {
              owned_characteristics_at_junction.replace_local_value(
                  owned_characteristics_at_junction.get_map().lid(global_id),
                  characteristic_W_outgoing);
            }
          }

          /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
          // when outer boundary, prescribed parameter and reflection boundary condition
          /+********************************************************************************/
          else
          {
            double u_condition = boundary_u;
            double A_condition = boundary_A;

            // precribed inflow
            double heavyside = 1.0;
            double time_cyc = time_n;
            if (auto period = input.boundary_conditions.cycle_period)
            {
              while (time_cyc > *period)
              {
                time_cyc -= *period;
              }
            }

            if (auto pulse = input.boundary_conditions.pulse_width)
            {
              if (time_cyc > *pulse)
              {
                heavyside = 0.0;
              }
            }

            // inlet
            if (normal_in_out[boundary_local_index] == -1)
            {
              //  modify outgoing characteristic if source term is used
              characteristic_W_outgoing +=
                  dt * (-input.fluid.viscous_resistance_K_R * boundary_u / boundary_A);

              // prescribed u
              if (input.boundary_conditions.input == "velocity")
              {
                u_condition = input.boundary_conditions.bc_fct->evaluate(time_n, 0) * heavyside;
                // A derived from characteristics and prescribed u
                A_condition = (pow(u_condition - characteristic_W_outgoing, 4) / 64) *
                              pow(input.fluid.density_rho / beta_element, 2);
              }
              else if (input.boundary_conditions.input == "area")
              {
                A_condition = input.boundary_conditions.bc_fct->evaluate(time_n, 0) * heavyside;
                u_condition = characteristic_W_outgoing +
                              4 * std::pow(A_condition, 0.25) *
                                  sqrt(0.5 * beta_element / input.fluid.density_rho);
              }
              else if (input.boundary_conditions.input == "pressure")
              {
                // prescribed p from function
                double pressure_fct =
                    input.boundary_conditions.bc_fct->evaluate(time_n, 0) * heavyside;
                A_condition = pow(
                    (pressure_fct - boundary_Pext) / beta_element + sqrt(reference_area_element),
                    2);
                u_condition = characteristic_W_outgoing +
                              4 * std::pow(A_condition, 0.25) *
                                  sqrt(0.5 * beta_element / input.fluid.density_rho);
              }
              else if (input.boundary_conditions.input == "flow")
              {
                // Newton-Raphson iteration to get conditions for A and u
                // Q = A(W1, W2) * u(W1,W2)
                double flow_fct = input.boundary_conditions.bc_fct->evaluate(time_n, 0) * heavyside;

                conditions_from_newton_raphson(input, flow_fct, reference_area_element,
                    characteristic_W_outgoing, beta_element, A_condition, u_condition);
              }
              else
              {
                FOUR_C_ASSERT(false, "no boundary condition provided");
              }
            }
            // outlet
            else
            {
              if (input.boundary_conditions.output == "reflection")
              {
                double reflection_factor = input.boundary_conditions.condition_outflow;
                double characteristic_W_incoming =
                    -4 * c_0 - reflection_factor * (characteristic_W_outgoing - 4 * c_0);

                FOUR_C_ASSERT(reflection_factor >= -1 && reflection_factor <= 1,
                    "Error in reflection factor computation.");
                A_condition =
                    pow((characteristic_W_outgoing - characteristic_W_incoming) * 0.25, 4) *
                    pow(input.fluid.density_rho * 0.5 / beta_element, 2);
                u_condition = (characteristic_W_outgoing + characteristic_W_incoming) * 0.5;
              }
              else if (input.boundary_conditions.output == "pressure")
              {
                // prescribed p at outlet
                A_condition = pow(
                    (input.boundary_conditions.condition_outflow - boundary_Pext) / beta_element +
                        sqrt(reference_area_element),
                    2);
                u_condition = characteristic_W_outgoing -
                              4 * std::pow(A_condition, 0.25) *
                                  sqrt(0.5 * beta_element / input.fluid.density_rho);
              }
              else if (input.boundary_conditions.output == "terminal_unit")
              {
                // access TerminalUnitModel at output node
                auto it = global_tu_id_to_index.find(element.node_ids()[boundary_local_index]);
                if (it == global_tu_id_to_index.end())
                {
                  FOUR_C_ASSERT(false, "Wrong Terminal Unit mapping.");
                  return;
                }
                ReducedLung1DPipe::TerminalUnit::TerminalUnitModel& terminal_unit =
                    all_terminal_units[it->second];

                auto residuum_and_jacobian = [&](double A_eval) -> std::tuple<double, double>
                {
                  return terminal_unit.evaluate_residual_jacobian(A_eval, reference_area_element,
                      beta_element, boundary_Pext, input.fluid.density_rho,
                      characteristic_W_outgoing, dt);
                };

                A_condition =
                    Core::Utils::solve_local_newton(residuum_and_jacobian, A_condition, 1e-8, 100);

                u_condition = characteristic_W_outgoing -
                              4 * std::pow(A_condition, 0.25) *
                                  sqrt(0.5 * beta_element / input.fluid.density_rho);
                double Q_condition = A_condition * u_condition;
                // update model data
                terminal_unit.update_terminal_unit_data(Q_condition, dt, A_condition, beta_element,
                    input.fluid.density_rho, characteristic_W_outgoing);
              }
              else
              {
                FOUR_C_ASSERT(false, "no outlet boundary condition provided");
              }
            }

            FOUR_C_ASSERT(
                (boundary_u + sound_speed_c_boundary) * dt / L < 1, "CFL condition violated.");

            double F1 = A_condition * u_condition;
            double F2 = 0.5 * pow(u_condition, 2) +
                        (beta_element * (sqrt(A_condition) - sqrt(reference_area_element))) /
                            input.fluid.density_rho;
            double F1_h = boundary_A * boundary_u;
            double F2_h = 0.5 * pow(boundary_u, 2) +
                          (beta_element * (sqrt(boundary_A) - sqrt(reference_area_element))) /
                              input.fluid.density_rho;

            element_rhs(2 * boundary_local_index) +=
                normal_in_out[boundary_local_index] * (F1_h - F1);
            element_rhs(2 * boundary_local_index + 1) +=
                normal_in_out[boundary_local_index] * (F2_h - F2);
          }  // boundary
        }  // boundary || junction
      }  // loop over nodes in element
    };

    /** --------------------------------------------------------------------------------------------
     * Time steps
     */
    Teuchos::ParameterList dummy;
    // Visualization setup
    Core::IO::DiscretizationVisualizationWriterMesh visualization_writer(
        discretization, Core::IO::visualization_parameters_factory(
                            context.io_parameters->sublist("RUNTIME VTK OUTPUT"),
                            *context.output_control_file, 0.0));
    visualization_writer.reset();
    visualization_writer.append_element_owner("Owner");
    visualization_writer.append_result_data_vector_with_context(
        solution, Core::IO::OutputEntity::dof, {"A", "u"});
    visualization_writer.append_result_data_vector_with_context(
        pressure_solution, Core::IO::OutputEntity::node, {"p"});
    visualization_writer.append_result_data_vector_with_context(
        flow_solution, Core::IO::OutputEntity::node, {"Q"});
    visualization_writer.append_result_data_vector_with_context(
        radius_solution, Core::IO::OutputEntity::node, {"r"});
    visualization_writer.write_to_disk(0.0, 0);

    // Setup solver
    Core::LinAlg::Solver solver(*context.linear_solver_parameters, discretization->get_comm(),
        context.solver_params_callback, context.verbosity);

    auto y = std::make_shared<Core::LinAlg::Vector<double>>(*discretization->dof_row_map());

    // Time loop
    for (int i = 0; i < input.n_steps; i++)
    {
      time_n = i * dt;
      const double time_np = time_n + dt;

      // get mass_matrix and rhs
      discretization->evaluate(dummy, strategy, compute_local_contributions);

      // export characteristics information to junction mapping
      Core::LinAlg::export_to(owned_characteristics_at_junction, characteristics_for_junctions);

      get_conditions_at_junctions(input.fluid.density_rho, characteristics_for_junctions,
          solution_for_junction, normals_for_junctions, area0_for_junctions, beta_for_junctions,
          all_junctions, updated_solutions_on_junctions);

      update_rhs_with_junction_properties(input.fluid.density_rho, *rhs_junction,
          updated_solutions_on_junctions, beta, reference_area_0, normals, solution, all_junctions);

      //     update rhs with contributions from junction
      rhs->update(1.0, *rhs_junction, 1.0);

      mass_matrix->epetra_matrix().FillComplete();

      solver.reset();
      // y = M^-1 * rhs
      [[maybe_unused]] int error = solver.solve(mass_matrix, y, rhs, {});
      FOUR_C_ASSERT(error == 0, "Error Code {}: solver solve failed.", error);

      mass_matrix->zero();

      // U_n+1 = 1.0 * U_n + dt*y
      solution.update(dt, y->as_multi_vector(), 1.0);
      Core::LinAlg::export_to(solution, solution_for_evaluation);
      Core::LinAlg::export_to(solution, solution_for_junction);

      // reset used vectors
      rhs->put_scalar(0.0);
      rhs_junction->put_scalar(0.0);
      y->put_scalar(0.0);

      // output for visualization
      for (const auto& node : discretization->my_row_node_range())
      {
        int local_id = discretization->node_row_map()->lid(node.global_id());
        int A_id = 2 * local_id;
        FOUR_C_ASSERT(solution.get_values()[A_id] > 0,
            "area in solution_np < 0 at node {}, time {}", node.global_id(), time_n);
        double pressure_p =
            beta.get_values()[local_id] *
            (sqrt(solution.get_values()[A_id]) - sqrt(reference_area_0.get_values()[local_id]));
        pressure_solution.replace_local_value(local_id, pressure_p);

        flow_solution.replace_local_value(
            local_id, solution.get_values()[A_id] * solution.get_values()[A_id + 1]);
      }

      if ((i + 1) % input.result_every == 0)
      {
        TEUCHOS_FUNC_TIME_MONITOR("Write output");
        visualization_writer.reset();
        visualization_writer.append_element_owner("Owner");
        visualization_writer.append_result_data_vector_with_context(
            solution, Core::IO::OutputEntity::dof, {"A", "u"});
        visualization_writer.append_result_data_vector_with_context(
            pressure_solution, Core::IO::OutputEntity::node, {"p"});
        visualization_writer.append_result_data_vector_with_context(
            flow_solution, Core::IO::OutputEntity::node, {"Q"});

        visualization_writer.write_to_disk(time_np, i + 1);
      }
    }
    Teuchos::TimeMonitor::summarize();
    // Result tests
    auto sol_ptr = std::make_shared<const Core::LinAlg::Vector<double>>(solution);
    std::shared_ptr<Core::Utils::ResultTest> resulttest =
        std::make_shared<ReducedLung1dPipeFlow::ResultTest>(discretization, sol_ptr);
    context.add_field_test(resulttest);
    context.test_all(discretization->get_comm());
  }

  void main()
  {
    const ReducedLung1dPipeFlowContext context =
        make_reduced_lung_1d_pipe_flow_context_from_problem(*Global::Problem::instance());
    run_reduced_lung_1d_pipe_flow(context);
  }
}  // namespace ReducedLung1dPipeFlow
FOUR_C_NAMESPACE_CLOSE

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
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_mat_vplast_law.hpp"
#include "4C_reduced_lung_1d_pipe_flow_input.hpp"
#include "4C_reduced_lung_1d_pipe_flow_resulttest.hpp"
#include "4C_structure_new_timint_base.hpp"
#include "4C_utils_function_of_time.hpp"

#include <boost/graph/subgraph.hpp>
#include <boost/xpressive/detail/core/matcher/logical_newline_matcher.hpp>
#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN
namespace ReducedLung1dPipeFlow
{
  void fill_parameters(Parameters& parameters, Core::LinAlg::Vector<double>& solution,
      Core::LinAlg::Vector<double>& reference_area, Core::LinAlg::Vector<double>& thickness,
      Core::LinAlg::Vector<double>& Young, Core::LinAlg::Vector<double>& beta,
      Core::LinAlg::Vector<double>& radius, const Core::FE::Discretization& discretization)
  {
    const auto& reduced_lung_parameters = Global::Problem::instance()->reduced_lung_parameters();
    parameters = reduced_lung_parameters.get<Parameters>("general");

    // constants to be computed from the input parameters
    parameters.fluid.K_R =
        8.0 * M_PI * parameters.fluid.viscosity_mu / parameters.fluid.density_rho;

    // elemental vectors
    for (const auto& node : discretization.my_row_node_range())
    {
      const int node_gid = node.global_id();
      const int element_global_id = node.adjacent_elements()[0].global_id();

      const double Young_value = parameters.material.Young_E.at(element_global_id);
      const double A0_value = parameters.geometry.reference_area_A0.at(element_global_id);
      const double r0_value = std::sqrt(A0_value / M_PI);
      const double th_value = parameters.geometry.thickness_th.at(element_global_id);
      const double beta_value =
          (sqrt(M_PI) * th_value * Young_value) /
          ((1 - std::pow(parameters.material.Poisson_ratio_nue, 2)) * A0_value);

      Young.replace_global_value(node_gid, Young_value);
      reference_area.replace_global_value(node_gid, A0_value);
      radius.replace_global_value(node_gid, r0_value);
      thickness.replace_global_value(node_gid, th_value);
      beta.replace_global_value(node_gid, beta_value);

      // initial state
      solution.replace_global_value(2 * node_gid, A0_value);
      solution.replace_global_value(2 * node_gid + 1, 0.0);
    }
    // Boundary conditions
    parameters.boundary_conditions.bc_fct =
        &Global::Problem::instance()
             ->function_manager()
             .function_by_id<Core::Utils::FunctionOfTime>(
                 parameters.boundary_conditions.function_id_inflow);
  }

  auto compute_length(const Core::Elements::Element& element)
  {
    const auto& p1 = element.nodes()[0]->x();
    const auto& p2 = element.nodes()[1]->x();

    // Calculate the length of artery element
    const double L = std::sqrt(
        std::pow(p1[0] - p2[0], 2) + std::pow(p1[1] - p2[1], 2) + std::pow(p1[2] - p2[2], 2));
    FOUR_C_ASSERT(L > 0, "Element length needs to be  > 0");
    return L;
  }

  auto compute_psi_matrix(Core::LinAlg::Matrix<2, 4>& Psi_matrix,
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
    // step 1: initial guess: Q(A0), W1 = 2Q/A0 - W2
    double W_in = 2 * Q_condition / boundary_A0 - characteristic_W_outgoing;
    double f = pow(W_in - characteristic_W_outgoing, 4) / 1024 *
               pow(input.fluid.density_rho / beta, 2) * 0.5 * (W_in + characteristic_W_outgoing);

    int itrs = 0;
    while (fabs(f) > 0.000001)
    {
      // step 2: calculate df/dW
      double dfdW1 = pow(input.fluid.density_rho / beta, 2) * (1.0 / 1024.0) *
                     pow(W_in - characteristic_W_outgoing, 3) *
                     (2.5 * W_in + 1.5 * characteristic_W_outgoing);
      // step 3: find W_in(n+1) = W_in(n) - f(n)/dfdW1(n)
      W_in = W_in - f / dfdW1;
      // step 4: calculate error f
      f = pow(W_in - characteristic_W_outgoing, 4) / 1024 * pow(input.fluid.density_rho / beta, 2) *
              0.5 * (W_in + characteristic_W_outgoing) -
          Q_condition;
      // prevent infinite loop
      itrs++;
      if (itrs >= 100)
      {
        FOUR_C_THROW("Newton Raphson method exceeded maximum allowed iterations");
      }
    }
    A_condition =
        pow(input.fluid.density_rho / beta, 2) * pow(W_in - characteristic_W_outgoing, 4) / 1024;
    u_condition = 0.5 * (W_in + characteristic_W_outgoing);
  }

  void main()
  {
    std::shared_ptr<Core::FE::Discretization> discretization =
        Global::Problem::instance()->get_dis("artery");
    if (!discretization->filled() || !discretization->have_dofs())
    {
      discretization->fill_complete({.assign_degrees_of_freedom = true,
          .init_elements = true,
          .do_boundary_conditions = true});
    }

    // parameters for time and blood from input file
    Parameters input;

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

    /************/

    double time_n = 0.0;
    double dt = input.final_time / input.n_steps;
    double min_time_step_size = 1.0;

    Core::FE::AssembleStrategy strategy(0, 0, mass_matrix, nullptr, rhs, nullptr, nullptr);

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

      // check if element is part of a boundary
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

        // check if junction, capillary or boundary
        if (normal_in_out[inode] != 0)
        {
          is_boundary[inode] = true;
        }
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
            (pow(reference_area_element * (1 - pow(input.material.Poisson_ratio_nue, 2)), 2));
        const double dp_dA0 = ((-0.5) * beta_element) / sqrt(reference_area_element);

        Core::LinAlg::Matrix<2, 1> S;
        S.put_scalar(0.0);
        S(1, 0) += (-1 / input.fluid.density_rho) *
                   (input.fluid.K_R * input.fluid.density_rho * velocity_u / area_A +
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
        if (is_boundary[boundary_local_index])
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

          // extrapolation: computation of outgoing characteristic extrapolated from last time step
          // (x = 0 + lambda_2 * dt)
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


          /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
          // when outer boundary, prescribed parameter and reflection boundary condition
          /+********************************************************************************/

          double u_condition = 0.0;
          double A_condition = boundary_A;

          // precribed inflow
          double heavyside = 1.0;
          double time_cyc = time_n;
          while (time_cyc > input.boundary_conditions.cycle_period)
          {
            time_cyc -= input.boundary_conditions.cycle_period;
          }
          if (time_cyc > input.boundary_conditions.pulse_width)
          {
            heavyside = 0.0;
          }
          // inlet
          if (normal_in_out[boundary_local_index] == -1)
          {
            //  modify outgoing characteristic if source term is used
            characteristic_W_outgoing += dt * (-input.fluid.K_R * boundary_u / boundary_A);

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
                  (pressure_fct - boundary_Pext) / beta_element + sqrt(reference_area_element), 2);
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
              A_condition = pow((characteristic_W_outgoing - characteristic_W_incoming) * 0.25, 4) *
                            pow(input.fluid.density_rho * 0.5 / beta_element, 2);
              u_condition = (characteristic_W_outgoing + characteristic_W_incoming) * 0.5;
            }
            else if (input.boundary_conditions.output == "pressure")
            {
              // prescribed p at outlet
              A_condition = pow((input.boundary_conditions.condition_outflow * 1333.22 -
                                    boundary_Pext) /  // Conversion mmHg -> dyn/cm2
                                        beta_element +
                                    sqrt(reference_area_element),
                  2);
              u_condition = characteristic_W_outgoing -
                            4 * std::pow(A_condition, 0.25) *
                                sqrt(0.5 * beta_element / input.fluid.density_rho);
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
      }  // loop over nodes in element
    };

    /** --------------------------------------------------------------------------------------------
     * Time steps
     */
    Teuchos::ParameterList dummy;
    // Visualization setup
    Core::IO::DiscretizationVisualizationWriterMesh visualization_writer(
        discretization, Core::IO::visualization_parameters_factory(
                            Global::Problem::instance()->io_params().sublist("RUNTIME VTK OUTPUT"),
                            *Global::Problem::instance()->output_control_file(), 0.0));
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

    auto y = std::make_shared<Core::LinAlg::Vector<double>>(*discretization->dof_row_map());

    // Time loop
    for (int i = 0; i < input.n_steps; i++)
    {
      time_n = i * dt;
      const double time_np = time_n + dt;

      // get mass_matrix and rhs
      discretization->evaluate(dummy, strategy, compute_local_contributions);
      mass_matrix->epetra_matrix().FillComplete();

      Core::LinAlg::Solver solver(Global::Problem::instance()->solver_params(1),
          discretization->get_comm(), Global::Problem::instance()->solver_params_callback(),
          Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
              Global::Problem::instance()->io_params(), "VERBOSITY"));

      // y = M^-1 * rhs
      [[maybe_unused]] int error = solver.solve(mass_matrix, y, rhs, {});
      FOUR_C_ASSERT(error == 0, "Error Code {}: solver solve failed.", error);

      mass_matrix->zero();

      // U_n+1 = 1.0 * U_n + dt*y
      solution.update(dt, y->as_multi_vector(), 1.0);
      Core::LinAlg::export_to(solution, solution_for_evaluation);

      // reset used vectors
      rhs->put_scalar(0.0);
      y->put_scalar(0.0);
      // output for visualization
      for (const auto& node : discretization->my_row_node_range())
      {
        int local_id = discretization->node_row_map()->lid(node.global_id());
        int A_id = 2 * local_id;
        FOUR_C_ASSERT(solution.get_values()[A_id] > 0,
            "area in solution_np < 0 at node {}, time {}", node.global_id(), time_n);
        double pressure_p =
            (1 / 1333.22) * beta.get_values()[local_id] *
            (sqrt(solution.get_values()[A_id]) - sqrt(reference_area_0.get_values()[local_id]));
        pressure_solution.replace_local_value(local_id, pressure_p);  // p in mmHg

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
        std::make_shared<ReducedLung1D::ResultTest>(discretization, sol_ptr);
    Global::Problem::instance()->add_field_test(resulttest);
    Global::Problem::instance()->test_all(discretization->get_comm());
  }
}  // namespace ReducedLung1dPipeFlow
FOUR_C_NAMESPACE_CLOSE

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_solver_nonlin_nox_adapter.hpp"

#include "4C_io_pstream.hpp"
#include "4C_linalg_map.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_linear_solver_method.hpp"
#include "4C_linear_solver_method_linalg.hpp"

#include <mpi.h>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCPDecl.hpp>

#include <cmath>
#include <memory>
#include <utility>

namespace
{
  using namespace FourC;

  Teuchos::ParameterList create_test_nox_parameter_list()
  {
    Teuchos::ParameterList nox_params;
    nox_params.set("Nonlinear Solver", "Line Search Based");

    auto& pdir = nox_params.sublist("Direction");
    pdir.set("Method", "Newton");
    pdir.sublist("Newton").sublist("Linear Solver");

    nox_params.sublist("Line Search").set("Method", "Full Step");

    auto& outer = nox_params.sublist("Status Test").sublist("Outer Status Test");
    outer.set("Test Type", "Combo");
    outer.set("Combo Type", "OR");

    auto& norm_f = outer.sublist("Test 0");
    norm_f.set("Test Type", "NormF");
    norm_f.set("Quantity Type", "Generic");
    norm_f.set("Tolerance Type", "Absolute");
    norm_f.set<double>("Tolerance", 1.0e-16);
    norm_f.set("Norm Type", "Two Norm");
    norm_f.set("Scale Type", "Unscaled");

    auto& max_iters = outer.sublist("Test 1");
    max_iters.set("Test Type", "MaxIters");
    max_iters.set<int>("Maximum Iterations", 40);

    return nox_params;
  }

  TEST(SolverNonlinNoxAdapter, SolvesScalarQuadratic)
  {
    MPI_Comm comm = MPI_COMM_WORLD;

    Core::LinAlg::Map map(1, 0, comm);
    Core::LinAlg::Vector<double> x(map, true);
    Core::LinAlg::Vector<double> rhs(map, true);
    Core::LinAlg::SparseMatrix jacobian(map, map, 1);

    if (x.local_length() > 0) x.replace_global_value(0, 2.0);

    Teuchos::ParameterList solver_params;
    solver_params.set("SOLVER", Core::LinearSolver::SolverType::UMFPACK);
    solver_params.set("NAME", "NoxAdapterSolver");
    const auto get_solver_params = [&](int) -> const Teuchos::ParameterList&
    { return solver_params; };

    auto linear_solver = Teuchos::make_rcp<Core::LinAlg::Solver>(
        solver_params, comm, get_solver_params, Core::IO::Verbositylevel::minimal);

    Teuchos::ParameterList nox_params = create_test_nox_parameter_list();

    std::map<FourC::NOX::Nln::SolutionType, Teuchos::RCP<Core::LinAlg::Solver>> solver_map;
    solver_map[FourC::NOX::Nln::sol_generic] = linear_solver;

    FourC::NOX::Nln::Adapter::ResidualCallback residual_callback =
        [&](const Core::LinAlg::Vector<double>& x_vec, Core::LinAlg::Vector<double>& f_vec,
            FourC::NOX::Nln::FillType) -> bool
    {
      if (x_vec.local_length() == 0) return true;

      const double x_val = x_vec.get_values()[0];
      f_vec.replace_local_value(0, x_val * x_val);
      return true;
    };

    FourC::NOX::Nln::Adapter::JacobianCallback jacobian_callback =
        [&](const Core::LinAlg::Vector<double>& x_vec, Core::LinAlg::SparseOperator& jac) -> bool
    {
      auto& jac_matrix = dynamic_cast<Core::LinAlg::SparseMatrix&>(jac);
      jac_matrix.reset();

      if (x_vec.local_length() > 0)
      {
        const double x_val = x_vec.get_values()[0];
        jac_matrix.set_value(2.0 * x_val, 0, 0);
      }

      jac_matrix.complete();
      return true;
    };

    FourC::NOX::Nln::Adapter adapter(comm, nox_params, solver_map, x, jacobian,
        std::move(residual_callback), std::move(jacobian_callback));

    const unsigned int iterations = adapter.solve();
    EXPECT_GT(iterations, 0u);

    if (x.local_length() > 0)
    {
      EXPECT_NEAR(x.get_values()[0], 0.0, 1.0e-8);
    }
  }

  TEST(SolverNonlinNoxAdapter, RepeatedSolveReinitializesState)
  {
    MPI_Comm comm = MPI_COMM_WORLD;

    Core::LinAlg::Map map(1, 0, comm);
    Core::LinAlg::Vector<double> x(map, true);
    Core::LinAlg::SparseMatrix jacobian(map, map, 1);

    if (x.local_length() > 0) x.replace_global_value(0, 0.0);

    Teuchos::ParameterList solver_params;
    solver_params.set("SOLVER", Core::LinearSolver::SolverType::UMFPACK);
    solver_params.set("NAME", "NoxAdapterSolver");
    const auto get_solver_params = [&](int) -> const Teuchos::ParameterList&
    { return solver_params; };

    auto linear_solver = Teuchos::make_rcp<Core::LinAlg::Solver>(
        solver_params, comm, get_solver_params, Core::IO::Verbositylevel::minimal);

    Teuchos::ParameterList nox_params = create_test_nox_parameter_list();

    std::map<FourC::NOX::Nln::SolutionType, Teuchos::RCP<Core::LinAlg::Solver>> solver_map;
    solver_map[FourC::NOX::Nln::sol_generic] = linear_solver;

    double target = 0.1;
    FourC::NOX::Nln::Adapter::ResidualCallback residual_callback =
        [&](const Core::LinAlg::Vector<double>& x_vec, Core::LinAlg::Vector<double>& f_vec,
            FourC::NOX::Nln::FillType) -> bool
    {
      if (x_vec.local_length() == 0) return true;
      f_vec.replace_local_value(0, x_vec.get_values()[0] - target);
      return true;
    };

    FourC::NOX::Nln::Adapter::JacobianCallback jacobian_callback =
        [&](const Core::LinAlg::Vector<double>& x_vec, Core::LinAlg::SparseOperator& jac) -> bool
    {
      auto& jac_matrix = dynamic_cast<Core::LinAlg::SparseMatrix&>(jac);
      jac_matrix.reset();
      if (x_vec.local_length() > 0) jac_matrix.set_value(1.0, 0, 0);
      jac_matrix.complete();
      return true;
    };

    FourC::NOX::Nln::Adapter adapter(comm, nox_params, solver_map, x, jacobian,
        std::move(residual_callback), std::move(jacobian_callback));

    const unsigned int iterations_step_1 = adapter.solve();
    EXPECT_GT(iterations_step_1, 0u);

    double x_after_step_1 = 0.0;
    if (x.local_length() > 0)
    {
      x_after_step_1 = x.get_values()[0];
      EXPECT_NEAR(x_after_step_1, 0.1, 1.0e-12);
    }

    target = 0.2;
    const unsigned int iterations_step_2 = adapter.solve();
    EXPECT_GT(iterations_step_2, 0u);

    if (x.local_length() > 0)
    {
      const double x_after_step_2 = x.get_values()[0];
      EXPECT_NEAR(x_after_step_2, 0.2, 1.0e-12);
      EXPECT_GT(std::abs(x_after_step_2 - x_after_step_1), 1.0e-6);
    }
  }
}  // namespace

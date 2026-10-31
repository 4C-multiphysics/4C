// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_reduced_lung_nox_solver.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_sparseoperator.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_utils_exceptions.hpp"

#include <map>

FOUR_C_NAMESPACE_OPEN

namespace ReducedLung
{
  NoxSolver::NoxSolver(const NoxSolverContext& context, double initial_time)
      : x_solution_(context.x),
        dofs_(context.dofs),
        locally_relevant_dofs_(context.locally_relevant_dofs),
        assembly_pipeline_(context.assembly_pipeline),
        dt_(context.dynamics.time_increment),
        current_time_(initial_time),
        linear_solver_(std::make_shared<Core::LinAlg::Solver>(context.linear_solver_parameters,
            context.comm, context.solver_params_callback, Core::IO::Verbositylevel::minimal))
  {
    if (!linear_solver_)
    {
      FOUR_C_THROW("ReducedLung::NoxSolver requires a valid linear solver instance.");
    }
    if (assembly_pipeline_.residual_assemblers.empty())
    {
      FOUR_C_THROW("ReducedLung::NoxSolver requires at least one residual assembler callback.");
    }
    if (assembly_pipeline_.jacobian_assemblers.empty())
    {
      FOUR_C_THROW("ReducedLung::NoxSolver requires at least one Jacobian assembler callback.");
    }

    // Create NOX parameter list
    Teuchos::ParameterList nox_params = create_nox_parameter_list(context.dynamics);

    // Create callbacks that bind to this instance's methods
    NOX::Nln::Adapter::ResidualCallback residual_callback =
        [this](const Core::LinAlg::Vector<double>& x_vec, Core::LinAlg::Vector<double>& rhs,
            NOX::Nln::FillType fill_type) -> bool { return this->residual(x_vec, rhs, fill_type); };

    NOX::Nln::Adapter::JacobianCallback jacobian_callback =
        [this](const Core::LinAlg::Vector<double>& x_vec, Core::LinAlg::SparseOperator& jac) -> bool
    { return this->jacobian(x_vec, jac); };

    // Setup solver map
    std::map<NOX::Nln::SolutionType, Teuchos::RCP<Core::LinAlg::Solver>> solver_map;
    solver_map[NOX::Nln::sol_generic] = Teuchos::rcpFromRef(*linear_solver_);


    // Construct NOX adapter in-place using std::optional::emplace
    adapter_.emplace(context.comm, nox_params, solver_map, context.x, context.jacobian,
        std::move(residual_callback), std::move(jacobian_callback));
  }

  unsigned int NoxSolver::solve(double time)
  {
    current_time_ = time;
    unsigned int iterations = adapter_->solve();
    sync_state_from_x(x_solution_);

    return iterations;
  }

  bool NoxSolver::residual(const Core::LinAlg::Vector<double>& x,
      Core::LinAlg::Vector<double>& residual, NOX::Nln::FillType /*fill_type*/)
  {
    sync_state_from_x(x);

    for (const auto& assemble_residual : assembly_pipeline_.residual_assemblers)
    {
      assemble_residual(residual, locally_relevant_dofs_, current_time_, dt_);
    }

    return true;
  }

  bool NoxSolver::jacobian(const Core::LinAlg::Vector<double>& x, Core::LinAlg::SparseOperator& jac)
  {
    sync_state_from_x(x);

    auto* jac_matrix = dynamic_cast<Core::LinAlg::SparseMatrix*>(&jac);
    if (jac_matrix == nullptr)
    {
      FOUR_C_THROW(
          "ReducedLung NOX assembly requires Core::LinAlg::SparseMatrix Jacobian operator.");
    }

    for (const auto& assemble_jacobian : assembly_pipeline_.jacobian_assemblers)
    {
      assemble_jacobian(*jac_matrix, locally_relevant_dofs_, current_time_, dt_);
    }

    if (!jac_matrix->filled())
    {
      jac_matrix->complete();
    }

    return true;
  }

  void NoxSolver::sync_state_from_x(const Core::LinAlg::Vector<double>& x)
  {
    if (Core::Communication::num_mpi_ranks(x.get_comm()) == 1)
    {
      dofs_.update(1.0, x, 0.0);
      locally_relevant_dofs_.update(1.0, x, 0.0);
    }
    else
    {
      Core::LinAlg::export_to(x, dofs_);
      Core::LinAlg::export_to(dofs_, locally_relevant_dofs_);
    }

    for (const auto& update_state : assembly_pipeline_.state_updaters)
    {
      update_state(locally_relevant_dofs_, dt_);
    }
  }

  Teuchos::ParameterList create_nox_parameter_list(const ReducedLungParameters::Dynamics& dynamics)
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

    auto& converged = outer.sublist("Test 0");
    converged.set("Test Type", "Combo");
    converged.set("Combo Type", "AND");

    auto& norm_f = converged.sublist("Test 0");
    norm_f.set("Test Type", "NormF");
    norm_f.set("Quantity Type", "Generic");
    norm_f.set("Tolerance Type", "Absolute");
    norm_f.set<double>("Tolerance", dynamics.nonlinear_residual_tolerance);
    norm_f.set("Norm Type", "Two Norm");
    norm_f.set("Scale Type", "Unscaled");

    auto& norm_update = converged.sublist("Test 1");
    norm_update.set("Test Type", "NormUpdateSkipFirstIter");
    norm_update.set("Quantity Type", "Generic");
    norm_update.set("Tolerance Type", "Absolute");
    norm_update.set<double>("Tolerance", dynamics.nonlinear_increment_tolerance);
    norm_update.set("Norm Type", "Two Norm");
    norm_update.set("Scale Type", "Unscaled");

    auto& max_iters = outer.sublist("Test 1");
    max_iters.set("Test Type", "MaxIters");
    max_iters.set<int>("Maximum Iterations", dynamics.max_nonlinear_iterations);

    return nox_params;
  }
}  // namespace ReducedLung

FOUR_C_NAMESPACE_CLOSE

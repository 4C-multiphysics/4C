// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_linalg_map.hpp"
#include "4C_linalg_multi_vector.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linear_solver_method.hpp"
#include "4C_linear_solver_method_direct.hpp"

#include <mpi.h>
#include <Teuchos_ParameterList.hpp>

#include <memory>
#include <numeric>
#include <vector>

namespace
{
  using namespace FourC;

  constexpr int num_rows = 20;

  class DirectSolverListMap : public ::testing::TestWithParam<Core::LinearSolver::SolverType>
  {
  };

  // A map built from the list [0, N) is not flagged as linear by Epetra but is contiguous
  TEST_P(DirectSolverListMap, OrderedListMapGivesCorrectSolution)
  {
    std::vector<int> gids(num_rows);
    std::iota(gids.begin(), gids.end(), 0);

    const Core::LinAlg::Map map(num_rows, std::span<const int>(gids), 0, MPI_COMM_WORLD);
    ASSERT_FALSE(map.linear_map());

    auto matrix = std::make_shared<Core::LinAlg::SparseMatrix>(map, 3);
    // not symmetric, so that a wrong numbering is not hidden
    for (int gid = 0; gid < num_rows; ++gid)
    {
      matrix->set_value(4.0, gid, gid);
      if (gid > 0) matrix->set_value(-1.0, gid, gid - 1);
      if (gid < num_rows - 1) matrix->set_value(-2.0, gid, gid + 1);
    }
    matrix->complete();

    auto b = std::make_shared<Core::LinAlg::MultiVector<double>>(map, 1, true);
    for (int gid = 0; gid < num_rows; ++gid) b->replace_global_value(gid, 0, 1.0 + gid * gid);

    Core::LinearSolver::DirectSolver solver(GetParam(), Teuchos::ParameterList());
    solver.setup(matrix, b, /*refactor=*/true);

    Core::LinAlg::MultiVector<double> x(map, 1, true);
    solver.solve(x);

    Core::LinAlg::MultiVector<double> residual(map, 1, true);
    matrix->multiply(false, x, residual);
    residual.update(1.0, *b, -1.0);

    double residual_norm = 0.0;
    double rhs_norm = 0.0;
    residual.norm_2(&residual_norm);
    b->norm_2(&rhs_norm);
    EXPECT_LT(residual_norm / rhs_norm, 1.0e-12);
  }

  INSTANTIATE_TEST_SUITE_P(AllDirectSolvers, DirectSolverListMap,
      ::testing::Values(Core::LinearSolver::SolverType::KLU2,
          Core::LinearSolver::SolverType::UMFPACK, Core::LinearSolver::SolverType::Superlu));
}  // namespace

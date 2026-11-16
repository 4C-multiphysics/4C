// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_reduced_lung_input.hpp"
#include "4C_reduced_lung_newton_sparse.hpp"

#include <Teuchos_ParameterList.hpp>

#include <string>

namespace
{
  using namespace FourC;
  using namespace FourC::ReducedLung;

  TEST(ReducedLungNewtonSparseParameters, ConvergesOnTheResidualAlone)
  {
    ReducedLungParameters::Dynamics dynamics{};
    dynamics.max_nonlinear_iterations = 7;
    dynamics.nonlinear_residual_tolerance = 1e-8;
    dynamics.nonlinear_increment_tolerance = 1e-10;

    const Teuchos::ParameterList nox_params = create_nox_parameter_list(dynamics);
    const auto& outer = nox_params.sublist("Status Test").sublist("Outer Status Test");

    EXPECT_EQ(outer.get<std::string>("Combo Type"), "OR");

    const auto& norm_f = outer.sublist("Test 0");
    EXPECT_EQ(norm_f.get<std::string>("Test Type"), "NormF");
    EXPECT_DOUBLE_EQ(norm_f.get<double>("Tolerance"), dynamics.nonlinear_residual_tolerance);

    const auto& max_iters = outer.sublist("Test 1");
    EXPECT_EQ(max_iters.get<std::string>("Test Type"), "MaxIters");
    EXPECT_EQ(max_iters.get<int>("Maximum Iterations"), dynamics.max_nonlinear_iterations);

    EXPECT_FALSE(outer.isSublist("Test 2"));
  }
}  // namespace

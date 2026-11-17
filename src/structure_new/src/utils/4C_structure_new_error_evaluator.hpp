// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_STRUCTURE_NEW_ERROR_EVALUATOR_HPP
#define FOUR_C_STRUCTURE_NEW_ERROR_EVALUATOR_HPP

#include "4C_config.hpp"

#include "4C_fem_discretization.hpp"

#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN

namespace Solid
{
  namespace ErrorEvaluator
  {
    struct Parameters
    {
      bool evaluate_error_analytical;
      int evaluate_error_analytical_displacement_function_id;
      int evaluate_error_analytical_lagrange_multiplier_function_id;
    };

    struct Results
    {
      std::optional<double> integrated_squared_error_displacements;
      std::optional<double> integrated_squared_gradient_error_displacements;
      std::optional<double> integrated_squared_displacements;
      std::optional<double> integrated_volume;

      std::optional<double> integrated_squared_error_lagrange_multipliers;
    };

    Parameters error_evaluator_parameter_factory(const Teuchos::ParameterList& parameter_list);

    void write_error(const Results& results, Core::FE::Discretization& discretization);

  }  // namespace ErrorEvaluator
}  // namespace Solid

FOUR_C_NAMESPACE_CLOSE

#endif

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
    };

    Parameters error_evaluator_parameter_factory(const Teuchos::ParameterList& parameter_list);

    /*! \brief compute L2 and H1 displacement error norm with respect to an analytical solution
     *
     *  \param error_evaluator_parameters          error evaluation parameters from input file
     *  \param discretization_evaluation_parameters        discretization parameter interface
     *  \param discretization  discretization object
     *
     */
    void evaluate_error(const Parameters& error_evaluator_parameters,
        Teuchos::ParameterList& discretization_evaluation_parameters,
        Core::FE::Discretization& discretization);

  }  // namespace ErrorEvaluator
}  // namespace Solid

FOUR_C_NAMESPACE_CLOSE

#endif

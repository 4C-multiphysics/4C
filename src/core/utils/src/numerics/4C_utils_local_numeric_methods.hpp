// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_UTILS_LOCAL_NUMERIC_METHODS_HPP
#define FOUR_C_UTILS_LOCAL_NUMERIC_METHODS_HPP

#include "4C_config.hpp"

#include <functional>

FOUR_C_NAMESPACE_OPEN

namespace Core::Utils
{
  /*!
   * @brief Structure holding the values of a function f(x) and its first derivative dfdx(x) and
   * second derivative ddfddx(x) at x
   */
  struct ValuesFunctAndFunctDerivs
  {
    double val_funct;
    double val_deriv_funct;
    double val_deriv_deriv_funct;
  };

  /*!
   * @brief Find a zero of a continuous function using the bisection method
   *
   * Find a zero of the scalar-valued continuous function func defined on an interval [a, b]
   * given the starting endpoints of the interval a_init and b_init. Note that f(a) and f(b)
   * need to have opposite signs.
   *
   * The input function needs to be defined as a function that accepts the argument x and that
   * returns a double holding f(x)
   *
   * @param[in]     func Scalar function f(x) whose zero should be found
   * @param[in]     a_init Starting point of a for the interval [a,b]
   * @param[in]     b_init Starting point of b for the interval [a,b]
   * @param[in]     tol Tolerance for the zero value f(c)
   * @param[in]     maxiter Maximal number of iterations
   * @return        c Midpoint c where f(c) is zero
   */
  double bisection(const std::function<double(double)>& func, const double a_init,
      const double b_init, const double tol, const int maxiter);

  /*!
   * @brief Evaluate a function f(x) and the first and second derivatives w.r.t. x using central
   * differences
   *
   * @param[in]     func Scalar function f(x) to be evaluated
   * @param[in]     x Point x where f(x) and derivatives should be evaluated
   * @param[in]     delta_x Step size x_{i+1}-x_i
   */
  ValuesFunctAndFunctDerivs evaluate_function_and_derivatives_central_differences(
      const std::function<double(double)>& func, const double x, const double delta_x);

  /*!
   * @brief Evaluate the first derivative of f(x) w.r.t. x using central differences
   *
   * @param[in]     f_i_minus_1 Solution f(x_{i-1})
   * @param[in]     f_i_plus_1 Solution f(x_{i+1})
   * @param[in]     delta_x Step size x_{i+1}-x_i
   * @param[out]    dfdx First derivative of f at x_i
   */
  double first_derivative_central_differences(
      const double f_i_minus_1, const double f_i_plus_1, const double delta_x);

  /*!
   * @brief Evaluate the second derivative of f(x) w.r.t. x using central differences
   *
   * @param[in]     f_i_minus_1 Solution f(x_{i-1})
   * @param[in]     f_i Solution f(x_{i})
   * @param[in]     f_i_plus_1 Solution f(x_{i+1})
   * @param[in]     delta_x Step size x_{i+1}-x_i
   * @param[out]    ddfddx Second derivative of f at x_i
   */
  double second_derivative_central_differences(
      const double f_i_minus_1, const double f_i, const double f_i_plus_1, const double delta_x);

}  // namespace Core::Utils

FOUR_C_NAMESPACE_CLOSE

#endif

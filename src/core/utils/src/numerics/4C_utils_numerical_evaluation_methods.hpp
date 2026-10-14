// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_UTILS_NUMERICAL_EVALUATION_METHODS_HPP
#define FOUR_C_UTILS_NUMERICAL_EVALUATION_METHODS_HPP

#include "4C_config.hpp"

#include <functional>
#include <initializer_list>
#include <span>

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


  /// Status for the computation of the general, signed log-sum-exp operation \f$ \log(y) =
  /// \log\left[\sum_{i=1}^{N} p_i \exp( x_i )\right] \f$
  enum class SignedLogSumExpStatus
  {
    success,        ///< successful computation
    zero_argument,  ///< argument \f$ \left[\sum_{i=1}^{N} p_i \exp( x_i )\right]\f$ is numerically
                    ///< too close to 0.0 such that evaluating its logarithm leads to \f$ - \infty
                    ///< \f$
    overflow,       ///< computation leads to overflow due to high input prefactors
  };

  /// Result struct for the general, signed log-sum-exp operation \f$ \log(y) =
  /// \log\left[\sum_{i=1}^{N} p_i \exp( x_i )\right] \f$
  struct SignedLogSumExpResult
  {
    /// logarithm of the absolute value \f$ \log(\left| y \right|) \f$
    double log_abs;

    /// sign \f$ \mathrm{sign} y \f$; either -1.0 or 1.0
    double sign;

    /// status of the evaluation
    SignedLogSumExpStatus eval_status;
  };

  /// Compute the general, signed log-sum-exp operation \f$ \log(y) =  \log\left[\sum_{i = 1}^{N}
  /// p_i  \exp( x_i )\right] \f$ for given \p log_values \f$ x_i \f$ and \p prefactors \f$ p_i \f$.
  /// Note that the function internally asserts that the given \p log_values and \p prefactors are
  /// finite doubles; the calling procedure must ensure this requirement. Precision will be lost for
  /// extremely high magnitude inputs. Also, the prefactors are not allowed to be under machine
  /// precision in the current implementation.
  SignedLogSumExpResult signed_log_sum_exp(
      std::span<const double> log_values, std::span<const double> prefactors);

  /// \overload Convenience function with initializer lists
  SignedLogSumExpResult signed_log_sum_exp(
      std::initializer_list<double> log_values, std::initializer_list<double> prefactors);

}  // namespace Core::Utils

FOUR_C_NAMESPACE_CLOSE

#endif

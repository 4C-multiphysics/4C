// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_utils_numerical_evaluation_methods.hpp"

#include "4C_utils_exceptions.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double Core::Utils::bisection(const std::function<double(double)>& funct, const double a_init,
    const double b_init, const double tol, const int maxiter)
{
  double a = a_init;
  double b = b_init;
  double c;
  int numiter = 0;

  while (numiter <= maxiter)
  {
    numiter++;
    c = (a + b) / 2;

    auto f_c = funct(c);
    auto f_a = funct(a);

    if (std::fabs(f_c) < tol)
    {
      return c;
    }
    // sgn(f(a)) == sgn(f(c))
    else if (f_c * f_a > 0)
    {
      a = c;
    }
    else
    {
      b = c;
    }
  }

  FOUR_C_THROW(
      "Maximal number of iterations reached for Bisection method ({} iterations). Error is "
      "still at {:14.14f}",
      numiter, (b - a) / 2);

  return c;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::Utils::ValuesFunctAndFunctDerivs
Core::Utils::evaluate_function_and_derivatives_central_differences(
    const std::function<double(double)>& func, const double x, const double delta_x)
{
  ValuesFunctAndFunctDerivs f_df_ddf;

  f_df_ddf.val_funct = func(x);

  double f_i_minus_1 = func(x - delta_x);
  double f_i_plus_1 = func(x + delta_x);
  f_df_ddf.val_deriv_funct = first_derivative_central_differences(f_i_minus_1, f_i_plus_1, delta_x);
  f_df_ddf.val_deriv_deriv_funct =
      second_derivative_central_differences(f_i_minus_1, f_df_ddf.val_funct, f_i_plus_1, delta_x);

  return f_df_ddf;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double Core::Utils::first_derivative_central_differences(
    const double f_i_minus_1, const double f_i_plus_1, const double delta_x)
{
  double dfdx = (f_i_plus_1 - f_i_minus_1) / (2 * delta_x);
  return dfdx;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double Core::Utils::second_derivative_central_differences(
    const double f_i_minus_1, const double f_i, const double f_i_plus_1, const double delta_x)
{
  double ddfddx = (f_i_plus_1 - 2 * f_i + f_i_minus_1) / (delta_x * delta_x);
  return ddfddx;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::Utils::SignedLogSumExpResult Core::Utils::signed_log_sum_exp(
    std::span<const double> log_values, std::span<const double> prefactors)
{
  FOUR_C_ASSERT_ALWAYS(not log_values.empty(), "No values were given for this computation");
  FOUR_C_ASSERT_ALWAYS(log_values.size() == prefactors.size(),
      "The number of values must match the number of prefactors: currently {} versus {}, "
      "respectively",
      log_values.size(), prefactors.size());

  // determine the maximum value \f$ M = \max x_i \f$
  const double max_value = *(std::max_element(log_values.begin(), log_values.end()));

  // compute reduced sum \f$ S = \sum_{i=1}^{N} p_i \exp( x_i - M) \f$
  double reduced_sum = 0.0;
  double reduced_sum_scale =
      1.0;  // scaling factor used to normalize partial sums in order to avoid overflow
  for (std::size_t i = 0; i < log_values.size(); ++i)
  {
    // assert that the given input prefactors and log values are finite
    FOUR_C_ASSERT_ALWAYS(std::isfinite(prefactors[i]),
        "Given prefactor {} with index {} is non-finite! The calling procedure must deal with such "
        "cases itself!",
        prefactors[i], i);
    FOUR_C_ASSERT_ALWAYS(std::abs(prefactors[i]) > std::numeric_limits<double>::epsilon(),
        "Given prefactor {} with index {} has an absolute value smaller than precision epsilon {}. "
        "This leads to "
        "inexact computations within the log-sum-exp.",
        prefactors[i], i, std::numeric_limits<double>::epsilon());
    FOUR_C_ASSERT_ALWAYS(std::isfinite(log_values[i]),
        "Given log_value {} with index {} is non-finite! The calling procedure must deal with such "
        "cases itself!",
        log_values[i], i);
    const double add_term = prefactors[i] * std::exp(log_values[i] - max_value);
    const double current_reduced_sum_scale = std::max(reduced_sum_scale, std::abs(add_term));
    reduced_sum *= reduced_sum_scale / current_reduced_sum_scale;
    reduced_sum += add_term / current_reduced_sum_scale;
    reduced_sum_scale = current_reduced_sum_scale;
  }
  // scale reduced sum back to its original order of magnitude
  reduced_sum *= reduced_sum_scale;
  const double sign_reduced_sum =
      std::copysign(1.0, reduced_sum);  // note that the sign of the reduced sum also directly gives
                                        // the sign of the argument \f$ \mathrm{sign} y \f$

  // We now have to verify problematic logarithms of the reduced sum; it could be \f$
  // \log(\left| S
  // \right|) = - \infty \f$, which implies a zero argument \f$ S = 0 \implies y = 0\f$.
  // Alternatively, it could overflow $\log(\left| S \right|) = \infty$. In both cases, we need to
  // return early with the specific evaluation status.
  const double log_abs_reduced_sum = std::log(std::abs(reduced_sum));
  if (not std::isfinite(log_abs_reduced_sum))
  {
    if (std::copysign(1.0, log_abs_reduced_sum) == -1.0)  // \f$ -\infty \f$: zero argument
    {
      return {
          .log_abs = -std::numeric_limits<double>::infinity(),
          .sign = 1.0,  // we define 0.0 to have a positive sign
          .eval_status = SignedLogSumExpStatus::zero_argument,
      };
    }
    else
    {
      return {
          .log_abs = std::numeric_limits<double>::infinity(),  // genuine overflow
          .sign = sign_reduced_sum,
          .eval_status = SignedLogSumExpStatus::overflow,
      };
    }
  }

  // compute \f$ \log y = M + \log(\left| S \right|)  \f$ and perform a similar verification as
  // above
  const double log_y = max_value + log_abs_reduced_sum;
  FOUR_C_ASSERT(std::isfinite(log_y),
      "This should be unreachable for the finite inputs max_value {} and log_abs_reduced_sum {}",
      max_value, log_abs_reduced_sum);

  return {
      .log_abs = log_y, .sign = sign_reduced_sum, .eval_status = SignedLogSumExpStatus::success};
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::Utils::SignedLogSumExpResult Core::Utils::signed_log_sum_exp(
    std::initializer_list<double> log_values, std::initializer_list<double> prefactors)
{
  return Core::Utils::signed_log_sum_exp(
      std::span<const double>(log_values.begin(), log_values.end()),
      std::span<const double>(prefactors.begin(), prefactors.end()));
}


FOUR_C_NAMESPACE_CLOSE

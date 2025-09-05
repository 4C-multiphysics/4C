// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_REDUCED_LUNG_AAA_APPROX_HPP
#define FOUR_C_REDUCED_LUNG_AAA_APPROX_HPP

#include "4C_config.hpp"

#include "4C_linalg_serialdensematrix.hpp"

#include <functional>

FOUR_C_NAMESPACE_OPEN

namespace ReducedLung
{
  /**
   * @brief Options for the AAA (Adaptive Antoulas-Anderson) rational approximation.
   *
   * Controls the stopping tolerance and the maximum number of support points
   * used by the greedy AAA iteration.
   */
  struct AAAOptions
  {
    /// Relative max-norm error tolerance on the sample set.
    double tol = 1e-13;

    /// Maximum number of iterations / support points.
    int mmax = 100;
  };

  /**
   * @brief Result of an AAA approximation in barycentric form.
   *
   * Holds the support points, function values, and barycentric weights that
   * define the rational approximant, plus a per-iteration error history.
   * Utility methods allow evaluation, pole/zero extraction, and gain.
   *
   * Reference: Nakatsukasa, Sete, Trefethen (SIAM SISC 2018).
   */
  struct AAAResult
  {
    /// Greedy-selected support points (size m).
    std::vector<double> z;

    /// Function values F(z_j) at the support points (size m).
    std::vector<double> f;

    /// Barycentric weights (size m).
    std::vector<double> w;

    /// Max-norm error per AAA iteration on the sample grid.
    std::vector<double> errvec;

    /**
     * @brief Evaluate the rational approximant at given points.
     * Exact support-point queries return the stored f_j.
     */
    std::vector<double> operator()(const std::vector<double>& zz) const;

    /**
     * @brief Compute the finite poles (roots of the denominator).
     * Returns complex poles; infinite eigenvalues are omitted.
     */
    std::vector<std::complex<double>> compute_poles() const;

    /**
     * @brief Compute the finite zeros (roots of the numerator).
     * Returns complex zeros; infinite eigenvalues are omitted.
     */
    std::vector<std::complex<double>> compute_zeros() const;

    /**
     * @brief Gain (value approached as |z|->infinity), equal to
     *        sum_j w_j f_j / sum_j w_j.
     */
    double compute_gain() const;
  };

  /**
   * @brief Compute residues at given poles from barycentric data (z,f,w).
   *
   * Uses a stable formula based on sums (no large products). For well-separated
   * data and moderate degree, double precision is typically sufficient.
   *
   * @param poles   Complex poles at which residues are requested.
   * @param z       Support points (real).
   * @param f       Function values at support points (real).
   * @param w       Barycentric weights (real).
   * @param tiny_abs Absolute fallback threshold for declaring the denominator
   *                 derivative effectively zero (0 = derive a relative threshold).
   * @param eps_rel  Relative proximity threshold to treat a pole as colliding
   *                 with a support point.
   * @return Residues in the same order as @p poles.
   *
   */
  std::vector<std::complex<double>> compute_residues(const std::vector<std::complex<double>>& poles,
      const std::vector<double>& z, const std::vector<double>& f, const std::vector<double>& w,
      const double tiny_abs = 0.0, const double eps_rel = 1e-14);

  using EvaluationFunction = std::function<std::vector<double>(const std::vector<double>& x)>;

  /**
   * @brief Adaptive Antoulas-Anderson (AAA) rational approximation on real samples.
   *
   * Given sample points Z and a target function, constructs a rational
   * approximant in barycentric form by greedily selecting support points and
   * solving a small SVD at each step.
   *
   * @param Z Real sample grid.
   * @param target_function Callable mapping Z to F(Z).
   * @param opts Tolerance and maximum support count.
   * @return AAAResult with (z,f,w) and an error history.
   *
   * Reference:
   *   Y. Nakatsukasa, O. Sete, L. N. Trefethen,
   *   "The AAA Algorithm for Rational Approximation,"
   *   SIAM Journal on Scientific Computing, 40(3), A1494-A1522, 2018.
   */
  AAAResult aaa(const std::vector<double>& Z, const EvaluationFunction& target_function,
      AAAOptions opts = {});
}  // namespace ReducedLung


FOUR_C_NAMESPACE_CLOSE

#endif
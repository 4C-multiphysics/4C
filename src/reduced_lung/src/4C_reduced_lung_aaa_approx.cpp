// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_reduced_lung_aaa_approx.hpp"

#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_utils_densematrix_eigen.hpp"
#include "4C_linalg_utils_densematrix_svd.hpp"

#include <complex>
#include <numeric>

FOUR_C_NAMESPACE_OPEN

namespace ReducedLung
{

  std::vector<double> AAAResult::operator()(const std::vector<double>& zz) const
  {
    const int m = z.size();
    std::vector<double> r(zz.size());

    for (size_t i = 0; i < zz.size(); ++i)
    {
      double num = 0.0;
      double den = 0.0;
      bool exact_match = false;

      for (int j = 0; j < m; ++j)
      {
        if (zz[i] == z[j])
        {
          r[i] = f[j];
          exact_match = true;
          break;
        }
        const auto c = 1.0 / (zz[i] - z[j]);
        num += c * w[j] * f[j];
        den += c * w[j];
      }

      if (!exact_match)
      {
        r[i] = num / den;
      }
    }
    return r;
  }

  std::vector<std::complex<double>> AAAResult::compute_poles() const
  {
    const int m = static_cast<int>(w.size()) + 1;
    Core::LinAlg::SerialDenseMatrix A(m, m), B(m, m);

    // Generalized eigenvalue problem in arrowhead form. A(0, 0) and B(0, 0) are 0.
    for (int i = 1; i < m; ++i)
    {
      A(0, i) = w[i - 1];
      A(i, 0) = 1.0;
      A(i, i) = z[i - 1];
      B(i, i) = 1.0;
    }

    return Core::LinAlg::generalized_eigen(A, B);
  }

  std::vector<std::complex<double>> AAAResult::compute_zeros() const
  {
    const int m = static_cast<int>(w.size()) + 1;
    Core::LinAlg::SerialDenseMatrix A(m, m, true), B(m, m, true);

    // Generalized eigenvalue problem in arrowhead form. A(0, 0) and B(0, 0) are 0.
    for (int i = 1; i < m; ++i)
    {
      A(0, i) = w[i - 1] * f[i - 1];
      A(i, 0) = 1.0;
      A(i, i) = z[i - 1];
      B(i, i) = 1.0;
    }

    return Core::LinAlg::generalized_eigen(A, B);
  }

  double AAAResult::compute_gain() const
  {
    double num = 0.0;
    double den = 0.0;
    for (size_t i = 0; i < w.size(); ++i)
    {
      num += w[i] * f[i];
      den += w[i];
    }
    return num / den;
  }

  std::vector<std::complex<double>> compute_residues(const std::vector<std::complex<double>>& poles,
      const std::vector<double>& z, const std::vector<double>& f, const std::vector<double>& w,
      const double tiny_abs, const double eps_rel)
  {
    if (z.size() != f.size() || z.size() != w.size())
      FOUR_C_THROW("In compute_barycentric_residues: z,f,w size mismatch.");

    std::vector<std::complex<double>> R(poles.size());
    if (poles.empty()) return R;

    // stable Kahan summation
    auto kahan_add =
        [](const std::complex<double>& x, std::complex<double>& sum, std::complex<double>& c)
    {
      const std::complex<double> y = x - c;
      const std::complex<double> t = sum + y;
      c = (t - sum) - y;
      sum = t;
    };

    for (std::size_t k = 0; k < poles.size(); ++k)
    {
      const std::complex<double> pk = poles[k];

      std::complex<double> sN(0.0, 0.0), cN(0.0, 0.0);    // N(pk)
      std::complex<double> sD2(0.0, 0.0), cD2(0.0, 0.0);  // sum w_j / (pk - z_j)^2
      double scaleD2 = 0.0;  // sum |w_j| / |pk - z_j|^2 (for relative tiny)

      for (std::size_t j = 0; j < z.size(); ++j)
      {
        if (w[j] == 0.0) continue;

        const std::complex<double> zj(z[j], 0.0);
        const std::complex<double> diff = pk - zj;
        const double diff_abs = std::abs(diff);

        // near-collision guard (pk almost equals some support point)
        const double rel_thresh = eps_rel * (std::abs(pk) + std::abs(zj) + 1.0);
        if (diff_abs <= rel_thresh)
        {
          FOUR_C_THROW(
              "In compute_barycentric_residues: pole nearly coincides with support point.");
        }

        const std::complex<double> inv = 1.0 / diff;
        const std::complex<double> inv2 = inv * inv;

        kahan_add((w[j] * f[j]) * inv, sN, cN);
        kahan_add(w[j] * inv2, sD2, cD2);

        scaleD2 += std::abs(w[j]) / (diff_abs * diff_abs);
      }

      // Form threshold
      double thresh = tiny_abs;
      if (thresh == 0.0)
      {
        // Relative threshold: a few epsilons times the natural scale of the sum
        thresh = 1e-12 * std::max(1.0, scaleD2);
      }

      if (std::abs(sD2) <= thresh)
      {
        FOUR_C_THROW(
            "In compute_barycentric_residues: D'(pk) ~ 0 (multiple pole / near-collision).");
      }
      else
      {
        // Residue = N(pk) / d/dz D(pk); d/dz D(pk) = -sum w_j / (pk - z_j)^2 = -sD2
        R[k] = -sN / sD2;
      }
    }

    return R;
  }

  AAAResult aaa(
      const std::vector<double>& Z, const EvaluationFunction& target_function, AAAOptions opts)
  {
    int M = Z.size();

    // Evaluate target function at support points
    const std::vector<double> F = target_function(Z);

    // J = candidate indices of support points
    std::vector<int> J(M);
    std::iota(J.begin(), J.end(), 0);

    // Output vectors of barycentric approximation
    std::vector<double> z, f, w, errvec;
    // Cauchy matrix (iteratively filled)
    Core::LinAlg::SerialDenseMatrix C(M, opts.mmax, true);

    // R = initial approximant = mean(F)
    std::vector<double> R(M);
    const double f_mean = std::accumulate(F.begin(), F.end(), 0.0) / M;
    const double f_max = *std::ranges::max_element(F);
    std::ranges::fill(R, f_mean);

    // Main loop with greedy selection of support points
    for (int m = 0; m < opts.mmax; ++m)
    {
      int j = -1;
      double max_err = 0.0;
      // Loop over all remaining support point indices
      for (int i : J)
      {
        // Calculate maximal approximation error
        double diff = std::abs(F[i] - R[i]);
        if (diff >= max_err)
        {
          max_err = diff;
          j = i;
        }
      }

      // Add support point with maximum error
      z.push_back(Z[j]);
      f.push_back(F[j]);

      // Remove j from J
      std::erase(J, j);

      // update C(:,m)
      for (int i = 0; i < M; ++i)
      {
        if (i == j)
        {
          C(i, m) = 0.0;  // unused row, avoid division by zero
        }
        else
        {
          C(i, m) = 1.0 / (Z[i] - Z[j]);
        }
      }

      // build Loewner matrix A(J,:)
      int n = static_cast<int>(J.size());
      Core::LinAlg::SerialDenseMatrix A(n, m + 1);
      for (int row = 0; row < n; ++row)
      {
        int i = J[row];
        for (int col = 0; col <= m; ++col) A(row, col) = F[i] * C(i, col) - C(i, col) * f[col];
      }

      // SVD of A
      Core::LinAlg::SerialDenseMatrix U(n, n, true);
      Core::LinAlg::SerialDenseMatrix S(n, m + 1, true);
      Core::LinAlg::SerialDenseMatrix Vt(m + 1, m + 1, true);
      Core::LinAlg::svd(A, U, S, Vt);

      // w = last column of V
      w.resize(m + 1);
      for (int k = 0; k <= m; ++k) w[k] = Vt(m, k);

      // Compute new approximation R(J) = N(J)/D(J)
      for (int idx : J)
      {
        double num = 0.0, den = 0.0;
        for (int k = 0; k <= m; ++k)
        {
          num += C(idx, k) * w[k] * f[k];
          den += C(idx, k) * w[k];
        }
        R[idx] = num / den;
      }
      R[j] = F[j];

      // Calculate new approximation error
      double err = 0.0;
      for (int i = 0; i < M; ++i) err = std::max(err, std::abs((F[i] - R[i]) / f_max));
      errvec.push_back(err);
      if (err <= opts.tol) break;
    }

    AAAResult result;
    result.z = std::move(z);
    result.f = std::move(f);
    result.w = std::move(w);
    result.errvec = std::move(errvec);

    return result;
  }
}  // namespace ReducedLung

FOUR_C_NAMESPACE_CLOSE

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_reduced_lung_aaa_approx.hpp"

FOUR_C_NAMESPACE_OPEN

namespace
{

  // Helper: compute max abs error between two vectors
  double max_error(const std::vector<double>& a, const std::vector<double>& b)
  {
    assert(a.size() == b.size());
    double err = 0.0;
    for (size_t i = 0; i < a.size(); ++i) err = std::max(err, abs(a[i] - b[i]));
    return err;
  }

  TEST(AAATests, ApproximatesExp)
  {
    // Sample points
    constexpr int M = 200;
    std::vector<double> Z(M);
    for (int i = 0; i < M; ++i) Z[i] = 6.0 * i / (M - 1);

    // True function
    auto F = [](const std::vector<double>& x)
    {
      std::vector<double> y(x.size());
      for (size_t i = 0; i < x.size(); ++i) y[i] = exp(x[i]);
      return y;
    };

    const auto result = ReducedLung::aaa(Z, F);

    // Evaluate approximation on a finer grid
    std::vector<double> z_test(400);
    for (int i = 0; i < 400; ++i) z_test[i] = 6.0 * i / 399.0;

    const auto f_test = F(z_test);
    const auto r_test = result(z_test);

    const double err = max_error(f_test, r_test);
    EXPECT_LT(err, 1e-6) << "AAA should approximate exp(x) on [0,6] with small error.";
  }

  TEST(AAATests, ExactOnSupportPoints)
  {
    constexpr int M = 200;
    std::vector<double> Z(M);
    for (int i = 0; i < M; ++i) Z[i] = 4 * std::numbers::pi / M * i;

    auto F = [](const std::vector<double>& x)
    {
      std::vector<double> y(x.size());
      for (size_t i = 0; i < x.size(); ++i) y[i] = sin(x[i]);
      return y;
    };

    const auto result = ReducedLung::aaa(Z, F);

    // Check interpolation: r(z_k) == f_k
    const auto val = result(result.z);
    for (size_t k = 0; k < result.z.size(); ++k)
    {
      EXPECT_NEAR(abs(val[k] - result.f[k]), 0.0, 1e-14)
          << "Rational approximant should interpolate at support points.";
    }
  }

  TEST(AAATests, HandlesConstantFunction)
  {
    constexpr int M = 50;
    std::vector<double> Z(M);
    for (int i = 0; i < M; ++i) Z[i] = i;

    auto F = [](const std::vector<double>& x)
    {
      std::vector<double> y(x.size(), 42.0);
      return y;
    };

    const auto result = ReducedLung::aaa(Z, F);

    const std::vector<double> z_test = {0.0, 1.0, 10.0, 49.0};
    const auto r_test = result(z_test);

    for (const auto val : r_test)
      EXPECT_NEAR(abs(val - 42.0), 0.0, 1e-14)
          << "AAA should reproduce a constant function exactly.";
  }

  TEST(AAATests, ErrorDecreasesWithMmax)
  {
    // Try approximating 1/(1+sqrt(x)) on [0,10] with different mmax
    constexpr int M = 100;
    std::vector<double> Z(M);
    for (int i = 0; i < M; ++i) Z[i] = 10.0 * i / (M - 1);

    auto F = [](const std::vector<double>& x)
    {
      std::vector<double> y(x.size());
      for (size_t i = 0; i < x.size(); ++i) y[i] = 1.0 / (1.0 + std::sqrt(x[i]));
      return y;
    };

    const auto result_small = ReducedLung::aaa(Z, F, {.tol = 1e-13, .mmax = 3});
    const auto result_large = ReducedLung::aaa(Z, F, {.tol = 1e-13, .mmax = 10});

    const std::vector<double> z_test = {2.0, 5.0, 9.0};
    const auto f_test = F(z_test);
    const auto r_small = result_small(z_test);
    const auto r_large = result_large(z_test);

    const double err_small = max_error(f_test, r_small);
    const double err_large = max_error(f_test, r_large);

    EXPECT_LT(err_large, err_small) << "Increasing mmax should reduce approximation error.";
  }

  void sort_by_real(std::vector<std::complex<double>>& v)
  {
    std::ranges::sort(v, [](const std::complex<double>& a, const std::complex<double>& b)
        { return a.real() < b.real(); });
  }

  TEST(AAA_ResultProcessing, PolesZerosTwoTermClosedForm)
  {
    ReducedLung::AAAResult R;
    R.z = {0.0, 2.0};  // size m-1 = 2  -> pencil size m = 3
    R.w = {1.0, 2.0};  // two terms => exactly one finite pole
    R.f = {42.0, -5.0};

    const double w0 = R.w[0], w1 = R.w[1], z0 = R.z[0], z1 = R.z[1], f0 = R.f[0], f1 = R.f[1];
    const double pole_expected = (w0 * z1 + w1 * z0) / (w0 + w1);
    const double zero_expected = (w0 * f0 * z1 + w1 * f1 * z0) / (w0 * f0 + w1 * f1);

    const auto poles = R.compute_poles();
    const auto zeros = R.compute_zeros();

    ASSERT_EQ(poles.size(), 1u) << "Two-term equation has one finite root.";
    ASSERT_EQ(zeros.size(), 1u) << "Two-term equation has one finite root.";
    EXPECT_NEAR(poles[0].real(), pole_expected, 1e-12);
    EXPECT_NEAR(poles[0].imag(), 0.0, 1e-14);
    EXPECT_NEAR(zeros[0].real(), zero_expected, 1e-12);
    EXPECT_NEAR(zeros[0].imag(), 0.0, 1e-14);
  }

  TEST(AAA_ResultProcessing, PolesThreeTermQuadraticClosedForm)
  {
    ReducedLung::AAAResult R;
    R.z = {-1.0, 2.0, 5.0};
    R.w = {1.0, 2.0, 3.0};
    R.f = {4.0, -1.0, 2.0};  // unused for poles

    const double z0 = R.z[0], z1 = R.z[1], z2 = R.z[2];
    const double w0 = R.w[0], w1 = R.w[1], w2 = R.w[2];

    const double A = (w0 + w1 + w2);
    const double B = -(w0 * (z1 + z2) + w1 * (z0 + z2) + w2 * (z0 + z1));
    const double C = (w0 * z1 * z2 + w1 * z0 * z2 + w2 * z0 * z1);

    const double disc = B * B - 4 * A * C;
    ASSERT_GT(disc, 0.0);
    const double r1 = (-B + std::sqrt(disc)) / (2 * A);
    const double r2 = (-B - std::sqrt(disc)) / (2 * A);

    const auto poles = R.compute_poles();
    ASSERT_EQ(poles.size(), 2u);

    std::vector<std::complex<double>> got = poles, exp = {std::complex<double>(r1, 0),
                                                       std::complex<double>(r2, 0)};
    sort_by_real(got);
    sort_by_real(exp);
    for (int i = 0; i < 2; ++i)
    {
      EXPECT_NEAR(got[i].real(), exp[i].real(), 1e-9);
      EXPECT_NEAR(got[i].imag(), 0.0, 1e-12);
    }
  }

  TEST(AAA_ResultProcessing, ZerosThreeTermQuadraticClosedForm)
  {
    ReducedLung::AAAResult R;
    R.z = {-2.0, 1.0, 4.0};
    R.w = {3.0, 2.0, 1.0};
    R.f = {5.0, -1.0, 2.0};

    const double z0 = R.z[0], z1 = R.z[1], z2 = R.z[2];
    const double a = R.w[0] * R.f[0];
    const double b = R.w[1] * R.f[1];
    const double c = R.w[2] * R.f[2];

    const double A = (a + b + c);
    const double B = -(a * (z1 + z2) + b * (z0 + z2) + c * (z0 + z1));
    const double C = (a * z1 * z2 + b * z0 * z2 + c * z0 * z1);

    const double disc = B * B - 4 * A * C;
    ASSERT_GT(disc, 0.0);
    const double r1 = (-B + std::sqrt(disc)) / (2 * A);
    const double r2 = (-B - std::sqrt(disc)) / (2 * A);

    const auto zeros = R.compute_zeros();
    ASSERT_EQ(zeros.size(), 2u);

    std::vector<std::complex<double>> got = zeros, exp = {std::complex<double>(r1, 0),
                                                       std::complex<double>(r2, 0)};
    sort_by_real(got);
    sort_by_real(exp);
    for (int i = 0; i < 2; ++i)
    {
      EXPECT_NEAR(got[i].real(), exp[i].real(), 1e-9);
      EXPECT_NEAR(got[i].imag(), 0.0, 1e-12);
    }
  }

  TEST(AAA_ResultProcessing, PolesCount)
  {
    ReducedLung::AAAResult R;
    R.z = {-3.0, -1.0, 2.0, 6.0};  // m-1 = 4  -> pencil size m = 5
    R.w = {1.0, 4.0, -2.0, 0.5};
    R.f = {1.0, 1.0, 1.0, 1.0};

    const auto poles = R.compute_poles();
    EXPECT_EQ(poles.size(), R.z.size() - 1);  // m-1 finite poles
  }

  TEST(AAA_ResultProcessing, ZerosCount)
  {
    ReducedLung::AAAResult R;
    R.z = {-3.0, -1.0, 2.0, 6.0};
    R.w = {1.0, 4.0, -2.0, 0.5};
    R.f = {2.0, -1.0, 0.5, 3.0};

    const auto zeros = R.compute_zeros();
    EXPECT_EQ(zeros.size(), R.z.size() - 1);  // m-1 finite zeros
  }

  TEST(AAA_ResultProcessing, PolesInvariantUnderScaling)
  {
    ReducedLung::AAAResult R;
    R.z = {0.0, 1.0, 3.0};
    R.w = {2.0, -1.0, 4.0};
    R.f = {7.0, 8.0, 9.0};

    auto poles1 = R.compute_poles();
    constexpr double alpha = 7.0;
    for (auto& wi : R.w) wi *= alpha;

    auto poles2 = R.compute_poles();

    ASSERT_EQ(poles1.size(), poles2.size());
    sort_by_real(poles1);
    sort_by_real(poles2);
    for (size_t i = 0; i < poles1.size(); ++i)
    {
      EXPECT_NEAR(poles1[i].real(), poles2[i].real(), 1e-12);
      EXPECT_NEAR(poles1[i].imag(), poles2[i].imag(), 1e-12);
    }
  }

  TEST(AAA_ResultProcessing, ZerosInvariantUnderScaling)
  {
    ReducedLung::AAAResult R;
    R.z = {-1.0, 2.0, 5.0};
    R.w = {3.0, 1.0, 4.0};
    R.f = {2.0, 5.0, -1.0};

    auto zeros1 = R.compute_zeros();

    constexpr double alpha = 10.0;
    for (size_t i = 0; i < R.w.size(); ++i) R.w[i] *= alpha;

    auto zeros2 = R.compute_zeros();

    ASSERT_EQ(zeros1.size(), zeros2.size());
    sort_by_real(zeros1);
    sort_by_real(zeros2);
    for (size_t i = 0; i < zeros1.size(); ++i)
    {
      EXPECT_NEAR(zeros1[i].real(), zeros2[i].real(), 1e-12);
      EXPECT_NEAR(zeros1[i].imag(), zeros2[i].imag(), 1e-12);
    }
  }

  std::complex<double> bary_eval(const std::complex<double>& zq, const std::vector<double>& z,
      const std::vector<double>& f, const std::vector<double>& w)
  {
    for (std::size_t j = 0; j < z.size(); ++j)
    {
      if (zq.real() == z[j] && zq.imag() == 0.0)
      {
        return std::complex<double>(f[j], 0.0);
      }
    }
    std::complex<double> num(0.0, 0.0), den(0.0, 0.0);
    for (std::size_t j = 0; j < z.size(); ++j)
    {
      std::complex<double> diff = zq - std::complex<double>(z[j], 0.0);
      std::complex<double> inv = std::complex<double>(1.0, 0.0) / diff;
      num += (w[j] * f[j]) * inv;
      den += (w[j]) * inv;
    }
    return num / den;
  }

  // ----- Helper: numerical residue via limit h*r(pk+h) with Richardson refinement
  std::complex<double> numeric_residue(const std::complex<double>& pk, const std::vector<double>& z,
      const std::vector<double>& f, const std::vector<double>& w)
  {
    // complex step directions to avoid directional bias
    const std::complex<double> i(0.0, 1.0);
    constexpr double base = 1e-6;  // step scale (safe for typical data)
    // two step sizes and Richardson extrapolation
    auto est = [&](double hscale)
    {
      const std::complex<double> h = hscale * (1.0 + i);  // diagonal complex step
      return h * bary_eval(pk + h, z, f, w);
    };
    const std::complex<double> R1 = est(base);
    const std::complex<double> R2 = est(base / 10.0);
    return (10.0 * R2 - R1) / 9.0;
  }

  TEST(AAA_ResultProcessing, TwoSupport_PoleAndResidue_MatchNumericalLimit)
  {
    const std::vector<double> z = {0.0, 4.0};
    const std::vector<double> f = {3.0, -1.0};
    const std::vector<double> w = {2.0, 5.0};

    const double w0 = w[0], w1 = w[1], z0 = z[0], z1 = z[1];
    const double lambda = (w0 * z1 + w1 * z0) / (w0 + w1);  // unique finite pole
    const std::vector<std::complex<double>> poles = {std::complex<double>(lambda, 0.0)};

    const auto R = ReducedLung::compute_residues(poles, z, f, w);

    const std::complex<double> Rnum = numeric_residue(poles[0], z, f, w);

    ASSERT_EQ(R.size(), 1u);
    EXPECT_NEAR(R[0].real(), Rnum.real(), 1e-8);
    EXPECT_NEAR(R[0].imag(), Rnum.imag(), 1e-8);
  }

  TEST(AAA_ResultProcessing, ThreeSupport_TwoPoles_ResiduesMatchNumericalLimit)
  {
    const std::vector<double> z = {-1.0, 2.0, 5.0};
    const std::vector<double> f = {4.0, -1.0, 2.0};
    const std::vector<double> w = {1.0, 2.0, 3.0};

    const double z0 = z[0], z1 = z[1], z2 = z[2];
    const double w0 = w[0], w1 = w[1], w2 = w[2];

    const double A = (w0 + w1 + w2);
    const double B = -(w0 * (z1 + z2) + w1 * (z0 + z2) + w2 * (z0 + z1));
    const double Cc = (w0 * z1 * z2 + w1 * z0 * z2 + w2 * z0 * z1);

    const double disc = B * B - 4 * A * Cc;
    ASSERT_GT(disc, 0.0);
    const double r1 = (-B + std::sqrt(disc)) / (2 * A);
    const double r2 = (-B - std::sqrt(disc)) / (2 * A);
    const std::vector<std::complex<double>> poles = {
        std::complex<double>(r1, 0.0), std::complex<double>(r2, 0.0)};

    const auto R = ReducedLung::compute_residues(poles, z, f, w);
    ASSERT_EQ(R.size(), 2u);

    const std::complex<double> Rnum1 = numeric_residue(poles[0], z, f, w);
    const std::complex<double> Rnum2 = numeric_residue(poles[1], z, f, w);
    EXPECT_NEAR(R[0].real(), Rnum1.real(), 1e-8);
    EXPECT_NEAR(R[0].imag(), Rnum1.imag(), 1e-8);
    EXPECT_NEAR(R[1].real(), Rnum2.real(), 1e-8);
    EXPECT_NEAR(R[1].imag(), Rnum2.imag(), 1e-8);
  }

  TEST(AAA_ResultProcessing, ScalingFScalesResidues)
  {
    const std::vector<double> z = {0.0, 3.0, 6.0};
    const std::vector<double> f = {2.0, -1.0, 4.0};
    const std::vector<double> w = {1.0, 2.0, 3.0};

    // Poles (roots of D): compute via quadratic as above
    const double z0 = z[0], z1 = z[1], z2 = z[2];
    const double w0 = w[0], w1 = w[1], w2 = w[2];
    const double A = (w0 + w1 + w2);
    const double B = -(w0 * (z1 + z2) + w1 * (z0 + z2) + w2 * (z0 + z1));
    const double Cc = (w0 * z1 * z2 + w1 * z0 * z2 + w2 * z0 * z1);
    const double disc = B * B - 4 * A * Cc;
    ASSERT_GT(disc, 0.0);
    const std::vector<std::complex<double>> poles = {
        std::complex<double>((-B + std::sqrt(disc)) / (2 * A), 0.0),
        std::complex<double>((-B - std::sqrt(disc)) / (2 * A), 0.0)};

    const auto R1 = ReducedLung::compute_residues(poles, z, f, w);

    constexpr double alpha = 7.5;
    std::vector<double> f2 = f;
    for (auto& v : f2) v *= alpha;

    const auto R2 = ReducedLung::compute_residues(poles, z, f2, w);

    ASSERT_EQ(R1.size(), R2.size());
    for (std::size_t k = 0; k < R1.size(); ++k)
    {
      EXPECT_NEAR(R2[k].real(), alpha * R1[k].real(), 1e-12);
      EXPECT_NEAR(R2[k].imag(), alpha * R1[k].imag(), 1e-12);
    }
  }

  TEST(AAA_ResultProcessing, PermuteSupport_Invariance)
  {
    const std::vector<double> z = {-2.0, 1.0, 4.0};
    const std::vector<double> f = {3.0, 2.0, -1.0};
    const std::vector<double> w = {2.0, 5.0, 1.0};

    // Poles from D: compute once
    const double z0 = z[0], z1 = z[1], z2 = z[2];
    const double w0 = w[0], w1 = w[1], w2 = w[2];
    const double A = (w0 + w1 + w2);
    const double B = -(w0 * (z1 + z2) + w1 * (z0 + z2) + w2 * (z0 + z1));
    const double Cc = (w0 * z1 * z2 + w1 * z0 * z2 + w2 * z0 * z1);
    const double disc = B * B - 4 * A * Cc;
    ASSERT_GT(disc, 0.0);
    const std::vector<std::complex<double>> poles = {
        std::complex<double>((-B + std::sqrt(disc)) / (2 * A), 0.0),
        std::complex<double>((-B - std::sqrt(disc)) / (2 * A), 0.0)};

    const auto R1 = ReducedLung::compute_residues(poles, z, f, w);

    // Permute (z,f,w) consistently
    const std::vector<int> perm = {2, 0, 1};
    std::vector<double> zP(3), fP(3), wP(3);
    for (int i = 0; i < 3; ++i)
    {
      zP[i] = z[perm[i]];
      fP[i] = f[perm[i]];
      wP[i] = w[perm[i]];
    }

    const auto R2 = ReducedLung::compute_residues(poles, zP, fP, wP);

    ASSERT_EQ(R1.size(), R2.size());
    for (std::size_t k = 0; k < R1.size(); ++k)
    {
      EXPECT_NEAR(R1[k].real(), R2[k].real(), 1e-12);
      EXPECT_NEAR(R1[k].imag(), R2[k].imag(), 1e-12);
    }
  }

}  // namespace

FOUR_C_NAMESPACE_CLOSE
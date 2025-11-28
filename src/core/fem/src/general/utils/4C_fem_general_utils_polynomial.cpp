// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_general_utils_polynomial.hpp"

#include "4C_fem_general_utils_integration.hpp"
#include "4C_utils_singleton_owner.hpp"

#include <Intrepid2_PointTools.hpp>
#include <Kokkos_DynRankView.hpp>
#include <Shards_CellTopology.hpp>

FOUR_C_NAMESPACE_OPEN

namespace
{
  inline Core::FE::GaussRule1D get_1d_lagrange_basis_lobatto_gauss_rule(const unsigned int degree)
  {
    switch (degree)
    {
      case 1:
        return Core::FE::GaussRule1D::line_lobatto2point;
      case 2:
        return Core::FE::GaussRule1D::line_lobatto3point;
      case 3:
        return Core::FE::GaussRule1D::line_lobatto4point;
      case 4:
        return Core::FE::GaussRule1D::line_lobatto5point;
      case 5:
        return Core::FE::GaussRule1D::line_lobatto6point;
      case 6:
        return Core::FE::GaussRule1D::line_lobatto7point;
      case 7:
        return Core::FE::GaussRule1D::line_lobatto8point;
      case 8:
        return Core::FE::GaussRule1D::line_lobatto9point;
      case 9:
        return Core::FE::GaussRule1D::line_lobatto10point;
      default:
        FOUR_C_THROW("The Lobatto Gauss rule is not implemented for degree {}", degree);
    }
  }
}  // namespace

namespace Core::FE
{
  /*!
   \brief generates complete Lagrange basis in 1D for [-1,1]^d elements
   */
  std::vector<LagrangePolynomial> generate_lagrange_basis1_d(const unsigned int degree)
  {
    std::vector<LagrangePolynomial> poly1d;
    if (degree == 0)
    {
      poly1d.push_back(Core::FE::LagrangePolynomial(std::vector<double>(), 0.));
      return poly1d;
    }

    Core::FE::IntegrationPoints1D gaussLobattoPoints(
        get_1d_lagrange_basis_lobatto_gauss_rule(degree));
    std::vector<double> points(degree);
    for (unsigned int i = 0; i <= degree; ++i)
    {
      for (unsigned int j = 0, c = 0; j <= degree; ++j)
        if (i != j)
        {
          points[c] = gaussLobattoPoints.qxg[j][0];
          ++c;
        }
      poly1d.push_back(Core::FE::LagrangePolynomial(points, gaussLobattoPoints.qxg[i][0]));
    }
    return poly1d;
  }


  /*!
   \brief generates complete Legendre basis in 1D

   Legendre polynomials are orthogonal on the unit interval [-1, 1]. As opposed
   to the usual mathematical definition, we also scale the polynomials such that
   they are actually orthonormal on the unit interval.
   */
  std::vector<Polynomial> generate_legendre_basis1_d(const unsigned int degree)
  {
    // Legendre polynomials are defined recursively by the following scheme:
    // L_0 (x) = 1
    // L_1 (x) = x
    // L_n (x) = (2n-1)/n * x * L_{n-1} (x) - (n-1)/n * L_{n-2}(x), n>=2
    // (This is usually referred to as Bonnet's recursion formula)

    std::vector<Polynomial> polynomials;

    std::vector<double> coefficients(1, 1.);
    polynomials.push_back(Polynomial(coefficients));
    if (degree == 0) return polynomials;

    std::vector<double> coefficients_n1(2, 0.);
    coefficients_n1[1] = 1.;
    for (unsigned int n = 2;; ++n)
    {
      FOUR_C_ASSERT(coefficients_n1.size() == n, "Internal error");
      {
        // Scale coefficients to make polynomials orthonormal
        std::vector<double> scaledCoefficients(coefficients_n1);
        for (unsigned int q = 0; q < scaledCoefficients.size(); ++q)
          scaledCoefficients[q] *= (2. * n - 1.) * 0.5;
        polynomials.push_back(Polynomial(scaledCoefficients));
      }

      if (n > degree) break;

      coefficients.resize(n + 1);
      for (unsigned int i = 0; i < n - 1; ++i) coefficients[i] *= -static_cast<double>(n - 1) / n;
      for (unsigned int i = 0; i < n; ++i)
        coefficients[i + 1] += coefficients_n1[i] * static_cast<double>(2 * n - 1) / n;

      coefficients.swap(coefficients_n1);
    }

    return polynomials;
  }



  /*
   \brief Evaluates the values of the whole polynomial space in the given point
   */
  template <int nsd, class POLY>
  void PolynomialSpaceTensor<nsd, POLY>::evaluate(
      const Core::LinAlg::Matrix<nsd, 1>& point, Core::LinAlg::SerialDenseVector& values) const
  {
    const unsigned int size = poly_space1d_.size();
    FOUR_C_ASSERT(size < 20, "Not implemented");

    // avoid memory allocation by allocating values on the stack
    double evaluation[nsd][20];
    for (unsigned int i = 0; i < size; ++i)
      for (unsigned int d = 0; d < nsd; ++d) evaluation[d][i] = poly_space1d_[i].evaluate(point(d));

    switch (nsd)
    {
      case 3:
        for (unsigned int i = 0; i < size; ++i)
          for (unsigned int j = 0; j < size; ++j)
            for (unsigned int k = 0; k < size; ++k)
              values(renumbering_[i * size * size + j * size + k]) =
                  evaluation[2][i] * evaluation[1][j] * evaluation[0][k];
        break;
      case 2:
        for (unsigned int j = 0; j < size; ++j)
          for (unsigned int k = 0; k < size; ++k)
            values(renumbering_[j * size + k]) = evaluation[1][j] * evaluation[0][k];
        break;
      case 1:
        for (unsigned int k = 0; k < size; ++k) values(renumbering_[k]) = evaluation[0][k];
        break;
      default:
        FOUR_C_THROW("Invalid dimension");
    }
  }


  /*
   \brief Evaluates the first derivative of the whole polynomial space in the given point
   */
  template <int nsd, class POLY>
  void PolynomialSpaceTensor<nsd, POLY>::evaluate_deriv1(
      const Core::LinAlg::Matrix<nsd, 1>& point, Core::LinAlg::SerialDenseMatrix& derivatives) const
  {
    const unsigned int size = poly_space1d_.size();
    FOUR_C_ASSERT(size < 20, "Not implemented");

    // avoid memory allocation by allocating values on the stack
    double evaluation[nsd][20], gradient[nsd][20];
    Core::LinAlg::Matrix<2, 1, double> eval;
    for (unsigned int i = 0; i < size; ++i)
      for (unsigned int d = 0; d < nsd; ++d)
      {
        poly_space1d_[i].evaluate(point(d), eval);
        evaluation[d][i] = eval(0);
        gradient[d][i] = eval(1);
      }

    switch (nsd)
    {
      case 3:
        for (unsigned int i = 0; i < size; ++i)
          for (unsigned int j = 0; j < size; ++j)
            for (unsigned int k = 0; k < size; ++k)
            {
              derivatives(0, renumbering_[i * size * size + j * size + k]) =
                  evaluation[2][i] * evaluation[1][j] * gradient[0][k];
              derivatives(1, renumbering_[i * size * size + j * size + k]) =
                  evaluation[2][i] * gradient[1][j] * evaluation[0][k];
              derivatives(2, renumbering_[i * size * size + j * size + k]) =
                  gradient[2][i] * evaluation[1][j] * evaluation[0][k];
            }
        break;
      case 2:
        for (unsigned int j = 0; j < size; ++j)
          for (unsigned int k = 0; k < size; ++k)
          {
            derivatives(0, renumbering_[j * size + k]) = evaluation[1][j] * gradient[0][k];
            derivatives(1, renumbering_[j * size + k]) = gradient[1][j] * evaluation[0][k];
          }
        break;
      case 1:
        for (unsigned int k = 0; k < size; ++k) derivatives(0, renumbering_[k]) = gradient[0][k];
        break;
      default:
        FOUR_C_THROW("Invalid dimension");
    }
  }



  /*
   \brief Evaluates the first derivative of the whole polynomial space in the given point
   */
  template <int nsd, class POLY>
  void PolynomialSpaceTensor<nsd, POLY>::evaluate_deriv2(
      const Core::LinAlg::Matrix<nsd, 1>& point, Core::LinAlg::SerialDenseMatrix& derivatives) const
  {
    const unsigned int size = poly_space1d_.size();
    FOUR_C_ASSERT(size < 20, "Not implemented");

    // avoid memory allocation by allocating values on the stack
    double evaluation[nsd][20], gradient[nsd][20], hessian[nsd][20];
    Core::LinAlg::Matrix<3, 1, double> eval;
    for (unsigned int i = 0; i < size; ++i)
      for (unsigned int d = 0; d < nsd; ++d)
      {
        poly_space1d_[i].evaluate(point(d), eval);
        evaluation[d][i] = eval(0);
        gradient[d][i] = eval(1);
        hessian[d][i] = eval(2);
      }

    switch (nsd)
    {
      case 3:
        for (unsigned int i = 0; i < size; ++i)
          for (unsigned int j = 0; j < size; ++j)
            for (unsigned int k = 0; k < size; ++k)
            {
              derivatives(0, renumbering_[i * size * size + j * size + k]) =
                  evaluation[2][i] * evaluation[1][j] * hessian[0][k];
              derivatives(1, renumbering_[i * size * size + j * size + k]) =
                  evaluation[2][i] * hessian[1][j] * evaluation[0][k];
              derivatives(2, renumbering_[i * size * size + j * size + k]) =
                  hessian[2][i] * evaluation[1][j] * evaluation[0][k];
              derivatives(3, renumbering_[i * size * size + j * size + k]) =
                  evaluation[2][i] * gradient[1][j] * gradient[0][k];
              derivatives(4, renumbering_[i * size * size + j * size + k]) =
                  gradient[2][i] * evaluation[1][j] * gradient[0][k];
              derivatives(5, renumbering_[i * size * size + j * size + k]) =
                  gradient[2][i] * gradient[1][j] * evaluation[0][k];
            }
        break;
      case 2:
        for (unsigned int j = 0; j < size; ++j)
          for (unsigned int k = 0; k < size; ++k)
          {
            derivatives(0, renumbering_[j * size + k]) = evaluation[1][j] * hessian[0][k];
            derivatives(1, renumbering_[j * size + k]) = hessian[1][j] * evaluation[0][k];
            derivatives(2, renumbering_[j * size + k]) = gradient[1][j] * gradient[0][k];
          }
        break;
      case 1:
        for (unsigned int k = 0; k < size; ++k) derivatives(0, renumbering_[k]) = hessian[0][k];
        break;
      default:
        FOUR_C_THROW("Invalid dimension");
    }
  }



  /*
   \brief Creates an array with coordinates of the nodes supporting the polynomials.
   */
  template <int nsd, class POLY>
  void PolynomialSpaceTensor<nsd, POLY>::fill_unit_node_points(
      Core::LinAlg::SerialDenseMatrix& matrix) const
  {
    matrix.shape(nsd, size());

    const unsigned int size = poly_space1d_.size();
    switch (nsd)
    {
      case 3:
        for (unsigned int i = 0; i < size; ++i)
          for (unsigned int j = 0; j < size; ++j)
            for (unsigned int k = 0; k < size; ++k)
            {
              matrix(0, renumbering_[i * size * size + j * size + k]) =
                  poly_space1d_[k].node_point();
              matrix(1, renumbering_[i * size * size + j * size + k]) =
                  poly_space1d_[j].node_point();
              matrix(2, renumbering_[i * size * size + j * size + k]) =
                  poly_space1d_[i].node_point();
            }
        break;
      case 2:
        for (unsigned int j = 0; j < size; ++j)
          for (unsigned int k = 0; k < size; ++k)
          {
            matrix(0, renumbering_[j * size + k]) = poly_space1d_[k].node_point();
            matrix(1, renumbering_[j * size + k]) = poly_space1d_[j].node_point();
          }
        break;
      case 1:
        for (unsigned int k = 0; k < size; ++k)
          matrix(0, renumbering_[k]) = poly_space1d_[k].node_point();
        break;
      default:
        FOUR_C_THROW("Invalid dimension");
    }
  }



  /*
   \brief Evaluates the values of the whole polynomial space in the given point
   */
  template <int nsd, class POLY>
  void PolynomialSpaceComplete<nsd, POLY>::evaluate(
      const Core::LinAlg::Matrix<nsd, 1>& point, Core::LinAlg::SerialDenseVector& values) const
  {
    const unsigned int size = poly_space1d_.size();
    FOUR_C_ASSERT(size < 20, "Not implemented");

    // avoid memory allocation by allocating values on the stack
    double evaluation[nsd][20];
    for (unsigned int i = 0; i < size; ++i)
      for (unsigned int d = 0; d < nsd; ++d) evaluation[d][i] = poly_space1d_[i].evaluate(point(d));

    unsigned int c = 0;
    switch (nsd)
    {
      case 3:
        for (unsigned int i = 0; i < size; ++i)
          for (unsigned int j = 0; j < size - i; ++j)
            for (unsigned int k = 0; k < size - i - j; ++k, ++c)
              values(renumbering_[c]) = evaluation[2][i] * evaluation[1][j] * evaluation[0][k];
        break;
      case 2:
        for (unsigned int j = 0; j < size; ++j)
          for (unsigned int k = 0; k < size - j; ++k, ++c)
            values(renumbering_[c]) = evaluation[1][j] * evaluation[0][k];
        break;
      case 1:
        for (unsigned int k = 0; k < size; ++k, ++c) values(renumbering_[k]) = evaluation[0][k];
        break;
      default:
        FOUR_C_THROW("Invalid dimension");
        break;
    }

    FOUR_C_ASSERT(c == PolynomialSpaceComplete::size(), "Internal error");
  }


  /*
   \brief Evaluates the first derivative of the whole polynomial space in the given point
   */
  template <int nsd, class POLY>
  void PolynomialSpaceComplete<nsd, POLY>::evaluate_deriv1(
      const Core::LinAlg::Matrix<nsd, 1>& point, Core::LinAlg::SerialDenseMatrix& derivatives) const
  {
    const unsigned int size = poly_space1d_.size();
    FOUR_C_ASSERT(size < 20, "Not implemented");

    // avoid memory allocation by allocating values on the stack
    double evaluation[nsd][20], gradient[nsd][20];
    Core::LinAlg::Matrix<2, 1, double> eval;
    for (unsigned int i = 0; i < size; ++i)
      for (unsigned int d = 0; d < nsd; ++d)
      {
        poly_space1d_[i].evaluate(point(d), eval);
        evaluation[d][i] = eval(0);
        gradient[d][i] = eval(1);
      }

    unsigned int c = 0;
    switch (nsd)
    {
      case 3:
        for (unsigned int i = 0; i < size; ++i)
          for (unsigned int j = 0; j < size - i; ++j)
            for (unsigned int k = 0; k < size - i - j; ++k, ++c)
            {
              derivatives(0, renumbering_[c]) =
                  evaluation[2][i] * evaluation[1][j] * gradient[0][k];
              derivatives(1, renumbering_[c]) =
                  evaluation[2][i] * gradient[1][j] * evaluation[0][k];
              derivatives(2, renumbering_[c]) =
                  gradient[2][i] * evaluation[1][j] * evaluation[0][k];
            }
        break;
      case 2:
        for (unsigned int j = 0; j < size; ++j)
          for (unsigned int k = 0; k < size - j; ++k, ++c)
          {
            derivatives(0, renumbering_[c]) = evaluation[1][j] * gradient[0][k];
            derivatives(1, renumbering_[c]) = gradient[1][j] * evaluation[0][k];
          }
        break;
      case 1:
        for (unsigned int k = 0; k < size; ++k, ++c)
          derivatives(0, renumbering_[k]) = gradient[0][k];
        break;
      default:
        FOUR_C_THROW("Invalid dimension");
        break;
    }

    FOUR_C_ASSERT(c == PolynomialSpaceComplete::size(), "Internal error");
  }



  /*
   \brief Evaluates the first derivative of the whole polynomial space in the given point
   */
  template <int nsd, class POLY>
  void PolynomialSpaceComplete<nsd, POLY>::evaluate_deriv2(
      const Core::LinAlg::Matrix<nsd, 1>& point, Core::LinAlg::SerialDenseMatrix& derivatives) const
  {
    const unsigned int size = poly_space1d_.size();
    FOUR_C_ASSERT(size < 20, "Not implemented");

    // avoid memory allocation by allocating values on the stack
    double evaluation[nsd][20], gradient[nsd][20], hessian[nsd][20];
    Core::LinAlg::Matrix<3, 1, double> eval;
    for (unsigned int i = 0; i < size; ++i)
      for (unsigned int d = 0; d < nsd; ++d)
      {
        poly_space1d_[i].evaluate(point(d), eval);
        evaluation[d][i] = eval(0);
        gradient[d][i] = eval(1);
        hessian[d][i] = eval(2);
      }

    unsigned int c = 0;
    switch (nsd)
    {
      case 3:
        for (unsigned int i = 0; i < size; ++i)
          for (unsigned int j = 0; j < size - i; ++j)
            for (unsigned int k = 0; k < size - i - j; ++k, ++c)
            {
              derivatives(0, renumbering_[c]) = evaluation[2][i] * evaluation[1][j] * hessian[0][k];
              derivatives(1, renumbering_[c]) = evaluation[2][i] * hessian[1][j] * evaluation[0][k];
              derivatives(2, renumbering_[c]) = hessian[2][i] * evaluation[1][j] * evaluation[0][k];
              derivatives(3, renumbering_[c]) = evaluation[2][i] * gradient[1][j] * gradient[0][k];
              derivatives(4, renumbering_[c]) = gradient[2][i] * evaluation[1][j] * gradient[0][k];
              derivatives(5, renumbering_[c]) = gradient[2][i] * gradient[1][j] * evaluation[0][k];
            }
        break;
      case 2:
        for (unsigned int j = 0; j < size; ++j)
          for (unsigned int k = 0; k < size - j; ++k, ++c)
          {
            derivatives(0, renumbering_[c]) = evaluation[1][j] * hessian[0][k];
            derivatives(1, renumbering_[c]) = hessian[1][j] * evaluation[0][k];
            derivatives(2, renumbering_[c]) = gradient[1][j] * gradient[0][k];
          }
        break;
      case 1:
        for (unsigned int k = 0; k < size; ++k, ++c)
          derivatives(0, renumbering_[k]) = hessian[0][k];
        break;
      default:
        FOUR_C_THROW("Invalid dimension");
        break;
    }
    FOUR_C_ASSERT(c == PolynomialSpaceComplete::size(), "Internal error");
  }



  /*
   \brief Creates an array with coordinates of the nodes supporting the polynomials.
   */
  template <int nsd, class POLY>
  void PolynomialSpaceComplete<nsd, POLY>::fill_unit_node_points(
      Core::LinAlg::SerialDenseMatrix& matrix) const
  {
    matrix.shape(nsd, size());

    const unsigned int size = poly_space1d_.size();
    unsigned int c = 0;
    switch (nsd)
    {
      case 3:
        for (unsigned int i = 0; i < size; ++i)
          for (unsigned int j = 0; j < size - i; ++j)
            for (unsigned int k = 0; k < size - i - j; ++k, ++c)
            {
              matrix(0, renumbering_[c]) = poly_space1d_[k].node_point();
              matrix(1, renumbering_[c]) = poly_space1d_[j].node_point();
              matrix(2, renumbering_[c]) = poly_space1d_[i].node_point();
            }
        break;
      case 2:
        for (unsigned int j = 0; j < size; ++j)
          for (unsigned int k = 0; k < size - j; ++k, ++c)
          {
            matrix(0, renumbering_[c]) = poly_space1d_[k].node_point();
            matrix(1, renumbering_[c]) = poly_space1d_[j].node_point();
          }
        break;
      case 1:
        for (unsigned int k = 0; k < size; ++k, ++c)
          matrix(0, renumbering_[c]) = poly_space1d_[k].node_point();
        break;
      default:
        FOUR_C_THROW("Invalid dimension");
        break;
    }
    FOUR_C_ASSERT(c == PolynomialSpaceComplete::size(), "Internal error");
  }



  template <int nsd>
  void LagrangeBasisTet<nsd>::evaluate(
      const Core::LinAlg::Matrix<nsd, 1>& point, Core::LinAlg::SerialDenseVector& values) const
  {
    legendre_.evaluate(point, values);
    vandermonde_factor_.setVectors(Teuchos::rcpFromRef(values), Teuchos::rcpFromRef(values));
    vandermonde_factor_.solve();
  }



  template <int nsd>
  void LagrangeBasisTet<nsd>::evaluate_deriv1(
      const Core::LinAlg::Matrix<nsd, 1>& point, Core::LinAlg::SerialDenseMatrix& derivatives) const
  {
    legendre_.evaluate_deriv1(point, derivatives);
    for (unsigned int d = 0; d < nsd; ++d)
    {
      for (unsigned int i = 0; i < size(); ++i) evaluate_vec_(i) = derivatives(d, i);
      vandermonde_factor_.setVectors(
          Teuchos::rcpFromRef(evaluate_vec_), Teuchos::rcpFromRef(evaluate_vec_));
      vandermonde_factor_.solve();
      for (unsigned int i = 0; i < size(); ++i) derivatives(d, i) = evaluate_vec_(i);
    }
  }



  template <int nsd>
  void LagrangeBasisTet<nsd>::evaluate_deriv2(
      const Core::LinAlg::Matrix<nsd, 1>& point, Core::LinAlg::SerialDenseMatrix& derivatives) const
  {
    legendre_.evaluate_deriv2(point, derivatives);
    for (unsigned int d = 0; d < (nsd * (nsd + 1)) / 2; ++d)
    {
      for (unsigned int i = 0; i < size(); ++i) evaluate_vec_(i) = derivatives(d, i);
      vandermonde_factor_.setVectors(
          Teuchos::rcpFromRef(evaluate_vec_), Teuchos::rcpFromRef(evaluate_vec_));
      vandermonde_factor_.solve();
      for (unsigned int i = 0; i < size(); ++i) derivatives(d, i) = evaluate_vec_(i);
    }
  }



  template <>
  void LagrangeBasisTet<2>::fill_fekete_points(const unsigned int degree)
  {
    fekete_points_.shape(2, size(degree));

    if (degree == 0)
    {
      fekete_points_(0, 0) = 1. / 3.;
      fekete_points_(1, 0) = 1. / 3.;
    }
    else if (degree == 1)
    {
      std::array<double, 6> list = {0, 0, 1, 0, 0, 1};
      for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int d = 0; d < 2; ++d) fekete_points_(d, i) = list[i * 2 + d];
    }
    else
    {
      Kokkos::DynRankView<double, Kokkos::HostSpace> wb_points("fekete_points", size(degree), 2);

      //  CellTopologyData
      shards::CellTopology myTri(shards::getCellTopologyData<shards::Triangle<3>>());
      Intrepid2::PointTools::getLattice(
          wb_points, myTri, degree, 0, Intrepid2::POINTTYPE_WARPBLEND);

      for (unsigned int i = 0; i < size(degree); ++i)
        for (int j = 0; j < 2; ++j) fekete_points_(j, i) = wb_points(i, j);
    }
  }



  template <>
  void LagrangeBasisTet<3>::fill_fekete_points(const unsigned int degree)
  {
    fekete_points_.shape(3, size(degree));
    unsigned int c = 0;
    if (degree == 0)
    {
      fekete_points_(0, 0) = 0.25;
      fekete_points_(1, 0) = 0.25;
      fekete_points_(2, 0) = 0.25;
    }
    else if (degree == 1)
    {
      for (unsigned int i = 0; i <= 1; ++i)
        for (unsigned int j = 0; j <= 1 - i; ++j)
          for (unsigned int k = 0; k <= 1 - i - j; ++k, ++c)
          {
            fekete_points_(0, c) = (double)k / degree;
            fekete_points_(1, c) = (double)j / degree;
            fekete_points_(2, c) = (double)i / degree;
          }
    }
    else
    {
      Kokkos::DynRankView<double, Kokkos::HostSpace> wb_points("fekete_points", size(degree), 3);

      //  CellTopologyData
      shards::CellTopology myTet(shards::getCellTopologyData<shards::Tetrahedron<4>>());
      Intrepid2::PointTools::getLattice(
          wb_points, myTet, degree, 0, Intrepid2::POINTTYPE_WARPBLEND);

      for (unsigned int i = 0; i < size(degree); ++i)
        for (int j = 0; j < 3; ++j) fekete_points_(j, i) = wb_points(i, j);
    }
  }



  template <int nsd>
  void Core::FE::LagrangeBasisTet<nsd>::fill_fekete_points(const unsigned int)
  {
    FOUR_C_THROW("Not implemented for dim = {}", nsd);
  }



  template <int nsd>
  void Core::FE::LagrangeBasisTet<nsd>::compute_vandermonde_matrices(const unsigned int degree)
  {
    vandermonde_.shape(size(), size());

    Core::LinAlg::SerialDenseVector values(size());
    Core::LinAlg::SerialDenseMatrix deriv1(nsd, size());
    Core::LinAlg::SerialDenseMatrix deriv2(nsd * (nsd + 1) / 2, size());
    Core::LinAlg::Matrix<nsd, 1> point;
    for (unsigned int i = 0; i < size(); ++i)
    {
      for (unsigned int d = 0; d < nsd; ++d) point(d, 0) = fekete_points_(d, i);

      legendre_.evaluate(point, values);
      for (unsigned int j = 0; j < size(); ++j) vandermonde_(j, i) = values(j);
    }

    vandermonde_factor_.setMatrix(Teuchos::rcpFromRef(vandermonde_.base()));
    vandermonde_factor_.factor();
    evaluate_vec_.shape(size(), 1);


    // Sanity check: Polynomials should be nodal in the Fekete points
#ifdef FOUR_C_ENABLE_ASSERTIONS
    for (unsigned int i = 0; i < size(); ++i)
    {
      for (unsigned int d = 0; d < nsd; ++d) point(d, 0) = fekete_points_(d, i);

      evaluate(point, values);
      for (unsigned int j = 0; j < size(); ++j)
        if (i != j)
          FOUR_C_ASSERT(std::abs(values(j)) < 1e-11,
              "Lagrange polynomial seems to not be nodal, p_j(xi_i) = {}!", values(j));
      FOUR_C_ASSERT(std::abs(values(i) - 1.) < 1e-11,
          "Lagrange polynomial seems to not be nodal, p_i(xi_i) = {}!", values(i));
    }
#endif
  }



  template <int nsd>
  void Core::FE::LagrangeBasisTet<nsd>::fill_unit_node_points(
      Core::LinAlg::SerialDenseMatrix& matrix) const
  {
    matrix.shape(fekete_points_.numRows(), fekete_points_.numCols());
    for (int i = 0; i < fekete_points_.numCols(); ++i)
      for (int j = 0; j < fekete_points_.numRows(); ++j) matrix(j, i) = fekete_points_(j, i);
  }

  template <int nsd>
  Core::FE::PolynomialSpaceCache<nsd>& Core::FE::PolynomialSpaceCache<nsd>::instance()
  {
    static Core::Utils::SingletonOwner<Core::FE::PolynomialSpaceCache<nsd>> owner(
        []()
        {
          return std::unique_ptr<Core::FE::PolynomialSpaceCache<nsd>>(
              new PolynomialSpaceCache<nsd>);
        });

    return *owner.instance(Core::Utils::SingletonAction::create);
  }

  template <int nsd>
  std::shared_ptr<Core::FE::PolynomialSpace<nsd>> Core::FE::PolynomialSpaceCache<nsd>::create(
      PolynomialSpaceParams params)
  {
    typename std::map<PolynomialSpaceParams,
        std::shared_ptr<Core::FE::PolynomialSpace<nsd>>>::iterator i = ps_cache_.find(params);
    if (i != ps_cache_.end())
    {
      return i->second;
    }

    // this is expensive and should not be done too often
    std::shared_ptr<PolynomialSpace<nsd>> ps;
    ps = std::make_shared<PolynomialSpace<nsd>>(params);

    ps_cache_[params] = ps;

    return ps;
  }


  // explicit instantiations
  template class PolynomialSpaceTensor<1, LagrangePolynomial>;
  template class PolynomialSpaceTensor<2, LagrangePolynomial>;
  template class PolynomialSpaceTensor<3, LagrangePolynomial>;

  template class PolynomialSpaceComplete<1, Polynomial>;
  template class PolynomialSpaceComplete<2, Polynomial>;
  template class PolynomialSpaceComplete<3, Polynomial>;

  template class LagrangeBasisTet<1>;
  template class LagrangeBasisTet<2>;
  template class LagrangeBasisTet<3>;

  template class PolynomialSpaceCache<1>;
  template class PolynomialSpaceCache<2>;
  template class PolynomialSpaceCache<3>;
}  // namespace Core::FE

FOUR_C_NAMESPACE_CLOSE

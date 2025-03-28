// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_linalg_utils_densematrix_eigen.hpp"

#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_utils_densematrix_multiply.hpp"
#include "4C_unittest_utils_assertions_test.hpp"

FOUR_C_NAMESPACE_OPEN

namespace
{
  /*
   * \note Helper functions for the dense eigen test routines.
   *
   */
  template <size_t length>
  void assert_eigen_values(const Core::LinAlg::SerialDenseVector& eigenvalues,
      const std::array<double, length>& eig_compare)
  {
    EXPECT_EQ(eigenvalues.length(), length);
    for (unsigned i = 0; i < length; ++i) EXPECT_NEAR(eigenvalues[i], eig_compare[i], 1e-9);
  }

  template <unsigned int size, size_t length>
  void assert_eigen_values(const Core::LinAlg::Matrix<size, size>& eigenvalues,
      const std::array<double, length>& eig_compare)
  {
    for (unsigned i = 0; i < size; ++i)
    {
      for (unsigned j = 0; j < size; ++j)
      {
        if (i == j)
          EXPECT_NEAR(eigenvalues(i, j), eig_compare[i], 1e-9) << "for i=" << i << ", j=" << j;
        else
          EXPECT_NEAR(eigenvalues(i, j), 0.0, 1e-9) << "for i=" << i << ", j=" << j;
      }
    }
  }

  template <unsigned int size, size_t length>
  void assert_eigen_values(
      const Core::LinAlg::Matrix<size, size, std::complex<double>>& eigenvalues,
      const std::array<std::complex<double>, length>& eig_compare)
  {
    for (unsigned i = 0; i < size; ++i)
    {
      for (unsigned j = 0; j < size; ++j)
      {
        if (i == j)
        {
          EXPECT_NEAR(eigenvalues(i, j).real(), eig_compare[i].real(), 1e-9)
              << " (real part) for i=" << i << ", j=" << j;
          EXPECT_NEAR(eigenvalues(i, j).imag(), eig_compare[i].imag(), 1e-9)
              << " (imaginary part) for i=" << i << ", j=" << j;
        }
        else
        {
          EXPECT_NEAR(eigenvalues(i, j).real(), 0.0, 1e-9) << "for i=" << i << ", j=" << j;
          EXPECT_NEAR(eigenvalues(i, j).imag(), 0.0, 1e-9) << "for i=" << i << ", j=" << j;
        }
      }
    }
  }

  template <size_t length>
  void assert_eigen_problem(const Core::LinAlg::SerialDenseMatrix& A,
      const Core::LinAlg::SerialDenseVector& eigenvalues,
      const Core::LinAlg::SerialDenseMatrix& eigenvectors,
      const std::array<double, length>& eig_compare)
  {
    EXPECT_EQ(eigenvalues.length(), length);
    EXPECT_EQ(A.numRows(), length);
    EXPECT_EQ(A.numCols(), length);
    EXPECT_EQ(eigenvectors.numRows(), length);
    EXPECT_EQ(eigenvectors.numCols(), length);

    assert_eigen_values(eigenvalues, eig_compare);

    Core::LinAlg::SerialDenseMatrix A_result(length, length, true);
    for (std::size_t i = 0; i < length; ++i)
    {
      Core::LinAlg::SerialDenseMatrix v(length, 1, false);
      for (std::size_t j = 0; j < length; ++j) v(j, 0) = eigenvectors(j, i);
      Core::LinAlg::multiply_nt(1.0, A_result, eigenvalues(i), v, v);
    }

    FOUR_C_EXPECT_NEAR(A, A_result, 1e-9);
  }

  template <unsigned int size, size_t length>
  void assert_eigen_problem(const Core::LinAlg::Matrix<size, size>& A,
      const Core::LinAlg::Matrix<size, size>& eigenvalues,
      const Core::LinAlg::Matrix<size, size>& eigenvectors,
      const std::array<double, length>& eig_compare)
  {
    assert_eigen_values(eigenvalues, eig_compare);

    Core::LinAlg::Matrix<size, size> A_result(Core::LinAlg::Initialization::zero);
    for (unsigned int i = 0; i < size; ++i)
    {
      Core::LinAlg::Matrix<size, 1> v(Core::LinAlg::Initialization::uninitialized);
      for (unsigned int j = 0; j < size; ++j) v(j, 0) = eigenvectors(j, i);
      A_result.multiply_nt(eigenvalues(i, i), v, v, 1.0);
    }

    FOUR_C_EXPECT_NEAR(A, A_result, 1e-9);
  }

  template <unsigned int size, size_t length>
  void assert_eigen_problem(const Core::LinAlg::Matrix<size, size>& A,
      const Core::LinAlg::Matrix<size, size, std::complex<double>>& eigenvalues,
      const Core::LinAlg::Matrix<size, size, std::complex<double>>& eigenvectors,
      const std::array<std::complex<double>, length>& eig_compare)
  {
    assert_eigen_values(eigenvalues, eig_compare);

    Core::LinAlg::Matrix<size, size> A_result(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<size, size, std::complex<double>> inv_eigenvectors(
        Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<size, size, std::complex<double>> temp(eigenvectors);
    inv_eigenvectors.invert(temp);
    Core::LinAlg::Matrix<size, size, std::complex<double>> output_complex(
        Core::LinAlg::Initialization::zero);
    temp.multiply_nn(eigenvectors, eigenvalues);
    output_complex.multiply_nn(temp, inv_eigenvectors);
    // restore complex to real form (guaranteed for a real input matrix)
    for (unsigned int i = 0; i < size; ++i)
    {
      for (unsigned int j = 0; j < size; ++j)
      {
        A_result(i, j) = output_complex(i, j).real();
      }
    }

    FOUR_C_EXPECT_NEAR(A, A_result, 1e-9);
  }

  /*
   * \note The values for the matrix used in tests below are generated with python/numpy
   */
  TEST(LinalgDenseMatrixEigenTest, 2x2SymmetricEigenValues)
  {
    Core::LinAlg::SerialDenseMatrix A(2, 2, false);
    A(0, 0) = 0.9964456203546112;
    A(0, 1) = 0.490484665405466;
    A(1, 0) = 0.490484665405466;
    A(1, 1) = 0.5611378979071144;

    std::array eigenvalues{0.24218351254540577, 1.3154000057163198};

    Core::LinAlg::SerialDenseVector L(2);
    Core::LinAlg::symmetric_eigen_values(A, L, false);

    assert_eigen_values(L, eigenvalues);
  }

  TEST(LinalgDenseMatrixEigenTest, 2x2SymmetricEigenProblem)
  {
    Core::LinAlg::SerialDenseMatrix A(2, 2, false);
    A(0, 0) = 0.9964456203546112;
    A(0, 1) = 0.490484665405466;
    A(1, 0) = 0.490484665405466;
    A(1, 1) = 0.5611378979071144;

    std::array eigenvalues{0.24218351254540577, 1.3154000057163198};

    Core::LinAlg::SerialDenseMatrix eigenvectors(A);
    Core::LinAlg::SerialDenseVector L(2);
    Core::LinAlg::symmetric_eigen_problem(eigenvectors, L, false);

    assert_eigen_problem(A, L, eigenvectors, eigenvalues);
  }

  TEST(LinalgDenseMatrixEigenTest, 2x2SymmetricEigenNoVectors)
  {
    Core::LinAlg::SerialDenseMatrix A(2, 2, false);
    A(0, 0) = 0.9964456203546112;
    A(0, 1) = 0.490484665405466;
    A(1, 0) = 0.490484665405466;
    A(1, 1) = 0.5611378979071144;

    std::array eigenvalues{0.24218351254540577, 1.3154000057163198};

    Core::LinAlg::SerialDenseVector L(2);
    Core::LinAlg::symmetric_eigen(A, L, 'N', false);

    assert_eigen_values(L, eigenvalues);
  }

  TEST(LinalgDenseMatrixEigenTest, 2x2SymmetricEigenVectors)
  {
    Core::LinAlg::SerialDenseMatrix A(2, 2, false);
    A(0, 0) = 0.9964456203546112;
    A(0, 1) = 0.490484665405466;
    A(1, 0) = 0.490484665405466;
    A(1, 1) = 0.5611378979071144;

    std::array eigenvalues{0.24218351254540577, 1.3154000057163198};

    Core::LinAlg::SerialDenseMatrix eigenvectors(A);
    Core::LinAlg::SerialDenseVector L(2);
    Core::LinAlg::symmetric_eigen(eigenvectors, L, 'V', false);

    assert_eigen_problem(A, L, eigenvectors, eigenvalues);
  }

  TEST(LinalgDenseMatrixEigenTest, 2x2SYEV)
  {
    Core::LinAlg::Matrix<2, 2> A(Core::LinAlg::Initialization::uninitialized);
    A(0, 0) = 0.9964456203546112;
    A(0, 1) = 0.490484665405466;
    A(1, 0) = 0.490484665405466;
    A(1, 1) = 0.5611378979071144;

    std::array eigenvalues{0.24218351254540577, 1.3154000057163198};

    Core::LinAlg::Matrix<2, 2> V(Core::LinAlg::Initialization::uninitialized);
    Core::LinAlg::Matrix<2, 2> S(Core::LinAlg::Initialization::uninitialized);
    Core::LinAlg::syev(A, S, V);

    assert_eigen_problem(A, S, V, eigenvalues);
  }

  TEST(LinalgDenseMatrixEigenTest, 2x2GEEV)
  {
    Core::LinAlg::Matrix<2, 2> A(Core::LinAlg::Initialization::uninitialized);
    A(0, 0) = 0.9964456203546112;
    A(0, 1) = -0.490484665405466;
    A(1, 0) = 0.2345123987123478;
    A(1, 1) = 0.5611378979071144;

    std::array eigenvalues{std::complex<double>(0.77879176, 0.260099081),
        std::complex<double>(0.77879176, -0.260099081)};
    Core::LinAlg::Matrix<2, 2, std::complex<double>> V(Core::LinAlg::Initialization::uninitialized);
    Core::LinAlg::Matrix<2, 2, std::complex<double>> S(Core::LinAlg::Initialization::uninitialized);
    Core::LinAlg::geev(A, S, V);

    assert_eigen_problem(A, S, V, eigenvalues);
  }

  TEST(LinalgDenseMatrixEigenTest, 3x3SymmetricEigenValues)
  {
    Core::LinAlg::SerialDenseMatrix A(3, 3, false);
    A(0, 0) = 1.2966342861458506;
    A(0, 1) = 0.8940941796919223;
    A(0, 2) = 0.16862685184206302;
    A(1, 0) = 0.8940941796919223;
    A(1, 1) = 0.9880908794535803;
    A(1, 2) = 0.06322733832497837;
    A(2, 0) = 0.16862685184206302;
    A(2, 1) = 0.06322733832497837;
    A(2, 2) = 0.047048409972083906;

    std::array eigenvalues{0.01628207201103285, 0.2515293645924337, 2.0639621389680487};

    Core::LinAlg::SerialDenseVector L(3);
    Core::LinAlg::symmetric_eigen_values(A, L, false);

    assert_eigen_values(L, eigenvalues);
  }

  TEST(LinalgDenseMatrixEigenTest, 3x3SymmetricEigenProblem)
  {
    Core::LinAlg::SerialDenseMatrix A(3, 3, false);
    A(0, 0) = 1.2966342861458506;
    A(0, 1) = 0.8940941796919223;
    A(0, 2) = 0.16862685184206302;
    A(1, 0) = 0.8940941796919223;
    A(1, 1) = 0.9880908794535803;
    A(1, 2) = 0.06322733832497837;
    A(2, 0) = 0.16862685184206302;
    A(2, 1) = 0.06322733832497837;
    A(2, 2) = 0.047048409972083906;

    std::array eigenvalues{0.01628207201103285, 0.2515293645924337, 2.0639621389680487};

    Core::LinAlg::SerialDenseMatrix V(A);
    Core::LinAlg::SerialDenseVector L(3);
    Core::LinAlg::symmetric_eigen_problem(V, L, false);

    assert_eigen_problem(A, L, V, eigenvalues);
  }

  TEST(LinalgDenseMatrixEigenTest, 3x3SymmetricEigenNoVectors)
  {
    Core::LinAlg::SerialDenseMatrix A(3, 3, false);
    A(0, 0) = 1.2966342861458506;
    A(0, 1) = 0.8940941796919223;
    A(0, 2) = 0.16862685184206302;
    A(1, 0) = 0.8940941796919223;
    A(1, 1) = 0.9880908794535803;
    A(1, 2) = 0.06322733832497837;
    A(2, 0) = 0.16862685184206302;
    A(2, 1) = 0.06322733832497837;
    A(2, 2) = 0.047048409972083906;

    std::array eigenvalues{0.01628207201103285, 0.2515293645924337, 2.0639621389680487};

    Core::LinAlg::SerialDenseVector L(3);
    Core::LinAlg::symmetric_eigen(A, L, 'N', false);

    assert_eigen_values(L, eigenvalues);
  }

  TEST(LinalgDenseMatrixEigenTest, 3x3SymmetricEigenVectors)
  {
    Core::LinAlg::SerialDenseMatrix A(3, 3, false);
    A(0, 0) = 1.2966342861458506;
    A(0, 1) = 0.8940941796919223;
    A(0, 2) = 0.16862685184206302;
    A(1, 0) = 0.8940941796919223;
    A(1, 1) = 0.9880908794535803;
    A(1, 2) = 0.06322733832497837;
    A(2, 0) = 0.16862685184206302;
    A(2, 1) = 0.06322733832497837;
    A(2, 2) = 0.047048409972083906;

    std::array eigenvalues{0.01628207201103285, 0.2515293645924337, 2.0639621389680487};

    Core::LinAlg::SerialDenseMatrix V(A);
    Core::LinAlg::SerialDenseVector L(3);
    Core::LinAlg::symmetric_eigen(V, L, 'V', false);

    assert_eigen_problem(A, L, V, eigenvalues);
  }

  TEST(LinalgDenseMatrixEigenTest, 3x3SYEV)
  {
    Core::LinAlg::Matrix<3, 3> A(Core::LinAlg::Initialization::uninitialized);
    A(0, 0) = 1.2966342861458506;
    A(0, 1) = 0.8940941796919223;
    A(0, 2) = 0.16862685184206302;
    A(1, 0) = 0.8940941796919223;
    A(1, 1) = 0.9880908794535803;
    A(1, 2) = 0.06322733832497837;
    A(2, 0) = 0.16862685184206302;
    A(2, 1) = 0.06322733832497837;
    A(2, 2) = 0.047048409972083906;

    std::array eigenvalues{0.01628207201103285, 0.2515293645924337, 2.0639621389680487};

    Core::LinAlg::Matrix<3, 3> V(Core::LinAlg::Initialization::uninitialized);
    Core::LinAlg::Matrix<3, 3> S(Core::LinAlg::Initialization::uninitialized);
    Core::LinAlg::syev(A, S, V);

    assert_eigen_problem(A, S, V, eigenvalues);
  }

  TEST(LinalgDenseMatrixEigenTest, 3x3GEEV)
  {
    Core::LinAlg::Matrix<3, 3> A(Core::LinAlg::Initialization::uninitialized);
    A(0, 0) = 0.9876543212345678;
    A(0, 1) = 1.2345678901234567;
    A(0, 2) = 2.3456789012345678;
    A(1, 0) = -1.2345678901234567;
    A(1, 1) = 0.8765432109876543;
    A(1, 2) = 3.4567890123456789;
    A(2, 0) = 1.6789012345678901;
    A(2, 1) = 3.4567890123456789;
    A(2, 2) = 2.3456789012345678;

    std::array eigenvalues{std::complex<double>(-2.262778181431353, 0.0),
        std::complex<double>(1.010906471515987, 0.0), std::complex<double>(5.461748143372156, 0.0)};

    Core::LinAlg::Matrix<3, 3, std::complex<double>> V(Core::LinAlg::Initialization::uninitialized);
    Core::LinAlg::Matrix<3, 3, std::complex<double>> S(Core::LinAlg::Initialization::uninitialized);
    Core::LinAlg::geev(A, S, V);

    assert_eigen_problem(A, S, V, eigenvalues);
  }


  TEST(LinalgDenseMatrixEigenTest, 4x4SymmetricEigenValues)
  {
    Core::LinAlg::SerialDenseMatrix A(4, 4, false);
    A(0, 0) = 0.5561130226871257;
    A(0, 1) = 1.0052918588741722;
    A(0, 2) = 0.8408494685470309;
    A(0, 3) = 0.8731301282118089;
    A(1, 0) = 1.0052918588741722;
    A(1, 1) = 2.023681530073728;
    A(1, 2) = 1.7222521019056944;
    A(1, 3) = 1.6511949164466262;
    A(2, 0) = 0.8408494685470309;
    A(2, 1) = 1.7222521019056944;
    A(2, 2) = 1.6035737196981317;
    A(2, 3) = 1.4613812746280035;
    A(3, 0) = 0.8731301282118089;
    A(3, 1) = 1.6511949164466262;
    A(3, 2) = 1.4613812746280035;
    A(3, 3) = 1.4335181777869124;

    std::array eigenvalues{
        0.00023212100268553735, 0.06219024553961773, 0.11100584442852221, 5.443458239275074};

    Core::LinAlg::SerialDenseVector L(4);
    Core::LinAlg::symmetric_eigen_values(A, L, false);

    assert_eigen_values(L, eigenvalues);
  }

  TEST(LinalgDenseMatrixEigenTest, 4x4SymmetricEigenProblem)
  {
    Core::LinAlg::SerialDenseMatrix A(4, 4, false);
    A(0, 0) = 0.5561130226871257;
    A(0, 1) = 1.0052918588741722;
    A(0, 2) = 0.8408494685470309;
    A(0, 3) = 0.8731301282118089;
    A(1, 0) = 1.0052918588741722;
    A(1, 1) = 2.023681530073728;
    A(1, 2) = 1.7222521019056944;
    A(1, 3) = 1.6511949164466262;
    A(2, 0) = 0.8408494685470309;
    A(2, 1) = 1.7222521019056944;
    A(2, 2) = 1.6035737196981317;
    A(2, 3) = 1.4613812746280035;
    A(3, 0) = 0.8731301282118089;
    A(3, 1) = 1.6511949164466262;
    A(3, 2) = 1.4613812746280035;
    A(3, 3) = 1.4335181777869124;

    std::array eigenvalues{
        0.00023212100268553735, 0.06219024553961773, 0.11100584442852221, 5.443458239275074};

    Core::LinAlg::SerialDenseMatrix V(A);
    Core::LinAlg::SerialDenseVector L(4);
    Core::LinAlg::symmetric_eigen_problem(V, L, false);

    assert_eigen_problem(A, L, V, eigenvalues);
  }

  TEST(LinalgDenseMatrixEigenTest, 4x4SymmetricEigenNoVectors)
  {
    Core::LinAlg::SerialDenseMatrix A(4, 4, false);
    A(0, 0) = 0.5561130226871257;
    A(0, 1) = 1.0052918588741722;
    A(0, 2) = 0.8408494685470309;
    A(0, 3) = 0.8731301282118089;
    A(1, 0) = 1.0052918588741722;
    A(1, 1) = 2.023681530073728;
    A(1, 2) = 1.7222521019056944;
    A(1, 3) = 1.6511949164466262;
    A(2, 0) = 0.8408494685470309;
    A(2, 1) = 1.7222521019056944;
    A(2, 2) = 1.6035737196981317;
    A(2, 3) = 1.4613812746280035;
    A(3, 0) = 0.8731301282118089;
    A(3, 1) = 1.6511949164466262;
    A(3, 2) = 1.4613812746280035;
    A(3, 3) = 1.4335181777869124;

    std::array eigenvalues{
        0.00023212100268553735, 0.06219024553961773, 0.11100584442852221, 5.443458239275074};

    Core::LinAlg::SerialDenseVector L(4);
    Core::LinAlg::symmetric_eigen(A, L, 'N', false);

    assert_eigen_values(L, eigenvalues);
  }

  TEST(LinalgDenseMatrixEigenTest, 4x4SymmetricEigenVectors)
  {
    Core::LinAlg::SerialDenseMatrix A(4, 4, false);
    A(0, 0) = 0.5561130226871257;
    A(0, 1) = 1.0052918588741722;
    A(0, 2) = 0.8408494685470309;
    A(0, 3) = 0.8731301282118089;
    A(1, 0) = 1.0052918588741722;
    A(1, 1) = 2.023681530073728;
    A(1, 2) = 1.7222521019056944;
    A(1, 3) = 1.6511949164466262;
    A(2, 0) = 0.8408494685470309;
    A(2, 1) = 1.7222521019056944;
    A(2, 2) = 1.6035737196981317;
    A(2, 3) = 1.4613812746280035;
    A(3, 0) = 0.8731301282118089;
    A(3, 1) = 1.6511949164466262;
    A(3, 2) = 1.4613812746280035;
    A(3, 3) = 1.4335181777869124;

    std::array eigenvalues{
        0.00023212100268553735, 0.06219024553961773, 0.11100584442852221, 5.443458239275074};

    Core::LinAlg::SerialDenseMatrix V(A);
    Core::LinAlg::SerialDenseVector L(4);
    Core::LinAlg::symmetric_eigen(V, L, 'V', false);

    assert_eigen_problem(A, L, V, eigenvalues);
  }

  TEST(LinalgDenseMatrixEigenTest, 4x4SYEV)
  {
    Core::LinAlg::Matrix<4, 4> A(Core::LinAlg::Initialization::uninitialized);
    A(0, 0) = 0.5561130226871257;
    A(0, 1) = 1.0052918588741722;
    A(0, 2) = 0.8408494685470309;
    A(0, 3) = 0.8731301282118089;
    A(1, 0) = 1.0052918588741722;
    A(1, 1) = 2.023681530073728;
    A(1, 2) = 1.7222521019056944;
    A(1, 3) = 1.6511949164466262;
    A(2, 0) = 0.8408494685470309;
    A(2, 1) = 1.7222521019056944;
    A(2, 2) = 1.6035737196981317;
    A(2, 3) = 1.4613812746280035;
    A(3, 0) = 0.8731301282118089;
    A(3, 1) = 1.6511949164466262;
    A(3, 2) = 1.4613812746280035;
    A(3, 3) = 1.4335181777869124;

    std::array eigenvalues{
        0.00023212100268553735, 0.06219024553961773, 0.11100584442852221, 5.443458239275074};

    Core::LinAlg::Matrix<4, 4> V(Core::LinAlg::Initialization::uninitialized);
    Core::LinAlg::Matrix<4, 4> S(Core::LinAlg::Initialization::uninitialized);
    Core::LinAlg::syev(A, S, V);

    assert_eigen_problem(A, S, V, eigenvalues);
  }

}  // namespace

FOUR_C_NAMESPACE_CLOSE

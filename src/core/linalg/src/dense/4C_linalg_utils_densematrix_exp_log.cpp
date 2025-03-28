// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_linalg_utils_densematrix_exp_log.hpp"

#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_fixedsizematrix_tensor_derivatives.hpp"
#include "4C_linalg_fixedsizematrix_tensor_products.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_linalg_four_tensor.hpp"
#include "4C_linalg_utils_densematrix_eigen.hpp"

FOUR_C_NAMESPACE_OPEN

using vmap = Core::LinAlg::Voigt::IndexMappings;


template <unsigned int dim>
Core::LinAlg::Matrix<dim, dim> Core::LinAlg::matrix_exp(const Core::LinAlg::Matrix<dim, dim>& input)
{
  // declare output matrix
  Core::LinAlg::Matrix<dim, dim> output(Core::LinAlg::Initialization::zero);

  // get norm of input matrix
  double mat_norm = input.norm2();

  // direct calculation for zero-matrix
  if (mat_norm == 0.)
  {
    for (unsigned int i = 0; i < dim; i++) output(i, i) = 1.;
    return output;
  }

  // compute matrix exponential via power series for small matrix norms. This is usually
  // faster than by spectral decomposition for matrices which are close to zero.
  if (mat_norm < 2.0)
  {
    // set maximum number of terms
    int n_max = 50;

    // set convergence tolerance for the employed series description
    double conv_tol = 1.0e-32;
    // loop through the terms of the power series
    int n = 0;
    int facn = 1;
    for (unsigned int i = 0; i < dim; i++) output(i, i) = 1.;
    Core::LinAlg::Matrix<dim, dim> tmp(output);
    Core::LinAlg::Matrix<dim, dim> tmp2(output);
    while (n < n_max && tmp.norm2() / facn > conv_tol)
    {
      // successively add terms from the power series to the output
      n++;
      facn *= n;
      tmp.multiply(1.0, tmp2, input, 0.0);
      tmp2 = tmp;
      output.update(1. / facn, tmp, 1.);
    }

    // throw error if no convergence is reached after the maximum number of terms
    FOUR_C_ASSERT_ALWAYS(n < n_max, "Matrix exponential unconverged in {} steps.", n);

    return output;
  }
  else
  {
    // spectral decomposition for higher matrix norms
    Core::LinAlg::Matrix<dim, dim, std::complex<double>> eigenval_matrix(
        Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<dim, dim, std::complex<double>> eigenvect_matrix(
        Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<dim, dim> temp_input(input);
    Core::LinAlg::geev(temp_input, eigenval_matrix, eigenvect_matrix);

    // get the exponentials of the eigenvalues
    for (unsigned int i = 0; i < dim; ++i)
    {
      eigenval_matrix(i, i) = std::exp(eigenval_matrix(i, i));
    }

    // get inverse of the eigenvector matrix
    Core::LinAlg::Matrix<dim, dim, std::complex<double>> inv_eigenvect_matrix(
        Core::LinAlg::Initialization::zero);
    inv_eigenvect_matrix.invert(eigenvect_matrix);

    // construct the exponential function
    Core::LinAlg::Matrix<dim, dim, std::complex<double>> temp(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<dim, dim, std::complex<double>> output_complex(
        Core::LinAlg::Initialization::zero);
    temp.multiply_nn(eigenvect_matrix, eigenval_matrix);
    output_complex.multiply_nn(temp, inv_eigenvect_matrix);
    // restore complex to real form (guaranteed for a real input matrix)
    for (unsigned int i = 0; i < dim; ++i)
    {
      for (unsigned int j = 0; j < dim; ++j)
      {
        output(i, j) = output_complex(i, j).real();
      }
    }

    return output;
  }
}

template <unsigned int dim>
Core::LinAlg::Matrix<dim, dim> Core::LinAlg::matrix_log(const Core::LinAlg::Matrix<dim, dim>& input)
{
  // auxiliaries
  Core::LinAlg::Matrix<dim, dim> id(Core::LinAlg::Initialization::zero);
  for (unsigned int i = 0; i < dim; ++i)
  {
    id(i, i) = 1.0;
  }
  Core::LinAlg::Matrix<dim, dim> temp(Core::LinAlg::Initialization::zero);

  // declare output matrix
  Core::LinAlg::Matrix<dim, dim> output(Core::LinAlg::Initialization::zero);

  // set convergence tolerance for the employed series description
  double conv_tol = 1.0e-10;

  // characteristic matrix \f$ \boldsymbol{I} - \boldsymbol{A} \f$
  Core::LinAlg::Matrix<dim, dim> id_minus_A(Core::LinAlg::Initialization::zero);
  id_minus_A.update(1.0, id, -1.0, input, 0.0);

  // employ Taylor series if matrix norm smaller than 1 for characteristic matrix
  if (id_minus_A.norm2() < 1.0)
  {
    // set initial exponent \f$ m \f$ and the maximum acceptable number of series terms
    int m = 1;
    int m_max = 50;

    // \f$ \boldsymbol{A} - \boldsymbol{I} \f$
    Core::LinAlg::Matrix<dim, dim> A_minus_id(Core::LinAlg::Initialization::zero);
    A_minus_id.update(1.0, input, -1.0, id);

    //  \f$ \left(\boldsymbol{A} - \boldsymbol{I}\right)^m \f$
    Core::LinAlg::Matrix<dim, dim> A_minus_id_m = A_minus_id;

    while (A_minus_id_m.norm2() > conv_tol)
    {
      // check whether the maximum number of terms was reached
      FOUR_C_ASSERT_ALWAYS(
          m <= m_max, "Couldn't compute the matrix logarithm using the Taylor series!");

      // update output matrix
      output.update(std::pow(-1.0, m + 1) / m, A_minus_id_m, 1.0);

      // update \f$ \left(\boldsymbol{A} - \boldsymbol{I}\right)^m \f$
      temp = A_minus_id_m;
      A_minus_id_m.multiply(1.0, temp, A_minus_id, 0.0);

      // increment m
      m += 1;
    }
    return output;
  }

  // \f$ \boldsymbol{I} + \boldsymbol{A} \f$
  Core::LinAlg::Matrix<dim, dim> id_plus_A(Core::LinAlg::Initialization::zero);
  id_plus_A.update(1.0, id, 1.0, input, 0.0);

  // \f$ \left( \boldsymbol{I} + \boldsymbol{A} \right)^{-1} \f$
  Core::LinAlg::Matrix<dim, dim> inv_id_plus_A(Core::LinAlg::Initialization::zero);
  inv_id_plus_A.invert(id_plus_A);

  // update matrix: \f$ \left[ \left( \boldsymbol{I} - \boldsymbol{A} \right) \left( \boldsymbol{I}
  // + \boldsymbol{A}
  // \right)^{-1} \right] \f$
  Core::LinAlg::Matrix<dim, dim> update_mat(Core::LinAlg::Initialization::zero);
  update_mat.multiply(1.0, id_minus_A, inv_id_plus_A, 0.0);

  // employ the Gregory series, if the norm of the first update matrix is smaller than 1.0
  if (update_mat.norm2() < 1.0)
  {
    // set initial exponent \f$ m \f$ and the maximum acceptable number of series terms
    int m = 0;
    int m_max = 50;

    // \f$ \left[ \left( \boldsymbol{I} - \boldsymbol{A} \right) \left( \boldsymbol{I} +
    // \boldsymbol{A} \right)^{-1} \right]^{2m+1}
    // \f$
    Core::LinAlg::Matrix<dim, dim> update_mat_2mpl1(Core::LinAlg::Initialization::zero);
    update_mat_2mpl1 = update_mat;

    output.clear();
    while (update_mat_2mpl1.norm2() > conv_tol)
    {
      // check whether the maximum number of terms was reached
      FOUR_C_ASSERT_ALWAYS(
          m <= m_max, "Couldn't compute the matrix logarithm using the Gregory series!");

      // update output matrix
      output.update(-2.0 / (2.0 * m + 1.0), update_mat_2mpl1, 1.0);

      // update update_mat_2mpl1
      temp.multiply(1.0, update_mat_2mpl1, update_mat, 0.0);
      update_mat_2mpl1.multiply(1.0, temp, update_mat, 0.0);

      // increment m
      m += 1;
    }

    return output;
  }
  // spectral decomposition as the last resort
  else
  {
    Core::LinAlg::Matrix<dim, dim, std::complex<double>> eigenval_matrix(
        Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<dim, dim, std::complex<double>> eigenvect_matrix(
        Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<dim, dim> temp_input(input);
    Core::LinAlg::geev(temp_input, eigenval_matrix, eigenvect_matrix);

    // get the (principal) logarithms of the eigenvalues
    for (unsigned int i = 0; i < dim; ++i)
    {
      FOUR_C_ASSERT_ALWAYS(eigenval_matrix(i, i).real() >= 0.0,
          "The current matrix logarithm implementation only considers the case where all "
          "eigenvalues "
          "possess positive real parts! This is not given here, real part:  {}",
          eigenval_matrix(i, i).real());
      eigenval_matrix(i, i) = std::log(eigenval_matrix(i, i));
    }

    // get inverse of the eigenvector matrix
    Core::LinAlg::Matrix<dim, dim, std::complex<double>> inv_eigenvect_matrix(
        Core::LinAlg::Initialization::zero);
    inv_eigenvect_matrix.invert(eigenvect_matrix);

    // construct the logarithm function
    Core::LinAlg::Matrix<dim, dim, std::complex<double>> tmp(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<dim, dim, std::complex<double>> output_complex(
        Core::LinAlg::Initialization::zero);
    tmp.multiply_nn(eigenvect_matrix, eigenval_matrix);
    output_complex.multiply_nn(tmp, inv_eigenvect_matrix);
    // restore complex to real form (guaranteed for a real input matrix)
    for (unsigned int i = 0; i < dim; ++i)
    {
      for (unsigned int j = 0; j < dim; ++j)
      {
        output(i, j) = output_complex(i, j).real();
      }
    }

    return output;
  }
}

Core::LinAlg::Matrix<9, 9> Core::LinAlg::matrix_3x3_exp_1st_deriv(
    const Core::LinAlg::Matrix<3, 3>& input)
{
  // declare output variable
  Core::LinAlg::Matrix<9, 9> output(Core::LinAlg::Initialization::zero);

  // see Souza-Neto: Computational Methods for plasticity, Box B.2.
  int nmax = 0;
  int nIter = 0;
  int nfac = 1;
  Core::LinAlg::Matrix<3, 3> tmp1;
  Core::LinAlg::Matrix<3, 3> tmp2(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; i++) tmp2(i, i) = 1.;

  // declare vector of all needed powers of X
  std::vector<Core::LinAlg::Matrix<3, 3>> Xn;
  Xn.resize(0);
  Xn.push_back(tmp2);

  // declare vector of all needed factorials
  std::vector<int> fac;
  fac.resize(0);
  fac.push_back(nfac);

  // compute nmax and Xn
  while (nIter < 50 && tmp2.norm2() / nfac > 1.e-32)
  {
    nIter++;
    nfac *= nIter;
    fac.push_back(nfac);
    tmp1.multiply(tmp2, input);
    Xn.push_back(tmp1);
    tmp2 = tmp1;
  }

  FOUR_C_ASSERT_ALWAYS(nIter < 50, "matrix exponential unconverged in {} steps", nIter);
  nmax = nIter;

  // compose derivative of matrix exponential (non-symmetric Voigt-notation)
  for (int n = 1; n <= nmax; n++)
    for (int m = 1; m <= n; m++)
      Core::LinAlg::Tensor::add_non_symmetric_product(
          1. / fac[n], Xn.at(m - 1), Xn.at(n - m), output);

  return output;
}

Core::LinAlg::Matrix<6, 6> Core::LinAlg::sym_matrix_3x3_exp_1st_deriv(
    const Core::LinAlg::Matrix<3, 3>& input)
{
  // get matrix norm
  double norm = input.norm2();

  // declare output variable
  Core::LinAlg::Matrix<6, 6> output(Core::LinAlg::Initialization::zero);

  // compute 4-th order identity tensor
  Core::LinAlg::Matrix<6, 6> id4sharp(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; i++) id4sharp(i, i) = 1.0;
  for (int i = 3; i < 6; i++) id4sharp(i, i) = 0.5;

  // direct calculation for zero-matrix
  if (norm == 0.)
  {
    return id4sharp;
  }

  // further norm checking to determine appropriate computation procedure
  if (norm < 0.3)
  {
    // see Souza-Neto: Computational Methods for plasticity, Box B.2.
    int nmax = 0;
    int nIter = 0;
    int nfac = 1;
    Core::LinAlg::Matrix<3, 3> tmp1;
    Core::LinAlg::Matrix<3, 3> tmp2(Core::LinAlg::Initialization::zero);
    for (int i = 0; i < 3; i++) tmp2(i, i) = 1.;

    // declare vector of all needed powers of X
    std::vector<Core::LinAlg::Matrix<3, 3>> Xn;
    Xn.resize(0);
    Xn.push_back(tmp2);

    // declare vector of all needed factorials
    std::vector<int> fac;
    fac.resize(0);
    fac.push_back(nfac);

    // compute nmax and Xn
    while (nIter < 50 && tmp2.norm2() / nfac > 1.e-32)
    {
      nIter++;
      nfac *= nIter;
      fac.push_back(nfac);
      tmp1.multiply(tmp2, input);
      Xn.push_back(tmp1);
      tmp2 = tmp1;
    }
    FOUR_C_ASSERT_ALWAYS(nIter < 50, "matrix exponential unconverged in {} steps", nIter);
    nmax = nIter;

    // compose derivative of matrix exponential (symmetric Voigt-notation)
    for (int n = 1; n <= nmax; n++)
    {
      for (int m = 1; m <= n / 2; m++)
        Core::LinAlg::Tensor::add_symmetric_holzapfel_product(
            output, Xn.at(m - 1), Xn.at(n - m), .5 / fac[n]);
      if (n % 2 == 1)
        Core::LinAlg::Tensor::add_symmetric_holzapfel_product(
            output, Xn.at((n - 1) / 2), Xn.at((n - 1) / 2), .25 / fac[n]);
    }
  }
  else
  {
    double EWtolerance = 1.e-12;

    Core::LinAlg::Matrix<3, 3> EV(input);
    Core::LinAlg::Matrix<3, 3> EW;
    Core::LinAlg::syev(EV, EW, EV);

    Core::LinAlg::Matrix<3, 1> vec1;
    Core::LinAlg::Matrix<3, 1> vec2;
    Core::LinAlg::Matrix<3, 3> tmp1;
    Core::LinAlg::Matrix<3, 3> tmp2;

    // souza eq. (A.52)
    // note: EW stored in ascending order

    //  d X^2 / d X  =  1/2 * (  delta_jk X_lj + delta_il X_kj
    //                         + delta_jl X_ik + delta_kj X_il )
    //
    // y_i = log(x_i)
    // dy_i / dx_j = delta_ij 1/x_i

    Core::LinAlg::Matrix<3, 3> id2(Core::LinAlg::Initialization::zero);
    for (int i = 0; i < 3; i++) id2(i, i) = 1.0;
    //  // --------------------------------- switch by number of equal eigenvalues
    if (abs(EW(0, 0) - EW(1, 1)) < EWtolerance &&
        abs(EW(1, 1) - EW(2, 2)) < EWtolerance)  // ------------------ x_a == x_b == x_c
    {
      // calculate derivative
      output = id4sharp;
      output.scale(exp(EW(0, 0)));
    }

    else if ((abs(EW(0, 0) - EW(1, 1)) < EWtolerance && abs(EW(1, 1) - EW(2, 2)) > EWtolerance) ||
             (abs(EW(0, 0) - EW(1, 1)) > EWtolerance &&
                 abs(EW(1, 1) - EW(2, 2)) <
                     EWtolerance))  // ---- x_a != x_b == x_c or x_a == x_b != x_c
    {
      // factors
      double s1 = 0.0;
      double s2 = 0.0;
      double s3 = 0.0;
      double s4 = 0.0;
      double s5 = 0.0;
      double s6 = 0.0;

      int a = 0;
      int c = 0;

      // switch which two EW are equal
      if (abs(EW(0, 0) - EW(1, 1)) < EWtolerance &&
          abs(EW(1, 1) - EW(2, 2)) > EWtolerance)  // ----------------------- x_a == x_b != x_c
      {
        a = 2;
        c = 0;
      }
      else if (abs(EW(0, 0) - EW(1, 1)) > EWtolerance &&
               abs(EW(1, 1) - EW(2, 2)) < EWtolerance)  // ------------------ x_a != x_b == x_c
      {
        a = 0;
        c = 2;
      }
      else
        FOUR_C_THROW("you should not be here");

      // in souza eq. (A.53):
      s1 = (exp(EW(a, a)) - exp(EW(c, c))) / (pow(EW(a, a) - EW(c, c), 2.0)) -
           exp(EW(c, c)) / (EW(a, a) - EW(c, c));
      s2 = 2.0 * EW(c, c) * (exp(EW(a, a)) - exp(EW(c, c))) / (pow(EW(a, a) - EW(c, c), 2.0)) -
           (EW(a, a) + EW(c, c)) / (EW(a, a) - EW(c, c)) * exp(EW(c, c));
      s3 = 2.0 * (exp(EW(a, a)) - exp(EW(c, c))) / (pow(EW(a, a) - EW(c, c), 3.0)) -
           (exp(EW(a, a)) + exp(EW(c, c))) / (pow(EW(a, a) - EW(c, c), 2.0));
      s4 = EW(c, c) * s3;
      s5 = s4;
      s6 = EW(c, c) * EW(c, c) * s3;

      // calculate derivative
      Core::LinAlg::Tensor::add_derivative_of_squared_tensor(output, s1, input, 1.);
      output.update(-s2, id4sharp, 1.);
      Core::LinAlg::Tensor::add_elasticity_tensor_product(output, -1. * s3, input, input, 1.);
      Core::LinAlg::Tensor::add_elasticity_tensor_product(output, s4, input, id2, 1.);
      Core::LinAlg::Tensor::add_elasticity_tensor_product(output, s5, id2, input, 1.);
      Core::LinAlg::Tensor::add_elasticity_tensor_product(output, -s6, id2, id2, 1.);
    }

    else if (abs(EW(0, 0) - EW(1, 1)) > EWtolerance &&
             abs(EW(1, 1) - EW(2, 2)) > EWtolerance)  // ----------------- x_a != x_b != x_c
    {
      for (int a = 0; a < 3; a++)  // loop over all eigenvalues
      {
        int b = (a + 1) % 3;
        int c = (a + 2) % 3;

        Core::LinAlg::Matrix<3, 1> ea;
        Core::LinAlg::Matrix<3, 1> eb;
        Core::LinAlg::Matrix<3, 1> ec;
        for (int i = 0; i < 3; i++)
        {
          ea(i) = EV(i, a);
          eb(i) = EV(i, b);
          ec(i) = EV(i, c);
        }
        Core::LinAlg::Matrix<3, 3> Ea;
        Ea.multiply_nt(ea, ea);
        Core::LinAlg::Matrix<3, 3> Eb;
        Eb.multiply_nt(eb, eb);
        Core::LinAlg::Matrix<3, 3> Ec;
        Ec.multiply_nt(ec, ec);

        double fac = exp(EW(a, a)) / ((EW(a, a) - EW(b, b)) * (EW(a, a) - EW(c, c)));

        // + d X^2 / d X
        Core::LinAlg::Tensor::add_derivative_of_squared_tensor(output, fac, input, 1.);

        // - (x_b + x_c) I_s
        output.update(-1. * (EW(b, b) + EW(c, c)) * fac, id4sharp, 1.);

        // - [(x_a - x_b) + (x_a - x_c)] E_a \dyad E_a
        Core::LinAlg::Tensor::add_elasticity_tensor_product(
            output, -1. * fac * ((EW(a, a) - EW(b, b)) + (EW(a, a) - EW(c, c))), Ea, Ea, 1.);


        // - (x_b - x_c) (E_b \dyad E_b)
        Core::LinAlg::Tensor::add_elasticity_tensor_product(
            output, -1. * fac * (EW(b, b) - EW(c, c)), Eb, Eb, 1.);

        // + (x_b - x_c) (E_c \dyad E_c)
        Core::LinAlg::Tensor::add_elasticity_tensor_product(
            output, fac * (EW(b, b) - EW(c, c)), Ec, Ec, 1.);

        // dy / dx_a E_a \dyad E_a
        Core::LinAlg::Tensor::add_elasticity_tensor_product(output, exp(EW(a, a)), Ea, Ea, 1.);
      }  // end loop over all eigenvalues
    }
    else
      FOUR_C_THROW("you should not be here.");
  }
  return output;
}

Core::LinAlg::Matrix<9, 9> Core::LinAlg::matrix_3x3_log_1st_deriv(
    const Core::LinAlg::Matrix<3, 3>& input)
{
  // auxiliaries
  Core::LinAlg::Matrix<3, 3> id_3x3(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; ++i)
  {
    id_3x3(i, i) = 1.0;
  }
  Core::LinAlg::Matrix<9, 9> id4(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Tensor::add_non_symmetric_product(1.0, id_3x3, id_3x3, id4);
  Core::LinAlg::FourTensor<3> id4_FourTensor(true);
  Core::LinAlg::Voigt::setup_four_tensor_from_9x9_voigt_matrix(id4_FourTensor, id4);
  Core::LinAlg::Matrix<3, 3> temp3x3(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<9, 9> temp9x9(Core::LinAlg::Initialization::zero);
  Core::LinAlg::FourTensor<3> tempFourTensor(true);
  Core::LinAlg::FourTensor<3> leftFourTensor(true);
  Core::LinAlg::FourTensor<3> rightFourTensor(true);

  // declare output variable
  Core::LinAlg::Matrix<9, 9> output(Core::LinAlg::Initialization::zero);
  // set convergence tolerance
  double conv_tol = 1.0e-10;
  // characteristic matrix \f$ \boldsymbol{I} - \boldsymbol{A} \f$
  Core::LinAlg::Matrix<3, 3> id_minus_A(Core::LinAlg::Initialization::zero);
  id_minus_A.update(1.0, id_3x3, -1.0, input, 0.0);

  // employ Taylor series if matrix norm smaller than 1 for characteristic matrix
  if (id_minus_A.norm2() < 1.0)
  {
    // set initial exponent \f$ m \f$ and the maximum acceptable number of series terms
    int m = 1;
    int m_max = 50;

    // \f$ \boldsymbol{A} - \boldsymbol{I} \f$
    Core::LinAlg::Matrix<3, 3> A_minus_id(Core::LinAlg::Initialization::zero);
    A_minus_id.update(1.0, input, -1.0, id_3x3);

    //\f$ \left( \bm{A} - \bm{I} \right)^T \f$
    Core::LinAlg::Matrix<3, 3> A_minus_idT(Core::LinAlg::Initialization::zero);
    A_minus_idT.multiply_tn(1.0, A_minus_id, id_3x3, 0.0);

    //\f$ \left( \bm{A} - \bm{I} \right)^{m-1} \f$
    Core::LinAlg::Matrix<3, 3> A_minus_id_mmin1 = id_3x3;

    // \f$ \frac{\partial \left( \bm{A} - \bm{I} \right)^{m}}{\partial \bm{A}} \f$
    Core::LinAlg::Matrix<9, 9> dA_minus_id_m_dA = id4;

    // \f$ \frac{\partial \left( \bm{A} - \bm{I} \right)^{m-1}}{\partial \bm{A}} \f$
    Core::LinAlg::FourTensor<3> dA_minus_id_mmin1_dA_FourTensor(true);

    // add series terms until convergence tolerance reached
    while (dA_minus_id_m_dA.norm2() > conv_tol)
    {
      // check whether the maximum number of terms was reached
      FOUR_C_ASSERT_ALWAYS(
          m <= m_max, "Couldn't compute the matrix logarithm derivative using the Taylor series!");

      // update output matrix
      output.update(std::pow(-1.0, m + 1.0) / m, dA_minus_id_m_dA, 1.0);

      // update A_minus_id_mmin1
      temp3x3 = A_minus_id_mmin1;
      A_minus_id_mmin1.multiply_nn(1.0, temp3x3, A_minus_id, 0.0);

      // get dA_minus_id_m_dA as a FourTensor
      Core::LinAlg::Voigt::setup_four_tensor_from_9x9_voigt_matrix(
          dA_minus_id_mmin1_dA_FourTensor, dA_minus_id_m_dA);

      // update derivative dA_minus_id_m_dA
      temp9x9.clear();
      Core::LinAlg::Tensor::add_non_symmetric_product(1.0, A_minus_id_mmin1, id_3x3, temp9x9);
      tempFourTensor.clear();
      Core::LinAlg::Voigt::setup_four_tensor_from_9x9_voigt_matrix(tempFourTensor, temp9x9);
      Core::LinAlg::Tensor::multiply_four_tensor_four_tensor(
          leftFourTensor, tempFourTensor, id4_FourTensor, true);
      Core::LinAlg::Voigt::setup_9x9_voigt_matrix_from_four_tensor(temp9x9, leftFourTensor);
      dA_minus_id_m_dA.update(1.0, temp9x9, 0.0);

      temp9x9.clear();
      Core::LinAlg::Tensor::add_non_symmetric_product(1.0, id_3x3, A_minus_id_mmin1, temp9x9);
      tempFourTensor.clear();
      Core::LinAlg::Voigt::setup_four_tensor_from_9x9_voigt_matrix(tempFourTensor, temp9x9);
      Core::LinAlg::Tensor::multiply_four_tensor_four_tensor(
          rightFourTensor, tempFourTensor, dA_minus_id_mmin1_dA_FourTensor, true);
      Core::LinAlg::Voigt::setup_9x9_voigt_matrix_from_four_tensor(temp9x9, rightFourTensor);
      dA_minus_id_m_dA.update(1.0, temp9x9, 1.0);
      // increment m
      m += 1;
    }
    return output;
  }
  // alternatively, employ the Gregory series
  else
  {
    // set initial exponent \f$ m \f$ and the maximum acceptable number of series terms
    int m = 1;
    int m_max = 2 * 50 + 1;

    // \f$ \bm{I} - \bm{A} \f$
    Core::LinAlg::Matrix<3, 3> id_minus_A(Core::LinAlg::Initialization::zero);
    id_minus_A.update(1.0, id_3x3, -1.0, input, 0.0);

    // \f$ \bm{I} + \bm{A} \f$
    Core::LinAlg::Matrix<3, 3> id_plus_A(Core::LinAlg::Initialization::zero);
    id_plus_A.update(1.0, id_3x3, 1.0, input, 0.0);

    // \f$ \left( \bm{I} + \bm{A} \right)^{-1} \f$
    Core::LinAlg::Matrix<3, 3> inv_id_plus_A(Core::LinAlg::Initialization::zero);
    inv_id_plus_A.invert(id_plus_A);

    // \f$ \frac{\partial  \left( \bm{I} + \bm{A} \right)^{-1}}{\partial \bm{A}}  \f$
    Core::LinAlg::Matrix<9, 9> dinv_id_plus_A_dA(Core::LinAlg::Initialization::zero);
    temp9x9.clear();
    Core::LinAlg::Tensor::add_non_symmetric_product(1.0, inv_id_plus_A, inv_id_plus_A, temp9x9);
    dinv_id_plus_A_dA.multiply_nn(-1.0, temp9x9, id4, 0.0);
    Core::LinAlg::FourTensor<3> dinv_id_plus_A_dA_FourTensor(true);
    Core::LinAlg::Voigt::setup_four_tensor_from_9x9_voigt_matrix(
        dinv_id_plus_A_dA_FourTensor, dinv_id_plus_A_dA);

    // update matrix: \f$ \left[ \left( \bm{I} - \bm{A} \right) \left( \bm{I} + \bm{A}
    // \right)^{-1} \right] \f$
    Core::LinAlg::Matrix<3, 3> updateMat(Core::LinAlg::Initialization::zero);
    updateMat.multiply_nn(1.0, id_minus_A, inv_id_plus_A, 0.0);

    // get derivative of the update matrix w.r.t. input matrix \f$ \bm{A} \f$
    Core::LinAlg::Matrix<9, 9> dupdateMat_dA(Core::LinAlg::Initialization::zero);
    temp9x9.clear();
    Core::LinAlg::Tensor::add_non_symmetric_product(1.0, id_minus_A, id_3x3, temp9x9);
    Core::LinAlg::Voigt::setup_four_tensor_from_9x9_voigt_matrix(tempFourTensor, temp9x9);
    Core::LinAlg::Tensor::multiply_four_tensor_four_tensor(
        leftFourTensor, tempFourTensor, dinv_id_plus_A_dA_FourTensor, true);
    Core::LinAlg::Voigt::setup_9x9_voigt_matrix_from_four_tensor(temp9x9, leftFourTensor);
    dupdateMat_dA.update(1.0, temp9x9, 0.0);

    temp9x9.clear();
    Core::LinAlg::Tensor::add_non_symmetric_product(1.0, id_3x3, inv_id_plus_A, temp9x9);
    Core::LinAlg::Voigt::setup_four_tensor_from_9x9_voigt_matrix(tempFourTensor, temp9x9);
    Core::LinAlg::Tensor::multiply_four_tensor_four_tensor(
        rightFourTensor, tempFourTensor, id4_FourTensor, true);
    Core::LinAlg::Voigt::setup_9x9_voigt_matrix_from_four_tensor(temp9x9, rightFourTensor);
    dupdateMat_dA.update(-1.0, temp9x9, 1.0);


    Core::LinAlg::FourTensor<3> dupdateMat_dA_FourTensor(true);
    Core::LinAlg::Voigt::setup_four_tensor_from_9x9_voigt_matrix(
        dupdateMat_dA_FourTensor, dupdateMat_dA);

    // declare first derivatives of the m-th and m-1-th update term
    Core::LinAlg::Matrix<9, 9> dupdateMat_dA_mmin1(Core::LinAlg::Initialization::zero);
    Core::LinAlg::FourTensor<3> dupdateMat_dA_mmin1_FourTensor(true);
    Core::LinAlg::Matrix<9, 9> dupdateMat_dA_m = dupdateMat_dA;
    // \f$ \left[ \left( \bm{I} - \bm{A} \right) \left( \bm{I} + \bm{A} \right)^{-1} \right]^{m-1}
    // \f$
    Core::LinAlg::Matrix<3, 3> updateMat_mmin1 = id_3x3;

    while (true)
    {
      // check whether we have reached the maximum number of terms
      FOUR_C_ASSERT_ALWAYS(
          m <= m_max, "Couldn't compute the matrix logarithm derivative using the Gregory series!");

      // update output matrix
      if ((m % 2 == 1))
      {
        output.update(-2.0 / m, dupdateMat_dA_m, 1.0);
      }

      // update updateMat_mmin1
      temp3x3 = updateMat_mmin1;
      updateMat_mmin1.multiply_nn(1.0, temp3x3, updateMat, 0.0);

      // update derivatives of the previous and current values for the next iteration
      dupdateMat_dA_mmin1 = dupdateMat_dA_m;
      Core::LinAlg::Voigt::setup_four_tensor_from_9x9_voigt_matrix(
          dupdateMat_dA_mmin1_FourTensor, dupdateMat_dA_mmin1);

      temp9x9.clear();
      Core::LinAlg::Tensor::add_non_symmetric_product(1.0, updateMat_mmin1, id_3x3, temp9x9);
      Core::LinAlg::Voigt::setup_four_tensor_from_9x9_voigt_matrix(tempFourTensor, temp9x9);
      Core::LinAlg::Tensor::multiply_four_tensor_four_tensor(
          leftFourTensor, tempFourTensor, dupdateMat_dA_FourTensor, true);
      Core::LinAlg::Voigt::setup_9x9_voigt_matrix_from_four_tensor(temp9x9, leftFourTensor);
      dupdateMat_dA_m.update(1.0, temp9x9, 0.0);

      temp9x9.clear();
      Core::LinAlg::Tensor::add_non_symmetric_product(1.0, id_3x3, updateMat, temp9x9);
      Core::LinAlg::Voigt::setup_four_tensor_from_9x9_voigt_matrix(tempFourTensor, temp9x9);
      Core::LinAlg::Tensor::multiply_four_tensor_four_tensor(
          rightFourTensor, tempFourTensor, dupdateMat_dA_mmin1_FourTensor, true);
      Core::LinAlg::Voigt::setup_9x9_voigt_matrix_from_four_tensor(temp9x9, rightFourTensor);
      dupdateMat_dA_m.update(1.0, temp9x9, 1.0);

      // return if converged
      if ((dupdateMat_dA_m.norm2() < conv_tol) && (m > 1))
      {
        return output;
      }

      // increment m
      m += 1;
    }
  }
}

void Core::LinAlg::sym_matrix_3x3_exp_2nd_deriv_voigt(const Core::LinAlg::Matrix<3, 3>& input,
    Core::LinAlg::Matrix<3, 3>& exp, Core::LinAlg::Matrix<6, 6>& dexp_mat,
    Core::LinAlg::Matrix<6, 6>* ddexp_mat)
{
  // declare arrays of derivatives
  Core::LinAlg::Matrix<3, 3> matrix_exp_1st_deriv[6];
  Core::LinAlg::Matrix<3, 3> matrix_exp_2nd_deriv[6][6];

  // temporary matrices
  Core::LinAlg::Matrix<3, 3> akm(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3> ak(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3> akmd[6];
  Core::LinAlg::Matrix<3, 3> akd[6];
  Core::LinAlg::Matrix<3, 3> akmdd[6][6];
  Core::LinAlg::Matrix<3, 3> akdd[6][6];

  // derivatives of A w.r.t. beta's
  Core::LinAlg::Matrix<3, 3> da[6];
  da[0](0, 0) = 1.;
  da[1](1, 1) = 1.;
  da[2](2, 2) = 1.;
  da[3](0, 1) = da[3](1, 0) = 0.5;
  da[4](1, 2) = da[4](2, 1) = 0.5;
  da[5](0, 2) = da[5](2, 0) = 0.5;

  // prepare
  exp.clear();

  // start with first entry
  int k = 0;
  int kmax = 200;
  for (int i = 0; i < 3; i++) exp(i, i) = 1.;

  // increment
  ++k;
  akm = exp;
  ak.multiply(1. / (double)k, akm, input);
  akd[0](0, 0) = 1.;
  akd[1](1, 1) = 1.;
  akd[2](2, 2) = 1.;
  akd[3](0, 1) = akd[3](1, 0) = 0.5;
  akd[4](1, 2) = akd[4](2, 1) = 0.5;
  akd[5](0, 2) = akd[5](2, 0) = 0.5;

  do
  {
    // add summand
    exp.update(1., ak, 1.);

    // increment
    ++k;
    akm.update(ak);
    ak.multiply(1. / (double)k, akm, input);

    // 1st derivatives
    for (int i = 0; i < 6; i++)
    {
      matrix_exp_1st_deriv[i].update(1., akd[i], 1.);
      // increment
      akmd[i].update(akd[i]);
      akd[i].multiply(1. / (double)k, akm, da[i]);
      akd[i].multiply(1. / (double)k, akmd[i], input, 1.);
    }

    // 2nd derivatives
    for (int i = 0; i < 6; i++)
      for (int j = i; j < 6; j++) matrix_exp_2nd_deriv[i][j].update(1., akdd[i][j], 1.);

    for (int i = 0; i < 6; i++)
      for (int j = i; j < 6; j++)
      {
        // increment
        akmdd[i][j] = akdd[i][j];
        akdd[i][j].multiply(1. / ((double)k), akmd[i], da[j]);
        akdd[i][j].multiply(1. / ((double)k), akmd[j], da[i], 1.);
        akdd[i][j].multiply(1. / ((double)k), akmdd[i][j], input, 1.);
      }

  } while (k < kmax && ak.norm2() > 1.e-16);


  FOUR_C_ASSERT_ALWAYS(k < kmax, "Matrix exponential unconverged with {} summands", k);

  // Additions: 1. Map first derivative from [6](3,3) to (6,6)
  for (int i = 0; i < 6; i++)
    for (int j = 0; j < 3; j++)
      for (int k = j; k < 3; k++)
        dexp_mat(i, vmap::non_symmetric_tensor_to_voigt9_index(j, k)) +=
            matrix_exp_1st_deriv[i](j, k);

  // 2. Map second derivative from [6][6]3x3 to [6]6x6
  for (int I = 0; I < 6; I++)
    for (int J = I; J < 6; J++)
      for (int k = 0; k < 3; k++)
        for (int l = k; l < 3; l++)
        {
          ddexp_mat[I](J, vmap::non_symmetric_tensor_to_voigt9_index(k, l)) =
              matrix_exp_2nd_deriv[I][J](k, l);
          ddexp_mat[J](I, vmap::non_symmetric_tensor_to_voigt9_index(k, l)) =
              matrix_exp_2nd_deriv[I][J](k, l);
        }
  return;
}


// explicit instantiation of template functions
template Core::LinAlg::Matrix<2, 2> Core::LinAlg::matrix_exp<2>(
    const Core::LinAlg::Matrix<2, 2>& input);
template Core::LinAlg::Matrix<3, 3> Core::LinAlg::matrix_exp<3>(
    const Core::LinAlg::Matrix<3, 3>& input);
template Core::LinAlg::Matrix<2, 2> Core::LinAlg::matrix_log<2>(
    const Core::LinAlg::Matrix<2, 2>& input);
template Core::LinAlg::Matrix<3, 3> Core::LinAlg::matrix_log<3>(
    const Core::LinAlg::Matrix<3, 3>& input);

FOUR_C_NAMESPACE_CLOSE

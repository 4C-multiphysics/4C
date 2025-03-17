// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_linalg_utils_densematrix_funct.hpp"

#include "4C_fem_general_cell_type.hpp"
#include "4C_fem_general_utils_integration.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_fixedsizematrix_tensor_derivatives.hpp"
#include "4C_linalg_fixedsizematrix_tensor_products.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_linalg_four_tensor.hpp"
#include "4C_linalg_utils_densematrix_eigen.hpp"
#include "4C_utils_exceptions.hpp"

#include <boost/fusion/view/joint_view/detail/deref_impl.hpp>

#include <iterator>
#include <map>
#include <string>


FOUR_C_NAMESPACE_OPEN

using vmap = Core::LinAlg::Voigt::IndexMappings;

namespace
{
  // theta_m values as shown in Higham: Functions of Matrices, 11.5 Inverse Scaling and Squaring
  // Method, Table 11.1, S. 277
  std::map<unsigned int, double> theta_m_map{{1, 1.10e-5}, {2, 1.82e-3}, {3, 1.62e-2}, {4, 5.39e-2},
      {5, 1.14e-1}, {6, 1.87e-1}, {7, 2.64e-1}, {8, 3.40e-1}, {9, 4.11e-1}, {10, 4.75e-1},
      {11, 5.31e-1}, {12, 5.81e-1}, {13, 6.24e-1}, {14, 6.62e-1}, {15, 6.95e-1}, {16, 7.24e-1}};

}  // namespace


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
template <unsigned int dim>
Core::LinAlg::Matrix<dim, dim> Core::LinAlg::matrix_sqrt(
    const Core::LinAlg::Matrix<dim, dim>& input, Core::LinAlg::MatrixFunctErrorType& err_status,
    unsigned int* num_of_iters, Core::LinAlg::MatrixSqrtCalcMethod calc_method)
{
  // set error status to no errors
  err_status = Core::LinAlg::MatrixFunctErrorType::no_errors;

  // return 0 if matrix is 0: determined with matrix norm
  if (input.norm2() == 0.0)
  {
    err_status = Core::LinAlg::MatrixFunctErrorType::no_errors;
    if (num_of_iters != nullptr) *num_of_iters = 0;
    return input;
  }

  if (calc_method == Core::LinAlg::MatrixSqrtCalcMethod::db_iter_scaled_product)
  {
    // compute dim-dimensional unit tensor
    Core::LinAlg::Matrix<dim, dim> id{true};
    for (unsigned int i = 0; i < dim; ++i) id(i, i) = 1.0;

    // compute exponent for scaling factor \f$ 1/(2 n) \f$
    const double exponent = 1.0 / 2.0 / dim;

    // initialize iteration matrices for the scaled DB iteration in product form, as shown in
    // Higham: Functions of Matrices, Chapter 6: Matrix Square Root, (6.29)
    // _k
    Core::LinAlg::Matrix<dim, dim> M_k{input};
    Core::LinAlg::Matrix<dim, dim> X_k{input};
    Core::LinAlg::Matrix<dim, dim> Y_k{id};
    // _{k+1}
    Core::LinAlg::Matrix<dim, dim> M_kp1{true};
    Core::LinAlg::Matrix<dim, dim> X_kp1{true};
    Core::LinAlg::Matrix<dim, dim> Y_kp1{true};

    // compute inverse of \f$ \boldsymbol{M}_k \f$
    Core::LinAlg::Matrix<dim, dim> invM_k{input};
    invM_k.invert(M_k);

    // compute inverse of \f$ \boldsymbol{X}_k \f$
    Core::LinAlg::Matrix<dim, dim> invX_k{input};
    invX_k.invert(X_k);
    double abs_invX_k = invX_k.norm2();

    // compute determinant of \f$ \boldsymbol{M}_k \f$
    double detM_k = M_k.determinant();
    double inv_abs_detM_k = 1.0 / std::abs(detM_k);

    // compute initial scaling factor \f$ \mu_k \f$
    double mu_k = std::pow(inv_abs_detM_k, exponent);

    // set relative error tolerance \f$ \eta \f$ from stopping criterion Eq. (6.31)
    const double err_tolerance = 1.0e-8;

    // set maximum number of iterations
    const unsigned int max_num_iter = 50;

    // declare distance norm \f$ \| \boldsymbol{X}_{k+1} - \boldsymbol{X}_{k} \| \f$
    Core::LinAlg::Matrix<dim, dim> distance{true};
    double distance_norm = 0.0;

    // iterator integer
    unsigned int iter{0};
    // iterate towards solution
    while (true)
    {
      // increment iterator
      ++iter;

      // check whether iteration number has exceeded its specified maximum
      if (iter > max_num_iter)
      {
        err_status = Core::LinAlg::MatrixFunctErrorType::failed_computation;
        if (num_of_iters != nullptr) *num_of_iters = 0;
        return Core::LinAlg::Matrix<dim, dim>{};
      }

      // compute \f$ \boldsymbol{M}_{k+1} \f$
      M_kp1.update(1.0 / 2.0, id, 0.0);
      M_kp1.update(1.0 / 4.0 * mu_k * mu_k, M_k, 1.0);
      M_kp1.update(1.0 / 4.0 / mu_k / mu_k, invM_k, 1.0);

      // compute \f$ \boldsymbol{X}_{k+1} \f$
      X_kp1.update(1.0 / 2.0 * mu_k, X_k, 0.0);
      X_kp1.multiply(1.0 / 2.0 / mu_k, X_k, invM_k, 1.0);

      // compute \f$ \boldsymbol{Y}_{k+1} \f$
      Y_kp1.update(1.0 / 2.0 * mu_k, Y_k, 0.0);
      Y_kp1.multiply(1.0 / 2.0 / mu_k, Y_k, invM_k, 1.0);

      // compute distance norm (we use the Euclidean 2-norm)
      distance.update(1.0, X_kp1, -1.0, X_k, 0.0);
      distance_norm = distance.norm2();

      // checking stopping criterion
      if (distance_norm * distance_norm < err_tolerance * X_kp1.norm2() / abs_invX_k)
      {
        err_status = MatrixFunctErrorType::no_errors;
        if (num_of_iters != nullptr) *num_of_iters = iter;
        return X_kp1;
      }

      // update relevant matrices and norms
      X_k = X_kp1;
      invX_k.invert(X_k);
      abs_invX_k = invX_k.norm2();
      Y_k = Y_kp1;
      M_k = M_kp1;
      detM_k = M_k.determinant();
      inv_abs_detM_k = 1.0 / std::abs(detM_k);
      invM_k.invert(M_k);
      mu_k = std::pow(inv_abs_detM_k, exponent);
    }
  }
  else
  {
    FOUR_C_THROW("The matrix sqrt function cannot be computed with the specified method yet!");
  }
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
template <unsigned int dim>
Core::LinAlg::Matrix<dim, dim> Core::LinAlg::matrix_exp(const Core::LinAlg::Matrix<dim, dim>& input,
    Core::LinAlg::MatrixFunctErrorType& err_status, Core::LinAlg::MatrixExpCalcMethod calc_method)
{
  // declare output matrix
  Core::LinAlg::Matrix<dim, dim> output(true);

  // get norm of input matrix
  double mat_norm = input.norm2();

  // direct calculation for zero-matrix
  if (mat_norm == 0.)
  {
    for (unsigned int i = 0; i < dim; i++) output(i, i) = 1.;
    err_status = MatrixFunctErrorType::no_errors;
    return output;
  }

  // determine computation method based on the matrix norm
  if (calc_method == Core::LinAlg::MatrixExpCalcMethod::default_method)
  {
    if (mat_norm < 2.0)
      calc_method = Core::LinAlg::MatrixExpCalcMethod::taylor_series;
    else
      calc_method = Core::LinAlg::MatrixExpCalcMethod::spectral_decomp;
  }


  // compute matrix exponential via power series for small matrix norms. This is usually
  // faster than by spectral decomposition for matrices which are close to zero.
  if (calc_method == Core::LinAlg::MatrixExpCalcMethod::taylor_series)
  {
    // consistency check
    if (mat_norm >= 2.0)
    {
      err_status = MatrixFunctErrorType::unsuitable_method;
      return Core::LinAlg::Matrix<dim, dim>{true};
    }

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

    // throw error if no convergence is reached after the maximum number
    // of terms
    if (n > n_max)
    {
      std::cout << "Matrix exponential unconverged in " << n
                << "steps, for the following matrix: " << std::endl;
      input.print(std::cout);
      err_status = MatrixFunctErrorType::failed_computation;
      return output;
    }

    err_status = MatrixFunctErrorType::no_errors;
    return output;
  }
  else if (calc_method == Core::LinAlg::MatrixExpCalcMethod::spectral_decomp)
  {
    // spectral decomposition for higher matrix norms
    Core::LinAlg::Matrix<dim, dim, std::complex<double>> eigenval_matrix(true);
    Core::LinAlg::Matrix<dim, dim, std::complex<double>> eigenvect_matrix(true);
    const Core::LinAlg::Matrix<dim, dim> temp_input(input);
    Core::LinAlg::geev(temp_input, eigenval_matrix, eigenvect_matrix);

    // get the exponentials of the eigenvalues
    for (unsigned int i = 0; i < dim; ++i)
    {
      eigenval_matrix(i, i) = std::exp(eigenval_matrix(i, i));
    }

    // get inverse of the eigenvector matrix
    Core::LinAlg::Matrix<dim, dim, std::complex<double>> inv_eigenvect_matrix(true);
    inv_eigenvect_matrix.invert(eigenvect_matrix);

    // construct the exponential function
    Core::LinAlg::Matrix<dim, dim, std::complex<double>> temp(true);
    Core::LinAlg::Matrix<dim, dim, std::complex<double>> output_complex(true);
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

    err_status = MatrixFunctErrorType::no_errors;
    return output;
  }
  else
  {
    FOUR_C_THROW("The provided computation method for the matrix exponential is not implemented!");
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
template <unsigned int dim>
Core::LinAlg::Matrix<dim, dim> Core::LinAlg::matrix_log(const Core::LinAlg::Matrix<dim, dim>& input,
    Core::LinAlg::MatrixFunctErrorType& err_status, Core::LinAlg::MatrixLogCalcMethod calc_method,
    unsigned int* pade_order)
{
  // auxiliaries
  Core::LinAlg::Matrix<dim, dim> id(true);
  for (unsigned int i = 0; i < dim; ++i)
  {
    id(i, i) = 1.0;
  }
  Core::LinAlg::Matrix<dim, dim> temp(true);

  // initialize error status to no errors(0)
  err_status = MatrixFunctErrorType::no_errors;

  // declare output matrix
  Core::LinAlg::Matrix<dim, dim> output(true);

  // ---> determine matrix logarithm
  // Pade approximation of a given order \f$ m \f$, using a partial fraction form
  if (calc_method == Core::LinAlg::MatrixLogCalcMethod::pade_part_fract)
  {
    // check whether an approximation order \f$ m \f$ was provided
    FOUR_C_ASSERT_ALWAYS(pade_order != nullptr,
        "You want to compute the matrix logarithm with a Pade partial fraction expansion, but have "
        "not provided an approximation order!");

    // subtract identity from input matrix to obtain the \f$ X \f$
    // matrix
    Core::LinAlg::Matrix<dim, dim> X{true};
    X.update(1.0, input, -1.0, id, 0.0);

    // return directly in the case that \f$ X \f$ has norm 0
    if (X.norm2() == 0.0)
    {
      err_status = MatrixFunctErrorType::no_errors;
      return Core::LinAlg::Matrix<dim, dim>{true};
    }

    // initialize m-point Gauss integration points and weights
    FE::IntPointsAndWeights<1> int_pts_wts{
        FE::num_gauss_points_to_gauss_rule<Core::FE::CellType::line2>(*pade_order)};

    // declare \f$ \boldsymbol{I} + \beta_j^{(m)} x  \f$ and its inverse
    Core::LinAlg::Matrix<dim, dim> id_pl_beta_x{true};
    Core::LinAlg::Matrix<dim, dim> inv_id_pl_beta_x{true};

    // declare current Gauss point and Gauss weight (transformed from
    // [-1, 1] onto [0, 1])
    double trafo_point{0.0};
    double trafo_weight{0.0};

    // add terms successively to the output matrix
    unsigned int pt = 0;  // point index
    while (true)
    {
      // transform Gauss points and weights from [-1, 1] onto the considered interval
      // [0, 1]
      trafo_point = 1.0 / 2.0 * (*int_pts_wts.point(pt) + 1.0);
      trafo_weight = 1.0 / 2.0 * int_pts_wts.weight(pt);

      // compute \f$ \boldsymbol{I} + \beta_j^{(m)} x  \f$ and its
      // inverse
      id_pl_beta_x.update(1.0, id, trafo_point, X, 0.0);
      inv_id_pl_beta_x.invert(id_pl_beta_x);

      // add contribution of the Gauss point to the output matrix
      output.multiply(trafo_weight, X, inv_id_pl_beta_x, 1.0);

      // increment point index
      ++pt;

      // exit loop if point index equal to Pade order
      if (pt == (*pade_order))
      {
        err_status = MatrixFunctErrorType::no_errors;
        return output;
      }
    }
  }
  // inverse scaling and squaring method
  else if (calc_method == Core::LinAlg::MatrixLogCalcMethod::inv_scal_square)
  {
    // initialize number of square roots \f$ k \f$, number of iterations for DB iteration \f$
    // \text{it} \f$, and number of norm checks \f$ p \f$
    unsigned int k = 0;
    unsigned int it = 5;
    unsigned int p = 0;

    // use pointer to iteration number it
    unsigned int* it_ptr = &it;

    // declare distance norm to identity tensor \f$ \tau \f$
    double tau{0.0};

    // declare Pade approx. orders \f$ j_1, j_2, m \f$
    unsigned int j_1, j_2, m = 0;

    // declare and initialize \f$ A \f$ (k-th square root of the input matrix)
    Core::LinAlg::Matrix<dim, dim> A{input};

    // loop to determine the matrix logarithm
    while (true)
    {
      // compute distance norm
      temp.update(1.0, A, -1.0, id, 0.0);
      tau = temp.norm1();

      // check whether the distance norm is already smaller than \f$ \theta_{16} \f$
      if (tau < theta_m_map[16])
      {
        // increment number of norm checks
        ++p;

        // compute the relevant Pade approx. order \f$ j_1 \f$
        for (auto theta_m_iter = theta_m_map.rbegin(); theta_m_iter != theta_m_map.rend();
            ++theta_m_iter)
        {
          // account for minimum value of j_1 = 3
          if (theta_m_iter->first < 3)
          {
            j_1 = 3;
            break;
          }


          if (tau > theta_m_iter->second)  // here, we have found the first Pade order j_1
          {
            j_1 = theta_m_iter->first + 1;

            // j_1 is determined -> break out of for-loop
            break;
          }
        }

        // compute the relevant Pade approx. order \f$ j_2 \f$
        for (auto theta_m_iter =
                 std::make_reverse_iterator(theta_m_map.find(static_cast<int>(j_1)));
            theta_m_iter != theta_m_map.rend(); ++theta_m_iter)
        {
          // account for minimum value of j_2 = 3
          if (theta_m_iter->first < 3)
          {
            j_2 = 3;
            break;
          }

          if (tau > theta_m_iter->second)  // here, we have found the first Pade order j_2
          {
            j_2 = theta_m_iter->first + 1;

            // j_2 is determined -> break out of for-loop
            break;
          }
        }

        // check whether loop exit condition is fulfilled
        if ((2 * (j_1 - j_2) <= 3 * (*it_ptr)) || (p == 2))
        {
          // set Pade approximation order \f$ m \f$
          m = j_1;

          // exit out of the loop
          break;
        }
      }

      // take one more square root
      A = matrix_sqrt(
          A, err_status, it_ptr, Core::LinAlg::MatrixSqrtCalcMethod::db_iter_scaled_product);

      // return with error if computation of the matrix sqrt fails
      if (err_status != MatrixFunctErrorType::no_errors)
      {
        return Core::LinAlg::Matrix<dim, dim>{true};
      }

      // increment square root iterator
      k += 1;
    }

    // determine the matrix logarithm of the k-th square root using the Pade approximation of order
    // m
    if (pade_order == nullptr)
    {
      pade_order = &m;  // bind the Pade order pointer to the local variable, which will go out of
                        // scope afterwards
    }
    else
    {
      *pade_order = m;  // save the correct order with the Pade order pointer
    }

    Core::LinAlg::Matrix<dim, dim> k_sqrt_log = matrix_log(A, err_status,
        calc_method = Core::LinAlg::MatrixLogCalcMethod::pade_part_fract, pade_order);

    // return scaled matrix logarithm
    if (err_status == MatrixFunctErrorType::no_errors)
    {
      output.update(std::pow(2.0, k), k_sqrt_log, 0.0);
      return output;
    }
    else
    {
      return Core::LinAlg::Matrix<dim, dim>{true};
    }
  }
  // series descriptions / spectral decomposition
  else
  {
    // set convergence tolerance for the employed series description
    double conv_tol = 1.0e-10;

    // characteristic matrix \f$ \boldsymbol{I} - \boldsymbol{A} \f$
    Core::LinAlg::Matrix<dim, dim> id_minus_A(true);
    id_minus_A.update(1.0, id, -1.0, input, 0.0);

    // \f$ \boldsymbol{I} + \boldsymbol{A} \f$
    Core::LinAlg::Matrix<dim, dim> id_plus_A(true);
    id_plus_A.update(1.0, id, 1.0, input, 0.0);

    // \f$ \left( \boldsymbol{I} + \boldsymbol{A} \right)^{-1} \f$
    Core::LinAlg::Matrix<dim, dim> inv_id_plus_A(true);
    inv_id_plus_A.invert(id_plus_A);

    // update matrix: \f$ \left[ \left( \boldsymbol{I} - \boldsymbol{A} \right) \left(
    // \boldsymbol{I}
    // + \boldsymbol{A}
    // \right)^{-1} \right] \f$
    Core::LinAlg::Matrix<dim, dim> update_mat(true);
    update_mat.multiply(1.0, id_minus_A, inv_id_plus_A, 0.0);

    // determine the computation method based on characteristic matrix norms
    if (calc_method == Core::LinAlg::MatrixLogCalcMethod::default_series)
    {
      // employ Taylor series if matrix norm smaller than 1 for characteristic matrix
      if (id_minus_A.norm2() < 1.0)
      {
        calc_method = Core::LinAlg::MatrixLogCalcMethod::taylor_series;
      }
      // employ the Gregory series, if the norm of the first update matrix is smaller than 1.0
      else if (update_mat.norm2() < 1.0)
      {
        calc_method = Core::LinAlg::MatrixLogCalcMethod::gregory_series;
      }
      // spectral decomposition as the last resort
      else
      {
        calc_method = Core::LinAlg::MatrixLogCalcMethod::spectral_decomp;
      }
    }

    // compute the matrix logarithm
    if (calc_method == Core::LinAlg::MatrixLogCalcMethod::taylor_series)
    {
      // consistency check
      if (id_minus_A.norm2() >= 1.0)
      {
        err_status = MatrixFunctErrorType::unsuitable_method;
        return Core::LinAlg::Matrix<dim, dim>{true};
      }

      // set initial exponent \f$ m \f$ and the maximum acceptable number of series terms
      int m = 1;
      int m_max = 50;

      // \f$ \boldsymbol{A} - \boldsymbol{I} \f$
      Core::LinAlg::Matrix<dim, dim> A_minus_id(true);
      A_minus_id.update(1.0, input, -1.0, id);

      //  \f$ \left(\boldsymbol{A} - \boldsymbol{I}\right)^m \f$
      Core::LinAlg::Matrix<dim, dim> A_minus_id_m = A_minus_id;

      while (A_minus_id_m.norm2() > conv_tol)
      {
        // check whether the maximum number of terms was reached
        if (m > m_max)
        {
          std::cout << "Couldn't compute the matrix logarithm using the Taylor series for the "
                       "following matrix: "
                    << std::endl;
          input.print(std::cout);
          err_status = MatrixFunctErrorType::failed_computation;
          return output;
        }
        FOUR_C_ASSERT_ALWAYS(m <= m_max, "");

        // update output matrix
        output.update(std::pow(-1.0, m + 1) / m, A_minus_id_m, 1.0);

        // update \f$ \left(\boldsymbol{A} - \boldsymbol{I}\right)^m \f$
        temp = A_minus_id_m;
        A_minus_id_m.multiply(1.0, temp, A_minus_id, 0.0);

        // increment m
        m += 1;
      }

      err_status = MatrixFunctErrorType::no_errors;
      return output;
    }
    else if (calc_method == Core::LinAlg::MatrixLogCalcMethod::gregory_series)
    {
      // consistency check
      if (update_mat.norm2() >= 1.0)
      {
        err_status = MatrixFunctErrorType::unsuitable_method;
        return Core::LinAlg::Matrix<dim, dim>{true};
      }

      // set initial exponent \f$ m \f$ and the maximum acceptable number of series terms
      int m = 0;
      int m_max = 50;

      // \f$ \left[ \left( \boldsymbol{I} - \boldsymbol{A} \right) \left( \boldsymbol{I} +
      // \boldsymbol{A} \right)^{-1} \right]^{2m+1}
      // \f$
      Core::LinAlg::Matrix<dim, dim> update_mat_2mpl1(true);
      update_mat_2mpl1 = update_mat;

      output.clear();
      while (update_mat_2mpl1.norm2() > conv_tol)
      {
        // check whether the maximum number of terms was reached
        if (m > m_max)
        {
          std::cout << "Couldn't compute the matrix logarithm using the Gregory series for the "
                       "following matrix: "
                    << std::endl;
          input.print(std::cout);
          err_status = MatrixFunctErrorType::failed_computation;
          return output;
        }

        // update output matrix
        output.update(-2.0 / (2.0 * m + 1.0), update_mat_2mpl1, 1.0);

        // update update_mat_2mpl1
        temp.multiply(1.0, update_mat_2mpl1, update_mat, 0.0);
        update_mat_2mpl1.multiply(1.0, temp, update_mat, 0.0);

        // increment m
        m += 1;
      }

      err_status = MatrixFunctErrorType::no_errors;
      return output;
    }
    else if (calc_method == Core::LinAlg::MatrixLogCalcMethod::spectral_decomp)
    {
      Core::LinAlg::Matrix<dim, dim, std::complex<double>> eigenval_matrix(true);
      Core::LinAlg::Matrix<dim, dim, std::complex<double>> eigenvect_matrix(true);
      Core::LinAlg::Matrix<dim, dim> temp_input(input);
      Core::LinAlg::geev(temp_input, eigenval_matrix, eigenvect_matrix);

      // get the (principal) logarithms of the eigenvalues
      for (unsigned int i = 0; i < dim; ++i)
      {
        if (eigenval_matrix(i, i).real() < 0.0)
        {
          std::cout
              << "The current matrix logarithm implementation only considers the case where all "
                 "eigenvalues "
                 "possess positive real parts! This is not given here, real part of eigenval "
                 "number "
              << i << ": " << eigenval_matrix(i, i).real() << std::endl;
          err_status = MatrixFunctErrorType::failed_computation;
          return output;
        }
        eigenval_matrix(i, i) = std::log(eigenval_matrix(i, i));
      }

      // get inverse of the eigenvector matrix
      Core::LinAlg::Matrix<dim, dim, std::complex<double>> inv_eigenvect_matrix(true);
      inv_eigenvect_matrix.invert(eigenvect_matrix);

      // construct the logarithm function
      Core::LinAlg::Matrix<dim, dim, std::complex<double>> tmp(true);
      Core::LinAlg::Matrix<dim, dim, std::complex<double>> output_complex(true);
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

      err_status = MatrixFunctErrorType::no_errors;
      return output;
    }
    else
    {
      FOUR_C_THROW("The provided computation method for the matrix logarithm is not implemented!");
    }
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Core::LinAlg::Matrix<9, 9> Core::LinAlg::matrix_3x3_exp_1st_deriv(
    const Core::LinAlg::Matrix<3, 3>& input, Core::LinAlg::MatrixFunctErrorType& err_status,
    Core::LinAlg::GenMatrixExpFirstDerivCalcMethod calc_method)
{
  // declare output variable
  Core::LinAlg::Matrix<9, 9> output(true);

  // determine the computation method (currently, only Taylor series implemented)
  if (calc_method == Core::LinAlg::GenMatrixExpFirstDerivCalcMethod::default_method)
  {
    calc_method = Core::LinAlg::GenMatrixExpFirstDerivCalcMethod::taylor_series;
  }

  if (calc_method == Core::LinAlg::GenMatrixExpFirstDerivCalcMethod::taylor_series)
  {
    // see Souza-Neto: Computational Methods for plasticity, Box B.2.
    int nmax = 0;
    int nIter = 0;
    int nfac = 1;
    Core::LinAlg::Matrix<3, 3> tmp1;
    Core::LinAlg::Matrix<3, 3> tmp2(true);
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

    if (nIter > 50)
    {
      std::cout << "Matrix exponential unconverged in " << nIter
                << " steps for the following matrix: " << std::endl;
      input.print(std::cout);
      err_status = MatrixFunctErrorType::failed_computation;
      return output;
    }
    nmax = nIter;

    // compose derivative of matrix exponential (non-symmetric Voigt-notation)
    for (int n = 1; n <= nmax; n++)
      for (int m = 1; m <= n; m++)
        Core::LinAlg::Tensor::add_non_symmetric_product(
            1. / fac[n], Xn.at(m - 1), Xn.at(n - m), output);

    err_status = MatrixFunctErrorType::no_errors;
    return output;
  }
  else
  {
    FOUR_C_THROW(
        "The provided computation method for first derivative of the matrix exponential of a "
        "general matrix is not implemented!");
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Core::LinAlg::Matrix<6, 6> Core::LinAlg::sym_matrix_3x3_exp_1st_deriv(
    const Core::LinAlg::Matrix<3, 3>& input, Core::LinAlg::MatrixFunctErrorType& err_status,
    Core::LinAlg::SymMatrixExpFirstDerivCalcMethod calc_method)
{
  // get matrix norm
  double norm = input.norm2();

  // declare output variable
  Core::LinAlg::Matrix<6, 6> output(true);

  // compute 4-th order identity tensor
  Core::LinAlg::Matrix<6, 6> id4sharp(true);
  for (int i = 0; i < 3; i++) id4sharp(i, i) = 1.0;
  for (int i = 3; i < 6; i++) id4sharp(i, i) = 0.5;

  // direct calculation for zero-matrix
  if (norm == 0.)
  {
    err_status = MatrixFunctErrorType::no_errors;
    return id4sharp;
  }

  // determine computation method based on the matrix norm
  if (calc_method == Core::LinAlg::SymMatrixExpFirstDerivCalcMethod::default_method)
  {
    if (norm < 0.3)
      calc_method = Core::LinAlg::SymMatrixExpFirstDerivCalcMethod::taylor_series;
    else
      calc_method = Core::LinAlg::SymMatrixExpFirstDerivCalcMethod::eigenproj_based;
  }

  // further norm checking to determine appropriate computation procedure
  if (calc_method == Core::LinAlg::SymMatrixExpFirstDerivCalcMethod::taylor_series)
  {
    // consistency check
    if (norm >= 0.3)
    {
      err_status = MatrixFunctErrorType::unsuitable_method;
      return Core::LinAlg::Matrix<6, 6>{true};
    }

    // see Souza-Neto: Computational Methods for plasticity, Box B.2.
    int nmax = 0;
    int nIter = 0;
    int nfac = 1;
    Core::LinAlg::Matrix<3, 3> tmp1;
    Core::LinAlg::Matrix<3, 3> tmp2(true);
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
    if (nIter > 50)
    {
      std::cout << "Matrix exponential unconverged in " << nIter
                << " steps for the following matrix: " << std::endl;
      input.print(std::cout);
      err_status = MatrixFunctErrorType::failed_computation;
      return output;
    }
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
  else if (calc_method == Core::LinAlg::SymMatrixExpFirstDerivCalcMethod::eigenproj_based)
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

    Core::LinAlg::Matrix<3, 3> id2(true);
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
      {
        std::cout << "You should not be here, in the matrix exponential evaluation of: "
                  << std::endl;
        input.print(std::cout);
        err_status = MatrixFunctErrorType::failed_computation;
        return output;
      }

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
    {
      std::cout << "You should not be here, in the matrix exponential evaluation of: " << std::endl;
      input.print(std::cout);
      err_status = MatrixFunctErrorType::failed_computation;
      return output;
    }
  }
  else
  {
    FOUR_C_THROW(
        "The provided computation method for first derivative of the matrix exponential of a "
        "symmetric matrix is not implemented!");
  }
  err_status = MatrixFunctErrorType::no_errors;
  return output;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Core::LinAlg::Matrix<9, 9> Core::LinAlg::matrix_3x3_log_1st_deriv(
    const Core::LinAlg::Matrix<3, 3>& input, Core::LinAlg::MatrixFunctErrorType& err_status,
    Core::LinAlg::GenMatrixLogFirstDerivCalcMethod calc_method, const unsigned int* pade_order)
{
  // auxiliaries
  Core::LinAlg::Matrix<3, 3> id_3x3(true);
  for (int i = 0; i < 3; ++i)
  {
    id_3x3(i, i) = 1.0;
  }
  Core::LinAlg::Matrix<9, 9> id4(true);
  Core::LinAlg::Tensor::add_non_symmetric_product(1.0, id_3x3, id_3x3, id4);
  Core::LinAlg::FourTensor<3> id4_FourTensor(true);
  Core::LinAlg::Voigt::setup_four_tensor_from_9x9_voigt_matrix(id4_FourTensor, id4);
  Core::LinAlg::Matrix<3, 3> temp3x3(true);
  Core::LinAlg::Matrix<9, 9> temp9x9(true);
  Core::LinAlg::FourTensor<3> tempFourTensor(true);
  Core::LinAlg::FourTensor<3> leftFourTensor(true);
  Core::LinAlg::FourTensor<3> rightFourTensor(true);

  // declare output variable
  Core::LinAlg::Matrix<9, 9> output(true);

  // computation using the Pade approximation (partial fraction
  // expansion)
  if (calc_method == Core::LinAlg::GenMatrixLogFirstDerivCalcMethod::pade_part_fract)
  {
    // check whether an approximation order \f$ m \f$ was provided
    FOUR_C_ASSERT_ALWAYS(pade_order != nullptr,
        "You want to compute the matrix logarithm with a Pade partial fraction expansion, but have "
        "not provided an approximation order!");

    // subtract identity from input matrix to obtain the \f$ X \f$
    // matrix
    Core::LinAlg::Matrix<3, 3> X{true};
    X.update(1.0, input, -1.0, id_3x3, 0.0);

    // return directly in the case that \f$ X \f$ has norm 0
    if (X.norm2() < 1.0e-8)
    {
      err_status = MatrixFunctErrorType::no_errors;
      Core::LinAlg::Matrix<9, 9> id9x9{true};
      Tensor::add_non_symmetric_product(1.0, id_3x3, id_3x3, id9x9);
      return id9x9;
    }

    // initialize m-point Gauss integration points and weights
    FE::IntPointsAndWeights<1> int_pts_wts{
        FE::num_gauss_points_to_gauss_rule<Core::FE::CellType::line2>(*pade_order)};

    // declare \f$ \boldsymbol{K} =  \boldsymbol{I} + \beta_j^{(m)} x  \f$ and its inverse
    Core::LinAlg::Matrix<3, 3> K{true};
    Core::LinAlg::Matrix<3, 3> invK{true};

    // declare \f$ \boldsymbol{I}_{AC} \boldsymbol{K}^{-1}_{DB}
    // \f$
    Core::LinAlg::Matrix<9, 9> id_invKT{true};

    // declare \f$ \boldsymbol{X}_{AI} \boldsymbol{K}^{-1}_{IB} \f$
    Core::LinAlg::Matrix<3, 3> XinvK{true};

    // declare \f$ \boldsymbol{X}_{AI} \boldsymbol{K}^{-1}_{IC} \boldsymbol{K}^{-1}_{DB}
    // \f$
    Core::LinAlg::Matrix<9, 9> XinvK_invKT{true};

    // declare current Gauss point and Gauss weight (transformed from
    // [-1, 1] onto [0, 1])
    double trafo_point{0.0};
    double trafo_weight{0.0};

    // add terms successively to the output matrix
    unsigned int pt = 0;  // point index
    while (true)
    {
      // transform Gauss points and weights from [-1, 1] onto the considered interval
      // [0, 1]
      trafo_point = 1.0 / 2.0 * (*int_pts_wts.point(pt) + 1.0);
      trafo_weight = 1.0 / 2.0 * int_pts_wts.weight(pt);

      // compute \f$ \boldsymbol{K}  \f$ and its
      // inverse
      K.update(1.0, id_3x3, trafo_point, X, 0.0);
      invK.invert(K);

      // compute \f$ \boldsymbol{I}_{AC} \boldsymbol{K}^{-1}_{DB}
      // \f$
      id_invKT.clear();
      Tensor::add_non_symmetric_product(1.0, id_3x3, invK, id_invKT);

      // compute \f$ \boldsymbol{X}_{AI} \boldsymbol{K}^{-1}_{IB} \f$
      XinvK.multiply(1.0, X, invK, 0.0);

      // compute \f$ \boldsymbol{X}_{AI} \boldsymbol{K}^{-1}_{IC} \boldsymbol{K}^{-1}_{DB}
      // \f$
      XinvK_invKT.clear();
      Tensor::add_non_symmetric_product(1.0, XinvK, invK, XinvK_invKT);

      // add contribution of the Gauss point to the output matrix
      output.update(-trafo_point * trafo_weight, XinvK_invKT, trafo_weight, id_invKT, 1.0);

      // increment point index
      ++pt;

      // exit loop if point index equal to Pade order
      if (pt == (*pade_order))
      {
        err_status = MatrixFunctErrorType::no_errors;
        return output;
      }
    }
  }
  // computation using Taylor / Gregory series
  else if ((calc_method == GenMatrixLogFirstDerivCalcMethod::default_series) ||
           (calc_method == GenMatrixLogFirstDerivCalcMethod::taylor_series) ||
           (calc_method == GenMatrixLogFirstDerivCalcMethod::gregory_series))
  {
    // set convergence tolerance
    double conv_tol = 1.0e-10;
    // characteristic matrix \f$ \boldsymbol{I} - \boldsymbol{A} \f$
    Core::LinAlg::Matrix<3, 3> id_minus_A(true);
    id_minus_A.update(1.0, id_3x3, -1.0, input, 0.0);

    // determine the computation method based on matrix characteristics
    if (calc_method == Core::LinAlg::GenMatrixLogFirstDerivCalcMethod::default_series)
    {
      if (id_minus_A.norm2() < 1.0)
        calc_method = Core::LinAlg::GenMatrixLogFirstDerivCalcMethod::taylor_series;
      else
        calc_method = Core::LinAlg::GenMatrixLogFirstDerivCalcMethod::gregory_series;
    }

    // employ Taylor series if matrix norm smaller than 1 for characteristic matrix
    if (calc_method == Core::LinAlg::GenMatrixLogFirstDerivCalcMethod::taylor_series)
    {
      // consistency check
      if (id_minus_A.norm2() >= 1.0)
      {
        err_status = MatrixFunctErrorType::unsuitable_method;
        return Core::LinAlg::Matrix<9, 9>{true};
      }

      // set initial exponent \f$ m \f$ and the maximum acceptable number of series terms
      int m = 1;
      int m_max = 50;

      // \f$ \boldsymbol{A} - \boldsymbol{I} \f$
      Core::LinAlg::Matrix<3, 3> A_minus_id(true);
      A_minus_id.update(1.0, input, -1.0, id_3x3);

      //\f$ \left( \bm{A} - \bm{I} \right)^T \f$
      Core::LinAlg::Matrix<3, 3> A_minus_idT(true);
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
        if (m > m_max)
        {
          std::cout
              << "Couldn't compute the matrix logarithm derivative using the Taylor series for "
                 "the following matrix: "
              << std::endl;
          input.print(std::cout);
          err_status = MatrixFunctErrorType::failed_computation;
          return output;
        }

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
      err_status = MatrixFunctErrorType::no_errors;
      return output;
    }
    // alternatively, employ the Gregory series
    else if (calc_method == Core::LinAlg::GenMatrixLogFirstDerivCalcMethod::gregory_series)
    {
      // set initial exponent \f$ m \f$ and the maximum acceptable number of series terms
      int m = 1;
      int m_max = 2 * 50 + 1;

      // \f$ \bm{I} - \bm{A} \f$
      Core::LinAlg::Matrix<3, 3> id_minus_A(true);
      id_minus_A.update(1.0, id_3x3, -1.0, input, 0.0);

      // \f$ \bm{I} + \bm{A} \f$
      Core::LinAlg::Matrix<3, 3> id_plus_A(true);
      id_plus_A.update(1.0, id_3x3, 1.0, input, 0.0);

      // \f$ \left( \bm{I} + \bm{A} \right)^{-1} \f$
      Core::LinAlg::Matrix<3, 3> inv_id_plus_A(true);
      inv_id_plus_A.invert(id_plus_A);

      // \f$ \frac{\partial  \left( \bm{I} + \bm{A} \right)^{-1}}{\partial \bm{A}}  \f$
      Core::LinAlg::Matrix<9, 9> dinv_id_plus_A_dA(true);
      temp9x9.clear();
      Core::LinAlg::Tensor::add_non_symmetric_product(1.0, inv_id_plus_A, inv_id_plus_A, temp9x9);
      dinv_id_plus_A_dA.multiply_nn(-1.0, temp9x9, id4, 0.0);
      Core::LinAlg::FourTensor<3> dinv_id_plus_A_dA_FourTensor(true);
      Core::LinAlg::Voigt::setup_four_tensor_from_9x9_voigt_matrix(
          dinv_id_plus_A_dA_FourTensor, dinv_id_plus_A_dA);

      // update matrix: \f$ \left[ \left( \bm{I} - \bm{A} \right) \left( \bm{I} + \bm{A}
      // \right)^{-1} \right] \f$
      Core::LinAlg::Matrix<3, 3> updateMat(true);
      updateMat.multiply_nn(1.0, id_minus_A, inv_id_plus_A, 0.0);

      // get derivative of the update matrix w.r.t. input matrix \f$ \bm{A} \f$
      Core::LinAlg::Matrix<9, 9> dupdateMat_dA(true);
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
      Core::LinAlg::Matrix<9, 9> dupdateMat_dA_mmin1(true);
      Core::LinAlg::FourTensor<3> dupdateMat_dA_mmin1_FourTensor(true);
      Core::LinAlg::Matrix<9, 9> dupdateMat_dA_m = dupdateMat_dA;
      // \f$ \left[ \left( \bm{I} - \bm{A} \right) \left( \bm{I} + \bm{A} \right)^{-1} \right]^{m-1}
      // \f$
      Core::LinAlg::Matrix<3, 3> updateMat_mmin1 = id_3x3;

      while (true)
      {
        // check whether we have reached the maximum number of terms
        if (m > m_max)
        {
          std::cout
              << "Couldn't compute the matrix logarithm derivative using the Gregory series for "
                 "the following matrix: "
              << std::endl;
          input.print(std::cout);
          err_status = MatrixFunctErrorType::failed_computation;
          return output;
        }


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
          err_status = MatrixFunctErrorType::no_errors;
          return output;
        }

        // increment m
        m += 1;
      }
    }
  }
  else
  {
    FOUR_C_THROW(
        "The provided computation method for first derivative of the matrix logarithm of a "
        "general matrix is not implemented!");
  }

  err_status = MatrixFunctErrorType::unsuitable_method;
  return Core::LinAlg::Matrix<9, 9>{true};
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Core::LinAlg::sym_matrix_3x3_exp_2nd_deriv_voigt(const Core::LinAlg::Matrix<3, 3>& input,
    Core::LinAlg::Matrix<3, 3>& exp, Core::LinAlg::Matrix<6, 6>& dexp_mat,
    Core::LinAlg::Matrix<6, 6>* ddexp_mat, Core::LinAlg::MatrixFunctErrorType& err_status)
{
  // declare arrays of derivatives
  Core::LinAlg::Matrix<3, 3> matrix_exp_1st_deriv[6];
  Core::LinAlg::Matrix<3, 3> matrix_exp_2nd_deriv[6][6];

  // temporary matrices
  Core::LinAlg::Matrix<3, 3> akm(true);
  Core::LinAlg::Matrix<3, 3> ak(true);
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

  if (k > kmax)
  {
    std::cout << "Matrix exponential unconverged with " << k
              << " summands for the following matrix: " << std::endl;
    input.print(std::cout);
    err_status = MatrixFunctErrorType::failed_computation;
    return;
  }



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
  err_status = MatrixFunctErrorType::no_errors;
  return;
}


// explicit instantiation of template functions
template Core::LinAlg::Matrix<2, 2> Core::LinAlg::matrix_sqrt<2>(
    const Core::LinAlg::Matrix<2, 2>& input, Core::LinAlg::MatrixFunctErrorType& err_status,
    unsigned int* num_of_iters, Core::LinAlg::MatrixSqrtCalcMethod calc_method);
template Core::LinAlg::Matrix<3, 3> Core::LinAlg::matrix_sqrt<3>(
    const Core::LinAlg::Matrix<3, 3>& input, Core::LinAlg::MatrixFunctErrorType& err_status,
    unsigned int* num_of_iters, Core::LinAlg::MatrixSqrtCalcMethod calc_method);
template Core::LinAlg::Matrix<2, 2> Core::LinAlg::matrix_exp<2>(
    const Core::LinAlg::Matrix<2, 2>& input, Core::LinAlg::MatrixFunctErrorType& err_status,
    Core::LinAlg::MatrixExpCalcMethod calc_method);
template Core::LinAlg::Matrix<3, 3> Core::LinAlg::matrix_exp<3>(
    const Core::LinAlg::Matrix<3, 3>& input, Core::LinAlg::MatrixFunctErrorType& err_status,
    Core::LinAlg::MatrixExpCalcMethod calc_method);
template Core::LinAlg::Matrix<2, 2> Core::LinAlg::matrix_log<2>(
    const Core::LinAlg::Matrix<2, 2>& input, Core::LinAlg::MatrixFunctErrorType& err_status,
    Core::LinAlg::MatrixLogCalcMethod calc_method, unsigned int* pade_order);
template Core::LinAlg::Matrix<3, 3> Core::LinAlg::matrix_log<3>(
    const Core::LinAlg::Matrix<3, 3>& input, Core::LinAlg::MatrixFunctErrorType& err_status,
    Core::LinAlg::MatrixLogCalcMethod calc_method, unsigned int* pade_order);

FOUR_C_NAMESPACE_CLOSE
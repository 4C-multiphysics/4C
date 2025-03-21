// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/*----------------------------------------------------------------------*/
/*! \file

\brief A collection of methods to evaluate matrix functions, such as the square root, the
exponential and the logarithm, along with specific derivatives in namespace Core::LinAlg

\level 0
*/
/*----------------------------------------------------------------------*/

#ifndef FOUR_C_LINALG_UTILS_DENSEMATRIX_FUNCT_HPP
#define FOUR_C_LINALG_UTILS_DENSEMATRIX_FUNCT_HPP

#include "4C_config.hpp"

#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_utils_densematrix_eigen.hpp"

#include <magic_enum/magic_enum.hpp>

#include <cstddef>
#include <optional>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  /// enum class: error types for the calculation of matrix functions
  enum class MatrixFunctErrorType
  {
    no_errors,           ///< evaluation without errors
    unsuitable_method,   ///< unsuitable computation method (e.g., the matrix norm is too large to
                         ///< use
                         ///< a Taylor series description)
    failed_computation,  ///< although the computation method is suitable, the evaluation fails
                         ///< nonetheless (e.g., the maximum number of
                         ///< iterations is exceeded)
  };

  /// enum class: computation method used for the calculation of the matrix square root
  enum class MatrixSqrtCalcMethod
  {
    db_iter_scaled_product,  ///< Product form of Denman and Beavers iteration (scaled), as
                             ///< described in Higham, Functions of Matrices, Chapter 6: Matrix
                             ///< Square Root, (6.29)
  };

  /*!
   * @name Computes the matrix square root of a given real matrix with a specified computation
   * method.
   *
   *
   * @param[in] input input matrix
   * @param[in,out] err_status Error status in the evaluation
   * @param[in] calc_method utilized computation method
   * @returns matrix square root of the input matrix
   */
  //! @{

  /*!
   * @brief Basic usage
   */
  template <unsigned int dim>
  Matrix<dim, dim> matrix_sqrt(const Matrix<dim, dim>& input, MatrixFunctErrorType& err_status,
      const MatrixSqrtCalcMethod calc_method = MatrixSqrtCalcMethod::db_iter_scaled_product);

  /*! @brief Additionally saves the number of iterations required to
   * compute the matrix square root, in the case of an iterative method
   * @param[out] num_of_iters number of iterations required to compute the matrix square root
   */
  template <unsigned int dim>
  Matrix<dim, dim> matrix_sqrt(const Matrix<dim, dim>& input, MatrixFunctErrorType& err_status,
      unsigned int& num_of_iters,
      const MatrixSqrtCalcMethod calc_method = MatrixSqrtCalcMethod::db_iter_scaled_product);
  //! @}

  /// enum class: computation method used for the calculation of the matrix exponential
  enum class MatrixExpCalcMethod
  {
    default_method,  ///< default computation, employing either of the other methods below depending
                     ///< on different characteristics, such as the matrix norm
    taylor_series,   ///< computation using the Taylor series,
    spectral_decomp,  ///< computation using the spectral decomposition,
  };

  /*!
   * @brief Computes the matrix exponential of a given real matrix, using the Taylor series or a
   * spectral decomposition
   *
   * @note Computation method depends on the norm of the input matrix.
   *
   * @param[in] input input matrix
   * @param[in,out] err_status Error status in the evaluation
   * @param[in] calc_method utilized computation method
   * @returns matrix exponential of the input matrix
   */
  template <unsigned int dim>
  Matrix<dim, dim> matrix_exp(const Matrix<dim, dim>& input, MatrixFunctErrorType& err_status,
      const MatrixExpCalcMethod calc_method = MatrixExpCalcMethod::default_method);

  /// enum class: computation method used for the calculation of the matrix logarithm
  enum class MatrixLogCalcMethod
  {
    default_series,  ///< default series computation, employing one of the series descriptions (or
                     ///< the spectral decomposition) below, depending on different characteristics,
                     ///< such as the matrix norm
    taylor_series,   ///< computation using the Taylor series,
    gregory_series,  ///< computation using the Gregory series,
    spectral_decomp,  ///< computation using the spectral decomposition,
    inv_scal_square,  ///< computation using the inverse scaling and squaring algorithm presented in
                      ///< Higham: Functions of Matrices, Chapter 11: Matrix Logarithm,
                      ///< Algorithm 11.10
    pade_part_fract,  ///< Pade approximation using a partial fraction expansion, as presented in
                      ///< Higham: Functions of Matrices, Chapter 11: Matrix Logarithm, Eq. 11.18
  };


  /*!
   * @name Computes the (principal) matrix logarithm of a given real
   * matrix using a specified computation method
   *
   *
   * For further information, refer to:
   *    -# Higham, Functions of Matrices: Theory and Computation, Society for Industrial and
   * Applied Mathematics, 2008
   *
   * @note The current implementation only takes input matrices into account, where all eigenvalues
   * possess positive real parts.
   *
   * @param[in] input input matrix
   * @param[in,out] err_status Error status in the evaluation
   * @param[in] calc_method utilized computation method
   * @returns principal matrix logarithm of the input matrix
   */
  //! @{
  /*!
   * @brief Basic usage for methods not based on the Pade approximation
   */
  template <unsigned int dim>
  Matrix<dim, dim> matrix_log(const Matrix<dim, dim>& input, MatrixFunctErrorType& err_status,
      const MatrixLogCalcMethod calc_method = MatrixLogCalcMethod::default_series);

  /*!
   * @brief Special usage for methods employing the Pade approximation, where the Pade
   * order can be either an input or/and an output parameter
   *
   * @note If the computation method does not employ a Pade
   * approximation (and thus does not require the Pade order as input/output), then the
   * matrix logarithm is evaluated via the basic usage function and the
   * input Pade order remains unchanged.
   *
   * @param[in, out] pade_order Pade approximation order. This is an input or output parameter
   * depending on the employed method (e.g.: input for Pade
   * approximation with partial fraction expansion (remains the same),
   * output for inverse scaling and squaring method (gets computed))
   */
  template <unsigned int dim>
  Matrix<dim, dim> matrix_log(const Matrix<dim, dim>& input, MatrixFunctErrorType& err_status,
      unsigned int& pade_order,
      const MatrixLogCalcMethod calc_method = MatrixLogCalcMethod::pade_part_fract);
  //! @}

  /// enum class: computation method used for the calculation of the first derivative of the matrix
  /// exponential for a general, not necessarily symmetric matrix
  enum class GenMatrixExpFirstDerivCalcMethod
  {
    default_method,  ///< default computation, employing either of the other methods below depending
                     ///< on different characteristics, such as the matrix norm
    taylor_series,   ///< computation using the Taylor series,
  };

  /*!
   * @brief Computes the first derivative of the matrix exponential (general, not necessarily
   * symmetric 3x3 matrix) with respect to its argument
   *
   * For further information, refer to:
   *    -# deSouza, Computational Methods for Plasticity: Theory and Applications, Wiley & Sons,
   * 2008, Section B.2
   *
   * @param[in] input input 3x3 matrix
   * @param[in,out] err_status Error status in the evaluation
   * @param[in] calc_method utilized computation method
   * @return first derivative of input matrix exponential w.r.t. input matrix, specified in Voigt
   * notation
   */
  Matrix<9, 9> matrix_3x3_exp_1st_deriv(const Matrix<3, 3>& input, MatrixFunctErrorType& err_status,
      GenMatrixExpFirstDerivCalcMethod calc_method =
          GenMatrixExpFirstDerivCalcMethod::default_method);

  /// enum class: computation method used for the calculation of the first derivative of the matrix
  /// logarithm for a general, not necessarily symmetric matrix
  enum class GenMatrixLogFirstDerivCalcMethod
  {
    default_series,  ///< default series computation, employing either of the series below depending
                     ///< on different characteristics, such as the matrix norm
    taylor_series,   ///< computation using the Taylor series,
    gregory_series,  ///< computation using the Gregory series,
    pade_part_fract,  ///< Pade approximation using a partial fraction expansion, as presented in
                      ///< Higham: Functions of Matrices, Chapter 11: Matrix Logarithm, Eq. 11.18
  };

  /*!
   * @name Computes the derivative of the matrix logarithm (general, not necessarily symmetric
   * matrix) with respect to its argument, using either the Taylor series or
   * the Gregory series
   *
   * For further information, refer to:
   *    -# Higham, Functions of Matrices: Theory and Computation, Society for Industrial and Applied
   * Mathematics, 2008
   *
   * @param[in] input input 3x3 matrix
   * @param[in,out] err_status Error status in the evaluation
   * @param[in] calc_method utilized computation method
   * @return derivative of input matrix logarithm w.r.t. input
   * matrix, specified in Voigt notation
   */

  //! @{

  /*! @brief Basic usage for methods not based on the Pade approximation
   */
  Matrix<9, 9> matrix_3x3_log_1st_deriv(const Matrix<3, 3>& input, MatrixFunctErrorType& err_status,
      GenMatrixLogFirstDerivCalcMethod calc_method =
          GenMatrixLogFirstDerivCalcMethod::default_series);

  /*!  @brief Special usage for methods employing the Pade approximation, where the Pade
   * order is also an input parameter
   *
   * @note If the computation method does not employ a Pade
   * approximation (and thus does not require the Pade order as input/output), then the
   * matrix logarithm is evaluated via the basic usage function and the
   * input Pade order remains unchanged.
   *
   * @param[in] pade_order Pade approximation order
   */
  Matrix<9, 9> matrix_3x3_log_1st_deriv(const Matrix<3, 3>& input, MatrixFunctErrorType& err_status,
      const unsigned int pade_order,
      GenMatrixLogFirstDerivCalcMethod calc_method =
          GenMatrixLogFirstDerivCalcMethod::pade_part_fract);

  //! @}

  /// enum class: computation method used for the calculation of the first derivative of the matrix
  /// exponential for a symmetric matrix
  enum class SymMatrixExpFirstDerivCalcMethod
  {
    default_method,  ///< default computation, employing either of the other methods below depending
                     ///< on different characteristics, such as the matrix norm
    taylor_series,   ///< computation using the Taylor series,
    eigenproj_based,  ///< computation based on eigenprojections, as shown in deSouza, Computational
                      ///< Methods for Plasticity: Theory and Applications, Wiley & Sons,
                      ///< 2008, Section A.5
  };

  /*!
   * @brief Computes the derivative of the matrix exponential (symmetric 3x3 matrix) with respect to
   * its argument
   *
   * For further information, refer to:
   *    -# deSouza, Computational Methods for Plasticity: Theory and Applications, Wiley & Sons,
   * 2008, Section A.5 / Box B.2
   *
   * @param[in] input input 3x3 matrix
   * @param[in,out] err_status Error status in the evaluation
   * @param[in] calc_method utilized computation method
   * @return derivative of input matrix exponential w.r.t. input matrix, specified in Voigt
   * stress-stress form
   */
  Matrix<6, 6> sym_matrix_3x3_exp_1st_deriv(const Matrix<3, 3>& input,
      MatrixFunctErrorType& err_status,
      SymMatrixExpFirstDerivCalcMethod calc_method =
          SymMatrixExpFirstDerivCalcMethod::default_method);

  /*!
   * @brief Computes the exponential of a symmetric matrix along with the first and second
   * derivatives with respect to the argument, in Voigt notation
   *
   *  For further information, refer to:
   *    -# Ortiz et al., The computation of the exponential and logarithmic mappings and their first
   * and second linearizations, Int. J. Number. Meth. Engng 52, 2001
   *
   * @param[in] input input 3x3 matrix
   * @param[out] exp exponential of the input matrix
   * @param[out] dexp_mat first derivative of exponential w.r.t. matrix
   * @param[in,out] err_status Error status in the evaluation
   * @param[out] ddexp_mat second derivative of exponential w.r.t. matrix
   * notation
   */
  void sym_matrix_3x3_exp_2nd_deriv_voigt(const Core::LinAlg::Matrix<3, 3>& input,
      Core::LinAlg::Matrix<3, 3>& exp, Core::LinAlg::Matrix<6, 6>& dexp_mat,
      Core::LinAlg::Matrix<6, 6>* ddexp_mat, MatrixFunctErrorType& err_status);

}  // namespace Core::LinAlg

FOUR_C_NAMESPACE_CLOSE

#endif
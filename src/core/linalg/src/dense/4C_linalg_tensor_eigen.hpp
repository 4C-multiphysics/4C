// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINALG_TENSOR_EIGEN_HPP
#define FOUR_C_LINALG_TENSOR_EIGEN_HPP

#include "4C_config.hpp"

#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_linalg_tensor.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_linalg_tensor_svd.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_LAPACK.hpp>

#include <complex>
#include <type_traits>


FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  /*!
   * @brief Computes the (complex) eigenvalue decomposition of a non-symmetric 2-tensor
   *
   * @note This function uses LAPACK's GEEV routine. It will return the (complex) eigenvalues
   * and (complex) eigenvectors of the input tensor.
   *
   * @throw if the tensor is not diagonalizable.
   *
   * @param t Input 2-tensor
   * @tparam Tensor
   * @return A tuple of a std::array containing the complex eigenvalues and a 2-tensor containing
   * the complex eigenvectors.
   */
  template <typename Tensor>
    requires(!is_compressed_tensor<Tensor> && Tensor::rank() == 2 && SquareTensor<Tensor> &&
             std::is_same_v<std::remove_cvref_t<typename Tensor::value_type>, double>)
  auto eig(const Tensor& t)
  {
    auto A = t;

    constexpr std::size_t size = Tensor::template extent<0>();

    Core::LinAlg::Tensor<double, size, size> eigenvectors;

    std::array<double, size> eigenvalues_real;
    std::array<double, size> eigenvalues_imag;
    // ----- perform eigendecomposition ----- //
    const int lwork = 4 * size * size;
    std::array<double, lwork> work;
    int info;
    Teuchos::LAPACK<int, double> lapack;
    lapack.GEEV('N', 'V', size, A.data(), size, eigenvalues_real.data(), eigenvalues_imag.data(),
        nullptr, size, eigenvectors.data(), size, work.data(), lwork, &info);
    FOUR_C_ASSERT_ALWAYS(info == 0, "Eigenvalue decomposition failed with error code {}", info);

    std::array<std::complex<double>, size> eigenvalues_complex;
    Core::LinAlg::Tensor<std::complex<double>, size, size> eigenvectors_complex;

    // transform to complex eigenvalues and eigenvectors
    for (std::size_t i = 0; i < size; ++i)
    {
      if (eigenvalues_imag[i] == 0.0)
      {
        // real eigenvalue and eigenvector
        eigenvalues_complex[i] = std::complex<double>(eigenvalues_real[i], 0.0);
        for (std::size_t j = 0; j < size; ++j)
        {
          eigenvectors_complex(j, i) = std::complex<double>(eigenvectors(j, i), 0.0);
        }
      }
      else
      {
        eigenvalues_complex[i] = std::complex<double>(eigenvalues_real[i], eigenvalues_imag[i]);
        eigenvalues_complex[i + 1] =
            std::complex<double>(eigenvalues_real[i], -eigenvalues_imag[i]);
        for (std::size_t j = 0; j < size; ++j)
        {
          eigenvectors_complex(j, i) =
              std::complex<double>(eigenvectors(j, i), eigenvectors(j, i + 1));
          eigenvectors_complex(j, i + 1) =
              std::complex<double>(eigenvectors(j, i), -eigenvectors(j, i + 1));
        }

        // skip the next eigenvalue (which is the conjugate pair)
        ++i;
      }
    }

    // Note: The determinant is quite sensitive, so we have a quite large tolerance for the
    // determinant. If determinant is smaller than this value, we consider the eigenvector matrix as
    // singular.
    constexpr double determinant_tolerance = 1e-12;
    FOUR_C_ASSERT_ALWAYS(std::abs(Core::LinAlg::det(eigenvectors_complex)) >= determinant_tolerance,
        "The computed eigenvector matrix is (close to) singular (determinant is zero). It looks "
        "like the input matrix is not diagonalizable.");

    return std::make_tuple(eigenvalues_complex, eigenvectors_complex);
  }

  /*!
   * @brief Computes a unit eigenvector for a given eigenvalue of a non-symmetric 2-tensor
   *
   * @note This function uses the singular value decomposition (SVD) to compute the eigenvector
   * corresponding to the provided eigenvalue.
   *
   * @throw if the provided eigenvalue is not an eigenvalue of the given tensor.
   *
   * @tparam Tensor
   * @param t 2-tensor Input 2-tensor
   * @param eigenvalue Eigenvalue for which the eigenvector should be computed
   * @return A unit eigenvector corresponding to the provided eigenvalue.
   */
  template <typename Tensor>
    requires(!is_compressed_tensor<Tensor> && Tensor::rank() == 2 && SquareTensor<Tensor> &&
             std::is_same_v<std::remove_cvref_t<typename Tensor::value_type>, double>)
  auto compute_eigenvector(const Tensor& t, const double eigenvalue)
  {
    constexpr std::size_t size = Tensor::template extent<0>();

    // get singular values of (A-lambda*I)
    auto [_, S, Vh] = Core::LinAlg::svd(
        t - Core::LinAlg::get_full(
                eigenvalue * Core::LinAlg::TensorGenerators::identity<double, size, size>));

    // The smallest singular value should be zero
    constexpr double singular_value_tolerance = 1e-16;
    FOUR_C_ASSERT_ALWAYS(S[size - 1] < singular_value_tolerance,
        "Could not compute eigenvector for eigenvalue {}. The provided eigenvalue is probably not "
        "an eigenvalue of the given matrix!",
        eigenvalue);

    // the eigenvector is the last column of V (corresponding to the smallest singular value)
    Core::LinAlg::Tensor<double, size> eigenvector;
    for (std::size_t i = 0; i < size; ++i) eigenvector(i) = Vh(size - 1, i);

    eigenvector /= Core::LinAlg::norm2(eigenvector);

    return eigenvector;
  }
}  // namespace Core::LinAlg

FOUR_C_NAMESPACE_CLOSE

#endif
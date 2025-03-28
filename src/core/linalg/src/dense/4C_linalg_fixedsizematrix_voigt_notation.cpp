// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"

#include <Sacado.hpp>

FOUR_C_NAMESPACE_OPEN

using NotationType = Core::LinAlg::Voigt::NotationType;

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::LinAlg::Voigt::matrix_3x3_to_9x1(
    Core::LinAlg::Matrix<3, 3> const& in, Core::LinAlg::Matrix<9, 1>& out)
{
  for (int i = 0; i < 3; ++i) out(i) = in(i, i);
  out(3) = in(0, 1);
  out(4) = in(1, 2);
  out(5) = in(0, 2);
  out(6) = in(1, 0);
  out(7) = in(2, 1);
  out(8) = in(2, 0);
}

void Core::LinAlg::Voigt::matrix_9x1_to_3x3(
    Core::LinAlg::Matrix<9, 1> const& in, Core::LinAlg::Matrix<3, 3>& out)
{
  for (int i = 0; i < 3; ++i) out(i, i) = in(i);
  out(0, 1) = in(3);
  out(1, 2) = in(4);
  out(0, 2) = in(5);
  out(1, 0) = in(6);
  out(2, 1) = in(7);
  out(2, 0) = in(8);
}

template <NotationType rows_notation, NotationType cols_notation>
void Core::LinAlg::Voigt::fourth_order_identity_matrix(Core::LinAlg::Matrix<6, 6>& id)
{
  id.clear();

  for (unsigned int i = 0; i < 3; ++i) id(i, i) = 1.0;

  for (unsigned int i = 3; i < 6; ++i)
    id(i, i) = 0.5 * VoigtUtils<rows_notation>::scale_factor(i) *
               VoigtUtils<cols_notation>::scale_factor(i);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Core::LinAlg::Matrix<6, 6> Core::LinAlg::Voigt::modify_voigt_representation(
    const Core::LinAlg::Matrix<6, 6>& input, const double scalar_row, const double scalar_col)
{
  Core::LinAlg::Matrix<6, 6> output(Core::LinAlg::Initialization::zero);

  output(0, 0) = 1.0 * input(0, 0) * 1.0;
  output(0, 1) = 1.0 * input(0, 1) * 1.0;
  output(0, 2) = 1.0 * input(0, 2) * 1.0;
  output(0, 3) = 1.0 * input(0, 3) * scalar_col;
  output(0, 4) = 1.0 * input(0, 4) * scalar_col;
  output(0, 5) = 1.0 * input(0, 5) * scalar_col;

  output(1, 0) = 1.0 * input(1, 0) * 1.0;
  output(1, 1) = 1.0 * input(1, 1) * 1.0;
  output(1, 2) = 1.0 * input(1, 2) * 1.0;
  output(1, 3) = 1.0 * input(1, 3) * scalar_col;
  output(1, 4) = 1.0 * input(1, 4) * scalar_col;
  output(1, 5) = 1.0 * input(1, 5) * scalar_col;

  output(2, 0) = 1.0 * input(2, 0) * 1.0;
  output(2, 1) = 1.0 * input(2, 1) * 1.0;
  output(2, 2) = 1.0 * input(2, 2) * 1.0;
  output(2, 3) = 1.0 * input(2, 3) * scalar_col;
  output(2, 4) = 1.0 * input(2, 4) * scalar_col;
  output(2, 5) = 1.0 * input(2, 5) * scalar_col;

  output(3, 0) = scalar_row * input(3, 0) * 1.0;
  output(3, 1) = scalar_row * input(3, 1) * 1.0;
  output(3, 2) = scalar_row * input(3, 2) * 1.0;
  output(3, 3) = scalar_row * input(3, 3) * scalar_col;
  output(3, 4) = scalar_row * input(3, 4) * scalar_col;
  output(3, 5) = scalar_row * input(3, 5) * scalar_col;

  output(4, 0) = scalar_row * input(4, 0) * 1.0;
  output(4, 1) = scalar_row * input(4, 1) * 1.0;
  output(4, 2) = scalar_row * input(4, 2) * 1.0;
  output(4, 3) = scalar_row * input(4, 3) * scalar_col;
  output(4, 4) = scalar_row * input(4, 4) * scalar_col;
  output(4, 5) = scalar_row * input(4, 5) * scalar_col;

  output(5, 0) = scalar_row * input(5, 0) * 1.0;
  output(5, 1) = scalar_row * input(5, 1) * 1.0;
  output(5, 2) = scalar_row * input(5, 2) * 1.0;
  output(5, 3) = scalar_row * input(5, 3) * scalar_col;
  output(5, 4) = scalar_row * input(5, 4) * scalar_col;
  output(5, 5) = scalar_row * input(5, 5) * scalar_col;

  return output;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
template <NotationType type>
void Core::LinAlg::Voigt::VoigtUtils<type>::symmetric_outer_product(
    const Core::LinAlg::Matrix<3, 1>& vec_a, const Core::LinAlg::Matrix<3, 1>& vec_b,
    Core::LinAlg::Matrix<6, 1>& ab_ba)
{
  std::fill(ab_ba.data(), ab_ba.data() + 6, 0.0);

  Core::LinAlg::Matrix<3, 3> outer_product;
  outer_product.multiply_nt(vec_a, vec_b);

  for (unsigned i = 0; i < 3; ++i)
    for (unsigned j = i; j < 3; ++j)
      ab_ba(IndexMappings::symmetric_tensor_to_voigt6_index(i, j)) +=
          outer_product(i, j) + outer_product(j, i);

  // scale off-diagonal values
  scale_off_diagonal_vals(ab_ba);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
template <NotationType type>
void Core::LinAlg::Voigt::VoigtUtils<type>::multiply_tensor_vector(
    const Core::LinAlg::Matrix<6, 1>& strain, const Core::LinAlg::Matrix<3, 1>& vec,
    Core::LinAlg::Matrix<3, 1>& res)
{
  for (unsigned i = 0; i < 3; ++i)
  {
    for (unsigned j = 0; j < 3; ++j)
    {
      const double fac = unscale_factor(IndexMappings::symmetric_tensor_to_voigt6_index(i, j));
      res(i, 0) += strain(IndexMappings::symmetric_tensor_to_voigt6_index(i, j)) * fac * vec(j, 0);
    }
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
template <NotationType type>
void Core::LinAlg::Voigt::VoigtUtils<type>::power_of_symmetric_tensor(const unsigned pow,
    const Core::LinAlg::Matrix<6, 1>& strain, Core::LinAlg::Matrix<6, 1>& strain_pow)
{
  std::copy(strain.data(), strain.data() + 6, strain_pow.data());

  if (pow > 1)
  {
    // unscale the off-diagonal values
    unscale_off_diagonal_vals(strain_pow);

    Core::LinAlg::Matrix<6, 1> prod(Core::LinAlg::Initialization::uninitialized);

    for (unsigned p = 1; p < pow; ++p)
    {
      std::fill(prod.data(), prod.data() + 6, 0.0);

      for (unsigned i = 0; i < 3; ++i)
      {
        for (unsigned j = i; j < 3; ++j)
        {
          for (unsigned k = 0; k < 3; ++k)
          {
            prod(IndexMappings::symmetric_tensor_to_voigt6_index(i, j), 0) +=
                strain_pow(IndexMappings::symmetric_tensor_to_voigt6_index(i, k), 0) *
                unscale_fac_[IndexMappings::symmetric_tensor_to_voigt6_index(k, j)] *
                strain(IndexMappings::symmetric_tensor_to_voigt6_index(k, j), 0);
          }
        }
      }

      std::copy(prod.data(), prod.data() + 6, strain_pow.data());
    }

    // scale the off-diagonal values again
    scale_off_diagonal_vals(strain_pow);
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
template <NotationType type>
void Core::LinAlg::Voigt::VoigtUtils<type>::inverse_tensor(
    const Core::LinAlg::Matrix<6, 1>& tens, Core::LinAlg::Matrix<6, 1>& tens_inv)
{
  double det = determinant(tens);
  tens_inv(0) = (tens(1) * tens(2) - unscale_factor(4) * unscale_factor(4) * tens(4) * tens(4)) /
                det * scale_factor(0);
  tens_inv(1) = (tens(0) * tens(2) - unscale_factor(5) * unscale_factor(5) * tens(5) * tens(5)) /
                det * scale_factor(1);
  tens_inv(2) = (tens(0) * tens(1) - unscale_factor(3) * unscale_factor(3) * tens(3) * tens(3)) /
                det * scale_factor(2);
  tens_inv(3) = (unscale_factor(5) * unscale_factor(4) * tens(5) * tens(4) -
                    unscale_factor(3) * unscale_factor(2) * tens(3) * tens(2)) /
                det * scale_factor(3);
  tens_inv(4) = (unscale_factor(3) * unscale_factor(5) * tens(3) * tens(5) -
                    unscale_factor(0) * unscale_factor(4) * tens(0) * tens(4)) /
                det * scale_factor(4);
  tens_inv(5) = (unscale_factor(3) * unscale_factor(4) * tens(3) * tens(4) -
                    unscale_factor(5) * unscale_factor(1) * tens(5) * tens(1)) /
                det * scale_factor(5);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
template <NotationType type>
void Core::LinAlg::Voigt::VoigtUtils<type>::to_stress_like(
    const Core::LinAlg::Matrix<6, 1>& vtensor_in, Core::LinAlg::Matrix<6, 1>& vtensor_out)
{
  for (unsigned i = 0; i < 6; ++i) vtensor_out(i) = unscale_factor(i) * vtensor_in(i);
}

template <NotationType type>
void Core::LinAlg::Voigt::VoigtUtils<type>::to_strain_like(
    const Core::LinAlg::Matrix<6, 1>& vtensor_in, Core::LinAlg::Matrix<6, 1>& vtensor_out)
{
  for (unsigned i = 0; i < 6; ++i)
    vtensor_out(i) = unscale_factor(i) * vtensor_in(i) * Strains::scale_factor(i);
}

template <NotationType type>
void Core::LinAlg::Voigt::VoigtUtils<type>::vector_to_matrix(
    const Core::LinAlg::Matrix<6, 1>& vtensor_in, Core::LinAlg::Matrix<3, 3>& tensor_out)
{
  for (int i = 0; i < 3; ++i) tensor_out(i, i) = vtensor_in(i);
  tensor_out(0, 1) = tensor_out(1, 0) = unscale_factor(3) * vtensor_in(3);
  tensor_out(1, 2) = tensor_out(2, 1) = unscale_factor(4) * vtensor_in(4);
  tensor_out(0, 2) = tensor_out(2, 0) = unscale_factor(5) * vtensor_in(5);
}

template <NotationType type>
template <typename T>
void Core::LinAlg::Voigt::VoigtUtils<type>::matrix_to_vector(
    const Core::LinAlg::Matrix<3, 3, T>& tensor_in, Core::LinAlg::Matrix<6, 1, T>& vtensor_out)
{
  for (int i = 0; i < 3; ++i) vtensor_out(i) = tensor_in(i, i);
  vtensor_out(3) = 0.5 * scale_factor(3) * (tensor_in(0, 1) + tensor_in(1, 0));
  vtensor_out(4) = 0.5 * scale_factor(4) * (tensor_in(1, 2) + tensor_in(2, 1));
  vtensor_out(5) = 0.5 * scale_factor(5) * (tensor_in(0, 2) + tensor_in(2, 0));
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
template <int dim>
void Core::LinAlg::Voigt::setup_four_tensor_from_6x6_voigt_matrix(
    Core::LinAlg::FourTensor<dim>& four_tensor, const Core::LinAlg::Matrix<6, 6>& matrix_voigt)
{
  static_assert(dim == 3, "Current implementation only valid for dim = 3.");

  // Setup 4-Tensor from 6x6 Voigt matrix (which has to be the representative of a 4 tensor with at
  // least minor symmetries)
  four_tensor(0, 0, 0, 0) = matrix_voigt(0, 0);  // C1111
  four_tensor(0, 0, 1, 1) = matrix_voigt(0, 1);  // C1122
  four_tensor(0, 0, 2, 2) = matrix_voigt(0, 2);  // C1133
  four_tensor(0, 0, 0, 1) = matrix_voigt(0, 3);  // C1112
  four_tensor(0, 0, 1, 0) = matrix_voigt(0, 3);  // C1121
  four_tensor(0, 0, 1, 2) = matrix_voigt(0, 4);  // C1123
  four_tensor(0, 0, 2, 1) = matrix_voigt(0, 4);  // C1132
  four_tensor(0, 0, 0, 2) = matrix_voigt(0, 5);  // C1113
  four_tensor(0, 0, 2, 0) = matrix_voigt(0, 5);  // C1131

  four_tensor(1, 1, 0, 0) = matrix_voigt(1, 0);  // C2211
  four_tensor(1, 1, 1, 1) = matrix_voigt(1, 1);  // C2222
  four_tensor(1, 1, 2, 2) = matrix_voigt(1, 2);  // C2233
  four_tensor(1, 1, 0, 1) = matrix_voigt(1, 3);  // C2212
  four_tensor(1, 1, 1, 0) = matrix_voigt(1, 3);  // C2221
  four_tensor(1, 1, 1, 2) = matrix_voigt(1, 4);  // C2223
  four_tensor(1, 1, 2, 1) = matrix_voigt(1, 4);  // C2232
  four_tensor(1, 1, 0, 2) = matrix_voigt(1, 5);  // C2213
  four_tensor(1, 1, 2, 0) = matrix_voigt(1, 5);  // C2231

  four_tensor(2, 2, 0, 0) = matrix_voigt(2, 0);  // C3311
  four_tensor(2, 2, 1, 1) = matrix_voigt(2, 1);  // C3322
  four_tensor(2, 2, 2, 2) = matrix_voigt(2, 2);  // C3333
  four_tensor(2, 2, 0, 1) = matrix_voigt(2, 3);  // C3312
  four_tensor(2, 2, 1, 0) = matrix_voigt(2, 3);  // C3321
  four_tensor(2, 2, 1, 2) = matrix_voigt(2, 4);  // C3323
  four_tensor(2, 2, 2, 1) = matrix_voigt(2, 4);  // C3332
  four_tensor(2, 2, 0, 2) = matrix_voigt(2, 5);  // C3313
  four_tensor(2, 2, 2, 0) = matrix_voigt(2, 5);  // C3331

  four_tensor(0, 1, 0, 0) = matrix_voigt(3, 0);
  four_tensor(1, 0, 0, 0) = matrix_voigt(3, 0);  // C1211 = C2111
  four_tensor(0, 1, 1, 1) = matrix_voigt(3, 1);
  four_tensor(1, 0, 1, 1) = matrix_voigt(3, 1);  // C1222 = C2122
  four_tensor(0, 1, 2, 2) = matrix_voigt(3, 2);
  four_tensor(1, 0, 2, 2) = matrix_voigt(3, 2);  // C1233 = C2133
  four_tensor(0, 1, 0, 1) = matrix_voigt(3, 3);
  four_tensor(1, 0, 0, 1) = matrix_voigt(3, 3);  // C1212 = C2112
  four_tensor(0, 1, 1, 0) = matrix_voigt(3, 3);
  four_tensor(1, 0, 1, 0) = matrix_voigt(3, 3);  // C1221 = C2121
  four_tensor(0, 1, 1, 2) = matrix_voigt(3, 4);
  four_tensor(1, 0, 1, 2) = matrix_voigt(3, 4);  // C1223 = C2123
  four_tensor(0, 1, 2, 1) = matrix_voigt(3, 4);
  four_tensor(1, 0, 2, 1) = matrix_voigt(3, 4);  // C1232 = C2132
  four_tensor(0, 1, 0, 2) = matrix_voigt(3, 5);
  four_tensor(1, 0, 0, 2) = matrix_voigt(3, 5);  // C1213 = C2113
  four_tensor(0, 1, 2, 0) = matrix_voigt(3, 5);
  four_tensor(1, 0, 2, 0) = matrix_voigt(3, 5);  // C1231 = C2131

  four_tensor(1, 2, 0, 0) = matrix_voigt(4, 0);
  four_tensor(2, 1, 0, 0) = matrix_voigt(4, 0);  // C2311 = C3211
  four_tensor(1, 2, 1, 1) = matrix_voigt(4, 1);
  four_tensor(2, 1, 1, 1) = matrix_voigt(4, 1);  // C2322 = C3222
  four_tensor(1, 2, 2, 2) = matrix_voigt(4, 2);
  four_tensor(2, 1, 2, 2) = matrix_voigt(4, 2);  // C2333 = C3233
  four_tensor(1, 2, 0, 1) = matrix_voigt(4, 3);
  four_tensor(2, 1, 0, 1) = matrix_voigt(4, 3);  // C2312 = C3212
  four_tensor(1, 2, 1, 0) = matrix_voigt(4, 3);
  four_tensor(2, 1, 1, 0) = matrix_voigt(4, 3);  // C2321 = C3221
  four_tensor(1, 2, 1, 2) = matrix_voigt(4, 4);
  four_tensor(2, 1, 1, 2) = matrix_voigt(4, 4);  // C2323 = C3223
  four_tensor(1, 2, 2, 1) = matrix_voigt(4, 4);
  four_tensor(2, 1, 2, 1) = matrix_voigt(4, 4);  // C2332 = C3232
  four_tensor(1, 2, 0, 2) = matrix_voigt(4, 5);
  four_tensor(2, 1, 0, 2) = matrix_voigt(4, 5);  // C2313 = C3213
  four_tensor(1, 2, 2, 0) = matrix_voigt(4, 5);
  four_tensor(2, 1, 2, 0) = matrix_voigt(4, 5);  // C2331 = C3231

  four_tensor(0, 2, 0, 0) = matrix_voigt(5, 0);
  four_tensor(2, 0, 0, 0) = matrix_voigt(5, 0);  // C1311 = C3111
  four_tensor(0, 2, 1, 1) = matrix_voigt(5, 1);
  four_tensor(2, 0, 1, 1) = matrix_voigt(5, 1);  // C1322 = C3122
  four_tensor(0, 2, 2, 2) = matrix_voigt(5, 2);
  four_tensor(2, 0, 2, 2) = matrix_voigt(5, 2);  // C1333 = C3133
  four_tensor(0, 2, 0, 1) = matrix_voigt(5, 3);
  four_tensor(2, 0, 0, 1) = matrix_voigt(5, 3);  // C1312 = C3112
  four_tensor(0, 2, 1, 0) = matrix_voigt(5, 3);
  four_tensor(2, 0, 1, 0) = matrix_voigt(5, 3);  // C1321 = C3121
  four_tensor(0, 2, 1, 2) = matrix_voigt(5, 4);
  four_tensor(2, 0, 1, 2) = matrix_voigt(5, 4);  // C1323 = C3123
  four_tensor(0, 2, 2, 1) = matrix_voigt(5, 4);
  four_tensor(2, 0, 2, 1) = matrix_voigt(5, 4);  // C1332 = C3132
  four_tensor(0, 2, 0, 2) = matrix_voigt(5, 5);
  four_tensor(2, 0, 0, 2) = matrix_voigt(5, 5);  // C1313 = C3113
  four_tensor(0, 2, 2, 0) = matrix_voigt(5, 5);
  four_tensor(2, 0, 2, 0) = matrix_voigt(5, 5);  // C1331 = C3131
}

template <int dim>
void Core::LinAlg::Voigt::setup_four_tensor_from_6x9_voigt_matrix(
    Core::LinAlg::FourTensor<dim>& fourTensor, const Core::LinAlg::Matrix<6, 9>& matrixVoigt)
{
  static_assert(dim == 3, "Current implementation only valid for dim = 3.");

  // Setup 4-Tensor from 9x9 Voigt matrix
  fourTensor(0, 0, 0, 0) = matrixVoigt(0, 0);
  fourTensor(0, 0, 1, 1) = matrixVoigt(0, 1);
  fourTensor(0, 0, 2, 2) = matrixVoigt(0, 2);
  fourTensor(0, 0, 0, 1) = matrixVoigt(0, 3);
  fourTensor(0, 0, 1, 0) = matrixVoigt(0, 6);
  fourTensor(0, 0, 1, 2) = matrixVoigt(0, 4);
  fourTensor(0, 0, 2, 1) = matrixVoigt(0, 7);
  fourTensor(0, 0, 0, 2) = matrixVoigt(0, 5);
  fourTensor(0, 0, 2, 0) = matrixVoigt(0, 8);

  fourTensor(1, 1, 0, 0) = matrixVoigt(1, 0);
  fourTensor(1, 1, 1, 1) = matrixVoigt(1, 1);
  fourTensor(1, 1, 2, 2) = matrixVoigt(1, 2);
  fourTensor(1, 1, 0, 1) = matrixVoigt(1, 3);
  fourTensor(1, 1, 1, 0) = matrixVoigt(1, 6);
  fourTensor(1, 1, 1, 2) = matrixVoigt(1, 4);
  fourTensor(1, 1, 2, 1) = matrixVoigt(1, 7);
  fourTensor(1, 1, 0, 2) = matrixVoigt(1, 5);
  fourTensor(1, 1, 2, 0) = matrixVoigt(1, 8);

  fourTensor(2, 2, 0, 0) = matrixVoigt(2, 0);
  fourTensor(2, 2, 1, 1) = matrixVoigt(2, 1);
  fourTensor(2, 2, 2, 2) = matrixVoigt(2, 2);
  fourTensor(2, 2, 0, 1) = matrixVoigt(2, 3);
  fourTensor(2, 2, 1, 0) = matrixVoigt(2, 6);
  fourTensor(2, 2, 1, 2) = matrixVoigt(2, 4);
  fourTensor(2, 2, 2, 1) = matrixVoigt(2, 7);
  fourTensor(2, 2, 0, 2) = matrixVoigt(2, 5);
  fourTensor(2, 2, 2, 0) = matrixVoigt(2, 8);

  fourTensor(0, 1, 0, 0) = matrixVoigt(3, 0);
  fourTensor(1, 0, 0, 0) = matrixVoigt(3, 0);
  fourTensor(0, 1, 1, 1) = matrixVoigt(3, 1);
  fourTensor(1, 0, 1, 1) = matrixVoigt(3, 1);
  fourTensor(0, 1, 2, 2) = matrixVoigt(3, 2);
  fourTensor(1, 0, 2, 2) = matrixVoigt(3, 2);
  fourTensor(0, 1, 0, 1) = matrixVoigt(3, 3);
  fourTensor(1, 0, 0, 1) = matrixVoigt(3, 3);
  fourTensor(0, 1, 1, 0) = matrixVoigt(3, 6);
  fourTensor(1, 0, 1, 0) = matrixVoigt(3, 6);
  fourTensor(0, 1, 1, 2) = matrixVoigt(3, 4);
  fourTensor(1, 0, 1, 2) = matrixVoigt(3, 4);
  fourTensor(0, 1, 2, 1) = matrixVoigt(3, 7);
  fourTensor(1, 0, 2, 1) = matrixVoigt(3, 7);
  fourTensor(0, 1, 0, 2) = matrixVoigt(3, 5);
  fourTensor(1, 0, 0, 2) = matrixVoigt(3, 5);
  fourTensor(0, 1, 2, 0) = matrixVoigt(3, 8);
  fourTensor(1, 0, 2, 0) = matrixVoigt(3, 8);

  fourTensor(1, 2, 0, 0) = matrixVoigt(4, 0);
  fourTensor(2, 1, 0, 0) = matrixVoigt(4, 0);
  fourTensor(1, 2, 1, 1) = matrixVoigt(4, 1);
  fourTensor(2, 1, 1, 1) = matrixVoigt(4, 1);
  fourTensor(1, 2, 2, 2) = matrixVoigt(4, 2);
  fourTensor(2, 1, 2, 2) = matrixVoigt(4, 2);
  fourTensor(1, 2, 0, 1) = matrixVoigt(4, 3);
  fourTensor(2, 1, 0, 1) = matrixVoigt(4, 3);
  fourTensor(1, 2, 1, 0) = matrixVoigt(4, 6);
  fourTensor(2, 1, 1, 0) = matrixVoigt(4, 6);
  fourTensor(1, 2, 1, 2) = matrixVoigt(4, 4);
  fourTensor(2, 1, 1, 2) = matrixVoigt(4, 4);
  fourTensor(1, 2, 2, 1) = matrixVoigt(4, 7);
  fourTensor(2, 1, 2, 1) = matrixVoigt(4, 7);
  fourTensor(1, 2, 0, 2) = matrixVoigt(4, 5);
  fourTensor(2, 1, 0, 2) = matrixVoigt(4, 5);
  fourTensor(1, 2, 2, 0) = matrixVoigt(4, 8);
  fourTensor(2, 1, 2, 0) = matrixVoigt(4, 8);

  fourTensor(0, 2, 0, 0) = matrixVoigt(5, 0);
  fourTensor(2, 0, 0, 0) = matrixVoigt(5, 0);
  fourTensor(0, 2, 1, 1) = matrixVoigt(5, 1);
  fourTensor(2, 0, 1, 1) = matrixVoigt(5, 1);
  fourTensor(0, 2, 2, 2) = matrixVoigt(5, 2);
  fourTensor(2, 0, 2, 2) = matrixVoigt(5, 2);
  fourTensor(0, 2, 0, 1) = matrixVoigt(5, 3);
  fourTensor(2, 0, 0, 1) = matrixVoigt(5, 3);
  fourTensor(0, 2, 1, 0) = matrixVoigt(5, 6);
  fourTensor(2, 0, 1, 0) = matrixVoigt(5, 6);
  fourTensor(0, 2, 1, 2) = matrixVoigt(5, 4);
  fourTensor(2, 0, 1, 2) = matrixVoigt(5, 4);
  fourTensor(0, 2, 2, 1) = matrixVoigt(5, 7);
  fourTensor(2, 0, 2, 1) = matrixVoigt(5, 7);
  fourTensor(0, 2, 0, 2) = matrixVoigt(5, 5);
  fourTensor(2, 0, 0, 2) = matrixVoigt(5, 5);
  fourTensor(0, 2, 2, 0) = matrixVoigt(5, 8);
  fourTensor(2, 0, 2, 0) = matrixVoigt(5, 8);
}

template <int dim>
void Core::LinAlg::Voigt::setup_four_tensor_from_9x6_voigt_matrix(
    Core::LinAlg::FourTensor<dim>& fourTensor, const Core::LinAlg::Matrix<9, 6>& matrixVoigt)
{
  static_assert(dim == 3, "Current implementation only valid for dim = 3.");

  // Setup 4-Tensor from 9x6 Voigt matrix

  fourTensor(0, 0, 0, 0) = matrixVoigt(0, 0);
  fourTensor(0, 0, 1, 1) = matrixVoigt(0, 1);
  fourTensor(0, 0, 2, 2) = matrixVoigt(0, 2);
  fourTensor(0, 0, 0, 1) = matrixVoigt(0, 3);
  fourTensor(0, 0, 1, 0) = matrixVoigt(0, 3);
  fourTensor(0, 0, 1, 2) = matrixVoigt(0, 4);
  fourTensor(0, 0, 2, 1) = matrixVoigt(0, 4);
  fourTensor(0, 0, 0, 2) = matrixVoigt(0, 5);
  fourTensor(0, 0, 2, 0) = matrixVoigt(0, 5);

  fourTensor(1, 1, 0, 0) = matrixVoigt(1, 0);
  fourTensor(1, 1, 1, 1) = matrixVoigt(1, 1);
  fourTensor(1, 1, 2, 2) = matrixVoigt(1, 2);
  fourTensor(1, 1, 0, 1) = matrixVoigt(1, 3);
  fourTensor(1, 1, 1, 0) = matrixVoigt(1, 3);
  fourTensor(1, 1, 1, 2) = matrixVoigt(1, 4);
  fourTensor(1, 1, 2, 1) = matrixVoigt(1, 4);
  fourTensor(1, 1, 0, 2) = matrixVoigt(1, 5);
  fourTensor(1, 1, 2, 0) = matrixVoigt(1, 5);

  fourTensor(2, 2, 0, 0) = matrixVoigt(2, 0);
  fourTensor(2, 2, 1, 1) = matrixVoigt(2, 1);
  fourTensor(2, 2, 2, 2) = matrixVoigt(2, 2);
  fourTensor(2, 2, 0, 1) = matrixVoigt(2, 3);
  fourTensor(2, 2, 1, 0) = matrixVoigt(2, 3);
  fourTensor(2, 2, 1, 2) = matrixVoigt(2, 4);
  fourTensor(2, 2, 2, 1) = matrixVoigt(2, 4);
  fourTensor(2, 2, 0, 2) = matrixVoigt(2, 5);
  fourTensor(2, 2, 2, 0) = matrixVoigt(2, 5);

  fourTensor(0, 1, 0, 0) = matrixVoigt(3, 0);
  fourTensor(1, 0, 0, 0) = matrixVoigt(6, 0);
  fourTensor(0, 1, 1, 1) = matrixVoigt(3, 1);
  fourTensor(1, 0, 1, 1) = matrixVoigt(6, 1);
  fourTensor(0, 1, 2, 2) = matrixVoigt(3, 2);
  fourTensor(1, 0, 2, 2) = matrixVoigt(6, 2);
  fourTensor(0, 1, 0, 1) = matrixVoigt(3, 3);
  fourTensor(1, 0, 0, 1) = matrixVoigt(6, 3);
  fourTensor(0, 1, 1, 0) = matrixVoigt(3, 3);
  fourTensor(1, 0, 1, 0) = matrixVoigt(6, 3);
  fourTensor(0, 1, 1, 2) = matrixVoigt(3, 4);
  fourTensor(1, 0, 1, 2) = matrixVoigt(6, 4);
  fourTensor(0, 1, 2, 1) = matrixVoigt(3, 4);
  fourTensor(1, 0, 2, 1) = matrixVoigt(6, 4);
  fourTensor(0, 1, 0, 2) = matrixVoigt(3, 5);
  fourTensor(1, 0, 0, 2) = matrixVoigt(6, 5);
  fourTensor(0, 1, 2, 0) = matrixVoigt(3, 5);
  fourTensor(1, 0, 2, 0) = matrixVoigt(6, 5);

  fourTensor(1, 2, 0, 0) = matrixVoigt(4, 0);
  fourTensor(2, 1, 0, 0) = matrixVoigt(7, 0);
  fourTensor(1, 2, 1, 1) = matrixVoigt(4, 1);
  fourTensor(2, 1, 1, 1) = matrixVoigt(7, 1);
  fourTensor(1, 2, 2, 2) = matrixVoigt(4, 2);
  fourTensor(2, 1, 2, 2) = matrixVoigt(7, 2);
  fourTensor(1, 2, 0, 1) = matrixVoigt(4, 3);
  fourTensor(2, 1, 0, 1) = matrixVoigt(7, 3);
  fourTensor(1, 2, 1, 0) = matrixVoigt(4, 3);
  fourTensor(2, 1, 1, 0) = matrixVoigt(7, 3);
  fourTensor(1, 2, 1, 2) = matrixVoigt(4, 4);
  fourTensor(2, 1, 1, 2) = matrixVoigt(7, 4);
  fourTensor(1, 2, 2, 1) = matrixVoigt(4, 4);
  fourTensor(2, 1, 2, 1) = matrixVoigt(7, 4);
  fourTensor(1, 2, 0, 2) = matrixVoigt(4, 5);
  fourTensor(2, 1, 0, 2) = matrixVoigt(7, 5);
  fourTensor(1, 2, 2, 0) = matrixVoigt(4, 5);
  fourTensor(2, 1, 2, 0) = matrixVoigt(7, 5);

  fourTensor(0, 2, 0, 0) = matrixVoigt(5, 0);
  fourTensor(2, 0, 0, 0) = matrixVoigt(8, 0);
  fourTensor(0, 2, 1, 1) = matrixVoigt(5, 1);
  fourTensor(2, 0, 1, 1) = matrixVoigt(8, 1);
  fourTensor(0, 2, 2, 2) = matrixVoigt(5, 2);
  fourTensor(2, 0, 2, 2) = matrixVoigt(8, 2);
  fourTensor(0, 2, 0, 1) = matrixVoigt(5, 3);
  fourTensor(2, 0, 0, 1) = matrixVoigt(8, 3);
  fourTensor(0, 2, 1, 0) = matrixVoigt(5, 3);
  fourTensor(2, 0, 1, 0) = matrixVoigt(8, 3);
  fourTensor(0, 2, 1, 2) = matrixVoigt(5, 4);
  fourTensor(2, 0, 1, 2) = matrixVoigt(8, 4);
  fourTensor(0, 2, 2, 1) = matrixVoigt(5, 4);
  fourTensor(2, 0, 2, 1) = matrixVoigt(8, 4);
  fourTensor(0, 2, 0, 2) = matrixVoigt(5, 5);
  fourTensor(2, 0, 0, 2) = matrixVoigt(8, 5);
  fourTensor(0, 2, 2, 0) = matrixVoigt(5, 5);
  fourTensor(2, 0, 2, 0) = matrixVoigt(8, 5);
}

template <int dim>
void Core::LinAlg::Voigt::setup_four_tensor_from_9x9_voigt_matrix(
    Core::LinAlg::FourTensor<dim>& fourTensor, const Core::LinAlg::Matrix<9, 9>& matrixVoigt)
{
  static_assert(dim == 3, "Current implementation only valid for dim = 3.");

  // Setup 4-Tensor from 9x9 Voigt matrix
  fourTensor(0, 0, 0, 0) = matrixVoigt(0, 0);
  fourTensor(0, 0, 1, 1) = matrixVoigt(0, 1);
  fourTensor(0, 0, 2, 2) = matrixVoigt(0, 2);
  fourTensor(0, 0, 0, 1) = matrixVoigt(0, 3);
  fourTensor(0, 0, 1, 0) = matrixVoigt(0, 6);
  fourTensor(0, 0, 1, 2) = matrixVoigt(0, 4);
  fourTensor(0, 0, 2, 1) = matrixVoigt(0, 7);
  fourTensor(0, 0, 0, 2) = matrixVoigt(0, 5);
  fourTensor(0, 0, 2, 0) = matrixVoigt(0, 8);

  fourTensor(1, 1, 0, 0) = matrixVoigt(1, 0);
  fourTensor(1, 1, 1, 1) = matrixVoigt(1, 1);
  fourTensor(1, 1, 2, 2) = matrixVoigt(1, 2);
  fourTensor(1, 1, 0, 1) = matrixVoigt(1, 3);
  fourTensor(1, 1, 1, 0) = matrixVoigt(1, 6);
  fourTensor(1, 1, 1, 2) = matrixVoigt(1, 4);
  fourTensor(1, 1, 2, 1) = matrixVoigt(1, 7);
  fourTensor(1, 1, 0, 2) = matrixVoigt(1, 5);
  fourTensor(1, 1, 2, 0) = matrixVoigt(1, 8);

  fourTensor(2, 2, 0, 0) = matrixVoigt(2, 0);
  fourTensor(2, 2, 1, 1) = matrixVoigt(2, 1);
  fourTensor(2, 2, 2, 2) = matrixVoigt(2, 2);
  fourTensor(2, 2, 0, 1) = matrixVoigt(2, 3);
  fourTensor(2, 2, 1, 0) = matrixVoigt(2, 6);
  fourTensor(2, 2, 1, 2) = matrixVoigt(2, 4);
  fourTensor(2, 2, 2, 1) = matrixVoigt(2, 7);
  fourTensor(2, 2, 0, 2) = matrixVoigt(2, 5);
  fourTensor(2, 2, 2, 0) = matrixVoigt(2, 8);

  fourTensor(0, 1, 0, 0) = matrixVoigt(3, 0);
  fourTensor(1, 0, 0, 0) = matrixVoigt(6, 0);
  fourTensor(0, 1, 1, 1) = matrixVoigt(3, 1);
  fourTensor(1, 0, 1, 1) = matrixVoigt(6, 1);
  fourTensor(0, 1, 2, 2) = matrixVoigt(3, 2);
  fourTensor(1, 0, 2, 2) = matrixVoigt(6, 2);
  fourTensor(0, 1, 0, 1) = matrixVoigt(3, 3);
  fourTensor(1, 0, 0, 1) = matrixVoigt(6, 3);
  fourTensor(0, 1, 1, 0) = matrixVoigt(3, 6);
  fourTensor(1, 0, 1, 0) = matrixVoigt(6, 6);
  fourTensor(0, 1, 1, 2) = matrixVoigt(3, 4);
  fourTensor(1, 0, 1, 2) = matrixVoigt(6, 4);
  fourTensor(0, 1, 2, 1) = matrixVoigt(3, 7);
  fourTensor(1, 0, 2, 1) = matrixVoigt(6, 7);
  fourTensor(0, 1, 0, 2) = matrixVoigt(3, 5);
  fourTensor(1, 0, 0, 2) = matrixVoigt(6, 5);
  fourTensor(0, 1, 2, 0) = matrixVoigt(3, 8);
  fourTensor(1, 0, 2, 0) = matrixVoigt(6, 8);

  fourTensor(1, 2, 0, 0) = matrixVoigt(4, 0);
  fourTensor(2, 1, 0, 0) = matrixVoigt(7, 0);
  fourTensor(1, 2, 1, 1) = matrixVoigt(4, 1);
  fourTensor(2, 1, 1, 1) = matrixVoigt(7, 1);
  fourTensor(1, 2, 2, 2) = matrixVoigt(4, 2);
  fourTensor(2, 1, 2, 2) = matrixVoigt(7, 2);
  fourTensor(1, 2, 0, 1) = matrixVoigt(4, 3);
  fourTensor(2, 1, 0, 1) = matrixVoigt(7, 3);
  fourTensor(1, 2, 1, 0) = matrixVoigt(4, 6);
  fourTensor(2, 1, 1, 0) = matrixVoigt(7, 6);
  fourTensor(1, 2, 1, 2) = matrixVoigt(4, 4);
  fourTensor(2, 1, 1, 2) = matrixVoigt(7, 4);
  fourTensor(1, 2, 2, 1) = matrixVoigt(4, 7);
  fourTensor(2, 1, 2, 1) = matrixVoigt(7, 7);
  fourTensor(1, 2, 0, 2) = matrixVoigt(4, 5);
  fourTensor(2, 1, 0, 2) = matrixVoigt(7, 5);
  fourTensor(1, 2, 2, 0) = matrixVoigt(4, 8);
  fourTensor(2, 1, 2, 0) = matrixVoigt(7, 8);

  fourTensor(0, 2, 0, 0) = matrixVoigt(5, 0);
  fourTensor(2, 0, 0, 0) = matrixVoigt(8, 0);
  fourTensor(0, 2, 1, 1) = matrixVoigt(5, 1);
  fourTensor(2, 0, 1, 1) = matrixVoigt(8, 1);
  fourTensor(0, 2, 2, 2) = matrixVoigt(5, 2);
  fourTensor(2, 0, 2, 2) = matrixVoigt(8, 2);
  fourTensor(0, 2, 0, 1) = matrixVoigt(5, 3);
  fourTensor(2, 0, 0, 1) = matrixVoigt(8, 3);
  fourTensor(0, 2, 1, 0) = matrixVoigt(5, 6);
  fourTensor(2, 0, 1, 0) = matrixVoigt(8, 6);
  fourTensor(0, 2, 1, 2) = matrixVoigt(5, 4);
  fourTensor(2, 0, 1, 2) = matrixVoigt(8, 4);
  fourTensor(0, 2, 2, 1) = matrixVoigt(5, 7);
  fourTensor(2, 0, 2, 1) = matrixVoigt(8, 7);
  fourTensor(0, 2, 0, 2) = matrixVoigt(5, 5);
  fourTensor(2, 0, 0, 2) = matrixVoigt(8, 5);
  fourTensor(0, 2, 2, 0) = matrixVoigt(5, 8);
  fourTensor(2, 0, 2, 0) = matrixVoigt(8, 8);
}

template <int dim>
void Core::LinAlg::Voigt::setup_6x6_voigt_matrix_from_four_tensor(
    Core::LinAlg::Matrix<6, 6>& matrix_voigt, const Core::LinAlg::FourTensor<dim>& four_tensor)
{
  static_assert(dim == 3, "Current implementation only valid for dim = 3.");


  // Setup 6x6 Voigt matrix from 4-Tensor
  matrix_voigt(0, 0) = four_tensor(0, 0, 0, 0);  // C1111
  matrix_voigt(0, 1) = four_tensor(0, 0, 1, 1);  // C1122
  matrix_voigt(0, 2) = four_tensor(0, 0, 2, 2);  // C1133
  matrix_voigt(0, 3) =
      0.5 * (four_tensor(0, 0, 0, 1) + four_tensor(0, 0, 1, 0));  // 0.5*(C1112+C1121)
  matrix_voigt(0, 4) =
      0.5 * (four_tensor(0, 0, 1, 2) + four_tensor(0, 0, 2, 1));  // 0.5*(C1123+C1132)
  matrix_voigt(0, 5) =
      0.5 * (four_tensor(0, 0, 0, 2) + four_tensor(0, 0, 2, 0));  // 0.5*(C1113+C1131)

  matrix_voigt(1, 0) = four_tensor(1, 1, 0, 0);  // C2211
  matrix_voigt(1, 1) = four_tensor(1, 1, 1, 1);  // C2222
  matrix_voigt(1, 2) = four_tensor(1, 1, 2, 2);  // C2233
  matrix_voigt(1, 3) =
      0.5 * (four_tensor(1, 1, 0, 1) + four_tensor(1, 1, 1, 0));  // 0.5*(C2212+C2221)
  matrix_voigt(1, 4) =
      0.5 * (four_tensor(1, 1, 1, 2) + four_tensor(1, 1, 2, 1));  // 0.5*(C2223+C2232)
  matrix_voigt(1, 5) =
      0.5 * (four_tensor(1, 1, 0, 2) + four_tensor(1, 1, 2, 0));  // 0.5*(C2213+C2231)

  matrix_voigt(2, 0) = four_tensor(2, 2, 0, 0);  // C3311
  matrix_voigt(2, 1) = four_tensor(2, 2, 1, 1);  // C3322
  matrix_voigt(2, 2) = four_tensor(2, 2, 2, 2);  // C3333
  matrix_voigt(2, 3) =
      0.5 * (four_tensor(2, 2, 0, 1) + four_tensor(2, 2, 1, 0));  // 0.5*(C3312+C3321)
  matrix_voigt(2, 4) =
      0.5 * (four_tensor(2, 2, 1, 2) + four_tensor(2, 2, 2, 1));  // 0.5*(C3323+C3332)
  matrix_voigt(2, 5) =
      0.5 * (four_tensor(2, 2, 0, 2) + four_tensor(2, 2, 2, 0));  // 0.5*(C3313+C3331)

  matrix_voigt(3, 0) =
      0.5 * (four_tensor(0, 1, 0, 0) + four_tensor(1, 0, 0, 0));  // 0.5*(C1211+C2111)
  matrix_voigt(3, 1) =
      0.5 * (four_tensor(0, 1, 1, 1) + four_tensor(1, 0, 1, 1));  // 0.5*(C1222+C2122)
  matrix_voigt(3, 2) =
      0.5 * (four_tensor(0, 1, 2, 2) + four_tensor(1, 0, 2, 2));  // 0.5*(C1233+C2133)
  matrix_voigt(3, 3) =
      0.25 * (four_tensor(0, 1, 0, 1) + four_tensor(1, 0, 0, 1) + four_tensor(0, 1, 1, 0) +
                 four_tensor(1, 0, 1, 0));  // 0.5*(C1212+C2112+C1221+C2121)
  matrix_voigt(3, 4) =
      0.25 * (four_tensor(0, 1, 1, 2) + four_tensor(1, 0, 1, 2) + four_tensor(0, 1, 2, 1) +
                 four_tensor(1, 0, 2, 1));  // 0.5*(C1223+C2123+C1232+C2132)
  matrix_voigt(3, 5) =
      0.25 * (four_tensor(0, 1, 0, 2) + four_tensor(1, 0, 0, 2) + four_tensor(0, 1, 2, 0) +
                 four_tensor(1, 0, 2, 0));  // 0.5*(C1213+C2113+C1231+C2131)

  matrix_voigt(4, 0) =
      0.5 * (four_tensor(1, 2, 0, 0) + four_tensor(2, 1, 0, 0));  // 0.5*(C2311+C3211)
  matrix_voigt(4, 1) =
      0.5 * (four_tensor(1, 2, 1, 1) + four_tensor(2, 1, 1, 1));  // 0.5*(C2322+C3222)
  matrix_voigt(4, 2) =
      0.5 * (four_tensor(1, 2, 2, 2) + four_tensor(2, 1, 2, 2));  // 0.5*(C2333+C3233)
  matrix_voigt(4, 3) =
      0.25 * (four_tensor(1, 2, 0, 1) + four_tensor(2, 1, 0, 1) + four_tensor(1, 2, 1, 0) +
                 four_tensor(2, 1, 1, 0));  // 0.5*(C2312+C3212+C2321+C3221)
  matrix_voigt(4, 4) =
      0.25 * (four_tensor(1, 2, 1, 2) + four_tensor(2, 1, 1, 2) + four_tensor(1, 2, 2, 1) +
                 four_tensor(2, 1, 2, 1));  // 0.5*(C2323+C3223+C2332+C3232)
  matrix_voigt(4, 5) =
      0.25 * (four_tensor(1, 2, 0, 2) + four_tensor(2, 1, 0, 2) + four_tensor(1, 2, 2, 0) +
                 four_tensor(2, 1, 2, 0));  // 0.5*(C2313+C3213+C2331+C3231)

  matrix_voigt(5, 0) =
      0.5 * (four_tensor(0, 2, 0, 0) + four_tensor(2, 0, 0, 0));  // 0.5*(C1311+C3111)
  matrix_voigt(5, 1) =
      0.5 * (four_tensor(0, 2, 1, 1) + four_tensor(2, 0, 1, 1));  // 0.5*(C1322+C3122)
  matrix_voigt(5, 2) =
      0.5 * (four_tensor(0, 2, 2, 2) + four_tensor(2, 0, 2, 2));  // 0.5*(C1333+C3133)
  matrix_voigt(5, 3) =
      0.25 * (four_tensor(0, 2, 0, 1) + four_tensor(2, 0, 0, 1) + four_tensor(0, 2, 1, 0) +
                 four_tensor(2, 0, 1, 0));  // 0.5*(C1312+C3112+C1321+C3121)
  matrix_voigt(5, 4) =
      0.25 * (four_tensor(0, 2, 1, 2) + four_tensor(2, 0, 1, 2) + four_tensor(0, 2, 2, 1) +
                 four_tensor(2, 0, 2, 1));  // 0.5*(C1323+C3123+C1332+C3132)
  matrix_voigt(5, 5) =
      0.25 * (four_tensor(0, 2, 0, 2) + four_tensor(2, 0, 0, 2) + four_tensor(0, 2, 2, 0) +
                 four_tensor(2, 0, 2, 0));  // 0.5*(C1313+C3113+C1331+C3131)
}

template <int dim>
void Core::LinAlg::Voigt::setup_9x6_voigt_matrix_from_four_tensor(
    Core::LinAlg::Matrix<9, 6>& matrixVoigt, const Core::LinAlg::FourTensor<dim>& fourTensor)
{
  static_assert(dim == 3, "Current implementation only valid for dim = 3.");

  // Setup 9x6 Voigt matrix from 4-Tensor
  matrixVoigt(0, 0) = fourTensor(0, 0, 0, 0);  // C1111
  matrixVoigt(0, 1) = fourTensor(0, 0, 1, 1);  // C1122
  matrixVoigt(0, 2) = fourTensor(0, 0, 2, 2);  // C1133
  matrixVoigt(0, 3) = 0.5 * (fourTensor(0, 0, 0, 1) + fourTensor(0, 0, 1, 0));
  // 0.5*(C1112+C1121)
  matrixVoigt(0, 4) = 0.5 * (fourTensor(0, 0, 1, 2) + fourTensor(0, 0, 2, 1));
  // 0.5*(C1123+C1132)
  matrixVoigt(0, 5) = 0.5 * (fourTensor(0, 0, 0, 2) + fourTensor(0, 0, 2, 0));
  // 0.5*(C1113+C1131)

  matrixVoigt(1, 0) = fourTensor(1, 1, 0, 0);  // C2211
  matrixVoigt(1, 1) = fourTensor(1, 1, 1, 1);  // C2222
  matrixVoigt(1, 2) = fourTensor(1, 1, 2, 2);  // C2233
  matrixVoigt(1, 3) = 0.5 * (fourTensor(1, 1, 0, 1) + fourTensor(1, 1, 1, 0));
  // 0.5*(C2212+C2221)
  matrixVoigt(1, 4) = 0.5 * (fourTensor(1, 1, 1, 2) + fourTensor(1, 1, 2, 1));
  // 0.5*(C2223+C2232)
  matrixVoigt(1, 5) = 0.5 * (fourTensor(1, 1, 0, 2) + fourTensor(1, 1, 2, 0));
  // 0.5*(C2213+C2231)

  matrixVoigt(2, 0) = fourTensor(2, 2, 0, 0);  // C3311
  matrixVoigt(2, 1) = fourTensor(2, 2, 1, 1);  // C3322
  matrixVoigt(2, 2) = fourTensor(2, 2, 2, 2);  // C3333
  matrixVoigt(2, 3) = 0.5 * (fourTensor(2, 2, 0, 1) + fourTensor(2, 2, 1, 0));
  // 0.5*(C3312+C3321)
  matrixVoigt(2, 4) = 0.5 * (fourTensor(2, 2, 1, 2) + fourTensor(2, 2, 2, 1));
  // 0.5*(C3323+C3332)
  matrixVoigt(2, 5) = 0.5 * (fourTensor(2, 2, 0, 2) + fourTensor(2, 2, 2, 0));
  // 0.5*(C3313+C3331)

  matrixVoigt(3, 0) = fourTensor(0, 1, 0, 0);  // C1211
  matrixVoigt(3, 1) = fourTensor(0, 1, 1, 1);  // C1222
  matrixVoigt(3, 2) = fourTensor(0, 1, 2, 2);  // C1233
  matrixVoigt(3, 3) = 0.5 * (fourTensor(0, 1, 0, 1) + fourTensor(0, 1, 1, 0));
  // 0.5*(C1212+C1221)
  matrixVoigt(3, 4) = 0.5 * (fourTensor(0, 1, 1, 2) + fourTensor(0, 1, 2, 1));
  // 0.5*(C1223+C2132)
  matrixVoigt(3, 5) = 0.5 * (fourTensor(0, 1, 0, 2) + fourTensor(0, 1, 2, 0));
  // 0.5*(C1213+C1231)

  matrixVoigt(4, 0) = fourTensor(1, 2, 0, 0);  // C2311
  matrixVoigt(4, 1) = fourTensor(1, 2, 1, 1);  // C2322
  matrixVoigt(4, 2) = fourTensor(1, 2, 2, 2);  // C2333
  matrixVoigt(4, 3) = 0.5 * (fourTensor(1, 2, 0, 1) + fourTensor(1, 2, 1, 0));
  // 0.5*(C2312+C2321)
  matrixVoigt(4, 4) = 0.5 * (fourTensor(1, 2, 1, 2) + fourTensor(1, 2, 2, 1));
  // 0.5*(C2323+C2332)
  matrixVoigt(4, 5) = 0.5 * (fourTensor(1, 2, 0, 2) + fourTensor(1, 2, 2, 0));
  // 0.5*(C2313+C2331)

  matrixVoigt(5, 0) = fourTensor(0, 2, 0, 0);  // C1311
  matrixVoigt(5, 1) = fourTensor(0, 2, 1, 1);  // C1322
  matrixVoigt(5, 2) = fourTensor(0, 2, 2, 2);  // C1333
  matrixVoigt(5, 3) = 0.5 * (fourTensor(0, 2, 0, 1) + fourTensor(0, 2, 1, 0));
  // 0.5*(C1312+C1321)
  matrixVoigt(5, 4) = 0.5 * (fourTensor(0, 2, 1, 2) + fourTensor(0, 2, 2, 1));
  // 0.5*(C1323+C1332)
  matrixVoigt(5, 5) = 0.5 * (fourTensor(0, 2, 0, 2) + fourTensor(0, 2, 2, 0));
  // 0.5*(C1313+C1331)

  matrixVoigt(6, 0) = fourTensor(1, 0, 0, 0);  // C2111
  matrixVoigt(6, 1) = fourTensor(1, 0, 1, 1);  // C2122
  matrixVoigt(6, 2) = fourTensor(1, 0, 2, 2);  // C2133
  matrixVoigt(6, 3) = 0.5 * (fourTensor(1, 0, 0, 1) + fourTensor(1, 0, 1, 0));
  // 0.5*(C2112+C2121)
  matrixVoigt(6, 4) = 0.5 * (fourTensor(1, 0, 1, 2) + fourTensor(1, 0, 2, 1));
  // 0.5*(C2123+C2132)
  matrixVoigt(6, 5) = 0.5 * (fourTensor(1, 0, 0, 2) + fourTensor(1, 0, 2, 0));
  // 0.5*(C2113+C2131)

  matrixVoigt(7, 0) = fourTensor(2, 1, 0, 0);  // C3211
  matrixVoigt(7, 1) = fourTensor(2, 1, 1, 1);  // C3222
  matrixVoigt(7, 2) = fourTensor(2, 1, 2, 2);  // C3233
  matrixVoigt(7, 3) = 0.5 * (fourTensor(2, 1, 0, 1) + fourTensor(2, 1, 1, 0));
  // 0.5*(C3212+C3221)
  matrixVoigt(7, 4) = 0.5 * (fourTensor(2, 1, 1, 2) + fourTensor(2, 1, 2, 1));
  // 0.5*(C3223+C3232)
  matrixVoigt(7, 5) = 0.5 * (fourTensor(2, 1, 0, 2) + fourTensor(2, 1, 2, 0));
  // 0.5*(C3213+C3231)

  matrixVoigt(8, 0) = fourTensor(2, 0, 0, 0);  // C3111
  matrixVoigt(8, 1) = fourTensor(2, 0, 1, 1);  // C3122
  matrixVoigt(8, 2) = fourTensor(2, 0, 2, 2);  // C3133
  matrixVoigt(8, 3) = 0.5 * (fourTensor(2, 0, 0, 1) + fourTensor(2, 0, 1, 0));
  // 0.5*(C3112+C3121)
  matrixVoigt(8, 4) = 0.5 * (fourTensor(2, 0, 1, 2) + fourTensor(2, 0, 2, 1));
  // 0.5*(C3123+C3132)
  matrixVoigt(8, 5) = 0.5 * (fourTensor(2, 0, 0, 2) + fourTensor(2, 0, 2, 0));
  // 0.5*(C3113+C3131)
}

template <int dim>
void Core::LinAlg::Voigt::setup_6x9_voigt_matrix_from_four_tensor(
    Core::LinAlg::Matrix<6, 9>& matrixVoigt, const Core::LinAlg::FourTensor<dim>& fourTensor)
{
  static_assert(dim == 3, "Current implementation only valid for dim = 3.");


  // Setup 9x6 Voigt matrix from 4-Tensor
  matrixVoigt(0, 0) = fourTensor(0, 0, 0, 0);  // C1111
  matrixVoigt(0, 1) = fourTensor(0, 0, 1, 1);  // C1122
  matrixVoigt(0, 2) = fourTensor(0, 0, 2, 2);  // C1133
  matrixVoigt(0, 3) = fourTensor(0, 0, 0, 1);  // C1112
  matrixVoigt(0, 4) = fourTensor(0, 0, 1, 2);  // C1123
  matrixVoigt(0, 5) = fourTensor(0, 0, 0, 2);  // C1113
  matrixVoigt(0, 6) = fourTensor(0, 0, 1, 0);  // C1121
  matrixVoigt(0, 7) = fourTensor(0, 0, 2, 1);  // C1132
  matrixVoigt(0, 8) = fourTensor(0, 0, 2, 0);  // C1131

  matrixVoigt(1, 0) = fourTensor(1, 1, 0, 0);  // C2211
  matrixVoigt(1, 1) = fourTensor(1, 1, 1, 1);  // C2222
  matrixVoigt(1, 2) = fourTensor(1, 1, 2, 2);  // C2233
  matrixVoigt(1, 3) = fourTensor(1, 1, 0, 1);  // C2212
  matrixVoigt(1, 4) = fourTensor(1, 1, 1, 2);  // C2223
  matrixVoigt(1, 5) = fourTensor(1, 1, 0, 2);  // C2213
  matrixVoigt(1, 6) = fourTensor(1, 1, 1, 0);  // C2221
  matrixVoigt(1, 7) = fourTensor(1, 1, 2, 1);  // C2232
  matrixVoigt(1, 8) = fourTensor(1, 1, 2, 0);  // C2231

  matrixVoigt(2, 0) = fourTensor(2, 2, 0, 0);  // C3311
  matrixVoigt(2, 1) = fourTensor(2, 2, 1, 1);  // C3322
  matrixVoigt(2, 2) = fourTensor(2, 2, 2, 2);  // C3333
  matrixVoigt(2, 3) = fourTensor(2, 2, 0, 1);  // C3312
  matrixVoigt(2, 4) = fourTensor(2, 2, 1, 2);  // C3323
  matrixVoigt(2, 5) = fourTensor(2, 2, 0, 2);  // C3313
  matrixVoigt(2, 6) = fourTensor(2, 2, 1, 0);  // C3321
  matrixVoigt(2, 7) = fourTensor(2, 2, 2, 1);  // C3332
  matrixVoigt(2, 8) = fourTensor(2, 2, 2, 0);  // C3331

  matrixVoigt(3, 0) =
      0.5 * (fourTensor(0, 1, 0, 0) + fourTensor(1, 0, 0, 0));  // 0.5 * (C1211 + C2111)
  matrixVoigt(3, 1) =
      0.5 * (fourTensor(0, 1, 1, 1) + fourTensor(1, 0, 1, 1));  // 0.5 * (C1222 + C2122)
  matrixVoigt(3, 2) =
      0.5 * (fourTensor(0, 1, 2, 2) + fourTensor(1, 0, 2, 2));  // 0.5 * (C1233 + C2133)
  matrixVoigt(3, 3) =
      0.5 * (fourTensor(0, 1, 0, 1) + fourTensor(1, 0, 0, 1));  // 0.5 * (C1212 + C2112)
  matrixVoigt(3, 4) =
      0.5 * (fourTensor(0, 1, 1, 2) + fourTensor(1, 0, 1, 2));  // 0.5 * (C1223 + C2123)
  matrixVoigt(3, 5) =
      0.5 * (fourTensor(0, 1, 0, 2) + fourTensor(1, 0, 0, 2));  // 0.5 * (C1213 + C2113)
  matrixVoigt(3, 6) =
      0.5 * (fourTensor(0, 1, 1, 0) + fourTensor(1, 0, 1, 0));  // 0.5 * (C1221 + C2121)
  matrixVoigt(3, 7) =
      0.5 * (fourTensor(0, 1, 2, 1) + fourTensor(1, 0, 2, 1));  // 0.5 * (C1232 + C2132)
  matrixVoigt(3, 8) =
      0.5 * (fourTensor(0, 1, 2, 0) + fourTensor(1, 0, 2, 0));  // 0.5 * (C1231 + C2131)

  matrixVoigt(4, 0) =
      0.5 * (fourTensor(1, 2, 0, 0) + fourTensor(2, 1, 0, 0));  // 0.5 * (C2311 + C3211)
  matrixVoigt(4, 1) =
      0.5 * (fourTensor(1, 2, 1, 1) + fourTensor(2, 1, 1, 1));  // 0.5 * (C2322 + C3222)
  matrixVoigt(4, 2) =
      0.5 * (fourTensor(1, 2, 2, 2) + fourTensor(2, 1, 2, 2));  // 0.5 * (C2333 + C3233)
  matrixVoigt(4, 3) =
      0.5 * (fourTensor(1, 2, 0, 1) + fourTensor(2, 1, 0, 1));  // 0.5 * (C2312 + C3212)
  matrixVoigt(4, 4) =
      0.5 * (fourTensor(1, 2, 1, 2) + fourTensor(2, 1, 1, 2));  // 0.5 * (C2323 + C3223)
  matrixVoigt(4, 5) =
      0.5 * (fourTensor(1, 2, 0, 2) + fourTensor(2, 1, 0, 2));  // 0.5 * (C2313 + C3213)
  matrixVoigt(4, 6) =
      0.5 * (fourTensor(1, 2, 1, 0) + fourTensor(2, 1, 1, 0));  // 0.5 * (C2321 + C3221)
  matrixVoigt(4, 7) =
      0.5 * (fourTensor(1, 2, 2, 1) + fourTensor(2, 1, 2, 1));  // 0.5 * (C2332 + C3232)
  matrixVoigt(4, 8) =
      0.5 * (fourTensor(1, 2, 2, 0) + fourTensor(2, 1, 2, 0));  // 0.5 * (C2331 + C3231)

  matrixVoigt(5, 0) =
      0.5 * (fourTensor(0, 2, 0, 0) + fourTensor(2, 0, 0, 0));  // 0.5 * (C1311 + C3111)
  matrixVoigt(5, 1) =
      0.5 * (fourTensor(0, 2, 1, 1) + fourTensor(2, 0, 1, 1));  // 0.5 * (C1322 + C3122)
  matrixVoigt(5, 2) =
      0.5 * (fourTensor(0, 2, 2, 2) + fourTensor(2, 0, 2, 2));  // 0.5 * (C1333 + C3133)
  matrixVoigt(5, 3) =
      0.5 * (fourTensor(0, 2, 0, 1) + fourTensor(2, 0, 0, 1));  // 0.5 * (C1312 + C3112)
  matrixVoigt(5, 4) =
      0.5 * (fourTensor(0, 2, 1, 2) + fourTensor(2, 0, 1, 2));  // 0.5 * (C1323 + C3123)
  matrixVoigt(5, 5) =
      0.5 * (fourTensor(0, 2, 0, 2) + fourTensor(2, 0, 0, 2));  // 0.5 * (C1313 + C3113)
  matrixVoigt(5, 6) =
      0.5 * (fourTensor(0, 2, 1, 0) + fourTensor(2, 0, 1, 0));  // 0.5 * (C1321 + C3121)
  matrixVoigt(5, 7) =
      0.5 * (fourTensor(0, 2, 2, 1) + fourTensor(2, 0, 2, 1));  // 0.5 * (C1332 + C3132)
  matrixVoigt(5, 8) =
      0.5 * (fourTensor(0, 2, 2, 0) + fourTensor(2, 0, 2, 0));  // 0.5 * (C1331 + C3131)
}

template <int dim>
void Core::LinAlg::Voigt::setup_9x9_voigt_matrix_from_four_tensor(
    Core::LinAlg::Matrix<9, 9>& matrixVoigt, const Core::LinAlg::FourTensor<dim>& fourTensor)
{
  static_assert(dim == 3, "Current implementation only valid for dim = 3.");


  // Setup 9x9 Voigt matrix from 4-Tensor
  matrixVoigt(0, 0) = fourTensor(0, 0, 0, 0);
  matrixVoigt(0, 1) = fourTensor(0, 0, 1, 1);
  matrixVoigt(0, 2) = fourTensor(0, 0, 2, 2);
  matrixVoigt(0, 3) = fourTensor(0, 0, 0, 1);
  matrixVoigt(0, 4) = fourTensor(0, 0, 1, 2);
  matrixVoigt(0, 5) = fourTensor(0, 0, 0, 2);
  matrixVoigt(0, 6) = fourTensor(0, 0, 1, 0);
  matrixVoigt(0, 7) = fourTensor(0, 0, 2, 1);
  matrixVoigt(0, 8) = fourTensor(0, 0, 2, 0);

  matrixVoigt(1, 0) = fourTensor(1, 1, 0, 0);
  matrixVoigt(1, 1) = fourTensor(1, 1, 1, 1);
  matrixVoigt(1, 2) = fourTensor(1, 1, 2, 2);
  matrixVoigt(1, 3) = fourTensor(1, 1, 0, 1);
  matrixVoigt(1, 4) = fourTensor(1, 1, 1, 2);
  matrixVoigt(1, 5) = fourTensor(1, 1, 0, 2);
  matrixVoigt(1, 6) = fourTensor(1, 1, 1, 0);
  matrixVoigt(1, 7) = fourTensor(1, 1, 2, 1);
  matrixVoigt(1, 8) = fourTensor(1, 1, 2, 0);

  matrixVoigt(2, 0) = fourTensor(2, 2, 0, 0);
  matrixVoigt(2, 1) = fourTensor(2, 2, 1, 1);
  matrixVoigt(2, 2) = fourTensor(2, 2, 2, 2);
  matrixVoigt(2, 3) = fourTensor(2, 2, 0, 1);
  matrixVoigt(2, 4) = fourTensor(2, 2, 1, 2);
  matrixVoigt(2, 5) = fourTensor(2, 2, 0, 2);
  matrixVoigt(2, 6) = fourTensor(2, 2, 1, 0);
  matrixVoigt(2, 7) = fourTensor(2, 2, 2, 1);
  matrixVoigt(2, 8) = fourTensor(2, 2, 2, 0);

  matrixVoigt(3, 0) = fourTensor(0, 1, 0, 0);
  matrixVoigt(3, 1) = fourTensor(0, 1, 1, 1);
  matrixVoigt(3, 2) = fourTensor(0, 1, 2, 2);
  matrixVoigt(3, 3) = fourTensor(0, 1, 0, 1);
  matrixVoigt(3, 4) = fourTensor(0, 1, 1, 2);
  matrixVoigt(3, 5) = fourTensor(0, 1, 0, 2);
  matrixVoigt(3, 6) = fourTensor(0, 1, 1, 0);
  matrixVoigt(3, 7) = fourTensor(0, 1, 2, 1);
  matrixVoigt(3, 8) = fourTensor(0, 1, 2, 0);

  matrixVoigt(4, 0) = fourTensor(1, 2, 0, 0);
  matrixVoigt(4, 1) = fourTensor(1, 2, 1, 1);
  matrixVoigt(4, 2) = fourTensor(1, 2, 2, 2);
  matrixVoigt(4, 3) = fourTensor(1, 2, 0, 1);
  matrixVoigt(4, 4) = fourTensor(1, 2, 1, 2);
  matrixVoigt(4, 5) = fourTensor(1, 2, 0, 2);
  matrixVoigt(4, 6) = fourTensor(1, 2, 1, 0);
  matrixVoigt(4, 7) = fourTensor(1, 2, 2, 1);
  matrixVoigt(4, 8) = fourTensor(1, 2, 2, 0);

  matrixVoigt(5, 0) = fourTensor(0, 2, 0, 0);
  matrixVoigt(5, 1) = fourTensor(0, 2, 1, 1);
  matrixVoigt(5, 2) = fourTensor(0, 2, 2, 2);
  matrixVoigt(5, 3) = fourTensor(0, 2, 0, 1);
  matrixVoigt(5, 4) = fourTensor(0, 2, 1, 2);
  matrixVoigt(5, 5) = fourTensor(0, 2, 0, 2);
  matrixVoigt(5, 6) = fourTensor(0, 2, 1, 0);
  matrixVoigt(5, 7) = fourTensor(0, 2, 2, 1);
  matrixVoigt(5, 8) = fourTensor(0, 2, 2, 0);

  matrixVoigt(6, 0) = fourTensor(1, 0, 0, 0);
  matrixVoigt(6, 1) = fourTensor(1, 0, 1, 1);
  matrixVoigt(6, 2) = fourTensor(1, 0, 2, 2);
  matrixVoigt(6, 3) = fourTensor(1, 0, 0, 1);
  matrixVoigt(6, 4) = fourTensor(1, 0, 1, 2);
  matrixVoigt(6, 5) = fourTensor(1, 0, 0, 2);
  matrixVoigt(6, 6) = fourTensor(1, 0, 1, 0);
  matrixVoigt(6, 7) = fourTensor(1, 0, 2, 1);
  matrixVoigt(6, 8) = fourTensor(1, 0, 2, 0);

  matrixVoigt(7, 0) = fourTensor(2, 1, 0, 0);
  matrixVoigt(7, 1) = fourTensor(2, 1, 1, 1);
  matrixVoigt(7, 2) = fourTensor(2, 1, 2, 2);
  matrixVoigt(7, 3) = fourTensor(2, 1, 0, 1);
  matrixVoigt(7, 4) = fourTensor(2, 1, 1, 2);
  matrixVoigt(7, 5) = fourTensor(2, 1, 0, 2);
  matrixVoigt(7, 6) = fourTensor(2, 1, 1, 0);
  matrixVoigt(7, 7) = fourTensor(2, 1, 2, 1);
  matrixVoigt(7, 8) = fourTensor(2, 1, 2, 0);

  matrixVoigt(8, 0) = fourTensor(2, 0, 0, 0);
  matrixVoigt(8, 1) = fourTensor(2, 0, 1, 1);
  matrixVoigt(8, 2) = fourTensor(2, 0, 2, 2);
  matrixVoigt(8, 3) = fourTensor(2, 0, 0, 1);
  matrixVoigt(8, 4) = fourTensor(2, 0, 1, 2);
  matrixVoigt(8, 5) = fourTensor(2, 0, 0, 2);
  matrixVoigt(8, 6) = fourTensor(2, 0, 1, 0);
  matrixVoigt(8, 7) = fourTensor(2, 0, 2, 1);
  matrixVoigt(8, 8) = fourTensor(2, 0, 2, 0);
}

template void Core::LinAlg::Voigt::setup_9x9_voigt_matrix_from_four_tensor<3>(
    Core::LinAlg::Matrix<9, 9>& matrix_voigt, const Core::LinAlg::FourTensor<3>& four_tensor);

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
template <NotationType type>
void Core::LinAlg::Voigt::VoigtUtils<type>::scale_off_diagonal_vals(
    Core::LinAlg::Matrix<6, 1>& strain)
{
  for (unsigned i = 3; i < 6; ++i) strain(i, 0) *= scale_factor(i);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
template <NotationType type>
void Core::LinAlg::Voigt::VoigtUtils<type>::unscale_off_diagonal_vals(
    Core::LinAlg::Matrix<6, 1>& strain)
{
  for (unsigned i = 3; i < 6; ++i) strain(i, 0) *= unscale_factor(i);
}

// explicit template declarations
template class Core::LinAlg::Voigt::VoigtUtils<NotationType::strain>;
template class Core::LinAlg::Voigt::VoigtUtils<NotationType::stress>;

template void Core::LinAlg::Voigt::VoigtUtils<Core::LinAlg::Voigt::NotationType::strain>::
    matrix_to_vector<double>(
        Core::LinAlg::Matrix<3, 3, double> const& in, Core::LinAlg::Matrix<6, 1, double>& out);
template void Core::LinAlg::Voigt::VoigtUtils<Core::LinAlg::Voigt::NotationType::stress>::
    matrix_to_vector<double>(
        Core::LinAlg::Matrix<3, 3, double> const& in, Core::LinAlg::Matrix<6, 1, double>& out);

using FAD = Sacado::Fad::DFad<double>;
template void
Core::LinAlg::Voigt::VoigtUtils<Core::LinAlg::Voigt::NotationType::strain>::matrix_to_vector<FAD>(
    Core::LinAlg::Matrix<3, 3, FAD> const& in, Core::LinAlg::Matrix<6, 1, FAD>& out);
template void
Core::LinAlg::Voigt::VoigtUtils<Core::LinAlg::Voigt::NotationType::stress>::matrix_to_vector<FAD>(
    Core::LinAlg::Matrix<3, 3, FAD> const& in, Core::LinAlg::Matrix<6, 1, FAD>& out);

template void Core::LinAlg::Voigt::fourth_order_identity_matrix<NotationType::stress,
    NotationType::stress>(Core::LinAlg::Matrix<6, 6>& id);
template void Core::LinAlg::Voigt::fourth_order_identity_matrix<NotationType::stress,
    NotationType::strain>(Core::LinAlg::Matrix<6, 6>& id);

template void Core::LinAlg::Voigt::setup_four_tensor_from_6x6_voigt_matrix<3>(
    Core::LinAlg::FourTensor<3>& four_tensor, const Core::LinAlg::Matrix<6, 6>& matrix_voigt);

template void Core::LinAlg::Voigt::setup_four_tensor_from_6x9_voigt_matrix<3>(
    Core::LinAlg::FourTensor<3>& four_tensor, const Core::LinAlg::Matrix<6, 9>& matrix_voigt);

template void Core::LinAlg::Voigt::setup_four_tensor_from_9x6_voigt_matrix<3>(
    Core::LinAlg::FourTensor<3>& four_tensor, const Core::LinAlg::Matrix<9, 6>& matrix_voigt);

template void Core::LinAlg::Voigt::setup_four_tensor_from_9x9_voigt_matrix<3>(
    Core::LinAlg::FourTensor<3>& four_tensor, const Core::LinAlg::Matrix<9, 9>& matrix_voigt);

template void Core::LinAlg::Voigt::setup_6x6_voigt_matrix_from_four_tensor<3>(
    Core::LinAlg::Matrix<6, 6>& matrix_voigt, const Core::LinAlg::FourTensor<3>& four_tensor);

template void Core::LinAlg::Voigt::setup_9x6_voigt_matrix_from_four_tensor<3>(
    Core::LinAlg::Matrix<9, 6>& matrix_voigt, const Core::LinAlg::FourTensor<3>& four_tensor);

template void Core::LinAlg::Voigt::setup_6x9_voigt_matrix_from_four_tensor<3>(
    Core::LinAlg::Matrix<6, 9>& matrix_voigt, const Core::LinAlg::FourTensor<3>& four_tensor);

FOUR_C_NAMESPACE_CLOSE
